"""Regression tests: wake word, follow-up listening, app launching, web search, and prompt-size handling."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from ev.audio.trigger import WakeWord
from ev.core.config import load_config
from ev.llm.brain import Brain
from ev.llm.client import LLMError, OllamaClient
from ev.llm.fastpath import match_fast
from ev.safety.gate import SafetyGate
from ev.tools import system
from ev.tools.builtin import _first_rows

from .fakes import FakeOllama, ndjson, text_reply


# ---- fast path: explicit searches go to web_search and then back through the LLM
@pytest.mark.parametrize("text,query", [
    ("search for the capital of Australia", "the capital of australia"),
    ("can you google best python book please", "best python book"),
    ("look up who won the cricket world cup", "who won the cricket world cup"),
])
def test_search_fastpath(text, query):
    m = match_fast(text)
    assert m is not None and m.tool == "web_search" and m.then_llm
    assert m.args == {"query": query}


@pytest.mark.parametrize("text,name", [
    ("Could you launch the Edge browser for me?", "edge"),
    ("open Google Chrome for me please", "google chrome"),
    ("open the browser", "browser"),
    ("start the calculator app", "calculator"),
])
def test_open_app_name_is_trimmed(text, name):
    m = match_fast(text)
    assert m is not None and m.tool == "open_app" and m.args == {"name": name}


@pytest.mark.parametrize("text", ["look up at the stars", "search me out", "I love the weather here"])
def test_search_fastpath_ignores_chat(text):
    assert match_fast(text) is None


def test_search_goes_through_llm_with_tool_result(cfg, memory, policy, registry):
    seen = []
    from ev.tools.base import Tool, ToolResult

    registry.register(Tool("web_search", "s", {"query": {"type": "string"}},
                           lambda query: seen.append(query) or ToolResult(
                               True, "- Canberra: capital", untrusted=True, source="web_search", hint="Be brief."),
                           ("query",)))
    fake = FakeOllama([text_reply("It's Canberra.")])
    client = OllamaClient(cfg, transport=fake.transport())
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy), persona="You are EV.")
    res = brain.respond("search for the capital of Australia")
    assert seen == ["the capital of australia"] and res.text == "It's Canberra."
    tool_msg = [m for m in fake.requests[0]["messages"] if m["role"] == "tool"][0]["content"]
    assert "<untrusted" in tool_msg and tool_msg.rstrip().endswith("Be brief.")


# ---- the prompt must fit in num_ctx or Ollama silently drops the system prompt + tools
def test_history_is_trimmed_to_context_budget(cfg, memory, policy, registry):
    client = OllamaClient(cfg, transport=FakeOllama([text_reply("ok")]).transport())
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy), persona="You are EV.")
    brain.history_turns = 50
    brain.history = [{"role": r, "content": "word " * 300} for _ in range(30) for r in ("user", "assistant")]
    msgs = brain._messages("next question")
    est = brain._est_tokens(msgs) + brain._est_tokens(registry.schemas())
    assert est <= brain.ctx_budget
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[-1]["content"] == "next question"
    assert len(msgs) < 62


def test_history_turns_setting_is_honoured(cfg, memory, policy, registry):
    client = OllamaClient(cfg, transport=FakeOllama([text_reply("ok")]).transport())
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy), persona="x")
    brain.history = [{"role": r, "content": f"m{i}"} for i in range(20) for r in ("user", "assistant")]
    assert len(brain._messages("q")) == 1 + brain.history_turns * 2 + 1


def test_shipped_context_fits_real_persona(cfg, memory, policy, registry):
    """The real persona plus tool list must leave room for history in the configured num_ctx."""
    client = OllamaClient(cfg, transport=FakeOllama([text_reply("ok")]).transport())
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy))
    fixed = brain._est_tokens(brain.system_prompt()) + brain._est_tokens(registry.schemas())
    assert fixed < brain.ctx_budget * 0.75, f"prompt ~{fixed} tokens leaves little room in {brain.ctx_budget}"


# ---- GPU out-of-memory: retry on CPU instead of failing the turn
def test_oom_retries_on_cpu(cfg):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["options"].get("num_gpu"))
        if body["options"].get("num_gpu") != 0:
            return httpx.Response(500, json={"error": "cudaMalloc failed: out of memory"})
        return httpx.Response(200, content=text_reply("hi there"))

    client = OllamaClient(cfg, transport=httpx.MockTransport(handler))
    assert client.chat([{"role": "user", "content": "hi"}]).text == "hi there"
    assert calls == [None, 0]
    client.chat([{"role": "user", "content": "again"}])      # stays on CPU, no second failed GPU attempt
    assert calls == [None, 0, 0]


def test_other_llm_errors_are_not_retried(cfg):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": "boom"})

    client = OllamaClient(cfg, transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_inline_error_line_is_an_error(cfg):
    client = OllamaClient(cfg, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=ndjson({"error": "model not found"}))))
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_oom_gets_honest_spoken_message(cfg, memory, policy, registry):
    client = OllamaClient(cfg, transport=httpx.MockTransport(
        lambda r: httpx.Response(500, json={"error": "out of memory"})))
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy), persona="x")
    spoken = []
    brain.respond("tell me something interesting", on_sentence=spoken.append)
    assert "out of memory" in spoken[0]


# ---- web search: DuckDuckGo sometimes serves a bot-check page ("No results found")
class _Ctx:
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        return self.fn

    def __exit__(self, *a):
        return False


def test_search_retries_then_succeeds():
    outcomes = [DDGSErr(), [{"title": "", "body": ""}], [{"title": "Canberra", "body": "capital"}]]

    def attempt(_d):
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    rows = _first_rows([attempt] * 3, lambda: _Ctx(None), pause_s=0)
    assert rows == [{"title": "Canberra", "body": "capital"}]


def test_search_reports_unavailable_vs_empty():
    def boom(_d):
        raise DDGSErr()

    assert _first_rows([boom, boom], lambda: _Ctx(None), pause_s=0) is None
    assert _first_rows([lambda _d: []], lambda: _Ctx(None), pause_s=0) == []


class DDGSErr(Exception):
    pass


# ---- app launching
def test_launch_resolves_exe_off_path(monkeypatch):
    started = []
    monkeypatch.setattr(system.shutil, "which", lambda n: None)
    monkeypatch.setattr(system, "resolve_exe", lambda n: r"C:\Apps\chrome.exe" if n == "chrome.exe" else None)
    monkeypatch.setattr(system.subprocess, "Popen", lambda argv, **k: started.append(argv))
    assert system.launch_app("Chrome") == "chrome"
    assert started == [[r"C:\Apps\chrome.exe"]]


def test_launch_falls_back_to_start_menu_when_exe_missing(monkeypatch, tmp_path):
    lnk = tmp_path / "Google Chrome.lnk"
    lnk.write_text("x")
    opened = []
    monkeypatch.setattr(system, "resolve_exe", lambda n: None)
    monkeypatch.setattr(system, "_start_menu_dirs", lambda: [tmp_path])
    monkeypatch.setattr(system.os, "startfile", lambda p: opened.append(p), raising=False)
    assert system.launch_app("chrome") == "Google Chrome"
    assert opened == [str(lnk)]


def test_start_menu_matching(monkeypatch, tmp_path):
    for n in ("Visual Studio Code.lnk", "Uninstall OBS Studio.lnk", "OBS Studio.lnk", "Spotify.lnk"):
        (tmp_path / n).write_text("x")
    monkeypatch.setattr(system, "_start_menu_dirs", lambda: [tmp_path])
    assert system.find_start_menu_entry("obs").stem == "OBS Studio"          # never the uninstaller
    assert system.find_start_menu_entry("spotifi").stem == "Spotify"         # speech-recognition slip
    assert system.find_start_menu_entry("visual studio code").stem == "Visual Studio Code"
    assert system.find_start_menu_entry("photoshop") is None


def test_launch_unknown_app_raises(monkeypatch):
    monkeypatch.setattr(system, "_start_menu_dirs", lambda: [])
    monkeypatch.setattr(system, "find_store_app", lambda n, loose=True: None)
    with pytest.raises(FileNotFoundError, match="couldn't find an app called zzz"):
        system.launch_app("zzz")


# ---- wake word
def test_wakeword_model_lookup(tmp_path):
    cfg = load_config(overrides={"trigger": {"wakeword_dir": str(tmp_path), "wakeword_model": "hey_jarvis"}})
    w = WakeWord(cfg, lambda: None)
    with pytest.raises(FileNotFoundError, match="download_models.py --wakeword"):
        w.model_path()
    (tmp_path / "hey_jarvis_v0.1.onnx").write_bytes(b"x")
    assert w.model_path().name == "hey_jarvis_v0.1.onnx"


def test_wakeword_missing_model_degrades_gracefully(tmp_path):
    cfg = load_config(overrides={"trigger": {"wakeword_dir": str(tmp_path)}})
    w = WakeWord(cfg, lambda: None)
    w._run()
    assert w.error and not w.ready.is_set()


def test_shipped_config_enables_wakeword_and_followups(cfg):
    assert cfg["trigger"]["mode"] == "wakeword"
    assert cfg["conversation"]["followup_enabled"] is True
    assert cfg["llm"]["num_ctx"] >= 4096


# ---- follow-up window logic (EVApp is heavy; exercise the pure method directly)
def _app(cfg):
    from ev.app import EVApp

    app = EVApp.__new__(EVApp)
    app.cfg = cfg
    return SimpleNamespace(wait=lambda h, r: EVApp._followup_wait(app, h, r))


def test_followup_windows(cfg):
    f = _app(cfg).wait
    assert f("what's the capital of France", "Paris.") == cfg["conversation"]["followup_window_s"]
    assert f("open the thing", "Which one do you mean?") == cfg["conversation"]["followup_question_window_s"]
    for bye in ("thanks", "That's all.", "never mind", "no thank you", "goodbye"):
        assert f(bye, "Anytime.") is None, bye


def test_followup_can_be_disabled():
    cfg = load_config(overrides={"conversation": {"followup_enabled": False}})
    assert _app(cfg).wait("hello", "Hi!") is None


# ---- voice interruption detector
def _feed(det, frames):
    return [det.feed(i, sp, o) for i, sp, o in frames]


def test_echo_alone_never_interrupts():
    from ev.audio.interrupt import EchoDetector

    det = EchoDetector()
    echo = [(0.06, True, 0.6)] * 60                  # EV's own voice leaking into the mic at a steady ratio
    assert not any(_feed(det, echo))


def test_user_louder_than_echo_interrupts_after_min_duration():
    from ev.audio.interrupt import EchoDetector

    det = EchoDetector(block_ms=30, min_ms=300)
    _feed(det, [(0.06, True, 0.6)] * 20)             # calibrate on echo only
    out = _feed(det, [(0.20, True, 0.6)] * 12)       # user starts talking over EV
    assert not any(out[:9]) and out[9]               # fires on the 10th block (300 ms), not before


def test_lagged_echo_and_sentence_gaps_do_not_self_interrupt():
    from ev.audio.interrupt import EchoDetector

    det = EchoDetector()
    # Real log: the mic heard almost nothing for the first ~200 ms of playback (output buffering), then EV's voice arrived,
    # with a quiet gap between sentences where the echo tail still rings. That used to calibrate the echo to ~0.
    reply = [(0.004, True, 0.6)] * 6 + [(0.12, True, 0.6)] * 30 + [(0.10, True, 0.0)] * 6 + [(0.12, True, 0.6)] * 30
    assert not any(_feed(det, reply))
    assert det.ratio is not None and det.ratio > 0.1


def test_short_noise_does_not_interrupt():
    from ev.audio.interrupt import EchoDetector

    det = EchoDetector()
    _feed(det, [(0.06, True, 0.6)] * 20)
    burst = [(0.3, True, 0.6)] * 4 + [(0.06, True, 0.6)] * 20     # a cough or door slam
    assert not any(_feed(det, burst))


def test_headphones_user_speech_in_gap_between_sentences():
    from ev.audio.interrupt import EchoDetector

    det = EchoDetector()
    _feed(det, [(0.003, True, 0.5)] * 20)            # headphones: almost no echo
    assert any(_feed(det, [(0.05, True, 0.0)] * 12))


def test_no_interrupt_before_calibration():
    from ev.audio.interrupt import EchoDetector

    det = EchoDetector()
    assert not any(_feed(det, [(0.5, True, 0.0)] * 40))     # nothing was playing yet, so nothing to compare to


def test_non_speech_never_interrupts():
    from ev.audio.interrupt import EchoDetector

    det = EchoDetector()
    _feed(det, [(0.06, True, 0.6)] * 20)
    assert not any(_feed(det, [(0.5, False, 0.6)] * 30))


def test_brain_stops_generating_when_interrupted(cfg, memory, policy, registry):
    fake = FakeOllama([text_reply("First sentence here. Second sentence here. Third sentence here. Fourth one.", parts=12)])
    client = OllamaClient(cfg, transport=fake.transport())
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy), persona="x")
    spoken: list[str] = []
    res = brain.respond("tell me things please", on_sentence=spoken.append,
                        should_stop=lambda: len(spoken) >= 1)
    assert res.interrupted and len(spoken) == 1
    assert brain.history[-1]["content"] == spoken[0]          # only what was actually said is remembered


# ---- weather is a real tool now (search snippets never contain the actual conditions)
@pytest.mark.parametrize("text,args", [
    ("what's the weather in Chennai", {"place": "chennai", "day": "today"}),
    ("what is the weather in Mumbai today", {"place": "mumbai", "day": "today"}),
    ("what's the weather like in London tomorrow", {"place": "london", "day": "tomorrow"}),
    ("weather tomorrow in Delhi", {"place": "delhi", "day": "tomorrow"}),
    ("how's the weather", {"day": "today"}),
    ("the weather.", {"day": "today"}),
    ("what's the weather like today", {"day": "today"}),
])
def test_weather_fastpath(text, args):
    m = match_fast(text)
    assert m is not None and m.tool == "get_weather" and not m.then_llm and m.args == args


def _weather_http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _om_handler(request: httpx.Request) -> httpx.Response:
    if "geocoding" in request.url.host:
        if request.url.params["name"].lower() == "nowhere":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"results": [{"name": "Chennai<script>", "latitude": 13.08, "longitude": 80.27}]})
    return httpx.Response(200, json={
        "current": {"temperature_2m": 28.4, "apparent_temperature": 34.2, "weather_code": 2},
        "daily": {"weather_code": [2, 61], "temperature_2m_max": [34.1, 31.0], "temperature_2m_min": [25.0, 24.6],
                  "precipitation_probability_max": [20, 70]}})


def test_weather_text_today_and_tomorrow():
    from ev.tools import weather

    today = weather.get_weather("Chennai", "today", http=_weather_http(_om_handler))
    assert today == ("Right now in Chennaiscript it's 28 degrees Celsius and partly cloudy, feels like 34. "
                     "Today's high is 34 and the low is 25, with a 20 percent chance of rain.")    # markup stripped from the name
    tom = weather.get_weather("Chennai", "tomorrow", http=_weather_http(_om_handler))
    assert "lightly raining" in tom and "70 percent" in tom and tom.startswith("Tomorrow in")


def test_weather_errors_are_speakable():
    from ev.tools import weather

    with pytest.raises(weather.WeatherError, match="couldn't find a place"):
        weather.get_weather("Nowhere", http=_weather_http(_om_handler))
    with pytest.raises(weather.WeatherError, match="Which city"):
        weather.get_weather("  ", http=_weather_http(_om_handler))
    boom = _weather_http(lambda r: httpx.Response(503))
    with pytest.raises(weather.WeatherError, match="couldn't reach"):
        weather.get_weather("Chennai", http=boom)


def test_weather_tool_uses_home_city_and_respects_switch(registry, memory, cfg, monkeypatch):
    from ev.tools import weather

    seen = []
    monkeypatch.setattr(weather, "get_weather", lambda place, day, units, timeout: seen.append((place, day)) or "ok")
    tool = registry.get("get_weather")
    memory.remember("city", "Pune")
    assert tool.func(day="tomorrow").message == "ok" and seen[-1] == ("Pune", "tomorrow")
    assert tool.func(place="Goa").message == "ok" and seen[-1] == ("Goa", "today")
    cfg["web"]["weather_enabled"] = False
    try:
        r = tool.func(place="Pune")
        assert not r.ok and "turned off" in r.message
    finally:
        cfg["web"]["weather_enabled"] = True


def test_weather_goes_through_gate_as_safe(policy):
    from ev.safety.policy import Verdict

    assert policy.evaluate("get_weather", {"place": "Chennai"}).verdict is Verdict.ALLOW


# ---- opening apps by voice
@pytest.mark.parametrize("text,name", [
    ("Can you open up Cloud Desktop in my PC?", "cloud desktop"),
    ("open up Claude Desktop", "claude desktop"),
    ("launch Spotify on my computer please", "spotify"),
    ("open vs code for me", "vs code"),
])
def test_open_variants(text, name):
    m = match_fast(text)
    assert m is not None and m.tool == "open_app" and m.args == {"name": name}


@pytest.mark.parametrize("text", ["open up to me about your day", "open up about it", "start my day", "run a test"])
def test_open_up_chat_is_not_an_app(text):
    assert match_fast(text) is None


def test_asr_mishearing_resolves_to_store_app(monkeypatch, tmp_path):
    (tmp_path / "Genshin Impact Cloud.lnk").write_text("x")
    monkeypatch.setattr(system, "_start_menu_dirs", lambda: [tmp_path])
    monkeypatch.setattr(system, "_store_entries", lambda: {"claude": ("Claude", "Claude_x!Claude")})
    started = []
    monkeypatch.setattr(system.subprocess, "Popen", lambda argv, **k: started.append(argv))
    assert system.launch_app("cloud desktop") == "Claude"                    # Whisper hears "Cloud" for "Claude"
    assert started == [["explorer.exe", "shell:AppsFolder\\Claude_x!Claude"]]
    opened = []
    monkeypatch.setattr(system.os, "startfile", lambda p: opened.append(p), raising=False)
    assert system.launch_app("genshin impact cloud") == "Genshin Impact Cloud"   # an exact name always wins


def test_unrelated_names_do_not_fuzzy_match(monkeypatch, tmp_path):
    monkeypatch.setattr(system, "_start_menu_dirs", lambda: [tmp_path])
    monkeypatch.setattr(system, "_store_entries", lambda: {"photos": ("Photos", "Photos!App")})
    assert system.find_store_app("photoshop") is None


# ---- screen questions go to the screen tool; internal tool names are never spoken
@pytest.mark.parametrize("text", ["can you see my screen right now?", "what's on my screen", "read my screen", "what am I looking at"])
def test_screen_fastpath(text):
    m = match_fast(text)
    assert m is not None and m.tool == "read_screen" and m.then_llm


def test_tool_names_are_not_spoken(cfg, memory, policy, registry):
    fake = FakeOllama([text_reply("Got it. I can use the web_search tool for that. It is sunny today.")])
    client = OllamaClient(cfg, transport=fake.transport())
    brain = Brain(cfg, client, memory, registry, SafetyGate(policy), persona="x")
    spoken: list[str] = []
    brain.respond("how is the mood outside", on_sentence=spoken.append, allow_fast=False)
    assert spoken and not any("web_search" in x for x in spoken) and any("sunny" in x for x in spoken)


def test_search_results_are_cached(cfg, memory, policy, monkeypatch):
    import datetime as dt

    from ev.tools import builtin
    from ev.tools.builtin import ToolContext, build_tools
    from ev.tools.scheduler import Scheduler

    calls = []
    monkeypatch.setattr(builtin, "_first_rows",
                        lambda attempts, make, pause_s=1.0: calls.append(1) or [{"title": "T", "body": "B"}])
    reg = build_tools(ToolContext(cfg, memory, Scheduler(memory, lambda t: None), policy, now=lambda: dt.datetime(2026, 1, 2)))
    tool = reg.get("web_search")
    assert tool.func(query="Capital of Australia").ok and tool.func(query="capital of australia ").ok
    assert len(calls) == 1
