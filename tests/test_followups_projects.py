"""Follow-ups after an action ("play it"), playing a video, the clock, "as admin", and opening folders on the D drive."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ev.llm.brain import _ASKS_CLOCK, Brain, claims_action
from ev.llm.client import OllamaClient
from ev.llm.fastpath import match_compound, match_fast
from ev.safety.gate import SafetyGate
from ev.safety.policy import Verdict
from ev.tools import system, web

from .fakes import FakeOllama, text_reply, tool_reply

LAST_YT = {"tool": "open_website", "args": {"site": "youtube", "query": "shape of you by ed sheeran"}, "message": "Searched YouTube."}


# ---- the clock
@pytest.mark.parametrize("said", [
    "what time is it", "whats the time", "what's the date today", "what day is it", "what is the date and time",
    "thats great can u tell me the current time and date", "can you tell me the time", "tell me the current time and date",
    "what is the current date", "what's today's date",
])
def test_the_clock_is_answered_by_the_tool_not_a_web_search(said):
    m = match_fast(said)
    assert m and m.tool == "get_time_date", said
    assert _ASKS_CLOCK.search(said)


@pytest.mark.parametrize("said", ["what time does the shop open", "what's the weather", "set a timer for five minutes"])
def test_other_questions_are_not_the_clock(said):
    m = match_fast(said)
    assert not (m and m.tool == "get_time_date"), said


# ---- "open cmd as admin"
def test_as_admin_becomes_a_real_admin_request_that_the_gate_confirms(policy):
    m = match_fast("open cmd as admin")
    assert m and m.tool == "open_app" and m.args == {"name": "cmd", "admin": True}
    assert policy.evaluate(m.tool, m.args).verdict is Verdict.CONFIRM
    plain = match_fast("open powershell")
    assert plain and plain.args == {"name": "powershell"}
    assert policy.evaluate(plain.tool, plain.args).verdict is Verdict.ALLOW
    for name in ("cmd", "command prompt", "powershell", "windows powershell"):
        assert name in system.KNOWN_APPS


# ---- playing a video
@pytest.mark.parametrize("said,args", [
    ("play shape of you by ed sheeran in youtube", {"site": "youtube", "query": "shape of you by ed sheeran", "play": True}),
    ("play despacito on youtube", {"site": "youtube", "query": "despacito", "play": True}),
    ("open youtube and play lofi beats", {"site": "youtube", "query": "lofi beats", "play": True}),
    ("search youtube for cats", {"site": "youtube", "query": "cats"}),                  # a search is not a play
    ("play shape of you on spotify", {"site": "spotify", "query": "shape of you"}),
])
def test_play_phrases(said, args):
    m = match_fast(said)
    assert m and m.tool == "open_website" and m.args == args, said


def test_open_spotify_alone_is_still_the_app():
    m = match_fast("open spotify")
    assert m and m.tool == "open_app"
    assert match_compound("open spotify and play despacito") is None          # "play ..." is not an app name


def test_follow_ups_use_what_ev_just_did():
    for said in ("play it", "can u play it", "play that", "play it again", "play the same song"):
        m = match_fast(said, LAST_YT)
        assert m and m.args == {"site": "youtube", "query": "shape of you by ed sheeran", "play": True}, said
    m = match_fast("play the same song on spotify", LAST_YT)
    assert m and m.args == {"site": "spotify", "query": "shape of you by ed sheeran"}
    m = match_fast("search it on youtube", LAST_YT)
    assert m and m.args == {"site": "youtube", "query": "shape of you by ed sheeran"}
    assert match_fast("play it") is None                                       # nothing to point at
    assert match_fast("play it", {"tool": "open_app", "args": {"name": "notepad"}, "message": ""}) is None
    assert match_fast("open it", LAST_YT).tool == "open_file"                  # "open it" keeps meaning the last file


def test_first_video_id_ignores_ads_shorts_and_bad_ids():
    page = '..."adSlotRenderer":{"videoId":"AAAAAAAAAAA"}..."videoRenderer":{"videoId":"JGwWNGJdvx8","thumbnail":...'
    assert web.first_video_id(page) == "JGwWNGJdvx8"
    assert web.first_video_id('"videoRenderer":{"videoId":"short"}') is None
    assert web.first_video_id("nothing here") is None
    assert web.watch_url("JGwWNGJdvx8") == "https://www.youtube.com/watch?v=JGwWNGJdvx8"
    for bad in ("../../etc", "a&b=c&d=e", "JGwWNGJdvx8x", ""):
        assert web.watch_url(bad) is None


def test_resolve_first_video_fetches_one_youtube_page(monkeypatch):
    seen = {}

    def fake_get(url, **kw):
        seen["url"], seen["follow"] = url, kw.get("follow_redirects")
        return httpx.Response(200, text='"videoRenderer":{"videoId":"JGwWNGJdvx8"')

    monkeypatch.setattr(web.httpx, "get", fake_get)
    assert web.resolve_first_video("shape of you") == "https://www.youtube.com/watch?v=JGwWNGJdvx8"
    assert seen["url"] == "https://www.youtube.com/results?search_query=shape+of+you" and seen["follow"] is False
    monkeypatch.setattr(web.httpx, "get", lambda url, **kw: httpx.Response(303, text=""))        # a consent redirect
    assert web.resolve_first_video("x") is None

    def boom(url, **kw):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(web.httpx, "get", boom)
    assert web.resolve_first_video("x") is None
    assert web.resolve_first_video("   ") is None


def test_spotify_search_url_uses_percent_encoding():
    assert web.build_url("spotify", "shape of you") == "https://open.spotify.com/search/shape%20of%20you"
    assert web.build_url("spotify") == "https://open.spotify.com"


def _opened(registry, monkeypatch, found):
    opened: list[str] = []
    monkeypatch.setattr(web, "open_url", lambda url, browser="", profile="": opened.append(url) or "your browser")
    monkeypatch.setattr(web, "resolve_first_video", lambda q, timeout_s=6.0: found)
    return opened


def test_open_website_play_opens_the_video_or_says_it_could_not(registry, monkeypatch):
    opened = _opened(registry, monkeypatch, "https://www.youtube.com/watch?v=JGwWNGJdvx8")
    r = registry.get("open_website").func(site="youtube", query="shape of you", play=True)
    assert r.ok and opened == ["https://www.youtube.com/watch?v=JGwWNGJdvx8"] and r.message.startswith("Opened the top YouTube result")
    opened = _opened(registry, monkeypatch, None)
    r = registry.get("open_website").func(site="youtube", query="shape of you", play=True)
    assert opened == ["https://www.youtube.com/results?search_query=shape+of+you"]
    assert "couldn't pick a video" in r.message and "Opened the top" not in r.message             # honest about what happened
    opened = _opened(registry, monkeypatch, "https://www.youtube.com/watch?v=JGwWNGJdvx8")
    r = registry.get("open_website").func(site="youtube", query="shape of you")            # no play: only the results page
    assert opened == ["https://www.youtube.com/results?search_query=shape+of+you"]
    r = registry.get("open_website").func(site="spotify", query="shape of you", play=True)
    assert "press play" in r.message and opened[-1] == "https://open.spotify.com/search/shape%20of%20you"


# ---- a whole conversation, as in the bug report
def _brain(cfg, memory, registry, policy, replies=None):
    fake = FakeOllama(replies or [text_reply("ok")])
    return Brain(cfg, OllamaClient(cfg, transport=fake.transport()), memory, registry, SafetyGate(policy), persona="x"), fake


def test_play_it_after_a_youtube_search_plays_it(cfg, memory, registry, policy, monkeypatch):
    opened = _opened(registry, monkeypatch, "https://www.youtube.com/watch?v=JGwWNGJdvx8")
    brain, fake = _brain(cfg, memory, registry, policy)
    first = brain.respond("search youtube for shape of you by ed sheeran")
    assert first.fast_path and opened[-1].startswith("https://www.youtube.com/results")
    assert brain.last_action and brain.last_action["args"]["query"] == "shape of you by ed sheeran"
    second = brain.respond("can u play it")
    assert second.fast_path and [t["tool"] for t in second.tool_log] == ["open_website"]
    assert opened[-1] == "https://www.youtube.com/watch?v=JGwWNGJdvx8" and second.text.startswith("Opened the top YouTube result")
    third = brain.respond("play the same song on spotify")
    assert third.fast_path and opened[-1] == "https://open.spotify.com/search/shape%20of%20you%20by%20ed%20sheeran"
    assert fake.requests == []                       # none of these needed the model


def test_the_model_is_told_what_it_just_did_only_when_the_request_points_back(cfg, memory, registry, policy, monkeypatch):
    _opened(registry, monkeypatch, None)
    brain, _ = _brain(cfg, memory, registry, policy)
    brain.respond("search youtube for lofi")
    assert "open_website(site=youtube, query=lofi)" in brain._last_action_note("do that again")
    assert brain._last_action_note("tell me a joke") == ""
    assert Brain(cfg, OllamaClient(cfg), memory, registry, SafetyGate(policy), persona="x")._last_action_note("play it") == ""


def test_a_tool_call_written_as_words_is_not_spoken_and_the_model_is_asked_again(cfg, memory, registry, policy, monkeypatch):
    opened = _opened(registry, monkeypatch, None)
    replies = [text_reply("Open_website spotify, query shape of you"),
               tool_reply("open_website", {"site": "spotify", "query": "shape of you"}), text_reply("Done.")]
    brain, fake = _brain(cfg, memory, registry, policy, replies)
    spoken: list[str] = []
    res = brain.respond("find some music on spotify please", on_sentence=spoken.append, allow_fast=False)
    assert not any("Open_website" in s for s in spoken)
    assert [t["tool"] for t in res.tool_log] == ["open_website"] and opened == ["https://open.spotify.com/search/shape%20of%20you"]
    assert len(fake.requests) >= 2


def test_claims_of_playing_without_a_tool_are_caught():
    assert claims_action("Controlled playback to play shape of you by ed sheeran.", "can u play it")
    assert claims_action("Started playing it.", "play it")
    assert not claims_action("The weather is sunny.", "weather")


# ---- the summary on exit
def test_a_session_summary_is_saved_when_it_ends(cfg, memory, registry, policy):
    brain, _ = _brain(cfg, memory, registry, policy, [text_reply("Sam works on EV and keeps projects on the D drive.")])
    assert brain.finish_session() is False                     # nothing said yet
    brain.history = [{"role": "user", "content": "my projects are on D"}, {"role": "assistant", "content": "ok"}]
    brain.turns = 2
    assert brain.finish_session() is True and "D drive" in memory.get_kv("summary")
    assert brain.finish_session() is False                     # already saved


# ---- folders on the D drive
@pytest.mark.parametrize("said,args", [
    ("open CLM in vs code", {"name": "clm", "how": "vscode"}),
    ("can u open my CLM project in a new VS CODE window", {"name": "clm", "how": "vscode"}),
    ("open the CLM project folder from my D Drive in the terminal", {"name": "clm", "how": "terminal"}),
    ("open EV_AI in the terminal and activate the ev environment", {"name": "ev_ai", "how": "terminal", "env": "ev"}),
    ("go to EV_AI and activate the ev environment", {"name": "ev_ai", "how": "terminal", "env": "ev"}),
    ("so with the powershell u have opened, go into my EV_AI project folder from D drive and activate the ev conda environment",
     {"name": "ev_ai", "how": "terminal", "env": "ev"}),
    ("open a terminal in EV_AI", {"name": "ev_ai", "how": "terminal"}),
    ("open the Riot Games folder", {"name": "riot games", "how": "folder"}),
    ("open projects in file explorer", {"name": "projects", "how": "folder"}),
])
def test_folder_phrases(said, args):
    m = match_fast(said)
    assert m and m.tool == "open_project" and m.args == args, said


@pytest.mark.parametrize("said", ["open that in the terminal", "open notepad", "open it in vs code", "open the screenshot you took"])
def test_vague_or_app_phrases_are_not_folders(said):
    m = match_fast(said)
    assert not (m and m.tool == "open_project"), said


def test_activate_after_a_terminal_is_one_request_not_two_clauses():
    assert match_compound("open EV_AI in the terminal and activate ev") is None


@pytest.mark.parametrize("args,verdict", [
    ({"name": "CLM_MVP", "how": "vscode"}, Verdict.ALLOW),
    ({"name": "Riot Games", "how": "folder"}, Verdict.ALLOW),
    ({"name": "EV_AI", "how": "terminal", "env": "ev"}, Verdict.ALLOW),
    ({"name": "EV_AI", "how": "terminal", "env": "base"}, Verdict.BLOCK),                 # not on the list
    ({"name": "EV_AI", "how": "terminal", "env": "ev; calc"}, Verdict.BLOCK),
    ({"name": "EV_AI", "how": "vscode", "env": "ev"}, Verdict.BLOCK),                     # env only goes with a terminal
    ({"name": "..\\Windows", "how": "folder"}, Verdict.BLOCK),
    ({"name": "C:\\Windows", "how": "folder"}, Verdict.BLOCK),
    ({"name": "D:/Projects/EV_AI", "how": "folder"}, Verdict.BLOCK),
    ({"name": "x&calc", "how": "vscode"}, Verdict.BLOCK),
    ({"name": "EV_AI --remote evil", "how": "vscode"}, Verdict.ALLOW),                    # plain words: no folder matches it, so nothing opens
    ({"name": "EV_AI", "how": "run"}, Verdict.BLOCK),
])
def test_open_project_policy(policy, args, verdict):
    assert policy.evaluate("open_project", args).verdict is verdict


def test_open_project_is_a_known_tool_and_cannot_run_without_a_level(policy):
    assert policy.levels["open_project"] == "safe"
    assert policy.project_roots and "ev" in policy.conda_envs


@pytest.fixture
def drive(tmp_path: Path):
    for d in ("Projects/CLM_MVP", "Projects/EV_AI", "Projects/WriteRightAI", "Games/Riot Games", "Games/HoYoPlay",
              "$RECYCLE.BIN/x", "Recovery", "ollama", ".hidden"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "Projects" / "notes.txt").write_text("a file, not a folder")
    return tmp_path


def test_folder_lookup_finds_by_spoken_name_and_stays_one_level_deep(drive):
    roots = [drive / "Projects", drive / "Games", drive]
    skip = frozenset({"$recycle.bin", "recovery"})
    assert system.find_folder("CLM", roots, skip) == drive / "Projects" / "CLM_MVP"
    assert system.find_folder("clm project", roots, skip) == drive / "Projects" / "CLM_MVP"
    assert system.find_folder("ev ai", roots, skip) == drive / "Projects" / "EV_AI"
    assert system.find_folder("riot games folder", roots, skip) == drive / "Games" / "Riot Games"
    assert system.find_folder("games", roots, skip) == drive / "Games"
    assert system.find_folder("ollama", roots, skip) == drive / "ollama"
    assert system.find_folder("notes", roots, skip) is None                      # a file
    assert system.find_folder("recovery", roots, skip) is None and system.find_folder(".hidden", roots, skip) is None
    assert system.find_folder("x", roots, skip) is None                          # only direct children: $RECYCLE.BIN\x is not one
    assert system.find_folder("anything", [drive / "missing drive"], skip) is None


def test_open_project_tool_opens_the_folder_and_says_only_what_it_did(registry, policy, drive, monkeypatch):
    policy.project_roots = [drive / "Projects", drive / "Games"]
    calls = []
    monkeypatch.setattr(system, "open_folder", lambda how, path, env="", agent="": calls.append((how, path, env)))
    tool = registry.get("open_project")
    r = tool.func(name="clm", how="terminal", env="ev")
    assert r.ok and calls == [("terminal", drive / "Projects" / "CLM_MVP", "ev")]
    assert "asked it to activate ev" in r.message and "CLM_MVP" in r.message
    assert not tool.func(name="nothing like this", how="folder").ok
    assert not tool.func(name="clm", how="terminal", env="base").ok and not tool.func(name="clm", how="vscode", env="ev").ok
    assert len(calls) == 1                                                       # the refused ones started nothing


def test_open_folder_builds_fixed_argument_lists(monkeypatch, tmp_path):
    started: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(system.subprocess, "Popen", lambda argv, **kw: started.append((argv, kw)))
    monkeypatch.setattr(system, "resolve_exe", lambda exe: {"wt.exe": "WT", "powershell.exe": "PS", "Code.exe": "CODE"}.get(exe))
    system.open_folder("terminal", tmp_path, "ev")
    assert started[-1][0] == ["WT", "-d", str(tmp_path), "PS", "-NoExit", "-Command", "conda activate ev"]
    system.open_folder("terminal", tmp_path)
    assert started[-1][0] == ["WT", "-d", str(tmp_path), "PS", "-NoExit"]
    system.open_folder("vscode", tmp_path)
    assert started[-1][0] == ["CODE", "--new-window", str(tmp_path)]
    for bad in ("ev; calc", "ev && calc", "$(calc)", "a b", ""):
        if bad:
            with pytest.raises(ValueError):
                system.open_folder("terminal", tmp_path, bad)
    with pytest.raises(ValueError):
        system.open_folder("run", tmp_path)
    monkeypatch.setattr(system, "resolve_exe", lambda exe: {"powershell.exe": "PS"}.get(exe))     # no Windows Terminal
    system.open_folder("terminal", tmp_path, "ev")
    assert started[-1][0] == ["PS", "-NoExit", "-Command", "conda activate ev"] and started[-1][1]["cwd"] == str(tmp_path)
    monkeypatch.setattr(system, "resolve_exe", lambda exe: None)
    with pytest.raises(FileNotFoundError):
        system.open_folder("vscode", tmp_path)


# ---- coding agents (Claude Code, Antigravity) started in a project's terminal
@pytest.mark.parametrize("said,args", [
    ("can u initialize a claude code env inside the GNSS Project",
     {"name": "gnss", "how": "terminal", "agent": "claude"}),
    ("start claude code in EV_AI", {"name": "ev_ai", "how": "terminal", "agent": "claude"}),
    ("open claude in the CLM project folder from my D drive", {"name": "clm", "how": "terminal", "agent": "claude"}),
    ("launch antigravity in GNSS", {"name": "gnss", "how": "terminal", "agent": "antigravity"}),
    ("run agy for the GNSS project", {"name": "gnss", "how": "terminal", "agent": "antigravity"}),
    ("open GNSS with claude code", {"name": "gnss", "how": "terminal", "agent": "claude"}),
    # the sentence that failed on 2026-11-14: a misspelt "initialized" and "in the terminal" after the agent
    ("can u open the GNSS project with claude code initilized in the terminal", {"name": "gnss", "how": "terminal", "agent": "claude"}),
    ("open the GNSS project with claude code initialised", {"name": "gnss", "how": "terminal", "agent": "claude"}),
    ("start claude code in the terminal in GNSS", {"name": "gnss", "how": "terminal", "agent": "claude"}),
    ("open GNSS in the terminal with antigravity started", {"name": "gnss", "how": "terminal", "agent": "antigravity"}),
])
def test_coding_agent_phrases(said, args):
    m = match_fast(said)
    assert m and m.tool == "open_project" and m.args == args, said


def test_an_agent_word_is_never_swallowed_into_a_folder_name():
    m = match_fast("open the GNSS project with claude code and some other words in the terminal")
    assert not (m and m.tool == "open_project" and "claude" in m.args["name"])


@pytest.mark.parametrize("said", ["open claude code", "open it in claude code", "start claude code in it", "open notepad"])
def test_a_coding_agent_needs_a_named_folder(said):
    m = match_fast(said)
    assert not (m and m.args.get("agent")), said


@pytest.mark.parametrize("args,verdict", [
    ({"name": "GNSS", "how": "terminal", "agent": "claude"}, Verdict.ALLOW),
    ({"name": "GNSS", "how": "terminal", "agent": "antigravity"}, Verdict.ALLOW),
    ({"name": "GNSS", "how": "terminal", "agent": "codex"}, Verdict.BLOCK),                 # not on the list
    ({"name": "GNSS", "how": "terminal", "agent": "claude; calc"}, Verdict.BLOCK),
    ({"name": "GNSS", "how": "vscode", "agent": "claude"}, Verdict.BLOCK),                  # a terminal only
    ({"name": "GNSS", "how": "terminal", "agent": "claude", "env": "ev"}, Verdict.BLOCK),   # on its own, no env
])
def test_coding_agent_policy(policy, args, verdict):
    assert policy.evaluate("open_project", args).verdict is verdict


def test_an_empty_agent_list_turns_it_off(policy):
    policy.coding_agents = frozenset()
    assert policy.evaluate("open_project", {"name": "GNSS", "how": "terminal", "agent": "claude"}).verdict is Verdict.BLOCK


def test_a_coding_agent_starts_a_fixed_program_without_the_user_profile(monkeypatch, tmp_path):
    started: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(system.subprocess, "Popen", lambda argv, **kw: started.append((argv, kw)))
    monkeypatch.setattr(system, "resolve_exe", lambda exe: {"wt.exe": "WT", "powershell.exe": "PS"}.get(exe))
    monkeypatch.setattr(system.shutil, "which", lambda prog: "C:/bin/" + prog)
    system.open_folder("terminal", tmp_path, agent="claude")
    assert started[-1][0] == ["WT", "-d", str(tmp_path), "PS", "-NoProfile", "-NoExit", "-Command", "claude"]
    system.open_folder("terminal", tmp_path, agent="antigravity")
    assert started[-1][0][-1] == "agy"
    for how, env, agent in (("vscode", "", "claude"), ("terminal", "ev", "claude"), ("terminal", "", "codex"), ("terminal", "", "claude && calc")):
        with pytest.raises(ValueError):
            system.open_folder(how, tmp_path, env, agent)
    monkeypatch.setattr(system.shutil, "which", lambda prog: None)
    with pytest.raises(FileNotFoundError, match="Claude Code"):
        system.open_folder("terminal", tmp_path, agent="claude")


def test_open_project_tool_says_it_asked_for_the_agent(registry, policy, drive, monkeypatch):
    policy.project_roots = [drive / "Projects"]
    calls = []
    monkeypatch.setattr(system, "open_folder", lambda how, path, env="", agent="": calls.append((how, path, env, agent)))
    tool = registry.get("open_project")
    r = tool.func(name="clm", how="terminal", agent="claude")
    assert r.ok and calls == [("terminal", drive / "Projects" / "CLM_MVP", "", "claude")]
    assert "asked it to start Claude Code" in r.message
    assert not tool.func(name="clm", how="vscode", agent="claude").ok and not tool.func(name="clm", how="terminal", agent="codex").ok
    assert len(calls) == 1


# ---- web search backups
class _Ctx:
    def __init__(self, **kw): ...
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_web_search_falls_back_to_the_next_engine_and_logs_why(registry, monkeypatch, caplog):
    seen: list[str] = []

    class FakeDDGS(_Ctx):
        def text(self, query, **kw):
            seen.append(kw["backend"])
            if kw["backend"] == "duckduckgo":
                raise RuntimeError("No results found.")
            return [{"title": "Nvidia CEO", "body": "Jensen Huang"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    monkeypatch.setattr("time.sleep", lambda s: None)
    import logging
    with caplog.at_level(logging.WARNING):
        r = registry.get("web_search").func(query="nvidia ceo")
    assert r.ok and "Jensen Huang" in r.message and seen == ["duckduckgo", "duckduckgo", "yahoo"]
    assert "web search attempt 1" in caplog.text and "No results found" in caplog.text


def test_web_search_stays_on_duckduckgo_when_the_lists_say_so(cfg, memory, scheduler, policy, monkeypatch):
    import datetime as dt

    from ev.tools.builtin import ToolContext, build_tools

    cfg["web"]["backends_text"] = ["duckduckgo"]
    cfg["web"]["backends_news"] = ["duckduckgo"]
    seen: list[str] = []

    class FakeDDGS(_Ctx):
        def text(self, query, **kw):
            seen.append(kw["backend"])
            raise RuntimeError("blocked")

        def news(self, query, **kw):
            seen.append(kw["backend"])
            raise RuntimeError("blocked")

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    monkeypatch.setattr("time.sleep", lambda s: None)
    reg = build_tools(ToolContext(cfg, memory, scheduler, policy, now=lambda: dt.datetime(2026, 1, 2)))
    r = reg.get("web_search").func(query="nvidia ceo")
    assert not r.ok and "isn't answering" in r.message and set(seen) == {"duckduckgo"}


def test_seed_facts_tell_ev_where_things_are():
    from ev.core.config import ROOT, load_toml

    facts = load_toml(ROOT / "config" / "seed_facts.toml")["facts"]
    assert "D:\\Projects" in facts["d_drive"] and "D:\\Games" in facts["d_drive"]
    assert all(e in facts["conda_envs"] for e in ("ev", "torch", "voice-agent")) and all(len(v) <= 200 for v in facts.values())


# ---- music and video open in the personal Firefox profile
def test_media_sites_open_in_the_configured_firefox_profile_but_a_named_browser_wins(registry, monkeypatch):
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(web, "open_url", lambda url, browser="", profile="": calls.append((url, browser, profile)) or browser)
    monkeypatch.setattr(web, "resolve_first_video", lambda q, timeout_s=6.0: "https://www.youtube.com/watch?v=JGwWNGJdvx8")
    tool = registry.get("open_website").func
    r = tool(site="youtube", query="shape of you", play=True)
    assert calls[-1] == ("https://www.youtube.com/watch?v=JGwWNGJdvx8", "firefox", "default-release") and " in firefox" not in r.message
    tool(site="spotify", query="shape of you")
    assert calls[-1][1:] == ("firefox", "default-release")
    tool(site="youtube", query="cats", browser="chrome")                       # the user named a browser: no profile
    assert calls[-1][1:] == ("chrome", "")
    tool(site="github")                                                        # not a media site: the system default
    assert calls[-1][1:] == ("", "")
    tool(url="https://example.com/x")
    assert calls[-1][1:] == ("", "")


def test_firefox_is_started_with_the_profile_name_from_config(monkeypatch):
    started: list[list[str]] = []
    monkeypatch.setattr(web.subprocess, "Popen", lambda argv, **kw: started.append(argv))
    monkeypatch.setattr(web.system, "resolve_exe", lambda exe: "FF" if exe == "firefox.exe" else "CH" if exe == "chrome.exe" else None)
    web.open_url("https://www.youtube.com/watch?v=JGwWNGJdvx8", "firefox", "default-release")
    assert started[-1] == ["FF", "-P", "default-release", "https://www.youtube.com/watch?v=JGwWNGJdvx8"]
    web.open_url("https://www.youtube.com", "firefox")
    assert started[-1] == ["FF", "https://www.youtube.com"]
    web.open_url("https://www.youtube.com", "chrome", "default-release")        # profiles are only for Firefox
    assert started[-1] == ["CH", "https://www.youtube.com"]
    for bad in ("-profile", "-no-remote", "a;b", "p\"q"):
        with pytest.raises(ValueError):
            web.open_url("https://www.youtube.com", "firefox", bad)


def test_the_media_profile_is_a_real_firefox_profile_name(cfg):
    assert cfg["web"]["media_browser"] == "firefox" and cfg["web"]["media_profile"] == "default-release"
