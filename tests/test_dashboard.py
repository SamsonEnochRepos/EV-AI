"""The dashboard: how it signs you in, what it refuses, what it can and cannot change, and that it only asks EV to do things
through the same gate as a spoken request."""
from __future__ import annotations

import hashlib
import http.client
import json
import shutil
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ev.core.config import ROOT, Config
from ev.core.events import State
from ev.dashboard import localconf
from ev.dashboard.backend import AppBackend
from ev.dashboard.hub import Hub
from ev.dashboard.server import FILES, MAX_BODY, TOKEN_TTL_S, DashboardError, DashboardServer
from ev.dashboard.settings_schema import ALLOWED, SettingError, coerce
from ev.memory.store import MemoryStore
from ev.safety.policy import Policy
from ev.skills.store import SkillStore
from ev.tools.base import Tool, ToolRegistry, ToolResult

from .fakes import text_reply
from .test_skills import Rig

# ---------------------------------------------------------------- a server with a fake backend


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        def call(*args: Any) -> dict[str, Any]:
            self.calls.append((name, args))
            return {"ok": True, "called": name}
        return call


class Browser:
    """A bare HTTP client that keeps cookies by hand, so a test can send exactly the headers it means to."""

    def __init__(self, server: DashboardServer):
        self.server, self.cookie = server, ""

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None,
                host: str | None = None, origin: str | None = "same", cookie: bool = True) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        h = {"Host": host or f"127.0.0.1:{self.server.port}"}
        if cookie and self.cookie:
            h["Cookie"] = self.cookie
        if origin:
            h["Origin"] = f"http://127.0.0.1:{self.server.port}" if origin == "same" else origin
        h.update(headers or {})
        conn.putrequest(method, path, skip_host=True)
        for k, v in h.items():
            conn.putheader(k, v)
        if body is not None:
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders(body)
        r = conn.getresponse()
        data = r.read()
        out = (r.status, {k.lower(): v for k, v in r.getheaders()}, data)
        conn.close()
        return out

    def sign_in(self) -> None:
        url = self.server.login_url()
        status, hdr, _ = self.request("GET", url.split(str(self.server.port), 1)[1], cookie=False)
        assert status == 302
        self.cookie = hdr["set-cookie"].split(";", 1)[0]

    def post(self, path: str, obj: Any, **kw: Any) -> tuple[int, dict[str, str], bytes]:
        h = {"Content-Type": "application/json", **kw.pop("headers", {})}
        return self.request("POST", path, json.dumps(obj).encode(), headers=h, **kw)


@pytest.fixture
def served():
    clock = {"t": 1000.0}
    backend, hub = FakeBackend(), Hub()
    server = DashboardServer(backend, hub, clock=lambda: clock["t"])
    server.start()
    yield SimpleNamespace(server=server, backend=backend, hub=hub, clock=clock, browser=Browser(server))
    server.stop()


def test_it_listens_on_loopback_only(served):
    assert served.server._httpd.server_address[0] == "127.0.0.1"


def test_nothing_works_without_signing_in(served):
    b = served.browser
    for path in ("/", "/app.js", "/api/status", "/api/facts", "/api/policy", "/api/events"):
        assert b.request("GET", path, cookie=False)[0] == 401
    assert b.post("/api/control", {"action": "kill"}, cookie=False)[0] == 401
    assert served.backend.calls == []


def test_the_sign_in_link_works_once_and_becomes_a_cookie(served):
    b = served.browser
    url = served.server.login_url()
    path = url.split(str(served.server.port), 1)[1]
    status, hdr, _ = b.request("GET", path, cookie=False)
    assert status == 302 and hdr["location"] == "/"
    cookie = hdr["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert b.request("GET", path, cookie=False)[0] == 403                # the link is spent
    b.cookie = cookie.split(";", 1)[0]
    status, hdr, body = b.request("GET", "/")
    assert status == 200 and b"EV dashboard" in body
    assert "script-src 'self'" in hdr["content-security-policy"] and "'unsafe-inline'" not in hdr["content-security-policy"]
    assert hdr["cache-control"] == "no-store"


def test_a_sign_in_link_expires(served):
    path = served.server.login_url().split(str(served.server.port), 1)[1]
    served.clock["t"] += TOKEN_TTL_S + 1
    assert served.browser.request("GET", path, cookie=False)[0] == 403


def test_a_guessed_token_or_cookie_is_refused(served):
    b = served.browser
    assert b.request("GET", "/?t=" + "a" * 32, cookie=False)[0] == 403
    b.cookie = "ev_dash=" + "b" * 43
    assert b.request("GET", "/api/status")[0] == 401


def test_a_wrong_host_is_refused_even_with_a_cookie(served):
    b = served.browser
    b.sign_in()
    assert b.request("GET", "/api/status", host="evil.example:80")[0] == 403          # DNS rebinding
    assert b.request("GET", "/api/status", host="127.0.0.1:1")[0] == 403
    assert b.request("GET", "/api/status")[0] == 200


def test_posts_need_the_right_origin_json_and_a_small_body(served):
    b = served.browser
    b.sign_in()
    assert b.post("/api/control", {"action": "pause"})[0] == 200
    assert b.post("/api/control", {"action": "pause"}, origin=None)[0] == 403
    assert b.post("/api/control", {"action": "pause"}, origin="http://evil.example")[0] == 403
    assert b.post("/api/control", {"action": "pause"}, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert b.request("POST", "/api/control", b"action=pause", headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    assert b.request("POST", "/api/control", b"not json", headers={"Content-Type": "application/json"})[0] == 400
    assert b.request("POST", "/api/control", b"[1]", headers={"Content-Type": "application/json"})[0] == 400
    assert b.request("POST", "/api/control", b"x" * (MAX_BODY + 1), headers={"Content-Type": "application/json"})[0] == 413
    assert b.post("/api/nothing", {})[0] == 404
    assert [c[0] for c in served.backend.calls] == ["control"]                         # only the one good request got through


def test_there_is_no_route_that_writes_policy_or_arbitrary_files(served):
    b = served.browser
    b.sign_in()
    for path in ("/api/policy", "/api/policy/save", "/api/config", "/api/file", "/api/exec", "/api/builder", "/api/vault/path"):
        assert b.post(path, {"text": "x"})[0] == 404
    assert b.request("PUT", "/api/policy", b"{}", headers={"Content-Type": "application/json"})[0] in (400, 404, 501)
    assert b.request("GET", "/../config/policy.toml")[0] == 404
    assert b.request("GET", "/static/../../config/ev.toml")[0] == 404


def test_events_stream_what_the_hub_publishes(served):
    b = served.browser
    b.sign_in()
    conn = http.client.HTTPConnection("127.0.0.1", served.server.port, timeout=5)
    conn.putrequest("GET", "/api/events", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{served.server.port}")
    conn.putheader("Cookie", b.cookie)
    conn.endheaders()
    r = conn.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    got: list[str] = []

    def read() -> None:
        while not any(g.startswith("data:") for g in got):
            line = r.fp.readline().decode()
            if not line:
                return
            got.append(line.strip())

    t = threading.Thread(target=read, daemon=True)
    t.start()
    for _ in range(100):                                    # wait until the handler has subscribed
        if served.hub.watchers:
            break
        threading.Event().wait(0.02)
    served.hub.publish("tool", tool="set_volume", ok=True, decision="allow")
    t.join(5)
    conn.close()
    assert "event: tool" in got and any('"decision": "allow"' in g for g in got)


def _open_events(served):
    b = served.browser
    b.sign_in()
    conn = http.client.HTTPConnection("127.0.0.1", served.server.port, timeout=5)
    conn.putrequest("GET", "/api/events", skip_host=True)
    conn.putheader("Host", f"127.0.0.1:{served.server.port}")
    conn.putheader("Cookie", b.cookie)
    conn.endheaders()
    return conn, conn.getresponse()


def _read_events(r, want: int) -> list[str]:
    got: list[str] = []
    while sum(g.startswith("data:") for g in got) < want:
        line = r.fp.readline().decode()
        if not line:
            break
        got.append(line.strip())
    return got


def test_a_page_that_connects_late_is_caught_up_on_recent_events_but_not_audio_levels(served):
    served.hub.publish("turn", user="hello there", reply="hi")
    served.hub.publish("level", v=0.7)
    served.hub.publish("tool", tool="get_time_date", ok=True, decision="allow")
    conn, r = _open_events(served)
    got = _read_events(r, 2)
    conn.close()
    events = [g for g in got if g.startswith("event:")]
    assert events == ["event: turn", "event: tool"]                      # oldest first, and the level is never replayed
    assert any("hello there" in g for g in got)


def test_levels_stream_live_to_a_watching_page(served):
    conn, r = _open_events(served)
    for _ in range(100):
        if served.hub.watchers:
            break
        threading.Event().wait(0.02)
    served.hub.publish("level", v=0.42)
    got = _read_events(r, 1)
    conn.close()
    assert "event: level" in got and any('"v": 0.42' in g for g in got)


def test_the_hub_never_blocks_on_a_browser_that_stopped_reading():
    hub = Hub()
    q = hub.subscribe()
    for i in range(1000):
        hub.publish("tool", n=i)
    assert q.qsize() == 200 and hub.recent()["tool"][-1]["n"] == 999
    hub.unsubscribe(q)
    assert hub.watchers == 0


def test_an_event_cannot_overwrite_its_own_kind():
    hub = Hub()
    hub.publish("proactive", kind="spoken", at="x")
    last = hub.recent()["proactive"][-1]
    assert last["kind"] == "proactive" and isinstance(last["at"], float)


# ---------------------------------------------------------------- settings: an allow-list, written to ev.local.toml only

def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class FakeSpeaker:
    def __init__(self, fail: bool = False):
        self.engine = SimpleNamespace(name="piper", voice_name="en_US-hfc_female-medium")
        self.pause_ms = 90
        self.fail, self.said = fail, []

    def swap_engine(self, new: Any) -> None:
        if self.fail:
            raise RuntimeError("model file is corrupt")
        self.engine = new

    def say(self, text: str) -> None:
        self.said.append(text)


def make_app(cfg: Config, tmp_path: Path, **over: Any) -> SimpleNamespace:
    ns = SimpleNamespace(
        cfg=Config(json.loads(json.dumps(cfg.raw)), cfg.root), sm=SimpleNamespace(state=State.IDLE), paused=False, muted=False,
        display_mode=False, memory=MemoryStore(tmp_path / "m.sqlite"), scheduler=SimpleNamespace(active_timers=lambda: []),
        client=SimpleNamespace(model="qwen2.5:3b", opts={"temperature": 0.6}, http=None),
        brain=SimpleNamespace(max_sentences=2, max_sentences_detail=6, filler_after_s=1.3, refresh_prompt_cache=lambda: True),
        speaker=FakeSpeaker(), overlay=SimpleNamespace(send_display=lambda _e: None), proactive=None, vault=None, index=None,
        context=SimpleNamespace(enabled=True, on={"foreground": True, "recent_files": True, "battery": True, "git": True, "calendar": True}),
        skills=SimpleNamespace(problems=[]), _turn_lock=threading.Lock(), toggle_pause=lambda: None, toggle_mute=lambda: None,
        kill=lambda: None)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def backend(cfg, tmp_path):
    app = make_app(cfg, tmp_path)
    be = AppBackend(app, tmp_path / "ev.local.toml")      # type: ignore[arg-type]
    yield be
    app.memory.close()


def test_settings_are_written_to_ev_local_toml_and_nothing_else_is_touched(backend, tmp_path):
    ev, pol = ROOT / "config" / "ev.toml", ROOT / "config" / "policy.toml"
    before = (digest(ev), digest(pol))
    out = backend.apply_settings({"llm.reply_max_sentences": 3, "conversation.followup_window_s": 9.5, "context.git": False})
    assert out["saved"] == ["context.git", "conversation.followup_window_s", "llm.reply_max_sentences"]
    assert tomllib.loads((tmp_path / "ev.local.toml").read_text())["llm"] == {"reply_max_sentences": 3}
    assert (digest(ev), digest(pol)) == before
    assert {p.name for p in tmp_path.iterdir() if p.is_file()} <= {"ev.local.toml", "m.sqlite", "m.sqlite-wal", "m.sqlite-shm"}


def test_live_settings_take_effect_now_and_restart_ones_say_so(backend):
    out = backend.apply_settings({"llm.reply_max_sentences": 4, "llm.temperature": 0.2, "trigger.wakeword_threshold": 0.7,
                                  "tts.sentence_pause_ms": 40, "context.git": False})
    assert backend.app.brain.max_sentences == 4 and backend.app.client.opts["temperature"] == 0.2
    assert backend.app.speaker.pause_ms == 40 and backend.app.context.on["git"] is False
    assert out["restart_needed"] == ["trigger.wakeword_threshold"]


def test_a_setting_that_is_not_listed_is_refused_and_nothing_is_written(backend, tmp_path):
    for key in ("builder.enabled", "vault.path", "web.trusted_hosts", "privacy.log_conversations", "llm.host", "general.policy_file",
                "context.repos", "dashboard.enabled", "policy.tools", "tools.run_command", "hotkeys.kill_switch", "llm.model"):
        with pytest.raises(DashboardError) as e:
            backend.apply_settings({key: True})
        assert e.value.status == 403
    with pytest.raises(DashboardError):                                      # one bad key spoils the whole request
        backend.apply_settings({"llm.reply_max_sentences": 3, "builder.enabled": True})
    assert not (tmp_path / "ev.local.toml").exists() and backend.app.brain.max_sentences == 2


@pytest.mark.parametrize("key,value", [("llm.reply_max_sentences", 99), ("llm.reply_max_sentences", 0), ("llm.reply_max_sentences", True),
                                       ("llm.reply_max_sentences", "3"), ("llm.temperature", float("nan")), ("llm.temperature", float("inf")),
                                       ("conversation.followup_window_s", -1), ("context.git", 1), ("context.git", "yes"),
                                       ("proactive.quiet_start", "25:00"), ("proactive.quiet_start", "22:30; rm"), ("proactive.quiet_start", 2230)])
def test_values_outside_their_type_or_range_are_refused(key, value):
    with pytest.raises(SettingError):
        coerce(key, value)


def test_ints_stay_ints_and_floats_stay_floats():
    assert isinstance(coerce("llm.reply_max_sentences", 3.0), int) and isinstance(coerce("llm.temperature", 1), float)


SAFE_SECTIONS = {"conversation", "trigger", "llm", "tts", "overlay", "context", "proactive"}
WIDENING = {"path", "dir", "file", "host", "port", "url", "repos", "source", "token", "trusted", "allow", "block", "policy", "builder",
            "hotkey", "model", "engine", "device", "key", "password", "rules"}          # whole words of a key's name (split on _)
NOT_WIDENING = {"overlay.display_mode", "context.enabled"}                       # EV's own text card / a switch for a read-only look


def widening_problems(allowed: dict[str, Any]) -> list[str]:
    """Everything in `allowed` that could widen what EV reaches or does, judged by section, by name and against every key of
    policy.toml (every policy value is file-only)."""
    pol = tomllib.loads((ROOT / "config" / "policy.toml").read_text(encoding="utf-8"))
    policy_keys = {f"{sec}.{k}" for sec, tab in pol.items() if isinstance(tab, dict) for k in tab}
    out = []
    for key in allowed:
        section, name = key.split(".", 1)
        if section not in SAFE_SECTIONS:
            out.append(f"{key}: not a tuning section")
        elif key not in NOT_WIDENING and WIDENING & set(name.split("_")):
            out.append(f"{key}: the name looks like it widens access")
        if key in policy_keys:
            out.append(f"{key}: is a policy value")
    return out


def test_the_allow_list_never_contains_anything_that_widens_access():
    """Walks ev.toml and policy.toml: a key that could widen what EV reaches or does must not be in the allow-list, so adding
    one there fails the build."""
    assert widening_problems(ALLOWED) == []
    ev = tomllib.loads((ROOT / "config" / "ev.toml").read_text(encoding="utf-8"))
    for key in ALLOWED:                                       # each allowed key belongs to a real section of ev.toml
        assert key.split(".")[0] in ev, key
    keys = [f"{sec}.{k}" for sec, tab in ev.items() if isinstance(tab, dict) for k in tab]
    assert len(keys) > 100                                    # the walk really covers the file
    # the sections that hold access, paths, hosts, privacy or the builder have no allowed key at all
    assert not {k for k in ALLOWED if k.split(".")[0] in {"web", "builder", "vault", "privacy", "dashboard", "general", "memory", "skills",
                                                          "screen", "control", "hotkeys", "audio", "stt", "vlm"}}


@pytest.mark.parametrize("key", ["web.trusted_hosts", "builder.enabled", "vault.path", "context.repos", "llm.host", "tools.run_command",
                                 "tts.piper_voice_dir", "proactive.rules_file", "context.calendar_source", "conversation.api_token"])
def test_the_widening_guard_catches_a_key_that_slips_in(key):
    assert widening_problems({key: None})


def test_the_local_file_keeps_other_keys_backs_up_once_and_survives_awkward_values(tmp_path):
    f = tmp_path / "ev.local.toml"
    f.write_text('[general]\nname = "x\\"y"\n\n[llm]\nmodel = "m"\n', encoding="utf-8")
    localconf.update(f, {"llm.temperature": 0.3, "proactive.quiet_start": "21:00"})
    localconf.update(f, {"llm.temperature": 0.4})
    data = tomllib.loads(f.read_text(encoding="utf-8"))
    assert data == {"general": {"name": 'x"y'}, "llm": {"model": "m", "temperature": 0.4}, "proactive": {"quiet_start": "21:00"}}
    assert "model = \"m\"" in (tmp_path / "ev.local.toml.bak").read_text(encoding="utf-8")


def test_a_broken_save_leaves_the_old_file(tmp_path, monkeypatch):
    f = tmp_path / "ev.local.toml"
    f.write_text("[llm]\ntemperature = 0.5\n", encoding="utf-8")
    monkeypatch.setattr(localconf, "dumps", lambda _d: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError):
        localconf.update(f, {"llm.temperature": 0.9})
    assert tomllib.loads(f.read_text(encoding="utf-8")) == {"llm": {"temperature": 0.5}}


def test_proactive_guards_are_tuned_by_the_settings_and_stay_in_range(tmp_path):
    from dataclasses import replace

    from ev.proactive.rules import Guards, apply_overrides

    g, problems = apply_overrides(Guards(), {"max_spoken_per_hour": 2, "quiet_start": "21:00", "defer_minutes": 5})
    assert (g.max_spoken_per_hour, g.quiet_hours, g.defer_minutes, problems) == (2, ("21:00", "07:30"), 5.0, [])
    g, problems = apply_overrides(replace(Guards(), max_spoken_per_hour=3), {"max_spoken_per_hour": 9999, "quiet_end": "late"})
    assert g.max_spoken_per_hour == 3 and g.quiet_hours == ("22:30", "07:30") and len(problems) == 2


def test_proactive_settings_apply_live_to_the_running_engine(cfg, tmp_path):
    from ev.proactive.rules import Guards

    engine = SimpleNamespace(guards=Guards())
    app = make_app(cfg, tmp_path, proactive=engine)
    be = AppBackend(app, tmp_path / "ev.local.toml")      # type: ignore[arg-type]
    be.apply_settings({"proactive.max_spoken_per_hour": 1, "proactive.quiet_start": "20:00"})
    assert engine.guards.max_spoken_per_hour == 1 and engine.guards.quiet_hours == ("20:00", "07:30")
    app.memory.close()


# ---------------------------------------------------------------- voice and model

def test_a_voice_that_fails_to_load_leaves_the_old_one_speaking(backend, monkeypatch):
    monkeypatch.setattr(backend, "_piper_voices", lambda: ["en_US-amy-medium"])
    monkeypatch.setattr("ev.dashboard.backend.make_engine", lambda *_a, **_k: SimpleNamespace(name="piper"))
    old = backend.app.speaker.engine
    backend.app.speaker.fail = True
    with pytest.raises(DashboardError) as e:
        backend.set_voice("piper", "en_US-amy-medium", 1.0, False, True)
    assert e.value.status == 422 and backend.app.speaker.engine is old
    assert backend.app.cfg["tts"]["piper_voice"] == "en_US-hfc_female-medium" and not backend.local_file.exists()


def test_a_voice_that_is_not_installed_or_a_silly_speed_is_refused(backend, monkeypatch):
    monkeypatch.setattr(backend, "_piper_voices", lambda: ["en_US-amy-medium"])
    for args in (("piper", "../../etc/passwd", 1.0), ("piper", "en_GB-alan-medium", 1.0), ("sapi", "x", 1.0), ("piper", "en_US-amy-medium", 9.0)):
        with pytest.raises(DashboardError):
            backend.set_voice(*args, False, False)
    assert backend.app.speaker.engine.name == "piper" and backend.app.speaker.said == []


def test_a_good_voice_is_swapped_previewed_and_saved(backend, monkeypatch):
    new = SimpleNamespace(name="piper", voice_name="en_US-amy-medium")
    seen: list[Config] = []
    monkeypatch.setattr(backend, "_piper_voices", lambda: ["en_US-amy-medium"])
    monkeypatch.setattr("ev.dashboard.backend.make_engine", lambda c, _k: seen.append(c) or new)
    backend.set_voice("piper", "en_US-amy-medium", 1.25, True, True)
    assert backend.app.speaker.engine is new and backend.app.speaker.said
    assert seen[0]["tts"]["piper_voice"] == "en_US-amy-medium" and seen[0]["tts"]["length_scale"] == 0.8
    assert tomllib.loads(backend.local_file.read_text())["tts"] == {"engine": "piper", "piper_voice": "en_US-amy-medium", "length_scale": 0.8}


def test_choosing_a_chat_model_checks_the_list_and_warns_about_size(backend):
    class Resp:
        def raise_for_status(self) -> None: ...
        def json(self) -> dict[str, Any]:
            return {"models": [{"name": "qwen2.5:3b", "size": 1.9e9}, {"name": "llama3.1:8b", "size": 4.9e9}]}

    backend.app.client.http = SimpleNamespace(get=lambda _p: Resp())
    m = backend.models()
    assert [(x["name"], x["too_big"]) for x in m["models"]] == [("qwen2.5:3b", False), ("llama3.1:8b", True)]
    with pytest.raises(DashboardError):
        backend.set_model("evil:latest", True)
    backend.set_model("llama3.1:8b", False)
    assert backend.app.client.model == "llama3.1:8b" and not backend.local_file.exists()


# ---------------------------------------------------------------- skills and memory go through the gate

@pytest.fixture
def skills_rig(cfg, tmp_path):
    sdir = tmp_path / "skills"
    shutil.copytree(ROOT / "skills", sdir)
    pol = Policy.load(ROOT / "config" / "policy.toml", places={"skills": sdir, "vault_ev": tmp_path / "v", "screenshots": tmp_path / "s"})
    store = SkillStore(sdir, pol)
    store.load()
    return lambda answer=False: _skills_backend(cfg, pol, store, tmp_path, answer)


def _skills_backend(cfg, pol, store, tmp_path, answer):
    rig = Rig(cfg, pol, store, [text_reply("unused")], answer)
    app = make_app(cfg, tmp_path, brain=rig.brain, gate=rig.gate, skills=store, registry=rig.brain.registry)
    return rig, store, AppBackend(app, tmp_path / "ev.local.toml")      # type: ignore[arg-type]


def test_running_a_skill_from_the_dashboard_gates_every_step(skills_rig):
    rig, _store, be = skills_rig()
    out = be.run_skill("work mode")
    assert out["ok"] and rig.tools() == ["open_app", "open_website"]
    assert [e["tool"] for e in rig.gate.audit] == ["run_skill", "open_app", "open_website"]


def test_a_risky_step_in_a_dashboard_run_still_asks_out_loud(skills_rig):
    from ev.skills.store import Skill, Step

    rig, store, be = skills_rig(answer=False)
    store.save(Skill("tidy", "d", ["tidy up"], [Step("set_volume", {"level": 10}), Step("close_app", {"name": "chrome", "force": True})]))
    out = be.run_skill("tidy")
    assert not out["ok"] and rig.asked == ["force close chrome. Anything unsaved in it will be lost"] and rig.tools() == ["set_volume"]


def test_a_blocked_step_cannot_be_run_through_the_dashboard_either(skills_rig):
    from ev.skills.store import Skill, Step

    rig, store, be = skills_rig(answer=True)
    store.save(Skill("sneaky", "d", ["sneaky"], [Step("open_app", {"name": "cmd.exe"})]))
    out = be.run_skill("sneaky")
    assert not out["ok"] and rig.tools() == [] and "not allowed" in out["message"]


def test_a_dashboard_run_waits_if_a_conversation_is_going_on(skills_rig):
    rig, _s, be = skills_rig()
    be.app._turn_lock.acquire()
    with pytest.raises(DashboardError) as e:
        be.run_skill("work mode")
    assert e.value.status == 409 and rig.tools() == []


def test_deleting_a_skill_needs_a_spoken_yes(skills_rig):
    rig, store, be = skills_rig(answer=False)
    assert be.delete_skill("work mode")["deleted"] is False and store.get("work mode") is not None
    assert rig.asked and "work mode" in rig.asked[0]
    rig2, store2, be2 = skills_rig(answer=True)
    assert be2.delete_skill("work mode") == {"deleted": True} and store2.get("work mode") is None


def test_saving_a_skill_from_the_dashboard_is_validated(skills_rig):
    _rig, store, be = skills_rig()
    ok = {"name": "quiet time", "description": "d", "triggers": ["quiet time"], "steps": [{"tool": "set_volume", "args": {"level": 5}}]}
    assert be.save_skill(ok) == {"saved": "quiet time"} and store.get("quiet time") is not None
    for bad in ({**ok, "name": "x/../y"}, {**ok, "steps": [{"tool": "run_skill", "args": {"name": "work mode"}}]},
                {**ok, "steps": [{"tool": "remember_fact", "args": {"key": "a", "value": "b"}}]},
                {**ok, "steps": [{"tool": "no_such_tool", "args": {}}]}, {**ok, "steps": []}, {**ok, "triggers": "nope"},
                {**ok, "steps": [{"tool": "set_volume", "args": {"level": {"x": 1}}}]}):
        with pytest.raises(DashboardError):
            be.save_skill(bad)


def test_a_skill_edit_cannot_pick_its_own_instructions_file(skills_rig):
    _rig, store, be = skills_rig()
    be.save_skill({"name": "work mode", "description": "d", "triggers": ["work mode"], "instructions": "../../secrets.md",
                   "steps": [{"tool": "set_volume", "args": {"level": 5}}]})
    assert (store.get("work mode").instructions_file or "") == ""


class MemRig:
    def __init__(self, cfg, pol, tmp_path, answer):
        from ev.llm.brain import Brain
        from ev.llm.client import OllamaClient
        from ev.memory.vault import Vault
        from ev.safety.gate import SafetyGate
        from ev.safety.provenance import TurnState
        from ev.tools.builtin import ToolContext, build_tools
        from ev.tools.scheduler import Scheduler

        from .fakes import FakeOllama

        self.asked: list[str] = []
        self.memory = MemoryStore(tmp_path / "mem.sqlite")
        self.vault = Vault(tmp_path / "vault", self.memory, policy=pol)
        self.vault.ensure()
        self.turn = TurnState()
        reg = build_tools(ToolContext(cfg, self.memory, Scheduler(self.memory, lambda _t: None), pol, vault=self.vault, turn=self.turn,
                                      notes_dir=tmp_path / "notes"))
        self.gate = SafetyGate(pol, ask=lambda r: self.asked.append(r) or answer)
        self.brain = Brain(cfg, OllamaClient(cfg, transport=FakeOllama([text_reply("x")]).transport()), self.memory, reg, self.gate,
                           persona="You are EV.", vault=self.vault, turn=self.turn)


@pytest.fixture
def mem_rig(cfg, tmp_path):
    made: list[MemRig] = []
    pol = Policy.load(ROOT / "config" / "policy.toml", places={"vault_ev": tmp_path / "vault" / "EV", "skills": tmp_path / "skills",
                                                               "screenshots": tmp_path / "s"})

    def make(answer: bool) -> tuple[MemRig, AppBackend]:
        rig = MemRig(cfg, pol, tmp_path, answer)
        made.append(rig)
        app = make_app(cfg, tmp_path, brain=rig.brain, gate=rig.gate, vault=rig.vault, memory=rig.memory, registry=rig.brain.registry)
        return rig, AppBackend(app, tmp_path / "ev.local.toml")      # type: ignore[arg-type]

    yield make
    for r in made:
        r.memory.close()


def test_forgetting_a_fact_from_the_dashboard_needs_a_spoken_yes(mem_rig):
    rig, be = mem_rig(False)
    rig.vault.add_fact("tea", "green", "user said")
    out = be.forget("tea")
    assert not out["ok"] and rig.asked and "tea" in rig.asked[0] and rig.vault.profile_facts()["tea"] == "green"
    rig2, be2 = mem_rig(True)
    rig2.vault.add_fact("tea", "green", "user said")
    assert be2.forget("tea")["ok"] and "tea" not in rig2.vault.profile_facts()


def test_confirming_an_unconfirmed_fact_is_a_spoken_readback_never_a_click(mem_rig):
    rig, be = mem_rig(False)
    path = rig.vault.ev_dir / "Facts.md"
    path.write_text(path.read_text(encoding="utf-8") + "- tea: green (source: user said, 2026-01-01)\n", encoding="utf-8")
    listed = be.facts()["unconfirmed"]
    assert [(u["key"], u["value"]) for u in listed] == [("tea", "green")] and "tea" not in be.facts()["facts"]
    out = be.keep(listed[0]["id"])
    assert not out["ok"] and rig.asked and "tea" in rig.asked[0] and "tea" not in rig.vault.profile_facts()
    rig2, be2 = mem_rig(True)
    p2 = rig2.vault.ev_dir / "Facts.md"
    p2.write_text(p2.read_text(encoding="utf-8") + "- tea: green (source: user said, 2026-01-01)\n", encoding="utf-8")
    assert be2.keep(be2.facts()["unconfirmed"][0]["id"])["ok"] and rig2.vault.profile_facts()["tea"] == "green"


def test_the_dashboard_cannot_ask_for_a_tool_the_model_could_not(mem_rig):
    rig, be = mem_rig(True)
    assert be._gated("run_command", {"command": "calc"})["ok"] is False


def test_the_policy_is_shown_but_there_is_nothing_to_save_it_with(backend):
    out = backend.policy_text()
    assert out["read_only"] and "[tools]" in out["text"]
    assert not any(n in dir(backend) for n in ("save_policy", "set_policy", "write_policy", "apply_policy"))


def test_pause_mute_and_stop_are_the_only_controls(backend):
    for a in ("pause", "resume", "mute", "unmute", "kill"):
        assert backend.control(a)["ok"]
    with pytest.raises(DashboardError):
        backend.control("quit")
    with pytest.raises(DashboardError):
        backend.control("unlock")


def test_open_dashboard_is_voice_only_and_handled_without_the_model(cfg):
    from ev.llm.fastpath import match_fast

    for said in ("open the dashboard", "show me the dashboard", "pull up my dashboard please", "okay open the dashboard"):
        assert (m := match_fast(said)) and m.tool == "open_dashboard", said
    assert (m := match_fast("open spotify")) and m.tool == "open_app"
    reg = ToolRegistry()
    reg.register(Tool("open_dashboard", "d", {}, lambda: ToolResult(True, "ok"), (), llm_visible=False))
    assert reg.schemas() == []


# ---------------------------------------------------------------- the page itself, and the real app

def test_the_page_puts_text_in_as_text_and_has_no_inline_script_or_style():
    static = ROOT / "ev" / "dashboard" / "static"
    for name in ("app.js", "orb.js"):
        js = (static / name).read_text(encoding="utf-8")
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "srcdoc", "javascript:"):
            assert bad not in js, (name, bad)
        assert " style=" not in js and ".setAttribute(\"style\"" not in js, name
        if name == "orb.js":                                             # the orb only draws on a canvas: no network, no markup
            assert not any(w in js for w in ("fetch(", "XMLHttpRequest", "EventSource", "WebSocket", "createElement")), name
    html = (static / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html and " style=" not in html and "onclick=" not in html and "http://" not in html and "https://" not in html
    css = (static / "app.css").read_text(encoding="utf-8")
    assert "@import" not in css and "url(http" not in css and "https://" not in css and "http://" not in css     # no outside fonts, images or sheets
    assert 'src="/orb.js"' in html and "/orb.js" in FILES


def test_the_real_app_serves_the_dashboard_and_open_dashboard_hands_out_a_working_link(tmp_path, monkeypatch):
    from ev.app import EVApp
    from ev.core.config import load_config

    cfg = load_config(overrides={"general": {"data_dir": str(tmp_path / "d"), "log_dir": str(tmp_path / "l")},
                                 "vault": {"path": str(tmp_path / "v")}, "overlay": {"enabled": False}, "tray": {"enabled": False},
                                 "dashboard": {"enabled": True, "port": 0}})
    app = EVApp(cfg)
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    try:
        app._start_dashboard()
        assert app.dashboard is not None and app.registry.get("open_dashboard").llm_visible is False
        res = app.brain.respond("open the dashboard")
        assert res.fast_path and len(opened) == 1 and opened[0].startswith(f"http://127.0.0.1:{app.dashboard.port}/?t=")
        b = Browser(app.dashboard)
        status, _hdr, _ = b.request("GET", opened[0].split(str(app.dashboard.port), 1)[1], cookie=False)
        assert status == 302
        b.cookie = _hdr["set-cookie"].split(";", 1)[0]
        for path in ("/api/status", "/api/settings", "/api/voices", "/api/skills", "/api/facts", "/api/policy"):
            code, _h, body = b.request("GET", path)
            assert code in (200, 503), (path, code, body)
        code, _h, body = b.request("GET", "/api/status")
        info = json.loads(body)
        assert info["state"] == "idle" and info["model"] and info["vault"]["unconfirmed"] == 0
        app.hub.publish("turn", user="hi", reply="hello")
        assert app.hub.recent()["turn"][-1]["reply"] == "hello"
    finally:
        app.shutdown()
    assert app.dashboard is not None and app.dashboard.stopping.is_set()
