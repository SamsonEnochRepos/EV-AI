"""The proactive engine: the rules, each source, each guard, and the skill path (which never asks a question out of the blue)."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from ev.context.calendar import CachedCalendar, Event
from ev.core.config import ROOT
from ev.llm.client import LLMError
from ev.proactive.engine import Actions, ProactiveEngine, fill, safe_field
from ev.proactive.probes import Probes
from ev.proactive.rules import Guards, RuleError, load_rules, parse_guards, parse_rule
from ev.tools.base import Tool, ToolResult

from .fakes import text_reply
from .test_skills import Rig, pol, sdir, store  # noqa: F401  (fixtures)

MON_9AM = dt.datetime(2026, 11, 2, 9, 5)          # a Monday


class Env:
    """An engine with a clock, probes and actions the test controls."""

    def __init__(self, rules: list[dict[str, Any]], guards: Guards | None = None, now: dt.datetime = MON_9AM, calendar=None, memory=None):
        self.dt = now
        self.t = 1000.0
        self.epoch_s = 1_800_000_000.0
        self.state_ok = True
        self.idle: float | None = 5.0
        self.full = False
        self.call = False
        self.fg = "code"
        self.batt: tuple[int, bool] | None = (80, False)
        self.procs: dict[int, tuple[str, float]] = {}
        self.listing: dict[str, tuple[float, int]] = {}
        self.spoken: list[str] = []
        self.notices: list[str] = []
        self.skill_calls: list[str] = []
        self.skill_says: str | None = "Good morning."
        probes = Probes(idle_s=lambda: self.idle, fullscreen=lambda: self.full, in_call=lambda: self.call,
                        foreground_exe=lambda: self.fg, battery=lambda: self.batt, processes=lambda names: dict(self.procs),
                        files=lambda pattern: dict(self.listing))
        actions = Actions(self.spoken.append, self.notices.append, lambda name: (self.skill_calls.append(name), self.skill_says)[1])
        g = guards or Guards(quiet_hours=None, max_spoken_per_hour=4, defer_minutes=10, process_poll_s=0)
        self.engine = ProactiveEngine([parse_rule(r) for r in rules], g, probes, actions, memory=memory, calendar=calendar,
                                      state_ok=lambda: self.state_ok, now=lambda: self.dt, clock=lambda: self.t,
                                      epoch=lambda: self.epoch_s)

    def advance(self, seconds: float = 0, minutes: float = 0) -> None:
        s = seconds + minutes * 60
        self.t += s
        self.epoch_s += s
        self.dt += dt.timedelta(seconds=s)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.engine.tick()


def rule(**kw: Any) -> dict[str, Any]:
    return {"name": "r", "source": "battery", "say": "hi", **kw}


# ---------------------------------------------------------------- rule files

def test_the_shipped_rules_load_without_problems_and_the_meeting_rule_ships_off():
    rules, guards, problems = load_rules(ROOT / "config" / "proactive.toml")
    assert problems == [] and {r.name: r.enabled for r in rules} == {
        "low battery": True, "long build finished": True, "morning brief": True, "meeting soon": False}
    assert guards.quiet_hours == ("22:30", "07:30") and guards.max_spoken_per_hour == 4 and guards.check_fullscreen and guards.check_calls


@pytest.mark.parametrize("bad", [
    {"source": "nonsense"}, {"say": "", "overlay": "", "skill": ""}, {"name": ""}, {"name": "x" * 60},
    {"source": "time", "at": "25:00"}, {"source": "time", "at": "9"}, {"source": "time", "at": "09:00", "days": ["funday"]},
    {"source": "battery", "below": 0}, {"source": "battery", "below": "low"}, {"source": "battery", "below": True},
    {"source": "battery", "say": "{nope}"}, {"source": "battery", "overlay": "{title}"},
    {"source": "process_exit", "names": []}, {"source": "process_exit", "names": ["a b"]}, {"source": "process_exit", "names": ["x"] * 21},
    {"source": "file", "glob": ""}, {"source": "file", "glob": "C:\\"}, {"source": "file", "glob": "~/../x/*"},
    {"source": "calendar", "minutes_before": 0}, {"source": "event", "topic": "Bad Topic"}, {"say": "x" * 200},
    {"source": "app_time", "minutes": 1},
])
def test_invalid_rules_are_refused(bad):
    with pytest.raises(RuleError):
        parse_rule({"name": "r", "source": "battery", "say": "hi", **bad})


def test_a_template_may_only_use_fields_its_source_provides():
    parse_rule({"name": "r", "source": "battery", "say": "{percent} percent"})
    parse_rule({"name": "r", "source": "process_exit", "names": ["cargo"], "say": "{name} after {minutes}"})
    with pytest.raises(RuleError, match="isn't available"):
        parse_rule({"name": "r", "source": "battery", "say": "{name}"})


def test_bad_rules_and_bad_files_are_skipped_not_fatal(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text('[[rules]]\nname="ok"\nsource="battery"\nsay="hi"\n[[rules]]\nname="bad"\nsource="time"\nat="99:99"\nsay="x"\n'
                 '[[rules]]\nname="ok"\nsource="battery"\nsay="dup"\n[guards]\nquiet_hours=["late","early"]\n', encoding="utf-8")
    rules, guards, problems = load_rules(p)
    assert [r.name for r in rules] == ["ok"] and len(problems) == 3 and guards == Guards()
    p.write_text("not [toml", encoding="utf-8")
    assert load_rules(p)[0] == [] and len(load_rules(p)[2]) == 1
    assert load_rules(tmp_path / "missing.toml") == ([], Guards(), [])


def test_guards_parse_and_bound_their_values():
    assert parse_guards({"quiet_hours": []}).quiet_hours is None
    with pytest.raises(RuleError):
        parse_guards({"max_spoken_per_hour": 9999})
    with pytest.raises(RuleError):
        parse_guards({"quiet_hours": ["22:30"]})


# ---------------------------------------------------------------- sources

def test_time_rule_fires_once_in_its_window_on_its_days():
    e = Env([rule(source="time", at="09:00", days=["mon", "tue"], window_min=30, say="It's {time}.")], now=dt.datetime(2026, 11, 2, 8, 59))
    e.tick()
    assert e.spoken == []
    e.advance(minutes=2)
    e.tick(3)
    assert e.spoken == ["It's 9:00 AM."]
    e.advance(minutes=60)
    e.tick()
    assert e.spoken == ["It's 9:00 AM."]                                        # once a day, and not after the window
    e.advance(minutes=24 * 60 - 60)                                              # Tuesday 09:01
    e.tick()
    assert len(e.spoken) == 2
    e.advance(minutes=24 * 60)                                                   # Wednesday: not a listed day
    e.tick()
    assert len(e.spoken) == 2


def test_battery_fires_below_the_line_once_and_rearms_when_charged():
    e = Env([rule(source="battery", below=20, say="Battery {percent}.")])
    e.batt = (50, False)
    e.tick()
    e.batt = (19, False)
    e.tick(3)
    assert e.spoken == ["Battery 19."]
    e.batt = (15, False)
    e.tick()
    assert len(e.spoken) == 1                                                    # still low: not again
    e.batt = (15, True)                                                          # plugged in: re-armed
    e.tick()
    e.batt = (18, False)
    e.tick()
    assert len(e.spoken) == 2
    e.batt = (18, True)
    e.tick()
    assert len(e.spoken) == 2                                                    # charging: no complaint
    assert Env([rule()]).engine is not None


def test_calendar_rule_fires_once_per_event_and_only_inside_the_window():
    ev = Event(MON_9AM + dt.timedelta(minutes=9), MON_9AM + dt.timedelta(minutes=40), "Standup <b>now</b>")
    cal = CachedCalendar(type("S", (), {"events_between": lambda self, a, b: [ev]})(), now=lambda: MON_9AM)
    cal.refresh()
    e = Env([rule(source="calendar", minutes_before=10, say="{title} in {minutes}.", overlay="{title}")], calendar=cal)
    e.tick(3)
    assert e.spoken == ["Standup now in 9."] and e.notices == ["Standup now"]
    e2 = Env([rule(source="calendar", minutes_before=5, say="x")], calendar=cal)
    e2.tick()
    assert e2.spoken == []                                                       # nine minutes away, window is five
    assert Env([rule(source="calendar", say="x")]).engine._due(parse_rule(rule(source="calendar", say="x")), MON_9AM, True) == []


def test_process_exit_fires_for_a_long_enough_job_and_ignores_short_ones_and_the_first_look():
    e = Env([rule(source="process_exit", names=["cargo"], min_runtime_s=120, say="{name} done after {minutes} minutes.")])
    e.procs = {10: ("cargo", e.epoch_s - 570), 11: ("cargo", e.epoch_s - 20)}
    e.tick()                                                                     # first look: what is running is the baseline
    assert e.spoken == []
    e.advance(seconds=30)
    e.procs = {}
    e.tick()
    assert e.spoken == ["cargo done after 10 minutes."]                          # the 10-minute job; the 50-second one is too short


def test_file_rule_waits_for_the_file_to_stop_growing_and_ignores_what_was_there():
    e = Env([rule(source="file", glob="~/Downloads/*.pdf", say="New file {name}.")])
    e.listing = {"C:/x/old.pdf": (1.0, 10)}
    e.tick()
    assert e.spoken == []
    e.listing["C:/x/new.pdf"] = (5.0, 100)
    e.tick()
    e.listing["C:/x/new.pdf"] = (6.0, 400)                                       # still being written
    e.tick()
    assert e.spoken == []
    e.tick()                                                                     # unchanged on two polls in a row
    assert e.spoken == ["New file new.pdf."]
    e.tick(3)
    assert len(e.spoken) == 1


def test_app_time_fires_once_per_stretch_and_a_long_break_starts_a_new_one():
    e = Env([rule(source="app_time", exe="code", minutes=90, say="{minutes} minutes in {app}.")])

    def work(minutes: int) -> None:                                              # the engine ticks every couple of seconds in life
        for _ in range(minutes):
            e.advance(minutes=1)
            e.tick()

    e.fg = "code"
    e.tick()
    work(91)
    assert e.spoken == ["90 minutes in code."]
    work(30)
    assert len(e.spoken) == 1                                                    # once per stretch
    e.fg = "firefox"
    for _ in range(5):                                                           # five minutes in another app: the stretch is over
        e.advance(minutes=1)
        e.tick()
    e.fg = "code"
    work(60)
    assert len(e.spoken) == 1                                                    # a fresh stretch, only an hour old
    work(31)
    assert len(e.spoken) == 2


def test_event_rules_fire_on_their_topic_with_sanitised_fields():
    e = Env([rule(source="event", topic="job_done", say="{label} is done, {count} files, {summary}.")])
    assert e.engine.fire_event("other") == 0
    n = e.engine.fire_event("job_done", label="the weather retry <script>", count=3, summary="x" * 200, secret="hidden")
    e.tick()
    assert n == 1 and len(e.spoken) == 1 and "<" not in e.spoken[0] and "hidden" not in e.spoken[0] and "3 files" in e.spoken[0]
    assert len(safe_field("y" * 500)) <= 60 and safe_field("a\n\n  b\u200b") == "a b"
    assert fill("{a} and {b}", {"a": "1"}) == "1 and"


# ---------------------------------------------------------------- guards

def blocked(e: Env, **setup: Any) -> str:
    for k, v in setup.items():
        setattr(e, k, v)
    e.engine.fire_event("t")
    e.tick()
    return e.engine.last_block


@pytest.fixture
def env():
    return Env([rule(source="event", topic="t", say="hello", active_within_s=300)])


@pytest.mark.parametrize("setup,why", [
    ({"state_ok": False}, "busy"), ({"full": True}, "full-screen app"), ({"call": True}, "call"), ({"fg": "zoom"}, "call"),
    ({"idle": 4000.0}, "nobody there"), ({"idle": 400.0}, "not active")])
def test_each_guard_holds_a_spoken_line_back(env, setup, why):
    assert blocked(env, **setup) == why and env.spoken == []
    for k in setup:                                                              # the condition ends: the held line goes out
        setattr(env, k, {"state_ok": True, "full": False, "call": False, "fg": "code", "idle": 5.0}[k])
    env.advance(seconds=30)
    env.tick()
    assert env.spoken == ["hello"]


def test_quiet_hours_wrap_midnight_and_a_held_line_is_dropped_not_delivered_stale():
    g = Guards(quiet_hours=("22:30", "07:30"), defer_minutes=10, process_poll_s=0)
    e = Env([rule(source="event", topic="t", say="hello")], g, now=dt.datetime(2026, 11, 2, 23, 0))
    e.engine.fire_event("t")
    e.tick()
    assert e.spoken == [] and e.engine.last_block == "quiet hours"
    e.advance(minutes=11)
    e.tick()
    assert e.engine._pending == [] and e.spoken == [] and e.engine.log[-1]["kind"] == "dropped"
    e.dt = dt.datetime(2026, 11, 3, 7, 31)                                       # morning: allowed again
    e.engine.fire_event("t")
    e.tick()
    assert e.spoken == ["hello"]
    e.dt = dt.datetime(2026, 11, 3, 3, 0)
    assert e.engine.speech_blocked() == "quiet hours"


def test_the_hourly_cap_holds_the_fifth_line_until_the_hour_has_passed():
    e = Env([rule(source="event", topic="t", say="line", cooldown_s=0)], Guards(quiet_hours=None, max_spoken_per_hour=4, defer_minutes=120, process_poll_s=0))
    for _ in range(5):
        e.engine.fire_event("t")
        e.tick()
        e.advance(minutes=1)
    assert len(e.spoken) == 4 and e.engine.last_block == "hourly cap"
    e.advance(minutes=60)
    e.tick()
    assert len(e.spoken) == 5


def test_a_notice_shows_in_quiet_hours_and_calls_but_never_over_a_full_screen_app():
    g = Guards(quiet_hours=("22:30", "07:30"), process_poll_s=0)
    e = Env([rule(source="event", topic="t", say="", overlay="Heads up")], g, now=dt.datetime(2026, 11, 2, 23, 0))
    e.call = True
    e.full = True
    e.engine.fire_event("t")
    e.tick()
    assert e.notices == []
    e.full = False
    e.tick()
    assert e.notices == ["Heads up"] and e.spoken == []                          # silent card: fine at night, fine in a call


def test_snooze_holds_speech_survives_a_restart_and_can_be_ended(memory):
    e = Env([rule(source="event", topic="t", say="hello")], memory=memory)
    e.engine.snooze(60)
    assert e.engine.snoozed and e.engine.snooze_minutes_left() == 60
    e.engine.fire_event("t")
    e.tick()
    assert e.spoken == [] and e.engine.last_block == "snoozed"
    again = Env([rule(source="event", topic="t", say="hello")], memory=memory)
    again.epoch_s = e.epoch_s + 600
    assert again.engine.snoozed                                                  # remembered across a restart (kv)
    e.engine.snooze(0)
    e.tick()
    assert e.spoken == ["hello"]


def test_cooldown_between_firings():
    e = Env([rule(source="event", topic="t", say="hi", cooldown_s=600)])
    e.engine.fire_event("t")
    e.tick()
    e.advance(minutes=5)
    e.engine.fire_event("t")
    e.tick()
    assert len(e.spoken) == 1
    e.advance(minutes=6)
    e.engine.fire_event("t")
    e.tick()
    assert len(e.spoken) == 2


def test_a_failing_action_is_not_retried_forever_and_a_failing_source_does_not_stop_the_others():
    e = Env([rule(name="a", source="event", topic="t", say="x"), rule(name="b", source="battery", below=50, say="low {percent}")])
    calls = []
    e.engine.actions = Actions(lambda s: (calls.append(s), 1 / 0)[0], e.notices.append, lambda n: None)
    e.engine.fire_event("t")
    e.batt = (10, False)
    e.tick(3)
    assert calls.count("x") == 1                                                 # raised once, never repeated
    e.engine.probes.battery = lambda: (_ for _ in ()).throw(RuntimeError("probe died"))
    e.tick()                                                                     # a dead probe doesn't raise out of tick()


def test_a_rule_with_a_skill_speaks_what_the_skill_returns_and_nothing_if_it_returns_nothing():
    e = Env([rule(source="event", topic="t", say="", skill="morning brief")])
    e.engine.fire_event("t")
    e.tick()
    assert e.skill_calls == ["morning brief"] and e.spoken == ["Good morning."]
    e.skill_says = None
    e.engine.fire_event("t")
    e.tick()
    assert e.spoken == ["Good morning."]
    e.engine.guards = Guards(quiet_hours=None, check_calls=False, defer_minutes=0)
    e.skill_says = "x"
    e.state_ok = False
    e.engine.fire_event("t")
    e.tick()
    assert e.skill_calls == ["morning brief", "morning brief"]                   # a blocked rule doesn't even start its skill


def test_the_engine_thread_starts_and_stops(memory):
    e = Env([rule(source="event", topic="t", say="hello")])
    e.engine.guards = Guards(quiet_hours=None, tick_s=0.2, process_poll_s=0)
    e.engine.fire_event("t")
    e.engine.start()
    import time as _t
    deadline = _t.time() + 5
    while _t.time() < deadline and not e.spoken:
        _t.sleep(0.05)
    e.engine.stop()
    assert e.spoken == ["hello"]


# ---------------------------------------------------------------- through the Brain: skills that run on their own

@pytest.fixture
def quiet_rig(cfg, pol, store):                                                  # noqa: F811
    return lambda replies=None, answer=True: Rig(cfg, pol, store, replies or [text_reply("unused")], answer)


def test_a_proactive_skill_runs_its_steps_through_the_gate(quiet_rig):
    r = quiet_rig()
    said = r.brain.run_skill_proactive("work mode")
    assert said == "open_app ok open_website ok Okay, let's get to work." and r.tools() == ["open_app", "open_website"]
    assert [e["tool"] for e in r.gate.audit] == ["run_skill", "open_app", "open_website"]


def test_a_proactive_skill_never_asks_a_question_out_of_the_blue(quiet_rig, store):                                  # noqa: F811
    from ev.skills.store import Skill, Step

    store.save(Skill("tidy", "d", ["tidy up"], [Step("set_volume", {"level": 10}), Step("close_app", {"name": "chrome", "force": True}),
                                                Step("set_timer", {"minutes": 1})]))
    r = quiet_rig(answer=True)                                                   # even with a "yes" ready, nobody is asked
    said = r.brain.run_skill_proactive("tidy")
    assert r.asked == [] and r.tools() == ["set_volume"]
    assert said.startswith("Stopped at step 2, close app: not allowed: that needs your approval, and I don't ask for it on my own")


def test_a_proactive_skill_that_is_blocked_or_unknown_says_nothing_or_names_the_step(quiet_rig, store):        # noqa: F811
    from ev.skills.store import Skill, Step

    store.save(Skill("sneaky", "d", ["sneaky"], [Step("open_app", {"name": "cmd.exe"})]))
    r = quiet_rig()
    assert r.brain.run_skill_proactive("sneaky").startswith("Stopped at step 1, open app: not allowed")
    assert r.brain.run_skill_proactive("no such skill") is None and r.tools() == []
    r.gate.kill()
    assert r.brain.run_skill_proactive("work mode") is None


def test_a_proactive_skill_that_read_outside_text_is_phrased_by_the_model_with_no_tools(quiet_rig):
    r = quiet_rig([text_reply("Good morning. Standup is first. Two reminders. Have a good one. Extra fourth sentence.")])
    said = r.brain.run_skill_proactive("morning brief")
    req = r.fake.requests[0]
    assert "tools" not in req and '<untrusted source="skill"' in req["messages"][-1]["content"] and "09:00 Standup" in req["messages"][-1]["content"]
    assert said == "Good morning. Standup is first. Two reminders." and r.asked == []


def test_if_the_model_cannot_phrase_the_update_nothing_is_said(quiet_rig, monkeypatch):
    r = quiet_rig()
    monkeypatch.setattr(r.brain.client, "chat", lambda *a, **k: (_ for _ in ()).throw(LLMError("down")))
    assert r.brain.run_skill_proactive("morning brief") is None


def test_the_phrasing_call_drops_sentences_that_name_a_tool(quiet_rig):
    r = quiet_rig([text_reply("I used get_weather for you. It is sunny.")])
    assert r.brain.phrase_for_speech("sunny", "") == "It is sunny."


# ---------------------------------------------------------------- the app

def test_the_app_builds_the_engine_with_the_shipped_rules_and_a_voice_only_snooze(tmp_path):
    from ev.app import EVApp
    from ev.core.config import load_config

    cfg = load_config(overrides={"general": {"data_dir": str(tmp_path / "d"), "log_dir": str(tmp_path / "l")},
                                 "vault": {"path": str(tmp_path / "v")}, "overlay": {"enabled": False}, "tray": {"enabled": False}})
    app = EVApp(cfg)
    try:
        assert app.proactive is not None and [r.name for r in app.proactive.rules] == ["low battery", "long build finished", "morning brief"]
        assert app.proactive.problems == []
        tool = app.registry.get("snooze_proactive")
        assert tool is not None and tool.llm_visible is False
        assert tool.func(minutes=60).message == "Okay, I'll stay quiet for 1 hour." and app.proactive.snoozed
        assert tool.func(minutes=90).message == "Okay, I'll stay quiet for 90 minutes."
        assert tool.func(minutes=0).message.startswith("Okay, I'll speak up again") and not app.proactive.snoozed
        assert app._may_speak_up() is True
        app._following_up = True
        assert app._may_speak_up() is False
    finally:
        app.shutdown()


def test_proactive_can_be_switched_off(tmp_path):
    from ev.app import EVApp
    from ev.core.config import load_config

    cfg = load_config(overrides={"general": {"data_dir": str(tmp_path / "d"), "log_dir": str(tmp_path / "l")},
                                 "vault": {"path": str(tmp_path / "v")}, "overlay": {"enabled": False}, "tray": {"enabled": False},
                                 "proactive": {"enabled": False}})
    app = EVApp(cfg)
    try:
        assert app.proactive is None and app.registry.get("snooze_proactive") is None
    finally:
        app.shutdown()


def test_a_skill_run_by_a_rule_leaves_the_state_machine_idle(tmp_path):
    """Regression: on_acting asked for IDLE -> ACTING (refused), then ACTING -> THINKING became IDLE -> THINKING and stuck,
    which would also have stopped every later proactive line (it only speaks when idle)."""
    from ev.app import EVApp
    from ev.core.config import load_config
    from ev.core.events import State

    cfg = load_config(overrides={"general": {"data_dir": str(tmp_path / "d"), "log_dir": str(tmp_path / "l")},
                                 "vault": {"path": str(tmp_path / "v")}, "skills": {"dir": str(tmp_path / "skills")},
                                 "overlay": {"enabled": False}, "tray": {"enabled": False}})
    app = EVApp(cfg)
    try:
        app.registry.register(Tool("get_time_date", "t", {}, lambda: ToolResult(True, "It's nine."), direct=True))
        from ev.skills.store import Skill, Step

        app.skills.save(Skill("tell time", "d", ["tell time"], [Step("get_time_date", {})]))
        assert app.sm.state is State.IDLE
        assert app.brain.run_skill_proactive("tell time") == "It's nine."
        assert app.sm.state is State.IDLE and app._may_speak_up() is True
    finally:
        app.shutdown()


@pytest.mark.parametrize("said,minutes", [("quiet for an hour", 60.0), ("be quiet for thirty minutes", 30.0), ("Hey EV, stop interrupting me for two hours", 120.0),
                                          ("don't bother me for 45 minutes", 45.0), ("snooze notifications for ten minutes", 10.0),
                                          ("quiet for 24 hours", 720.0), ("you can talk again", 0.0), ("stop being quiet", 0.0)])
def test_snooze_by_voice(said, minutes):
    from ev.llm.fastpath import match_fast

    m = match_fast(said)
    assert m is not None and m.tool == "snooze_proactive" and m.args == {"minutes": minutes}, (said, m)


@pytest.mark.parametrize("said", ["be quiet", "quiet please", "set a timer for an hour", "I need quiet for the exam", "keep the volume quiet for an hour"])
def test_other_phrases_do_not_snooze(said):
    from ev.llm.fastpath import match_fast

    m = match_fast(said)
    assert m is None or m.tool != "snooze_proactive", (said, m)
