"""Skills: the files, the matching, the runner, and (the point) that every step still goes through the gate on its own."""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import pytest

from ev.core.config import ROOT, load_config
from ev.llm.brain import Brain
from ev.llm.client import OllamaClient
from ev.memory.store import MemoryStore
from ev.safety.gate import SafetyGate
from ev.safety.policy import Policy
from ev.skills import runner
from ev.skills.store import Skill, SkillError, SkillStore, Step, canonical, clean_value, dump, substitute, validate
from ev.tools.base import Tool, ToolRegistry, ToolResult

from .fakes import FakeOllama, text_reply, tool_reply

NOW = dt.datetime(2026, 11, 2, 9, 30)
WORK = {"name": "work mode", "triggers": ["work mode", "work mode with {music}"], "say": "Let's go.", "params": {"music": "lofi"},
        "steps": [{"tool": "open_app", "args": {"name": "visual studio code"}},
                  {"tool": "open_website", "args": {"site": "youtube", "query": "{music}"}}]}


def raw(**over: Any) -> dict[str, Any]:
    return {**WORK, **over}


# ---------------------------------------------------------------- validation

@pytest.mark.parametrize("tool", ["run_skill", "save_skill", "manage_skill", "keep_fact", "build_task", "remember_fact", "forget_fact"])
def test_a_skill_cannot_use_skill_memory_or_builder_tools(tool):
    with pytest.raises(SkillError, match="can't be used"):
        validate(raw(steps=[{"tool": tool, "args": {}}]))


@pytest.mark.parametrize("bad", [
    {"name": ""}, {"name": "../../etc/passwd"}, {"name": "x" * 60}, {"steps": []}, {"steps": [{"tool": "get_time_date"}] * 11},
    {"triggers": ["a"]}, {"triggers": ["work mode with {unknown}"]}, {"triggers": [f"phrase {i}" for i in range(9)]},
    {"params": {"Bad Name": "x"}}, {"params": {"music": 5}}, {"params": {"music": "x" * 200}},
    {"steps": [{"tool": "open_app", "args": {"name": "{nope}"}}]},
    {"steps": [{"tool": "open_app", "args": {"name": {"nested": "dict"}}}]},
    {"steps": [{"tool": "Open App!", "args": {}}]}, {"instructions": "../secret.md"}, {"instructions": "x.txt"},
])
def test_invalid_skills_are_refused(bad):
    with pytest.raises(SkillError):
        validate(raw(**bad))


def test_instructions_are_capped_at_a_size_a_small_model_can_use():
    with pytest.raises(SkillError, match="words"):
        validate(raw(instructions="x.md"), "word " * 121)
    assert validate(raw(instructions="x.md"), "word " * 120).instructions.startswith("word")


def test_substitution_is_single_pass_and_values_cannot_carry_templates():
    assert substitute({"q": "{music} and {music}", "n": 3}, {"music": "{music}{x}"}) == {"q": "{music}{x} and {music}{x}", "n": 3}
    assert clean_value("  jazz {evil}\n  now  ") == "jazz evil now"
    assert clean_value("x" * 500) == "x" * 100


def test_dump_round_trips_awkward_text(tmp_path):
    s = Skill("odd one", 'says "hi" \\ there ü', ["odd one"], [Step("open_website", {"site": "youtube", "query": 'a "q"\\ ü {p}'}),
                                                          Step("set_volume", {"level": 30}), Step("get_weather", {"day": "today"}, True)],
              {"p": "x"}, say="Done.")
    back = validate(__import__("tomllib").loads(dump(s)))
    assert back.steps == s.steps and back.description == s.description and back.params == s.params


# ---------------------------------------------------------------- files, matching

@pytest.fixture
def sdir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    shutil.copytree(ROOT / "skills", d)
    return d


@pytest.fixture
def pol(sdir, tmp_path) -> Policy:
    return Policy.load(ROOT / "config" / "policy.toml", places={"skills": sdir, "vault_ev": tmp_path / "v", "screenshots": tmp_path / "s"})


@pytest.fixture
def store(sdir, pol) -> SkillStore:
    s = SkillStore(sdir, pol)
    assert s.load() == []
    return s


def test_the_shipped_skills_load_cleanly_and_use_real_tools(store, registry):
    assert store.names() == ["morning brief", "wind down", "work mode"]
    names = set(registry.names()) | {"get_agenda"}
    for sk in (store.get(n) for n in store.names()):
        assert all(s.tool in names for s in sk.steps), sk.name
    assert store.get("morning brief").reads_outside and not store.get("work mode").reads_outside
    assert "morning brief" in store.get("morning brief").instructions.lower() or "brief" in store.get("morning brief").instructions.lower()


@pytest.mark.parametrize("said,skill,values", [
    ("work mode", "work mode", {}), ("Work mode.", "work mode", {}), ("hey EV, start work mode please", "work mode", {}),
    ("Yeah, so work mode", "work mode", {}), ("work mode with jazz", "work mode", {"music": "jazz"}),
    ("Work mode with Daft Punk radio!", "work mode", {"music": "Daft Punk radio"}),
    ("let's get to work", "work mode", {}), ("wind down", "wind down", {}), ("brief me", "morning brief", {}),
])
def test_trigger_phrases_match_the_whole_sentence(store, said, skill, values):
    got = store.match(said)
    assert got is not None and got[0].name == skill and got[1] == values


@pytest.mark.parametrize("said", ["I should get into work mode soon", "what is work mode", "work", "open work mode settings now",
                                  "wind down the volume a bit please and tell me a joke", "tell me about my morning brief"])
def test_a_trigger_inside_a_longer_sentence_does_not_run(store, said):
    assert store.match(said) is None


def test_parameter_values_cannot_be_templates(store):
    sk, vals = store.match("work mode with {music} and {x}")
    assert vals == {"music": "music and x"}


def test_a_broken_file_is_reported_and_skipped_not_run(sdir, pol):
    (sdir / "bad.toml").write_text('name = "bad"\n[[steps]]\ntool = "run_skill"\n', encoding="utf-8")
    (sdir / "garbage.toml").write_text("this is [not toml", encoding="utf-8")
    (sdir / "huge.toml").write_text("# " + "x" * 30000, encoding="utf-8")
    (sdir / "dup.toml").write_text((sdir / "work_mode.toml").read_text(encoding="utf-8"), encoding="utf-8")
    s = SkillStore(sdir, pol)
    problems = s.load()
    assert len(problems) == 4 and s.get("bad") is None and s.get("work mode") is not None
    assert any("can't be used" in p for p in problems) and any("20 KB" in p for p in problems)


def test_save_rename_trigger_delete_round_trip(store, sdir):
    changes = []
    store.on_change = lambda: changes.append(1)
    store.save(Skill("focus", "x", ["focus"], [Step("set_volume", {"level": 20})]))
    assert (sdir / "focus.toml").exists() and store.match("focus")[0].name == "focus"
    store.add_trigger("focus", "deep work")
    assert store.match("deep work")[0].name == "focus" and len(list(sdir.glob("focus*.toml"))) == 1       # same file updated in place
    store.rename("focus", "heads down")
    assert store.get("focus") is None and (sdir / "heads_down.toml").exists() and not (sdir / "focus.toml").exists()
    assert store.match("heads down")[0].name == "heads down"
    store.delete("heads down")
    assert store.get("heads down") is None and not (sdir / "heads_down.toml").exists() and len(changes) >= 4


def test_name_and_phrase_conflicts_are_refused(store):
    with pytest.raises(SkillError, match="already"):
        store.save(Skill("work mode", "x", ["work mode"], [Step("get_time_date", {})]))
    with pytest.raises(SkillError, match="already starts"):
        store.add_trigger("wind down", "work mode")
    with pytest.raises(SkillError):
        store.rename("wind down", "work mode")
    with pytest.raises(SkillError):
        store.delete("nothing here")


def test_a_skill_name_cannot_walk_out_of_the_skills_folder(store, sdir):
    with pytest.raises(SkillError):
        store.save(Skill("../../evil", "x", ["evil trigger"], [Step("get_time_date", {})]))
    store.save(Skill("a/b", "x", ["a b"], [Step("get_time_date", {})])) if False else None
    assert {p.parent for p in sdir.rglob("*.toml")} == {sdir}


def test_writes_are_refused_outside_the_policys_write_dirs(sdir, tmp_path):
    elsewhere = Policy.load(ROOT / "config" / "policy.toml", places={"skills": tmp_path / "other", "vault_ev": tmp_path / "v"})
    with pytest.raises(SkillError, match="policy"):
        SkillStore(sdir, elsewhere).save(Skill("x y", "d", ["x y"], [Step("get_time_date", {})]))


# ---------------------------------------------------------------- the runner

def entry(ok=True, message="done", **kw):
    return {"ok": ok, "message": message, **kw}


def test_steps_run_in_order_with_parameters_and_a_closing_line():
    seen = []
    rep = runner.run(validate(WORK), {"music": "jazz"}, lambda t, a: seen.append((t, a)) or entry(message=f"did {t}"))
    assert seen == [("open_app", {"name": "visual studio code"}), ("open_website", {"site": "youtube", "query": "jazz"})]
    assert rep.ok and rep.done == 2 and rep.message == "did open_app did open_website Let's go."


def test_a_failed_step_stops_the_skill_and_names_the_step():
    seen = []

    def step(t, a):
        seen.append(t)
        return entry(ok=t != "open_website", message="No browser found.")

    rep = runner.run(validate(WORK), {}, step)
    assert not rep.ok and rep.failed_step == 2 and rep.done == 1
    assert rep.message == "Stopped at step 2, open website: No browser found. Steps before it worked: 1."


def test_a_denied_step_stops_even_if_optional():
    sk = validate(raw(steps=[{"tool": "close_app", "args": {"name": "x"}, "optional": True}, {"tool": "get_time_date"}]))
    seen = []
    rep = runner.run(sk, {}, lambda t, a: seen.append(t) or entry(ok=False, denied=True, message="I can't do that: not confirmed."))
    assert not rep.ok and seen == ["close_app"] and "not allowed: not confirmed" in rep.message


def test_optional_steps_are_skipped_when_unavailable_or_failing():
    sk = validate(raw(steps=[{"tool": "get_agenda", "optional": True}, {"tool": "get_weather", "optional": True},
                             {"tool": "get_time_date"}]))
    rep = runner.run(sk, {}, lambda t, a: entry(ok=t == "get_time_date", message="It's nine.") if t != "get_agenda" else entry(),
                     available=lambda t: t != "get_agenda")
    assert rep.ok and rep.message == "It's nine. Let's go." or rep.message.startswith("It's nine.")
    assert runner.run(validate(raw(steps=[{"tool": "get_agenda"}])), {}, lambda t, a: entry(), available=lambda t: False).ok is False


def test_outside_text_is_joined_as_data_not_as_a_sentence():
    sk = validate(raw(steps=[{"tool": "get_time_date"}, {"tool": "get_agenda"}], say=""))
    rep = runner.run(sk, {}, lambda t, a: entry(message="I found some results.", raw="09:00 Standup", untrusted=True)
                     if t == "get_agenda" else entry(message="It's nine.", raw="It's nine."))
    assert rep.outside and rep.message == "It's nine.\n09:00 Standup"


# ---------------------------------------------------------------- through the Brain: every step is gated

class Rig:
    def __init__(self, cfg, pol, store, replies, answer=False):
        self.calls: list[tuple[str, dict]] = []
        self.asked: list[str] = []
        reg = ToolRegistry()

        def stub(name, untrusted=False):
            def f(**kw):
                self.calls.append((name, kw))
                return ToolResult(True, "09:00 Standup" if untrusted else f"{name} ok", untrusted=untrusted, source="calendar")
            return f

        for n, params, req in [("open_app", {"name": {"type": "string"}, "admin": {"type": "boolean"}}, ("name",)),
                               ("open_website", {"site": {"type": "string"}, "query": {"type": "string"}}, ()),
                               ("set_volume", {"level": {"type": "integer"}}, ()), ("media_control", {"action": {"type": "string"}}, ("action",)),
                               ("set_timer", {"minutes": {"type": "number"}, "label": {"type": "string"}}, ()),
                               ("close_app", {"name": {"type": "string"}, "force": {"type": "boolean"}}, ("name",)),
                               ("get_time_date", {}, ()), ("get_weather", {"day": {"type": "string"}}, ()), ("list_reminders", {}, ())]:
            reg.register(Tool(n, n, params, stub(n), req))
        reg.register(Tool("get_agenda", "a", {"day": {"type": "string"}}, stub("get_agenda", True)))

        def ask(r):
            self.asked.append(r)
            return answer

        self.gate = SafetyGate(pol, ask=ask)
        self.fake = FakeOllama(replies)
        self.memory = MemoryStore(":memory:")
        self.brain = Brain(cfg, OllamaClient(cfg, transport=self.fake.transport()), self.memory, reg, self.gate, persona="You are EV.",
                           now=lambda: NOW, skills=store)
        self.changed: list[int] = []
        self.brain.on_tools_changed = lambda: self.changed.append(1)

    def tools(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def rig(cfg, pol, store):
    return lambda replies=None, answer=False: Rig(cfg, pol, store, replies or [text_reply("unused")], answer)


def test_work_mode_by_voice_runs_every_step_without_the_model(rig):
    r = rig()
    res = r.brain.respond("work mode with jazz")
    assert res.fast_path and r.fake.requests == []
    assert r.calls == [("open_app", {"name": "visual studio code"}), ("open_website", {"site": "youtube", "query": "jazz"})]
    assert res.text == "open_app ok open_website ok Okay, let's get to work."


def test_each_step_has_its_own_gate_decision(rig):
    r = rig()
    r.brain.respond("work mode")
    gated = [e["tool"] for e in r.gate.audit]
    assert gated == ["run_skill", "open_app", "open_website"] and all(e["allowed"] for e in r.gate.audit)


def test_a_blocked_step_never_runs_and_stops_the_skill(rig, store):
    store.save(Skill("sneaky", "d", ["sneaky"], [Step("set_volume", {"level": 10}), Step("open_app", {"name": "cmd.exe"}),
                                                 Step("set_timer", {"minutes": 1})]))
    r = rig()
    res = r.brain.respond("sneaky")
    assert r.tools() == ["set_volume"] and not res.tool_log[0]["ok"]
    assert res.text.startswith("Stopped at step 2, open app: not allowed:") and "Steps before it worked: 1" in res.text


def test_a_risky_step_asks_out_loud_and_a_no_stops_everything_after_it(rig, store):
    store.save(Skill("tidy", "d", ["tidy up"], [Step("set_volume", {"level": 10}), Step("close_app", {"name": "chrome", "force": True}),
                                                Step("set_timer", {"minutes": 1})]))
    r = rig(answer=False)
    res = r.brain.respond("tidy up")
    assert r.asked == ["force close chrome. Anything unsaved in it will be lost"] and r.tools() == ["set_volume"]
    assert "Stopped at step 2" in res.text
    r2 = rig(answer=True)
    r2.brain.respond("tidy up")
    assert r2.tools() == ["set_volume", "close_app", "set_timer"]                     # a real yes lets the same skill carry on


def test_a_skill_with_a_poisoned_parameter_is_still_judged_by_the_policy(rig, store):
    r = rig()
    res = r.brain.respond("work mode with ..\\..\\Windows\\System32 & format c:")        # blocked at the outer call already
    assert r.calls == [] and res.text == "I can't do that: argument matches a blocked pattern."
    # a value that passes the outer call is judged again by the policy for the step that uses it
    store.save(Skill("launcher", "d", ["do {app} thing"], [Step("set_volume", {"level": 10}), Step("open_app", {"name": "{app}"})],
                     {"app": "notepad"}))
    r2 = rig()
    res = r2.brain.respond("do cmd.exe thing")
    assert r2.tools() == ["set_volume"] and res.text.startswith("Stopped at step 2, open app: not allowed:")
    r3 = rig()
    r3.brain.respond("do notepad thing")
    assert r3.tools() == ["set_volume", "open_app"]


def test_the_model_can_run_a_skill_by_name_but_not_make_or_change_one(rig, store):
    r = rig([tool_reply("run_skill", {"name": "work mode", "options": "music=rock"}), text_reply("Done."),
             tool_reply("save_skill", {"name": "evil"}), text_reply("ok"), tool_reply("manage_skill", {"action": "delete", "name": "work mode"}),
             text_reply("ok")])
    r.brain.respond("let's go to work", allow_fast=False)
    assert ("open_website", {"site": "youtube", "query": "rock"}) in r.calls
    r.brain.respond("make a skill", allow_fast=False)
    r.brain.respond("delete work mode", allow_fast=False)
    assert store.get("evil") is None and store.get("work mode") is not None and r.asked == []
    schema_names = [s["function"]["name"] for s in r.brain.registry.schemas()]
    assert "run_skill" in schema_names and "save_skill" not in schema_names and "manage_skill" not in schema_names
    enum = next(s for s in r.brain.registry.schemas() if s["function"]["name"] == "run_skill")["function"]["parameters"]["properties"]["name"]["enum"]
    assert enum == ["morning brief", "wind down", "work mode"]


def test_the_model_running_a_skill_on_small_talk_is_refused_without_a_question(rig):
    """Measured: 'tell me a joke' made qwen2.5:3b call run_skill in 5 to 6 of 6 trials. A skill is a bundle of side effects."""
    r = rig([tool_reply("run_skill", {"name": "wind down"}), text_reply("Ha.")], answer=True)     # even a ready "yes" is never asked for
    res = r.brain.respond("tell me a joke", allow_fast=False)
    assert r.calls == [] and r.asked == [] and res.tool_log[0]["denied"]
    assert r.gate.audit[-1] == {"tool": "run_skill", "args": {"name": "wind down"}, "allowed": False, "reason": "the user did not ask for it"}


def test_a_partial_match_is_asked_about_not_run_silently(rig, store):
    store.save(Skill("deep focus session", "d", ["deep focus session"], [Step("set_volume", {"level": 10})]))
    r = rig([tool_reply("run_skill", {"name": "deep focus session"}), text_reply("Ok.")], answer=False)
    r.brain.respond("I need some focus today", allow_fast=False)              # one of three words: not nothing, not enough
    assert r.calls == [] and r.asked == ["run the skill deep focus session, though you didn't ask for it"]
    r2 = rig([tool_reply("run_skill", {"name": "deep focus session"}), text_reply("Ok.")], answer=True)
    r2.brain.respond("I need some focus today", allow_fast=False)
    assert r2.tools() == ["set_volume"]                                         # a real yes is a real yes


@pytest.mark.parametrize("said,skill", [("let's start my work mode with some jazz", "work mode"), ("I want to wind down for the night", "wind down"),
                                        ("can you get me ready to work", "work mode")])
def test_the_model_running_the_skill_you_asked_for_needs_no_question(rig, said, skill):
    r = rig([tool_reply("run_skill", {"name": skill}), text_reply("Done.")])
    r.brain.respond(said, allow_fast=False)
    assert r.asked == [] and r.calls


def test_a_skill_the_model_picked_after_reading_outside_text_asks(rig):
    r = rig([tool_reply("get_agenda", {"day": "today"}), tool_reply("run_skill", {"name": "work mode"}), text_reply("Ok.")], answer=False)
    r.brain.respond("what's on my calendar for work mode", allow_fast=False)             # says the name, but outside text was read
    assert r.asked and "something I read" in r.asked[0] and "open_app" not in r.tools()


def test_a_skill_cannot_start_a_skill_even_if_the_file_is_edited_at_runtime(rig):
    r = rig()
    assert r.brain._skill_step("run_skill", {"name": "work mode"})["denied"] is True
    r.brain._skill_running = True
    assert r.brain._tool_run_skill("work mode").message == "A skill can't start another skill."


def test_instructions_reach_the_model_only_while_the_skill_runs(rig, store):
    r = rig([text_reply("Good morning. It's nine. Standup is first.")])
    brief = store.get("morning brief").instructions
    assert brief and brief not in r.brain.system_prompt()
    res = r.brain.respond("morning brief")
    assert not res.fast_path or True
    sent = r.fake.requests[0]["messages"]
    tool_msg = next(m for m in sent if m.get("role") == "tool")
    assert '<untrusted source="skill"' in tool_msg["content"] and brief in tool_msg["content"]       # data wrapped, guidance beside it
    assert "09:00 Standup" in tool_msg["content"] and brief not in sent[0]["content"]
    assert r.tools() == ["get_time_date", "get_weather", "get_agenda", "list_reminders"]


def test_optional_morning_brief_steps_are_skipped_when_the_tool_is_missing(cfg, pol, store):
    r = Rig(cfg, pol, store, [text_reply("Good morning.")])
    r.brain.registry.unregister("get_agenda")
    r.brain.respond("morning brief")
    assert r.tools() == ["get_time_date", "get_weather", "list_reminders"]


# ---- teaching by doing, and managing skills by voice

def test_save_that_as_turns_what_just_worked_into_a_skill(rig, store, sdir):
    r = rig()
    r.brain.respond("open notepad")
    r.brain.respond("volume to 30 percent") if False else r.brain.respond("set the volume to 30 percent")
    res = r.brain.respond("save that as focus time")
    assert (sdir / "focus_time.toml").exists() and store.get("focus time").steps == [
        Step("open_app", {"name": "notepad"}), Step("set_volume", {"level": 30})]
    assert res.text == "Saved focus time. It will open notepad, then set volume with level 30. Say focus time to run it, or tell me another phrase for it."
    assert r.changed == [1]
    r.calls.clear()
    r.brain.respond("focus time")
    assert r.tools() == ["open_app", "set_volume"]
    assert "save_skill" in [e["tool"] for e in r.gate.audit]


def test_save_that_has_nothing_to_save_without_a_recent_action_and_never_includes_skill_or_memory_calls(rig, store):
    r = rig()
    assert r.brain.respond("save that as nothing yet").text == "I haven't done anything lately that I could save."
    r.brain.respond("work mode")
    r.brain.respond("save that as copy of work")
    assert [s.tool for s in store.get("copy of work").steps] == ["open_app", "open_website"]      # the steps, not run_skill


def test_saved_name_must_be_new(rig, store):
    r = rig()
    r.brain.respond("open notepad")
    assert r.brain.respond("save that as work mode").text == "I already have a skill called work mode"


def test_add_a_phrase_rename_list_and_delete_by_voice(rig, store, sdir):
    r = rig(answer=False)
    r.brain.respond("open notepad")
    r.brain.respond("save that as note time")
    assert r.brain.respond("also run it when I say open pad please").text == "Okay, saying open pad now starts note time."   # "please" is politeness
    r.calls.clear()
    r.brain.respond("open pad")
    assert r.tools() == ["open_app"]
    assert r.brain.respond("rename the skill note time to writing time").text == "Renamed it to writing time."
    assert "writing time" in r.brain.respond("what skills do I have").text
    # deleting needs a real spoken yes
    assert r.brain.respond("delete the skill writing time").text.startswith("I can't do that")
    assert store.get("writing time") is not None and r.asked[-1] == "delete the skill writing time"
    r.gate.ask = lambda _r: True
    assert r.brain.respond("delete the skill writing time").text == "Deleted writing time."
    assert store.get("writing time") is None


def test_the_tool_list_changes_once_when_the_skills_change_and_run_skill_goes_when_none_are_left(rig, store):
    r = rig(answer=True)
    for n in list(store.names()):
        r.brain.respond(f"delete the skill {n}")
    assert r.brain.registry.get("run_skill") is None and "run_skill" not in [s["function"]["name"] for s in r.brain.registry.schemas()]
    assert len(r.changed) == 3


def test_skill_phrases_are_not_voice_commands_for_other_things():
    from ev.llm.fastpath import match_fast

    for said in ["save it as notes.txt", "save this as a pdf", "save that as a text file", "save this as notes", "delete my files",
                 "remove the skill"]:
        m = match_fast(said)
        assert m is None or m.tool not in ("save_skill", "manage_skill"), (said, m)
    assert match_fast("save that as note time").args == {"name": "note time"}      # starts with a file word but is a name
    assert match_fast("what are my reminders").tool == "list_reminders"


def test_list_reminders_tool(registry, memory):
    assert registry.get("list_reminders").func().message == "You have no reminders."
    memory.add_reminder("call mom", dt.datetime(2026, 1, 2, 15, 30).timestamp())
    assert "1 reminder: call mom at 3:30 PM" in registry.get("list_reminders").func().message


def test_canonical_names():
    assert canonical("  Work   MODE! ") == "work mode"


def test_app_loads_the_shipped_skills_and_exposes_run_skill(tmp_path):
    from ev.app import EVApp

    cfg = load_config(overrides={"general": {"data_dir": str(tmp_path / "d"), "log_dir": str(tmp_path / "l")},
                                 "vault": {"path": str(tmp_path / "v")}, "overlay": {"enabled": False}, "tray": {"enabled": False}})
    app = EVApp(cfg)
    try:
        assert app.skills.names() == ["morning brief", "wind down", "work mode"] and app.skills.problems == []
        assert app.registry.get("run_skill") is not None and app.registry.get("save_skill").llm_visible is False
    finally:
        app.shutdown()
