"""Running a skill: its steps in order, each through whatever `step` does (in the app, `Brain._skill_step`, i.e. the same
validation, SafetyGate and provenance as a spoken request). A failed or denied step stops the skill and says which one."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ev.skills.store import Skill, substitute

StepFn = Callable[[str, dict[str, Any]], dict[str, Any]]      # -> a Brain entry: ok, message, raw, untrusted, denied, hallucinated


@dataclass
class Report:
    ok: bool
    message: str
    outside: bool = False                  # some step returned text from outside; the message is data for the model, not a sentence
    done: int = 0
    failed_step: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)


def words(tool: str) -> str:
    return tool.replace("_", " ")


def run(skill: Skill, values: dict[str, str], step: StepFn, available: Callable[[str], bool] = lambda _t: True) -> Report:
    rep = Report(True, "")
    lines: list[str] = []
    for i, s in enumerate(skill.steps, 1):
        if not available(s.tool):
            if s.optional:
                continue
            return _fail(rep, lines, i, s.tool, "that isn't available")
        entry = step(s.tool, substitute(s.args, values))
        rep.entries.append(entry)
        if entry.get("hallucinated"):
            if s.optional:
                continue
            return _fail(rep, lines, i, s.tool, "that isn't available")
        if not entry.get("ok"):
            if s.optional and not entry.get("denied"):
                continue                       # "optional" skips a step that can't be done; a refusal by the gate always stops
            return _fail(rep, lines, i, s.tool, _why(entry))
        rep.done += 1
        outside = bool(entry.get("untrusted"))
        rep.outside = rep.outside or outside
        text = str(entry.get("raw") if outside else entry.get("message", "")).strip()
        if text:
            lines.append(text)
    if skill.say:
        lines.append(skill.say)
    rep.message = ("\n" if rep.outside else " ").join(lines) or f"Done: {skill.name}."
    return rep


def _why(entry: dict[str, Any]) -> str:
    msg = str(entry.get("message", "")).strip()
    return re.sub(r"^I can't do that:\s*", "not allowed: ", msg).rstrip(".") or "it didn't work"


def _fail(rep: Report, done_lines: list[str], i: int, tool: str, why: str) -> Report:
    rep.ok = False
    rep.failed_step = i
    done = f" Steps before it worked: {rep.done}." if rep.done else ""
    rep.message = f"Stopped at step {i}, {words(tool)}: {why}.{done}"
    return rep
