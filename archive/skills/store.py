"""Skills: named routines made of ordinary tool calls, kept as small TOML files in `skills/`.

    name = "work mode"
    description = "Open the coding setup and start some music"
    triggers = ["work mode", "work mode with {music}"]      # said as a whole sentence; {music} is filled from what was said
    say = "Okay, let's go."                                  # optional: spoken after the last step
    instructions = "morning_brief.md"                        # optional: a short Markdown file, shown to the model only while this skill runs
    [params]
    music = "lofi beats"
    [[steps]]
    tool = "open_website"
    args = { site = "youtube", query = "{music}" }

A skill can never do more than the user could ask for directly: each step is run through the same validation, `SafetyGate` and
provenance checks as a spoken request (see `Brain._skill_step`). This module only reads, checks and writes the files; the
files themselves are user-owned configuration, so a bad one is reported and skipped, never run half-checked.
"""
from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ev.core.log import get_logger
from ev.safety.policy import Policy, clean_text

log = get_logger("skills")

MAX_FILE_BYTES = 20 * 1024
MAX_FILES = 100
MAX_STEPS = 10
MAX_TRIGGERS = 8
MAX_INSTRUCTION_WORDS = 120
MAX_VALUE_CHARS = 100

# A skill cannot call these: they would let a skill start another skill, write a skill, or write to memory behind the user's back.
FORBIDDEN_STEP_TOOLS = frozenset({"run_skill", "save_skill", "manage_skill", "keep_fact", "build_task", "remember_fact",
                                  "forget_fact"})
# Tools whose results are text from outside (web, screen, notes, calendar): a skill that uses one hands the model data, not a sentence.
OUTSIDE_TOOLS = frozenset({"web_search", "read_screen", "screenshot_describe", "get_agenda", "recall"})

_NAME = re.compile(r"[a-z0-9][a-z0-9 '_-]{0,39}")
_PARAM = re.compile(r"[a-z][a-z0-9_]{0,19}")
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]{0,19})\}")
_TOOL = re.compile(r"[a-z][a-z0-9_]{0,39}")


class SkillError(ValueError):
    """Something EV can say out loud about why a skill couldn't be saved, changed or run."""


@dataclass(frozen=True)
class Step:
    tool: str
    args: dict[str, Any]
    optional: bool = False       # skipped (not an error) when the tool isn't available, e.g. get_agenda with no calendar


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: list[str]
    steps: list[Step]
    params: dict[str, str] = field(default_factory=dict)
    say: str = ""
    instructions_file: str = ""
    instructions: str = ""       # the text of that file (loaded and length-checked by the store)

    @property
    def reads_outside(self) -> bool:
        return any(s.tool in OUTSIDE_TOOLS for s in self.steps)


def canonical(name: str) -> str:
    return re.sub(r"\s+", " ", clean_text(name).lower()).strip(" .,!?")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", canonical(name)).strip("_")[:40] or "skill"


def substitute(args: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    """{param} in string arguments, in one pass: a value that itself contains {x} is not expanded again."""
    def fill(v: Any) -> Any:
        return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), v) if isinstance(v, str) else v
    return {k: fill(v) for k, v in args.items()}


def clean_value(v: str) -> str:
    """What the user said for a {param}: one short plain line, no braces, so it can't carry a template of its own."""
    return re.sub(r"\s+", " ", clean_text(v).replace("{", "").replace("}", "")).strip()[:MAX_VALUE_CHARS]


def validate(raw: dict[str, Any], instructions_text: str = "") -> Skill:
    """A Skill from parsed TOML, or SkillError saying what is wrong. Never partial."""
    name = canonical(str(raw.get("name", "")))
    if not _NAME.fullmatch(name):
        raise SkillError("a skill needs a short name of letters, numbers and spaces")
    params = raw.get("params", {})
    if not isinstance(params, dict) or any(not _PARAM.fullmatch(str(k)) or not isinstance(v, str) or len(v) > MAX_VALUE_CHARS
                                           for k, v in params.items()):
        raise SkillError("params must be short lowercase names with short text defaults")
    triggers = [canonical(str(t)) for t in raw.get("triggers", [])] or [name]
    if len(triggers) > MAX_TRIGGERS or any(not 2 <= len(t) <= 60 for t in triggers):
        raise SkillError(f"a skill takes up to {MAX_TRIGGERS} trigger phrases of 2 to 60 characters")
    for t in triggers:
        if any(p not in params for p in _PLACEHOLDER.findall(t)):
            raise SkillError(f"the phrase '{t}' uses a {{parameter}} the skill doesn't declare")
    steps_raw = raw.get("steps", [])
    if not isinstance(steps_raw, list) or not 1 <= len(steps_raw) <= MAX_STEPS:
        raise SkillError(f"a skill needs between 1 and {MAX_STEPS} steps")
    steps: list[Step] = []
    for i, s in enumerate(steps_raw, 1):
        tool, args = str(s.get("tool", "")) if isinstance(s, dict) else "", s.get("args", {}) if isinstance(s, dict) else None
        if not _TOOL.fullmatch(tool) or tool in FORBIDDEN_STEP_TOOLS:
            raise SkillError(f"step {i}: '{tool}' can't be used in a skill")
        if not isinstance(args, dict) or any(isinstance(v, dict | list) for v in args.values()):
            raise SkillError(f"step {i}: arguments must be plain values")
        for v in args.values():
            if isinstance(v, str) and any(p not in params for p in _PLACEHOLDER.findall(v)):
                raise SkillError(f"step {i}: uses a {{parameter}} the skill doesn't declare")
        steps.append(Step(tool, dict(args), bool(s.get("optional", False))))
    say = str(raw.get("say", ""))[:120]
    file = str(raw.get("instructions", ""))
    if file and not re.fullmatch(r"[a-z0-9_-]+\.md", file):
        raise SkillError("instructions must be the name of a .md file in the skills folder")
    if len(instructions_text.split()) > MAX_INSTRUCTION_WORDS:
        raise SkillError(f"instructions are over {MAX_INSTRUCTION_WORDS} words; a small model can't use more")
    return Skill(name, str(raw.get("description", ""))[:120], triggers, steps, {str(k): v for k, v in params.items()}, say,
                 file, instructions_text.strip())


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    return json.dumps(str(v))                    # a JSON string is a valid TOML basic string


def dump(skill: Skill) -> str:
    out = [f"name = {_toml_value(skill.name)}", f"description = {_toml_value(skill.description)}",
           "triggers = [" + ", ".join(_toml_value(t) for t in skill.triggers) + "]"]
    if skill.say:
        out.append(f"say = {_toml_value(skill.say)}")
    if skill.instructions_file:
        out.append(f"instructions = {_toml_value(skill.instructions_file)}")
    if skill.params:
        out += ["", "[params]"] + [f"{k} = {_toml_value(v)}" for k, v in skill.params.items()]
    for s in skill.steps:
        out += ["", "[[steps]]", f"tool = {_toml_value(s.tool)}"]
        if s.optional:
            out.append("optional = true")
        out.append("args = { " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in s.args.items()) + " }" if s.args else "args = {}")
    return "\n".join(out) + "\n"


class SkillStore:
    def __init__(self, directory: Path, policy: Policy | None = None, on_change: Callable[[], None] | None = None):
        self.dir = directory
        self.policy = policy
        self.on_change = on_change or (lambda: None)
        self._skills: dict[str, Skill] = {}
        self._files: dict[str, Path] = {}
        self.problems: list[str] = []
        self.last: str = ""                      # the skill saved or run most recently: what "run it when I say..." refers to
        self._matchers: list[tuple[re.Pattern[str], str, list[str]]] = []

    # ---- reading
    def load(self) -> list[str]:
        """Read every skill file. A file that fails a check is reported in `problems` and skipped."""
        self._skills.clear()
        self._files.clear()
        self.problems = []
        files = sorted(self.dir.glob("*.toml"))[:MAX_FILES] if self.dir.is_dir() else []
        for f in files:
            try:
                if f.stat().st_size > MAX_FILE_BYTES:
                    raise SkillError("the file is over 20 KB")
                raw = tomllib.loads(f.read_text(encoding="utf-8"))
                text = ""
                if raw.get("instructions"):
                    md = self.dir / str(raw["instructions"])
                    text = md.read_text(encoding="utf-8") if re.fullmatch(r"[a-z0-9_-]+\.md", md.name) and md.is_file() else ""
                skill = validate(raw, text)
                if skill.name in self._skills:
                    raise SkillError(f"another file already defines '{skill.name}'")
            except (OSError, tomllib.TOMLDecodeError, SkillError, UnicodeDecodeError) as e:
                self.problems.append(f"{f.name}: {e}")
                log.warning("skill file %s skipped: %s", f.name, e)
                continue
            self._skills[skill.name] = skill
            self._files[skill.name] = f
        self._matchers = self._build_matchers()
        return self.problems

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(canonical(name))

    def __len__(self) -> int:
        return len(self._skills)

    def _build_matchers(self) -> list[tuple[re.Pattern[str], str, list[str]]]:
        out = []
        for skill in self._skills.values():
            for trig in skill.triggers:
                names: list[str] = []
                parts = []
                for piece in re.split(r"(\{[a-z][a-z0-9_]*\})", re.escape(trig).replace(r"\{", "{").replace(r"\}", "}")):
                    if m := _PLACEHOLDER.fullmatch(piece):
                        names.append(m.group(1))
                        parts.append(f"(?P<{m.group(1)}>.+?)")
                    else:
                        parts.append(piece)
                rx = re.compile(r"^\s*(?:(?:yeah|yes|okay|ok|so|well|um|uh|and|now|alright)[, ]+){0,2}(?:(?:hey |ok(?:ay)? )?ev[, ]+)?"
                                r"(?:please |can you |could you |would you )?(?:start |run |activate |begin |do |launch )?(?:the )?"
                                + "".join(parts) + r"(?: skill| routine| mode)?[\s.!?]*(?:please)?[\s.!?]*$", re.I)
                out.append((rx, skill.name, names))
        # longer (more specific) phrases first, so "work mode with jazz" wins over "work mode"
        return sorted(out, key=lambda m: -len(m[0].pattern))

    def match(self, text: str) -> tuple[Skill, dict[str, str]] | None:
        """The skill whose trigger phrase is the whole of what was said, and the {parameter} values from it."""
        t = re.sub(r"\s+", " ", text.strip())
        for rx, name, names in self._matchers:
            if m := rx.match(t):
                return self._skills[name], {n: clean_value(m.group(n)) for n in names}
        return None

    def values_for(self, skill: Skill, said: dict[str, str]) -> dict[str, str]:
        return {**skill.params, **{k: v for k, v in said.items() if k in skill.params and v}}

    # ---- writing (user-owned files; confined to the skills folder)
    def _path(self, name: str) -> Path:
        p = self.dir / f"{slug(name)}.toml"
        base, real = self.dir.resolve(), p.resolve()
        if real.parent != base:
            raise SkillError("that isn't a place I can write a skill")
        if self.policy is not None and not self.policy.path_in_dirs(str(p), self.policy.write_dirs):
            raise SkillError("my policy doesn't let me write there")
        return p

    def save(self, skill: Skill, *, overwrite: bool = False) -> Skill:
        skill = validate(json.loads(json.dumps(_as_raw(skill))), skill.instructions)      # nothing is written that wouldn't load
        path = self._files[skill.name] if overwrite and skill.name in self._files else self._path(skill.name)
        if (skill.name in self._skills or path.exists()) and not overwrite:
            raise SkillError(f"I already have a skill called {skill.name}")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".evtmp")
            tmp.write_text(dump(skill), encoding="utf-8", newline="\n")
            tmp.replace(path)
        except OSError as e:
            raise SkillError("I couldn't write the skill file") from e
        self.load()
        self.last = skill.name
        self.on_change()
        return skill

    def delete(self, name: str) -> None:
        skill = self.get(name)
        if skill is None:
            raise SkillError(f"I don't have a skill called {name}")
        try:
            self._files[skill.name].unlink()
        except OSError as e:
            raise SkillError("I couldn't delete the skill file") from e
        self.load()
        self.on_change()

    def rename(self, old: str, new: str) -> Skill:
        skill = self.get(old)
        if skill is None:
            raise SkillError(f"I don't have a skill called {old}")
        new_name = canonical(new)
        if self.get(new_name):
            raise SkillError(f"I already have a skill called {new_name}")
        triggers = [new_name if t == skill.name else t for t in skill.triggers]
        old_file = self._files[skill.name]
        renamed = self.save(replace(skill, name=new_name, triggers=triggers), overwrite=False)
        old_file.unlink(missing_ok=True)
        self.load()
        self.on_change()
        return renamed

    def add_trigger(self, name: str, phrase: str) -> Skill:
        skill = self.get(name or self.last)
        if skill is None:
            raise SkillError("which skill? say its name")
        phrase = canonical(phrase)
        if phrase in skill.triggers:
            return skill
        owner = next((s for s in self._skills.values() if phrase in s.triggers), None)
        if owner:
            raise SkillError(f"'{phrase}' already starts {owner.name}")
        updated = replace(skill, triggers=[*skill.triggers, phrase])
        return self.save(updated, overwrite=True)


def _as_raw(s: Skill) -> dict[str, Any]:
    return {"name": s.name, "description": s.description, "triggers": s.triggers, "say": s.say, "params": s.params,
            "instructions": s.instructions_file,
            "steps": [{"tool": x.tool, "args": x.args, "optional": x.optional} for x in s.steps]}
