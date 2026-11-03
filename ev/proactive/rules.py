"""Proactive rules (config/proactive.toml): something EV notices, a condition, and what to do about it.

    [[rules]]
    name = "low battery"
    source = "battery"          # time | calendar | process_exit | file | battery | app_time | event
    below = 20                  # source-specific settings
    say = "Your battery is at {percent} percent."     # speak a line   (system text built from the fields, never model text)
    overlay = "Battery {percent}%"                    # and/or show a short card
    skill = "wind down"                               # and/or run a skill (every step still goes through the safety gate)

A rule that fails any check is reported and skipped: a typo must not turn into a rule that fires every few seconds.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ev.core.log import get_logger

log = get_logger("proactive")

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
SOURCES = ("time", "calendar", "process_exit", "file", "battery", "app_time", "event")
MAX_RULES = 40
MAX_TEMPLATE = 160
_NAME = re.compile(r"[a-z0-9][a-z0-9 '_-]{0,39}", re.I)
_PROC = re.compile(r"[a-z0-9][a-z0-9._+-]{0,39}", re.I)
_HHMM = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")
_FIELD = re.compile(r"\{(\w+)\}")


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    name: str
    source: str
    say: str = ""
    overlay: str = ""
    skill: str = ""
    enabled: bool = True
    active_within_s: int = 0         # only if the user touched keyboard or mouse this recently (0 = don't check)
    cooldown_s: int = 0              # at least this long between two firings of the rule
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Guards:
    quiet_hours: tuple[str, str] | None = ("22:30", "07:30")
    max_spoken_per_hour: int = 4
    max_idle_s: int = 1800               # nobody at the keyboard for this long: don't talk to an empty room
    defer_minutes: float = 10            # a line held back by a guard is dropped after this long
    tick_s: float = 2.0
    process_poll_s: float = 10.0
    call_apps: tuple[str, ...] = ("zoom", "teams", "ms-teams", "webex", "skype", "slack", "discord", "meet")
    check_fullscreen: bool = True
    check_calls: bool = True


def _num(raw: dict[str, Any], key: str, lo: float, hi: float, default: float | None = None) -> float:
    v = raw.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float) or not lo <= v <= hi:
        raise RuleError(f"{key} must be a number from {lo:g} to {hi:g}")
    return float(v)


def parse_rule(raw: dict[str, Any]) -> Rule:
    name = str(raw.get("name", "")).strip()
    if not _NAME.fullmatch(name):
        raise RuleError("a rule needs a short name")
    source = str(raw.get("source", ""))
    if source not in SOURCES:
        raise RuleError(f"source must be one of {', '.join(SOURCES)}")
    say, overlay, skill = (str(raw.get(k, "")).strip() for k in ("say", "overlay", "skill"))
    if not (say or overlay or skill):
        raise RuleError("a rule needs say, overlay or skill")
    if len(say) > MAX_TEMPLATE or len(overlay) > MAX_TEMPLATE or len(skill) > 40:
        raise RuleError("say and overlay are limited to 160 characters")
    p: dict[str, Any] = {}
    if source == "time":
        m = _HHMM.fullmatch(str(raw.get("at", "")))
        if not m:
            raise RuleError("at must be a time like 09:00")
        p["at"] = (int(m.group(1)), int(m.group(2)))
        days = [str(d).lower()[:3] for d in raw.get("days", DAYS)]
        if not days or any(d not in DAYS for d in days):
            raise RuleError("days must be a list like [\"mon\", \"tue\"]")
        p["days"] = tuple(DAYS.index(d) for d in days)
        p["window_min"] = _num(raw, "window_min", 1, 720, 30)
    elif source == "calendar":
        p["minutes_before"] = _num(raw, "minutes_before", 1, 240, 10)
    elif source == "process_exit":
        names = raw.get("names", [])
        if not isinstance(names, list) or not 1 <= len(names) <= 20 or any(not _PROC.fullmatch(str(n)) for n in names):
            raise RuleError("names must be a list of 1 to 20 process names")
        p["names"] = tuple(str(n).lower().removesuffix(".exe") for n in names)
        p["min_runtime_s"] = _num(raw, "min_runtime_s", 0, 86400, 120)
    elif source == "file":
        pattern = str(raw.get("glob", "")).strip()
        if not pattern or len(pattern) > 200 or ".." in pattern or re.fullmatch(r"[A-Za-z]:?[\\/]*\**", pattern):
            raise RuleError("glob must be a folder pattern like ~/Downloads/*.pdf")
        p["glob"] = pattern
    elif source == "battery":
        p["below"] = _num(raw, "below", 1, 99, 20)
        p["rearm_above"] = _num(raw, "rearm_above", 2, 100, p["below"] + 5)
    elif source == "app_time":
        p["exe"] = str(raw.get("exe", "")).lower().removesuffix(".exe")
        p["minutes"] = _num(raw, "minutes", 5, 720, 90)
    elif source == "event":
        topic = str(raw.get("topic", ""))
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,29}", topic):
            raise RuleError("topic must be a short lowercase name")
        p["topic"] = topic
    for text in (say, overlay):
        for f in _FIELD.findall(text):
            if f not in FIELDS[source]:
                raise RuleError(f"{{{f}}} isn't available for a {source} rule (it has: {', '.join(sorted(FIELDS[source]))})")
    return Rule(name, source, say, overlay, skill, bool(raw.get("enabled", True)),
                int(_num(raw, "active_within_s", 0, 86400, 0)), int(_num(raw, "cooldown_s", 0, 86400, 0)), p)


# What each source can put into a template. Anything else is refused when the rule is read.
FIELDS: dict[str, frozenset[str]] = {
    "time": frozenset({"time"}), "calendar": frozenset({"title", "minutes"}), "process_exit": frozenset({"name", "minutes"}),
    "file": frozenset({"name"}), "battery": frozenset({"percent"}), "app_time": frozenset({"app", "minutes"}),
    "event": frozenset({"summary", "count", "label", "branch"}),
}


def parse_guards(raw: dict[str, Any]) -> Guards:
    q = raw.get("quiet_hours", ["22:30", "07:30"])
    quiet: tuple[str, str] | None = None
    if q:
        if not (isinstance(q, list) and len(q) == 2 and all(_HHMM.fullmatch(str(x)) for x in q)):
            raise RuleError("quiet_hours must be two times like [\"22:30\", \"07:30\"], or [] for none")
        quiet = (str(q[0]), str(q[1]))
    d = Guards()
    return Guards(quiet, int(_num(raw, "max_spoken_per_hour", 0, 60, d.max_spoken_per_hour)),
                  int(_num(raw, "max_idle_s", 30, 86400, d.max_idle_s)), _num(raw, "defer_minutes", 0, 240, d.defer_minutes),
                  _num(raw, "tick_s", 0.2, 60, d.tick_s), _num(raw, "process_poll_s", 2, 600, d.process_poll_s),
                  tuple(str(a).lower() for a in raw.get("call_apps", d.call_apps)),
                  bool(raw.get("check_fullscreen", True)), bool(raw.get("check_calls", True)))


def apply_overrides(guards: Guards, section: dict[str, Any]) -> tuple[Guards, list[str]]:
    """The few guard values the dashboard may tune live in [proactive] of ev.toml / ev.local.toml. They replace the rules file's
    values but go through the same range checks; a bad one is reported and ignored."""
    problems: list[str] = []
    changes: dict[str, Any] = {}
    for key, lo, hi, cast in (("max_spoken_per_hour", 0, 60, int), ("max_idle_s", 30, 86400, int), ("defer_minutes", 0, 240, float)):
        if key in section:
            try:
                changes[key] = cast(_num(section, key, lo, hi))
            except RuleError as e:
                problems.append(f"[proactive] {e}")
    if "quiet_start" in section or "quiet_end" in section:
        cur = guards.quiet_hours or ("22:30", "07:30")
        pair = (str(section.get("quiet_start", cur[0])), str(section.get("quiet_end", cur[1])))
        if all(_HHMM.fullmatch(x) for x in pair):
            changes["quiet_hours"] = pair
        else:
            problems.append("[proactive] quiet_start and quiet_end must be times like 22:30")
    return replace(guards, **changes), problems


def load_rules(path: Path) -> tuple[list[Rule], Guards, list[str]]:
    """(rules, guards, problems). A missing file means no rules; a bad rule is skipped and listed in problems."""
    if not path.is_file():
        return [], Guards(), []
    problems: list[str] = []
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return [], Guards(), [f"{path.name}: {e}"]
    try:
        guards = parse_guards(raw.get("guards", {}))
    except RuleError as e:
        problems.append(f"[guards]: {e}; using the defaults")
        guards = Guards()
    rules: list[Rule] = []
    seen: set[str] = set()
    for i, r in enumerate(raw.get("rules", [])[:MAX_RULES], 1):
        try:
            rule = parse_rule(r if isinstance(r, dict) else {})
            if rule.name.lower() in seen:
                raise RuleError("another rule has the same name")
            seen.add(rule.name.lower())
            rules.append(rule)
        except RuleError as e:
            problems.append(f"rule {i} ({r.get('name', '?') if isinstance(r, dict) else '?'}): {e}")
    for p in problems:
        log.warning("proactive: %s", p)
    return rules, guards, problems
