"""Where did this memory write come from? Escalation only: a fact the user didn't plainly say gets a spoken yes first.

`Policy.evaluate` depends on config and arguments only. Whether a write to memory is the user's own words or something a
web page, a screen or a note talked the model into depends on the turn, so it is decided here from facts the *system*
observed (the fast path matched the transcript; an untrusted tool result arrived this turn; how much the value overlaps
what the user actually said). The result can only turn an ALLOW into a CONFIRM in `SafetyGate.check`; it can never relax
a BLOCK or a CONFIRM, and nothing the model, the screen, the web or the vault says can lower it.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

# Tools whose writes to memory need to be the user's own words.
MEMORY_TOOLS = frozenset({"remember_fact", "forget_fact"})

SOURCE_FAST = "fast_path"
SOURCE_SKILL = "skill"        # a step of a skill: gated like any call, and it is not "the user's own words" for memory purposes

# What a fact's `source` says (kept in the vault line for people and in SQLite for EV). Set by the system, never the model.
SOURCE_SAID = "user said"
SOURCE_SAID_CONFIRMED = "user said, confirmed"
SOURCE_KEPT = "user confirmed"

REASON_UNTRUSTED = "that came from something I read, not from you"
REASON_NOT_SAID = "you didn't say that"
REASON_NOT_ASKED = "you didn't ask for it"

_STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been am it its this that these
those i me my mine we our you your he she they them his her their there here so do does did have has had will would can could
should just very really also then than too not no yes ok okay please about into over up down out""".split())
_WORD = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> set[str]:
    """Lower-case words that carry meaning: stop-words and one-letter leftovers ("dog's" -> "dog", "s") are dropped, numbers stay."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and (len(w) > 1 or w.isdigit())}


def overlap(text: str, utterances: list[str] | deque[str]) -> float:
    """Share of the content words in `text` that the user actually said. 0.0 when `text` has none to compare."""
    want = content_tokens(text)
    if not want:
        return 0.0
    said: set[str] = set()
    for u in utterances:
        said |= content_tokens(u)
    return len(want & said) / len(want)


@dataclass
class TurnState:
    """What the system has seen this turn. Owned by `Brain`; the memory tools read `call_source` to label what they save."""

    utterances: deque[str] = field(default_factory=lambda: deque(maxlen=3))   # the user's own transcripts, never tool or model text
    untrusted_seen: bool = False       # a web / screen / vault / file result (or recalled note) reached the model this turn
    call_source: str = ""              # label for the call about to run; set by Brain right before a gated tool executes

    def begin_turn(self, user_text: str) -> None:
        self.utterances.append(user_text)
        self.untrusted_seen = False
        self.call_source = ""


def skill_unasked(phrases: list[str], source: str, turn: TurnState) -> bool:
    """A model-made run_skill with no word in common with anything the user said: almost certainly a mistake, so it is refused
    outright rather than turned into a question about something the user never mentioned."""
    if source in (SOURCE_FAST, SOURCE_SKILL):
        return False
    return max((overlap(p, turn.utterances) for p in phrases), default=0.0) == 0.0


def skill_escalation(phrases: list[str], source: str, turn: TurnState, min_overlap: float = 0.5) -> str | None:
    """A model-made run_skill: silent only if the user said (most of) the skill's name or one of its trigger phrases and nothing
    from outside was read this turn. A 3B model reaches for a tool with a list of names on small talk ("tell me a joke" ran
    'wind down' in 5 of 6 trials), and a skill is a bundle of side effects. The user's own phrase (fast path) needs no check."""
    if source == SOURCE_FAST or source == SOURCE_SKILL:
        return None
    if turn.untrusted_seen:
        return REASON_UNTRUSTED
    if max((overlap(p, turn.utterances) for p in phrases), default=0.0) < min_overlap:
        return REASON_NOT_ASKED
    return None


def escalation(tool: str, args: dict[str, object], source: str, turn: TurnState, min_overlap: float) -> str | None:
    """A short reason to ask the user out loud first, or None when the write may go ahead."""
    if tool not in MEMORY_TOOLS:
        return None
    if source == SOURCE_FAST:                        # matched straight from the transcript of what the user said
        return None
    if turn.untrusted_seen:
        return REASON_UNTRUSTED
    text = str(args.get("value" if tool == "remember_fact" else "key", ""))
    if overlap(text, turn.utterances) < min_overlap:
        return REASON_NOT_SAID
    return None
