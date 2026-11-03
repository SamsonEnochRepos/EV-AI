"""The settings the dashboard may change: an explicit allow-list of tuning knobs, each with a type and a range.

The rule is structural, not a filter: a key that is not listed here does not exist as far as the dashboard is concerned, so
anything that could *widen* what EV can do or reach stays file-only. That includes the builder, the vault and every path,
web and trusted-host settings, privacy and retention, hotkeys, the calendar and the repos to watch, the model server address,
how the microphone is triggered, the rules of the proactive engine, and every value in policy.toml. Only things that tune
how EV feels (timings, thresholds, reply length, what the context block shows, how often proactive lines may come) are here.
`tests/test_dashboard.py` walks every key in ev.toml and policy.toml and fails if a widening-looking one is ever added.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class SettingError(ValueError):
    pass


@dataclass(frozen=True)
class Setting:
    key: str                       # "section.name", as in ev.toml
    kind: str                      # bool | int | float | time
    label: str
    lo: float = 0
    hi: float = 0
    live: bool = False             # applied at once; otherwise "restart needed"
    help: str = ""

    @property
    def section(self) -> str:
        return self.key.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.key.split(".", 1)[1]


SETTINGS: tuple[Setting, ...] = (
    # conversation feel
    Setting("conversation.followup_enabled", "bool", "Keep listening after I answer", live=True),
    Setting("conversation.followup_window_s", "float", "Follow-up window (s)", 1, 30, True, "How long she waits for you to keep talking."),
    Setting("conversation.followup_question_window_s", "float", "Window after a question (s)", 1, 30, True),
    Setting("conversation.followup_patience_s", "float", "Extra patience per turn (s)", 0, 10, True),
    Setting("conversation.followup_window_max_s", "float", "Longest window (s)", 5, 60, True),
    Setting("conversation.followup_max_turns", "int", "Back-and-forth turns per wake-up", 1, 20, True),
    Setting("conversation.turn_gap_ms", "int", "Pause before the mic reopens (ms)", 0, 2000, True),
    Setting("conversation.voice_interrupt", "bool", "Talk over her to stop her", live=True),
    Setting("conversation.interrupt_min_ms", "int", "Talk-over time needed to stop her (ms)", 100, 1500, True),
    Setting("conversation.interrupt_margin", "float", "Talk-over loudness margin", 1.0, 4.0, True,
            "Raise it if she cuts herself off on loud speakers."),
    Setting("conversation.interrupt_floor", "float", "Talk-over minimum level", 0.005, 0.1, True),
    Setting("trigger.wakeword_threshold", "float", "Wake word sensitivity threshold", 0.3, 0.95, False,
            "Higher = fewer false wake-ups, more misses."),
    # replies
    Setting("llm.reply_max_sentences", "int", "Reply length (sentences)", 1, 6, True),
    Setting("llm.reply_max_sentences_detail", "int", "Reply length when you ask for detail", 2, 12, True),
    Setting("llm.temperature", "float", "Model temperature", 0.0, 1.5, True),
    Setting("llm.filler_after_s", "float", "Say \"one sec\" after (s, 0 = never)", 0, 5, True),
    Setting("tts.sentence_pause_ms", "int", "Pause between sentences (ms)", 0, 500, True),
    # the pill
    Setting("overlay.display_mode", "bool", "Display mode (text card)", live=True),
    Setting("overlay.display_linger_s", "float", "Card stays for (s)", 2, 30, False),
    Setting("overlay.display_font_pt", "int", "Card text size", 8, 24, False),
    # what the context block shows (switches only)
    Setting("context.enabled", "bool", "Give her a look at what is on my computer", live=True),
    Setting("context.foreground", "bool", "...the app and window I am in", live=True),
    Setting("context.recent_files", "bool", "...files that just appeared", live=True),
    Setting("context.battery", "bool", "...battery", live=True),
    Setting("context.git", "bool", "...git branch", live=True),
    # how much she may speak up on her own (the rules themselves stay in config/proactive.toml)
    Setting("proactive.max_spoken_per_hour", "int", "Most spoken interruptions per hour", 0, 20, True),
    Setting("proactive.quiet_start", "time", "Quiet hours start", live=True),
    Setting("proactive.quiet_end", "time", "Quiet hours end", live=True),
    Setting("proactive.max_idle_s", "int", "Stay quiet if I've been away (s)", 60, 86400, True),
    Setting("proactive.defer_minutes", "float", "Drop a held-back line after (min)", 0, 120, True),
)

ALLOWED: dict[str, Setting] = {s.key: s for s in SETTINGS}
_HHMM = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


def coerce(key: str, raw: Any) -> bool | int | float | str:
    """The value for an allowed key, checked against its type and range, or SettingError. An unlisted key is refused."""
    s = ALLOWED.get(key)
    if s is None:
        raise SettingError(f"{key} can't be changed from the dashboard")
    if s.kind == "bool":
        if not isinstance(raw, bool):
            raise SettingError(f"{key} must be true or false")
        return raw
    if s.kind == "time":
        if not isinstance(raw, str) or not _HHMM.fullmatch(raw):
            raise SettingError(f"{key} must be a time like 22:30")
        return raw
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise SettingError(f"{key} must be a number")
    if raw != raw or not s.lo <= raw <= s.hi:                      # NaN fails both
        raise SettingError(f"{key} must be between {s.lo:g} and {s.hi:g}")
    return int(raw) if s.kind == "int" else float(raw)
