"""Phase 1 tools. Each is small, with a strict schema and a short description."""
from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ev.core.config import Config
from ev.memory.store import MemoryStore
from ev.safety.policy import Policy

from . import system
from .base import Tool, ToolRegistry, ToolResult
from .scheduler import Scheduler


@dataclass
class ToolContext:
    cfg: Config
    memory: MemoryStore
    scheduler: Scheduler
    policy: Policy
    now: Any = dt.datetime.now  # injectable for tests


def _spoken_duration(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s and not h:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " and ".join(parts) or "0 seconds"


def build_tools(ctx: ToolContext, reg: ToolRegistry | None = None) -> ToolRegistry:
    reg = reg or ToolRegistry()

    def get_time_date() -> ToolResult:
        n = ctx.now()
        return ToolResult(True, n.strftime("It's %I:%M %p on %A, %B %d, %Y").replace(" 0", " "))

    def open_app(name: str) -> ToolResult:
        try:
            launched = system.launch_app(name, ctx.policy.start_menu)
        except FileNotFoundError as e:
            return ToolResult(False, str(e))
        return ToolResult(True, f"Opened {launched}.")

    def set_timer(minutes: float = 0, seconds: int = 0, label: str = "") -> ToolResult:
        total = int(round(minutes * 60 + seconds))
        if total <= 0:
            return ToolResult(False, "I need a duration for the timer.")
        if total > 24 * 3600:
            return ToolResult(False, "That's longer than a day; try a reminder instead.")
        ctx.scheduler.add_timer(total, label.strip())
        return ToolResult(True, f"Timer set for {_spoken_duration(total)}.", {"seconds": total})

    def set_reminder(text: str, in_minutes: float = 0, at: str = "") -> ToolResult:
        now = ctx.now()
        if at:
            m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*([ap]m)?\s*", at, re.I)
            if not m:
                return ToolResult(False, "I couldn't read that time. Try something like 3:30 pm.")
            h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
            if ap == "pm" and h < 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            if not (0 <= h < 24 and 0 <= mi < 60):
                return ToolResult(False, "That time doesn't look right.")
            due_dt = now.replace(hour=h, minute=mi, second=0, microsecond=0)
            if due_dt <= now:
                due_dt += dt.timedelta(days=1)
            due = due_dt.timestamp()
        elif in_minutes > 0:
            due = now.timestamp() + in_minutes * 60
        else:
            return ToolResult(False, "When should I remind you?")
        ctx.memory.add_reminder(text.strip(), due)
        when = dt.datetime.fromtimestamp(due).strftime("%I:%M %p").lstrip("0")
        return ToolResult(True, f"Okay, I'll remind you at {when}.")

    def media_control(action: str) -> ToolResult:
        system.press_media_key(action)
        return ToolResult(True, f"Done: {action.replace('_', ' ')}.")

    def set_volume(level: int = -1, direction: str = "") -> ToolResult:
        if direction in ("up", "down"):
            system.press_media_key("volume_up" if direction == "up" else "volume_down", 5)
            return ToolResult(True, f"Volume {direction}.")
        if direction == "mute":
            system.press_media_key("mute")
            return ToolResult(True, "Toggled mute.")
        if 0 <= level <= 100:
            system.set_master_volume(level)
            return ToolResult(True, f"Volume set to {level} percent.")
        return ToolResult(False, "Tell me a level from zero to one hundred.")

    def web_search(query: str) -> ToolResult:
        from ddgs import DDGS  # type: ignore[import-untyped]

        web = ctx.cfg["web"]
        if not web.get("enabled", True):
            return ToolResult(False, "Web search is turned off.")
        with DDGS(timeout=web["timeout_s"]) as d:
            # backend pinned: ddgs "auto" fans out to Bing/Brave/Google/Yahoo/..., but only DuckDuckGo is allowed
            rows = list(d.text(query, max_results=web["max_results"], backend="duckduckgo"))
        if not rows:
            return ToolResult(False, "I didn't find anything.")
        text = "\n".join(f"- {r.get('title','')}: {r.get('body','')}" for r in rows)
        return ToolResult(True, text, {"count": len(rows)}, untrusted=True, source="web_search")

    def remember_fact(key: str, value: str) -> ToolResult:
        ctx.memory.remember(key, value)
        return ToolResult(True, "Got it, I'll remember that.")

    def forget_fact(key: str) -> ToolResult:
        ok = ctx.memory.forget(key)
        return ToolResult(ok, "Forgotten." if ok else "I don't have that saved.")

    def take_note(text: str) -> ToolResult:
        ctx.memory.add_note(text)
        return ToolResult(True, "Noted.")

    S = "string"
    reg.register(Tool("get_time_date", "Get the current time and date.", {}, get_time_date))
    reg.register(Tool("open_app", "Open an application by name.",
                      {"name": {"type": S, "description": "App name, e.g. Spotify", "maxLength": 60}},
                      open_app, ("name",)))
    reg.register(Tool("set_timer", "Start a countdown timer.", {
        "minutes": {"type": "number", "minimum": 0, "maximum": 1440},
        "seconds": {"type": "integer", "minimum": 0, "maximum": 86400},
        "label": {"type": S, "maxLength": 40}}, set_timer))
    reg.register(Tool("set_reminder", "Remind the user of something later. Use in_minutes or at (like 3:30 pm).", {
        "text": {"type": S, "maxLength": 200},
        "in_minutes": {"type": "number", "minimum": 0, "maximum": 525600},
        "at": {"type": S, "maxLength": 12}}, set_reminder, ("text",)))
    reg.register(Tool("media_control", "Control media playback.", {
        "action": {"type": S, "enum": ["play_pause", "next", "previous", "stop"]}},
        media_control, ("action",)))
    reg.register(Tool("set_volume", "Change system volume: set level 0-100, or direction up/down/mute.", {
        "level": {"type": "integer", "minimum": 0, "maximum": 100},
        "direction": {"type": S, "enum": ["up", "down", "mute"]}}, set_volume))
    reg.register(Tool("web_search", "Search the web for current information.",
                      {"query": {"type": S, "maxLength": 200}}, web_search, ("query",), timeout_s=15))
    reg.register(Tool("remember_fact", "Save a fact about the user for later.", {
        "key": {"type": S, "description": "short label, e.g. favorite_color", "maxLength": 60},
        "value": {"type": S, "maxLength": 300}}, remember_fact, ("key", "value")))
    reg.register(Tool("forget_fact", "Delete a saved fact about the user.",
                      {"key": {"type": S, "maxLength": 60}}, forget_fact, ("key",)))
    reg.register(Tool("take_note", "Save a quick note for the user.",
                      {"text": {"type": S, "maxLength": 1000}}, take_note, ("text",)))
    return reg


def notes_dir(cfg: Config) -> Path:
    p = cfg.root / "data" / "notes"
    p.mkdir(parents=True, exist_ok=True)
    return p


_ = time  # keep import for monkeypatching in tests
