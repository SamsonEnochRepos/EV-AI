"""Phase 1 tools. Each is small, with a strict schema and a short description."""
from __future__ import annotations

import datetime as dt
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ev.core.config import Config
from ev.memory.store import MemoryStore
from ev.safety.policy import Policy

from . import system, weather, web, windows
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

    def open_app(name: str, admin: bool = False) -> ToolResult:
        try:
            launched = system.launch_app(name, ctx.policy.start_menu, elevate=admin)
        except FileNotFoundError as e:
            return ToolResult(False, str(e))
        except PermissionError as e:
            return ToolResult(False, f"I couldn't start {name} as administrator: {e}.")
        except OSError as e:
            return ToolResult(False, f"I couldn't start {name}: {e.strerror or type(e).__name__}.")
        return ToolResult(True, f"Opened {launched}" + (" as administrator." if admin else "."))

    def open_website(site: str = "", query: str = "", url: str = "", browser: str = "") -> ToolResult:
        target = web.build_url(site, query, url)
        if target is None:
            return ToolResult(False, "That doesn't look like a website I can open.")
        try:
            web.open_url(target, browser)
        except FileNotFoundError as e:
            return ToolResult(False, str(e))
        except OSError as e:
            return ToolResult(False, f"I couldn't open the browser: {e.strerror or type(e).__name__}.")
        label = _SITE_LABELS.get(web.canonical_site(site), web.host_of(target).removeprefix("www."))
        where = f" in {browser}" if browser else ""
        if query.strip() and not url:
            return ToolResult(True, f"Searched {label} for {query.strip()}{where}.")
        return ToolResult(True, f"Opened {label}{where}.")

    def close_app(name: str, force: bool = False) -> ToolResult:
        try:
            wins = windows.list_windows()
            fg = windows.foreground()
        except OSError as e:
            return ToolResult(False, f"I can't close windows here: {e}")
        key = name.strip().lower()
        this = key in windows.THIS_WINDOW
        found = ([fg] if fg else []) if this else windows.match(name, wins)
        label = "that window" if this else name
        own = windows.own_pids()
        ours = [w for w in found if w.pid in own]
        guarded = [w for w in found if w.exe.replace("-", "") in ctx.policy.protected_apps]
        target = [w for w in found if w not in ours and w not in guarded]
        if not target:
            if ours:
                return ToolResult(False, "That's the window I'm running in, so I'm leaving it open.")
            if guarded:
                return ToolResult(False, f"I won't close {label}; Windows or EV depends on it.")
            stems = windows.stems_for(name)
            hidden = [] if this else windows.running_pids(stems - ctx.policy.protected_apps, own)
            if hidden:
                return ToolResult(False, f"{label} is running in the background with no window. "
                                         "I can force close it if you say so.")
            return ToolResult(False, f"I don't see {label} open.")
        if force:
            ended, refused = windows.terminate(sorted({w.pid for w in target}))
            if refused and not ended:
                return ToolResult(False, f"{label} is running as administrator, so I can't force it closed.")
            return ToolResult(True, f"Force closed {label}.")
        refused_wins = [w for w in target if not windows.request_close(w)]
        if len(refused_wins) == len(target):
            return ToolResult(False, f"{label} is running as administrator, so I can't close it without admin rights.")
        left = windows.wait_gone([w for w in target if w not in refused_wins])
        if left:
            return ToolResult(False, f"I asked {label} to close, but it's still open. It may be asking to save something.")
        return ToolResult(True, f"Closed {label}.")

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

    searches: dict[str, tuple[float, list[dict[str, Any]]]] = {}      # query -> (time, rows): repeats skip DuckDuckGo

    def get_weather(place: str = "", day: str = "today") -> ToolResult:
        web = ctx.cfg["web"]
        if not (web.get("enabled", True) and web.get("weather_enabled", True)):
            return ToolResult(False, "Weather lookups are turned off.")
        place = place.strip() or _home_place(ctx.memory)
        try:
            text = weather.get_weather(place, day, web.get("units", "metric"), float(web["timeout_s"]))
        except weather.WeatherError as e:
            return ToolResult(False, str(e))
        return ToolResult(True, text, {"place": place, "day": day})

    def web_search(query: str) -> ToolResult:
        from ddgs import DDGS  # type: ignore[import-untyped]

        web = ctx.cfg["web"]
        if not web.get("enabled", True):
            return ToolResult(False, "Web search is turned off.")
        n = int(web["max_results"])
        key = query.strip().lower()
        hit = searches.get(key)
        if hit and time.monotonic() - hit[0] < float(web.get("cache_s", 600)):
            return _search_result(hit[1])
        # backend pinned: ddgs "auto" fans out to Bing/Brave/Google/Yahoo/..., but only DuckDuckGo is allowed
        rows = _first_rows(
            [lambda d: d.text(query, max_results=n, backend="duckduckgo")] * 3
            + [lambda d: d.news(query, max_results=n, backend="duckduckgo")],
            lambda: DDGS(timeout=web["timeout_s"]), pause_s=float(web.get("retry_pause_s", 1.5)))
        if rows:
            searches[key] = (time.monotonic(), rows)
        return _search_result(rows)

    def _search_result(rows: list[dict[str, Any]] | None) -> ToolResult:
        if rows is None:
            return ToolResult(False, "The search service isn't answering right now. Try again in a moment.")
        if not rows:
            return ToolResult(False, "I didn't find anything.")
        text = "\n".join(f"- {r.get('title', '')}: {r.get('body', '')}".strip() for r in rows)
        return ToolResult(True, text, {"count": len(rows)}, untrusted=True, source="web_search",
                          hint=("Answer the user's question in one or two short spoken sentences using only facts that are written "
                                "in these results. Don't read out site names or links. Never invent numbers, "
                                "conditions or names. If the results only describe what a website offers and don't "
                                "state the answer (typical for live weather), say you couldn't get the live "
                                "details and suggest checking a weather site."))

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
    reg.register(Tool("open_app", "Open an application by name. admin only if asked for administrator rights.",
                      {"name": {"type": S, "description": "App name, e.g. Spotify", "maxLength": 60},
                       "admin": {"type": "boolean"}},
                      open_app, ("name",), timeout_s=20))
    reg.register(Tool("open_website", "Open a website, or search one, in the browser. Set browser only if the user names one.",
                      {"site": {"type": S, "description": "youtube, github, a domain...", "maxLength": 60},
                       "query": {"type": S, "maxLength": 200},
                       "url": {"type": S, "description": "only if the user gave an address", "maxLength": 500},
                       "browser": {"type": S, "maxLength": 20}},
                      open_website, timeout_s=15))
    reg.register(Tool("close_app", "Close an app by name, or 'this' for the front window. force only for force quit or kill.",
                      {"name": {"type": S, "maxLength": 60},
                       "force": {"type": "boolean"}},
                      close_app, ("name",), timeout_s=15))
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
    reg.register(Tool("get_weather", "Get the live weather for a city, today or tomorrow. Leave place empty for the user's home city.", {
        "place": {"type": S, "maxLength": 60},
        "day": {"type": S, "enum": ["today", "tomorrow"]}}, get_weather, timeout_s=20))
    reg.register(Tool("web_search", "Search the web for news, facts and other current information (not weather).",
                      {"query": {"type": S, "maxLength": 200}}, web_search, ("query",), timeout_s=30))
    reg.register(Tool("remember_fact", "Save a fact about the user for later.", {
        "key": {"type": S, "description": "short label, e.g. favorite_color", "maxLength": 60},
        "value": {"type": S, "maxLength": 300}}, remember_fact, ("key", "value")))
    reg.register(Tool("forget_fact", "Delete a saved fact about the user.",
                      {"key": {"type": S, "maxLength": 60}}, forget_fact, ("key",)))
    reg.register(Tool("take_note", "Save a quick note for the user.",
                      {"text": {"type": S, "maxLength": 1000}}, take_note, ("text",)))
    return reg


_SITE_LABELS = {"youtube": "YouTube", "github": "GitHub", "gmail": "Gmail", "linkedin": "LinkedIn", "maps": "Google Maps",
                "stackoverflow": "Stack Overflow", "duckduckgo": "DuckDuckGo", "twitter": "X", "google": "Google",
                "wikipedia": "Wikipedia", "reddit": "Reddit", "amazon": "Amazon"}


def _first_rows(attempts: list[Callable[[Any], Any]], make_client: Callable[[], Any], pause_s: float = 1.0
                ) -> list[dict[str, Any]] | None:
    """Run search attempts in order until one yields usable rows.

    DuckDuckGo intermittently serves a bot-check page, which ddgs reports as "No results found"; a short pause and
    a retry usually clears it. Returns [] if a search ran but found nothing, None if every attempt errored.
    """
    errored = 0
    for k, attempt in enumerate(attempts):
        if k:
            time.sleep(pause_s * k)
        try:
            with make_client() as d:
                rows = [r for r in attempt(d) if r.get("title") or r.get("body")]
        except Exception:
            errored += 1
            continue
        if rows:
            return rows
    return None if errored == len(attempts) else []


_HOME_KEYS = ("location", "city", "home_city", "hometown", "home", "lives_in", "current_city", "place")


def _home_place(memory: MemoryStore) -> str:
    facts = memory.facts(100)
    for k in _HOME_KEYS:
        if facts.get(k):
            return facts[k]
    return ""


def notes_dir(cfg: Config) -> Path:
    p = cfg.root / "data" / "notes"
    p.mkdir(parents=True, exist_ok=True)
    return p


_ = time  # keep import for monkeypatching in tests
