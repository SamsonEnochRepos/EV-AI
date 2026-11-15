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
from ev.core.log import get_logger
from ev.memory.store import MemoryStore
from ev.memory.vault import Vault, VaultError
from ev.memory.vault_index import VaultIndex
from ev.safety.policy import Policy, Verdict
from ev.safety.provenance import SOURCE_SAID, TurnState

from . import system, weather, web, windows
from .base import Tool, ToolRegistry, ToolResult
from .scheduler import Scheduler

log = get_logger("tools")


@dataclass
class ToolContext:
    cfg: Config
    memory: MemoryStore
    scheduler: Scheduler
    policy: Policy
    now: Any = dt.datetime.now  # injectable for tests
    notes_dir: Path | None = None  # where take_note keeps its daily file; None = data/notes (tests and evals point it elsewhere)
    vault: Vault | None = None     # where facts live; None = the SQLite facts table (vault switched off)
    index: VaultIndex | None = None
    turn: TurnState | None = None  # what the system observed this turn; says how a fact write should be labelled


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

    def open_project(name: str, how: str = "folder", env: str = "", agent: str = "") -> ToolResult:
        pol = ctx.policy
        path = system.find_folder(name, pol.project_roots, pol.project_skip)
        if path is None:
            return ToolResult(False, f"I couldn't find a folder called {name.strip()} on your drives.")
        if env and (how != "terminal" or env not in pol.conda_envs):          # policy already checked; the tool doesn't trust it
            return ToolResult(False, "I can only activate an environment from my list, in a terminal.")
        if agent and (how != "terminal" or env or agent not in pol.coding_agents or agent not in system.AGENT_CLIS):
            return ToolResult(False, "I can only start a coding agent from my list, in a terminal on its own.")
        try:
            system.open_folder(how, path, env, agent)
        except FileNotFoundError as e:
            return ToolResult(False, str(e))
        except (OSError, ValueError) as e:
            return ToolResult(False, f"I couldn't open {path.name}: {getattr(e, 'strerror', None) or e}.")
        where = {"vscode": "in VS Code", "terminal": "in a terminal", "folder": "in File Explorer"}[how]
        said = f"Opened {path.name} {where}"
        if agent:
            said += f" and asked it to start {system.AGENT_CLIS[agent][1]}"      # same: it may still be loading or ask to sign in
        if env:
            said += f" and asked it to activate {env}"          # can't be seen from here, so say only what was asked
        return ToolResult(True, said + ".", {"name": path.name, "how": how})

    def open_file(path: str = "", which: str = "") -> ToolResult:
        target = path.strip()
        if not target:
            target, kind = ctx.memory.last_artifact()
            if not target:
                return ToolResult(False, "I haven't made anything yet that I could open.")
        else:
            kind = ""
        # Policy judged the arguments; "last" has no path in them, so judge the file this resolved to as well.
        decision = ctx.policy.check_open_path(target)
        if decision.verdict is Verdict.BLOCK:
            return ToolResult(False, f"I can't open that: {decision.reason}.")
        p = Path(target)
        p = p if p.is_absolute() else ctx.policy.root / p
        if not p.is_file():
            return ToolResult(False, "That file isn't there any more.")
        try:
            system.open_path(str(p))
        except OSError as e:
            return ToolResult(False, f"I couldn't open it: {e.strerror or type(e).__name__}.")
        return ToolResult(True, f"Opened the {kind}." if kind else "Opened the file.", {"path": str(p)})

    def open_website(site: str = "", query: str = "", url: str = "", browser: str = "", play: bool = False) -> ToolResult:
        target = web.build_url(site, query, url)
        if target is None:
            return ToolResult(False, "That doesn't look like a website I can open.")
        key = web.canonical_site(site)
        q = query.strip()
        wants_video = bool(play and q and not url and key == "youtube")
        video = None
        if wants_video and ctx.cfg["web"].get("enabled", True):
            video = web.resolve_first_video(q, float(ctx.cfg["web"].get("timeout_s", 8)))
        where = f" in {browser}" if browser else ""                       # only a browser the user named is spoken
        use, profile = browser, ""
        if not browser and not url and key in _MEDIA_SITES and (media := ctx.cfg["web"].get("media_browser", "")):
            use, profile = str(media), str(ctx.cfg["web"].get("media_profile", ""))   # music and video go to the personal profile
        try:
            web.open_url(video or target, use, profile)
        except FileNotFoundError as e:
            return ToolResult(False, str(e))
        except (OSError, ValueError) as e:
            return ToolResult(False, f"I couldn't open the browser: {getattr(e, 'strerror', None) or type(e).__name__}.")
        label = _SITE_LABELS.get(key, web.host_of(target).removeprefix("www."))
        if video:
            # EV can't see the page, and browsers often block autoplay on a tab you haven't clicked, so it only says it opened it
            return ToolResult(True, f"Opened the top YouTube result for {q}{where}. If it's paused, your browser blocked autoplay, so press play.",
                              {"query": q})
        if wants_video:           # said honestly: nothing was picked, the user is looking at search results
            return ToolResult(True, f"I couldn't pick a video, so I searched YouTube for {q}{where}.", {"query": q})
        if q and not url:
            tail = " Spotify won't start a song from a link, so press play on the one you want." if key == "spotify" else ""
            return ToolResult(True, f"Searched {label} for {q}{where}.{tail}", {"query": q})
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

    def list_reminders() -> ToolResult:
        rows = ctx.memory.pending_reminders()
        if not rows:
            return ToolResult(True, "You have no reminders.")
        parts = [f"{text} at {dt.datetime.fromtimestamp(due).strftime('%I:%M %p').lstrip('0')}" for _id, text, due in rows[:5]]
        n = len(rows)
        return ToolResult(True, f"You have {n} reminder{'s' if n != 1 else ''}: " + "; ".join(parts) + ".")

    def media_control(action: str) -> ToolResult:
        system.press_media_key(action)
        return ToolResult(True, _MEDIA_SAID.get(action, "Done."))

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
        place = place.strip() or _home_place(ctx.vault.profile_facts(100) if ctx.vault is not None else ctx.memory.facts(100))
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
        # Backends are named one at a time, never ddgs "auto" (which fans out to every engine). DuckDuckGo goes first; the
        # engines after it in [web] backends_text / backends_news are tried only if it errors (it serves a bot check to some
        # networks), so a query reaches them only then. An empty list means DuckDuckGo alone.
        text_be = [str(b) for b in web.get("backends_text") or ["duckduckgo"]]
        news_be = [str(b) for b in web.get("backends_news") or ["duckduckgo"]]

        def text_try(be: str, **kw: Any) -> Callable[[Any], Any]:
            return lambda d: d.text(query, max_results=n, backend=be, **kw)

        def news_try(be: str) -> Callable[[Any], Any]:
            return lambda d: d.news(query, max_results=n, backend=be)

        text_search = [text_try(text_be[0]), text_try(text_be[0])] + [_fallback(text_try(b)) for b in text_be[1:]]
        news_search = [news_try(news_be[0])] + [_fallback(news_try(b)) for b in news_be[1:]]
        # Text search ranks evergreen pages (last year's storm, a Wikipedia stub), and the model can't tell they are old.
        # For anything recent, ask for the past week first, then dated news, then the past month.
        recent_search = ([text_try(text_be[0], timelimit="w"), news_try(news_be[0])]
                         + [_fallback(text_try(b, timelimit="w")) for b in text_be[1:]]
                         + [_fallback(news_try(b)) for b in news_be[1:]]
                         + [text_try(text_be[0], timelimit="w"), text_try(text_be[0], timelimit="m")])
        rows = _first_rows(
            recent_search if _RECENT.search(query) else text_search + news_search,
            lambda: DDGS(timeout=web["timeout_s"]), pause_s=float(web.get("retry_pause_s", 1.5)))
        if rows:
            searches[key] = (time.monotonic(), rows)
        return _search_result(rows)

    def _search_result(rows: list[dict[str, Any]] | None) -> ToolResult:
        if rows is None:
            return ToolResult(False, "The search service isn't answering right now. Try again in a moment.")
        if not rows:
            return ToolResult(False, "I didn't find anything.")
        text = "\n".join(
            "- " + " ".join(p for p in (str(r.get("date") or "")[:10], f"{r.get('title', '')}: {r.get('body', '')}".strip()) if p)
            for r in rows)
        return ToolResult(True, text, {"count": len(rows)}, untrusted=True, source="web_search",
                          hint=("Answer the user's question in one or two short spoken sentences using only facts that are written "
                                "in these results. Where results carry dates, trust the newest one. Don't read out site names or links. Never invent numbers, "
                                "conditions or names. If the results only describe what a website offers and don't "
                                "state the answer (typical for live weather), say you couldn't get the live "
                                "details and suggest checking a weather site."))

    def remember_fact(key: str, value: str, about: str = "") -> ToolResult:
        if ctx.vault is None:
            ctx.memory.remember(key, value)
            return ToolResult(True, "Got it, I'll remember that.")
        # The label comes from the system (Brain sets it after the gate has run), never from the model's arguments.
        source = (ctx.turn.call_source if ctx.turn else "") or SOURCE_SAID
        try:
            ctx.vault.add_fact(key, value, source, about)
        except VaultError as e:
            return ToolResult(False, str(e))
        return ToolResult(True, "Got it, I'll remember that.")

    def forget_fact(key: str, about: str = "") -> ToolResult:
        try:
            ok = ctx.vault.forget_fact(key, about) if ctx.vault is not None else ctx.memory.forget(key)
        except VaultError as e:
            return ToolResult(False, str(e))
        return ToolResult(ok, "Forgotten." if ok else "I don't have that saved.")

    def keep_fact(id: str) -> ToolResult:               # noqa: A002  (the argument is called id in the tool schema)
        assert ctx.vault is not None
        try:
            fact = ctx.vault.keep_unconfirmed(id)
        except VaultError as e:
            return ToolResult(False, str(e))
        return ToolResult(True, f"Okay, I'll keep that: {fact.key.replace('_', ' ')}.")

    def recall(query: str = "", when: str = "") -> ToolResult:
        assert ctx.index is not None
        k = int(ctx.cfg.get("vault", "recall_k", 4))
        hits = ctx.index.search(query, when or None, k)
        if not hits:
            return ToolResult(True, "Nothing in the notes about that. Tell the user you don't remember it. Don't guess or "
                                    "make anything up; you can offer to remember it if they tell you.")
        text = "\n".join(f"- {h.path} ({h.date}), {h.heading}: {h.text}" for h in hits)
        return ToolResult(True, text, {"count": len(hits)}, untrusted=True, source="vault",
                          hint=("Answer in one or two short spoken sentences using only what these notes say, and say roughly "
                                "when it was written if that helps. Don't read out file names or paths. If they don't answer "
                                "the question, say you don't remember it. Never add anything the notes don't say."))

    def take_note(text: str) -> ToolResult:
        ctx.memory.add_note(text)
        path = _append_note_file(ctx.notes_dir or notes_dir(ctx.cfg), text, ctx.now())
        if path:
            ctx.memory.set_last_artifact(path, "note")
        return ToolResult(True, "Noted.", {"path": str(path)} if path else {})

    S = "string"
    reg.register(Tool("get_time_date", "Get the current time and date.", {}, get_time_date, direct=True))
    reg.register(Tool("open_app", "Open an application by name. admin only if asked for administrator rights.",
                      {"name": {"type": S, "description": "App name, e.g. Spotify", "maxLength": 60},
                       "admin": {"type": "boolean"}},
                      open_app, ("name",), timeout_s=20, direct=True))
    reg.register(Tool("open_project", "Open a folder from the user's D drive (a project or game) by name, in vscode, a file "
                      "explorer window (folder) or a terminal. env only with terminal. agent starts that coding CLI in a terminal.",
                      {"name": {"type": S, "description": "folder name, e.g. CLM_MVP", "maxLength": 60},
                       "how": {"type": S, "enum": ["vscode", "folder", "terminal"]},
                       "env": {"type": S, "description": "conda environment to activate", "maxLength": 30},
                       "agent": {"type": S, "enum": ["claude", "antigravity"]}},
                      open_project, ("name",), timeout_s=15, direct=True))
    reg.register(Tool("open_file", "Open the last screenshot or note EV saved (which=last), or a file path the user gave.",
                      {"which": {"type": S, "enum": ["last"]}, "path": {"type": S, "maxLength": 300}},
                      open_file, timeout_s=15, direct=True, llm_visible=False))      # voice-routed: costs the model no prompt
    reg.register(Tool("open_website", "Open a website, or search one, in the browser. Set browser only if the user names one. "
                      "play=true with site youtube and a query starts the top video.",
                      {"site": {"type": S, "description": "youtube, spotify, github, a domain...", "maxLength": 60},
                       "query": {"type": S, "maxLength": 200},
                       "url": {"type": S, "description": "only if the user gave an address", "maxLength": 500},
                       "browser": {"type": S, "maxLength": 20},
                       "play": {"type": "boolean"}},
                      open_website, timeout_s=15, direct=True))
    reg.register(Tool("close_app", "Close an app by name, or 'this' for the front window. force only for force quit or kill.",
                      {"name": {"type": S, "maxLength": 60},
                       "force": {"type": "boolean"}},
                      close_app, ("name",), timeout_s=15, direct=True))
    reg.register(Tool("set_timer", "Start a countdown timer.", {
        "minutes": {"type": "number", "minimum": 0, "maximum": 1440},
        "seconds": {"type": "integer", "minimum": 0, "maximum": 86400},
        "label": {"type": S, "maxLength": 40}}, set_timer, direct=True))
    reg.register(Tool("set_reminder", "Remind the user of something later. Use in_minutes or at (like 3:30 pm).", {
        "text": {"type": S, "maxLength": 200},
        "in_minutes": {"type": "number", "minimum": 0, "maximum": 525600},
        "at": {"type": S, "maxLength": 12}}, set_reminder, ("text",), direct=True))
    reg.register(Tool("list_reminders", "List the user's pending reminders.", {}, list_reminders, direct=True,
                      llm_visible=False))
    reg.register(Tool("media_control", "Control media playback.", {
        "action": {"type": S, "enum": ["play_pause", "next", "previous", "stop"]}},
        media_control, ("action",), direct=True))
    reg.register(Tool("set_volume", "Change system volume: set level 0-100, or direction up/down/mute.", {
        "level": {"type": "integer", "minimum": 0, "maximum": 100},
        "direction": {"type": S, "enum": ["up", "down", "mute"]}}, set_volume, direct=True))
    reg.register(Tool("get_weather", "Get the live weather for a city, today or tomorrow. Leave place empty for the user's home city.", {
        "place": {"type": S, "maxLength": 60},
        "day": {"type": S, "enum": ["today", "tomorrow"]}}, get_weather, timeout_s=20, direct=True))
    reg.register(Tool("web_search", "Search the web for news, facts and other current information (not weather).",
                      {"query": {"type": S, "maxLength": 200}}, web_search, ("query",), timeout_s=30))
    about_spec = {"type": S, "description": "another person or project; omit for the user", "maxLength": 40}
    reg.register(Tool("remember_fact", "Save a fact about the user for later.", {
        "key": {"type": S, "description": "short label, e.g. favorite_color", "maxLength": 60},
        "value": {"type": S, "maxLength": 300}, "about": about_spec}, remember_fact, ("key", "value"), direct=True))
    reg.register(Tool("forget_fact", "Delete a saved fact about the user.",
                      {"key": {"type": S, "maxLength": 60}, "about": about_spec}, forget_fact, ("key",), direct=True,
                      llm_visible=False))
    if ctx.index is not None:
        reg.register(Tool("recall", "Search the user's notes for what they said or wrote earlier. when: today, yesterday, last week, a month.",
                          {"query": {"type": S, "maxLength": 120}, "when": {"type": S, "maxLength": 30}}, recall,
                          timeout_s=10, llm_visible=False))
    if ctx.vault is not None:
        reg.register(Tool("keep_fact", "Keep a hand-written note in the facts file as a fact.",
                          {"id": {"type": S, "maxLength": 12}}, keep_fact, ("id",), direct=True, llm_visible=False))
    reg.register(Tool("take_note", "Save a quick note for the user.",
                      {"text": {"type": S, "maxLength": 1000}}, take_note, ("text",), direct=True))
    return reg


_MEDIA_SITES = frozenset({"youtube", "spotify"})     # opened in [web] media_browser / media_profile when the user names no browser
_MEDIA_SAID = {"play_pause": "Okay, toggled playback.", "next": "Skipping ahead.", "previous": "Going back a track.",
               "stop": "Stopped."}
_SITE_LABELS = {"youtube": "YouTube", "github": "GitHub", "gmail": "Gmail", "linkedin": "LinkedIn", "maps": "Google Maps",
                "stackoverflow": "Stack Overflow", "duckduckgo": "DuckDuckGo", "twitter": "X", "google": "Google",
                "wikipedia": "Wikipedia", "reddit": "Reddit", "amazon": "Amazon", "spotify": "Spotify"}


def _fallback(attempt: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Mark an attempt as a different search engine from the one before it: no pause first."""
    attempt.fallback = True  # type: ignore[attr-defined]
    return attempt


def _first_rows(attempts: list[Callable[[Any], Any]], make_client: Callable[[], Any], pause_s: float = 1.0
                ) -> list[dict[str, Any]] | None:
    """Run search attempts in order until one yields usable rows.

    DuckDuckGo intermittently serves a bot-check page, which ddgs reports as "No results found"; a short pause and
    a retry usually clears it. Returns [] if a search ran but found nothing, None if every attempt errored.
    """
    errored = 0
    for k, attempt in enumerate(attempts):
        if k and not getattr(attempt, "fallback", False):
            time.sleep(pause_s)                  # a retry of the same engine waits; a different engine is tried at once
        try:
            with make_client() as d:
                rows = [r for r in attempt(d) if r.get("title") or r.get("body")]
        except Exception as e:                  # noqa: BLE001  (every engine error means "try the next one")
            errored += 1
            log.warning("web search attempt %d of %d failed: %s: %s", k + 1, len(attempts), type(e).__name__, str(e)[:160])
            continue
        if rows:
            return rows
    return None if errored == len(attempts) else []


_RECENT = re.compile(r"\b(latest|recent|recently|today|tonight|yesterday|now|current|currently|news|breaking|this (?:week|month)|"
                     r"happening|just)\b", re.I)
_HOME_KEYS = ("location", "city", "home_city", "hometown", "home", "lives_in", "current_city", "place")


def _home_place(facts: dict[str, str]) -> str:
    for k in _HOME_KEYS:
        if facts.get(k):
            return facts[k]
    return ""


def notes_dir(cfg: Config) -> Path:
    p = cfg.root / "data" / "notes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _append_note_file(folder: Path, text: str, when: dt.datetime) -> Path | None:
    """Add the note to today's file, one line per note. The SQLite copy is the record; a failure here only means "open it" has no file."""
    path = folder / f"{when:%Y-%m-%d}.md"
    line = " ".join(text.split())
    try:
        folder.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            if new:
                f.write(f"# Notes {when:%Y-%m-%d}\n\n")
            f.write(f"- {when:%H:%M} {line}\n")
    except OSError:
        return None
    return path


_ = time  # keep import for monkeypatching in tests
