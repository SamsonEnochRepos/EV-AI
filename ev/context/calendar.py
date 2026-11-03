"""Read-only calendar sources for the context snapshot and `get_agenda`: an .ics file, or Outlook through COM. Never writes.

Nothing here runs on the request path. `CachedCalendar` refreshes in the background every few minutes and the snapshot only
reads the cache, so a slow or missing Outlook can't delay a reply.
"""
from __future__ import annotations

import datetime as dt
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ev.core.log import get_logger

log = get_logger("calendar")

MAX_ICS_BYTES = 5 * 1024 * 1024
MAX_EVENTS = 50
TITLE_CHARS = 80


@dataclass(frozen=True)
class Event:
    start: dt.datetime      # naive local time
    end: dt.datetime
    title: str
    all_day: bool = False


class CalendarSource(Protocol):
    def events_between(self, start: dt.datetime, end: dt.datetime) -> list[Event]: ...


def _clean_title(s: str) -> str:
    s = s.replace("\\n", " ").replace("\\N", " ").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:TITLE_CHARS] or "(no title)"


def _parse_when(params: str, value: str) -> tuple[dt.datetime, bool] | None:
    value = value.strip()
    try:
        if "VALUE=DATE" in params and "VALUE=DATE-TIME" not in params or len(value) == 8:
            return dt.datetime.strptime(value[:8], "%Y%m%d"), True
        naive = dt.datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
        if value.endswith("Z"):
            return naive.replace(tzinfo=dt.UTC).astimezone().replace(tzinfo=None), False
        return naive, False                  # a TZID or a floating time: taken as local time
    except ValueError:
        return None


_UNFOLD = re.compile(r"\r?\n[ \t]")


def parse_ics(text: str) -> list[Event]:
    """Single events only. Recurring events (RRULE) show up once, on their first date; expanding them is not supported."""
    events: list[Event] = []
    cur: dict[str, tuple[str, str]] | None = None
    for ln in _UNFOLD.sub("", text).splitlines():
        if ln == "BEGIN:VEVENT":
            cur = {}
        elif ln == "END:VEVENT":
            if cur is not None and (ev := _make_event(cur)):
                events.append(ev)
            cur = None
        elif cur is not None and ":" in ln:
            head, _, value = ln.partition(":")
            name = head.split(";")[0].upper()
            if name in ("DTSTART", "DTEND", "SUMMARY", "STATUS"):
                cur[name] = (head.upper(), value)
    return sorted(events, key=lambda e: e.start)


def _make_event(p: dict[str, tuple[str, str]]) -> Event | None:
    if p.get("STATUS", ("", ""))[1].strip().upper() == "CANCELLED" or "DTSTART" not in p:
        return None
    start = _parse_when(*p["DTSTART"])
    if start is None:
        return None
    end = _parse_when(*p["DTEND"]) if "DTEND" in p else None
    all_day = start[1]
    end_dt = end[0] if end else start[0] + (dt.timedelta(days=1) if all_day else dt.timedelta(hours=1))
    return Event(start[0], max(end_dt, start[0]), _clean_title(p.get("SUMMARY", ("", ""))[1]), all_day)


class IcsCalendar:
    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()
        self._stamp: tuple[int, int] | None = None
        self._events: list[Event] = []

    def events_between(self, start: dt.datetime, end: dt.datetime) -> list[Event]:
        try:
            st = self.path.stat()
            if st.st_size > MAX_ICS_BYTES:
                log.warning("calendar file %s is over %d MB; ignoring it", self.path.name, MAX_ICS_BYTES // 2**20)
                return []
            if self._stamp != (st.st_mtime_ns, st.st_size):
                self._events = parse_ics(self.path.read_text(encoding="utf-8", errors="replace"))
                self._stamp = (st.st_mtime_ns, st.st_size)
        except OSError as e:
            log.warning("calendar file not readable: %s", e)
            return []
        return [e for e in self._events if e.end > start and e.start < end][:MAX_EVENTS]


class OutlookCalendar:
    """Read-only through COM (needs classic Outlook, installed and signed in). Written against the documented object model and
    tested with fakes only: Outlook is not installed on the development machine."""

    def events_between(self, start: dt.datetime, end: dt.datetime) -> list[Event]:
        try:
            import pythoncom
            import win32com.client
        except ImportError:
            log.warning("pywin32 is not installed; the Outlook calendar is unavailable")
            return []
        pythoncom.CoInitialize()                      # this runs on the refresh thread, which needs its own COM apartment
        try:
            folder = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI").GetDefaultFolder(9)   # olFolderCalendar
            items = folder.Items
            items.IncludeRecurrences = True
            items.Sort("[Start]")
            fmt = "%m/%d/%Y %I:%M %p"
            found = items.Restrict(f"[Start] < '{end.strftime(fmt)}' AND [End] > '{start.strftime(fmt)}'")
            out: list[Event] = []
            for it in found:
                s, e = it.Start, it.End
                out.append(Event(dt.datetime(s.year, s.month, s.day, s.hour, s.minute),
                                 dt.datetime(e.year, e.month, e.day, e.hour, e.minute),
                                 _clean_title(str(it.Subject)), bool(it.AllDayEvent)))
                if len(out) >= MAX_EVENTS:
                    break
            return out
        except Exception as e:                        # noqa: BLE001  (Outlook not running, not configured, or a COM error)
            log.warning("Outlook calendar unavailable: %s", e)
            return []
        finally:
            pythoncom.CoUninitialize()


class CachedCalendar:
    """The only thing the rest of EV talks to. Refreshes in the background; reads never touch the source."""

    def __init__(self, source: CalendarSource, refresh_s: float = 300, horizon_days: int = 2,
                 now: Callable[[], dt.datetime] = dt.datetime.now):
        self.source = source
        self.refresh_s = refresh_s
        self.horizon = dt.timedelta(days=horizon_days)
        self.now = now
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh(self) -> None:
        today = self.now().replace(hour=0, minute=0, second=0, microsecond=0)
        events = self.source.events_between(today, today + self.horizon)
        with self._lock:
            self._events = events

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="calendar", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception:                          # noqa: BLE001
                log.exception("calendar refresh failed")
            if self._stop.wait(self.refresh_s):
                return

    def on_day(self, day: dt.date) -> list[Event]:
        with self._lock:
            return [e for e in self._events if e.start.date() <= day <= (e.end - dt.timedelta(seconds=1)).date()]

    def next_event(self, now: dt.datetime | None = None) -> Event | None:
        """The first timed event that hasn't ended yet. All-day events are not 'next'; they are listed by on_day."""
        now = now or self.now()
        with self._lock:
            return next((e for e in self._events if not e.all_day and e.end > now), None)


def make_calendar(source: str, path: str, refresh_s: float) -> CachedCalendar | None:
    """None when the calendar is off (the default) or its file isn't configured."""
    if source == "ics" and path:
        return CachedCalendar(IcsCalendar(path), refresh_s)
    if source == "outlook":
        return CachedCalendar(OutlookCalendar(), refresh_s)
    return None
