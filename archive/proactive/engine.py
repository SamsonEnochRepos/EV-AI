"""The proactive engine: EV speaks up, or shows a card, or runs a skill, when something it was told to watch for happens.

It is deliberately timid. Nothing it does skips the safety layer (a skill's steps are gated one by one, and a step that would
need a spoken yes is refused, because EV doesn't start a question on its own), and it holds its tongue when it shouldn't talk:
during quiet hours, full-screen apps and presentations, calls, a turn or confirmation in progress, an empty room, after
`max_spoken_per_hour` lines, and while snoozed ("quiet for an hour"). A line a guard holds back is retried for
`defer_minutes` and then dropped, so EV never delivers stale news. The spoken lines are system text built from a rule's own
template and a few sanitised fields (a process name, an event title), never model output, like the "one sec" fillers (D21).
"""
from __future__ import annotations

import datetime as dt
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ev.context.calendar import CachedCalendar
from ev.core.events import EventBus
from ev.core.log import get_logger
from ev.llm.spoken import spoken_text
from ev.memory.store import MemoryStore
from ev.proactive.probes import Probes
from ev.proactive.rules import FIELDS, Guards, Rule
from ev.safety.policy import clean_text

log = get_logger("proactive")

SNOOZE_KEY = "proactive.snooze_until"
_FIELD = re.compile(r"\{(\w+)\}")
_FIELD_CHARS = 60
_SEEN_CAP = 500


@dataclass
class Actions:
    """What the engine can do. The app supplies these; the engine never reaches past them."""

    say: Callable[[str], None]                       # speak a system line
    notice: Callable[[str], None]                    # show a short card
    skill: Callable[[str], str | None]               # run a skill (gated, never asks); returns what to say, or None


@dataclass
class _Pending:
    rule: Rule
    fields: dict[str, str]
    created: float
    noticed: bool = False                            # the card part is done; the spoken part is still waiting
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


def safe_field(v: Any) -> str:
    """Text from outside (a process name, an event title, a file name) as it may appear in a spoken line or a card."""
    s = re.sub(r"<[^>]*>", " ", clean_text(str(v)))                  # a tag is dropped whole, not left as the word "b"
    s = re.sub(r"\s+", " ", s.replace("<", " ").replace(">", " ")).strip()
    return (spoken_text(s) or s)[:_FIELD_CHARS]


def fill(template: str, fields: dict[str, str]) -> str:
    return re.sub(r"\s+", " ", _FIELD.sub(lambda m: fields.get(m.group(1), ""), template)).strip()


def _in_window(now: dt.time, start: str, end: str) -> bool:
    def at(hhmm: str) -> dt.time:
        h, m = hhmm.split(":")
        return dt.time(int(h), int(m))

    s, e = at(start), at(end)
    return s <= now < e if s <= e else (now >= s or now < e)          # a window like 22:30 to 07:30 wraps midnight


class ProactiveEngine:
    def __init__(self, rules: list[Rule], guards: Guards, probes: Probes, actions: Actions, *,
                 memory: MemoryStore | None = None, calendar: CachedCalendar | None = None, bus: EventBus | None = None,
                 state_ok: Callable[[], bool] = lambda: True, now: Callable[[], dt.datetime] = dt.datetime.now,
                 clock: Callable[[], float] = time.monotonic, epoch: Callable[[], float] = time.time):
        self.rules = [r for r in rules if r.enabled]
        self.guards = guards
        self.probes = probes
        self.actions = actions
        self.memory = memory
        self.calendar = calendar
        self.bus = bus
        self.state_ok = state_ok
        self.now, self.clock, self.epoch = now, clock, epoch
        self._lock = threading.RLock()
        self._pending: list[_Pending] = []
        self._seen: dict[str, deque[str]] = {r.name: deque(maxlen=_SEEN_CAP) for r in self.rules}
        self._last_fired: dict[str, float] = {}
        self._spoken: deque[float] = deque(maxlen=200)       # clock times of spoken lines, for the hourly cap
        self._armed: dict[str, bool] = {}
        self._procs: dict[int, tuple[str, float]] = {}
        self._procs_ready = False
        self._known: dict[str, dict[str, tuple[float, int]]] = {}                  # file rules: files already there or announced
        self._cand: dict[str, dict[str, tuple[float, int]]] = {}                   # ...and files seen once, waiting to settle
        self._stretch: dict[str, tuple[str, float, float, bool]] = {}              # rule -> (exe, since, last seen, fired)
        self._slow_at = -1e9
        self.last_block = ""                                                       # why the last attempt to speak was held back
        self.problems: list[str] = []                                              # rules that were skipped, and why
        self.log: deque[dict[str, Any]] = deque(maxlen=100)                         # recent events, for the dashboard
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snooze_until = float(memory.get_kv(SNOOZE_KEY, "0") or 0) if memory else 0.0

    # ---- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="proactive", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.guards.tick_s):
            try:
                self.tick()
            except Exception:                                     # noqa: BLE001  (a bad probe must never kill the engine)
                log.exception("proactive tick failed")

    # ---- snooze: "quiet for an hour"
    def snooze(self, minutes: float) -> None:
        with self._lock:
            self._snooze_until = self.epoch() + minutes * 60 if minutes > 0 else 0.0
            if self.memory:
                self.memory.set_kv(SNOOZE_KEY, str(self._snooze_until))

    @property
    def snoozed(self) -> bool:
        return self.epoch() < self._snooze_until

    def snooze_minutes_left(self) -> int:
        return max(0, int((self._snooze_until - self.epoch() + 59) // 60))

    # ---- events from elsewhere in EV (a finished Claude Code job, a timer...)
    def fire_event(self, topic: str, **fields: Any) -> int:
        """Rules with source = "event" and this topic fire now. Returns how many matched."""
        n = 0
        with self._lock:
            for rule in self.rules:
                if rule.source == "event" and rule.params["topic"] == topic:
                    self._enqueue(rule, {k: safe_field(v) for k, v in fields.items() if k in FIELDS["event"]})
                    n += 1
        return n

    # ---- the tick
    def tick(self) -> None:
        """Only the engine's own thread calls this. The looking (a process scan can take seconds the first time) happens
        without the lock, so a snooze or an event from elsewhere never waits on it; the lock covers only the hand-over."""
        now = self.now()
        slow = self.clock() - self._slow_at >= self.guards.process_poll_s
        if slow:
            self._slow_at = self.clock()
        due: list[tuple[Rule, dict[str, str]]] = []
        for rule in self.rules:
            try:
                due += [(rule, fields) for fields in self._due(rule, now, slow)]
            except Exception:                                     # noqa: BLE001
                log.warning("rule %r failed", rule.name, exc_info=True)
        with self._lock:
            for rule, fields in due:
                self._enqueue(rule, fields)
            self._flush()

    def _seen_new(self, rule: Rule, key: str) -> bool:
        seen = self._seen[rule.name]
        if key in seen:
            return False
        seen.append(key)
        return True

    def _due(self, rule: Rule, now: dt.datetime, slow: bool) -> list[dict[str, str]]:
        p, src = rule.params, rule.source
        if src == "time":
            at = now.replace(hour=p["at"][0], minute=p["at"][1], second=0, microsecond=0)
            if now.weekday() in p["days"] and at <= now < at + dt.timedelta(minutes=p["window_min"]) \
                    and self._seen_new(rule, f"{now.date()}"):
                return [{"time": f"{at:%I:%M %p}".lstrip("0")}]
        elif src == "calendar" and self.calendar:
            ev = self.calendar.next_event(now)
            if ev and dt.timedelta(0) < ev.start - now <= dt.timedelta(minutes=p["minutes_before"]) \
                    and self._seen_new(rule, f"{ev.start.isoformat()}|{ev.title}"):
                return [{"title": safe_field(ev.title), "minutes": str(max(1, round((ev.start - now).total_seconds() / 60)))}]
        elif src == "battery":
            b = self.probes.battery()
            if b is None:
                return []
            pct, plugged = b
            armed = self._armed.get(rule.name, True)
            if plugged or pct >= p["rearm_above"]:
                self._armed[rule.name] = True
            elif armed and pct < p["below"]:
                self._armed[rule.name] = False
                return [{"percent": str(pct)}]
        elif src == "app_time":
            return self._app_time(rule)
        elif src == "process_exit" and slow:
            return self._process_exit(rule)
        elif src == "file" and slow:
            return self._new_files(rule)
        return []

    def _app_time(self, rule: Rule) -> list[dict[str, str]]:
        exe = self.probes.foreground_exe()
        t = self.clock()
        want = rule.params["exe"]
        cur = self._stretch.get(rule.name)
        if not exe or (want and exe != want):
            return []
        if cur is None or cur[0] != exe or t - cur[2] > 120:            # a new app, or two minutes away from it: a new stretch
            self._stretch[rule.name] = (exe, t, t, False)
            return []
        _exe, since, _last, fired = cur
        self._stretch[rule.name] = (exe, since, t, fired)
        if not fired and t - since >= rule.params["minutes"] * 60:
            self._stretch[rule.name] = (exe, since, t, True)
            return [{"app": safe_field(exe), "minutes": str(int((t - since) // 60))}]
        return []

    def _process_exit(self, rule: Rule) -> list[dict[str, str]]:
        cur = self.probes.processes(rule.params["names"])
        out: list[dict[str, str]] = []
        if self._procs_ready:
            for pid, (name, started) in self._procs.items():
                if pid not in cur and name in rule.params["names"]:
                    runtime = self.epoch() - started
                    if runtime >= rule.params["min_runtime_s"]:
                        out.append({"name": safe_field(name), "minutes": str(max(1, round(runtime / 60)))})
        self._procs = {**{k: v for k, v in self._procs.items() if k in cur}, **cur}
        self._procs_ready = True
        return out

    def _new_files(self, rule: Rule) -> list[dict[str, str]]:
        """A file that appeared or changed and has stopped changing (same size and time on two polls in a row), so a download
        that is still arriving isn't announced half done. What was already there at start-up is the baseline, not news."""
        listing = self.probes.files(rule.params["glob"])
        known = self._known.get(rule.name)
        if known is None:
            self._known[rule.name] = dict(listing)
            return []
        cand = self._cand.setdefault(rule.name, {})
        out: list[dict[str, str]] = []
        for path, sig in listing.items():
            if known.get(path) == sig:
                continue
            if cand.get(path) == sig:
                known[path] = sig
                del cand[path]
                out.append({"name": safe_field(path.replace("\\", "/").rsplit("/", 1)[-1])})
            else:
                cand[path] = sig
        for path in [p for p in cand if p not in listing]:
            del cand[path]
        return out

    # ---- delivering, or holding back
    def _enqueue(self, rule: Rule, fields: dict[str, str]) -> None:
        last = self._last_fired.get(rule.name)
        if rule.cooldown_s and last is not None and self.clock() - last < rule.cooldown_s:
            return
        self._pending.append(_Pending(rule, fields, self.clock()))

    def speech_blocked(self, rule: Rule | None = None) -> str:
        """Why EV should not speak right now, or "" if it may."""
        g = self.guards
        if not self.state_ok():
            return "busy"
        if self.snoozed:
            return "snoozed"
        if g.quiet_hours and _in_window(self.now().time(), *g.quiet_hours):
            return "quiet hours"
        if g.check_fullscreen and self.probes.fullscreen():
            return "full-screen app"
        if g.check_calls and (self.probes.in_call() or self.probes.foreground_exe() in g.call_apps):
            return "call"
        idle = self.probes.idle_s()
        if idle is not None:
            if idle > g.max_idle_s:
                return "nobody there"
            if rule and rule.active_within_s and idle > rule.active_within_s:
                return "not active"
        horizon = self.clock() - 3600
        if sum(1 for t in self._spoken if t >= horizon) >= g.max_spoken_per_hour:
            return "hourly cap"
        return ""

    def notice_blocked(self) -> str:
        if self.guards.check_fullscreen and self.probes.fullscreen():
            return "full-screen app"
        return ""

    def _flush(self) -> None:
        keep: list[_Pending] = []
        for p in self._pending:
            try:
                done = self._deliver(p)
            except Exception:                                     # noqa: BLE001  (a failing action must not repeat every tick)
                log.exception("proactive rule %r failed while delivering; not retrying it", p.rule.name)
                continue
            if done:
                continue
            if self.clock() - p.created > self.guards.defer_minutes * 60:
                self._note(p.rule, "dropped", self.last_block)
                log.info("proactive %r dropped after waiting (%s)", p.rule.name, self.last_block)
                continue
            keep.append(p)
        self._pending = keep

    def _deliver(self, p: _Pending) -> bool:
        """True when the rule is fully done (delivered, or has nothing left to deliver)."""
        r = p.rule
        if r.overlay and not p.noticed:
            if self.notice_blocked():
                self.last_block = self.notice_blocked()
                return False
            self.actions.notice(fill(r.overlay, p.fields))
            p.noticed = True
            self._note(r, "notice", fill(r.overlay, p.fields))
        if r.say or r.skill:
            self.last_block = self.speech_blocked(r)
            if self.last_block:
                return False
            lines: list[str] = []
            if r.say:
                lines.append(fill(r.say, p.fields))
            if r.skill:
                if (said := self.actions.skill(r.skill)):
                    lines.append(said)
            for line in lines:
                if line:
                    self.actions.say(line)
                    self._spoken.append(self.clock())
                    self._note(r, "spoken", line)
        self._last_fired[r.name] = self.clock()
        return True

    def _note(self, rule: Rule, kind: str, text: str) -> None:
        entry = {"rule": rule.name, "kind": kind, "text": text, "at": self.now().isoformat(timespec="seconds")}
        self.log.append(entry)
        if self.bus:
            self.bus.publish("proactive", **entry)
