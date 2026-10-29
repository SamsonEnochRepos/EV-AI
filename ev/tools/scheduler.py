"""Timers and persisted reminders. Fires a callback (which speaks the alert)."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

from ev.core.log import get_logger
from ev.memory.store import MemoryStore

log = get_logger("scheduler")


class Scheduler:
    def __init__(self, memory: MemoryStore, on_fire: Callable[[str], None], tick_s: float = 1.0):
        self.memory = memory
        self.on_fire = on_fire
        self.tick_s = tick_s
        self._timers: dict[int, tuple[float, str]] = {}
        self._next = 1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)

    def add_timer(self, seconds: float, label: str = "") -> int:
        with self._lock:
            tid = self._next
            self._next += 1
            self._timers[tid] = (time.monotonic() + seconds, label)
        return tid

    def active_timers(self) -> list[tuple[int, float, str]]:
        now = time.monotonic()
        with self._lock:
            return [(i, max(0.0, d - now), lab) for i, (d, lab) in self._timers.items()]

    def cancel_all_timers(self) -> int:
        with self._lock:
            n = len(self._timers)
            self._timers.clear()
        return n

    def poll_once(self) -> None:
        now = time.monotonic()
        with self._lock:
            done = [(i, lab) for i, (d, lab) in self._timers.items() if d <= now]
            for i, _ in done:
                del self._timers[i]
        for _, label in done:
            self._fire(f"Your {label} timer is up." if label else "Your timer is up.")
        for rid, text in self.memory.due_reminders():
            self.memory.mark_fired(rid)
            self._fire(f"Reminder: {text}")

    def _fire(self, msg: str) -> None:
        try:
            self.on_fire(msg)
        except Exception:
            log.exception("alert callback failed")

    def _loop(self) -> None:
        while not self._stop.wait(self.tick_s):
            try:
                self.poll_once()
            except Exception:
                log.exception("scheduler tick failed")
