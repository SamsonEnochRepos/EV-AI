"""Recent events for the live view, and fan-out to browsers watching over Server-Sent Events."""
from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from typing import Any

KEEP = {"turn": 30, "tool": 60, "proactive": 30, "job": 30, "state": 1, "level": 1}
QUEUE_SIZE = 200


class Hub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=30))
        for kind, n in KEEP.items():
            self._recent[kind] = deque(maxlen=n)
        self._subs: list[queue.Queue[dict[str, Any]]] = []

    def publish(self, kind: str, /, **payload: Any) -> None:
        item = {**payload, "kind": kind, "at": round(time.time(), 3)}     # the event's own name and time always win
        with self._lock:
            self._recent[kind].append(item)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(item)
            except queue.Full:                      # a browser that stopped reading must not slow EV down
                pass

    def recent(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {k: list(v) for k, v in self._recent.items()}

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=QUEUE_SIZE)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    @property
    def watchers(self) -> int:
        with self._lock:
            return len(self._subs)
