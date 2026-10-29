"""Tiny synchronous-publish, thread-safe event bus plus the state enum."""
from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any

from .log import get_logger

log = get_logger("events")


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    ACTING = "acting"
    SPEAKING = "speaking"
    ERROR = "error"
    PAUSED = "paused"


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[..., None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, fn: Callable[..., None]) -> None:
        with self._lock:
            self._subs[topic].append(fn)

    def publish(self, topic: str, **payload: Any) -> None:
        with self._lock:
            subs = list(self._subs.get(topic, ()))
        for fn in subs:
            try:
                fn(**payload)
            except Exception:
                log.exception("subscriber failed for %s", topic)
