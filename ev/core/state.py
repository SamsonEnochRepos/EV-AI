"""State machine: IDLE -> LISTENING -> TRANSCRIBING -> THINKING -> (ACTING) -> SPEAKING -> IDLE."""
from __future__ import annotations

import threading

from .events import EventBus, State
from .log import get_logger

log = get_logger("state")

_ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.LISTENING, State.THINKING, State.PAUSED, State.SPEAKING, State.ERROR},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE, State.PAUSED, State.ERROR},
    State.TRANSCRIBING: {State.THINKING, State.IDLE, State.ERROR, State.LISTENING},
    State.THINKING: {State.ACTING, State.SPEAKING, State.IDLE, State.ERROR, State.LISTENING},
    State.ACTING: {State.THINKING, State.SPEAKING, State.IDLE, State.ERROR, State.LISTENING},
    State.SPEAKING: {State.IDLE, State.LISTENING, State.THINKING, State.ACTING, State.ERROR},
    State.ERROR: {State.IDLE, State.PAUSED, State.LISTENING},
    State.PAUSED: {State.IDLE},
}


class StateMachine:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._state = State.IDLE
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        return self._state

    def to(self, new: State, force: bool = False) -> bool:
        with self._lock:
            old = self._state
            if new == old:
                return True
            if not force and new not in _ALLOWED[old]:
                log.warning("illegal transition %s -> %s", old.value, new.value)
                return False
            self._state = new
        log.debug("state %s -> %s", old.value, new.value)
        self.bus.publish("state", old=old, new=new)
        return True
