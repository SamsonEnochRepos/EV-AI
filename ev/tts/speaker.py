"""Speaker: sentences in -> synthesized and played as early as possible. Supports barge-in."""
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from ev.audio.playback import Player
from ev.core.log import get_logger

from .base import TTSEngine, TTSError

log = get_logger("speaker")


class Speaker:
    def __init__(self, engine: TTSEngine, player: Player,
                 on_error: Callable[[Exception], None] | None = None, pause_ms: int = 0):
        self.engine = engine
        self.player = player
        self.pause_ms = pause_ms
        self.on_error = on_error or (lambda _e: None)
        self._q: queue.Queue[str | None] = queue.Queue()
        self._gen = 0                      # bumped on stop(); stale work is discarded
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()
        self.first_chunk_ms: list[float] = []
        self._t = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._t.start()

    def say(self, sentence: str) -> None:
        s = sentence.strip()
        if not s:
            return
        with self._lock:
            self._pending += 1
            self._idle.clear()
            gen = self._gen
        self._q.put(f"{gen}\x00{s}")

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
        try:
            while True:
                if self._q.get_nowait() is not None:
                    self._finish_one()
        except queue.Empty:
            pass
        self.player.stop()

    def wait_done(self, timeout: float | None = None) -> bool:
        if not self._idle.wait(timeout):
            return False
        return self.player.wait_idle(timeout)

    @property
    def is_speaking(self) -> bool:
        return not self._idle.is_set() or self.player.is_playing

    def _finish_one(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            gen_s, text = item.split("\x00", 1)
            try:
                if int(gen_s) != self._gen:
                    continue
                t0 = time.perf_counter()
                first = True
                for chunk in self.engine.synthesize(text):
                    if int(gen_s) != self._gen:
                        break
                    if first:
                        self.first_chunk_ms.append((time.perf_counter() - t0) * 1000)
                        first = False
                    self.player.play(chunk, self.engine.sample_rate)
                if self.pause_ms and int(gen_s) == self._gen:
                    self.player.play(np.zeros(self.engine.sample_rate * self.pause_ms // 1000, np.float32),
                                     self.engine.sample_rate)
            except TTSError as e:
                log.error("tts failed: %s", e)
                self.on_error(e)
            except Exception as e:
                log.exception("unexpected tts failure")
                self.on_error(e)
            finally:
                self._finish_one()

    def close(self) -> None:
        self.stop()
        self._q.put(None)
        self.player.close()
