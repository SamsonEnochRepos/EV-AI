"""Speaker playback with output-level metering and instant stop (barge-in)."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from ev.core.log import get_logger

log = get_logger("playback")


class Player:
    """Plays queued float32 chunks. Level callback gets 0..1 RMS per output block."""

    def __init__(self, on_level: Callable[[float], None] | None = None, device: str | None = None,
                 sink: Callable[[np.ndarray, int], None] | None = None):
        self.on_level = on_level or (lambda _l: None)
        self.device = device or None
        self.sink = sink  # test hook: bypass the sound card
        self._q: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
        self._stop_evt = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._pending = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="player", daemon=True)
        self._thread.start()
        self.error: Exception | None = None

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        with self._lock:
            self._pending += 1
            self._idle.clear()
        self._q.put((audio, sample_rate))

    def stop(self) -> None:
        """Immediately silence output and drop everything queued."""
        self._stop_evt.set()
        try:
            while True:
                item = self._q.get_nowait()
                if item is not None:
                    self._done()
        except queue.Empty:
            pass
        self.on_level(0.0)

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    @property
    def is_playing(self) -> bool:
        return not self._idle.is_set()

    def close(self) -> None:
        self.stop()
        self._q.put(None)

    def _done(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()

    def _run(self) -> None:
        stream: Any = None
        cur_sr = 0
        while True:
            item = self._q.get()
            if item is None:
                break
            audio, sr = item
            try:
                if self._stop_evt.is_set():
                    self._stop_evt.clear()
                if self.sink is not None:
                    self._meter_and_sink(audio, sr)
                else:
                    if stream is None or sr != cur_sr:
                        if stream is not None:
                            stream.close()
                        import sounddevice as sd  # type: ignore[import-untyped]

                        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                                 device=self.device)
                        stream.start()
                        cur_sr = sr
                    block = max(256, sr // 50)
                    for i in range(0, len(audio), block):
                        if self._stop_evt.is_set():
                            break
                        seg = audio[i:i + block].astype(np.float32)
                        self.on_level(min(1.0, float(np.sqrt(np.mean(seg * seg))) * 5))
                        stream.write(seg.reshape(-1, 1))
            except Exception as e:  # missing speaker, device unplugged...
                self.error = e
                log.error("playback failed: %s", e)
                stream = None
            finally:
                self._done()
                if self._q.empty():
                    self.on_level(0.0)

    def _meter_and_sink(self, audio: np.ndarray, sr: int) -> None:
        block = max(256, sr // 50)
        for i in range(0, len(audio), block):
            if self._stop_evt.is_set():
                break
            seg = audio[i:i + block]
            self.on_level(min(1.0, float(np.sqrt(np.mean(seg * seg))) * 5))
            assert self.sink is not None
            self.sink(seg, sr)
