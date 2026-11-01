"""Microphone capture with VAD endpointing. The mic stream is open only while recording."""
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

from .vad import rms

log = get_logger("capture")


class MicError(RuntimeError):
    pass


def mic_blocks(cfg: Config, stop: Callable[[], bool]) -> Iterator[np.ndarray]:
    """Yield int16 mono blocks from the real microphone until stop() or the generator closes."""
    import sounddevice as sd  # type: ignore[import-untyped]

    a = cfg["audio"]
    sr, block = a["sample_rate"], int(a["sample_rate"] * a["block_ms"] / 1000)
    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    def cb(indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    device = a.get("input_device") or None
    try:
        stream = sd.InputStream(samplerate=sr, channels=1, dtype="int16", blocksize=block,
                                device=device, callback=cb)
        stream.start()
    except Exception as e:
        raise MicError(f"can't open microphone: {e}") from e
    try:
        while not stop():
            try:
                yield q.get(timeout=0.5)
            except queue.Empty:
                continue
    finally:
        stream.stop()
        stream.close()


class SharedMic:
    """One microphone stream kept open for a whole conversation, fanned out to whoever is listening.

    The per-turn `mic_blocks` closes and reopens the device between turns (about 100-200 ms) and the barge-in monitor
    and the recorder each open their own stream, so anything said at the hand-off is lost. Here the stream stays open;
    `blocks()` subscribes a listener (the monitor while EV speaks, the recorder while it listens) and unsubscribes when
    that listener stops. Blocks are only delivered to current subscribers, nothing is buffered or stored.
    """

    def __init__(self, cfg: Config):
        a = cfg["audio"]
        self.sr = int(a["sample_rate"])
        self.block = int(self.sr * a["block_ms"] / 1000)
        self.device = a.get("input_device") or None
        self._subs: list[queue.Queue[np.ndarray]] = []
        self._lock = threading.Lock()
        self._stream: Any = None

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def open(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd  # type: ignore[import-untyped]

        def cb(indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
            self._publish(indata[:, 0].copy())

        try:
            stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="int16", blocksize=self.block,
                                    device=self.device, callback=cb)
            stream.start()
        except Exception as e:
            raise MicError(f"can't open microphone: {e}") from e
        self._stream = stream

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            self._subs.clear()

    def _publish(self, blk: np.ndarray) -> None:
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(blk)
                except queue.Full:
                    pass                              # a listener that stopped reading just misses blocks

    def blocks(self, stop: Callable[[], bool]) -> MicListener:
        """Blocks from this moment on (int16 mono) until stop() or close(). Same shape as mic_blocks().

        Subscribes immediately rather than at the first read, so nothing between creating the listener and reading it is lost."""
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        with self._lock:
            self._subs.append(q)
        return MicListener(self, q, stop)

    def _unsubscribe(self, q: queue.Queue[np.ndarray]) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


class MicListener:
    """One listener's view of a SharedMic. Iterate it for blocks; close() (or dropping it) unsubscribes."""

    def __init__(self, mic: SharedMic, q: queue.Queue[np.ndarray], stop: Callable[[], bool]):
        self._mic, self._q, self._stop = mic, q, stop

    def __iter__(self) -> MicListener:
        return self

    def __next__(self) -> np.ndarray:
        while not self._stop():
            try:
                return self._q.get(timeout=0.5)
            except queue.Empty:
                continue
        self.close()
        raise StopIteration

    def close(self) -> None:
        self._mic._unsubscribe(self._q)

    def __del__(self) -> None:
        self.close()


class Recorder:
    def __init__(self, cfg: Config, vad: Any, on_level: Callable[[float], None] | None = None):
        a = cfg["audio"]
        self.sr = a["sample_rate"]
        self.block_ms = a["block_ms"]
        self.silence_blocks = max(1, int(a["silence_ms_to_end"] / self.block_ms))
        self.max_blocks = int(a["max_utterance_s"] * 1000 / self.block_ms)
        self.min_blocks = int(a["min_utterance_ms"] / self.block_ms)
        self.pre_roll = int(a.get("pre_roll_ms", 300) / self.block_ms)   # VAD fires late; keep the clipped first syllable
        self.vad = vad
        self.on_level = on_level or (lambda _l: None)
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def record(self, blocks: Iterator[np.ndarray], wait_for_speech_s: float = 6.0) -> np.ndarray | None:
        """Consume blocks until end of speech. Returns int16 audio, or None if no speech heard."""
        self.cancelled = False
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        pre: list[np.ndarray] = []
        voiced: list[np.ndarray] = []
        started = False
        speech_blocks = 0
        silent = 0
        max_wait = int(wait_for_speech_s * 1000 / self.block_ms)
        n = 0
        for blk in blocks:
            n += 1
            if self.cancelled:
                return None
            self.on_level(min(1.0, rms(blk) * 8))
            speech = self.vad.is_speech(blk)
            if not started:
                pre.append(blk)
                pre = pre[-self.pre_roll:]
                if speech:
                    started = True
                    voiced = list(pre)
                    speech_blocks = 1
                    silent = 0
                elif n > max_wait:
                    return None
                continue
            voiced.append(blk)
            speech_blocks += 1 if speech else 0
            silent = 0 if speech else silent + 1
            if silent >= self.silence_blocks or len(voiced) >= self.max_blocks:
                break
        if not started or speech_blocks < self.min_blocks:
            return None
        return np.concatenate(voiced)
