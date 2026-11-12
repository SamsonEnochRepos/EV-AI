"""Microphone capture with VAD endpointing. The mic stream is open only while recording."""
from __future__ import annotations

import queue
import threading
from collections import deque
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
    that listener stops. Blocks are delivered to current subscribers; the last couple of seconds are also kept in memory
    (never written anywhere) so a listener can start from a given moment (`blocks(since=...)`): the recorder that takes over
    from the wake word or from the barge-in monitor then hears everything said during the hand-off.
    """

    RING_S = 3.0

    def __init__(self, cfg: Config):
        a = cfg["audio"]
        self.sr = int(a["sample_rate"])
        self.block = int(self.sr * a["block_ms"] / 1000)
        self.device = a.get("input_device") or None
        self._subs: list[queue.Queue[tuple[int, np.ndarray]]] = []
        self._lock = threading.Lock()
        self._stream: Any = None
        self._seq = 0                                                  # number of blocks published so far
        self._ring: deque[tuple[int, np.ndarray]] = deque(maxlen=max(1, int(self.RING_S * 1000 / a["block_ms"])))

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
            item = (self._seq, blk)
            self._seq += 1
            self._ring.append(item)
            for q in self._subs:
                try:
                    q.put_nowait(item)
                except queue.Full:
                    pass                              # a listener that stopped reading just misses blocks

    def mark(self, back: int = 0) -> int:
        """A position in the stream: the number of the next block to arrive, minus `back` (for `blocks(since=...)`)."""
        with self._lock:
            return max(0, self._seq - back)

    def blocks(self, stop: Callable[[], bool], since: int | None = None) -> MicListener:
        """Blocks from this moment on (int16 mono) until stop() or close(). Same shape as mic_blocks().

        Subscribes immediately rather than at the first read, so nothing between creating the listener and reading it is lost.
        With `since` (a value from mark() or a listener's resume_from) the blocks kept since then come first, then the live ones,
        with no gap in between."""
        q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=max(200, self._ring.maxlen or 0))
        with self._lock:
            if since is not None:
                for item in self._ring:
                    if item[0] >= since:
                        q.put_nowait(item)
            self._subs.append(q)
        return MicListener(self, q, stop)

    def _unsubscribe(self, q: queue.Queue[tuple[int, np.ndarray]]) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


class MicListener:
    """One listener's view of a SharedMic. Iterate it for blocks; close() (or dropping it) unsubscribes."""

    def __init__(self, mic: SharedMic, q: queue.Queue[tuple[int, np.ndarray]], stop: Callable[[], bool]):
        self._mic, self._q, self._stop = mic, q, stop
        self.last_seq = -1                    # number of the last block handed out: the next listener can resume at last_seq + 1

    @property
    def resume_from(self) -> int:
        return self.last_seq + 1

    def __iter__(self) -> MicListener:
        return self

    def __next__(self) -> np.ndarray:
        while not self._stop():
            try:
                self.last_seq, blk = self._q.get(timeout=0.5)
                return blk
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
        self.last: dict[str, float | str] = {}       # how the last recording went (for the log): seconds, longest pause, why it ended

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
        longest_pause = 0
        ended = "no speech"
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
            if speech:
                longest_pause = max(longest_pause, silent)       # the pause that just ended, inside the utterance
            silent = 0 if speech else silent + 1
            if silent >= self.silence_blocks:
                ended = f"{self.silence_blocks * self.block_ms} ms of silence"
                break
            if len(voiced) >= self.max_blocks:
                ended = "the length limit"
                break
        self.last = {"seconds": len(voiced) * self.block_ms / 1000, "longest_pause_ms": longest_pause * self.block_ms, "ended": ended}
        if not started or speech_blocks < self.min_blocks:
            return None
        return np.concatenate(voiced)
