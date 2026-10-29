"""Microphone capture with VAD endpointing. The mic stream is open only while recording."""
from __future__ import annotations

import queue
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


class Recorder:
    def __init__(self, cfg: Config, vad: Any, on_level: Callable[[float], None] | None = None):
        a = cfg["audio"]
        self.sr = a["sample_rate"]
        self.block_ms = a["block_ms"]
        self.silence_blocks = max(1, int(a["silence_ms_to_end"] / self.block_ms))
        self.max_blocks = int(a["max_utterance_s"] * 1000 / self.block_ms)
        self.min_blocks = int(a["min_utterance_ms"] / self.block_ms)
        self.pre_roll = int(300 / self.block_ms)
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
