"""Voice activity detection. Energy-based (zero deps) with an adaptive noise floor;
optional Silero (bundled with faster-whisper) for better accuracy."""
from __future__ import annotations

import numpy as np


def rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    x = block.astype(np.float32)
    if block.dtype == np.int16:
        x = x / 32768.0
    return float(np.sqrt(np.mean(x * x)))


class EnergyVAD:
    """is_speech(block) -> bool, with an adaptive noise floor."""

    def __init__(self, ratio: float = 3.0, floor_min: float = 0.004, adapt: float = 0.05):
        self.ratio = ratio
        self.floor_min = floor_min
        self.adapt = adapt
        self.floor = floor_min

    def reset(self) -> None:
        self.floor = self.floor_min

    def is_speech(self, block: np.ndarray) -> bool:
        level = rms(block)
        speech = level > max(self.floor * self.ratio, self.floor_min * 2)
        if not speech:  # only learn the floor from non-speech
            self.floor = (1 - self.adapt) * self.floor + self.adapt * max(level, self.floor_min)
        return speech


class SileroVAD:
    """Streaming Silero VAD using the ONNX model bundled with faster-whisper (no extra download).

    faster-whisper's wrapper is stateless/batch-oriented, so this drives its ONNX session directly with a
    persistent LSTM state and 64-sample context, 512 samples (32 ms at 16 kHz) at a time.
    """

    CHUNK = 512
    CONTEXT = 64

    def __init__(self, threshold: float = 0.5):
        from faster_whisper.vad import get_vad_model  # type: ignore[import-untyped]

        self.session = get_vad_model().session
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, self.CONTEXT), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0

    def _prob(self, chunk: np.ndarray) -> float:
        inp = np.concatenate([self._ctx, chunk[None, :]], axis=1)
        out, self._h, self._c = self.session.run(None, {"input": inp, "h": self._h, "c": self._c})
        self._ctx = chunk[None, -self.CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])

    def is_speech(self, block: np.ndarray) -> bool:
        x = block.astype(np.float32)
        if block.dtype == np.int16:
            x = x / 32768.0
        self._buf = np.concatenate([self._buf, x])
        prob = self.last_prob
        while len(self._buf) >= self.CHUNK:
            chunk, self._buf = self._buf[: self.CHUNK], self._buf[self.CHUNK:]
            prob = self._prob(chunk)
        self.last_prob = prob
        return prob >= self.threshold


def make_vad(kind: str = "energy", threshold: float = 0.5):
    if kind == "silero":
        try:
            return SileroVAD(threshold)
        except Exception:  # noqa: S110  (Silero unavailable -> energy VAD)
            pass
    return EnergyVAD()
