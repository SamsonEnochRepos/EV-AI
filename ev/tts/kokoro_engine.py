"""Kokoro TTS engine (ONNX, CPU) via kokoro-onnx. Optional upgrade over Piper; same interface."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

from .base import TTSError

log = get_logger("tts.kokoro")


class KokoroEngine:
    name = "kokoro"

    def __init__(self, cfg: Config):
        t = cfg["tts"]
        self.voice = t["kokoro_voice"]
        self.dir = cfg.path("tts", "kokoro_dir")
        self.speed = float(t.get("kokoro_speed", 1.0))
        self.sample_rate = 24000
        self._k: Any = None

    def load(self) -> None:
        if self._k is not None:
            return
        model, voices = self.dir / "kokoro-v1.0.onnx", self.dir / "voices-v1.0.bin"
        if not model.exists() or not voices.exists():
            raise TTSError(f"Kokoro files missing in {self.dir}. See README (Kokoro upgrade).")
        try:
            from kokoro_onnx import Kokoro  # type: ignore[import-untyped]
        except ImportError as e:
            raise TTSError("kokoro-onnx is not installed (pip install kokoro-onnx)") from e
        self._k = Kokoro(str(model), str(voices))

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        self.load()
        try:
            samples, sr = self._k.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        except Exception as e:
            raise TTSError(f"kokoro failed: {e}") from e
        self.sample_rate = int(sr)
        yield np.asarray(samples, dtype=np.float32)
