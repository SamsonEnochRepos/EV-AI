"""Piper TTS engine (CPU, ONNX)."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

from .base import TTSError

log = get_logger("tts.piper")


class PiperEngine:
    name = "piper"

    def __init__(self, cfg: Config):
        t = cfg["tts"]
        self.voice_name = t["piper_voice"]
        self.dir = cfg.path("tts", "piper_voice_dir")
        self.length_scale = float(t.get("length_scale", 1.0))
        self.sample_rate = 22050
        self._voice: Any = None

    def load(self) -> None:
        if self._voice is not None:
            return
        from piper import PiperVoice  # type: ignore[import-untyped]

        model = self.dir / f"{self.voice_name}.onnx"
        if not model.exists():
            raise TTSError(f"Piper voice missing: {model}. Run scripts/download_models.py")
        self._voice = PiperVoice.load(str(model))
        self.sample_rate = int(self._voice.config.sample_rate)

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        self.load()
        v = self._voice
        assert v is not None
        try:
            if hasattr(v, "synthesize_stream_raw"):  # piper-tts 1.2.x
                for raw in v.synthesize_stream_raw(text, length_scale=self.length_scale):
                    yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:  # piper-tts >= 1.3
                from piper import SynthesisConfig  # type: ignore[import-untyped]

                sc = SynthesisConfig(length_scale=self.length_scale)
                for chunk in v.synthesize(text, syn_config=sc):
                    yield chunk.audio_float_array.astype(np.float32)
        except Exception as e:
            raise TTSError(f"piper failed: {e}") from e
