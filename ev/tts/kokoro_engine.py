"""Kokoro TTS engine (ONNX, CPU) via kokoro-onnx. Optional upgrade over Piper; same interface."""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

from .base import TTSError

log = get_logger("tts.kokoro")


def parse_voice(spec: str) -> list[tuple[str, float]]:
    """"af_heart" -> [("af_heart", 1.0)]; "af_heart:70,af_bella:30" -> two voices whose weights add up to 1 (a blend).
    Raises ValueError for anything else, so a typo in ev.local.toml is reported rather than guessed at."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[tuple[str, float]] = []
    for p in parts:
        name, _, weight = p.partition(":")
        name = name.strip()
        if not re.fullmatch(r"[a-z]{2}_[a-z]+", name):
            raise ValueError(f"not a Kokoro voice name: {name!r}")
        try:
            w = float(weight) if weight.strip() else 1.0
        except ValueError as e:
            raise ValueError(f"bad weight for {name}: {weight!r}") from e
        if w <= 0:
            raise ValueError(f"weight for {name} must be above zero")
        out.append((name, w))
    if not out or len(out) > 4:
        raise ValueError("give one to four Kokoro voices")
    total = sum(w for _, w in out)
    return [(n, w / total) for n, w in out]


def voice_lang(name: str) -> str:
    """Kokoro voices starting with b (bf_*, bm_*) are British English; the phonemes have to match or words come out wrong."""
    return "en-gb" if name.startswith("b") else "en-us"


class KokoroEngine:
    name = "kokoro"

    def __init__(self, cfg: Config):
        t = cfg["tts"]
        self.voice = t["kokoro_voice"]
        self.mix = parse_voice(str(self.voice))
        self.lang = voice_lang(self.mix[0][0])
        self.dir = cfg.path("tts", "kokoro_dir")
        self.speed = float(t.get("kokoro_speed", 1.0))
        self.sample_rate = 24000
        self._k: Any = None
        self._style: Any = None

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
        if len(self.mix) == 1:
            self._style = self.mix[0][0]                          # a plain voice name
        else:                                                      # a blend: the weighted sum of the voices' style vectors
            try:
                self._style = sum(w * self._k.get_voice_style(n) for n, w in self.mix)
            except Exception as e:
                raise TTSError(f"kokoro voice blend {self.voice!r} failed: {e}") from e
        try:
            self._k.create("Hi.", voice=self._style, speed=self.speed, lang=self.lang)   # first call is ~3x slower
        except Exception as e:
            log.warning("kokoro warm-up failed: %s", e)

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        self.load()
        try:
            samples, sr = self._k.create(text, voice=self._style, speed=self.speed, lang=self.lang)
        except Exception as e:
            raise TTSError(f"kokoro failed: {e}") from e
        self.sample_rate = int(sr)
        yield np.asarray(samples, dtype=np.float32)
