"""TTS interface. Engines are swappable; each yields float32 mono chunks for a sentence."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def load(self) -> None: ...

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 mono audio chunks (range -1..1) for one sentence."""
        ...


class TTSError(RuntimeError):
    pass


def make_engine(cfg) -> TTSEngine:  # noqa: ANN001
    kind = cfg["tts"]["engine"]
    if kind == "kokoro":
        from .kokoro_engine import KokoroEngine

        return KokoroEngine(cfg)
    from .piper_engine import PiperEngine

    return PiperEngine(cfg)
