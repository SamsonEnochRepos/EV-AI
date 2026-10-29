"""faster-whisper wrapper (CPU, int8). Lazily loaded."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

log = get_logger("stt")


@dataclass
class Transcript:
    text: str
    seconds: float       # wall time spent transcribing
    audio_seconds: float

    @property
    def rtf(self) -> float:
        return self.seconds / self.audio_seconds if self.audio_seconds else 0.0


class STT:
    def __init__(self, cfg: Config, model_name: str | None = None):
        self.c = cfg["stt"]
        self.model_name = model_name or self.c["model"]
        self.sr = cfg["audio"]["sample_rate"]
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            t0 = time.perf_counter()
            try:
                # Offline by default: faster-whisper otherwise pings Hugging Face on every load.
                self._model = WhisperModel(
                    self.model_name, device=self.c["device"], compute_type=self.c["compute_type"],
                    cpu_threads=self.c["cpu_threads"], local_files_only=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Whisper model '{self.model_name}' is not downloaded. Run: python scripts/download_models.py "
                    f"--whisper {self.model_name}") from e
            log.info("whisper %s loaded in %.1fs", self.model_name, time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        self.load()
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x = x / 32768.0
        t0 = time.perf_counter()
        assert self._model is not None
        segs, _ = self._model.transcribe(
            x, language=self.c["language"], beam_size=self.c["beam_size"],
            vad_filter=True, condition_on_previous_text=False,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        return Transcript(text, time.perf_counter() - t0, len(x) / self.sr)

    def unload(self) -> None:
        with self._lock:
            self._model = None
