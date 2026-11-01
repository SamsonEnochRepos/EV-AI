"""faster-whisper wrapper (CPU, int8). Lazily loaded."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

log = get_logger("stt")

_TARGET_PEAK = 0.9
_MAX_GAIN = 12.0       # don't blow up a near-silent clip into hiss


@dataclass
class Transcript:
    text: str
    seconds: float       # wall time spent transcribing
    audio_seconds: float

    @property
    def rtf(self) -> float:
        return self.seconds / self.audio_seconds if self.audio_seconds else 0.0


# Names Whisper splits or mishears in real use ("Open Fire Fox", "open you due"). Kept deliberately small and literal.
_NAME_FIXES = [
    (re.compile(r"\bfire[ -]?fox\b", re.I), "Firefox"),
    (re.compile(r"\b(?:you|u)[ -]?tube\b", re.I), "YouTube"),
    (re.compile(r"\b(open|launch|go to|on|and|to) (?:you due|you dude|u tube)\b", re.I), r"\1 YouTube"),
    (re.compile(r"\bg[ -]?mail\b", re.I), "Gmail"),
    (re.compile(r"\bgit[ -]hub\b", re.I), "GitHub"),
    (re.compile(r"\bdis?play mode\b", re.I), "display mode"),
]


def fix_names(text: str) -> str:
    for rx, rep in _NAME_FIXES:
        text = rx.sub(rep, text)
    return text


def clean_audio(x: np.ndarray, sr: int) -> np.ndarray:
    """Remove DC and low rumble, then bring quiet speech up to a consistent level. Whisper degrades on quiet mics."""
    if x.size == 0:
        return x
    x = x - float(np.mean(x))
    try:
        from scipy.signal import butter, sosfilt

        x = sosfilt(butter(2, 80, btype="highpass", fs=sr, output="sos"), x).astype(np.float32)
    except ImportError:
        pass
    peak = float(np.max(np.abs(x)))
    if peak > 1e-4:
        x = np.clip(x * min(_TARGET_PEAK / peak, _MAX_GAIN), -1.0, 1.0)
    return x.astype(np.float32)


class STT:
    def __init__(self, cfg: Config, model_name: str | None = None):
        self.c = cfg["stt"]
        self.model_name = model_name or self.c["model"]
        self.sr = cfg["audio"]["sample_rate"]
        self.hotwords = " ".join(self.c.get("hotwords", [])) or None
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
        audio_s = len(x) / self.sr
        x = clean_audio(x, self.sr)
        t0 = time.perf_counter()
        assert self._model is not None
        segs, _ = self._model.transcribe(
            x, language=self.c["language"], beam_size=self.c["beam_size"],
            vad_filter=True, condition_on_previous_text=False, without_timestamps=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 250},
            temperature=[0.0, 0.2, 0.4],            # retry hotter only when the greedy pass looks like garbage
            no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
            hotwords=self.hotwords,
        )
        text = fix_names(" ".join(s.text.strip() for s in segs).strip())
        return Transcript(text, time.perf_counter() - t0, audio_s)

    def warm(self) -> None:
        """Run one throwaway decode so the first real utterance doesn't pay the one-time start-up cost."""
        self.load()
        assert self._model is not None
        noise = (np.random.default_rng(0).standard_normal(self.sr) * 0.01).astype(np.float32)
        segs, _ = self._model.transcribe(noise, language=self.c["language"], beam_size=1, vad_filter=False,
                                         without_timestamps=True)
        for _ in segs:
            pass

    def unload(self) -> None:
        with self._lock:
            self._model = None
