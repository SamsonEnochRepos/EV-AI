"""faster-whisper wrapper: the GPU when it works, otherwise the CPU (int8). Lazily loaded."""
from __future__ import annotations

import importlib.util
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    avg_logprob: float = 0.0       # Whisper's own confidence: 0 is certain, below about -1 is a guess
    no_speech_prob: float = 0.0    # 1 = it thinks this was not speech at all

    @property
    def rtf(self) -> float:
        return self.seconds / self.audio_seconds if self.audio_seconds else 0.0


# What Whisper writes for silence, a click or the tail of the wake word. None of it is a request.
_HALLUCINATION = re.compile(
    r"^\W*(?:thanks?(?: you)? for (?:watching|listening)|please (?:like and )?subscribe|like and subscribe|"
    r"see you (?:in the )?next (?:time|video)|\[?(?:blank[_ ]audio|silence|music|noise|applause)\]?|"
    r"you|uh+|um+|hm+|mm+|ah+|oh+|huh)\W*$", re.I)
_SHORT_WORDS = 3            # "You what?" is three words or fewer; a real command that short is almost always confident
_SHORT_MIN_LOGPROB = -0.9   # below this, a very short transcript is a guess
_MIN_AUDIO_S = 0.3          # shorter than this is a click or a cough


def looks_like_junk(t: Transcript) -> bool:
    """True when a transcript should be dropped instead of answered: empty, a known Whisper hallucination, a blip of audio,
    or a very short guess Whisper itself is unsure of. "Bye" and "thank you" are real phrases EV answers, so they stay."""
    text = t.text.strip()
    if not text:
        return True
    if t.audio_seconds and t.audio_seconds < _MIN_AUDIO_S:
        return True
    if _HALLUCINATION.match(text):
        return True
    words = len(text.split())
    if t.no_speech_prob > 0.8 and t.avg_logprob < -1.0:
        return True
    if words <= _SHORT_WORDS and (t.avg_logprob < _SHORT_MIN_LOGPROB or t.no_speech_prob > 0.5):
        return True
    return False


# Names Whisper splits or mishears in real use ("Open Fire Fox", "open you due"). Kept deliberately small and literal.
_NAME_FIXES = [
    (re.compile(r"\bfire[ -]?fox\b", re.I), "Firefox"),
    (re.compile(r"\b(?:you|u)[ -]?tube\b", re.I), "YouTube"),
    (re.compile(r"\b(open|launch|go to|on|and|to) (?:you due|you dude|u tube)\b", re.I), r"\1 YouTube"),
    (re.compile(r"\b(?:makersoft|macrosoft|micro soft|microsft)\b", re.I), "Microsoft"),
    (re.compile(r"\bg[ -]?mail\b", re.I), "Gmail"),
    (re.compile(r"\bgit[ -]hub\b", re.I), "GitHub"),
    (re.compile(r"\bdis?play mode\b", re.I), "display mode"),
    # "dashboard" is the word Whisper breaks up most: "dash board", "dash bored", "dashable", "the dash" after open/show
    (re.compile(r"\bdash[ -]?(?:board|bored|bord|boards|able|abel|aboard)\b", re.I), "dashboard"),
    (re.compile(r"\b(open|show|launch|pull up|bring up)( me)?( the| my)? dash\b", re.I), r"\1\2\3 dashboard"),
    (re.compile(r"\b(new|another|different|same|next|this|that|the) (?:talk|top|tap|tub|tag)\b(?! (?:about|with|to|of|show|is|was))", re.I), r"\1 tab"),    # "open YouTube in a new talk"
    (re.compile(r"^\W*(?:pin|pen|opened|opening)\s+(?:the\s+)?dashboard\b", re.I), "open the dashboard"),   # heard in real use: "Pin Dashable"
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


def add_cuda_dll_dirs(roots: list[Path] | None = None) -> list[str]:
    """Let Windows find the CUDA libraries that `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` puts under
    site-packages/nvidia/<lib>/bin. CTranslate2 loads them by name, and pip does not put them on the DLL search path.
    Returns the folders added (none when those packages are not installed)."""
    if roots is None:
        spec = importlib.util.find_spec("nvidia")
        roots = [Path(p) for p in (spec.submodule_search_locations or [])] if spec else []
    added: list[str] = []
    for root in roots:
        for bin_dir in sorted(root.glob("*/bin")):
            if not bin_dir.is_dir() or str(bin_dir) in added:
                continue
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(bin_dir))
    return added


class STT:
    def __init__(self, cfg: Config, model_name: str | None = None):
        self.c = cfg["stt"]
        self.model_name = model_name or self.c["model"]
        self.sr = cfg["audio"]["sample_rate"]
        self.hotwords = " ".join(self.c.get("hotwords", [])) or None
        self._model: Any = None
        self.device = "cpu"                  # where the loaded model really runs (set by load)
        self._lock = threading.Lock()
        self._decode_lock = threading.Lock()   # the barge-in monitor and the main turn must not decode on the same model at once

    def _build(self, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        # Offline by default: faster-whisper otherwise pings Hugging Face on every load.
        return WhisperModel(self.model_name, device=device, compute_type=compute_type,
                            cpu_threads=self.c["cpu_threads"], local_files_only=True)

    def _build_gpu(self) -> Any | None:
        """The model on the GPU, or None when that does not work here. A CUDA model loads fine without the cuBLAS library and
        only fails on the first decode, so a one-second decode is the real test."""
        try:
            add_cuda_dll_dirs()
            model = self._build("cuda", str(self.c.get("gpu_compute_type", "int8_float16")))
            segs, _ = model.transcribe(np.zeros(self.sr, np.float32), language=self.c["language"], beam_size=1,
                                       vad_filter=False, without_timestamps=True)
            for _ in segs:
                pass
            return model
        except Exception as e:                      # missing DLL, no CUDA device, out of memory: all mean "use the CPU"
            log.warning("whisper on the GPU is not available (%s); using the CPU", str(e).splitlines()[0][:140] if str(e) else type(e).__name__)
            return None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            t0 = time.perf_counter()
            device = str(self.c["device"])
            try:
                model = self._build_gpu() if device in ("auto", "cuda") else None
                if model is not None:
                    self.device = "cuda"
                else:
                    self.device = "cpu"
                    model = self._build("cpu", self.c["compute_type"])
                self._model = model
            except Exception as e:
                raise RuntimeError(
                    f"Whisper model '{self.model_name}' is not downloaded. Run: python scripts/download_models.py "
                    f"--whisper {self.model_name}") from e
            log.info("whisper %s loaded on %s in %.1fs", self.model_name, self.device, time.perf_counter() - t0)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        self.load()
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x = x / 32768.0
        audio_s = len(x) / self.sr
        x = clean_audio(x, self.sr)
        t0 = time.perf_counter()
        assert self._model is not None
        with self._decode_lock:
            segs, _ = self._model.transcribe(
                x, language=self.c["language"], beam_size=self.c["beam_size"],
                vad_filter=bool(self.c.get("internal_vad", True)), condition_on_previous_text=False, without_timestamps=True,
                vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 250},
                temperature=[0.0, 0.2, 0.4],            # retry hotter only when the greedy pass looks like garbage
                no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
                hotwords=self.hotwords,
            )
            parts = list(segs)
        text = fix_names(" ".join(s.text.strip() for s in parts).strip())
        logprob = sum(s.avg_logprob for s in parts) / len(parts) if parts else 0.0
        no_speech = max((s.no_speech_prob for s in parts), default=0.0)
        return Transcript(text, time.perf_counter() - t0, audio_s, logprob, no_speech)

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
