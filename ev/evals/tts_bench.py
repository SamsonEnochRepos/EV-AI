"""TTS engines head to head: load time, first-chunk latency, RTF, memory, and round-trip intelligibility (Whisper WER).

Naturalness is subjective and is NOT measured here; WER only shows how intelligible the speech is to an ASR.
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np
import psutil

from ev.core.config import load_config
from ev.stt.whisper import STT
from ev.tts.base import make_engine

from .common import RESULTS
from .stt_eval import resample_16k, wer

SENTENCES = [
    "Sure thing, I'll set that timer for ten minutes.",
    "It's three forty five in the afternoon on Thursday.",
    "I couldn't find that app, but I can search the web for it if you like.",
    "Good morning! You have two reminders today and the weather looks lovely.",
    "Okay, opening Spotify now.",
    "Just checking, do you want me to delete that file?",
    "Your battery is at eighty seven percent, so you're good for a few hours.",
    "I can't do that one, sorry, but I'm happy to help with something else.",
]


def bench(engine_name: str) -> dict:
    cfg = load_config(overrides={"tts": {"engine": engine_name}})
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    eng = make_engine(cfg)
    eng.load()
    load_s = time.perf_counter() - t0
    list(eng.synthesize("warm up"))
    first, rtf, audios = [], [], []
    for s in SENTENCES:
        t = time.perf_counter()
        f = None
        chunks = []
        for ch in eng.synthesize(s):
            if f is None:
                f = (time.perf_counter() - t) * 1000
            chunks.append(ch)
        total = time.perf_counter() - t
        a = np.concatenate(chunks)
        first.append(f or 0.0)
        rtf.append(total / (len(a) / eng.sample_rate))
        audios.append(a)
    stt = STT(cfg, "base")
    w = [wer(s, stt.transcribe(resample_16k(a, eng.sample_rate)).text) for s, a in zip(SENTENCES, audios, strict=True)]
    return {"load_s": round(load_s, 2), "first_chunk_ms_median": round(statistics.median(first), 1),
            "first_chunk_ms_max": round(max(first), 1), "rtf": round(statistics.median(rtf), 3),
            "rss_delta_mb": round((proc.memory_info().rss - rss0) / 1e6), "sample_rate": eng.sample_rate,
            "roundtrip_wer": round(statistics.mean(w), 3)}


def main() -> int:
    out = {}
    for name in sys.argv[1:] or ["piper", "kokoro"]:
        try:
            out[name] = bench(name)
        except Exception as e:      # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
        print(name, out[name])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bench_tts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
