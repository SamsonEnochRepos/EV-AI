"""Soak test: periodic synthetic interactions while sampling RSS / VRAM / CPU. `python -m ev.evals.soak --minutes 120`.

Interaction = Piper-synthesized utterance -> VAD -> Whisper -> brain (Ollama, stub tools) -> Piper -> sink.
Reports memory growth of the EV process (first-10% baseline vs last-10%), crashes, and latency drift.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np
import psutil

from ev.audio.capture import Recorder
from ev.audio.playback import Player
from ev.audio.vad import make_vad
from ev.core.config import load_config
from ev.stt.whisper import STT
from ev.tts.piper_engine import PiperEngine
from ev.tts.speaker import Speaker

from .bench import vram_mb
from .common import RESULTS
from .functional import make_recording_brain
from .stt_eval import UTTERANCES, resample_16k

BLOCK = 480


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--every", type=float, default=25.0, help="seconds between interactions")
    ap.add_argument("--sample", type=float, default=30.0, help="seconds between resource samples")
    a = ap.parse_args()

    cfg = load_config()
    tts = PiperEngine(cfg)
    tts.load()
    stt = STT(cfg)
    stt.load()
    brain, calls, mem = make_recording_brain(cfg)
    brain.client.warm_up()
    player = Player(sink=lambda seg, sr: None)
    speaker = Speaker(tts, player)
    clips = []
    for u in UTTERANCES:
        x = (resample_16k(np.concatenate(list(tts.synthesize(u))), tts.sample_rate) * 32767).astype(np.int16)
        clips.append((u, np.concatenate([np.zeros(4800, np.int16), x, np.zeros(14400, np.int16)])))

    me = psutil.Process()
    samples: list[dict[str, Any]] = []
    stop = threading.Event()

    def sampler() -> None:
        me.cpu_percent(None)
        while not stop.wait(a.sample):
            samples.append({"t": time.time(), "rss_mb": me.memory_info().rss / 1e6,
                            "cpu_pct": me.cpu_percent(None) / psutil.cpu_count(), "vram_mb": vram_mb(),
                            "threads": me.num_threads(), "sys_ram_free_mb": psutil.virtual_memory().available / 1e6})

    threading.Thread(target=sampler, daemon=True).start()
    rng = random.Random(7)
    ok = errors = 0
    lat: list[float] = []
    err_samples: list[str] = []
    t_end = time.time() + a.minutes * 60
    while time.time() < t_end:
        u, audio = rng.choice(clips)
        t0 = time.perf_counter()
        try:
            vr = Recorder(cfg, make_vad("energy"))
            blocks = (audio[i:i + BLOCK] for i in range(0, len(audio) - BLOCK + 1, BLOCK))
            got = vr.record(blocks)
            if got is None:
                raise RuntimeError("no speech detected")
            text = stt.transcribe(got).text
            brain.history.clear() if rng.random() < 0.5 else None
            res = brain.respond(text, on_sentence=speaker.say)
            speaker.wait_done(30)
            if not res.text:
                raise RuntimeError("empty reply")
            ok += 1
            lat.append(time.perf_counter() - t0)
        except Exception:      # noqa: BLE001
            errors += 1
            err_samples.append(traceback.format_exc()[-300:])
        time.sleep(max(0.0, a.every - (time.perf_counter() - t0)))
    stop.set()
    speaker.close()
    mem.close()

    def growth(key: str) -> float | None:
        v = [s[key] for s in samples if s.get(key) is not None]
        if len(v) < 10:
            return None
        k = max(2, len(v) // 10)
        return round(100 * (statistics.mean(v[-k:]) - statistics.mean(v[:k])) / max(1e-9, statistics.mean(v[:k])), 2)

    third = max(1, len(lat) // 3)
    out = {
        "minutes": a.minutes, "interactions_ok": ok, "errors": errors, "error_samples": err_samples[:3],
        "rss_growth_pct": growth("rss_mb"), "vram_growth_pct": growth("vram_mb"),
        "rss_mb_start": samples[0]["rss_mb"] if samples else None, "rss_mb_end": samples[-1]["rss_mb"] if samples else None,
        "rss_mb_max": max((s["rss_mb"] for s in samples), default=None),
        "vram_mb_max": max((s["vram_mb"] for s in samples if s["vram_mb"]), default=None),
        "cpu_pct_mean": round(statistics.mean(s["cpu_pct"] for s in samples), 2) if samples else None,
        "latency_first_third_s": round(statistics.mean(lat[:third]), 3) if lat else None,
        "latency_last_third_s": round(statistics.mean(lat[-third:]), 3) if lat else None,
        "min_sys_ram_free_mb": round(min((s["sys_ram_free_mb"] for s in samples), default=0)),
        "samples": samples,
        "note": "performance throttling not directly measurable here; compare first vs last third latency for drift",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "soak.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print({k: v for k, v in out.items() if k not in ("samples", "error_samples")})
    return 0 if errors == 0 and (out["rss_growth_pct"] or 0) <= 10 else 1


if __name__ == "__main__":
    sys.exit(main())
