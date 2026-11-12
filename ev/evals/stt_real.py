"""Which speech setting understands YOUR voice on YOUR microphone?  python -m ev.evals.stt_real

The other speech evals use a synthetic voice, which is too clean to show what goes wrong in real use. This one asks you to read
a short list of sentences aloud (the kind of thing you say to EV, including the words it keeps getting wrong), records each one
from the real microphone, and then runs every candidate model / setting over the same recordings and scores them.

Close EV first (it uses the microphone and the GPU). Nothing is saved: the audio stays in memory and is dropped when this ends.
The scores and what each model heard (text only) go to evals/results/stt_real.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from typing import Any

import numpy as np

from ev.audio.capture import Recorder, mic_blocks
from ev.audio.vad import make_vad
from ev.core.config import Config, load_config
from ev.stt.whisper import STT

SENTENCES = [
    "open the dashboard",
    "can you open YouTube in a new tab",
    "set a timer for ten minutes",
    "what time is it",
    "open Notepad and type hello world",
    "remind me in twenty minutes to stretch",
    "turn the volume up a little",
    "search for the height of Mount Everest",
    "tell me a joke",
    "what is on my screen right now",
    "play some music on Spotify",
    "I want you to open it in Firefox instead",
    "Evie, don't open it",
    "Evie, stop",
]


def norm(t: str) -> str:
    t = t.lower().replace("-", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", t)).strip()


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref).split(), (norm(hyp) or "<nothing>").split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(len(r), 1)


def record_all(cfg: Config) -> list[tuple[str, np.ndarray, float]]:
    vad = make_vad("silero", cfg["audio"]["vad_aggressiveness_threshold"])
    rec = Recorder(cfg, vad)
    clips: list[tuple[str, np.ndarray, float]] = []
    print("\nRead each sentence in your normal voice, at your normal distance from the laptop, right after you press Enter.")
    print("If it did not catch you, it will say so and ask again.\n")
    for n, s in enumerate(SENTENCES, 1):
        while True:
            input(f"[{n}/{len(SENTENCES)}]  press Enter, then say:  \"{s}\"  ")
            print("   listening...", flush=True)
            audio = rec.record(mic_blocks(cfg, lambda: False), wait_for_speech_s=6)
            if audio is None:
                print("   I did not hear anything. Try again.")
                continue
            peak = float(np.max(np.abs(audio))) / 32768
            print(f"   got {len(audio) / 16000:.1f} s (loudest point {peak:.2f}{' - quiet, move closer or turn the mic up' if peak < 0.05 else ''})")
            clips.append((s, audio, peak))
            break
    return clips


def candidates(cfg: Config) -> list[dict[str, Any]]:
    from huggingface_hub import try_to_load_from_cache  # type: ignore[import-untyped]

    def have(repo: str) -> bool:
        return try_to_load_from_cache(repo, "model.bin") is not None

    out: list[dict[str, Any]] = [{"name": "distil-small.en (today)", "model": "distil-small.en"}]
    out.append({"name": "distil-small.en on the CPU (the old way)", "model": "distil-small.en", "device": "cpu",
                "hotwords": ["Evie", "EV", "Sam", "Spotify", "Firefox", "Chrome", "Discord", "Notepad", "Ollama", "VS Code", "YouTube",
                             "display mode"]})
    out.append({"name": "distil-small.en, no hint words", "model": "distil-small.en", "hotwords": []})
    out.append({"name": "distil-small.en, no second voice filter", "model": "distil-small.en", "internal_vad": False})
    if have("Systran/faster-whisper-small"):
        out.append({"name": "small", "model": "small"})
    if have("Systran/faster-distil-whisper-large-v3"):
        out.append({"name": "distil-large-v3 (GPU)", "model": "distil-large-v3"})
        out.append({"name": "distil-large-v3 (CPU)", "model": "distil-large-v3", "device": "cpu"})
    return out


def run_candidate(base: Config, spec: dict[str, Any], clips: list[tuple[str, np.ndarray, float]]) -> dict[str, Any]:
    cfg = load_config()
    stt_cfg = cfg.raw["stt"]
    for k in ("hotwords", "internal_vad", "device"):
        if k in spec:
            stt_cfg[k] = spec[k]
    stt = STT(cfg, spec["model"])
    stt.load()
    stt.warm()
    rows, secs, wers = [], [], []
    for said, audio, _peak in clips:
        t = stt.transcribe(audio)
        w = wer(said, t.text)
        rows.append({"said": said, "heard": t.text, "wer": round(w, 2), "seconds": round(t.seconds, 2)})
        secs.append(t.seconds)
        wers.append(w)
    stt.unload()
    return {"name": spec["name"], "device": stt.device, "wer": round(statistics.mean(wers), 3), "exact": sum(w == 0 for w in wers),
            "median_s": round(statistics.median(secs), 2), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m ev.evals.stt_real")
    ap.add_argument("--only", help="run just the candidates whose name contains this text")
    a = ap.parse_args()
    cfg = load_config()
    clips = record_all(cfg)
    results = []
    for spec in candidates(cfg):
        if a.only and a.only.lower() not in spec["name"].lower():
            continue
        print(f"\nscoring: {spec['name']} ...", flush=True)
        t0 = time.perf_counter()
        try:
            results.append(run_candidate(cfg, spec, clips))
        except Exception as e:                           # noqa: BLE001  (a model that cannot load is a result too)
            print(f"   could not run: {e}")
            continue
        r = results[-1]
        print(f"   word error {r['wer'] * 100:.0f}%   exact {r['exact']}/{len(clips)}   {r['median_s']} s each on the {r['device']}"
              f"   (scored in {time.perf_counter() - t0:.0f} s)")
    print("\n=== summary (lower word error is better) ===")
    for r in sorted(results, key=lambda r: r["wer"]):
        print(f"  {r['wer'] * 100:5.1f}%   exact {r['exact']:>2}/{len(clips)}   {r['median_s']:>5} s   {r['name']}")
    if results:
        best, today = min(results, key=lambda r: r["wer"]), results[0]
        print(f"\nwhat each of the two heard, sentence by sentence ({today['name']}  vs  {best['name']}):")
        for a_row, b_row in zip(today["rows"], best["rows"], strict=True):
            print(f"  said : {a_row['said']}\n    today: {a_row['heard']}\n    best : {b_row['heard']}")
    from ev.evals.common import RESULTS  # noqa: PLC0415

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stt_real.json").write_text(
        json.dumps({"clips": [{"said": s, "loudest": round(p, 3)} for s, _a, p in clips], "results": results}, indent=2), encoding="utf-8")
    print(f"\nscores saved to {RESULTS / 'stt_real.json'} (text only, no audio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
