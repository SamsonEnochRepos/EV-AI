"""Synthesise wake-word training audio with Piper (hfc_female, EV's own voice, gets the biggest share).

Positives, near-miss negatives and generic-speech negatives all draw from the *same* voice mix and the same prosody
ranges, otherwise the classifier would learn "synthetic voice vs. not" instead of "Evie vs. not".

    python scripts/wakeword/synth.py --pos 6000 --adv 5000 --gen 5000 --workers 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    ADVERSARIAL, COMMANDS, PIPER_DIR, POSITIVE_PHRASES, POSITIVE_WEIGHTS, RAW, SR, SYNTH,
)

# (voice, share). hfc_female is EV's main voice; the others add speaker variety.
VOICES = [
    ("en_US-hfc_female-medium", 0.45), ("en_US-amy-medium", 0.10), ("en_US-lessac-medium", 0.08),
    ("en_US-ryan-medium", 0.08), ("en_GB-alan-medium", 0.07), ("en_US-libritts_r-medium", 0.22),
]
_voices: dict[str, object] = {}
_nspk: dict[str, int] = {}


def _load(name: str):
    if name not in _voices:
        from piper import PiperVoice

        _voices.clear()                  # one voice per worker at a time; jobs are sorted by voice (see main)

        _voices[name] = PiperVoice.load(str(PIPER_DIR / f"{name}.onnx"))
        cfg = json.loads((PIPER_DIR / f"{name}.onnx.json").read_text(encoding="utf-8"))
        _nspk[name] = int(cfg.get("num_speakers", 1))
    return _voices[name]


def synth_one(job: dict) -> str | None:
    from piper import SynthesisConfig

    out = SYNTH / job["kind"] / f"{job['id']:06d}.wav"
    if out.exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        v = _load(job["voice"])
        sc = SynthesisConfig(
            speaker_id=job["speaker"] if _nspk[job["voice"]] > 1 else None, length_scale=job["length"],
            noise_scale=job["noise"], noise_w_scale=job["noise_w"], volume=job["volume"])
        chunks = [c.audio_float_array for c in v.synthesize(job["text"], syn_config=sc)]
        if not chunks:
            return None
        a = np.concatenate(chunks).astype(np.float32)
        g = math.gcd(v.config.sample_rate, SR)
        a = resample_poly(a, SR // g, v.config.sample_rate // g)
        wavfile.write(out, SR, (np.clip(a, -1, 1) * 32767).astype(np.int16))
        return str(out)
    except Exception as e:  # one bad text must not kill a 20-minute run
        print("synth failed:", job["text"][:40], e, file=sys.stderr)
        return None


def _prosody(rng: random.Random) -> dict:
    names, shares = zip(*VOICES, strict=True)
    voice = rng.choices(names, shares)[0]
    return {
        "voice": voice, "speaker": rng.randrange(900), "length": rng.uniform(0.8, 1.3),
        "noise": rng.uniform(0.4, 0.9), "noise_w": rng.uniform(0.4, 1.0), "volume": rng.uniform(0.6, 1.0),
    }


def _punct(rng: random.Random, text: str) -> str:
    if text.startswith("hey ") and rng.random() < 0.4:
        text = "hey, " + text[4:]
    return text + rng.choice(["", "", ".", "?", "!"])


def transcripts() -> list[str]:
    lines = []
    for f in (RAW / "librispeech-dev").rglob("*.trans.txt"):
        for ln in f.read_text(encoding="utf-8").splitlines():
            lines.append(ln.split(" ", 1)[1].lower())
    return lines


def plan(n_pos: int, n_adv: int, n_gen: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    jobs: list[dict] = []
    for i in range(n_pos):
        text = _punct(rng, rng.choices(POSITIVE_PHRASES, POSITIVE_WEIGHTS)[0])
        jobs.append({"kind": "pos", "id": i, "text": text, **_prosody(rng)})
    for i in range(n_adv):
        text = _punct(rng, rng.choice(ADVERSARIAL))
        jobs.append({"kind": "adv", "id": i, "text": text, **_prosody(rng)})
    sents = transcripts()
    if not sents:
        print("no LibriSpeech transcripts yet (run fetch_data.py); skipping generic negatives", file=sys.stderr)
        n_gen = 0
    for i in range(n_gen):
        r = rng.random()
        if r < 0.25:
            text = rng.choice(COMMANDS)
        else:
            words = rng.choice(sents).split()
            n = rng.choice([1, 2, 3, 4, 6, 9, 14])        # short utterances too, so length isn't a giveaway
            start = rng.randrange(max(1, len(words) - n + 1))
            text = " ".join(words[start:start + n])
        jobs.append({"kind": "gen", "id": i, "text": text, **_prosody(rng)})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=6000)
    ap.add_argument("--adv", type=int, default=5000)
    ap.add_argument("--gen", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = plan(a.pos, a.adv, a.gen)
    jobs.sort(key=lambda j: (j["voice"], j["kind"], j["id"]))   # consecutive jobs share a voice -> few model loads
    print(f"{len(jobs)} clips with {a.workers} workers")
    done = 0
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(synth_one, jobs, chunksize=16):
            done += r is not None
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"done: {done} clips in {SYNTH}")


if __name__ == "__main__":
    main()
