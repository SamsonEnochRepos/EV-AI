"""Record your own voice for the wake-word model (strongly recommended: real mic + real voice beat any TTS).

    .venv\\Scripts\\python scripts/wakeword/record.py            # ~3 min: positives, then 90 s of everyday talk
    .venv\\Scripts\\python scripts/wakeword/record.py --reps 6   # more takes per phrase

Positives are saved as 2 s clips in data/wakeword/recordings/pos, negatives in .../neg. Re-run features.py and
train.py afterwards; every 5th clip is held out so evaluate.py can report recall on your voice.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))
from common import POSITIVE_PHRASES, SR, WORK  # noqa: E402


def rec(seconds: float, device: int | str | None) -> np.ndarray:
    a = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16", device=device)
    sd.wait()
    return a[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4, help="takes per phrase")
    ap.add_argument("--neg-seconds", type=int, default=90, help="seconds of ordinary talking to record as negatives")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    pos, neg = WORK / "recordings" / "pos", WORK / "recordings" / "neg"
    pos.mkdir(parents=True, exist_ok=True)
    neg.mkdir(parents=True, exist_ok=True)
    n_pos = len(list(pos.glob("*.wav")))

    phrases = ["hey evie", "hey E V", "evie", "E V"] * 1
    print(f"\nSay each phrase naturally when it appears; vary distance, tone and speed.\n"
          f"Phrases: {phrases}, {a.reps} takes each. Press Enter before each take (2 s recording).")
    takes = [p for p in phrases for _ in range(a.reps)]
    random.shuffle(takes)
    for i, p in enumerate(takes, 1):
        input(f"[{i}/{len(takes)}] say  \"{p}\"   (Enter, then speak) ")
        time.sleep(0.15)
        clip = rec(2.0, a.device)
        wavfile.write(pos / f"{n_pos:04d}.wav", SR, clip)
        n_pos += 1

    input(f"\nNow {a.neg_seconds} s of ordinary talking, reading, humming, typing: anything EXCEPT the wake phrase. "
          "Press Enter to start. ")
    long = rec(a.neg_seconds, a.device)
    n_neg = len(list(neg.glob("*.wav")))
    for s in range(0, len(long) - 2 * SR + 1, 2 * SR):
        wavfile.write(neg / f"{n_neg:04d}.wav", SR, long[s:s + 2 * SR])
        n_neg += 1
    print(f"\nsaved {n_pos} positive and {n_neg} negative clips under {WORK / 'recordings'}")
    print("next: features.py -> train.py -> evaluate.py")


if __name__ == "__main__":
    main()
