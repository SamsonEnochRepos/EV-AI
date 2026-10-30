"""Measure the trained model the way EV uses it: streamed through openwakeword.Model in 80 ms frames.

  recall            held-out *Kokoro* voices (a different TTS than training) saying each phrase, plus your own
                    recordings if you made some (scripts/wakeword/record.py)
  false triggers/h  LibriSpeech test-clean (5 h of read speech, never seen in training) and ESC-50 fold 5 noise
  near-miss         Kokoro voices saying words that sound like the wake phrase

    python scripts/wakeword/evaluate.py [--model hey_evie] [--hours 2]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
from common import ADVERSARIAL, OWW_DIR, POSITIVE_PHRASES, RAW, ROOT, SR, WORK  # noqa: E402
from features import read_wav  # noqa: E402

BLOCK = 1280
REFRACTORY_S = 2.0
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_model(name: str):
    from openwakeword.model import Model

    return Model(wakeword_models=[str(OWW_DIR / f"{name}.onnx")], inference_framework="onnx",
                 melspec_model_path=str(OWW_DIR / "melspectrogram.onnx"),
                 embedding_model_path=str(OWW_DIR / "embedding_model.onnx"))


def scores(model, audio: np.ndarray) -> np.ndarray:
    """Per-80 ms-frame scores for int16 audio, exactly as EV's WakeWord loop would feed them."""
    model.reset()
    out = []
    for i in range(0, len(audio) - BLOCK + 1, BLOCK):
        out.append(max(model.predict(audio[i:i + BLOCK]).values()))
    return np.array(out)


def count_triggers(s: np.ndarray, thr: float) -> int:
    n, last = 0, -1e9
    for i, v in enumerate(s):
        if v >= thr and (i - last) * BLOCK / SR >= REFRACTORY_S:
            n += 1
            last = i
    return n


def to_int16(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


def kokoro_clips(texts: list[str], speeds: tuple[float, ...], voices: list[str], lead_s: float = 1.2) -> list[np.ndarray]:
    from kokoro_onnx import Kokoro

    k = Kokoro(str(ROOT / "models/kokoro/kokoro-v1.0.onnx"), str(ROOT / "models/kokoro/voices-v1.0.bin"))
    rng = random.Random(3)
    clips = []
    for v in voices:
        for sp in speeds:
            for t in texts:
                a, sr = k.create(t, voice=v, speed=sp, lang="en-us" if v[0] == "a" else "en-gb")
                a = resample_poly(a, SR, sr).astype(np.float32)
                pad = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(int(lead_s * SR)) * 0.002
                clips.append(to_int16(np.concatenate([pad, a, pad[: SR]])))
    return clips


def long_audio(files: list[Path], seconds: float, flac: bool) -> np.ndarray:
    parts, n = [], 0
    for f in files:
        a = sf.read(f, dtype="float32")[0] if flac else read_wav(f)
        parts.append(a)
        n += len(a)
        if n >= seconds * SR:
            break
    return to_int16(np.concatenate(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hey_evie")
    ap.add_argument("--hours", type=float, default=2.0, help="hours of LibriSpeech test-clean to stream")
    a = ap.parse_args()
    model = load_model(a.model)
    print(f"model: {a.model}   thresholds: {THRESHOLDS}\n")

    # ---- recall on unseen synthetic voices
    voices = ["af_heart", "af_bella", "af_nicole", "af_sky", "af_sarah", "am_adam", "am_michael", "am_eric", "am_liam",
              "bf_emma", "bf_isabella", "bm_george", "bm_daniel", "bm_lewis"]
    phrases = sorted(set(POSITIVE_PHRASES))
    pos_scores: dict[str, list[float]] = {p: [] for p in phrases}
    for p in phrases:
        for c in kokoro_clips([p], (0.9, 1.05, 1.2), voices):
            pos_scores[p].append(float(scores(model, c).max()))
    print("recall on held-out Kokoro voices (fraction of clips whose peak score reaches the threshold)")
    print(f"  {'phrase':14s}" + "".join(f"{t:>7.1f}" for t in THRESHOLDS))
    allp = np.concatenate([np.array(v) for v in pos_scores.values()])
    for p in phrases:
        v = np.array(pos_scores[p])
        print(f"  {p:14s}" + "".join(f"{(v >= t).mean():7.2f}" for t in THRESHOLDS))
    print(f"  {'ALL':14s}" + "".join(f"{(allp >= t).mean():7.2f}" for t in THRESHOLDS))

    # ---- your own voice
    mine = sorted((WORK / "recordings" / "pos").glob("*.wav"))[::5]
    if mine:
        ms = np.array([float(scores(model, to_int16(np.concatenate([np.zeros(SR), read_wav(f), np.zeros(SR)]))).max())
                       for f in mine])
        print(f"  {'YOUR VOICE':14s}" + "".join(f"{(ms >= t).mean():7.2f}" for t in THRESHOLDS) + f"   ({len(mine)} held-out clips)")

    # ---- near-miss words from unseen voices
    adv = sorted(set(ADVERSARIAL))
    advs = np.array([float(scores(model, c).max()) for c in kokoro_clips(adv, (1.05,), voices[:6])])
    print(f"\nnear-miss words from unseen voices that trigger ({len(advs)} clips)")
    print(f"  {'':14s}" + "".join(f"{(advs >= t).mean():7.3f}" for t in THRESHOLDS))

    # ---- false triggers per hour on real speech + noise
    files = sorted((RAW / "librispeech-test").rglob("*.flac"))
    random.Random(5).shuffle(files)
    speech = long_audio(files, a.hours * 3600, flac=True)
    hours = len(speech) / SR / 3600
    ss = scores(model, speech)
    esc = RAW / "esc50" / "ESC-50-master"
    rows = (esc / "meta" / "esc50.csv").read_text().splitlines()[1:] if esc.exists() else []
    noise_files = [esc / "audio" / r.split(",")[0] for r in rows if r.split(",")[1] == "5"]
    ns = scores(model, long_audio(noise_files, 1e9, flac=False)) if noise_files else np.array([0.0])
    nh = len(ns) * BLOCK / SR / 3600
    print(f"\nfalse triggers per hour ({hours:.1f} h LibriSpeech test-clean, {nh * 60:.0f} min ESC-50 fold 5)")
    print(f"  {'speech':14s}" + "".join(f"{count_triggers(ss, t) / hours:7.2f}" for t in THRESHOLDS))
    print(f"  {'noise':14s}" + "".join(f"{count_triggers(ns, t) / nh:7.2f}" for t in THRESHOLDS))


if __name__ == "__main__":
    main()
