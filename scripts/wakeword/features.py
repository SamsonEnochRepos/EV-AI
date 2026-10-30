"""Turn the synthesised + public audio into augmented openWakeWord embeddings (16 x 96 per 2 s clip).

Every training example is one 2 s clip, which the shared melspectrogram + embedding models turn into exactly one
16x96 window, the same thing the runtime classifier sees on every 80 ms step. Positives end their phrase in the last
0-0.35 s of the clip; negatives are anything else (near-miss words placed anywhere, real speech, noise, silence).

    python scripts/wakeword/features.py            # needs synth.py and fetch_data.py to have run
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

sys.path.insert(0, str(Path(__file__).parent))
from common import FEATS, OWW_DIR, RAW, SR, SYNTH, WORK  # noqa: E402

CLIP = 2 * SR


def read_wav(p: Path) -> np.ndarray:
    sr, a = wavfile.read(p)
    a = a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = resample_poly(a, SR, sr)
    return a


def trim_silence(a: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    idx = np.flatnonzero(np.abs(a) > thresh * max(1e-6, np.abs(a).max()) * 5)
    return a[max(0, idx[0] - 800): idx[-1] + 800] if len(idx) else a


def read_flac_like(p: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(p, dtype="float32")
    return a if sr == SR else resample_poly(a, SR, sr)


class Noise:
    """Background sources: ESC-50 clips (by fold, so validation noise is unseen), LibriSpeech babble, synthetic noise."""

    def __init__(self, esc_folds: set[int], speech_files: list[Path]):
        meta = RAW / "esc50" / "ESC-50-master"
        rows = (meta / "meta" / "esc50.csv").read_text().splitlines()[1:] if meta.exists() else []
        self.esc = [meta / "audio" / r.split(",")[0] for r in rows if int(r.split(",")[1]) in esc_folds]
        self.speech = speech_files
        self._cache: dict[Path, np.ndarray] = {}

    def _load(self, p: Path, flac: bool) -> np.ndarray:
        if p not in self._cache:
            if len(self._cache) > 400:
                self._cache.clear()
            self._cache[p] = read_flac_like(p) if flac else read_wav(p)
        return self._cache[p]

    def crop(self, rng: random.Random, n: int, kind: str | None = None) -> np.ndarray:
        kind = kind or rng.choice(["esc", "esc", "speech", "white", "pink", "none"])
        if kind == "esc" and self.esc:
            a = self._load(rng.choice(self.esc), False)
        elif kind == "speech" and self.speech:
            a = self._load(rng.choice(self.speech), True)
        elif kind in ("white", "pink"):
            w = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(n).astype(np.float32)
            if kind == "pink":
                w = sosfilt(butter(1, 800, "low", fs=SR, output="sos"), w).astype(np.float32)
            return w / (np.abs(w).max() + 1e-6)
        else:
            return np.zeros(n, np.float32)
        if len(a) < n:
            a = np.tile(a, n // len(a) + 1)
        s = rng.randrange(len(a) - n + 1)
        return a[s:s + n].astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-9)


def reverb(x: np.ndarray, rng: random.Random) -> np.ndarray:
    rt60 = rng.uniform(0.1, 0.6)
    t = np.arange(int(rt60 * SR)) / SR
    rir = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    y = fftconvolve(x, rir.astype(np.float32))[: len(x)]
    return (y * rms(x) / rms(y)).astype(np.float32)


def colour(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random mic/room EQ: cheap laptop mics roll off at both ends."""
    if rng.random() < 0.5:
        x = sosfilt(butter(2, rng.uniform(60, 250), "high", fs=SR, output="sos"), x)
    if rng.random() < 0.4:
        x = sosfilt(butter(2, rng.uniform(3500, 7000), "low", fs=SR, output="sos"), x)
    return x.astype(np.float32)


def place(speech: np.ndarray, rng: random.Random, end_back: tuple[float, float]) -> np.ndarray:
    """Put `speech` in a 2 s clip so that it ends `end_back` seconds (a range) before the clip end."""
    out = np.zeros(CLIP, np.float32)
    end = CLIP - int(rng.uniform(*end_back) * SR)
    end = max(min(end, CLIP), min(len(speech), CLIP) // 2)
    start = end - len(speech)
    s0, d0 = (0, start) if start >= 0 else (-start, 0)
    seg = speech[s0: s0 + (CLIP - d0)][: max(0, end - d0)]
    out[d0: d0 + len(seg)] = seg
    return out


def augment(speech: np.ndarray | None, noise: Noise, rng: random.Random, end_back: tuple[float, float]) -> np.ndarray:
    """One 2 s float clip: optional speech placed in it, then reverb, gain, background and EQ."""
    if speech is None:
        clip = np.zeros(CLIP, np.float32)
    else:
        f = rng.uniform(0.9, 1.1)                         # speed + pitch shift in one resample
        speech = resample_poly(speech, 100, int(round(100 * f))).astype(np.float32) if abs(f - 1) > 0.01 else speech
        clip = place(speech, rng, end_back)
        if rng.random() < 0.35:
            clip = reverb(clip, rng)
        clip *= 10 ** (rng.uniform(-14, -1) / 20) / (np.abs(clip).max() + 1e-6)
    bg = noise.crop(rng, CLIP)
    if np.any(bg):
        level = rms(clip) / (10 ** (rng.uniform(0, 30) / 20)) if speech is not None else 10 ** (rng.uniform(-45, -12) / 20)
        clip = clip + bg / rms(bg) * level
    clip = colour(clip, rng)
    return (np.clip(clip, -1, 1) * 32767).astype(np.int16)


def embed(clips: list[np.ndarray]) -> np.ndarray:
    from openwakeword.utils import AudioFeatures

    f = AudioFeatures(melspec_model_path=str(OWW_DIR / "melspectrogram.onnx"),
                      embedding_model_path=str(OWW_DIR / "embedding_model.onnx"), ncpu=4)
    out = f.embed_clips(np.stack(clips), batch_size=64)
    return out[:, -16:, :].astype(np.float32)


def build(name: str, jobs: list[tuple[str, np.ndarray | None, tuple[float, float], int]], noise: Noise, seed: int,
          label: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """jobs: (kind, speech, end_back range, n_augmentations). Returns X, y, kinds."""
    rng = random.Random(seed)
    xs, ys, ks = [], [], []
    clips: list[np.ndarray] = []
    kinds: list[str] = []

    def flush() -> None:
        if clips:
            xs.append(embed(clips))
            ks.extend(kinds)
            clips.clear()
            kinds.clear()

    for i, (kind, speech, end_back, reps) in enumerate(jobs):
        for _ in range(reps):
            clips.append(augment(speech, noise, rng, end_back))
            kinds.append(kind)
            ys.append(label[kind])
        if len(clips) >= 512:
            flush()
            print(f"  {name}: {i + 1}/{len(jobs)}", flush=True)
    flush()
    return np.concatenate(xs), np.array(ys, np.float32), np.array(ks)


def synth_jobs(kind: str, label_kind: str, reps: int, end_back: tuple[float, float], split: str) -> list:
    files = sorted((SYNTH / kind).glob("*.wav"))
    keep = [f for f in files if (int(f.stem) % 10 == 0) == (split == "val")]
    return [(label_kind, trim_silence(read_wav(f)), end_back, reps) for f in keep]


def libri_windows(files: list[Path], per_file: int, rng: random.Random) -> list:
    import soundfile as sf

    jobs = []
    for f in files:
        a, sr = sf.read(f, dtype="float32")
        for _ in range(per_file):
            if len(a) <= CLIP:
                continue
            s = rng.randrange(len(a) - CLIP)
            jobs.append(("real", a[s:s + CLIP], (0.0, 0.0), 1))
    return jobs


def user_jobs(split: str, reps: int) -> list:
    """Your own voice (scripts/wakeword/record.py): every 5th file is held out for validation."""
    jobs = []
    for sub, kind, end_back in (("pos", "user_pos", (0.0, 0.35)), ("neg", "user_neg", (0.0, 1.2))):
        files = sorted((WORK / "recordings" / sub).glob("*.wav"))
        for i, f in enumerate(files):
            if (i % 5 == 0) == (split == "val"):
                jobs.append((kind, trim_silence(read_wav(f)), end_back, reps))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="augmented copies per synthesised clip")
    a = ap.parse_args()
    FEATS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)

    dev = sorted((RAW / "librispeech-dev").rglob("*.flac"))
    label = {"pos": 1, "user_pos": 1, "adv": 0, "gen": 0, "real": 0, "noise": 0, "silence": 0, "user_neg": 0}
    for split in ("train", "val"):
        noise = Noise({1, 2, 3, 4} if split == "train" else {5}, dev)
        val = split == "val"
        reps = 1 if val else a.reps
        jobs = (
            synth_jobs("pos", "pos", reps, (0.0, 0.35), split)
            + synth_jobs("adv", "adv", reps, (0.0, 1.4), split)
            + synth_jobs("gen", "gen", reps, (0.0, 1.4), split)
            + user_jobs(split, 20 if not val else 1)
        )
        real_files = [f for i, f in enumerate(dev) if (i % 10 == 0) == val]
        jobs += libri_windows(real_files, 3 if not val else 1, rng)
        jobs += [("noise", None, (0.0, 0.0), 1)] * (3000 if not val else 300)         # noise-only / near silence
        rng.shuffle(jobs)
        print(f"{split}: {len(jobs)} base clips")
        X, y, kinds = build(split, jobs, noise, 11 if not val else 12, label)
        np.savez(FEATS / f"{split}.npz", X=X, y=y, kinds=kinds)
        print(f"{split}: X {X.shape}, positives {int(y.sum())}")


if __name__ == "__main__":
    main()
