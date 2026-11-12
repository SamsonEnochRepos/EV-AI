"""STT accuracy (WER) and real-time factor, on utterances synthesized with Piper (clean + noisy)."""
from __future__ import annotations

import re
import statistics
import sys

import numpy as np

from ev.core.config import load_config
from ev.stt.whisper import STT
from ev.tts.piper_engine import PiperEngine

from .common import Recorder

UTTERANCES = [
    # commands
    "open Spotify", "set a timer for ten minutes", "what time is it", "turn the volume up",
    "remind me in twenty minutes to stretch", "pause the music", "open Notepad and type hello world",
    "set the volume to forty percent", "take a note buy oat milk", "skip this song",
    "what's on my screen right now", "search the web for the height of Mount Everest",
    # numbers / app names
    "set a timer for forty five seconds", "remind me at three thirty pm to call mom",
    "open Visual Studio Code", "open Discord", "launch Google Chrome", "set the volume to seventy five percent",
    # conversation
    "how are you doing today", "tell me a joke", "I had a really long day at work",
    "what should I make for dinner tonight", "can you remember that my dog is called Biscuit",
    "forget my favorite color", "good morning EV", "thanks that was really helpful",
    # longer
    "I need to finish my report by Friday so remind me tomorrow morning at nine",
    "could you open the calculator and then turn the music down a little",
    "what is the capital of Australia and how far is it from Sydney",
    "please read the text on the screen and tell me if anything looks suspicious",
]


def norm(t: str) -> str:
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(ref: str, hyp: str) -> float:
    import jiwer

    return float(jiwer.wer(norm(ref), norm(hyp) or "<empty>"))


def add_noise(x: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = float(np.mean(x ** 2)) + 1e-9
    n = rng.normal(0, np.sqrt(p / (10 ** (snr_db / 10))), x.shape)
    return np.clip(x + n, -1, 1).astype(np.float32)


def resample_16k(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def run(models: tuple[str, ...] = ("base", "small")) -> Recorder:
    cfg = load_config()
    rec = Recorder("stt")
    tts = PiperEngine(cfg)
    tts.load()
    clips = []
    for u in UTTERANCES:
        a = np.concatenate(list(tts.synthesize(u)))
        a = resample_16k(a, tts.sample_rate)
        pad = np.zeros(4800, np.float32)
        clips.append((u, np.concatenate([pad, a, pad])))
    summary: dict[str, dict[str, float]] = {}
    for m in models:
        stt = STT(cfg, m)
        stt.load()
        res: dict[str, list[float]] = {"clean": [], "noisy10": [], "rtf": []}
        for u, a in clips:
            t = stt.transcribe(a)
            res["clean"].append(wer(u, t.text))
            res["rtf"].append(t.rtf)
            tn = stt.transcribe(add_noise(a, 10))
            res["noisy10"].append(wer(u, tn.text))
            rec.add(f"{m}:{u[:40]}", res["clean"][-1] <= 0.25, "info", f"clean={t.text!r} wer={res['clean'][-1]:.2f} noisy_wer={res['noisy10'][-1]:.2f}")
        summary[m] = {"wer_clean": statistics.mean(res["clean"]), "wer_noisy_10dB": statistics.mean(res["noisy10"]),
                      "rtf_median": statistics.median(res["rtf"]), "rtf_p95": float(np.percentile(res["rtf"], 95))}
        stt.unload()
    rec.summary.update({"models": summary, "n_utterances": len(UTTERANCES),
                        "note": "synthetic Piper speech (one voice), so WER is optimistic vs real human speech"})
    rec.save()
    return rec


if __name__ == "__main__":
    r = run()
    for m, s in r.summary["models"].items():
        print(m, {k: round(v, 3) for k, v in s.items()})
    sys.exit(0)
