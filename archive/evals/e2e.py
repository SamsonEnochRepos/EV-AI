"""End-to-end: synthesized speech -> recorder/VAD -> Whisper -> brain (Ollama) -> Piper -> player sink.

Tools are stubbed (no side effects on the machine). Reports per-stage timing and the latency budget:
"end of speech to first audio". The VAD silence window (silence_ms_to_end) is reported separately because it is
the wait before EV even knows you stopped talking.
"""
from __future__ import annotations

import statistics
import sys
import time
from typing import Any

import numpy as np

from ev.audio.capture import Recorder
from ev.audio.playback import Player
from ev.audio.vad import make_vad
from ev.core.config import load_config
from ev.stt.whisper import STT
from ev.tts.piper_engine import PiperEngine
from ev.tts.speaker import Speaker

from .common import Recorder as ResultRecorder
from .functional import make_recording_brain
from .stt_eval import UTTERANCES, resample_16k

BLOCK = 480  # 30 ms at 16 kHz


def _blocks(audio: np.ndarray):  # noqa: ANN202
    for i in range(0, len(audio) - BLOCK + 1, BLOCK):
        yield audio[i:i + BLOCK]


def run(n: int = 20) -> ResultRecorder:
    cfg = load_config()
    rec = ResultRecorder("e2e")
    tts = PiperEngine(cfg)
    tts.load()
    stt = STT(cfg)
    stt.load()
    brain, calls, mem = make_recording_brain(cfg)
    brain.client.warm_up()
    first_audio: list[float] = []
    player = Player(sink=lambda seg, sr: first_audio.append(time.perf_counter()) if not first_audio else None)
    speaker = Speaker(tts, player)
    rows: list[dict[str, Any]] = []
    sil = np.zeros(int(16000 * 0.9), np.int16)
    for u in UTTERANCES[:n]:
        a = np.concatenate(list(tts.synthesize(u)))
        a16 = (resample_16k(a, tts.sample_rate) * 32767).astype(np.int16)
        audio_in = np.concatenate([sil[:4800], a16, sil])
        vr = Recorder(cfg, make_vad("energy"))
        t0 = time.perf_counter()
        got = vr.record(_blocks(audio_in))
        t_vad = time.perf_counter() - t0
        if got is None:
            rec.add(f"e2e:{u[:40]}", False, "info", "recorder found no speech")
            continue
        t_end = time.perf_counter()                 # end of speech detected
        tr = stt.transcribe(got)
        t_stt = time.perf_counter()
        first_audio.clear()
        calls.clear()
        brain.history.clear()
        res = brain.respond(tr.text, on_sentence=speaker.say)
        t_brain = time.perf_counter()
        speaker.wait_done(30)
        ttfa = (first_audio[0] - t_end) if first_audio else None
        ok = bool(tr.text.strip()) and bool(res.text.strip()) and ttfa is not None
        rows.append({"utt": u, "heard": tr.text, "reply": res.text[:80], "fast": res.fast_path,
                     "tools": [c["tool"] for c in calls], "stt_s": t_stt - t_end,
                     "llm_first_token_s": res.timings.get("first_token", 0.0), "brain_total_s": t_brain - t_stt,
                     "tts_first_chunk_ms": speaker.first_chunk_ms[-1] if speaker.first_chunk_ms else None,
                     "end_of_speech_to_first_audio_s": ttfa, "vad_s": t_vad})
        rec.add(f"e2e:{u[:40]}", ok, "info", f"heard={tr.text!r} reply={res.text[:50]!r} ttfa={ttfa if ttfa is None else round(ttfa, 2)}")
    speaker.close()
    mem.close()
    ttfa_all = [r["end_of_speech_to_first_audio_s"] for r in rows if r["end_of_speech_to_first_audio_s"] is not None]
    conv = [r["end_of_speech_to_first_audio_s"] for r in rows if not r["fast"] and not r["tools"] and r["end_of_speech_to_first_audio_s"]]
    tool = [r["end_of_speech_to_first_audio_s"] for r in rows if r["tools"] and not r["fast"] and r["end_of_speech_to_first_audio_s"]]
    fast = [r["end_of_speech_to_first_audio_s"] for r in rows if r["fast"] and r["end_of_speech_to_first_audio_s"]]

    def med(x: list[float]) -> float | None:
        return round(statistics.median(x), 3) if x else None

    rec.summary.update({
        "runs": len(rows), "success_rate": rec.passed / max(1, len(rec.checks)),
        "ttfa_median_s": med(ttfa_all), "ttfa_p95_s": round(float(np.percentile(ttfa_all, 95)), 3) if ttfa_all else None,
        "ttfa_conversational_median_s": med(conv), "ttfa_tool_median_s": med(tool), "ttfa_fast_path_median_s": med(fast),
        "n_conversational": len(conv), "n_tool": len(tool), "n_fast": len(fast),
        "stt_median_s": med([r["stt_s"] for r in rows]), "llm_first_token_median_s": med([r["llm_first_token_s"] for r in rows if not r["fast"]]),
        "tts_first_chunk_median_ms": med([r["tts_first_chunk_ms"] for r in rows if r["tts_first_chunk_ms"]]),
        "vad_endpoint_silence_s": cfg["audio"]["silence_ms_to_end"] / 1000,
        "rows": rows,
    })
    rec.save()
    return rec


if __name__ == "__main__":
    r = run()
    print(f"e2e: {r.passed}/{len(r.checks)}")
    print({k: v for k, v in r.summary.items() if k != "rows"})
    sys.exit(0)
