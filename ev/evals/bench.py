"""Benchmarks: LLM candidates (speed + VRAM), overlay toolkits (CPU/RAM). Writes evals/results/*.json."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess  # noqa: S404
import sys
import threading
import time
from typing import Any

import httpx

from ev.core.config import ROOT

from .common import RESULTS

PROMPTS = [
    "Tell me something interesting about octopuses.",
    "What should I make for dinner tonight? I have eggs, rice and spinach.",
    "Explain in two sentences why the sky is blue.",
]


def vram_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],  # noqa: S603, S607
                             capture_output=True, text=True, timeout=5, check=False).stdout.strip().splitlines()
        return int(out[0])
    except Exception:      # noqa: BLE001
        return None


def sample_vram(stop: threading.Event, out: list[int]) -> None:
    while not stop.is_set():
        v = vram_mb()
        if v is not None:
            out.append(v)
        stop.wait(0.25)


def ollama_ps() -> list[dict[str, Any]]:
    try:
        return httpx.get("http://127.0.0.1:11434/api/ps", timeout=5).json().get("models", [])
    except Exception:      # noqa: BLE001
        return []


def bench_llm(model: str, num_ctx: int = 2048, persona: str | None = None) -> dict[str, Any]:
    persona = persona or (ROOT / "config" / "persona.md").read_text(encoding="utf-8")
    httpx.post("http://127.0.0.1:11434/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(1.5)
    base = vram_mb()
    samples: list[int] = []
    stop = threading.Event()
    th = threading.Thread(target=sample_vram, args=(stop, samples), daemon=True)
    th.start()
    runs = []
    load_s = None
    for i, p in enumerate(PROMPTS):
        body = {"model": model, "stream": False, "think": False, "keep_alive": "5m",
                "options": {"num_ctx": num_ctx, "temperature": 0.6, "num_predict": 120},
                "messages": [{"role": "system", "content": persona}, {"role": "user", "content": p}]}
        t0 = time.perf_counter()
        r = httpx.post("http://127.0.0.1:11434/api/chat", json=body, timeout=300).json()
        wall = time.perf_counter() - t0
        if i == 0:
            load_s = r.get("load_duration", 0) / 1e9
        ev = r.get("eval_count", 0) / max(r.get("eval_duration", 1) / 1e9, 1e-6)
        pe = r.get("prompt_eval_count", 0) / max(r.get("prompt_eval_duration", 1) / 1e9, 1e-6)
        runs.append({"tok_s": ev, "prompt_tok_s": pe, "wall_s": wall, "tokens": r.get("eval_count", 0),
                     "ttft_est_s": (r.get("load_duration", 0) + r.get("prompt_eval_duration", 0)) / 1e9 if i else
                     r.get("prompt_eval_duration", 0) / 1e9})
    ps = [m for m in ollama_ps() if m.get("name", "").startswith(model.split(":")[0])]
    stop.set()
    th.join(1)
    size = ps[0].get("size", 0) if ps else 0
    size_vram = ps[0].get("size_vram", 0) if ps else 0
    return {
        "model": model, "num_ctx": num_ctx, "load_s": load_s,
        "tok_s_median": statistics.median(r["tok_s"] for r in runs),
        "prompt_tok_s_median": statistics.median(r["prompt_tok_s"] for r in runs),
        "ttft_s_median": statistics.median(r["ttft_est_s"] for r in runs),
        "peak_vram_mb": max(samples) if samples else None, "baseline_vram_mb": base,
        "model_bytes": size, "model_bytes_in_vram": size_vram,
        "spill_to_ram_pct": round(100 * (1 - size_vram / size), 1) if size else None,
        "runs": runs,
    }


def bench_overlay(toolkit: str, seconds: float = 12.0) -> dict[str, Any]:
    import psutil

    env_cfg = f'import ev.core.config as c; cfg=c.load_config(overrides={{"overlay":{{"toolkit":"{toolkit}"}}}}); '
    code = env_cfg + (
        "from ev.overlay.qt_app import run_demo; run_demo(cfg)" if toolkit == "qt"
        else "from ev.overlay.tk_app import run_demo; run_demo(cfg)")
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
    time.sleep(3)
    root = psutil.Process(proc.pid)
    tree = [root, *root.children(recursive=True)]     # venv python.exe is a launcher shim
    for p in tree:
        p.cpu_percent(None)
    cpu, rss = [], []
    t_end = time.time() + seconds
    while time.time() < t_end:
        time.sleep(0.5)
        cpu.append(sum(p.cpu_percent(None) for p in tree) / psutil.cpu_count())
        rss.append(sum(p.memory_info().rss for p in tree) / 1e6)
    for p in reversed(tree):
        p.kill()
    return {"toolkit": toolkit, "cpu_pct_mean": statistics.mean(cpu), "cpu_pct_max": max(cpu),
            "rss_mb_mean": statistics.mean(rss), "rss_mb_max": max(rss), "seconds": seconds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["llm", "overlay"])
    ap.add_argument("--models", nargs="*", default=["qwen2.5:3b", "llama3.2:latest", "qwen3:4b"])
    ap.add_argument("--ctx", nargs="*", type=int, default=[2048])
    ap.add_argument("--toolkits", nargs="*", default=["qt", "tk"])
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if a.what == "llm":
        res = [bench_llm(m, c) for m in a.models for c in a.ctx]
        out = RESULTS / "bench_llm.json"
    else:
        res = [bench_overlay(t) for t in a.toolkits]
        out = RESULTS / "bench_overlay.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for r in res:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "runs"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
