"""Launch the real EV process, measure startup time and idle CPU / RAM / VRAM, then stop it."""
from __future__ import annotations

import json
import statistics
import subprocess  # noqa: S404
import sys
import time

import psutil

from ev.core.config import ROOT
from ev.overlay.client import kill_tree

from .bench import vram_mb
from .common import RESULTS


def main(seconds: float = 45.0) -> int:
    log = ROOT / "logs" / "ev.log"
    start_size = log.stat().st_size if log.exists() else 0
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "ev"], cwd=str(ROOT), stdout=subprocess.DEVNULL,  # noqa: S603
                            stderr=subprocess.DEVNULL)
    ready_s = None
    warm_s = None
    while time.perf_counter() - t0 < 120:
        time.sleep(0.25)
        if log.exists():
            with open(log, encoding="utf-8", errors="replace") as f:
                f.seek(start_size)
                txt = f.read()
            if ready_s is None and "EV ready" in txt:
                ready_s = time.perf_counter() - t0
            if warm_s is None and "whisper base loaded" in txt:
                warm_s = time.perf_counter() - t0
            if ready_s and warm_s and time.perf_counter() - t0 > max(ready_s, warm_s) + 8:
                break
        if proc.poll() is not None:
            break
    alive = proc.poll() is None
    cpu: list[float] = []
    rss: list[float] = []
    sample_error = ""
    if alive:
        root = psutil.Process(proc.pid)
        tree = [root, *root.children(recursive=True)]
        for p in tree:
            p.cpu_percent(None)
        t_end = time.time() + seconds
        while time.time() < t_end:
            time.sleep(1.0)
            try:
                cpu.append(sum(p.cpu_percent(None) for p in tree) / psutil.cpu_count())
                rss.append(sum(p.memory_info().rss for p in tree) / 1e6)
            except psutil.Error as e:
                sample_error = str(e)
                break
        procs = [{"pid": p.pid, "rss_mb": round(p.memory_info().rss / 1e6, 1), "cmd": " ".join(p.cmdline()[-3:])}
                 for p in tree if p.is_running()]
    else:
        procs = []
    vr = vram_mb()
    tail = ""
    if log.exists():
        with open(log, encoding="utf-8", errors="replace") as f:
            f.seek(start_size)
            tail = f.read()[-1500:]
    kill_tree(proc.pid)
    out = {"alive": alive, "ready_s": ready_s, "stt_warm_s": warm_s,
           "idle_cpu_pct_mean": round(statistics.mean(cpu), 2) if cpu else None,
           "idle_cpu_pct_max": round(max(cpu), 2) if cpu else None,
           "idle_rss_mb_mean": round(statistics.mean(rss), 1) if rss else None,
           "idle_rss_mb_max": round(max(rss), 1) if rss else None,
           "vram_mb_total_used": vr, "processes": procs, "sample_error": sample_error, "log_tail": tail}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "idle.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print({k: v for k, v in out.items() if k != "log_tail"})
    print(tail[-700:])
    return 0 if alive else 1


if __name__ == "__main__":
    sys.exit(main())
