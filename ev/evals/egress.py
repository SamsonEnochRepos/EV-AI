"""Network egress audit: what does the real EV process tree connect to / listen on?"""
from __future__ import annotations

import ipaddress
import json
import subprocess  # noqa: S404
import sys
import threading
import time

import psutil

from ev.core.config import ROOT, load_config
from ev.overlay.client import kill_tree

from .common import RESULTS, Recorder


def _is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip.split("%")[0]).is_loopback
    except ValueError:
        return False


def _tree(pid: int) -> list[psutil.Process]:
    root = psutil.Process(pid)
    return [root, *root.children(recursive=True)]


def _snapshot(pids: set[int]) -> list[dict]:
    out = []
    for c in psutil.net_connections(kind="inet"):
        if c.pid in pids:
            out.append({"pid": c.pid, "status": c.status, "laddr": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "",
                        "raddr": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else ""})
    return out


def run() -> Recorder:
    rec = Recorder("egress")
    proc = subprocess.Popen([sys.executable, "-m", "ev"], cwd=str(ROOT), stdout=subprocess.DEVNULL,  # noqa: S603
                            stderr=subprocess.DEVNULL)
    seen: list[dict] = []
    try:
        time.sleep(14)                                  # startup + warm-up (Ollama chat, Whisper load)
        t_end = time.time() + 25
        while time.time() < t_end:
            pids = {p.pid for p in _tree(proc.pid)}
            seen += _snapshot(pids)
            time.sleep(0.5)
        remote = {(s["raddr"]) for s in seen if s["raddr"]}
        non_loop = sorted(r for r in remote if not _is_loopback(r.rsplit(":", 1)[0]))
        listeners = sorted({s["laddr"] for s in seen if s["status"] == "LISTEN"})
        bad_listen = [a for a in listeners if not _is_loopback(a.rsplit(":", 1)[0])]
        rec.add("egress:idle_no_non_loopback_connections", not non_loop, "critical", f"remote={non_loop}")
        rec.add("egress:listeners_loopback_only", not bad_listen, "high", f"listeners={listeners}")
        rec.summary.update({"idle_remote_endpoints": sorted(remote), "listeners": listeners})
    finally:
        kill_tree(proc.pid)

    # The one permitted egress: web_search -> DuckDuckGo. Measure exactly which remotes it touches.
    from ev.memory.store import MemoryStore
    from ev.safety.policy import Policy
    from ev.tools.builtin import ToolContext, build_tools
    from ev.tools.scheduler import Scheduler

    cfg = load_config()
    mem = MemoryStore(":memory:")
    reg = build_tools(ToolContext(cfg, mem, Scheduler(mem, lambda _m: None), Policy.load(cfg.path("general", "policy_file"))))
    me = psutil.Process()
    stop = threading.Event()
    remotes: set[str] = set()

    def watch() -> None:
        while not stop.is_set():
            for c in psutil.net_connections(kind="inet"):
                if c.pid == me.pid and c.raddr and not _is_loopback(c.raddr.ip):
                    remotes.add(f"{c.raddr.ip}:{c.raddr.port}")
            time.sleep(0.05)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    tool = reg.get("web_search")
    assert tool
    res = reg.run(tool, {"query": "height of mount everest"})
    time.sleep(0.5)
    stop.set()
    th.join(1)
    mem.close()
    rec.add("egress:web_search_works_or_fails_gracefully", isinstance(res.ok, bool), "info", res.message[:80].replace("\n", " "))
    rec.summary["web_search_remote_endpoints"] = sorted(remotes)
    rec.summary["web_search_ok"] = res.ok
    rec.add("egress:web_search_only_remote_traffic_recorded", True, "info", f"{len(remotes)} remote endpoints: {sorted(remotes)[:6]}")
    rec.save()
    return rec


if __name__ == "__main__":
    r = run()
    print(f"egress: {r.passed}/{len(r.checks)}")
    print(json.dumps(r.summary, indent=1)[:1500])
    _ = RESULTS
