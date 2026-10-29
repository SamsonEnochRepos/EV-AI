"""Overlay evals: renders each state to PNG, window-style/focus checks, idle hiding, level response, resilience."""
from __future__ import annotations

import ctypes
import os
import subprocess  # noqa: S404
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

from ev.core.config import ROOT, load_config
from ev.overlay import protocol
from ev.overlay.client import OverlayClient

from .common import RESULTS, Recorder

OUT = RESULTS / "overlay"
user32 = ctypes.windll.user32
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x20, 0x80, 0x08000000


def render_snapshots(rec: Recorder) -> None:
    """Offscreen render of every state with synthetic bars. Needs no display focus."""
    code = r"""
import os, sys, time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from ev.core.config import load_config
from ev.overlay.qt_app import Pill
from pathlib import Path
from typing import Any
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
app = QApplication([])
cfg = load_config()
p = Pill(cfg)
scale = 4
for state, level in [('idle',0),('listening',0.15),('listening',0.9),('transcribing',0),('thinking',0),('acting',0),('speaking',0.8),('paused',0)]:
    p.state = state; p.anim.state = state; p.opacity = 1.0; p.setWindowOpacity(1.0)
    p.set_level(level)
    for i in range(90): p.anim.step(i/30, level)
    pm = p.grab()
    name = f"{state}_{int(level*100):02d}.png"
    pm.save(str(out / name))
print('ok')
"""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code, str(OUT)], capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=120, check=False)  # noqa: S603
    rec.add("snapshots_rendered", r.returncode == 0 and "ok" in r.stdout, "info", (r.stderr or r.stdout)[-160:])


def analyse_snapshots(rec: Recorder) -> None:
    from PIL import Image

    cfg = load_config()
    w, h = cfg["overlay"]["width"], cfg["overlay"]["height"]
    stats: dict[str, dict[str, Any]] = {}
    for p in sorted(OUT.glob("*.png")):
        im = Image.open(p).convert("RGBA")
        px: Any = im.load()
        W, H = im.size
        white = sum(1 for x in range(W) for y in range(H) if px[x, y][3] > 200 and min(px[x, y][:3]) > 220)
        dark = sum(1 for x in range(W) for y in range(H) if px[x, y][3] > 150 and max(px[x, y][:3]) < 40)
        corner = px[0, 0][3]
        stats[p.stem] = {"size": (W, H), "white_px": white, "dark_px": dark, "corner_alpha": corner}
    rec.summary["snapshot_stats"] = stats
    if not stats:
        rec.add("snapshot_stats", False, "high", "no snapshots")
        return
    any_size = next(iter(stats.values()))["size"]
    rec.add("pill_size_matches_config", abs(any_size[0] / any_size[1] - w / h) < 0.05, "medium", f"{any_size} vs {w}x{h}")
    rec.add("pill_corners_transparent", all(s["corner_alpha"] < 30 for s in stats.values()), "medium", "fully rounded ends")
    rec.add("pill_fill_dark", all(s["dark_px"] > 0.4 * s["size"][0] * s["size"][1] * 0.6 for s in stats.values()), "medium")
    lo = stats.get("listening_15", {}).get("white_px", 0)
    hi = stats.get("listening_90", {}).get("white_px", 0)
    rec.add("bars_grow_with_mic_level", hi > lo * 1.5, "high", f"white px {lo} -> {hi}")
    sp = stats.get("speaking_80", {}).get("white_px", 0)
    rec.add("bars_grow_with_speaker_level", sp > stats.get("thinking_00", {}).get("white_px", 0) * 1.3, "high", f"speaking={sp}")
    pa = stats.get("paused_00", {}).get("white_px", 1)
    rec.add("paused_is_dimmed_and_flat", pa < lo, "medium", f"paused white px={pa}")
    rec.add("all_states_render", {"listening_15", "thinking_00", "acting_00", "speaking_80", "paused_00"} <= set(stats), "high")


def _tree_pids(pid: int) -> set[int]:
    import psutil

    try:
        return {pid, *(c.pid for c in psutil.Process(pid).children(recursive=True))}
    except psutil.Error:
        return {pid}


def _find_windows(pid: int) -> list[int]:
    found: list[int] = []
    pids = _tree_pids(pid)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd: int, _l: int) -> bool:
        p = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value in pids:
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found


def live_window_checks(rec: Recorder) -> None:
    """Real overlay process via the real client: styles, focus, idle hiding, resilience."""
    cfg = load_config(overrides={"overlay": {"hide_delay_ms": 300, "fade_ms": 100}})
    cl = OverlayClient(cfg, restart_cooldown_s=0.2)
    fg_before = user32.GetForegroundWindow()
    if not cl.start() or cl.proc is None:
        rec.add("live:overlay_starts", False, "high")
        return
    rec.add("live:overlay_starts", True, "high")
    pid = cl.proc.pid
    cl.send_state("listening")
    time.sleep(1.5)
    wins = [h for h in _find_windows(pid) if user32.IsWindowVisible(h)]
    pill = next((h for h in wins if user32.GetWindowRect(h, ctypes.byref(wintypes.RECT())) and
                 (lambda r: (user32.GetWindowRect(h, ctypes.byref(r)), r.right - r.left)[1] <= 200)(wintypes.RECT())), None)
    rec.add("live:visible_when_listening", pill is not None, "high", f"{len(wins)} visible windows")
    if pill:
        ex = user32.GetWindowLongW(pill, GWL_EXSTYLE) & 0xFFFFFFFF
        rec.add("live:click_through", bool(ex & WS_EX_TRANSPARENT), "high", hex(ex))
        rec.add("live:no_activate", bool(ex & WS_EX_NOACTIVATE), "high")
        rec.add("live:tool_window_no_taskbar", bool(ex & WS_EX_TOOLWINDOW), "medium")
        rec.add("live:topmost", bool(ex & 0x8), "medium", "WS_EX_TOPMOST")
        r = wintypes.RECT()
        user32.GetWindowRect(pill, ctypes.byref(r))
        sw = user32.GetSystemMetrics(0)
        rec.add("live:top_center_position", abs((r.left + r.right) / 2 - sw / 2) < 8 and r.top < 60, "medium", f"{r.left},{r.top},{r.right},{r.bottom} on {sw}px screen")
    rec.add("live:never_took_focus", user32.GetForegroundWindow() == fg_before, "high", "foreground window unchanged")
    for lv in (0.1, 0.9, 0.4):
        cl.send_level(lv)
        time.sleep(0.05)
    cl.send_state("idle")
    time.sleep(1.5)
    still = [h for h in _find_windows(pid) if user32.IsWindowVisible(h)]
    rec.add("live:hidden_when_idle", not still, "high", f"{len(still)} visible after idle")
    # resilience: overlay dies -> core keeps going and restarts it
    from ev.overlay.client import kill_tree
    kill_tree(cl.proc.pid)
    time.sleep(0.5)
    t0 = time.perf_counter()
    for _ in range(3):
        cl.send_state("listening")
        time.sleep(0.3)
    survived = (time.perf_counter() - t0) < 10
    rec.add("live:core_survives_overlay_crash", survived, "high")
    time.sleep(1.5)
    rec.add("live:overlay_restarted", cl.proc is not None and cl.proc.poll() is None, "medium")
    # core dies -> overlay exits
    pid2 = cl.proc.pid if cl.proc else 0
    if cl.sock:
        cl.sock.close()
        cl.sock = None
    time.sleep(2.5)
    alive = cl.proc is not None and cl.proc.poll() is None
    rec.add("live:overlay_exits_when_core_dies", not alive, "high", f"pid {pid2}")
    cl.stop()


def run() -> Recorder:
    rec = Recorder("overlay")
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (render_snapshots, analyse_snapshots, live_window_checks):
        try:
            fn(rec)
        except Exception as e:      # noqa: BLE001
            rec.add(f"{fn.__name__}:error", False, "high", f"{type(e).__name__}: {e}"[:160])
    rec.save()
    return rec


if __name__ == "__main__":
    r = run()
    print(f"overlay: {r.passed}/{len(r.checks)}")
    for c in r.checks:
        if not c.passed:
            print("  FAIL", c.name, c.detail)
    _ = (protocol, Path)
