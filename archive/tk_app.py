"""Tkinter pill, kept for the toolkit benchmark (docs/DECISIONS.md D5). Demo-only.

Tk has no per-pixel alpha: rounded edges use a colour-key transparency, which leaves jagged
(aliased) edges, so this is not used in production.
"""
from __future__ import annotations

import time
import tkinter as tk

from ev.core.config import Config

from .anim import BarAnimator, demo_level, demo_schedule

KEY = "#010203"


def run_demo(cfg: Config) -> int:
    o = cfg["overlay"]
    w, h = o["width"], o["height"]
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True, "-transparentcolor", KEY)
    sw = root.winfo_screenwidth()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{o['offset_y']}")
    cv = tk.Canvas(root, width=w, height=h, bg=KEY, highlightthickness=0)
    cv.pack()
    anim = BarAnimator(o["bar_count"])
    sched = demo_schedule()
    idx = [0]
    last = [time.monotonic()]
    t0 = time.monotonic()

    def draw() -> None:
        now = time.monotonic()
        name, dur = sched[idx[0] % len(sched)]
        if now - last[0] >= dur:
            idx[0] += 1
            last[0] = now
            name, dur = sched[idx[0] % len(sched)]
        anim.state = name
        hs = anim.step(now - t0, demo_level(name, now - t0))
        cv.delete("all")
        r = h // 2
        cv.create_oval(0, 0, h, h, fill="#111111", outline="#3a3a3a")
        cv.create_oval(w - h, 0, w, h, fill="#111111", outline="#3a3a3a")
        cv.create_rectangle(r, 0, w - r, h, fill="#111111", outline="")
        total = anim.n * 6
        x0 = (w - total) // 2
        for i, hh in enumerate(hs):
            bh = max(3, hh * (h - 12))
            x = x0 + i * 6
            cv.create_rectangle(x, (h - bh) / 2, x + 3, (h + bh) / 2, fill="white", outline="")
        root.after(int(1000 / o["fps"]), draw)

    draw()
    root.mainloop()
    return 0


def run_server(cfg: Config) -> int:   # pragma: no cover
    raise NotImplementedError("Tk pill is benchmark-only; use toolkit = 'qt'")
