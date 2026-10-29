"""Voice barge-in: notice the user talking over EV and hand the mic back.

The mic is only open while EV speaks, and nothing is stored. On speakers EV's own voice leaks into the mic, so
the detector learns how strongly the playback echoes (input level / output level) during the first moments of a
reply and only fires when the user is clearly louder than that expected echo. With headphones the echo is ~0 and
it reacts to any real speech.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

from .vad import rms

log = get_logger("interrupt")


class EchoDetector:
    """feed() one mic block at a time; returns True once the user has talked over EV long enough."""

    OUT_MIN = 0.05            # playback level (0..1) needed for an echo-ratio sample to count
    LAG_BLOCKS = 12           # ~360 ms: the output meter runs ahead of the sound reaching the mic, and tails linger

    def __init__(self, block_ms: float = 30, min_ms: float = 300, margin: float = 2.0, floor: float = 0.015,
                 calib_blocks: int = 20):
        self.need = max(1, math.ceil(min_ms / block_ms))
        self.margin = margin
        self.floor = floor
        self.calib_blocks = calib_blocks
        self.ratio: float | None = None
        self._samples: list[float] = []
        self._recent_out: deque[float] = deque(maxlen=self.LAG_BLOCKS)
        self._run = 0

    def feed(self, in_rms: float, speech: bool, out_level: float) -> bool:
        self._recent_out.append(out_level)
        out_ref = max(self._recent_out)              # what the mic could be hearing right now, latency included
        if self.ratio is None:                       # learn the echo while EV's own voice is playing
            if out_ref > self.OUT_MIN:
                self._samples.append(in_rms / out_ref)
                if len(self._samples) >= self.calib_blocks:
                    # Upper quartile, not median: the first blocks are measured before the echo has reached the mic.
                    self.ratio = float(np.percentile(self._samples, 75))
            return False
        expected_echo = self.ratio * out_ref
        loud_enough = in_rms > max(self.floor, self.margin * expected_echo + self.floor / 2)
        self._run = self._run + 1 if (speech and loud_enough) else 0
        return self._run >= self.need


class InterruptMonitor:
    """Background mic listener that runs while EV speaks. Calls on_interrupt(recent_blocks) at most once."""

    PREROLL_BLOCKS = 20       # ~600 ms handed to the recorder so the start of the user's sentence isn't lost

    def __init__(self, cfg: Config, vad: Any, out_level: Callable[[], float],
                 on_interrupt: Callable[[list[np.ndarray]], None],
                 blocks: Callable[[Config, Callable[[], bool]], Iterator[np.ndarray]]):
        conv = cfg.raw.get("conversation", {})
        self.detector = EchoDetector(
            block_ms=float(cfg["audio"]["block_ms"]), min_ms=float(conv.get("interrupt_min_ms", 300)),
            margin=float(conv.get("interrupt_margin", 2.0)), floor=float(conv.get("interrupt_floor", 0.015)))
        self.cfg = cfg
        self.vad = vad
        self.out_level = out_level
        self.on_interrupt = on_interrupt
        self._blocks = blocks
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="interrupt-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop and wait until the mic is released, so the recorder can open it."""
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(2.0)

    def _run(self) -> None:
        ring: deque[np.ndarray] = deque(maxlen=self.PREROLL_BLOCKS)
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        gen = self._blocks(self.cfg, self._stop.is_set)
        try:
            for blk in gen:
                ring.append(blk)
                if self.detector.feed(rms(blk), self.vad.is_speech(blk), self.out_level()):
                    log.info("user talked over EV (echo ratio %.2f)", self.detector.ratio or 0.0)
                    self._stop.set()
                    self.on_interrupt(list(ring))
                    return
        except Exception as e:                       # no mic, device unplugged: EV just can't be interrupted by voice
            log.warning("voice interruption unavailable: %s", e)
        finally:
            getattr(gen, "close", lambda: None)()
