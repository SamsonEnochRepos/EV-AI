"""Bar animation math, independent of any GUI toolkit (so it is unit-testable)."""
from __future__ import annotations

import math

VISIBLE_STATES = {"listening", "transcribing", "thinking", "acting", "speaking", "paused", "muted"}
ACTIVE_LEVEL_STATES = {"listening", "speaking"}
CALM_STATES = {"thinking", "transcribing", "acting"}


def _noise(i: int, t: float) -> float:
    """Cheap deterministic pseudo-noise in 0..1, smooth in t."""
    return 0.5 + 0.5 * math.sin(t * (5.0 + (i * 1.7) % 3.1) + i * 2.399)


class BarAnimator:
    def __init__(self, bars: int = 11, smoothing: float = 0.55):
        self.n = bars
        self.smooth = smoothing
        self.h = [0.1] * bars
        self.level = 0.0          # smoothed input level
        self.state = "idle"

    def envelope(self, i: int) -> float:
        c = (self.n - 1) / 2
        return 0.45 + 0.55 * (1 - abs(i - c) / (c + 0.001))

    def target(self, i: int, t: float, level: float) -> float:
        s = self.state
        if s in ACTIVE_LEVEL_STATES:
            amp = min(1.0, level)
            return 0.12 + 0.88 * amp * self.envelope(i) * (0.55 + 0.45 * _noise(i, t))
        if s in CALM_STATES:
            wave = 0.5 + 0.5 * math.sin(2 * math.pi * (t * 0.9 - i / self.n))
            return 0.14 + 0.2 * wave * self.envelope(i)
        if s == "paused" or s == "muted":
            return 0.08
        return 0.1

    def step(self, t: float, level: float = 0.0) -> list[float]:
        # fast attack, slower release on the level so bars feel alive but not jittery
        k = 0.6 if level > self.level else 0.25
        self.level += (level - self.level) * k
        for i in range(self.n):
            tgt = self.target(i, t, self.level)
            self.h[i] += (tgt - self.h[i]) * (1 - self.smooth)
        return list(self.h)


def demo_schedule() -> list[tuple[str, float]]:
    """(state, seconds) cycle used by --demo."""
    return [("listening", 4.0), ("transcribing", 1.0), ("thinking", 2.5), ("acting", 2.5),
            ("speaking", 4.0), ("paused", 2.0), ("idle", 2.0)]


def demo_level(state: str, t: float) -> float:
    """Synthetic audio level for the demo."""
    if state not in ACTIVE_LEVEL_STATES:
        return 0.0
    syllable = max(0.0, math.sin(t * 7.0)) ** 2
    phrase = 0.5 + 0.5 * math.sin(t * 1.3)
    return min(1.0, 0.15 + 0.85 * syllable * phrase)
