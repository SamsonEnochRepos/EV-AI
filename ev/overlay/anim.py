"""Island animation math (level bars, shape springs), independent of any GUI toolkit so it is unit-testable."""
from __future__ import annotations

import math
from dataclasses import dataclass

VISIBLE_STATES = {"listening", "transcribing", "thinking", "acting", "speaking", "paused", "muted", "error"}
ACTIVE_LEVEL_STATES = {"listening", "speaking"}
CALM_STATES = {"thinking", "transcribing", "acting"}
WIDE_STATES = {"listening", "speaking"}              # the island opens to its full width
MID_STATES = {"transcribing", "thinking", "acting", "error"}   # half way: it is busy, not talking
ASLEEP_STATES = {"paused", "muted"}

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def approach(cur: float, target: float, dt: float, rate: float) -> float:
    """Exponential ease of `cur` toward `target`; frame-rate independent."""
    return target + (cur - target) * math.exp(-max(0.0, rate) * max(0.0, dt))


class Spring:
    """A slightly bouncy spring (ease with a little overshoot), so the island feels like it has weight."""

    def __init__(self, value: float, stiffness: float = 260.0, damping_ratio: float = 0.72):
        self.x = value
        self.v = 0.0
        self.k = stiffness
        self.zeta = damping_ratio

    def step(self, target: float, dt: float) -> float:
        dt = clamp(dt, 0.0, 0.1)
        c = 2 * self.zeta * math.sqrt(self.k)
        n = int(dt * 240) + 1                      # sub-steps keep a long frame from blowing the integration up
        h = dt / n
        for _ in range(n):
            self.v += (self.k * (target - self.x) - c * self.v) * h
            self.x += self.v * h
        return self.x

    def settled(self, target: float, eps: float = 0.25) -> bool:
        return abs(target - self.x) < eps and abs(self.v) < eps * 4


@dataclass(frozen=True)
class IslandSizes:
    idle_w: float = 116.0
    idle_h: float = 34.0
    wide_w: float = 236.0
    wide_h: float = 42.0
    card_w: float = 460.0       # width when display mode has text to show
    strip_h: float = 42.0       # the bar strip at the top of a card-sized island
    card_radius: float = 22.0


def island_shape(state: str, s: IslandSizes, card_h: float = 0.0) -> tuple[float, float]:
    """The (width, height) the island is heading for. Text (display mode / a notice) takes it over from the state."""
    if card_h > 0:
        return s.card_w, s.strip_h + card_h
    if state in WIDE_STATES:
        return s.wide_w, s.wide_h
    if state in MID_STATES:
        return (s.idle_w + s.wide_w) / 2, (s.idle_h + s.wide_h) / 2
    return s.idle_w, s.idle_h


def expansion(w: float, s: IslandSizes) -> float:
    """0 = resting pill, 1 = fully open. Things that only belong on an open island (the bars) fade with this."""
    return clamp((w - s.idle_w) / max(1.0, s.wide_w - s.idle_w))


def corner_radius(w: float, h: float, s: IslandSizes) -> float:
    """A full pill at rest, a rounded rectangle when open, never rounder than a card."""
    grow = clamp((h - s.idle_h) / max(1.0, s.wide_h - s.idle_h))
    return min(h * (0.5 - 0.08 * grow), s.card_radius, w / 2)


def place_window(anchor: str, sw: int, sh: int, w: int, h: int, ox: int, oy: int, margin: int) -> tuple[int, int]:
    """Top-left of the (margin-padded) window so that the island body sits `ox`/`oy` from the anchored screen edge."""
    v, _, hz = anchor.partition("-")
    x = {"left": ox - margin, "right": sw - w + margin - ox}.get(hz, (sw - w) // 2 + ox)
    y = oy - margin if v == "top" else sh - h + margin - oy if v == "bottom" else (sh - h) // 2 + oy
    return x, y


def body_origin(horizontal: str, vertical: str, win_w: int, win_h: int, w: float, h: float, margin: int) -> tuple[float, float]:
    """Top-left of the island body inside its window: pinned to the anchored edge, growing away from it."""
    x = {"left": float(margin), "right": win_w - margin - w}.get(horizontal, (win_w - w) / 2)
    y = win_h - margin - h if vertical == "bottom" else float(margin)
    return x, y


def alert_target(state: str) -> float:
    """1 while EV is working on the screen or something went wrong (the island shows its accent colour and a soft ring), else 0."""
    return 1.0 if state in ("acting", "error") else 0.0


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = clamp(t)
    return (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t), round(a[2] + (b[2] - a[2]) * t))


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
