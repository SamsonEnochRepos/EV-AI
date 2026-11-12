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
    """feed() one mic block at a time; returns True once the user has talked over EV long enough.

    Measured on a laptop: EV's voice reaches its own mic ~0.5 s after the playback meter says it is playing, and the
    echo is as loud as normal speech (mic rms 0.02-0.05 against 0.05-0.1 for a person). Comparing "mic now" with
    "playback now" therefore mistakes EV's own voice for the user. This detector instead
      * learns the speaker-to-mic delay by correlating the playback and mic envelopes (devices differ widely),
      * predicts the echo from what was playing that long ago, scaled by a learned echo gain,
      * stays disarmed until it has heard the echo (a person rarely interrupts in the first second), and
      * stops learning while a possible interruption is in progress, so the user's voice is never taken for echo.
    """

    OUT_MIN = 0.05            # playback level (0..1) that counts as EV speaking
    GAIN_MIN_OUT = 0.15       # echo-gain samples are only taken when the (delayed) playback is clearly audible
    HISTORY = 64              # blocks of playback/mic envelope kept for the delay estimate (~1.9 s)
    MAX_LAG = 40              # longest speaker-to-mic delay considered (~1.2 s, Bluetooth included)
    DEFAULT_LAG = 16          # ~480 ms until the first estimate
    MIN_GAIN_SAMPLES = 12

    QUIET_EXTRA = 8           # blocks of silence from EV (beyond the echo delay) before the room counts as quiet

    def __init__(self, block_ms: float = 30, min_ms: float = 300, margin: float = 2.0, floor: float = 0.015,
                 arm_ms: float = 1200, gain_quantile: float = 75, quiet_fires: bool = False):
        # quiet_fires: while EV itself has made no sound for longer than the echo takes to arrive (it is still thinking, or
        # between replies) there is no echo to model, so plain speech above the floor counts as the user talking.
        self.quiet_fires = quiet_fires
        self.need = max(1, math.ceil(min_ms / block_ms))
        self.margin = margin
        self.floor = floor
        self.arm_blocks = max(1, math.ceil(arm_ms / block_ms))
        self.gain_q = gain_quantile
        self.block_ms = block_ms
        self.lag = self.DEFAULT_LAG
        self.ratio: float | None = None                 # learned echo gain: mic rms per unit of (delayed) playback level
        self._out: deque[float] = deque(maxlen=self.HISTORY)
        self._in: deque[float] = deque(maxlen=self.HISTORY)
        self._gains: deque[float] = deque(maxlen=100)
        self._played = 0
        self._since_est = 0
        self._run = 0

    @property
    def lag_ms(self) -> float:
        return self.lag * self.block_ms

    def _estimate_lag(self) -> None:
        o, i = np.asarray(self._out), np.asarray(self._in)
        k = np.ones(3) / 3
        o, i = np.convolve(o, k, "same"), np.convolve(i, k, "same")
        if o.std() < 1e-3 or i.std() < 1e-4:
            return
        best, best_c = self.lag, 0.3                    # need a real correlation before moving off the current delay
        for lag in range(self.MAX_LAG + 1):
            n = len(o) - lag
            if n < 24:
                break
            c = float(np.corrcoef(o[:n], i[lag:lag + n])[0, 1])
            if c > best_c:
                best, best_c = lag, c
        self.lag = best

    def rearm(self) -> None:
        """Forget the speech that just fired (it was not for EV), so the next interruption needs a fresh run of speech."""
        self._run = 0

    def feed(self, in_rms: float, speech: bool, out_level: float) -> bool:
        self._out.append(out_level)
        self._in.append(in_rms)
        if out_level > self.OUT_MIN:
            self._played += 1
        self._since_est += 1
        if self._played >= 30 and self._since_est >= 10:
            self._since_est = 0
            self._estimate_lag()
        if self.quiet_fires and max(list(self._out)[-(self.lag + self.QUIET_EXTRA):]) <= self.OUT_MIN:
            self._run = self._run + 1 if (speech and in_rms > self.floor) else 0
            return self._run >= self.need
        out_d = self._out[-1 - self.lag] if len(self._out) > self.lag else 0.0     # playback as the mic hears it now
        gain =float(np.percentile(self._gains, self.gain_q)) if len(self._gains) >= self.MIN_GAIN_SAMPLES else None
        self.ratio = gain
        if self._played < self.arm_blocks or gain is None:
            if out_d > self.GAIN_MIN_OUT:
                self._gains.append(in_rms / out_d)
            return False
        loud_enough = in_rms > max(self.floor, self.margin * gain * out_d + self.floor / 2)
        suspect = speech and loud_enough
        self._run = self._run + 1 if suspect else 0
        if not suspect and out_d > self.GAIN_MIN_OUT:   # keep tracking the echo, but never learn the user's voice
            self._gains.append(in_rms / out_d)
        return self._run >= self.need


class InterruptMonitor:
    """Background mic listener that runs while EV speaks. Calls on_interrupt(recent_blocks) at most once.

    With `verify` set (conversation.interrupt_needs_name), hearing someone talk is not enough: the start of what they say is
    collected and handed to verify(), which says whether it was addressed to EV ("EV, don't open it..."). Anything else (a
    side conversation, the TV, a murmur) is ignored and EV carries on."""

    PREROLL_BLOCKS = 20       # ~600 ms handed to the recorder so the start of the user's sentence isn't lost
    SETTLE_MS = 400           # this much silence ends the collected speech early

    def __init__(self, cfg: Config, vad: Any, out_level: Callable[[], float],
                 on_interrupt: Callable[[list[np.ndarray]], None],
                 blocks: Callable[[Config, Callable[[], bool]], Iterator[np.ndarray]]):
        conv = cfg.raw.get("conversation", {})
        self.detector = EchoDetector(
            block_ms=float(cfg["audio"]["block_ms"]), min_ms=float(conv.get("interrupt_min_ms", 300)),
            margin=float(conv.get("interrupt_margin", 2.0)), floor=float(conv.get("interrupt_floor", 0.015)),
            quiet_fires=bool(conv.get("listen_while_thinking", False)))
        self.cfg = cfg
        self.vad = vad
        self.out_level = out_level
        self.on_interrupt = on_interrupt
        self._blocks = blocks
        self.verify: Callable[[list[np.ndarray]], bool] | None = None
        self.listen_blocks = max(1, math.ceil(float(conv.get("interrupt_listen_s", 1.4)) * 1000 / float(cfg["audio"]["block_ms"])))
        self.settle_blocks = max(1, math.ceil(self.SETTLE_MS / float(cfg["audio"]["block_ms"])))
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

    def _collect(self, gen: Any, heard: list[np.ndarray]) -> list[np.ndarray]:
        """The start of what the user is saying: up to listen_s of it, or until they pause."""
        silent = 0
        for blk in gen:
            heard.append(blk)
            silent = 0 if self.vad.is_speech(blk) else silent + 1
            if silent >= self.settle_blocks or len(heard) >= self.listen_blocks + self.PREROLL_BLOCKS:
                break
        return heard

    def _run(self) -> None:
        ring: deque[np.ndarray] = deque(maxlen=self.PREROLL_BLOCKS)
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        gen = self._blocks(self.cfg, self._stop.is_set)
        try:
            for blk in gen:
                ring.append(blk)
                if self.detector.feed(rms(blk), self.vad.is_speech(blk), self.out_level()):
                    heard = list(ring)
                    if self.verify is not None:
                        heard = self._collect(gen, heard)
                        if not self.verify(heard):
                            self.detector.rearm()                 # not for EV: keep talking, keep listening
                            ring.clear()
                            continue
                    log.info("user talked over EV (echo gain %.2f, delay %d ms)", self.detector.ratio or 0.0, self.detector.lag_ms)
                    self._stop.set()
                    resume = getattr(gen, "resume_from", None)       # where on a shared microphone the recorder picks up
                    if resume is None:
                        self.on_interrupt(heard)
                    else:
                        self.on_interrupt(heard, resume)             # type: ignore[call-arg]
                    return
        except Exception as e:                       # no mic, device unplugged: EV just can't be interrupted by voice
            log.warning("voice interruption unavailable: %s", e)
        finally:
            getattr(gen, "close", lambda: None)()
