"""Voice barge-in detector. Built around real levels recorded on a laptop whose speakers feed its own mic."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ev.audio.interrupt import EchoDetector
from ev.core.config import load_config

DATA = json.loads((Path(__file__).parent / "data" / "echo_laptop_speakers.json").read_text())
RUNS = {"run1": DATA["run1"], "run2": DATA["run2"]}


def shipped() -> EchoDetector:
    c = load_config().raw["conversation"]
    return EchoDetector(block_ms=30, min_ms=c["interrupt_min_ms"], margin=c["interrupt_margin"], floor=c["interrupt_floor"])


def feed_all(det, rows, user=None):
    """Feed recorded [mic rms, vad speech, playback] rows; `user=(start_s, rms)` mixes in someone talking for 1.5 s.
    Returns the second at which it fired, or None."""
    for n, (i, sp, o) in enumerate(rows):
        t = n * 0.03
        if user and user[0] <= t < user[0] + 1.5:
            i = math.sqrt(i * i + (user[1] * (0.6 + 0.4 * abs(math.sin(2 * math.pi * 4 * t)))) ** 2)
            sp = 1
        if det.feed(i, bool(sp), o):
            return t
    return None


@pytest.mark.parametrize("run", RUNS)
def test_ev_hearing_itself_never_interrupts(run):
    """The bug from the logs: EV's own voice, nobody talking, used to fire a false interrupt ('echo ratio 0.01')."""
    det = shipped()
    assert feed_all(det, RUNS[run]) is None


@pytest.mark.parametrize("run", RUNS)
def test_it_learns_the_speaker_to_mic_delay(run):
    det = shipped()
    feed_all(det, RUNS[run])
    assert 420 <= det.lag_ms <= 570                      # measured by hand on this laptop: ~480 ms


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("start", [1.6, 4.2])
def test_a_person_talking_over_the_echo_interrupts_quickly(run, start):
    fired = feed_all(shipped(), RUNS[run], user=(start, 0.12))
    assert fired is not None and start <= fired <= start + 1.0


def test_nothing_fires_before_the_echo_has_been_heard():
    det = shipped()
    loud_from_the_start = [(0.3, True, 0.6)] * 30        # 0.9 s of loud "speech" right as EV begins
    assert not any(det.feed(i, sp, o) for i, sp, o in loud_from_the_start)


def test_a_cough_does_not_interrupt():
    rows = [list(r) for r in RUNS["run1"]]
    for n in range(100, 104):                            # 120 ms burst, shorter than interrupt_min_ms
        rows[n] = [0.4, 1, rows[n][2]]
    assert feed_all(shipped(), rows) is None


def test_non_speech_never_interrupts():
    rows = [list(r) for r in RUNS["run1"]]
    for n in range(60, 160):
        rows[n] = [0.5, 0, rows[n][2]]                   # very loud, but the VAD says it isn't speech
    assert feed_all(shipped(), rows) is None


def test_with_headphones_a_quiet_voice_is_enough():
    det = shipped()
    out = [0.5 if (n // 12) % 2 == 0 else 0.0 for n in range(120)]      # EV's speech, almost no echo reaches the mic
    rows = [(0.003, 1, o) for o in out]
    assert not any(det.feed(*r) for r in rows)
    assert any(det.feed(0.05, True, o) for o in out)     # the user speaks at a normal level


def test_the_users_voice_is_not_learned_as_echo():
    """A long interruption must not teach the detector that 'loud' is normal and then stop noticing it."""
    det = shipped()
    rows = RUNS["run1"]
    first = feed_all(det, rows, user=(3.0, 0.12))
    assert first is not None
    assert det._run >= det.need                          # still seeing it as an interruption, not as echo


# ---- quiet_fires: the mic is also open while EV thinks (conversation.listen_while_thinking)
def quiet_detector(**kw) -> EchoDetector:
    c = load_config().raw["conversation"]
    return EchoDetector(block_ms=30, min_ms=c["interrupt_min_ms"], margin=c["interrupt_margin"], floor=c["interrupt_floor"], **kw)


def test_speech_while_ev_is_silent_fires_only_when_asked():
    talking = [(0.06, True, 0.0)] * 15                    # 0.45 s of a person talking, EV making no sound
    on, off = quiet_detector(quiet_fires=True), quiet_detector()
    assert any([on.feed(*row) for row in talking])
    assert not any([off.feed(*row) for row in talking])                     # off: unchanged behaviour


def test_quiet_mode_ignores_noise_the_vad_does_not_call_speech():
    det = quiet_detector(quiet_fires=True)
    assert not any([det.feed(0.06, False, 0.0) for _ in range(60)])
    det = quiet_detector(quiet_fires=True)
    assert not any([det.feed(0.005, True, 0.0) for _ in range(60)])         # below the floor


@pytest.mark.parametrize("run", RUNS)
def test_quiet_mode_still_never_mistakes_ev_s_own_echo_for_the_user(run):
    assert feed_all(quiet_detector(quiet_fires=True), RUNS[run]) is None


def test_quiet_mode_waits_out_the_echo_after_ev_stops():
    delay, gain = 16, 0.075                               # blocks the sound takes to reach the mic; mic rms per unit of playback
    env = np.convolve(np.random.default_rng(3).uniform(0.2, 1.0, 100), np.ones(3) / 3, "same")      # speech-like, not periodic
    out = [0.4 * float(e) for e in env[:90]] + [0.0] * 40
    rows = [(gain * out[n - delay] if n >= delay else 0.0, True, out[n]) for n in range(len(out))]
    det = quiet_detector(quiet_fires=True)
    assert [det.feed(*row) for row in rows].count(True) == 0   # EV talking, then its echo dying away: nobody interrupted
    assert 14 <= det.lag <= 18
