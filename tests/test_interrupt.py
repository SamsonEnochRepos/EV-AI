"""Voice barge-in detector. Built around real levels recorded on a laptop whose speakers feed its own mic."""
from __future__ import annotations

import json
import math
from pathlib import Path

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
