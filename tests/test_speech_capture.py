"""Speech capture: nothing said is lost at a hand-off, a pause is not the end of a sentence, and only EV's name interrupts it."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from ev.app import _ADDRESSED
from ev.audio.capture import Recorder, SharedMic
from ev.audio.endpoint import looks_unfinished
from ev.audio.interrupt import InterruptMonitor
from ev.audio.trigger import WakeWord
from ev.core.config import load_config
from ev.llm.fastpath import match_fast
from ev.stt.whisper import fix_names


def blk(level: float = 0.0, n: int = 480) -> np.ndarray:
    return np.full(n, int(level * 32767), np.int16)


@pytest.fixture
def cfg():
    return load_config()


def open_mic(cfg) -> SharedMic:
    mic = SharedMic(cfg)
    mic._stream = object()                       # "open" without a sound card; blocks are published by hand
    return mic


# ---- the shared microphone keeps a short memory so a listener can start from a given moment
def test_a_listener_can_start_from_a_mark_and_gets_every_block_once(cfg):
    mic = open_mic(cfg)
    for i in range(5):
        mic._publish(blk(i / 100))
    mark = mic.mark()
    for i in range(5, 9):
        mic._publish(blk(i / 100))                                  # said while nobody was listening
    ln = mic.blocks(lambda: False, since=mark)
    mic._publish(blk(0.09))                                         # arrives live afterwards
    got = [next(ln) for _ in range(5)]
    assert [int(b[0]) for b in got] == [int(i / 100 * 32767) for i in range(5, 10)]
    assert ln.resume_from == 10                                     # the next listener continues exactly here


def test_without_since_only_live_blocks_arrive(cfg):
    mic = open_mic(cfg)
    mic._publish(blk(0.5))
    ln = mic.blocks(lambda: True)
    assert list(ln) == []                                           # stop() is already true: nothing, and no old audio


def test_the_memory_is_a_few_seconds_not_a_recording(cfg):
    mic = open_mic(cfg)
    for _ in range(1000):
        mic._publish(blk(0.1))
    assert len(mic._ring) <= int(mic.RING_S * 1000 / cfg["audio"]["block_ms"])


# ---- the wake word listens on that microphone and remembers when it fired
class OneShotModel:
    """Scores 1.0 for any chunk that contains a loud sample, else 0."""

    def predict(self, chunk):
        return {"hey_evie": 1.0 if int(np.max(chunk)) > 20000 else 0.0}

    def reset(self): ...


def test_wake_word_on_the_shared_mic_marks_where_the_recorder_should_start(cfg):
    mic = open_mic(cfg)
    ww = WakeWord(cfg, lambda: None)
    ww.source = mic
    result: list[bool] = []
    t = threading.Thread(target=lambda: result.append(ww._listen_shared(OneShotModel())))
    t.start()
    time.sleep(0.2)
    for _ in range(12):
        mic._publish(blk(0.0))                                      # quiet
    for _ in range(3):
        mic._publish(blk(0.8))                                      # "Evie"
    later = mic.mark()
    for _ in range(8):
        mic._publish(blk(0.0))
    t.join(5)
    assert result == [True]
    mark = ww.take_mark()
    assert mark is not None and mark <= later and mark >= later - 12      # just before / around the moment it was heard
    assert ww.take_mark() is None                                       # used once


def test_a_stale_wake_mark_is_not_used(cfg):
    ww = WakeWord(cfg, lambda: None)
    ww._mark = (5, time.monotonic() - ww.MARK_TTL_S - 1)
    assert ww.take_mark() is None


# ---- a pause is not the end of the sentence
class ScriptVAD:
    def __init__(self, pattern: str):
        self.pattern, self.i = pattern, 0

    def reset(self): ...

    def is_speech(self, _b):
        c = self.pattern[min(self.i, len(self.pattern) - 1)]
        self.i += 1
        return c == "S"


def record(cfg, pattern: str, silence_ms: int) -> tuple[np.ndarray | None, Recorder]:
    cfg.raw["audio"]["silence_ms_to_end"] = silence_ms
    r = Recorder(cfg, ScriptVAD(pattern))
    return r.record(iter([blk(0.1)] * len(pattern)), wait_for_speech_s=2), r


def test_shipped_silence_tolerates_a_thinking_pause(cfg):
    pattern = "S" * 12 + "." * 22 + "S" * 12 + "." * 40          # "open the dashboard" ... 660 ms ... "in Firefox"
    audio, rec = record(cfg, pattern, cfg["audio"]["silence_ms_to_end"])
    assert audio is not None and len(audio) // 480 >= 12 + 22 + 12      # both halves are in one recording
    assert rec.last["longest_pause_ms"] == 660
    assert "silence" in str(rec.last["ended"])


def test_the_old_half_second_cuts_the_same_sentence_in_two(cfg):
    pattern = "S" * 12 + "." * 22 + "S" * 12 + "." * 40
    audio, _ = record(cfg, pattern, 550)
    assert audio is not None and len(audio) // 480 < 12 + 22 + 12       # what the log showed: only the first half


@pytest.mark.parametrize("said", [
    "open the", "Can you?", "can you", "Evie", "hey Evie, can you", "open the dashboard and", "set a timer for", "checks, I mean,",
    "what's the capital of", "I want to", "open it in", "tell me about my", "turn the", "search for", "I need you to", "remind me to",
    "so, um,", "and then",
])
def test_these_read_like_the_middle_of_a_sentence(said):
    assert looks_unfinished(said)


@pytest.mark.parametrize("said", [
    "open the dashboard", "what time is it", "thanks", "yes", "what?", "stop", "set a timer for ten minutes", "open it", "do that",
    "turn it up", "can you open the dashboard", "no thank you", "how are you doing today", "pause the music", "show me", "thank you",
    "", "play",
])
def test_these_are_finished(said):
    assert not looks_unfinished(said)


# ---- only EV's name interrupts EV
@pytest.mark.parametrize("said", [
    "EV, don't open it", "Evie, wait", "oh Evie can you not open the dashboard", "hey ev stop", "Eevee, open it in Firefox",
    "Okay, Evie. Stop.", "wait, EV", "sorry Evie",
])
def test_speech_that_starts_with_the_name_is_for_ev(said):
    assert _ADDRESSED.match(said)


@pytest.mark.parametrize("said", [
    "I think the dashboard is open", "every time I try", "that's what I said", "can you hear that", "Give me?", "checks, I mean,",
    "the TV is loud", "", "never mind", "evening everyone",
])
def test_everything_else_is_not(said):
    assert not _ADDRESSED.match(said)


class FakeListener:
    """Stands in for a MicListener: yields blocks, and says where a recorder should pick up."""

    def __init__(self, blocks):
        self._it = iter(blocks)
        self.resume_from = 4242

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self): ...


class LoudVAD:
    def reset(self): ...

    def is_speech(self, b):
        return int(np.max(np.abs(b))) > 1000


def run_monitor(cfg, verify, calls):
    cfg.raw["conversation"]["listen_while_thinking"] = True        # EV is silent here: plain speech counts, no echo to model
    stream = [blk(0.0)] * 5 + [blk(0.08)] * 60 + [blk(0.0)] * 40
    mon = InterruptMonitor(cfg, LoudVAD(), lambda: 0.0, lambda *a: calls.append(a), lambda _c, _s: FakeListener(stream))
    mon.verify = verify
    mon._run()


def test_speech_that_is_not_for_ev_does_not_stop_it(cfg):
    calls: list = []
    asked: list = []
    run_monitor(cfg, lambda blocks: asked.append(len(blocks)) or False, calls)
    assert asked and calls == []                                    # it was checked, found not to be for EV, and ignored


def test_speech_for_ev_stops_it_and_hands_over_everything_heard(cfg):
    calls: list = []
    run_monitor(cfg, lambda blocks: True, calls)
    assert len(calls) == 1
    blocks, resume = calls[0]
    assert resume == 4242                                           # the recorder resumes exactly where the monitor stopped
    assert len(blocks) >= 10                                        # the start of the sentence is kept, not just the last 600 ms


def test_without_the_name_check_any_speech_interrupts_as_before(cfg):
    calls: list = []
    run_monitor(cfg, None, calls)
    assert len(calls) == 1 and calls[0][0]                          # interrupted, with the speech, no check in between


# ---- the words Whisper breaks up, and how a request that starts with the name is read
@pytest.mark.parametrize("heard,fixed", [
    ("Pin Dashable.", "open the dashboard"), ("Can you open the dash board?", "Can you open the dashboard?"),
    ("open the dash", "open the dashboard"), ("show me the Dash", "show me the dashboard"),
    ("a quick dash to the shop", "a quick dash to the shop"),
    ("Can you open YouTube in a new talk?", "Can you open YouTube in a new tab?"),
    ("open it in a different top", "open it in a different tab"),
    ("let's talk about it", "let's talk about it"),
    ("a new talk about whales", "a new talk about whales"),
])
def test_dashboard_is_put_back_together(heard, fixed):
    assert fix_names(heard).rstrip(".") == fixed.rstrip(".")


@pytest.mark.parametrize("said", ["open the dashboard", "Evie, open the dashboard", "oh Evie, can you open the dashboard",
                                  "hey ev open the dashboard please"])
def test_a_request_may_start_with_either_spelling_of_the_name(said):
    m = match_fast(said)
    assert m is not None and m.tool == "open_dashboard"
