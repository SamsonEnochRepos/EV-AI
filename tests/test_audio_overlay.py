import threading
import time

import numpy as np
import pytest

from ev.audio.capture import Recorder
from ev.audio.playback import Player
from ev.audio.vad import EnergyVAD, rms
from ev.overlay import protocol
from ev.overlay.anim import BarAnimator, demo_level, demo_schedule
from ev.tts.speaker import Speaker

SR = 16000
BLK = 480


def tone(ms, amp=0.2):
    n = SR * ms // 1000
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype(np.int16)


def silence(ms):
    return (np.random.default_rng(0).normal(0, 20, SR * ms // 1000)).astype(np.int16)


def blocks(*segs):
    audio = np.concatenate(segs)
    for i in range(0, len(audio) - BLK + 1, BLK):
        yield audio[i:i + BLK]


def test_recorder_endpoints_speech(cfg):
    levels = []
    rec = Recorder(cfg, EnergyVAD(), on_level=levels.append)
    out = rec.record(blocks(silence(500), tone(1200), silence(1500)))
    assert out is not None
    assert 1.2 <= len(out) / SR <= 2.4
    assert max(levels) > 0.3


def test_recorder_no_speech_returns_none(cfg):
    assert Recorder(cfg, EnergyVAD()).record(blocks(silence(8000)), wait_for_speech_s=2) is None


def test_recorder_ignores_blips(cfg):
    assert Recorder(cfg, EnergyVAD()).record(blocks(silence(500), tone(60), silence(3000))) is None


def test_recorder_cancel(cfg):
    rec = Recorder(cfg, EnergyVAD())
    def gen():
        for i, b in enumerate(blocks(silence(300), tone(5000))):
            if i == 20:
                rec.cancel()
            yield b
    assert rec.record(gen()) is None


def test_recorder_max_length(cfg):
    out = Recorder(cfg, EnergyVAD()).record(blocks(silence(300), tone(30000)))
    assert out is not None and len(out) / SR <= cfg["audio"]["max_utterance_s"] + 0.5


def test_rms():
    assert rms(np.zeros(10, np.int16)) == 0
    assert rms(tone(100, 0.5)) == pytest.approx(0.35, abs=0.03)


def test_player_levels_and_barge_in():
    levels, played = [], []
    p = Player(on_level=levels.append, sink=lambda seg, sr: (played.append(len(seg)), time.sleep(0.02)))
    p.play(np.ones(22050 * 2, np.float32) * 0.2, 22050)       # 2 s of audio
    time.sleep(0.15)
    assert p.is_playing
    t0 = time.perf_counter()
    p.stop()
    assert p.wait_idle(1.0)
    assert (time.perf_counter() - t0) < 0.5
    assert sum(played) < 22050 * 2 and max(levels) > 0
    assert levels[-1] == 0.0
    p.close()


class FakeEngine:
    name = "fake"
    sample_rate = 22050

    def __init__(self, fail=False, delay=0.0):
        self.fail, self.delay, self.texts = fail, delay, []

    def load(self): ...

    def synthesize(self, text):
        from ev.tts.base import TTSError
        self.texts.append(text)
        time.sleep(self.delay)
        if self.fail:
            raise TTSError("boom")
        yield np.ones(2205, np.float32) * 0.1


def test_speaker_sentence_streaming_and_order():
    got = []
    p = Player(sink=lambda seg, sr: got.append(len(seg)))
    eng = FakeEngine()
    s = Speaker(eng, p)
    for t in ["One.", "Two.", "Three."]:
        s.say(t)
    assert s.wait_done(3)
    assert eng.texts == ["One.", "Two.", "Three."] and s.first_chunk_ms
    s.close()


def test_speaker_barge_in_discards_queue():
    p = Player(sink=lambda seg, sr: time.sleep(0.01))
    eng = FakeEngine(delay=0.2)
    s = Speaker(eng, p)
    for i in range(5):
        s.say(f"Sentence {i}.")
    time.sleep(0.05)
    s.stop()
    assert s.wait_done(2)
    time.sleep(0.5)
    assert len(eng.texts) <= 2
    s.say("After.")
    assert s.wait_done(3) and eng.texts[-1] == "After."
    s.close()


def test_speaker_tts_failure_reported_not_fatal():
    errs = []
    s = Speaker(FakeEngine(fail=True), Player(sink=lambda *a: None), on_error=errs.append)
    s.say("Hello.")
    assert s.wait_done(2) and errs
    s.close()


# ---- overlay math / protocol
def test_bars_follow_input_level():
    a = BarAnimator(11)
    a.state = "listening"
    quiet = loud = None
    for i in range(60):
        quiet = a.step(i / 30, 0.02)
    mq = sum(quiet)
    for i in range(60, 120):
        loud = a.step(i / 30, 0.9)
    assert sum(loud) > mq * 2


def test_calm_states_ignore_mic_level():
    a = BarAnimator(11)
    a.state = "thinking"
    for i in range(60):
        h = a.step(i / 30, 1.0)
    assert max(h) < 0.5


def test_paused_is_flat():
    a = BarAnimator(11)
    a.state = "paused"
    for i in range(90):
        h = a.step(i / 30, 1.0)
    assert max(h) - min(h) < 0.02 and max(h) < 0.15


def test_demo_covers_all_states():
    names = {n for n, _ in demo_schedule()}
    assert {"listening", "thinking", "acting", "speaking", "paused", "idle"} <= names
    assert 0 <= demo_level("speaking", 3.3) <= 1 and demo_level("thinking", 1) == 0


def test_protocol_validation():
    assert protocol.decode(b'{"t":"level","v":0.5}\n') == {"t": "level", "v": 0.5}
    assert protocol.decode(b"[1]") is None and protocol.decode(b"\xff\xfe") is None
    assert protocol.decode(b"x" * 5000) is None
    assert protocol.valid_message({"t": "state", "v": "listening"})
    assert not protocol.valid_message({"t": "level", "v": "high"})
    assert not protocol.valid_message({"t": "level", "v": True})
    assert not protocol.valid_message({"t": "exec", "v": "calc"})
    assert protocol.token_ok("abc", "abc") and not protocol.token_ok("abc", "abd")
    assert protocol.is_loopback("127.0.0.1") and not protocol.is_loopback("192.168.0.4")


def test_bind_is_loopback():
    s = protocol.bind_loopback()
    try:
        assert s.getsockname()[0] == "127.0.0.1"
    finally:
        s.close()


def test_scheduler_thread_fires(memory):
    fired = []
    from ev.tools.scheduler import Scheduler
    sc = Scheduler(memory, fired.append, tick_s=0.05)
    sc.add_timer(0.1, "tea")
    sc.start()
    time.sleep(0.5)
    sc.stop()
    assert fired and "tea" in fired[0]
    assert threading.active_count() >= 1


# ---- SharedMic: one stream, many listeners
def test_shared_mic_fans_out_to_every_listener_and_unsubscribes():
    import threading as th

    from ev.audio.capture import SharedMic
    from ev.core.config import load_config

    mic = SharedMic(load_config())
    stop_a, stop_b = th.Event(), th.Event()
    got_a, got_b = [], []
    ga, gb = mic.blocks(stop_a.is_set), mic.blocks(stop_b.is_set)

    def drain(gen, out, n):
        for blk in gen:
            out.append(int(blk[0]))
            if len(out) >= n:
                return

    ta = th.Thread(target=drain, args=(ga, got_a, 3))
    tb = th.Thread(target=drain, args=(gb, got_b, 2))
    ta.start(), tb.start()
    for _ in range(100):                                   # both generators must have subscribed before we publish
        if len(mic._subs) == 2:
            break
        time.sleep(0.01)
    for i in range(3):
        mic._publish(np.full(480, i, np.int16))
    ta.join(2), tb.join(2)
    assert got_a == [0, 1, 2] and got_b == [0, 1]          # same blocks, same order, to both
    ga.close(), gb.close()
    assert mic._subs == []                                 # a finished listener leaves no queue behind


def test_shared_mic_only_delivers_blocks_published_after_subscribing():
    from ev.audio.capture import SharedMic
    from ev.core.config import load_config

    mic = SharedMic(load_config())
    mic._publish(np.ones(480, np.int16))                   # nobody listening: dropped, never stored
    stop = [False]
    gen = mic.blocks(lambda: stop[0])                      # subscribes now, even before the first read
    mic._publish(np.full(480, 7, np.int16))
    assert int(next(gen)[0]) == 7
    stop[0] = True
    assert list(gen) == [] and mic._subs == []             # stopping ends the iteration and unsubscribes


def test_dropping_an_unread_listener_unsubscribes_it():
    from ev.audio.capture import SharedMic
    from ev.core.config import load_config

    mic = SharedMic(load_config())
    mic.blocks(lambda: False)                              # never iterated, immediately garbage
    assert mic._subs == []
