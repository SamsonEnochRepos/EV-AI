"""Integration tests: the real EVApp wiring with mocked mic / STT / speaker / Ollama."""
from __future__ import annotations

import threading
import time

import httpx
import numpy as np
import pytest

from ev.app import EVApp
from ev.audio.capture import MicError
from ev.audio.playback import Player
from ev.core.config import load_config
from ev.core.events import State
from ev.llm.client import OllamaClient
from ev.tools.base import Tool, ToolResult
from ev.tts.speaker import Speaker

from .fakes import FakeOllama, text_reply, tool_reply

pytestmark = pytest.mark.integration

SR = 16000


class Engine:
    name = "fake"
    sample_rate = 22050

    def __init__(self, fail=False, delay=0.0):
        self.fail, self.delay, self.said = fail, delay, []

    def load(self): ...

    def synthesize(self, text):
        from ev.tts.base import TTSError
        self.said.append(text)
        time.sleep(self.delay)
        if self.fail:
            raise TTSError("boom")
        yield np.ones(4410, np.float32) * 0.1


def speech_blocks():
    t = np.arange(SR) / SR
    tone = (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype(np.int16)
    audio = np.concatenate([np.zeros(4800, np.int16), tone, np.zeros(SR, np.int16)])
    for i in range(0, len(audio) - 480 + 1, 480):
        yield audio[i:i + 480]


@pytest.fixture
def app(tmp_path, monkeypatch):
    cfg = load_config(overrides={"general": {"data_dir": str(tmp_path / "data"), "log_dir": str(tmp_path / "logs")},
                                 "overlay": {"enabled": False}, "tray": {"enabled": False}})
    a = EVApp(cfg)
    a.overlay.enabled = False
    from ev.audio.vad import EnergyVAD
    a.recorder.vad = EnergyVAD()          # synthetic tone is not speech for Silero; this tests wiring
    a.engine = Engine()
    a.sink_audio = []
    a.speaker = Speaker(a.engine, Player(sink=lambda seg, sr: (a.sink_audio.append(len(seg)), time.sleep(0.005))))
    monkeypatch.setattr("ev.app.mic_blocks", lambda cfg, stop: speech_blocks())
    a.stt.transcribe = lambda audio: type("T", (), {"text": a.heard, "rtf": 0.1})()   # type: ignore[method-assign]
    a.heard = "what time is it"
    a.states = []
    a.bus.subscribe("state", lambda old, new: a.states.append(new))
    yield a
    a.shutdown()


def use_fake_llm(a, fake):
    a.client = OllamaClient(a.cfg, transport=fake.transport())
    a.brain.client = a.client


def test_full_voice_turn_fast_path(app):
    app.on_trigger()
    assert app.engine.said and "PM" in app.engine.said[0] or "AM" in app.engine.said[0]
    assert app.sm.state is State.IDLE
    seq = [s for s in app.states]
    assert State.LISTENING in seq and State.TRANSCRIBING in seq and State.THINKING in seq and State.SPEAKING in seq
    assert seq.index(State.LISTENING) < seq.index(State.TRANSCRIBING) < seq.index(State.THINKING) < seq.index(State.SPEAKING)


def test_llm_turn_streams_sentences_and_returns_to_idle(app):
    app.heard = "how are you doing today"
    use_fake_llm(app, FakeOllama([text_reply("Doing well, thanks for asking. How about you?")]))
    app.on_trigger()
    assert app.engine.said == ["Doing well, thanks for asking.", "How about you?"]
    assert app.sm.state is State.IDLE


def test_tool_turn_enters_acting_state(app):
    app.heard = "remember my dog is Biscuit"
    use_fake_llm(app, FakeOllama([tool_reply("remember_fact", {"key": "dog", "value": "Biscuit"}), text_reply("Got it.")]))
    app.on_trigger()
    assert app.memory.recall("dog") == "Biscuit"
    assert State.ACTING in app.states


def test_confirmation_by_hotkey_runs_action(app):
    ran = []
    app.registry.register(Tool("delete_file", "d", {"path": {"type": "string"}},
                               lambda path: ran.append(path) or ToolResult(True, "gone"), ("path",)))
    app.heard = "delete my temp file"
    use_fake_llm(app, FakeOllama([tool_reply("delete_file", {"path": "data/sandbox/a.txt"}), text_reply("Deleted.")]))
    threading.Timer(0.8, lambda: app._answers.put(True)).start()      # user presses the confirm hotkey
    app.on_trigger()
    assert ran == ["data/sandbox/a.txt"]
    assert any("delete file" in s.lower() for s in app.engine.said), app.engine.said   # exact readback was spoken


def test_confirmation_declined_blocks_action(app):
    ran = []
    app.registry.register(Tool("delete_file", "d", {"path": {"type": "string"}},
                               lambda path: ran.append(path) or ToolResult(True, "gone"), ("path",)))
    app.heard = "delete my temp file"
    use_fake_llm(app, FakeOllama([tool_reply("delete_file", {"path": "data/sandbox/a.txt"}), text_reply("Okay, left it.")]))
    threading.Timer(0.8, lambda: app._answers.put(False)).start()
    app.on_trigger()
    assert ran == []


def test_confirmation_timeout_denies(app):
    ran = []
    app.registry.register(Tool("delete_file", "d", {"path": {"type": "string"}},
                               lambda path: ran.append(path) or ToolResult(True, "gone"), ("path",)))
    app.policy.d["defaults"]["confirm_timeout_s"] = 1
    app.heard = "delete my temp file"
    use_fake_llm(app, FakeOllama([tool_reply("delete_file", {"path": "data/sandbox/a.txt"}), text_reply("Skipped.")]))
    app.recorder.record = lambda *a, **k: None          # type: ignore[method-assign]   # nobody answers by voice
    app.on_trigger()
    assert ran == []


def test_kill_switch_stops_speech_fast(app):
    app.engine.delay = 0.3
    for i in range(6):
        app.speaker.say(f"Sentence number {i}.")
    time.sleep(0.1)
    t0 = time.perf_counter()
    app.kill()
    ms = (time.perf_counter() - t0) * 1000
    assert ms < 500
    assert app.gate.killed.is_set()
    assert not app.gate.check("get_time_date", {}).allowed
    assert app.sm.state is State.IDLE
    time.sleep(1.3)                                      # re-arms
    assert app.gate.check("get_time_date", {}).allowed


def test_barge_in_interrupts_speech(app):
    app.engine.delay = 0.2
    for i in range(5):
        app.speaker.say(f"Long sentence {i}.")
    time.sleep(0.1)
    assert app.speaker.is_speaking
    app.heard = "what time is it"
    app.on_trigger()                                     # hotkey while speaking
    assert not any(f"Long sentence {i}." in app.engine.said[1:] for i in range(3, 5))
    assert app.sm.state is State.IDLE


def test_ollama_down_is_spoken_not_fatal(app):
    def boom(request):
        raise httpx.ConnectError("refused")
    app.client = OllamaClient(app.cfg, transport=httpx.MockTransport(boom))
    app.brain.client = app.client
    app.heard = "tell me a story"
    app.on_trigger()
    assert any("Ollama" in s or "brain" in s for s in app.engine.said)
    assert app.sm.state is State.IDLE


def test_missing_microphone_gives_spoken_error(app, monkeypatch):
    def nomic(cfg, stop):
        raise MicError("no device")
        yield  # pragma: no cover
    monkeypatch.setattr("ev.app.mic_blocks", nomic)
    app.on_trigger()
    assert any("microphone" in s.lower() for s in app.engine.said)


def test_tts_failure_does_not_crash_turn(app):
    app.engine.fail = True
    app.on_trigger()
    assert app.sm.state in (State.IDLE, State.SPEAKING)
    app.engine.fail = False
    app.heard = "what time is it"
    app.on_trigger()
    assert app.engine.said


def test_no_speaker_at_all_is_survivable(app):
    app.speaker = None
    app.on_trigger()
    assert app.sm.state is State.IDLE


def test_paused_and_muted_ignore_trigger(app):
    app.paused = True
    app.on_trigger()
    assert app.engine.said == []
    app.paused = False
    app.muted = True
    app.on_trigger()
    assert app.engine.said == []


def test_reminder_alert_is_spoken(app):
    app.scheduler.poll_once()
    app.memory.add_reminder("stretch", time.time() - 1)
    app.scheduler.poll_once()
    app.speaker.wait_done(3)
    assert any("stretch" in s for s in app.engine.said)


def test_kill_switch_under_cpu_load(app):
    """Kill must stay fast while the machine is saturated (6 separate busy processes + speech queued)."""
    import subprocess
    import sys

    burners = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(6)]   # noqa: S603
    try:
        app.engine.delay = 0.2
        for i in range(8):
            app.speaker.say(f"Busy sentence {i}.")
        time.sleep(0.15)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            app.kill()
            times.append((time.perf_counter() - t0) * 1000)
            time.sleep(1.2)
        assert max(times) < 500, times
    finally:
        for b in burners:
            b.kill()
