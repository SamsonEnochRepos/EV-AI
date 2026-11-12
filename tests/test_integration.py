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
from ev.stt.whisper import Transcript
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
                                 "vault": {"path": str(tmp_path / "vault")},        # never the real Documents folder
                                 "overlay": {"enabled": False}, "tray": {"enabled": False},
                                 "conversation": {"followup_enabled": False, "voice_interrupt": False,    # fake mic always "hears" speech
                                                  "continuous_mic": False}})            # tests that want the shared stream turn it on and fake it
    a = EVApp(cfg)
    a.overlay.enabled = False
    from ev.audio.vad import EnergyVAD
    a.recorder.vad = EnergyVAD()          # synthetic tone is not speech for Silero; this tests wiring
    a.engine = Engine()
    a.sink_audio = []
    a.speaker = Speaker(a.engine, Player(sink=lambda seg, sr: (a.sink_audio.append(len(seg)), time.sleep(0.005))))
    monkeypatch.setattr("ev.app.mic_blocks", lambda cfg, stop: speech_blocks())
    a.stt.transcribe = lambda audio: Transcript(a.heard, 0.1, 2.0)   # type: ignore[method-assign]
    a.heard = "what time is it"
    a.states = []
    a.bus.subscribe("state", lambda old, new: a.states.append(new))
    yield a
    a.shutdown()


def use_fake_llm(a, fake):
    a.client = OllamaClient(a.cfg, transport=fake.transport())
    a.client._last_used = time.monotonic()        # warm: no background re-warm request eating a scripted reply
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


def test_real_prompt_leaves_room_for_history(app):
    """The whole tool list as the running app registers it (screen, control, vault tools included), not the lightweight test
    registry. If this fails, conversation history is being squeezed out of num_ctx: trim a description or a persona line."""
    b = app.brain
    fixed = b._est_tokens(b.system_prompt()) + b._est_tokens(app.registry.schemas())
    assert len(app.registry.schemas()) >= 15, "the tools did not register; this test would prove nothing"
    assert b.ctx_budget - fixed >= 400, f"fixed prompt ~{fixed} of {b.ctx_budget}: under 400 tokens left for history"


def test_tool_turn_enters_acting_state(app):
    app.heard = "my dog is Biscuit, keep that in mind"        # not "remember ...": that would take the fast path, not the model
    use_fake_llm(app, FakeOllama([tool_reply("remember_fact", {"key": "dog", "value": "Biscuit"}), text_reply("Got it.")]))
    app.on_trigger()
    assert app.vault is not None and app.vault.profile_facts()["dog"] == "Biscuit"     # facts live in the vault now
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


# ---- follow-up listening: after EV answers, the mic stays open for the next sentence (no hotkey / wake word)
def _silence_blocks():
    for _ in range(300):
        yield np.zeros(480, np.int16)


def _mic_script(monkeypatch, kinds):
    """Each mic open yields the next kind of audio: 'speech' or 'silence' (the last kind repeats)."""
    opened = []

    def fake(cfg, stop):
        k = kinds[min(len(opened), len(kinds) - 1)]
        opened.append(k)
        return speech_blocks() if k == "speech" else _silence_blocks()

    monkeypatch.setattr("ev.app.mic_blocks", fake)
    return opened


def test_followup_turn_without_retrigger(app, monkeypatch):
    app.cfg.raw["conversation"]["followup_enabled"] = True
    app.cfg.raw["conversation"]["followup_window_s"] = 1
    opened = _mic_script(monkeypatch, ["speech", "speech", "silence"])
    heard = iter(["what time is it", "what's the date today"])
    app.stt.transcribe = lambda audio: Transcript(next(heard), 0.1, 2.0)   # type: ignore[method-assign]
    app.on_trigger()
    assert opened == ["speech", "speech", "silence"]       # turn 1, follow-up turn, then the window times out
    assert len(app.engine.said) == 2
    assert app.sm.state is State.IDLE and not app._following_up


def test_wrap_up_phrase_ends_conversation(app, monkeypatch):
    app.cfg.raw["conversation"]["followup_enabled"] = True
    opened = _mic_script(monkeypatch, ["speech"])
    app.heard = "thanks"
    use_fake_llm(app, FakeOllama([text_reply("Anytime.")]))
    app.on_trigger()
    assert opened == ["speech"]                            # no follow-up window after "thanks"
    assert app.sm.state is State.IDLE


def test_followup_turns_are_capped(app, monkeypatch):
    app.cfg.raw["conversation"]["followup_enabled"] = True
    app.cfg.raw["conversation"]["followup_max_turns"] = 3
    opened = _mic_script(monkeypatch, ["speech"])
    app.on_trigger()                                       # fake mic never goes quiet
    assert len(opened) == 3
    assert app.sm.state is State.IDLE


def test_hotkey_during_followup_window_is_ignored(app):
    app._following_up = True
    app.on_trigger()
    assert app.engine.said == []


# ---- voice barge-in: the user talks over EV, speech stops, and what they say next is heard
class FakeMonitor:
    """Stands in for the mic monitor: fires 'user talked over EV' shortly after the reply starts."""

    def __init__(self, cfg, vad, out_level, on_interrupt, blocks):
        self.on_interrupt = on_interrupt
        self.stopped = False

    def start(self):
        threading.Timer(0.25, lambda: self.on_interrupt([np.ones(480, np.int16)] * 3)).start()

    def stop(self):
        self.stopped = True


def test_talking_over_ev_stops_speech_and_listens(app, monkeypatch):
    app.cfg.raw["conversation"].update(voice_interrupt=True, followup_window_s=1)
    app.engine.delay = 0.2                                     # each sentence takes a while to "speak"
    monitors = []
    monkeypatch.setattr("ev.app.InterruptMonitor", lambda *a: monitors.append(FakeMonitor(*a)) or monitors[-1])
    opened = _mic_script(monkeypatch, ["speech", "speech", "silence"])
    heard = iter(["tell me a story", "actually never mind the story, what time is it"])
    app.stt.transcribe = lambda audio: Transcript(next(heard), 0.1, 2.0)   # type: ignore[method-assign]
    use_fake_llm(app, FakeOllama([text_reply("Once upon a time there was a fox. " * 8, parts=24)]))
    app.on_trigger()
    assert monitors and monitors[0].stopped                    # the mic was released before re-recording
    assert opened[:2] == ["speech", "speech"]                  # story request, then the interrupting sentence
    assert app.brain.history[-2]["content"].startswith("actually never mind") or "time" in app.brain.history[-1]["content"].lower()
    assert app.sm.state is State.IDLE
    spoken_story = [s for s in app.engine.said if "fox" in s]
    assert 0 < len(spoken_story) < 8                           # the story was cut short, not read to the end


def test_voice_interrupt_off_never_starts_monitor(app, monkeypatch):
    started = []
    monkeypatch.setattr("ev.app.InterruptMonitor", lambda *a: started.append(1))
    app.on_trigger()
    assert started == []


class RecordingOverlay:
    """Stands in for the overlay process and records everything EV tells it."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, *a))


def test_display_mode_by_voice_shows_both_sides_of_the_conversation(app):
    app.overlay = RecordingOverlay()
    app.heard = "activate display mode"
    app.on_trigger()
    assert app.display_mode and app.memory.get_kv("display_mode") == "1"
    assert ("send_display", True) in app.overlay.calls
    assert ("send_text", "user", "activate display mode") in app.overlay.calls       # the command itself is shown too
    assert ("send_text", "ev", "Display mode on.", True) in app.overlay.calls

    app.overlay.calls.clear()
    app.heard = "what time is it"
    app.on_trigger()
    texts = [c for c in app.overlay.calls if c[0] == "send_text"]
    assert texts[0] == ("send_text", "user", "what time is it", False)
    assert texts[1][:2] == ("send_text", "ev") and texts[1][3] is True


def test_deactivating_display_mode_stops_sending_text(app):
    app.overlay = RecordingOverlay()
    app.display_mode = True
    app.heard = "deactivate display mode"
    app.on_trigger()
    assert not app.display_mode and app.memory.get_kv("display_mode") == "0"
    assert ("send_display", False) in app.overlay.calls
    app.overlay.calls.clear()
    app.heard = "what time is it"
    app.on_trigger()
    assert not [c for c in app.overlay.calls if c[0] == "send_text"]


def test_display_mode_is_off_by_default_and_remembered(app, tmp_path):
    assert app.display_mode is False
    app.memory.set_kv("display_mode", "1")
    from ev.app import EVApp
    from ev.core.config import load_config

    again = EVApp(load_config(overrides={"general": {"data_dir": str(tmp_path / "data"), "log_dir": str(tmp_path / "logs")},
                                         "vault": {"path": str(tmp_path / "vault")},
                                         "overlay": {"enabled": False}, "tray": {"enabled": False}}))
    try:
        assert again.display_mode is True
    finally:
        again.shutdown()


# ---- conversation feel: patience grows with the back-and-forth; filler noise is dropped; smalltalk is instant
def test_followup_window_grows_with_the_conversation(app):
    app.cfg.raw["conversation"].update(followup_enabled=True, followup_window_s=7, followup_patience_s=2,
                                       followup_window_max_s=12, followup_question_window_s=12)
    waits = [app._followup_wait("what's up", "Not much.", turn) for turn in range(5)]
    assert waits == [7, 9, 11, 12, 12]                       # +2 s per turn, capped at followup_window_max_s
    assert app._followup_wait("what's up", "Want me to?", 3) == 12      # a question already gets the long window
    assert app._followup_wait("thanks", "Anytime.", 2) is None          # wrapping up still ends it


def test_hesitation_sounds_in_a_followup_are_not_a_request(app, monkeypatch):
    app.cfg.raw["conversation"]["followup_enabled"] = True
    _mic_script(monkeypatch, ["speech", "speech", "silence"])
    heard = iter(["what time is it", "hmm"])
    app.stt.transcribe = lambda audio: Transcript(next(heard), 0.1, 2.0)   # type: ignore[method-assign]
    use_fake_llm(app, FakeOllama([text_reply("should never be asked")]))
    app.on_trigger()
    assert len(app.engine.said) == 1                         # only the time answer; "hmm" never reached the model
    assert app.sm.state is State.IDLE


def test_thanks_gets_an_instant_reply_without_the_model(app):
    fake = FakeOllama([text_reply("unused")])
    use_fake_llm(app, fake)
    app.heard = "thanks"
    app.on_trigger()
    assert fake.requests == [] and len(app.engine.said) == 1


# ---- continuous mic: one stream for the whole conversation
class FakeSharedMic:
    instances: list = []

    def __init__(self, cfg):
        self.is_open = False
        self.opened = self.closed = 0
        self.subscriptions = 0
        FakeSharedMic.instances.append(self)

    def open(self):
        self.is_open, self.opened = True, self.opened + 1

    def close(self):
        self.is_open, self.closed = False, self.closed + 1

    def mark(self, back=0):
        return 0

    def blocks(self, stop, since=None):
        self.subscriptions += 1
        return speech_blocks()


def test_continuous_mic_opens_once_per_conversation_and_closes_before_idle(app, monkeypatch):
    FakeSharedMic.instances = []
    monkeypatch.setattr("ev.app.SharedMic", FakeSharedMic)
    monkeypatch.setattr("ev.app.mic_blocks", lambda cfg, stop: (_ for _ in ()).throw(AssertionError("per-turn stream used")))
    app.cfg.raw["conversation"].update(continuous_mic=True, followup_enabled=True, followup_window_s=1, followup_max_turns=2)
    app.on_trigger()
    (mic,) = FakeSharedMic.instances
    assert mic.opened == 1 and mic.closed == 1             # one open for both turns, closed when the conversation ended
    assert mic.subscriptions == 2 and app._mic is None     # each turn's recorder subscribed to the same stream
    assert app.sm.state is State.IDLE


def test_continuous_mic_off_means_one_stream_per_turn(app, monkeypatch):
    FakeSharedMic.instances = []
    monkeypatch.setattr("ev.app.SharedMic", FakeSharedMic)
    app.on_trigger()                                       # the fixture turns it off
    assert FakeSharedMic.instances == []
    assert load_config().raw["conversation"]["continuous_mic"] is True      # the shipped default is on (D33): nothing lost at a hand-off


def test_continuous_mic_failure_falls_back_to_per_turn_streams(app, monkeypatch):
    class Broken(FakeSharedMic):
        def open(self):
            raise MicError("busy")

    monkeypatch.setattr("ev.app.SharedMic", Broken)
    app.cfg.raw["conversation"]["continuous_mic"] = True
    app.on_trigger()                                       # the patched per-turn mic still works
    assert app.engine.said and app._mic is None


# ---- listening while EV thinks: speaking again before EV answers cancels the turn and joins it to the next utterance
class OnceMonitor:
    """Fires 'the user spoke' the moment it starts, but only the first one created; later monitors stay silent."""
    created = 0

    def __init__(self, cfg, vad, out_level, on_interrupt, blocks):
        type(self).created += 1
        self.fire = type(self).created == 1
        self.on_interrupt = on_interrupt

    def start(self):
        if self.fire:
            self.on_interrupt([np.ones(480, np.int16)] * 3)

    def stop(self): ...


def _thinking_app(app, monkeypatch, kinds, window=1):
    OnceMonitor.created = 0
    app.cfg.raw["conversation"].update(voice_interrupt=True, listen_while_thinking=True, followup_window_s=window)
    app.brain.lookup_planner = False
    monkeypatch.setattr("ev.app.InterruptMonitor", OnceMonitor)
    return _mic_script(monkeypatch, kinds)


def test_speaking_again_before_ev_answers_joins_the_two_halves(app, monkeypatch):
    opened = _thinking_app(app, monkeypatch, ["speech", "speech"])
    heard = iter(["what's the capital of", "Australia"])
    app.stt.transcribe = lambda audio: Transcript(next(heard), 0.1, 2.0)        # type: ignore[method-assign]
    use_fake_llm(app, FakeOllama([text_reply("Cut off."), text_reply("Canberra.")]))
    app.on_trigger()
    users = [m["content"] for m in app.brain.history if m["role"] == "user"]
    assert users == ["what's the capital of Australia"]                          # one request, not two fragments
    assert "Canberra." in app.engine.said and "Cut off." not in app.engine.said
    assert opened == ["speech", "speech"] and app._held == ""


def test_if_nothing_more_is_said_the_held_request_is_still_answered(app, monkeypatch):
    opened = _thinking_app(app, monkeypatch, ["speech", "silence"])
    heard = iter(["what's the capital of France"])
    app.stt.transcribe = lambda audio: Transcript(next(heard), 0.1, 2.0)        # type: ignore[method-assign]
    use_fake_llm(app, FakeOllama([text_reply("Cut off."), text_reply("Paris.")]))
    app.on_trigger()                                                              # a cough, not a sentence
    assert "Paris." in app.engine.said
    assert [m["content"] for m in app.brain.history if m["role"] == "user"] == ["what's the capital of France"]
    assert opened == ["speech", "silence"]


def test_a_request_that_already_ran_a_tool_is_never_held_or_repeated(app, monkeypatch):
    _thinking_app(app, monkeypatch, ["speech", "silence"])
    app.heard = "what time is it"                                                 # fast path: the tool runs at once
    app.on_trigger()
    assert app._held == ""                                                        # nothing to merge into later


def test_listen_while_thinking_is_off_by_default(app, monkeypatch):
    started = []
    monkeypatch.setattr("ev.app.InterruptMonitor", lambda *a: started.append(1))
    app.cfg.raw["conversation"]["voice_interrupt"] = True
    app.heard = "how are you doing today"
    use_fake_llm(app, FakeOllama([text_reply("Fine.")]))
    app.on_trigger()
    assert started == [1]                                                         # only the one that starts with speech, as before


# ---- "think harder": the bigger model, only when asked
def _models_of(fake):
    return [r.get("model") for r in fake.requests if "messages" in r]


def test_think_harder_uses_the_bigger_model_and_says_so(app):
    app.brain.deep_model = "qwen2.5:7b"
    app.brain.lookup_planner = False
    app.heard = "think harder about whether I should learn Rust or Go"
    fake = FakeOllama([text_reply("Go is gentler for a Python person. Rust pays off later.")])
    fake.models.append("qwen2.5:7b")
    use_fake_llm(app, fake)
    app.on_trigger()
    assert _models_of(fake) == ["qwen2.5:7b"]
    assert app.engine.said[0] == "Okay, let me think that through."
    assert fake.requests[0]["options"]["num_predict"] == app.brain.deep_num_predict
    assert app.client.is_cold(90)                       # the main model was pushed out: it reloads while the user speaks next


def test_ordinary_questions_and_missing_model_stay_on_the_main_model(app):
    app.brain.deep_model = "qwen2.5:7b"
    app.brain.lookup_planner = False
    fake = FakeOllama([text_reply("Go is gentler.")])
    use_fake_llm(app, fake)
    app.heard = "should I learn Rust or Go first"
    app.on_trigger()
    app.heard = "think harder about it"                 # asked, but the model isn't installed (the fake lists only two)
    app.on_trigger()
    assert set(_models_of(fake)) == {app.client.model}
    assert "Okay, let me think that through." not in app.engine.said


# ---- speech capture: nothing lost at a hand-off, a pause is not the end of a sentence (D33)
class HandoffMic:
    is_open = True

    def __init__(self):
        self.since_args = []

    def mark(self, back=0):
        return 7

    def blocks(self, stop, since=None):
        self.since_args.append(since)
        return speech_blocks()

    def close(self): ...


def test_an_unfinished_sentence_is_waited_for_and_joined(app):
    app.cfg.raw["conversation"].update(finish_rounds=2, finish_window_s=1)
    app._mic = HandoffMic()
    said = iter(["what time is", "what time is it"])
    lengths = []
    app.stt.transcribe = lambda audio: (lengths.append(len(audio)), Transcript(next(said), 0.1, 2.0))[1]   # type: ignore[method-assign]
    app.on_trigger()
    assert len(lengths) == 2 and lengths[1] > lengths[0]                  # the second decode holds both parts
    assert any("AM" in s or "PM" in s for s in app.engine.said)           # and the joined sentence is what was answered


def test_a_finished_sentence_is_not_made_to_wait(app):
    app._mic = HandoffMic()
    calls = []
    app.stt.transcribe = lambda audio: (calls.append(1), Transcript("what time is it", 0.1, 2.0))[1]   # type: ignore[method-assign]
    app.on_trigger()
    assert len(calls) == 1


def test_the_recording_starts_from_the_moment_the_wake_word_fired(app):
    mic = app._mic = HandoffMic()
    app.wake._mark = (7, time.monotonic())
    app.on_trigger()
    assert mic.since_args[0] == 7


def test_after_a_barge_in_the_recorder_resumes_exactly_where_the_monitor_stopped(app):
    mic = app._mic = HandoffMic()
    app._on_talk_over([np.ones(480, np.int16)], 99)
    app.on_trigger()
    assert mic.since_args[0] == 99


def test_talking_over_ev_is_checked_for_its_name_unless_that_is_turned_off(app, monkeypatch):
    made = []

    class Spy:
        def __init__(self, *a):
            made.append(self)
            self.verify = None

        def start(self): ...

    monkeypatch.setattr("ev.app.InterruptMonitor", Spy)
    app.cfg.raw["conversation"]["voice_interrupt"] = True
    assert app._start_monitor() is made[0] and made[0].verify == app._addressed_to_ev
    app.cfg.raw["conversation"]["interrupt_needs_name"] = False
    app._start_monitor()
    assert made[1].verify is None


def test_the_name_check_reads_what_was_said_and_fails_closed(app):
    app._name_score = lambda blocks: 0.0                                             # type: ignore[method-assign]
    app.stt.transcribe = lambda audio: Transcript("Evie, don't open it", 0.1, 2.0)    # type: ignore[method-assign]
    assert app._addressed_to_ev([np.ones(480, np.int16)])
    app.stt.transcribe = lambda audio: Transcript("so then I told him", 0.1, 2.0)     # type: ignore[method-assign]
    assert not app._addressed_to_ev([np.ones(480, np.int16)])

    def boom(audio):
        raise RuntimeError("no model")

    app.stt.transcribe = boom                                                        # type: ignore[method-assign]
    assert not app._addressed_to_ev([np.ones(480, np.int16)])                         # nothing can interrupt without a way to check


@pytest.mark.parametrize("said,score,addressed", [
    ("Eveled.", 0.0, True),                   # what Whisper wrote for "Evie" in the real log
    ("Eevee, stop", 0.0, True),
    ("Evvy wait", 0.0, True),
    ("so then I told him", 0.9, True),         # the speech model missed it; the wake-word model heard the name
    ("so then I told him", 0.1, False),
    ("every time I try it", 1.0, False),       # sounds like the name, is an ordinary word
    ("even if it works", 1.0, False),
    ("the TV is loud", 0.0, False),
])
def test_a_misheard_name_still_counts_but_ordinary_words_do_not(app, said, score, addressed):
    app.stt.transcribe = lambda audio: Transcript(said, 0.1, 2.0)                    # type: ignore[method-assign]
    app._name_score = lambda blocks: score                                           # type: ignore[method-assign]
    assert app._addressed_to_ev([np.ones(480, np.int16)]) is addressed
