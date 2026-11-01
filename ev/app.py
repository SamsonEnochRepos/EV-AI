"""EV application: wires triggers, audio, STT, brain, TTS, overlay, tray into the core loop."""
from __future__ import annotations

import itertools
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from ev.audio.capture import MicError, Recorder, SharedMic, mic_blocks
from ev.audio.interrupt import InterruptMonitor
from ev.audio.playback import Player
from ev.audio.trigger import Hotkeys, WakeWord
from ev.audio.vad import make_vad
from ev.core.config import Config
from ev.core.events import EventBus, State
from ev.core.log import get_logger, setup_logging
from ev.core.state import StateMachine
from ev.llm.brain import Brain
from ev.llm.client import LLMError, OllamaClient, OllamaDown
from ev.memory.store import MemoryStore
from ev.overlay.client import OverlayClient
from ev.safety.gate import SafetyGate, parse_yes_no
from ev.safety.policy import Policy
from ev.stt.whisper import STT
from ev.tools.base import ToolRegistry
from ev.tools.builtin import ToolContext, build_tools
from ev.tools.scheduler import Scheduler
from ev.tts.base import make_engine
from ev.tts.speaker import Speaker

log = get_logger("app")

# Whisper tends to invent these from room noise; in a follow-up window they mean "nobody spoke".
_NOISE = re.compile(r"^\W*(you|bye|(?:u+|a+|e+)?[hm]+|um+|uh+|er+|thanks for watching|thank you for watching)\W*$", re.I)
# The user is wrapping up: answer, but don't keep the mic open afterwards.
_WRAP_UP = re.compile(r"^\W*(no,? )?(thanks|thank you|that'?s all|that'?s it|nothing|nope|never ?mind|goodbye|bye|stop)\b", re.I)


class EVApp:
    def __init__(self, cfg: Config, debug: bool = False):
        self.cfg = cfg
        setup_logging(cfg.path("general", "log_dir"), debug, cfg.get("general", "log_conversations", True))
        self.bus = EventBus()
        self.sm = StateMachine(self.bus)
        self.memory = MemoryStore(cfg.data_path(cfg["memory"]["db_file"]))
        self.policy = Policy.load(cfg.path("general", "policy_file"))
        self.gate = SafetyGate(self.policy, ask=self.confirm)
        self.overlay = OverlayClient(cfg)
        self.client = OllamaClient(cfg)
        self.scheduler = Scheduler(self.memory, self._on_alert)
        self.registry: ToolRegistry = build_tools(ToolContext(cfg, self.memory, self.scheduler, self.policy))
        # display mode: the pill shows a text card with what the user said and what EV answers
        self.display_mode = self.memory.get_kv("display_mode", "1" if cfg["overlay"].get("display_mode") else "0") == "1"
        self._last_heard = ""
        self.overlay.send_display(self.display_mode)
        self._register_display_tool()
        self._register_screen_tools()
        self.brain = Brain(cfg, self.client, self.memory, self.registry, self.gate, on_acting=self._on_acting)
        self.stt = STT(cfg)
        self.vad = make_vad("silero", cfg["audio"]["vad_aggressiveness_threshold"])
        self.recorder = Recorder(cfg, self.vad, on_level=self.overlay.send_level)
        self.speaker: Speaker | None = None
        self.muted = False
        self.paused = False
        self._turn_lock = threading.Lock()
        self._answers: queue.Queue[bool] = queue.Queue()
        self._confirming = False
        self._abort = threading.Event()
        self._following_up = False                        # True while the follow-up window is listening
        self._interrupted = threading.Event()             # the user talked over EV's reply
        self._carry: list[np.ndarray] = []                # mic blocks captured while interrupting, fed to the next recording
        self._mic: SharedMic | None = None                # one stream for the whole conversation (conversation.continuous_mic)
        self._out_level = 0.0
        self.hotkeys = Hotkeys(cfg, {
            "push_to_talk": self.on_trigger, "kill_switch": self.kill,
            "confirm_yes": lambda: self._answers.put(True), "confirm_no": lambda: self._answers.put(False),
        })
        self.wake = WakeWord(cfg, self.on_trigger) if cfg["trigger"]["mode"] == "wakeword" else None
        self.tray: Any = None
        self.bus.subscribe("state", self._on_state)

    # ---- optional tools (screen/control) registered lazily so lightweight setups still start
    def _register_screen_tools(self) -> None:
        try:
            from ev.control.tools import register_control_tools
            from ev.screen.tools import register_screen_tools

            register_screen_tools(self.registry, self.cfg, self.policy, self.client)
            register_control_tools(self.registry, self.cfg, self.policy, self.gate)
            self._register_agent_tool()
        except Exception:
            log.exception("screen/control tools unavailable")

    def _register_display_tool(self) -> None:
        from ev.tools.base import Tool, ToolResult

        def set_display_mode(enabled: bool) -> ToolResult:
            if enabled == self.display_mode:
                return ToolResult(True, f"Display mode is already {'on' if enabled else 'off'}.")
            self.display_mode = enabled
            self.memory.set_kv("display_mode", "1" if enabled else "0")
            self.overlay.send_display(enabled)
            if enabled and self._last_heard:
                self.overlay.send_text("user", self._last_heard)       # show the command that turned it on
            return ToolResult(True, "Display mode on." if enabled else "Display mode off.")

        self.registry.register(Tool(
            "set_display_mode", "Turn display mode on or off: show the conversation as text on screen.",
            {"enabled": {"type": "boolean"}}, set_display_mode, ("enabled",), direct=True))

    def _show(self, who: str, text: str, append: bool = False) -> None:
        """Put a line of the conversation on the pill's text card (only does anything in display mode)."""
        if self.display_mode:
            self.overlay.send_text(who, text, append)

    def _register_agent_tool(self) -> None:
        from ev.control.agent import ScreenAgent
        from ev.tools.base import Tool, ToolResult

        def observe() -> str:
            tool = self.registry.get("read_screen")
            return tool.func().message if tool else ""

        def do_on_screen(goal: str) -> ToolResult:
            agent = ScreenAgent(self.client, self.registry, self.gate, observe,
                                max_steps=int(self.cfg["screen"]["agent_max_steps"]),
                                narrate=self.say_direct)
            r = agent.run(goal)
            return ToolResult(r.ok, r.summary, {"steps": r.steps})

        self.registry.register(Tool(
            "do_on_screen", "Carry out a multi-step task on screen, like opening an app and typing in it.",
            {"goal": {"type": "string", "maxLength": 300}}, do_on_screen, ("goal",), timeout_s=180))

    # ---- lifecycle
    def start(self) -> None:
        t0 = time.perf_counter()
        self.overlay.start()
        self._init_speaker()
        self._select_model()
        self.scheduler.start()
        self.hotkeys.start()
        if self.wake:
            self.wake.start()
            threading.Thread(target=self._report_wake, name="wake-status", daemon=True).start()
        threading.Thread(target=self._warm, name="warm", daemon=True).start()
        if self.cfg["tray"]["enabled"]:
            self._start_tray()
        log.info("EV ready in %.1fs", time.perf_counter() - t0)

    def _report_wake(self) -> None:
        assert self.wake is not None
        for _ in range(100):                      # model load takes a moment; give up after ~20 s
            if self.wake.ready.wait(0.2) or self.wake.error:
                break
        if self.wake.error:
            self.overlay.send_state("error")
            log.warning("wake word is off; push-to-talk (%s) still works", self.cfg["hotkeys"]["push_to_talk"])

    def _warm(self) -> None:
        """Load and exercise every slow part at once (speech model, voice, LLM prompt cache, app list) so the first
        request after start-up isn't the one that pays for them."""
        from ev.tools import system

        def run(label: str, fn: Any) -> None:
            t0 = time.perf_counter()
            try:
                fn()
                log.info("warm: %s ready in %.1fs", label, time.perf_counter() - t0)
            except Exception:
                log.warning("warm-up of %s incomplete", label, exc_info=True)

        jobs = [("speech model", self.stt.warm),
                ("llm", self.brain.warm),
                ("app list", system.warm_app_index)]
        if self.speaker:
            jobs.append(("voice", self.speaker.warm))
        threads = [threading.Thread(target=run, args=job, name=f"warm-{job[0]}", daemon=True) for job in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _init_speaker(self) -> None:
        try:
            engine = make_engine(self.cfg)
            try:
                engine.load()
            except Exception as e:
                if engine.name == "piper":
                    raise
                log.warning("%s voice unavailable (%s); falling back to piper", engine.name, e)
                engine = make_engine(self.cfg, "piper")
                engine.load()
            player = Player(on_level=self._on_out_level, device=self.cfg["audio"]["output_device"] or None)
            self.speaker = Speaker(engine, player, on_error=lambda e: log.error("speech error: %s", e),
                                   pause_ms=int(self.cfg.get("tts", "sentence_pause_ms", 0)))
        except Exception as e:
            log.error("TTS unavailable: %s", e)
            self.speaker = None

    def _select_model(self) -> None:
        try:
            have = self.client.available_models()
        except OllamaDown:
            log.error("Ollama is not running")
            return
        want = self.client.model
        if want not in have and self.client.fallback in have:
            log.warning("model %s missing; falling back to %s", want, self.client.fallback)
            self.client.model = self.client.fallback

    def run_forever(self) -> None:
        self.start()
        try:
            while not getattr(self, "_quit", False):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._quit = True
        self.kill()
        self.hotkeys.stop()
        if self.wake:
            self.wake.stop()
        self.scheduler.stop()
        self.overlay.stop()
        if self.speaker:
            self.speaker.close()
        if self.tray:
            self.tray.stop()
        self.memory.close()

    # ---- event hooks
    def _on_state(self, old: State, new: State) -> None:
        self.overlay.send_state(new.value)
        if self.wake:
            (self.wake.resume if new is State.IDLE else self.wake.pause)()

    def _on_acting(self, active: bool) -> None:
        if active:
            self.sm.to(State.ACTING)
        else:
            self.sm.to(State.THINKING)

    def _on_out_level(self, level: float) -> None:
        self._out_level = level
        self.overlay.send_level(level)

    def _on_alert(self, text: str) -> None:
        if self.muted or self.paused:
            return
        self.say_direct(text)

    # ---- speech helpers
    def say_direct(self, text: str) -> None:
        if not self.speaker:
            log.warning("cannot speak (no TTS): %s", text)
            return
        if self.sm.state in (State.IDLE, State.SPEAKING):
            self.sm.to(State.SPEAKING)
        self._show("ev", text)
        self.speaker.say(text)
        threading.Thread(target=self._finish_speaking, daemon=True).start()

    def _finish_speaking(self) -> None:
        if self.speaker:
            self.speaker.wait_done(120)
        if self.sm.state is State.SPEAKING:
            self.sm.to(State.IDLE)

    # ---- confirmation (system-driven, never by the model)
    def confirm(self, readback: str) -> bool | None:
        self._confirming = True
        while not self._answers.empty():
            self._answers.get_nowait()
        try:
            self.say_direct(f"Just checking. Do you want me to {readback}? Say yes or no.")
            if self.speaker:
                self.speaker.wait_done(30)
            timeout = float(self.policy.d["defaults"]["confirm_timeout_s"])
            stop = threading.Event()

            def listen() -> None:
                try:
                    audio = self.recorder.record(self._blocks(self.cfg, stop.is_set), wait_for_speech_s=timeout)
                    if audio is not None and not stop.is_set():
                        said = self.stt.transcribe(audio).text
                        self._show("user", said)
                        ans = parse_yes_no(said)
                        if ans is not None:
                            self._answers.put(ans)
                except MicError:
                    log.error("mic unavailable for confirmation")

            th = threading.Thread(target=listen, daemon=True)
            th.start()
            try:
                return self._answers.get(timeout=timeout)
            except queue.Empty:
                return None
            finally:
                stop.set()
                self.recorder.cancel()
        finally:
            self._confirming = False

    # ---- kill switch
    def kill(self) -> None:
        t0 = time.perf_counter()
        self.gate.kill()
        self._abort.set()
        self.recorder.cancel()
        if self.speaker:
            self.speaker.stop()
        self._answers.put(False)
        self.sm.to(State.IDLE, force=True)
        log.warning("KILL SWITCH fired; stopped in %.0f ms", (time.perf_counter() - t0) * 1000)
        threading.Timer(1.0, self._rearm).start()

    def _rearm(self) -> None:
        self._abort.clear()
        self.gate.reset_kill()

    # ---- tray controls
    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.sm.to(State.PAUSED if self.paused else State.IDLE, force=True)

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        self.overlay.send_state("muted" if self.muted else "idle")

    def _start_tray(self) -> None:
        try:
            import pystray  # type: ignore[import-untyped]
            from PIL import Image, ImageDraw

            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((2, 18, 62, 46), radius=14, fill=(17, 17, 17, 255), outline=(90, 90, 90, 255))
            for i, h in enumerate((6, 12, 18, 12, 6)):
                x = 16 + i * 8
                d.rounded_rectangle((x, 32 - h // 2, x + 3, 32 + h // 2), radius=1, fill="white")
            menu = pystray.Menu(
                pystray.MenuItem("Pause", lambda *_: self.toggle_pause(), checked=lambda _i: self.paused),
                pystray.MenuItem("Mute mic", lambda *_: self.toggle_mute(), checked=lambda _i: self.muted),
                pystray.MenuItem("Quit", lambda *_: setattr(self, "_quit", True)),
            )
            self.tray = pystray.Icon("EV", img, "EV", menu)
            self.tray.run_detached()
        except Exception:
            log.exception("tray unavailable")

    def _blocks(self, cfg: Config, stop: Callable[[], bool]) -> Iterator[np.ndarray]:
        """Mic blocks for whoever is listening: the conversation's shared stream if one is open, else a stream of its own."""
        mic = self._mic
        return mic.blocks(stop) if mic is not None and mic.is_open else mic_blocks(cfg, stop)

    def _open_shared_mic(self) -> None:
        if not self.cfg.raw.get("conversation", {}).get("continuous_mic", False) or self._mic is not None:
            return
        mic = SharedMic(self.cfg)
        try:
            mic.open()
        except MicError as e:
            log.warning("continuous mic unavailable (%s); using one stream per turn", e)
            return
        self._mic = mic

    def _close_shared_mic(self) -> None:
        mic, self._mic = self._mic, None
        if mic is not None:
            mic.close()

    # ---- the core loop for one interaction
    def on_trigger(self) -> None:
        if self.paused or self.muted or self._confirming:
            return
        if self._following_up:                            # already listening for the user's next sentence
            return
        if self.speaker and self.speaker.is_speaking:     # barge-in
            self.speaker.stop()
        if not self._turn_lock.acquire(blocking=False):
            self.recorder.cancel()                        # second press during a turn cancels it
            return
        try:
            self._abort.clear()
            self._conversation()
        except Exception:
            log.exception("turn failed")
            self.sm.to(State.ERROR, force=True)
            self.say_direct("Sorry, something broke on my end.")
        finally:
            self._following_up = False
            self._turn_lock.release()
            if self.sm.state not in (State.SPEAKING, State.PAUSED):
                self.sm.to(State.IDLE, force=True)

    def _conversation(self) -> None:
        """One activation: the first turn, then follow-up turns while the user keeps talking."""
        conv = self.cfg["conversation"] if "conversation" in self.cfg.raw else {}
        max_turns = int(conv.get("followup_max_turns", 8))
        wait: float | None = 6.0
        self._open_shared_mic()                           # stays open across the turns below, closed before the wake word resumes
        try:
            for n in range(max_turns):
                self._following_up = n > 0
                wait = self._one_turn(wait or 6.0, followup=n > 0, turn=n)
                if wait is None or self._abort.is_set():
                    break
                if not self._interrupted.is_set():        # after an interruption the user is already talking
                    time.sleep(float(conv.get("turn_gap_ms", 300)) / 1000)   # let the speaker tail die so EV doesn't hear itself
        finally:
            self._close_shared_mic()
        self._following_up = False
        if self.sm.state is State.SPEAKING:               # conversation ended right after a reply; wake word resumes on IDLE
            self.sm.to(State.IDLE, force=True)

    def _followup_wait(self, heard: str, reply: str, turn: int = 0) -> float | None:
        """Seconds to keep listening after this reply, or None to end the conversation.

        The deeper into a back-and-forth, the more patient EV is (followup_patience_s per turn, up to
        followup_window_max_s): people in the middle of a conversation take longer to answer, and re-saying the wake
        word mid-flow is what makes an assistant feel like a walkie-talkie."""
        conv = self.cfg.raw.get("conversation", {})
        if not conv.get("followup_enabled", True) or _WRAP_UP.match(heard):
            return None
        base = float(conv.get("followup_question_window_s", 12) if reply.rstrip().endswith("?")
                     else conv.get("followup_window_s", 7))
        extra = float(conv.get("followup_patience_s", 0)) * turn
        return min(base + extra, max(base, float(conv.get("followup_window_max_s", base))))

    def _start_monitor(self) -> InterruptMonitor | None:
        conv = self.cfg.raw.get("conversation", {})
        if not conv.get("voice_interrupt", True) or not self.speaker:
            return None
        mon = InterruptMonitor(self.cfg, make_vad("silero", self.cfg["audio"]["vad_aggressiveness_threshold"]),
                               lambda: self._out_level, self._on_talk_over, self._blocks)
        mon.start()
        return mon

    def _on_talk_over(self, blocks: list[np.ndarray]) -> None:
        self._carry = blocks
        self._interrupted.set()
        if self.speaker:
            self.speaker.stop()

    def _one_turn(self, wait_s: float = 6.0, followup: bool = False, turn: int = 0) -> float | None:
        """Listen, think, speak. Returns the follow-up listening window in seconds, or None if we're done."""
        self.sm.to(State.LISTENING)
        self.brain.rewarm_if_cold()                       # reload the model while the user is still talking
        carried, self._carry = self._carry, []
        self._interrupted.clear()
        try:
            audio = self.recorder.record(itertools.chain(carried, self._blocks(self.cfg, lambda: False)),
                                         wait_for_speech_s=wait_s)
        except MicError:
            self.sm.to(State.ERROR, force=True)
            self.say_direct("I can't hear anything. I can't open the microphone.")
            return None
        if audio is None or self._abort.is_set():
            self.sm.to(State.IDLE, force=True)
            return None
        self.sm.to(State.TRANSCRIBING)
        text = self.stt.transcribe(audio).text
        log.info("heard%s: %s", " (follow-up)" if followup else "", text)
        if not text.strip() or self._abort.is_set() or (followup and _NOISE.match(text)):
            self.sm.to(State.IDLE, force=True)
            return None
        self._last_heard = text
        self._show("user", text)
        self.sm.to(State.THINKING)
        first = {"v": True}
        monitor: InterruptMonitor | None = None

        def on_sentence(s: str) -> None:
            nonlocal monitor
            if self._abort.is_set() or self._interrupted.is_set():
                return
            if first["v"]:
                first["v"] = False
                self.sm.to(State.SPEAKING, force=True)
                monitor = self._start_monitor()
            self._show("ev", s, append=True)
            if self.speaker:
                self.speaker.say(s)

        try:
            try:
                res = self.brain.respond(text, on_sentence=on_sentence,
                                         should_stop=lambda: self._interrupted.is_set() or self._abort.is_set())
            except LLMError:
                self.say_direct("I can't reach my brain right now.")
                return None
            log.info("reply%s: %s", " (interrupted)" if self._interrupted.is_set() else "", res.text)
            if self.speaker and not self._interrupted.is_set():
                self.speaker.wait_done(120)               # the monitor keeps listening meanwhile and can cut this short
        finally:
            if monitor:
                monitor.stop()                            # releases the mic before the recorder reopens it
        if self._interrupted.is_set() and not self._abort.is_set():
            log.info("interrupted by voice; listening")
            return float(self.cfg.raw.get("conversation", {}).get("followup_window_s", 7))
        if res.error or self._abort.is_set():
            self.sm.to(State.IDLE, force=True)
            return None
        return self._followup_wait(text, res.text, turn)
