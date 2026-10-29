"""EV application: wires triggers, audio, STT, brain, TTS, overlay, tray into the core loop."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

from ev.audio.capture import MicError, Recorder, mic_blocks
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
        threading.Thread(target=self._warm, name="warm", daemon=True).start()
        if self.cfg["tray"]["enabled"]:
            self._start_tray()
        log.info("EV ready in %.1fs", time.perf_counter() - t0)

    def _warm(self) -> None:
        try:
            self.stt.load()
            self.client.warm_up()
        except Exception:
            log.warning("warm-up incomplete", exc_info=True)

    def _init_speaker(self) -> None:
        try:
            engine = make_engine(self.cfg)
            engine.load()
            player = Player(on_level=self._on_out_level, device=self.cfg["audio"]["output_device"] or None)
            self.speaker = Speaker(engine, player, on_error=lambda e: log.error("speech error: %s", e))
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
            (self.wake.pause if new not in (State.IDLE, State.PAUSED) else self.wake.resume)()

    def _on_acting(self, active: bool) -> None:
        if active:
            self.sm.to(State.ACTING)
        else:
            self.sm.to(State.THINKING)

    def _on_out_level(self, level: float) -> None:
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
                    audio = self.recorder.record(mic_blocks(self.cfg, stop.is_set), wait_for_speech_s=timeout)
                    if audio is not None and not stop.is_set():
                        ans = parse_yes_no(self.stt.transcribe(audio).text)
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

    # ---- the core loop for one interaction
    def on_trigger(self) -> None:
        if self.paused or self.muted or self._confirming:
            return
        if self.speaker and self.speaker.is_speaking:     # barge-in
            self.speaker.stop()
        if not self._turn_lock.acquire(blocking=False):
            self.recorder.cancel()                        # second press during a turn cancels it
            return
        try:
            self._abort.clear()
            self._one_turn()
        except Exception:
            log.exception("turn failed")
            self.sm.to(State.ERROR, force=True)
            self.say_direct("Sorry, something broke on my end.")
        finally:
            self._turn_lock.release()
            if self.sm.state not in (State.SPEAKING, State.PAUSED):
                self.sm.to(State.IDLE, force=True)

    def _one_turn(self) -> None:
        self.sm.to(State.LISTENING)
        try:
            audio = self.recorder.record(mic_blocks(self.cfg, lambda: False))
        except MicError:
            self.sm.to(State.ERROR, force=True)
            self.say_direct("I can't hear anything. I can't open the microphone.")
            return
        if audio is None or self._abort.is_set():
            self.sm.to(State.IDLE, force=True)
            return
        self.sm.to(State.TRANSCRIBING)
        text = self.stt.transcribe(audio).text
        log.info("heard: %s", text)
        if not text.strip() or self._abort.is_set():
            self.sm.to(State.IDLE, force=True)
            return
        self.sm.to(State.THINKING)
        first = {"v": True}

        def on_sentence(s: str) -> None:
            if self._abort.is_set():
                return
            if first["v"]:
                first["v"] = False
                self.sm.to(State.SPEAKING, force=True)
            if self.speaker:
                self.speaker.say(s)

        try:
            res = self.brain.respond(text, on_sentence=on_sentence)
            log.info("reply: %s", res.text)
        except LLMError:
            self.say_direct("I can't reach my brain right now.")
            return
        if self.speaker:
            self.speaker.wait_done(120)
        self.sm.to(State.IDLE, force=True)
