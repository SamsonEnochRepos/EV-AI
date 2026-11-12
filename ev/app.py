"""EV application: wires triggers, audio, STT, brain, TTS, overlay, tray into the core loop."""
from __future__ import annotations

import itertools
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ev.agents.listener import AgentListener, load_or_create_token
from ev.agents.rules import Rules
from ev.agents.supervisor import AgentSupervisor
from ev.audio.capture import MicError, Recorder, SharedMic, mic_blocks
from ev.audio.endpoint import looks_unfinished
from ev.audio.interrupt import InterruptMonitor
from ev.audio.playback import Player
from ev.audio.trigger import Hotkeys, WakeWord
from ev.audio.vad import make_vad
from ev.core.config import Config
from ev.core.events import EventBus, State
from ev.core.log import get_logger, setup_logging
from ev.core.state import StateMachine
from ev.dashboard.hub import Hub
from ev.dashboard.server import DashboardServer
from ev.llm.brain import Brain
from ev.llm.client import LLMError, OllamaClient, OllamaDown
from ev.llm.fastpath import is_stop
from ev.memory.migrations import open_vault, seed_from_persona
from ev.memory.store import MemoryStore
from ev.overlay.client import OverlayClient
from ev.safety.gate import SafetyGate, parse_yes_no
from ev.safety.policy import Policy
from ev.safety.provenance import TurnState
from ev.stt.whisper import STT, Transcript, looks_like_junk
from ev.tools.base import ToolRegistry
from ev.tools.builtin import ToolContext, build_tools
from ev.tools.scheduler import Scheduler
from ev.tts.base import make_engine
from ev.tts.speaker import Speaker

log = get_logger("app")

# Whisper tends to invent these from room noise; in a follow-up window they mean "nobody spoke".
_NOISE = re.compile(r"^\W*(you|bye|(?:u+|a+|e+)?[hm]+|um+|uh+|er+|thanks for watching|thank you for watching)\W*$", re.I)
# The user is wrapping up: answer, but don't keep the mic open afterwards.
# Talking over EV counts only when it starts with EV's name ("EV, don't open it", "oh Evie, wait"): anything else is a side
# conversation, the TV, or a murmur. Whisper spells the name several ways, so the likely ones are all accepted.
_ADDRESSED = re.compile(
    r"^\W*(?:(?:hey|oh|ok(?:ay)?|um|uh|so|well|yeah|wait|hold on|sorry)[, ]+){0,2}"
    r"(?:evie|evy|evi|eevee|eevie|eve|ev|e\.? ?v\.?)\b", re.I)
# Whisper often does not write the name down ("Eveled", "Eevee", "Evvy"). A first word that sounds like it still counts, unless it is
# an ordinary word that merely starts the same way. As a second opinion the wake-word model, which was trained on the name, listens to
# the same audio (`_name_score`).
_FIRST_WORD = re.compile(r"^\W*(?:(?:hey|oh|ok(?:ay)?|um|uh|so|well|yeah|wait|hold on|sorry)[, ]+){0,2}(?P<w>[a-z']+)", re.I)
_SOUNDS_LIKE_EV = re.compile(r"^(?:ev|eev|eve|evi|evy|ee|yev)[a-z]{0,5}$")
_NOT_THE_NAME = frozenset(
    "even evenly ever every everyone everybody everything everywhere event events eventually evening evil evidence evident evict "
    "evade evoke evolve evolved evolution evaluate evaporate".split())
_WRAP_UP = re.compile(r"^\W*(no,? )?(thanks|thank you|that'?s all|that'?s it|nothing|nope|never ?mind|goodbye|bye|stop)\b", re.I)


class EVApp:
    def __init__(self, cfg: Config, debug: bool = False):
        self.cfg = cfg
        setup_logging(cfg.path("general", "log_dir"), debug, cfg.get("general", "log_conversations", True))
        self.bus = EventBus()
        self.sm = StateMachine(self.bus)
        self.memory = MemoryStore(cfg.data_path(cfg["memory"]["db_file"]))
        seed_from_persona(self.memory, cfg.path("memory", "seed_file"))
        self.policy = Policy.load(cfg.path("general", "policy_file"), places=cfg.places())
        self.gate = SafetyGate(self.policy, ask=self.confirm)
        self.overlay = OverlayClient(cfg)
        self.client = OllamaClient(cfg)
        self.scheduler = Scheduler(self.memory, self._on_alert)
        self.turn = TurnState()
        self.vault, self.index = open_vault(cfg, self.memory, self.policy)
        self.registry: ToolRegistry = build_tools(ToolContext(cfg, self.memory, self.scheduler, self.policy,
                                                              vault=self.vault, index=self.index, turn=self.turn))
        # display mode: the pill shows a text card with what the user said and what EV answers
        self.display_mode = self.memory.get_kv("display_mode", "1" if cfg["overlay"].get("display_mode") else "0") == "1"
        self._last_heard = ""
        self.overlay.send_display(self.display_mode)
        self._register_display_tool()
        self._register_screen_tools()
        self.brain = Brain(cfg, self.client, self.memory, self.registry, self.gate, on_acting=self._on_acting,
                           vault=self.vault, index=self.index, turn=self.turn)
        self.brain.on_facts_changed = self.brain.refresh_prompt_cache      # a fact changed: re-prime the prompt cache once
        self.stt = STT(cfg)
        self.vad = make_vad("silero", cfg["audio"]["vad_aggressiveness_threshold"])
        self.recorder = Recorder(cfg, self.vad, on_level=self._on_level)
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
        self._name_model: object | None = None            # a second wake-word model, used only to hear whether EV was called by name
        self._name_lock = threading.Lock()
        self._since: int | None = None                    # where on the shared microphone the next recording picks up (after a barge-in)
        self._mic_permanent = False                       # the shared microphone stays open all the time (the wake word listens on it)
        self._held = ""                                   # a request cut off by the user speaking again before EV did or ran anything
        self._mic: SharedMic | None = None                # one stream for the whole conversation (conversation.continuous_mic)
        self._out_level = 0.0
        self._level_pub_at = 0.0
        self.hotkeys = Hotkeys(cfg, {
            "push_to_talk": self.on_trigger, "kill_switch": self.kill,
            "confirm_yes": lambda: self._answers.put(True), "confirm_no": lambda: self._answers.put(False),
        })
        self.wake = WakeWord(cfg, self.on_trigger) if cfg["trigger"]["mode"] == "wakeword" else None
        self.tray: Any = None
        self.bus.subscribe("state", self._on_state)
        # the dashboard: a window onto the same events the pill gets, plus a server that only runs if [dashboard] enabled
        self.hub = Hub()
        self.bus.subscribe("state", lambda old, new: self.hub.publish("state", old=old.value, new=new.value))
        self.brain.on_tool_result = lambda e: self.hub.publish("tool", **e)
        self.dashboard: DashboardServer | None = None
        self.agents, self.agent_listener = self._build_agents()
        self._register_dashboard_tool()

    # ---- optional tools (screen/control) registered lazily so lightweight setups still start
    def _register_screen_tools(self) -> None:
        try:
            from ev.control.tools import register_control_tools
            from ev.screen.tools import register_screen_tools

            register_screen_tools(self.registry, self.cfg, self.policy, on_artifact=self.memory.set_last_artifact)
            register_control_tools(self.registry, self.cfg, self.policy, self.gate)
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
            {"enabled": {"type": "boolean"}}, set_display_mode, ("enabled",), direct=True, llm_visible=False))

    # ---- dashboard: a loopback page, opened on request with a one-time sign-in link
    def _register_dashboard_tool(self) -> None:
        from ev.tools.base import Tool, ToolResult

        def open_dashboard() -> ToolResult:
            if not self.open_dashboard():
                return ToolResult(False, "The dashboard is turned off.")
            return ToolResult(True, "Opening the dashboard.")

        self.registry.register(Tool("open_dashboard", "Open the dashboard in the browser.", {}, open_dashboard, (),
                                    direct=True, llm_visible=False))

    # ---- coding agents (Claude Code, Codex) asking for approval: ev/agents
    def _build_agents(self) -> tuple[AgentSupervisor | None, AgentListener | None]:
        ac = self.cfg.raw.get("agents", {})
        if not ac.get("enabled", False):
            return None, None
        from ev.tools.base import Tool, ToolResult

        rules = Rules.load(self.cfg.root / str(ac.get("rules_file", "config/agent_rules.toml")),
                           strict_project_code=bool(ac.get("strict_project_code", False)))
        sup = AgentSupervisor(rules, self.cfg.data_path("agent_audit.jsonl"), ask=self._agent_ask, notify=self.overlay.send_notice,
                              publish=lambda e: self.hub.publish("agent", **e), ask_wait_s=float(ac.get("ask_wait_s", 25)))
        token = load_or_create_token(self.cfg.data_path("agent_token"))
        if ac.get("advisor", False):
            from ev.agents.advisor import Advisor

            sup.advisor = Advisor()

        def agents_control(action: str, minutes: float = 60.0) -> ToolResult:
            if action == "pause":
                sup.pause(minutes)
                return ToolResult(True, f"Okay, I'll stop answering the coding agents for {_span(minutes)}. Their own prompts will show.")
            if action == "resume":
                sup.pause(0)
                return ToolResult(True, "Okay, I'm answering the coding agents again.")
            return ToolResult(True, sup.status() + (" Approvals are paused." if sup.paused else ""))

        def agents_allow(group: str, minutes: float = 60.0) -> ToolResult:
            if not sup.last_cwd:
                return ToolResult(False, "I haven't seen a coding agent ask for anything yet, so I don't know which project you mean.")
            project = Path(sup.pinned_cwd or sup.last_cwd).name           # the one named in the question, not whoever asked since
            label = sup.grant(group, minutes)
            return ToolResult(True, f"Okay, the agents can {label} in {project} for {_span(minutes)}.")

        self.registry.register(Tool("agents_control", "Pause, resume or report on answering coding agents' approval requests.", {
            "action": {"type": "string", "enum": ["pause", "resume", "status"]},
            "minutes": {"type": "number", "minimum": 1, "maximum": 720}}, agents_control, ("action",), direct=True, llm_visible=False))
        self.registry.register(Tool("agents_allow", "Let the coding agents do one kind of thing in their current project for a while.", {
            "group": {"type": "string", "enum": sorted(rules.groups)},
            "minutes": {"type": "number", "minimum": 1, "maximum": 240}}, agents_allow, ("group",), direct=True, llm_visible=False))
        def describe_allow(a: dict[str, Any]) -> str:
            project = Path(sup.pin_project()).name or "their current project"        # the project is fixed here, as the question is asked
            what = rules.groups.get(str(a.get("group")), {}).get("label", "do that")
            return f"let the coding agents {what} in {project} for {_span(float(a.get('minutes', 60)))}"

        self.gate.describers["agents_allow"] = describe_allow
        return sup, AgentListener(sup, token, int(ac.get("port", 47831)))

    def _agent_ask(self, readback: str) -> bool | None:
        """A coding agent's request fell between the rules: ask out loud, but never in the middle of a conversation."""
        if self.paused or self.muted or not self._turn_lock.acquire(timeout=20):
            return None
        try:
            return self.confirm(readback)
        finally:
            self._turn_lock.release()

    def _start_dashboard(self) -> None:
        dc = self.cfg.raw.get("dashboard", {})
        if not dc.get("enabled", False) or self.dashboard is not None:
            return
        from ev.dashboard.backend import AppBackend

        try:
            server = DashboardServer(AppBackend(self), self.hub, port=int(dc.get("port", 0)))
            server.start()
        except OSError as e:
            log.error("dashboard could not start: %s", e)
            return
        self.dashboard = server

    def open_dashboard(self) -> bool:
        """Open the dashboard in the default browser through a one-time link. False if it is off."""
        if self.dashboard is None:
            return False
        import webbrowser

        webbrowser.open(self.dashboard.login_url())
        return True

    def _show(self, who: str, text: str, append: bool = False) -> None:
        """Put a line of the conversation on the pill's text card (only does anything in display mode)."""
        if self.display_mode:
            self.overlay.send_text(who, text, append)

    # ---- lifecycle
    def start(self) -> None:
        t0 = time.perf_counter()
        self.overlay.start()
        self._init_speaker()
        self._select_model()
        if self.index:
            self.index.start()                                  # indexes the vault now, then every vault.poll_s
        if self.vault and (n := len(self.vault.unconfirmed_facts())):
            log.warning("%d unconfirmed fact(s) in the vault; say \"review my unconfirmed facts\" to go through them", n)
        self.scheduler.start()
        self._start_dashboard()
        if self.agent_listener:
            self.agent_listener.start()
        self.hotkeys.start()
        if self.wake:
            self._open_permanent_mic()
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
        if self.wake:
            jobs.append(("name model", self._warm_name_model))
        threads = [threading.Thread(target=run, args=job, name=f"warm-{job[0]}", daemon=True) for job in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _warm_name_model(self) -> None:
        self._name_score([np.zeros(1280, np.int16)])      # loads the model now, not in the middle of someone's interruption

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
        self._mic_permanent = False
        self._close_shared_mic()
        self.scheduler.stop()
        if self.dashboard:
            self.dashboard.stop()
        if self.agent_listener:
            self.agent_listener.stop()
        self.overlay.stop()
        if self.speaker:
            self.speaker.close()
        if self.tray:
            self.tray.stop()
        if self.index:
            self.index.close()
        self.memory.close()

    # ---- event hooks
    def _on_state(self, old: State, new: State) -> None:
        self.overlay.send_state(new.value)
        if self.wake:
            (self.wake.resume if new is State.IDLE else self.wake.pause)()

    def _on_acting(self, active: bool) -> None:
        if self.sm.state is State.IDLE:
            return          # started by the dashboard: no turn is running, so don't fake one (IDLE -> THINKING would stick)
        if active:
            self.sm.to(State.ACTING)
        else:
            self.sm.to(State.THINKING)

    def _on_level(self, level: float) -> None:
        """Mic or speaker level: to the island, and (throttled, only while a dashboard is open) to the dashboard's orb."""
        self.overlay.send_level(level)
        now = time.monotonic()
        if now - self._level_pub_at >= 1 / 15 and self.hub.watchers:
            self._level_pub_at = now
            self.hub.publish("level", v=round(float(level), 2))

    def _on_out_level(self, level: float) -> None:
        self._out_level = level
        self._on_level(level)

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
                pystray.MenuItem("Dashboard", lambda *_: self.open_dashboard(), enabled=lambda _i: self.dashboard is not None),
                pystray.MenuItem("Pause", lambda *_: self.toggle_pause(), checked=lambda _i: self.paused),
                pystray.MenuItem("Mute mic", lambda *_: self.toggle_mute(), checked=lambda _i: self.muted),
                pystray.MenuItem("Quit", lambda *_: setattr(self, "_quit", True)),
            )
            self.tray = pystray.Icon("EV", img, "EV", menu)
            self.tray.run_detached()
        except Exception:
            log.exception("tray unavailable")

    def _blocks(self, cfg: Config, stop: Callable[[], bool], since: int | None = None) -> Iterator[np.ndarray]:
        """Mic blocks for whoever is listening: the shared stream if one is open (from `since` on, if given), else a stream of its own."""
        mic = self._mic
        return mic.blocks(stop, since=since) if mic is not None and mic.is_open else mic_blocks(cfg, stop)

    def _mark_now(self) -> int | None:
        mic = self._mic
        return mic.mark() if mic is not None and mic.is_open else None

    def _open_permanent_mic(self) -> None:
        """With the wake word on, keep ONE microphone stream open and let the wake word listen on it. The stream was open anyway
        (the wake word held its own); the difference is that the recorder can now start from the moment the wake word fired
        instead of from whenever a second stream finished opening, which lost the first words of "Evie, open the dashboard"."""
        if self.wake is None or not self.cfg.raw.get("conversation", {}).get("continuous_mic", True) or self._mic is not None:
            return
        mic = SharedMic(self.cfg)
        try:
            mic.open()
        except MicError as e:
            log.warning("shared microphone unavailable (%s); the wake word opens its own stream", e)
            return
        self._mic, self._mic_permanent = mic, True
        self.wake.source = mic

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
        if self._mic_permanent:
            return                                        # the wake word is listening on it
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
        if conv.get("interrupt_needs_name", True):
            mon.verify = self._addressed_to_ev
        mon.start()
        return mon

    def _addressed_to_ev(self, blocks: list[np.ndarray]) -> bool:
        """Was the speech heard over EV addressed to it? Only then does it stop talking."""
        try:
            said = self.stt.transcribe(np.concatenate(blocks)).text
        except Exception:                                  # noqa: BLE001  (no speech model: nothing can interrupt by voice)
            log.exception("could not check whether EV was addressed")
            return False
        why = ""
        ok = bool(_ADDRESSED.match(said)) or is_stop(said)          # a bare "stop" / "be quiet" is never anything but a command
        if ok:
            why = "starts with its name"
        else:
            m = _FIRST_WORD.match(said)
            word = m.group("w").lower() if m else ""
            if word in _NOT_THE_NAME:
                why = ""
            elif _SOUNDS_LIKE_EV.match(word):
                ok, why = True, f"{word!r} sounds like its name"
            else:
                score = self._name_score(blocks)
                if score >= float(self.cfg.raw.get("conversation", {}).get("interrupt_name_score", 0.5)):
                    ok, why = True, f"the wake-word model heard its name ({score:.2f})"
        log.info("heard over EV: %r -> %s", said, f"addressed to EV ({why}), stopping" if ok else "not for EV, carrying on")
        return ok

    def _name_score(self, blocks: list[np.ndarray]) -> float:
        """How much the wake-word model thinks EV's name is in this audio (0..1). It was trained on the name, so it can hear "Evie"
        when the speech model writes something else down. 0 when there is no wake-word model (hotkey mode)."""
        if self.wake is None:
            return 0.0
        try:
            with self._name_lock:
                if self._name_model is None:
                    self._name_model = self.wake._load()
                model: Any = self._name_model
                pcm = np.concatenate([np.concatenate(blocks).astype(np.int16)[: 2 * 16000], np.zeros(8000, np.int16)])
                model.reset()
                top = 0.0
                for i in range(0, len(pcm) - 1279, 1280):
                    top = max(top, max(model.predict(pcm[i:i + 1280]).values(), default=0.0))
                return float(top)
        except Exception:                                  # noqa: BLE001  (no model: the transcript alone decides)
            log.warning("could not score the name with the wake-word model", exc_info=True)
            return 0.0

    def _on_talk_over(self, blocks: list[np.ndarray], resume_from: int | None = None) -> None:
        self._carry = blocks
        self._since = resume_from
        self._interrupted.set()
        if self.speaker:
            self.speaker.stop()

    def _transcribe_whole_sentence(self, audio: np.ndarray) -> Transcript:
        """Transcribe, and if what was said stops on a word no sentence ends on ("open the", "can you", "and"), keep the mic
        open a little longer for the rest and transcribe the two parts together. A pause is not the end of a sentence."""
        conv = self.cfg.raw.get("conversation", {})
        rounds, window = int(conv.get("finish_rounds", 2)), float(conv.get("finish_window_s", 1.8))
        mark = self._mark_now()                           # what is said while this is being transcribed is kept on the shared mic
        heard = self.stt.transcribe(audio)
        for _ in range(max(0, rounds) if mark is not None else 0):
            if self._abort.is_set() or not looks_unfinished(heard.text):
                break
            log.info("sounds unfinished (%r); listening %.1f s more", heard.text, window)
            self.sm.to(State.LISTENING)
            more = self.recorder.record(self._blocks(self.cfg, lambda: False, mark), wait_for_speech_s=window)
            self.sm.to(State.TRANSCRIBING)
            if more is None:
                break
            audio = np.concatenate([audio, more])
            mark = self._mark_now()
            heard = self.stt.transcribe(audio)
        return heard

    def _one_turn(self, wait_s: float = 6.0, followup: bool = False, turn: int = 0) -> float | None:
        """Listen, think, speak. Returns the follow-up listening window in seconds, or None if we're done."""
        self.sm.to(State.LISTENING)
        self.brain.rewarm_if_cold()                       # reload the model while the user is still talking
        carried, self._carry = self._carry, []
        since, self._since = self._since, None
        if since is None and not followup and self.wake is not None:
            since = self.wake.take_mark()                 # the wake word fired here: start listening from just before this moment
        held, self._held = self._held, ""                 # a request the user cut off while EV was still thinking
        self._interrupted.clear()
        try:
            audio = self.recorder.record(itertools.chain(carried, self._blocks(self.cfg, lambda: False, since)),
                                         wait_for_speech_s=wait_s)
        except MicError:
            self.sm.to(State.ERROR, force=True)
            self.say_direct("I can't hear anything. I can't open the microphone.")
            return None
        if self._abort.is_set() or (audio is None and not held):
            self.sm.to(State.IDLE, force=True)
            return None
        text = held
        if audio is not None:
            self.sm.to(State.TRANSCRIBING)
            rec = dict(self.recorder.last)
            heard = self._transcribe_whole_sentence(audio)
            text = heard.text
            log.info("heard%s: %s   [%.1f s, longest pause %d ms, ended on %s]", " (follow-up)" if followup else "", text,
                     float(rec.get("seconds", 0)), int(rec.get("longest_pause_ms", 0)), rec.get("ended", "?"))
            # With a held request the new words only need to be real words: "Australia" alone is a short, shaky transcript,
            # but it finishes "what's the capital of".
            if self._abort.is_set():
                self.sm.to(State.IDLE, force=True)
                return None
            junk = not text.strip() if held else looks_like_junk(heard)
            if junk or (followup and _NOISE.match(text) and not held):
                if text.strip():
                    log.info("ignored as noise (logprob %.2f, no-speech %.2f)", heard.avg_logprob, heard.no_speech_prob)
                if not held:
                    self.sm.to(State.IDLE, force=True)
                    return None
                text = held                              # nothing usable came after all: answer what was asked
            elif held:
                text = f"{held} {text}".strip()
                log.info("merged with the request it continues: %s", text)
        else:
            log.info("nothing more was said; answering the held request: %s", held)
        if is_stop(text):                                 # "Evie stop": silence, not a question for the model (it used to answer with nonsense)
            log.info("stop command (%r): staying quiet", text)
            if self.speaker:
                self.speaker.stop()
            self.sm.to(State.IDLE, force=True)
            return None
        self._last_heard = text
        self._show("user", text)
        self.sm.to(State.THINKING)
        first = {"v": True}
        # Listening while EV thinks (conversation.listen_while_thinking): the user speaking again before EV has said or done
        # anything cancels this turn and joins it to what they say next.
        monitor: InterruptMonitor | None = (
            self._start_monitor() if self.cfg.raw.get("conversation", {}).get("listen_while_thinking", False) else None)

        def on_sentence(s: str) -> None:
            nonlocal monitor
            if self._abort.is_set() or self._interrupted.is_set():
                return
            if first["v"]:
                first["v"] = False
                self.sm.to(State.SPEAKING, force=True)
                monitor = monitor or self._start_monitor()
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
            self.hub.publish("turn", user=text, reply=res.text, interrupted=self._interrupted.is_set())
            if self.speaker and not self._interrupted.is_set():
                self.speaker.wait_done(120)               # the monitor keeps listening meanwhile and can cut this short
        finally:
            if monitor:
                monitor.stop()                            # releases the mic before the recorder reopens it
        if self._interrupted.is_set() and not self._abort.is_set():
            if first["v"] and not res.tool_log and not res.error:
                self._held = text                         # EV had said and done nothing yet: this request waits for the rest
                log.info("interrupted before EV answered; holding %r for what comes next", text)
            log.info("interrupted by voice; listening")
            return float(self.cfg.raw.get("conversation", {}).get("followup_window_s", 7))
        if res.error or self._abort.is_set():
            self.sm.to(State.IDLE, force=True)
            return None
        return self._followup_wait(text, res.text, turn)


def _span(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60} hour{'s' if m // 60 != 1 else ''}" if m >= 60 and m % 60 == 0 else f"{m} minute{'s' if m != 1 else ''}"
