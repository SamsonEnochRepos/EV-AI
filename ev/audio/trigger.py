"""Triggers: global hotkeys (push-to-talk, kill switch, confirm yes/no) and optional wake word."""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ev.core.config import Config
from ev.core.log import get_logger

log = get_logger("trigger")

_SPECIAL = {"ctrl", "alt", "shift", "cmd", "space", "enter", "esc", "tab", "up", "down", "left", "right", "delete"}


def to_pynput(combo: str) -> str:
    """'ctrl+alt+space' -> '<ctrl>+<alt>+<space>'."""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    return "+".join(f"<{p}>" if p in _SPECIAL or (len(p) > 1 and p.startswith("f") and p[1:].isdigit()) else p for p in parts)


class Hotkeys:
    def __init__(self, cfg: Config, handlers: dict[str, Callable[[], None]]):
        self.map = {to_pynput(cfg["hotkeys"][name]): fn for name, fn in handlers.items() if name in cfg["hotkeys"]}
        self._listener: Any = None

    def start(self) -> None:
        from pynput import keyboard  # type: ignore[import-untyped]

        def wrap(fn: Callable[[], None]) -> Callable[[], None]:
            def run() -> None:
                threading.Thread(target=fn, daemon=True).start()   # never block the hook thread
            return run

        self._listener = keyboard.GlobalHotKeys({k: wrap(v) for k, v in self.map.items()})
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()


class WakeWord:
    """Optional openWakeWord listener, fully offline.

    Models are read from `trigger.wakeword_dir` (fetch them once with
    `python scripts/download_models.py --wakeword hey_jarvis`); `trigger.wakeword_model` is a model name or a path
    to a custom .onnx. The mic stream is closed *before* the callback runs so the recorder owns the mic, and the
    listener stays paused until the whole turn is over so EV can't wake itself with its own session.
    """

    BLOCK = 1280                      # 80 ms at 16 kHz, the frame size openWakeWord expects
    COOLDOWN_S = 1.0

    def __init__(self, cfg: Config, on_wake: Callable[[], None]):
        self.cfg = cfg
        self.on_wake = on_wake
        t = cfg["trigger"]
        self.threshold = float(t["wakeword_threshold"])
        self.model_name = str(t["wakeword_model"])
        self.model_dir = cfg.path("trigger", "wakeword_dir") if "wakeword_dir" in t else cfg.root / "models" / "openwakeword"
        self.device = cfg["audio"].get("input_device") or None
        self.ready = threading.Event()
        self.error = ""
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None

    def pause(self) -> None:      # mic is released while EV is listening/speaking
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="wakeword", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def model_path(self) -> Path:
        p = Path(self.model_name)
        if p.suffix == ".onnx":
            return p if p.is_absolute() else self.cfg.root / p
        found = sorted(self.model_dir.glob(f"{self.model_name}*.onnx"))
        if not found:
            raise FileNotFoundError(
                f"no wake-word model '{self.model_name}' in {self.model_dir}; "
                f"run: python scripts/download_models.py --wakeword {self.model_name}")
        return found[0]

    def _load(self) -> Any:
        from openwakeword.model import Model  # type: ignore[import-untyped]

        return Model(
            wakeword_models=[str(self.model_path())], inference_framework="onnx",
            melspec_model_path=str(self.model_dir / "melspectrogram.onnx"),
            embedding_model_path=str(self.model_dir / "embedding_model.onnx"))

    def _fail(self, msg: str) -> None:
        self.error = msg
        log.error("wake word unavailable (hotkey still works): %s", msg)

    def _run(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-untyped]
        except ImportError as e:
            self._fail(f"{e}; install with: pip install openwakeword")
            return
        try:
            model = self._load()
        except ImportError:
            self._fail("openwakeword is not installed; install with: pip install openwakeword")
            return
        except Exception as e:
            self._fail(str(e))
            return
        log.info("wake word ready: say '%s' (threshold %.2f)", self.model_name.replace("_", " "), self.threshold)
        self.ready.set()
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.2)
                continue
            hit = False
            try:
                model.reset()
                with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                                    blocksize=self.BLOCK, device=self.device) as s:
                    while not self._stop.is_set() and not self._paused.is_set():
                        data, _ = s.read(self.BLOCK)
                        scores = model.predict(np.asarray(data[:, 0]))
                        top = max(scores.values(), default=0.0)
                        if top >= 0.2:
                            log.debug("wake score %.2f", top)
                        if top >= self.threshold:
                            hit = True
                            break
            except Exception as e:
                log.warning("wake word mic error: %s; retrying", e)
                self._stop.wait(2.0)
                continue
            if hit:
                self.pause()                                   # stay paused until the turn is over
                threading.Thread(target=self._fire, name="wake-turn", daemon=True).start()

    def _fire(self) -> None:
        try:
            log.info("wake word heard")
            self.on_wake()
        except Exception:
            log.exception("wake-triggered turn failed")
        finally:
            self._stop.wait(self.COOLDOWN_S)
            self.resume()
