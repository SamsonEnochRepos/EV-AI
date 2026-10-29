"""Triggers: global hotkeys (push-to-talk, kill switch, confirm yes/no) and optional wake word."""
from __future__ import annotations

import threading
from collections.abc import Callable
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
    """Optional openWakeWord listener. Pluggable: any model name or .onnx path in ev.toml."""

    def __init__(self, cfg: Config, on_wake: Callable[[], None]):
        self.cfg = cfg
        self.on_wake = on_wake
        self.threshold = float(cfg["trigger"]["wakeword_threshold"])
        self.model_name = cfg["trigger"]["wakeword_model"]
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

    def _run(self) -> None:
        try:
            import openwakeword  # type: ignore[import-untyped]
            import sounddevice as sd  # type: ignore[import-untyped]
            from openwakeword.model import Model  # type: ignore[import-untyped]
        except ImportError:
            log.error("wake word requires: pip install openwakeword")
            return
        try:
            openwakeword.utils.download_models([self.model_name])
            model = Model(wakeword_models=[self.model_name], inference_framework="onnx")
        except Exception as e:
            log.error("wake word model failed: %s", e)
            return
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.2)
                continue
            with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1280) as s:
                while not self._stop.is_set() and not self._paused.is_set():
                    data, _ = s.read(1280)
                    scores = model.predict(np.asarray(data[:, 0]))
                    if any(v >= self.threshold for v in scores.values()):
                        model.reset()
                        self.on_wake()
                        break
