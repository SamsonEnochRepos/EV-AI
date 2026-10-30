"""Core-side overlay controller: spawns the overlay process and streams state/levels to it.

Failure isolation: any overlay error is swallowed (EV keeps working) and the process is
restarted lazily, rate-limited. If EV dies, the socket closes and the overlay exits.
"""
from __future__ import annotations

import socket
import subprocess  # noqa: S404
import sys
import threading
import time

from ev.core.config import Config
from ev.core.log import get_logger

from . import protocol

log = get_logger("overlay.client")

CREATE_NO_WINDOW = 0x08000000


def kill_tree(pid: int) -> None:
    """Kill a process and its children (venv python.exe is a launcher that spawns the real interpreter)."""
    import psutil

    try:
        root = psutil.Process(pid)
        procs = [*root.children(recursive=True), root]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass


class OverlayClient:
    def __init__(self, cfg: Config, restart_cooldown_s: float = 5.0):
        self.cfg = cfg
        self.enabled = bool(cfg["overlay"].get("enabled", True))
        self.cooldown = restart_cooldown_s
        self.proc: subprocess.Popen[str] | None = None
        self.sock: socket.socket | None = None
        self.token = ""
        self._lock = threading.Lock()
        self._last_start = 0.0
        self._last_level_sent = 0.0
        self._last_state = "idle"
        self._display = False

    # ---- lifecycle
    def start(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> bool:
        if time.monotonic() - self._last_start < self.cooldown:
            return False
        self._last_start = time.monotonic()
        self._teardown_locked()
        try:
            self.token = protocol.new_token()
            self.proc = subprocess.Popen(  # noqa: S603
                [sys.executable, "-m", "ev.overlay", "--serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            assert self.proc.stdin and self.proc.stdout
            self.proc.stdin.write(self.token + "\n")
            self.proc.stdin.flush()
            port = int(self.proc.stdout.readline().strip())
            s = socket.create_connection((protocol.LOOPBACK, port), timeout=3)
            s.settimeout(0.5)
            s.sendall(protocol.encode({"token": self.token}))
            self.sock = s
            s.sendall(protocol.encode({"t": "state", "v": self._last_state}))   # lock already held
            if self._display:                                                   # a restarted overlay keeps display mode
                s.sendall(protocol.encode({"t": "display", "v": True}))
            return True
        except Exception as e:
            log.warning("overlay failed to start: %s", e)
            self._teardown_locked()
            return False

    def _teardown_locked(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        if self.proc and self.proc.poll() is None:
            kill_tree(self.proc.pid)
        self.proc = None

    def stop(self) -> None:
        with self._lock:
            if self.sock:
                try:
                    self.sock.sendall(protocol.encode({"t": "quit"}))
                except OSError:
                    pass
            self._teardown_locked()

    # ---- sending
    def _send(self, msg: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.sock is None:
                if not self._start_locked():
                    return
            try:
                assert self.sock
                self.sock.sendall(protocol.encode(msg))
            except (OSError, AssertionError):
                self._teardown_locked()   # overlay died; EV carries on, restart later

    def send_state(self, state: str) -> None:
        self._last_state = state
        self._send({"t": "state", "v": state})

    def send_display(self, on: bool) -> None:
        """Turn the text card on or off. Remembered, so an overlay that restarts comes back in the same mode."""
        self._display = on
        if self.sock is not None:
            self._send({"t": "display", "v": on})

    def send_text(self, who: str, text: str, append: bool = False) -> None:
        """Show a line of conversation on the card. Does nothing unless display mode is on."""
        text = " ".join(text.split())
        if not self._display or not text:
            return
        if len(text) > protocol.MAX_TEXT:
            text = text[: protocol.MAX_TEXT - 1].rstrip() + "…"
        self._send({"t": "text", "who": who, "v": text, "append": append})

    def send_level(self, level: float) -> None:
        now = time.monotonic()
        if now - self._last_level_sent < 1 / 40:   # throttle to ~40 Hz
            return
        self._last_level_sent = now
        self._send({"t": "level", "v": round(float(level), 3)})
