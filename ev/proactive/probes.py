"""What the proactive engine looks at: how long the user has been idle, whether something is full-screen or a call is going,
which build tools are running, and which files appeared. Each is a plain function so the tests can swap them for fakes.
Nothing here ever acts on anything; it only reads."""
from __future__ import annotations

import ctypes
import glob
import os
import sys
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from ev.tools import windows

# SHQueryUserNotificationState values that mean "don't interrupt": a full-screen app (2), a full-screen Direct3D game (3), a presentation (4).
_BUSY_STATES = frozenset({2, 3, 4})
_MIC_KEY = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
_MAX_GLOB_MATCHES = 200


def idle_seconds() -> float | None:
    """Seconds since the last keyboard or mouse input, or None if Windows won't say."""
    if sys.platform != "win32":
        return None

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    info = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO), 0)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    return float((ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def notification_state() -> int | None:
    if sys.platform != "win32":
        return None
    state = ctypes.c_int(0)
    if ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state)) != 0:
        return None
    return int(state.value)


def fullscreen_busy() -> bool:
    """A full-screen app, game or presentation is in front. Unknown counts as 'not busy' (don't go silent forever on an error)."""
    return notification_state() in _BUSY_STATES


def mic_in_use_by_others() -> bool:
    """Another program (not EV's own Python) is using the microphone right now: a call. Windows records, per app, when the
    microphone was last started and stopped; an app that has started and not stopped is using it."""
    if sys.platform != "win32":
        return False
    import winreg

    def scan(root: int, path: str) -> bool:
        try:
            with winreg.OpenKey(root, path) as k:
                n = winreg.QueryInfoKey(k)[0]
                for i in range(n):
                    name = winreg.EnumKey(k, i)
                    if name == "NonPackaged" or "python" in name.lower():       # EV itself listens through python(w).exe
                        continue
                    with winreg.OpenKey(k, name) as app:
                        try:
                            start = winreg.QueryValueEx(app, "LastUsedTimeStart")[0]
                            stop = winreg.QueryValueEx(app, "LastUsedTimeStop")[0]
                        except OSError:
                            continue
                        if start and not stop:
                            return True
        except OSError:
            return False
        return False

    return scan(winreg.HKEY_CURRENT_USER, _MIC_KEY) or scan(winreg.HKEY_CURRENT_USER, _MIC_KEY + r"\NonPackaged")


def foreground_exe() -> str:
    try:
        w = windows.foreground()
    except OSError:
        return ""
    return w.exe if w else ""


def battery() -> tuple[int, bool] | None:
    try:
        b = psutil.sensors_battery()
    except (AttributeError, OSError):
        return None
    return (int(b.percent), bool(b.power_plugged)) if b else None


_own_cache: set[int] | None = None


def own_pids() -> set[int]:
    """EV and the terminal it runs in. Walking every process to find them costs ~80 ms, so it is done once."""
    global _own_cache
    if _own_cache is None:
        try:
            _own_cache = windows.own_pids()
        except OSError:
            _own_cache = {os.getpid()}
    return _own_cache


def processes(names: tuple[str, ...], own: set[int]) -> dict[int, tuple[str, float]]:
    """{pid: (name, started at epoch seconds)} for running processes whose name is one of `names`, excluding EV's own."""
    out: dict[int, tuple[str, float]] = {}
    for p in psutil.process_iter(["pid", "name", "create_time"]):
        try:
            stem = (p.info["name"] or "").lower().removesuffix(".exe")
            if stem in names and p.info["pid"] not in own:
                out[int(p.info["pid"])] = (stem, float(p.info["create_time"]))
        except (psutil.Error, OSError):
            continue
    return out


def files(pattern: str) -> dict[str, tuple[float, int]]:
    """{path: (modified time, size)} for files matching a glob (~ expanded), at most 200, folders skipped."""
    out: dict[str, tuple[float, int]] = {}
    for p in glob.iglob(os.path.expanduser(pattern)):
        try:
            st = os.stat(p)
        except OSError:
            continue
        if Path(p).is_file():
            out[p] = (st.st_mtime, st.st_size)
        if len(out) >= _MAX_GLOB_MATCHES:
            break
    return out


@dataclass
class Probes:
    """Everything the engine reads. The defaults are the real thing; tests pass fakes."""

    idle_s: Callable[[], float | None] = idle_seconds
    fullscreen: Callable[[], bool] = fullscreen_busy
    in_call: Callable[[], bool] = mic_in_use_by_others
    foreground_exe: Callable[[], str] = foreground_exe
    battery: Callable[[], tuple[int, bool] | None] = battery
    processes: Callable[[tuple[str, ...]], dict[int, tuple[str, float]]] = field(
        default=lambda names: processes(names, own_pids()))
    files: Callable[[str], dict[str, tuple[float, int]]] = files
