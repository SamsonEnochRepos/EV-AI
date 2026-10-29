"""Windows system helpers: media keys, volume, app launching."""
from __future__ import annotations

import ctypes
import os
import re
import subprocess  # noqa: S404
import sys
from pathlib import Path

VK = {
    "play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2,
    "volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
}
KEYEVENTF_KEYUP = 0x0002

# Fixed executables for common apps: resolved by name, never via a shell string.
KNOWN_APPS: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"], "calc": ["calc.exe"],
    "paint": ["mspaint.exe"],
    "explorer": ["explorer.exe"], "file explorer": ["explorer.exe"],
    "settings": ["explorer.exe", "ms-settings:"],
    "task manager": ["taskmgr.exe"],
    "terminal": ["wt.exe"], "windows terminal": ["wt.exe"],
    "edge": ["msedge.exe"], "microsoft edge": ["msedge.exe"],
    "chrome": ["chrome.exe"], "google chrome": ["chrome.exe"],
    "firefox": ["firefox.exe"],
}


def press_media_key(name: str, times: int = 1) -> None:
    if sys.platform != "win32":
        raise OSError("media keys need Windows")
    code = VK[name]
    for _ in range(times):
        ctypes.windll.user32.keybd_event(code, 0, 0, 0)
        ctypes.windll.user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)


def set_master_volume(percent: int) -> None:
    from comtypes import CLSCTX_ALL  # type: ignore[import-untyped]
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore[import-untyped]

    dev = AudioUtilities.GetSpeakers()
    iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    vol = ctypes.cast(iface, ctypes.POINTER(IAudioEndpointVolume))
    vol.SetMasterVolumeLevelScalar(max(0, min(100, percent)) / 100.0, None)


def _start_menu_dirs() -> list[Path]:
    dirs = []
    for env in ("ProgramData", "APPDATA"):
        base = os.environ.get(env)
        if base:
            dirs.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def find_start_menu_entry(name: str) -> Path | None:
    want = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
    best: Path | None = None
    for d in _start_menu_dirs():
        if not d.exists():
            continue
        for lnk in d.rglob("*.lnk"):
            stem = re.sub(r"[^a-z0-9 ]", "", lnk.stem.lower()).strip()
            if stem == want:
                return lnk
            if want and want in stem and best is None:
                best = lnk
    return best


def launch_app(name: str, allow_start_menu: bool = True) -> str:
    """Launch an app by friendly name. Returns a short description of what was launched."""
    key = name.strip().lower()
    if key in KNOWN_APPS:
        argv = KNOWN_APPS[key]
        if len(argv) > 1 and argv[1].endswith(":"):
            os.startfile(argv[1])  # type: ignore[attr-defined]  # noqa: S606
        else:
            subprocess.Popen(argv, close_fds=True)  # noqa: S603
        return key
    if allow_start_menu:
        lnk = find_start_menu_entry(key)
        if lnk:
            os.startfile(str(lnk))  # type: ignore[attr-defined]  # noqa: S606
            return lnk.stem
    raise FileNotFoundError(f"couldn't find an app called {name}")
