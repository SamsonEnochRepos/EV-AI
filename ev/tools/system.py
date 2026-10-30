"""Windows system helpers: media keys, volume, app launching."""
from __future__ import annotations

import ctypes
import difflib
import json
import os
import re
import shutil
import subprocess  # noqa: S404
import sys
import time
from pathlib import Path
from typing import Any

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
    "vscode": ["Code.exe"], "vs code": ["Code.exe"], "visual studio code": ["Code.exe"], "code": ["Code.exe"],
    "word": ["winword.exe"], "excel": ["excel.exe"], "powerpoint": ["powerpnt.exe"],
}

_APP_PATHS = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
_FUZZY_CUTOFF = 0.85          # tolerate speech-recognition slips ("spotifi", "note pad") but not unrelated names


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


def resolve_exe(exe: str) -> str | None:
    """Find an executable the way Windows' Run box does: PATH first, then the registered App Paths."""
    found = shutil.which(exe)
    if found:
        return found
    if sys.platform != "win32":
        return None
    import winreg

    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, rf"{_APP_PATHS}\{exe}") as k:
                target = os.path.expandvars(str(winreg.QueryValueEx(k, "")[0])).strip('"')
        except OSError:
            continue
        if target and Path(target).exists():
            return target
    return None


def _start_menu_dirs() -> list[Path]:
    dirs = []
    for env in ("ProgramData", "APPDATA"):
        base = os.environ.get(env)
        if base:
            dirs.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


# Words Whisper reliably mishears in app names ("Cloud Desktop" -> Claude Desktop). Applied only as a second guess.
_ASR_FIXES = {"cloud": "claude", "claud": "claude", "clod": "claude", "clawed": "claude", "claudes": "claude"}


def _variants(name: str) -> tuple[str, str | None]:
    base = _norm(name)
    fixed = " ".join(_ASR_FIXES.get(w, w) for w in base.split())
    return base, (fixed if fixed != base else None)


def _match(entries: dict[str, Any], want: str, loose: bool) -> Any | None:
    """Exact name first; with loose=True also whole-word containment ("claude desktop" -> "Claude"), then fuzzy."""
    if not want:
        return None
    if want in entries:
        return entries[want]
    if not loose:
        return None
    wt = set(want.split())
    contained = [s for s in entries if len(s) >= 4 and (set(s.split()) <= wt or wt <= set(s.split()))]
    if contained:
        return entries[min(contained, key=len)]          # "chrome" -> "google chrome", not a longer sibling
    close = difflib.get_close_matches(want, list(entries), n=1, cutoff=_FUZZY_CUTOFF)
    return entries[close[0]] if close else None


def _lnk_entries() -> dict[str, Path]:
    entries: dict[str, Path] = {}
    for d in _start_menu_dirs():
        if d.exists():
            for lnk in d.rglob("*.lnk"):
                stem = _norm(lnk.stem)
                if stem and "uninstall" not in stem:
                    entries.setdefault(stem, lnk)
    return entries


_store_cache: tuple[float, dict[str, tuple[str, str]]] = (0.0, {})


def _store_entries() -> dict[str, tuple[str, str]]:
    """Microsoft Store / UWP apps have no .lnk; ask Windows for its Start apps (cached for five minutes)."""
    global _store_cache
    if sys.platform != "win32":
        return {}
    if time.monotonic() - _store_cache[0] < 300 and _store_cache[1]:
        return _store_cache[1]
    try:
        argv = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "Get-StartApps | Select-Object Name,AppID | ConvertTo-Json -Compress"]   # fixed argv, no user text
        out = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, timeout=12, check=False, creationflags=0x08000000)
        rows = json.loads(out.stdout or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    apps = {_norm(r["Name"]): (r["Name"], r["AppID"]) for r in rows if r.get("Name") and r.get("AppID")}
    _store_cache = (time.monotonic(), apps)
    return apps


def find_start_menu_entry(name: str, loose: bool = True) -> Path | None:
    base, fixed = _variants(name)
    for want in ((fixed, base) if loose and fixed else (base, fixed)):
        hit = _match(_lnk_entries(), want or "", loose)
        if hit:
            return hit  # type: ignore[no-any-return]
    return None


def find_store_app(name: str, loose: bool = True) -> tuple[str, str] | None:
    base, fixed = _variants(name)
    for want in ((fixed, base) if loose and fixed else (base, fixed)):
        hit = _match(_store_entries(), want or "", loose)
        if hit:
            return hit  # type: ignore[no-any-return]
    return None


def _run_elevated(path: str, args: str | None = None) -> None:
    """Ask Windows to start `path` as administrator. Windows shows its own UAC prompt; EV can't answer it."""
    rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", path, args, None, 1)
    if rc <= 32:
        raise PermissionError("Windows didn't allow that, or the administrator prompt was declined")


def launch_app(name: str, allow_start_menu: bool = True, elevate: bool = False) -> str:
    """Launch an app by friendly name. Returns a short description of what was launched.

    elevate=True starts it as administrator; callers only pass it after the user has confirmed out loud."""
    key = name.strip().lower()
    if key in KNOWN_APPS:
        argv = KNOWN_APPS[key]
        if len(argv) > 1 and argv[1].endswith(":"):
            os.startfile(argv[1])  # type: ignore[attr-defined]  # noqa: S606
            return key
        exe = resolve_exe(argv[0])
        if exe:
            if elevate:
                _run_elevated(exe, " ".join(argv[1:]) or None)
            else:
                subprocess.Popen([exe, *argv[1:]], close_fds=True)  # noqa: S603
            return key
        # not installed under that exe name: fall through to Start-menu resolution
    if allow_start_menu:
        # An exact name anywhere beats a loose match anywhere ("cloud" must not open "Genshin Impact Cloud").
        for loose in (False, True):
            lnk = find_start_menu_entry(key, loose)
            if lnk:
                if elevate:
                    _run_elevated(str(lnk))
                else:
                    os.startfile(str(lnk))  # type: ignore[attr-defined]  # noqa: S606
                return lnk.stem
            store = find_store_app(key, loose)
            if store:
                subprocess.Popen(["explorer.exe", "shell:AppsFolder\\" + store[1]], close_fds=True)  # noqa: S603, S607  # AppID comes from Windows Get-StartApps
                return store[0]
    raise FileNotFoundError(f"I couldn't find an app called {name}")
