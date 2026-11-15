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
    "cmd": ["cmd.exe"], "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe"], "windows powershell": ["powershell.exe"],
    "edge": ["msedge.exe"], "microsoft edge": ["msedge.exe"],
    "chrome": ["chrome.exe"], "google chrome": ["chrome.exe"],
    "firefox": ["firefox.exe"],
    "vscode": ["Code.exe"], "vs code": ["Code.exe"], "visual studio code": ["Code.exe"], "code": ["Code.exe"],
    "word": ["winword.exe"], "excel": ["excel.exe"], "powerpoint": ["powerpnt.exe"],
}

_CONSOLE_EXES = frozenset({"cmd.exe", "powershell.exe"})
_NEW_CONSOLE = 0x00000010      # CREATE_NEW_CONSOLE
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


_INDEX_TTL_S = 300
_lnk_cache: tuple[float, tuple[str, ...], dict[str, Path]] = (0.0, (), {})


def _lnk_entries() -> dict[str, Path]:
    """Start-menu shortcuts by name. The folder walk costs 50-300 ms and a launch asks several times, so it is
    cached for five minutes (and rebuilt at once if an app can't be found; see launch_app)."""
    global _lnk_cache
    dirs = _start_menu_dirs()
    key = tuple(str(d) for d in dirs)
    if _lnk_cache[1] == key and time.monotonic() - _lnk_cache[0] < _INDEX_TTL_S:
        return _lnk_cache[2]
    entries: dict[str, Path] = {}
    for d in dirs:
        if d.exists():
            for lnk in d.rglob("*.lnk"):
                stem = _norm(lnk.stem)
                if stem and "uninstall" not in stem:
                    entries.setdefault(stem, lnk)
    _lnk_cache = (time.monotonic(), key, entries)
    return entries


def forget_app_index() -> None:
    """Drop the cached app lists so the next lookup sees apps installed in the last few minutes."""
    global _lnk_cache, _store_cache
    _lnk_cache = (0.0, (), {})
    _store_cache = (0.0, {})


def warm_app_index() -> None:
    """Build both app lists ahead of time (the Store list starts PowerShell, 1-2 s) so the first "open X" is quick."""
    _lnk_entries()
    _store_entries()


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


def open_path(path: str) -> None:
    """Open a file with the app Windows has registered for its type. The caller has already checked the path against policy."""
    os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606


# ---- folders: "open CLM in VS Code", "open the Riot Games folder", "open EV_AI in a terminal with the ev environment"
_FOLDER_WORDS = re.compile(r"\s+(?:project|folder|directory|repo|repository)$")
_ENV_NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")


def _folder_key(name: str) -> str:
    """"CLM_MVP" and "clm mvp" and "Clm-MVP" all become "clm mvp"."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def list_folders(roots: list[Path], skip: frozenset[str]) -> dict[str, Path]:
    """The folders directly inside each root, by spoken-style name. Symlinks, junctions, hidden and `skip` names are left out.
    An earlier root wins when two folders have the same name. A root that isn't there (a drive that isn't mounted) is ignored."""
    out: dict[str, Path] = {}
    for root in roots:
        try:
            kids = sorted(root.iterdir())
        except OSError:
            continue
        for p in kids:
            if p.name.startswith((".", "$")) or p.name.lower() in skip:
                continue
            try:
                if p.is_symlink() or getattr(p, "is_junction", lambda: False)() or not p.is_dir():
                    continue
            except OSError:
                continue
            if key := _folder_key(p.name):
                out.setdefault(key, p)
    return out


def find_folder(name: str, roots: list[Path], skip: frozenset[str] = frozenset()) -> Path | None:
    """The folder a spoken name means: exact first, then whole-word containment ("clm" -> CLM_MVP), then a close spelling."""
    want = _folder_key(_FOLDER_WORDS.sub("", name.strip().lower()))
    return _match(list_folders(roots, skip), want, True) if want else None  # type: ignore[no-any-return]


# Coding-agent CLIs a terminal may start: spoken name -> (program on PATH, what EV calls it). The program is fixed here, never
# taken from speech or config, and nothing is passed to it, so it opens at its own prompt and waits for the user.
AGENT_CLIS: dict[str, tuple[str, str]] = {
    "claude": ("claude", "Claude Code"),
    "antigravity": ("agy", "Antigravity"),
}


def open_folder(how: str, path: Path, env: str = "", agent: str = "") -> None:
    """Open `path` in VS Code, File Explorer or a new terminal. Everything started here is built from a fixed program name,
    the folder, and (for a terminal) `conda activate <env>` with an env name that has passed _ENV_NAME and the policy list,
    or one of the fixed coding-agent programs in AGENT_CLIS. No shell string is built and nothing the user said reaches a
    command line except those checked names."""
    if agent:
        if how != "terminal" or env or agent not in AGENT_CLIS:
            raise ValueError("a coding agent starts only in a terminal, on its own")
        prog, label = AGENT_CLIS[agent]
        if not shutil.which(prog):
            raise FileNotFoundError(f"I couldn't find {label} on this PC")
        # -NoProfile: the user's profile (conda init, prompt theme) takes several seconds and the agent needs none of it; the
        # program is found through the PATH Windows already gives every process.
        shell = [resolve_exe("powershell.exe") or "powershell.exe", "-NoProfile", "-NoExit", "-Command", prog]
        wt = resolve_exe("wt.exe")
        if wt:
            subprocess.Popen([wt, "-d", str(path), *shell], close_fds=True)  # noqa: S603
        else:
            subprocess.Popen(shell, cwd=str(path), close_fds=True, creationflags=_NEW_CONSOLE)  # noqa: S603
        return
    if how == "folder":
        os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
    elif how == "vscode":
        code = resolve_exe("Code.exe")
        if not code:
            raise FileNotFoundError("I couldn't find VS Code on this PC")
        subprocess.Popen([code, "--new-window", str(path)], close_fds=True)  # noqa: S603
    elif how == "terminal":
        if env and not _ENV_NAME.fullmatch(env):
            raise ValueError("bad environment name")
        # the user's PowerShell profile runs `conda init`, so "conda activate" is a plain command there (no ; needed, which
        # Windows Terminal would read as its own separator)
        shell = [resolve_exe("powershell.exe") or "powershell.exe", "-NoExit", *(["-Command", f"conda activate {env}"] if env else [])]
        wt = resolve_exe("wt.exe")
        if wt:
            subprocess.Popen([wt, "-d", str(path), *shell], close_fds=True)  # noqa: S603
        else:
            subprocess.Popen(shell, cwd=str(path), close_fds=True, creationflags=_NEW_CONSOLE)  # noqa: S603
    else:
        raise ValueError("unknown way to open a folder")


def launch_app(name: str, allow_start_menu: bool = True, elevate: bool = False) -> str:
    """Launch an app by friendly name. Returns a short description of what was launched.

    elevate=True starts it as administrator; callers only pass it after the user has confirmed out loud."""
    try:
        return _launch(name, allow_start_menu, elevate)
    except FileNotFoundError:
        if not allow_start_menu or (_lnk_cache[0] == 0.0 and _store_cache[0] == 0.0):
            raise
        forget_app_index()                      # maybe it was installed after the lists were cached: look once more
        return _launch(name, allow_start_menu, elevate)


def _launch(name: str, allow_start_menu: bool, elevate: bool) -> str:
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
                # a console program started from a console would share EV's own window: give it a new one
                flags = _NEW_CONSOLE if argv[0].lower() in _CONSOLE_EXES else 0
                subprocess.Popen([exe, *argv[1:]], close_fds=True, creationflags=flags)  # noqa: S603
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
