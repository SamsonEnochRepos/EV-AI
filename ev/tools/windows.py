"""Top-level window helpers: find an app's windows, ask them to close, force-quit as a last resort.

Closing is polite by default (WM_CLOSE, exactly what the X button sends), so apps still offer to save unsaved work.
EV never elevates: a window or process that needs administrator rights is reported as such and left alone.
"""
from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

import psutil

from .system import KNOWN_APPS

WM_CLOSE = 0x0010
_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x80
_DWMWA_CLOAKED = 13
_SHELL_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "NotifyIconOverflowWindow"}
_UWP_HOSTS = {"applicationframehost"}     # Store apps (Calculator, Settings...) live inside this host process
_OWN_HELPERS = {"python", "pythonw"}
_BROWSERS = {"firefox", "chrome", "msedge", "brave", "opera", "vivaldi"}
THIS_WINDOW = {"this", "this window", "the window", "window", "current window", "active window", "this app",
               "current app", "the current window", "the active window", "front window"}

# friendly name -> process stems, for apps whose process name differs from what people say
_ALIASES: dict[str, set[str]] = {
    "calculator": {"calculatorapp", "calculator", "calc"}, "calc": {"calculatorapp", "calculator", "calc"},
    "settings": {"systemsettings"}, "terminal": {"windowsterminal", "wt"}, "windows terminal": {"windowsterminal", "wt"},
    "word": {"winword"}, "powerpoint": {"powerpnt"}, "paint": {"mspaint"}, "edge": {"msedge"},
    "microsoft edge": {"msedge"}, "google chrome": {"chrome"}, "task manager": {"taskmgr"},
    "file explorer": {"explorer"}, "files": {"explorer"}, "vs code": {"code"}, "vscode": {"code"},
    "visual studio code": {"code"}, "teams": {"ms-teams", "teams"}, "whatsapp": {"whatsapp", "whatsapp.root"},
}


class WindowsOnly(OSError):
    pass


@dataclass(frozen=True)
class Win:
    hwnd: int
    title: str
    pid: int
    exe: str          # process name without ".exe", lowercase
    cls: str = ""


def _user32() -> ctypes.WinDLL:
    if sys.platform != "win32":
        raise WindowsOnly("closing apps needs Windows")
    return ctypes.windll.user32  # type: ignore[no-any-return]


def _stem(pid: int) -> str:
    try:
        return psutil.Process(pid).name().lower().removesuffix(".exe")
    except (psutil.Error, OSError):
        return ""


def _describe(u: ctypes.WinDLL, hwnd: int) -> Win | None:
    buf = ctypes.create_unicode_buffer(256)
    u.GetWindowTextW(hwnd, buf, 256)
    cls = ctypes.create_unicode_buffer(128)
    u.GetClassNameW(hwnd, cls, 128)
    pid = wintypes.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return Win(int(hwnd), buf.value, int(pid.value), _stem(int(pid.value)), cls.value)


def list_windows() -> list[Win]:
    """Visible, titled, real top-level windows (no tool windows, hidden UWP ghosts or the desktop shell)."""
    u = _user32()
    out: list[Win] = []
    dwm = ctypes.windll.dwmapi
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd: int, _lp: int) -> bool:
        if not u.IsWindowVisible(hwnd) or u.GetWindowTextLengthW(hwnd) == 0:
            return True
        if u.GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_TOOLWINDOW:
            return True
        cloaked = wintypes.DWORD()
        dwm.DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        if cloaked.value:
            return True
        w = _describe(u, hwnd)
        if w and w.cls not in _SHELL_CLASSES:
            out.append(w)
        return True

    u.EnumWindows(proto(cb), 0)
    return out


def foreground() -> Win | None:
    u = _user32()
    hwnd = u.GetForegroundWindow()
    w = _describe(u, hwnd) if hwnd else None
    return w if w and w.cls not in _SHELL_CLASSES else None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def stems_for(name: str) -> set[str]:
    k = _norm(name)
    stems = {k, k.replace(" ", "")} | _ALIASES.get(k, set())
    for exe in KNOWN_APPS.get(k, []):
        if exe.lower().endswith(".exe"):
            stems.add(exe.lower().removesuffix(".exe"))
    return {s for s in stems if s}


def match(name: str, wins: list[Win]) -> list[Win]:
    """Windows that belong to the app the user named. Browser tabs never match by title: closing the window would
    close every tab, so "close YouTube" must not quietly shut Firefox."""
    k = _norm(name)
    if not k:
        return []
    stems = stems_for(name)
    compact = k.replace(" ", "")
    hits = []
    for w in wins:
        if w.exe in stems or (len(compact) >= 4 and w.exe.startswith(compact)):
            hits.append(w)
        elif w.exe in _UWP_HOSTS and len(k) >= 3 and k in w.title.lower():
            hits.append(w)
        elif w.exe not in _BROWSERS and len(k) >= 4 and re.search(rf"\b{re.escape(k)}\b", w.title.lower()):
            hits.append(w)
    return hits


BROWSER_STEMS = frozenset(_BROWSERS)
_BROWSER_WORDS = frozenset({"browser", "web browser", "the browser", "internet", "internet browser", "my browser"})
_PREVIOUS_WORDS = frozenset({"previous", "last", "back", "previous window", "last window", "previous app", "last app",
                             "the previous window", "the last window", "other window", "the other window"})
_NEVER_SWITCH_TO = frozenset({"python", "pythonw"})      # EV's own overlay, never "the previous window"


def is_previous_word(name: str) -> bool:
    return _norm(name) in _PREVIOUS_WORDS


def is_private_browser_window(w: Win) -> bool:
    return w.exe in _BROWSERS and ("private browsing" in w.title.lower() or "inprivate" in w.title.lower()
                                   or "incognito" in w.title.lower())


def browser_windows(wins: list[Win], fg: Win | None) -> list[Win]:
    """Browser windows, front-most first. A private window is left out unless it is the one the user is already in."""
    return [w for w in wins if w.exe in _BROWSERS and (not is_private_browser_window(w) or (fg is not None and w.hwnd == fg.hwnd))]


def pick_for_focus(name: str, wins: list[Win], fg: Win | None) -> Win | None:
    """The window "switch to X" means: an app by name (VS Code, Firefox), "the browser", or "the previous window".

    `wins` is in Z-order (front first), so the first hit is the one used most recently. A window that is already in front
    is passed over when another one of the same app exists, so "switch to Firefox" from a second Firefox window goes to the first."""
    k = _norm(name)
    if k in _PREVIOUS_WORDS:
        return next((w for w in wins if (fg is None or w.hwnd != fg.hwnd) and w.exe not in _NEVER_SWITCH_TO), None)
    hits = browser_windows(wins, fg) if k in _BROWSER_WORDS else match(name, wins)
    hits = [w for w in hits if w.exe not in _NEVER_SWITCH_TO and not is_private_browser_window(w)] or [
        w for w in hits if w.exe not in _NEVER_SWITCH_TO]
    others = [w for w in hits if fg is None or w.hwnd != fg.hwnd]
    chosen = others or hits
    return chosen[0] if chosen else None


def bring_to_front(w: Win) -> bool:
    """Make `w` the foreground window. Windows only lets the program that last received input take focus, so this joins the
    current foreground window's input queue for the moment (the usual way; it sends no keystrokes, so nothing is typed or opened).
    True if the window really is in front afterwards."""
    u = _user32()
    if u.IsIconic(w.hwnd):
        u.ShowWindow(w.hwnd, 9)                                   # SW_RESTORE
    fg = u.GetForegroundWindow()
    me = ctypes.windll.kernel32.GetCurrentThreadId()
    fg_tid = int(u.GetWindowThreadProcessId(fg, None)) if fg else 0
    attached = bool(fg_tid) and fg_tid != me and bool(u.AttachThreadInput(me, fg_tid, True))
    try:
        u.BringWindowToTop(w.hwnd)
        u.SetForegroundWindow(w.hwnd)
    finally:
        if attached:
            u.AttachThreadInput(me, fg_tid, False)
    time.sleep(0.12)
    return int(u.GetForegroundWindow()) == w.hwnd


def own_pids() -> set[int]:
    """EV, what launched it (the terminal it runs in) and its own Python helpers (the overlay).

    Apps EV opened for the user are children too, but they are the user's to close, so they are not listed."""
    me = psutil.Process(os.getpid())
    pids = {me.pid}
    try:
        pids |= {p.pid for p in me.parents()}
        pids |= {p.pid for p in me.children(recursive=True) if p.name().lower().removesuffix(".exe") in _OWN_HELPERS}
    except psutil.Error:
        pass
    return pids


def request_close(w: Win) -> bool:
    """Send WM_CLOSE. False if Windows refused (the window belongs to an elevated process)."""
    return bool(_user32().PostMessageW(w.hwnd, WM_CLOSE, 0, 0))


def wait_gone(wins: list[Win], timeout_s: float = 2.5) -> list[Win]:
    """Windows that are still open after waiting up to timeout_s."""
    u = _user32()
    left = list(wins)
    end = time.monotonic() + timeout_s
    while left and time.monotonic() < end:
        left = [w for w in left if u.IsWindow(w.hwnd)]
        if left:
            time.sleep(0.1)
    return left


def running_pids(stems: set[str], exclude: set[int]) -> list[int]:
    out = []
    for p in psutil.process_iter(["pid", "name"]):
        n = (p.info["name"] or "").lower().removesuffix(".exe")
        if n in stems and p.info["pid"] not in exclude:
            out.append(p.info["pid"])
    return out


def terminate(pids: list[int]) -> tuple[int, int]:
    """Force-quit. Returns (ended, refused); refused means Windows said access denied (needs admin)."""
    procs, refused = [], 0
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.terminate()
            procs.append(p)
        except psutil.AccessDenied:
            refused += 1
        except psutil.NoSuchProcess:
            continue
    gone, alive = psutil.wait_procs(procs, timeout=2)
    return len(gone), refused + len(alive)
