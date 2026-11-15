"""Mouse/keyboard control. Every action re-checks the *resolved* target against policy first."""
from __future__ import annotations

import re
import time
from pathlib import Path

from ev.control import trust
from ev.core.config import Config
from ev.core.log import get_logger
from ev.safety.gate import SafetyGate
from ev.safety.policy import Policy, Verdict
from ev.tools import windows
from ev.tools.base import Tool, ToolRegistry, ToolResult

log = get_logger("control")

_TRUST_WAIT_S = 12.0                                    # how long trust_agent_folder waits for the prompt to show
_TERMINAL_EXES = frozenset({"windowsterminal"})          # Windows Terminal is the one terminal whose text UIA can read


def _pag():  # noqa: ANN202
    import pyautogui

    pyautogui.FAILSAFE = True       # slam the mouse into a corner to abort
    pyautogui.PAUSE = 0.05
    return pyautogui


def _clicked(label: str) -> str:
    """What was clicked, as a short spoken line. A match that is a whole line of text (a paragraph, a log line) is not a name to read out."""
    label = " ".join(label.split())
    return f"Clicked {label}." if 0 < len(label) <= 40 else "Clicked it."


def register_control_tools(reg: ToolRegistry, cfg: Config, policy: Policy, gate: SafetyGate) -> None:
    from ev.screen import uia

    def _killed() -> bool:
        return gate.killed.is_set()

    def _active_window_check() -> ToolResult | None:
        win = uia.foreground_window()
        if win is not None:
            d = policy.check_sensitive_target(uia.window_title(win))
            if d.verdict is Verdict.BLOCK:
                return ToolResult(False, "That window looks sensitive, so I'm not touching it.")
        return None

    def click_element(name: str) -> ToolResult:
        if _killed():
            return ToolResult(False, "Stopped.")
        win = uia.foreground_window()
        title = uia.window_title(win) if win is not None else ""
        if win is not None:
            els = uia.elements(win)
            el = uia.find_element(els, name)
            if el is not None:
                d = policy.check_sensitive_target(title, el.name, el.is_password)
                if d.verdict is Verdict.BLOCK:
                    return ToolResult(False, "That control looks sensitive, so I won't click it.")
                _pag().click(*el.center)
                return ToolResult(True, _clicked(el.name), {"layer": "uia"})
        d = policy.check_sensitive_target(title, name)
        if d.verdict is Verdict.BLOCK:
            return ToolResult(False, "That looks sensitive, so I won't click it.")
        return ToolResult(False, f"I couldn't find {name} on screen.")

    def type_text(text: str) -> ToolResult:
        if _killed():
            return ToolResult(False, "Stopped.")
        blocked = _active_window_check()
        if blocked:
            return blocked
        win = uia.foreground_window()
        if win is not None:
            try:
                focused = win.get_focus()
                if focused is not None and bool(focused.element_info.element.CurrentIsPassword):
                    return ToolResult(False, "That's a password field. I never type into those.")
            except Exception:  # noqa: S110  (focus probe is best effort; policy already checked the window)
                pass
        _pag().write(text, interval=0.01)
        return ToolResult(True, "Typed it.")

    def press_keys(combo: str) -> ToolResult:
        if _killed():
            return ToolResult(False, "Stopped.")
        blocked = _active_window_check()
        if blocked:
            return blocked
        keys = [k.strip().lower() for k in combo.replace("+", " ").split() if k.strip()]
        _pag().hotkey(*keys)
        return ToolResult(True, f"Pressed {combo}.")

    def scroll(direction: str, amount: int = 3) -> ToolResult:
        _pag().scroll(amount * 120 * (1 if direction == "up" else -1))
        return ToolResult(True, f"Scrolled {direction}.")

    def _app_label(w: windows.Win) -> str:
        """"EV assistant... - EV_AI - Visual Studio Code" -> "Visual Studio Code": the app, not a page or file title."""
        return re.split(r"\s[-—–]\s", w.title)[-1].strip() or w.exe

    def focus_window(title: str) -> ToolResult:
        d = policy.check_sensitive_target(title)
        if d.verdict is Verdict.BLOCK:
            return ToolResult(False, "I won't switch to that window.")
        try:
            wins, fg = windows.list_windows(), windows.foreground()
        except OSError:
            wins, fg = [], None
        target = windows.pick_for_focus(title, wins, fg)          # by app ("VS Code", "the browser", "previous window")
        if target is not None:
            if policy.check_sensitive_target(target.title).verdict is Verdict.BLOCK:      # the window it resolved to, not the words
                return ToolResult(False, "I won't switch to that window.")
            if not windows.bring_to_front(target):
                return ToolResult(False, "Windows wouldn't let me switch to it.")
            return ToolResult(True, f"Switched to {_app_label(target)}.")
        if windows.is_previous_word(title):
            return ToolResult(False, "There's no other window to go back to.")
        w = uia.find_window(title)
        if w is None:
            return ToolResult(False, f"I couldn't find a window called {title}.")
        try:
            w.set_focus()
        except Exception:
            return ToolResult(False, "Windows wouldn't let me switch to it.")
        time.sleep(0.15)
        return ToolResult(True, f"Switched to {uia.window_title(w)}.")

    def switch_tab(name: str) -> ToolResult:
        """Go to the browser tab whose title has these words. The window in front is searched first, then the other browser
        windows (a private window only if it is the one in front). The reply uses the user's words, never the page's title."""
        if _killed():
            return ToolResult(False, "Stopped.")
        if policy.check_sensitive_target(name).verdict is Verdict.BLOCK:
            return ToolResult(False, "I won't switch to that tab.")
        try:
            wins, fg = windows.list_windows(), windows.foreground()
        except OSError:
            return ToolResult(False, "I can't look at windows here.")
        cands = windows.browser_windows(wins, fg)
        if not cands:
            return ToolResult(False, "I don't see a browser open.")
        front = [w for w in cands if fg is not None and w.hwnd == fg.hwnd]
        for w in front + [w for w in cands if w not in front]:
            tab = uia.find_tab(uia.browser_tabs(uia.window_for(w.hwnd)), name)
            if tab is None:
                continue
            if policy.check_sensitive_target(w.title, tab.name).verdict is Verdict.BLOCK:      # the tab it resolved to
                return ToolResult(False, "I won't switch to that tab.")
            if not windows.bring_to_front(w):
                return ToolResult(False, "Windows wouldn't let me switch to it.")
            if not uia.select_tab(tab):
                _pag().click(*tab.center)
            return ToolResult(True, f"Switched to the {name.strip()} tab.")
        return ToolResult(False, f"I couldn't find a tab called {name.strip()}.")

    def trust_agent_folder() -> ToolResult:
        """Answer Claude Code's "trust this folder?" prompt with yes, only when everything checks out on screen: the window in front
        is Windows Terminal, its text is that prompt, the folder it names sits directly inside a [projects] root, and "Yes" is the
        highlighted choice. It waits a few seconds for the prompt to appear (it is slow to come up right after a launch)."""
        deadline = time.monotonic() + _TRUST_WAIT_S
        fg: windows.Win | None = None
        prompt = None
        while True:
            if _killed():
                return ToolResult(False, "Stopped.")
            try:
                fg = windows.foreground()
            except OSError:
                fg = None
            if fg is not None and fg.exe in _TERMINAL_EXES:
                prompt = trust.parse_claude_trust(uia.terminal_text(fg.hwnd))
                if prompt is not None:
                    break
            if time.monotonic() >= deadline:
                return ToolResult(False, "I don't see Claude Code asking to trust a folder in the window in front.")
            time.sleep(0.5)
        assert fg is not None
        if policy.check_sensitive_target(fg.title).verdict is Verdict.BLOCK:
            return ToolResult(False, "That window looks sensitive, so I'm not touching it.")
        folder = Path(prompt.path)
        if not policy.is_project_folder(folder):
            return ToolResult(False, "That folder isn't one of my project folders, so I'll leave the trust question to you.")
        if not prompt.yes_selected:
            return ToolResult(False, "Yes isn't the highlighted choice, so I'm leaving it to you.")
        _pag().press("enter")
        time.sleep(1.2)
        after = trust.parse_claude_trust(uia.terminal_text(fg.hwnd))
        if after is not None and after.path == prompt.path:
            return ToolResult(False, "I pressed enter, but the trust question is still showing.")
        return ToolResult(True, f"Trusted {folder.name} for Claude Code.")

    def copy_selection() -> ToolResult:
        blocked = _active_window_check()
        if blocked:
            return blocked
        _pag().hotkey("ctrl", "c")
        return ToolResult(True, "Copied.")

    def paste() -> ToolResult:
        blocked = _active_window_check()
        if blocked:
            return blocked
        _pag().hotkey("ctrl", "v")
        return ToolResult(True, "Pasted.")

    S = "string"
    reg.register(Tool("click_element", "Click a button, link or control by its visible name.",
                      {"name": {"type": S, "maxLength": 80}}, click_element, ("name",), timeout_s=20, llm_visible=False))
    reg.register(Tool("type_text", "Type text into the focused field.",
                      {"text": {"type": S, "maxLength": 500}}, type_text, ("text",), timeout_s=30, llm_visible=False))
    reg.register(Tool("press_keys", "Press a key or shortcut such as ctrl+s.",
                      {"combo": {"type": S, "maxLength": 30}}, press_keys, ("combo",), llm_visible=False))
    reg.register(Tool("scroll", "Scroll the active window.", {
        "direction": {"type": S, "enum": ["up", "down"]},
        "amount": {"type": "integer", "minimum": 1, "maximum": 20}}, scroll, ("direction",), llm_visible=False))
    reg.register(Tool("focus_window", "Bring a window to the front by part of its title.",
                      {"title": {"type": S, "maxLength": 80}}, focus_window, ("title",), llm_visible=False))
    reg.register(Tool("switch_tab", "Switch to a browser tab by a few words of its title.",
                      {"name": {"type": S, "maxLength": 80}}, switch_tab, ("name",), timeout_s=20, llm_visible=False))
    reg.register(Tool("trust_agent_folder", "Say yes to Claude Code's trust-this-folder question in the terminal in front.",
                      {}, trust_agent_folder, timeout_s=_TRUST_WAIT_S + 10, llm_visible=False))
    reg.register(Tool("copy_selection", "Copy the current selection.", {}, copy_selection, llm_visible=False))
    reg.register(Tool("paste", "Paste the clipboard.", {}, paste, llm_visible=False))
