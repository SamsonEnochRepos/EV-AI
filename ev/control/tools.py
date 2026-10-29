"""Mouse/keyboard control. Every action re-checks the *resolved* target against policy first."""
from __future__ import annotations

import time

from ev.core.config import Config
from ev.core.log import get_logger
from ev.safety.gate import SafetyGate
from ev.safety.policy import Policy, Verdict
from ev.tools.base import Tool, ToolRegistry, ToolResult

log = get_logger("control")


def _pag():  # noqa: ANN202
    import pyautogui

    pyautogui.FAILSAFE = True       # slam the mouse into a corner to abort
    pyautogui.PAUSE = 0.05
    return pyautogui


def register_control_tools(reg: ToolRegistry, cfg: Config, policy: Policy, gate: SafetyGate) -> None:
    from ev.screen import capture, uia
    from ev.screen.ocr import OCR

    ocr = OCR(cfg)

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
        # layer 1: UI Automation
        if win is not None:
            els = uia.elements(win)
            el = uia.find_element(els, name)
            if el is not None:
                d = policy.check_sensitive_target(title, el.name, el.is_password)
                if d.verdict is Verdict.BLOCK:
                    return ToolResult(False, "That control looks sensitive, so I won't click it.")
                _pag().click(*el.center)
                return ToolResult(True, f"Clicked {el.name}.", {"layer": "uia"})
        # layer 2: OCR
        d = policy.check_sensitive_target(title, name)
        if d.verdict is Verdict.BLOCK:
            return ToolResult(False, "That looks sensitive, so I won't click it.")
        img = capture.grab(cfg)
        hit = OCR.find(ocr.read(img), name)
        if hit is None:
            return ToolResult(False, f"I couldn't find {name} on screen.")
        scale_x = _pag().size()[0] / img.size[0]
        cx, cy = hit.center
        _pag().click(int(cx * scale_x), int(cy * scale_x))
        return ToolResult(True, f"Clicked {hit.text}.", {"layer": "ocr"})

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

    def focus_window(title: str) -> ToolResult:
        d = policy.check_sensitive_target(title)
        if d.verdict is Verdict.BLOCK:
            return ToolResult(False, "I won't switch to that window.")
        w = uia.find_window(title)
        if w is None:
            return ToolResult(False, f"I couldn't find a window called {title}.")
        try:
            w.set_focus()
        except Exception:
            return ToolResult(False, "Windows wouldn't let me switch to it.")
        time.sleep(0.15)
        return ToolResult(True, f"Switched to {uia.window_title(w)}.")

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
                      {"name": {"type": S, "maxLength": 80}}, click_element, ("name",), timeout_s=20))
    reg.register(Tool("type_text", "Type text into the focused field.",
                      {"text": {"type": S, "maxLength": 500}}, type_text, ("text",), timeout_s=30))
    reg.register(Tool("press_keys", "Press a key or shortcut such as ctrl+s.",
                      {"combo": {"type": S, "maxLength": 30}}, press_keys, ("combo",)))
    reg.register(Tool("scroll", "Scroll the active window.", {
        "direction": {"type": S, "enum": ["up", "down"]},
        "amount": {"type": "integer", "minimum": 1, "maximum": 20}}, scroll, ("direction",)))
    reg.register(Tool("focus_window", "Bring a window to the front by part of its title.",
                      {"title": {"type": S, "maxLength": 80}}, focus_window, ("title",)))
    reg.register(Tool("copy_selection", "Copy the current selection.", {}, copy_selection))
    reg.register(Tool("paste", "Paste the clipboard.", {}, paste))
