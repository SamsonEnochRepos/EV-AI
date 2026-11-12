"""Screen tools, voice-routed only: read_screen (UI Automation, no OCR) and take_screenshot."""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

from ev.core.config import Config
from ev.safety.policy import Policy
from ev.tools.base import Tool, ToolRegistry, ToolResult

from . import capture, uia


def register_screen_tools(reg: ToolRegistry, cfg: Config, policy: Policy,
                          on_artifact: Callable[[Path, str], None] | None = None) -> None:
    def read_screen() -> ToolResult:
        """UI Automation only: cheap, and never leaves the machine."""
        text = ""
        win = uia.foreground_window()
        if win is not None:
            title = uia.window_title(win)
            blocked = policy.check_sensitive_target(title)
            if blocked.verdict.value == "block":
                return ToolResult(False, "That window looks sensitive, so I won't read it.")
            text = uia.summarize(win, uia.elements(win))
        if not text.strip():
            return ToolResult(False, "I couldn't find any readable text on screen.")
        return ToolResult(True, text[:3000], untrusted=True, source="screen",
                          hint="Tell the user in one short casual sentence what is on their screen. "
                               "Don't read it out word for word.")

    def take_screenshot() -> ToolResult:
        """Save the whole screen as a PNG the user can find. Only on request; nothing else here writes screenshots."""
        win = uia.foreground_window()
        if win is not None and policy.check_sensitive_target(uia.window_title(win)).verdict.value == "block":
            return ToolResult(False, "That window looks sensitive, so I won't screenshot it.")
        folder = cfg.screenshot_dir()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"EV {dt.datetime.now():%Y-%m-%d %H-%M-%S}.png"
        capture.grab(cfg, full=True).save(path)
        if on_artifact:
            on_artifact(path, "screenshot")        # so "open it" / "open the screenshot you just took" can find it
        return ToolResult(True, "Took a screenshot and saved it.", {"path": str(path)})

    reg.register(Tool("take_screenshot", "Take a screenshot of the screen and save it as an image file for the user.",
                      {}, take_screenshot, timeout_s=15, direct=True, llm_visible=False))
    reg.register(Tool("read_screen", "Read the text currently visible on the user's screen.",
                      {}, read_screen, timeout_s=20, llm_visible=False))
