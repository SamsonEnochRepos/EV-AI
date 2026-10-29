"""Screen-awareness tools: read_screen (UIA -> OCR) and screenshot_describe (VLM)."""
from __future__ import annotations

from ev.core.config import Config
from ev.llm.client import OllamaClient
from ev.safety.policy import Policy
from ev.tools.base import Tool, ToolRegistry, ToolResult

from . import capture, uia
from .ocr import OCR


def register_screen_tools(reg: ToolRegistry, cfg: Config, policy: Policy, client: OllamaClient) -> None:
    ocr = OCR(cfg)

    def read_screen(method: str = "auto") -> ToolResult:
        """Layered: UI Automation first (cheap), OCR if UIA yields too little."""
        text = ""
        if method in ("auto", "uia"):
            win = uia.foreground_window()
            if win is not None:
                title = uia.window_title(win)
                blocked = policy.check_sensitive_target(title)
                if blocked.verdict.value == "block":
                    return ToolResult(False, "That window looks sensitive, so I won't read it.")
                text = uia.summarize(win, uia.elements(win))
        if method == "ocr" or (method == "auto" and len(text) < 80):
            img = capture.grab(cfg)
            text = OCR.as_text(ocr.read(img))
        if not text.strip():
            return ToolResult(False, "I couldn't find any readable text on screen.")
        return ToolResult(True, text[:3000], untrusted=True, source="screen",
                          hint="Tell the user in one short casual sentence what is on their screen. "
                               "Don't read it out word for word.")

    def screenshot_describe(question: str = "") -> ToolResult:
        from .vlm import VLM

        img = capture.grab(cfg)
        r = VLM(cfg, client).describe(img, question or "Describe what is on this screen briefly.")
        return ToolResult(True, r.text, {"load_s": r.load_s}, untrusted=True, source="screen_vlm")

    reg.register(Tool("read_screen", "Read the text currently visible on the user's screen.", {
        "method": {"type": "string", "enum": ["auto", "uia", "ocr"]}}, read_screen, timeout_s=20))
    reg.register(Tool("screenshot_describe", "Describe the screen visually (slow; use only if read_screen is not enough).", {
        "question": {"type": "string", "maxLength": 200}}, screenshot_describe, timeout_s=90))
