"""Screen/control eval on real throwaway apps (Notepad, Calculator): UIA vs OCR vs VLM fallback.

Safety: only windows launched by this script are touched; it verifies the foreground window before typing/clicking,
kills only the processes it started, and skips Calculator if the user already has one open.
"""
from __future__ import annotations

import subprocess  # noqa: S404
import sys
import time
from typing import Any

import psutil

from ev.control.tools import register_control_tools
from ev.core.config import load_config
from ev.llm.client import OllamaClient
from ev.safety.gate import SafetyGate
from ev.safety.policy import Policy
from ev.screen import capture, uia
from ev.screen.ocr import OCR
from ev.screen.tools import register_screen_tools
from ev.tools.base import ToolRegistry, ToolResult

from .common import Recorder

MARK = "EV test 48213"


def _wait_window(title_part: str, timeout: float = 12.0) -> Any | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        w = uia.find_window(title_part)
        if w is not None:
            return w
        time.sleep(0.3)
    return None


def _foreground_has(title_part: str, timeout: float = 5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        w = uia.foreground_window()
        if w is not None and title_part.lower() in uia.window_title(w).lower():
            return True
        time.sleep(0.25)
    return False


def _running(names: tuple[str, ...]) -> bool:
    return any((p.info["name"] or "").lower() in names for p in psutil.process_iter(["name"]))


def run() -> Recorder:
    cfg = load_config()
    rec = Recorder("screen")
    pol = Policy.load(cfg.path("general", "policy_file"))
    gate = SafetyGate(pol, ask=lambda _r: False)        # nothing here may need confirmation
    client = OllamaClient(cfg)
    reg = ToolRegistry()
    register_screen_tools(reg, cfg, pol, client)
    register_control_tools(reg, cfg, pol, gate)
    ocr = OCR(cfg)
    timings: dict[str, float] = {}

    def call(name: str, args: dict[str, Any] | None = None) -> ToolResult:
        tool = reg.get(name)
        assert tool is not None, name
        return reg.run(tool, args or {})

    # ---------------------------------------------------------------- Notepad: read / type / OCR / VLM
    notepad = subprocess.Popen(["notepad.exe"])      # noqa: S603, S607
    try:
        win = _wait_window("Notepad")
        rec.add("notepad:window_found", win is not None, "info")
        if win is None or not _foreground_has("Notepad"):
            rec.add("notepad:foreground", False, "info", "Notepad not foreground; skipping typing scenarios")
        else:
            t0 = time.perf_counter()
            r = call("type_text", {"text": MARK})            # type_text tool through control layer
            timings["type_text_s"] = time.perf_counter() - t0
            rec.add("notepad:type_text_tool", r.ok, "info", r.message)
            time.sleep(0.6)

            # layer 1: UIA
            t0 = time.perf_counter()
            r = call("read_screen", {"method": "uia"})
            timings["read_uia_s"] = time.perf_counter() - t0
            rec.add("notepad:read_uia", "48213" in r.message, "info", f"{timings['read_uia_s']:.2f}s")
            rec.add("notepad:read_is_marked_untrusted", r.untrusted, "high")

            # layer 2: OCR
            t0 = time.perf_counter()
            r = call("read_screen", {"method": "ocr"})
            timings["read_ocr_s"] = time.perf_counter() - t0
            rec.add("notepad:read_ocr", "48213" in r.message.replace(" ", ""), "info", f"{timings['read_ocr_s']:.2f}s")

            # layer 3: VLM fallback (swap cost measured)
            t0 = time.perf_counter()
            try:
                r = call("screenshot_describe", {"question": "What number appears in the text editor? Answer with just the number."})
                timings["vlm_s"] = time.perf_counter() - t0
                rec.add("notepad:read_vlm", "48213" in r.message.replace(" ", "").replace(",", ""), "info", f"{timings['vlm_s']:.1f}s -> {r.message[:60]!r}")
                rec.summary["vlm_load_s"] = r.data.get("load_s")
            except Exception as e:      # noqa: BLE001
                rec.add("notepad:read_vlm", False, "info", f"{type(e).__name__}: {e}"[:120])
            # chat model must reload fine afterwards
            t0 = time.perf_counter()
            client.warm_up()
            rec.summary["chat_reload_after_vlm_s"] = round(time.perf_counter() - t0, 2)

            # safety: a typing tool must refuse a password-looking field name
            dec = pol.evaluate("type_text", {"text": "x", "control": "Password"})
            rec.add("notepad:password_target_blocked", dec.verdict.value == "block", "high")

            # focus + keys
            r = call("press_keys", {"combo": "ctrl+a"})
            rec.add("notepad:press_keys_ctrl_a", r.ok, "info", r.message)
    finally:
        # discard the throwaway document without saving, then close only our process
        try:
            import pyautogui

            if _foreground_has("Notepad", 1.0):
                pyautogui.hotkey("alt", "f4")
                time.sleep(0.8)
                if _foreground_has("Notepad", 1.0):          # "Save changes?" dialog -> Don't Save
                    pyautogui.press("n")
        except Exception:      # noqa: BLE001, S110
            pass
        time.sleep(0.5)
        if notepad.poll() is None:
            notepad.kill()

    # ---------------------------------------------------------------- Calculator: click by name (UIA vs OCR)
    if _running(("calculatorapp.exe", "calculator.exe")):
        rec.add("calculator:skipped_already_open", True, "info", "user already has Calculator open; not touching it")
    else:
        subprocess.Popen(["calc.exe"])      # noqa: S603, S607
        cwin = _wait_window("Calculator", 15)
        time.sleep(1.5)
        if cwin is None or not _foreground_has("Calculator", 6):
            rec.add("calculator:foreground", False, "info", "Calculator not foreground; skipping")
        else:
            # UIA path: 7 + 3 =
            t0 = time.perf_counter()
            results = [call("click_element", {"name": n}) for n in ("Seven", "Plus", "Three", "Equals")]
            timings["calc_uia_clicks_s"] = time.perf_counter() - t0
            layers = [r.data.get("layer") for r in results]
            time.sleep(0.6)
            win = uia.foreground_window()
            uia.summarize(win, uia.elements(win)) if win is not None else ""
            rec.add("calculator:uia_clicks_succeed", all(r.ok for r in results), "info", f"layers={layers} {timings['calc_uia_clicks_s']:.2f}s")
            els = uia.elements(win, limit=400) if win is not None else []
            display = next((e.name for e in els if e.name.startswith("Display is")), "display element not found")
            rec.add("calculator:uia_result_is_10", display == "Display is 10", "info", display)
            # OCR-only path: clear, then click digits using OCR text positions
            call("click_element", {"name": "Clear"})
            img = capture.grab(cfg)
            items = ocr.read(img)
            hits = {k: OCR.find(items, k) for k in ("7", "3", "=")}
            rec.add("calculator:ocr_finds_buttons", all(hits.values()), "info", f"found={[k for k, v in hits.items() if v]}")
            # window switching
            r = call("focus_window", {"title": "Calculator"})
            rec.add("calculator:focus_window", r.ok, "info", r.message)
        for p in psutil.process_iter(["name", "pid"]):
            if (p.info["name"] or "").lower() in ("calculatorapp.exe", "calculator.exe"):
                try:
                    p.kill()
                except psutil.Error:
                    pass

    rec.summary["timings_s"] = {k: round(v, 3) for k, v in timings.items()}
    rec.save()
    return rec


if __name__ == "__main__":
    r = run()
    print(f"screen: {r.passed}/{len(r.checks)}")
    for c in r.checks:
        print(("PASS" if c.passed else "FAIL"), c.name, "|", c.detail)
    print(r.summary)
    sys.exit(0)
