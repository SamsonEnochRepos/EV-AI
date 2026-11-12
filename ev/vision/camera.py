"""On-request camera vision: take one frame from the webcam, ask the local vision model about it, forget the frame.

The camera is opened only for the moment of the grab and closed again, so its light is on for about a second and nothing runs
in the background. The frame lives in memory only: it is never written to disk and never leaves the machine.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from PIL import Image

from ev.core.config import Config
from ev.core.log import get_logger
from ev.llm.client import OllamaClient
from ev.safety.policy import Policy
from ev.tools.base import Tool, ToolRegistry, ToolResult

log = get_logger("camera")

_GENERIC = ("Describe the person in front of the camera and anything they are holding or showing, in one or two short "
            "sentences.")
_FRAME_PROMPT = ("This is one photo from the user's webcam. Answer their request about it in one or two short sentences. "
                 "Only say what you can actually see; if you can't tell, say so. Request: ")
_DARK = 25.0          # average pixel value (0-255) below which a frame is the camera still adjusting its exposure
_STEADY = 2.0         # the picture counts as settled when its average brightness moves less than this between frames


class CameraError(RuntimeError):
    """The camera could not be opened or gave no picture. The message is safe to say aloud."""


def grab_frame(index: int = 0, warmup_frames: int = 6, timeout_s: float = 5.0) -> Image.Image:
    """One RGB frame from the webcam. A freshly opened camera spends its first frames adjusting exposure, so frames are read
    and thrown away until the picture is bright enough and has stopped changing (at least `warmup_frames`, at most `timeout_s`).
    The camera keeps its own default resolution: asking for another one costs over a second and the vision model does not need it."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)          # DirectShow opens and closes much faster than the default backend here
    try:
        if not cap.isOpened():
            raise CameraError("I can't open the camera. Another app might be using it, or it's switched off.")
        deadline = time.monotonic() + timeout_s
        frame = None
        read = 0
        prev = -1000.0
        while time.monotonic() < deadline:
            ok, got = cap.read()
            if not ok or got is None:
                continue
            read += 1
            frame, level = got, float(got.mean())
            if read >= warmup_frames and level > _DARK and abs(level - prev) < _STEADY:
                break
            prev = level
        if frame is None:
            raise CameraError("The camera isn't giving me a picture.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def make_question(said: str) -> str:
    """What to ask the vision model: the user's own words about the photo, or a general description."""
    said = " ".join(said.split())[:200]
    return _FRAME_PROMPT + said if said else _GENERIC


def register_camera_tool(reg: ToolRegistry, cfg: Config, policy: Policy, client: OllamaClient,
                         grab: Callable[..., Image.Image] = grab_frame) -> None:
    """look_at_camera: voice only. It is matched from what the user says (ev/llm/fastpath.py) and is not in the model's tool
    list, so text on a web page or a screen can never make EV turn the camera on."""
    c: dict[str, Any] = cfg.raw.get("camera", {})
    if not c.get("enabled", False):
        return

    def look_at_camera(question: str = "") -> ToolResult:
        from ev.screen.vlm import VLM

        try:
            img = grab(int(c.get("index", 0)), int(c.get("warmup_frames", 6)))
        except CameraError as e:
            return ToolResult(False, str(e))
        except Exception as e:                                # no OpenCV, driver trouble: the user hears a plain sentence
            log.warning("camera failed: %s", e)
            return ToolResult(False, "I couldn't use the camera.")
        r = VLM(cfg, client).describe(img, make_question(question))
        del img                                               # the frame is not kept anywhere
        if not r.text:
            return ToolResult(False, "I looked, but I couldn't make anything out.")
        return ToolResult(True, r.text, {"load_s": r.load_s}, untrusted=True, source="camera",
                          hint="This is what the camera showed. Tell the user in one or two short casual sentences. "
                               "Don't mention a photo or an image, you are looking through the camera right now.")

    reg.register(Tool("look_at_camera", "Look through the camera and say what is there.",
                      {"question": {"type": "string", "maxLength": 200}}, look_at_camera, timeout_s=90, llm_visible=False))
