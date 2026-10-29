"""VLM fallback: load on demand via Ollama, run once, unload immediately. Logs swap cost."""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass

from PIL import Image

from ev.core.config import Config
from ev.core.log import get_logger
from ev.llm.client import OllamaClient

from .capture import downscale, to_jpeg_bytes

log = get_logger("vlm")


@dataclass
class VlmResult:
    text: str
    load_s: float
    total_s: float


class VLM:
    def __init__(self, cfg: Config, client: OllamaClient):
        v = cfg["vlm"]
        self.model = v["model"]
        self.max_px = int(v["max_image_px"])
        self.keep_alive = str(v["keep_alive"])
        self.client = client
        self.chat_model = client.model

    def describe(self, img: Image.Image, question: str = "Describe what is on this screen briefly.") -> VlmResult:
        """The chat model is unloaded first so VRAM never holds both."""
        t0 = time.perf_counter()
        self.client.unload(self.client.model)
        b64 = base64.b64encode(to_jpeg_bytes(downscale(img, self.max_px))).decode()
        t_load = time.perf_counter()
        r = self.client.chat(
            [{"role": "user", "content": question, "images": [b64]}],
            model=self.model, keep_alive=self.keep_alive,
            options={"num_ctx": 2048, "num_predict": 160},
        )
        self.client.unload(self.model)
        total = time.perf_counter() - t0
        load_ns = r.stats.get("load_duration", 0)
        log.info("VLM swap: model load %.2fs, total %.2fs", load_ns / 1e9, total)
        # chat model reloads lazily on the next turn (keep_alive on first request)
        return VlmResult(r.text.strip(), load_ns / 1e9 if load_ns else t_load - t0, total)
