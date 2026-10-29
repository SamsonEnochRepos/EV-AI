"""RapidOCR (ONNX, CPU): text with bounding boxes. Loaded lazily."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ev.core.config import Config


@dataclass
class OcrItem:
    text: str
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in the analysed image's pixels
    conf: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


class OCR:
    def __init__(self, cfg: Config):
        self.min_conf = float(cfg["screen"]["ocr_min_confidence"])
        self._engine = None
        self._lock = threading.Lock()

    def _get(self):  # noqa: ANN202
        with self._lock:
            if self._engine is None:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]

                self._engine = RapidOCR()
            return self._engine

    def read(self, img: Image.Image) -> list[OcrItem]:
        result, _ = self._get()(np.asarray(img.convert("RGB")))
        items: list[OcrItem] = []
        for box, text, conf in result or []:
            c = float(conf)
            if c < self.min_conf or not str(text).strip():
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            items.append(OcrItem(str(text).strip(), (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), c))
        return items

    @staticmethod
    def find(items: list[OcrItem], needle: str) -> OcrItem | None:
        n = needle.lower().strip()
        exact = [i for i in items if i.text.lower() == n]
        if exact:
            return exact[0]
        part = [i for i in items if n in i.text.lower()]
        return min(part, key=lambda i: len(i.text)) if part else None

    @staticmethod
    def as_text(items: list[OcrItem]) -> str:
        rows = sorted(items, key=lambda i: (i.box[1] // 12, i.box[0]))
        return "\n".join(i.text for i in rows)
