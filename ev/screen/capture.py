"""Screen capture with mss. Images live in memory only; nothing is written unless the debug flag is on."""
from __future__ import annotations

import atexit
import io
from pathlib import Path

from PIL import Image

from ev.core.config import Config

_debug_files: list[Path] = []


def grab(cfg: Config, monitor: int = 1, region: tuple[int, int, int, int] | None = None,
         full: bool = False) -> Image.Image:
    """Capture the screen. full=True keeps native resolution (for screenshots the user asked to keep)."""
    import mss

    with mss.mss() as sct:
        mon = sct.monitors[monitor] if region is None else {
            "left": region[0], "top": region[1], "width": region[2], "height": region[3]}
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    if full:
        return img
    img = downscale(img, int(cfg["screen"]["max_capture_px"]))
    _maybe_debug_save(cfg, img)
    return img


def downscale(img: Image.Image, max_px: int) -> Image.Image:
    w, h = img.size
    s = max_px / max(w, h)
    if s >= 1:
        return img
    return img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)


def to_jpeg_bytes(img: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _maybe_debug_save(cfg: Config, img: Image.Image) -> None:
    if not cfg.get("privacy", "debug_save_screenshots", False):
        return
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="ev_dbg_"))
    p = d / "shot.png"
    img.save(p)
    _debug_files.append(p)


@atexit.register
def _cleanup() -> None:
    for p in _debug_files:
        try:
            p.unlink(missing_ok=True)
            p.parent.rmdir()
        except OSError:
            pass
