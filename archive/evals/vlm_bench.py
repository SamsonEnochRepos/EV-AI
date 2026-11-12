"""VLM candidates on rendered UI-like images: text reading, button listing, speed, VRAM and swap cost."""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont

from .bench import vram_mb
from .common import RESULTS

FONT = "C:/Windows/Fonts/segoeui.ttf"


def make_images() -> list[tuple[str, Image.Image, str, list[str]]]:
    f = ImageFont.truetype(FONT, 26)
    fs = ImageFont.truetype(FONT, 18)
    a = Image.new("RGB", (900, 500), "#f3f3f3")
    d = ImageDraw.Draw(a)
    d.rectangle((0, 0, 900, 40), fill="#0078d4")
    d.text((12, 6), "Untitled - Notepad", fill="white", font=fs)
    d.text((30, 80), "Meeting moved to Thursday at 3 PM", fill="black", font=f)
    d.text((30, 130), "Order number 48213", fill="black", font=f)
    for i, label in enumerate(["Save", "Cancel", "Delete"]):
        d.rounded_rectangle((30 + i * 160, 400, 170 + i * 160, 450), radius=6, fill="#e1e1e1", outline="#888")
        d.text((60 + i * 160, 412), label, fill="black", font=f)
    b = Image.new("RGB", (900, 500), "white")
    d2 = ImageDraw.Draw(b)
    d2.text((40, 40), "Settings", fill="black", font=f)
    for i, t in enumerate(["Wi-Fi: connected", "Bluetooth: off", "Battery: 87 percent"]):
        d2.text((60, 110 + i * 60), t, fill="black", font=f)
    return [
        ("read_order_number", a, "What is the order number shown? Answer with just the number.", ["48213"]),
        ("read_meeting_day", a, "On which day was the meeting moved to?", ["thursday"]),
        ("list_buttons", a, "List the buttons at the bottom of the window.", ["save", "cancel", "delete"]),
        ("battery", b, "What is the battery level?", ["87"]),
        ("bluetooth_state", b, "Is Bluetooth on or off?", ["off"]),
    ]


def ask(model: str, img: Image.Image, q: str) -> tuple[str, float, float]:
    buf = io.BytesIO()
    img.resize((768, int(768 * img.size[1] / img.size[0]))).save(buf, "JPEG", quality=85)
    body = {"model": model, "stream": False, "keep_alive": "5m",
            "messages": [{"role": "user", "content": q, "images": [base64.b64encode(buf.getvalue()).decode()]}],
            "options": {"num_ctx": 2048, "num_predict": 120, "temperature": 0.1}}
    t0 = time.perf_counter()
    r = httpx.post("http://127.0.0.1:11434/api/chat", json=body, timeout=300).json()
    return r.get("message", {}).get("content", ""), time.perf_counter() - t0, r.get("load_duration", 0) / 1e9


def main() -> int:
    out = {}
    for model in sys.argv[1:] or ["moondream", "qwen2.5vl:3b"]:
        httpx.post("http://127.0.0.1:11434/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        time.sleep(1.0)
        rows: list[dict[str, Any]] = []
        vr: list[int] = []
        first_load = None
        for name, img, q, want in make_images():
            ans, secs, load = ask(model, img, q)
            if first_load is None:
                first_load = load
            ok = all(w in ans.lower() for w in want)
            rows.append({"task": name, "ok": ok, "answer": ans[:100], "seconds": round(secs, 2)})
            v = vram_mb()
            if v:
                vr.append(v)
        ps = httpx.get("http://127.0.0.1:11434/api/ps", timeout=5).json().get("models", [])
        mine = [m for m in ps if m["name"].startswith(model.split(":")[0])]
        warm = [r["seconds"] for r in rows[1:]]
        out[model] = {"accuracy": sum(r["ok"] for r in rows) / len(rows), "first_load_s": round(first_load or 0, 2),
                      "warm_s_median": sorted(warm)[len(warm) // 2], "peak_vram_mb_total": max(vr) if vr else None,
                      "spill_pct": round(100 * (1 - mine[0]["size_vram"] / mine[0]["size"]), 1) if mine else None,
                      "rows": rows}
        httpx.post("http://127.0.0.1:11434/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        print(model, {k: v for k, v in out[model].items() if k != "rows"})
        for r in rows:
            print("   ", r)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bench_vlm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
