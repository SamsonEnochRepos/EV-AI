"""Download free models from official sources and record SHA-256 checksums.

Piper voices come from the rhasspy/piper-voices Hugging Face repo. Whisper models are fetched
by faster-whisper from Systran on Hugging Face on first use (or pass --whisper).
Ollama models are pulled with `ollama pull` (see README).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

VOICES = {
    "en_US-amy-medium": "en/en_US/amy/medium",
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_US-kristin-medium": "en/en_US/kristin/medium",
    "en_GB-jenny_dioco-medium": "en/en_GB/jenny_dioco/medium",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    if not url.startswith("https://huggingface.co/"):
        raise ValueError("only official Hugging Face URLs are allowed")
    if dest.exists():
        return
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310  (host pinned above)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en_US-amy-medium", choices=sorted(VOICES))
    ap.add_argument("--whisper", nargs="*", default=[], help="e.g. base small")
    a = ap.parse_args()

    out = ROOT / "models" / "piper"
    out.mkdir(parents=True, exist_ok=True)
    base = f"{HF}/{VOICES[a.voice]}/{a.voice}"
    for ext in (".onnx", ".onnx.json"):
        fetch(base + ext, out / (a.voice + ext))

    sums_path = ROOT / "models" / "checksums.json"
    sums = json.loads(sums_path.read_text()) if sums_path.exists() else {}
    for f in out.glob(a.voice + "*"):
        sums[f"piper/{f.name}"] = sha256(f)

    for name in a.whisper:
        from faster_whisper.utils import download_model  # type: ignore[import-untyped]

        d = Path(download_model(name))
        for f in d.glob("*"):
            if f.is_file():
                sums[f"whisper-{name}/{f.name}"] = sha256(f)
    sums_path.write_text(json.dumps(sums, indent=2))
    print(f"checksums recorded in {sums_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
