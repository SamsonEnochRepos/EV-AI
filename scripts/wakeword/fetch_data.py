"""Download the public audio used as negatives / background noise for wake-word training.

Training-time only (EV itself never downloads anything at runtime). Everything lands in data/wakeword/raw (git-ignored).

  LibriSpeech dev-clean   ~340 MB  real read speech, used to train against false triggers
  LibriSpeech test-clean  ~350 MB  held out, only used to *measure* false triggers per hour
  ESC-50                  ~600 MB  2000 five-second environmental noise clips, used for background mixing
"""
from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "wakeword" / "raw"

SOURCES = {
    "librispeech-dev": ("https://www.openslr.org/resources/12/dev-clean.tar.gz", "tar"),
    "librispeech-test": ("https://www.openslr.org/resources/12/test-clean.tar.gz", "tar"),
    "esc50": ("https://github.com/karoldvl/ESC-50/archive/master.zip", "zip"),
}
ALLOWED = ("https://www.openslr.org/", "https://github.com/karoldvl/ESC-50/")


def fetch(name: str) -> None:
    url, kind = SOURCES[name]
    if not url.startswith(ALLOWED):
        raise ValueError(url)
    out = RAW / name
    if (out / ".done").exists():
        print(f"{name}: already there")
        return
    out.mkdir(parents=True, exist_ok=True)
    archive = RAW / f"{name}.{kind}"
    if not archive.exists():
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, archive)  # noqa: S310  (host pinned above)
    print(f"{name}: extracting")
    if kind == "tar":
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    archive.unlink()
    (out / ".done").write_text("ok")


if __name__ == "__main__":
    for n in SOURCES:
        fetch(n)
    print("done")
