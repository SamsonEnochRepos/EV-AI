"""Incremental sentence splitter for streaming LLM output into TTS."""
from __future__ import annotations

import re

_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|(?<=[.!?])[\"')\]]*$|\n+")
_ABBR = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|vs|etc|e\.g|i\.e|a\.m|p\.m)\.$", re.I)


class SentenceSplitter:
    def __init__(self, min_chars: int = 12):
        self.buf = ""
        self.min_chars = min_chars

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            m = None
            for cand in _END.finditer(self.buf):
                head = self.buf[: cand.end()].strip()
                if cand.end() >= len(self.buf) and not self.buf.endswith(("\n", " ")):
                    break  # trailing punctuation without following whitespace: wait for more text
                if _ABBR.search(head) or re.search(r"\d\.$", head):
                    continue
                if len(head) < self.min_chars and out == [] and "\n" not in cand.group(0):
                    continue
                m = cand
                break
            if not m:
                break
            sent = self.buf[: m.end()].strip()
            self.buf = self.buf[m.end():]
            if sent:
                out.append(sent)
        return out

    def flush(self) -> list[str]:
        rest = self.buf.strip()
        self.buf = ""
        return [rest] if rest else []
