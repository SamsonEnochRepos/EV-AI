"""Incremental sentence splitter for streaming LLM output into TTS."""
from __future__ import annotations

import re

_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|(?<=[.!?])[\"')\]]*$|\n+")
_ABBR = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|vs|etc|e\.g|i\.e|a\.m|p\.m)\.$", re.I)
_CLAUSE = re.compile(r"[,;:]\s+")        # needs the space, so "1,000" is never split


class SentenceSplitter:
    """feed() text as it streams in; complete sentences come out as soon as they are known.

    clause_chars > 0 lets the *first* chunk of a reply end at a comma once it is at least that long, so speech starts
    a beat earlier on a long opening sentence. Later chunks are always whole sentences (a clause chunk is flagged so
    callers don't count it as a sentence).
    """

    def __init__(self, min_chars: int = 12, clause_chars: int = 0):
        self.buf = ""
        self.min_chars = min_chars
        self.clause_chars = clause_chars
        self._emitted = False

    def feed(self, text: str) -> list[str]:
        return [s for s, _ in self.feed_ex(text)]

    def feed_ex(self, text: str) -> list[tuple[str, bool]]:
        """Like feed(), but each chunk comes with a flag: True if it is a whole sentence, False for an early clause."""
        self.buf += text
        out: list[tuple[str, bool]] = []
        while True:
            m = None
            for cand in _END.finditer(self.buf):
                head = self.buf[: cand.end()].strip()
                if cand.end() >= len(self.buf) and not self.buf.endswith(("\n", " ")):
                    break  # trailing punctuation without following whitespace: wait for more text
                if _ABBR.search(head) or re.search(r"\d\.$", head):
                    continue
                if len(head) < self.min_chars and not out and "\n" not in cand.group(0):
                    continue
                m = cand
                break
            if m:
                sent = self.buf[: m.end()].strip()
                self.buf = self.buf[m.end():]
                if sent:
                    out.append((sent, True))
                    self._emitted = True
                continue
            clause = self._early_clause()
            if clause is None:
                break
            out.append((clause, False))
            self._emitted = True
        return out

    def _early_clause(self) -> str | None:
        """Split the first chunk at a comma when the sentence is long and no full stop has arrived yet."""
        if not self.clause_chars or self._emitted:
            return None
        for cand in _CLAUSE.finditer(self.buf):
            if cand.end() >= len(self.buf):
                break                                    # nothing after the comma yet; wait for more text
            head = self.buf[: cand.start() + 1].strip()
            if len(head) >= self.clause_chars:
                self.buf = self.buf[cand.end():]
                return head
        return None

    def flush(self) -> list[str]:
        rest = self.buf.strip()
        self.buf = ""
        return [rest] if rest else []
