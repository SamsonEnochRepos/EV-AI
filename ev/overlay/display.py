"""Display mode model: what the card under the pill says. Toolkit-independent so it is unit-testable."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

ELLIPSIS = "… "


@dataclass
class Exchange:
    """The latest thing the user said and EV's reply to it."""

    user: str = ""
    ev: str = ""

    @property
    def empty(self) -> bool:
        return not (self.user or self.ev)

    def clear(self) -> None:
        self.user = self.ev = ""

    def apply(self, who: str, text: str, append: bool = False) -> None:
        """A new user line starts a new exchange; EV lines build up the reply sentence by sentence."""
        text = " ".join(text.split())
        if who == "user":
            self.user, self.ev = text, ""
        elif append and self.ev:
            self.ev = f"{self.ev} {text}".strip()
        else:
            self.ev = text


def tail_that_fits(text: str, fits: Callable[[str], bool]) -> str:
    """The end of `text` that fits: the whole text if it does, else the longest word-aligned ending with a leading
    ellipsis. A reply that outgrows the card keeps scrolling, so the newest words are always visible."""
    if fits(text):
        return text
    words = text.split(" ")
    for i in range(1, len(words)):
        cand = ELLIPSIS + " ".join(words[i:])
        if fits(cand):
            return cand
    return ELLIPSIS + words[-1]
