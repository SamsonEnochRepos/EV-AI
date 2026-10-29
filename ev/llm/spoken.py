"""Turn model text into something safe to speak: no markdown, lists, emojis, symbols or URLs.

The persona asks for this, but a 3B model slips, so it is enforced here on every sentence before TTS.
"""
from __future__ import annotations

import re

_URL = re.compile(r"(?:https?://|www\.)([^\s/<>\"')]+)(?:/[^\s<>\"')]*)?", re.I)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
_BULLET = re.compile(r"(?:^|(?<=\s))(?:[-•*–—]|\d{1,2}[.)])\s+(?=[A-Z0-9\"'(])")
_LIST_BREAK = re.compile(r"([^.!?:;,\s])\s+(?:[-•*–—]|\d{1,2}[.)])\s+(?=[A-Z])")
_MD = re.compile(r"(\*\*|__|`+|^#+\s*|~~)", re.M)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")


def _domain(m: re.Match[str]) -> str:
    host = re.sub(r"^www\.", "", m.group(1))
    return host.replace(".", " dot ")


def spoken_text(text: str) -> str:
    t = _URL.sub(_domain, text)
    t = _EMOJI.sub("", t)
    t = _MD.sub("", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)     # markdown links keep their label
    t = _LIST_BREAK.sub(r"\1. ", t)    # "item one - item two" -> "item one. item two"
    t = _BULLET.sub(" ", t)
    t = _PERCENT.sub(r"\1 percent", t)
    t = _MONEY.sub(lambda m: m.group(1).replace(",", "") + " dollars", t)
    t = t.replace("&", " and ").replace("@", " at ")
    t = re.sub(r"\s+", " ", t).strip()
    return t
