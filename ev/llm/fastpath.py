"""Deterministic intent matching for common commands. Bypasses the LLM for speed/reliability."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
    "half": 0.5,
}


@dataclass
class FastMatch:
    tool: str
    args: dict[str, Any]
    pattern: str
    then_llm: bool = False      # run the tool, then let the LLM phrase the answer (e.g. web search results)


def _num(s: str) -> float | None:
    s = s.strip().lower().replace("-", " ")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    if s in _NUM_WORDS:
        return float(_NUM_WORDS[s])
    m = re.fullmatch(r"(twenty|thirty|forty|fifty|sixty)[ ](one|two|three|four|five|six|seven|eight|nine)", s)
    if m:
        return float(_NUM_WORDS[m.group(1)] + _NUM_WORDS[m.group(2)])
    return None


_NUM = r"(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty[ -]five|forty|fifty|sixty|ninety|half)"
_UNIT = r"(hours?|hrs?|minutes?|mins?|seconds?|secs?)"

# "open up to me", "start my day", "run a test": conversational, not an app name
_CHATTY_OPEN = re.compile(r"(up|to|my|me|your|our|it|that|this|these|those|about|a|an|with|for|on|in|out|over|down|into)\b")

# "look up at the stars", "search me out": not a search query
_CHATTY_SEARCH = re.compile(r"(up|at|to|me|my|it|that|this|out|over|down|on)\b")

_PRE = r"^\s*(?:(?:hey |ok(?:ay)? )?ev[, ]+)?(?:please |can you |could you |would you )?"
_POST = r"[\s.!?]*(?:please)?[\s.!?]*$"


def _dur(n: str, unit: str) -> dict[str, Any] | None:
    v = _num(n)
    if v is None:
        return None
    u = unit.lower()
    if u.startswith("h"):
        return {"minutes": v * 60}
    if u.startswith("m"):
        return {"minutes": v}
    return {"seconds": int(v)}


def match_fast(text: str) -> FastMatch | None:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)

    if re.search(_PRE + r"(what(?:'s| is)? the time|what time is it|what(?:'s| is)? the date(?: today)?|what day is it(?: today)?|what(?:'s| is)? today'?s date|tell me the time|what(?:'s| is)? the (?:current )?time(?: right now| now)?)" + _POST, t):
        return FastMatch("get_time_date", {}, "time")

    m = re.search(_PRE + r"(?:set|start|create)\s+(?:a |an )?(?:(\w+) )?timer (?:for|of)\s+" + _NUM + r"\s*" + _UNIT + _POST, t)
    if m:
        d = _dur(m.group(2), m.group(3))
        if d:
            return FastMatch("set_timer", d, "timer")
    m = re.search(_PRE + r"(?:set|start)\s+(?:a |an )?" + _NUM + r"[ -]?" + _UNIT + r"[ -]?timer" + _POST, t)
    if m:
        d = _dur(m.group(1), m.group(2))
        if d:
            return FastMatch("set_timer", d, "timer2")
    m = re.search(_PRE + r"(?:remind me|set a reminder)\s+in\s+" + _NUM + r"\s*" + _UNIT + r"\s+(?:to|that|about)\s+(.+?)" + _POST, t)
    if m:
        d = _dur(m.group(1), m.group(2))
        if d and "minutes" in d:
            return FastMatch("set_reminder", {"text": m.group(3), "in_minutes": d["minutes"]}, "reminder_in")
        if d:
            return FastMatch("set_reminder", {"text": m.group(3), "in_minutes": d["seconds"] / 60}, "reminder_in")

    m = re.search(
        _PRE + r"(?:search(?: the (?:web|internet))?(?: online)?(?: for| up)?|google|look up|look online for|search up)\s+"
        r"(.{2,120}?)" + _POST, t)
    if m and not _CHATTY_SEARCH.match(m.group(1)):
        return FastMatch("web_search", {"query": m.group(1).strip()}, "search", then_llm=True)
    _day = r"(today|tomorrow|right now|now|tonight)"
    m = re.search(
        _PRE + r"(?:what(?:'s| is)|how(?:'s| is)|tell me|give me|check)?\s*(?:the |today'?s )?(?:weather|temperature|forecast)"
        r"(?: like)?(?: " + _day + r")?(?: (?:in|for|at) (.{2,60}?))?(?: " + _day + r")?" + _POST, t)
    if m:
        args = {"day": "tomorrow" if "tomorrow" in (m.group(1), m.group(3)) else "today"}
        if m.group(2):
            args["place"] = m.group(2).strip()
        return FastMatch("get_weather", args, "weather")

    m = re.search(
        _PRE + r"(?:what(?:'s| is) (?:on|in) (?:my |the )?(?:screen|display|monitor)"
        r"|(?:see|look at|read|check|describe) (?:my |the )?(?:screen|display)"
        r"|what am i (?:looking at|seeing)|what do you see(?: on (?:my |the )?screen)?|read (?:this|the) (?:page|window))"
        r"(?: right now| now| for me)?" + _POST, t)
    if m:
        return FastMatch("read_screen", {}, "screen", then_llm=True)

    m = re.search(_PRE + r"(?:open|launch|start|run)(?: up)?\s+(?:the )?(?:app )?([a-z0-9 .+-]{2,40}?)(?: app| application| browser)?"
                  r"(?:\s+(?:in|on) (?:my |the )?(?:pc|computer|laptop|machine|system))?(?: for me)?(?: right now| now)?" + _POST, t)
    if m and not re.match(r"(a |an )?(timer|reminder)", m.group(1)) and not _CHATTY_OPEN.match(m.group(1)):
        return FastMatch("open_app", {"name": m.group(1).strip()}, "open")

    _where = r"(?: in here| right now| a bit| a little)?"
    if re.search(
        _PRE + r"(?:turn (?:the )?(?:volume )?up|volume up|louder|raise (?:the )?volume|increase (?:the )?volume"
        r"|turn it up|make it (?:a bit |way |much )?louder"
        r"|(?:it's|it is|that's) (?:way |much |a bit )?(?:too )?(?:quiet|low|soft)"
        r"|i can't hear (?:anything|you|it))" + _where + _POST, t):
        return FastMatch("set_volume", {"direction": "up"}, "vol_up")
    if re.search(
        _PRE + r"(?:turn (?:the )?(?:volume )?down|volume down|quieter|lower (?:the )?volume|decrease (?:the )?volume"
        r"|turn it down|make it (?:a bit |way |much )?quieter"
        r"|(?:it's|it is|that's) (?:way |much |a bit )?(?:too )?(?:loud|noisy))" + _where + _POST, t):
        return FastMatch("set_volume", {"direction": "down"}, "vol_down")
    if re.search(_PRE + r"(?:mute|unmute)(?: (?:the )?(?:sound|volume|audio|it))?" + _POST, t):
        return FastMatch("set_volume", {"direction": "mute"}, "mute")
    m = re.search(_PRE + r"(?:set|put|turn) (?:the )?volume (?:to|at)\s+(\d{1,3})\s*(?:percent|%)?" + _POST, t)
    if m and int(m.group(1)) <= 100:
        return FastMatch("set_volume", {"level": int(m.group(1))}, "vol_set")

    if re.search(_PRE + r"(?:pause|resume|play|unpause)(?: the| my)?(?: music| song| video| media| playback)?" + _POST, t) and not re.search(r"play \w+ \w+", t):
        return FastMatch("media_control", {"action": "play_pause"}, "playpause")
    if re.search(_PRE + r"(?:next|skip)(?: the)?(?: song| track)?" + _POST, t):
        return FastMatch("media_control", {"action": "next"}, "next")
    if re.search(_PRE + r"(?:previous|go back)(?: song| track)?" + _POST, t):
        return FastMatch("media_control", {"action": "previous"}, "prev")

    m = re.search(
        _PRE + r"(?:(?:forget|stop remembering) (?:that )?my (.{2,60}?)(?: from (?:your )?memory)?"
        r"|(?:forget|delete|remove) (?:the |that )?(.{2,60}?) from (?:your )?memory)" + _POST, t)
    key = (m.group(1) or m.group(2)) if m else None
    if key and not re.match(r"(it|that|this|about|what|everything|all|files?|folders?|documents?)\b", key):
        return FastMatch("forget_fact", {"key": key.strip()}, "forget")

    if re.search(_PRE + r"copy (?:that|this|it|the selection|selection)" + _POST, t):
        return FastMatch("copy_selection", {}, "copy")
    if re.search(_PRE + r"paste(?: it| that| this)?(?: in| here)?" + _POST, t):
        return FastMatch("paste", {}, "paste")

    m = re.search(_PRE + r"(?:take a note|note down|make a note)(?: that| to)?[: ]+(.+?)" + _POST, t)
    if m:
        return FastMatch("take_note", {"text": m.group(1)}, "note")

    return None
