"""Deterministic intent matching for common commands. Bypasses the LLM for speed/reliability."""
from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ev.tools.web import BROWSERS, SITE_ALIASES, SITES

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

# "open that", "open the screenshot you just took", "open the last file": these point at something, they don't name an app.
# "file explorer" is an app, so that one is let through.
_REFERENT = re.compile(
    r"\b(?:it|that|this|these|those|just|last|latest|recent|previous|newest|particular|you|one (?:i|we|that|from)"
    r"|screenshots?|screen ?shots?|screen captures?|files?(?! explorer)|pdfs?|images?|pictures?|documents?)\b")

_ARTIFACT_NOUN = r"(?:screenshot|screen ?shot|screen capture|picture|photo|image|file|note|document|pdf)"
_MADE = r"(?:(?:that |which )?(?:you|i) (?:just )?(?:took|made|saved|created|wrote|captured|grabbed)|just now)"

# "look up at the stars", "search me out": not a search query
_CHATTY_SEARCH = re.compile(r"(up|at|to|me|my|it|that|this|out|over|down|on)\b")

def _alt(words: list[str]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_SITE = rf"(?:{_alt([*SITES, *(a for a in SITE_ALIASES if len(a) > 1)])})(?:\.com)?"
_BROWSER = rf"(?:{_alt(list(BROWSERS))})"
_OPEN_V = r"(?:open|launch|start|go to|pull up|bring up)(?: up)?"
_AND = r"(?:\s*,\s*|\s+)(?:and\s+|then\s+)*"
_FIND_V = r"(?:search(?: for| up)?|look up|look for|find|play|watch)"
# Google/DuckDuckGo searches stay on web_search: the user wants the answer spoken, not a browser tab.
_NOT_ANSWER_SITES = ("google", "duckduckgo")

# "activate display mode" / "deactivate display mode": the on-screen text of the conversation
_DISPLAY = r"(?:display|captions?|subtitles?|transcript|text|text display)"
_ON_V = r"(?:activate|enable|turn on|switch on|start|show|enter)"
_OFF_V = r"(?:deactivate|disable|turn off|switch off|stop|hide|exit|leave|end|close)"

# Whisper keeps the "Yeah, so" / "okay, um" in front of a request, so a few leading fillers don't stop a match.
_PRE = (r"^\s*(?:(?:yeah|yes|yep|okay|ok|so|well|um|uh|and|now|alright|all right)[, ]+){0,2}"
        r"(?:(?:hey |ok(?:ay)? )?ev[, ]+)?(?:please |can you |could you |would you )?")
_POST = r"[\s.!?]*(?:please)?[\s.!?]*$"

# "remember that my dog's name is Biscuit": matched on the transcript as spoken (case kept), and the value is the user's own
# words. This is the one way a fact reaches memory without a spoken yes (see ev/safety/provenance.py).
_REMEMBER_MY = re.compile(
    _PRE + r"remember(?: that)?\s+my\s+(?P<k>[a-z0-9' -]{2,40}?)\s+(?:is|are|was|were)\s+(?P<v>.{1,200}?)" + _POST, re.I)
_REMEMBER_THAT = re.compile(_PRE + r"remember that\s+(?P<s>.{3,200}?)" + _POST, re.I)
# "don't forget my sister is Asha", "keep in mind that I work nights"
_REMEMBER_LEADING = re.compile(_PRE + r"(?:keep in mind|bear in mind|don'?t forget|do not forget)(?: that)?\s+(?P<s>.{3,200}?)" + _POST, re.I)
# "my dog's name is Biscuit, remember that" / "I'm allergic to peanuts, keep that in mind": the comma is what separates a
# request from "I will remember that" or "do you remember that".
_REMEMBER_TRAILING = re.compile(
    r"^\s*(?:(?:hey |ok(?:ay)? )?ev[, ]+)?(?P<s>.{6,200}?)[,.]\s+(?:please\s+)?"
    r"(?:remember (?:that|this|it)|keep (?:that|this|it) in mind|don'?t forget (?:that|it)|do not forget (?:that|it))" + _POST, re.I)
_NOT_A_FACT = re.compile(r"(?:to|i will|i'll|i can|i do|i did|you|we|do you|did you|can you|could you|would you|will you|should)\b", re.I)
_MY_X_IS_Y = re.compile(r"^my\s+(?P<k>[a-z0-9' -]{2,40}?)\s+(?:is|are|was|were)\s+(?P<v>.+)$", re.I)
_KEY_STOP = frozenset("i m s t ll ve d a an the my is are was were of to and that it in on at for".split())


def _fact_key(sentence: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", sentence.lower()) if w not in _KEY_STOP]
    return " ".join(words[:4]) or "note"


def _fact_from(sentence: str) -> dict[str, Any]:
    """remember_fact arguments for what the user said: "my X is Y" -> X / Y, anything else -> a short key and the sentence."""
    s = sentence.strip(" .!?,")
    m = _MY_X_IS_Y.match(s)
    return {"key": m["k"].strip(), "value": m["v"].strip(" .!?,")} if m else {"key": _fact_key(s), "value": s}


_WHEN = r"(?:today|yesterday|this week|last week|this month|last month|last night)"


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


_THIS = r"(?:this|that|it|this window|that window|the window|current window|active window|this app|that app)"


def _site_command(t: str) -> FastMatch | None:
    """"open youtube", "open firefox and search youtube for X", "search for X on reddit", "go to example.com"."""
    def args(site: str, q: str | None, b: str | None, url: str = "") -> dict[str, Any]:
        a: dict[str, Any] = {"site": site} if site else {}
        if url:
            a["url"] = url
        if q:
            a["query"] = q.strip(" .,")
        if b:
            a["browser"] = b
        return a

    m = re.search(
        _PRE + rf"(?:{_OPEN_V}\s+(?:the\s+)?(?P<b1>{_BROWSER})(?:{_AND}))?(?:{_OPEN_V}\s+(?:the\s+)?)?"
        rf"(?P<site>{_SITE})(?:\s+(?:in|on|with)\s+(?:the\s+)?(?P<b2>{_BROWSER}))?"
        rf"(?:{_AND}{_FIND_V}\s+(?:for\s+)?(?P<q>.+?))?(?:\s+(?:in|on|with)\s+(?:the\s+)?(?P<b3>{_BROWSER}))?" + _POST, t)
    if m and re.search(rf"\b{_OPEN_V}\b", t):
        return FastMatch("open_website", args(m["site"], m["q"], m["b1"] or m["b2"] or m["b3"]), "site")

    m = re.search(
        _PRE + rf"{_FIND_V}\s+(?:for\s+)?(?P<q>.+?)\s+(?:on|in|at)\s+(?:the\s+)?(?P<site>{_SITE})"
        rf"(?:\s+(?:in|on|with)\s+(?:the\s+)?(?P<b>{_BROWSER}))?" + _POST, t)
    if m is None:
        m = re.search(_PRE + rf"{_FIND_V}\s+(?:on\s+)?(?P<site>{_SITE})\s+for\s+(?P<q>.+?)"
                      rf"(?:\s+(?:in|on|with)\s+(?:the\s+)?(?P<b>{_BROWSER}))?" + _POST, t)
    if m and m["site"].removesuffix(".com") not in _NOT_ANSWER_SITES:
        return FastMatch("open_website", args(m["site"], m["q"], m["b"]), "site_search")

    m = re.search(_PRE + rf"(?:{_OPEN_V}|visit)\s+(?:the\s+)?(?P<b>{_BROWSER})(?:{_AND})(?:search(?: for| up)?|look up|google)\s+(?P<q>.+?)" + _POST, t)
    if m:
        return FastMatch("open_website", args("google", m["q"], m["b"]), "browser_search")

    m = re.search(_PRE + r"(?:open|go to|visit|launch)\s+(?P<d>[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|in|dev|ai|co|edu|gov))" + _POST, t)
    if m:
        return FastMatch("open_website", {"url": m["d"]}, "domain")
    return None


def _close_command(t: str) -> FastMatch | None:
    m = re.search(_PRE + r"(?P<f>force[ -]?)?(?P<v>close|quit|kill|exit)(?: down)?\s+(?:the |my )?(?P<n>[a-z0-9 .+-]{1,40}?)"
                  r"(?: app| application| window| program)?(?: for me)?(?: now| right now)?" + _POST, t)
    if not m:
        return None
    name = m["n"].strip()
    if re.fullmatch(_THIS, name):
        name = "this"
    elif re.match(r"(tabs?|this tab|that tab|the tab|browser|everything|all|up|out|down|in|off|on|a |an )\b|tabs?$", name):
        return None                      # tabs and vague targets go to the model, which can pick the right tool
    args: dict[str, Any] = {"name": name}
    if m["f"] or m["v"] == "kill":
        args["force"] = True
    return FastMatch("close_app", args, "close")


# Pure pleasantries get an instant, varied reply with no model call. Only whole utterances match, and never
# "yes/ok/sure" (those can be answers to a question EV just asked, so they need the model and the history).
_NAME = r"(?:\s+(?:ev|evie))?"
_SMALLTALK: list[tuple[str, tuple[str, ...]]] = [
    (rf"(?:thanks|thank you|thx|cheers)(?:\s+(?:so much|a lot|a ton|very much|again))?{_NAME}",
     ("Anytime!", "You got it.", "Happy to help.", "Of course!", "No problem.", "Anytime, really.")),
    (rf"(?:hi|hello|hey|hiya|howdy|yo|heya)(?:\s+there)?{_NAME}",
     ("Hey! What's up?", "Hi there. What do you need?", "Hey you. What's going on?", "Hello! I'm all ears.")),
    (rf"good\s+(?:morning|afternoon|evening){_NAME}",
     ("Hey, good to hear from you. What's the plan?", "Hi! What are we getting into?", "Hey there. What do you need?")),
    (rf"(?:good\s?night|night night|nighty night|sleep well){_NAME}",
     ("Night! Sleep well.", "Goodnight! Catch you tomorrow.", "Sleep tight!")),
    (rf"(?:bye|goodbye|bye bye|see you(?: later| soon| tomorrow)?|talk to you later|catch you later){_NAME}",
     ("Bye! Talk soon.", "See you!", "Catch you later.", "Later!")),
]
_SMALLTALK_RES = [(re.compile(_PRE + p + _POST), replies) for p, replies in _SMALLTALK]


def match_smalltalk(text: str, pick: Callable[[tuple[str, ...]], str] = random.choice) -> str | None:
    """A ready-made spoken reply for a bare "thanks" / "hello" / "goodbye", or None."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    for rx, replies in _SMALLTALK_RES:
        if rx.match(t):
            return pick(replies)
    return None


_CALL_ME = re.compile(_PRE + r"(?:from now on,? |going forward,? )?(?:you can |you should |please )?(?:just )?call me\s+(?P<n>[A-Za-z][A-Za-z'-]{1,19}(?: [A-Za-z][A-Za-z'-]{1,19})?)" + _POST, re.I)
_NOT_A_NAME = re.compile(r"(?:back|later|tomorrow|tonight|today|now|soon|when|if|at|in|on|after|before|please|again|up|maybe|first|then|a|an|the|my|me)(?![A-Za-z])", re.I)


def _memory_command(orig: str, t: str) -> FastMatch | None:
    """Remember, recall and review. Matched before the web-search and open patterns: "look up X in my notes" is a search of
    the user's own notes and must never be sent to a search engine. `orig` keeps the user's capitalisation for what is saved."""
    m = _CALL_ME.search(orig)             # "call me boss": the persona reads this fact (preferred_name) and uses the name
    if m and not _NOT_A_NAME.match(m["n"]):
        return FastMatch("remember_fact", {"key": "preferred_name", "value": re.sub(r"\s+please$", "", m["n"].strip(), flags=re.I)}, "preferred_name")
    m = _REMEMBER_MY.search(orig)
    if m:
        return FastMatch("remember_fact", {"key": m["k"].strip(), "value": m["v"].strip(" .!?,")}, "remember_my")
    m = _REMEMBER_THAT.search(orig)
    if m and not re.match(r"(?:to|when|how|why|what|if|you)\b", m["s"], re.I):
        return FastMatch("remember_fact", _fact_from(m["s"]), "remember_that")
    m = _REMEMBER_LEADING.search(orig)
    if m and not _NOT_A_FACT.match(m["s"]):
        return FastMatch("remember_fact", _fact_from(m["s"]), "remember_leading")
    m = _REMEMBER_TRAILING.search(orig)
    if m and not _NOT_A_FACT.match(m["s"]):
        return FastMatch("remember_fact", _fact_from(m["s"]), "remember_trailing")

    # the user's own notes and memory: "what did I say about the budget last week", "what do my notes say about Priya"
    # "Do you remember my dog Bistic?": the model will happily say yes to anything. Look in the notes, and the answer is
    # grounded in what is there (recall says "nothing" when there is nothing, and the model is told not to guess).
    m = re.search(_PRE + r"do you (?:still )?remember[?,]?\s+(?:about\s+)?(?!(?:that|it|this)\W*$)(?P<q>.{3,150}?)" + _POST, t)
    if m:
        return FastMatch("recall", {"query": m["q"].strip()}, "recall_remember", then_llm=True)
    m = re.search(_PRE + r"you (?:still )?remember\s+(?!(?:that|it|this)\W*$)(?P<q>.{3,150}?)(?:,?\s+(?:right|don'?t you|do you|huh))?"
                  + _POST, t)
    if m:                                                    # "you remember my dog, right?" is a question, not a command
        return FastMatch("recall", {"query": m["q"].strip()}, "recall_you_remember", then_llm=True)
    m = re.search(_PRE + r"what did (?:we|i) (?:talk|speak|chat) about\s+(?P<w>" + _WHEN + r")" + _POST, t)
    if m:                                                    # no topic, just a day: the newest notes from that day
        return FastMatch("recall", {"query": "", "when": m["w"]}, "recall_when", then_llm=True)
    m = (re.search(_PRE + r"what (?:did|have) (?:i|we) (?:say|said|tell|told|write|wrote|note|noted|mention|mentioned|talk(?:ed)?|"
                   r"discuss(?:ed)?)(?: (?:you|to you))? (?:about|on|regarding)\s+(?P<q>.+?)(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t)
         or re.search(_PRE + r"what do my (?:notes|memories|files) (?:say|have|mention|tell me)(?: about| on| regarding)\s+"
                      r"(?P<q>.+?)(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t)
         or re.search(_PRE + r"(?:search|look (?:through|in)|check|go through) my (?:notes|vault|memory|memories)(?: for| about)\s+"
                      r"(?P<q>.+?)(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t)
         or re.search(_PRE + r"what(?:'s| is| do i have| have i got) in my (?:notes|vault|memory|memories)(?: about| on| regarding)\s+"
                      r"(?P<q>.+?)(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t)
         or re.search(_PRE + r"(?:do|did) i have (?:any )?notes?(?: written)? (?:about|on|regarding)\s+"
                      r"(?P<q>.+?)(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t)
         or re.search(_PRE + r"(?:find|show|pull up|get)(?: me)?(?: my| the)? notes (?:about|on|regarding)\s+"
                      r"(?P<q>.+?)(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t)
         or re.search(_PRE + r"what notes do i have (?:about|on|regarding)\s+(?P<q>.+?)(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t)
         or re.search(_PRE + r"(?:look up|find|search for)\s+(?P<q>.+?)\s+in my (?:notes|vault|memory|memories)"
                      r"(?:\s+(?P<w>" + _WHEN + r"))?" + _POST, t))
    if m:
        return FastMatch("recall", {"query": m["q"].strip(), **({"when": m["w"]} if m["w"] else {})}, "recall", then_llm=True)
    if re.search(_PRE + r"(?:review|go through|check|list)(?: my| the| any)?(?: (?:unconfirmed|unverified|pending|unchecked))"
                 r" (?:facts|notes|memories)" + _POST, t) or re.search(
                 _PRE + r"(?:review|go through) the (?:facts|notes) i (?:haven'?t|have not) confirmed" + _POST, t):
        return FastMatch("review_facts", {}, "review_facts")
    return None


# "save this as a pdf" is a request about a file, not the name of a routine; "note time" or "text adventure" are fine names.
_NOT_A_SKILL_NAME = re.compile(r"(?:a |an |the )?(?:pdf|doc|docx|word|excel|text|txt|file|image|picture|png|jpg|csv|note|notes)"
                               r"(?: file| document)?", re.I)
_SKILL_NAME = r"(?P<n>[a-z0-9][a-z0-9 '_-]{1,39}?)"


def _skill_command(t: str) -> FastMatch | None:
    """Saving, naming, renaming, deleting and listing routines, by voice only: a skill is a standing automation, so the model
    can't do any of this itself (these tools are not in its tool list)."""
    m = re.search(_PRE + r"(?:save|store|keep|make) (?:that|this|it|what (?:i|we) (?:just )?did) as (?:a |an |the )?"
                  r"(?:(?:skill|routine|shortcut|macro) )?(?:called |named )?" + _SKILL_NAME + _POST, t)
    if m and not _NOT_A_SKILL_NAME.fullmatch(m["n"].strip()):
        return FastMatch("save_skill", {"name": m["n"].strip()}, "skill_save")
    m = re.search(_PRE + r"(?:also )?(?:run|start|do|trigger) (?:it|that|this) (?:when|if|whenever) i (?:say|ask for) (?P<p>.{2,60}?)" + _POST, t)
    if m:
        return FastMatch("manage_skill", {"action": "add_trigger", "name": "", "value": m["p"].strip(" '\"")}, "skill_trigger")
    m = re.search(_PRE + r"(?:add|use) (?:the )?(?:phrase|trigger) (?P<p>.{2,60}?) (?:for|to) (?:the )?(?:(?:skill|routine) )?"
                  r"(?P<n>[a-z0-9][a-z0-9 '_-]{1,39}?)" + _POST, t)
    if m:
        return FastMatch("manage_skill", {"action": "add_trigger", "name": m["n"].strip(), "value": m["p"].strip(" '\"")}, "skill_trigger")
    m = re.search(_PRE + r"rename (?:the |my )?(?:skill|routine) (?P<a>[a-z0-9][a-z0-9 '_-]{1,39}?) (?:to|as) (?P<b>[a-z0-9][a-z0-9 '_-]{1,39}?)"
                  + _POST, t)
    if m:
        return FastMatch("manage_skill", {"action": "rename", "name": m["a"].strip(), "value": m["b"].strip()}, "skill_rename")
    m = (re.search(_PRE + r"(?:delete|remove) (?:the |my )?(?:skill|routine) " + _SKILL_NAME + _POST, t)
         or re.search(_PRE + r"(?:delete|remove) (?:the |my )" + _SKILL_NAME + r" (?:skill|routine)" + _POST, t))
    if m:
        return FastMatch("manage_skill", {"action": "delete", "name": m["n"].strip()}, "skill_delete")
    if re.search(_PRE + r"(?:what|which) (?:skills|routines) (?:do (?:i|you) have|are there)|list (?:my |the )?(?:skills|routines)" + _POST, t):
        return FastMatch("manage_skill", {"action": "list"}, "skill_list")
    return None


def match_fast(text: str) -> FastMatch | None:
    orig = re.sub(r"\s+", " ", text.strip())
    t = orig.lower()
    t = re.sub(r"[.?!]+\s+(?=[a-z])", ", ", t)       # Whisper punctuates "open firefox? open youtube and ..." as sentences

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

    if re.search(_PRE + r"(?:open|show|pull up|bring up|launch)\s+(?:me\s+)?(?:the |my )?(?:evie |ev )?dashboard" + _POST, t):
        return FastMatch("open_dashboard", {}, "dashboard")

    m = re.search(_PRE + rf"(?:(?P<on>{_ON_V})|(?P<off>{_OFF_V}))\s+(?:the |my )?{_DISPLAY} mode" + _POST, t) or re.search(
        _PRE + rf"(?:turn |switch )?{_DISPLAY} mode (?:(?P<on>on)|(?P<off>off))" + _POST, t)
    if m:
        return FastMatch("set_display_mode", {"enabled": bool(m["on"])}, "display_mode")

    site_cmd = _site_command(t)
    if site_cmd:
        return site_cmd

    m = re.search(_PRE + r"(?:take|grab|capture|snap)(?: me)?(?: a| the)?\s+(?:screenshot|screen ?shot|screen capture|picture of (?:my |the )?screen)"
                  r"(?: of (?:my |the )?(?:screen|desktop|display))?(?: for me)?(?: now| right now)?" + _POST, t)
    if m or re.search(_PRE + r"screenshot(?: this| my screen| the screen| it)?" + _POST, t):
        return FastMatch("take_screenshot", {}, "screenshot")

    cm = _close_command(t)
    if cm:
        return cm

    mem = _memory_command(orig, t)
    if mem:
        return mem
    sk = _skill_command(t)
    if sk:
        return sk
    # "quiet for an hour": stop EV speaking up on its own (proactive lines) for a while; "you can talk again" ends it
    m = re.search(_PRE + r"(?:be quiet|stay quiet|go quiet|keep quiet|quiet|stop (?:interrupting|bothering) me|don'?t (?:interrupt|bother) me"
                  r"|snooze(?: the)?(?: proactive| notifications?| alerts?)?)(?: for)?(?: the next)?\s+" + _NUM + r"\s*" + _UNIT + _POST, t)
    if m and (d := _dur(m.group(1), m.group(2))):
        return FastMatch("snooze_proactive", {"minutes": min(720.0, float(d.get("minutes", d.get("seconds", 0) / 60)))}, "snooze")
    if re.search(_PRE + r"(?:you can (?:talk|speak|interrupt) (?:to me )?again|stop being quiet|unsnooze|resume (?:the )?(?:proactive|notifications?|alerts?))"
                 + _POST, t):
        return FastMatch("snooze_proactive", {"minutes": 0.0}, "unsnooze")
    if re.search(_PRE + r"(?:what(?:'s| is| are) (?:on )?my (?:reminders?|reminder list)|list (?:my )?reminders|do i have any reminders)"
                 + _POST, t):
        return FastMatch("list_reminders", {}, "list_reminders")

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

    # the file EV last made: "open it", "open the screenshot you just took", "show me the last note"
    if re.search(
        _PRE + r"(?:open|show|pull up|bring up|display)(?: up)?(?: me)?\s+(?:(?:it|that|this)(?: (?:file|one))?"
        rf"|(?:the |that |this |my )?(?:(?:last|latest|most recent|previous|newest|particular)\s+)*{_ARTIFACT_NOUN}(?: {_MADE})?)"
        r"(?: again| now)?" + _POST, t):
        return FastMatch("open_file", {"which": "last"}, "open_file")

    m = re.search(_PRE + r"(?:open|launch|start|run)(?: up)?\s+(?:the )?(?:app )?([a-z0-9 .+-]{2,40}?)(?: app| application| browser)?"
                  r"(?:\s+(?:in|on) (?:my |the )?(?:pc|computer|laptop|machine|system))?(?: for me)?(?: right now| now)?" + _POST, t)
    if (m and not re.match(r"(a |an )?(timer|reminder)", m.group(1)) and not _CHATTY_OPEN.match(m.group(1))
            and not _REFERENT.search(m.group(1))):
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


# Question-shaped speech: the only turns worth a planner call (ev/llm/brain.py::_plan_lookup). Deliberately loose; the
# planner, not this pattern, decides whether the web is needed.
_QUESTION = re.compile(
    r"\?|\b(?:who|whom|whose|what|which|when|where|why|how|is there|are there|was there|were there|did|does|do you (?:know|have)|"
    r"any idea|tell me|heard|news|happening|happened)\b", re.I)


_ABOUT_EV = re.compile(r"\b(?:how are you|how'?s it going|what'?s up|who are you|what can you do|your name|are you (?:there|ok|okay|real))\b", re.I)


def looks_like_question(text: str) -> bool:
    return len(text.split()) >= 4 and bool(_QUESTION.search(text)) and not _ABOUT_EV.search(text)
