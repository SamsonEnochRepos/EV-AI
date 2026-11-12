"""Did the speaker stop because they were finished, or because they paused to think?

A pause of half a second is normal in the middle of a sentence ("open the dashboard ... in Firefox"). The recorder can only
hear silence, so after it ends a recording EV reads what was transcribed: if it stops on a word no sentence ends on ("the",
"and", "in", "can you"), EV keeps listening a little longer and joins what comes next. Only the words are looked at; this
never decides anything about what EV is allowed to do.
"""
from __future__ import annotations

import re

# Words a finished request does not end on.
_DANGLING = frozenset("""a an the and or but so if because then than as of to in on at for with from by about into onto over
under between through my your his her their our its this these those some any no um uh er erm hmm like also is are was were am
open close launch search find tell show take write type click press remind call send switch put remember forget make
turn set add what which who whose how why when where whether not""".split())

# "can you", "i want to", "let me" ...: a request that has been announced but not made.
_OPENER_ONLY = re.compile(
    r"(?:^|\b)(?:(?:hey|oh|ok(?:ay)?|so|well|um|uh|and|but|also|now|alright)\s+)*"
    r"(?:(?:ev|evie)\s+)?"
    r"(?:(?:can|could|would|will)\s+you(?:\s+please)?|please|i\s+(?:want|need|would\s+like|'d\s+like|wanna|have)\s+(?:you\s+)?to|"
    r"i\s+want|i\s+need|let\s+me|let's|how\s+do\s+i|how\s+do\s+you|what\s+is|what's|tell\s+me|do\s+you\s+know|"
    r"i\s+was\s+(?:wondering|thinking)|is\s+there)\s*$", re.I)
_JUST_A_NAME = re.compile(r"^(?:(?:hey|oh|ok(?:ay)?|so|well|um|uh)\s+)?(?:ev|evie)$", re.I)


def looks_unfinished(text: str) -> bool:
    """True when the transcript reads like the middle of a sentence."""
    raw = text.strip()
    if not raw:
        return False
    if raw.endswith((",", "-", "—", "...", "…")):       # Whisper writes a trailing comma / dash when the voice hangs
        return True
    words = re.findall(r"[a-z0-9']+", raw.lower())
    if not words:
        return False
    plain = " ".join(words)
    if _JUST_A_NAME.match(plain) or _OPENER_ONLY.search(plain):
        return True
    # A single word is a whole answer ("yes", "what?", "stop"), so only the longer ones are judged by their last word.
    return len(words) >= 2 and words[-1] in _DANGLING
