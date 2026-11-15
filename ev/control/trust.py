"""Reading Claude Code's "do you trust this folder?" screen from a terminal's text.

Nothing here presses a key. It only says whether the text on screen really is that prompt, which folder it is about, and
whether "Yes, I trust this folder" is the highlighted choice. The tool that acts on it (control/tools.py::trust_agent_folder)
re-checks the folder against policy before it presses anything.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MARK = "❯"                                   # the heavy right-pointing angle that marks the highlighted choice
_YES = re.compile(rf"^(?P<sel>[{_MARK}>›»]\s*)?(?:1\.\s*)?Yes, I trust this folder\b", re.I)
_QUESTION = re.compile(r"Is this a project you created or one you trust", re.I)
_PATH = re.compile(r"^[A-Za-z]:\\[^\r\n<>\"|?*]{1,240}$")


@dataclass(frozen=True)
class TrustPrompt:
    path: str              # the folder the prompt names, as printed
    yes_selected: bool     # "Yes, I trust this folder" is the highlighted choice, so Enter would take it


def parse_claude_trust(text: str) -> TrustPrompt | None:
    """The last trust prompt in `text`, or None. All three parts must be there: the "Accessing workspace:" line with a
    full Windows path under it, the question, and the "Yes, I trust this folder" choice. Anything less is not the prompt."""
    marker = "Accessing workspace:"
    at = text.rfind(marker)
    if at < 0:
        return None
    lines = [ln.strip() for ln in text[at + len(marker):].splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines or not _PATH.match(lines[0]):
        return None
    rest = lines[1:]
    if not any(_QUESTION.search(ln) for ln in rest):
        return None
    yes = next((m for ln in rest if (m := _YES.match(ln))), None)
    if yes is None:
        return None
    return TrustPrompt(lines[0].rstrip("\\"), bool(yes["sel"]))
