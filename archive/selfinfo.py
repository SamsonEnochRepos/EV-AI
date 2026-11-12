"""What EV knows about itself: its models, voice, hearing, and how it was built.

Every sentence here is made from the live config (and, where it can differ, from what is really loaded: the speech model's
device, the voice that survived a fallback, the chat model that was picked), so it never goes stale and never has to be guessed
by the 3B chat model, which gets "what's your voice model?" wrong (it answers about the foreground app instead).

Two ways it reaches the user, both system text and never the model's own claim:
  * `match_self` in fastpath.py routes a plain question ("what text to speech model do you use?") to the voice-only
    `about_self` tool, whose reply is `SelfInfo.answer(topic)`;
  * for a looser self question that no pattern caught, `Brain._beside_the_request` puts `SelfInfo.block()` beside the request.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ev.core.config import Config

TOPICS = ("overview", "llm", "stt", "tts", "wake", "vision", "memory", "build", "safety")


def spoken_model(name: str) -> str:
    """"qwen2.5:3b" -> "Qwen 2 point 5, 3 billion". Model ids read badly aloud."""
    base, _, tag = name.strip().partition(":")
    base = re.sub(r"([a-z])(\d)", r"\1 \2", base.lower())
    base = re.sub(r"(\d)\s*\.\s*(\d)", r"\1 point \2", base)
    base = re.sub(r"(\d)(vl)\b", r"\1 \2", base)
    words = base.replace("-", " ").replace("_", " ").split()
    out = " ".join(w if w == "point" or w[0].isdigit() else "VL" if w == "vl" else w.capitalize() for w in words)
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)b", tag):
        out += f", {m[1]} billion"
    return out


def spoken_voice(name: str) -> str:
    """"en_US-hfc_female-medium" -> "hfc female, medium quality"."""
    parts = name.split("-")
    if len(parts) == 3:
        return f"{parts[1].replace('_', ' ')}, {parts[2]} quality"
    return name.replace("_", " ").replace("-", " ")


class SelfInfo:
    def __init__(self, cfg: Config, live: Callable[[], dict[str, Any]] | None = None):
        self.cfg = cfg
        self._live = live or (lambda: {})

    def _l(self) -> dict[str, Any]:
        try:
            return self._live() or {}
        except Exception:                                                       # noqa: BLE001  (never break a turn over a status read)
            return {}

    # ---- one spoken answer per topic
    def llm(self) -> str:
        live = self._l()
        c = self.cfg["llm"]
        main = str(live.get("llm_model") or c["model"])
        s = f"I think with {spoken_model(main)}, a small language model running on this computer through Ollama."
        if c.get("fallback_model"):
            s += f" If that's missing I fall back to {spoken_model(str(c['fallback_model']))}."
        if c.get("deep_model"):
            s += f" When you ask me to think harder I borrow the bigger {spoken_model(str(c['deep_model']))}."
        return s

    def stt(self) -> str:
        live = self._l()
        model = str(live.get("stt_model") or self.cfg["stt"]["model"])
        where = {"cuda": "on the graphics card", "cpu": "on the processor"}.get(str(live.get("stt_device", "")), "")
        return f"I hear you with Whisper {model}, run through faster-whisper{' ' + where if where else ''}. It all stays on this machine."

    def tts(self) -> str:
        live = self._l()
        t = self.cfg["tts"]
        engine = str(live.get("tts_engine") or t["engine"])
        if engine == "kokoro":
            return f"I speak with Kokoro, the {t['kokoro_voice'].replace('_', ' ')} voice, generated locally."
        return f"I speak with Piper, using the {spoken_voice(str(t['piper_voice']))} voice, generated locally on this machine."

    def wake(self) -> str:
        t = self.cfg["trigger"]
        if t.get("mode") != "wakeword":
            return f"I start listening when you press {self.cfg['hotkeys']['push_to_talk']}."
        name = str(t["wakeword_model"]).replace("_", " ")
        return f"I wake on '{name}', a small model that runs through openWakeWord. The push to talk hotkey works too."

    def vision(self) -> str:
        v = self.cfg["vlm"]["model"]
        s = f"To read the screen I try the Windows accessibility tree first, then RapidOCR, then the {spoken_model(str(v))} vision model."
        if self.cfg.get("camera", "enabled", False):
            s += " The camera is on request only, one frame at a time."
        return s

    def memory(self) -> str:
        return ("What I remember lives in plain Markdown files in your notes vault, searched with SQLite. "
                "Reminders and timers are in a small local database. Nothing goes to the cloud.")

    def build(self) -> str:
        return ("I'm written in Python and run entirely on this laptop. Hearing is Whisper through faster-whisper. "
                f"Thinking is {spoken_model(str(self._l().get('llm_model') or self.cfg['llm']['model']))} on Ollama, "
                "speaking is Piper, and the little pill on screen is built with PySide6. "
                "Every action goes through a safety gate first.")

    def safety(self) -> str:
        return ("Every action I take goes through a safety gate that follows a fixed policy file, not what I or any web page says. "
                "Risky things like deleting files or admin rights need your spoken yes.")

    def overview(self) -> str:
        return self.build() + " Ask me about my model, my voice or how I hear for the details."

    def answer(self, topic: str) -> str:
        fn = getattr(self, topic, None) if topic in TOPICS else None
        return fn() if callable(fn) else self.overview()

    # ---- all of it, for the chat model, when a self question wasn't matched by a pattern
    def block(self) -> str:
        return "\n".join(f"- {t}: {self.answer(t)}" for t in ("llm", "stt", "tts", "wake", "vision", "memory", "build"))


# A question about EV itself that the fast path did not take: worth putting the facts beside it. Loose on purpose; the block is
# short, and a false positive costs a few tokens, not a wrong action.
_ABOUT_SELF = re.compile(
    r"\b(?:you|your|you're|yourself|evie)\b.{0,60}\b(?:model|llm|brain|voice|engine|built|made|created|programmed|powered|running|"
    r"technolog\w*|tech stack|stack|hear|listen|speak|talk|work|ocr|wake ?word|whisper|piper|kokoro|ollama|python)\b", re.I)


def looks_about_self(text: str) -> bool:
    return bool(_ABOUT_SELF.search(text))
