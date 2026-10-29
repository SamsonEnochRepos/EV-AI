"""The brain: persona + memory + history -> LLM -> gated tool loop -> streamed sentences."""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ev.core.config import Config
from ev.core.log import get_logger
from ev.memory.store import MemoryStore
from ev.safety.gate import SafetyGate
from ev.safety.untrusted import InjectionDetector, wrap_untrusted
from ev.tools.base import ToolRegistry, ToolResult, validate_args

from .client import LLMError, OllamaClient, ToolCall
from .fastpath import match_fast
from .sentences import SentenceSplitter
from .spoken import spoken_text

log = get_logger("brain")

_TOOL_JSON = re.compile(r"\{\s*\"name\"\s*:\s*\"(\w+)\"\s*,\s*\"(?:arguments|parameters)\"\s*:\s*(\{.*?\})\s*\}", re.S)


@dataclass
class TurnResult:
    text: str
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    fast_path: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    injection_flags: list[str] = field(default_factory=list)
    error: str = ""


class Brain:
    def __init__(self, cfg: Config, client: OllamaClient, memory: MemoryStore,
                 registry: ToolRegistry, gate: SafetyGate,
                 persona: str | None = None,
                 on_acting: Callable[[bool], None] | None = None,
                 now: Callable[[], dt.datetime] = dt.datetime.now):
        self.cfg = cfg
        self.client = client
        self.memory = memory
        self.registry = registry
        self.gate = gate
        self.persona = persona if persona is not None else cfg.path("general", "persona_file").read_text(encoding="utf-8")
        self.history: list[dict[str, Any]] = []
        self.max_steps = int(cfg.get("general", "max_tool_steps", 5))
        self.history_turns = int(cfg.get("llm", "history_turns", 6))
        self.facts_limit = int(cfg.get("memory", "max_facts_in_prompt", 12))
        self.on_acting = on_acting or (lambda _b: None)
        self.detector = InjectionDetector(gate.policy.d.get("injection", {}).get("phrases"))
        self.now = now
        self.turns = 0

    # ---- prompt building
    def system_prompt(self) -> str:
        parts = [self.persona.strip()]
        facts = self.memory.facts(self.facts_limit)
        if facts:
            parts.append("What you know about the user:\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items()))
        summ = self.memory.get_kv("summary")
        if summ:
            parts.append("Earlier conversation summary: " + summ)
        parts.append(self.now().strftime("Current time: %A %B %d %Y, %I:%M %p."))
        return "\n\n".join(parts)

    def _trim_history(self) -> None:
        keep = self.history_turns * 2
        if len(self.history) > keep * 3:
            self.history = self.history[-keep * 3:]

    def _messages(self, user_text: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt()}, *self.history,
                {"role": "user", "content": user_text}]

    # ---- fast path
    def try_fast(self, text: str) -> TurnResult | None:
        fm = match_fast(text)
        if not fm or fm.tool not in self.registry.names():
            return None
        t0 = time.perf_counter()
        res = self._execute(fm.tool, fm.args)
        out = TurnResult(res["message"], [res], fast_path=True)
        out.timings["total"] = time.perf_counter() - t0
        self._remember_turn(text, out.text)
        return out

    # ---- main entry
    def respond(self, user_text: str, on_sentence: Callable[[str], None] | None = None,
                allow_fast: bool = True) -> TurnResult:
        on_sentence = on_sentence or (lambda _s: None)
        t_start = time.perf_counter()
        if allow_fast:
            fast = self.try_fast(user_text)
            if fast:
                on_sentence(fast.text)
                return fast

        result = TurnResult("")
        messages = self._messages(user_text)
        schemas = self.registry.schemas()
        spoken: list[str] = []
        first_token: float | None = None
        try:
            for step in range(self.max_steps + 1):
                splitter = SentenceSplitter(int(self.cfg.get("tts", "first_chunk_min_chars", 12)))
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                use_tools = schemas if step < self.max_steps else None
                for ch in self.client.chat_stream(messages, tools=use_tools):
                    if ch.text:
                        if first_token is None:
                            first_token = time.perf_counter() - t_start
                        text_parts.append(ch.text)
                        # Only stream speech if this turn is not turning into a tool call.
                        if not calls and not _looks_like_tool_json("".join(text_parts)):
                            for s in splitter.feed(ch.text):
                                if (s := spoken_text(s)):
                                    spoken.append(s)
                                    on_sentence(s)
                    calls += ch.tool_calls
                full = "".join(text_parts)
                if not calls:
                    calls = _extract_json_calls(full)
                    if calls:
                        full = ""
                if not calls:
                    for s in splitter.flush():
                        if (s := spoken_text(s)):
                            spoken.append(s)
                            on_sentence(s)
                    break
                messages.append({"role": "assistant", "content": full,
                                 "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
                for c in calls[:3]:
                    r = self._execute(c.name, c.arguments)
                    result.tool_log.append(r)
                    if r.get("flags"):
                        result.injection_flags += r["flags"]
                    messages.append({"role": "tool", "tool_name": c.name, "content": r["model_view"]})
        except LLMError as e:
            result.error = type(e).__name__
            log.error("LLM failure: %s", e)
            msg = "I can't reach my brain right now. Is Ollama running?" if type(e).__name__ == "OllamaDown" else "Something went wrong while thinking. Let me know if you want to try again."
            spoken = [msg]
            on_sentence(msg)

        text = " ".join(spoken).strip()
        if not text and result.tool_log:
            text = spoken_text(result.tool_log[-1]["message"])
            on_sentence(text)
        elif not text and not result.error:
            text = "Sorry, I blanked on that. Could you say it again?"
            on_sentence(text)
        result.text = text
        result.timings["first_token"] = first_token or 0.0
        result.timings["total"] = time.perf_counter() - t_start
        if not result.error:
            self._remember_turn(user_text, text, result.tool_log)
        return result

    # ---- tool execution through the gate
    def _execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {"tool": name, "args": args, "ok": False, "message": "", "model_view": ""}
        tool = self.registry.get(name)
        if tool is None:
            entry["message"] = f"I don't have a tool called {name}."
            entry["model_view"] = f"ERROR: no such tool '{name}'. Do not invent tools; answer in plain words."
            entry["hallucinated"] = True
            return entry
        clean, err = validate_args(tool, args)
        if clean is None:
            entry["message"] = "I couldn't work out the details for that."
            entry["model_view"] = f"ERROR: invalid arguments: {err}"
            return entry
        entry["args"] = clean
        gr = self.gate.check(name, clean)
        if not gr.allowed:
            entry["message"] = f"I can't do that: {gr.reason}."
            entry["model_view"] = f"DENIED by safety policy: {gr.reason}. Tell the user you did not do it."
            entry["denied"] = True
            return entry
        self.on_acting(True)
        try:
            res: ToolResult = self.registry.run(tool, clean)
        finally:
            self.on_acting(False)
        entry["ok"] = res.ok
        entry["message"] = res.message
        if res.untrusted:
            wrapped, flags = wrap_untrusted(res.message, res.source or name, self.detector)
            entry["flags"] = flags
            entry["model_view"] = wrapped
            entry["message"] = "I found some results."
        else:
            entry["model_view"] = ("OK: " if res.ok else "FAILED: ") + res.message
        return entry

    # ---- memory
    def _remember_turn(self, user: str, reply: str, tool_log: list[dict[str, Any]] | None = None) -> None:
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
        self._trim_history()
        self.turns += 1
        every = int(self.cfg.get("memory", "summary_every_turns", 8))
        if every and self.turns % every == 0:
            try:
                self.update_summary()
            except LLMError:
                log.warning("summary update skipped")

    def update_summary(self) -> None:
        old = self.memory.get_kv("summary")
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-self.history_turns * 2:] if m.get("content"))
        prompt = [{"role": "system", "content": "Update the running summary of a conversation between a user and their assistant EV. Keep every durable fact about the user (names, preferences, plans). Max 80 words. Output only the summary."},
                  {"role": "user", "content": f"Current summary: {old or '(none)'}\n\nRecent conversation:\n{convo}"}]
        r = self.client.chat(prompt, options={"num_predict": 140, "temperature": 0.2})
        if r.text.strip():
            self.memory.set_kv("summary", r.text.strip()[:700])


def _looks_like_tool_json(s: str) -> bool:
    return s.lstrip().startswith("{") or s.lstrip().startswith("```")


def _extract_json_calls(text: str) -> list[ToolCall]:
    out = []
    for m in _TOOL_JSON.finditer(text):
        try:
            out.append(ToolCall(m.group(1), json.loads(m.group(2))))
        except json.JSONDecodeError:
            continue
    return out
