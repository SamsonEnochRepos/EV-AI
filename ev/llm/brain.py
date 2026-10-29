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

# When the user asks for depth, the short-reply cap is relaxed for that turn.
_WANTS_DETAIL = re.compile(
    r"\b(in detail|detailed|step by step|walk me through|at length|elaborate|tell me (?:a|the) story|"
    r"tell me (?:more|everything)|explain (?:it |that |this )?(?:fully|properly|more)|longer (?:answer|version))\b", re.I)

_TOOL_JSON =re.compile(r"\{\s*\"name\"\s*:\s*\"(\w+)\"\s*,\s*\"(?:arguments|parameters)\"\s*:\s*(\{.*?\})\s*\}", re.S)


@dataclass
class TurnResult:
    text: str
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    fast_path: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    injection_flags: list[str] = field(default_factory=list)
    error: str = ""
    interrupted: bool = False   # the user talked over the reply; text is only what was actually spoken


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
        llm = cfg["llm"]
        self.ctx_budget = int(llm["num_ctx"]) - int(llm["num_predict"]) - 128   # leave room for the reply + template
        self.max_sentences = int(llm.get("reply_max_sentences", 2))
        self.max_sentences_detail = int(llm.get("reply_max_sentences_detail", 6))

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

    @staticmethod
    def _est_tokens(obj: Any) -> int:
        return int(len(json.dumps(obj, ensure_ascii=False)) / 3.5) + 1     # conservative for Qwen/Llama tokenizers

    def _messages(self, user_text: str) -> list[dict[str, Any]]:
        """System prompt + as much recent history as fits. Ollama silently drops the *start* of an over-long
        prompt (our system prompt and tool list), so we trim the oldest turns ourselves instead."""
        system = {"role": "system", "content": self.system_prompt()}
        user = {"role": "user", "content": user_text}
        fixed = self._est_tokens([system, user]) + self._est_tokens(self.registry.schemas())
        hist = self.history[-self.history_turns * 2:]
        while hist and fixed + self._est_tokens(hist) > self.ctx_budget:
            hist = hist[2:]                                                 # drop the oldest user/assistant pair
        if fixed > self.ctx_budget:
            log.warning("prompt (~%d tokens) exceeds the context budget (%d); raise llm.num_ctx or shorten the persona",
                        fixed, self.ctx_budget)
        return [system, *hist, user]

    def _names_a_tool(self, sentence: str) -> bool:
        """Small models narrate "I can use the web_search tool..."; internal tool names are never spoken."""
        return any(n in sentence for n in self.registry.names() if "_" in n)

    # ---- fast path
    def try_fast(self, text: str) -> TurnResult | None:
        fm = match_fast(text)
        if not fm or fm.then_llm or fm.tool not in self.registry.names():
            return None
        t0 = time.perf_counter()
        res = self._execute(fm.tool, fm.args)
        out = TurnResult(res["message"], [res], fast_path=True)
        out.timings["total"] = time.perf_counter() - t0
        self._remember_turn(text, out.text)
        return out

    # ---- main entry
    def respond(self, user_text: str, on_sentence: Callable[[str], None] | None = None,
                allow_fast: bool = True, should_stop: Callable[[], bool] | None = None) -> TurnResult:
        on_sentence = on_sentence or (lambda _s: None)
        should_stop = should_stop or (lambda: False)
        t_start = time.perf_counter()
        if allow_fast:
            fast = self.try_fast(user_text)
            if fast:
                on_sentence(fast.text)
                return fast

        result = TurnResult("")
        messages = self._messages(user_text)
        schemas = self.registry.schemas()
        forced = match_fast(user_text) if allow_fast else None
        if forced and forced.then_llm and forced.tool in self.registry.names():
            # explicit "search for X" / "weather in X": call the tool ourselves, then let the model phrase the answer
            r = self._execute(forced.tool, forced.args)
            result.tool_log.append(r)
            result.injection_flags += r.get("flags", [])
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"function": {"name": forced.tool, "arguments": r["args"]}}]})
            messages.append({"role": "tool", "tool_name": forced.tool, "content": r["model_view"]})
        spoken: list[str] = []
        first_token: float | None = None
        limit = self.max_sentences_detail if _WANTS_DETAIL.search(user_text) else self.max_sentences
        capped = False
        try:
            for step in range(self.max_steps + 1):
                splitter = SentenceSplitter(int(self.cfg.get("tts", "first_chunk_min_chars", 12)))
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                use_tools = schemas if step < self.max_steps else None
                for ch in self.client.chat_stream(messages, tools=use_tools):
                    if should_stop():
                        break
                    if ch.text:
                        if first_token is None:
                            first_token = time.perf_counter() - t_start
                        text_parts.append(ch.text)
                        # Only stream speech if this turn is not turning into a tool call.
                        if not calls and not _looks_like_tool_json("".join(text_parts)):
                            for s in splitter.feed(ch.text):
                                if len(spoken) < limit and (s := spoken_text(s)) and not self._names_a_tool(s):
                                    spoken.append(s)
                                    on_sentence(s)
                            if len(spoken) >= limit:
                                capped = True          # enough said; stop generating the rest
                                break
                    calls += ch.tool_calls
                if should_stop():
                    result.interrupted = True
                    break
                full = "".join(text_parts)
                if not calls:
                    calls = _extract_json_calls(full)
                    if calls:
                        full = ""
                if not calls:
                    for s in ([] if capped else splitter.flush()):
                        if (s := spoken_text(s)) and not self._names_a_tool(s):
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
            if type(e).__name__ == "OllamaDown":
                msg = "I can't reach my brain right now. Is Ollama running?"
            elif "out of memory" in str(e).lower():
                msg = "I'm out of memory to think with. Closing a game or a heavy app should help."
            else:
                msg = "Something went wrong while thinking. Let me know if you want to try again."
            spoken = [msg]
            on_sentence(msg)

        text = " ".join(spoken).strip()
        if result.interrupted:
            pass
        elif not text and result.tool_log:
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
            entry["model_view"] = wrapped + (f"\n{res.hint}" if res.hint else "")
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
