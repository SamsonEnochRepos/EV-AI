"""The brain: persona + memory + history -> LLM -> gated tool loop -> streamed sentences."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import random
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ev.core.config import Config
from ev.core.log import get_logger
from ev.memory.store import MemoryStore
from ev.safety.gate import SafetyGate
from ev.safety.untrusted import InjectionDetector, wrap_untrusted
from ev.tools.base import Tool, ToolRegistry, ToolResult, validate_args

from .client import LLMError, OllamaClient, ToolCall
from .fastpath import FastMatch, match_fast, match_smalltalk
from .sentences import SentenceSplitter
from .spoken import spoken_text

log = get_logger("brain")

# When the user asks for depth, the short-reply cap is relaxed for that turn.
_WANTS_DETAIL = re.compile(
    r"\b(in detail|detailed|step by step|walk me through|at length|elaborate|tell me (?:a|the) story|"
    r"tell me (?:more|everything)|explain (?:it |that |this )?(?:fully|properly|more)|longer (?:answer|version))\b", re.I)

_TOOL_JSON =re.compile(r"\{\s*\"name\"\s*:\s*\"(\w+)\"\s*,\s*\"(?:arguments|parameters)\"\s*:\s*(\{.*?\})\s*\}", re.S)

# Tools with no on-screen or ordering side effects, so a batch of them may run at the same time. UI tools (open, close,
# click, type...) stay in the order the model asked for.
_PARALLEL_SAFE = frozenset({"get_weather", "web_search", "get_time_date", "remember_fact", "forget_fact", "take_note",
                            "set_timer", "set_reminder"})
# Said by the system, never the model, when a slow tool is still running: silence reads as "it didn't hear me".
_FILLERS = ("One sec.", "Let me check.", "Hang on a sec.", "Checking now.", "Looking into it.")


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
        self.direct_replies = bool(llm.get("direct_tool_replies", False))
        self.filler_after_s = float(llm.get("filler_after_s", 1.3))
        self.filler_tools = frozenset(llm.get("filler_tools", ["web_search", "get_weather", "read_screen", "screenshot_describe"]))
        self.rewarm_after_s = float(llm.get("rewarm_after_s", 90))
        self.first_chunk_min = int(cfg.get("tts", "first_chunk_min_chars", 12))
        self.clause_chars = int(cfg.get("tts", "first_clause_chars", 0))
        self.background_summary = True         # evals and tests that need the summary at once call update_summary() themselves
        self.rng = random.Random()  # noqa: S311  (picks a varied phrase; not security-related)
        self._summary_lock = threading.Lock()
        self._warm_lock = threading.Lock()
        self._schema_tokens: tuple[list[dict[str, Any]], int] | None = None

    # ---- prompt building
    def system_prompt(self) -> str:
        """Kept byte-identical from turn to turn (the date and part of the day change at most a few times a day), because
        Ollama only reuses its cached prompt up to the first token that differs. The exact clock is the get_time_date tool's job."""
        parts = [self.persona.strip()]
        facts = self.memory.facts(self.facts_limit)
        if facts:
            parts.append("What you know about the user:\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items()))
        summ = self.memory.get_kv("summary")
        if summ:
            parts.append("Earlier conversation summary: " + summ)
        now = self.now()
        part = ("morning" if 5 <= now.hour < 12 else "afternoon" if 12 <= now.hour < 17
                else "evening" if 17 <= now.hour < 22 else "night")
        parts.append(f"Today is {now:%A}, {now:%B} {now.day}, {now.year} ({part}).")
        return "\n\n".join(parts)

    def _trim_history(self) -> None:
        keep = self.history_turns * 2
        if len(self.history) > keep * 3:
            self.history = self.history[-keep * 3:]

    @staticmethod
    def _est_tokens(obj: Any) -> int:
        return int(len(json.dumps(obj, ensure_ascii=False)) / 3.5) + 1     # conservative for Qwen/Llama tokenizers

    def _schemas_tokens(self, schemas: list[dict[str, Any]]) -> int:
        """The tool list never changes between turns, so measure it once per registry change."""
        if self._schema_tokens is None or self._schema_tokens[0] is not schemas:
            self._schema_tokens = (schemas, self._est_tokens(schemas))
        return self._schema_tokens[1]

    def _messages(self, user_text: str) -> list[dict[str, Any]]:
        """System prompt + as much recent history as fits. Ollama silently drops the *start* of an over-long
        prompt (our system prompt and tool list), so we trim the oldest turns ourselves instead."""
        system = {"role": "system", "content": self.system_prompt()}
        user = {"role": "user", "content": user_text}
        fixed = self._est_tokens([system, user]) + self._schemas_tokens(self.registry.schemas())
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

    # ---- warm-up: keep Ollama's prompt cache holding the real system prompt + tool list
    def warm(self) -> None:
        """Load the model and process the real system prompt and tool list once, so the first reply of a session (or
        the first after the cache was lost) doesn't pay for ~2k tokens of prompt."""
        self.client.warm_up([{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": "hi"}],
                            self.registry.schemas())

    def rewarm_if_cold(self) -> bool:
        """Called when the user starts talking: if the model was unloaded (screen reading swaps it out) or has been idle,
        reload it in the background while they speak and the speech is transcribed."""
        if not self.client.is_cold(self.rewarm_after_s) or not self._warm_lock.acquire(blocking=False):
            return False

        def run() -> None:
            try:
                self.warm()
            except LLMError as e:
                log.warning("re-warm skipped: %s", e)
            finally:
                self._warm_lock.release()

        threading.Thread(target=run, name="rewarm", daemon=True).start()
        return True

    # ---- fast path
    def try_fast(self, text: str) -> TurnResult | None:
        return self._run_fast(text, match_fast(text))

    def _run_fast(self, text: str, fm: FastMatch | None, filler: Callable[[], None] | None = None) -> TurnResult | None:
        if not fm or fm.then_llm or fm.tool not in self.registry.names():
            return None
        t0 = time.perf_counter()
        res = self._execute(fm.tool, fm.args, filler)
        out = TurnResult(res["message"], [res], fast_path=True)
        out.timings["total"] = time.perf_counter() - t0
        self._remember_turn(text, out.text)
        return out

    def _smalltalk(self, text: str) -> TurnResult | None:
        reply = match_smalltalk(text, self.rng.choice)
        if not reply:
            return None
        self._remember_turn(text, reply)
        return TurnResult(reply, fast_path=True)

    # ---- main entry
    def respond(self, user_text: str, on_sentence: Callable[[str], None] | None = None,
                allow_fast: bool = True, should_stop: Callable[[], bool] | None = None) -> TurnResult:
        on_sentence = on_sentence or (lambda _s: None)
        should_stop = should_stop or (lambda: False)
        t_start = time.perf_counter()
        spoken: list[str] = []
        sentences = 0                                  # whole sentences spoken; an early clause chunk doesn't count
        said_filler = False

        def filler() -> None:                          # runs on a timer thread while a slow tool is still going
            nonlocal said_filler
            if said_filler or spoken or should_stop():
                return
            said_filler = True
            on_sentence(self.rng.choice(_FILLERS))

        forced = match_fast(user_text) if allow_fast else None     # matched once: it serves the fast path and the forced call
        if forced:
            fast = self._run_fast(user_text, forced, filler)
            if fast:
                on_sentence(fast.text)
                return fast
        elif allow_fast:
            chat = self._smalltalk(user_text)
            if chat:
                on_sentence(chat.text)
                return chat

        result = TurnResult("")
        messages = self._messages(user_text)
        schemas = self.registry.schemas()
        if forced and forced.then_llm and forced.tool in self.registry.names():
            # explicit "search for X" / "weather in X": call the tool ourselves, then let the model phrase the answer
            r = self._execute(forced.tool, forced.args, filler)
            result.tool_log.append(r)
            result.injection_flags += r.get("flags", [])
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"function": {"name": forced.tool, "arguments": r["args"]}}]})
            messages.append({"role": "tool", "tool_name": forced.tool, "content": r["model_view"]})
        first_token: float | None = None
        limit = self.max_sentences_detail if _WANTS_DETAIL.search(user_text) else self.max_sentences
        capped = False
        try:
            for step in range(self.max_steps + 1):
                splitter = SentenceSplitter(self.first_chunk_min, self.clause_chars)
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                use_tools = schemas if step < self.max_steps else None
                # closing() ends the HTTP stream the moment we stop reading, so Ollama stops generating too
                with contextlib.closing(self.client.chat_stream(messages, tools=use_tools)) as chunks:
                    for ch in chunks:
                        if should_stop():
                            break
                        if ch.text:
                            if first_token is None:
                                first_token = time.perf_counter() - t_start
                            text_parts.append(ch.text)
                            # Only stream speech if this turn is not turning into a tool call.
                            if not calls and not _looks_like_tool_json("".join(text_parts)):
                                for s, whole in splitter.feed_ex(ch.text):
                                    if sentences < limit and (s := spoken_text(s)) and not self._names_a_tool(s):
                                        spoken.append(s)
                                        sentences += whole
                                        on_sentence(s)
                                if sentences >= limit:
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
                entries = self._execute_batch(calls[:3], filler)
                for r in entries:
                    result.tool_log.append(r)
                    if r.get("flags"):
                        result.injection_flags += r["flags"]
                    messages.append({"role": "tool", "tool_name": r["tool"], "content": r["model_view"]})
                if self._speakable_as_is(entries, full):
                    # every result is already a finished sentence ("Opened Spotify."): say it and skip a whole
                    # model round trip, the biggest single delay in an action turn
                    for r in entries:
                        if s := spoken_text(r["message"]):
                            spoken.append(s)
                            on_sentence(s)
                    break
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

    def _speakable_as_is(self, entries: list[dict[str, Any]], model_text: str) -> bool:
        return (self.direct_replies and bool(entries) and not model_text.strip()
                and all(e["ok"] and e.get("direct") for e in entries))

    # ---- tool execution through the gate
    def _prepare(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], Tool | None]:
        """Validate and gate a call. The entry is final (and the tool None) if the call won't run."""
        entry: dict[str, Any] = {"tool": name, "args": args, "ok": False, "message": "", "model_view": ""}
        tool = self.registry.get(name)
        if tool is None:
            entry["message"] = f"I don't have a tool called {name}."
            entry["model_view"] = f"ERROR: no such tool '{name}'. Do not invent tools; answer in plain words."
            entry["hallucinated"] = True
            return entry, None
        clean, err = validate_args(tool, args)
        if clean is None:
            entry["message"] = "I couldn't work out the details for that."
            entry["model_view"] = f"ERROR: invalid arguments: {err}"
            return entry, None
        entry["args"] = clean
        gr = self.gate.check(name, clean)
        if not gr.allowed:
            entry["message"] = f"I can't do that: {gr.reason}."
            entry["model_view"] = f"DENIED by safety policy: {gr.reason}. Tell the user you did not do it."
            entry["denied"] = True
            return entry, None
        return entry, tool

    def _finish(self, entry: dict[str, Any], tool: Tool, res: ToolResult) -> None:
        entry["ok"] = res.ok
        entry["message"] = res.message
        entry["direct"] = tool.direct and not res.untrusted
        if res.untrusted:
            wrapped, flags = wrap_untrusted(res.message, res.source or tool.name, self.detector)
            entry["flags"] = flags
            entry["model_view"] = wrapped + (f"\n{res.hint}" if res.hint else "")
            entry["message"] = "I found some results."
        else:
            entry["model_view"] = ("OK: " if res.ok else "FAILED: ") + res.message

    @contextlib.contextmanager
    def _filler_guard(self, names: list[str], filler: Callable[[], None] | None) -> Iterator[None]:
        """If a slow tool is still running after filler_after_s, have `filler` say a short "one sec"."""
        if filler is None or self.filler_after_s <= 0 or not self.filler_tools.intersection(names):
            yield
            return
        lock = threading.Lock()
        finished = False

        def fire() -> None:
            with lock:
                if not finished:
                    filler()

        timer = threading.Timer(self.filler_after_s, fire)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            with lock:
                finished = True
            timer.cancel()

    def _execute(self, name: str, args: dict[str, Any], filler: Callable[[], None] | None = None) -> dict[str, Any]:
        entry, tool = self._prepare(name, args)
        if tool is None:
            return entry
        self.on_acting(True)
        try:
            with self._filler_guard([name], filler):
                res: ToolResult = self.registry.run(tool, entry["args"])
        finally:
            self.on_acting(False)
        self._finish(entry, tool, res)
        return entry

    def _execute_batch(self, calls: list[ToolCall], filler: Callable[[], None] | None = None) -> list[dict[str, Any]]:
        """Run one model turn's tool calls. A batch made only of side-effect-free tools runs concurrently
        (two searches take as long as one); anything else runs in order. Gating is always one call at a time."""
        if len(calls) < 2 or not all(c.name in _PARALLEL_SAFE for c in calls):
            return [self._execute(c.name, c.arguments, filler) for c in calls]
        prepared = [self._prepare(c.name, c.arguments) for c in calls]     # confirmations are spoken, so never concurrent
        jobs = [(t, e["args"]) for e, t in prepared if t is not None]
        results: Iterator[ToolResult] = iter(())
        if jobs:
            self.on_acting(True)
            try:
                with self._filler_guard([t.name for t, _ in jobs], filler):
                    results = iter(self.registry.run_many(jobs))
            finally:
                self.on_acting(False)
        for entry, tool in prepared:
            if tool is not None:
                self._finish(entry, tool, next(results))
        return [e for e, _ in prepared]

    # ---- memory
    def _remember_turn(self, user: str, reply: str, tool_log: list[dict[str, Any]] | None = None) -> None:
        self.history += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
        self._trim_history()
        self.turns += 1
        every = int(self.cfg.get("memory", "summary_every_turns", 8))
        if every and self.turns % every == 0:
            if self.background_summary:
                self._summarize_in_background()
            else:
                try:
                    self.update_summary()
                except LLMError:
                    log.warning("summary update skipped")

    def _summarize_in_background(self) -> None:
        """The summary is an extra model call; done inline it delays the end of the turn (and so the follow-up
        listening) by a second or two."""
        if not self._summary_lock.acquire(blocking=False):
            return

        def run() -> None:
            try:
                self.update_summary()
                self.warm()          # the summary request replaced Ollama's cached prompt; put the real one back
            except LLMError:
                log.warning("summary update skipped")
            except Exception:
                log.exception("summary update failed")
            finally:
                self._summary_lock.release()

        threading.Thread(target=run, name="summary", daemon=True).start()

    def update_summary(self) -> None:
        old = self.memory.get_kv("summary")
        recent = list(self.history[-self.history_turns * 2:])       # a copy: the next turn may append while this runs
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in recent if m.get("content"))
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
