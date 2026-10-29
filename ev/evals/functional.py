"""Functional evals (12A): intent/tool accuracy, persona quality, memory."""
from __future__ import annotations

import re
import statistics
import time
from typing import Any

from ev.core.config import ROOT, Config, load_config
from ev.llm.brain import Brain
from ev.llm.client import OllamaClient
from ev.llm.fastpath import match_fast
from ev.memory.store import MemoryStore
from ev.safety.gate import SafetyGate
from ev.safety.policy import Policy
from ev.tools.base import Tool, ToolRegistry, ToolResult
from ev.tools.builtin import ToolContext, build_tools
from ev.tools.scheduler import Scheduler

from .common import Recorder, pct
from .datasets import INTENT, PERSONA

ALL_TOOL_NAMES = ["get_time_date", "open_app", "set_timer", "set_reminder", "media_control", "set_volume",
                  "web_search", "remember_fact", "forget_fact", "take_note", "read_screen", "click_element",
                  "type_text", "press_keys", "scroll", "focus_window", "copy_selection", "paste", "do_on_screen"]


def make_recording_brain(cfg: Config, model: str | None = None, ask=None) -> tuple[Brain, list[dict[str, Any]], MemoryStore]:  # noqa: ANN001
    """Brain with stub tools that record calls instead of touching the machine."""
    mem = MemoryStore(":memory:")
    pol = Policy.load(ROOT / "config" / "policy.toml")
    real = build_tools(ToolContext(cfg, mem, Scheduler(mem, lambda _m: None), pol))
    reg = ToolRegistry()
    calls: list[dict[str, Any]] = []
    for name in real.names():
        t = real.get(name)
        assert t
        if name in ("remember_fact", "forget_fact", "take_note", "get_time_date", "set_reminder", "set_timer"):
            reg.register(t)
            continue
        def stub(_n: str = name, **kw: Any) -> ToolResult:
            return ToolResult(True, f"{_n} ok")
        reg.register(Tool(name, t.description, t.parameters, stub, t.required))
    # screen/control tool schemas without real side effects
    extra: dict[str, tuple[dict[str, Any], tuple[str, ...], str]] = {
        "read_screen": ({"method": {"type": "string", "enum": ["auto", "uia", "ocr"]}}, (), "Read the text currently visible on the user's screen."),
        "click_element": ({"name": {"type": "string"}}, ("name",), "Click a button, link or control by its visible name."),
        "type_text": ({"text": {"type": "string"}}, ("text",), "Type text into the focused field."),
        "press_keys": ({"combo": {"type": "string"}}, ("combo",), "Press a key or shortcut such as ctrl+s."),
        "scroll": ({"direction": {"type": "string", "enum": ["up", "down"]}, "amount": {"type": "integer"}}, ("direction",), "Scroll the active window."),
        "focus_window": ({"title": {"type": "string"}}, ("title",), "Bring a window to the front by part of its title."),
        "copy_selection": ({}, (), "Copy the current selection."),
        "paste": ({}, (), "Paste the clipboard."),
        "do_on_screen": ({"goal": {"type": "string"}}, ("goal",), "Carry out a multi-step task on screen, like opening an app and typing in it."),
    }
    for n, (params, req, desc) in extra.items():
        reg.register(Tool(n, desc, params, lambda _n=n, **kw: ToolResult(True, f"{_n} ok"), req))

    wrapped = ToolRegistry()
    for n in reg.names():
        t = reg.get(n)
        assert t

        def rec_fn(_t: Tool = t, **kw: Any) -> ToolResult:
            calls.append({"tool": _t.name, "args": kw})
            return _t.func(**kw)
        wrapped.register(Tool(t.name, t.description, t.parameters, rec_fn, t.required))
    gate = SafetyGate(pol, ask=ask or (lambda _r: False))
    client = OllamaClient(cfg)
    if model:
        client.model = model
    return Brain(cfg, client, mem, wrapped, gate), calls, mem


def _args_match(expect: dict[str, Any], got: dict[str, Any]) -> bool:
    for k, v in expect.items():
        g = got.get(k)
        if g is None:
            return False
        if isinstance(v, (int, float)):
            try:
                if abs(float(g) - float(v)) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif str(v).lower() not in str(g).lower():
            return False
    return True


def intent_accuracy(model: str | None = None, rec: Recorder | None = None, use_fastpath: bool = True) -> Recorder:
    cfg = load_config()
    rec = rec or Recorder("intent")
    brain, calls, mem = make_recording_brain(cfg, model)
    totals = {"tool_ok": 0, "arg_ok": 0, "arg_n": 0, "halluc": 0, "n": 0, "fast_hit": 0, "fast_n": 0}
    lat: list[float] = []
    for it in INTENT:
        calls.clear()
        brain.history.clear()
        t0 = time.perf_counter()
        fm = match_fast(it["u"]) if use_fastpath else None
        if it["cat"] == "fast":
            totals["fast_n"] += 1
            totals["fast_hit"] += 1 if fm else 0
        first: dict[str, Any] | None
        if fm:
            first = {"tool": fm.tool, "args": fm.args}
        else:
            res = brain.respond(it["u"], allow_fast=False)
            first = calls[0] if calls else None
            halluc = any(t.get("hallucinated") for t in res.tool_log)
            totals["halluc"] += 1 if halluc else 0
        lat.append(time.perf_counter() - t0)
        accept = it.get("accept", [it["tool"]])
        got_tool = first["tool"] if first else None
        ok = got_tool in accept or (got_tool == it["tool"])
        totals["n"] += 1
        totals["tool_ok"] += ok
        if first and it["tool"] and got_tool == it["tool"] and it["args"]:
            totals["arg_n"] += 1
            totals["arg_ok"] += _args_match(it["args"], first["args"])
        rec.add(f"{it['cat']}:{it['u'][:48]}", ok, "info", f"want={it['tool']} got={got_tool} args={first['args'] if first else None}")
    mem.close()
    rec.summary.update({
        "model": model or cfg["llm"]["model"],
        "correct_tool_rate": totals["tool_ok"] / totals["n"],
        "correct_arg_rate": totals["arg_ok"] / totals["arg_n"] if totals["arg_n"] else None,
        "hallucinated_tool_rate": totals["halluc"] / totals["n"],
        "fast_path_coverage": totals["fast_hit"] / totals["fast_n"] if totals["fast_n"] else None,
        "median_latency_s": statistics.median(lat),
        "n": totals["n"],
    })
    return rec


# ---------------------------------------------------------------- persona
_BANNED = [(re.compile(r"[*#`]"), "markdown chars"), (re.compile(r"^\s*(?:[-•]|\d+[.)])\s", re.M), "list"),
           (re.compile("[\U0001F300-\U0001FAFF☀-➿]"), "emoji"), (re.compile(r"https?://|www\."), "url"),
           (re.compile(r"\d+%"), "percent symbol")]
_CLAIM = re.compile(r"\b(i('ve| have)? (sent|saved|deleted|emailed|transferred)|done[.!]|all set)\b", re.I)
_PERSON = re.compile(r"\b(i am|i'm) (scarlett|peter|a human|a real person)\b", re.I)


def persona_checks(prompt: str, kind: str, reply: str, tools_called: list[str]) -> dict[str, bool]:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", reply.strip()) if s]
    long_ok = kind in ("concise_long_ok",)
    out = {
        "nonempty": bool(reply.strip()),
        "no_banned_formatting": not any(rx.search(reply) for rx, _ in _BANNED),
        "concise": len(sents) <= (12 if long_ok else 4) and len(reply) <= (1200 if long_ok else 380),
        "no_false_action_claim": not (kind == "no_false_claim" and _CLAIM.search(reply) and not tools_called),
        "not_impersonating": not _PERSON.search(reply),
        "no_all_caps": not (kind == "robust" and sum(c.isupper() for c in reply) > 0.5 * max(1, sum(c.isalpha() for c in reply))),
    }
    return out


JUDGE_RUBRIC = (
    "You grade a voice assistant named EV. Persona: warm, quick-witted, playful, concise, caring, honest about "
    "limits, speaks like a person. Score the reply 1-5 for each: character, naturalness_for_speech, honesty. "
    'Reply ONLY with JSON like {"character":4,"naturalness_for_speech":5,"honesty":5}.'
)


def persona_eval(model: str | None = None, judge: bool = True) -> Recorder:
    import json

    cfg = load_config()
    rec = Recorder("persona")
    brain, calls, mem = make_recording_brain(cfg, model)
    jclient = OllamaClient(cfg)
    scores: list[float] = []
    for it in PERSONA:
        calls.clear()
        brain.history.clear()
        res = brain.respond(it["p"], allow_fast=False)
        checks = persona_checks(it["p"], it["kind"], res.text, [c["tool"] for c in calls])
        ok = all(checks.values())
        detail = ",".join(k for k, v in checks.items() if not v)
        if judge and res.text:
            try:
                j = jclient.chat([{"role": "system", "content": JUDGE_RUBRIC},
                                  {"role": "user", "content": f"User said: {it['p']}\nEV replied: {res.text}"}],
                                 options={"temperature": 0.0, "num_predict": 60})
                m = re.search(r"\{.*?\}", j.text, re.S)
                if m:
                    d = json.loads(m.group(0))
                    s = statistics.mean(float(v) for v in d.values())
                    scores.append(s)
                    detail += f" judge={s:.1f}"
            except Exception:      # noqa: BLE001
                pass
        rec.add(f"{it['kind']}:{it['p'][:40]}", ok, "info", f"{detail} | {res.text[:70]!r}")
    mem.close()
    rec.summary.update({"model": model or cfg["llm"]["model"], "deterministic_pass_rate": rec.passed / len(rec.checks),
                        "judge_mean_1to5": statistics.mean(scores) if scores else None, "judged": len(scores)})
    return rec


# ---------------------------------------------------------------- memory
def memory_eval() -> Recorder:
    cfg = load_config()
    rec = Recorder("memory")
    m = MemoryStore(":memory:")
    rec.add("store_and_recall", (m.remember("favorite color", "green"), m.recall("Favorite_Color"))[1] == "green")
    m.remember("favorite color", "blue")
    rec.add("update_overwrites", m.recall("favorite color") == "blue")
    rec.add("forget_removes", m.forget("favorite color") and m.recall("favorite color") is None)
    rec.add("forget_missing_false", not m.forget("never_saved"))
    m.close()
    brain, calls, mem = make_recording_brain(cfg)
    try:
        brain.respond("remember that my dog's name is Biscuit", allow_fast=False)
        stored = any("biscuit" in v.lower() for v in mem.facts().values())
        rec.add("llm_stores_fact_from_speech", stored, "info", str(mem.facts()))
        brain.history.clear()
        r = brain.respond("what is my dog's name?", allow_fast=False)
        rec.add("llm_recalls_fact_in_new_session", "biscuit" in r.text.lower(), "info", r.text[:80])
        brain.respond("forget my dog's name", allow_fast=False)
        rec.add("llm_forgets_on_request", not any("biscuit" in v.lower() for v in mem.facts().values()), "info", str(mem.facts()))
        # rolling summary keeps key facts
        brain.history.clear()
        mem.facts()
        for t in ["Hi, I'm Priya and I live in Lisbon.", "I work as a nurse on night shifts.", "I'm training for a half marathon in March."]:
            brain.respond(t, allow_fast=False)
        brain.update_summary()
        summ = mem.get_kv("summary").lower()
        keep = sum(k in summ for k in ("priya", "lisbon", "nurse", "marathon"))
        rec.add("summary_keeps_key_facts", keep >= 3, "info", f"{keep}/4 kept: {summ[:100]}")
    finally:
        mem.close()
    rec.summary["pass_rate"] = rec.passed / len(rec.checks)
    return rec


def summarize_rates(rec: Recorder) -> str:
    s = rec.summary
    return ", ".join(f"{k}={pct(int(v*1000), 1000) if isinstance(v, float) and v <= 1 else v}" for k, v in s.items())
