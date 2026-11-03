"""Safety gate: the only path from a model-proposed tool call to execution.

Confirmation is a separate, out-of-band step: the exact action is read back by the system
(not the model) and only a real user yes (voice transcript matched here, or hotkey) approves it.
The model never sees a way to answer the prompt itself.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ev.core.log import get_logger

from .policy import Decision, Policy, Verdict

log = get_logger("gate")

YES = re.compile(r"^\s*(yes|yeah|yep|yup|sure|go ahead|do it|confirm|confirmed|okay do it)\b[\s.!]*$", re.I)
NO = re.compile(r"^\s*(no|nope|nah|cancel|stop|don'?t|never mind|abort)\b", re.I)


def parse_yes_no(text: str) -> bool | None:
    if NO.search(text):
        return False
    if YES.search(text):
        return True
    return None


@dataclass
class GateResult:
    allowed: bool
    reason: str
    confirmed: bool = False


# ask(readback) -> True / False / None(timeout). Supplied by the voice layer; hotkeys can also set it.
AskFn = Callable[[str], "bool | None"]


class SafetyGate:
    def __init__(self, policy: Policy, ask: AskFn | None = None, describe: Callable[[str, dict[str, Any]], str] | None = None):
        self.policy = policy
        self.ask = ask
        self.describe = describe or default_describe
        self.describers: dict[str, Callable[[dict[str, Any]], str]] = {}   # a tool whose readback needs more than its arguments
        self.killed = threading.Event()
        self.audit: list[dict[str, Any]] = []

    def kill(self) -> None:
        self.killed.set()

    def reset_kill(self) -> None:
        self.killed.clear()

    def check(self, tool: str, args: dict[str, Any] | None, escalate: str | None = None, ask_allowed: bool = True) -> GateResult:
        """`escalate` is a reason, computed by the system (ev/safety/provenance.py), to ask the user first. It can only turn
        ALLOW into CONFIRM: a BLOCK stays a BLOCK, a CONFIRM stays a CONFIRM, and nothing here can approve anything.

        `ask_allowed=False` is for things EV starts on its own (a proactive rule): anything that would need a spoken yes is
        refused instead of asking, because nobody asked EV to speak up. It can only make the answer stricter."""
        if self.killed.is_set():
            return self._rec(tool, args, GateResult(False, "kill switch active"))
        d: Decision = self.policy.evaluate(tool, args)
        if d.verdict is Verdict.BLOCK:
            return self._rec(tool, args, GateResult(False, d.reason))
        if d.verdict is Verdict.ALLOW and not escalate:
            return self._rec(tool, args, GateResult(True, "ok"))
        # CONFIRM: separate step with exact readback; fail closed.
        if not ask_allowed:
            return self._rec(tool, args, GateResult(False, "that needs your approval, and I don't ask for it on my own"), escalate)
        if self.ask is None:
            return self._rec(tool, args, GateResult(False, "confirmation unavailable"), escalate)
        readback = self.describer(tool)(args or {})
        if escalate:
            readback += f", though {escalate}"
        answer = self.ask(readback)
        if answer is True and not self.killed.is_set():
            return self._rec(tool, args, GateResult(True, "confirmed", True), escalate)
        return self._rec(tool, args, GateResult(False, "not confirmed" if answer is False else "confirmation timed out"),
                         escalate)

    def describer(self, tool: str) -> Callable[[dict[str, Any]], str]:
        own = self.describers.get(tool)
        return own if own else (lambda a: self.describe(tool, a))

    def _rec(self, tool: str, args: Any, r: GateResult, escalated: str | None = None) -> GateResult:
        self.audit.append({"tool": tool, "args": args, "allowed": r.allowed, "reason": r.reason,
                           **({"escalated": escalated} if escalated else {})})
        del self.audit[:-200]
        log.info("gate %s -> %s (%s)", tool, "ALLOW" if r.allowed else "DENY", r.reason)
        return r


def default_describe(tool: str, args: dict[str, Any]) -> str:
    if tool == "close_app" and args.get("force"):
        return f"force close {args.get('name', 'that app')}. Anything unsaved in it will be lost"
    if tool == "open_app" and args.get("admin"):
        return f"open {args.get('name', 'that app')} as administrator"
    if tool == "open_app":
        return f"open {args.get('name', 'that app')}"
    if tool == "open_website":
        return "open " + (args.get("url") or f"{args.get('site', 'a website')} searching for {args.get('query')}")
    if tool == "run_skill":
        return f"run the skill {args.get('name', '')}"
    if tool == "remember_fact":
        about = f"{args['about']}'s " if args.get("about") else ""
        return f"remember that {about}{args.get('key', 'that')} is {args.get('value', '')}"
    if tool == "forget_fact":
        about = f"{args['about']}'s " if args.get("about") else ""
        return f"forget {about}{args.get('key', 'that')}"
    bits = ", ".join(f"{k} {v}" for k, v in args.items())
    return f"{tool.replace('_', ' ')}" + (f" with {bits}" if bits else "")
