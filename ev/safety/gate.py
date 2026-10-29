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
        self.killed = threading.Event()
        self.audit: list[dict[str, Any]] = []

    def kill(self) -> None:
        self.killed.set()

    def reset_kill(self) -> None:
        self.killed.clear()

    def check(self, tool: str, args: dict[str, Any] | None) -> GateResult:
        if self.killed.is_set():
            return self._rec(tool, args, GateResult(False, "kill switch active"))
        d: Decision = self.policy.evaluate(tool, args)
        if d.verdict is Verdict.BLOCK:
            return self._rec(tool, args, GateResult(False, d.reason))
        if d.verdict is Verdict.ALLOW:
            return self._rec(tool, args, GateResult(True, "ok"))
        # CONFIRM: separate step with exact readback; fail closed.
        if self.ask is None:
            return self._rec(tool, args, GateResult(False, "confirmation unavailable"))
        readback = self.describe(tool, args or {})
        answer = self.ask(readback)
        if answer is True and not self.killed.is_set():
            return self._rec(tool, args, GateResult(True, "confirmed", True))
        return self._rec(tool, args, GateResult(False, "not confirmed" if answer is False else "confirmation timed out"))

    def _rec(self, tool: str, args: Any, r: GateResult) -> GateResult:
        self.audit.append({"tool": tool, "args": args, "allowed": r.allowed, "reason": r.reason})
        del self.audit[:-200]
        log.info("gate %s -> %s (%s)", tool, "ALLOW" if r.allowed else "DENY", r.reason)
        return r


def default_describe(tool: str, args: dict[str, Any]) -> str:
    bits = ", ".join(f"{k} {v}" for k, v in args.items())
    return f"{tool.replace('_', ' ')}" + (f" with {bits}" if bits else "")
