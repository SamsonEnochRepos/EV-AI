"""Answers a coding agent's approval request: rules first, the user's voice for the grey area, the agent's own prompt as the fallback.

The agent (Claude Code, Codex) POSTs the request to `AgentListener`; `decide()` returns the hook response, or None for "no
decision", in which case the agent shows its normal prompt. None is what happens when EV is paused, can't ask, or isn't
answered in time, so a stopped EV never approves anything by accident.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ev.core.log import get_logger

from .rules import Decision, Rules, Verdict, describe, evaluate

log = get_logger("agents")

MAX_GRANT_MIN = 240
AGENT_NAMES = {"claude": "Claude Code", "codex": "Codex"}


class AgentSupervisor:
    def __init__(self, rules: Rules, audit_path: Path | None = None, ask: Callable[[str], bool | None] | None = None,
                 notify: Callable[[str], None] | None = None, publish: Callable[[dict[str, Any]], None] | None = None,
                 advisor: Callable[[str, dict[str, Any]], Decision | None] | None = None,
                 now: Callable[[], float] = time.time, ask_wait_s: float = 25.0):
        self.rules = rules
        self.audit_path = audit_path
        self.ask = ask                         # speaks a readback and returns True / False / None (no answer); None = can't ask
        self.notify = notify or (lambda _t: None)
        self.publish = publish or (lambda _e: None)
        self.advisor = advisor
        self.now = now
        self.ask_wait_s = ask_wait_s
        self.paused_until = 0.0
        self.last_cwd = ""
        self._grants: dict[tuple[str, str], float] = {}
        self._ask_lock = threading.Lock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=200)

    # ---- what the user can say
    def pause(self, minutes: float) -> None:
        self.paused_until = self.now() + minutes * 60 if minutes > 0 else 0.0

    @property
    def paused(self) -> bool:
        return self.now() < self.paused_until

    def grant(self, group: str, minutes: float) -> str:
        """Open one rule group for the project the agent was last working in. Returns the group's spoken label."""
        if group not in self.rules.groups:
            raise ValueError(f"unknown group {group!r}")
        key = (group, _key(self.last_cwd))
        self._grants[key] = self.now() + min(max(minutes, 1), MAX_GRANT_MIN) * 60
        return str(self.rules.groups[group]["label"])

    def revoke_all(self) -> None:
        self._grants.clear()

    def granted(self, cwd: str) -> frozenset[str]:
        t = self.now()
        self._grants = {k: v for k, v in self._grants.items() if v > t}
        return frozenset(g for (g, c), _ in self._grants.items() if c == _key(cwd))

    def status(self, window_s: float = 3600) -> str:
        t = self.now()
        recent = [e for e in self._recent if t - e["at"] <= window_s]
        if not recent:
            return "No agent requests in the last hour."
        n = {v: sum(1 for e in recent if e["verdict"] == v) for v in ("allow", "deny", "ask")}
        asked = sum(1 for e in recent if e.get("by") == "you")
        bit = f" You were asked {asked} time{'s' if asked != 1 else ''}." if asked else ""
        return f"In the last hour: {n['allow']} allowed, {n['deny']} blocked, {n['ask']} needed a decision.{bit}"

    # ---- the decision
    def decide(self, agent: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        tool = payload.get("tool_name")
        ti = payload.get("tool_input")
        cwd = str(payload.get("cwd") or "")
        if not isinstance(tool, str) or not isinstance(ti, dict):
            return None                                                    # not something we understand: leave it to the agent
        if cwd:
            self.last_cwd = cwd
        if self.paused:
            self._log(agent, tool, ti, cwd, Verdict.ASK, "approvals are paused", "paused")
            return None
        d = evaluate(self.rules, tool, ti, cwd, self.granted(cwd))
        if d.verdict is not Verdict.DENY and self.advisor is not None:
            d = self.advisor(tool, ti) or d                                # it can only make the answer stricter
        if d.verdict is Verdict.DENY:
            self._log(agent, tool, ti, cwd, d.verdict, d.reason, "rules")
            self.notify(f"Blocked {AGENT_NAMES.get(agent, agent)}: {d.reason}")
            return _response("deny", f"EV blocked this: {d.reason}. Don't retry it; tell the user what you needed instead.")
        if d.verdict is Verdict.ALLOW:
            self._log(agent, tool, ti, cwd, d.verdict, d.reason, "rules")
            return _response("allow")
        return self._ask_user(agent, tool, ti, cwd, d)

    def _ask_user(self, agent: str, tool: str, ti: dict[str, Any], cwd: str, d: Decision) -> dict[str, Any] | None:
        if self.ask is None:
            self._log(agent, tool, ti, cwd, Verdict.ASK, d.reason, "no way to ask")
            return None
        if not self._ask_lock.acquire(timeout=self.ask_wait_s):          # one question at a time; the rest wait their turn
            self._log(agent, tool, ti, cwd, Verdict.ASK, d.reason, "busy")
            return None
        try:
            who = AGENT_NAMES.get(agent, agent)
            where = Path(cwd).name if cwd else ""
            answer = self.ask(f"let {who} {describe(tool, ti)}" + (f" in {where}" if where else ""))
        finally:
            self._ask_lock.release()
        if answer is True:
            self._log(agent, tool, ti, cwd, Verdict.ASK, d.reason, "you", approved=True)
            return _response("allow")
        if answer is False:
            self._log(agent, tool, ti, cwd, Verdict.ASK, d.reason, "you", approved=False)
            return _response("deny", "The user said no. Don't retry it; ask them what they'd like instead.")
        self._log(agent, tool, ti, cwd, Verdict.ASK, d.reason, "no answer")
        return None                                                        # nobody answered: the agent's own prompt takes over

    # ---- the record
    def _log(self, agent: str, tool: str, ti: dict[str, Any], cwd: str, verdict: Verdict, reason: str, by: str,
             approved: bool | None = None) -> None:
        entry = {"at": self.now(), "agent": agent, "tool": tool, "what": describe(tool, ti)[:200], "cwd": cwd[:200],
                 "verdict": verdict.value if approved is None else ("allow" if approved else "deny"),
                 "reason": reason, "by": by}
        if approved is not None:
            entry["verdict_before"] = "ask"
        self._recent.append(entry)
        self.publish(entry)
        if self.audit_path is not None:
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as e:
                log.warning("could not write the agent audit log: %s", e)


def _key(cwd: str) -> str:
    return cwd.replace("\\", "/").rstrip("/").lower()


def _response(behavior: str, message: str = "") -> dict[str, Any]:
    decision: dict[str, Any] = {"behavior": behavior}
    if behavior == "deny":
        decision.update({"message": message, "interrupt": False})
    return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": decision}}
