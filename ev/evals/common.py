"""Shared eval helpers: result recording and report fragments."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ev.core.config import ROOT

RESULTS = ROOT / "evals" / "results"


@dataclass
class Check:
    suite: str
    name: str
    passed: bool
    severity: str = "medium"       # for security items
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class Recorder:
    def __init__(self, suite: str):
        self.suite = suite
        self.checks: list[Check] = []
        self.meta: dict[str, Any] = {"suite": suite, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                     "python": platform.python_version()}
        self.summary: dict[str, Any] = {}

    def add(self, name: str, passed: bool, severity: str = "medium", detail: str = "", **metrics: Any) -> Check:
        c = Check(self.suite, name, bool(passed), severity, detail, metrics)
        self.checks.append(c)
        return c

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.checks)

    def save(self, name: str | None = None) -> Path:
        RESULTS.mkdir(parents=True, exist_ok=True)
        p = RESULTS / f"{name or self.suite}.json"
        data = {"meta": self.meta, "summary": {**self.summary, "passed": self.passed, "total": len(self.checks)},
                "checks": [asdict(c) for c in self.checks]}
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return p

    def table(self) -> str:
        rows = ["| Test | Result | Severity | Detail |", "|---|---|---|---|"]
        for c in self.checks:
            rows.append(f"| {c.name} | {'PASS' if c.passed else '**FAIL**'} | {c.severity} | {c.detail[:90]} |")
        return "\n".join(rows)


def pct(n: int, d: int) -> str:
    return f"{(100 * n / d):.1f}%" if d else "n/a"
