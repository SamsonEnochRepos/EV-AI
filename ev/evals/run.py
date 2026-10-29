"""python -m ev.evals.run <suite> [--model NAME]   suites: security, security-nollm, intent, persona, memory, all"""
from __future__ import annotations

import argparse
import sys

from .common import Recorder


def _print(rec: Recorder) -> None:
    fails = [c for c in rec.checks if not c.passed]
    print(f"[{rec.suite}] {rec.passed}/{len(rec.checks)} passed")
    for c in fails[:40]:
        print(f"  FAIL {c.name}: {c.detail}")
    if rec.summary:
        print("  summary:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.summary.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["security", "security-nollm", "intent", "persona", "memory", "all"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None, help="suffix for the results file (e.g. model name)")
    a = ap.parse_args()
    recs: list[Recorder] = []

    if a.suite in ("security", "security-nollm", "all"):
        from . import security
        recs.append(security.run(include_llm=a.suite != "security-nollm"))
    if a.suite in ("intent", "all"):
        from . import functional
        r = functional.intent_accuracy(a.model)
        r.save(f"intent_{a.tag}" if a.tag else "intent")
        recs.append(r)
    if a.suite in ("persona", "all"):
        from . import functional
        r = functional.persona_eval(a.model)
        r.save(f"persona_{a.tag}" if a.tag else "persona")
        recs.append(r)
    if a.suite in ("memory", "all"):
        from . import functional
        r = functional.memory_eval()
        r.save()
        recs.append(r)
    for r in recs:
        _print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
