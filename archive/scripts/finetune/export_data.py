"""Turn opt-in conversation logs and the eval datasets into a fine-tuning file. Nothing here runs unless you run it.

    python scripts/finetune/export_data.py --yes                       # data/training/turns.jsonl (+ eval seeds) -> data/training/export/
    python scripts/finetune/export_data.py --yes --no-evals            # only your own turns
    python scripts/finetune/export_data.py                             # a dry run: counts and a sample, writes nothing

EV does not record turns today. Collecting them is a separate opt-in that has not been built (the notebook outline in
qlora_qwen25_3b.ipynb explains why it is left to a decision of yours). If you build or hand-make `data/training/turns.jsonl`, one JSON
object per line:

    {"ts": "2026-11-02T09:30:00", "user": "what time is it", "assistant": "Half past nine.",
     "tools": [{"name": "get_time_date", "args": {}, "ok": true}], "untrusted": false, "keep": true}

`untrusted: true` marks a turn in which web, screen, file or note text reached the model; those turns are always dropped (the
reply may quote text someone else wrote, and training on it would teach the model to follow it). `keep: false` drops a turn too.

Everything that is written is redacted first (ev.core.log.redact plus emails, phone numbers, URLs, Windows user folders and long
digit runs), and the export refuses to write unless `--yes` is given, so a stray run can't produce a file of your conversations.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ev.core.log import redact  # noqa: E402
from ev.evals.datasets import INTENT  # noqa: E402

_EXTRA = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"https?://\S+|www\.\S+"), "<url>"),
    (re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)"), "<number>"),
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", re.I), r"C:\\Users\\<user>"),
    (re.compile(r"\b\d{9,}\b"), "<number>"),
]
SYSTEM = "You are Evie, a warm, quick-witted voice assistant on the user's laptop. Reply in one or two short spoken sentences."


def clean(text: str) -> str:
    text = redact(text)
    for pattern, repl in _EXTRA:
        text = pattern.sub(repl, text)
    return text.strip()


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def example(user: str, assistant: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """One chat-format training example. A tool turn is the user's words then the assistant's tool call(s), nothing else:
    what the tool returned is not something to learn to say."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": clean(user)}]
    if tools:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": t["name"], "arguments": _scrub(t.get("args", {}))}} for t in tools]})
    else:
        msgs.append({"role": "assistant", "content": clean(assistant)})
    return {"messages": msgs}


def from_turns(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    skipped = {"untrusted": 0, "not_kept": 0, "failed_tool": 0, "bad_row": 0, "too_short": 0}
    if not path.is_file():
        return out, skipped
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            user, assistant = str(row["user"]), str(row.get("assistant", ""))
            tools = [t for t in row.get("tools", []) if isinstance(t, dict) and "name" in t]
        except (ValueError, KeyError, TypeError):
            skipped["bad_row"] += 1
            continue
        if row.get("untrusted"):
            skipped["untrusted"] += 1
        elif row.get("keep") is False:
            skipped["not_kept"] += 1
        elif any(not t.get("ok", True) for t in tools):
            skipped["failed_tool"] += 1                       # don't teach calls that did not work
        elif len(user.split()) < 2 or (not tools and len(assistant.split()) < 2):
            skipped["too_short"] += 1
        else:
            out.append(example(user, assistant, tools))
    return out, skipped


def from_evals() -> list[dict[str, Any]]:
    """The intent eval's hand-written utterances as tool-call examples. Weak on its own (72 items) but it is the one dataset
    that is known to be correct, and it must also stay a held-out test: see the notebook, which splits by utterance."""
    out = []
    for it in INTENT:
        if it.get("tool"):
            out.append(example(it["u"], "", [{"name": it["tool"], "args": it.get("args", {})}]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", type=Path, default=ROOT / "data" / "training" / "turns.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "training" / "export")
    ap.add_argument("--no-evals", action="store_true", help="leave out the intent-eval examples")
    ap.add_argument("--val", type=float, default=0.1, help="share of your own turns held out for validation")
    ap.add_argument("--yes", action="store_true", help="actually write the files (without it this is a dry run)")
    a = ap.parse_args()

    mine, skipped = from_turns(a.turns)
    seeds = [] if a.no_evals else from_evals()
    random.Random(7).shuffle(mine)  # noqa: S311  (a repeatable split, not security)
    cut = int(len(mine) * a.val)
    val, train = mine[:cut], mine[cut:] + seeds
    print(f"turns file: {a.turns} ({'found' if a.turns.is_file() else 'not found'})")
    print(f"your turns kept {len(mine)}; skipped {skipped}; eval seeds {len(seeds)}; train {len(train)}, validation {len(val)}")
    for ex in (train[:2] + val[:1]):
        print("sample:", json.dumps(ex, ensure_ascii=False)[:300])
    if not a.yes:
        print("dry run: nothing written (add --yes to write)")
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        (a.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {a.out / 'train.jsonl'} and {a.out / 'val.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
