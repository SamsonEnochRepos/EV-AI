"""python -m ev.agents.install   prints the hook settings that point Claude Code (and Codex) at EV. It changes nothing by itself.

Add --write-claude to merge the Claude Code entry into ~/.claude/settings.json (a backup is kept next to it). Codex's hook
format has changed between versions and is not verified here: the Codex snippet is a starting point to check against
`codex --help` / its docs before use.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ev.core.config import load_config

from .listener import load_or_create_token


def claude_hook(port: int, token: str) -> dict[str, Any]:
    return {"matcher": "*", "hooks": [{"type": "http", "url": f"http://127.0.0.1:{port}/v1/permission/claude",
                                       "headers": {"Authorization": f"Bearer {token}"}, "timeout": 60}]}


def codex_hook(port: int, token: str, shim: Path) -> dict[str, Any]:
    return {"PermissionRequest": [{"hooks": [{"type": "command", "command": f'python "{shim}"',
                                              "env": {"EV_AGENT_PORT": str(port), "EV_AGENT_TOKEN": token}}]}]}


def merge_claude(settings_path: Path, entry: dict[str, Any]) -> None:
    data: dict[str, Any] = {}
    if settings_path.exists():
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        shutil.copy2(settings_path, settings_path.with_suffix(".json.ev-backup"))
    lst = data.setdefault("hooks", {}).setdefault("PermissionRequest", [])
    lst[:] = [e for e in lst if "/v1/permission/claude" not in json.dumps(e)] + [entry]     # replace an older EV entry, keep the rest
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(prog="ev.agents.install")
    ap.add_argument("--write-claude", action="store_true", help="merge the entry into ~/.claude/settings.json")
    a = ap.parse_args()
    cfg = load_config()
    port = int(cfg.get("agents", "port", 47831))
    token = load_or_create_token(cfg.data_path("agent_token"))
    entry = claude_hook(port, token)
    print("Claude Code: put this under \"hooks\" -> \"PermissionRequest\" in ~/.claude/settings.json (then check /hooks in Claude Code):\n")
    print(json.dumps({"hooks": {"PermissionRequest": [entry]}}, indent=2))
    shim = Path(__file__).with_name("ev_hook.py")
    print("\nCodex (UNVERIFIED format; check Codex's hooks docs first): ~/.codex/hooks.json\n")
    print(json.dumps({"hooks": codex_hook(port, token, shim)}, indent=2))
    if a.write_claude:
        p = Path.home() / ".claude" / "settings.json"
        merge_claude(p, entry)
        print(f"\nWrote the Claude Code entry to {p} (backup: settings.json.ev-backup).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
