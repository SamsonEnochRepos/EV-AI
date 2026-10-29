"""Policy engine. Decides allow / confirm / block for every tool call from config alone.

Nothing the model says, and nothing found in untrusted data, can change a decision.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ev.core.config import ROOT, load_toml
from ev.core.log import get_logger

log = get_logger("policy")


class Verdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""


def clean_text(s: str) -> str:
    """NFKC-normalise and strip invisible/format/control chars so 'S\\u200bend' == 'Send'."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc") or ch in "\n\t")


def _strings(obj: Any) -> list[str]:
    """All string leaves of an arbitrary JSON-ish structure (bounded depth)."""
    out: list[str] = []
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        v, d = stack.pop()
        if d > 6:
            continue
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            stack += [(x, d + 1) for x in list(v.keys()) + list(v.values())]
        elif isinstance(v, list | tuple):
            stack += [(x, d + 1) for x in v]
    return out


def normalize_combo(combo: str) -> str:
    parts = [p.strip().lower() for p in re.split(r"[+\-\s]+", combo) if p.strip()]
    order = {"ctrl": 0, "control": 0, "alt": 1, "shift": 2, "win": 3, "windows": 3}
    mods = sorted((("ctrl" if p == "control" else "win" if p == "windows" else p)
                   for p in parts if p in order), key=lambda m: order[m])
    keys = [p for p in parts if p not in order]
    return "+".join(mods + keys)


class Policy:
    def __init__(self, data: dict[str, Any], root: Path = ROOT):
        self.d = data
        self.root = root
        self.levels: dict[str, str] = data.get("tools", {})
        self.unknown = data.get("defaults", {}).get("unknown_tool", "blocked")
        self.arg_res = [re.compile(p, re.I) for p in data.get("blocked_patterns", {}).get("args", [])]
        ctl = data.get("control", {})
        self.ctl_patterns = [p.lower() for p in ctl.get("blocked_control_patterns", [])]
        self.win_patterns = [p.lower() for p in ctl.get("blocked_window_patterns", [])]
        self.blocked_combos = {normalize_combo(c) for c in ctl.get("blocked_key_combos", [])}
        self.confirm_combos = {normalize_combo(c) for c in ctl.get("confirm_key_combos", [])}
        self.typing_max = int(ctl.get("typing_max_chars", 500))
        self.confirm_ctl = [p.lower() for p in ctl.get("confirm_control_patterns", [])]
        self.safe_dirs = [
            (root / p).resolve() for p in data.get("files", {}).get("safe_dirs", [])
        ]
        self.shell_enabled = bool(data.get("shell", {}).get("enabled", False))
        self.shell_allow = data.get("shell", {}).get("allowlist", [])
        self.app_allow = {a.lower() for a in data.get("apps", {}).get("allowlist", [])}
        self.start_menu = bool(data.get("apps", {}).get("allow_start_menu_resolution", False))
        self.inj_res = [re.compile(p, re.I) for p in data.get("injection", {}).get("phrases", [])]

    @classmethod
    def load(cls, path: Path) -> Policy:
        return cls(load_toml(path))

    # path sandbox
    def path_in_safe_dirs(self, raw: str) -> bool:
        if "%" in raw or raw.startswith("~") or "$" in raw:
            return False  # no env-var / home expansion tricks
        try:
            p = Path(raw)
            p = p if p.is_absolute() else self.root / p
            if self.d.get("files", {}).get("deny_symlinks", True):
                cur = p
                while True:
                    if cur.is_symlink():
                        return False
                    if cur.parent == cur:
                        break
                    cur = cur.parent
            real = Path(os.path.realpath(p))
        except (OSError, ValueError):
            return False
        return any(real == s or s in real.parents for s in self.safe_dirs)

    def evaluate(self, tool: str, args: dict[str, Any] | None) -> Decision:
        args = args if isinstance(args, dict) else {}
        level = self.levels.get(tool, self.unknown)
        if level == "blocked":
            return Decision(Verdict.BLOCK, f"{tool} is not permitted")

        raw_strs = _strings(args)
        if any("\x00" in s for s in raw_strs):
            return Decision(Verdict.BLOCK, "null byte in argument")
        strs = [clean_text(s) for s in raw_strs]
        for s in strs:
            for rx in self.arg_res:
                if rx.search(s):
                    return Decision(Verdict.BLOCK, "argument matches a blocked pattern")
        args = {k: clean_text(v) if isinstance(v, str) else v for k, v in args.items()}

        if tool in ("delete_file", "move_file", "take_note_file"):
            for key in ("path", "src", "dst", "dest"):
                if key in args and not self.path_in_safe_dirs(str(args[key])):
                    return Decision(Verdict.BLOCK, "path is outside the safe folders")

        if tool == "open_app":
            name = str(args.get("name", "")).strip().lower()
            if (
                not re.fullmatch(r"[a-z0-9][a-z0-9 .+_-]{0,59}", name)
                or re.search(r"(^|\s)[-/]", name)  # argument injection ("notepad -arg")
                or re.search(r"\.(exe|bat|cmd|ps1|vbs|lnk|msi|dll|js|com|scr)$", name)
            ):
                return Decision(Verdict.BLOCK, "invalid app name")
            if name not in self.app_allow and not self.start_menu:
                return Decision(Verdict.BLOCK, "app is not on the allowlist")

        if tool == "run_shell" and not self.shell_enabled:
            return Decision(Verdict.BLOCK, "shell is disabled")

        if tool in ("click_element", "type_text", "paste", "press_keys", "focus_window"):
            for key in ("name", "control", "target", "window", "title"):
                v = str(args.get(key, "")).lower()
                if v and any(p in v for p in self.ctl_patterns + self.win_patterns):
                    return Decision(Verdict.BLOCK, "target looks like a sensitive field or window")

        if tool == "click_element":
            nm = str(args.get("name", "")).lower()
            if any(re.search(rf"\b{re.escape(p)}", nm) for p in self.confirm_ctl):
                return Decision(Verdict.CONFIRM, f"click {args.get('name')}")

        if tool == "type_text":
            if len(str(args.get("text", ""))) > self.typing_max:
                return Decision(Verdict.BLOCK, "text too long to type")

        if tool == "press_keys":
            combo = normalize_combo(str(args.get("combo", "")))
            if not combo:
                return Decision(Verdict.BLOCK, "empty key combo")
            if combo in self.blocked_combos:
                return Decision(Verdict.BLOCK, f"{combo} is not permitted")
            if combo in self.confirm_combos:
                return Decision(Verdict.CONFIRM, f"press {combo}")

        if level == "confirm":
            return Decision(Verdict.CONFIRM, f"{tool} needs your approval")
        return Decision(Verdict.ALLOW)

    def check_sensitive_target(self, window_title: str, control_name: str = "",
                               is_password: bool = False) -> Decision:
        """Called by control layer with the *actual* resolved UI element before acting."""
        if is_password:
            return Decision(Verdict.BLOCK, "password field")
        blob = f"{window_title} {control_name}".lower()
        if any(p in blob for p in self.ctl_patterns + self.win_patterns):
            return Decision(Verdict.BLOCK, "sensitive window or control")
        return Decision(Verdict.ALLOW)
