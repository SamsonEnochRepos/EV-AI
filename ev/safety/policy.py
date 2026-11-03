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
from ev.tools import web

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


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class Policy:
    def __init__(self, data: dict[str, Any], root: Path = ROOT, places: dict[str, Path] | None = None):
        self.d = data
        self.root = root
        self.places = places or {}
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
        files = data.get("files", {})
        self.safe_dirs = self._dirs(files.get("safe_dirs", []))
        self.open_dirs = self._dirs(files.get("open_dirs", []))
        self.write_dirs = self._dirs(files.get("write_dirs", []))     # where EV's own code may create files (the vault checks this too)
        self.open_exts = {str(e).lower().lstrip(".") for e in files.get("open_extensions", [])}
        self.shell_enabled = bool(data.get("shell", {}).get("enabled", False))
        self.shell_allow = data.get("shell", {}).get("allowlist", [])
        self.app_allow = {a.lower() for a in data.get("apps", {}).get("allowlist", [])}
        self.start_menu = bool(data.get("apps", {}).get("allow_start_menu_resolution", False))
        self.inj_res = [re.compile(p, re.I) for p in data.get("injection", {}).get("phrases", [])]
        apps = data.get("apps", {})
        self.protected_apps = {_compact(a) for a in apps.get("protected", [])}      # close_app never touches these
        self.no_force_apps = {_compact(a) for a in apps.get("no_force", [])}        # may be closed politely, never killed
        self.trusted_hosts = [h.lower().lstrip(".") for h in data.get("web", {}).get("trusted_hosts", [])]
        cp = data.get("confirm_patterns", {})
        self.confirm_res = [re.compile(p, re.I) for p in cp.get("args", [])]
        self.confirm_pattern_tools = set(cp.get("tools", []))

    @classmethod
    def load(cls, path: Path, places: dict[str, Path] | None = None) -> Policy:
        return cls(load_toml(path), places=places)

    def _dirs(self, entries: list[str]) -> list[Path]:
        """Folder list from policy.toml. {name} entries are filled from `places`; one that can't be (feature off) is dropped."""
        out: list[Path] = []
        for e in entries:
            names = _PLACEHOLDER.findall(e)
            if any(n not in self.places for n in names):
                continue
            out.append((self.root / _PLACEHOLDER.sub(lambda m: str(self.places[m.group(1)]), e)).resolve())
        return out

    # path sandbox
    def path_in_safe_dirs(self, raw: str) -> bool:
        return self.path_in_dirs(raw, self.safe_dirs)

    def path_in_dirs(self, raw: str, dirs: list[Path]) -> bool:
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
        return any(real == s or s in real.parents for s in dirs)

    def check_open_path(self, raw: str) -> Decision:
        """Whether open_file may open this path. Called on the arguments, and again by the tool on the path it resolved
        itself (the 'last' file), because opening a file can run it: only listed folders and only document/media types."""
        raw = clean_text(raw).strip()
        if not raw:
            return Decision(Verdict.BLOCK, "no file given")
        ext = Path(raw).suffix.lower().lstrip(".")
        if ext not in self.open_exts:
            return Decision(Verdict.BLOCK, "that kind of file can't be opened")
        if not self.path_in_dirs(raw, self.open_dirs):
            return Decision(Verdict.BLOCK, "that file is outside the folders I can open from")
        return Decision(Verdict.ALLOW)

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

        if tool == "open_file":
            path = str(args.get("path", "")).strip()
            if path:
                od = self.check_open_path(path)
                if od.verdict is Verdict.BLOCK:
                    return od
            elif args.get("which") != "last":
                return Decision(Verdict.BLOCK, "no file given")

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

        if tool == "open_website":
            d = self._website(args)
            if d is not None:
                return d

        if tool == "close_app":
            d = self._close_app(args)
            if d is not None:
                return d

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

        if tool == "manage_skill" and str(args.get("action")) == "delete":
            return Decision(Verdict.CONFIRM, "delete a skill")

        if tool == "open_app" and args.get("admin") is True:
            return Decision(Verdict.CONFIRM, "open an app with administrator rights")
        if tool in self.confirm_pattern_tools and any(rx.search(s) for s in strs for rx in self.confirm_res):
            return Decision(Verdict.CONFIRM, "that involves administrator rights")

        if level == "confirm":
            return Decision(Verdict.CONFIRM, f"{tool} needs your approval")
        return Decision(Verdict.ALLOW)

    def _website(self, args: dict[str, Any]) -> Decision | None:
        browser = str(args.get("browser", "")).strip().lower()
        if browser and browser not in web.BROWSERS:
            return Decision(Verdict.BLOCK, "unknown browser")
        target = web.build_url(str(args.get("site", "")), str(args.get("query", "")), str(args.get("url", "")))
        if target is None:
            return Decision(Verdict.BLOCK, "not a public website address")
        # Data can leave in a hostname, path, query or fragment, and policy can't tell a link the user spoke from one a
        # web page or screen suggested. So any address that isn't a known site or on the trusted list asks first.
        named_site = web.canonical_site(str(args.get("site", ""))) in web.SITES
        if (args.get("url") or not named_site) and not self._trusted_host(web.host_of(target)):
            if args.get("url") or args.get("site"):
                return Decision(Verdict.CONFIRM, f"open {web.host_of(target)}, which isn't a site I know")
        return None

    def _trusted_host(self, host: str) -> bool:
        host = host.lower()
        return host in web.KNOWN_HOSTS or any(host == t or host.endswith("." + t) for t in self.trusted_hosts)

    def _close_app(self, args: dict[str, Any]) -> Decision | None:
        name = str(args.get("name", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .+_-]{0,59}", name) or re.search(r"(^|\s)[-/]", name):
            return Decision(Verdict.BLOCK, "invalid app name")
        key = _compact(name)
        if key in self.protected_apps:
            return Decision(Verdict.BLOCK, f"{name} is protected")
        if args.get("force") is True:
            if key in self.no_force_apps:
                return Decision(Verdict.BLOCK, f"{name} can't be force closed")
            return Decision(Verdict.CONFIRM, f"force close {name}; unsaved work would be lost")
        return None

    def check_sensitive_target(self, window_title: str, control_name: str = "",
                               is_password: bool = False) -> Decision:
        """Called by control layer with the *actual* resolved UI element before acting."""
        if is_password:
            return Decision(Verdict.BLOCK, "password field")
        blob = f"{window_title} {control_name}".lower()
        if any(p in blob for p in self.ctl_patterns + self.win_patterns):
            return Decision(Verdict.BLOCK, "sensitive window or control")
        return Decision(Verdict.ALLOW)
