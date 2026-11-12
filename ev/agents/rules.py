"""Decide whether a coding agent's request may go ahead: allow, deny, or ask the user.

This is deliberately plain code. It reads only the request (tool name, command, path, project folder) and the rules in
`config/agent_rules.toml`. No model, web page or file content takes part in the decision, so nothing an attacker can write
into a repo can talk it into approving. Order of authority: deny, then ask, then allow. A voice-granted group
("let Claude install packages for an hour") can only turn an `ask` into an `allow`; it never touches `deny`.
"""
from __future__ import annotations

import re
import shlex
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any


class Verdict(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str
    group: str = ""            # for ASK: the rule group a voice grant could open ("installs"), if there is one


@dataclass
class Rules:
    deny_words: re.Pattern[str]
    deny_patterns: list[re.Pattern[str]]
    sensitive: list[re.Pattern[str]]
    read_tools: frozenset[str]
    write_tools: frozenset[str]
    commands: list[tuple[str, ...]]
    risky_flags: dict[str, list[str]] = field(default_factory=dict)
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str, strict_project_code: bool = False) -> Rules:
        d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        if not strict_project_code:
            d["allow"]["commands"] = [*d["allow"]["commands"], *d["allow"].get("project_code", [])]
        words = "|".join(re.escape(w) for w in d["deny"]["words"])
        return cls(
            deny_words=re.compile(rf"(?<![\w.\-])(?:{words})(?![\w.\-])", re.I),
            deny_patterns=[re.compile(p, re.I) for p in d["deny"]["patterns"]],
            sensitive=[re.compile(p, re.I) for p in d["deny"]["sensitive"]],
            read_tools=frozenset(d["allow"]["read_tools"]),
            write_tools=frozenset(d["allow"]["write_tools"]),
            commands=[tuple(c.lower().split()) for c in d["allow"]["commands"]],
            risky_flags={k.lower(): [f.lower() for f in v] for k, v in d["allow"].get("risky_flags", {}).items()},
            groups={k: {"label": v.get("label", k), "commands": [tuple(c.lower().split()) for c in v.get("commands", [])],
                        "tools": frozenset(v.get("tools", []))} for k, v in d.get("groups", {}).items()},
        )


# ---- reading a shell command -------------------------------------------------------------------------------------------
_SEPARATORS = ("&&", "||", ";", "|", "&", "\n")
_SAFE_REDIRECT = re.compile(r"\d?>>?\s*(?:&\d|nul\b|/dev/null)|\d?>&\d", re.I)
_SUBSTITUTION = re.compile(r"\$\(|`|<\(|>\(|\$\{")


class Unparseable(Exception):
    pass


def split_command(cmd: str) -> list[str]:
    """The parts of a command line separated by `&&`, `||`, `;`, `|`, `&` or a newline, ignoring separators inside quotes.
    Raises Unparseable for an unbalanced quote or a substitution (`$(...)`, backticks), which cannot be judged part by part."""
    parts: list[str] = []
    cur: list[str] = []
    quote = ""
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            if ch == quote:
                quote = ""
            elif quote == '"' and ch == "`":
                raise Unparseable("a command inside quotes")
            elif quote == '"' and ch == "$" and cmd[i + 1:i + 2] in ("(", "{"):
                raise Unparseable("a command inside quotes")
            cur.append(ch)
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            cur.append(ch)
            i += 1
            continue
        if _SUBSTITUTION.match(cmd, i):
            raise Unparseable("a command built from another command")
        sep = next((x for x in _SEPARATORS if cmd.startswith(x, i)), None)
        if sep == "&" and (cmd[i - 1:i] == ">" or cmd[i + 1:i + 2] == ">"):
            sep = None                          # `2>&1` and `&>` are redirects, not separators
        if sep:
            parts.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    if quote:
        raise Unparseable("an unfinished quote")
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def tokens(part: str) -> list[str]:
    try:
        toks = shlex.split(part, posix=False)
    except ValueError as e:
        raise Unparseable(str(e)) from e
    return [t.strip("\"'") for t in toks]


def _norm_cmd(tok: str) -> str:
    t = tok.lower().replace("\\", "/").rsplit("/", 1)[-1]
    return t[:-4] if t.endswith(".exe") else t


def _match_prefix(toks: list[str], prefixes: list[tuple[str, ...]]) -> tuple[str, ...] | None:
    low = [_norm_cmd(toks[0])] + [t.lower() for t in toks[1:]] if toks else []
    return next((p for p in prefixes if len(low) >= len(p) and tuple(low[:len(p)]) == p), None)


def _matches(toks: list[str], prefixes: list[tuple[str, ...]]) -> bool:
    return _match_prefix(toks, prefixes) is not None


def _risky_flag(toks: list[str], flags: list[str]) -> str | None:
    """The first argument that is one of `flags`: exact, `--flag=value`, a `name*` prefix, or a short flag inside a bundle."""
    for t in (x.lower() for x in toks[1:]):
        for f in flags:
            if f.endswith("*") and t.startswith(f[:-1]):
                return t
            if t == f or (f.startswith("--") and t.startswith(f + "=")):
                return t
            if len(f) == 2 and f[0] == "-" and f[1] != "-" and t.startswith("-") and not t.startswith("--") and f[1] in t[1:] and t[1:].isalpha():
                return t
    return None


_VARIABLE = re.compile(r"\$|%[^%\s]+%")
# A backslash or caret before a quote or an operator means different things to bash, PowerShell and cmd, so this parser cannot tell
# where a quoted part ends. Windows paths are fine (C:\Users\x); only \" \; ^& and the like are refused.
_ESCAPES = re.compile(r"\\['\"`;&|()<>$\n]|\^['\"&|()<>]")
_ODD_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u2028\u2029\u00a0\u200b-\u200f\ufeff]")


# ---- paths -------------------------------------------------------------------------------------------------------------
def resolve(path: str, cwd: str) -> Path | None:
    try:
        p = Path(path).expanduser()
        if not p.is_absolute() and not PureWindowsPath(path).is_absolute():
            p = Path(cwd) / p
        return p.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def inside(path: str, cwd: str) -> bool:
    p, root = resolve(path, cwd), resolve(cwd, cwd)
    if p is None or root is None:
        return False
    return p == root or root in p.parents


def _looks_like_path(tok: str) -> bool:
    return bool(tok) and not tok.startswith("-") and (
        "/" in tok or "\\" in tok or tok.startswith(("~", ".")) or re.match(r"^[a-zA-Z]:", tok) is not None)


# ---- the decision ------------------------------------------------------------------------------------------------------
def evaluate(rules: Rules, tool: str, tool_input: dict[str, Any], cwd: str, granted: frozenset[str] = frozenset()) -> Decision:
    """`granted` is the set of group names the user opened by voice for this project right now."""
    cwd = cwd or "."
    if tool in rules.write_tools or tool in rules.read_tools:
        return _file_tool(rules, tool, tool_input, cwd)
    if tool in ("Bash", "PowerShell", "shell", "local_shell", "exec_command"):
        raw = tool_input.get("command") or tool_input.get("cmd") or ""
        if isinstance(raw, list):
            raw = " ".join(str(x) for x in raw)
        return _shell(rules, str(raw), cwd, granted)
    for name in granted:
        if tool in rules.groups.get(name, {}).get("tools", ()):
            return Decision(Verdict.ALLOW, f"you allowed the agent to {rules.groups[name]['label']}")
    for name, g in rules.groups.items():
        if tool in g["tools"]:
            return Decision(Verdict.ASK, f"it wants to {g['label']}", name)
    return Decision(Verdict.ASK, f"it wants to use {tool}")


def _file_tool(rules: Rules, tool: str, ti: dict[str, Any], cwd: str) -> Decision:
    paths = [str(ti[k]) for k in ("file_path", "path", "notebook_path") if ti.get(k)]
    # A search pattern is not a path, except Glob's pattern and Grep's glob: those can point anywhere (C:/Users/*/.ssh/*, ../**).
    globs = [str(ti[k]) for k in ("glob", "pattern" if tool == "Glob" else "glob") if isinstance(ti.get(k), str) and ti.get(k)]
    for p in paths + globs:
        if any(rx.search(p) for rx in rules.sensitive):
            return Decision(Verdict.DENY, "that file holds credentials or controls this approval system")
    if tool in rules.write_tools:
        if not paths:
            return Decision(Verdict.ASK, f"{tool} without a file path")
        if all(inside(p, cwd) for p in paths):
            return Decision(Verdict.ALLOW, "editing inside the project folder")
        return Decision(Verdict.ASK, "it wants to write outside the project folder")
    if all(inside(p, cwd) for p in paths) and not any(_escapes_project(g) for g in globs):
        return Decision(Verdict.ALLOW, "reading inside the project folder")
    return Decision(Verdict.ASK, "it wants to read outside the project folder")


def _escapes_project(pattern: str) -> bool:
    """A glob that is absolute, starts from home or a drive, climbs with `..`, or uses a variable can match outside the project."""
    p = pattern.replace("\\", "/")
    return p.startswith(("/", "~")) or re.match(r"^[a-zA-Z]:", p) is not None or ".." in p.split("/") or "$" in p or "%" in p


def _shell(rules: Rules, raw: str, cwd: str, granted: frozenset[str]) -> Decision:
    # Newlines are command separators, so they stay; only runs of spaces and tabs are collapsed.
    text = re.sub(r"[ \t]+", " ", raw.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text:
        return Decision(Verdict.ASK, "an empty command")
    if _ODD_CHARS.search(text):
        return Decision(Verdict.ASK, "it contains unusual control characters")
    if _ESCAPES.search(text):
        return Decision(Verdict.ASK, "it uses escapes I can't read reliably")
    # 1. things that are never fine, found anywhere in the text, even inside bash -c "..."
    if m := rules.deny_words.search(text):
        return Decision(Verdict.DENY, f"it would delete or destroy things ({m.group(0).lower()})")
    for rx in rules.deny_patterns:
        if rx.search(text):
            return Decision(Verdict.DENY, "that command is on the never-allowed list")
    if any(rx.search(text) for rx in rules.sensitive):
        return Decision(Verdict.DENY, "it touches credentials or this approval system")
    # 2. every part must be known-safe, or be in a group the user opened
    try:
        parts = split_command(text)
    except Unparseable as e:
        return Decision(Verdict.ASK, f"I can't tell what this does ({e})")
    asks: list[Decision] = []
    for part in parts:
        d = _part(rules, part, cwd, granted)
        if d.verdict is Verdict.DENY:
            return d
        if d.verdict is Verdict.ASK:
            asks.append(d)
    if asks:
        return asks[0]
    return Decision(Verdict.ALLOW, "read-only or check command")


def _part(rules: Rules, part: str, cwd: str, granted: frozenset[str]) -> Decision:
    cleaned = _SAFE_REDIRECT.sub(" ", part)
    if ">" in cleaned or "<" in cleaned:
        return Decision(Verdict.ASK, "it redirects output into a file")
    if _VARIABLE.search(cleaned):                    # judged before cd, voice grants and the allow-list: inside() can't resolve it
        return Decision(Verdict.ASK, "it uses a variable I can't resolve")
    try:
        toks = tokens(cleaned)
    except Unparseable as e:
        return Decision(Verdict.ASK, f"I can't tell what this does ({e})")
    if not toks:
        return Decision(Verdict.ALLOW, "")
    if toks[0].lower() in ("cd", "pushd", "set-location", "chdir") and len(toks) > 1:
        return Decision(Verdict.ALLOW, "") if inside(toks[1], cwd) else Decision(Verdict.ASK, "it moves outside the project folder")
    for name in granted:
        if _matches(toks, rules.groups.get(name, {}).get("commands", [])):
            return Decision(Verdict.ALLOW, f"you allowed the agent to {rules.groups[name]['label']}", name)
    if (prefix := _match_prefix(toks, rules.commands)) is not None:
        if (bad := _risky_flag(toks, rules.risky_flags.get(" ".join(prefix), []))) is not None:
            return Decision(Verdict.ASK, f"{bad} makes that command run a program or write a file")
        outside = [t for t in toks[1:] if _looks_like_path(t) and not inside(t, cwd)]
        if outside:
            return Decision(Verdict.ASK, "it reads outside the project folder")
        return Decision(Verdict.ALLOW, "")
    for name, g in rules.groups.items():
        if _matches(toks, g["commands"]):
            return Decision(Verdict.ASK, f"it wants to {g['label']}", name)
    return Decision(Verdict.ASK, f"I don't know {_norm_cmd(toks[0])}")


def describe(tool: str, tool_input: dict[str, Any]) -> str:
    """A short spoken description of the request, for the question. Long commands are cut; nothing here is executed."""
    if tool in ("Bash", "PowerShell", "shell", "local_shell", "exec_command"):
        cmd = re.sub(r"\s+", " ", str(tool_input.get("command") or tool_input.get("cmd") or "")).strip()
        return "run " + (cmd if len(cmd) <= 90 else cmd[:87] + "...")
    for k in ("file_path", "path", "notebook_path", "url"):
        if tool_input.get(k):
            name = str(tool_input[k]).replace("\\", "/").rsplit("/", 1)[-1] if k != "url" else str(tool_input[k])[:80]
            return f"{tool.lower()} {name}"
    return f"use {tool}"
