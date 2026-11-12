"""Is the package an agent wants to install current and safe? Asks the registries' structured data, never a web page.

Off unless `[agents] advisor = true`, because it contacts pypi.org, registry.npmjs.org and api.osv.dev (the only egress EV
has apart from DuckDuckGo). It can only make an answer stricter: a known vulnerability turns an install into a refusal, and
a package that is missing, deprecated, abandoned or installed from a URL turns it into a question. It never approves.
Only IDs and registry flags are used; no free text from a registry reaches the model or the speech.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from ev.core.log import get_logger

from .rules import Decision, Unparseable, Verdict, split_command, tokens

log = get_logger("agents.advisor")

HOSTS = ("pypi.org", "registry.npmjs.org", "api.osv.dev")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,80}$|^@[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")
_INSTALLERS: dict[tuple[str, ...], str] = {
    ("pip", "install"): "PyPI", ("python", "-m", "pip", "install"): "PyPI", ("uv", "pip", "install"): "PyPI",
    ("uv", "add"): "PyPI", ("poetry", "add"): "PyPI", ("npm", "install"): "npm", ("npm", "i"): "npm", ("npm", "add"): "npm",
    ("pnpm", "add"): "npm", ("pnpm", "install"): "npm", ("yarn", "add"): "npm"}
_TAKES_VALUE = frozenset({"-r", "--requirement", "-e", "--editable", "-c", "--constraint", "--index-url", "-i", "--registry"})


def packages(command: str) -> list[tuple[str, str, str, str]]:
    """(ecosystem, name, version, problem) for each package named in an install command. `problem` is set for a URL or path."""
    out: list[tuple[str, str, str, str]] = []
    try:
        parts = split_command(command)
    except Unparseable:                                            # the rules already ask about what can't be read
        return out
    for part in parts:
        try:
            toks = tokens(part)
        except Unparseable:
            continue
        low = tuple(t.lower() for t in toks)
        key = next((k for k in _INSTALLERS if low[:len(k)] == k), None)
        if key is None:
            continue
        eco = _INSTALLERS[key]
        skip = False
        for t in toks[len(key):]:
            if skip:
                skip = False
                continue
            if t in _TAKES_VALUE:
                skip = True
                continue
            if t.startswith("-"):
                continue
            if "://" in t or t.startswith((".", "/", "~", "git+")) or "\\" in t or re.match(r"^[a-zA-Z]:", t) or "/" in t and not t.startswith("@"):
                out.append((eco, t[:60], "", "it installs from a URL or a path"))
                continue
            name, ver = t, ""
            if eco == "PyPI":
                m = re.match(r"^([A-Za-z0-9._\-]+)\s*(?:==\s*([\w.\-+!]+))?(?:[<>=!~].*)?$", t.split("[")[0])
                name, ver = (m.group(1), m.group(2) or "") if m else (t, "")
            elif "@" in t[1:]:
                name, ver = t.rsplit("@", 1)
            if _NAME.match(name):
                out.append((eco, name, ver if re.match(r"^[\w.\-+]+$", ver or "x") else "", ""))
    return out


class Advisor:
    def __init__(self, client: httpx.Client | None = None, ttl_s: float = 3600, now: Callable[[], float] = time.time):
        self.http = client or httpx.Client(timeout=4.0, follow_redirects=False, headers={"User-Agent": "ev-package-check"})
        self.ttl_s = ttl_s
        self.now = now
        self._cache: dict[tuple[str, str, str], tuple[float, Decision | None]] = {}

    def __call__(self, tool: str, ti: dict[str, Any]) -> Decision | None:
        if tool not in ("Bash", "PowerShell", "shell"):
            return None
        worst: Decision | None = None
        for eco, name, ver, problem in packages(str(ti.get("command", ""))):
            d = Decision(Verdict.ASK, f"{name}: {problem}") if problem else self._check(eco, name, ver)
            if d is not None and (worst is None or (d.verdict is Verdict.DENY and worst.verdict is not Verdict.DENY)):
                worst = d
        return worst

    def _check(self, eco: str, name: str, ver: str) -> Decision | None:
        key = (eco, name, ver)
        hit = self._cache.get(key)
        if hit and self.now() - hit[0] < self.ttl_s:
            return hit[1]
        try:
            d = self._pypi(name, ver) if eco == "PyPI" else self._npm(name, ver)
        except (httpx.HTTPError, ValueError) as e:
            log.info("package check for %s failed (%s)", name, e)
            d = Decision(Verdict.ASK, f"I couldn't check {name} against the registry")
        self._cache[key] = (self.now(), d)
        return d

    def _pypi(self, name: str, ver: str) -> Decision | None:
        r = self.http.get(f"https://pypi.org/pypi/{quote(name)}/{quote(ver) + '/' if ver else ''}json")
        if r.status_code == 404:
            return Decision(Verdict.ASK, f"{name} is not on PyPI")
        r.raise_for_status()
        j = r.json()
        info = j.get("info", {})
        vulns = [v.get("id") for v in j.get("vulnerabilities", []) if isinstance(v, dict) and v.get("id") and not v.get("withdrawn")]
        if vulns:
            return Decision(Verdict.DENY, f"{name} {info.get('version', '')} has a known vulnerability ({', '.join(map(str, vulns[:2]))})")
        if info.get("yanked"):
            return Decision(Verdict.DENY, f"{name} {info.get('version', '')} was yanked")
        if any("Inactive" in c for c in info.get("classifiers", [])):
            return Decision(Verdict.ASK, f"{name} is marked inactive")
        return None

    def _npm(self, name: str, ver: str) -> Decision | None:
        r = self.http.get(f"https://registry.npmjs.org/{quote(name, safe='@')}/{quote(ver) if ver else 'latest'}")
        if r.status_code == 404:
            return Decision(Verdict.ASK, f"{name} is not on npm")
        r.raise_for_status()
        j = r.json()
        version = str(j.get("version", ver))
        o = self.http.post("https://api.osv.dev/v1/query", json={"package": {"name": name, "ecosystem": "npm"}, "version": version})
        o.raise_for_status()
        ids = [v.get("id") for v in o.json().get("vulns", []) if isinstance(v, dict) and v.get("id")]
        if ids:
            return Decision(Verdict.DENY, f"{name} {version} has a known vulnerability ({', '.join(map(str, ids[:2]))})")
        if j.get("deprecated"):
            return Decision(Verdict.ASK, f"{name} {version} is deprecated")
        return None
