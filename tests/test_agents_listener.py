"""The approval endpoint: who may talk to it, what comes back, and that nothing gets approved by accident."""
from __future__ import annotations

import http.client
import json
from pathlib import Path

import pytest

from ev.agents.listener import AgentListener, load_or_create_token
from ev.agents.rules import Rules
from ev.agents.supervisor import AgentSupervisor

ROOT = Path(__file__).resolve().parents[1]
CWD = str(ROOT)
TOKEN = "t" * 40


def req(tool, **ti):
    return {"tool_name": tool, "tool_input": ti, "cwd": CWD, "hook_event_name": "PermissionRequest"}


def behavior(out):
    return None if out is None else out["hookSpecificOutput"]["decision"]["behavior"]


@pytest.fixture()
def rules():
    return Rules.load(ROOT / "config" / "agent_rules.toml")


@pytest.fixture()
def sup(rules, tmp_path):
    asked: list[str] = []
    s = AgentSupervisor(rules, tmp_path / "audit.jsonl", ask=lambda q: asked.append(q) or None)
    s.asked = asked                                              # type: ignore[attr-defined]
    return s


def test_allow_and_deny_are_answered_without_asking(sup):
    assert behavior(sup.decide("claude", req("Bash", command="git status"))) == "allow"
    out = sup.decide("claude", req("Bash", command="rm -rf build"))
    assert behavior(out) == "deny" and "EV blocked" in out["hookSpecificOutput"]["decision"]["message"]
    assert sup.asked == []


def test_a_grey_area_request_is_asked_by_voice_and_the_answer_is_used(rules):
    for answer, want in ((True, "allow"), (False, "deny"), (None, None)):
        s = AgentSupervisor(rules, ask=lambda q, a=answer: a)
        assert behavior(s.decide("claude", req("Bash", command="npm install left-pad"))) == want


def test_the_question_names_the_agent_and_the_project(sup):
    sup.decide("claude", req("Bash", command="npm install left-pad"))
    assert sup.asked and sup.asked[0].startswith("let Claude Code run npm install left-pad in ")


def test_with_no_way_to_ask_the_agent_keeps_its_own_prompt(rules):
    assert AgentSupervisor(rules, ask=None).decide("claude", req("Bash", command="npm install x")) is None


def test_paused_means_no_decisions_at_all_not_even_allow(rules):
    s = AgentSupervisor(rules, ask=lambda q: True)
    s.pause(10)
    assert s.decide("claude", req("Bash", command="git status")) is None
    assert s.decide("claude", req("Bash", command="rm -rf x")) is None       # the agent's own prompt still guards it
    s.pause(0)
    assert behavior(s.decide("claude", req("Bash", command="git status"))) == "allow"


def test_a_voice_grant_is_scoped_to_the_project_and_expires(rules):
    t = [1000.0]
    s = AgentSupervisor(rules, ask=lambda q: None, now=lambda: t[0])
    s.decide("claude", req("Bash", command="git status"))                    # EV now knows which project the agent is in
    assert s.grant("installs", 30) == "install packages"
    assert behavior(s.decide("claude", req("Bash", command="pip install requests"))) == "allow"
    other = {**req("Bash", command="pip install requests"), "cwd": "C:/elsewhere"}
    assert s.decide("claude", other) is None                                 # another project: asked (and nobody answered)
    t[0] += 31 * 60
    assert s.decide("claude", req("Bash", command="pip install requests")) is None
    with pytest.raises(ValueError):
        s.grant("everything", 5)


def test_an_advisor_can_only_make_things_stricter(rules):
    from ev.agents.rules import Decision, Verdict

    s = AgentSupervisor(rules, ask=lambda q: None, advisor=lambda tool, ti: Decision(Verdict.DENY, "a known vulnerability"))
    assert behavior(s.decide("claude", req("Bash", command="git status"))) == "deny"
    s2 = AgentSupervisor(rules, ask=lambda q: None, advisor=lambda tool, ti: None)
    assert behavior(s2.decide("claude", req("Bash", command="git status"))) == "allow"


def test_malformed_requests_get_no_decision(sup):
    assert sup.decide("claude", {}) is None
    assert sup.decide("claude", {"tool_name": "Bash", "tool_input": "rm -rf /"}) is None


def test_every_decision_is_written_to_the_audit_log(sup, tmp_path):
    sup.decide("claude", req("Bash", command="git status"))
    sup.decide("claude", req("Bash", command="rm -rf x"))
    rows = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [r["verdict"] for r in rows] == ["allow", "deny"] and rows[1]["what"] == "run rm -rf x"
    assert "3 " not in sup.status() and "1 allowed, 1 blocked" in sup.status()


# ---- the HTTP edge
@pytest.fixture()
def server(sup):
    ln = AgentListener(sup, TOKEN, 0)
    assert ln.start()
    yield ln
    ln.stop()


def post(ln, path="/v1/permission/claude", body=None, token=TOKEN, ctype="application/json", host=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", ln.port, timeout=5)
    h = {"Content-Type": ctype, "Host": host or f"127.0.0.1:{ln.port}"}
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    c.request("POST", path, body=raw if raw is not None else json.dumps(body if body is not None else req("Bash", command="git status")), headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def test_a_good_request_gets_the_hook_response(server):
    status, data = post(server)
    assert status == 200 and json.loads(data)["hookSpecificOutput"]["decision"]["behavior"] == "allow"
    status, data = post(server, body=req("Bash", command="rm -rf x"))
    assert status == 200 and json.loads(data)["hookSpecificOutput"]["decision"]["behavior"] == "deny"


def test_no_decision_is_an_empty_204(server):
    assert post(server, body=req("Bash", command="npm install x")) == (204, b"")


@pytest.mark.parametrize("kw,code", [
    ({"token": None}, 401), ({"token": "wrong"}, 401), ({"host": "evil.example:80"}, 403), ({"path": "/v1/permission/other"}, 404),
    ({"path": "/"}, 404), ({"ctype": "text/plain"}, 415), ({"raw": "not json"}, 400), ({"raw": "[1,2]"}, 400),
])
def test_bad_requests_are_refused_before_anything_is_decided(server, sup, kw, code):
    assert post(server, **kw)[0] == code
    assert sup.asked == []


def test_a_huge_body_is_refused(server):
    assert post(server, raw="x" * 1_000_001)[0] == 413


def test_get_is_refused(server):
    c = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    c.request("GET", "/v1/permission/claude")
    assert c.getresponse().status == 405


def test_the_token_is_created_once_and_reused(tmp_path):
    a = load_or_create_token(tmp_path / "tok")
    assert len(a) >= 32 and load_or_create_token(tmp_path / "tok") == a
