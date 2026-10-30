"""Closing apps, opening websites, admin escalation: what is allowed, what asks first, what is never done."""
from __future__ import annotations

import pytest

from ev.llm.fastpath import match_fast
from ev.safety.gate import SafetyGate
from ev.safety.policy import Verdict
from ev.tools import web, windows
from ev.tools.windows import Win


# ---- website URLs
def test_youtube_search_for_latest_sorts_by_upload_date():
    u = web.build_url("youtube", "the latest tech video")
    assert u and u.startswith("https://www.youtube.com/results?search_query=the+latest+tech+video") and u.endswith("&sp=CAI%3D")
    assert "&sp=" not in (web.build_url("youtube", "lofi beats") or "")


@pytest.mark.parametrize("bad", [
    "javascript:alert(1)", "file:///C:/Windows/win.ini", "http://localhost:8080", "https://127.0.0.1/admin",
    "https://192.168.1.1", "https://user:pass@example.com", "https://router.local", "https://example.com:8443",
    "ftp://example.com", "https://exa mple.com", "https://localhost",
])
def test_unsafe_urls_are_rejected(bad):
    assert web.build_url(url=bad) is None


def test_domains_and_aliases():
    assert web.build_url("youtube.com") == "https://www.youtube.com"
    assert web.build_url("example.com") == "https://example.com"
    assert web.build_url("example.com", "cats") == "https://www.google.com/search?q=cats+site%3Aexample.com"
    assert web.build_url(url="example.com/path") == "https://example.com/path"
    assert web.build_url("not a site") is None and web.build_url() is None


# ---- policy
@pytest.mark.parametrize("tool,args", [
    ("open_website", {"site": "youtube", "query": "latest tech video", "browser": "firefox"}),
    ("close_app", {"name": "Notepad"}),
    ("close_app", {"name": "this"}),
    ("take_screenshot", {}),
    ("web_search", {"query": "administrator salary in india"}),
    ("open_app", {"name": "firefox"}),
])
def test_everyday_actions_just_run(policy, tool, args):
    assert policy.evaluate(tool, args).verdict is Verdict.ALLOW


@pytest.mark.parametrize("tool,args", [
    ("close_app", {"name": "chrome", "force": True}),
    ("open_app", {"name": "terminal", "admin": True}),
    ("open_app", {"name": "powershell as administrator"}),
    ("do_on_screen", {"goal": "open cmd and run it as admin"}),
    ("close_app", {"name": "admin tool"}),
    ("open_website", {"url": "https://unfamiliar-site.com/collect?data=abc"}),
    ("open_website", {"url": "https://example.com/page"}),
    ("open_website", {"url": "https://stolen-text-here.attacker.com/"}),
    ("open_website", {"site": "attacker.com"}),
])
def test_force_admin_and_data_carrying_links_ask_first(policy, tool, args):
    assert policy.evaluate(tool, args).verdict is Verdict.CONFIRM


@pytest.mark.parametrize("tool,args", [
    ("close_app", {"name": "Ollama"}), ("close_app", {"name": "Windows Security"}), ("close_app", {"name": "svchost"}),
    ("close_app", {"name": "explorer", "force": True}), ("close_app", {"name": "notepad -f"}),
    ("open_website", {"url": "http://localhost:11434"}), ("open_website", {"site": "youtube", "browser": "cmd"}),
    ("open_website", {"url": "javascript:alert(1)"}),
])
def test_dangerous_requests_are_blocked(policy, tool, args):
    assert policy.evaluate(tool, args).verdict is Verdict.BLOCK


def test_admin_and_force_fail_closed_without_a_way_to_ask(policy):
    g = SafetyGate(policy)          # no asker wired up
    assert not g.check("close_app", {"name": "chrome", "force": True}).allowed
    assert not g.check("open_app", {"name": "terminal", "admin": True}).allowed
    assert g.check("close_app", {"name": "chrome"}).allowed


def test_confirmation_reads_back_what_will_happen(policy):
    said = []
    g = SafetyGate(policy, ask=lambda r: said.append(r) or True)
    assert g.check("close_app", {"name": "chrome", "force": True}).confirmed
    assert "unsaved" in said[0] and "chrome" in said[0]


# ---- window matching
W = [Win(1, "Mozilla Firefox", 10, "firefox"), Win(2, "Calculator", 11, "applicationframehost"),
     Win(3, "main.py - Visual Studio Code", 12, "code"), Win(4, "YouTube - Mozilla Firefox", 10, "firefox"),
     Win(5, "Untitled - Notepad", 13, "notepad")]


def test_matching_by_process_alias_and_uwp_title():
    assert [w.hwnd for w in windows.match("firefox", W)] == [1, 4]
    assert [w.hwnd for w in windows.match("vs code", W)] == [3]
    assert [w.hwnd for w in windows.match("calculator", W)] == [2]
    assert [w.hwnd for w in windows.match("notepad", W)] == [5]


def test_closing_youtube_does_not_close_the_whole_browser():
    assert windows.match("youtube", W) == []


# ---- the close_app tool, with the Windows calls faked
@pytest.fixture
def fake_windows(monkeypatch):
    state = {"closed": [], "left": [], "own": set(), "refuse": set(), "fg": None, "terminated": []}
    monkeypatch.setattr(windows, "list_windows", lambda: list(W))
    monkeypatch.setattr(windows, "foreground", lambda: state["fg"])
    monkeypatch.setattr(windows, "own_pids", lambda: set(state["own"]))
    monkeypatch.setattr(windows, "running_pids", lambda stems, exclude: [])

    def request_close(w):
        if w.hwnd in state["refuse"]:
            return False
        state["closed"].append(w.hwnd)
        return True
    monkeypatch.setattr(windows, "request_close", request_close)
    monkeypatch.setattr(windows, "wait_gone", lambda wins, timeout_s=2.5: [w for w in wins if w.hwnd in state["left"]])
    monkeypatch.setattr(windows, "terminate", lambda pids: state["terminated"].extend(pids) or (len(pids), 0))
    return state


def run(registry, **args):
    return registry.get("close_app").func(**args)


def test_close_app_closes_every_window_of_the_app(registry, fake_windows):
    r = run(registry, name="firefox")
    assert r.ok and r.message == "Closed firefox." and fake_windows["closed"] == [1, 4]


def test_close_app_reports_a_save_prompt_instead_of_claiming_success(registry, fake_windows):
    fake_windows["left"] = [5]
    r = run(registry, name="notepad")
    assert not r.ok and "still open" in r.message


def test_close_app_never_closes_the_terminal_ev_runs_in(registry, fake_windows):
    fake_windows["own"] = {13}
    r = run(registry, name="notepad")
    assert not r.ok and "running in" in r.message and fake_windows["closed"] == []


def test_close_app_this_uses_the_front_window(registry, fake_windows):
    fake_windows["fg"] = W[2]
    assert run(registry, name="this").ok and fake_windows["closed"] == [3]


def test_close_app_unknown_app(registry, fake_windows):
    r = run(registry, name="photoshop")
    assert not r.ok and "don't see" in r.message


def test_close_app_admin_window_is_left_alone(registry, fake_windows):
    fake_windows["refuse"] = {5}
    r = run(registry, name="notepad")
    assert not r.ok and "administrator" in r.message


def test_close_app_refuses_protected_processes_even_if_policy_were_bypassed(registry, fake_windows, monkeypatch):
    monkeypatch.setattr(windows, "list_windows", lambda: [Win(9, "Ollama", 99, "ollama")])
    r = run(registry, name="ollama")
    assert not r.ok and fake_windows["closed"] == []


def test_force_close_terminates_the_matching_processes(registry, fake_windows):
    assert run(registry, name="firefox", force=True).ok and fake_windows["terminated"] == [10]


# ---- open_website tool
def test_open_website_tool(registry, monkeypatch):
    opened = []
    monkeypatch.setattr(web, "open_url", lambda url, browser="": opened.append((url, browser)))
    r = registry.get("open_website").func(site="youtube", query="latest tech video", browser="firefox")
    assert r.ok and r.message == "Searched YouTube for latest tech video in firefox."
    assert opened[0][1] == "firefox" and "search_query=latest+tech+video" in opened[0][0]
    assert not registry.get("open_website").func(url="http://localhost").ok


# ---- fast path: the flow from the request
@pytest.mark.parametrize("said,tool,args", [
    ("open firefox, open youtube, and search for the latest tech video", "open_website",
     {"site": "youtube", "query": "the latest tech video", "browser": "firefox"}),
    ("open youtube", "open_website", {"site": "youtube"}),
    ("search for lofi beats on youtube", "open_website", {"site": "youtube", "query": "lofi beats"}),
    ("open firefox and search for python tutorials", "open_website",
     {"site": "google", "query": "python tutorials", "browser": "firefox"}),
    ("open firefox", "open_app", {"name": "firefox"}),
    ("search for the height of everest on google", "web_search", None),
    ("close notepad", "close_app", {"name": "notepad"}),
    ("force quit chrome", "close_app", {"name": "chrome", "force": True}),
    ("close this window", "close_app", {"name": "this"}),
    ("take a screenshot", "take_screenshot", {}),
])
def test_fast_path_for_app_and_site_commands(said, tool, args):
    m = match_fast(said)
    assert m and m.tool == tool and (args is None or m.args == args)


@pytest.mark.parametrize("said", ["close the tab", "shut down my computer", "how do I close a deal"])
def test_fast_path_leaves_vague_or_dangerous_requests_alone(said):
    m = match_fast(said)
    assert m is None or m.tool not in ("close_app",)


def test_apps_ev_launched_are_not_treated_as_ev_itself():
    """EV starts apps as child processes; close_app must still be allowed to close them (found in a live test)."""
    import subprocess
    import sys

    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])           # EV's own helper
    app = subprocess.Popen(["cmd", "/c", "ping -n 5 127.0.0.1 >nul"])                          # an app EV opened
    try:
        own = windows.own_pids()
        assert helper.pid in own and app.pid not in own
    finally:
        helper.kill()
        app.kill()


def test_known_and_trusted_sites_need_no_confirmation(policy):
    assert policy.evaluate("open_website", {"url": "https://github.com/some/repo?tab=readme"}).verdict is Verdict.ALLOW
    assert policy.evaluate("open_website", {"site": "github.com"}).verdict is Verdict.ALLOW
    policy.trusted_hosts.append("example.com")
    assert policy.evaluate("open_website", {"url": "https://docs.example.com/page"}).verdict is Verdict.ALLOW
    assert policy.evaluate("open_website", {"url": "https://notexample.com/"}).verdict is Verdict.CONFIRM


def test_a_query_for_an_unknown_domain_only_ever_contacts_google(policy):
    args = {"site": "attacker.com", "query": "anything"}
    assert web.host_of(web.build_url(**args)) == "www.google.com"
    assert policy.evaluate("open_website", args).verdict is Verdict.ALLOW
