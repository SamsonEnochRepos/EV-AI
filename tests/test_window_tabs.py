"""Switching windows ("switch to VS Code", "switch back to the browser") and browser tabs ("the YouTube tab", "next tab")."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ev.control.tools import register_control_tools
from ev.llm.fastpath import match_fast
from ev.safety.policy import Verdict
from ev.screen import uia
from ev.tools import windows
from ev.tools.windows import Win

CODE = Win(1, "main.py - EV_AI - Visual Studio Code", 10, "code")
FF_A = Win(2, "Shape of You - YouTube — SamsonPC — Mozilla Firefox", 11, "firefox")
FF_PRIVATE = Win(3, "Some page — SamsonPC — Mozilla Firefox Private Browsing", 11, "firefox")
FF_B = Win(4, "Inbox — Agents — Mozilla Firefox", 12, "firefox")
TERM = Win(5, "PowerShell", 13, "windowsterminal")
OVERLAY = Win(6, "EV overlay", 14, "python")
ALL = [FF_PRIVATE, CODE, OVERLAY, FF_A, TERM, FF_B]            # front to back


# ---- which window "switch to X" means
@pytest.mark.parametrize("said,fg,want", [
    ("vs code", FF_A, CODE), ("visual studio code", TERM, CODE), ("vscode", FF_A, CODE),
    ("firefox", CODE, FF_A),                          # a private window is passed over for the same app
    ("the browser", CODE, FF_A), ("browser", TERM, FF_A),
    ("terminal", CODE, TERM),
    ("previous window", FF_PRIVATE, CODE),
    ("previous", CODE, FF_PRIVATE),                   # whatever was behind the front window, even a private one
])
def test_pick_for_focus(said, fg, want):
    assert windows.pick_for_focus(said, ALL, fg) == want, said


def test_the_overlay_is_never_the_previous_window_and_nothing_to_go_back_to_is_none():
    assert windows.pick_for_focus("previous", [CODE, OVERLAY], CODE) is None
    assert windows.pick_for_focus("vs code", [FF_A], FF_A) is None
    assert windows.pick_for_focus("notepad", ALL, CODE) is None


def test_a_second_window_of_the_same_app_is_preferred_to_the_one_in_front():
    other = Win(7, "Other — SamsonPC — Mozilla Firefox", 11, "firefox")
    assert windows.pick_for_focus("firefox", [FF_A, other], FF_A) == other


def test_browser_windows_leave_out_private_ones_unless_in_front():
    assert windows.browser_windows(ALL, CODE) == [FF_A, FF_B]
    assert windows.browser_windows(ALL, FF_PRIVATE) == [FF_PRIVATE, FF_A, FF_B]


# ---- what is said
@pytest.mark.parametrize("said,tool,args", [
    ("switch to vs code", "focus_window", {"title": "vs code"}),
    ("can you switch to VS Code", "focus_window", {"title": "vs code"}),
    ("switch back to the browser", "focus_window", {"title": "browser"}),
    ("go back to vs code", "focus_window", {"title": "vs code"}),
    ("go to the browser", "focus_window", {"title": "browser"}),
    ("switch back", "focus_window", {"title": "previous"}),
    ("switch to the previous window", "focus_window", {"title": "previous"}),
    ("alt tab", "focus_window", {"title": "previous"}),
    ("next tab", "press_keys", {"combo": "ctrl+pagedown"}),
    ("switch to the previous tab", "press_keys", {"combo": "ctrl+pageup"}),
    ("go to the third tab", "press_keys", {"combo": "ctrl+3"}),
    ("switch to tab two", "press_keys", {"combo": "ctrl+2"}),
    ("switch to the last tab", "press_keys", {"combo": "ctrl+9"}),
    ("switch to the youtube tab", "switch_tab", {"name": "youtube"}),
    ("can you open the whatsapp tab", "switch_tab", {"name": "whatsapp"}),
    ("go to the github tab", "switch_tab", {"name": "github"}),
])
def test_switch_phrases(said, tool, args):
    m = match_fast(said)
    assert m and m.tool == tool and m.args == args, said


@pytest.mark.parametrize("said,not_tool", [
    ("go to youtube", "focus_window"), ("go to github", "focus_window"), ("open a new tab", "switch_tab"),
    ("switch to the new tab", "switch_tab"), ("go back", "focus_window"), ("get back to work", "focus_window"),
    ("close the tab", "switch_tab"), ("switch to dark mode", "focus_window"),
])
def test_phrases_that_are_not_switching(said, not_tool):
    m = match_fast(said)
    assert not (m and m.tool == not_tool), said


def test_go_back_is_still_the_previous_track():
    m = match_fast("go back")
    assert m and m.tool == "media_control"


# ---- tab strip handling
class _El:
    def __init__(self, name, left, top, right, bottom):
        self.element_info = SimpleNamespace(
            name=name, rectangle=SimpleNamespace(left=left, top=top, right=right, bottom=bottom,
                                                 width=lambda: right - left, height=lambda: bottom - top))


def _strip():
    top = [_El("Google", 0, 0, 100, 44), _El("Shape of You - YouTube", 100, 0, 200, 44), _El("YouTube Music - Home page and more", 200, 0, 300, 44)]
    page = [_El("Chats", 76, 190, 115, 219), _El("Calls", 122, 190, 208, 219)]            # tabs inside a web page
    return SimpleNamespace(descendants=lambda control_type=None: top + page)


def test_only_the_tab_row_counts_and_a_title_is_found_by_its_words():
    tabs = uia.browser_tabs(_strip())
    assert [t.name for t in tabs] == ["Google", "Shape of You - YouTube", "YouTube Music - Home page and more"]
    assert uia.find_tab(tabs, "youtube").name == "Shape of You - YouTube"            # the shorter title wins
    assert uia.find_tab(tabs, "youtube music").name.startswith("YouTube Music")
    assert uia.find_tab(tabs, "shape you").name == "Shape of You - YouTube"
    assert uia.find_tab(tabs, "whatsapp") is None and uia.find_tab(tabs, "  ") is None
    assert uia.browser_tabs(SimpleNamespace(descendants=lambda control_type=None: 1 / 0)) == []


# ---- the tools, with the desktop faked
@pytest.fixture
def desk(cfg, registry, policy, gate, monkeypatch):
    state = SimpleNamespace(fg=CODE, wins=list(ALL), fronted=[], selected=[], clicked=[], tabs={})
    monkeypatch.setattr(windows, "list_windows", lambda: list(state.wins))
    monkeypatch.setattr(windows, "foreground", lambda: state.fg)
    monkeypatch.setattr(windows, "bring_to_front", lambda w: state.fronted.append(w) or True)
    monkeypatch.setattr(uia, "window_for", lambda hwnd: hwnd)
    monkeypatch.setattr(uia, "browser_tabs", lambda hwnd: state.tabs.get(hwnd, []))
    monkeypatch.setattr(uia, "select_tab", lambda tab: state.selected.append(tab.name) or True)
    register_control_tools(registry, cfg, policy, gate)
    state.reg = registry
    return state


def _tab(name):
    return uia.UIElement(name, "TabItem", (0, 0, 10, 10), False, None)


def test_switch_to_an_app_brings_its_window_forward_and_says_the_app_not_the_file(desk):
    r = desk.reg.get("focus_window").func(title="vs code")
    assert r.ok and r.message == "Switched to Visual Studio Code." and desk.fronted == [CODE]
    desk.fg = CODE
    r = desk.reg.get("focus_window").func(title="the browser")
    assert r.ok and r.message == "Switched to Mozilla Firefox." and desk.fronted[-1] == FF_A


def test_switch_back_goes_to_the_window_behind_and_says_so_when_there_is_none(desk):
    desk.fg, desk.wins = FF_A, [FF_A, CODE]
    assert desk.reg.get("focus_window").func(title="previous").ok and desk.fronted == [CODE]
    desk.wins = [FF_A]
    r = desk.reg.get("focus_window").func(title="previous")
    assert not r.ok and "no other window" in r.message


def test_a_sensitive_window_is_not_switched_to_even_when_the_words_are_innocent(desk):
    secret = Win(9, "Registry Editor", 15, "regedit")
    desk.wins = [CODE, secret]
    r = desk.reg.get("focus_window").func(title="regedit")
    assert not r.ok and not desk.fronted


def test_switch_tab_selects_the_named_tab_and_speaks_the_users_words(desk):
    desk.fg = CODE
    desk.tabs = {FF_A.hwnd: [_tab("Google"), _tab("Shape of You - YouTube")], FF_B.hwnd: [_tab("Inbox")]}
    r = desk.reg.get("switch_tab").func(name="youtube")
    assert r.ok and r.message == "Switched to the youtube tab." and desk.selected == ["Shape of You - YouTube"] and desk.fronted == [FF_A]
    desk.selected.clear()
    r = desk.reg.get("switch_tab").func(name="inbox")                      # found in the second browser window
    assert r.ok and desk.fronted[-1] == FF_B and desk.selected == ["Inbox"]
    r = desk.reg.get("switch_tab").func(name="netflix")
    assert not r.ok and "couldn't find a tab called netflix" in r.message


def test_switch_tab_leaves_private_windows_alone_unless_the_user_is_in_one(desk):
    desk.tabs = {FF_PRIVATE.hwnd: [_tab("Secret thing")], FF_A.hwnd: [_tab("Google")]}
    desk.fg = CODE
    assert not desk.reg.get("switch_tab").func(name="secret").ok and not desk.fronted
    desk.fg = FF_PRIVATE
    assert desk.reg.get("switch_tab").func(name="secret").ok


def test_switch_tab_needs_a_browser_and_refuses_sensitive_tab_names(desk):
    desk.tabs = {FF_A.hwnd: [_tab("Windows Security - settings")]}
    assert not desk.reg.get("switch_tab").func(name="windows security").ok
    desk.wins = [CODE, TERM]
    r = desk.reg.get("switch_tab").func(name="youtube")
    assert not r.ok and "don't see a browser" in r.message


def test_policy_knows_the_new_tool_and_checks_what_it_is_told_to_switch_to(policy):
    assert policy.levels["switch_tab"] == "safe"
    assert policy.evaluate("switch_tab", {"name": "youtube"}).verdict is Verdict.ALLOW
    assert policy.evaluate("switch_tab", {"name": "registry editor"}).verdict is Verdict.BLOCK
    assert policy.evaluate("focus_window", {"title": "vs code"}).verdict is Verdict.ALLOW
