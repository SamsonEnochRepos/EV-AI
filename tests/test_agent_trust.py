"""Answering Claude Code's "trust this folder?" question by voice, and only when the screen really shows it."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ev.control import trust
from ev.control.tools import register_control_tools
from ev.llm.fastpath import match_compound, match_fast
from ev.safety.policy import Verdict
from ev.screen import uia
from ev.tools import windows
from ev.tools.windows import Win

MARK = "\u276f"


def screen(path="D:\\Projects\\GNSS", yes_selected=True, question=True, yes=True):
    """Roughly what Windows Terminal shows for the prompt (padded lines, as the terminal returns them)."""
    lines = ["", "Accessing workspace:", "", path, "",
             "Quick safety check: Is this a project you created or one you trust? (Like your own code, a well-known open source" if question else "",
             "project, or work from your team). If not, take a moment to review what's in this folder first.", "",
             "Claude Code'll be able to read, edit, and execute files here.", "", "Security guide", "",
             "  No, exit"]
    if yes:
        lines.append(f"{MARK} Yes, I trust this folder" if yes_selected else "  Yes, I trust this folder")
    lines += ["", "Enter to confirm \u00b7 Esc to cancel"]
    return "\r\n".join(ln.ljust(120) for ln in lines)


# ---- reading the screen
def test_the_real_prompt_is_read():
    p = trust.parse_claude_trust(screen())
    assert p == trust.TrustPrompt("D:\\Projects\\GNSS", True)


@pytest.mark.parametrize("text", [
    "",
    "you> open GNSS in claude code\r\nev> Opened GNSS in a terminal.",
    screen(question=False),                       # no question line
    screen(yes=False),                            # no "Yes, I trust this folder" choice
    screen(path="not a path"),
    screen(path="D:\\Projects\\GNSS" + "x" * 300),
    "Accessing workspace:\r\n\r\nD:\\Projects\\GNSS\r\n",                        # nothing else
    "Yes, I trust this folder\r\nIs this a project you created or one you trust",  # no workspace line
])
def test_anything_less_than_the_whole_prompt_is_not_it(text):
    assert trust.parse_claude_trust(text) is None


def test_it_says_when_yes_is_not_the_highlighted_choice():
    assert trust.parse_claude_trust(screen(yes_selected=False)).yes_selected is False


def test_only_the_last_prompt_on_screen_counts():
    older = screen(path="D:\\Projects\\Old")
    assert trust.parse_claude_trust(older + "\r\n" + screen()).path == "D:\\Projects\\GNSS"
    assert trust.parse_claude_trust(screen() + "\r\nAccessing workspace:\r\n\r\nD:\\Projects\\Next\r\nnothing else yet") is None


# ---- what is said
@pytest.mark.parametrize("said", [
    "trust this folder", "trust the folder", "Trust it.", "yes trust this folder", "yes, I trust this folder", "I trust this project",
    "can u trust this folder", "trust this folder for claude code", "accept the trust prompt", "accept the folder trust",
    "say yes to the trust prompt", "hey evie, trust the folder please",
])
def test_trust_phrases(said):
    m = match_fast(said)
    assert m and m.tool == "trust_agent_folder" and m.args == {}, said


@pytest.mark.parametrize("said", [
    "accept it", "yes", "i trust you", "trust me", "do you trust this website", "should I trust this folder", "open the trust folder",
    "trust this folder and delete everything", "accept the cookies",
])
def test_other_speech_is_not_the_trust_command(said):
    m = match_fast(said)
    assert not (m and m.tool == "trust_agent_folder"), said


def test_open_with_claude_and_trust_it_is_two_gated_steps():
    ms = match_compound("open the GNSS project with claude code and trust it")
    assert ms and [m.tool for m in ms] == ["open_project", "trust_agent_folder"]
    assert ms[0].args == {"name": "gnss", "how": "terminal", "agent": "claude"}


def test_the_trust_tool_is_voice_only_and_has_a_level(policy):
    assert policy.levels["trust_agent_folder"] == "safe"
    assert policy.evaluate("trust_agent_folder", {}).verdict is Verdict.ALLOW


# ---- the tool
TERM = Win(7, "claude", 70, "windowsterminal")


@pytest.fixture
def ui(registry, cfg, policy, gate, monkeypatch, tmp_path):
    projects = tmp_path / "Projects"
    (projects / "GNSS").mkdir(parents=True)
    (tmp_path / "Elsewhere" / "deep" / "thing").mkdir(parents=True)
    policy.project_roots = [projects]
    st = SimpleNamespace(fg=TERM, text=screen(str(projects / "GNSS")), keys=[], after=None, folder=projects / "GNSS", projects=projects,
                         tmp=tmp_path)

    def press(key):
        st.keys.append(key)
        if st.after is not None:
            st.text = st.after

    fake = SimpleNamespace(press=press)
    monkeypatch.setattr("ev.control.tools._pag", lambda: fake)
    monkeypatch.setattr("ev.control.tools._TRUST_WAIT_S", 0.2)
    monkeypatch.setattr("ev.control.tools.time.sleep", lambda s: None)
    monkeypatch.setattr(windows, "foreground", lambda: st.fg)
    monkeypatch.setattr(uia, "terminal_text", lambda hwnd: st.text)
    register_control_tools(registry, cfg, policy, gate)
    st.run = registry.get("trust_agent_folder").func
    return st


def test_it_presses_enter_once_and_says_what_it_saw(ui):
    ui.after = "claude is ready"                                       # the prompt is gone afterwards
    r = ui.run()
    assert r.ok and ui.keys == ["enter"] and "Trusted GNSS" in r.message


def test_it_does_not_claim_success_if_the_prompt_is_still_there(ui):
    r = ui.run()
    assert not r.ok and ui.keys == ["enter"] and "still showing" in r.message


def test_it_waits_for_the_prompt_and_then_gives_up_without_pressing_anything(ui):
    ui.text = "PS D:\\Projects\\GNSS> claude"
    r = ui.run()
    assert not r.ok and ui.keys == [] and "don't see" in r.message


@pytest.mark.parametrize("fg", [Win(8, "Mozilla Firefox", 80, "firefox"), Win(9, "claude", 90, "code"), None])
def test_it_only_acts_in_windows_terminal(ui, fg):
    ui.fg = fg
    assert not ui.run().ok and ui.keys == []


def test_it_refuses_a_folder_that_is_not_a_project_folder(ui):
    ui.text = screen(str(ui.tmp / "Elsewhere" / "deep" / "thing"))
    r = ui.run()
    assert not r.ok and ui.keys == [] and "isn't one of my project folders" in r.message
    ui.text = screen(str(ui.projects / "NotThere"))
    assert not ui.run().ok and ui.keys == []
    ui.text = screen(str(ui.projects))                                   # the root itself is not a folder inside a root
    assert not ui.run().ok and ui.keys == []


def test_it_leaves_it_to_the_user_when_yes_is_not_highlighted(ui):
    ui.text = screen(str(ui.folder), yes_selected=False)
    r = ui.run()
    assert not r.ok and ui.keys == [] and "highlighted" in r.message


def test_a_model_cannot_call_it(ui, registry):
    assert registry.get("trust_agent_folder").llm_visible is False
    assert "trust_agent_folder" not in [t["function"]["name"] for t in registry.schemas()]


def test_project_folder_check(policy, tmp_path):
    (tmp_path / "P" / "A").mkdir(parents=True)
    (tmp_path / "P" / "A" / "inner").mkdir()
    policy.project_roots = [tmp_path / "P"]
    assert policy.is_project_folder(tmp_path / "P" / "A")
    assert not policy.is_project_folder(tmp_path / "P")
    assert not policy.is_project_folder(tmp_path / "P" / "A" / "inner")
    assert not policy.is_project_folder(tmp_path / "P" / "missing")
    assert not policy.is_project_folder(Path("Z:\\nowhere\\GNSS"))
