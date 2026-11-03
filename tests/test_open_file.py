"""open_file: policy decides from config + arguments, the tool re-checks the file it resolved, and last_artifact is remembered."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from ev.core.config import ROOT, load_config
from ev.safety.gate import SafetyGate
from ev.safety.policy import Policy, Verdict
from ev.tools.base import validate_args
from ev.tools.builtin import ToolContext, build_tools
from ev.tools.scheduler import Scheduler


@pytest.fixture
def shots(tmp_path: Path) -> Path:
    d = tmp_path / "Screenshots"
    d.mkdir()
    return d


@pytest.fixture
def pol(shots: Path) -> Policy:
    return Policy.load(ROOT / "config" / "policy.toml", places={"screenshots": shots, "skills": ROOT / "skills"})


def verdict(pol: Policy, **args) -> Verdict:
    return pol.evaluate("open_file", args).verdict


def test_screenshot_folder_placeholder_is_filled_in(pol, shots):
    assert shots.resolve() in pol.open_dirs
    assert verdict(pol, path=str(shots / "EV 2026-01-02 15-04-00.png")) is Verdict.ALLOW


def test_unresolved_placeholders_are_dropped(pol):
    """{vault_ev} isn't known until the vault exists (Phase 2); an unknown placeholder must not become a folder."""
    assert all("{" not in str(d) for d in pol.open_dirs)
    assert len(pol.open_dirs) == 3                      # data/notes, data/sandbox, the screenshot folder


def test_which_last_is_allowed_without_a_path(pol):
    assert verdict(pol, which="last") is Verdict.ALLOW


def test_nothing_to_open_is_blocked(pol):
    assert verdict(pol) is Verdict.BLOCK
    assert verdict(pol, which="first") is Verdict.BLOCK


@pytest.mark.parametrize("name", ["setup.exe", "run.bat", "x.cmd", "x.ps1", "x.lnk", "x.vbs", "x.js", "page.html", "x.msi", "x.dll", "noext"])
def test_runnable_and_unknown_types_are_blocked(pol, shots, name):
    assert verdict(pol, path=str(shots / name)) is Verdict.BLOCK


@pytest.mark.parametrize("path", [
    "C:\\Windows\\System32\\notepad.exe", "C:\\Users\\Public\\x.png", "..\\..\\secret.png",
    "data/sandbox/../../secret.txt", "%USERPROFILE%\\Pictures\\x.png", "~/Pictures/x.png", "$HOME/x.png",
    "\\\\attacker\\share\\x.png",
])
def test_paths_outside_the_open_folders_are_blocked(pol, path):
    assert verdict(pol, path=path) is Verdict.BLOCK


def test_extension_check_is_case_insensitive_and_double_extensions_use_the_last(pol, shots):
    assert verdict(pol, path=str(shots / "SHOT.PNG")) is Verdict.ALLOW
    assert verdict(pol, path=str(shots / "photo.png.exe")) is Verdict.BLOCK


def test_invisible_characters_do_not_smuggle_a_type(pol, shots):
    assert verdict(pol, path=str(shots / "x.ex\u200be")) is Verdict.BLOCK


def test_symlink_out_of_the_folder_is_blocked(pol, shots, tmp_path):
    target = tmp_path / "outside.png"
    target.write_bytes(b"x")
    link = shots / "link.png"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks need privileges on this machine")
    assert verdict(pol, path=str(link)) is Verdict.BLOCK


# ---- the tool

@pytest.fixture
def env(memory, shots, tmp_path, monkeypatch):
    cfg = load_config()
    pol = Policy.load(ROOT / "config" / "policy.toml", places={"screenshots": shots, "skills": ROOT / "skills"})
    reg = build_tools(ToolContext(cfg, memory, Scheduler(memory, lambda _m: None), pol,
                                  now=lambda: dt.datetime(2026, 1, 2, 15, 4), notes_dir=tmp_path / "notes"))
    opened: list[str] = []
    monkeypatch.setattr("ev.tools.system.open_path", opened.append)
    return reg, pol, opened


def call(reg, **args):
    tool = reg.get("open_file")
    clean, err = validate_args(tool, args)
    assert clean is not None, err
    return reg.run(tool, clean)


def test_opens_the_last_screenshot(env, memory, shots):
    reg, _pol, opened = env
    shot = shots / "EV 2026-01-02 15-04-00.png"
    shot.write_bytes(b"png")
    memory.set_last_artifact(shot, "screenshot")
    r = call(reg, which="last")
    assert r.ok and r.message == "Opened the screenshot."
    assert opened == [str(shot)]


def test_nothing_made_yet(env):
    reg, _pol, opened = env
    r = call(reg, which="last")
    assert not r.ok and "haven't made anything" in r.message and not opened


def test_file_gone(env, memory, shots):
    reg, _pol, opened = env
    memory.set_last_artifact(shots / "gone.png", "screenshot")
    r = call(reg, which="last")
    assert not r.ok and "isn't there" in r.message and not opened


def test_tool_rechecks_the_file_it_resolved_itself(env, memory, tmp_path):
    """'last' carries no path for policy to judge, so a bad last_artifact must still be refused by the tool."""
    reg, _pol, opened = env
    evil = tmp_path / "evil.exe"
    evil.write_bytes(b"MZ")
    memory.set_last_artifact(evil, "file")
    r = call(reg, which="last")
    assert not r.ok and "can't open" in r.message and not opened


def test_explicit_path_through_the_gate(env, shots):
    reg, pol, opened = env
    gate = SafetyGate(pol, ask=lambda _r: False)
    ok = shots / "a.png"
    ok.write_bytes(b"x")
    assert gate.check("open_file", {"path": str(ok)}).allowed
    assert not gate.check("open_file", {"path": str(shots / "a.exe")}).allowed
    assert not opened                                    # checking never opens anything


def test_take_note_remembers_its_file_and_it_opens(env, memory, tmp_path):
    reg, pol, opened = env
    pol.open_dirs.append((tmp_path / "notes").resolve())    # stands in for data/notes, which is on the list in policy.toml
    tool = reg.get("take_note")
    assert reg.run(tool, {"text": "buy milk\nand eggs"}).ok
    assert memory.notes() == ["buy milk\nand eggs"]
    path, kind = memory.last_artifact()
    assert kind == "note" and Path(path).parent == tmp_path / "notes"
    assert "- 15:04 buy milk and eggs" in Path(path).read_text(encoding="utf-8")
    # the notes folder is on the open list, so "open it" works right after taking a note
    assert call(env[0], which="last").ok and opened == [path]
    reg.run(tool, {"text": "second"})
    assert Path(path).read_text(encoding="utf-8").count("\n- ") == 2     # appended to today's file, header written once
