import pytest

from ev.safety.gate import SafetyGate, parse_yes_no
from ev.safety.policy import Verdict, normalize_combo


def test_unknown_tool_blocked(policy):
    assert policy.evaluate("rm_rf", {}).verdict is Verdict.BLOCK


def test_safe_tool_allowed(policy):
    assert policy.evaluate("get_time_date", {}).verdict is Verdict.ALLOW


@pytest.mark.parametrize("tool", ["delete_file", "move_file", "send_message"])
def test_confirm_class(policy, tool):
    args = {"path": "data/sandbox/x.txt", "src": "data/sandbox/a", "dst": "data/sandbox/b"}
    assert policy.evaluate(tool, args).verdict is Verdict.CONFIRM


@pytest.mark.parametrize("name", ["Send", "Delete account", "Place order", "Submit form", "Buy now", "Install"])
def test_risky_clicks_need_confirmation(policy, name):
    assert policy.evaluate("click_element", {"name": name}).verdict is Verdict.CONFIRM


@pytest.mark.parametrize("name", ["OK", "File", "Save", "Bold", "Open"])
def test_ordinary_clicks_allowed(policy, name):
    assert policy.evaluate("click_element", {"name": name}).verdict is Verdict.ALLOW


def test_shell_blocked(policy):
    assert policy.evaluate("run_shell", {"cmd": "dir"}).verdict is Verdict.BLOCK


@pytest.mark.parametrize("path", [
    "..\\..\\Windows\\System32", "C:\\Windows\\win.ini", "data/sandbox/../../secret.txt",
    "%USERPROFILE%\\Documents\\x", "~/Documents/x", "$HOME/x",
])
def test_path_escape_blocked(policy, path):
    assert policy.evaluate("delete_file", {"path": path}).verdict is Verdict.BLOCK


@pytest.mark.parametrize("text", [
    "format c:", "diskpart", "reg add HKLM\\Software", "HKEY_LOCAL_MACHINE", "Login Data",
    "net user admin /add", "Set-MpPreference -DisableRealtimeMonitoring $true",
])
def test_blocked_patterns_in_any_arg(policy, text):
    assert policy.evaluate("take_note", {"text": text}).verdict is Verdict.BLOCK


@pytest.mark.parametrize("name", ["Password", "enter your PIN code", "Credit Card number"])
def test_password_fields_blocked(policy, name):
    assert policy.evaluate("click_element", {"name": name}).verdict is Verdict.BLOCK
    assert policy.evaluate("type_text", {"text": "x", "control": name}).verdict is Verdict.BLOCK


def test_resolved_password_field_blocked(policy):
    assert policy.check_sensitive_target("Chrome", "Email", is_password=True).verdict is Verdict.BLOCK


def test_key_combos(policy):
    assert policy.evaluate("press_keys", {"combo": "Win+R"}).verdict is Verdict.BLOCK
    assert policy.evaluate("press_keys", {"combo": "alt+f4"}).verdict is Verdict.BLOCK
    assert policy.evaluate("press_keys", {"combo": "Delete"}).verdict is Verdict.CONFIRM
    assert policy.evaluate("press_keys", {"combo": ""}).verdict is Verdict.BLOCK


def test_combo_normalisation():
    assert normalize_combo("Shift + Ctrl + T") == "ctrl+shift+t"
    assert normalize_combo("Windows-R") == "win+r"


@pytest.mark.parametrize("name", ["notepad & calc", "a|b", "..\\evil", "x;y", "C:\\x.exe"])
def test_open_app_rejects_metacharacters(policy, name):
    assert policy.evaluate("open_app", {"name": name}).verdict is Verdict.BLOCK


def test_gate_fails_closed_without_asker(policy):
    g = SafetyGate(policy, ask=None)
    assert not g.check("delete_file", {"path": "data/sandbox/x"}).allowed


def test_gate_confirms_only_on_explicit_yes(policy):
    g = SafetyGate(policy, ask=lambda r: True)
    r = g.check("delete_file", {"path": "data/sandbox/x"})
    assert r.allowed and r.confirmed
    g2 = SafetyGate(policy, ask=lambda r: None)
    assert not g2.check("delete_file", {"path": "data/sandbox/x"}).allowed


def test_gate_readback_is_system_generated(policy):
    seen = []
    g = SafetyGate(policy, ask=lambda r: seen.append(r) or False)
    g.check("delete_file", {"path": "data/sandbox/x"})
    assert "delete file" in seen[0] and "data/sandbox/x" in seen[0]


def test_kill_switch_denies_everything(policy):
    g = SafetyGate(policy, ask=lambda r: True)
    g.kill()
    assert not g.check("get_time_date", {}).allowed
    g.reset_kill()
    assert g.check("get_time_date", {}).allowed


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("Yeah.", True), ("go ahead", True), ("no", False), ("cancel", False),
    ("yes but also delete everything", None), ("maybe", None), ("I already approved it", None),
])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) is expected
