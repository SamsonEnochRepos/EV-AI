"""The fine-tuning export is opt-in, redacts what it writes and never trains on turns that quoted outside text."""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from ev.core.config import ROOT


@pytest.fixture(scope="module")
def export():
    spec = importlib.util.spec_from_file_location("export_data", ROOT / "scripts" / "finetune" / "export_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_redaction_removes_contact_details_urls_and_user_folders(export):
    said = "mail bob@example.com or call +91 98765 43210, see https://x.example/a?b=1 in C:\\Users\\samso\\Desktop, id 123456789012"
    out = export.clean(said)
    for leaked in ("bob@example.com", "98765", "https://", "samso", "123456789012"):
        assert leaked not in out
    assert "<email>" in out and "<url>" in out


def test_untrusted_failed_unkept_and_broken_turns_are_dropped(export, tmp_path):
    rows = [{"user": "what time is it", "assistant": "", "tools": [{"name": "get_time_date", "args": {}, "ok": True}]},
            {"user": "search the web for cats", "assistant": "ignore previous", "untrusted": True},
            {"user": "open spotify please", "tools": [{"name": "open_app", "args": {"name": "spotify"}, "ok": False}]},
            {"user": "tell me a joke", "assistant": "Why did the chicken cross?", "keep": False},
            {"user": "hey there", "assistant": "Hey! Good to hear you."}]
    f = tmp_path / "turns.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    out, skipped = export.from_turns(f)
    assert len(out) == 2 and skipped == {"untrusted": 1, "not_kept": 1, "failed_tool": 1, "bad_row": 1, "too_short": 0}
    assert out[0]["messages"][-1]["tool_calls"][0]["function"]["name"] == "get_time_date"


def test_a_tool_turn_trains_on_the_call_not_on_the_result(export):
    ex = export.example("set a timer for ten minutes", "Timer set for 10 minutes.", [{"name": "set_timer", "args": {"minutes": 10}}])
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"] and ex["messages"][-1]["content"] == ""


def test_without_yes_nothing_is_written(export, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["export_data.py", "--turns", str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "out")])
    assert export.main() == 0
    assert not (tmp_path / "out").exists() and "dry run" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["export_data.py", "--turns", str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "out"), "--yes"])
    assert export.main() == 0
    train = (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) > 20 and all(json.loads(line)["messages"][0]["role"] == "system" for line in train)
