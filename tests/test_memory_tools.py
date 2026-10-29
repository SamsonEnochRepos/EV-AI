import time

import pytest

from ev.tools.base import validate_args


def run(registry, name, **args):
    tool = registry.get(name)
    clean, err = validate_args(tool, args)
    assert clean is not None, err
    return registry.run(tool, clean)


def test_remember_update_forget(registry, memory):
    run(registry, "remember_fact", key="Favorite Color", value="green")
    assert memory.recall("favorite_color") == "green"
    run(registry, "remember_fact", key="favorite color", value="blue")
    assert memory.recall("favorite_color") == "blue"
    assert run(registry, "forget_fact", key="favorite color").ok
    assert memory.recall("favorite_color") is None
    assert not run(registry, "forget_fact", key="nothing").ok


def test_time_tool(registry):
    assert "3:04 PM" in run(registry, "get_time_date").message


def test_timer_fires(registry, scheduler, fired):
    r = run(registry, "set_timer", minutes=0, seconds=1)
    assert r.ok
    time.sleep(1.1)
    scheduler.poll_once()
    assert fired and "timer" in fired[0].lower()


def test_timer_requires_duration(registry):
    assert not run(registry, "set_timer").ok


def test_reminder_persists_and_fires(cfg, policy, memory, scheduler, fired):
    from ev.tools.builtin import ToolContext, build_tools

    registry = build_tools(ToolContext(cfg, memory, scheduler, policy))   # real clock
    assert run(registry, "set_reminder", text="call mom", in_minutes=1).ok
    assert len(memory.pending_reminders()) == 1
    memory.add_reminder("past", time.time() - 5)
    scheduler.poll_once()
    assert any("past" in f for f in fired)
    assert len(memory.pending_reminders()) == 1


def test_reminder_at_time(registry):
    assert run(registry, "set_reminder", text="x", at="4:30 pm").ok
    assert not run(registry, "set_reminder", text="x", at="banana").ok


def test_notes(registry, memory):
    run(registry, "take_note", text="buy milk")
    assert memory.notes() == ["buy milk"]


def test_validation_rejects_bad_input(registry):
    t = registry.get("set_timer")
    assert validate_args(t, {"minutes": "ten"})[0] is None
    assert validate_args(t, {"minutes": 10, "evil": 1})[0] is None
    assert validate_args(t, {"minutes": 99999})[0] is None
    assert validate_args(registry.get("media_control"), {"action": "explode"})[0] is None
    assert validate_args(registry.get("remember_fact"), {"key": "a"})[0] is None
    assert validate_args(t, "not a dict")[0] is None
    assert validate_args(t, {"minutes": True})[0] is None


def test_schema_cache(registry):
    assert registry.schemas() is registry.schemas()
    assert all(s["function"]["description"] for s in registry.schemas())


def test_summary_persists(memory):
    memory.set_kv("summary", "likes jazz")
    assert memory.snapshot()["summary"] == "likes jazz"


@pytest.mark.parametrize("bad", ["", "   ", "!!!"])
def test_empty_fact_rejected(memory, bad):
    with pytest.raises(ValueError):
        memory.remember(bad, "")


def test_forget_fuzzy_unique_match(memory):
    memory.remember("dog_name", "Biscuit")
    memory.remember("favorite_color", "green")
    assert memory.forget("my dog's name")
    assert memory.recall("dog_name") is None and memory.recall("favorite_color") == "green"


def test_forget_fuzzy_refuses_ambiguous(memory):
    memory.remember("dog_name", "Biscuit")
    memory.remember("cat_name", "Mochi")
    assert not memory.forget("name")          # matches two keys -> do nothing
    assert memory.recall("dog_name") and memory.recall("cat_name")


def test_web_search_pins_duckduckgo_backend(registry, monkeypatch):
    seen = {}

    class FakeDDGS:
        def __init__(self, **kw): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, query, **kw):
            seen.update(kw)
            return [{"title": "T", "body": "B"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    r = run(registry, "web_search", query="everest")
    assert r.ok and r.untrusted and seen["backend"] == "duckduckgo"


def test_whisper_loads_offline_only(cfg, monkeypatch):
    import sys
    import types

    from ev.stt.whisper import STT

    calls = {}

    class FakeModel:
        def __init__(self, name, **kw):
            calls.update(kw)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    STT(cfg).load()
    assert calls["local_files_only"] is True


def test_whisper_missing_model_gives_actionable_error(cfg, monkeypatch):
    import sys
    import types

    from ev.stt.whisper import STT

    def boom(*a, **k):
        raise OSError("not cached")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=boom))
    with pytest.raises(RuntimeError, match="download_models"):
        STT(cfg).load()
