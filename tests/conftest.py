from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from ev.core.config import ROOT, load_config
from ev.memory.store import MemoryStore
from ev.safety.gate import SafetyGate
from ev.safety.policy import Policy
from ev.tools.base import ToolRegistry
from ev.tools.builtin import ToolContext, build_tools
from ev.tools.scheduler import Scheduler


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture(autouse=True)
def _no_lookup_planner(request, monkeypatch):
    """The planner is one extra model call before the main one; tests that script Ollama replies by position don't expect it.
    A test that wants it asks for the `planner_on` fixture."""
    if "planner_on" not in request.fixturenames:
        monkeypatch.setattr("ev.llm.brain.Brain._plan_lookup", lambda self, text: None)


@pytest.fixture
def planner_on():
    return True


@pytest.fixture
def policy():
    return Policy.load(ROOT / "config" / "policy.toml")


@pytest.fixture
def memory(tmp_path: Path):
    m = MemoryStore(tmp_path / "t.sqlite")
    yield m
    m.close()


@pytest.fixture
def fired():
    return []


@pytest.fixture
def scheduler(memory, fired):
    return Scheduler(memory, fired.append)


@pytest.fixture
def gate(policy):
    return SafetyGate(policy, ask=lambda _r: False)


@pytest.fixture
def registry(cfg, memory, scheduler, policy, tmp_path: Path) -> ToolRegistry:
    ctx = ToolContext(cfg, memory, scheduler, policy, now=lambda: dt.datetime(2026, 1, 2, 15, 4),
                      notes_dir=tmp_path / "notes")
    return build_tools(ctx)
