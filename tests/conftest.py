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
def registry(cfg, memory, scheduler, policy) -> ToolRegistry:
    ctx = ToolContext(cfg, memory, scheduler, policy, now=lambda: dt.datetime(2026, 1, 2, 15, 4))
    return build_tools(ctx)
