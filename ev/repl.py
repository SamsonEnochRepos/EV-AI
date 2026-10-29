"""Developer-only text REPL for testing the brain without audio. Not the product UI."""
from __future__ import annotations

import datetime as dt

from ev.core.config import Config
from ev.core.log import setup_logging
from ev.llm.brain import Brain
from ev.llm.client import OllamaClient
from ev.memory.store import MemoryStore
from ev.safety.gate import SafetyGate, parse_yes_no
from ev.safety.policy import Policy
from ev.tools.builtin import ToolContext, build_tools
from ev.tools.scheduler import Scheduler


def build_brain(cfg: Config, memory: MemoryStore, ask=None, model: str | None = None) -> tuple[Brain, Scheduler]:  # noqa: ANN001
    policy = Policy.load(cfg.path("general", "policy_file"))
    gate = SafetyGate(policy, ask=ask)
    sched = Scheduler(memory, lambda m: print(f"\n[alert] {m}"))
    registry = build_tools(ToolContext(cfg, memory, sched, policy))
    client = OllamaClient(cfg)
    if model:
        client.model = model
    return Brain(cfg, client, memory, registry, gate, now=dt.datetime.now), sched


def run_repl(cfg: Config, debug: bool = False) -> int:
    setup_logging(cfg.path("general", "log_dir"), debug)

    def ask(readback: str) -> bool | None:
        return parse_yes_no(input(f"[confirm] {readback}? (yes/no) "))

    memory = MemoryStore(cfg.data_path(cfg["memory"]["db_file"]))
    brain, sched = build_brain(cfg, memory, ask)
    sched.start()
    print("EV dev REPL. Ctrl+C or 'quit' to exit.")
    try:
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            res = brain.respond(text, on_sentence=lambda s: print("ev>", s))
            if debug:
                print(f"   [{res.timings} fast={res.fast_path} tools={[t['tool'] for t in res.tool_log]}]")
    except KeyboardInterrupt:
        pass
    finally:
        sched.stop()
        memory.close()
    return 0
