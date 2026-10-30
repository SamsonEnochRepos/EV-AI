# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EV: a fully local, voice-first desktop assistant for Windows (Python 3.11/3.12). Speech → faster-whisper (CPU) → Ollama LLM on `127.0.0.1` (default `qwen2.5:3b`, fallback `llama3.2`) → tools → Piper/Kokoro TTS. The only UI is a small overlay "pill" plus a tray icon. No cloud, no telemetry; the only runtime egress is the optional DuckDuckGo search.

## Commands

The environment is a conda env named `ev` (the owner uses conda only; `.venv` is an unused alternative). The Makefile assumes `.venv/Scripts/python.exe`.

```powershell
make check                                      # ruff + mypy + pytest
make lint                                       # ruff check ev tests
make type                                       # mypy ev
make unit                                       # pytest -m "not integration"
make integration                                # pytest -m integration (mocked hardware)
.venv\Scripts\python -m pytest tests/test_policy.py::test_name -q   # single test

.venv\Scripts\python -m ev --debug              # run the assistant with console logs
.venv\Scripts\python -m ev --repl --debug       # developer text REPL (no audio)
.venv\Scripts\python -m ev.overlay --demo       # preview the pill only

.venv\Scripts\python -m ev.evals.run security   # red-team suite
.venv\Scripts\python -m ev.evals.run all        # choices: security, security-nollm, intent, persona, memory, all (needs Ollama)
```

Pytest `addopts` excludes the `eval` and `hardware` markers by default. Ruff runs at line length 120 with rules `E,F,I,B,S,UP`; bandit rules (`S`) are disabled for `tests/` and `ev/evals/`.

## Architecture

The pipeline is described in `docs/ARCHITECTURE.md`. Read that first, plus `docs/SECURITY.md` and `docs/DECISIONS.md` (numbered decisions such as D4/D15 that justify model and engine choices).

- **Orchestration**: `ev/app.py` wires everything together. `core/state.py` is the state machine (IDLE → LISTENING → TRANSCRIBING → THINKING → ACTING → SPEAKING, plus ERROR/PAUSED). It publishes on the event bus in `core/events.py`, and the overlay client and wake-word logic subscribe to it.
- **Brain**: `llm/brain.py` streams Ollama `/api/chat` with tools and splits the output into sentences incrementally (`llm/sentences.py`) so TTS starts early. `llm/fastpath.py` handles common intents (time, timers, open app, volume, media, reminders, notes) with regexes and skips the LLM.
- **Tools**: `tools/base.py` defines `Tool`/`ToolResult`/`ToolRegistry` with strict arg validation and per-tool timeouts. Built-in tools are assembled in `tools/builtin.py::build_tools(ToolContext)`. Screen tools live in `screen/` (UIA → RapidOCR → VLM, in that order) and mouse/keyboard tools in `control/`. `control/agent.py` is the bounded observe→act→verify `ScreenAgent`.
- **Safety (an invariant, don't break it)**: every tool call goes through `safety/gate.py::SafetyGate` → `safety/policy.py::Policy`. That includes LLM calls, fast-path calls and screen-agent calls. Policy decisions depend only on `config/policy.toml` and the call arguments, never on model or screen text. Confirmation is a separate system-driven readback, and with no asker it fails closed. `unknown_tool = "blocked"`, so **a new tool must be given a level in `config/policy.toml` `[tools]`** or it will be blocked. Results from external text sources (web, screen, files) must set `ToolResult.untrusted=True` so `safety/untrusted.py` wraps them as data.
- **Overlay**: runs as a separate process (`overlay/qt_app.py` uses PySide6, `tk_app.py` is the fallback). It speaks newline-delimited JSON over loopback with a per-session token passed on stdin (`overlay/protocol.py`).
- **Memory**: SQLite in `memory/store.py` (facts, notes, reminders, kv). There is no vector DB.
- **Evals**: `ev/evals/` holds benchmark and eval suites driven by `ev.evals.run`. Results go to `evals/results/`.

## Config

- `config/ev.toml` holds the main settings. `config/ev.local.toml` (git-ignored) is deep-merged over it for per-machine overrides (`core/config.py`).
- `config/persona.md` is the system persona, read at startup. After editing it, run `python -m ev.evals.run persona`.
- `config/policy.toml` holds tool permission levels (safe/confirm/blocked), the app allowlist, safe folders and blocked patterns.

## Testing notes

- `tests/conftest.py` provides `cfg`, `policy`, `memory` (tmp SQLite), `scheduler`, `gate` (always declines confirmations) and `registry` (fixed clock `2026-01-02 15:04`) fixtures.
- `tests/fakes.py::FakeOllama` is an `httpx` transport that replays scripted ndjson replies (`text_reply`, `tool_reply`). Use it instead of a real Ollama in unit and integration tests.
