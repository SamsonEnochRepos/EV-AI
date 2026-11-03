# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EV: a fully local, voice-first desktop assistant for Windows (Python 3.11/3.12). Speech → faster-whisper (CPU) → Ollama LLM on `127.0.0.1` (default `qwen2.5:3b`, fallback `llama3.2`) → tools → Piper/Kokoro TTS. The only UI is a small overlay "pill" plus a tray icon. No cloud, no telemetry; the only runtime egress is the optional DuckDuckGo search.

## Commands

The environment is a conda env named `ev` (the owner uses conda only; `.venv` is an unused leftover). The Makefile still assumes `.venv/Scripts/python.exe`, so run the underlying commands with the env's Python: `conda activate ev` and plain `python`, or `& "$env:USERPROFILE\miniconda3\envs\ev\python.exe"` when `conda` isn't on PATH.

```powershell
make check                                      # ruff + mypy + pytest
make lint                                       # ruff check ev tests
make type                                       # mypy ev
make unit                                       # pytest -m "not integration"
make integration                                # pytest -m integration (mocked hardware)
python -m ruff check ev tests; python -m mypy ev; python -m pytest -q   # what `make check` does, with the conda env
python -m pytest tests/test_policy.py::test_name -q   # single test

python -m ev --debug              # run the assistant with console logs
python -m ev --repl --debug       # developer text REPL (no audio)
python -m ev.overlay --demo       # preview the pill only

python -m ev.evals.run security   # red-team suite
python -m ev.evals.run all        # choices: security, security-nollm, intent, persona, memory, all (needs Ollama)
```

Pytest `addopts` excludes the `eval` and `hardware` markers by default. Ruff runs at line length 120 with rules `E,F,I,B,S,UP`; bandit rules (`S`) are disabled for `tests/` and `ev/evals/`.

## Architecture

The pipeline is described in `docs/ARCHITECTURE.md`. Read that first, plus `docs/SECURITY.md` and `docs/DECISIONS.md` (numbered decisions such as D4/D15 that justify model and engine choices).

- **Orchestration**: `ev/app.py` wires everything together. `core/state.py` is the state machine (IDLE → LISTENING → TRANSCRIBING → THINKING → ACTING → SPEAKING, plus ERROR/PAUSED). It publishes on the event bus in `core/events.py`, and the overlay client and wake-word logic subscribe to it.
- **Brain**: `llm/brain.py` streams Ollama `/api/chat` with tools and splits the output into sentences incrementally (`llm/sentences.py`) so TTS starts early. `llm/fastpath.py` handles common intents (time, timers, open app, volume, media, reminders, notes) with regexes and skips the LLM.
- **Tools**: `tools/base.py` defines `Tool`/`ToolResult`/`ToolRegistry` with strict arg validation and per-tool timeouts. Built-in tools are assembled in `tools/builtin.py::build_tools(ToolContext)`. Screen tools live in `screen/` (UIA → RapidOCR → VLM, in that order) and mouse/keyboard tools in `control/`. `control/agent.py` is the bounded observe→act→verify `ScreenAgent`.
- **Safety (an invariant, don't break it)**: every tool call goes through `safety/gate.py::SafetyGate` → `safety/policy.py::Policy`. That includes LLM calls, fast-path calls and screen-agent calls. Policy decisions depend only on `config/policy.toml` and the call arguments, never on model or screen text. Confirmation is a separate system-driven readback, and with no asker it fails closed. `unknown_tool = "blocked"`, so **a new tool must be given a level in `config/policy.toml` `[tools]`** or it will be blocked. Results from external text sources (web, screen, files) must set `ToolResult.untrusted=True` so `safety/untrusted.py` wraps them as data.
- **Overlay**: the "island" (a persistent pill with cursor-following eyes that springs open for speech and grows to hold display-mode text; maths in `overlay/anim.py`) runs as a separate process (`overlay/qt_app.py` uses PySide6, `tk_app.py` is the fallback). It speaks newline-delimited JSON over loopback with a per-session token passed on stdin (`overlay/protocol.py`).
- **Memory**: facts, people, projects and daily summaries are Markdown files in a vault folder (`memory/vault.py`, `[vault]` in `ev.toml`, default `~/Documents/EV-Vault/EV/`); EV writes only there and searches the whole vault through a rebuildable SQLite FTS5 index (`memory/vault_index.py`). SQLite (`memory/store.py`) holds reminders, notes, kv and `fact_records`. **A line of `Facts.md` is trusted only if its SHA-256 matches a record EV wrote; the `source:` label in the file is never believed** (a hand edit or forged line is an "unconfirmed fact": an ordinary note until the user confirms it by voice via `keep_fact`). Every fact write goes through `Vault.add_fact` (sanitising, caps, injection-phrase refusal). There is no vector DB. What Evie knows about the user is seeded once from `config/seed_facts.toml` (`memory/migrations.py`); `persona.md` holds character and rules only. `kv.last_artifact` is the last screenshot or note EV made, for `open_file`.
- **Memory provenance**: a `remember_fact` / `forget_fact` that isn't plainly the user's own words (not the fast path, little overlap with their last three utterances, or any untrusted tool result or recalled note this turn) is escalated to a spoken yes by `safety/provenance.py` -> `SafetyGate.check(..., escalate=)`. Escalation can only add a confirmation, never relax anything. Voice-only tools set `Tool.llm_visible = False` and a model-made call to one is refused. Don't add a tool that writes user-visible memory without going through this.
- **Skills**: `skills/*.toml` are routines of ordinary tool calls (`skills/store.py`, `skills/runner.py`, run by `Brain._execute` with `source="skill"` so **each step is gated on its own**). A skill step can never be `run_skill`/`save_skill`/`manage_skill`/`keep_fact`/`build_task`/`remember_fact`/`forget_fact`. `save_skill` and `manage_skill` are voice-only (not in the model's tool list); a model-made `run_skill` for something the user didn't mention is refused. Don't add a path that runs skill steps without `_execute`.
- **Proactive**: `proactive/engine.py` + `config/proactive.toml`. EV only volunteers things when idle and within the guards; spoken lines are system text from a rule's template, never model output (except `Brain.phrase_for_speech`, one tool-less call over wrapped data). A skill run by a rule uses `SafetyGate.check(ask_allowed=False)`: it must never ask a question out of the blue. Slow probes belong on the engine thread, outside its lock.
- **Context**: `context/snapshot.py` appends a short wrapped block (foreground app, next event, recent files, battery, git) to the latest user message on the model path; it must stay out of the system prompt (D21) and must not do slow work on the request path (the process walk is done once at start-up; the calendar refreshes in the background).
- **Dashboard**: `dashboard/` is a stdlib loopback server (`server.py` transport + auth, `backend.py` what it may ask of EV, `settings_schema.py` the allow-list of editable keys, `hub.py` events, `static/` the page). Settings are an allow-list written only to `config/ev.local.toml`; anything that widens access (paths, hosts, builder, vault, privacy, every policy value) stays file-only, and `tests/test_dashboard.py` walks `ev.toml`/`policy.toml` to enforce it. Never add a route that writes policy or that acts without going through `SafetyGate` (run a skill = `Brain.run_skill_for_user`; forget/keep a fact = `Brain._execute(..., source="dashboard")`). The page must set text with `textContent`, never `innerHTML`.
- **Prompt budget**: `python -m pytest tests/test_integration.py -k real_prompt` guards the real tool list. New tools and persona lines cost context (measure with Ollama's `prompt_eval_count`); `Brain._est_tokens` is calibrated to 4.0 chars per token from that measurement.
- **Evals**: `ev/evals/` holds benchmark and eval suites driven by `ev.evals.run`. Results go to `evals/results/`.

## Config

- `config/ev.toml` holds the main settings. `config/ev.local.toml` (git-ignored) is deep-merged over it for per-machine overrides (`core/config.py`).
- `config/persona.md` is the system persona, read at startup. After editing it, run `python -m ev.evals.run persona`.
- `config/policy.toml` holds tool permission levels (safe/confirm/blocked), the app allowlist, safe folders and blocked patterns.

## Testing notes

- `tests/conftest.py` provides `cfg`, `policy`, `memory` (tmp SQLite), `scheduler`, `gate` (always declines confirmations) and `registry` (fixed clock `2026-01-02 15:04`) fixtures.
- `tests/fakes.py::FakeOllama` is an `httpx` transport that replays scripted ndjson replies (`text_reply`, `tool_reply`). Use it instead of a real Ollama in unit and integration tests.
