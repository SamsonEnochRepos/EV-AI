# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project skeleton for EV, a local voice-first assistant: `pyproject.toml`, `Makefile`, `start_ev.bat`, and the `ev/` package (core, audio, stt, tts, llm, tools, screen, control, memory, overlay, safety, evals).
- Config files: `config/ev.toml`, `config/persona.md`, `config/policy.toml`.
- Core: config loader, redacting file logger, event bus, state machine.
- LLM layer: loopback-only Ollama client, brain with a gated tool-call loop, regex fast path, sentence splitter, developer REPL.
- Safety: policy engine, confirmation gate with a kill switch, untrusted-data wrapper and injection detector.
- Tools: time/date, open app, timers, reminders, media, volume, web search, facts, notes, plus screen and control tools and a bounded screen agent.
- Audio and speech: VAD, mic recorder, faster-whisper wrapper, Piper TTS engine, speaker with barge-in, player with level metering.
- Overlay: Qt pill, token-authenticated loopback IPC, `--demo` mode, Tk pill for benchmarking.
- Eval harness: datasets, security, functional and benchmark runners.
- Unit tests and docs: README, ARCHITECTURE, DECISIONS, SECURITY, PROGRESS.
- STT eval (`ev/evals/stt_eval.py`) and overlay eval (`ev/evals/overlay_eval.py`).
- Evals: end-to-end, idle/startup, network egress, screen (real Notepad/Calculator), VLM benchmark, soak test, and `report.py` that generates `docs/EVAL_REPORT.md` from result JSONs.
- 15 integration tests of the real `EVApp` (turns, confirmation, kill switch under load, barge-in, Ollama down, no mic, TTS failure).
- `ev/llm/spoken.py` speech sanitiser; Kokoro TTS engine (optional); fuzzy `forget` for memory; fast-path patterns for louder/quieter, resume, copy/paste, forget.
- Docs: decisions D10-D16, README offline/Kokoro/troubleshooting sections.
- TTS benchmark (`ev/evals/tts_bench.py`: Piper vs Kokoro) and model checksums for Piper and Kokoro in `models/checksums.json`.
- Conda support: `environment.yml`, `start_ev_conda.bat`, `run_ev_debug_conda.bat` (verified by creating the `ev` env and starting EV from it).
- Models: pulled `qwen3:4b` and `moondream` in Ollama; downloaded the Piper `en_US-amy-medium` voice.
- `PROJECT_DOCUMENTATION.md`: full project documentation generated from the code (architecture diagrams, modules, tools, configuration, data model, evals, security model, limitations and suggested improvements).
- `CLAUDE.md` with commands, architecture overview, safety invariants and test fixtures for Claude Code.

### Changed
- Defaults from measurements: STT `base` (RTF 0.23 vs 0.65), VLM `qwen2.5vl:3b` (5/5 vs 2/5), VAD silence window 550 ms, persona tool guidance (tool accuracy 81.8% -> 92.4%).
- Runtime is offline for models (`local_files_only`, `HF_HUB_OFFLINE`); `web_search` pinned to the DuckDuckGo backend.
- Security eval suite: 88/88 deterministic checks pass, 0 confirmation bypasses. 123 unit tests pass.
- Ruff line length set to 120; E501 and UP042 ignored.
- Recorded the LLM speed/VRAM benchmark in `docs/DECISIONS.md` (D4): qwen3:4b rejected (27.6% spills to RAM, 28 tok/s); qwen2.5:3b and llama3.2 fit fully in VRAM at about 77-80 tok/s.
- Control tools (click, type, press keys, paste) are `safe` at tool level; risky targets and key combos escalate to confirmation by argument.

### Fixed
- Overlay client deadlocked at startup (re-entrant lock); overlay teardown now kills the whole process tree.
- Silero VAD was driven through faster-whisper's batch API and crashed every turn; now a streaming ONNX session with a reset per utterance.
- EV contacted Hugging Face at idle and `web_search` queried several engines; both removed (egress audit: loopback only at idle).
- Fast path no longer treats "delete my X" or chatty "open up..." as commands.
- Recorder counted pre-roll as speech, so tiny blips passed the minimum-length check.
- `policy.toml` had invalid TOML (Python-style raw string), breaking every policy-dependent test.
- Fast path matched conversational phrases such as "open up to me" as `open_app`; added a stoplist.
- MemoryStore reads are now locked and safe after close; scheduler joins its thread on stop (fixed an access-violation crash).
- Overlay server left rejected sockets open (closed the makefile wrapper too).
- `focus_window` now checks sensitive-window patterns; `open_app` rejects newlines, argument injection and executable extensions.
- Added injection phrases ("note to AI", "always trust", tool names in data); removed a stray backspace character from a regex.
- Policy now NFKC-normalises and strips invisible characters before matching, closing a zero-width-character bypass.

### Removed
- Nothing.
