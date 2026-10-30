# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `docs/PROJECT_DOCUMENTATION.md`: full project documentation (architecture, modules, tools, safety model, setup for the conda env, limitations) including the trained "Hey Evie" wake-word results from `scripts/wakeword/trained_output/`.
- **Wake-word training kit** (`scripts/wakeword/`) for a custom openWakeWord model, `hey_evie`, that answers to "hey evie", "hey EV", "evie" and "EV". Piper speech (EV's `hfc_female` voice is the largest share, plus five other voices) with noise/reverb/EQ augmentation, near-miss and real-speech negatives (LibriSpeech, ESC-50), a small FCN head trained with torch and exported to ONNX, and a streaming evaluator (recall on unseen voices, false triggers/hour). `train_cloud.ipynb` runs it all on Colab/Kaggle; `record.py` records your own voice. See `scripts/wakeword/README.md`. The model itself is not trained yet.
- `scripts/download_models.py` can fetch three more Piper voices (`ryan`, `libritts_r`, `alan`), used only for wake-word training data.

### Fixed
- Wake-word training: the near-miss list no longer contains spellings of the wake word itself ("eevee", "Eevie", "eevee evolution", "hi Evie's friend"), which labelled the same sound both positive and negative.

### Removed
- Project cleanup: one-off install/pull logs and leftovers in `scripts/`, `docs/PROGRESS.md` (stale Phase 0 notes) and the generated `PROJECT_DOCUMENTATION.md`. Dangling doc references were fixed.

### Changed
- `config/ev.toml`: `tts.engine` is now `"piper"` (`en_US-hfc_female-medium`) instead of `"kokoro"`; Kokoro stays available as an option.

### Added
- **Display mode**: say "activate display mode" (also "enable/turn on ... display/caption/subtitle mode") and the pill grows a text card showing what you said and EV's reply as it is spoken, sentence by sentence; "deactivate display mode" puts it away. The mode is remembered across restarts (`overlay.display_mode` sets the default), the card stays `overlay.display_linger_s` after EV finishes, and long replies scroll so the newest words stay visible. Voice command is a fast-path match to the new `set_display_mode` tool. Overlay protocol gained `display` and `text` messages; the wire format is now UTF-8 instead of `\uXXXX` escapes so curly quotes and non-English text fit the line limit. `python -m ev.overlay --demo --display` previews it.
- `ev.stt.whisper.clean_audio`, `Speaker(pause_ms=...)`, `make_engine(cfg, kind)` and tests for the reply cap and audio cleanup (`tests/test_stt_clean.py`, `tests/test_brain.py`).
- Decision D17 in `docs/DECISIONS.md` recording the STT/TTS/reply-length changes and what was and wasn't measured.

### Added
- **Close apps by voice**: `close_app` ("close Notepad", "close this window") asks the app's windows to close like the X button, so unsaved work still gets a save prompt. "Force quit" / "kill" needs a spoken yes. Windows services, Defender, Ollama, EV itself and the terminal it runs in are never touched.
- **Websites**: `open_website` opens a site or searches it in the default browser or a named one ("open Firefox, open YouTube and search for the latest tech video" is one fast-path call; "latest" sorts YouTube by upload date). Only public https sites; any address that isn't a built-in site or in `[web] trusted_hosts` (policy.toml) asks first, since data can leave in a hostname or path.
- **Screenshots**: `take_screenshot` saves a full-resolution PNG to `Pictures\EV Screenshots` (`screen.screenshot_dir`), refusing if the active window looks sensitive.
- **Administrator rights always ask**: `open_app admin=true` and any admin/elevate/run-as wording in app, website, window or screen-agent requests need a spoken yes (`[confirm_patterns]`); EV never elevates itself, Windows' UAC prompt still appears.
- The screen agent can use `open_website` and `close_app`; confirmations read back what will happen ("force close chrome. Anything unsaved in it will be lost").

### Changed
- Config: `hotwords` gain "YouTube" and "display mode"; `interrupt_margin` 2.0 -> 1.5. Added `tests/test_interrupt.py` and the recorded-levels fixture `tests/data/echo_laptop_speakers.json` (levels only, no audio).
- Overlay wire format is UTF-8 (`ensure_ascii=False`); `Pill` now sizes itself from its content and positions the pill by anchor inside a wider window. Tests added for display mode (`tests/test_display_mode.py`, plus integration turns in `tests/test_integration.py`).
- `read_screen` hint now asks for one short casual sentence instead of one or two.
- Speech recognition: default model is now `distil-small.en` (beam 3, 6 threads) instead of `base`; input audio is DC/rumble filtered and level-normalised before decoding, Whisper gets `hotwords` for EV's common vocabulary and a temperature fallback, and the pre-speech buffer grew from 300 to 480 ms (`audio.pre_roll_ms`). Costs about 1.3 s more per turn than `base`; set `stt.model = "base"` to trade back.
- Voice: default TTS engine is Kokoro (`af_heart`, speed 1.05) with a 90 ms pause between sentences (`tts.sentence_pause_ms`); EV falls back to Piper if Kokoro can't load.
- Replies: `llm.reply_max_sentences = 2` is now enforced in `Brain` (generation stops once reached; 6 when the user asks for detail), `num_predict` dropped from 220 to 110, and the persona's speaking rules were rewritten for a casual, answer-first voice.

### Fixed
- EV interrupting itself, again (`user talked over EV (echo ratio 0.01)`). Measured on a laptop: EV's voice reaches its own mic about 0.5 s after the playback meter shows it, and its echo is as loud as normal speech, so the detector's "mic vs playback right now" comparison took EV's own voice for the user (2 of 3 recordings fired with nobody talking). The detector now learns the speaker-to-mic delay itself, predicts echo from what was playing that long ago, stays disarmed for the first 1.2 s of a reply and never learns the user's voice as echo. 0 false interrupts on four real recordings; `conversation.interrupt_margin` is now 1.5. Tests use the recorded levels (`tests/data/echo_laptop_speakers.json`).
- Whisper mishearings of names EV hears often ("Open Fire Fox", "open you due" for YouTube) are corrected after transcription, and sentence breaks in a multi-command utterance ("Open Firefox? Open YouTube and search...") no longer stop the fast path from matching.
- EV heard its own voice and answered it: the voice-interrupt detector calibrated the speaker echo before the echo reached the mic (ratio ~0), so EV's own speech counted as the user talking over it and was transcribed as the next question. Calibration now spans ~600 ms (75th percentile) and the expected echo uses the loudest playback of the last ~360 ms, which also covers the gaps between sentences.
- Tool calling for app launches: "open **up** Claude Desktop", "... in my PC" and "... for me" now reach `open_app`; Store apps (Claude, etc.) resolve, Whisper mishearings like "Cloud" -> "Claude" are corrected as a second guess, an exact name always beats a loose match (no more "cloud" -> Genshin Impact Cloud), and fuzzy matching no longer turns "photoshop" into Photos.
- EV no longer speaks sentences that name internal tools ("I can use the web_search tool..."), and the persona stops it claiming it can't open apps or see the screen.
- Web search: results are cached for ten minutes and retried with longer spacing when DuckDuckGo serves its bot check.
- Wake word never started: `trigger.mode` was `hotkey` and `openwakeword` was not installed. The listener now loads its models offline from `models/openwakeword/` (fetch with `python scripts/download_models.py --wakeword hey_jarvis`), releases the mic before the turn starts, stays paused for the whole turn, and degrades to hotkey-only with a clear log line if the model is missing.
- Prompt overflow: persona + 20 tool schemas used ~1900 of `num_ctx = 2048` tokens, so Ollama silently truncated the system prompt/tools on later turns (lost context, tools stopped being called). `num_ctx` is now 4096 and `Brain` trims history to the context budget and honours `llm.history_turns`.
- `open_app` failed with `WinError 2` for Chrome/Firefox/Edge (registered in App Paths, not on PATH). Executables are now resolved via PATH, App Paths, Start-menu shortcuts (fuzzy, never uninstallers) and Store apps; "for me"/"browser" suffixes are trimmed.
- Web search: DuckDuckGo's intermittent bot-check ("No results found") is retried, falls back to news, and reports honestly when the service is unavailable. Explicit searches and weather requests call `web_search` deterministically, and the result carries a system-authored hint that stops the model inventing details.
- GPU out-of-memory from Ollama now retries on CPU for five minutes, and EV says it is out of memory instead of "something went wrong".

### Changed
- Evie's voice is now the Piper voice `en_US-hfc_female-medium` (`tts.piper_voice` in `config/ev.toml`); the voice is also listed in `scripts/download_models.py`. Its training data is CC BY-NC-SA 4.0 (non-commercial).

### Added
- `get_weather` tool (`ev/tools/weather.py`): live conditions and tomorrow's forecast from Open-Meteo, answered directly without the LLM. DuckDuckGo snippets only describe weather sites, so web search could never answer "what's the weather". Sends only the city name; switch off with `[web] weather_enabled = false`. A saved `city`/`location` fact is used when no place is spoken.
- Fast-path routing for screen questions ("can you see my screen", "what's on my screen") to `read_screen`.
- Voice barge-in (`ev/audio/interrupt.py`): talk over EV while it speaks and it stops mid-sentence, stops generating, and listens to what you say next. An echo-aware detector learns how loud EV's own voice leaks into the mic so speakers don't self-interrupt; tune with `voice_interrupt`, `interrupt_min_ms`, `interrupt_margin`, `interrupt_floor` in `[conversation]`. EV remembers only the part of the reply it actually said.
- Follow-up listening (`[conversation]` in `config/ev.toml`): after EV answers it keeps listening without a wake word or hotkey, for longer if its reply was a question. "Thanks", "that's all", "never mind" end the conversation.
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
