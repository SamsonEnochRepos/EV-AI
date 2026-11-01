# Decisions

Each entry: what was decided, why, and what evidence backs it. Entries marked **(pending)** get
their numbers filled in once the benchmark has actually been run on this machine.

## D1. Python 3.12 via uv, not the system Python 3.14
The machine has only Python 3.14. `ctranslate2` (faster-whisper), `onnxruntime`, `piper-tts` and
`PySide6` ship wheels for 3.11/3.12 first; 3.14 wheels are not reliably available. `uv` was
installed with `pip install --user uv` and provisioned CPython 3.12.16 into `.venv`.

## D2. Real OS is Windows 10 Home (build 19045), not Windows 11
The brief says Windows 11. `Get-CimInstance Win32_OperatingSystem` reports Windows 10 Home.
Impact: none of the chosen components need Win11. The overlay uses Win32 extended window styles
that exist on both. Win11-only effects (Mica, rounded-corner DWM attributes) are not used.

## D3. Hotkey defaults
`Ctrl+Shift+Esc` opens Task Manager, so it is not used. Defaults (all in `config/ev.toml`):
push-to-talk `Ctrl+Alt+Space`, kill switch `Ctrl+Alt+Shift+K`, confirm yes `Ctrl+Alt+Y`,
confirm no `Ctrl+Alt+N`.

## D4. Ollama model candidates
Already installed: `llama3.2` (3B), `qwen2.5:3b`, `qwen2.5:7b`. Pulled for the benchmark:
`qwen3:4b` (thinking disabled via `think=false`), `moondream` (VLM). `qwen2.5:7b` is excluded
up front: 4.7 GB cannot fit in 4 GB VRAM.

Speed/VRAM benchmark (`python -m ev.evals.bench llm`, num_ctx 2048, 3 prompts, raw data in
`evals/results/bench_llm.json`):

| Model | tok/s (median) | Model size | In VRAM | Spill to RAM |
|---|---|---|---|---|
| qwen2.5:3b | 79.6 | 2.09 GB | 2.09 GB | 0% |
| llama3.2 (3B) | 76.9 | 2.32 GB | 2.32 GB | 0% |
| qwen3:4b | 28.5 | 3.23 GB | 2.34 GB | **27.6%** |

`qwen3:4b` fails the "no spill into system RAM" requirement at 2048 context and is about 3x slower,
so it is rejected for the voice path. The peak-VRAM column in the raw JSON is not reliable for the
second and third models because the previous model had not yet unloaded when the baseline was taken.
Both remaining candidates meet the 20 tok/s budget by a wide margin. The tool-accuracy and persona
comparison between qwen2.5:3b and llama3.2 is still **pending** (needs the venv install to finish).
Working default: `qwen2.5:3b`, fallback `llama3.2`.

## D5. Overlay toolkit
Candidates: PySide6 (Qt) vs Tkinter canvas. Tk cannot do per-pixel alpha or anti-aliased edges on
a transparent window without a colour-key hack (jagged edges). Qt does both natively. Final
choice: **PySide6 (Qt)**. Measured by `python -m ev.evals.bench overlay` (12 s of the `--demo` cycle,
whole process tree, raw data in `evals/results/bench_overlay.json`):

| Toolkit | CPU (mean, % of all cores) | CPU max | RSS | Edges |
|---|---|---|---|---|
| Qt | 0.01% | 0.26% | 52 MB | anti-aliased, per-pixel alpha |
| Tk canvas | 0.14% | 0.52% | 34 MB | aliased (colour-key transparency) |

Both meet the animating-CPU (< 1-2%) budget. Qt is 52 MB, inside the 60 MB budget but with less headroom.
Tk is lighter on RAM but cannot render smooth translucent rounded ends, so it is demo/benchmark-only.
Caveat: the 12 s sample covers part of the demo cycle including its hidden phase, so mean CPU is a blend.
Note: on this machine `.venv\Scripts\python.exe` is a launcher shim that spawns the real interpreter, so the
overlay is a process *tree*; the client kills the whole tree and benchmarks sum over it.

## D6. Overlay IPC
Overlay process is the TCP server bound to `127.0.0.1` on an ephemeral port. A random per-session
token is passed to it over stdin (not argv, which other processes can read) and must be the first
line a client sends (constant-time compare, 2 s deadline). Non-loopback peers are dropped. When the
authenticated core connection closes, the overlay exits; if the overlay dies, the core swallows the
error and restarts it with a cooldown.

## D7. Control tools are `safe` at tool level, with argument-level escalation
Per-click confirmation would make screen control unusable. Instead the policy engine escalates
based on arguments: clicking a control named like send/submit/delete/buy/pay/install... and
pressing Enter/Delete/Ctrl+W need spoken confirmation; password-like targets and sensitive windows
are blocked, and the control layer re-checks the *resolved* UIA element (including the
`IsPassword` property) before acting. Residual risk: Enter inside a web form is confirmed
conservatively, which costs one extra "yes" in normal typing flows. Edit `confirm_key_combos`
in `policy.toml` to relax.

## D8. Wake word
openWakeWord has no pretrained "Hey EV". Wake word is an optional, pluggable listener
(`trigger.mode = "wakeword"`, `wakeword_model`); it started with the placeholder `hey_jarvis`. A custom `hey_evie`
model was then trained with `scripts/wakeword/` and is now the configured default; the push-to-talk hotkey still
works in every mode. See `docs/PROJECT_DOCUMENTATION.md` for the training results and what was not measured. Models live in `models/openwakeword/` and are
fetched once by `scripts/download_models.py --wakeword`; nothing is downloaded at runtime. The push-to-talk
hotkey keeps working in wake-word mode.

## D9. Vector memory
SQLite facts + rolling summary only. ChromaDB would cost hundreds of MB of RAM on an 8 GB machine.

## D10. STT model: `base` (int8, CPU)
`python -m ev.evals.stt_eval`: 30 Piper-synthesized utterances (commands, numbers, app names, chat), clean and
with white noise at 10 dB SNR (raw data `evals/results/stt.json`).

| Model | WER clean | WER 10 dB | RTF median | RTF p95 |
|---|---|---|---|---|
| base | 8.6% | 8.6% | 0.23 | 0.33 |
| small | 9.0% | 8.2% | 0.65 | 0.93 |

`small` misses the RTF < 0.5 budget on this CPU and is not more accurate here, so `base` is the default
(`stt.model` in `ev.toml`). **Caveat:** the test audio is synthetic speech from one voice, so absolute WER is
optimistic versus real human speech, accents and room noise. No real-speaker recordings were available to me.

## D11. Vision fallback model: `qwen2.5vl:3b` (replaces moondream as default)
`python -m ev.evals.vlm_bench`, 5 UI-reading tasks on rendered images (read an order number, a meeting day, list
buttons, battery level, Bluetooth state), raw data `evals/results/bench_vlm.json`:

| Model | Accuracy | Warm latency | First load |
|---|---|---|---|
| moondream | 2/5 (40%) | 0.4 s | 4.1 s |
| qwen2.5vl:3b | 5/5 (100%) | 1.5 s | 5.9 s |

Moondream often returned an empty string or misread text. Qwen2.5-VL 3B is 3.2 GB, so it is **only usable because the
chat model is unloaded first** (`VLM.describe` does that, then unloads itself with `keep_alive=0`). The spill figure
measured while the chat model was still resident is not representative; the clean figure is in `EVAL_REPORT.md`.
MiniCPM-V and larger models were not tried: they do not fit in 4 GB VRAM next to a KV cache.
The VLM is a last resort. For reading text, UIA (0.03 s) and OCR (about 4 s cold) are used first.

## D12. Runtime is offline for models
A network audit of the real process showed a connection to a CDN at idle: faster-whisper checks the Hugging Face Hub on
every model load. EV now loads Whisper with `local_files_only=True`, sets `HF_HUB_OFFLINE=1` and the telemetry opt-outs
in `ev/__main__.py`, and fails with an actionable message if the model is missing (`scripts/download_models.py`).
`web_search` pins ddgs to the `duckduckgo` backend (default `auto` queried several search engines).
After the fix the idle process tree talks only to `127.0.0.1`.

## D13. VAD: Silero (streaming) with energy fallback
`faster_whisper.vad.SileroVADModel` is a stateless batch wrapper; EV drives its ONNX session directly with persistent
LSTM state, 512-sample chunks and a 64-sample context. On Piper speech vs noise/silence: Silero detected 94% of speech
blocks (energy VAD 83%) with 0% false positives on noise and silence, at about 4 ms per audio-second.
If Silero cannot load, EV falls back to the energy VAD.

## D14. Spoken-text sanitiser
A 3B model ignores "no lists / no percent signs / no URLs" often enough to matter. `ev/llm/spoken.py` enforces the voice
rules on every sentence before TTS (markdown, list markers, emoji, `%`, `$`, URLs become spoken forms such as "google dot com").
Persona pass rates in the report are for what EV would actually say, not raw model output.

## D15. Models and tools that were rejected
* `qwen2.5:7b`: 4.7 GB, cannot fit. `qwen3:4b`: 27.6% of the weights spill to RAM at 2048 context and it is about 3x slower.
* `llama3.2` 3B: fits and is fast, but scored 68% on tool choice and called `type_text` for plain chat.
* Tk overlay: kept only for the toolkit benchmark.

## D16. Git
`git init` was run and `.gitignore` written, but **nothing has been committed**: the housekeeping hook configured for this
workspace says not to commit.

## D17. Real-use fixes: STT model, Kokoro voice, reply length
Live testing showed mishearing, a robotic voice and long replies. Changes, with what was and was not measured:
* **STT**: `distil-small.en`, beam 3. On this CPU (20 synthetic utterances, `evals/stt_eval.py` clips) RTF was base 0.25,
  small 0.8, distil-small.en 0.65. WER was about 11% for all of them, so **this benchmark cannot show the accuracy gain**
  (same caveat as D10: synthetic single-voice speech). The choice rests on `small`-class models being known to handle accents
  and noise better than `base`, not on a measurement here. Whisper encodes a fixed 30 s window, so cost is roughly
  per-utterance, not per-second. Roll back with `stt.model = "base"`.
* **TTS**: Kokoro was already installed. Steady-state RTF about 0.44 vs Piper 0.09, so the first sentence starts about 1 s
  later; it is warmed up at load and `app.py` falls back to Piper if it fails.
* **Length**: the 3B model ignored "one to three sentences", so `Brain` now stops generation at `llm.reply_max_sentences`.
  Persona eval after the rewrite: 30/30 deterministic, judge mean 4.69/5.

## D18. Closing apps, websites, screenshots, administrator rights
* **Ask only for what is risky.** `close_app` (polite), `open_website` and `take_screenshot` are `safe`. Force-closing, anything mentioning administrator rights, and any address that isn't a known or trusted site (an automated review showed that gating only `?`/`#` missed data in hostnames and paths) escalates to confirm in `policy.py`; protected processes are blocked. All of it is decided from config and arguments, never from model or screen text.
* **Why `open_website` and not clicking.** "Open YouTube and search X" through the screen agent is 4-6 model steps of clicking; building the search URL is one deterministic call. `build_url` is a pure function the policy also calls, so the policy judges the exact URL that would open. Its first version crashed the policy on `javascript:alert(1)` (found by a test); it now rejects non-web schemes and never raises.
* **Polite close.** `WM_CLOSE` to every window of the app (same as the X button). Browser tabs never match by title, so "close YouTube" can't shut the whole browser. EV's own process, its terminal and its Python helpers are skipped; apps EV launched are not, which a live test showed (EV started Notepad as a child process and then refused to close it).
* **Not built:** a `delete_file` tool. The policy still lists `delete_file`, `move_file` and `send_message` as `confirm`, but nothing registers them, so EV cannot delete or send anything yet.
* **Measured:** 307+ unit tests, security suite 96/96 with 0 bypasses, intent eval 62/72 (4 of 6 new phrasings route to the right tool on `qwen2.5:3b`; "get rid of the Notepad window" and "pull up the Rust subreddit on Reddit" got a text reply instead of a tool call). Persona eval is 29-30/30 and varies run to run on a 3B model.

## D19. Display mode (text card on the pill)
* **One window, not two.** The card is drawn by the existing pill widget (the window just gets taller), so it shares the pill's fade, always-on-top, click-through and no-focus behaviour. The pill stays where it is; the toolkit-independent model is `overlay/display.py`.
* **Only while on.** The core sends transcript text to the overlay only in display mode, and the overlay keeps just the latest exchange in memory. Nothing is written to disk.
* **What it shows.** The user's line appears after transcription (Whisper runs on the finished utterance, so there are no live partial words), and EV's reply is added sentence by sentence as each is queued for speech; an interrupted reply shows only what was spoken. Confirmation prompts and agent narration appear too.
* **Tried and removed: hiding the card from screen capture.** The idea was to keep EV's OCR and screenshots from reading its own transcript. `SetWindowDisplayAffinity` fails on the pill (error 8, even in monitor mode; probably because it is a layered, translucent window, not confirmed) and the card showed up in captures. It was removed rather than shipped as a no-op. Consequence: in display mode, `screenshot_describe` and OCR-based `read_screen` can see the card. UI Automation reads, the default, only see the foreground window.

## D20. Voice barge-in on speakers: model the echo, don't compare levels
Recording EV speaking through laptop speakers while the mic listened (nobody talking): mic rms 0.02-0.05 while EV talks (a person at the mic is about 0.05-0.1), arriving about 480 ms after the playback meter, envelope correlation 0.88 once aligned. The first detector compared mic level to playback level at the same instant and calibrated in the first 600 ms, before the echo arrived (ratio 0.009), so it fired on 2 of 3 recordings. The replacement estimates the delay by cross-correlation, predicts echo as gain x playback delayed by that amount (gain = 75th percentile of recent samples, frozen while a possible interruption is in progress), and arms only after 1.2 s of playback. Margin 1.5 was chosen from a grid over margin and gain percentile: the longest false "suspect" run on real echo was 5 of the 10 blocks needed to fire, and simulated talk-over at 0.08 rms or louder was detected in every trial (median 0.4-0.7 s). **Limits:** the user's voice is simulated by adding a modulated tone to real echo levels, not recorded; a quiet voice (0.06) is caught in about 4 of 6 trials; without echo cancellation, a quiet talker over loud speakers cannot be told from EV's own voice. Headphones remove the problem. If EV still cuts itself off, raise `interrupt_margin` or set `voice_interrupt = false`.

## D21. Response time and conversation feel: what was measured decides what is on
Goal: faster replies and a less walkie-talkie feel, without touching the safety path. Measured on this machine with the real Ollama (`qwen2.5:3b`), one run each unless a count is given:
* **Warm up with the real prompt (on).** The old warm-up sent a bare "hi", so the first real turn still processed the persona and ~20 tool schemas (~2.9k tokens). `Brain.warm()` now sends the real system prompt and tool list, so Ollama's prompt cache holds them. First speech after a cold model: **0.23 s** after the real-prompt warm-up vs **1.04 s** after the old bare warm-up (second turn 0.26 s). The cache only helps while the prefix is byte-identical, so the system prompt no longer carries the minute-level clock (it was invalidating the cache every minute): it has the date and part of the day, and the exact time is the `get_time_date` tool's job (the fast path already answers "what time is it"). The cache is also lost when the screen tools swap the chat model out, or after the summary call, so the client tracks that (`OllamaClient.is_cold`) and `Brain.rewarm_if_cold()` reloads the model in the background while the user is still speaking; the summary thread re-warms after itself.
* **Summary off the critical path (on).** The rolling-summary model call (every 8th turn) ran inline and delayed the end of that turn; it now runs on a background thread.
* **One fast-path match per turn, one tool-schema token count per registry change (on).** Small, but it ran on every turn.
* **Start-up warm-ups in parallel (on).** Speech model (plus a throwaway decode), voice model (a discarded synthesis, serialised with real speech by a lock), LLM prompt, and the app list. The Start-menu walk (50-300 ms, done up to four times per launch) is cached for 5 minutes and rebuilt once on a miss; the Store-app list (a PowerShell call, 1-2 s) is built ahead of time. Weather reuses one HTTP connection and caches geocoding (24 h) and forecasts (5 min). **Not measured:** the gain of these four; they remove work, but no before/after timing was taken.
* **Whisper `without_timestamps` (on).** Utterances are at most 15 s (one window), so timestamps aren't needed. **Not measured.**
* **Parallel tool batches (on, narrow).** When one model turn asks for several tools and all are side-effect free (`get_weather`, `web_search`, `get_time_date`, memory, timers, reminders), they run at the same time (test: two calls that only finish if concurrent). UI tools keep their order. Gating still happens one call at a time, on the calling thread, because confirmations are spoken. The tool pool grew from 2 to 6 workers so a batch plus the screen agent's nested calls cannot starve.
* **"One sec" filler (on).** If a slow tool (search, weather, screen read) is still running after `llm.filler_after_s = 1.3`, the system says a short varied phrase. It is system text, not model output, and is not stored in history. **Not measured live:** the timing is covered by tests only.
* **Smalltalk without the model (on).** A bare "thanks", greeting or goodbye gets a varied canned reply instantly. "yes", "okay" and "later" are deliberately excluded: they can answer a question EV just asked and need the history.
* **Follow-up window grows with the conversation (on).** +2 s per back-to-back turn up to 15 s (`conversation.followup_patience_s`, `followup_window_max_s`), and hesitation sounds ("hmm", "um") in a follow-up are dropped like other noise.
* **Direct tool replies (built, off).** Speaking a finished tool result ("Opened Spotify.") instead of a second model call looked like the biggest saving. Measured over 6 action turns each: **median 0.30 s direct vs 0.28 s rephrased**, i.e. nothing, because the warm prompt cache makes the second call nearly free, and the rephrased replies were more natural ("Remembered. Asha is your sister's name." vs "Noted."). So `llm.direct_tool_replies = false`; it may pay off on slow hardware or after a CPU fallback.
* **Early clause chunk (built, off).** Speaking a long first sentence from its first comma (`tts.first_clause_chars`) can't help when first speech already starts ~0.25 s after the question, and it costs prosody. Off; a clause chunk does not count toward the sentence cap.
* **Continuous mic (built, off until tried).** `SharedMic` keeps one input stream open for a whole conversation and fans it out: the barge-in monitor while EV speaks and the recorder while it listens each subscribe (immediately, not at first read) and unsubscribe when done, so the device is not closed and reopened between turns and nothing is lost at the hand-off. `conversation.continuous_mic = false` by default because the echo behaviour D20 was tuned on has only been exercised with a fake stream here. **Measured on the real microphone:** two listeners got 20 blocks in 312 ms (real time), a third joined the open stream in 30 ms (one block), nothing left subscribed after close; the first device open took 690 ms (a later re-open was not timed, so no per-turn saving is claimed). **Not verified:** barge-in and the echo tail with a real voice over speakers with this on; try it, and if EV starts hearing itself, raise `conversation.turn_gap_ms` or turn it off.
* **Considered, not built.** Speaking while EV is still thinking (the echo detector stays disarmed until EV has talked for 1.2 s, so speech before the first word of the reply is not caught; handling it means cancelling and merging the turn), speculative transcription at the first pause (a mid-sentence pause would make the final decode wait behind a stale one), and per-query tool subsets (sending fewer schemas would break the prompt-cache prefix this decision depends on and risks missing the right tool).
* **Safety is unchanged.** Parallel batches, direct replies, fillers and smalltalk all sit behind or beside `SafetyGate`; nothing here reads text to decide a permission. Security suite after the change: 88/88, 0 bypasses.
