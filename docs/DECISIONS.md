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
openWakeWord has no pretrained "Hey EV". Push-to-talk is the default. Wake word is an optional,
pluggable listener (`trigger.mode = "wakeword"`, `wakeword_model`), defaulting to the placeholder
`hey_jarvis`. See README for training a custom model. Models live in `models/openwakeword/` and are
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
workspace says not to commit. Suggested phase commits are listed at the end of `docs/PROGRESS.md`.

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

