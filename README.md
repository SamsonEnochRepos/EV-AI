# EV

A fully local, voice-first assistant for Windows. No cloud, no API keys, no telemetry.
The only UI is a tiny pill that appears while she listens, thinks and speaks, plus a tray icon.

## Quick start

```powershell
# 1. Conda environment (Python 3.12)
conda env create -f environment.yml
conda activate ev
pip install -e ".[dev]"          # tests, ruff, mypy (optional)
pip install openwakeword         # only for wake-word mode

# 2. Free models (all local)
ollama pull qwen2.5:3b          # default brain (fits fully in 4 GB VRAM, ~80 tok/s here)
ollama pull llama3.2            # fallback brain (measured weaker at tool choice; see docs/DECISIONS.md D4/D15)
ollama pull qwen2.5vl:3b        # vision fallback, loaded only on demand while the chat model is unloaded
python scripts\download_models.py --voice en_US-hfc_female-medium --whisper distil-small.en --wakeword hey_jarvis
# hey_evie.onnx (the trained wake word) goes in models\openwakeword
# 3. Run
start_ev_conda.bat              # background, no console window
run_ev_debug_conda.bat          # foreground with logs; or: python -m ev --debug
```

Ollama must be running (`ollama serve`, or the tray app). It is only ever contacted on `127.0.0.1`.

EV is **offline at runtime**: models are downloaded once by the commands above, and EV never contacts a model hub.
The only outbound traffic is the optional DuckDuckGo search (`web.enabled = false` in `ev.toml` turns it off).
If Whisper's model is missing, EV tells you to run `scripts\download_models.py --whisper distil-small.en`.

## Using her

| Action                                                                         | Default hotkey (edit in `config/ev.toml`) |
| ------------------------------------------------------------------------------ | ------------------------------------------- |
| Talk (push-to-talk). Pressing it while she speaks interrupts her               | `Ctrl+Alt+Space`                          |
| **Kill switch**: stops speech, listening and any screen action instantly | `Ctrl+Alt+Shift+K`                        |
| Approve a confirmation                                                         | say "yes", or `Ctrl+Alt+Y`                |
| Decline a confirmation                                                         | say "no", or `Ctrl+Alt+N`                 |

Try: "open Spotify", "set a timer for ten minutes", "remind me in twenty minutes to stretch",
"volume up", "what's on my screen", "search the web for ...", "remember that my dog is called Biscuit".
Moving the mouse into a screen corner also aborts any mouse automation (pyautogui failsafe).

Preview the pill without running the assistant: `python -m ev.overlay --demo`.
Developer text REPL (not the product UI): `python -m ev --repl --debug`.

## Customising

* **Persona:** edit `config/persona.md`. It is read at startup. Run `python -m ev.evals.run persona` afterwards
  to check you did not break her voice rules.
* **Safety policy:** `config/policy.toml` (tool permission levels, app allowlist, safe folders, blocked patterns).
  See `docs/SECURITY.md`.
* **Everything else:** `config/ev.toml`. Put per-machine overrides in `config/ev.local.toml` (git-ignored).
* **Speed and conversation feel:** `llm.rewarm_after_s`, `llm.filler_after_s` / `filler_tools` (the "one sec" for slow tools), `conversation.followup_patience_s` / `followup_window_max_s` (how long she keeps listening mid-conversation) `conversation.turn_gap_ms`, and `conversation.continuous_mic` (one mic stream for the whole conversation; off until you've tried it on your mic and speakers). Two options are off because they measured no gain here: `llm.direct_tool_replies` and `tts.first_clause_chars` (see `docs/DECISIONS.md` D21).

## Wake word ("Hey EV")

`config/ev.toml` ships with `trigger.mode = "wakeword"` and the trained `hey_evie` model (answers to "hey evie",
"hey EV", "evie" and "EV"). It needs `pip install openwakeword` and `models/openwakeword/hey_evie.onnx`; the push-to-talk
hotkey always works too. Set `trigger.mode = "hotkey"` for push-to-talk only. To retrain or record your own voice, see
`scripts/wakeword/README.md` (training kit, `train_cloud.ipynb`); `trigger.wakeword_model` also accepts a path to any
openWakeWord `.onnx`. `hey_jarvis` (`download_models.py --wakeword hey_jarvis`) is a pretrained stand-in.

## Kokoro voice (optional upgrade)

Piper (`en_US-hfc_female-medium`) is the default. To try Kokoro: `pip install kokoro-onnx`, download `kokoro-v1.0.onnx` and
`voices-v1.0.bin` from the kokoro-onnx GitHub release `model-files-v1.0` into `models/kokoro/`, then set `tts.engine = "kokoro"`.
Measured comparison: `docs/EVAL_REPORT.md` (TTS section) and `docs/DECISIONS.md`.
Voices are generic/original; do not use a clone of a real person's voice.

## Troubleshooting

* "I can't reach my brain": start Ollama. EV falls back to `llama3.2` if the configured model is missing.
* Nothing happens on the hotkey: another app may own `Ctrl+Alt+Space`; change `hotkeys.push_to_talk` in `config/ev.toml`.
* Logs are in `logs/ev.log` (redacted). `python -m ev --debug` also prints them.
* Panic button: `Ctrl+Alt+Shift+K` stops speech, listening and any screen action immediately.

## Autostart (opt-in)

```powershell
schtasks /Create /TN "EV" /TR "\"%CD%\start_ev.bat\"" /SC ONLOGON /RL LIMITED
```

Remove with `schtasks /Delete /TN "EV" /F`. Runs as your normal user; EV never needs admin rights.

## Tests and evals

```powershell
make check            # lint + types + unit/integration tests
python -m pytest -m integration
python -m ev.evals.run security     # red-team suite
python -m ev.evals.run all          # all functional + security evals (needs Ollama)
```

Reports: `docs/EVAL_REPORT.md`; machine-readable results in `evals/results/`.

## Docs

`docs/ARCHITECTURE.md`, `DECISIONS.md`, `SECURITY.md`, `EVAL_REPORT.md` (generated by the eval run).
