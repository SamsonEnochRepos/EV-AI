# Architecture

```
 hotkey / wake word
        |
        v
  [audio.capture] --mic blocks--> VAD endpointing --> [stt.whisper] (CPU, int8)
        |                                                  |
        | level                                            v text
        v                                            [llm.brain]
  [overlay.client] <--state/level/text--  core  --------> fast path (regex) or Ollama (GPU)
   (separate process,                 ^                     |
    token-auth loopback)              |             tool calls (validated)
                                      |                     v
                              [safety.gate] <------- [safety.policy] (config only)
                                      |  confirm? --> system readback --> real "yes" (voice / hotkey)
                                      v
                                [tools.*]  [screen.*: UIA -> OCR -> VLM]  [control.*]
                                      |
                                      v sentences (streamed)
                                [tts.speaker] --> [audio.playback] --level--> overlay
```

* **State machine** (`core/state.py`): IDLE → LISTENING → TRANSCRIBING → THINKING → (ACTING) → SPEAKING → IDLE, plus ERROR / PAUSED. Transitions publish `state` events on the bus; the overlay client and wake-word pause/resume subscribe.
* **Brain** (`llm/brain.py`): persona + remembered facts + rolling summary + short history → Ollama `/api/chat` (streaming, tools, `think=false`). Sentences are split incrementally and handed to TTS as soon as each completes. Tool calls go: schema validation → `SafetyGate` → execution (thread pool with timeout) → result back to the model. Results from untrusted sources (web, screen, files) are wrapped as data and scanned for injection phrasing.
* **Fast path** (`llm/fastpath.py`): regexes for time, timers, open app, volume, media, reminders, notes. They bypass the LLM entirely and still go through the gate. `match_smalltalk` answers a bare "thanks" / greeting / goodbye with a varied canned line.
* **Latency design** (D21): the system prompt is kept byte-identical between turns and `Brain.warm()` primes Ollama's prompt cache with the real prompt and tool list at start-up and whenever the model was unloaded or idle (`rewarm_if_cold`, triggered when the user starts speaking); the rolling summary runs on a background thread; start-up warms the speech model, voice, LLM and app list in parallel. A batch of side-effect-free tool calls runs concurrently (gating stays sequential), and a slow tool gets a spoken "one sec" after `llm.filler_after_s`.
* **Conversation** (`app.py`): after a reply the mic reopens without a wake word; the listening window grows with each back-to-back turn (`followup_patience_s`, capped by `followup_window_max_s`); talking over EV stops it (`audio/interrupt.py`) and the interrupting speech is carried into the next recording. With `conversation.continuous_mic = true` one `SharedMic` stream stays open for the whole conversation and the monitor and the recorder both listen to it (no re-open between turns, nothing lost at the hand-off); the wake word's own stream resumes only after it is closed.
* **Safety** (`safety/`): `Policy` decides allow/confirm/block from `config/policy.toml` and the arguments only. `SafetyGate` performs the confirmation as a separate system-driven step with the exact action read back; with no asker it fails closed. A kill event denies everything.
* **Screen** (`screen/`): UI Automation tree first, RapidOCR second, `qwen2.5vl:3b` VLM last (D11). The VLM path unloads the chat model first and itself immediately after (`keep_alive=0`), so VRAM never holds both.
* **Control** (`control/`): pyautogui/pywinauto actions re-check the resolved UIA element (password flag, sensitive window) right before acting. `ScreenAgent` is the bounded observe→act→verify loop (default 8 steps, stuck-detection, kill aware).
* **Overlay** (`overlay/`): independent process. Core → overlay is newline-delimited JSON over `127.0.0.1` with a per-session token delivered via stdin. The overlay exits when the core connection closes; the core restarts it (rate limited) if it dies.
* **Memory** (`memory/store.py`): SQLite (facts, notes, reminders, kv). No vector DB.
* **Degradation**: Ollama down → spoken message; configured model missing → fallback model; no mic → spoken message; TTS or speaker failure → logged, EV keeps running; overlay failure → ignored.
