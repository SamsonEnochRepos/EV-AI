# Architecture

```
 hotkey / wake word
        |
        v
  [audio.capture] --mic blocks--> VAD endpointing --> [stt.whisper] (CPU, int8)
        |                                                  |
        | level                                            v text
        v                                            [llm.brain]
  [overlay.client] <--state/level--  core  --------> fast path (regex) or Ollama (GPU)
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
* **Fast path** (`llm/fastpath.py`): regexes for time, timers, open app, volume, media, reminders, notes. They bypass the LLM entirely and still go through the gate.
* **Safety** (`safety/`): `Policy` decides allow/confirm/block from `config/policy.toml` and the arguments only. `SafetyGate` performs the confirmation as a separate system-driven step with the exact action read back; with no asker it fails closed. A kill event denies everything.
* **Screen** (`screen/`): UI Automation tree first, RapidOCR second, Moondream VLM last. The VLM path unloads the chat model first and itself immediately after (`keep_alive=0`), so VRAM never holds both.
* **Control** (`control/`): pyautogui/pywinauto actions re-check the resolved UIA element (password flag, sensitive window) right before acting. `ScreenAgent` is the bounded observe→act→verify loop (default 8 steps, stuck-detection, kill aware).
* **Overlay** (`overlay/`): independent process. Core → overlay is newline-delimited JSON over `127.0.0.1` with a per-session token delivered via stdin. The overlay exits when the core connection closes; the core restarts it (rate limited) if it dies.
* **Memory** (`memory/store.py`): SQLite (facts, notes, reminders, kv). No vector DB.
* **Degradation**: Ollama down → spoken message; configured model missing → fallback model; no mic → spoken message; TTS or speaker failure → logged, EV keeps running; overlay failure → ignored.
