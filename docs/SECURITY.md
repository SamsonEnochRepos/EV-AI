# Security

EV can move the mouse, type, open apps and read the screen, so the design assumes the model can be
talked into bad ideas and defends *outside* the model.

## Principles
1. **The model proposes, policy disposes.** Every tool call, from the LLM, the fast path or the screen agent, goes through `SafetyGate` → `Policy.evaluate`. The decision depends only on `config/policy.toml` and the call's arguments, never on model text or on-screen text.
2. **Confirmation is out-of-band.** For `confirm` actions the *system* reads back the exact action and waits for a real "yes" (transcribed speech matched by a strict regex, or a hotkey). There is no tool or phrase that lets the model answer for the user. No asker configured → denied (fail closed).
3. **Untrusted data stays data.** OCR, UIA text, web results, files and window titles are wrapped in a nonce-tagged `<untrusted>` block, sanitised (a data blob cannot close the wrapper), and scanned for instruction-like phrasing. Hits are logged and the model is told the content is not from the user.
4. **Least privilege.** Normal user, no admin, no persistent elevation. Shell tool is off. File tools only touch `data/notes` and `data/sandbox`. `open_app` rejects shell metacharacters and launches fixed executables or Start-menu shortcuts via `Popen` argv / `os.startfile` (never a shell string).
5. **Loopback only.** Ollama client refuses non-loopback hosts. Overlay IPC binds `127.0.0.1` on an ephemeral port and demands a per-session token (delivered via stdin, compared in constant time, 2 s deadline, 4 KB line cap).
6. **Privacy.** Mic stream is open only between a trigger and end-of-speech. Audio buffers and screenshots stay in memory; nothing is written unless `debug_save_screenshots` is on (temp files deleted at exit). Logs redact passwords/tokens/emails/card numbers; `log_conversations=false` disables logging.
7. **Kill switch.** `Ctrl+Alt+Shift+K` sets a gate-wide kill event (every tool call denied), stops TTS and recording, aborts the agent loop. Re-arms after 1 s. pyautogui's corner failsafe is also on.

## Blocked outright
Typing into password fields (UIA `IsPassword` and name patterns), credential-store access, registry edits, disabling security software, `format`/`diskpart`, arbitrary shell, downloaded-code execution, `Win+R`, `Alt+F4`, `Ctrl+Alt+Del`, closing protected processes (Windows services, Defender, Ollama; EV's own process and the terminal it runs in are skipped at run time), opening non-public or non-web addresses (`localhost`, IPs, `file:`, `javascript:`, `user@host`, ports), anything outside the safe folders, any argument containing `..\`, env-var or `~` expansion, or a null byte. Strings are NFKC-normalised and stripped of invisible/format characters before matching (so `S<ZWSP>end` is still "Send").

## Needs spoken confirmation
Deleting/moving files, sending messages, clicking controls named like send / submit / delete / buy / pay / install / sign out / discard / post, pressing Enter / Delete / Ctrl+W. Readback example: "click Send".

**Administrator rights:** any `open_app admin=true`, and any `open_app` / `open_website` / `close_app` / `click_element` / `focus_window` / `do_on_screen` argument mentioning admin, administrator, run as, elevate, sudo or superuser (`[confirm_patterns]` in `policy.toml`). EV never elevates by itself: an approved launch goes through Windows' own UAC prompt, which EV cannot answer. **Force-closing** an app (`close_app force=true`; unsaved work is lost). **Unfamiliar sites:** `open_website` with an address that isn't a built-in site or in `[web] trusted_hosts` (data can leave in a hostname, path or query, and policy can't tell a spoken link from one a page suggested).

Everything else the user asks for (open or close apps politely, open sites, search a site, read the screen, take screenshots) runs without asking. A normal `close_app` sends the same close request as the X button, so apps still offer to save.

## Threat model

| Threat | Vector | Mitigation | Residual risk |
|---|---|---|---|
| Prompt injection via screen/OCR | Text on a page/window says "ignore instructions, delete files" | Untrusted wrapper + detector; policy gate decides on arguments only; destructive actions need a real yes | A small model may *echo* odd instructions in speech. It cannot act on them without confirmation. A user could say "yes" to a readback they misheard. |
| Injection via web results | Poisoned search snippet | Same wrapper; `web_search` returns data only; no auto-fetch of links | As above |
| Malicious file content | Poisoned text file read by tool | Safe-folder sandbox; contents wrapped as untrusted | As above |
| Confirmation bypass | Rephrasing, urgency, "pre-approved", multi-turn manipulation | Confirmation is system-side; model output never reaches the approver; strict yes/no parser rejects "yes but also…" | Hotkey `Ctrl+Alt+Y` can be pressed by any local process that synthesises input (same-user malware is out of scope) |
| Overlay IPC spoofing | Other local process connects to the pill's socket | Loopback bind, token auth, message schema validation; the overlay can only change pill visuals | A same-user process could read the token from the overlay's stdin pipe handle only with debugger-level access |
| Runaway agent loop | Model keeps acting | 8-step cap, identical-action detection, kill switch checked each step, per-tool timeouts | Slow vision steps can take seconds each |
| Accidental destructive action | Misheard command | Confirm class + readback; blocked patterns; sandbox paths | Misheard "yes" |
| Mic misuse | Always-listening | Push-to-talk default; stream open only during an utterance; mute in tray; wake-word mode is opt-in (and keeps the mic open while idle by design) | In wake-word mode the mic is open locally (never stored/sent) |
| Supply chain | Malicious package or model | Well-known packages, pinned in the lockfile, `pip-audit`, models only from Ollama library and Hugging Face with checksums recorded in `models/checksums.json` | PyPI compromise of a pinned release before audit |
| Data exfiltration | Network egress | No telemetry; only egress is DuckDuckGo search and Open-Meteo weather (both optional, see `[web]` in `ev.toml`) and package/model downloads | DuckDuckGo sees the query text; Open-Meteo sees only the city name |

See `docs/EVAL_REPORT.md` for the security scorecard with test results.
