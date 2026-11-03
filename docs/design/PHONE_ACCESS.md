# Phone access (design only, nothing built)

**Status:** a design note. No code, nothing measured, no network listener beyond what exists today.

## Goal

Talk to EV from a phone (or any second device) when away from the laptop: ask a question, set a reminder, see the dashboard,
run a skill, and later get EV's warnings on the phone. The brain stays on the laptop. The phone is a microphone, a screen and
a speaker.

## What stays true

* **Every action goes through `SafetyGate` -> `Policy`**, decided on config and arguments only, exactly as for the voice path.
  A phone is just another source of the user's words, so it needs the same provenance rules (it is not the fast path unless
  the transcript is the user's own).
* **Nothing is exposed to the internet.** EV never accepts connections from the public internet and never opens a port on the
  router. The only paths in are (a) the home Wi-Fi, or (b) a private overlay network (Tailscale or WireGuard) the user sets up.
* **A stolen or compromised phone must not be able to widen anything.** The per-device tier below is the control, and
  `policy.toml` stays file-only on the laptop (D28).

## Shape

```
phone (PWA in a browser)  --WSS-->  hub listener on the laptop (its own port, tailnet/LAN interface only)
                                      |-- speech: audio in -> faster-whisper (existing)
                                      |-- Brain.respond(text, source="phone")   -> gate -> tools
                                      '-- text and audio out -> phone
```

* **A separate listener, not the dashboard's.** The dashboard (D28) is loopback-only by design; do not widen it. The hub reuses its
  pieces (one-time links, exact `Host`/`Origin` checks, strict CSP, size caps) but binds to the **Tailscale or LAN interface
  address only**, never `0.0.0.0`, and refuses to start if it would.
* **PWA over HTTPS.** Browsers only give a page the microphone over HTTPS. With Tailscale, `tailscale cert` gives a real
  certificate for the machine's `*.ts.net` name; on a LAN alone, a local CA (mkcert) installed on the phone. No third-party relay,
  no cloud account of ours.
* **Audio.** Push-to-talk in the page (hold a button), 16 kHz mono chunks over the WebSocket, transcribed by the existing STT. No
  wake word on the phone, no always-on mic. The reply comes back as text first and, optionally, as audio from Piper.
* **Local-first.** If the laptop is asleep or EV is not running, the phone says so. Nothing queues in the cloud.

## Pairing

1. At the laptop, say "pair my phone" or press a button in the dashboard. EV shows a **QR code** (overlay card or dashboard) with
   a one-time URL: `https://<machine>.<tailnet>.ts.net:<port>/pair?t=<token>`, valid for two minutes and one use.
2. The phone opens it. The page generates a keypair in the browser (WebCrypto, non-extractable), sends the **public key**, and the
   laptop stores it with a device name and a tier. The private key never leaves the phone.
3. EV **speaks and shows the device name and a short code** ("Pairing Sam's Pixel, code four seven two") and needs a spoken yes. A
   QR code seen over someone's shoulder is therefore not enough.
4. Later connections sign a server-chosen nonce with the device key. No passwords, no long-lived bearer token in a cookie.

Revoking a device is a voice command and a dashboard button; revoking is always allowed and needs no confirmation.

## Tiers (per device, set at pairing, changed only on the laptop)

| Tier | Can do |
|---|---|
| 0 watch | see state, conversation and notices; nothing else |
| 1 chat | ask questions, reminders, timers, notes, web search; nothing that touches apps or files |
| 2 act | everything the voice path can do **except** that anything the policy marks `confirm` cannot be confirmed from the phone |
| 3 act and confirm | as 2, and a `confirm` action can be approved from the phone with a number-match challenge (the laptop shows a number; the phone must send it back), never a bare "yes" |

Default is tier 1. Memory writes from any phone tier are never "plainly the user's words": they need the spoken yes on the laptop,
or, at tier 3 only, the number-match approval. Skills (`run_skill`) are tier 2 and up; saving or deleting one is laptop-only. A
phone can never change settings beyond the dashboard's tuning allow-list, and can never reach `policy.toml`.

## Threats to write into SECURITY.md when this is built

| Threat | Mitigation | Residual |
|---|---|---|
| Stolen unlocked phone | tier limits, revoke by voice, per-device rate limit, no standing destructive power at tier 1 or 2 | tier 3 on an unlocked phone can approve confirmations in the user's name |
| Pairing link or QR seen by another person | one use, two minutes, spoken confirmation of the device name and code | an attacker who is in the room and gets the owner to say yes |
| Another device on the home Wi-Fi | interface-bound listener, mutual keys, no unauthenticated route, strict `Host`/`Origin` | a LAN attacker can still see that a port is open (use the tailnet to hide it) |
| Tailnet account compromise | device keys are separate from the tailnet identity, so the attacker also needs a paired device | tailnet admin could add a node that can reach the port (pairing still blocks it) |
| Prompt injection through the phone path | same wrapper and gate; phone text is the user's words, tool results are still untrusted | as everywhere else |
| Mic abuse | push-to-talk only, indicator on the laptop while a phone mic is open | a paired phone can listen in while its button is held, by design |
| Replay | per-connection server nonce, signed | none known |
| Resource abuse | message size caps, one active session per device, hourly cap per tier | none known |

## Calls and "warn me about problems in live apps"

Out of scope here and different problems: taking phone calls needs telephony (SIP, a number, regulation, a lot of risk of EV
acting for the user without the user), and warning about live apps is an extension of the proactive engine (D27) with a push path
to the phone (Web Push, which needs a service worker and a push service, an external dependency to think about). Each gets its own
design note before any code.

## Order of work, when you want it

1. Decide tailnet vs LAN-only and the HTTPS route; try `tailscale cert` on this laptop first.
2. Read-only tier 0 hub (state and conversation), pairing and revoking, with the tests for the pairing and tier rules.
3. Tier 1 chat with push-to-talk audio; measure STT latency over Wi-Fi and the battery cost on the phone.
4. Tiers 2 and 3, with the number-match approval; extend the security suite (phone-sourced memory writes, confirm from a
   tier 2 device, replayed nonce, wrong `Host`, listener bound to the wrong interface).
5. Then, only if wanted, push notices.

## Not decided, needs the owner

Tailscale vs a plain LAN; whether tier 3 should exist at all; whether the phone may ever hear EV's audio replies outside the home;
and whether a phone mic counts as "the user's words" for the fast path (this design says no: it needs speaker verification first, see
`SPEAKER_VERIFICATION.md`).
