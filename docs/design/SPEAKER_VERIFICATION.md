# Speaker verification (design only, nothing built)

**Status:** a design note. No code, no model, no measurement. Everything numeric below is a thing to measure, not a result.

## Why

EV acts on whatever the microphone hears after the wake word. Two consequences:

1. **A spoken "remember that ..." is trusted** (`ev/safety/provenance.py`: the fast path counts as the user's own words). Audio
   from the speakers (a video, a call, a podcast that says "Hey Evie, remember that my email is ...") can plant a fact. It is
   listed as a residual risk in `docs/SECURITY.md` (D23). The same hole applies to anything else the fast path does without
   asking: open an app, change the volume, start a skill.
2. **Other people in the room.** A housemate or a visitor can command EV, including things the user would not want.

The gate already stops anything dangerous without a spoken yes, so this is about *who is allowed to say the safe things*, mainly memory
writes, skills and the dashboard.

## What it would do

Compare each utterance's voice with an enrolled voice and produce `speaker_score` (0 to 1). That score is **escalation-only**,
like provenance: it can add a confirmation or refuse a fast-path shortcut, never relax a decision.

| Score | Treatment |
|---|---|
| high | as today |
| middle | fast-path memory writes, skill runs and anything the policy marks `confirm` ask for a spoken yes first |
| low | EV answers chat but does not act and does not save facts ("I don't think that was you") |
| verification unavailable (no enrolment, model failed) | as today, and logged once; never block the owner out |

Two thresholds, not one: a low one so the owner is rarely refused when asking for the time, a high one for anything that writes
memory or runs a skill. Choose them from measured error rates (below), not from a guess.

## Pieces

* **Embedding model.** A small speaker-embedding model that runs on the CPU: ECAPA-TDNN (SpeechBrain), a WeSpeaker ResNet exported
  to ONNX, or Resemblyzer. Prefer an ONNX export run with `onnxruntime` (already a dependency through other models), to avoid
  adding PyTorch to a laptop with about 7 GB of RAM. *Licence, size and latency of each must be checked before choosing.*
* **Enrolment, by voice only** ("learn my voice"): EV asks for five or six short sentences in a normal voice, averages their
  embeddings and stores **only the vector** (a few hundred floats, in SQLite `kv` or a file in `data/`), never the audio. "Forget
  my voice" deletes it. Re-enrolling is a spoken yes, because it decides who is trusted. It must not be reachable from the
  dashboard or the model.
* **Hook.** The recorder already has the utterance's audio before it goes to speech-to-text. Compute the embedding in parallel with
  transcription (the embedding must not add latency to the first spoken word; measure it) and attach `speaker_score` to the turn
  (`TurnState`). `provenance.escalation` and the fast-path executor read it.
* **Confirmations.** The yes/no after a read-back could also be verified; a "yes" from a different voice should not count. Short
  utterances give poor embeddings, so this one needs its own, looser threshold and testing.
* **Rolling adaptation (maybe, later).** Updating the stored embedding with confident new samples helps with a cold, a different
  microphone or a headset, and is also how a poisoned enrolment would drift. Not in the first version.

## What it will not do

* **Replays of the owner's own voice** (a recording, a voice clone, a video of the owner talking) will pass. Verification stops
  strangers and ambient audio, not a targeted attack. Liveness detection is out of scope.
* **It is not authentication for anything that is not already confirmable.** The gate stays the authority; the score only adds
  questions.
* It does not make the wake word safer (the wake word fires on the model's own score; that is separate).

## Measure before building

1. Record the owner (about 40 utterances over several days: tired, hoarse, near and far, headset and laptop mic) and a few other
   voices (family, friends who agree; public speech datasets for volume). Record the **same** speakers *through the laptop's
   speakers* back into the mic, which is the actual attack.
2. Compute scores, plot genuine vs impostor, read off false-accept and false-reject at a few thresholds, **separately for short
   (under 2 s) and normal utterances**.
3. Time the embedding on the Ryzen 5 with Ollama running, and its RAM.
4. Decide: is there a threshold where the owner is refused less than about one time in twenty and played-back strangers are
   refused nearly always? If not, do not ship it; the residual risk stays documented.

## Alternatives, cheaper and available now

* **Say "yes" for the first memory write after wake-word audio that arrived while the speakers were playing.** EV knows when it
  and the system are playing sound (the output level, the "someone is using the speakers" state) and could treat a command heard
  during playback as untrusted. That needs no model and covers the video case, not the housemate case.
* **Turn memory writes through the fast path into spoken readbacks** ("Remember that your email is ... ?"). Safe, a little
  annoying, and a one-line policy change. This is the fallback if verification does not pan out.

## Tests and docs when it is built

Property test: `speaker_score` can never lower a decision (same shape as the `escalate` property in the security suite). Cases:
no enrolment, model error, short utterance, a low score on a fast-path `remember_fact`, a low score on a skill run, a low score on
a confirmation. `docs/SECURITY.md`: move the fast-path residual risk to "mitigated by speaker verification, except replays", add
a threat row for enrolment tampering, and record the measured error rates in a new D-entry.
