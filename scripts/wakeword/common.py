"""Shared constants for the wake-word training scripts (see scripts/wakeword/README.md)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "wakeword"          # git-ignored
RAW = WORK / "raw"
SYNTH = WORK / "synth"
FEATS = WORK / "features"
PIPER_DIR = ROOT / "models" / "piper"
OWW_DIR = ROOT / "models" / "openwakeword"

SR = 16000
MODEL_NAME = "hey_evie"                     # -> models/openwakeword/hey_evie.onnx

# The one model answers to all four phrasings. Spellings are tuned for how Piper's espeak front end reads them:
# "Evie" can be said /ev-ee/ or /ee-vee/, "EV" is the letters E-V.
POSITIVE_PHRASES = [
    "hey evie", "hey eevee", "hey evvy",
    "hey E V", "hey ee vee",
    "evie", "eevee", "evvy",
    "E V", "ee vee",
]
# Weighted so a bare "evie"/"EV" (the hardest to tell from ordinary speech) still gets plenty of examples.
POSITIVE_WEIGHTS = [5, 2, 1, 4, 1, 4, 2, 1, 3, 1]

# Near misses: things the model has to learn are NOT the wake word.
ADVERSARIAL = [
    "every", "everyday", "eve", "even", "evening", "ever", "evil", "heavy", "heavy V", "heavy E",
    "Evelyn", "Eva", "Ava", "Eddie", "Emmy", "Evan", "Ivy", "Abby", "Avi", "Effie",
    "hey Eddie", "hey Emmy", "hey Eve", "hey Ivy", "hey Eva", "hey Evan", "hey Abby", "hey Siri", "hey Steve",
    "hey Kevin", "hey Devi", "hey Levi", "hey Stevie", "hey Jeevy", "hey Bevy", "hey guys", "hey there", "hey you",
    "hey everyone", "hey every", "hey heavy", "hey even", "hey ever",
    "TV", "B V", "D V D", "P V C", "E T", "E Z", "E E", "V V", "V P", "U V", "G V", "T V remote", "V B", "B E",
    "E V charger", "electric vehicle", "E Vs", "evidence", "evolve", "evoke", "evasive", "evaluate", "event", "eventually",
    "he", "key", "bee", "see", "me", "we", "knee", "tea", "pea", "free", "three", "believe", "achieve", "receive",
    "easy", "ease", "eat", "heat", "heavy", "heaven", "seven", "eleven",
    "okay", "hey", "hello", "hi there", "Eve and Adam", "Steve", "Genevieve", "Evita",
]

# Things people actually say to EV. As negatives these keep "ordinary commands" from tripping the model.
COMMANDS = [
    "what time is it", "set a timer for five minutes", "open Spotify", "turn the volume up", "play some music",
    "what's the weather like", "remind me to call mum at six", "take a note", "open Chrome", "pause the music",
    "next song", "mute", "what's on my screen", "close the window", "search the web for pizza near me",
    "how long until my timer is done", "good morning", "thank you", "never mind", "stop", "cancel that",
    "turn it down a bit", "open the browser", "what day is it today", "tell me a joke", "what's two plus two",
    "can you help me with something", "I need to focus", "switch to display mode", "read that back to me",
    "I'll be right back", "did you see that", "where did I put my keys", "dinner is ready", "turn off the lights",
    "that is not what I meant", "let's try again", "how do you spell that", "what does that word mean",
]
