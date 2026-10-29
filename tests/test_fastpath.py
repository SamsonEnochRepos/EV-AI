import pytest

from ev.llm.fastpath import match_fast
from ev.llm.sentences import SentenceSplitter


@pytest.mark.parametrize("text,tool,args", [
    ("what time is it", "get_time_date", {}),
    ("Hey EV, what's the date?", "get_time_date", {}),
    ("set a timer for 10 minutes", "set_timer", {"minutes": 10.0}),
    ("please set a timer for five minutes", "set_timer", {"minutes": 5.0}),
    ("set a timer for 30 seconds", "set_timer", {"seconds": 30}),
    ("start a 25 minute timer", "set_timer", {"minutes": 25.0}),
    ("set a timer for two hours", "set_timer", {"minutes": 120.0}),
    ("open Spotify", "open_app", {"name": "spotify"}),
    ("can you open notepad please", "open_app", {"name": "notepad"}),
    ("launch the calculator", "open_app", {"name": "calculator"}),
    ("volume up", "set_volume", {"direction": "up"}),
    ("turn it down", "set_volume", {"direction": "down"}),
    ("mute", "set_volume", {"direction": "mute"}),
    ("set the volume to 40 percent", "set_volume", {"level": 40}),
    ("pause the music", "media_control", {"action": "play_pause"}),
    ("skip", "media_control", {"action": "next"}),
    ("remind me in 20 minutes to stretch", "set_reminder", {"text": "stretch", "in_minutes": 20.0}),
    ("take a note buy milk", "take_note", {"text": "buy milk"}),
    ("what's the date today", "get_time_date", {}),
    ("it's way too loud in here", "set_volume", {"direction": "down"}),
    ("make it louder", "set_volume", {"direction": "up"}),
    ("resume my song", "media_control", {"action": "play_pause"}),
    ("copy that", "copy_selection", {}),
    ("paste it in", "paste", {}),
    ("forget my dog's name", "forget_fact", {"key": "dog's name"}),
    ("forget my favorite color", "forget_fact", {"key": "favorite color"}),
    ("what's the weather like", "get_weather", {"day": "today"}),
    ("would you open up Claude Desktop", "open_app", {"name": "claude desktop"}),
])
def test_fast_matches(text, tool, args):
    m = match_fast(text)
    assert m is not None, text
    assert m.tool == tool and m.args == args


@pytest.mark.parametrize("text", [
    "how are you today", "tell me a joke", "open up to me about your day",
    "who won the game", "set a timer", "play something by Daft Punk",
    "it's too loud outside, let's go for a walk", "I can't hear the birds today",
    "delete my temp file", "remove my old notes", "forget it", "forget about it", "forget what I said", "delete my files", "delete everything", "forget the whole thing",
])
def test_fast_ignores_chat(text):
    assert match_fast(text) is None


def test_sentence_splitter_streams():
    s = SentenceSplitter(min_chars=5)
    out = []
    for piece in ["Sure thing. I'll ", "do that now! Anything ", "else"]:
        out += s.feed(piece)
    out += s.flush()
    assert out == ["Sure thing.", "I'll do that now!", "Anything else"]


def test_splitter_abbreviation_and_decimal():
    s = SentenceSplitter(min_chars=5)
    out = s.feed("Dr. Smith owes me 3.5 dollars. Ok then. ") + s.flush()
    assert out[0].startswith("Dr. Smith owes me 3.5 dollars.")
