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


@pytest.mark.parametrize("text", [
    "Open the particular screenshot you just took",        # the failing example from PROJECT_DOCUMENTATION.md
    "open it", "open that", "open this please", "can you open it again",
    "open the screenshot", "show me the screenshot", "pull up the last screenshot",
    "open the latest file", "open the note you just made", "open the file you just saved",
    "Hey EV, open the screenshot that you just took.",
])
def test_open_it_goes_to_open_file(text):
    m = match_fast(text)
    assert m is not None, text
    assert m.tool == "open_file" and m.args == {"which": "last"}


@pytest.mark.parametrize("text", [
    "open the pdf you downloaded yesterday", "open the picture of my dog", "open that website you mentioned",
    "open the file with my budget in it", "open the last thing we talked about", "open the one I used yesterday",
])
def test_open_with_a_referent_never_becomes_open_app(text):
    """Phrases that point at something are for the model, never an app lookup that ends in 'couldn't find an app called...'."""
    m = match_fast(text)
    assert m is None or m.tool != "open_app", (text, m)


@pytest.mark.parametrize("text,name", [
    ("open file explorer", "file explorer"), ("open the file explorer", "file explorer"),
    ("open Spotify", "spotify"), ("open visual studio code", "visual studio code"), ("launch itunes", "itunes"),
    ("open one note", "one note"),
])
def test_open_app_still_works_for_real_apps(text, name):
    m = match_fast(text)
    assert m is not None and m.tool == "open_app" and m.args == {"name": name}


@pytest.mark.parametrize("text,args", [
    ("what did I say about the budget", {"query": "the budget"}),
    ("What did I tell you about Priya last week?", {"query": "priya", "when": "last week"}),
    ("what do my notes say about the trip", {"query": "the trip"}),
    ("search my notes for tomatoes", {"query": "tomatoes"}),
    ("Hey EV, check my notes about lisbon yesterday", {"query": "lisbon", "when": "yesterday"}),
    ("what did we talk about yesterday", {"query": "", "when": "yesterday"}),
    ("what's in my notes about the tomatoes", {"query": "the tomatoes"}),
    ("What do I have in my notes about Lisbon last week?", {"query": "lisbon", "when": "last week"}),
    ("do I have any notes on the budget", {"query": "the budget"}),
    ("find my notes about the trip", {"query": "the trip"}),
    ("what notes do I have about Priya", {"query": "priya"}),
    ("look up the invoice in my notes", {"query": "the invoice"}),
    # real transcript lines from the first run: Whisper keeps the filler, and the model must not answer "do you remember" itself
    ("Yeah, so what is in my notes about Lisbon?", {"query": "lisbon"}),
    ("Do you remember? My dog Bistic?", {"query": "my dog bistic"}),
    ("do you remember my name", {"query": "my name"}),
    ("You remember my dog Bistic, right?", {"query": "my dog bistic"}),
    ("you still remember my sister Asha don't you", {"query": "my sister asha"}),
    ("Okay, so do you remember when we went to Goa", {"query": "when we went to goa"}),
])
def test_recall_fast_path(text, args):
    m = match_fast(text)
    assert m is not None and m.tool == "recall" and m.args == args and m.then_llm


@pytest.mark.parametrize("text", [
    "look up the invoice in my notes", "search for the budget in my notes", "search my notes for tomatoes",
    "find the trip in my vault",
])
def test_searching_your_own_notes_is_never_a_web_search(text):
    """The query would leave the machine. Anything that names the user's notes goes to recall."""
    m = match_fast(text)
    assert m is not None and m.tool == "recall", (text, m)


@pytest.mark.parametrize("text,args", [
    ("remember that my dog's name is Biscuit", {"key": "dog's name", "value": "Biscuit"}),
    ("Remember my wifi network is Blue Door.", {"key": "wifi network", "value": "Blue Door"}),
    ("please remember that I'm allergic to peanuts", {"key": "allergic peanuts", "value": "I'm allergic to peanuts"}),
    ("My dog's name is Biscuit, remember that", {"key": "dog's name", "value": "Biscuit"}),
    ("I'm allergic to peanuts, keep that in mind.", {"key": "allergic peanuts", "value": "I'm allergic to peanuts"}),
    ("Don't forget my sister's name is Asha", {"key": "sister's name", "value": "Asha"}),
    ("keep in mind that I work night shifts", {"key": "work night shifts", "value": "I work night shifts"}),
    ("my birthday is June third, please remember that", {"key": "birthday", "value": "June third"}),
])
def test_remember_fast_path_keeps_the_users_wording(text, args):
    m = match_fast(text)
    assert m is not None and m.tool == "remember_fact" and m.args == args and not m.then_llm


@pytest.mark.parametrize("text", [
    "review my unconfirmed facts", "Review the unverified notes", "go through my pending facts", "list unconfirmed memories",
    "review the facts I haven't confirmed",
])
def test_review_fast_path(text):
    m = match_fast(text)
    assert m is not None and m.tool == "review_facts"


@pytest.mark.parametrize("text", [
    "remember to call mom", "remember when we went to Goa", "what did I have for lunch",
    "what did I say", "review my code", "check my notes", "I will remember that", "I'll keep that in mind",
    "you should remember that", "do you remember that", "do you remember it?", "can you keep that in mind", "I don't forget that", "remember that",
    "don't forget to call mom", "will you remember that", "what's in my pocket", "do I have any plans", "find my keys",
    "look up the weather", "what notes should I take",
])
def test_memory_phrases_that_are_not_commands(text):
    m = match_fast(text)
    assert m is None or m.tool not in ("remember_fact", "recall", "review_facts"), (text, m)


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


@pytest.mark.parametrize("said,name", [("call me boss", "boss"), ("you can call me Chief", "Chief"), ("from now on call me Sam please", "Sam"),
                                       ("please call me captain", "captain")])
def test_call_me_saves_a_preferred_name(said, name):
    m = match_fast(said)
    assert m and m.tool == "remember_fact" and m.args == {"key": "preferred_name", "value": name}


@pytest.mark.parametrize("said", ["call me back later", "call me at five", "call me tomorrow", "call me a taxi", "call me later", "call my mom"])
def test_call_me_something_else_is_not_a_name(said):
    m = match_fast(said)
    assert not (m and m.tool == "remember_fact")


@pytest.mark.parametrize("text,expected", [
    ("what was the cyclone's name that recently appeared in Chennai?", True),
    ("I asked what was the cyclone's name?", True),
    ("do you have any idea what that is about", True),
    ("I like the rain a lot", False),
    ("who is it", False),
    ("how are you doing today", False),
])
def test_looks_like_question(text, expected):
    from ev.llm.fastpath import looks_like_question

    assert looks_like_question(text) is expected
