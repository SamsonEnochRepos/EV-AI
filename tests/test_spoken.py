import pytest

from ev.llm.spoken import spoken_text


@pytest.mark.parametrize("raw,expected", [
    ("25% of 80 is 20.", "25 percent of 80 is 20."),
    ("Here it is: https://www.google.com/search?q=x", "Here it is: google dot com"),
    ("**Bold** and `code` here", "Bold and code here"),
    ("- Germany invades Poland. - Japan attacks Pearl Harbor.", "Germany invades Poland. Japan attacks Pearl Harbor."),
    ("1. First thing 2. Second thing", "First thing. Second thing"),
    ("Great job 🎉😀", "Great job"),
    ("That costs $1,200 & more", "That costs 1200 dollars and more"),
    ("See [the docs](http://x.io/a) now", "See the docs now"),
    ("Poland in 1939 - Japan in 1941 - Midway in 1942", "Poland in 1939. Japan in 1941. Midway in 1942"),
    ("", ""),
])
def test_spoken_text(raw, expected):
    assert spoken_text(raw) == expected


def test_plain_speech_untouched():
    s = "Sure thing, I'll set that timer for ten minutes."
    assert spoken_text(s) == s
    assert spoken_text("It's a well-known, state-of-the-art idea.") == "It's a well-known, state-of-the-art idea."


@pytest.mark.parametrize("raw,expected", [
    ("Boss, I'm not sure what email you mean.", "I'm not sure what email you mean."),
    ("boss! that was quick", "That was quick"),
    ("Sam: sounds good.", "Sounds good."),
    ("Night, boss. Sleep well.", "Night, boss. Sleep well."),            # a name later in the sentence is the point of the rule
    ("Boss battles are the best part.", "Boss battles are the best part."),
    ("Samsung makes good phones.", "Samsung makes good phones."),
    ("Boss!", "Boss!"),                                                  # never empty a reply
])
def test_a_reply_never_opens_with_a_name(raw, expected):
    from ev.llm.spoken import drop_name_opener

    assert drop_name_opener(raw) == expected


def test_the_brain_strips_a_name_opener_from_the_first_sentence_only(cfg, memory, registry, gate):
    from ev.llm.brain import Brain
    from ev.llm.client import OllamaClient

    from .fakes import FakeOllama, text_reply

    fake = FakeOllama([text_reply("Boss, that sounds fun. Boss, enjoy it.")])
    brain = Brain(cfg, OllamaClient(cfg, transport=fake.transport()), memory, registry, gate, persona="You are EV.")
    heard: list[str] = []
    res = brain.respond("I'm going to a concert", on_sentence=heard.append, allow_fast=False)
    assert heard[0] == "That sounds fun." and res.text.startswith("That sounds fun.")
