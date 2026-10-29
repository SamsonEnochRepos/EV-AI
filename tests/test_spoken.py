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
