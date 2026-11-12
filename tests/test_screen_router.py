"""Screen and click phrases are routed in code (ev/llm/fastpath.py), not left to a 3B model to pick a tool for."""
from __future__ import annotations

import pytest

from ev.llm.fastpath import looks_like_question, match_fast
from ev.safety.policy import Verdict


@pytest.mark.parametrize("said, tool, args", [
    ("click the camera button", "click_element", {"name": "camera"}),
    ("Click on Send.", "click_element", {"name": "send"}),
    ("double click the folder", "click_element", {"name": "folder"}),
    ("tap the settings icon", "click_element", {"name": "settings"}),
    ("press the play button", "click_element", {"name": "play"}),
    ("press control s", "press_keys", {"combo": "ctrl+s"}),
    ("press ctrl plus shift t", "press_keys", {"combo": "ctrl+shift+t"}),
    ("hit escape", "press_keys", {"combo": "esc"}),
    ("scroll down", "scroll", {"direction": "down", "amount": 3}),
    ("scroll up a lot", "scroll", {"direction": "up", "amount": 10}),
    ("scroll down 5 times", "scroll", {"direction": "down", "amount": 5}),
    ("Type Hello World.", "type_text", {"text": "Hello World"}),
    ("switch to chrome window", "focus_window", {"title": "chrome"}),
    ("bring notepad to the front", "focus_window", {"title": "notepad"}),
])
def test_screen_phrases_are_routed(said, tool, args):
    m = match_fast(said)
    assert m is not None and (m.tool, m.args) == (tool, args)


@pytest.mark.parametrize("said", [
    "click it", "click here", "type of cake", "type in the search box cats", "switch to dark mode", "press a", "press play",
    "tell me a joke", "scroll", "focus",
])
def test_vague_or_ordinary_speech_is_left_alone(said):
    m = match_fast(said)
    assert m is None or m.tool not in {"click_element", "type_text", "press_keys", "scroll", "focus_window"}


def test_what_do_you_see_alone_still_means_the_screen():
    assert match_fast("what do you see").tool == "read_screen"
    assert match_fast("can you look at my screen and tell me what's on it").tool == "read_screen"
    assert match_fast("summarize this page").tool == "read_screen"


def test_routed_calls_still_meet_the_policy(policy):
    """The router only picks the tool. Whether it may run is decided from the arguments, as for any call."""
    assert policy.evaluate("click_element", match_fast("click the send button").args).verdict is Verdict.CONFIRM
    assert policy.evaluate("press_keys", match_fast("press alt f4").args).verdict is Verdict.BLOCK
    assert policy.evaluate("press_keys", match_fast("press enter").args).verdict is Verdict.CONFIRM
    assert policy.evaluate("click_element", match_fast("click the password field").args).verdict is Verdict.BLOCK


@pytest.mark.parametrize("said", ["Can you do that for me?", "try that again", "go ahead and do it", "can you redo it please"])
def test_do_that_requests_are_not_lookups(said):
    assert not looks_like_question(said)


def test_real_questions_still_are():
    assert looks_like_question("who won the match last night?") and looks_like_question("what's the latest news")
