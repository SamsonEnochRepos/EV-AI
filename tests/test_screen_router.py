"""Screen, click and camera phrases are routed in code (ev/llm/fastpath.py), not left to a 3B model to pick a tool for."""
from __future__ import annotations

import pytest

from ev.llm.brain import Brain
from ev.llm.client import OllamaClient
from ev.llm.fastpath import looks_like_question, match_fast
from ev.safety.gate import SafetyGate
from ev.safety.policy import Verdict
from tests.fakes import FakeOllama, text_reply


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
    ("open notepad and type hello", "do_on_screen", {"goal": "open notepad and type hello"}),
    ("click the search box and type cats", "do_on_screen", {"goal": "click the search box and type cats"}),
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
    assert m is None or m.tool not in {"click_element", "type_text", "press_keys", "scroll", "focus_window", "do_on_screen"}


@pytest.mark.parametrize("said", [
    "what am I holding", "can you see me", "how do I look", "look at me", "what's this thing I'm holding",
    "what do you see on the camera", "hey ev what am i wearing",
])
def test_camera_phrases(said):
    m = match_fast(said)
    assert m is not None and m.tool == "look_at_camera" and m.then_llm and m.args["question"].startswith(said[:10])


def test_what_do_you_see_alone_still_means_the_screen():
    assert match_fast("what do you see").tool == "read_screen"
    assert match_fast("can you look at my screen and tell me what's on it").tool == "read_screen"
    assert match_fast("summarize this page").tool == "read_screen"
    assert match_fast("describe this image on my screen").tool == "screenshot_describe"


def test_routed_calls_still_meet_the_policy(policy):
    """The router only picks the tool. Whether it may run is decided from the arguments, as for any call."""
    assert policy.evaluate("click_element", match_fast("click the send button").args).verdict is Verdict.CONFIRM
    assert policy.evaluate("press_keys", match_fast("press alt f4").args).verdict is Verdict.BLOCK
    assert policy.evaluate("press_keys", match_fast("press enter").args).verdict is Verdict.CONFIRM
    assert policy.evaluate("click_element", match_fast("click the password field").args).verdict is Verdict.BLOCK
    assert policy.evaluate("look_at_camera", {"question": "what am i holding"}).verdict is Verdict.ALLOW


def test_asking_for_the_camera_when_it_is_off_says_so(cfg, memory, policy, registry):
    """With the camera tool not registered, the model must not be left to guess what it can "see"."""
    fake = FakeOllama([text_reply("You are holding a cup.")])
    brain = Brain(cfg, OllamaClient(cfg, transport=fake.transport()), memory, registry, SafetyGate(policy), persona="You are EV.")
    said: list[str] = []
    res = brain.respond("what am I holding", on_sentence=said.append)
    assert "camera is turned off" in res.text and said == [res.text] and not fake.requests


@pytest.mark.parametrize("said", ["Can you do that for me?", "try that again", "go ahead and do it", "can you redo it please"])
def test_do_that_requests_are_not_lookups(said):
    assert not looks_like_question(said)


def test_real_questions_still_are():
    assert looks_like_question("who won the match last night?") and looks_like_question("what's the latest news")
