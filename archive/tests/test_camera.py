"""look_at_camera: one frame, described by the local vision model, never offered to the chat model as a tool."""
from __future__ import annotations

import json

from PIL import Image

from ev.llm.client import OllamaClient
from ev.safety.gate import SafetyGate
from ev.tools.base import ToolRegistry
from ev.vision.camera import CameraError, make_question, register_camera_tool
from tests.fakes import FakeOllama, text_reply


def _registry(cfg, policy, grab, reply="A person holding a red mug."):
    fake = FakeOllama([text_reply(reply)])
    reg = ToolRegistry()
    register_camera_tool(reg, cfg, policy, OllamaClient(cfg, transport=fake.transport()), grab=grab)
    return reg, fake


def _frame(*_a, **_k):
    return Image.new("RGB", (64, 48), (120, 90, 60))


def test_it_describes_one_frame_and_marks_the_answer_untrusted(cfg, policy):
    reg, fake = _registry(cfg, policy, _frame)
    r = reg.get("look_at_camera").func(question="what am I holding?")
    assert r.ok and r.message == "A person holding a red mug." and r.untrusted and r.source == "camera"
    vlm_call = [q for q in fake.requests if q.get("model") == cfg["vlm"]["model"]][0]
    msg = vlm_call["messages"][0]
    assert "what am I holding?" in msg["content"] and len(msg["images"]) == 1          # the photo goes to the local VLM only


def test_the_model_cannot_call_it(cfg, policy):
    """Voice only: text on a page or a screen must never be able to switch the camera on."""
    reg, _ = _registry(cfg, policy, _frame)
    assert reg.get("look_at_camera") is not None
    assert "look_at_camera" not in json.dumps(reg.schemas())


def test_a_camera_that_wont_open_is_a_spoken_sentence_not_a_crash(cfg, policy):
    def busy(*_a, **_k):
        raise CameraError("I can't open the camera. Another app might be using it, or it's switched off.")

    reg, fake = _registry(cfg, policy, busy)
    r = reg.get("look_at_camera").func(question="")
    assert not r.ok and "can't open the camera" in r.message and not fake.requests        # no VLM call without a picture


def test_an_unexpected_driver_error_does_not_leak(cfg, policy):
    def broken(*_a, **_k):
        raise OSError("0x8007001F device not functioning")

    reg, _ = _registry(cfg, policy, broken)
    r = reg.get("look_at_camera").func(question="")
    assert not r.ok and r.message == "I couldn't use the camera."


def test_off_in_config_means_no_tool(cfg, policy):
    cfg.raw["camera"] = {**cfg.raw.get("camera", {}), "enabled": False}
    reg, _ = _registry(cfg, policy, _frame)
    assert reg.get("look_at_camera") is None


def test_gate_allows_the_tool_only_as_configured(cfg, policy):
    assert SafetyGate(policy).check("look_at_camera", {"question": "can you see me"}).allowed


def test_the_question_is_the_users_words_or_a_general_description():
    assert make_question("what colour is this?").endswith("what colour is this?")
    assert "Describe the person" in make_question("   ")
    assert len(make_question("x" * 900)) < 400
