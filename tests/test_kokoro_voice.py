"""Kokoro voice names: single voices, blends, and the language that has to match the accent."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ev.tts.kokoro_engine import KokoroEngine, parse_voice, voice_lang


def test_a_single_voice_and_a_blend_are_read_the_same_way():
    assert parse_voice("af_heart") == [("af_heart", 1.0)]
    assert parse_voice("af_heart:60,af_bella:40") == [("af_heart", 0.6), ("af_bella", 0.4)]
    assert parse_voice("af_heart, af_bella") == [("af_heart", 0.5), ("af_bella", 0.5)]
    assert parse_voice("af_heart:3,bf_emma:1") == [("af_heart", 0.75), ("bf_emma", 0.25)]


@pytest.mark.parametrize("bad", ["", "  ", "heart", "af_heart:x", "af_heart:0", "af_heart:-1", "af_heart;calc", "../af_heart",
                                 "a_b,c_d,e_f,g_h,i_j"])
def test_a_typo_in_the_voice_is_an_error_not_a_guess(bad):
    with pytest.raises(ValueError):
        parse_voice(bad)


def test_british_voices_use_british_phonemes():
    assert voice_lang("bf_emma") == "en-gb" and voice_lang("bm_george") == "en-gb"
    assert voice_lang("af_heart") == "en-us" and voice_lang("am_michael") == "en-us"


class _FakeKokoro:
    calls: list[dict] = []

    def __init__(self, model, voices): ...

    def get_voice_style(self, name):
        return np.full(4, {"af_heart": 1.0, "af_bella": 3.0}[name], dtype=np.float32)

    def create(self, text, voice, speed, lang):
        _FakeKokoro.calls.append({"voice": voice, "speed": speed, "lang": lang})
        return np.zeros(240, dtype=np.float32), 24000


@pytest.fixture
def engine_for(cfg, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=_FakeKokoro))
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"x")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"x")
    cfg["tts"]["kokoro_dir"] = str(tmp_path)

    def make(voice):
        cfg["tts"]["kokoro_voice"] = voice
        _FakeKokoro.calls.clear()
        return KokoroEngine(cfg)

    return make


def test_the_engine_uses_the_voice_s_language_and_blends_style_vectors(engine_for):
    eng = engine_for("bf_emma")
    list(eng.synthesize("Hello there."))
    assert _FakeKokoro.calls[-1]["voice"] == "bf_emma" and _FakeKokoro.calls[-1]["lang"] == "en-gb"
    eng = engine_for("af_heart:75,af_bella:25")
    list(eng.synthesize("Hello there."))
    style = _FakeKokoro.calls[-1]["voice"]
    assert np.allclose(style, 0.75 * 1.0 + 0.25 * 3.0) and _FakeKokoro.calls[-1]["lang"] == "en-us"
