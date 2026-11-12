"""A transcript that is noise, a Whisper hallucination or a low-confidence blip is dropped instead of answered."""
from __future__ import annotations

import pytest

from ev.stt.whisper import STT, Transcript, add_cuda_dll_dirs, looks_like_junk


def t(text: str, logprob: float = -0.3, no_speech: float = 0.05, audio_s: float = 2.0) -> Transcript:
    return Transcript(text, 0.5, audio_s, logprob, no_speech)


@pytest.mark.parametrize("tr", [
    t(""), t("   "), t("you"), t("You."), t("Thanks for watching!"), t("Thank you for watching."), t("[BLANK_AUDIO]"),
    t("Please subscribe"), t("um"), t("hmm"),
    t("You what?", logprob=-1.2),                          # the real failure: a short guess Whisper itself did not trust
    t("open", no_speech=0.7),
    t("a blip", audio_s=0.2),
    t("something long enough to look real", logprob=-1.4, no_speech=0.9),
])
def test_junk_is_dropped(tr):
    assert looks_like_junk(tr)


@pytest.mark.parametrize("tr", [
    t("pause"), t("mute"), t("next"), t("yes"), t("no"), t("bye"), t("thank you"), t("hello"),
    t("what time is it"), t("open spotify", logprob=-0.8), t("You what?", logprob=-0.4),
    t("what's the weather like in London tomorrow", logprob=-0.95),       # long, so a lower score is still a real sentence
])
def test_real_speech_is_kept(tr):
    assert not looks_like_junk(tr)


def test_gpu_failure_falls_back_to_the_cpu(cfg, monkeypatch):
    """The CUDA model loads without cuBLAS and only fails on the first decode: that must end up on the CPU, not as an error."""
    calls: list[tuple[str, str]] = []

    class Boom:
        def transcribe(self, *a, **k):
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    stt = STT(cfg)
    stt.c = {**stt.c, "device": "auto"}

    def build(device: str, compute: str):
        calls.append((device, compute))
        return Boom() if device == "cuda" else object()

    monkeypatch.setattr(stt, "_build", build)
    stt.load()
    assert stt.device == "cpu" and [c[0] for c in calls] == ["cuda", "cpu"]


def test_cpu_is_never_probed_for_a_gpu(cfg, monkeypatch):
    stt = STT(cfg)
    stt.c = {**stt.c, "device": "cpu"}
    seen: list[str] = []
    monkeypatch.setattr(stt, "_build", lambda d, c: seen.append(d) or object())
    stt.load()
    assert seen == ["cpu"] and stt.device == "cpu"


def test_a_working_gpu_is_used(cfg, monkeypatch):
    class Fine:
        def transcribe(self, *a, **k):
            return iter(()), None

    stt = STT(cfg)
    stt.c = {**stt.c, "device": "auto"}
    monkeypatch.setattr(stt, "_build", lambda d, c: Fine())
    stt.load()
    assert stt.device == "cuda"


def test_cuda_dll_folders_are_found_when_pip_installed_them(tmp_path, monkeypatch):
    (tmp_path / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "cudnn" / "bin").mkdir(parents=True)
    (tmp_path / "cuda_nvrtc" / "lib").mkdir(parents=True)                 # not a bin folder: ignored
    monkeypatch.setenv("PATH", "C:\base")
    added = add_cuda_dll_dirs([tmp_path])
    assert [p.split("\\")[-2] if "\\" in p else p.split("/")[-2] for p in added] == ["cublas", "cudnn"]
    assert add_cuda_dll_dirs([tmp_path / "nothing"]) == []
