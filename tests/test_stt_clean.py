import numpy as np

from ev.stt.whisper import clean_audio


def test_quiet_speech_is_brought_up_and_dc_removed():
    t = np.arange(16000) / 16000
    x = (0.1 * np.sin(2 * np.pi * 300 * t) + 0.05).astype(np.float32)   # modest tone riding on a DC offset
    y = clean_audio(x, 16000)
    assert 0.8 < float(np.max(np.abs(y))) <= 1.0
    assert abs(float(np.mean(y))) < 0.01


def test_silence_is_not_amplified_into_noise():
    y = clean_audio(np.zeros(1600, np.float32), 16000)
    assert float(np.max(np.abs(y))) == 0.0
