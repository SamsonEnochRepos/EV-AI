# Training the EV wake word

One small openWakeWord model (`hey_evie.onnx`, ~100k parameters) that answers to **"hey evie", "hey EV", "evie" and
"EV"**. EV's `WakeWord` class already loads any custom `.onnx` through `trigger.wakeword_model`, so training only has to
produce that file.

## How it works

The shared openWakeWord melspectrogram + embedding models turn 2 s of audio into one 16x96 window. We train only the
small classifier head on those windows, using

* **positives**: Piper speech, with EV's own `hfc_female` voice as the largest share (45%), plus amy, lessac, ryan, alan
  and `libritts_r` (900+ speakers), with random speed, pitch, noise, reverb and mic EQ;
* **negatives**: near-miss words ("every", "Eddie", "TV", "hey Siri", ...), generic Piper speech from the *same* voice
  mix (so "synthetic voice" is not the giveaway), real LibriSpeech speech, and ESC-50 noise;
* **your own voice** (optional, strongly recommended): `record.py`.

## Run it (Colab / Kaggle, recommended)

`train_cloud.ipynb` does everything. Upload `data/wakeword/ev_wakeword_kit.zip` (scripts + any recordings) and run the
cells; download `hey_evie.onnx` at the end. It needs ~1-2 h on CPU and no GPU.

## Run it locally

Needs ~8 GB+ RAM free; the machine this was built on has 7 GB, hence the notebook.

```powershell
.venv\Scripts\python scripts/wakeword/fetch_data.py                 # LibriSpeech + ESC-50 (~1.3 GB)
.venv\Scripts\python scripts/wakeword/synth.py --workers 3          # Piper clips
.venv\Scripts\python scripts/wakeword/features.py                   # augment + embed
data\wakeword\venv\Scripts\python scripts/wakeword/train.py         # torch lives in its own venv
.venv\Scripts\python scripts/wakeword/evaluate.py                   # recall, false triggers/hour
```

`evaluate.py` runs the model through `openwakeword.Model` in 80 ms frames, like EV does, against voices and noise it
never saw, and prints a threshold table; pick `trigger.wakeword_threshold` from it.

## Record your own voice

```powershell
.venv\Scripts\python scripts/wakeword/record.py
```

Then re-run `features.py` and `train.py` (or re-zip the kit so the cloud run includes your clips). Synthetic voices are
a good start, but a real mic and a real voice are what the model is judged on.
