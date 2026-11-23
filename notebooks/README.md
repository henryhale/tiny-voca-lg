# Notebooks

## train-kaggle.ipynb

Trains tiny-voca-lg on [Kaggle](https://www.kaggle.com) with two free T4 GPUs:
from the Waxal Luganda recordings to a packaged model (PyTorch, ONNX and model
card), in about two hours. It clones this repository and runs its code, with
the settings the released weights were trained with.

### Run it

1. On Kaggle, create a notebook and import `train-kaggle.ipynb` (*File* →
   *Import notebook*).
2. In the settings, choose Accelerator **GPU T4 ×2** and turn Internet **on**.
3. *Run All*.

### Settings

Cell 1 holds everything you may want to change.

<details>
<summary>All settings</summary>

| Setting | Default | Meaning |
|---|---|---|
| `REPO_REF` | `master` | Branch or tag of this repository to train with; pin a tag to repeat a run exactly |
| `FEMALE`, `MALE` | `5`, `1` | Waxal `speaker_id`s for the F2 and M2 voices |
| `MINUTES_PER_VOICE` | 100 | Training audio kept per voice |
| `STEPS` | 4000 | Training steps (one batch per GPU each) |
| `LEARNING_RATE` | 2e-4 | For two GPUs |
| `BATCH_FRAMES` | 9000 | Per GPU; drops to 6000 by itself if a T4 runs out of memory |
| `MAX_TRAIN_SECONDS` | 3 hours | Training time per session |
| `RESUME_FROM` | `None` | A previous version's output, to continue a stopped run |

</details>

Changing the speakers or the minutes after the data steps have run is refused:
delete `stages`, `data`, `features`, `annotations` and `run` in
`/kaggle/working` to start over.

### Steps

<details>
<summary>What each cell does</summary>

| Cell | Step | Code |
|---|---|---|
| 1–3 | Settings, code and downloads, front-end check | |
| 4 | Export the two speakers at 24 kHz | `scripts/export_waxal.py` |
| 5 | Cut long clips into 4–18 s chunks, one speaker per GPU | `scripts/split_clips.py` |
| 6 | Choose the audio, acoustic features | `scripts/make_manifests.py`, `tiny_voca.prepare` |
| 7 | Phoneme alignment | `tiny_voca.annotate` |
| 8 | Training, one model on both GPUs | `tiny_voca.train` |
| 9 | Listen to the last checkpoints | `scripts/export_checkpoint.py` |
| 10 | Package the chosen checkpoint | `tiny_voca.export`, `scripts/export_gguf.py` |

</details>

### Stopping and resuming

Everything is kept in `/kaggle/working/`, and every step resumes. A finished
step leaves a marker in `/kaggle/working/stages/` and is skipped next time. An
interrupted step continues where it stopped; training restarts from its last
save, at most 500 steps back.

- **Interactive session:** Settings → *Persistence* → **Files only**, then
  *Run All* again.
- **Saved version:** *Add Input* → your previous version's output, set
  `RESUME_FROM` in cell 1, then *Save & Run All*.

### After training

Listen to the samples that cell 9 plays. To keep an earlier checkpoint, set
`BEST` in cell 10 (for example `"step-00003000"`) and run it again. The model
ends up in `/kaggle/working/luganda/model/`:

- `tiny-voca-lug.pt`
- `model.onnx`
- `model.json`
- `gguf/` with the f32, f16, q8_0 and q4_0 files

The end of cell 10 has commented-out lines that upload the folder to Hugging
Face with an `HF_TOKEN` secret. The model card is kept on Hugging Face, and the
upload leaves it as it is.

To train without Kaggle, follow the command-line steps in the
[main README](../README.md#training).
