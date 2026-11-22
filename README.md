# tiny-voca-lg

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/henryhale/tiny-voca-lg)
[![Base model](https://img.shields.io/badge/base%20model-n4ze3m%2Ftiny--voca-yellow?logo=huggingface)](https://huggingface.co/n4ze3m/tiny-voca)
[![Dataset](https://img.shields.io/badge/dataset-waxal--lug--clean-orange?logo=huggingface)](https://huggingface.co/datasets/CraneAILabs/waxal-lug-clean)
[![Language: Luganda](https://img.shields.io/badge/language-Luganda-green)](#how-luganda-is-read)
[![Parameters: 4.5M](https://img.shields.io/badge/parameters-4.5M-lightgrey)](#features)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2017-005CED?logo=onnx)](#onnx)

tiny-voca-lg is a small Luganda text-to-speech model. It is
[tiny-voca](https://github.com/n4ze3m/tiny-voca), a 4.5M-parameter English
model, fine-tuned on recorded Luganda speech. It has two voices and runs on a
CPU or in the browser.

This repository is a fork of tiny-voca with the Luganda changes built in:

- a Luganda front end (`src/tiny_voca/lug.py`) that turns spelling into
  phonemes with plain rules, so no dictionary, eSpeak or dizione is needed;
- `--init-weights`, to start training from the released English weights;
- data-parallel training on several GPUs (`torchrun`);
- alignment tuned for Luganda vowels;
- a browser demo that reads Luganda (`web/`).

## Features

- **Small**: 4,537,291 parameters, 18 MB, waveform generator included.
- **Two voices**: F2 (female) and M2 (male), Luganda, 24 kHz.
- **Runs anywhere**: PyTorch, ONNX Runtime, and ONNX Runtime Web.
- **No phonemizer to install**: Luganda spelling is close to phonemic, and the
  rules fit in about 100 lines of Python (and the same in JavaScript).
- **Cheap to make**: about 3.2 hours of speech and one hour of training on two
  free Kaggle T4 GPUs.

## Install

```bash
git clone https://github.com/henryhale/tiny-voca-lg.git
cd tiny-voca-lg
pip install -e .
hf download henryhale/tiny-voca-lg --local-dir weights
```

## Usage

```bash
python -m tiny_voca.infer \
  --checkpoint weights/tiny-voca-lug.pt \
  --voice F2 \
  --text "Oli otya? Nkwagala nnyo, ssebo." \
  --output oli-otya.wav
```

The command finds the front end the checkpoint was trained with (`lug`) by
itself. In your own Python code, select it first:

```python
from tiny_voca.text import set_frontend
set_frontend("lug")             # or set TINY_VOCA_FRONTEND=lug

import soundfile as sf
from tiny_voca.infer import load_model, speak_text

model = load_model("weights/tiny-voca-lug.pt")
wave, spoken = speak_text(model, "Webale nnyo okujja. Weeraba!", "M2")
sf.write("webale.wav", wave.numpy(), 24000)
```

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a file instead of `--text` |
| `--voice` | `F2` | `F2` or `M2` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Raise or lower the pitch |
| `--pitch-range` | 1.0 | More or less pitch movement |

Long text is fine. It is read one sentence at a time and the audio is joined.
A text that ends without a mark gets a full stop, so the last word does not
break off. A sentence with nothing to read, such as one in another script, is
skipped. The output never peaks above 0.99, so it does not clip as a 16-bit
WAV.

## How Luganda is read

`lug.py` maps spelling onto the phonemes the English model already knows:

| Spelling | Phonemes | Example |
|---|---|---|
| a, e, i, o, u | ɐ, ɛ, i, ɔ, ʊ | *bala* → b ɐ l ɐ |
| aa, ee, ii, oo, uu (long vowels) | ɑː, eɪ, iː, oː, uː | *kitaawe* → k i t ɑː w ɛ |
| double consonant (geminate) | the phone twice | *bajja* → b ɐ dʒ dʒ ɐ |
| ny | n j | *nnyo* → n n j ɔ |
| ng' or ŋ | ŋ | *ng'ombe* → ŋ ɔ m b ɛ |
| ky, gy | tʃ, dʒ | |
| r | ɾ | |
| other apostrophes (elision) | dropped | *n'abaana* → n ɐ b ɑː n ɐ |

Tone is not written in Luganda and is not modelled. Digits are not read:
write numbers out in words. To see the phonemes of a text:

```bash
TINY_VOCA_FRONTEND=lug python -c "from tiny_voca.text import phonemize; print(' '.join(phonemize('Oli otya?').symbols))"
```

The rule tables are part of the checkpoint's front-end fingerprint. If you
change them, the released weights no longer load, and you have to train again.

## ONNX

```bash
python -m tiny_voca.export --checkpoint weights/tiny-voca-lug.pt --output tiny-voca-lug.onnx
```

A ready file is in the weights repository as `model.onnx`, with its metadata
in `model.json`. The graph takes phoneme ids and returns the waveform:

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tiny_voca.text import phonemize, set_frontend

set_frontend("lug")
phones = phonemize("Oli otya? Nkwagala nnyo.")
one = lambda value: np.array([value], dtype=np.float32)

session = ort.InferenceSession("weights/model.onnx")
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),      # 0 = F2, 1 = M2
    "speed": one(1.0), "pitch_shift": one(0.0), "pitch_range": one(1.0),
})
audio = audio[0] * min(1.0, 0.99 / np.abs(audio).max())
sf.write("out.wav", audio, 24000)
```

The model reads one utterance per call, up to 512 tokens and 32 seconds.
Split longer text into sentences; `tiny_voca.infer.speak_text` does this.

## Web demo

`web/` runs the model in the browser with ONNX Runtime Web. It needs no
Python and no Pyodide: `web/public/lug.js` is the Luganda front end in plain
JavaScript and gives the same ids as `lug.py`. The model (21 MB) is loaded
from Hugging Face on the first visit and then kept in the browser's cache.
Each sentence plays as soon as it is made, and the whole clip stays in the
player.

Any static file server works:

```bash
cd web/public
python3 -m http.server 8000
```

Then open http://localhost:8000. To deploy it on Cloudflare, run
`npx wrangler deploy` in `web/`. To serve the model files yourself, run
`bash web/fetch_assets.sh` and set `BASE = "assets/"` in
`web/public/worker.js`.

## Training

To train on Kaggle with two free T4 GPUs, see [notebooks/](notebooks/).

The steps from the command line, with a GPU:

```bash
pip install -e ".[data,prepare,annotate]"
export TINY_VOCA_FRONTEND=lug
hf download n4ze3m/tiny-voca tiny-voca.pt --local-dir english   # the starting weights

# 1. Two speakers of CraneAILabs/waxal-lug-clean at 24 kHz: speaker 5 as F2, speaker 1 as M2
python scripts/export_waxal.py --output data

# 2. Cut the long recordings (about 43 s) into 4-18 s chunks at pauses between words
python scripts/split_clips.py --clips data/clips-5.jsonl --output data/chunks-5
python scripts/split_clips.py --clips data/clips-1.jsonl --output data/chunks-1

# 3. Keep the 100 minutes per voice the aligner is surest of; one sentence in 25 for validation
python scripts/make_manifests.py --data data

# 4. Acoustic features
python -m tiny_voca.prepare --inputs data/train.jsonl --split train --output features
python -m tiny_voca.prepare --inputs data/validation.jsonl --split validation --output features

# 5. Phoneme durations by forced alignment
python -m tiny_voca.annotate --features features --output annotations \
  --dataset-revision CraneAILabs/waxal-lug-clean --splits train validation \
  --min-token-probability 0.35 --max-weak-phone-fraction 0.5 --max-phone-frames 90

# 6. Fine-tune from the English weights (two GPUs; on one, use python -m tiny_voca.train)
torchrun --standalone --nproc_per_node 2 -m tiny_voca.train \
  --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output runs/lug --init-weights english/tiny-voca.pt \
  --steps 4000 --learning-rate 2e-4 --warmup 300 \
  --gan-start 1000 --gan-ramp 1000 \
  --teacher-hold 500 --teacher-ramp 1500 --teacher-final 0 \
  --batch-frames 9000 --save-every 500 --keep-every 1000 --log-every 50

# 7. Inference checkpoint and ONNX
python scripts/export_checkpoint.py --checkpoint runs/lug/step-00004000.pt --output tiny-voca-lug.pt
python -m tiny_voca.export --checkpoint tiny-voca-lug.pt --output model.onnx
```

Every step can be restarted: `split_clips.py` skips finished clips, `prepare`
and `annotate` take `--resume`, and training continues with
`--resume runs/lug/last.pt` (pass the same `--init-weights`). If a GPU runs
out of memory, use `--batch-frames 6000`. The 2e-4 learning rate was chosen for
two GPUs; on one GPU, a lower rate such as 1e-4 is probably better, but it has
not been tried.

The released checkpoint is step 4,000: 3.2 hours of aligned speech (95
minutes F2, 96 minutes M2), about an hour on two T4s.

## Tests

```bash
PYTHONPATH=src:tests python -m unittest discover -s tests
```

`tests/test_lug.py` checks the front end against the tokens the model was
trained on. When node is installed, it also checks that `web/public/lug.js`
gives the same ids as Python.

## Limits

- Luganda only, in Latin letters. Two fixed voices. No voice cloning.
- Pronunciation comes from spelling rules. Tone and other things the spelling
  does not show are not modelled.
- Digits are not read: write numbers out in words.
- English loanwords and names are read with Luganda rules.
- Trained on about 3 hours from two speakers, so expect some mistakes in
  rhythm and sound quality.
- No formal evaluation or listening test has been run.

## Credits

- [tiny-voca](https://github.com/n4ze3m/tiny-voca) by Muhammed Nazeem: the
  model, the code and the English weights this is fine-tuned from
  (Apache-2.0).
- [CraneAILabs/waxal-lug-clean](https://huggingface.co/datasets/CraneAILabs/waxal-lug-clean),
  from WaxalNLP: the Luganda recordings (CC BY 4.0).
- `facebook/wav2vec2-lv-60-espeak-cv-ft`: phoneme alignment of the training
  data only. It is not part of the model.

## License

Apache-2.0 for the code, as upstream. The training data is CC BY 4.0.

## Citation

Please cite upstream tiny-voca:

```bibtex
@software{nazeem2026tinyvoca,
  author = {Nazeem, Muhammed},
  title  = {tiny-voca: a 4.5M-parameter two-voice English text-to-speech model},
  year   = {2026},
  url    = {https://huggingface.co/n4ze3m/tiny-voca}
}
```
