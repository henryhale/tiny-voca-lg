---
license: apache-2.0
language:
- lg
pipeline_tag: text-to-speech
library_name: pytorch
base_model: n4ze3m/tiny-voca
datasets:
- CraneAILabs/waxal-lug-clean
tags:
- text-to-speech
- luganda
- onnx
---
# tiny-voca-lg

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Code](https://img.shields.io/badge/code-GitHub-181717?logo=github)](https://github.com/henryhale/tiny-voca-lg)
[![Base model](https://img.shields.io/badge/base%20model-n4ze3m%2Ftiny--voca-yellow?logo=huggingface)](https://huggingface.co/n4ze3m/tiny-voca)
[![Dataset](https://img.shields.io/badge/dataset-waxal--lug--clean-orange?logo=huggingface)](https://huggingface.co/datasets/CraneAILabs/waxal-lug-clean)
[![Language: Luganda](https://img.shields.io/badge/language-Luganda-green)](https://github.com/henryhale/tiny-voca-lg#how-luganda-is-read)
[![Parameters: 4.5M](https://img.shields.io/badge/parameters-4.5M-lightgrey)](#files)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2017-005CED?logo=onnx)](#onnx)

A 4.5M-parameter Luganda text-to-speech model with two voices: F2 (female) and
M2 (male). It is fine-tuned from the English
[n4ze3m/tiny-voca](https://huggingface.co/n4ze3m/tiny-voca) (Apache-2.0) on
[CraneAILabs/waxal-lug-clean](https://huggingface.co/datasets/CraneAILabs/waxal-lug-clean)
(CC BY 4.0, from WaxalNLP): speaker 5 as F2 and speaker 1 as M2.

- Code, Luganda front end and browser demo: https://github.com/henryhale/tiny-voca-lg
- Training notebooks: https://github.com/henryhale/tiny-voca-lg/tree/master/notebooks

| | |
|---|---|
| Parameters | 4,537,291 |
| Voices | F2, M2 (Luganda) |
| Output | 24 kHz mono |
| Input | Luganda text, converted to phonemes with spelling rules (`lug.py`) |
| Checkpoint | step 4,000 |
| License | Apache-2.0 |

## Files

| File | Contents |
|---|---|
| `tiny-voca-lug.pt` | PyTorch weights, float32 (18 MB) |
| `model.onnx` | ONNX export, float32, opset 17 (21.4 MB) |
| `model.json` | Sample rate, voices, phoneme symbols and front-end fingerprint for `model.onnx` |

## Usage

### PyTorch

```bash
pip install git+https://github.com/henryhale/tiny-voca-lg
hf download henryhale/tiny-voca-lg --local-dir tiny-voca-lg

python -m tiny_voca.infer \
  --checkpoint tiny-voca-lg/tiny-voca-lug.pt \
  --voice F2 \
  --text "Oli otya? Nkwagala nnyo." \
  --output out.wav
```

The package installs as `tiny_voca`, like upstream tiny-voca, so install it
in its own environment. The command picks the Luganda front end from the
checkpoint by itself. In Python, call `tiny_voca.text.set_frontend("lug")` (or
set `TINY_VOCA_FRONTEND=lug`) before you load the model.

<details>
<summary>Command options</summary>

| Option | Default | Meaning |
|---|---|---|
| `--text-file` | | Read the text from a file instead of `--text`. |
| `--voice` | `F2` | `F2` or `M2` |
| `--speed` | 1.0 | Speaking rate |
| `--pitch-shift` | 0 | Added to log F0. 0.058 is about one semitone. |
| `--pitch-range` | 1.0 | Scales pitch movement around the utterance mean. |

</details>

### ONNX

The graph takes phoneme ids and stress values and returns the waveform. Text
is converted to ids outside the graph; the ids are positions in `symbols` in
`model.json`.

<details>
<summary>Run the ONNX model from Python</summary>

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tiny_voca.text import phonemize, set_frontend

set_frontend("lug")
phones = phonemize("Oli otya? Nkwagala nnyo.")
one = lambda value: np.array([value], dtype=np.float32)

session = ort.InferenceSession("tiny-voca-lg/model.onnx")
audio, durations = session.run(None, {
    "tokens": np.array([phones.ids], dtype=np.int64),
    "stress": np.array([phones.stress], dtype=np.int64),
    "voice": np.array([0], dtype=np.int64),      # 0 = F2, 1 = M2
    "speed": one(1.0),
    "pitch_shift": one(0.0),
    "pitch_range": one(1.0),
})
audio = audio[0]
audio = audio * min(1.0, 0.99 / np.abs(audio).max())    # the model can peak above 1.0
sf.write("out.wav", audio, 24000)
```

</details>

The model reads one utterance per call, up to 512 tokens and 32 seconds. Split
longer text into sentences; `tiny_voca.infer.speak_text` does this in Python.
End each utterance with a mark such as a full stop.

### Browser

The same `model.onnx` runs in onnxruntime-web. The GitHub repository has a demo
page (`web/`) with the Luganda front end in plain JavaScript (`lug.js`), so
the browser needs no Python.

## Training

| | |
|---|---|
| Data | 3.2 hours after cutting and alignment: 95 min F2, 96 min M2; 1,040 training and 39 validation chunks of 4–18 s |
| Starting point | The English tiny-voca weights (`--init-weights`) |
| Steps | 4,000, batch 9,000 frames per GPU, learning rate 2e-4 |
| Hardware | Two Kaggle T4 GPUs, data-parallel, about 1 hour |

The long Waxal recordings were cut at pauses between words. Phoneme durations
come from forced alignment with `facebook/wav2vec2-lv-60-espeak-cv-ft`, which
is used only to label the training data. Luganda spelling is mapped onto the
English model's phoneme inventory, so no new symbols were added.

## Limitations

- Pronunciation comes from spelling rules. Tone is not written in Luganda and
  is not modelled.
- Digits are not read: write numbers out in words.
- English loanwords and names are read with Luganda rules.
- Trained on about 3 hours from two speakers. No formal evaluation or
  listening test has been run.

## License and attribution

The weights are fine-tuned from tiny-voca (Apache-2.0, Muhammed Nazeem). The
training data is CraneAILabs/waxal-lug-clean (CC BY 4.0), derived from
WaxalNLP.
