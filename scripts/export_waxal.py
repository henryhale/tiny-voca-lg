"""Export two speakers of the Waxal Luganda dataset as 24 kHz WAV files.

Step 1 of training: the train and validation splits of one female and one male speaker, resampled
to 24 kHz, peaks kept below 0.99. Clips whose text has digits are skipped,
because the front end does not read digits.

    python scripts/export_waxal.py --output data

Writes DATA/clips/*.wav, DATA/clips-<speaker>.jsonl (the input of
split_clips.py, one file per speaker) and DATA/test_sentences.txt (four
sentences of the test split, for listening). WAV files that already exist are
not written again, so a stopped run can be restarted.
Needs: pip install -e ".[data]"
"""

import argparse
import json
from math import gcd
from pathlib import Path
import re

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000
DIGITS = re.compile(r"\d")   # the front end does not read digits


def to_24k(audio):
    x = np.asarray(audio["array"], dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0 if x.shape[0] < x.shape[-1] else 1)
    sr = int(audio["sampling_rate"])
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        g = gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g)
    peak = np.abs(x).max()
    if peak > 0.99:
        x *= 0.99 / peak
    return x.astype(np.float32)


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Data folder (DATA above)")
    parser.add_argument("--dataset", default="CraneAILabs/waxal-lug-clean")
    parser.add_argument("--female", default="5", help="speaker_id of the F2 voice (5 has the most audio)")
    parser.add_argument("--male", default="1", help="speaker_id of the M2 voice (1 has the most audio)")
    args = parser.parse_args()

    from datasets import load_dataset

    voices = {args.female: "F2", args.male: "M2"}
    ds = load_dataset(args.dataset)
    clips, skipped = {s: [] for s in voices}, 0
    (args.output / "clips").mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        part = ds[split].filter(lambda s: str(s) in voices, input_columns="speaker_id")
        for row in part:
            if DIGITS.search(row["text"]):
                skipped += 1
                continue
            name = f"{split}-{row['id']}"
            wav = args.output / "clips" / f"{name}.wav"
            if not wav.exists():
                sf.write(wav, to_24k(row["audio"]), SAMPLE_RATE)
            speaker = str(row["speaker_id"])
            clips[speaker].append({"id": name, "text": row["text"].strip(), "speaker": voices[speaker], "audio": str(wav)})
    for s, rows in clips.items():
        write_jsonl(args.output / f"clips-{s}.jsonl", rows)
        print(f"speaker {s} -> {voices[s]}: {len(rows)} clips")
    print(f"skipped {skipped} clips with digits in the text")
    tests = [t.strip() for t in ds["test"]["text"] if not DIGITS.search(t) and 30 <= len(t) <= 160][:4]
    (args.output / "test_sentences.txt").write_text("\n".join(tests) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
