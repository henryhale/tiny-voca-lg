"""Choose the training audio and write the train and validation manifests.

Step 3 of training: for each voice, keep the chunks the aligner was surest of until
--minutes of audio are kept, then put one sentence in 25 into validation. The
split is made by the text, so the same sentence never lands in both splits.

    python scripts/make_manifests.py --data data

Reads DATA/chunks-<speaker>/chunks.jsonl (from split_clips.py) and writes
DATA/train.jsonl and DATA/validation.jsonl, the --inputs of
`python -m tiny_voca.prepare`. The manifests are written once: an existing train.jsonl is left alone, so a
restarted run sees identical inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Data folder (DATA above)")
    parser.add_argument("--female", default="5", help="speaker_id of the F2 voice")
    parser.add_argument("--male", default="1", help="speaker_id of the M2 voice")
    parser.add_argument("--minutes", type=float, default=100, help="Training audio per voice")
    args = parser.parse_args()

    if (args.data / "train.jsonl").exists():
        raise SystemExit(f"{args.data / 'train.jsonl'} exists: delete train*.jsonl and validation*.jsonl to choose again")
    voices = {args.female: "F2", args.male: "M2"}
    rows = {"train": [], "validation": []}
    for s, voice in voices.items():
        chunks = sorted(read_jsonl(args.data / f"chunks-{s}" / "chunks.jsonl"), key=lambda r: (-r["confidence"], r["id"]))
        total = 0.0
        for r in chunks:
            if total >= args.minutes * 60:
                break
            total += r["seconds"]
            # split by text so the same sentence never lands in both splits
            bucket = int(hashlib.sha256(r["text"].strip().encode()).hexdigest()[:8], 16) % 25
            rows["validation" if bucket == 0 else "train"].append(
                {"id": r["id"], "text": r["text"], "voice_id": voice, "audio": r["audio"]})
        print(f"{voice}: {total / 60:.0f} min kept from {len(chunks)} chunks")
    for split, items in rows.items():
        write_jsonl(args.data / f"{split}.jsonl", items)
        print(split, len(items), "chunks")


if __name__ == "__main__":
    main()
