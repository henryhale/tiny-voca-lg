"""Cut long recordings into 4-18 s chunks at pauses between words.

tiny-voca accepts at most 32 s (and 512 phones) per clip, and the Waxal clips
average about 43 s. Each clip is force-aligned with the same phoneme recogniser
that annotation uses, then cut at the quietest point between two words, with a
preference for long pauses and sentence ends.

Usage: python scripts/split_clips.py --clips clips.jsonl --output DIR [--device cuda]
Input rows: {"id", "text", "speaker", "audio": path to a 24 kHz mono WAV}
Output: DIR/chunks.jsonl and DIR/wav/*.wav (24 kHz). Resumable: clips listed in
DIR/done.txt are skipped, and DIR/COMPLETE appears when every clip is done.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME = 0.02            # aligner frame, seconds
ONSET = 0.019           # a phone starts this long before its first emission frame
LEVEL_HOP = 0.01        # energy frames for finding the quietest cut point


def word_units(text, phonemize, phone_set, unk):
    """Split text into words that carry at least one phone; a word with none
    (a lone number or symbol) is glued to the previous word."""
    units = []
    for raw in text.split():
        try:
            phones = [s for s in phonemize(raw).symbols if s in phone_set or s == unk]
        except ValueError:
            phones = []
        if phones or not units:
            units.append({"text": raw, "phones": phones})
        else:
            units[-1]["text"] += " " + raw
    if units and not units[0]["phones"]:
        if len(units) == 1:
            return []
        units[1]["text"] = units[0]["text"] + " " + units[1]["text"]
        units = units[1:]
    return units


def level_db(wave, sample_rate):
    hop = int(LEVEL_HOP * sample_rate)
    frames = len(wave) // hop
    rms = np.sqrt(np.mean(wave[:frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-10)
    return 20 * np.log10(rms)


def boundaries(units, spans, level, duration):
    """Candidate cut points between words: (time, score, after_unit_index)."""
    result, start = [], 0
    starts, ends = [], []
    for unit in units:
        count = len(unit["phones"])
        starts.append(max(0.0, spans[start][0] * FRAME - ONSET))
        ends.append((spans[start + count - 1][1] + 1) * FRAME)
        start += count
    quiet = level.max() - 35.0
    for i in range(len(units) - 1):
        a, b = ends[i], starts[i + 1]
        lo, hi = int(a / LEVEL_HOP), int(b / LEVEL_HOP)
        if hi <= lo:
            cut, pause = b, 0.0
        else:
            window = level[lo:hi]
            cut = (lo + int(np.argmin(window))) * LEVEL_HOP
            pause = float((window < quiet).sum()) * LEVEL_HOP
        score = pause
        if re.search(r"[.!?]['\"”’)]*$", units[i]["text"]):
            score += 0.3
        elif re.search(r"[,;:]['\"”’)]*$", units[i]["text"]):
            score += 0.1
        result.append((min(max(cut, 0.0), duration), score, i))
    return result


def plan(units, spans, level, duration, shortest=4.0, longest=18.0):
    """Greedy cuts: each chunk ends at the best-scoring word gap in its window.
    Returns [(start_s, end_s, first_unit, last_unit)]."""
    gaps = boundaries(units, spans, level, duration)
    chunks, start, first = [], 0.0, 0
    while duration - start > longest:
        window = [g for g in gaps if g[2] >= first and start + shortest <= g[0] <= start + longest]
        if not window:
            # no usable word gap (a very long word run): skip to the next gap after the window
            later = [g for g in gaps if g[2] >= first and g[0] > start + longest]
            if not later:
                return chunks
            start, first = later[0][0], later[0][2] + 1
            continue
        cut, _, last = max(window, key=lambda g: (g[1], g[0]))
        chunks.append((start, cut, first, last))
        start, first = cut, last + 1
    if duration - start >= 1.0 and first < len(units):
        chunks.append((start, duration, first, len(units) - 1))
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-probability", type=float, default=0.25, help="Drop chunks the aligner is unsure of")
    args = parser.parse_args()

    os.environ.setdefault("TINY_VOCA_FRONTEND", "lug")
    from tiny_voca.align import PhoneRecognizer, token_spans
    from tiny_voca.text import PHONE_SET, UNK, phonemize, set_frontend
    set_frontend("lug")

    recognizer = PhoneRecognizer(args.device)
    (args.output / "wav").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.clips.read_text(encoding="utf-8").splitlines() if line.strip()]
    done_path, chunks_path = args.output / "done.txt", args.output / "chunks.jsonl"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    # Keep only chunks of fully finished clips (a clip interrupted half-way is redone).
    previous = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()] if chunks_path.exists() else []
    previous = [r for r in previous if r["source"] in done]
    chunks_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in previous), encoding="utf-8")
    kept, dropped, seconds = len(previous), 0, sum(r["seconds"] for r in previous)
    print(f"resuming: {len(done)} clips already done" if done else "starting", flush=True)
    started = time.time()
    with chunks_path.open("a", encoding="utf-8") as out, done_path.open("a") as progress:
        for n, row in enumerate(rows):
            if row["id"] in done:
                continue
            wave, sample_rate = sf.read(row["audio"], dtype="float32")
            duration = len(wave) / sample_rate
            units = word_units(row["text"], phonemize, PHONE_SET, UNK)
            labels = [recognizer.label(p) for unit in units for p in unit["phones"]]
            try:
                if not labels:
                    raise ValueError("no phones")
                log_probs = recognizer.log_probs(wave, sample_rate)
                spans = token_spans(log_probs, labels, recognizer.blank)
            except ValueError:
                dropped += 1
                progress.write(row["id"] + "\n")
                progress.flush()
                continue
            probability = np.array([float(np.exp(log_probs[a:b + 1, label]).mean()) for (a, b), label in zip(spans, labels)])
            level = level_db(wave, sample_rate)
            offsets = np.cumsum([0] + [len(u["phones"]) for u in units])
            for k, (start, end, first, last) in enumerate(plan(units, spans, level, duration)):
                confidence = float(probability[offsets[first]:offsets[last + 1]].mean())
                if confidence < args.min_probability or end - start < 1.0:
                    dropped += 1
                    continue
                piece = wave[int(start * sample_rate):int(end * sample_rate)]
                name = f"{row['id']}_{k:02d}"
                path = args.output / "wav" / f"{name}.wav"
                sf.write(path, piece, sample_rate)
                text = " ".join(u["text"] for u in units[first:last + 1])
                out.write(json.dumps({"id": name, "text": text, "speaker": row["speaker"], "audio": str(path),
                                      "seconds": round(end - start, 3), "confidence": round(confidence, 3), "source": row["id"]},
                                     ensure_ascii=False) + "\n")
                kept += 1
                seconds += end - start
            out.flush()
            progress.write(row["id"] + "\n")
            progress.flush()
            if (n + 1) % 25 == 0:
                print(f"{n + 1}/{len(rows)} clips, {kept} chunks, {seconds / 60:.1f} min, "
                      f"{time.time() - started:.0f} s", flush=True)
    (args.output / "COMPLETE").write_text("ok\n")
    print(f"done: {kept} chunks ({seconds / 60:.1f} min), {dropped} dropped this run", flush=True)


if __name__ == "__main__":
    main()