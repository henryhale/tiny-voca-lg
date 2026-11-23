"""Speak text with a GGUF file made by ``export_gguf.py``.

    python scripts/infer_gguf.py --gguf gguf/tiny-voca-lg-q8_0.gguf \
      --voice F2 --text "Oli otya? Nkwagala nnyo, ssebo." --output out.wav

The file is read here, its tensors are turned back into float32 (f32, f16,
q8_0 and q4_0 are understood) and loaded into the PyTorch model, so the
options are the same as ``python -m tiny_voca.infer``.
"""

import argparse
import json
from pathlib import Path
import struct

import numpy as np
import soundfile as sf
import torch

from tiny_voca.config import Config
from tiny_voca.infer import speak_text
from tiny_voca.model import TinyVoca
from tiny_voca.text import FRONTENDS, frontend_fingerprint, set_frontend

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    """(metadata, {name: float32 array}, {embedded file name: bytes})."""
    data = Path(path).read_bytes()
    at = 0

    def take(fmt):
        nonlocal at
        (item,) = struct.unpack_from(fmt, data, at)
        at += struct.calcsize(fmt)
        return item

    def string():
        nonlocal at
        size = take("<Q")
        at += size
        return data[at - size:at].decode("utf-8")

    def value(kind):
        nonlocal at
        if kind == 8:
            return string()
        if kind == 9:
            inner, count = take("<I"), take("<Q")
            if inner in (0, 1):
                at += count
                return data[at - count:at]
            return [value(inner) for _ in range(count)]
        return take(SCALARS[kind])

    if take("<I") != 0x46554747:
        raise ValueError(f"{path}: not a GGUF file")
    take("<I")
    tensor_count, key_count = take("<Q"), take("<Q")
    meta = {}
    for _ in range(key_count):
        name = string()
        meta[name] = value(take("<I"))
    infos = []
    for _ in range(tensor_count):
        name = string()
        shape = [take("<Q") for _ in range(take("<I"))][::-1]
        infos.append((name, shape, take("<I"), take("<Q")))
    alignment = meta.get("general.alignment", 32)
    start = -(-at // alignment) * alignment
    tensors = {name: unpack(data, start + offset, kind, shape) for name, shape, kind, offset in infos}

    names, offsets = meta.get("audiocpp.embedded_files.names", []), meta.get("audiocpp.embedded_files.offsets")
    files = {name: meta["audiocpp.embedded_files.data"][offsets[i]:offsets[i + 1]] for i, name in enumerate(names)}
    return meta, tensors, files


def unpack(data, start, kind, shape):
    count = int(np.prod(shape)) if shape else 1
    if kind == 0:  # F32
        out = np.frombuffer(data, np.float32, count, start)
    elif kind == 1:  # F16
        out = np.frombuffer(data, np.float16, count, start).astype(np.float32)
    elif kind == 8:  # Q8_0: f16 scale + 32 int8
        blocks = np.frombuffer(data, np.uint8, count // 32 * 34, start).reshape(-1, 34)
        scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        out = scale * blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    elif kind == 2:  # Q4_0: f16 scale + 16 bytes of nibbles, low halves first
        blocks = np.frombuffer(data, np.uint8, count // 32 * 18, start).reshape(-1, 18)
        scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        nibbles = blocks[:, 2:]
        out = scale * (np.concatenate([nibbles & 15, nibbles >> 4], axis=1).astype(np.float32) - 8)
    else:
        raise ValueError(f"tensor type {kind} is not supported")
    return out.reshape(shape).copy()


def select_frontend(fingerprint):
    for name in FRONTENDS:
        set_frontend(name)
        try:
            if frontend_fingerprint() == fingerprint:
                return name
        except ImportError:
            continue
    raise SystemExit("no front end in this code base matches the file's tiny_voca.frontend_fingerprint")


def load_gguf(path, device="cpu"):
    meta, tensors, files = read_gguf(path)
    select_frontend(meta["tiny_voca.frontend_fingerprint"])
    settings = json.loads(files["config.json"])
    model = TinyVoca(Config.from_dict(settings["model"]))
    state = {name[len("model/"):]: torch.from_numpy(array) for name, array in tensors.items() if name.startswith("model/")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description="Speak text with a tiny-voca GGUF file")
    parser.add_argument("--gguf", type=Path, required=True)
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    parser.add_argument("--voice", default="F2")
    parser.add_argument("--output", type=Path, default=Path("out.wav"))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--pitch-range", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_gguf(args.gguf)
    if args.voice not in model.config.speakers:
        raise SystemExit(f"unknown voice {args.voice!r}; this file has {', '.join(model.config.speakers)}")
    content = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    with torch.no_grad():
        wave, _ = speak_text(model, content, args.voice, speed=args.speed, pitch_shift=args.pitch_shift,
                             seed=args.seed, pitch_range=args.pitch_range)
    sf.write(args.output, wave.numpy(), model.config.sample_rate)
    print(f"wrote {args.output} ({len(wave) / model.config.sample_rate:.1f} s) from {args.gguf.name}")


if __name__ == "__main__":
    main()
