"""Write a tiny-voca checkpoint as GGUF files.

    python scripts/export_gguf.py --checkpoint weights/tiny-voca-lug.pt --output-dir gguf

This writes one file per type, named like the upstream files on Hugging Face:

    tiny-voca-lg-f32.gguf   every tensor in float32 (the "orig" file)
    tiny-voca-lg-f16.gguf   matrices and kernels in float16
    tiny-voca-lg-q8_0.gguf  transformer matrices in Q8_0, the rest float32
    tiny-voca-lg-q4_0.gguf  transformer matrices in Q4_0, the rest float32

The layout follows what upstream ``js/tiny-voca.js`` reads (the audio.cpp style):

- tensors are named ``model/<state_dict key>``;
- ``config.json`` is embedded through the ``audiocpp.embedded_files.*`` keys and
  holds ``{"model": <config>, "vocab": {symbol: id}}``;
- extra ``tiny_voca.*`` keys record the front end, so a reader can tell this
  Luganda model from the English one.

Only numpy and torch are needed; the GGUF bytes are written by hand.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import struct

import numpy as np
import torch

from tiny_voca.config import Config
from tiny_voca.infer import match_frontend
from tiny_voca.text import SYMBOL_TO_ID, frontend_fingerprint, frontend_identity

ALIGNMENT = 32
MAGIC, VERSION = 0x46554747, 3

# GGUF metadata value types.
UINT8, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64 = 0, 5, 6, 7, 8, 9, 10
# ggml tensor types.
F32, F16, Q4_0, Q8_0 = 0, 1, 2, 8
TYPES = {"f32": F32, "f16": F16, "q8_0": Q8_0, "q4_0": Q4_0}


# --- quantization -------------------------------------------------------------

def q8_0(data):
    """Blocks of 32: a float16 scale, then 32 int8 values."""
    blocks = data.reshape(-1, 32).astype(np.float32)
    scale = np.abs(blocks).max(axis=1) / 127
    inverse = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)
    codes = np.round(blocks * inverse[:, None]).astype(np.int8)
    out = np.empty((len(blocks), 34), dtype=np.uint8)
    out[:, :2] = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = codes.view(np.uint8)
    return out.tobytes()


def q4_0(data):
    """Blocks of 32: a float16 scale, then 16 bytes. Value i sits in the low
    nibble of byte i for i < 16 and in the high nibble of byte i - 16 after."""
    blocks = data.reshape(-1, 32).astype(np.float32)
    largest = blocks[np.arange(len(blocks)), np.abs(blocks).argmax(axis=1)]
    scale = largest / -8
    inverse = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale != 0)
    codes = np.clip((blocks * inverse[:, None] + 8.5).astype(np.int32), 0, 15).astype(np.uint8)
    out = np.empty((len(blocks), 18), dtype=np.uint8)
    out[:, :2] = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = codes[:, :16] | (codes[:, 16:] << 4)
    return out.tobytes()


def pack(name, array, wanted):
    """The ggml type and bytes for one tensor. Vectors (norms, biases) stay
    float32; quantized types apply to the transformer matrices only, as in
    upstream ``js/pack.mjs``."""
    array = np.ascontiguousarray(array, dtype=np.float32)
    if wanted == F16 and array.ndim >= 2:
        return F16, array.astype(np.float16).tobytes()
    if wanted in (Q8_0, Q4_0) and array.ndim == 2 and ".blocks." in name and array.shape[-1] % 32 == 0:
        return wanted, (q8_0 if wanted == Q8_0 else q4_0)(array)
    return F32, array.tobytes()


# --- GGUF writing -------------------------------------------------------------

def string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def value(kind, item):
    if kind == STRING:
        return string(item)
    if kind == UINT64:
        return struct.pack("<Q", item)
    if kind == INT32:
        return struct.pack("<i", item)
    if kind == FLOAT32:
        return struct.pack("<f", item)
    if kind == BOOL:
        return struct.pack("<B", bool(item))
    if kind == UINT8:
        return struct.pack("<B", item)
    raise ValueError(kind)


def key(name, kind, item, inner=None):
    out = string(name) + struct.pack("<I", kind)
    if kind != ARRAY:
        return out + value(kind, item)
    out += struct.pack("<IQ", inner, len(item))
    if inner == UINT8:
        return out + bytes(item)
    return out + b"".join(value(inner, element) for element in item)


def padding(size):
    return -size % ALIGNMENT


def write_gguf(path, metadata, files, tensors):
    """``metadata``: list of (name, kind, value[, inner]); ``files``: name -> bytes;
    ``tensors``: list of (name, shape, ggml type, bytes)."""
    names = list(files)
    offsets = [0]
    for name in names:
        offsets.append(offsets[-1] + len(files[name]))
    entries = [key(*item) for item in metadata] + [
        key("audiocpp.embedded_files.names", ARRAY, names, STRING),
        key("audiocpp.embedded_files.offsets", ARRAY, offsets, UINT64),
        key("audiocpp.embedded_files.data", ARRAY, b"".join(files[name] for name in names), UINT8),
    ]

    infos, offset = [], 0
    for name, shape, kind, data in tensors:
        # GGUF lists dimensions fastest first: the reverse of the torch shape.
        infos.append(string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", n) for n in reversed(shape))
                     + struct.pack("<IQ", kind, offset))
        offset += len(data) + padding(len(data))

    header = struct.pack("<IIQQ", MAGIC, VERSION, len(tensors), len(entries)) + b"".join(entries) + b"".join(infos)
    with open(path, "wb") as out:
        out.write(header + b"\0" * padding(len(header)))
        for _, _, _, data in tensors:
            out.write(data + b"\0" * padding(len(data)))


# --- checkpoint ---------------------------------------------------------------

def convert(checkpoint, output_dir, name, types):
    frontend = match_frontend(checkpoint)
    if frontend is None:
        raise SystemExit("no front end in this code base matches the checkpoint fingerprint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = asdict(Config.from_dict(state["config"]))
    config["speakers"] = list(config["speakers"])
    settings = {"model": config, "vocab": dict(SYMBOL_TO_ID)}
    files = {"config.json": json.dumps(settings, ensure_ascii=False, indent=1).encode("utf-8")}
    identity = frontend_identity()

    metadata = [
        ("general.architecture", STRING, "tiny-voca"),
        ("general.name", STRING, name),
        ("general.alignment", UINT64, ALIGNMENT),
        ("general.license", STRING, "cc-by-sa-4.0"),   # the weights; the code is Apache-2.0
        ("tiny_voca.sample_rate", UINT64, config["sample_rate"]),
        ("tiny_voca.hop", UINT64, config["hop"]),
        ("tiny_voca.voices", ARRAY, config["speakers"], STRING),
        ("tiny_voca.symbols", ARRAY, list(SYMBOL_TO_ID), STRING),
        ("tiny_voca.frontend", STRING, identity.get("phonemizer", frontend)),
        ("tiny_voca.frontend_fingerprint", STRING, frontend_fingerprint()),
    ]
    if state.get("step") is not None:
        metadata.append(("tiny_voca.step", UINT64, int(state["step"])))

    weights = {k: v.detach().float().numpy() for k, v in state["model"].items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in types:
        tensors, packed = [], 0
        for key_name, array in weights.items():
            kind, data = pack(key_name, array, TYPES[label])
            packed += kind != F32
            tensors.append(("model/" + key_name, array.shape, kind, data))
        path = output_dir / f"{name}-{label}.gguf"
        write_gguf(path, metadata + [("tiny_voca.file_type", STRING, label)], files, tensors)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(tensors)} tensors, {packed} as {label})")


def main():
    parser = argparse.ArgumentParser(description="Export a tiny-voca checkpoint to GGUF")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gguf"))
    parser.add_argument("--name", default="tiny-voca-lg", help="file name prefix and general.name")
    parser.add_argument("--types", nargs="+", choices=list(TYPES), default=list(TYPES))
    args = parser.parse_args()
    convert(args.checkpoint, args.output_dir, args.name, args.types)


if __name__ == "__main__":
    main()
