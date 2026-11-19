#!/usr/bin/env python3
"""Losslessly build the 51.51 GB layer-50 truncated Qwen3-VL text encoder for
MiniMax-H3-Image from the official 66.71 GB 14-shard MiniMaxAI/MiniMax-H3 text_encoder
using Python's standard library only (zero PyTorch RAM overhead).

Drops only the unused layers 50-63, final norm, and lm_head (156 tensors),
keeping embed_tokens, language layers 0-49, and the full vision tower (902 tensors).

Usage:
    python3 tools/build_te_layer50.py \
        --official-te-dir /path/to/MiniMax-H3/text_encoder \
        --out ./build/qwen3vl_32b_minimax_h3_bf16.safetensors
"""
import argparse
import glob
import hashlib
import json
import os
import re
import struct

CHUNK = 64 << 20


def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def comfy_name(official: str) -> str | None:
    if official.startswith("model.language_model."):
        rest = official[len("model.language_model."):]
        m = re.match(r"layers\.(\d+)\.", rest)
        if rest == "norm.weight" or (m and int(m.group(1)) >= 50):
            return None
        return "model." + rest
    if official.startswith("model.visual."):
        return "visual." + official[len("model.visual."):]
    if official == "lm_head.weight":
        return None
    raise SystemExit("Unmapped official tensor: " + official)


def dump_header(h: dict, align: int = 8) -> bytes:
    s = json.dumps(h, separators=(",", ":")).encode()
    return s + b" " * ((-len(s)) % align)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-te-dir", required=True, help="Directory holding model-*.safetensors shards")
    ap.add_argument("--out", required=True, help="Output 51.51 GB safetensors path")
    args = ap.parse_args()

    src = {}
    dropped = []
    for shard in sorted(glob.glob(os.path.join(args.official_te_dir, "model-*.safetensors"))):
        h, start = read_header(shard)
        for k, v in h.items():
            c = comfy_name(k)
            if c is None:
                dropped.append(k)
                continue
            src[c] = (shard, start, v)

    assert len(src) == 902 and len(dropped) == 156, (len(src), len(dropped))
    order = sorted(src.keys())

    oh = {
        "__metadata__": {
            "num_hidden_layers": "50",
            "output": "unnormalized_hidden_after_layer_50",
        }
    }
    off = 0
    for k in order:
        _, _, v = src[k]
        nb = v["data_offsets"][1] - v["data_offsets"][0]
        oh[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + nb]}
        off += nb

    hdr = dump_header(oh)
    sha = hashlib.sha256()
    tmp = args.out + ".partial"
    with open(tmp, "wb") as out:
        prefix = struct.pack("<Q", len(hdr)) + hdr
        out.write(prefix)
        sha.update(prefix)
        fds = {}
        for k in order:
            shard, start, v = src[k]
            f = fds.setdefault(shard, open(shard, "rb"))
            a, b = v["data_offsets"]
            f.seek(start + a)
            left = b - a
            while left:
                buf = f.read(min(CHUNK, left))
                assert buf, k
                out.write(buf)
                sha.update(buf)
                left -= len(buf)
        for f in fds.values():
            f.close()
        out.flush()
        os.fsync(out.fileno())

    os.replace(tmp, args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out)} bytes, sha256={sha.hexdigest()})")


if __name__ == "__main__":
    main()
