#!/usr/bin/env python3
"""Stream-graft exact 7-step one-hot adaLN lookup vectors onto any ComfyUI curve-format
MiniMax-H3-Image DiT checkpoint (BF16, INT8 ConvRot, or NVFP4) without loading the full
checkpoint into RAM.

Usage:
    python3 tools/build_exact_dit.py \
        --src minimax_h3_image_pruned_bf16.safetensors \
        --exact-vectors build/adaln_exact_vectors.safetensors \
        --out build/minimax_h3_image_exact_adaln.safetensors
"""
import argparse
import json
import math
import os
import struct

import torch
from safetensors import safe_open

GRID = 1025
CHUNK = 128 << 20  # 128 MiB streaming buffer


def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return n, raw, json.loads(raw)


def dump_header(h: dict, align: int = 8) -> bytes:
    s = json.dumps(h, separators=(",", ":")).encode()
    return s + b" " * ((-len(s)) % align)


def build_one_hot_table(t: torch.Tensor) -> torch.Tensor:
    tt = [float(x) for x in t]
    table = torch.zeros(GRID, len(tt), dtype=torch.float32)
    for r in range(GRID):
        p = r / (GRID - 1)
        table[r, min(range(len(tt)), key=lambda i: abs(p - tt[i]))] = 1.0
    for i, ti in enumerate(t):
        pos = float(ti) * (GRID - 1)
        i0 = min(math.floor(pos), GRID - 2)
        for r in (i0, i0 + 1):
            assert table[r].argmax() == i and table[r].sum() == 1.0, (i, r)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Input curve-format H3-Image safetensors file")
    ap.add_argument("--exact-vectors", required=True, help="adaln_exact_vectors.safetensors from adaln_exact.py")
    ap.add_argument("--out", required=True, help="Output exact-adaLN safetensors path")
    args = ap.parse_args()

    n, _, h = read_header(args.src)
    meta = h.get("__metadata__", {})
    names = sorted([k for k in h if k != "__metadata__"], key=lambda k: h[k]["data_offsets"][0])
    data_start = 8 + n

    vf = safe_open(args.exact_vectors, framework="pt")
    t = vf.get_tensor("t_official")
    assert t.shape == (7,), f"Expected 7 timesteps, got {t.shape}"
    table = build_one_hot_table(t)

    new = {"adaln_t_table": table}
    for k in [kk for kk in names if kk.endswith(".adaln_proj.linear.weight")]:
        key = k.replace(".adaln_proj.linear.weight", "")
        A = vf.get_tensor(key + ".A")
        assert A.dtype == torch.float32 and A.shape[0] == 7
        assert torch.equal(A, A.to(torch.bfloat16).to(torch.float32)), f"{key}.A is not BF16-exact"
        new[k] = A.T.contiguous()
        new[k.replace(".weight", ".bias")] = torch.zeros(A.shape[1], dtype=torch.float32)
    assert len(new) == 103, len(new)

    oh = {"__metadata__": meta}
    off = 0
    for k in names:
        if k in new:
            v = new[k]
            nb = v.numel() * 4
            oh[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + nb]}
        else:
            a, b = h[k]["data_offsets"]
            nb = b - a
            oh[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + nb]}
        off += nb

    hdr = dump_header(oh)
    tmp = args.out + ".partial"
    with open(args.src, "rb") as src, open(tmp, "wb") as dst:
        dst.write(struct.pack("<Q", len(hdr)))
        dst.write(hdr)
        for k in names:
            if k in new:
                dst.write(new[k].numpy().tobytes())
                continue
            a, b = h[k]["data_offsets"]
            src.seek(data_start + a)
            left = b - a
            while left:
                buf = src.read(min(CHUNK, left))
                assert buf, k
                dst.write(buf)
                left -= len(buf)
        dst.flush()
        os.fsync(dst.fileno())

    size = os.path.getsize(tmp)
    assert size == 8 + len(hdr) + off, (size, len(hdr), off)
    os.replace(tmp, args.out)
    print(f"Wrote {args.out} ({size} bytes, {len(names)} tensors, 103 exact adaLN tensors replaced)")


if __name__ == "__main__":
    main()
