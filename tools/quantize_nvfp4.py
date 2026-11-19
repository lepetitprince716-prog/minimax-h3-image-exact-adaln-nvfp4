#!/usr/bin/env python3
"""Quantize minimax_h3_image_exact_adaln.safetensors (BF16) to Blackwell NVFP4
using ComfyUI's comfy-kitchen TensorCoreNVFP4Layout quantizer (streaming tensor-by-tensor).

Quantizes the 200 DiT block linear projections:
    blocks.0..49.{attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}
while keeping token_refiner, condition_proj, video_patch_proj, final_layer.video_out,
all norms, and the exact 7-step adaLN lookup table untouched in BF16 / FP32.

Usage:
    python3 tools/quantize_nvfp4.py \
        --src build/minimax_h3_image_exact_adaln.safetensors \
        --out build/minimax_h3_image_exact_adaln_nvfp4.safetensors
"""
import argparse
import json
import os
import re
import struct

import torch
from safetensors import safe_open
from comfy_kitchen.tensor import TensorCoreNVFP4Layout

TARGET_RE = re.compile(r"^blocks\.(\d+)\.(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))\.weight$")
CHUNK = 128 << 20


def dump_header(h: dict, align: int = 8) -> bytes:
    s = json.dumps(h, separators=(",", ":")).encode()
    return s + b" " * ((-len(s)) % align)


def torch_dtype_to_st(dt: torch.dtype) -> str:
    return {
        torch.uint8: "U8",
        torch.float8_e4m3fn: "F8_E4M3",
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
    }[dt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Input Exact-adaLN BF16 safetensors")
    ap.add_argument("--out", required=True, help="Output Exact-adaLN NVFP4 safetensors")
    ap.add_argument("--device", default="cuda", help="CUDA device for comfy_kitchen.quantize_nvfp4")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(hdr_len))
    data_start = 8 + hdr_len
    src_keys = sorted([k for k in h if k != "__metadata__"], key=lambda k: h[k]["data_offsets"][0])

    quant_layers = {}
    quant_blobs = {}
    sf = safe_open(args.src, framework="pt", device=args.device)

    for k in src_keys:
        if not TARGET_RE.match(k):
            continue
        prefix = k[:-7]  # strip ".weight"
        w = sf.get_tensor(k)
        qdata, params = TensorCoreNVFP4Layout.quantize(w)
        lmeta = {
            "format": "nvfp4",
            "group_size": 16,
            "orig_dtype": str(params.orig_dtype),
            "orig_shape": list(params.orig_shape),
        }
        quant_layers[prefix] = lmeta
        cq_bytes = json.dumps(lmeta).encode("utf-8")
        quant_blobs[prefix] = {
            f"{prefix}.comfy_quant": ("U8", [len(cq_bytes)], cq_bytes),
            f"{prefix}.weight": (
                "U8",
                list(qdata.view(torch.uint8).shape),
                qdata.view(torch.uint8).cpu().contiguous().numpy().tobytes(),
            ),
            f"{prefix}.weight_scale": (
                torch_dtype_to_st(params.block_scale.dtype),
                list(params.block_scale.shape),
                params.block_scale.view(torch.uint8).cpu().contiguous().numpy().tobytes(),
            ),
            f"{prefix}.weight_scale_2": (
                "F32",
                [],
                params.scale.to(torch.float32).cpu().contiguous().numpy().tobytes(),
            ),
        }

    assert len(quant_layers) == 200, f"Expected 200 quantized projections, got {len(quant_layers)}"

    meta = dict(h.get("__metadata__", {}))
    meta["_quantization_metadata"] = json.dumps({"layers": quant_layers})

    oh = {"__metadata__": meta}
    plan = []
    off = 0
    for k in src_keys:
        if TARGET_RE.match(k):
            prefix = k[:-7]
            for sub_k, (dt_s, shp, raw_b) in quant_blobs[prefix].items():
                nb = len(raw_b)
                oh[sub_k] = {"dtype": dt_s, "shape": shp, "data_offsets": [off, off + nb]}
                plan.append(("mem", raw_b))
                off += nb
        else:
            a, b = h[k]["data_offsets"]
            nb = b - a
            oh[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + nb]}
            plan.append(("copy", (a, b)))
            off += nb

    hdr = dump_header(oh)
    tmp = args.out + ".partial"
    with open(args.src, "rb") as src, open(tmp, "wb") as dst:
        dst.write(struct.pack("<Q", len(hdr)))
        dst.write(hdr)
        for kind, payload in plan:
            if kind == "mem":
                dst.write(payload)
            else:
                a, b = payload
                src.seek(data_start + a)
                left = b - a
                while left:
                    buf = src.read(min(CHUNK, left))
                    assert buf
                    dst.write(buf)
                    left -= len(buf)
        dst.flush()
        os.fsync(dst.fileno())

    os.replace(tmp, args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out)} bytes, {len(quant_layers)} NVFP4 layers)")


if __name__ == "__main__":
    main()
