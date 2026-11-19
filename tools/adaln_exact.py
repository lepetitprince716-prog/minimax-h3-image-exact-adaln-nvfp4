#!/usr/bin/env python3
"""Compute exact 7-step official adaLN modulation vectors for MiniMax-H3-Image
and measure the BF16 ULP error of ComfyUI's 8-dimensional curve approximation.

Usage:
    python3 tools/adaln_exact.py \
        --official-dir /path/to/MiniMax-H3-Image \
        --comfy-pruned /path/to/minimax_h3_image_pruned_bf16.safetensors \
        --out-dir ./build

Writes:
    ./build/adaln_exact_vectors.safetensors
    ./build/adaln_report.json
"""
import argparse
import json
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as Fn
from safetensors import safe_open
from safetensors.torch import save_file

VISUAL_COND_TIMESTEP = 0.999


def f32(x: float) -> float:
    return float(torch.tensor([x], dtype=torch.float32)[0])


def manual_sigmas(text: str) -> torch.Tensor:
    s = re.findall(r"[-+]?(?:\d*\.*\d+)", text)
    return torch.FloatTensor([float(i) for i in s])


def compute_timesteps(official_dir: str, comfy_pruned: str):
    with open(os.path.join(official_dir, "pdd_config.json")) as f:
        sig = json.load(f)["sigmas"]
    with safe_open(comfy_pruned, framework="pt") as cf:
        meta_sig = json.loads(cf.metadata()["config"])["transformer"]["pdd_sigmas"]
    assert sig == meta_sig, "pdd_config sigmas differ from ComfyUI metadata"

    t1 = torch.tensor([1.0 - s for s in sig[:-1]], dtype=torch.float32)
    sig_rt = torch.cat([1.0 - t1, torch.zeros(1)])
    sig_rt = torch.as_tensor(sig_rt, dtype=torch.float64).flatten().cpu().to(torch.float32)
    t_sched = 1.0 - sig_rt[:-1]
    t_rows = torch.tensor([float(x) for x in t_sched], dtype=torch.float32)
    t_ref = torch.tensor([VISUAL_COND_TIMESTEP], dtype=torch.float32)
    t_off = torch.cat([t_rows, t_ref])

    grid = torch.tensor(sig, dtype=torch.float32).flip(0)
    mult = 1000

    def sigma_fn(timestep):
        n = grid.shape[0] - 1
        pos = (timestep / mult).clamp(0.0, 1.0) * n
        i = pos.floor().long().clamp(max=n - 1)
        return torch.lerp(grid[i], grid[i + 1], pos - i)

    sigmas = sigma_fn((torch.arange(1, 1001, 1) / 1000) * mult)
    steps = 6
    ss = len(sigmas) / steps
    simple_sg = torch.FloatTensor([float(sigmas[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])
    man_pdd = ", ".join(repr(float(s)) for s in sig)

    sched = {"manual_pdd": manual_sigmas(man_pdd), "simple": simple_sg}
    comfy_info = {}
    for name, sg32 in sched.items():
        calls = []
        for i in range(len(sg32) - 1):
            s = sg32[i]
            if float(s) == 0.0:
                continue
            ts = (s.reshape(1) * mult).float()
            sigma = float(ts.flatten()[0]) / 1000.0
            t32 = f32(1.0 - sigma)
            calls.append({
                "sigma_in": float(s),
                "t_fp32": t32,
                "t_fp32_minus_official": t32 - float(t_off[i]) if i < 6 else None,
            })
        comfy_info[name] = {
            "sigmas": [float(x) for x in sg32],
            "calls": calls,
            "max_abs_t_minus_official": max(abs(c["t_fp32_minus_official"]) for c in calls[:6]),
        }

    t_manual = torch.tensor([c["t_fp32"] for c in comfy_info["manual_pdd"]["calls"]] + [VISUAL_COND_TIMESTEP], dtype=torch.float32)
    t_simple = torch.tensor([c["t_fp32"] for c in comfy_info["simple"]["calls"]] + [VISUAL_COND_TIMESTEP], dtype=torch.float32)
    return t_off, t_manual, t_simple, {"pdd_sigmas": sig, "manual_pdd_string": man_pdd, "comfy": comfy_info}


def official_tensor(official_dir: str, name: str, index: dict) -> torch.Tensor:
    with safe_open(os.path.join(official_dir, "transformer", index[name]), framework="pt") as f:
        return f.get_tensor(name)


def time_embed(t: torch.Tensor, te: dict, dtype: torch.dtype) -> torch.Tensor:
    half = te["proj_in.weight"].shape[1] // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.to(torch.float32)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
    w = {k: v.to(dtype) for k, v in te.items()}
    h = Fn.linear(emb, w["proj_in.weight"], w["proj_in.bias"])
    h = Fn.silu(h)
    return Fn.linear(h, w["proj_out.weight"], w["proj_out.bias"])


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    a = x.abs().double().clamp(min=2.0 ** -126)
    _, e = torch.frexp(a)
    return torch.pow(2.0, (e - 1 - 7).double())


def cmp_metrics(approx: torch.Tensor, exact: torch.Tensor) -> dict:
    d = (approx.double() - exact.double()).abs()
    e = exact.double().abs()
    nz = e > 0
    rel = d[nz] / e[nz]
    u = d / ulp_bf16(exact)
    return {
        "max_abs": float(d.max()),
        "rel_l2": float(d.norm() / e.norm()),
        "ulp_median": float(u.median()),
        "ulp_p99": float(np.quantile(u.reshape(-1).numpy(), 0.99)),
        "ulp_max": float(u.max()),
        "frac_neq": float((d > 0).double().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-dir", required=True, help="Local directory of MiniMaxAI/MiniMax-H3-Image")
    ap.add_argument("--comfy-pruned", required=True, help="Path to minimax_h3_image_pruned_bf16.safetensors")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    t_off, t_manual, t_simple, s2 = compute_timesteps(args.official_dir, args.comfy_pruned)
    with open(os.path.join(args.official_dir, "transformer", "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]

    te = {
        k: official_tensor(args.official_dir, "time_embedder." + k, index)
        for k in ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")
    }
    te32 = time_embed(t_off, te, torch.float32)
    xA = bf16_round(Fn.silu(te32)).double()

    cf = safe_open(args.comfy_pruned, framework="pt")
    table = cf.get_tensor("adaln_t_table")

    def coords_at(tv: torch.Tensor) -> torch.Tensor:
        pos = tv.to(torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    coords = coords_at(t_manual)
    names = [f"blocks.{i}.adaln_proj.linear" for i in range(50)] + ["final_layer.adaln_proj.linear"]

    vecs = {"t_official": t_off}
    per_block = {}
    ulp_blocks = []

    for name in names:
        W = official_tensor(args.official_dir, name + ".weight", index).double()
        b = official_tensor(args.official_dir, name + ".bias", index).double()
        yA = bf16_round((xA @ W.T + b).float())

        Wc = cf.get_tensor(name + ".weight").float()
        bc = cf.get_tensor(name + ".bias").float()
        is_final = name.startswith("final_layer")
        yC32 = Fn.linear(coords, Wc, bc)
        yC = yC32 if is_final else bf16_round(yC32)

        key = name.replace(".adaln_proj.linear", "")
        vecs[key + ".A"] = yA.contiguous()
        per_block[key] = {"curve_vs_official": cmp_metrics(yC, yA)}
        if not is_final:
            u = ((yC.double() - yA.double()).abs() / ulp_bf16(yA)).numpy()
            ulp_blocks.append(u)
        print(f"{key:<14s} curve_vs_official p99={per_block[key]['curve_vs_official']['ulp_p99']:.1f} ulp", flush=True)

    U = np.stack(ulp_blocks)
    report = {
        "timesteps": s2,
        "curve_vs_official_50_blocks": {
            "ulp_median": float(np.median(U)),
            "ulp_p99": float(np.quantile(U, 0.99)),
            "ulp_max": float(U.max()),
            "per_t_p99_ulp": [float(np.quantile(U[:, i], 0.99)) for i in range(7)],
        },
        "per_block": per_block,
    }
    vec_out = os.path.join(args.out_dir, "adaln_exact_vectors.safetensors")
    rep_out = os.path.join(args.out_dir, "adaln_report.json")
    save_file(vecs, vec_out)
    with open(rep_out, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved:", vec_out, "and", rep_out)


if __name__ == "__main__":
    main()
