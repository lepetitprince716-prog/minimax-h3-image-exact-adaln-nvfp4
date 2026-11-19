# MiniMax-H3-Image Exact-adaLN NVFP4

A **12.61 GB** (`11.74 GiB`) Blackwell **NVFP4** checkpoint of [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) (33.1B DiT) for ComfyUI, paired with a **zero-ULP-error 7-step one-hot `adaLN` lookup table** in ComfyUI's native curve format (no ComfyUI patches or custom nodes required).

Governed by the [MiniMax-H3-Image Community License Agreement](LICENSE-MODEL.md) (note the upstream territory exclusions in Section I.5: EU, UK, Republic of Korea, and USA). Scripts and workflows in this repository are MIT-licensed. Not an official MiniMax release.

**Download weights:** [Hugging Face (`kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4`)](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4) and [Civitai (`3057968`)](https://civitai.com/models/3057968/minimax-h3-image-exact-adaln-nvfp4). This GitHub repository holds the build scripts, error measurements, and ComfyUI workflows because GitHub caps release files at 2 GB.

![Sample 01](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4/resolve/main/samples/01-alpine-knight-sunrise.jpg)

## Overview of What Was Changed

### 1. Exact 7-Step One-Hot `adaLN` Lookup Table (`0` BF16 ULP Error)

In the upstream `MiniMaxAI/MiniMax-H3-Image` DiT (`66.28 GB` BF16), each of the 50 transformer blocks holds a dense `adaln_proj.linear` layer (`2688 -> 96768` with bias) taking `silu(time_embedder(t))` as input. Together those 50 MLPs occupy **26.0 GB (39.5% of the entire DiT)** even though their output depends solely on the scalar timestep `t`.

Comfy-Org's pruned checkpoint (`minimax_h3_image_pruned_bf16.safetensors`, `40.23 GB`) compresses those MLPs into a curve format (`adaln_t_table` `F32 [1025, 8]` linearly interpolated and multiplied by an `F16 [96768, 8]` basis matrix per block). Measuring that 8-dimensional `F16` curve against the official `FP64`-accumulated `BF16` runtime path across all 50 blocks shows:

| Comparison vs Official `adaLN` Output | Median Error | `p99` Error | Worst `t` (`p99`) | Fraction `!=` Official |
|---|---|---|---|---|
| **Standard 8-dim `F16` curve** (`Comfy-Org` pruned) | `0` BF16 ULP | **`5` BF16 ULP** | `t = 0.0` (`7` ULP), `t = 0.999` ref (`9` ULP) | `13% - 15%` |
| **Exact 7-step one-hot table (this release)** | **`0` BF16 ULP** | **`0` BF16 ULP** | **`0` BF16 ULP** on all 7 timesteps | **`0.0%`** |

Because `MiniMax-H3-Image` is a 6-step PDD flow model, the DiT is **only ever invoked at 7 fixed timesteps**:
- the 6 target PDD timesteps: `t = 0.0, 0.0324763, 0.1124515, 0.25, 0.4672261, 0.7679911`
- the visual reference-image conditioning timestep: `t = 0.999` (`VISUAL_COND_TIMESTEP` in ComfyUI `comfy/ldm/minimax/model.py`)

ComfyUI auto-detects curve format from the shape of `adaln_t_table` (`[grid_size, cols]`) without hardcoding `cols = 8` (`comfy/model_detection.py`), skips `silu`, and computes `coords @ W.T + bias` in `FP32` before casting block modulations to `BF16`. We exploit this directly with **zero code changes to ComfyUI**:

- `adaln_t_table` (`F32 [1025, 7]`): row `r` is set to the one-hot vector $e_i$ of the official timestep $t_i$ nearest to $r / 1024$. Because all 7 $t_i$ are separated by at least `0.032` (`> 32` grid cells), both endpoints `floor(1024 * t_i)` and `floor(1024 * t_i) + 1` of ComfyUI's linear interpolation equal the **same** $e_i$, making `lerp` return exact $e_i$ even under `FP32` sigma round-trip jitter.
- `blocks.{0..49}.adaln_proj.linear.weight` (`F32 [96768, 7]`) and `final_layer.adaln_proj.linear.weight` (`F32 [10752, 7]`): column `i` stores the exact official `adaln_proj.linear(bf16(silu(time_embedder(t_i))))` output vector, with `bias = 0`.
- Total lookup footprint across all 50 blocks + `final_layer`: **~136 MB** instead of **26.0 GB**.

### 2. Blackwell Native `NVFP4` Quantization (`comfy-kitchen`)

Using ComfyUI's `comfy_kitchen.tensor.TensorCoreNVFP4Layout.quantize` (`group_size = 16`), only the **200 heavy DiT block linear projections** (`blocks.0..49.{attn.qkv_proj, attn.out_proj, mlp.fc1, mlp.fc2}`) are quantized to `NVFP4` (`E2M1` packed `U8` weights + `F8_E4M3` per-16 block scales + `F32` per-tensor scale).

**Kept unquantized (`BF16` / `FP32`):**
- Exact 7-step `adaLN` lookup table (`adaln_t_table`, `blocks.*.adaln_proj`, `final_layer.adaln_proj`)
- Multimodal condition & patch embedders (`token_refiner.*`, `condition_proj`, `video_patch_proj`)
- All 6-step PDD output heads (`final_layer.video_out`) and all `q_norm` / `k_norm` / `norm1` / `norm2` layers

| File | Bytes | SHA-256 |
|---|---|---|
| `minimax_h3_image_exact_adaln_nvfp4.safetensors` (this release) | `12,605,496,924` | `e5a9122b14dc31ab4da67b1a7513041afb3ced7d31dd81f501c1aa16b958aa2a` |

## Speed, Memory, and Quality

Measured in ComfyUI `0.45.0` (`comfy-kitchen 0.2.41`, PyTorch `2.13.0+cu130`, `--bf16-text-enc --fp32-vae`, 6 PDD steps via `ManualSigmas`, 4 MP canvas):

| Variant | File Size | Resident DiT Weights | Peak DiT VRAM (4 MP T2I) | Peak DiT VRAM (4 MP Dual-Ref Edit) | 6-Step DiT Time (4 MP T2I) | 6-Step DiT Time (4 MP Dual-Ref) |
|---|---|---|---|---|---|---|
| Upstream unpruned `BF16` DiT | `66.28 GB` | `~61.7 GiB` | — | — | — | — |
| Exact-adaLN `BF16` | `40.30 GB` | `37.53 GiB` | `38.11 GiB` | `40.78 GiB` | `15.02 s` (`1.00x`) | `30.38 s` (`1.00x`) |
| Exact-adaLN `INT8 ConvRot` | `21.05 GB` | `19.62 GiB` | `20.06 GiB` | `21.59 GiB` | `12.09 s` (`1.24x`) | `31.36 s` (`0.97x`) |
| **Exact-adaLN `NVFP4` (this)** | **`12.61 GB`** | **`11.75 GiB`** | **`12.34 GiB`** | **`14.00 GiB`** | **`10.55 s` (`1.42x`)** | **`20.97 s` (`1.45x`)** |

At `12.34 GiB` peak DiT VRAM for 4 MP generation, the DiT fits inside **16 GB Blackwell GPUs** (such as RTX 5080 / RTX 5070 Ti) when ComfyUI offloads the Qwen3-VL text encoder between encoding and sampling (or when paired with a quantized text encoder such as `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`). Non-Blackwell GPUs can still load the checkpoint via ComfyUI's on-the-fly dequantization fallback (saving disk and host RAM without native FP4 Tensor Core speedup).

### Visual Comparison (`Exact-adaLN BF16` vs `Exact-adaLN NVFP4`)

4-bit weights alter the 6-step PDD trajectory slightly on unconditioned text-to-image seeds (`~13.4 dB` PSNR), while dual-reference editing remains closely locked (`~15.1 dB` PSNR) and fine CJK / Latin typography stays sharp:

![Exact-adaLN BF16 (left) vs Exact-adaLN NVFP4 (right)](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4/resolve/main/samples/ab-bf16-vs-nvfp4.jpg)

| | | |
|---|---|---|
| ![](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4/resolve/main/samples/02-vermilion-lantern-rain.jpg) | ![](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4/resolve/main/samples/03-conservatory-astrolabe.jpg) | ![](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4/resolve/main/samples/04-winter-terrace-tea.jpg) |
| ![](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4/resolve/main/samples/05-autumn-gorge-viaduct.jpg) | ![](https://huggingface.co/kefallrain/MiniMax-H3-Image-Exact-adaLN-NVFP4/resolve/main/samples/06-basalt-coastal-lighthouse.jpg) | |

## ComfyUI Usage & Exact `ManualSigmas` Schedule

1. Place `minimax_h3_image_exact_adaln_nvfp4.safetensors` in `ComfyUI/models/diffusion_models/`.
2. Place `qwen3vl_32b_minimax_h3_bf16.safetensors` (or `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`) in `ComfyUI/models/text_encoders/` and pass `--bf16-text-enc` if using the BF16 text encoder (without `--bf16-text-enc`, ComfyUI defaults the H3 text encoder to `fp16`).
3. Place `minimax_h3_image_vae_fp32.safetensors` (from `MiniMaxAI/MiniMax-H3-Image/vae`, run with `--fp32-vae`) or `minimax_h3_image_vae_fp16.safetensors` in `ComfyUI/models/vae/`.
4. Load [`workflows/minimax-h3-image-exact-adaln-nvfp4-t2i.json`](workflows/minimax-h3-image-exact-adaln-nvfp4-t2i.json) (text-to-image) or [`workflows/minimax-h3-image-exact-adaln-nvfp4-edit.json`](workflows/minimax-h3-image-exact-adaln-nvfp4-edit.json) (reference-image editing).

### Why `ManualSigmas` Instead of `KSampler (simple, 6 steps)`?

ComfyUI's built-in `simple` 6-step scheduler indexes a 1000-point table at non-integer PDD intervals on 4 of the 6 steps, causing sigma to drift up to **`8.7e-4`** from the official `pdd_config.json` schedule. Passing the exact 7 PDD sigmas via `ManualSigmas` + `SamplerCustomAdvanced` + `KSamplerSelect (euler)` + `BasicGuider` matches the official schedule to **`< 1.5e-8`**:

```text
1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0
```

## Reproducing Everything From Upstream Checkpoints

All conversion scripts stream safetensors in `128 MiB` chunks without holding `40+ GB` in host RAM:

```bash
# 1. Extract the 7-step exact official adaLN vectors (and audit curve ULP error)
python3 tools/adaln_exact.py \
  --official-dir /path/to/MiniMaxAI/MiniMax-H3-Image \
  --comfy-pruned /path/to/minimax_h3_image_pruned_bf16.safetensors \
  --out-dir ./build

# 2. Stream-graft the [1025, 7] one-hot table + exact vectors onto Comfy's pruned BF16 DiT
python3 tools/build_exact_dit.py \
  --src /path/to/minimax_h3_image_pruned_bf16.safetensors \
  --exact-vectors ./build/adaln_exact_vectors.safetensors \
  --out ./build/minimax_h3_image_exact_adaln.safetensors

# 3. Quantize the 200 DiT linear projections to Blackwell NVFP4 via comfy-kitchen
python3 tools/quantize_nvfp4.py \
  --src ./build/minimax_h3_image_exact_adaln.safetensors \
  --out ./build/minimax_h3_image_exact_adaln_nvfp4.safetensors

# (Optional) Build the 51.51 GB layer-50 truncated Qwen3-VL text encoder from official shards
python3 tools/build_te_layer50.py \
  --official-te-dir /path/to/MiniMaxAI/MiniMax-H3/text_encoder \
  --out ./build/qwen3vl_32b_minimax_h3_bf16.safetensors
```

## License

- **Model weights (`minimax_h3_image_exact_adaln_nvfp4.safetensors`):** Derivative of [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) and [`Comfy-Org/MiniMax-H3-Image`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), distributed under the [MiniMax-H3-Image Community License Agreement](LICENSE-MODEL.md). Please review the territory exclusions (EU, UK, Republic of Korea, USA) and Acceptable Use Policy in [`LICENSE-MODEL.md`](LICENSE-MODEL.md) before downloading or deploying the weights.
- **Scripts and workflows in this repository:** [MIT License](LICENSE).
