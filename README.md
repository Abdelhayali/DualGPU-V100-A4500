# DualGPU: LLMs on an RTX A4500 + Tesla V100

Run and tune local LLMs with llama.cpp split over two different GPUs, served on `http://127.0.0.1:8899/v1` (OpenAI API).

| | GPU | VRAM | memory BW | arch |
|---|---|---|---|---|
| CUDA0 | RTX A4500 | 20 GB | 640 GB/s | sm_86 |
| CUDA1 | Tesla V100-SXM2 (PCIe adapter) | 16 GB | 900 GB/s | sm_70 |

Rest of the PC: Xeon E5-2660 v4, X99, 64 GB RAM, Windows 11, both cards on PCIe 3.0 x16.
CUDA0/CUDA1 are pinned by GPU UUID in `tools/common.ps1`, so `-ts a,b` always means A4500,V100 whatever slots the cards are in.

## Quick start

**GUI** (same launcher as V100-LLM): `START-GUI.bat` opens http://127.0.0.1:7860. Pick a model and preset, Load, chat with
images, edit any flag, and watch tok/s plus VRAM / load / power for each GPU. The model is served at
`http://127.0.0.1:8899/v1` (OpenAI API) for any other app.

**Command line:**

```bat
start.bat                                 :: list models and presets
start.bat orcacyber-nano-27b              :: a model's default preset
start.bat minicpm-v4.7-35b-a3b fast       :: a named preset
```

## Leaderboard (hard suite, 72 tasks, same tasks and scoring as V100-LLM)

| # | model (quant, size) | preset | ctx | gen tok/s | coding | reasoning | instr | search | agent | vision | longctx | **overall** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OrcaCyber-Nano 1.0 27B Mix-15.7G (MTP), vision | `128k` | 128k | 61.6 | 94 | 100 | 50 | 100 | 100 | 88 | 100 | **90.2** |
| 2 | Qwen3.6-35B-A3B UD-Q5_K_XL (25.3 GB, MoE, MTP), vision | `128k` | 128k | 99.6 | 94 | 100 | 50 | 88 | 100 | 88 | 100 | **88.4** |
| 3 | MiniCPM-V 4.7 35B-A3B Q6_K (26.6 GB), vision | `128k` | 128k | 83.6 | 94 | 100 | 33 | 88 | 100 | 88 | 100 | **86.0** |
| 4 | Qwen3.8-27B UD-Q5_K_XL (20.9 GB, MTP), vision | `128k` | 128k | 51.3 | 88 | 95 | 58 | 100 | 100 | 88 | 50 | **82.6** |
| 5 | Kimi-Linear 48B-A3B abliterated i1-Q4_K_M (27.7 GB), text only | `262k` | 262k | 100.9 | 56 | 60 | 50 | 88 | 100 | - | 50 | **57.7**\* |

\* Kimi has no vision, which the suite scores as 0; its mean over the 6 text categories is 67.3. Many reasoning misses are
format (it ends with a LaTeX `\[ ... \]` block, so the graded last line is `\]`), but the coding misses are real bugs.

The V100-only references on the same suite: OrcaCyber Mix-13G 89.8 at 60 tok/s (32k), MiniCPM-V 4.7 Q3 mix 87.5 at
93 tok/s, Qwen3.6-35B-A3B UD-IQ3_XXS 88.4 at 119 tok/s (64k). Two GPUs give the bigger quants and 2-8x the context; for the
small-active MoE models one V100 with a 3-bit quant is faster (the whole model sits on the faster card, no hand-over).

## Presets

| model | preset | config | ctx | KV | tok/s (mean of code + chat) | 7k prompt |
|---|---|---|---|---|---|---|
| orcacyber-nano-27b | **128k** (default) | `auto-mtp3-128k` | 128k | f16 | **58.2** | 974 t/s |
| | code | `auto-mtp4-32k` | 32k | f16 | 57.9 (69.9 on code) | 970 t/s |
| | 262k | `l55-mtp3-262k-q8` | 262k | q8_0 | 56.5 | 958 t/s |
| qwen3.6-35b-a3b | **128k** (default) | `auto-mtp3-128k` | 128k | f16 | **95.2** (109.7 on code) | 919 t/s |
| | fast | `l60-mtp3-32k` | 32k | f16 | 96.6 (111.0 on code) | 989 t/s |
| | 262k | `auto-mtp3-262k-q8` | 262k | q8_0 | 86.0 | 916 t/s |
| minicpm-v4.7-35b-a3b | **128k** (default) | `auto-nodraft-128k` | 128k | f16 | **87.0** | 1055 t/s |
| | fast | `auto-nodraft-32k` | 32k | f16 | 91.6 | 1055 t/s |
| | 262k | `auto-nodraft-262k-q8` | 262k | q8_0 | 82.3 | 1051 t/s |
| kimi-linear-48b-a3b | **262k** (default) | `auto-262k` | 262k | f16 | **104.8** | 1299 t/s |
| | fast | `auto-32k` | 32k | f16 | 106.0 | 1276 t/s |
| | 512k | `auto-512k` | 512k | f16 | 102.1 | 1259 t/s |
| qwen3.8-27b | **128k** (default) | `l60-mtp3-128k` | 128k | f16 | **48.3** | 917 t/s |
| | fast | `layer-auto-mtp3-32k` | 32k | f16 | 50.2 | 952 t/s |
| | 262k | `l65-mtp3-262k-q4-ub256` | 262k | q4_0 | 46.0 | 846 t/s |

All presets pass the image test (vision models) and a ~7k-token prompt. Per-model findings:

- **OrcaCyber**: same architecture as Qwen3.8-27B. MTP 3-4 beats its DFlash2 drafter (50.6-51.1). Both GPUs: 57.3 tok/s vs
  47.5 on the A4500 alone. 262k f16 KV does not fit; q8 does, with both cards nearly full.
- **Qwen3.6-35B-A3B**: unsloth's MTP build (MTP head in the file). MTP 3 95 tok/s vs 88 without (MTP 4: 112 on code,
  78 on chat). Splits within 3 tok/s. 262k f16 KV runs text (93.6) but the vision encoder runs out of memory on the
  A4500, so the 262k preset uses q8 KV.
- **MiniCPM-V 4.7**: MoE with ~3B active, so the DSpark draft is a loss here (n8 66.0, n15 54.2 vs 91.6 without, 18-32%
  accepted). 40/60 split and 262k f16 KV do not fit.
- **Kimi-Linear**: 20 linear-attention (KDA) + 7 MLA layers, so 512k context costs little. 1M does not fit (needs a 4.6 GB
  more buffer on the A4500); q8_0 KV is impossible (V head dim 72), and mixed K/V types are not supported for this model.

### Qwen3.8-27B presets in detail

| preset | config | ctx | KV | split A4500/V100 | tok/s (code / chat) | 7k prompt |
|---|---|---|---|---|---|---|
| **128k** (default) | `l60-mtp3-128k` | 128k | f16 | 60/40 | **48.3** (54.2 / 42.3) | 917 t/s |
| fast | `layer-auto-mtp3-32k` | 32k | f16 | auto | **50.2** (55.9 / 44.4) | 952 t/s |
| 262k | `l65-mtp3-262k-q4-ub256` | 262k | q4_0 | 65/35 | **46.0** (51.8 / 40.2) | 846 t/s |

All three use `-sm layer`, MTP 3 (built-in draft head), `-b 2048 -ub 512` (262k: ub 256), flash attention, vision on.

## Qwen3.8-27B results

Model: `unsloth/Qwen3.8-27B-GGUF` UD-Q5_K_XL (20.9 GB) + mmproj-F16. Speed = mean of a code and a chat reply
(300 tokens, temperature 0, thinking off). Every config must also read the test image and answer a ~7k-token prompt.
Power limits during these runs: V100 300 W, A4500 170-200 W.

### Hard suite (same tasks and scoring as the V100-only setup)

| setup | overall | coding | reasoning | ifeval | search | agent | vision | longctx | gen tok/s | prompt tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| **this: Q5_K_XL, A4500 + V100, 128k** | **82.6** | 87.5 | 95.0 | 58.3 | 100 | 100 | 87.5 | 50 | 51.3 | 884 |
| ref: unsloth UD-IQ3_S, V100 only, 32k | 88.4 | 93.8 | 100 | 50.0 | 100 | 100 | 75.0 | 100 | 52.1 | 460 |
| ref: ByteShape IQ4_XS, V100 only, 32k | 84.2 | 93.8 | 100 | 58.3 | 100 | 100 | 87.5 | 50 | 53.7 | 531 |

The suite is small: longctx is 2 questions (one miss = -7 overall), coding 16 tasks. The gap to IQ3_S is one long-context
question (Teo's daughter's birth year) plus two coding tasks, so quality is about the same; the bigger quant did not buy points
here. What two GPUs do buy: the same generation speed with a 5-6 GB larger quant, 4-8x the context (128k-262k vs 32k),
and ~1.9x faster prompt processing.

### Launch-config tuning (`results/tune-qwen3.8-27b.jsonl`, `python tools/report.py`)

| what | result |
|---|---|
| split ratio (32k, MTP 3) | auto 50.2, 45/55 49.7, 60/40 49.2, 50/50 49.1, 40/60 47.0 tok/s: flat within ~1 tok/s |
| speculative decoding | none 26.7, MTP 2 45.9, **MTP 3 49.7**, MTP 4 49.1 (code 59.6, chat 38.6), DFlash2 n4 46.7, n6 47.9 |
| DFlash draft on the A4500 | does not start (draft shares `output.weight`, which lives on the V100) |
| KV cache (32k) | f16 49.7, q8_0 47.5 |
| ubatch (b 2048) | ub 256 49.2 / 800 t/s prefill, **ub 512** 49.7 / 896, ub 1024 48.9 / 904 |
| b = ub = 256 (no pipelined prefill) | 34.7 tok/s, 405 t/s prefill (measured with both cards capped at 100 W) |
| 128k | 45/55: V100 out of memory; **60/40 f16 48.3**, 60/40 q8 47.3 |
| 262k | q8 does not fit (or crashes on the image); **q4 60/40 47.0** (V100 at 16.1 GB, too tight), **q4 65/35 ub 256 46.0** |
| `-sm row` | does not start in this build ("CUDA0 does not support split buffers") |
| `-sm tensor` | ~55 tok/s on code, then **cut the power** of the PC (both GPUs at full load together). Do not use. |

## Power notes (read this before you try heavy configs)

- With the original **500 W PSU** the PC shut off hard (Kernel-Power 41, no BSOD) four times: three times in pipelined prefill
  (`-b 2048 -ub 512`, both GPUs computing at once) and once even with b = ub. Cause: the PSU, not PCIe bandwidth
  (layer split moves ~40 MB per ubatch).
- After the PSU was replaced, all layer-split configs pass at full power limits. `-sm tensor` still cut the power, because
  it runs both cards at full load for every token.
- On a weak PSU: set `"b"` equal to `"ub"` in a config (the cards take turns), lower the power limits, and avoid tensor split.

## Layout

```
START-GUI.bat           the GUI (gui/launcher.py + ui.html, from V100-LLM, adapted to two GPUs) on :7860
catalog.json            what the GUI shows: generated by tools\make-catalog.ps1 (do not edit by hand)
start.bat               start a preset:  start.bat <model> [preset]
presets.json            tested presets per model
configs/<model>.json    model files, sampling, every launch config and named sets
tools/tune.ps1          run configs: tools\tune.ps1 <model> [names] | -Set <set>   -> results/tune-<model>.jsonl/.log
tools/eval.ps1          hard suite:  tools\eval.ps1 <model> <config>               -> results/hard/
tools/report.py         markdown tables from the results
tools/fill-hashes.py    sizes + sha256 of model files from Hugging Face -> configs/files.json (GUI downloads)
tools/make-catalog.ps1  configs + presets + results -> catalog.json (same launch args as start.bat)
tools/build-kernel.bat  build llama.cpp (CUDA 12.9, sm_70 + sm_86) into kernel/bin
kernel/                 SOURCE.md + patches (source and binaries are not in git)
models/<model>/         GGUF files (not in git)
```

Kernel: llama.cpp `f336359` + V100 MMVQ tuning (`kernel/patches/0001-*.patch`), built with CUDA 12.9 (CUDA 13 dropped Volta).
Prebuilt Windows binaries: GitHub release `kernel-da8fd08` (unzip into `kernel\bin`). See `kernel/SOURCE.md`.

## Add a model

1. Put the GGUF (+ mmproj / draft) in `models/<id>/`.
2. Write `configs/<id>.json` (copy `qwen3.8-27b.json`: `gguf`, `mmproj`, `sampling`, `configs`, `sets`).
3. `tools\tune.ps1 <id> -Set <set>`, then `tools\eval.ps1 <id> <best config>`.
4. Add presets to `presets.json` and a section here.
5. `python tools\fill-hashes.py <id>` and `tools\make-catalog.ps1`: the model shows up in the GUI.

The GUI runs one model at a time and kills any other `llama-server` when it loads one, so don't use it while `tune.ps1` or
`eval.ps1` is running.
