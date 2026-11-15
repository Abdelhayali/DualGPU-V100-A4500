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
| 1 | Qwen3.8-27B UD-Q4_K_M (16.6 GB, dense, MTP), vision | `128k` | 128k | 58.8 | 94 | 100 | 42 | 100 | 100 | 100 | 100 | **90.8** |
| 2 | Huihui Gemma-4-26B-A4B-it QAT abliterated Q4_K (16.8 GB, MoE), vision | `262k` | 262k | 86.5 | 88 | 100 | 83 | 88 | 100 | 75 | 100 | **90.5** |
| 3 | OrcaCyber-Nano 1.0 27B Mix-15.7G (MTP), vision | `128k` | 128k | 61.6 | 94 | 100 | 50 | 100 | 100 | 88 | 100 | **90.2** |
| 4 | Qwen3.6-35B-A3B UD-Q5_K_XL (25.3 GB, MoE, MTP), vision | `128k` | 128k | 99.6 | 94 | 100 | 50 | 88 | 100 | 88 | 100 | **88.4** |
| 5 | MiniCPM-V 4.7 35B-A3B Q6_K (26.6 GB), vision | `128k` | 128k | 83.6 | 94 | 100 | 33 | 88 | 100 | 88 | 100 | **86.0** |
| 6 | Qwen3.8-27B UD-Q5_K_XL (20.9 GB, MTP), vision | `128k` | 128k | 51.3 | 88 | 95 | 58 | 100 | 100 | 88 | 50 | **82.6** |
| 7 | Qwen3.8-27B UD-Q5_K_XL (21.0 GB, unsloth update 2026-11-12, MTP), vision | `128k` | 128k | 51.3 | 75 | 100 | 50 | 100 | 100 | 88 | 50 | **80.4** |
| 8 | Ripple-65B Preview AD-IQ3_XXS (28.0 GB, dense), text only | `64k` | 64k | 13.9 | 75 | 100 | 83 | 100 | 100 | - | 100 | **79.8**\* |
| 9 | Kimi-Linear 48B-A3B abliterated i1-Q4_K_M (27.7 GB), text only | `262k` | 262k | 100.9 | 56 | 60 | 50 | 88 | 100 | - | 50 | **57.7**\* |

\* Text-only models: the suite scores the missing vision category as 0. Over the 6 text categories Ripple averages 93.1
(the best instruction-following score here, 83) and Kimi 67.3. Many reasoning misses are
format (it ends with a LaTeX `\[ ... \]` block, so the graded last line is `\]`), but the coding misses are real bugs.

The V100-only references on the same suite: OrcaCyber Mix-13G 89.8 at 60 tok/s (32k), MiniCPM-V 4.7 Q3 mix 87.5 at
93 tok/s, Qwen3.6-35B-A3B UD-IQ3_XXS 88.4 at 119 tok/s (64k). Two GPUs give the bigger quants and 2-8x the context; for the
small-active MoE models one V100 with a 3-bit quant is faster (the whole model sits on the faster card, no hand-over).

## Presets

| model | preset | config | ctx | KV | tok/s (mean of code + chat) | 7k prompt |
|---|---|---|---|---|---|---|
| orcacyber-nano-27b | **128k** (default) | `auto-mtp3-128k` | 128k | f16 | **58.2** | 974 t/s |
| | code | `auto-dsp15-32k` (DSpine) | 32k | f16 | 52.1 (**78.2** on code) | 793 t/s |
| | 262k | `l55-mtp3-262k-q8` | 262k | q8_0 | 56.5 | 958 t/s |
| qwen3.6-35b-a3b | **128k** (default) | `auto-mtp3-128k` | 128k | f16 | **95.2** (109.7 on code) | 919 t/s |
| | fast | `l60-mtp3-32k` | 32k | f16 | 96.6 (111.0 on code) | 989 t/s |
| | 262k | `auto-mtp3-262k-q8` | 262k | q8_0 | 86.0 | 916 t/s |
| minicpm-v4.7-35b-a3b | **128k** (default) | `auto-nodraft-128k` | 128k | f16 | **87.0** | 1055 t/s |
| | fast | `auto-nodraft-32k` | 32k | f16 | 91.6 | 1055 t/s |
| | 262k | `auto-nodraft-262k-q8` | 262k | q8_0 | 82.3 | 1051 t/s |
| gemma4-31b-ara-heretic | **64k** (default) | `auto-mtp4-64k` | 64k | f16 | **42.7** (48.5 on code) | 784 t/s |
| | fast | `auto-mtp4-32k` | 32k | f16 | 43.8 (50.0 on code) | 831 t/s |
| | 128k | `auto-mtp4-128k-q8` | 128k | q8_0 | 39.3 | 829 t/s |
| qwen3.8-27b-nov12 | **128k** (default) | `l60-mtp3-128k` | 128k | f16 | **47.2** (53.6 on code) | 901 t/s |
| | fast | `l45-mtp4-32k` | 32k | f16 | 48.9 (58.2 on code) | 866 t/s |
| | 262k | `l65-mtp3-262k-q4-ub256` | 262k | q4_0 | 45.3 | 844 t/s |
| qwen3.8-27b-q4km | **128k** (default) | `l60-mtp3-128k` | 128k | f16 | **53.5** (60.2 on code) | 961 t/s |
| | fast | `l50-mtp3-32k` | 32k | f16 | 55.0 (62.0 on code) | 935 t/s |
| | 262k | `l60-mtp3-262k-q8` | 262k | q8_0 | 51.7 | 942 t/s |
| qwen3.8-40b-heretic | **64k** (default) | `auto-mtp2-64k` | 64k | f16 | **35.7** (37.5 on code) | 580 t/s |
| | code | `auto-mtp4-32k` | 32k | f16 | 34.9 (**38.8** on code) | 586 t/s |
| | 128k | `l55-mtp3-128k-q4` | 128k | q4_0 | 34.5 | 578 t/s |
| muse-glimmer-30b | **128k** (default) | `auto-dfl15-128k` (DFlash) | 128k | f16 | **40.6** (45.5 on code) | 801 t/s |
| | fast | `auto-dfl12-32k` (DFlash) | 32k | f16 | 41.3 (46.8 on code) | 810 t/s |
| | nodraft | `auto-nodraft-128k` | 128k | f16 | 26.6 | 1165 t/s |
| huihui-gemma4-26b-a4b | **262k** (default) | `auto-nodraft-262k` | 262k | f16 | **87.5** | 1808 t/s |
| | fast | `l60-nodraft-32k` | 32k | f16 | 88.9 | 1904 t/s |
| | code | `auto-mtp2-262k` | 262k | f16 | 88.4 (**96.6** on code) | 1699 t/s |
| huihui-qwen3.6-35b-a3b | **128k** (default) | `auto-mtp3-128k` | 128k | f16 | **97.0** (105.4 on code) | 931 t/s |
| | fast | `auto-mtp4-32k` | 32k | f16 | 103.6 (120.9 on code) | 939 t/s |
| | 262k | `auto-mtp3-262k` | 262k | f16 | 93.7 | 927 t/s |
| kimi-linear-48b-a3b | **262k** (default) | `auto-262k` | 262k | f16 | **104.8** | 1299 t/s |
| | fast | `auto-32k` | 32k | f16 | 106.0 | 1276 t/s |
| | 512k | `auto-512k` | 512k | f16 | 102.1 | 1259 t/s |
| nanbeige4.3-33b | **64k** (default) | `auto-64k-q8` | 64k | q8_0 | **23.1** | 1045 t/s |
| | fast | `l50-32k` | 32k | f16 | 23.8 | 1094 t/s |
| | 128k | `auto-128k-q4` | 128k | q4_0 | 22.9 | 1035 t/s |
| ripple-65b | **64k** (default) | `auto-64k` | 64k | f16 | **14.8** | 427 t/s |
| | fast | `l50-32k` | 32k | f16 | 15.2 | 417 t/s |
| | 128k | `auto-128k-q8` | 128k | q8_0 | 14.4 | 404 t/s |
| | 262k | `auto-262k-q4` | 262k | q4_0 | 14.4 | 405 t/s |
| qwen3.8-27b | **128k** (default) | `l60-mtp3-128k` | 128k | f16 | **48.3** | 917 t/s |
| | fast | `layer-auto-mtp3-32k` | 32k | f16 | 50.2 | 952 t/s |
| | code | `auto-dsp15-32k` (DSpine) | 32k | f16 | 49.2 (**71.3** on code) | 759 t/s |
| | 262k | `l65-mtp3-262k-q4-ub256` | 262k | q4_0 | 46.0 | 846 t/s |

All presets pass the image test (vision models) and a ~7k-token prompt. Per-model findings:

- **DSpine** (tencent's 16-token drafter for Qwen3.8-27B, `Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf`, kernel `bin-dspine` =
  the tested source + the DSpine commit of Ajay9o9/llama.cpp `dspine`, `kernel/patches/0003`): best on code, worse on chat.

  | target, 32k f16 | DSpine 15 drafted: code / chat / mean | DSpine 7 | MTP 3 mean |
  |---|---|---|---|
  | Qwen3.8-27B UD-Q5_K_XL | **71.3** / 27.2 / 49.2 (24% accepted) | 58.0 / 28.9 / 43.5 | 50.2 |
  | OrcaCyber Mix-15.7G | **78.2** / 26.1 / 52.1 (23% accepted) | 66.2 / 29.0 / 47.6 | 57.3 |

  It also works on OrcaCyber (a Qwen3.8-27B fine-tune), so both get a `code` preset with it. q8 KV is slower (47.1);
  OrcaCyber runs it at 128k (52.1), Qwen3.8-27B Q5_K_XL does not fit the drafter at 128k.
- **OrcaCyber**: same architecture as Qwen3.8-27B. MTP 3-4 beats its DFlash2 drafter (50.6-51.1). Both GPUs: 57.3 tok/s vs
  47.5 on the A4500 alone. 262k f16 KV does not fit; q8 does, with both cards nearly full.
- **Qwen3.6-35B-A3B**: unsloth's MTP build (MTP head in the file). MTP 3 95 tok/s vs 88 without (MTP 4: 112 on code,
  78 on chat). Splits within 3 tok/s. 262k f16 KV runs text (93.6) but the vision encoder runs out of memory on the
  A4500, so the 262k preset uses q8 KV.
- **Huihui Qwen3.6-35B-A3B abliterated** (Q4_K, 21.7 GB, MTP kept): as fast as the unsloth Q5 (MTP 3 102 tok/s at 32k,
  MTP 4 104, no draft 96) and 3.6 GB smaller, so 262k f16 KV fits with the vision encoder (93.7). Not run on the hard suite.
- **Gemma-4-31B-it ARA heretic** (richardyoung's refusal-removed fine-tune, mradermacher i1-Q5_K_M 21.9 GB, dense,
  vision; main kernel) with Google's MTP drafter for the base model (`gemma-4-31B-it-assistant.Q8_0.gguf` from
  AtomicChat, 0.5 GB): the fine-tune still accepts 61-76% of its drafts, so MTP 4 gives 1.77x (43.8 vs 24.8 tok/s at 32k;
  MTP 2 42.8, 3 42.7, 6 37.7). Splits within 1 tok/s (auto, 50/50, 60/40). Its global layers keep big KV: 64k f16 fits
  (19.6 + 14.2 GB), 128k needs q8 (39.3) or q4 (38.6), 262k does not fit even with q4. GPU log of the whole tune (1 s):
  both cards peak at 100% and average ~50% while busy (layer split = they take turns); V100 power spikes to 271 W over
  its 230 W limit and 84 C, A4500 170 W / 66 C. No crash. Not run on the hard suite.
- **Qwen3.8-27B, unsloth update of 2026-11-12** (`Qwen3.8-27B-UD-Q5_K_XL.gguf` changed in the repo; our old copy is
  kept, so the older results stay valid): MTP built in (72-74% accepted). 45/55 split runs out of V100 memory at 128k and
  262k, so the long-context presets use 60/40 and 65/35. 128k f16 60/40 47.2, 65/35 46.3; 262k q4 with ub 256 45.3; 262k q8
  60/40 does not start. Not run on the hard suite yet.
- **Qwen3.8-27B UD-Q4_K_M** (same repo, 16.6 GB; hard suite 90.8 on `128k`, the top of the leaderboard, with instruction following at 42, the lowest of the top models): about 13% faster than Q5 (53.5 vs 47.2 at 128k). 262k q8 fits at 60/40
  (51.7). Its 128k run with the 45/55 split crashed the CUDA process (the V100 was full); the 60/40 split works.
  GPU log (1 s) for the Q5 tune: V100 peaks 265 W over its 230 W limit at 84 C; A4500 167 W / 66 C. No crash.
- **Qwen3.8-40B Ultimate Heretic Turbo-Brilliance** (DavidAU: Qwen3.8-27B heretic fine-tune expanded to 40B and trained
  again; `MTP-Q4_K_M` 24.2 GB, dense, MTP built in, vision; main kernel): MTP gives 1.65x (MTP 2 35.6, 3 35.3, 4 34.9 tok/s
  vs 21.5 without, 58-74% accepted). Splits within 0.3 tok/s (auto, 50/50, 55/45, 60/40). Its KV is large (128k f16 needs
  ~14 GB), so 64k f16 is the default; 128k runs with q4 KV (34.5) or q8 KV at 60/40 (32.3, A4500 full at 20.2 GB);
  128k q8 at 55/45 crashed on the image test; 262k does not fit even with q4. Not run on the hard suite.
- **Muse Glimmer 30B** (Meta, unsloth UD-Q5_K_XL 21.8 GB, dense, vision; runs on the main kernel): 52 layers, 39 of them
  sliding (2048) and only 2 KV heads, so 128k f16 KV is cheap. Meta's DFlash drafter (`dflash-kquant.gguf`, 1.63 GB, block
  16) gives 1.5x: no draft 26.6, DFlash 7 31.0, 10 40.1, 12 41.3, 15 40.9 at 32k (15-23% accepted), 128k 40.6; q8 KV is
  slower (35.2); auto split beats 50/50 (36.5) and 60/40 (36.8). DFlash costs prompt speed (7k prompt 800 vs 1165 t/s).
  It always reasons first (`reasoning_strength` low..xhigh, default high; `enable_thinking=false` has no effect), so the
  tune image test now allows 1500 tokens (200 was used up by its reasoning; vision itself works).
  Hard suite: the PC went down twice at the same point (the `search` category, ~5 min in, `128k` preset): first a
  bugcheck 0x133 (DPC watchdog, corrected PCIe errors on the A4500's root port), then a freeze. Before that it had coding
  87.5, reasoning 100, instruction following 91.7. Not rerun; every other model passes the suite.
- **Huihui Gemma-4-26B-A4B-it QAT abliterated** (Q4_K, 16.8 GB, separate MTP drafter `mtp-ggml-model-bf16.gguf`,
  arch `gemma4-assistant`; runs on the main kernel): ~4B active, fastest prompt processing here (1700-1900 t/s). The MTP
  drafter barely pays off (no draft 86-89, MTP 1 84.2, MTP 2 86.8, MTP 4 82.6, MTP 6 74.4 at 32k): MTP 2 helps code
  (95-97) but costs chat (76-82 vs 86-89), so the default is no draft and `code` uses MTP 2. Most layers use sliding
  windows, so 262k f16 KV with vision costs nothing (87.5 vs 86.1 at 32k, 14.4 + 12.2 GB); q8 KV is slower (69.9).
  Hard suite on `262k`: **90.5**, the top score here (instruction following 83, tied best; vision 75).
- **MiniCPM-V 4.7**: MoE with ~3B active, so the DSpark draft is a loss here (n8 66.0, n15 54.2 vs 91.6 without, 18-32%
  accepted). 40/60 split and 262k f16 KV do not fit.
- **Kimi-Linear**: 20 linear-attention (KDA) + 7 MLA layers, so 512k context costs little. 1M does not fit (needs a 4.6 GB
  more buffer on the A4500); q8_0 KV is impossible (V head dim 72), and mixed K/V types are not supported for this model.
- **Nanbeige4.3-33B**: dense Llama (runs on the main kernel). ~23 tok/s whatever the split (50/50 23.8, auto 23.5,
  60/40 23.0). Every one of its 52 layers keeps a full KV cache (~210 KB/token f16), so context is the limit: 64k needs q8
  KV, 128k q4, 262k does not fit; with a 50/50 split the 64k/128k configs crash on the 7k prompt (V100 full).
  Hard suite: only the coding category was run (stopped on request): **81.2** on `64k` (Ripple 75, the top models 94).
- **Ripple-65B Preview**: 65B dense, so every token reads all 28 GB: ~15 tok/s whatever the split (auto 15.0, 50/50 15.2,
  60/40 14.5). Only 16 of 80 layers keep a full KV cache (the rest slide over 1024 tokens), so 262k fits with q4 KV at the
  same speed. New `ricursive` architecture: it runs on `kernel\bin-ripple`, the tested source plus AtomicChat's port
  (`kernel/patches/0002-ricursive-port-applied.patch`), built with `tools\build-kernel.bat <cuda> ripple`.

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
tools/build-kernel.bat  build llama.cpp (CUDA 12.9, sm_70 + sm_86) into kernel/bin; [cuda] <x>: kernel/src-x -> bin-x
kernel/                 SOURCE.md + patches (source and binaries are not in git); bin-ripple = + Ripple port
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
