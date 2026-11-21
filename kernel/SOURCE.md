# Kernel (llama.cpp) source

`kernel/src` (not in git) = upstream llama.cpp `f336359` ("Add more benchmarks to llama-eval (#26174)") plus
patch 1 of `patches/llamacpp-f336359-v100-mmvq.patch` (V100 MMVQ tuning for i-quants, commit `6f7e9e2`).
This is the same source as the V100-only kernel in `D:\V100-LLM\_kernels\llamacpp` (version "build 2, commit 6f7e9e2"),
which only contains sm_70 code. The V100 tuning is compiled for sm_70 only and has no effect on the A4500.

Rebuild:

```bat
git clone <llama.cpp> kernel\src && cd kernel\src && git checkout f336359
git am ..\patches\0001-cuda-mmvq-v100-tuning-applied.patch   (exactly the tested build, commit da8fd08)
tools\build-kernel.bat
```

`tools\build-kernel.bat` builds with CUDA 12.9 (`D:\pxa\cuda-12.9`; CUDA 13 dropped Volta), `CMAKE_CUDA_ARCHITECTURES=70;86`
(V100 = sm_70, RTX A4500 = sm_86), and copies the executables plus the CUDA 12 runtime DLLs into `kernel\bin`.

`patches/0001-cuda-mmvq-v100-tuning-applied.patch` is the one commit on top of `f336359` in the tested build;
`patches/llamacpp-f336359-v100-mmvq.patch` is the original 2-patch file from the V100-only setup (patch 2 is not applied).

Prebuilt binaries (Windows x64, the exact `kernel\bin` used for every result here: llama-server/bench/cli + CUDA 12.9 runtime
and cuBLAS DLLs + MSVC runtime) are attached to the GitHub release `kernel-da8fd08`. Unzip into `kernel\bin`.

## Variant: `bin-ripple` (Ripple-65B Preview, `ricursive` architecture)

`kernel/src-ripple` = the same `da8fd08` tree plus `patches/0002-ricursive-port-applied.patch`: AtomicChat's
`ricursive-port.patch` (from `AtomicChat/Ripple-65B-Preview-GGUF`, written for llama.cpp `9e2a04a`, upstream PR #31692)
with the two architecture-list hunks merged by hand. It only adds the model (no new ggml op), so every other model runs the
same code as in `kernel/bin`. Rebuild:

```bat
git -C kernel\src worktree add ..\src-ripple da8fd08
git -C kernel\src-ripple apply ..\patches\0002-ricursive-port-applied.patch
tools\build-kernel.bat D:\pxa\cuda-12.9 ripple      (-> kernel\bin-ripple)
```

## Variant: `bin-dspine` (DSpine drafter for Qwen3.8-27B)

`kernel/src-dspine` = `da8fd08` plus `patches/0003-dspine-draft-applied.patch`: commit `d493f84` ("dflash : add DSpine
draft model support") of https://github.com/Ajay9o9/llama.cpp/tree/dspine, without its `llama-quant.cpp` hunk (it only
keeps DSpine tensors in BF16 when quantizing). Build: `tools\build-kernel.bat D:\pxa\cuda-12.9 dspine`. Configs pick it per
config with `"kernel": "dspine"` (draft type `draft-dflash`, `--spec-draft-n-max 15`).

Models pick the Ripple build with `"kernel": "ripple"` in `configs/<id>.json` (tools) and the catalog kernel `dualgpu-ripple` (GUI).

## Variant: `bin-inklings` (Inkling-2-Mini, realugbun's inkling2-speed fork) - used by the Inkling presets

`kernel/src-inklings` = https://github.com/realugbun/llama.cpp branch `inkling2-speed` at `7669d899`: upstream `e50c254a`
(the `bin-inklingm` source) plus 4 commits - the depth states normalized once per block, each glue op chain kept together
in the graph and run as one fused CUDA kernel (`ggml/src/ggml-cuda/inkling2.cu`), a faster pad for narrow rows. Other
models are unaffected. Here: A4500 36-37 -> 53-58 tok/s, V100 33 -> 48, prompts about 1.4-1.9x.

```bat
git -C kernel\src fetch https://github.com/realugbun/llama.cpp inkling2-speed
git -C kernel\src worktree add --detach ..\src-inklings 7669d899
tools\build-kernel.bat D:\pxa\cuda-12.9 inklings      (-> kernel\bin-inklings)
```

## Variant: `bin-inklingm` (Inkling-2-Mini, merged `inkling2_loop`, upstream e50c254a)

`kernel/src-inklingm` = upstream llama.cpp `e50c254a` (2026-11-21) with Inkling-2-Mini merged (#32694, plus #32704 "share
the loop slots after the scale pass"); no V100 MMVQ patch. Loads GGUFs in the merged layout (arch `inkling2_loop`:
ggml-org/Inkling-2-Mini-GGUF, AtomicChat's files after their 2026-11-21 conversion). 37.4 tok/s on one A4500 vs 29.7 for
`bin-inkling` below.

```bat
git -C kernel\src worktree add --detach ..\src-inklingm e50c254a      (fetch https://github.com/ggml-org/llama.cpp first)
tools\build-kernel.bat D:\pxa\cuda-12.9 inklingm      (-> kernel\bin-inklingm)
```

## Variant: `bin-inkling` (superseded: Inkling-2-Mini via the unmerged PR #32695, unsloth's first GGUF layout)

`kernel/src-inkling` = upstream llama.cpp `91979450` (2026-11-20, 904 commits after `f336359`; has Inkling 2 text + vision,
the Inkling chat parser and the attention-mask fixes) plus `patches/0004-inkling2-mini-loop-pr32695-applied.patch`:
PR #32695 (Daniel Han, "Inkling-2-Mini - looped arch", 2 commits, not merged), which matches unsloth's GGUFs. The V100
MMVQ patch does not apply to this tree, so it is left out (it only tuned i-quants on the V100).

```bat
git -C kernel\src worktree add --detach ..\src-inkling 91979450      (fetch https://github.com/ggml-org/llama.cpp first)
git -C kernel\src-inkling apply ..\patches\0004-inkling2-mini-loop-pr32695-applied.patch
tools\build-kernel.bat D:\pxa\cuda-12.9 inkling      (-> kernel\bin-inkling)
```

Note: in this build CUDA device 0 is the V100 (PCI order) when CUDA_VISIBLE_DEVICES is not set; the tools set it, A4500 first.
