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

## Engine: `build-maya` (Maya `maya-ik2`, Inkling-2-Mini only, not llama.cpp)

`kernel/src-maya` = https://github.com/alesha-pro/maya-inkling-mini at `c520fd1` (Project Maya v1.1.70 with the Inkling-2-Mini
engine: all 3,072 experts resident in VRAM, a token = one CUDA graph of ~780 launches, NVFP4 experts decoded in registers
for the tensor cores, FP8 KV, the official DSpark drafter). Reads unsloth's `UD-Q4_K_XL` only. Built with CUDA 13.4 for
sm_86 (the A4500; the V100 is not supported). Its image encoder `maya-vision` links llama.cpp `b12590`
(`kernel/src-maya/third_party/llama.cpp-b12590`, a worktree of `kernel/src`).

```bat
git clone https://github.com/alesha-pro/maya-inkling-mini kernel\src-maya     (c520fd1)
rem in a VS 2022 x64 prompt with CUDA 13.4 on PATH:
cmake -S kernel\src-maya -B kernel\build-maya -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build kernel\build-maya --target maya-ik2
git -C kernel\src worktree add --detach ..\src-maya\third_party\llama.cpp-b12590 b12590
cmake -S kernel\src-maya\tools\vision -B kernel\build-maya-vision -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DLLAMA_DIR=kernel/src-maya/third_party/llama.cpp-b12590 -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build kernel\build-maya-vision --target maya-vision
python -m venv D:\venvs\maya  &&  D:\venvs\maya\Scripts\pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub
D:\venvs\maya\Scripts\python kernel\src-maya\tools\strata_tokenizer.py --gguf models\inkling-2-mini-maya\Inkling-2-Mini-UD-Q4_K_XL.gguf ^
  --out models\inkling-2-mini-maya\ik2 --chat-template kernel\src-maya\tools\ik2\chat_template.jinja --check
```

Files in `models\inkling-2-mini-maya`: unsloth `Inkling-2-Mini-UD-Q4_K_XL.gguf` (sha256 e62d463c...) and `mmproj-BF16.gguf`
(2c165c78...), Thinking Machines' `mtp.safetensors` (the drafter, 7c3b6513...). Start: `start-maya.bat [64k|code] [port]`.

## Engine: `build-maya2` (Maya v1.1.73 `maya-ik2`: the V100, experts past VRAM in RAM)

`kernel/src-maya2` = https://github.com/mw00/project-maya tag `v1.1.73` (commit `f9a053d`), unchanged. Volta kernels
(`mma.m8n8k4`) and the offload path (pinned RAM, a VRAM tier from `data/expert-profile-ik2mini.bin` via `IK2_WARM`).
Inkling 2 on Windows is outside Maya's preview: `-DSTRATA_IK2_WINDOWS=ON` and `IK2_UNSUPPORTED=1` at run time. CUDA 12.9
(CUDA 13 has no sm_70); its `cudart64_12` / `cublas64_12` / `cublasLt64_12` DLLs are copied beside the exe. Uses the
templates' own server (`"src": "src-maya2"`) and the same venv, tokenizer folder and `maya-vision` as `build-maya`.

```bat
git clone --depth 1 --branch v1.1.73 https://github.com/mw00/project-maya kernel\src-maya2
rem in a VS 2022 x64 prompt with CUDA 12.9 on PATH:
cmake -S kernel\src-maya2 -B kernel\build-maya2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="70;86" -DSTRATA_IK2_WINDOWS=ON -DSTRATA_BUILD_TESTS=OFF
cmake --build kernel\build-maya2 --target maya-ik2
copy %CUDA%\bin\cudart64_12.dll kernel\build-maya2 & copy %CUDA%\bin\cublas64_12.dll kernel\build-maya2 & copy %CUDA%\bin\cublasLt64_12.dll kernel\build-maya2
```

Preset `configs\maya\inkling-2-mini-v100.json` ("card": the GPU it gets alone).

## Engine: `build-maya3` (Maya v1.1.74 `maya-ik2`: expert lending, 128k / 262k at full speed)

`kernel/src-maya3` = https://github.com/mw00/project-maya tag `v1.1.74` (commit `e8d4f29`), unchanged. Built exactly as
`build-maya2` above with `src-maya3` / `build-maya3` (CUDA 12.9, `-DCMAKE_CUDA_ARCHITECTURES="70;86"`; `STRATA_IK2_WINDOWS`
is on by default since v1.1.74), the three CUDA DLLs copied beside the exe. Presets
`configs\maya\inkling-2-mini-{64k-new,128k,262k}.json` (`"src": "src-maya3"`).

## Variant: `bin-kintsugi` (Kintsugi-27B: Escha W2 / EXL3 trellis weights)

`kernel/src-kintsugi/kintsugi-engine-5f393adb5` = the release source `Kintsugi-27B-v1.1-engine-src-5f393adb5.tar.gz` of
https://github.com/professorpalmer/Kintsugi-27B (release v1.1.1, 37,360,262 bytes, sha256 `ded8fd4e...caf4a6`, MIT): a
llama.cpp fork with the Escha op, exllamav3 trellis GEMV kernels, a tiered K/V cache, a two-stage head and MTP / DFlash2
drafting. Built with their release flags for sm_86 only (their kernels target RTX 20/30/40; the V100 has no kernel
image, so the configs use `gpu: A4500` and the GUI hides the V100 from it):

```bat
tar -xzf Kintsugi-27B-v1.1-engine-src-5f393adb5.tar.gz -C kernel\src-kintsugi
rem in a VS 2022 x64 prompt with CUDA 12.9 on PATH:
cmake -S kernel\src-kintsugi\kintsugi-engine-5f393adb5 -B kernel\build-kintsugi -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real -DGGML_CUDA_MIRAI=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF ^
  -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF ^
  -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DGGML_CUDA_NCCL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build kernel\build-kintsugi --target llama-server      (copy bin\llama-*.exe + the CUDA 12 DLLs to kernel\bin-kintsugi)
```

## Engine: `strata-inkling` (Strata 0.2.19, prebuilt, Inkling-2-Mini `.strata` packs)

`kernel/strata-inkling` = `strata-inkling-0.1.65-windows-x86_64-cuda-cuda12.tar.gz` from
https://github.com/Niko1221/Strata/releases/tag/v0.2.19 (571,479,044 bytes, sha256 `27c313bc...8022f02` as in the
release's `SHA256SUMS.txt`; MIT; built by the release with CUDA 12.9 for sm_60..sm_120, so the V100 is included). It
speaks the same stdin line protocol as Maya's engine, so Maya's server (`kernel\src-maya2\serve\server.py`) drives it
for text; Maya's image-encoder command is not understood ("unknown command"), so the preset is text only.

```bat
gh release download v0.2.19 -R Niko1221/Strata -p strata-inkling-0.1.65-windows-x86_64-cuda-cuda12.tar.gz -p SHA256SUMS.txt -D kernel\strata-dl
mkdir kernel\strata-inkling && tar -xzf kernel\strata-dl\strata-inkling-0.1.65-windows-x86_64-cuda-cuda12.tar.gz -C kernel\strata-inkling
hf download strata-engine/Inkling-2-Mini-Strata Inkling-2-Mini-SQ-15.strata --local-dir models\inkling-2-mini-maya
```

Preset `configs\maya\inkling-2-mini-strata.json` (`start-maya.bat strata`). `--bench` only reads `.strata` packs.

## Engine: `build-maya4` (Maya v1.1.75 `maya-ik2`: GGUF drafter, two GPUs on Windows)

`kernel/src-maya4` = https://github.com/mw00/project-maya tag `v1.1.75` (commit `8621f7a`), unchanged. Built as `build-maya3`
plus `-DMAYA_GGML_DIR=D:/DualGPU/kernel/build-maya3/_deps/strata_llamacpp-src` (the same pinned ggml commit `3cf0325`;
its own fetch came back empty here). Drafter: `peasantsmith/Inkling-2-Mini-Maya-GGUF` revision `3702771`,
`drafter/dspark-Inkling-2-Mini-Q8_0.gguf` (sha256 `2ec78510...3773`) into `models/inkling-2-mini-maya`. Preset
`configs/maya/inkling-2-mini-2gpu.json` (`"card": "A4500|V100"`, `--gpus 2`).

Presets `262k-fast` / `128k-fast` read `models/inkling-2-mini-maya/Inkling-2-Mini-UD-Q4_K_XL-nvfp4tail.gguf`, made once from
UD-Q4_K_XL (the `Old_Support` layout) with:

```bat
python -I tools\ik2_nvfp4_tail.py models\inkling-2-mini-maya\Inkling-2-Mini-UD-Q4_K_XL.gguf models\inkling-2-mini-maya\Inkling-2-Mini-UD-Q4_K_XL-nvfp4tail.gguf
```
