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

Prebuilt binaries (Windows x64, the exact `kernelin` used for every result here: llama-server/bench/cli + CUDA 12.9 runtime
and cuBLAS DLLs + MSVC runtime) are attached to the GitHub release `kernel-da8fd08`. Unzip into `kernelin`.

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

Models pick it with `"kernel": "ripple"` in `configs/<id>.json` (tools) and the catalog kernel `dualgpu-ripple` (GUI).
