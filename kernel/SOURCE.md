# Kernel (llama.cpp) source

`kernel/src` (not in git) = upstream llama.cpp `f336359` ("Add more benchmarks to llama-eval (#26174)") plus
patch 1 of `patches/llamacpp-f336359-v100-mmvq.patch` (V100 MMVQ tuning for i-quants, commit `6f7e9e2`).
This is the same source as the V100-only kernel in `D:\V100-LLM\_kernels\llamacpp` (version "build 2, commit 6f7e9e2"),
which only contains sm_70 code. The V100 tuning is compiled for sm_70 only and has no effect on the A4500.

Rebuild:

```bat
git clone <llama.cpp> kernel\src && cd kernel\src && git checkout f336359
git am --3way ..\patches\llamacpp-f336359-v100-mmvq.patch   (or apply patch 1 only to match the tested build)
tools\build-kernel.bat
```

`tools\build-kernel.bat` builds with CUDA 12.9 (`D:\pxa\cuda-12.9`; CUDA 13 dropped Volta), `CMAKE_CUDA_ARCHITECTURES=70;86`
(V100 = sm_70, RTX A4500 = sm_86), and copies the executables plus the CUDA 12 runtime DLLs into `kernel\bin`.
