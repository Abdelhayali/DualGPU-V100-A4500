@echo off
rem Builds kernel\src (llama.cpp) with CUDA for both cards: V100 = sm_70, A4500 = sm_86.
rem CUDA 13 dropped Volta, so this uses the CUDA 12.9 toolkit. Output: kernel\bin (exe + runtime DLLs).
rem Usage: tools\build-kernel.bat [cuda root] [variant]   (default D:\pxa\cuda-12.9)
rem   variant X builds kernel\src-X into kernel\bin-X, e.g. "ripple" = the tested source + the Ripple-65B port patch
setlocal
set ROOT=%~dp0..
set CUDA=%~1
if "%CUDA%"=="" set CUDA=D:\pxa\cuda-12.9
set V=
if not "%~2"=="" set V=-%~2
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set PATH=%CUDA%\bin;%PATH%
cmake -S "%ROOT%\kernel\src%V%" -B "%ROOT%\kernel\build%V%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;86" -DCUDAToolkit_ROOT="%CUDA%" -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" ^
  -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON ^
  -DLLAMA_BUILD_TESTS=OFF || exit /b 1
cmake --build "%ROOT%\kernel\build%V%" --config Release -j 24 --target llama-server llama-bench llama-cli || exit /b 1
if not exist "%ROOT%\kernel\bin%V%" mkdir "%ROOT%\kernel\bin%V%"
copy /y "%ROOT%\kernel\build%V%\bin\llama-*.exe" "%ROOT%\kernel\bin%V%\" >nul
rem CUDA runtime + MSVC/OpenMP DLLs: the same set the V100-LLM kernel ships with (copied, the original stays untouched)
copy /y "D:\V100-LLM\_kernels\llamacpp\*.dll" "%ROOT%\kernel\bin%V%\" >nul
"%ROOT%\kernel\bin%V%\llama-server.exe" --version
