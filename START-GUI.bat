@echo off
rem DualGPU: starts the model launcher GUI at http://127.0.0.1:7860 (opens your browser). Models are served on :8899.
rem Python: runtime\python (portable) if present, else the V100-LLM portable runtime, else python on PATH.
cd /d "%~dp0"
set "PY=%~dp0runtime\python\python.exe"
if not exist "%PY%" set "PY=D:\V100-LLM\runtime\python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0gui\launcher.py" %*
if errorlevel 1 pause
