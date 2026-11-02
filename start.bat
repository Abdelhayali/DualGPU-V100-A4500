@echo off
rem Starts a tested preset on port 8899:   start.bat <model> [preset]     (no arguments = list of models and presets)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\serve.ps1" %*
