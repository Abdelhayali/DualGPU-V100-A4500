@echo off
rem Starts Inkling-2-Mini on Maya's engine (maya-ik2) on the RTX A4500, port 8899:   start-maya.bat [64k|code] [port]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\serve-maya.ps1" %*
