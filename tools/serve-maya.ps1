# Starts Maya's Inkling-2-Mini engine (maya-ik2, kernel\build-maya; see kernel\SOURCE.md) with its OpenAI-compatible
# server on the RTX A4500 alone, in this window. Ctrl+C stops it.
# Usage: tools\serve-maya.ps1 [preset] [port]      presets: configs\maya\inkling-2-mini-<preset>.json (64k = default, code)
# It needs every expert in VRAM (14.7 GiB + 1.1 GiB dense): the A4500 only. Another port leaves other servers running,
# e.g. start-maya.bat 64k 8899 next to start.bat inkling-2-mini-uncensored v100 8900 (V100).
param([string] $Preset = '64k', [int] $ListenPort = 8899)
. "$PSScriptRoot\common.ps1"
$src = "$Root\configs\maya\inkling-2-mini-$Preset.json"
if (-not (Test-Path $src)) { throw "no preset $Preset (configs\maya\inkling-2-mini-*.json)" }
$cfg = (Get-Content $src -Raw).Replace('{root}', $Root.Replace([string][char]92, '/'))
$tmp = "$env:TEMP\maya-inkling-2-mini-$Preset.json"
[IO.File]::WriteAllText($tmp, $cfg, [Text.UTF8Encoding]::new($false))
$py = 'D:\venvs\maya\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "no Python venv at $py (python -m venv D:\venvs\maya; pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil 'jsonschema>=4.18,<5' huggingface_hub)" }
if ($ListenPort -eq $Port) { Stop-Srv }
$env:CUDA_VISIBLE_DEVICES = ($Gpus | Where-Object name -match 'A4500').uuid
"maya-ik2 / $Preset : $((Get-Content $src -Raw | ConvertFrom-Json).note)  (port $ListenPort, A4500 only)"
Push-Location "$Root\kernel\src-maya"
try { & $py serve\server.py --engine maya --config $tmp --host 127.0.0.1 --port $ListenPort }
finally { Pop-Location }
