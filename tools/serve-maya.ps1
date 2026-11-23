# Starts Maya's Inkling-2-Mini engine (maya-ik2; see kernel\SOURCE.md) with its OpenAI-compatible server on one card,
# in this window. Ctrl+C stops it.
# Usage: tools\serve-maya.ps1 [preset] [port]      presets: configs\maya\inkling-2-mini-<preset>.json
#   64k (default), code: the first Maya build (kernel\build-maya, every expert in VRAM, A4500)
#   v100: Maya v1.1.73 (kernel\build-maya2, its server in kernel\src-maya2; experts past VRAM in pinned RAM)
#   64k-new, 128k, 262k: Maya v1.1.74 (kernel\build-maya3, kernel\src-maya3; expert lending)
#   2gpu: Maya v1.1.75 (kernel\build-maya4, kernel\src-maya4): A4500 + V100 with --gpus 2
#   262k-fast, 128k-fast: Maya v1.1.75, the NVFP4-tail file (tools\ik2_nvfp4_tail.py) + GGUF drafter, A4500: 150+ tok/s on code
# The template's "card" names the card (default A4500) and "src" the Maya source folder (default src-maya). Another port
# leaves other servers running, e.g. start-maya.bat v100 8900 (V100) next to start-maya.bat 64k (A4500).
param([string] $Preset = '64k', [int] $ListenPort = 8899)
. "$PSScriptRoot\common.ps1"
$src = "$Root\configs\maya\inkling-2-mini-$Preset.json"
if (-not (Test-Path $src)) { throw "no preset $Preset (configs\maya\inkling-2-mini-*.json)" }
$cfg = (Get-Content $src -Raw).Replace('{root}', $Root.Replace([string][char]92, '/'))
$j = $cfg | ConvertFrom-Json
$card = if ($j.card) { $j.card } else { 'A4500' }
$msrc = if ($j.src) { $j.src } else { 'src-maya' }
$tmp = "$env:TEMP\maya-inkling-2-mini-$Preset.json"
[IO.File]::WriteAllText($tmp, $cfg, [Text.UTF8Encoding]::new($false))
$py = 'D:\venvs\maya\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "no Python venv at $py (python -m venv D:\venvs\maya; pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil 'jsonschema>=4.18,<5' huggingface_hub)" }
if ($ListenPort -eq $Port) { Stop-Srv }
$env:CUDA_VISIBLE_DEVICES = @(($Gpus | Where-Object name -match $card).uuid) -join ','   # "card": "A4500|V100" = both, A4500 first (--gpus 2)
$env:PATH = (Split-Path $j.exe -Parent) + ';' + $env:PATH   # the engine's CUDA runtime DLLs sit beside it
"maya-ik2 / $Preset : $($j.note)  (port $ListenPort, $card only)"
Push-Location "$Root\kernel\$msrc"
try { & $py serve\server.py --engine maya --config $tmp --host 127.0.0.1 --port $ListenPort }
finally { Pop-Location }
