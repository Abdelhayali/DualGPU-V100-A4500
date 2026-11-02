# Hard suite (copied from D:\V100-LLM\eval, same tasks and scoring) for one model + config/preset.
# Starts the server on port 8899, runs tools\eval\run_eval_hard.py, writes results\hard\<model>@<config>.json.
# Usage: tools\eval.ps1 <model id> <config or preset name>
param([Parameter(Mandatory)][string] $Model, [Parameter(Mandatory)][string] $Config)
. "$PSScriptRoot\common.ps1"
$m = Get-ModelCfg $Model
$c = @($m.configs) | Where-Object name -eq $Config
if (-not $c) { throw "no config $Config in configs\$Model.json" }
New-Item -ItemType Directory -Force "$Root\results\logs", "$Root\results\hard" | Out-Null
if (-not (Start-Srv (Get-SrvArgs $m $c) "$Root\results\logs\$Model-$Config-eval.log")) { throw "server did not start" }
$skip = if ($m.mmproj) { @() } else { @('--skip', 'vision') }
python -I "$Root\tools\eval\run_eval_hard.py" --label "$Model@$Config" --port $Port --out-dir "..\..\results\hard" @skip
Stop-Srv
