# Starts llama-server for a model + preset from presets.json (or any config name in configs\<model>.json)
# in this window, on port 8899 (OpenAI API: http://127.0.0.1:8899/v1). Ctrl+C stops it.
# Usage: tools\serve.ps1 <model id> [preset]     (no preset = the model's default preset)
param([string] $Model, [string] $Preset)
. "$PSScriptRoot\common.ps1"
$all = Get-Content "$Root\presets.json" -Raw | ConvertFrom-Json
if (-not $Model) {
    "Usage: start.bat <model> [preset]`n"
    foreach ($p in $all.models.PSObject.Properties) {
        "  $($p.Name)   (default preset: $($p.Value.default))"
        foreach ($q in $p.Value.presets) { "      {0,-26} {1}" -f $q.name, $q.summary }
    }
    exit 1
}
$pm = $all.models.$Model
if (-not $Preset) { $Preset = if ($pm) { $pm.default } else { throw "unknown model $Model" } }
$p = @($pm.presets) | Where-Object name -eq $Preset
$name = if ($p) { $p.config } else { $Preset }
$m = Get-ModelCfg $Model
$c = @($m.configs) | Where-Object name -eq $name
if (-not $c) { throw "unknown preset/config $Preset for $Model" }
$a = Get-SrvArgs $m $c
Stop-Srv
"$Model / $Preset" + $(if ($p) { " : $($p.summary)" } else { '' })
"llama-server $($a -join ' ')`n"
& "$Kd\llama-server.exe" @a
