# Starts llama-server for a model + preset from presets.json (or any config name in configs\<model>.json)
# in this window, on port 8899 (OpenAI API: http://127.0.0.1:8899/v1). Ctrl+C stops it.
# Usage: tools\serve.ps1 <model id> [preset] [port]     (no preset = the model's default preset)
# With the default port it first stops any running llama-server. With another port it leaves running servers alone,
# so a one-card preset (config "gpu", e.g. a "v100" preset) can run next to a model on the other card:
#   start.bat inkling-2-mini            (A4500, port 8899)
#   start.bat inkling-2-mini-uncensored v100 8900      (V100, port 8900)
param([string] $Model, [string] $Preset, [int] $ListenPort = 8899)
. "$PSScriptRoot\common.ps1"
$all = Get-Content "$Root\presets.json" -Raw | ConvertFrom-Json
if (-not $Model) {
    "Usage: start.bat <model> [preset] [port]`n"
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
$a = @(Get-SrvArgs $m $c)
$i = [array]::IndexOf($a, '--port'); $a[$i + 1] = "$ListenPort"
if ($ListenPort -eq $Port) { Stop-Srv }
$a = Use-SrvGpu $a
"$Model / $Preset" + $(if ($p) { " : $($p.summary)" } else { '' })
"llama-server $($a -join ' ')   (CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES)`n"
Start-Process "$Kd\llama-server.exe" -NoNewWindow -Wait -ArgumentList (Get-QuotedArgs $a)
