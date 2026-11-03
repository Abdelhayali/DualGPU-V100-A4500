# Runs launch configs from configs\<model>.json on port 8899 and records speed, VRAM and the two must-pass checks
# (test image + ~7k-token prompt). Appends one JSON line per config to results\tune-<model>.jsonl.
# Usage: tools\tune.ps1 <model id> [config name ...]      (no names = every config in the file)
#        tools\tune.ps1 <model id> -Set <set name>        (a named group from "sets" in the config file)
[CmdletBinding(PositionalBinding = $false)]
param([Parameter(Mandatory, Position = 0)][string] $Model, [string] $Set, [Parameter(ValueFromRemainingArguments)] [string[]] $Names)
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\common.ps1"
$m = Get-ModelCfg $Model
$configs = @($m.configs)
if ($Set) { $Names = @($m.sets.$Set) }
if ($Set -and -not $Names) { throw "no set $Set in configs\$Model.json" }
if ($Names) {
    $configs = @($configs | Where-Object { $Names -contains $_.name })
    if ($configs.Count -ne @($Names).Count) { throw "unknown config name(s): $($Names -join ', ')" }
}
New-Item -ItemType Directory -Force "$Root\results\logs" | Out-Null
$out = "$Root\results\tune-$Model.jsonl"; $txt = "$Root\results\tune-$Model.log"
function L($s) {
    $s = "[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $s; $s | Out-Host
    for ($i = 0; $i -lt 20; $i++) { try { Add-Content $txt $s -EA Stop; break } catch { Start-Sleep -Milliseconds 250 } }   # a reader (tail) may hold the file
}

foreach ($c in $configs) {
    $a = Get-SrvArgs $m $c
    $r = [ordered]@{ model = $Model; config = $c.name; time = (Get-Date -Format s); args = ($a -join ' ').Replace("$Root\", '') }
    L "$($c.name) : loading"      # step lines go to disk first, so a crash or power loss shows which step it hit
    if (-not (Start-Srv $a "$Root\results\logs\$Model-$($c.name).log")) {
        $r.status = 'no-start'; L "$($c.name) : does not start"
    } else {
        $r.vram_mib = Get-Vram
        L "$($c.name) : loaded, vram $($r.vram_mib) MiB; speed test"
        $s = Test-Speed
        if (-not $s) { $r.status = 'gen-failed'; L "$($c.name) : vram $($r.vram_mib), generation failed" }
        else {
            L "$($c.name) : speed $($s.mean) tok/s; image test"
            $img = Test-Image
            L "$($c.name) : image $img; 7k prompt"
            $lp = Test-LongPrompt
            $r.code = $s.code; $r.chat = $s.chat; $r.mean = $s.mean; $r.accept = $s.acc
            $r.image = $img; $r.longprompt = $lp.ok; $r.pp = $lp.pp; $r.vram_peak_mib = Get-Vram
            $r.status = if ($img -and $lp.ok) { 'pass' } else { 'FAIL' }
            L ("$($c.name) : MEAN $($s.mean) tok/s (code $($s.code) / chat $($s.chat))" + $(if ($s.acc) { ", accept $($s.acc)%" } else { '' }) +
               ", vram $($r.vram_peak_mib) MiB, image $img, 7k prompt $($lp.ok) @ $($lp.pp) t/s -> $($r.status)")
        }
    }
    Add-Content $out ($r | ConvertTo-Json -Compress)
}
Stop-Srv
