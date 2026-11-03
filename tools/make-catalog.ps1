# Writes catalog.json for the GUI (gui\launcher.py, START-GUI.bat) from configs\<model>.json, presets.json,
# configs\files.json (sizes + hashes, tools\fill-hashes.py) and the tune results. Presets from presets.json come first;
# every other config that passed tune.ps1 follows as an extra preset. Parameters come from Get-SrvArgs, so the GUI starts
# exactly what tune.ps1 / start.bat tested. Usage: tools\make-catalog.ps1
. "$PSScriptRoot\common.ps1"
$presets = Get-Content "$Root\presets.json" -Raw | ConvertFrom-Json
$hashes = Get-Content "$Root\configs\files.json" -Raw | ConvertFrom-Json

function Get-Results($id) {   # latest tune run per config
    $r = @{}
    $f = "$Root\results\tune-$id.jsonl"
    if (Test-Path $f) { Get-Content $f | Where-Object { $_.Trim() } | ForEach-Object { $j = $_ | ConvertFrom-Json; $r[$j.config] = $j } }
    return $r
}

function Get-Params($m, $c) {   # llama-server args minus the parts the launcher adds itself (-m, --mmproj, --host, --port)
    $a = @(Get-SrvArgs $m $c); $md = "$Root\models\$($m.id)\"; $out = @(); $skip = @('-m', '--mmproj', '--host', '--port')
    for ($i = 0; $i -lt $a.Count; $i++) {
        $flag = $a[$i]; $val = ''
        if ($i + 1 -lt $a.Count -and $a[$i + 1] -notmatch '^--?[a-z]') { $val = $a[$i + 1]; $i++ }
        if ($skip -contains $flag) { continue }
        if ($val.StartsWith($md)) { $val = '{model:' + $val.Substring($md.Length) + '}' }
        $out += , @($flag, $val, '')
    }
    return , $out
}

function Get-Bench($r) {
    if (-not $r -or -not $r.mean) { return '' }
    $s = "$($r.mean) tok/s (code $($r.code) / chat $($r.chat))"
    if ($r.accept) { $s += ", draft accept $($r.accept)%" }
    if ($r.pp) { $s += ", 7k prompt $($r.pp) t/s" }
    return "$s, VRAM $($r.vram_peak_mib) MiB (A4500+V100)"
}

$models = @()
foreach ($f in Get-ChildItem "$Root\configs\*.json" -Exclude files.json) {
    $m = Get-Content $f.FullName -Raw | ConvertFrom-Json
    $res = Get-Results $m.id
    $pm = $presets.models.($m.id)
    $files = @($m.files | ForEach-Object {
        $h = $hashes."$($m.id)/$($_.file)"
        if (-not $h) { throw "no size/hash for $($m.id)/$($_.file): run python tools\fill-hashes.py $($m.id)" }
        [ordered]@{ name = $_.file; repo = $_.repo; size = $h.size; sha256 = $h.sha256; url = $h.url } })
    # files the configs never use (no draft config -> no draft file) are still listed, so "Download missing" gets them all
    $list = @()
    foreach ($p in @($pm.presets)) {
        if (-not $p) { continue }
        $c = @($m.configs) | Where-Object name -eq $p.config
        $list += [ordered]@{ id = $p.name; name = "$($p.name): $($p.summary)"; kernel = 'dualgpu'; kernels = @('dualgpu')
                             bench = Get-Bench $res[$c.name]; params = Get-Params $m $c }
    }
    $used = @($pm.presets | ForEach-Object { $_.config })
    foreach ($c in @($m.configs)) {
        $r = $res[$c.name]
        if ($used -contains $c.name -or -not $r -or $r.status -ne 'pass') { continue }
        $list += [ordered]@{ id = $c.name; name = "config $($c.name) (tested)"; kernel = 'dualgpu'; kernels = @('dualgpu')
                             bench = Get-Bench $r; params = Get-Params $m $c }
    }
    if (-not $list) { Write-Warning "$($m.id): no tested presets yet, skipped"; continue }
    $e = [ordered]@{ id = $m.id; name = $m.name; desc = $m.notes; files = $files; model = $m.gguf; presets = $list }
    if ($m.mmproj) { $e.mmproj = $m.mmproj }
    $models += $e
    "$($m.id): $($list.Count) presets"
}

$cat = [ordered]@{
    server_port = $Port
    kernels = [ordered]@{ dualgpu = [ordered]@{
        name = 'llama.cpp da8fd08 (CUDA 12.9, sm_70 + sm_86: A4500 + V100)'
        desc = 'llama.cpp f336359 + V100 MMVQ tuning, built by tools\build-kernel.bat. kernel\bin.'
        exe = 'llama-server.exe'
        download = @([ordered]@{ url = 'https://github.com/Abdelhayali/DualGPU-V100-A4500/releases/download/kernel-da8fd08/llamacpp-da8fd08-cuda12.9-sm70-sm86-win64.zip' }) } }
    models = $models
}
[IO.File]::WriteAllText("$Root\catalog.json", ($cat | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
"wrote catalog.json ($($models.Count) models)"
