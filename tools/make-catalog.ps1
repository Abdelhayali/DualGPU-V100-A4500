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
    # a config with its own gguf keeps its -m: the launcher puts the model's default -m first, llama.cpp takes the last one
    $a = @(Get-SrvArgs $m $c); $md = "$Root\models\$($m.id)\"; $out = @(); $skip = @('--mmproj', '--host', '--port')
    if (-not $c.gguf) { $skip += '-m' }
    for ($i = 0; $i -lt $a.Count; $i++) {
        $flag = $a[$i]; $val = ''
        if ($i + 1 -lt $a.Count -and $a[$i + 1] -notmatch '^--?[a-z]') { $val = $a[$i + 1]; $i++ }
        if ($skip -contains $flag) { continue }
        if ($val.StartsWith($md)) { $val = '{model:' + $val.Substring($md.Length) + '}' }
        $out += , @($flag, $val, '')
    }
    return , $out
}

function Get-Kernel($m, $c) {   # catalog kernel id: dualgpu, or dualgpu-<variant> for a config/model on kernel\bin-<variant>
    $v = if ($c.kernel) { $c.kernel } else { $m.kernel }
    if ($v) { "dualgpu-$v" } else { 'dualgpu' }
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
    if ($m.wip) { Write-Warning "$($m.id): work in progress ($($m.wip)), skipped"; continue }
    $res = Get-Results $m.id
    $pm = $presets.models.($m.id)
    $files = @($m.files | ForEach-Object {
        $h = $hashes."$($m.id)/$($_.file)"
        if (-not $h) { throw "no size/hash for $($m.id)/$($_.file): run python tools\fill-hashes.py $($m.id)" }
        [ordered]@{ name = $_.file; repo = $_.repo; size = $h.size; sha256 = $h.sha256; url = $h.url } })
    # files the configs never use (no draft config -> no draft file) are still listed, so "Download missing" gets them all
    $list = @()
    if ($m.engine -eq 'maya') {
        # Maya (its own engine + Python server): each preset is a template in configs\maya, started by name
        foreach ($p in @($m.maya.presets)) {
            $list += [ordered]@{ id = $p.name; name = "$($p.name): $($p.summary)"; kernel = 'maya'; kernels = @('maya')
                                 bench = $p.bench; params = @(, @('--preset', $p.name, '')) }
        }
        $models += [ordered]@{ id = $m.id; name = $m.name; desc = $m.notes; files = $files; model = $m.gguf
                               mmproj = $m.mmproj; maya = [ordered]@{ template = $m.maya.template }; presets = $list }
        "$($m.id): $($list.Count) presets (Maya)"
        continue
    }
    foreach ($p in @($pm.presets)) {
        if (-not $p) { continue }
        $c = @($m.configs) | Where-Object name -eq $p.config
        $k = Get-Kernel $m $c
        $list += [ordered]@{ id = $p.name; name = "$($p.name): $($p.summary)"; kernel = $k; kernels = @($k)
                             bench = Get-Bench $res[$c.name]; params = Get-Params $m $c }
    }
    $used = @($pm.presets | ForEach-Object { $_.config })
    foreach ($c in @($m.configs)) {
        $r = $res[$c.name]
        if ($used -contains $c.name -or -not $r -or $r.status -ne 'pass') { continue }
        $k = Get-Kernel $m $c
        $list += [ordered]@{ id = $c.name; name = "config $($c.name) (tested)"; kernel = $k; kernels = @($k)
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
        download = @([ordered]@{ url = 'https://github.com/Abdelhayali/DualGPU-V100-A4500/releases/download/kernel-da8fd08/llamacpp-da8fd08-cuda12.9-sm70-sm86-win64.zip' }) }
      'dualgpu-ripple' = [ordered]@{
        name = 'llama.cpp da8fd08 + Ripple port (ricursive architecture)'
        desc = 'The same source plus AtomicChat ricursive-port.patch (kernel\patches), built by tools\build-kernel.bat <cuda> ripple. kernel\bin-ripple.'
        exe = 'llama-server.exe'; dir = 'bin-ripple' }
      'dualgpu-dspine' = [ordered]@{
        name = 'llama.cpp da8fd08 + DSpine drafter (Qwen3.8-27B)'
        desc = 'The same source plus the DSpine commit of Ajay9o9/llama.cpp dspine (kernel\patches\0003), built by tools\build-kernel.bat <cuda> dspine. kernel\bin-dspine.'
        exe = 'llama-server.exe'; dir = 'bin-dspine' }
      'maya' = [ordered]@{
        name = 'Maya maya-ik2 (Inkling-2-Mini engine, RTX A4500 only)'
        desc = 'alesha-pro/maya-inkling-mini c520fd1 built with CUDA 13.4 for sm_86 (kernel\build-maya), its Python server in kernel\src-maya, venv D:\venvs\maya; see kernel\SOURCE.md. Every expert in VRAM, one CUDA graph per token.'
        type = 'maya'; exe = 'maya-ik2.exe'; dir = 'build-maya'; src = 'src-maya'; python = 'D:/venvs/maya/Scripts/python.exe' }
      'dualgpu-inklings' = [ordered]@{
        name = 'llama.cpp e50c254a + realugbun inkling2-speed (fused Inkling 2 CUDA kernels)'
        desc = 'https://github.com/realugbun/llama.cpp inkling2-speed 7669d899: upstream e50c254a plus 4 commits with fused CUDA kernels for Inkling 2; built by tools\build-kernel.bat <cuda> inklings. kernel\bin-inklings.'
        exe = 'llama-server.exe'; dir = 'bin-inklings' }
      'dualgpu-inklingm' = [ordered]@{
        name = 'llama.cpp master e50c254a (Inkling-2-Mini, merged #32694)'
        desc = 'Upstream llama.cpp e50c254a (2026-11-21) with the merged looped Inkling-2-Mini support, no V100 MMVQ patch; built by tools\build-kernel.bat <cuda> inklingm. kernel\bin-inklingm.'
        exe = 'llama-server.exe'; dir = 'bin-inklingm' } }
    models = $models
}
[IO.File]::WriteAllText("$Root\catalog.json", ($cat | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false))
"wrote catalog.json ($($models.Count) models)"
