# Shared helpers for tools\tune.ps1 and tools\serve.ps1: turns a config (from configs\<model>.json) into
# llama-server arguments, starts/stops the server, and runs the checks every config must pass.
$Root = Split-Path $PSScriptRoot -Parent
$Kd = "$Root\kernel\bin"
$Port = 8899
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
# Pin CUDA0 = RTX A4500, CUDA1 = Tesla V100 by UUID, whatever slots the cards sit in (configs' ts/mg/devd assume this order)
$Gpus = @(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | ForEach-Object { $u, $n = $_ -split ', ', 2; [pscustomobject]@{ uuid = $u; name = $n } } |
          Sort-Object { $_.name -notmatch 'A4500' })
$env:CUDA_VISIBLE_DEVICES = ($Gpus.uuid -join ',')

function Get-ModelCfg($id) {
    $f = "$Root\configs\$id.json"
    if (-not (Test-Path $f)) { throw "no config file $f" }
    $m = Get-Content $f -Raw | ConvertFrom-Json
    # a model on an architecture the main kernel lacks names a variant build: "kernel": "ripple" -> kernel\bin-ripple
    $script:Kd = if ($m.kernel) { "$Root\kernel\bin-$($m.kernel)" } else { "$Root\kernel\bin" }
    return $m
}

# One config -> llama-server argument list. Fields (all optional except ctx):
#   gguf (overrides the model default), ctx, kv (f16/q8_0/q4_0), b, ub, sm (layer/row/tensor/none), ts ("a,b"), mg,
#   spec (mtp/dflash), n, draft (draft file, default the model's), kernel (variant build for this config, default the
#   model's), mmgpu (vision projector on GPU, default true), cpumoe (experts of the first N layers stay in
#   system RAM, for models bigger than both GPUs), fit (no -ngl: llama.cpp's --fit picks the GPU/RAM split), extra (array of raw args)
function Get-SrvArgs($m, $c) {
    $k = if ($c.kernel) { $c.kernel } else { $m.kernel }
    $script:Kd = if ($k) { "$Root\kernel\bin-$k" } else { "$Root\kernel\bin" }
    $draft = if ($c.draft) { $c.draft } else { $m.draft }
    $md = "$Root\models\$($m.id)"
    $gguf = if ($c.gguf) { $c.gguf } else { $m.gguf }
    $kv = if ($c.kv) { $c.kv } else { 'f16' }
    $ub = if ($c.ub) { $c.ub } else { 512 }
    # b > ub lets both GPUs process prompt chunks at the same time (pipeline): 952 vs 405 t/s on the 7k prompt. With the
    # old 500 W PSU that peak cut the power (4 hard resets on 2026-11-02); after the PSU swap it passes. Set b = ub to make
    # the cards take turns again on a weak PSU.
    $b = if ($c.b) { $c.b } else { 2048 }
    $a = @('-m', "$md\$gguf", '-c', "$($c.ctx)", '-np', '1', '-fa', 'on', '-b', "$b", '-ub', "$ub",
           '--jinja', '--cache-ram', '2048', '--host', '127.0.0.1', '--port', "$Port")
    if (-not $c.fit) { $a += @('-ngl', 'all') }
    if ($m.mmproj) { $a += @('--mmproj', "$md\$($m.mmproj)"); if ($c.mmgpu -eq $false) { $a += '--no-mmproj-offload' } }
    if ($m.sampling) { $a += @($m.sampling) }
    if ($kv -ne 'f16') { $a += @('-ctk', $kv, '-ctv', $kv) }
    if ($c.sm) { $a += @('-sm', $c.sm) }
    if ($c.ts) { $a += @('-ts', $c.ts) }
    if ($null -ne $c.mg) { $a += @('-mg', "$($c.mg)") }
    if ($c.cpumoe) { $a += @('--n-cpu-moe', "$($c.cpumoe)") }
    if ($c.spec) {
        $type = @{ mtp = 'draft-mtp'; dflash = 'draft-dflash'; dspark = 'draft-dspark' }[$c.spec]
        $a += @('--spec-type', $type, '--spec-draft-n-max', "$($c.n)", '--spec-draft-p-min', '0')
        # MTP heads normally sit in the model file; m.mtp names a separate head file (Qwen3.8-Flash-Next)
        if ($c.spec -ne 'mtp') { $a += @('--spec-draft-model', "$md\$draft", '--spec-draft-ngl', 'all') }
        elseif ($m.mtp) { $a += @('--spec-draft-model', "$md\$($m.mtp)", '--spec-draft-ngl', 'all') }
        if ($c.devd) { $a += @('--spec-draft-device', $c.devd) }
        if ($kv -ne 'f16') { $a += @('--spec-draft-type-k', $kv, '--spec-draft-type-v', $kv) }
    }
    if ($c.extra) { $a += @($c.extra) }
    return $a
}

function Stop-Srv { Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force; Start-Sleep 3 }

function Start-Srv($a, $errlog) {
    Stop-Srv
    $p = Start-Process "$Kd\llama-server.exe" -PassThru -WindowStyle Hidden -RedirectStandardError $errlog -ArgumentList $a
    for ($i = 0; $i -lt 200; $i++) {
        try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3).status -eq 'ok') { return $true } } catch {}
        if ($p.HasExited) { return $false }; Start-Sleep 3
    }
    return $false
}

function Get-Vram { ($Gpus | ForEach-Object { nvidia-smi -i $_.uuid --query-gpu=memory.used --format=csv,noheader,nounits }) -join '+' }   # MiB, A4500+V100

function Gen($text, $max = 300) {
    $b = @{ messages = @(@{ role = 'user'; content = $text }); max_tokens = $max; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 5
    try { return (Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 600).timings } catch { return $null }
}

# Speed = mean tok/s of a code and a chat reply (300 tokens, temperature 0, thinking off)
function Test-Speed {
    $null = Gen 'Hi' 16
    $a = Gen 'Write a Python function that parses a CSV file and returns the average of a numeric column, with error handling.'
    $b = Gen 'Explain in a few paragraphs why the sky is blue and why sunsets are red.'
    if (-not $a -or -not $b) { return $null }
    $acc = if ($a.draft_n) { [math]::Round(100 * ($a.draft_n_accepted + $b.draft_n_accepted) / ($a.draft_n + $b.draft_n)) } else { $null }
    return [pscustomobject]@{ code = [math]::Round($a.predicted_per_second, 1); chat = [math]::Round($b.predicted_per_second, 1)
                              mean = [math]::Round(($a.predicted_per_second + $b.predicted_per_second) / 2, 1); acc = $acc }
}

# ~7k-token prompt: short prompts alone miss out-of-memory crashes in the prompt-processing buffers
function Test-LongPrompt {
    $txt = 'Here is some code:' + "`n" + ("def f(x):`n    return x * 2 + 1  # helper`n" * 400) + 'Rewrite f to be faster and explain.'
    $b = @{ messages = @(@{ role = 'user'; content = $txt }); max_tokens = 100; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 5
    try { $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 600
          return [pscustomobject]@{ ok = $true; n = $r.timings.prompt_n; pp = [math]::Round($r.timings.prompt_per_second) } } catch { return [pscustomobject]@{ ok = $false } }
}

# The test image shows shapes and the text "TESLA V100 = 16 GB"; the model must read it back
function Test-Image {
    $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("$Root\tools\test-image.png"))
    $b = @{ messages = @(@{ role = 'user'; content = @(
              @{ type = 'text'; text = 'Describe this image: list every shape with its colour, and read the text exactly.' },
              @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$b64" } }) })
            max_tokens = 1500; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 8
    try { $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 300
          return [bool]($r.choices[0].message.content -match 'TESLA V100 = 16 GB') } catch { return $false }
}
