# Shared helpers for tools\tune.ps1 and tools\serve.ps1: turns a config (from configs\<model>.json) into
# llama-server arguments, starts/stops the server, and runs the checks every config must pass.
$Root = Split-Path $PSScriptRoot -Parent
$Kd = "$Root\kernel\bin"
$Port = 8899
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # CUDA0 = nvidia-smi 0 = RTX A4500, CUDA1 = Tesla V100

function Get-ModelCfg($id) {
    $f = "$Root\configs\$id.json"
    if (-not (Test-Path $f)) { throw "no config file $f" }
    return Get-Content $f -Raw | ConvertFrom-Json
}

# One config -> llama-server argument list. Fields (all optional except ctx):
#   gguf (overrides the model default), ctx, kv (f16/q8_0/q4_0), b, ub, sm (layer/row/tensor/none), ts ("a,b"), mg,
#   spec (mtp/dflash), n, mmgpu (vision projector on GPU, default true), extra (array of raw args)
function Get-SrvArgs($m, $c) {
    $md = "$Root\models\$($m.id)"
    $gguf = if ($c.gguf) { $c.gguf } else { $m.gguf }
    $kv = if ($c.kv) { $c.kv } else { 'f16' }
    $ub = if ($c.ub) { $c.ub } else { 256 }
    # b > ub lets both GPUs process prompt chunks at the same time (pipeline). On this PC that peak crashed the V100 / the
    # whole system twice (b 2048 / ub 512 during a 7k prompt), so b defaults to ub: the cards take turns.
    $b = if ($c.b) { $c.b } else { $ub }
    $a = @('-m', "$md\$gguf", '-ngl', 'all', '-c', "$($c.ctx)", '-np', '1', '-fa', 'on', '-b', "$b", '-ub', "$ub",
           '--jinja', '--cache-ram', '2048', '--host', '127.0.0.1', '--port', "$Port")
    if ($m.mmproj) { $a += @('--mmproj', "$md\$($m.mmproj)"); if ($c.mmgpu -eq $false) { $a += '--no-mmproj-offload' } }
    if ($m.sampling) { $a += @($m.sampling) }
    if ($kv -ne 'f16') { $a += @('-ctk', $kv, '-ctv', $kv) }
    if ($c.sm) { $a += @('-sm', $c.sm) }
    if ($c.ts) { $a += @('-ts', $c.ts) }
    if ($null -ne $c.mg) { $a += @('-mg', "$($c.mg)") }
    if ($c.spec) {
        $type = @{ mtp = 'draft-mtp'; dflash = 'draft-dflash' }[$c.spec]
        $a += @('--spec-type', $type, '--spec-draft-n-max', "$($c.n)", '--spec-draft-p-min', '0')
        if ($c.spec -eq 'dflash') { $a += @('--spec-draft-model', "$md\$($m.draft)", '--spec-draft-ngl', 'all') }
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

function Get-Vram { (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join '+' }   # MiB, GPU0+GPU1

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
            max_tokens = 200; temperature = 0; chat_template_kwargs = @{ enable_thinking = $false } } | ConvertTo-Json -Depth 8
    try { $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 300
          return [bool]($r.choices[0].message.content -match 'TESLA V100 = 16 GB') } catch { return $false }
}
