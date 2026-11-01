# Qwen4 (Qwen3.8-Flash-Next, qwen4exp) throughput: airun vs Strata vs llama.cpp.
#
# Same raw 792-token prompt (the start of AGENTS.md wrapped in a chat turn), 128 greedy tokens,
# engines alternated each round. Writes qwen4.csv and qwen4.log next to this script.
#
#   llama.cpp: a build with qwen4exp support (the stock one has none), experts on the CPU
#              (--n-cpu-moe 48), everything else on the GPUs; timings from its /completion reply.
#   Strata:    engine 0.1.44.1 with a native pack built from the same GGUF
#              (tools/iq_pack.py --gguf <shard 1> --out <pack>); its default expert profile and
#              cache, --spec 4 (lookup drafting; there is no MTP head for this fine-tune).
#   airun:     defaults (experts that do not fit in VRAM mapped from pinned RAM).
param(
    [int]$Rounds = 2,
    [int]$N = 128,
    [string]$Model = "D:\modelsAI\gguf\qwen3.8\Next\Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
    [string]$Airun = "F:\coding\ia\airun\target\release\airun.exe",
    [string]$Strata = "E:\Strata",
    [string]$StrataPack = "E:\Strata-data\packs\iq3_s-abl",
    [string]$StrataExe = "",
    [int]$Batch = 512,
    [string]$LlamaBin = "",
    [string]$Work = $env:TEMP
)
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$csv = "$here\qwen4-b$Batch.csv"
$log = "$here\qwen4-b$Batch.log"
"round,engine,prompt_tps,gen_tps" | Set-Content $csv
Set-Content -Path $log -Value "" -Encoding utf8
function Note($s) { $s; Add-Content -Path $log -Value $s -Encoding utf8 }

$text = (Get-Content "$here\..\..\AGENTS.md" -Raw).Substring(0, 2200)
$raw = "<|im_start|>user`nSummarize the following notes in three sentences.`n`n$text<|im_end|>`n<|im_start|>assistant`n"
$promptFile = "$Work\q4bench.txt"
[IO.File]::WriteAllText($promptFile, $raw)
$ids = (& $Airun tokenize $Model -f $promptFile --ids 2>$null) -join ""
$ids = $ids.Trim().TrimStart('[').TrimEnd(']') -replace '\s', ''
[IO.File]::WriteAllText("$Work\q4bench.ids", $ids)
Note "prompt: $(($ids -split ',').Count) tokens, $N generated"
$shard2 = $Model -replace '00001-of-', '00002-of-'
if (-not $StrataExe) { $StrataExe = "$Strata\engine\strata.exe" }

for ($r = 1; $r -le $Rounds; $r++) {
    # airun
    $o = cmd /c "type `"$promptFile`" | `"$Airun`" run `"$Model`" --raw -n $N --temp 0 --repeat-penalty 1.0 -c 4096 -b $Batch --stats 2>&1 1>nul"
    $line = ($o | Select-String "prompt tokens in") -join ""
    Note "round $r airun    : $line"
    if ($line -match '\(([\d.]+) tok/s\).*\(([\d.]+) tok/s\)') { "$r,airun,$($Matches[1]),$($Matches[2])" | Add-Content $csv }

    # Strata
    $env:PATH = "$Strata\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;$env:PATH"
    Push-Location $Strata
    $o = cmd /c "`"$StrataExe`" --pack `"$StrataPack`" --native `"$Model`" --ple-gguf `"$shard2`" --expert-profile `"$Strata\data\expert-profile.bin`" --expert-cache auto --prefill $Batch --spec 4 --spec-min-p 0.5 --max-context 4096 --tokens-file `"$Work\q4bench.ids`" --max-new $N 2>nul"
    Pop-Location
    $p = ($o | Select-String '^prefill\s.*->\s*([\d.]+) tok/s').Matches
    $g = ($o | Select-String '^decode\s.*->\s*([\d.]+) tok/s').Matches
    $pt = if ($p) { $p[0].Groups[1].Value } else { "" }; $gt = if ($g) { $g[0].Groups[1].Value } else { "" }
    Note "round $r Strata   : prompt $pt tok/s, gen $gt tok/s"
    if ($pt -and $gt) { "$r,strata,$pt,$gt" | Add-Content $csv }

    # llama.cpp (optional: needs a build with qwen4exp)
    if ($LlamaBin) {
        $env:PATH = "$LlamaBin;$env:PATH"
        $srv = Start-Process -FilePath "$LlamaBin\llama-server.exe" -ArgumentList "-m `"$Model`" -ngl 99 --n-cpu-moe 48 -c 4096 -np 1 -fa on --port 8094 --host 127.0.0.1" -WindowStyle Hidden -PassThru -RedirectStandardError "$Work\q4bench-llama.err"
        $deadline = (Get-Date).AddSeconds(900)
        do { Start-Sleep 5; $ok = $false; try { $ok = (Invoke-RestMethod http://127.0.0.1:8094/health -TimeoutSec 3).status -eq "ok" } catch {} } until ($ok -or $srv.HasExited -or (Get-Date) -gt $deadline)
        if ($ok) {
            $body = @{ prompt = $raw; n_predict = $N; temperature = 0; cache_prompt = $false } | ConvertTo-Json
            $res = Invoke-RestMethod -Uri http://127.0.0.1:8094/completion -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType "application/json; charset=utf-8"
            $pt = [math]::Round($res.timings.prompt_per_second, 1); $gt = [math]::Round($res.timings.predicted_per_second, 2)
            Note "round $r llama.cpp: prompt $pt tok/s, gen $gt tok/s"
            "$r,llama.cpp,$pt,$gt" | Add-Content $csv
        } else { Note "round $r llama.cpp failed to start" }
        if (-not $srv.HasExited) { Stop-Process -Id $srv.Id }
        Start-Sleep 3
    }
}
