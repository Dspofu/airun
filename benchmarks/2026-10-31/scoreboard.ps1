# Interleaved scoreboard, airun vs llama.cpp, Qwen3.8-27B NVFP4 on one GPU.
# Each round runs both engines back to back so background load hits both alike.
param(
    [string]$Model = "D:\modelsAI\gguf\qwen3.8\Qwen3.8-27B-NVFP4-Q5K-mtp-yarn1m.gguf",
    [string]$Llama = "D:\win-llamacpp",
    [int]$Rounds = 3
)
$ErrorActionPreference = "Continue"
$airun = Join-Path $PSScriptRoot "..\..\target\release\airun.exe"
$prompts = @(
    "Write a Python function that parses a CSV file and computes the average of each numeric column. Explain briefly.",
    "Explain how a hash map works, with a short example in Rust."
)
$tmp = Join-Path $env:TEMP "airun-score"; New-Item -ItemType Directory -Force $tmp | Out-Null
$rows = @()
$log = Join-Path $PSScriptRoot "scoreboard.log"
function Step($what) { Add-Content $log "[$(Get-Date -Format HH:mm:ss)] $what" }
Set-Content $log ""
for ($r = 1; $r -le $Rounds; $r++) {
    Step "round $r airun bench"
    # Throughput: prefill 512/2048 and decode 128 tokens.
    $a = & $airun bench $Model --prefill 512,2048 --decode 128 -c 4096 -b 512 --runs 1 --warmup 1 2>&1 | Out-String
    foreach ($m in [regex]::Matches($a, "(prefill \d+|decode \d+ @ \d+)\s+([\d.]+) t/s")) {
        $rows += [pscustomobject]@{ round = $r; engine = "airun"; test = $m.Groups[1].Value; tps = [double]$m.Groups[2].Value }
    }
    Step "round $r llama-bench"
    $l = & "$Llama\llama-bench.exe" -m $Model -dev CUDA0 -p 512,2048 -n 128 -r 1 -fa 1 -o csv 2>$null | ConvertFrom-Csv
    foreach ($x in $l) {
        $t = if ([int]$x.n_prompt -gt 0) { "prefill $($x.n_prompt)" } else { "decode $($x.n_gen) @ 16" }
        $rows += [pscustomobject]@{ round = $r; engine = "llama.cpp"; test = $t; tps = [double]$x.avg_ts }
    }
    Step "round $r llama-server"
    # Speculative decoding with the model's MTP head, greedy, same raw prompts.
    $srv = Start-Process -FilePath "$Llama\llama-server.exe" -ArgumentList "-m `"$Model`" -dev CUDA0 -ngl 99 -c 4096 -np 1 -ub 256 -fa on --spec-type draft-mtp --spec-draft-n-max 3 --port 8091 --host 127.0.0.1" -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddSeconds(240)
    do { Start-Sleep 3; $ok = $false; try { $ok = (Invoke-RestMethod http://127.0.0.1:8091/health -TimeoutSec 3).status -eq "ok" } catch {} } until ($ok -or $srv.HasExited -or (Get-Date) -gt $deadline)
    Step "round $r llama-server ready=$ok exited=$($srv.HasExited)"
    $i = 0
    foreach ($q in $prompts) {
        $i++
        $raw = "<|im_start|>user`n$q<|im_end|>`n<|im_start|>assistant`n"
        $body = @{ prompt = $raw; n_predict = 400; temperature = 0; cache_prompt = $false } | ConvertTo-Json
        $res = Invoke-RestMethod -Uri http://127.0.0.1:8091/completion -Method Post -Body $body -ContentType "application/json"
        $rows += [pscustomobject]@{ round = $r; engine = "llama.cpp"; test = "mtp prompt $i"; tps = [double]$res.timings.predicted_per_second }
    }
    if (-not $srv.HasExited) { Stop-Process -Id $srv.Id }
    Step "round $r airun mtp"
    $i = 0
    foreach ($q in $prompts) {
        $i++
        [IO.File]::WriteAllText("$tmp\p$i.txt", "<|im_start|>user`n$q<|im_end|>`n<|im_start|>assistant`n")
        $o = cmd /c "type $tmp\p$i.txt | `"$airun`" run `"$Model`" --raw -n 400 --temp 0 --repeat-penalty 1.0 --stats 2>&1 1>nul"
        $m = [regex]::Match(($o | Out-String), "generated in \d+ ms \(([\d.]+) tok/s\)")
        $rows += [pscustomobject]@{ round = $r; engine = "airun"; test = "mtp prompt $i"; tps = [double]$m.Groups[1].Value }
    }
}
$rows | Export-Csv -NoTypeInformation (Join-Path $PSScriptRoot "scoreboard.csv")
$rows | Group-Object test | ForEach-Object {
    $a = ($_.Group | ? engine -eq "airun" | Measure-Object tps -Maximum).Maximum
    $l = ($_.Group | ? engine -eq "llama.cpp" | Measure-Object tps -Maximum).Maximum
    [pscustomobject]@{ test = $_.Name; airun = $a; llama = $l; ratio = [math]::Round($a / $l, 3) }
} | Format-Table -AutoSize
