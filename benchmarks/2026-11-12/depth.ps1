# depth.ps1 -Extra "<serve flags>" -Depths 1000,4000,... : prefill and decode speed by prompt size,
# on real source code (this repository's), each prompt starting differently so nothing is reused.
# Run from the repository root:
#   .\benchmarks\2026-11-12\depth.ps1 -Model <file.gguf> -Extra "--draft 4 -c 131072"
param([Parameter(Mandatory)][string]$Model, [string]$Extra = "",
      [int[]]$Depths = @(1000, 4000, 8000, 16000, 32000, 64000, 120000), [int]$N = 200)
$s = $env:TEMP
$m = $Model
$code = (Get-ChildItem .\crates -Recurse -Include *.rs | Sort-Object FullName | ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n"
$a = @("serve", "`"$m`"", "--port", "5012", "--kv-type", "q4_0") + ($Extra.Split(" ") | Where-Object { $_ })
$p = Start-Process -NoNewWindow -FilePath .\target\release\airun.exe -ArgumentList $a -RedirectStandardError "$s\d.err" -RedirectStandardOutput "$s\d.out" -PassThru
try {
    $ok = $false
    for ($i = 0; $i -lt 150; $i++) { if ($p.HasExited) { break }; try { Invoke-RestMethod http://127.0.0.1:5012/health -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { Start-Sleep 1 } }
    if (-not $ok) { "server did not start"; Get-Content "$s\d.err" | Select-Object -Last 6; return }
    $pr = Invoke-RestMethod http://127.0.0.1:5012/props
    "ctx {0} draft {1} | {2}" -f $pr.ctx, $pr.details.mtp.draft, (($pr.details.gpus | ForEach-Object { "{0}: {1}" -f ($_.name -replace "NVIDIA GeForce ", ""), $_.layers.count }) -join " | ")
    $off = 0
    foreach ($d in $Depths) {
        $chars = [int]($d * 3.1)
        $off = ($off + 7919) % 200000
        $text = $code.Substring($off, [math]::Min($chars, $code.Length - $off))
        while ($text.Length -lt $chars) { $text += $code.Substring(0, [math]::Min($chars - $text.Length, $code.Length)) }
        $prompt = "<|im_start|>user`nRun $d. Here is some source code:`n`n$text`n`nExplain in detail what this code does and how its parts fit together.<|im_end|>`n<|im_start|>assistant`n<think>`n`n</think>`n`n"
        $body = [Text.Encoding]::UTF8.GetBytes((@{ prompt = $prompt; n_predict = $N; temperature = 0; cache_prompt = $false } | ConvertTo-Json))
        try {
            $t = (Invoke-RestMethod -Method Post http://127.0.0.1:5012/completion -Body $body -ContentType application/json -TimeoutSec 900).timings
            "depth {0,7}  cached {1,5}  prefill {2,7:N0} tok/s  decode {3,6:N1} tok/s  drafts {4}/{5}" -f $t.prompt_n, $t.cache_n, $t.prompt_per_second, $t.predicted_per_second, $t.draft_n_accepted, $t.draft_n
        } catch { "depth ~$d failed: " + $_.ErrorDetails.Message }
    }
} finally { if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; Start-Sleep 2 }
