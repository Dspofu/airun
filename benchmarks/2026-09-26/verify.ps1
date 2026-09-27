# Full verification after the hybrid fixes and the quantized KV cache.
# Run from D:\airun. Every result lands in benchmarks/2026-09-26/v2-*.
$ErrorActionPreference = 'Continue'
$b = 'benchmarks/2026-09-26'
$exe = '.\target\release\airun.exe'
$dense = 'E:\models\imagem\Qwen3-4B-Q8_0.gguf'
$moe = 'E:\models\LLMs\qwen3.8\Qwen3.8-35B-A3B-Q4_K_M.gguf'

& $exe selftest *> "$b/v2-selftest.log"
"selftest exit=$LASTEXITCODE"; Get-Content "$b/v2-selftest.log" -Tail 2
Select-String "$b/v2-selftest.log" -Pattern 'FAIL|kv_store|over q' | ForEach-Object { $_.Line }

# Dense control: must stay at 2.6689 / 53.5% with f16, and quantized KV should cost little.
foreach ($kv in 'f16', 'q8_0', 'q4_0') {
    $r = & $exe perplexity $dense --kv-type $kv 2>&1 | Out-String
    "dense kv=$kv  " + ([regex]'mean loss\s+([\d.]+)').Match($r).Groups[1].Value + '  top1 ' + ([regex]'top-1\s+([\d.]+%)').Match($r).Groups[1].Value
}
foreach ($kv in 'f16', 'q8_0', 'q4_0') {
    $r = & $exe perplexity $moe --expert-cache 96 --kv-type $kv 2>&1 | Out-String
    "35B kv=$kv  " + ([regex]'mean loss\s+([\d.]+)').Match($r).Groups[1].Value + '  top1 ' + ([regex]'top-1\s+([\d.]+%)').Match($r).Groups[1].Value
}

& $exe run $dense -p 'What is 2+2?' -n 20 --temp 0 --stats *> "$b/v2-run.log"
Get-Content "$b/v2-run.log" -Tail 3
