# airun vs llama.cpp, alternated, best run of each (interference on this PC only
# ever adds time, so the fastest run is the robust estimate). Qwen3-4B Q8_0.
param([int]$Rounds = 5, [string]$Model = 'E:\models\imagem\Qwen3-4B-Q8_0.gguf')
Set-Location D:\airun
$best = @{}
function Keep($k, $v) { if (-not $best.ContainsKey($k) -or $v -gt $best[$k]) { $best[$k] = $v } }
foreach ($round in 1..$Rounds) {
    $j = cmd /c ".\target\release\airun.exe bench $Model --prefill 128,512 --decode-at 16,3072 --decode 64 --runs 3 --warmup 1 --json 2>nul" | Out-String | ConvertFrom-Json
    foreach ($r in $j.results) {
        $k = switch -regex ($r.label) { 'prefill 128' { 'p128' } 'prefill 512' { 'p512' } '@ 16$' { 'd16' } '@ 3072' { 'd3072' } }
        Keep "a$k" ([double]$r.max_tps)
    }
    $l = cmd /c "E:\llamacpp\llama-bench.exe -m $Model -p 128,512 -n 0 -ngl 99 -fa on -r 3 -o json 2>nul" | Out-String | ConvertFrom-Json
    foreach ($r in $l) { Keep "lp$($r.n_prompt)" (($r.samples_ts | Measure-Object -Maximum).Maximum) }
    $l = cmd /c "E:\llamacpp\llama-bench.exe -m $Model -p 0 -n 64 -d 16,3072 -ngl 99 -fa on -r 3 -o json 2>nul" | Out-String | ConvertFrom-Json
    foreach ($r in $l) { Keep "ld$($r.n_depth)" (($r.samples_ts | Measure-Object -Maximum).Maximum) }
}
foreach ($t in @(@('prefill 128', 'p128'), @('prefill 512', 'p512'), @('decode @16', 'd16'), @('decode @3072', 'd3072'))) {
    $a = $best["a$($t[1])"]; $l = $best["l$($t[1])"]
    '{0,-14} airun {1,8:N1}   llama.cpp {2,8:N1}   airun/llama {3:N2}' -f $t[0], $a, $l, ($a / $l)
}
