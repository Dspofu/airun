# Per-SASS-instruction stall samples for one MMQ launch (Qwen3-4B prefill 512).
# Needs GPU performance counters enabled for all users (NVIDIA Control Panel).
param([string]$Kernel = 'matvec_mmq_async', [int]$Skip = 20, [int]$Top = 14)
Set-Location D:\airun
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.0\ncu.bat'
$rep = "$env:TEMP\sass_profile"
cmd /c "`"$ncu`" --kernel-name regex:$Kernel --launch-skip $Skip --launch-count 1 --section SourceCounters --section SpeedOfLight -f -o $rep .\target\release\airun.exe bench E:\models\imagem\Qwen3-4B-Q8_0.gguf --prefill 512 --decode-at 1 --decode 1 --runs 1 --warmup 0 >nul 2>&1"
$dur = cmd /c "`"$ncu`" --import $rep.ncu-rep --page details 2>nul" | Select-String 'Duration' | Select-Object -First 1
"duration: $($dur.Line.Trim())"
$csv = cmd /c "`"$ncu`" --import $rep.ncu-rep --page source --csv --print-source sass 2>nul"
$rows = $csv | Select-Object -Skip 1 | ConvertFrom-Csv
$tot = ($rows | Measure-Object -Property 'Warp Stall Sampling (All Samples)' -Sum).Sum
$i = 0
$rows | ForEach-Object { $_ | Add-Member -NotePropertyName idx -NotePropertyValue ($i++) }
$rows | ForEach-Object {
    $op = ($_.Source.Trim() -replace '^@!?P\d+\s+', '' -split '\s+')[0]
    [pscustomobject]@{ op = ($op -split '\.')[0]; s = [double]$_.'Warp Stall Sampling (All Samples)' }
} | Group-Object op | ForEach-Object {
    [pscustomobject]@{ op = $_.Name; pct = [math]::Round(100 * ($_.Group | Measure-Object s -Sum).Sum / $tot, 1); n = $_.Count }
} | Sort-Object pct -Descending | Select-Object -First $Top | Format-Table -AutoSize
