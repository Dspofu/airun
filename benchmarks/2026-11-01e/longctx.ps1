# Long prompts and a large context: airun vs llama.cpp on Qwen3.8-27B NVFP4,
# both GPUs (layer split), 512-token prompt batches on both engines.
#
#   prefill N        : one prompt of N tokens into an empty cache
#   decode 128 @ D   : 128 greedy tokens with D tokens already in the cache
#
# airun: `airun bench` (no MTP drafting in bench). llama.cpp: `llama-bench`
# with flash attention, which has no MTP either. The 128k runs get their own
# invocation with a context sized for them and no warm-up (the warm-up would
# double a minute-long run; the shorter runs before them warm the GPUs).
# Writes longctx.csv and longctx.log next to this script.
param(
    [string]$Model = "D:\modelsAI\gguf\qwen3.8\Qwen3.8-27B-NVFP4-Q5K-mtp-yarn1m.gguf",
    [string]$Llama = "D:\win-llamacpp",
    [int]$Rounds = 2
)
$ErrorActionPreference = "Continue"
$airun = Join-Path $PSScriptRoot "..\..\target\release\airun.exe"
$log = Join-Path $PSScriptRoot "longctx.log"
$csv = Join-Path $PSScriptRoot "longctx.csv"
Set-Content $log "" -Encoding utf8
function Note($s) { "[$(Get-Date -Format HH:mm:ss)] $s" | Tee-Object -FilePath $log -Append }
$rows = @()

function Airun($label, [string[]]$a) {
    Note "airun bench $label"
    $o = & $airun bench $Model @a 2>&1 | Out-String
    Add-Content $log $o -Encoding utf8
    foreach ($m in [regex]::Matches($o, "(prefill \d+|decode \d+ @ \d+)\s+([\d.]+) t/s")) {
        $script:rows += [pscustomobject]@{ round = $r; engine = "airun"; test = $m.Groups[1].Value; tps = [double]$m.Groups[2].Value }
    }
}

function Llama($label, [string[]]$a) {
    Note "llama-bench $label"
    $o = & "$Llama\llama-bench.exe" -m $Model -fa 1 -ub 512 -b 512 -r 1 -o csv @a 2>$null
    Add-Content $log ($o -join "`n") -Encoding utf8
    foreach ($x in ($o | ConvertFrom-Csv)) {
        $t = if ([int]$x.n_prompt -gt 0) { "prefill $($x.n_prompt)" } else { "decode $($x.n_gen) @ $([math]::Max(16, [int]$x.n_depth))" }
        $script:rows += [pscustomobject]@{ round = $r; engine = "llama.cpp"; test = $t; tps = [double]$x.avg_ts }
    }
}

for ($r = 1; $r -le $Rounds; $r++) {
    Airun "8k/32k" @("--prefill", "8192,32768", "--decode", "128", "--decode-at", "16,32768", "-c", "36864", "-b", "512", "--runs", "1", "--warmup", "1")
    Llama "8k/32k" @("-p", "8192,32768", "-n", "0")
    Llama "decode @16/32k" @("-p", "0", "-n", "128", "-d", "0,32768")
    Airun "128k" @("--prefill", "131072", "--decode", "128", "--decode-at", "131072", "-c", "131328", "-b", "512", "--runs", "1", "--warmup", "0")
    Llama "128k" @("-p", "131072", "-n", "0")
    Llama "decode @128k" @("-p", "0", "-n", "128", "-d", "131072")
    $rows | Export-Csv -NoTypeInformation $csv
}

Note "done"
$rows | Group-Object test | ForEach-Object {
    $a = ($_.Group | Where-Object engine -eq "airun" | Measure-Object tps -Maximum).Maximum
    $l = ($_.Group | Where-Object engine -eq "llama.cpp" | Measure-Object tps -Maximum).Maximum
    [pscustomobject]@{ test = $_.Name; airun = $a; llama = $l; ratio = if ($l) { [math]::Round($a / $l, 2) } else { $null } }
} | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $log -Append
