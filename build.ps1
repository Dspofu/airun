# build.ps1 — compila o airun para a GPU local (Windows PowerShell).
#
# Uso:
#   .\build.ps1                  # detecta a GPU e compila só para ela
#   .\build.ps1 -Arch 120        # força uma arquitetura (ou "86,120" para várias)
#   .\build.ps1 -All             # build universal (RTX 20 até 50, padrão do repo)
#   .\build.ps1 -NoGpu           # sem CUDA (CPU only, para quem não tem GPU)
#   .\build.ps1 -SkipCheck       # não roda gpu/selftest depois do build
#   .\build.ps1 -Debug           # build de debug (mais rápido, mais lento em runtime)
#
# Mapeia o compute capability do nvidia-smi para o código do airun:
#   7.5 -> 75 (RTX 20)   8.0 -> 80 (A100)   8.6 -> 86 (RTX 30)
#   8.9 -> 89 (RTX 40)   9.0 -> 90 (H100)   12.0 -> 120 (RTX 50)
#
# O build dedicado é menor e mais rápido de compilar, mas SÓ roda naquela GPU.
# Para distribuir, use -All.

param(
    [string]$Arch = "",
    [switch]$All,
    [switch]$NoGpu,
    [switch]$SkipCheck,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

function ConvertTo-AirunArch([string]$cc) {
    # "8.6" -> "86"; "12.0" -> "120"
    return ($cc -replace '\.', '')
}

# --- decide as arquiteturas -------------------------------------------------
$mode = "auto"
if ($NoGpu) {
    $mode = "none"
} elseif ($All) {
    $mode = "all"
} elseif ($Arch -ne "") {
    $mode = "manual"
} else {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Write-Warning "nvidia-smi nao encontrado; usando build universal."
        $mode = "all"
    }
}

$archs = $Arch
if ($mode -eq "auto") {
    $rows = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "nvidia-smi falhou; usando build universal."
        $mode = "all"
    } else {
        $set = @{}
        foreach ($row in $rows) {
            $parts = $row.Split(",")
            $name = $parts[0].Trim()
            $cc = $parts[1].Trim()
            $a = ConvertTo-AirunArch $cc
            $set[$a] = $name
            Write-Host ("  GPU: {0}  (compute {1} -> arch {2})" -f $name, $cc, $a)
        }
        $archs = ($set.Keys | Sort-Object { [int]$_ }) -join ","
    }
}

# --- monta o ambiente -------------------------------------------------------
$env:AIRUN_CUDA_ARCHS = $null
$env:AIRUN_SKIP_CUDA = $null
switch ($mode) {
    "none"   { $env:AIRUN_SKIP_CUDA = "1" }
    "all"    { Write-Host "Archs: universal (padrao do repo: 75 80 86 89 120)" }
    default  { Write-Host "Archs: $archs" }
}
if ($mode -in @("manual", "auto")) { $env:AIRUN_CUDA_ARCHS = $archs }

# --- compila ----------------------------------------------------------------
$profile = if ($Debug) { "debug" } else { "release" }
Write-Host "`ncargo build --$profile ..."
cargo build --$profile
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build falhou."
    exit 1
}

$bin = Join-Path (Join-Path $PWD "target\$profile") "airun.exe"
Write-Host "`nOK: $bin"

# --- verifica ---------------------------------------------------------------
if ($SkipCheck -or $mode -eq "none") { return }
& $bin gpu
if ($LASTEXITCODE -ne 0) { Write-Warning "airun gpu falhou."; exit 1 }
Write-Host "`nselftest (kernels vs referencia CPU) ..."
& $bin selftest --device 0
if ($LASTEXITCODE -ne 0) { Write-Error "selftest falhou."; exit 1 }
Write-Host "`nBuild verificado e pronto para usar."
