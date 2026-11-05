# build.ps1 — compila o airun para as GPUs desta máquina (Windows PowerShell).
# No Linux, use ./build.sh (mesmas opções).
#
# Uso:
#   .\build.ps1                  # detecta TODAS as GPUs e compila para as arquiteturas delas
#   .\build.ps1 -Arch 120        # força uma arquitetura (ou "86,120" para várias)
#   .\build.ps1 -All             # build universal (RTX 20 até 50, padrão do repo)
#   .\build.ps1 -NoGpu           # sem CUDA (CPU only, para quem não tem GPU)
#   .\build.ps1 -SkipCheck       # não roda gpu/selftest depois do build
#   .\build.ps1 -Debug           # build de debug (compila mais rápido, roda mais devagar)
#
# Mapeia o compute capability do nvidia-smi para o código do airun:
#   7.5 -> 75 (RTX 20)   8.0 -> 80 (A100)   8.6 -> 86 (RTX 30)
#   8.9 -> 89 (RTX 40)   9.0 -> 90 (H100)   12.0 -> 120 (RTX 50)
#
# GPUs de gerações diferentes no mesmo PC (ex.: RTX 3060 + RTX 5060 Ti) são
# detectadas juntas: o binário leva o código de cada uma ("86,120") e cada
# placa roda o seu. O build dedicado é menor e mais rápido de compilar, mas SÓ
# roda nessas GPUs. Para distribuir, use -All.

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
} elseif (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Warning "nvidia-smi nao encontrado; usando build universal (-Arch escolhe as arquiteturas)."
    $mode = "all"
}

$archs = $Arch -replace '\s+', ','
$gpuCount = 0
if ($mode -eq "auto") {
    $rows = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    if ($LASTEXITCODE -ne 0 -or -not $rows) {
        Write-Warning "nvidia-smi falhou; usando build universal (-Arch escolhe as arquiteturas)."
        $mode = "all"
    } else {
        $set = @{}
        foreach ($row in @($rows)) {
            $parts = $row.Split(",")
            $name = $parts[0].Trim()
            $cc = $parts[1].Trim()
            $a = ConvertTo-AirunArch $cc
            $set[$a] = $name
            Write-Host ("  GPU {0}: {1}  (compute {2} -> arch {3})" -f $gpuCount, $name, $cc, $a)
            $gpuCount++
        }
        $archs = ($set.Keys | Sort-Object { [int]$_ }) -join ","
    }
}

# --- monta o ambiente (restaurado no fim) -----------------------------------
$savedArchs = $env:AIRUN_CUDA_ARCHS
$savedSkip = $env:AIRUN_SKIP_CUDA
$env:AIRUN_CUDA_ARCHS = $null
$env:AIRUN_SKIP_CUDA = $null
switch ($mode) {
    "none"   { $env:AIRUN_SKIP_CUDA = "1"; Write-Host "Sem CUDA (CPU only)" }
    "all"    { Write-Host "Archs: universal (padrao do repo: 75 80 86 89 120)" }
    default  {
        $env:AIRUN_CUDA_ARCHS = $archs
        Write-Host "Archs: $archs"
        if (($archs -split ',') -notcontains "120") {
            Write-Host "  (sem 120: os kernels NVFP4 nativos das RTX 50 ficam de fora)"
        }
    }
}

try {
    # Daqui em diante os erros vêm dos códigos de saída: no PowerShell 5.1,
    # "Stop" faria o progresso que o cargo escreve no stderr abortar o build
    # quando a saída é redirecionada.
    $ErrorActionPreference = "Continue"

    # --- compila ------------------------------------------------------------
    $buildProfile = if ($Debug) { "debug" } else { "release" }
    $cargoArgs = @("build")
    if (-not $Debug) { $cargoArgs += "--release" }
    Write-Host "`ncargo $($cargoArgs -join ' ') ..."
    & cargo @cargoArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build falhou. Veja o erro acima; 'cargo build -vv' mostra o comando do nvcc."
        exit 1
    }

    $bin = Join-Path (Join-Path $PWD "target\$buildProfile") "airun.exe"
    Write-Host "`nOK: $bin"

    # --- verifica: selftest em cada GPU (cada uma roda o código da sua arch) -
    if ($SkipCheck -or $mode -eq "none") { return }
    & $bin gpu
    if ($LASTEXITCODE -ne 0) { Write-Warning "airun gpu falhou."; exit 1 }
    if ($gpuCount -eq 0) { $gpuCount = 1 }
    for ($d = 0; $d -lt $gpuCount; $d++) {
        Write-Host "`nselftest na GPU $d (kernels vs referencia CPU) ..."
        & $bin selftest --device $d
        if ($LASTEXITCODE -ne 0) { Write-Error "selftest falhou na GPU $d."; exit 1 }
    }
    Write-Host "`nBuild verificado e pronto para usar."
} finally {
    $env:AIRUN_CUDA_ARCHS = $savedArchs
    $env:AIRUN_SKIP_CUDA = $savedSkip
}
