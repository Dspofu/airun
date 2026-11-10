# build.ps1 — builds airun for this machine's GPUs (Windows PowerShell).
# On Linux, use ./build.sh (same options).
#
# Usage:
#   .\build.ps1                  # detects ALL GPUs and builds for their architectures
#   .\build.ps1 -Arch 120        # forces an architecture (or "86,120" for several)
#   .\build.ps1 -All             # universal build (RTX 20 through 50, the repo default)
#   .\build.ps1 -NoGpu           # no CUDA (CPU only, for machines without a GPU)
#   .\build.ps1 -SkipCheck       # does not run gpu/selftest after the build
#   .\build.ps1 -Debug           # debug build (compiles faster, runs slower)
#
# Maps nvidia-smi's compute capability to airun's code:
#   7.5 -> 75 (RTX 20)   8.0 -> 80 (A100)   8.6 -> 86 (RTX 30)
#   8.9 -> 89 (RTX 40)   9.0 -> 90 (H100)   12.0 -> 120 (RTX 50)
#
# GPUs of different generations in the same PC (e.g. RTX 3060 + RTX 5060 Ti) are
# detected together: the binary carries each one's code ("86,120") and each
# card runs its own. The dedicated build is smaller and faster to compile, but
# ONLY runs on those GPUs. To distribute, use -All.

param(
    [string]$Arch = "",
    [switch]$All,
    [switch]$NoGpu,
    [switch]$SkipCheck,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

# --- find Rust --------------------------------------------------------------
# With Rust installed outside the default location (e.g. E:\rust\cargo), a new
# terminal does not have cargo on PATH. That folder's cargo.exe is the rustup
# proxy, which also needs RUSTUP_HOME to find the toolchain.
$savedPath = $env:PATH
$savedCargoHome = $env:CARGO_HOME
$savedRustupHome = $env:RUSTUP_HOME
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $homes = @()
    if ($env:CARGO_HOME) { $homes += $env:CARGO_HOME }
    $homes += Join-Path $env:USERPROFILE ".cargo"
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem).Root) {
        $homes += Join-Path $drive "rust\cargo"
    }
    $cargoHome = $homes | Where-Object { Test-Path (Join-Path $_ "bin\cargo.exe") } | Select-Object -First 1
    if (-not $cargoHome) {
        Write-Error ("cargo not found on PATH nor in: " + ($homes -join ", ") +
            ". Install Rust (https://rustup.rs) or set CARGO_HOME (and RUSTUP_HOME) to the folder where it lives.")
        exit 1
    }
    $env:CARGO_HOME = $cargoHome
    $env:PATH = (Join-Path $cargoHome "bin") + ";" + $env:PATH
    $rustupHome = Join-Path (Split-Path -Parent $cargoHome) "rustup"
    if (-not $env:RUSTUP_HOME -and (Test-Path $rustupHome)) { $env:RUSTUP_HOME = $rustupHome }
    Write-Host "Rust: $cargoHome (not on PATH; set CARGO_HOME, RUSTUP_HOME and PATH in Windows to not depend on this)"
}

function ConvertTo-AirunArch([string]$cc) {
    # "8.6" -> "86"; "12.0" -> "120"
    return ($cc -replace '\.', '')
}

# --- pick the architectures -------------------------------------------------
$mode = "auto"
if ($NoGpu) {
    $mode = "none"
} elseif ($All) {
    $mode = "all"
} elseif ($Arch -ne "") {
    $mode = "manual"
} elseif (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Warning "nvidia-smi not found; using the universal build (-Arch picks the architectures)."
    $mode = "all"
}

$archs = $Arch -replace '\s+', ','
$gpuCount = 0
if ($mode -eq "auto") {
    $rows = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    if ($LASTEXITCODE -ne 0 -or -not $rows) {
        Write-Warning "nvidia-smi failed; using the universal build (-Arch picks the architectures)."
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

# --- set up the environment (restored at the end) ---------------------------
$savedArchs = $env:AIRUN_CUDA_ARCHS
$savedSkip = $env:AIRUN_SKIP_CUDA
$env:AIRUN_CUDA_ARCHS = $null
$env:AIRUN_SKIP_CUDA = $null
switch ($mode) {
    "none"   { $env:AIRUN_SKIP_CUDA = "1"; Write-Host "No CUDA (CPU only)" }
    "all"    { Write-Host "Archs: universal (repo default: 75 80 86 89 120)" }
    default  {
        $env:AIRUN_CUDA_ARCHS = $archs
        Write-Host "Archs: $archs"
        if (($archs -split ',') -notcontains "120") {
            Write-Host "  (no 120: the RTX 50 native NVFP4 kernels are left out)"
        }
    }
}

try {
    # From here on errors come from exit codes, not from stderr.
    $ErrorActionPreference = "Continue"

    # --- build --------------------------------------------------------------
    $buildProfile = if ($Debug) { "debug" } else { "release" }
    $cargoArgs = @("build")
    if (-not $Debug) { $cargoArgs += "--release" }
    Write-Host "`ncargo $($cargoArgs -join ' ') ..."
    Write-Host "(the airun-cuda step compiles the CUDA kernels and takes a few minutes without printing anything)"
    # cargo writes progress to stderr, and PowerShell 5.1 turns every stderr
    # line into a red "NativeCommandError". cmd merges stderr into stdout
    # before PowerShell sees it; the exit code is cargo's.
    & cmd /c "cargo $($cargoArgs -join ' ') 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed. See the error above; 'cargo build -vv' shows the nvcc command."
        exit 1
    }

    $bin = Join-Path (Join-Path $PWD "target\$buildProfile") "airun.exe"
    Write-Host "`nOK: $bin"

    # --- verify: selftest on each GPU (each runs its own arch's code) -------
    if ($SkipCheck -or $mode -eq "none") { return }
    & $bin gpu
    if ($LASTEXITCODE -ne 0) { Write-Warning "airun gpu failed."; exit 1 }
    if ($gpuCount -eq 0) { $gpuCount = 1 }
    for ($d = 0; $d -lt $gpuCount; $d++) {
        Write-Host "`nselftest on GPU $d (kernels vs CPU reference) ..."
        & $bin selftest --device $d
        if ($LASTEXITCODE -ne 0) { Write-Error "selftest failed on GPU $d."; exit 1 }
    }
    Write-Host "`nBuild verified and ready to use."
} finally {
    $env:AIRUN_CUDA_ARCHS = $savedArchs
    $env:AIRUN_SKIP_CUDA = $savedSkip
    $env:PATH = $savedPath
    $env:CARGO_HOME = $savedCargoHome
    $env:RUSTUP_HOME = $savedRustupHome
}
