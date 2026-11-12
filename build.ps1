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
#   .\build.ps1 -Yes             # installs whatever is missing without asking
#
# Requires: Rust 1.85+, Visual Studio Build Tools (C++ / MSVC) and CUDA Toolkit
# 12.8+ (12.8 is the first with RTX 50 / sm_120). When one is missing the
# script says what to install and offers to install it (winget, or rustup-init
# for Rust). The NVIDIA driver itself is only pointed at.
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
    [switch]$Debug,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$savedPath = $env:PATH
$savedCargoHome = $env:CARGO_HOME
$savedRustupHome = $env:RUSTUP_HOME
$savedCudaPath = $env:CUDA_PATH

function Fail([string]$msg) {
    Write-Host "error: $msg" -ForegroundColor Red
    $env:PATH = $savedPath
    $env:CARGO_HOME = $savedCargoHome
    $env:RUSTUP_HOME = $savedRustupHome
    $env:CUDA_PATH = $savedCudaPath
    exit 1
}

# --- helpers for "X is missing: install it?" --------------------------------
# "yes", "no", or "cannot" (no console to ask and no -Yes).
function Confirm-Install([string]$question) {
    if ($Yes) { return "yes" }
    if (-not [Environment]::UserInteractive -or [Console]::IsInputRedirected) { return "cannot" }
    $ans = Read-Host "$question [Y/n]"
    if ($ans -eq "" -or $ans -match '^(y|yes|s|sim)$') { return "yes" }
    return "no"
}

# Explains what is missing, asks, installs; or stops with the manual guide.
function Need([string]$what, [string]$guide, [scriptblock]$install) {
    Write-Host ""
    Write-Host "Missing: $what" -ForegroundColor Yellow
    if ($install) {
        $answer = Confirm-Install "Install it now?"
        if ($answer -eq "yes") {
            $ok = $false
            try { $ok = & $install } catch { Write-Host $_ -ForegroundColor Red }
            if ($ok) { return }
            Fail "installing $what failed (see above). Install it by hand:`n$guide"
        }
        if ($answer -eq "cannot") { $guide += "`n(or run .\build.ps1 -Yes to let this script install it)" }
    }
    Fail "$what is required. To install it:`n$guide`nThen run .\build.ps1 again."
}

$hasWinget = [bool](Get-Command winget -ErrorAction SilentlyContinue)

function Install-Winget([string]$id, [string[]]$extra = @()) {
    if (-not $hasWinget) {
        Write-Host "winget not found (it comes with 'App Installer' from the Microsoft Store)."
        return $false
    }
    Write-Host "winget install $id ..."
    & winget install --id $id -e --accept-source-agreements --accept-package-agreements @extra | Out-Host
    # 0 = installed; -1978335189 = already installed, no newer version.
    return ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq -1978335189)
}

# "1.93.1" >= "1.85"
function Test-VersionAtLeast([string]$have, [string]$want) {
    $h = [version](($have -split '[^0-9.]')[0] -replace '^(\d+\.\d+).*$', '$1')
    $w = [version]$want
    return $h -ge $w
}

# --- Rust -------------------------------------------------------------------
# With Rust installed outside the default location (e.g. E:\rust\cargo), a new
# terminal does not have cargo on PATH. That folder's cargo.exe is the rustup
# proxy, which also needs RUSTUP_HOME to find the toolchain.
$MinRust = "1.85"
$RustGuide = "    winget install Rustlang.Rustup`n  or download and run rustup-init.exe from https://rustup.rs (then open a new terminal)"

function Find-Cargo {
    if (Get-Command cargo -ErrorAction SilentlyContinue) { return $true }
    $homes = @()
    if ($env:CARGO_HOME) { $homes += $env:CARGO_HOME }
    $homes += Join-Path $env:USERPROFILE ".cargo"
    foreach ($drive in (Get-PSDrive -PSProvider FileSystem).Root) {
        $homes += Join-Path $drive "rust\cargo"
    }
    $cargoHome = $homes | Where-Object { Test-Path (Join-Path $_ "bin\cargo.exe") } | Select-Object -First 1
    if (-not $cargoHome) { return $false }
    $env:CARGO_HOME = $cargoHome
    $env:PATH = (Join-Path $cargoHome "bin") + ";" + $env:PATH
    $rustupHome = Join-Path (Split-Path -Parent $cargoHome) "rustup"
    if (-not $env:RUSTUP_HOME -and (Test-Path $rustupHome)) { $env:RUSTUP_HOME = $rustupHome }
    Write-Host "Rust: $cargoHome (not on PATH; set CARGO_HOME, RUSTUP_HOME and PATH in Windows to not depend on this)"
    return $true
}

function Get-CargoVersion {
    $out = & cargo --version 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return "" }
    return ($out -split ' ')[1]
}

function Install-Rust {
    $ok = Install-Winget "Rustlang.Rustup"
    if (-not $ok) {
        $exe = Join-Path $env:TEMP "rustup-init.exe"
        Write-Host "Downloading rustup-init.exe from static.rust-lang.org ..."
        Invoke-WebRequest "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe" -OutFile $exe -UseBasicParsing
        & $exe -y --profile minimal --default-toolchain stable | Out-Host
        $ok = ($LASTEXITCODE -eq 0)
        Remove-Item $exe -ErrorAction SilentlyContinue
        if (-not $ok) { return $false }
    }
    # The installer edits the user PATH; this terminal still has the old one.
    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    $env:PATH = "$cargoBin;" + $env:PATH
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { return $false }
    & rustup default stable 2>&1 | Out-Null
    Write-Host "Rust installed: $(& cargo --version). Open a new terminal to use cargo outside this script."
    return (Test-VersionAtLeast (Get-CargoVersion) $MinRust)
}

if (-not (Find-Cargo)) {
    Need "Rust $MinRust+ (cargo)" $RustGuide { Install-Rust }
} elseif (-not (Test-VersionAtLeast (Get-CargoVersion) $MinRust)) {
    if (Get-Command rustup -ErrorAction SilentlyContinue) {
        Write-Host "Rust $(Get-CargoVersion) is older than $MinRust; updating with rustup ..."
        & rustup update stable
        if (-not (Test-VersionAtLeast (Get-CargoVersion) $MinRust)) { Fail "cargo is still older than $MinRust after rustup update. Check 'rustup show'." }
    } else {
        Need "Rust $MinRust+ (found $(Get-CargoVersion); airun uses edition 2024)" $RustGuide { Install-Rust }
    }
}
Write-Host "Rust: $(& cargo --version)"

# --- MSVC (Rust's linker and nvcc's host compiler) --------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
function Test-Msvc {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return $true }
    if (-not (Test-Path $vswhere)) { return $false }
    $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return [bool]$p
}
$MsvcGuide = "    winget install Microsoft.VisualStudio.2022.BuildTools --override `"--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`"`n  or https://visualstudio.microsoft.com/visual-cpp-build-tools/ (select 'Desktop development with C++')"
if (-not (Test-Msvc)) {
    Need "Visual Studio Build Tools with C++ (MSVC), which Rust and nvcc use to compile and link" $MsvcGuide {
        $ok = Install-Winget "Microsoft.VisualStudio.2022.BuildTools" @("--override", "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended")
        $ok -and (Test-Msvc)
    }
}

function ConvertTo-AirunArch([string]$cc) {
    # "8.6" -> "86"; "12.0" -> "120"
    return ($cc -replace '\.', '')
}

# --- pick the architectures -------------------------------------------------
$DriverHint = "https://www.nvidia.com/drivers (or the NVIDIA App), then reboot"
$mode = "auto"
if ($NoGpu) {
    $mode = "none"
} elseif ($All) {
    $mode = "all"
} elseif ($Arch -ne "") {
    $mode = "manual"
} elseif (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Warning "nvidia-smi not found, so the NVIDIA driver is probably not installed. Install it: $DriverHint. Building for every architecture meanwhile (-Arch picks them)."
    $mode = "all"
}

$archs = $Arch -replace '\s+', ','
$gpuCount = 0
if ($mode -eq "auto") {
    $rows = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    if ($LASTEXITCODE -ne 0 -or -not $rows) {
        Write-Warning "nvidia-smi failed (driver not loaded? a reboot after installing it fixes that); using the universal build (-Arch picks the architectures)."
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
# The universal build includes 120 too.
$minCuda = "11.8"
if ($mode -eq "all" -or ($mode -ne "none" -and ($archs -split ',') -contains "120")) { $minCuda = "12.8" }

# --- CUDA toolkit -----------------------------------------------------------
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

# Every nvcc build.rs would also consider, in its order.
function Get-NvccCandidates {
    $list = @()
    foreach ($v in @($env:CUDA_PATH, $env:CUDA_HOME, $env:CUDA_ROOT)) {
        if ($v -and (Test-Path (Join-Path $v "bin\nvcc.exe"))) { $list += (Join-Path $v "bin\nvcc.exe") }
    }
    $onPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($onPath) { $list += $onPath.Source }
    if (Test-Path $CudaRoot) {
        Get-ChildItem $CudaRoot -Directory |
            Sort-Object { try { [version]($_.Name.TrimStart('v')) } catch { [version]"0.0" } } -Descending |
            ForEach-Object { $n = Join-Path $_.FullName "bin\nvcc.exe"; if (Test-Path $n) { $list += $n } }
    }
    return $list
}
function Get-NvccVersion([string]$nvcc) {
    $out = (& $nvcc --version 2>$null) -join "`n"
    if ($out -match 'release (\d+\.\d+)') { return $Matches[1] }
    return "0.0"
}
# The first candidate at least $minCuda, else the first at all (Old = $true).
function Select-Nvcc {
    $all = @(Get-NvccCandidates)
    if ($all.Count -eq 0) { return $null }
    foreach ($c in $all) {
        if (Test-VersionAtLeast (Get-NvccVersion $c) $minCuda) { return @{ Path = $c; Old = $false } }
    }
    return @{ Path = $all[0]; Old = $true }
}
$CudaGuide = "    winget install Nvidia.CUDA`n  or https://developer.nvidia.com/cuda-downloads (Windows > x86_64 > exe)`n  If CUDA is installed somewhere else: set CUDA_PATH to its folder.`n  For a build without GPU: .\build.ps1 -NoGpu"
$installCuda = {
    $ok = Install-Winget "Nvidia.CUDA"
    $s = Select-Nvcc
    $ok -and $s -and -not $s.Old
}

$nvcc = $null
if ($mode -ne "none") {
    $sel = Select-Nvcc
    if (-not $sel) {
        Need "CUDA Toolkit $minCuda+ (nvcc)" $CudaGuide $installCuda
        $sel = Select-Nvcc
    } elseif ($sel.Old) {
        Need ("CUDA Toolkit $minCuda+ (found only $(Get-NvccVersion $sel.Path) at $($sel.Path); RTX 50 / sm_120 needs 12.8+." +
              "`n  Or build for older GPUs only with -Arch, e.g. -Arch 86)") $CudaGuide $installCuda
        $sel = Select-Nvcc
    }
    $nvcc = $sel.Path
    # build.rs looks in the same places; pointing it at the one checked here
    # keeps the two from disagreeing.
    $env:CUDA_PATH = Split-Path -Parent (Split-Path -Parent $nvcc)
    Write-Host "CUDA: $(Get-NvccVersion $nvcc) at $nvcc"
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
    if ($mode -ne "none") {
        Write-Host "(the airun-cuda step compiles the CUDA kernels and takes a few minutes without printing anything)"
    }
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
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Host "`nBuilt, but not verified: no NVIDIA driver. Install it ($DriverHint) and run: $bin selftest"
        return
    }
    # A CUDA installed by this run is not on this terminal's PATH yet, and
    # airun.exe needs its cudart DLL.
    if ($nvcc) { $env:PATH = (Split-Path -Parent $nvcc) + ";" + $env:PATH }
    & $bin gpu
    if ($LASTEXITCODE -ne 0) { Write-Warning "airun gpu failed. If the driver was just installed, reboot and run: $bin gpu"; exit 1 }
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
    $env:CUDA_PATH = $savedCudaPath
}
