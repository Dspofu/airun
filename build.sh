#!/usr/bin/env bash
# build.sh — builds airun for this machine's GPUs (Linux).
# On Windows, use .\build.ps1 (same options).
#
# Usage:
#   ./build.sh                 # detects ALL GPUs and builds for their architectures
#   ./build.sh --arch 120      # forces an architecture (or "86,120" for several)
#   ./build.sh --all           # universal build (RTX 20 through 50, the repo default)
#   ./build.sh --no-gpu        # no CUDA (CPU only, for machines without a GPU)
#   ./build.sh --skip-check    # does not run gpu/selftest after the build
#   ./build.sh --debug         # debug build (compiles faster, runs slower)
#   ./build.sh --yes           # installs whatever is missing without asking
#
# Requires: Rust 1.85+, a C++ compiler (g++) and CUDA Toolkit 12.8+ (12.8 is
# the first with RTX 50 / sm_120). When one is missing the script says what to
# install and offers to install it (rustup for Rust, the distro's package
# manager for g++ and CUDA). The NVIDIA driver itself is only pointed at.
#
# GPUs of different generations in the same PC (e.g. RTX 3060 + RTX 5060 Ti) are
# detected together: the binary carries each one's code ("86,120") and each
# card runs its own. To distribute, use --all.

set -euo pipefail
cd "$(dirname "$0")"

arch="" mode="auto" check=1 debug=0 assume_yes=0
while [ $# -gt 0 ]; do
    case "$1" in
        -a|--arch) arch="${2:?--arch needs a value, e.g. --arch 86,120}"; mode="manual"; shift ;;
        --arch=*) arch="${1#*=}"; mode="manual" ;;
        --all) mode="all" ;;
        --no-gpu) mode="none" ;;
        --skip-check) check=0 ;;
        --debug) debug=1 ;;
        -y|--yes|--install-rust) assume_yes=1 ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see ./build.sh --help)" >&2; exit 2 ;;
    esac
    shift
done

die() { echo "error: $*" >&2; exit 1; }

# --- helpers for "X is missing: install it?" --------------------------------
# 0 = yes, 1 = no, 2 = cannot ask (no terminal and no --yes).
confirm() {
    [ "$assume_yes" = 1 ] && return 0
    [ -t 0 ] || return 2
    local ans
    read -r -p "$1 [Y/n] " ans || return 1
    case "$ans" in ""|y|Y|yes|s|S|sim) return 0 ;; *) return 1 ;; esac
}

# need "<what is missing>" "<how to install it by hand>" <install function>
# Explains, asks, installs; or stops with the manual guide.
need() {
    local what="$1" guide="$2" fn="$3"
    echo
    echo "Missing: $what"
    if [ -n "$fn" ]; then
        local rc=0
        confirm "Install it now?" || rc=$?
        if [ "$rc" = 0 ]; then
            "$fn" && return 0
            die "installing $what failed (see above). Install it by hand:
$guide"
        fi
        [ "$rc" = 2 ] && guide="$guide
(or run ./build.sh --yes to let this script install it)"
    fi
    die "$what is required. To install it:
$guide
Then run ./build.sh again."
}

# Distro and package manager, for the install commands.
os_id="" os_ver="" os_like=""
if [ -r /etc/os-release ]; then
    os_id="$(. /etc/os-release; echo "${ID:-}")"
    os_ver="$(. /etc/os-release; echo "${VERSION_ID:-}")"
    os_like="$(. /etc/os-release; echo "${ID_LIKE:-}")"
fi
pm=""
for p in apt-get dnf pacman zypper; do command -v "$p" >/dev/null 2>&1 && { pm="$p"; break; }; done
sudo=""
if [ "$(id -u)" != 0 ]; then
    command -v sudo >/dev/null 2>&1 && sudo="sudo"
fi

pkg_install() {
    [ -n "$sudo" ] || [ "$(id -u)" = 0 ] || { echo "sudo not found; run this as root." >&2; return 1; }
    case "$pm" in
        apt-get) $sudo apt-get update && $sudo apt-get install -y "$@" ;;
        dnf)     $sudo dnf install -y "$@" ;;
        pacman)  $sudo pacman -S --needed --noconfirm "$@" ;;
        zypper)  $sudo zypper install -y "$@" ;;
        *) return 1 ;;
    esac
}

fetch() {  # fetch URL FILE  (FILE "-" = stdout)
    if command -v curl >/dev/null 2>&1; then curl --proto =https --tlsv1.2 -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then wget -q --https-only -O "$2" "$1"
    else echo "neither curl nor wget is installed (sudo apt install curl)." >&2; return 1
    fi
}

ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }  # ver_ge A B: A >= B

# --- Rust -------------------------------------------------------------------
MIN_RUST="1.85"
RUST_GUIDE="    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  then open a new terminal (or run '. ~/.cargo/env'). See https://rustup.rs"

# A shell that did not source ~/.cargo/env (e.g. non-interactive) does not
# have cargo on PATH, even with rustup installed. The rustup proxy in that
# folder also needs RUSTUP_HOME when Rust lives outside the default location.
find_cargo() {
    command -v cargo >/dev/null 2>&1 && return 0
    for d in "${CARGO_HOME:-}" "$HOME/.cargo"; do
        if [ -n "$d" ] && [ -x "$d/bin/cargo" ]; then
            export CARGO_HOME="$d" PATH="$d/bin:$PATH"
            local rh; rh="$(dirname "$d")/rustup"
            if [ -z "${RUSTUP_HOME:-}" ] && [ ! -d "$HOME/.rustup" ] && [ -d "$rh" ]; then
                export RUSTUP_HOME="$rh"
            fi
            echo "Rust: $d (not on PATH; add '. \"$d/env\"' to ~/.bashrc to not depend on this)"
            return 0
        fi
    done
    return 1
}

cargo_version() { cargo --version 2>/dev/null | awk '{print $2}'; }
cargo_new_enough() { local v; v="$(cargo_version)"; [ -n "$v" ] && ver_ge "$v" "$MIN_RUST"; }

install_rust() {
    command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
        || { [ -n "$pm" ] && pkg_install curl; } || return 1
    local tmp; tmp="$(mktemp)"
    fetch https://sh.rustup.rs "$tmp" || { rm -f "$tmp"; return 1; }
    echo "Installing Rust (stable) with rustup into ~/.cargo ..."
    # The PATH check would stop on a distro cargo in /usr/bin; ~/.cargo/bin
    # goes first on PATH below, so rustup's toolchain wins.
    RUSTUP_INIT_SKIP_PATH_CHECK=yes sh "$tmp" -y --profile minimal --default-toolchain stable \
        || { rm -f "$tmp"; return 1; }
    rm -f "$tmp"
    export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
    export PATH="$CARGO_HOME/bin:$PATH"
    hash -r
    echo "Rust installed: $(cargo --version). Open a new terminal (or run '. ~/.cargo/env') to use cargo outside this script."
    cargo_new_enough
}

if ! find_cargo; then
    need "Rust $MIN_RUST+ (cargo)" "$RUST_GUIDE" install_rust
elif ! cargo_new_enough; then
    if command -v rustup >/dev/null 2>&1; then
        echo "Rust $(cargo_version) is older than $MIN_RUST; updating with rustup ..."
        rustup update stable || die "rustup update failed. Run 'rustup update stable' by hand and try again."
        cargo_new_enough || die "cargo is still older than $MIN_RUST after rustup update. Check 'rustup show'."
    else
        need "Rust $MIN_RUST+ (found $(cargo_version) at $(command -v cargo); airun uses edition 2024)" \
            "$RUST_GUIDE" install_rust
    fi
fi
echo "Rust: $(cargo --version) at $(command -v cargo)"

# --- pick the architectures -------------------------------------------------
gpus=0
driver_hint=""
case "$os_id $os_like" in
    *ubuntu*) driver_hint="sudo ubuntu-drivers install   (then reboot)" ;;
    *fedora*) driver_hint="enable RPM Fusion and: sudo dnf install akmod-nvidia   (then reboot)" ;;
    *arch*)   driver_hint="sudo pacman -S nvidia-open   (then reboot)" ;;
    *)        driver_hint="your distro's NVIDIA driver package, or https://www.nvidia.com/drivers" ;;
esac
if [ "$mode" = "auto" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "warning: nvidia-smi not found, so the NVIDIA driver is probably not installed." >&2
        echo "  Install it with: $driver_hint" >&2
        echo "  Building for every architecture meanwhile (--arch picks them)." >&2
        mode="all"
    elif ! rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)" || [ -z "$rows" ]; then
        echo "warning: nvidia-smi failed (driver not loaded? a reboot after installing it fixes that)." >&2
        echo "  Building for every architecture meanwhile (--arch picks them)." >&2
        mode="all"
    else
        list=""
        while IFS=, read -r name cc; do
            name="$(echo "$name" | xargs)"; cc="$(echo "$cc" | xargs)"
            a="${cc//./}"
            echo "  GPU $gpus: $name  (compute $cc -> arch $a)"
            list="$list $a"
            gpus=$((gpus + 1))
        done <<< "$rows"
        arch="$(echo $list | tr ' ' '\n' | sort -n -u | paste -sd, -)"
    fi
fi
arch="$(echo "$arch" | tr ' ' ',')"
# The universal build includes 120 too.
min_cuda="11.8"
case "$mode" in
    all) min_cuda="12.8" ;;
    auto|manual) case ",$arch," in *,120,*) min_cuda="12.8" ;; esac ;;
esac

# --- C++ compiler and CUDA toolkit ------------------------------------------
gxx_pkg() { case "$pm" in apt-get) echo g++ ;; pacman) echo gcc ;; dnf|zypper) echo gcc-c++ ;; esac; }
install_gxx() { [ -n "$pm" ] && pkg_install "$(gxx_pkg)"; }

# Every nvcc build.rs would also consider, in its order.
nvcc_candidates() {
    local d
    for d in "${CUDA_PATH:-}" "${CUDA_HOME:-}" "${CUDA_ROOT:-}"; do
        [ -n "$d" ] && [ -x "$d/bin/nvcc" ] && echo "$d/bin/nvcc"
    done
    command -v nvcc 2>/dev/null || true
    for d in /usr/local/cuda $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V -r) /opt/cuda; do
        [ -x "$d/bin/nvcc" ] && echo "$d/bin/nvcc"
    done
    return 0
}
nvcc_version() { "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p'; }

# Sets $nvcc to the first candidate at least $min_cuda (or the first at all,
# with $nvcc_old=1). Returns 1 when there is none.
pick_nvcc() {
    nvcc="" nvcc_old=0
    local c first=""
    while read -r c; do
        [ -n "$c" ] || continue
        [ -n "$first" ] || first="$c"
        if ver_ge "$(nvcc_version "$c")" "$min_cuda"; then nvcc="$c"; return 0; fi
    done < <(nvcc_candidates)
    [ -n "$first" ] || return 1
    nvcc="$first" nvcc_old=1
}

cuda_guide() {
    case "$pm" in
        apt-get) echo "    sudo apt install cuda-toolkit
  (if your distro's cuda-toolkit is older than $min_cuda, add NVIDIA's repository first:
   https://developer.nvidia.com/cuda-downloads > Linux > x86_64 > your distro > deb (network))" ;;
        pacman)  echo "    sudo pacman -S cuda" ;;
        dnf)     echo "    add NVIDIA's repository (https://developer.nvidia.com/cuda-downloads > Linux > Fedora > rpm (network))
    then: sudo dnf install cuda-toolkit" ;;
        *)       echo "    https://developer.nvidia.com/cuda-downloads" ;;
    esac
    echo "  If CUDA is installed somewhere else: CUDA_PATH=/path/to/cuda ./build.sh
  For a build without GPU: ./build.sh --no-gpu"
}

# NVIDIA's repository name for this distro ("ubuntu2404", "debian12", "fedora42").
nvidia_repo() {
    local m; case "$(uname -m)" in x86_64) m=x86_64 ;; aarch64) m=sbsa ;; *) return 1 ;; esac
    local id="$os_id"
    [ "$id" = debian ] && os_ver="${os_ver%%.*}"
    echo "https://developer.download.nvidia.com/compute/cuda/repos/${id}${os_ver//./}/$m"
}

install_cuda() {
    case "$pm" in
        apt-get)
            # The distro's own package when new enough (Ubuntu 26.04 ships 13.x);
            # otherwise NVIDIA's repository through its keyring package.
            local cand
            cand="$(apt-cache policy cuda-toolkit 2>/dev/null | awk '/Candidat|Candidate/ {print $2; exit}')"
            cand="${cand#*:}"
            if [ -n "$cand" ] && [ "$cand" != "(none)" ] && [ "$cand" != "(nenhum)" ] && ver_ge "$cand" "$min_cuda"; then
                echo "Installing cuda-toolkit $cand from the distro repository ..."
            else
                local repo; repo="$(nvidia_repo)" || return 1
                echo "Adding NVIDIA's CUDA repository ($repo) ..."
                local tmp; tmp="$(mktemp --suffix=.deb)"
                fetch "$repo/cuda-keyring_1.1-1_all.deb" "$tmp" \
                    || { rm -f "$tmp"; echo "NVIDIA has no repository for $os_id $os_ver." >&2; return 1; }
                $sudo dpkg -i "$tmp" || { rm -f "$tmp"; return 1; }
                rm -f "$tmp"
            fi
            pkg_install cuda-toolkit ;;
        pacman) pkg_install cuda ;;
        dnf)
            local repo; repo="$(nvidia_repo)" || return 1
            local file="$repo/cuda-${os_id}${os_ver//./}.repo"
            echo "Adding NVIDIA's CUDA repository ($file) ..."
            $sudo dnf config-manager addrepo --from-repofile="$file" 2>/dev/null \
                || $sudo dnf config-manager --add-repo "$file" || return 1
            pkg_install cuda-toolkit ;;
        *) return 1 ;;
    esac
    hash -r
    pick_nvcc && [ "$nvcc_old" = 0 ]
}

if [ "$mode" != "none" ]; then
    if [ -z "${AIRUN_NVCC_CCBIN:-}" ] && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
        hint="    sudo apt install g++   (Fedora: sudo dnf install gcc-c++, Arch: sudo pacman -S gcc)"
        [ -n "$pm" ] && hint="    $sudo ${pm/apt-get/apt} install $(gxx_pkg)"
        need "a C++ compiler (g++), which nvcc uses for the kernels' host code" "$hint" \
            "$([ -n "$pm" ] && echo install_gxx)"
    fi

    if ! pick_nvcc; then
        need "CUDA Toolkit $min_cuda+ (nvcc)" "$(cuda_guide)" "$([ -n "$pm" ] && echo install_cuda)"
    elif [ "$nvcc_old" = 1 ]; then
        need "CUDA Toolkit $min_cuda+ (found only $(nvcc_version "$nvcc") at $nvcc; RTX 50 / sm_120 needs 12.8+.
  Or build for older GPUs only with --arch, e.g. --arch 86)" "$(cuda_guide)" \
            "$([ -n "$pm" ] && echo install_cuda)"
    fi
    # build.rs looks in the same places; pointing it at the one checked here
    # keeps the two from disagreeing.
    export CUDA_PATH="$(dirname "$(dirname "$nvcc")")"
    echo "CUDA: $(nvcc_version "$nvcc") at $nvcc"
fi

unset AIRUN_CUDA_ARCHS AIRUN_SKIP_CUDA
case "$mode" in
    none) export AIRUN_SKIP_CUDA=1; echo "No CUDA (CPU only)" ;;
    all)  echo "Archs: universal (repo default: 75 80 86 89 120)" ;;
    *)
        export AIRUN_CUDA_ARCHS="$arch"
        echo "Archs: $arch"
        case ",$arch," in *,120,*) ;; *) echo "  (no 120: the RTX 50 native NVFP4 kernels are left out)" ;; esac
        ;;
esac

# --- build ------------------------------------------------------------------
if [ "$debug" = 1 ]; then
    profile=debug; cargo_args=(build)
else
    profile=release; cargo_args=(build --release)
fi
echo
echo "cargo ${cargo_args[*]} ..."
[ "$mode" = "none" ] || echo "(the airun-cuda step compiles the CUDA kernels and takes a few minutes without printing anything)"
if ! cargo "${cargo_args[@]}"; then
    die "Build failed. See the error above; 'cargo build -vv' shows the nvcc command."
fi

bin="$PWD/target/$profile/airun"
echo
echo "OK: $bin"

# The binary links libcudart.so dynamically. NVIDIA's .deb/.rpm packages
# register its folder with ldconfig; the .run installer does not.
if [ "$mode" != "none" ] && command -v ldd >/dev/null 2>&1; then
    missing="$(ldd "$bin" 2>/dev/null | awk '/not found/ {print $1}' | paste -sd' ' -)"
    if [ -n "$missing" ]; then
        libdir="$CUDA_PATH/lib64"
        echo >&2
        echo "warning: the binary cannot find: $missing" >&2
        echo "  Add the CUDA libraries to the loader path, once:" >&2
        echo "    echo $libdir | sudo tee /etc/ld.so.conf.d/cuda.conf && sudo ldconfig" >&2
        echo "  or per shell: export LD_LIBRARY_PATH=$libdir:\${LD_LIBRARY_PATH:-}" >&2
        export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
    fi
fi

# --- verify: selftest on each GPU (each runs its own arch's code) -----------
[ "$check" = 1 ] && [ "$mode" != "none" ] || exit 0
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "Built, but not verified: no NVIDIA driver. Install it ($driver_hint)"
    echo "and run: $bin selftest"
    exit 0
fi
"$bin" gpu || die "airun gpu failed. If the driver was just installed, reboot and run: $bin gpu"
[ "$gpus" -gt 0 ] || gpus=1
for ((d = 0; d < gpus; d++)); do
    echo
    echo "selftest on GPU $d (kernels vs CPU reference) ..."
    "$bin" selftest --device "$d" || die "selftest failed on GPU $d."
done
echo
echo "Build verified and ready to use."
