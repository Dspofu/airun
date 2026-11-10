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
#
# Requires: Rust (rustup), CUDA Toolkit 12.8+ (nvcc on PATH or CUDA_PATH /
# /usr/local/cuda) and a C++ compiler (g++).
#
# GPUs of different generations in the same PC (e.g. RTX 3060 + RTX 5060 Ti) are
# detected together: the binary carries each one's code ("86,120") and each
# card runs its own. To distribute, use --all.

set -euo pipefail
cd "$(dirname "$0")"

# A shell that did not source ~/.cargo/env (e.g. non-interactive) does not
# have cargo on PATH, even with rustup installed.
if ! command -v cargo >/dev/null 2>&1; then
    for d in "${CARGO_HOME:-}" "$HOME/.cargo"; do
        if [ -n "$d" ] && [ -x "$d/bin/cargo" ]; then
            export PATH="$d/bin:$PATH"
            echo "Rust: $d (not on PATH)"
            break
        fi
    done
    if ! command -v cargo >/dev/null 2>&1; then
        echo "cargo not found on PATH, in \$CARGO_HOME or in ~/.cargo. Install Rust" \
             "(https://rustup.rs) or set CARGO_HOME to the folder where it lives." >&2
        exit 1
    fi
fi

arch="" mode="auto" check=1 debug=0
while [ $# -gt 0 ]; do
    case "$1" in
        -a|--arch) arch="${2:?--arch needs a value, e.g. --arch 86,120}"; mode="manual"; shift ;;
        --arch=*) arch="${1#*=}"; mode="manual" ;;
        --all) mode="all" ;;
        --no-gpu) mode="none" ;;
        --skip-check) check=0 ;;
        --debug) debug=1 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see ./build.sh --help)" >&2; exit 2 ;;
    esac
    shift
done

gpus=0
if [ "$mode" = "auto" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "warning: nvidia-smi not found; using the universal build (--arch picks the architectures)." >&2
        mode="all"
    elif ! rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)" || [ -z "$rows" ]; then
        echo "warning: nvidia-smi failed; using the universal build (--arch picks the architectures)." >&2
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

unset AIRUN_CUDA_ARCHS AIRUN_SKIP_CUDA
case "$mode" in
    none) export AIRUN_SKIP_CUDA=1; echo "No CUDA (CPU only)" ;;
    all)  echo "Archs: universal (repo default: 75 80 86 89 120)" ;;
    *)
        arch="$(echo "$arch" | tr ' ' ',')"
        export AIRUN_CUDA_ARCHS="$arch"
        echo "Archs: $arch"
        case ",$arch," in *,120,*) ;; *) echo "  (no 120: the RTX 50 native NVFP4 kernels are left out)" ;; esac
        ;;
esac

if [ "$debug" = 1 ]; then
    profile=debug; cargo_args=(build)
else
    profile=release; cargo_args=(build --release)
fi
echo
echo "cargo ${cargo_args[*]} ..."
if ! cargo "${cargo_args[@]}"; then
    echo "Build failed. See the error above; 'cargo build -vv' shows the nvcc command." >&2
    exit 1
fi

bin="$PWD/target/$profile/airun"
echo
echo "OK: $bin"

[ "$check" = 1 ] && [ "$mode" != "none" ] || exit 0
"$bin" gpu || { echo "airun gpu failed." >&2; exit 1; }
[ "$gpus" -gt 0 ] || gpus=1
for ((d = 0; d < gpus; d++)); do
    echo
    echo "selftest on GPU $d (kernels vs CPU reference) ..."
    "$bin" selftest --device "$d" || { echo "selftest failed on GPU $d." >&2; exit 1; }
done
echo
echo "Build verified and ready to use."
