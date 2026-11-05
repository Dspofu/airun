#!/usr/bin/env bash
# build.sh — compila o airun para as GPUs desta máquina (Linux).
# No Windows, use .\build.ps1 (mesmas opções).
#
# Uso:
#   ./build.sh                 # detecta TODAS as GPUs e compila para as arquiteturas delas
#   ./build.sh --arch 120      # força uma arquitetura (ou "86,120" para várias)
#   ./build.sh --all           # build universal (RTX 20 até 50, padrão do repo)
#   ./build.sh --no-gpu        # sem CUDA (CPU only, para quem não tem GPU)
#   ./build.sh --skip-check    # não roda gpu/selftest depois do build
#   ./build.sh --debug         # build de debug (compila mais rápido, roda mais devagar)
#
# Requer: Rust (rustup), CUDA Toolkit 12.8+ (nvcc no PATH ou CUDA_PATH /
# /usr/local/cuda) e um compilador C++ (g++).
#
# GPUs de gerações diferentes no mesmo PC (ex.: RTX 3060 + RTX 5060 Ti) são
# detectadas juntas: o binário leva o código de cada uma ("86,120") e cada
# placa roda o seu. Para distribuir, use --all.

set -euo pipefail
cd "$(dirname "$0")"

arch="" mode="auto" check=1 debug=0
while [ $# -gt 0 ]; do
    case "$1" in
        -a|--arch) arch="${2:?--arch precisa de um valor, ex.: --arch 86,120}"; mode="manual"; shift ;;
        --arch=*) arch="${1#*=}"; mode="manual" ;;
        --all) mode="all" ;;
        --no-gpu) mode="none" ;;
        --skip-check) check=0 ;;
        --debug) debug=1 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "opção desconhecida: $1 (veja ./build.sh --help)" >&2; exit 2 ;;
    esac
    shift
done

gpus=0
if [ "$mode" = "auto" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "aviso: nvidia-smi não encontrado; usando build universal (--arch escolhe as arquiteturas)." >&2
        mode="all"
    elif ! rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)" || [ -z "$rows" ]; then
        echo "aviso: nvidia-smi falhou; usando build universal (--arch escolhe as arquiteturas)." >&2
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
    none) export AIRUN_SKIP_CUDA=1; echo "Sem CUDA (CPU only)" ;;
    all)  echo "Archs: universal (padrão do repo: 75 80 86 89 120)" ;;
    *)
        arch="$(echo "$arch" | tr ' ' ',')"
        export AIRUN_CUDA_ARCHS="$arch"
        echo "Archs: $arch"
        case ",$arch," in *,120,*) ;; *) echo "  (sem 120: os kernels NVFP4 nativos das RTX 50 ficam de fora)" ;; esac
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
    echo "Build falhou. Veja o erro acima; 'cargo build -vv' mostra o comando do nvcc." >&2
    exit 1
fi

bin="$PWD/target/$profile/airun"
echo
echo "OK: $bin"

[ "$check" = 1 ] && [ "$mode" != "none" ] || exit 0
"$bin" gpu || { echo "airun gpu falhou." >&2; exit 1; }
[ "$gpus" -gt 0 ] || gpus=1
for ((d = 0; d < gpus; d++)); do
    echo
    echo "selftest na GPU $d (kernels vs referência CPU) ..."
    "$bin" selftest --device "$d" || { echo "selftest falhou na GPU $d." >&2; exit 1; }
done
echo
echo "Build verificado e pronto para usar."
