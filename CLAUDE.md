# airun

Engine de inferência de LLM escrita do zero em Rust + CUDA.

**Leia [AGENTS.md](AGENTS.md) antes de alterar qualquer coisa.** Ele tem o
estado atual verificado, o mapa do código, as armadilhas conhecidas e o
procedimento de verificação. Este arquivo é só o resumo.

## Regras invioláveis

1. **Não adicione dependências de inferência externas.** Nada de llama.cpp,
   ggml, PyTorch, ONNX, cuBLAS no caminho de execução. Foi decisão explícita do
   usuário. Utilitários (clap, serde, tokio, axum, memmap2, half) são ok.

2. **Os decodificadores de quantização existem duas vezes**, em
   `crates/airun-gguf/src/dequant.rs` (CPU, referência) e
   `crates/airun-cuda/kernels/dequant.cuh` (GPU). Mudou um, mude o outro e rode
   `airun selftest`. Um layout errado não trava — produz texto quase certo.

3. **Toda mensagem de erro diz ao usuário o que fazer.** Veja
   `crates/airun-core/src/error.rs` como padrão.

## Ambiente

Rust fica em `E:\rust` (fora do padrão):

```powershell
$env:RUSTUP_HOME="E:\rust\rustup"; $env:CARGO_HOME="E:\rust\cargo"
$env:PATH="E:\rust\cargo\bin;$env:PATH"
$env:AIRUN_CUDA_ARCHS="120"   # build rápido em dev; remova para a build final
```

Hardware: RTX 5070 Ti + RTX 5060 Ti (ambas sm_120), CUDA 13.2, MSVC 2022.
**Sem P2P entre as placas** — cópias entre GPUs passam pela RAM.

Modelos de teste: `G:\modelsAI\gguf\` (GGUF) e `G:\modelsAI\origin\` (safetensors).

## Verificação

```powershell
cargo test --release
cargo build --release
.\target\release\airun.exe selftest
.\target\release\airun.exe run G:\modelsAI\gguf\qwen3\Qwen3-1.7B-UD-Q8_K_XL.gguf -p "2+2?" -n 20 --temp 0
```

## Git

Não há repositório ainda. Não rode `git commit` a menos que o usuário peça.
