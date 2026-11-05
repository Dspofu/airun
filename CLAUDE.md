# airun

Engine de inferência para modelos **Qwen** (Qwen3, Qwen3.5/3.8, Qwen3-Next,
Qwen4), escrita do zero em Rust + CUDA.

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

Hardware: RTX 5060 Ti + RTX 3060 (sm_120 + sm_86; às vezes 2× 5060 Ti ou
5070 Ti), CUDA 13.x, MSVC 2022. Compile com `AIRUN_CUDA_ARCHS="86,120"` quando
a 3060 estiver no PC.
**Sem P2P entre as placas** — cópias entre GPUs passam pela RAM.

Modelos de teste: `D:\modelsAI\gguf\` (Qwen3.8-27B NVFP4, 35B-A3B, Qwen4 Flash-Next,
mmproj) e `E:\modelos IA\` (Qwen3-1.7B, Qwen3.5-9B e outros menores).

## Verificação

```powershell
cargo test --release
cargo build --release
.\target\release\airun.exe selftest
.\target\release\airun.exe run "E:\modelos IA\qwen3\Qwen3-1.7B-UD-Q8_K_XL.gguf" -p "2+2?" -n 20 --temp 0
```

## Git

Repositório git com `main`. Não rode `git commit` nem `git push` a menos que o
usuário peça.
