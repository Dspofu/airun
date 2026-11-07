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

- **Rust** 1.85+ (via rustup). Se ele estiver fora do PATH, defina
  `CARGO_HOME`/`RUSTUP_HOME` para onde está — ou deixe o `build.ps1`/`build.sh`
  descobrirem sozinhos.
- **CUDA Toolkit** 12.x ou 13.x (sm_120 precisa de 12.8+). Com mais de um
  CUDA instalado, aponte `CUDA_PATH` para o certo.
- Durante o desenvolvimento, compile só para a sua GPU:
  `$env:AIRUN_CUDA_ARCHS="120"` (ou `"86,120"`, etc.). Remova na build final.

GPUs de gerações diferentes no mesmo PC funcionam juntas; sem P2P entre as
placas, cópias entre GPUs passam pela RAM.

Modelos de teste: qualquer GGUF da família Qwen que você tenha local (um
pequeno, como o Qwen3-1.7B, serve para os testes rápidos abaixo).

## Verificação

```powershell
cargo test --release
cargo build --release
.\target\release\airun.exe selftest
.\target\release\airun.exe run modelo-pequeno.gguf -p "2+2?" -n 20 --temp 0
```

## Git

Repositório git com `main`. Não rode `git commit` nem `git push` a menos que o
usuário peça.
