# airun

Inference engine for **Qwen** models (Qwen3, Qwen3.5/3.8, Qwen3-Next,
Qwen4), written from scratch in Rust + CUDA.

**Read [AGENTS.md](AGENTS.md) before changing anything.** It has the verified
current state, the code map, the known pitfalls and the verification
procedure. This file is only the summary.

## Inviolable rules

1. **Do not add external inference dependencies.** No llama.cpp, ggml,
   PyTorch, ONNX, cuBLAS on the execution path. That was an explicit decision
   of the user. Utilities (clap, serde, tokio, axum, memmap2, half) are fine.

2. **The quantization decoders exist twice**, in
   `crates/airun-gguf/src/dequant.rs` (CPU, reference) and
   `crates/airun-cuda/kernels/dequant.cuh` (GPU). Changed one, change the
   other and run `airun selftest`. A wrong layout does not crash — it produces
   almost-right text.

3. **Every error message tells the user what to do.** See
   `crates/airun-core/src/error.rs` for the pattern.

4. **Keep the docs clean** ([AGENTS.md §13](AGENTS.md#13-keeping-the-docs-clean)).
   The `.md` files describe the current state: update sections in place, no
   dated session logs, no per-run numbers or machine descriptions, failed
   experiments as one line in AGENTS.md §12, and no references to files
   outside the repository.

5. **Tasks and progress go in [Tasks.md](Tasks.md).** Create or update its
   entry when you start, finish or find a problem: `In progress`, `Fixed` or
   `Warn`, in the format at the top of the file. Use `pending` as the hash
   until the commit exists, then replace it. Never delete entries: the
   maintainer prunes the file.

6. **Other tracking or monitoring files are never committed.** Notes, logs,
   benchmark output, profiles and scratch scripts go in `benchmarks/`,
   `local/` or a `*.local.*` name, all git-ignored. A new location goes into
   `.gitignore` in the same change.

## Environment

- **Rust** 1.85+ (via rustup). If it is not on PATH, set
  `CARGO_HOME`/`RUSTUP_HOME` to where it lives — or let `build.ps1`/`build.sh`
  find it.
- **CUDA Toolkit** 12.x or 13.x (sm_120 needs 12.8+). With more than one CUDA
  installed, point `CUDA_PATH` at the right one.
- During development, build only for your GPU:
  `$env:AIRUN_CUDA_ARCHS="120"` (or `"86,120"`, etc.). Remove it for the final
  build.

GPUs of different generations in the same PC work together; without P2P
between the cards, copies between GPUs go through RAM.

Test models: any Qwen-family GGUF you have locally (a small one, like
Qwen3-1.7B, is enough for the quick tests below).

## Verification

```bash
cargo test --release
cargo build --release
./target/release/airun selftest
./target/release/airun run small-model.gguf -p "2+2?" -n 20 --temp 0
```

On Windows the binary is `.\target\release\airun.exe`.

## Git

Git repository with `main`. Do not run `git commit` or `git push` unless the
user asks.
