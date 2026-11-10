# Contributing to airun

Thanks for your interest. airun is an inference engine written from scratch in
Rust + CUDA, and some rules exist because a mistake here rarely crashes: it
produces *almost* right text. This guide says what is accepted, how to set up
the environment, and how to prove a change is correct and faster.

The scope is the **Qwen family** (architectures `qwen3`, `qwen35`, `qwen35moe`,
`qwen3next` and `qwen4exp`; the full list is in the [README](README.md#which-models)).
Support for another architecture is welcome, as long as it comes with
perplexity measured against llama.cpp.

Before touching code, read **[AGENTS.md](AGENTS.md)**. It has the verified
state of the project, the code map and the pitfalls that have already cost
days.

---

## The rules that do not change

1. **No external inference dependencies.** No llama.cpp, ggml, PyTorch, ONNX,
   cuBLAS, cuDNN or CUTLASS on the execution path. Every kernel lives in
   `crates/airun-cuda/kernels/` and belongs to this project. Utilities (clap,
   serde, tokio, axum, memmap2, half, minijinja) are welcome. Reading another
   project's code to understand a format is great; copying code is not. If a
   piece of data or a design comes from another project, cite it in the
   comment and in [NOTICE](NOTICE).

2. **Every quantization format exists twice:** in readable Rust, in
   `crates/airun-gguf/src/dequant.rs`, and in CUDA, in
   `crates/airun-cuda/kernels/dequant.cuh`. Changed one, change the other and
   run `airun selftest`. The selftest is the only defense against a swapped
   layout.

3. **Every error message tells the user what to do.** "failed" is not enough;
   "does not fit: raise --ctx-size or use --kv-type q8_0" is. See
   `crates/airun-core/src/error.rs`.

4. **Performance is proven with numbers, and so is correctness.** An
   optimization comes with the before and after measurement; a numeric change
   comes with the before and after perplexity. "It feels faster" and "the text
   looks fine" are not evidence.

---

## Environment

| | |
|---|---|
| GPU | NVIDIA with compute capability 7.5+ (RTX 20 or newer) |
| CUDA Toolkit | 12.x or 13.x (sm_120 needs 12.8+) |
| Rust | 1.85+ (`rust-toolchain.toml` pins the version) |
| Host compiler | MSVC 2022 on Windows, gcc/clang on Linux |

```powershell
# During development, build only for your GPU (minutes saved):
.\build.ps1                          # detects the GPUs and builds (and runs the selftest on each)
./build.sh                           # the same on Linux
# or by hand:
$env:AIRUN_CUDA_ARCHS = "120"     # 75, 80, 86, 89 or 120
cargo build --release
```

`build.rs` finds CUDA and the host compiler on its own. If you have more than
one CUDA installed, point `CUDA_PATH` at the right one. Do not leave
`AIRUN_CUDA_ARCHS` set in anything that goes into a release: the default
builds for RTX 20 through 50.

---

## How to verify a change

Always, in this order:

```powershell
cargo test --release                    # unit tests (no GPU)
cargo build --release
.\target\release\airun.exe selftest     # every kernel against the CPU reference
.\target\release\airun.exe run model.gguf -p "2+2?" -n 20 --temp 0
.\target\release\airun.exe perplexity model.gguf    # loss and top-1
```

- **The selftest passes and generation turned to garbage?** The problem is in
  the runtime (order of operations, KV cache, RoPE), not in the kernels.
- **Do not judge correctness by reading the text.** A model with a swapped
  layout writes fluent English about something else. Use `airun perplexity`
  and compare with the earlier measurement, on the same text and chunk size.
- **Touched `airun serve`?** Run the conformance test against a llama-server
  with the same model:

  ```powershell
  python tests\api_conformance.py --ref http://127.0.0.1:8090 --test http://127.0.0.1:8080
  ```

  It compares the responses key by key and the SSE event names. airun's API
  follows llama-server's; a missing field is a broken client.

---

## Benchmarks

A measurement that goes into the repository lives in `benchmarks/<date>/` with
three things: the **script** that produced it, the **raw output** (`.csv`,
`.log`) and a **RESULTS.md** with the table and what changed since the
previous one. A reader should be able to repeat the measurement with one
command.

- **Same conditions on both sides:** same model, same prompt, same prompt
  batch size, same context depth.
- **Engines alternated per round,** so background load and GPU temperature hit
  all of them equally. Two rounds at least.
- **Decode is not a single rate:** it drops with context. Say at what depth
  you measured (`airun bench --decode-at 16,32768`).
- **State the machine:** GPUs, PCIe link, RAM, driver, OS.

---

## Code

- **Comments explain why**, not what: the measurement that justified a choice,
  the bug a line avoids, the format a layout follows. Follow the density and
  tone of the file you are in.
- **Code, comments and project documentation** (`AGENTS.md`, `docs/`,
  benchmark results) **in English.**
- `cargo fmt` is not required on existing code, but do not reformat whole
  files in a PR that changes something else: the diff must show only the
  change.
- A fixed bug gets a test that failed before the fix, when it can be tested
  without a GPU; when it cannot, a case in `selftest`.
- New flags and environment variables go into [docs/FLAGS.md](docs/FLAGS.md).

---

## Pull requests

1. Open an issue before a large change (new architecture, new kernel, format
   change), to agree on the approach.
2. One PR, one subject. An optimization and a refactor go in separate PRs.
3. Describe in the PR:
   - what changed and why;
   - how you verified it (tests, selftest, perplexity before/after);
   - for performance, the before/after table and the machine.
4. Commit messages: a one-line summary and, in the body, what changed and the
   numbers that matter.

By submitting a contribution, you agree that it is distributed under the
project's [Apache 2.0 license](LICENSE).

---

## Reporting bugs

Include:

- the full command and its output (`--stats` helps);
- `airun gpu` (GPUs, VRAM, driver);
- the model (file name and where it came from) and `airun inspect model.gguf`
  if it is an unusual format;
- for a correctness problem: the prompt, the expected and the actual output,
  preferably with `--temp 0`.

Security problems in `airun serve` (authentication, file reads) do not go in a
public issue: use GitHub's private reporting (**Security → Report a
vulnerability** tab).
