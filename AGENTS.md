# Guide for AI agents working on airun

Read this file **before** changing anything. It describes what the project
is, what already works, where the pitfalls are, and how to verify you did
not break anything.

---

## 1. What airun is

An LLM inference engine **written from scratch**, in Rust + CUDA.

**This is not a wrapper.** There is no llama.cpp, ggml, PyTorch, ONNX or
cuBLAS on the execution path. Every CUDA kernel in `crates/airun-cuda/kernels/`
was written for this project. If you catch yourself about to add an external
inference dependency, stop: that was an explicit decision of the user.

Allowed dependencies: utilities (clap, serde, memmap2, half, thiserror,
anyhow, tokio, axum). The CUDA Runtime is used only to manage memory, streams
and launch *our* kernels.

---

## 2. Current state (verified on this device)

airun is built for the **Qwen family**. Architectures (`general.architecture`):

| Architecture | Models | State |
|---|---|---|
| `qwen35` | Dense hybrid Qwen3.5 / Qwen3.8 (e.g. 27B) | main focus: MTP, vision, `--parallel`, NVFP4 |
| `qwen35moe` | Qwen3.5 / Qwen3.8 MoE (e.g. 35B-A3B) | complete (MTP off by default) |
| `qwen4exp` | Qwen4 Flash-Next (hyper-connections, PLE, QSA) | complete, no `--parallel` |
| `qwen3next`, `qwen3nextmoe` | Qwen3-Next | same hybrid block |
| `qwen3` | Dense Qwen3 | complete |
| `qwen2`, `llama` | same block layout | works, lightly tested |
| `qwen3moe` | Qwen3-30B-A3B / 235B | **no** (the loader refuses with a message) |

| Component | State |
|---|---|
| GGUF (v1–v3) and safetensors reading | works |
| Quantization decoders | bit-exact against the CPU reference |
| Quantizers (writing) | round-trip tested |
| CUDA kernels | 289 cases in the selftest (fewer on Ampere, without the FP4 ones) |
| Multi-GPU, including different generations (build `86,120`) | validated (2× 5060 Ti; 5060 Ti + 3060) |
| VRAM planner | tested, with fallback and automatic context |
| llama-server-format API | 56/56 in `tests/api_conformance.py` |
| Sampling and constrained generation (GBNF, JSON schema) | tested |
| MTP (speculative), also with temperature | 27B: 26 → 68 tok/s greedy, 25.7 → 47.2 at temp 0.7 |
| Vision (`mmproj` qwen3vl_merger) | reads text in images like llama.cpp |
| `serve --parallel` | 4 requests on the 27B: ~85 tok/s summed (one alone with MTP: 38); a lone request same as without the flag (MTP included) |
| Stream replay, reasoning levels, load progress bar | tested |

**Not yet:** `qwen3moe`, audio, LoRA, rerank, video, Qwen3-VL with deepstack.

---

## 3. Code map

```
crates/
  airun-gguf/          file format — no CUDA, no network I/O
    ggml_type.rs       type table; unknown IDs are NOT an error
    reader.rs          GGUF parser (mmap, everything bounds-checked)
    dequant.rs         CPU decoders — the ORACLE for the GPU kernels
    quantize.rs        encoders (f32 -> blocks)
    writer.rs          streaming GGUF writer
    safetensors.rs     HF checkpoint reading

  airun-cuda/          GPU
    build.rs           finds CUDA + MSVC, compiles kernels/*.cu -> .lib
    kernels/*.cu       THE KERNELS. Hand-written.
    kernels/dequant.cuh  GPU decoders — mirror of dequant.rs
    src/ffi.rs         FFI (fixed-width types only)
    src/ops.rs         safe wrappers

  airun-core/          runtime
    config.rs          hyperparameters from the GGUF metadata
    plan.rs            VRAM planner / multi-GPU split
    tokenizer.rs       byte-level BPE (GPT-2/Qwen)
    model.rs           weight upload
    runtime.rs         forward pass
    kv_loan.rs         KV cache rows lent to MoE experts
    spec.rs            MTP speculative decoding
    cpu.rs             CPU matmuls for layers kept in RAM
    sample.rs          sampling
    loader.rs          orchestration + fallback ladder

  airun-cli/           commands
```

---

## 4. The most important rule

> **`crates/airun-gguf/src/dequant.rs` and
> `crates/airun-cuda/kernels/dequant.cuh` implement the SAME thing twice, on
> purpose.**

The Rust file is the readable reference; the `.cuh` is the GPU version.
`airun selftest` compares the two numerically.

If you change one, you **must** change the other and run the selftest. A
subtly wrong quantization layout does not crash — it produces text that looks
almost right. That is much worse than a crash, and the selftest is the only
defense.

---

## 5. How to verify your work

Always, in this order:

```powershell
# If Rust or CUDA are not on PATH, set CARGO_HOME/RUSTUP_HOME/CUDA_PATH
# (build.ps1 and build.sh find them on their own; a manual cargo needs the variables).
$env:AIRUN_CUDA_ARCHS="120"     # local arch only: fast build during dev (drop it for the final)

cargo test --release            # 60+ unit tests
cargo build --release
.\target\release\airun.exe selftest        # every GPU kernel against the CPU reference
.\target\release\airun.exe run <small-model.gguf> -p "What is 2+2?" -n 20 --temp 0

# and the quality metric, which is what catches subtle errors (any model on a known text):
.\target\release\airun.exe perplexity <model.gguf>
```

If the selftest passes but generation turns to garbage, the bug is in the
*runtime* (order of operations, KV cache format, RoPE), not in the kernels.

**Decode is not a single rate.** It drops with context (1.7B: 234 t/s empty,
128 t/s at 3072). When comparing measurements, compare at the same depth.

**Do not judge correctness by reading the generated text.** A model with a
swapped layout writes fluent English answering something else, and a short
continuation gets lucky — this has already reversed a decision in this
project. Use `airun perplexity`: it scores a known passage and returns loss
and top-1, so any hypothesis becomes a number comparable against a dense
control model.

**Do not drop `AIRUN_CUDA_ARCHS` handling from the final build.** The default
compiles for sm_75/80/86/89/120 and takes several minutes, but it is what
supports RTX 20 through 50.

---

### The three matvec paths

Decode (n=1) runs on the exact f32 kernel — generation is bit-identical to the
CPU reference. Batches of 2–63 tokens use dp4a over quantized activations
(block_q8a in `vecdotq.cuh`, mirrored in `quantize_q8a_mirror` in the
selftest — changed one, change the other). Batches of 64+ go to **int8 tensor
cores** (MMQ, `mmq.cuh`) on sm_80+ for Q8_0/Q4_K/Q5_K/Q6_K — the weights
become the same integer codes as dp4a; Q8_0/Q4_K/Q5_K use the `cp.async`
version (`matvec_mmq_async_kernel`). Other types and older GPUs go to fp16
tensor cores (`matvec_wmma_kernel`). Each path has its selftest cases with a
backward error bound; `AIRUN_MATVEC_F32=1` forces the exact one everywhere,
`AIRUN_NO_MMQ=1`, `AIRUN_NO_MMQ_ASYNC=1` and `AIRUN_NO_WMMA=1` turn each off.

**NVFP4** (Qwen3.8-27B NVFP4) takes the integer path at every size, decode
included (`KernelType::prefers_q8_decode`): at 4.5 bits the f32 vector each
warp rereads weighs more than the weights. FP4 nibbles become int8 with
`__byte_perm` (`fp4_codes8` in `dequant.cuh`) — the old `__constant__` table,
indexed by a different nibble in each lane, serialized and left the 27B at
5 tok/s. Batches of 2–8 tokens (the MTP verify) use `matvec_q8_multi_kernel`,
which decodes each group once (`mmq_decode`) and spends it on every token.
NVFP4 has async MMQ and fused gate/up (`mmq_async<NVFP4>`, 36-byte blocks
copied like Q8_0's). NVFP4 decode uses `matvec_q8_wide_kernel` (128 threads
cover the K of 2 rows: with one warp per 4 rows, a 5120 row was only 5
iterations per lane and bandwidth stayed at 73%); the MTP verify uses the
multi-token version (`matvec_q8_wide_multi_kernel`).

**NVFP4 prefill on Blackwell (sm_120): FP4 tensor cores.**
`kernels/sm120a/fp4mma.cu`, compiled separately for `compute_120a` (`mma ...
kind::mxf4nvf4` only exists on the specific target; `build.rs` produces
`airun_fp4.lib`). Weights go in as they are: word t of an NVFP4 block already
is the A fragment register and the scale word `d[4]` is the scale register;
the activation is quantized to the same block layout, plus one f32 scale per
token. 4-bit activations, as llama.cpp does: in the 27B's batched perplexity
(128 chunks, LF text) int8 2.0727, FP4 2.0838, llama.cpp GPU 2.0862. Batches
of 32+ tokens; `AIRUN_NO_FP4=1` goes back to int8 MMQ. SwiGLU goes inside the
quantization of the `down` input (`quantize_glu_nvfp4`). The quantizer has two
bit-identical forms: the vectorized one (4 lanes per run of 16, coalesced
float4, row in registers up to K = 20480) for 16-byte-aligned rows and the
scalar one for the rest; the selftest compares the two. The vectorized one
took SwiGLU from 214 to 66 µs per layer (512 tokens, K 17408) and pp2048 from
3623 to 3690 t/s. Putting SwiGLU in the epilogue of the `up` GEMM (with
per-row amax via atomicMax) measured slower: 3450/3380 t/s. `0x7F` scales
(zero for ggml, NaN for the hardware) are zeroed at load.

Async MMQ uses **stream-K** when the tiles do not close whole waves of SMs
(`MmqStreamK` in `mmq.cuh`): full waves run whole tiles and the leftover tiles
are split into equal K slices across one wave of blocks. The block holding the
start of a tile ("head") waits for the flags of the following blocks
("tails") and adds the partials in block order — deterministic. Waits only
point at higher-index blocks and the host limits stream-K to one resident
wave, so it does not deadlock. Prefill gate+up+SwiGLU are a single launch
(`GLU=true`), stream-K included (the slots carry the raw gate and up sums;
SwiGLU runs in the head). Do not capture MMQ in a CUDA graph: the flags'
`epoch` would freeze (`mmq_streamk_plan` refuses on a capturing stream).

In decode, Q8_0 loads the bytes of all the warp's rows before using any of
them (`q8_0_load`/`q8_0_decode` in `dequant.cuh`). With a `break` or
interleaved decoding, the compiler held the next row's loads behind the
previous one's arithmetic — one row in flight per warp. The result stays
bit-identical.

Greedy decoding is a pipeline (`ChainGraph` in `runtime.rs`): the graph reads
token and position from the GPU, the argmax writes the next token there and
advances the position, and the host queues step i+1 before reading step i's
token. A speculative step that the end of generation discards is not
committed (`n_past` and `history` do not advance). That is why it does not
run on hybrids: the recurrent state cannot undo a step.

Attention also has paths by size: prefill of 16+ tokens uses flash attention
on fp16 tensor cores (`AIRUN_NO_FLASH=1` turns it off), and decode uses a
kernel per (KV head, split) that reads the cache once for the whole GQA group
(`AIRUN_NO_GQA_DECODE=1` goes back to the old one).
`crates/airun-core/tests/prefill.rs` (ignored;
`AIRUN_TEST_DENSE_GGUF`/`AIRUN_TEST_MOE_GGUF`) checks that a batched prefill
reaches the same logits as feeding token by token — it is the only test that
runs those kernels with real weights.

**Long-context prefill (2026-11-01).** Long-prompt prefill lost 28% to 37% to
llama.cpp with the 27B split across the two GPUs. Two causes, in this order of
weight:

1. **The GPUs ran in series.** The `handoff` between shards synchronized the
   source GPU through the host, and `airun bench` read the logits of every
   prompt chunk, which also synchronizes. A 5060 Ti (half the SMs) behind the
   5070 Ti dropped a 4k prompt from 3653 (one GPU) to 2501 tok/s; llama.cpp,
   which chains the GPUs, kept 3514.
   - Now `handoff` only orders by events: the destination stream waits on the
     source's `sent`, and the source waits on the destination's `copied`
     before overwriting `x`. GPU 0 runs chunk c+1 while GPU 1 runs chunk c.
   - The bench uses `Runtime::prefill_chunk` for the middle chunks, like
     `generate`.
   - Whoever fills a pinned host buffer must wait for the previous upload,
     because the host no longer stops between chunks. PLE has the `uploaded`
     event; MoE already synchronized. Split: 4k at 3566 tok/s.
2. **Prefill attention was at 55 TFLOPS** with head_dim 256
   (`crates/airun-cuda/tests/attention_bench.rs`, ignored, measures in
   isolation). With 255 registers only one 8-warp block fits per SM, and the
   two warps of each partition do the softmax together while the tensor core
   waits, so what weighs is the per-tile cost outside the MMAs. Three cuts
   took it to 72.7 TFLOPS:
   - **P·V in fp16 per key tile:** a sum of at most 32 terms ≤ 1 times V,
     added to an fp32 O. It is the double-rate MMA, without keeping a long
     sum in fp16; llama.cpp keeps the whole O in fp16. The kernel only uses
     it when the batch sees 4096+ keys (`keys` of `attention_masked`). On
     short prompts the fp16 sum flipped a near-tied Qwen4 token against the
     reference, and the gain is small. Perplexity with 8k chunks: 1.2960
     against 1.2999 in fp32. `AIRUN_ATTN_PV_F32=1` turns it off.
   - **Unmasked path:** a whole tile before all of the warp's queries needs
     neither a mask nor the `-inf` guards.
   - **Lazy rescale:** O is only rescaled when some row's maximum changes.

**Allocation: `--alloc auto|gpu|memory` and `--main-gpu` (2026-11-02).**

- **Planner** (`plan.rs`):
  - `AllocMode` decides whether weights may go to RAM.
  - `--main-gpu` becomes the first of `ordered`.
  - `fill_fastest_first(.., fewest)` uses only the GPUs needed in `auto`.
  - `spill_to_ram` fills the other GPUs with the last layers and leaves in
    RAM, with the KV in the main GPU's VRAM, what the main GPU cannot hold.
  - The result goes in the `Shard`: `ram_layers` and `staging_bytes`.
- **Loader** (`model.rs`):
  - `load_ram` keeps the matrix in pinned RAM, with the NVFP4 sanitize done
    on the CPU, and `GpuTensor::ptr()` points at the staging.
  - `LayerWeights::ram_copies` lists the layer's copies.
- **Runtime:** `queue_shard` queues the copies before the layer, on the same
  stream. In the decode graph they become copy nodes.
- **Validated:** 120 identical tokens with 22 layers in RAM against the model
  in VRAM, and Qwen4 identical to the reference.
- **Limits:**
  - Whole MoE layers in RAM do not exist yet; for MoE the planner does not
    spill and the experts use the old path.
  - The copy does not overlap the compute. The link is at its limit (~26
    GB/s), and overlapping would hide only the ~20% the compute takes.

**Long-context decode.** `attention_decode_gqa_kernel` reads K and V at ~80%
of bandwidth (711 GB/s on the 5070 Ti, 323 on the 5060 Ti, at 128k;
`decode_attention_bandwidth` in `attention_bench.rs`, with
`AIRUN_BENCH_DEVICE`). It splits the cache into up to 256 pieces, and 4 KV
heads × 256 make 1024 blocks. On the 5060 Ti (144 blocks at a time) that made
7.1 waves, the eighth almost empty. `attention_active_splits` now rounds to
whole waves, with the GPU's real concurrency: +2% decode at 128k.
`AIRUN_NO_SPLIT_WAVES=1` turns it off.
- **cp.async did not help:** I tried loading the tile with `cp.async` (every
  piece in flight) and it dropped to 636 GB/s. Per-tile latency was not the
  bottleneck.

Grouping the GQA heads would not reduce traffic: the intensity per K/V tile
is the same, because the kernel is compute-bound, not memory-bound.

**Three kernel pitfalls that cost hours on 2026-09-26:**
- A register array (`acc[h]`) indexed in a loop with a run-time bound goes
  entirely to local memory. Use `#pragma unroll` with the constant bound and
  `if (h >= G) continue`. Check `STACK:` in `cuobjdump --dump-resource-usage`
  — it must be 0.
- `(const half2*)&raw` over a register variable also sends it to local
  memory. Extract with bit operations.
- Without performance counters (they need admin on Windows), measure with
  `clock64()` inside the kernel and one block's `printf` — that is what found
  the three bottlenecks.

**When tuning kernel tiles: the sizes appear in the template AND in the grid
math.** Sweeping one without the other computes a fraction of the output and
looks like a huge gain — `airun bench` does not validate outputs. Change the
defines, never the instantiations, and run the selftest at every measured
point. This has already fooled a whole session.

## 6. Pitfalls that have already cost time

1. **CUDA ordinal ≠ nvidia-smi index.** CUDA orders by speed by default. On
   this PC, CUDA 0 = RTX 5070 Ti, but nvidia-smi 0 = RTX 5060 Ti. Always show
   the PCI bus id next to the ordinal.

2. **These GPUs have no P2P.** `cudaDeviceCanAccessPeer` returns 0. Copies
   between GPUs go through RAM. That is why the planner prefers a single GPU
   when the model fits and, when it does not, fills the fastest first
   (`fill_fastest_first`, in `plan.rs`) — unless the GPUs are twins (perf
   score within 10%): then it splits evenly (`split_by_fraction` filtered by
   `fits_budgets`), which doubles prefill and does not change decode.
   `plan_experts` also orders the slow GPUs last and passes that order to the
   planner (`PlacementRequest.device_order`), without assuming the highest
   ordinal is the slowest. Splitting by free VRAM put half of the 27B on the
   5060 Ti, which has half the 5070 Ti's bandwidth. Since decode runs the cards
   in series, that cost 28%: decode @16 at 33.7 against 43.0 tok/s and @32k at
   28.9 against 37.1. In decode the split costs: on the 1.7B, 94 t/s against
   234 t/s on a single card, with an empty context. In prefill the GPUs
   overlap (see "Long-context prefill" above) and the split barely loses.
   Never synchronize a stream through the host between shards: that puts the
   GPUs in series.
   **Updated (2026-11-05):** between unequal GPUs, without `--main-gpu` and
   with an untied output, the planner uses `fill_fastest_last`. The fastest
   gets the OUTPUT and the last layers, and the slow one the first layers and
   the embedding. The output is read whole every token (and again for every
   MTP draft); the embedding is only looked up by row. Qwen3.8-27B NVFP4 with
   a 2.5 GB BF16 lm_head, 5070 Ti + 5060, MTP 3, `-c 131072`: 44.1 → 52.2
   tok/s (`run`); `serve` 34–36 → 43–47. With the automatic context (262k)
   the new layout does not fit (the 5070 Ti would have output + KV) and the
   planner falls back to the old one.

3. **CUDA 13 removed fields from `cudaDeviceProp`** (`memoryClockRate`). Never
   bind that struct directly from Rust — use the shim in `kernels/device.cu`.

4. **`gen` is a reserved keyword in edition 2024.** Do not use it as a
   variable name.

5. **Tied embeddings.** Qwen3-1.7B has no `output.weight`; the final
   projection reuses `token_embd.weight`. In a multi-GPU split that forces
   mirroring the table — the planner already accounts for it.

6. **Relative error is the wrong metric for a dot product.** When terms
   cancel, the result can be arbitrarily small next to the values that
   produced it. The selftest uses *backward* error: a bound proportional to
   `sum|w*x|`. Do not "fix" this by loosening the tolerance.

7. **Test data must be physically plausible.** Random bytes in MXFP4's E8M0
   exponent produce scales of 2^±127 and overflow f32. The decode is right;
   the data was unrealistic.

8. **A GPU-vs-CPU test does not prove the operation is right** — it proves
   both sides agree. If both encode the same wrong assumption, the test passes
   and the model stays broken. It happened with `rmsnorm_gated`, where the
   "CPU reference" repeated the wrong gate order. When a kernel passes but the
   model is no good, suspect the reference before the kernel.

9. **A token is not a character.** In byte-level BPE an accented letter often
   takes two tokens — `"Órbita"` comes out as `<20 C3>` `<93>` `"r"`.
   Decoding token by token runs `from_utf8_lossy` on half a character, which
   returns `U+FFFD` and **destroys the bytes**: not even when the other half
   arrives can they be recovered. All streaming must go through
   `StreamDecoder`, which holds the incomplete tail. It applies to accents,
   CJK and emoji, on any model — it is not a broken-model problem.

10. **NVFP4 has an absolute magnitude floor.** The block carries no
    per-tensor scale, only four UE4M3 scales (one per 16 elements), and the
    smallest UE4M3 step is ~9.8e-4. A run of 16 weights whose peak is below
    ~5.9e-3 encodes as **zeros**, without any warning. That is ggml's format,
    not an encoder bug — but it means a round-trip test with small-magnitude
    data "fails" for the wrong reason (which is what happened in the first
    version of the test). Real weights stay well above the floor.

11. **The repeat penalty no longer accumulates per occurrence.** It is applied
    once per *distinct* token in the window, which is what modern llama.cpp
    does. The previous version divided the logit once per occurrence, so a
    token repeated 3 times was penalized `penalty^3` — the effect depended on
    how full the window was. If you compare outputs with an old session, the
    difference comes from here, not from a new bug.

12. **`llama-perplexity -f` on Windows reads the file in text mode** and turns
    CRLF into LF; `airun perplexity -f` reads the bytes as they are. On a CRLF
    file the two score different texts and llama.cpp looks ~0.1 nat better.
    Normalize to LF before comparing. With the same text, airun's `--chunk N`
    reproduces `llama-perplexity -c N` chunk by chunk (27B NVFP4, 128 chunks:
    airun 7.9356, llama.cpp CPU 7.9438). An f64 numpy reference of the qwen35
    forward (outside the repo) confirmed airun layer by layer.

13. **YaRN is not linear scaling.** `rope.scaling.type = yarn` (the "yarn1m"
    27B) only interpolates the slow dimensions and multiplies cos/sin by
    `1 + 0.1 ln(factor)`; treating the factor as linear scaling changes the
    model even at short context. `RopeParams` in `ops.rs`, mirrored in
    `airun.cuh`.

14. **NVFP4 exists in two conventions; the loader reads both (2026-11-03).**
    airun's encoder is *self-contained*: the block carries only the four
    UE4M3 scales (item 10) and has no sibling tensor. But outside NVFP4 files
    (vllm/NVIDIA — e.g. the "Cold Fusion" Qwen3.8-27B) use the *global-scale*
    convention: each NVFP4 weight has a `<base>.scale` F32[1] sidecar (≈1e-4)
    and the per-block dequant comes out ~10⁴ small without it. The loader
    reads the sidecar and applies it in `GpuTensor::global_scale` (a scale
    epilogue on every matvec; in the GLU fusions the scale turns the fusion
    off, because SiLU is not scale-linear). Without it the vllm 27B gave
    **NaN** (NaN perplexity, `!!!!` generation) and the selftest passed — the
    selftest uses synthetic weights in the kernel's own layout, never reads
    real bytes nor applies the sidecar. Diagnosis: dequantize the real bytes
    with and without the `.scale` and see which gives a sensible weight
    magnitude (`benchmarks/nvfp4_magnitude.py`).

## 7. SSM/MoE hybrids — fixed on 2026-09-26

**The output is correct now.** Two bugs masked each other:

1. **V-head → K-head mapping in the delta rule.** The GGUF repeats the whole
   K-head axis to reach the number of V-heads (`kh = vh % n_k_heads`, pattern
   `0,1,0,1…`), like llama.cpp's `ggml_repeat`. The kernel used adjacent
   pairs (`kh = vh / groups`, pattern `0,0,1,1…`), which is HuggingFace's
   `repeat_interleave` — but the converter reorders the V-heads precisely so
   that the tile works. `ssm.cu`, two kernels.
2. **Gate order in `rmsnorm_gated`.** It is norm → weight → gate (SiLU), like
   Qwen3-Next's `RMSNormGated` and llama.cpp's `build_norm_gated`. The project
   had chosen "gate first" because, **with bug 1 present**, that order scored
   better (4.40 against 4.89). With bug 1 fixed, the right order goes to 1.43
   and the old one stays at 5.12.

| Qwen3.8-35B-A3B Q4_K_M | Loss | Top-1 |
|---|---|---|
| before | 7.02 | 8.8% |
| fix 1 only | 5.12 | 14.5% |
| fixes 1 + 2 | **1.43** | **57.2%** |
| Qwen3-4B Q8_0 (dense, control) | 2.67 | 53.5% |

The `docs/FLAGS.md` sweep was **redone on the fixed base** and all 15
alternatives score worse (table in `benchmarks/2026-09-26/sweep2.txt`).

**Lesson for the whole project:** an A/B decision taken with another bug
present is not reliable. When fixing a layout, redo the earlier layout
decisions. The selftest does not catch this (pitfall 8): both sides encoded
the same assumption.

Earlier bugs, still valid (do not reintroduce):
1. The planner must allocate `cfg.n_run_layers()`; the last layer is MTP and
   is not loaded.
2. The `q/k/v` buffers need `group_count * state_size` in the SSM layers.
3. **`ssm_a` already comes as `-exp(A_log)`**; `load_delta_decay_coeff`
   normalizes by the sign.

### Expert offload (`--expert-cache N`)

Keeps N experts per layer in VRAM (LRU) and reads the missing ones from the
GGUF, without requantizing — the test `crates/airun-core/tests/offload.rs`
(ignored; run with `AIRUN_TEST_MOE_GGUF=<35B>`) requires **identical** logits
to the resident model.

Misses are read with parallel positional reads (`upload_experts` in
`runtime.rs`, one file handle per thread). Through mmap they arrived one page
fault at a time, at the SSD's queue-depth-1 speed, and that was almost all of
the decode time.

**Measure with natural text.** `airun bench` uses a repeated token on dense
models, but on a MoE that always routes to the same experts and the cache
looks perfect (99.8% hits, 54 t/s) when with real text it hits ~88% (3 t/s
before the parallel reader). The bench now uses an English passage for MoE and
shows the hit rate; also use `--temp 0.7`, because greedy falls into loops and
flatters the cache the same way.

### Experts mapped in RAM (default when the MoE does not fit the GPUs)

Without `--expert-cache`, a MoE too big goes to `plan_experts` (`loader.rs`):
`slots` experts per layer stay in VRAM and the rest in a pinned, mapped
allocation (`MappedExperts` in `model.rs`), which the GPU reads over PCIe. The
kernels find each expert through an address table (`moe_*_tab`), so nothing
waits on the host and decode fits in a CUDA graph. File streaming (section
above) remains with `AIRUN_EXPERTS=stream`.

- **Decode:** before the MoE, `moe_stage_plan` + `moe_stage_copy` copy into a
  VRAM staging area only the routed experts that are in RAM (coalesced 16 B
  loads). Reading straight from RAM with the decoders (narrow loads) was 5x
  slower.
- **Batches (≥ 16 tokens):** `moe_grouped` sorts the (token, slot) pairs by
  expert and reads each expert's weights once for all its tokens, copying the
  block's rows into shared memory with wide loads. It is bit-identical to the
  per-token path (selftest).
- **Adaptive residency:** the MoE counts each expert's use on the GPU
  (`moe_count`); every 8 steps `rebalance_experts` swaps the most used ones
  in RAM for the least used in VRAM (up to 16 per layer, with a margin).
- **Slow link:** `plan_experts` measures host→GPU bandwidth. A GPU below
  8 GB/s (this machine's 5060 Ti is on a PCIe 3.0 x4 slot: 1.6 GB/s
  zero-copy) only gets whole layers, with every expert in VRAM; the rest go
  to the fast-link GPU with partial experts.
- **Graphs per GPU:** decode captures one graph per shard; the residual moves
  between GPUs outside the graphs, only the batch's rows. PLE is prepared on
  the host in a pinned buffer before each replay.

Qwen4 Flash-Next (78 GB, 5070 Ti + 5060 Ti, 64 GB of RAM), 792-token prompt +
128 generated (`benchmarks/2026-11-01/qwen4.ps1`): see the numbers in
`benchmarks/2026-11-01/RESULTS.md`. What took airun from 92 / 19 tok/s to
~287 / ~39 (ahead of Strata 0.1.44, ~270 / ~33):

- **`__constant__` tables:** `kvalues_iq4nl` indexed by different nibbles in
  each lane serialized up to 16x. IQ4_NL/IQ4_XS now use `__byte_perm`
  (`iq4_codes8`), like FP4. That alone was worth ~5 ms per token.
- **Routing profile** (`expert_profile.rs`): the load puts in VRAM the
  experts most used in previous runs, and the rebalance scores start from the
  profile. A swap only happens with an advantage of ~4 routings, because each
  one moves both experts over PCIe.
- **Overlapped copy:** decode copies the missing experts on a side stream
  (`DeviceState.side`, `fork`/`join` events, which work inside the CUDA graph)
  while the resident ones compute (`Staged.only`).
- **One-warp router** (`moe_router_warp_kernel`): 25 → 4 µs per layer.
- **Grouped MoE on tensor cores** (`moe_mma_*`, batches of 64+): decodes each
  64-wide slice of K once to fp16 and uses `mma.sync`. Experts in VRAM are
  read directly, without copying the rows, which gives 3x higher occupancy;
  the ones in RAM are copied with wide loads. Below 64 tokens the f32 kernels
  stay, which keeps short prompts identical to the reference.
- **Parallel PLE** (`gather_pool.rs`): the 26.8 GB table does not fit the file
  cache, so each row is a random SSD read. Read one by one, they left the GPU
  idle 1.4 s on a 792-token prompt. 32 persistent threads read them together.

Loading takes ~80 s (pinning ~24 GB of RAM).

### QSA sparse attention (qwen4exp)

Attention layers with `compress_ratio` r > 0 only attend to `top_k + r - 1`
cells (2051 on Qwen4): the incomplete last block and the r-token blocks with
the highest score. Kernels in `kernels/qsa.cu`, wrappers in
`airun-cuda/src/qsa.rs`, orchestration in `Runtime::qsa_select`:

- `qsa_store`: each token's raw indexer key goes into an f16 ring
  (`QsaCache.ring`, like llama.cpp's indexer cache, which is f16).
- `qsa_pool`: when a block's last token arrives, mean of the r keys →
  RMSNorm (`indexer.k_norm`) → RoPE at the block's first position. Stored in
  `QsaCache.pooled` ([ctx/r][128] f32).
- `qsa_score`: score = Σ_heads relu(q_h · key), q with RMSNorm and RoPE at the
  token's position. Skipped when the query's context fits the budget.
- `qsa_select`: radix select; ties go to the lower-index block and a block
  taken in part keeps its first cells (the same as Strata's `qsa_select.cu`).
  Output: a `[count-1, cells...]` list in decode, which `attention_split_sel`
  uses instead of the position (fits the CUDA graph), or a bit mask per query
  in batches (`attention_masked`, which flash2 uses to skip tiles with no
  chosen cell).

With context ≤ 2051 the selection is everything, and the output is identical
to dense. The selftest compares pooling, scores, selection and both
attentions with the CPU. Perplexity in 4096 chunks (positions 2048–4095
scored) on a 25k-token text: reference 5.1563, airun with QSA 5.1684, airun
dense (`AIRUN_QSA=off`) 5.1717. At 2048 (all dense) the baseline difference
airun vs reference is already ±0.5% per chunk, from rounding.

### HTTP API (`airun serve`) in llama-server's format

Routes, request fields and responses follow `llama-server`'s, so that clients
built for it work unchanged:
- **Health and metadata:** `/health`, `/v1/health`, `/props`, `/models`,
  `/v1/models`, `/slots` and `/metrics` (501, like llama-server without
  `--metrics`).
- **Hardware (airun's own):** `/hardware` (behind the API key, like
  `/slots`). Per GPU, from NVML (`airun-cuda/src/nvml.rs`, opened at run
  time from the driver, so not a dependency): load, memory-bus load, VRAM
  (`nvmlDeviceGetMemoryInfo_v2`, which matches nvidia-smi), temperature,
  power and limit, clocks, fan, P-state, PCIe link and traffic. Without NVML
  only VRAM, from `cudaMemGetInfo`. RAM: system, this process
  (`disk::process_memory`) and pinned (`airun_cuda::pinned_bytes`, counted
  in `buffer.rs`). A reading costs ~250 ms on two GPUs (NVML samples PCIe for
  ~20 ms per counter), so it is cached for 0.7 s and shared between clients.
  The web UI polls it every 2.5 s for the sidebar block, every second while
  its Hardware panel is open, every 10 s in a background tab.
- **Generation:** `/completion` and `/completions` (native), `/infill`,
  `/v1/completions`, `/chat/completions` and `/v1/chat/completions`,
  `/v1/responses` (and `/responses`, OpenAI Responses) and `/v1/messages`
  (Anthropic Messages), all with streaming.
- **Counting:** `/v1/chat/completions/input_tokens`,
  `/v1/responses/input_tokens` and `/v1/messages/count_tokens` (with and
  without `/v1`).
- **Utilities:** `/tokenize` (with `with_pieces`), `/detokenize` and
  `/apply-template`.
- **Embeddings** (`--embeddings`): native `/embedding(s)` and
  `/v1/embeddings` (float or base64), from the final state (after the output
  norm) via `Runtime::hidden_states`; pooling `--pooling none|mean|cls|last`
  (default: the GGUF's `pooling_type`, else `last`, which in a generative
  model is the token that read the whole text), `embd_normalize` as in
  llama.cpp. Without the flag: 501.
- **Slots** (`--slot-save-path DIR`): `POST /slots/0?action=save|restore|
  erase` save/load the whole cache (`Runtime::save_state`/`load_state`: used
  KV rows, delta-net matrices, conv, QSA ring, PLE history). On a hybrid it is
  the only way to resume a long prompt without reprocessing; the continuation
  comes out identical token by token. The 27B's file is large (~283 MB at
  1925 tokens: the 48 delta-net matrices are f32). File name without a
  directory. Without the flag: 501.
- **Controlled reasoning:** `reasoning_budget_tokens` (or
  `thinking_budget_tokens`, or `--reasoning-budget`) and
  `reasoning_budget_message` force `<think>` closed after N tokens
  (`ReasoningBudget` in the sampler); with `reasoning_control: true`,
  `POST /v1/chat/completions/control {"id", "action": "reasoning_end"}`
  closes the thinking of a generation in progress (works for chat, Responses
  and Messages). Both turn off the GPU argmax and MTP for that request.
- **Unsupported, answering like llama-server without the feature:** rerank
  (`/rerank`, `/reranking`, `/v1/rerank(ing)`; needs a reranker model with a
  classification head), `POST /props`, `/v1/audio/transcriptions` (501; the
  models have no audio encoder); `/lora-adapters` (empty list / `success`).
  These handlers read the body before answering: without that the connection
  closed with the client still sending and it saw a reset instead of the 501.

Every response carries `timings`, `usage.prompt_tokens_details.cached_tokens`
and `system_fingerprint`; streaming honors `stream_options.include_usage`.
Errors come as `{"error": {"code", "message", "type"}}`.
- **Template:** the chat prompt is the GGUF's own Jinja template (`chat.rs`,
  minijinja with Python string methods), like llama.cpp's `--jinja`. Tools,
  tool responses, `enable_thinking` and `reasoning_effort` (in the body or in
  `chat_template_kwargs`) reach the template. Tool call arguments sent as a
  JSON string become objects first.
- **Response:** it is split into `reasoning_content` (the `<think>` block),
  `content` and `tool_calls` (`<tool_call>` in Qwen3.5+'s
  `<function=..><parameter=..>` XML format or Qwen2.5/3's JSON), with
  `finish_reason: tool_calls`. `reasoning_format: none` (or
  `--reasoning-format none`) leaves the thinking in `content`.
- **Responses and Messages** are converted into the chat form
  (`responses_to_chat`, `anthropic_to_chat`: `instructions`/`system`,
  `function_call`/`function_call_output` items, `tool_use`/`tool_result`/
  `thinking` blocks, `text.format`, `thinking.type`, `stop_sequences`) and
  take the same path (`chat_job`, `stream_chat`); only the output changes.
  Responses events carry `sequence_number`/`output_index` like OpenAI's;
  Messages blocks close one before the next opens, as at Anthropic.
- **Sampling:** llama.cpp's full chain (penalties → DRY → top-n-sigma → top-k
  → typical → top-p → min-p → XTC → temperature, with dynamic temperature;
  mirostat 1/2 replaces what comes after DRY). The `dry_sequence_breakers`
  become a token mask (the token whose text contains a breaker). `ignore_eos`
  is a -inf `logit_bias` on the end tokens (reported as `{"bias": null,
  "token"}`, like llama-server).
- **Grammar** (`airun-core/src/grammar.rs`): full GBNF (rules, groups,
  classes, `* + ? {m,n}`, escapes), a stack automaton over code points (a
  character split across tokens is carried), and a llama.cpp-style JSON
  schema converter (property order preserved; that is why `serde_json` uses
  `preserve_order`). `grammar` > `json_schema` > `response_format`. The mask
  tests the 256 most likely tokens and only falls back to the whole vocabulary
  if none fits. With the prompt opening `<think>`, the grammar waits for
  `</think>` (`Grammar::after`): the model thinks freely and the constraint
  applies to the answer. Cost: ~38 against ~49 tok/s on the 27B.
- **Probabilities:** `n_probs` (native), `logprobs: N` (/v1/completions) and
  `logprobs` + `top_logprobs` (chat) come from the softmax of the raw logits,
  before the samplers, as in llama.cpp, the end token included; also when
  streaming.
- **Infill:** `<|repo_name|>myproject`, each `input_extra`, `<|file_sep|>
  filename`, then `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>` —
  llama.cpp's format. `<|fim_pad|>`, `<|repo_name|>` and `<|file_sep|>` count
  as end of generation in the tokenizer, as in llama.cpp.
- **Verification:** `tests/api_conformance.py` records one server's responses
  and compares them key by key (and the SSE event names) with another's. On
  2026-11-01, with Qwen3.8-27B, it gave 56/56 against llama-server b11760:
  grammar, JSON schema, `ignore_eos`, `tool_choice: none`, `json_object`, tool
  calls and infill with the same text; `n_probs` within 0.002–0.013. On the
  short chat prompt without thinking "Hi" comes out with logprob -0.175
  against -0.42 in llama.cpp, with identical prompt and ids. airun's all-f32
  path (`-b 1`, `AIRUN_MATVEC_F32=1 AIRUN_NO_FP4=1 AIRUN_NO_MMQ=1`) gives
  -0.183: normal airun stays within 0.01 of the exact computation and
  llama.cpp 0.24 away (its activation quantization).
- **Default difference:** OpenAI requests without `repeat_penalty` use the
  server's `--repeat-penalty` (1.1); llama-server uses 1.0. The native
  `/completion` uses 1.0, like llama-server.
- **Not implemented yet:** rerank, audio and LoRA (none of the target models
  has a rerank head, an audio encoder or adapters).
- **Bug fixed alongside (2026-11-01):** with `-b 1` the prefill went token by
  token through the decode graph without synchronizing, and each replay read
  the token and position already overwritten in the pinned buffer ("The
  capital of France is" → "a 2013 American comedy").
  `forward_decode_graphed` now waits for the previous replay before rewriting
  the slots; decode does not change (48.4 tok/s).

### Quantized KV cache (`--kv-type f16|q8_0|q4_0`)

GGUF Q8_0/Q4_0 blocks per head row, decoded by the same `dequant`. The GPU
encoder mirrors `encode_q8_0`/`encode_q4_0` in `quantize.rs` **bit for bit**
(the selftest compares the bytes); the `__f*_rn` intrinsics exist because nvcc
would contract `x*id + 8.5` into an FMA that rounds differently.

## 8. Other fronts

1. **Decode that does not drop with context — done.** In decode, attention
   splits the KV range across blocks (`gridDim.z`, `attention_splits(ctx)`
   blocks) and combines the `(max, sum, acc)` partials in a second kernel,
   like llama.cpp's `flash_attn_ext`. Qwen3-4B Q8_0 on an RTX 3060: 38.9 →
   53.4 t/s at 3072 tokens (llama.cpp: 49.0). `AIRUN_NO_SPLIT_ATTN=1` goes
   back to the one-block-per-head kernel. The selftest covers up to position
   212991.

   Later, on 2026-09-26/27: int8 MMQ with `cp.async`, flash attention with
   `mma.sync`, stream-K, fusions (QKV, gate/up/SwiGLU, attention prologue,
   add+norm), GPU argmax and the decode pipeline. Interleaved scoreboard
   against llama.cpp, Qwen3-4B Q8_0, RTX 3060, best of 3 rounds
   (`benchmarks/2026-09-26/scoreboard.ps1`):

   | Test | airun | llama.cpp | ratio |
   |---|---|---|---|
   | prefill 128 | 3535 t/s | 3118 t/s | 1.13x |
   | prefill 512 | 3979 t/s | 3829 t/s | 1.04x |
   | decode @16 | 70.3 t/s | 69.1 t/s | 1.02x |
   | decode @3072 | 63.7 t/s | 62.9 t/s | 1.01x |

   Decode is at the memory limit: the matvecs run at 88–96% of real bandwidth
   (the 3060 sits in P2 during compute, memory at 7301 MHz instead of 7501).
   What is still left: short-context decode attention (~16 µs per layer
   against ~10 in llama.cpp, latency-bound with 8 warps on the whole GPU) and
   ~4 µs of ramp+tail per small matvec (o and QKV at 88%).

   **Tried and discarded (measured, do not repeat without a new reason):** L2
   prefetch on a second stream during the small kernels (1–2% slower: the
   graph's extra branches cost more than the L2 saves); process GPU priority
   via `D3DKMTSetProcessSchedulingPriorityClass` (no effect, not even with
   another CUDA load on the card); 1 or 2 rows per warp in the decode GLU (1
   drops to 85% of bandwidth, 2 ties with 4); double K/V buffer in decode
   attention (66 against 52 µs at 3072).

   **Measuring requires alternating.** On this machine another program
   drawing on the GPU (a browser, the chat app itself) stretches a kernel by
   ~300 µs every ~4 ms — ~8% of decode — and it goes away when the program
   stops. Both engines suffer equally; only interleaved runs in the same time
   window are comparable.
2. **Multi-modal.** `mmproj-f16.gguf` is a 27-block ViT (patch 16, embd 1152,
   ffn 4304) with a `qwen3vl_merger` projector, 2x2 spatial merge and
   deepstack layers. Structure already mapped via `inspect --tensors`. It
   depends on the previous item because the text model that comes with it is
   qwen3.6.

3. **Paged KV cache** for several concurrent sequences. Today the server
   serializes requests because the cache is a single sequence.

4. **MTP — done for dense qwen35 (2026-10-31).** `--draft N` (default 4 in
   `run` since 2026-11-05, 3 before; 0 in `serve` — the flag exists in both
   since 2026-11-03) loads `blk.{n_run}` as the draft head and decodes
   speculatively when sampling is pure greedy (`--temp 0 --repeat-penalty
   1`). The output follows normal decoding, but **is not guaranteed
   identical**: the verify runs batches of 2–4 tokens in other kernels
   (`matvec_q8_multi`, a different sum order), and on a near-tie of logits
   the argmax may flip — the text diverges from there on, equally plausible.
   `crates/airun-core/src/spec.rs`: the verify runs the delta-net layers
   without writing state (it keeps the inputs in `DeltaStash`) and `commit`
   pushes only the accepted tokens — `test_delta_defer` in the selftest
   requires bit-for-bit equality. The MTP KV row at position q is the pair
   (h_{q-1}, x_q), as in llama.cpp's draft-mtp. Between two verifies,
   `advance_graphed` runs commit + head over the verify's rows + the k chained
   drafts (each step's pick becomes the next one's token on the device) as one
   CUDA graph per accepted count m: one launch and one read, where before the
   host waited for each head step. With graphs on; on several GPUs the head
   and advance graphs are captured on the output GPU's stream and only
   `commit` (which touches all of them) stays outside (2026-11-05). The
   verify still has no graph on several GPUs. 27B NVFP4 on a 5070 Ti:
   93.8–113.6 tok/s against 49 without MTP (llama.cpp with draft-mtp, same
   prompts: 91.6–104.2).
   **MoE (2026-11-05):** the MoE head loads through the same expert path
   (`load_mtp_weights` builds `FfnWeights::Moe`). It works, but it is slower:
   35B-A3B on a 5060 Ti 115 → 91 tok/s with 64% accepted, because a 4-token
   verify wakes many more experts. That is why `--draft` without a value
   (`LoadOptions.draft = None`) is `DEFAULT_DRAFT` on dense and 0 on MoE.
   **Changed on 2026-11-06:** with the verify in a graph per GPU, a MoE fully
   in VRAM gains with MTP, and the default became `MOE_DEFAULT_DRAFT` (2) in
   that case; with experts in RAM it stays 0 (see "Decode measurements with
   MTP" below).
   The head is block `n_run_layers()` (`block_count` counts it); the old bug
   looked for `blk.{n_layers}` and never found it. 27B on a 5060 Ti: 26.4 →
   68.4. Missing: non-greedy sampling.

   **Non-silent activation (2026-11-03):** `--draft` only engages with pure
   greedy sampling (`temp 0` and no penalty — `run`/`serve` default to
   `--repeat-penalty 1.1`, which turns it off). When the user asks for
   `--draft` but the generation will not speculate, the runtime prints a
   `note:` explaining why (the same at load, when the file has no usable MTP
   head). Before, that fallback was silent: the user passed `--draft 3 --temp
   0`, got no speculation and did not know why.

5. **Jinja template.** The GGUF carries `tokenizer.chat_template`, which today
   is stored but not evaluated; we use built-in ChatML.

---

## 8. Style

- Comments explain **why**, not what. If a comment paraphrases the code,
  delete it.
- Every error message must tell the user **what to do**. Compare: "failed to
  load model" (bad) vs. the message in `error.rs` that says how much VRAM was
  missing and which flags to change (good).
- Every `unsafe` block needs a `// SAFETY:` comment justifying it.
- Tests check behavior, not implementation. Prefer tests that would fail if
  the logic were wrong, not just if it changed.
- Code, comments and documentation are in English.

## NVFP4 outside Blackwell (2026-11-05)

Without the FP4 tensor cores (sm_120a), NVFP4 runs on int8 MMQ (prefill) and
dp4a (decode), with the same result; `fp4_supported()` decides per GPU
(`cudaGetDevice`), so in a mixed PC each card uses its own path.

- RTX 3060, Qwen3-1.7B NVFP4 (converted by `airun quantize`), warm: prefill
  128/512/2048 at 6826/8031/7423 t/s, decode 226 tok/s. llama.cpp b11760,
  same file and card: pp512 5983, pp2048 6248, tg128 195.
- 27B on a 5060 Ti: the int8 path gives 889 t/s on a 512 prefill, the same as
  the model's Q4_K (864), against 2016 with FP4. There is no bottleneck
  specific to the format.
- Sweep on the 3060 (`AIRUN_MMQ_TC=64`, `AIRUN_NO_MMQ_STREAMK`,
  `AIRUN_NO_MMQ_SPLIT`): the defaults are already the best. Each session's
  first measurement comes out ~8% lower (card clock ramping up); compare only
  after a warm-up round.
- The installed Nsight Compute 2022.4 profiles the 3060 (sm_86), not the RTX
  50s.

## Vision, MTP in its own file, stream replay (2026-11-05)

- **Vision** (`crates/airun-core/src/vision.rs`, `kernels/vision.cu`): the
  `qwen3vl_merger` ViT as llama.cpp's `tools/mtmd/models/qwen3vl.cpp`. It runs
  on the text kernels: BF16 matmuls, the KV cache attention with every query
  at the last position (= full attention) and the 72-wide heads zero-padded to
  96; FFN of 4304 padded to 4352 (WMMA's k % 64). Selftest: section "vision
  encoder and image positions".
- **Image positions in the text:** `airun_rope_params` gained `shift` (one i32
  on the GPU per shard; the text after an image runs at slot - shift, and the
  CUDA graphs follow the value) and `mpos` (the (t, row, column) positions of
  the image tokens, spread over the interleaved M-RoPE sections). Without an
  image the shift is 0 and everything is bit-identical to before.
- **Images in the prompt** are sentinel ids >= 2^31 (`IMAGE_TOKEN_BASE`), from
  the hash of the embeddings (`Runtime::register_image`), so
  `reusable_prefix` recognizes the same image in the next turn. `generate`
  runs the images through `prefill_images`; with an image MTP is off.
  `save_state` refuses chats with images.
- **MTP in its own file:** `loader::mtp_source` (`--mtp`, or a compatible
  `mtp-*.gguf` in the model's folder). `Model::load` receives where to read
  the head from.
- **Stream replay** (`crates/airun-cli/src/replay.rs`): a middleware copies
  the SSE of requests with `X-Conversation-Id` into the session and consumes
  the generation without the client; `DELETE` drops the body and the
  generation stops at the next token.
- **MTP during prefill no longer waits for the head** on every batch
  (`mtp_queue` instead of `mtp_run`): the `synchronize` kept the first GPU
  from starting the next batch. 27B on two GPUs, 61k prompt: 615 -> 899
  tok/s.
- **GQA decode reads V straight from the f16 cache** (without going through
  shared memory): 128k, 27B on 5060 Ti + 3060, 12.4 -> 13.4 tok/s. The limit
  now is occupancy: the K tile leaves 4 warps per SM.

## Where the time goes (2026-11-05, Nsight Systems 2026.3)

Nsight Systems 2026.x's `nsys` profiles the RTX 50s (Nsight Compute 2022.4
does not). Decode only shows kernel by kernel with `AIRUN_NO_GRAPH=1
AIRUN_NO_DECODE_PIPELINE=1` (or with `--cuda-graph-trace=node`).
`nsys profile -t cuda -o x airun.exe bench ...` and then
`nsys stats -r cuda_gpu_kern_sum -f csv -o x x.nsys-rep`.

27B NVFP4, one 5060 Ti:
- **Decode:** `matvec_q8_wide<NVFP4>` 83% (~388 GB/s, 87% of bandwidth),
  Q5_K output head 5% (~95% of bandwidth), delta-net 2%, attention 1.3% at
  1k. It is already at the memory limit; only reading fewer bytes helps (MTP,
  smaller KV/weights).
- **Prefill:** `matmul_nvfp4` 57% (~190 TFLOPS, half the FP4 peak),
  `delta_rule_reg` 19% (token-by-token recurrence), activation quantization to
  FP4 8%, `add_rmsnorm` 4%, attention 2% at 2k.

## Chunked prefill delta-net, on the tensor cores (2026-11-05)

`kernels/delta_chunk.cu`: the WY form of the gated delta rule (like
flash-linear-attention's `chunk_gated_delta_rule`) in 64-token chunks. A
parallel step per (head, chunk) builds W, U, decayed Q and K and the masked Q
K^T; a sequential step per (head, 32 columns) walks the chunks. Products in
bf16 with fp32 accumulation (WMMA, sm_80+); the state stays in fp32. It kicks
in when the batch has 64+ tokens and 128-wide heads; decode and the MTP verify
stay on the recurrence. `AIRUN_NO_DELTA_CHUNK=1` turns it off.

- 27B on a 5060 Ti: prefill 512 2020 -> 2192 t/s, 2048 1987 -> 2157 (+8.5%).
  35B-A3B on the two GPUs: 1856 -> 1990, 2666 -> 2845 (+7%).
- Perplexity (16k tokens of code, `--chunk 1024 --batched`) unchanged: 2.00 on
  the 27B, 2.06 on the 35B. Selftest: 0.5% of range against the f64
  recurrence.
- Greedy text may flip on a near-tie (it is no longer bit-identical to
  token-by-token decoding); the 61k-token answer stayed right.
- llama.cpp b11760 uses the recurrence here ("TODO: chunked kernel").
- **Measurement:** `nsys` kernel times misled on this kernel (they showed the
  chunked one as slower); decide with `bench` alternating with the variable.
- **Attempts that did not win** (reorganized recurrence): loading k/q in tiles
  of 4 or 16 tokens (-3%), one column per warp with shuffle reduction (-7%), 4
  slices of 32 instead of 8 of 16 (-2.5%). FP4 matmul with 2-block stages and
  4 buffers (8-byte copies): -26%.
- GPU 0 also draws the screen: another program using the GPU dropped prefill
  from 2018 to ~1650 in one round. Always compare at the same moment.

## Load progress bar and reasoning levels (2026-11-05)

- `crates/airun-cli/src/progress.rs`: model loading shows step, percentage,
  layers, GiB and time left (on a line that redraws itself in a terminal;
  every 25% in a log). `run`, `serve`, `bench` and `perplexity`. With several
  GPUs, each has its own line ("GPU 1 RTX 5060, layers 0..15"), which stays on
  screen with "ok" when it finishes. The labels come from
  `loader::upload_labels`. The bar's GiB are the file size times the fraction
  of steps (not each GPU's real bytes); the plan printed afterwards has the
  right values.
- `return_progress` (llama-server): when streaming, the chat sends a chunk
  with `prompt_progress` {total, cache, processed, time_ms} and `choices: []`
  on every prefill batch (`Runtime::set_prefill_hook`, also in MTP and
  `--parallel`). Without waiting on the GPU: the launch queue already keeps
  the host at most one batch ahead. Clients that think they talk to llama.cpp
  (`/props` has `default_generation_settings`) expect this.
- **Cache on hybrids with thinking: missing.** Qwen's template strips
  `<think>` from old answers, so the next prompt does not start with the whole
  history and `reusable_prefix` gives 0 (the delta-net state does not go
  back). In an agent with thinking, every turn reprocesses the context
  (measured: 7.5k tokens, turn 2 with 0 from cache; without thinking, 7,537
  from cache). Path: save the delta-net state at the end of the prompt and
  restore it.
- `/props` carries `details` (training and started context, KV, layers per
  GPU/RAM, MoE with active/total experts, RoPE, MTP, planner notes), and the
  web UI's settings panel shows all of it.
- `crates/airun-cli/src/reasoning.rs`: levels none/low/medium/high/xhigh/max
  (xhigh = 32768 tokens; it used to be a synonym of max). The web UI shows
  off/default/low/medium/high/very high/max; "default" sends no level (the
  server's `--reasoning-effort` applies, else the template). `serve` tries the
  level in the template (Qwen3.8 accepts low, medium, xhigh) and, if the
  template does not read it or refuses the name, uses the thinking budget.
- **`run` did not think** with Qwen3.5/3.8: the plain ChatML prompt did not
  open `<think>` the way the template does. It now follows the template's
  rule (`Thinking::from_template`), like `serve`.
- FP4 gate/up/SwiGLU in one launch (`matmul_nvfp4_kernel<true>`): +2%
  prefill.
## Several requests at once and MTP with temperature (2026-11-05)

- **Slots** (`Runtime::set_slots`, `serve --parallel N`): each slot has its own
  set of `LayerState` (KV and delta-net) per shard; the active one lives in
  the shards and the others in `Runtime::slots`. `switch_slot` swaps the sets
  and discards the decode graphs (they kept the addresses). The planner counts
  the cache N times (`Config::slots`).
- **`decode_rows`**: one token from each slot in a single pass. The matmuls
  and the per-token work run batched; at the 4 stateful points (KV write and
  attention, delta-net convolution and recurrence) the runtime calls the usual
  kernel once per row with that row's slot state (`self.mixed`). The RoPE
  position comes from `mpos` (all three axes equal), so each slot keeps its
  own image shift. No CUDA graph on this path.
- **Scheduler** (`crates/airun-cli/src/sched.rs`): one thread admits one
  request per step (prefill into the free slot with the longest common
  prefix) and advances the active ones together. `run_generation` sends the
  request and receives the tokens over a channel; stop sequences and streaming
  stay on the request's thread. With `--parallel 1` none of this runs.
- Models with QSA, hyper-connections or PLE refuse `--parallel` > 1.
- **MTP with temperature** (`Sampler::sample_draft`): the draft (the head's
  argmax) is accepted with the probability the sampling chain gives it and,
  when refused, the token comes from the distribution without it; test
  `sampling_against_a_draft_keeps_the_distribution` (200k draws). 27B: 25.7
  -> 47.2 tok/s at temp 0.7. Mirostat still has no MTP.

## `--parallel` without a loss for a lone request (2026-11-06, 5070 Ti + 5060)

Measured with `serve`, requests of 200–300 tokens, base (`--parallel 1`) and
`--parallel 4` alternated in the same session (the PC was in use; rounds with
a general ~35% drop were discarded).

- **What was lost:** with `--parallel` > 1 the loader did not load the MTP
  head and every request went through `decode_rows` (whole logits to the host,
  no GPU argmax, no graph chain). 27B NVFP4 `--draft 4`, lone request: greedy
  38.6 -> 27.4 tok/s (-29%), temp 0.7 33.6 -> 27.4 (-18%). Qwen3-4B Q8_0:
  greedy 134.5 -> 126.8 (-6%). Besides, `slot_prefix` ignored the turn
  snapshot: the same prompt again went to an empty slot (0 of 23 tokens from
  cache, prefill 340 against 75 ms).
- **Solo mode** (`sched.rs`, `solo`): with nobody active or queued, the
  request runs through `Runtime::generate_with` on its slot — the same path as
  `--parallel 1`. The callback checks the channel on every token; when another
  request arrives, it returns false, and the refused token (delivered, outside
  the caches) becomes the batched step's `next`, as before. `generate_with`
  receives the `StreamDecoder` from outside and does not flush the tail, so a
  character cut at the pause continues. When the others finish, what is left
  goes back to solo with prompt = slot history + `next` (a 1-token prefill).
  MTP on that return only if the head covers the cache (`spec_keeps_prefix`)
  or redoing costs up to 256 tokens (`REDO_FOR_DRAFTS`): there is one head and
  it goes stale on every slot switch or batched step.
  `AIRUN_NO_PARALLEL_SOLO=1` turns it off.
- **Tokenizing without the runtime lock:** `Runtime::tokenizer` is an `Arc`
  and `AppState` keeps a copy. Without it the solo, which holds the runtime
  during generation, kept the new request from reaching the queue (the 4
  concurrent ones dropped to 42.6 tok/s, one after the other).
- **Snapshot per slot:** `SlotStore::turn_snap`, swapped in `switch_slot`
  (which clears its `pending`: the head was written by others). `slot_begin`
  also takes the snapshot, and `slot_prefix` uses the same rule as
  `reusable_prefix` (`prefix_of`). 27B, `--parallel 2`, two chats with a
  1,237-token system prompt, also at the same time: turn 2 with 1,243 from
  cache in each.
- **Graph for the batched step** (`queue_rows_graphed`, `MixedGraphs`):
  everything that depends on position in the `mixed` path comes from GPU
  buffers (`positions`, `mpos`) and each slot's state buffers never change
  address (`switch_slot` moves the handles), so one graph per shard, per
  ordered slot list, holds while the same slots advance together (up to 8
  lists). If capture fails on the first shard, it goes back to the old path
  with a `note:`. `AIRUN_NO_ROWS_GRAPH=1` turns it off. 4 requests (3
  alternated rounds):

  | Model | no graph | with graph |
  |---|---|---|
  | 27B, temp 0.7 | 57.7–58.6 | **67.3–67.8** |
  | 27B, greedy | 59.2–60.9 | **71.1–72.8** (text identical to no graph) |
  | 4B, greedy | 146–217 | 161–246 (+10–13%, noisy) |

- **Result, lone request:** 27B `--draft 4` greedy 38.2–38.7 against
  36.0–38.3 for the base; temp 0.7 32.3–33.8 against 32.8–33.6; 4B greedy
  126.6/127.2 against 123.9/127.6. Greedy text identical to the base.
- **Still costs:** the cache's VRAM (N times, the automatic context drops) and
  the MTP head, which now also loads with `--parallel` when there is
  `--draft`. Updated in the next section (batched step).
- **Greedy text with several requests at once differs from one at a time**
  (it was already so before this change, checked with
  `AIRUN_NO_PARALLEL_SOLO=1`): the batch sums in another order and a near-tie
  flips; it also depends on when each request comes in. The "answers equal to
  one at a time" sentence in §2's table only holds for requests that do not
  overlap.

## `--parallel` batched step: where the time went (2026-11-06, 5070 Ti + 5060)

Profiled with nsys (`--cuda-graph-trace=node`) and `AIRUN_SCHED_TIMING=1`. 27B
NVFP4 `--tensor-split 5,2` (the BF16 head stays on the 5060), 4 requests. The
stateful kernels added up to only ~5% of the GPU; what weighed:

- **Batch logits: 13 ms per step.** The BF16 output (248k x 5120, 2.5 GB) with
  4 rows went to the 16-token tiled kernel (`matvec_tiled`), and `decode_rows`
  allocated (`cudaMalloc`/`cudaFree`, which synchronizes) and copied to
  pageable memory on every step. Now: `matvec_half_kernel` (BF16/F16, 1..8
  tokens, 16 bytes per lane, each weight read once for all tokens;
  `AIRUN_NO_HALF_MATVEC=1` goes back to the old one) and fixed buffers
  (`rows_logits`, GPU + pinned host), returned as a slice. With 1 token on the
  5070 Ti: 3.30 -> 3.02 ms (the old one already did 770 GB/s; small gain). On
  the 5060 both stay at ~6 ms, which is its bandwidth. **Careful:** this PC's
  nsys labels the GPUs swapped; check by bandwidth.
- **Narrow F32 matvec with several rows:** the delta-net gates (48 outputs)
  with 4 tokens went to the tiled kernel with 3 grid blocks: 55 us against 5
  with 1 token, ~2.8 ms per step. `matvec_tallk_kernel` now takes up to 16
  tokens (`blockIdx.y`; `AIRUN_NO_TALLK_TOKENS=1` turns it off). This also
  applies to the **MTP verify** (5 rows), which fell into the same 3-block
  kernel: 27B `--draft 4`, one request, 3 alternated rounds, greedy 41.2–41.9
  -> **43.9–44.7** tok/s, temp 0.7 35.5–35.6 -> **37.3–38.3**; 4 requests
  80–81 -> 84–86. A few-token prefill (the rest of a cached prompt) also goes
  through it, so a turn's greedy text may flip on a near-tie compared with
  before (seen on the 27B, at character 305).
- **Attention, delta-net and conv once per row:** `attention_split_rows`,
  `delta_rule_rows`, `causal_conv1d_rows`, with the row in
  `blockIdx.z`/`y` and a pointer table per shard (`rows_tab`, K/state of the n
  rows and then V/conv), rewritten before each step; the graphs keep only the
  table's address. One token's attention with short context takes ~70 us
  barely using the GPU (4 one-warp blocks), so joining the 4 rows takes ~3 ms
  off each step. Same arithmetic: bit-for-bit equal (case in the selftest).
  `AIRUN_NO_ROWS_BATCH=1` goes back to the loop.
- **Sampling:** each row on a thread (`sample_rows`); it cost little (~1 ms for
  the 4), but it was serial.

Result, without per-step logging, alternated rounds (summed t/s, greedy /
temp 0.7); greedy text identical across variants in every round:

| | all | no batching (`NO_ROWS_BATCH`) | no `matvec_half` |
|---|---|---|---|
| 27B, 4 requests (3 rounds) | **81–86 / 89–92** | 75–80 / 84–88 | 79–82 / 85–90 |
| 4B, 4 requests (2 rounds) | **362–377 / 361–380** | 294–306 / 317–326 | 351 / 375–383 |

Before this round the 4B with 4 requests did 233–246, and the 27B 67–73
(`-c 16384`, which makes attention a bit more expensive).

- **Measure without per-step logging.** With `AIRUN_SCHED_TIMING=1` (one
  stderr line per step), the interval between steps grew from 45 to 50–62 ms
  (p90 +26 ms) and t/s dropped to 51–60. Use the log to see how the step
  splits, not to measure speed.
- **Missing:** admission does one prefill at a time (4 short prompts: ~1.5 s
  before the first batched step), and the rest of the step is weight matvecs,
  which already read each weight once for all rows.

## Automatic context and the head on the fast GPU (2026-11-06, 5070 Ti + 5060)

The user saw `--parallel 4` faster than without the flag with a single
request. Measured (27B NVFP4, `--mmproj`, `--kv-type q4_0 -b 256 --draft 3`,
no `-c`, 3 alternated rounds): 63.8 against 52.5 tok/s greedy, 47.6 against
36.4 at temp 0.7. It was not `--parallel`: without the flag the automatic
context rose to 127,488, where the layout with the fast GPU last
(`fill_fastest_last`) no longer fit, and the planner fell back to "filling GPU
0 first", which leaves the output head (and MTP, which reads it on every
draft) on the 5060. With `--parallel 4` the context was 25,856 and the good
layout fit.

`auto_ctx` (`loader.rs`) now, between unequal GPUs (without `--main-gpu` or
`--tensor-split`), if the largest context puts the head off the fastest GPU,
looks for the largest context, up to 1/4 smaller, at which it stays on the
fastest, and warns with a `note:`. Here: 126,976 instead of 127,488, and
without the flag it went to 63.8–63.9 / 47.3–47.6, the same as `--parallel
4`. An explicit `-c` does not change.

## RAM during loading (2026-11-07)

**The symptom:** while loading, RAM rose up to the model's size and only came
down at the end. On a PC with other apps open, they had to be closed to load,
even though it fit afterwards.

**The cause:** the GGUF is read through mmap. Every page read for the upload
stayed in the process's working set until the end of the load. Measured on
the 35B-A3B Q4_K_M (5060 + 3060): working set up to 25.6 GiB, and the system's
free RAM dropped from 47 to 24 GiB.

**The fix:** `GgufFile::release(prefix)` (`airun-gguf/src/reader.rs`) takes the
pages of the tensors already uploaded out of the working set, with
`VirtualUnlock` on unlocked pages. The data stays valid: a later read, for
example of experts read from the file, brings it back. `Model::load` calls it
after the embedding, the output and each layer. Outside Windows it does
nothing, because mapped pages are already reclaimable cache.

**After:** the working set stayed stable at ~5 GiB (the experts pinned in RAM,
which stay during use) and free RAM did not drop below 44.9 GiB. The load took
19.6 s; before, 25–55 s, depending on the disk cache. Greedy text is
identical.

**What remains:** "private bytes" reach ~21.7 GiB. On WDDM, every VRAM
allocation counts against the process's commit (a whole 4B in VRAM: 5.5 GiB
private with 286 MiB working set), and the pinned experts add up. That does
not take physical RAM, but it must fit the commit limit (RAM + page file).
With the page file off or small, loading can fail even with free RAM.

## Out of memory on the first request with MTP (2026-11-07)

**The symptom:** a 35B-A3B at the VRAM limit loaded, but the first request
failed with "out of memory on GPU 0: tried to allocate 16.12 MiB, but only 0 B
of 0 B is free". With `--draft 0`, it worked. VRAM also rose and fell between
requests.

**The causes:**
- **Verify scratch outside the per-GPU plan:** `verify_partial`
  (`(draft+1) × n_head × 256 splits × (head_dim+2)` floats; 16.12 MiB with
  `--draft 3`) and the delta-net stash are allocated on **every** GPU, but
  `spec_scratch_bytes` charged both only to the output head's GPU. The other
  GPU, full up to its budget, went without them.
- **Lazy allocation:** `spec_init`, the per-slot head caches and the batched
  step's `rows_logits` were only allocated on the first request. `rows_logits`
  was freed and reallocated larger every time the batch grew: that is the VRAM
  that rises and falls. A failure there reached the user, instead of falling
  to the loader's next rung.
- **Decode split-attention** was counted with 128 splits; `attention_splits`
  goes up to 256.
- **"0 B of 0 B":** `cudaMemGetInfo` right after the failure sometimes does
  not answer. The hint also cited llama.cpp flags (`--gpu-layers`,
  `--split-mode`), which airun does not have.

**The fixes:**
- `Config::spec_rows`: each GPU's `plan::scratch_bytes` counts `(1 +
  spec_rows + slots if > 1) × 256 splits`, and one slot's logits per row.
- `model::spec_stash_bytes` goes into `sizes.per_layer` of each linear layer.
  `spec_scratch_bytes` kept only what stays on the head's GPU.
- `Runtime::reserve_buffers` (called in the loader, after `set_slots`)
  allocates MTP and `rows_logits` at maximum size. A capacity failure there
  goes to `next_rung`.
- The OOM message omits the numbers when the driver does not give them, and
  suggests `-c`, `--kv-type`, `--draft 0`, `--vram-reserve`.

**Tested** with the 35B-A3B Q4_K_M, `--expert-cache 200 --draft 3` (5060 +
3060; the plan fills the 3060 to 10.28 of 10.61 GiB):
- 4 requests in a row: all OK, VRAM stable (+6 MiB per request, graphs).
- `--parallel 2`, two rounds of 2 concurrent requests: OK, VRAM the same
  between rounds.

**Side finding, not fixed:** `Qwen3.6-35B-A3B-NVFP4-MTP-HQ` with
`--expert-cache` fails on the first request with "gate and up experts must
share a type and layout".

## Image marker (2026-11-07)

**The error:** a chat that only quoted `<__media__>` (code, a log, airun's
documentation) gave HTTP 400: "the prompt has 1 image markers (<__media__>)
and 0 images were given".

**The fix:**
- `attach_images` now puts a per-process random marker in place of each image
  (`media::slot()`, as llama-server does). `tokenize` splits the text only on
  it.
- `<__media__>` only becomes an image slot where airun documents it that way:
  `run --image` and `/completion` with `multimodal_data`, via
  `media::mark_slots`.
- Elsewhere, `<__media__>` is plain text.

**Tested in `serve`** (27B with mmproj):
- chat quoting the marker without an image;
- image together with text quoting the marker;
- `/completion` with `multimodal_data`;
- image in Anthropic's format.

## MTP verify attention at long context (2026-11-07)

**The symptom.** In `serve` with MTP, generation dropped with the
conversation: ~50 tok/s up to ~9k tokens, 26 tok/s at 27k. The verify (1 +
drafts tokens) went to `attention_masked`, the prefill kernel, with grid
(tokens, heads): 96 blocks, each reading the whole context.

**The fix:**
- **Path:** `ops::attention_split_multi`, chosen in `runtime::split_verify`;
  it also serves the batched verify and the MTP head.
- **Kernel:** `attention_decode_gqa_kernel` with W warps per block, one per
  query. They share each K/V tile in shared memory, loaded by all the warps,
  and each warp masks the keys past its own position.
- **In isolation** (`tests/attention_bench.rs`), per layer, 4 queries over a
  q4_0 cache, 5070 Ti:
  - 25k cells: 537 µs, against 1520 in the old kernel and 1205 splitting per
    row;
  - 8k cells: 167 µs, against 490.

**Determinism:**
- **Fixed boundaries:** in multi-warp mode, the splits have fixed boundaries
  (`verify_split_keys` = max(64, ⌈ctx/splits⌉)), not dependent on position.
- **No occupancy rounding:** there is no "wave" in multi-warp mode.
- **Merge:** joins up to the last query's split, with `pos_stride` 0.
- **Why it matters:** the same token is query 0 of a lone verify and query 2
  of a batched verify. With boundaries that depended on position,
  `--parallel`'s greedy text diverged from the lone text; now it comes out
  identical with 2, 3 and 4 requests.
- **Tests:** `verify_attention_ignores_the_drafts_count` (bench) and
  `test_split_multi` (selftest).
- **`gqa_stage`:** now does not write rows past n, because the warps share
  the tile.

**Measured** (27B NVFP4, 5070 Ti + 5060 Ti, `-c 32768 --kv-type q4_0 --draft
3`, 300 tokens):

| prompt | before | after |
|---|---|---|
| 25k tokens | 34.2 tok/s (verify 75 ms) | 52.8 tok/s (verify 42 ms) |
| short | 54 tok/s | 54.8 tok/s |

`AIRUN_NO_SPLIT_VERIFY=1` goes back to the old kernel.

## RAM layers multiplied on the CPU (2026-11-06, RTX 5060 8 GB alone)

**The problem:** the machine was left with only the 5060, on a PCIe 3.0 x1
link (~0.8 GB/s; `nvidia-smi -q`: Link Width Current 1x, Max 16x). The 27B
NVFP4 left 47 layers (9.4 GiB) in RAM, copied to VRAM on every pass: 11 s per
token, 0.1 tok/s.

**The solution (`cpu.rs`):**
- **AVX2 NVFP4 kernel:** `pshufb` with the e2m1 LUT, `sign`/`maddubs`/`madd`,
  UE4M3 scale × activation scale, int8 per 32 like the GPU's `quantize_act`.
  It takes 4 tokens at a time, decoding the block only once for the 4.
- **Fallback:** `dequantize` plus an f32 product, for F32 and the other types.
- **Own thread pool:** 32-row chunks taken dynamically (P-cores and E-cores
  perform differently) and 200 µs of spinning between jobs, because there are
  ~300 matmuls per token.
- **Isolated measurement** (`tests/cpu_bench.rs`, 17408×5120 matrix): 50–58
  GB/s from 1 to 4 tokens; 26 GB/s with 8 tokens, already compute-bound.

**Runtime integration:**
- **Per-tensor decision:** `on_cpu(w)` holds for `w.ram` present and
  `cpu::pass_on_cpu()`. `queue_shard` calls `cpu::set_pass(n)`, which turns
  the CPU on for passes of up to `AIRUN_CPU_MAX_TOKENS` (64) tokens.
- **Prompt:** larger chunks copy the layers as before. On x1, a 2550-token
  prompt did 40 tok/s copying and 10.6 on the CPU; the break-even is at ~150
  tokens per chunk.
- **`cpu_matvec`:** activation to pinned memory (once for q/k/v:
  `host_valid`), sync, CPU, product back.
- **`cpu_ffn`:** gate, up, SwiGLU and down at once.
- **Fused paths bypassed:** fused qkv, fused GLU and prefill FP4 skip these
  tensors.
- **CUDA graphs:** off when there are layers on the CPU (`graphs_disabled()
  || cpu::in_use()`).

**Planner:** when weights go to RAM anyway, the KV stays in VRAM first. On a
slow link, the KV in RAM cost ~100 ms per token with 4096 cells.

**Measured** (27B NVFP4, 5060 x1, i7-13700, DDR5-4800):

| | before | after |
|---|---|---|
| generation without drafting | 0.1 tok/s | 3.2 |
| generation with `--draft 2` | — | 5.3 (74% accepted) |
| short prompt | 1.1 tok/s | 10.7 |

- **Exactness:** greedy text identical to `AIRUN_RAM_ON_GPU=1`.
- **Threads:** 8, 16 and 24 give the same result, because RAM bandwidth
  saturates.
- **Where the time goes:** ~70% is CPU compute, close to bandwidth; ~20% is
  waiting for the GPU (17 layers, BF16 head and the round trips over the x1
  link).

## MTP in `serve --parallel`'s batched steps (2026-11-06)

**Head per slot.**
- `spec::SlotHead` (KV, `pending`, `synced`) lives in `SlotStore::head`; the
  active slot's stays in `Spec`.
- `switch_slot` swaps the heads (`Spec::swap_head`) instead of marking the
  head as out of sync.
- `decode_rows` only marks out of sync the rows it advanced.
- `slot_begin` feeds the head during prefill (`head_prefill_chunk`) when there
  are drafts, with the same rules as `generate_spec`.
- The planner already counted one head per slot (`kv_bytes_per_layer` ×
  slots).

**Batched speculative step** (`sched::spec_step`):
1. `draft_rows`: first drafts for whoever has none, from the slot's `pending`.
2. `verify_rows`: a mixed pass with `mixed_t = 1 + k` tokens per row, `defer`
   on and delta-net stash for every row. It is captured as a graph in
   `MixedGraphs`, with the key including `t`.
3. The scheduler picks the accepted ones (greedy from the GPU's picks;
   sampling, `sample_draft` on threads).
4. `accept_rows` does three things:
   - `commit_rows` advances each delta-net with only the m accepted;
   - the head absorbs each slot's t rows (extra rows are rewritten before
     anyone attends to them);
   - `chain_rows` generates the next drafts.

Whoever has no synced head goes in with filler drafts, which stays exact.

**Runtime changes in mixed mode:**
- **Attention** (`run_attention`, `mixed_t > 1`): each row writes t cells and
  uses `attention_masked`. The `keys` argument is constant (`ctx`), so the
  graph holds at any position.
- **`mixed_head`:** swaps the cache for the slot's head.
- **Delta-net:** `oop` conv and `delta_rule_ex` per row, with t tokens.

**Kernels:**
- **NVFP4, 9 to 16 tokens:** two passes of up to 8 in
  `matvec_q8_wide_multi_kernel` (~175 µs on the 17408×5120 FFN with 16
  tokens). The tiled kernel took 734 µs; one 16-token pass, 204 µs.
- **MMQ:** would take 147 µs, but stream-K numbers the launches (epoch passed
  as an argument) and does not work in a graph.
- **Types without a multi-token kernel** (IQ4_XS, which is this 27B's MTP head
  layer): from 2 to 16 tokens, one call per token. Before, 1.67 ms for 16
  tokens in the tiled kernel.
- **Isolated measurement:** `crates/airun-cuda/tests/matvec_bench.rs`.

**Measurements** (27B NVFP4, 5070 Ti + 5060, `--mmproj ... --kv-type q4_0 -b
256 --draft 3 --parallel 4`, 200 tokens, greedy, summed tok/s):
- **Drafts by number of requests** (`ROWS_DRAFT = [2, 1, 1]`):
  - 2 requests: 81.5 with 2 drafts, 78.5 with 3, against 59 without MTP;
  - 3 requests: 98 with 1, 81 with 2, against 82;
  - 4 requests: 117 with 1, 98 with 2, against 106;
  - 4 requests, temp 0.7: 118 against 106.
- **`-c 262144` with KV in RAM:** 2 requests 78 against 54; 4 requests 107
  against 88.5.
- **Exactness:** at temp 0, each request's text in the batch is identical to
  its text alone (2 and 4 requests).

**VRAM pitfall.**
- **What happens:** the layout that puts the output head on the fast GPU
  fills that GPU up to its budget. Buffers allocated after the plan pushed it
  ~125 MiB past it: per-slot stash, batch logits, head rows and graphs.
- **Effect:** Windows started paging VRAM. Steps of 110 ms went to 300–650 ms,
  suddenly and mid-generation, with one GPU at 100% and the other idle. Even
  the lone request dropped from 57–63 to 45–55 t/s.
- **Fix:** `model::spec_scratch_bytes` goes into the plan together with the
  head.
- **Diagnosis:** if it shows up again, try `--vram-reserve 900`. If that
  solves it, it is an allocation outside the plan.

## KV cache in RAM before weights, and `--hide-gpu` (2026-11-06)

**KV in RAM.** With an explicit `-c` that does not fit, the planner
(`plan.rs`, `plan()` → `place`) tries, in this order:
1. everything in VRAM;
2. the KV cache of the last k attention layers in RAM, with k growing;
3. all the KV in RAM plus the weights left over (`spill_to_ram`).

How it works:
- **Why the KV goes first:** weights are read whole on every pass; of the
  cache only the filled cells.
- **Planner:** `try_place(.., kv_ram, spill)`.
  - The stage (K+V of one layer of **one** slot, `Config::kv_stage_bytes`) is
    charged to the last layer moved during the split and, at the end, to each
    shard that has any (`Shard::kv_stage_bytes`).
  - Only for an `alloc` that allows RAM, without MoE, and with cache rows that
    are multiples of 4 bytes.
- **Memory:** `DeviceBuffer::in_ram` (mapped pinned; `is_in_ram`).
- **Runtime:**
  - `kv_stage_in` uses `ops::kv_stage(to_stage=true)` to copy cells `[0, max
    pos+1)` of each head into `DeviceState::kv_stage`. The cache write and the
    attention use the stage.
  - `kv_stage_out` returns only the new cells.
  - The length comes from `positions` on the GPU, so the decode graphs hold at
    any position.
  - In the batched step (`mixed`), layers with KV in RAM go row by row,
    without the batched kernel.
- **Memory copies:** now use `cudaMemcpyDefault` (`device.cu`), because a
  "device" buffer can be mapped RAM.

Measured on the 27B NVFP4 (5070 Ti + 5060, `--kv-type q4_0 -b 256 --draft 3`):

| Case | Before | After |
|---|---|---|
| `-c 262144`, serve | 15.7 tok/s (13 weight layers in RAM) | 63.0 (KV of 10 of the 16 layers in RAM) |
| `-c 262144`, temp 0.7 | 12.3 | 49.3 |
| `-c 262144 --parallel 4` | 6.1 (context cut to 131072, 43 layers in RAM) | 60.9 (context kept, 18 GiB of pinned KV) |
| Decode at position 65536 | 18.9 | 18.9 |

- With greedy `run` the text comes out identical to `AIRUN_NO_KV_RAM=1`.
- The 921-token prompt's prefill dropped from 1526 to 1249 tok/s, because the
  layout changed.
- Selftest: `test_kv_in_ram`. Planner test:
  `the_kv_cache_goes_to_ram_before_weights`.

**`--hide-gpu` / `AIRUN_HIDE_GPUS`** (`airun-cli/src/hide.rs`, called at the
start of `main`):
- A child `airun gpu --json` lists the GPUs. Then `CUDA_VISIBLE_DEVICES`
  receives the UUIDs of the ones that stay, so the runtime never opens a
  context on the hidden ones (measured: 5060 with 0 MiB).
- Each item accepts:
  - a number, which is an ordinal if some GPU has it, else part of the name
    (`5060`);
  - a UUID;
  - part of the name.
- The rest are renumbered from 0. A `note:` gives the new numbering, and
  `airun gpu` lists the hidden ones.
- Does not combine with an already-set `CUDA_VISIBLE_DEVICES`.

## Decode measurements with MTP: reserve, draft, head (2026-11-05, 2× 5060 Ti)

Machine: 2× RTX 5060 Ti (twins, sm_120; the one on bus 03 draws the screen).
Qwen3.8-27B NVFP4, `-c 131072 -b 256 --kv-type q4_0 --temp 0 --repeat-penalty
1 -n 300`, `AIRUN_SPEC_TIMING=1`, variants always alternated in the same
session. On this machine the rounds varied < 1% (mean clock 2.8 GHz, no
drops); the old PC's 20–40 t/s noise did not appear. Split, the planner puts
32/32 layers and each GPU sits at ~9 GiB.

- **`--vram-reserve` 384 against 1024 (WDDM hypothesis):** changes nothing.
  Split: 44.3–44.8 t/s in both. To reproduce the monitor GPU full, the whole
  model on it (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 --alloc
  gpu`, context reduced to 16k, peak 15.7 of 16.3 GiB): 54.4 t/s in both; the
  drop in rounds 3–4 (to 48.1) hit both variants equally. The default reserve
  stays at 384. Without `--alloc gpu`, `auto` sent 2.4 GiB of weights to RAM
  and decode dropped to 3 t/s (verify 882 ms): if a user reports slow decode
  with the GPU full, check the "loaded X GiB".
- **`--draft`** (t/s, draft acceptance in parentheses):

  | Config | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|
  | split, prose (5+4 rounds) | 40.3 (69%) | 43.2 (57%) | **45.0** (53%) | 42.9 (44%) |
  | split, code (4 rounds) | — | 47.6 (67%) | 48.2 (58%) | **49.0** (53%) |
  | one GPU, prose (4 rounds) | — | 54.1 | **55.0** | 51.4 |

  Verify per step: 52.7 / 56.2 / 58.4 ms (3/4/5) split; commit+absorb 10.2 /
  13.0 / 15.7 ms. 4 beat 3 in every configuration (and in the old PC's sweep,
  61.4 against 52.3), so `DEFAULT_DRAFT` became 4. 5 only wins on very
  predictable text.
- **Split verify with one graph per GPU (2026-11-06):** `verify_sharded` in
  `spec.rs` captures each shard's part (and the picks on the last one) as one
  CUDA graph per GPU, with the residual moving between them outside the
  graphs (`handoff`), like decode. Before, split, the verify was launched
  kernel by kernel (only one GPU had a graph).
  `AIRUN_NO_VERIFY_SHARD_GRAPHS=1` goes back to the old one. `switch_slot`
  now also discards the MTP graphs.
  - 27B, 128k q4_0, draft 4, 4 alternated rounds: 44.4 → **51.8 t/s**, verify
    57.1 → 46.9 ms, text identical to the old path.
  - 35B-A3B Q5_K_M, `-c 32768`, 2 alternated rounds (before the graph, draft
    4 gave 96):

    | 35B-A3B | no MTP | draft 2 | draft 3 |
    |---|---|---|---|
    | 2 GPUs, all in VRAM, prose | 109.6 | **137.6** (76%) | 128.2 (62%) |
    | 2 GPUs, all in VRAM, code | 108.8 | **132.1** (71%) | 127.4 (62%) |
    | 1 GPU, experts in RAM | **44.7** | 34.5 | 32.0 |

    That is why `MOE_DEFAULT_DRAFT = 2` (`loader.rs`) only applies when
    `plan_experts`, counting the head and its cache, says the model fits
    whole; otherwise MTP stays off, with a `note:`. `--alloc gpu` turns it on
    (nothing goes to RAM); `--expert-cache` turns it off.
  - Qwen4 Flash-Next has no MTP head; it does not go through here.
  - This machine: GPU 0 (bus 03, the monitor's) is on PCIe 3.0 x4; GPU 1 on
    4.0 x8. Plain split decode 25.1 against 26.4 t/s on a single GPU.
- **Draft with a cut vocabulary (FR-Spec, arXiv 2502.14856):**
  `AIRUN_DRAFT_VOCAB=K` makes the head choose only among the first K rows of
  `output.weight` (in BPE, low ids ≈ most frequent merges); the verify keeps
  the whole vocabulary, so the output does not change. **Off by default**,
  because it depends on the language (27B split, draft 4, t/s and
  acceptance):

  | Prompt | whole | K=151643 | K=65536 |
  |---|---|---|---|
  | code (EN) | 55.3 (58%) | 58.6 (58%) | **59.5** (55%) |
  | English story | 44.2 (41%) | 46.9 | **49.1** (40%) |
  | Portuguese prose | **51.8** (53%) | 49.7 (45%) | 52.2 (45%) |

  The advance drops from 13.2 to 6.9 ms with K=65536. Portuguese uses the
  vocabulary extension's tokens (ids > 151k), which the prefix cuts. The path
  to turn it on by default: prefix + the tokens already seen in the chat
  (rows copied into their own buffer, with an index → id map after the
  argmax). On the 35B-A3B with experts in RAM (2026-11-10), K=16384/32768 cut
  the head from 2.7 to 1.3 ms but acceptance fell to 34–57%: no gain.
- **Cache between turns on hybrids (2026-11-06):** `TurnSnapshot` in
  `runtime.rs`. KV rows hold up to any position, but the recurrent state
  (delta-net, conv, QSA ring and `pooled`, PLE history) only moves forward;
  since the template rewrites the previous answer's `<think>`, the next
  request diverged right after the assistant header and a hybrid reprocessed
  the whole conversation every turn. Now the prefill stops exactly before the
  prompt's last `<|im_start|>` (`snapshot_point`), copies that state into a
  reused pinned buffer (`take_snapshot`, asynchronous copies on each GPU's
  stream, without waiting) and continues; in the next turn `reusable_prefix`
  accepts that point and `rewind_to` restores the state
  (`restore_snapshot`), plus the MTP head's `pending` row when the prefill
  that took the snapshot ran the head. `walk_state(false, ..)` is
  `save_state` without the KV rows. `AIRUN_NO_STATE_SNAPSHOT=1` turns it off.
  - 27B split, `serve`, ~2,760-token system prompt, 3 turns: turn 2 reused
    2,760 (prefill 443 ms, most of it building the decode graph for the first
    time) against 878 ms without the snapshot; turn 3 reused 2,788, prefill
    **160 ms** against 895.
  - Correctness: the same request twice (the second restores and reprocesses
    from the same boundary) comes out **identical** on the 27B (MTP on and
    off) and on the 35B-A3B fully in VRAM.
  - Splitting the prefill at that point changes the first turn's batch
    boundaries, so its greedy text may flip on a near-tie (like MTP).
- **Side bug, found in the test above:** MoE with experts in RAM and batches
  of 2–15 tokens is not deterministic. 35B-A3B on one GPU,
  `AIRUN_NO_STATE_SNAPSHOT=1 -b 4`: three equal requests, three different
  texts (with large batches the same test comes out identical). Cause, as
  confirmed on 2026-11-10: with the two passes (resident, then copied) the sum
  order depends on which experts are resident, and residency changes between
  runs (`rebalance_experts`). With `AIRUN_MOE_OVERLAP=0` the text is
  identical. That is why Qwen4 (experts in RAM) is not good for validating the
  snapshot without that variable.
- **A Q8_0 copy of the head for MTP: not worth it.** This file's
  `output.weight` is Q5_K (874 MB), not BF16 — the old comment in `plan.rs`
  was wrong. A Q8_0 (1.35 GB) would read more bytes, not fewer.


## Disk offload: `--disk-offload` (2026-11-09)

**What it does:** the order becomes VRAM → RAM → disk. The layers the planner
would already send to RAM (`Shard::ram_layers`) that do not fit the free RAM
stay in the GGUF itself, without a copy.

- **Decision** (`loader.rs`, `place_on_disk`, after `plan()`): budget = RAM
  available now (`disk::available_ram`: `GlobalMemoryStatusEx` /
  `MemAvailable`) minus `--ram-reserve` (default 1/10 of RAM, 2–8 GiB), minus
  the KV in RAM, the host embedding and the mapped experts.
  `plan::spill_to_disk` keeps pinned the first RAM layers that fit and puts
  the rest in `Shard::disk_layers` (a subset of `ram_layers`, bytes also in
  `ram_bytes`). Without the flag, only a `note:` suggesting `--disk-offload`.
- **Load** (`model.rs`, `load_disk`): `RamWeights::host` became
  `HostBytes::{Pinned, File}`. `File` keeps the `Arc<GgufFile>` and the
  `TensorInfo`, and `as_slice()` is the mmap slice. Nothing is read at load
  (27B: 4.3 s against 23 s pinning 6.2 GiB).
- **CPU (generation):** `cpu::matvec` reads the mmap slice; the OS pages from
  the SSD. No sanitize needed: `ue4m3_to_f32(0x7F)` is already 0.
- **GPU (prompt chunks > `AIRUN_CPU_MAX_TOKENS`, or `AIRUN_RAM_ON_GPU=1`):**
  `RamCopy` (replaces the `ram_copies` tuple) copies from pageable memory and
  runs `nvfp4_sanitize` on the staging when `nvfp4_blocks > 0`. A copy from
  pageable memory does not go into a graph: `disk::in_use()` turns graphs
  off.
- **Prefetch** (`disk.rs`, `Prefetcher`): `queue_shard` calls
  `before(layer)`; 4 threads (`AIRUN_DISK_THREADS`) do
  `PrefetchVirtualMemory`/`madvise(WILLNEED)` and touch one page every 4 KiB
  of the next 2 disk layers (`AIRUN_DISK_AHEAD`), in 8 MiB pieces, wrapping
  around for the next token. The threads hold an `Arc` of the file, so the
  mmap lives while they run.
- **MoE:** if the experts that would go to mapped RAM (`MappedExperts`)
  exceed the budget, `model::stream_experts()` forces the
  `AIRUN_EXPERTS=stream` path (misses read from the file). It is all or
  nothing; there is no half mapped, half from the file.

**Validated** (5070 Ti, `--alloc memory --vram-reserve` to push layers out,
`--ram-reserve 1000000` to put all of them on disk), greedy text identical to
pinned RAM:
- Qwen3-1.7B Q8_K_XL, 26 layers on disk: 12- and 358-token prompts, with and
  without `AIRUN_RAM_ON_GPU=1`.
- 27B NVFP4 (MTP on), 31 layers / 6.19 GiB on disk: 14- and 360-token
  prompts. Generation 9.6–9.7 tok/s against 10.0 in RAM.
- Planner test: `what_the_ram_cannot_hold_goes_to_disk`.

**Not measured:** a cold file cache. In the tests above the file was in the
Windows cache (spare RAM), so the number is RAM's, not the SSD's. With RAM
actually full, the ceiling is the SSD bandwidth divided by the bytes on disk,
per token. The MoE expert path with the flag was not run either (there is no
small MoE on this machine); it uses the streaming that already existed.

**Speed analysis (2026-11-09, SK hynix HFS256GEJ9X110N NVMe on D:, cold read;
`tests/disk_bench.rs`):**

| Method | GB/s |
|---|---|
| page faults, 1 thread | 0.93 |
| page faults, 8 threads | 1.58 |
| hint + touch, 4 threads, 8 MiB (the `Prefetcher`) | 1.61 |
| hint + touch, 8 threads | 1.61 |
| hint + touch, 16 threads, 4 MiB | 1.23 |
| cached ReadFile, 8 threads | 1.20 |
| unbuffered ReadFile (`FILE_FLAG_NO_BUFFERING`), 8 / 16 threads | 1.32 / 1.38 |

- **Reading:** the `Prefetcher` is already at the disk's ceiling. Reading
  unbuffered into a pinned ring gains no bandwidth; it would only make memory
  use fixed.
- **Decode:** with a cold disk, the time per pass is bytes on disk ÷ bandwidth
  (27B: 6.65 GB → ~4.1 s). What speeds it up is more tokens per pass. MTP
  tokens per pass (27B, 160 tokens of code): draft 2 → 2.8; 4 → 3.8; 6 → 4.3;
  8 → 4.3. That is why `DISK_DEFAULT_DRAFT = 6` (loader.rs): with layers on
  disk and no `--draft`, it projects 1.05 against 0.92 tok/s. The MTP buffers
  are sized for 6 before the plan; if nothing goes to disk, the runtime uses 4
  within them. With the file in the OS cache (spare RAM, as in the tests
  here), the smaller draft wins (9.9 / 9.1 / 8.0 tok/s with 2 / 4 / 6); but
  layers on disk only exist when RAM has run out.
- **Layer-by-layer prefill (`step_layered`, runtime.rs):** a `-b`-token chunk
  reread every disk layer; a 4096 prompt with `-b 512` read the disk 8 times.
  Now `layered_take` sets aside the whole chunks before the last one (or
  before the snapshot point), up to `AIRUN_LAYERED_TOKENS` (8192), and
  `step_layered` runs them layer by layer: the VRAM layers batched per chunk,
  each RAM/disk layer alone over all chunks. Between one stretch and the
  next, each chunk's `x` and `xb` stay in a buffer (VRAM, or mapped RAM if
  short), because with fused norms the `xb` left by the previous layer is the
  next one's input. `n_past` is adjusted per chunk (attention uses `n_past +
  n`), and the positions are re-sent. `queue_shard` became `shard_input` +
  `shard_layers` + `shard_output`. In `generate_spec`, the callback runs
  `stage_h`/`save_pending`/`mtp_queue` of each chunk right after the final
  norm. 4420-token prompt: the disk is read 2 times instead of 9; cold, at 1.6
  GB/s, ~8 s instead of ~37 s of reading on the 27B (projection; here the file
  was cached).
  - **Exactness:** greedy text identical to `AIRUN_NO_LAYERED=1` with 4420
    tokens: 1.7B with disk; 27B with disk with and without MTP; 27B with disk
    on two GPUs.
  - **Only with layers on disk:** with layers only in pinned RAM it lost (27B,
    4420 tokens: 3.7 → 3.8 s on one GPU, 3.2 → 3.9 s on two, which stop
    overlapping), so `layered_take` requires `model.disk`. Also excluded:
    hyper-connections, PLE, images and `serve --parallel`'s batched prefill.
- **Side finding:** on Qwen3-1.7B Q8_K_XL, the RAM layers multiplied on the CPU
  gave 2.5 tok/s and copied to the GPU (`AIRUN_RAM_ON_GPU=1`) 16.7 tok/s. The
  `dequantize` + f32 fallback in `cpu.rs` (types other than NVFP4) is slow,
  and with fast PCIe copying wins. The CPU × copy choice does not look at the
  type or the link's bandwidth. (Q4_K/Q5_K/Q6_K/Q8_0 got their own kernels on
  2026-11-10; see below.)

## RAM layers in K-quants and MoE with experts in RAM (2026-11-10, RTX 3060 12 GB alone)

**CPU matmuls for Q4_K, Q5_K, Q6_K and Q8_0** (`cpu.rs`,
`avx2::q4k_row4`/`q5k_row4`/`q6k_row4`/`q8_0_row4`):
- Before, only NVFP4 had a kernel; the other types went to `dequantize` + f32,
  at ~1.5 GB/s. Qwen3.6-27B Q4_K_M (84% Q4_K/Q6_K) with 7.5 GB in RAM
  generated at 0.2 tok/s.
- Weights are decoded in registers and multiplied as int8 (`maddubs`), as in
  NVFP4. The K-quants' minimum goes in as `dmin * mn * activation block sum`
  (`quantize_sums`).
- `tests/cpu_bench.rs`: 1 token at 49–52 GB/s (Q4_K/Q5_K/Q6_K), Q8_0 at 27
  GB/s.
- A matmul with less than 1 MiB of weights runs on the calling thread
  (`run_sized`): waking the pool cost 205 µs for 138 KB.
- **Do not** raise the threads' spin wait (`AIRUN_CPU_SPIN_US`): at 5 ms
  generation dropped from 3.0 to 0.9 tok/s, because 23 spinning threads take
  CPU away from the thread driving the GPU.
- `AIRUN_CPU_STATS=1`: every 1000 matmuls, the time inside them and the
  bandwidth.
- **Result** (27B Q4_K_M, 3060, `--draft 0`): generation 0.2 → 4.6–4.8 tok/s;
  copying to the GPU (`AIRUN_RAM_ON_GPU=1`), 2.8. Greedy text identical on
  both paths.
- Measuring in sequence inflated rounds: a run right after another that freed
  a lot of pinned memory came out at ~3 tok/s. Repeat the measurement.

**MoE experts in decode on the integer path** (`moe.cu`,
`moe_gate_up_q8_kernel`, `moe_down_q8_kernel`):
- The expert kernels decoded each weight to float: on the 35B-A3B they read
  ~566 MB per token at ~18% of bandwidth.
- With the `q8` buffer (`MoeScratch::q8`, `hybrid::moe_q8_scratch_bytes`), up
  to 4 tokens quantize the input (and `h`) to int8 per 32 and use `q8dot`
  (`dp4a`), like the dense matrices. `AIRUN_MOE_F32=1` goes back to the float
  kernels.
- Selftest: "moe expert ffn, int8 decode kernels" (1.1% of peak with random
  data). The 35B's greedy text identical to the float path's.

**Copying the missing experts with few blocks** (`moe_stage_copy_kernel`,
`AIRUN_STAGE_BLOCKS`, default 4):
- The RAM → VRAM copy runs on a queue parallel to the resident experts. With
  32 blocks per expert (up to 256 at a time, each stuck waiting on PCIe), it
  took the GPU and delayed the other queue's kernels.
- Now it is 4 blocks per expert, with 8 16-byte loads in flight per thread.

**Result on the 35B-A3B Q4_K_M** (`-c 246000 --kv-type q8_0 --temp 0.7`, 7.16
GiB of weights in VRAM, ~78% of experts in VRAM):

| Step | tok/s |
|---|---|
| before | 42–44 |
| int8 experts | 48 |
| copy: 32 → 4 blocks per expert | 58.7–60.6 |

- Block sweep: 32 → 51.6; 8 → 55; 4 → 58.7–60.6; 3 → 60.0–60.4; 1 →
  55.7–57.4.
- MTP still lost with experts in RAM: `--draft 1` 56.4–56.8; `--draft 2`
  48.3–49.3 (acceptance 45–65% at temp 0.7).
- Keys per attention split (`GQA_KEYS_PER_SPLIT`) 64 / 32 / 16: 60.3–60.7 /
  61.3–62.2 / 57.8–64.0, within noise; it stayed 64 (superseded by the
  adaptive split below).

With the context full, just reading the KV every token already costs ~7.5 ms
on the 3060. So 90 tok/s is only reachable with a short conversation.

## KV lent to experts, short-context attention and MTP with experts in RAM (2026-11-10, afternoon)

Same scenario (35B-A3B Q4_K_M, RTX 3060 12 GB, `-c 246000 --kv-type q8_0`).
Result, alternated rounds in the same session, 300 tokens:

| | temp 0.7 | greedy |
|---|---|---|
| before (`AIRUN_NO_KV_LOAN=1 AIRUN_NO_SPLIT_WAVES=1`) | 59.9–62.2 | 59.8–62.0 |
| now | 73.0–73.6 | 73.7–74.7 |
| now + `--draft 1` | 64.8–67.3 | 68.2–69.4 |

With `serve ... --kv-type q8_0 -b 256 -c 262144` and ~2.3k-token prompts
(400 generated, 5 requests): prefill 553 tok/s, generation 61.0 tok/s.

**KV lent to experts** (`kv_loan.rs`, `model::Loan`,
`MappedExperts::{fill, release_below}`):
- The 246k KV reserves ~2.7 GB that a short conversation does not touch. The
  rows not yet reached hold copies of experts from RAM (1013 experts, 2.43
  GiB on the 35B): cache hits 82% → 89%, 56 → 66 tok/s on its own.
- The KV layout is `[head][position][row]`; each lent piece sits at a
  position `p0` of a head and holds while the sequence writes below it. The
  host keeps the expert's original copy, so giving it back is just pointing
  the table back (nothing crosses PCIe).
- `kv_loan_guard(upto)` runs before any KV write: `step_inner`,
  `step_layered`, `chain_launch`, `decode_rows`, `load_state`,
  `spec::verify`, `verify_rows` and, as a safety net, `queue_forward_with`.
  On coming within `margin` (= batch + 1024) rows of the lowest loan, it gives
  back everything below `upto + 2 * margin`.
- Rebalancing fills free loans above `max(n_past, history.len()) + 2 *
  margin` with the hottest experts (one copy, against three for a swap).
  `reset()` frees the rows again.
- Off with `--parallel` > 1 (each slot would need its own), KV in RAM, a
  sparse-attention index, or `AIRUN_NO_KV_LOAN=1`.
- **Exactness:** greedy text identical with and without loans, including the
  give-back in the middle of generation (4964 prompt tokens, 2200 generated,
  `-c 12288`), **as long as** `AIRUN_MOE_OVERLAP=0`. With the two passes
  (resident, then copied), an expert that changes sides changes the float sum
  order and the text may diverge at a near-tie. That is not corruption:
  always compare with `AIRUN_MOE_OVERLAP=0`.

**MTP: the verifies did not rebalance the experts** (`spec.rs`):
`rebalance_experts` only ran in `step()`; with `--draft` the choice of experts
in VRAM froze at the prompt. Now it runs after every `verify` and
`accept_rows`. `--draft 1`: 56 → 60–67 tok/s.

**Short-context decode attention** (`attention_decode_gqa_kernel`,
`gqa_decode_tail<D, KV, GS>`):
- With ~300 tokens there were 5 splits × 2 KV heads = 10 warps on the whole
  GPU, ~90 µs per layer (~1 ms per token, 0.8% of bandwidth). Nsight Compute:
  ~18k instructions per warp at ~9.5 cycles each, half the wait in
  instruction-fetch misses (the unrolled code is walked only once per block).
- The heads loop unrolled `GQA_MAX` = 8 with a run-time `if (h >= G)
  continue`: the body is now instantiated with 1/2/4/8 fixed heads (generic
  for other G and for the multi-warp verify, whose numerics must not depend on
  the path).
- `attention_active_splits` picks 16 to 64 keys per split, the smallest that
  still fits a wave (`wave`). Attention and merge call the same function, so
  they agree. No effect at long context (the splits reach the 256 ceiling).
  `AIRUN_NO_SPLIT_WAVES=1` goes back to fixed: 69–72 against 74–75 tok/s.
- It changes the sum order: Qwen3-4B's greedy text diverges from the fixed
  one at a near-tie (character 501 of 1627); both coherent.
- **Tried and discarded:** spreading a group's 8 query heads over several
  blocks. With the specialized body it came to 72.4–73.3 against 73.6–74.1:
  the block's time is the cold pass through the code and the q8_0 byte-by-byte
  decoding (74% excess sectors), not the heads.

**Expert prefetch before the FFN** (`Prefetch`, `moe_prefetch`,
`moe_stage_plan_pre_kernel`), **optional** with `AIRUN_EXPERT_PREFETCH=1`:
- At the start of the layer, its router applied to the incoming residual
  (normalized with the FFN weight) predicts the experts. The predicted ones in
  RAM start copying while attention runs, into the copy area's first slots; in
  the FFN only what the prediction missed is copied (`first` in
  `moe_stage_copy`).
- It predicts 72% of the copies and cuts the PCIe wait from 2.1 to 0.8 ms per
  token (nsys), but loses: 67–69 against 70 tok/s. The prediction costs ~20
  µs per layer, and the copies beside the attention make the other kernels
  ~2 ms slower per token. Predicting only 4 experts (31% hits, almost no
  waste), it loses the same.

**Where the time goes now** (nsys, without prefetch, ~14 ms per token outside
the profiler):
- compute ~11.7 ms: dense matvecs 5.4 (the 248k × 2048 Q6_K output alone
  ~1.2), experts 2.8, `matvec_tallk` 0.9, attention ~0.5;
- GPU waiting on the expert copy ~2.1 ms;
- the profiler inflated the gap between tokens: `cudaGraphLaunch` 1.2 ms per
  token with tracing. Without it the real gap is ~0.6 ms. Sampling on the CPU
  at temp 0.7 costs only ~0.25 ms more than greedy.

**MTP still loses with experts in RAM:** verifying 2 tokens costs 22 ms
against 14–15 for 1 (each token picks its own 8 experts, so copies and expert
compute almost double), and the head plus commit, 2.7 ms. With 58–66%
acceptance it stays below plain decoding.

**Next levers:**
- the PCIe wait (~2.1 ms): it only hides if the copy gets off the path without
  competing for the GPU. A prefetch only wins if the prediction costs almost
  nothing (for example the next layer's router together with the current
  one's, in a single matvec) and the copies do not slow the attention;
- dequantizing the q8_0 KV row in attention with wide loads, instead of byte
  by byte;
- dense matvecs are already close to bandwidth; the 248k-row output is ~8% of
  the token.

## Automatic dose: `--prefer`, MTP on in `serve`, and a verify kernel that lost (2026-11-12, 5070 Ti + 5060 Ti)

Goal of the session: Qwen3.8-27B NVFP4 as fast as it goes with no flags. The
"300 tok/s on a 3090" numbers going around come from 3-bit weights (TQ3, 11.9
GB against this file's 14.6) plus a separate block drafter (DFlash2, ~5.7
accepted tokens per pass on reasoning prompts); with the model's own MTP head
the best published 3090 figures are 112-117 tok/s. Neither a 3-bit format nor
a drafter exists here yet: those are the two levers left for a big jump.

**What changed.**
- **MTP only with `--draft`, in `run` and `serve` alike** (the user's rule:
  the engine may point the head out, not turn it on). Without the flag the
  loader prints a `note:` with the number of drafts it would have used
  (`DEFAULT_DRAFT`, `MOE_DEFAULT_DRAFT`, `DISK_DEFAULT_DRAFT`) and loads no
  head. `run` used to draft by default; `serve` never did (48 against 108
  tok/s on the 27B with `--draft 4`).
- **`--prefer speed|balanced|context`** (`loader.rs`, `auto_ctx`): between
  GPUs of different speeds the automatic context is bounded by a predicted
  decode cost, `SPLIT_SHARE` (6.5% of a step for each hop between cards) plus
  `SLOW_SHARE` (35% of a layer's share for each time the card's bus is
  narrower). The constants are measured, not derived: the comment on them has
  the table. Bus width alone (2x) predicted 1.56% a layer; measured 0.42-0.7%.
  A slower GPU now joins the automatic context's GPUs (it used to stay out
  when the model fit one card, which left `serve` at 18k tokens).
- Same prompt, `serve`, one request, thinking off, 300 tokens: old flags
  (`--draft 3 --parallel 4 -c 248144`) 89 tok/s; `--draft 4` alone 108-111
  at 80k; with `--prefer speed` 120 at 4.8k; with `--prefer context` 91 at
  728k; with `--parallel 4` 98 alone, 100 summed with 4 requests.

**By context depth** (`benchmarks/2026-11-12`, `-c 131072`, real code, one
run per point; prompt tokens -> prefill / decode with `--draft 4` / decode
without MTP, tok/s): 764 -> 1944 / 74.0 / 41.7; 3.5k -> 3665 / 76.1 / 40.7;
6.9k -> 3684 / 68.6 / 39.4; 14k -> 3341 / 68.5 / 35.9; 28k -> 2738 / 59.1 /
31.4; 61k -> 1859 / 40.4 / 24.6; 112k -> 1260 / 26.8 / 18.4. On code the
drafts are accepted 52-64% of the time, against 82% on the predictable prompt
of the table above: compare speeds on the same prompt only.

**Where a speculative step goes** (5070 Ti alone, `--draft 4`, 33 ms):
verify 26.3 ms (the NVFP4 matmuls 25.2 for 5 rows, against ~20.5 for one),
advance 6.7 ms, of which 4.3 ms are the four reads of the 874 MB output head
for the drafts' argmax. `AIRUN_DRAFT_VOCAB=32768` cuts those (89 -> 102
tok/s on English) but loses acceptance on Portuguese and code: the next step
worth taking is a learned shortlist (rows of the tokens seen, plus a
frequency profile), which keeps the gain without the language dependence.
Longer drafts do not pay: acceptance falls to 48% at 6 and the verify jumps
to 48 ms at 9 rows (the multi-token kernel takes 8).

**A tensor-core verify kernel that lost.** `matvec_nvfp4_mma8_kernel`: int8
`m16n8k16` MMAs straight from global memory (a lane loads the 16 bits that
hold its four codes, the neighbour's half arrives by shuffle, k permuted the
same way in A and B), 64 rows per warp. In `matvec_bench` on the 5070 Ti it
won (5 tokens 66 -> 55 us, 8 tokens 82 -> 58), and in the model it lost
badly: verify 26.5 -> 37.6 ms. Two lessons:
- **`matvec_bench` does not measure memory on the 5070 Ti.** Its 50 MB matrix
  fits the card's L2 (1870 GB/s "of weights" for one token), so it times
  arithmetic only. The model's 13.6 GB come from DRAM, where 16 warps
  striding the K of 64 rows read far worse than two rows per block in order.
  Decide kernels with `run`/`bench` on the model, or make the bench matrix
  larger than L2.
- What the first version paid for was `ue4m3_to_f32` (19 of 66 us: its
  branches and int-to-float) and shared memory (a 32 KiB reduction buffer per
  block comes out of L1). Both are fixable and neither was the problem.

**Also measured.** The fixed cost of a split (verify +1.8 ms, advance +1.1)
does not come from waiting: synchronizing the first GPU from the host before
the handoff changed nothing. With MTP off and one conversation, the 27B fits
the 5060 Ti alone and `auto` picks it over the 5070 Ti, which has a little
less free VRAM (decode 23 tok/s): a model that fits only the slower card
should still be weighed against splitting.
