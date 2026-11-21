# Guide for AI agents working on airun

Read this file **before** changing anything. It describes what the project
is, what already works, where the pitfalls are, and how to verify you did
not break anything. It describes the **current** state; see
[§13](#13-keeping-the-docs-clean) before adding to it.

---

## 1. What airun is

An LLM inference engine **written from scratch**, in Rust + CUDA.

**This is not a wrapper.** There is no llama.cpp, ggml, PyTorch, ONNX or
cuBLAS on the execution path. Every CUDA kernel in `crates/airun-cuda/kernels/`
was written for this project. If you catch yourself about to add an external
inference dependency, stop: that was an explicit decision of the user.

Allowed dependencies: utilities (clap, serde, memmap2, half, thiserror,
anyhow, tokio, axum, minijinja). The CUDA Runtime is used only to manage
memory, streams and launch *our* kernels.

---

## 2. Current state

airun is built for the **Qwen family**. Architectures (`general.architecture`):

| Architecture | Models | State |
|---|---|---|
| `qwen35` | Dense hybrid Qwen3.5 / Qwen3.8 (e.g. 27B) | main focus: MTP, vision, `--parallel`, NVFP4 |
| `qwen35moe` | Qwen3.5 / Qwen3.8 MoE (e.g. 35B-A3B) | complete |
| `qwen4exp` | Qwen4 Flash-Next (hyper-connections, PLE, QSA) | complete, no `--parallel`, no MTP head |
| `qwen3next`, `qwen3nextmoe` | Qwen3-Next | same hybrid block |
| `qwen3` | Dense Qwen3 | complete |
| `qwen2`, `llama` | same block layout | works, lightly tested |
| `qwen3moe` | Qwen3-30B-A3B / 235B | **no** (the loader refuses with a message) |

| Component | State |
|---|---|
| GGUF (v1–v3) and safetensors reading | works |
| Quantization decoders | bit-exact against the CPU reference |
| Quantizers (writing) | round-trip tested |
| CUDA kernels | every kernel in the selftest (the FP4 cases only on sm_120) |
| Multi-GPU, including different generations (build `86,120`) | validated |
| VRAM planner | tested, with fallback, automatic context and batch |
| llama-server-format API | 56/56 in `tests/api_conformance.py` |
| Sampling and constrained generation (GBNF, JSON schema) | tested |
| MTP (speculative), also with temperature and in `--parallel` | tested |
| Vision (`mmproj` qwen3vl_merger) | reads text in images like llama.cpp |
| Layers, FFN columns and MoE experts in RAM (CPU AVX2 kernels), disk offload | tested |

**Not yet:** `qwen3moe`, audio, LoRA, rerank, video, Qwen3-VL with deepstack,
a paged KV cache.

---

## 3. Code map

```
crates/
  airun-gguf/          file format — no CUDA, no network I/O
    ggml_type.rs       type table; unknown IDs are NOT an error
    reader.rs          GGUF parser (mmap, everything bounds-checked)
    dequant.rs         CPU decoders — the ORACLE for the GPU kernels
    quantize.rs        encoders (f32 -> blocks)
    safetensors.rs     HF checkpoint reading

  airun-cuda/          GPU
    build.rs           finds CUDA + the host compiler, compiles kernels/*.cu
    kernels/*.cu       THE KERNELS. Hand-written.
    kernels/dequant.cuh  GPU decoders — mirror of dequant.rs
    kernels/sm120a/    FP4 tensor-core kernels (compute_120a only)
    src/ffi.rs         FFI (fixed-width types only)
    src/ops.rs         safe wrappers
    src/nvml.rs        NVML, opened at run time (not a dependency)

  airun-core/          runtime
    config.rs          hyperparameters from the GGUF metadata
    plan.rs            VRAM planner / multi-GPU split
    loader.rs          orchestration, automatic context/batch, fallback ladder
    model.rs           weight upload (VRAM, pinned RAM, file)
    runtime.rs         forward pass
    spec.rs            MTP speculative decoding
    cpu.rs             CPU (AVX2) matmuls for weights kept in RAM
    moe_cpu.rs         MoE experts in RAM computed on the CPU
    expert_*.rs        expert cache, profile, reader, RAM tier
    kv_loan.rs         KV cache rows lent to MoE experts
    disk.rs            disk offload, prefetcher, RAM probes
    vision.rs          the mmproj ViT
    grammar.rs         GBNF and JSON schema
    tokenizer.rs, sample.rs

  airun-cli/           commands (serve.rs, sched.rs for --parallel, chat.rs
                       for the Jinja template, replay.rs, hide.rs,
                       document.rs for text/PDF attachments, ...)
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

Other mirrors that must stay in step: `block_q8a` in `vecdotq.cuh` and
`quantize_q8a_mirror` in the selftest; `encode_q8_0`/`encode_q4_0` in
`quantize.rs` and the GPU KV encoder (compared byte for byte; the `__f*_rn`
intrinsics stop nvcc from contracting into an FMA that rounds differently);
`RopeParams` in `ops.rs` and `airun.cuh`.

---

## 5. How to verify your work

Always, in this order:

```bash
# If Rust or CUDA are not on PATH, set CARGO_HOME/RUSTUP_HOME/CUDA_PATH
# (build.ps1 and build.sh find them on their own; a manual cargo needs the variables).
export AIRUN_CUDA_ARCHS=120     # Windows: $env:AIRUN_CUDA_ARCHS="120"; drop it for the final build

cargo test --release
cargo build --release
./target/release/airun selftest        # every GPU kernel against the CPU reference
./target/release/airun run <small-model.gguf> -p "What is 2+2?" -n 20 --temp 0
./target/release/airun perplexity <model.gguf>   # the metric that catches subtle errors
```

On Windows the binary is `.\target\release\airun.exe`.

If the selftest passes but generation turns to garbage, the bug is in the
*runtime* (order of operations, KV cache format, RoPE), not in the kernels.

**Do not judge correctness by reading the generated text.** A model with a
swapped layout writes fluent English answering something else, and a short
continuation gets lucky — this has already reversed a decision in this
project. Use `airun perplexity`: it scores a known passage and returns loss
and top-1, so any hypothesis becomes a number comparable against a dense
control model.

**Do not drop `AIRUN_CUDA_ARCHS` handling from the final build.** The default
compiles for sm_75/80/86/89/120 and takes several minutes, but it is what
supports RTX 20 through 50.

**The selftest uses synthetic weights in the kernel's own layout.** It never
reads real file bytes or applies a sidecar scale (see pitfall 14), and it
does not exercise kernels with real weights; `crates/airun-core/tests/prefill.rs`
(ignored; `AIRUN_TEST_DENSE_GGUF`/`AIRUN_TEST_MOE_GGUF`) checks that a batched
prefill reaches the same logits as token by token.

---

## 6. How to measure

Every one of these has already produced a wrong conclusion here.

- **Alternate the variants in the same session, two rounds at least.**
  Another program drawing on the GPU (a browser, the chat app) stretches
  kernels by ~8%, and two separate sessions vary by ~15%. The first
  measurement of a session comes out ~8% low (clocks ramping up).
- **Decode is not a single rate.** It drops with context; compare at the same
  depth and on the same prompt (MTP acceptance moves with the text: 82% on a
  predictable prompt, 52–64% on code).
- **MoE: measure with natural text and `--temp 0.7`.** A repeated token, or
  greedy loops, route to the same experts and flatter the cache.
- **Judge by `run`/`bench` on the model, not by nsys or `matvec_bench`.**
  nsys inflates graph launches and short kernels; `matvec_bench`'s matrices
  fit the 5070 Ti's L2, so it measures arithmetic, not memory. Make a bench
  matrix larger than L2 or test in the model.
- **Measure without per-step logging** (`AIRUN_SCHED_TIMING`,
  `AIRUN_SPEC_TIMING`): use the logs to see how a step splits, not its speed.
- **Disk/streamed experts depend on the OS file cache.** The same build gave
  29/15 tok/s with a warm cache and 18/8 cold.
- **Windows power throttling (EcoQoS)** moves a console process without a
  foreground window to E-cores a few seconds after start;
  `cpu::keep_full_speed` turns it off. CPU numbers taken without it are
  wrong.
- **A run right after another that freed a lot of pinned memory** comes out
  slow; repeat it.
- **Greedy text may flip on a near-tie** whenever the sum order changes
  (batch boundaries, MTP verify, CPU vs GPU path, adaptive splits). Equal
  perplexity and coherent text is not a bug. For exact comparisons with MoE
  experts in RAM, set `AIRUN_MOE_OVERLAP=0` (the two-pass order depends on
  residency).
- `bench --random-tokens` builds prompts like `llama-bench`, for
  like-for-like comparisons (not for MoE cache judgements).
  `AIRUN_BENCH_PROGRESS=1` prints a line per run.

---

## 7. Kernel paths

### Matvec / matmul by batch size

- **Decode (n=1)** runs on the exact f32 kernel (bit-identical to the CPU
  reference), except NVFP4 and Q6_K, which take the integer path at every
  size (`KernelType::prefers_q8_decode`): at 4.5–6.5 bits the f32 vector each
  warp rereads weighs more than the weights. `AIRUN_Q6K_F32_DECODE=1` puts
  Q6_K back on f32. The IQ types and Q2_0 take dp4a up to 63 tokens
  (`AIRUN_FOR_EACH_Q8_DECODE_ONLY_TYPE`, `Q8_DECODE_ONLY_MAX`), in passes of
  up to 8. Their f32 kernel was compute-bound on the grid lookups, and a 27B
  IQ2_M in VRAM ran several times slower on it. From 64 tokens the grid
  types go to MMQ (below); IQ4_NL, IQ4_XS and Q2_0 have no `mmq_decode` and
  stay on fp16 tensor cores. The grid types (IQ2_S, IQ3_S, IQ3_XXS) use
  `matvec_iq_kernel` up to 63 tokens:
  - a warp per row (2 rows a warp for one token, 4 for more), so a 5120-wide
    row is 5 groups a lane;
  - the codebook copied into shared memory, with persistent blocks, as many
    as the kernel's occupancy allows (`AIRUN_IQ_BLOCKS_PER_SM` overrides);
  - each group decoded once (`q8decode` to `q8codes`) and spent on every
    token;
  - for 2–4 tokens, the activations staged in shared memory and read as
    16-byte loads (`AIRUN_NO_IQ_STAGE=1` turns this off). Rows wider than
    `IQ_STAGE_GROUPS` (160 groups, a 5120-wide row) are staged chunk by chunk
    (`matvec_iq_chunk_kernel`);
  - three blocks an SM for 2+ tokens (`__launch_bounds__`). At two, set by
    registers and shared memory, a third of the warps were resident and the
    kernel waited on latency.

  The other decode-only types use `matvec_q8_wide_kernel` and
  `matvec_q8_dot_multi_kernel`. `AIRUN_NO_IQ_KERNEL=1` sends the grid types
  there too, and `AIRUN_IQ_F32_DECODE=1` puts the IQ types back on f32.
- **2–8 tokens:** dp4a over quantized activations; `matvec_q8_multi_kernel`
  and `matvec_q8_wide_multi_kernel` decode each group once and spend it on
  every token (the MTP verify lives here).
- **From 9 tokens (`AIRUN_MMQ_MIN`):** int8 tensor cores (MMQ, `mmq.cuh`) on
  sm_80+ for Q8_0/Q4_K/Q5_K/Q6_K/NVFP4, `cp.async` version
  (`matvec_mmq_async_kernel`); from 64 tokens also for IQ2_S, IQ3_XXS (both
  `cp.async`, copied like Q6_K) and IQ3_S (synchronous: its 110-byte blocks
  would need copies past the matrix's end), with the codebook staged in shared
  memory (`mmq_decode_iq`; from global memory it ran at half the speed) and
  gate+up+SwiGLU fused. A 27B IQ2_M prefilled 512 tokens 3.5x faster than on
  WMMA, about llama.cpp's rate. `AIRUN_NO_IQ_MMQ=1`. Other types and older
  GPUs go to fp16 tensor cores (`matvec_wmma_kernel`). While a graph is being captured, batches under
  32 keep the few-token kernels.
- `AIRUN_MATVEC_F32=1` forces the exact path everywhere; `AIRUN_NO_MMQ=1`,
  `AIRUN_NO_MMQ_ASYNC=1` and `AIRUN_NO_WMMA=1` turn each off. Each path has
  selftest cases with a backward error bound.
- Small matrices against a batch have their own kernels: `matvec_tallk_kernel`
  (narrow F32, up to 16 tokens), `matvec_few_rows_kernel` (≤ 4 rows),
  `matvec_f32_rows_kernel` (the router), `matvec_half_*` (BF16/F16).

### NVFP4

- **The block:** 64 values as four runs of 16, each with a UE4M3 scale; 36
  bytes. FP4 nibbles become int8 with `__byte_perm` (`fp4_codes8` in
  `dequant.cuh`): E2M1 values doubled are the integers 0..12, the ½ goes into
  the scale. A `__constant__` table indexed by a different nibble per lane
  serializes (it held the 27B at 5 tok/s; IQ4_NL/IQ4_XS had the same problem,
  `iq4_codes8`).
- **Decode:** `matvec_q8_wide_kernel` (128 threads cover the K of 2 rows).
- **Prefill on Blackwell (sm_120): FP4 tensor cores** from 65 tokens
  (`AIRUN_FP4_MIN`; MMQ is faster up to one 64-token tile and its 8-bit
  activations are closer). `kernels/sm120a/fp4mma.cu`, compiled for
  `compute_120a` (`mma ... kind::mxf4nvf4` only exists there). Weights go in
  as stored (word t of a block is the A fragment register, `d[4]` the scale
  register); activations are quantized to the same layout, SwiGLU inside the
  `down` input's quantization (`quantize_glu_nvfp4`, a vectorized and a
  scalar form that the selftest compares). `AIRUN_NO_FP4=1` goes back to MMQ.
- **Without FP4 tensor cores** (RTX 20/30/40): int8 MMQ for prefill, dp4a for
  decode, same result. `fp4_supported()` decides per GPU, so a mixed PC uses
  each card's own path. On a 5060 Ti the int8 path prefills at ~45% of FP4,
  the same as the model's Q4_K: no format-specific bottleneck.
- `0x7F` scales (zero for ggml, NaN for the hardware) are zeroed at load.

### MMQ details

- **Stream-K** (`MmqStreamK`) when the tiles do not close whole waves: the
  head block of a tile waits on its tails' flags and adds partials in block
  order (deterministic); limited to one resident wave, so it does not
  deadlock. **Only from 32 tokens**: on smaller batches it brought a 5070 Ti
  down ("GPU is lost", TDR) every few runs, cause not found. If a card is lost
  again, try `AIRUN_NO_MMQ_STREAMK=1` first.
- **Do not capture MMQ stream-K in a CUDA graph:** the flags' `epoch` would
  freeze (`mmq_streamk_plan` refuses on a capturing stream).
- Prefill gate+up+SwiGLU is one launch (`GLU=true`), stream-K included.
- Q5_K/Q6_K superblocks are copied once for two chunks (`BLOCK2`), with two
  superblock buffers up to 64-token tiles (`mmq_async_sb2`). Q6_K's rows are
  not 16-byte multiples: each row's copies are aligned on their own
  (`ROW_SKEW`), and the launch needs a 16-byte-aligned matrix
  (`mmq_async_fits`).
- Q8_0 decode loads all of a warp's rows before using any
  (`q8_0_load`/`q8_0_decode`); otherwise the compiler serializes rows.

### Attention

- Prefill of 16+ tokens: flash attention on fp16 tensor cores
  (`AIRUN_NO_FLASH=1`). P·V in fp16 per key tile only when the batch sees
  4096+ keys (`AIRUN_ATTN_PV_F32=1` turns it off; on short prompts it flipped
  a near-tie).
- Decode on sm_80+ with up to 8 query heads per KV head:
  `attention_decode_mma_kernel`, one warp per (KV head, split). The group's
  heads are the rows of fp16 `mma.m16n8k16` (rows past the group are zero),
  for both S = QKᵀ and O += PV. P goes from the S accumulators straight to
  the A fragments, and V's B fragments come from `ldmatrix.trans`. Tiles are
  16 keys (`MMA_DECODE_TILE`). Quantized rows that are whole 16-byte units are
  copied raw with 16-byte loads, with the next tile's K and V in flight while
  this one computes, and decoded in shared memory (`kv_piece`). Q and P are
  fp16, like the prefill flash kernel. `AIRUN_NO_MMA_DECODE=1` turns it off.
  At long context with a quantized cache it is several times faster than the
  scalar kernel, which is bound by FP32 math and per-element decoding.
- Decode otherwise (older GPUs, bigger groups, QSA, verify):
  `attention_decode_gqa_kernel`, one kernel per (KV head, split), reading the
  cache once for the whole GQA group, with the body instantiated for 1/2/4/8
  heads. Splits adapt to fill whole waves (16–64 keys per split, up to 256
  splits; `AIRUN_NO_SPLIT_WAVES=1`). With an f16 cache it reads K/V at ~80% of
  bandwidth. With quantized caches it is limited by decoding: q4_0 tiles are
  staged raw (`gqa_scalar_raw`), q8_0 tiles go piece by piece.
- `cargo test --release -p airun-cuda --test attention_bench decode_attention_moe_shape -- --ignored --nocapture`
  times decode attention in the 35B-A3B's shape for every KV format.
- MTP verify and batched verify: `attention_split_multi`, one warp per query
  sharing K/V tiles. Its split boundaries are **fixed**
  (`verify_split_keys`), not position-dependent, so a token gives the same
  result as query 0 of a lone verify or query 2 of a batched one
  (`--parallel` text stays identical). `AIRUN_NO_SPLIT_VERIFY=1`.
- Quantized KV cache (`--kv-type f16|q8_0|q4_0`): GGUF blocks per head row,
  decoded by the same `dequant`.
- QSA sparse attention (qwen4exp, `kernels/qsa.cu`, `Runtime::qsa_select`):
  layers with `compress_ratio` r attend to `top_k + r - 1` cells chosen by an
  indexer (store → pool → score → radix select, ties to the lower block). With
  context ≤ 2051 it is identical to dense; `AIRUN_QSA=off` forces dense.

### Delta-net (Qwen3.5/3.8 and Qwen3-Next hybrids)

- Prefill of 64+ tokens with 128-wide heads: chunked WY form on tensor cores
  (`kernels/delta_chunk.cu`, bf16 products, fp32 state; +8% prefill).
  `AIRUN_NO_DELTA_CHUNK=1`. Decode and the MTP verify stay on the recurrence.
- V-head → K-head mapping is `kh = vh % n_k_heads` (ggml's repeat), and
  `rmsnorm_gated` is norm → weight → gate. Both were wrong once and masked
  each other (see pitfall 15).
- `ssm_a` already comes as `-exp(A_log)`; the q/k/v buffers need
  `group_count * state_size` in the SSM layers.
- The planner allocates `cfg.n_run_layers()`: the last block is the MTP head
  and is not a model layer.

### Decode pipeline and graphs

- Greedy decode is a pipeline (`ChainGraph`): the graph reads token and
  position from the GPU, the argmax writes the next ones, and the host queues
  step i+1 before reading step i. Not on hybrids (recurrent state cannot undo
  a discarded step).
- One graph per shard; the residual moves between GPUs outside the graphs
  (`handoff`, ordered by events only). **Never synchronize a stream through
  the host between shards**: that puts the GPUs in series. Since the host no
  longer stops between prompt chunks, whoever fills a pinned host buffer must
  wait for the previous upload of it (PLE has the `uploaded` event).
- Qwen4's decode graph is cut in pieces (`decode_pieces`,
  `AIRUN_GRAPH_LAYERS`) so the host prepares PLE rows while the GPU runs.

### Kernel pitfalls

- A register array indexed in a loop with a run-time bound goes to local
  memory. Use `#pragma unroll` with the constant bound and `if (h >= G)
  continue`; check `STACK:` in `cuobjdump --dump-resource-usage` (must be 0).
- `(const half2*)&raw` over a register variable also sends it to local
  memory. Extract with bit operations.
- Without performance counters, measure with `clock64()` inside the kernel
  and one block's `printf`.
- **When tuning tiles, the sizes appear in the template AND in the grid
  math.** Sweeping one computes a fraction of the output and looks like a huge
  gain — `airun bench` does not validate outputs. Change the defines, never
  the instantiations, and run the selftest at every measured point.
- Nsight Compute 2022.4 does not profile sm_120; nsys 2026.x does. Decode
  shows kernel by kernel only with `AIRUN_NO_GRAPH=1
  AIRUN_NO_DECODE_PIPELINE=1` (or `--cuda-graph-trace=node`).

---

## 8. Memory: planner, multi-GPU, RAM and disk

### Planner (`plan.rs`, `loader.rs`)

- If the model fits one GPU, use one (these GPUs have no P2P; copies between
  them go through RAM). Unequal GPUs: fill the fastest first
  (`fill_fastest_first`); twin GPUs (perf within 10%): split evenly, which
  nearly doubles prefill.
- Between unequal GPUs with an untied output, `fill_fastest_last` puts the
  OUTPUT head and last layers on the fastest card: the head is read whole
  every token and every MTP draft, the embedding only by row. A model that
  fits the fastest GPU only at a reduced batch gives the embedding and a layer
  or two to the slower card (`edge`) and keeps the batch.
- **Automatic context** (`auto_ctx`): bounded by a predicted decode cost
  between unequal GPUs (`--prefer speed|balanced|context`; `SPLIT_SHARE`,
  `SLOW_SHARE` are measured constants, the table is in the code). It prefers a
  context up to 1/4 smaller that keeps the head on the fastest GPU, with a
  `note:`.
- **Automatic batch** (`auto_batch`): 256 or 128 instead of 512 when the
  smaller scratch moves enough weight off the CPU. An explicit `-b` is
  respected.
- **Fallback ladder:** with an explicit `-c` that does not fit: everything in
  VRAM → the KV of the last attention layers in RAM → all KV in RAM plus
  weights (`spill_to_ram`). The KV goes first because weights are read whole
  every pass, the cache only where filled.
- **Buffers allocated after the plan must be in the plan.** MTP scratch, the
  delta-net stash (on every GPU), batch logits and per-slot heads are counted
  (`Config::spec_rows`, `spec_stash_bytes`, `spec_scratch_bytes`) and
  allocated at maximum size in `Runtime::reserve_buffers`, so a failure goes
  to the next rung. Pushing a GPU past its budget made WDDM page VRAM (steps
  3–6x slower, suddenly). If it reappears, try `--vram-reserve 900`: if that
  fixes it, something is allocated outside the plan.
- **KV in RAM** needs an `alloc` that allows RAM, no MoE, and cache rows that
  are multiples of 4 bytes. A "device" buffer can then be mapped RAM
  (`DeviceBuffer::in_ram`), which is why memory copies use
  `cudaMemcpyDefault`.
- Not counted by the plan: the CUDA context (~140 MiB), the prompt kernels'
  lazy scratch, the turn snapshot. `split_ffn` leaves `SPLIT_SLACK` on top of
  `--vram-reserve`. cudaMalloc rounds large allocations to 2 MiB, so a layer
  is one allocation (`LayerArena`).
- `AIRUN_SHOW_PLAN=1` prints the plan from any command.

### Weights in RAM

- `--alloc auto|gpu|memory`, `--main-gpu`, `--hide-gpu` (`hide.rs`, sets
  `CUDA_VISIBLE_DEVICES` by UUID before CUDA starts).
- **Whole layers in RAM** run on the CPU (`cpu.rs`, AVX2 kernels for NVFP4,
  Q4_K/Q5_K/Q6_K/Q8_0: decode in registers, int8 `maddubs`, the token count
  as a compile-time constant, software prefetch) for passes up to
  `AIRUN_CPU_MAX_TOKENS` (64); larger prompt chunks copy the layer to VRAM and
  run it layer by layer over all chunks (`layered_take`). `AIRUN_RAM_ON_GPU=1`
  forces the copy. CUDA graphs are off while the CPU is in use.
- **FFN columns in RAM** (`plan::split_ffn`): on one GPU with dense
  feed-forwards, every layer stays in VRAM except the last N intermediate
  columns, computed on the CPU in parallel with the GPU part.
  `AIRUN_NO_FFN_SPLIT=1`. Up to 3/4 of the feed-forward can go to RAM
  (`AIRUN_FFN_SPLIT_MAX`, a fraction). The old cap of 1/2 left a 27B Q4_K_M
  on a 12 GB card with whole layers in RAM, which was slower. Past that, the
  CPU's share reads at about what pinned RAM gives, so a model that needs
  most of its weights in RAM is bound by RAM bandwidth, not by a kernel.
- The output head goes in as Q8_0 when the weights do not fit
  (`AIRUN_NO_HEAD_Q8=1`); hyper-connection matrices are converted to Q8_0 at
  load (`AIRUN_HC_Q8=0`). The vision encoder is uploaded per image when it
  does not fit beside the model (`AIRUN_VISION_RESIDENT=1`).
- **Disk offload** (`--disk-offload`): VRAM → RAM → disk. Layers that do not
  fit free RAM (minus `--ram-reserve`) stay in the mmap; a prefetcher with
  `PrefetchVirtualMemory`/`madvise` reads the next layers, already at the
  SSD's ceiling. If the MoE experts do not fit the budget either, all of them
  are streamed from the file (no half-mapped state). With layers on disk, MTP defaults to 6 drafts
  (`DISK_DEFAULT_DRAFT`): more tokens per pass is what helps.
- On Windows, `GgufFile::release` takes uploaded pages out of the working set
  during load; without it RAM rose to the model's size. Every VRAM allocation
  still counts against the commit limit on WDDM: a small page file can fail a
  load with free RAM.

---

## 9. MoE

- **Experts in mapped RAM** (default when the MoE does not fit;
  `plan_experts`): `slots` experts per layer in VRAM, the rest pinned and
  mapped, found through address tables (`moe_*_tab`) so decode fits a graph.
  Decode copies the routed RAM experts into a VRAM staging area on a side
  stream (few blocks: `AIRUN_STAGE_BLOCKS`/`AIRUN_STAGE_STREAMS`; many blocks
  stuck on PCIe take the GPU from the resident kernels).
- **Experts in RAM on the CPU** (`moe_cpu.rs`): the GPU posts the missed
  experts to a mailbox in mapped memory, a spinning leader runs them on the
  CPU pool while the GPU computes the resident ones. The GPU bumps the
  mailbox counter with a plain volatile increment: atomics on host-mapped
  memory need PCIe atomics. If the CPU answers late
  (50 ms), the GPU takes them back over PCIe; the generation must not abort.
  `AIRUN_MOE_CPU=0` turns it off.
- **Adaptive residency:** `rebalance_experts` every 8 steps (also after every
  MTP verify), swaps batched (`SwapBatch`; it counts the shared stage area in
  **bytes**, because expert sizes differ by layer — counting in experts wrote
  garbage). A GPU with a host link below 8 GB/s gets only whole layers with
  every expert in VRAM. A routing profile (`expert_profile.rs`) seeds residency at load.
- **KV lent to experts** (`kv_loan.rs`): KV rows a short conversation has not
  reached hold copies of hot experts, given back before any KV write reaches
  them (`kv_loan_guard`). Off with `--parallel` > 1, KV in RAM, QSA, or
  `AIRUN_NO_KV_LOAN=1`.
- **Streamed from the file** (`--expert-cache N`, `AIRUN_EXPERTS=stream`, or
  forced by disk offload): parallel positional reads (through mmap they
  arrived at queue-depth 1). Batches of 16+ run expert-major in waves of as
  many experts as the slots hold, so each missing expert is read once per
  batch. `crates/airun-core/tests/offload.rs` (ignored,
  `AIRUN_TEST_MOE_GGUF`) requires logits identical to the resident model. A
  pinned RAM tier (`expert_ram.rs`) is turned on only when it covers
  85% of the missing experts (`EXPERT_RAM_COVER`); below that it took RAM
  from the file cache and lost. The tier keeps experts by routing count, not
  recency: a prompt batch walks a layer's experts in order, so LRU evicts
  exactly what the next batch asks for first. The rule's "on" side is not
  measured yet. `AIRUN_EXPERT_RAM=<MiB>` forces it.
- **Kernels:** decode on the integer path for up to 4 tokens (all K-quants,
  IQ types and Q2_0; `AIRUN_MOE_F32=1`); grouped MMQ from 8 tokens for experts
  in VRAM (`airun_moe_mmq`, `AIRUN_NO_MOE_MMQ=1`); grouped fp16 MMA kernels
  from 16 tokens otherwise (`AIRUN_MOE_MMA_MIN`).
- **Refused at load:** gate and up experts of different types (every MoE
  kernel reads both with the gate's type), and NVFP4 expert stacks with one
  scale per expert (vllm/ModelOpt `.scale` F32[n_experts]). Both loaded
  before and wrote garbage (Qwen3.6-35B-A3B-NVFP4-MTP-HQ has both).
- **MTP with experts in RAM loses** (each verified token wakes its own
  experts), so MoE drafts by default only when the model fits whole
  (`MOE_DEFAULT_DRAFT` = 2).
- With experts in RAM and batches of 2–15, the text is not deterministic
  across runs (sum order depends on residency); `AIRUN_MOE_OVERLAP=0` makes
  it so.

---

## 10. MTP and serving

### MTP (`spec.rs`)

- **MTP only with `--draft`, in `run` and `serve` alike** — the user's rule:
  the engine may suggest the head, not turn it on. Without the flag the loader
  prints a `note:` with the drafts it would use (`DEFAULT_DRAFT` = 4 dense,
  `MOE_DEFAULT_DRAFT`, `DISK_DEFAULT_DRAFT`). When `--draft` is given but
  cannot speculate, a `note:` says why.
- The head is block `n_run_layers()` (`block_count` counts it), or a separate
  `mtp-*.gguf` (`--mtp`). The verify runs the delta-net without writing state
  (`DeltaStash`) and `commit` pushes only the accepted tokens
  (`test_delta_defer` requires bit equality). The verify and the advance are
  CUDA graphs, one per GPU when split.
- **With temperature** (`Sampler::sample_draft`): the draft is accepted with
  the probability the sampling chain gives it, else the token comes from the
  distribution without it. Mirostat has no MTP.
- The text follows normal decoding but is not guaranteed identical (near-ties
  in the multi-token kernels).
- The draft's argmax covers the first 65536 rows of the head
  (`DEFAULT_DRAFT_VOCAB` in spec.rs). On a 248k vocabulary every draft read
  the whole head, about a fifth of a 27B IQ2_M verify step. 64k rows kept the
  acceptance on English code and Portuguese prose, where 16k–48k lost it on
  Portuguese. `AIRUN_DRAFT_VOCAB=K` sets it, and 0 means the whole
  vocabulary. A learned shortlist is the open lever.

### `serve`

- **`--parallel N`** (`sched.rs`): slots with their own KV/delta-net state;
  `decode_rows` advances one token per slot in one pass (batched matmuls, the
  stateful kernels per row via a pointer table), captured as graphs
  (`MixedGraphs`). A lone request runs the normal path ("solo",
  `AIRUN_NO_PARALLEL_SOLO=1`), so `--parallel` costs nothing alone. MTP works
  in the batched steps (`spec_step`, head per slot). Models with QSA,
  hyper-connections or PLE refuse `--parallel` > 1. Overlapping requests'
  greedy text may differ from one at a time (sum order).
- **Turn cache on hybrids** (`TurnSnapshot`): the template rewrites the last
  answer's `<think>`, so the prefill stops before the prompt's last
  `<|im_start|>`, snapshots the recurrent state and restores it next turn
  (`AIRUN_NO_STATE_SNAPSHOT=1`). Tokenizing happens outside the runtime lock.
- **Images:** sentinel ids from the embeddings' hash, so the next turn reuses
  them. `<__media__>` is a slot only in `run --image` and `/completion` with
  `multimodal_data`; elsewhere a per-process random marker is used and the
  literal text stays text. With an image, MTP is off and `save_state`
  refuses the chat.
- **Stream replay** (`replay.rs`): requests with `X-Conversation-Id` keep
  generating without the client.

### HTTP API in llama-server's format

Routes, request fields and responses follow `llama-server`'s, so clients
built for it work unchanged:
- **Health and metadata:** `/health`, `/v1/health`, `/props` (with `details`),
  `/models`, `/v1/models`, `/slots`, `/metrics` (501).
- **Hardware (airun's own):** `/hardware`, from NVML (VRAM via
  `nvmlDeviceGetMemoryInfo_v2`, matching nvidia-smi); cached 0.7 s because a
  reading costs ~250 ms on two GPUs.
- **Generation:** `/completion(s)`, `/infill`, `/v1/completions`,
  `/(v1/)chat/completions`, `/(v1/)responses` (OpenAI Responses),
  `/v1/messages` (Anthropic), all streaming; token counting endpoints;
  `/tokenize`, `/detokenize`, `/apply-template`.
- **Documents:** `document.rs` turns text files and PDFs (`pdf-extract`, a
  utility, behind `catch_unwind`) into text; pages with no text give their
  largest image (lopdf; JPEG passed through, CCITT via `fax`, Flate/raw in the
  common color spaces) as a 1280 px JPEG. `attach_files` in serve.rs swaps
  `file` / `input_file` / `document` parts for a `<file name>` text part plus
  image parts before images are handled, and `/extract` serves the web UI's
  attachments (the page stores the text and page images, not the file).
- **Vision VRAM on Windows:** WDDM's reported free memory counts what other
  programs hold, so with `--mmproj` the loader plans from NVML's free memory
  and `vision_reserve` adds a margin for the CUDA context; forcing the encoder
  past the real free memory made Windows page the model (3x slower decode).
- **Embeddings** (`--embeddings`, `--pooling`), **slots** save/restore
  (`--slot-save-path`; on a hybrid the only way to resume a long prompt), and
  **controlled reasoning** (budgets, `reasoning_control` +
  `POST /v1/chat/completions/control`; both turn off GPU argmax and MTP).
- **Unsupported** endpoints answer like llama-server without the feature
  (rerank, audio, `POST /props`, `/lora-adapters`), and read the body first
  (otherwise the client sees a reset instead of the 501).
- **Template:** the GGUF's own Jinja template (`chat.rs`, minijinja with
  Python string methods), like `--jinja`. The response is split into
  `reasoning_content`, `content` and `tool_calls` (Qwen3.5+ XML or JSON).
  Responses and Messages are converted to the chat form and back.
- **Sampling:** llama.cpp's full chain (penalties → DRY → top-n-sigma →
  top-k → typical → top-p → min-p → XTC → temperature; mirostat 1/2).
  `ignore_eos` is a -inf `logit_bias`. Probabilities come from the raw
  logits' softmax, before the samplers.
- **Grammar** (`grammar.rs`): full GBNF and a llama.cpp-style JSON schema
  converter (property order kept: `serde_json` uses `preserve_order`). With
  `<think>` open, the grammar waits for `</think>`.
- **Default difference:** OpenAI requests without `repeat_penalty` use
  `--repeat-penalty` (1.1); llama-server uses 1.0. The native `/completion`
  uses 1.0.
- **Verification:** `tests/api_conformance.py` compares two servers key by
  key and by SSE event names.

---

## 11. Pitfalls that have already cost time

1. **CUDA ordinal ≠ nvidia-smi index.** CUDA orders by speed by default.
   Always show the PCI bus id next to the ordinal; nsys may label GPUs
   swapped, check by bandwidth.
2. **No P2P on consumer GPUs** (`cudaDeviceCanAccessPeer` = 0): copies between
   GPUs go through RAM, and splitting costs decode. Splitting by free VRAM
   once put half the 27B on the slower card (-28% decode).
3. **CUDA 13 removed fields from `cudaDeviceProp`.** Never bind that struct
   from Rust — use the shim in `kernels/device.cu`.
4. **`gen` is a reserved keyword in edition 2024.**
5. **Tied embeddings** (Qwen3-1.7B has no `output.weight`): a multi-GPU split
   mirrors the table; the planner accounts for it.
6. **Relative error is the wrong metric for a dot product.** The selftest
   uses *backward* error (bound ∝ `sum|w*x|`). Do not loosen the tolerance.
7. **Test data must be physically plausible.** Random bytes in MXFP4's E8M0
   overflow f32; random f16 scales on the CPU hit subnormal arithmetic (18x a
   normal FMA; the pool threads flush denormals).
8. **A GPU-vs-CPU test proves both sides agree, not that they are right.**
   `rmsnorm_gated`'s "reference" repeated the wrong gate order. When a kernel
   passes and the model is bad, suspect the reference.
9. **A token is not a character.** Byte-level BPE splits accented letters;
   `from_utf8_lossy` on half a character destroys the bytes. All streaming
   goes through `StreamDecoder`.
10. **NVFP4 has an absolute magnitude floor.** A run of 16 weights whose peak
    is below ~5.9e-3 encodes as zeros. That is the format; a round-trip test
    with tiny data "fails" for the wrong reason.
11. **The repeat penalty applies once per distinct token** in the window
    (modern llama.cpp), not once per occurrence.
12. **`llama-perplexity -f` on Windows turns CRLF into LF**; airun reads the
    bytes. Normalize to LF before comparing. With the same text, `--chunk N`
    reproduces `llama-perplexity -c N`.
13. **YaRN is not linear scaling**: only the slow dimensions interpolate, and
    cos/sin scale by `1 + 0.1 ln(factor)`.
14. **NVFP4 exists in two conventions.** airun's blocks are self-contained;
    vllm/NVIDIA files add a `<base>.scale` F32[1] sidecar (≈1e-4) that the
    loader applies in `GpuTensor::global_scale` (it turns the GLU fusions off:
    SiLU is not scale-linear). Without it the model gives NaN while the
    selftest passes. Diagnose by dequantizing real bytes with and without the
    scale and checking the weight magnitude.
15. **An A/B decision taken with another bug present is not reliable.** The
    delta-net gate order "won" only because the head mapping was also wrong.
    When fixing a layout, redo the earlier layout decisions.
16. **`-b 1` prefill** goes through the decode graph; each replay must wait
    for the previous one before rewriting the pinned token/position slots.
17. **A failed scratch cudaMalloc leaves the error** for the next launch check;
    clear it when falling back.
18. **`cargo test <filter>` runs matching tests in parallel**: two benchmarks
    at once measure nothing.
19. **`pkill -f "airun serve"` kills the shell running it**; use `pkill -x
    airun`.

---

## 12. Tried and discarded

Measured; do not repeat without a new reason.

- L2 prefetch on a second stream; process GPU priority
  (`D3DKMTSetProcessSchedulingPriorityClass`); 1–2 rows per warp in the decode
  GLU; a double K/V buffer in decode attention; `cp.async` tiles in decode
  attention; grouping GQA heads or spreading them over several blocks.
- SwiGLU in the epilogue of the `up` GEMM (slower than inside the FP4
  quantizer). FP4 matmul with 2-block stages and 4 buffers (-26%).
- A tensor-core int8 verify kernel (`matvec_nvfp4_mma8_kernel`): won in
  `matvec_bench` (L2-resident), lost in the model (verify 26.5 → 37.6 ms).
- Reorganized delta-net recurrences (tiles of 4/16 tokens, one column per
  warp, 4 slices of 32).
- Expert prefetch before the FFN (`AIRUN_EXPERT_PREFETCH=1`): predicts 72% of
  copies but the copies beside attention slow the other kernels.
- A Q8_0 copy of a Q5_K output head for MTP (more bytes, not fewer).
- Longer drafts than 4–5 (acceptance falls, the multi-token kernel takes 8).
- MTP with MoE experts in RAM.
- A raised CPU spin wait (`AIRUN_CPU_SPIN_US`): spinning threads starve the
  thread driving the GPU.
- Unpinned memory for the expert RAM tier (page file); VRAM expert slots
  evicted by count instead of recency; more VRAM slots via a smaller reserve.
- Rebalancing less often (hit rate falls more than swaps cost).
- Layer-by-layer prefill with layers only in pinned RAM (loses the GPU
  overlap; kept only with layers on disk or a single GPU).

**Open levers:** a 3-bit weight format and a block drafter (the two big
jumps); per-expert NVFP4 scales and mixed gate/up types in the MoE kernels
(the up and down scales fold into the routing weights, the gate's sits inside
the SiLU); the PCIe wait for experts in RAM; wide-load q8_0 KV dequant in
attention; Q6_K's split k16 MMAs; automatic `-b` for streamed experts;
admission of several prefills at once in `--parallel`; a learned draft
vocabulary; the CPU-vs-copy choice for RAM layers ignores the weight type and
the link's bandwidth; a model that fits only the slower GPU should be weighed
against splitting.

---

## 13. Keeping the docs clean

This file, `README.md`, `CONTRIBUTING.md` and `docs/FLAGS.md` describe the
project **as it is now**. They are not a lab notebook.

- **Update in place.** When something changes, edit the section that
  describes it; do not append a dated section that contradicts an older one.
  If a statement is no longer true, delete it.
- **No session logs.** Before/after tables, per-run numbers, "today I tried",
  and descriptions of one machine (its PCIe slot, its other programs) do not
  belong here. Keep the result and, if it matters, one number.
- **Failed experiments go to §12** as one line with the reason, not as a
  section.
- **Reference only files that are in the repository.** `benchmarks/` is
  git-ignored: do not point readers at it.
- **Tasks and progress go in [Tasks.md](Tasks.md)**, the one tracking file
  that is committed. Add or move an entry when you start (`In progress`),
  finish (`Fixed`) or find a problem you leave open (`Warn`), in the format at
  the top of that file; `pending` as the hash until the commit exists. Do not
  delete entries: the maintainer prunes it. A known open problem goes there,
  not in a new section here.
- **Any other file you create to track or monitor something** (notes, logs,
  benchmark output, profiler captures, scratch scripts) goes in a git-ignored
  place: `benchmarks/`, `local/`, or any `*.local.*` name. If you need a new
  location, add it to `.gitignore` in the same change. Never commit them.

---

## 14. Style

- Comments explain **why**, not what. If a comment paraphrases the code,
  delete it.
- Every error message must tell the user **what to do**. Compare: "failed to
  load model" (bad) vs. the message in `error.rs` that says how much VRAM was
  missing and which flags to change (good).
- Every `unsafe` block needs a `// SAFETY:` comment justifying it.
- Tests check behavior, not implementation. Prefer tests that would fail if
  the logic were wrong, not just if it changed.
- Code, comments and documentation are in English.
