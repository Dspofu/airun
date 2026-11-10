# 2026-11-01 (evening, final): ahead of llama.cpp at every context

Same script and conditions as [`../2026-11-01e/`](../2026-11-01e/RESULTS.md)
and [`../2026-11-01f/`](../2026-11-01f/RESULTS.md):

- **Model:** Qwen3.8-27B NVFP4 on the two GPUs.
- **Prompt:** 512 batches.
- **Rounds:** 2, alternated between engines, with the best of the two in
  tok/s.
- **Raw output:** `longctx.csv` and `longctx.log`.

| Test | e (start) | f (prefill) | **g (now)** | llama.cpp | g / llama.cpp |
|---|---:|---:|---:|---:|---:|
| 8,192 prompt | 2348 | 3519 | **3694** | 3496 | 1.06 |
| 32,768 prompt | 1853 | 2821 | **3055** | 2862 | 1.07 |
| 131,072 prompt | 1012 | 1735 | **2428** | 1597 | 1.52 |
| generation @ 16 | 33.3 | 31.3 | **42.1** | 32.7 | 1.29 |
| generation @ 32,768 | 28.5 | 27.5 | **37.4** | 29.1 | 1.29 |
| generation @ 131,072 | 20.9 | 19.7 | **23.5** | 20.4 | 1.15 |

llama.cpp varied between tonight's rounds, for example from 20.4 to 21.8 in
generation @ 128k. Even against its best numbers, airun stays ahead in
everything.

## What changed since `f`

1. **The planner fills the fastest GPU first** (`fill_fastest_first`, in
   `plan.rs`).
   - **Before:** a model that did not fit one card was split in proportion to
     free VRAM. With two 16 GB cards, half on each.
   - **Why it matters:** decode goes through the GPUs one after the other, and
     the 5060 Ti has half the 5070 Ti's bandwidth. Every layer on it cost
     double.
   - **Now:** the 5070 Ti takes 59 of the 64 layers with up to 32k context
     and 43 with 128k (the KV takes the rest). The 5060 Ti takes the rest.
   - **Prefill:** in a pipeline, the step is the slowest card's, which now
     carries little. Hence the +40% at 128k over `f`.
2. **Decode splits in whole waves.** Decode attention splits the cache into up
   to 256 pieces per KV head.
   - **Before:** 4 × 256 = 1024 blocks made 7.1 waves on the 5060 Ti (144
     blocks at a time), and the eighth ran almost empty.
   - **Now:** the number of pieces drops to a multiple of the GPU's real
     concurrency.
   - **Gain:** +2% decode @ 128k (23.6 against 23.1, measured in isolation).
     `AIRUN_NO_SPLIT_WAVES=1` turns it off.

## Correctness

- `cargo test`: the new planner test
  (`an_oversized_model_fills_the_fastest_gpu_first`) fails with the old split.
- selftest: 275 of 275.
- Qwen4 Flash-Next: identical to the reference on 3 prompts × 80 tokens.
- MTP on the 27B with 64k context (53/11 layers): 70 tok/s, 77% of drafts
  accepted, against 116 tok/s on one GPU with 4k context.

## What did not help

Loading the decode tiles with `cp.async`, to put every piece in flight,
dropped bandwidth from 711 to 636 GB/s on the 5070 Ti. Per-tile latency was
not the bottleneck.
