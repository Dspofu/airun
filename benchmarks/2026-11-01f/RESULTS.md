# 2026-11-01 (evening, later): long prefill ahead of llama.cpp

Same script and conditions as [`../2026-11-01e/`](../2026-11-01e/RESULTS.md):

- **Model:** Qwen3.8-27B NVFP4 split across the two GPUs.
- **Prompt:** 512 batches.
- **Rounds:** 2, alternated between engines, with the best of the two in
  tok/s.
- **Raw output:** `longctx.csv` and `longctx.log`.

| Test | airun before | **airun now** | llama.cpp | now / llama.cpp |
|---|---:|---:|---:|---:|
| 8,192-token prompt | 2348 | **3519** | 3211 | 1.10 |
| 32,768-token prompt | 1853 | **2821** | 2631 | 1.07 |
| 131,072-token prompt | 1012 | **1735** | 1599 | 1.09 |
| generation @ 16 | 33.3 | **31.3** | 30.8 | 1.02 |
| generation @ 32,768 | 28.5 | 27.5 | **27.8** | 0.99 |
| generation @ 131,072 | 20.9 | 19.7 | **21.8** | 0.91 |

## What changed

1. **The GPUs now work at the same time during prefill.** This was the
   biggest gain. The `handoff` between the cards synchronized the source one
   through the host, and `airun bench` read the logits of every prompt chunk.
   Both put the GPUs in series: the 5060 Ti (half the SMs) waited for the
   5070 Ti, and vice versa.
   - Now the hand-off between GPUs is ordered only by CUDA events. GPU 0
     processes chunk c+1 while GPU 1 processes chunk c.
   - On a 4k prompt: 3653 tok/s on one GPU, 2501 split before, 3566 split
     now. llama.cpp does 3514 split.
2. **Prefill attention from 55 to 72.7 TFLOPS** (head_dim 256, measured in
   isolation by `crates/airun-cuda/tests/attention_bench.rs`):
   - **P·V in fp16 per key tile:** sums of at most 32 terms, with the total
     in fp32. Only from 4096 keys, and with the same perplexity: 1.2960
     against 1.2999 at 16k tokens in 8k chunks.
   - **Unmasked path** for fully visible tiles.
   - **Accumulator rescale** only when some row's maximum changes.

## Generation

- **Decode did not change.** Run in isolation, it gives the same as before:
  - @ 16: 33.7 against 33.6 with the old hand-off;
  - @ 32k: 28.9 against 28.8;
  - @ 128k: 20.9 in both modes.
- **The table's numbers come right after heavy prefills.** Those prefills now
  keep both cards busy all the time, so they reach decode hotter. That is a
  hypothesis I did not measure; the clocks at the end of the isolated round
  were 2835 and 2010–2332 MHz.
- **At 128k context llama.cpp still generates ~4% faster** (21.4–21.8 against
  20.9 measured in isolation). It is the next target.

## Correctness

- **selftest:** 275 of 275, with a new case for the fp16 P·V path.
- **Qwen4 Flash-Next:** greedy output of 3 prompts × 80 tokens is still
  identical to the reference. Qwen4 also uses both GPUs and did not lose
  speed: prompt 452–496 and generation 39.1 tok/s, against 449 / 39.4 before,
  with a 1024 batch.
- **Qwen3.8-27B:** "The capital of France is" → " Paris." with `-b 512` and
  `-b 1`.
