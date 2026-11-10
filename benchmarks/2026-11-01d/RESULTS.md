# 2026-11-01 (evening): optimized prefill against Strata 0.1.55

`qwen4.ps1 -Batch N`: the three engines with the same prompt chunk size
(airun `-b N`, Strata `--prefill N`), 792-token prompt and 128 greedy tokens,
2 rounds per size. Values in tok/s.

| Chunk | Round | airun | Strata 0.1.55 | llama.cpp |
|---|---|---|---|---|
| 1024 | 1 | **447.0 / 39.2** | 414.4 / 34.1 | 35.9 / 9.9 |
| 1024 | 2 | **449.3 / 39.4** | 426.6 / 33.9 | 35.0 / 9.9 |
| 512 | 1 | **404.5 / 40.7** | 315.6 / 34.4 | 34.8 / 9.9 |
| 512 | 2 | **417.7 / 40.6** | 303.1 / 34.4 | 35.6 / 9.9 |

Values are prompt / generation.

- **1024 chunk:** prompt +7%, generation +16%.
- **512 chunk:** prompt +33%, generation +18%.
- **Quality:** greedy output matches the reference on 3 prompts × 80 tokens;
  perplexity (2048 chunks) 5.5145.

What changed since `../2026-11-01c/` (290 tok/s prompt):
- **BF16 matmul:** a smaller tile, for projections with few rows and long K,
  fills the GPU.
- **MoE on tensor cores (v2):** fp16 activations once per layer, read
  straight from L2. K=128 slices with all 128 threads decoding, and `h` in
  fp16.
- **Experts in RAM:** row copy with 16-byte loads, four in flight per thread.
  These experts' step (PCIe-bound) runs on a stream parallel to the VRAM
  experts' (compute-bound).
- **PLE:** a single `PrefetchVirtualMemory` asks Windows for every page of the
  rows before the gather (456 → 130 ms over 12,672 rows). More page-fault
  threads only fought over the working-set lock, and unbuffered reads on a
  SATA SSD yielded ~7k rows per second.
