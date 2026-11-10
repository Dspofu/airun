# 2026-11-01 (evening): long prompt and large context, Qwen3.8-27B NVFP4

> **Superseded by [`../2026-11-01f/`](../2026-11-01f/RESULTS.md).** The
> diagnosis below was wrong. The main cause was the split running the GPUs in
> series; attention was compute-bound, not bound by re-reading K/V. This
> folder's measurement stays as the record of the before.

`longctx.ps1`: airun against llama.cpp b11760, 2 alternated rounds, best of
the two, in tok/s. Raw output in `longctx.csv` and `longctx.log`.

Machine: RTX 5070 Ti 16 GB (PCIe 4.0 x16) + RTX 5060 Ti 16 GB (PCIe 3.0 x4),
Ryzen 7 5700X3D, 64 GB of RAM, Windows 11, no P2P.

- **GPUs:** with 32k or more context the 27B does not fit one card, so both
  engines split the layers between the two (layer split).
- **Prompt batch:** 512 on both.
- **llama.cpp:** `llama-bench` with flash attention.
- **airun:** `airun bench`.
- **MTP:** neither uses it (`llama-bench` has no MTP).

| Test | airun | llama.cpp | airun / llama.cpp |
|---|---:|---:|---:|
| 8,192-token prompt | 2348 | **3276** | 0.72 |
| 32,768-token prompt | 1853 | **2835** | 0.65 |
| 131,072-token prompt | 1012 | **1598** | 0.63 |
| generation @ 16 tokens of context | **33.3** | 31.4 | 1.06 |
| generation @ 32,768 | 28.5 | **29.1** | 0.98 |
| generation @ 131,072 | 20.9 | **21.4** | 0.98 |

- **Generation:** a tie. Ahead with short context, 2% behind with full
  context.
- **Long prompt:** the current weakness, and the gap grows with context. On
  short prompts airun wins (3797 against 3472 at 512 tokens on one GPU,
  `../2026-11-01/`). The rest of the forward is already faster, so what loses
  is prefill attention, whose cost grows with the cache.
  - At 128k, attention costs ~98 s of airun's 130 s, against ~48 s in
    llama.cpp. The estimate subtracts the matmul cost at 3700 tok/s.
- **Why attention loses** (`flash2_attention_kernel` with head_dim 256):
  - **K/V read 6 times:** each of the 6 query heads sharing a KV head reads
    the whole cache again (~134 MB of K/V per block at 128k). llama.cpp groups
    the GQA heads and reads once.
  - **32-key tiles.**
  - **Low occupancy:** one 128-query block per SM, 96 blocks for 70 SMs. That
    is a wave and a half, the second half empty.
- **Next step:** queries of a group's 6 heads in the same block and the KV
  range split across blocks when the context is long.
