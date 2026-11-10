# Benchmarks 2026-09-26 — RTX 3060 12 GB, i5-10400F, 16 GB DDR4-2666, NVMe (E:)

PCIe 3.0 x16. Built with `AIRUN_CUDA_ARCHS=86`. llama.cpp in `E:\llamacpp`.

## Qwen3-4B Q8_0 (dense) — final scoreboard, 2026-09-27

Alternated in the same session, `scoreboard.ps1 -Rounds 3`, best run of each
(interference on this machine only adds time, so the fastest is the robust
estimate):

| Test | airun | llama.cpp | airun / llama |
|---|---|---|---|
| prefill 128 | 3535 | 3118 | **1.13** |
| prefill 512 | 3979 | 3829 | **1.04** |
| decode @ 16 | 70.3 | 69.1 | 1.02 |
| decode @ 3072 | 63.7 | 62.9 | 1.01 |

Decode is tied within noise (±2%); prefill wins comfortably. The absolute
numbers in this table come from a window without interference: with another
program drawing on the GPU, both engines drop ~8% (~64 t/s decode) and the
ratios hold. What changed since the table below:

- stream-K in MMQ: prefill 128 from 2767 to 3269 t/s (+18%), 512 +5%;
- explicit load of the Q8_0 rows in the decode matvec: gate/up from 170 to
  161 µs per layer (llama.cpp 159), registers from 128 to 80;
- two-pass argmax: 46 → 5 µs per token;
- greedy decode pipeline (the next step is queued before the host reads the
  token): ~0.3%.

Verification on the final binary: selftest 226/226, identical perplexity (4B
2.6689 / 53.5%, 35B 1.4291 / 57.2%), batched prefill = token by token (cosine
0.99979 on the 4B, 0.99897 on the 35B), expert offload identical to resident.

## Qwen3-4B Q8_0 (dense) — 2026-09-26 scoreboard, measured interleaved

**Correction:** an earlier version of this table said airun's decode beat
llama.cpp (65.5 against 54.4). The two measurements were taken at different
times and this machine's speed varies up to ~15% between sessions (the same
llama.cpp binary measured 54, 59 and 64 t/s during the day). It was not
right. From here on, airun and llama.cpp run **alternated in the same
session**, 3 rounds, median:

| Test | airun | llama.cpp | airun / llama |
|---|---|---|---|
| prefill 128 | 1343 | 2344 | 0.57 |
| prefill 512 | 1525 | 2970 | 0.51 |
| decode @ 16 | 48.1 | 52.3 | 0.92 |
| decode @ 3072 | 42.9 | 47.4 | 0.91 |

airun's gain against itself (binary from the start of the session,
interleaved): prefill 512 from ~695 to ~1760 t/s (2.5x); decode @ 3072 +13% to
+25%; short decode unchanged. They came from: int8 MMQ on tensor cores with
`cp.async` (prefill), flash attention (prefill), and decode attention grouped
by KV head.

## Quality (`airun perplexity`, built-in passage, 159 tokens)

| Model | KV f16 | KV q8_0 | KV q4_0 |
|---|---|---|---|
| Qwen3-4B Q8_0 | 2.6689 / 53.5% | 2.6622 / 53.5% | 2.5301 / 53.5% |
| Qwen3.8-35B-A3B Q4_K_M | 1.4291 / 57.2% | 1.4304 / 56.6% | 1.4479 / 57.2% |

The 35B scored 7.02 / 8.8% at the start of the session (see AGENTS.md §7).
The 4B's improvement with q4_0 is noise from a short passage, not a real gain.

Sweep of the layout interpretations over the fixed base: `sweep2.txt`.

## Qwen3.8-35B-A3B Q4_K_M (20.2 GB) — 212992 context, KV q4_0

Real generation: 25-token chat prompt, 400 tokens at temperature 0.7, seed 1.

| Engine | Configuration | Decode |
|---|---|---|
| airun, mmap + page faults | `--expert-cache 104` | ~3 t/s (bench) |
| airun, parallel reader | `--expert-cache 104` | **17.0–17.4 t/s** (89–90% hits) |
| llama.cpp | `-ncmoe 24` | 15.3 t/s |
| llama.cpp | `-ncmoe 28` | 16.1 t/s |

Where the time goes (airun, 425 tokens with prefill): 18.1 s reading misses
from the GGUF, 5.8 s of GPU work. Misses add up to ~61 MB/token; the disk
reads ~84 MB/s during generation (~7% of the misses) because 16 GB of RAM
cannot hold the ~11 GB of experts that stay outside VRAM.

`airun bench` with a repeated token measures 54 t/s — that is degenerate
routing (99.8% hits), not real performance. The bench now uses natural text
on MoE.
