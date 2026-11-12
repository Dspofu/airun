# Speed by context depth (2026-11-12)

Qwen3.8-27B NVFP4 (Q5_K output head, MTP head in the file), RTX 5070 Ti +
RTX 5060 Ti (no P2P), `airun serve --kv-type q4_0 -c 131072`, one request at
a time. The planner put 53 layers on the 5070 Ti and 11 on the 5060 Ti with
`--draft 4`, 55 and 9 without.

Script: [`depth.ps1`](depth.ps1). Each prompt is a slice of this repository's
Rust source followed by "explain what this code does", thinking off, greedy,
200 tokens generated, `cache_prompt: false`, and every prompt starts
differently, so nothing is reused between requests. One run per point.

| Prompt tokens | Prefill tok/s | Time to read the prompt | Decode tok/s, `--draft 4` | Drafts accepted | Decode tok/s, no MTP |
|---:|---:|---:|---:|---:|---:|
| 764 | 1944 | 0.4 s | 74.0 | 136/260 | 41.7 |
| 3,467 | 3665 | 0.9 s | 76.1 | 137/252 | 40.7 |
| 6,851 | 3684 | 1.9 s | 68.6 | 134/272 | 39.4 |
| 14,006 | 3341 | 4.2 s | 68.5 | 142/240 | 35.9 |
| 27,885 | 2738 | 10.2 s | 59.1 | 145/228 | 31.4 |
| 61,491 | 1859 | 33.1 s | 40.4 | 141/252 | 24.6 |
| 111,761 | 1260 | 88.7 s | 26.8 | 135/264 | 18.4 |

Prefill figures are from the `--draft 4` server (the one without MTP read the
same prompts 1-5% faster, except the shortest: 1676 tok/s).

Reading it:
- Decode holds within 10% of its best up to ~14k tokens, has lost a quarter by
  28k and two thirds by 112k: every token's attention reads the whole cache
  of the 16 attention layers (and the MTP head's).
- MTP's gain shrinks with depth, from 1.8-1.9x to 1.5x at 112k: a verify's
  rows each read the cache.
- Prefill peaks between 3k and 7k tokens (the first batches pay for the
  kernels' first use) and falls to a third by 112k: attention over the prompt
  grows with its square.

Also measured the same day, `serve`, a short and very predictable prompt
(one request, 300 tokens, thinking off; `--draft 4` unless noted):

| Flags | Context chosen | Layers on 5070 Ti / 5060 Ti | tok/s |
|---|---:|---:|---:|
| `--draft 3 --parallel 4 -c 248144` | 248,144 | 34 / 30 (KV of 5 layers in RAM) | 89 |
| none (no MTP) | 94,976 | 58 / 6 | 43 |
| `--draft 4` | 62,976 | 58 / 6 | 106 |
| `--draft 4 --parallel 4` | 76,800 per slot | 42 / 22 | 100 (103 summed over 4 requests) |
| `--draft 4 --prefer speed` | 4,864 | 64 / 0 | 120 |
| `--draft 4 --prefer context` | 728,832 | 29 / 35 | 91 |

The last two rows were measured before the final calibration of the planner's
pace model; the context each picks may differ by a few thousand tokens now.
