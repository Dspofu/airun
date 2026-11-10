# 2026-11-01: Qwen4 against Strata and llama.cpp, and QSA

Machine: RTX 5070 Ti (PCIe 4.0 x16) + RTX 5060 Ti (PCIe 3.0 x4), 64 GB of RAM,
no P2P.

## Qwen4 Flash-Next (qwen4exp, IQ3_S, 78 GB)

`qwen4.ps1`: the same 792-token prompt, 128 greedy tokens, engines alternated
every round (`qwen4.csv`, `qwen4.log`).

| Engine | Prompt (tok/s) | Generation (tok/s) |
|---|---|---|
| Strata 0.1.44.1 | 271 / 282 | 32.7 / 32.1 |
| airun | 93 / 90 | 19.4 / 18.9 |
| llama.cpp (fork with qwen4exp, `--n-cpu-moe 48`) | 30 / 32 | 9.5 / 9.6 |

- Strata uses only the 5070 Ti. Experts not in the VRAM cache are computed on
  the CPU (AVX2, 8 threads), and the cache is pre-filled from a routing
  profile. The pack was generated from the same GGUF with
  `tools/iq_pack.py --gguf <shard 1> --out E:\Strata-data\packs\iq3_s-abl`.
  It ran with `--spec 4` (lookup drafting; 3 of 369 accepted) and without MTP,
  because this fine-tune has no MTP head.
- airun uses both GPUs and reads the experts in pinned RAM over PCIe, instead
  of computing them on the CPU.

## Qwen3.8-27B NVFP4

Strata does not run this model (only Flash-Next). Against llama.cpp, see
`../2026-10-31/`: airun 3694 / 3703 / 49.1 against 3460 / 3663 / 47.9
(pp512 / pp2048 / decode). This round repeated `airun bench` after QSA:
3796 / 3688 / 49.1.

## Repeat after the QSA commit (340a77c), 3 rounds

Strata refuses the 27B (`iq_pack.py`: "the model has no expert tensors"); it
only runs Flash-Next.

**Qwen3.8-27B NVFP4** (`scoreboard-27b.ps1`, best of 3, tok/s):

| Test | airun | llama.cpp | airun / llama.cpp |
|---|---|---|---|
| prompt 512 | 3797 | 3472 | 1.09 |
| prompt 2048 | 3690 | 3652 | 1.01 |
| generation 128 | 49.0 | 47.9 | 1.02 |
| MTP, prompt 1 | 113.7 | 104.6 | 1.09 |
| MTP, prompt 2 | 94.9 | 91.7 | 1.03 |

**Qwen4 Flash-Next** (`qwen4.ps1`, prompt / generation in tok/s):

| Round | airun | Strata | llama.cpp |
|---|---|---|---|
| 1 | 88.6 / 19.1 | 263.7 / 32.9 | 36.8 / 9.5 |
| 2 | 90.3 / 19.2 | 136.2 / 33.0 | 37.4 / 9.6 |
| 3 | 82.3 / 18.7 | 110.6 / 33.5 | 35.4 / 9.8 |

Strata's prompt dropped in rounds 2 and 3, which ran right after the other
engines. A plausible cause, which I did not verify, is the part of the prompt
it computes on the CPU competing for RAM and file cache. Its generation
stayed stable at ~33 tok/s.

## QSA (qwen4exp sparse attention)

Perplexity on a 25k-token text (AGENTS.md + FLAGS.md + Strata docs). `airun
perplexity --chunk N --batched` against `llama-perplexity -c N`:

| Chunk | Reference | airun QSA | airun dense (`AIRUN_QSA=off`) |
|---|---|---|---|
| 4096 (positions 2048–4095, sparse) | 5.1563 | 5.1684 | 5.1717 |
| 2048 (all dense, baseline) | 5.5440 | 5.5269 | — |

At 2048 airun and the reference already differ by up to ±0.5% per chunk, from
rounding. At 4096 the QSA version stays closer to the reference than the
dense one in almost every chunk. Kernel exactness is in the selftest: the
selection is bit-identical, and attention differs by 1e-7 (decode) and 7e-5
(flash).

On a 4920-token prompt, the three versions give the same answer. airun's with
QSA differs from the reference only in the position of a `**`.
