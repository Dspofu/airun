# Tasks

> What is being worked on, what was finished and what is known to be wrong.
> Agents add and update entries here; the maintainer prunes the file by hand.
> Rules: [AGENTS.md §13](AGENTS.md#13-keeping-the-docs-clean).

<!--
Entry: - commit `{hash}` | `{author}` - `{AI}` | {message}
  - {hash}: short hash; `pending` until committed, then replace it.
  - {author}: the git author. {AI}: the agent that did the work, or `-`.
  - {message}: one line. Optional detail as one indented line below it.
Newest first. Move an entry between sections instead of duplicating it.
-->

## In progress

## Fixed

[X] - commit `d951d88` | `Dspofu` - `Claude` | FFN split allowed up to 3/4 of the feed-forward in RAM (was 1/2)
  - 27B Q4_K_M on the 3060: 6.1 to 6.5 tok/s, 11.7 to 13.0 with --draft 3. Q8_0 (needs more) unchanged at 2.8 / 5.4: bound by RAM bandwidth.

[X] - commit `d951d88` | `Dspofu` - `Claude` | Decode attention on fp16 tensor cores (sm_80+), quantized KV staged raw
  - 35B-A3B Q4_K_M on the 3060, 256k ctx, q4_0 KV, 160k filled: 16.2 to 49.6 tok/s. New selftest: random-cache split decode against the CPU.

[X] - commit `d951d88` | `Dspofu` - `Claude` | Dense IQ types on integer kernels up to 63 tokens (matvec_iq_kernel), draft vocabulary 64k by default
  - 27B IQ2_M on the 3060, 400 tokens: decode 4.5 to 24.8 tok/s; --draft 3 3.5 to ~46 tok/s (code) / ~42 (Portuguese); 32-token prefill 12 to ~71 tok/s.

[X] - commit `12e1276` | `Dspofu` - `Claude` | Docs cleanup: AGENTS.md rewritten as current state, rules against doc junk, Tasks.md
  - Also documents NVFP4 on GPUs without FP4 tensor cores in the README.

## Warn

- commit `631bba0` | `Dspofu` - `-` | Stream-K MMQ under 32 tokens brought a 5070 Ti down (TDR); cause not found
  - Stream-K now starts at 32 tokens. If a card is lost again, try `AIRUN_NO_MMQ_STREAMK=1` first.
- commit `-` | `Dspofu` - `-` | `--expert-cache` with Qwen3.6-35B-A3B-NVFP4-MTP-HQ fails on the first request
  - "gate and up experts must share a type and layout".
- commit `-` | `Dspofu` - `-` | MoE with experts in RAM and batches of 2–15 tokens is not deterministic across runs
  - Sum order depends on residency; `AIRUN_MOE_OVERLAP=0` makes it deterministic.
