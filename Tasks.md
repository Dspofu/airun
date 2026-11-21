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

[X] - commit `pending` | `Dspofu` - `Claude` | Dense IQ prefill on int8 tensor cores (MMQ) from 64 tokens: IQ2_S, IQ3_XXS (cp.async) and IQ3_S, codebook in shared memory, fused gate/up
  - 27B IQ2_M on the 3060, pp512: 139 to 492 t/s (llama-bench 509), pp128 91 to 443. Perplexity unchanged (11.70 both paths). AIRUN_NO_IQ_MMQ=1 goes back.

[X] - commit `pending` | `Dspofu` - `Claude` | MoE files the kernels misread were loaded and wrote "!!!!"; now refused with a message
  - Per-expert NVFP4 `.scale` sidecars (F32[n_experts]) and gate/up experts of different types. Qwen3.6-35B-A3B-NVFP4-MTP-HQ has both. Support is open (see Warn).

[X] - commit `09ab5c4` | `Dspofu` - `Claude` | Web UI and API take documents: text files in any encoding and PDFs; scanned pages go to the vision model as images
  - Attach button always visible; /extract for the page; file / input_file / document parts in the APIs; request limit 64 MiB. Scans in JPEG, CCITT, Flate; not JBIG2 / JPEG 2000.

[X] - commit `09ab5c4` | `Dspofu` - `Claude` | --mmproj on a full 12 GB card failed ("0 B free") or, forced, paged the model to RAM
  - WDDM over-reports free VRAM; with vision the plan uses NVML's free memory plus a 320 MiB margin. 27B IQ2_M + vision: loads, ~13 tok/s instead of failing.

[X] - commit `d951d88` | `Dspofu` - `Claude` | FFN split allowed up to 3/4 of the feed-forward in RAM (was 1/2)
  - 27B Q4_K_M on the 3060: 6.1 to 6.5 tok/s, 11.7 to 13.0 with --draft 3. Q8_0 (needs more) unchanged at 2.8 / 5.4: bound by RAM bandwidth.

[X] - commit `d951d88` | `Dspofu` - `Claude` | Decode attention on fp16 tensor cores (sm_80+), quantized KV staged raw
  - 35B-A3B Q4_K_M on the 3060, 256k ctx, q4_0 KV, 160k filled: 16.2 to 49.6 tok/s. New selftest: random-cache split decode against the CPU.

[X] - commit `d951d88` | `Dspofu` - `Claude` | Dense IQ types on integer kernels up to 63 tokens (matvec_iq_kernel), draft vocabulary 64k by default
  - 27B IQ2_M on the 3060, 400 tokens: decode 4.5 to 24.8 tok/s; --draft 3 3.5 to ~46 tok/s (code) / ~42 (Portuguese); 32-token prefill 12 to ~71 tok/s.

[X] - commit `12e1276` | `Dspofu` - `Claude` | Docs cleanup: AGENTS.md rewritten as current state, rules against doc junk, Tasks.md
  - Also documents NVFP4 on GPUs without FP4 tensor cores in the README.

## Warn

- commit `pending` | `Dspofu` - `Claude` | MoE kernels cannot apply per-expert NVFP4 scales nor read gate/up of different types (such files are refused)
  - Qwen3.6-35B-A3B-NVFP4-MTP-HQ needs both: layers 32-39 have gate NVFP4 and up Q4_K.

- commit `-` | `Dspofu` - `Claude` | 35B-A3B Q4_K_M decode measured 63.6 tok/s (natural text, q8_0 KV) where 73-75 was recorded; not bisected
  - Different prompt from the earlier measurement; rerun the same way before calling it a regression.

- commit `631bba0` | `Dspofu` - `-` | Stream-K MMQ under 32 tokens brought a 5070 Ti down (TDR); cause not found
  - Stream-K now starts at 32 tokens. If a card is lost again, try `AIRUN_NO_MMQ_STREAMK=1` first.
- commit `-` | `Dspofu` - `-` | `--expert-cache` with Qwen3.6-35B-A3B-NVFP4-MTP-HQ fails on the first request
  - "gate and up experts must share a type and layout".
- commit `-` | `Dspofu` - `-` | MoE with experts in RAM and batches of 2–15 tokens is not deterministic across runs
  - Sum order depends on residency; `AIRUN_MOE_OVERLAP=0` makes it deterministic.
