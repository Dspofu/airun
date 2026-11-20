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

[X] - commit `12e1276` | `Dspofu` - `Claude` | Docs cleanup: AGENTS.md rewritten as current state, rules against doc junk, Tasks.md
  - Also documents NVFP4 on GPUs without FP4 tensor cores in the README.

## Warn

- commit `631bba0` | `Dspofu` - `-` | Stream-K MMQ under 32 tokens brought a 5070 Ti down (TDR); cause not found
  - Stream-K now starts at 32 tokens. If a card is lost again, try `AIRUN_NO_MMQ_STREAMK=1` first.
- commit `-` | `Dspofu` - `-` | `--expert-cache` with Qwen3.6-35B-A3B-NVFP4-MTP-HQ fails on the first request
  - "gate and up experts must share a type and layout".
- commit `-` | `Dspofu` - `-` | MoE with experts in RAM and batches of 2–15 tokens is not deterministic across runs
  - Sum order depends on residency; `AIRUN_MOE_OVERLAP=0` makes it deterministic.
