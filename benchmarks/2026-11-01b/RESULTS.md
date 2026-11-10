# 2026-11-01 (afternoon): airun ahead of Strata on Qwen4

Same procedure as `../2026-11-01/` (`qwen4.ps1`): 792-token prompt, 128
greedy tokens, the three engines alternated over 3 rounds. The data is in
`qwen4.csv` and `qwen4.log`.

| Round | airun | Strata 0.1.44.1 | llama.cpp |
|---|---|---|---|
| 1 | **295.3 / 38.7** | 250.2 / 32.5 | 36.7 / 9.5 |
| 2 | **282.4 / 39.0** | 266.4 / 32.2 | 35.1 / 9.7 |
| 3 | **292.5 / 37.6** | 270.1 / 34.0 | 36.3 / 9.9 |

Values are prompt / generation, in tok/s. The same morning, airun did 90 / 19.

- **Best of 3:** prompt 295 against 270 (+9%), generation 39.0 against 34.0
  (+15%).
- **Mean:** prompt 290 against 262 (+11%), generation 38.4 against 32.9
  (+17%).
- **Quality:** greedy output still matches the llama.cpp reference on 3
  prompts × 80 tokens. Perplexity in 2048 chunks: 5.5145, against 5.5269
  before and 5.5440 for the reference.
- **Routing profile:** came from 3 runs with prompts other than the
  benchmark's. Strata uses the profile it ships with.
- **Strata version:** this is the installed executable (0.1.44.1). The code
  was updated to 0.1.55, but the executable only changes by building or
  downloading the new release.

What changed in airun is in `AGENTS.md`, section on experts mapped in RAM.
