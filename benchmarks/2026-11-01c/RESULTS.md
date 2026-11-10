# 2026-11-01 (evening): against Strata 0.1.55 (release v0.2.9)

`qwen4.ps1 -StrataExe E:\Strata\engine-0.1.55\strata.exe -StrataPack E:\Strata-data\packs\iq3_s-abl-055`:
the same 792-token prompt and 128 greedy tokens, 3 alternated rounds. The
pack was rebuilt with that version's `iq_pack.py`.

| Round | airun | Strata 0.1.55 | llama.cpp |
|---|---|---|---|
| 1 | 286.7 / **39.3** | **295.4** / 33.5 | 36.8 / 9.8 |
| 2 | 293.9 / **39.0** | **305.9** / 32.6 | 37.5 / 9.5 |
| 3 | 288.8 / **39.0** | **293.7** / 32.3 | 36.4 / 9.6 |

Values are prompt / generation, in tok/s.

- **Generation:** airun stays ahead, 39.1 against 32.8 on average (+19%).
- **Prompt:** Strata 0.1.55 came out ~3% ahead, 298 against 290 on average.
  0.1.44 did ~262.
