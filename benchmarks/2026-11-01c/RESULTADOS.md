# 2026-11-01 (noite): contra o Strata 0.1.55 (release v0.2.9)

`qwen4.ps1 -StrataExe E:\Strata\engine-0.1.55\strata.exe -StrataPack E:\Strata-data\packs\iq3_s-abl-055`:
o mesmo prompt de 792 tokens e 128 tokens gulosos, 3 rodadas alternadas. O pack
foi refeito com o `iq_pack.py` dessa versão.

| Rodada | airun | Strata 0.1.55 | llama.cpp |
|---|---|---|---|
| 1 | 286,7 / **39,3** | **295,4** / 33,5 | 36,8 / 9,8 |
| 2 | 293,9 / **39,0** | **305,9** / 32,6 | 37,5 / 9,5 |
| 3 | 288,8 / **39,0** | **293,7** / 32,3 | 36,4 / 9,6 |

Os valores são prompt / geração, em tok/s.

- **Geração:** a airun fica à frente, 39,1 contra 32,8 de média (+19%).
- **Prompt:** o Strata 0.1.55 ficou ~3% à frente, 298 contra 290 de média. O
  0.1.44 fazia ~262.
