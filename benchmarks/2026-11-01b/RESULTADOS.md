# 2026-11-01 (tarde): airun à frente do Strata no Qwen4

Mesmo procedimento de `../2026-11-01/` (`qwen4.ps1`): prompt de 792 tokens, 128
tokens gulosos, os três motores alternados em 3 rodadas. Os dados estão em
`qwen4.csv` e `qwen4.log`.

| Rodada | airun | Strata 0.1.44.1 | llama.cpp |
|---|---|---|---|
| 1 | **295,3 / 38,7** | 250,2 / 32,5 | 36,7 / 9,5 |
| 2 | **282,4 / 39,0** | 266,4 / 32,2 | 35,1 / 9,7 |
| 3 | **292,5 / 37,6** | 270,1 / 34,0 | 36,3 / 9,9 |

Os valores são prompt / geração, em tok/s. No mesmo dia, de manhã, a airun
fazia 90 / 19.

- **Melhor de 3:** prompt 295 contra 270 (+9%), geração 39,0 contra 34,0 (+15%).
- **Média:** prompt 290 contra 262 (+11%), geração 38,4 contra 32,9 (+17%).
- **Qualidade:** a saída gulosa continua igual à da referência llama.cpp em 3
  prompts × 80 tokens. Perplexidade em pedaços de 2048: 5,5145, contra 5,5269
  antes e 5,5440 da referência.
- **Perfil de roteamento:** veio de 3 execuções com prompts diferentes do
  benchmark. O Strata usa o perfil que vem com ele.
- **Versão do Strata:** este é o executável instalado (0.1.44.1). O código foi
  atualizado para 0.1.55, mas o executável só muda compilando ou baixando o
  release novo.

O que mudou na airun está em `AGENTS.md`, seção "Experts mapeados em RAM".
