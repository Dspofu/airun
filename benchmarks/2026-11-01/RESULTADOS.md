# 2026-11-01: Qwen4 contra Strata e llama.cpp, e QSA

Máquina: RTX 5070 Ti (PCIe 4.0 x16) + RTX 5060 Ti (PCIe 3.0 x4), 64 GB de RAM,
sem P2P.

## Qwen4 Flash-Next (qwen4exp, IQ3_S, 78 GB)

`qwen4.ps1`: o mesmo prompt de 792 tokens, 128 tokens gulosos, motores
alternados a cada rodada (`qwen4.csv`, `qwen4.log`).

| Motor | Prompt (tok/s) | Geração (tok/s) |
|---|---|---|
| Strata 0.1.44.1 | 271 / 282 | 32,7 / 32,1 |
| airun | 93 / 90 | 19,4 / 18,9 |
| llama.cpp (fork com qwen4exp, `--n-cpu-moe 48`) | 30 / 32 | 9,5 / 9,6 |

- O Strata usa só a 5070 Ti. Os experts que não estão no cache da VRAM são
  calculados na CPU (AVX2, 8 threads), e o cache vem preenchido por um perfil de
  roteamento. O pack foi gerado do mesmo GGUF com
  `tools/iq_pack.py --gguf <shard 1> --out E:\Strata-data\packs\iq3_s-abl`.
  Rodou com `--spec 4` (rascunho por lookup; 3 de 369 aceitos) e sem MTP, porque
  esse fine-tune não tem cabeça MTP.
- A airun usa as duas GPUs e lê pela PCIe os experts que estão em RAM fixada,
  em vez de calculá-los na CPU.

## Qwen3.8-27B NVFP4

O Strata não roda esse modelo (só o Flash-Next). Contra o llama.cpp, veja
`../2026-10-31/`: airun 3694 / 3703 / 49,1 contra 3460 / 3663 / 47,9
(pp512 / pp2048 / decode). Esta rodada repetiu o `airun bench` depois do QSA:
3796 / 3688 / 49,1.

## Repetição depois do commit do QSA (340a77c), 3 rodadas

O Strata recusa o 27B (`iq_pack.py`: "the model has no expert tensors"); ele só
roda o Flash-Next.

**Qwen3.8-27B NVFP4** (`scoreboard-27b.ps1`, melhor de 3, tok/s):

| Teste | airun | llama.cpp | airun / llama.cpp |
|---|---|---|---|
| prompt 512 | 3797 | 3472 | 1,09 |
| prompt 2048 | 3690 | 3652 | 1,01 |
| geração 128 | 49,0 | 47,9 | 1,02 |
| MTP, prompt 1 | 113,7 | 104,6 | 1,09 |
| MTP, prompt 2 | 94,9 | 91,7 | 1,03 |

**Qwen4 Flash-Next** (`qwen4.ps1`, prompt / geração em tok/s):

| Rodada | airun | Strata | llama.cpp |
|---|---|---|---|
| 1 | 88,6 / 19,1 | 263,7 / 32,9 | 36,8 / 9,5 |
| 2 | 90,3 / 19,2 | 136,2 / 33,0 | 37,4 / 9,6 |
| 3 | 82,3 / 18,7 | 110,6 / 33,5 | 35,4 / 9,8 |

O prompt do Strata caiu nas rodadas 2 e 3, que rodaram logo depois das outras
engines. Uma causa plausível, que não verifiquei, é a parte do prompt que ele
calcula na CPU disputando RAM e cache de arquivo. A geração dele ficou estável
em ~33 tok/s.

## QSA (atenção esparsa do qwen4exp)

Perplexidade num texto de 25 mil tokens (AGENTS.md + FLAGS.md + docs do
Strata). `airun perplexity --chunk N --batched` contra `llama-perplexity -c N`:

| Pedaço | Referência | airun QSA | airun densa (`AIRUN_QSA=off`) |
|---|---|---|---|
| 4096 (posições 2048–4095, esparsas) | 5,1563 | 5,1684 | 5,1717 |
| 2048 (tudo denso, base) | 5,5440 | 5,5269 | — |

Em 2048 a airun e a referência já diferem até ±0,5% por pedaço, por
arredondamento. Em 4096 a versão com QSA fica mais perto da referência que a
densa em quase todos os pedaços. A exatidão dos kernels está no selftest: a
seleção é idêntica bit a bit, e a atenção difere em 1e-7 (decode) e 7e-5 (flash).

Num prompt de 4920 tokens, as três versões dão a mesma resposta. A da airun com
QSA difere da referência só na posição de um `**`.
