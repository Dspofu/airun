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
