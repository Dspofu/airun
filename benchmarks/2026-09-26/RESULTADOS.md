# Benchmarks 2026-09-26 — RTX 3060 12 GB, i5-10400F, 16 GB DDR4-2666, NVMe (E:)

PCIe 3.0 x16. Build com `AIRUN_CUDA_ARCHS=86`. llama.cpp em `E:\llamacpp`.

## Qwen3-4B Q8_0 (denso) — placar final, 2026-09-27

Alternados na mesma sessão, `scoreboard.ps1 -Rounds 3`, melhor execução de
cada (interferência nesta máquina só acrescenta tempo, então a mais rápida é a
estimativa robusta):

| Teste | airun | llama.cpp | airun / llama |
|---|---|---|---|
| prefill 128 | 3535 | 3118 | **1,13** |
| prefill 512 | 3979 | 3829 | **1,04** |
| decode @ 16 | 70,3 | 69,1 | 1,02 |
| decode @ 3072 | 63,7 | 62,9 | 1,01 |

O decode está empatado dentro do ruído (±2%); o prefill vence com folga. Os
números absolutos desta tabela são de uma janela sem interferência: com outro
programa desenhando na GPU, os dois motores caem ~8% (~64 t/s de decode) e as
razões se mantêm. O que mudou desde a tabela abaixo:

- stream-K no MMQ: prefill 128 de 2767 para 3269 t/s (+18%), 512 +5%;
- carga explícita das linhas Q8_0 no matvec de decode: gate/up de 170 para
  161 µs por camada (llama.cpp 159), registradores de 128 para 80;
- argmax em duas passadas: 46 → 5 µs por token;
- pipeline do decode guloso (o passo seguinte é enfileirado antes do host ler
  o token): ~0,3%.

Verificação no binário final: selftest 226/226, perplexidade idêntica (4B
2.6689 / 53.5%, 35B 1.4291 / 57.2%), prefill em lote = token a token (cosseno
0.99979 no 4B, 0.99897 no 35B), offload de experts idêntico ao residente.

## Qwen3-4B Q8_0 (denso) — placar de 2026-09-26, medido intercalado

**Correção:** uma versão anterior desta tabela dizia que o decode do airun
vencia o llama.cpp (65,5 contra 54,4). As duas medições foram feitas em
momentos diferentes e a velocidade desta máquina varia até ~15% entre sessões
(o mesmo binário do llama.cpp mediu 54, 59 e 64 t/s ao longo do dia). Não
estava certo. A partir daqui, airun e llama.cpp rodam **alternados na mesma
sessão**, 3 rodadas, mediana:

| Teste | airun | llama.cpp | airun / llama |
|---|---|---|---|
| prefill 128 | 1343 | 2344 | 0,57 |
| prefill 512 | 1525 | 2970 | 0,51 |
| decode @ 16 | 48,1 | 52,3 | 0,92 |
| decode @ 3072 | 42,9 | 47,4 | 0,91 |

Ganho do airun contra ele mesmo (binário do início da sessão, intercalado):
prefill 512 de ~695 para ~1760 t/s (2,5x); decode @ 3072 +13% a +25%; decode
curto igual. Vieram de: MMQ int8 em tensor core com `cp.async` (prefill),
flash attention (prefill), e atenção de decode agrupada por head de KV.

## Qualidade (`airun perplexity`, passagem embutida, 159 tokens)

| Modelo | KV f16 | KV q8_0 | KV q4_0 |
|---|---|---|---|
| Qwen3-4B Q8_0 | 2.6689 / 53.5% | 2.6622 / 53.5% | 2.5301 / 53.5% |
| Qwen3.8-35B-A3B Q4_K_M | 1.4291 / 57.2% | 1.4304 / 56.6% | 1.4479 / 57.2% |

O 35B marcava 7.02 / 8.8% no início da sessão (veja AGENTS.md §7). A melhora do
4B com q4_0 é ruído de uma passagem curta, não ganho real.

Varredura das interpretações de layout sobre a base corrigida: `sweep2.txt`.

## Qwen3.8-35B-A3B Q4_K_M (20.2 GB) — 212992 de contexto, KV q4_0

Geração real: prompt de chat de 25 tokens, 400 tokens a temperatura 0.7, seed 1.

| Engine | Configuração | Decode |
|---|---|---|
| airun, mmap + page faults | `--expert-cache 104` | ~3 t/s (bench) |
| airun, leitor paralelo | `--expert-cache 104` | **17.0–17.4 t/s** (89–90% de acerto) |
| llama.cpp | `-ncmoe 24` | 15.3 t/s |
| llama.cpp | `-ncmoe 28` | 16.1 t/s |

Onde o tempo vai (airun, 425 tokens com prefill): 18.1 s lendo misses do GGUF,
5.8 s de trabalho da GPU. Os misses somam ~61 MB/token; o disco lê ~84 MB/s
durante a geração (~7% dos misses) porque 16 GB de RAM não comportam os
~11 GB de experts que ficam fora da VRAM.

`airun bench` com um token repetido mede 54 t/s — é roteamento degenerado
(99.8% de acerto), não desempenho real. O bench agora usa texto natural em MoE.
