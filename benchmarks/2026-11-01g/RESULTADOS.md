# 2026-11-01 (noite, final): à frente do llama.cpp em todo contexto

Mesmo script e condições de [`../2026-11-01e/`](../2026-11-01e/RESULTADOS.md)
e [`../2026-11-01f/`](../2026-11-01f/RESULTADOS.md):

- **Modelo:** Qwen3.8-27B NVFP4 nas duas GPUs.
- **Prompt:** lotes de 512.
- **Rodadas:** 2, alternadas entre os motores, com o melhor das duas em tok/s.
- **Saída bruta:** `longctx.csv` e `longctx.log`.

| Teste | e (início) | f (prefill) | **g (agora)** | llama.cpp | g / llama.cpp |
|---|---:|---:|---:|---:|---:|
| prompt de 8.192 | 2348 | 3519 | **3694** | 3496 | 1,06 |
| prompt de 32.768 | 1853 | 2821 | **3055** | 2862 | 1,07 |
| prompt de 131.072 | 1012 | 1735 | **2428** | 1597 | 1,52 |
| geração @ 16 | 33,3 | 31,3 | **42,1** | 32,7 | 1,29 |
| geração @ 32.768 | 28,5 | 27,5 | **37,4** | 29,1 | 1,29 |
| geração @ 131.072 | 20,9 | 19,7 | **23,5** | 20,4 | 1,15 |

O llama.cpp variou entre as rodadas desta noite, por exemplo de 20,4 a 21,8
na geração @ 128k. Mesmo contra os melhores números dele, a airun fica à
frente em tudo.

## O que mudou desde `f`

1. **O planner enche a GPU mais rápida primeiro** (`fill_fastest_first`, em
   `plan.rs`).
   - **Antes:** um modelo que não cabia numa placa era dividido em
     proporção à VRAM livre. Com duas placas de 16 GB, metade em cada.
   - **Por que importa:** o decode passa pelas GPUs uma depois da outra, e a
     5060 Ti tem metade da banda da 5070 Ti. Cada camada nela custava o
     dobro.
   - **Agora:** a 5070 Ti fica com 59 das 64 camadas com até 32k de contexto
     e 43 com 128k (o KV ocupa o resto). A 5060 Ti fica com o restante.
   - **Prefill:** em pipeline, o passo é o da placa mais lenta, que agora
     carrega pouco. Daí os +40% em 128k sobre `f`.
2. **Divisões do decode em ondas inteiras.** A atenção do decode divide o
   cache em até 256 pedaços por cabeça de KV.
   - **Antes:** 4 × 256 = 1024 blocos davam 7,1 ondas na 5060 Ti (144 blocos
     por vez), e a oitava rodava quase vazia.
   - **Agora:** o número de pedaços desce para um múltiplo da concorrência
     real da GPU.
   - **Ganho:** +2% no decode @ 128k (23,6 contra 23,1, medido isolado).
     `AIRUN_NO_SPLIT_WAVES=1` desliga.

## Correção

- `cargo test`: o teste novo do planner (`an_oversized_model_fills_the_fastest_gpu_first`)
  falha com a divisão antiga.
- selftest: 275 de 275.
- Qwen4 Flash-Next: idêntico à referência em 3 prompts × 80 tokens.
- MTP no 27B com 64k de contexto (53/11 camadas): 70 tok/s, 77% dos
  rascunhos aceitos, contra 116 tok/s numa GPU com 4k de contexto.

## O que não ajudou

Carregar os tiles do decode com `cp.async`, para pôr todas as peças em voo,
derrubou a banda de 711 para 636 GB/s na 5070 Ti. A latência por tile não era
o gargalo.
