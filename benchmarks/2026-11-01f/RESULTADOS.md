# 2026-11-01 (noite, depois): prefill longo à frente do llama.cpp

Mesmo script e condições de [`../2026-11-01e/`](../2026-11-01e/RESULTADOS.md):

- **Modelo:** Qwen3.8-27B NVFP4 dividido entre as duas GPUs.
- **Prompt:** lotes de 512.
- **Rodadas:** 2, alternadas entre os motores, com o melhor das duas em tok/s.
- **Saída bruta:** `longctx.csv` e `longctx.log`.

| Teste | airun antes | **airun agora** | llama.cpp | agora / llama.cpp |
|---|---:|---:|---:|---:|
| prompt de 8.192 tokens | 2348 | **3519** | 3211 | 1,10 |
| prompt de 32.768 tokens | 1853 | **2821** | 2631 | 1,07 |
| prompt de 131.072 tokens | 1012 | **1735** | 1599 | 1,09 |
| geração @ 16 | 33,3 | **31,3** | 30,8 | 1,02 |
| geração @ 32.768 | 28,5 | 27,5 | **27,8** | 0,99 |
| geração @ 131.072 | 20,9 | 19,7 | **21,8** | 0,91 |

## O que mudou

1. **As GPUs passaram a trabalhar ao mesmo tempo no prefill.** Este foi o
   ganho maior. O `handoff` entre as placas sincronizava a de origem pelo
   host, e o `airun bench` lia os logits de todo pedaço de prompt. Os dois
   punham as GPUs em série: a 5060 Ti (metade dos SMs) esperava a 5070 Ti, e
   vice-versa.
   - Agora a passagem entre GPUs é ordenada só por eventos CUDA. A GPU 0
     processa o pedaço c+1 enquanto a GPU 1 processa o c.
   - Num prompt de 4k: 3653 tok/s numa GPU, 2501 em split antes, 3566 em
     split agora. O llama.cpp faz 3514 em split.
2. **Atenção do prefill de 55 para 72,7 TFLOPS** (head_dim 256, medida
   isolada por `crates/airun-cuda/tests/attention_bench.rs`):
   - **P·V em fp16 por tile de chaves:** somas de no máximo 32 termos, com o
     total em fp32. Só a partir de 4096 chaves, e com perplexidade igual:
     1,2960 contra 1,2999 em 16k tokens com pedaços de 8k.
   - **Caminho sem máscara** para tiles inteiramente visíveis.
   - **Reescala do acumulador** só quando o máximo de alguma linha muda.

## Geração

- **O decode não mudou.** Rodado isolado, ele dá o mesmo que antes:
  - @ 16: 33,7 contra 33,6 com a passagem antiga;
  - @ 32k: 28,9 contra 28,8;
  - @ 128k: 20,9 nos dois modos.
- **Os números da tabela vêm logo depois de prefills pesados.** Agora esses
  prefills deixam as duas placas ocupadas o tempo todo, então elas chegam ao
  decode mais quentes. É uma hipótese que não medi; os clocks no fim da
  rodada isolada foram 2835 e 2010–2332 MHz.
- **Com 128k de contexto o llama.cpp ainda gera ~4% mais rápido** (21,4–21,8
  contra 20,9 em medição isolada). É o próximo alvo.

## Correção

- **selftest:** 275 de 275, com um caso novo para o caminho de P·V em fp16.
- **Qwen4 Flash-Next:** a saída gulosa de 3 prompts × 80 tokens continua
  idêntica à da referência. O Qwen4 também usa as duas GPUs e não perdeu
  velocidade: prompt 452–496 e geração 39,1 tok/s, contra 449 / 39,4 antes,
  com lote de 1024.
- **Qwen3.8-27B:** "The capital of France is" → " Paris." com `-b 512` e
  `-b 1`.
