# 2026-11-01 (noite): prompt longo e contexto grande, Qwen3.8-27B NVFP4

> **Superado por [`../2026-11-01f/`](../2026-11-01f/RESULTADOS.md).** O
> diagnóstico abaixo estava errado. A causa principal era o split rodar as
> GPUs em série; a atenção era limitada por cálculo, não pela leitura repetida
> de K/V. A medição desta pasta fica como registro do antes.

`longctx.ps1`: airun contra llama.cpp b11760, 2 rodadas alternadas, melhor
das duas, em tok/s. Saída bruta em `longctx.csv` e `longctx.log`.

Máquina: RTX 5070 Ti 16 GB (PCIe 4.0 x16) + RTX 5060 Ti 16 GB (PCIe 3.0 x4),
Ryzen 7 5700X3D, 64 GB de RAM, Windows 11, sem P2P.

- **GPUs:** com 32k ou mais de contexto o 27B não cabe numa placa, então os
  dois motores dividem as camadas entre as duas (layer split).
- **Lote de prompt:** 512 nos dois.
- **llama.cpp:** `llama-bench` com flash attention.
- **airun:** `airun bench`.
- **MTP:** nenhum dos dois usa (o `llama-bench` não tem MTP).

| Teste | airun | llama.cpp | airun / llama.cpp |
|---|---:|---:|---:|
| prompt de 8.192 tokens | 2348 | **3276** | 0,72 |
| prompt de 32.768 tokens | 1853 | **2835** | 0,65 |
| prompt de 131.072 tokens | 1012 | **1598** | 0,63 |
| geração @ 16 tokens de contexto | **33,3** | 31,4 | 1,06 |
| geração @ 32.768 | 28,5 | **29,1** | 0,98 |
| geração @ 131.072 | 20,9 | **21,4** | 0,98 |

- **Geração:** empate. À frente com contexto curto, 2% atrás com contexto
  cheio.
- **Prompt longo:** é a fraqueza atual, e a diferença cresce com o contexto.
  Em prompts curtos a airun vence (3797 contra 3472 em 512 tokens numa GPU,
  `../2026-11-01/`). O resto do forward já é mais rápido, então o que perde é
  a atenção do prefill, cujo custo cresce com o cache.
  - Em 128k, a atenção custa ~98 s dos 130 s da airun, contra ~48 s no
    llama.cpp. A estimativa desconta o custo dos matmuls a 3700 tok/s.
- **Por que a atenção perde** (`flash2_attention_kernel` com head_dim 256):
  - **K/V lidos 6 vezes:** cada uma das 6 cabeças de consulta que dividem uma
    cabeça de KV lê o cache inteiro de novo (~134 MB de K/V por bloco em
    128k). O llama.cpp agrupa as cabeças do GQA e lê uma vez.
  - **Tiles de 32 chaves.**
  - **Ocupação baixa:** um bloco de 128 consultas por SM, 96 blocos para 70
    SMs. É uma onda e meia, com a segunda pela metade.
- **Próximo passo:** consultas das 6 cabeças de um grupo no mesmo bloco e o
  intervalo de KV dividido entre blocos quando o contexto é longo.
