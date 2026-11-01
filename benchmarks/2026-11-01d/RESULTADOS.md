# 2026-11-01 (noite): prefill otimizado contra o Strata 0.1.55

`qwen4.ps1 -Batch N`: os três motores com o mesmo tamanho de bloco de prompt
(airun `-b N`, Strata `--prefill N`), prompt de 792 tokens e 128 tokens
gulosos, 2 rodadas por tamanho. Valores em tok/s.

| Bloco | Rodada | airun | Strata 0.1.55 | llama.cpp |
|---|---|---|---|---|
| 1024 | 1 | **447,0 / 39,2** | 414,4 / 34,1 | 35,9 / 9,9 |
| 1024 | 2 | **449,3 / 39,4** | 426,6 / 33,9 | 35,0 / 9,9 |
| 512 | 1 | **404,5 / 40,7** | 315,6 / 34,4 | 34,8 / 9,9 |
| 512 | 2 | **417,7 / 40,6** | 303,1 / 34,4 | 35,6 / 9,9 |

Os valores são prompt / geração.

- **Bloco de 1024:** prompt +7%, geração +16%.
- **Bloco de 512:** prompt +33%, geração +18%.
- **Qualidade:** saída gulosa igual à da referência em 3 prompts × 80 tokens;
  perplexidade (blocos de 2048) 5,5145.

O que mudou desde `../2026-11-01c/` (290 tok/s de prompt):
- **Matmul BF16:** um tile menor, para projeções com poucas linhas e K longo,
  enche a GPU.
- **MoE em tensor cores (v2):** ativações em fp16 uma vez por camada, lidas
  direto do L2. Fatias de K=128 com as 128 threads decodificando, e `h` em
  fp16.
- **Experts na RAM:** cópia das linhas com leituras de 16 bytes, quatro em voo
  por thread. O passo desses experts (limitado pela PCIe) roda num stream
  paralelo ao dos experts na VRAM (limitado pelo cálculo).
- **PLE:** um único `PrefetchVirtualMemory` pede ao Windows todas as páginas
  das linhas antes do gather (456 → 130 ms em 12.672 linhas). Mais threads de
  page fault só disputavam o lock do working set, e leituras sem buffer em SSD
  SATA renderam ~7 mil linhas por segundo.
