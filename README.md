# airun

Engine de inferência de LLMs para GPUs NVIDIA, escrita do zero em Rust + CUDA.

Sem llama.cpp, sem ggml, sem PyTorch, sem cuBLAS no caminho de execução. Os
kernels CUDA são deste projeto.

```
airun run modelo.gguf -p "Explique o que é um kernel de GPU" -n 200
```

---

## Por que Rust

O requisito era: mais rápida possível, econômica em recursos, boa para cálculo
matricial em GPUs NVIDIA.

O trabalho pesado acontece nos kernels CUDA — a linguagem host não faz
multiplicação de matriz. O que ela precisa fazer bem é: não introduzir overhead
entre lançamentos de kernel, não desperdiçar RAM enquanto a VRAM está no limite,
e falar com C sem custo.

| | Rust | C++ | Go | Python | Zig |
|---|---|---|---|---|---|
| Sem GC / pausas | sim | sim | **não** | não | sim |
| FFI com CUDA sem custo | sim | nativo | cgo (caro) | ctypes (caro) | sim |
| Memória determinística | sim | sim | não | não | sim |
| Segurança de memória | **verificada** | manual | GC | GC | manual |
| Servidor async maduro | tokio/axum | manual | sim | sim | imaturo |
| Binário único | sim | sim | sim | não | sim |
| Ecossistema estável | sim | sim | sim | sim | **não** |

Go morre no `cgo`: cada chamada CUDA cruza a fronteira, e num forward pass são
milhares. Python não consegue orquestrar sem overhead. C++ empata em desempenho,
mas troca as garantias de memória por nada — e aqui se manipula ponteiro de
device em três crates diferentes. Zig teria servido, mas o ecossistema ainda não
sustenta um servidor HTTP de produção.

Rust dá o desempenho do C++, FFI sem custo com os kernels, e o compilador
verifica a propriedade de cada buffer de GPU.

---

## Requisitos

- GPU NVIDIA com compute capability **7.5+** (RTX 20 ou mais nova)
- CUDA Toolkit 11.8+
- Rust 1.85+
- Windows: MSVC Build Tools · Linux: gcc/clang

```powershell
airun gpu     # confirma o que foi detectado
```

## Build

```powershell
cargo build --release
```

O `build.rs` acha o CUDA e o compilador host sozinho — não é preciso Developer
Prompt. Por padrão compila para sm_75/80/86/89/120 (RTX 20 até 50), o que leva
alguns minutos. Durante desenvolvimento:

```powershell
$env:AIRUN_CUDA_ARCHS="120"   # só a sua arquitetura
```

Sem CUDA instalado o build ainda funciona; os comandos de GPU informam que os
kernels não foram compilados.

---

## Comandos

```
airun run        gerar texto
airun serve      API HTTP compatível com OpenAI
airun inspect    descrever um GGUF (metadados, tensores, mistura de quantização)
airun gpu        listar GPUs, VRAM, P2P, geração
airun convert    safetensors (HuggingFace) -> GGUF
airun quantize   GGUF -> GGUF em outra precisão
airun bench      medir throughput de prefill e decode
airun perplexity pontuar um texto conhecido (perda, perplexidade, top-1)
airun selftest   validar os kernels GPU contra a referência CPU
```

Referência completa: **[docs/FLAGS.md](docs/FLAGS.md)**

---

## Exemplos

**Gerar texto, determinístico**

```powershell
airun run modelo.gguf -p "O que é RoPE?" -n 200 --temp 0
```

**Servir a API**

```powershell
airun serve modelo.gguf --port 8080 --api-key minha-chave -c 8192
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer minha-chave" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Olá"}],"max_tokens":100}'
```

**Converter e quantizar um checkpoint HuggingFace**

```powershell
airun convert G:\modelsAI\origin\Qwen3-8B -o E:\models\Qwen3-8B-Q4_K.gguf --type Q4_K
airun run E:\models\Qwen3-8B-Q4_K.gguf -p "Teste" -n 50
```

**Dividir um modelo grande entre GPUs**

```powershell
airun run modelo-grande.gguf -p "..." --tensor-split 0.6,0.4 --stats
```

Sem `--tensor-split`, o airun decide sozinho — e usa **uma** GPU só quando o
modelo cabe, porque atravessar PCIe custa mais do que a segunda placa rende.

---

## Alocação de VRAM

O planner lê os tamanhos exatos dos tensores do cabeçalho GGUF. Não estima.

```
memory plan (context 4096 tokens):
  GPU 0 NVIDIA GeForce RTX 5070 Ti layers 0..27 (28)       +embed +output
      weights   2.17 GiB  kv 448.00 MiB  scratch  60.73 MiB  =  2.66 GiB of 14.31 GiB budget
  note: whole model fits on GPU 0, keeping it on one device to avoid cross-GPU transfers
```

Quando não cabe, em vez de falhar com "out of memory", ele desce uma escada:
batch pela metade → contexto pela metade → reserva menor. E diz o que mudou.

---

## Correção numérica

Um layout de quantização sutilmente errado não trava — produz texto que parece
quase certo. Por isso cada tipo é implementado **duas vezes**: uma vez em Rust
(CPU, legível) e uma vez em CUDA. `airun selftest` compara as duas.

```
airun selftest
  dequant Q4_K    max_abs 0.000e0   max_rel 0.000e0   (16384 elements)
  matvec Q6_K     max_abs 2.560e2   error/bound 0.00x
  ...
  78 passed, 0 failed
```

Isso prova que os dois lados **concordam**, não que a operação está certa: se a
referência CPU repetir a mesma suposição errada, o teste passa e o modelo
continua quebrado. O contrapeso é `airun perplexity`, que pontua uma passagem
conhecida e mede o modelo inteiro de ponta a ponta:

```
airun perplexity E:\airun-models\Qwen3-8B-Q4_K.gguf
  159 tokens scored
    mean loss   2.2018 nats/token
    perplexity  9.04
    top-1       50.9% (81/159)
```

Os 17 decodificadores batem **bit a bit**, e o decode (token a token) roda
inteiro no caminho f32 exato — a geração é bit-idêntica à referência CPU. O
prefill troca essa identidade por velocidade de forma controlada: lotes de
2–63 tokens usam produto escalar inteiro (dp4a) sobre ativações quantizadas em
Q8, e lotes de 64+ usam tensor cores em fp16 — cada um com seu caso no
selftest, limitado por erro *backward* (proporcional a `sum|w*x|`, a métrica
correta para produto escalar: erro relativo puro é enganoso quando os termos
se cancelam) e vigiado fim-a-fim pela perplexidade. `AIRUN_MATVEC_F32=1` força
o caminho exato em tudo.

---

## Suporte

**Arquiteturas:** `qwen3`, `qwen2`, `llama` (decoders densos).

**Quantizações — leitura:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K,
Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, MXFP4, NVFP4.

**Quantizações — escrita:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K,
Q5_K, Q6_K, NVFP4.

**Ainda não:** modelos híbridos SSM/atenção (`qwen35`, `qwen35moe`), MoE,
multi-modal (`mmproj`). O loader recusa com mensagem explícita em vez de gerar
lixo. Detalhes e roteiro em [AGENTS.md](AGENTS.md).

---

## Medições neste dispositivo

RTX 5070 Ti (16 GB) + RTX 5060 Ti (16 GB), CUDA 13.2, Windows 11.

Prefill medido com lotes de 512 tokens. O decode é dado em três profundidades
de contexto porque **ele não é constante**: a atenção lê o cache inteiro a cada
token, então uma taxa medida com o contexto vazio não descreve uma conversa
longa. `airun bench --decode-at 16,1024,3072` reproduz a tabela.

| Modelo | Config | Prefill | Decode @16 | @1024 | @3072 |
|---|---|---|---|---|---|
| Qwen3-1.7B Q8_0/BF16 | 1 GPU | 5650 t/s | 234 t/s | 180 t/s | 128 t/s |
| Qwen3-1.7B Q8_0/BF16 | split 50/50 | 4146 t/s | 94 t/s | 88 t/s | 74 t/s |
| Qwen3-8B Q4_K (convertido pelo airun) | 1 GPU | 1485 t/s | 113 t/s | 95 t/s | 73 t/s |

O split entre placas é mais lento de propósito: estas GPUs não têm P2P, então
cada handoff passa pela RAM. Use uma GPU só quando o modelo couber.

A queda do decode com o contexto é a fraqueza atual. A atenção paraleliza sobre
`n_tokens * n_heads` blocos, o que no decode dá 16 blocos numa placa de 70 SMs —
a placa fica ociosa e o custo por posição de KV é dominado por latência, não por
banda. O bloco já usa 32 warps no decode (contra 8 no prefill, onde a grade
satura sozinha), o que dobrou a taxa a 3072 tokens; o próximo passo é dividir o
intervalo de KV entre blocos e combinar os parciais, como faz o
`flash_attn_ext` do llama.cpp.

---

## Documentação

- **[docs/FLAGS.md](docs/FLAGS.md)** — toda flag de todo comando
- **[AGENTS.md](AGENTS.md)** — guia para agentes de IA e novos contribuidores:
  estado atual, armadilhas conhecidas, como verificar mudanças
- **[tasks.md](tasks.md)** — estado de cada item do escopo original
