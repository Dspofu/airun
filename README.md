<h1 align="center">airun</h1>

<p align="center">
  <b>Engine de inferência de LLMs para GPUs NVIDIA, escrita do zero em Rust + CUDA.</b><br>
  Sem llama.cpp, sem ggml, sem PyTorch, sem cuBLAS. Cada kernel é deste projeto.
</p>

<p align="center">
  <a href="LICENSE"><img alt="Licença Apache 2.0" src="https://img.shields.io/badge/licen%C3%A7a-Apache%202.0-blue"></a>
  <img alt="Rust 1.85+" src="https://img.shields.io/badge/Rust-1.85%2B-orange">
  <img alt="CUDA 12/13" src="https://img.shields.io/badge/CUDA-12%20%7C%2013-76b900">
  <img alt="RTX 20 a 50" src="https://img.shields.io/badge/GPU-RTX%2020%20a%2050-76b900">
</p>

```powershell
airun serve Qwen3.8-27B-NVFP4.gguf          # API no formato do llama-server, OpenAI e Anthropic
airun run   Qwen3.8-27B-NVFP4.gguf -p "Explique o que é um kernel de GPU" -n 200
```

---

## Destaques

- **Mais rápida que o llama.cpp** no Qwen3.8-27B NVFP4 em todos os testes:
  prompt de 512 a 131.072 tokens, geração com 16 a 131.072 de contexto e
  decodificação especulativa com MTP. E
  **mais rápida que o Strata e o llama.cpp** no Qwen4 Flash-Next (MoE de
  78 GB numa máquina com 32 GB de VRAM). Com contexto longo, a vantagem chega
  a +52% no prompt de 128k tokens e +15% na geração.
- **Modelos maiores que a VRAM:** `--alloc auto` (o padrão) enche a GPU
  principal, depois as outras, e só então usa a RAM para o que sobrou.
  - **Modelo denso:** camadas inteiras ficam em RAM fixada e são copiadas
    para a VRAM antes de rodar, com o contexto mantido e resultado idêntico.
  - **MoE:** os experts menos usados ficam em RAM e são lidos pela PCIe,
    guiados por um perfil de roteamento que a airun aprende e guarda por
    modelo.
  - **Outros modos:** `--alloc gpu` usa só a VRAM, e `--main-gpu` escolhe a
    principal.
- **API completa no formato do `llama-server`:** OpenAI (chat, completions,
  responses), Anthropic (messages), `/completion` e `/infill` nativos,
  tool calls, reasoning, logprobs, gramáticas GBNF e JSON schema, embeddings,
  slots. Um cliente feito para o llama-server funciona sem mudança, e o teste
  de conformidade confere isso chave a chave (56/56).
- **Correção verificada, não presumida:** cada formato de quantização existe
  em Rust (referência) e em CUDA, e `airun selftest` compara os dois (274
  casos). No caminho normal os logprobs ficam a 0,01 do cálculo exato em f32.
- **Um binário, sem Python:** carrega GGUF direto, converte safetensors,
  quantiza, mede e serve.

---

## Resultados

Máquina: RTX 5070 Ti 16 GB (PCIe 4.0 x16) + RTX 5060 Ti 16 GB (PCIe 3.0 x4),
64 GB de RAM, Windows 11, CUDA 13. As placas **não têm P2P**: o que cruza de
uma para a outra passa pela RAM. Cada medição tem script, saída bruta e
relatório em [`benchmarks/`](benchmarks/).

### Qwen3.8-27B NVFP4 contra o llama.cpp

Uma GPU, melhor de 3 rodadas alternadas, tok/s
([`benchmarks/2026-11-01`](benchmarks/2026-11-01/RESULTADOS.md)):

| Teste | airun | llama.cpp b11760 | |
|---|---:|---:|---:|
| prompt de 512 tokens | **3797** | 3472 | +9% |
| prompt de 2048 tokens | **3690** | 3652 | +1% |
| geração | **49,0** | 47,9 | +2% |
| geração com MTP, prompt 1 | **113,7** | 104,6 | +9% |
| geração com MTP, prompt 2 | **94,9** | 91,7 | +3% |

**Condição da vantagem no decode:** os números acima são de uma GPU (ou das
duas, na seção de contexto longo) onde o 27B **cabe na VRAM** — o decode é
limitado pela banda de memória, e o airun extrai mais dos matvecs quantizados
e do pipeline de grafos. Quando o modelo **não cabe** (ex.: 27B num GPU de 8 GB),
o gargalo muda: o airun mantém as camadas excedentes em RAM fixada e as lê pela
PCIe a cada token, enquanto o llama.cpp as derrama nos núcleos da CPU. Nesse
caso o llama.cpp fica à frente no decode (medido: 0,87 vs 0,4 t/s) e também no
prefill (kernels de matmul denso mais otimizados). A vantagem do airun se
aplica, portanto, a **decode com o modelo residente na VRAM** — o regime
multi-GPU/longo-contexto acima e o MoE que cabe numa placa.

### Contexto longo

Mesmo modelo dividido entre as duas GPUs (com 32k+ de contexto ele não cabe
numa placa), lotes de prompt de 512, melhor de 2 rodadas, tok/s
([`benchmarks/2026-11-01g`](benchmarks/2026-11-01g/RESULTADOS.md)):

| Teste | airun | llama.cpp b11760 | |
|---|---:|---:|---:|
| prompt de 8k tokens | **3694** | 3496 | +6% |
| prompt de 32k tokens | **3055** | 2862 | +7% |
| prompt de 128k tokens | **2428** | 1597 | +52% |
| geração @ 16 de contexto | **42,1** | 32,7 | +29% |
| geração @ 32k | **37,4** | 29,1 | +29% |
| geração @ 128k | **23,5** | 20,4 | +15% |

No início da noite o prompt longo perdia de 28% a 37%
([antes](benchmarks/2026-11-01e/RESULTADOS.md)). Três mudanças viraram o
jogo:

- **GPUs em paralelo no prefill:** as duas placas trabalham ao mesmo tempo,
  cada uma num pedaço do prompt.
- **Atenção do prefill:** subiu de 55 para 73 TFLOPS.
- **O planner enche a GPU mais rápida primeiro,** em vez de dividir o modelo
  ao meio entre uma placa rápida e uma com metade da banda.

### Qwen4 Flash-Next (MoE, IQ3_S, 78 GB) contra Strata e llama.cpp

Prompt de 792 tokens, 128 tokens gulosos, mesmo tamanho de bloco de prompt
nos três motores, prompt / geração em tok/s
([`benchmarks/2026-11-01d`](benchmarks/2026-11-01d/RESULTADOS.md)):

| Bloco | airun | Strata 0.1.55 | llama.cpp |
|---|---:|---:|---:|
| 1024 | **449 / 39,4** | 427 / 33,9 | 36 / 9,9 |
| 512 | **418 / 40,7** | 316 / 34,4 | 36 / 9,9 |

O modelo não cabe nas duas GPUs juntas. O Strata calcula na CPU os experts que
estão fora da VRAM; o llama.cpp os deixa na CPU (`--n-cpu-moe`). A airun os
mantém em RAM fixada e a GPU os lê pela PCIe num stream paralelo ao dos experts
residentes.

---

## Início rápido

### Requisitos

- GPU NVIDIA com compute capability **7.5+** (RTX 20, 30, 40 ou 50)
- CUDA Toolkit 12.x ou 13.x (RTX 50 precisa de 12.8+)
- Rust 1.85+
- Windows: MSVC Build Tools 2022 · Linux: gcc ou clang

### Build

```powershell
git clone https://github.com/<você>/airun
cd airun
.\build.ps1                          # Windows: detecta as GPUs e compila só para elas
./build.sh                           # Linux: o mesmo, com --arch, --all, --no-gpu, --debug, --skip-check
# ou manualmente:
$env:AIRUN_CUDA_ARCHS = "86"        # 75, 80, 86, 89 ou 120
cargo build --release
.\target\release\airun.exe gpu        # o que foi detectado
.\target\release\airun.exe selftest   # cada kernel contra a referência CPU
```

O `build.rs` acha o CUDA e o compilador host sozinho. Por padrão compila para
RTX 20 até 50, o que leva alguns minutos; durante o desenvolvimento, compile só
para a sua placa com `$env:AIRUN_CUDA_ARCHS="120"` (ou 75, 80, 86, 89).
`build.ps1` (Windows) e `build.sh` (Linux) automatizam isso e aceitam
`-Arch 120` / `--arch 120`, `-All` / `--all` (universal, para distribuir),
`-NoGpu`, `-Debug` e `-SkipCheck`; ao final rodam `gpu` e o `selftest` em cada
GPU para confirmar.

**GPUs de gerações diferentes no mesmo PC** (ex.: RTX 3060 + RTX 5060 Ti): o
script detecta todas e compila para a união das arquiteturas (`86,120`); cada
placa roda o código da sua. À mão: `AIRUN_CUDA_ARCHS="86,120"`. Os kernels
NVFP4 de tensor core FP4 só existem nas RTX 50 e só entram se `120` estiver na
lista; nas outras placas o NVFP4 roda no caminho int8, com o mesmo resultado.

### Gerar texto

```powershell
airun run modelo.gguf -p "O que é RoPE?" -n 300 --temp 0
Get-Content prompt.txt | airun run modelo.gguf -n 500 --stats   # prompt pela entrada padrão, com tempos
```

### Servir a API

```powershell
airun serve modelo.gguf --port 8080 -c 32768 --api-key minha-chave
```

Abra `http://localhost:8080/` para o chat embutido, ou use qualquer cliente
OpenAI, Anthropic ou llama.cpp:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer minha-chave" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Olá!"}],"max_tokens":200}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="minha-chave")
r = client.chat.completions.create(model="airun", messages=[{"role": "user", "content": "Olá!"}])
print(r.choices[0].message.content)
```

---

### Contexto longo com código de verdade

27B NVFP4 numa RTX 5060 Ti + RTX 3060 (arquiteturas diferentes no mesmo PC),
prompts tirados de 1 MB de código C++/CUDA do llama.cpp, tok/s:

| Teste | airun | llama.cpp b11760 | |
|---|---:|---:|---:|
| prompt de 8k | **1150** | 1079 | +7% |
| prompt de 32k | **1039** | 987 | +5% |
| prompt de 64k | **910** | 874 | +4% |
| prompt de 128k | **745** | 712 | +5% |
| geração @ 8k | **22,4** | 20,1 | +12% |
| geração @ 64k | **16,9** | 16,4 | +3% |
| geração @ 128k | 13,4 | 13,5 | empate |

Um pedido real de ponta a ponta (o `ggml-cuda.cu` inteiro, 61.634 tokens, e
uma pergunta sobre ele) com MTP: prompt a 899 tok/s, resposta a 19,8 tok/s
com 95% dos rascunhos aceitos.

## A API

Rotas, campos e respostas seguem os do `llama-server`, para que o web UI dele,
os SDKs da OpenAI e da Anthropic e agentes que chamam ferramentas funcionem
sem adaptação. [`tests/api_conformance.py`](tests/api_conformance.py) grava as
respostas de dois servidores e as compara chave a chave: 56 de 56 casos iguais
ao llama-server b11760.

| Rota | O quê |
|---|---|
| `/v1/chat/completions` | Chat OpenAI: stream, tool calls, `reasoning_content`, logprobs, `response_format` |
| `/v1/responses` | OpenAI Responses API, com stream e chamadas de função |
| `/v1/messages` | Anthropic Messages API: blocos `thinking`, `tool_use`, stream |
| `/v1/completions`, `/completion` | Completions OpenAI e nativa do llama.cpp, com `n_probs` |
| `/infill` | Preenchimento no meio (FIM) para código |
| `/v1/embeddings`, `/embedding` | Embeddings do estado final, com pooling configurável (`--embeddings`) |
| `/tokenize`, `/detokenize`, `/apply-template` | Utilitários |
| `…/input_tokens`, `/v1/messages/count_tokens` | Contagem de tokens de um pedido |
| `/slots/0?action=save\|restore\|erase` | Salva e retoma o cache de uma conversa (`--slot-save-path`) |
| `/v1/chat/completions/control` | Encerra o raciocínio de uma geração em andamento |
| `/health`, `/props`, `/v1/models`, `/slots` | Estado e metadados |
| `/v1/stream/{id}`, `/v1/streams/lookup` | Stream replay: a geração sobrevive ao cliente cair e é relida de onde parou |

**Imagens:** com `--mmproj`, o chat aceita imagens nos formatos da OpenAI, da
Responses e da Anthropic (data URL / base64), e o `run` aceita `--image`.

**Amostragem:** a cadeia inteira do llama.cpp, na mesma ordem: penalidades,
DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC, temperatura dinâmica e
mirostat 1/2, com `logit_bias` e `seed`.

**Saída restrita:** `grammar` (GBNF), `json_schema` e `response_format`. Num
modelo que pensa antes de responder, a gramática espera o fim do pensamento e
restringe só a resposta.

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Invente uma pessoa."}],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {
    "type": "object", "properties": {"nome": {"type": "string"}, "idade": {"type": "integer"}},
    "required": ["nome", "idade"]}}}}'
```

**Raciocínio sob controle:** `reasoning_budget_tokens` limita quantos tokens o
modelo pensa; com `reasoning_control: true`, um `POST` em `/control` encerra o
pensamento no meio do stream e o modelo passa direto para a resposta.

---

## Modelos e formatos

| | |
|---|---|
| **Arquiteturas** | `qwen3`, `qwen2`, `llama` (densos) · `qwen35`, `qwen35moe`, `qwen3next` (híbridos delta-net/atenção, MoE) · `qwen4exp` (Qwen4 Flash-Next: hyper-connections, PLE, atenção esparsa QSA) |
| **Quantizações lidas** | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4 |
| **Quantizações escritas** | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, NVFP4 |
| **KV cache** | f16, q8_0, q4_0 (`--kv-type`) |
| **Entrada** | GGUF; safetensors do HuggingFace via `airun convert` |

**Ainda não:** visão (`mmproj`), áudio, LoRA e rerank. O loader recusa um
modelo que não suporta com uma mensagem explícita em vez de gerar lixo.

**NVFP4 em duas convenções.** O encoder próprio é autocontido (o bloco carrega
as escalas UE4M3 e não tem tensor-irmão). Arquivos NVFP4 de fora (vllm/NVIDIA)
usam a convenção *global-scale*: cada peso tem um sidecar `<base>.scale` F32[1]
(≈1e‑4) que o loader lê e aplica — sem ele, os pesos saem ~10⁴ maiores e o
modelo degrada a NaN. As duas convenções funcionam; a escala é um no‑op nos
outros formatos.

---

## Como ela é rápida

- **NVFP4 em tensor cores FP4** nas RTX 50 (sm_120), com fallback exato nas
  placas anteriores.
- **Decode em grafos CUDA** encadeados: o passo seguinte entra na fila antes de
  o host ler o token do passo atual, então a GPU não espera o host.
- **Decodificação especulativa com a cabeça MTP** do próprio modelo.
- **MoE agrupado em tensor cores** no prefill; no decode, os experts ausentes
  são copiados num stream lateral enquanto os residentes calculam.
- **Perfil de roteamento persistente:** os experts mais usados de cada modelo
  começam na VRAM, e o rebalanceamento só troca com margem.
- **Atenção esparsa QSA** do Qwen4: cada consulta lê os blocos de KV que o
  indexador escolhe, em vez do cache inteiro.
- **GPUs em pipeline no prefill:** com o modelo dividido, a primeira placa
  processa o pedaço seguinte do prompt enquanto a segunda termina o atual.
  A passagem entre elas é ordenada só por eventos CUDA, mesmo sem P2P.
- **Flash attention própria** em `mma.sync`. Com contexto longo, P·V é
  somado em fp16 por tile de chaves (no máximo 32 termos) e acumulado em
  fp32, o que dá a taxa dupla do tensor core sem a imprecisão de guardar o
  acumulador inteiro em fp16.
- **Planejador de VRAM** que lê o tamanho exato de cada tensor do GGUF:
  - **Uma GPU só** quando o modelo cabe nela, porque cruzar PCIe sem P2P custa
    mais do que a segunda placa rende.
  - **Quando não cabe, enche a GPU mais rápida primeiro.** O decode passa
    pelas placas uma depois da outra, e a lenta deve ficar só com o resto.
    Isso deu +28% no decode do 27B em relação a dividir pela VRAM livre.
  - **Quando nem assim cabe,** reduz lote, contexto ou reserva, nessa ordem,
    e diz o que mudou.

---

## Correção numérica

Um layout de quantização errado não trava: ele produz texto que parece quase
certo. Por isso:

- **Cada formato existe duas vezes,** em Rust (CPU, legível) e em CUDA.
  `airun selftest` compara os dois kernel a kernel (274 casos), com limites
  de erro proporcionais a `sum|w*x|`, a medida certa para produtos escalares.
- **`airun perplexity`** mede o modelo inteiro de ponta a ponta numa passagem
  conhecida. É o que pega o erro sutil que o selftest não vê.
- **`AIRUN_MATVEC_F32=1 AIRUN_NO_FP4=1`** forçam o caminho exato em f32, que
  serve de referência: no Qwen3.8-27B o caminho normal fica a 0,01 dele em
  logprob, onde o llama.cpp se afasta 0,24 no mesmo prompt.
- **NVFP4 global-scale (bug de 2026-11-03):** o selftest não lê bytes reais
  nem sidecars, então um fator global em falta passa nele — o 27B vllm dava
  perplexidade NaN e geração `!!!!` com o selftest verde. A métrica que pega
  é a `airun perplexity` (NaN → 3.93 depois do fix).

---

## Comandos

```
airun run         gerar texto
airun serve       API HTTP no formato do llama-server (OpenAI + Anthropic + nativa)
airun bench       medir prefill e decode, em várias profundidades de contexto
airun perplexity  pontuar um texto conhecido (perda, perplexidade, top-1)
airun inspect     descrever um GGUF: metadados, tensores, mistura de quantização
airun tokenize    ver como o tokenizer divide um texto
airun convert     safetensors (HuggingFace) -> GGUF
airun quantize    GGUF -> GGUF em outra precisão
airun gpu         listar GPUs, VRAM, P2P e geração
airun selftest    validar os kernels GPU contra a referência CPU
```

Todas as flags: **[docs/FLAGS.md](docs/FLAGS.md)**.

---

## Estrutura

```
crates/
  airun-gguf/   leitura e escrita de GGUF e safetensors; quantização de referência em Rust
  airun-cuda/   os kernels CUDA (kernels/*.cu) e a camada Rust que os lança
  airun-core/   tokenizer, planejador de VRAM, runtime, amostragem, gramáticas, MTP
  airun-cli/    o binário airun: run, serve (API), bench, perplexity, convert...
tests/          conformidade da API contra o llama-server
benchmarks/     cada medição publicada, com script e saída bruta
docs/           referência de flags
```

---

## Documentação e contribuição

- **[docs/FLAGS.md](docs/FLAGS.md)**: cada flag de cada comando.
- **[AGENTS.md](AGENTS.md)**: o estado verificado do projeto, o mapa do
  código e as armadilhas conhecidas. Leia antes de mudar algo.
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: regras, ambiente, como verificar uma
  mudança e como publicar um benchmark.

## Licença

[Apache 2.0](LICENSE). Os codebooks IQ vêm do ggml, e a seleção de blocos da
QSA segue a do Strata, ambos sob MIT; os avisos estão em [NOTICE](NOTICE).
