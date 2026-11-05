<h1 align="center">airun</h1>

<p align="center">
  <b>Engine de inferência para modelos <u>Qwen</u> em GPUs NVIDIA, escrita do zero em Rust + CUDA.</b><br>
  Sem llama.cpp, sem ggml, sem PyTorch, sem cuBLAS: todo kernel é deste projeto.
</p>

<p align="center">
  <a href="LICENSE"><img alt="Licença Apache 2.0" src="https://img.shields.io/badge/licen%C3%A7a-Apache%202.0-blue"></a>
  <img alt="Rust 1.85+" src="https://img.shields.io/badge/Rust-1.85%2B-orange">
  <img alt="CUDA 12/13" src="https://img.shields.io/badge/CUDA-12%20%7C%2013-76b900">
  <img alt="RTX 20 a 50" src="https://img.shields.io/badge/GPU-RTX%2020%20a%2050-76b900">
</p>

```powershell
airun run   Qwen3.8-27B-NVFP4.gguf -p "Explique o que é um kernel de GPU"
airun serve Qwen3.8-27B-NVFP4.gguf          # API compatível com llama-server, OpenAI e Anthropic
```

---

## Para quais modelos

O airun é feito para a **família Qwen**. O que decide se um arquivo roda é a
arquitetura gravada nele (`general.architecture` no GGUF; `airun inspect
modelo.gguf` mostra):

| Arquitetura | Modelos | Situação |
|---|---|---|
| **`qwen35`** | **Qwen3.5 e Qwen3.8 densos** (ex.: Qwen3.8-27B) | **Foco principal.** Tudo funciona: MTP, visão, `--parallel`, NVFP4 |
| **`qwen35moe`** | **Qwen3.5 e Qwen3.8 MoE** (ex.: Qwen3.8-35B-A3B) | Completo. MTP existe, mas vem desligado (fica mais lento em MoE) |
| `qwen4exp` | Qwen4 Flash-Next | Completo, menos `--parallel` |
| `qwen3next`, `qwen3nextmoe` | Qwen3-Next | Mesmo bloco do `qwen35` |
| `qwen3` | Qwen3 denso (0.6B a 32B) | Completo (esses modelos não têm MTP nem visão) |
| `qwen2`, `llama` | Modelos com o mesmo formato de bloco | Funcionam, mas são pouco testados |
| `qwen3moe` | Qwen3-30B-A3B, Qwen3-235B-A22B | **Ainda não** |

Os Qwen3.5/3.8/Next são **híbridos**: três de cada quatro camadas são
"delta-net" (atenção linear, com um estado de tamanho fixo em vez de cache que
cresce) e uma é atenção normal. É por isso que eles aguentam contextos de
centenas de milhares de tokens com pouca VRAM, e é nessa arquitetura que o
airun concentra as otimizações.

Outras famílias (Gemma, Llama 3+, Mistral, DeepSeek, GLM…) **não são
suportadas**: o airun recusa o arquivo com uma mensagem clara em vez de gerar
texto errado.

---

## Destaques

- **Mais rápido que o llama.cpp** no Qwen3.8-27B NVFP4: no prompt, na geração,
  com contexto de até 128 mil tokens e com decodificação especulativa (MTP).
  No Qwen4 Flash-Next (78 GB, maior que a VRAM) também passa o Strata.
- **MTP: o modelo rascunha os próximos tokens com a cabeça extra que ele já
  traz,** e o modelo principal confere vários de uma vez. No 27B a geração
  quase dobra (26 → 68 tok/s), com temperatura ou sem. A cabeça pode vir num
  arquivo separado (`mtp-*.gguf`, como o llama.cpp gera).
- **Visão:** com o `mmproj` do modelo, ele lê imagens (fotos, prints, páginas
  com texto). Funciona no `run` e na API.
- **Várias pessoas ao mesmo tempo** (`serve --parallel N`): cada conversa
  guarda o próprio cache, e as que estão gerando avançam juntas. Com 4
  pedidos simultâneos o servidor entrega 2,2 vezes mais tokens.
- **Modelos maiores que a VRAM:** enche a GPU principal, depois as outras, e
  só então usa a RAM. Num MoE, os experts menos usados ficam na RAM.
- **API igual à do `llama-server`:** OpenAI (chat, completions, responses),
  Anthropic (messages), rotas nativas do llama.cpp, tool calls, raciocínio,
  gramáticas, JSON schema, embeddings, slots e stream replay. Clientes feitos
  para o llama-server funcionam sem mudança (56 de 56 casos conferidos).
- **Correção verificada:** cada formato de quantização existe em Rust (a
  referência) e em CUDA, e `airun selftest` compara os dois (mais de 270
  casos).
- **Um binário, sem Python:** carrega GGUF, converte safetensors, quantiza,
  mede e serve.

---

## Início rápido

### Requisitos

- GPU NVIDIA RTX 20, 30, 40 ou 50 (compute capability 7.5 ou mais)
- CUDA Toolkit 12.x ou 13.x (as RTX 50 precisam do 12.8 ou mais novo)
- Rust 1.85 ou mais novo
- Windows: MSVC Build Tools 2022 · Linux: gcc ou clang

### Compilar

```powershell
git clone https://github.com/<você>/airun
cd airun
.\build.ps1      # Windows
./build.sh       # Linux
```

O script descobre as suas GPUs, compila só para elas e, no fim, roda o
`selftest` em cada uma para confirmar que está tudo certo. Opções:

| Windows | Linux | O que faz |
|---|---|---|
| `-Arch 120` | `--arch 120` | Compila para uma arquitetura escolhida (ou várias: `86,120`) |
| `-All` | `--all` | Compila para RTX 20 até 50 (para distribuir o binário) |
| `-NoGpu` | `--no-gpu` | Sem CUDA |
| `-Debug` | `--debug` | Build de depuração |
| `-SkipCheck` | `--skip-check` | Não roda o selftest no fim |

**GPUs de gerações diferentes no mesmo PC** (ex.: RTX 3060 + RTX 5060 Ti)
funcionam: o script compila para as duas, e cada placa roda o seu código.

Para compilar à mão: `$env:AIRUN_CUDA_ARCHS = "120"` (ou `75`, `80`, `86`,
`89`, ou uma lista) e `cargo build --release`.

### Gerar texto

```powershell
airun run modelo.gguf -p "O que é RoPE?"
airun run modelo.gguf -p "Resolva: 17 x 23" --reasoning-effort low     # pensa pouco
airun run modelo.gguf --mmproj mmproj.gguf --image foto.jpg -p "O que tem na imagem?"
Get-Content prompt.txt | airun run modelo.gguf --stats                   # prompt de um arquivo, com tempos
```

### Servir a API

```powershell
airun serve modelo.gguf --port 8080 --api-key minha-chave
```

Abra `http://localhost:8080/` para o chat embutido, ou use qualquer cliente
OpenAI, Anthropic ou llama.cpp:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer minha-chave" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Olá!"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="minha-chave")
r = client.chat.completions.create(model="airun", messages=[{"role": "user", "content": "Olá!"}])
print(r.choices[0].message.content)
```

Sem `-c`, o contexto é o maior que cabe nas GPUs com todos os pesos na VRAM.

---

## Resultados

Cada medição tem script, saída bruta e relatório em [`benchmarks/`](benchmarks/).
As GPUs destes testes **não têm P2P**: o que passa de uma placa para a outra
vai pela RAM. Números em tokens por segundo.

### Qwen3.8-27B NVFP4, uma GPU (RTX 5070 Ti)

Melhor de 3 rodadas alternadas
([`benchmarks/2026-11-01`](benchmarks/2026-11-01/RESULTADOS.md)):

| Teste | airun | llama.cpp b11760 | |
|---|---:|---:|---:|
| prompt de 512 tokens | **3797** | 3472 | +9% |
| prompt de 2048 tokens | **3690** | 3652 | +1% |
| geração | **49,0** | 47,9 | +2% |
| geração com MTP | **113,7** | 104,6 | +9% |

### Contexto longo, código de verdade (RTX 5060 Ti + RTX 3060)

O 27B dividido entre as duas placas, com prompts tirados de 1 MB de código
C++/CUDA:

| Teste | airun | llama.cpp b11760 | |
|---|---:|---:|---:|
| prompt de 8 mil tokens | **1150** | 1079 | +7% |
| prompt de 32 mil | **1039** | 987 | +5% |
| prompt de 128 mil | **745** | 712 | +5% |
| geração com 8 mil de contexto | **22,4** | 20,1 | +12% |
| geração com 128 mil de contexto | 13,4 | 13,5 | empate |

Um pedido real (o arquivo `ggml-cuda.cu` inteiro, 61.634 tokens, mais uma
pergunta sobre ele) leu o prompt a 899 tok/s e respondeu a 19,8 tok/s, com
95% dos rascunhos do MTP aceitos. A resposta estava certa.

### Qwen4 Flash-Next (MoE de 78 GB, maior que as duas GPUs juntas)

Prompt de 792 tokens, prompt / geração
([`benchmarks/2026-11-01d`](benchmarks/2026-11-01d/RESULTADOS.md)):

| | airun | Strata 0.1.55 | llama.cpp |
|---|---:|---:|---:|
| prompt / geração | **418 / 40,7** | 316 / 34,4 | 36 / 9,9 |

Os experts que não cabem na VRAM ficam na RAM; o airun os traz pela PCIe
enquanto a GPU calcula os outros.

### Quando o airun perde

Quando o modelo **não cabe** na VRAM de jeito nenhum e camadas inteiras ficam
na RAM (ex.: o 27B numa GPU de 8 GB), o llama.cpp é mais rápido: ele calcula
essas camadas na CPU, e o airun as copia pela PCIe a cada token (medido: 0,87
contra 0,4 tok/s). As vantagens acima valem com o modelo na VRAM.

---

## A API

Rotas, campos e respostas seguem os do `llama-server`, então o web UI dele e
os SDKs da OpenAI e da Anthropic funcionam sem adaptação.
[`tests/api_conformance.py`](tests/api_conformance.py) compara as respostas
com as do llama-server, campo a campo.

| Rota | Para quê |
|---|---|
| `/v1/chat/completions` | Chat da OpenAI: stream, tool calls, raciocínio, logprobs, saída em JSON |
| `/v1/responses` | Responses API da OpenAI |
| `/v1/messages` | Messages API da Anthropic |
| `/v1/completions`, `/completion` | Completar texto (OpenAI e o formato nativo do llama.cpp) |
| `/infill` | Completar código no meio de um arquivo (FIM) |
| `/v1/embeddings`, `/embedding` | Embeddings (com `--embeddings`) |
| `/tokenize`, `/detokenize`, `/apply-template` | Utilitários |
| `/v1/stream/{id}` | Reler uma geração que continuou depois que o cliente caiu |
| `/slots`, `/health`, `/props`, `/v1/models` | Estado do servidor |

- **Imagens:** com `--mmproj`, mande a imagem como data URL (OpenAI) ou base64
  (Anthropic).
- **Raciocínio:** `reasoning_effort` (`none`, `low`, `medium`, `high`, `max`)
  controla quanto o modelo pensa; `reasoning_budget_tokens` dá o limite exato.
- **Amostragem:** a mesma cadeia do llama.cpp (penalidades, DRY, top-k, top-p,
  min-p, XTC, mirostat…), com `seed` e `logit_bias`.
- **Saída restrita:** gramática GBNF, JSON schema ou `response_format`. Num
  modelo que pensa antes de responder, a regra vale só para a resposta.

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Invente uma pessoa."}],
  "response_format": {"type": "json_schema", "json_schema": {"schema": {
    "type": "object", "properties": {"nome": {"type": "string"}, "idade": {"type": "integer"}},
    "required": ["nome", "idade"]}}}}'
```

---

## Formatos

| | |
|---|---|
| **Quantizações lidas** | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4 |
| **Quantizações escritas** | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, NVFP4 |
| **KV cache** | f16, q8_0, q4_0 (`--kv-type`; q4_0 cabe ~3,6x mais contexto) |
| **Entrada** | GGUF; safetensors do HuggingFace via `airun convert` |
| **Visão** | `mmproj` dos Qwen3.5/3.8 (projetor `qwen3vl_merger`) |

**Ainda não:** áudio, LoRA, rerank, vídeo e a arquitetura `qwen3moe`.

**NVFP4 vindo de outras ferramentas** (vllm, NVIDIA) guarda um fator de escala
num arquivo ao lado de cada peso (`<nome>.scale`). O airun lê e aplica esse
fator; sem ele, o modelo gera lixo.

---

## Por que é rápido

- **NVFP4 nos tensor cores FP4** das RTX 50. Nas placas anteriores roda em
  int8, com o mesmo resultado (e ainda à frente do llama.cpp).
- **A delta-net dos Qwen3.5/3.8 em blocos nos tensor cores** durante o prompt,
  em vez de token a token (+8% no prompt).
- **Decode em grafos CUDA encadeados:** o próximo passo já está na fila antes
  de o processador ler o token atual, então a GPU não fica esperando.
- **MTP**, com temperatura ou sem.
- **Várias GPUs trabalhando ao mesmo tempo no prompt:** a primeira placa já
  começa o próximo pedaço enquanto a segunda termina o atual.
- **Um planejador de VRAM** que lê o tamanho exato de cada tensor:
  - se o modelo cabe numa GPU, usa uma só (passar pela PCIe sem P2P custa caro);
  - se não cabe e as GPUs são diferentes, enche primeiro a mais rápida;
  - se são iguais, divide por igual (o prompt fica quase duas vezes mais rápido);
  - se nem assim cabe, reduz lote, contexto ou reserva, e avisa o que mudou.

---

## Como a correção é garantida

Um formato de quantização decodificado errado não trava o programa: ele gera
texto que *parece* quase certo. Por isso:

- **Cada formato existe duas vezes:** em Rust, simples de ler (a referência),
  e em CUDA. `airun selftest` compara os dois kernel a kernel.
- **`airun perplexity`** mede o modelo inteiro num texto conhecido. Pega os
  erros sutis que o selftest não vê.
- **`AIRUN_MATVEC_F32=1 AIRUN_NO_FP4=1`** forçam o cálculo exato em f32. No
  27B o caminho normal fica a 0,01 dele (em logprob); o llama.cpp fica a 0,24.

---

## Comandos

```
airun run         gerar texto
airun serve       servidor HTTP (OpenAI + Anthropic + llama.cpp)
airun bench       medir prompt e geração, em várias profundidades de contexto
airun perplexity  medir a qualidade num texto conhecido
airun inspect     descrever um GGUF: arquitetura, tensores, quantização
airun tokenize    ver como o tokenizer divide um texto
airun convert     safetensors (HuggingFace) -> GGUF
airun quantize    GGUF -> GGUF em outra precisão
airun gpu         listar as GPUs, VRAM e P2P
airun selftest    conferir os kernels da GPU contra a referência em Rust
```

Todas as opções: **[docs/FLAGS.md](docs/FLAGS.md)**.

---

## Estrutura

```
crates/
  airun-gguf/   leitura e escrita de GGUF e safetensors; quantização de referência
  airun-cuda/   os kernels CUDA (kernels/*.cu) e o código Rust que os chama
  airun-core/   tokenizer, planejador de VRAM, runtime, amostragem, gramáticas, MTP, visão
  airun-cli/    o binário: run, serve, bench, perplexity, convert...
tests/          conformidade da API contra o llama-server
benchmarks/     cada medição publicada, com script e saída bruta
docs/           referência das opções
```

## Documentação

- **[docs/FLAGS.md](docs/FLAGS.md):** todas as opções de todos os comandos.
- **[AGENTS.md](AGENTS.md):** como o código está organizado, o que já foi
  medido e as armadilhas conhecidas. Leia antes de mudar algo.
- **[CONTRIBUTING.md](CONTRIBUTING.md):** regras, ambiente e como verificar
  uma mudança.

## Licença

[Apache 2.0](LICENSE). Os codebooks IQ vêm do ggml, e a seleção de blocos da
atenção esparsa segue a do Strata, ambos sob MIT; os avisos estão em
[NOTICE](NOTICE).
