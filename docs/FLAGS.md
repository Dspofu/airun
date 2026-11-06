# Opções do airun

Todas as opções de todos os comandos. `airun <comando> --help` mostra a versão
curta.

```
airun run         gerar texto
airun serve       servidor HTTP (OpenAI + Anthropic + llama.cpp)
airun bench       medir prompt e geração
airun perplexity  medir a qualidade num texto conhecido
airun inspect     descrever um arquivo GGUF
airun gpu         listar as GPUs
airun convert     safetensors (HuggingFace) -> GGUF
airun quantize    GGUF -> GGUF em outra precisão
airun selftest    conferir os kernels da GPU contra a referência em Rust
```

---

## Memória, GPUs e modelo (valem para `run`, `serve` e `bench`)

| Opção | Padrão | O que faz |
|---|---|---|
| `-c`, `--ctx-size <N>` | automático | Tamanho do contexto, em tokens. É o que mais gasta VRAM depois dos pesos. Sem a opção, veja [Contexto automático](#contexto-automático). |
| `-b`, `--batch-size <N>` | `512` | Quantos tokens do prompt são processados por vez. Maior = prompt mais rápido e mais VRAM. Não muda a geração. |
| `--kv-type <tipo>` | `f16` | Formato do cache: `f16`, `q8_0` ou `q4_0`. Os dois últimos cabem 2x e 3,6x mais contexto, com perda de qualidade muito pequena. |
| `--gpus <lista>` | todas | Quais GPUs usar, ex.: `--gpus 0,1`. O número é o do CUDA, que pode não bater com o do `nvidia-smi`: confira com `airun gpu`. |
| `--main-gpu <N>` | a mais rápida | A GPU que é enchida primeiro. Escolha a que está no slot PCIe mais rápido. |
| `--alloc <modo>` | `auto` | Onde os pesos podem ficar. `auto`: nas GPUs necessárias e, só se faltar, na RAM. `gpu`: só na VRAM (se não couber, reduz o contexto ou dá erro). `memory`: em todas as GPUs e depois na RAM. |
| `--tensor-split <frações>` | — | Divide as camadas à mão, ex.: `0.7,0.3`. Desliga a divisão automática. |
| `--single-gpu <N>` | — | Põe o modelo inteiro numa GPU. |
| `--vram-reserve <MiB>` | `384` | VRAM deixada livre em cada GPU (o Windows não entrega os últimos megabytes). |
| `--no-fallback` | desligado | Dá erro em vez de reduzir lote ou contexto para caber. Útil em testes automáticos. |
| `--expert-cache <N>` | todos | Só MoE: quantos experts por camada ficam na VRAM; os outros são lidos do arquivo quando o modelo os pede. |
| `--draft <N>` | veja [MTP](#mtp-geração-mais-rápida) | Quantos tokens a cabeça MTP rascunha por passo. `0` desliga. |
| `--mtp <arquivo>` | — | A cabeça MTP num arquivo separado. Veja [MTP](#mtp-geração-mais-rápida). |
| `--mmproj <arquivo>` | — | O codificador de visão do modelo. Veja [Imagens](#imagens). |
| `--image-max-tokens <N>` | `1024` | Máximo de tokens que uma imagem pode ocupar. |

### Contexto automático

Sem `-c`, `run` e `serve` usam **o maior contexto que cabe nas GPUs com todos
os pesos na VRAM**, no mínimo 4096 e no máximo o contexto de treino do modelo.
Um modelo que cabe numa GPU continua numa GPU: o airun não espalha o modelo
por duas placas só para ganhar contexto. Se nem os pesos cabem (um MoE com
experts na RAM), fica em 4096. `bench` e `perplexity` usam 4096.

### MTP (geração mais rápida)

Os Qwen3.5/3.8 trazem uma camada extra, a "cabeça MTP", que adivinha os
próximos tokens. O airun usa essa cabeça para rascunhar alguns tokens, e o
modelo principal confere todos de uma vez. Os que ele aprova saem de graça.

- **Padrão:** `run` usa 4 rascunhos nos modelos densos. Nos MoE fica desligado,
  porque conferir 4 tokens acorda muito mais experts e ficou mais lento
  (35B-A3B: 115 → 91 tok/s); `--draft 3` liga mesmo assim. `serve` usa 0.
- **Ganho:** no 27B, 26 → 68 tok/s em modo guloso (`--temp 0`) e 25,7 → 47,2
  tok/s com `--temp 0.7`.
- **Funciona com qualquer amostragem menos mirostat.** Com temperatura, cada
  rascunho é aceito com a probabilidade que o modelo lhe dá, então o texto
  segue a mesma distribuição de antes; o MTP só muda a velocidade.
- **Em modo guloso** o texto pode, raramente, sair diferente do decode sem MTP:
  a conferência soma os números em outra ordem, e num quase-empate a escolha
  vira.
- **Cabeça em arquivo separado:** o conversor do llama.cpp pode gravar o modelo
  sem a cabeça (`--no-mtp`) e a cabeça num `mtp-<nome>.gguf` ao lado. O airun
  acha esse arquivo sozinho na pasta do modelo (confere se é do mesmo modelo),
  ou use `--mtp <arquivo>`.
- **Sem MTP** quando há imagem no prompt ou com `serve --parallel` acima de 1.
- Quando o MTP não pode rodar numa geração, o airun imprime uma `note:` dizendo
  por quê.

### MoE: o que é automático

- **Experts na RAM:** um MoE que não cabe nas GPUs fica com os experts mais
  usados na VRAM e o resto na RAM. A GPU lê da RAM os que faltam, num canal
  paralelo enquanto calcula os outros.
- **Perfil de uso:** o airun anota quais experts cada camada usa e guarda em
  `%LOCALAPPDATA%\airun\profiles\` (ou `~/.cache/airun/profiles/`). Na
  próxima vez os mais usados já começam na VRAM. `AIRUN_EXPERT_PROFILE=off`
  desliga.

### Como as camadas são divididas entre as GPUs

1. O airun lê o tamanho **exato** de cada tensor no arquivo e soma pesos,
   cache e memória de trabalho.
2. **Se cabe numa GPU, usa uma só** (a `--main-gpu`, ou a mais rápida). Passar
   dados pela PCIe sem P2P custa mais do que a segunda placa rende.
3. Se não cabe:
   - **GPUs diferentes:** enche a mais rápida primeiro e dá à outra só o que
     sobrar. Na geração as placas trabalham uma depois da outra, então cada
     camada na placa lenta custa mais. (27B numa 5070 Ti + 5060 Ti: 33,7 →
     43,0 tok/s em comparação a dividir ao meio.)
   - **GPUs iguais** (até 10% de diferença): divide ao meio. A geração fica
     igual e o prompt quase dobra, porque as duas trabalham juntas.
4. Se nem todas as GPUs juntas comportam o modelo, as camadas que sobram ficam
   na RAM e são copiadas para a GPU antes de rodar. O resultado é idêntico,
   mas fica lento: cada uma dessas camadas passa pela PCIe a cada token (27B
   com 12 camadas na RAM: 45 → 9 tok/s). Com `--alloc gpu` isso não acontece.

`--stats` mostra o que foi decidido.

### Quando não cabe

O airun tenta nesta ordem e avisa o que mudou:

1. **Lote pela metade** (até 64): só o prompt fica mais lento.
2. **Contexto pela metade** (até 512).
3. **Reserva de VRAM em 128 MiB:** último recurso.

`--no-fallback` desliga essas reduções.

---

## `airun run`

Gera texto e escreve no stdout.

### Prompt

| Flag | Padrão | O que faz |
|---|---|---|
| `-p`, `--prompt <texto>` | stdin | O prompt. Sem esta flag, lê do stdin. |
| `--system <texto>` | — | Mensagem de sistema, inserida antes do prompt. |
| `--raw` | desligado | Envia o prompt exatamente como está, sem template de chat. Use quando estiver montando o ChatML você mesmo. |
| `-n`, `--predict <N>` | `256` | Máximo de tokens a gerar. |

### Amostragem

| Flag | Padrão | O que faz |
|---|---|---|
| `--temp <f>` | `0.8` | Temperatura. **`0` = greedy determinístico.** |
| `--top-k <N>` | `40` | Mantém só os N tokens mais prováveis. `0` desliga. |
| `--top-p <f>` | `0.95` | Nucleus sampling. `1.0` desliga. |
| `--min-p <f>` | `0.05` | Descarta tokens abaixo desta fração da probabilidade do melhor token. `0` desliga. |
| `--typical-p <f>` | `1.0` | Amostragem localmente típica: mantém os tokens cuja surpresa está mais perto da entropia da distribuição, não os mais prováveis. `1.0` desliga. |
| `--top-n-sigma <f>` | `0.0` | Mantém só os logits a até N desvios-padrão do melhor. `0` desliga. Ao contrário do top-p, o corte **não** se alarga com a temperatura — é a propriedade pela qual o filtro foi proposto. |
| `--repeat-penalty <f>` | `1.1` | Penalidade multiplicativa em tokens recentes. `1.0` desliga. Aplicada **uma vez por token distinto** na janela, não uma vez por ocorrência. |
| `--presence-penalty <f>` | `0.0` | Subtraído uma vez de qualquer token já visto. Escala aditiva da OpenAI. |
| `--frequency-penalty <f>` | `0.0` | Subtraído por ocorrência do token. Escala aditiva da OpenAI. |
| `--repeat-last-n <N>` | `64` | Janela que as três penalidades enxergam. |
| `--seed <N>` | `0` | Semente do RNG. `0` sorteia uma. Um valor fixo torna a saída reproduzível. |

Os filtros são aplicados nesta ordem, a mesma do llama.cpp: viés →
penalidades → top-n-sigma → top-k → typical → top-p → min-p → temperatura. A
ordem muda o resultado (o top-p depois da temperatura, por exemplo, escolheria
outros tokens). Nenhum filtro deixa a lista de candidatos vazia: sobra sempre
pelo menos um.

### Raciocínio

| Flag | Padrão | Descrição |
|---|---|---|
| `--reasoning <on\|off\|auto>` | `auto` | Liga ou desliga o pensamento (o bloco `<think>`). `auto` deixa como o modelo faz por padrão. |
| `--reasoning-effort <nível>` | — | Quanto o modelo pode pensar: `none` (nada), `low` (até 1024 tokens), `medium` (4096), `high` (16384) ou `max` (sem limite). |
| `--reasoning-budget <N>` | — | Limite exato de tokens de pensamento; vale mais que o nível. `-1` = sem limite. |

Quando o limite acaba, o airun fecha o `</think>` e o modelo passa direto
para a resposta. Se o próprio template do modelo entende níveis (o do Qwen3.8
tem `low`, `medium` e `xhigh`), o nível vai para ele. As três opções valem
também no `serve`, como padrão dos pedidos. Na API: `reasoning_effort`
(OpenAI), `reasoning.effort` (Responses), `thinking.budget_tokens`
(Anthropic) e `reasoning_budget_tokens` (llama-server).

### Imagens

| Flag | Padrão | Descrição |
|---|---|---|
| `--image <arquivo>` | — | Uma imagem (PNG, JPEG ou WebP) para o prompt; repita para várias. Precisa de `--mmproj`. Ponha `<__media__>` no prompt onde cada uma vai; sem marcador, elas vão antes do texto. |

```powershell
airun run modelo.gguf --mmproj mmproj-BF16.gguf --image foto.jpg -p "O que tem nesta imagem?" --temp 0
```

O processamento da imagem segue o do llama.cpp (`tools/mtmd`), então a
mesma imagem gera a mesma descrição nos dois. Hoje funciona o `mmproj` dos
Qwen3.5/3.8. Com imagem no prompt o MTP fica desligado naquela geração.

### Saída

| Flag | Padrão | O que faz |
|---|---|---|
| `--stats` | desligado | Imprime o plano de memória e o resumo de tempos. |
| `-q`, `--quiet` | desligado | Silencia o progresso de carga no stderr. |

**Exemplos**

```powershell
# Determinístico, uma GPU
airun run modelo.gguf -p "Explique RoPE" -n 200 --temp 0

# Contexto grande, forçando split entre duas placas
airun run modelo.gguf -p "..." -c 32768 --tensor-split 0.6,0.4 --stats

# Prompt via pipe
Get-Content prompt.txt | airun run modelo.gguf -n 500
```

---

## `airun serve`

Sobe uma API HTTP no formato do llama-server: OpenAI (chat, completions,
responses), Anthropic (messages), a nativa do llama.cpp (`/completion`,
`/infill`) e os utilitários. Os campos de amostragem (`dry_*`, `xtc_*`,
`dynatemp_*`, `mirostat*`), `grammar`, `json_schema`, `response_format`,
`n_probs`/`logprobs` e `ignore_eos` vão na requisição, como no llama-server;
não há flags de servidor para eles.

| Flag | Padrão | O que faz |
|---|---|---|
| `--host <ip>` | `127.0.0.1` | Endereço de bind. Use `0.0.0.0` para expor na rede. |
| `--port <N>` | `8080` | Porta. |
| `--parallel <N>` (ou `--np`) | `1` | Quantos pedidos são atendidos ao mesmo tempo. Veja [Vários pedidos ao mesmo tempo](#vários-pedidos-ao-mesmo-tempo). |
| `--api-key <chave>` | — | Exige `Authorization: Bearer <chave>`. Também lida da variável `AIRUN_API_KEY`. **Sem esta flag não há autenticação nenhuma** — não use `--host 0.0.0.0` sem ela. |
| `--served-model-name <nome>` (ou `--alias`) | o nome no arquivo | Nome mostrado em `/v1/models`. Sem a opção, vem dos metadados do GGUF; se não houver, do nome do arquivo. |
| `--default-max-tokens <N>` | `512` | Limite das requisições OpenAI que não especificam um. O `/completion` nativo gera até o fim, como o llama-server. |
| `--reasoning-format <deepseek\|none>` | `deepseek` | Onde vai o pensamento (`<think>`) das respostas de chat: em `reasoning_content` (padrão do llama-server) ou no `content`. O `reasoning_format` da requisição tem prioridade. |
| `--repeat-penalty <f>` | `1.1` | Penalidade multiplicativa em tokens recentes quando a requisição não pede uma. `1.0` desliga. |
| `--repeat-last-n <N>` | `64` | Quantos tokens recentes as penalidades consideram. |
| `--reasoning-budget <N>` | `-1` | Tokens de pensamento antes de o `</think>` ser forçado; `-1` é sem limite. O `reasoning_budget_tokens` da requisição tem prioridade. |
| `--reasoning-budget-message <texto>` | — | Texto escrito antes do `</think>` forçado. |
| `--slot-save-path <dir>` | — | Habilita `POST /slots/0?action=save\|restore\|erase`, que gravam e carregam o cache da conversa nesse diretório. Sem ela, 501. |
| `--embeddings` | desligado | Habilita `/embedding` e `/v1/embeddings`. Sem ela, 501. |
| `--pooling <none\|mean\|cls\|last>` | do GGUF, senão `last` | Como os estados dos tokens viram um embedding. `none` devolve um vetor por token (só no endpoint nativo). |

Mais todas as flags de memória e alocação da seção acima.

**Sobre as penalidades:** a API da OpenAI só tem `presence_penalty` e
`frequency_penalty`. A `repeat_penalty` (que evita o modelo ficar repetindo)
não tem campo lá, então vale a do servidor; um cliente pode mandar
`repeat_penalty` no corpo para trocar.

### Vários pedidos ao mesmo tempo

Com `--parallel N`, até N pedidos são atendidos juntos, como os slots do
llama-server:

- Cada pedido ganha um **slot** com o próprio cache. A VRAM do cache cresce N
  vezes, então o contexto automático fica menor.
- Os que estão gerando avançam **juntos**: uma passada pelo modelo dá um token
  para cada um, lendo os pesos uma vez só. No 27B (5060 Ti + 3060): 1 pedido
  sozinho 22,7 tok/s; 4 juntos somam 50,3 tok/s (cada um a ~13).
- Um pedido novo vai para o slot que já tem o começo da conversa em cache, então
  o próximo turno de um chat não reprocessa o histórico.
- Com N maior que 1 o MTP fica desligado, e modelos `qwen4exp` não aceitam.

### Cache de prompt

O servidor reaproveita o prefixo do prompt que já está nas caches, então num
chat só o último turno é processado a cada mensagem. O ganho é grande porque o
prefill é lento (veja abaixo): num Qwen3.6-27B, o segundo turno de uma conversa
de ~880 tokens caiu de ~78s para 1,5s.

Nos modelos **híbridos** (Qwen3.5/3.8/Next) há uma limitação que vem da
arquitetura: o estado da delta-net é um resumo de tudo que ela leu, e não dá
para "desler" um token. Então o cache só é aproveitado quando o prompt novo
**continua exatamente** o anterior; editar uma mensagem antiga faz reprocessar
tudo. Nos modelos densos (Qwen3) dá para voltar a qualquer ponto.

Cada requisição imprime uma linha no stderr dizendo quanto foi aproveitado:

```
prompt 884 tok (869 cached, 15 new)  prefill 1285 ms (11.7 tok/s)  decode 1 tok in 116 ms (8.6 tok/s)
```

### Endpoints

| Endpoint | Método | Observações |
|---|---|---|
| `/health` | GET | Sem autenticação. Para health checks. |
| `/v1/models` | GET | Requer autenticação. |
| `/v1/chat/completions` | POST | Suporta `stream: true` (SSE). |
| `/v1/completions` | POST | Suporta `stream: true` (SSE). |
| `/v1/stream/{id}?from=N` | GET | Relê uma geração a partir do byte N e continua ao vivo (veja abaixo). |
| `/v1/streams/lookup` | POST | `{"conversation_ids": [...]}`: diz quais dessas gerações existem e se já acabaram. |
| `/v1/stream/{id}` | DELETE | Para a geração e apaga o que estava guardado. |

**Stream replay:** um pedido com o cabeçalho `X-Conversation-Id: <id>`
continua gerando mesmo se o cliente cair (aba fechada, rede caiu). O cliente
volta e relê o texto pelo `/v1/stream/<id>`. O servidor guarda os últimos 4
MiB e apaga 5 minutos depois que a geração acaba. É o mesmo recurso do
llama-server.

**Imagens (com `--mmproj`):** mande a imagem dentro do pedido, em base64:
`image_url` com data URL (OpenAI), `input_image` (Responses), bloco `image`
(Anthropic) ou `multimodal_data` (`/completion`). O servidor não baixa URLs
http. A mesma imagem nos turnos seguintes de um chat não é processada de novo.

Campos aceitos: `messages`, `prompt`, `max_tokens`, `max_completion_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `top_n_sigma`, `seed`,
`presence_penalty`, `frequency_penalty`, `repeat_penalty`, `logit_bias`,
`stream`, `stop`.

`logit_bias` é o objeto da OpenAI — chave é o id do token, valor é o quanto
somar ao logit dele. `-inf` (ou um valor bem negativo) bane o token. Entradas
que não parseiam são ignoradas em vez de derrubar a requisição.

`finish_reason` é `"stop"` quando o modelo emitiu um token de fim ou uma stop
sequence bateu, e `"length"` quando o orçamento de tokens ou o contexto acabou.

Uma stop sequence que cai entre dois tokens **não vaza** para o cliente: o
servidor segura o pedaço duvidoso até saber se ele vira a stop sequence.

**Vários clientes:** com `--parallel 1` (o padrão) os pedidos são atendidos um
de cada vez, e os outros esperam na fila. Para atender vários juntos, use
`--parallel N`.

```powershell
airun serve modelo.gguf --port 8080 --api-key minha-chave -c 8192
```

---

## `airun convert`

Converte um checkpoint HuggingFace (safetensors) em GGUF.

| Flag | Padrão | O que faz |
|---|---|---|
| `<input>` | — | Diretório com `config.json` e os `.safetensors`. Shards são unidos automaticamente. |
| `-o`, `--output <path>` | — | Arquivo `.gguf` de saída. Diretórios pai são criados. |
| `--type <tipo>` | `F16` | Tipo alvo. Veja a tabela de tipos abaixo. |
| `--pure` | desligado | Aplica `--type` a todas as matrizes, sem exceções por tensor. |
| `--name <nome>` | nome da pasta | Nome gravado no arquivo. |

Por padrão (sem `--pure`), a precisão é elevada em tensores sensíveis:
`token_embd` e `output` para Q6_K; `attn_v` e `ffn_down` um passo acima do alvo.
Normas ficam sempre em F32. É o mesmo raciocínio das variantes `_M` do
llama.cpp.

```powershell
airun convert G:\modelsAI\origin\Qwen3-8B -o E:\models\Qwen3-8B-Q4_K.gguf --type Q4_K
```

---

## `airun quantize`

Reescreve um GGUF em outra precisão.

| Flag | Padrão | O que faz |
|---|---|---|
| `<input>` `<output>` | — | Arquivos de origem e destino. |
| `--type <tipo>` | `Q4_K` | Tipo alvo. |
| `--pure` | desligado | Sem exceções por tensor. |
| `--dry-run` | desligado | Mostra o que seria escrito, sem escrever. |

Nunca aumenta a precisão: um tensor já em 4 bits não ganha nada sendo reescrito
em 8 — o arquivo só cresceria carregando o mesmo erro. Tensores cujo tipo não
muda são copiados byte a byte, sem round-trip com perda.

---

## Tipos de quantização

| Tipo | Bits/peso | Leitura | Escrita | Observação |
|---|---|---|---|---|
| `F32` | 32 | sim | sim | |
| `F16` | 16 | sim | sim | |
| `BF16` | 16 | sim | sim | Faixa maior, menos mantissa |
| `Q8_0` | 8.5 | sim | sim | Quase sem perda |
| `Q6_K` | 6.56 | sim | sim | |
| `Q5_K` | 5.5 | sim | sim | |
| `Q5_0` / `Q5_1` | 5.5 / 6.0 | sim | sim | |
| `Q4_K` | 4.5 | sim | sim | Melhor equilíbrio geral |
| `Q4_0` / `Q4_1` | 4.5 / 5.0 | sim | sim | Legados |
| `Q2_K` / `Q3_K` | 2.63 / 3.44 | sim | não | |
| `IQ4_NL` / `IQ4_XS` | 4.5 / 4.25 | sim | não | Codebook não-linear |
| `MXFP4` | 4.25 | sim | não | Escala E8M0 |

`Q4_K_M` e `Q4_K_S` são aceitos como sinônimos de `Q4_K`: o sufixo nomeia uma
política de mistura por tensor, não um formato de bloco.

---

## `airun inspect`

| Flag | O que faz |
|---|---|
| `--tensors` | Lista cada tensor com forma, tipo e offset. |
| `--all-metadata` | Inclui os arrays enormes do tokenizer. |
| `--json` | Saída legível por máquina. |

---

## `airun bench`

| Flag | Padrão | O que faz |
|---|---|---|
| `--prefill <lista>` | `128,512` | Tamanhos de prompt a medir. |
| `--decode <N>` | `128` | Tokens gerados na medição de decode. |
| `--decode-at <lista>` | `16` | Profundidades de contexto onde medir o decode, ex.: `--decode-at 16,1024,3072`. O prompt é ingerido antes e só a geração é cronometrada. |
| `--runs <N>` | `3` | Execuções medidas por configuração. |
| `--warmup <N>` | `1` | Execuções descartadas (JIT de kernel, subida de clock). |
| `--text-file <arquivo>` | — | Monta os prompts a partir deste texto em vez de um token repetido. Modelos MoE usam uma passagem embutida por padrão. |
| `--temp <T>` | `0` | Temperatura da amostragem no decode; 0 é greedy. |
| `--json` | — | Saída legível por máquina. |

**Em MoE com `--expert-cache`, meça com texto natural e `--temp 0.7`.** Um
token repetido roteia sempre para os mesmos experts, e o greedy entra em laço;
os dois fazem o cache parecer perfeito. A coluna *expert hit* mostra a taxa de
acerto de cada teste.

Reporta a **mediana**, não a média: uma única interrupção de escalonamento
distorce demais a média de três execuções.

**O decode não é uma taxa só.** A atenção lê o cache inteiro a cada token, então
a taxa cai com o contexto — num Qwen3-1.7B, de 234 t/s com o contexto vazio para
128 t/s a 3072 tokens. O default (`--decode-at 16`) mede só o caso vazio, que é
o melhor caso e não descreve uma conversa longa. Ao comparar duas versões,
compare na mesma profundidade.

---

## `airun perplexity`

Pontua um texto conhecido e reporta a perda média, a perplexidade e a taxa de
acerto top-1. Serve para responder "este modelo está *certo*?", que a geração de
texto não responde: um decodificador com um layout trocado ainda escreve inglês
fluente enquanto responde a outra pergunta.

| Flag | Padrão | O que faz |
|---|---|---|
| `-t`, `--text <TEXTO>` | passagem embutida | Texto a pontuar. |
| `-f`, `--file <ARQUIVO>` | — | Lê o texto de um arquivo. Conflita com `--text`. |
| `-n`, `--limit <N>` | `0` (tudo) | Para depois de N tokens. |
| `--worst <N>` | `0` | Lista os N tokens pior previstos, do pior para o melhor. |
| `--batched` | desligado | Com `--chunk`: cada bloco numa passada só, pelos kernels de prefill (como o `llama-perplexity`), em vez de token a token. Mede o custo de qualidade do caminho de prefill (ex.: ativações FP4). |
| `--chunk <N>` | `0` | Pontua como o `llama-perplexity -c N`: blocos de N tokens, contexto reiniciado em cada um, só a segunda metade conta. Mesmo texto e N dão números comparáveis entre os dois (normalize CRLF→LF antes: o llama.cpp lê `-f` em modo texto no Windows). |
| `-c`, `--ctx-size <N>` | `4096` | Tamanho do contexto. |
| `--gpus`, `--tensor-split`, `--single-gpu` | | Iguais às de `run`. |

Os tokens são alimentados **um a um**, que é o caminho que a decodificação
realmente toma — um estado recorrente atualizado errado seria invisível numa
única passagem em lote.

Valores de referência nesta máquina, na passagem embutida:

| Modelo | Perda | Top-1 |
|---|---|---|
| Qwen3-8B Q4_K (denso, correto) | 2.20 | 50.9% |
| Qwen3-1.7B Q8_K_XL (denso, correto) | 2.43 | 47.2% |
| Qwen3.5-9B Q4_K_M (híbrido) | 4.40 | 20.1% |
| Qwen3.6-35B-A3B Q4_K_XL (híbrido + MoE) | 11.86 | 1.9% |

`ln(248320) = 12.4` é o valor de um modelo que chuta uniformemente.

---

## `airun selftest`

| Flag | Padrão | O que faz |
|---|---|---|
| `--device <ordinal>` | `0` | GPU a testar. |
| `--verbose` | desligado | Mostra cada caso, inclusive os que passam. |

Compara os 16 decodificadores GPU, o matvec de cada tipo, rmsnorm, swiglu, rope
e atenção contra a referência CPU. **Rode isto depois de mexer em qualquer
kernel.**

---

## Variáveis de ambiente

| Variável | O que faz |
|---|---|
| `AIRUN_API_KEY` | Alternativa a `--api-key`. |
| `AIRUN_CUDA_ARCHS` | Arquiteturas CUDA compiladas (build). Ex.: `120` durante desenvolvimento, para builds rápidas. Padrão: `75,80,86,89,120`. |
| `AIRUN_NVCC_CCBIN` | Diretório do compilador host para o nvcc, se a detecção automática falhar. |
| `AIRUN_SKIP_CUDA` | Compila sem kernels GPU (só parsing de arquivos). |
| `CUDA_PATH` | Onde achar o toolkit CUDA. |

### Variáveis de depuração dos híbridos

Existem porque várias ambiguidades de layout produzem tensores da **mesma
forma**: nenhuma validação as distingue, só a saída do modelo. Cada uma liga a
interpretação alternativa; meça a diferença com `airun perplexity`, não olhando
o texto gerado. Todas as escolhas atuais foram medidas de novo depois da correção de 2026-09-26 e são as melhores.

| Variável | Interpretação alternativa que liga |
|---|---|
| `AIRUN_ROPE_SCALING=none\|linear\|yarn` | Ignora o `rope.scaling.type` do arquivo, como o `--rope-scaling` do llama.cpp. Para A/B. |
| `AIRUN_ROPE_MSCALE=<f>` | Substitui o fator de magnitude do YaRN (padrão `1 + 0.1 ln fator`). Para A/B. |
| `AIRUN_EMBED_ON_GPU=1` | Mantém a tabela de embeddings na VRAM. Por padrão ela fica em RAM pinned (como no llama.cpp) quando não é amarrada à cabeça de saída: ~0,8 GiB a menos num vocabulário de 250k. |
| `AIRUN_NO_FP4=1` | Prefill NVFP4 no MMQ int8 (ativações de 8 bits) em vez dos tensor cores FP4 do Blackwell (ativações de 4 bits). Mais lento, um pouco mais exato. |
| `AIRUN_NO_Q8_WIDE=1` | Decode e verify NVFP4 voltam aos kernels de um warp por quatro linhas. |
| `AIRUN_NO_Q8_MULTI=1` | Lotes de 2–8 tokens voltam ao kernel tiled em vez do que decodifica cada grupo uma vez para todos os tokens. |
| `AIRUN_SPEC_TIMING=1` | Imprime o tempo médio de verify, commit e rascunho por passo especulativo. |
| `AIRUN_MATVEC_F32=1` | Todos os matvecs no caminho f32 exato (o de referência). Sem ela, o decode já é exato; lotes pequenos usam dp4a sobre ativações Q8 e lotes grandes usam tensor cores int8 (MMQ) — ambos validados no selftest por limite de erro backward e fim-a-fim por perplexidade. Use quando "idêntico à referência" importa mais que velocidade. |
| `AIRUN_NO_WMMA=1` | Mantém lotes grandes nos kernels de registrador (mede o caminho tensor-core contra eles sem rebuild). |
| `AIRUN_NO_MMQ=1` | Lotes de 64+ tokens saem dos tensor cores int8 (MMQ) e voltam para o fp16 (WMMA). |
| `AIRUN_NO_MMQ_ASYNC=1` | Mantém o MMQ, mas na versão síncrona (sem `cp.async`). |
| `AIRUN_NO_FLASH=1` | O prefill usa o kernel de atenção antigo (um bloco por query) em vez da flash attention. |
| `AIRUN_NO_GQA_DECODE=1` | O decode usa o kernel de atenção por head de query em vez do agrupado por head de KV. |
| `AIRUN_ABLATE=ssm,attn,ffn` | Zera a contribuição desses blocos ao residual. Um bloco correto sempre pontua melhor que bloco nenhum. Sanidade: `ffn` no 8B denso leva a perda de 2.20 para 25.96 — se ela **não** se mexer, a flag não está ligada naquele caminho. |
| `AIRUN_TRACE_LAYERS=1` | Imprime rms/média de cada bloco e do residual, camada a camada. |
| `AIRUN_TRACE_SSM=<camada>` | Imprime as faixas de q, k, v, z, decay, beta, estado e saída dessa camada delta-net. |
| `AIRUN_SSM_GATE_BEFORE_NORM=1` | Aplica o gate antes da normalização (a ordem antiga, errada). |
| `AIRUN_SSM_GATE_SIGMOID=1` | Sigmoid em vez de SiLU no gate. |
| `AIRUN_SSM_QKV_ORDER=<perm>` | Ordem dos três blocos em `attn_qkv`, ex.: `kqv`. Padrão `qkv`. |
| `AIRUN_SSM_QKV_GROUPED=1` | `attn_qkv` agrupado por k-head em vez de blocos contíguos. |
| `AIRUN_SSM_SWAP_AB=1` | Troca `ssm_alpha` e `ssm_beta`. |
| `AIRUN_SSM_CONV_REVERSE=1` | Inverte a ordem dos taps da convolução. |
| `AIRUN_SSM_NO_L2NORM=1` | Não normaliza q/k para norma unitária. |
| `AIRUN_SSM_NO_QSCALE=1` | Não escala q por `1/sqrt(head_k_dim)`. |
| `AIRUN_SSM_NO_CONV_SILU=1` | Sem SiLU depois da convolução. |
| `AIRUN_SSM_A_IS_LOG=1` | Trata `ssm_a` como `A_log` cru mesmo já sendo negativo. |
| `AIRUN_ATTN_GATE_BLOCKED=1` | `attn_q` como `[todos os q][todos os gates]` em vez de intercalado por head. |
| `AIRUN_ATTN_GATE_FIRST=1` | O gate é a **primeira** metade da fatia de cada head. |
| `AIRUN_ATTN_NO_GATE=1` | Não aplica o gate sigmoid na saída da atenção. |
| `AIRUN_ROPE_FULL=1` | Rotaciona `head_dim` inteiro em vez de `rope.dimension_count`. |
| `AIRUN_ROPE_INTERLEAVED=1` | Pareamento GPT-J `(2i, 2i+1)` em vez de NeoX `(i, i + dim/2)`. |
| `AIRUN_ROPE_OFF=1` | Sem RoPE nenhum. O custo disso mede quanto o modelo usa posição: no 8B denso vale +2.75 nats, no 9B híbrido +0.04. |
| `AIRUN_ATTN_WINDOW=<n>` | Força uma janela deslizante. Com `1` cada query só se vê; o custo mede quanto a atenção tira do histórico. |

### Variáveis de desempenho

Cada uma desliga (ou força) uma otimização, para medir o ganho dela sem
recompilar. Nenhuma muda o resultado além de arredondamento — as que mexem na
ordem das somas estão cobertas pelo selftest e pela perplexidade. Meça sempre
alternando as execuções (veja `benchmarks/2026-09-26/scoreboard.ps1`): nesta
máquina, outro programa desenhando na GPU atrasa kernels em ~300 µs a cada 4 ms,
e duas sessões separadas variam ~15%.

| Variável | Efeito |
|---|---|
| `AIRUN_NO_MMQ_STREAMK=1` | O MMQ assíncrono volta à grade de tiles inteiros (com split-K onde ela ocupa mal a GPU). Com stream-K as últimas tiles são divididas em fatias de K iguais entre todos os SMs: prefill de 128 tokens no Qwen3-4B de 2767 para 3269 t/s. |
| `AIRUN_NO_FP4_GLU=1` | Num FFN NVFP4 nas RTX 50, gate e up voltam a ser dois matmuls FP4, com o SwiGLU na quantização do down. Por padrão os três são um launch só (SwiGLU no epílogo): +2% de prefill no 27B. |
| `AIRUN_NO_DELTA_CHUNK=1` | O prefill das camadas delta-net volta à recorrência token a token. Por padrão, lotes de 64+ tokens usam a forma em blocos nos tensor cores (bf16, acumulação fp32): +8,5% de prefill no 27B, +7% no 35B, perplexidade igual. |
| `AIRUN_NO_MMQ_SPLIT=1` | Sem split-K no MMQ (onde o stream-K não roda: kernel síncrono, Q6_K). |
| `AIRUN_MMQ_TC=64` ou `=128` | Força a largura da tile de tokens do MMQ. Padrão: 128 a partir de 128 tokens. |
| `AIRUN_NO_MMQ_GLU=1` | Gate, up e SwiGLU do prefill em três lançamentos em vez de um. |
| `AIRUN_ATTN_PV_F32=1` | O prefill soma P·V em fp32 também com contexto longo (a partir de 4096 chaves, o padrão é fp16 por tile de chaves, somado em fp32). |
| `AIRUN_NO_SPLIT_WAVES=1` | O decode usa todas as divisões do cache que a posição permite, sem arredondar para ondas inteiras de blocos (veja `attention_active_splits`). |
| `AIRUN_NO_FLASH2=1` | O prefill usa a flash attention WMMA antiga em vez da de `mma.sync` (head_dim 64/128). |
| `AIRUN_NO_SPLIT_ATTN=1` | O decode usa o kernel de atenção de um bloco por head, sem dividir o contexto. |
| `AIRUN_NO_FUSED_PROLOGUE=1` | q/k norm, RoPE e escrita no cache em lançamentos separados. |
| `AIRUN_NO_FUSED_NORMS=1` | Soma residual e RMSNorm seguinte em dois lançamentos. |
| `AIRUN_NO_GRAPH=1` | O decode lança os kernels um a um em vez de repetir um CUDA graph capturado. |
| `AIRUN_NO_GPU_ARGMAX=1` | Decodificação gulosa copia os logits para o host e escolhe lá. |
| `AIRUN_NO_DECODE_PIPELINE=1` | Decodificação gulosa espera cada token antes de enfileirar o passo seguinte. Com o pipeline o passo i+1 lê o token do passo i direto da GPU; o texto gerado é idêntico. Não roda em híbridos (o estado recorrente absorveria o passo especulativo). |
