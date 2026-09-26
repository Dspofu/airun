# Referência de flags do airun

Todas as flags de todos os comandos. Rode `airun <comando> --help` para a versão
curta.

```
airun run        gerar texto
airun serve      API HTTP compatível com OpenAI
airun inspect    descrever um arquivo GGUF
airun gpu        listar GPUs CUDA detectadas
airun convert    safetensors (HuggingFace) -> GGUF
airun quantize   GGUF -> GGUF em outra precisão
airun bench      medir throughput
airun perplexity medir quão bem um modelo prevê um texto
airun selftest   validar kernels GPU contra a referência CPU
```

---

## Flags de memória e alocação (compartilhadas por `run`, `serve`, `bench`)

Estas são as flags que decidem **o que vai para qual GPU**.

| Flag | Padrão | O que faz |
|---|---|---|
| `-c`, `--ctx-size <N>` | `4096` | Tamanho do contexto em tokens. É o principal controle de VRAM: o KV cache cresce linearmente com ele. Limitado ao contexto de treino do modelo. |
| `-b`, `--batch-size <N>` | `512` | Tokens por forward pass durante a ingestão do prompt. Maior = prefill mais rápido, mais VRAM de scratch. Não afeta a geração. |
| `--gpus <lista>` | todas | Ordinais CUDA a usar, ex.: `--gpus 0,1`. **Atenção:** ordinal CUDA não é o índice do `nvidia-smi`. Rode `airun gpu` para ver a correspondência. |
| `--tensor-split <fracoes>` | — | Proporção manual de camadas por GPU, ex.: `--tensor-split 0.7,0.3`. Uma entrada por GPU selecionada. Desliga o modo automático. |
| `--single-gpu <ordinal>` | — | Força o modelo inteiro numa placa. Conflita com `--tensor-split`. |
| `--vram-reserve <MiB>` | `384` | VRAM deixada livre em cada GPU. O Windows/WDDM não entrega os últimos megabytes; reduza só se souber o que está fazendo. |
| `--no-fallback` | desligado | Falha em vez de reduzir contexto/batch automaticamente para caber. Útil em CI, onde uma redução silenciosa mascara uma regressão. |

### Como a alocação automática decide

Sem `--tensor-split` nem `--single-gpu`, o planner:

1. Lê os tamanhos **exatos** de cada tensor no cabeçalho GGUF (não estima).
2. Soma pesos + KV cache + scratch.
3. **Se couber numa GPU só, usa uma GPU só** — mesmo havendo outras livres.
   Atravessar PCIe custa mais do que a segunda placa ganha, especialmente sem
   P2P.
4. Se não couber, divide as camadas em faixas contíguas proporcionais à VRAM
   livre de cada placa, colocando mais na mais rápida.
5. Se ainda não couber, desce a escada de fallback (veja abaixo).

Tudo que ele decidiu aparece em `--stats`.

### Escada de fallback

Quando não cabe, o loader tenta nesta ordem, e informa o que mudou:

1. **Batch pela metade** (até 64) — custa só velocidade de prefill.
2. **Contexto pela metade** (até 512) — muda o que o modelo consegue fazer.
3. **Reserva de VRAM para 128 MiB** — último recurso, arrisca falha de driver.

`--no-fallback` desliga tudo isso.

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

Ordem de aplicação: viés → penalidades → top-n-sigma → top-k → typical → top-p
→ min-p → temperatura. É a ordem do llama.cpp e não é cosmética: aplicar
temperatura antes mudaria o que o top-p seleciona, e rodar top-p antes do top-k
tornaria o top-k inócuo. Todo filtro preserva **pelo menos um** candidato — um
filtro que esvazia o conjunto vira uma troca silenciosa de amostrador.

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

Sobe uma API HTTP compatível com OpenAI.

| Flag | Padrão | O que faz |
|---|---|---|
| `--host <ip>` | `127.0.0.1` | Endereço de bind. Use `0.0.0.0` para expor na rede. |
| `--port <N>` | `8080` | Porta. |
| `--api-key <chave>` | — | Exige `Authorization: Bearer <chave>`. Também lida da variável `AIRUN_API_KEY`. **Sem esta flag não há autenticação nenhuma** — não use `--host 0.0.0.0` sem ela. |
| `--served-model-name <nome>` | nome do arquivo | Nome reportado em `/v1/models`. |
| `--default-max-tokens <N>` | `512` | Limite quando a requisição não especifica um. |
| `--repeat-penalty <f>` | `1.1` | Penalidade multiplicativa em tokens recentes quando a requisição não pede uma. `1.0` desliga. |
| `--repeat-last-n <N>` | `64` | Quantos tokens recentes as penalidades consideram. |

Mais todas as flags de memória e alocação da seção acima.

**Sobre as penalidades:** o esquema da OpenAI só tem as duas aditivas
(`presence_penalty` e `frequency_penalty`), que passam direto para o
amostrador. A multiplicativa não tem campo lá, então ela fica no default do
servidor — um cliente que manda `frequency_penalty: 0` continua protegido
contra loops. Um cliente de llama.cpp pode mandar `repeat_penalty` no corpo
para sobrescrever.

### Cache de prompt

O servidor reaproveita o prefixo do prompt que já está nas caches, então num
chat só o último turno é processado a cada mensagem. O ganho é grande porque o
prefill é lento (veja abaixo): num Qwen3.6-27B, o segundo turno de uma conversa
de ~880 tokens caiu de ~78s para 1,5s.

Há uma restrição que vem da arquitetura, não da implementação: num modelo
**híbrido** o estado do delta-net é um resumo corrido de tudo que ele viu, sem
como subtrair um token de volta. Então lá o cache só serve quando o novo prompt
**estende exatamente** o anterior — qualquer edição no histórico obriga a
reprocessar tudo. Num modelo denso o KV cache é endereçado por posição e dá para
voltar a qualquer ponto anterior.

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

Campos aceitos: `messages`, `prompt`, `max_tokens`, `max_completion_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `top_n_sigma`, `seed`,
`presence_penalty`, `frequency_penalty`, `repeat_penalty`, `logit_bias`,
`stream`, `stop`.

`logit_bias` é o objeto da OpenAI — chave é o id do token, valor é o quanto
somar ao logit dele. `-inf` (ou um valor bem negativo) bane o token. Entradas
que não parseiam são ignoradas em vez de derrubar a requisição.

`finish_reason` é `"stop"` quando o modelo emitiu um token de fim ou uma stop
sequence bateu, e `"length"` quando o orçamento de tokens ou o contexto acabou.

Uma stop sequence partida entre dois tokens **não vaza**: o servidor segura a
cauda ambígua até saber se ela vira a stop sequence ou não. Um chunk SSE já
enviado não tem como ser desfeito.

**Concorrência:** as requisições são serializadas. O KV cache é uma sequência
única; duas gerações simultâneas se corromperiam. Clientes que assumem
concorrência funcionam — apenas ficam na fila.

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
| `--json` | — | Saída legível por máquina. |

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
o texto gerado. Todas as escolhas atuais já foram medidas e são as melhores —
veja a tabela em `tasks.md`.

| Variável | Interpretação alternativa que liga |
|---|---|
| `AIRUN_MATVEC_F32=1` | Todos os matvecs no caminho f32 exato (o de referência). Sem ela, o decode já é exato; lotes de 2–63 tokens usam dp4a sobre ativações Q8 e 64+ usam tensor cores fp16 — ambos validados no selftest por limite de erro backward e fim-a-fim por perplexidade. Use quando "idêntico à referência" importa mais que velocidade. |
| `AIRUN_NO_WMMA=1` | Mantém lotes grandes nos kernels de registrador (mede o caminho tensor-core contra eles sem rebuild). |
| `AIRUN_ABLATE=ssm,attn,ffn` | Zera a contribuição desses blocos ao residual. Um bloco correto sempre pontua melhor que bloco nenhum. Sanidade: `ffn` no 8B denso leva a perda de 2.20 para 25.96 — se ela **não** se mexer, a flag não está ligada naquele caminho. |
| `AIRUN_TRACE_LAYERS=1` | Imprime rms/média de cada bloco e do residual, camada a camada. |
| `AIRUN_TRACE_SSM=<camada>` | Imprime as faixas de q, k, v, z, decay, beta, estado e saída dessa camada delta-net. |
| `AIRUN_SSM_GATE_AFTER_NORM=1` | Aplica o gate depois da normalização, não antes. |
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
