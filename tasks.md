# airun — estado do projeto e handoff

Este arquivo é o ponto de entrada para retomar o trabalho em uma sessão nova.
Leia isto e `AGENTS.md` antes de tocar em qualquer código. `AGENTS.md` tem o
mapa do código e as armadilhas gerais; este arquivo tem o estado exato de cada
frente de trabalho e o que fazer a seguir.

Última atualização: 14/08/2026.

---

## 0. Ambiente — copiar e colar para retomar

```powershell
$env:RUSTUP_HOME="E:\rust\rustup"; $env:CARGO_HOME="E:\rust\cargo"
$env:PATH="E:\rust\cargo\bin;$env:PATH"
$env:AIRUN_CUDA_ARCHS="120"   # build rápido em dev; remova para build final (75,80,86,89,120)
Set-Location "d:\coding\ia\airun"
```

Hardware: RTX 5070 Ti (16GB) + RTX 5060 Ti (16GB), CUDA 13.2, Windows 11,
MSVC Build Tools 2022. **Sem P2P entre as placas.** CUDA ordinal 0 = 5070 Ti,
mas nvidia-smi 0 = 5060 Ti — não são o mesmo índice.

Modelos de teste:
- `G:\modelsAI\gguf\qwen3\Qwen3-1.7B-UD-Q8_K_XL.gguf` — denso, referência de regressão
- `G:\modelsAI\gguf\qwen3.5\Qwen3.5-9B-Q4_K_M.gguf` — híbrido SSM, 9B
- `G:\modelsAI\gguf\qwen3.6\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` — híbrido SSM+MoE, 35B
- `G:\modelsAI\gguf\qwen3.6\mmproj-f16.gguf` — encoder de visão (não implementado)
- `E:\airun-models\Qwen3-8B-Q4_K.gguf` — convertido por nós de
  `G:\modelsAI\origin\Qwen3-8B` (safetensors), usado para validar `convert`/`quantize`
- `G:\modelsAI\origin\gemma-4-12B` — safetensors, não usado ainda

**Primeiro passo sempre**, para confirmar que nada regrediu:

```powershell
cargo test --release 2>&1 | Select-String "test result:|error"
cargo build --release --message-format short 2>&1 | Select-String "error"
.\target\release\airun.exe selftest
# esperado: NN passed, 0 failed — hoje NN=55
.\target\release\airun.exe run G:\modelsAI\gguf\qwen3\Qwen3-1.7B-UD-Q8_K_XL.gguf -p "The capital of France is" -n 12 --temp 0 -q --raw
# esperado: " Paris. The capital of Italy is Rome. The capital of"
.\target\release\airun.exe perplexity G:\modelsAI\gguf\qwen3\Qwen3-1.7B-UD-Q8_K_XL.gguf
# esperado: perda 2.4348, top-1 47.2%
.\target\release\airun.exe perplexity E:\airun-models\Qwen3-8B-Q4_K.gguf
# esperado: perda 2.2018, top-1 50.9%
```

Esses dois últimos números são a **linha de base de qualidade**. Eles não se
movem quando o trabalho é só nos híbridos; se moverem, você quebrou o caminho
denso.

---

## 1. Mapa do código

```
crates/
  airun-gguf/          formato de arquivo, sem CUDA
    ggml_type.rs        tabela de tipos ggml
    reader.rs           parser GGUF (mmap)
    dequant.rs           decodificadores CPU — ORÁCULO dos kernels GPU
    quantize.rs          11 codificadores (escrita)
    writer.rs            escritor GGUF streaming
    safetensors.rs        leitor de checkpoints HF

  airun-cuda/          GPU
    build.rs             acha CUDA+MSVC, compila kernels/*.cu
    kernels/dequant.cuh   decodificadores GPU — espelho de dequant.rs
    kernels/matmul.cu     matvec quantizado + dequantize
    kernels/ops.cu        rmsnorm, rope, swiglu, attention, kv_store, embed
    kernels/ssm.cu        NOVO: gated delta-net (conv1d, l2norm, delta rule, gates)
    kernels/moe.cu        NOVO: router + FFN de experts
    kernels/device.cu     device mgmt (shim sobre cudaDeviceProp)
    src/ops.rs            wrappers seguros dos kernels densos
    src/hybrid.rs          NOVO: wrappers seguros dos kernels híbridos/MoE

  airun-core/          runtime
    config.rs            hiperparâmetros a partir do GGUF + SsmConfig/MoeConfig
    plan.rs               planner de VRAM/split multi-GPU
    tokenizer.rs           BPE byte-level nativo
    model.rs               upload de pesos (denso + híbrido + MoE)
    runtime.rs             forward pass (denso + híbrido + MoE)
    sample.rs               amostragem
    loader.rs               orquestração + fallback

  airun-cli/           comandos: run, serve, inspect, gpu, convert, quantize,
                        bench, perplexity, selftest, tokenize
    perplexity.rs        NOVO: métrica de qualidade (perda/top-1) — a ferramenta
                          que torna qualquer hipótese um número
```

Docs: `README.md`, `AGENTS.md` (leia a seção 7 antes de tocar em híbridos),
`CLAUDE.md`, `docs/FLAGS.md`.

---

## 2. O que está 100% concluído e verificado

Modelos **densos** (`qwen3`, `qwen2`, `llama`): pipeline completo, ponta a
ponta, sem ressalvas.

- Parser/escritor GGUF, 16 decodificadores + 11 quantizadores, todos bit-exatos
  ou dentro de tolerância testada contra referência CPU
- `airun selftest`: valida cada kernel GPU contra CPU antes de qualquer coisa
- `airun convert`: safetensors→GGUF testado com Qwen3-8B real (15,2GB BF16 →
  5,06GB Q4_K, roda e responde corretamente)
- `airun quantize`: GGUF→GGUF em outra precisão
- `airun run`/`serve`: geração real, API OpenAI (chat, completions, SSE, auth)
  testada por HTTP de verdade
- Multi-GPU: split de camadas, planner automático (uma GPU quando cabe,
  proporcional quando não cabe), `--tensor-split`/`--single-gpu` manuais
- `airun bench`, `airun gpu`, `airun tokenize`, `airun inspect --values`
  (dump de tensor com min/max/média — ferramenta chave para debug)
- `airun perplexity`: perda/perplexidade/top-1 numa passagem conhecida. É a
  única forma de dizer se um modelo está *certo*, e não só fluente

Regressão confirmada nesta sessão: Qwen3-1.7B e Qwen3-8B continuam corretos
depois de todo o trabalho em híbridos — texto idêntico e perda idêntica
(2.4348 e 2.2018), com `cargo test` 52/52 e `selftest` 55/55.

---

## 3. EM ANDAMENTO — híbridos SSM/MoE (qwen35, qwen35moe)

**Não pare para "confirmar direção" nisto — é debug direto, continue sozinho.**

### O que é

Qwen3.5/3.6 = arquitetura **Qwen3-Next Gated DeltaNet**: a maioria das camadas
é atenção linear recorrente (não softmax), só 1 em cada `full_attention_interval`
(=4) é atenção softmax de verdade, e essa atenção tem um gate sigmoid extra. O
35B ainda soma MoE (256 experts, 8 ativos, + 1 expert compartilhado sempre-ativo).
A última camada de `block_count` é uma cabeça MTP e **não é executada**.

### O que já funciona

- Config parseia tudo: `SsmConfig`, `MoeConfig`, `full_attention_interval`,
  `attn_output_gate`, `n_run_layers()` (exclui a MTP)
- Model carrega os pesos híbridos: `attn_qkv`, `attn_gate`, `ssm_*`,
  `ffn_*_exps`, `ffn_*_shexp`, tudo com validação de forma
- Planner contabiliza estado recorrente (tamanho fixo) nas camadas lineares em
  vez de KV cache (que cresce com contexto) — e divide corretamente entre GPUs
- Runtime executa o forward completo: RoPE parcial, delta rule, conv1d causal
  com estado persistente, router MoE, FFN de experts
- 8 kernels CUDA novos, todos em `kernels/ssm.cu` e `kernels/moe.cu`,
  wrappers em `src/hybrid.rs`
- `selftest`: **55/55**, incluindo:
  - delta rule streaming (token a token) == batched, bit-exato
  - conv1d causal streaming == batched, bit-exato
  - atenção testada nas formas exatas dos modelos reais (head_dim 256, GQA 4x)
  - MoE router (softmax+top-k) e FFN de experts contra referência CPU
  - as duas ordens do rmsnorm com gate, mais um caso que falha se elas
    deixarem de ser diferentes
- Qwen3.5-9B **carrega e gera** (5,3GB, cabe numa GPU)
- Qwen3.6-35B-A3B **carrega e gera**, dividido automaticamente entre as 2 GPUs
  (20,78GB — não cabe numa só)
- Recuperação de longo alcance **funciona**: num texto repetido 3x o 9B acerta
  71,3% dos tokens (o 1.7B denso acerta 77,0%). Ou seja, as camadas de atenção
  copiam do contexto; o que está quebrado não é o alcance.

### O que NÃO funciona — agora com número

A gramática está intacta e a **semântica está destruída**. Exemplos com
`--temp 0 --raw`:

```
"The capital of Japan is"                    -> " a Japanese TV show, and the Show"
"The largest planet in the solar system is"  -> " 1096."
"2 + 2 ="                                    -> " 1"
```

Medido com `airun perplexity` (passagem embutida, 159 tokens), **todo o parque
de modelos deste PC**, tudo depois da correção do gate:

| Modelo | FFN | Perda | Top-1 | Veredito |
|---|---|---|---|---|
| Qwen3-8B Q4_K — denso, controle | densa | **2.20** | 50.9% | correto |
| Qwen3-1.7B Q8_K_XL — denso, controle | densa | **2.43** | 47.2% | correto |
| Qwen3.5-9B Q4_K_M — híbrido | densa | **4.40** | 20.1% | um defeito |
| Qwen3.6-27B Q4_K_M — híbrido | densa | **4.87** | 15.1% | um defeito |
| Qwen3.6-35B-A3B Q4_K_XL — híbrido | MoE | **6.99** | 8.2% | dois defeitos |
| Qwen3.5-35B-A3B MXFP4_MOE — híbrido | MoE | **6.94** | 7.5% | dois defeitos |

`ln(248320) = 12.4` é o chute uniforme. O padrão é em degraus limpos:

- **denso: ~2.2–2.4** — correto
- **híbrido com FFN densa: ~4.4–4.9** — um defeito, e ele é o mesmo em 9B e 27B
- **híbrido + MoE: ~6.9–7.0** — algo a mais, ~2 nats além do degrau anterior

Os dois modelos MoE têm quantização de expert **completamente diferente**
(Q4_K contra MXFP4) e pontuam praticamente igual, 6.99 contra 6.94. Isso
**elimina a quantização** como explicação do degrau do MoE.

Uma ressalva honesta sobre esse segundo degrau: com `AIRUN_ABLATE=ffn` o MoE
vale 5.5 nats (6.99 → 12.50), ou seja **ele não está morto**, está degradado
como o resto. E são modelos A3B, com só 3B de parâmetros ativos, o que
justificaria *parte* da diferença — mas não 2 nats.

**Diagnóstico: não é kernel, não é layout, e não é o split multi-GPU.** Tudo
isso foi medido.

### Bugs já encontrados e corrigidos (não reintroduzir)

1. **Planner alocava a camada MTP.** `cfg.n_layers` inclui a MTP;
   `cfg.n_run_layers()` não. O planner e o model loader devem sempre iterar
   `0..cfg.n_run_layers()`, nunca `0..cfg.n_layers`.

2. **Overflow nos buffers q/k/v.** Eram dimensionados só pela atenção
   (`n_head_kv * head_dim`). As camadas SSM escrevem `group_count * state_size`
   nesses mesmos buffers, que no Qwen3.5-9B é o **dobro** (2048 vs 1024). Só
   estourava em prompts >256 tokens. Corrigido em `runtime.rs::new()`,
   dimensionando pelo `max()` das duas necessidades.

3. **`ssm_a` já é `-exp(A_log)`, não `A_log` cru.** Confirmei olhando os
   valores reais com `inspect --values blk.0.ssm_a`: todos negativos, entre
   -0,14 e -0,009 — isso é claramente já-exponenciado-e-negado, não um log.
   Aplicar `-exp()` de novo no kernel multiplicava o decaimento por ~13x; o
   estado recorrente esquecia quase tudo a cada token. Corrigido em
   `model.rs::load_delta_decay_coeff`, que detecta pelo sinal (se todos os
   valores são negativos, já está pronto; senão aplica `-exp()`).

4. **Bug no próprio harness de teste**, não no produto: a função `compare()` em
   `selftest.rs` usava `f32::max`, que ignora NaN silenciosamente. Bytes de
   teste aleatórios geram ~3% de escalas f16 NaN, então dois arrays cheios de
   NaN comparavam como "iguais" (erro 0.000e0) — teste que não podia falhar.
   Corrigido: `compare()` agora trata qualquer NaN como erro infinito, e os
   geradores de dados de teste usam escalas plausíveis em vez de bytes
   totalmente aleatórios. **Se você vir `0.000e0` de erro num teste com dados
   aleatórios, desconfie — verifique que os arrays não são zero/NaN antes de
   confiar no PASS.**

5. **`rmsnorm_gated` aplicava o gate depois da normalização.** O
   `RMSNormGated` do Mamba-2, de onde o do Qwen3-Next vem, multiplica pelo
   `silu(z)` **antes** de medir a linha, de modo que a normalização divide a
   escala do próprio gate de volta. Fazendo depois, cada uma das 24 camadas
   lineares reescala o residual livremente. Corrigido em `ssm.cu`; vale 4.89 →
   4.40 de perda. `AIRUN_SSM_GATE_AFTER_NORM=1` volta ao antigo.

   **Outro bug de harness junto com esse:** o caso do `selftest` reimplementava
   a mesma ordem errada como "referência CPU", então não podia falhar. Agora
   testa as duas ordens e tem um caso que falha se elas pararem de ser
   diferentes. Quando um teste GPU-vs-CPU passa mas o modelo está errado,
   suspeite que os dois lados codificam a mesma suposição.

### Espaço de busca já varrido — não repita nada disto

Cada linha foi medida com `airun perplexity` no Qwen3.5-9B. A configuração
atual é a melhor de todas; a coluna mostra o que acontece ao ligar a
alternativa. **Baseline atual: 4.40.**

| Alternativa (env var) | Perda | Conclusão |
|---|---|---|
| — (configuração atual) | **4.40** | melhor |
| `AIRUN_SSM_GATE_AFTER_NORM=1` | 4.89 | gate antes da norma vence (mudança desta sessão) |
| `AIRUN_SSM_A_IS_LOG=1` | 4.61 | `ssm_a` já é `-exp(A_log)`, heurística de sinal certa |
| `AIRUN_ROPE_INTERLEAVED=1` | 4.43 | **inconclusivo** — veja "onde procurar" §3 |
| `AIRUN_ROPE_FULL=1` | 4.45 | **inconclusivo** — idem |
| `AIRUN_ROPE_OFF=1` | 4.45 | **posição não está sendo usada** — veja §3 |
| `AIRUN_SSM_SWAP_AB=1` | 4.58 | `alpha`→decay, `beta`→update certo |
| `AIRUN_SSM_QKV_ORDER=kqv` | 5.59 | ordem `[q][k][v]` certa |
| `AIRUN_SSM_QKV_ORDER=qvk/kvq/vqk/vkq` | 12.8–13.4 | idem |
| `AIRUN_SSM_CONV_REVERSE=1` | 6.31 | ordem dos taps certa |
| `AIRUN_SSM_NO_QSCALE=1` | 8.05 | escala `1/sqrt(d_k)` em q é necessária |
| `AIRUN_SSM_NO_CONV_SILU=1` | 8.20 | SiLU depois da conv é necessário |
| `AIRUN_SSM_NO_L2NORM=1` | NaN | l2norm em q/k é necessário |
| `AIRUN_SSM_GATE_SIGMOID=1` | 14.40 | SiLU no gate, não sigmoid |
| `AIRUN_ATTN_GATE_FIRST=1` | 10.47 | q é a primeira metade da fatia, gate a segunda |
| `AIRUN_ATTN_GATE_BLOCKED=1` | 10.77 | `attn_q` intercalado por head certo |
| `AIRUN_SSM_QKV_GROUPED=1` | 12.17 | blocos contíguos, não agrupado por k-head |
| `AIRUN_ATTN_NO_GATE=1` | 15.84 | o gate da atenção está sendo aplicado certo |

Ablações (`AIRUN_ABLATE=`), para calibrar quanto cada bloco contribui:

| Ablação | Perda |
|---|---|
| nenhuma | 4.40 |
| `ssm` (24 blocos fora) | 11.00 |
| `attn` (8 blocos fora) | 13.72 |
| `ssm,attn` (tudo fora) | 13.92 |

**Leitura:** tirar as 8 camadas de atenção chega quase no mesmo lugar que tirar
tudo (13.72 contra 13.92) — as 24 camadas delta-net sozinhas quase não
contribuem. Já tirar as 24 delta-net deixa em 11.00, então elas contribuem
quando a atenção está presente. Nenhum bloco é "pior que nada", que era a
hipótese mais simples.

### Outras coisas descartadas nesta sessão

- **Split multi-GPU.** Hipótese plausível, porque *todos* os modelos corretos
  cabem numa GPU e *todos* os quebrados são pesados o bastante para dividir —
  as duas coisas estavam completamente confundidas no parque de modelos. Testado
  forçando cada caso e o resultado é **bit a bit idêntico**:

  | Modelo | 1 GPU (ord. 0) | 1 GPU (ord. 1) | split 50/50 |
  |---|---|---|---|
  | Qwen3-8B denso | 2.2018 | — | **2.2018** |
  | Qwen3.5-9B híbrido | 4.4045 | 4.4045 | **4.4045** |

  O handoff de ativações entre shards, o `Placement` e o estado recorrente
  dividido estão corretos. Não perca tempo aqui de novo.
- **Tokenizador.** `airun tokenize` no 9B e no 1.7B dá a mesma segmentação e
  round-trip exato. Não é isso.
- **Prefill em lote.** `-b 512`, `-b 4` e `-b 1` dão saída idêntica, então o
  estado recorrente e a conv atravessam os limites de bloco corretamente.
- **Camada MTP.** `blk.32` tem os tensores `nextn.*`; pulá-la está certo.
- **Decodificador Q5_K** (só usado nos 24 `ssm_out` deste arquivo): os valores
  desquantizados são simétricos, média ~0, 50% negativos. Distribuição sã.
- **mRoPE.** `rope.dimension_sections = [11,11,10,0]` soma 32 = `rope_dim/2`.
  Em `apply_multimodal_rotary_pos_emb` as três seções são reconcatenadas a
  partir de `cos.split(sections*2)`; com texto puro os três eixos de posição
  são idênticos, então o resultado é bit a bit o RoPE padrão. Não precisa ser
  implementado para texto — só para imagens.
- **Offset DC no residual.** `AIRUN_TRACE_LAYERS=1` mostra médias por bloco de
  ~1e-3 tanto no 9B quanto no 8B denso. O perfil de magnitude também é
  parecido. Não há deriva de escala nem viés acumulado.
- **`post_attention_norm`.** O arquivo tem exatamente dois layernorms por
  bloco (`attn_norm` e `post_attention_norm`) e o Qwen3-Next tem exatamente
  dois pre-norms, então o mapeamento é forçado. Não é o "attn_post_norm" do
  Gemma2.

### Onde procurar a seguir

1. **O caminho MoE**, isolado no 35B. Ele está em 11.86 (quase aleatório)
   contra 4.40 do 9B, que não tem MoE — quase certamente há um defeito
   separado ali. É a pista mais forte que sobrou porque é a maior diferença
   entre dois modelos que compartilham todo o resto. Comece medindo
   `AIRUN_ABLATE=ffn` no 35B e comparando os experts contra uma referência CPU
   com os pesos reais.

2. **Uma referência CPU de uma camada delta-net inteira, com os pesos reais do
   arquivo.** É o próximo passo natural do método do projeto (`dequant.rs` é o
   oráculo dos kernels; falta o oráculo de uma *camada*). Pega erros de
   composição — stride, aliasing de buffer, ordem de argumento — que os testes
   por kernel não pegam porque constroem as entradas corretamente por
   construção. É o único caminho que sobrou que não é adivinhação.

3. **A pista mais forte: o híbrido quase não usa posição.** Medido, não
   deduzido — desligue o RoPE inteiro com `AIRUN_ROPE_OFF=1`:

   | Modelo | RoPE ligado | RoPE desligado | Custo |
   |---|---|---|---|
   | Qwen3-8B denso | 2.20 | 4.95 | **+2.75** |
   | Qwen3.5-9B híbrido | 4.40 | **4.45** | **+0.04** |

   Tirar toda a informação de posição do híbrido custa 0,04 nats. É por isso
   que *todos* os testes de RoPE deram empate (parcial 4.40, total 4.45,
   intercalado 4.43, desligado 4.45): não havia sinal posicional chegando à
   saída para eles mexerem. **Os resultados de RoPE da tabela acima não
   provam que o RoPE está certo — provam que ele é irrelevante hoje.**

   Note também que o 8B denso *sem posição nenhuma* pontua 4.95, quase o
   mesmo que o híbrido completo em 4.40. O híbrido se comporta como um modelo
   sem posição.

   Quanto da atenção é posicional, usando `AIRUN_ATTN_WINDOW=<n>` para medir
   quanto ela tira do histórico:

   | Janela | 8B denso | 9B híbrido |
   |---|---|---|
   | 1 (só o próprio token) | 9.13 | 5.26 |
   | 8 | 4.46 | 4.75 |
   | 64 | 2.94 | 4.45 |
   | completa | 2.20 | 4.40 |

   No denso a atenção vale 6.93 nats e 2.75 deles (40%) são posicionais. No
   híbrido ela vale 0.86 nats e só 0.04 (5%) são posicionais — ou seja, as 8
   camadas softmax fazem recuperação por conteúdo, sem ordem.

   **Ressalva honesta:** as 24 camadas delta-net carregam posição
   implicitamente pela recorrência, então parte dessa insensibilidade pode ser
   legítima da arquitetura. O que não dá para explicar assim é a razão 5%
   contra 40%. Este é o ponto onde eu começaria.

   Isso também recoloca a atenção sob suspeita, apesar de todos os toggles
   dela terem sido descartados: a capacidade de copiar texto repetido (71,3%)
   pode vir inteira do delta-net, que é uma memória associativa k→v e faz
   exatamente isso. Não é prova de que a atenção funciona.

### Método (atualizado — o smoke test antigo era enganoso)

1. **Meça, não olhe.** `airun perplexity` em vez de ler o texto gerado. Isto
   inverteu uma decisão nesta sessão: com o gate depois da norma o modelo
   continua `"...the lazy"` com `" dog"` e com o gate antes produz bobagem,
   mas a perda diz o contrário (4.89 contra 4.40), e a inversão se confirma
   numa segunda passagem independente (5.75 contra 5.08). **O `" dog"` era
   sorte de um modelo quebrado, não sinal de correção.**

2. **Sempre contra um controle.** `E:\airun-models\Qwen3-8B-Q4_K.gguf` é o
   controle certo: mesmo tamanho, mesma precisão, caminho denso conhecido.

3. **Olhe os valores reais.** `airun inspect --values` para pesos,
   `AIRUN_TRACE_SSM=<camada>` para intermediários em execução. Foi assim que
   se achou o `-exp(A_log)` já aplicado, e é assim que se vê que os taps da
   conv1d têm o último ~100x maior que os outros — o que confirma sozinho o
   layout e a direção da convolução.

4. **Ablate para separar "este bloco está errado" de "o erro está em outro
   lugar".** Um bloco correto sempre pontua melhor que bloco nenhum.

5. Depois de mexer em kernel: `selftest` **e** a perplexidade dos dois modelos
   densos de controle. O selftest não pega composição; os controles densos
   pegam regressão no que já funciona.

### Comandos de teste rápidos para retomar

```powershell
# métrica de qualidade (o teste que vale)
.\target\release\airun.exe perplexity G:\modelsAI\gguf\qwen3.5\Qwen3.5-9B-Q4_K_M.gguf
# hoje: perda 4.4045, top-1 20.1%   (controle denso: 2.20 / 50.9%)

# A/B de uma interpretação alternativa
$env:AIRUN_SSM_QKV_ORDER="kqv"; .\target\release\airun.exe perplexity <modelo>

# quanto cada tipo de bloco contribui
$env:AIRUN_ABLATE="ssm"; .\target\release\airun.exe perplexity <modelo>

# intermediários de uma camada delta-net, num forward real
$env:AIRUN_TRACE_SSM="0"; .\target\release\airun.exe run <modelo> -p "..." -n 1 --temp 0 -q --raw

# perfil de magnitude camada a camada
$env:AIRUN_TRACE_LAYERS="1"; .\target\release\airun.exe run <modelo> -p "..." -n 1 --temp 0 -q --raw

# inspecionar valores de um tensor
.\target\release\airun.exe inspect G:\modelsAI\gguf\qwen3.5\Qwen3.5-9B-Q4_K_M.gguf `
  --values blk.0.ssm_a --count 8
```

A lista completa das variáveis está em `docs/FLAGS.md`.

---

## 4. NÃO INICIADO — multi-modal (encoder de visão)

Estrutura já mapeada via `inspect --tensors` em `mmproj-f16.gguf`, nenhum
código escrito ainda:

- ViT: 27 blocos, patch 16x16, embd 1152, ffn 4304, `attn_qkv`/`attn_out` com
  bias, `ln1`/`ln2` pre-norm, `post_ln` final
- Projetor `qwen3vl_merger`: merge espacial 2x2, camadas deepstack
  (`clip.vision.is_deepstack_layers`), projeção final `mm.0`/`mm.1` para 4608
- Depende do item 3 (híbridos) estar correto primeiro, porque o modelo de
  texto que acompanha o mmproj é qwen3.6, não um denso

A API (`serve.rs`) já aceita o formato de conteúdo multi-parte da OpenAI
(`content: [{type: text, ...}, {type: image_url, ...}]`) sem falhar no parse —
só ignora as partes não-texto por enquanto.

---

## 5. Outras frentes conhecidas

### Prefill com tensor cores — a maior otimização disponível

`matvec_kernel` indexa o token por `blockIdx.y` e **redecodifica a linha de peso
para cada token**, então o prefill relê o modelo inteiro uma vez por token. Não
há reuso entre tokens: o prefill é literalmente N decodes em sequência.

Medido num Qwen3.6-27B Q4_K_M (17,2 GiB, dividido entre as duas placas):

| | tok/s |
|---|---|
| prefill | **11,3** |
| decode | 8,9 |

**O prefill é só 27% mais rápido que o decode.** Num modelo saudável a razão
seria de uma ordem de grandeza. É o custo de ler 17,2 GiB por token, com as duas
GPUs em série porque não há P2P. Na prática: 2000 tokens de prompt = 3 minutos
até o primeiro token.

Um GEMM com `mma.sync`/WMMA que carregue o bloco de pesos uma vez e o aplique a
vários tokens é o conserto. Ortogonal ao trabalho de híbridos.

### Feito nesta sessão

- **Cache de prompt no `serve`**: `Runtime::reusable_prefix` acha o prefixo já
  processado e `generate` pula ele. Num chat só o último turno é processado.
  No 27B, o segundo turno de uma conversa de ~880 tokens caiu de ~78s para
  **1,5s** (869 de 884 tokens reaproveitados), com saída idêntica à do caminho
  sem cache.

  A restrição é arquitetural: em híbridos o estado do delta-net não é
  rebobinável, então o cache só vale quando o novo prompt estende exatamente o
  anterior. `reusable_prefix` retorna 0 nos outros casos. Em modelo denso o KV
  cache é endereçado por posição e dá para voltar a qualquer ponto.

- **Penalidade de repetição no `/v1/chat/completions`**: montava
  `1.0 + frequency_penalty`, e como a maioria dos clientes manda `0` explícito
  o resultado era `1.0` — penalidade desligada justo no endpoint que as UIs de
  chat usam, enquanto `/v1/completions` e `airun run` usavam 1.1. Agora só um
  `frequency_penalty` **positivo** sobrescreve o default, que virou a flag
  `--repeat-penalty` (1.1).

- **Log por requisição** no stderr do `serve`, com tokens aproveitados do cache
  e as taxas de prefill/decode — antes não havia como atribuir um TTFT lento.

- **UTF-8 quebrado no streaming.** `decode_one` rodava `from_utf8_lossy` num
  token só. Como o BPE byte-level parte caracteres acentuados entre tokens
  (`"Órbita"` = `<20 C3>` `<93>` `"r"`), metade de um caractere virava `U+FFFD`
  e os bytes se perdiam — a saída aparecia como `Anigku-in «s ——e-e n—i—m`.
  Atingia **qualquer modelo**, inclusive os densos corretos, em qualquer texto
  não-ASCII. Corrigido com `StreamDecoder` em `tokenizer.rs`, que segura a
  cauda incompleta até o token que a completa; `generate` usa ele e dá flush no
  fim. Três testes cobrem isso, incluindo um que corta uma frase com acento,
  CJK e emoji em **todos** os limites de byte e exige o mesmo resultado.

  O `airun tokenize` agora mostra fragmentos como `<20 C3>` em vez de ``,
  porque era justamente a ferramenta de diagnóstico exibindo o sintoma.

### Ainda não iniciadas

- **NVFP4 (ggml type id 40).** A tabela em `ggml_type.rs` termina em
  `MXFP4 = 39`. **Metade dos modelos pesados deste PC é NVFP4 e não abre** —
  os três 27B (`Unsloth-...-NVFP4-A`, `...-NVFP4-LOW`,
  `...-NVFP4-MTP-Q8_K_P`) param todos no mesmo ponto, com a mensagem correta:

  ```
  tensor "blk.0.attn_qkv.weight" uses ggml type id 40, which this build does
  not know; the file was likely produced by a newer converter
  ```

  É trabalho bem delimitado e independente do bug dos híbridos: bloco de 16
  valores E2M1 com escala por bloco em FP8 E4M3. Precisa do decoder em
  `dequant.rs`, do espelho em `dequant.cuh`, da entrada na tabela de tipos e
  de um caso no `selftest` — o mesmo roteiro dos outros 16.
- **KV cache paginado** para múltiplas sequências concorrentes (hoje `serve`
  serializa requisições, e o cache de prompt é de uma sequência só).
- **Template Jinja**: `tokenizer.chat_template` é armazenado mas não avaliado;
  usa-se ChatML fixo.
