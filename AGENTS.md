# Guia para agentes de IA que trabalham no airun

Leia este arquivo **antes** de alterar qualquer coisa. Ele descreve o que o
projeto é, o que já funciona, onde estão as armadilhas, e como verificar que
você não quebrou nada.

---

## 1. O que é o airun

Uma engine de inferência de LLMs **escrita do zero**, em Rust + CUDA.

**Isto não é um wrapper.** Não há llama.cpp, ggml, PyTorch, ONNX ou cuBLAS no
caminho de execução. Todo kernel CUDA em `crates/airun-cuda/kernels/` foi
escrito para este projeto. Se você se pegar prestes a adicionar uma dependência
de inferência externa, pare: essa foi uma decisão explícita do usuário.

Dependências permitidas: utilitários (clap, serde, memmap2, half, thiserror,
anyhow, tokio, axum). O CUDA Runtime é usado apenas para gerenciar memória,
streams e lançar *nossos* kernels.

---

## 2. Estado atual (verificado neste dispositivo)

| Componente | Estado |
|---|---|
| Leitura GGUF (v1–v3) | funciona |
| 17 decodificadores de quantização | **bit-exatos** vs. referência CPU |
| 12 quantizadores (escrita) | round-trip testado |
| Escrita GGUF | round-trip testado |
| Leitura safetensors (multi-shard) | funciona |
| Kernels CUDA (matvec f32/dp4a/tensor-core, attention, rope, rmsnorm, swiglu, embed, kv) | 226/226 no selftest |
| Runtime Qwen3 denso | gera texto coerente |
| Multi-GPU (split por camada) | validado com 2 placas |
| Planner de VRAM | testado, com fallback automático |
| API OpenAI (chat, completions, SSE, auth) | validada |
| Amostragem (top-k/p, min-p, typical, top-n-sigma, 3 penalidades, logit_bias) | testada |
| Conversão HF → GGUF | Qwen3-8B convertido e executado |
| Benchmark | funciona |

**Arquiteturas suportadas hoje:** `qwen3`, `qwen2`, `llama` (decoders densos),
`qwen35` / `qwen35moe` (híbridos SSM/atenção + MoE — veja a seção 7).

**Não suportado ainda** (o loader recusa com mensagem clara):
- `clip` (mmproj) — encoder de visão.

---

## 3. Mapa do código

```
crates/
  airun-gguf/          formato de arquivo — sem CUDA, sem I/O de rede
    ggml_type.rs       tabela de tipos; IDs desconhecidos NÃO são erro
    reader.rs          parser GGUF (mmap, tudo com bounds-check)
    dequant.rs         decodificadores CPU — ORÁCULO dos kernels GPU
    quantize.rs        codificadores (f32 -> blocos)
    writer.rs          escrita GGUF em streaming
    safetensors.rs     leitura de checkpoints HF

  airun-cuda/          GPU
    build.rs           acha CUDA + MSVC, compila kernels/*.cu -> .lib
    kernels/*.cu       OS KERNELS. Escritos à mão.
    kernels/dequant.cuh  decodificadores GPU — espelho de dequant.rs
    src/ffi.rs         FFI (só tipos de largura fixa)
    src/ops.rs         wrappers seguros

  airun-core/          runtime
    config.rs          hiperparâmetros a partir dos metadados GGUF
    plan.rs            planner de VRAM / split multi-GPU
    tokenizer.rs       BPE byte-level (GPT-2/Qwen)
    model.rs           upload de pesos
    runtime.rs         forward pass
    sample.rs          amostragem
    loader.rs          orquestração + escada de fallback

  airun-cli/           comandos
```

---

## 4. A regra mais importante

> **`crates/airun-gguf/src/dequant.rs` e
> `crates/airun-cuda/kernels/dequant.cuh` implementam a MESMA coisa duas
> vezes, de propósito.**

O arquivo Rust é a referência legível; o `.cuh` é a versão GPU. `airun selftest`
compara os dois numericamente.

Se você mudar um, **precisa** mudar o outro e rodar o selftest. Um layout de
quantização sutilmente errado não trava — ele produz texto que parece quase
certo. Isso é muito pior que um crash, e o selftest é a única defesa.

---

## 5. Como verificar seu trabalho

Sempre, nesta ordem:

```powershell
$env:RUSTUP_HOME="E:\rust\rustup"; $env:CARGO_HOME="E:\rust\cargo"
$env:PATH="E:\rust\cargo\bin;$env:PATH"
$env:AIRUN_CUDA_ARCHS="120"     # só a arch local: build rápido durante dev

cargo test --release            # 60+ testes unitários
cargo build --release
.\target\release\airun.exe selftest        # 78 casos GPU vs CPU
.\target\release\airun.exe run G:\modelsAI\gguf\qwen3\Qwen3-1.7B-UD-Q8_K_XL.gguf `
    -p "What is 2+2?" -n 20 --temp 0

# e a métrica de qualidade, que é o que pega erro sutil:
.\target\release\airun.exe perplexity E:\airun-models\Qwen3-8B-Q4_K.gguf
# esperado: perda 2.2018 nats/token, top-1 50.9%
```

Se o selftest passar mas a geração virar lixo, o bug está no *runtime*
(ordem das operações, formato do KV cache, RoPE), não nos kernels.

**Decode não é uma taxa só.** Ele cai com o contexto (1.7B: 234 t/s vazio,
128 t/s a 3072). Ao comparar medições, compare na mesma profundidade.

**Não julgue correção lendo o texto gerado.** Um modelo com um layout trocado
escreve inglês fluente respondendo outra coisa, e uma continuação curta acerta
por sorte — isso já inverteu uma decisão neste projeto. Use
`airun perplexity`: ele pontua uma passagem conhecida e devolve perda e top-1,
então qualquer hipótese vira um número comparável contra um modelo denso de
controle.

**Não remova `AIRUN_CUDA_ARCHS` da build final.** O padrão compila para
sm_75/80/86/89/120 e leva vários minutos, mas é o que dá suporte a RTX 20 até 50.

---

### Os três caminhos de matvec

O decode (n=1) roda no kernel f32 exato — geração é bit-idêntica à referência
CPU. Lotes de 2–63 tokens usam dp4a sobre ativações quantizadas (block_q8a em
`vecdotq.cuh`, espelho em `quantize_q8a_mirror` no selftest — mudou um, mude o
outro). Lotes de 64+ vão para **tensor cores int8** (MMQ, `mmq.cuh`) em sm_80+
para Q8_0/Q4_K/Q5_K/Q6_K — os pesos viram os mesmos códigos inteiros do dp4a;
Q8_0/Q4_K/Q5_K usam a versão com `cp.async` (`matvec_mmq_async_kernel`).
Outros tipos e GPUs mais antigas vão para tensor cores fp16
(`matvec_wmma_kernel`). Cada caminho tem seus casos no selftest com limite de
erro backward; `AIRUN_MATVEC_F32=1` força o exato em tudo, `AIRUN_NO_MMQ=1`,
`AIRUN_NO_MMQ_ASYNC=1` e `AIRUN_NO_WMMA=1` desligam cada um.

O MMQ assíncrono usa **stream-K** quando as tiles não fecham ondas inteiras de
SMs (`MmqStreamK` em `mmq.cuh`): as ondas cheias rodam tiles inteiras e as
tiles que sobram são divididas em fatias de K iguais entre uma onda de blocos.
O bloco que tem o começo de uma tile ("head") espera as flags dos blocos
seguintes ("tails") e soma os parciais na ordem dos blocos — determinístico.
As esperas só apontam para blocos de índice maior e o host limita o stream-K a
uma onda residente, então não trava. Gate+up+SwiGLU do prefill são um
lançamento só (`GLU=true`), inclusive no stream-K (os slots levam as somas
cruas de gate e up; o SwiGLU roda no head). Não capture MMQ num CUDA graph: o
`epoch` das flags congelaria (`mmq_streamk_plan` recusa em stream capturando).

No decode, o Q8_0 carrega os bytes de todas as linhas do warp antes de usar
qualquer uma (`q8_0_load`/`q8_0_decode` em `dequant.cuh`). Com um `break` ou
decodificação intercalada, o compilador segurava as cargas da linha seguinte
atrás da aritmética da anterior — uma linha em voo por warp. O resultado segue
bit-idêntico.

A decodificação gulosa é um pipeline (`ChainGraph` em `runtime.rs`): o grafo
lê token e posição da GPU, o argmax escreve o próximo token lá e avança a
posição, e o host enfileira o passo i+1 antes de ler o token do passo i. Um
passo especulativo que o fim da geração descarta não é confirmado (`n_past` e
`history` não andam). Por isso não roda em híbridos: o estado recorrente não
tem como desfazer um passo.

A atenção também tem caminhos por tamanho: prefill de 16+ tokens usa flash
attention em tensor core fp16 (`AIRUN_NO_FLASH=1` desliga), e o decode usa um
kernel por (head de KV, split) que lê o cache uma vez para todo o grupo GQA
(`AIRUN_NO_GQA_DECODE=1` volta ao antigo). `crates/airun-core/tests/prefill.rs`
(ignorado; `AIRUN_TEST_DENSE_GGUF`/`AIRUN_TEST_MOE_GGUF`) confere que um
prefill em lote chega aos mesmos logits que alimentar token a token — é o
único teste que roda esses kernels com pesos reais.

**Três armadilhas de kernel que custaram horas em 2026-09-26:**
- Um array de registradores (`acc[h]`) indexado num laço com limite de tempo
  de execução vai inteiro para a memória local. Use `#pragma unroll` com o
  limite constante e `if (h >= G) continue`. Confira `STACK:` em
  `cuobjdump --dump-resource-usage` — deve ser 0.
- `(const half2*)&raw` sobre uma variável de registrador também a manda para
  a memória local. Extraia com operações de bits.
- Sem contadores de desempenho (exigem admin no Windows), meça com `clock64()`
  dentro do kernel e `printf` de um bloco — foi o que achou os três gargalos.

**Ao ajustar tiles dos kernels: os tamanhos aparecem no template E na conta do
grid.** Varrer um sem o outro computa fração da saída e parece um ganho enorme
— `airun bench` não valida saídas. Mude as defines, nunca as instanciações, e
rode o selftest a cada ponto medido. Isso já enganou uma sessão inteira.

## 6. Armadilhas que já custaram tempo

1. **Ordinal CUDA ≠ índice do nvidia-smi.** O CUDA ordena por velocidade por
   padrão. Neste PC, CUDA 0 = RTX 5070 Ti, mas nvidia-smi 0 = RTX 5060 Ti.
   Sempre mostre o PCI bus id junto do ordinal.

2. **Estas GPUs não têm P2P.** `cudaDeviceCanAccessPeer` retorna 0. Cópias
   entre GPUs passam pela RAM. Por isso o planner prefere uma GPU só quando o
   modelo cabe — e por isso o split 50/50 é mais lento (94 t/s) que uma placa
   só (234 t/s) no modelo de 1.7B, medido com o contexto vazio.

3. **CUDA 13 removeu campos de `cudaDeviceProp`** (`memoryClockRate`). Nunca
   faça bind direto dessa struct do Rust — use o shim em `kernels/device.cu`.

4. **`gen` é keyword reservada na edition 2024.** Não use como nome de variável.

5. **Embeddings amarrados (tied).** Qwen3-1.7B não tem `output.weight`; a
   projeção final reusa `token_embd.weight`. Em split multi-GPU isso obriga a
   espelhar a tabela — o planner já contabiliza isso.

6. **Erro relativo é métrica errada para produto escalar.** Quando termos se
   cancelam, o resultado pode ser arbitrariamente pequeno perto dos valores que
   o geraram. O selftest usa erro *backward*: limite proporcional a
   `sum|w*x|`. Não "conserte" isso afrouxando tolerância.

7. **Dados de teste precisam ser fisicamente plausíveis.** Bytes aleatórios no
   expoente E8M0 do MXFP4 geram escalas de 2^±127 e estouram f32. O decode
   está certo; os dados é que eram irreais.

8. **Um teste GPU-vs-CPU não prova que a operação está certa** — prova que os
   dois lados concordam. Se ambos codificam a mesma suposição errada, o teste
   passa e o modelo continua quebrado. Aconteceu com o `rmsnorm_gated`, onde a
   "referência CPU" repetia a ordem errada do gate. Quando um kernel passa mas
   o modelo não presta, desconfie da referência antes do kernel.

9. **Um token não é um caractere.** No BPE byte-level um acentuado costuma
   ocupar dois tokens — `"Órbita"` sai como `<20 C3>` `<93>` `"r"`. Decodificar
   token a token roda `from_utf8_lossy` em meio caractere, o que devolve
   `U+FFFD` e **destrói os bytes**: nem quando a outra metade chega dá para
   recuperar. Todo streaming tem que passar por `StreamDecoder`, que segura a
   cauda incompleta. Vale para acentuação, CJK e emoji, em qualquer modelo —
   não é problema de modelo quebrado.

10. **NVFP4 tem um piso de magnitude absoluta.** O bloco não carrega escala por
    tensor, só quatro escalas UE4M3 (uma por 16 elementos), e o menor passo do
    UE4M3 é ~9.8e-4. Uma corrida de 16 pesos cujo pico esteja abaixo de ~5.9e-3
    codifica como **zeros**, sem aviso nenhum. Isso é o formato do ggml, não um
    bug do encoder — mas significa que um teste de round-trip com dados de
    magnitude pequena "falha" por motivo errado (foi o que aconteceu na primeira
    versão do teste). Pesos reais ficam bem acima do piso.

11. **A penalidade de repetição não acumula mais por ocorrência.** Ela é
    aplicada uma vez por token *distinto* na janela, que é o que o llama.cpp
    moderno faz. A versão anterior dividia o logit uma vez por ocorrência, então
    um token repetido 3 vezes era penalizado `penalty^3` — o efeito dependia de
    quão cheia a janela estava. Se você comparar saídas com uma sessão antiga,
    a diferença vem daqui, não de um bug novo.

---

## 7. Híbridos SSM/MoE — corrigidos em 2026-09-26

**A saída está correta agora.** Dois bugs se mascaravam um ao outro:

1. **Associação V-head → K-head no delta rule.** O GGUF repete o eixo inteiro
   de K-heads para chegar ao número de V-heads (`kh = vh % n_k_heads`, padrão
   `0,1,0,1…`), como o `ggml_repeat` do llama.cpp. O kernel usava pares
   adjacentes (`kh = vh / grupos`, padrão `0,0,1,1…`), que é o
   `repeat_interleave` do HuggingFace — mas o conversor reordena as V-heads
   justamente para que o tile funcione. `ssm.cu`, dois kernels.
2. **Ordem do gate no `rmsnorm_gated`.** É norma → peso → gate (SiLU), como o
   `RMSNormGated` do Qwen3-Next e o `build_norm_gated` do llama.cpp. O projeto
   tinha escolhido "gate antes" porque, **com o bug 1 presente**, essa ordem
   pontuava melhor (4.40 contra 4.89). Com o bug 1 corrigido, a ordem certa vai
   a 1.43 e a antiga fica em 5.12.

| Qwen3.8-35B-A3B Q4_K_M | Perda | Top-1 |
|---|---|---|
| antes | 7.02 | 8.8% |
| só a correção 1 | 5.12 | 14.5% |
| correções 1 + 2 | **1.43** | **57.2%** |
| Qwen3-4B Q8_0 (denso, controle) | 2.67 | 53.5% |

A varredura de `docs/FLAGS.md` foi **refeita sobre a base corrigida** e todas
as 15 alternativas pontuam pior (tabela em `benchmarks/2026-09-26/sweep2.txt`).

**Lição que vale para o projeto inteiro:** uma decisão A/B tomada com outro bug
presente não é confiável. Ao corrigir um layout, refaça as decisões de layout
anteriores. O selftest não pega isso (armadilha 8): os dois lados codificavam a
mesma suposição.

Bugs anteriores, ainda válidos (não reintroduza):
1. O planner deve alocar `cfg.n_run_layers()`; a última camada é MTP e não é
   carregada.
2. Os buffers `q/k/v` precisam de `group_count * state_size` nas camadas SSM.
3. **`ssm_a` já vem como `-exp(A_log)`**; `load_delta_decay_coeff` normaliza
   pelo sinal.

### Offload de experts (`--expert-cache N`)

Mantém N experts por camada na VRAM (LRU) e lê os que faltam do GGUF, sem
requantizar — o teste `crates/airun-core/tests/offload.rs` (ignorado; rode com
`AIRUN_TEST_MOE_GGUF=<35B>`) exige logits **idênticos** ao modelo residente.

Os misses são lidos com leituras posicionais paralelas (`upload_experts` em
`runtime.rs`, um handle de arquivo por thread). Pelo mmap eles chegavam um
page fault de cada vez, à velocidade de fila-1 do SSD, e isso era quase todo o
tempo de decode.

**Meça com texto natural.** `airun bench` usa um token repetido em modelos
densos, mas num MoE isso roteia sempre para os mesmos experts e o cache parece
perfeito (99,8% de acerto, 54 t/s) quando com texto real acerta ~88% (3 t/s
antes do leitor paralelo). O bench agora usa uma passagem em inglês para MoE e
mostra a taxa de acerto; use também `--temp 0.7`, porque o greedy entra em
laço e flatteria o cache da mesma forma.

### KV cache quantizado (`--kv-type f16|q8_0|q4_0`)

Blocos Q8_0/Q4_0 do GGUF por linha de head, decodificados pelos mesmos
`dequant`. O encoder GPU espelha `encode_q8_0`/`encode_q4_0` de `quantize.rs`
**bit a bit** (selftest compara os bytes); os intrinsics `__f*_rn` existem
porque o nvcc contrairia `x*id + 8.5` num FMA que arredonda diferente.
## 8. Outras frentes

1. **Decode que não cai com o contexto — feito.** No decode a atenção divide o
   intervalo de KV entre blocos (`gridDim.z`, `attention_splits(ctx)` blocos) e
   combina os parciais `(max, sum, acc)` num segundo kernel, como o
   `flash_attn_ext` do llama.cpp. Qwen3-4B Q8_0 numa RTX 3060: 38,9 → 53,4 t/s a
   3072 tokens (llama.cpp: 49,0). `AIRUN_NO_SPLIT_ATTN=1` volta ao kernel de um
   bloco por head. O selftest cobre até a posição 212991.

   Depois, em 2026-09-26/27: MMQ int8 com `cp.async`, flash attention com
   `mma.sync`, stream-K, fusões (QKV, gate/up/SwiGLU, prólogo da atenção,
   soma+norm), argmax na GPU e o pipeline do decode. Placar intercalado contra
   o llama.cpp, Qwen3-4B Q8_0, RTX 3060, melhor de 3 rodadas
   (`benchmarks/2026-09-26/scoreboard.ps1`):

   | Teste | airun | llama.cpp | razão |
   |---|---|---|---|
   | prefill 128 | 3535 t/s | 3118 t/s | 1,13x |
   | prefill 512 | 3979 t/s | 3829 t/s | 1,04x |
   | decode @16 | 70,3 t/s | 69,1 t/s | 1,02x |
   | decode @3072 | 63,7 t/s | 62,9 t/s | 1,01x |

   O decode está no limite da memória: os matvecs rodam a 88–96% da banda real
   (a 3060 fica em P2 durante compute, memória a 7301 MHz em vez de 7501). O
   que ainda sobra: atenção de decode em contexto curto (~16 µs por camada
   contra ~10 do llama.cpp, limitada por latência com 8 warps na GPU inteira)
   e ~4 µs de rampa+cauda por matvec pequeno (o e QKV a 88%).

   **Tentado e descartado (medido, não repita sem motivo novo):** prefetch
   para o L2 num segundo stream durante os kernels pequenos (1–2% mais lento: os
   ramos extras do grafo custam mais do que o L2 poupa); prioridade de GPU do
   processo via `D3DKMTSetProcessSchedulingPriorityClass` (nenhum efeito, nem
   com outra carga CUDA na placa); 1 ou 2 linhas por warp no GLU do decode (1
   cai para 85% da banda, 2 empata com 4); buffer duplo K/V na atenção de decode
   (66 contra 52 µs a 3072).

   **Medir exige alternar.** Nesta máquina outro programa que desenha na GPU
   (navegador, o próprio app de chat) estica um kernel em ~300 µs a cada ~4 ms —
   ~8% do decode — e some quando o programa para. Os dois motores sofrem igual;
   só execuções intercaladas na mesma janela de tempo são comparáveis.
2. **Multi-modal.** `mmproj-f16.gguf` é um ViT de 27 blocos (patch 16, embd
   1152, ffn 4304) com projetor `qwen3vl_merger`, merge espacial 2x2 e camadas
   deepstack. Estrutura já mapeada via `inspect --tensors`. Depende do item
   anterior porque o modelo de texto que o acompanha é qwen3.6.

3. **KV cache paginado** para múltiplas sequências concorrentes. Hoje o
   servidor serializa requisições porque o cache é uma sequência só.

4. **MTP.** As camadas `nextn.*` são carregadas? Não — são puladas. Servem para
   decodificação especulativa, não para a saída principal.

5. **Template Jinja.** O GGUF traz `tokenizer.chat_template`, que hoje é
   armazenado mas não avaliado; usamos ChatML embutido.

---

## 8. Estilo

- Comentários explicam **por quê**, não o quê. Se um comentário parafraseia o
  código, apague.
- Toda mensagem de erro deve dizer ao usuário **o que fazer**. Compare:
  "failed to load model" (ruim) vs. a mensagem em `error.rs` que diz quanta
  VRAM faltou e quais flags mudar (bom).
- Todo bloco `unsafe` precisa de um comentário `// SAFETY:` justificando.
- Testes verificam comportamento, não implementação. Prefira testes que
  falhariam se a lógica estivesse errada, não só se ela mudasse.
