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
| API no formato do llama-server (OpenAI chat/completions/responses, Anthropic messages, nativa, infill, SSE, auth, reasoning, tool calls, logprobs) | 56/56 em `tests/api_conformance.py` |
| Amostragem (top-k/p, min-p, typical, top-n-sigma, 3 penalidades, logit_bias, DRY, XTC, dynatemp, mirostat 1/2) | testada |
| Geração restrita (GBNF, JSON schema, `response_format`) | testada; saída idêntica à do llama-server |
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
# CUDA_PATH do sistema aponta para o 12.0, que não conhece sm_120:
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"

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

**NVFP4** (Qwen3.8-27B NVFP4) segue o caminho inteiro em todos os tamanhos,
inclusive no decode (`KernelType::prefers_q8_decode`): a 4,5 bits o vetor f32
que cada warp relê pesa mais que os pesos. Os nibbles FP4 viram int8 com
`__byte_perm` (`fp4_codes8` em `dequant.cuh`) — a tabela `__constant__` antiga,
indexada por um nibble diferente em cada lane, serializava e deixava o 27B em
5 tok/s. Lotes de 2–8 tokens (o verify do MTP) usam `matvec_q8_multi_kernel`,
que decodifica cada grupo uma vez (`mmq_decode`) e o gasta em todos os tokens.
NVFP4 tem MMQ assíncrono e gate/up fundido (`mmq_async<NVFP4>`, blocos de 36
bytes copiados como os do Q8_0). O decode NVFP4 usa `matvec_q8_wide_kernel`
(128 threads cobrem o K de 2 linhas: com um warp por 4 linhas, uma linha de 5120
era só 5 iterações por lane e a banda ficava em 73%); o verify do MTP usa a
versão com vários tokens (`matvec_q8_wide_multi_kernel`).

**Prefill NVFP4 no Blackwell (sm_120): tensor cores FP4.**
`kernels/sm120a/fp4mma.cu`, compilado à parte para `compute_120a` (o
`mma ... kind::mxf4nvf4` só existe no alvo específico; o `build.rs` gera
`airun_fp4.lib`). Os pesos entram como estão: a palavra t de um bloco NVFP4 já
é o registrador do fragmento A e a palavra de escalas `d[4]` é o registrador de
escala; a ativação é quantizada para o mesmo layout de bloco, mais uma escala
f32 por token. Ativação em 4 bits, como o llama.cpp faz: na perplexidade em lote
do 27B (blocos de 128, texto LF) int8 2,0727, FP4 2,0838, llama.cpp GPU 2,0862.
Lotes de 32+ tokens; `AIRUN_NO_FP4=1` volta ao MMQ int8. O SwiGLU vai dentro da
quantização da entrada da `down` (`quantize_glu_nvfp4`). O quantizador tem duas
formas bit-idênticas: a vetorizada (4 lanes por run de 16, float4 coalescido,
linha em registradores até K = 20480) para linhas alinhadas em 16 bytes e a
escalar para o resto; o selftest compara as duas. A vetorizada levou o SwiGLU
de 214 para 66 µs por camada (512 tokens, K 17408) e o pp2048 de 3623 para
3690 t/s. Pôr o SwiGLU no epílogo do GEMM do `up` (com amax por linha via
atomicMax) foi medido mais lento: 3450/3380 t/s. Escalas `0x7F` (zero
para o ggml, NaN para o hardware) são zeradas no carregamento.

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

**Prefill com contexto longo (2026-11-01).** O prefill de prompts longos
perdia de 28% a 37% para o llama.cpp com o 27B dividido entre as duas GPUs.
Duas causas, nessa ordem de peso:

1. **As GPUs rodavam em série.** O `handoff` entre shards sincronizava a GPU
   de origem pelo host, e o `airun bench` lia os logits de todo pedaço de
   prompt, o que sincroniza também. Uma 5060 Ti (metade dos SMs) atrás da
   5070 Ti derrubava 4k de prompt de 3653 (uma GPU) para 2501 tok/s; o
   llama.cpp, que encadeia as GPUs, mantinha 3514.
   - Agora o `handoff` só ordena por eventos: o stream de destino espera o
     `sent` da origem, e a origem espera o `copied` do destino antes de
     sobrescrever `x`. A GPU 0 roda o pedaço c+1 enquanto a GPU 1 roda o c.
   - O bench usa `Runtime::prefill_chunk` nos pedaços intermediários, como o
     `generate`.
   - Quem enche um buffer pinado no host precisa esperar o upload anterior,
     porque o host não para mais entre os pedaços. O PLE tem o evento
     `uploaded`; o MoE já sincronizava. Split: 4k a 3566 tok/s.
2. **A atenção do prefill estava em 55 TFLOPS** com head_dim 256
   (`crates/airun-cuda/tests/attention_bench.rs`, ignorado, mede isolado).
   Com 255 registradores só cabe um bloco de 8 warps por SM, e os dois
   warps de cada partição fazem a softmax juntos enquanto o tensor core
   espera, então o que pesa é o custo por tile fora das MMAs. Três cortes
   levaram a 72,7 TFLOPS:
   - **P·V em fp16 por tile de chaves:** soma de no máximo 32 termos ≤ 1
     vezes V, somada a um O em fp32. É a MMA de taxa dupla, sem guardar uma
     soma longa em fp16; o llama.cpp guarda o O inteiro em fp16. O kernel
     só usa isso quando o lote vê 4096+ chaves (`keys` de
     `attention_masked`). Em prompt curto a soma em fp16 virava um token
     quase empatado do Qwen4 contra a referência, e o ganho é pequeno.
     Perplexidade com pedaços de 8k: 1,2960 contra 1,2999 em fp32.
     `AIRUN_ATTN_PV_F32=1` desliga.
   - **Caminho sem máscara:** um tile inteiro antes de todas as consultas
     do warp não precisa de máscara nem das guardas de `-inf`.
   - **Reescala preguiçosa:** o O só é reescalado quando o máximo de alguma
     linha muda.

Agrupar as cabeças do GQA não reduziria o tráfego: a intensidade por tile
de K/V é a mesma, porque o kernel é limitado por cálculo, não por memória.

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
   modelo cabe. No decode o split custa: no 1.7B, 94 t/s contra 234 t/s numa
   placa só, com o contexto vazio. No prefill as GPUs se sobrepõem (veja
   "Prefill com contexto longo" acima) e o split quase não perde. Nunca
   sincronize um stream pelo host entre shards: isso põe as GPUs em série.

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

12. **`llama-perplexity -f` no Windows lê o arquivo em modo texto** e troca
    CRLF por LF; o `airun perplexity -f` lê os bytes como estão. Num arquivo com
    CRLF os dois pontuam textos diferentes e o llama.cpp parece ~0,1 nat melhor.
    Normalize para LF antes de comparar. Com o mesmo texto, `--chunk N` do airun
    reproduz o `llama-perplexity -c N` bloco a bloco (27B NVFP4, blocos de 128:
    airun 7,9356, llama.cpp CPU 7,9438). Uma referência f64 em numpy do forward
    qwen35 (fora do repo) confirmou o airun camada a camada.

13. **YaRN não é escala linear.** `rope.scaling.type = yarn` (o 27B "yarn1m")
    só interpola as dimensões lentas e multiplica cos/sin por
    `1 + 0.1 ln(fator)`; tratar o fator como escala linear muda o modelo mesmo
    em contexto curto. `RopeParams` em `ops.rs`, espelho em `airun.cuh`.

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

### Experts mapeados em RAM (padrão quando o MoE não cabe nas GPUs)

Sem `--expert-cache`, um MoE grande demais vai para `plan_experts`
(`loader.rs`): `slots` experts por camada ficam na VRAM e os demais numa
alocação fixada e mapeada (`MappedExperts` em `model.rs`), que a GPU lê pelo
PCIe. Os kernels acham cada expert por uma tabela de endereços
(`moe_*_tab`), então nada espera o host e o decode cabe num CUDA graph. O
streaming do arquivo (seção acima) continua com `AIRUN_EXPERTS=stream`.

- **Decode:** antes do MoE, `moe_stage_plan` + `moe_stage_copy` copiam para um
  staging na VRAM só os experts roteados que estão na RAM (leituras de 16 B
  coalescidas). Ler direto da RAM com os decodificadores (leituras estreitas)
  era 5x mais lento.
- **Lotes (≥ 16 tokens):** `moe_grouped` ordena os pares (token, slot) por
  expert e lê os pesos de cada expert uma vez para todos os seus tokens,
  copiando as linhas do bloco para a memória compartilhada com leituras
  largas. É bit-idêntico ao caminho por token (selftest).
- **Residência adaptativa:** o MoE conta o uso de cada expert na GPU
  (`moe_count`); a cada 8 passos `rebalance_experts` troca os mais usados que
  estão na RAM pelos menos usados da VRAM (até 16 por camada, com margem).
- **Link lento:** `plan_experts` mede a banda host→GPU. Uma GPU abaixo de
  8 GB/s (a 5060 Ti desta máquina está num slot PCIe 3.0 x4: 1,6 GB/s em
  zero-copy) recebe só camadas inteiras, com todos os experts na VRAM; as
  demais vão para a GPU de link rápido com experts parciais.
- **Grafos por GPU:** o decode captura um grafo por shard; o resíduo passa
  entre as GPUs fora dos grafos, só as linhas do lote. O PLE é preparado no
  host num buffer fixado antes de cada replay.

Qwen4 Flash-Next (78 GB, 5070 Ti + 5060 Ti, 64 GB de RAM), prompt de 792
tokens + 128 gerados (`benchmarks/2026-11-01/qwen4.ps1`): veja os números em
`benchmarks/2026-11-01/RESULTADOS.md`. O que levou a airun de 92 / 19 tok/s
para ~287 / ~39 (à frente do Strata 0.1.44, ~270 / ~33):

- **Tabelas `__constant__`:** `kvalues_iq4nl` indexada por nibbles diferentes
  em cada lane serializava até 16x. IQ4_NL/IQ4_XS agora usam `__byte_perm`
  (`iq4_codes8`), como o FP4. Só isso valia ~5 ms por token.
- **Perfil de roteamento** (`expert_profile.rs`): a carga põe na VRAM os
  experts mais usados nas execuções anteriores, e as notas do rebalanceamento
  partem do perfil. Uma troca só acontece com ~4 roteamentos de vantagem,
  porque cada uma move os dois experts pela PCIe.
- **Cópia sobreposta:** o decode copia os experts ausentes num stream lateral
  (`DeviceState.side`, eventos `fork`/`join`, que funcionam dentro do CUDA
  graph) enquanto os residentes calculam (`Staged.only`).
- **Router em um warp** (`moe_router_warp_kernel`): 25 → 4 µs por camada.
- **MoE agrupado em tensor cores** (`moe_mma_*`, lotes de 64+): decodifica
  cada fatia de 64 de K uma vez para fp16 e usa `mma.sync`. Experts na VRAM
  são lidos direto, sem copiar as linhas, o que dá ocupação 3x maior; os da
  RAM são copiados com leituras largas. Abaixo de 64 tokens ficam os kernels
  f32, o que mantém prompts curtos idênticos à referência.
- **PLE em paralelo** (`gather_pool.rs`): a tabela de 26,8 GB não cabe no
  cache de arquivos, então cada linha é uma leitura aleatória no SSD. Lidas
  uma a uma, deixavam a GPU parada 1,4 s num prompt de 792 tokens. 32 threads
  persistentes as leem juntas.

A carga leva ~80 s (fixar ~24 GB de RAM).

### Atenção esparsa QSA (qwen4exp)

As camadas de atenção com `compress_ratio` r > 0 só atendem a `top_k + r - 1`
células (2051 no Qwen4): o bloco final incompleto e os blocos de r tokens com
maior nota. Kernels em `kernels/qsa.cu`, wrappers em `airun-cuda/src/qsa.rs`,
orquestração em `Runtime::qsa_select`:

- `qsa_store`: a chave crua do indexador de cada token vai para um anel f16
  (`QsaCache.ring`, como o cache do indexador do llama.cpp, que é f16).
- `qsa_pool`: quando o último token de um bloco chega, média das r chaves →
  RMSNorm (`indexer.k_norm`) → RoPE na primeira posição do bloco. Fica em
  `QsaCache.pooled` ([ctx/r][128] f32).
- `qsa_score`: nota = Σ_cabeças relu(q_h · chave), q com RMSNorm e RoPE da
  posição do token. Pulado quando o contexto da query cabe no orçamento.
- `qsa_select`: radix select; empate vai para o bloco de menor índice e um
  bloco pego em parte fica com as primeiras células (o mesmo que o
  `qsa_select.cu` do Strata). Saída: lista `[count-1, células...]` no decode,
  que `attention_split_sel` usa no lugar da posição (cabe no CUDA graph), ou
  uma máscara de bits por query nos lotes (`attention_masked`, que o flash2 usa
  para pular tiles sem nenhuma célula escolhida).

Com contexto ≤ 2051 a seleção é tudo, e a saída é idêntica à densa. O selftest
compara pooling, notas, seleção e as duas atenções com a CPU. Perplexidade em
pedaços de 4096 (posições 2048–4095 pontuadas) num texto de 25 mil tokens:
referência 5,1563, airun com QSA 5,1684, airun densa (`AIRUN_QSA=off`) 5,1717.
Em 2048 (tudo denso) a diferença de base airun vs referência já é ±0,5% por
pedaço, por arredondamento.

### API HTTP (`airun serve`) no formato do llama-server

As rotas, os campos de requisição e as respostas seguem os do `llama-server`,
para que clientes feitos para ele funcionem sem mudança:
- **Saúde e metadados:** `/health`, `/v1/health`, `/props`, `/models`,
  `/v1/models`, `/slots` e `/metrics` (501, como o llama-server sem
  `--metrics`).
- **Geração:** `/completion` e `/completions` (nativos), `/infill`,
  `/v1/completions`, `/chat/completions` e `/v1/chat/completions`,
  `/v1/responses` (e `/responses`, OpenAI Responses) e `/v1/messages`
  (Anthropic Messages), todos com stream.
- **Contagem:** `/v1/chat/completions/input_tokens`,
  `/v1/responses/input_tokens` e `/v1/messages/count_tokens` (com e sem
  `/v1`).
- **Utilitários:** `/tokenize` (com `with_pieces`), `/detokenize` e
  `/apply-template`.
- **Embeddings** (`--embeddings`): `/embedding(s)` nativo e
  `/v1/embeddings` (float ou base64), do estado final (depois da norma de
  saída) via `Runtime::hidden_states`; pooling `--pooling
  none|mean|cls|last` (padrão: o `pooling_type` do GGUF, senão `last`, que
  num modelo gerativo é o token que leu o texto todo), `embd_normalize` como
  no llama.cpp. Sem a flag: 501.
- **Slots** (`--slot-save-path DIR`): `POST /slots/0?action=save|restore|
  erase` gravam/carregam o cache inteiro (`Runtime::save_state`/
  `load_state`: linhas de KV usadas, matrizes delta-net, conv, anel QSA,
  histórico PLE). Num híbrido é o único jeito de retomar um prompt longo sem
  reprocessar; a continuação sai idêntica token a token. O arquivo do 27B é
  grande (~283 MB em 1925 tokens: as 48 matrizes delta-net são f32). Nome de
  arquivo sem diretório. Sem a flag: 501.
- **Raciocínio controlado:** `reasoning_budget_tokens` (ou
  `thinking_budget_tokens`, ou `--reasoning-budget`) e
  `reasoning_budget_message` fecham o `<think>` à força depois de N tokens
  (`ReasoningBudget` no amostrador); com `reasoning_control: true`,
  `POST /v1/chat/completions/control {"id", "action": "reasoning_end"}`
  fecha o pensamento de uma geração em andamento (vale para chat, Responses
  e Messages). Ambos desligam o argmax na GPU e o MTP daquela requisição.
- **Sem suporte, respondendo como o llama-server sem a feature:** rerank
  (`/rerank`, `/reranking`, `/v1/rerank(ing)`; precisa de um modelo
  reranker com cabeça de classificação), `POST /props`,
  `/v1/audio/transcriptions` (501; os modelos não têm encoder de áudio);
  `/lora-adapters` (lista vazia / `success`). Esses handlers leem o corpo
  antes de responder: sem isso a conexão fechava com o cliente ainda
  enviando e ele via um reset em vez do 501.

Cada resposta traz `timings`, `usage.prompt_tokens_details.cached_tokens` e
`system_fingerprint`; o stream respeita `stream_options.include_usage`. Erros
vêm como `{"error": {"code", "message", "type"}}`.

- **Template:** o prompt de chat é o template Jinja do próprio GGUF
  (`chat.rs`, minijinja com métodos de string do Python), como o `--jinja` do
  llama.cpp. Ferramentas, respostas de ferramentas, `enable_thinking` e
  `reasoning_effort` (no corpo ou em `chat_template_kwargs`) chegam ao
  template. Argumentos de tool call enviados como string JSON viram objetos
  antes.
- **Resposta:** é dividida em `reasoning_content` (o bloco `<think>`),
  `content` e `tool_calls` (`<tool_call>` no formato XML
  `<function=..><parameter=..>` do Qwen3.5+ ou JSON do Qwen2.5/3), com
  `finish_reason: tool_calls`. `reasoning_format: none` (ou
  `--reasoning-format none`) deixa o pensamento no `content`.
- **Responses e Messages** são convertidos para a forma de chat
  (`responses_to_chat`, `anthropic_to_chat`: `instructions`/`system`,
  itens `function_call`/`function_call_output`, blocos `tool_use`/
  `tool_result`/`thinking`, `text.format`, `thinking.type`,
  `stop_sequences`) e passam pelo mesmo caminho (`chat_job`,
  `stream_chat`); só a saída muda. Os eventos do Responses levam
  `sequence_number`/`output_index` como os da OpenAI; os blocos do Messages
  fecham um antes do próximo abrir, como na Anthropic.
- **Amostragem:** a cadeia completa do llama.cpp (penalties → DRY →
  top-n-sigma → top-k → typical → top-p → min-p → XTC → temperatura, com
  temperatura dinâmica; mirostat 1/2 substitui o que vem depois do DRY). Os
  `dry_sequence_breakers` viram uma máscara de tokens (o token cujo texto
  contém um breaker). `ignore_eos` é `logit_bias` -inf nos tokens de fim
  (reportado como `{"bias": null, "token"}`, como o llama-server).
- **Gramática** (`airun-core/src/grammar.rs`): GBNF completo (regras,
  grupos, classes, `* + ? {m,n}`, escapes), autômato com pilhas sobre code
  points (um caractere partido entre tokens é carregado), e conversor de JSON
  schema no estilo do llama.cpp (ordem das propriedades preservada; por isso
  o `serde_json` usa `preserve_order`). `grammar` > `json_schema` >
  `response_format`. A máscara testa os 256 tokens mais prováveis e só cai
  para o vocabulário inteiro se nenhum servir. Com o prompt abrindo
  `<think>`, a gramática espera o `</think>` (`Grammar::after`): o modelo
  pensa livre e a restrição vale para a resposta. Custo: ~38 contra ~49
  tok/s no 27B.
- **Probabilidades:** `n_probs` (nativo), `logprobs: N` (/v1/completions) e
  `logprobs` + `top_logprobs` (chat) vêm do softmax dos logits crus, antes
  dos amostradores, como no llama.cpp, inclusive o token de fim; também no
  stream.
- **Infill:** `<|repo_name|>myproject`, cada `input_extra`, `<|file_sep|>
  filename`, depois `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>` — o
  formato do llama.cpp. `<|fim_pad|>`, `<|repo_name|>` e `<|file_sep|>`
  contam como fim de geração no tokenizer, como no llama.cpp.
- **Verificação:** `tests/api_conformance.py` grava as respostas de um
  servidor e as compara chave a chave (e os nomes de evento SSE) com as de
  outro. Em 2026-11-01, com o Qwen3.8-27B, deu 56/56 contra o llama-server
  b11760: gramática, JSON schema, `ignore_eos`, `tool_choice: none`,
  `json_object`, tool calls e infill com o mesmo texto; `n_probs` dentro de
  0,002–0,013. No prompt de chat curto sem pensamento o "Hi" sai com
  logprob -0,175 contra -0,42 no llama.cpp, com prompt e ids idênticos. O
  caminho todo em f32 do airun (`-b 1`, `AIRUN_MATVEC_F32=1
  AIRUN_NO_FP4=1 AIRUN_NO_MMQ=1`) dá -0,183: o airun normal fica a 0,01 do
  cálculo exato e o llama.cpp a 0,24 (quantização das ativações dele).
- **Diferença de padrão:** pedidos OpenAI sem `repeat_penalty` usam o
  `--repeat-penalty` do servidor (1,1); o llama-server usa 1,0. O
  `/completion` nativo usa 1,0, como o llama-server.
- **Ainda não implementado:** rerank, áudio e LoRA (nenhum dos modelos
  alvo tem cabeça de rerank, encoder de áudio ou adaptadores).
- **Bug corrigido junto (2026-11-01):** com `-b 1` o prefill passava token a
  token pelo grafo de decode sem sincronizar, e cada replay lia o token e a
  posição já sobrescritos no buffer pinado ("The capital of France is" →
  "a 2013 American comedy"). `forward_decode_graphed` agora espera o replay
  anterior antes de reescrever os slots; o decode não muda (48,4 tok/s).

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

4. **MTP — feito para o qwen35 denso (2026-10-31).** `--draft N` (padrão 3)
   carrega `blk.{n_run}` como cabeça de rascunho e decodifica especulativamente
   quando a amostragem é gulosa pura (`--temp 0 --repeat-penalty 1`). A saída
   segue o decode normal, mas **não é garantidamente idêntica**: o verify roda
   lotes de 2–4 tokens em outros kernels (`matvec_q8_multi`, ordem de soma
   diferente), e num quase-empate de logits o argmax pode virar — o texto
   diverge dali em diante, igualmente plausível. `crates/airun-core/src/spec.rs`: o verify roda
   as camadas delta-net sem gravar estado (guarda as entradas em `DeltaStash`) e
   `commit` empurra só os tokens aceitos — `test_delta_defer` no selftest exige
   igualdade bit a bit. A linha do KV do MTP na posição q é o par
   (h_{q-1}, x_q), como no draft-mtp do llama.cpp. Entre dois verifies,
   `advance_graphed` roda commit + cabeça sobre as linhas do verify + os k
   rascunhos encadeados (o pick de cada passo vira o token do seguinte no
   device) como um CUDA graph por contagem aceita m: um lançamento e uma
   leitura, onde antes o host esperava cada passo da cabeça. Só com uma GPU e
   grafos ligados. 27B NVFP4 numa 5070 Ti: 93.8–113.6 tok/s contra 49 sem MTP
   (llama.cpp com draft-mtp, mesmos prompts: 91.6–104.2).
   Falta: cabeça MTP dos MoE (35B-A3B), amostragem não gulosa.

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
