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
| Kernels CUDA (matvec f32/dp4a/tensor-core, attention, rope, rmsnorm, swiglu, embed, kv) | 78/78 no selftest |
| Runtime Qwen3 denso | gera texto coerente |
| Multi-GPU (split por camada) | validado com 2 placas |
| Planner de VRAM | testado, com fallback automático |
| API OpenAI (chat, completions, SSE, auth) | validada |
| Amostragem (top-k/p, min-p, typical, top-n-sigma, 3 penalidades, logit_bias) | testada |
| Conversão HF → GGUF | Qwen3-8B convertido e executado |
| Benchmark | funciona |

**Arquiteturas suportadas hoje:** `qwen3`, `qwen2`, `llama` (decoders densos).

**Não suportado ainda** (o loader recusa com mensagem clara):
- `qwen35` / `qwen35moe` — camadas híbridas SSM/atenção + MoE. Veja a seção 7.
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
outro). Lotes de 64+ vão para tensor cores fp16 (`matvec_wmma_kernel`), que
serve todos os 17 tipos via os mesmos `decode_group`. Cada caminho tem seus
casos no selftest com limite de erro backward; `AIRUN_MATVEC_F32=1` força o
exato em tudo, `AIRUN_NO_WMMA=1` desliga só o tensor-core.

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

## 7. Híbridos SSM/MoE — estado e o que falta

**Implementado e rodando, mas a saída ainda não está correta.** Leia isto antes
de mexer.

O que já funciona:
- Config, carga de pesos, planner e split multi-GPU para `qwen35` e `qwen35moe`
- Todos os kernels novos validados no `selftest` (delta rule, conv1d causal,
  l2norm, gates, rmsnorm gated, router MoE, FFN de experts)
- Qwen3.5-9B carrega e gera; Qwen3.6-35B-A3B carrega e divide entre as 2 GPUs
- Recuperação de longo alcance funciona (71,3% de acerto em texto repetido,
  contra 77,0% do 1.7B denso)

O que não funciona — **a semântica**, com a gramática intacta:
`"The capital of Japan is"` → `" a Japanese TV show"`. Em número:

| Modelo | Perda | Top-1 |
|---|---|---|
| Qwen3-8B Q4_K (denso, controle) | 2.20 | 50.9% |
| Qwen3.5-9B (híbrido) | 4.40 | 20.1% |
| Qwen3.6-35B-A3B (híbrido + MoE) | 11.86 | 1.9% |

O 35B está quase no chute uniforme (`ln(248320) = 12.4`), então há um segundo
defeito no caminho MoE além do que os dois compartilham.

Bugs reais já encontrados e corrigidos (não os reintroduza):
1. O planner alocava `block_count` camadas, mas a última é MTP e não é
   carregada → índice fora dos limites. Use sempre `cfg.n_run_layers()`.
2. Os buffers `q/k/v` eram dimensionados só pela atenção. As camadas SSM
   precisam de `group_count * state_size`, que no 9B é o dobro do
   `n_head_kv * head_dim`. Escrita fora dos limites em prompts > 256 tokens.
3. **`ssm_a` já vem como `-exp(A_log)`**, não como `A_log`. Aplicar `-exp()` de
   novo multiplica o decaimento por ~13x e o estado esquece tudo a cada token.
   `load_delta_decay_coeff` normaliza pelo sinal.
4. **O gate do `rmsnorm_gated` vem antes da normalização**, não depois — é o
   que o `RMSNormGated` do Mamba-2 faz, e faz a normalização dividir a escala
   do próprio gate de volta. Vale 4.89 → 4.40 de perda.

**Todas as ambiguidades de layout já foram varridas e medidas** — ordem de
`[q][k][v]`, agrupamento por k-head, `alpha`/`beta`, metade do gate em
`attn_q`, ordem dos taps da conv, pareamento do RoPE, RoPE parcial vs total,
l2norm, escala de q, SiLU da conv, SiLU vs sigmoid no gate, interpretação do
`ssm_a`. A tabela com os números de cada uma está em `tasks.md` §3; a lista das
variáveis de ambiente está em `docs/FLAGS.md`. **Não refaça essa varredura.**

Também já descartados: tokenizador, prefill em lote, camada MTP, decodificador
Q5_K, mRoPE (com texto puro reduz exatamente ao RoPE padrão), offset DC no
residual e a semântica do `post_attention_norm`.

A pista mais forte que sobrou: **o híbrido quase não usa posição.** Desligar o
RoPE inteiro (`AIRUN_ROPE_OFF=1`) custa +2.75 nats no 8B denso e **+0.04** no
9B híbrido. Por isso todos os testes de RoPE deram empate — não há sinal
posicional chegando à saída para eles mexerem. Os números completos e a
ressalva (o delta-net carrega posição implícita, então parte disso pode ser
legítimo) estão em `tasks.md` §3.

O resto está no mesmo lugar: o caminho MoE (isolado pelo 35B), e uma referência
CPU de uma *camada* delta-net inteira com os pesos reais, que é o oráculo que
falta (`dequant.rs` é o oráculo dos kernels; não existe um para composição).

## 8. Outras frentes

1. **Decode que não cai com o contexto.** Esta é agora a maior perda medida.
   A atenção lança `n_tokens * n_heads` blocos, o que no decode dá 16 blocos
   numa placa de 70 SMs. O custo por posição de KV é de *latência*, não de
   banda: a 3072 tokens o kernel gasta ~12x o que a leitura do cache justifica.
   Já foi feito o que dava dentro de um bloco — 32 warps no decode contra 8 no
   prefill (`ATTN_WARPS_DECODE`/`ATTN_WARPS_PREFILL` em `ops.cu`), que dobrou a
   taxa a 3072. O passo que falta é dividir o intervalo de KV entre blocos
   (`gridDim.z`) e combinar os parciais `(max, sum, acc)` num segundo kernel,
   como o `flash_attn_ext` do llama.cpp. Precisa de um buffer de scratch de
   `n_tokens * n_heads * n_splits * (head_dim + 2)` floats — ~133 KB no decode.

   Use `airun bench --decode-at 16,512,1024,2048,3072` para medir. **O bench sem
   `--decode-at` mede só com o contexto vazio e é cego para isso** — foi assim
   que a regressão passou despercebida até um usuário notar na conversa.

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
