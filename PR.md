# PR: NVFP4 global-scale fix, `serve --draft`, MTP warnings, help & docs

**Branch:** `fix/nvfp4-global-scale-and-serve-draft` → `main`
**Status:** pronto para revisão; o autor continua testando/modificando antes de aprovar.

---

## Resumo

Corrige um bug que fazia **modelos NVFP4 de fora (vllm/NVIDIA)** gerarem lixo
(perplexidade `NaN`, geração `!!!!`) e adiciona a flag `--draft` ao `airun serve`,
avisos não-silenciosos para a decodificação especulativa, e melhora a
documentação (help raiz, FLAGS.md, README, AGENTS.md).

O bug era real e reproduzível: o mesmo arquivo NVFP4 funcionava no llama.cpp e
quebrava no airun, enquanto um denso Q4_K da mesma arquitetura funcionava.

---

## 1. Bug principal: NVFP4 global scale não era lido

### Sintoma
- `airun run`/`serve` num NVFP4 (ex. Qwen3.8-27B "Cold Fusion", gerado pelo
  vllm): geração em loop (`!!!!`), `airun perplexity` = **NaN**.
- O mesmo arquivo no **llama.cpp**: texto coerente (arquivo íntegro).
- Um **Q4_K da mesma arquitetura** (`qwen35`, 65 blocos, mesmo ssm/head): funciona.

### Causa raiz
O NVFP4 existe em **duas convenções**:

1. **Autocontida** (encoder próprio do airun, estilo MXFP4): o bloco de 64
   elementos carrega as 4 escalas UE4M3 e não tem tensor-irmão.
2. **Global-scale** (vllm/NVIDIA, arquivos de fora): cada peso NVFP4 tem um
   sidecar `<base>.scale` F32[1] (≈1e-4). O dequant por bloco sai ~10⁴ pequeno
   sem ele; a magnitude real vem de `bloco × .scale`.

O airun **nunca lia o sidecar** (nenhuma referência a `.scale`/`global_scale`
no código). Dequantizar os bytes reais do arquivo confirma a direção:
`raw` → maxAbs **132** (≈2750× acima do sensato); `raw × .scale` → maxAbs
**0.048** (magnitude de peso normal). Sem o fator, os pesos NVFP4 saíam ~2750×
maiores → hidden states estouravam para **inf/NaN** a partir da ~5ª camada →
logits NaN → `!!!!`.

### Por que os testes existentes não pegavam
- **`selftest`**: compara GPU-vs-CPU com pesos **sintéticos no layout do próprio
  kernel** — nunca lê bytes reais nem aplica um sidecar. Passava com o bug.
- **`AIRUN_MATVEC_F32=1`**: força o caminho f32 exato, mas o que faltava era o
  **fator global**, não o kernel — então continuava NaN.
- **`perplexity`**: a métrica que pega (NaN → 3.93 depois do fix).

### Correção
- `GpuTensor::global_scale` (`model.rs`): lido do sidecar `<base>.scale` F32[1]
  no load (função `global_scale_of`); `1.0` quando não há sidecar (todo
  formato não-NVFP4, e NVFP4 autocontido).
- Epílogo de escala em todos os caminhos de projeção (`runtime.rs`):
  - `matvec` / `matvec_reuse` (decode, q8, prefill FP4) via `apply_global_scale`.
  - Projeções de saída diretas (2 sites) com `ops::scale`.
  - **Fusões GLU** (gate+up num lançamento): o scale **desliga a fusão** quando
    presente, porque `SiLU` não é linear-escala (`SiLU(g·x) ≠ g·SiLU(x)`); o
    caminho separado escala `gate` e `up` antes do SwiGLU.
- **Sem regressão:** para Q4_K/IQ/MXFP4 (sem sidecar) `global_scale == 1.0` e o
  epílogo não roda — saída bit-idêntica.

### Verificação
| Teste | Antes | Depois |
|---|---|---|
| `perplexity` 27B NVFP4 | **NaN** | **3.93** (top-1 61%) |
| `run --temp 0` 27B NVFP4 | `!!!!` | texto coerente (idêntico ao llama.cpp) |
| `serve` + `/v1/chat/completions` 27B NVFP4 | lixo | responde certo |
| Q4_K denso (mesma arch) | ok | ok (invariado) |
| MoE 35B-A3B Q4_K | ok | ok (invariado) |
| `cargo test --release` | 101 | **101/101** |

---

## 2. `airun serve` ganha `--draft`

- A flag **não existia** no `serve` (só no `run`). Quem passava `--draft` ao
  `serve` recebia "unexpected argument".
- `serve` agora tem `--draft <N>` (padrão `0`), e o `run` continua com padrão `3`.
- O `serve` repassa `draft` para o `LoadOptions` via `run_args.draft`.

---

## 3. Decodificação especulativa (MTP): não mais silenciosa

O MTP **já funcionava** (27B: `--draft 3 --temp 0 --repeat-penalty 1.0` →
**72% acceptance**, ~3× mais rápido). O problema era a **ativação silenciosa**:
- O gate exige amostragem **gulosa pura** (`temp 0` **e** nenhuma penalidade).
- `run`/`serve` têm `--repeat-penalty 1.1` por padrão → qualquer penalty
  desligava a especulação **sem aviso**. O usuário passava `--draft 3 --temp 0`,
  não recebia especulação e não sabia por quê.

Agora:
- **No load:** se `--draft > 0` mas o arquivo não tem cabeça MTP utilizável
  (MoE, ou sem `nextn_predict_layers`), imprime `note:` explicando o motivo.
- **Na geração:** se o modelo suporta MTP mas o sampler não é guloso puro,
  imprime `note:` dizendo exatamente o que desliga (`--temp 0` e
  `--repeat-penalty 1.0`).
- `run`/`serve`/`bench`/`perplexity`/`selftest`/`quantize`/`convert`/`inspect`
  agora têm `--draft` documentado com o requisito de greedy.

> **Ainda não implementado:** cabeça MTP de modelos **MoE** (`qwen35moe`). O
> aviso deixa isso explícito em vez de silenciar.

---

## 4. `airun --help` (raiz)

- `long_about` agora lista as **opções comuns por comando** (`-c`, `--kv-type`,
  `-b`, `--gpus`, `--alloc`, `--main-gpu`, `--draft`, `--temp`, `--port`,
  `--alias`, `-p`, `-n`, `--hf`, `--from/--to`) e **exemplos** prontos.
- Cada comando mantém a lista completa via `airun <COMMAND> --help`.

---

## 5. Documentação

- **`docs/FLAGS.md`**: `--draft` atualizado (dois comandos, requisito de greedy
  com o `--repeat-penalty 1.1` padrão, avisos).
- **`README.md`**:
  - "Modelos e formatos": nota "NVFP4 em duas convenções".
  - "Correção numérica": item do bug global-scale (NaN → 3.93) como lição de por
    que o selftest não basta.
  - "Resultados": **nota de condição da vantagem no decode** — os números do 27B
    são com o modelo **residente na VRAM** (regime multi-GPU/longo-contexto e
    MoE que cabe). Quando não cabe (27B num GPU de 8 GB), o gargalo muda:
    airun lê as camadas excedentes pela **PCIe** a cada token; llama.cpp as
    derrama na **CPU** (medido: 0,87 vs 0,4 t/s) e ganha também no prefill.
- **`AGENTS.md`**:
  - §6 **armadilha #14**: as duas convenções NVFP4, o bug, por que o selftest
    não pegava, e o script de diagnóstico.
  - §8.4 (MTP): `--draft` nos dois comandos + bloco "Ativação não silenciosa".
- **`benchmarks/`**: 3 utilitários de diagnóstico (parser GGUF v3):
  - `gguf_tensors.py` — lista tensores (nome, tipo, dims).
  - `meta_dump.py` — metadados de arquitetura.
  - `nvfp4_magnitude.py` — dequantiza bytes reais com/sem `.scale` (o teste que
    provou a direção do bug).

---

## Arquivos alterados

```
crates/airun-core/src/model.rs     # GpuTensor.global_scale + global_scale_of
crates/airun-core/src/runtime.rs   # epílogo de escala + fusões GLU + avisos
crates/airun-core/src/loader.rs    # aviso no load quando --draft é ignorado
crates/airun-core/src/lib.rs       # reexporta mtp_supported
crates/airun-cli/src/serve.rs      # --draft + aviso + repasse para LoadOptions
crates/airun-cli/src/run.rs        # texto do --draft (requisito de greedy)
crates/airun-cli/src/main.rs       # help raiz (opções comuns + exemplos)
docs/FLAGS.md                      # --draft atualizado
README.md                          # NVFP4, correção, condição do decode
AGENTS.md                          # armadilha #14 + MTP não-silencioso
benchmarks/{gguf_tensors,meta_dump,nvfp4_magnitude}.py
```

## Como verificar (checklist do revisor)

```powershell
cargo test --release                     # 101/101
.\target\release\airun.exe selftest     # kernels vs referência CPU
.\target\release\airun.exe perplexity <27B-NVFP4.gguf>   # ~3.93 (não NaN)
.\target\release\airun.exe run <27B-NVFP4.gguf> -p "2+2?" -n 20 --temp 0
.\target\release\airun.exe run <27B-NVFP4.gguf> -p "hi" -n 16 --temp 0 --repeat-penalty 1.0 --draft 3 --stats
.\target\release\airun.exe --help       # opções comuns + exemplos
```
