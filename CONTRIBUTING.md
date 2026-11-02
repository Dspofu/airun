# Contribuindo com o airun

Obrigado pelo interesse. O airun é uma engine de inferência escrita do zero em
Rust + CUDA, e algumas regras existem porque um erro aqui raramente trava: ele
produz texto *quase* certo. Este guia diz o que é aceito, como montar o
ambiente e como provar que uma mudança está correta e é mais rápida.

Antes de mexer em código, leia o **[AGENTS.md](AGENTS.md)**. Ele tem o estado
verificado do projeto, o mapa do código e as armadilhas que já custaram dias.

---

## As regras que não mudam

1. **Nada de dependências de inferência externas.** Sem llama.cpp, ggml,
   PyTorch, ONNX, cuBLAS, cuDNN ou CUTLASS no caminho de execução. Todo kernel
   está em `crates/airun-cuda/kernels/` e é deste projeto. Utilitários (clap,
   serde, tokio, axum, memmap2, half, minijinja) são bem-vindos. Ler o código de
   outro projeto para entender um formato é ótimo; copiar código não é. Se um
   dado ou desenho vier de outro projeto, cite-o no comentário e no
   [NOTICE](NOTICE).

2. **Todo formato de quantização existe duas vezes:** em Rust, legível, em
   `crates/airun-gguf/src/dequant.rs`, e em CUDA, em
   `crates/airun-cuda/kernels/dequant.cuh`. Mudou um, mude o outro e rode
   `airun selftest`. O selftest é a única defesa contra um layout trocado.

3. **Toda mensagem de erro diz ao usuário o que fazer.** "falhou" não serve;
   "não cabe: aumente --ctx-size ou use --kv-type q8_0" serve. Veja
   `crates/airun-core/src/error.rs`.

4. **Desempenho se prova com número, e correção também.** Uma otimização entra
   com a medição antes e depois; uma mudança numérica entra com a perplexidade
   antes e depois. "Parece mais rápido" e "o texto ficou bom" não são evidência.

---

## Ambiente

| | |
|---|---|
| GPU | NVIDIA com compute capability 7.5+ (RTX 20 ou mais nova) |
| CUDA Toolkit | 12.x ou 13.x (sm_120 precisa de 12.8+) |
| Rust | 1.85+ (o `rust-toolchain.toml` fixa a versão) |
| Compilador host | MSVC 2022 no Windows, gcc/clang no Linux |

```powershell
# Durante o desenvolvimento, compile só para a sua GPU (minutos a menos):
$env:AIRUN_CUDA_ARCHS = "120"     # 75, 80, 86, 89 ou 120
cargo build --release
```

O `build.rs` acha o CUDA e o compilador host sozinho. Se você tem mais de um
CUDA instalado, aponte `CUDA_PATH` para o certo. Não deixe `AIRUN_CUDA_ARCHS`
fixo em nada que vá para uma release: o padrão compila para RTX 20 até 50.

---

## Como verificar uma mudança

Sempre, nesta ordem:

```powershell
cargo test --release                    # testes unitários (sem GPU)
cargo build --release
.\target\release\airun.exe selftest     # cada kernel contra a referência CPU
.\target\release\airun.exe run modelo.gguf -p "2+2?" -n 20 --temp 0
.\target\release\airun.exe perplexity modelo.gguf    # perda e top-1
```

- **O selftest passa e a geração virou lixo?** O problema está no runtime
  (ordem das operações, KV cache, RoPE), não nos kernels.
- **Não julgue correção lendo o texto.** Um modelo com um layout trocado
  escreve inglês fluente sobre outra coisa. Use `airun perplexity` e compare
  com a medição de antes, no mesmo texto e no mesmo tamanho de pedaço.
- **Mexeu no `airun serve`?** Rode o teste de conformidade contra um
  llama-server com o mesmo modelo:

  ```powershell
  python tests\api_conformance.py --ref http://127.0.0.1:8090 --test http://127.0.0.1:8080
  ```

  Ele compara as respostas chave a chave e os nomes dos eventos SSE. A API do
  airun segue a do llama-server; um campo a menos é um cliente quebrado.

---

## Benchmarks

Uma medição que vai para o repositório fica em `benchmarks/<data>/` com três
coisas: o **script** que a produziu, a **saída bruta** (`.csv`, `.log`) e um
**RESULTADOS.md** com a tabela e o que mudou desde a anterior. Quem ler deve
conseguir repetir a medição com um comando.

- **Mesmas condições nos dois lados:** mesmo modelo, mesmo prompt, mesmo
  tamanho de bloco de prompt, mesma profundidade de contexto.
- **Motores alternados por rodada,** para que carga de fundo e temperatura da
  GPU atinjam todos igual. Duas rodadas no mínimo.
- **O decode não é uma taxa só:** ele cai com o contexto. Diga em que
  profundidade mediu (`airun bench --decode-at 16,32768`).
- **Diga a máquina:** GPUs, link PCIe, RAM, driver, sistema.

---

## Código

- **Comentários explicam o porquê**, não o quê: a medição que justificou uma
  escolha, o bug que uma linha evita, o formato que um layout segue. Siga a
  densidade e o tom do arquivo em que você está.
- **Nomes e comentários de código em inglês;** documentação do projeto
  (`AGENTS.md`, `docs/`, resultados de benchmark) em português.
- `cargo fmt` não é obrigatório no código existente, mas não reformate
  arquivos inteiros num PR que muda outra coisa: o diff precisa mostrar só a
  mudança.
- Um bug corrigido ganha um teste que falhava antes da correção, quando der
  para testar sem GPU; quando não der, um caso no `selftest`.
- Flags e variáveis de ambiente novas entram em
  [docs/FLAGS.md](docs/FLAGS.md).

---

## Pull requests

1. Abra uma issue antes de uma mudança grande (arquitetura nova, kernel novo,
   mudança de formato), para alinhar a abordagem.
2. Um PR, um assunto. Uma otimização e uma refatoração vão em PRs separados.
3. Descreva no PR:
   - o que mudou e por quê;
   - como verificou (testes, selftest, perplexidade antes/depois);
   - para desempenho, a tabela antes/depois e a máquina.
4. Mensagens de commit: uma linha de resumo e, no corpo, o que mudou e os
   números que importam.

Ao enviar uma contribuição, você concorda que ela é distribuída sob a
[licença Apache 2.0](LICENSE) do projeto.

---

## Reportando bugs

Inclua:

- o comando completo e a saída (`--stats` ajuda);
- `airun gpu` (GPUs, VRAM, driver);
- o modelo (nome do arquivo e de onde veio) e `airun inspect modelo.gguf`
  se for um formato incomum;
- se o problema é de correção: o prompt, a saída esperada e a obtida, de
  preferência com `--temp 0`.

Problemas de segurança no `airun serve` (autenticação, leitura de arquivos)
não vão em issue pública: use o relato privado do GitHub (aba **Security →
Report a vulnerability**).
