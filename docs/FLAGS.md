# airun options

Every option of every command. `airun <command> --help` shows the short
version.

```
airun run         generate text
airun serve       HTTP server (OpenAI + Anthropic + llama.cpp)
airun bench       measure prompt and generation
airun perplexity  measure quality on a known text
airun inspect     describe a GGUF file
airun gpu         list the GPUs
airun convert     safetensors (HuggingFace) -> GGUF
airun quantize    GGUF -> GGUF at another precision
airun selftest    check the GPU kernels against the Rust reference
```

---

## Memory, GPUs and model (apply to `run`, `serve` and `bench`)

| Option | Default | What it does |
|---|---|---|
| `-c`, `--ctx-size <N>` | automatic | Context size, in tokens. It is what uses the most VRAM after the weights. Without the option, see [Automatic context](#automatic-context). |
| `-b`, `--batch-size <N>` | `512` | How many prompt tokens are processed at a time. Larger = faster prompt and more VRAM. Does not change generation. |
| `--kv-type <type>` | `f16` | Cache format: `f16`, `q8_0` or `q4_0`. The last two fit 2x and 3.6x more context, with a very small quality loss. |
| `--gpus <list>` | all | Which GPUs to use, e.g. `--gpus 0,1`. The number is CUDA's, which may not match `nvidia-smi`'s: check with `airun gpu`. |
| `--hide-gpu <list>` | none | GPUs airun **never** touches, not even to open a context: CUDA number (as `airun gpu` shows with nothing hidden), UUID or part of the name, e.g. `--hide-gpu 5060`. Applies to any command. The others are renumbered from 0, and `--gpus`/`--main-gpu`/`--tensor-split` use the new numbering. To always hide, set `AIRUN_HIDE_GPUS` in the environment. |
| `--main-gpu <N>` | the fastest | The GPU filled first. Pick the one in the fastest PCIe slot. |
| `--alloc <mode>` | `auto` | Where weights and cache may live. `auto`: on the GPUs needed and, only if short, in RAM (cache first, then weights). `gpu`: VRAM only (if it does not fit, lowers the context or errors). `memory`: on every GPU and then in RAM. |
| `--disk-offload` | off | When VRAM and RAM run out, the leftover layers (or, for MoE, the experts) are read **straight from the .gguf file on disk** as they run, instead of the load failing. See [Disk offload](#disk-offload). |
| `--ram-reserve <MiB>` | 1/10 of RAM (2–8 GiB) | RAM left free for the system and other programs when deciding what the weights may pin. With `--disk-offload`, what exceeds it goes to disk. |
| `--tensor-split <fractions>` | — | Splits the layers by hand, e.g. `0.7,0.3`. Turns off the automatic split. |
| `--single-gpu <N>` | — | Puts the whole model on one GPU. |
| `--vram-reserve <MiB>` | `384` | VRAM left free on each GPU (Windows does not hand out the last megabytes). |
| `--no-fallback` | off | Errors instead of lowering batch or context to fit. Useful in automated tests. |
| `--expert-cache <N>` | all | MoE only: how many experts per layer stay in VRAM; the others are read from the file when the model asks for them. |
| `--draft <N>` | see [MTP](#mtp-faster-generation) | How many tokens the MTP head drafts per step. `0` turns it off. |
| `--mtp <file>` | — | The MTP head in a separate file. See [MTP](#mtp-faster-generation). |
| `--mmproj <file>` | — | The model's vision encoder. See [Images](#images). |
| `--image-max-tokens <N>` | `1024` | Maximum tokens an image may take. |

### Automatic context

Without `-c`, `run` and `serve` use **the largest context that fits the GPUs
with every weight in VRAM**, at least 4096 and at most the model's training
context. A model that fits one GPU stays on one GPU: airun does not spread the
model over two cards just to gain context. If not even the weights fit (a MoE
with experts in RAM), it stays at 4096. `bench` and `perplexity` use 4096.

### MTP (faster generation)

Qwen3.5/3.8 ship an extra layer, the "MTP head", that guesses the next tokens.
airun uses that head to draft a few tokens, and the main model checks all of
them at once. The ones it approves come for free.

- **Default:** `run` uses 4 drafts on dense models and 2 on MoEs that fit the
  GPUs whole (35B-A3B on 2× 5060 Ti: 109.6 → 137.6 tok/s). On a MoE with
  experts in RAM it is off, because checking several tokens pulls many more
  experts over PCIe (35B-A3B on one RTX 3060: 73–75 tok/s without, 65–69 with
  `--draft 1`); `--draft 2` turns it on anyway. `serve` uses 0.
- **Gain:** on the 27B, 26 → 68 tok/s greedy (`--temp 0`) and 25.7 → 47.2
  tok/s with `--temp 0.7`.
- **Works with any sampling except mirostat.** With temperature, each draft is
  accepted with the probability the model gives it, so the text follows the
  same distribution as before; MTP only changes the speed.
- **In greedy mode** the text may, rarely, come out different from decoding
  without MTP: the check adds the numbers in another order, and on a near-tie
  the choice flips.
- **Head in a separate file:** llama.cpp's converter can write the model
  without the head (`--no-mtp`) and the head into an `mtp-<name>.gguf` next to
  it. airun finds that file on its own in the model's folder (it checks it is
  from the same model), or use `--mtp <file>`.
- **No MTP** when there is an image in the prompt. With `serve --parallel`
  above 1, MTP runs only while one request is generating alone (see below).
- When MTP cannot run in a generation, airun prints a `note:` saying why.

### MoE: what is automatic

- **Experts in RAM:** a MoE that does not fit the GPUs keeps the most used
  experts in VRAM and the rest in RAM. The GPU reads the missing ones from RAM
  on a parallel channel while it computes the others, and the hottest migrate
  into VRAM as routing shifts.
- **Experts in the unused KV cache:** a long context reserves its whole KV
  cache up front. The rows the conversation has not reached yet hold copies of
  more experts (35B-A3B at 262k in q8_0: ~1,100 experts, 2.6 GB), handed back
  as the context fills. `AIRUN_NO_KV_LOAN=1` turns it off.
- **Usage profile:** airun records which experts each layer uses and stores it
  in `%LOCALAPPDATA%\airun\profiles\` (or `~/.cache/airun/profiles/`). Next
  time the most used ones already start in VRAM. `AIRUN_EXPERT_PROFILE=off`
  turns it off.

### How layers are split across GPUs

1. airun reads the **exact** size of each tensor in the file and adds up
   weights, cache and working memory.
2. **If it fits one GPU, it uses only one** (the `--main-gpu`, or the
   fastest). Moving data over PCIe without P2P costs more than the second card
   yields.
3. If it does not fit:
   - **Different GPUs:** fills the fastest first and gives the other only what
     is left. During generation the cards work one after the other, so every
     layer on the slow card costs more. (27B on a 5070 Ti + 5060 Ti: 33.7 →
     43.0 tok/s compared with splitting in half.)
   - **Equal GPUs** (within 10%): splits in half. Generation stays the same and
     the prompt almost doubles, because both work together.
4. If the context asked with `-c` does not fit, **the KV cache of the last
   attention layers goes to RAM** (only what is needed). Each pass brings only
   the cells already filled into VRAM, so in a short conversation almost
   nothing crosses PCIe. The result is identical. 27B NVFP4 on 5070 Ti + 5060,
   `-c 262144`: 15.7 → 63.0 tok/s; with `--parallel 4`, 6.1 → 60.9, and the
   context no longer dropped to 131072. In a long conversation the copy grows:
   at position 65536, 18.9 tok/s, the same as before. `AIRUN_NO_KV_RAM=1` goes
   back to the old behavior.
5. If not even with the whole cache in RAM the weights fit, the cache goes
   back to VRAM and the leftover layers stay in RAM. **During generation**
   (passes of up to 64 tokens), those layers' matmuls run **on the CPU**,
   reading straight from RAM (AVX2 kernels for NVFP4, Q4_K, Q5_K, Q6_K and
   Q8_0). Only the activations, a few KB, cross PCIe; attention, delta-net and
   norms stay on the GPU. **During the prompt** (larger chunks), they are
   copied to the GPU as before, because the CPU becomes compute-bound with many
   tokens. The result is the same as copying, with identical greedy text.

   27B NVFP4 on an RTX 5060 8 GB alone, on PCIe 3.0 x1, i7-13700 with
   DDR5-4800:

   | | before (layers copied over the link) | now |
   |---|---|---|
   | generation, no draft | 0.1 tok/s | 3.2 |
   | generation, `--draft 2` | — | 5.3 |
   | 26-token prompt | 1.1 tok/s | 10.7 |
   | 2550-token prompt | — | 40 (copied, 512-token chunks) |

   Qwen3.6-27B Q4_K_M on an RTX 3060 12 GB with ~7.5 GB in RAM: 0.2 → 4.6–4.8
   tok/s (copying to the GPU: 2.8).

   `AIRUN_RAM_ON_GPU=1` always copies again. With `--alloc gpu` none of this
   happens (the context shrinks). MoE models follow the expert plan.

   **Check the card's slot:** `nvidia-smi -q` shows the link width under "Link
   Width". An x1 slot (or riser) makes copying weights 8–16× slower than x8/x16,
   and it is the copy that weighs on a long prompt.

To take a GPU out for good (for example, the one kept for games or for the
monitor), use `--hide-gpu` or `AIRUN_HIDE_GPUS`. `--gpus` only chooses where
the model goes, and CUDA may still open a context (a few hundred MiB) on the
others.

`--stats` shows what was decided.

### When it does not fit

airun tries in this order and says what changed:

1. **Half the batch** (down to 64): only the prompt gets slower.
2. **Half the context** (down to 512).
3. **VRAM reserve at 128 MiB:** last resort.

`--no-fallback` turns these reductions off.

### Disk offload

With `--disk-offload`, the order becomes: **VRAM → RAM → disk**. The layers the
planner sends to RAM (see above) that do not fit the free RAM, after
`--ram-reserve`, are not copied: they stay in the .gguf itself, memory-mapped,
and the operating system reads them from the SSD when the CPU multiplies them,
and drops them when something else needs the memory. Nothing is pinned for
them, so they take neither RAM nor Windows' commit limit.

- While a layer runs, 4 threads bring the **next 2** in from disk
  (`AIRUN_DISK_AHEAD`, `AIRUN_DISK_THREADS`), with large reads
  (`PrefetchVirtualMemory` on Windows, `madvise` on Linux).
- **MoE:** if the experts that do not fit VRAM do not fit RAM either, they are
  read from the file when routed (the same path as `AIRUN_EXPERTS=stream`).
- `--stats` shows which layers ended up on disk.
- **MTP:** with layers on disk and no `--draft`, the head drafts 6 tokens per
  step instead of 4. Each pass reads the disk once, so more tokens per pass is
  faster (27B on NVMe: ~0.92 → ~1.05 tok/s).
- **Long prompt:** the `-b`-token chunks go together through each disk layer,
  which is read once for all of them (up to 8192 tokens at a time), not once
  per chunk. A 4096-token prompt with `-b 512` reads the disk 2 times instead
  of 8. The result is the same.

**It is slow.** Every token rereads those whole layers from disk: the speed is
the SSD's divided by the bytes on disk (an NVMe at ~5 GB/s with 5 GB on disk
gives ~1 token/s, at best). On a hard disk it becomes unusable. It is for
running a model that would not load otherwise; for speed, prefer a smaller
quantization or less context.

Without the option, when RAM cannot hold the plan, airun warns and suggests
`--disk-offload`.

---

## `airun run`

Generates text and writes it to stdout.

### Prompt

| Flag | Default | What it does |
|---|---|---|
| `-p`, `--prompt <text>` | stdin | The prompt. Without this flag, reads from stdin. |
| `--system <text>` | — | System message, inserted before the prompt. |
| `--raw` | off | Sends the prompt exactly as given, without a chat template. Use it when building ChatML yourself. |
| `-n`, `--predict <N>` | `256` | Maximum tokens to generate. |

### Sampling

| Flag | Default | What it does |
|---|---|---|
| `--temp <f>` | `0.8` | Temperature. **`0` = deterministic greedy.** |
| `--top-k <N>` | `40` | Keeps only the N most likely tokens. `0` turns it off. |
| `--top-p <f>` | `0.95` | Nucleus sampling. `1.0` turns it off. |
| `--min-p <f>` | `0.05` | Drops tokens below this fraction of the best token's probability. `0` turns it off. |
| `--typical-p <f>` | `1.0` | Locally typical sampling: keeps the tokens whose surprise is closest to the distribution's entropy, not the most likely ones. `1.0` turns it off. |
| `--top-n-sigma <f>` | `0.0` | Keeps only logits within N standard deviations of the best. `0` turns it off. Unlike top-p, the cut does **not** widen with temperature — the property the filter was proposed for. |
| `--repeat-penalty <f>` | `1.1` | Multiplicative penalty on recent tokens. `1.0` turns it off. Applied **once per distinct token** in the window, not once per occurrence. |
| `--presence-penalty <f>` | `0.0` | Subtracted once from any token already seen. OpenAI's additive scale. |
| `--frequency-penalty <f>` | `0.0` | Subtracted per occurrence of the token. OpenAI's additive scale. |
| `--repeat-last-n <N>` | `64` | Window the three penalties look at. |
| `--seed <N>` | `0` | RNG seed. `0` draws one. A fixed value makes the output reproducible. |

Filters are applied in this order, the same as llama.cpp: bias → penalties →
top-n-sigma → top-k → typical → top-p → min-p → temperature. The order changes
the result (top-p after temperature, for example, would pick other tokens). No
filter leaves the candidate list empty: at least one always remains.

### Reasoning

| Flag | Default | Description |
|---|---|---|
| `--reasoning <on\|off\|auto>` | `auto` | Turns thinking (the `<think>` block) on or off. `auto` leaves it as the model does by default. |
| `--reasoning-effort <level>` | — | How much the model may think: `none` (nothing), `low` (up to 1024 tokens), `medium` (4096), `high` (16384) or `max` (no limit). |
| `--reasoning-budget <N>` | — | Exact limit of thinking tokens; overrides the level. `-1` = no limit. |

When the limit runs out, airun closes `</think>` and the model goes straight
to the answer. If the model's own template understands levels (Qwen3.8's has
`low`, `medium` and `xhigh`), the level goes to it. All three options also
apply to `serve`, as the request default. In the API: `reasoning_effort`
(OpenAI), `reasoning.effort` (Responses), `thinking.budget_tokens`
(Anthropic) and `reasoning_budget_tokens` (llama-server).

### Images

| Flag | Default | Description |
|---|---|---|
| `--image <file>` | — | An image (PNG, JPEG or WebP) for the prompt; repeat for several. Needs `--mmproj`. Put `<__media__>` in the prompt where each goes; without a marker, they go before the text. Without `--image`, `<__media__>` is plain text. |

```powershell
airun run model.gguf --mmproj mmproj-BF16.gguf --image photo.jpg -p "What is in this image?" --temp 0
```

Image processing follows llama.cpp's (`tools/mtmd`), so the same image yields
the same description in both. Today the Qwen3.5/3.8 `mmproj` works. With an
image in the prompt MTP is off for that generation.

### Output

| Flag | Default | What it does |
|---|---|---|
| `--stats` | off | Prints the memory plan and the timing summary. |
| `-q`, `--quiet` | off | Silences load progress on stderr. |

**Examples**

```powershell
# Deterministic, one GPU
airun run model.gguf -p "Explain RoPE" -n 200 --temp 0

# Large context, forcing a split between two cards
airun run model.gguf -p "..." -c 32768 --tensor-split 0.6,0.4 --stats

# Prompt through a pipe
Get-Content prompt.txt | airun run model.gguf -n 500
```

---

## `airun serve`

Starts an HTTP API in llama-server's format: OpenAI (chat, completions,
responses), Anthropic (messages), llama.cpp's native one (`/completion`,
`/infill`) and the utilities. The sampling fields (`dry_*`, `xtc_*`,
`dynatemp_*`, `mirostat*`), `grammar`, `json_schema`, `response_format`,
`n_probs`/`logprobs` and `ignore_eos` go in the request, as in llama-server;
there are no server flags for them.

| Flag | Default | What it does |
|---|---|---|
| `--host <ip>` | `127.0.0.1` | Bind address. Use `0.0.0.0` to expose on the network. |
| `--port <N>` | `8080` | Port. |
| `--parallel <N>` (or `--np`) | `1` | How many requests are served at once. See [Several requests at once](#several-requests-at-once). |
| `--api-key <key>` | — | Requires `Authorization: Bearer <key>`. Also read from the `AIRUN_API_KEY` variable. **Without this flag there is no authentication at all** — do not use `--host 0.0.0.0` without it. |
| `--served-model-name <name>` (or `--alias`) | the name in the file | Name shown in `/v1/models`. Without the option, it comes from the GGUF metadata; if absent, from the file name. |
| `--default-max-tokens <N>` | `512` | Limit for OpenAI requests that do not set one. The native `/completion` generates to the end, like llama-server. |
| `--reasoning-format <deepseek\|none>` | `deepseek` | Where chat responses' thinking (`<think>`) goes: in `reasoning_content` (llama-server's default) or in `content`. The request's `reasoning_format` takes priority. |
| `--repeat-penalty <f>` | `1.1` | Multiplicative penalty on recent tokens when the request does not ask for one. `1.0` turns it off. |
| `--repeat-last-n <N>` | `64` | How many recent tokens the penalties consider. |
| `--reasoning-budget <N>` | `-1` | Thinking tokens before `</think>` is forced; `-1` is no limit. The request's `reasoning_budget_tokens` takes priority. |
| `--reasoning-budget-message <text>` | — | Text written before the forced `</think>`. |
| `--slot-save-path <dir>` | — | Enables `POST /slots/0?action=save\|restore\|erase`, which save and load the conversation cache in that directory. Without it, 501. |
| `--embeddings` | off | Enables `/embedding` and `/v1/embeddings`. Without it, 501. |
| `--pooling <none\|mean\|cls\|last>` | from the GGUF, else `last` | How token states become an embedding. `none` returns one vector per token (native endpoint only). |

Plus every memory and allocation flag from the section above.

**About penalties:** the OpenAI API only has `presence_penalty` and
`frequency_penalty`. `repeat_penalty` (which keeps the model from repeating
itself) has no field there, so the server's applies; a client can send
`repeat_penalty` in the body to change it.

### Several requests at once

With `--parallel N`, up to N requests are served together, like
llama-server's slots:

- Each request gets a **slot** with its own cache. The cache's VRAM grows N
  times, so the automatic context gets smaller.
- The ones generating advance **together**: one pass through the model gives
  one token to each, reading the weights only once. On the 27B (5060 Ti +
  3060): 1 request alone 22.7 tok/s; 4 together add up to 50.3 tok/s (~13
  each).
- A new request goes to the slot that already has the start of the
  conversation cached, so a chat's next turn does not reprocess the history.
- **A request alone runs as without the flag:** with nobody generating or
  queued, it takes the same path as `--parallel 1` (MTP with `--draft`, greedy
  pick on the GPU, decode graphs, turn snapshot). When another arrives, it
  pauses at the token it is delivering and both continue together. On the 27B
  NVFP4 with `--draft 4` (5070 Ti + 5060): alone 38.3 tok/s with or without
  `--parallel 4` (before the change, 27.4 with the flag).
  `AIRUN_NO_PARALLEL_SOLO=1` turns it off.
- **Batched steps also use MTP.** Each slot has its own head cache, and one
  pass checks the drafts of all requests together. The text is the same each
  request would generate alone (temp 0). The more requests, the fewer drafts
  per request: checking many tokens at once costs more per token. It is 2
  drafts with 2 requests and 1 with 3 or 4; with more than 4, the batch runs
  without drafts. 27B NVFP4 (5070 Ti + 5060, `--draft 3`), summed tok/s:

  | requests | no MTP in the batch | MTP in the batch |
  |---|---|---|
  | 2 | 59 | 81.5 |
  | 3 | 82 | 98 |
  | 4 | 106 | 117 |
  | 4, `-c 262144` (KV in RAM) | 88.5 | 107 |

  `AIRUN_NO_ROWS_SPEC=1` turns it off; `AIRUN_ROWS_DRAFT=2,1,1` picks the
  drafts for 2, 3, 4... requests.
- Each slot keeps its own turn snapshot (hybrids), so two alternating
  conversations each reuse their own history.
- `qwen4exp` models do not accept N greater than 1.

### Prompt cache

The server reuses the prompt prefix already in the caches, so in a chat only
the last turn is processed on each message. The gain is large because prefill
is slow (see below): on a Qwen3.6-27B, the second turn of a ~880-token
conversation went from ~78 s to 1.5 s.

On **hybrid** models (Qwen3.5/3.8/Next) there is a limitation that comes from
the architecture: the delta-net state is a summary of everything it has read,
and a token cannot be "unread". So the cache is only reused when the new
prompt **continues exactly** the previous one; editing an old message
reprocesses everything. On dense models (Qwen3) it can go back to any point.

Each request prints a line on stderr saying how much was reused:

```
prompt 884 tok (869 cached, 15 new)  prefill 1285 ms (11.7 tok/s)  decode 1 tok in 116 ms (8.6 tok/s)
```

### Endpoints

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | No authentication. For health checks. |
| `/v1/models` | GET | Requires authentication. |
| `/v1/chat/completions` | POST | Supports `stream: true` (SSE). |
| `/v1/completions` | POST | Supports `stream: true` (SSE). |
| `/v1/stream/{id}?from=N` | GET | Replays a generation from byte N and continues live (see below). |
| `/v1/streams/lookup` | POST | `{"conversation_ids": [...]}`: says which of these generations exist and whether they have finished. |
| `/v1/stream/{id}` | DELETE | Stops the generation and deletes what was stored. |

**Stream replay:** a request with the `X-Conversation-Id: <id>` header keeps
generating even if the client drops (tab closed, network down). The client
comes back and rereads the text through `/v1/stream/<id>`. The server keeps
the last 4 MiB and deletes it 5 minutes after the generation ends. It is the
same feature as llama-server's.

**Images (with `--mmproj`):** send the image inside the request, in base64:
`image_url` with a data URL (OpenAI), `input_image` (Responses), an `image`
block (Anthropic) or `multimodal_data` (`/completion`). The server does not
download http URLs. The same image in later turns of a chat is not processed
again.

Accepted fields: `messages`, `prompt`, `max_tokens`, `max_completion_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `top_n_sigma`, `seed`,
`presence_penalty`, `frequency_penalty`, `repeat_penalty`, `logit_bias`,
`stream`, `stop`.

`logit_bias` is OpenAI's object — the key is the token id, the value how much
to add to its logit. `-inf` (or a very negative value) bans the token. Entries
that do not parse are ignored instead of failing the request.

`finish_reason` is `"stop"` when the model emitted an end token or a stop
sequence matched, and `"length"` when the token budget or the context ran
out.

A stop sequence that falls between two tokens **does not leak** to the
client: the server holds the doubtful piece until it knows whether it becomes
the stop sequence.

**Several clients:** with `--parallel 1` (the default) requests are served one
at a time, and the others wait in the queue. To serve several together, use
`--parallel N`.

```powershell
airun serve model.gguf --port 8080 --api-key my-key -c 8192
```

---

## `airun convert`

Converts a HuggingFace checkpoint (safetensors) into GGUF.

| Flag | Default | What it does |
|---|---|---|
| `<input>` | — | Directory with `config.json` and the `.safetensors`. Shards are joined automatically. |
| `-o`, `--output <path>` | — | Output `.gguf` file. Parent directories are created. |
| `--type <type>` | `F16` | Target type. See the type table below. |
| `--pure` | off | Applies `--type` to every matrix, with no per-tensor exceptions. |
| `--name <name>` | folder name | Name recorded in the file. |

By default (without `--pure`), precision is raised on sensitive tensors:
`token_embd` and `output` to Q6_K; `attn_v` and `ffn_down` one step above the
target. Norms always stay F32. It is the same reasoning as llama.cpp's `_M`
variants.

```powershell
airun convert model-folder -o out.gguf --type Q4_K
```

---

## `airun quantize`

Rewrites a GGUF at another precision.

| Flag | Default | What it does |
|---|---|---|
| `<input>` `<output>` | — | Source and destination files. |
| `--type <type>` | `Q4_K` | Target type. |
| `--pure` | off | No per-tensor exceptions. |
| `--dry-run` | off | Shows what would be written, without writing. |

It never raises precision: a tensor already at 4 bits gains nothing from
being rewritten at 8 — the file would only grow carrying the same error.
Tensors whose type does not change are copied byte for byte, with no lossy
round trip.

---

## Quantization types

| Type | Bits/weight | Read | Write | Note |
|---|---|---|---|---|
| `F32` | 32 | yes | yes | |
| `F16` | 16 | yes | yes | |
| `BF16` | 16 | yes | yes | Wider range, less mantissa |
| `Q8_0` | 8.5 | yes | yes | Nearly lossless |
| `Q6_K` | 6.56 | yes | yes | |
| `Q5_K` | 5.5 | yes | yes | |
| `Q5_0` / `Q5_1` | 5.5 / 6.0 | yes | yes | |
| `Q4_K` | 4.5 | yes | yes | Best overall balance |
| `Q4_0` / `Q4_1` | 4.5 / 5.0 | yes | yes | Legacy |
| `Q2_K` / `Q3_K` | 2.63 / 3.44 | yes | no | |
| `IQ4_NL` / `IQ4_XS` | 4.5 / 4.25 | yes | no | Non-linear codebook |
| `MXFP4` | 4.25 | yes | no | E8M0 scale |

`Q4_K_M` and `Q4_K_S` are accepted as synonyms of `Q4_K`: the suffix names a
per-tensor mixing policy, not a block format.

---

## `airun inspect`

| Flag | What it does |
|---|---|
| `--tensors` | Lists each tensor with shape, type and offset. |
| `--all-metadata` | Includes the huge tokenizer arrays. |
| `--json` | Machine-readable output. |

---

## `airun bench`

| Flag | Default | What it does |
|---|---|---|
| `--prefill <list>` | `128,512` | Prompt sizes to measure. |
| `--decode <N>` | `128` | Tokens generated in the decode measurement. |
| `--decode-at <list>` | `16` | Context depths at which to measure decode, e.g. `--decode-at 16,1024,3072`. The prompt is ingested first and only generation is timed. |
| `--runs <N>` | `3` | Measured runs per configuration. |
| `--warmup <N>` | `1` | Discarded runs (kernel JIT, clock ramp-up). |
| `--text-file <file>` | — | Builds the prompts from this text instead of a repeated token. MoE models use a built-in passage by default. |
| `--temp <T>` | `0` | Sampling temperature in decode; 0 is greedy. |
| `--json` | — | Machine-readable output. |

**On MoE with experts outside VRAM, measure with natural text and
`--temp 0.7`.** A repeated token always routes to the same experts, and greedy
falls into loops; both make the cache look perfect. The *expert hit* column
shows each test's hit rate.

It reports the **median**, not the mean: a single scheduling interruption
skews the mean of three runs too much.

**Decode is not a single rate.** Attention reads the whole cache on every
token, so the rate drops with context — on a Qwen3-1.7B, from 234 t/s with an
empty context to 128 t/s at 3072 tokens. The default (`--decode-at 16`)
measures only the empty case, which is the best case and does not describe a
long conversation. When comparing two versions, compare at the same depth.

---

## `airun perplexity`

Scores a known text and reports the mean loss, the perplexity and the top-1
hit rate. It answers "is this model *right*?", which text generation does not:
a decoder with a swapped layout still writes fluent English while answering
another question.

| Flag | Default | What it does |
|---|---|---|
| `-t`, `--text <TEXT>` | built-in passage | Text to score. |
| `-f`, `--file <FILE>` | — | Reads the text from a file. Conflicts with `--text`. |
| `-n`, `--limit <N>` | `0` (all) | Stops after N tokens. |
| `--worst <N>` | `0` | Lists the N worst-predicted tokens, worst first. |
| `--batched` | off | With `--chunk`: each chunk in a single pass, through the prefill kernels (like `llama-perplexity`), instead of token by token. Measures the quality cost of the prefill path (e.g. FP4 activations). |
| `--chunk <N>` | `0` | Scores like `llama-perplexity -c N`: chunks of N tokens, context reset in each, only the second half counts. Same text and N give comparable numbers between the two (normalize CRLF→LF first: llama.cpp reads `-f` in text mode on Windows). |
| `-c`, `--ctx-size <N>` | `4096` | Context size. |
| `--gpus`, `--tensor-split`, `--single-gpu` | | Same as `run`. |

Tokens are fed **one at a time**, which is the path decoding actually takes —
a wrongly updated recurrent state would be invisible in a single batched
pass.

Reference values on this machine, on the built-in passage:

| Model | Loss | Top-1 |
|---|---|---|
| Qwen3-8B Q4_K (dense, correct) | 2.20 | 50.9% |
| Qwen3-1.7B Q8_K_XL (dense, correct) | 2.43 | 47.2% |
| Qwen3.5-9B Q4_K_M (hybrid) | 4.40 | 20.1% |
| Qwen3.6-35B-A3B Q4_K_XL (hybrid + MoE) | 11.86 | 1.9% |

`ln(248320) = 12.4` is the value of a model guessing uniformly.

---

## `airun selftest`

| Flag | Default | What it does |
|---|---|---|
| `--device <ordinal>` | `0` | GPU to test. |
| `--verbose` | off | Shows every case, including the passing ones. |

Compares the GPU decoders, the matvec of each type, rmsnorm, swiglu, rope,
attention and the MoE expert kernels against the CPU reference. **Run it after
touching any kernel.**

---

## Environment variables

| Variable | What it does |
|---|---|
| `AIRUN_API_KEY` | Alternative to `--api-key`. |
| `AIRUN_CUDA_ARCHS` | CUDA architectures compiled (build). E.g. `120` during development, for fast builds. Default: `75,80,86,89,120`. |
| `AIRUN_NVCC_CCBIN` | Host compiler directory for nvcc, if automatic detection fails. |
| `AIRUN_SKIP_CUDA` | Builds without GPU kernels (file parsing only). |
| `CUDA_PATH` | Where to find the CUDA toolkit. |

### Hybrid debugging variables

They exist because several layout ambiguities produce tensors of the **same
shape**: no validation tells them apart, only the model's output. Each one
turns on the alternative interpretation; measure the difference with `airun
perplexity`, not by looking at the generated text. Every current choice was
measured again after the 2026-09-26 fix and is the best.

| Variable | Alternative interpretation it turns on |
|---|---|
| `AIRUN_ROPE_SCALING=none\|linear\|yarn` | Ignores the file's `rope.scaling.type`, like llama.cpp's `--rope-scaling`. For A/B. |
| `AIRUN_ROPE_MSCALE=<f>` | Replaces YaRN's magnitude factor (default `1 + 0.1 ln factor`). For A/B. |
| `AIRUN_EMBED_ON_GPU=1` | Keeps the embedding table in VRAM. By default it stays in pinned RAM (as in llama.cpp) when not tied to the output head: ~0.8 GiB less on a 250k vocabulary. |
| `AIRUN_NO_FP4=1` | NVFP4 prefill on the int8 MMQ (8-bit activations) instead of Blackwell's FP4 tensor cores (4-bit activations). Slower, slightly more exact. |
| `AIRUN_NO_Q8_WIDE=1` | NVFP4 decode and verify go back to the one-warp-per-four-rows kernels. |
| `AIRUN_NO_Q8_MULTI=1` | Batches of 2–8 tokens go back to the tiled kernel instead of the one that decodes each group once for all tokens. |
| `AIRUN_SPEC_TIMING=1` | Prints the mean verify, commit and draft time per speculative step. |
| `AIRUN_MATVEC_F32=1` | Every matvec on the exact f32 path (the reference one). Without it, decode is already exact; small batches use dp4a over Q8 activations and large ones int8 tensor cores (MMQ) — both validated in the selftest by a backward error bound and end to end by perplexity. Use it when "identical to the reference" matters more than speed. |
| `AIRUN_NO_WMMA=1` | Keeps large batches on the register kernels (measures the tensor-core path against them without a rebuild). |
| `AIRUN_NO_MMQ=1` | Batches of 64+ tokens leave the int8 tensor cores (MMQ) and go back to fp16 (WMMA). |
| `AIRUN_NO_MMQ_ASYNC=1` | Keeps MMQ, but the synchronous version (no `cp.async`). |
| `AIRUN_NO_FLASH=1` | Prefill uses the old attention kernel (one block per query) instead of flash attention. |
| `AIRUN_NO_GQA_DECODE=1` | Decode uses the per-query-head attention kernel instead of the one grouped by KV head. |
| `AIRUN_ABLATE=ssm,attn,ffn` | Zeroes these blocks' contribution to the residual. A correct block always scores better than no block. Sanity: `ffn` on the dense 8B takes the loss from 2.20 to 25.96 — if it does **not** move, the flag is not wired on that path. |
| `AIRUN_TRACE_LAYERS=1` | Prints rms/mean of each block and of the residual, layer by layer. |
| `AIRUN_TRACE_SSM=<layer>` | Prints the ranges of q, k, v, z, decay, beta, state and output of that delta-net layer. |
| `AIRUN_SSM_GATE_BEFORE_NORM=1` | Applies the gate before normalization (the old, wrong order). |
| `AIRUN_SSM_GATE_SIGMOID=1` | Sigmoid instead of SiLU in the gate. |
| `AIRUN_SSM_QKV_ORDER=<perm>` | Order of the three blocks in `attn_qkv`, e.g. `kqv`. Default `qkv`. |
| `AIRUN_SSM_QKV_GROUPED=1` | `attn_qkv` grouped by k-head instead of contiguous blocks. |
| `AIRUN_SSM_SWAP_AB=1` | Swaps `ssm_alpha` and `ssm_beta`. |
| `AIRUN_SSM_CONV_REVERSE=1` | Reverses the order of the convolution taps. |
| `AIRUN_SSM_NO_L2NORM=1` | Does not normalize q/k to unit norm. |
| `AIRUN_SSM_NO_QSCALE=1` | Does not scale q by `1/sqrt(head_k_dim)`. |
| `AIRUN_SSM_NO_CONV_SILU=1` | No SiLU after the convolution. |
| `AIRUN_SSM_A_IS_LOG=1` | Treats `ssm_a` as raw `A_log` even when it is already negative. |
| `AIRUN_ATTN_GATE_BLOCKED=1` | `attn_q` as `[all q][all gates]` instead of interleaved per head. |
| `AIRUN_ATTN_GATE_FIRST=1` | The gate is the **first** half of each head's slice. |
| `AIRUN_ATTN_NO_GATE=1` | Does not apply the sigmoid gate on the attention output. |
| `AIRUN_ROPE_FULL=1` | Rotates the whole `head_dim` instead of `rope.dimension_count`. |
| `AIRUN_ROPE_INTERLEAVED=1` | GPT-J pairing `(2i, 2i+1)` instead of NeoX `(i, i + dim/2)`. |
| `AIRUN_ROPE_OFF=1` | No RoPE at all. Its cost measures how much the model uses position: on the dense 8B +2.75 nats, on the hybrid 9B +0.04. |
| `AIRUN_ATTN_WINDOW=<n>` | Forces a sliding window. With `1` each query sees only itself; the cost measures how much attention takes from the history. |

### Performance variables

Each one turns off (or forces) an optimization, to measure its gain without
recompiling. None changes the result beyond rounding — the ones that change
the order of sums are covered by the selftest and perplexity. Always measure
alternating the runs (see `benchmarks/2026-09-26/scoreboard.ps1`): on this
machine, another program drawing on the GPU delays kernels by ~300 µs every
4 ms, and two separate sessions vary by ~15%.

| Variable | Effect |
|---|---|
| `AIRUN_NO_MMQ_STREAMK=1` | Async MMQ goes back to the whole-tile grid (with split-K where it fills the GPU poorly). With stream-K the last tiles are split into equal K slices across all SMs: 128-token prefill on Qwen3-4B from 2767 to 3269 t/s. |
| `AIRUN_NO_FP4_GLU=1` | In an NVFP4 FFN on RTX 50, gate and up go back to two FP4 matmuls, with SwiGLU in the down's quantization. By default the three are one launch (SwiGLU in the epilogue): +2% prefill on the 27B. |
| `AIRUN_NO_DELTA_CHUNK=1` | Delta-net layer prefill goes back to the token-by-token recurrence. By default, batches of 64+ tokens use the chunked form on the tensor cores (bf16, fp32 accumulation): +8.5% prefill on the 27B, +7% on the 35B, same perplexity. |
| `AIRUN_NO_MMQ_SPLIT=1` | No split-K in MMQ (where stream-K does not run: synchronous kernel, Q6_K). |
| `AIRUN_MMQ_TC=64` or `=128` | Forces MMQ's token tile width. Default: 128 from 128 tokens. |
| `AIRUN_NO_MMQ_GLU=1` | Prefill gate, up and SwiGLU in three launches instead of one. |
| `AIRUN_ATTN_PV_F32=1` | Prefill sums P·V in fp32 also with long context (from 4096 keys, the default is fp16 per key tile, summed in fp32). |
| `AIRUN_NO_SPLIT_WAVES=1` | Decode uses every cache split the position allows, without rounding to whole waves of blocks, and keeps 64 keys per split at short contexts (see `attention_active_splits`; 35B-A3B on an RTX 3060: 69–72 against 74–75 tok/s). |
| `AIRUN_NO_FLASH2=1` | Prefill uses the old WMMA flash attention instead of the `mma.sync` one (head_dim 64/128). |
| `AIRUN_NO_SPLIT_ATTN=1` | Decode uses the one-block-per-head attention kernel, without splitting the context. |
| `AIRUN_NO_FUSED_PROLOGUE=1` | q/k norm, RoPE and cache write in separate launches. |
| `AIRUN_NO_FUSED_NORMS=1` | Residual add and the next RMSNorm in two launches. |
| `AIRUN_NO_GRAPH=1` | Decode launches the kernels one by one instead of replaying a captured CUDA graph. |
| `AIRUN_NO_ROWS_GRAPH=1` | The `serve --parallel` batched step (`decode_rows`) launches the kernels one by one. By default it is one graph per slot list: 4 requests on the 27B from 59 to 70 tok/s summed. |
| `AIRUN_NO_ROWS_BATCH=1` | In the batched step, attention, delta-net and convolution run one kernel per row instead of one for all (same result; 4 requests on the 4B: 300 against 370 tok/s). |
| `AIRUN_NO_TALLK_TOKENS=1` | Narrow matrices (fewer than 512 rows, like the delta-net gates) with 2 to 16 tokens go back to the tiled kernel. By default they stay on the one-block-per-row-and-token one: MTP verify on the 27B +7%. |
| `AIRUN_NO_HALF_MATVEC=1` | BF16/F16 matrices with 1 to 8 tokens (the output head of several GGUFs) go back to the per-group kernels. |
| `AIRUN_SCHED_TIMING=1` | One line per `serve --parallel` batched step: forward, sampling, delivery. Disturbs speed measurement. |
| `AIRUN_HIDE_GPUS=<list>` | Like `--hide-gpu`, for every command: GPUs airun never touches. Does not combine with `CUDA_VISIBLE_DEVICES`. |
| `AIRUN_NO_KV_RAM=1` | When the requested context does not fit, takes weights out of VRAM instead of the KV cache (the old behavior). |
| `AIRUN_NO_SPLIT_VERIFY=1` | MTP verify attention (several tokens over one cache) goes back to the prefill kernel, which reads the whole context per block: 27B at 25k tokens, 52.8 → 34.2 tok/s. |
| `AIRUN_RAM_ON_GPU=1` | Layers that stay in RAM are always copied to VRAM before running, instead of multiplied on the CPU during generation. |
| `AIRUN_CPU_THREADS=<n>` | Threads for CPU matmuls (default: one per logical core). On the 13700's DDR5, 8 already saturate the bandwidth. |
| `AIRUN_CPU_SPIN_US=<n>` | How long CPU worker threads spin waiting for the next matmul (default 200 µs). Much longer starves the thread driving the GPU (5 ms: 3.0 → 0.9 tok/s). |
| `AIRUN_CPU_STATS=1` | Every 1000 CPU matmuls, prints the time inside them and the bandwidth. |
| `AIRUN_DISK_AHEAD=<n>` | With `--disk-offload`, how many disk layers are read ahead while the current one runs (default 2). Each takes its share of the file cache. |
| `AIRUN_DISK_THREADS=<n>` | Threads reading those layers ahead (default 4). |
| `AIRUN_NO_LAYERED=1` | With layers on disk, the prompt goes back to passing chunk by chunk through all layers (reading the disk once per chunk). |
| `AIRUN_LAYERED_TOKENS=<n>` | Prompt tokens that pass together through each disk layer (default 8192). Each token costs 8 bytes × model width of temporary memory. |
| `AIRUN_CPU_MAX_TOKENS=<n>` | Passes of up to n tokens multiply the RAM layers on the CPU; larger ones copy to the GPU (default 64). |
| `AIRUN_NO_ROWS_SPEC=1` | With `serve --parallel`, batched steps run without MTP drafts (only the lone request drafts). |
| `AIRUN_ROWS_DRAFT=<k2,k3,...>` | Drafts per request in a batched step of 2, 3... requests (default `2,1,1`; 0 = no draft). |
| `AIRUN_NO_SPLIT_FEW=1` | 2 to 16 tokens on a type without a multi-token kernel (IQ4_XS, for example) go to the tiled kernel instead of one call per token. |
| `AIRUN_NO_PARALLEL_SOLO=1` | With `serve --parallel`, a lone request also goes through the batched step (no MTP, logits copied to the host). |
| `AIRUN_NO_GPU_ARGMAX=1` | Greedy decoding copies the logits to the host and picks there. |
| `AIRUN_NO_DECODE_PIPELINE=1` | Greedy decoding waits for each token before queuing the next step. With the pipeline step i+1 reads step i's token straight from the GPU; the generated text is identical. Does not run on hybrids (the recurrent state would absorb the speculative step). |
| `AIRUN_NO_KV_LOAN=1` | MoE experts in RAM no longer borrow the KV cache's still-unused rows (see [MoE: what is automatic](#moe-what-is-automatic)). |
| `AIRUN_MOE_OVERLAP=0` | MoE decode computes resident and copied experts in one pass on one stream, instead of the copy running beside the resident ones. Slower; makes the sum order independent of where experts live (use it to compare greedy text). |
| `AIRUN_MOE_F32=1` | MoE decode experts on the float kernels instead of int8 (dp4a). |
| `AIRUN_STAGE_BLOCKS=<n>` | Blocks per expert in the RAM→VRAM expert copy (default 4). |
| `AIRUN_EXPERT_PREFETCH=1` | Predicts a layer's experts as it starts and copies the ones in RAM while attention runs. Off by default: it lost on an RTX 3060 (67–69 against 70 tok/s). |
