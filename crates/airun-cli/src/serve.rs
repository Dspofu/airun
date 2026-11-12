//! `airun serve` -- an HTTP API in the shape of llama.cpp's `llama-server`.
//!
//! The routes, request fields and response bodies follow llama-server's, so a
//! client written for it (its web UI, the OpenAI SDKs, agents that call tools)
//! works unchanged: llama.cpp's native `/completion`, the OpenAI
//! `/v1/completions` and `/v1/chat/completions` with llama.cpp's additions
//! (`timings`, `reasoning_content`, `tool_calls`, cached-token usage),
//! the OpenAI Responses API (`/v1/responses`), the Anthropic Messages API
//! (`/v1/messages`), `/infill`, token counting, `/tokenize`, `/detokenize`,
//! `/apply-template`, `/props`, `/models`, `/slots` and `/health`; the
//! endpoints of features airun lacks (embeddings, reranking, audio, slot
//! files) answer 501 as llama-server does without them.
//! `tests/api_conformance.py` compares the two servers' responses key by key.
//!
//! Sampling takes llama.cpp's whole chain (DRY, XTC, dynamic temperature,
//! mirostat), constrained by a GBNF `grammar`, a `json_schema` or a
//! `response_format`, and reports per-token probabilities (`n_probs`,
//! `logprobs`).
//!
//! Chat prompts come from the GGUF's own Jinja template (see chat.rs), as
//! llama-server's do with `--jinja`, and replies are split into reasoning,
//! content and tool calls the same way.
//!
//! One model is held in memory and requests are serialized against it: the KV
//! cache is a single sequence (one slot), so two concurrent generations would
//! corrupt each other. Requests queue on a mutex rather than failing.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use airun_core::grammar::{Grammar, json_object_grammar, json_schema_grammar};
use airun_core::{ProbsSink, Runtime, Sampler, SamplingParams, TokenProbs, loader};
use anyhow::{Context, Result};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Args;
use serde_json::{Map, Value, json};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::chat::{ChatTemplate, Delta, DeltaSplitter, Reply, ToolCall, parse_reply, parse_reply_without_tools};
use crate::run::RunArgs;

#[derive(Args)]
pub struct ServeArgs {
    /// Path to a .gguf model file.
    pub model: PathBuf,

    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Require this key as a Bearer token. Unset means no authentication.
    #[arg(long, env = "AIRUN_API_KEY")]
    pub api_key: Option<String>,

    /// Name reported by /v1/models and echoed in responses. Default: the
    /// model's name in the file's metadata (general.name), else the file name.
    #[arg(long, alias = "alias")]
    pub served_model_name: Option<String>,

    /// Context length in tokens. Default: the largest the GPUs hold with
    /// every weight in VRAM (at least 4096), on the GPUs the model needs.
    #[arg(short = 'c', long)]
    pub ctx_size: Option<usize>,

    /// Resident experts per layer; stream cache misses from the GGUF without re-quantizing.
    #[arg(long)]
    pub expert_cache: Option<usize>,

    /// KV cache format: f16, q8_0 or q4_0.
    #[arg(long, default_value = "f16", value_parser = crate::run::parse_kv_type)]
    pub kv_type: airun_cuda::ops::KvType,

    /// Fail instead of silently shrinking the requested context or batch.
    #[arg(long)]
    pub no_fallback: bool,

    /// Tokens per forward pass during prompt ingestion.
    #[arg(short = 'b', long, default_value_t = 512)]
    pub batch_size: usize,

    /// Speculative decoding with the model's own MTP head: tokens drafted per
    /// step. Off unless given; the load says when the model has a head and
    /// how many drafts suit it. The drafts are verified by the model, and
    /// sampling keeps its distribution, but the verify runs in a small
    /// batch: on a near-tie the text can differ from one-at-a-time decoding.
    /// Files without a usable MTP head ignore it.
    #[arg(long)]
    pub draft: Option<usize>,

    /// The MTP head in a file of its own (mtp-*.gguf). Default: the model's
    /// own head, or an mtp-*.gguf of the same model in the model's folder.
    #[arg(long, value_name = "FILE")]
    pub mtp: Option<PathBuf>,

    /// The vision encoder (mmproj-*.gguf): chat requests may then carry
    /// images (OpenAI image_url / input_image data URLs, Anthropic base64
    /// image blocks, /completion's multimodal_data).
    #[arg(long, value_name = "FILE")]
    pub mmproj: Option<PathBuf>,

    /// Requests served at once (llama-server's --parallel): each gets a slot
    /// with its own KV cache and state at the full context, and the running
    /// ones advance together, one forward pass per token for all of them.
    /// VRAM for the caches grows with N. A request running alone decodes as
    /// with --parallel 1 (MTP with --draft included); batched steps do not draft.
    #[arg(long, visible_alias = "np", default_value_t = 1)]
    pub parallel: usize,

    /// Most tokens one image may take (a 32 x 32 pixel area each).
    #[arg(long, default_value_t = 1024)]
    pub image_max_tokens: usize,

    /// CUDA ordinals to use.
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<i32>,

    /// Layer split proportions per GPU.
    #[arg(long, value_delimiter = ',')]
    pub tensor_split: Vec<f32>,

    /// Keep the whole model on one GPU, by ordinal.
    #[arg(long)]
    pub single_gpu: Option<i32>,

    #[command(flatten)]
    pub alloc: crate::run::AllocArgs,

    /// VRAM left free on each GPU, in MiB.
    #[arg(long, default_value_t = 384)]
    pub vram_reserve: u64,

    /// Default maximum tokens when an OpenAI request does not specify one
    /// (llama.cpp's native /completion generates until the end or the
    /// context, as llama-server does).
    #[arg(long, default_value_t = 512)]
    pub default_max_tokens: usize,

    /// Penalty on recently generated tokens when the request does not ask for
    /// one. 1.0 disables it, at the cost of letting a model loop.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// How many recent tokens the repetition penalty considers.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// How chat replies return their thinking: `deepseek` (in
    /// `reasoning_content`, llama-server's default) or `none` (left in the
    /// content). A request's own `reasoning_format` wins.
    #[arg(long, default_value = "deepseek")]
    pub reasoning_format: String,

    /// Tokens a chat reply may spend thinking before `</think>` is forced;
    /// -1 is unlimited. A request's `reasoning_budget_tokens` (or
    /// `thinking_budget_tokens`) wins.
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    pub reasoning_budget: i64,

    /// Text written before the forced `</think>` when the budget runs out.
    #[arg(long, default_value = "")]
    pub reasoning_budget_message: String,

    /// Thinking on, off, or as the template decides (auto), as
    /// llama-server's --reasoning. A request's `enable_thinking` wins.
    #[arg(long, value_name = "on|off|auto", default_value = "auto", value_parser = crate::reasoning::parse_switch_arg)]
    pub reasoning: String,

    /// Default reasoning level: none, low (1024 thinking tokens), medium
    /// (4096), high (16384), xhigh (32768) or max (unlimited). A request's
    /// `reasoning_effort` (or Responses' `reasoning.effort`) wins; an
    /// explicit budget wins over both.
    #[arg(long, value_name = "LEVEL", value_parser = crate::reasoning::parse_arg)]
    pub reasoning_effort: Option<String>,

    /// Directory where `POST /slots/0?action=save|restore` keeps the cache
    /// of a conversation. Unset: slot actions answer 501, as in llama-server.
    #[arg(long)]
    pub slot_save_path: Option<PathBuf>,

    /// Serve `/embedding` and `/v1/embeddings` from the model's final hidden
    /// state. Without it they answer 501, as in llama-server.
    #[arg(long, alias = "embedding")]
    pub embeddings: bool,

    /// How a text's token states become one embedding: none (one per
    /// token, native endpoint only), mean, cls (the first token) or last.
    /// Default: the GGUF's pooling_type, or last for a generative model,
    /// whose last token is the one that has read the whole text.
    #[arg(long, value_parser = ["none", "mean", "cls", "last"])]
    pub pooling: Option<String>,
}

/// What the last request asked for and how far it got, for `/slots`.
#[derive(Default)]
struct SlotInfo {
    id_task: u64,
    n_prompt_tokens: usize,
    n_prompt_tokens_processed: usize,
    n_prompt_tokens_cache: usize,
    n_decoded: usize,
    params: Value,
}

struct AppState {
    runtime: Arc<Mutex<Runtime>>,
    /// The runtime's tokenizer, usable while a generation holds the runtime
    /// (with --parallel, a request has to reach the scheduler to be batched).
    tokenizer: Arc<airun_core::tokenizer::Tokenizer>,
    /// With --parallel above 1: the scheduler that runs generations.
    sched: Option<std::sync::Mutex<std::sync::mpsc::Sender<crate::sched::Job>>>,
    /// The vision encoder, with --mmproj.
    vision: Option<std::sync::Mutex<airun_core::vision::VisionModel>>,
    /// Images already encoded, by a hash of their file bytes: a chat sends
    /// the same image every turn, and its tokens are found again without
    /// running the encoder (while the runtime still has them registered).
    image_memo: std::sync::Mutex<Vec<(u64, Vec<u32>)>>,
    model_name: String,
    api_key: Option<String>,
    default_max_tokens: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    reasoning_format: String,
    /// airun's /props fields (its web UI reads them) merged with llama-server's.
    props: Value,
    /// /v1/models' body.
    models: Value,
    template: Option<ChatTemplate>,
    /// How the GGUF's chat template handles reasoning, for the fallback prompt
    /// when the template does not render.
    thinking: Thinking,
    ctx: usize,
    processing: AtomicBool,
    tasks: AtomicU64,
    slot: std::sync::Mutex<SlotInfo>,
    reasoning_budget: Option<usize>,
    reasoning_budget_message: String,
    /// --reasoning and --reasoning-effort.
    reasoning_default: Option<bool>,
    reasoning_effort: Option<String>,
    /// The template reads `reasoning_effort` itself.
    template_effort: bool,
    slot_save_path: Option<PathBuf>,
    embeddings: bool,
    pooling: Pooling,
    /// The token sequences of `<think>` and `</think>`, if the template thinks.
    think_tags: Option<(Vec<u32>, Vec<u32>)>,
    /// Chat completions in flight, by id: whether they accept `/control`, and
    /// the flag that ends their reasoning.
    active: std::sync::Mutex<std::collections::HashMap<String, (bool, Arc<AtomicBool>)>>,
    /// Each token's bytes as a client sees them (special tokens empty), for
    /// grammars, probabilities and DRY's sequence breakers.
    pieces: Arc<Vec<Vec<u8>>>,
    eog: Arc<Vec<u32>>,
    /// The tokens llama.cpp's default DRY sequence breakers mark.
    default_breakers: Arc<Vec<bool>>,
    /// Fill-in-the-middle markers, if the vocabulary has them.
    fim: Option<Fim>,
    /// Every visible GPU, for /hardware: (ordinal, name, PCI bus id).
    hw_devices: Vec<(i32, String, String)>,
    /// The last /hardware reading and when it was taken. A reading costs
    /// ~50 ms per GPU (NVML samples PCIe traffic), so several open pages
    /// share one.
    hw_cache: std::sync::Mutex<Option<(std::time::Instant, Value)>>,
}

#[derive(Clone, Copy)]
struct Fim {
    prefix: u32,
    suffix: u32,
    middle: u32,
    /// Qwen's repository-level context markers.
    repo: Option<u32>,
    file: Option<u32>,
}

/// llama.cpp's default DRY sequence breakers.
const DRY_BREAKERS: [&str; 4] = ["
", ":", "\"", "*"];

/// Which tokens contain one of `breakers`.
fn breaker_mask(pieces: &[Vec<u8>], breakers: &[String]) -> Vec<bool> {
    pieces.iter().map(|p| {
        let text = String::from_utf8_lossy(p);
        breakers.iter().any(|b| !b.is_empty() && text.contains(b.as_str()))
    }).collect()
}

/// Reasoning support read from the chat template, used when the prompt is
/// built without it.
#[derive(Clone, Copy, Default)]
pub(crate) struct Thinking {
    /// The template knows `<think>` blocks, so `enable_thinking` means something.
    pub(crate) supported: bool,
    /// The generation prompt already opens the block (Qwen3.5 and later), so the
    /// model never emits `<think>` itself.
    prompt_opens_block: bool,
}

impl Thinking {
    pub(crate) fn from_template(template: &str) -> Self {
        Self {
            supported: template.contains("<think>"),
            prompt_opens_block: template.contains("'<think>\\n'"),
        }
    }

    /// Text appended after `<|im_start|>assistant\n`.
    pub(crate) fn generation_prefix(self, enabled: bool) -> &'static str {
        match (self.supported, enabled) {
            (false, _) => "",
            (true, false) => "<think>\n\n</think>\n\n",
            (true, true) if self.prompt_opens_block => "<think>\n",
            (true, true) => "",
        }
    }
}

/// The answer part of an assistant turn, and the thinking before it.
fn strip_reasoning(text: &str) -> (&str, Option<&str>) {
    match text.rfind("</think>") {
        Some(end) => {
            let before = text[..end].trim_start();
            let before = before.strip_prefix("<think>").unwrap_or(before);
            (text[end + "</think>".len()..].trim_start(), Some(before.trim()))
        }
        // A reply stopped mid-thought has no answer at all.
        None if text.trim_start().starts_with("<think>") => ("", None),
        None => (text, None),
    }
}

/// One line per request, so a slow first token can be attributed.
fn log_request(s: &airun_core::GenerationStats) {
    eprintln!(
        "  prompt {} tok ({} cached, {} new)  prefill {:.0} ms ({:.1} tok/s)  \
         decode {} tok in {:.0} ms ({:.1} tok/s)",
        s.prompt_tokens, s.cached_tokens, s.processed_tokens(), s.prefill_ms, s.prefill_tps(),
        s.generated_tokens, s.decode_ms, s.decode_tps(),
    );
}

/// `logit_bias` in either shape: OpenAI's object keyed by token id, or
/// llama.cpp's `[[id, bias], ...]` (a bias of `false` bans the token).
/// Unparseable entries are dropped rather than failing the request.
fn logit_bias_from(v: Option<&Value>) -> Vec<(u32, f32)> {
    match v {
        Some(Value::Object(map)) => map.iter()
            .filter_map(|(k, v)| Some((k.parse::<u32>().ok()?, v.as_f64()? as f32)))
            .collect(),
        Some(Value::Array(a)) => a.iter().filter_map(|e| {
            let pair = e.as_array()?;
            let id = pair.first()?.as_u64()? as u32;
            let bias = match pair.get(1)? {
                Value::Bool(false) => f32::NEG_INFINITY,
                b => b.as_f64()? as f32,
            };
            Some((id, bias))
        }).collect(),
        _ => Vec::new(),
    }
}

/// How many trailing bytes of `text` might still become a stop sequence.
///
/// A stop string split across two tokens would otherwise reach the client
/// before it is recognised, and an SSE chunk cannot be unsent.
fn ambiguous_tail(text: &str, stops: &[String]) -> usize {
    let mut hold = 0;
    for stop in stops {
        // Proper prefixes only: a whole match is a stop, not an ambiguity.
        for (len, _) in stop.char_indices().skip(1) {
            if text.ends_with(&stop[..len]) {
                hold = hold.max(len);
            }
        }
    }
    hold
}

/// Where the earliest stop sequence begins in `text`, and which one.
fn stop_at<'a>(text: &str, stops: &'a [String]) -> Option<(usize, &'a str)> {
    stops.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str()).map(|i| (i, s.as_str())))
        .min_by_key(|x| x.0)
}

/// llama.cpp's stop_type for a run that ended this way.
fn stop_type(stop: airun_core::StopReason, word: bool) -> &'static str {
    match stop {
        _ if word => "word",
        airun_core::StopReason::EndOfGeneration => "eos",
        airun_core::StopReason::MaxTokens | airun_core::StopReason::ContextFull => "limit",
        airun_core::StopReason::Cancelled => "none",
    }
}

/// OpenAI's finish reason for a llama.cpp stop_type.
fn finish_reason(stop_type: &str, tool_calls: bool) -> &'static str {
    match stop_type {
        "eos" | "word" if tool_calls => "tool_calls",
        "eos" | "word" => "stop",
        // A client that hung up reads nothing; "stop" is the honest default.
        "none" => "stop",
        _ => "length",
    }
}

fn timings(s: &airun_core::GenerationStats) -> Value {
    let pn = s.processed_tokens();
    let per = |ms: f64, n: usize| if n > 0 { ms / n as f64 } else { 0.0 };
    let rate = |ms: f64, n: usize| if ms > 0.0 { n as f64 * 1000.0 / ms } else { 0.0 };
    let mut t = json!({
        "cache_n": s.cached_tokens,
        "prompt_n": pn,
        "prompt_ms": s.prefill_ms,
        "prompt_per_token_ms": per(s.prefill_ms, pn),
        "prompt_per_second": rate(s.prefill_ms, pn),
        "predicted_n": s.generated_tokens,
        "predicted_ms": s.decode_ms,
        "predicted_per_token_ms": per(s.decode_ms, s.generated_tokens),
        "predicted_per_second": rate(s.decode_ms, s.generated_tokens),
    });
    if s.drafted > 0 {
        t["draft_n"] = json!(s.drafted);
        t["draft_n_accepted"] = json!(s.accepted);
    }
    t
}

fn usage(s: &airun_core::GenerationStats) -> Value {
    json!({
        "completion_tokens": s.generated_tokens,
        "prompt_tokens": s.prompt_tokens,
        "total_tokens": s.prompt_tokens + s.generated_tokens,
        "prompt_tokens_details": { "cached_tokens": s.cached_tokens },
    })
}

const FINGERPRINT: &str = concat!("airun-", env!("CARGO_PKG_VERSION"));

/// A completion id in llama-server's form: a prefix and 32 random letters.
fn completion_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut x = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64).unwrap_or(0) ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let s: String = (0..32).map(|_| {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        ALPHA[(x % ALPHA.len() as u64) as usize] as char
    }).collect();
    format!("{prefix}{s}")
}

pub fn run(args: ServeArgs) -> Result<()> {
    let mut run_args = RunArgs::bench_defaults(args.model.clone());
    run_args.ctx_size = args.ctx_size;
    run_args.expert_cache = args.expert_cache;
    run_args.kv_type = args.kv_type;
    run_args.no_fallback = args.no_fallback;
    run_args.batch_size = args.batch_size;
    run_args.gpus = args.gpus.clone();
    run_args.tensor_split = args.tensor_split.clone();
    run_args.single_gpu = args.single_gpu;
    run_args.alloc = args.alloc.clone();
    run_args.vram_reserve = args.vram_reserve;
    run_args.draft = args.draft;
    run_args.mtp = args.mtp.clone();
    run_args.mmproj = args.mmproj.clone();
    run_args.image_max_tokens = args.image_max_tokens;

    eprintln!("loading {}", args.model.display());
    let mut load_opts = run_args.load_options();
    load_opts.slots = args.parallel.max(1);
    if let Some(mm) = &args.mmproj {
        load_opts.reserve_bytes += crate::media::vision_reserve(mm, args.image_max_tokens)?;
    }
    let mut bar = crate::progress::LoadBar::new(&args.model);
    let (runtime, report) = loader::load(&args.model, &load_opts, |stage, d, t| bar.update(stage, d, t))
        .with_context(|| format!("failed to load {}", args.model.display()))?;
    bar.finish();
    eprintln!("{}", report.config.summary());
    eprint!("{}", report.placement.report());
    let vision = match &args.mmproj {
        Some(mm) => {
            let v = crate::media::load_vision(mm, &runtime, args.image_max_tokens)?;
            eprintln!("vision encoder: {} on GPU {} (up to {} tokens per image)", mm.display(), v.device,
                args.image_max_tokens);
            Some(std::sync::Mutex::new(v))
        }
        None => None,
    };

    // --alias, else the name the file's metadata gives, else the file name.
    let model_name = args.served_model_name.clone()
        .or_else(|| (report.config.name != "unnamed").then(|| report.config.name.clone()))
        .unwrap_or_else(|| args.model.file_stem().and_then(|s| s.to_str()).unwrap_or("airun").to_string());
    let ctx = runtime.ctx();
    let cfg = &report.config;

    let gguf = airun_gguf::GgufFile::open(&args.model).ok();
    let meta_str = |k: &str| gguf.as_ref().and_then(|g| g.get(k)).and_then(|v| v.as_str()).map(str::to_owned);
    let template_src = meta_str("tokenizer.chat_template").unwrap_or_default();
    let token_text = |id: Option<u32>| id.and_then(|i| runtime.tokenizer.token_text(i)).unwrap_or("").to_string();
    let bos = token_text(runtime.tokenizer.bos);
    let eos = token_text(runtime.tokenizer.eos);
    let thinking = Thinking::from_template(&template_src);
    let think_tags = thinking.supported.then(|| {
        (runtime.tokenizer.encode("<think>", false), runtime.tokenizer.encode("</think>", false))
    }).filter(|(a, b)| !a.is_empty() && !b.is_empty());
    let template = if template_src.is_empty() {
        None
    } else {
        match ChatTemplate::new(template_src.clone(), bos.clone(), eos.clone()) {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("warning: {e}; chat prompts are built as plain ChatML instead");
                None
            }
        }
    };
    let pooling = match args.pooling.as_deref() {
        Some(p) => Pooling::from_name(p),
        None => {
            let arch = meta_str("general.architecture").unwrap_or_default();
            match gguf.as_ref().and_then(|g| g.get(&format!("{arch}.pooling_type"))).and_then(|v| v.as_u64()) {
                Some(0) => Pooling::None,
                Some(1) => Pooling::Mean,
                Some(2) => Pooling::Cls,
                _ => Pooling::Last,
            }
        }
    };
    let n_params: u64 = gguf.as_ref().map_or(0, |g| g.tensors().iter().map(|t| t.n_elements()).sum());
    let size = gguf.as_ref().map_or(0, |g| g.file_len());
    let ftype = gguf.as_ref().and_then(|g| g.type_histogram().into_iter().max_by_key(|(_, (_, b))| *b))
        .map(|(t, _)| t).unwrap_or_default();
    let model_path = args.model.display().to_string();

    // What the settings panel shows about the model and where it runs: the
    // file's architecture, how this server started it, and the memory plan.
    let placement = &report.placement;
    let range = |v: &[usize]| match (v.first(), v.last()) {
        (Some(a), Some(b)) => json!({ "first": a, "last": b, "count": v.len() }),
        _ => Value::Null,
    };
    let details = json!({
        "file": model_path,
        "file_bytes": size,
        "params": n_params,
        "ftype": ftype,
        "weight_bytes": report.weight_bytes,
        "load_ms": report.load_ms,
        "train_ctx": cfg.train_ctx,
        "ctx": ctx,
        "batch": placement.batch,
        "parallel": args.parallel.max(1),
        "kv_type": cfg.kv_type.name(),
        "layers": cfg.n_run_layers(),
        "attention_layers": cfg.n_attention_layers(),
        "linear_layers": cfg.ssm.as_ref().map(|_| cfg.n_run_layers() - cfg.n_attention_layers()),
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "n_head_kv": cfg.n_head_kv,
        "head_dim": cfg.head_dim,
        "ffn": cfg.ffn_len,
        "vocab": cfg.vocab_size,
        "rope_base": cfg.rope.base,
        "rope_scale": (cfg.rope.freq_scale != 1.0).then(|| 1.0 / cfg.rope.freq_scale),
        "sliding_window": (cfg.sliding_window > 0).then_some(cfg.sliding_window),
        "tied_embeddings": cfg.tied_embeddings,
        "moe": cfg.moe.as_ref().map(|m| json!({
            "experts": m.n_experts, "active": m.n_experts_used, "expert_ffn": m.expert_ff,
            "shared_ffn": (m.shared_expert_ff > 0).then_some(m.shared_expert_ff),
        })),
        "mtp": {
            "head": runtime.model.mtp.is_some(),
            "draft": if runtime.model.mtp.is_some() { runtime.draft() } else { 0 },
        },
        "gpus": placement.shards.iter().map(|s| {
            let vram: Vec<usize> = s.layers.iter().copied().filter(|l| !s.ram_layers.contains(l)).collect();
            json!({
                "device": s.device, "name": s.device_name,
                "layers": range(&vram), "ram_layers": range(&s.ram_layers),
                "embedding": s.hosts_embedding, "output": s.hosts_output,
                "weight_bytes": s.weight_bytes, "kv_bytes": s.kv_bytes, "scratch_bytes": s.scratch_bytes,
                "budget_bytes": s.budget_bytes, "ram_bytes": s.ram_bytes,
            })
        }).collect::<Vec<_>>(),
        "notes": placement.notes.iter().chain(report.fallbacks.iter()).collect::<Vec<_>>(),
    });

    let pieces: Arc<Vec<Vec<u8>>> = Arc::new((0..runtime.tokenizer.vocab_size() as u32)
        .map(|id| runtime.tokenizer.piece_bytes(id)).collect());
    let eog: Arc<Vec<u32>> = Arc::new({
        let mut e = runtime.tokenizer.eog.clone();
        e.extend(runtime.tokenizer.eos);
        e.sort_unstable();
        e.dedup();
        e
    });
    let default_breakers = Arc::new(breaker_mask(&pieces, &DRY_BREAKERS.map(str::to_string)));
    let fim = {
        let id = |t: &str| runtime.tokenizer.id_of(t);
        match (id("<|fim_prefix|>"), id("<|fim_suffix|>"), id("<|fim_middle|>")) {
            (Some(prefix), Some(suffix), Some(middle)) => Some(Fim {
                prefix, suffix, middle, repo: id("<|repo_name|>"), file: id("<|file_sep|>"),
            }),
            _ => None,
        }
    };

    let defaults = GenSpec {
        tokens: Vec::new(),
        params: SamplingParams {
            temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05,
            repeat_penalty: args.repeat_penalty, repeat_last_n: args.repeat_last_n,
            dry_penalty_last_n: args.repeat_last_n,
            ..SamplingParams::greedy()
        },
        max_tokens: None,
        stops: Vec::new(),
        stream: false,
        reasoning_format: args.reasoning_format.clone(),
        ignore_eos: false,
        grammar: None,
        n_probs: None,
        dry_breakers: DRY_BREAKERS.map(str::to_string).to_vec(),
        reasoning: None,
        progress: None,
    };
    let props = json!({
        // llama-server's
        "default_generation_settings": {
            "params": generation_settings(&defaults),
            "n_ctx": ctx,
        },
        "total_slots": 1,
        "model_alias": model_name,
        "model_ftype": ftype,
        "model_path": model_path,
        "modalities": { "vision": vision.is_some(), "video": false, "audio": false },
        "endpoint_slots": true,
        "endpoint_props": false,
        "endpoint_metrics": false,
        "ui": true,
        "ui_settings": {},
        "chat_template": template_src,
        // What the template handles, read from its text as llama.cpp probes it.
        "chat_template_caps": {
            "supports_tools": template_src.contains("tools"),
            "supports_tool_calls": template_src.contains("tool_call"),
            "supports_system_role": template_src.contains("system"),
            "supports_parallel_tool_calls": template_src.contains("tool_call"),
            "supports_object_arguments": template_src.contains("arguments is mapping")
                || template_src.contains("arguments|items"),
            "supports_preserve_reasoning": template_src.contains("reasoning_content"),
            "supports_reasoning_effort": template_src.contains("reasoning_effort"),
            "supports_string_content": true,
            "supports_typed_content": template_src.contains("content is string")
                || template_src.contains("'type'") || template_src.contains("\"type\""),
        },
        // No media input yet; the marker a client would put in a prompt.
        "media_marker": "<__media__>",
        "bos_token": bos,
        "eos_token": eos,
        "build_info": FINGERPRINT,
        "is_sleeping": false,
        "cors_proxy_enabled": false,
        // airun's web UI
        "model": model_name,
        "arch": cfg.arch_name,
        "n_layers": cfg.n_layers,
        "n_embd": cfg.n_embd,
        "vocab_size": cfg.vocab_size,
        "ctx": ctx,
        "gpus": report.placement.devices(),
        "auth_required": args.api_key.is_some(),
        "thinking": thinking.supported,
        "details": details,
    });
    let created = now();
    let models = json!({
        "models": [{
            "name": model_name, "model": model_name, "modified_at": "", "size": "", "digest": "",
            "type": "model", "description": "", "tags": [""], "capabilities": ["completion"],
            "parameters": "",
            "details": { "parent_model": "", "format": "gguf", "family": "", "families": [""],
                         "parameter_size": "", "quantization_level": "" },
        }],
        "object": "list",
        "data": [{
            "id": model_name,
            "aliases": [model_name],
            "tags": [],
            "object": "model",
            "architecture": { "input_modalities": if vision.is_some() { json!(["text", "image"]) } else { json!(["text"]) },
                              "output_modalities": ["text"] },
            "created": created,
            "owned_by": "airun",
            "meta": {
                "vocab_type": 2, "n_vocab": cfg.vocab_size, "n_ctx": ctx, "n_ctx_train": cfg.train_ctx,
                "n_embd": cfg.n_embd, "n_params": n_params, "size": size, "ftype": ftype,
            },
        }],
    });

    let tokenizer = Arc::clone(&runtime.tokenizer);
    let shared_rt = Arc::new(Mutex::new(runtime));
    let sched = (args.parallel > 1).then(|| std::sync::Mutex::new(crate::sched::spawn(Arc::clone(&shared_rt))));
    let state = Arc::new(AppState {
        sched,
        vision,
        image_memo: std::sync::Mutex::new(Vec::new()),
        runtime: shared_rt,
        tokenizer,
        model_name: model_name.clone(),
        api_key: args.api_key.clone(),
        default_max_tokens: args.default_max_tokens,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        reasoning_format: args.reasoning_format.clone(),
        props,
        models,
        template,
        thinking,
        ctx,
        processing: AtomicBool::new(false),
        tasks: AtomicU64::new(0),
        slot: std::sync::Mutex::new(SlotInfo::default()),
        pieces,
        eog,
        default_breakers,
        fim,
        reasoning_budget: (args.reasoning_budget >= 0).then_some(args.reasoning_budget as usize),
        reasoning_budget_message: args.reasoning_budget_message.clone(),
        reasoning_default: crate::reasoning::parse_switch(&args.reasoning).unwrap_or(None),
        reasoning_effort: args.reasoning_effort.clone(),
        template_effort: template_src.contains("reasoning_effort"),
        slot_save_path: args.slot_save_path.clone(),
        embeddings: args.embeddings,
        pooling,
        think_tags,
        active: Default::default(),
        hw_devices: airun_cuda::devices().unwrap_or_default().into_iter()
            .map(|d| (d.ordinal, d.name.clone(), d.pci_bus_id())).collect(),
        hw_cache: std::sync::Mutex::new(None),
    });

    let app = Router::new()
        .route("/", get(web_ui))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/props", get(props_handler).post(props_post))
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models))
        .route("/slots", get(slots_handler))
        .route("/slots/{id}", post(slot_action))
        .route("/metrics", get(metrics_handler))
        .route("/hardware", get(hardware_handler))
        .route("/completion", post(native_completion))
        .route("/completions", post(native_completion))
        .route("/v1/completions", post(oai_completions))
        .route("/chat/completions", post(chat_completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/chat/completions/input_tokens", post(chat_input_tokens))
        .route("/v1/chat/completions/input_tokens", post(chat_input_tokens))
        .route("/chat/completions/control", post(completion_control))
        .route("/v1/chat/completions/control", post(completion_control))
        .route("/responses", post(responses))
        .route("/v1/responses", post(responses))
        .route("/responses/input_tokens", post(responses_input_tokens))
        .route("/v1/responses/input_tokens", post(responses_input_tokens))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/infill", post(infill))
        .route("/embedding", post(embeddings_native))
        .route("/embeddings", post(embeddings_native))
        .route("/v1/embeddings", post(embeddings_oai))
        .route("/rerank", post(no_rerank))
        .route("/reranking", post(no_rerank))
        .route("/v1/rerank", post(no_rerank))
        .route("/v1/reranking", post(no_rerank))
        .route("/audio/transcriptions", post(no_transcriptions))
        .route("/v1/audio/transcriptions", post(no_transcriptions))
        .route("/lora-adapters", get(lora_list).post(lora_set))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
        .route("/v1/stream/{conv_id}", get(stream_replay).delete(stream_stop))
        .route("/v1/streams/lookup", post(streams_lookup))
        .layer(axum::middleware::from_fn(crate::replay::tee))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let auth = if args.api_key.is_some() { "enabled" } else { "disabled (no --api-key)" };

    // The runtime is created here rather than via #[tokio::main] so that model
    // loading happens on the main thread, before any async machinery starts.
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        crate::replay::spawn_gc();
        let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("could not bind {addr}"))?;
        eprintln!("\nairun serving {model_name} on http://{addr}");
        eprintln!("  web ui:         http://{addr}/");
        eprintln!("  authentication: {auth}");
        eprintln!("  endpoints: /v1/chat/completions /v1/responses /v1/messages /v1/completions /completion /infill /tokenize \
                   /detokenize /apply-template /v1/models /props /slots /health /hardware /v1/stream/{{id}}");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("\nshutting down");
            })
            .await
            .context("server error")
    })
}

/// llama-server's generation_settings object for these parameters.
fn generation_settings(spec: &GenSpec) -> Value {
    let p = &spec.params;
    let n_predict = spec.max_tokens.map_or(-1, |m| m as i64);
    json!({
        "seed": if p.seed == 0 { 4_294_967_295u64 } else { p.seed },
        "temperature": p.temperature,
        "dynatemp_range": p.dynatemp_range, "dynatemp_exponent": p.dynatemp_exponent,
        "top_k": p.top_k, "top_p": p.top_p, "min_p": p.min_p,
        "top_n_sigma": if p.top_n_sigma > 0.0 { p.top_n_sigma } else { -1.0 },
        "xtc_probability": p.xtc_probability, "xtc_threshold": p.xtc_threshold,
        "typical_p": p.typical_p,
        "repeat_last_n": p.repeat_last_n, "repeat_penalty": p.repeat_penalty,
        "presence_penalty": p.presence_penalty, "frequency_penalty": p.frequency_penalty,
        "dry_multiplier": p.dry_multiplier, "dry_base": p.dry_base, "dry_allowed_length": p.dry_allowed_length,
        "dry_penalty_last_n": p.dry_penalty_last_n,
        "dry_sequence_breakers": spec.dry_breakers,
        "mirostat": p.mirostat, "mirostat_tau": p.mirostat_tau, "mirostat_eta": p.mirostat_eta,
        "adaptive_target": -1.0, "adaptive_decay": 0.9,
        "stop": spec.stops,
        "max_tokens": n_predict, "n_predict": n_predict,
        "n_keep": 0, "n_discard": 0,
        "ignore_eos": spec.ignore_eos,
        "stream": spec.stream,
        // A banned token's bias is -inf, which JSON writes as null, as llama-server does.
        "logit_bias": p.logit_bias.iter().map(|(t, b)| json!({ "bias": b, "token": t })).collect::<Vec<_>>(),
        "n_probs": spec.n_probs.unwrap_or(0), "min_keep": 0,
        "grammar": spec.grammar.as_ref().map_or("", |g| g.0.as_str()),
        "grammar_lazy": false, "grammar_triggers": [], "preserved_tokens": [],
        "chat_format": "Content-only",
        "reasoning_format": spec.reasoning_format,
        "reasoning_in_content": false,
        "generation_prompt": "",
        "samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
        "speculative.types": "none",
        "timings_per_token": false, "post_sampling_probs": false, "backend_sampling": false,
        "lora": [],
    })
}

// --- errors, auth -----------------------------------------------------------

/// llama-server's error body: `{"error": {"code", "message", "type"}}`.
fn api_error(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    let body = json!({ "error": { "code": status.as_u16(), "message": message.into(), "type": kind } });
    (status, Json(body)).into_response()
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = &state.api_key else { return Ok(()) };
    let provided = headers.get("authorization").and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ")).map(str::trim);
    match provided {
        Some(k) if k == expected => Ok(()),
        Some(_) => Err(api_error(StatusCode::UNAUTHORIZED, "Invalid API Key", "authentication_error")),
        None => Err(api_error(StatusCode::UNAUTHORIZED,
            "Missing bearer token. Send `Authorization: Bearer <key>`.", "authentication_error")),
    }
}

// --- stream replay (see replay.rs) -------------------------------------------

#[derive(serde::Deserialize)]
struct ReplayQuery {
    from: Option<String>,
}

/// GET /v1/stream/{conv_id}?from=N: the buffered SSE bytes from N, then the
/// live ones until the generation ends.
async fn stream_replay(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(conv_id): Path<String>,
                       Query(q): Query<ReplayQuery>) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return r;
    }
    let Some(session) = crate::replay::get(&conv_id) else {
        return api_error(StatusCode::NOT_FOUND,
            "Stream not found or expired (a finished stream is kept for 5 minutes); send the request again",
            "not_found_error");
    };
    let from = match q.from.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None => 0,
        Some(s) => match s.parse::<usize>() {
            Ok(n) => n,
            Err(_) => return api_error(StatusCode::BAD_REQUEST,
                format!("Invalid 'from' offset {s:?}: pass the number of bytes already received"),
                "invalid_request_error"),
        },
    };
    if from < session.dropped_bytes() {
        return api_error(StatusCode::BAD_REQUEST,
            "Stream offset lost: only the last 4 MiB of a stream are kept; send the request again",
            "invalid_request_error");
    }
    Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
        .header(axum::http::header::CACHE_CONTROL, "no-cache")
        .body(crate::replay::reader(session, from))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// POST /v1/streams/lookup {"conversation_ids": [...]}: the sessions of ids
/// the caller already knows.
async fn streams_lookup(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return r;
    }
    let req = match parse_body(&body) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let ids: Vec<String> = req["conversation_ids"].as_array().map(|a| a.iter()
        .filter_map(|v| v.as_str()).filter(|s| !s.is_empty()).map(String::from).collect()).unwrap_or_default();
    let found: Vec<Value> = crate::replay::lookup(&ids).iter().map(|s| json!({
        "conversation_id": s.id,
        "is_done": s.is_done(),
        "total_bytes": s.total_bytes(),
        "started_at": s.started_at,
        "completed_at": s.completed_at(),
    })).collect();
    Json(Value::Array(found)).into_response()
}

/// DELETE /v1/stream/{conv_id}: the user's Stop. Cancels the generation and
/// drops its buffer; 204 even when there was nothing to stop.
async fn stream_stop(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(conv_id): Path<String>) -> Response {
    if let Err(r) = check_auth(&state, &headers) {
        return r;
    }
    crate::replay::cancel(&conv_id);
    StatusCode::NO_CONTENT.into_response()
}

/// The body as JSON, or llama-server's 400 when it is not.
fn parse_body(body: &axum::body::Bytes) -> Result<Value, Response> {
    serde_json::from_slice(body).map_err(|e|
        api_error(StatusCode::BAD_REQUEST, format!("the request body is not valid JSON: {e}"), "invalid_request_error"))
}

// --- small endpoints ---------------------------------------------------------

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// The chat page, compiled into the binary.
async fn web_ui() -> impl IntoResponse {
    axum::response::Html(include_str!("webui.html"))
}

/// Model and server facts. Unauthenticated, as llama-server's: it exposes
/// only what the operator chose to serve, and the web UI needs it to tell
/// "wrong key" from "server down".
async fn props_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.props.clone())
}

async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    Json(state.models.clone()).into_response()
}

async fn slots_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let s = state.slot.lock().unwrap();
    Json(json!([{
        "id": 0,
        "n_ctx": state.ctx,
        "speculative": false,
        "is_processing": state.processing.load(Ordering::Relaxed),
        "id_task": s.id_task,
        "n_prompt_tokens": s.n_prompt_tokens,
        "n_prompt_tokens_processed": s.n_prompt_tokens_processed,
        "n_prompt_tokens_cache": s.n_prompt_tokens_cache,
        "params": s.params,
        "next_token": [{ "has_next_token": false, "has_new_line": false, "n_remain": -1, "n_decoded": s.n_decoded }],
    }])).into_response()
}

/// llama-server answers 501 unless started with `--metrics`; airun has none.
/// Live hardware readings: each GPU's load, memory, temperature, power,
/// clocks and PCIe traffic (from NVML, when the driver has it), and the RAM
/// of the machine and of this process, pinned RAM included. airun's own plan
/// per GPU is in /props (`details.gpus`).
async fn hardware_handler(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let st = Arc::clone(&state);
    match tokio::task::spawn_blocking(move || hardware_reading(&st)).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("could not read the hardware: {e}"),
            "server_error"),
    }
}

fn hardware_reading(state: &AppState) -> Value {
    const FRESH: std::time::Duration = std::time::Duration::from_millis(700);
    // Held across the reading, so concurrent requests wait for one reading
    // instead of each sampling the GPUs.
    let mut cache = state.hw_cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, v)) = cache.as_ref() {
        if at.elapsed() < FRESH {
            return v.clone();
        }
    }
    let nvml = airun_cuda::nvml::available();
    let gpus: Vec<Value> = state.hw_devices.iter().map(|(ordinal, name, pci)| {
        let s = airun_cuda::nvml::gpu_stats(pci).unwrap_or_default();
        // Without NVML, the VRAM figures still come from CUDA.
        let (total, used) = match (s.mem_total, s.mem_used) {
            (Some(t), Some(u)) => (Some(t), Some(u)),
            _ => match airun_cuda::device::mem_info(*ordinal) {
                Ok((free, total)) => (Some(total), Some(total.saturating_sub(free))),
                Err(_) => (None, None),
            },
        };
        json!({
            "device": ordinal, "name": name, "pci_bus_id": pci,
            "vram_total": total, "vram_used": used,
            "util_gpu": s.util_gpu, "util_mem": s.util_mem,
            "temp_c": s.temp_c, "power_w": s.power_w, "power_limit_w": s.power_limit_w,
            "clock_sm_mhz": s.clock_sm_mhz, "clock_sm_max_mhz": s.clock_sm_max_mhz,
            "clock_mem_mhz": s.clock_mem_mhz, "fan_pct": s.fan_pct,
            "pcie_tx": s.pcie_tx, "pcie_rx": s.pcie_rx, "pcie_gen": s.pcie_gen, "pcie_width": s.pcie_width,
            "pstate": s.pstate,
        })
    }).collect();
    let process = airun_core::disk::process_memory();
    let v = json!({
        "time_ms": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64),
        "nvml": nvml,
        "driver": airun_cuda::nvml::driver_version(),
        "gpus": gpus,
        "ram": {
            "total": airun_core::disk::total_ram(),
            "available": airun_core::disk::available_ram(),
            "process_resident": process.map(|p| p.0),
            "process_committed": process.and_then(|p| p.1),
            "pinned": airun_cuda::pinned_bytes(),
        },
        "processing": state.processing.load(Ordering::Relaxed),
    });
    *cache = Some((std::time::Instant::now(), v.clone()));
    v
}

async fn metrics_handler() -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED,
        "This server does not support metrics endpoint. Start it with `--metrics`", "not_supported_error")
}

// The answers llama-server gives for features it was started without, which
// airun does not have: a client probing for them gets the same signal. Each
// reads the body it ignores, or the connection can close while the client is
// still sending it and the client sees a reset instead of the answer.

fn no_embeddings() -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED,
        "This server does not support embeddings. Start it with `--embeddings`", "not_supported_error")
}

// --- embeddings ------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Pooling {
    None,
    Mean,
    Cls,
    Last,
}

impl Pooling {
    fn from_name(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "mean" => Self::Mean,
            "cls" => Self::Cls,
            _ => Self::Last,
        }
    }
}

/// llama.cpp's `embd_normalize`: -1 none, 0 max-absolute scaled to int16,
/// 1 taxicab, 2 euclidean (the default), p > 2 the p-norm.
fn normalize(v: &mut [f32], p: i64) {
    let norm = match p {
        ..0 => return,
        0 => v.iter().fold(0f32, |m, x| m.max(x.abs())) / 32760.0,
        1 => v.iter().map(|x| x.abs()).sum(),
        2 => v.iter().map(|x| x * x).sum::<f32>().sqrt(),
        p => v.iter().map(|x| x.abs().powi(p as i32)).sum::<f32>().powf(1.0 / p as f32),
    };
    let scale = if norm > 0.0 { 1.0 / norm } else { 0.0 };
    v.iter_mut().for_each(|x| *x *= scale);
}

fn base64(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= c.len() { A[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// The texts of an embedding request, tokenized: a string, token ids, or a
/// list of either (llama.cpp's prompt shapes).
fn embedding_inputs(rt: &Runtime, v: &Value) -> Vec<Vec<u32>> {
    let one = |x: &Value| -> Option<Vec<u32>> {
        match x {
            Value::String(s) => Some(rt.tokenizer.encode(s, true)),
            Value::Array(a) if a.iter().all(Value::is_u64) => Some(a.iter().map(|t| t.as_u64().unwrap() as u32).collect()),
            Value::Array(a) => {
                // Mixed strings and ids form one prompt.
                let mut ids = Vec::new();
                for p in a {
                    match p {
                        Value::String(s) => ids.extend(rt.tokenizer.encode(s, ids.is_empty())),
                        t => ids.push(t.as_u64()? as u32),
                    }
                }
                Some(ids)
            }
            _ => None,
        }
    };
    match v {
        Value::Array(a) if !a.is_empty() && !a.iter().all(Value::is_u64)
            && a.iter().all(|x| x.is_string() || x.is_array()) => a.iter().filter_map(one).collect(),
        x => one(x).into_iter().collect(),
    }
}

/// Runs each input and pools it: one vector per input, or one per token
/// with no pooling. Returns the vectors and the tokens each input took.
fn embed_all(state: &AppState, inputs: &[Vec<u32>], norm: i64) -> airun_core::Result<Vec<(Vec<Vec<f32>>, usize)>> {
    let mut rt = state.runtime.blocking_lock();
    let n_embd = rt.model.config.n_embd;
    let mut out = Vec::new();
    for tokens in inputs {
        let h = rt.hidden_states(tokens)?;
        let rows: Vec<&[f32]> = h.chunks_exact(n_embd).collect();
        let vecs = match state.pooling {
            // Token states come out as they are, as llama.cpp's do.
            Pooling::None => rows.iter().map(|r| r.to_vec()).collect(),
            pooling => {
                let mut v = match pooling {
                    Pooling::Mean => {
                        let mut m = vec![0f32; n_embd];
                        for r in &rows {
                            m.iter_mut().zip(r.iter()).for_each(|(a, b)| *a += b);
                        }
                        m.iter_mut().for_each(|a| *a /= rows.len() as f32);
                        m
                    }
                    Pooling::Cls => rows[0].to_vec(),
                    _ => rows[rows.len() - 1].to_vec(),
                };
                normalize(&mut v, norm);
                vec![v]
            }
        };
        out.push((vecs, tokens.len()));
    }
    Ok(out)
}

/// The inputs and normalization of an embedding request, or the error.
async fn embedding_request(state: &AppState, req: &Value, input: Option<&Value>)
    -> Result<(Vec<Vec<u32>>, i64), Response> {
    let Some(input) = input else {
        return Err(api_error(StatusCode::BAD_REQUEST, "\"input\" or \"content\" must be provided",
            "invalid_request_error"));
    };
    let inputs = embedding_inputs(&*state.runtime.lock().await, input);
    if inputs.is_empty() || inputs.iter().any(Vec::is_empty) {
        return Err(api_error(StatusCode::BAD_REQUEST, "Input content cannot be empty", "invalid_request_error"));
    }
    Ok((inputs, req.get("embd_normalize").and_then(Value::as_i64).unwrap_or(2)))
}

/// llama.cpp's native `/embedding`: `[{index, embedding: [[...]]}]`, one
/// entry per input, with a vector per token when pooling is none.
async fn embeddings_native(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    if !state.embeddings {
        return no_embeddings();
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let (inputs, norm) = match embedding_request(&state, &req, req.get("content").or_else(|| req.get("input"))).await {
        Ok(x) => x,
        Err(e) => return e,
    };
    let st = Arc::clone(&state);
    match tokio::task::spawn_blocking(move || embed_all(&st, &inputs, norm)).await {
        Ok(Ok(all)) => Json(Value::Array(all.into_iter().enumerate()
            .map(|(i, (vecs, _))| json!({ "index": i, "embedding": vecs })).collect())).into_response(),
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, e.to_string(), "invalid_request_error"),
        Err(e) => generation_failed(e),
    }
}

/// OpenAI's `/v1/embeddings`, floats or base64.
async fn embeddings_oai(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    if !state.embeddings {
        return no_embeddings();
    }
    if state.pooling == Pooling::None {
        return api_error(StatusCode::BAD_REQUEST,
            "Pooling type 'none' is not OAI compatible. Please use a different pooling type (--pooling)",
            "invalid_request_error");
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let base64_out = match req.get("encoding_format").and_then(Value::as_str) {
        None | Some("float") => false,
        Some("base64") => true,
        Some(_) => return api_error(StatusCode::BAD_REQUEST,
            "The format to return the embeddings in. Can be either float or base64", "invalid_request_error"),
    };
    let (inputs, norm) = match embedding_request(&state, &req, req.get("input")).await {
        Ok(x) => x,
        Err(e) => return e,
    };
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let st = Arc::clone(&state);
    match tokio::task::spawn_blocking(move || embed_all(&st, &inputs, norm)).await {
        Ok(Ok(all)) => {
            let n: usize = all.iter().map(|x| x.1).sum();
            let data: Vec<Value> = all.into_iter().enumerate().map(|(i, (mut vecs, _))| {
                let v = vecs.swap_remove(0);
                if base64_out {
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    json!({ "embedding": base64(&bytes), "index": i, "object": "embedding", "encoding_format": "base64" })
                } else {
                    json!({ "embedding": v, "index": i, "object": "embedding" })
                }
            }).collect();
            Json(json!({ "model": model, "object": "list", "usage": { "prompt_tokens": n, "total_tokens": n },
                "data": data })).into_response()
        }
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, e.to_string(), "invalid_request_error"),
        Err(e) => generation_failed(e),
    }
}

async fn no_rerank(_body: axum::body::Bytes) -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED,
        "This server does not support reranking. Start it with `--reranking`", "not_supported_error")
}

async fn no_transcriptions(_body: axum::body::Bytes) -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED, "The current model does not support audio input.", "not_supported_error")
}

async fn props_post(_body: axum::body::Bytes) -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED,
        "This server does not support changing global properties. Start it with `--props`", "not_supported_error")
}

/// llama-server's slot actions on the one slot: `save` writes the cache of
/// the current conversation to a file under `--slot-save-path`, `restore`
/// loads one back (the next request that continues it skips the replay, which
/// on a hybrid model is the only way to resume a long prompt without
/// reprocessing it), `erase` empties the cache.
async fn slot_action(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>,
    Query(query): Query<std::collections::HashMap<String, String>>, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let Some(dir) = state.slot_save_path.clone() else {
        return api_error(StatusCode::NOT_IMPLEMENTED,
            "This server does not support slots action. Start it with `--slot-save-path`", "not_supported_error");
    };
    let bad = |m: &str| api_error(StatusCode::BAD_REQUEST, m, "invalid_request_error");
    if id.parse::<i64>().ok() != Some(0) {
        return bad("Invalid slot ID");
    }
    let action = query.get("action").map(String::as_str).unwrap_or_default().to_string();
    if !["save", "restore", "erase"].contains(&action.as_str()) {
        return bad("Invalid action");
    }
    let req = if body.is_empty() { json!({}) } else { match parse_body(&body) { Ok(v) => v, Err(e) => return e } };
    let filename = req.get("filename").and_then(Value::as_str).unwrap_or_default().to_string();
    // A bare file name: no directories, so a request cannot reach outside
    // the save path.
    let valid_name = !filename.is_empty() && filename.len() <= 255 && filename != "." && filename != ".."
        && !filename.chars().any(|c| matches!(c, '/' | '\\' | ':' | '\0') || c.is_control());
    if action != "erase" && !valid_name {
        return bad("Invalid filename");
    }
    let st = Arc::clone(&state);
    let out = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let mut rt = st.runtime.blocking_lock();
        let start = std::time::Instant::now();
        let path = dir.join(&filename);
        match action.as_str() {
            "save" => {
                let bytes = rt.save_state().map_err(|e| e.to_string())?;
                std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, &bytes))
                    .map_err(|e| format!("could not write {}: {e}; check --slot-save-path", path.display()))?;
                Ok(json!({ "id_slot": 0, "filename": filename, "n_saved": rt.n_past(), "n_written": bytes.len(),
                    "timings": { "save_ms": start.elapsed().as_secs_f64() * 1000.0 } }))
            }
            "restore" => {
                let bytes = std::fs::read(&path)
                    .map_err(|e| format!("Unable to restore slot: could not read {}: {e}", path.display()))?;
                let n = rt.load_state(&bytes).map_err(|e| format!("Unable to restore slot: {e}"))?;
                Ok(json!({ "id_slot": 0, "filename": filename, "n_restored": n, "n_read": bytes.len(),
                    "timings": { "restore_ms": start.elapsed().as_secs_f64() * 1000.0 } }))
            }
            _ => {
                let n = rt.n_past();
                rt.reset();
                Ok(json!({ "id_slot": 0, "n_erased": n }))
            }
        }
    }).await;
    match out {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => bad(&e),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("slot action failed: {e}"), "server_error"),
    }
}

/// No LoRA adapters are loaded, so there is nothing to list or scale.
async fn lora_list(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    Json(json!([])).into_response()
}

async fn lora_set(State(state): State<Arc<AppState>>, headers: HeaderMap, _body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    Json(json!({ "success": true })).into_response()
}

/// llama-server's control of a running completion: `reasoning_end` makes a
/// chat completion that asked for `reasoning_control` close its thinking at
/// the next token and move on to the answer. It runs beside the generation
/// (it never takes the model lock), so a client sends it mid-stream.
async fn completion_control(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let Some(id) = req.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()) else {
        return api_error(StatusCode::BAD_REQUEST, "missing completion id", "invalid_request_error");
    };
    if req.get("action").and_then(Value::as_str) != Some("reasoning_end") {
        return api_error(StatusCode::BAD_REQUEST,
            "unknown control action; the only one is \"reasoning_end\"", "invalid_request_error");
    }
    let found = state.active.lock().unwrap().get(id).cloned();
    let body = match found {
        None => json!({ "success": false, "message": "no active completion for this id" }),
        Some((false, _)) => json!({ "success": false,
            "message": "reasoning control not enabled for this completion; send `reasoning_control: true` with it" }),
        Some((true, stop)) => {
            stop.store(true, Ordering::Relaxed);
            json!({ "success": true })
        }
    };
    Json(body).into_response()
}

async fn tokenize(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let content = req.get("content").and_then(Value::as_str).unwrap_or_default().to_string();
    let add_special = req.get("add_special").and_then(Value::as_bool).unwrap_or(false);
    let with_pieces = req.get("with_pieces").and_then(Value::as_bool).unwrap_or(false);
    let tk = &*state.tokenizer;
    let ids = tk.encode(&content, add_special);
    let tokens: Vec<Value> = if with_pieces {
        ids.iter().map(|&id| {
            let bytes = tk.decode_bytes(&[id]);
            let piece = match String::from_utf8(bytes.clone()) {
                Ok(s) => Value::String(s),
                // A piece that is part of a multi-byte character, as bytes.
                Err(_) => json!(bytes),
            };
            json!({ "id": id, "piece": piece })
        }).collect()
    } else {
        ids.iter().map(|&id| json!(id)).collect()
    };
    Json(json!({ "tokens": tokens })).into_response()
}

async fn detokenize(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let ids: Vec<u32> = req.get("tokens").and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|t| t.as_u64().map(|x| x as u32)).collect()).unwrap_or_default();
    let content = state.tokenizer.decode(&ids, false);
    Json(json!({ "content": content })).into_response()
}

async fn apply_template(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    match chat_prompt(&state, &req) {
        Ok((prompt, _)) => Json(json!({ "prompt": prompt })).into_response(),
        Err(e) => e,
    }
}

// --- generation --------------------------------------------------------------

/// What every generating endpoint asks of the model.
struct GenSpec {
    tokens: Vec<u32>,
    params: SamplingParams,
    /// None: until the end of generation or of the context.
    max_tokens: Option<usize>,
    stops: Vec<String>,
    stream: bool,
    reasoning_format: String,
    ignore_eos: bool,
    /// The grammar's text, as generation_settings reports it, and the grammar.
    grammar: Option<(String, Grammar)>,
    /// Report each token's probability and this many alternatives.
    n_probs: Option<usize>,
    dry_breakers: Vec<String>,
    /// A limit on the reply's reasoning, or a way to end it from `/control`.
    reasoning: Option<airun_core::ReasoningBudget>,
    /// Told after each prefill batch (llama-server's `return_progress`).
    progress: Option<ProgressFn>,
}

type ProgressFn = Arc<dyn Fn(airun_core::runtime::PrefillProgress) + Send + Sync>;

/// llama-server's `prompt_progress` object.
fn prompt_progress_json(p: airun_core::runtime::PrefillProgress) -> Value {
    json!({ "total": p.total, "cache": p.cached, "processed": p.processed, "time_ms": p.elapsed_ms })
}

/// How a finished generation went.
struct GenEnd {
    stats: airun_core::GenerationStats,
    stop_type: &'static str,
    stopping_word: String,
    /// One per sampled token, the end-of-generation token included, when asked.
    probs: Vec<TokenProbs>,
}

impl AppState {
    /// Everything a generating request asks for but its prompt and length.
    /// `n_probs` is how many alternatives per token the endpoint's own field
    /// asks to report. Fails on an invalid grammar or schema.
    fn spec_from(&self, req: &Value, native: bool, tokens: Vec<u32>, max_tokens: Option<usize>,
        n_probs: Option<usize>) -> Result<GenSpec, Response> {
        let mut params = self.params_from(req, native);
        let ignore_eos = req.get("ignore_eos").and_then(Value::as_bool).unwrap_or(false);
        if ignore_eos {
            params.logit_bias.extend(self.eog.iter().map(|&t| (t, f32::NEG_INFINITY)));
        }
        let dry_breakers = match req.get("dry_sequence_breakers").and_then(Value::as_array) {
            Some(a) => {
                let b: Vec<String> = a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect();
                params.dry_breakers = Arc::new(breaker_mask(&self.pieces, &b));
                b
            }
            None => {
                params.dry_breakers = Arc::clone(&self.default_breakers);
                DRY_BREAKERS.map(str::to_string).to_vec()
            }
        };
        Ok(GenSpec {
            tokens,
            params,
            max_tokens,
            stops: stop_sequences(req.get("stop")),
            stream: req.get("stream").and_then(Value::as_bool).unwrap_or(false),
            reasoning_format: self.reasoning_format_of(req),
            ignore_eos,
            grammar: self.grammar_of(req)?,
            n_probs,
            dry_breakers,
            reasoning: None,
            progress: None,
        })
    }

    /// The constraint a request asks for: a GBNF `grammar`, a `json_schema`,
    /// or an OpenAI `response_format`, in that order.
    fn grammar_of(&self, req: &Value) -> Result<Option<(String, Grammar)>, Response> {
        let bad = |m: String| api_error(StatusCode::BAD_REQUEST, m, "invalid_request_error");
        let from_schema = |s: &Value| json_schema_grammar(s)
            .map_err(|e| bad(format!("json_schema cannot be turned into a grammar: {e}")));
        let text = if let Some(g) = req.get("grammar").and_then(Value::as_str).filter(|g| !g.trim().is_empty()) {
            g.to_string()
        } else if let Some(s) = req.get("json_schema").filter(|s| s.is_object()) {
            from_schema(s)?
        } else if let Some(rf) = req.get("response_format").filter(|r| r.is_object()) {
            match rf.get("type").and_then(Value::as_str) {
                Some("json_object") => match rf.get("schema") {
                    Some(s) => from_schema(s)?,
                    None => json_object_grammar(),
                },
                Some("json_schema") => {
                    let s = rf.pointer("/json_schema/schema").or_else(|| rf.get("schema")).ok_or_else(|| {
                        bad("response_format of type json_schema needs `json_schema.schema`".into())
                    })?;
                    from_schema(s)?
                }
                Some("text") | None => return Ok(None),
                Some(t) => return Err(bad(format!(
                    "response_format type '{t}' is not supported; use text, json_object or json_schema"))),
            }
        } else {
            return Ok(None);
        };
        let g = Grammar::new(&text, Arc::clone(&self.pieces), Arc::clone(&self.eog))
            .map_err(|e| bad(format!("invalid grammar: {e}")))?;
        Ok(Some((text, g)))
    }

    fn params_from(&self, req: &Value, native: bool) -> SamplingParams {
        let f = |k: &str| req.get(k).and_then(Value::as_f64).map(|x| x as f32);
        let u = |k: &str| req.get(k).and_then(Value::as_i64);
        let last_n = |k: &str, default: usize| u(k).map_or(default, |x| if x < 0 { self.ctx } else { x as usize });
        SamplingParams {
            dynatemp_range: f("dynatemp_range").unwrap_or(0.0),
            dynatemp_exponent: f("dynatemp_exponent").unwrap_or(1.0),
            xtc_probability: f("xtc_probability").unwrap_or(0.0),
            xtc_threshold: f("xtc_threshold").unwrap_or(0.1),
            dry_multiplier: f("dry_multiplier").unwrap_or(0.0),
            dry_base: f("dry_base").unwrap_or(1.75),
            dry_allowed_length: u("dry_allowed_length").map_or(2, |x| x.max(0) as usize),
            dry_penalty_last_n: last_n("dry_penalty_last_n", self.repeat_last_n),
            dry_breakers: Arc::clone(&self.default_breakers),
            mirostat: u("mirostat").map_or(0, |x| x.clamp(0, 2) as u8),
            mirostat_tau: f("mirostat_tau").unwrap_or(5.0),
            mirostat_eta: f("mirostat_eta").unwrap_or(0.1),
            temperature: f("temperature").unwrap_or(0.8),
            top_k: u("top_k").map_or(40, |x| x.max(0) as usize),
            top_p: f("top_p").unwrap_or(0.95),
            min_p: f("min_p").unwrap_or(0.05),
            typical_p: f("typical_p").unwrap_or(1.0),
            top_n_sigma: f("top_n_sigma").filter(|&x| x > 0.0).unwrap_or(0.0),
            // OpenAI's schema has no multiplicative penalty, so OpenAI requests
            // keep the operator's default; llama.cpp's native one defaults to off.
            repeat_penalty: f("repeat_penalty").unwrap_or(if native { 1.0 } else { self.repeat_penalty }),
            presence_penalty: f("presence_penalty").unwrap_or(0.0),
            frequency_penalty: f("frequency_penalty").unwrap_or(0.0),
            repeat_last_n: last_n("repeat_last_n", self.repeat_last_n),
            logit_bias: logit_bias_from(req.get("logit_bias")),
            seed: u("seed").filter(|&s| s >= 0 && s != u32::MAX as i64).map_or(0, |s| s as u64),
        }
    }

    fn reasoning_format_of(&self, req: &Value) -> String {
        req.get("reasoning_format").and_then(Value::as_str).unwrap_or(&self.reasoning_format).to_string()
    }
}

fn stop_sequences(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// Runs a generation on the calling (blocking) thread. `on_text(accumulated,
/// hold, token, probs)` sees the text so far after each token, with how many
/// trailing bytes may still turn into a stop sequence and the probabilities
/// recorded since the last call (when asked for); returning false stops early.
/// [`run_generation`] through the --parallel scheduler: the job goes to its
/// thread and the tokens come back here, where stop sequences and the
/// callback run as before.
fn run_generation_scheduled(state: &AppState, spec: &GenSpec,
    mut on_text: impl FnMut(&str, usize, u32, &[TokenProbs]) -> bool) -> Result<(GenEnd, String)> {
    let sink: ProbsSink = Default::default();
    let mut sampler = Sampler::new(spec.params.clone());
    if let Some((_, g)) = &spec.grammar {
        sampler = sampler.with_grammar(g.clone());
    }
    if let Some(n) = spec.n_probs {
        sampler = sampler.with_probs(n, Arc::clone(&sink));
    }
    if let Some(r) = &spec.reasoning {
        sampler = sampler.with_reasoning_budget(r.clone());
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let (etx, erx) = std::sync::mpsc::channel();
    let job = crate::sched::Job {
        tokens: spec.tokens.clone(),
        max_tokens: spec.max_tokens.unwrap_or(state.ctx),
        sampler,
        events: etx,
        cancel: Arc::clone(&cancel),
        progress: spec.progress.as_ref().map(|p| {
            let p = Arc::clone(p);
            Box::new(move |x| p(x)) as airun_core::runtime::PrefillHook
        }),
    };
    state.sched.as_ref().expect("checked by the caller").lock().unwrap_or_else(|e| e.into_inner()).send(job)
        .map_err(|_| anyhow::anyhow!("the scheduler has stopped; restart the server"))?;
    state.processing.store(true, Ordering::Relaxed);
    let mut text = String::new();
    let mut word: Option<String> = None;
    let mut seen = 0usize;
    let result = loop {
        match erx.recv() {
            Ok(crate::sched::Event::Token(tok, piece)) => {
                if cancel.load(Ordering::Relaxed) {
                    continue;
                }
                let fresh: Vec<TokenProbs> = {
                    let s = sink.lock().unwrap();
                    let v = s[seen.min(s.len())..].to_vec();
                    seen = s.len();
                    v
                };
                text.push_str(&piece);
                let keep = if let Some((cut, w)) = stop_at(&text, &spec.stops) {
                    text.truncate(cut);
                    word = Some(w.to_string());
                    on_text(&text, 0, tok, &fresh);
                    false
                } else {
                    on_text(&text, ambiguous_tail(&text, &spec.stops), tok, &fresh)
                };
                if !keep {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Ok(crate::sched::Event::Done(r)) => break r,
            Err(_) => break Err("the scheduler dropped the request".to_string()),
        }
    };
    state.processing.store(false, Ordering::Relaxed);
    let mut stats = result.map_err(|e| anyhow::anyhow!("{e}"))?;
    if word.is_some() {
        stats.stop = airun_core::runtime::StopReason::Cancelled;
    }
    log_request(&stats);
    let id_task = state.tasks.fetch_add(1, Ordering::Relaxed);
    *state.slot.lock().unwrap() = SlotInfo {
        id_task,
        n_prompt_tokens: stats.prompt_tokens,
        n_prompt_tokens_processed: stats.processed_tokens(),
        n_prompt_tokens_cache: stats.cached_tokens,
        n_decoded: stats.generated_tokens,
        params: generation_settings(spec),
    };
    let st = stop_type(stats.stop, word.is_some());
    let probs = std::mem::take(&mut *sink.lock().unwrap());
    Ok((GenEnd { stats, stop_type: st, stopping_word: word.unwrap_or_default(), probs }, text))
}

fn run_generation(state: &AppState, spec: &GenSpec,
    mut on_text: impl FnMut(&str, usize, u32, &[TokenProbs]) -> bool) -> Result<(GenEnd, String)> {
    if state.sched.is_some() {
        return run_generation_scheduled(state, spec, on_text);
    }
    let mut rt = state.runtime.blocking_lock();
    state.processing.store(true, Ordering::Relaxed);
    let sink: ProbsSink = Default::default();
    let mut sampler = Sampler::new(spec.params.clone());
    if let Some((_, g)) = &spec.grammar {
        sampler = sampler.with_grammar(g.clone());
    }
    if let Some(n) = spec.n_probs {
        sampler = sampler.with_probs(n, Arc::clone(&sink));
    }
    if let Some(r) = &spec.reasoning {
        sampler = sampler.with_reasoning_budget(r.clone());
    }
    let mut text = String::new();
    let mut word: Option<String> = None;
    let mut seen = 0usize;
    let max = spec.max_tokens.unwrap_or(state.ctx);
    if let Some(p) = &spec.progress {
        let p = Arc::clone(p);
        rt.set_prefill_hook(Some(Box::new(move |x| p(x))));
    }
    // No reset: `generate` reuses whatever prefix of this prompt the caches hold.
    let result = rt.generate(&spec.tokens, max, &mut sampler, |tok, piece| {
        let fresh: Vec<TokenProbs> = {
            let s = sink.lock().unwrap();
            let v = s[seen.min(s.len())..].to_vec();
            seen = s.len();
            v
        };
        text.push_str(piece);
        if let Some((cut, w)) = stop_at(&text, &spec.stops) {
            text.truncate(cut);
            word = Some(w.to_string());
            on_text(&text, 0, tok, &fresh);
            return false;
        }
        on_text(&text, ambiguous_tail(&text, &spec.stops), tok, &fresh)
    });
    rt.set_prefill_hook(None);
    state.processing.store(false, Ordering::Relaxed);
    let stats = result?;
    log_request(&stats);
    let id_task = state.tasks.fetch_add(1, Ordering::Relaxed);
    *state.slot.lock().unwrap() = SlotInfo {
        id_task,
        n_prompt_tokens: stats.prompt_tokens,
        n_prompt_tokens_processed: stats.processed_tokens(),
        n_prompt_tokens_cache: stats.cached_tokens,
        n_decoded: stats.generated_tokens,
        params: generation_settings(spec),
    };
    let st = stop_type(stats.stop, word.is_some());
    let probs = std::mem::take(&mut *sink.lock().unwrap());
    Ok((GenEnd { stats, stop_type: st, stopping_word: word.unwrap_or_default(), probs }, text))
}

/// One token's probabilities in llama-server's form.
fn probs_json(state: &AppState, p: &TokenProbs) -> Value {
    let one = |id: u32, logprob: f32| {
        let bytes = state.pieces.get(id as usize).cloned().unwrap_or_default();
        json!({ "id": id, "token": String::from_utf8_lossy(&bytes), "bytes": bytes, "logprob": logprob })
    };
    let mut v = one(p.token, p.logprob);
    v["top_logprobs"] = p.top.iter().map(|&(id, lp)| one(id, lp)).collect();
    v
}

fn probs_list(state: &AppState, probs: &[TokenProbs]) -> Value {
    Value::Array(probs.iter().map(|p| probs_json(state, p)).collect())
}

type Tx = mpsc::UnboundedSender<std::result::Result<Event, std::convert::Infallible>>;

fn sse(tx: &Tx, v: &Value) -> bool {
    tx.send(Ok(Event::default().data(v.to_string()))).is_ok()
}

fn sse_stream(rx: mpsc::UnboundedReceiver<std::result::Result<Event, std::convert::Infallible>>) -> Response {
    Sse::new(UnboundedReceiverStream::new(rx)).into_response()
}

// --- llama.cpp's native /completion -------------------------------------------

async fn native_completion(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let req = match attach_images(&state, req).await { Ok(v) => v, Err(e) => return e };
    let images = request_images(&req);
    let (prompt_text, tokens) = {
        let rt = &*state.tokenizer;
        match req.get("prompt") {
            // multimodal_data: the prompt's <__media__> markers are the images' places.
            Some(Value::String(s)) if !images.is_empty() => match crate::media::tokenize(rt, &crate::media::mark_slots(s), &images, true) {
                Ok(t) => (s.clone(), t),
                Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("{e:#}"), "invalid_request_error"),
            },
            Some(Value::String(s)) => (s.clone(), rt.encode(s, true)),
            // Token ids, or a mix of ids and strings, as llama-server takes.
            Some(Value::Array(a)) if a.iter().all(|x| x.is_u64() || x.is_string()) && !a.is_empty() => {
                let mut ids = Vec::new();
                for x in a {
                    match x {
                        Value::String(s) => ids.extend(rt.encode(s, ids.is_empty())),
                        v => ids.push(v.as_u64().unwrap_or(0) as u32),
                    }
                }
                (rt.decode(&ids, false), ids)
            }
            _ => return api_error(StatusCode::BAD_REQUEST, "\"prompt\" must be a string, a list of tokens, \
                or a list of mixed strings and tokens", "invalid_request_error"),
        }
    };
    native_generate(state, req, prompt_text, tokens).await
}

/// `/infill`: code completion between `input_prefix` and `input_suffix`, in
/// the model's fill-in-the-middle format (prefix, suffix, then middle, where
/// generation starts after any `prompt` text). Before them, as llama.cpp
/// builds it, a repository header, each `input_extra` file, and the file
/// being edited.
async fn infill(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let Some(fim) = state.fim else {
        return api_error(StatusCode::NOT_IMPLEMENTED,
            "this model has no fill-in-the-middle tokens (<|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>), \
             so /infill cannot work with it; use /completion", "not_supported_error");
    };
    let field = |k: &str| req.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
    let (prompt_text, tokens) = {
        let tk = &*state.tokenizer;
        let enc = |s: &str| tk.encode(s, false);
        let mut t = tk.encode("", true);
        if let Some(repo) = fim.repo {
            t.push(repo);
            t.extend(enc("myproject\n"));
        }
        for f in req.get("input_extra").and_then(Value::as_array).into_iter().flatten() {
            if let Some(sep) = fim.file {
                t.push(sep);
                t.extend(enc(&format!("{}\n", f.get("filename").and_then(Value::as_str).unwrap_or_default())));
            }
            t.extend(enc(f.get("text").and_then(Value::as_str).unwrap_or_default()));
        }
        if let Some(sep) = fim.file {
            t.push(sep);
            t.extend(enc("filename\n"));
        }
        t.push(fim.prefix);
        t.extend(enc(&field("input_prefix")));
        t.push(fim.suffix);
        t.extend(enc(&field("input_suffix")));
        t.push(fim.middle);
        t.extend(enc(&field("prompt")));
        (tk.decode(&t, false), t)
    };
    native_generate(state, req, prompt_text, tokens).await
}

/// llama.cpp's native completion of `tokens`, streamed or not.
async fn native_generate(state: Arc<AppState>, req: Value, prompt_text: String, tokens: Vec<u32>) -> Response {
    if tokens.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "the prompt is empty", "invalid_request_error");
    }
    let n_predict = req.get("n_predict").or_else(|| req.get("max_tokens")).and_then(Value::as_i64).unwrap_or(-1);
    let n_probs = req.get("n_probs").and_then(Value::as_u64).filter(|&n| n > 0).map(|n| n as usize);
    let spec = match state.spec_from(&req, true, tokens, (n_predict >= 0).then_some(n_predict as usize), n_probs) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let return_tokens = req.get("return_tokens").and_then(Value::as_bool).unwrap_or(false);
    let model = state.model_name.clone();

    let final_body = move |end: &GenEnd, content: &str, ids: &[u32], spec: &GenSpec, prompt: &str| json!({
        "index": 0,
        "content": content,
        "tokens": if return_tokens { ids.to_vec() } else { Vec::new() },
        "id_slot": 0,
        "stop": true,
        "model": model,
        "tokens_predicted": end.stats.generated_tokens,
        "tokens_evaluated": end.stats.prompt_tokens,
        "generation_settings": generation_settings(spec),
        "prompt": prompt,
        "has_new_line": content.contains('\n'),
        "truncated": false,
        "stop_type": end.stop_type,
        "stopping_word": end.stopping_word,
        "tokens_cached": end.stats.prompt_tokens + end.stats.generated_tokens,
        "timings": timings(&end.stats),
    });

    if !spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
            let mut ids = Vec::new();
            let r = run_generation(&st, &spec, |_, _, tok, _| { ids.push(tok); true });
            r.map(|(end, text)| {
                let mut body = final_body(&end, &text, &ids, &spec, &prompt_text);
                if spec.n_probs.is_some() {
                    body["completion_probabilities"] = probs_list(&st, &end.probs);
                }
                body
            })
        }).await;
        return match out {
            Ok(Ok(body)) => Json(body).into_response(),
            Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "server_error"),
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("generation task failed: {e}"), "server_error"),
        };
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let st = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let mut sent = 0usize;
        let mut pending: Vec<u32> = Vec::new();
        let mut pending_probs: Vec<TokenProbs> = Vec::new();
        let mut n = 0usize;
        let n_prompt = spec.tokens.len();
        let with_probs = spec.n_probs.is_some();
        let chunk = |st: &AppState, text: &str, toks: Vec<u32>, probs: &[TokenProbs], n: usize| {
            let mut c = json!({
                "index": 0, "content": text, "tokens": toks,
                "stop": false, "id_slot": -1, "tokens_predicted": n, "tokens_evaluated": n_prompt,
            });
            if with_probs {
                c["completion_probabilities"] = probs_list(st, probs);
            }
            c
        };
        let r = run_generation(&st, &spec, |text, hold, tok, probs| {
            pending.push(tok);
            pending_probs.extend_from_slice(probs);
            n += 1;
            let release = floor_boundary(text, text.len().saturating_sub(hold));
            if release <= sent {
                return true;
            }
            let c = chunk(&st, &text[sent..release], std::mem::take(&mut pending),
                &std::mem::take(&mut pending_probs), n);
            sent = release;
            sse(&tx, &c)
        });
        match r {
            Ok((end, text)) => {
                if sent < text.len() {
                    sse(&tx, &chunk(&st, &text[sent..], pending, &pending_probs, n));
                }
                sse(&tx, &final_body(&end, "", &[], &spec, &prompt_text));
            }
            Err(e) => {
                sse(&tx, &json!({ "error": { "code": 500, "message": e.to_string(), "type": "server_error" } }));
            }
        }
    });
    sse_stream(rx)
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

// --- OpenAI /v1/completions ---------------------------------------------------

async fn oai_completions(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let prompt = match req.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        // The array form is a batch in the OpenAI API; airun runs one sequence
        // at a time, so only the first entry is honoured.
        Some(Value::Array(a)) => a.first().and_then(Value::as_str).unwrap_or_default().to_string(),
        _ => String::new(),
    };
    if prompt.is_empty() {
        return api_error(StatusCode::BAD_REQUEST, "\"prompt\" must be provided", "invalid_request_error");
    }
    let tokens = state.tokenizer.encode(&prompt, true);
    let max = req.get("max_tokens").or_else(|| req.get("n_predict")).and_then(Value::as_u64)
        .map_or(state.default_max_tokens, |m| m as usize);
    // `logprobs` is the number of alternatives here (chat's is a flag).
    let n_probs = req.get("logprobs").and_then(Value::as_u64).map(|n| n as usize);
    let spec = match state.spec_from(&req, false, tokens, Some(max), n_probs) { Ok(s) => s, Err(e) => return e };
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let id = completion_id("chatcmpl-");
    let created = now();
    let chunk = move |text: &str, finish: Value, logprobs: Value| json!({
        "choices": [{ "text": text, "index": 0, "logprobs": logprobs, "finish_reason": finish }],
        "created": created, "model": model, "system_fingerprint": FINGERPRINT, "object": "text_completion",
        "id": id,
    });

    if !spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
            run_generation(&st, &spec, |_, _, _, _| true).map(|(end, text)| {
                let lp = if spec.n_probs.is_some() {
                    json!({ "content": probs_list(&st, &end.probs) })
                } else {
                    Value::Null
                };
                let mut body = chunk(&text, json!(finish_reason(end.stop_type, false)), lp);
                body["usage"] = usage(&end.stats);
                body["timings"] = timings(&end.stats);
                body
            })
        }).await;
        return match out {
            Ok(Ok(body)) => Json(body).into_response(),
            Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "server_error"),
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("generation task failed: {e}"), "server_error"),
        };
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let st = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let mut sent = 0usize;
        let mut pending: Vec<TokenProbs> = Vec::new();
        let with_probs = spec.n_probs.is_some();
        let lp = |st: &AppState, p: &[TokenProbs]| if with_probs {
            json!({ "content": probs_list(st, p) })
        } else {
            Value::Null
        };
        let r = run_generation(&st, &spec, |text, hold, _, probs| {
            pending.extend_from_slice(probs);
            let release = floor_boundary(text, text.len().saturating_sub(hold));
            if release <= sent {
                return true;
            }
            let ok = sse(&tx, &chunk(&text[sent..release], Value::Null, lp(&st, &std::mem::take(&mut pending))));
            sent = release;
            ok
        });
        match r {
            Ok((end, text)) => {
                if sent < text.len() {
                    sse(&tx, &chunk(&text[sent..], Value::Null, lp(&st, &pending)));
                }
                let mut last = chunk("", json!(finish_reason(end.stop_type, false)), Value::Null);
                last["usage"] = usage(&end.stats);
                last["timings"] = timings(&end.stats);
                sse(&tx, &last);
            }
            Err(e) => {
                sse(&tx, &json!({ "error": { "code": 500, "message": e.to_string(), "type": "server_error" } }));
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });
    sse_stream(rx)
}

// --- chat: prompts and the generation all chat APIs share --------------------------

/// The chat prompt for a request, and whether it leaves a `<think>` block open.
// --- images (see media.rs) ---------------------------------------------------

/// Where a request's images are kept once encoded: the runtime's sentinel ids
/// of each, in prompt order.
const IMAGES_KEY: &str = "__airun_images";

/// Encodes every image the request carries and puts the marker text in its
/// place, so the format converters and the chat template see text; the ids
/// go under IMAGES_KEY. Image parts: OpenAI `image_url` ({url} or a string)
/// and Responses `input_image` (image_url) with a data: URL, Anthropic
/// `image` with a base64 source, and /completion's `multimodal_data` (base64
/// strings for the prompt's <__media__> markers).
async fn attach_images(state: &AppState, mut req: Value) -> Result<Value, Response> {
    let mut found: Vec<Vec<u8>> = Vec::new();
    let mut err: Option<String> = None;
    fn walk(v: &mut Value, found: &mut Vec<Vec<u8>>, err: &mut Option<String>) {
        match v {
            Value::Array(a) => a.iter_mut().for_each(|x| walk(x, found, err)),
            Value::Object(o) => {
                let ty = o.get("type").and_then(Value::as_str).unwrap_or("");
                if matches!(ty, "image_url" | "input_image" | "image") {
                    let src = match ty {
                        "image" => o.get("source").map(|s| {
                            let data = s.get("data").and_then(Value::as_str).unwrap_or("");
                            match s.get("type").and_then(Value::as_str) {
                                Some("url") => s.get("url").and_then(Value::as_str).unwrap_or("").to_string(),
                                _ => format!("data:;base64,{data}"),
                            }
                        }),
                        _ => o.get("image_url").and_then(|u| u.as_str().map(str::to_string)
                            .or_else(|| u.get("url").and_then(Value::as_str).map(str::to_string))),
                    };
                    match src.as_deref().map(decode_data_url) {
                        Some(Ok(bytes)) => found.push(bytes),
                        Some(Err(e)) => *err = Some(e),
                        None => *err = Some(format!("an {ty} part has no image data")),
                    }
                    *v = json!({ "type": "text", "text": crate::media::slot() });
                    return;
                }
                o.values_mut().for_each(|x| walk(x, found, err));
            }
            _ => {}
        }
    }
    for key in ["messages", "input"] {
        if let Some(m) = req.get_mut(key) {
            walk(m, &mut found, &mut err);
        }
    }
    if let Some(Value::Array(mm)) = req.get("multimodal_data") {
        for d in mm {
            // Bare base64, as llama-server takes it, or a whole data: URL.
            let url = |s: &str| if s.starts_with("data:") { s.to_string() } else { format!("data:;base64,{s}") };
            match d.as_str().map(|s| decode_data_url(&url(s))) {
                Some(Ok(b)) => found.push(b),
                Some(Err(e)) => err = Some(e),
                None => err = Some("multimodal_data holds base64 strings".into()),
            }
        }
    }
    if let Some(e) = err {
        return Err(api_error(StatusCode::BAD_REQUEST, e, "invalid_request_error"));
    }
    if found.is_empty() {
        return Ok(req);
    }
    let Some(vision) = &state.vision else {
        return Err(api_error(StatusCode::BAD_REQUEST,
            "this server has no vision encoder: start it with --mmproj <mmproj-*.gguf> to send images",
            "invalid_request_error"));
    };
    let mut rt = state.runtime.lock().await;
    let mut ids = Vec::new();
    for bytes in &found {
        let key = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        let known = state.image_memo.lock().unwrap_or_else(|e| e.into_inner()).iter()
            .find(|(k, ids)| *k == key && rt.has_image(ids)).map(|(_, ids)| ids.clone());
        if let Some(enc) = known {
            ids.push(json!(enc));
            continue;
        }
        let img = crate::media::decode_image(bytes)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("{e:#}"), "invalid_request_error"))?;
        let v = vision.lock().unwrap_or_else(|e| e.into_inner());
        let enc = tokio::task::block_in_place(|| crate::media::encode_image(&v, &mut rt, &img))
            .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"), "server_error"))?;
        let mut memo = state.image_memo.lock().unwrap_or_else(|e| e.into_inner());
        memo.push((key, enc.clone()));
        if memo.len() > 32 {
            memo.remove(0);
        }
        ids.push(json!(enc));
    }
    req[IMAGES_KEY] = Value::Array(ids);
    Ok(req)
}

/// The encoded images [`attach_images`] left in the request.
fn request_images(req: &Value) -> Vec<Vec<u32>> {
    req.get(IMAGES_KEY).and_then(Value::as_array).map(|a| a.iter().map(|ids| ids.as_array()
        .map(|v| v.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect()).unwrap_or_default())
        .collect()).unwrap_or_default()
}

/// Keeps the images across a format conversion.
fn carry_images(from: &Value, to: &mut Value) {
    if let Some(ids) = from.get(IMAGES_KEY) {
        to[IMAGES_KEY] = ids.clone();
    }
}

/// The bytes of a `data:[type];base64,...` URL.
fn decode_data_url(url: &str) -> Result<Vec<u8>, String> {
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(if url.starts_with("http://") || url.starts_with("https://") {
            "image URLs are not fetched by the server; send the image inline as a data: URL \
             (data:image/png;base64,...)".into()
        } else {
            "the image is not a data: URL; send data:image/png;base64,... (or jpeg/webp)".into()
        });
    };
    let Some((meta, data)) = rest.split_once(',') else { return Err("the data: URL has no ',' before its data".into()) };
    if !meta.ends_with(";base64") && !meta.is_empty() {
        return Err("the data: URL is not base64 (expected ;base64,)".into());
    }
    base64_decode(data).ok_or_else(|| "the image's base64 data is malformed".into())
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in s.as_bytes() {
        if c == b'=' {
            break;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        acc = (acc << 6) | val(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

fn chat_prompt(state: &AppState, req: &Value) -> Result<(String, bool), Response> {
    chat_prompt_ex(state, req).map(|(p, o, _)| (p, o))
}

/// [`chat_prompt`], and whether the template took the reasoning level itself.
fn chat_prompt_ex(state: &AppState, req: &Value) -> Result<(String, bool, bool), Response> {
    let Some(messages) = req.get("messages").filter(|m| m.as_array().is_some_and(|a| !a.is_empty())) else {
        return Err(api_error(StatusCode::BAD_REQUEST, "'messages' is required", "invalid_request_error"));
    };
    // Earlier assistant turns that carry their thinking inline give it to the
    // template as reasoning_content, which is how it expects to see it.
    let messages = Value::Array(messages.as_array().unwrap().iter().map(|m| {
        let mut m = m.clone();
        if m.get("role").and_then(Value::as_str) == Some("assistant") {
            if let Some(text) = m.get("content").and_then(Value::as_str).map(str::to_owned) {
                let (answer, thought) = strip_reasoning(&text);
                if let Some(t) = thought.filter(|_| m.get("reasoning_content").is_none()) {
                    m["reasoning_content"] = json!(t);
                }
                m["content"] = json!(answer);
            }
        }
        m
    }).collect());
    let mut kwargs: Map<String, Value> = req.get("chat_template_kwargs").and_then(Value::as_object).cloned()
        .unwrap_or_default();
    for k in ["enable_thinking", "reasoning_effort"] {
        if let Some(v) = req.get(k) {
            kwargs.entry(k).or_insert_with(|| v.clone());
        }
    }
    // Levels: "none" turns thinking off; the others go to a template that
    // reads reasoning_effort (Qwen3.8's knows low, medium and xhigh), and
    // become a thinking budget (in chat_job) where it does not, or where it
    // rejects the name.
    let effort = request_effort(state, req);
    kwargs.remove("reasoning_effort");
    if effort == Some(crate::reasoning::Level::Off) {
        kwargs.entry("enable_thinking").or_insert(json!(false));
    }
    if let Some(on) = state.reasoning_default {
        kwargs.entry("enable_thinking").or_insert(json!(on));
    }
    let named = req.get("reasoning_effort").and_then(Value::as_str).map(str::to_string)
        .or_else(|| state.reasoning_effort.clone())
        .filter(|_| state.template_effort && effort != Some(crate::reasoning::Level::Off));
    let mut native = false;
    let prompt = match &state.template {
        Some(t) => {
            let with_effort = named.as_ref().and_then(|e| {
                let mut k = kwargs.clone();
                k.insert("reasoning_effort".into(), json!(e));
                t.render(&messages, req.get("tools"), true, &k).ok()
            });
            match with_effort {
                Some(p) => {
                    native = true;
                    p
                }
                None => t.render(&messages, req.get("tools"), true, &kwargs)
                    .map_err(|e| api_error(StatusCode::BAD_REQUEST, e.to_string(), "invalid_request_error"))?,
            }
        }
        None => fallback_chatml(state, &messages, kwargs.get("enable_thinking").and_then(Value::as_bool)),
    };
    let opened = prompt.trim_end_matches(['\n', ' ']).ends_with("<think>");
    Ok((prompt, opened, native))
}

/// ChatML by hand, for a GGUF without a usable template.
fn fallback_chatml(state: &AppState, messages: &Value, enable_thinking: Option<bool>) -> String {
    let mut out = String::new();
    for m in messages.as_array().into_iter().flatten() {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let role = if role == "tool" { "user" } else { role };
        out.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", text_of(m.get("content"))));
    }
    out.push_str("<|im_start|>assistant\n");
    out.push_str(state.thinking.generation_prefix(enable_thinking.unwrap_or(true)));
    out
}

/// The text of a message's content: a string, or the text of its parts.
fn text_of(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// A chat request made runnable: what to generate and how to read the reply.
struct ChatJob {
    spec: GenSpec,
    /// The prompt left a `<think>` block open.
    opened: bool,
    extract: bool,
    /// Tool calls are parsed out of the reply (not with `tool_choice: "none"`).
    tools: bool,
    /// The request asked for `reasoning_control`, and the flag `/control` raises.
    control: (bool, Arc<AtomicBool>),
}

impl ChatJob {
    fn reply(&self, text: &str) -> Reply {
        if self.tools {
            parse_reply(text, self.opened, self.extract)
        } else {
            parse_reply_without_tools(text, self.opened, self.extract)
        }
    }

    fn splitter(&self) -> DeltaSplitter {
        let s = DeltaSplitter::new(self.opened, self.extract);
        if self.tools { s } else { s.without_tools() }
    }
}

/// `req`, in chat-completions form, ready to run. `n_probs`: report this many
/// alternatives per token.
async fn chat_job(state: &AppState, req: &Value, n_probs: Option<usize>) -> Result<ChatJob, Response> {
    let (prompt, opened, native_effort) = chat_prompt_ex(state, req)?;
    let tokens = {
        crate::media::tokenize(&state.tokenizer, &prompt, &request_images(req), true)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("{e:#}"), "invalid_request_error"))?
    };
    let max = req.get("max_completion_tokens").or_else(|| req.get("max_tokens")).or_else(|| req.get("n_predict"))
        .and_then(Value::as_u64).map_or(state.default_max_tokens, |m| m as usize);
    let mut spec = state.spec_from(req, false, tokens, Some(max), n_probs)?;
    // A reply that starts inside its reasoning is free until the reasoning
    // ends; the constraint is on the answer.
    if opened {
        spec.grammar = spec.grammar.take().map(|(text, g)| (text, g.after("</think>")));
    }
    let extract = spec.reasoning_format != "none";
    let tools = req.get("tool_choice").and_then(Value::as_str) != Some("none");
    // llama.cpp's reasoning budget: a token limit on the thinking, and/or a
    // way to end it from outside.
    let reasoning_control = req.get("reasoning_control").and_then(Value::as_bool).unwrap_or(false);
    let stop = Arc::new(AtomicBool::new(false));
    // An explicit budget, else the level's (when the template does not read
    // levels itself), else --reasoning-budget.
    let level_budget = match request_effort(state, req) {
        Some(crate::reasoning::Level::Budget(b)) if !native_effort => Some(b),
        _ => None,
    };
    let budget = match req.get("reasoning_budget_tokens").or_else(|| req.get("thinking_budget_tokens"))
        .and_then(Value::as_i64) {
        Some(b) => (b >= 0).then_some(b as usize),
        None => level_budget.unwrap_or(state.reasoning_budget),
    };
    if let Some((start, end)) = &state.think_tags {
        if budget.is_some() || reasoning_control {
            let message = req.get("reasoning_budget_message").and_then(Value::as_str)
                .unwrap_or(&state.reasoning_budget_message);
            let mut forced = if message.is_empty() { Vec::new() } else { state.tokenizer.encode(message, false) };
            forced.extend_from_slice(end);
            spec.reasoning = Some(airun_core::ReasoningBudget {
                start: start.clone(), end: end.clone(), forced, budget, stop: Arc::clone(&stop), opened,
            });
        }
    }
    Ok(ChatJob { spec, opened, extract, tools, control: (reasoning_control, stop) })
}

/// The reasoning level a request asks for (its `reasoning_effort`, else
/// --reasoning-effort); an unknown name counts as none given.
fn request_effort(state: &AppState, req: &Value) -> Option<crate::reasoning::Level> {
    req.get("reasoning_effort").and_then(Value::as_str).map(str::to_string)
        .or_else(|| state.reasoning_effort.clone())
        .and_then(|e| crate::reasoning::parse(&e))
}

/// Keeps a completion addressable by `/control` while it runs.
struct Active {
    state: Arc<AppState>,
    id: String,
}

impl Active {
    fn register(state: &Arc<AppState>, id: &str, job: &ChatJob) -> Self {
        state.active.lock().unwrap().insert(id.to_string(), (job.control.0, Arc::clone(&job.control.1)));
        Self { state: Arc::clone(state), id: id.to_string() }
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        self.state.active.lock().unwrap().remove(&self.id);
    }
}

/// Runs a chat job, handing `emit` each reasoning or content delta with the
/// probabilities of the tokens behind it; returning false stops generation.
/// Text from a tool call on is held back: it comes out parsed, at the end.
fn stream_chat(state: &AppState, job: &ChatJob, mut emit: impl FnMut(Delta, Vec<TokenProbs>) -> bool)
    -> Result<(GenEnd, String)> {
    let mut splitter = job.splitter();
    let mut pending: Vec<TokenProbs> = Vec::new();
    let r = run_generation(state, &job.spec, |text, hold, _, probs| {
        pending.extend_from_slice(probs);
        for d in splitter.feed(text, hold) {
            if !emit(d, std::mem::take(&mut pending)) {
                return false; // the client hung up
            }
        }
        true
    });
    if let Ok((_, text)) = &r {
        for d in splitter.feed(text, 0) {
            emit(d, std::mem::take(&mut pending));
        }
    }
    r
}

/// The tokens of a chat prompt, for the `input_tokens` endpoints.
async fn count_chat_tokens(state: &AppState, req: &Value) -> Result<usize, Response> {
    let (prompt, _) = chat_prompt(state, req)?;
    Ok(state.tokenizer.encode(&prompt, true).len())
}

fn generation_failed(e: impl std::fmt::Display) -> Response {
    api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("generation failed: {e}"), "server_error")
}

// --- OpenAI /v1/chat/completions ------------------------------------------------

async fn chat_completions(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let req = match attach_images(&state, req).await { Ok(v) => v, Err(e) => return e };
    let n_probs = req.get("logprobs").and_then(Value::as_bool).unwrap_or(false)
        .then(|| req.get("top_logprobs").and_then(Value::as_u64).unwrap_or(0) as usize);
    let mut job = match chat_job(&state, &req, n_probs).await { Ok(j) => j, Err(e) => return e };
    let include_usage = req.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
    let return_progress = req.get("return_progress").and_then(Value::as_bool).unwrap_or(false);
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let id = completion_id("chatcmpl-");
    let active = Active::register(&state, &id, &job);
    let created = now();
    let envelope = move |choices: Value| json!({
        "choices": choices, "created": created, "id": id, "model": model,
        "system_fingerprint": FINGERPRINT, "object": "chat.completion.chunk",
    });

    if !job.spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
            let _active = active;
            run_generation(&st, &job.spec, |_, _, _, _| true).map(|(end, text)| {
                let reply = job.reply(&text);
                let mut message = json!({ "role": "assistant", "content": reply.content });
                if let Some(r) = &reply.reasoning {
                    message["reasoning_content"] = json!(r);
                }
                if !reply.tool_calls.is_empty() {
                    message["tool_calls"] = Value::Array(reply.tool_calls.iter().enumerate().map(|(i, c)| {
                        let mut j = c.to_json(i, &completion_id(""));
                        j.as_object_mut().unwrap().remove("index");
                        j
                    }).collect());
                }
                let mut choice = json!({
                    "finish_reason": finish_reason(end.stop_type, !reply.tool_calls.is_empty()),
                    "index": 0, "message": message,
                });
                if job.spec.n_probs.is_some() {
                    choice["logprobs"] = json!({ "content": probs_list(&st, &end.probs) });
                }
                let mut body = envelope(json!([choice]));
                body["object"] = json!("chat.completion");
                body["usage"] = usage(&end.stats);
                body["timings"] = timings(&end.stats);
                body
            })
        }).await;
        return match out {
            Ok(Ok(body)) => Json(body).into_response(),
            Ok(Err(e)) => generation_failed(e),
            Err(e) => generation_failed(e),
        };
    }

    let (tx, rx) = mpsc::unbounded_channel();
    // llama-server's return_progress: a chunk with `prompt_progress` (and no
    // choices) after each prefill batch. Besides showing progress, it keeps a
    // proxy or tunnel from closing a connection that would otherwise stay
    // silent through a long prefill.
    if return_progress {
        let ptx = tx.clone();
        let envelope = envelope.clone();
        job.spec.progress = Some(Arc::new(move |p| {
            let mut chunk = envelope(json!([]));
            chunk["prompt_progress"] = prompt_progress_json(p);
            sse(&ptx, &chunk);
        }));
    }
    let st = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let _active = active;
        let with_probs = job.spec.n_probs.is_some();
        let delta = |d: Value, probs: &[TokenProbs]| {
            let mut c = json!({ "finish_reason": null, "index": 0, "delta": d });
            if with_probs && !probs.is_empty() {
                c["logprobs"] = json!({ "content": probs_list(&st, probs) });
            }
            envelope(json!([c]))
        };
        sse(&tx, &delta(json!({ "role": "assistant", "content": null }), &[]));
        let r = stream_chat(&st, &job, |d, probs| match d {
            Delta::Reasoning(s) => sse(&tx, &delta(json!({ "reasoning_content": s }), &probs)),
            Delta::Content(s) => sse(&tx, &delta(json!({ "content": s }), &probs)),
        });
        match r {
            Ok((end, text)) => {
                let reply = job.reply(&text);
                if !reply.tool_calls.is_empty() {
                    let calls: Vec<Value> = reply.tool_calls.iter().enumerate()
                        .map(|(i, c)| c.to_json(i, &completion_id(""))).collect();
                    sse(&tx, &delta(json!({ "tool_calls": calls }), &[]));
                }
                let finish = finish_reason(end.stop_type, !reply.tool_calls.is_empty());
                let mut last = envelope(json!([{ "finish_reason": finish, "index": 0, "delta": {} }]));
                if include_usage {
                    sse(&tx, &last);
                    last = envelope(json!([]));
                    last["usage"] = usage(&end.stats);
                }
                last["timings"] = timings(&end.stats);
                sse(&tx, &last);
            }
            Err(e) => {
                sse(&tx, &json!({ "error": { "code": 500, "message": e.to_string(), "type": "server_error" } }));
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });
    sse_stream(rx)
}

async fn chat_input_tokens(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    match count_chat_tokens(&state, &req).await {
        Ok(n) => Json(json!({ "input_tokens": n, "object": "response.input_tokens" })).into_response(),
        Err(e) => e,
    }
}

// --- OpenAI /v1/responses -----------------------------------------------------------

/// A Responses API request in chat-completions form. Fields both APIs share
/// (sampling, `stream`, llama.cpp's extras) pass through unchanged.
fn responses_to_chat(req: &Value) -> Result<Value, Response> {
    let mut messages = Vec::new();
    if let Some(i) = req.get("instructions").and_then(Value::as_str) {
        messages.push(json!({ "role": "system", "content": i }));
    }
    match req.get("input") {
        Some(Value::String(s)) => messages.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => {
            // Reasoning items precede the assistant turn they belong to.
            let mut reasoning: Option<String> = None;
            for it in items {
                let field = |k: &str| it.get(k).and_then(Value::as_str).unwrap_or_default();
                match it.get("type").and_then(Value::as_str).unwrap_or("message") {
                    "message" => {
                        let role = match field("role") { "" => "user", "developer" => "system", r => r };
                        let mut m = json!({ "role": role, "content": text_of(it.get("content")) });
                        if role == "assistant" {
                            if let Some(r) = reasoning.take() {
                                m["reasoning_content"] = json!(r);
                            }
                        }
                        messages.push(m);
                    }
                    "reasoning" => {
                        let text = text_of(it.get("content"));
                        let text = if text.is_empty() { text_of(it.get("summary")) } else { text };
                        reasoning = Some(text);
                    }
                    "function_call" => {
                        let call = json!({ "id": field("call_id"), "type": "function",
                            "function": { "name": field("name"), "arguments": field("arguments") } });
                        // Calls of one turn belong to one assistant message.
                        match messages.last_mut() {
                            Some(m) if m["role"] == "assistant" && m.get("tool_calls").is_some() => {
                                m["tool_calls"].as_array_mut().unwrap().push(call);
                            }
                            _ => {
                                let mut m = json!({ "role": "assistant", "content": "", "tool_calls": [call] });
                                if let Some(r) = reasoning.take() {
                                    m["reasoning_content"] = json!(r);
                                }
                                messages.push(m);
                            }
                        }
                    }
                    "function_call_output" => {
                        let output = match it.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v @ Value::Array(_)) => text_of(Some(v)),
                            Some(v) => v.to_string(),
                            None => String::new(),
                        };
                        messages.push(json!({ "role": "tool", "tool_call_id": field("call_id"), "content": output }));
                    }
                    _ => {}
                }
            }
        }
        _ => return Err(api_error(StatusCode::BAD_REQUEST, "'input' is required", "invalid_request_error")),
    }
    let mut chat = req.as_object().cloned().unwrap_or_default();
    for k in ["input", "instructions", "tools", "text", "reasoning", "max_output_tokens"] {
        chat.remove(k);
    }
    chat.insert("messages".into(), Value::Array(messages));
    if let Some(m) = req.get("max_output_tokens") {
        chat.insert("max_tokens".into(), m.clone());
    }
    if let Some(tools) = req.get("tools").and_then(Value::as_array) {
        let fns: Vec<Value> = tools.iter().filter(|t| t.get("type").and_then(Value::as_str) == Some("function"))
            .map(|t| json!({ "type": "function", "function": {
                "name": t.get("name"), "description": t.get("description"), "parameters": t.get("parameters"),
            } })).collect();
        if !fns.is_empty() {
            chat.insert("tools".into(), Value::Array(fns));
        }
    }
    if let Some(fmt) = req.pointer("/text/format") {
        match fmt.get("type").and_then(Value::as_str) {
            Some("json_schema") => {
                let schema = fmt.get("schema").cloned().unwrap_or(json!({}));
                chat.insert("response_format".into(), json!({ "type": "json_schema", "json_schema": { "schema": schema } }));
            }
            Some("json_object") => {
                chat.insert("response_format".into(), json!({ "type": "json_object" }));
            }
            _ => {}
        }
    }
    if let Some(effort) = req.pointer("/reasoning/effort") {
        chat.insert("reasoning_effort".into(), effort.clone());
    }
    Ok(Value::Object(chat))
}

/// A Responses API output item for a reasoning block, a message or a call.
fn reasoning_item(id: &str, text: Option<&str>, status: Option<&str>) -> Value {
    let mut v = json!({
        "id": id, "summary": [], "type": "reasoning",
        "content": text.map_or(json!([]), |t| json!([{ "text": t, "type": "reasoning_text" }])),
        "encrypted_content": "",
    });
    if let Some(s) = status {
        v["status"] = json!(s);
    }
    v
}

fn output_text(text: &str) -> Value {
    json!({ "type": "output_text", "annotations": [], "logprobs": [], "text": text })
}

fn message_item(id: &str, text: Option<&str>, status: &str) -> Value {
    json!({
        "type": "message", "status": status, "id": id, "role": "assistant",
        "content": text.map_or(json!([]), |t| json!([output_text(t)])),
    })
}

fn function_call_item(id: &str, call: &ToolCall, status: &str) -> Value {
    json!({
        "id": format!("fc_{id}"), "type": "function_call", "status": status,
        "arguments": call.arguments.to_string(), "call_id": format!("call_{id}"), "name": call.name,
    })
}

fn response_usage(s: &airun_core::GenerationStats) -> Value {
    json!({
        "input_tokens": s.prompt_tokens, "output_tokens": s.generated_tokens,
        "total_tokens": s.prompt_tokens + s.generated_tokens,
        "input_tokens_details": { "cached_tokens": s.cached_tokens },
        "output_tokens_details": { "reasoning_tokens": 0 },
    })
}

/// The finished response object; `end` is None while it is in progress.
fn response_object(id: &str, model: &str, created: u64, output: Vec<Value>, end: Option<&GenEnd>) -> Value {
    let mut v = json!({
        "id": id, "object": "response", "created_at": created, "model": model, "output": output,
        "status": "in_progress",
    });
    if let Some(end) = end {
        if end.stop_type == "limit" {
            v["status"] = json!("incomplete");
            v["incomplete_details"] = json!({ "reason": "max_output_tokens" });
        } else {
            v["status"] = json!("completed");
        }
        v["completed_at"] = json!(now());
        v["usage"] = response_usage(&end.stats);
    }
    v
}

async fn responses(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let req = match attach_images(&state, req).await { Ok(v) => v, Err(e) => return e };
    let mut chat = match responses_to_chat(&req) { Ok(c) => c, Err(e) => return e };
    carry_images(&req, &mut chat);
    let job = match chat_job(&state, &chat, None).await { Ok(j) => j, Err(e) => return e };
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let id = completion_id("resp_");
    let active = Active::register(&state, &id, &job);
    let (rs_id, msg_id) = (completion_id("rs_"), completion_id("msg_"));
    let created = now();

    if !job.spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
            let _active = active;
            run_generation(&st, &job.spec, |_, _, _, _| true).map(|(end, text)| {
                let reply = job.reply(&text);
                let mut output = Vec::new();
                if let Some(r) = &reply.reasoning {
                    output.push(reasoning_item(&rs_id, Some(r), Some("completed")));
                }
                if !reply.content.is_empty() || reply.tool_calls.is_empty() {
                    output.push(message_item(&msg_id, Some(&reply.content), "completed"));
                }
                for c in &reply.tool_calls {
                    output.push(function_call_item(&completion_id(""), c, "completed"));
                }
                let mut body = response_object(&id, &model, created, output, Some(&end));
                body["timings"] = timings(&end.stats);
                body
            })
        }).await;
        return match out {
            Ok(Ok(body)) => Json(body).into_response(),
            Ok(Err(e)) => generation_failed(e),
            Err(e) => generation_failed(e),
        };
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let st = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let _active = active;
        let mut seq = 0u64;
        // Every event names its type twice (SSE `event:` and `type`) and is
        // numbered, as OpenAI's are.
        let mut send = |name: &str, mut data: Value| {
            data["type"] = json!(name);
            data["sequence_number"] = json!(seq);
            seq += 1;
            tx.send(Ok(Event::default().event(name).data(data.to_string()))).is_ok()
        };
        let pending = response_object(&id, &model, created, Vec::new(), None);
        send("response.created", json!({ "response": pending }));
        send("response.in_progress", json!({ "response": pending }));
        // Output items in the order they opened: (kind, index).
        let mut rs_index: Option<usize> = None;
        let mut msg_index: Option<usize> = None;
        let mut items = 0usize;
        let r = stream_chat(&st, &job, |d, _| match d {
            Delta::Reasoning(s) => {
                let i = *rs_index.get_or_insert_with(|| {
                    items += 1;
                    send("response.output_item.added", json!({
                        "output_index": items - 1, "item": reasoning_item(&rs_id, None, Some("in_progress")) }));
                    items - 1
                });
                send("response.reasoning_text.delta",
                    json!({ "item_id": rs_id, "output_index": i, "content_index": 0, "delta": s }))
            }
            Delta::Content(s) => {
                let i = *msg_index.get_or_insert_with(|| {
                    items += 1;
                    send("response.output_item.added", json!({
                        "output_index": items - 1, "item": message_item(&msg_id, None, "in_progress") }));
                    send("response.content_part.added", json!({
                        "item_id": msg_id, "output_index": items - 1, "content_index": 0,
                        "part": { "type": "output_text", "text": "" } }));
                    items - 1
                });
                send("response.output_text.delta",
                    json!({ "item_id": msg_id, "output_index": i, "content_index": 0, "delta": s }))
            }
        });
        let (end, text) = match r {
            Ok(x) => x,
            Err(e) => {
                send("error", json!({ "code": "server_error", "message": e.to_string(), "param": null }));
                return;
            }
        };
        let reply = job.reply(&text);
        let mut output = Vec::new();
        if let Some(i) = rs_index {
            let item = reasoning_item(&rs_id, reply.reasoning.as_deref(), Some("completed"));
            send("response.reasoning_text.done", json!({
                "item_id": rs_id, "output_index": i, "content_index": 0,
                "text": reply.reasoning.as_deref().unwrap_or_default() }));
            send("response.output_item.done", json!({ "output_index": i, "item": item }));
            output.push(item);
        }
        if let Some(i) = msg_index {
            let item = message_item(&msg_id, Some(&reply.content), "completed");
            send("response.output_text.done",
                json!({ "item_id": msg_id, "output_index": i, "content_index": 0, "text": reply.content }));
            send("response.content_part.done", json!({
                "item_id": msg_id, "output_index": i, "content_index": 0, "part": output_text(&reply.content) }));
            send("response.output_item.done", json!({ "output_index": i, "item": item }));
            output.push(item);
        }
        for c in &reply.tool_calls {
            let cid = completion_id("");
            let i = items;
            items += 1;
            let item_id = format!("fc_{cid}");
            let mut added = function_call_item(&cid, c, "in_progress");
            added["arguments"] = json!("");
            send("response.output_item.added", json!({ "output_index": i, "item": added }));
            let args = c.arguments.to_string();
            send("response.function_call_arguments.delta", json!({ "item_id": item_id, "output_index": i, "delta": args }));
            send("response.function_call_arguments.done",
                json!({ "item_id": item_id, "output_index": i, "arguments": args }));
            let item = function_call_item(&cid, c, "completed");
            send("response.output_item.done", json!({ "output_index": i, "item": item }));
            output.push(item);
        }
        let mut done = json!({ "response": response_object(&id, &model, created, output, Some(&end)) });
        done["timings"] = timings(&end.stats);
        let name = if end.stop_type == "limit" { "response.incomplete" } else { "response.completed" };
        send(name, done);
    });
    sse_stream(rx)
}

async fn responses_input_tokens(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let req = match attach_images(&state, req).await { Ok(v) => v, Err(e) => return e };
    let mut chat = match responses_to_chat(&req) { Ok(c) => c, Err(e) => return e };
    carry_images(&req, &mut chat);
    match count_chat_tokens(&state, &chat).await {
        Ok(n) => Json(json!({ "input_tokens": n, "object": "response.input_tokens" })).into_response(),
        Err(e) => e,
    }
}

// --- Anthropic /v1/messages -------------------------------------------------------

/// An Anthropic Messages request in chat-completions form.
fn anthropic_to_chat(req: &Value) -> Result<Value, Response> {
    let mut messages = Vec::new();
    match req.get("system") {
        Some(Value::String(s)) => messages.push(json!({ "role": "system", "content": s })),
        Some(v @ Value::Array(_)) => messages.push(json!({ "role": "system", "content": text_of(Some(v)) })),
        _ => {}
    }
    let Some(turns) = req.get("messages").and_then(Value::as_array).filter(|a| !a.is_empty()) else {
        return Err(api_error(StatusCode::BAD_REQUEST, "'messages' is required", "invalid_request_error"));
    };
    for m in turns {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let blocks = match m.get("content") {
            Some(Value::String(s)) => {
                messages.push(json!({ "role": role, "content": s }));
                continue;
            }
            Some(Value::Array(b)) => b.as_slice(),
            _ => &[],
        };
        let mut text = Vec::new();
        let mut thinking = Vec::new();
        let mut calls = Vec::new();
        for b in blocks {
            let field = |k: &str| b.get(k).and_then(Value::as_str).unwrap_or_default();
            match field("type") {
                "text" => text.push(field("text").to_string()),
                "thinking" => thinking.push(field("thinking").to_string()),
                "tool_use" => calls.push(json!({ "id": field("id"), "type": "function", "function": {
                    "name": field("name"), "arguments": b.get("input").cloned().unwrap_or(json!({})).to_string() } })),
                // Results answer the previous turn's calls, ahead of any text.
                "tool_result" => messages.push(json!({ "role": "tool", "tool_call_id": field("tool_use_id"),
                    "content": text_of(b.get("content")) })),
                _ => {}
            }
        }
        if role == "assistant" {
            let mut a = json!({ "role": "assistant", "content": text.join("\n") });
            if !thinking.is_empty() {
                a["reasoning_content"] = json!(thinking.join("\n"));
            }
            if !calls.is_empty() {
                a["tool_calls"] = Value::Array(calls);
            }
            messages.push(a);
        } else if !text.is_empty() {
            messages.push(json!({ "role": role, "content": text.join("\n") }));
        }
    }
    let mut chat = req.as_object().cloned().unwrap_or_default();
    for k in ["system", "tools", "tool_choice", "stop_sequences", "thinking", "metadata"] {
        chat.remove(k);
    }
    chat.insert("messages".into(), Value::Array(messages));
    if let Some(tools) = req.get("tools").and_then(Value::as_array) {
        chat.insert("tools".into(), tools.iter().map(|t| json!({ "type": "function", "function": {
            "name": t.get("name"), "description": t.get("description"), "parameters": t.get("input_schema"),
        } })).collect());
    }
    if let Some(tc) = req.pointer("/tool_choice/type").and_then(Value::as_str) {
        chat.insert("tool_choice".into(), json!(match tc { "none" => "none", "any" | "tool" => "required", _ => "auto" }));
    }
    if let Some(stops) = req.get("stop_sequences") {
        chat.insert("stop".into(), stops.clone());
    }
    if let Some(t) = req.pointer("/thinking/type").and_then(Value::as_str) {
        let kwargs = chat.entry("chat_template_kwargs").or_insert_with(|| json!({}));
        if let Some(k) = kwargs.as_object_mut() {
            k.insert("enable_thinking".into(), json!(t != "disabled"));
        }
        // Anthropic's thinking budget is llama-server's thinking_budget_tokens.
        if t == "enabled" {
            if let Some(b) = req.pointer("/thinking/budget_tokens").and_then(Value::as_i64) {
                chat.insert("thinking_budget_tokens".into(), json!(b));
            }
        }
    }
    Ok(Value::Object(chat))
}

/// Anthropic's stop reason for how a generation ended.
fn anthropic_stop(end: &GenEnd, tool_calls: bool) -> (&'static str, Value) {
    match end.stop_type {
        "word" => ("stop_sequence", json!(end.stopping_word)),
        "limit" => ("max_tokens", Value::Null),
        _ if tool_calls => ("tool_use", Value::Null),
        _ => ("end_turn", Value::Null),
    }
}

fn anthropic_usage(prompt: usize, cached: usize, output: usize) -> Value {
    json!({ "cache_read_input_tokens": cached, "input_tokens": prompt - cached.min(prompt), "output_tokens": output })
}

async fn anthropic_messages(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let req = match attach_images(&state, req).await { Ok(v) => v, Err(e) => return e };
    let mut chat = match anthropic_to_chat(&req) { Ok(c) => c, Err(e) => return e };
    carry_images(&req, &mut chat);
    let job = match chat_job(&state, &chat, None).await { Ok(j) => j, Err(e) => return e };
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let id = completion_id("msg_");
    let active = Active::register(&state, &id, &job);

    if !job.spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
            let _active = active;
            run_generation(&st, &job.spec, |_, _, _, _| true).map(|(end, text)| {
                let reply = job.reply(&text);
                let mut content = Vec::new();
                if let Some(r) = &reply.reasoning {
                    content.push(json!({ "type": "thinking", "thinking": r, "signature": "" }));
                }
                if !reply.content.is_empty() {
                    content.push(json!({ "type": "text", "text": reply.content }));
                }
                for c in &reply.tool_calls {
                    content.push(json!({ "type": "tool_use", "id": completion_id("toolu_"), "name": c.name,
                        "input": c.arguments }));
                }
                let (reason, seq) = anthropic_stop(&end, !reply.tool_calls.is_empty());
                json!({
                    "id": id, "type": "message", "role": "assistant", "content": content, "model": model,
                    "stop_reason": reason, "stop_sequence": seq,
                    "usage": anthropic_usage(end.stats.prompt_tokens, end.stats.cached_tokens,
                        end.stats.generated_tokens),
                })
            })
        }).await;
        return match out {
            Ok(Ok(body)) => Json(body).into_response(),
            Ok(Err(e)) => generation_failed(e),
            Err(e) => generation_failed(e),
        };
    }

    let cached = state.runtime.lock().await.reusable_prefix(&job.spec.tokens);
    let (tx, rx) = mpsc::unbounded_channel();
    let st = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let _active = active;
        let send = |name: &str, mut data: Value| {
            data["type"] = json!(name);
            tx.send(Ok(Event::default().event(name).data(data.to_string()))).is_ok()
        };
        send("message_start", json!({ "message": {
            "id": id, "type": "message", "role": "assistant", "content": [], "model": model,
            "stop_reason": null, "stop_sequence": null,
            "usage": anthropic_usage(job.spec.tokens.len(), cached, 0),
        } }));
        // The open content block: its index and whether it is thinking.
        let mut open: Option<(usize, bool)> = None;
        let mut next = 0usize;
        // Starts a block of this kind unless one is open, closing the other kind.
        let mut block = |thinking: bool, open: &mut Option<(usize, bool)>| -> usize {
            if let Some((i, t)) = *open {
                if t == thinking {
                    return i;
                }
                if t {
                    send("content_block_delta",
                        json!({ "index": i, "delta": { "type": "signature_delta", "signature": "" } }));
                }
                send("content_block_stop", json!({ "index": i }));
            }
            let i = next;
            next += 1;
            let cb = if thinking { json!({ "type": "thinking", "thinking": "" }) } else { json!({ "type": "text", "text": "" }) };
            send("content_block_start", json!({ "index": i, "content_block": cb }));
            *open = Some((i, thinking));
            i
        };
        let r = stream_chat(&st, &job, |d, _| match d {
            Delta::Reasoning(s) => {
                let i = block(true, &mut open);
                send("content_block_delta", json!({ "index": i, "delta": { "type": "thinking_delta", "thinking": s } }))
            }
            Delta::Content(s) => {
                let i = block(false, &mut open);
                send("content_block_delta", json!({ "index": i, "delta": { "type": "text_delta", "text": s } }))
            }
        });
        let (end, text) = match r {
            Ok(x) => x,
            Err(e) => {
                send("error", json!({ "error": { "type": "api_error", "message": e.to_string() } }));
                return;
            }
        };
        if let Some((i, t)) = open {
            if t {
                send("content_block_delta", json!({ "index": i, "delta": { "type": "signature_delta", "signature": "" } }));
            }
            send("content_block_stop", json!({ "index": i }));
        }
        let reply = job.reply(&text);
        for c in &reply.tool_calls {
            let i = next;
            next += 1;
            send("content_block_start", json!({ "index": i, "content_block": {
                "type": "tool_use", "id": completion_id("toolu_"), "name": c.name, "input": {} } }));
            send("content_block_delta", json!({ "index": i, "delta": {
                "type": "input_json_delta", "partial_json": c.arguments.to_string() } }));
            send("content_block_stop", json!({ "index": i }));
        }
        let (reason, seq) = anthropic_stop(&end, !reply.tool_calls.is_empty());
        send("message_delta", json!({ "delta": { "stop_reason": reason, "stop_sequence": seq },
            "usage": { "output_tokens": end.stats.generated_tokens } }));
        send("message_stop", json!({}));
    });
    sse_stream(rx)
}

async fn anthropic_count_tokens(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let req = match attach_images(&state, req).await { Ok(v) => v, Err(e) => return e };
    let mut chat = match anthropic_to_chat(&req) { Ok(c) => c, Err(e) => return e };
    carry_images(&req, &mut chat);
    match count_chat_tokens(&state, &chat).await {
        Ok(n) => Json(json!({ "input_tokens": n })).into_response(),
        Err(e) => e,
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this guards: a stop sequence that arrives split across two
    /// tokens used to be streamed to the client before it was recognised.
    #[test]
    fn a_stop_sequence_split_across_tokens_is_held_back() {
        let stops = vec!["<|end|>".to_string()];
        assert_eq!(ambiguous_tail("hello <|en", &stops), 4);
        assert_eq!(ambiguous_tail("hello <", &stops), 1);
        assert_eq!(ambiguous_tail("hello there", &stops), 0);
        assert_eq!(ambiguous_tail("a < b", &stops), 0);
    }

    #[test]
    fn the_longest_candidate_stop_wins_the_holdback() {
        let stops = vec!["END".to_string(), "ENDING".to_string()];
        assert_eq!(ambiguous_tail("text EN", &stops), 2);
        assert_eq!(ambiguous_tail("text ENDIN", &stops), 5);
    }

    #[test]
    fn holding_back_never_splits_a_character() {
        let stops = vec!["â€¦done".to_string()];
        let text = "wait â€¦do";
        let hold = ambiguous_tail(text, &stops);
        assert!(text.is_char_boundary(text.len() - hold));
        assert_eq!(&text[text.len() - hold..], "â€¦do");
    }

    #[test]
    fn the_earliest_stop_sequence_wins() {
        let stops = vec!["World".to_string(), "Hello".to_string()];
        assert_eq!(stop_at("say Hello World", &stops), Some((4, "Hello")));
        assert_eq!(stop_at("say nothing", &stops), None);
        // An empty stop string would match at 0 and truncate everything.
        assert_eq!(stop_at("anything", &[String::new()]), None);
    }

    #[test]
    fn logit_bias_parses_both_shapes_and_skips_junk() {
        let v = json!({ "42": -1.5, "7": 100, "not-a-token": 1.0, "9": "x" });
        let mut bias = logit_bias_from(Some(&v));
        bias.sort_by_key(|b| b.0);
        assert_eq!(bias, vec![(7, 100.0), (42, -1.5)]);
        let llama = json!([[15, -2.0], [16, false], ["x", 1]]);
        assert_eq!(logit_bias_from(Some(&llama)), vec![(15, -2.0), (16, f32::NEG_INFINITY)]);
        assert!(logit_bias_from(None).is_empty());
    }

    #[test]
    fn earlier_reasoning_moves_out_of_the_transcript() {
        assert_eq!(strip_reasoning("<think>\nhmm\n</think>\n\nParis."), ("Paris.", Some("hmm")));
        // Qwen3.5+ opens the block in the prompt, so replies lack <think>.
        assert_eq!(strip_reasoning("hmm\n</think>\n\nParis."), ("Paris.", Some("hmm")));
        assert_eq!(strip_reasoning("Paris."), ("Paris.", None));
        assert_eq!(strip_reasoning("<think>\nstill going"), ("", None));
    }

    #[test]
    fn thinking_prefix_follows_the_template() {
        let opens = Thinking::from_template(
            "{%- if enable_thinking is defined and enable_thinking is false %}\
             {{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}{%- endif %}");
        assert_eq!(opens.generation_prefix(true), "<think>\n");
        assert_eq!(opens.generation_prefix(false), "<think>\n\n</think>\n\n");
        let plain = Thinking::from_template("{{ messages }}");
        assert_eq!(plain.generation_prefix(false), "");
    }

    /// A stream that ended on the model's own end-of-generation token used to
    /// report "length", which tells a client its answer was truncated.
    #[test]
    fn stop_types_map_to_openai_finish_reasons() {
        use airun_core::StopReason;
        assert_eq!(stop_type(StopReason::EndOfGeneration, false), "eos");
        assert_eq!(stop_type(StopReason::Cancelled, true), "word");
        assert_eq!(stop_type(StopReason::MaxTokens, false), "limit");
        assert_eq!(finish_reason("eos", false), "stop");
        assert_eq!(finish_reason("eos", true), "tool_calls");
        assert_eq!(finish_reason("limit", false), "length");
    }

    #[test]
    fn completion_ids_look_like_llama_servers() {
        let a = completion_id("chatcmpl-");
        assert!(a.starts_with("chatcmpl-") && a.len() == 9 + 32);
        assert_ne!(a, completion_id("chatcmpl-"));
    }
}
