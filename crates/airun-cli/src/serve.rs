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
use axum::extract::State;
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

    /// Name reported by /v1/models and echoed in responses.
    #[arg(long, alias = "alias")]
    pub served_model_name: Option<String>,

    /// Context length in tokens.
    #[arg(short = 'c', long, default_value_t = 4096)]
    pub ctx_size: usize,

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

    /// CUDA ordinals to use.
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<i32>,

    /// Layer split proportions per GPU.
    #[arg(long, value_delimiter = ',')]
    pub tensor_split: Vec<f32>,

    /// Keep the whole model on one GPU, by ordinal.
    #[arg(long)]
    pub single_gpu: Option<i32>,

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
    runtime: Mutex<Runtime>,
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
    /// Each token's bytes as a client sees them (special tokens empty), for
    /// grammars, probabilities and DRY's sequence breakers.
    pieces: Arc<Vec<Vec<u8>>>,
    eog: Arc<Vec<u32>>,
    /// The tokens llama.cpp's default DRY sequence breakers mark.
    default_breakers: Arc<Vec<bool>>,
    /// Fill-in-the-middle markers, if the vocabulary has them.
    fim: Option<Fim>,
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
struct Thinking {
    /// The template knows `<think>` blocks, so `enable_thinking` means something.
    supported: bool,
    /// The generation prompt already opens the block (Qwen3.5 and later), so the
    /// model never emits `<think>` itself.
    prompt_opens_block: bool,
}

impl Thinking {
    fn from_template(template: &str) -> Self {
        Self {
            supported: template.contains("<think>"),
            prompt_opens_block: template.contains("'<think>\\n'"),
        }
    }

    /// Text appended after `<|im_start|>assistant\n`.
    fn generation_prefix(self, enabled: bool) -> &'static str {
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
    run_args.vram_reserve = args.vram_reserve;

    eprintln!("loading {}", args.model.display());
    let (runtime, report) = loader::load(&args.model, &run_args.load_options(), |stage, d, t| {
        if t > 0 && d % 64 == 0 {
            eprint!("\r{stage}: {d}/{t}   ");
        }
    })
    .with_context(|| format!("failed to load {}", args.model.display()))?;
    eprintln!("\r{}", report.config.summary());
    eprint!("{}", report.placement.report());

    let model_name = args.served_model_name.clone().unwrap_or_else(|| {
        args.model.file_stem().and_then(|s| s.to_str()).unwrap_or("airun").to_string()
    });
    let ctx = runtime.ctx();
    let cfg = &report.config;

    let gguf = airun_gguf::GgufFile::open(&args.model).ok();
    let meta_str = |k: &str| gguf.as_ref().and_then(|g| g.get(k)).and_then(|v| v.as_str()).map(str::to_owned);
    let template_src = meta_str("tokenizer.chat_template").unwrap_or_default();
    let token_text = |id: Option<u32>| id.and_then(|i| runtime.tokenizer.token_text(i)).unwrap_or("").to_string();
    let bos = token_text(runtime.tokenizer.bos);
    let eos = token_text(runtime.tokenizer.eos);
    let thinking = Thinking::from_template(&template_src);
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
    let n_params: u64 = gguf.as_ref().map_or(0, |g| g.tensors().iter().map(|t| t.n_elements()).sum());
    let size = gguf.as_ref().map_or(0, |g| g.file_len());
    let ftype = gguf.as_ref().and_then(|g| g.type_histogram().into_iter().max_by_key(|(_, (_, b))| *b))
        .map(|(t, _)| t).unwrap_or_default();
    let model_path = args.model.display().to_string();

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
        "modalities": { "vision": false, "video": false, "audio": false },
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
            "architecture": { "input_modalities": ["text"], "output_modalities": ["text"] },
            "created": created,
            "owned_by": "airun",
            "meta": {
                "vocab_type": 2, "n_vocab": cfg.vocab_size, "n_ctx": ctx, "n_ctx_train": cfg.train_ctx,
                "n_embd": cfg.n_embd, "n_params": n_params, "size": size, "ftype": ftype,
            },
        }],
    });

    let state = Arc::new(AppState {
        runtime: Mutex::new(runtime),
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
        .route("/embedding", post(no_embeddings))
        .route("/embeddings", post(no_embeddings))
        .route("/v1/embeddings", post(no_embeddings))
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
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let auth = if args.api_key.is_some() { "enabled" } else { "disabled (no --api-key)" };

    // The runtime is created here rather than via #[tokio::main] so that model
    // loading happens on the main thread, before any async machinery starts.
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("could not bind {addr}"))?;
        eprintln!("\nairun serving {model_name} on http://{addr}");
        eprintln!("  web ui:         http://{addr}/");
        eprintln!("  authentication: {auth}");
        eprintln!("  endpoints: /v1/chat/completions /v1/responses /v1/messages /v1/completions /completion /infill /tokenize \
                   /detokenize /apply-template /v1/models /props /slots /health");
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
async fn metrics_handler() -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED,
        "This server does not support metrics endpoint. Start it with `--metrics`", "not_supported_error")
}

// The answers llama-server gives for features it was started without, which
// airun does not have: a client probing for them gets the same signal. Each
// reads the body it ignores, or the connection can close while the client is
// still sending it and the client sees a reset instead of the answer.

async fn no_embeddings(_body: axum::body::Bytes) -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED,
        "This server does not support embeddings. Start it with `--embeddings`", "not_supported_error")
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

async fn slot_action(_body: axum::body::Bytes) -> Response {
    api_error(StatusCode::NOT_IMPLEMENTED,
        "This server does not support slots action. Start it with `--slot-save-path`", "not_supported_error")
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

/// llama-server's control of a running completion (ending its reasoning
/// early). Generations here are not addressable by id while they run, so
/// every id is unknown, which is what llama-server says for a finished one.
async fn completion_control(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes)
    -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    if req.get("id").and_then(Value::as_str).is_none_or(str::is_empty) {
        return api_error(StatusCode::BAD_REQUEST, "missing completion id", "invalid_request_error");
    }
    Json(json!({ "success": false, "message": "no active completion for this id" })).into_response()
}

async fn tokenize(State(state): State<Arc<AppState>>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let req = match parse_body(&body) { Ok(v) => v, Err(e) => return e };
    let content = req.get("content").and_then(Value::as_str).unwrap_or_default().to_string();
    let add_special = req.get("add_special").and_then(Value::as_bool).unwrap_or(false);
    let with_pieces = req.get("with_pieces").and_then(Value::as_bool).unwrap_or(false);
    let rt = state.runtime.lock().await;
    let ids = rt.tokenizer.encode(&content, add_special);
    let tokens: Vec<Value> = if with_pieces {
        ids.iter().map(|&id| {
            let bytes = rt.tokenizer.decode_bytes(&[id]);
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
    let rt = state.runtime.lock().await;
    let content = rt.tokenizer.decode(&ids, false);
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
fn run_generation(state: &AppState, spec: &GenSpec,
    mut on_text: impl FnMut(&str, usize, u32, &[TokenProbs]) -> bool) -> Result<(GenEnd, String)> {
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
    let mut text = String::new();
    let mut word: Option<String> = None;
    let mut seen = 0usize;
    let max = spec.max_tokens.unwrap_or(state.ctx);
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
    let (prompt_text, tokens) = {
        let rt = state.runtime.lock().await;
        match req.get("prompt") {
            Some(Value::String(s)) => (s.clone(), rt.tokenizer.encode(s, true)),
            // Token ids, or a mix of ids and strings, as llama-server takes.
            Some(Value::Array(a)) if a.iter().all(|x| x.is_u64() || x.is_string()) && !a.is_empty() => {
                let mut ids = Vec::new();
                for x in a {
                    match x {
                        Value::String(s) => ids.extend(rt.tokenizer.encode(s, ids.is_empty())),
                        v => ids.push(v.as_u64().unwrap_or(0) as u32),
                    }
                }
                (rt.tokenizer.decode(&ids, false), ids)
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
        let rt = state.runtime.lock().await;
        let enc = |s: &str| rt.tokenizer.encode(s, false);
        let mut t = rt.tokenizer.encode("", true);
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
        (rt.tokenizer.decode(&t, false), t)
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
    let tokens = state.runtime.lock().await.tokenizer.encode(&prompt, true);
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
fn chat_prompt(state: &AppState, req: &Value) -> Result<(String, bool), Response> {
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
    let prompt = match &state.template {
        Some(t) => t.render(&messages, req.get("tools"), true, &kwargs)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, e.to_string(), "invalid_request_error"))?,
        None => fallback_chatml(state, &messages, kwargs.get("enable_thinking").and_then(Value::as_bool)),
    };
    let opened = prompt.trim_end_matches(['\n', ' ']).ends_with("<think>");
    Ok((prompt, opened))
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
    let (prompt, opened) = chat_prompt(state, req)?;
    let tokens = state.runtime.lock().await.tokenizer.encode(&prompt, true);
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
    Ok(ChatJob { spec, opened, extract, tools })
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
    Ok(state.runtime.lock().await.tokenizer.encode(&prompt, true).len())
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
    let n_probs = req.get("logprobs").and_then(Value::as_bool).unwrap_or(false)
        .then(|| req.get("top_logprobs").and_then(Value::as_u64).unwrap_or(0) as usize);
    let job = match chat_job(&state, &req, n_probs).await { Ok(j) => j, Err(e) => return e };
    let include_usage = req.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let id = completion_id("chatcmpl-");
    let created = now();
    let envelope = move |choices: Value| json!({
        "choices": choices, "created": created, "id": id, "model": model,
        "system_fingerprint": FINGERPRINT, "object": "chat.completion.chunk",
    });

    if !job.spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
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
    let st = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
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
    let chat = match responses_to_chat(&req) { Ok(c) => c, Err(e) => return e };
    let job = match chat_job(&state, &chat, None).await { Ok(j) => j, Err(e) => return e };
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let id = completion_id("resp_");
    let (rs_id, msg_id) = (completion_id("rs_"), completion_id("msg_"));
    let created = now();

    if !job.spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
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
    let chat = match responses_to_chat(&req) { Ok(c) => c, Err(e) => return e };
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
    let chat = match anthropic_to_chat(&req) { Ok(c) => c, Err(e) => return e };
    let job = match chat_job(&state, &chat, None).await { Ok(j) => j, Err(e) => return e };
    let model = req.get("model").and_then(Value::as_str).map_or_else(|| state.model_name.clone(), str::to_owned);
    let id = completion_id("msg_");

    if !job.spec.stream {
        let st = Arc::clone(&state);
        let out = tokio::task::spawn_blocking(move || {
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
    let chat = match anthropic_to_chat(&req) { Ok(c) => c, Err(e) => return e };
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
        let stops = vec!["…done".to_string()];
        let text = "wait …do";
        let hold = ambiguous_tail(text, &stops);
        assert!(text.is_char_boundary(text.len() - hold));
        assert_eq!(&text[text.len() - hold..], "…do");
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
