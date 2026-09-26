//! `airun serve` -- an OpenAI-compatible HTTP API.
//!
//! One model is held in memory and requests are serialized against it: the KV
//! cache is a single sequence, so two concurrent generations would corrupt each
//! other. Requests queue on a mutex rather than failing, which keeps clients
//! that assume server-side concurrency working.

use std::path::PathBuf;
use std::sync::Arc;

use airun_core::{Runtime, Sampler, SamplingParams, apply_chat_template, loader};
use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response, Sse, sse::Event};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;

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
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Context length in tokens.
    #[arg(short = 'c', long, default_value_t = 4096)]
    pub ctx_size: usize,

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

    /// Default maximum tokens when a request does not specify one.
    #[arg(long, default_value_t = 512)]
    pub default_max_tokens: usize,

    /// Penalty on recently generated tokens when the request does not ask for
    /// one. 1.0 disables it, at the cost of letting a model loop.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// How many recent tokens the repetition penalty considers.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,
}

struct AppState {
    runtime: Mutex<Runtime>,
    model_name: String,
    api_key: Option<String>,
    default_max_tokens: usize,
    repeat_penalty: f32,
    repeat_last_n: usize,
    /// Everything the web UI shows about the loaded model. Built once at
    /// startup so `/props` never has to touch the runtime mutex, which a
    /// generation in flight would otherwise hold for the whole response.
    props: serde_json::Value,
}

/// One line per request, so a slow first token can be attributed instead of
/// guessed at: `cached` shows how much of the prompt the cache absorbed, and
/// the prefill rate is over the tokens actually recomputed.
fn log_request(s: &airun_core::GenerationStats) {
    eprintln!(
        "  prompt {} tok ({} cached, {} new)  prefill {:.0} ms ({:.1} tok/s)  \
         decode {} tok in {:.0} ms ({:.1} tok/s)",
        s.prompt_tokens,
        s.cached_tokens,
        s.processed_tokens(),
        s.prefill_ms,
        s.prefill_tps(),
        s.generated_tokens,
        s.decode_ms,
        s.decode_tps(),
    );
}

/// Parses OpenAI's `logit_bias`: an object keyed by token id, valued by the
/// amount to add to that token's logit.
///
/// Unparseable entries are dropped rather than failing the request; a client
/// that sends a key the tokenizer does not have should get a completion, not a
/// 400. `-inf` is accepted as a way to ban a token outright.
fn logit_bias_from(v: &Option<serde_json::Value>) -> Vec<(u32, f32)> {
    let Some(serde_json::Value::Object(map)) = v else {
        return Vec::new();
    };
    map.iter()
        .filter_map(|(k, v)| {
            let id = k.parse::<u32>().ok()?;
            let bias = v.as_f64()? as f32;
            Some((id, bias))
        })
        .collect()
}

/// How many trailing bytes of `text` might still become a stop sequence.
///
/// A stop string split across two tokens would otherwise reach the client
/// before it is recognised, and an SSE chunk cannot be unsent. Holding back the
/// ambiguous tail costs at most one token of latency and only when the text
/// actually starts to look like a stop sequence.
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

/// Where the earliest stop sequence begins in `text`, if any.
fn stop_at(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// What OpenAI calls the finish reason for a run that ended this way.
fn finish_reason(stop: airun_core::StopReason) -> &'static str {
    match stop {
        // A stop sequence and a client disconnect both arrive as Cancelled;
        // "stop" is the honest answer for the first and the only sensible one
        // for the second, since nothing will read it.
        airun_core::StopReason::EndOfGeneration | airun_core::StopReason::Cancelled => "stop",
        airun_core::StopReason::MaxTokens | airun_core::StopReason::ContextFull => "length",
    }
}

pub fn run(args: ServeArgs) -> Result<()> {
    let mut run_args = RunArgs::bench_defaults(args.model.clone());
    run_args.ctx_size = args.ctx_size;
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
        args.model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("airun")
            .to_string()
    });

    let ctx = runtime.ctx();
    let cfg = &report.config;

    // A hybrid loads and generates fluent text while answering the wrong
    // question, so the UI has to say so out loud -- otherwise the first thing a
    // new user sees is a confident wrong answer with nothing marking it.
    let note = match cfg.architecture {
        airun_core::Architecture::Qwen3NextHybrid => Some(
            "Hybrid SSM/attention architecture: this model loads and produces \
             grammatical text, but its output is known to be semantically wrong \
             (perplexity ~16x a dense model of the same size). Do not judge \
             quality from it yet.",
        ),
        _ => None,
    };

    let props = json!({
        "model": model_name,
        "arch": cfg.arch_name,
        "n_layers": cfg.n_layers,
        "n_embd": cfg.n_embd,
        "vocab_size": cfg.vocab_size,
        "ctx": ctx,
        "gpus": report.placement.devices(),
        "auth_required": args.api_key.is_some(),
        "note": note,
    });

    let state = Arc::new(AppState {
        runtime: Mutex::new(runtime),
        model_name: model_name.clone(),
        api_key: args.api_key.clone(),
        default_max_tokens: args.default_max_tokens,
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        props,
    });

    let app = Router::new()
        .route("/", get(web_ui))
        .route("/props", get(props_handler))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let auth = if args.api_key.is_some() {
        "enabled"
    } else {
        "disabled (no --api-key)"
    };

    // The runtime is created here rather than via #[tokio::main] so that model
    // loading happens on the main thread, before any async machinery starts.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("could not bind {addr}"))?;
        eprintln!("\nairun serving {model_name} on http://{addr}");
        eprintln!("  web ui:         http://{addr}/");
        eprintln!("  authentication: {auth}");
        eprintln!("  endpoints: /v1/chat/completions  /v1/completions  /v1/models  /props  /health");

        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("\nshutting down");
            })
            .await
            .context("server error")
    })
}

// --- request and response shapes -------------------------------------------

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    /// Accepts both a plain string and the multi-part array form, so
    /// multi-modal clients do not fail at the parse step.
    #[serde(default)]
    content: serde_json::Value,
}

impl ChatMessage {
    /// Flattens the content into text, ignoring non-text parts.
    fn text(&self) -> String {
        match &self.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    typical_p: Option<f32>,
    #[serde(default)]
    top_n_sigma: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    /// llama.cpp's multiplicative penalty, which the OpenAI schema has no field
    /// for. Named the same so a llama.cpp client's request works unchanged.
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: serde_json::Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    typical_p: Option<f32>,
    #[serde(default)]
    top_n_sigma: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ApiError {
    error: ApiErrorBody,
}

#[derive(Serialize)]
struct ApiErrorBody {
    message: String,
    r#type: String,
    code: Option<String>,
}

fn api_error(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    (
        status,
        Json(ApiError {
            error: ApiErrorBody {
                message: message.into(),
                r#type: kind.to_string(),
                code: None,
            },
        }),
    )
        .into_response()
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = &state.api_key else {
        return Ok(());
    };
    let provided = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);

    match provided {
        Some(k) if k == expected => Ok(()),
        Some(_) => Err(api_error(
            StatusCode::UNAUTHORIZED,
            "Incorrect API key provided.",
            "invalid_request_error",
        )),
        None => Err(api_error(
            StatusCode::UNAUTHORIZED,
            "Missing bearer token. Send `Authorization: Bearer <key>`.",
            "invalid_request_error",
        )),
    }
}

// --- handlers ---------------------------------------------------------------

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// The chat page, compiled into the binary so `airun serve` stays one file with
/// nothing to install and no assets to locate at runtime.
async fn web_ui() -> impl IntoResponse {
    axum::response::Html(include_str!("webui.html"))
}

/// Model facts for the UI. Deliberately unauthenticated: it exposes only what
/// the operator already chose to serve, and requiring the key here would leave
/// the page unable to tell "wrong key" from "server down".
async fn props_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.props.clone())
}

async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    // Model discovery is behind the same key as inference: the served model name
    // is information about the deployment, not a public fact.
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": now(),
            "owned_by": "airun",
        }],
    }))
    .into_response()
}

fn stop_sequences(v: &Option<serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    if req.messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "`messages` must contain at least one message.",
            "invalid_request_error",
        );
    }

    // Fold the conversation into ChatML. Only the trailing user turn drives the
    // template; earlier turns are replayed verbatim so context is preserved.
    let mut system = None;
    let mut transcript = String::new();
    for m in &req.messages {
        match m.role.as_str() {
            "system" if system.is_none() && transcript.is_empty() => system = Some(m.text()),
            "assistant" => {
                transcript.push_str("<|im_start|>assistant\n");
                transcript.push_str(&m.text());
                transcript.push_str("<|im_end|>\n");
            }
            _ => {
                transcript.push_str("<|im_start|>user\n");
                transcript.push_str(&m.text());
                transcript.push_str("<|im_end|>\n");
            }
        }
    }
    let mut prompt = String::new();
    if let Some(sys) = &system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(sys);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str(&transcript);
    prompt.push_str("<|im_start|>assistant\n");

    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.8),
        top_k: req.top_k.unwrap_or(40),
        top_p: req.top_p.unwrap_or(0.95),
        min_p: req.min_p.unwrap_or(0.05),
        typical_p: req.typical_p.unwrap_or(1.0),
        top_n_sigma: req.top_n_sigma.unwrap_or(0.0),
        // OpenAI's two additive penalties pass straight through. The
        // multiplicative one has no field in that schema, so it keeps the
        // server default -- a client that sends frequency_penalty: 0 still gets
        // the loop protection the operator configured.
        repeat_penalty: req.repeat_penalty.unwrap_or(state.repeat_penalty),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        repeat_last_n: state.repeat_last_n,
        logit_bias: logit_bias_from(&req.logit_bias),
        seed: req.seed.unwrap_or(0),
    };
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(state.default_max_tokens);
    let stops = stop_sequences(&req.stop);
    let model = req.model.clone().unwrap_or_else(|| state.model_name.clone());

    if req.stream {
        stream_response(state, prompt, params, max_tokens, stops, model, true).await
    } else {
        blocking_response(state, prompt, params, max_tokens, stops, model, true).await
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if let Err(e) = check_auth(&state, &headers) {
        return e;
    }
    let prompt = match &req.prompt {
        serde_json::Value::String(s) => s.clone(),
        // The array form is a batch in the OpenAI API; airun runs one sequence
        // at a time, so only the first entry is honoured.
        serde_json::Value::Array(a) => a
            .first()
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    };
    if prompt.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "`prompt` must be a non-empty string.",
            "invalid_request_error",
        );
    }

    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.8),
        top_k: req.top_k.unwrap_or(40),
        top_p: req.top_p.unwrap_or(0.95),
        min_p: req.min_p.unwrap_or(0.05),
        typical_p: req.typical_p.unwrap_or(1.0),
        top_n_sigma: req.top_n_sigma.unwrap_or(0.0),
        repeat_penalty: req.repeat_penalty.unwrap_or(state.repeat_penalty),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        repeat_last_n: state.repeat_last_n,
        logit_bias: logit_bias_from(&req.logit_bias),
        seed: req.seed.unwrap_or(0),
    };
    let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens);
    let stops = stop_sequences(&req.stop);
    let model = req.model.clone().unwrap_or_else(|| state.model_name.clone());

    if req.stream {
        stream_response(state, prompt, params, max_tokens, stops, model, false).await
    } else {
        blocking_response(state, prompt, params, max_tokens, stops, model, false).await
    }
}

/// Generates on a blocking thread and returns one JSON body.
async fn blocking_response(
    state: Arc<AppState>,
    prompt: String,
    params: SamplingParams,
    max_tokens: usize,
    stops: Vec<String>,
    model: String,
    chat: bool,
) -> Response {
    let result = tokio::task::spawn_blocking(move || {
        // The lock is held across the whole generation: the KV cache is a single
        // sequence and cannot interleave.
        let mut rt = state.runtime.blocking_lock();
        // No reset: `generate` reuses whatever prefix of this prompt is already
        // in the caches, which in a chat is the entire transcript bar the last
        // turn. Resetting here would throw that away every message.
        let tokens = rt.tokenizer.encode(&prompt, true);
        let mut sampler = Sampler::new(params);
        let mut text = String::new();

        let stats = rt.generate(&tokens, max_tokens, &mut sampler, |_, piece| {
            text.push_str(piece);
            stop_at(&text, &stops).is_none()
        });

        stats.map(|s| {
            if let Some(cut) = stop_at(&text, &stops) {
                text.truncate(cut);
            }
            log_request(&s);
            (text, s)
        })
    })
    .await;

    let (text, stats) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "server_error");
        }
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("generation task failed: {e}"),
                "server_error",
            );
        }
    };

    let finish = finish_reason(stats.stop);

    let usage = json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.generated_tokens,
        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
    });

    let body = if chat {
        json!({
            "id": format!("chatcmpl-{}", now()),
            "object": "chat.completion",
            "created": now(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": finish,
            }],
            "usage": usage,
        })
    } else {
        json!({
            "id": format!("cmpl-{}", now()),
            "object": "text_completion",
            "created": now(),
            "model": model,
            "choices": [{ "index": 0, "text": text, "finish_reason": finish }],
            "usage": usage,
        })
    };
    Json(body).into_response()
}

/// Generates on a blocking thread, forwarding pieces over SSE as they appear.
async fn stream_response(
    state: Arc<AppState>,
    prompt: String,
    params: SamplingParams,
    max_tokens: usize,
    stops: Vec<String>,
    model: String,
    chat: bool,
) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<std::result::Result<Event, std::convert::Infallible>>();
    let id = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, now());
    let created = now();

    let model_for_task = model.clone();
    let id_for_task = id.clone();

    tokio::task::spawn_blocking(move || {
        let mut rt = state.runtime.blocking_lock();
        // See the note in `blocking_response`: resetting would defeat the
        // prompt cache.
        let tokens = rt.tokenizer.encode(&prompt, true);
        let mut sampler = Sampler::new(params);

        if chat {
            // OpenAI clients expect the role to arrive in the first chunk.
            let first = json!({
                "id": id_for_task, "object": "chat.completion.chunk", "created": created,
                "model": model_for_task,
                "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
            });
            let _ = tx.send(Ok(Event::default().data(first.to_string())));
        }

        let mut accumulated = String::new();
        // How much of `accumulated` the client already has. Text is released
        // from here rather than forwarded token by token, because a stop
        // sequence can straddle two tokens and an SSE chunk cannot be unsent.
        let mut sent = 0usize;

        let emit = |text: &str| {
            let chunk = if chat {
                json!({
                    "id": id_for_task, "object": "chat.completion.chunk", "created": created,
                    "model": model_for_task,
                    "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": null }],
                })
            } else {
                json!({
                    "id": id_for_task, "object": "text_completion", "created": created,
                    "model": model_for_task,
                    "choices": [{ "index": 0, "text": text, "finish_reason": null }],
                })
            };
            // A send failure means the client hung up; stop generating.
            tx.send(Ok(Event::default().data(chunk.to_string()))).is_ok()
        };

        let result = rt.generate(&tokens, max_tokens, &mut sampler, |_, piece| {
            accumulated.push_str(piece);

            if let Some(cut) = stop_at(&accumulated, &stops) {
                // Release whatever preceded the stop sequence, then finish.
                if cut > sent {
                    emit(&accumulated[sent..cut]);
                    sent = cut;
                }
                return false;
            }

            let hold = ambiguous_tail(&accumulated, &stops);
            let release = accumulated.len() - hold;
            if release <= sent {
                return true; // the whole new piece could still start a stop
            }
            let text = accumulated[sent..release].to_string();
            sent = release;
            emit(&text)
        });

        if let Err(e) = result {
            let err = json!({ "error": { "message": e.to_string(), "type": "server_error" } });
            let _ = tx.send(Ok(Event::default().data(err.to_string())));
        } else {
            // Held-back text that never became a stop sequence is still output.
            if sent < accumulated.len() && stop_at(&accumulated, &stops).is_none() {
                let tail = accumulated[sent..].to_string();
                emit(&tail);
            }
            if let Ok(s) = &result {
                log_request(s);
            }
            let finish = result.as_ref().map(|s| finish_reason(s.stop)).unwrap_or("stop");
            let last = if chat {
                json!({
                    "id": id_for_task, "object": "chat.completion.chunk", "created": created,
                    "model": model_for_task,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": finish }],
                })
            } else {
                json!({
                    "id": id_for_task, "object": "text_completion", "created": created,
                    "model": model_for_task,
                    "choices": [{ "index": 0, "text": "", "finish_reason": finish }],
                })
            };
            let _ = tx.send(Ok(Event::default().data(last.to_string())));
        }
        // The sentinel the OpenAI streaming protocol ends on.
        let _ = tx.send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(UnboundedReceiverStream::new(rx)).into_response()
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this guards: a stop sequence that arrives split across two
    /// tokens used to be streamed to the client before it was recognised.
    #[test]
    fn a_stop_sequence_split_across_tokens_is_held_back() {
        let stops = vec!["<|end|>".to_string()];

        // A tail that could still become the stop sequence must not go out.
        assert_eq!(ambiguous_tail("hello <|en", &stops), 4);
        assert_eq!(ambiguous_tail("hello <", &stops), 1);

        // Text that cannot lead anywhere is released immediately.
        assert_eq!(ambiguous_tail("hello there", &stops), 0);
        assert_eq!(ambiguous_tail("a < b", &stops), 0);
    }

    #[test]
    fn the_longest_candidate_stop_wins_the_holdback() {
        let stops = vec!["END".to_string(), "ENDING".to_string()];
        // "EN" is a prefix of both; the longer one does not change the answer
        // here, but "ENDIN" is only a prefix of the second and must still hold.
        assert_eq!(ambiguous_tail("text EN", &stops), 2);
        assert_eq!(ambiguous_tail("text ENDIN", &stops), 5);
    }

    #[test]
    fn holding_back_never_splits_a_character() {
        // A multi-byte stop sequence: the held-back length has to land on a
        // character boundary or slicing the buffer panics.
        let stops = vec!["…done".to_string()];
        let text = "wait …do";
        let hold = ambiguous_tail(text, &stops);
        assert!(text.is_char_boundary(text.len() - hold));
        assert_eq!(&text[text.len() - hold..], "…do");
    }

    #[test]
    fn the_earliest_stop_sequence_wins() {
        let stops = vec!["World".to_string(), "Hello".to_string()];
        assert_eq!(stop_at("say Hello World", &stops), Some(4));
        assert_eq!(stop_at("say nothing", &stops), None);
        // An empty stop string would match at 0 and truncate everything.
        assert_eq!(stop_at("anything", &[String::new()]), None);
    }

    #[test]
    fn logit_bias_parses_openai_shape_and_skips_junk() {
        let v = serde_json::json!({ "42": -1.5, "7": 100, "not-a-token": 1.0, "9": "x" });
        let mut bias = logit_bias_from(&Some(v));
        bias.sort_by_key(|b| b.0);
        assert_eq!(bias, vec![(7, 100.0), (42, -1.5)]);

        assert!(logit_bias_from(&None).is_empty());
        assert!(logit_bias_from(&Some(serde_json::json!([1, 2]))).is_empty());
    }

    /// A stream that ended on the model's own end-of-generation token used to
    /// report "length", which tells a client its answer was truncated.
    #[test]
    fn finish_reason_distinguishes_eos_from_truncation() {
        use airun_core::StopReason;
        assert_eq!(finish_reason(StopReason::EndOfGeneration), "stop");
        assert_eq!(finish_reason(StopReason::Cancelled), "stop");
        assert_eq!(finish_reason(StopReason::MaxTokens), "length");
        assert_eq!(finish_reason(StopReason::ContextFull), "length");
    }
}
