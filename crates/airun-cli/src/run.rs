//! `airun run` -- generate text from a GGUF model.

use std::io::Write;
use std::path::PathBuf;

use airun_cuda::ops::KvType;
use airun_core::{LoadOptions, Sampler, SamplingParams, SplitMode, apply_chat_template, loader};
use anyhow::{Context, Result};
use clap::Args;

#[derive(Args, Clone)]
pub struct RunArgs {
    /// Path to a .gguf model file.
    pub model: PathBuf,

    // --- prompt ---
    /// Prompt text. Reads stdin when omitted.
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// System message placed ahead of the prompt.
    #[arg(long)]
    pub system: Option<String>,

    /// Send the prompt exactly as given, with no chat template.
    #[arg(long)]
    pub raw: bool,

    /// Maximum tokens to generate.
    #[arg(short = 'n', long, default_value_t = 256)]
    pub predict: usize,

    // --- memory and placement ---
    /// Context length in tokens.
    #[arg(short = 'c', long, default_value_t = 4096)]
    pub ctx_size: usize,

    /// Resident experts per layer; stream cache misses from the GGUF without re-quantizing.
    #[arg(long)]
    pub expert_cache: Option<usize>,

    /// KV cache format: f16, q8_0 or q4_0. The quantized formats fit 2x and
    /// 3.6x more context in the same VRAM, at a small cost in accuracy.
    #[arg(long, default_value = "f16", value_parser = parse_kv_type)]
    pub kv_type: KvType,

    /// Tokens processed per forward pass during prompt ingestion.
    #[arg(short = 'b', long, default_value_t = 512)]
    pub batch_size: usize,

    /// Speculative decoding with the model's own MTP head: tokens drafted per
    /// step (0 turns it off). Greedy decoding only (--temp 0); the output is the
    /// same text plain decoding writes. Ignored for files without an MTP head.
    #[arg(long, default_value_t = 3)]
    pub draft: usize,

    /// CUDA ordinals to use, comma separated. Defaults to every supported GPU.
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<i32>,

    /// Layer split proportions per GPU, comma separated (e.g. 0.7,0.3).
    #[arg(long, value_delimiter = ',', conflicts_with = "single_gpu")]
    pub tensor_split: Vec<f32>,

    /// Keep the whole model on one GPU, by ordinal.
    #[arg(long)]
    pub single_gpu: Option<i32>,

    /// VRAM left free on each GPU, in MiB.
    #[arg(long, default_value_t = 384)]
    pub vram_reserve: u64,

    /// Fail instead of shrinking context or batch to fit.
    #[arg(long)]
    pub no_fallback: bool,

    // --- sampling ---
    /// Sampling temperature. 0 selects greedy decoding.
    #[arg(long, default_value_t = 0.8)]
    pub temp: f32,

    /// Keep only the top K tokens. 0 disables.
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    /// Nucleus sampling threshold. 1.0 disables.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Drop tokens below this fraction of the best token's probability.
    #[arg(long, default_value_t = 0.05)]
    pub min_p: f32,

    /// Locally typical sampling: keep the tokens whose surprise is closest to
    /// the distribution's entropy. 1.0 disables.
    #[arg(long, default_value_t = 1.0)]
    pub typical_p: f32,

    /// Keep only logits within this many standard deviations of the best one.
    /// 0 disables. Unlike top-p, the cut does not widen with temperature.
    #[arg(long, default_value_t = 0.0)]
    pub top_n_sigma: f32,

    /// Penalty applied to recently generated tokens. 1.0 disables.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// Subtracted once from any token already seen. 0 disables.
    #[arg(long, default_value_t = 0.0)]
    pub presence_penalty: f32,

    /// Subtracted per occurrence of the token. 0 disables.
    #[arg(long, default_value_t = 0.0)]
    pub frequency_penalty: f32,

    /// How many recent tokens the three penalties consider.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// RNG seed. 0 picks a random one.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    // --- output ---
    /// Print the memory plan and timing summary.
    #[arg(long)]
    pub stats: bool,

    /// Suppress load progress on stderr.
    #[arg(short, long)]
    pub quiet: bool,
}

impl RunArgs {
    /// Defaults for callers that only need the placement flags, such as `bench`.
    pub fn bench_defaults(model: PathBuf) -> Self {
        Self {
            model,
            prompt: None,
            system: None,
            raw: true,
            predict: 128,
            ctx_size: 4096,
            batch_size: 512,
            draft: 0,
            gpus: Vec::new(),
            tensor_split: Vec::new(),
            single_gpu: None,
            expert_cache: None,
            kv_type: KvType::F16,
            vram_reserve: 384,
            no_fallback: false,
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            top_n_sigma: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: 0,
            seed: 1,
            stats: false,
            quiet: true,
        }
    }

}

pub fn parse_kv_type(s: &str) -> Result<KvType, String> {
    KvType::parse(s).ok_or_else(|| format!("unknown KV cache type '{s}'; use f16, q8_0 or q4_0"))
}

impl RunArgs {
    pub fn load_options(&self) -> LoadOptions {
        let split = if let Some(g) = self.single_gpu {
            SplitMode::Single(g)
        } else if !self.tensor_split.is_empty() {
            SplitMode::Manual(self.tensor_split.clone())
        } else {
            SplitMode::Auto
        };

        LoadOptions {
            ctx: self.ctx_size,
            batch: self.batch_size,
            gpus: self.gpus.clone(),
            split,
            gpu_layers: None,
            expert_cache: self.expert_cache,
            kv_type: self.kv_type,
            reserve_bytes: self.vram_reserve << 20,
            allow_fallback: !self.no_fallback,
            draft: self.draft,
        }
    }

    pub fn sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            typical_p: self.typical_p,
            top_n_sigma: self.top_n_sigma,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_last_n: self.repeat_last_n,
            logit_bias: Vec::new(),
            seed: self.seed,
        }
    }
}

pub fn run(args: RunArgs) -> Result<()> {
    let prompt_text = match &args.prompt {
        Some(p) => p.clone(),
        None => {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
                .context("failed to read the prompt from stdin")?;
            buf
        }
    };
    if prompt_text.trim().is_empty() {
        anyhow::bail!("no prompt given: pass --prompt or pipe text on stdin");
    }

    let quiet = args.quiet;
    let mut last_stage = String::new();
    let (mut runtime, report) = loader::load(&args.model, &args.load_options(), |stage, done, total| {
        if quiet {
            return;
        }
        if stage != last_stage {
            last_stage = stage.to_string();
        }
        if total > 0 && (done % 32 == 0 || done == total) {
            eprint!("\r{stage}: {done}/{total}   ");
            let _ = std::io::stderr().flush();
        }
    })
    .with_context(|| format!("failed to load {}", args.model.display()))?;

    if !quiet {
        eprint!("\r");
        eprintln!("{}", report.config.summary());
        eprintln!(
            "loaded {} of weights in {:.1}s",
            airun_cuda::human_bytes(report.weight_bytes),
            report.load_ms / 1000.0
        );
        for f in &report.fallbacks {
            eprintln!("  adjusted: {f}");
        }
    }
    if args.stats {
        eprint!("{}", report.placement.report());
    }

    let full_prompt = if args.raw {
        prompt_text
    } else {
        apply_chat_template(args.system.as_deref(), &prompt_text, None)
    };

    let tokens = runtime.tokenizer.encode(&full_prompt, true);
    if !quiet {
        eprintln!("prompt: {} tokens\n", tokens.len());
    }

    let mut sampler = Sampler::new(args.sampling());
    let mut stdout = std::io::stdout();

    let stats = runtime.generate(&tokens, args.predict, &mut sampler, |_, text| {
        print!("{text}");
        let _ = stdout.flush();
        true
    })?;

    println!();
    if args.stats || !quiet {
        eprintln!(
            "\n{} prompt tokens in {:.0} ms ({:.1} tok/s)  |  {} generated in {:.0} ms ({:.1} tok/s)",
            stats.prompt_tokens,
            stats.prefill_ms,
            stats.prefill_tps(),
            stats.generated_tokens,
            stats.decode_ms,
            stats.decode_tps(),
        );
        if stats.drafted > 0 {
            eprintln!("speculation: {} of {} drafted tokens accepted ({:.0}%)", stats.accepted, stats.drafted,
                100.0 * stats.accepted as f64 / stats.drafted as f64);
        }
        let (hits, misses) = runtime.expert_cache_stats();
        if hits + misses > 0 {
            eprintln!(
                "expert cache: {:.1}% hits ({hits} hits, {misses} misses)",
                100.0 * hits as f64 / (hits + misses) as f64
            );
            let (wait, read) = runtime.expert_stream_timing();
            eprintln!("expert streaming: {wait:.0} ms waiting on the GPU for routing, {read:.0} ms reading misses");
        }
    }
    Ok(())
}
