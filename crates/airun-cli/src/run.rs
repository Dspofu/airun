//! `airun run` -- generate text from a GGUF model.

use std::io::Write;
use std::path::PathBuf;

use airun_cuda::ops::KvType;
use airun_core::{LoadOptions, Sampler, SamplingParams, SplitMode, apply_chat_template, loader};
use anyhow::{Context, Result};
use clap::Args;

/// Where weights may live; shared by every command that loads a model.
#[derive(Args, Clone, Debug, Default)]
pub struct AllocArgs {
    /// Where the weights go: auto (the GPUs the model needs, the main one
    /// first, then RAM for what they cannot hold), gpu (VRAM only: a model
    /// that does not fit gets a smaller context or an error) or memory (every
    /// GPU in --gpus, then RAM). RAM layers are read across PCIe each step,
    /// so they are much slower than VRAM ones.
    #[arg(long, default_value = "auto", value_parser = airun_core::AllocMode::parse)]
    pub alloc: airun_core::AllocMode,

    /// CUDA ordinal of the GPU filled first, which also runs the layers kept
    /// in RAM (give it the fastest PCIe slot). Default: the fastest GPU.
    #[arg(long)]
    pub main_gpu: Option<i32>,
}

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
    /// Context length in tokens. Default: the largest the GPUs hold with
    /// every weight in VRAM (at least 4096), on the GPUs the model needs.
    #[arg(short = 'c', long)]
    pub ctx_size: Option<usize>,

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
    /// step (0 turns it off; default 4 on dense models, 2 on MoE models that
    /// fit the GPUs whole, off on MoE models with experts in RAM, where it
    /// measured slower). Engages only when the sampler is pure
    /// greedy: --temp 0 AND no penalties (this command defaults to
    /// --repeat-penalty 1.1, which disables it; pass --repeat-penalty 1.0).
    /// The drafts are verified by the model itself, so the text is the
    /// model's greedy text, but the verify runs a small batch: on a near-tie
    /// it can pick the other token than one-at-a-time decoding. Files
    /// without a usable MTP head ignore it (a note is printed).
    #[arg(long)]
    pub draft: Option<usize>,

    /// The MTP head in a file of its own (mtp-*.gguf, as llama.cpp's
    /// converter writes with --mtp). Default: the model's own head, or an
    /// mtp-*.gguf of the same model in the model's folder.
    #[arg(long, value_name = "FILE")]
    pub mtp: Option<PathBuf>,

    /// The vision encoder (mmproj-*.gguf) that lets the model see images.
    #[arg(long, value_name = "FILE")]
    pub mmproj: Option<PathBuf>,

    /// An image for the prompt (PNG, JPEG or WebP); repeat for several. Put
    /// <__media__> in the prompt where each goes, or they go before it.
    #[arg(long, value_name = "FILE", requires = "mmproj")]
    pub image: Vec<PathBuf>,

    /// Most tokens one image may take (each is a 32 x 32 pixel area; larger
    /// images are scaled down). More reads finer detail and needs more VRAM.
    #[arg(long, default_value_t = 1024)]
    pub image_max_tokens: usize,

    /// Thinking (Qwen3.5/3.8 and other models with a <think> block): on, off,
    /// or the model's default (auto).
    #[arg(long, value_name = "on|off|auto", default_value = "auto", value_parser = crate::reasoning::parse_switch_arg)]
    pub reasoning: String,

    /// How long the model may think: none, low (1024 tokens), medium (4096),
    /// high (16384), xhigh (32768) or max (unlimited). Past it `</think>` is forced and the
    /// model answers.
    #[arg(long, value_name = "LEVEL", value_parser = crate::reasoning::parse_arg)]
    pub reasoning_effort: Option<String>,

    /// Thinking tokens allowed, overriding --reasoning-effort; -1 unlimited.
    #[arg(long, allow_negative_numbers = true)]
    pub reasoning_budget: Option<i64>,

    /// CUDA ordinals to use, comma separated. Defaults to every supported GPU.
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<i32>,

    /// Layer split proportions per GPU, comma separated (e.g. 0.7,0.3).
    #[arg(long, value_delimiter = ',', conflicts_with = "single_gpu")]
    pub tensor_split: Vec<f32>,

    /// Keep the whole model on one GPU, by ordinal.
    #[arg(long)]
    pub single_gpu: Option<i32>,

    #[command(flatten)]
    pub alloc: AllocArgs,

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
            ctx_size: Some(4096),
            batch_size: 512,
            draft: Some(0),
            mtp: None,
            mmproj: None,
            image: Vec::new(),
            image_max_tokens: 1024,
            reasoning: "auto".into(),
            reasoning_effort: None,
            reasoning_budget: None,
            gpus: Vec::new(),
            tensor_split: Vec::new(),
            single_gpu: None,
            alloc: AllocArgs::default(),
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
            ctx: self.ctx_size.unwrap_or(4096),
            ctx_auto: self.ctx_size.is_none(),
            batch: self.batch_size,
            gpus: self.gpus.clone(),
            split,
            gpu_layers: None,
            expert_cache: self.expert_cache,
            kv_type: self.kv_type,
            reserve_bytes: self.vram_reserve << 20,
            allow_fallback: !self.no_fallback,
            draft: self.draft,
            mtp_file: self.mtp.clone(),
            slots: 1,
            alloc: self.alloc.alloc,
            main_gpu: self.alloc.main_gpu,
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
            ..Default::default()
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
    let mut opts = args.load_options();
    if let Some(mm) = &args.mmproj {
        opts.reserve_bytes += crate::media::vision_reserve(mm, args.image_max_tokens)?;
    }
    let images: Vec<airun_core::vision::Rgb> =
        args.image.iter().map(|p| crate::media::read_image(p)).collect::<Result<_>>()?;
    let mut bar = crate::progress::LoadBar::new(&args.model);
    let (mut runtime, report) = loader::load(&args.model, &opts, |stage, done, total| {
        if !quiet {
            bar.update(stage, done, total);
        }
    })
    .with_context(|| format!("failed to load {}", args.model.display()))?;

    if !quiet {
        bar.finish();
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

    let mut image_ids = Vec::new();
    if let Some(mm) = &args.mmproj {
        let t = std::time::Instant::now();
        let vision = crate::media::load_vision(mm, &runtime, args.image_max_tokens)?;
        for img in &images {
            image_ids.push(crate::media::encode_image(&vision, &mut runtime, img)?);
        }
        if !quiet && !images.is_empty() {
            let n: usize = image_ids.iter().map(Vec::len).sum();
            eprintln!("{} image(s) -> {n} tokens in {:.0} ms (vision on GPU {})", images.len(),
                t.elapsed().as_secs_f64() * 1000.0, vision.device);
        }
    }
    // Images without a marker go before the text.
    let mut prompt_text = prompt_text;
    let marked = prompt_text.matches(crate::media::MARKER).count();
    if marked == 0 && !image_ids.is_empty() {
        prompt_text = format!("{}{prompt_text}", format!("{}\n", crate::media::MARKER).repeat(image_ids.len()));
    }
    // Thinking off, by switch or level: an empty block the model continues
    // after, as the Qwen templates write it with enable_thinking false.
    let level = args.reasoning_effort.as_deref().and_then(crate::reasoning::parse);
    let think_off = crate::reasoning::parse_switch(&args.reasoning) == Ok(Some(false)) || level == Some(crate::reasoning::Level::Off);
    // The block opened or closed as the model's own template does it: Qwen3.5
    // and later open `<think>` in the generation prompt when thinking is on
    // (their default), so the plain ChatML prompt alone left them not
    // thinking at all.
    let template = airun_gguf::GgufFile::open(&args.model).ok()
        .and_then(|g| g.get("tokenizer.chat_template").and_then(|v| v.as_str()).map(str::to_string))
        .unwrap_or_default();
    let prefix = crate::serve::Thinking::from_template(&template).generation_prefix(!think_off);
    let opened = prefix.ends_with("<think>\n");
    let full_prompt = if args.raw {
        prompt_text
    } else {
        apply_chat_template(args.system.as_deref(), &prompt_text, Some(prefix))
    };
    let budget = match (args.reasoning_budget, level) {
        (Some(b), _) => (b >= 0).then_some(b as usize),
        (None, Some(crate::reasoning::Level::Budget(b))) => b,
        _ => None,
    };

    let tokens = crate::media::tokenize(&runtime, &full_prompt, &image_ids, true)?;
    if !quiet {
        eprintln!("prompt: {} tokens\n", tokens.len());
    }

    let mut sampler = Sampler::new(args.sampling());
    if let Some(b) = budget.filter(|_| !think_off) {
        let (start, end) = (runtime.tokenizer.encode("<think>", false), runtime.tokenizer.encode("</think>", false));
        if start.is_empty() || end.is_empty() {
            eprintln!("note: --reasoning-effort/--reasoning-budget ignored: the vocabulary has no <think> tokens");
        } else {
            sampler = sampler.with_reasoning_budget(airun_core::ReasoningBudget {
                start, forced: end.clone(), end, budget: Some(b),
                stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)), opened,
            });
        }
    }
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
