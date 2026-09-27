//! `airun perplexity` -- score a known text, so model quality is a number.
//!
//! Generated text tells you whether a model is catastrophically broken, but not
//! whether it is slightly broken: a decoder with one transposed layout still
//! writes fluent English while answering the wrong question. Scoring a text the
//! model has certainly seen the like of turns that into a measurement, which is
//! what makes A/B testing two candidate layouts possible at all.
//!
//! Tokens are fed one at a time rather than in batches, because that is the path
//! decoding actually takes -- a recurrent state that is updated wrongly per token
//! would be invisible to a single batched pass.

use std::path::PathBuf;

use airun_core::loader;
use anyhow::{Context, Result};
use clap::Args;

use crate::run::RunArgs;

/// Ordinary English, no rare words and no formatting: a model that predicts this
/// badly is broken, not merely unfamiliar with the domain.
pub(crate) const DEFAULT_TEXT: &str = "\
The capital of France is Paris, and the capital of Italy is Rome. Water freezes \
at zero degrees Celsius and boils at one hundred. The sun rises in the east and \
sets in the west. A year has twelve months, and most of them have thirty or \
thirty-one days. When it rains, the streets become wet, and people open their \
umbrellas or wait indoors until the weather clears. Children go to school in the \
morning and come home in the afternoon, where they eat, do their homework, and \
play until it is time to sleep. The doctor asked how long the pain had lasted, \
and the patient said it had started three days earlier, after a long walk in the \
cold. She wrote the answer on the board so that everyone in the room could read \
it clearly from the back.";

#[derive(Args)]
pub struct PerplexityArgs {
    /// Path to a .gguf model file.
    pub model: PathBuf,

    /// Text to score. Defaults to a built-in English passage.
    #[arg(short, long)]
    pub text: Option<String>,

    /// Read the text to score from a file instead.
    #[arg(short, long, conflicts_with = "text")]
    pub file: Option<PathBuf>,

    /// Stop after this many tokens. 0 scores everything.
    #[arg(short = 'n', long, default_value_t = 0)]
    pub limit: usize,

    /// Print the per-token loss, highest first.
    #[arg(long, default_value_t = 0)]
    pub worst: usize,

    /// Context length in tokens.
    #[arg(short = 'c', long, default_value_t = 4096)]
    pub ctx_size: usize,

    /// Resident experts per layer; stream cache misses from the GGUF without re-quantizing.
    #[arg(long)]
    pub expert_cache: Option<usize>,

    /// KV cache format: f16, q8_0 or q4_0.
    #[arg(long, default_value = "f16", value_parser = crate::run::parse_kv_type)]
    pub kv_type: airun_cuda::ops::KvType,

    /// CUDA ordinals to use.
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<i32>,

    /// Layer split proportions per GPU.
    #[arg(long, value_delimiter = ',')]
    pub tensor_split: Vec<f32>,

    /// Keep the whole model on one GPU, by ordinal.
    #[arg(long)]
    pub single_gpu: Option<i32>,
}

pub fn run(args: PerplexityArgs) -> Result<()> {
    let text = match (&args.text, &args.file) {
        (Some(t), _) => t.clone(),
        (None, Some(p)) => {
            std::fs::read_to_string(p).with_context(|| format!("failed to read {}", p.display()))?
        }
        (None, None) => DEFAULT_TEXT.to_string(),
    };

    let mut run_args = RunArgs::bench_defaults(args.model.clone());
    run_args.ctx_size = args.ctx_size;
    run_args.expert_cache = args.expert_cache;
    run_args.kv_type = args.kv_type;
    run_args.batch_size = 1;
    run_args.gpus = args.gpus.clone();
    run_args.tensor_split = args.tensor_split.clone();
    run_args.single_gpu = args.single_gpu;

    let (mut runtime, report) = loader::load(&args.model, &run_args.load_options(), |_, _, _| {})
        .with_context(|| format!("failed to load {}", args.model.display()))?;

    println!("{}", report.config.summary());

    let tokens = runtime.tokenizer.encode(&text, true);
    if tokens.len() < 2 {
        anyhow::bail!("need at least two tokens to score; the text produced {}", tokens.len());
    }
    let scored = match args.limit {
        0 => tokens.len(),
        n => n.min(tokens.len()),
    };
    if scored + 1 >= runtime.ctx() {
        anyhow::bail!(
            "the text is {scored} tokens but the context is {}; raise --ctx-size",
            runtime.ctx()
        );
    }

    runtime.reset();

    let mut losses: Vec<(usize, u32, f32)> = Vec::with_capacity(scored - 1);
    let mut total = 0f64;
    let mut top1 = 0usize;

    for i in 0..scored - 1 {
        let logits = runtime.forward(&tokens[i..i + 1])?;
        let target = tokens[i + 1] as usize;

        // Log-softmax the stable way; a 250k-entry vocabulary makes the naive
        // form overflow long before the answer is wrong.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
        let nll = -((logits[target] - max) - sum.ln());

        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if best == target {
            top1 += 1;
        }

        total += nll as f64;
        losses.push((i + 1, tokens[i + 1], nll));
    }

    let n = losses.len() as f64;
    let mean = total / n;
    println!(
        "\n{} tokens scored\n  mean loss   {:.4} nats/token\n  perplexity  {:.2}\n  top-1       {:.1}% ({}/{})",
        losses.len(),
        mean,
        mean.exp(),
        100.0 * top1 as f64 / n,
        top1,
        losses.len()
    );

    if args.worst > 0 {
        let mut sorted = losses.clone();
        sorted.sort_by(|a, b| b.2.total_cmp(&a.2));
        println!("\nworst-predicted tokens:");
        for (pos, tok, nll) in sorted.iter().take(args.worst) {
            println!(
                "  pos {pos:>4}  loss {nll:>8.3}  {:?}",
                runtime.tokenizer.decode_one(*tok)
            );
        }
    }

    Ok(())
}
