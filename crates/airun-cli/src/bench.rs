//! `airun bench` -- measure prefill and decode throughput.

use std::path::PathBuf;

use airun_core::{Sampler, SamplingParams, loader};
use anyhow::{Context, Result};
use clap::Args;

use crate::run::RunArgs;

#[derive(Args)]
pub struct BenchArgs {
    /// Path to a .gguf model file.
    pub model: PathBuf,

    /// Prompt lengths to measure prefill at, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = [128usize, 512])]
    pub prefill: Vec<usize>,

    /// Tokens to generate when measuring decode.
    #[arg(long, default_value_t = 128)]
    pub decode: usize,

    /// Context depths to measure decode at, comma separated. Decode cost is not
    /// flat: attention reads the whole cache every token, so a rate measured on
    /// an empty context says nothing about the rate a long chat will see.
    #[arg(long, value_delimiter = ',', default_values_t = [16usize])]
    pub decode_at: Vec<usize>,

    /// Measured runs per configuration.
    #[arg(long, default_value_t = 3)]
    pub runs: usize,

    /// Discarded warm-up runs, which pay for kernel JIT and clock ramp-up.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,

    /// Context length in tokens.
    #[arg(short = 'c', long, default_value_t = 4096)]
    pub ctx_size: usize,

    /// Tokens per forward pass during prefill.
    #[arg(short = 'b', long, default_value_t = 512)]
    pub batch_size: usize,

    /// CUDA ordinals to use.
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<i32>,

    /// Layer split proportions per GPU.
    #[arg(long, value_delimiter = ',')]
    pub tensor_split: Vec<f32>,

    /// Emit machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

struct Measurement {
    label: String,
    tokens: usize,
    samples: Vec<f64>,
}

impl Measurement {
    fn tps(&self) -> Vec<f64> {
        self.samples
            .iter()
            .map(|ms| self.tokens as f64 / (ms / 1000.0))
            .collect()
    }

    /// Median is reported rather than mean: a single scheduling hiccup or a
    /// clock drop skews a three-run mean badly, and the median does not move.
    fn median_tps(&self) -> f64 {
        let mut v = self.tps();
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            return 0.0;
        }
        v[v.len() / 2]
    }

    fn min_tps(&self) -> f64 {
        self.tps().iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max_tps(&self) -> f64 {
        self.tps().iter().copied().fold(0.0, f64::max)
    }
}

pub fn run(args: BenchArgs) -> Result<()> {
    let mut run_args = RunArgs::bench_defaults(args.model.clone());
    run_args.ctx_size = args.ctx_size;
    run_args.batch_size = args.batch_size;
    run_args.gpus = args.gpus.clone();
    run_args.tensor_split = args.tensor_split.clone();

    let (mut runtime, report) = loader::load(&args.model, &run_args.load_options(), |_, _, _| {})
        .with_context(|| format!("failed to load {}", args.model.display()))?;

    if !args.json {
        println!("{}", report.config.summary());
        println!(
            "weights {} loaded in {:.1}s",
            airun_cuda::human_bytes(report.weight_bytes),
            report.load_ms / 1000.0
        );
        print!("{}", report.placement.report());
        println!();
    }

    // A fixed repeated token keeps tokenization out of the measurement and makes
    // every run identical in shape.
    let filler = runtime
        .tokenizer
        .id_of("the")
        .or_else(|| runtime.tokenizer.id_of(" the"))
        .unwrap_or(1);

    let mut results = Vec::new();

    for &n in &args.prefill {
        if n + 8 >= runtime.ctx() {
            eprintln!("skipping prefill {n}: it does not fit the {} context", runtime.ctx());
            continue;
        }
        let prompt = vec![filler; n];
        let mut samples = Vec::new();

        for i in 0..(args.warmup + args.runs) {
            runtime.reset();
            let start = std::time::Instant::now();
            let mut fed = 0;
            while fed < prompt.len() {
                let take = (prompt.len() - fed).min(args.batch_size);
                runtime.forward(&prompt[fed..fed + take])?;
                fed += take;
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if i >= args.warmup {
                samples.push(ms);
            }
        }
        results.push(Measurement {
            label: format!("prefill {n}"),
            tokens: n,
            samples,
        });
    }

    // Decode at each requested depth. The prompt is prefilled first and only the
    // generation is timed, so the number is per-token latency with that much
    // cache already behind it.
    for &depth in &args.decode_at {
        if depth + args.decode + 8 >= runtime.ctx() {
            eprintln!(
                "skipping decode at {depth}: {depth} + {} generated does not fit the {} context",
                args.decode,
                runtime.ctx()
            );
            continue;
        }
        let prompt = vec![filler; depth.max(1)];
        let mut samples = Vec::new();
        for i in 0..(args.warmup + args.runs) {
            runtime.reset();
            let mut sampler = Sampler::new(SamplingParams::greedy());
            let stats = runtime.generate(&prompt, args.decode, &mut sampler, |_, _| true)?;
            if i >= args.warmup && stats.generated_tokens > 0 {
                // Normalize to the tokens actually produced; an early stop would
                // otherwise inflate the rate.
                samples.push(stats.decode_ms * args.decode as f64 / stats.generated_tokens as f64);
            }
        }
        results.push(Measurement {
            label: format!("decode {} @ {depth}", args.decode),
            tokens: args.decode,
            samples,
        });
    }

    if args.json {
        let rows: Vec<_> = results
            .iter()
            .map(|m| {
                serde_json::json!({
                    "label": m.label,
                    "tokens": m.tokens,
                    "median_tps": m.median_tps(),
                    "min_tps": m.min_tps(),
                    "max_tps": m.max_tps(),
                    "samples_ms": m.samples,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": args.model.display().to_string(),
                "architecture": report.config.arch_name,
                "parameters_summary": report.config.summary(),
                "weight_bytes": report.weight_bytes,
                "ctx": runtime.ctx(),
                "gpus": report.placement.devices(),
                "results": rows,
            }))?
        );
        return Ok(());
    }

    println!("{:<16} {:>12} {:>12} {:>12}", "test", "median", "min", "max");
    println!("{:-<56}", "");
    for m in &results {
        println!(
            "{:<16} {:>9.1} t/s {:>9.1} t/s {:>9.1} t/s",
            m.label,
            m.median_tps(),
            m.min_tps(),
            m.max_tps()
        );
    }
    println!(
        "\n{} run(s) per test after {} warm-up run(s); median reported",
        args.runs, args.warmup
    );
    Ok(())
}
