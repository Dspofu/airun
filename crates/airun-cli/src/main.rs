//! airun -- multi-modal LLM runtime for NVIDIA GPUs.

mod bench;
mod convert;
mod gpu;
mod inspect;
mod perplexity;
mod quantize;
mod run;
mod selftest;
mod tokenize;
mod serve;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "airun",
    version,
    about = "Multi-modal LLM runtime for NVIDIA GPUs",
    long_about = "airun runs GGUF models on NVIDIA GPUs with its own CUDA kernels, \
                  converts and quantizes checkpoints, and serves an OpenAI-compatible API."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text from a GGUF model.
    Run(run::RunArgs),

    /// Describe a GGUF model: metadata, tensors and quantization mix.
    Inspect(inspect::InspectArgs),

    /// Serve an OpenAI-compatible HTTP API.
    Serve(serve::ServeArgs),

    /// Convert a HuggingFace safetensors checkpoint into GGUF.
    Convert(convert::ConvertArgs),

    /// Rewrite a GGUF file at a different precision.
    Quantize(quantize::QuantizeArgs),

    /// Measure prefill and decode throughput.
    Bench(bench::BenchArgs),

    /// Score a text to measure how well a model predicts it.
    Perplexity(perplexity::PerplexityArgs),

    /// Show how a model's tokenizer splits text.
    Tokenize(tokenize::TokenizeArgs),

    /// Report the CUDA devices airun can see.
    Gpu(gpu::GpuArgs),

    /// Check every GPU kernel against the CPU reference implementation.
    Selftest(selftest::SelftestArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run::run(args),
        Command::Inspect(args) => inspect::run(args),
        Command::Serve(args) => serve::run(args),
        Command::Convert(args) => convert::run(args),
        Command::Quantize(args) => quantize::run(args),
        Command::Bench(args) => bench::run(args),
        Command::Perplexity(args) => perplexity::run(args),
        Command::Tokenize(args) => tokenize::run(args),
        Command::Gpu(args) => gpu::run(args),
        Command::Selftest(args) => selftest::run(args),
    }
}
