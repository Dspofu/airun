#![recursion_limit = "256"]
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
mod chat;
mod serve;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "airun",
    version,
    about = "Multi-modal LLM runtime for NVIDIA GPUs",
    long_about = "airun runs GGUF models on NVIDIA GPUs with its own CUDA kernels, \
                  converts and quantizes checkpoints, and serves an OpenAI-compatible API.",
    after_long_help = "Common options (full list for each command: `airun <COMMAND> --help`):
  run, bench, perplexity, quantize
      -c, --ctx-size <N>          context length in tokens (default 4096)
          --kv-type <f16|q8_0|q4_0>  KV cache format; q4_0 fits ~3.6x more context
      -b, --batch-size <N>        prompt tokens per forward pass (default 512)
          --gpus <0,1>            CUDA ordinals to use
          --alloc <auto|gpu|memory>  where the weights live
          --main-gpu <N>          GPU filled first (and running the RAM layers)
  run, serve
          --draft <N>             MTP speculative decoding, drafts per step (run: 3, serve: 0)
          --temp <T>              sampling temperature; 0 is greedy
  run
      -p, --prompt <TEXT>         the prompt (reads stdin when omitted)
      -n, --predict <N>           maximum tokens to generate (default 256)
  serve
          --port <PORT>           HTTP port (default 8080)
          --alias <NAME>          name advertised by /v1/models
  convert
          --hf <REPO>            Hugging Face repo, optional :QUANT suffix
  quantize
          --from <FILE> --to <FILE>  rewrite a GGUF at another precision

Examples:
  airun run model.gguf -p \"Hello\" -n 128
  airun serve model.gguf --port 8080 --draft 3 --kv-type q4_0
  airun bench model.gguf --decode 128 --decode-at 512
  airun perplexity model.gguf --worst 10"
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
