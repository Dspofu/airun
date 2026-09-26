//! `airun inspect` -- describe a GGUF file.

use std::collections::BTreeMap;
use std::path::PathBuf;

use airun_gguf::{GgufFile, Support};
use anyhow::{Context, Result};
use clap::Args;

#[derive(Args)]
pub struct InspectArgs {
    /// Path to a .gguf model file.
    pub model: PathBuf,

    /// List every tensor with its shape, type and offset.
    #[arg(long)]
    pub tensors: bool,

    /// Show all metadata keys, including the tokenizer arrays.
    #[arg(long)]
    pub all_metadata: bool,

    /// Emit machine-readable JSON instead of a report.
    #[arg(long)]
    pub json: bool,

    /// Decode a tensor and print its first values, with summary statistics.
    /// Useful for checking whether a parameter is stored raw or pre-transformed.
    #[arg(long, value_name = "TENSOR")]
    pub values: Option<String>,

    /// How many values `--values` prints.
    #[arg(long, default_value_t = 16)]
    pub count: usize,
}

pub fn run(args: InspectArgs) -> Result<()> {
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("failed to open {}", args.model.display()))?;

    if let Some(name) = &args.values {
        return dump_values(&gguf, name, args.count);
    }

    if args.json {
        return emit_json(&gguf, &args);
    }

    println!("file      {}", gguf.path().display());
    println!(
        "size      {}  (header {}, tensor data {})",
        human_bytes(gguf.file_len()),
        human_bytes(gguf.data_offset()),
        human_bytes(gguf.tensor_data_size())
    );
    println!(
        "gguf      v{}  alignment {}  {} tensors  {} metadata keys",
        gguf.version(),
        gguf.alignment(),
        gguf.tensors().len(),
        gguf.metadata().len()
    );

    println!("\n== metadata ==");
    // Tokenizer arrays are enormous and rarely what you are looking for, so they
    // stay behind --all-metadata.
    let hidden_prefixes = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"];
    let mut hidden = 0usize;
    for (key, value) in gguf.metadata() {
        if !args.all_metadata && hidden_prefixes.iter().any(|p| key.starts_with(p)) {
            hidden += 1;
            continue;
        }
        println!("  {key:<44} {value}");
    }
    if hidden > 0 {
        println!("  ({hidden} bulk tokenizer keys hidden; pass --all-metadata to show)");
    }

    println!("\n== quantization mix ==");
    let hist = gguf.type_histogram();
    let total_bytes: u64 = hist.values().map(|(_, b)| *b).sum();
    let mut rows: Vec<_> = hist.iter().collect();
    rows.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
    for (ty_name, (count, bytes)) in rows {
        let pct = if total_bytes > 0 {
            *bytes as f64 * 100.0 / total_bytes as f64
        } else {
            0.0
        };
        println!(
            "  {ty_name:<10} {count:>5} tensors  {:>10}  {pct:>5.1}%",
            human_bytes(*bytes)
        );
    }

    // Anything airun cannot yet compute on is worth calling out up front rather
    // than failing halfway through a load.
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    for t in gguf.tensors() {
        if !matches!(t.ty.support(), Support::Full) {
            *unsupported.entry(format!("{} ({})", t.ty, t.ty.support().label())).or_default() += 1;
        }
    }
    if unsupported.is_empty() {
        println!("\n  all tensor types are supported by the airun compute path");
    } else {
        println!("\n  types without a fused GPU kernel:");
        for (label, count) in &unsupported {
            println!("    {label:<28} {count} tensors");
        }
    }

    let n_elements: u64 = gguf.tensors().iter().map(|t| t.n_elements()).sum();
    println!("\n== totals ==");
    println!("  parameters      {}", human_count(n_elements));
    if n_elements > 0 {
        println!(
            "  effective bpw   {:.2} bits/weight",
            total_bytes as f64 * 8.0 / n_elements as f64
        );
    }

    if args.tensors {
        println!("\n== tensors ==");
        println!(
            "  {:<48} {:<10} {:>18} {:>14} {:>14}",
            "name", "type", "shape", "bytes", "offset"
        );
        for t in gguf.tensors() {
            let shape = t
                .dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            println!(
                "  {:<48} {:<10} {:>18} {:>14} {:>14}",
                t.name,
                t.ty.to_string(),
                shape,
                t.nbytes,
                t.file_offset
            );
        }
    }

    Ok(())
}

fn emit_json(gguf: &GgufFile, args: &InspectArgs) -> Result<()> {
    use serde_json::{Map, Value as J, json};

    let mut meta = Map::new();
    for (k, v) in gguf.metadata() {
        if !args.all_metadata && (k.starts_with("tokenizer.ggml.tokens") || k.starts_with("tokenizer.ggml.merges")) {
            continue;
        }
        meta.insert(k.clone(), J::String(v.to_string()));
    }

    let tensors: Vec<J> = if args.tensors {
        gguf.tensors()
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "type": t.ty.to_string(),
                    "type_id": t.ty.id(),
                    "support": t.ty.support().label(),
                    "dims": t.dims,
                    "bytes": t.nbytes,
                    "offset": t.file_offset,
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    let hist: Map<String, J> = gguf
        .type_histogram()
        .into_iter()
        .map(|(k, (count, bytes))| (k, json!({ "tensors": count, "bytes": bytes })))
        .collect();

    let n_elements: u64 = gguf.tensors().iter().map(|t| t.n_elements()).sum();
    let out = json!({
        "path": gguf.path().display().to_string(),
        "file_size": gguf.file_len(),
        "gguf_version": gguf.version(),
        "alignment": gguf.alignment(),
        "data_offset": gguf.data_offset(),
        "tensor_count": gguf.tensors().len(),
        "parameters": n_elements,
        "architecture": gguf.architecture(),
        "quantization_mix": hist,
        "metadata": meta,
        "tensors": tensors,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

/// Decodes a tensor and prints values plus statistics.
///
/// Statistics matter more than the values: whether a parameter is stored as a
/// log, already exponentiated, or negated is invisible in a shape but obvious
/// from its range.
fn dump_values(gguf: &GgufFile, name: &str, count: usize) -> Result<()> {
    let info = gguf
        .tensor(name)
        .with_context(|| format!("no tensor named {name:?} in this file"))?;

    let n = info.n_elements() as usize;
    let take = count.min(n);
    let mut all = vec![0f32; n];
    airun_gguf::dequant::dequantize(info.ty, gguf.tensor_bytes(info)?, n, &mut all)?;

    let shape = info
        .dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("x");
    println!("{name}  {}  shape {shape}  ({n} values)", info.ty);

    let finite: Vec<f32> = all.iter().copied().filter(|v| v.is_finite()).collect();
    if !finite.is_empty() {
        let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
        let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = finite.iter().sum::<f32>() / finite.len() as f32;
        let neg = finite.iter().filter(|v| **v < 0.0).count();
        println!(
            "  min {min:.6}  max {max:.6}  mean {mean:.6}  negative {neg}/{}",
            finite.len()
        );
        if all.len() != finite.len() {
            println!("  WARNING: {} non-finite values", all.len() - finite.len());
        }
    }

    println!("  first {take}:");
    for chunk in all[..take].chunks(8) {
        let row: Vec<String> = chunk.iter().map(|v| format!("{v:>12.6}")).collect();
        println!("    {}", row.join(" "));
    }
    Ok(())
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2} B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2} M", n as f64 / 1e6)
    } else {
        n.to_string()
    }
}
