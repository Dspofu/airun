//! `airun quantize` -- rewrite an existing GGUF at a different precision.

use std::io::Write;
use std::path::PathBuf;

use airun_gguf::{GgmlType, GgufFile, GgufWriter, dequant, quantize as q};
use anyhow::{Context, Result, bail};
use clap::Args;

#[derive(Args)]
pub struct QuantizeArgs {
    /// Source .gguf file.
    pub input: PathBuf,

    /// Destination .gguf file.
    pub output: PathBuf,

    /// Target type for the bulk of the weights.
    #[arg(long, default_value = "Q4_K")]
    pub r#type: String,

    /// Quantize every matrix to --type, with no per-tensor exceptions.
    #[arg(long)]
    pub pure: bool,

    /// Describe what would be written without writing it.
    #[arg(long)]
    pub dry_run: bool,
}

pub fn run(args: QuantizeArgs) -> Result<()> {
    let target = q::parse_type(&args.r#type)
        .with_context(|| format!("unknown quantization type {:?}", args.r#type))?;
    if !q::can_encode(target) {
        bail!(
            "airun cannot write {target}. Supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, \
             Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, NVFP4"
        );
    }

    let src = GgufFile::open(&args.input)
        .with_context(|| format!("failed to open {}", args.input.display()))?;

    // Decide every tensor's fate before touching the output, so a type we cannot
    // read fails immediately rather than halfway through a multi-gigabyte write.
    struct Job {
        name: String,
        dims: Vec<u64>,
        from: GgmlType,
        to: GgmlType,
        elements: u64,
    }

    let mut jobs = Vec::new();
    let mut in_bytes = 0u64;
    let mut out_bytes = 0u64;

    for t in src.tensors() {
        if !dequant::can_decode(t.ty) {
            bail!(
                "tensor {:?} is stored as {}, which airun cannot decode, so it cannot be \
                 requantized.\n  Run `airun inspect --tensors {}` to see the full type mix.",
                t.name,
                t.ty,
                args.input.display()
            );
        }

        let to = choose_type(&t.name, t.dims.len(), t.ty, target, args.pure);
        let n = t.n_elements();
        let row_len = t.row_size();
        let block = to.layout().unwrap().elements;
        if row_len % block != 0 {
            bail!(
                "tensor {:?} has rows of {row_len}, which is not a multiple of the {to} \
                 block size {block}. Pick a type with a smaller block, such as Q8_0.",
                t.name
            );
        }

        in_bytes += t.nbytes;
        out_bytes += to.nbytes(n).unwrap_or(0);
        jobs.push(Job {
            name: t.name.clone(),
            dims: t.dims.clone(),
            from: t.ty,
            to,
            elements: n,
        });
    }

    println!(
        "{} -> {}\n  {} tensors, {} -> {} ({:.1}% of the original)",
        args.input.display(),
        args.output.display(),
        jobs.len(),
        airun_cuda::human_bytes(in_bytes),
        airun_cuda::human_bytes(out_bytes),
        out_bytes as f64 * 100.0 / in_bytes.max(1) as f64
    );

    let changed = jobs.iter().filter(|j| j.from != j.to).count();
    println!("  {changed} tensors change type, {} stay as they are", jobs.len() - changed);

    if args.dry_run {
        println!("\ndry run: nothing was written");
        return Ok(());
    }

    let mut w = GgufWriter::create(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;

    // Carry the source metadata across so the tokenizer and shape keys survive.
    for (key, value) in src.metadata() {
        if key == "general.file_type" {
            continue;
        }
        w.set_metadata(key.clone(), value.clone());
    }
    w.set_str("general.quantized_by", "airun");

    for j in &jobs {
        w.add_tensor(j.name.clone(), j.dims.clone(), j.to)?;
    }
    w.write_header()?;

    let mut values: Vec<f32> = Vec::new();
    let mut packed: Vec<u8> = Vec::new();

    for (i, j) in jobs.iter().enumerate() {
        let info = src.tensor(&j.name).unwrap();
        let bytes = src.tensor_bytes(info)?;

        if j.from == j.to {
            // Same type: copy the packed bytes straight through, which avoids a
            // lossy decode/re-encode round trip.
            w.write_tensor_data(&j.name, bytes)?;
        } else {
            values.clear();
            values.resize(j.elements as usize, 0.0);
            dequant::dequantize(j.from, bytes, j.elements as usize, &mut values)?;

            packed.clear();
            q::quantize(j.to, &values, info.row_size() as usize, &mut packed)?;
            w.write_tensor_data(&j.name, &packed)?;
        }

        if i % 8 == 0 || i + 1 == jobs.len() {
            eprint!(
                "\r[{}/{}] {:<46} {} -> {}   ",
                i + 1,
                jobs.len(),
                j.name,
                j.from,
                j.to
            );
            let _ = std::io::stderr().flush();
        }
    }

    let path = w.finish()?;
    eprintln!();
    println!("\nwrote {}", path.display());
    Ok(())
}

/// Per-tensor policy, mirroring `airun convert`.
///
/// Requantizing upward is refused: a tensor already stored at 4 bits gains
/// nothing from being rewritten at 8, and the file would just get larger while
/// still carrying the original quantization error.
fn choose_type(
    name: &str,
    n_dims: usize,
    from: GgmlType,
    target: GgmlType,
    pure: bool,
) -> GgmlType {
    if n_dims < 2 || name.ends_with("_norm.weight") || name == "output_norm.weight" {
        return from;
    }
    if !target.is_quantized() {
        return target;
    }

    let want = if pure {
        target
    } else {
        match name {
            "token_embd.weight" | "output.weight" => GgmlType::Q6_K,
            n if n.ends_with("attn_v.weight") || n.ends_with("ffn_down.weight") => {
                match target {
                    GgmlType::Q4_K => GgmlType::Q6_K,
                    GgmlType::Q4_0 | GgmlType::Q4_1 => GgmlType::Q5_1,
                    other => other,
                }
            }
            _ => target,
        }
    };

    // Never spend more bits than the source already has.
    match (want.bits_per_weight(), from.bits_per_weight()) {
        (Some(w), Some(f)) if w >= f => from,
        _ => want,
    }
}
