//! `airun convert` -- turn a HuggingFace checkpoint into a GGUF file.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use airun_gguf::{GgmlType, GgufWriter, SafeTensors, quantize};
use anyhow::{Context, Result, bail};
use clap::Args;
use serde_json::Value as J;

#[derive(Args)]
pub struct ConvertArgs {
    /// Directory holding config.json and the .safetensors shards.
    pub input: PathBuf,

    /// Output .gguf path.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Target type for the bulk of the weights.
    #[arg(long, default_value = "F16")]
    pub r#type: String,

    /// Quantize every matrix to --type, with no per-tensor exceptions.
    #[arg(long)]
    pub pure: bool,

    /// Override the model name written into the file.
    #[arg(long)]
    pub name: Option<String>,
}

pub fn run(args: ConvertArgs) -> Result<()> {
    let target = quantize::parse_type(&args.r#type)
        .with_context(|| format!("unknown quantization type {:?}", args.r#type))?;
    if !quantize::can_encode(target) {
        bail!(
            "airun cannot write {target}. Supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, \
             Q5_1, Q8_0, Q4_K, Q5_K, Q6_K"
        );
    }

    let dir = &args.input;
    let config: J = read_json(&dir.join("config.json"))
        .context("config.json is required to know the model's shape")?;

    let arch_list = config["architectures"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let model_type = config["model_type"].as_str().unwrap_or("");

    // Only shapes the runtime can actually execute are worth writing.
    let arch = match model_type {
        "qwen3" => "qwen3",
        "qwen2" => "qwen2",
        "llama" => "llama",
        other => bail!(
            "model_type {other:?} (architectures: {arch_list:?}) is not supported by \
             `airun convert` yet.\n  \
             Supported: qwen3, qwen2 and llama-shaped dense decoders."
        ),
    };

    let st = SafeTensors::open(dir).context("failed to read the safetensors checkpoint")?;
    eprintln!(
        "read {} tensors from {} shard(s)",
        st.tensors().len(),
        st.paths().len()
    );

    let n_layers = usize_at(&config, "num_hidden_layers")?;
    let n_embd = usize_at(&config, "hidden_size")?;
    let n_head = usize_at(&config, "num_attention_heads")?;
    let n_head_kv = config["num_key_value_heads"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(n_head);
    let head_dim = config["head_dim"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(n_embd / n_head);
    let ffn_len = usize_at(&config, "intermediate_size")?;
    let ctx_len = config["max_position_embeddings"]
        .as_u64()
        .unwrap_or(4096) as usize;
    let rms_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let rope_base = config["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let tie = config["tie_word_embeddings"].as_bool().unwrap_or(false);

    let name = args
        .name
        .clone()
        .or_else(|| {
            dir.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "converted".into());

    // --- build the tensor plan ---
    let mapping = build_mapping(n_layers, tie);
    let mut plan: Vec<(String, String, GgmlType)> = Vec::new();

    for (hf_name, gguf_name) in &mapping {
        let Some(t) = st.get(hf_name) else {
            // A tied model legitimately has no lm_head; anything else is missing.
            if gguf_name == "output.weight" {
                continue;
            }
            bail!(
                "checkpoint is missing {hf_name:?}, which maps to {gguf_name:?}.\n  \
                 The download may be incomplete, or the model may not be a plain {arch} \
                 decoder."
            );
        };

        let ty = choose_type(gguf_name, t.shape.len(), target, args.pure);
        plan.push((hf_name.clone(), gguf_name.clone(), ty));
    }

    // --- write ---
    let mut w = GgufWriter::create(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;

    w.set_str("general.architecture", arch);
    w.set_str("general.name", &name);
    w.set_str("general.type", "model");
    w.set_u32("general.quantization_version", 2);
    w.set_str("general.converted_by", "airun");

    w.set_u32(format!("{arch}.block_count"), n_layers as u32);
    w.set_u32(format!("{arch}.embedding_length"), n_embd as u32);
    w.set_u32(format!("{arch}.feed_forward_length"), ffn_len as u32);
    w.set_u32(format!("{arch}.context_length"), ctx_len as u32);
    w.set_u32(format!("{arch}.attention.head_count"), n_head as u32);
    w.set_u32(format!("{arch}.attention.head_count_kv"), n_head_kv as u32);
    w.set_u32(format!("{arch}.attention.key_length"), head_dim as u32);
    w.set_u32(format!("{arch}.attention.value_length"), head_dim as u32);
    w.set_f32(format!("{arch}.attention.layer_norm_rms_epsilon"), rms_eps);
    w.set_f32(format!("{arch}.rope.freq_base"), rope_base);
    w.set_u32(format!("{arch}.rope.dimension_count"), head_dim as u32);

    write_tokenizer(&mut w, dir).context("failed to convert the tokenizer")?;

    for (_, gguf_name, ty) in &plan {
        let hf = plan
            .iter()
            .find(|(_, g, _)| g == gguf_name)
            .map(|(h, _, _)| h)
            .unwrap();
        let t = st.get(hf).unwrap();
        w.add_tensor(gguf_name.clone(), t.gguf_dims(), *ty)?;
    }
    w.write_header()?;

    let total = plan.len();
    let mut packed = Vec::new();
    let mut out_bytes = 0u64;

    for (i, (hf_name, gguf_name, ty)) in plan.iter().enumerate() {
        let t = st.get(hf_name).unwrap();
        let values = st.to_f32(t)?;

        // Rows are the fastest-varying axis, which is the last one in the
        // safetensors shape and the first in GGUF's.
        let row_len = *t.shape.last().unwrap_or(&1) as usize;

        packed.clear();
        quantize::quantize(*ty, &values, row_len, &mut packed).with_context(|| {
            format!("failed to quantize {gguf_name} ({} elements) to {ty}", values.len())
        })?;

        w.write_tensor_data(gguf_name, &packed)?;
        out_bytes += packed.len() as u64;

        eprint!(
            "\r[{}/{}] {:<44} {} -> {}   ",
            i + 1,
            total,
            gguf_name,
            t.dtype.name(),
            ty
        );
        let _ = std::io::stderr().flush();
    }

    let path = w.finish()?;
    eprintln!("\n");
    eprintln!(
        "wrote {} ({} tensors, {})",
        path.display(),
        total,
        airun_cuda::human_bytes(out_bytes)
    );
    Ok(())
}

/// Per-tensor type policy.
///
/// Norms stay f32 -- they are a rounding error's worth of space and a large
/// share of the numerical sensitivity. The embedding and output matrices keep
/// extra precision because every token passes through them, which is the same
/// reasoning behind llama.cpp's `_M` variants.
fn choose_type(gguf_name: &str, n_dims: usize, target: GgmlType, pure: bool) -> GgmlType {
    // 1-D tensors are norms and biases: never quantized.
    if n_dims < 2 || gguf_name.ends_with("_norm.weight") || gguf_name == "output_norm.weight" {
        return GgmlType::F32;
    }
    if pure || !target.is_quantized() {
        return target;
    }

    let bump = |better: GgmlType| -> GgmlType {
        // Only raise precision, never lower it below what was asked for.
        if better.bits_per_weight() > target.bits_per_weight() {
            better
        } else {
            target
        }
    };

    match gguf_name {
        "token_embd.weight" | "output.weight" => bump(GgmlType::Q6_K),
        // Value and down projections carry the most error sensitivity in
        // practice, so they get one step more precision.
        n if n.ends_with("attn_v.weight") || n.ends_with("ffn_down.weight") => {
            bump(match target {
                GgmlType::Q4_K => GgmlType::Q6_K,
                GgmlType::Q4_0 | GgmlType::Q4_1 => GgmlType::Q5_1,
                other => other,
            })
        }
        _ => target,
    }
}

/// HuggingFace tensor name -> GGUF tensor name.
fn build_mapping(n_layers: usize, tie: bool) -> Vec<(String, String)> {
    let mut m = vec![
        ("model.embed_tokens.weight".to_string(), "token_embd.weight".to_string()),
        ("model.norm.weight".to_string(), "output_norm.weight".to_string()),
    ];
    if !tie {
        m.push(("lm_head.weight".into(), "output.weight".into()));
    }

    for l in 0..n_layers {
        let pairs = [
            ("input_layernorm", "attn_norm"),
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("self_attn.q_norm", "attn_q_norm"),
            ("self_attn.k_norm", "attn_k_norm"),
            ("post_attention_layernorm", "ffn_norm"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ];
        for (hf, gg) in pairs {
            m.push((
                format!("model.layers.{l}.{hf}.weight"),
                format!("blk.{l}.{gg}.weight"),
            ));
        }
    }
    m
}

/// Copies the BPE vocabulary and merges into GGUF metadata.
fn write_tokenizer(w: &mut GgufWriter, dir: &Path) -> Result<()> {
    let tok_path = dir.join("tokenizer.json");
    let (vocab, merges, added): (BTreeMap<String, u32>, Vec<String>, Vec<(u32, String, bool)>) =
        if tok_path.exists() {
            let tok: J = read_json(&tok_path)?;
            let model = &tok["model"];

            let vocab: BTreeMap<String, u32> = model["vocab"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
                        .collect()
                })
                .unwrap_or_default();

            // Newer tokenizer.json files store merges as ["a","b"] pairs rather
            // than a single "a b" string.
            let merges: Vec<String> = model["merges"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|m| match m {
                            J::String(s) => Some(s.clone()),
                            J::Array(pair) if pair.len() == 2 => Some(format!(
                                "{} {}",
                                pair[0].as_str().unwrap_or(""),
                                pair[1].as_str().unwrap_or("")
                            )),
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default();

            let added: Vec<(u32, String, bool)> = tok["added_tokens"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|t| {
                            Some((
                                t["id"].as_u64()? as u32,
                                t["content"].as_str()?.to_string(),
                                t["special"].as_bool().unwrap_or(true),
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default();

            (vocab, merges, added)
        } else {
            // Fall back to the older vocab.json + merges.txt pair.
            let vocab: BTreeMap<String, u32> = read_json(&dir.join("vocab.json"))
                .context("neither tokenizer.json nor vocab.json is present")?
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
                        .collect()
                })
                .unwrap_or_default();

            let merges = std::fs::read_to_string(dir.join("merges.txt"))
                .map(|s| {
                    s.lines()
                        .filter(|l| !l.starts_with("#version") && !l.trim().is_empty())
                        .map(|l| l.to_string())
                        .collect()
                })
                .unwrap_or_default();

            (vocab, merges, Vec::new())
        };

    if vocab.is_empty() {
        bail!("the tokenizer has an empty vocabulary");
    }

    // Rebuild a dense id-indexed list; the JSON is keyed by token, not by id.
    let max_id = vocab
        .values()
        .copied()
        .chain(added.iter().map(|(id, _, _)| *id))
        .max()
        .unwrap_or(0) as usize;

    let mut tokens = vec![String::new(); max_id + 1];
    let mut types = vec![1i32; max_id + 1];
    for (text, id) in &vocab {
        tokens[*id as usize] = text.clone();
    }
    for (id, text, special) in &added {
        let i = *id as usize;
        tokens[i] = text.clone();
        // 3 = control, 4 = user defined.
        types[i] = if *special { 3 } else { 4 };
    }
    // Ids with no token would decode to nothing; give them a visible placeholder.
    for (i, t) in tokens.iter_mut().enumerate() {
        if t.is_empty() {
            *t = format!("<|unused_{i}|>");
            types[i] = 5; // unused
        }
    }

    let find = |needle: &str| -> Option<u32> {
        tokens.iter().position(|t| t == needle).map(|i| i as u32)
    };

    w.set_str("tokenizer.ggml.model", "gpt2");
    w.set_str("tokenizer.ggml.pre", "qwen2");
    w.set_strings("tokenizer.ggml.tokens", tokens.clone());
    w.set_i32_array("tokenizer.ggml.token_type", types);
    w.set_strings("tokenizer.ggml.merges", merges);
    w.set_bool("tokenizer.ggml.add_bos_token", false);

    // Prefer the chat end-of-turn marker; plain endoftext is the fallback.
    if let Some(id) = find("<|im_end|>").or_else(|| find("<|endoftext|>")) {
        w.set_u32("tokenizer.ggml.eos_token_id", id);
    }
    if let Some(id) = find("<|endoftext|>") {
        w.set_u32("tokenizer.ggml.bos_token_id", id);
        w.set_u32("tokenizer.ggml.padding_token_id", id);
    }

    Ok(())
}

fn read_json(path: &Path) -> Result<J> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("could not read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("{} is not valid JSON", path.display()))
}

fn usize_at(v: &J, key: &str) -> Result<usize> {
    v[key]
        .as_u64()
        .map(|x| x as usize)
        .with_context(|| format!("config.json is missing {key:?}"))
}
