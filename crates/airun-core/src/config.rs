//! Model hyperparameters, read out of GGUF metadata.
//!
//! GGUF namespaces most architectural keys under the architecture name
//! (`qwen3.block_count`, `llama.block_count`, ...), so every lookup goes through
//! [`Config::arch_key`] rather than hard-coding one prefix.

use airun_gguf::{GgufFile, Value};

use crate::error::{CoreError, Result};

/// Architectures the runtime has a forward pass for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Dense decoder: RMSNorm, GQA with per-head q/k norm, SwiGLU, NeoX RoPE.
    /// Covers `qwen3` and, with the q/k norms absent, `llama`-shaped models.
    Qwen3Dense,
    /// Qwen3-Next style hybrid: most layers are gated-delta-net linear
    /// attention, every `full_attention_interval`-th layer is gated softmax
    /// attention. The feed-forward may be dense or mixture-of-experts.
    Qwen3NextHybrid,
    /// Recognised in the file but not yet executable.
    Unsupported(&'static str),
}

impl Architecture {
    pub fn from_name(name: &str) -> Self {
        match name {
            "qwen3" => Architecture::Qwen3Dense,
            // Same block shape; the q/k norm tensors are simply absent.
            "llama" | "qwen2" => Architecture::Qwen3Dense,
            "qwen35" | "qwen3next" | "qwen35moe" | "qwen3nextmoe" => {
                Architecture::Qwen3NextHybrid
            }
            "qwen3moe" => Architecture::Unsupported(
                "dense-attention mixture-of-experts (qwen3moe) is not implemented yet",
            ),
            "clip" => Architecture::Unsupported(
                "this is a vision projector (mmproj), not a language model; pass it with --mmproj",
            ),
            _ => Architecture::Unsupported("unknown architecture"),
        }
    }
}

/// What a given block computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Softmax attention over a KV cache.
    Attention,
    /// Gated delta-net linear attention over a fixed-size recurrent state.
    LinearAttention,
}

/// Shape of the gated delta-net layers.
///
/// GGUF stores these under `ssm.*` keys inherited from Mamba, but the meaning
/// here is Qwen3-Next's: `group_count` is the number of key heads,
/// `time_step_rank` the number of value heads, and `state_size` the key head
/// width.
#[derive(Debug, Clone, Copy)]
pub struct SsmConfig {
    pub conv_kernel: usize,
    /// Total width of the value path.
    pub inner_size: usize,
    /// Key/query head width, and one side of the recurrent state.
    pub state_size: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
}

impl SsmConfig {
    pub fn head_k_dim(&self) -> usize {
        self.state_size
    }

    pub fn head_v_dim(&self) -> usize {
        if self.n_v_heads == 0 {
            0
        } else {
            self.inner_size / self.n_v_heads
        }
    }

    /// Width of the packed q/k/v projection.
    pub fn qkv_dim(&self) -> usize {
        2 * self.n_k_heads * self.head_k_dim() + self.n_v_heads * self.head_v_dim()
    }

    pub fn k_dim(&self) -> usize {
        self.n_k_heads * self.head_k_dim()
    }

    pub fn v_dim(&self) -> usize {
        self.n_v_heads * self.head_v_dim()
    }

    /// Floats of recurrent state per layer. Independent of context length --
    /// this is why hybrid models advertise very long contexts cheaply.
    pub fn state_elements(&self) -> usize {
        self.n_v_heads * self.head_k_dim() * self.head_v_dim()
    }

    /// Floats of convolution history per layer.
    pub fn conv_state_elements(&self) -> usize {
        self.qkv_dim() * self.conv_kernel.saturating_sub(1)
    }
}

/// Shape of the mixture-of-experts feed-forward.
#[derive(Debug, Clone, Copy)]
pub struct MoeConfig {
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub expert_ff: usize,
    /// 0 when the layer has no always-on shared expert.
    pub shared_expert_ff: usize,
    /// Renormalize the selected experts' weights to sum to one.
    pub norm_topk: bool,
}

/// Everything the forward pass needs to know about a model's shape.
#[derive(Debug, Clone)]
pub struct Config {
    pub architecture: Architecture,
    pub arch_name: String,
    pub name: String,

    pub n_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub ffn_len: usize,
    pub vocab_size: usize,

    /// Maximum context the weights were trained for.
    pub train_ctx: usize,

    pub rms_eps: f32,
    /// Frequency base and context scaling (linear or YaRN) of the rotary
    /// embedding.
    pub rope: airun_cuda::ops::RopeParams,
    /// Rotary dimensions; may be smaller than `head_dim` for partial rotary.
    pub rope_dim: usize,

    /// 0 when every layer attends to the full context.
    pub sliding_window: usize,

    /// True when the output projection reuses the embedding table.
    pub tied_embeddings: bool,
    /// Present on Qwen3: RMSNorm applied per head before RoPE.
    pub has_qk_norm: bool,

    /// Extra tokens predicted per step by an MTP head, if the file carries one.
    pub nextn_predict_layers: usize,

    /// Gated delta-net parameters, on hybrid models.
    pub ssm: Option<SsmConfig>,
    /// Mixture-of-experts parameters, when the feed-forward is sparse.
    pub moe: Option<MoeConfig>,
    /// Every Nth layer uses softmax attention; the rest use linear attention.
    /// 0 means every layer is softmax attention.
    pub full_attention_interval: usize,
    /// The q projection also emits a per-head sigmoid gate for the attention
    /// output, doubling its width.
    pub attn_output_gate: bool,
    /// The FFN pre-norm is named `post_attention_norm` rather than `ffn_norm`.
    pub ffn_norm_name: &'static str,

    /// Storage format of the KV cache. A runtime choice, not a file property:
    /// from_gguf leaves it at f16 and the loader applies the user's option.
    pub kv_type: airun_cuda::ops::KvType,
}

impl Config {
    /// Reads the architecture-namespaced key `<arch>.<suffix>`.
    fn arch_key<'a>(gguf: &'a GgufFile, arch: &str, suffix: &str) -> Option<&'a Value> {
        gguf.metadata().get(&format!("{arch}.{suffix}"))
    }

    fn require_usize(gguf: &GgufFile, arch: &str, suffix: &str) -> Result<usize> {
        Self::arch_key(gguf, arch, suffix)
            .and_then(|v| v.as_usize())
            .ok_or_else(|| CoreError::MissingConfig {
                key: format!("{arch}.{suffix}"),
            })
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch_name = gguf
            .architecture()
            .ok_or_else(|| CoreError::MissingConfig {
                key: "general.architecture".into(),
            })?
            .to_string();

        let architecture = Architecture::from_name(&arch_name);

        let n_layers = Self::require_usize(gguf, &arch_name, "block_count")?;
        let n_embd = Self::require_usize(gguf, &arch_name, "embedding_length")?;
        let n_head = Self::require_usize(gguf, &arch_name, "attention.head_count")?;
        let n_head_kv =
            Self::arch_key(gguf, &arch_name, "attention.head_count_kv")
                .and_then(|v| v.as_usize())
                // No GQA key means multi-head attention.
                .unwrap_or(n_head);

        // key_length is authoritative when present; otherwise heads divide the
        // embedding evenly, which is the pre-GQA convention.
        let head_dim = Self::arch_key(gguf, &arch_name, "attention.key_length")
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| if n_head > 0 { n_embd / n_head } else { 0 });

        // A pure mixture-of-experts model has no dense feed-forward, so it does
        // not declare a width for one. Fall back to the expert width, which is
        // what the scratch buffers need to cover.
        let ffn_len = Self::arch_key(gguf, &arch_name, "feed_forward_length")
            .and_then(|v| v.as_usize())
            .or_else(|| {
                Self::arch_key(gguf, &arch_name, "expert_feed_forward_length")
                    .and_then(|v| v.as_usize())
            })
            .ok_or_else(|| CoreError::MissingConfig {
                key: format!("{arch_name}.feed_forward_length"),
            })?;

        let rms_eps = Self::arch_key(gguf, &arch_name, "attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-6);
        let rope_dim = Self::arch_key(gguf, &arch_name, "rope.dimension_count")
            .and_then(|v| v.as_usize())
            .unwrap_or(head_dim);

        let train_ctx = Self::arch_key(gguf, &arch_name, "context_length")
            .and_then(|v| v.as_usize())
            .unwrap_or(4096);

        let rope = Self::rope_params(gguf, &arch_name, rope_dim)?;

        let sliding_window = Self::arch_key(gguf, &arch_name, "attention.sliding_window")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);

        let nextn_predict_layers = Self::arch_key(gguf, &arch_name, "nextn_predict_layers")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);

        // The vocabulary size comes from the token list; the embedding tensor is
        // the fallback for files that ship a tokenizer separately.
        let vocab_size = gguf
            .metadata()
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .or_else(|| {
                gguf.tensor("token_embd.weight")
                    .and_then(|t| t.dims.get(1).copied())
                    .map(|d| d as usize)
            })
            .ok_or_else(|| CoreError::MissingConfig {
                key: "tokenizer.ggml.tokens".into(),
            })?;

        // No `output.weight` means the model reuses the embedding table for the
        // final projection.
        let tied_embeddings = gguf.tensor("output.weight").is_none();
        // Hybrids put their q/k norms only on the softmax-attention layers, so
        // probe by tensor name across all layers rather than assuming layer 0.
        let has_qk_norm = (0..n_layers)
            .any(|l| gguf.tensor(&format!("blk.{l}.attn_q_norm.weight")).is_some());

        // Gated delta-net layers are present when the ssm.* keys are.
        let ssm = Self::arch_key(gguf, &arch_name, "ssm.inner_size")
            .and_then(|v| v.as_usize())
            .map(|inner_size| SsmConfig {
                conv_kernel: Self::arch_key(gguf, &arch_name, "ssm.conv_kernel")
                    .and_then(|v| v.as_usize())
                    .unwrap_or(4),
                inner_size,
                state_size: Self::arch_key(gguf, &arch_name, "ssm.state_size")
                    .and_then(|v| v.as_usize())
                    .unwrap_or(128),
                n_k_heads: Self::arch_key(gguf, &arch_name, "ssm.group_count")
                    .and_then(|v| v.as_usize())
                    .unwrap_or(1),
                n_v_heads: Self::arch_key(gguf, &arch_name, "ssm.time_step_rank")
                    .and_then(|v| v.as_usize())
                    .unwrap_or(1),
            });

        let moe = Self::arch_key(gguf, &arch_name, "expert_count")
            .and_then(|v| v.as_usize())
            .filter(|&n| n > 0)
            .map(|n_experts| MoeConfig {
                n_experts,
                n_experts_used: Self::arch_key(gguf, &arch_name, "expert_used_count")
                    .and_then(|v| v.as_usize())
                    .unwrap_or(1),
                expert_ff: Self::arch_key(gguf, &arch_name, "expert_feed_forward_length")
                    .and_then(|v| v.as_usize())
                    .unwrap_or(ffn_len),
                shared_expert_ff: Self::arch_key(
                    gguf,
                    &arch_name,
                    "expert_shared_feed_forward_length",
                )
                .and_then(|v| v.as_usize())
                .unwrap_or(0),
                // Qwen-family routers renormalize; the key is only present when
                // a converter chose to record it.
                norm_topk: Self::arch_key(gguf, &arch_name, "expert_weights_norm")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
            });

        let full_attention_interval = Self::arch_key(gguf, &arch_name, "full_attention_interval")
            .and_then(|v| v.as_usize())
            .unwrap_or(0);

        // A q projection twice as wide as the heads need is carrying an output
        // gate alongside the queries.
        let attn_output_gate = (0..n_layers)
            .find_map(|l| gguf.tensor(&format!("blk.{l}.attn_q.weight")))
            .map(|t| t.n_rows() as usize == 2 * n_head * head_dim)
            .unwrap_or(false);

        // Hybrids name the FFN pre-norm after the block it follows.
        let ffn_norm_name = if (0..n_layers)
            .any(|l| gguf.tensor(&format!("blk.{l}.ffn_norm.weight")).is_some())
        {
            "ffn_norm"
        } else {
            "post_attention_norm"
        };

        let cfg = Config {
            architecture,
            arch_name,
            name: gguf
                .metadata()
                .get("general.name")
                .and_then(|v| v.as_str())
                .unwrap_or("unnamed")
                .to_string(),
            n_layers,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            ffn_len,
            vocab_size,
            train_ctx,
            rms_eps,
            rope,
            rope_dim,
            sliding_window,
            tied_embeddings,
            has_qk_norm,
            nextn_predict_layers,
            ssm,
            moe,
            full_attention_interval,
            attn_output_gate,
            ffn_norm_name,
            kv_type: Default::default(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// RoPE frequencies, including the context-extension scaling the file
    /// declares. `rope.scaling.type` decides how `rope.scaling.factor` applies:
    /// linear interpolation divides every frequency, YaRN only the slow ones
    /// (and rescales attention), so the two are not interchangeable even at
    /// short contexts.
    fn rope_params(gguf: &GgufFile, arch: &str, rope_dim: usize) -> Result<airun_cuda::ops::RopeParams> {
        use airun_cuda::ops::RopeParams;
        let f32_key = |k: &str| Self::arch_key(gguf, arch, k).and_then(|v| v.as_f32());
        let base = f32_key("rope.freq_base").unwrap_or(10_000.0);
        let factor = f32_key("rope.scaling.factor").filter(|f| *f > 0.0).unwrap_or(1.0);
        // AIRUN_ROPE_SCALING=none|linear|yarn overrides the file, like
        // llama.cpp's --rope-scaling; for A/B checks against it.
        let kind = std::env::var("AIRUN_ROPE_SCALING").ok()
            .or_else(|| Self::arch_key(gguf, arch, "rope.scaling.type").and_then(|v| v.as_str().map(str::to_string)))
            .unwrap_or_else(|| if factor == 1.0 { "none".into() } else { "linear".into() });
        match kind.as_str() {
            "none" => Ok(RopeParams::plain(base, 1.0)),
            "linear" => Ok(RopeParams::plain(base, 1.0 / factor)),
            "yarn" => {
                let orig = Self::arch_key(gguf, arch, "rope.scaling.original_context_length")
                    .and_then(|v| v.as_usize())
                    .ok_or_else(|| CoreError::MissingConfig {
                        key: format!("{arch}.rope.scaling.original_context_length"),
                    })?;
                if f32_key("rope.scaling.yarn_log_multiplier").is_some_and(|m| m != 0.0) {
                    return Err(CoreError::Other(format!(
                        "{arch}.rope.scaling.yarn_log_multiplier is set (DeepSeek-style YaRN), which airun does not                          implement; remove the key from the file only if the model card says plain YaRN"
                    )));
                }
                let beta_fast = f32_key("rope.scaling.yarn_beta_fast").unwrap_or(32.0);
                let beta_slow = f32_key("rope.scaling.yarn_beta_slow").unwrap_or(1.0);
                let attn_factor = f32_key("rope.scaling.attn_factor").unwrap_or(1.0);
                let mut p = RopeParams::yarn(base, factor, rope_dim, orig, beta_fast, beta_slow, attn_factor);
                if let Some(ext) = f32_key("rope.scaling.yarn_ext_factor").filter(|e| *e >= 0.0) {
                    p.ext_factor = ext;
                }
                if let Some(m) = std::env::var("AIRUN_ROPE_MSCALE").ok().and_then(|v| v.parse().ok()) {
                    p.mscale = m;
                }
                Ok(p)
            }
            other => Err(CoreError::Other(format!(
                "{arch}.rope.scaling.type is \"{other}\", which airun does not implement (it knows none, linear                  and yarn); this file needs a newer airun"
            ))),
        }
    }

    /// Layers the forward pass actually runs.
    ///
    /// `block_count` includes the multi-token-prediction blocks appended at the
    /// end. Those are a speculative-decoding head, not part of the main
    /// computation, so running them would corrupt the output.
    pub fn n_run_layers(&self) -> usize {
        self.n_layers.saturating_sub(self.nextn_predict_layers)
    }

    /// Whether block `layer` is softmax attention or linear attention.
    pub fn layer_kind(&self, layer: usize) -> LayerKind {
        if self.ssm.is_none() || self.full_attention_interval == 0 {
            return LayerKind::Attention;
        }
        // Attention lands on the last layer of each group, so a group is always
        // "some linear layers, then one attention layer".
        if (layer + 1) % self.full_attention_interval == 0 {
            LayerKind::Attention
        } else {
            LayerKind::LinearAttention
        }
    }

    pub fn n_attention_layers(&self) -> usize {
        (0..self.n_run_layers())
            .filter(|&l| self.layer_kind(l) == LayerKind::Attention)
            .count()
    }

    pub fn n_linear_layers(&self) -> usize {
        self.n_run_layers() - self.n_attention_layers()
    }

    pub fn is_hybrid(&self) -> bool {
        self.ssm.is_some()
    }

    /// Catches shapes the kernels cannot execute, before any memory is allocated.
    fn validate(&self) -> Result<()> {
        let mut problems = Vec::new();
        if self.n_head == 0 || self.n_head_kv == 0 {
            problems.push("head counts must be non-zero".to_string());
        } else if self.n_head % self.n_head_kv != 0 {
            problems.push(format!(
                "head_count {} is not a multiple of head_count_kv {}",
                self.n_head, self.n_head_kv
            ));
        }
        if self.head_dim == 0 || self.head_dim % 32 != 0 {
            problems.push(format!(
                "head dimension {} must be a non-zero multiple of 32",
                self.head_dim
            ));
        }
        if self.head_dim > 256 {
            problems.push(format!(
                "head dimension {} exceeds the 256 the attention kernel supports",
                self.head_dim
            ));
        }
        if self.n_embd % 32 != 0 {
            problems.push(format!(
                "embedding length {} must be a multiple of 32",
                self.n_embd
            ));
        }
        if !problems.is_empty() {
            return Err(CoreError::UnsupportedShape {
                model: self.name.clone(),
                problems,
            });
        }
        Ok(())
    }

    /// Attention softmax scale.
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Queries per key/value head.
    pub fn gqa_ratio(&self) -> usize {
        self.n_head / self.n_head_kv
    }

    /// Width of the packed Q projection output.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Width of the packed K (or V) projection output.
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// Bytes of KV cache for one layer at `ctx` tokens, both K and V.
    pub fn kv_bytes_per_layer(&self, ctx: usize) -> u64 {
        2 * (self.n_head_kv * ctx) as u64 * self.kv_type.row_bytes(self.head_dim) as u64
    }

    /// KV cache across the whole model.
    ///
    /// On hybrids only the softmax-attention layers hold a KV cache; the linear
    /// layers keep a fixed-size recurrent state instead, which is why a hybrid
    /// at 262144 context costs a fraction of what a dense model would.
    pub fn kv_bytes_total(&self, ctx: usize) -> u64 {
        self.kv_bytes_per_layer(ctx) * self.n_attention_layers().max(1) as u64
    }

    /// Recurrent and convolution state bytes for one linear-attention layer.
    pub fn ssm_bytes_per_layer(&self) -> u64 {
        match &self.ssm {
            Some(s) => ((s.state_elements() + s.conv_state_elements()) * 4) as u64,
            None => 0,
        }
    }

    pub fn ssm_bytes_total(&self) -> u64 {
        self.ssm_bytes_per_layer() * self.n_linear_layers() as u64
    }

    /// Human summary for `airun inspect` and load-time logging.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} ({}) - {} layers, {} embd, {} heads",
            self.name, self.arch_name, self.n_layers, self.n_embd, self.n_head
        );
        if self.n_head_kv != self.n_head {
            s.push_str(&format!(" ({} kv, GQA {}x)", self.n_head_kv, self.gqa_ratio()));
        }
        s.push_str(&format!(
            ", head_dim {}, ffn {}, vocab {}",
            self.head_dim, self.ffn_len, self.vocab_size
        ));
        if self.tied_embeddings {
            s.push_str(", tied embeddings");
        }
        if self.has_qk_norm {
            s.push_str(", q/k norm");
        }
        if self.nextn_predict_layers > 0 {
            s.push_str(&format!(
                ", {} MTP layer(s) skipped",
                self.nextn_predict_layers
            ));
        }
        let r = &self.rope;
        if r.ext_factor != 0.0 {
            s.push_str(&format!(
                "\n  rope: {} of {} dims, base {}, YaRN x{} (dims {}..{} ramp, mscale {:.4})",
                self.rope_dim, self.head_dim, r.base, 1.0 / r.freq_scale, r.corr_lo, r.corr_hi, r.mscale
            ));
        } else if r.freq_scale != 1.0 {
            s.push_str(&format!("\n  rope: {} of {} dims, base {}, linear x{}",
                self.rope_dim, self.head_dim, r.base, 1.0 / r.freq_scale));
        }
        if let Some(ssm) = &self.ssm {
            s.push_str(&format!(
                "\n  hybrid: {} linear-attention + {} attention layers; delta-net {} v-heads x \
                 {} k-heads, state {}x{}, conv {}",
                self.n_linear_layers(),
                self.n_attention_layers(),
                ssm.n_v_heads,
                ssm.n_k_heads,
                ssm.head_k_dim(),
                ssm.head_v_dim(),
                ssm.conv_kernel
            ));
        }
        if let Some(moe) = &self.moe {
            s.push_str(&format!(
                "\n  MoE: {} experts, {} active, expert ffn {}",
                moe.n_experts, moe.n_experts_used, moe.expert_ff
            ));
            if moe.shared_expert_ff > 0 {
                s.push_str(&format!(", shared expert ffn {}", moe.shared_expert_ff));
            }
        }
        if self.attn_output_gate {
            s.push_str("\n  gated attention output");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_1_7b() -> Config {
        Config {
            architecture: Architecture::Qwen3Dense,
            arch_name: "qwen3".into(),
            name: "test".into(),
            n_layers: 28,
            n_embd: 2048,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 128,
            ffn_len: 6144,
            vocab_size: 151936,
            train_ctx: 40960,
            rms_eps: 1e-6,
            rope: airun_cuda::ops::RopeParams::plain(1e6, 1.0),
            rope_dim: 128,
            sliding_window: 0,
            tied_embeddings: true,
            has_qk_norm: true,
            nextn_predict_layers: 0,
            ssm: None,
            moe: None,
            full_attention_interval: 0,
            attn_output_gate: false,
            ffn_norm_name: "ffn_norm",
            kv_type: Default::default(),
        }
    }

    /// Shaped like Qwen3.5-9B: 33 blocks of which the last is an MTP head, and
    /// softmax attention on every fourth layer.
    fn qwen35_9b() -> Config {
        Config {
            architecture: Architecture::Qwen3NextHybrid,
            arch_name: "qwen35".into(),
            n_layers: 33,
            n_head: 16,
            n_head_kv: 4,
            head_dim: 256,
            nextn_predict_layers: 1,
            full_attention_interval: 4,
            attn_output_gate: true,
            ffn_norm_name: "post_attention_norm",
            kv_type: Default::default(),
            ssm: Some(SsmConfig {
                conv_kernel: 4,
                inner_size: 4096,
                state_size: 128,
                n_k_heads: 16,
                n_v_heads: 32,
            }),
            ..qwen3_1_7b()
        }
    }

    #[test]
    fn hybrid_layer_schedule_matches_the_file() {
        let c = qwen35_9b();
        // The MTP block is excluded from the main path.
        assert_eq!(c.n_run_layers(), 32);
        // Attention on 3, 7, 11, ... 31 -- the last layer of each group of four.
        let attn: Vec<usize> = (0..c.n_run_layers())
            .filter(|&l| c.layer_kind(l) == LayerKind::Attention)
            .collect();
        assert_eq!(attn, vec![3, 7, 11, 15, 19, 23, 27, 31]);
        assert_eq!(c.n_attention_layers(), 8);
        assert_eq!(c.n_linear_layers(), 24);
    }

    #[test]
    fn delta_net_dimensions_match_the_projection_widths() {
        let ssm = qwen35_9b().ssm.unwrap();
        assert_eq!(ssm.head_v_dim(), 128);
        assert_eq!(ssm.head_k_dim(), 128);
        assert_eq!(ssm.k_dim(), 2048);
        assert_eq!(ssm.v_dim(), 4096);
        // The packed qkv projection in the file is 4096 -> 8192.
        assert_eq!(ssm.qkv_dim(), 8192);
        // State is independent of context length.
        assert_eq!(ssm.state_elements(), 32 * 128 * 128);
        assert_eq!(ssm.conv_state_elements(), 8192 * 3);
    }

    #[test]
    fn hybrid_kv_cache_only_counts_attention_layers() {
        let c = qwen35_9b();
        // 8 attention layers, not 33: the linear layers hold a constant-size
        // state instead.
        assert_eq!(c.kv_bytes_total(4096), c.kv_bytes_per_layer(4096) * 8);
        assert!(c.ssm_bytes_total() > 0);
        // At long context the KV cache dominates; the SSM state never grows.
        assert_eq!(c.ssm_bytes_total(), c.ssm_bytes_per_layer() * 24);
    }

    #[test]
    fn derived_dimensions_match_the_model() {
        let c = qwen3_1_7b();
        assert_eq!(c.q_dim(), 2048);
        assert_eq!(c.kv_dim(), 1024);
        assert_eq!(c.gqa_ratio(), 2);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn kv_cache_size_is_exact() {
        let c = qwen3_1_7b();
        // 2 (K and V) * 8 heads * 4096 ctx * 128 dim * 2 bytes = 16 MiB per layer
        assert_eq!(c.kv_bytes_per_layer(4096), 16 * 1024 * 1024);
        assert_eq!(c.kv_bytes_total(4096), 28 * 16 * 1024 * 1024);
    }

    #[test]
    fn validation_rejects_shapes_the_kernels_cannot_run() {
        let mut c = qwen3_1_7b();
        c.head_dim = 100; // not a multiple of 32
        assert!(c.validate().is_err());

        let mut c = qwen3_1_7b();
        c.head_dim = 512; // past the attention kernel's limit
        assert!(c.validate().is_err());

        let mut c = qwen3_1_7b();
        c.n_head_kv = 5; // does not divide head_count
        assert!(c.validate().is_err());
    }
}
