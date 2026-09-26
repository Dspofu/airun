//! Loading model weights onto the GPUs described by a [`Placement`].
//!
//! Quantized tensors are uploaded in their packed form and stay that way: the
//! matmul kernels decode blocks straight into registers, so nothing is ever
//! expanded to f32 in device memory. Norm weights and the small delta-net
//! parameters are the exception -- they are tiny and read every layer, so they
//! are widened once at load time.

use airun_cuda::{DeviceBuffer, Stream};
use airun_gguf::{GgufFile, TensorInfo};

use crate::config::{Config, LayerKind};
use crate::error::{CoreError, Result};
use crate::plan::Placement;

/// A weight matrix on a device, kept in its on-disk layout.
pub struct GpuTensor {
    pub data: DeviceBuffer<u8>,
    pub ty: airun_gguf::GgmlType,
    /// Row length: the matmul's inner dimension.
    pub k: usize,
    /// Number of rows in one expert (equal to the total for non-expert tensors).
    pub rows: usize,
    pub row_bytes: usize,
    /// Bytes between consecutive experts; 0 when the tensor is not expert-indexed.
    pub expert_stride: usize,
    pub n_experts: usize,
    pub device: i32,
}

impl GpuTensor {
    pub fn kernel_type(&self) -> airun_cuda::ops::KernelType {
        airun_cuda::ops::KernelType(self.ty.id() as i32)
    }

    pub fn bytes(&self) -> usize {
        self.data.bytes()
    }
}

/// Softmax-attention block weights.
pub struct AttentionWeights {
    /// Twice as wide as the heads need when the model gates its output.
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub output: GpuTensor,
    pub q_norm: Option<DeviceBuffer<f32>>,
    pub k_norm: Option<DeviceBuffer<f32>>,
}

/// Gated delta-net (linear attention) block weights.
pub struct LinearAttentionWeights {
    /// Packed q/k/v projection.
    pub qkv: GpuTensor,
    /// Output gate projection.
    pub gate: GpuTensor,
    /// Depthwise convolution taps, [channels][kernel].
    pub conv1d: DeviceBuffer<f32>,
    /// Per-head log decay rate.
    pub a_log: DeviceBuffer<f32>,
    pub dt_bias: DeviceBuffer<f32>,
    /// Projections producing the per-token decay and beta coefficients.
    pub alpha: GpuTensor,
    pub beta: GpuTensor,
    /// RMSNorm applied per value head before the output projection.
    pub norm: DeviceBuffer<f32>,
    pub out: GpuTensor,
}

/// The always-on expert every token passes through.
pub struct SharedExpert {
    /// Produces one logit per token, gating how much the shared expert counts.
    pub router: GpuTensor,
    pub gate: GpuTensor,
    pub up: GpuTensor,
    pub down: GpuTensor,
}

/// Feed-forward, dense or sparse.
pub enum FfnWeights {
    Dense {
        gate: GpuTensor,
        up: GpuTensor,
        down: GpuTensor,
    },
    Moe {
        router: GpuTensor,
        gate_exps: GpuTensor,
        up_exps: GpuTensor,
        down_exps: GpuTensor,
        shared: Option<SharedExpert>,
    },
}

pub enum Block {
    Attention(AttentionWeights),
    Linear(LinearAttentionWeights),
}

pub struct LayerWeights {
    pub device: i32,
    pub attn_norm: DeviceBuffer<f32>,
    pub block: Block,
    pub ffn_norm: DeviceBuffer<f32>,
    pub ffn: FfnWeights,
}

/// The loaded model.
pub struct Model {
    pub config: Config,
    pub placement: Placement,
    pub layers: Vec<LayerWeights>,
    pub token_embd: GpuTensor,
    pub output_norm: DeviceBuffer<f32>,
    /// The output projection. When embeddings are tied this is a second copy of
    /// the embedding table on the output device.
    pub output: GpuTensor,
    pub total_bytes: u64,
}

impl Model {
    /// Uploads every weight named by `placement`.
    ///
    /// `progress` is called with (layers done, total) so the caller can show
    /// something during the multi-second upload of a large model.
    pub fn load(
        gguf: &GgufFile,
        config: Config,
        placement: Placement,
        streams: &[(i32, Stream)],
        mut progress: impl FnMut(usize, usize),
    ) -> Result<Self> {
        let stream_for = |dev: i32| -> Result<&Stream> {
            streams
                .iter()
                .find(|(d, _)| *d == dev)
                .map(|(_, s)| s)
                .ok_or_else(|| CoreError::Other(format!("no stream for GPU {dev}")))
        };

        let n_run = config.n_run_layers();
        let total_steps = n_run + 3;
        let mut done = 0usize;
        let mut total_bytes = 0u64;

        let embed_dev = placement.embedding_device();
        let out_dev = placement.output_device();

        let token_embd = load_packed(gguf, "token_embd.weight", embed_dev, stream_for(embed_dev)?, &config)?;
        total_bytes += token_embd.bytes() as u64;
        done += 1;
        progress(done, total_steps);

        let output_norm = load_f32(gguf, "output_norm.weight", out_dev, stream_for(out_dev)?)?;
        done += 1;
        progress(done, total_steps);

        // A tied model has no `output.weight`; the embedding table doubles as
        // the projection. When the two live on different GPUs it has to be
        // uploaded twice, which the planner already accounted for.
        let output = if config.tied_embeddings {
            if out_dev == embed_dev {
                GpuTensor {
                    data: DeviceBuffer::<u8>::new(out_dev, 0)?,
                    ty: token_embd.ty,
                    k: token_embd.k,
                    rows: token_embd.rows,
                    row_bytes: token_embd.row_bytes,
                    expert_stride: 0,
                    n_experts: 1,
                    device: out_dev,
                }
            } else {
                let t = load_packed(gguf, "token_embd.weight", out_dev, stream_for(out_dev)?, &config)?;
                total_bytes += t.bytes() as u64;
                t
            }
        } else {
            let t = load_packed(gguf, "output.weight", out_dev, stream_for(out_dev)?, &config)?;
            total_bytes += t.bytes() as u64;
            t
        };
        done += 1;
        progress(done, total_steps);

        let mut layers = Vec::with_capacity(n_run);
        for l in 0..n_run {
            let dev = placement
                .device_for_layer(l)
                .ok_or_else(|| CoreError::Other(format!("layer {l} was not placed on any GPU")))?;
            let stream = stream_for(dev)?;
            let name = |suffix: &str| format!("blk.{l}.{suffix}");

            let block = match config.layer_kind(l) {
                LayerKind::Attention => Block::Attention(AttentionWeights {
                    q: load_packed(gguf, &name("attn_q.weight"), dev, stream, &config)?,
                    k: load_packed(gguf, &name("attn_k.weight"), dev, stream, &config)?,
                    v: load_packed(gguf, &name("attn_v.weight"), dev, stream, &config)?,
                    output: load_packed(gguf, &name("attn_output.weight"), dev, stream, &config)?,
                    q_norm: load_f32_optional(gguf, &name("attn_q_norm.weight"), dev, stream)?,
                    k_norm: load_f32_optional(gguf, &name("attn_k_norm.weight"), dev, stream)?,
                }),
                LayerKind::LinearAttention => Block::Linear(LinearAttentionWeights {
                    qkv: load_packed(gguf, &name("attn_qkv.weight"), dev, stream, &config)?,
                    gate: load_packed(gguf, &name("attn_gate.weight"), dev, stream, &config)?,
                    conv1d: load_f32(gguf, &name("ssm_conv1d.weight"), dev, stream)?,
                    a_log: load_delta_decay_coeff(gguf, &name("ssm_a"), dev, stream)?,
                    dt_bias: load_f32(gguf, &name("ssm_dt.bias"), dev, stream)?,
                    alpha: load_packed(gguf, &name("ssm_alpha.weight"), dev, stream, &config)?,
                    beta: load_packed(gguf, &name("ssm_beta.weight"), dev, stream, &config)?,
                    norm: load_f32(gguf, &name("ssm_norm.weight"), dev, stream)?,
                    out: load_packed(gguf, &name("ssm_out.weight"), dev, stream, &config)?,
                }),
            };

            let ffn = if config.moe.is_some() {
                let shared = if gguf.tensor(&name("ffn_gate_shexp.weight")).is_some() {
                    Some(SharedExpert {
                        router: load_packed(gguf, &name("ffn_gate_inp_shexp.weight"), dev, stream, &config)?,
                        gate: load_packed(gguf, &name("ffn_gate_shexp.weight"), dev, stream, &config)?,
                        up: load_packed(gguf, &name("ffn_up_shexp.weight"), dev, stream, &config)?,
                        down: load_packed(gguf, &name("ffn_down_shexp.weight"), dev, stream, &config)?,
                    })
                } else {
                    None
                };
                FfnWeights::Moe {
                    router: load_packed(gguf, &name("ffn_gate_inp.weight"), dev, stream, &config)?,
                    gate_exps: load_packed(gguf, &name("ffn_gate_exps.weight"), dev, stream, &config)?,
                    up_exps: load_packed(gguf, &name("ffn_up_exps.weight"), dev, stream, &config)?,
                    down_exps: load_packed(gguf, &name("ffn_down_exps.weight"), dev, stream, &config)?,
                    shared,
                }
            } else {
                FfnWeights::Dense {
                    gate: load_packed(gguf, &name("ffn_gate.weight"), dev, stream, &config)?,
                    up: load_packed(gguf, &name("ffn_up.weight"), dev, stream, &config)?,
                    down: load_packed(gguf, &name("ffn_down.weight"), dev, stream, &config)?,
                }
            };

            let w = LayerWeights {
                device: dev,
                attn_norm: load_f32(gguf, &name("attn_norm.weight"), dev, stream)?,
                block,
                ffn_norm: load_f32(gguf, &name(&format!("{}.weight", config.ffn_norm_name)), dev, stream)?,
                ffn,
            };
            total_bytes += layer_bytes(&w);
            layers.push(w);

            done += 1;
            progress(done, total_steps);
        }

        // Verify the shapes actually match what the metadata promised. A file
        // whose header and tensors disagree would otherwise produce garbage.
        validate_shapes(&config, &layers, &token_embd)?;

        for (_, s) in streams {
            s.synchronize()?;
        }

        Ok(Self {
            config,
            placement,
            layers,
            token_embd,
            output_norm,
            output,
            total_bytes,
        })
    }

    /// The tensor used for the final projection to logits.
    pub fn output_tensor(&self) -> &GpuTensor {
        if self.config.tied_embeddings && self.output.data.is_empty() {
            &self.token_embd
        } else {
            &self.output
        }
    }
}

fn layer_bytes(w: &LayerWeights) -> u64 {
    let block = match &w.block {
        Block::Attention(a) => a.q.bytes() + a.k.bytes() + a.v.bytes() + a.output.bytes(),
        Block::Linear(l) => {
            l.qkv.bytes() + l.gate.bytes() + l.alpha.bytes() + l.beta.bytes() + l.out.bytes()
        }
    };
    let ffn = match &w.ffn {
        FfnWeights::Dense { gate, up, down } => gate.bytes() + up.bytes() + down.bytes(),
        FfnWeights::Moe {
            router,
            gate_exps,
            up_exps,
            down_exps,
            shared,
        } => {
            router.bytes()
                + gate_exps.bytes()
                + up_exps.bytes()
                + down_exps.bytes()
                + shared.as_ref().map_or(0, |s| {
                    s.router.bytes() + s.gate.bytes() + s.up.bytes() + s.down.bytes()
                })
        }
    };
    (block + ffn) as u64
}

fn tensor_info<'a>(gguf: &'a GgufFile, name: &str, arch: &str) -> Result<&'a TensorInfo> {
    gguf.tensor(name).ok_or_else(|| CoreError::MissingTensor {
        name: name.to_string(),
        arch: arch.to_string(),
    })
}

/// Uploads a tensor in its packed form.
///
/// A three-dimensional tensor is an expert stack: the trailing axis indexes
/// experts, and `rows` becomes the per-expert row count.
fn load_packed(
    gguf: &GgufFile,
    name: &str,
    device: i32,
    stream: &Stream,
    config: &Config,
) -> Result<GpuTensor> {
    let info = tensor_info(gguf, name, &config.arch_name)?;

    if !airun_cuda::ops::KernelType(info.ty.id() as i32).is_supported() {
        return Err(CoreError::UnsupportedTensorType {
            name: name.to_string(),
            ty: info.ty.to_string(),
        });
    }

    let k = info.row_size() as usize;
    if k % 32 != 0 {
        return Err(CoreError::ShapeMismatch {
            name: name.to_string(),
            found: info.dims.clone(),
            expected: vec![32],
        });
    }

    let layout = info.ty.layout().expect("supported types have a layout");
    let row_bytes = (k / layout.elements as usize) * layout.bytes as usize;

    let (rows, n_experts) = match info.dims.len() {
        3 => (info.dims[1] as usize, info.dims[2] as usize),
        _ => (info.n_rows() as usize, 1),
    };
    let expert_stride = if n_experts > 1 { rows * row_bytes } else { 0 };

    let bytes = gguf.tensor_bytes(info)?;
    let data = DeviceBuffer::<u8>::from_slice(device, bytes, stream)?;

    Ok(GpuTensor {
        data,
        ty: info.ty,
        k,
        rows,
        row_bytes,
        expert_stride,
        n_experts,
        device,
    })
}

/// Uploads a tensor widened to f32. Used for norms and delta-net coefficients.
fn load_f32(gguf: &GgufFile, name: &str, device: i32, stream: &Stream) -> Result<DeviceBuffer<f32>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| CoreError::MissingTensor {
            name: name.to_string(),
            arch: String::new(),
        })?;
    let n = info.n_elements() as usize;
    let mut host = vec![0f32; n];
    airun_gguf::dequant::dequantize(info.ty, gguf.tensor_bytes(info)?, n, &mut host)?;
    Ok(DeviceBuffer::<f32>::from_slice(device, &host, stream)?)
}

/// Loads the delta-net decay coefficient, normalized to its negative form.
///
/// The reference formulation is `g = -exp(A_log) * softplus(...)`, but converters
/// disagree about what lands in the file: llama.cpp's applies `-exp()` during
/// conversion and stores the result, while a converter that copies the raw
/// parameter leaves `A_log` behind. The two are told apart by sign -- `A_log` is
/// a log of a positive rate and `-exp(A_log)` is strictly negative.
///
/// Guessing wrong is not a crash. It exponentiates the decay a second time, the
/// state forgets almost everything within a couple of tokens, and the model
/// emits text that is locally fluent and globally meaningless.
fn load_delta_decay_coeff(
    gguf: &GgufFile,
    name: &str,
    device: i32,
    stream: &Stream,
) -> Result<DeviceBuffer<f32>> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| CoreError::MissingTensor {
            name: name.to_string(),
            arch: String::new(),
        })?;
    let n = info.n_elements() as usize;
    let mut host = vec![0f32; n];
    airun_gguf::dequant::dequantize(info.ty, gguf.tensor_bytes(info)?, n, &mut host)?;

    // AIRUN_SSM_A_IS_LOG=1 overrides the detection and treats the stored values
    // as raw `A_log` even when they are already negative.
    let already_negative =
        host.iter().all(|&v| v < 0.0) && std::env::var("AIRUN_SSM_A_IS_LOG").is_err();
    if !already_negative {
        for v in host.iter_mut() {
            *v = -v.exp();
        }
    }

    Ok(DeviceBuffer::<f32>::from_slice(device, &host, stream)?)
}

fn load_f32_optional(
    gguf: &GgufFile,
    name: &str,
    device: i32,
    stream: &Stream,
) -> Result<Option<DeviceBuffer<f32>>> {
    if gguf.tensor(name).is_none() {
        return Ok(None);
    }
    load_f32(gguf, name, device, stream).map(Some)
}

/// Cross-checks loaded tensor shapes against the config derived from metadata.
fn validate_shapes(cfg: &Config, layers: &[LayerWeights], embd: &GpuTensor) -> Result<()> {
    if embd.k != cfg.n_embd {
        return Err(CoreError::ShapeMismatch {
            name: "token_embd.weight".into(),
            found: vec![embd.k as u64, embd.rows as u64],
            expected: vec![cfg.n_embd as u64, cfg.vocab_size as u64],
        });
    }

    let mut check = |name: String, found: (usize, usize), expected: (usize, usize)| -> Result<()> {
        if found != expected {
            return Err(CoreError::ShapeMismatch {
                name,
                found: vec![found.0 as u64, found.1 as u64],
                expected: vec![expected.0 as u64, expected.1 as u64],
            });
        }
        Ok(())
    };

    for (i, layer) in layers.iter().enumerate() {
        match &layer.block {
            Block::Attention(a) => {
                // A gated model's q projection carries the gate alongside the
                // queries, so it is twice as wide.
                let q_width = if cfg.attn_output_gate {
                    cfg.q_dim() * 2
                } else {
                    cfg.q_dim()
                };
                check(format!("blk.{i}.attn_q.weight"), (a.q.k, a.q.rows), (cfg.n_embd, q_width))?;
                check(format!("blk.{i}.attn_k.weight"), (a.k.k, a.k.rows), (cfg.n_embd, cfg.kv_dim()))?;
                check(format!("blk.{i}.attn_v.weight"), (a.v.k, a.v.rows), (cfg.n_embd, cfg.kv_dim()))?;
                check(
                    format!("blk.{i}.attn_output.weight"),
                    (a.output.k, a.output.rows),
                    (cfg.q_dim(), cfg.n_embd),
                )?;
            }
            Block::Linear(l) => {
                let ssm = cfg.ssm.as_ref().ok_or_else(|| CoreError::Other(
                    "a linear-attention layer was loaded but the model declares no ssm.* config"
                        .into(),
                ))?;
                check(
                    format!("blk.{i}.attn_qkv.weight"),
                    (l.qkv.k, l.qkv.rows),
                    (cfg.n_embd, ssm.qkv_dim()),
                )?;
                check(
                    format!("blk.{i}.attn_gate.weight"),
                    (l.gate.k, l.gate.rows),
                    (cfg.n_embd, ssm.inner_size),
                )?;
                check(
                    format!("blk.{i}.ssm_out.weight"),
                    (l.out.k, l.out.rows),
                    (ssm.inner_size, cfg.n_embd),
                )?;
                check(
                    format!("blk.{i}.ssm_alpha.weight"),
                    (l.alpha.k, l.alpha.rows),
                    (cfg.n_embd, ssm.n_v_heads),
                )?;
                check(
                    format!("blk.{i}.ssm_beta.weight"),
                    (l.beta.k, l.beta.rows),
                    (cfg.n_embd, ssm.n_v_heads),
                )?;
                if l.conv1d.len() != ssm.qkv_dim() * ssm.conv_kernel {
                    return Err(CoreError::ShapeMismatch {
                        name: format!("blk.{i}.ssm_conv1d.weight"),
                        found: vec![l.conv1d.len() as u64],
                        expected: vec![(ssm.qkv_dim() * ssm.conv_kernel) as u64],
                    });
                }
            }
        }

        match &layer.ffn {
            FfnWeights::Dense { gate, up, down } => {
                check(format!("blk.{i}.ffn_gate.weight"), (gate.k, gate.rows), (cfg.n_embd, cfg.ffn_len))?;
                check(format!("blk.{i}.ffn_up.weight"), (up.k, up.rows), (cfg.n_embd, cfg.ffn_len))?;
                check(format!("blk.{i}.ffn_down.weight"), (down.k, down.rows), (cfg.ffn_len, cfg.n_embd))?;
            }
            FfnWeights::Moe {
                router,
                gate_exps,
                up_exps,
                down_exps,
                ..
            } => {
                let moe = cfg.moe.as_ref().expect("MoE weights imply a MoE config");
                check(
                    format!("blk.{i}.ffn_gate_inp.weight"),
                    (router.k, router.rows),
                    (cfg.n_embd, moe.n_experts),
                )?;
                for (n, t) in [("gate", gate_exps), ("up", up_exps)] {
                    check(
                        format!("blk.{i}.ffn_{n}_exps.weight"),
                        (t.k, t.rows),
                        (cfg.n_embd, moe.expert_ff),
                    )?;
                    if t.n_experts != moe.n_experts {
                        return Err(CoreError::ShapeMismatch {
                            name: format!("blk.{i}.ffn_{n}_exps.weight"),
                            found: vec![t.n_experts as u64],
                            expected: vec![moe.n_experts as u64],
                        });
                    }
                }
                check(
                    format!("blk.{i}.ffn_down_exps.weight"),
                    (down_exps.k, down_exps.rows),
                    (moe.expert_ff, cfg.n_embd),
                )?;
            }
        }
    }
    Ok(())
}
