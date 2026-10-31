//! The forward pass and the generation loop.
//!
//! Handles both plain dense decoders and Qwen3-Next style hybrids, where most
//! layers are gated delta-net linear attention and only every Nth is softmax
//! attention. The two layer kinds differ in what they carry between tokens: a
//! KV cache that grows with context, or a recurrent state that does not.

use airun_cuda::{DeviceBuffer, Stream, device, hc, hybrid, ops};

use crate::config::{Config, LayerKind};
use crate::error::{CoreError, Result};
use crate::model::{Block, FfnWeights, GpuTensor, Model};
use crate::plan::Placement;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;

#[path = "spec.rs"]
mod spec;

/// What a layer carries from one token to the next.
enum LayerState {
    /// Softmax attention: keys and values for every position so far.
    Kv {
        // Raw rows in cfg.kv_type's format.
        k: DeviceBuffer<u8>,
        v: DeviceBuffer<u8>,
    },
    /// Linear attention: a fixed-size recurrent matrix plus the convolution's
    /// short history. Neither grows with sequence length.
    Delta {
        state: DeviceBuffer<f32>,
        conv: DeviceBuffer<f32>,
    },
}

/// Scratch buffers only hybrid models need.
struct HybridScratch {
    qkv: DeviceBuffer<f32>,
    /// The convolved qkv of a batch (the batch convolution is out of place).
    conv_out: DeviceBuffer<f32>,
    z: DeviceBuffer<f32>,
    core: DeviceBuffer<f32>,
    a: DeviceBuffer<f32>,
    b: DeviceBuffer<f32>,
    decay: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    /// The wide q projection before the output gate is split off.
    q_wide: DeviceBuffer<f32>,
    attn_gate: DeviceBuffer<f32>,
}

/// Scratch for hyper-connection models (see config::HcConfig).
struct HcScratch {
    /// The streams normalized by the current mix, [b][hc][d].
    xn: DeviceBuffer<f32>,
    /// The mix's gate, [b][hc][d]; the PLE reuses it for its normalized value.
    gate: DeviceBuffer<f32>,
    /// The low-rank bottleneck, [b][low_rank].
    lo: DeviceBuffer<f32>,
    /// Stream weights for the combine after the block, [b][hc].
    inject: DeviceBuffer<f32>,
    /// On the device that runs the PLE layer.
    ple: Option<PleScratch>,
}

struct PleScratch {
    /// The batch's n-gram embeddings, gathered on the host.
    emb: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    /// The convolution's history: the last (kernel-1)*ngram normalized rows.
    hist: DeviceBuffer<f32>,
    hist_scratch: DeviceBuffer<f32>,
}

/// Scratch buffers only mixture-of-experts models need.
struct MoeScratch {
    logits: DeviceBuffer<f32>,
    ids: DeviceBuffer<i32>,
    weights: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    shared_logit: DeviceBuffer<f32>,
    upload: Option<std::cell::RefCell<airun_cuda::PinnedBuffer<u8>>>,
    cached_ids: DeviceBuffer<i32>,
    cached_host_ids: std::cell::RefCell<airun_cuda::PinnedBuffer<i32>>,
    cache_group: usize,
    /// Present only when experts are streamed.
    reader: Option<ExpertReader>,
    /// Milliseconds spent waiting for routing results and reading misses.
    timing: std::cell::Cell<(f64, f64)>,
}

/// What a delta-net layer saw during a speculative verify, so the accepted
/// prefix can be pushed into its state afterwards: the pre-convolution rows,
/// and the recurrence's inputs.
struct DeltaStash {
    raw: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    decay: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
}

/// The activation-quantization scratch and its freshness flag.
///
/// `valid` means: `buf` holds the quantization of the input the current group
/// of matvec calls shares. [`matvec`] clears and lazily sets it; interior
/// mutability because the runtime hands out `&DeviceState` during a layer.
struct QAct {
    buf: DeviceBuffer<u8>,
    valid: std::cell::Cell<bool>,
    /// The same input as NVFP4 blocks for the FP4 tensor cores, where the
    /// device and the model have them; same freshness protocol.
    fp4: Option<DeviceBuffer<u8>>,
    fp4_valid: std::cell::Cell<bool>,
}

/// Everything that lives on one GPU.
struct DeviceState {
    device: i32,
    stream: Stream,

    x: DeviceBuffer<f32>,
    xb: DeviceBuffer<f32>,
    xb2: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    attn_partial: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    positions: DeviceBuffer<i32>,
    tokens: DeviceBuffer<i32>,
    logits: Option<DeviceBuffer<f32>>,

    /// Activation quantized to 8-bit groups, plus whether its contents match
    /// the input of the current matvec group. Sized for the widest matvec
    /// input. See [`matvec`] for the freshness protocol.
    q8act: QAct,

    hybrid: Option<HybridScratch>,
    moe: Option<MoeScratch>,
    hc: Option<HcScratch>,

    /// One entry per layer on this device, in the same order as `layers`.
    states: Vec<LayerState>,
    layers: Vec<usize>,
    /// Per layer, parallel to `states`; allocated for delta-net layers once
    /// speculative decoding starts.
    stash: Vec<Option<DeltaStash>>,
}

/// The captured decode step and the pinned staging its replays read.
struct DecodeGraph {
    graph: Option<airun_cuda::Graph>,
    token: airun_cuda::PinnedBuffer<i32>,
    position: airun_cuda::PinnedBuffer<i32>,
}

/// Whether AIRUN_NO_GRAPH=1 turned off decode graph capture.
fn graphs_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_GRAPH").is_ok_and(|v| v.trim() != "0"))
}

/// A loaded model ready to generate.
pub struct Runtime {
    pub model: Model,
    pub tokenizer: Tokenizer,
    states: Vec<DeviceState>,
    ctx: usize,
    max_batch: usize,
    /// Tokens currently in the caches.
    n_past: usize,
    /// Which tokens those are, so a follow-up prompt that starts the same way
    /// can skip replaying the part already processed.
    history: Vec<u32>,
    /// Captured decode step; built on the first single-token forward.
    decode_graph: Option<DecodeGraph>,
    /// Greedy decoding's pick on the device, its pinned landing slot and the
    /// argmax scratch.
    pick: Option<(DeviceBuffer<i32>, airun_cuda::PinnedBuffer<i32>, DeviceBuffer<i32>)>,
    /// The pipelined greedy decode step; built on first use.
    chain: Option<ChainGraph>,
    /// Pinned landing buffer for the logits a sampler reads, reused per token.
    logits_host: Option<airun_cuda::PinnedBuffer<f32>>,
    /// Set during a speculative verify: delta-net layers stash their inputs
    /// instead of advancing their state.
    defer: bool,
    /// Set when the caller projects the logits itself (a verify needs every
    /// row, not just the last).
    skip_logits: bool,
    /// The MTP draft head's state, once speculative decoding has run.
    spec: Option<spec::Spec>,
    /// Drafts per speculative step (see `set_draft`); 0 decodes plainly.
    draft: usize,
}

/// Greedy decode as a chain of graph launches (see [`Runtime::generate`]): the
/// step reads its token and position from the device, and its argmax writes
/// the next token there and advances the position, so step i+1 can be queued
/// before the host has read step i's pick. Between steps the GPU used to idle
/// 170-340 us while the host copied the pick back, synchronized and relaunched
/// -- 1-2% of a Qwen3-4B decode step.
struct ChainGraph {
    graph: Option<airun_cuda::Graph>,
    scratch: DeviceBuffer<i32>,
    /// Each step's pick lands here, alternating between the two slots: the
    /// host reads one while the step after it fills the other.
    slots: airun_cuda::PinnedBuffer<i32>,
    ready: [airun_cuda::Event; 2],
    /// Token and position a chain starts from.
    seed: airun_cuda::PinnedBuffer<i32>,
    launches: usize,
}

/// AIRUN_NO_DECODE_PIPELINE=1 waits for every greedy pick before queuing the
/// next step.
fn decode_pipeline_disabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("AIRUN_NO_DECODE_PIPELINE").is_ok_and(|v| v.trim() != "0"))
}

impl Runtime {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        _streams: Vec<(i32, Stream)>,
        max_batch: usize,
    ) -> Result<Self> {
        let cfg = &model.config;
        let placement: &Placement = &model.placement;
        let ctx = placement.ctx;
        let b = max_batch;

        // Scratch buffers are shared between the attention and linear-attention
        // paths, so each must be sized for whichever path needs it wider. These
        // do not track each other: a hybrid's delta-net key width comes from
        // `group_count * state_size` and can easily exceed the attention
        // `n_head_kv * head_dim`, which is exactly the case on Qwen3.5-9B
        // (2048 against 1024).
        let ffn_width = cfg
            .ffn_len
            .max(cfg.moe.as_ref().map_or(0, |m| m.shared_expert_ff));

        let (ssm_k, ssm_v) = match &cfg.ssm {
            Some(s) => (s.k_dim(), s.v_dim()),
            None => (0, 0),
        };
        let q_width = cfg.q_dim().max(ssm_k);
        let k_width = cfg.kv_dim().max(ssm_k);
        let v_width = cfg.kv_dim().max(ssm_v);
        let attn_width = cfg.q_dim().max(ssm_v);

        // Hyper-connections widen the residual to `count` streams.
        let streams = cfg.hc.map_or(1, |h| h.count);
        let hc_width = streams * cfg.n_embd;

        let model_has_nvfp4 = model.layers.iter().any(|l| match &l.ffn {
            FfnWeights::Dense { gate, .. } => gate.kernel_type().0 == 40,
            FfnWeights::Moe { gate_exps, .. } => gate_exps.kernel_type().0 == 40,
        });
        let mut states = Vec::new();
        for shard in &placement.shards {
            let dev = shard.device;
            device::set_device(dev)?;
            let stream = Stream::new(dev)?;

            let mut layer_states = Vec::with_capacity(shard.layers.len());
            for &l in &shard.layers {
                layer_states.push(match cfg.layer_kind(l) {
                    LayerKind::Attention => LayerState::Kv {
                        k: DeviceBuffer::zeroed(dev, cfg.n_head_kv * ctx * cfg.kv_type.row_bytes(cfg.head_dim), &stream)?,
                        v: DeviceBuffer::zeroed(dev, cfg.n_head_kv * ctx * cfg.kv_type.row_bytes(cfg.head_dim), &stream)?,
                    },
                    LayerKind::LinearAttention => {
                        let ssm = cfg.ssm.as_ref().ok_or_else(|| {
                            CoreError::Other("linear layer without an ssm config".into())
                        })?;
                        LayerState::Delta {
                            state: DeviceBuffer::zeroed(dev, ssm.state_elements(), &stream)?,
                            conv: DeviceBuffer::zeroed(dev, ssm.conv_state_elements(), &stream)?,
                        }
                    }
                });
            }

            let hybrid_scratch = match &cfg.ssm {
                Some(ssm) => Some(HybridScratch {
                    qkv: DeviceBuffer::new(dev, b * ssm.qkv_dim())?,
                    conv_out: DeviceBuffer::new(dev, b * ssm.qkv_dim())?,
                    z: DeviceBuffer::new(dev, b * ssm.inner_size)?,
                    core: DeviceBuffer::new(dev, b * ssm.v_dim())?,
                    a: DeviceBuffer::new(dev, b * ssm.n_v_heads)?,
                    b: DeviceBuffer::new(dev, b * ssm.n_v_heads)?,
                    decay: DeviceBuffer::new(dev, b * ssm.n_v_heads)?,
                    beta: DeviceBuffer::new(dev, b * ssm.n_v_heads)?,
                    q_wide: DeviceBuffer::new(dev, b * cfg.q_dim() * 2)?,
                    attn_gate: DeviceBuffer::new(dev, b * cfg.q_dim())?,
                }),
                None => None,
            };

            let moe_scratch = match &cfg.moe {
                Some(moe) => Some(MoeScratch {
                    logits: DeviceBuffer::new(dev, b * moe.n_experts)?,
                    ids: DeviceBuffer::new(dev, b * moe.n_experts_used)?,
                    weights: DeviceBuffer::new(dev, b * moe.n_experts_used)?,
                    h: DeviceBuffer::new(dev, b * moe.n_experts_used * moe.expert_ff)?,
                    shared_logit: DeviceBuffer::new(dev, b)?,
                    upload: {
                        let staging_bytes = model.layers.iter().filter_map(|l| match &l.ffn {
                            FfnWeights::Moe { gate_exps, up_exps, down_exps, .. }
                                if gate_exps.host_experts.is_some() => Some(
                                    gate_exps.expert_stride.max(up_exps.expert_stride).max(down_exps.expert_stride)
                                    * gate_exps.cache_capacity().min(b * moe.n_experts_used)),
                            _ => None,
                        }).max().unwrap_or(0);
                        if staging_bytes == 0 { None } else {
                            Some(std::cell::RefCell::new(airun_cuda::PinnedBuffer::new(3 * staging_bytes)?))
                        }
                    },
                    cached_ids: DeviceBuffer::new(dev, b * moe.n_experts_used)?,
                    cached_host_ids: std::cell::RefCell::new(airun_cuda::PinnedBuffer::new(b * moe.n_experts_used)?),
                    reader: model.layers.iter().find_map(|l| match &l.ffn {
                        FfnWeights::Moe { gate_exps, .. } => gate_exps.expert_file_paths(),
                        _ => None,
                    }).map(|paths| ExpertReader::new(&paths, EXPERT_READERS))
                    .transpose()
                    .map_err(|e| CoreError::Other(format!(
                        "cannot reopen the model to stream experts ({e}); check the file is still readable")))?,
                    timing: std::cell::Cell::new((0.0, 0.0)),
                    cache_group: model.layers.iter().filter_map(|l| match &l.ffn {
                        FfnWeights::Moe { gate_exps, .. } if gate_exps.host_experts.is_some() =>
                            Some(gate_exps.cache_capacity() / moe.n_experts_used), _ => None,
                    }).min().unwrap_or(1).max(1),
                }),
                None => None,
            };

            let hc_scratch = match &cfg.hc {
                Some(h) => Some(HcScratch {
                    xn: DeviceBuffer::new(dev, b * hc_width)?,
                    gate: DeviceBuffer::new(dev, b * hc_width)?,
                    lo: DeviceBuffer::new(dev, b * h.low_rank)?,
                    inject: DeviceBuffer::new(dev, b * h.count)?,
                    ple: match &cfg.ple {
                        Some(p) if shard.layers.contains(&p.layer) => Some(PleScratch {
                            emb: DeviceBuffer::new(dev, b * p.n_heads() * p.head_dim)?,
                            key: DeviceBuffer::new(dev, b * hc_width)?,
                            value: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                            gated: DeviceBuffer::new(dev, b * hc_width)?,
                            hist: DeviceBuffer::zeroed(dev, p.conv_history() * hc_width, &stream)?,
                            hist_scratch: DeviceBuffer::new(dev, p.conv_history() * hc_width)?,
                        }),
                        _ => None,
                    },
                }),
                None => None,
            };

            states.push(DeviceState {
                device: dev,
                stream,
                x: DeviceBuffer::new(dev, b * hc_width)?,
                xb: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                xb2: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                q: DeviceBuffer::new(dev, b * q_width)?,
                k: DeviceBuffer::new(dev, b * k_width)?,
                v: DeviceBuffer::new(dev, b * v_width)?,
                attn: DeviceBuffer::new(dev, b * attn_width)?,
                attn_partial: DeviceBuffer::new(dev, cfg.n_head * ops::attention_splits(ctx) * (cfg.head_dim + 2))?,
                gate: DeviceBuffer::new(dev, b * ffn_width.max(1))?,
                up: DeviceBuffer::new(dev, b * ffn_width.max(1))?,
                positions: DeviceBuffer::new(dev, b)?,
                tokens: DeviceBuffer::new(dev, b)?,
                logits: if shard.hosts_output {
                    Some(DeviceBuffer::new(dev, cfg.vocab_size)?)
                } else {
                    None
                },
                // 36 bytes per 32 floats; sized for the widest input any
                // matvec on this device sees (xb, gate, or attn).
                q8act: QAct {
                    buf: DeviceBuffer::new(
                        dev,
                        b * cfg.n_embd.max(ffn_width).max(attn_width).max(hc_width) / 32 * 36,
                    )?,
                    valid: std::cell::Cell::new(false),
                    fp4: if model_has_nvfp4 && ops::fp4_supported() {
                        Some(DeviceBuffer::new(dev, ops::nvfp4_act_bytes(b,
                            cfg.n_embd.max(ffn_width).max(attn_width).next_multiple_of(64)))?)
                    } else {
                        None
                    },
                    fp4_valid: std::cell::Cell::new(false),
                },
                hybrid: hybrid_scratch,
                moe: moe_scratch,
                hc: hc_scratch,

                states: layer_states,
                layers: shard.layers.clone(),
                stash: Vec::new(),
            });
        }

        Ok(Self {
            model,
            tokenizer,
            states,
            ctx,
            max_batch,
            n_past: 0,
            history: Vec::new(),
            decode_graph: None,
            pick: None,
            chain: None,
            logits_host: None,
            defer: false,
            skip_logits: false,
            spec: None,
            draft: 0,
        })
    }

    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// Tensor-cache hits/misses, including prefill and warm-up.
    /// Milliseconds spent (waiting on the GPU for routing, reading expert
    /// misses) since load; the first includes all GPU work queued before it.
    pub fn expert_stream_timing(&self) -> (f64, f64) {
        self.states.iter().filter_map(|s| s.moe.as_ref())
            .fold((0.0, 0.0), |(w, r), m| { let (a, b) = m.timing.get(); (w + a, r + b) })
    }

    pub fn expert_cache_stats(&self) -> (u64, u64) {
        self.model.layers.iter().fold((0, 0), |(hits, misses), layer| {
            if let FfnWeights::Moe { gate_exps, up_exps, down_exps, .. } = &layer.ffn {
                [gate_exps, up_exps, down_exps].iter().fold((hits, misses), |(h, m), w| {
                    let (wh, wm) = w.cache_stats(); (h + wh, m + wm)
                })
            } else { (hits, misses) }
        })
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// Starts a fresh sequence.
    ///
    /// A KV cache only needs its position counter reset, because entries are
    /// overwritten by position. A recurrent state does not: it accumulates every
    /// token it has seen, so it has to be zeroed or the next sequence would
    /// continue the previous one.
    pub fn reset(&mut self) {
        self.n_past = 0;
        self.history.clear();
        for st in &self.states {
            if device::set_device(st.device).is_err() {
                continue;
            }
            for state in &st.states {
                if let LayerState::Delta { state, conv } = state {
                    let _ = state.fill_zero(&st.stream);
                    let _ = conv.fill_zero(&st.stream);
                }
            }
            if let Some(p) = st.hc.as_ref().and_then(|h| h.ple.as_ref()) {
                let _ = p.hist.fill_zero(&st.stream);
            }
            let _ = st.stream.synchronize();
        }
    }

    /// Runs `tokens` through the model and returns the logits for the last one.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.step(tokens)?;
        let st = &self.states[self.states.len() - 1];
        device::set_device(st.device)?;
        let logits = st
            .logits
            .as_ref()
            .ok_or_else(|| CoreError::Other("output shard has no logits buffer".into()))?;
        Ok(logits.to_vec(&st.stream)?)
    }

    /// Runs `tokens` through the model in one pass, as a prefill does, and hands
    /// every row's logits to `each(row, logits)` in order -- what a batched
    /// perplexity scores. The head runs eight rows at a time, so the logits of a
    /// whole chunk never sit in VRAM at once.
    pub fn forward_all(&mut self, tokens: &[u32], mut each: impl FnMut(usize, &[f32])) -> Result<()> {
        let n = tokens.len();
        if n > self.max_batch || self.n_past + n > self.ctx {
            return Err(CoreError::Other(format!(
                "{n} tokens at position {} do not fit a batch of {} and a context of {};                  raise --batch-size or --ctx-size", self.n_past, self.max_batch, self.ctx)));
        }
        if let Some(sp) = self.spec.as_mut() {
            sp.synced = false;
        }
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let positions: Vec<i32> = (0..n).map(|i| (self.n_past + i) as i32).collect();
        self.skip_logits = true;
        let queued = self.queue_forward(&ids, &positions);
        self.skip_logits = false;
        queued?;
        self.n_past += n;
        self.history.extend_from_slice(tokens);

        let cfg = &self.model.config;
        let st = self.states.last().expect("a runtime has at least one device");
        device::set_device(st.device)?;
        let out_w = self.model.output_tensor();
        const ROWS: usize = 8;
        let buf = DeviceBuffer::<f32>::new(st.device, ROWS * cfg.vocab_size)?;
        let mut host = vec![0f32; ROWS * cfg.vocab_size];
        for r0 in (0..n).step_by(ROWS) {
            let m = ROWS.min(n - r0);
            ops::matvec(out_w.kernel_type(), out_w.data.as_ptr(), st.xb.offset_ptr(r0 * cfg.n_embd),
                buf.as_mut_ptr(), cfg.n_embd, cfg.vocab_size, m, out_w.row_bytes, &st.stream)?;
            buf.copy_to_host(&mut host[..m * cfg.vocab_size], &st.stream)?;
            st.stream.synchronize()?;
            for i in 0..m {
                each(r0 + i, &host[i * cfg.vocab_size..(i + 1) * cfg.vocab_size]);
            }
        }
        Ok(())
    }

    /// [`forward`], with the logits landing in a reused pinned buffer rather
    /// than a fresh vector: a pageable copy of a 152k vocabulary goes through a
    /// driver staging buffer every token.
    fn forward_pinned(&mut self, tokens: &[u32]) -> Result<&mut [f32]> {
        self.step(tokens)?;
        let st = &self.states[self.states.len() - 1];
        device::set_device(st.device)?;
        let logits = st
            .logits
            .as_ref()
            .ok_or_else(|| CoreError::Other("output shard has no logits buffer".into()))?;
        if self.logits_host.as_ref().is_none_or(|h| h.len() != logits.len()) {
            self.logits_host = Some(airun_cuda::PinnedBuffer::new(logits.len())?);
        }
        let host = self.logits_host.as_mut().expect("allocated above");
        logits.copy_to_host(host.as_mut_slice(), &st.stream)?;
        st.stream.synchronize()?;
        Ok(host.as_mut_slice())
    }

    /// [`forward`] for greedy decoding: the argmax of the logits, computed on
    /// the GPU, so four bytes cross the bus instead of the whole vocabulary.
    /// Picks the same token as [`crate::Sampler`] with temperature 0.
    fn forward_argmax(&mut self, tokens: &[u32]) -> Result<u32> {
        self.step(tokens)?;
        let last = self.states.len() - 1;
        let st = &self.states[last];
        device::set_device(st.device)?;
        let logits = st
            .logits
            .as_ref()
            .ok_or_else(|| CoreError::Other("output shard has no logits buffer".into()))?;
        if self.pick.is_none() {
            self.pick = Some((
                DeviceBuffer::new(st.device, 1)?,
                airun_cuda::PinnedBuffer::new(1)?,
                DeviceBuffer::new(st.device, ops::ARGMAX_SCRATCH)?,
            ));
        }
        let (dev, host, scratch) = self.pick.as_mut().expect("allocated above");
        ops::argmax(logits.as_ptr(), logits.len(), dev.as_mut_ptr(), scratch.as_mut_ptr(),
                    std::ptr::null_mut(), &st.stream)?;
        dev.copy_to_host(host.as_mut_slice(), &st.stream)?;
        st.stream.synchronize()?;
        Ok(host.as_slice()[0] as u32)
    }

    /// Whether greedy decode can run as a [`ChainGraph`]: where the graphed
    /// step runs, and not on a hybrid, whose recurrent state would absorb a
    /// speculative step that the end of a generation then throws away.
    fn chain_eligible(&self) -> bool {
        self.states.len() == 1
            && !graphs_disabled()
            && !decode_pipeline_disabled()
            && !self.model.config.is_hybrid()
            && !self.states[0].moe.as_ref().is_some_and(|m| m.upload.is_some())
    }

    /// Puts `token` and the current position where the chained step reads them.
    fn chain_seed(&mut self, token: u32) -> Result<()> {
        let mut ch = match self.chain.take() {
            Some(ch) => ch,
            None => {
                let dev = self.states[0].device;
                ChainGraph {
                    graph: None,
                    scratch: DeviceBuffer::new(dev, ops::ARGMAX_SCRATCH)?,
                    slots: airun_cuda::PinnedBuffer::new(2)?,
                    ready: [airun_cuda::Event::new()?, airun_cuda::Event::new()?],
                    seed: airun_cuda::PinnedBuffer::new(2)?,
                    launches: 0,
                }
            }
        };
        // The previous chain's seed copies finished before its first pick was
        // read, so the staging is free to overwrite.
        ch.seed.as_mut_slice()[0] = token as i32;
        ch.seed.as_mut_slice()[1] = self.n_past as i32;
        let st = &mut self.states[0];
        device::set_device(st.device)?;
        let queued = st.tokens.copy_from_host(&ch.seed.as_slice()[..1], &st.stream)
            .and_then(|_| st.positions.copy_from_host(&ch.seed.as_slice()[1..], &st.stream));
        self.chain = Some(ch);
        Ok(queued?)
    }

    /// Queues one chained step; returns the slot its pick lands in.
    fn chain_launch(&mut self) -> Result<usize> {
        let mut ch = self.chain.take().ok_or_else(|| {
            CoreError::Other("chained decode launched before it was seeded; this is a bug".into())
        })?;
        device::set_device(self.states[0].device)?;
        if ch.graph.is_none() {
            // SAFETY of the capture: everything queued lives in `self` or in
            // `ch`, both of which outlive the graph.
            airun_cuda::Graph::begin_capture(&self.states[0].stream)?;
            let queued = self.queue_forward_with(&[0], &[0], false).and_then(|_| {
                let st = &self.states[0];
                let logits = st.logits.as_ref().ok_or_else(|| {
                    CoreError::Other("output shard has no logits buffer".into())
                })?;
                ops::argmax(logits.as_ptr(), logits.len(), st.tokens.as_mut_ptr(),
                            ch.scratch.as_mut_ptr(), st.positions.as_mut_ptr(), &st.stream)?;
                Ok(())
            });
            let ended = airun_cuda::Graph::end_capture(&self.states[0].stream);
            let built = queued.and_then(|_| Ok(ended?));
            match built {
                Ok(graph) => ch.graph = Some(graph),
                Err(e) => {
                    self.chain = Some(ch);
                    return Err(e);
                }
            }
        }
        let st = &self.states[0];
        let slot = ch.launches % 2;
        let queued = ch.graph.as_ref().expect("graph was just built").launch(&st.stream)
            .and_then(|_| st.tokens.copy_to_host(&mut ch.slots.as_mut_slice()[slot..slot + 1], &st.stream))
            .and_then(|_| ch.ready[slot].record(&st.stream));
        ch.launches += 1;
        self.chain = Some(ch);
        queued?;
        Ok(slot)
    }

    /// Waits for the chained step that fills `slot` and returns its pick.
    fn chain_read(&self, slot: usize) -> Result<u32> {
        let ch = self.chain.as_ref().ok_or_else(|| {
            CoreError::Other("chained decode read before it was seeded; this is a bug".into())
        })?;
        ch.ready[slot].synchronize()?;
        Ok(ch.slots.as_slice()[slot] as u32)
    }

    /// Runs `tokens` through the model, leaving the last one's logits on the
    /// device.
    fn step(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Err(CoreError::Other("forward called with no tokens".into()));
        }
        if tokens.len() > self.max_batch {
            return Err(CoreError::Other(format!(
                "batch of {} exceeds the {} this runtime was built for; \
                 raise --batch-size or feed the prompt in chunks",
                tokens.len(),
                self.max_batch
            )));
        }
        if self.n_past + tokens.len() > self.ctx {
            return Err(CoreError::Other(format!(
                "sequence would reach {} tokens, past the {} context this model was \
                 loaded with; raise --ctx-size or start a new conversation",
                self.n_past + tokens.len(),
                self.ctx
            )));
        }

        // The MTP head's cache does not follow a plain pass.
        if let Some(sp) = self.spec.as_mut() {
            sp.synced = false;
        }

        // Decode goes through a captured CUDA graph where possible: a step is
        // hundreds of tiny launches, and issuing them one by one measured
        // ~40% of a decode step as GPU idle time on a 1.7B.
        if tokens.len() == 1 && self.states.len() == 1 && !graphs_disabled()
            && self.model.config.ple.is_none()
            && !self.states[0].moe.as_ref().is_some_and(|m| m.upload.is_some()) {
            return self.forward_decode_graphed(tokens[0]);
        }

        let n = tokens.len();
        let positions: Vec<i32> = (0..n).map(|i| (self.n_past + i) as i32).collect();
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.queue_forward(&token_ids, &positions)?;
        self.n_past += n;
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    /// One decode step through a captured graph.
    ///
    /// The first step records the whole forward pass -- kernel launches and
    /// the two host-to-device copies -- into a graph; later steps overwrite
    /// the pinned staging values and replay it. The capture stores pointers,
    /// which is why the token and position live in pinned buffers owned by
    /// the runtime: their addresses must outlive every replay.
    fn forward_decode_graphed(&mut self, token: u32) -> Result<()> {
        let mut dg = match self.decode_graph.take() {
            Some(dg) => dg,
            None => DecodeGraph {
                graph: None,
                token: airun_cuda::PinnedBuffer::new(1)?,
                position: airun_cuda::PinnedBuffer::new(1)?,
            },
        };
        dg.token.as_mut_slice()[0] = token as i32;
        dg.position.as_mut_slice()[0] = self.n_past as i32;

        device::set_device(self.states[0].device)?;
        if dg.graph.is_none() {
            // SAFETY of the capture: everything queue_forward touches lives in
            // `self` or in `dg`, both of which outlive the graph.
            airun_cuda::Graph::begin_capture(&self.states[0].stream)?;
            let queued = self.queue_forward(dg.token.as_slice(), dg.position.as_slice());
            let ended = airun_cuda::Graph::end_capture(&self.states[0].stream);
            queued?;
            dg.graph = Some(ended?);
        }
        dg.graph
            .as_ref()
            .expect("graph was just built")
            .launch(&self.states[0].stream)?;

        self.decode_graph = Some(dg);
        self.n_past += 1;
        self.history.push(token);
        Ok(())
    }

    /// Queues the whole forward pass for `token_ids` at `positions` on the
    /// device streams, through the logits projection. Nothing synchronizes;
    /// the caller reads the logits back. Factored out of [`Self::forward`] so
    /// the decode path can record exactly this work into a CUDA graph.
    fn queue_forward(&mut self, token_ids: &[i32], positions: &[i32]) -> Result<()> {
        self.queue_forward_with(token_ids, positions, true)
    }

    /// [`Self::queue_forward`]; with `upload` false the tokens and positions are
    /// already in the device buffers, and the slices only give their count.
    fn queue_forward_with(&mut self, token_ids: &[i32], positions: &[i32], upload: bool) -> Result<()> {
        let cfg = self.model.config.clone();
        let n = token_ids.len();

        // --- embedding, on the shard that holds the table ---
        {
            let st = &mut self.states[0];
            device::set_device(st.device)?;
            if upload {
                st.tokens.copy_from_host(&token_ids, &st.stream)?;
                st.positions.copy_from_host(&positions, &st.stream)?;
            }

            let embd = &self.model.token_embd;
            // With hyper-connections the embedding lands in xb and is copied
            // into every stream.
            let dst = if cfg.hc.is_some() { st.xb.as_mut_ptr() } else { st.x.as_mut_ptr() };
            ops::embed(
                embd.kernel_type(),
                dst,
                self.model.embedding_ptr(st.device)?,
                st.tokens.as_ptr(),
                n,
                cfg.n_embd,
                embd.row_bytes,
                embd.rows,
                &st.stream,
            )?;
            if let Some(h) = &cfg.hc {
                hc::hc_expand(st.x.as_mut_ptr(), st.xb.as_ptr(), n, h.count, cfg.n_embd, &st.stream)?;
            }
        }
        self.ple_stage(token_ids)?;

        // --- layers, shard by shard ---
        for si in 0..self.states.len() {
            if si > 0 {
                // Hand the activations to the next GPU. Both sides synchronize
                // here; without peer access the copy stages through host memory
                // anyway, so overlapping would buy little.
                let (left, right) = self.states.split_at_mut(si);
                let src = &left[si - 1];
                let dst = &mut right[0];
                src.stream.synchronize()?;
                device::set_device(dst.device)?;
                dst.x.copy_from_device(&src.x, &dst.stream)?;
                dst.positions.copy_from_host(&positions, &dst.stream)?;
                dst.stream.synchronize()?;
            }

            device::set_device(self.states[si].device)?;
            for local in 0..self.states[si].layers.len() {
                let layer = self.states[si].layers[local];
                self.run_layer(si, local, layer, n, &cfg)?;
            }

        }

        // --- final norm and projection, last token only ---
        let last = self.states.len() - 1;
        let st = &self.states[last];
        device::set_device(st.device)?;

        // The last layer's residual add already left this norm in xb; on a
        // hyper-connection model the final mix is the norm.
        if let Some(head) = &self.model.hc_head {
            self.hc_mix(last, head, n, &cfg)?;
        } else if fused_norms_disabled() || st.layers.is_empty() {
            ops::rmsnorm(
                st.xb.as_mut_ptr(),
                st.x.as_ptr(),
                self.model.output_norm.as_ptr(),
                cfg.n_embd,
                n,
                cfg.rms_eps,
                &st.stream,
            )?;
        }

        if self.skip_logits {
            return Ok(());
        }
        let out_w = self.model.output_tensor();
        let logits = st
            .logits
            .as_ref()
            .ok_or_else(|| CoreError::Other("output shard has no logits buffer".into()))?;

        // Only the final position matters for sampling, so project one row.
        // A single row is decode, and decode stays on the exact f32 kernels --
        // see matvec_reuse for the measured reasoning.
        let last_row = st.xb.offset_ptr((n - 1) * cfg.n_embd);
        ops::matvec(
            out_w.kernel_type(),
            out_w.data.as_ptr(),
            last_row,
            logits.as_mut_ptr(),
            cfg.n_embd,
            cfg.vocab_size,
            1,
            out_w.row_bytes,
            &st.stream,
        )?;

        if std::env::var("AIRUN_TRACE_LAYERS").is_ok_and(|v| v.trim() == "sum") {
            let norm = st.xb.to_vec(&st.stream)?;
            let norm = &norm[(n - 1) * cfg.n_embd..n * cfg.n_embd];
            let l = logits.to_vec(&st.stream)?;
            let l = &l[..cfg.vocab_size];
            let best = (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap_or(0);
            eprintln!("  result_norm sum {:>12.6}  result_output sum {:>14.4}  [{:.4}, {:.4}, {:.4}]  argmax {best} ({:.4})",
                norm.iter().map(|&v| v as f64).sum::<f64>(), l.iter().map(|&v| v as f64).sum::<f64>(),
                l[0], l[1], l[2], l[best]);
        }

        Ok(())
    }

    /// How much of `tokens` the caches already hold and need not be replayed.
    ///
    /// A KV cache is addressed by position, so a dense model can drop back to
    /// any earlier point and let the next tokens overwrite from there. A
    /// delta-net layer cannot: its state is a running summary of everything it
    /// has seen, and there is no way to subtract a token back out of it. On a
    /// hybrid the caches are therefore reusable only when the new sequence
    /// extends the old one exactly -- anything else has to start over.
    ///
    /// One token is always left to replay, because the logits come out of a
    /// forward pass and a prompt entirely in cache would produce none.
    pub fn reusable_prefix(&self, tokens: &[u32]) -> usize {
        let limit = tokens.len().saturating_sub(1);
        let common = self
            .history
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(limit);

        if self.model.config.is_hybrid() && common < self.history.len() {
            return 0;
        }
        common
    }

    /// Drops the cache cursor back to `n`, discarding what came after.
    ///
    /// Only ever called with a value [`reusable_prefix`] returned, which is what
    /// makes it safe on hybrids: there it is either the full history or zero.
    fn rewind_to(&mut self, n: usize) {
        if n == 0 {
            self.reset();
            return;
        }
        debug_assert!(
            !self.model.config.is_hybrid() || n == self.history.len(),
            "a hybrid's recurrent state cannot be rewound to an arbitrary point"
        );
        self.n_past = n;
        self.history.truncate(n);
    }

    fn run_layer(
        &mut self,
        si: usize,
        local: usize,
        layer: usize,
        n: usize,
        cfg: &Config,
    ) -> Result<()> {
        // The previous layer on this device already left this norm in xb (see
        // the end of run_block).
        let norm_first = !(local > 0 && !fused_norms_disabled());
        let next = self.next_norm(si, local);
        let stash = self.states[si].stash.get(local).and_then(|s| s.as_ref()).filter(|_| self.defer);
        if cfg.hc.is_some() {
            return self.run_block_hc(si, &self.model.layers[layer], &self.states[si].states[local], stash,
                layer, n, cfg);
        }
        self.run_block(si, &self.model.layers[layer], &self.states[si].states[local], stash, layer,
            n, cfg, norm_first, next)
    }

    /// One block: token mixing then feed-forward, each with its residual add.
    /// `next` is the norm whose output the final residual add leaves in xb
    /// (the following block's, or the output norm); `None` leaves only x.
    #[allow(clippy::too_many_arguments)]
    fn run_block(
        &self,
        si: usize,
        w: &crate::model::LayerWeights,
        state: &LayerState,
        stash: Option<&DeltaStash>,
        layer: usize,
        n: usize,
        cfg: &Config,
        norm_first: bool,
        next: Option<*const std::ffi::c_void>,
    ) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;

        // --- token mixing ---
        if norm_first {
            ops::rmsnorm(
                st.xb.as_mut_ptr(),
                st.x.as_ptr(),
                w.attn_norm.as_ptr(),
                cfg.n_embd,
                n,
                cfg.rms_eps,
                stream,
            )?;
        }

        let kind = match &w.block {
            Block::Attention(_) => "attn",
            Block::Linear(_) => "ssm",
        };
        if ablated(kind) {
            st.xb2.fill_zero(stream)?;
        } else {
            match &w.block {
                Block::Attention(a) => self.run_attention(si, state, a, n, cfg)?,
                Block::Linear(l) => self.run_linear_attention(si, state, stash, layer, l, n, cfg)?,
            }
        }
        trace_block(st, layer, kind, n, cfg)?;

        // --- feed-forward ---

        if fused_norms_disabled() {
            ops::add(st.x.as_mut_ptr(), st.xb2.as_ptr(), n * cfg.n_embd, stream)?;
            ops::rmsnorm(
                st.xb.as_mut_ptr(),
                st.x.as_ptr(),
                w.ffn_norm.as_ptr(),
                cfg.n_embd,
                n,
                cfg.rms_eps,
                stream,
            )?;
        } else {
            ops::add_rmsnorm(st.xb.as_mut_ptr(), st.x.as_mut_ptr(), st.xb2.as_ptr(),
                w.ffn_norm.as_ptr(), cfg.n_embd, n, cfg.rms_eps, stream)?;
        }

        if ablated("ffn") {
            st.xb2.fill_zero(stream)?;
        } else {
            match &w.ffn {
                FfnWeights::Dense { gate, up, down } => {
                    let mut down_done = false;
                    // Prefill takes gate, up and SwiGLU as one integer tensor-core
                    // launch where it can (see matvec_mmq_async_kernel's GLU).
                    let glu = n > 1 && !matvec_f32_forced() && same_shape(&[gate, up])
                        && !fp4_prefill(gate, n, &st.q8act)
                        && ops::matvec_supports_q8(gate.kernel_type())
                        && ops::mmq_glu_supported(gate.kernel_type(), gate.data.as_ptr(),
                            up.data.as_ptr(), st.q8act.buf.as_ptr(), gate.k, gate.row_bytes, n);
                    if n == 1 && same_shape(&[gate, up]) && !q8_decode(gate) {
                        // Decode: gate, up and SwiGLU in one launch, bit-identical.
                        ops::matvec_glu(gate.kernel_type(), gate.data.as_ptr(), up.data.as_ptr(),
                            st.xb.as_ptr(), st.gate.as_mut_ptr(), gate.k, gate.rows, gate.row_bytes,
                            stream)?;
                    } else if glu {
                        ops::quantize_act(st.xb.as_ptr(), st.q8act.buf.as_mut_ptr(), n * gate.k, stream)?;
                        ops::matvec_q8_glu(gate.kernel_type(), gate.data.as_ptr(), up.data.as_ptr(),
                            st.q8act.buf.as_ptr(), st.gate.as_mut_ptr(), gate.k, gate.rows, n,
                            gate.row_bytes, stream)?;
                    } else if fp4_prefill(gate, n, &st.q8act) && fp4_prefill(down, n, &st.q8act) {
                        // The SwiGLU goes into the down projection's quantization.
                        matvec(gate, &st.xb, &st.gate, n, &st.q8act, stream)?;
                        matvec_reuse(up, &st.xb, &st.up, n, &st.q8act, stream)?;
                        let buf = st.q8act.fp4.as_ref().expect("checked by fp4_prefill");
                        ops::quantize_glu_nvfp4(st.gate.as_ptr(), st.up.as_ptr(), buf.as_mut_ptr(), n, down.k,
                            stream)?;
                        st.q8act.valid.set(false);
                        st.q8act.fp4_valid.set(true);
                        matvec_reuse(down, &st.gate, &st.xb2, n, &st.q8act, stream)?;
                        down_done = true;
                    } else {
                        matvec(gate, &st.xb, &st.gate, n, &st.q8act, stream)?;
                        matvec_reuse(up, &st.xb, &st.up, n, &st.q8act, stream)?;
                        ops::swiglu(st.gate.as_mut_ptr(), st.up.as_ptr(), n * cfg.ffn_len, stream)?;
                    }
                    if !down_done {
                        matvec(down, &st.gate, &st.xb2, n, &st.q8act, stream)?;
                    }
                }
                FfnWeights::Moe {
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                    shared,
                } => self
                    .run_moe(si, router, gate_exps, up_exps, down_exps, shared.as_ref(), n, cfg)?,
            }
        }
        trace_block(st, layer, "ffn", n, cfg)?;
        // The residual add, fused with whichever norm reads x next: the next
        // layer's on this device, or the output norm after the last layer.
        // Across devices x moves first, so the add stays plain.
        match next {
            Some(norm) if !fused_norms_disabled() => ops::add_rmsnorm(st.xb.as_mut_ptr(),
                st.x.as_mut_ptr(), st.xb2.as_ptr(), norm, cfg.n_embd, n, cfg.rms_eps, stream)?,
            _ => ops::add(st.x.as_mut_ptr(), st.xb2.as_ptr(), n * cfg.n_embd, stream)?,
        }

        Ok(())
    }

    /// One block of a hyper-connection model: PLE where the layer has it, then
    /// mix -> token mixer -> combine, mix -> feed-forward -> combine. The
    /// mixers read xb and write xb2 as in [`Self::run_block`].
    #[allow(clippy::too_many_arguments)]
    fn run_block_hc(
        &self,
        si: usize,
        w: &crate::model::LayerWeights,
        state: &LayerState,
        stash: Option<&DeltaStash>,
        layer: usize,
        n: usize,
        cfg: &Config,
    ) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;
        let streams = cfg.hc.as_ref().map_or(1, |h| h.count);
        let h = st.hc.as_ref().ok_or_else(|| CoreError::Other("hyper-connection model without its scratch".into()))?;
        let (mix_attn, mix_ffn) = w.hc.as_ref().ok_or_else(|| CoreError::Other(format!(
            "layer {layer} has no hyper-connection weights; the file is incomplete")))?;

        if let Some(p) = &w.ple {
            self.run_ple(si, p, n, cfg)?;
        }

        self.hc_mix(si, mix_attn, n, cfg)?;
        match &w.block {
            Block::Attention(a) => self.run_attention(si, state, a, n, cfg)?,
            Block::Linear(l) => self.run_linear_attention(si, state, stash, layer, l, n, cfg)?,
        }
        trace_block(st, layer, "mixer", n, cfg)?;
        hc::hc_post(st.x.as_mut_ptr(), st.xb2.as_ptr(), h.inject.as_ptr(), n, streams, cfg.n_embd, stream)?;

        self.hc_mix(si, mix_ffn, n, cfg)?;
        match &w.ffn {
            FfnWeights::Moe { router, gate_exps, up_exps, down_exps, shared } =>
                self.run_moe(si, router, gate_exps, up_exps, down_exps, shared.as_ref(), n, cfg)?,
            FfnWeights::Dense { .. } => return Err(CoreError::Other(
                "a hyper-connection model with a dense feed-forward is not supported yet".into())),
        }
        trace_block(st, layer, "ffn", n, cfg)?;
        hc::hc_post(st.x.as_mut_ptr(), st.xb2.as_ptr(), h.inject.as_ptr(), n, streams, cfg.n_embd, stream)?;
        Ok(())
    }

    /// A hyper-connection mix of the streams in x into xb (and, when the mix
    /// has one, the stream weights for the next combine into the inject
    /// scratch).
    fn hc_mix(&self, si: usize, mix: &crate::model::HcMix, n: usize, cfg: &Config) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;
        let hcfg = cfg.hc.ok_or_else(|| CoreError::Other("hyper-connection mix without a config".into()))?;
        let h = st.hc.as_ref().ok_or_else(|| CoreError::Other("hyper-connection model without its scratch".into()))?;
        let d = cfg.n_embd;
        hc::rmsnorm_grouped(h.xn.as_mut_ptr(), st.x.as_ptr(), mix.norm.as_ptr(), d, n * hcfg.count, hcfg.count,
            cfg.rms_eps, stream)?;
        matvec(&mix.down, &h.xn, &h.lo, n, &st.q8act, stream)?;
        hc::silu_scale(h.lo.as_mut_ptr(), n * hcfg.low_rank, 1.0 / hcfg.count as f32, stream)?;
        matvec(&mix.up, &h.lo, &h.gate, n, &st.q8act, stream)?;
        hc::hc_pre(st.xb.as_mut_ptr(), h.xn.as_ptr(), h.gate.as_ptr(), n, hcfg.count, d, stream)?;
        if let Some(inject) = &mix.inject {
            matvec(inject, &h.xn, &h.inject, n, &st.q8act, stream)?;
        }
        Ok(())
    }

    /// The per-layer n-gram embedding block, in place on x. Its inputs were
    /// staged by [`Self::ple_stage`].
    fn run_ple(&self, si: usize, p: &crate::model::PleWeights, n: usize, cfg: &Config) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;
        let (hcfg, pc) = match (&cfg.hc, &cfg.ple) {
            (Some(h), Some(p)) => (h, p),
            _ => return Err(CoreError::Other("PLE weights without the hc and ple configs".into())),
        };
        let h = st.hc.as_ref().ok_or_else(|| CoreError::Other("hyper-connection model without its scratch".into()))?;
        let ps = h.ple.as_ref().ok_or_else(|| CoreError::Other("the PLE layer's GPU has no PLE scratch".into()))?;
        let (d, nh) = (cfg.n_embd, hcfg.count);
        matvec(&p.key, &ps.emb, &ps.key, n, &st.q8act, stream)?;
        matvec_reuse(&p.value, &ps.emb, &ps.value, n, &st.q8act, stream)?;
        hc::rmsnorm_grouped(ps.key.as_mut_ptr(), ps.key.as_ptr(), p.norm_key.as_ptr(), d, n * nh, nh, cfg.rms_eps, stream)?;
        hc::rmsnorm_grouped(h.xn.as_mut_ptr(), st.x.as_ptr(), p.norm_query.as_ptr(), d, n * nh, nh, cfg.rms_eps, stream)?;
        hc::ple_gate(ps.gated.as_mut_ptr(), ps.key.as_ptr(), h.xn.as_ptr(), ps.value.as_ptr(), n, nh, d, stream)?;
        hc::rmsnorm_grouped(h.gate.as_mut_ptr(), ps.gated.as_ptr(), p.norm_conv.as_ptr(), d, n * nh, nh, cfg.rms_eps, stream)?;
        hc::ple_conv(st.x.as_mut_ptr(), ps.gated.as_ptr(), h.gate.as_ptr(), ps.hist.as_mut_ptr(),
            ps.hist_scratch.as_mut_ptr(), p.conv.as_ptr(), n, nh * d, pc.conv_kernel, pc.ngram, stream)?;
        Ok(())
    }

    /// Hashes each token of the batch with its predecessors (from the batch
    /// and the history) into PLE table rows, decodes them on the host and
    /// uploads them to the PLE layer's GPU.
    fn ple_stage(&self, token_ids: &[i32]) -> Result<()> {
        let (Some(pc), Some(table)) = (&self.model.config.ple, &self.model.ple_table) else { return Ok(()) };
        let Some(st) = self.states.iter().find(|s| s.layers.contains(&pc.layer)) else { return Ok(()) };
        let ps = st.hc.as_ref().and_then(|h| h.ple.as_ref())
            .ok_or_else(|| CoreError::Other("the PLE layer's GPU has no PLE scratch".into()))?;
        let keep = pc.ngram - 1;
        let mut seq: Vec<u32> = self.history[self.history.len().saturating_sub(keep)..].to_vec();
        let prefix = seq.len();
        seq.extend(token_ids.iter().map(|&t| t as u32));
        let mut rows = Vec::with_capacity(token_ids.len() * pc.n_heads());
        for i in 0..token_ids.len() {
            pc.rows(&seq[..prefix + i + 1], &mut rows);
        }
        let mut emb = vec![0f32; rows.len() * pc.head_dim];
        table.gather(&rows, &mut emb)?;
        device::set_device(st.device)?;
        ps.emb.copy_from_host_at(0, &emb, &st.stream)?;
        Ok(())
    }

    /// The norm applied to x right after layer `local` of shard `si`, when it
    /// runs on the same device: the next layer's attention norm, or the output
    /// norm after the model's last layer.
    fn next_norm(&self, si: usize, local: usize) -> Option<*const std::ffi::c_void> {
        let layers = &self.states[si].layers;
        if local + 1 < layers.len() {
            Some(self.model.layers[layers[local + 1]].attn_norm.as_ptr())
        } else if si + 1 == self.states.len() {
            Some(self.model.output_norm.as_ptr())
        } else {
            None
        }
    }

    /// Softmax attention, optionally with a per-head sigmoid output gate.
    fn run_attention(
        &self,
        si: usize,
        kv_state: &LayerState,
        a: &crate::model::AttentionWeights,
        n: usize,
        cfg: &Config,
    ) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;

        // Decode runs q, k and v as one launch (same f32 kernel, same result).
        let fused_qkv = n == 1 && !cfg.attn_output_gate && same_shape(&[&a.q, &a.k, &a.v])
            && !q8_decode(&a.q);
        if cfg.attn_output_gate {
            // The q projection packs a gate next to each head's queries; they
            // interleave per head rather than sitting in two blocks.
            let h = st.hybrid.as_ref().ok_or_else(|| {
                CoreError::Other("gated attention needs the hybrid scratch buffers".into())
            })?;
            matvec(&a.q, &st.xb, &h.q_wide, n, &st.q8act, stream)?;
            // AIRUN_ATTN_GATE_BLOCKED=1 selects the [all q][all gate] layout,
            // for files whose converter reordered the projection.
            let interleaved = std::env::var("AIRUN_ATTN_GATE_BLOCKED").is_err();
            // Which half of each head's slice is the query and which is the
            // gate is pure convention -- both halves are head_dim wide, so a
            // converter that emitted them the other way round looks identical.
            // AIRUN_ATTN_GATE_FIRST=1 reads the gate out of the first half.
            let (q_dst, gate_dst) = if std::env::var("AIRUN_ATTN_GATE_FIRST").is_ok() {
                (h.attn_gate.as_mut_ptr(), st.q.as_mut_ptr())
            } else {
                (st.q.as_mut_ptr(), h.attn_gate.as_mut_ptr())
            };
            hybrid::split_head_gate(
                h.q_wide.as_ptr(),
                q_dst,
                gate_dst,
                n,
                cfg.n_head,
                cfg.head_dim,
                interleaved,
                stream,
            )?;
        } else if fused_qkv {
            // Decode: q, k and v in one launch. k and v alone are grids too
            // small to keep DRAM busy (see matvec3_kernel).
            ops::matvec3(
                a.q.kernel_type(),
                [a.q.data.as_ptr(), a.k.data.as_ptr(), a.v.data.as_ptr()],
                st.xb.as_ptr(),
                [st.q.as_mut_ptr(), st.k.as_mut_ptr(), st.v.as_mut_ptr()],
                a.q.k,
                [a.q.rows, a.k.rows, a.v.rows],
                a.q.row_bytes,
                stream,
            )?;
        } else {
            matvec(&a.q, &st.xb, &st.q, n, &st.q8act, stream)?;
        }
        if !fused_qkv {
            matvec_reuse(&a.k, &st.xb, &st.k, n, &st.q8act, stream)?;
            matvec_reuse(&a.v, &st.xb, &st.v, n, &st.q8act, stream)?;
        }


        // Partial rotary: `rope_dim` may be a fraction of `head_dim`, leaving the
        // tail of each head unrotated. AIRUN_ROPE_FULL=1 rotates the whole head
        // instead, for files whose metadata understates the rotary width.
        let rope_dim = if std::env::var("AIRUN_ROPE_FULL").is_ok() {
            cfg.head_dim
        } else {
            cfg.rope_dim
        };
        // AIRUN_ROPE_INTERLEAVED=1 pairs (2i, 2i+1) instead of (i, i + dim/2).
        let rope_interleaved = std::env::var("AIRUN_ROPE_INTERLEAVED").is_ok();
        // AIRUN_ROPE_OFF=1 removes position entirely. Its cost measures how much
        // the attention layers actually use position: a model that barely
        // notices is not attending the way it was trained to.
        let rope_off = std::env::var("AIRUN_ROPE_OFF").is_ok();

        let LayerState::Kv { k: kc, v: vc } = kv_state else {
            return Err(CoreError::Other(
                "an attention layer was given a recurrent state".into(),
            ));
        };

        if !fused_prologue_disabled() {
            // q/k norms, RoPE and the cache write in one launch (see
            // qk_norm_rope_store_kernel); the same arithmetic as the steps below.
            let null = std::ptr::null::<std::ffi::c_void>();
            ops::qk_norm_rope_store(
                st.q.as_mut_ptr(), st.k.as_ptr(), st.v.as_ptr(),
                a.q_norm.as_ref().map_or(null, |w| w.as_ptr()),
                a.k_norm.as_ref().map_or(null, |w| w.as_ptr()),
                kc.as_mut_ptr(), vc.as_mut_ptr(), st.positions.as_ptr(),
                n, cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.rms_eps,
                (!rope_off).then_some((rope_dim, cfg.rope, rope_interleaved)),
                cfg.kv_type, stream,
            )?;
        } else {
            // Per-head RMSNorm before RoPE. Rows are contiguous head_dim slices, so
            // this is one call over n * heads rows.
            if let Some(qn) = &a.q_norm {
                ops::rmsnorm(
                    st.q.as_mut_ptr(),
                    st.q.as_ptr(),
                    qn.as_ptr(),
                    cfg.head_dim,
                    n * cfg.n_head,
                    cfg.rms_eps,
                    stream,
                )?;
            }
            if let Some(kn) = &a.k_norm {
                ops::rmsnorm(
                    st.k.as_mut_ptr(),
                    st.k.as_ptr(),
                    kn.as_ptr(),
                    cfg.head_dim,
                    n * cfg.n_head_kv,
                    cfg.rms_eps,
                    stream,
                )?;
            }

            if !rope_off {
                ops::rope(
                    st.q.as_mut_ptr(),
                    st.positions.as_ptr(),
                    n,
                    cfg.n_head,
                    cfg.head_dim,
                    rope_dim,
                    cfg.rope,
                    rope_interleaved,
                    stream,
                )?;
                ops::rope(
                    st.k.as_mut_ptr(),
                    st.positions.as_ptr(),
                    n,
                    cfg.n_head_kv,
                    cfg.head_dim,
                    rope_dim,
                    cfg.rope,
                    rope_interleaved,
                    stream,
                )?;
            }

            ops::kv_store(
                kc.as_mut_ptr(),
                vc.as_mut_ptr(),
                st.k.as_ptr(),
                st.v.as_ptr(),
                st.positions.as_ptr(),
                n,
                cfg.n_head_kv,
                cfg.head_dim,
                self.ctx,
                cfg.kv_type,
                stream,
            )?;
        }

        let window = std::env::var("AIRUN_ATTN_WINDOW").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(cfg.sliding_window);
        if n == 1 && std::env::var("AIRUN_NO_SPLIT_ATTN").is_err() {
            ops::attention_split(st.attn.as_mut_ptr(), &st.attn_partial,
                st.q.as_ptr(), kc.as_ptr(), vc.as_ptr(), st.positions.as_ptr(),
                cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.attn_scale(),
                window, ops::attention_splits(self.ctx), cfg.kv_type, stream)?;
        } else {
            ops::attention(st.attn.as_mut_ptr(), st.q.as_ptr(), kc.as_ptr(), vc.as_ptr(),
                st.positions.as_ptr(), n, cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                self.ctx, cfg.attn_scale(), window, cfg.kv_type, stream)?;
        }
        // AIRUN_ATTN_NO_GATE=1 drops the gate entirely, which says whether the
        // gate is the thing going wrong or merely riding along.
        if cfg.attn_output_gate && std::env::var("AIRUN_ATTN_NO_GATE").is_err() {
            let h = st.hybrid.as_ref().unwrap();
            hybrid::sigmoid_gate(
                st.attn.as_mut_ptr(),
                h.attn_gate.as_ptr(),
                n * cfg.q_dim(),
                stream,
            )?;
        }

        matvec(&a.output, &st.attn, &st.xb2, n, &st.q8act, stream)?;
        Ok(())
    }

    /// Gated delta-net linear attention.
    fn run_linear_attention(
        &self,
        si: usize,
        delta_state: &LayerState,
        stash: Option<&DeltaStash>,
        layer: usize,
        l: &crate::model::LinearAttentionWeights,
        n: usize,
        cfg: &Config,
    ) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;
        let ssm = cfg
            .ssm
            .as_ref()
            .ok_or_else(|| CoreError::Other("linear layer without an ssm config".into()))?;
        let h = st
            .hybrid
            .as_ref()
            .ok_or_else(|| CoreError::Other("linear layer without hybrid scratch".into()))?;

        let LayerState::Delta { state, conv } = delta_state else {
            return Err(CoreError::Other(
                "a linear-attention layer was given a KV cache".into(),
            ));
        };

        // Projections.
        matvec(&l.qkv, &st.xb, &h.qkv, n, &st.q8act, stream)?;
        matvec_reuse(&l.gate, &st.xb, &h.z, n, &st.q8act, stream)?;
        // `ssm_alpha` feeds the decay and `ssm_beta` the update strength. The
        // upstream reference splits this projection as (b, a) in that order, so
        // a converter naming the halves positionally could swap them.
        // AIRUN_SSM_SWAP_AB=1 exchanges the two.
        let (a_dst, b_dst) = if std::env::var("AIRUN_SSM_SWAP_AB").is_ok() {
            (&h.b, &h.a)
        } else {
            (&h.a, &h.b)
        };
        matvec_reuse(&l.alpha, &st.xb, a_dst, n, &st.q8act, stream)?;
        matvec_reuse(&l.beta, &st.xb, b_dst, n, &st.q8act, stream)?;

        // decay = exp(-exp(A) * softplus(a + dt_bias)), beta = sigmoid(b).
        hybrid::delta_gates(
            h.a.as_ptr(),
            h.b.as_ptr(),
            l.a_log.as_ptr(),
            l.dt_bias.as_ptr(),
            h.decay.as_mut_ptr(),
            h.beta.as_mut_ptr(),
            n,
            ssm.n_v_heads,
            stream,
        )?;

        // Short causal convolution over the packed q/k/v, with SiLU.
        // AIRUN_SSM_CONV_REVERSE=1 flips the tap order.
        // A speculative verify leaves the recurrent state as it was and keeps
        // what it needs to advance it later by however many tokens are
        // accepted (see spec.rs).
        if let Some(sh) = stash {
            copy_rows(&sh.raw, &h.qkv, n * ssm.qkv_dim(), stream)?;
        }
        // A batch convolves out of place, in parallel over its tokens, into
        // conv_out; one token stays in place.
        let silu = std::env::var("AIRUN_SSM_NO_CONV_SILU").is_err();
        let reverse = std::env::var("AIRUN_SSM_CONV_REVERSE").is_ok();
        let conv_rows = if n > 1 {
            hybrid::causal_conv1d_oop(h.qkv.as_ptr(), h.conv_out.as_mut_ptr(), l.conv1d.as_ptr(),
                conv.as_mut_ptr(), n, ssm.qkv_dim(), ssm.conv_kernel, silu, reverse, stash.is_none(), stream)?;
            &h.conv_out
        } else {
            hybrid::causal_conv1d_ex(h.qkv.as_mut_ptr(), l.conv1d.as_ptr(), conv.as_mut_ptr(), n, ssm.qkv_dim(),
                ssm.conv_kernel, silu, reverse, stash.is_none(), stream)?;
            &h.qkv
        };

        if trace_ssm(layer) {
            stat(
                "qkv (post-conv, pre-split)",
                conv_rows,
                (n - 1) * ssm.qkv_dim(),
                ssm.qkv_dim(),
                stream,
            )?;
        }

        // Split the convolved projection. Reusing q/k/v here is safe: this layer
        // has no softmax attention to clobber.
        //
        // This projection *is* three contiguous blocks, unlike the attention
        // q/gate projection which keeps HuggingFace's per-head interleave. The
        // difference is not arbitrary: extracting the delta-net gate into its
        // own tensor forces the converter to deinterleave, while the attention
        // gate rides along in one tensor and is left alone. Both layouts were
        // confirmed by running the model each way.
        //
        // AIRUN_SSM_QKV_GROUPED=1 selects the per-key-head layout instead.
        let k_dim = ssm.k_dim();
        if std::env::var("AIRUN_SSM_QKV_GROUPED").is_err() {
            let [q_at, k_at, v_at] = qkv_offsets(ssm)?;
            hybrid::slice_columns(conv_rows.as_ptr(), st.q.as_mut_ptr(), n, ssm.qkv_dim(), q_at, k_dim, stream)?;
            hybrid::slice_columns(conv_rows.as_ptr(), st.k.as_mut_ptr(), n, ssm.qkv_dim(), k_at, k_dim, stream)?;
            hybrid::slice_columns(
                conv_rows.as_ptr(),
                h.core.as_mut_ptr(),
                n,
                ssm.qkv_dim(),
                v_at,
                ssm.v_dim(),
                stream,
            )?;
        } else {
            hybrid::split_qkv_grouped(
                conv_rows.as_ptr(),
                st.q.as_mut_ptr(),
                st.k.as_mut_ptr(),
                h.core.as_mut_ptr(),
                n,
                ssm.n_k_heads,
                ssm.head_k_dim(),
                ssm.head_v_dim(),
                ssm.n_v_heads / ssm.n_k_heads,
                stream,
            )?;
        }

        // Queries and keys are unit vectors; the 1/sqrt(d) scale then plays the
        // same role it does in softmax attention.
        // AIRUN_SSM_NO_L2NORM=1 and AIRUN_SSM_NO_QSCALE=1 drop either step.
        if std::env::var("AIRUN_SSM_NO_L2NORM").is_err() {
            hybrid::l2norm(st.q.as_mut_ptr(), n * ssm.n_k_heads, ssm.head_k_dim(), 1e-6, stream)?;
            hybrid::l2norm(st.k.as_mut_ptr(), n * ssm.n_k_heads, ssm.head_k_dim(), 1e-6, stream)?;
        }
        if std::env::var("AIRUN_SSM_NO_QSCALE").is_err() {
            ops::scale(
                st.q.as_mut_ptr(),
                1.0 / (ssm.head_k_dim() as f32).sqrt(),
                n * k_dim,
                stream,
            )?;
        }

        if let Some(sh) = stash {
            copy_rows(&sh.q, &st.q, n * k_dim, stream)?;
            copy_rows(&sh.k, &st.k, n * k_dim, stream)?;
            copy_rows(&sh.v, &h.core, n * ssm.v_dim(), stream)?;
            copy_rows(&sh.decay, &h.decay, n * ssm.n_v_heads, stream)?;
            copy_rows(&sh.beta, &h.beta, n * ssm.n_v_heads, stream)?;
        }

        // The recurrence writes over the value slice, which is no longer needed.
        hybrid::delta_rule_ex(
            state.as_mut_ptr(),
            st.q.as_ptr(),
            st.k.as_ptr(),
            h.core.as_ptr(),
            h.decay.as_ptr(),
            h.beta.as_ptr(),
            st.attn.as_mut_ptr(),
            n,
            ssm.n_k_heads,
            ssm.n_v_heads,
            ssm.head_k_dim(),
            ssm.head_v_dim(),
            stash.is_none(),
            stream,
        )?;

        // Normalize, scale by the learned weight, then gate -- the order of
        // Qwen3-Next's RMSNormGated and of llama.cpp's build_norm_gated.
        //
        // The gate-first order once scored better (4.40 against 4.89 nats), but
        // only while the delta rule paired V heads with the wrong K heads. With
        // the K-head broadcast fixed the 35B scores 1.43 gate-last against 5.12
        // gate-first. Two layout bugs can hide each other: re-measure old A/B
        // decisions after fixing one.
        //
        // AIRUN_SSM_GATE_BEFORE_NORM=1 restores the other order.
        hybrid::rmsnorm_gated(
            st.attn.as_mut_ptr(),
            l.norm.as_ptr(),
            h.z.as_ptr(),
            ssm.head_v_dim(),
            n * ssm.n_v_heads,
            cfg.rms_eps,
            std::env::var("AIRUN_SSM_GATE_BEFORE_NORM").is_ok(),
            cfg.ssm_gate_sigmoid || std::env::var("AIRUN_SSM_GATE_SIGMOID").is_ok(),
            stream,
        )?;

        if trace_ssm(layer) {
            let row = n - 1;
            eprintln!("  -- delta-net layer {layer}, token {row} --");
            stat("q (l2-normed, scaled)", &st.q, row * k_dim, k_dim, stream)?;
            stat("k (l2-normed)", &st.k, row * k_dim, k_dim, stream)?;
            stat("v (post-conv)", &h.core, row * ssm.v_dim(), ssm.v_dim(), stream)?;
            stat("z (gate)", &h.z, row * ssm.inner_size, ssm.inner_size, stream)?;
            stat("decay", &h.decay, row * ssm.n_v_heads, ssm.n_v_heads, stream)?;
            stat("beta", &h.beta, row * ssm.n_v_heads, ssm.n_v_heads, stream)?;
            stat("state", state, 0, ssm.state_elements(), stream)?;
            stat("out (post gated norm)", &st.attn, row * ssm.v_dim(), ssm.v_dim(), stream)?;
        }

        matvec(&l.out, &st.attn, &st.xb2, n, &st.q8act, stream)?;
        Ok(())
    }

    /// Sparse feed-forward: route each token to a few experts, plus the shared
    /// expert every token passes through.
    #[allow(clippy::too_many_arguments)]
    fn run_moe(
        &self,
        si: usize,
        router: &GpuTensor,
        gate_exps: &GpuTensor,
        up_exps: &GpuTensor,
        down_exps: &GpuTensor,
        shared: Option<&crate::model::SharedExpert>,
        n: usize,
        cfg: &Config,
    ) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;
        let moe = cfg
            .moe
            .as_ref()
            .ok_or_else(|| CoreError::Other("MoE layer without a MoE config".into()))?;
        let m = st
            .moe
            .as_ref()
            .ok_or_else(|| CoreError::Other("MoE layer without MoE scratch".into()))?;

        matvec(router, &st.xb, &m.logits, n, &st.q8act, stream)?;
        hybrid::moe_router(
            m.logits.as_ptr(),
            m.ids.as_mut_ptr(),
            m.weights.as_mut_ptr(),
            n,
            moe.n_experts,
            moe.n_experts_used,
            moe.norm_topk,
            stream,
        )?;

        // The shared expert writes the output first; the routed experts then
        // accumulate on top, which avoids a separate buffer and a second add.
        if let Some(s) = shared {
            matvec_reuse(&s.router, &st.xb, &m.shared_logit, n, &st.q8act, stream)?;
            matvec_reuse(&s.gate, &st.xb, &st.gate, n, &st.q8act, stream)?;
            matvec_reuse(&s.up, &st.xb, &st.up, n, &st.q8act, stream)?;
            ops::swiglu(
                st.gate.as_mut_ptr(),
                st.up.as_ptr(),
                n * moe.shared_expert_ff,
                stream,
            )?;
            matvec(&s.down, &st.gate, &st.xb2, n, &st.q8act, stream)?;
            hybrid::sigmoid_scale_rows(
                st.xb2.as_mut_ptr(),
                m.shared_logit.as_ptr(),
                n,
                cfg.n_embd,
                stream,
            )?;
        } else {
            st.xb2.fill_zero(stream)?;
        }

        if let Some(upload) = &m.upload {
            let mut ids = vec![0i32; n * moe.n_experts_used];
            let waited = std::time::Instant::now();
            m.ids.copy_to_host(&mut ids, stream)?;
            stream.synchronize()?;
            let wait_ms = waited.elapsed().as_secs_f64() * 1000.0;
            let mut staging = upload.borrow_mut();
            let mut host_ids = m.cached_host_ids.borrow_mut();
            let tensors = [gate_exps, up_exps, down_exps];
            for t in (0..n).step_by(m.cache_group) {
                // The next token may evict a slot, and pinned staging cannot be
                // rewritten while the DMA engine is still reading it.
                if t > 0 { stream.synchronize()?; }
                let count = m.cache_group.min(n - t);
                let selected = &ids[t * moe.n_experts_used..(t + count) * moe.n_experts_used];
                let mut picks = Vec::with_capacity(3);
                for w in tensors {
                    match w.select_experts(selected) {
                        Ok(p) => picks.push(p),
                        Err(e) => { tensors.iter().for_each(|w| w.invalidate_experts()); return Err(e); }
                    }
                }
                if picks[0].0 != picks[1].0 || picks[0].0 != picks[2].0 {
                    tensors.iter().for_each(|w| w.invalidate_experts());
                    return Err(CoreError::Other("expert caches disagree; reload the model and run airun selftest".into()));
                }
                let reader = m.reader.as_ref().ok_or_else(|| CoreError::Other(
                    "streamed experts without a reader; reload the model".into()))?;
                let read = std::time::Instant::now();
                let uploaded = upload_experts(&tensors, &picks, staging.as_mut_slice(), reader, stream);
                let (w, r) = m.timing.get();
                m.timing.set((w + if t == 0 { wait_ms } else { 0.0 }, r + read.elapsed().as_secs_f64() * 1000.0));
                if let Err(e) = uploaded {
                    tensors.iter().for_each(|w| w.invalidate_experts());
                    return Err(e);
                }
                let gate_ids = &picks[0].0;
                host_ids.as_mut_slice()[..gate_ids.len()].copy_from_slice(gate_ids);
                m.cached_ids.copy_from_host_at(0, &host_ids.as_slice()[..gate_ids.len()], stream)?;
                hybrid::moe_gate_up(
                    gate_exps.kernel_type(), gate_exps.data.as_ptr(), up_exps.data.as_ptr(),
                    st.xb.offset_ptr(t * cfg.n_embd), m.cached_ids.as_ptr(), m.h.as_mut_ptr(),
                    cfg.n_embd, moe.expert_ff, count, moe.n_experts_used,
                    gate_exps.row_bytes, gate_exps.expert_stride, stream,
                )?;
                hybrid::moe_down(
                    down_exps.kernel_type(), down_exps.data.as_ptr(), m.h.as_ptr(),
                    m.cached_ids.as_ptr(), m.weights.offset_ptr(t * moe.n_experts_used),
                    st.xb2.offset_ptr(t * cfg.n_embd), moe.expert_ff, cfg.n_embd,
                    count, moe.n_experts_used, down_exps.row_bytes, down_exps.expert_stride, stream,
                )?;
                // Both staging arrays belong to the runtime. The next routing
                // read synchronizes before either is reused by another layer.
            }
            return Ok(());
        }

        hybrid::moe_gate_up(
            gate_exps.kernel_type(),
            gate_exps.data.as_ptr(),
            up_exps.data.as_ptr(),
            st.xb.as_ptr(),
            m.ids.as_ptr(),
            m.h.as_mut_ptr(),
            cfg.n_embd,
            moe.expert_ff,
            n,
            moe.n_experts_used,
            gate_exps.row_bytes,
            gate_exps.expert_stride,
            stream,
        )?;

        hybrid::moe_down(
            down_exps.kernel_type(),
            down_exps.data.as_ptr(),
            m.h.as_ptr(),
            m.ids.as_ptr(),
            m.weights.as_ptr(),
            st.xb2.as_mut_ptr(),
            moe.expert_ff,
            cfg.n_embd,
            n,
            moe.n_experts_used,
            down_exps.row_bytes,
            down_exps.expert_stride,
            stream,
        )?;

        Ok(())
    }

    /// Generates up to `max_tokens`, calling `on_token` for each one.
    ///
    /// Returning `false` from `on_token` stops generation, which is how the
    /// server implements client disconnects and stop sequences.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        sampler: &mut Sampler,
        mut on_token: impl FnMut(u32, &str) -> bool,
    ) -> Result<GenerationStats> {
        if sampler.is_pure_greedy() && !gpu_argmax_disabled() && self.speculates() {
            return self.generate_spec(prompt, max_tokens, &mut on_token);
        }
        let start = std::time::Instant::now();

        // Prefill, skipping whatever the caches already hold. In a chat each
        // request repeats the whole transcript and appends one turn, so this is
        // the difference between replaying thousands of tokens per message and
        // replaying the last few.
        let cached = self.reusable_prefix(prompt);
        self.rewind_to(cached);

        // Greedy decoding picks each token on the GPU (see forward_argmax);
        // anything else samples from logits copied to pinned host memory.
        let greedy = sampler.is_pure_greedy() && !gpu_argmax_disabled();
        let mut next: Option<u32> = None;
        let mut prompt_done = cached;
        while prompt_done < prompt.len() {
            let take = (prompt.len() - prompt_done).min(self.max_batch);
            let chunk = &prompt[prompt_done..prompt_done + take];
            prompt_done += take;
            if prompt_done < prompt.len() {
                self.step(chunk)?;
            } else if greedy {
                next = Some(self.forward_argmax(chunk)?);
            } else {
                self.forward_pinned(chunk)?;
            }
        }
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = std::time::Instant::now();
        let mut generated = 0usize;
        let mut history: Vec<u32> = prompt.to_vec();
        // Streaming has to decode across tokens, not per token: a character can
        // straddle two of them, and decoding each alone would destroy it.
        let mut decoder = self.tokenizer.stream_decoder();
        let mut last_token = 0u32;

        // Chained greedy steps queued and not yet read, oldest first: the slot
        // each one's pick lands in. At most two -- the step for the token just
        // accepted, and the one after it.
        let chain = greedy && self.chain_eligible();
        let mut inflight: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        let mut stop = StopReason::MaxTokens;
        for _ in 0..max_tokens {
            let token = match next.take() {
                Some(t) => t,
                None => {
                    let logits = self.logits_host.as_mut().ok_or_else(|| CoreError::Other(
                        "generate has no logits to sample; the prompt must not be empty".into()))?;
                    sampler.sample(logits.as_mut_slice(), &history)
                }
            };
            if self.tokenizer.is_eog(token) {
                stop = StopReason::EndOfGeneration;
                break;
            }
            let text = decoder.push(&self.tokenizer, token);
            last_token = token;
            generated += 1;
            history.push(token);
            if !on_token(token, &text) {
                stop = StopReason::Cancelled;
                break;
            }
            if self.n_past >= self.ctx {
                stop = StopReason::ContextFull;
                break;
            }
            if chain {
                if inflight.is_empty() {
                    self.chain_seed(token)?;
                    inflight.push_back(self.chain_launch()?);
                }
                // That step (queued now or one iteration ago) is `token`'s.
                self.n_past += 1;
                self.history.push(token);
                // The next step's input is this step's pick, which the GPU
                // already holds: queue it before reading the pick back, so the
                // GPU is not idle while the host waits, copies and hands the
                // token out. If the pick ends generation, that step is never
                // committed: the position stays put and its cache row is
                // overwritten by whatever comes next.
                if generated < max_tokens && self.n_past < self.ctx {
                    inflight.push_back(self.chain_launch()?);
                }
                let slot = inflight.pop_front().expect("a chained step is in flight");
                next = Some(self.chain_read(slot)?);
            } else if greedy {
                next = Some(self.forward_argmax(&[token])?);
            } else {
                self.forward_pinned(&[token])?;
            }
        }
        if !inflight.is_empty() {
            // A speculative step outlived the loop; let it finish before its
            // pinned staging can be reused.
            let st = &self.states[0];
            device::set_device(st.device)?;
            st.stream.synchronize()?;
        }

        // A generation that stops mid-character leaves bytes behind; emit them
        // rather than dropping text on the floor.
        let tail = decoder.flush();
        if !tail.is_empty() {
            on_token(last_token, &tail);
        }

        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        Ok(GenerationStats {
            prompt_tokens: prompt.len(),
            cached_tokens: cached,
            generated_tokens: generated,
            prefill_ms,
            decode_ms,
            stop,
            drafted: 0,
            accepted: 0,
        })
    }
}

/// Why a generation loop ended.
///
/// The caller cannot infer this from the counts: a run that emits exactly
/// `max_tokens` and then would have stopped anyway is indistinguishable from
/// one that was cut off, and an OpenAI client is told a different
/// `finish_reason` for each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted an end-of-generation token.
    EndOfGeneration,
    /// The token budget ran out.
    MaxTokens,
    /// The context filled up.
    ContextFull,
    /// The callback asked to stop -- a stop sequence, or a client that hung up.
    Cancelled,
}

/// Timings for one generation call.
#[derive(Debug, Clone, Copy)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    /// Of those, how many were already in cache and were not recomputed.
    pub cached_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub stop: StopReason,
    /// Speculative decoding: tokens the MTP head drafted, and how many of them
    /// the main model accepted. Both 0 without speculation.
    pub drafted: usize,
    pub accepted: usize,
}

impl GenerationStats {
    /// Tokens the prefill actually had to process.
    pub fn processed_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.cached_tokens)
    }

    /// Rate over the tokens actually processed -- counting cached ones would
    /// report a throughput the hardware never reached.
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_ms <= 0.0 {
            return 0.0;
        }
        self.processed_tokens() as f64 / (self.prefill_ms / 1000.0)
    }

    pub fn decode_tps(&self) -> f64 {
        if self.decode_ms <= 0.0 {
            return 0.0;
        }
        self.generated_tokens as f64 / (self.decode_ms / 1000.0)
    }
}

/// Where q, k and v start inside the packed delta-net projection.
///
/// The three are contiguous blocks, but nothing in the file records their order,
/// and q and k are the same width -- so a converter that emitted them swapped
/// would pass every shape check. AIRUN_SSM_QKV_ORDER is a permutation of "qkv",
/// defaulting to the order the tensor is named for.
fn qkv_offsets(ssm: &crate::config::SsmConfig) -> Result<[usize; 3]> {
    let order = std::env::var("AIRUN_SSM_QKV_ORDER").unwrap_or_else(|_| "qkv".to_string());
    let mut offsets = [usize::MAX; 3];
    let mut at = 0usize;
    for c in order.chars() {
        let (slot, width) = match c {
            'q' => (0, ssm.k_dim()),
            'k' => (1, ssm.k_dim()),
            'v' => (2, ssm.v_dim()),
            _ => return Err(CoreError::Other(format!(
                "AIRUN_SSM_QKV_ORDER must be a permutation of \"qkv\", found {order:?}"
            ))),
        };
        if offsets[slot] != usize::MAX {
            return Err(CoreError::Other(format!(
                "AIRUN_SSM_QKV_ORDER repeats {c:?}; it must be a permutation of \"qkv\""
            )));
        }
        offsets[slot] = at;
        at += width;
    }
    if offsets.contains(&usize::MAX) {
        return Err(CoreError::Other(format!(
            "AIRUN_SSM_QKV_ORDER must name all of q, k and v, found {order:?}"
        )));
    }
    Ok(offsets)
}

/// Whether AIRUN_ABLATE names this block kind, in which case it contributes
/// nothing to the residual.
///
/// This separates "this block is wrong" from "this block is fine and the fault
/// is elsewhere". A correctly composed block always scores better than no block
/// at all, so if skipping one *improves* the loss, that block is the bug --
/// something a per-kernel test cannot tell you, because every kernel involved
/// can be individually correct while the composition is not.
///
/// Accepts a comma-separated list: AIRUN_ABLATE=ssm,ffn.
fn ablated(kind: &str) -> bool {
    match std::env::var("AIRUN_ABLATE") {
        Ok(v) => v.split(',').any(|part| part.trim() == kind),
        Err(_) => false,
    }
}

/// Whether AIRUN_TRACE_SSM names this layer, e.g. AIRUN_TRACE_SSM=0.
fn trace_ssm(layer: usize) -> bool {
    match std::env::var("AIRUN_TRACE_SSM") {
        Ok(v) => v.trim() == layer.to_string(),
        Err(_) => false,
    }
}

/// Prints the range of one intermediate tensor.
///
/// Every quantity inside a delta-net layer has a range it must be in if it is
/// what it claims to be: a decay is in (0, 1), an l2-normalized head has norm 1,
/// a gate straddles zero. Reading them off a real forward pass is how the
/// already-exponentiated `ssm_a` was caught, and it costs nothing to keep the
/// same window open on the rest.
fn stat(
    label: &str,
    buf: &DeviceBuffer<f32>,
    offset: usize,
    len: usize,
    stream: &Stream,
) -> Result<()> {
    let host = buf.to_vec(stream)?;
    let slice = &host[offset..offset + len];
    let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
    let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = slice.iter().sum::<f32>() / len as f32;
    let rms = (slice.iter().map(|v| v * v).sum::<f32>() / len as f32).sqrt();
    let nan = slice.iter().filter(|v| !v.is_finite()).count();
    eprintln!(
        "     {label:<24} min {min:>11.5}  max {max:>11.5}  mean {mean:>11.5}  rms {rms:>10.5}  \
         nonfinite {nan}"
    );
    Ok(())
}

/// Prints how big each block's contribution is next to the residual it joins.
///
/// A block that is composed wrongly -- a transposed layout, a gate on the wrong
/// side of a norm -- usually still produces plausible-looking text for a few
/// tokens, but its contribution to the residual stream is off by an order of
/// magnitude from the start. That shows up here immediately, and the profile of
/// a model known to be correct is the yardstick to compare against.
///
/// Enabled by AIRUN_TRACE_LAYERS=1. Costs a device synchronization per block,
/// so it is a debugging tool, not something to leave on.
fn trace_block(st: &DeviceState, layer: usize, kind: &str, n: usize, cfg: &Config) -> Result<()> {
    if std::env::var("AIRUN_TRACE_LAYERS").is_err() {
        return Ok(());
    }

    // Only the last token is reported: during prefill it is the one whose state
    // every later token builds on, and printing every row would bury the signal.
    // The mean is reported next to the RMS because a block that adds a constant
    // offset to every channel corrupts the directions the feed-forward layers
    // match against while leaving the magnitudes -- and the grammar -- intact.
    let moments = |buf: &DeviceBuffer<f32>| -> Result<(f32, f32)> {
        let host = buf.to_vec(&st.stream)?;
        let row = &host[(n - 1) * cfg.n_embd..n * cfg.n_embd];
        let mean = row.iter().sum::<f32>() / cfg.n_embd as f32;
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / cfg.n_embd as f32).sqrt();
        Ok((mean, rms))
    };

    let (bm, br) = moments(&st.xb2)?;
    let (xm, xr) = moments(&st.x)?;
    eprintln!(
        "  layer {layer:>3} {kind:<4}  block rms {br:>9.4} mean {bm:>9.4}  \
         residual rms {xr:>9.4} mean {xm:>9.4}"
    );
    // The block output's sum over every token and the last token's first
    // values, in the form llama.cpp's eval-callback prints them (as
    // linear_attn_out-N / attn_output-N / ffn_out-N), so the same prompt
    // compares layer by layer.
    if std::env::var("AIRUN_TRACE_LAYERS").is_ok_and(|v| v.trim() == "sum") {
        let host = st.xb2.to_vec(&st.stream)?;
        let row = &host[(n - 1) * cfg.n_embd..n * cfg.n_embd];
        let sum: f64 = host[..n * cfg.n_embd].iter().map(|&v| v as f64).sum();
        let last: f64 = row.iter().map(|&v| v as f64).sum();
        eprintln!("    sum {sum:>12.6}  last {last:>12.6}  [{:.4}, {:.4}, {:.4}, ..., {:.4}, {:.4}, {:.4}]",
            row[0], row[1], row[2], row[cfg.n_embd - 3], row[cfg.n_embd - 2], row[cfg.n_embd - 1]);
    }
    Ok(())
}

/// Copies the first `len` floats of `src` into `dst`, on the stream.
fn copy_rows(dst: &DeviceBuffer<f32>, src: &DeviceBuffer<f32>, len: usize, stream: &Stream) -> Result<()> {
    let bytes = len * std::mem::size_of::<f32>();
    Ok(ops::copy_2d(dst.as_mut_ptr(), bytes, src.as_ptr(), bytes, bytes, 1, stream)?)
}

/// Whether this matmul takes the FP4 tensor cores: an NVFP4 weight, a batch
/// big enough to fill their 128-token tiles reasonably, and a device and
/// build that have them (see ops::fp4_supported; AIRUN_NO_FP4=1 opts out).
/// The activation is then 4-bit, as in llama.cpp's Blackwell path; decode and
/// small batches keep 8-bit activations.
fn fp4_prefill(w: &GpuTensor, n: usize, q8: &QAct) -> bool {
    n >= 32 && q8.fp4.is_some() && w.kernel_type().0 == 40 && w.k % 256 == 0 && !matvec_f32_forced()
}

/// Decode on the integer kernel for this weight; see
/// [`ops::KernelType::prefers_q8_decode`].
fn q8_decode(w: &GpuTensor) -> bool {
    !matvec_f32_forced() && w.kernel_type().prefers_q8_decode() && ops::matvec_supports_q8(w.kernel_type())
}

/// Whether AIRUN_MATVEC_F32=1 pinned the matvec to the exact f32 path.
///
/// The integer path quantizes the activation to 8 bits, so it is fast but not
/// bit-identical to the CPU reference; this switch is the escape hatch for
/// numerical debugging, where "exactly the reference" matters more than speed.
fn matvec_f32_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| std::env::var("AIRUN_MATVEC_F32").is_ok_and(|v| v.trim() != "0"))
}

/// Concurrent reads when streaming experts. An NVMe drive only reaches its
/// bandwidth with several requests in flight.
const EXPERT_READERS: usize = 8;

use crate::expert_reader::ExpertReader;

/// Reads every expert miss of one MoE step from the GGUF into pinned staging,
/// then queues the uploads.
///
/// The misses are a few dozen ~0.6 MiB pieces scattered over a 20 GB file. Read
/// through the mmap they arrived one page fault at a time, the SSD's queue-depth-1
/// speed of ~220 MB/s, and that was nearly all of decode. Parallel positional
/// reads (see ExpertReader) keep the drive's queue full.
fn upload_experts(
    tensors: &[&GpuTensor; 3],
    picks: &[(Vec<i32>, Vec<(usize, usize)>)],
    staging: &mut [u8],
    reader: &ExpertReader,
    stream: &Stream,
) -> Result<()> {
    // (tensor, slot, (shard, file offset), staging range)
    let mut jobs = Vec::new();
    let mut used = 0usize;
    for (w, (_, uploads)) in tensors.iter().zip(picks) {
        for &(slot, expert) in uploads {
            let offset = w.expert_file_offset(expert).ok_or_else(|| CoreError::Other(
                "an expert miss on a tensor that is not streamed; reload the model".into()))?;
            let end = used + w.expert_stride;
            if end > staging.len() {
                return Err(CoreError::Other(
                    "expert uploads exceed the staging buffer; lower --batch-size".into()));
            }
            jobs.push((*w, slot, offset, used..end));
            used = end;
        }
    }
    if jobs.is_empty() {
        return Ok(());
    }

    let mut rest = &mut staging[..used];
    let mut reads = Vec::with_capacity(jobs.len());
    for (w, _, offset, _) in &jobs {
        let (head, tail) = rest.split_at_mut(w.expert_stride);
        reads.push((offset.0, offset.1, head));
        rest = tail;
    }
    reader.read_all(reads).map_err(|e| CoreError::Other(format!(
        "reading experts from the model file failed ({e}); check the drive and that the file is unchanged")))?;
    for (w, slot, _, range) in jobs {
        w.data.copy_from_host_at(slot * w.expert_stride, &staging[range], stream)?;
    }
    Ok(())
}

/// AIRUN_NO_GPU_ARGMAX=1 copies the logits to the host for greedy decoding too.
fn gpu_argmax_disabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("AIRUN_NO_GPU_ARGMAX").is_ok_and(|v| v.trim() != "0"))
}

/// AIRUN_NO_FUSED_NORMS=1 keeps each residual add and the norm after it as two
/// launches, to compare against add_rmsnorm.
fn fused_norms_disabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("AIRUN_NO_FUSED_NORMS").is_ok_and(|v| v.trim() != "0"))
}

/// AIRUN_NO_FUSED_PROLOGUE=1 runs the q/k norms, RoPE and the cache write as
/// separate launches, to compare against the fused kernel.
fn fused_prologue_disabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("AIRUN_NO_FUSED_PROLOGUE").is_ok_and(|v| v.trim() != "0"))
}

/// Weights that one fused decode launch can serve together: same kernel type,
/// inner size and row size, with row counts the fused kernel can partition.
fn same_shape(ws: &[&GpuTensor]) -> bool {
    let first = ws[0];
    ws.iter().all(|w| {
        w.ty == first.ty && w.k == first.k && w.row_bytes == first.row_bytes
            && w.rows % 4 == 0 && w.n_experts <= 1
    })
}


/// `out = w @ x` for `n` tokens.
///
/// Quantizes `x` into `q8` and takes the integer kernel when the weight type
/// has one, the f32 kernel otherwise. The quantization happens even when this
/// particular weight cannot use it, so a following [`matvec_reuse`] against
/// the same `x` is valid regardless of how the types are mixed.
fn matvec(
    w: &GpuTensor,
    x: &DeviceBuffer<f32>,
    out: &DeviceBuffer<f32>,
    n: usize,
    q8: &QAct,
    stream: &Stream,
) -> Result<()> {
    // A fresh input invalidates whatever quantization is sitting in the
    // scratch; it is rebuilt lazily by the first call that can use it, so a
    // group whose weights are all float types never pays for quantization.
    q8.valid.set(false);
    q8.fp4_valid.set(false);
    matvec_reuse(w, x, out, n, q8, stream)
}

/// [`matvec`] whose `x` is the same buffer and contents as the previous call's
/// -- the point is to not re-quantize when q, k and v all project the same
/// normalized activation.
fn matvec_reuse(
    w: &GpuTensor,
    x: &DeviceBuffer<f32>,
    out: &DeviceBuffer<f32>,
    n: usize,
    q8: &QAct,
    stream: &Stream,
) -> Result<()> {
    // Which kernel serves which batch size, all of it measured:
    //   n == 1     exact f32 -- decode is bandwidth-bound on the weights, the
    //              same bytes move either way, and skipping the quantize
    //              launches measured faster (148 vs 133 tok/s, Q8_0 1.7B).
    //              This also keeps generation bit-identical to the reference.
    //              NVFP4 is the exception (prefers_q8_decode): at 4.5 bits the
    //              f32 vector, not the weights, is what each warp reads most.
    //   2..63      dp4a on quantized activations, when the type has a kernel.
    //   64+        int8 tensor cores (MMQ) on sm_80+ for the same types.
    //              Other types and older GPUs go to ops::matvec's fp16
    //              tensor-core kernel.
    if fp4_prefill(w, n, q8) {
        let buf = q8.fp4.as_ref().expect("checked by fp4_prefill");
        if !q8.fp4_valid.get() {
            ops::quantize_act_nvfp4(x.as_ptr(), buf.as_mut_ptr(), n, w.k, stream)?;
            q8.fp4_valid.set(true);
        }
        ops::matmul_nvfp4(w.data.as_ptr(), w.row_bytes, buf.as_ptr(), out.as_mut_ptr(), w.k, w.rows, n, stream)?;
        return Ok(());
    }
    let batch_q8 = n < 64 || ops::mmq_supported();
    let wants_q8 = if n == 1 { q8_decode(w) } else { batch_q8 };
    if wants_q8 && !matvec_f32_forced() && ops::matvec_supports_q8(w.kernel_type()) {
        if !q8.valid.get() {
            ops::quantize_act(x.as_ptr(), q8.buf.as_mut_ptr(), n * w.k, stream)?;
            q8.valid.set(true);
        }
        ops::matvec_q8(
            w.kernel_type(),
            w.data.as_ptr(),
            q8.buf.as_ptr(),
            out.as_mut_ptr(),
            w.k,
            w.rows,
            n,
            w.row_bytes,
            stream,
        )?;
        return Ok(());
    }
    ops::matvec(
        w.kernel_type(),
        w.data.as_ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        w.k,
        w.rows,
        n,
        w.row_bytes,
        stream,
    )?;
    Ok(())
}
