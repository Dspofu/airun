//! The forward pass and the generation loop.
//!
//! Handles both plain dense decoders and Qwen3-Next style hybrids, where most
//! layers are gated delta-net linear attention and only every Nth is softmax
//! attention. The two layer kinds differ in what they carry between tokens: a
//! KV cache that grows with context, or a recurrent state that does not.

use airun_cuda::{DeviceBuffer, Stream, device, hybrid, ops};
use half::f16;

use crate::config::{Config, LayerKind};
use crate::error::{CoreError, Result};
use crate::model::{Block, FfnWeights, GpuTensor, Model};
use crate::plan::Placement;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;

/// What a layer carries from one token to the next.
enum LayerState {
    /// Softmax attention: keys and values for every position so far.
    Kv {
        k: DeviceBuffer<f16>,
        v: DeviceBuffer<f16>,
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

/// Scratch buffers only mixture-of-experts models need.
struct MoeScratch {
    logits: DeviceBuffer<f32>,
    ids: DeviceBuffer<i32>,
    weights: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    shared_logit: DeviceBuffer<f32>,
}

/// The activation-quantization scratch and its freshness flag.
///
/// `valid` means: `buf` holds the quantization of the input the current group
/// of matvec calls shares. [`matvec`] clears and lazily sets it; interior
/// mutability because the runtime hands out `&DeviceState` during a layer.
struct QAct {
    buf: DeviceBuffer<u8>,
    valid: std::cell::Cell<bool>,
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

    /// One entry per layer on this device, in the same order as `layers`.
    states: Vec<LayerState>,
    layers: Vec<usize>,
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

        let mut states = Vec::new();
        for shard in &placement.shards {
            let dev = shard.device;
            device::set_device(dev)?;
            let stream = Stream::new(dev)?;

            let mut layer_states = Vec::with_capacity(shard.layers.len());
            for &l in &shard.layers {
                layer_states.push(match cfg.layer_kind(l) {
                    LayerKind::Attention => LayerState::Kv {
                        k: DeviceBuffer::zeroed(dev, cfg.n_head_kv * ctx * cfg.head_dim, &stream)?,
                        v: DeviceBuffer::zeroed(dev, cfg.n_head_kv * ctx * cfg.head_dim, &stream)?,
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
                }),
                None => None,
            };

            states.push(DeviceState {
                device: dev,
                stream,
                x: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                xb: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                xb2: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                q: DeviceBuffer::new(dev, b * q_width)?,
                k: DeviceBuffer::new(dev, b * k_width)?,
                v: DeviceBuffer::new(dev, b * v_width)?,
                attn: DeviceBuffer::new(dev, b * attn_width)?,
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
                        b * cfg.n_embd.max(ffn_width).max(attn_width) / 32 * 36,
                    )?,
                    valid: std::cell::Cell::new(false),
                },
                hybrid: hybrid_scratch,
                moe: moe_scratch,
                states: layer_states,
                layers: shard.layers.clone(),
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
        })
    }

    pub fn ctx(&self) -> usize {
        self.ctx
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
            let _ = st.stream.synchronize();
        }
    }

    /// Runs `tokens` through the model and returns the logits for the last one.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
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

        // Decode goes through a captured CUDA graph where possible: a step is
        // hundreds of tiny launches, and issuing them one by one measured
        // ~40% of a decode step as GPU idle time on a 1.7B.
        if tokens.len() == 1 && self.states.len() == 1 && !graphs_disabled() {
            return self.forward_decode_graphed(tokens[0]);
        }

        let n = tokens.len();
        let positions: Vec<i32> = (0..n).map(|i| (self.n_past + i) as i32).collect();
        let token_ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.queue_forward(&token_ids, &positions)?;

        let st = &self.states[self.states.len() - 1];
        device::set_device(st.device)?;
        let logits = st
            .logits
            .as_ref()
            .ok_or_else(|| CoreError::Other("output shard has no logits buffer".into()))?;
        let out = logits.to_vec(&st.stream)?;
        self.n_past += n;
        self.history.extend_from_slice(tokens);
        Ok(out)
    }

    /// One decode step through a captured graph.
    ///
    /// The first step records the whole forward pass -- kernel launches and
    /// the two host-to-device copies -- into a graph; later steps overwrite
    /// the pinned staging values and replay it. The capture stores pointers,
    /// which is why the token and position live in pinned buffers owned by
    /// the runtime: their addresses must outlive every replay.
    fn forward_decode_graphed(&mut self, token: u32) -> Result<Vec<f32>> {
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

        let st = &self.states[0];
        let logits = st
            .logits
            .as_ref()
            .ok_or_else(|| CoreError::Other("output shard has no logits buffer".into()))?;
        let out = logits.to_vec(&st.stream)?;
        self.decode_graph = Some(dg);
        self.n_past += 1;
        self.history.push(token);
        Ok(out)
    }

    /// Queues the whole forward pass for `token_ids` at `positions` on the
    /// device streams, through the logits projection. Nothing synchronizes;
    /// the caller reads the logits back. Factored out of [`Self::forward`] so
    /// the decode path can record exactly this work into a CUDA graph.
    fn queue_forward(&mut self, token_ids: &[i32], positions: &[i32]) -> Result<()> {
        let cfg = self.model.config.clone();
        let n = token_ids.len();

        // --- embedding, on the shard that holds the table ---
        {
            let st = &mut self.states[0];
            device::set_device(st.device)?;
            st.tokens.copy_from_host(&token_ids, &st.stream)?;
            st.positions.copy_from_host(&positions, &st.stream)?;

            let embd = &self.model.token_embd;
            ops::embed(
                embd.kernel_type(),
                st.x.as_mut_ptr(),
                embd.data.as_ptr(),
                st.tokens.as_ptr(),
                n,
                cfg.n_embd,
                embd.row_bytes,
                embd.rows,
                &st.stream,
            )?;
        }

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

        ops::rmsnorm(
            st.xb.as_mut_ptr(),
            st.x.as_ptr(),
            self.model.output_norm.as_ptr(),
            cfg.n_embd,
            n,
            cfg.rms_eps,
            &st.stream,
        )?;

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
        let w = &self.model.layers[layer];
        let st = &self.states[si];
        let stream = &st.stream;

        // --- token mixing ---
        ops::rmsnorm(
            st.xb.as_mut_ptr(),
            st.x.as_ptr(),
            w.attn_norm.as_ptr(),
            cfg.n_embd,
            n,
            cfg.rms_eps,
            stream,
        )?;

        let kind = match &w.block {
            Block::Attention(_) => "attn",
            Block::Linear(_) => "ssm",
        };
        if ablated(kind) {
            st.xb2.fill_zero(stream)?;
        } else {
            match &w.block {
                Block::Attention(a) => self.run_attention(si, local, a, n, cfg)?,
                Block::Linear(l) => self.run_linear_attention(si, local, layer, l, n, cfg)?,
            }
        }
        trace_block(st, layer, kind, n, cfg)?;
        ops::add(st.x.as_mut_ptr(), st.xb2.as_ptr(), n * cfg.n_embd, stream)?;

        // --- feed-forward ---
        ops::rmsnorm(
            st.xb.as_mut_ptr(),
            st.x.as_ptr(),
            w.ffn_norm.as_ptr(),
            cfg.n_embd,
            n,
            cfg.rms_eps,
            stream,
        )?;

        if ablated("ffn") {
            st.xb2.fill_zero(stream)?;
        } else {
            match &w.ffn {
                FfnWeights::Dense { gate, up, down } => {
                    matvec(gate, &st.xb, &st.gate, n, &st.q8act, stream)?;
                    matvec_reuse(up, &st.xb, &st.up, n, &st.q8act, stream)?;
                    ops::swiglu(st.gate.as_mut_ptr(), st.up.as_ptr(), n * cfg.ffn_len, stream)?;
                    matvec(down, &st.gate, &st.xb2, n, &st.q8act, stream)?;
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
        ops::add(st.x.as_mut_ptr(), st.xb2.as_ptr(), n * cfg.n_embd, stream)?;

        Ok(())
    }

    /// Softmax attention, optionally with a per-head sigmoid output gate.
    fn run_attention(
        &self,
        si: usize,
        local: usize,
        a: &crate::model::AttentionWeights,
        n: usize,
        cfg: &Config,
    ) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;

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
        } else {
            matvec(&a.q, &st.xb, &st.q, n, &st.q8act, stream)?;
        }
        matvec_reuse(&a.k, &st.xb, &st.k, n, &st.q8act, stream)?;
        matvec_reuse(&a.v, &st.xb, &st.v, n, &st.q8act, stream)?;

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

        if !rope_off {
            ops::rope(
                st.q.as_mut_ptr(),
                st.positions.as_ptr(),
                n,
                cfg.n_head,
                cfg.head_dim,
                rope_dim,
                cfg.rope_base,
                cfg.rope_scale,
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
                cfg.rope_base,
                cfg.rope_scale,
                rope_interleaved,
                stream,
            )?;
        }

        let LayerState::Kv { k: kc, v: vc } = &st.states[local] else {
            return Err(CoreError::Other(
                "an attention layer was given a recurrent state".into(),
            ));
        };

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
            stream,
        )?;

        ops::attention(
            st.attn.as_mut_ptr(),
            st.q.as_ptr(),
            kc.as_ptr(),
            vc.as_ptr(),
            st.positions.as_ptr(),
            n,
            cfg.n_head,
            cfg.n_head_kv,
            cfg.head_dim,
            self.ctx,
            cfg.attn_scale(),
            // AIRUN_ATTN_WINDOW=<n> forces a sliding window. Setting it to 1
            // leaves each query able to see only its own position, so the cost
            // of doing that measures how much the softmax layers draw from
            // history at all.
            match std::env::var("AIRUN_ATTN_WINDOW") {
                Ok(v) => v.parse().unwrap_or(cfg.sliding_window),
                Err(_) => cfg.sliding_window,
            },
            stream,
        )?;

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
        local: usize,
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

        let LayerState::Delta { state, conv } = &st.states[local] else {
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
        hybrid::causal_conv1d(
            h.qkv.as_mut_ptr(),
            l.conv1d.as_ptr(),
            conv.as_mut_ptr(),
            n,
            ssm.qkv_dim(),
            ssm.conv_kernel,
            std::env::var("AIRUN_SSM_NO_CONV_SILU").is_err(),
            std::env::var("AIRUN_SSM_CONV_REVERSE").is_ok(),
            stream,
        )?;

        if trace_ssm(layer) {
            stat(
                "qkv (post-conv, pre-split)",
                &h.qkv,
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
            hybrid::slice_columns(h.qkv.as_ptr(), st.q.as_mut_ptr(), n, ssm.qkv_dim(), q_at, k_dim, stream)?;
            hybrid::slice_columns(h.qkv.as_ptr(), st.k.as_mut_ptr(), n, ssm.qkv_dim(), k_at, k_dim, stream)?;
            hybrid::slice_columns(
                h.qkv.as_ptr(),
                h.core.as_mut_ptr(),
                n,
                ssm.qkv_dim(),
                v_at,
                ssm.v_dim(),
                stream,
            )?;
        } else {
            hybrid::split_qkv_grouped(
                h.qkv.as_ptr(),
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

        // The recurrence writes over the value slice, which is no longer needed.
        hybrid::delta_rule(
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
            stream,
        )?;

        // Gate per value head, then normalize, then scale by the learned weight.
        //
        // Mamba-2's fused RMSNormGated -- which Qwen3-Next's is derived from --
        // gates before it measures the row, so the normalization divides the
        // gate's own scale back out instead of leaving it in the output. Worth
        // knowing that the short smoke test prefers the other order: it
        // continues "...the lazy" with " dog" when the gate comes last and with
        // nonsense when it comes first. Scored over a passage the ranking
        // reverses and stays reversed on a second passage (4.40 against 4.89
        // nats, then 5.08 against 5.75), which is the ordering kept here.
        //
        // AIRUN_SSM_GATE_AFTER_NORM=1 restores the other one.
        hybrid::rmsnorm_gated(
            st.attn.as_mut_ptr(),
            l.norm.as_ptr(),
            h.z.as_ptr(),
            ssm.head_v_dim(),
            n * ssm.n_v_heads,
            cfg.rms_eps,
            std::env::var("AIRUN_SSM_GATE_AFTER_NORM").is_err(),
            std::env::var("AIRUN_SSM_GATE_SIGMOID").is_ok(),
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
        let start = std::time::Instant::now();

        // Prefill, skipping whatever the caches already hold. In a chat each
        // request repeats the whole transcript and appends one turn, so this is
        // the difference between replaying thousands of tokens per message and
        // replaying the last few.
        let cached = self.reusable_prefix(prompt);
        self.rewind_to(cached);

        let mut logits = Vec::new();
        let mut prompt_done = cached;
        while prompt_done < prompt.len() {
            let take = (prompt.len() - prompt_done).min(self.max_batch);
            logits = self.forward(&prompt[prompt_done..prompt_done + take])?;
            prompt_done += take;
        }
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = std::time::Instant::now();
        let mut generated = 0usize;
        let mut history: Vec<u32> = prompt.to_vec();
        // Streaming has to decode across tokens, not per token: a character can
        // straddle two of them, and decoding each alone would destroy it.
        let mut decoder = self.tokenizer.stream_decoder();
        let mut last_token = 0u32;

        let mut stop = StopReason::MaxTokens;
        for _ in 0..max_tokens {
            let token = sampler.sample(&mut logits, &history);
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
            logits = self.forward(&[token])?;
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
    Ok(())
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
    //   2..63      dp4a on quantized activations, when the type has a kernel.
    //   64+        handled inside ops::matvec, which sends large batches to
    //              the fp16 tensor-core kernel for every weight type.
    if n > 1 && n < 64 && !matvec_f32_forced() && ops::matvec_supports_q8(w.kernel_type()) {
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
