//! Speculative decoding with the model's own MTP head (Qwen3.5/3.8 `nextn`).
//!
//! Each step drafts `k` tokens with the MTP block, then runs the main model
//! once over the newest token plus the drafts. Greedy decoding keeps the
//! longest prefix of drafts the main model agrees with, plus the main model's
//! own next token, so the output is exactly what plain greedy decoding writes:
//! the drafts only decide how many tokens one pass over the weights yields.
//!
//! Attention layers need no undo: cache rows past the accepted prefix are
//! overwritten before anything attends to them. A delta-net layer's state
//! absorbs every token it sees, so the verify pass runs those layers without
//! writing their state (`Runtime::defer`), stashing their inputs, and
//! [`Runtime::commit`] then pushes only the accepted tokens.
//!
//! The MTP head's KV row at position q holds the pair (h_{q-1}, x_q) -- the
//! main model's normed final hidden at q-1 and the token at q -- which is the
//! convention of llama.cpp's draft-mtp. Rows the drafting writes from the
//! head's own hiddens are rewritten from the verify pass's before anything
//! attends to them.

use super::*;

pub(super) struct Spec {
    /// Drafts per step.
    k: usize,
    /// The MTP head's KV cache, on the output device.
    kv: LayerState,
    /// Hidden rows the next MTP call reads, one per token.
    h: DeviceBuffer<f32>,
    /// Embedding rows, normed in place.
    e: DeviceBuffer<f32>,
    /// `[enorm(e) | hnorm(h)]` rows: eh_proj's input.
    cat: DeviceBuffer<f32>,
    /// The main model's hidden at the last position in the caches: the h of
    /// the first row the next prefill gives the head.
    pending: DeviceBuffer<f32>,
    /// k + 1 rows of logits from a verify.
    logits: DeviceBuffer<f32>,
    picks: DeviceBuffer<i32>,
    picks_host: airun_cuda::PinnedBuffer<i32>,
    scratch: DeviceBuffer<i32>,
    /// The head's cache and `pending` describe exactly the first `n_past`
    /// positions. Anything that advances the caches without the head clears it.
    pub(super) synced: bool,
    /// The verify of k + 1 tokens as a CUDA graph (one device only).
    vgraph: Option<VerifyGraph>,
    /// The one-row head step as a CUDA graph.
    mgraph: Option<VerifyGraph>,
    /// [`Runtime::advance_graphed`]'s graphs, by accepted count - 1.
    agraphs: Vec<Option<AdvanceGraph>>,
}

/// A captured advance (commit, head rows, k drafts) and its pinned inputs.
struct AdvanceGraph {
    graph: Option<airun_cuda::Graph>,
    tokens: airun_cuda::PinnedBuffer<i32>,
    positions: airun_cuda::PinnedBuffer<i32>,
}

/// A captured verify pass and the pinned staging its replays read, as
/// [`DecodeGraph`] for decode: a verify is ~2000 launches, which issued one by
/// one left the GPU idle for a quarter of the pass.
struct VerifyGraph {
    graph: Option<airun_cuda::Graph>,
    tokens: airun_cuda::PinnedBuffer<i32>,
    positions: airun_cuda::PinnedBuffer<i32>,
}

impl Runtime {
    /// Drafts per speculative step; 0 turns speculation off. Has an effect only
    /// when the model was loaded with its MTP head.
    pub fn set_draft(&mut self, k: usize) {
        self.draft = if self.model.mtp.is_some() { k } else { 0 };
    }

    /// Whether greedy generation speculates.
    pub fn speculates(&self) -> bool {
        self.draft > 0 && self.model.mtp.is_some()
    }

    fn spec_init(&mut self) -> Result<()> {
        let k = self.draft;
        if self.spec.as_ref().is_some_and(|s| s.k == k) {
            return Ok(());
        }
        let cfg = self.model.config.clone();
        let ssm = cfg.ssm.as_ref().ok_or_else(|| CoreError::Other(
            "speculative decoding needs the delta-net config of a hybrid model".into()))?;
        let rows = k + 1;
        if rows > self.max_batch {
            return Err(CoreError::Other(format!(
                "--draft {k} verifies {rows} tokens at once; raise --batch-size to at least {rows}")));
        }

        for st in &mut self.states {
            device::set_device(st.device)?;
            let dev = st.device;
            st.stash = st.layers.iter().map(|&l| -> Result<Option<DeltaStash>> {
                Ok(match cfg.layer_kind(l) {
                    LayerKind::LinearAttention => Some(DeltaStash {
                        raw: DeviceBuffer::new(dev, rows * ssm.qkv_dim())?,
                        q: DeviceBuffer::new(dev, rows * ssm.k_dim())?,
                        k: DeviceBuffer::new(dev, rows * ssm.k_dim())?,
                        v: DeviceBuffer::new(dev, rows * ssm.v_dim())?,
                        decay: DeviceBuffer::new(dev, rows * ssm.n_v_heads)?,
                        beta: DeviceBuffer::new(dev, rows * ssm.n_v_heads)?,
                    }),
                    LayerKind::Attention => None,
                })
            }).collect::<Result<Vec<_>>>()?;
        }

        let st = self.states.last().expect("a runtime has at least one device");
        device::set_device(st.device)?;
        let dev = st.device;
        let b = self.max_batch;
        let kv_bytes = cfg.n_head_kv * self.ctx * cfg.kv_type.row_bytes(cfg.head_dim);
        self.spec = Some(Spec {
            k,
            kv: LayerState::Kv {
                k: DeviceBuffer::zeroed(dev, kv_bytes, &st.stream)?,
                v: DeviceBuffer::zeroed(dev, kv_bytes, &st.stream)?,
                qsa: None,
            },
            h: DeviceBuffer::new(dev, b * cfg.n_embd)?,
            e: DeviceBuffer::new(dev, b * cfg.n_embd)?,
            cat: DeviceBuffer::new(dev, b * 2 * cfg.n_embd)?,
            pending: DeviceBuffer::zeroed(dev, cfg.n_embd, &st.stream)?,
            logits: DeviceBuffer::new(dev, rows * cfg.vocab_size)?,
            picks: DeviceBuffer::new(dev, rows)?,
            picks_host: airun_cuda::PinnedBuffer::new(rows)?,
            scratch: DeviceBuffer::new(dev, ops::ARGMAX_SCRATCH)?,
            synced: false,
            vgraph: None,
            mgraph: None,
            agraphs: Vec::new(),
        });
        Ok(())
    }

    fn sp(&self) -> &Spec {
        self.spec.as_ref().expect("speculation was set up by spec_init")
    }

    /// Fills the head's hidden rows: `pending` first when `from_pending`, then
    /// rows `0..count` of xb on the output device (the last pass's hiddens).
    fn stage_h(&self, from_pending: bool, count: usize) -> Result<()> {
        let st = self.states.last().expect("a runtime has at least one device");
        let sp = self.sp();
        let d = self.model.config.n_embd;
        let row = d * std::mem::size_of::<f32>();
        device::set_device(st.device)?;
        let at = from_pending as usize;
        if from_pending {
            ops::copy_2d(sp.h.as_mut_ptr(), row, sp.pending.as_ptr(), row, row, 1, &st.stream)?;
        }
        ops::copy_2d(sp.h.offset_ptr(at * d), row, st.xb.as_ptr(), row, row, count, &st.stream)?;
        Ok(())
    }

    /// `pending` = row `i` of xb on the output device.
    fn save_pending(&self, i: usize) -> Result<()> {
        let st = self.states.last().expect("a runtime has at least one device");
        let d = self.model.config.n_embd;
        let row = d * std::mem::size_of::<f32>();
        device::set_device(st.device)?;
        ops::copy_2d(self.sp().pending.as_mut_ptr(), row, st.xb.offset_ptr(i * d), row, row, 1, &st.stream)?;
        Ok(())
    }

    /// The MTP head over `tokens` at positions from `pos0`, reading the hidden
    /// rows already in `spec.h`. Leaves the head's normed output rows in xb of
    /// the output device and returns the greedy pick after the last row.
    fn mtp_run(&mut self, tokens: &[u32], pos0: usize) -> Result<u32> {
        if tokens.len() == 1 && !graphs_disabled() {
            return self.mtp_run_graphed(tokens[0], pos0);
        }
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let pos: Vec<i32> = (0..tokens.len()).map(|i| (pos0 + i) as i32).collect();
        self.mtp_queue(&ids, &pos)?;
        let st = self.states.last().expect("a runtime has at least one device");
        st.stream.synchronize()?;
        Ok(self.sp().picks_host.as_slice()[0] as u32)
    }

    /// A one-row head step (each draft after the first) as a captured graph:
    /// ~80 small launches and a readback, which issued one by one were a third
    /// of the step. The head lives on the output device, so this is captured
    /// on its stream even when the model is split across GPUs.
    fn mtp_run_graphed(&mut self, token: u32, pos: usize) -> Result<u32> {
        let si = self.states.len() - 1;
        let mut mg = match self.spec.as_mut().expect("set up by spec_init").mgraph.take() {
            Some(mg) => mg,
            None => VerifyGraph {
                graph: None,
                tokens: airun_cuda::PinnedBuffer::new(1)?,
                positions: airun_cuda::PinnedBuffer::new(1)?,
            },
        };
        mg.tokens.as_mut_slice()[0] = token as i32;
        mg.positions.as_mut_slice()[0] = pos as i32;
        device::set_device(self.states[si].device)?;
        let built = if mg.graph.is_none() {
            // SAFETY of the capture: everything queued lives in `self` or `mg`,
            // and `mg` goes back into `self.spec` below.
            airun_cuda::Graph::begin_capture(&self.states[si].stream)?;
            let queued = self.mtp_queue(mg.tokens.as_slice(), mg.positions.as_slice());
            let ended = airun_cuda::Graph::end_capture(&self.states[si].stream);
            queued.and_then(|_| Ok(ended?)).map(|g| mg.graph = Some(g))
        } else {
            Ok(())
        };
        let launched = built.and_then(|_| {
            Ok(mg.graph.as_ref().expect("graph was just built").launch(&self.states[si].stream)?)
        });
        self.spec.as_mut().expect("set up by spec_init").mgraph = Some(mg);
        launched?;
        self.states[si].stream.synchronize()?;
        Ok(self.sp().picks_host.as_slice()[0] as u32)
    }

    /// The MTP head over `ids` at `pos`, reading the hidden rows already in
    /// `spec.h`. Leaves the head's normed output rows in xb of the output device
    /// and queues the last row's greedy pick into `picks_host[0]`.
    fn mtp_queue(&mut self, ids: &[i32], pos: &[i32]) -> Result<()> {
        {
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            st.tokens.copy_from_host_at(0, ids, &st.stream)?;
            st.positions.copy_from_host_at(0, pos, &st.stream)?;
        }
        self.mtp_body(ids.len(), 0)?;
        let st = self.states.last().expect("a runtime has at least one device");
        let sp = self.spec.as_mut().expect("set up by spec_init");
        sp.picks.copy_to_host(&mut sp.picks_host.as_mut_slice()[..1], &st.stream)?;
        Ok(())
    }

    /// The head over the `n` tokens and positions already on the output
    /// device, reading `spec.h`; its normed output rows land in xb and the last
    /// row's greedy pick in `picks[slot]`.
    fn mtp_body(&self, n: usize, slot: usize) -> Result<()> {
        let cfg = &self.model.config;
        let mtp = self.model.mtp.as_ref().ok_or_else(|| CoreError::Other(
            "speculative decoding without an MTP head; this is a bug".into()))?;
        let sp = self.sp();
        let si = self.states.len() - 1;
        let st = &self.states[si];
        let stream = &st.stream;
        let d = cfg.n_embd;
        let embd = &self.model.token_embd;
        ops::embed(embd.kernel_type(), sp.e.as_mut_ptr(), self.model.embedding_ptr(st.device)?,
            st.tokens.as_ptr(), n, d, embd.row_bytes, embd.rows, stream)?;
        ops::rmsnorm(sp.e.as_mut_ptr(), sp.e.as_ptr(), mtp.enorm.as_ptr(), d, n, cfg.rms_eps, stream)?;
        ops::rmsnorm(st.xb2.as_mut_ptr(), sp.h.as_ptr(), mtp.hnorm.as_ptr(), d, n, cfg.rms_eps, stream)?;
        let row = d * std::mem::size_of::<f32>();
        ops::copy_2d(sp.cat.as_mut_ptr(), 2 * row, sp.e.as_ptr(), row, row, n, stream)?;
        ops::copy_2d(sp.cat.offset_ptr(d), 2 * row, st.xb2.as_ptr(), row, row, n, stream)?;
        matvec(&mtp.eh_proj, &sp.cat, &st.x, n, &st.q8act, stream)?;

        self.run_block(si, &mtp.layer, &sp.kv, None, cfg.n_run_layers(), n, cfg, true, None)?;
        ops::rmsnorm(st.xb.as_mut_ptr(), st.x.as_ptr(), mtp.head_norm.as_ptr(), d, n, cfg.rms_eps, stream)?;

        let out_w = self.model.output_tensor();
        ops::matvec(out_w.kernel_type(), out_w.ptr(), st.xb.offset_ptr((n - 1) * d),
            sp.logits.as_mut_ptr(), d, cfg.vocab_size, 1, out_w.row_bytes, stream)?;
        ops::argmax(sp.logits.as_ptr(), cfg.vocab_size, sp.picks.offset_ptr(slot), sp.scratch.as_mut_ptr(),
            std::ptr::null_mut(), stream)?;
        Ok(())
    }

    /// Everything between two verifies as one captured graph per accepted
    /// count `m`: push the m tokens into the delta-net states, feed the head
    /// the verify's rows, and draft k tokens, each step's pick written straight
    /// into the next step's token on the device. One launch and one readback,
    /// where the host had waited on each head step in turn. `toks` are the m
    /// tokens after the verify's first, `pos0` the position of the first.
    fn advance_graphed(&mut self, toks: &[u32], pos0: usize) -> Result<Vec<u32>> {
        let k = self.draft;
        let m = toks.len();
        let taken = {
            let sp = self.spec.as_mut().expect("set up by spec_init");
            if sp.agraphs.len() < m {
                sp.agraphs.resize_with(m, || None);
            }
            sp.agraphs[m - 1].take()
        };
        let mut g = match taken {
            Some(g) => g,
            None => AdvanceGraph {
                graph: None,
                tokens: airun_cuda::PinnedBuffer::new(m)?,
                positions: airun_cuda::PinnedBuffer::new(m + k)?,
            },
        };
        for (i, &t) in toks.iter().enumerate() {
            g.tokens.as_mut_slice()[i] = t as i32;
        }
        // The head's rows at pos0 .. pos0+m-1, then the chained steps after.
        for i in 0..m + k {
            g.positions.as_mut_slice()[i] = (pos0 + i) as i32;
        }
        // Everything after the commit runs on the output device, where the
        // head lives. With one GPU the commit goes into the graph too; split
        // across GPUs it touches every device, so it is queued on their own
        // streams first (no wait: each stream orders it before its next pass).
        let si = self.states.len() - 1;
        let with_commit = si == 0;
        if !with_commit {
            self.commit(m)?;
        }
        device::set_device(self.states[si].device)?;
        let built = if g.graph.is_none() {
            // SAFETY of the capture: everything queued lives in `self` or `g`,
            // and `g` goes back into `self.spec` below.
            airun_cuda::Graph::begin_capture(&self.states[si].stream)?;
            let queued = self.queue_advance(m, g.tokens.as_slice(), g.positions.as_slice(), with_commit);
            let ended = airun_cuda::Graph::end_capture(&self.states[si].stream);
            queued.and_then(|_| Ok(ended?)).map(|gr| g.graph = Some(gr))
        } else {
            Ok(())
        };
        let launched = built.and_then(|_| {
            Ok(g.graph.as_ref().expect("graph was just built").launch(&self.states[si].stream)?)
        });
        self.spec.as_mut().expect("set up by spec_init").agraphs[m - 1] = Some(g);
        launched?;
        self.states[si].stream.synchronize()?;
        Ok(self.sp().picks_host.as_slice()[..k].iter().map(|&t| t as u32).collect())
    }

    /// The advance on the output device's stream; `with_commit` includes the
    /// delta-net commit, which is only capturable when that is the one device.
    fn queue_advance(&mut self, m: usize, toks: &[i32], positions: &[i32], with_commit: bool) -> Result<()> {
        let k = self.draft;
        let d = self.model.config.n_embd;
        let row = d * std::mem::size_of::<f32>();
        let si = self.states.len() - 1;
        if with_commit {
            self.commit(m)?;
        }
        self.stage_h(false, m)?;
        self.save_pending(m - 1)?;
        {
            let st = &self.states[si];
            st.tokens.copy_from_host_at(0, toks, &st.stream)?;
            st.positions.copy_from_host_at(0, &positions[..m], &st.stream)?;
        }
        self.mtp_body(m, 0)?;
        for i in 1..k {
            let st = &self.states[si];
            let sp = self.sp();
            let last = if i == 1 { m - 1 } else { 0 };
            ops::copy_2d(sp.h.as_mut_ptr(), row, st.xb.offset_ptr(last * d), row, row, 1, &st.stream)?;
            ops::copy_2d(st.tokens.as_mut_ptr(), 4, sp.picks.offset_ptr(i - 1), 4, 4, 1, &st.stream)?;
            st.positions.copy_from_host_at(0, &positions[m - 1 + i..m + i], &st.stream)?;
            self.mtp_body(1, i)?;
        }
        let st = &self.states[si];
        let sp = self.spec.as_mut().expect("set up by spec_init");
        sp.picks.copy_to_host(&mut sp.picks_host.as_mut_slice()[..k], &st.stream)?;
        Ok(())
    }

    /// Reads `picks[0..n]` back to the host.
    fn read_picks(&mut self, n: usize) -> Result<Vec<u32>> {
        let st = self.states.last().expect("a runtime has at least one device");
        let sp = self.spec.as_mut().expect("speculation was set up by spec_init");
        device::set_device(st.device)?;
        sp.picks.copy_to_host(&mut sp.picks_host.as_mut_slice()[..n], &st.stream)?;
        st.stream.synchronize()?;
        Ok(sp.picks_host.as_slice()[..n].iter().map(|&t| t as u32).collect())
    }

    /// Drafts `count` tokens starting from `first`, the pick of the head's
    /// row at `pos` (whose output hidden is the last of `rows` rows in xb):
    /// each further step pairs the previous step's hidden with its pick.
    fn mtp_chain(&mut self, first: u32, pos: usize, rows: usize, count: usize) -> Result<Vec<u32>> {
        let mut drafts = vec![first];
        let mut last_row = rows - 1;
        let d = self.model.config.n_embd;
        while drafts.len() < count {
            {
                let st = self.states.last().expect("a runtime has at least one device");
                let row = d * std::mem::size_of::<f32>();
                ops::copy_2d(self.sp().h.as_mut_ptr(), row, st.xb.offset_ptr(last_row * d), row, row, 1,
                    &st.stream)?;
            }
            let prev = *drafts.last().expect("drafts start with one token");
            let pick = self.mtp_run(&[prev], pos + drafts.len())?;
            drafts.push(pick);
            last_row = 0;
        }
        Ok(drafts)
    }

    /// Pushes the first `m` tokens of the last verify into every delta-net
    /// layer's convolution history and recurrent state.
    fn commit(&self, m: usize) -> Result<()> {
        let cfg = &self.model.config;
        let ssm = cfg.ssm.as_ref().expect("speculation runs on hybrids");
        for st in &self.states {
            device::set_device(st.device)?;
            let h = st.hybrid.as_ref().expect("hybrid scratch on a hybrid model");
            for (local, sh) in st.stash.iter().enumerate() {
                let Some(sh) = sh else { continue };
                let LayerState::Delta { state, conv } = &st.states[local] else { continue };
                let Block::Linear(l) = &self.model.layers[st.layers[local]].block else { continue };
                // The convolution's outputs land in scratch and are dropped;
                // only its history advances.
                copy_rows(&h.qkv, &sh.raw, m * ssm.qkv_dim(), &st.stream)?;
                hybrid::causal_conv1d_ex(h.qkv.as_mut_ptr(), l.conv1d.as_ptr(), conv.as_mut_ptr(), m,
                    ssm.qkv_dim(), ssm.conv_kernel, std::env::var("AIRUN_SSM_NO_CONV_SILU").is_err(),
                    std::env::var("AIRUN_SSM_CONV_REVERSE").is_ok(), true, &st.stream)?;
                hybrid::delta_rule_ex(state.as_mut_ptr(), sh.q.as_ptr(), sh.k.as_ptr(), sh.v.as_ptr(),
                    sh.decay.as_ptr(), sh.beta.as_ptr(), st.attn.as_mut_ptr(), m, ssm.n_k_heads,
                    ssm.n_v_heads, ssm.head_k_dim(), ssm.head_v_dim(), true, &st.stream)?;
            }
        }
        Ok(())
    }

    /// The main model over `tokens` from the current position, advancing
    /// nothing, and every row's greedy pick. The hidden rows stay in xb of the
    /// output device until the next pass.
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        let n = tokens.len();
        let graphed = n == self.draft + 1 && self.states.len() == 1 && !graphs_disabled();
        if !graphed {
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let positions: Vec<i32> = (0..n).map(|i| (self.n_past + i) as i32).collect();
            self.queue_verify(&ids, &positions)?;
            self.queue_picks(n)?;
            return self.read_picks(n);
        }

        let mut vg = match self.spec.as_mut().expect("set up by spec_init").vgraph.take() {
            Some(vg) => vg,
            None => VerifyGraph {
                graph: None,
                tokens: airun_cuda::PinnedBuffer::new(n)?,
                positions: airun_cuda::PinnedBuffer::new(n)?,
            },
        };
        for (i, &t) in tokens.iter().enumerate() {
            vg.tokens.as_mut_slice()[i] = t as i32;
            vg.positions.as_mut_slice()[i] = (self.n_past + i) as i32;
        }
        device::set_device(self.states[0].device)?;
        let built = if vg.graph.is_none() {
            // SAFETY of the capture: everything queued lives in `self` or `vg`,
            // and `vg` goes back into `self.spec` below.
            airun_cuda::Graph::begin_capture(&self.states[0].stream)?;
            let queued = self.queue_verify(vg.tokens.as_slice(), vg.positions.as_slice())
                .and_then(|_| self.queue_picks(n));
            let ended = airun_cuda::Graph::end_capture(&self.states[0].stream);
            queued.and_then(|_| Ok(ended?)).map(|g| vg.graph = Some(g))
        } else {
            Ok(())
        };
        let launched = built.and_then(|_| {
            Ok(vg.graph.as_ref().expect("graph was just built").launch(&self.states[0].stream)?)
        });
        self.spec.as_mut().expect("set up by spec_init").vgraph = Some(vg);
        launched?;
        let st = &self.states[0];
        st.stream.synchronize()?;
        let sp = self.sp();
        Ok(sp.picks_host.as_slice()[..n].iter().map(|&t| t as u32).collect())
    }

    /// The verify's forward pass, with delta-net layers deferring.
    fn queue_verify(&mut self, ids: &[i32], positions: &[i32]) -> Result<()> {
        self.defer = true;
        self.skip_logits = true;
        let queued = self.queue_forward(ids, positions);
        self.defer = false;
        self.skip_logits = false;
        queued
    }

    /// Every row's logits, their argmaxes, and the copy of those to the host.
    fn queue_picks(&mut self, n: usize) -> Result<()> {
        let cfg = &self.model.config;
        let st = self.states.last().expect("a runtime has at least one device");
        let sp = self.spec.as_mut().expect("set up by spec_init");
        device::set_device(st.device)?;
        let out_w = self.model.output_tensor();
        matvec(out_w, &st.xb, &sp.logits, n, &st.q8act, &st.stream)?;
        for i in 0..n {
            ops::argmax(sp.logits.offset_ptr(i * cfg.vocab_size), cfg.vocab_size, sp.picks.offset_ptr(i),
                sp.scratch.as_mut_ptr(), std::ptr::null_mut(), &st.stream)?;
        }
        sp.picks.copy_to_host(&mut sp.picks_host.as_mut_slice()[..n], &st.stream)?;
        Ok(())
    }

    /// [`Runtime::generate`] for greedy decoding with the MTP head drafting.
    pub(super) fn generate_spec(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        sampler: &mut crate::sample::Sampler,
        on_token: &mut dyn FnMut(u32, &str) -> bool,
    ) -> Result<GenerationStats> {
        // Pure greedy compares the drafts with the GPU's argmaxes; anything
        // else reads the verified rows' logits and samples against the drafts
        // (Sampler::sample_draft), which keeps the sampler's distribution.
        let sampling = !sampler.is_pure_greedy();
        let vocab = self.model.config.vocab_size;
        let mut row_logits: Vec<f32> = Vec::new();
        let mut hist: Vec<u32> = prompt.iter().copied().filter(|&t| !crate::runtime::is_image_token(t)).collect();
        let start = std::time::Instant::now();
        self.spec_init()?;
        let k = self.draft;

        // The head's cache must cover every cached position; one a plain
        // generation left behind is rebuilt along with the rest.
        let mut cached = self.reusable_prefix(prompt);
        if cached > 0 && !self.sp().synced {
            cached = 0;
        }
        self.rewind_to(cached);
        if cached == 0 {
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            self.sp().pending.fill_zero(&st.stream)?;
        }
        self.spec.as_mut().expect("set up above").synced = false;

        // Prefill: each chunk through the main model, then its rows through
        // the head, pairing each token with the hidden before it.
        let mut prompt_done = cached;
        let mut t = 0u32;
        while prompt_done < prompt.len() {
            let take = (prompt.len() - prompt_done).min(self.max_batch);
            let chunk = prompt[prompt_done..prompt_done + take].to_vec();
            let pos0 = self.n_past;
            self.step(&chunk)?;
            prompt_done += take;
            if prompt_done == prompt.len() {
                let st = self.states.last().expect("a runtime has at least one device");
                device::set_device(st.device)?;
                let logits = st.logits.as_ref().ok_or_else(|| CoreError::Other(
                    "output shard has no logits buffer".into()))?;
                if sampling {
                    row_logits.resize(vocab, 0.0);
                    logits.copy_to_host(&mut row_logits[..vocab], &st.stream)?;
                    st.stream.synchronize()?;
                    t = sampler.sample(&mut row_logits[..vocab], &hist);
                } else {
                    let sp = self.sp();
                    ops::argmax(logits.as_ptr(), logits.len(), sp.picks.as_mut_ptr(), sp.scratch.as_mut_ptr(),
                        std::ptr::null_mut(), &st.stream)?;
                    t = self.read_picks(1)?[0];
                }
            }
            self.stage_h(true, take - 1)?;
            self.save_pending(take - 1)?;
            // Queued, not waited on: its pick is not needed, and a wait here
            // would stop the first GPU from starting the next chunk while the
            // last one finishes this one (27B on two GPUs, 61k-token prompt:
            // 615 -> see AGENTS.md).
            let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
            let pos: Vec<i32> = (0..chunk.len()).map(|i| (pos0 + i) as i32).collect();
            self.mtp_queue(&ids, &pos)?;
        }
        self.spec.as_mut().expect("set up above").synced = true;
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = std::time::Instant::now();
        let mut decoder = self.tokenizer.stream_decoder();
        let mut generated = 0usize;
        let mut last_token = t;
        let (mut drafted, mut accepted) = (0usize, 0usize);
        let mut stop = StopReason::MaxTokens;
        let mut timing = [0f64; 3];
        let mut steps = 0usize;

        // Hands one token out; Some(reason) when generation ends at it.
        let mut emit = |rt: &Runtime, tok: u32, generated: &mut usize,
                        decoder: &mut crate::tokenizer::StreamDecoder| -> Option<StopReason> {
            if *generated >= max_tokens {
                return Some(StopReason::MaxTokens);
            }
            if rt.tokenizer.is_eog(tok) {
                return Some(StopReason::EndOfGeneration);
            }
            let text = decoder.push(&rt.tokenizer, tok);
            *generated += 1;
            if !on_token(tok, &text) {
                return Some(StopReason::Cancelled);
            }
            None
        };

        // t is the newest token: handed out, not yet in the caches.
        hist.push(t);
        if let Some(s) = emit(self, t, &mut generated, &mut decoder) {
            stop = s;
        } else {
            // The head's row at n_past pairs the last prompt hidden with t.
            let mut drafts = Vec::new();
            let room = k.min(self.ctx.saturating_sub(self.n_past + 1));
            if room > 0 {
                self.stage_h(true, 0)?;
                let first = self.mtp_run(&[t], self.n_past)?;
                drafts = self.mtp_chain(first, self.n_past, 1, room)?;
            }
            loop {
                if self.n_past + 1 + drafts.len() > self.ctx {
                    stop = StopReason::ContextFull;
                    break;
                }
                let p = self.n_past;
                let mut input = vec![t];
                input.extend_from_slice(&drafts);
                let t0 = std::time::Instant::now();
                let picks = self.verify(&input)?;
                timing[0] += t0.elapsed().as_secs_f64();
                let mut a = 0;
                let mut sampled = None;
                if sampling {
                    // The verify left every row's logits on the output device.
                    let n = input.len();
                    row_logits.resize(n * vocab, 0.0);
                    {
                        let st = self.states.last().expect("a runtime has at least one device");
                        device::set_device(st.device)?;
                        self.sp().logits.copy_to_host(&mut row_logits[..n * vocab], &st.stream)?;
                        st.stream.synchronize()?;
                    }
                    let mut h = hist.clone();
                    while a < drafts.len() {
                        let row = &mut row_logits[a * vocab..(a + 1) * vocab];
                        let (tok, took) = sampler.sample_draft(row, &h, drafts[a]);
                        if !took {
                            sampled = Some(tok);
                            break;
                        }
                        h.push(tok);
                        a += 1;
                    }
                    if sampled.is_none() {
                        let row = &mut row_logits[a * vocab..(a + 1) * vocab];
                        sampled = Some(sampler.sample(row, &h));
                    }
                } else {
                    while a < drafts.len() && drafts[a] == picks[a] {
                        a += 1;
                    }
                }
                drafted += drafts.len();
                accepted += a;

                // Hand out the accepted drafts, then the main model's own next
                // token, stopping wherever plain decoding would.
                let mut committed = 1; // t
                let mut ended = None;
                for &tok in &drafts[..a] {
                    if let Some(s) = emit(self, tok, &mut generated, &mut decoder) {
                        ended = Some(s);
                        break;
                    }
                    last_token = tok;
                    hist.push(tok);
                    committed += 1;
                }
                let next = match sampled {
                    Some(s) if committed - 1 == a => s,
                    _ => picks[committed - 1],
                };
                if ended.is_none() {
                    ended = emit(self, next, &mut generated, &mut decoder);
                    if ended.is_none() {
                        last_token = next;
                        hist.push(next);
                    }
                }

                let t1 = std::time::Instant::now();
                let room = k.min(self.ctx.saturating_sub(p + committed + 1));
                if ended.is_none() && room == k && k > 0 && !graphs_disabled() {
                    let mut toks = input[1..committed].to_vec();
                    toks.push(next);
                    drafts = self.advance_graphed(&toks, p + 1)?;
                    self.n_past = p + committed;
                    self.history.extend_from_slice(&input[..committed]);
                    t = next;
                    timing[1] += t1.elapsed().as_secs_f64();
                    steps += 1;
                    continue;
                }
                self.commit(committed)?;
                self.n_past = p + committed;
                self.history.extend_from_slice(&input[..committed]);

                // The head's rows p+1 .. pair the verify hiddens with the
                // tokens after them; `pending` keeps the last committed one for
                // whatever prefill comes next.
                self.stage_h(false, committed)?;
                self.save_pending(committed - 1)?;
                if ended.is_some() {
                    if committed > 1 {
                        self.mtp_run(&input[1..committed], p + 1)?;
                    }
                    stop = ended.expect("checked");
                    break;
                }
                let mut toks = input[1..committed].to_vec();
                toks.push(next);
                let first = self.mtp_run(&toks, p + 1)?;
                timing[1] += t1.elapsed().as_secs_f64();
                let t2 = std::time::Instant::now();
                t = next;
                let room = k.min(self.ctx.saturating_sub(self.n_past + 1));
                drafts = if room == 0 { Vec::new() } else {
                    self.mtp_chain(first, self.n_past, toks.len(), room)?
                };
                timing[2] += t2.elapsed().as_secs_f64();
                steps += 1;
            }
        }

        drop(emit);
        if std::env::var("AIRUN_SPEC_TIMING").is_ok() && steps > 0 {
            let ms = |x: f64| 1000.0 * x / steps as f64;
            eprintln!("spec step: verify {:.2} ms, commit+absorb {:.2} ms, chain {:.2} ms ({steps} steps)",
                ms(timing[0]), ms(timing[1]), ms(timing[2]));
        }
        let tail = decoder.flush();
        if !tail.is_empty() {
            on_token(last_token, &tail);
        }
        Ok(GenerationStats {
            prompt_tokens: prompt.len(),
            cached_tokens: cached,
            generated_tokens: generated,
            prefill_ms,
            decode_ms: decode_start.elapsed().as_secs_f64() * 1000.0,
            stop,
            drafted,
            accepted,
        })
    }
}
