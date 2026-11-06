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
    pub(super) pending: DeviceBuffer<f32>,
    /// k + 1 rows of logits from a verify.
    logits: DeviceBuffer<f32>,
    picks: DeviceBuffer<i32>,
    picks_host: airun_cuda::PinnedBuffer<i32>,
    scratch: DeviceBuffer<i32>,
    /// The head's cache and `pending` describe exactly the first `n_past`
    /// positions. Anything that advances the caches without the head clears it.
    pub(super) synced: bool,
    /// The verify of k + 1 tokens as a CUDA graph (one device only).
    pub(super) vgraph: Option<VerifyGraph>,
    /// The same split across GPUs: one graph per shard.
    pub(super) vsgraph: Option<ShardedVerifyGraph>,
    /// The one-row head step as a CUDA graph.
    mgraph: Option<VerifyGraph>,
    /// [`Runtime::advance_graphed`]'s graphs, by accepted count - 1.
    pub(super) agraphs: Vec<Option<AdvanceGraph>>,
    /// Batched steps (verify_rows / accept_rows / draft_rows): each row's
    /// head hidden for the next draft, the draft logits and the picks
    /// ([step][row] for drafts, [row][token] for a verify).
    rows_hsel: DeviceBuffer<f32>,
    rows_logits: DeviceBuffer<f32>,
    rows_picks: DeviceBuffer<i32>,
    rows_picks_host: airun_cuda::PinnedBuffer<i32>,
}

/// The MTP head state of a slot that is not the active one (the active
/// slot's lives in [`Spec`]): its cache, the hidden its next row pairs with,
/// and whether both cover exactly what the slot holds.
pub(super) struct SlotHead {
    kv: LayerState,
    pending: DeviceBuffer<f32>,
    synced: bool,
}

/// What a batched verify ran, for [`Runtime::accept_rows`]: each row's slot
/// and tokens (the newest, then its drafts), all of `t` tokens.
pub(super) struct RowsSpec {
    rows: Vec<(usize, Vec<u32>)>,
    t: usize,
}

/// A batched verify's result: every row's greedy pick after each of its
/// tokens (`t` per row), and the logits behind them when asked for.
pub struct RowsVerify<'a> {
    pub picks: Vec<u32>,
    pub t: usize,
    pub logits: Option<&'a mut [f32]>,
}

impl Spec {
    /// Puts `incoming` (the newly active slot's head) in place and returns
    /// the one it replaces. A slot without a head of its own gets this one,
    /// out of sync.
    pub(super) fn swap_head(&mut self, incoming: Option<SlotHead>) -> Option<SlotHead> {
        let Some(h) = incoming else {
            self.synced = false;
            return None;
        };
        Some(SlotHead {
            kv: std::mem::replace(&mut self.kv, h.kv),
            pending: std::mem::replace(&mut self.pending, h.pending),
            synced: std::mem::replace(&mut self.synced, h.synced),
        })
    }

    /// Drops the graph that captured the head cache's address.
    pub(super) fn forget_head_graph(&mut self) {
        self.mgraph = None;
    }
}

/// A verify split across GPUs, captured as one graph per shard and replayed on
/// each device's stream; the residual rows cross between GPUs outside the
/// graphs (handoff), as in [`DecodeGraph`]. Issued launch by launch, a split
/// verify of the 27B took 50.7 ms against 40.5 ms graphed on one GPU.
pub(super) struct ShardedVerifyGraph {
    graphs: Vec<Option<airun_cuda::Graph>>,
    tokens: airun_cuda::PinnedBuffer<i32>,
    positions: airun_cuda::PinnedBuffer<i32>,
}

/// How many tokens the draft head chooses among: the first rows of the output
/// head, which in a byte-level BPE are the earliest merges and so roughly the
/// most frequent tokens (FR-Spec, arXiv 2502.14856, ranks by corpus counts).
/// Every draft is still checked by the full-vocabulary verify, so this trades
/// acceptance for a cheaper head step, never the output. AIRUN_DRAFT_VOCAB
/// sets it; 0 or anything at least the vocabulary means all of it.
fn draft_vocab(vocab: usize) -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let k = *K.get_or_init(|| std::env::var("AIRUN_DRAFT_VOCAB").ok()
        .and_then(|v| v.trim().parse().ok()).unwrap_or(0));
    if k == 0 { vocab } else { k.min(vocab) }
}

/// Drafts per conversation in a batched step of 2, 3, 4... (see
/// Runtime::rows_draft); past the end, a plain batched step. Qwen3.8-27B
/// NVFP4 on 5070 Ti + 5060, --draft 3, tok/s summed (plain batching in
/// brackets): 2 rows 81.5 with 2 drafts, 78.5 with 3 [59]; 3 rows 98 with
/// 1, 81 with 2 [82]; 4 rows 117 with 1, 98 with 2 [106].
const ROWS_DRAFT: &[usize] = &[2, 1, 1];

/// AIRUN_NO_VERIFY_SHARD_GRAPHS=1 issues a split verify launch by launch.
fn shard_graphs_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_VERIFY_SHARD_GRAPHS").is_ok_and(|v| v.trim() != "0"))
}

/// A captured advance (commit, head rows, k drafts) and its pinned inputs.
pub(super) struct AdvanceGraph {
    graph: Option<airun_cuda::Graph>,
    tokens: airun_cuda::PinnedBuffer<i32>,
    positions: airun_cuda::PinnedBuffer<i32>,
}

/// A captured verify pass and the pinned staging its replays read, as
/// [`DecodeGraph`] for decode: a verify is ~2000 launches, which issued one by
/// one left the GPU idle for a quarter of the pass.
pub(super) struct VerifyGraph {
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

    pub fn draft(&self) -> usize {
        self.draft
    }

    /// Whether a speculative [`Runtime::generate`] of `prompt` keeps the
    /// prefix the caches already hold. It does only when the head's cache
    /// covers that prefix too; otherwise it prefills the whole prompt again
    /// (after another --parallel slot ran, say), and a long conversation is
    /// better continued without drafts.
    pub fn spec_keeps_prefix(&self, prompt: &[u32]) -> bool {
        let cached = self.reusable_prefix(prompt);
        if cached == 0 {
            return true;
        }
        if cached < self.history.len() {
            self.snapshot_has_head()
        } else {
            self.spec.as_ref().is_some_and(|s| s.synced)
        }
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
            // Room for a batched verify of every slot (verify_rows).
            let all = rows * self.slots.len().max(1);
            st.stash = st.layers.iter().map(|&l| -> Result<Option<DeltaStash>> {
                Ok(match cfg.layer_kind(l) {
                    LayerKind::LinearAttention => Some(DeltaStash {
                        raw: DeviceBuffer::new(dev, all * ssm.qkv_dim())?,
                        q: DeviceBuffer::new(dev, all * ssm.k_dim())?,
                        k: DeviceBuffer::new(dev, all * ssm.k_dim())?,
                        v: DeviceBuffer::new(dev, all * ssm.v_dim())?,
                        decay: DeviceBuffer::new(dev, all * ssm.n_v_heads)?,
                        beta: DeviceBuffer::new(dev, all * ssm.n_v_heads)?,
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
            vsgraph: None,
            mgraph: None,
            agraphs: Vec::new(),
            rows_hsel: DeviceBuffer::new(dev, self.slots.len().max(1) * cfg.n_embd)?,
            rows_logits: DeviceBuffer::new(dev, self.slots.len().max(1) * draft_vocab(cfg.vocab_size))?,
            rows_picks: DeviceBuffer::new(dev, self.slots.len().max(1) * rows)?,
            rows_picks_host: airun_cuda::PinnedBuffer::new(self.slots.len().max(1) * rows)?,
        });
        for s in 0..self.slots.len() {
            self.slots[s].head = None;
        }
        self.ensure_heads()?;
        Ok(())
    }

    /// A head cache for every slot that has none (the planner counted one
    /// per slot: Config::kv_bytes_per_layer).
    fn ensure_heads(&mut self) -> Result<()> {
        let cfg = &self.model.config;
        let st = self.states.last().expect("a runtime has at least one device");
        device::set_device(st.device)?;
        let kv_bytes = cfg.n_head_kv * self.ctx * cfg.kv_type.row_bytes(cfg.head_dim);
        for s in 0..self.slots.len() {
            if s == self.active_slot || self.slots[s].head.is_some() {
                continue;
            }
            self.slots[s].head = Some(SlotHead {
                kv: LayerState::Kv {
                    k: DeviceBuffer::zeroed(st.device, kv_bytes, &st.stream)?,
                    v: DeviceBuffer::zeroed(st.device, kv_bytes, &st.stream)?,
                    qsa: None,
                },
                pending: DeviceBuffer::zeroed(st.device, cfg.n_embd, &st.stream)?,
                synced: false,
            });
        }
        Ok(())
    }

    /// Slot `slot`'s head cache.
    pub(super) fn head_kv(&self, slot: usize) -> Result<&LayerState> {
        if slot == self.active_slot {
            return Ok(&self.sp().kv);
        }
        self.slots[slot].head.as_ref().map(|h| &h.kv).ok_or_else(|| CoreError::Other(
            "a batched draft reached a slot without a head cache; this is a bug, run with --draft 0".into()))
    }

    fn head_pending(&self, slot: usize) -> Option<&DeviceBuffer<f32>> {
        if slot == self.active_slot {
            self.spec.as_ref().map(|sp| &sp.pending)
        } else {
            self.slots[slot].head.as_ref().map(|h| &h.pending)
        }
    }

    /// Whether slot `slot`'s head covers exactly what the slot holds, so its
    /// drafts are worth verifying.
    pub fn head_synced(&self, slot: usize) -> bool {
        let Some(sp) = self.spec.as_ref() else { return false };
        if slot == self.active_slot { sp.synced } else { self.slots[slot].head.as_ref().is_some_and(|h| h.synced) }
    }

    pub(super) fn set_head_synced(&mut self, slot: usize, synced: bool) {
        if slot == self.active_slot {
            if let Some(sp) = self.spec.as_mut() {
                sp.synced = synced;
            }
        } else if let Some(h) = self.slots.get_mut(slot).and_then(|s| s.head.as_mut()) {
            h.synced = synced;
        }
    }

    /// Drafts per conversation for a batched speculative step over `rows`
    /// of them; 0 means a plain batched step. A verify of rows x (1 + k)
    /// tokens costs more per token the more there are (an NVFP4 matmul of the
    /// 27B: 50 us for 4 tokens, 86 for 8, ~175 for 16), so k shrinks as rows
    /// grow. AIRUN_ROWS_DRAFT="k2,k3,k4,..." sets it for 2, 3, 4... rows.
    pub fn rows_draft(&self, rows: usize) -> usize {
        if !self.speculates() || rows < 2 || rows > self.slots.len() || std::env::var("AIRUN_NO_ROWS_SPEC").is_ok() {
            return 0;
        }
        static TABLE: std::sync::OnceLock<Option<Vec<usize>>> = std::sync::OnceLock::new();
        let table = TABLE.get_or_init(|| std::env::var("AIRUN_ROWS_DRAFT").ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()));
        let k = match table {
            Some(t) => t.get(rows - 2).or(t.last()).copied().unwrap_or(0),
            None => ROWS_DRAFT.get(rows - 2).copied().unwrap_or(0),
        };
        let k = k.min(self.draft);
        if rows * (k + 1) > self.max_batch { 0 } else { k }
    }

    /// The prefill's head step for the chunk just run through the main model
    /// (`pos0` its first position): rows pairing each token with the hidden
    /// before it, as generate_spec does. Queued, not waited on.
    pub(super) fn head_prefill_chunk(&mut self, chunk: &[u32], pos0: usize) -> Result<()> {
        self.spec_init()?;
        self.stage_h(true, chunk.len() - 1)?;
        self.save_pending(chunk.len() - 1)?;
        let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
        let pos: Vec<i32> = (0..chunk.len()).map(|i| (pos0 + i) as i32).collect();
        self.mtp_queue(&ids, &pos)
    }

    /// Zeroes the active slot's `pending` (a prefill from position 0).
    pub(super) fn head_reset(&mut self) -> Result<()> {
        self.spec_init()?;
        let st = self.states.last().expect("a runtime has at least one device");
        device::set_device(st.device)?;
        self.sp().pending.fill_zero(&st.stream)?;
        Ok(())
    }

    /// One batched speculative verify: row b runs `rows[b].1` (the slot's
    /// newest token, then its drafts; every row the same length) through the
    /// main model at the slot's next positions, advancing nothing. Returns
    /// each token's greedy pick, plus the logits when `logits` (for sampled
    /// rows). Complete it with [`Self::accept_rows`] before anything else
    /// runs on these slots.
    pub fn verify_rows(&mut self, rows: &[(usize, Vec<u32>)], logits: bool) -> Result<RowsVerify<'_>> {
        self.spec_init()?;
        self.ensure_heads()?;
        let n_rows = rows.len();
        let t = rows.first().map_or(0, |r| r.1.len());
        if t == 0 || rows.iter().any(|r| r.1.len() != t) || n_rows * t > self.max_batch {
            return Err(CoreError::Other("verify_rows needs rows of one length within the batch".into()));
        }
        let n = n_rows * t;
        let mut seen = std::collections::HashSet::new();
        let (mut ids, mut positions, mut rope, mut past) = (Vec::with_capacity(n), Vec::with_capacity(n),
            vec![0i32; 3 * n], Vec::with_capacity(n_rows));
        for (b, (slot, toks)) in rows.iter().enumerate() {
            let slot = *slot;
            if slot >= self.slots.len() || !seen.insert(slot) {
                return Err(CoreError::Other(format!("verify_rows needs distinct, existing slots (row {b})")));
            }
            let (p, shift) = if slot == self.active_slot { (self.n_past, self.rope_shift) }
                else { (self.slots[slot].n_past, self.slots[slot].rope_shift) };
            if p + t > self.ctx {
                return Err(CoreError::Other(format!(
                    "slot {slot} has no room for {t} tokens in its {} context; this is a bug", self.ctx)));
            }
            past.push(p);
            for (i, &tok) in toks.iter().enumerate() {
                ids.push(tok as i32);
                positions.push((p + i) as i32);
                for a in 0..3 {
                    rope[a * n + b * t + i] = (p + i) as i32 - shift;
                }
            }
        }
        for st in &self.states {
            device::set_device(st.device)?;
            st.mpos.copy_from_host_at(0, &rope, &st.stream)?;
        }
        let slot_rows: Vec<(usize, u32)> = rows.iter().map(|(s, v)| (*s, v[0])).collect();
        self.fill_rows_tab(&slot_rows)?;
        self.mixed = Some(rows.iter().map(|r| r.0).collect());
        self.mixed_t = t;
        self.mixed_past = past;
        self.image_rows = Some(n);
        self.defer = true;
        self.skip_logits = true;
        let graphed = !graphs_disabled() && !mixed_graphs_disabled() && !self.mixed_graph_off
            && !self.states.iter().any(|s| s.moe.as_ref().is_some_and(|m| m.upload.is_some()));
        let queued = if graphed {
            let mut launched = false;
            match self.queue_rows_graphed(&ids, &positions, &mut launched) {
                Err(e) if !launched => {
                    eprintln!("note: batched verify runs without CUDA graphs: {e}");
                    self.mixed_graph_off = true;
                    self.queue_forward(&ids, &positions)
                }
                r => r,
            }
        } else {
            self.queue_forward(&ids, &positions)
        };
        self.mixed = None;
        self.mixed_t = 1;
        self.image_rows = None;
        self.defer = false;
        self.skip_logits = false;
        queued?;

        // Every token's logits and pick on the output device, and its hidden
        // kept for the head.
        let cfg = &self.model.config;
        let (vocab, d) = (cfg.vocab_size, cfg.n_embd);
        let st = self.states.last().expect("a runtime has at least one device");
        device::set_device(st.device)?;
        let row = d * std::mem::size_of::<f32>();
        ops::copy_2d(self.sp().h.as_mut_ptr(), row, st.xb.as_ptr(), row, row, n, &st.stream)?;
        let out_w = self.model.output_tensor();
        let need = n * vocab;
        if self.rows_logits.as_ref().is_none_or(|(_, h)| h.len() < need) {
            self.rows_logits = None;
            self.rows_logits = Some((DeviceBuffer::<f32>::new(st.device, need)?, airun_cuda::PinnedBuffer::new(need)?));
        }
        let sp = self.spec.as_mut().expect("set up by spec_init");
        let (dev, host) = self.rows_logits.as_mut().expect("allocated above");
        const AT_ONCE: usize = 8;
        for r0 in (0..n).step_by(AT_ONCE) {
            let m = AT_ONCE.min(n - r0);
            let out = dev.offset_ptr(r0 * vocab);
            ops::matvec(out_w.kernel_type(), out_w.ptr(), st.xb.offset_ptr(r0 * d), out, d, vocab, m,
                out_w.row_bytes, &st.stream)?;
            if out_w.global_scale != 1.0 {
                ops::scale(out, out_w.global_scale, m * vocab, &st.stream)?;
            }
        }
        for i in 0..n {
            ops::argmax(dev.offset_ptr(i * vocab), vocab, sp.rows_picks.offset_ptr(i), sp.scratch.as_mut_ptr(),
                std::ptr::null_mut(), &st.stream)?;
        }
        sp.rows_picks.copy_to_host(&mut sp.rows_picks_host.as_mut_slice()[..n], &st.stream)?;
        if logits {
            dev.copy_to_host(&mut host.as_mut_slice()[..need], &st.stream)?;
        }
        st.stream.synchronize()?;
        let picks = sp.rows_picks_host.as_slice()[..n].iter().map(|&p| p as u32).collect();
        self.rows_spec = Some(RowsSpec { rows: rows.to_vec(), t });
        let logits = if logits { Some(&mut host.as_mut_slice()[..need]) } else { None };
        Ok(RowsVerify { picks, t, logits })
    }

    /// Completes the last [`Self::verify_rows`]: `accept[b] = (m, next)` --
    /// row b's first m tokens go into its slot's caches (the newest and the
    /// m - 1 drafts accepted), and `next`, the token after them, starts its
    /// next step. Feeds every row's head and returns its next drafts.
    pub fn accept_rows(&mut self, accept: &[(usize, u32)], k: usize) -> Result<Vec<Vec<u32>>> {
        let RowsSpec { rows, t } = self.rows_spec.take().ok_or_else(|| CoreError::Other(
            "accept_rows without a verify_rows before it; this is a bug".into()))?;
        if accept.len() != rows.len() || accept.iter().any(|&(m, _)| m == 0 || m > t) {
            return Err(CoreError::Other("accept_rows needs 1..=t tokens for every verified row".into()));
        }
        self.commit_rows(&rows, t, accept)?;
        let mut bases = Vec::with_capacity(rows.len());
        for ((slot, toks), &(m, _)) in rows.iter().zip(accept) {
            let slot = *slot;
            let p = if slot == self.active_slot {
                let p = self.n_past;
                self.n_past += m;
                self.history.extend_from_slice(&toks[..m]);
                p
            } else {
                let s = &mut self.slots[slot];
                let p = s.n_past;
                s.n_past += m;
                s.history.extend_from_slice(&toks[..m]);
                p
            };
            bases.push(p);
        }
        // The head's rows p+1 .. p+t pair the verify's hiddens (already in
        // spec.h) with the tokens after them: the accepted drafts and next,
        // then filler rows past the end, rewritten before anything attends to
        // them. Its pick after row m - 1 is the first draft.
        let d = self.model.config.n_embd;
        let n = rows.len() * t;
        let (mut ids, mut pos, mut sel) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::new());
        for (b, ((_, toks), &(m, next))) in rows.iter().zip(accept).enumerate() {
            for i in 0..t {
                ids.push(if i + 1 < m { toks[i + 1] } else { next } as i32);
                pos.push((bases[b] + 1 + i).min(self.ctx - 1) as i32);
            }
            sel.push(m - 1);
        }
        {
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            let row = d * std::mem::size_of::<f32>();
            for (b, ((slot, _), &(m, _))) in rows.iter().zip(accept).enumerate() {
                if let Some(pend) = self.head_pending(*slot) {
                    ops::copy_2d(pend.as_mut_ptr(), row, self.sp().h.offset_ptr((b * t + m - 1) * d), row, row, 1,
                        &st.stream)?;
                }
            }
        }
        let slots: Vec<usize> = rows.iter().map(|r| r.0).collect();
        let first: Vec<usize> = (0..rows.len()).map(|b| bases[b] + accept[b].0).collect();
        self.head_rows(&slots, t, &ids, &pos, &sel, 0)?;
        self.chain_rows(&slots, &first, k)
    }

    /// First drafts for slots that have none: row b's head row at the
    /// slot's next position pairs its `pending` hidden with `rows[b].1`.
    pub fn draft_rows(&mut self, rows: &[(usize, u32)], k: usize) -> Result<Vec<Vec<u32>>> {
        self.spec_init()?;
        self.ensure_heads()?;
        let d = self.model.config.n_embd;
        let mut pos = Vec::with_capacity(rows.len());
        {
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            let row = d * std::mem::size_of::<f32>();
            for (b, &(slot, _)) in rows.iter().enumerate() {
                let pend = self.head_pending(slot).ok_or_else(|| CoreError::Other(
                    "a batched draft reached a slot without a head cache; this is a bug".into()))?;
                ops::copy_2d(self.sp().h.offset_ptr(b * d), row, pend.as_ptr(), row, row, 1, &st.stream)?;
                pos.push(if slot == self.active_slot { self.n_past } else { self.slots[slot].n_past });
            }
        }
        let slots: Vec<usize> = rows.iter().map(|r| r.0).collect();
        let ids: Vec<i32> = rows.iter().map(|r| r.1 as i32).collect();
        let p32: Vec<i32> = pos.iter().map(|&p| p as i32).collect();
        self.head_rows(&slots, 1, &ids, &p32, &vec![0; rows.len()], 0)?;
        self.chain_rows(&slots, &pos, k)
    }

    /// The remaining k - 1 drafts after [`Self::head_rows`] left each row's
    /// first in picks[0..rows] and its hidden in rows_hsel: one head row per
    /// slot at a time, at `first[b] + i`. Returns every row's k drafts.
    fn chain_rows(&mut self, slots: &[usize], first: &[usize], k: usize) -> Result<Vec<Vec<u32>>> {
        let k = k.clamp(1, self.draft.max(1));
        let r = slots.len();
        let d = self.model.config.n_embd;
        for i in 1..k {
            {
                let st = self.states.last().expect("a runtime has at least one device");
                let sp = self.sp();
                let row = d * std::mem::size_of::<f32>();
                ops::copy_2d(sp.h.as_mut_ptr(), row, sp.rows_hsel.as_ptr(), row, row, r, &st.stream)?;
                ops::copy_2d(st.tokens.as_mut_ptr(), 4, sp.rows_picks.offset_ptr((i - 1) * r), 4, 4, r, &st.stream)?;
            }
            let pos: Vec<i32> = first.iter().map(|&p| (p + i).min(self.ctx - 1) as i32).collect();
            self.head_rows(slots, 1, &[], &pos, &vec![0; r], i)?;
        }
        let st = self.states.last().expect("a runtime has at least one device");
        let sp = self.spec.as_mut().expect("set up by spec_init");
        sp.rows_picks.copy_to_host(&mut sp.rows_picks_host.as_mut_slice()[..k * r], &st.stream)?;
        st.stream.synchronize()?;
        let h = sp.rows_picks_host.as_slice();
        Ok((0..r).map(|b| (0..k).map(|i| h[i * r + b] as u32).collect()).collect())
    }

    /// The head over `t` tokens for each of `slots` (row b: tokens
    /// `ids[b*t..]` at `pos[b*t..]`; empty `ids` keeps the tokens already on
    /// the device), reading spec.h, against each slot's own head cache. Row
    /// b's output at token `sel[b]` lands in rows_hsel[b] and its draft pick
    /// in picks[step * rows + b].
    fn head_rows(&mut self, slots: &[usize], t: usize, ids: &[i32], pos: &[i32], sel: &[usize], step: usize)
        -> Result<()> {
        let r = slots.len();
        let n = r * t;
        let mut rope = vec![0i32; 3 * n];
        for (b, &slot) in slots.iter().enumerate() {
            let shift = if slot == self.active_slot { self.rope_shift } else { self.slots[slot].rope_shift };
            for i in 0..t {
                for a in 0..3 {
                    rope[a * n + b * t + i] = pos[b * t + i] - shift;
                }
            }
        }
        {
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            if !ids.is_empty() {
                st.tokens.copy_from_host_at(0, ids, &st.stream)?;
            }
            st.positions.copy_from_host_at(0, pos, &st.stream)?;
            st.mpos.copy_from_host_at(0, &rope, &st.stream)?;
        }
        self.mixed = Some(slots.to_vec());
        self.mixed_t = t;
        self.mixed_past = (0..r).map(|b| pos[b * t] as usize).collect();
        self.mixed_head = true;
        self.image_rows = Some(n);
        let ran = self.head_rows_body(n, sel, t, step);
        self.mixed = None;
        self.mixed_t = 1;
        self.mixed_head = false;
        self.image_rows = None;
        ran
    }

    fn head_rows_body(&self, n: usize, sel: &[usize], t: usize, step: usize) -> Result<()> {
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
        let r = sel.len();
        for (b, &i) in sel.iter().enumerate() {
            ops::copy_2d(sp.rows_hsel.offset_ptr(b * d), row, st.xb.offset_ptr((b * t + i) * d), row, row, 1, stream)?;
        }
        let out_w = self.model.output_tensor();
        let vocab = draft_vocab(cfg.vocab_size);
        const AT_ONCE: usize = 8;
        for r0 in (0..r).step_by(AT_ONCE) {
            let m = AT_ONCE.min(r - r0);
            ops::matvec(out_w.kernel_type(), out_w.ptr(), sp.rows_hsel.offset_ptr(r0 * d),
                sp.rows_logits.offset_ptr(r0 * vocab), d, vocab, m, out_w.row_bytes, stream)?;
        }
        for b in 0..r {
            ops::argmax(sp.rows_logits.offset_ptr(b * vocab), vocab, sp.rows_picks.offset_ptr(step * r + b),
                sp.scratch.as_mut_ptr(), std::ptr::null_mut(), stream)?;
        }
        Ok(())
    }

    /// Pushes row b's first `accept[b].0` verified tokens into its slot's
    /// delta-net states (the batched [`Self::commit`]).
    fn commit_rows(&self, rows: &[(usize, Vec<u32>)], t: usize, accept: &[(usize, u32)]) -> Result<()> {
        let cfg = &self.model.config;
        let ssm = cfg.ssm.as_ref().expect("speculation runs on hybrids");
        let silu = std::env::var("AIRUN_SSM_NO_CONV_SILU").is_err();
        let reverse = std::env::var("AIRUN_SSM_CONV_REVERSE").is_ok();
        let (qd, kd, vd, nv) = (ssm.qkv_dim(), ssm.k_dim(), ssm.v_dim(), ssm.n_v_heads);
        for (si, st) in self.states.iter().enumerate() {
            device::set_device(st.device)?;
            let h = st.hybrid.as_ref().expect("hybrid scratch on a hybrid model");
            for (local, sh) in st.stash.iter().enumerate() {
                let Some(sh) = sh else { continue };
                let Block::Linear(l) = &self.model.layers[st.layers[local]].block else { continue };
                for (b, ((slot, _), &(m, _))) in rows.iter().zip(accept).enumerate() {
                    let LayerState::Delta { state, conv } = self.slot_state(*slot, si, local) else { continue };
                    let r = b * t;
                    let bytes = m * qd * std::mem::size_of::<f32>();
                    ops::copy_2d(h.qkv.as_mut_ptr(), bytes, sh.raw.offset_ptr(r * qd), bytes, bytes, 1, &st.stream)?;
                    hybrid::causal_conv1d_ex(h.qkv.as_mut_ptr(), l.conv1d.as_ptr(), conv.as_mut_ptr(), m, qd,
                        ssm.conv_kernel, silu, reverse, true, &st.stream)?;
                    hybrid::delta_rule_ex(state.as_mut_ptr(), sh.q.offset_ptr(r * kd), sh.k.offset_ptr(r * kd),
                        sh.v.offset_ptr(r * vd), sh.decay.offset_ptr(r * nv), sh.beta.offset_ptr(r * nv),
                        st.attn.as_mut_ptr(), m, ssm.n_k_heads, nv, ssm.head_k_dim(), ssm.head_v_dim(), true,
                        &st.stream)?;
                }
            }
        }
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

        // The draft picks among the first rows of the output head only (see
        // draft_vocab); the verify still scores the whole vocabulary.
        let out_w = self.model.output_tensor();
        let vocab = draft_vocab(cfg.vocab_size);
        ops::matvec(out_w.kernel_type(), out_w.ptr(), st.xb.offset_ptr((n - 1) * d),
            sp.logits.as_mut_ptr(), d, vocab, 1, out_w.row_bytes, stream)?;
        ops::argmax(sp.logits.as_ptr(), vocab, sp.picks.offset_ptr(slot), sp.scratch.as_mut_ptr(),
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
        let full = n == self.draft + 1 && !graphs_disabled();
        if full && self.states.len() > 1 && !shard_graphs_disabled() {
            return self.verify_sharded(tokens);
        }
        let graphed = full && self.states.len() == 1;
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

    /// [`Self::verify`] on a model split across GPUs: each shard's part of the
    /// pass (and the picks, on the last) replays as its own graph, with the
    /// residual handed between them as in decode.
    fn verify_sharded(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        let n = tokens.len();
        let shards = self.states.len();
        let mut vg = match self.spec.as_mut().expect("set up by spec_init").vsgraph.take() {
            Some(vg) => vg,
            None => ShardedVerifyGraph {
                graphs: (0..shards).map(|_| None).collect(),
                tokens: airun_cuda::PinnedBuffer::new(n)?,
                positions: airun_cuda::PinnedBuffer::new(n)?,
            },
        };
        // The previous replays copied these slots when they ran; the last
        // shard's was waited on below, and every earlier shard finished
        // before it could. A commit queued since does not read them.
        for (i, &t) in tokens.iter().enumerate() {
            vg.tokens.as_mut_slice()[i] = t as i32;
            vg.positions.as_mut_slice()[i] = (self.n_past + i) as i32;
        }
        let mut result = Ok(());
        for si in 0..shards {
            if si > 0 {
                result = self.handoff(si, n);
                if result.is_err() { break; }
            }
            device::set_device(self.states[si].device)?;
            if vg.graphs[si].is_none() {
                // SAFETY of the capture: everything queued lives in `self` or
                // `vg`, and `vg` goes back into `self.spec` below.
                self.defer = true;
                self.skip_logits = true;
                airun_cuda::Graph::begin_capture(&self.states[si].stream)?;
                let mut queued = self.queue_shard(si, vg.tokens.as_slice(), vg.positions.as_slice(), true, true, false);
                if si + 1 == shards {
                    queued = queued.and_then(|_| self.queue_picks(n));
                }
                let ended = airun_cuda::Graph::end_capture(&self.states[si].stream);
                self.defer = false;
                self.skip_logits = false;
                match queued.and_then(|_| Ok(ended?)) {
                    Ok(g) => vg.graphs[si] = Some(g),
                    Err(e) => { result = Err(e); break; }
                }
            }
            result = vg.graphs[si].as_ref().expect("graph was just built")
                .launch(&self.states[si].stream).map_err(Into::into);
            if result.is_err() { break; }
        }
        self.spec.as_mut().expect("set up by spec_init").vsgraph = Some(vg);
        result?;
        let st = self.states.last().expect("a runtime has at least one device");
        device::set_device(st.device)?;
        st.stream.synchronize()?;
        Ok(self.sp().picks_host.as_slice()[..n].iter().map(|&t| t as u32).collect())
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
        decoder: &mut crate::tokenizer::StreamDecoder,
        flush: bool,
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
        // Back to the turn snapshot: the head's cache is good only if the
        // prefill that took it ran the head too.
        let to_snapshot = cached > 0 && cached < self.history.len();
        if if to_snapshot { !self.snapshot_has_head() } else { cached > 0 && !self.sp().synced } {
            cached = 0;
        }
        self.rewind_to(cached);
        let cached = self.n_past.min(cached);
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
        let snap_at = self.snapshot_point(prompt, cached);
        self.prefill_report(prompt_done, prompt.len(), cached, start);
        while prompt_done < prompt.len() {
            let mut take = (prompt.len() - prompt_done).min(self.max_batch);
            if let Some(p) = snap_at.filter(|&p| p > prompt_done) {
                take = take.min(p - prompt_done);
            }
            let chunk = prompt[prompt_done..prompt_done + take].to_vec();
            let pos0 = self.n_past;
            self.step(&chunk)?;
            prompt_done += take;
            self.prefill_report(prompt_done, prompt.len(), cached, start);
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
            if snap_at == Some(prompt_done) {
                self.take_snapshot(true)?;
            }
        }
        self.spec.as_mut().expect("set up above").synced = true;
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = std::time::Instant::now();
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
        if let Some(s) = emit(self, t, &mut generated, decoder) {
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
                    if let Some(s) = emit(self, tok, &mut generated, decoder) {
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
                    ended = emit(self, next, &mut generated, decoder);
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
        if flush {
            let tail = decoder.flush();
            if !tail.is_empty() {
                on_token(last_token, &tail);
            }
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
