//! The KV cache's unused rows lend their VRAM to MoE experts kept in RAM.
//!
//! A long context reserves its whole KV cache up front: the 35B-A3B at 246k
//! tokens in q8_0 holds ~2.7 GB that a short conversation never touches,
//! while a fifth of its experts cross PCIe every token. The rows past the
//! sequence hold copies of host-side experts instead ([`crate::model::Loan`]);
//! each loan sits at some row `p0` of a head's span, and stays good while the
//! sequence writes below it. Before a pass would come within `margin` rows of
//! the lowest loan in use, every loan below twice that is handed back: the
//! expert points at its host copy again, which moves nothing. Rebalancing
//! refills loans above the sequence with the hottest host-side experts.
//!
//! Off with several slots (each one's cache would need its own loans), KV in
//! RAM, sparse-attention indexes, or AIRUN_NO_KV_LOAN=1.

use super::*;
use crate::model::Loan;

pub(super) struct LoanState {
    /// Rows kept clear between the furthest a pass may write and a loan.
    margin: usize,
    /// The lowest row of a loan holding an expert.
    floor: usize,
}

fn loans_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_KV_LOAN").is_ok_and(|v| v.trim() != "0"))
}

/// The three expert tensors of a MoE layer whose experts are mapped.
fn mapped_triple(ffn: &FfnWeights) -> Option<[&GpuTensor; 3]> {
    match ffn {
        FfnWeights::Moe { gate_exps, up_exps, down_exps, .. }
            if gate_exps.mapped.is_some() && up_exps.mapped.is_some() && down_exps.mapped.is_some() =>
            Some([gate_exps, up_exps, down_exps]),
        _ => None,
    }
}

impl Runtime {
    /// Carves the unused KV rows into loans and fills them with the host-side
    /// experts the placement ranked first. Called once the caches exist.
    pub(super) fn setup_kv_loans(&mut self) -> Result<()> {
        let cfg = &self.model.config;
        if loans_disabled() || cfg.indexer.is_some() || self.slots.len() > 1 {
            return Ok(());
        }
        let row = cfg.kv_type.row_bytes(cfg.head_dim);
        let margin = self.max_batch + 1024;
        let mut lent = 0usize;
        let mut experts = 0usize;
        for st in &self.states {
            let moe: Vec<[&GpuTensor; 3]> = st.layers.iter()
                .filter_map(|&l| mapped_triple(&self.model.layers[l].ffn)).collect();
            if moe.is_empty() {
                continue;
            }
            // One size for every loan on this device: the widest expert.
            let size = moe.iter().flatten().map(|t| t.expert_stride).max().unwrap_or(0).div_ceil(256) * 256;
            if size == 0 {
                continue;
            }
            let mut chunks: Vec<Loan> = Vec::new();
            for (local, &l) in st.layers.iter().enumerate() {
                if self.model.placement.kv_in_ram(l) {
                    continue;
                }
                let LayerState::Kv { k, v, .. } = &st.states[local] else { continue };
                for buf in [k, v] {
                    let base = buf.as_ptr() as u64;
                    for h in 0..cfg.n_head_kv {
                        let start = h * self.ctx * row;
                        let mut end = start + self.ctx * row;
                        // From the far end of the span down: the last rows
                        // a conversation reaches lend the longest.
                        while end >= start + size {
                            let at = (end - size) & !255;
                            if at < start {
                                break;
                            }
                            chunks.push(Loan { addr: base + at as u64, p0: (at - start) / row });
                            end = at;
                        }
                    }
                }
            }
            if chunks.is_empty() {
                continue;
            }
            chunks.sort_by(|a, b| b.p0.cmp(&a.p0));
            // Each layer takes loans in turn, three at a time (gate, up and
            // down share one row limit), so all of them lose loans evenly
            // as the context fills.
            let room: Vec<usize> = moe.iter().map(|t| {
                let m = t[0].mapped.as_ref().expect("mapped_triple checked it");
                m.place.borrow().iter().filter(|p| p.is_err()).count()
            }).collect();
            let mut per: Vec<Vec<[Loan; 3]>> = vec![Vec::new(); moe.len()];
            let mut it = chunks.chunks_exact(3);
            'deal: loop {
                let mut dealt = false;
                for (i, p) in per.iter_mut().enumerate() {
                    if p.len() >= room[i] {
                        continue;
                    }
                    let Some(c) = it.next() else { break 'deal };
                    let p0 = c.iter().map(|x| x.p0).min().expect("three chunks");
                    p.push([c[0], c[1], c[2]].map(|x| Loan { addr: x.addr, p0 }));
                    dealt = true;
                }
                if !dealt {
                    break;
                }
            }
            device::set_device(st.device)?;
            for (t, loans) in moe.iter().zip(&per) {
                // Host copies are in the placement's order, most routed
                // first; the gate's order stands for all three.
                let mut outside: Vec<(usize, usize)> = t[0].mapped.as_ref().expect("checked").place.borrow()
                    .iter().enumerate().filter_map(|(e, p)| p.err().map(|h| (h, e))).collect();
                outside.sort();
                let usable: Vec<usize> = (0..loans.len()).filter(|&i| loans[i][0].p0 >= 2 * margin).collect();
                for (j, tensor) in t.iter().enumerate() {
                    let m = tensor.mapped.as_ref().expect("mapped_triple checked it");
                    m.set_loans(loans.iter().map(|l| l[j]).collect());
                    for (&li, &(_, e)) in usable.iter().zip(&outside) {
                        m.fill(tensor.expert_stride, e, li, &st.stream)?;
                    }
                    m.publish(&tensor.data, tensor.expert_stride, &st.stream)?;
                    // The table's pinned mirror is reused by the next publish.
                    st.stream.synchronize()?;
                }
                lent += usable.len().min(outside.len()) * 3 * size;
                experts += usable.len().min(outside.len());
            }
        }
        if experts == 0 {
            return Ok(());
        }
        eprintln!("experts: {experts} more in VRAM ({}) lent by the KV cache's unused rows, \
                   given back as the context fills (AIRUN_NO_KV_LOAN=1 turns it off)",
                  airun_cuda::human_bytes(lent as u64));
        self.loan = Some(LoanState { margin, floor: 0 });
        self.loan_floor_refresh();
        Ok(())
    }

    fn loan_floor_refresh(&mut self) {
        let floor = self.model.layers.iter().filter_map(|l| mapped_triple(&l.ffn))
            .map(|t| t[0].mapped.as_ref().expect("mapped_triple checked it").loan_floor())
            .min().unwrap_or(usize::MAX);
        if let Some(l) = self.loan.as_mut() {
            l.floor = floor;
        }
    }

    /// Hands back every loan a pass writing rows below `upto` could reach.
    /// Called before anything writes KV rows.
    pub(crate) fn kv_loan_guard(&mut self, upto: usize) -> Result<()> {
        let Some(l) = &self.loan else { return Ok(()) };
        if upto + l.margin <= l.floor {
            return Ok(());
        }
        let limit = upto + 2 * l.margin;
        for st in &self.states {
            let mut moved = false;
            for &layer in &st.layers {
                let Some(t) = mapped_triple(&self.model.layers[layer].ffn) else { continue };
                let mut any = false;
                for tensor in t {
                    any |= tensor.mapped.as_ref().expect("checked").release_below(limit);
                }
                if any {
                    device::set_device(st.device)?;
                    for tensor in t {
                        tensor.mapped.as_ref().expect("checked").publish(&tensor.data, tensor.expert_stride,
                            &st.stream)?;
                    }
                    moved = true;
                }
            }
            if moved {
                st.stream.synchronize()?;
            }
        }
        self.loan_floor_refresh();
        Ok(())
    }

    /// The row from which loans may take experts: past everything the caches
    /// hold or a prefix match could reuse, with room for the next passes.
    pub(super) fn loan_limit(&self) -> Option<usize> {
        self.loan.as_ref().map(|l| self.n_past.max(self.history.len()) + 2 * l.margin)
    }

    /// Fills free loans of one layer with its hottest host-side experts, at
    /// most `max` of them; `scores` ranks the experts. Returns how many.
    pub(super) fn loan_fill(&self, st: &DeviceState, layer: usize, scores: &[f32], max: usize, min_score: f32)
        -> Result<usize> {
        let Some(limit) = self.loan_limit() else { return Ok(0) };
        let Some(t) = mapped_triple(&self.model.layers[layer].ffn) else { return Ok(0) };
        let gm = t[0].mapped.as_ref().expect("checked");
        let free = gm.free_loans(limit);
        if free.is_empty() {
            return Ok(0);
        }
        let mut outside: Vec<usize> = gm.place.borrow().iter().enumerate()
            .filter_map(|(e, p)| p.is_err().then_some(e)).filter(|&e| scores[e] > min_score).collect();
        outside.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let mut n = 0;
        for (&li, &e) in free.iter().zip(&outside).take(max) {
            for tensor in t {
                tensor.mapped.as_ref().expect("checked").fill(tensor.expert_stride, e, li, &st.stream)?;
            }
            n += 1;
        }
        Ok(n)
    }

    /// After loans were filled: the floor may have dropped.
    pub(super) fn loan_filled(&mut self) {
        if self.loan.is_some() {
            self.loan_floor_refresh();
        }
    }

    /// Drops every loan for good (several slots would each need their own).
    pub(super) fn drop_kv_loans(&mut self) -> Result<()> {
        if self.loan.take().is_none() {
            return Ok(());
        }
        for st in &self.states {
            device::set_device(st.device)?;
            for &layer in &st.layers {
                let Some(t) = mapped_triple(&self.model.layers[layer].ffn) else { continue };
                for tensor in t {
                    let m = tensor.mapped.as_ref().expect("checked");
                    m.set_loans(Vec::new());
                    m.publish(&tensor.data, tensor.expert_stride, &st.stream)?;
                }
                st.stream.synchronize()?;
            }
        }
        Ok(())
    }
}
