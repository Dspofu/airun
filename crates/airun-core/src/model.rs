//! Loading model weights onto the GPUs described by a [`Placement`].
//!
//! Quantized tensors are uploaded in their packed form and stay that way: the
//! matmul kernels decode blocks straight into registers, so nothing is ever
//! expanded to f32 in device memory. Norm weights and the small delta-net
//! parameters are the exception -- they are tiny and read every layer, so they
//! are widened once at load time.

use airun_cuda::{DeviceBuffer, Stream};
use airun_gguf::{GgufFile, TensorInfo};
use std::{cell::RefCell, sync::Arc};
use crate::expert_cache::ExpertSlots;

use crate::config::{Config, LayerKind};
use crate::error::{CoreError, Result};
use crate::plan::Placement;

/// A weight matrix on a device, kept in its on-disk layout.
pub struct GpuTensor {
    pub data: DeviceBuffer<u8>,
    pub host_experts: Option<HostExperts>,
    /// Experts split between VRAM slots (`data`) and mapped host memory.
    pub mapped: Option<MappedExperts>,
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
    /// Per-tensor NVFP4 global scale, read from the file's one-element
    /// `<base>.scale` sidecar next to `<base>.weight`. NVFP4 weights are
    /// stored as tiny per-block values that only become real weights once
    /// multiplied by it; without it the dequant is thousands of times too
    /// large and the hidden states overflow to NaN. 1.0 for every other type.
    pub global_scale: f32,
    /// Set when the weights stay in RAM (`--alloc auto|memory` with a model the
    /// GPUs cannot hold): `data` is empty and the kernels read the copy the
    /// runtime stages into VRAM right before the layer runs.
    pub ram: Option<RamWeights>,
}

/// A tensor kept in pinned RAM, and where its staged copy lands in VRAM.
pub struct RamWeights {
    pub host: HostBytes,
    /// Device address inside the layer's GPU's staging buffer.
    pub staged: u64,
}

/// Where the bytes of a [`RamWeights`] are.
pub enum HostBytes {
    /// Copied into page-locked RAM at load.
    Pinned(airun_cuda::PinnedBuffer<u8>),
    /// Left in the model file (`--disk-offload`): a slice of its mapping,
    /// paged in from the disk as it is read (see disk.rs). Unlike the pinned
    /// copy, NVFP4 scales are not sanitized here; the CPU decoder reads 0x7F
    /// as zero already, and a copy into VRAM is sanitized on the GPU.
    File { file: Arc<GgufFile>, info: TensorInfo },
}

impl HostBytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Pinned(b) => b.as_slice(),
            Self::File { file, info } => file.tensor_bytes(info).expect("the tensor was read at load"),
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn on_disk(&self) -> bool {
        matches!(self, Self::File { .. })
    }
}

/// One copy that stages a RAM layer's matrix into VRAM before the layer runs
/// on the GPU (a prompt chunk; see cpu::set_pass).
#[derive(Clone, Copy, Debug)]
pub struct RamCopy {
    /// Host address: pinned RAM, or the model file's mapping.
    pub host: usize,
    /// Device address inside the staging buffer.
    pub dev: u64,
    pub bytes: usize,
    /// NVFP4 blocks to sanitize after the copy (see load_cached): a matrix
    /// read from the file still has its 0x7F scales. 0 otherwise.
    pub nvfp4_blocks: usize,
    /// The host side is file pages, not pinned memory.
    pub on_disk: bool,
}

/// An expert stack too big for VRAM, every expert addressable by the kernels:
/// `slots` of them in the tensor's VRAM, the rest in page-locked host memory
/// mapped into the device's address space, read across PCIe as the kernels
/// touch them. A table of per-expert addresses tells the kernels where each
/// one is, so routing never waits on the host; [`Self::swap`] moves experts
/// between the two as their use shifts.
pub struct MappedExperts {
    host: airun_cuda::PinnedBuffer<u8>,
    /// Device address of `host`.
    host_dev: u64,
    /// The kernels' table: one address per expert.
    pub table: DeviceBuffer<u64>,
    /// 1 per expert in VRAM, 0 per expert in host memory: which ones the
    /// kernels must stage before computing (see hybrid::moe_stage_plan).
    pub resident: DeviceBuffer<u8>,
    resident_mirror: RefCell<airun_cuda::PinnedBuffer<u8>>,
    /// Its host copy, staged through pinned memory.
    mirror: RefCell<airun_cuda::PinnedBuffer<u64>>,
    /// Where each expert is: `Ok(slot)` in VRAM, `Err(index)` in `host`.
    /// Slots from `base` on are loans (see [`Loan`]).
    pub place: RefCell<Vec<std::result::Result<usize, usize>>>,
    /// Slots in the tensor's own VRAM.
    base: usize,
    /// VRAM lent by the KV cache's still unused rows; slot `base + i` is
    /// `loans[i]`.
    loans: RefCell<Vec<Loan>>,
    /// For an expert in a loan, the host copy it keeps (`usize::MAX` for the
    /// others): a loan holds a copy, so taking it back moves nothing.
    home: RefCell<Vec<usize>>,
}

/// One expert's worth of VRAM in the KV cache's unused rows, usable while
/// the sequence stays below row `p0` (see kv_loan.rs).
#[derive(Clone, Copy, Debug)]
pub struct Loan {
    pub addr: u64,
    pub p0: usize,
}

impl MappedExperts {
    fn addr(&self, data: &DeviceBuffer<u8>, stride: usize, p: std::result::Result<usize, usize>) -> u64 {
        match p {
            Ok(slot) if slot >= self.base => self.loans.borrow()[slot - self.base].addr,
            Ok(slot) => data.as_ptr() as u64 + (slot * stride) as u64,
            Err(i) => self.host_dev + (i * stride) as u64,
        }
    }

    fn host_ptr(&self, j: usize, stride: usize) -> *const std::ffi::c_void {
        (self.host.as_raw() as *const u8).wrapping_add(j * stride) as *const std::ffi::c_void
    }

    /// Replaces the loans; any expert in the old ones goes back to its host
    /// copy. The caller publishes the table.
    pub fn set_loans(&self, loans: Vec<Loan>) {
        self.release_below(usize::MAX);
        *self.loans.borrow_mut() = loans;
    }

    pub fn loans(&self) -> Vec<Loan> {
        self.loans.borrow().clone()
    }

    /// Sends every expert in a loan starting below row `limit` back to its
    /// host copy. Returns whether any moved; the caller then publishes.
    pub fn release_below(&self, limit: usize) -> bool {
        let loans = self.loans.borrow();
        let mut place = self.place.borrow_mut();
        let mut home = self.home.borrow_mut();
        let mut moved = false;
        for e in 0..place.len() {
            if let Ok(s) = place[e] {
                if s >= self.base && loans[s - self.base].p0 < limit {
                    place[e] = Err(home[e]);
                    home[e] = usize::MAX;
                    moved = true;
                }
            }
        }
        moved
    }

    /// Loans usable below row `limit` that hold no expert.
    pub fn free_loans(&self, limit: usize) -> Vec<usize> {
        let loans = self.loans.borrow();
        let mut used = vec![false; loans.len()];
        for p in self.place.borrow().iter() {
            if let Ok(s) = *p {
                if s >= self.base {
                    used[s - self.base] = true;
                }
            }
        }
        (0..loans.len()).filter(|&i| !used[i] && loans[i].p0 >= limit).collect()
    }

    /// The lowest row of a loan that holds an expert (`usize::MAX` if none).
    pub fn loan_floor(&self) -> usize {
        let loans = self.loans.borrow();
        self.place.borrow().iter().filter_map(|p| match *p {
            Ok(s) if s >= self.base => Some(loans[s - self.base].p0),
            _ => None,
        }).min().unwrap_or(usize::MAX)
    }

    /// Copies expert `hot` (in host memory) into the free loan `li`; its host
    /// copy stays. The caller publishes the table.
    pub fn fill(&self, stride: usize, hot: usize, li: usize, stream: &Stream) -> Result<()> {
        let mut place = self.place.borrow_mut();
        let Err(j) = place[hot] else {
            return Err(CoreError::Other("expert loaned while already in VRAM; this is a bug".into()));
        };
        let addr = self.loans.borrow()[li].addr;
        airun_cuda::copy_h2d(addr as *mut std::ffi::c_void, self.host_ptr(j, stride), stride, stream)?;
        place[hot] = Ok(self.base + li);
        self.home.borrow_mut()[hot] = j;
        Ok(())
    }

    /// Exchanges expert `hot` (in host memory) with `cold` (in VRAM), through
    /// `staging` (one stride of VRAM), on `stream`. The caller uploads the
    /// table afterwards with [`Self::publish`] and must not touch the host
    /// copy of either expert until the stream has passed this point.
    pub fn swap(&self, data: &DeviceBuffer<u8>, stride: usize, hot: usize, cold: usize,
        staging: &DeviceBuffer<u8>, stream: &Stream) -> Result<()> {
        let mut place = self.place.borrow_mut();
        let (Err(j), Ok(slot)) = (place[hot], place[cold]) else {
            return Err(CoreError::Other("expert swap between two experts on the same side; this is a bug".into()));
        };
        if slot >= self.base {
            // A loan holds a copy: `cold` goes back to its own, `hot` keeps
            // its host one.
            let addr = self.loans.borrow()[slot - self.base].addr;
            airun_cuda::copy_h2d(addr as *mut std::ffi::c_void, self.host_ptr(j, stride), stride, stream)?;
            let mut home = self.home.borrow_mut();
            place[cold] = Err(home[cold]);
            home[cold] = usize::MAX;
            place[hot] = Ok(slot);
            home[hot] = j;
            return Ok(());
        }
        let slot_ptr = data.offset_ptr(slot * stride);
        let host_ptr = (self.host.as_raw() as *mut u8).wrapping_add(j * stride);
        airun_cuda::copy_d2d(staging.as_mut_ptr(), slot_ptr, stride, stream)?;
        airun_cuda::copy_h2d(slot_ptr, host_ptr as *const std::ffi::c_void, stride, stream)?;
        airun_cuda::copy_d2h(host_ptr as *mut std::ffi::c_void, staging.as_ptr(), stride, stream)?;
        place[hot] = Ok(slot);
        place[cold] = Err(j);
        Ok(())
    }

    /// Rewrites the kernels' table from `place`.
    pub fn publish(&self, data: &DeviceBuffer<u8>, stride: usize, stream: &Stream) -> Result<()> {
        let mut mirror = self.mirror.borrow_mut();
        let mut res = self.resident_mirror.borrow_mut();
        for (e, p) in self.place.borrow().iter().enumerate() {
            mirror.as_mut_slice()[e] = self.addr(data, stride, *p);
            res.as_mut_slice()[e] = p.is_ok() as u8;
        }
        self.table.copy_from_host_at(0, mirror.as_slice(), stream)?;
        self.resident.copy_from_host_at(0, res.as_slice(), stream)?;
        Ok(())
    }
}

pub struct HostExperts {
    file: Arc<GgufFile>,
    info: TensorInfo,
    slots: RefCell<ExpertSlots>,
}

impl GpuTensor {
    /// Where the kernels read the weights: the tensor's VRAM, or for one kept
    /// in RAM its staged copy.
    pub fn ptr(&self) -> *const std::ffi::c_void {
        match &self.ram {
            Some(r) => r.staged as *const std::ffi::c_void,
            None => self.data.as_ptr(),
        }
    }

    /// Assigns a device slot to every id, returning the slots and the
    /// `(slot, expert)` pairs that must be uploaded before the kernels run. The
    /// caller synchronizes before evicting slots a previous token still reads,
    /// and calls `invalidate_experts` if the uploads then fail.
    pub fn select_experts(&self, ids: &[i32]) -> Result<(Vec<i32>, Vec<(usize, usize)>)> {
        let Some(host) = &self.host_experts else { return Ok((ids.to_vec(), Vec::new())); };
        host.slots.borrow_mut().select(ids, self.n_experts)
    }

    pub fn invalidate_experts(&self) {
        if let Some(host) = &self.host_experts { host.slots.borrow_mut().invalidate(); }
    }

    /// Shard and absolute offset in it of one expert's packed weights.
    pub fn expert_file_offset(&self, expert: usize) -> Option<(usize, u64)> {
        let host = self.host_experts.as_ref()?;
        Some((host.info.shard, host.info.file_offset + (expert * self.expert_stride) as u64))
    }

    /// The GGUF files the experts are streamed from, in shard order.
    pub fn expert_file_paths(&self) -> Option<Vec<&std::path::Path>> {
        self.host_experts.as_ref().map(|h| h.file.shard_paths())
    }

    pub fn cache_stats(&self) -> (u64, u64) {
        self.host_experts.as_ref().map_or((0, 0), |h| {
            let c = h.slots.borrow();
            (c.hits, c.misses)
        })
    }

    pub fn cache_capacity(&self) -> usize {
        if self.host_experts.is_some() || self.mapped.is_some() {
            self.data.len() / self.expert_stride
        } else {
            self.n_experts
        }
    }

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
    /// The QSA indexer, on layers that attend sparsely.
    pub indexer: Option<IndexerWeights>,
}

/// The QSA indexer of one attention layer: `heads` query heads and one key
/// head of `dim`, each RMS-normalized and rotated like the attention's own.
pub struct IndexerWeights {
    /// [n_embd] -> [heads * dim].
    pub q: GpuTensor,
    /// [n_embd] -> [dim].
    pub k: GpuTensor,
    pub q_norm: DeviceBuffer<f32>,
    pub k_norm: DeviceBuffer<f32>,
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

/// One hyper-connection mix (see [`crate::config::HcConfig`]): a grouped
/// RMSNorm over the streams, a low-rank gate, and the stream weights for the
/// combine after the block.
pub struct HcMix {
    /// [count * n_embd]: one gamma per stream, already folded to (1 + w).
    pub norm: DeviceBuffer<f32>,
    /// [count * n_embd] -> [low_rank].
    pub down: GpuTensor,
    /// [low_rank] -> [count * n_embd].
    pub up: GpuTensor,
    /// [count * n_embd] -> [count]; absent on the final (output) mix.
    pub inject: Option<GpuTensor>,
}

/// The per-layer n-gram embedding block (see [`crate::config::PleConfig`]).
pub struct PleWeights {
    /// Embedding -> one key per stream, [n_embd] -> [count * n_embd].
    pub key: GpuTensor,
    /// Embedding -> the value every stream gates, [n_embd] -> [n_embd].
    pub value: GpuTensor,
    pub norm_key: DeviceBuffer<f32>,
    pub norm_query: DeviceBuffer<f32>,
    pub norm_conv: DeviceBuffer<f32>,
    /// Depthwise taps, [count * n_embd][kernel].
    pub conv: DeviceBuffer<f32>,
}

/// The n-gram embedding table, left in the file: at 26.8 GiB for the Qwen4
/// preview it fits no GPU, and a token reads just 16 rows of it.
pub struct PleTable {
    file: Arc<GgufFile>,
    info: TensorInfo,
    row_bytes: usize,
    head_dim: usize,
    /// Reads the rows in parallel: most are random SSD reads (gather_pool).
    pool: crate::gather_pool::GatherPool,
}

/// Rows the gather threads keep in flight. More only contended on the
/// working-set lock their page faults take (128 measured slower than 32).
const PLE_READERS: usize = 32;

/// Asks Windows to read the pages under `ranges` ((address, length) pairs)
/// in one call: the memory manager queues the reads together, where faulting
/// them in from many threads contends on the process's working-set lock.
/// Elsewhere, and on failure, the gather simply faults them in.
fn prefetch_pages(ranges: &[(usize, usize)]) {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Range {
            addr: *mut std::ffi::c_void,
            len: usize,
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn PrefetchVirtualMemory(process: *mut std::ffi::c_void, n: usize, ranges: *const Range, flags: u32)
                -> i32;
        }
        let list: Vec<Range> = ranges.iter().map(|&(a, l)| Range { addr: a as *mut _, len: l }).collect();
        // SAFETY: every range lies inside the live read-only mapping of the
        // model file; the call only schedules reads of it.
        unsafe { PrefetchVirtualMemory(GetCurrentProcess(), list.len(), list.as_ptr(), 0) };
    }
    #[cfg(not(windows))]
    let _ = ranges;
}

/// A pointer the gather threads write disjoint rows through.
struct RowsOut(*mut f32);
// SAFETY: every row index is written by exactly one thread, and the buffer
// outlives the pool run that writes it.
unsafe impl Sync for RowsOut {}

impl RowsOut {
    /// Row i of `hd` floats. SAFETY: the caller owns row i and it is in bounds.
    unsafe fn row(&self, i: usize, hd: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(i * hd), hd) }
    }
}

impl PleTable {
    /// Decodes `rows` into `out`, `head_dim` floats each, in order.
    pub fn gather(&self, rows: &[u64], out: &mut [f32]) -> Result<()> {
        let bytes = self.file.tensor_bytes(&self.info)?;
        let n_rows = self.info.n_rows();
        if let Some(r) = rows.iter().find(|&&r| r >= n_rows) {
            return Err(CoreError::Other(format!(
                "PLE row {r} is past the table's {n_rows}; the ple.* metadata does not match the table")));
        }
        if out.len() < rows.len() * self.head_dim {
            return Err(CoreError::Other("PLE gather given too small an output; this is a bug".into()));
        }
        let (hd, rb, ty) = (self.head_dim, self.row_bytes, self.info.ty);
        if rows.len() > 64 {
            const PAGE: usize = 4096;
            let base = bytes.as_ptr() as usize;
            let mut ranges: Vec<(usize, usize)> = rows.iter().map(|&r| {
                let a = base + r as usize * rb;
                let start = a / PAGE * PAGE;
                (start, (a + rb).div_ceil(PAGE) * PAGE - start)
            }).collect();
            ranges.sort_unstable();
            ranges.dedup();
            prefetch_pages(&ranges);
        }
        let dst = RowsOut(out.as_mut_ptr());
        let ok = self.pool.run(rows.len(), &|i| {
            let at = rows[i] as usize * rb;
            // SAFETY: row i owns out[i * hd..(i + 1) * hd], checked in bounds above.
            let row = unsafe { dst.row(i, hd) };
            airun_gguf::dequant::dequantize(ty, &bytes[at..at + rb], hd, row).is_ok()
        });
        if !ok {
            return Err(CoreError::Other(format!(
                "cannot decode the PLE table's {ty} rows; check the model file with airun inspect")));
        }
        Ok(())
    }
}

pub struct LayerWeights {
    pub device: i32,
    /// Empty on hyper-connection models, whose mixes normalize instead.
    pub attn_norm: DeviceBuffer<f32>,
    pub block: Block,
    pub ffn_norm: DeviceBuffer<f32>,
    pub ffn: FfnWeights,
    /// The mixes before the token mixer and before the feed-forward.
    pub hc: Option<(HcMix, HcMix)>,
    pub ple: Option<PleWeights>,
    /// For a layer whose weights stay in RAM (or on disk), the copies that
    /// stage them into VRAM before it runs on the GPU.
    pub ram_copies: Vec<RamCopy>,
}

/// The multi-token-prediction head stored after the main layers
/// (`blk.{n_run}`): a draft model for speculative decoding. It reads the main
/// model's final hidden state and the next token, and predicts the token after.
pub struct MtpWeights {
    /// A full-attention block with a dense feed-forward, in the main layers' form.
    pub layer: LayerWeights,
    /// Projects `[enorm(embedding) | hnorm(hidden)]` back to the model width.
    pub eh_proj: GpuTensor,
    pub enorm: DeviceBuffer<f32>,
    pub hnorm: DeviceBuffer<f32>,
    /// The norm before the shared output head; its output is also the hidden
    /// the next draft step starts from.
    pub head_norm: DeviceBuffer<f32>,
}

/// The loaded model.
pub struct Model {
    pub config: Config,
    pub placement: Placement,
    pub layers: Vec<LayerWeights>,
    /// Shape and type of the embedding table. Its bytes are in `data`, or in
    /// `token_embd_host` when the table stays in RAM (see [`embeds_on_host`]).
    pub token_embd: GpuTensor,
    pub token_embd_host: Option<airun_cuda::PinnedBuffer<u8>>,
    pub output_norm: DeviceBuffer<f32>,
    /// The output projection. When embeddings are tied this is a second copy of
    /// the embedding table on the output device.
    pub output: GpuTensor,
    /// The final hyper-connection mix, which stands in for `output_norm`.
    pub hc_head: Option<HcMix>,
    pub ple_table: Option<PleTable>,
    /// Loaded when the file has one and speculative decoding was asked for.
    pub mtp: Option<MtpWeights>,
    pub total_bytes: u64,
    /// Routing counts that chose the resident experts, updated as the
    /// runtime routes and saved when it is done (see expert_profile).
    pub expert_profile: Option<RefCell<crate::expert_profile::ExpertProfile>>,
    /// The VRAM that layers kept in RAM are staged through, one per GPU that
    /// runs any (sized by the planner for the largest such layer).
    pub ram_staging: Vec<DeviceBuffer<u8>>,
    /// Fetches the layers read from the file ahead of use (--disk-offload).
    pub disk: Option<crate::disk::Prefetcher>,
}

impl Model {
    /// Uploads every weight named by `placement`.
    ///
    /// `progress` is called with (layers done, total) so the caller can show
    /// something during the multi-second upload of a large model.
    pub fn load(
        gguf: &Arc<GgufFile>,
        config: Config,
        placement: Placement,
        streams: &[(i32, Stream)],
        expert_slots: Option<&[usize]>,
        profile: Option<&crate::expert_profile::ExpertProfile>,
        // Where the MTP head is read from: the model file, a file of its own
        // (see loader::mtp_source), or None to leave it out.
        mtp_src: Option<&Arc<GgufFile>>,
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

        let (token_embd, token_embd_host) = if embeds_on_host(&config) {
            let info = tensor_info(gguf, "token_embd.weight", &config.arch_name)?;
            let bytes = gguf.tensor_bytes(info)?;
            let mut host = airun_cuda::PinnedBuffer::<u8>::new(bytes.len())?;
            host.as_mut_slice().copy_from_slice(bytes);
            let mut t = describe_packed(gguf, "token_embd.weight", embed_dev, &config)?;
            t.data = DeviceBuffer::<u8>::new(embed_dev, 0)?;
            (t, Some(host))
        } else {
            let t = load_packed(gguf, "token_embd.weight", embed_dev, stream_for(embed_dev)?, &config)?;
            total_bytes += t.bytes() as u64;
            (t, None)
        };
        // Each upload's file pages leave the working set as soon as they are
        // on the GPU (GgufFile::release): the load holds one layer of RAM,
        // not the whole model.
        gguf.release("token_embd.");
        done += 1;
        progress(done, total_steps);

        let hc_on = config.hc.is_some();
        let output_norm = if hc_on {
            DeviceBuffer::<f32>::new(out_dev, 0)?
        } else {
            load_f32(gguf, "output_norm.weight", out_dev, stream_for(out_dev)?)?
        };
        let hc_head = if hc_on {
            Some(load_hc_mix(gguf, "output_hc", out_dev, stream_for(out_dev)?, &config, false)?)
        } else {
            None
        };
        let ple_table = match &config.ple {
            Some(p) => {
                let info = tensor_info(gguf, "per_layer_token_embd.weight", &config.arch_name)?.clone();
                let layout = info.ty.layout().ok_or_else(|| CoreError::UnsupportedTensorType {
                    name: info.name.clone(), ty: info.ty.to_string() })?;
                if info.row_size() as usize != p.head_dim || !airun_gguf::dequant::can_decode(info.ty) {
                    return Err(CoreError::ShapeMismatch {
                        name: info.name.clone(), found: info.dims.clone(), expected: vec![p.head_dim as u64] });
                }
                let row_bytes = p.head_dim / layout.elements as usize * layout.bytes as usize;
                Some(PleTable { file: Arc::clone(gguf), info, row_bytes, head_dim: p.head_dim,
                    pool: crate::gather_pool::GatherPool::new(PLE_READERS) })
            }
            None => None,
        };
        done += 1;
        progress(done, total_steps);

        // A tied model has no `output.weight`; the embedding table doubles as
        // the projection. When the two live on different GPUs it has to be
        // uploaded twice, which the planner already accounted for.
        let output = if config.tied_embeddings {
            if out_dev == embed_dev {
                GpuTensor {
                    data: DeviceBuffer::<u8>::new(out_dev, 0)?,
                    host_experts: None,
                    mapped: None,
                    ty: token_embd.ty,
                    k: token_embd.k,
                    rows: token_embd.rows,
                    row_bytes: token_embd.row_bytes,
                    expert_stride: 0,
                    n_experts: 1,
                    device: out_dev,
                    global_scale: token_embd.global_scale,
                    ram: None,
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
        gguf.release("output");
        gguf.release("token_embd.");
        done += 1;
        progress(done, total_steps);

        // Layers in RAM multiply on the CPU (cpu.rs) unless AIRUN_RAM_ON_GPU=1
        // keeps copying them into VRAM before they run.
        if placement.shards.iter().any(|s| !s.ram_layers.is_empty())
            && !std::env::var("AIRUN_RAM_ON_GPU").is_ok_and(|v| v.trim() != "0") {
            crate::cpu::set_in_use();
        }
        if placement.shards.iter().any(|s| !s.disk_layers.is_empty()) {
            crate::disk::set_in_use();
        }
        let mut ram_staging: Vec<DeviceBuffer<u8>> = Vec::new();
        for dev in placement.devices() {
            let bytes = placement.staging_bytes(dev);
            if bytes > 0 && !ram_staging.iter().any(|b| b.device() == dev) {
                ram_staging.push(DeviceBuffer::<u8>::new(dev, bytes as usize)?);
            }
        }

        let mut layers = Vec::with_capacity(n_run);
        for l in 0..n_run {
            let dev = placement
                .device_for_layer(l)
                .ok_or_else(|| CoreError::Other(format!("layer {l} was not placed on any GPU")))?;
            let stream = stream_for(dev)?;
            let name = |suffix: &str| format!("blk.{l}.{suffix}");
            let order = profile.filter(|p| p.knows(l)).map(|p| p.order(l));
            // A layer kept in RAM: its matrices are pinned host copies, staged
            // one after another into its GPU's staging buffer.
            let on_disk = placement.layer_on_disk(l);
            let staging = placement.layer_in_ram(l).then(|| {
                ram_staging.iter().find(|b| b.device() == dev).map(|b| (b.as_ptr() as u64, b.len()))
            }).flatten();
            let cursor = std::cell::Cell::new(0usize);
            let copies = RefCell::new(Vec::new());
            let packed = |n: &str| -> Result<GpuTensor> {
                match staging {
                    Some((base, cap)) => {
                        let t = if on_disk {
                            load_disk(gguf, n, dev, &config, base + cursor.get() as u64)?
                        } else {
                            load_ram(gguf, n, dev, &config, base + cursor.get() as u64)?
                        };
                        let len = t.ram.as_ref().map_or(0, |r| r.host.len());
                        if cursor.get() + len > cap {
                            return Err(CoreError::Other(format!(
                                "layer {l} does not fit the {} staging buffer planned for it; this is a bug",
                                airun_cuda::human_bytes(cap as u64))));
                        }
                        if let Some(r) = &t.ram {
                            let sanitize = r.host.on_disk() && t.ty == airun_gguf::GgmlType::NVFP4
                                && airun_cuda::ops::fp4_supported();
                            copies.borrow_mut().push(RamCopy {
                                host: r.host.as_slice().as_ptr() as usize,
                                dev: r.staged,
                                bytes: len,
                                nvfp4_blocks: if sanitize { len / 36 } else { 0 },
                                on_disk: r.host.on_disk(),
                            });
                        }
                        cursor.set(cursor.get() + len.div_ceil(256) * 256);
                        Ok(t)
                    }
                    None => load_packed(gguf, n, dev, stream, &config),
                }
            };

            let block = match config.layer_kind(l) {
                LayerKind::Attention => Block::Attention(AttentionWeights {
                    q: packed(&name("attn_q.weight"))?,
                    k: packed(&name("attn_k.weight"))?,
                    v: packed(&name("attn_v.weight"))?,
                    output: packed(&name("attn_output.weight"))?,
                    q_norm: load_f32_optional(gguf, &name("attn_q_norm.weight"), dev, stream)?,
                    k_norm: load_f32_optional(gguf, &name("attn_k_norm.weight"), dev, stream)?,
                    indexer: match &config.indexer {
                        Some(ix) if ix.ratio(l) > 0 => Some(IndexerWeights {
                            q: packed(&name("indexer.q_proj.weight"))?,
                            k: packed(&name("indexer.k_proj.weight"))?,
                            q_norm: load_f32(gguf, &name("indexer.q_norm.weight"), dev, stream)?,
                            k_norm: load_f32(gguf, &name("indexer.k_norm.weight"), dev, stream)?,
                        }),
                        _ => None,
                    },
                }),
                LayerKind::LinearAttention => Block::Linear(LinearAttentionWeights {
                    qkv: packed(&name("attn_qkv.weight"))?,
                    gate: packed(&name("attn_gate.weight"))?,
                    conv1d: load_f32(gguf, &name("ssm_conv1d.weight"), dev, stream)?,
                    a_log: load_delta_decay_coeff(gguf, &name("ssm_a"), dev, stream)?,
                    dt_bias: load_f32(gguf, &name("ssm_dt.bias"), dev, stream)?,
                    alpha: packed(&name("ssm_alpha.weight"))?,
                    beta: packed(&name("ssm_beta.weight"))?,
                    norm: load_f32(gguf, &name("ssm_norm.weight"), dev, stream)?,
                    out: packed(&name("ssm_out.weight"))?,
                }),
            };

            let ffn = if config.moe.is_some() {
                let shared = if gguf.tensor(&name("ffn_gate_shexp.weight")).is_some() {
                    Some(SharedExpert {
                        router: packed(&name("ffn_gate_inp_shexp.weight"))?,
                        gate: packed(&name("ffn_gate_shexp.weight"))?,
                        up: packed(&name("ffn_up_shexp.weight"))?,
                        down: packed(&name("ffn_down_shexp.weight"))?,
                    })
                } else {
                    None
                };
                FfnWeights::Moe {
                    router: packed(&name("ffn_gate_inp.weight"))?,
                    gate_exps: load_cached(gguf, &name("ffn_gate_exps.weight"), dev, stream, &config, expert_slots.map(|s| s[l]),
                        order.as_deref())?,
                    up_exps: load_cached(gguf, &name("ffn_up_exps.weight"), dev, stream, &config, expert_slots.map(|s| s[l]),
                        order.as_deref())?,
                    down_exps: load_cached(gguf, &name("ffn_down_exps.weight"), dev, stream, &config, expert_slots.map(|s| s[l]),
                        order.as_deref())?,
                    shared,
                }
            } else {
                FfnWeights::Dense {
                    gate: packed(&name("ffn_gate.weight"))?,
                    up: packed(&name("ffn_up.weight"))?,
                    down: packed(&name("ffn_down.weight"))?,
                }
            };

            let empty = || DeviceBuffer::<f32>::new(dev, 0);
            let w = LayerWeights {
                device: dev,
                attn_norm: if hc_on { empty()? } else { load_f32(gguf, &name("attn_norm.weight"), dev, stream)? },
                block,
                ffn_norm: if hc_on {
                    empty()?
                } else {
                    load_f32(gguf, &name(&format!("{}.weight", config.ffn_norm_name)), dev, stream)?
                },
                ffn,
                hc: if hc_on {
                    Some((load_hc_mix(gguf, &name("hc_attn"), dev, stream, &config, true)?,
                          load_hc_mix(gguf, &name("hc_ffn"), dev, stream, &config, true)?))
                } else {
                    None
                },
                ple: match &config.ple {
                    Some(p) if p.layer == l => Some(PleWeights {
                        key: packed(&name("ple_key.weight"))?,
                        value: packed(&name("ple_value.weight"))?,
                        norm_key: load_f32(gguf, &name("ple_norm_key.weight"), dev, stream)?,
                        norm_query: load_f32(gguf, &name("ple_norm_query.weight"), dev, stream)?,
                        norm_conv: load_f32(gguf, &name("ple_norm_conv.weight"), dev, stream)?,
                        conv: load_f32(gguf, &name("ple_conv1d.weight"), dev, stream)?,
                    }),
                    _ => None,
                },
                ram_copies: copies.into_inner(),
            };
            total_bytes += layer_bytes(&w);
            layers.push(w);
            gguf.release(&format!("blk.{l}."));

            done += 1;
            progress(done, total_steps);
        }

        // Verify the shapes actually match what the metadata promised. A file
        // whose header and tensors disagree would otherwise produce garbage.
        validate_shapes(&config, &layers, &token_embd)?;

        let mtp = if let Some(src) = mtp_src {
            let m = load_mtp_weights(src, &config, out_dev, stream_for(out_dev)?)?;
            total_bytes += layer_bytes(&m.layer) + m.eh_proj.bytes() as u64;
            Some(m)
        } else {
            None
        };

        for (_, s) in streams {
            s.synchronize()?;
        }

        // The layers read from the file, in run order, with their ranges.
        let disk_layers: Vec<(usize, Vec<(usize, usize)>)> = layers.iter().enumerate()
            .filter(|(l, _)| placement.layer_on_disk(*l))
            .map(|(l, w)| (l, w.ram_copies.iter().filter(|c| c.on_disk).map(|c| (c.host, c.bytes)).collect()))
            .collect();
        let disk = (!disk_layers.is_empty()).then(|| crate::disk::Prefetcher::new(gguf, disk_layers));

        Ok(Self {
            config,
            placement,
            layers,
            token_embd,
            token_embd_host,
            output_norm,
            output,
            hc_head,
            ple_table,
            mtp,
            total_bytes,
            expert_profile: None,
            ram_staging,
            disk,
        })
    }

    /// Where the embedding kernel on `device` reads the table.
    pub fn embedding_ptr(&self, device: i32) -> Result<*const std::ffi::c_void> {
        match &self.token_embd_host {
            Some(host) => Ok(host.device_ptr(device)?),
            None => Ok(self.token_embd.data.as_ptr()),
        }
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

/// `<prefix>_norm/_down/_up/_inject.weight`.
fn load_hc_mix(gguf: &Arc<GgufFile>, prefix: &str, dev: i32, stream: &Stream, config: &Config,
    inject: bool) -> Result<HcMix> {
    Ok(HcMix {
        norm: load_f32(gguf, &format!("{prefix}_norm.weight"), dev, stream)?,
        down: load_packed(gguf, &format!("{prefix}_down.weight"), dev, stream, config)?,
        up: load_packed(gguf, &format!("{prefix}_up.weight"), dev, stream, config)?,
        inject: if inject {
            Some(load_packed(gguf, &format!("{prefix}_inject.weight"), dev, stream, config)?)
        } else {
            None
        },
    })
}

fn hc_bytes(m: &HcMix) -> usize {
    m.down.bytes() + m.up.bytes() + m.inject.as_ref().map_or(0, |t| t.bytes())
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
    let hc = w.hc.as_ref().map_or(0, |(a, f)| hc_bytes(a) + hc_bytes(f));
    let ple = w.ple.as_ref().map_or(0, |p| p.key.bytes() + p.value.bytes());
    (block + ffn + hc + ple) as u64
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
/// Whether this file's MTP head is one the runtime can draft with: a single
/// head whose block is attention with a dense feed-forward (Qwen3.5/3.8
/// dense). The MoE heads of the A3B models are not wired up yet.
/// A file runs MTP speculative decoding when it declares exactly one draft
/// head (`nextn_predict_layers == 1`), the head is a layer the runtime can
/// execute (a hybrid block: dense and MoE hybrids both qualify, the MoE head
/// running through the same expert path as the other layers), and the head's
/// tensors are actually present in the file. That last check matters: some
/// converters keep the `nextn_predict_layers` metadata while stripping the
/// head weights, and loading would then fail on a missing tensor. The
/// `nextn.eh_proj` input projection is the head's signature tensor.
///
/// The head is block `n_run_layers()`: `block_count` counts it, so it is the
/// last block, not one past it.
pub fn mtp_supported(gguf: &GgufFile, config: &Config) -> bool {
    config.nextn_predict_layers == 1 && config.is_hybrid() && mtp_head_present(gguf, config)
}

/// Bytes of the MTP head's own tensors (block `n_run_layers()`) in `src`; a
/// head file's copies of the embedding and output are not loaded.
pub fn mtp_head_bytes(src: &GgufFile, config: &Config) -> u64 {
    let prefix = format!("blk.{}.", config.n_run_layers());
    src.tensors().iter().filter(|t| t.name.starts_with(&prefix)).map(|t| t.nbytes).sum()
}

/// VRAM speculative decoding allocates once it starts (runtime::spec), on
/// top of the head and its cache: the head's input rows, the verify's
/// logits, the delta-net stash for every slot's verified rows, the batched
/// verify's logits, and room for the graphs. Left out of the plan, the fast
/// GPU (filled to its budget when it holds the output head) ran ~125 MiB
/// over on the 27B with --parallel 4, and Windows began paging VRAM: steps
/// went from 110 to 400 ms.
pub fn spec_scratch_bytes(config: &Config, batch: usize, draft: usize) -> u64 {
    let slots = config.slots.max(1) as u64;
    let rows = draft as u64 + 1;
    let (d, vocab) = (config.n_embd as u64, config.vocab_size as u64);
    let f = std::mem::size_of::<f32>() as u64;
    let head_rows = 4 * batch as u64 * d * f;
    let logits = rows * vocab * f + slots * (vocab + d) * f;
    let batched = if slots > 1 { slots * rows * vocab * f } else { 0 };
    // The delta-net stash and the verify's attention scratch sit on every
    // GPU: spec_stash_bytes per layer, plan::scratch_bytes per device.
    head_rows + logits + batched + (32 << 20)
}

/// The delta-net stash one linear-attention layer keeps for a speculative
/// verify (every slot's verified rows), on that layer's GPU.
pub fn spec_stash_bytes(config: &Config, draft: usize) -> u64 {
    config.ssm.as_ref().map_or(0, |ssm| {
        let per_row = (ssm.qkv_dim() + 2 * ssm.k_dim() + ssm.v_dim() + 2 * ssm.n_v_heads) as u64 * 4;
        config.slots.max(1) as u64 * (draft as u64 + 1) * per_row
    })
}

/// Whether the file carries the MTP head's weights (see [`mtp_supported`]).
pub fn mtp_head_present(gguf: &GgufFile, config: &Config) -> bool {
    gguf.tensor(&format!("blk.{}.nextn.eh_proj.weight", config.n_run_layers())).is_some()
}

fn load_mtp_weights(gguf: &Arc<GgufFile>, config: &Config, dev: i32, stream: &Stream) -> Result<MtpWeights> {
    let l = config.n_run_layers();
    let name = |suffix: &str| format!("blk.{l}.{suffix}");
    let layer = LayerWeights {
        device: dev,
        attn_norm: load_f32(gguf, &name("attn_norm.weight"), dev, stream)?,
        block: Block::Attention(AttentionWeights {
            q: load_packed(gguf, &name("attn_q.weight"), dev, stream, config)?,
            k: load_packed(gguf, &name("attn_k.weight"), dev, stream, config)?,
            v: load_packed(gguf, &name("attn_v.weight"), dev, stream, config)?,
            output: load_packed(gguf, &name("attn_output.weight"), dev, stream, config)?,
            q_norm: load_f32_optional(gguf, &name("attn_q_norm.weight"), dev, stream)?,
            k_norm: load_f32_optional(gguf, &name("attn_k_norm.weight"), dev, stream)?,
            indexer: None,
        }),
        ffn_norm: load_f32(gguf, &name(&format!("{}.weight", config.ffn_norm_name)), dev, stream)?,
        // A MoE model's head is a MoE block too; its experts all stay on the
        // GPU (the planner counted the whole head with the output device).
        ffn: if config.moe.is_some() {
            let shared = if gguf.tensor(&name("ffn_gate_shexp.weight")).is_some() {
                Some(SharedExpert {
                    router: load_packed(gguf, &name("ffn_gate_inp_shexp.weight"), dev, stream, config)?,
                    gate: load_packed(gguf, &name("ffn_gate_shexp.weight"), dev, stream, config)?,
                    up: load_packed(gguf, &name("ffn_up_shexp.weight"), dev, stream, config)?,
                    down: load_packed(gguf, &name("ffn_down_shexp.weight"), dev, stream, config)?,
                })
            } else {
                None
            };
            FfnWeights::Moe {
                router: load_packed(gguf, &name("ffn_gate_inp.weight"), dev, stream, config)?,
                gate_exps: load_packed(gguf, &name("ffn_gate_exps.weight"), dev, stream, config)?,
                up_exps: load_packed(gguf, &name("ffn_up_exps.weight"), dev, stream, config)?,
                down_exps: load_packed(gguf, &name("ffn_down_exps.weight"), dev, stream, config)?,
                shared,
            }
        } else {
            FfnWeights::Dense {
                gate: load_packed(gguf, &name("ffn_gate.weight"), dev, stream, config)?,
                up: load_packed(gguf, &name("ffn_up.weight"), dev, stream, config)?,
                down: load_packed(gguf, &name("ffn_down.weight"), dev, stream, config)?,
            }
        },
        hc: None,
        ple: None,
        ram_copies: Vec::new(),
    };
    let head_norm = if gguf.tensor(&name("nextn.shared_head_norm.weight")).is_some() {
        load_f32(gguf, &name("nextn.shared_head_norm.weight"), dev, stream)?
    } else {
        load_f32(gguf, "output_norm.weight", dev, stream)?
    };
    Ok(MtpWeights {
        layer,
        eh_proj: load_packed(gguf, &name("nextn.eh_proj.weight"), dev, stream, config)?,
        enorm: load_f32(gguf, &name("nextn.enorm.weight"), dev, stream)?,
        hnorm: load_f32(gguf, &name("nextn.hnorm.weight"), dev, stream)?,
        head_norm,
    })
}

/// Whether the token embedding table stays in pinned RAM instead of VRAM, as
/// llama.cpp keeps it. A decode step reads one row (3.5 KiB for a 5120-wide
/// Q5_K table) over PCIe, while the table itself is ~0.8 GiB of VRAM on a
/// 250k vocabulary -- enough to decide whether a 27B fits one 16 GB card. A
/// tied table is also the output projection, which reads all of it every
/// token, so that one stays on the GPU. AIRUN_EMBED_ON_GPU=1 keeps it there.
pub fn embeds_on_host(config: &Config) -> bool {
    !config.tied_embeddings && std::env::var("AIRUN_EMBED_ON_GPU").map_or(true, |v| v.trim() == "0")
}

/// A tensor's shape and type, with an empty device buffer.
fn describe_packed(gguf: &Arc<GgufFile>, name: &str, device: i32, config: &Config) -> Result<GpuTensor> {
    let info = tensor_info(gguf, name, &config.arch_name)?;
    let k = info.row_size() as usize;
    let layout = info.ty.layout().ok_or_else(|| CoreError::UnsupportedTensorType {
        name: name.to_string(),
        ty: info.ty.to_string(),
    })?;
    Ok(GpuTensor {
        data: DeviceBuffer::<u8>::new(device, 0)?,
        host_experts: None,
        mapped: None,
        ty: info.ty,
        k,
        rows: info.n_rows() as usize,
        row_bytes: (k / layout.elements as usize) * layout.bytes as usize,
        expert_stride: 0,
        n_experts: 1,
        device,
        global_scale: global_scale_of(gguf, name),
        ram: None,
    })
}

/// The per-tensor NVFP4 global scale, stored in these files as a one-element
/// `<base>.scale` F32 sidecar next to `<base>.weight` (the per-block scales
/// alone are ~1e-4 of the true weight magnitude). 1.0 when the file carries
/// no sidecar, which is every non-NVFP4 model.
fn global_scale_of(gguf: &GgufFile, name: &str) -> f32 {
    let Some(base) = name.strip_suffix(".weight") else { return 1.0 };
    let Some(info) = gguf.tensor(&format!("{base}.scale")) else { return 1.0 };
    if info.n_elements() != 1 {
        return 1.0;
    }
    let Ok(bytes) = gguf.tensor_bytes(info) else { return 1.0 };
    let mut v = vec![0f32; 1];
    airun_gguf::dequant::dequantize(info.ty, bytes, 1, &mut v).map(|_| v[0]).unwrap_or(1.0)
}

fn load_packed(
    gguf: &Arc<GgufFile>,
    name: &str,
    device: i32,
    stream: &Stream,
    config: &Config,
) -> Result<GpuTensor> {
    load_cached(gguf, name, device, stream, config, None, None)
}

fn load_cached(
    gguf: &Arc<GgufFile>, name: &str, device: i32, stream: &Stream,
    config: &Config, slots: Option<usize>, order: Option<&[usize]>,
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

    let mapped_mode = experts_mapped();
    let mut mapped = None;
    let (data, host_experts) = if let Some(slots) = slots.filter(|&s| s < n_experts && mapped_mode) {
        let bytes = gguf.tensor_bytes(info)?;
        // The profile's most routed experts take the VRAM slots; without one,
        // the first `slots` by index do.
        let order: Vec<usize> = match order {
            Some(o) if o.len() == n_experts => o.to_vec(),
            _ => (0..n_experts).collect(),
        };
        let expert = |e: usize| &bytes[e * expert_stride..(e + 1) * expert_stride];
        let data = DeviceBuffer::<u8>::new(device, slots * expert_stride)?;
        for (slot, &e) in order[..slots].iter().enumerate() {
            data.copy_from_host_at(slot * expert_stride, expert(e), stream)?;
        }
        let mut host = airun_cuda::PinnedBuffer::<u8>::new((n_experts - slots) * expert_stride).map_err(|e|
            CoreError::Other(format!("cannot pin {} of host memory for the experts that do not fit in VRAM ({e}); \
                free RAM, or set AIRUN_EXPERTS=stream to read them from the file instead",
                airun_cuda::human_bytes(((n_experts - slots) * expert_stride) as u64))))?;
        for (i, &e) in order[slots..].iter().enumerate() {
            host.as_mut_slice()[i * expert_stride..(i + 1) * expert_stride].copy_from_slice(expert(e));
        }
        let mut place = vec![Ok(0); n_experts];
        for (i, &e) in order.iter().enumerate() {
            place[e] = if i < slots { Ok(i) } else { Err(i - slots) };
        }
        let host_dev = host.device_ptr(device)? as u64;
        let m = MappedExperts {
            host,
            host_dev,
            table: DeviceBuffer::new(device, n_experts)?,
            mirror: RefCell::new(airun_cuda::PinnedBuffer::new(n_experts)?),
            resident: DeviceBuffer::new(device, n_experts)?,
            resident_mirror: RefCell::new(airun_cuda::PinnedBuffer::new(n_experts)?),
            place: RefCell::new(place),
            base: slots,
            loans: RefCell::new(Vec::new()),
            home: RefCell::new(vec![usize::MAX; n_experts]),
        };
        m.publish(&data, expert_stride, stream)?;
        // The pinned mirror is reused by the next publish; let this one land.
        stream.synchronize()?;
        mapped = Some(m);
        (data, None)
    } else if let Some(slots) = slots.filter(|&s| s < n_experts) {
        (DeviceBuffer::new(device, slots * expert_stride)?, Some(HostExperts {
            file: Arc::clone(gguf), info: info.clone(), slots: RefCell::new(ExpertSlots::new(slots)),
        }))
    } else {
        let data = DeviceBuffer::<u8>::from_slice(device, gguf.tensor_bytes(info)?, stream)?;
        // ggml reads a 0x7F NVFP4 run scale as zero and the FP4 tensor cores as
        // NaN; zeroing it keeps every path on ggml's reading.
        if info.ty == airun_gguf::GgmlType::NVFP4 && airun_cuda::ops::fp4_supported() {
            airun_cuda::ops::nvfp4_sanitize(data.as_mut_ptr(), data.len() / 36, stream)?;
        }
        (data, None)
    };

    let global_scale = global_scale_of(gguf, name);
    if n_experts > 1 && global_scale != 1.0 {
        return Err(CoreError::Other(format!(
            "{name} is an NVFP4 expert stack with a per-tensor global scale (a `.scale` sidecar), which the \
             MoE kernels do not apply yet; convert the file without the sidecar (airun's own NVFP4 encoder \
             writes self-contained blocks) or use another quantization of the model")));
    }

    Ok(GpuTensor {
        data,
        host_experts,
        mapped,
        ty: info.ty,
        k,
        rows,
        row_bytes,
        expert_stride,
        n_experts,
        device,
        global_scale,
        ram: None,
    })
}

/// A matrix kept in pinned RAM, to be staged to `staged` before use (see
/// [`RamWeights`]).
fn load_ram(gguf: &Arc<GgufFile>, name: &str, device: i32, config: &Config, staged: u64) -> Result<GpuTensor> {
    let info = tensor_info(gguf, name, &config.arch_name)?;
    if !airun_cuda::ops::KernelType(info.ty.id() as i32).is_supported() {
        return Err(CoreError::UnsupportedTensorType { name: name.to_string(), ty: info.ty.to_string() });
    }
    let bytes = gguf.tensor_bytes(info)?;
    let mut host = airun_cuda::PinnedBuffer::<u8>::new(bytes.len()).map_err(|e| CoreError::Other(format!(
        "cannot pin {} of RAM for {name} ({e}); free RAM, lower --ctx-size so fewer layers stay in RAM, \
         pass --disk-offload to read what the RAM cannot hold from the SSD, or use --alloc gpu",
        airun_cuda::human_bytes(bytes.len() as u64))))?;
    host.as_mut_slice().copy_from_slice(bytes);
    // As load_cached does on the GPU: ggml reads a 0x7F NVFP4 run scale as
    // zero, the FP4 tensor cores as NaN. Each 36-byte block starts with its
    // four scales.
    if info.ty == airun_gguf::GgmlType::NVFP4 && airun_cuda::ops::fp4_supported() {
        for block in host.as_mut_slice().chunks_exact_mut(36) {
            for d in &mut block[..4] {
                if *d == 0x7F {
                    *d = 0;
                }
            }
        }
    }
    let mut t = describe_packed(gguf, name, device, config)?;
    t.ram = Some(RamWeights { host: HostBytes::Pinned(host), staged });
    Ok(t)
}

/// A matrix left in the model file (`--disk-offload`): nothing is read now;
/// the CPU kernels read the mapping, and a pass on the GPU stages it to
/// `staged` (see [`HostBytes::File`]).
fn load_disk(gguf: &Arc<GgufFile>, name: &str, device: i32, config: &Config, staged: u64) -> Result<GpuTensor> {
    let info = tensor_info(gguf, name, &config.arch_name)?;
    if !airun_cuda::ops::KernelType(info.ty.id() as i32).is_supported() {
        return Err(CoreError::UnsupportedTensorType { name: name.to_string(), ty: info.ty.to_string() });
    }
    // Bounds-checks the slice now, so HostBytes::as_slice cannot fail later.
    gguf.tensor_bytes(info)?;
    let info = info.clone();
    let mut t = describe_packed(gguf, name, device, config)?;
    t.ram = Some(RamWeights { host: HostBytes::File { file: Arc::clone(gguf), info }, staged });
    Ok(t)
}

/// Set by the loader when --disk-offload finds no RAM for mapped experts.
static STREAM_EXPERTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Experts that do not fit in VRAM are read from the file from now on.
pub fn stream_experts() {
    STREAM_EXPERTS.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether experts that do not fit in VRAM live in mapped host memory (the
/// default) or are read from the GGUF on each miss (AIRUN_EXPERTS=stream, or
/// --disk-offload with too little RAM).
pub fn experts_mapped() -> bool {
    !STREAM_EXPERTS.load(std::sync::atomic::Ordering::Relaxed)
        && std::env::var("AIRUN_EXPERTS").map_or(true, |v| v.trim() != "stream")
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

    let check = |name: String, found: (usize, usize), expected: (usize, usize)| -> Result<()> {
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
