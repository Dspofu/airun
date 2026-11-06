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

use airun_cuda::qsa;

#[path = "spec.rs"]
mod spec;
pub use spec::RowsVerify;

/// What a layer carries from one token to the next.
enum LayerState {
    /// Softmax attention: keys and values for every position so far.
    Kv {
        // Raw rows in cfg.kv_type's format.
        k: DeviceBuffer<u8>,
        v: DeviceBuffer<u8>,
        /// The QSA indexer's keys, on layers that attend sparsely.
        qsa: Option<QsaCache>,
    },
    /// Linear attention: a fixed-size recurrent matrix plus the convolution's
    /// short history. Neither grows with sequence length.
    Delta {
        state: DeviceBuffer<f32>,
        conv: DeviceBuffer<f32>,
    },
}

/// A conversation held apart from the active one (see [`Runtime::set_slots`]):
/// its layer states, one set per shard, and where its sequence stands. The
/// active slot's states live in the shards themselves; its entry is empty.
#[derive(Default)]
struct SlotStore {
    states: Vec<Vec<LayerState>>,
    n_past: usize,
    history: Vec<u32>,
    rope_shift: i32,
    image_spans: Vec<(usize, usize, i32)>,
    /// Its own turn snapshot, so a request on another slot does not take it.
    turn_snap: Option<TurnSnapshot>,
    /// Its own MTP head cache (the active slot's is in `Runtime::spec`).
    head: Option<spec::SlotHead>,
}

/// A hybrid's recurrent state (delta-net matrices and convolutions, QSA keys,
/// PLE history) as it stood just before the prompt's last `<|im_start|>`.
///
/// The KV rows are kept by position, so cutting the cache back is free, but a
/// recurrent state only moves forward. A chat's next request repeats the
/// transcript up to the assistant turn just generated and then differs (the
/// template drops or rewrites the previous `<think>`), so without this a
/// hybrid reprocessed the whole conversation every turn. Restoring the state
/// at that header reuses everything before it.
struct TurnSnapshot {
    /// Tokens before the header; the state is what they leave.
    tokens: Vec<u32>,
    /// [`Runtime::walk_state`] without the KV rows, in pinned memory: the
    /// copies are queued on each shard's stream, in order with the prefill,
    /// and nothing waits for them. Kept across turns, as its size is fixed.
    bytes: airun_cuda::PinnedBuffer<u8>,
    /// Whether `bytes` ends with the MTP head's `pending` row at the same
    /// point (hidden of the last token): the prefill that took it ran the
    /// head too, so the head's cache rows before it are good.
    pending: bool,
}

/// AIRUN_NO_STATE_SNAPSHOT=1 turns the turn snapshot off.
fn snapshot_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_STATE_SNAPSHOT").is_ok_and(|v| v.trim() != "0"))
}

/// The first bytes of a [`Runtime::save_state`] file, with its format version.
const STATE_MAGIC: &[u8] = b"AIRUNST1";

/// A cache buffer of any element type, for saving and restoring it.
enum StateBuf<'a> {
    U8(&'a DeviceBuffer<u8>),
    U16(&'a DeviceBuffer<u16>),
    F32(&'a DeviceBuffer<f32>),
}

impl StateBuf<'_> {
    /// Device address and byte count of `len` elements from `offset`.
    fn raw(&self, offset: usize, len: usize) -> (*mut std::ffi::c_void, usize) {
        fn go<T: Copy + Default>(b: &DeviceBuffer<T>, offset: usize, len: usize) -> (*mut std::ffi::c_void, usize) {
            let size = std::mem::size_of::<T>();
            (b.as_mut_ptr().wrapping_byte_add(offset * size), len * size)
        }
        match self {
            Self::U8(b) => go(b, offset, len),
            Self::U16(b) => go(b, offset, len),
            Self::F32(b) => go(b, offset, len),
        }
    }

    /// Appends `len` elements from `offset` to `out`, as little-endian bytes.
    fn save(&self, offset: usize, len: usize, stream: &Stream, out: &mut Vec<u8>) -> Result<()> {
        fn go<T: Copy + Default>(b: &DeviceBuffer<T>, offset: usize, len: usize, stream: &Stream,
            out: &mut Vec<u8>) -> Result<()> {
            let mut host = vec![T::default(); len];
            b.copy_to_host_at(offset, &mut host, stream)?;
            stream.synchronize()?;
            // SAFETY: plain integers and floats, read as their bytes (all
            // supported hosts are little-endian, as the GPUs are).
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(host.as_ptr().cast::<u8>(), std::mem::size_of_val(host.as_slice()))
            });
            Ok(())
        }
        match self {
            Self::U8(b) => go(b, offset, len, stream, out),
            Self::U16(b) => go(b, offset, len, stream, out),
            Self::F32(b) => go(b, offset, len, stream, out),
        }
    }

    /// Fills `len` elements from `offset` with the next bytes of `input`.
    fn load(&self, offset: usize, len: usize, stream: &Stream, input: &mut &[u8]) -> Result<()> {
        fn go<T: Copy + Default>(b: &DeviceBuffer<T>, offset: usize, len: usize, stream: &Stream,
            input: &mut &[u8]) -> Result<()> {
            let n = len * std::mem::size_of::<T>();
            let (head, rest) = input.split_at_checked(n).ok_or_else(|| CoreError::Other(
                "the saved state cannot be restored: it is truncated; save it again".into()))?;
            let mut host = vec![T::default(); len];
            // SAFETY: `host` has exactly n bytes of plain-old-data elements.
            unsafe { std::ptr::copy_nonoverlapping(head.as_ptr(), host.as_mut_ptr().cast::<u8>(), n) };
            b.copy_from_host_at(offset, &host, stream)?;
            stream.synchronize()?;
            *input = rest;
            Ok(())
        }
        match self {
            Self::U8(b) => go(b, offset, len, stream, input),
            Self::U16(b) => go(b, offset, len, stream, input),
            Self::F32(b) => go(b, offset, len, stream, input),
        }
    }
}

/// A sparse attention layer's indexer keys (see airun_cuda::qsa).
struct QsaCache {
    /// Raw keys of the newest tokens, [QSA_RING + batch][dim] f16.
    ring: DeviceBuffer<u16>,
    ring_len: usize,
    /// One pooled key per complete block, [ctx / ratio][dim].
    pooled: DeviceBuffer<f32>,
    ratio: usize,
}

/// Extra ring rows beyond the batch: a call reads up to ratio - 1 earlier keys.
const QSA_RING: usize = 64;
/// Queries scored and selected per launch in a batch, bounding the scores.
const QSA_GROUP: usize = 64;

/// Indexer scratch for the sparse attention layers of one device.
struct QsaScratch {
    /// Indexer queries [b][heads][dim] and raw keys [b][dim].
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    /// Block scores, [QSA_GROUP][max_blocks].
    scores: DeviceBuffer<f32>,
    max_blocks: usize,
    /// Decode: [count - 1, cells...], what attention_split_sel reads.
    list: DeviceBuffer<i32>,
    /// Batches: [b][words] bits of the cells each query sees.
    mask: DeviceBuffer<u32>,
    words: usize,
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
    /// The batch's n-gram embeddings, gathered on the host into pinned
    /// memory (a captured decode graph uploads from that address) ...
    emb_host: std::cell::RefCell<airun_cuda::PinnedBuffer<f32>>,
    /// ... and on the device.
    emb: DeviceBuffer<f32>,
    key: DeviceBuffer<f32>,
    value: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    /// The convolution's history: the last (kernel-1)*ngram normalized rows.
    hist: DeviceBuffer<f32>,
    hist_scratch: DeviceBuffer<f32>,
    /// Recorded once the staged rows have been copied to the device, so the
    /// host does not overwrite them while the upload is still queued.
    uploaded: airun_cuda::Event,
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
    /// With mapped experts: per local MoE layer, how often each expert was
    /// routed to since the last rebalance (u32 on the device) ...
    counts: Option<DeviceBuffer<u32>>,
    /// ... the decayed history the rebalance ranks by ...
    scores: std::cell::RefCell<Vec<f32>>,
    /// ... one expert's worth of VRAM to swap through ...
    staging: Option<DeviceBuffer<u8>>,
    /// ... and the batch's host-side experts copied into VRAM before the
    /// expert kernels run: gate, up and down areas of STAGE_CAP experts, the
    /// per-id staging index, the staged experts and their count.
    stage: Option<([DeviceBuffer<u8>; 3], DeviceBuffer<i32>, DeviceBuffer<i32>, DeviceBuffer<i32>)>,
    /// ... and the routed experts found in VRAM vs in host memory.
    mapped_hits: std::cell::Cell<(u64, u64)>,
    /// Expert-major batches (see hybrid::moe_grouped): per-expert counts,
    /// offsets, the sorted pairs, and per-pair down projections.
    /// (counts, offsets, perm, ypair, fp16 activations) of the expert-major batch path.
    grouped: Option<(DeviceBuffer<i32>, DeviceBuffer<i32>, DeviceBuffer<i32>, DeviceBuffer<f32>, DeviceBuffer<u16>)>,
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
    /// For matmuls on the CPU (layers in RAM, see cpu.rs): the input and
    /// output in pinned memory, and whether `host_x` holds the current input.
    host_x: std::cell::RefCell<Option<airun_cuda::PinnedBuffer<f32>>>,
    host_y: std::cell::RefCell<Option<airun_cuda::PinnedBuffer<f32>>>,
    host_valid: std::cell::Cell<bool>,
    /// A feed-forward's gate and up products on the CPU (cpu_ffn).
    host_mid: std::cell::RefCell<(Vec<f32>, Vec<f32>)>,
    /// The same input as NVFP4 blocks for the FP4 tensor cores, where the
    /// device and the model have them; same freshness protocol.
    fp4: Option<DeviceBuffer<u8>>,
    fp4_valid: std::cell::Cell<bool>,
}

/// Everything that lives on one GPU.
struct DeviceState {
    device: i32,
    stream: Stream,
    /// Copies host-side experts into the staging area while `stream`
    /// computes the resident ones (see run_moe), forked and joined by events.
    side: Stream,
    fork: airun_cuda::Event,
    join: airun_cuda::Event,
    /// Cross-GPU handoff (see Runtime::handoff): this shard finished the
    /// rows it passes on, and the rows it received have been copied in.
    sent: airun_cuda::Event,
    copied: airun_cuda::Event,

    x: DeviceBuffer<f32>,
    xb: DeviceBuffer<f32>,
    xb2: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    attn: DeviceBuffer<f32>,
    attn_partial: DeviceBuffer<f32>,
    /// With --parallel: per local layer, 2 x slots device pointers -- a
    /// decode_rows step's K (or delta-net state) caches for rows 0..n, then
    /// its V (or convolution) ones -- rewritten before each step, so the
    /// batched kernels can reach every row's own state in one launch.
    rows_tab: Option<DeviceBuffer<u64>>,
    /// attn_partial for every slot at once, for the batched attention.
    rows_partial: Option<DeviceBuffer<f32>>,
    /// The VRAM copy (K, V) of one layer's cache that the attention layers
    /// whose KV cache stays in RAM run on (see Shard::kv_ram_layers).
    kv_stage: Option<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    positions: DeviceBuffer<i32>,
    tokens: DeviceBuffer<i32>,
    logits: Option<DeviceBuffer<f32>>,
    /// What the text's rotation positions lag its KV slots by: the positions
    /// images saved (see RopeParams::shift). One i32, read by the kernels, so
    /// captured graphs follow it.
    rope_shift: DeviceBuffer<i32>,
    /// An image chunk's (t, row, col) rotation positions, [3][batch].
    mpos: DeviceBuffer<i32>,

    /// Activation quantized to 8-bit groups, plus whether its contents match
    /// the input of the current matvec group. Sized for the widest matvec
    /// input. See [`matvec`] for the freshness protocol.
    q8act: QAct,

    hybrid: Option<HybridScratch>,
    moe: Option<MoeScratch>,
    hc: Option<HcScratch>,
    qsa: Option<QsaScratch>,

    /// One entry per layer on this device, in the same order as `layers`.
    states: Vec<LayerState>,
    layers: Vec<usize>,
    /// Per layer, parallel to `states`; allocated for delta-net layers once
    /// speculative decoding starts.
    stash: Vec<Option<DeltaStash>>,
}

/// The captured decode step and the pinned staging its replays read.
struct DecodeGraph {
    /// One per shard, each replayed on its device's stream; the residual
    /// rows cross between GPUs outside the graphs.
    graphs: Vec<Option<airun_cuda::Graph>>,
    token: airun_cuda::PinnedBuffer<i32>,
    position: airun_cuda::PinnedBuffer<i32>,
}

/// Captured [`Runtime::decode_rows`] passes, one per list of slots (in row
/// order), each a graph per shard as [`DecodeGraph`]. Every per-row state
/// buffer belongs to its slot for good (switch_slot moves the handles, not
/// the memory) and positions come from device buffers, so a graph stays good
/// for as long as the same slots decode together. Without them a batched
/// step issued ~2 launches per row and stateful layer one by one.
struct MixedGraphs {
    entries: Vec<(Vec<usize>, Vec<Option<airun_cuda::Graph>>)>,
    tokens: airun_cuda::PinnedBuffer<i32>,
    positions: airun_cuda::PinnedBuffer<i32>,
}

/// Slot lists kept captured at once; the oldest goes first.
const MIXED_GRAPHS: usize = 8;

/// AIRUN_NO_ROWS_BATCH=1 runs decode_rows' stateful kernels once per row.
fn rows_batched_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_ROWS_BATCH").is_ok_and(|v| v.trim() != "0"))
}

/// AIRUN_NO_ROWS_GRAPH=1 runs decode_rows without graphs.
fn mixed_graphs_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_ROWS_GRAPH").is_ok_and(|v| v.trim() != "0"))
}

/// Whether AIRUN_NO_GRAPH=1 turned off decode graph capture, or some
/// layer's matmuls run on the CPU, which a graph cannot hold.
fn graphs_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_GRAPH").is_ok_and(|v| v.trim() != "0")) || crate::cpu::in_use()
}

/// Whether `w`'s matmuls run on the CPU: its weights stay in RAM and
/// AIRUN_RAM_ON_GPU=1 did not ask for the old copy into VRAM (see cpu.rs).
fn on_cpu(w: &GpuTensor) -> bool {
    w.ram.is_some() && crate::cpu::pass_on_cpu()
}

/// A dense feed-forward whose three matrices are on the CPU, there in one
/// go: `out = W_down (silu(W_gate x) * W_up x)`. One trip of the input down
/// and the product up, where three matvecs moved the 17408-wide gate and up
/// across PCIe too.
fn cpu_ffn(gate: &GpuTensor, up: &GpuTensor, down: &GpuTensor, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>,
           n: usize, q8: &QAct, stream: &Stream) -> Result<()> {
    let (nx, nf, ny) = (n * gate.k, n * gate.rows, n * down.rows);
    let grow = |slot: &std::cell::RefCell<Option<airun_cuda::PinnedBuffer<f32>>>, len: usize| -> Result<()> {
        if slot.borrow().as_ref().is_none_or(|b| b.len() < len) {
            *slot.borrow_mut() = Some(airun_cuda::PinnedBuffer::new(len)?);
        }
        Ok(())
    };
    grow(&q8.host_x, nx)?;
    grow(&q8.host_y, ny)?;
    let mut hx = q8.host_x.borrow_mut();
    let hx = hx.as_mut().expect("grown above");
    x.copy_to_host_at(0, &mut hx.as_mut_slice()[..nx], stream)?;
    stream.synchronize()?;
    // host_x now holds the FFN input, not what an earlier matvec cached.
    q8.host_valid.set(false);
    fn host(w: &GpuTensor) -> &[u8] {
        w.ram.as_ref().map(|r| r.host.as_slice()).expect("on_cpu checked it")
    }
    let mut mid = q8.host_mid.borrow_mut();
    let (g, u) = &mut *mid;
    g.resize(nf, 0.0);
    u.resize(nf, 0.0);
    let xs = &hx.as_slice()[..nx];
    crate::cpu::matvec(gate.ty, host(gate), gate.row_bytes, gate.k, gate.rows, xs, n, g)?;
    crate::cpu::matvec(up.ty, host(up), up.row_bytes, up.k, up.rows, xs, n, u)?;
    let (sg, su) = (gate.global_scale, up.global_scale);
    for (a, b) in g.iter_mut().zip(u.iter()) {
        let gv = *a * sg;
        *a = gv / (1.0 + (-gv).exp()) * (*b * su);
    }
    let mut hy = q8.host_y.borrow_mut();
    let hy = hy.as_mut().expect("grown above");
    crate::cpu::matvec(down.ty, host(down), down.row_bytes, down.k, down.rows, g, n, &mut hy.as_mut_slice()[..ny])?;
    if down.global_scale != 1.0 {
        hy.as_mut_slice()[..ny].iter_mut().for_each(|v| *v *= down.global_scale);
    }
    out.copy_from_host_at(0, &hy.as_slice()[..ny], stream)?;
    Ok(())
}

/// [`matvec_reuse`] on the CPU threads: the input comes to pinned memory
/// (once for several matrices of the same input), the product goes back.
fn cpu_matvec(w: &GpuTensor, x: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, n: usize, q8: &QAct,
              stream: &Stream) -> Result<()> {
    let ram = w.ram.as_ref().expect("on_cpu checked it");
    let (nx, ny) = (n * w.k, n * w.rows);
    let grow = |slot: &std::cell::RefCell<Option<airun_cuda::PinnedBuffer<f32>>>, len: usize| -> Result<()> {
        if slot.borrow().as_ref().is_none_or(|b| b.len() < len) {
            *slot.borrow_mut() = Some(airun_cuda::PinnedBuffer::new(len)?);
        }
        Ok(())
    };
    grow(&q8.host_x, nx)?;
    grow(&q8.host_y, ny)?;
    let mut hx = q8.host_x.borrow_mut();
    let hx = hx.as_mut().expect("grown above");
    if !q8.host_valid.get() {
        x.copy_to_host_at(0, &mut hx.as_mut_slice()[..nx], stream)?;
    }
    // Also orders the previous product's upload out of host_y before it is
    // overwritten.
    stream.synchronize()?;
    q8.host_valid.set(true);
    let mut hy = q8.host_y.borrow_mut();
    let hy = hy.as_mut().expect("grown above");
    crate::cpu::matvec(w.ty, ram.host.as_slice(), w.row_bytes, w.k, w.rows, &hx.as_slice()[..nx], n,
        &mut hy.as_mut_slice()[..ny])?;
    if w.global_scale != 1.0 {
        hy.as_mut_slice()[..ny].iter_mut().for_each(|v| *v *= w.global_scale);
    }
    out.copy_from_host_at(0, &hy.as_slice()[..ny], stream)?;
    Ok(())
}

/// A loaded model ready to generate.
/// How far a prompt's prefill has got: `processed` of `total` tokens, the
/// first `cached` of them reused from the cache, `elapsed_ms` since it began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillProgress {
    pub total: usize,
    pub cached: usize,
    pub processed: usize,
    pub elapsed_ms: u64,
}

/// See [`Runtime::set_prefill_hook`].
pub type PrefillHook = Box<dyn FnMut(PrefillProgress) + Send>;

pub struct Runtime {
    pub model: Model,
    /// Shared, so a server can tokenize while a generation holds the runtime.
    pub tokenizer: std::sync::Arc<Tokenizer>,
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
    mixed_graphs: Option<MixedGraphs>,
    /// A batched step failed to capture once; they run uncaptured since.
    mixed_graph_off: bool,
    /// Hybrids only: the recurrent state at the last turn boundary prefilled.
    turn_snap: Option<TurnSnapshot>,
    /// Greedy decoding's pick on the device, its pinned landing slot and the
    /// argmax scratch.
    pick: Option<(DeviceBuffer<i32>, airun_cuda::PinnedBuffer<i32>, DeviceBuffer<i32>)>,
    /// The pipelined greedy decode step; built on first use.
    chain: Option<ChainGraph>,
    /// Pinned landing buffer for the logits a sampler reads, reused per token.
    logits_host: Option<airun_cuda::PinnedBuffer<f32>>,
    /// decode_rows' logits, on the output device and pinned on the host,
    /// kept: allocating them each step synchronized the device (cudaFree).
    rows_logits: Option<(DeviceBuffer<f32>, airun_cuda::PinnedBuffer<f32>)>,
    /// Set during a speculative verify: delta-net layers stash their inputs
    /// instead of advancing their state.
    defer: bool,
    /// Set when the caller projects the logits itself (a verify needs every
    /// row, not just the last).
    skip_logits: bool,
    /// The MTP draft head's state, once speculative decoding has run.
    spec: Option<spec::Spec>,
    /// Images a prompt may name by their sentinel ids (see [`ImageInput`]).
    images: Vec<std::sync::Arc<ImageInput>>,
    /// While an image chunk runs: its row count, so RoPE reads the chunk's
    /// (t, row, col) positions; and that its embeddings are already in place
    /// of the token lookup.
    image_rows: Option<usize>,
    embeds_ready: bool,
    /// Rotation positions the cached images saved (each takes max(rows,
    /// cols) positions for rows * cols slots): the text's position is its slot
    /// minus this. Mirrored into every shard's `rope_shift`.
    rope_shift: i32,
    /// (first slot, slots, shift before it) of every cached image, to rewind.
    image_spans: Vec<(usize, usize, i32)>,
    /// Conversations kept side by side (`set_slots`), the active one's state
    /// in the shards and the others' here.
    slots: Vec<SlotStore>,
    active_slot: usize,
    /// During a multi-slot decode step: the slot of each row.
    mixed: Option<Vec<usize>>,
    /// Tokens per row of a mixed pass: 1 for decode_rows, 1 + drafts for a
    /// batched verify (spec::verify_rows) or the head's rows.
    mixed_t: usize,
    /// Each row's first position in a mixed pass of several tokens a row.
    mixed_past: Vec<usize>,
    /// The mixed pass is the MTP head's: attention reads each row's head cache.
    mixed_head: bool,
    /// The last batched verify's rows, which spec::accept_rows completes.
    rows_spec: Option<spec::RowsSpec>,
    /// Drafts per speculative step (see `set_draft`); 0 decodes plainly.
    draft: usize,
    /// Called after each prefill batch (see `set_prefill_hook`).
    prefill_hook: Option<PrefillHook>,
    /// Forward passes since the mapped experts were last rebalanced.
    since_rebalance: usize,
    /// Rebalances so far; every PROFILE_SAVE_EVERY of them saves the profile.
    rebalances: usize,
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

impl Drop for Runtime {
    fn drop(&mut self) {
        // The counts since the last rebalance are lost; the rest is kept.
        self.save_expert_profile();
    }
}

/// A zeroed K or V cache of `bytes`, in VRAM or (`in_ram`) pinned RAM.
fn kv_cache(dev: i32, bytes: usize, in_ram: bool, stream: &Stream) -> Result<DeviceBuffer<u8>> {
    Ok(if in_ram { DeviceBuffer::in_ram(dev, bytes)? } else { DeviceBuffer::zeroed(dev, bytes, stream)? })
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
            let rope_shift = DeviceBuffer::zeroed(dev, 1, &stream)?;

            let mut layer_states = Vec::with_capacity(shard.layers.len());
            for &l in &shard.layers {
                layer_states.push(match cfg.layer_kind(l) {
                    LayerKind::Attention => LayerState::Kv {
                        k: kv_cache(dev, cfg.n_head_kv * ctx * cfg.kv_type.row_bytes(cfg.head_dim),
                            placement.kv_in_ram(l), &stream)?,
                        v: kv_cache(dev, cfg.n_head_kv * ctx * cfg.kv_type.row_bytes(cfg.head_dim),
                            placement.kv_in_ram(l), &stream)?,
                        qsa: match qsa_ratio(cfg, l, ctx) {
                            Some(ratio) => {
                                let dim = cfg.indexer.as_ref().expect("qsa_ratio checked it").dim;
                                Some(QsaCache {
                                    ring: DeviceBuffer::zeroed(dev, (QSA_RING + b) * dim, &stream)?,
                                    ring_len: QSA_RING + b,
                                    pooled: DeviceBuffer::zeroed(dev, ctx / ratio * dim, &stream)?,
                                    ratio,
                                })
                            }
                            None => None,
                        },
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

            // Layers of this shard whose experts are split VRAM/host, and the
            // widest expert among them.
            let mapped: Vec<&GpuTensor> = shard.layers.iter().filter_map(|&l| match &model.layers[l].ffn {
                FfnWeights::Moe { gate_exps, .. } if gate_exps.mapped.is_some() => Some(gate_exps),
                _ => None,
            }).collect();
            let mapped_layers = if mapped.is_empty() { 0 } else { shard.layers.len() };
            let mapped_stride = shard.layers.iter().filter_map(|&l| match &model.layers[l].ffn {
                FfnWeights::Moe { gate_exps, up_exps, down_exps, .. } =>
                    Some(gate_exps.expert_stride.max(up_exps.expert_stride).max(down_exps.expert_stride)),
                _ => None,
            }).max().unwrap_or(0);

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
                    counts: if mapped_layers > 0 {
                        Some(DeviceBuffer::zeroed(dev, mapped_layers * moe.n_experts, &stream)?)
                    } else {
                        None
                    },
                    // Seeded from the routing profile at the decayed counts'
                    // steady state, so a rebalance weighs what a few tokens
                    // route against the long-run use, not against nothing.
                    scores: std::cell::RefCell::new({
                        let mut sc = vec![0.0; mapped_layers * moe.n_experts];
                        if let (true, Some(p)) = (mapped_layers > 0, &model.expert_profile) {
                            let steady = REBALANCE_EVERY as f32 * moe.n_experts_used as f32 / (1.0 - REBALANCE_DECAY);
                            for (local, &l) in shard.layers.iter().enumerate() {
                                if let Some(sh) = p.borrow().shares(l) {
                                    for (d, x) in sc[local * moe.n_experts..][..moe.n_experts].iter_mut().zip(sh) {
                                        *d = x * steady;
                                    }
                                }
                            }
                        }
                        sc
                    }),
                    staging: if mapped_layers > 0 { Some(DeviceBuffer::new(dev, mapped_stride)?) } else { None },
                    stage: if mapped_layers > 0 {
                        Some(([DeviceBuffer::new(dev, STAGE_CAP * mapped_stride)?,
                               DeviceBuffer::new(dev, STAGE_CAP * mapped_stride)?,
                               DeviceBuffer::new(dev, STAGE_CAP * mapped_stride)?],
                              DeviceBuffer::new(dev, STAGE_CAP)?, DeviceBuffer::new(dev, STAGE_CAP)?,
                              DeviceBuffer::new(dev, 1)?))
                    } else {
                        None
                    },
                    mapped_hits: std::cell::Cell::new((0, 0)),
                    grouped: if b >= GROUPED_MIN {
                        Some((DeviceBuffer::new(dev, moe.n_experts)?, DeviceBuffer::new(dev, moe.n_experts + 1)?,
                              DeviceBuffer::new(dev, b * moe.n_experts_used)?,
                              DeviceBuffer::new(dev, b * moe.n_experts_used * cfg.n_embd)?,
                              DeviceBuffer::new(dev, b * cfg.n_embd)?))
                    } else {
                        None
                    },
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
                            emb_host: std::cell::RefCell::new(airun_cuda::PinnedBuffer::new(b * p.n_heads() * p.head_dim)?),
                            emb: DeviceBuffer::new(dev, b * p.n_heads() * p.head_dim)?,
                            key: DeviceBuffer::new(dev, b * hc_width)?,
                            value: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                            gated: DeviceBuffer::new(dev, b * hc_width)?,
                            hist: DeviceBuffer::zeroed(dev, p.conv_history() * hc_width, &stream)?,
                            hist_scratch: DeviceBuffer::new(dev, p.conv_history() * hc_width)?,
                            uploaded: airun_cuda::Event::new()?,
                        }),
                        _ => None,
                    },
                }),
                None => None,
            };

            // The smallest block size among this device's sparse layers sets
            // the most blocks a query can score.
            let qsa_scratch = match (&cfg.indexer, shard.layers.iter().filter_map(|&l| qsa_ratio(cfg, l, ctx)).min()) {
                (Some(ix), Some(ratio)) => {
                    let max_blocks = ctx / ratio;
                    let words = ctx.div_ceil(32);
                    Some(QsaScratch {
                        q: DeviceBuffer::new(dev, b * ix.heads * ix.dim)?,
                        k: DeviceBuffer::new(dev, b * ix.dim)?,
                        scores: DeviceBuffer::new(dev, QSA_GROUP.min(b) * max_blocks)?,
                        max_blocks,
                        list: DeviceBuffer::new(dev, 1 + shard.layers.iter()
                            .filter_map(|&l| qsa_ratio(cfg, l, ctx)).map(|r| ix.width(r)).max().unwrap_or(0))?,
                        mask: DeviceBuffer::new(dev, b * words)?,
                        words,
                    })
                }
                _ => None,
            };

            states.push(DeviceState {
                device: dev,
                stream,
                side: Stream::new(dev)?,
                fork: airun_cuda::Event::new()?,
                join: airun_cuda::Event::new()?,
                sent: airun_cuda::Event::new()?,
                copied: airun_cuda::Event::new()?,
                x: DeviceBuffer::new(dev, b * hc_width)?,
                xb: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                xb2: DeviceBuffer::new(dev, b * cfg.n_embd)?,
                q: DeviceBuffer::new(dev, b * q_width)?,
                k: DeviceBuffer::new(dev, b * k_width)?,
                v: DeviceBuffer::new(dev, b * v_width)?,
                attn: DeviceBuffer::new(dev, b * attn_width)?,
                attn_partial: DeviceBuffer::new(dev, cfg.n_head * ops::attention_splits(ctx) * (cfg.head_dim + 2))?,
                rows_tab: None,
                rows_partial: None,
                kv_stage: if shard.kv_ram_layers.is_empty() { None } else {
                    let bytes = cfg.n_head_kv * ctx * cfg.kv_type.row_bytes(cfg.head_dim);
                    Some((DeviceBuffer::new(dev, bytes)?, DeviceBuffer::new(dev, bytes)?))
                },
                gate: DeviceBuffer::new(dev, b * ffn_width.max(1))?,
                up: DeviceBuffer::new(dev, b * ffn_width.max(1))?,
                positions: DeviceBuffer::new(dev, b)?,
                tokens: DeviceBuffer::new(dev, b)?,
                rope_shift,
                mpos: DeviceBuffer::new(dev, 3 * b)?,
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
                    host_x: std::cell::RefCell::new(None),
                    host_y: std::cell::RefCell::new(None),
                    host_valid: std::cell::Cell::new(false),
                    host_mid: std::cell::RefCell::new((Vec::new(), Vec::new())),
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
                qsa: qsa_scratch,

                states: layer_states,
                layers: shard.layers.clone(),
                stash: Vec::new(),
            });
        }

        Ok(Self {
            model,
            tokenizer: std::sync::Arc::new(tokenizer),
            states,
            ctx,
            max_batch,
            n_past: 0,
            history: Vec::new(),
            decode_graph: None,
            mixed_graphs: None,
            rows_logits: None,
            mixed_graph_off: false,
            turn_snap: None,
            pick: None,
            chain: None,
            logits_host: None,
            defer: false,
            skip_logits: false,
            spec: None,
            images: Vec::new(),
            image_rows: None,
            embeds_ready: false,
            rope_shift: 0,
            image_spans: Vec::new(),
            slots: vec![SlotStore::default()],
            active_slot: 0,
            mixed: None,
            mixed_t: 1,
            mixed_past: Vec::new(),
            mixed_head: false,
            rows_spec: None,
            draft: 0,
            prefill_hook: None,
            since_rebalance: 0,
            rebalances: 0,
        })
    }

    /// Installs (or with None removes) a callback told after each prefill
    /// batch how far the prompt has got -- what llama-server's
    /// `return_progress` streams as `prompt_progress`. The batch is queued,
    /// not waited on, when the call comes; the launch queue keeps the host at
    /// most about a batch ahead of the GPU.
    pub fn set_prefill_hook(&mut self, hook: Option<PrefillHook>) {
        self.prefill_hook = hook;
    }

    /// Reports `processed` of `total` prompt tokens (`cached` of them from
    /// the cache) to the prefill hook, if any.
    pub(crate) fn prefill_report(&mut self, processed: usize, total: usize, cached: usize, start: std::time::Instant) {
        if let Some(h) = self.prefill_hook.as_mut() {
            h(PrefillProgress { total, cached, processed, elapsed_ms: start.elapsed().as_millis() as u64 });
        }
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
        let stream = self.model.layers.iter().fold((0, 0), |(hits, misses), layer| {
            if let FfnWeights::Moe { gate_exps, up_exps, down_exps, .. } = &layer.ffn {
                [gate_exps, up_exps, down_exps].iter().fold((hits, misses), |(h, m), w| {
                    let (wh, wm) = w.cache_stats(); (h + wh, m + wm)
                })
            } else { (hits, misses) }
        });
        // Mapped experts count once per routed expert, not per tensor.
        let (mh, mm) = self.states.iter().filter_map(|s| s.moe.as_ref())
            .fold((0, 0), |(h, m), s| { let (a, b) = s.mapped_hits.get(); (h + a, m + b) });
        (stream.0 + mh, stream.1 + mm)
    }

    /// Tokens per forward pass this runtime was built for, which the planner
    /// may have lowered from the batch asked for.
    pub fn max_batch(&self) -> usize {
        self.max_batch
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
        // Its KV rows are about to be overwritten.
        self.turn_snap = None;
        self.image_spans.clear();
        let _ = self.set_rope_shift(0);
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

    /// The sequence in the caches, as bytes: its tokens, the KV rows they
    /// filled, and every recurrent state (delta-net matrices and convolution
    /// histories, the QSA key ring, the PLE history). [`load_state`] puts it
    /// back, so a conversation resumes without replaying its prompt --
    /// llama-server's slot save and restore.
    pub fn save_state(&self) -> Result<Vec<u8>> {
        if !self.image_spans.is_empty() {
            return Err(CoreError::Other(
                "a conversation holding images cannot be saved yet; send the images again after a restore".into()));
        }
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        for v in self.state_fingerprint() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.n_past as u64).to_le_bytes());
        out.extend_from_slice(&(self.history.len() as u64).to_le_bytes());
        for t in &self.history {
            out.extend_from_slice(&t.to_le_bytes());
        }
        self.walk_state(true, |buf, offset, len, stream| buf.save(offset, len, stream, &mut out))?;
        Ok(out)
    }

    /// Restores what [`save_state`] wrote, for this same model and context.
    /// Returns how many tokens the caches now hold.
    pub fn load_state(&mut self, bytes: &[u8]) -> Result<usize> {
        let bad = |why: &str| CoreError::Other(format!(
            "the saved state cannot be restored: {why}; save it again with this model and --ctx-size"));
        let mut input = bytes.strip_prefix(STATE_MAGIC).ok_or_else(|| bad("it is not an airun state file"))?;
        let u64_at = |input: &mut &[u8]| -> Result<u64> {
            let (head, rest) = input.split_at_checked(8).ok_or_else(|| bad("it is truncated"))?;
            *input = rest;
            Ok(u64::from_le_bytes(head.try_into().unwrap()))
        };
        for want in self.state_fingerprint() {
            if u64_at(&mut input)? != want {
                return Err(bad("it was saved from a different model or context size"));
            }
        }
        let n_past = u64_at(&mut input)? as usize;
        let n_hist = u64_at(&mut input)? as usize;
        if n_past > self.ctx || n_hist != n_past {
            return Err(bad("its token count does not fit this context"));
        }
        let (hist, rest) = input.split_at_checked(n_hist * 4).ok_or_else(|| bad("it is truncated"))?;
        let history: Vec<u32> = hist.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        input = rest;
        // The rows written depend on the position, so it must be set first.
        self.n_past = n_past;
        self.turn_snap = None;
        let loaded = self.walk_state(true, |buf, offset, len, stream| buf.load(offset, len, stream, &mut input));
        if let Err(e) = loaded {
            // Half-written caches are garbage: start over rather than resume them.
            self.reset();
            return Err(e);
        }
        if !input.is_empty() {
            self.reset();
            return Err(bad("it has trailing data"));
        }
        self.history = history;
        if let Some(sp) = self.spec.as_mut() {
            sp.synced = false;
        }
        Ok(n_past)
    }

    /// The final hidden state (after the output norm) of every token of
    /// `tokens`, run as a fresh sequence, `n_embd` floats per token: what
    /// embeddings are pooled from. The caches then hold `tokens`.
    pub fn hidden_states(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.len() > self.ctx {
            return Err(CoreError::Other(format!(
                "{} tokens do not fit a context of {}; shorten the input or raise --ctx-size",
                tokens.len(), self.ctx)));
        }
        self.reset();
        if let Some(sp) = self.spec.as_mut() {
            sp.synced = false;
        }
        let n_embd = self.model.config.n_embd;
        let mut out = vec![0f32; tokens.len() * n_embd];
        for start in (0..tokens.len()).step_by(self.max_batch.max(1)) {
            let chunk = &tokens[start..(start + self.max_batch).min(tokens.len())];
            let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
            let positions: Vec<i32> = (0..chunk.len()).map(|i| (self.n_past + i) as i32).collect();
            self.skip_logits = true;
            let queued = self.queue_forward(&ids, &positions);
            self.skip_logits = false;
            queued?;
            self.n_past += chunk.len();
            self.history.extend_from_slice(chunk);
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            st.xb.copy_to_host(&mut out[start * n_embd..(start + chunk.len()) * n_embd], &st.stream)?;
            st.stream.synchronize()?;
        }
        Ok(out)
    }

    /// What a state file must agree with to be loaded here.
    fn state_fingerprint(&self) -> [u64; 6] {
        let cfg = &self.model.config;
        [cfg.n_layers as u64, cfg.n_embd as u64, cfg.vocab_size as u64, self.ctx as u64,
            cfg.kv_type.row_bytes(cfg.head_dim) as u64, cfg.n_head_kv as u64]
    }

    /// Calls `f(buffer, offset, len, stream)` on every stretch of cache that
    /// holds the current sequence, in a fixed order. Without `kv`, only the
    /// state that does not stay valid when the sequence is cut back (see
    /// [`TurnSnapshot`]).
    fn walk_state(&self, kv: bool, mut f: impl FnMut(StateBuf<'_>, usize, usize, &Stream) -> Result<()>)
        -> Result<()> {
        let cfg = &self.model.config;
        let row = cfg.kv_type.row_bytes(cfg.head_dim);
        for st in &self.states {
            device::set_device(st.device)?;
            for state in &st.states {
                match state {
                    LayerState::Kv { k, v, qsa } => {
                        // [head][ctx][row]: the first n_past rows of each head.
                        for h in (0..cfg.n_head_kv).filter(|_| kv) {
                            f(StateBuf::U8(k), h * self.ctx * row, self.n_past * row, &st.stream)?;
                            f(StateBuf::U8(v), h * self.ctx * row, self.n_past * row, &st.stream)?;
                        }
                        if let Some(q) = qsa {
                            f(StateBuf::U16(&q.ring), 0, q.ring.len(), &st.stream)?;
                            f(StateBuf::F32(&q.pooled), 0, q.pooled.len(), &st.stream)?;
                        }
                    }
                    LayerState::Delta { state, conv } => {
                        f(StateBuf::F32(state), 0, state.len(), &st.stream)?;
                        f(StateBuf::F32(conv), 0, conv.len(), &st.stream)?;
                    }
                }
            }
            if let Some(p) = st.hc.as_ref().and_then(|h| h.ple.as_ref()) {
                f(StateBuf::F32(&p.hist), 0, p.hist.len(), &st.stream)?;
            }
        }
        Ok(())
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
            ops::matvec(out_w.kernel_type(), out_w.ptr(), st.xb.offset_ptr(r0 * cfg.n_embd),
                buf.as_mut_ptr(), cfg.n_embd, cfg.vocab_size, m, out_w.row_bytes, &st.stream)?;
            if out_w.global_scale != 1.0 {
                ops::scale(buf.as_mut_ptr(), out_w.global_scale, m * cfg.vocab_size, &st.stream)?;
            }
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

    /// Moves the most used host-side experts into VRAM in place of the least
    /// used resident ones, layer by layer, from the routing counts the MoE
    /// layers accumulated on the device. Runs every few forward passes: the
    /// counts read back is a synchronization, and each swap moves three
    /// experts each way across PCIe.
    fn rebalance_experts(&mut self) -> Result<()> {
        const MAX_SWAPS: usize = 16;
        self.since_rebalance += 1;
        if self.since_rebalance < REBALANCE_EVERY {
            return Ok(());
        }
        self.since_rebalance = 0;
        self.rebalances += 1;
        let Some(ne) = self.model.config.moe.as_ref().map(|m| m.n_experts) else { return Ok(()) };
        for st in &self.states {
            let Some(m) = &st.moe else { continue };
            let (Some(counts), Some(staging)) = (&m.counts, &m.staging) else { continue };
            device::set_device(st.device)?;
            let host = counts.to_vec(&st.stream)?;
            counts.fill_zero(&st.stream)?;
            let mut scores = m.scores.borrow_mut();
            let (mut hits, mut misses) = m.mapped_hits.get();
            for (local, &layer) in st.layers.iter().enumerate() {
                let FfnWeights::Moe { gate_exps, up_exps, down_exps, .. } = &self.model.layers[layer].ffn else {
                    continue;
                };
                let tensors = [gate_exps, up_exps, down_exps];
                let Some(gm) = &gate_exps.mapped else { continue };
                let sc = &mut scores[local * ne..(local + 1) * ne];
                let c = &host[local * ne..(local + 1) * ne];
                let place = gm.place.borrow().clone();
                for e in 0..ne {
                    if place[e].is_ok() { hits += u64::from(c[e]) } else { misses += u64::from(c[e]) }
                    sc[e] = sc[e] * REBALANCE_DECAY + c[e] as f32;
                }
                if let Some(p) = &self.model.expert_profile {
                    p.borrow_mut().add(layer, c);
                }
                let mut resident: Vec<usize> = (0..ne).filter(|&e| place[e].is_ok()).collect();
                let mut outside: Vec<usize> = (0..ne).filter(|&e| place[e].is_err()).collect();
                resident.sort_by(|&a, &b| sc[a].total_cmp(&sc[b]));
                outside.sort_by(|&a, &b| sc[b].total_cmp(&sc[a]));
                let mut swapped = 0;
                for (&hot, &cold) in outside.iter().zip(&resident).take(MAX_SWAPS) {
                    // A swap moves both experts across PCIe, three times
                    // what one miss costs; it pays only when the hot expert
                    // is expected to be routed a few times more than the
                    // cold one (the scores count about the last 80 tokens).
                    if sc[hot] <= sc[cold] * 1.25 + SWAP_MARGIN {
                        break;
                    }
                    for t in tensors {
                        let tm = t.mapped.as_ref().ok_or_else(|| CoreError::Other(
                            "a layer's expert tensors disagree on where their experts live; reload the model".into()))?;
                        tm.swap(&t.data, t.expert_stride, hot, cold, staging, &st.stream)?;
                    }
                    swapped += 1;
                }
                if swapped > 0 {
                    for t in tensors {
                        if let Some(tm) = &t.mapped {
                            tm.publish(&t.data, t.expert_stride, &st.stream)?;
                        }
                    }
                }
            }
            m.mapped_hits.set((hits, misses));
            // The table mirrors are reused by the next publish.
            st.stream.synchronize()?;
        }
        if self.rebalances % PROFILE_SAVE_EVERY == 0 {
            self.save_expert_profile();
        }
        Ok(())
    }

    /// Saves the routing profile; a failure is reported and otherwise ignored,
    /// since the profile only speeds up the next load.
    fn save_expert_profile(&self) {
        if let Some(p) = &self.model.expert_profile {
            let mut p = p.borrow_mut();
            if let Err(e) = p.save() {
                eprintln!("warning: cannot save the expert routing profile to {} ({e}); the next load                     will place experts without it -- check the directory is writable, or set                     AIRUN_EXPERT_PROFILE to another path", p.path().display());
            }
        }
    }

    /// Queues `tokens` into the caches without reading anything back: a
    /// prompt chunk that is not the last. Successive chunks overlap across
    /// GPUs; [`Self::forward`] on the last one waits for all of them.
    pub fn prefill_chunk(&mut self, tokens: &[u32]) -> Result<()> {
        self.step(tokens)
    }

    /// Runs `tokens` through the model, leaving the last one's logits on the
    /// device.
    fn step(&mut self, tokens: &[u32]) -> Result<()> {
        self.step_inner(tokens)?;
        self.rebalance_experts()
    }

    fn step_inner(&mut self, tokens: &[u32]) -> Result<()> {
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
        if tokens.len() == 1 && !graphs_disabled()
            && !self.states.iter().any(|s| s.moe.as_ref().is_some_and(|m| m.upload.is_some())) {
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
                graphs: (0..self.states.len()).map(|_| None).collect(),
                token: airun_cuda::PinnedBuffer::new(1)?,
                position: airun_cuda::PinnedBuffer::new(1)?,
            },
        };
        // The previous replay copies its token and position out of these
        // pinned slots when it runs, not when it is launched. Decode reads
        // each step's result before the next, but a batch-1 prefill queues
        // step after step: overwriting the slots first fed every queued
        // replay the latest token and position, and the prompt came out
        // garbled (`-b 1` answered "The capital of France is" with "a 2013
        // American comedy"). Wait for the previous replay to have run.
        if dg.graphs.iter().any(Option::is_some) {
            for st in &self.states {
                device::set_device(st.device)?;
                st.stream.synchronize()?;
            }
        }
        dg.token.as_mut_slice()[0] = token as i32;
        dg.position.as_mut_slice()[0] = self.n_past as i32;
        // The PLE rows are hashed on the host; a captured graph only copies
        // them from the pinned staging, so refresh it before replaying.
        let ple_built = self.model.config.ple.as_ref()
            .and_then(|p| self.states.iter().position(|s| s.layers.contains(&p.layer)))
            .is_some_and(|si| dg.graphs[si].is_some());
        if ple_built {
            self.ple_fill(dg.token.as_slice())?;
        }

        let mut result = Ok(());
        for si in 0..self.states.len() {
            if si > 0 {
                result = self.handoff(si, 1);
                if result.is_err() { break; }
            }
            device::set_device(self.states[si].device)?;
            if dg.graphs[si].is_none() {
                // SAFETY of the capture: everything queue_shard touches lives
                // in `self` or in `dg`, both of which outlive the graph.
                airun_cuda::Graph::begin_capture(&self.states[si].stream)?;
                let queued = self.queue_shard(si, dg.token.as_slice(), dg.position.as_slice(), true, true, false);
                let ended = airun_cuda::Graph::end_capture(&self.states[si].stream);
                match queued.and_then(|_| Ok(ended?)) {
                    Ok(g) => dg.graphs[si] = Some(g),
                    Err(e) => { result = Err(e); break; }
                }
            }
            result = dg.graphs[si].as_ref().expect("graph was just built")
                .launch(&self.states[si].stream).map_err(Into::into);
            if result.is_err() { break; }
        }
        self.decode_graph = Some(dg);
        result?;
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
        // The previous batch's PLE upload may still be queued behind a shard
        // that runs asynchronously (see handoff); its staging is refilled
        // below. (The decode graph path synchronizes every stream instead.)
        for st in &self.states {
            if let Some(ps) = st.hc.as_ref().and_then(|h| h.ple.as_ref()) {
                ps.uploaded.synchronize()?;
            }
        }
        for si in 0..self.states.len() {
            if si > 0 {
                self.handoff(si, token_ids.len())?;
            }
            self.queue_shard(si, token_ids, positions, upload, true, true)?;
        }
        Ok(())
    }

    /// Hands the first `n` rows of the residual from shard `si - 1` to shard
    /// `si`, after the earlier shard's work, without blocking the host.
    ///
    /// The receiving stream waits on an event for the sender's rows, and the
    /// sender's stream waits for the copy before it can overwrite them with
    /// the next batch. So a prefill pipelines: the first GPU runs chunk c+1
    /// while the second runs chunk c. The handoff used to synchronize the
    /// sender from the host, which ran the GPUs strictly in turn: split over
    /// a 5070 Ti and a 5060 Ti (half the SMs), a 4k prompt fell from 3653 to
    /// 2501 tok/s where llama.cpp's pipelined split kept 3514.
    fn handoff(&self, si: usize, n: usize) -> Result<()> {
        let (src, dst) = (&self.states[si - 1], &self.states[si]);
        device::set_device(src.device)?;
        src.sent.record(&src.stream)?;
        device::set_device(dst.device)?;
        dst.stream.wait_for(&src.sent)?;
        let width = self.model.config.hc.map_or(1, |h| h.count) * self.model.config.n_embd;
        airun_cuda::copy_peer(dst.x.as_mut_ptr(), dst.device, src.x.as_ptr(), src.device,
            n * width * std::mem::size_of::<f32>(), &dst.stream)?;
        dst.copied.record(&dst.stream)?;
        device::set_device(src.device)?;
        src.stream.wait_for(&dst.copied)?;
        device::set_device(dst.device)?;
        Ok(())
    }

    /// One shard's part of the forward pass on its stream: the embedding on
    /// the first, the layers, the final norm and logits on the last. With
    /// `fill_ple` the PLE rows are hashed and gathered here; a graph replay
    /// does that beforehand (see forward_decode_graphed).
    fn queue_shard(&mut self, si: usize, token_ids: &[i32], positions: &[i32], upload: bool, fill_ple: bool, record_ple_event: bool)
        -> Result<()> {
        let cfg = self.model.config.clone();
        let n = token_ids.len();
        device::set_device(self.states[si].device)?;
        crate::cpu::set_pass(n);

        // --- embedding, on the shard that holds the table ---
        if si == 0 {
            let st = &self.states[0];
            if upload {
                st.tokens.copy_from_host_at(0, token_ids, &st.stream)?;
                st.positions.copy_from_host_at(0, positions, &st.stream)?;
            }
            let embd = &self.model.token_embd;
            // With hyper-connections the embedding lands in xb and is copied
            // into every stream.
            let dst = if cfg.hc.is_some() { st.xb.as_mut_ptr() } else { st.x.as_mut_ptr() };
            if !self.embeds_ready { ops::embed(
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
            // An NVFP4 table with a global-scale sidecar: the gathered rows
            // are its stored magnitude until scaled.
            if embd.global_scale != 1.0 {
                ops::scale(dst, embd.global_scale, n * cfg.n_embd, &st.stream)?;
            } }
            if let Some(h) = &cfg.hc {
                hc::hc_expand(st.x.as_mut_ptr(), st.xb.as_ptr(), n, h.count, cfg.n_embd, &st.stream)?;
            }
        } else if upload {
            let st = &self.states[si];
            st.positions.copy_from_host_at(0, positions, &st.stream)?;
        }
        if cfg.ple.as_ref().is_some_and(|p| self.states[si].layers.contains(&p.layer)) {
            if fill_ple {
                self.ple_fill(token_ids)?;
            }
            self.ple_upload(si, n, record_ple_event)?;
        }

        for local in 0..self.states[si].layers.len() {
            let layer = self.states[si].layers[local];
            // A layer kept in RAM (--alloc): copy its weights into the staging
            // VRAM, in stream order, so they land after the previous such
            // layer is done with it. In a captured decode graph these are copy
            // nodes from the same pinned addresses.
            for &(host, dev, bytes) in self.model.layers[layer].ram_copies.iter().filter(|_| !crate::cpu::pass_on_cpu()) {
                airun_cuda::copy_h2d(dev as *mut std::ffi::c_void, host as *const std::ffi::c_void, bytes,
                    &self.states[si].stream)?;
            }
            self.run_layer(si, local, layer, n, &cfg)?;
        }
        if si + 1 < self.states.len() {
            return Ok(());
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
            out_w.ptr(),
            last_row,
            logits.as_mut_ptr(),
            cfg.n_embd,
            cfg.vocab_size,
            1,
            out_w.row_bytes,
            &st.stream,
        )?;
        if out_w.global_scale != 1.0 {
            ops::scale(logits.as_mut_ptr(), out_w.global_scale, cfg.vocab_size, &st.stream)?;
        }

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
    /// The RoPE parameters shard `si` runs with: the model's, positioned by
    /// the shard's shift and, during an image chunk, its 3D positions.
    fn rope_for(&self, si: usize, cfg: &Config) -> ops::RopeParams {
        let st = &self.states[si];
        cfg.rope.positioned(st.rope_shift.as_ptr() as u64, self.image_rows.map(|n| (st.mpos.as_ptr() as u64, n)))
    }

    fn set_rope_shift(&mut self, shift: i32) -> Result<()> {
        self.rope_shift = shift;
        for st in &mut self.states {
            device::set_device(st.device)?;
            st.rope_shift.copy_from_host(&[shift], &st.stream)?;
        }
        Ok(())
    }

    /// Whether this model can take image embeddings: its RoPE has the M-RoPE
    /// sections that place an image's rows and columns.
    pub fn takes_images(&self) -> bool {
        self.model.config.rope.sections.iter().any(|&s| s > 0)
    }

    /// Registers an image's embeddings (`nx * ny` rows of `n_embd`, row-major
    /// over the merged patch grid) and returns the sentinel ids a prompt
    /// carries in its place, between the model's vision start and end tokens.
    /// The ids come from the embeddings' hash, so the same image in the next
    /// turn of a chat is recognized in the cache and not run again.
    pub fn register_image(&mut self, embeds: Vec<f32>, nx: usize, ny: usize) -> Result<Vec<u32>> {
        let n_embd = self.model.config.n_embd;
        if embeds.len() != nx * ny * n_embd {
            return Err(CoreError::Other(format!(
                "image embeddings of {} floats do not make {nx} x {ny} rows of {n_embd}; the mmproj file does not \
                 belong to this model (its projection width must equal the model's)", embeds.len())));
        }
        if !self.takes_images() {
            return Err(CoreError::Other(format!(
                "{} has no M-RoPE sections (rope.dimension_sections), so it cannot place image tokens; use the \
                 vision model's own GGUF", self.model.config.arch_name)));
        }
        let mut h: u64 = 0xcbf29ce484222325;
        for v in &embeds {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        let ids: Vec<u32> = (0..(nx * ny) as u64)
            .map(|i| IMAGE_TOKEN_BASE | ((h ^ i.wrapping_mul(0x9E3779B97F4A7C15)) as u32 & !IMAGE_TOKEN_BASE))
            .collect();
        if !self.images.iter().any(|im| im.ids == ids) {
            self.images.push(std::sync::Arc::new(ImageInput { ids: ids.clone(), embeds, nx, ny }));
            // Keep the recent ones: a chat names the same few images turn after turn.
            if self.images.len() > 16 {
                self.images.remove(0);
            }
        }
        Ok(ids)
    }

    /// Whether the image these ids stand for is still registered.
    pub fn has_image(&self, ids: &[u32]) -> bool {
        self.images.iter().any(|im| im.ids == ids)
    }

    /// Runs `prompt[from..]` up to the end of its last image (text in chunks,
    /// images through [`Self::prefill_image`]) and returns where it stopped;
    /// `from` itself when the rest has no image.
    fn prefill_images(&mut self, prompt: &[u32], from: usize) -> Result<usize> {
        let Some(last) = prompt.iter().rposition(|&t| is_image_token(t)) else { return Ok(from) };
        let end = last + 1;
        let mut done = from;
        let start = std::time::Instant::now();
        while done < end {
            if is_image_token(prompt[done]) {
                let run = prompt[done..].iter().take_while(|&&t| is_image_token(t)).count();
                let ids = &prompt[done..done + run];
                let img = self.images.iter().find(|im| im.ids.as_slice() == ids).cloned().ok_or_else(||
                    CoreError::Other("the prompt names an image that is not registered (register_image)".into()))?;
                self.prefill_image(&img)?;
                done += run;
                self.prefill_report(done, prompt.len(), from, start);
            } else {
                let text = prompt[done..end].iter().take_while(|&&t| !is_image_token(t)).count();
                let mut i = 0;
                while i < text {
                    let take = (text - i).min(self.max_batch);
                    self.step(&prompt[done + i..done + i + take])?;
                    i += take;
                    self.prefill_report(done + i, prompt.len(), from, start);
                }
                done += text;
            }
        }
        Ok(done)
    }

    /// Where a cached prefix may end: not inside an image.
    fn image_safe_prefix(prompt: &[u32], cached: usize) -> usize {
        let mut c = cached;
        while c > 0 && c < prompt.len() && is_image_token(prompt[c]) && is_image_token(prompt[c - 1]) {
            c -= 1;
        }
        c
    }

    /// Runs an image's embeddings at the next cache slots. Its tokens rotate
    /// by (t, t + row, t + col), t the next text position, as Qwen-VL places
    /// them; the text after it continues at t + max(rows, cols).
    fn prefill_image(&mut self, img: &ImageInput) -> Result<()> {
        let n = img.nx * img.ny;
        let n_embd = self.model.config.n_embd;
        if self.n_past + n > self.ctx {
            return Err(CoreError::Other(format!(
                "the image's {n} tokens do not fit: the conversation would reach {} of a {} context; raise \
                 --ctx-size or send a smaller image", self.n_past + n, self.ctx)));
        }
        if let Some(sp) = self.spec.as_mut() {
            sp.synced = false;
        }
        let start_slot = self.n_past;
        let before = self.rope_shift;
        let t0 = self.n_past as i32 - self.rope_shift;
        let mut i0 = 0;
        while i0 < n {
            let len = (n - i0).min(self.max_batch);
            let mut mpos = vec![0i32; 3 * len];
            for i in 0..len {
                let g = i0 + i;
                mpos[i] = t0;
                mpos[len + i] = t0 + (g / img.nx) as i32;
                mpos[2 * len + i] = t0 + (g % img.nx) as i32;
            }
            for st in &self.states {
                device::set_device(st.device)?;
                st.mpos.copy_from_host_at(0, &mpos, &st.stream)?;
            }
            {
                let st = &self.states[0];
                device::set_device(st.device)?;
                let dst = if self.model.config.hc.is_some() { &st.xb } else { &st.x };
                dst.copy_from_host_at(0, &img.embeds[i0 * n_embd..(i0 + len) * n_embd], &st.stream)?;
            }
            let positions: Vec<i32> = (0..len).map(|i| (self.n_past + i) as i32).collect();
            let ids = vec![0i32; len];
            self.embeds_ready = true;
            self.image_rows = Some(len);
            let r = self.queue_forward(&ids, &positions);
            self.embeds_ready = false;
            self.image_rows = None;
            r?;
            self.n_past += len;
            self.history.extend_from_slice(&img.ids[i0..i0 + len]);
            self.rebalance_experts()?;
            i0 += len;
        }
        self.image_spans.push((start_slot, n, before));
        self.set_rope_shift(before + n as i32 - img.nx.max(img.ny) as i32)
    }

    // --- slots: several conversations, each with its own caches ---------------

    /// Keeps `n` conversations' caches side by side (llama-server's
    /// --parallel). Each slot gets its own KV cache and recurrent state at the
    /// full context; the planner must have counted them (Config::slots).
    /// [`Self::decode_rows`] then advances several slots in one forward pass.
    /// Each slot keeps its own MTP head cache too (swapped in with its
    /// states), so a slot's head stays good across switches and batched steps
    /// draft as well (spec::verify_rows).
    pub fn set_slots(&mut self, n: usize) -> Result<()> {
        let n = n.max(1);
        let cfg = &self.model.config;
        if n > 1 && (cfg.indexer.is_some() || cfg.hc.is_some() || cfg.ple.is_some()) {
            return Err(CoreError::Other(format!(
                "{} keeps state (sparse-attention index, hyper-connections or n-gram tables) that airun does not \
                 hold per conversation yet; run it with --parallel 1", cfg.arch_name)));
        }
        if n > self.max_batch {
            return Err(CoreError::Other(format!(
                "--parallel {n} exceeds the prompt batch of {}; raise --batch-size", self.max_batch)));
        }
        while self.slots.len() < n {
            let mut sets = Vec::with_capacity(self.states.len());
            for si in 0..self.states.len() {
                sets.push(self.fresh_states(si)?);
            }
            self.slots.push(SlotStore { states: sets, ..Default::default() });
        }
        self.slots.truncate(n);
        // Their graphs held the dropped or new slots' buffers.
        self.mixed_graphs = None;
        for st in &mut self.states {
            if n > 1 {
                device::set_device(st.device)?;
                let cfg = &self.model.config;
                st.rows_tab = Some(DeviceBuffer::zeroed(st.device, st.layers.len().max(1) * 2 * n, &st.stream)?);
                st.rows_partial = Some(DeviceBuffer::new(st.device,
                    n * cfg.n_head * ops::attention_splits(self.ctx) * (cfg.head_dim + 2))?);
            } else {
                st.rows_tab = None;
                st.rows_partial = None;
            }
        }
        Ok(())
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn active_slot(&self) -> usize {
        self.active_slot
    }

    /// The tokens slot `s` holds.
    pub fn slot_history(&self, s: usize) -> &[u32] {
        if s == self.active_slot { &self.history } else { &self.slots[s].history }
    }

    /// How much of `tokens` slot `s` already holds (see [`Self::reusable_prefix`]).
    pub fn slot_prefix(&self, s: usize, tokens: &[u32]) -> usize {
        if s == self.active_slot {
            self.reusable_prefix(tokens)
        } else {
            self.prefix_of(&self.slots[s].history, self.slots[s].turn_snap.as_ref(), tokens)
        }
    }

    /// Zeroed layer states for shard `si`, as the runtime allocates them.
    fn fresh_states(&self, si: usize) -> Result<Vec<LayerState>> {
        let cfg = &self.model.config;
        let st = &self.states[si];
        device::set_device(st.device)?;
        let mut out = Vec::with_capacity(st.layers.len());
        for &l in &st.layers {
            out.push(match cfg.layer_kind(l) {
                LayerKind::Attention => {
                    let bytes = cfg.n_head_kv * self.ctx * cfg.kv_type.row_bytes(cfg.head_dim);
                    let in_ram = self.model.placement.kv_in_ram(l);
                    LayerState::Kv {
                        k: kv_cache(st.device, bytes, in_ram, &st.stream)?,
                        v: kv_cache(st.device, bytes, in_ram, &st.stream)?,
                        qsa: None,
                    }
                }
                LayerKind::LinearAttention => {
                    let ssm = cfg.ssm.as_ref().ok_or_else(|| CoreError::Other("linear layer without an ssm config".into()))?;
                    LayerState::Delta {
                        state: DeviceBuffer::zeroed(st.device, ssm.state_elements(), &st.stream)?,
                        conv: DeviceBuffer::zeroed(st.device, ssm.conv_state_elements(), &st.stream)?,
                    }
                }
            });
        }
        st.stream.synchronize()?;
        Ok(out)
    }

    /// The layer state of slot `slot` for shard `si`'s `local`-th layer.
    fn slot_state(&self, slot: usize, si: usize, local: usize) -> &LayerState {
        if slot == self.active_slot { &self.states[si].states[local] } else { &self.slots[slot].states[si][local] }
    }

    /// Makes slot `s` the one single-sequence calls run on. Its states swap
    /// into the shards; captured decode graphs held the old ones' addresses.
    pub fn switch_slot(&mut self, s: usize) -> Result<()> {
        if s == self.active_slot {
            return Ok(());
        }
        if s >= self.slots.len() {
            return Err(CoreError::Other(format!("slot {s} does not exist ({} slots)", self.slots.len())));
        }
        let a = self.active_slot;
        for si in 0..self.states.len() {
            let cur = std::mem::take(&mut self.states[si].states);
            self.slots[a].states.resize_with(self.states.len(), Vec::new);
            self.slots[a].states[si] = cur;
            self.states[si].states = std::mem::take(&mut self.slots[s].states[si]);
        }
        self.slots[a].n_past = self.n_past;
        self.slots[a].history = std::mem::take(&mut self.history);
        self.slots[a].rope_shift = self.rope_shift;
        self.slots[a].image_spans = std::mem::take(&mut self.image_spans);
        self.slots[a].turn_snap = self.turn_snap.take();
        self.n_past = self.slots[s].n_past;
        self.history = std::mem::take(&mut self.slots[s].history);
        self.image_spans = std::mem::take(&mut self.slots[s].image_spans);
        self.turn_snap = self.slots[s].turn_snap.take();
        let shift = self.slots[s].rope_shift;
        self.active_slot = s;
        self.decode_graph = None;
        self.chain = None;
        if let Some(sp) = self.spec.as_mut() {
            // Each slot keeps its own head cache: swap it in with the states.
            let incoming = self.slots[s].head.take();
            self.slots[a].head = sp.swap_head(incoming);
            // These captured the old slot's delta-net, KV and head addresses.
            sp.vgraph = None;
            sp.vsgraph = None;
            sp.agraphs.clear();
            sp.forget_head_graph();
        }
        self.set_rope_shift(shift)
    }

    /// Prefills slot `s` with `prompt`, reusing what it already holds, and
    /// returns the last token's logits and how many tokens came from cache.
    pub fn slot_begin(&mut self, s: usize, prompt: &[u32]) -> Result<(Vec<f32>, usize)> {
        if prompt.is_empty() {
            return Err(CoreError::Other("an empty prompt has nothing to continue".into()));
        }
        self.switch_slot(s)?;
        let mut cached = Self::image_safe_prefix(prompt, self.reusable_prefix(prompt));
        // With drafting on, the MTP head is fed as in generate, so the slot
        // drafts in batched steps (spec::verify_rows). A head that missed
        // part of the cached prefix makes a short one worth prefilling again;
        // a long one goes on without it. No head through images.
        let mut head = self.speculates() && !prompt.iter().any(|&t| is_image_token(t));
        if head && cached > 0 {
            let valid = if cached < self.history.len() { self.snapshot_has_head() } else { self.head_synced(s) };
            if !valid {
                if cached <= 256 { cached = 0 } else { head = false }
            }
        }
        self.rewind_to(cached);
        let cached = self.n_past.min(cached);
        self.set_head_synced(s, false);
        if head && cached == 0 {
            self.head_reset()?;
        }
        let start = std::time::Instant::now();
        let mut done = self.prefill_images(prompt, cached)?;
        let mut logits = Vec::new();
        // As generate: the state before the last header, for the next turn.
        let snap_at = self.snapshot_point(prompt, done);
        self.prefill_report(done, prompt.len(), cached, start);
        while done < prompt.len() {
            let mut take = (prompt.len() - done).min(self.max_batch);
            if let Some(p) = snap_at.filter(|&p| p > done) {
                take = take.min(p - done);
            }
            let chunk = &prompt[done..done + take];
            let pos0 = self.n_past;
            done += take;
            if done < prompt.len() {
                self.step(chunk)?;
            } else {
                logits = self.forward_pinned(chunk)?.to_vec();
            }
            if head {
                self.head_prefill_chunk(chunk, pos0)?;
            }
            if snap_at == Some(done) {
                self.take_snapshot(head)?;
            }
            self.prefill_report(done, prompt.len(), cached, start);
        }
        self.set_head_synced(s, head);
        Ok((logits, cached))
    }

    /// One decode step for several conversations at once: row b feeds token
    /// `rows[b].1` to slot `rows[b].0` (distinct slots) and gets that slot's
    /// next-token logits, `vocab_size` floats a row. The matmuls read each
    /// weight once for all the rows -- what makes serving several requests
    /// faster than taking them in turn.
    pub fn decode_rows(&mut self, rows: &[(usize, u32)]) -> Result<&mut [f32]> {
        if rows.len() == 1 {
            self.switch_slot(rows[0].0)?;
            return self.forward_pinned(&[rows[0].1]);
        }
        let n = rows.len();
        if n > self.max_batch {
            return Err(CoreError::Other(format!("{n} rows exceed the batch of {}", self.max_batch)));
        }
        let mut seen = std::collections::HashSet::new();
        let mut positions = Vec::with_capacity(n);
        let mut rope = vec![0i32; 3 * n];
        for (b, &(slot, _)) in rows.iter().enumerate() {
            if slot >= self.slots.len() || !seen.insert(slot) {
                return Err(CoreError::Other(format!("decode_rows needs distinct, existing slots (row {b})")));
            }
            let (past, shift) = if slot == self.active_slot { (self.n_past, self.rope_shift) }
                else { (self.slots[slot].n_past, self.slots[slot].rope_shift) };
            if past + 1 > self.ctx {
                return Err(CoreError::Other(format!(
                    "slot {slot} is at its {} context; raise --ctx-size or start a new conversation", self.ctx)));
            }
            positions.push(past as i32);
            for a in 0..3 {
                rope[a * n + b] = past as i32 - shift;
            }
        }
        for st in &self.states {
            device::set_device(st.device)?;
            st.mpos.copy_from_host_at(0, &rope, &st.stream)?;
        }
        for &(slot, _) in rows {
            self.set_head_synced(slot, false);
        }
        let ids: Vec<i32> = rows.iter().map(|&(_, t)| t as i32).collect();
        self.fill_rows_tab(rows)?;
        self.mixed = Some(rows.iter().map(|&(s, _)| s).collect());
        self.image_rows = Some(n);
        self.skip_logits = true;
        let graphed = !graphs_disabled() && !mixed_graphs_disabled() && !self.mixed_graph_off
            && !self.states.iter().any(|s| s.moe.as_ref().is_some_and(|m| m.upload.is_some()));
        let queued = if graphed {
            let mut launched = false;
            match self.queue_rows_graphed(&ids, &positions, &mut launched) {
                // Nothing ran: the first shard did not capture.
                Err(e) if !launched => {
                    eprintln!("note: batched decode runs without CUDA graphs: {e}");
                    self.mixed_graph_off = true;
                    self.queue_forward(&ids, &positions)
                }
                r => r,
            }
        } else {
            self.queue_forward(&ids, &positions)
        };
        self.mixed = None;
        self.image_rows = None;
        self.skip_logits = false;
        queued?;
        for &(slot, tok) in rows {
            if slot == self.active_slot {
                self.n_past += 1;
                self.history.push(tok);
            } else {
                self.slots[slot].n_past += 1;
                self.slots[slot].history.push(tok);
            }
        }
        self.rebalance_experts()?;

        let cfg = &self.model.config;
        let vocab = cfg.vocab_size;
        let n_embd = cfg.n_embd;
        let st = self.states.last().expect("a runtime has at least one device");
        device::set_device(st.device)?;
        let out_w = self.model.output_tensor();
        let need = n * vocab;
        if self.rows_logits.as_ref().is_none_or(|(_, h)| h.len() < need) {
            self.rows_logits = None;
            self.rows_logits = Some((DeviceBuffer::<f32>::new(st.device, need)?, airun_cuda::PinnedBuffer::new(need)?));
        }
        let (dev, host) = self.rows_logits.as_mut().expect("allocated above");
        // The matvec takes up to 8 rows at once (each weight read once for them).
        const ROWS: usize = 8;
        for r0 in (0..n).step_by(ROWS) {
            let m = ROWS.min(n - r0);
            let out = dev.offset_ptr(r0 * vocab);
            ops::matvec(out_w.kernel_type(), out_w.ptr(), st.xb.offset_ptr(r0 * n_embd), out,
                n_embd, vocab, m, out_w.row_bytes, &st.stream)?;
            if out_w.global_scale != 1.0 {
                ops::scale(out, out_w.global_scale, m * vocab, &st.stream)?;
            }
        }
        dev.copy_to_host(&mut host.as_mut_slice()[..need], &st.stream)?;
        st.stream.synchronize()?;
        Ok(&mut host.as_mut_slice()[..need])
    }

    /// Writes each shard's `rows_tab` for these rows (see the field). Queued on
    /// the shard's stream ahead of the step, which the previous step's
    /// logits read has already waited out.
    fn fill_rows_tab(&self, rows: &[(usize, u32)]) -> Result<()> {
        let n = rows.len();
        let stride = 2 * self.slots.len();
        for si in 0..self.states.len() {
            let st = &self.states[si];
            let Some(tab) = st.rows_tab.as_ref() else { continue };
            let mut host = vec![0u64; st.layers.len().max(1) * stride];
            for local in 0..st.layers.len() {
                for (b, &(slot, _)) in rows.iter().enumerate() {
                    let (a, c) = match self.slot_state(slot, si, local) {
                        LayerState::Kv { k, v, .. } => (k.as_ptr() as u64, v.as_ptr() as u64),
                        LayerState::Delta { state, conv } => (state.as_ptr() as u64, conv.as_ptr() as u64),
                    };
                    host[local * stride + b] = a;
                    host[local * stride + n + b] = c;
                }
            }
            device::set_device(st.device)?;
            tab.copy_from_host_at(0, &host, &st.stream)?;
        }
        Ok(())
    }

    /// Row `b`'s table entries for shard `si`'s `local`-th layer: the K /
    /// delta-state pointers of the n rows, then the V / conv ones.
    fn rows_tab_at(&self, si: usize, local: usize) -> Option<(*const std::ffi::c_void, *const std::ffi::c_void)> {
        if rows_batched_disabled() {
            return None;
        }
        let n = self.mixed.as_ref()?.len();
        let tab = self.states[si].rows_tab.as_ref()?;
        let base = local * 2 * self.slots.len();
        Some((tab.offset_ptr(base) as *const _, tab.offset_ptr(base + n) as *const _))
    }

    /// [`Self::queue_forward`] for decode_rows (`self.mixed` set) through a
    /// graph per shard captured for these slots. `launched` tells whether
    /// any work was queued (when not, an error can be retried uncaptured).
    fn queue_rows_graphed(&mut self, ids: &[i32], positions: &[i32], launched: &mut bool) -> Result<()> {
        // A batched verify (several tokens a row) is a graph of its own.
        let mut key = self.mixed.clone().expect("decode_rows sets the rows");
        if self.mixed_t > 1 {
            key.push(usize::MAX - self.mixed_t);
        }
        let n = ids.len();
        let mut mg = match self.mixed_graphs.take() {
            Some(mg) => mg,
            None => MixedGraphs {
                entries: Vec::new(),
                tokens: airun_cuda::PinnedBuffer::new(self.max_batch)?,
                positions: airun_cuda::PinnedBuffer::new(self.max_batch)?,
            },
        };
        // decode_rows waited for the previous replay (its logits), so the
        // staging it copied from is free.
        mg.tokens.as_mut_slice()[..n].copy_from_slice(ids);
        mg.positions.as_mut_slice()[..n].copy_from_slice(positions);
        let at = match mg.entries.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                if mg.entries.len() >= MIXED_GRAPHS {
                    mg.entries.remove(0);
                }
                mg.entries.push((key, (0..self.states.len()).map(|_| None).collect()));
                mg.entries.len() - 1
            }
        };
        let mut result = Ok(());
        for si in 0..self.states.len() {
            if si > 0 {
                result = self.handoff(si, n);
                if result.is_err() { break; }
            }
            device::set_device(self.states[si].device)?;
            if mg.entries[at].1[si].is_none() {
                // Everything queue_shard touches lives in `self` or `mg`.
                airun_cuda::Graph::begin_capture(&self.states[si].stream)?;
                let queued = self.queue_shard(si, &mg.tokens.as_slice()[..n], &mg.positions.as_slice()[..n],
                    true, true, false);
                let ended = airun_cuda::Graph::end_capture(&self.states[si].stream);
                match queued.and_then(|_| Ok(ended?)) {
                    Ok(g) => mg.entries[at].1[si] = Some(g),
                    Err(e) => {
                        mg.entries.remove(at);
                        self.mixed_graph_off = true;
                        result = Err(e);
                        break;
                    }
                }
            }
            *launched = true;
            result = mg.entries[at].1[si].as_ref().expect("graph was just built")
                .launch(&self.states[si].stream).map_err(Into::into);
            if result.is_err() { break; }
        }
        self.mixed_graphs = Some(mg);
        result
    }

    pub fn reusable_prefix(&self, tokens: &[u32]) -> usize {
        self.prefix_of(&self.history, self.turn_snap.as_ref(), tokens)
    }

    /// How much of `tokens` a sequence holding `history` (and `snap`) reuses.
    fn prefix_of(&self, history: &[u32], snap: Option<&TurnSnapshot>, tokens: &[u32]) -> usize {
        let limit = tokens.len().saturating_sub(1);
        let common = history
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            .min(limit);

        if self.model.config.is_hybrid() && common < history.len() {
            // The recurrent state is the whole history's; it can only go back
            // to the turn snapshot, when the prompt still agrees up to it.
            return snap
                .filter(|s| s.tokens.len() <= common && history[..s.tokens.len()] == s.tokens[..])
                .map_or(0, |s| s.tokens.len());
        }
        common
    }

    /// Drops the cache cursor back to `n`, discarding what came after.
    ///
    /// Only ever called with a value [`reusable_prefix`] returned, which is what
    /// makes it safe on hybrids: there it is the full history, the turn
    /// snapshot's point (whose recurrent state is put back here) or zero.
    fn rewind_to(&mut self, n: usize) {
        if n == 0 {
            self.reset();
            return;
        }
        if self.model.config.is_hybrid() && n < self.history.len() {
            if let Err(e) = self.restore_snapshot(n) {
                eprintln!("warning: could not restore the turn snapshot ({e}); reprocessing the prompt");
                self.reset();
                return;
            }
        }
        // An image is all or nothing: a cut inside one drops it whole.
        let mut n = n;
        while let Some(&(start, len, before)) = self.image_spans.last() {
            if start + len <= n {
                break;
            }
            n = n.min(start);
            self.image_spans.pop();
            let _ = self.set_rope_shift(before);
        }
        if n == 0 {
            self.reset();
            return;
        }
        self.n_past = n;
        self.history.truncate(n);
    }

    /// Where a prefill of `prompt` from `from` stops to take the turn
    /// snapshot: just before the last `<|im_start|>`, if it lies ahead.
    pub(crate) fn snapshot_point(&self, prompt: &[u32], from: usize) -> Option<usize> {
        if !self.model.config.is_hybrid() || snapshot_disabled() || !self.image_spans.is_empty()
            || prompt.iter().any(|&t| is_image_token(t)) {
            return None;
        }
        let start = self.tokenizer.id_of("<|im_start|>")?;
        let p = prompt.iter().rposition(|&t| t == start)?;
        (p > from).then_some(p)
    }

    /// Saves the recurrent state the caches hold now (n_past tokens), and the
    /// MTP head's `pending` row when `with_head`.
    pub(crate) fn take_snapshot(&mut self, with_head: bool) -> Result<()> {
        let mut total = 0usize;
        self.walk_state(false, |buf, offset, len, _| { total += buf.raw(offset, len).1; Ok(()) })?;
        let pending_bytes = self.model.config.n_embd * std::mem::size_of::<f32>();
        let with_head = with_head && self.spec.is_some();
        let need = total + if with_head { pending_bytes } else { 0 };
        // The previous snapshot's buffer, if it is big enough. Its last
        // restore was queued on the streams that queue the copies below.
        let mut bytes = match self.turn_snap.take() {
            Some(s) if s.bytes.len() >= need => s.bytes,
            _ => airun_cuda::PinnedBuffer::new(total + pending_bytes)?,
        };
        let base = bytes.as_raw();
        let mut at = 0usize;
        self.walk_state(false, |buf, offset, len, stream| {
            let (src, n) = buf.raw(offset, len);
            airun_cuda::copy_d2h(base.wrapping_byte_add(at), src, n, stream)?;
            at += n;
            Ok(())
        })?;
        if with_head {
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            let sp = self.spec.as_ref().expect("checked above");
            airun_cuda::copy_d2h(base.wrapping_byte_add(at), sp.pending.as_ptr(), pending_bytes, &st.stream)?;
        }
        self.turn_snap = Some(TurnSnapshot { tokens: self.history[..self.n_past].to_vec(), bytes, pending: with_head });
        Ok(())
    }

    /// Puts back the snapshot's recurrent state, `n` being its length. The
    /// copies are queued on each shard's stream, ahead of the prefill that
    /// follows.
    fn restore_snapshot(&mut self, n: usize) -> Result<()> {
        let snap = self.turn_snap.take().filter(|s| s.tokens.len() == n).ok_or_else(|| CoreError::Other(
            "no turn snapshot at that point; this is a bug".into()))?;
        let base = snap.bytes.as_raw();
        let mut at = 0usize;
        let loaded = self.walk_state(false, |buf, offset, len, stream| {
            let (dst, bytes) = buf.raw(offset, len);
            airun_cuda::copy_h2d(dst, base.wrapping_byte_add(at), bytes, stream)?;
            at += bytes;
            Ok(())
        });
        let head = snap.pending && self.spec.is_some();
        if head {
            let st = self.states.last().expect("a runtime has at least one device");
            device::set_device(st.device)?;
            let sp = self.spec.as_ref().expect("checked above");
            airun_cuda::copy_h2d(sp.pending.as_mut_ptr(), base.wrapping_byte_add(at),
                self.model.config.n_embd * std::mem::size_of::<f32>(), &st.stream)?;
        }
        if let Some(sp) = self.spec.as_mut() {
            // The head's cache rows before n were written with these tokens.
            sp.synced = head;
        }
        self.turn_snap = Some(snap);
        loaded
    }

    /// Whether the turn snapshot carries the MTP head's state.
    pub(crate) fn snapshot_has_head(&self) -> bool {
        self.turn_snap.as_ref().is_some_and(|s| s.pending)
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
                Block::Attention(a) => self.run_attention(si, state, layer, a, n, cfg)?,
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
                FfnWeights::Dense { gate, up, down } if on_cpu(gate) && on_cpu(up) && on_cpu(down)
                    && gate.k == up.k && gate.rows == up.rows && down.k == gate.rows => {
                    cpu_ffn(gate, up, down, &st.xb, &st.xb2, n, &st.q8act, stream)?;
                }
                FfnWeights::Dense { gate, up, down } => {
                    let mut down_done = false;
                    // Prefill takes gate, up and SwiGLU as one integer tensor-core
                    // launch where it can (see matvec_mmq_async_kernel's GLU).
                    // The fused launches write silu(Wg·x)⊙(Wu·x) directly, so they
                    // cannot apply a per-tensor scale afterwards (silu is not
                    // scale-linear). Only fuse when neither projection carries a
                    // global scale; otherwise the separate path scales gate and
                    // up before the SwiGLU.
                    // Matrices on the CPU take the plain path (each through matvec).
                    let no_scale = gate.global_scale == 1.0 && up.global_scale == 1.0 && !on_cpu(gate);
                    let glu = n > 1 && no_scale && !matvec_f32_forced() && same_shape(&[gate, up])
                        && !fp4_prefill(gate, n, &st.q8act)
                        && ops::matvec_supports_q8(gate.kernel_type())
                        && ops::mmq_glu_supported(gate.kernel_type(), gate.ptr(),
                            up.ptr(), st.q8act.buf.as_ptr(), gate.k, gate.row_bytes, n);
                    if n == 1 && no_scale && same_shape(&[gate, up]) && !q8_decode(gate) {
                        // Decode: gate, up and SwiGLU in one launch, bit-identical.
                        ops::matvec_glu(gate.kernel_type(), gate.ptr(), up.ptr(),
                            st.xb.as_ptr(), st.gate.as_mut_ptr(), gate.k, gate.rows, gate.row_bytes,
                            stream)?;
                    } else if glu {
                        ops::quantize_act(st.xb.as_ptr(), st.q8act.buf.as_mut_ptr(), n * gate.k, stream)?;
                        ops::matvec_q8_glu(gate.kernel_type(), gate.ptr(), up.ptr(),
                            st.q8act.buf.as_ptr(), st.gate.as_mut_ptr(), gate.k, gate.rows, n,
                            gate.row_bytes, stream)?;
                    } else if no_scale && same_shape(&[gate, up]) && fp4_prefill(gate, n, &st.q8act)
                        && fp4_prefill(down, n, &st.q8act) && !fp4_glu_disabled() {
                        // Gate, up and SwiGLU in one FP4 launch (silu(g) * u in
                        // its epilogue); the down projection quantizes that.
                        let buf = st.q8act.fp4.as_ref().expect("checked by fp4_prefill");
                        st.q8act.valid.set(false);
                        st.q8act.fp4_valid.set(false);
                        ops::quantize_act_nvfp4(st.xb.as_ptr(), buf.as_mut_ptr(), n, gate.k, stream)?;
                        ops::matmul_nvfp4_glu(gate.ptr(), up.ptr(), gate.row_bytes, buf.as_ptr(),
                            st.gate.as_mut_ptr(), gate.k, gate.rows, n, stream)?;
                        matvec(down, &st.gate, &st.xb2, n, &st.q8act, stream)?;
                        down_done = true;
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
                    .run_moe(si, layer, router, gate_exps, up_exps, down_exps, shared.as_ref(), n, cfg)?,
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
            Block::Attention(a) => self.run_attention(si, state, layer, a, n, cfg)?,
            Block::Linear(l) => self.run_linear_attention(si, state, stash, layer, l, n, cfg)?,
        }
        trace_block(st, layer, "mixer", n, cfg)?;
        hc::hc_post(st.x.as_mut_ptr(), st.xb2.as_ptr(), h.inject.as_ptr(), n, streams, cfg.n_embd, stream)?;

        self.hc_mix(si, mix_ffn, n, cfg)?;
        match &w.ffn {
            FfnWeights::Moe { router, gate_exps, up_exps, down_exps, shared } =>
                self.run_moe(si, layer, router, gate_exps, up_exps, down_exps, shared.as_ref(), n, cfg)?,
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
    /// staged by [`Self::ple_fill`].
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
    /// and the history) into PLE table rows and decodes them on the host into
    /// the PLE layer's pinned staging.
    fn ple_fill(&self, token_ids: &[i32]) -> Result<()> {
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
        let mut host = ps.emb_host.borrow_mut();
        let t = std::time::Instant::now();
        table.gather(&rows, &mut host.as_mut_slice()[..rows.len() * pc.head_dim])?;
        if std::env::var("AIRUN_PLE_TIMING").is_ok() && rows.len() > 64 {
            eprintln!("PLE gather: {} rows in {:.1} ms", rows.len(), t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(())
    }

    /// Queues the upload of `n` staged PLE rows on shard `si`.
    fn ple_upload(&self, si: usize, n: usize, record_event: bool) -> Result<()> {
        let (Some(pc), st) = (&self.model.config.ple, &self.states[si]) else { return Ok(()) };
        let ps = st.hc.as_ref().and_then(|h| h.ple.as_ref())
            .ok_or_else(|| CoreError::Other("the PLE layer's GPU has no PLE scratch".into()))?;
        let host = ps.emb_host.borrow();
        ps.emb.copy_from_host_at(0, &host.as_slice()[..n * pc.n_heads() * pc.head_dim], &st.stream)?;
        // The staging guard is a CUDA event the batch path host-synchronizes
        // on before refilling the pinned rows. Recording it inside the decode
        // graph capture leaves the event graph-owned, and the next batch
        // prefill's cudaEventSynchronize on it faults the whole context (a
        // sticky cudaErrorInvalidValue that kills every later request). The
        // graph path guards the pinned buffer with a full stream sync instead,
        // so it skips the record.
        if record_event {
            ps.uploaded.record(&st.stream)?;
        }
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

    /// The QSA indexer of one sparse attention layer for the batch's n tokens:
    /// updates the layer's key cache, then selects the cells every query sees
    /// into `qs.list` (one token) or `qs.mask` (a batch).
    #[allow(clippy::too_many_arguments)]
    fn qsa_select(
        &self,
        st: &DeviceState,
        qc: &QsaCache,
        iw: &crate::model::IndexerWeights,
        qs: &QsaScratch,
        ix: &crate::config::IndexerConfig,
        n: usize,
        rope_dim: usize,
        cfg: &Config,
    ) -> Result<()> {
        let stream = &st.stream;
        let pos = st.positions.as_ptr();
        let width = ix.width(qc.ratio);
        // The indexer reads the same normalized input as q, k and v.
        matvec_reuse(&iw.q, &st.xb, &qs.q, n, &st.q8act, stream)?;
        matvec_reuse(&iw.k, &st.xb, &qs.k, n, &st.q8act, stream)?;
        ops::rmsnorm(qs.q.as_mut_ptr(), qs.q.as_ptr(), iw.q_norm.as_ptr(), ix.dim, n * ix.heads, cfg.rms_eps,
            stream)?;
        ops::rope(qs.q.as_mut_ptr(), pos, n, ix.heads, ix.dim, rope_dim, cfg.rope, false, stream)?;
        // Raw keys wait in the ring; blocks they complete are pooled, normed
        // and rotated -- the order llama.cpp applies them in.
        qsa::store(qc.ring.as_mut_ptr(), qs.k.as_ptr(), pos, n, ix.dim, qc.ring_len, stream)?;
        qsa::pool(qc.pooled.as_mut_ptr(), qc.ring.as_ptr(), iw.k_norm.as_ptr(), pos, n, ix.dim, qc.ratio,
            qc.ring_len, self.ctx / qc.ratio, cfg.rms_eps, rope_dim, cfg.rope, stream)?;
        let max_blocks = self.ctx / qc.ratio;
        for t0 in (0..n).step_by(QSA_GROUP) {
            let g = QSA_GROUP.min(n - t0);
            qsa::score(qs.scores.as_mut_ptr(), qs.max_blocks, qs.q.as_ptr(), qc.pooled.as_ptr(), pos, t0, g,
                ix.heads, ix.dim, qc.ratio, width, max_blocks, stream)?;
            let (list, mask) = if n == 1 {
                (qs.list.as_mut_ptr(), std::ptr::null_mut())
            } else {
                (std::ptr::null_mut(), qs.mask.as_mut_ptr())
            };
            qsa::select(qs.scores.as_ptr(), qs.max_blocks, pos, t0, g, qc.ratio, width, list, mask, qs.words,
                stream)?;
        }
        Ok(())
    }

    /// Softmax attention, optionally with a per-head sigmoid output gate.
    /// For a layer whose KV cache stays in RAM: copies the cells this pass
    /// reads into the shard's stage and returns the stage's (K, V), which the
    /// cache write and the attention then use. A cache in VRAM is returned
    /// as it is.
    fn kv_stage_in(&self, st: &DeviceState, kc: &DeviceBuffer<u8>, vc: &DeviceBuffer<u8>,
                   positions: *const std::ffi::c_void, n: usize)
        -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
        if !kc.is_in_ram() {
            return Ok((kc.as_mut_ptr(), vc.as_mut_ptr()));
        }
        let cfg = &self.model.config;
        let (sk, sv) = st.kv_stage.as_ref().ok_or_else(|| CoreError::Other(
            "a KV cache in RAM on a GPU without a stage; restart with --alloc gpu and report this".into()))?;
        let row = cfg.kv_type.row_bytes(cfg.head_dim);
        for (stage, cache) in [(sk, kc), (sv, vc)] {
            ops::kv_stage(stage.as_mut_ptr(), cache.as_ptr(), positions, n, cfg.n_head_kv, self.ctx, row,
                true, &st.stream)?;
        }
        Ok((sk.as_mut_ptr(), sv.as_mut_ptr()))
    }

    /// Writes the cells this pass stored in the stage back to a KV cache in
    /// RAM (see [`Self::kv_stage_in`]); nothing for one in VRAM.
    fn kv_stage_out(&self, st: &DeviceState, kc: &DeviceBuffer<u8>, vc: &DeviceBuffer<u8>,
                    positions: *const std::ffi::c_void, n: usize) -> Result<()> {
        let Some((sk, sv)) = st.kv_stage.as_ref().filter(|_| kc.is_in_ram()) else { return Ok(()) };
        let cfg = &self.model.config;
        let row = cfg.kv_type.row_bytes(cfg.head_dim);
        for (stage, cache) in [(sk, kc), (sv, vc)] {
            ops::kv_stage(cache.as_mut_ptr(), stage.as_ptr(), positions, n, cfg.n_head_kv, self.ctx, row,
                false, &st.stream)?;
        }
        Ok(())
    }

    fn run_attention(
        &self,
        si: usize,
        kv_state: &LayerState,
        layer: usize,
        a: &crate::model::AttentionWeights,
        n: usize,
        cfg: &Config,
    ) -> Result<()> {
        let st = &self.states[si];
        let stream = &st.stream;

        // Decode runs q, k and v as one launch (same f32 kernel, same result).
        // matvec3 writes q, k and v with no epilogue, so a global scale
        // (NVFP4 sidecar) sends them through matvec, which applies it.
        let fused_qkv = n == 1 && !cfg.attn_output_gate && !on_cpu(&a.q) && same_shape(&[&a.q, &a.k, &a.v])
            && !q8_decode(&a.q) && [&a.q, &a.k, &a.v].iter().all(|t| t.global_scale == 1.0);
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
                [a.q.ptr(), a.k.ptr(), a.v.ptr()],
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

        let LayerState::Kv { k: kc, v: vc, qsa: qsa_cache } = kv_state else {
            return Err(CoreError::Other(
                "an attention layer was given a recurrent state".into(),
            ));
        };

        if let Some(rows) = self.mixed.as_deref() {
            // Rows of `t` tokens, one row per conversation: each row's cache
            // write and attention against its own slot's cache (or, for the
            // head's rows, its slot's head cache), the projections around them
            // batched.
            let t = self.mixed_t;
            let local = if self.mixed_head { 0 } else {
                self.states[si].layers.iter().position(|&l| l == layer)
                    .ok_or_else(|| CoreError::Other("a mixed decode reached a layer its shard does not hold".into()))?
            };
            let null = std::ptr::null::<std::ffi::c_void>();
            let (qw, kw) = (cfg.n_head * cfg.head_dim, cfg.n_head_kv * cfg.head_dim);
            let window = std::env::var("AIRUN_ATTN_WINDOW").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(cfg.sliding_window);
            // Every row's attention in one launch when the kernel takes the shape
            // (and the caches are in VRAM: one in RAM goes through the stage).
            let batched = self.rows_tab_at(si, local)
                .filter(|_| t == 1 && !self.mixed_head)
                .filter(|_| ops::attention_split_rows_ok(cfg.n_head, cfg.n_head_kv, cfg.head_dim))
                .filter(|_| !kc.is_in_ram())
                .zip(st.rows_partial.as_ref());
            for (b, &slot) in rows.iter().enumerate() {
                let state = if self.mixed_head { self.head_kv(slot)? } else { self.slot_state(slot, si, local) };
                let LayerState::Kv { k: kc, v: vc, .. } = state else {
                    return Err(CoreError::Other("an attention layer was given a recurrent state".into()));
                };
                let at = b * t;
                let pos = st.positions.offset_ptr(at);
                let (kp, vp) = self.kv_stage_in(st, kc, vc, pos, t)?;
                // The rotation position comes from this row of mpos (all three
                // axes equal: text), so each slot keeps its own image shift.
                let rope = cfg.rope.positioned(0, Some((st.mpos.as_ptr() as u64 + 4 * at as u64, rows.len() * t)));
                ops::qk_norm_rope_store(
                    st.q.offset_ptr(at * qw), st.k.offset_ptr(at * kw), st.v.offset_ptr(at * kw),
                    a.q_norm.as_ref().map_or(null, |w| w.as_ptr()),
                    a.k_norm.as_ref().map_or(null, |w| w.as_ptr()),
                    kp, vp, pos,
                    t, cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.rms_eps,
                    (!rope_off).then_some((rope_dim, rope, rope_interleaved)),
                    cfg.kv_type, stream,
                )?;
                self.kv_stage_out(st, kc, vc, pos, t)?;
                if t > 1 {
                    // The cells come from `positions` on the device; `keys` only
                    // picks a long-context variant, and a constant keeps a
                    // captured verify valid at every position.
                    let keys = if self.mixed_head { self.mixed_past.get(b).copied().unwrap_or(0) + t } else { self.ctx };
                    ops::attention_masked(st.attn.offset_ptr(at * qw), st.q.offset_ptr(at * qw), kp, vp, pos,
                        t, cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.attn_scale(), window,
                        cfg.kv_type, None, keys, stream)?;
                } else if batched.is_none() {
                    ops::attention_split_sel(st.attn.offset_ptr(at * qw), &st.attn_partial,
                        st.q.offset_ptr(at * qw), kp, vp, pos,
                        cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.attn_scale(),
                        window, ops::attention_splits(self.ctx), cfg.kv_type, std::ptr::null(), stream)?;
                }
            }
            if let Some(((kv, _), partial)) = batched {
                ops::attention_split_rows(st.attn.as_mut_ptr(), partial, st.q.as_ptr(), kv, st.positions.as_ptr(),
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.attn_scale(), window,
                    ops::attention_splits(self.ctx), cfg.kv_type, rows.len(), stream)?;
            }
        } else {
            let (kp, vp) = self.kv_stage_in(st, kc, vc, st.positions.as_ptr(), n)?;
            if !fused_prologue_disabled() {
                // q/k norms, RoPE and the cache write in one launch (see
                // qk_norm_rope_store_kernel); the same arithmetic as the steps below.
                let null = std::ptr::null::<std::ffi::c_void>();
                ops::qk_norm_rope_store(
                    st.q.as_mut_ptr(), st.k.as_ptr(), st.v.as_ptr(),
                    a.q_norm.as_ref().map_or(null, |w| w.as_ptr()),
                    a.k_norm.as_ref().map_or(null, |w| w.as_ptr()),
                    kp, vp, st.positions.as_ptr(),
                    n, cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.rms_eps,
                    (!rope_off).then_some((rope_dim, self.rope_for(si, cfg), rope_interleaved)),
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
                        self.rope_for(si, cfg),
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
                        self.rope_for(si, cfg),
                        rope_interleaved,
                        stream,
                    )?;
                }

                ops::kv_store(
                    kp,
                    vp,
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

            self.kv_stage_out(st, kc, vc, st.positions.as_ptr(), n)?;

            // QSA: the cells each query attends to, as a list for decode and a
            // mask for batches; None attends densely.
            let sparse = match (qsa_cache, &a.indexer, &st.qsa, &cfg.indexer) {
                (Some(qc), Some(iw), Some(qs), Some(ix)) => {
                    self.qsa_select(st, qc, iw, qs, ix, n, cfg.rope_dim, cfg)?;
                    Some(qs)
                }
                _ => None,
            };

            let window = std::env::var("AIRUN_ATTN_WINDOW").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(cfg.sliding_window);
            if n == 1 && std::env::var("AIRUN_NO_SPLIT_ATTN").is_err() {
                ops::attention_split_sel(st.attn.as_mut_ptr(), &st.attn_partial,
                    st.q.as_ptr(), kp, vp, st.positions.as_ptr(),
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim, self.ctx, cfg.attn_scale(),
                    window, ops::attention_splits(self.ctx), cfg.kv_type,
                    sparse.map_or(std::ptr::null(), |qs| qs.list.as_ptr()), stream)?;
            } else {
                // A single query without the split kernel reads the mask too.
                if n == 1 {
                    if let (Some(qs), Some(qc)) = (sparse, qsa_cache) {
                        qsa::select(qs.scores.as_ptr(), qs.max_blocks, st.positions.as_ptr(), 0, 1, qc.ratio,
                            cfg.indexer.as_ref().map_or(0, |ix| ix.width(qc.ratio)), std::ptr::null_mut(),
                            qs.mask.as_mut_ptr(), qs.words, stream)?;
                    }
                }
                ops::attention_masked(st.attn.as_mut_ptr(), st.q.as_ptr(), kp, vp,
                    st.positions.as_ptr(), n, cfg.n_head, cfg.n_head_kv, cfg.head_dim,
                    self.ctx, cfg.attn_scale(), window, cfg.kv_type,
                    sparse.map(|qs| (qs.mask.as_ptr(), qs.words)), self.n_past + n, stream)?;
            }
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
        let conv_rows = if let Some(rows) = self.mixed.as_deref().filter(|_| self.mixed_t > 1) {
            // Several tokens per conversation (a batched verify): each row
            // convolves out of place against its own slot's history, which
            // moves only when no stash defers it.
            let (t, qd) = (self.mixed_t, ssm.qkv_dim());
            let local = self.states[si].layers.iter().position(|&x| x == layer)
                .ok_or_else(|| CoreError::Other("a mixed decode reached a layer its shard does not hold".into()))?;
            for (b, &slot) in rows.iter().enumerate() {
                let LayerState::Delta { conv: c, .. } = self.slot_state(slot, si, local) else {
                    return Err(CoreError::Other("a linear layer was given a KV cache".into()));
                };
                hybrid::causal_conv1d_oop(h.qkv.offset_ptr(b * t * qd), h.conv_out.offset_ptr(b * t * qd),
                    l.conv1d.as_ptr(), c.as_mut_ptr(), t, qd, ssm.conv_kernel, silu, reverse, stash.is_none(), stream)?;
            }
            &h.conv_out
        } else if let Some(rows) = self.mixed.as_deref() {
            // One token per conversation: each row convolves in place against
            // its own slot's history.
            let local = self.states[si].layers.iter().position(|&x| x == layer)
                .ok_or_else(|| CoreError::Other("a mixed decode reached a layer its shard does not hold".into()))?;
            if let Some((_, convs)) = self.rows_tab_at(si, local) {
                hybrid::causal_conv1d_rows(h.qkv.as_mut_ptr(), l.conv1d.as_ptr(), convs, rows.len(),
                    ssm.qkv_dim(), ssm.conv_kernel, silu, reverse, stream)?;
            } else {
                for (b, &slot) in rows.iter().enumerate() {
                    let LayerState::Delta { conv: c, .. } = self.slot_state(slot, si, local) else {
                        return Err(CoreError::Other("a linear layer was given a KV cache".into()));
                    };
                    hybrid::causal_conv1d_ex(h.qkv.offset_ptr(b * ssm.qkv_dim()), l.conv1d.as_ptr(), c.as_mut_ptr(), 1,
                        ssm.qkv_dim(), ssm.conv_kernel, silu, reverse, true, stream)?;
                }
            }
            &h.qkv
        } else if n > 1 {
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
        if let Some(rows) = self.mixed.as_deref().filter(|_| self.mixed_t > 1) {
            let t = self.mixed_t;
            let local = self.states[si].layers.iter().position(|&x| x == layer)
                .ok_or_else(|| CoreError::Other("a mixed decode reached a layer its shard does not hold".into()))?;
            for (b, &slot) in rows.iter().enumerate() {
                let LayerState::Delta { state: sl, .. } = self.slot_state(slot, si, local) else {
                    return Err(CoreError::Other("a linear layer was given a KV cache".into()));
                };
                let r = b * t;
                hybrid::delta_rule_ex(sl.as_mut_ptr(), st.q.offset_ptr(r * k_dim), st.k.offset_ptr(r * k_dim),
                    h.core.offset_ptr(r * ssm.v_dim()), h.decay.offset_ptr(r * ssm.n_v_heads),
                    h.beta.offset_ptr(r * ssm.n_v_heads), st.attn.offset_ptr(r * ssm.v_dim()), t,
                    ssm.n_k_heads, ssm.n_v_heads, ssm.head_k_dim(), ssm.head_v_dim(), stash.is_none(), stream)?;
            }
        } else if let Some(rows) = self.mixed.as_deref() {
            let local = self.states[si].layers.iter().position(|&x| x == layer)
                .ok_or_else(|| CoreError::Other("a mixed decode reached a layer its shard does not hold".into()))?;
            let batched = self.rows_tab_at(si, local).filter(|_|
                hybrid::delta_rule_rows_ok(ssm.n_k_heads, ssm.n_v_heads, ssm.head_k_dim(), ssm.head_v_dim()));
            if let Some((states, _)) = batched {
                hybrid::delta_rule_rows(states, st.q.as_ptr(), st.k.as_ptr(), h.core.as_ptr(), h.decay.as_ptr(),
                    h.beta.as_ptr(), st.attn.as_mut_ptr(), rows.len(), ssm.n_k_heads, ssm.n_v_heads,
                    ssm.head_k_dim(), ssm.head_v_dim(), stream)?;
            }
            for (b, &slot) in rows.iter().enumerate() {
                if batched.is_some() {
                    break;
                }
                let LayerState::Delta { state: sl, .. } = self.slot_state(slot, si, local) else {
                    return Err(CoreError::Other("a linear layer was given a KV cache".into()));
                };
                hybrid::delta_rule_ex(sl.as_mut_ptr(), st.q.offset_ptr(b * k_dim), st.k.offset_ptr(b * k_dim),
                    h.core.offset_ptr(b * ssm.v_dim()), h.decay.offset_ptr(b * ssm.n_v_heads),
                    h.beta.offset_ptr(b * ssm.n_v_heads), st.attn.offset_ptr(b * ssm.v_dim()), 1,
                    ssm.n_k_heads, ssm.n_v_heads, ssm.head_k_dim(), ssm.head_v_dim(), true, stream)?;
            }
        } else {
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
        }

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
        layer: usize,
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

        if gate_exps.mapped.is_some() {
            if let Some(counts) = &m.counts {
                let local = st.layers.iter().position(|&l| l == layer).unwrap_or(0);
                hybrid::moe_count(m.ids.as_ptr(), counts.offset_ptr(local * moe.n_experts),
                    n * moe.n_experts_used, stream)?;
            }
        }
        // A batch goes expert-major: each expert's weights are read once for
        // all the tokens routed to it, wherever they live.
        if let (Some((scratch, offsets, perm, ypair, xh)), true) = (&m.grouped, n >= GROUPED_MIN && m.upload.is_none()) {
            let k = moe.n_experts_used;
            hybrid::moe_sort(m.ids.as_ptr(), n * k, moe.n_experts, scratch.as_mut_ptr(), offsets.as_mut_ptr(),
                perm.as_mut_ptr(), stream)?;
            let ew = |t: &GpuTensor| hybrid::ExpertWeights {
                ty: t.kernel_type(),
                tab: t.mapped.as_ref().map_or(std::ptr::null(), |mp| mp.table.as_ptr()),
                base: t.ptr(),
                stride: t.expert_stride,
                row_bytes: t.row_bytes,
                resident: t.mapped.as_ref().map_or(std::ptr::null(), |mp| mp.resident.as_ptr()),
            };
            hybrid::moe_grouped(ew(gate_exps), ew(up_exps), ew(down_exps), st.xb.as_ptr(), perm.as_ptr(),
                offsets.as_ptr(), m.weights.as_ptr(), m.h.as_mut_ptr(), ypair.as_mut_ptr(), st.xb2.as_mut_ptr(),
                cfg.n_embd, moe.expert_ff, n, k, moe.n_experts, xh.as_mut_ptr(),
                moe_overlap().then_some((&st.side, &st.fork, &st.join)), stream)?;
            return Ok(());
        }
        if let (Some(gm), Some(um), Some(dm)) = (&gate_exps.mapped, &up_exps.mapped, &down_exps.mapped) {
            let (areas, sidx, list, n_list) = m.stage.as_ref().ok_or_else(|| CoreError::Other(
                "mapped experts without a staging area; this is a bug".into()))?;
            if gate_exps.expert_stride != up_exps.expert_stride {
                return Err(CoreError::Other("gate and up experts differ in size; this model needs the streamed expert path (AIRUN_EXPERTS=stream)".into()));
            }
            let k = moe.n_experts_used;
            // Tokens per pass, so the host-side experts they route to fit
            // the staging area even if none of them is resident.
            let per = (STAGE_CAP / k).max(1);
            for t0 in (0..n).step_by(per) {
                let cnt = per.min(n - t0);
                let ids = m.ids.offset_ptr(t0 * k);
                hybrid::moe_stage_plan(ids, cnt * k, gm.resident.as_ptr(), moe.n_experts, sidx.as_mut_ptr(),
                    list.as_mut_ptr(), n_list.as_mut_ptr(), stream)?;
                // The copy crosses PCIe on the side stream while the resident
                // experts compute here; the staged ones follow once it lands.
                // AIRUN_MOE_OVERLAP=0 keeps it all on one stream.
                let overlap = moe_overlap();
                let copy_on = if overlap {
                    st.fork.record(stream)?;
                    st.side.wait_for(&st.fork)?;
                    &st.side
                } else {
                    stream
                };
                for (t, area) in [gate_exps, up_exps, down_exps].iter().zip(areas) {
                    let tm = t.mapped.as_ref().expect("checked above");
                    hybrid::moe_stage_copy(tm.table.as_ptr(), list.as_ptr(), n_list.as_ptr(), area.as_mut_ptr(),
                        t.expert_stride, cnt * k, copy_on)?;
                }
                let staged = |t: &GpuTensor, only: i32| hybrid::Staged { sidx: sidx.as_ptr(), stride: t.expert_stride, only };
                let passes: &[i32] = if overlap { &[1, 2] } else { &[0] };
                for &only in passes {
                    if only == 2 {
                        st.join.record(&st.side)?;
                        stream.wait_for(&st.join)?;
                    }
                    hybrid::moe_gate_up_tab(gate_exps.kernel_type(), gm.table.as_ptr(), um.table.as_ptr(),
                        st.xb.offset_ptr(t0 * cfg.n_embd), ids, m.h.as_mut_ptr(), cfg.n_embd, moe.expert_ff, cnt, k,
                        gate_exps.row_bytes, Some((staged(gate_exps, only), areas[0].as_ptr(), areas[1].as_ptr())),
                        stream)?;
                    hybrid::moe_down_tab(down_exps.kernel_type(), dm.table.as_ptr(), m.h.as_ptr(), ids,
                        m.weights.offset_ptr(t0 * k), st.xb2.offset_ptr(t0 * cfg.n_embd), moe.expert_ff, cfg.n_embd,
                        cnt, k, down_exps.row_bytes, Some((staged(down_exps, only), areas[2].as_ptr())), stream)?;
                }
            }
            return Ok(());
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
                    gate_exps.kernel_type(), gate_exps.ptr(), up_exps.ptr(),
                    st.xb.offset_ptr(t * cfg.n_embd), m.cached_ids.as_ptr(), m.h.as_mut_ptr(),
                    cfg.n_embd, moe.expert_ff, count, moe.n_experts_used,
                    gate_exps.row_bytes, gate_exps.expert_stride, stream,
                )?;
                hybrid::moe_down(
                    down_exps.kernel_type(), down_exps.ptr(), m.h.as_ptr(),
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
            gate_exps.ptr(),
            up_exps.ptr(),
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
            down_exps.ptr(),
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
        let mut decoder = self.tokenizer.stream_decoder();
        self.generate_with(prompt, max_tokens, sampler, &mut decoder, true, &mut on_token)
    }

    /// [`Self::generate`] with the caller's text decoder, for a generation
    /// that may go on elsewhere (the --parallel scheduler pauses one when
    /// another request arrives). With `flush` false the bytes of a character
    /// cut at the end stay in `decoder`. The token `on_token` refuses is handed
    /// out but not in the caches: what continues the generation feeds it.
    pub fn generate_with(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        sampler: &mut Sampler,
        decoder: &mut crate::tokenizer::StreamDecoder,
        flush: bool,
        on_token: &mut dyn FnMut(u32, &str) -> bool,
    ) -> Result<GenerationStats> {
        // Speculation keeps any sampler's distribution (Sampler::sample_draft)
        // but mirostat's, whose running state would follow rejected drafts.
        let has_images = prompt.iter().any(|&t| is_image_token(t));
        let can_spec = !gpu_argmax_disabled() && (sampler.is_pure_greedy() || sampler.drafts_ok());
        if self.speculates() && !can_spec {
            eprintln!("note: speculative decoding (--draft) is off for this generation: mirostat sampling \
                       does not run with drafts");
        }
        if can_spec && self.speculates() && !has_images {
            return self.generate_spec(prompt, max_tokens, sampler, decoder, flush, on_token);
        }
        let start = std::time::Instant::now();

        // Prefill, skipping whatever the caches already hold. In a chat each
        // request repeats the whole transcript and appends one turn, so this is
        // the difference between replaying thousands of tokens per message and
        // replaying the last few.
        let cached = Self::image_safe_prefix(prompt, self.reusable_prefix(prompt));
        self.rewind_to(cached);
        let cached = self.n_past.min(cached);
        let cached = self.prefill_images(prompt, cached)?;

        // Greedy decoding picks each token on the GPU (see forward_argmax);
        // anything else samples from logits copied to pinned host memory.
        let greedy = sampler.is_pure_greedy() && !gpu_argmax_disabled();
        let mut next: Option<u32> = None;
        let mut prompt_done = cached;
        let prompt_cached = cached;
        let snap_at = self.snapshot_point(prompt, cached);
        self.prefill_report(prompt_done, prompt.len(), prompt_cached, start);
        while prompt_done < prompt.len() {
            let mut take = (prompt.len() - prompt_done).min(self.max_batch);
            if let Some(p) = snap_at.filter(|&p| p > prompt_done) {
                take = take.min(p - prompt_done);
            }
            let chunk = &prompt[prompt_done..prompt_done + take];
            prompt_done += take;
            if prompt_done < prompt.len() {
                self.step(chunk)?;
                if snap_at == Some(prompt_done) {
                    self.take_snapshot(false)?;
                }
            } else if greedy {
                next = Some(self.forward_argmax(chunk)?);
            } else {
                self.forward_pinned(chunk)?;
            }
            self.prefill_report(prompt_done, prompt.len(), prompt_cached, start);
        }
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = std::time::Instant::now();
        let mut generated = 0usize;
        // Penalties look at real tokens only, not an image's sentinels.
        let mut history: Vec<u32> = prompt.iter().copied().filter(|&t| !is_image_token(t)).collect();
        // Streaming has to decode across tokens, not per token: a character can
        // straddle two of them, and decoding each alone would destroy it.
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
        if flush {
            let tail = decoder.flush();
            if !tail.is_empty() {
                on_token(last_token, &tail);
            }
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

/// Sentinel ids at and above this stand for image tokens in a prompt: no
/// vocabulary comes near 2^31.
pub const IMAGE_TOKEN_BASE: u32 = 0x8000_0000;

pub fn is_image_token(t: u32) -> bool {
    t & IMAGE_TOKEN_BASE != 0
}

/// An image's embeddings in the text model's width, row-major over its
/// merged patch grid of `nx` columns and `ny` rows, and the sentinel ids
/// that stand for them in a prompt.
pub struct ImageInput {
    pub ids: Vec<u32>,
    pub embeds: Vec<f32>,
    pub nx: usize,
    pub ny: usize,
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
/// AIRUN_NO_FP4_GLU=1 runs an NVFP4 FFN's gate and up as two FP4 matmuls with
/// the SwiGLU in the down projection's quantization, as before the fused one.
fn fp4_glu_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_FP4_GLU").is_ok_and(|v| v.trim() != "0"))
}

fn fp4_prefill(w: &GpuTensor, n: usize, q8: &QAct) -> bool {
    !on_cpu(w) && n >= 32 && q8.fp4.is_some() && w.kernel_type().0 == 40 && w.k % 256 == 0 && !matvec_f32_forced()
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

/// Batches at least this big run the experts expert-major (see run_moe).
const GROUPED_MIN: usize = 16;
/// Whether run_moe copies host-side experts on a side stream while the
/// resident ones compute (AIRUN_MOE_OVERLAP=0 turns it off).
fn moe_overlap() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AIRUN_MOE_OVERLAP").as_deref() != Ok("0"))
}

/// Forward passes between expert rebalances, and the decay of the routing
/// scores at each one.
const REBALANCE_EVERY: usize = 8;
const REBALANCE_DECAY: f32 = 0.9;
/// Routings a swapped-in expert must be ahead of the one it evicts by.
const SWAP_MARGIN: f32 = 4.0;
/// Rebalances between saves of the routing profile (also saved on drop).
const PROFILE_SAVE_EVERY: usize = 64;

/// Experts a mapped MoE layer stages into VRAM per pass (see run_moe).
const STAGE_CAP: usize = 64;

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
/// The QSA block size of layer `l` when it attends sparsely at this context:
/// a context no longer than the budget is attended densely, which is exact.
/// AIRUN_QSA=off attends densely everywhere.
fn qsa_ratio(cfg: &Config, l: usize, ctx: usize) -> Option<usize> {
    let ix = cfg.indexer.as_ref()?;
    let ratio = ix.ratio(l);
    (ratio > 0 && ctx > ix.width(ratio) && std::env::var("AIRUN_QSA").as_deref() != Ok("off")).then_some(ratio)
}

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
    q8.host_valid.set(false);
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
    if on_cpu(w) {
        return cpu_matvec(w, x, out, n, q8, stream);
    }
    if fp4_prefill(w, n, q8) {
        let buf = q8.fp4.as_ref().expect("checked by fp4_prefill");
        if !q8.fp4_valid.get() {
            ops::quantize_act_nvfp4(x.as_ptr(), buf.as_mut_ptr(), n, w.k, stream)?;
            q8.fp4_valid.set(true);
        }
        ops::matmul_nvfp4(w.ptr(), w.row_bytes, buf.as_ptr(), out.as_mut_ptr(), w.k, w.rows, n, stream)?;
        apply_global_scale(w, out, n, stream)?;
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
            w.ptr(),
            q8.buf.as_ptr(),
            out.as_mut_ptr(),
            w.k,
            w.rows,
            n,
            w.row_bytes,
            stream,
        )?;
        apply_global_scale(w, out, n, stream)?;
        return Ok(());
    }
    ops::matvec(
        w.kernel_type(),
        w.ptr(),
        x.as_ptr(),
        out.as_mut_ptr(),
        w.k,
        w.rows,
        n,
        w.row_bytes,
        stream,
    )?;
    apply_global_scale(w, out, n, stream)?;
    Ok(())
}

/// NVFP4 weights are stored ~1e-4 of their true magnitude; the per-tensor
/// global scale (see `GpuTensor::global_scale`, read from the file's
/// `<base>.scale` sidecar) restores it. Without it the dequant is thousands
/// of times too large and the hidden states overflow to NaN. A no-op for every
/// non-NVFP4 weight, whose scale is 1.0.
fn apply_global_scale(w: &GpuTensor, out: &DeviceBuffer<f32>, n: usize, stream: &Stream) -> Result<()> {
    if w.global_scale != 1.0 {
        ops::scale(out.as_mut_ptr(), w.global_scale, n * w.rows, stream)?;
    }
    Ok(())
}
