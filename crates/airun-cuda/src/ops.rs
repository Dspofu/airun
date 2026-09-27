//! Safe wrappers over the kernels.
//!
//! Every entry point takes a [`Stream`] and returns as soon as the work is
//! queued; nothing here synchronizes. Shape checks happen on the Rust side so a
//! mismatch is a typed error rather than an illegal memory access inside a
//! kernel.

use std::ffi::c_void;

use crate::error::{CudaError, Result};
use crate::ffi::{self, Status};
use crate::stream::Stream;

// Declared here rather than in ffi.rs to keep the kernel entry points next to
// the wrappers that give them meaning.
#[cfg(airun_cuda)]
unsafe extern "C" {
    fn airun_mmq_supported() -> i32;
    fn airun_argmax(x: *const c_void, n: i32, out: *mut c_void, scratch: *mut c_void,
        bump: *mut c_void, stream: ffi::Stream) -> Status;    fn airun_add_rmsnorm(out: *mut c_void, x: *mut c_void, b: *const c_void, weight: *const c_void,
        n: i32, rows: i32, eps: f32, stream: ffi::Stream) -> Status;
    #[allow(clippy::too_many_arguments)]
    fn airun_qk_norm_rope_store(q: *mut c_void, k: *const c_void, v: *const c_void,
        q_norm: *const c_void, k_norm: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void,
        positions: *const c_void, n_tokens: i32, n_heads: i32, n_kv_heads: i32, head_dim: i32,
        max_seq: i32, eps: f32, rope_dim: i32, freq_base: f32, freq_scale: f32,
        interleaved: i32, rope_on: i32, kv_type: i32, stream: ffi::Stream) -> Status;
    fn airun_attention_split(out: *mut c_void, partial: *mut c_void,
        q: *const c_void, k: *const c_void, v: *const c_void, positions: *const c_void,
        heads: i32, kv_heads: i32, dim: i32, max_seq: i32, scale: f32,
        window: i32, splits: i32, kv_type: i32, stream: ffi::Stream) -> Status;
    fn airun_matvec(
        type_id: i32,
        w: *const c_void,
        x: *const c_void,
        y: *mut c_void,
        k: i32,
        n_rows: i32,
        n_tokens: i32,
        row_bytes: u64,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_matvec_has_q8(type_id: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn airun_matvec_glu(type_id: i32, wg: *const c_void, wu: *const c_void, x: *const c_void,
        y: *mut c_void, k: i32, n_rows: i32, row_bytes: u64, stream: ffi::Stream) -> Status;
    fn airun_mmq_glu_supported(type_id: i32, wg: *const c_void, wu: *const c_void,
        xq: *const c_void, k: i32, row_bytes: u64, n_tokens: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn airun_matvec_q8_glu(type_id: i32, wg: *const c_void, wu: *const c_void,
        xq: *const c_void, y: *mut c_void, k: i32, n_rows: i32, n_tokens: i32,
        row_bytes: u64, stream: ffi::Stream) -> Status;
    #[allow(clippy::too_many_arguments)]
    fn airun_matvec3(
        type_id: i32,
        w0: *const c_void, w1: *const c_void, w2: *const c_void,
        x: *const c_void,
        y0: *mut c_void, y1: *mut c_void, y2: *mut c_void,
        k: i32, n0: i32, n1: i32, n2: i32,
        row_bytes: u64,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_quantize_act(
        x: *const c_void,
        out: *mut c_void,
        n_elements: i64,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_matvec_q8(
        type_id: i32,
        w: *const c_void,
        xq: *const c_void,
        y: *mut c_void,
        k: i32,
        n_rows: i32,
        n_tokens: i32,
        row_bytes: u64,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_dequantize(
        type_id: i32,
        src: *const c_void,
        dst: *mut c_void,
        n_elements: i64,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_rmsnorm(
        out: *mut c_void,
        x: *const c_void,
        weight: *const c_void,
        n: i32,
        rows: i32,
        eps: f32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_rope(
        x: *mut c_void,
        positions: *const c_void,
        n_tokens: i32,
        n_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        freq_base: f32,
        freq_scale: f32,
        interleaved: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_swiglu(gate: *mut c_void, up: *const c_void, n: i64, stream: ffi::Stream) -> Status;
    fn airun_add(a: *mut c_void, b: *const c_void, n: i64, stream: ffi::Stream) -> Status;
    fn airun_scale(a: *mut c_void, s: f32, n: i64, stream: ffi::Stream) -> Status;
    fn airun_gelu(a: *mut c_void, n: i64, stream: ffi::Stream) -> Status;

    fn airun_embed(
        type_id: i32,
        out: *mut c_void,
        table: *const c_void,
        tokens: *const c_void,
        n_tokens: i32,
        n_embd: i32,
        row_bytes: u64,
        vocab_size: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_kv_store(
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        k: *const c_void,
        v: *const c_void,
        positions: *const c_void,
        n_tokens: i32,
        n_kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
        kv_type: i32,
        stream: ffi::Stream,
    ) -> Status;

    #[allow(clippy::too_many_arguments)]
    fn airun_attention(
        out: *mut c_void,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        positions: *const c_void,
        n_tokens: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        max_seq: i32,
        scale: f32,
        sliding_window: i32,
        kv_type: i32,
        stream: ffi::Stream,
    ) -> Status;
}

/// Without CUDA every kernel reports the same "not built with CUDA" status.
#[cfg(not(airun_cuda))]
mod nocuda {
    use super::*;
    /// Matches the stub `airun_error_name`, which maps to `NotBuiltWithCuda`.
    pub const NO_CUDA: Status = 100;
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn unavailable() -> Status {
        NO_CUDA
    }
}

/// A tensor's element type, as the kernels' dispatch switch understands it.
///
/// The discriminants are GGUF type ids, so this converts from
/// `airun_gguf::GgmlType` by id without a translation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelType(pub i32);

impl KernelType {
    /// Type ids the kernels have a compiled decoder for.
    pub const SUPPORTED: &'static [i32] =
        &[0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23, 30, 39, 40];

    pub fn is_supported(self) -> bool {
        Self::SUPPORTED.contains(&self.0)
    }
}

fn bad_arg(what: impl Into<String>) -> CudaError {
    CudaError::Runtime {
        code: -1,
        name: "AIRUN_BAD_ARG".into(),
        message: what.into(),
    }
}

fn unsupported_type(ty: KernelType) -> CudaError {
    CudaError::Runtime {
        code: -1,
        name: "AIRUN_UNSUPPORTED_TYPE".into(),
        message: format!(
            "no GPU kernel for ggml type id {}; airun can compute on ids {:?}",
            ty.0,
            KernelType::SUPPORTED
        ),
    }
}

/// `y[token][row] = dot(w[row], x[token])`.
///
/// `w` is the packed tensor exactly as stored in the GGUF file: `n_rows` rows of
/// `row_bytes`, each holding `k` logical values.
#[allow(clippy::too_many_arguments)]
pub fn matvec(
    ty: KernelType,
    w: *const c_void,
    x: *const c_void,
    y: *mut c_void,
    k: usize,
    n_rows: usize,
    n_tokens: usize,
    row_bytes: usize,
    stream: &Stream,
) -> Result<()> {
    if !ty.is_supported() {
        return Err(unsupported_type(ty));
    }
    if k % 32 != 0 {
        return Err(bad_arg(format!(
            "matvec needs an inner dimension divisible by 32, got {k}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: the pointers are device allocations owned by the caller and sized
    // to the dimensions passed; the kernel only reads/writes inside them.
    unsafe {
        ffi::check(airun_matvec(
            ty.0,
            w,
            x,
            y,
            k as i32,
            n_rows as i32,
            n_tokens as i32,
            row_bytes as u64,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (w, x, y, n_rows, n_tokens, row_bytes, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// One-token products of three matrices that share `x`, inner size, weight
/// type and row size -- the q, k and v projections of a decode step -- in one
/// launch. Bit-identical to three [`matvec`] calls. `ws[i]`, `ys[i]` and
/// `rows[i]` describe matrix i; the first two row counts must be multiples of 4.
pub fn matvec3(
    ty: KernelType,
    ws: [*const c_void; 3],
    x: *const c_void,
    ys: [*mut c_void; 3],
    k: usize,
    rows: [usize; 3],
    row_bytes: usize,
    stream: &Stream,
) -> Result<()> {
    if !ty.is_supported() {
        return Err(unsupported_type(ty));
    }
    if k % 32 != 0 || rows[0] % 4 != 0 || rows[1] % 4 != 0 {
        return Err(bad_arg(format!(
            "matvec3 needs k divisible by 32 and the first two row counts by 4, got k {k}, rows {rows:?}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: device allocations owned by the caller, sized to the dimensions
    // passed; the kernel reads and writes only inside them.
    unsafe {
        ffi::check(airun_matvec3(
            ty.0, ws[0], ws[1], ws[2], x, ys[0], ys[1], ys[2],
            k as i32, rows[0] as i32, rows[1] as i32, rows[2] as i32,
            row_bytes as u64, stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (ws, x, ys, rows, row_bytes, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
/// One-token `y = silu(gate x) * (up x)`, the decode FFN's gate, up and SwiGLU
/// in one launch; bit-identical to [`matvec`] twice and [`swiglu`].
#[allow(clippy::too_many_arguments)]
pub fn matvec_glu(ty: KernelType, wg: *const c_void, wu: *const c_void, x: *const c_void,
    y: *mut c_void, k: usize, n_rows: usize, row_bytes: usize, stream: &Stream) -> Result<()> {
    if !ty.is_supported() {
        return Err(unsupported_type(ty));
    }
    #[cfg(airun_cuda)]
    // SAFETY: device allocations owned by the caller, sized for n_rows x k
    // weights, k inputs and n_rows outputs.
    unsafe {
        ffi::check(airun_matvec_glu(ty.0, wg, wu, x, y, k as i32, n_rows as i32,
            row_bytes as u64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (wg, wu, x, y, k, n_rows, row_bytes, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
/// Whether [`matvec_q8_glu`] can run gate and up with these pointers and sizes:
/// integer tensor cores on this GPU, a type the asynchronous MMQ kernel reads,
/// aligned buffers and enough tokens.
#[allow(clippy::too_many_arguments)]
pub fn mmq_glu_supported(ty: KernelType, wg: *const c_void, wu: *const c_void, xq: *const c_void,
    k: usize, row_bytes: usize, n_tokens: usize) -> bool {
    #[cfg(airun_cuda)]
    // SAFETY: inspects the arguments and the device's capability; touches no memory.
    unsafe {
        airun_mmq_glu_supported(ty.0, wg, wu, xq, k as i32, row_bytes as u64, n_tokens as i32) != 0
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (ty, wg, wu, xq, k, row_bytes, n_tokens);
        false
    }
}

/// `y = silu(gate x) * (up x)` for `n_tokens` quantized activations: the FFN's
/// gate and up projections and its SwiGLU in one launch. `gate` and `up` share
/// type, shape and row size. Check [`mmq_glu_supported`] first.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q8_glu(ty: KernelType, wg: *const c_void, wu: *const c_void, xq: *const c_void,
    y: *mut c_void, k: usize, n_rows: usize, n_tokens: usize, row_bytes: usize,
    stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: device allocations owned by the caller, sized for n_rows x k
    // weights, n_tokens activation rows and n_tokens x n_rows outputs.
    unsafe {
        ffi::check(airun_matvec_q8_glu(ty.0, wg, wu, xq, y, k as i32, n_rows as i32,
            n_tokens as i32, row_bytes as u64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (ty, wg, wu, xq, y, k, n_rows, n_tokens, row_bytes, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
/// Whether [`matvec_q8`] has an integer kernel for this weight type.
pub fn matvec_supports_q8(ty: KernelType) -> bool {
    #[cfg(airun_cuda)]
    // SAFETY: pure function of the id, touches nothing.
    unsafe {
        airun_matvec_has_q8(ty.0) != 0
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = ty;
        false
    }
}

/// Whether [`matvec_q8`] runs large batches on integer tensor cores on the
/// current device (sm_80+, and not disabled with `AIRUN_NO_MMQ=1`). Where it
/// does, that beats the fp16 path of [`matvec`] for batches of 64 and more.
pub fn mmq_supported() -> bool {
    #[cfg(airun_cuda)]
    // SAFETY: reads the current device's compute capability and nothing else.
    unsafe {
        airun_mmq_supported() != 0
    }
    #[cfg(not(airun_cuda))]
    {
        false
    }
}

/// Quantizes `n_elements` f32 activations into `block_q8a` groups of 32
/// (36 bytes per group: the codes, then scale and folded sum as half2).
pub fn quantize_act(
    x: *const c_void,
    out: *mut c_void,
    n_elements: usize,
    stream: &Stream,
) -> Result<()> {
    if n_elements % 32 != 0 {
        return Err(bad_arg(format!(
            "quantize_act needs an element count divisible by 32, got {n_elements}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `x` holds `n_elements` floats, `out` holds `n_elements/32` blocks
    // of 36 bytes; the kernel writes only inside them.
    unsafe {
        ffi::check(airun_quantize_act(x, out, n_elements as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (x, out, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// [`matvec`] against an activation already quantized by [`quantize_act`].
///
/// Not bit-identical to the f32 path -- the activation carries 8-bit codes --
/// but validated by the selftest's backward-error bound against a CPU mirror
/// of the same algorithm. Only types where [`matvec_supports_q8`] is true.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q8(
    ty: KernelType,
    w: *const c_void,
    xq: *const c_void,
    y: *mut c_void,
    k: usize,
    n_rows: usize,
    n_tokens: usize,
    row_bytes: usize,
    stream: &Stream,
) -> Result<()> {
    if !matvec_supports_q8(ty) {
        return Err(bad_arg(format!(
            "no integer matvec kernel for ggml type id {}; check matvec_supports_q8 first",
            ty.0
        )));
    }
    if k % 32 != 0 {
        return Err(bad_arg(format!(
            "matvec needs an inner dimension divisible by 32, got {k}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: pointers are device allocations sized to the dimensions passed.
    unsafe {
        ffi::check(airun_matvec_q8(
            ty.0,
            w,
            xq,
            y,
            k as i32,
            n_rows as i32,
            n_tokens as i32,
            row_bytes as u64,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (w, xq, y, n_rows, n_tokens, row_bytes, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Expands a packed tensor to f32 on the device.
pub fn dequantize(
    ty: KernelType,
    src: *const c_void,
    dst: *mut c_void,
    n_elements: usize,
    stream: &Stream,
) -> Result<()> {
    if !ty.is_supported() {
        return Err(unsupported_type(ty));
    }
    if n_elements % 32 != 0 {
        return Err(bad_arg(format!(
            "dequantize needs an element count divisible by 32, got {n_elements}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `dst` holds `n_elements` floats and `src` the matching packed bytes.
    unsafe {
        ffi::check(airun_dequantize(ty.0, src, dst, n_elements as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (src, dst, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// RMSNorm over the last axis of `rows x n`. `weight` may be null.
pub fn rmsnorm(
    out: *mut c_void,
    x: *const c_void,
    weight: *const c_void,
    n: usize,
    rows: usize,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `out` and `x` each hold `rows * n` floats; `weight` holds `n` or is null.
    unsafe {
        ffi::check(airun_rmsnorm(
            out,
            x,
            weight,
            n as i32,
            rows as i32,
            eps,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, x, weight, n, rows, eps, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// In-place rotary embedding over `[n_tokens][n_heads][head_dim]`.
///
/// `rope_dim` may be smaller than `head_dim`, leaving the tail unrotated, which
/// is how partial rotary factors are expressed. `interleaved` selects the GPT-J
/// pairing of adjacent dimensions instead of the NeoX pairing GGUF normally
/// carries.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: *mut c_void,
    positions: *const c_void,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    freq_scale: f32,
    interleaved: bool,
    stream: &Stream,
) -> Result<()> {
    if rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0 {
        return Err(bad_arg(format!(
            "rope_dim {rope_dim} must be even and at most head_dim {head_dim}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `x` holds n_tokens * n_heads * head_dim floats; `positions` holds n_tokens i32s.
    unsafe {
        ffi::check(airun_rope(
            x,
            positions,
            n_tokens as i32,
            n_heads as i32,
            head_dim as i32,
            rope_dim as i32,
            freq_base,
            freq_scale,
            interleaved as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (x, positions, n_tokens, n_heads, freq_base, freq_scale, interleaved, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// `gate = silu(gate) * up`, in place.
pub fn swiglu(gate: *mut c_void, up: *const c_void, n: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: both buffers hold at least `n` floats.
    unsafe {
        ffi::check(airun_swiglu(gate, up, n as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (gate, up, n, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// `a += b`, in place.
pub fn add(a: *mut c_void, b: *const c_void, n: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: both buffers hold at least `n` floats.
    unsafe {
        ffi::check(airun_add(a, b, n as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (a, b, n, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// `a *= s`, in place.
pub fn scale(a: *mut c_void, s: f32, n: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `a` holds at least `n` floats.
    unsafe {
        ffi::check(airun_scale(a, s, n as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (a, s, n, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Tanh-approximate GELU, in place.
pub fn gelu(a: *mut c_void, n: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `a` holds at least `n` floats.
    unsafe {
        ffi::check(airun_gelu(a, n as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (a, n, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Gathers embedding rows for `tokens` straight from the packed table.
#[allow(clippy::too_many_arguments)]
pub fn embed(
    ty: KernelType,
    out: *mut c_void,
    table: *const c_void,
    tokens: *const c_void,
    n_tokens: usize,
    n_embd: usize,
    row_bytes: usize,
    vocab_size: usize,
    stream: &Stream,
) -> Result<()> {
    if !ty.is_supported() {
        return Err(unsupported_type(ty));
    }
    if n_embd % 32 != 0 {
        return Err(bad_arg(format!(
            "embedding width must be divisible by 32, got {n_embd}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `out` holds n_tokens * n_embd floats; the kernel clamps token ids
    // against `vocab_size` before indexing the table.
    unsafe {
        ffi::check(airun_embed(
            ty.0,
            out,
            table,
            tokens,
            n_tokens as i32,
            n_embd as i32,
            row_bytes as u64,
            vocab_size as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, table, tokens, n_tokens, row_bytes, vocab_size, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Element format of the KV cache. The quantized formats are GGUF's Q8_0 and
/// Q4_0 blocks, one per 32 values of a head row, so they decode exactly like
/// weights of those types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvType {
    #[default]
    F16,
    Q8_0,
    Q4_0,
}

impl KvType {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" => Some(Self::F16),
            "q8_0" | "q8" => Some(Self::Q8_0),
            "q4_0" | "q4" => Some(Self::Q4_0),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    /// Bytes of one cached head row of head_dim values.
    pub fn row_bytes(self, head_dim: usize) -> usize {
        match self {
            Self::F16 => head_dim * 2,
            Self::Q8_0 => head_dim / 32 * 34,
            Self::Q4_0 => head_dim / 32 * 18,
        }
    }

    #[cfg_attr(not(airun_cuda), allow(dead_code))]
    fn id(self) -> i32 {
        match self {
            Self::F16 => 0,
            Self::Q8_0 => 1,
            Self::Q4_0 => 2,
        }
    }
}

/// Device scratch [`argmax`] needs, in 4-byte elements.
pub const ARGMAX_SCRATCH: usize = 256;

/// Index of the first maximum of `n` floats, written to one i32 on the device;
/// the same pick as the sampler's greedy argmax. `scratch` holds
/// [`ARGMAX_SCRATCH`] 4-byte elements. A non-null `bump` points at an i32 that
/// is incremented once the index is written.
pub fn argmax(x: *const c_void, n: usize, out: *mut c_void, scratch: *mut c_void,
              bump: *mut c_void, stream: &Stream) -> Result<()> {
    if n == 0 || n > i32::MAX as usize {
        return Err(bad_arg(format!("argmax over {n} values; the vocabulary must be non-empty")));
    }
    if scratch.is_null() {
        return Err(bad_arg("argmax needs a scratch buffer of ARGMAX_SCRATCH elements"));
    }
    #[cfg(airun_cuda)]
    // SAFETY: x holds n floats, out one i32, scratch ARGMAX_SCRATCH 4-byte
    // elements and bump (if non-null) one i32: device memory the caller owns.
    unsafe { ffi::check(airun_argmax(x, n as i32, out, scratch, bump, stream.raw())) }
    #[cfg(not(airun_cuda))]
    { let _ = (x, out, bump, stream); Err(CudaError::NotBuiltWithCuda) }
}
/// `x += b; out = rmsnorm(x) * weight`, per row of `n`: a residual add and the
/// norm after it in one launch. Identical to [`add`] then [`rmsnorm`].
#[allow(clippy::too_many_arguments)]
pub fn add_rmsnorm(out: *mut c_void, x: *mut c_void, b: *const c_void, weight: *const c_void,
    n: usize, rows: usize, eps: f32, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: out, x and b hold rows * n floats and weight n; all owned by the
    // caller for the duration of the stream's work.
    unsafe { ffi::check(airun_add_rmsnorm(out, x, b, weight, n as i32, rows as i32, eps, stream.raw())) }
    #[cfg(not(airun_cuda))]
    { let _ = (out, x, b, weight, n, rows, eps, stream); Err(CudaError::NotBuiltWithCuda) }
}
/// The attention prologue in one launch: per-head RMSNorm of q and k (skipped
/// when the weight is null), RoPE on both (unless `rope`is None), and k and v
/// into the cache. Identical to [`rmsnorm`], [`rope`] and [`kv_store`] in turn.
/// `rope` is (rope_dim, freq_base, freq_scale, interleaved).
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_store(q: *mut c_void, k: *const c_void, v: *const c_void,
    q_norm: *const c_void, k_norm: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void,
    positions: *const c_void, n_tokens: usize, n_heads: usize, n_kv_heads: usize,
    head_dim: usize, max_seq: usize, eps: f32, rope: Option<(usize, f32, f32, bool)>,
    kv_type: KvType, stream: &Stream) -> Result<()> {
    if head_dim % 32 != 0 || head_dim > 1024 {
        return Err(bad_arg(format!("the fused attention prologue needs head_dim divisible by 32, got {head_dim}")));
    }
    let (rope_dim, base, scale, interleaved) = rope.unwrap_or((2, 1.0, 1.0, false));
    #[cfg(airun_cuda)]
    // SAFETY: q/k/v hold n_tokens rows of their head counts, the caches
    // n_kv_heads * max_seq rows in kv_type's format; positions outside
    // [0, max_seq) are skipped by the kernel.
    unsafe {
        ffi::check(airun_qk_norm_rope_store(q, k, v, q_norm, k_norm, k_cache, v_cache, positions,
            n_tokens as i32, n_heads as i32, n_kv_heads as i32, head_dim as i32, max_seq as i32,
            eps, rope_dim as i32, base, scale, interleaved as i32, rope.is_some() as i32,
            kv_type.id(), stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (q, k, v, q_norm, k_norm, k_cache, v_cache, positions, n_tokens, n_heads,
            n_kv_heads, max_seq, eps, rope_dim, base, scale, interleaved, kv_type, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
/// Writes freshly computed K/V for each token into the cache at its position.
#[allow(clippy::too_many_arguments)]
pub fn kv_store(
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    k: *const c_void,
    v: *const c_void,
    positions: *const c_void,
    n_tokens: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    kv_type: KvType,
    stream: &Stream,
) -> Result<()> {
    if kv_type != KvType::F16 && head_dim % 32 != 0 {
        return Err(bad_arg(format!(
            "a {} KV cache needs head_dim divisible by 32, got {head_dim}; use --kv-type f16",
            kv_type.name()
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: the caches hold n_kv_heads * max_seq rows of kv_type.row_bytes(head_dim); the kernel
    // drops any position outside [0, max_seq).
    unsafe {
        ffi::check(airun_kv_store(
            k_cache,
            v_cache,
            k,
            v,
            positions,
            n_tokens as i32,
            n_kv_heads as i32,
            head_dim as i32,
            max_seq as i32,
            kv_type.id(),
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (k_cache, v_cache, k, v, positions, n_tokens, n_kv_heads, head_dim, max_seq, kv_type, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Causal attention against the KV cache.
///
/// `sliding_window` of 0 means full causal attention.
/// KV splits for decode attention. The decode kernel runs one warp per (KV head,
/// split), so it needs hundreds of splits to fill a GPU when there are few KV
/// heads (2 on the 35B); capped so the merge stays cheap.
pub fn attention_splits(ctx: usize) -> usize { ctx.div_ceil(64).clamp(16, 256) }

/// Decode-only attention with a bounded scratch reduction across KV splits.
#[allow(clippy::too_many_arguments)]
pub fn attention_split(out: *mut c_void, partial: &crate::DeviceBuffer<f32>,
    q: *const c_void, k: *const c_void, v: *const c_void, positions: *const c_void,
    heads: usize, kv_heads: usize, dim: usize, ctx: usize, scale: f32,
    window: usize, splits: usize, kv_type: KvType, stream: &Stream) -> Result<()> {
    if dim == 0 || dim > 256 || dim % 32 != 0 || kv_heads == 0 || heads % kv_heads != 0
        || splits == 0 || splits > 256 || partial.len() < heads * splits * (dim + 2) {
        return Err(bad_arg("invalid split attention dimensions or scratch; resize the attention buffers"));
    }
    #[cfg(airun_cuda)]
    // SAFETY: scratch and shape are checked above; runtime owns the Q/K/V and
    // position buffers until the stream completes, just as for attention().
    unsafe { ffi::check(airun_attention_split(out, partial.as_mut_ptr(), q, k, v,
        positions, heads as i32, kv_heads as i32, dim as i32, ctx as i32,
        scale, window as i32, splits as i32, kv_type.id(), stream.raw())) }
    #[cfg(not(airun_cuda))]
    { let _ = (out, q, k, v, positions, ctx, scale, window, kv_type, stream); Err(CudaError::NotBuiltWithCuda) }
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    out: *mut c_void,
    q: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    positions: *const c_void,
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    scale: f32,
    sliding_window: usize,
    kv_type: KvType,
    stream: &Stream,
) -> Result<()> {
    if head_dim % 32 != 0 || head_dim > 256 {
        return Err(bad_arg(format!(
            "attention supports head_dim divisible by 32 and at most 256, got {head_dim}"
        )));
    }
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(bad_arg(format!(
            "n_heads {n_heads} must be a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: shapes are validated above and the caches are sized by the caller
    // to n_kv_heads * max_seq * head_dim.
    unsafe {
        ffi::check(airun_attention(
            out,
            q,
            k_cache,
            v_cache,
            positions,
            n_tokens as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            max_seq as i32,
            scale,
            sliding_window as i32,
            kv_type.id(),
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, q, k_cache, v_cache, positions, n_tokens, n_heads, max_seq, scale, sliding_window, kv_type, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
