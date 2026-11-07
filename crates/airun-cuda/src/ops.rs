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
    fn airun_fp4_supported() -> i32;
    fn airun_nvfp4_act_bytes(n_tokens: i32, k: i32) -> u64;
    fn airun_quantize_act_nvfp4(x: *const c_void, out: *mut c_void, n_tokens: i32, k: i32,
        stream: ffi::Stream) -> Status;
    fn airun_quantize_glu_nvfp4(gate: *const c_void, up: *const c_void, out: *mut c_void, n_tokens: i32,
        k: i32, stream: ffi::Stream) -> Status;
    fn airun_matmul_nvfp4_glu(w: *const c_void, w2: *const c_void, row_bytes: u64, xq: *const c_void,
        y: *mut c_void, k: i32, n_rows: i32, n_tokens: i32, stream: ffi::Stream) -> Status;
    fn airun_matmul_nvfp4(w: *const c_void, row_bytes: u64, xq: *const c_void, y: *mut c_void,
        k: i32, n_rows: i32, n_tokens: i32, stream: ffi::Stream) -> Status;
    fn airun_nvfp4_sanitize(w: *mut c_void, n_blocks: i64, stream: ffi::Stream) -> Status;
    fn airun_argmax(x: *const c_void, n: i32, out: *mut c_void, scratch: *mut c_void,
        bump: *mut c_void, stream: ffi::Stream) -> Status;    fn airun_add_rmsnorm(out: *mut c_void, x: *mut c_void, b: *const c_void, weight: *const c_void,
        n: i32, rows: i32, eps: f32, stream: ffi::Stream) -> Status;
    #[allow(clippy::too_many_arguments)]
    fn airun_qk_norm_rope_store(q: *mut c_void, k: *const c_void, v: *const c_void,
        q_norm: *const c_void, k_norm: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void,
        positions: *const c_void, n_tokens: i32, n_heads: i32, n_kv_heads: i32, head_dim: i32,
        max_seq: i32, eps: f32, rope_dim: i32, rope: RopeParams,
        interleaved: i32, rope_on: i32, kv_type: i32, stream: ffi::Stream) -> Status;
    fn airun_attention_split_rows(out: *mut c_void, partial: *mut c_void, q: *const c_void,
        kv_rows: *const *const c_void, positions: *const c_void, heads: i32, kv_heads: i32, dim: i32,
        max_seq: i32, scale: f32, window: i32, splits: i32, kv_type: i32, rows: i32, stream: ffi::Stream) -> Status;
    fn airun_attention_split(out: *mut c_void, partial: *mut c_void,
        q: *const c_void, k: *const c_void, v: *const c_void, positions: *const c_void,
        heads: i32, kv_heads: i32, dim: i32, max_seq: i32, scale: f32,
        window: i32, splits: i32, kv_type: i32, sel: *const c_void, stream: ffi::Stream) -> Status;
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
        rope: RopeParams,
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

    fn airun_attention_split_multi(
        out: *mut c_void,
        partial: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        positions: *const c_void,
        heads: i32,
        kv_heads: i32,
        dim: i32,
        max_seq: i32,
        scale: f32,
        window: i32,
        splits: i32,
        kv_type: i32,
        rows: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_kv_stage(
        dst: *mut c_void,
        src: *const c_void,
        positions: *const c_void,
        n_tokens: i32,
        n_kv_heads: i32,
        max_seq: i32,
        row_bytes: i32,
        to_stage: i32,
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
        mask: *const c_void,
        mask_words: i32,
        keys: i32,
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
        &[0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 18, 20, 21, 22, 23, 30, 39, 40, 42];

    pub fn is_supported(self) -> bool {
        Self::SUPPORTED.contains(&self.0)
    }

    /// Whether decode (one token) should take the integer dp4a kernel rather
    /// than the exact f32 one. At 4.5 bits per weight the f32 vector each warp
    /// re-reads is ~3.5x the bytes of the weights it is spent against, and the
    /// quantized vector is a quarter of that; at Q8_0's 8.5 bits the weights
    /// dominate and f32 measured faster (148 vs 133 tok/s, 1.7B).
    pub fn prefers_q8_decode(self) -> bool {
        self.0 == 40   // NVFP4
    }
}

pub(crate) fn bad_arg(what: impl Into<String>) -> CudaError {
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

/// Whether this build and the current device run NVFP4 x NVFP4 on Blackwell's
/// FP4 tensor cores (sm_120a; see kernels/sm120a/fp4mma.cu). Not on
/// AIRUN_NO_FP4=1, which keeps NVFP4 prefill on the int8 MMQ path.
pub fn fp4_supported() -> bool {
    if std::env::var("AIRUN_NO_FP4").is_ok_and(|v| v.trim() != "0") {
        return false;
    }
    #[cfg(airun_cuda)]
    // SAFETY: reads the current device's compute capability and nothing else.
    unsafe {
        airun_fp4_supported() != 0
    }
    #[cfg(not(airun_cuda))]
    {
        false
    }
}

/// Bytes [`quantize_act_nvfp4`] writes for `n_tokens` rows of `k` values.
pub fn nvfp4_act_bytes(n_tokens: usize, k: usize) -> usize {
    n_tokens * (k / 64) * 36 + n_tokens * 4
}

/// f32 activations `[n_tokens][k]` into NVFP4 blocks plus one f32 scale per
/// row, the activation side of [`matmul_nvfp4`]. `k` must be a multiple of 64.
pub fn quantize_act_nvfp4(x: *const c_void, out: *mut c_void, n_tokens: usize, k: usize,
    stream: &Stream) -> Result<()> {
    if k % 64 != 0 {
        return Err(bad_arg(format!("NVFP4 activations need K divisible by 64, got {k}")));
    }
    #[cfg(airun_cuda)]
    // SAFETY: x holds n_tokens * k floats; out holds nvfp4_act_bytes(n_tokens, k).
    unsafe {
        debug_assert_eq!(airun_nvfp4_act_bytes(n_tokens as i32, k as i32) as usize, nvfp4_act_bytes(n_tokens, k));
        ffi::check(airun_quantize_act_nvfp4(x, out, n_tokens as i32, k as i32, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (x, out, n_tokens, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// [`quantize_act_nvfp4`] of `silu(gate) * up`: the SwiGLU fused into the
/// down projection's input.
pub fn quantize_glu_nvfp4(gate: *const c_void, up: *const c_void, out: *mut c_void, n_tokens: usize, k: usize,
    stream: &Stream) -> Result<()> {
    if k % 64 != 0 {
        return Err(bad_arg(format!("NVFP4 activations need K divisible by 64, got {k}")));
    }
    #[cfg(airun_cuda)]
    // SAFETY: gate and up hold n_tokens * k floats; out nvfp4_act_bytes(n_tokens, k).
    unsafe {
        ffi::check(airun_quantize_glu_nvfp4(gate, up, out, n_tokens as i32, k as i32, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (gate, up, out, n_tokens, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// `y[t][row] = W[row] . x[t]` on the FP4 tensor cores, W NVFP4 as stored and x
/// from [`quantize_act_nvfp4`]. Check [`fp4_supported`] first.
#[allow(clippy::too_many_arguments)]
pub fn matmul_nvfp4(w: *const c_void, row_bytes: usize, xq: *const c_void, y: *mut c_void, k: usize,
    n_rows: usize, n_tokens: usize, stream: &Stream) -> Result<()> {
    if k % 256 != 0 || row_bytes != k / 64 * 36 {
        return Err(bad_arg(format!("NVFP4 matmul: K {k} must divide by 256 and rows be {} bytes", k / 64 * 36)));
    }
    #[cfg(airun_cuda)]
    // SAFETY: w holds n_rows rows of row_bytes; xq the quantized n_tokens rows;
    // y n_tokens * n_rows floats.
    unsafe {
        ffi::check(airun_matmul_nvfp4(w, row_bytes as u64, xq, y, k as i32, n_rows as i32, n_tokens as i32,
            stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (w, xq, y, n_rows, n_tokens, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// `y[t][row] = silu(Wg[row] . x[t]) * (Wu[row] . x[t])` on the FP4 tensor
/// cores: an FFN's gate, up and SwiGLU in one launch ([`matmul_nvfp4`] rules).
#[allow(clippy::too_many_arguments)]
pub fn matmul_nvfp4_glu(wg: *const c_void, wu: *const c_void, row_bytes: usize, xq: *const c_void, y: *mut c_void,
    k: usize, n_rows: usize, n_tokens: usize, stream: &Stream) -> Result<()> {
    if k % 256 != 0 || row_bytes != k / 64 * 36 {
        return Err(bad_arg(format!("NVFP4 matmul: K {k} must divide by 256 and rows be {} bytes", k / 64 * 36)));
    }
    #[cfg(airun_cuda)]
    // SAFETY: wg and wu hold n_rows rows of row_bytes; xq the quantized
    // n_tokens rows; y n_tokens * n_rows floats.
    unsafe {
        ffi::check(airun_matmul_nvfp4_glu(wg, wu, row_bytes as u64, xq, y, k as i32, n_rows as i32,
            n_tokens as i32, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (wg, wu, xq, y, n_rows, n_tokens, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Rewrites NVFP4 run scales of 0x7F -- ggml's "zero", the hardware's NaN --
/// to 0, over `n_blocks` blocks in place. Same weights to the int8 decoders.
pub fn nvfp4_sanitize(w: *mut c_void, n_blocks: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: w holds n_blocks 36-byte NVFP4 blocks.
    unsafe {
        ffi::check(airun_nvfp4_sanitize(w, n_blocks as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (w, n_blocks, stream);
        Err(CudaError::NotBuiltWithCuda)
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

/// Copies `rows` runs of `width` bytes from `src` (rows `spitch` bytes apart)
/// to `dst` (rows `dpitch` apart) on one device: placing two row blocks side
/// by side, or taking columns out of one.
#[allow(clippy::too_many_arguments)]
pub fn copy_2d(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize,
    rows: usize, stream: &Stream) -> Result<()> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    // SAFETY: the caller guarantees both regions hold `rows` rows at their pitches.
    ffi::check(unsafe {
        ffi::airun_memcpy2d_d2d(dst, dpitch as u64, src, spitch as u64, width as u64, rows as u64, stream.raw())
    })
}

/// RoPE frequency parameters, laid out as `airun_rope_params` in airun.cuh.
///
/// Plain RoPE multiplies the position by `freq_scale`. YaRN (`ext_factor` not
/// zero) mixes interpolated and extrapolated frequencies per dimension pair
/// and scales cos/sin by `mscale`; see [`RopeParams::yarn`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParams {
    pub base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub mscale: f32,
    pub corr_lo: f32,
    pub corr_hi: f32,
    /// Device address of an i32 subtracted from the KV slot to get the
    /// rotation position (text after an image), or 0. An address, not a
    /// pointer, so the parameters stay Send; see airun.cuh.
    pub shift: u64,
    /// Device address of [3][n_tok] i32 (t, row, col) positions for an
    /// image's tokens, split over the rotary pairs by `sections`, or 0.
    pub mpos: u64,
    pub n_tok: i32,
    /// Interleaved M-RoPE sections, in rotary pairs (`rope.dimension_sections`).
    pub sections: [i32; 3],
}

impl RopeParams {
    /// Unscaled RoPE, or linear position interpolation when `freq_scale < 1`.
    pub fn plain(base: f32, freq_scale: f32) -> Self {
        Self { base, freq_scale, ext_factor: 0.0, mscale: 1.0, corr_lo: 0.0, corr_hi: 0.0, shift: 0, mpos: 0,
               n_tok: 0, sections: [0; 3] }
    }

    /// These parameters with the rotation position taken from `shift` and,
    /// for an image's tokens, `mpos` (see the fields).
    pub fn positioned(mut self, shift: u64, mpos: Option<(u64, usize)>) -> Self {
        self.shift = shift;
        (self.mpos, self.n_tok) = mpos.map_or((0, 0), |(p, n)| (p, n as i32));
        self
    }

    /// YaRN over `rope_dim` rotary dimensions trained at `orig_ctx` positions.
    ///
    /// The correction range and the magnitude factor follow llama.cpp
    /// (`ggml_rope_yarn_corr_dims`, and `llama_context`'s attn_factor, which
    /// cancels its own get_mscale against the kernel's so that the net factor
    /// is `attn_factor * (1 + 0.1 ln factor)`), which in turn match
    /// HuggingFace's `_compute_yarn_parameters` with `truncate=True`.
    pub fn yarn(base: f32, factor: f32, rope_dim: usize, orig_ctx: usize, beta_fast: f32,
        beta_slow: f32, attn_factor: f32) -> Self {
        let corr_dim = |n_rot: f32| {
            rope_dim as f32 * (orig_ctx as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let lo = corr_dim(beta_fast).floor().max(0.0);
        let hi = corr_dim(beta_slow).ceil().min(rope_dim as f32 - 1.0);
        Self {
            base,
            freq_scale: 1.0 / factor,
            ext_factor: 1.0,
            mscale: attn_factor * (1.0 + 0.1 * factor.ln()),
            corr_lo: lo,
            corr_hi: hi,
            ..Self::plain(base, 1.0)
        }
    }

    /// The angle's (cos, sin) for one dimension pair, as `rope_cos_sin` in
    /// airun.cuh computes it. The CPU oracle for the kernels.
    pub fn cos_sin(&self, pair: usize, rope_dim: usize, pos: i32) -> (f32, f32) {
        let inv_freq = self.base.powf(-2.0 * pair as f32 / rope_dim as f32);
        let theta = if self.ext_factor == 0.0 {
            (pos as f32 * self.freq_scale) * inv_freq
        } else {
            let extrap = pos as f32 * inv_freq;
            let interp = self.freq_scale * extrap;
            let y = (pair as f32 - self.corr_lo) / (self.corr_hi - self.corr_lo).max(0.001);
            let ramp = (1.0 - y.clamp(0.0, 1.0)) * self.ext_factor;
            interp * (1.0 - ramp) + extrap * ramp
        };
        let (s, c) = theta.sin_cos();
        (c * self.mscale, s * self.mscale)
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
    params: RopeParams,
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
            params,
            interleaved as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (x, positions, n_tokens, n_heads, params, interleaved, stream);
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
/// `rope` is (rope_dim, params, interleaved).
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_store(q: *mut c_void, k: *const c_void, v: *const c_void,
    q_norm: *const c_void, k_norm: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void,
    positions: *const c_void, n_tokens: usize, n_heads: usize, n_kv_heads: usize,
    head_dim: usize, max_seq: usize, eps: f32, rope: Option<(usize, RopeParams, bool)>,
    kv_type: KvType, stream: &Stream) -> Result<()> {
    if head_dim % 32 != 0 || head_dim > 1024 {
        return Err(bad_arg(format!("the fused attention prologue needs head_dim divisible by 32, got {head_dim}")));
    }
    let (rope_dim, params, interleaved) = rope.unwrap_or((2, RopeParams::plain(1.0, 1.0), false));
    #[cfg(airun_cuda)]
    // SAFETY: q/k/v hold n_tokens rows of their head counts, the caches
    // n_kv_heads * max_seq rows in kv_type's format; positions outside
    // [0, max_seq) are skipped by the kernel.
    unsafe {
        ffi::check(airun_qk_norm_rope_store(q, k, v, q_norm, k_norm, k_cache, v_cache, positions,
            n_tokens as i32, n_heads as i32, n_kv_heads as i32, head_dim as i32, max_seq as i32,
            eps, rope_dim as i32, params, interleaved as i32, rope.is_some() as i32,
            kv_type.id(), stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (q, k, v, q_norm, k_norm, k_cache, v_cache, positions, n_tokens, n_heads,
            n_kv_heads, max_seq, eps, rope_dim, params, interleaved, kv_type, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
/// For a KV cache kept in RAM: with `to_stage`, copies the cells of
/// `[0, max position + 1)` of every head from `src` (the cache in RAM) into
/// `dst` (its VRAM copy); otherwise the cells at `positions` from `src` (the
/// VRAM copy) back to `dst`. Both are `[head][max_seq][row_bytes]`, the
/// positions are read on the device, and `row_bytes` must be a multiple of 4.
#[allow(clippy::too_many_arguments)]
pub fn kv_stage(
    dst: *mut c_void,
    src: *const c_void,
    positions: *const c_void,
    n_tokens: usize,
    n_kv_heads: usize,
    max_seq: usize,
    row_bytes: usize,
    to_stage: bool,
    stream: &Stream,
) -> Result<()> {
    if row_bytes % 4 != 0 {
        return Err(bad_arg(format!(
            "a KV cache in RAM needs rows of a multiple of 4 bytes, got {row_bytes}; use --alloc gpu              (everything in VRAM) or another --kv-type"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: both buffers hold n_kv_heads * max_seq rows of row_bytes; the
    // kernel clamps every cell to [0, max_seq).
    unsafe {
        ffi::check(airun_kv_stage(dst, src, positions, n_tokens as i32, n_kv_heads as i32, max_seq as i32,
            row_bytes as i32, to_stage as i32, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (dst, src, positions, n_tokens, n_kv_heads, max_seq, row_bytes, to_stage, stream);
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
    attention_split_sel(out, partial, q, k, v, positions, heads, kv_heads, dim, ctx, scale, window, splits,
        kv_type, std::ptr::null(), stream)
}

/// `rows` queries over one cache (a speculative verify's tokens), query r at
/// `positions[r]` and causal up to it, split across the cache as one-token
/// decode is. `partial` holds `rows` x heads x splits x (dim + 2) floats.
/// Takes the shapes [`attention_split_rows_ok`] accepts.
#[allow(clippy::too_many_arguments)]
pub fn attention_split_multi(out: *mut c_void, partial: *mut c_void, q: *const c_void, k: *const c_void,
    v: *const c_void, positions: *const c_void, heads: usize, kv_heads: usize, dim: usize, ctx: usize,
    scale: f32, window: usize, splits: usize, kv_type: KvType, rows: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: the caches hold kv_heads * ctx rows; q rows * heads * dim floats,
    // positions rows ints, partial as documented; the kernels stay inside them.
    unsafe {
        ffi::check(airun_attention_split_multi(out, partial, q, k, v, positions, heads as i32, kv_heads as i32,
            dim as i32, ctx as i32, scale, window as i32, splits as i32, kv_type.id(), rows as i32, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, partial, q, k, v, positions, heads, kv_heads, dim, ctx, scale, window, splits, kv_type, rows, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Whether [`attention_split_rows`] takes these shapes (the GQA decode kernel's).
pub fn attention_split_rows_ok(heads: usize, kv_heads: usize, dim: usize) -> bool {
    kv_heads > 0 && heads % kv_heads == 0 && heads / kv_heads <= 8 && matches!(dim, 64 | 128 | 256)
        && std::env::var("AIRUN_NO_GQA_DECODE").is_err()
}

/// [`attention_split`] for one token of each of `rows` conversations: row r's
/// query, output and position are row r, its cache `kv_rows[r]` (K) and
/// `kv_rows[rows + r]` (V), a device array of pointers. `partial` must hold
/// `rows` times the one-row scratch. One launch pair for a --parallel step.
#[allow(clippy::too_many_arguments)]
pub fn attention_split_rows(out: *mut c_void, partial: &crate::DeviceBuffer<f32>, q: *const c_void,
    kv_rows: *const c_void, positions: *const c_void, heads: usize, kv_heads: usize, dim: usize, ctx: usize,
    scale: f32, window: usize, splits: usize, kv_type: KvType, rows: usize, stream: &Stream) -> Result<()> {
    if !attention_split_rows_ok(heads, kv_heads, dim) || splits == 0 || splits > 256
        || partial.len() < rows * heads * splits * (dim + 2) {
        return Err(bad_arg("invalid batched split attention dimensions or scratch"));
    }
    #[cfg(airun_cuda)]
    // SAFETY: shapes and scratch checked above; the pointer table and the
    // caches it names outlive the stream's work, as for attention_split.
    unsafe { ffi::check(airun_attention_split_rows(out, partial.as_mut_ptr(), q, kv_rows as *const *const c_void,
        positions, heads as i32, kv_heads as i32, dim as i32, ctx as i32, scale, window as i32, splits as i32,
        kv_type.id(), rows as i32, stream.raw())) }
    #[cfg(not(airun_cuda))]
    { let _ = (out, q, kv_rows, positions, ctx, scale, window, kv_type, rows, stream); Err(CudaError::NotBuiltWithCuda) }
}

/// [`attention_split`] over the cells a QSA selection names: `sel` is
/// `[count - 1, cache row...]` (qsa_select's list) and replaces the position.
#[allow(clippy::too_many_arguments)]
pub fn attention_split_sel(out: *mut c_void, partial: &crate::DeviceBuffer<f32>,
    q: *const c_void, k: *const c_void, v: *const c_void, positions: *const c_void,
    heads: usize, kv_heads: usize, dim: usize, ctx: usize, scale: f32,
    window: usize, splits: usize, kv_type: KvType, sel: *const c_void, stream: &Stream) -> Result<()> {
    if dim == 0 || dim > 256 || dim % 32 != 0 || kv_heads == 0 || heads % kv_heads != 0
        || splits == 0 || splits > 256 || partial.len() < heads * splits * (dim + 2) {
        return Err(bad_arg("invalid split attention dimensions or scratch; resize the attention buffers"));
    }
    #[cfg(airun_cuda)]
    // SAFETY: scratch and shape are checked above; runtime owns the Q/K/V and
    // position buffers until the stream completes, just as for attention().
    unsafe { ffi::check(airun_attention_split(out, partial.as_mut_ptr(), q, k, v,
        positions, heads as i32, kv_heads as i32, dim as i32, ctx as i32,
        scale, window as i32, splits as i32, kv_type.id(), sel, stream.raw())) }
    #[cfg(not(airun_cuda))]
    { let _ = (out, q, k, v, positions, ctx, scale, window, kv_type, sel, stream); Err(CudaError::NotBuiltWithCuda) }
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
    attention_masked(out, q, k_cache, v_cache, positions, n_tokens, n_heads, n_kv_heads, head_dim, max_seq,
        scale, sliding_window, kv_type, None, 0, stream)
}

/// [`attention`] where query t sees key j only if bit j of `mask` row t is set
/// (`mask` = (words, words per row), qsa_select's mask). `keys` is how many
/// keys the batch's last query sees: from 4096 on, the prefill kernel sums
/// P.V in fp16 per key tile (at most 32 terms, added to an fp32 total), 20%
/// faster at long context with no measurable loss; below it, and with 0, the
/// sums stay fp32.
#[allow(clippy::too_many_arguments)]
pub fn attention_masked(
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
    mask: Option<(*const c_void, usize)>,
    keys: usize,
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
            mask.map_or(std::ptr::null(), |m| m.0),
            mask.map_or(0, |m| m.1 as i32),
            keys.min(i32::MAX as usize) as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, q, k_cache, v_cache, positions, n_tokens, n_heads, max_seq, scale, sliding_window, kv_type, mask, keys, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

#[cfg(test)]
mod tests {
    use super::RopeParams;

    /// Qwen3.8-27B's yarn1m file, worked by hand from llama.cpp's
    /// ggml_rope_yarn_corr_dims: 64 * ln(262144 / (32 * 2pi)) / (2 ln 1e7) =
    /// 14.24 and, for beta 1, 21.12; floor and ceil give [14, 22].
    #[test]
    fn yarn_matches_llama_cpp_for_qwen38_27b() {
        let p = RopeParams::yarn(1e7, 4.0, 64, 262_144, 32.0, 1.0, 1.0);
        assert_eq!((p.corr_lo, p.corr_hi), (14.0, 22.0));
        assert_eq!(p.freq_scale, 0.25);
        assert!((p.mscale - 1.138_629_4).abs() < 1e-6);
        // Fast pairs are not interpolated, slow ones are fully interpolated.
        let (c0, _) = p.cos_sin(0, 64, 1000);
        assert!((c0 - (1000f32).cos() * p.mscale).abs() < 1e-4);
        let slow = 1e7f32.powf(-2.0 * 30.0 / 64.0);
        let (c30, _) = p.cos_sin(30, 64, 1000);
        assert!((c30 - (1000.0 * slow * 0.25).cos() * p.mscale).abs() < 1e-5);
    }

    #[test]
    fn plain_rope_is_unscaled() {
        let p = RopeParams::plain(1e6, 1.0);
        let (c, s) = p.cos_sin(0, 128, 5);
        assert_eq!((c, s), (5f32.cos(), 5f32.sin()));
    }
}
