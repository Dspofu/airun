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
    stream: &Stream,
) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: the caches hold n_kv_heads * max_seq * head_dim halves; the kernel
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
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (k_cache, v_cache, k, v, positions, n_tokens, n_kv_heads, head_dim, max_seq, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Causal attention against the KV cache.
///
/// `sliding_window` of 0 means full causal attention.
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
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, q, k_cache, v_cache, positions, n_tokens, n_heads, max_seq, scale, sliding_window, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
