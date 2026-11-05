//! Vision-encoder ops (kernels/vision.cu): LayerNorm with bias, a per-column
//! bias, and the ViT's QKV split with its 2D rotary embedding.

use std::ffi::c_void;

#[cfg(not(airun_cuda))]
use crate::error::CudaError;
use crate::error::Result;
#[cfg(airun_cuda)]
use crate::ffi::{self, Status};
use crate::stream::Stream;

#[cfg(airun_cuda)]
unsafe extern "C" {
    fn airun_layernorm(out: *mut c_void, x: *const c_void, w: *const c_void, b: *const c_void, rows: i32, n: i32,
                       eps: f32, stream: ffi::Stream) -> Status;
    fn airun_add_bias(y: *mut c_void, bias: *const c_void, rows: i32, n: i32, stream: ffi::Stream) -> Status;
    fn airun_vision_qkv(q: *mut c_void, k: *mut c_void, v: *mut c_void, qkv: *const c_void, pos_y: *const c_void,
                        pos_x: *const c_void, n: i32, n_head: i32, hd: i32, hdp: i32, base: f32,
                        stream: ffi::Stream) -> Status;
    fn airun_vision_unpad(out: *mut c_void, input: *const c_void, rows: i32, hd: i32, hdp: i32,
                          stream: ffi::Stream) -> Status;
}

/// LayerNorm of `rows` rows of `n` floats: (x - mean) / sqrt(var + eps) * w + b.
/// `w` and `b` may be null (1 and 0).
#[allow(clippy::too_many_arguments)]
pub fn layernorm(out: *mut c_void, x: *const c_void, w: *const c_void, b: *const c_void, rows: usize, n: usize,
                 eps: f32, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: out and x hold rows * n floats, w and b n floats or are null.
    unsafe {
        ffi::check(airun_layernorm(out, x, w, b, rows as i32, n as i32, eps, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, x, w, b, rows, n, eps, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// y[r][i] += bias[i] for `rows` rows of `n`.
pub fn add_bias(y: *mut c_void, bias: *const c_void, rows: usize, n: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: y holds rows * n floats and bias n.
    unsafe {
        ffi::check(airun_add_bias(y, bias, rows as i32, n as i32, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (y, bias, rows, n, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// Splits the fused projection `qkv` ([n][3][n_head][hd] floats) into q
/// ([n][n_head][hdp] f32) and k, v ([n_head][n][hdp] f16, the KV-cache layout
/// of an n-position cache), rotating q and k by the 2D rotary embedding of
/// patch (pos_y, pos_x) and zero-filling columns hd..hdp.
#[allow(clippy::too_many_arguments)]
pub fn vision_qkv(q: *mut c_void, k: *mut c_void, v: *mut c_void, qkv: *const c_void, pos_y: *const c_void,
                  pos_x: *const c_void, n: usize, n_head: usize, hd: usize, hdp: usize, base: f32,
                  stream: &Stream) -> Result<()> {
    if hd % 4 != 0 || hdp < hd {
        return Err(crate::ops::bad_arg(format!(
            "vision_qkv needs a head width divisible by 4 and padding at least as wide, got {hd} -> {hdp}")));
    }
    #[cfg(airun_cuda)]
    // SAFETY: shapes as documented; the kernel writes only inside them.
    unsafe {
        ffi::check(airun_vision_qkv(q, k, v, qkv, pos_y, pos_x, n as i32, n_head as i32, hd as i32, hdp as i32,
            base, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (q, k, v, qkv, pos_y, pos_x, n, n_head, hd, hdp, base, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}

/// `rows` rows of `hdp` floats to rows of their first `hd`.
pub fn vision_unpad(out: *mut c_void, input: *const c_void, rows: usize, hd: usize, hdp: usize,
                    stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: input holds rows * hdp floats and out rows * hd.
    unsafe {
        ffi::check(airun_vision_unpad(out, input, rows as i32, hd as i32, hdp as i32, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    {
        let _ = (out, input, rows, hd, hdp, stream);
        Err(CudaError::NotBuiltWithCuda)
    }
}
