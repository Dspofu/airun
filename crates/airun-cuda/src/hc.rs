//! Hyper-connection and per-layer n-gram embedding kernels (qwen4exp); see
//! `kernels/hc.cu`. Every buffer is f32 and laid out [token][stream][width].

use std::ffi::c_void;

#[cfg(not(airun_cuda))]
use crate::error::CudaError;
use crate::error::Result;
use crate::ffi::{self, Status};
use crate::stream::Stream;

#[cfg(airun_cuda)]
unsafe extern "C" {
    fn airun_rmsnorm_grouped(out: *mut c_void, x: *const c_void, w: *const c_void, d: i32, n_rows: i32,
        group: i32, eps: f32, stream: ffi::Stream) -> Status;
    fn airun_hc_pre(out: *mut c_void, xn: *const c_void, gate: *const c_void, n_tokens: i32, hc: i32, d: i32,
        stream: ffi::Stream) -> Status;
    fn airun_hc_expand(x: *mut c_void, e: *const c_void, n_tokens: i32, hc: i32, d: i32,
        stream: ffi::Stream) -> Status;
    fn airun_silu_scale(x: *mut c_void, n: i32, scale: f32, stream: ffi::Stream) -> Status;
    fn airun_hc_post(x: *mut c_void, b: *const c_void, inject: *const c_void, n_tokens: i32, hc: i32, d: i32,
        stream: ffi::Stream) -> Status;
    fn airun_ple_gate(gated: *mut c_void, key: *const c_void, query: *const c_void, value: *const c_void,
        n_tokens: i32, hc: i32, d: i32, stream: ffi::Stream) -> Status;
    fn airun_ple_conv(x: *mut c_void, gated: *const c_void, xn: *const c_void, hist: *mut c_void,
        scratch: *mut c_void, w: *const c_void, n_tokens: i32, ch: i32, kernel: i32, dil: i32,
        stream: ffi::Stream) -> Status;
}

macro_rules! call {
    ($e:expr, $($unused:expr),*) => {{
        #[cfg(airun_cuda)]
        // SAFETY: the callers' doc comments state the buffer sizes each kernel
        // reads and writes; every pointer comes from a live DeviceBuffer.
        unsafe { ffi::check($e) }
        #[cfg(not(airun_cuda))]
        {
            $( let _ = &$unused; )*
            Err(CudaError::NotBuiltWithCuda)
        }
    }};
}

/// RMSNorm of `n_rows` rows of `d`; row r scales by gamma `r % group` of `w`
/// ([group * d]).
pub fn rmsnorm_grouped(out: *mut c_void, x: *const c_void, w: *const c_void, d: usize, n_rows: usize,
    group: usize, eps: f32, stream: &Stream) -> Result<()> {
    call!(airun_rmsnorm_grouped(out, x, w, d as i32, n_rows as i32, group as i32, eps, stream.raw()),
        out, x, w, d, n_rows, group, eps, stream)
}

/// `out[t] = mean_c(xn[t][c] * sigmoid(gate[t][c]))`: out is [n][d], xn and
/// gate [n][hc][d].
pub fn hc_pre(out: *mut c_void, xn: *const c_void, gate: *const c_void, n_tokens: usize, hc: usize, d: usize,
    stream: &Stream) -> Result<()> {
    call!(airun_hc_pre(out, xn, gate, n_tokens as i32, hc as i32, d as i32, stream.raw()),
        out, xn, gate, n_tokens, hc, d, stream)
}

/// `x[t][c] = e[t]` for every stream c: x [n][hc][d], e [n][d].
pub fn hc_expand(x: *mut c_void, e: *const c_void, n_tokens: usize, hc: usize, d: usize, stream: &Stream)
    -> Result<()> {
    call!(airun_hc_expand(x, e, n_tokens as i32, hc as i32, d as i32, stream.raw()), x, e, n_tokens, hc, d, stream)
}

/// `x = silu(x * scale)` over `n` floats.
pub fn silu_scale(x: *mut c_void, n: usize, scale: f32, stream: &Stream) -> Result<()> {
    call!(airun_silu_scale(x, n as i32, scale, stream.raw()), x, n, scale, stream)
}

/// `x[t][c] += b[t] * 2 sigmoid(inject[t][c] / hc)`: x [n][hc][d], b [n][d],
/// inject [n][hc].
pub fn hc_post(x: *mut c_void, b: *const c_void, inject: *const c_void, n_tokens: usize, hc: usize, d: usize,
    stream: &Stream) -> Result<()> {
    call!(airun_hc_post(x, b, inject, n_tokens as i32, hc as i32, d as i32, stream.raw()),
        x, b, inject, n_tokens, hc, d, stream)
}

/// The PLE gate: gated [n][hc][d] from key and query [n][hc][d] and value [n][d].
pub fn ple_gate(gated: *mut c_void, key: *const c_void, query: *const c_void, value: *const c_void,
    n_tokens: usize, hc: usize, d: usize, stream: &Stream) -> Result<()> {
    call!(airun_ple_gate(gated, key, query, value, n_tokens as i32, hc as i32, d as i32, stream.raw()),
        gated, key, query, value, n_tokens, hc, d, stream)
}

/// `x += gated + silu(dilated causal conv of xn)`, all [n][ch]; `hist` holds
/// the previous (kernel-1)*dil rows of xn and is advanced, through `scratch`
/// of the same size.
#[allow(clippy::too_many_arguments)]
pub fn ple_conv(x: *mut c_void, gated: *const c_void, xn: *const c_void, hist: *mut c_void, scratch: *mut c_void,
    w: *const c_void, n_tokens: usize, ch: usize, kernel: usize, dil: usize, stream: &Stream) -> Result<()> {
    call!(airun_ple_conv(x, gated, xn, hist, scratch, w, n_tokens as i32, ch as i32, kernel as i32, dil as i32,
        stream.raw()), x, gated, xn, hist, scratch, w, n_tokens, ch, kernel, dil, stream)
}
