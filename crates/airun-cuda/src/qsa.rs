//! QSA sparse attention (qwen4exp); see `kernels/qsa.cu`. The indexer's raw
//! keys wait in a ring until their block completes, then pool into one key per
//! block; each query scores the blocks it sees and selects the cells it
//! attends to, as a list (decode) or a bit mask (prefill) for the attention
//! kernels.

use std::ffi::c_void;

#[cfg(not(airun_cuda))]
use crate::error::CudaError;
use crate::error::Result;
use crate::ffi::{self, Status};
use crate::ops::RopeParams;
use crate::stream::Stream;

#[cfg(airun_cuda)]
unsafe extern "C" {
    fn airun_qsa_store(ring: *mut c_void, k: *const c_void, pos: *const c_void, n: i32, dim: i32, ring_len: i32,
        stream: ffi::Stream) -> Status;
    fn airun_qsa_pool(pooled: *mut c_void, ring: *const c_void, norm: *const c_void, pos: *const c_void, n: i32,
        dim: i32, ratio: i32, ring_len: i32, max_blocks: i32, eps: f32, rope_dim: i32, rope: RopeParams,
        stream: ffi::Stream) -> Status;
    fn airun_qsa_score(scores: *mut c_void, stride: i32, q: *const c_void, pooled: *const c_void,
        pos: *const c_void, t0: i32, n_queries: i32, heads: i32, dim: i32, ratio: i32, width: i32,
        max_blocks: i32, stream: ffi::Stream) -> Status;
    fn airun_qsa_select(scores: *const c_void, stride: i32, pos: *const c_void, t0: i32, n_queries: i32,
        ratio: i32, width: i32, list: *mut c_void, mask: *mut c_void, words: i32, stream: ffi::Stream) -> Status;
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

/// `ring[pos[t] % ring_len] = k[t]` as f16: ring [ring_len][dim] halves, k [n][dim].
pub fn store(ring: *mut c_void, k: *const c_void, pos: *const c_void, n: usize, dim: usize, ring_len: usize,
    stream: &Stream) -> Result<()> {
    call!(airun_qsa_store(ring, k, pos, n as i32, dim as i32, ring_len as i32, stream.raw()),
        ring, k, pos, n, dim, ring_len, stream)
}

/// Pools the blocks of `ratio` the n consecutive positions completed into
/// `pooled` ([max_blocks][dim]): mean, RMSNorm by `norm`, RoPE at the block's
/// first position. `ring_len` must exceed n + ratio.
#[allow(clippy::too_many_arguments)]
pub fn pool(pooled: *mut c_void, ring: *const c_void, norm: *const c_void, pos: *const c_void, n: usize,
    dim: usize, ratio: usize, ring_len: usize, max_blocks: usize, eps: f32, rope_dim: usize, rope: RopeParams,
    stream: &Stream) -> Result<()> {
    call!(airun_qsa_pool(pooled, ring, norm, pos, n as i32, dim as i32, ratio as i32, ring_len as i32,
        max_blocks as i32, eps, rope_dim as i32, rope, stream.raw()),
        pooled, ring, norm, pos, n, dim, ratio, ring_len, max_blocks, eps, rope_dim, rope, stream)
}

/// Block scores of queries t0..t0+n_queries ([n_queries][stride] f32) from their
/// indexer queries `q` ([n][heads][dim]); queries whose context fits `width`
/// are skipped.
#[allow(clippy::too_many_arguments)]
pub fn score(scores: *mut c_void, stride: usize, q: *const c_void, pooled: *const c_void, pos: *const c_void,
    t0: usize, n_queries: usize, heads: usize, dim: usize, ratio: usize, width: usize, max_blocks: usize,
    stream: &Stream) -> Result<()> {
    call!(airun_qsa_score(scores, stride as i32, q, pooled, pos, t0 as i32, n_queries as i32, heads as i32,
        dim as i32, ratio as i32, width as i32, max_blocks as i32, stream.raw()),
        scores, stride, q, pooled, pos, t0, n_queries, heads, dim, ratio, width, max_blocks, stream)
}

/// The cells queries t0..t0+n_queries attend to, from their scores: into
/// `list` ([1 + width] i32, one query only) and/or rows of `mask` ([n][words]).
#[allow(clippy::too_many_arguments)]
pub fn select(scores: *const c_void, stride: usize, pos: *const c_void, t0: usize, n_queries: usize, ratio: usize,
    width: usize, list: *mut c_void, mask: *mut c_void, words: usize, stream: &Stream) -> Result<()> {
    call!(airun_qsa_select(scores, stride as i32, pos, t0 as i32, n_queries as i32, ratio as i32, width as i32,
        list, mask, words as i32, stream.raw()),
        scores, stride, pos, t0, n_queries, ratio, width, list, mask, words, stream)
}
