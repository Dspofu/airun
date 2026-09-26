//! Kernels for hybrid SSM/attention models and mixture-of-experts layers.
//!
//! These back the Qwen3-Next family: linear-attention layers whose state is a
//! fixed-size matrix per head, gated softmax-attention layers, and MoE feed
//! forwards that run only the experts a token routes to.

use std::ffi::c_void;

use crate::error::{CudaError, Result};
use crate::ffi::{self, Status};
use crate::ops::KernelType;
use crate::stream::Stream;

#[cfg(airun_cuda)]
unsafe extern "C" {
    fn airun_causal_conv1d(
        x: *mut c_void,
        weight: *const c_void,
        state: *mut c_void,
        n_tokens: i32,
        channels: i32,
        kernel: i32,
        apply_silu: i32,
        reverse: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_l2norm(x: *mut c_void, rows: i32, head_dim: i32, eps: f32, stream: ffi::Stream) -> Status;

    fn airun_delta_gates(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        decay_out: *mut c_void,
        beta_out: *mut c_void,
        n_tokens: i32,
        n_v_heads: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_delta_rule(
        state: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        decay: *const c_void,
        beta: *const c_void,
        out: *mut c_void,
        n_tokens: i32,
        n_k_heads: i32,
        n_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_rmsnorm_gated(
        x: *mut c_void,
        weight: *const c_void,
        gate: *const c_void,
        n: i32,
        rows: i32,
        eps: f32,
        gate_first: i32,
        sigmoid_gate: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_split_head_gate(
        src: *const c_void,
        value: *mut c_void,
        gate: *mut c_void,
        n_tokens: i32,
        n_heads: i32,
        head_dim: i32,
        interleaved: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_sigmoid_gate(out: *mut c_void, gate: *const c_void, n: i64, stream: ffi::Stream) -> Status;

    fn airun_split_qkv_grouped(
        src: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        n_tokens: i32,
        n_k_heads: i32,
        head_k: i32,
        head_v: i32,
        v_per_k: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_slice_columns(
        src: *const c_void,
        dst: *mut c_void,
        n_tokens: i32,
        src_width: i32,
        offset: i32,
        width: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_moe_router(
        logits: *const c_void,
        out_ids: *mut c_void,
        out_weights: *mut c_void,
        n_tokens: i32,
        n_experts: i32,
        top_k: i32,
        normalize: i32,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_moe_gate_up(
        type_id: i32,
        gate_w: *const c_void,
        up_w: *const c_void,
        x: *const c_void,
        ids: *const c_void,
        h: *mut c_void,
        k: i32,
        ff: i32,
        n_tokens: i32,
        top_k: i32,
        row_bytes: u64,
        expert_stride: u64,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_moe_down(
        type_id: i32,
        down_w: *const c_void,
        h: *const c_void,
        ids: *const c_void,
        weights: *const c_void,
        y: *mut c_void,
        ff: i32,
        n_embd: i32,
        n_tokens: i32,
        top_k: i32,
        row_bytes: u64,
        expert_stride: u64,
        stream: ffi::Stream,
    ) -> Status;

    fn airun_sigmoid_scale_rows(
        x: *mut c_void,
        logit: *const c_void,
        n_tokens: i32,
        width: i32,
        stream: ffi::Stream,
    ) -> Status;
}

/// Every entry point degrades to the same error without compiled kernels.
macro_rules! needs_cuda {
    ($($unused:expr),* $(,)?) => {{
        $( let _ = $unused; )*
        return Err(CudaError::NotBuiltWithCuda);
    }};
}

fn bad_arg(what: impl Into<String>) -> CudaError {
    CudaError::Runtime {
        code: -1,
        name: "AIRUN_BAD_ARG".into(),
        message: what.into(),
    }
}

/// Causal depthwise convolution over `channels`, updating `state` in place.
///
/// `state` holds the previous `kernel - 1` values per channel, which is what
/// makes single-token decoding produce the same result as a full-sequence pass.
///
/// `reverse` flips which end of the tap window the current token multiplies --
/// the difference between a convolution and a correlation, which no shape check
/// can tell apart.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d(
    x: *mut c_void,
    weight: *const c_void,
    state: *mut c_void,
    n_tokens: usize,
    channels: usize,
    kernel: usize,
    apply_silu: bool,
    reverse: bool,
    stream: &Stream,
) -> Result<()> {
    if !(1..=8).contains(&kernel) {
        return Err(bad_arg(format!(
            "convolution kernel width {kernel} is outside the supported 1..=8"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `x` holds n_tokens * channels floats, `weight` channels * kernel,
    // and `state` channels * (kernel - 1).
    unsafe {
        ffi::check(airun_causal_conv1d(
            x,
            weight,
            state,
            n_tokens as i32,
            channels as i32,
            kernel as i32,
            apply_silu as i32,
            reverse as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(x, weight, state, n_tokens, channels, apply_silu, reverse, stream)
}

/// Normalizes each of `rows` vectors of `head_dim` to unit length.
pub fn l2norm(x: *mut c_void, rows: usize, head_dim: usize, eps: f32, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `x` holds rows * head_dim floats.
    unsafe {
        ffi::check(airun_l2norm(x, rows as i32, head_dim as i32, eps, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(x, rows, head_dim, eps, stream)
}

/// Computes the per-token decay and beta coefficients of the delta rule.
#[allow(clippy::too_many_arguments)]
pub fn delta_gates(
    a: *const c_void,
    b: *const c_void,
    a_log: *const c_void,
    dt_bias: *const c_void,
    decay_out: *mut c_void,
    beta_out: *mut c_void,
    n_tokens: usize,
    n_v_heads: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `a` and `b` hold n_tokens * n_v_heads floats; `a_log` and
    // `dt_bias` hold n_v_heads; the outputs match `a`.
    unsafe {
        ffi::check(airun_delta_gates(
            a,
            b,
            a_log,
            dt_bias,
            decay_out,
            beta_out,
            n_tokens as i32,
            n_v_heads as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(a, b, a_log, dt_bias, decay_out, beta_out, n_tokens, n_v_heads, stream)
}

/// The gated delta rule over `n_tokens`, carrying `state` forward.
///
/// The token loop lives inside the kernel because the recurrence is sequential;
/// parallelism comes from the heads and the state dimensions.
#[allow(clippy::too_many_arguments)]
pub fn delta_rule(
    state: *mut c_void,
    q: *const c_void,
    k: *const c_void,
    v: *const c_void,
    decay: *const c_void,
    beta: *const c_void,
    out: *mut c_void,
    n_tokens: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    stream: &Stream,
) -> Result<()> {
    if n_k_heads == 0 || n_v_heads % n_k_heads != 0 {
        return Err(bad_arg(format!(
            "value head count {n_v_heads} must be a multiple of key head count {n_k_heads}"
        )));
    }
    if head_v_dim > 1024 || head_k_dim > 1024 {
        return Err(bad_arg(format!(
            "delta rule supports head dimensions up to 1024, got k={head_k_dim} v={head_v_dim}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `state` holds n_v_heads * head_k_dim * head_v_dim floats; q/k are
    // n_tokens * n_k_heads * head_k_dim; v and out are n_tokens * n_v_heads * head_v_dim.
    unsafe {
        ffi::check(airun_delta_rule(
            state,
            q,
            k,
            v,
            decay,
            beta,
            out,
            n_tokens as i32,
            n_k_heads as i32,
            n_v_heads as i32,
            head_k_dim as i32,
            head_v_dim as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(state, q, k, v, decay, beta, out, n_tokens, stream)
}

/// RMSNorm over the last axis, fused with a SiLU gate.
///
/// `gate_first` selects where the gate lands. The reference formulation gates
/// before measuring the row, so the normalization divides the gate's own scale
/// back out; gating afterwards instead leaves it in the output. Both are
/// implemented because they are indistinguishable from shapes alone.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated(
    x: *mut c_void,
    weight: *const c_void,
    gate: *const c_void,
    n: usize,
    rows: usize,
    eps: f32,
    gate_first: bool,
    sigmoid_gate: bool,
    stream: &Stream,
) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `x` and `gate` hold rows * n floats; `weight` holds n.
    unsafe {
        ffi::check(airun_rmsnorm_gated(
            x,
            weight,
            gate,
            n as i32,
            rows as i32,
            eps,
            gate_first as i32,
            sigmoid_gate as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(x, weight, gate, n, rows, eps, gate_first, sigmoid_gate, stream)
}

/// Splits a packed value/gate projection.
///
/// `interleaved` selects the layout: per head (`[head][value|gate]`, the
/// HuggingFace view) or blocked (`[all values][all gates]`).
#[allow(clippy::too_many_arguments)]
pub fn split_head_gate(
    src: *const c_void,
    value: *mut c_void,
    gate: *mut c_void,
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    interleaved: bool,
    stream: &Stream,
) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `src` holds n_tokens * n_heads * 2 * head_dim floats; each output
    // holds half that, under either layout.
    unsafe {
        ffi::check(airun_split_head_gate(
            src,
            value,
            gate,
            n_tokens as i32,
            n_heads as i32,
            head_dim as i32,
            interleaved as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(src, value, gate, n_tokens, n_heads, head_dim, interleaved, stream)
}

/// `out *= sigmoid(gate)`, element-wise.
pub fn sigmoid_gate(out: *mut c_void, gate: *const c_void, n: usize, stream: &Stream) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: both buffers hold at least `n` floats.
    unsafe {
        ffi::check(airun_sigmoid_gate(out, gate, n as i64, stream.raw()))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(out, gate, n, stream)
}

/// Extracts q, k and v from a projection grouped by key head.
///
/// Each key head's slice holds its queries, its keys, and the value heads that
/// share it, in that order.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_grouped(
    src: *const c_void,
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    n_tokens: usize,
    n_k_heads: usize,
    head_k: usize,
    head_v: usize,
    v_per_k: usize,
    stream: &Stream,
) -> Result<()> {
    if n_k_heads == 0 || v_per_k == 0 {
        return Err(bad_arg("head counts must be non-zero"));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `src` holds n_tokens * n_k_heads * (2*head_k + v_per_k*head_v)
    // floats; the destinations together hold exactly the same count.
    unsafe {
        ffi::check(airun_split_qkv_grouped(
            src,
            q,
            k,
            v,
            n_tokens as i32,
            n_k_heads as i32,
            head_k as i32,
            head_v as i32,
            v_per_k as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(src, q, k, v, n_tokens, n_k_heads, head_k, head_v, v_per_k, stream)
}

/// Copies a column range out of each row of a wider buffer.
pub fn slice_columns(
    src: *const c_void,
    dst: *mut c_void,
    n_tokens: usize,
    src_width: usize,
    offset: usize,
    width: usize,
    stream: &Stream,
) -> Result<()> {
    if offset + width > src_width {
        return Err(bad_arg(format!(
            "slice [{offset}, {}) does not fit a row of {src_width}",
            offset + width
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: bounds checked above; `dst` holds n_tokens * width floats.
    unsafe {
        ffi::check(airun_slice_columns(
            src,
            dst,
            n_tokens as i32,
            src_width as i32,
            offset as i32,
            width as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(src, dst, n_tokens, src_width, offset, width, stream)
}

/// Softmax over expert logits, then the top-k experts and their weights.
#[allow(clippy::too_many_arguments)]
pub fn moe_router(
    logits: *const c_void,
    out_ids: *mut c_void,
    out_weights: *mut c_void,
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    normalize: bool,
    stream: &Stream,
) -> Result<()> {
    if top_k == 0 || top_k > n_experts {
        return Err(bad_arg(format!(
            "top_k {top_k} must be between 1 and the expert count {n_experts}"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `logits` holds n_tokens * n_experts floats; the outputs hold
    // n_tokens * top_k each.
    unsafe {
        ffi::check(airun_moe_router(
            logits,
            out_ids,
            out_weights,
            n_tokens as i32,
            n_experts as i32,
            top_k as i32,
            normalize as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(logits, out_ids, out_weights, n_tokens, n_experts, top_k, normalize, stream)
}

/// `h = silu(gate_e @ x) * (up_e @ x)` for each token's selected experts.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up(
    ty: KernelType,
    gate_w: *const c_void,
    up_w: *const c_void,
    x: *const c_void,
    ids: *const c_void,
    h: *mut c_void,
    k: usize,
    ff: usize,
    n_tokens: usize,
    top_k: usize,
    row_bytes: usize,
    expert_stride: usize,
    stream: &Stream,
) -> Result<()> {
    if k % 32 != 0 {
        return Err(bad_arg(format!(
            "expert inner dimension {k} must be divisible by 32"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `h` holds n_tokens * top_k * ff floats; expert weights are indexed
    // by ids, which the router constrained to [0, n_experts).
    unsafe {
        ffi::check(airun_moe_gate_up(
            ty.0,
            gate_w,
            up_w,
            x,
            ids,
            h,
            k as i32,
            ff as i32,
            n_tokens as i32,
            top_k as i32,
            row_bytes as u64,
            expert_stride as u64,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(gate_w, up_w, x, ids, h, ff, n_tokens, top_k, row_bytes, expert_stride, stream)
}

/// `y += sum_k weight_k * (down_e @ h_k)`, accumulating into `y`.
#[allow(clippy::too_many_arguments)]
pub fn moe_down(
    ty: KernelType,
    down_w: *const c_void,
    h: *const c_void,
    ids: *const c_void,
    weights: *const c_void,
    y: *mut c_void,
    ff: usize,
    n_embd: usize,
    n_tokens: usize,
    top_k: usize,
    row_bytes: usize,
    expert_stride: usize,
    stream: &Stream,
) -> Result<()> {
    if ff % 32 != 0 {
        return Err(bad_arg(format!(
            "expert feed-forward width {ff} must be divisible by 32"
        )));
    }
    #[cfg(airun_cuda)]
    // SAFETY: `y` holds n_tokens * n_embd floats and is accumulated into, not
    // overwritten, so the caller must have initialized it.
    unsafe {
        ffi::check(airun_moe_down(
            ty.0,
            down_w,
            h,
            ids,
            weights,
            y,
            ff as i32,
            n_embd as i32,
            n_tokens as i32,
            top_k as i32,
            row_bytes as u64,
            expert_stride as u64,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(down_w, h, ids, weights, y, n_embd, n_tokens, top_k, row_bytes, expert_stride, stream)
}

/// Scales each token's row by `sigmoid(logit[token])`.
pub fn sigmoid_scale_rows(
    x: *mut c_void,
    logit: *const c_void,
    n_tokens: usize,
    width: usize,
    stream: &Stream,
) -> Result<()> {
    #[cfg(airun_cuda)]
    // SAFETY: `x` holds n_tokens * width floats; `logit` holds n_tokens.
    unsafe {
        ffi::check(airun_sigmoid_scale_rows(
            x,
            logit,
            n_tokens as i32,
            width as i32,
            stream.raw(),
        ))
    }
    #[cfg(not(airun_cuda))]
    needs_cuda!(x, logit, n_tokens, width, stream)
}
