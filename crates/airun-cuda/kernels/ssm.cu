// Gated DeltaNet: the linear-attention layers of Qwen3-Next style hybrids.
//
// These layers replace softmax attention with a recurrent state of shape
// [head_k_dim, head_v_dim] per head. The state is a fixed size regardless of
// sequence length, which is why these models can advertise 262144-token
// contexts without a KV cache that grows with them.
//
// The update is the gated delta rule:
//
//     S_t = S_{t-1} * exp(g_t)                       decay
//     m_t = S_t^T k_t                                what the state predicts for k
//     S_t = S_t + k_t (v_t - m_t)^T * beta_t         correct toward v
//     o_t = S_t^T q_t                                read out with q
//
// Matches `torch_recurrent_gated_delta_rule` in HuggingFace transformers.

#include "airun.cuh"

// ---------------------------------------------------------------------------
// Causal depthwise convolution
// ---------------------------------------------------------------------------

// Short depthwise conv over the channel axis, mixing each channel with its own
// previous `kernel-1` values. `state` carries those values across calls so
// decoding one token at a time gives the same answer as a full-sequence pass.
//
// Weights are [kernel, channels] in ggml order, so channel c's taps are
// contiguous at w[c * kernel .. ].
__global__ void causal_conv1d_kernel(
    float* __restrict__ x,                 // [n_tokens][channels], updated in place
    const float* __restrict__ weight,      // [channels][kernel]
    float* __restrict__ state,             // [channels][kernel-1] ring of past values
    const int n_tokens,
    const int channels,
    const int kernel,
    const int apply_silu,
    const int reverse,
    const int keep_state,
    float* const* __restrict__ states = nullptr) {  // rows mode: row blockIdx.y, one token each

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    if (states) {
        state = states[blockIdx.y];
        x += (size_t)blockIdx.y * channels;
    }

    const int hist = kernel - 1;
    const float* w = weight + (size_t)c * kernel;
    float* st = state + (size_t)c * hist;

    // Which tap the current token gets. A convolution reads its window in the
    // opposite direction from a correlation, and a file can carry either
    // convention with the same shape; `reverse` picks the other one.
    const int cur_tap = reverse ? 0 : hist;

    // Past values, oldest first.
    float window[8]; // kernel <= 8 in every published config
#pragma unroll
    for (int i = 0; i < 8; ++i) window[i] = (i < hist) ? st[i] : 0.0f;

    for (int t = 0; t < n_tokens; ++t) {
        const float cur = x[(size_t)t * channels + c];

        float acc = 0.0f;
        for (int i = 0; i < hist; ++i) {
            acc = fmaf(w[reverse ? (hist - i) : i], window[i], acc);
        }
        acc = fmaf(w[cur_tap], cur, acc);

        if (apply_silu) acc = acc / (1.0f + __expf(-acc));
        x[(size_t)t * channels + c] = acc;

        // Slide the window; the convolution consumes the pre-activation input.
        for (int i = 0; i + 1 < hist; ++i) window[i] = window[i + 1];
        if (hist > 0) window[hist - 1] = cur;
    }

    // A speculative verify reads the history but leaves it; the accepted
    // tokens are pushed afterwards by a call that keeps it.
    if (keep_state) {
        for (int i = 0; i < hist; ++i) st[i] = window[i];
    }
}

// causal_conv1d_kernel's arithmetic, out of place and parallel over tokens as
// well as channels: each thread does CONV_T tokens of one channel, reading the
// inputs (and, before the first token, the history) without writing them. The
// in-place kernel walked all tokens of a channel in one thread -- 40 blocks for
// a 512-token prefill of a 10240-channel layer, 7% of the prefill.
#define CONV_T 8
__global__ void causal_conv1d_oop_kernel(
    const float* __restrict__ x, float* __restrict__ out, const float* __restrict__ weight,
    const float* __restrict__ state, const int n_tokens, const int channels, const int kernel,
    const int apply_silu, const int reverse) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int hist = kernel - 1;
    const float* w = weight + (size_t)c * kernel;
    const float* st = state + (size_t)c * hist;
    const int cur_tap = reverse ? 0 : hist;
    const int t0 = blockIdx.y * CONV_T;
    for (int t = t0; t < min(t0 + CONV_T, n_tokens); ++t) {
        float acc = 0.0f;
        for (int i = 0; i < hist; ++i) {
            const int src = t - hist + i;   // window[i], oldest first
            const float v = src >= 0 ? x[(size_t)src * channels + c] : st[hist + src];
            acc = fmaf(w[reverse ? (hist - i) : i], v, acc);
        }
        acc = fmaf(w[cur_tap], x[(size_t)t * channels + c], acc);
        if (apply_silu) acc = acc / (1.0f + __expf(-acc));
        out[(size_t)t * channels + c] = acc;
    }
}

// The history after the last token: the last `kernel - 1` inputs, reaching
// into the old history when fewer tokens than that arrived.
__global__ void conv_state_push_kernel(
    const float* __restrict__ x, float* __restrict__ state, const int n_tokens, const int channels,
    const int kernel) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int hist = kernel - 1;
    float* st = state + (size_t)c * hist;
    float next[8];
    for (int i = 0; i < hist; ++i) {
        const int src = n_tokens - hist + i;
        next[i] = src >= 0 ? x[(size_t)src * channels + c] : st[hist + src];
    }
    for (int i = 0; i < hist; ++i) st[i] = next[i];
}

// ---------------------------------------------------------------------------
// Per-head L2 normalization
// ---------------------------------------------------------------------------

// Scales each head's vector to unit length. One block per (token, head).
__global__ void l2norm_kernel(
    float* __restrict__ x,
    const int head_dim,
    const float eps) {

    extern __shared__ float shared[];
    float* row = x + (size_t)blockIdx.x * head_dim;

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = row[i];
        sumsq = fmaf(v, v, sumsq);
    }
    sumsq = block_reduce_sum(sumsq, shared);

    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(sumsq + eps);
    __syncthreads();

    const float s = inv;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) row[i] *= s;
}

// ---------------------------------------------------------------------------
// Gate coefficients
// ---------------------------------------------------------------------------

__device__ __forceinline__ float softplus(float x) {
    // log1p(exp(x)) overflows for large x, where it is just x to full precision.
    return (x > 20.0f) ? x : log1pf(__expf(x));
}

// beta = sigmoid(b), and decay = exp(a_neg * softplus(a + dt_bias)).
//
// `a_neg` is the per-head coefficient already in its negative form (`-exp(A_log)`
// in the reference formulation). The loader normalizes it, because converters
// disagree about whether the file stores the log or the exponentiated value --
// applying the exponential twice silently multiplies the decay rate and makes
// the layer forget almost everything within a few tokens.
//
// A negative coefficient times a positive softplus keeps decay in (0, 1): the
// state always shrinks, never grows.
__global__ void delta_gates_kernel(
    const float* __restrict__ a,        // [n_tokens][n_v_heads]
    const float* __restrict__ b,        // [n_tokens][n_v_heads]
    const float* __restrict__ a_neg,    // [n_v_heads], already negative
    const float* __restrict__ dt_bias,  // [n_v_heads]
    float* __restrict__ decay_out,      // [n_tokens][n_v_heads]
    float* __restrict__ beta_out,       // [n_tokens][n_v_heads]
    const int n_tokens,
    const int n_v_heads) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * n_v_heads) return;
    const int h = idx % n_v_heads;

    const float g = a_neg[h] * softplus(a[idx] + dt_bias[h]);
    decay_out[idx] = __expf(g);
    beta_out[idx]  = 1.0f / (1.0f + __expf(-b[idx]));
}

// ---------------------------------------------------------------------------
// The delta rule itself
// ---------------------------------------------------------------------------

// One block per v-head; the token loop is inside because the recurrence is
// strictly sequential.
//
// State is laid out [head_k_dim][head_v_dim] with v fastest. Thread `v` walks
// k, so on every iteration the 128 threads of a head read one contiguous row --
// fully coalesced, which matters because the state is re-read twice per token.
// The delta rule with the key dimension split across threadIdx.y.
//
// The one-warp-column version below leaves the GPU nearly idle at decode: 32
// blocks of 128 threads, each walking the whole 128-wide key dimension twice
// per token, measured 215us per layer for one token -- 23x the memory traffic
// would justify. Splitting dk across DELTA_DK_PAR thread groups multiplies the
// resident parallelism by that factor; the two dot products over dk become
// shared-memory tree reductions. The reduction changes the summation order, so
// this path is validated by the selftest bound rather than bit equality.
#define DELTA_DK_PAR 8
// Value columns per block of the register kernel (see delta_rule_reg_kernel).
#define DELTA_COLS 32

__global__ void delta_rule_par_kernel(
    float* __restrict__ state,          // [n_v_heads][head_k_dim][head_v_dim]
    const float* __restrict__ q,        // [n_tokens][n_k_heads][head_k_dim]
    const float* __restrict__ k,        // [n_tokens][n_k_heads][head_k_dim]
    const float* __restrict__ v,        // [n_tokens][n_v_heads][head_v_dim]
    const float* __restrict__ decay,    // [n_tokens][n_v_heads]
    const float* __restrict__ beta,     // [n_tokens][n_v_heads]
    float* __restrict__ out,            // [n_tokens][n_v_heads][head_v_dim]
    const int n_tokens,
    const int n_k_heads,
    const int n_v_heads,
    const int head_k_dim,
    const int head_v_dim) {

    const int vh = blockIdx.x;
    const int dv = threadIdx.x;          // 0..head_v_dim
    const int j  = threadIdx.y;          // 0..DELTA_DK_PAR
    // GGUF repeats the complete K-head axis to reach the V-head count.
    // Grouping adjacent V heads instead silently selects different Q/K rows.
    const int kh = vh % n_k_heads;

    const int slice = head_k_dim / DELTA_DK_PAR;
    const int dk0 = j * slice;
    const int dk1 = dk0 + slice;

    float* S = state + (size_t)vh * head_k_dim * head_v_dim;

    extern __shared__ float smem[];
    float* k_sh = smem;                              // head_k_dim
    float* q_sh = k_sh + head_k_dim;                 // head_k_dim
    float* red  = q_sh + head_k_dim;                 // DELTA_DK_PAR * head_v_dim
    float* d_sh = red + DELTA_DK_PAR * head_v_dim;   // head_v_dim

    const int tid = j * head_v_dim + dv;
    const int nthreads = head_v_dim * DELTA_DK_PAR;

    for (int t = 0; t < n_tokens; ++t) {
        const float d = decay[(size_t)t * n_v_heads + vh];
        const float b = beta[(size_t)t * n_v_heads + vh];

        for (int i = tid; i < head_k_dim; i += nthreads) {
            k_sh[i] = k[((size_t)t * n_k_heads + kh) * head_k_dim + i];
            q_sh[i] = q[((size_t)t * n_k_heads + kh) * head_k_dim + i];
        }
        __syncthreads();

        // Decay this thread's dk slice and accumulate its share of what the
        // state predicts for k.
        float pm = 0.0f;
        for (int dk = dk0; dk < dk1; ++dk) {
            const float sv = S[(size_t)dk * head_v_dim + dv] * d;
            S[(size_t)dk * head_v_dim + dv] = sv;
            pm = fmaf(sv, k_sh[dk], pm);
        }
        red[j * head_v_dim + dv] = pm;
        __syncthreads();

        // One group folds the partials into the correction for this dv.
        if (j == 0) {
            float mem = 0.0f;
#pragma unroll
            for (int jj = 0; jj < DELTA_DK_PAR; ++jj) {
                mem += red[jj * head_v_dim + dv];
            }
            d_sh[dv] = (v[((size_t)t * n_v_heads + vh) * head_v_dim + dv] - mem) * b;
        }
        __syncthreads();

        const float delta = d_sh[dv];
        float po = 0.0f;
        for (int dk = dk0; dk < dk1; ++dk) {
            const float sv = fmaf(k_sh[dk], delta, S[(size_t)dk * head_v_dim + dv]);
            S[(size_t)dk * head_v_dim + dv] = sv;
            po = fmaf(sv, q_sh[dk], po);
        }
        red[j * head_v_dim + dv] = po;
        __syncthreads();

        if (j == 0) {
            float o = 0.0f;
#pragma unroll
            for (int jj = 0; jj < DELTA_DK_PAR; ++jj) {
                o += red[jj * head_v_dim + dv];
            }
            out[((size_t)t * n_v_heads + vh) * head_v_dim + dv] = o;
        }
        __syncthreads();
    }
}

// delta_rule_par_kernel with each thread's dk slice of the state held in
// registers for the whole call: the state is read once and written once (or,
// with keep_state 0, not at all -- a speculative verify that must leave the
// state as it was). The arithmetic is the in-place kernel's, operation for
// operation, so the results are bit-identical to it.
// Blocks reach 128 x 8 = 1024 threads, so registers must stay at 64 per thread.
// A block-free variant -- a column's eight slices as adjacent lanes reducing
// by shuffle, each thread loading its own k/q slices a token ahead -- measured
// slower on the 27B (prefill 2048 3537 vs 3692 t/s, decode 48.9 vs 49.2): the
// per-thread global loads cost more than the syncs they replaced.
template <int SLICE>
__global__ __launch_bounds__(256) void delta_rule_reg_kernel(
    float* __restrict__ state,          // [n_v_heads][head_k_dim][head_v_dim]
    const float* __restrict__ q,        // [n_tokens][n_k_heads][head_k_dim]
    const float* __restrict__ k,        // [n_tokens][n_k_heads][head_k_dim]
    const float* __restrict__ v,        // [n_tokens][n_v_heads][head_v_dim]
    const float* __restrict__ decay,    // [n_tokens][n_v_heads]
    const float* __restrict__ beta,     // [n_tokens][n_v_heads]
    float* __restrict__ out,            // [n_tokens][n_v_heads][head_v_dim]
    const int n_tokens,
    const int n_k_heads,
    const int n_v_heads,
    const int head_k_dim,
    const int head_v_dim,
    const int keep_state,
    float* const* __restrict__ states = nullptr) {  // rows mode: row blockIdx.z, one token each

    if (states) {
        const size_t r = blockIdx.z;
        state = states[r];
        q += r * n_k_heads * head_k_dim;
        k += r * n_k_heads * head_k_dim;
        v += r * n_v_heads * head_v_dim;
        out += r * n_v_heads * head_v_dim;
        decay += r * n_v_heads;
        beta += r * n_v_heads;
    }
    // Value columns of the state never interact (each has its own memory
    // read, correction and output), so a head's columns split across
    // blockIdx.y: one block per head left 22 of a 70-SM card idle.
    const int vh = blockIdx.x;
    const int cols = blockDim.x;
    const int dv = blockIdx.y * cols + threadIdx.x;
    const int lc = threadIdx.x;
    const int j  = threadIdx.y;
    const int kh = vh % n_k_heads;
    const int dk0 = j * SLICE;

    float* S = state + (size_t)vh * head_k_dim * head_v_dim;

    extern __shared__ float smem[];
    float* k_sh = smem;
    float* q_sh = k_sh + head_k_dim;
    float* red  = q_sh + head_k_dim;
    float* d_sh = red + DELTA_DK_PAR * cols;

    const int tid = j * cols + lc;
    const int nthreads = cols * DELTA_DK_PAR;

    float s[SLICE];
#pragma unroll
    for (int i = 0; i < SLICE; ++i) s[i] = S[(size_t)(dk0 + i) * head_v_dim + dv];

    for (int t = 0; t < n_tokens; ++t) {
        const float d = decay[(size_t)t * n_v_heads + vh];
        const float b = beta[(size_t)t * n_v_heads + vh];

        for (int i = tid; i < head_k_dim; i += nthreads) {
            k_sh[i] = k[((size_t)t * n_k_heads + kh) * head_k_dim + i];
            q_sh[i] = q[((size_t)t * n_k_heads + kh) * head_k_dim + i];
        }
        __syncthreads();

        float pm = 0.0f;
#pragma unroll
        for (int i = 0; i < SLICE; ++i) {
            s[i] = s[i] * d;
            pm = fmaf(s[i], k_sh[dk0 + i], pm);
        }
        red[j * cols + lc] = pm;
        __syncthreads();

        if (j == 0) {
            float mem = 0.0f;
#pragma unroll
            for (int jj = 0; jj < DELTA_DK_PAR; ++jj) mem += red[jj * cols + lc];
            d_sh[lc] = (v[((size_t)t * n_v_heads + vh) * head_v_dim + dv] - mem) * b;
        }
        __syncthreads();

        const float delta = d_sh[lc];
        float po = 0.0f;
#pragma unroll
        for (int i = 0; i < SLICE; ++i) {
            s[i] = fmaf(k_sh[dk0 + i], delta, s[i]);
            po = fmaf(s[i], q_sh[dk0 + i], po);
        }
        red[j * cols + lc] = po;
        __syncthreads();

        if (j == 0) {
            float o = 0.0f;
#pragma unroll
            for (int jj = 0; jj < DELTA_DK_PAR; ++jj) o += red[jj * cols + lc];
            out[((size_t)t * n_v_heads + vh) * head_v_dim + dv] = o;
        }
        __syncthreads();
    }

    if (keep_state) {
#pragma unroll
        for (int i = 0; i < SLICE; ++i) S[(size_t)(dk0 + i) * head_v_dim + dv] = s[i];
    }
}

__global__ void delta_rule_kernel(
    float* __restrict__ state,          // [n_v_heads][head_k_dim][head_v_dim]
    const float* __restrict__ q,        // [n_tokens][n_k_heads][head_k_dim], pre-scaled
    const float* __restrict__ k,        // [n_tokens][n_k_heads][head_k_dim]
    const float* __restrict__ v,        // [n_tokens][n_v_heads][head_v_dim]
    const float* __restrict__ decay,    // [n_tokens][n_v_heads]
    const float* __restrict__ beta,     // [n_tokens][n_v_heads]
    float* __restrict__ out,            // [n_tokens][n_v_heads][head_v_dim]
    const int n_tokens,
    const int n_k_heads,
    const int n_v_heads,
    const int head_k_dim,
    const int head_v_dim) {

    const int vh = blockIdx.x;
    if (vh >= n_v_heads) return;

    // Value heads outnumber key heads; several v-heads share one k/q vector.
    const int kh = vh % n_k_heads;
    const int dv = threadIdx.x;
    if (dv >= head_v_dim) return;

    float* S = state + (size_t)vh * head_k_dim * head_v_dim;

    extern __shared__ float smem[];
    float* k_sh = smem;                 // head_k_dim
    float* q_sh = smem + head_k_dim;    // head_k_dim

    for (int t = 0; t < n_tokens; ++t) {
        const float d = decay[(size_t)t * n_v_heads + vh];
        const float b = beta[(size_t)t * n_v_heads + vh];

        // q and k are read head_k_dim times each below; stage them once.
        for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
            k_sh[i] = k[((size_t)t * n_k_heads + kh) * head_k_dim + i];
            q_sh[i] = q[((size_t)t * n_k_heads + kh) * head_k_dim + i];
        }
        __syncthreads();

        // Decay, and read out what the state currently predicts for k.
        float mem = 0.0f;
        for (int dk = 0; dk < head_k_dim; ++dk) {
            const float s = S[(size_t)dk * head_v_dim + dv] * d;
            S[(size_t)dk * head_v_dim + dv] = s;
            mem = fmaf(s, k_sh[dk], mem);
        }

        // Correct the state toward v, then read out with q.
        const float delta = (v[((size_t)t * n_v_heads + vh) * head_v_dim + dv] - mem) * b;

        float o = 0.0f;
        for (int dk = 0; dk < head_k_dim; ++dk) {
            const float s = fmaf(k_sh[dk], delta, S[(size_t)dk * head_v_dim + dv]);
            S[(size_t)dk * head_v_dim + dv] = s;
            o = fmaf(s, q_sh[dk], o);
        }

        out[((size_t)t * n_v_heads + vh) * head_v_dim + dv] = o;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Gated norms and splits
// ---------------------------------------------------------------------------

// RMSNorm over one head, combined with a SiLU gate.
//
// The gate multiplies the row *before* the row is measured, which is what
// Mamba-2's RMSNormGated does and what Qwen3-Next inherited. That ordering is
// the whole point of the fused op: the gate steers which channels dominate the
// variance, and the normalization then divides it back out, so the output scale
// stays fixed no matter how large the gate grows.
//
// Gating afterwards is a different function -- it leaves the gate's magnitude in
// the output, and 24 linear-attention layers of unnormalized rescaling drift the
// residual stream far enough that the model stops recovering anything from its
// prompt. `gate_first = 0` keeps that older ordering for A/B comparison.
__global__ void rmsnorm_gated_kernel(
    float* __restrict__ x,               // [rows][n]
    const float* __restrict__ weight,    // [n]
    const float* __restrict__ gate,      // [rows][n]
    const int n,
    const float eps,
    const int gate_first,
    const int sigmoid_gate) {

    extern __shared__ float shared[];
    const size_t base = (size_t)blockIdx.x * n;

    // SiLU is what the reference uses; a plain sigmoid is the other activation
    // a gate of this shape could plausibly carry, and only the output tells them
    // apart.
    auto activate = [sigmoid_gate](float g) {
        const float s = 1.0f / (1.0f + __expf(-g));
        return sigmoid_gate ? s : g * s;
    };

    if (gate_first) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            x[base + i] *= activate(gate[base + i]);
        }
        __syncthreads();
    }

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[base + i];
        sumsq = fmaf(v, v, sumsq);
    }
    sumsq = block_reduce_sum(sumsq, shared);

    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sumsq / (float)n + eps);
    __syncthreads();

    const float s = inv_rms;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[base + i] * s * weight[i];
        if (!gate_first) v *= activate(gate[base + i]);
        x[base + i] = v;
    }
}

// Splits a projection that packs a value and a gate into one row.
//
// Two layouts exist in the wild for the same logical tensor. HuggingFace views
// the projection as [head][2 * head_dim] and chunks each head's slice, so value
// and gate interleave per head. A converter that reorders instead writes
// [all values][all gates]. The shapes are identical, so nothing but the output
// distinguishes them -- picking wrong feeds gate values in as queries.
__global__ void split_head_gate_kernel(
    const float* __restrict__ src,   // [n_tokens][2 * n_heads * head_dim]
    float* __restrict__ value,       // [n_tokens][n_heads][head_dim]
    float* __restrict__ gate,        // [n_tokens][n_heads][head_dim]
    const int n_tokens,
    const int n_heads,
    const int head_dim,
    const int interleaved) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_heads * head_dim;
    if (idx >= total) return;

    const int d = idx % head_dim;
    const int rest = idx / head_dim;
    const int h = rest % n_heads;
    const int t = rest / n_heads;

    const int width = n_heads * head_dim;
    if (interleaved) {
        const size_t base = ((size_t)t * n_heads + h) * 2 * head_dim;
        value[idx] = src[base + d];
        gate[idx]  = src[base + head_dim + d];
    } else {
        const size_t base = (size_t)t * 2 * width + (size_t)h * head_dim + d;
        value[idx] = src[base];
        gate[idx]  = src[base + width];
    }
}

// out *= sigmoid(gate)
__global__ void sigmoid_gate_kernel(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const int64_t n) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] *= 1.0f / (1.0f + __expf(-gate[i]));
}

// Extracts q, k and v from a delta-net projection grouped by key head.
//
// HuggingFace views this projection as [k_head][q | k | v...], so one key head's
// queries, keys and its share of the value heads sit together. A converter that
// only strips the gate out leaves that grouping intact, which means q is not a
// contiguous prefix of the row.
//
// The value heads still come out in order: within a key head they are
// consecutive, and key heads are consecutive, so concatenating each group's
// value block reproduces the natural value-head ordering.
__global__ void split_qkv_grouped_kernel(
    const float* __restrict__ src,   // [n_tokens][n_k_heads][2*head_k + v_per_k*head_v]
    float* __restrict__ q,           // [n_tokens][n_k_heads][head_k]
    float* __restrict__ k,           // [n_tokens][n_k_heads][head_k]
    float* __restrict__ v,           // [n_tokens][n_v_heads][head_v]
    const int n_tokens,
    const int n_k_heads,
    const int head_k,
    const int head_v,
    const int v_per_k) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int group = 2 * head_k + v_per_k * head_v;
    const int total = n_tokens * n_k_heads * group;
    if (idx >= total) return;

    const int off = idx % group;
    const int rest = idx / group;
    const int kh = rest % n_k_heads;
    const int t = rest / n_k_heads;

    const float value = src[idx];

    if (off < head_k) {
        q[((size_t)t * n_k_heads + kh) * head_k + off] = value;
    } else if (off < 2 * head_k) {
        k[((size_t)t * n_k_heads + kh) * head_k + (off - head_k)] = value;
    } else {
        const int vo = off - 2 * head_k;
        v[((size_t)t * n_k_heads + kh) * v_per_k * head_v + vo] = value;
    }
}

// Copies a contiguous slice out of a wider row-major buffer.
__global__ void slice_columns_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int n_tokens,
    const int src_width,
    const int offset,
    const int width) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * width) return;
    const int c = idx % width;
    const int t = idx / width;
    dst[idx] = src[(size_t)t * src_width + offset + c];
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

#define AIRUN_ERR_BAD_ARG 1

extern "C" {

airun_status airun_causal_conv1d(
    void* x, const void* weight, void* state,
    int32_t n_tokens, int32_t channels, int32_t kernel, int32_t apply_silu,
    int32_t reverse, int32_t keep_state, cudaStream_t stream) {

    if (n_tokens <= 0 || channels <= 0) return AIRUN_OK;
    if (kernel < 1 || kernel > 8) return AIRUN_ERR_BAD_ARG;

    const int threads = 256;
    const int blocks = (channels + threads - 1) / threads;
    causal_conv1d_kernel<<<blocks, threads, 0, stream>>>(
        (float*)x, (const float*)weight, (float*)state,
        n_tokens, channels, kernel, apply_silu, reverse, keep_state);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_causal_conv1d_oop(
    const void* x, void* out, const void* weight, void* state,
    int32_t n_tokens, int32_t channels, int32_t kernel, int32_t apply_silu,
    int32_t reverse, int32_t keep_state, cudaStream_t stream) {

    if (n_tokens <= 0 || channels <= 0) return AIRUN_OK;
    if (kernel < 1 || kernel > 8) return AIRUN_ERR_BAD_ARG;
    const int threads = 128;
    const dim3 grid((channels + threads - 1) / threads, (n_tokens + CONV_T - 1) / CONV_T);
    causal_conv1d_oop_kernel<<<grid, threads, 0, stream>>>(
        (const float*)x, (float*)out, (const float*)weight, (const float*)state,
        n_tokens, channels, kernel, apply_silu, reverse);
    if (keep_state) {
        conv_state_push_kernel<<<(channels + 255) / 256, 256, 0, stream>>>(
            (const float*)x, (float*)state, n_tokens, channels, kernel);
    }
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_l2norm(
    void* x, int32_t rows, int32_t head_dim, float eps, cudaStream_t stream) {

    if (rows <= 0 || head_dim <= 0) return AIRUN_OK;
    const int threads = (head_dim < 256) ? ((head_dim + 31) / 32) * 32 : 256;
    const size_t shared = ((threads + 31) / 32) * sizeof(float);
    l2norm_kernel<<<rows, threads, shared, stream>>>((float*)x, head_dim, eps);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_delta_gates(
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    void* decay_out, void* beta_out,
    int32_t n_tokens, int32_t n_v_heads, cudaStream_t stream) {

    if (n_tokens <= 0 || n_v_heads <= 0) return AIRUN_OK;
    const int total = n_tokens * n_v_heads;
    const int threads = 256;
    delta_gates_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        (const float*)a, (const float*)b, (const float*)a_log, (const float*)dt_bias,
        (float*)decay_out, (float*)beta_out, n_tokens, n_v_heads);
    AIRUN_CHECK_LAUNCH();
}

int32_t airun_delta_chunk_ok(int32_t n_tokens, int32_t head_k_dim, int32_t head_v_dim, int32_t keep_state);
airun_status airun_delta_chunk(void* state, const void* q, const void* k, const void* v, const void* decay,
                               const void* beta, void* out, int32_t n_tokens, int32_t n_k_heads,
                               int32_t n_v_heads, cudaStream_t stream);

airun_status airun_delta_rule(
    void* state, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* out,
    int32_t n_tokens, int32_t n_k_heads, int32_t n_v_heads,
    int32_t head_k_dim, int32_t head_v_dim, int32_t keep_state, cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (n_k_heads <= 0 || n_v_heads <= 0 || n_v_heads % n_k_heads != 0) return AIRUN_ERR_BAD_ARG;
    // A prompt goes through the chunked form on tensor cores (delta_chunk.cu).
    if (airun_delta_chunk_ok(n_tokens, head_k_dim, head_v_dim, keep_state))
        return airun_delta_chunk(state, q, k, v, decay, beta, out, n_tokens, n_k_heads, n_v_heads, stream);
    if (head_v_dim > 1024 || head_k_dim > 1024) return AIRUN_ERR_BAD_ARG;

    // The dk-parallel form needs the key dim to split evenly and the thread
    // block to fit; anything else takes the one-warp-column original.
    if (head_k_dim == 16 * DELTA_DK_PAR && head_v_dim % DELTA_COLS == 0) {
        const dim3 threads(DELTA_COLS, DELTA_DK_PAR);
        const dim3 grid(n_v_heads, head_v_dim / DELTA_COLS);
        const size_t shared =
            (2 * (size_t)head_k_dim +
             (size_t)(DELTA_DK_PAR + 1) * DELTA_COLS) * sizeof(float);
        delta_rule_reg_kernel<16><<<grid, threads, shared, stream>>>(
            (float*)state, (const float*)q, (const float*)k, (const float*)v,
            (const float*)decay, (const float*)beta, (float*)out,
            n_tokens, n_k_heads, n_v_heads, head_k_dim, head_v_dim, keep_state);
        AIRUN_CHECK_LAUNCH();
    }
    // The in-place kernels below always write the state back.
    if (!keep_state) return AIRUN_ERR_BAD_ARG;
    if (head_k_dim % DELTA_DK_PAR == 0 &&
        head_v_dim * DELTA_DK_PAR <= 1024) {
        const dim3 threads(head_v_dim, DELTA_DK_PAR);
        const size_t shared =
            (2 * (size_t)head_k_dim +
             (size_t)(DELTA_DK_PAR + 1) * head_v_dim) * sizeof(float);
        delta_rule_par_kernel<<<n_v_heads, threads, shared, stream>>>(
            (float*)state, (const float*)q, (const float*)k, (const float*)v,
            (const float*)decay, (const float*)beta, (float*)out,
            n_tokens, n_k_heads, n_v_heads, head_k_dim, head_v_dim);
    } else {
        const int threads = head_v_dim;
        const size_t shared = 2 * (size_t)head_k_dim * sizeof(float);
        delta_rule_kernel<<<n_v_heads, threads, shared, stream>>>(
            (float*)state, (const float*)q, (const float*)k, (const float*)v,
            (const float*)decay, (const float*)beta, (float*)out,
            n_tokens, n_k_heads, n_v_heads, head_k_dim, head_v_dim);
    }
    AIRUN_CHECK_LAUNCH();
}

// One token for each of `rows` conversations, each with its own state
// (states: device array of `rows` pointers): a --parallel decode step in one
// launch instead of one per row. Only the register kernel's shapes (Qwen3.5 /
// 3.8: 128 x 128); anything else is refused and the caller loops.
airun_status airun_delta_rule_rows(
    void* const* states, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* out,
    int32_t rows, int32_t n_k_heads, int32_t n_v_heads,
    int32_t head_k_dim, int32_t head_v_dim, cudaStream_t stream) {

    if (rows <= 0) return AIRUN_OK;
    if (n_k_heads <= 0 || n_v_heads <= 0 || n_v_heads % n_k_heads != 0) return AIRUN_ERR_BAD_ARG;
    if (!(head_k_dim == 16 * DELTA_DK_PAR && head_v_dim % DELTA_COLS == 0)) return AIRUN_ERR_BAD_ARG;
    const dim3 threads(DELTA_COLS, DELTA_DK_PAR);
    const dim3 grid(n_v_heads, head_v_dim / DELTA_COLS, rows);
    const size_t shared =
        (2 * (size_t)head_k_dim + (size_t)(DELTA_DK_PAR + 1) * DELTA_COLS) * sizeof(float);
    delta_rule_reg_kernel<16><<<grid, threads, shared, stream>>>(
        nullptr, (const float*)q, (const float*)k, (const float*)v,
        (const float*)decay, (const float*)beta, (float*)out,
        1, n_k_heads, n_v_heads, head_k_dim, head_v_dim, 1, (float* const*)states);
    AIRUN_CHECK_LAUNCH();
}

// causal_conv1d for one token of each of `rows` conversations, in place, row
// r's history at states[r].
airun_status airun_causal_conv1d_rows(
    void* x, const void* weight, void* const* states,
    int32_t rows, int32_t channels, int32_t kernel, int32_t apply_silu,
    int32_t reverse, cudaStream_t stream) {

    if (rows <= 0 || channels <= 0) return AIRUN_OK;
    if (kernel < 1 || kernel > 8) return AIRUN_ERR_BAD_ARG;
    const int threads = 256;
    const dim3 grid((channels + threads - 1) / threads, rows);
    causal_conv1d_kernel<<<grid, threads, 0, stream>>>(
        (float*)x, (const float*)weight, nullptr,
        1, channels, kernel, apply_silu, reverse, 1, (float* const*)states);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_rmsnorm_gated(
    void* x, const void* weight, const void* gate,
    int32_t n, int32_t rows, float eps, int32_t gate_first, int32_t sigmoid_gate,
    cudaStream_t stream) {

    if (n <= 0 || rows <= 0) return AIRUN_OK;
    const int threads = (n < 1024) ? ((n + 31) / 32) * 32 : 1024;
    const size_t shared = ((threads + 31) / 32) * sizeof(float);
    rmsnorm_gated_kernel<<<rows, threads, shared, stream>>>(
        (float*)x, (const float*)weight, (const float*)gate, n, eps, gate_first, sigmoid_gate);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_split_head_gate(
    const void* src, void* value, void* gate,
    int32_t n_tokens, int32_t n_heads, int32_t head_dim, int32_t interleaved,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    const int total = n_tokens * n_heads * head_dim;
    const int threads = 256;
    split_head_gate_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        (const float*)src, (float*)value, (float*)gate,
        n_tokens, n_heads, head_dim, interleaved);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_sigmoid_gate(void* out, const void* gate, int64_t n, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    const int threads = 256;
    sigmoid_gate_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(
        (float*)out, (const float*)gate, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_split_qkv_grouped(
    const void* src, void* q, void* k, void* v,
    int32_t n_tokens, int32_t n_k_heads, int32_t head_k, int32_t head_v, int32_t v_per_k,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (n_k_heads <= 0 || v_per_k <= 0) return AIRUN_ERR_BAD_ARG;

    const int group = 2 * head_k + v_per_k * head_v;
    const int total = n_tokens * n_k_heads * group;
    const int threads = 256;
    split_qkv_grouped_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        (const float*)src, (float*)q, (float*)k, (float*)v,
        n_tokens, n_k_heads, head_k, head_v, v_per_k);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_slice_columns(
    const void* src, void* dst,
    int32_t n_tokens, int32_t src_width, int32_t offset, int32_t width,
    cudaStream_t stream) {

    if (n_tokens <= 0 || width <= 0) return AIRUN_OK;
    const int total = n_tokens * width;
    const int threads = 256;
    slice_columns_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        (const float*)src, (float*)dst, n_tokens, src_width, offset, width);
    AIRUN_CHECK_LAUNCH();
}

} // extern "C"
