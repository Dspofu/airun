// Mixture-of-experts routing and expert feed-forward.
//
// Only `top_k` of the experts run per token, so the cost is set by top_k and
// not by the expert count: a 256-expert layer with 8 active experts does the
// work of 8 small FFNs. The expert weights are one 3-D tensor with a fixed
// per-expert stride, so an expert is selected by pointer arithmetic rather than
// by gathering rows -- or, when the experts do not all fit in VRAM, through a
// table of per-expert addresses, some in VRAM and some in mapped host memory
// that the kernel reads across PCIe (see airun-core's MappedExperts).

#include "dequant.cuh"

#define MOE_ROWS_PER_BLOCK 4

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

// Softmax over expert logits, then the top-k experts and their weights.
//
// One block per token. Selection is k passes of a masked argmax rather than a
// sort: with k around 8 and a few hundred experts that is far cheaper, and it
// avoids needing scratch proportional to the expert count.
__global__ void moe_router_kernel(
    const float* __restrict__ logits,   // [n_tokens][n_experts]
    int32_t* __restrict__ out_ids,      // [n_tokens][top_k]
    float* __restrict__ out_weights,    // [n_tokens][top_k]
    const int n_experts,
    const int top_k,
    const int normalize) {

    extern __shared__ float smem[];
    float* probs = smem;                       // n_experts
    float* reduce = smem + n_experts;          // blockDim.x / 32

    const int token = blockIdx.x;
    const float* row = logits + (size_t)token * n_experts;

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    local_max = block_reduce_max(local_max, reduce);
    __shared__ float max_logit;
    if (threadIdx.x == 0) max_logit = local_max;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
        const float p = __expf(row[i] - max_logit);
        probs[i] = p;
        local_sum += p;
    }
    local_sum = block_reduce_sum(local_sum, reduce);
    __shared__ float total;
    if (threadIdx.x == 0) total = local_sum;
    __syncthreads();

    const float inv_total = 1.0f / total;
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) probs[i] *= inv_total;
    __syncthreads();

    // Selection is serial in one thread: k is small, and a parallel argmax plus
    // the masking barrier per pass would cost more than it saves.
    if (threadIdx.x == 0) {
        float weight_sum = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            int best = 0;
            float best_p = -1.0f;
            for (int i = 0; i < n_experts; ++i) {
                if (probs[i] > best_p) {
                    best_p = probs[i];
                    best = i;
                }
            }
            out_ids[(size_t)token * top_k + j] = best;
            out_weights[(size_t)token * top_k + j] = best_p;
            weight_sum += best_p;
            probs[best] = -1.0f; // mask so the next pass skips it
        }
        if (normalize && weight_sum > 0.0f) {
            const float inv = 1.0f / weight_sum;
            for (int j = 0; j < top_k; ++j) out_weights[(size_t)token * top_k + j] *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Expert feed-forward
// ---------------------------------------------------------------------------

// h[token][k][row] = silu(gate_e[row] . x[token]) * (up_e[row] . x[token])
//
// One warp per output row, matching the dense matvec; the only difference is
// that the weight base is offset by the selected expert.
template <int TYPE>
__global__ __launch_bounds__(32 * MOE_ROWS_PER_BLOCK) void moe_gate_up_kernel(
    const uint8_t* __restrict__ gate_w,
    const uint8_t* __restrict__ up_w,
    const float* __restrict__ x,          // [n_tokens][K]
    const int32_t* __restrict__ ids,      // [n_tokens][top_k]
    float* __restrict__ h,                // [n_tokens][top_k][ff]
    const int K,
    const int ff,
    const int top_k,
    const size_t row_bytes,
    const size_t expert_stride,
    const unsigned long long* __restrict__ gate_tab,
    const unsigned long long* __restrict__ up_tab,
    const int32_t* __restrict__ sidx,
    const uint8_t* __restrict__ gate_stage,
    const uint8_t* __restrict__ up_stage) {

    const int row = blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= ff) return;

    const int slot  = blockIdx.y;   // which of the token's top_k experts
    const int token = blockIdx.z;
    const int expert = ids[(size_t)token * top_k + slot];

    const uint8_t* g_base = gate_tab ? (const uint8_t*)gate_tab[expert] : gate_w + (size_t)expert * expert_stride;
    const uint8_t* u_base = up_tab ? (const uint8_t*)up_tab[expert] : up_w + (size_t)expert * expert_stride;
    // An expert outside VRAM was copied into the staging area beforehand.
    const int st = sidx ? sidx[(size_t)token * top_k + slot] : -1;
    if (st >= 0) {
        g_base = gate_stage + (size_t)st * expert_stride;
        u_base = up_stage + (size_t)st * expert_stride;
    }
    const uint8_t* g_row = g_base + (size_t)row * row_bytes;
    const uint8_t* u_row = u_base + (size_t)row * row_bytes;
    const float4* xv = (const float4*)(x + (size_t)token * K);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int grp = threadIdx.x; grp < K / 32; grp += 32) {
        float gv[32], uv[32];
        decode_group<TYPE>(g_row, grp, gv);
        decode_group<TYPE>(u_row, grp, uv);

        const float4* xg = xv + grp * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 xx = xg[i];
            acc_g = fmaf(gv[i * 4 + 0], xx.x, acc_g);
            acc_g = fmaf(gv[i * 4 + 1], xx.y, acc_g);
            acc_g = fmaf(gv[i * 4 + 2], xx.z, acc_g);
            acc_g = fmaf(gv[i * 4 + 3], xx.w, acc_g);
            acc_u = fmaf(uv[i * 4 + 0], xx.x, acc_u);
            acc_u = fmaf(uv[i * 4 + 1], xx.y, acc_u);
            acc_u = fmaf(uv[i * 4 + 2], xx.z, acc_u);
            acc_u = fmaf(uv[i * 4 + 3], xx.w, acc_u);
        }
    }

    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);

    if (threadIdx.x == 0) {
        const float silu = acc_g / (1.0f + __expf(-acc_g));
        h[((size_t)token * top_k + slot) * ff + row] = silu * acc_u;
    }
}

// y[token][row] += sum_k weight[token][k] * (down_e[row] . h[token][k])
//
// Summing over k inside the kernel keeps every write exclusive, so no atomics
// are needed even though all k contribute to the same output row.
template <int TYPE>
__global__ __launch_bounds__(32 * MOE_ROWS_PER_BLOCK) void moe_down_kernel(
    const uint8_t* __restrict__ down_w,
    const float* __restrict__ h,          // [n_tokens][top_k][ff]
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    float* __restrict__ y,                // [n_tokens][n_embd], accumulated
    const int ff,
    const int n_embd,
    const int top_k,
    const size_t row_bytes,
    const size_t expert_stride,
    const unsigned long long* __restrict__ down_tab,
    const int32_t* __restrict__ sidx,
    const uint8_t* __restrict__ down_stage) {

    const int row = blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= n_embd) return;
    const int token = blockIdx.y;

    float total = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int expert = ids[(size_t)token * top_k + slot];
        const float w = weights[(size_t)token * top_k + slot];

        const uint8_t* d_base = down_tab ? (const uint8_t*)down_tab[expert] : down_w + (size_t)expert * expert_stride;
        const int st = sidx ? sidx[(size_t)token * top_k + slot] : -1;
        if (st >= 0) d_base = down_stage + (size_t)st * expert_stride;
        const uint8_t* d_row = d_base + (size_t)row * row_bytes;
        const float4* hv = (const float4*)(h + ((size_t)token * top_k + slot) * ff);

        float acc = 0.0f;
        for (int grp = threadIdx.x; grp < ff / 32; grp += 32) {
            float dv[32];
            decode_group<TYPE>(d_row, grp, dv);
            const float4* hg = hv + grp * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 hh = hg[i];
                acc = fmaf(dv[i * 4 + 0], hh.x, acc);
                acc = fmaf(dv[i * 4 + 1], hh.y, acc);
                acc = fmaf(dv[i * 4 + 2], hh.z, acc);
                acc = fmaf(dv[i * 4 + 3], hh.w, acc);
            }
        }
        acc = warp_reduce_sum(acc);
        total = fmaf(w, acc, total);
    }

    if (threadIdx.x == 0) {
        y[(size_t)token * n_embd + row] += total;
    }
}

// Which of the batch's routed experts live outside VRAM, each given one
// staging index: sidx[i] for routed id i (-1 when resident), list[j] the
// expert staged at j, n_list[0] how many. One block; the scan over the experts
// is serial in one thread, which at a few hundred experts is nothing.
__global__ void moe_stage_plan_kernel(
    const int32_t* __restrict__ ids, const int n_ids, const uint8_t* __restrict__ resident, const int n_experts,
    int32_t* __restrict__ sidx, int32_t* __restrict__ list, int32_t* __restrict__ n_list) {
    extern __shared__ int32_t at[];   // n_experts
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) at[e] = -1;
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int e = ids[i];
        if (!resident[e]) at[e] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int e = 0; e < n_experts; ++e) {
            if (at[e] == 0) { at[e] = n; list[n] = e; ++n; }
        }
        n_list[0] = n;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) sidx[i] = at[ids[i]];
}

// Copies the staged experts out of host memory: 16-byte loads, many blocks in
// flight, which is what PCIe needs to run at its bandwidth -- the expert
// kernels' own narrow, dependent reads ran five times slower across it.
__global__ void moe_stage_copy_kernel(
    const unsigned long long* __restrict__ tab, const int32_t* __restrict__ list, const int32_t* __restrict__ n_list,
    uint8_t* __restrict__ stage, const size_t stride) {
    const int j = blockIdx.y;
    if (j >= n_list[0]) return;
    const uint4* src = (const uint4*)tab[list[j]];
    uint4* dst = (uint4*)(stage + (size_t)j * stride);
    const size_t n16 = stride / 16;
    // Four independent loads per thread before any store: a link with
    // microseconds of latency needs many requests in flight, and a
    // load-then-store loop keeps only one per thread.
    const size_t step = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * step < n16; i += 4 * step) {
        const uint4 a = src[i], b = src[i + step], c = src[i + 2 * step], d = src[i + 3 * step];
        dst[i] = a; dst[i + step] = b; dst[i + 2 * step] = c; dst[i + 3 * step] = d;
    }
    for (; i < n16; i += step) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Expert-major batches
// ---------------------------------------------------------------------------
//
// A prefill routes hundreds of tokens; the token-major kernels above read
// each expert's weights once per token that picked it, which across PCIe (an
// expert in host memory) is the whole cost. Here the (token, slot) pairs are
// first sorted by expert, and each expert's rows are decoded once per chunk
// of up to MOE_GROUP_T of its tokens. The arithmetic per pair is the token-major
// kernels', in the same order, so the results are bit-identical to them.
//
// A block's rows are contiguous, so it first copies them into shared memory
// with full-width coalesced loads: read in place, the decoders' narrow loads
// crossed PCIe at a fraction of its bandwidth for an expert in host memory.

#define MOE_GROUP_T 16

// `n` bytes from `src` to shared `dst`, 4 at a time when both allow it.
__device__ __forceinline__ void moe_load_rows(uint8_t* dst, const uint8_t* src, int n) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nt = blockDim.x * blockDim.y;
    if (((uintptr_t)src & 3) == 0 && (n & 3) == 0) {
        for (int i = tid; i < n / 4; i += nt) ((uint32_t*)dst)[i] = ((const uint32_t*)src)[i];
    } else {
        for (int i = tid; i < n; i += nt) dst[i] = src[i];
    }
}

// offsets[e]..offsets[e+1] index perm, which lists the pairs routed to e.
// counts must be zero on entry; it is left holding the per-expert totals.
__global__ void moe_sort_count_kernel(const int32_t* __restrict__ ids, int32_t* __restrict__ counts, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + ids[i], 1);
}

__global__ void moe_sort_scan_kernel(int32_t* __restrict__ counts, int32_t* __restrict__ offsets, const int n_experts) {
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < n_experts; ++e) {
            offsets[e] = acc;
            acc += counts[e];
            counts[e] = offsets[e];   // becomes the scatter cursor
        }
        offsets[n_experts] = acc;
    }
}

__global__ void moe_sort_scatter_kernel(const int32_t* __restrict__ ids, int32_t* __restrict__ cursor,
                                        int32_t* __restrict__ perm, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) perm[atomicAdd(cursor + ids[i], 1)] = i;
}

// h[pair][row] for every pair routed to expert blockIdx.y.
template <int TYPE>
__global__ __launch_bounds__(32 * MOE_ROWS_PER_BLOCK) void moe_grouped_gate_up_kernel(
    const unsigned long long* __restrict__ gate_tab, const unsigned long long* __restrict__ up_tab,
    const uint8_t* __restrict__ gate_w, const uint8_t* __restrict__ up_w, const size_t expert_stride,
    const float* __restrict__ x, const int32_t* __restrict__ perm, const int32_t* __restrict__ offsets,
    float* __restrict__ h, const int K, const int ff, const int top_k, const size_t row_bytes) {
    extern __shared__ __align__(16) uint8_t wsmem[];
    const int e = blockIdx.y;
    const int p0 = offsets[e], p1 = offsets[e + 1];
    if (p0 == p1) return;
    const int row0 = blockIdx.x * MOE_ROWS_PER_BLOCK;
    const int rows = min(MOE_ROWS_PER_BLOCK, ff - row0);
    const uint8_t* g_src = (gate_tab ? (const uint8_t*)gate_tab[e] : gate_w + (size_t)e * expert_stride) + (size_t)row0 * row_bytes;
    const uint8_t* u_src = (up_tab ? (const uint8_t*)up_tab[e] : up_w + (size_t)e * expert_stride) + (size_t)row0 * row_bytes;
    moe_load_rows(wsmem, g_src, rows * (int)row_bytes);
    moe_load_rows(wsmem + MOE_ROWS_PER_BLOCK * row_bytes, u_src, rows * (int)row_bytes);
    __syncthreads();
    const int row = row0 + threadIdx.y;
    if (threadIdx.y >= rows) return;
    const uint8_t* g_row = wsmem + threadIdx.y * row_bytes;
    const uint8_t* u_row = wsmem + (MOE_ROWS_PER_BLOCK + threadIdx.y) * row_bytes;

    for (int c0 = p0; c0 < p1; c0 += MOE_GROUP_T) {
        const int nt = min(MOE_GROUP_T, p1 - c0);
        float acc_g[MOE_GROUP_T], acc_u[MOE_GROUP_T];
        const float4* xv[MOE_GROUP_T];
#pragma unroll
        for (int t = 0; t < MOE_GROUP_T; ++t) {
            acc_g[t] = 0.0f;
            acc_u[t] = 0.0f;
            xv[t] = (const float4*)(x + (size_t)(perm[c0 + min(t, nt - 1)] / top_k) * K);
        }
        for (int grp = threadIdx.x; grp < K / 32; grp += 32) {
            float gv[32], uv[32];
            decode_group<TYPE>(g_row, grp, gv);
            decode_group<TYPE>(u_row, grp, uv);
#pragma unroll
            for (int t = 0; t < MOE_GROUP_T; ++t) {
                if (t < nt) {
                    const float4* xg = xv[t] + grp * 8;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float4 xx = xg[i];
                        acc_g[t] = fmaf(gv[i * 4 + 0], xx.x, acc_g[t]);
                        acc_g[t] = fmaf(gv[i * 4 + 1], xx.y, acc_g[t]);
                        acc_g[t] = fmaf(gv[i * 4 + 2], xx.z, acc_g[t]);
                        acc_g[t] = fmaf(gv[i * 4 + 3], xx.w, acc_g[t]);
                        acc_u[t] = fmaf(uv[i * 4 + 0], xx.x, acc_u[t]);
                        acc_u[t] = fmaf(uv[i * 4 + 1], xx.y, acc_u[t]);
                        acc_u[t] = fmaf(uv[i * 4 + 2], xx.z, acc_u[t]);
                        acc_u[t] = fmaf(uv[i * 4 + 3], xx.w, acc_u[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MOE_GROUP_T; ++t) {
            if (t < nt) {
                const float g = warp_reduce_sum(acc_g[t]);
                const float u = warp_reduce_sum(acc_u[t]);
                if (threadIdx.x == 0) h[(size_t)perm[c0 + t] * ff + row] = g / (1.0f + __expf(-g)) * u;
            }
        }
    }
}

// ypair[pair][row] = down_e[row] . h[pair], for the pairs routed to blockIdx.y.
template <int TYPE>
__global__ __launch_bounds__(32 * MOE_ROWS_PER_BLOCK) void moe_grouped_down_kernel(
    const unsigned long long* __restrict__ down_tab, const uint8_t* __restrict__ down_w, const size_t expert_stride,
    const float* __restrict__ h, const int32_t* __restrict__ perm, const int32_t* __restrict__ offsets,
    float* __restrict__ ypair, const int ff, const int n_embd, const size_t row_bytes) {
    extern __shared__ __align__(16) uint8_t wsmem[];
    const int e = blockIdx.y;
    const int p0 = offsets[e], p1 = offsets[e + 1];
    if (p0 == p1) return;
    const int row0 = blockIdx.x * MOE_ROWS_PER_BLOCK;
    const int rows = min(MOE_ROWS_PER_BLOCK, n_embd - row0);
    moe_load_rows(wsmem, (down_tab ? (const uint8_t*)down_tab[e] : down_w + (size_t)e * expert_stride)
        + (size_t)row0 * row_bytes, rows * (int)row_bytes);
    __syncthreads();
    const int row = row0 + threadIdx.y;
    if (threadIdx.y >= rows) return;
    const uint8_t* d_row = wsmem + threadIdx.y * row_bytes;

    for (int c0 = p0; c0 < p1; c0 += MOE_GROUP_T) {
        const int nt = min(MOE_GROUP_T, p1 - c0);
        float acc[MOE_GROUP_T];
        const float4* hv[MOE_GROUP_T];
#pragma unroll
        for (int t = 0; t < MOE_GROUP_T; ++t) {
            acc[t] = 0.0f;
            hv[t] = (const float4*)(h + (size_t)perm[c0 + min(t, nt - 1)] * ff);
        }
        for (int grp = threadIdx.x; grp < ff / 32; grp += 32) {
            float dv[32];
            decode_group<TYPE>(d_row, grp, dv);
#pragma unroll
            for (int t = 0; t < MOE_GROUP_T; ++t) {
                if (t < nt) {
                    const float4* hg = hv[t] + grp * 8;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float4 hh = hg[i];
                        acc[t] = fmaf(dv[i * 4 + 0], hh.x, acc[t]);
                        acc[t] = fmaf(dv[i * 4 + 1], hh.y, acc[t]);
                        acc[t] = fmaf(dv[i * 4 + 2], hh.z, acc[t]);
                        acc[t] = fmaf(dv[i * 4 + 3], hh.w, acc[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MOE_GROUP_T; ++t) {
            if (t < nt) {
                const float a = warp_reduce_sum(acc[t]);
                if (threadIdx.x == 0) ypair[(size_t)perm[c0 + t] * n_embd + row] = a;
            }
        }
    }
}

// y[token][row] += sum over slots, in slot order, of weight * ypair.
__global__ void moe_combine_kernel(const float* __restrict__ ypair, const float* __restrict__ weights,
                                   float* __restrict__ y, const int n_tokens, const int top_k, const int n_embd) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n_tokens * n_embd) return;
    const int t = (int)(idx / n_embd), row = (int)(idx % n_embd);
    float total = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        total = fmaf(weights[(size_t)t * top_k + k], ypair[((size_t)t * top_k + k) * n_embd + row], total);
    }
    y[idx] += total;
}

// counts[e] += how often the batch routed to expert e.
__global__ void moe_count_kernel(const int32_t* __restrict__ ids, unsigned* __restrict__ counts, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + ids[i], 1u);
}

// Scales each token's row by a per-token sigmoid gate. Used by the shared
// expert, which every token passes through but with a learned strength.
__global__ void sigmoid_scale_rows_kernel(
    float* __restrict__ x,
    const float* __restrict__ logit,   // [n_tokens]
    const int n_tokens,
    const int width) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * width) return;
    const int t = idx / width;
    x[idx] *= 1.0f / (1.0f + __expf(-logit[t]));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

#define AIRUN_ERR_BAD_ARG 1

#define AIRUN_FOR_EACH_TYPE(X)   \
    X(AIRUN_TYPE_F32)            \
    X(AIRUN_TYPE_F16)            \
    X(AIRUN_TYPE_BF16)           \
    X(AIRUN_TYPE_Q4_0)           \
    X(AIRUN_TYPE_Q4_1)           \
    X(AIRUN_TYPE_Q5_0)           \
    X(AIRUN_TYPE_Q5_1)           \
    X(AIRUN_TYPE_Q8_0)           \
    X(AIRUN_TYPE_Q2_K)           \
    X(AIRUN_TYPE_Q3_K)           \
    X(AIRUN_TYPE_Q4_K)           \
    X(AIRUN_TYPE_Q5_K)           \
    X(AIRUN_TYPE_Q6_K)           \
    X(AIRUN_TYPE_IQ4_NL)         \
    X(AIRUN_TYPE_IQ4_XS)         \
    X(AIRUN_TYPE_IQ2_S)          \
    X(AIRUN_TYPE_IQ3_XXS)        \
    X(AIRUN_TYPE_IQ3_S)          \
    X(AIRUN_TYPE_Q2_0)           \
    X(AIRUN_TYPE_MXFP4)         \
    X(AIRUN_TYPE_NVFP4)

#define AIRUN_CASE(TYPE) case TYPE: LAUNCH(TYPE); break;

extern "C" {

airun_status airun_moe_router(
    const void* logits, void* out_ids, void* out_weights,
    int32_t n_tokens, int32_t n_experts, int32_t top_k, int32_t normalize,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (top_k <= 0 || top_k > n_experts) return AIRUN_ERR_BAD_ARG;

    const int threads = (n_experts < 256) ? ((n_experts + 31) / 32) * 32 : 256;
    const size_t shared = ((size_t)n_experts + (threads + 31) / 32) * sizeof(float);
    moe_router_kernel<<<n_tokens, threads, shared, stream>>>(
        (const float*)logits, (int32_t*)out_ids, (float*)out_weights,
        n_experts, top_k, normalize);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_moe_gate_up_tab(
    int32_t type_id,
    const void* gate_w, const void* up_w, const void* x, const void* ids, void* h,
    int32_t k, int32_t ff, int32_t n_tokens, int32_t top_k,
    uint64_t row_bytes, uint64_t expert_stride, const void* gate_tab, const void* up_tab,
    const void* sidx, const void* gate_stage, const void* up_stage,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (k % 32 != 0) return AIRUN_ERR_BAD_ARG;

    const dim3 block(32, MOE_ROWS_PER_BLOCK);
    const dim3 grid((ff + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK, top_k, n_tokens);

#define LAUNCH(TYPE)                                                          \
    moe_gate_up_kernel<TYPE><<<grid, block, 0, stream>>>(                     \
        (const uint8_t*)gate_w, (const uint8_t*)up_w, (const float*)x,        \
        (const int32_t*)ids, (float*)h, k, ff, top_k,                         \
        (size_t)row_bytes, (size_t)expert_stride,                             \
        (const unsigned long long*)gate_tab, (const unsigned long long*)up_tab,  \
        (const int32_t*)sidx, (const uint8_t*)gate_stage, (const uint8_t*)up_stage);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_ARG;
    }
#undef LAUNCH
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_moe_down_tab(
    int32_t type_id,
    const void* down_w, const void* h, const void* ids, const void* weights, void* y,
    int32_t ff, int32_t n_embd, int32_t n_tokens, int32_t top_k,
    uint64_t row_bytes, uint64_t expert_stride, const void* down_tab,
    const void* sidx, const void* down_stage,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (ff % 32 != 0) return AIRUN_ERR_BAD_ARG;

    const dim3 block(32, MOE_ROWS_PER_BLOCK);
    const dim3 grid((n_embd + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK, n_tokens);

#define LAUNCH(TYPE)                                                          \
    moe_down_kernel<TYPE><<<grid, block, 0, stream>>>(                        \
        (const uint8_t*)down_w, (const float*)h, (const int32_t*)ids,         \
        (const float*)weights, (float*)y, ff, n_embd, top_k,                  \
        (size_t)row_bytes, (size_t)expert_stride, (const unsigned long long*)down_tab, \
        (const int32_t*)sidx, (const uint8_t*)down_stage);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_ARG;
    }
#undef LAUNCH
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_moe_gate_up(
    int32_t type_id,
    const void* gate_w, const void* up_w, const void* x, const void* ids, void* h,
    int32_t k, int32_t ff, int32_t n_tokens, int32_t top_k,
    uint64_t row_bytes, uint64_t expert_stride,
    cudaStream_t stream) {
    return airun_moe_gate_up_tab(type_id, gate_w, up_w, x, ids, h, k, ff, n_tokens, top_k, row_bytes,
                                 expert_stride, nullptr, nullptr, nullptr, nullptr, nullptr, stream);
}

airun_status airun_moe_down(
    int32_t type_id,
    const void* down_w, const void* h, const void* ids, const void* weights, void* y,
    int32_t ff, int32_t n_embd, int32_t n_tokens, int32_t top_k,
    uint64_t row_bytes, uint64_t expert_stride,
    cudaStream_t stream) {
    return airun_moe_down_tab(type_id, down_w, h, ids, weights, y, ff, n_embd, n_tokens, top_k, row_bytes,
                              expert_stride, nullptr, nullptr, nullptr, stream);
}

airun_status airun_moe_stage_plan(const void* ids, int32_t n_ids, const void* resident, int32_t n_experts,
                                  void* sidx, void* list, void* n_list, cudaStream_t stream) {
    if (n_ids <= 0) return AIRUN_OK;
    moe_stage_plan_kernel<<<1, 256, n_experts * sizeof(int32_t), stream>>>(
        (const int32_t*)ids, n_ids, (const uint8_t*)resident, n_experts, (int32_t*)sidx, (int32_t*)list,
        (int32_t*)n_list);
    AIRUN_CHECK_LAUNCH();
}

// Up to `cap` experts of `stride` bytes (a multiple of 16).
airun_status airun_moe_stage_copy(const void* tab, const void* list, const void* n_list, void* stage,
                                  uint64_t stride, int32_t cap, cudaStream_t stream) {
    if (cap <= 0) return AIRUN_OK;
    if (stride % 16 != 0) return AIRUN_ERR_BAD_ARG;
    const dim3 grid(32, cap);
    moe_stage_copy_kernel<<<grid, 256, 0, stream>>>(
        (const unsigned long long*)tab, (const int32_t*)list, (const int32_t*)n_list, (uint8_t*)stage,
        (size_t)stride);
    AIRUN_CHECK_LAUNCH();
}

// Sorts the n_tokens * top_k routed pairs by expert: scratch holds
// n_experts ints (zeroed here), offsets n_experts + 1, perm n_tokens * top_k.
airun_status airun_moe_sort(const void* ids, int32_t n, int32_t n_experts, void* scratch, void* offsets, void* perm,
                            cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    cudaMemsetAsync(scratch, 0, (size_t)n_experts * sizeof(int32_t), stream);
    moe_sort_count_kernel<<<(n + 255) / 256, 256, 0, stream>>>((const int32_t*)ids, (int32_t*)scratch, n);
    moe_sort_scan_kernel<<<1, 32, 0, stream>>>((int32_t*)scratch, (int32_t*)offsets, n_experts);
    moe_sort_scatter_kernel<<<(n + 255) / 256, 256, 0, stream>>>((const int32_t*)ids, (int32_t*)scratch,
                                                                 (int32_t*)perm, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_moe_grouped(
    int32_t gate_type, int32_t down_type,
    const void* gate_tab, const void* up_tab, const void* down_tab,
    const void* gate_w, const void* up_w, const void* down_w, uint64_t gate_stride, uint64_t down_stride,
    const void* x, const void* perm, const void* offsets, const void* weights, void* h, void* ypair, void* y,
    int32_t k, int32_t ff, int32_t n_tokens, int32_t top_k, int32_t n_experts,
    uint64_t gate_row_bytes, uint64_t down_row_bytes, cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    if (k % 32 != 0 || ff % 32 != 0) return AIRUN_ERR_BAD_ARG;
    // The rows staged in shared memory must fit the default 48 KiB.
    if (2 * MOE_ROWS_PER_BLOCK * gate_row_bytes > 48 * 1024 || MOE_ROWS_PER_BLOCK * down_row_bytes > 48 * 1024)
        return AIRUN_ERR_BAD_ARG;
    const dim3 block(32, MOE_ROWS_PER_BLOCK);
    {
        const dim3 grid((ff + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK, n_experts);
#define LAUNCH(TYPE)                                                                                   \
        moe_grouped_gate_up_kernel<TYPE><<<grid, block, 2 * MOE_ROWS_PER_BLOCK * gate_row_bytes, stream>>>( \
            (const unsigned long long*)gate_tab, (const unsigned long long*)up_tab,                    \
            (const uint8_t*)gate_w, (const uint8_t*)up_w, (size_t)gate_stride, (const float*)x,        \
            (const int32_t*)perm, (const int32_t*)offsets, (float*)h, k, ff, top_k, (size_t)gate_row_bytes);
        switch (gate_type) {
            AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_ARG;
        }
#undef LAUNCH
    }
    {
        const int n_embd = k;
        const dim3 grid((n_embd + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK, n_experts);
#define LAUNCH(TYPE)                                                                                   \
        moe_grouped_down_kernel<TYPE><<<grid, block, MOE_ROWS_PER_BLOCK * down_row_bytes, stream>>>(    \
            (const unsigned long long*)down_tab, (const uint8_t*)down_w, (size_t)down_stride,          \
            (const float*)h, (const int32_t*)perm, (const int32_t*)offsets, (float*)ypair, ff, n_embd, \
            (size_t)down_row_bytes);
        switch (down_type) {
            AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_ARG;
        }
#undef LAUNCH
        const size_t total = (size_t)n_tokens * n_embd;
        moe_combine_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
            (const float*)ypair, (const float*)weights, (float*)y, n_tokens, top_k, n_embd);
    }
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_moe_count(const void* ids, void* counts, int32_t n, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    moe_count_kernel<<<(n + 255) / 256, 256, 0, stream>>>((const int32_t*)ids, (unsigned*)counts, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_sigmoid_scale_rows(
    void* x, const void* logit, int32_t n_tokens, int32_t width, cudaStream_t stream) {

    if (n_tokens <= 0 || width <= 0) return AIRUN_OK;
    const int total = n_tokens * width;
    const int threads = 256;
    sigmoid_scale_rows_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        (float*)x, (const float*)logit, n_tokens, width);
    AIRUN_CHECK_LAUNCH();
}

} // extern "C"
