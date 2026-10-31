// Mixture-of-experts routing and expert feed-forward.
//
// Only `top_k` of the experts run per token, so the cost is set by top_k and
// not by the expert count: a 256-expert layer with 8 active experts does the
// work of 8 small FFNs. The expert weights are one 3-D tensor with a fixed
// per-expert stride, so an expert is selected by pointer arithmetic rather than
// by gathering rows.

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
    const size_t expert_stride) {

    const int row = blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= ff) return;

    const int slot  = blockIdx.y;   // which of the token's top_k experts
    const int token = blockIdx.z;
    const int expert = ids[(size_t)token * top_k + slot];

    const uint8_t* g_row = gate_w + (size_t)expert * expert_stride + (size_t)row * row_bytes;
    const uint8_t* u_row = up_w   + (size_t)expert * expert_stride + (size_t)row * row_bytes;
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
    const size_t expert_stride) {

    const int row = blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= n_embd) return;
    const int token = blockIdx.y;

    float total = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int expert = ids[(size_t)token * top_k + slot];
        const float w = weights[(size_t)token * top_k + slot];

        const uint8_t* d_row = down_w + (size_t)expert * expert_stride + (size_t)row * row_bytes;
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

airun_status airun_moe_gate_up(
    int32_t type_id,
    const void* gate_w, const void* up_w, const void* x, const void* ids, void* h,
    int32_t k, int32_t ff, int32_t n_tokens, int32_t top_k,
    uint64_t row_bytes, uint64_t expert_stride,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (k % 32 != 0) return AIRUN_ERR_BAD_ARG;

    const dim3 block(32, MOE_ROWS_PER_BLOCK);
    const dim3 grid((ff + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK, top_k, n_tokens);

#define LAUNCH(TYPE)                                                          \
    moe_gate_up_kernel<TYPE><<<grid, block, 0, stream>>>(                     \
        (const uint8_t*)gate_w, (const uint8_t*)up_w, (const float*)x,        \
        (const int32_t*)ids, (float*)h, k, ff, top_k,                         \
        (size_t)row_bytes, (size_t)expert_stride);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_ARG;
    }
#undef LAUNCH
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_moe_down(
    int32_t type_id,
    const void* down_w, const void* h, const void* ids, const void* weights, void* y,
    int32_t ff, int32_t n_embd, int32_t n_tokens, int32_t top_k,
    uint64_t row_bytes, uint64_t expert_stride,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (ff % 32 != 0) return AIRUN_ERR_BAD_ARG;

    const dim3 block(32, MOE_ROWS_PER_BLOCK);
    const dim3 grid((n_embd + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK, n_tokens);

#define LAUNCH(TYPE)                                                          \
    moe_down_kernel<TYPE><<<grid, block, 0, stream>>>(                        \
        (const uint8_t*)down_w, (const float*)h, (const int32_t*)ids,         \
        (const float*)weights, (float*)y, ff, n_embd, top_k,                  \
        (size_t)row_bytes, (size_t)expert_stride);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_ARG;
    }
#undef LAUNCH
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
