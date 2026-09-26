// Transformer element-wise ops, RoPE, KV-cache maintenance and attention.

#include "dequant.cuh"

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

// RMSNorm over the last axis: out = x * rsqrt(mean(x^2) + eps) * weight.
// One block per row; `weight` may be null for a plain normalization.
__global__ void rmsnorm_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const int n,
    const float eps) {

    extern __shared__ float shared[];
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * n;
    float* outr = out + (size_t)row * n;

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sumsq = fmaf(v, v, sumsq);
    }
    sumsq = block_reduce_sum(sumsq, shared);

    // Broadcast the block-wide result: only thread 0 holds it after the reduce.
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sumsq / (float)n + eps);
    __syncthreads();

    const float scale = inv_rms;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float w = weight ? weight[i] : 1.0f;
        outr[i] = xr[i] * scale * w;
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

// Rotary embedding. NeoX pairs index i with i + rope_dim/2, which is what
// HuggingFace exports and therefore what GGUF carries; `interleaved` selects the
// older GPT-J pairing of (2i, 2i+1) instead. Dimensions at or beyond `rope_dim`
// are left untouched, covering the partial rotary factor newer Qwen variants use.
__global__ void rope_kernel(
    float* __restrict__ x,               // [n_tokens][n_heads][head_dim]
    const int32_t* __restrict__ positions,
    const int n_heads,
    const int head_dim,
    const int rope_dim,
    const float freq_base,
    const float freq_scale,
    const int interleaved) {

    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int half  = rope_dim / 2;

    const int i = blockIdx.z * blockDim.x + threadIdx.x;
    if (i >= half) return;

    const float pos = (float)positions[token] * freq_scale;
    const float inv_freq = __powf(freq_base, -2.0f * (float)i / (float)rope_dim);
    const float theta = pos * inv_freq;

    float sin_t, cos_t;
    __sincosf(theta, &sin_t, &cos_t);

    float* base = x + ((size_t)token * n_heads + head) * head_dim;
    const int lo = interleaved ? (2 * i) : i;
    const int hi = interleaved ? (2 * i + 1) : (i + half);
    const float x0 = base[lo];
    const float x1 = base[hi];
    base[lo] = x0 * cos_t - x1 * sin_t;
    base[hi] = x0 * sin_t + x1 * cos_t;
}

// ---------------------------------------------------------------------------
// Element-wise
// ---------------------------------------------------------------------------

// SwiGLU: out = silu(gate) * up, written in place over `gate`.
__global__ void swiglu_kernel(
    float* __restrict__ gate,
    const float* __restrict__ up,
    const int64_t n) {

    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + __expf(-g))) * up[i];
}

__global__ void add_kernel(float* __restrict__ a, const float* __restrict__ b, const int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void scale_kernel(float* __restrict__ a, const float s, const int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= s;
}

// GELU with the tanh approximation, which is what Gemma-family configs specify.
__global__ void gelu_kernel(float* __restrict__ a, const int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i];
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    a[i] = 0.5f * x * (1.0f + tanhf(inner));
}

// ---------------------------------------------------------------------------
// Embedding lookup
// ---------------------------------------------------------------------------

// Gathers token rows straight out of the packed embedding table, so the table
// never has to be materialised in f32.
template <int TYPE>
__global__ void embed_kernel(
    float* __restrict__ out,             // [n_tokens][n_embd]
    const uint8_t* __restrict__ table,
    const int32_t* __restrict__ tokens,
    const int n_embd,
    const size_t row_bytes,
    const int32_t vocab_size) {

    const int token = blockIdx.y;
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = n_embd / 32;
    if (group >= groups) return;

    int id = tokens[token];
    // A token id past the table would read out of bounds; clamping keeps a bad
    // prompt from taking the process down.
    if (id < 0 || id >= vocab_size) id = 0;

    float vals[32];
    decode_group<TYPE>(table + (size_t)id * row_bytes, group, vals);

    float* dst = out + (size_t)token * n_embd + group * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = vals[i];
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

// Cache layout is [kv_head][max_seq][head_dim] so the keys for one head are
// contiguous along the sequence, which is the access order attention wants.
__global__ void kv_store_kernel(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,          // [n_tokens][n_kv_heads][head_dim]
    const float* __restrict__ v,
    const int32_t* __restrict__ positions,
    const int n_kv_heads,
    const int head_dim,
    const int max_seq) {

    const int token = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int d     = blockIdx.z * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    const int pos = positions[token];
    if (pos < 0 || pos >= max_seq) return;

    const size_t src = ((size_t)token * n_kv_heads + kvh) * head_dim + d;
    const size_t dst = ((size_t)kvh * max_seq + pos) * head_dim + d;
    k_cache[dst] = __float2half(k[src]);
    v_cache[dst] = __float2half(v[src]);
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#define MAX_DIM_PER_LANE 8   // supports head_dim up to 256

// How many warps cooperate on one (token, head) during prefill and during
// decode. The grid is `n_tokens * n_heads` blocks, so prefill saturates the GPU
// on its own and wide blocks only cost it occupancy -- while decode has one
// token and therefore a handful of blocks, and its only source of parallelism
// is inside the block. Measured on a 1.7B at 3072 tokens of context: 8 warps
// gives 61 tok/s and 32 gives 123, while at 512 prefill tokens 32 warps costs
// 8% against 8. Hence one kernel, two instantiations.
#define ATTN_WARPS_PREFILL 8
#define ATTN_WARPS_DECODE 32

// Streaming (online-softmax) attention.
//
// One block per (query token, head). Each warp walks a strided subset of the
// key positions keeping its own running max/sum/accumulator, and the partials
// are merged once at the end. Nothing proportional to sequence length is ever
// stored, so context length costs no scratch memory.
template <int ATTN_WARPS>
__global__ __launch_bounds__(32 * ATTN_WARPS) void attention_kernel(
    float* __restrict__ out,              // [n_tokens][n_heads][head_dim]
    const float* __restrict__ q,          // [n_tokens][n_heads][head_dim]
    const half* __restrict__ k_cache,     // [n_kv_heads][max_seq][head_dim]
    const half* __restrict__ v_cache,
    const int32_t* __restrict__ positions,
    const int n_heads,
    const int n_kv_heads,
    const int head_dim,
    const int max_seq,
    const float scale,
    const int sliding_window) {           // 0 disables the window

    extern __shared__ float smem[];
    // Layout: [ATTN_WARPS] maxima, [ATTN_WARPS] sums, then the accumulators.
    float* s_max = smem;
    float* s_sum = smem + ATTN_WARPS;
    float* s_acc = smem + 2 * ATTN_WARPS;

    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int warp  = threadIdx.y;
    const int lane  = threadIdx.x;
    const int dpl   = head_dim / 32;

    const int kv_head = head / (n_heads / n_kv_heads);
    const int pos = positions[token];

    const float* qv = q + ((size_t)token * n_heads + head) * head_dim;
    float qreg[MAX_DIM_PER_LANE];
#pragma unroll
    for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
        qreg[i] = (i < dpl) ? qv[lane + i * 32] : 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[MAX_DIM_PER_LANE];
#pragma unroll
    for (int i = 0; i < MAX_DIM_PER_LANE; ++i) acc[i] = 0.0f;

    const half* kbase = k_cache + (size_t)kv_head * max_seq * head_dim;
    const half* vbase = v_cache + (size_t)kv_head * max_seq * head_dim;

    // Causal: a query at `pos` sees keys 0..pos inclusive.
    for (int j = warp; j <= pos; j += ATTN_WARPS) {
        if (sliding_window > 0 && (pos - j) >= sliding_window) continue;

        const half* kj = kbase + (size_t)j * head_dim;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
            if (i < dpl) dot = fmaf(qreg[i], __half2float(kj[lane + i * 32]), dot);
        }
        // xor-reduce leaves every lane with the same score.
        dot = warp_reduce_sum(dot) * scale;

        const float new_max = fmaxf(running_max, dot);
        // On the first key running_max is -inf, so corr is exactly 0.
        const float corr = __expf(running_max - new_max);
        const float p    = __expf(dot - new_max);

        running_sum = running_sum * corr + p;
        const half* vj = vbase + (size_t)j * head_dim;
#pragma unroll
        for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
            if (i < dpl) acc[i] = fmaf(p, __half2float(vj[lane + i * 32]), acc[i] * corr);
        }
        running_max = new_max;
    }

    if (lane == 0) {
        s_max[warp] = running_max;
        s_sum[warp] = running_sum;
    }
#pragma unroll
    for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
        if (i < dpl) s_acc[(size_t)warp * head_dim + lane + i * 32] = acc[i];
    }
    __syncthreads();

    if (warp != 0) return;

    float global_max = -INFINITY;
    for (int w = 0; w < ATTN_WARPS; ++w) global_max = fmaxf(global_max, s_max[w]);

    float global_sum = 0.0f;
    float weight[ATTN_WARPS];
    for (int w = 0; w < ATTN_WARPS; ++w) {
        weight[w] = __expf(s_max[w] - global_max);
        global_sum += s_sum[w] * weight[w];
    }
    // Every warp came up empty (possible only with a degenerate window).
    const float inv_sum = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;

    float* dst = out + ((size_t)token * n_heads + head) * head_dim;
#pragma unroll
    for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
        if (i >= dpl) continue;
        float v = 0.0f;
        for (int w = 0; w < ATTN_WARPS; ++w) {
            v = fmaf(s_acc[(size_t)w * head_dim + lane + i * 32], weight[w], v);
        }
        dst[lane + i * 32] = v * inv_sum;
    }
}

// ---------------------------------------------------------------------------
// C entry points
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
    X(AIRUN_TYPE_MXFP4)         \
    X(AIRUN_TYPE_NVFP4)

#define AIRUN_CASE(TYPE) case TYPE: LAUNCH(TYPE); break;

extern "C" {

airun_status airun_rmsnorm(
    void* out, const void* x, const void* weight,
    int32_t n, int32_t rows, float eps, cudaStream_t stream) {

    if (n <= 0 || rows <= 0) return AIRUN_OK;
    const int threads = (n < 1024) ? ((n + 31) / 32) * 32 : 1024;
    const size_t shared = ((threads + 31) / 32) * sizeof(float);
    rmsnorm_kernel<<<rows, threads, shared, stream>>>(
        (float*)out, (const float*)x, (const float*)weight, n, eps);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_rope(
    void* x, const void* positions,
    int32_t n_tokens, int32_t n_heads, int32_t head_dim, int32_t rope_dim,
    float freq_base, float freq_scale, int32_t interleaved, cudaStream_t stream) {

    if (n_tokens <= 0 || n_heads <= 0) return AIRUN_OK;
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) return AIRUN_ERR_BAD_ARG;

    const int half = rope_dim / 2;
    const int threads = (half < 256) ? half : 256;
    const dim3 grid(n_tokens, n_heads, (half + threads - 1) / threads);
    rope_kernel<<<grid, threads, 0, stream>>>(
        (float*)x, (const int32_t*)positions,
        n_heads, head_dim, rope_dim, freq_base, freq_scale, interleaved);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_swiglu(void* gate, const void* up, int64_t n, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    const int threads = 256;
    swiglu_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(
        (float*)gate, (const float*)up, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_add(void* a, const void* b, int64_t n, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    const int threads = 256;
    add_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(
        (float*)a, (const float*)b, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_scale(void* a, float s, int64_t n, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    const int threads = 256;
    scale_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>((float*)a, s, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_gelu(void* a, int64_t n, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    const int threads = 256;
    gelu_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>((float*)a, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_embed(
    int32_t type_id, void* out, const void* table, const void* tokens,
    int32_t n_tokens, int32_t n_embd, uint64_t row_bytes, int32_t vocab_size,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (n_embd % 32 != 0) return AIRUN_ERR_BAD_ARG;

    const int groups = n_embd / 32;
    const int threads = (groups < 256) ? groups : 256;
    const dim3 grid((groups + threads - 1) / threads, n_tokens);

#define LAUNCH(TYPE)                                                        \
    embed_kernel<TYPE><<<grid, threads, 0, stream>>>(                       \
        (float*)out, (const uint8_t*)table, (const int32_t*)tokens,         \
        n_embd, (size_t)row_bytes, vocab_size);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_ARG;
    }
#undef LAUNCH
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_kv_store(
    void* k_cache, void* v_cache, const void* k, const void* v, const void* positions,
    int32_t n_tokens, int32_t n_kv_heads, int32_t head_dim, int32_t max_seq,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    const int threads = (head_dim < 256) ? head_dim : 256;
    const dim3 grid(n_tokens, n_kv_heads, (head_dim + threads - 1) / threads);
    kv_store_kernel<<<grid, threads, 0, stream>>>(
        (half*)k_cache, (half*)v_cache, (const float*)k, (const float*)v,
        (const int32_t*)positions, n_kv_heads, head_dim, max_seq);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_attention(
    void* out, const void* q, const void* k_cache, const void* v_cache,
    const void* positions,
    int32_t n_tokens, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t max_seq, float scale, int32_t sliding_window,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (head_dim % 32 != 0 || head_dim > 32 * MAX_DIM_PER_LANE) return AIRUN_ERR_BAD_ARG;
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) return AIRUN_ERR_BAD_ARG;

    const dim3 grid(n_tokens, n_heads);

#define LAUNCH_ATTENTION(WARPS)                                                   do {                                                                              const dim3 block(32, WARPS);                                                  const size_t shared =                                                             (2 * (WARPS) + (size_t)(WARPS) * head_dim) * sizeof(float);               attention_kernel<WARPS><<<grid, block, shared, stream>>>(                         (float*)out, (const float*)q, (const half*)k_cache,                           (const half*)v_cache, (const int32_t*)positions, n_heads,                     n_kv_heads, head_dim, max_seq, scale, sliding_window);                } while (0)

    // Below roughly a couple of blocks per SM the grid cannot fill the card, so
    // the parallelism has to come from inside the block instead.
    if (n_tokens * n_heads <= 128) {
        LAUNCH_ATTENTION(ATTN_WARPS_DECODE);
    } else {
        LAUNCH_ATTENTION(ATTN_WARPS_PREFILL);
    }
#undef LAUNCH_ATTENTION

    AIRUN_CHECK_LAUNCH();
}

} // extern "C"
