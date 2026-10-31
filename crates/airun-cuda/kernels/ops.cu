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

// x += b, then out = rmsnorm(x) * weight: the residual add and the next norm of
// a layer in one pass over the row instead of two launches. Same thread layout
// and per-thread order as add_kernel followed by rmsnorm_kernel, so the result
// is identical.
__global__ void add_rmsnorm_kernel(
    float* __restrict__ out,
    float* __restrict__ x,
    const float* __restrict__ b,
    const float* __restrict__ weight,
    const int n,
    const float eps) {

    extern __shared__ float shared[];
    const int row = blockIdx.x;
    float* xr = x + (size_t)row * n;
    const float* br = b + (size_t)row * n;
    float* outr = out + (size_t)row * n;

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + br[i];
        xr[i] = v;
        sumsq = fmaf(v, v, sumsq);
    }
    sumsq = block_reduce_sum(sumsq, shared);

    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sumsq / (float)n + eps);
    __syncthreads();

    const float scale = inv_rms;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float w = weight ? weight[i] : 1.0f;
        outr[i] = xr[i] * scale * w;
    }
}
// Greedy decoding without the logits leaving the GPU: the index of the first
// maximum, as sample.rs's argmax picks it (strictly greater wins, NaN never
// does, index 0 if nothing beats -inf). Copying the 600 KB of a 152k vocabulary
// to the host and scanning it there held the GPU idle ~0.4 ms per token.
//
// Two passes: ARGMAX_BLOCKS blocks each reduce a stride of x to one (value,
// index), and one block reduces those. A single 1024-thread block took 46 us
// per token on a 152k vocabulary, the 600 KB read by one SM.
#define ARGMAX_BLOCKS 128
#define ARGMAX_THREADS 256

// Folds every thread's (best, idx) into thread 0's. Ties go to the smaller
// index, so the first maximum overall survives.
__device__ __forceinline__ void argmax_block(float& best, int& idx, const int n) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
    }
    __shared__ float sb[32];
    __shared__ int si[32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) { sb[warp] = best; si[warp] = idx; }
    __syncthreads();
    if (warp == 0) {
        const int warps = blockDim.x / 32;
        best = lane < warps ? sb[lane] : -INFINITY;
        idx = lane < warps ? si[lane] : n;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
    }
}

// scratch: ARGMAX_BLOCKS floats, then ARGMAX_BLOCKS ints.
__global__ __launch_bounds__(ARGMAX_THREADS) void argmax_partial_kernel(
    const float* __restrict__ x, const int n, float* __restrict__ scratch) {
    float best = -INFINITY;
    int idx = n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = x[i];
        if (v > best) { best = v; idx = i; }   // ascending i: keeps the first
    }
    argmax_block(best, idx, n);
    if (threadIdx.x == 0) {
        scratch[blockIdx.x] = best;
        ((int*)scratch)[ARGMAX_BLOCKS + blockIdx.x] = idx;
    }
}

// `bump`, if set, is incremented by one: the chained decode graph advances its
// position on the device (see Runtime::forward_decode_chained).
__global__ __launch_bounds__(ARGMAX_THREADS) void argmax_final_kernel(
    const float* __restrict__ scratch, const int n, int32_t* __restrict__ out,
    int32_t* __restrict__ bump) {
    float best = -INFINITY;
    int idx = n;
    if (threadIdx.x < ARGMAX_BLOCKS) {
        best = scratch[threadIdx.x];
        idx = ((const int*)scratch)[ARGMAX_BLOCKS + threadIdx.x];
    }
    argmax_block(best, idx, n);
    if (threadIdx.x == 0) {
        *out = idx < n ? idx : 0;
        if (bump) *bump += 1;
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
    const airun_rope_params rp,
    const int interleaved) {

    const int token = blockIdx.x;
    const int head  = blockIdx.y;
    const int half  = rope_dim / 2;

    const int i = blockIdx.z * blockDim.x + threadIdx.x;
    if (i >= half) return;

    float sin_t, cos_t;
    rope_cos_sin(rp, i, rope_dim, positions[token], &cos_t, &sin_t);

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

// Quantized caches store each head row as GGUF Q8_0 or Q4_0 blocks of 32, so the
// values decode exactly as dequant.cuh decodes weights. The encoders mirror
// encode_q8_0 / encode_q4_0 in quantize.rs bit for bit -- the *_rn intrinsics
// stop nvcc from contracting x*id + 8.5 into an FMA that rounds differently.
#define KV_F16  0
#define KV_Q8_0 1
#define KV_Q4_0 2

template <int KV> struct KvBlock;
template <> struct KvBlock<KV_Q8_0> { static constexpr int BYTES = 34; };
template <> struct KvBlock<KV_Q4_0> { static constexpr int BYTES = 18; };

template <int KV>
__device__ __forceinline__ size_t kv_row_bytes(int head_dim) {
    if constexpr (KV == KV_F16) return (size_t)head_dim * 2;
    else return (size_t)(head_dim / 32) * KvBlock<KV>::BYTES;
}

template <int KV>
__device__ __forceinline__ float kv_load(const uint8_t* row, int e) {
    if constexpr (KV == KV_F16) {
        return __half2float(((const half*)row)[e]);
    } else if constexpr (KV == KV_Q8_0) {
        const uint8_t* b = row + (e >> 5) * 34;
        return __half2float(*(const half*)b) * (float)(int8_t)b[2 + (e & 31)];
    } else {
        const uint8_t* b = row + (e >> 5) * 18;
        const int j = e & 31;
        const uint8_t byte = b[2 + (j & 15)];
        const int q = (j < 16 ? (byte & 0xF) : (byte >> 4)) - 8;
        return __half2float(*(const half*)b) * (float)q;
    }
}

// One warp encodes one block; lane i holds element i.
template <int KV>
__device__ __forceinline__ void kv_encode_block(uint8_t* b, float x, int lane) {
    const unsigned full = 0xffffffffu;
    if constexpr (KV == KV_Q8_0) {
        float amax = fabsf(x);
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(full, amax, o));
        const float d = __fdiv_rn(amax, 127.0f);
        const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
        const float q = fminf(fmaxf(roundf(__fmul_rn(x, id)), -127.0f), 127.0f);
        b[2 + lane] = (uint8_t)(int8_t)q;
        if (lane == 0) *(half*)b = __float2half_rn(d);
    } else {
        // The scale comes from the first element of largest magnitude, sign kept.
        float best = fabsf(x);
        int idx = lane;
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(full, best, o);
            const int oi = __shfl_xor_sync(full, idx, o);
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
        float vmax = __shfl_sync(full, x, idx);
        if (best == 0.0f) vmax = 0.0f;   // the CPU scan never picks a -0.0
        const float d = __fdiv_rn(vmax, -8.0f);
        const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
        const int q = min(max(0, (int)__fadd_rn(__fmul_rn(x, id), 8.5f)), 15);
        const int hi = __shfl_down_sync(full, q, 16);
        if (lane < 16) b[2 + lane] = (uint8_t)(q | (hi << 4));
        if (lane == 0) *(half*)b = __float2half_rn(d);
    }
}

template <int KV>
__global__ void kv_store_quant_kernel(
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    const float* __restrict__ k,          // [n_tokens][n_kv_heads][head_dim]
    const float* __restrict__ v,
    const int32_t* __restrict__ positions,
    const int n_kv_heads,
    const int head_dim,
    const int max_seq) {

    const int token = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int blk   = threadIdx.y;
    const int lane  = threadIdx.x;

    const int pos = positions[token];
    if (pos < 0 || pos >= max_seq) return;

    const size_t src = ((size_t)token * n_kv_heads + kvh) * head_dim + blk * 32 + lane;
    const size_t dst = ((size_t)kvh * max_seq + pos) * kv_row_bytes<KV>(head_dim)
                     + (size_t)blk * KvBlock<KV>::BYTES;
    kv_encode_block<KV>(k_cache + dst, k[src], lane);
    kv_encode_block<KV>(v_cache + dst, v[src], lane);
}

// Everything between the q/k/v projections and attention, for one token row
// per block: per-head RMSNorm of q and k (when the model has it), RoPE on both,
// and k and v into the cache in its storage format. Five launches were ~2.5 us
// each per layer on a decode step, a twentieth of it. Blocks are head_dim
// threads -- as rmsnorm_kernel gets for head rows -- and the arithmetic is
// the separate kernels' expression for expression, so the result is identical.
// blockIdx.y: [0, n_heads) q heads, then n_kv_heads k heads, then v heads.
template <int KV>
__global__ void qk_norm_rope_store_kernel(
    float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    uint8_t* __restrict__ k_cache, uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ positions,
    const int n_heads, const int n_kv_heads, const int head_dim, const int max_seq,
    const float eps, const int rope_dim, const airun_rope_params rp,
    const int interleaved, const int rope_on) {

    extern __shared__ float qk_smem[];   // [head_dim] row, then the reduce scratch
    float* row = qk_smem;
    float* red = qk_smem + head_dim;
    const int token = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    const int pos = positions[token];
    const bool is_q = h < n_heads;
    const bool is_k = !is_q && h < n_heads + n_kv_heads;

    if (!is_q && !is_k) {
        // v: straight into the cache.
        const int kvh = h - n_heads - n_kv_heads;
        const float val = v[((size_t)token * n_kv_heads + kvh) * head_dim + i];
        if (pos < 0 || pos >= max_seq) return;
        const size_t rb = kv_row_bytes<KV>(head_dim);
        uint8_t* dst = v_cache + ((size_t)kvh * max_seq + pos) * rb;
        if constexpr (KV == KV_F16) ((half*)dst)[i] = __float2half(val);
        else kv_encode_block<KV>(dst + (i / 32) * KvBlock<KV>::BYTES, val, i % 32);
        return;
    }

    const int kvh = h - n_heads;
    const float* src = is_q ? q + ((size_t)token * n_heads + h) * head_dim
                            : k + ((size_t)token * n_kv_heads + kvh) * head_dim;
    const float* w = is_q ? q_norm : k_norm;
    float x = src[i];
    if (w) {
        float sumsq = fmaf(x, x, 0.0f);
        sumsq = block_reduce_sum(sumsq, red);
        __shared__ float inv_rms;
        if (i == 0) inv_rms = rsqrtf(sumsq / (float)head_dim + eps);
        __syncthreads();
        x = x * inv_rms * w[i];
    }
    row[i] = x;
    __syncthreads();

    if (rope_on) {
        const int half_ = rope_dim / 2;
        // The thread owning each pair's first element rotates it, reading the
        // unrotated pair from shared memory.
        int pair = -1, other = -1;
        bool first = false;
        if (interleaved) {
            if (i < rope_dim) { pair = i / 2; first = (i % 2) == 0; other = first ? i + 1 : i - 1; }
        } else if (i < rope_dim) {
            pair = i < half_ ? i : i - half_;
            first = i < half_;
            other = first ? i + half_ : i - half_;
        }
        if (pair >= 0) {
            float sin_t, cos_t;
            rope_cos_sin(rp, pair, rope_dim, pos, &cos_t, &sin_t);
            const float x0 = first ? row[i] : row[other];
            const float x1 = first ? row[other] : row[i];
            x = first ? x0 * cos_t - x1 * sin_t : x0 * sin_t + x1 * cos_t;
        }
    }

    if (is_q) {
        q[((size_t)token * n_heads + h) * head_dim + i] = x;
    } else {
        if (pos < 0 || pos >= max_seq) return;
        const size_t rb = kv_row_bytes<KV>(head_dim);
        uint8_t* dst = k_cache + ((size_t)kvh * max_seq + pos) * rb;
        if constexpr (KV == KV_F16) ((half*)dst)[i] = __float2half(x);
        else kv_encode_block<KV>(dst + (i / 32) * KvBlock<KV>::BYTES, x, i % 32);
    }
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
template <int ATTN_WARPS, int KV = KV_F16, bool SPLIT = false>
__global__ __launch_bounds__(32 * ATTN_WARPS) void attention_kernel(
    float* __restrict__ out,              // [n_tokens][n_heads][head_dim]
    const float* __restrict__ q,          // [n_tokens][n_heads][head_dim]
    const uint8_t* __restrict__ k_cache,  // [n_kv_heads][max_seq][row of KV]
    const uint8_t* __restrict__ v_cache,
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

    const size_t row = kv_row_bytes<KV>(head_dim);
    const uint8_t* kbase = k_cache + (size_t)kv_head * max_seq * row;
    const uint8_t* vbase = v_cache + (size_t)kv_head * max_seq * row;

    // Causal: a query at `pos` sees keys 0..pos inclusive.
    const int begin = SPLIT ? (int)(((int64_t)pos + 1) * blockIdx.z / gridDim.z) : 0;
    const int end = SPLIT ? (int)(((int64_t)pos + 1) * (blockIdx.z + 1) / gridDim.z) : pos + 1;
    for (int j = begin + warp; j < end; j += ATTN_WARPS) {
        if (sliding_window > 0 && (pos - j) >= sliding_window) continue;

        const uint8_t* kj = kbase + (size_t)j * row;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
            if (i < dpl) dot = fmaf(qreg[i], kv_load<KV>(kj, lane + i * 32), dot);
        }
        // xor-reduce leaves every lane with the same score.
        dot = warp_reduce_sum(dot) * scale;

        const float new_max = fmaxf(running_max, dot);
        // On the first key running_max is -inf, so corr is exactly 0.
        const float corr = __expf(running_max - new_max);
        const float p    = __expf(dot - new_max);

        running_sum = running_sum * corr + p;
        const uint8_t* vj = vbase + (size_t)j * row;
#pragma unroll
        for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
            if (i < dpl) acc[i] = fmaf(p, kv_load<KV>(vj, lane + i * 32), acc[i] * corr);
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
        weight[w] = s_sum[w] > 0.0f ? __expf(s_max[w] - global_max) : 0.0f;
        global_sum += s_sum[w] * weight[w];
    }
    // Every warp came up empty (possible only with a degenerate window).
    const float inv_sum = (global_sum > 0.0f) ? 1.0f / global_sum : 0.0f;

    float* dst;
    if constexpr (SPLIT) {
        dst = out + (((size_t)token * n_heads + head) * gridDim.z + blockIdx.z) * (head_dim + 2);
        if (lane == 0) { dst[0] = global_max; dst[1] = global_sum; }
        dst += 2;
    } else {
        dst = out + ((size_t)token * n_heads + head) * head_dim;
    }
#pragma unroll
    for (int i = 0; i < MAX_DIM_PER_LANE; ++i) {
        if (i >= dpl) continue;
        float v = 0.0f;
        for (int w = 0; w < ATTN_WARPS; ++w) {
            v = fmaf(s_acc[(size_t)w * head_dim + lane + i * 32], weight[w], v);
        }
        dst[lane + i * 32] = SPLIT ? v : v * inv_sum;
    }
}

// ---------------------------------------------------------------------------
// A second reduction lets independent blocks cover a long cache. Empty splits
// have sum=0; never evaluate exp(-inf - -inf) for them.
// Splits that hold keys at this position: one per GQA_KEYS_PER_SPLIT keys, so a
// short context does not launch and merge dozens of empty splits. The grid is
// fixed when the decode step is captured into a CUDA graph, so the count has
// to be worked out on the device from the position.
#define GQA_KEYS_PER_SPLIT 64
__device__ __forceinline__ int attention_active_splits(const int32_t* positions, int window,
                                                       int splits, int keys_per_split) {
    if (keys_per_split <= 0) return splits;
    const int pos = positions[0];
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    return min(splits, max(1, (pos + 1 - lo + keys_per_split - 1) / keys_per_split));
}

// One block per head, head_dim threads. The splits' (max, sum) pairs are
// loaded in parallel, a warp reduces the global max and the weights once, and
// each thread then sums its dimension over the splits with independent loads.
// The first version walked the splits twice with one dependent load at a time:
// 17 us per layer at 3072 tokens, a sixth of the decode step's attention.
#define MERGE_MAX_SPLITS 256
__global__ void attention_merge_kernel(float* out, const float* partial,
    int head_dim, int splits, const int32_t* positions, int window, int keys_per_split) {
    __shared__ float s_m[MERGE_MAX_SPLITS];
    __shared__ float s_w[MERGE_MAX_SPLITS];
    __shared__ float s_total;
    const int head = blockIdx.x;
    const float* base = partial + (size_t)head * splits * (head_dim + 2);
    const int active = min(MERGE_MAX_SPLITS,
                           attention_active_splits(positions, window, splits, keys_per_split));
    for (int s = threadIdx.x; s < active; s += blockDim.x) {
        const float* p = base + (size_t)s * (head_dim + 2);
        s_m[s] = p[1] > 0.0f ? p[0] : -INFINITY;   // empty splits carry no weight
        s_w[s] = p[1];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float m = -INFINITY;
        for (int s = threadIdx.x; s < active; s += 32) m = fmaxf(m, s_m[s]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float total = 0.0f;
        for (int s = threadIdx.x; s < active; s += 32) {
            // exp(-inf - m) is 0 for empty splits; m is -inf only if all are.
            const float w = s_m[s] == -INFINITY ? 0.0f : __expf(s_m[s] - m);
            total += s_w[s] * w;
            s_w[s] = w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
        if (threadIdx.x == 0) s_total = total;
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll 8
    for (int s = 0; s < active; ++s) acc = fmaf(base[(size_t)s * (head_dim + 2) + 2 + threadIdx.x], s_w[s], acc);
    out[(size_t)head * head_dim + threadIdx.x] = s_total > 0.0f ? acc / s_total : 0.0f;
}
// ---------------------------------------------------------------------------
// Decode attention, one warp per (KV head, split of the context).
//
// The split attention_kernel gives each *query* head its blocks, so under GQA
// every K/V row is fetched once per query head sharing it (4x on Qwen3-4B, 8x
// on the 35B), and each key costs a dependent chain: load, dot, five-step warp
// reduction, exp. At 3072 tokens that was 108 us per layer where the bytes
// justify ~35. Here the query heads of one KV head go together, so the cache
// is read once; each tile of 32 keys is staged with all its loads issued at
// once and coalesced; a lane then owns a key, so the dots need no reduction.
//
// Blocks are a single warp with ~12 KB of shared memory, so up to 16 fit an SM
// and the latency hides behind other warps rather than inside one. Measured
// with clock64: a 4-warp version with 41 KB blocks spent ~13k cycles per tile
// waiting on loads at two blocks per SM.
//
// Output is the split kernel's partial (max, sum, unnormalized acc) per query
// head, so attention_merge_kernel finishes the job.
// ---------------------------------------------------------------------------
#define GQA_MAX 8        // query heads per KV head this kernel handles
#define GQA_TILE 32      // keys per staged tile

template <int D>
struct GqaShape {
    static constexpr int LD = D + 8;   // halves per staged row: 16-byte loads stay conflict-free
    static constexpr size_t Q_BYTES = (size_t)GQA_MAX * D * 4;
    static constexpr size_t T_BYTES = (size_t)GQA_TILE * LD * 2;
    static constexpr size_t P_BYTES = (size_t)GQA_MAX * GQA_TILE * 4;
    static constexpr size_t SMEM = Q_BYTES + T_BYTES + P_BYTES;
};

// Stages rows [t0, t0 + n) of a cache head into `tile` as fp16. The warp's
// lanes take consecutive 16-byte pieces, and a batch of them is loaded before
// any is stored, so the requests overlap.
template <int D, int KV>
__device__ __forceinline__ void gqa_stage(half* tile, const uint8_t* base, size_t row_bytes,
                                          int t0, int n, int lane) {
    constexpr int PIECES = D / 8;                // 16-byte pieces per fp16 row
    constexpr int TOTAL = GQA_TILE * PIECES;
    constexpr int BATCH = 8;
#pragma unroll
    for (int b0 = 0; b0 < TOTAL / 32; b0 += BATCH) {
        uint4 v[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const int idx = (b0 + u) * 32 + lane;
            const int r = idx / PIECES, c = idx % PIECES;
            if (r < n) {
                const uint8_t* row = base + (size_t)(t0 + r) * row_bytes;
                if constexpr (KV == KV_F16) {
                    v[u] = ((const uint4*)row)[c];
                } else {
                    uint32_t wv[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const unsigned short lo = __half_as_ushort(__float2half(kv_load<KV>(row, 8 * c + 2 * e)));
                        const unsigned short hi = __half_as_ushort(__float2half(kv_load<KV>(row, 8 * c + 2 * e + 1)));
                        wv[e] = (uint32_t)lo | ((uint32_t)hi << 16);
                    }
                    v[u] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
                }
            } else {
                v[u] = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const int idx = (b0 + u) * 32 + lane;
            const int r = idx / PIECES, c = idx % PIECES;
            *(uint4*)&tile[r * GqaShape<D>::LD + 8 * c] = v[u];
        }
    }
}

template <int D, int KV>
// minBlocks = 1 lifts ptxas's register target: left to itself it stopped the
// f16 D=128 build at 128 registers and spilled 64 bytes to local memory.
__global__ __launch_bounds__(32, 1) void attention_decode_gqa_kernel(
    float* __restrict__ partial,          // [n_heads][splits][D + 2]
    const float* __restrict__ q,          // [n_heads][D], one token
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ positions,
    const int n_heads,
    const int n_kv_heads,
    const int max_seq,
    const float scale,
    const int sliding_window) {

    using S = GqaShape<D>;
    constexpr int LD = S::LD;
    constexpr int DP = D / 64;             // half2 pairs per lane across a row
    extern __shared__ __align__(16) unsigned char gqa_smem[];
    // Sized for this model's group, not GQA_MAX: with 4 heads per KV head the
    // block needs 11 KB instead of 14, and more of them fit an SM.
    const int G = n_heads / n_kv_heads;
    float* qs = (float*)gqa_smem;                                   // [G][D], pre-scaled
    // One tile buffer, K then V: a second buffer so both could load at once
    // measured slower (66 us against 52 at 3072 tokens) -- the extra shared
    // memory cost more resident blocks than the saved round trip bought.
    half* tile = (half*)(gqa_smem + (size_t)G * D * 4);             // [TILE][LD], K then V
    float* ps = (float*)(gqa_smem + (size_t)G * D * 4 + S::T_BYTES); // [G][TILE]

    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int lane = threadIdx.x;
    const int pos = positions[0];
    const int lo = sliding_window > 0 ? max(0, pos - sliding_window + 1) : 0;
    const int span = pos + 1 - lo;
    const int splits = attention_active_splits(positions, sliding_window, gridDim.y, GQA_KEYS_PER_SPLIT);
    if (split >= splits) return;   // the merge reads only the active ones
    const int begin = lo + (int)((int64_t)span * split / splits);
    const int end = lo + (int)((int64_t)span * (split + 1) / splits);

    for (int i = lane; i < G * D; i += 32) qs[i] = q[(size_t)(kvh * G) * D + i] * scale;
    __syncwarp();

    const size_t row_bytes = kv_row_bytes<KV>(D);
    const uint8_t* kbase = k_cache + (size_t)kvh * max_seq * row_bytes;
    const uint8_t* vbase = v_cache + (size_t)kvh * max_seq * row_bytes;

    float m[GQA_MAX], l[GQA_MAX], acc[GQA_MAX][DP][2];
#pragma unroll
    for (int h = 0; h < GQA_MAX; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < DP; ++i) acc[h][i][0] = acc[h][i][1] = 0.0f;
    }

    for (int t0 = begin; t0 < end; t0 += GQA_TILE) {
        const int n = min(GQA_TILE, end - t0);
        gqa_stage<D, KV>(tile, kbase, row_bytes, t0, n, lane);
        __syncwarp();

        // Lane `lane` scores key t0 + lane against every query head.
        float s[GQA_MAX];
#pragma unroll
        for (int h = 0; h < GQA_MAX; ++h) s[h] = 0.0f;
        const half* kr = tile + lane * LD;
#pragma unroll 4
        for (int c = 0; c < D / 8; ++c) {
            const uint4 raw = *(const uint4*)&kr[8 * c];
            // Unpacked with bit operations: taking the address of `raw` to view
            // it as half2 put it in local memory.
            const uint32_t w4[4] = {raw.x, raw.y, raw.z, raw.w};
            float kf[8];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                kf[2 * e] = __half2float(__ushort_as_half((unsigned short)(w4[e] & 0xFFFFu)));
                kf[2 * e + 1] = __half2float(__ushort_as_half((unsigned short)(w4[e] >> 16)));
            }
#pragma unroll
            for (int h = 0; h < GQA_MAX; ++h) {
                if (h >= G) continue;
                const float4 q0 = *(const float4*)&qs[h * D + 8 * c];
                const float4 q1 = *(const float4*)&qs[h * D + 8 * c + 4];
                s[h] = fmaf(q0.x, kf[0], fmaf(q0.y, kf[1], fmaf(q0.z, kf[2], fmaf(q0.w, kf[3], s[h]))));
                s[h] = fmaf(q1.x, kf[4], fmaf(q1.y, kf[5], fmaf(q1.z, kf[6], fmaf(q1.w, kf[7], s[h]))));
            }
        }
        const bool valid = lane < n;
#pragma unroll
        for (int h = 0; h < GQA_MAX; ++h) {
            if (h >= G) continue;
            const float sv = valid ? s[h] : -INFINITY;
            float mt = sv;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            const float m_new = fmaxf(m[h], mt);
            const float corr = __expf(m[h] - m_new);   // m[h] = -inf only before any key: corr = 0
            const float p = valid ? __expf(sv - m_new) : 0.0f;
            float sum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            l[h] = l[h] * corr + sum;
            m[h] = m_new;
#pragma unroll
            for (int i = 0; i < DP; ++i) { acc[h][i][0] *= corr; acc[h][i][1] *= corr; }
            ps[h * GQA_TILE + lane] = p;
        }
        __syncwarp();   // every lane is done reading K before V overwrites it

        gqa_stage<D, KV>(tile, vbase, row_bytes, t0, n, lane);
        __syncwarp();
        for (int j = 0; j < n; ++j) {
            const half* vr = tile + j * LD;
#pragma unroll
            for (int i = 0; i < DP; ++i) {
                const float2 v = __half22float2(*(const half2*)&vr[2 * lane + 64 * i]);
#pragma unroll
                for (int h = 0; h < GQA_MAX; ++h) {
                    if (h >= G) continue;
                    const float pj = ps[h * GQA_TILE + j];
                    acc[h][i][0] = fmaf(pj, v.x, acc[h][i][0]);
                    acc[h][i][1] = fmaf(pj, v.y, acc[h][i][1]);
                }
            }
        }
        __syncwarp();   // the next tile's K overwrites these rows
    }

    // A compile-time trip count: indexing m/l/acc with a runtime bound put all
    // three on the stack (224 bytes of local memory) and made every inner loop
    // go through it -- 20k cycles per tile instead of a few thousand.
#pragma unroll
    for (int h = 0; h < GQA_MAX; ++h) {
        if (h >= G) continue;
        float* out = partial + ((size_t)(kvh * G + h) * gridDim.y + split) * (D + 2);
        if (lane == 0) { out[0] = m[h]; out[1] = l[h]; }
#pragma unroll
        for (int i = 0; i < DP; ++i) {
            out[2 + 2 * lane + 64 * i] = acc[h][i][0];
            out[2 + 2 * lane + 64 * i + 1] = acc[h][i][1];
        }
    }
}
template <int D>
static cudaError_t launch_gqa_decode(int kv_type, float* partial, const float* q, const uint8_t* k,
                                     const uint8_t* v, const int32_t* pos, int heads, int kv_heads,
                                     int max_seq, float scale, int window, int splits,
                                     cudaStream_t stream) {
    const dim3 grid(kv_heads, splits);
    const int G = heads / kv_heads;
    const size_t smem = (size_t)G * D * 4 + GqaShape<D>::T_BYTES + (size_t)G * GQA_TILE * 4;
#define GQA_LAUNCH(KVT)                                                                     \
    do {                                                                                   \
        cudaFuncSetAttribute(attention_decode_gqa_kernel<D, KVT>,                          \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
        attention_decode_gqa_kernel<D, KVT><<<grid, 32, smem, stream>>>(                    \
            partial, q, k, v, pos, heads, kv_heads, max_seq, scale, window);               \
    } while (0)
    switch (kv_type) {
        case KV_Q8_0: GQA_LAUNCH(KV_Q8_0); break;
        case KV_Q4_0: GQA_LAUNCH(KV_Q4_0); break;
        default:      GQA_LAUNCH(KV_F16); break;
    }
#undef GQA_LAUNCH
    return cudaGetLastError();
}

// AIRUN_NO_GQA_DECODE=1 keeps decode on the per-query-head split kernel.
static bool gqa_decode_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_GQA_DECODE");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// ---------------------------------------------------------------------------
// Prefill attention on tensor cores (flash attention).
//
// attention_kernel gives every (query, head) its own block and walks the whole
// cache from it, so a 512-token prefill re-reads the cache 512 times per head
// and does every product on the FP32 pipes: 38% of a Qwen3-4B prefill. Here a
// block owns 16 queries per warp for one head, stages a tile of keys and values
// in shared memory once for all of them, and runs Q.K^T and P.V as fp16 WMMA
// with fp32 accumulation, keeping the running softmax per query row.
//
// Q and P are rounded to fp16 (P lies in [0, 1]); K and V already are fp16 in
// the cache, or decode to it from q8_0/q4_0. selftest bounds the difference to
// the fp32 reference.
// ---------------------------------------------------------------------------
#include <mma.h>

#define FLASH_KT 32   // keys per staged tile

template <int D>
struct FlashShape {
    // Warps per block, each owning 16 queries. The accumulator lives in shared
    // memory (16 x D floats per warp), which is what bounds this at D = 256.
    static constexpr int WARPS = D >= 256 ? 2 : 4;
    static constexpr int QT = 16 * WARPS;
    static constexpr int LDH = D + 8;          // halves per staged K/V/Q row
    static constexpr int LDS = FLASH_KT + 4;   // floats per score row
    static constexpr int LDP = FLASH_KT + 8;   // halves per probability row
    static constexpr size_t K_BYTES = (size_t)FLASH_KT * LDH * 2;
    static constexpr size_t Q_BYTES = (size_t)QT * LDH * 2;
    static constexpr size_t KV_BYTES = 2 * K_BYTES > Q_BYTES ? 2 * K_BYTES : Q_BYTES;
    static constexpr size_t S_BYTES = (size_t)WARPS * 16 * LDS * 4;
    static constexpr size_t P_BYTES = (size_t)WARPS * 16 * LDP * 2;
    static constexpr size_t O_BYTES = (size_t)WARPS * 16 * D * 4;
    static constexpr size_t SMEM = KV_BYTES + S_BYTES + P_BYTES + O_BYTES + 3 * QT * 4;
};

template <int D, int KV>
__global__ __launch_bounds__(32 * FlashShape<D>::WARPS) void flash_attention_kernel(
    float* __restrict__ out,              // [n_tokens][n_heads][D]
    const float* __restrict__ q,          // [n_tokens][n_heads][D]
    const uint8_t* __restrict__ k_cache,  // [n_kv_heads][max_seq][row of KV]
    const uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ positions,
    const int n_tokens,
    const int n_heads,
    const int n_kv_heads,
    const int max_seq,
    const float scale,
    const int sliding_window) {

    using namespace nvcuda::wmma;
    using S = FlashShape<D>;
    constexpr int WARPS = S::WARPS, QT = S::QT, LDH = S::LDH, LDS = S::LDS, LDP = S::LDP;

    extern __shared__ __align__(32) unsigned char flash_smem[];
    half* ks = (half*)flash_smem;                          // [KT][LDH]
    half* vs = (half*)(flash_smem + S::K_BYTES);           // [KT][LDH]
    half* qs = (half*)flash_smem;                          // [QT][LDH], only before the loop
    float* ss = (float*)(flash_smem + S::KV_BYTES);        // [WARPS][16][LDS]
    half* ps = (half*)(flash_smem + S::KV_BYTES + S::S_BYTES);            // [WARPS][16][LDP]
    float* os = (float*)(flash_smem + S::KV_BYTES + S::S_BYTES + S::P_BYTES); // [WARPS][16][D]
    float* row_max = os + WARPS * 16 * D;                  // [QT]
    float* row_sum = row_max + QT;                         // [QT]
    int* row_pos = (int*)(row_sum + QT);                   // [QT], -1 for padding

    const int tok0 = blockIdx.x * QT;
    const int head = blockIdx.y;
    const int kvh = head / (n_heads / n_kv_heads);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid = threadIdx.x;
    constexpr int THREADS = 32 * WARPS;
    const size_t row_bytes = kv_row_bytes<KV>(D);
    const uint8_t* kbase = k_cache + (size_t)kvh * max_seq * row_bytes;
    const uint8_t* vbase = v_cache + (size_t)kvh * max_seq * row_bytes;

    // Stage Q (pre-scaled) and the bookkeeping; padding rows are zero queries
    // that attend to nothing.
    for (int i = tid; i < QT * D; i += THREADS) {
        const int r = i / D, d = i % D;
        const int tok = tok0 + r;
        const float v = tok < n_tokens ? q[((size_t)tok * n_heads + head) * D + d] * scale : 0.0f;
        qs[r * LDH + d] = __float2half(v);
    }
    for (int r = tid; r < QT; r += THREADS) {
        const int tok = tok0 + r;
        row_pos[r] = tok < n_tokens ? positions[tok] : -1;
        row_max[r] = -INFINITY;
        row_sum[r] = 0.0f;
    }
    for (int i = tid; i < WARPS * 16 * D; i += THREADS) os[i] = 0.0f;
    __syncthreads();

    fragment<matrix_a, 16, 16, 16, half, row_major> qf[D / 16];
#pragma unroll
    for (int c = 0; c < D / 16; ++c) load_matrix_sync(qf[c], qs + (warp * 16) * LDH + c * 16, LDH);

    // Key range the whole tile can see.
    int hi = -1, lo = 0x7fffffff;
    for (int r = 0; r < QT; ++r) {
        const int p = row_pos[r];
        if (p < 0) continue;
        hi = max(hi, p);
        lo = min(lo, sliding_window > 0 ? max(0, p - sliding_window + 1) : 0);
    }
    __syncthreads();   // everyone has its Q fragments before K overwrites qs
    if (hi < 0) return;
    const int k_begin = (lo / FLASH_KT) * FLASH_KT;

    float* my_s = ss + warp * 16 * LDS;
    half* my_p = ps + warp * 16 * LDP;
    float* my_o = os + warp * 16 * D;
    // Two lanes per query row, each covering half the keys of a tile.
    const int prow = lane / 2;
    const int phalf = lane % 2;
    const int my_pos = row_pos[warp * 16 + prow];

    for (int k0 = k_begin; k0 <= hi; k0 += FLASH_KT) {
        // Stage K and V as fp16; keys past `hi` stage zeros and are masked.
        for (int i = tid; i < FLASH_KT * D; i += THREADS) {
            const int j = i / D, d = i % D;
            const int key = k0 + j;
            half kv = __float2half(0.0f), vv = kv;
            if (key <= hi && key < max_seq) {
                kv = __float2half(kv_load<KV>(kbase + (size_t)key * row_bytes, d));
                vv = __float2half(kv_load<KV>(vbase + (size_t)key * row_bytes, d));
            }
            ks[j * LDH + d] = kv;
            vs[j * LDH + d] = vv;
        }
        __syncthreads();

        // S = Q K^T for this warp's 16 queries.
#pragma unroll
        for (int n = 0; n < FLASH_KT / 16; ++n) {
            fragment<accumulator, 16, 16, 16, float> sf;
            fill_fragment(sf, 0.0f);
#pragma unroll
            for (int c = 0; c < D / 16; ++c) {
                fragment<matrix_b, 16, 16, 16, half, col_major> kf;
                load_matrix_sync(kf, ks + (n * 16) * LDH + c * 16, LDH);
                mma_sync(sf, qf[c], kf, sf);
            }
            store_matrix_sync(my_s + n * 16, sf, LDS, mem_row_major);
        }
        __syncwarp();

        // Online softmax on the row, masked by causality and the window.
        float m_tile = -INFINITY;
        const int j0 = phalf * (FLASH_KT / 2);
        for (int j = j0; j < j0 + FLASH_KT / 2; ++j) {
            const int key = k0 + j;
            const bool ok = my_pos >= 0 && key <= my_pos &&
                            (sliding_window <= 0 || my_pos - key < sliding_window);
            const float s = ok ? my_s[prow * LDS + j] : -INFINITY;
            my_s[prow * LDS + j] = s;
            m_tile = fmaxf(m_tile, s);
        }
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 1));
        const float m_old = row_max[warp * 16 + prow];
        const float m_new = fmaxf(m_old, m_tile);
        // A row with nothing visible yet keeps m = -inf; exp(-inf - -inf) is NaN.
        const float corr = m_new == -INFINITY ? 1.0f : __expf(m_old - m_new);
        float sum = 0.0f;
        for (int j = j0; j < j0 + FLASH_KT / 2; ++j) {
            const float s = my_s[prow * LDS + j];
            const float p = s == -INFINITY ? 0.0f : __expf(s - m_new);
            sum += p;
            my_p[prow * LDP + j] = __float2half(p);
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        for (int d = phalf; d < D; d += 2) my_o[prow * D + d] *= corr;
        __syncwarp();
        if (phalf == 0) {
            row_max[warp * 16 + prow] = m_new;
            row_sum[warp * 16 + prow] = row_sum[warp * 16 + prow] * corr + sum;
        }

        // O += P V
#pragma unroll
        for (int c = 0; c < D / 16; ++c) {
            fragment<accumulator, 16, 16, 16, float> of;
            load_matrix_sync(of, my_o + c * 16, D, mem_row_major);
#pragma unroll
            for (int kk = 0; kk < FLASH_KT / 16; ++kk) {
                fragment<matrix_a, 16, 16, 16, half, row_major> pf;
                fragment<matrix_b, 16, 16, 16, half, row_major> vf;
                load_matrix_sync(pf, my_p + kk * 16, LDP);
                load_matrix_sync(vf, vs + (kk * 16) * LDH + c * 16, LDH);
                mma_sync(of, pf, vf, of);
            }
            store_matrix_sync(my_o + c * 16, of, D, mem_row_major);
        }
        __syncthreads();   // K/V tiles are rewritten next iteration
    }

    for (int i = lane; i < 16 * D; i += 32) {
        const int r = i / D, d = i % D;
        const int tok = tok0 + warp * 16 + r;
        if (tok >= n_tokens) continue;
        const float l = row_sum[warp * 16 + r];
        out[((size_t)tok * n_heads + head) * D + d] = l > 0.0f ? my_o[r * D + d] / l : 0.0f;
    }
}

template <int D>
static cudaError_t launch_flash(int kv_type, float* out, const float* q, const uint8_t* k,
                                const uint8_t* v, const int32_t* pos, int n_tokens, int n_heads,
                                int n_kv_heads, int max_seq, float scale, int window,
                                cudaStream_t stream) {
    using S = FlashShape<D>;
    const dim3 grid((n_tokens + S::QT - 1) / S::QT, n_heads);
#define FLASH_LAUNCH(KVT)                                                                   \
    do {                                                                                   \
        cudaFuncSetAttribute(flash_attention_kernel<D, KVT>,                               \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)S::SMEM);   \
        flash_attention_kernel<D, KVT><<<grid, 32 * S::WARPS, S::SMEM, stream>>>(          \
            out, q, k, v, pos, n_tokens, n_heads, n_kv_heads, max_seq, scale, window);     \
    } while (0)
    switch (kv_type) {
        case KV_Q8_0: FLASH_LAUNCH(KV_Q8_0); break;
        case KV_Q4_0: FLASH_LAUNCH(KV_Q4_0); break;
        default:      FLASH_LAUNCH(KV_F16); break;
    }
#undef FLASH_LAUNCH
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Prefill attention, FlashAttention-2 style, for head_dim 64 and 128.
//
// flash_attention_kernel above uses the WMMA API, whose fragment layout is
// opaque: to rescale the running output it had to store O to shared memory and
// load it back on every key tile. It took 39 ms of a 512-token Qwen3-4B
// prefill where llama.cpp's took 8. Here everything a warp owns stays in
// registers -- Q as A fragments, the scores S, the output O -- because
// mma.sync's m16n8k16 layout is documented: a thread holds rows gid and gid+8,
// so the softmax works on registers, and two adjacent S tiles are exactly the
// A fragment P needs for P.V. K and V tiles arrive by cp.async, double
// buffered, and feed the MMAs through ldmatrix (.trans for V). Key tiles past
// a warp's last query position are skipped: half the work of a causal prefill.
// ---------------------------------------------------------------------------
#define F2_KT 64
#define F2_WARPS 8
#define F2_QT (16 * F2_WARPS)

template <int D>
struct F2Shape {
    // Keys per tile. At head_dim 256 a warp already holds Q (64 registers) and
    // O (128); 64-key tiles left no room for the scores and needed 135 KiB of
    // shared memory, so the widest heads (Qwen3.5/3.8) take 32.
    static constexpr int KT = D >= 256 ? 32 : F2_KT;
    static constexpr int LD = D + 8;                           // halves; 16B-aligned rows, conflict-free ldmatrix
    static constexpr size_t TILE = (size_t)KT * LD * 2;        // one K or V tile
    static constexpr size_t SMEM = 4 * TILE;                   // K and V, two stages
};

__device__ __forceinline__ uint32_t f2_pack(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ void f2_mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void f2_ldsm(uint32_t r[4], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void f2_ldsm_t(uint32_t r[4], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

// cp.async that zero-fills when `valid` is false, so keys past the end stage
// as zeros without a branch around the copy.
__device__ __forceinline__ void f2_cp16(void* smem, const void* gmem, bool valid) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(n));
}

template <int D, int KV>
__global__ __launch_bounds__(32 * F2_WARPS, 1) void flash2_attention_kernel(
    float* __restrict__ out,              // [n_tokens][n_heads][D]
    const float* __restrict__ q,          // [n_tokens][n_heads][D]
    const uint8_t* __restrict__ k_cache,  // [n_kv_heads][max_seq][row of KV]
    const uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ positions,
    const int n_tokens,
    const int n_heads,
    const int n_kv_heads,
    const int max_seq,
    const float scale,
    const int sliding_window) {

    using S = F2Shape<D>;
    constexpr int LD = S::LD;
    constexpr int NJ = S::KT / 8;       // key n8 tiles per KV tile
    constexpr int ND = D / 8;           // head-dim n8 tiles
    extern __shared__ __align__(16) unsigned char f2_smem[];
    half* kbuf = (half*)f2_smem;                              // [2][KT][LD]
    half* vbuf = (half*)(f2_smem + 2 * S::TILE);              // [2][KT][LD]
    __shared__ int warp_hi[F2_WARPS], warp_lo[F2_WARPS];

    const int tok0 = blockIdx.x * F2_QT;
    const int head = blockIdx.y;
    const int kvh = head / (n_heads / n_kv_heads);
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int wq0 = tok0 + warp * 16;
    const int pos0 = wq0 + gid < n_tokens ? positions[wq0 + gid] : -1;
    const int pos1 = wq0 + gid + 8 < n_tokens ? positions[wq0 + gid + 8] : -1;
    const size_t row_bytes = kv_row_bytes<KV>(D);
    const uint8_t* kbase = k_cache + (size_t)kvh * max_seq * row_bytes;
    const uint8_t* vbase = v_cache + (size_t)kvh * max_seq * row_bytes;

    // The keys this warp and this block can see.
    int whi = max(pos0, pos1);
    int wlo = 0x7fffffff;
    if (pos0 >= 0) wlo = sliding_window > 0 ? max(0, pos0 - sliding_window + 1) : 0;
    if (pos1 >= 0) wlo = min(wlo, sliding_window > 0 ? max(0, pos1 - sliding_window + 1) : 0);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        whi = max(whi, __shfl_xor_sync(0xffffffffu, whi, o));
        wlo = min(wlo, __shfl_xor_sync(0xffffffffu, wlo, o));
    }
    if (lane == 0) { warp_hi[warp] = whi; warp_lo[warp] = wlo; }
    __syncthreads();
    int hi = -1, lo = 0x7fffffff;
#pragma unroll
    for (int w = 0; w < F2_WARPS; ++w) { hi = max(hi, warp_hi[w]); lo = min(lo, warp_lo[w]); }
    if (hi < 0) return;
    const int k_begin = (lo / S::KT) * S::KT;

    // Q as A fragments, pre-scaled by scale * log2(e) so the softmax is exp2.
    const float qs = scale * 1.4426950408889634f;
    uint32_t qa[D / 16][4];
    {
        const float* q0 = q + ((size_t)(wq0 + gid) * n_heads + head) * D;
        const float* q1 = q + ((size_t)(wq0 + gid + 8) * n_heads + head) * D;
        const bool v0 = pos0 >= 0, v1 = pos1 >= 0;
#pragma unroll
        for (int c = 0; c < D / 16; ++c) {
            const int d = 16 * c + tig * 2;
            const float2 a = v0 ? *(const float2*)(q0 + d) : make_float2(0.0f, 0.0f);
            const float2 b = v1 ? *(const float2*)(q1 + d) : make_float2(0.0f, 0.0f);
            const float2 e = v0 ? *(const float2*)(q0 + d + 8) : make_float2(0.0f, 0.0f);
            const float2 f = v1 ? *(const float2*)(q1 + d + 8) : make_float2(0.0f, 0.0f);
            qa[c][0] = f2_pack(a.x * qs, a.y * qs);
            qa[c][1] = f2_pack(b.x * qs, b.y * qs);
            qa[c][2] = f2_pack(e.x * qs, e.y * qs);
            qa[c][3] = f2_pack(f.x * qs, f.y * qs);
        }
    }

    float o[ND][4];
#pragma unroll
    for (int j = 0; j < ND; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    // Stage the KV tile starting at key k0 into buffer `st`.
    auto stage = [&](int k0, int st) {
        half* kd = kbuf + st * S::KT * LD;
        half* vd = vbuf + st * S::KT * LD;
        constexpr int PIECES = D / 8;                 // 16-byte pieces per row
        for (int i = tid; i < S::KT * PIECES; i += 32 * F2_WARPS) {
            const int r = i / PIECES, c = i % PIECES;
            const int key = k0 + r;
            const bool ok = key <= hi && key < max_seq;
            if constexpr (KV == KV_F16) {
                const size_t off = (size_t)(ok ? key : 0) * row_bytes + c * 16;
                f2_cp16(kd + r * LD + c * 8, kbase + off, ok);
                f2_cp16(vd + r * LD + c * 8, vbase + off, ok);
            } else {
                uint32_t kw[4], vw[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int d = c * 8 + 2 * e;
                    kw[e] = ok ? f2_pack(kv_load<KV>(kbase + (size_t)key * row_bytes, d),
                                         kv_load<KV>(kbase + (size_t)key * row_bytes, d + 1)) : 0u;
                    vw[e] = ok ? f2_pack(kv_load<KV>(vbase + (size_t)key * row_bytes, d),
                                         kv_load<KV>(vbase + (size_t)key * row_bytes, d + 1)) : 0u;
                }
                *(uint4*)(kd + r * LD + c * 8) = make_uint4(kw[0], kw[1], kw[2], kw[3]);
                *(uint4*)(vd + r * LD + c * 8) = make_uint4(vw[0], vw[1], vw[2], vw[3]);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    stage(k_begin, 0);
    int st = 0;
    for (int k0 = k_begin; k0 <= hi; k0 += S::KT, st ^= 1) {
        if (k0 + S::KT <= hi) {
            stage(k0 + S::KT, st ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        if (k0 <= whi && k0 + S::KT > wlo) {
            const half* kt = kbuf + st * S::KT * LD;
            const half* vt = vbuf + st * S::KT * LD;

            // S = Q K^T for 16 queries x 64 keys.
            float s[NJ][4];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
                for (int c = 0; c < D / 16; c += 2) {
                    uint32_t b[4];
                    f2_ldsm(b, kt + (8 * j + (lane & 7)) * LD + 16 * c + (lane >> 3) * 8);
                    f2_mma(s[j], qa[c], b[0], b[1]);
                    f2_mma(s[j], qa[c + 1], b[2], b[3]);
                }
            }

            // Mask, then the online softmax on rows gid (s[.][0..1]) and gid+8.
            float t0 = -INFINITY, t1 = -INFINITY;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int key = k0 + 8 * j + tig * 2 + (e & 1);
                    const int pos = e < 2 ? pos0 : pos1;
                    const bool ok = key <= pos && (sliding_window <= 0 || pos - key < sliding_window);
                    s[j][e] = ok ? s[j][e] : -INFINITY;
                }
                t0 = fmaxf(t0, fmaxf(s[j][0], s[j][1]));
                t1 = fmaxf(t1, fmaxf(s[j][2], s[j][3]));
            }
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, 1));
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, 2));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, 1));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, 2));
            const float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);
            // A row with nothing visible yet keeps -inf; exp2(-inf - -inf) is NaN.
            const float c0 = n0 == -INFINITY ? 1.0f : exp2f(m0 - n0);
            const float c1 = n1 == -INFINITY ? 1.0f : exp2f(m1 - n1);
            float r0 = 0.0f, r1 = 0.0f;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                s[j][0] = s[j][0] == -INFINITY ? 0.0f : exp2f(s[j][0] - n0);
                s[j][1] = s[j][1] == -INFINITY ? 0.0f : exp2f(s[j][1] - n0);
                s[j][2] = s[j][2] == -INFINITY ? 0.0f : exp2f(s[j][2] - n1);
                s[j][3] = s[j][3] == -INFINITY ? 0.0f : exp2f(s[j][3] - n1);
                r0 += s[j][0] + s[j][1];
                r1 += s[j][2] + s[j][3];
            }
            r0 += __shfl_xor_sync(0xffffffffu, r0, 1);
            r0 += __shfl_xor_sync(0xffffffffu, r0, 2);
            r1 += __shfl_xor_sync(0xffffffffu, r1, 1);
            r1 += __shfl_xor_sync(0xffffffffu, r1, 2);
            l0 = l0 * c0 + r0;
            l1 = l1 * c1 + r1;
            m0 = n0;
            m1 = n1;
#pragma unroll
            for (int j = 0; j < ND; ++j) { o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1; }

            // O += P V. Two adjacent S tiles are one k16 A fragment of P.
#pragma unroll
            for (int kk = 0; kk < S::KT / 16; ++kk) {
                const uint32_t pa[4] = {
                    f2_pack(s[2 * kk][0], s[2 * kk][1]), f2_pack(s[2 * kk][2], s[2 * kk][3]),
                    f2_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), f2_pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
                for (int jd = 0; jd < ND; jd += 2) {
                    uint32_t b[4];
                    f2_ldsm_t(b, vt + (16 * kk + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + 8 * jd + (lane >> 4) * 8);
                    f2_mma(o[jd], pa, b[0], b[1]);
                    f2_mma(o[jd + 1], pa, b[2], b[3]);
                }
            }
        }
        __syncthreads();   // the stage just read is refilled next iteration
    }

    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < ND; ++j) {
        const int d = 8 * j + tig * 2;
        if (pos0 >= 0) *(float2*)(out + ((size_t)(wq0 + gid) * n_heads + head) * D + d) = make_float2(o[j][0] * i0, o[j][1] * i0);
        if (pos1 >= 0) *(float2*)(out + ((size_t)(wq0 + gid + 8) * n_heads + head) * D + d) = make_float2(o[j][2] * i1, o[j][3] * i1);
    }
}

template <int D>
static cudaError_t launch_flash2(int kv_type, float* out, const float* q, const uint8_t* k,
                                 const uint8_t* v, const int32_t* pos, int n_tokens, int n_heads,
                                 int n_kv_heads, int max_seq, float scale, int window,
                                 cudaStream_t stream) {
    const dim3 grid((n_tokens + F2_QT - 1) / F2_QT, n_heads);
    const size_t smem = F2Shape<D>::SMEM;
#define F2_LAUNCH(KVT)                                                                      \
    do {                                                                                   \
        cudaFuncSetAttribute(flash2_attention_kernel<D, KVT>,                              \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);      \
        flash2_attention_kernel<D, KVT><<<grid, 32 * F2_WARPS, smem, stream>>>(            \
            out, q, k, v, pos, n_tokens, n_heads, n_kv_heads, max_seq, scale, window);     \
    } while (0)
    switch (kv_type) {
        case KV_Q8_0: F2_LAUNCH(KV_Q8_0); break;
        case KV_Q4_0: F2_LAUNCH(KV_Q4_0); break;
        default:      F2_LAUNCH(KV_F16); break;
    }
#undef F2_LAUNCH
    return cudaGetLastError();
}

// AIRUN_NO_FLASH2=1 keeps head_dim 64/128 on the WMMA flash kernel.
static bool flash2_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_FLASH2");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// Batches this small keep attention_kernel: a tile of 32-64 queries would be
// mostly padding, and decode has its own split kernel.
#define FLASH_MIN_TOKENS 16

// AIRUN_NO_FLASH=1 keeps prefill on attention_kernel, to measure against it.
static bool flash_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_FLASH");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

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
    X(AIRUN_TYPE_IQ2_S)          \
    X(AIRUN_TYPE_IQ3_XXS)        \
    X(AIRUN_TYPE_IQ3_S)          \
    X(AIRUN_TYPE_Q2_0)           \
    X(AIRUN_TYPE_MXFP4)         \
    X(AIRUN_TYPE_NVFP4)

#define AIRUN_CASE(TYPE) case TYPE: LAUNCH(TYPE); break;

extern "C" {

airun_status airun_attention_split(
    void* out, void* partial, const void* q, const void* k, const void* v,
    const void* positions, int32_t heads, int32_t kv_heads, int32_t dim,
    int32_t max_seq, float scale, int32_t window, int32_t splits, int32_t kv_type,
    cudaStream_t stream) {
    if (dim <= 0 || dim % 32 || dim > 256 || kv_heads <= 0 || heads % kv_heads || splits <= 0)
        return AIRUN_ERR_BAD_ARG;
    const dim3 grid(1, heads, splits), block(32, ATTN_WARPS_PREFILL);
    const size_t shared = ATTN_WARPS_PREFILL * (dim + 2) * sizeof(float);
#define LAUNCH_SPLIT(KV) attention_kernel<ATTN_WARPS_PREFILL, KV, true><<<grid, block, shared, stream>>>( \
        (float*)partial, (const float*)q, (const uint8_t*)k, (const uint8_t*)v,                        \
        (const int32_t*)positions, heads, kv_heads, dim, max_seq, scale, window)
    if (kv_type < KV_F16 || kv_type > KV_Q4_0) return AIRUN_ERR_BAD_ARG;
    const bool gqa = heads / kv_heads <= GQA_MAX && !gqa_decode_disabled() &&
        (dim == 64 || dim == 128 || dim == 256);
    if (gqa) {
        const float* qq = (const float*)q;
        const uint8_t* kk = (const uint8_t*)k;
        const uint8_t* vv = (const uint8_t*)v;
        const int32_t* pp = (const int32_t*)positions;
        float* part = (float*)partial;
        cudaError_t e;
        if (dim == 64)       e = launch_gqa_decode<64>(kv_type, part, qq, kk, vv, pp, heads, kv_heads, max_seq, scale, window, splits, stream);
        else if (dim == 128) e = launch_gqa_decode<128>(kv_type, part, qq, kk, vv, pp, heads, kv_heads, max_seq, scale, window, splits, stream);
        else                 e = launch_gqa_decode<256>(kv_type, part, qq, kk, vv, pp, heads, kv_heads, max_seq, scale, window, splits, stream);
        AIRUN_TRY(e);
    } else {
        switch (kv_type) {
            case KV_F16:  LAUNCH_SPLIT(KV_F16); break;
            case KV_Q8_0: LAUNCH_SPLIT(KV_Q8_0); break;
            default:      LAUNCH_SPLIT(KV_Q4_0); break;
        }
    }
#undef LAUNCH_SPLIT
    AIRUN_TRY(cudaGetLastError());
    attention_merge_kernel<<<heads, dim, 0, stream>>>((float*)out, (const float*)partial, dim, splits,
        (const int32_t*)positions, window, gqa ? GQA_KEYS_PER_SPLIT : 0);
    AIRUN_CHECK_LAUNCH();
}

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

airun_status airun_add_rmsnorm(
    void* out, void* x, const void* b, const void* weight,
    int32_t n, int32_t rows, float eps, cudaStream_t stream) {

    if (n <= 0 || rows <= 0) return AIRUN_OK;
    const int threads = (n < 1024) ? ((n + 31) / 32) * 32 : 1024;   // as airun_rmsnorm
    const size_t shared = ((threads + 31) / 32) * sizeof(float);
    add_rmsnorm_kernel<<<rows, threads, shared, stream>>>(
        (float*)out, (float*)x, (const float*)b, (const float*)weight, n, eps);
    AIRUN_CHECK_LAUNCH();
}
airun_status airun_argmax(const void* x, int32_t n, void* out, void* scratch, void* bump,
                          cudaStream_t stream) {
    if (n <= 0 || !scratch) return AIRUN_ERR_BAD_ARG;
    argmax_partial_kernel<<<ARGMAX_BLOCKS, ARGMAX_THREADS, 0, stream>>>(
        (const float*)x, n, (float*)scratch);
    argmax_final_kernel<<<1, ARGMAX_THREADS, 0, stream>>>(
        (const float*)scratch, n, (int32_t*)out, (int32_t*)bump);
    AIRUN_CHECK_LAUNCH();
}
airun_status airun_rope(
    void* x, const void* positions,
    int32_t n_tokens, int32_t n_heads, int32_t head_dim, int32_t rope_dim,
    airun_rope_params rp, int32_t interleaved, cudaStream_t stream) {

    if (n_tokens <= 0 || n_heads <= 0) return AIRUN_OK;
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) return AIRUN_ERR_BAD_ARG;

    const int half = rope_dim / 2;
    const int threads = (half < 256) ? half : 256;
    const dim3 grid(n_tokens, n_heads, (half + threads - 1) / threads);
    rope_kernel<<<grid, threads, 0, stream>>>(
        (float*)x, (const int32_t*)positions,
        n_heads, head_dim, rope_dim, rp, interleaved);
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
    int32_t kv_type, cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (kv_type != KV_F16) {
        if (head_dim % 32 != 0 || head_dim > 1024) return AIRUN_ERR_BAD_ARG;
        const dim3 grid(n_tokens, n_kv_heads), block(32, head_dim / 32);
#define LAUNCH_STORE(KV) kv_store_quant_kernel<KV><<<grid, block, 0, stream>>>(            \
            (uint8_t*)k_cache, (uint8_t*)v_cache, (const float*)k, (const float*)v,     \
            (const int32_t*)positions, n_kv_heads, head_dim, max_seq)
        switch (kv_type) {
            case KV_Q8_0: LAUNCH_STORE(KV_Q8_0); break;
            case KV_Q4_0: LAUNCH_STORE(KV_Q4_0); break;
            default: return AIRUN_ERR_BAD_ARG;
        }
#undef LAUNCH_STORE
        AIRUN_CHECK_LAUNCH();
    }
    const int threads = (head_dim < 256) ? head_dim : 256;
    const dim3 grid(n_tokens, n_kv_heads, (head_dim + threads - 1) / threads);
    kv_store_kernel<<<grid, threads, 0, stream>>>(
        (half*)k_cache, (half*)v_cache, (const float*)k, (const float*)v,
        (const int32_t*)positions, n_kv_heads, head_dim, max_seq);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_qk_norm_rope_store(
    void* q, const void* k, const void* v, const void* q_norm, const void* k_norm,
    void* k_cache, void* v_cache, const void* positions,
    int32_t n_tokens, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim, int32_t max_seq,
    float eps, int32_t rope_dim, airun_rope_params rp, int32_t interleaved,
    int32_t rope_on, int32_t kv_type, cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (head_dim % 32 != 0 || head_dim > 1024 || kv_type < KV_F16 || kv_type > KV_Q4_0) return AIRUN_ERR_BAD_ARG;
    if (rope_on && (rope_dim <= 0 || rope_dim > head_dim || rope_dim % 2 != 0)) return AIRUN_ERR_BAD_ARG;
    const dim3 grid(n_tokens, n_heads + 2 * n_kv_heads);
    const size_t shared = (head_dim + 32) * sizeof(float);
#define QK_LAUNCH(KVT) qk_norm_rope_store_kernel<KVT><<<grid, head_dim, shared, stream>>>(        \
        (float*)q, (const float*)k, (const float*)v, (const float*)q_norm, (const float*)k_norm, \
        (uint8_t*)k_cache, (uint8_t*)v_cache, (const int32_t*)positions, n_heads, n_kv_heads,     \
        head_dim, max_seq, eps, rope_dim, rp, interleaved, rope_on)
    switch (kv_type) {
        case KV_Q8_0: QK_LAUNCH(KV_Q8_0); break;
        case KV_Q4_0: QK_LAUNCH(KV_Q4_0); break;
        default:      QK_LAUNCH(KV_F16); break;
    }
#undef QK_LAUNCH
    AIRUN_CHECK_LAUNCH();
}
airun_status airun_attention(
    void* out, const void* q, const void* k_cache, const void* v_cache,
    const void* positions,
    int32_t n_tokens, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t max_seq, float scale, int32_t sliding_window, int32_t kv_type,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (head_dim % 32 != 0 || head_dim > 32 * MAX_DIM_PER_LANE) return AIRUN_ERR_BAD_ARG;
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) return AIRUN_ERR_BAD_ARG;
    if (kv_type < KV_F16 || kv_type > KV_Q4_0) return AIRUN_ERR_BAD_ARG;

    if (n_tokens >= FLASH_MIN_TOKENS && !flash_disabled() &&
        (head_dim == 64 || head_dim == 128 || head_dim == 256)) {
        // The tiles need more shared memory than Turing's 64 KiB.
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major >= 8) {
            const float* qq = (const float*)q;
            const uint8_t* kk = (const uint8_t*)k_cache;
            const uint8_t* vv = (const uint8_t*)v_cache;
            const int32_t* pp = (const int32_t*)positions;
            cudaError_t e;
            const bool f2 = !flash2_disabled();
            if (head_dim == 64)       e = f2 ? launch_flash2<64>(kv_type, (float*)out, qq, kk, vv, pp, n_tokens, n_heads, n_kv_heads, max_seq, scale, sliding_window, stream)
                                         : launch_flash<64>(kv_type, (float*)out, qq, kk, vv, pp, n_tokens, n_heads, n_kv_heads, max_seq, scale, sliding_window, stream);
            else if (head_dim == 128) e = f2 ? launch_flash2<128>(kv_type, (float*)out, qq, kk, vv, pp, n_tokens, n_heads, n_kv_heads, max_seq, scale, sliding_window, stream)
                                         : launch_flash<128>(kv_type, (float*)out, qq, kk, vv, pp, n_tokens, n_heads, n_kv_heads, max_seq, scale, sliding_window, stream);
            else                      e = f2 ? launch_flash2<256>(kv_type, (float*)out, qq, kk, vv, pp, n_tokens, n_heads, n_kv_heads, max_seq, scale, sliding_window, stream)
                                         : launch_flash<256>(kv_type, (float*)out, qq, kk, vv, pp, n_tokens, n_heads, n_kv_heads, max_seq, scale, sliding_window, stream);
            AIRUN_TRY(e);
            return AIRUN_OK;
        }
    }

    const dim3 grid(n_tokens, n_heads);

#define LAUNCH_ATTENTION(WARPS, KV)                                             \
    attention_kernel<WARPS, KV><<<grid, dim3(32, WARPS),                        \
        (2 * (WARPS) + (size_t)(WARPS) * head_dim) * sizeof(float), stream>>>(  \
        (float*)out, (const float*)q, (const uint8_t*)k_cache,                  \
        (const uint8_t*)v_cache, (const int32_t*)positions, n_heads,            \
        n_kv_heads, head_dim, max_seq, scale, sliding_window)
#define LAUNCH_KV(WARPS)                                                        \
    switch (kv_type) {                                                          \
        case KV_Q8_0: LAUNCH_ATTENTION(WARPS, KV_Q8_0); break;                  \
        case KV_Q4_0: LAUNCH_ATTENTION(WARPS, KV_Q4_0); break;                  \
        default:      LAUNCH_ATTENTION(WARPS, KV_F16); break;                   \
    }

    // Below roughly a couple of blocks per SM the grid cannot fill the card, so
    // the parallelism has to come from inside the block instead.
    if (n_tokens * n_heads <= 128) {
        LAUNCH_KV(ATTN_WARPS_DECODE);
    } else {
        LAUNCH_KV(ATTN_WARPS_PREFILL);
    }
#undef LAUNCH_KV
#undef LAUNCH_ATTENTION

    AIRUN_CHECK_LAUNCH();
}

} // extern "C"
