// QSA sparse attention (qwen4exp).
//
// An attention layer with a compress ratio r scores whole blocks of r cached
// tokens and attends to the best of them plus the incomplete tail block:
// `width = top_k + r - 1` cells in all, every visible cell while the context is
// no longer than that. A block's key is the mean of its tokens' raw indexer
// keys, RMS-normalized and rotated at the block's first position; a query's
// score for it is the sum over the indexer heads of relu(q_h . key). The
// arithmetic follows llama.cpp's qwen4exp graph (build_qsa_top_k), and the
// selection Strata's qsa_select: the `width` best cells, the tail first, ties
// going to the lower block and a partly selected block keeping its first cells.
//
// The cache holds the pooled keys of complete blocks ([ctx / r][dim], f32) and
// the raw keys of the newest tokens in a ring ([ring][dim], f16 like
// llama.cpp's indexer cache), from which a block is pooled once its last token
// arrives. Positions in one call are consecutive.

#include "airun.cuh"

// ring[pos % ring_len] = k, rounded to f16.
__global__ void qsa_store_kernel(half* __restrict__ ring, const float* __restrict__ k,
                                 const int32_t* __restrict__ pos, const int dim, const int ring_len) {
    const int t = blockIdx.x;
    half* dst = ring + (size_t)(pos[t] % ring_len) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = __float2half(k[(size_t)t * dim + i]);
}

// Pools every block the call completed: block b once position b*r + r-1 is in.
// One CUDA block of `dim` threads per candidate block.
__global__ void qsa_pool_kernel(float* __restrict__ pooled, const half* __restrict__ ring,
                                const float* __restrict__ norm, const int32_t* __restrict__ pos,
                                const int n, const int dim, const int ratio, const int ring_len,
                                const int max_blocks, const float eps, const int rope_dim,
                                const airun_rope_params rp) {
    extern __shared__ float row[];
    __shared__ float red[32];
    __shared__ float inv;
    const int b = pos[0] / ratio + blockIdx.x;
    if (b >= (pos[n - 1] + 1) / ratio || b >= max_blocks) return;
    const int i = threadIdx.x;
    // The slices summed in order, then scaled: llama.cpp's pooling.
    float s = 0.0f;
    for (int j = 0; j < ratio; ++j) s += __half2float(ring[(size_t)((b * ratio + j) % ring_len) * dim + i]);
    s *= 1.0f / (float)ratio;
    const float ss = block_reduce_sum(s * s, red);
    if (i == 0) inv = rsqrtf(ss / (float)dim + eps);
    __syncthreads();
    row[i] = s * inv * norm[i];
    __syncthreads();
    float v = row[i];
    const int half_ = rope_dim / 2;
    if (i < rope_dim) {   // NeoX pairs (p, p + rope_dim/2), as the attention's own RoPE
        const int pair = i < half_ ? i : i - half_;
        float c, sn;
        rope_cos_sin(rp, pair, rope_dim, b * ratio, &c, &sn);
        const float x0 = row[pair], x1 = row[pair + half_];
        v = i < half_ ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    pooled[(size_t)b * dim + i] = v;
}

// scores[qi][b] = sum_h relu(q[t0 + qi][h] . pooled[b]) for the complete blocks a
// query sees, skipped for queries whose whole context fits the budget.
// grid (x, queries), 8 warps; a warp scores one block at a time.
#define QSA_SCORE_WARPS 8
__global__ void __launch_bounds__(QSA_SCORE_WARPS * 32) qsa_score_kernel(
    float* __restrict__ scores, const int stride, const float* __restrict__ q,
    const float* __restrict__ pooled, const int32_t* __restrict__ pos, const int t0,
    const int heads, const int dim, const int ratio, const int width) {
    extern __shared__ float qs[];   // [heads][dim]
    const int qi = blockIdx.y;
    const int p = pos[t0 + qi];
    if (p + 1 <= width) return;
    const int nb = (p + 1) / ratio;
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) qs[i] = q[(size_t)(t0 + qi) * heads * dim + i];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int b = blockIdx.x * QSA_SCORE_WARPS + warp; b < nb; b += gridDim.x * QSA_SCORE_WARPS) {
        const float* kb = pooled + (size_t)b * dim;
        float s = 0.0f;
        for (int h = 0; h < heads; ++h) {
            float d = 0.0f;
            for (int i = lane; i < dim; i += 32) d = fmaf(qs[h * dim + i], kb[i], d);
            d = warp_reduce_sum(d);
            s += d > 0.0f ? d : 0.0f;
        }
        if (lane == 0) scores[(size_t)qi * stride + b] = s;
    }
}

// A total order on floats as unsigned keys.
__device__ __forceinline__ uint32_t qsa_key(float s) {
    const uint32_t b = __float_as_uint(s + 0.0f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// The cells query t0 + qi attends to: the incomplete tail block, then the
// best-scored complete blocks up to `width` cells. Writes `list` ([0] = count - 1,
// then the cells in ascending order: the decode kernels' `positions` and row
// indices) and/or a row of `mask` (bit j of word j/32: the prefill kernels').
// Only the words up to the query's position are written.
#define QSA_T 256
__global__ void __launch_bounds__(QSA_T) qsa_select_kernel(
    const float* __restrict__ scores, const int stride, const int32_t* __restrict__ pos,
    const int t0, const int ratio, const int width, int32_t* __restrict__ list,
    uint32_t* __restrict__ mask, const int words) {
    __shared__ int hist[256];
    __shared__ int s_gt[QSA_T], s_eq[QSA_T];
    __shared__ int s_digit, s_above;
    const int qi = blockIdx.x, t = t0 + qi, tid = threadIdx.x;
    const int p = pos[t];
    const int n_vis = p + 1;
    uint32_t* mrow = mask ? mask + (size_t)t * words : nullptr;
    if (n_vis <= width) {   // everything is visible
        if (list) {
            if (tid == 0) list[0] = p;
            for (int j = tid; j < n_vis; j += QSA_T) list[1 + j] = j;
        }
        if (mrow) for (int w = tid; w <= (p >> 5); w += QSA_T) mrow[w] = 0xffffffffu;
        return;
    }
    const int tail_start = n_vis / ratio * ratio;
    const int nb = tail_start / ratio;
    const int budget = width - (n_vis - tail_start);
    const float* sc = scores + (size_t)qi * stride;
    const int per = (nb + QSA_T - 1) / QSA_T;
    const int b0 = min(nb, tid * per), b1 = min(nb, b0 + per);

    // Radix select, 8 bits at a time: the key of the budget-th best cell.
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += QSA_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int b = b0; b < b1; ++b) {
            const uint32_t k = qsa_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], ratio);
        }
        __syncthreads();
        if (tid == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= budget) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;

    // Cells above the threshold all go; the ones at it fill what is left,
    // lowest block first.
    int gt = 0, eq = 0;
    for (int b = b0; b < b1; ++b) {
        const uint32_t k = qsa_key(sc[b]);
        gt += k > thr ? ratio : 0;
        eq += k == thr ? ratio : 0;
    }
    s_gt[tid] = gt;
    s_eq[tid] = eq;
    __syncthreads();
    if (tid == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < QSA_T; ++i) {
            const int g = s_gt[i], e = s_eq[i];
            s_gt[i] = ag;
            s_eq[i] = ae;
            ag += g;
            ae += e;
        }
    }
    __syncthreads();
    int my_eq = budget - above - s_eq[tid];
    my_eq = max(0, min(my_eq, eq));
    const int my_before_gt = s_gt[tid], my_before_eq = max(0, min(budget - above, s_eq[tid]));
    __syncthreads();

    if (mrow) {
        for (int w = tid; w <= (p >> 5); w += QSA_T) mrow[w] = 0u;
        __syncthreads();
    }
    int out = my_before_gt + my_before_eq;   // cells of lower blocks already placed
    int left = my_eq;
    for (int b = b0; b < b1; ++b) {
        const uint32_t k = qsa_key(sc[b]);
        int take = 0;
        if (k > thr) take = ratio;
        else if (k == thr) { take = min(ratio, left); left -= take; }
        for (int c = 0; c < take; ++c) {
            const int cell = b * ratio + c;
            if (list) list[1 + out] = cell;
            if (mrow) atomicOr(&mrow[cell >> 5], 1u << (cell & 31));
            ++out;
        }
    }
    for (int j = tail_start + tid; j < n_vis; j += QSA_T) {
        if (list) list[1 + budget + (j - tail_start)] = j;
        if (mrow) atomicOr(&mrow[j >> 5], 1u << (j & 31));
    }
    if (list && tid == 0) list[0] = width - 1;
}

extern "C" {

airun_status airun_qsa_store(void* ring, const void* k, const void* pos, int32_t n, int32_t dim,
                             int32_t ring_len, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    if (dim <= 0 || ring_len <= 0) return 1;
    qsa_store_kernel<<<n, min(dim, 256), 0, stream>>>((half*)ring, (const float*)k, (const int32_t*)pos,
                                                      dim, ring_len);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_qsa_pool(void* pooled, const void* ring, const void* norm, const void* pos, int32_t n,
                            int32_t dim, int32_t ratio, int32_t ring_len, int32_t max_blocks, float eps,
                            int32_t rope_dim, airun_rope_params rp, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    if (dim <= 0 || dim > 1024 || dim % 32 || ratio <= 0 || ring_len < n + ratio ||
        rope_dim < 0 || rope_dim > dim || rope_dim % 2) return 1;
    // A call of n consecutive positions completes at most n / ratio + 1 blocks.
    qsa_pool_kernel<<<n / ratio + 1, dim, dim * sizeof(float), stream>>>(
        (float*)pooled, (const half*)ring, (const float*)norm, (const int32_t*)pos, n, dim, ratio, ring_len,
        max_blocks, eps, rope_dim, rp);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_qsa_score(void* scores, int32_t stride, const void* q, const void* pooled, const void* pos,
                             int32_t t0, int32_t n_queries, int32_t heads, int32_t dim, int32_t ratio,
                             int32_t width, int32_t max_blocks, cudaStream_t stream) {
    if (n_queries <= 0) return AIRUN_OK;
    if (heads <= 0 || dim <= 0 || ratio <= 0 || stride < max_blocks) return 1;
    const int gx = max(1, min(128, (max_blocks + QSA_SCORE_WARPS - 1) / QSA_SCORE_WARPS));
    qsa_score_kernel<<<dim3(gx, n_queries), QSA_SCORE_WARPS * 32, heads * dim * sizeof(float), stream>>>(
        (float*)scores, stride, (const float*)q, (const float*)pooled, (const int32_t*)pos, t0, heads, dim,
        ratio, width);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_qsa_select(const void* scores, int32_t stride, const void* pos, int32_t t0,
                              int32_t n_queries, int32_t ratio, int32_t width, void* list, void* mask,
                              int32_t words, cudaStream_t stream) {
    if (n_queries <= 0) return AIRUN_OK;
    if (ratio <= 0 || width < ratio || (!list && !mask) || (list && n_queries != 1)) return 1;
    qsa_select_kernel<<<n_queries, QSA_T, 0, stream>>>((const float*)scores, stride, (const int32_t*)pos, t0,
                                                       ratio, width, (int32_t*)list, (uint32_t*)mask, words);
    AIRUN_CHECK_LAUNCH();
}

}  // extern "C"
