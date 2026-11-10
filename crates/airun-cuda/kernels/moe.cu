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
#include "vecdotq.cuh"

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

// moe_router_kernel for up to 32 * ROUTER_PER_LANE experts in one warp per
// token: every lane keeps its experts' logits in registers, and each of the k
// picks is a register max per lane and a five-step shuffle argmax, instead of
// one thread scanning all the experts k times (25 us per layer at 512
// experts). Same picks: the largest probability, the lowest index on a tie.
#define ROUTER_PER_LANE 16
__global__ void moe_router_warp_kernel(
    const float* __restrict__ logits, int32_t* __restrict__ out_ids, float* __restrict__ out_weights,
    const int n_experts, const int top_k, const int normalize) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    const float* row = logits + (size_t)token * n_experts;
    float v[ROUTER_PER_LANE];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < ROUTER_PER_LANE; ++i) {
        const int e = i * 32 + lane;
        v[i] = e < n_experts ? row[e] : -INFINITY;
        m = fmaxf(m, v[i]);
    }
    m = warp_reduce_max(m);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < ROUTER_PER_LANE; ++i) {
        v[i] = i * 32 + lane < n_experts ? __expf(v[i] - m) : -1.0f;
        sum += v[i] > 0.0f ? v[i] : 0.0f;
    }
    const float inv_total = 1.0f / warp_reduce_sum(sum);
    float weight_sum = 0.0f;
    float picked = 0.0f;   // lane j keeps pick j's probability
    for (int j = 0; j < top_k; ++j) {
        float best = -1.0f;
        int best_i = 0;
#pragma unroll
        for (int i = 0; i < ROUTER_PER_LANE; ++i) {
            if (v[i] > best) { best = v[i]; best_i = i; }
        }
        int best_e = best_i * 32 + lane;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
            if (ob > best || (ob == best && oe < best_e)) { best = ob; best_e = oe; }
        }
        const float p = best * inv_total;
        if (lane == (best_e & 31)) {
#pragma unroll
            for (int i = 0; i < ROUTER_PER_LANE; ++i) if (i == (best_e >> 5)) v[i] = -1.0f;
        }
        if (lane == 0) out_ids[(size_t)token * top_k + j] = best_e;
        if (lane == j) picked = p;
        weight_sum += p;
    }
    if (lane < top_k) {
        const float w = normalize && weight_sum > 0.0f ? picked * (1.0f / weight_sum) : picked;
        out_weights[(size_t)token * top_k + lane] = w;
    }
}

// ---------------------------------------------------------------------------
// Expert feed-forward
// ---------------------------------------------------------------------------

// h[token][k][row] = silu(gate_e[row] . x[token]) * (up_e[row] . x[token])
//
// Half a warp per output row: a 2560-wide row is 80 groups, which split
// evenly over 16 lanes (5 each) where a whole warp left them 3/2/2 apart, and
// the warp covers two rows.
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
    const uint8_t* __restrict__ up_stage,
    const int only) {                     // 0 every slot, 1 the resident ones, 2 the staged ones

    const int row = (blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y) * 2 + (threadIdx.x >> 4);
    const int sub = threadIdx.x & 15;
    // Both halves stay for the shuffles below; a missing row computes nothing.
    const bool live = row < ff;

    const int slot  = blockIdx.y;   // which of the token's top_k experts
    const int token = blockIdx.z;
    const int expert = ids[(size_t)token * top_k + slot];

    const uint8_t* g_base = gate_tab ? (const uint8_t*)gate_tab[expert] : gate_w + (size_t)expert * expert_stride;
    const uint8_t* u_base = up_tab ? (const uint8_t*)up_tab[expert] : up_w + (size_t)expert * expert_stride;
    // An expert outside VRAM was copied into the staging area beforehand.
    const int st = sidx ? sidx[(size_t)token * top_k + slot] : -1;
    if (only != 0 && (only == 2) != (st >= 0)) return;   // the whole block is one slot
    if (st >= 0) {
        g_base = gate_stage + (size_t)st * expert_stride;
        u_base = up_stage + (size_t)st * expert_stride;
    }
    const uint8_t* g_row = g_base + (size_t)(live ? row : 0) * row_bytes;
    const uint8_t* u_row = u_base + (size_t)(live ? row : 0) * row_bytes;
    const float4* xv = (const float4*)(x + (size_t)token * K);

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int grp = sub; live && grp < K / 32; grp += 16) {
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

#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        acc_g += __shfl_xor_sync(0xffffffffu, acc_g, o);
        acc_u += __shfl_xor_sync(0xffffffffu, acc_u, o);
    }

    if (live && sub == 0) {
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
    const uint8_t* __restrict__ down_stage,
    const int only) {                     // as moe_gate_up_kernel

    const int row = blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= n_embd) return;
    const int token = blockIdx.y;
    const int lane = threadIdx.x;

    // Lane s < top_k resolves slot s's expert row once; the others read it by
    // shuffle. The lanes then split the (slot, group) pairs between them: with
    // a short down row (640 = 20 groups) one slot at a time left 12 lanes idle
    // and chained the slots' loads one after another.
    const uint8_t* my_row = nullptr;
    float my_w = 0.0f;
    if (lane < top_k) {
        const int expert = ids[(size_t)token * top_k + lane];
        my_w = weights[(size_t)token * top_k + lane];
        const uint8_t* d_base = down_tab ? (const uint8_t*)down_tab[expert] : down_w + (size_t)expert * expert_stride;
        const int st = sidx ? sidx[(size_t)token * top_k + lane] : -1;
        if (st >= 0) d_base = down_stage + (size_t)st * expert_stride;
        my_row = d_base + (size_t)row * row_bytes;
        if (only != 0 && (only == 2) != (st >= 0)) my_row = nullptr;   // another pass's slot
    }
    const int groups = ff / 32;
    const int pairs = top_k * groups;
    float total = 0.0f;
    for (int p0 = 0; p0 < pairs; p0 += 32) {
        const int p = p0 + lane;
        const int slot = min(p, pairs - 1) / groups;
        const int grp = min(p, pairs - 1) - slot * groups;
        const unsigned long long rp = __shfl_sync(0xffffffffu, (unsigned long long)my_row, slot);
        const float w = __shfl_sync(0xffffffffu, my_w, slot);
        if (p >= pairs || rp == 0) continue;
        float dv[32];
        decode_group<TYPE>((const uint8_t*)rp, grp, dv);
        const float4* hg = (const float4*)(h + ((size_t)token * top_k + slot) * ff) + grp * 8;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 hh = hg[i];
            acc = fmaf(dv[i * 4 + 0], hh.x, acc);
            acc = fmaf(dv[i * 4 + 1], hh.y, acc);
            acc = fmaf(dv[i * 4 + 2], hh.z, acc);
            acc = fmaf(dv[i * 4 + 3], hh.w, acc);
        }
        total = fmaf(w, acc, total);
    }
    total = warp_reduce_sum(total);

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

// moe_stage_plan_kernel after a prefetch: the experts `pre_list` already put
// at staging slots 0..n_pre[0] keep them, and only the other host-side ones
// are listed, from slot n_pre[0] on (`first[0]`, where the copy starts).
// `stats` (nullable) adds up prefetched experts used, experts the prediction
// missed, and prefetched ones not routed.
__global__ void moe_stage_plan_pre_kernel(
    const int32_t* __restrict__ ids, const int n_ids, const uint8_t* __restrict__ resident, const int n_experts,
    const int32_t* __restrict__ pre_list, const int32_t* __restrict__ n_pre,
    int32_t* __restrict__ sidx, int32_t* __restrict__ list, int32_t* __restrict__ n_list,
    int32_t* __restrict__ first, unsigned int* __restrict__ stats) {
    extern __shared__ int32_t at[];   // n_experts
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) at[e] = -1;
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
        const int e = ids[i];
        if (!resident[e]) at[e] = -2;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int np = n_pre[0];
        int used = 0;
        for (int j = 0; j < np; ++j) {
            const int e = pre_list[j];
            if (at[e] == -2) { at[e] = j; ++used; }
        }
        int m = 0;
        for (int e = 0; e < n_experts; ++e) {
            if (at[e] == -2) { at[e] = np + m; list[m] = e; ++m; }
        }
        n_list[0] = m;
        first[0] = np;
        if (stats) {
            atomicAdd(stats + 0, (unsigned)used);
            atomicAdd(stats + 1, (unsigned)m);
            atomicAdd(stats + 2, (unsigned)(np - used));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) sidx[i] = at[ids[i]];
}

// Copies the staged experts out of host memory: 16-byte loads, many blocks in
// flight, which is what PCIe needs to run at its bandwidth -- the expert
// kernels' own narrow, dependent reads ran five times slower across it.
// Listed expert j lands at slot j + first[0] (`first` nullable: 0).
__global__ void moe_stage_copy_kernel(
    const unsigned long long* __restrict__ tab, const int32_t* __restrict__ list, const int32_t* __restrict__ n_list,
    const int32_t* __restrict__ first, uint8_t* __restrict__ stage, const size_t stride) {
    const int j = blockIdx.y;
    if (j >= n_list[0]) return;
    const uint4* src = (const uint4*)tab[list[j]];
    uint4* dst = (uint4*)(stage + (size_t)(j + (first ? first[0] : 0)) * stride);
    const size_t n16 = stride / 16;
    // Four independent loads per thread before any store: a link with
    // microseconds of latency needs many requests in flight, and a
    // load-then-store loop keeps only one per thread.
    const size_t step = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 7 * step < n16; i += 8 * step) {
        uint4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = src[i + u * step];
#pragma unroll
        for (int u = 0; u < 8; ++u) dst[i + u * step] = v[u];
    }
    for (; i < n16; i += step) dst[i] = src[i];
}


// ---------------------------------------------------------------------------
// Decode: the dense decode's integer path
// ---------------------------------------------------------------------------
//
// A decode step routes one token (a few with a speculative verify) to top_k
// experts. The kernels above decode every weight to a float and pay an FMA per
// element; for one token that is ALU- and latency-bound: on an RTX 3060 the
// 35B-A3B's expert kernels read their ~566 MB a token at ~18% of the memory
// bandwidth. Here the input is quantized to int8 per 32 (block_q8a, as the
// dense decode does) and each group is one __dp4a dot (vecdotq.cuh) -- the
// same trade llama.cpp's mmvq makes for its MoE. Not bit-identical to the
// float kernels: the activations round to 8 bits per 32-value group.

#define MOE_Q8_MAX_TOKENS 4   // decode and a speculative verify of up to 3 drafts
#define MOE_Q8_ROWS 2         // output rows per warp in gate_up

// Bytes of block_q8a scratch the launchers below need for `n_tokens` tokens.
static size_t moe_q8_scratch_bytes(int n_tokens, int k, int top_k, int ff) {
    const size_t elems = (size_t)n_tokens * (size_t)max(k, top_k * ff);
    return elems / 32 * sizeof(block_q8a);
}

template <int TYPE>
__global__ __launch_bounds__(32 * MOE_ROWS_PER_BLOCK) void moe_gate_up_q8_kernel(
    const uint8_t* __restrict__ gate_w,
    const uint8_t* __restrict__ up_w,
    const block_q8a* __restrict__ xq,     // [n_tokens][K / 32]
    const int32_t* __restrict__ ids,
    float* __restrict__ h,
    const int K,
    const int ff,
    const int top_k,
    const size_t row_bytes,
    const size_t expert_stride,
    const unsigned long long* __restrict__ gate_tab,
    const unsigned long long* __restrict__ up_tab,
    const int32_t* __restrict__ sidx,
    const uint8_t* __restrict__ gate_stage,
    const uint8_t* __restrict__ up_stage,
    const int only) {                     // as moe_gate_up_kernel

    const int slot  = blockIdx.y;
    const int token = blockIdx.z;
    const int expert = ids[(size_t)token * top_k + slot];
    const int st = sidx ? sidx[(size_t)token * top_k + slot] : -1;
    if (only != 0 && (only == 2) != (st >= 0)) return;   // the whole block is one slot

    const int row0 = (blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y) * MOE_Q8_ROWS;
    if (row0 >= ff) return;
    const uint8_t* g_base = gate_tab ? (const uint8_t*)gate_tab[expert] : gate_w + (size_t)expert * expert_stride;
    const uint8_t* u_base = up_tab ? (const uint8_t*)up_tab[expert] : up_w + (size_t)expert * expert_stride;
    if (st >= 0) {
        g_base = gate_stage + (size_t)st * expert_stride;
        u_base = up_stage + (size_t)st * expert_stride;
    }
    const int groups = K / 32;
    const block_q8a* a = xq + (size_t)token * groups;

    float acc_g[MOE_Q8_ROWS], acc_u[MOE_Q8_ROWS];
#pragma unroll
    for (int r = 0; r < MOE_Q8_ROWS; ++r) { acc_g[r] = 0.0f; acc_u[r] = 0.0f; }
    for (int g = threadIdx.x; g < groups; g += 32) {
#pragma unroll
        for (int r = 0; r < MOE_Q8_ROWS; ++r) {
            const size_t off = (size_t)min(row0 + r, ff - 1) * row_bytes;
            acc_g[r] += q8dot<TYPE>::dot(g_base + off, g, a);
            acc_u[r] += q8dot<TYPE>::dot(u_base + off, g, a);
        }
    }
#pragma unroll
    for (int r = 0; r < MOE_Q8_ROWS; ++r) {
        const float gs = warp_reduce_sum(acc_g[r]);
        const float us = warp_reduce_sum(acc_u[r]);
        if (threadIdx.x == 0 && row0 + r < ff) {
            const float silu = gs / (1.0f + __expf(-gs));
            h[((size_t)token * top_k + slot) * ff + row0 + r] = silu * us;
        }
    }
}

// y[token][row] += sum_k weight[token][k] * (down_e[row] . h[token][k]), with
// h quantized per (token, slot) row: hq[(token * top_k + slot) * ff / 32 + g].
template <int TYPE>
__global__ __launch_bounds__(32 * MOE_ROWS_PER_BLOCK) void moe_down_q8_kernel(
    const uint8_t* __restrict__ down_w,
    const block_q8a* __restrict__ hq,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    float* __restrict__ y,
    const int ff,
    const int n_embd,
    const int top_k,
    const size_t row_bytes,
    const size_t expert_stride,
    const unsigned long long* __restrict__ down_tab,
    const int32_t* __restrict__ sidx,
    const uint8_t* __restrict__ down_stage,
    const int only) {

    const int row = blockIdx.x * MOE_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= n_embd) return;
    const int token = blockIdx.y;
    const int lane = threadIdx.x;

    // As moe_down_kernel: lane s < top_k resolves slot s's row, the lanes
    // split the (slot, group) pairs.
    const uint8_t* my_row = nullptr;
    float my_w = 0.0f;
    if (lane < top_k) {
        const int expert = ids[(size_t)token * top_k + lane];
        my_w = weights[(size_t)token * top_k + lane];
        const uint8_t* d_base = down_tab ? (const uint8_t*)down_tab[expert] : down_w + (size_t)expert * expert_stride;
        const int st = sidx ? sidx[(size_t)token * top_k + lane] : -1;
        if (st >= 0) d_base = down_stage + (size_t)st * expert_stride;
        my_row = d_base + (size_t)row * row_bytes;
        if (only != 0 && (only == 2) != (st >= 0)) my_row = nullptr;
    }
    const int groups = ff / 32;
    const int pairs = top_k * groups;
    float total = 0.0f;
    for (int p0 = 0; p0 < pairs; p0 += 32) {
        const int p = p0 + lane;
        const int slot = min(p, pairs - 1) / groups;
        const int grp = min(p, pairs - 1) - slot * groups;
        const unsigned long long rp = __shfl_sync(0xffffffffu, (unsigned long long)my_row, slot);
        const float w = __shfl_sync(0xffffffffu, my_w, slot);
        if (p >= pairs || rp == 0) continue;
        const block_q8a* a = hq + ((size_t)token * top_k + slot) * groups;
        total = fmaf(w, q8dot<TYPE>::dot((const uint8_t*)rp, grp, a), total);
    }
    total = warp_reduce_sum(total);
    if (lane == 0) y[(size_t)token * n_embd + row] += total;
}

// Quantizes n_groups 32-value groups of `x` into block_q8a, as the dense
// decode's quantize_q8a_kernel (matmul.cu) does: amax/127 scale, round to
// nearest even, the code sum folded into ds.y.
__global__ void moe_quantize_q8a_kernel(const float* __restrict__ x, block_q8a* __restrict__ out, const int n_groups) {
    const int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (g >= n_groups) return;
    const float v = x[(size_t)g * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    const int qi = __float2int_rn(v * id);
    int sum = qi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    out[g].qs[lane] = (int8_t)qi;
    if (lane == 0) out[g].ds = __floats2half2_rn(d, d * (float)sum);
}

static void moe_quantize_q8a(const float* x, block_q8a* out, int64_t n_groups, cudaStream_t stream) {
    const int per_block = 8;   // groups (warps) per block
    moe_quantize_q8a_kernel<<<(unsigned)((n_groups + per_block - 1) / per_block), 32 * per_block, 0, stream>>>(
        x, out, (int)n_groups);
}

// Whether the integer decode kernels serve this call.
static bool moe_q8_ok(int32_t type_id, const void* q8, int32_t n_tokens) {
    if (!q8 || n_tokens > MOE_Q8_MAX_TOKENS) return false;
    static const bool off = getenv("AIRUN_MOE_F32") != nullptr;
    if (off) return false;
    switch (type_id) {
#define Q8CASE(TYPE) case TYPE:
        AIRUN_FOR_EACH_Q8_TYPE(Q8CASE)
#undef Q8CASE
        return true;
        default:
            return false;
    }
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

// `n` bytes from `src` to shared `dst`: 16 at a time, four loads in flight
// per thread, when both allow it (an expert in host memory is read across
// PCIe, whose latency needs many wide requests outstanding), else 4 or 1.
__device__ __forceinline__ void moe_load_rows(uint8_t* dst, const uint8_t* src, int n) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nt = blockDim.x * blockDim.y;
    if ((((uintptr_t)src | (uintptr_t)dst) & 15) == 0 && (n & 15) == 0) {
        const uint4* s4 = (const uint4*)src;
        uint4* d4 = (uint4*)dst;
        const int n16 = n / 16;
        int i = tid;
        for (; i + 3 * nt < n16; i += 4 * nt) {
            const uint4 a = s4[i], b = s4[i + nt], c = s4[i + 2 * nt], d = s4[i + 3 * nt];
            d4[i] = a; d4[i + nt] = b; d4[i + 2 * nt] = c; d4[i + 3 * nt] = d;
        }
        for (; i < n16; i += nt) d4[i] = s4[i];
    } else if (((uintptr_t)src & 3) == 0 && (n & 3) == 0) {
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

// ---------------------------------------------------------------------------
// Grouped experts on tensor cores (sm_80+).
//
// The kernels above dequantize a row and FMA it in f32 against up to 16 of the
// expert's tokens at a time, four rows per block: at prefill sizes that ran at
// ~2% of the card's arithmetic. Here a block owns a tile of an expert's rows
// and all of its tokens. It decodes each 128-wide slice of the rows to fp16 in
// shared memory once per 32 tokens -- every thread one 32-element group -- and
// multiplies on the tensor cores (mma.sync m16n8k16, f32 accumulation). The
// activations are fp16 copies made once per layer (moe_half_kernel for x; the
// gate/up kernel writes h in fp16), read as B fragments straight from L2, so a
// block stages nothing but weights and syncs twice per slice.
//
// Experts in VRAM are decoded in place (`direct`); staging their rows only
// cost shared memory and occupancy. Experts in host memory are first copied
// into shared memory with wide loads, which their PCIe reads need. These
// results match the f32 kernels to fp16 rounding, not bit for bit.
// ---------------------------------------------------------------------------

#define MMA_KC 128         // K elements decoded per slice: 4 groups per row
#define MMA_TT 32          // tokens per pass
#define MMA_LD (MMA_KC + 8) // halves per staged row: conflict-free fragment loads
#define MMA_GU_ROWS 16     // gate rows (and up rows) per gate/up block
#define MMA_DN_ROWS 32     // down rows per block
#define MMA_MIN_TOKENS 64  // smaller batches stay on the f32 grouped kernels

__device__ __forceinline__ void moe_mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)c; (void)a; (void)b0; (void)b1;   // m16n8k16 is sm_80+; sub-sm_80 never launch this kernel
#endif
}

__device__ __forceinline__ half moe_to_half(float v) {
    return __float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

// xh = fp16(x), saturated at fp16's range.
__global__ void moe_half_kernel(const float* __restrict__ x, half* __restrict__ xh, const size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) xh[i] = moe_to_half(x[i]);
}

// One 32-element group of a row, decoded to fp16 at dst.
template <int TYPE>
__device__ __forceinline__ void moe_decode_half(const uint8_t* row, int grp, half* dst) {
    float v[32];
    decode_group<TYPE>(row, grp, v);
#pragma unroll
    for (int i = 0; i < 32; i += 2) *(half2*)(dst + i) = __floats2half2_rn(v[i], v[i + 1]);
}

// Decodes rows [0, n_rows) of a slice into wt: thread t takes row t / 4,
// group t % 4. A row past `rows` decodes as zero.
template <int TYPE>
__device__ __forceinline__ void moe_decode_slice(half* wt, const uint8_t* const* src, int rows_a, int rows_b,
                                                 int split, size_t row_bytes, int kc) {
    const int r = threadIdx.x >> 2, g = threadIdx.x & 3;
    half* dst = wt + r * MMA_LD + g * 32;
    const bool second = r >= split;
    const int rr = second ? r - split : r;
    if (rr < (second ? rows_b : rows_a)) {
        moe_decode_half<TYPE>(src[second] + (size_t)rr * row_bytes, kc / 32 + g, dst);
    } else {
#pragma unroll
        for (int i = 0; i < 32; i += 2) *(half2*)(dst + i) = __floats2half2_rn(0.0f, 0.0f);
    }
}

// acc[nt][4] += W[16 rows at w0] . X^T over one slice, X's rows for the lane's
// two n8 tiles at xr[0], xr[1] (fp16, global, K contiguous).
__device__ __forceinline__ void moe_mma_slice(float (&acc)[2][4], const half* w0, const half* const (&xr)[2],
                                              int kc, int lane) {
    const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
    for (int ks = 0; ks < MMA_KC; ks += 16) {
        uint32_t a[4];
        a[0] = *(const uint32_t*)(w0 + gid * MMA_LD + ks + tig * 2);
        a[1] = *(const uint32_t*)(w0 + (gid + 8) * MMA_LD + ks + tig * 2);
        a[2] = *(const uint32_t*)(w0 + gid * MMA_LD + ks + tig * 2 + 8);
        a[3] = *(const uint32_t*)(w0 + (gid + 8) * MMA_LD + ks + tig * 2 + 8);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const half* p = xr[nt] + kc + ks + tig * 2;
            moe_mma16816(acc[nt], a, __ldg((const unsigned int*)p), __ldg((const unsigned int*)(p + 8)));
        }
    }
}

// hh[pair][row] = fp16(silu(gate . x) * (up . x)) for every pair routed to
// expert blockIdx.y, rows blockIdx.x * 16 ...; 4 warps: (gate, up) x (tokens
// 0-15, 16-31). With a residency map the direct launch takes the experts in
// VRAM and the staged one those in host memory.
template <int TYPE>
__global__ __launch_bounds__(128) void moe_mma_gate_up_kernel(
    const unsigned long long* __restrict__ gate_tab, const unsigned long long* __restrict__ up_tab,
    const uint8_t* __restrict__ gate_w, const uint8_t* __restrict__ up_w, const size_t expert_stride,
    const half* __restrict__ xh, const int32_t* __restrict__ perm, const int32_t* __restrict__ offsets,
    half* __restrict__ hh, const int K, const int ff, const int top_k, const size_t row_bytes,
    const uint8_t* __restrict__ resident, const int direct) {
    extern __shared__ __align__(16) uint8_t msmem[];
    const int e = blockIdx.y;
    const int p0 = offsets[e], p1 = offsets[e + 1];
    if (p0 == p1) return;
    if (resident && (resident[e] != 0) != (direct != 0)) return;
    const int row0 = blockIdx.x * MMA_GU_ROWS;
    const int rows = min(MMA_GU_ROWS, ff - row0);
    const size_t raw_bytes = direct ? 0 : (size_t)MMA_GU_ROWS * row_bytes;
    uint8_t* raw = msmem;                                                // [gate rows][up rows]
    half* wt = (half*)(msmem + ((2 * raw_bytes + 15) & ~(size_t)15));   // [32][MMA_LD]
    float* out = (float*)wt;                                             // [2][16][MMA_TT], after a pass
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint8_t* g_src = (gate_tab ? (const uint8_t*)gate_tab[e] : gate_w + (size_t)e * expert_stride) + (size_t)row0 * row_bytes;
    const uint8_t* u_src = (up_tab ? (const uint8_t*)up_tab[e] : up_w + (size_t)e * expert_stride) + (size_t)row0 * row_bytes;
    if (!direct) {
        moe_load_rows(raw, g_src, rows * (int)row_bytes);
        moe_load_rows(raw + raw_bytes, u_src, rows * (int)row_bytes);
        g_src = raw;
        u_src = raw + raw_bytes;
    }
    const uint8_t* src[2] = {g_src, u_src};
    const int mat = warp >> 1, half_t = warp & 1;
    const int gid = lane >> 2, tig = lane & 3;

    for (int c0 = p0; c0 < p1; c0 += MMA_TT) {
        const int nt = min(MMA_TT, p1 - c0);
        const half* xr[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            // Columns past the expert's tokens read its last one; never stored.
            const int t = min(half_t * 16 + i * 8 + gid, nt - 1);
            xr[i] = xh + (size_t)(perm[c0 + t] / top_k) * K;
        }
        float acc[2][4] = {};
        for (int kc = 0; kc < K; kc += MMA_KC) {
            __syncthreads();   // the previous slice is read; staged rows have landed
            moe_decode_slice<TYPE>(wt, src, rows, rows, MMA_GU_ROWS, row_bytes, kc);
            __syncthreads();
            moe_mma_slice(acc, wt + mat * MMA_GU_ROWS * MMA_LD, xr, kc, lane);
        }
        __syncthreads();
        // Gate and up of a row sit in different warps: meet in shared memory.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int t = half_t * 16 + i * 8 + tig * 2;
            float* o = out + mat * MMA_GU_ROWS * MMA_TT;
            o[gid * MMA_TT + t] = acc[i][0];
            o[gid * MMA_TT + t + 1] = acc[i][1];
            o[(gid + 8) * MMA_TT + t] = acc[i][2];
            o[(gid + 8) * MMA_TT + t + 1] = acc[i][3];
        }
        __syncthreads();
        for (int i = tid; i < MMA_GU_ROWS * MMA_TT; i += 128) {
            const int r = i / MMA_TT, t = i % MMA_TT;
            if (t < nt && r < rows) {
                const float g = out[r * MMA_TT + t], u = out[MMA_GU_ROWS * MMA_TT + r * MMA_TT + t];
                hh[(size_t)perm[c0 + t] * ff + row0 + r] = moe_to_half(g / (1.0f + __expf(-g)) * u);
            }
        }
    }
}

// ypair[pair][row] = down_e[row] . h[pair] for the pairs routed to blockIdx.y,
// rows blockIdx.x * 32 ...; 4 warps: (rows 0-15, 16-31) x (tokens 0-15, 16-31).
template <int TYPE>
__global__ __launch_bounds__(128) void moe_mma_down_kernel(
    const unsigned long long* __restrict__ down_tab, const uint8_t* __restrict__ down_w, const size_t expert_stride,
    const half* __restrict__ hh, const int32_t* __restrict__ perm, const int32_t* __restrict__ offsets,
    float* __restrict__ ypair, const int ff, const int n_embd, const size_t row_bytes,
    const uint8_t* __restrict__ resident, const int direct) {
    extern __shared__ __align__(16) uint8_t msmem[];
    const int e = blockIdx.y;
    const int p0 = offsets[e], p1 = offsets[e + 1];
    if (p0 == p1) return;
    if (resident && (resident[e] != 0) != (direct != 0)) return;
    const int row0 = blockIdx.x * MMA_DN_ROWS;
    const int rows = min(MMA_DN_ROWS, n_embd - row0);
    const size_t raw_bytes = direct ? 0 : (size_t)MMA_DN_ROWS * row_bytes;
    uint8_t* raw = msmem;
    half* wt = (half*)(msmem + ((raw_bytes + 15) & ~(size_t)15));   // [32][MMA_LD]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const uint8_t* d_src = (down_tab ? (const uint8_t*)down_tab[e] : down_w + (size_t)e * expert_stride)
        + (size_t)row0 * row_bytes;
    if (!direct) {
        moe_load_rows(raw, d_src, rows * (int)row_bytes);
        d_src = raw;
    }
    const uint8_t* src[2] = {d_src, d_src + 16 * row_bytes};
    const int mt = warp >> 1, half_t = warp & 1;
    const int gid = lane >> 2, tig = lane & 3;

    for (int c0 = p0; c0 < p1; c0 += MMA_TT) {
        const int nt = min(MMA_TT, p1 - c0);
        const half* xr[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int t = min(half_t * 16 + i * 8 + gid, nt - 1);
            xr[i] = hh + (size_t)perm[c0 + t] * ff;
        }
        float acc[2][4] = {};
        for (int kc = 0; kc < ff; kc += MMA_KC) {
            __syncthreads();
            moe_decode_slice<TYPE>(wt, src, min(rows, 16), max(rows - 16, 0), 16, row_bytes, kc);
            __syncthreads();
            moe_mma_slice(acc, wt + mt * 16 * MMA_LD, xr, kc, lane);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int t = half_t * 16 + i * 8 + tig * 2;
            const int r = mt * 16 + gid;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int tt = t + (j & 1), rr = r + (j >> 1) * 8;
                if (tt < nt && rr < rows) ypair[(size_t)perm[c0 + tt] * n_embd + row0 + rr] = acc[i][j];
            }
        }
    }
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

// The tensor-core grouped kernels: sm_80+, K and ff in whole 64-wide slices.
// AIRUN_MOE_MMA=0 keeps the f32 kernels, to compare against.
static bool moe_mma_usable(int k, int ff) {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_MOE_MMA");
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cached = (major >= 8 && !(v && v[0] == '0')) ? 1 : 0;
    }
    return cached == 1 && k % MMA_KC == 0 && ff % MMA_KC == 0;
}

extern "C" {

airun_status airun_moe_router(
    const void* logits, void* out_ids, void* out_weights,
    int32_t n_tokens, int32_t n_experts, int32_t top_k, int32_t normalize,
    cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (top_k <= 0 || top_k > n_experts) return AIRUN_ERR_BAD_ARG;

    if (n_experts <= 32 * ROUTER_PER_LANE && top_k <= 32) {
        moe_router_warp_kernel<<<n_tokens, 32, 0, stream>>>(
            (const float*)logits, (int32_t*)out_ids, (float*)out_weights, n_experts, top_k, normalize);
        AIRUN_CHECK_LAUNCH();
    }
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
    int32_t only, void* q8, cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (k % 32 != 0) return AIRUN_ERR_BAD_ARG;

    const dim3 block(32, MOE_ROWS_PER_BLOCK);
    if (moe_q8_ok(type_id, q8, n_tokens)) {
        moe_quantize_q8a((const float*)x, (block_q8a*)q8, (int64_t)n_tokens * k / 32, stream);
        const dim3 qgrid((ff + MOE_Q8_ROWS * MOE_ROWS_PER_BLOCK - 1) / (MOE_Q8_ROWS * MOE_ROWS_PER_BLOCK), top_k,
                         n_tokens);
#define LAUNCH(TYPE)                                                          \
    moe_gate_up_q8_kernel<TYPE><<<qgrid, block, 0, stream>>>(                 \
        (const uint8_t*)gate_w, (const uint8_t*)up_w, (const block_q8a*)q8,   \
        (const int32_t*)ids, (float*)h, k, ff, top_k,                         \
        (size_t)row_bytes, (size_t)expert_stride,                             \
        (const unsigned long long*)gate_tab, (const unsigned long long*)up_tab,  \
        (const int32_t*)sidx, (const uint8_t*)gate_stage, (const uint8_t*)up_stage, only);
        switch (type_id) {
            AIRUN_FOR_EACH_Q8_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_ARG;
        }
#undef LAUNCH
        AIRUN_CHECK_LAUNCH();
    }
    const dim3 grid((ff + 2 * MOE_ROWS_PER_BLOCK - 1) / (2 * MOE_ROWS_PER_BLOCK), top_k, n_tokens);

#define LAUNCH(TYPE)                                                          \
    moe_gate_up_kernel<TYPE><<<grid, block, 0, stream>>>(                     \
        (const uint8_t*)gate_w, (const uint8_t*)up_w, (const float*)x,        \
        (const int32_t*)ids, (float*)h, k, ff, top_k,                         \
        (size_t)row_bytes, (size_t)expert_stride,                             \
        (const unsigned long long*)gate_tab, (const unsigned long long*)up_tab,  \
        (const int32_t*)sidx, (const uint8_t*)gate_stage, (const uint8_t*)up_stage, only);

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
    int32_t only, void* q8, cudaStream_t stream) {

    if (n_tokens <= 0) return AIRUN_OK;
    if (ff % 32 != 0) return AIRUN_ERR_BAD_ARG;

    const dim3 block(32, MOE_ROWS_PER_BLOCK);
    const dim3 grid((n_embd + MOE_ROWS_PER_BLOCK - 1) / MOE_ROWS_PER_BLOCK, n_tokens);
    if (moe_q8_ok(type_id, q8, n_tokens)) {
        moe_quantize_q8a((const float*)h, (block_q8a*)q8, (int64_t)n_tokens * top_k * ff / 32, stream);
#define LAUNCH(TYPE)                                                          \
    moe_down_q8_kernel<TYPE><<<grid, block, 0, stream>>>(                     \
        (const uint8_t*)down_w, (const block_q8a*)q8, (const int32_t*)ids,    \
        (const float*)weights, (float*)y, ff, n_embd, top_k,                  \
        (size_t)row_bytes, (size_t)expert_stride, (const unsigned long long*)down_tab, \
        (const int32_t*)sidx, (const uint8_t*)down_stage, only);
        switch (type_id) {
            AIRUN_FOR_EACH_Q8_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_ARG;
        }
#undef LAUNCH
        AIRUN_CHECK_LAUNCH();
    }

#define LAUNCH(TYPE)                                                          \
    moe_down_kernel<TYPE><<<grid, block, 0, stream>>>(                        \
        (const uint8_t*)down_w, (const float*)h, (const int32_t*)ids,         \
        (const float*)weights, (float*)y, ff, n_embd, top_k,                  \
        (size_t)row_bytes, (size_t)expert_stride, (const unsigned long long*)down_tab, \
        (const int32_t*)sidx, (const uint8_t*)down_stage, only);

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
    void* q8, cudaStream_t stream) {
    return airun_moe_gate_up_tab(type_id, gate_w, up_w, x, ids, h, k, ff, n_tokens, top_k, row_bytes,
                                 expert_stride, nullptr, nullptr, nullptr, nullptr, nullptr, 0, q8, stream);
}

airun_status airun_moe_down(
    int32_t type_id,
    const void* down_w, const void* h, const void* ids, const void* weights, void* y,
    int32_t ff, int32_t n_embd, int32_t n_tokens, int32_t top_k,
    uint64_t row_bytes, uint64_t expert_stride,
    void* q8, cudaStream_t stream) {
    return airun_moe_down_tab(type_id, down_w, h, ids, weights, y, ff, n_embd, n_tokens, top_k, row_bytes,
                              expert_stride, nullptr, nullptr, nullptr, 0, q8, stream);
}

// Bytes of scratch the q8 argument of the expert launchers needs; 0 when the
// integer kernels never run (more tokens than a decode).
uint64_t airun_moe_q8_scratch_bytes(int32_t k, int32_t top_k, int32_t ff) {
    return moe_q8_scratch_bytes(MOE_Q8_MAX_TOKENS, k, top_k, ff);
}

airun_status airun_moe_stage_plan(const void* ids, int32_t n_ids, const void* resident, int32_t n_experts,
                                  void* sidx, void* list, void* n_list, cudaStream_t stream) {
    if (n_ids <= 0) return AIRUN_OK;
    moe_stage_plan_kernel<<<1, 256, n_experts * sizeof(int32_t), stream>>>(
        (const int32_t*)ids, n_ids, (const uint8_t*)resident, n_experts, (int32_t*)sidx, (int32_t*)list,
        (int32_t*)n_list);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_moe_stage_plan_pre(const void* ids, int32_t n_ids, const void* resident, int32_t n_experts,
                                      const void* pre_list, const void* n_pre, void* sidx, void* list, void* n_list,
                                      void* first, void* stats, cudaStream_t stream) {
    if (n_ids <= 0) return AIRUN_OK;
    moe_stage_plan_pre_kernel<<<1, 256, n_experts * sizeof(int32_t), stream>>>(
        (const int32_t*)ids, n_ids, (const uint8_t*)resident, n_experts, (const int32_t*)pre_list,
        (const int32_t*)n_pre, (int32_t*)sidx, (int32_t*)list, (int32_t*)n_list, (int32_t*)first,
        (unsigned int*)stats);
    AIRUN_CHECK_LAUNCH();
}

// Up to `cap` experts of `stride` bytes (a multiple of 16), from slot first[0]
// on (`first` nullable: 0).
airun_status airun_moe_stage_copy(const void* tab, const void* list, const void* n_list, const void* first,
                                  void* stage, uint64_t stride, int32_t cap, cudaStream_t stream) {
    if (cap <= 0) return AIRUN_OK;
    if (stride % 16 != 0) return AIRUN_ERR_BAD_ARG;
    // Blocks per expert: AIRUN_STAGE_BLOCKS, else 4. The copy runs beside the
    // resident experts' kernels, and each of its blocks holds SM slots for the
    // whole PCIe round trip: at 32 blocks an expert (256 at once) it filled
    // the GPU and stalled them. Four, with eight 16-byte loads in flight per
    // thread, still keep ~130 KB of reads outstanding, past what PCIe 4.0 x16
    // needs. 35B-A3B Q4_K_M on an RTX 3060, 22% of experts from RAM:
    // 32 blocks 51.6 tok/s, 8 55, 4 58.7-60.6, 3 60.0-60.4, 1 55.7-57.4.
    static const int per = [] {
        const char* v = getenv("AIRUN_STAGE_BLOCKS");
        const int n = v ? atoi(v) : 0;
        return n > 0 ? n : 4;
    }();
    const dim3 grid(per, cap);
    moe_stage_copy_kernel<<<grid, 256, 0, stream>>>(
        (const unsigned long long*)tab, (const int32_t*)list, (const int32_t*)n_list, (const int32_t*)first,
        (uint8_t*)stage, (size_t)stride);
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
    uint64_t gate_row_bytes, uint64_t down_row_bytes, const void* gate_res, const void* down_res,
    void* xh, void* side, void* ev_fork, void* ev_join, cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    if (k % 32 != 0 || ff % 32 != 0) return AIRUN_ERR_BAD_ARG;
    // Below MMA_MIN_TOKENS the f32 kernels are cheap enough and keep a short
    // prompt's numbers those of the reference; fp16 can tip a near tie.
    if (n_tokens >= MMA_MIN_TOKENS && xh && moe_mma_usable(k, ff)) {
        // Experts in VRAM decode straight from it: staging their rows only
        // cost shared memory, and with it occupancy (20 ms against 6 per
        // layer on the 5060 Ti). Experts in host memory are staged with wide
        // loads, which their PCIe reads need. A table without a residency
        // map is all staged; no table is all in VRAM.
        const size_t gu_tile = (size_t)2 * MMA_GU_ROWS * MMA_LD * sizeof(half);
        const size_t dn_tile = (size_t)MMA_DN_ROWS * MMA_LD * sizeof(half);
        const size_t gu_stage = ((2 * MMA_GU_ROWS * gate_row_bytes + 15) & ~(size_t)15) + gu_tile;
        const size_t dn_stage = ((MMA_DN_ROWS * down_row_bytes + 15) & ~(size_t)15) + dn_tile;
        if (gu_stage <= 96 * 1024 && dn_stage <= 96 * 1024) {
            // (direct, residency map) per pass
            struct Pass { int direct; const uint8_t* res; };
            Pass gu_pass[2], dn_pass[2];
            int n_gu = 0, n_dn = 0;
            if (!gate_tab) gu_pass[n_gu++] = {1, nullptr};
            else if (!gate_res) gu_pass[n_gu++] = {0, nullptr};
            else { gu_pass[n_gu++] = {1, (const uint8_t*)gate_res}; gu_pass[n_gu++] = {0, (const uint8_t*)gate_res}; }
            if (!down_tab) dn_pass[n_dn++] = {1, nullptr};
            else if (!down_res) dn_pass[n_dn++] = {0, nullptr};
            else { dn_pass[n_dn++] = {1, (const uint8_t*)down_res}; dn_pass[n_dn++] = {0, (const uint8_t*)down_res}; }
            const int n_embd = k;
            auto launch_gu = [&](int pi, cudaStream_t st) -> airun_status {
                const int direct = gu_pass[pi].direct;
                const uint8_t* res = gu_pass[pi].res;
                const size_t gu_smem = direct ? gu_tile : gu_stage;
                const dim3 grid((ff + MMA_GU_ROWS - 1) / MMA_GU_ROWS, n_experts);
#define LAUNCH(TYPE)                                                                                                   cudaFuncSetAttribute(moe_mma_gate_up_kernel<TYPE>,                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gu_smem);                                        moe_mma_gate_up_kernel<TYPE><<<grid, 128, gu_smem, st>>>(                                                  (const unsigned long long*)gate_tab, (const unsigned long long*)up_tab,                                (const uint8_t*)gate_w, (const uint8_t*)up_w, (size_t)gate_stride, (const half*)xh,                     (const int32_t*)perm, (const int32_t*)offsets, (half*)h, k, ff, top_k, (size_t)gate_row_bytes,                      res, direct);
                switch (gate_type) {
                    AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
                    default: return AIRUN_ERR_BAD_ARG;
                }
#undef LAUNCH
                return AIRUN_OK;
            };
            auto launch_dn = [&](int pi, cudaStream_t st) -> airun_status {
                const int direct = dn_pass[pi].direct;
                const uint8_t* res = dn_pass[pi].res;
                const size_t dn_smem = direct ? dn_tile : dn_stage;
                const dim3 grid((n_embd + MMA_DN_ROWS - 1) / MMA_DN_ROWS, n_experts);
#define LAUNCH(TYPE)                                                                                                   cudaFuncSetAttribute(moe_mma_down_kernel<TYPE>,                                                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dn_smem);                                        moe_mma_down_kernel<TYPE><<<grid, 128, dn_smem, st>>>(                                                     (const unsigned long long*)down_tab, (const uint8_t*)down_w, (size_t)down_stride,                      (const half*)h, (const int32_t*)perm, (const int32_t*)offsets, (float*)ypair, ff, n_embd,                     (size_t)down_row_bytes, res, direct);
                switch (down_type) {
                    AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
                    default: return AIRUN_ERR_BAD_ARG;
                }
#undef LAUNCH
                return AIRUN_OK;
            };
            // The activations in fp16 once, read by every block as B fragments.
            {
                const size_t nx = (size_t)n_tokens * k;
                moe_half_kernel<<<(unsigned)((nx + 255) / 256), 256, 0, stream>>>((const float*)x, (half*)xh, nx);
            }
            // The experts in host memory are bound by PCIe, the ones in VRAM by
            // the arithmetic: with a side stream the two passes run at once.
            // Each pass touches only its own experts' pairs, so they share h
            // and ypair without overlapping.
            if (side && n_gu == 2 && n_dn == 2) {
                AIRUN_TRY(cudaEventRecord((cudaEvent_t)ev_fork, stream));
                AIRUN_TRY(cudaStreamWaitEvent((cudaStream_t)side, (cudaEvent_t)ev_fork, 0));
                for (int pi = 0; pi < 2; ++pi) {
                    const cudaStream_t st = pi == 0 ? stream : (cudaStream_t)side;
                    airun_status e = launch_gu(pi, st);
                    if (e != AIRUN_OK) return e;
                    e = launch_dn(pi, st);
                    if (e != AIRUN_OK) return e;
                }
                AIRUN_TRY(cudaEventRecord((cudaEvent_t)ev_join, (cudaStream_t)side));
                AIRUN_TRY(cudaStreamWaitEvent(stream, (cudaEvent_t)ev_join, 0));
            } else {
                for (int pi = 0; pi < n_gu; ++pi) {
                    const airun_status e = launch_gu(pi, stream);
                    if (e != AIRUN_OK) return e;
                }
                for (int pi = 0; pi < n_dn; ++pi) {
                    const airun_status e = launch_dn(pi, stream);
                    if (e != AIRUN_OK) return e;
                }
            }
            const size_t total = (size_t)n_tokens * n_embd;
            moe_combine_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
                (const float*)ypair, (const float*)weights, (float*)y, n_tokens, top_k, n_embd);
            AIRUN_CHECK_LAUNCH();
        }
    }
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
