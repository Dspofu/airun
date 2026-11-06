// Quantized matrix-vector / matrix-matrix products.
//
// Decode (one token) is memory bound: the weights are read once and every byte
// fetched is used exactly once, so the win comes from reading the packed form
// straight into registers and never materialising a dequantized copy. Each
// warp owns a few output rows and strides over their 32-element groups.
//
// Prefill (many tokens) amortizes each fetched weight group across a register
// tile of tokens x rows, and additionally runs on the integer (dp4a) path
// against quantized activations where the weight type has one. Decode stays on
// the exact f32 path, which is also bit-identical to the CPU reference.
//
// A caution for future tuning, learned the hard way: the tile sizes appear
// both in the template instantiation and in the dispatch's grid arithmetic.
// Sweeping one without the other computes a fraction of the output and looks
// like a huge speedup while producing garbage -- `airun bench` does not check
// outputs. Change the defines, never the instantiations, and hold perplexity
// constant across any kernel experiment.

#include <cmath>
#include "dequant.cuh"
#include "vecdotq.cuh"
#include "mmq.cuh"

#define ROWS_PER_BLOCK 4

// f32 prefill tile (tokens x rows). vals[ROWS][32] keeps ROWS small; 16x4
// measured best (see the q8 defines below for the sweep method).
#define TOKENS_PER_BLOCK 16

// How many output rows one warp carries in the decode kernel.
//
// A lane needs 8 float4 loads to pull its 32-element slice of `x`, against 2
// 16-byte loads for a Q4_K weight group -- so at one row per warp the vector,
// not the weights, dominates the load pipeline, and every warp re-reads all of
// `x` for its own row. Holding several rows per warp loads that slice once and
// spends it against each row's weights. Widening the *block* does not help:
// the redundancy is per warp, and each warp reads the whole vector regardless.
//
// 4 is where this peaked on a Q4_K 9B: 1 row gives 57 tok/s, 4 gives 75, and 8
// falls back to 61 as the extra accumulators cost more occupancy than the saved
// loads buy back.
#define ROWS_PER_WARP 4

// y[0][row] = dot(W[row], x[0]) for ROWS_PER_WARP rows -- the decode path.
//
// W is row-major with `row_bytes` between rows -- rows are individually block
// aligned, which is how GGUF stores them.
//
// Kept separate from the token-tiled kernel below rather than instantiated from
// it at TOKENS == 1: even with the tile bounds constant-folded, the tiled form
// measured ~3% slower here (134 -> 130 tok/s on a 1.7B Q8_0). Decode is the
// path a user waits on token by token, so it gets its own kernel.
//
// Each row still accumulates its groups in the same order as before, so the
// result is bit-identical to the one-row-per-warp version.
// ROWS_PER_WARP rows starting at row0 of one matrix, for one token: the body of
// the decode kernels below.
template <int TYPE>
__device__ __forceinline__ void matvec_rows(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int K,
    const int row0,
    const int n_rows,
    const size_t row_bytes) {

    // Warp-uniform: derived from blockIdx and threadIdx.y only, so the loops
    // below exit together and warp_reduce_sum always sees a full warp.
    const int n_row = min(ROWS_PER_WARP, n_rows - row0);
    const int groups = K / 32;

    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += 32) {
        // Q8_0: every row's bytes are requested before any is used; left to
        // itself the compiler held rows 2 and 3 back until rows 0 and 1 were
        // done.
        q8_0_raw raw[ROWS_PER_WARP];
        if constexpr (TYPE == AIRUN_TYPE_Q8_0) {
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                q8_0_load(w + (size_t)(row0 + min(r, n_row - 1)) * row_bytes, g, raw[r]);
            }
        }
        // 32 floats == 8 float4 loads; x is 16-byte aligned at every group.
        float xv[32];
        const float4* xg = (const float4*)x + g * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 v = xg[i];
            xv[i * 4 + 0] = v.x;
            xv[i * 4 + 1] = v.y;
            xv[i * 4 + 2] = v.z;
            xv[i * 4 + 3] = v.w;
        }

        // No early exit for a short last warp: rows past the end re-read the
        // last row and their sums are dropped below. A `break` here made every
        // row's loads wait for the previous row's arithmetic -- one row in
        // flight per warp, and ncu put nearly all stall samples on the first
        // use of each load. Branch-free, the loads of all rows go out together.
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            float vals[32];
            if constexpr (TYPE == AIRUN_TYPE_Q8_0) {
                q8_0_decode(raw[r], vals);
            } else {
                decode_group<TYPE>(w + (size_t)(row0 + min(r, n_row - 1)) * row_bytes, g, vals);
            }
#pragma unroll
            for (int i = 0; i < 32; ++i) acc[r] = fmaf(vals[i], xv[i], acc[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        if (r >= n_row) break;
        const float s = warp_reduce_sum(acc[r]);
        if (threadIdx.x == 0) {
            y[row0 + r] = s;
        }
    }
}

template <int TYPE>
__global__ __launch_bounds__(32 * ROWS_PER_BLOCK) void matvec_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const size_t row_bytes) {

    const int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * ROWS_PER_WARP;
    if (row0 >= n_rows) return;
    matvec_rows<TYPE>(w, x, y, K, row0, n_rows, row_bytes);
}

// Decode FFN: gate and up rows against the same x, SwiGLU applied on the way
// out -- y = silu(gate x) * (up x) -- in one launch, as llama.cpp's fused
// mul_mat_vec_q does. Each warp carries the same R rows of both matrices, so
// every x slice it loads serves 2R rows. Measured on Qwen3-4B: 178 us for gate
// and up as two launches, against 163 us in llama.cpp's; Q8_0's explicit
// load-then-decode (below) took it from 128 registers to 80 and from 170 to
// 161 us, llama.cpp's 159.
template <int TYPE, int R>
__global__ __launch_bounds__(32 * ROWS_PER_BLOCK) void matvec_glu_kernel(
    const uint8_t* __restrict__ wg,
    const uint8_t* __restrict__ wu,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const size_t row_bytes) {

    const int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * R;
    if (row0 >= n_rows) return;
    const int n_row = min(R, n_rows - row0);
    const int groups = K / 32;

    float ag[R], au[R];
#pragma unroll
    for (int r = 0; r < R; ++r) ag[r] = au[r] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += 32) {
        q8_0_raw rg[R], ru[R];
        if constexpr (TYPE == AIRUN_TYPE_Q8_0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const size_t off = (size_t)(row0 + min(r, n_row - 1)) * row_bytes;
                q8_0_load(wg + off, g, rg[r]);
                q8_0_load(wu + off, g, ru[r]);
            }
        }
        float xv[32];
        const float4* xg = (const float4*)x + g * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 v = xg[i];
            xv[i * 4 + 0] = v.x;
            xv[i * 4 + 1] = v.y;
            xv[i * 4 + 2] = v.z;
            xv[i * 4 + 3] = v.w;
        }
        // Branch-free for the reason in matvec_rows.
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const size_t off = (size_t)(row0 + min(r, n_row - 1)) * row_bytes;
            float vals[32];
            if constexpr (TYPE == AIRUN_TYPE_Q8_0) q8_0_decode(rg[r], vals);
            else decode_group<TYPE>(wg + off, g, vals);
#pragma unroll
            for (int i = 0; i < 32; ++i) ag[r] = fmaf(vals[i], xv[i], ag[r]);
            if constexpr (TYPE == AIRUN_TYPE_Q8_0) q8_0_decode(ru[r], vals);
            else decode_group<TYPE>(wu + off, g, vals);
#pragma unroll
            for (int i = 0; i < 32; ++i) au[r] = fmaf(vals[i], xv[i], au[r]);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= n_row) break;
        const float g = warp_reduce_sum(ag[r]);
        const float u = warp_reduce_sum(au[r]);
        // Same accumulation order as matvec_kernel and the same SwiGLU as
        // swiglu_kernel, so decode stays bit-identical to the unfused path.
        if (threadIdx.x == 0) y[row0 + r] = (g / (1.0f + __expf(-g))) * u;
    }
}
// Three matrices against the same x in one launch: the q, k and v projections
// of a decode step. On their own k and v (1024 rows on Qwen3-4B) are 64-block
// grids that kept DRAM at half its bandwidth -- 15.6 us each where the bytes
// justify 8 -- and q at 71%. As one 6144-row grid they stream like the big FFN
// matrices. Each warp's rows lie in one matrix: the boundaries are multiples of
// ROWS_PER_WARP, which dispatch checks.
template <int TYPE>
__global__ __launch_bounds__(32 * ROWS_PER_BLOCK) void matvec3_kernel(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1, const uint8_t* __restrict__ w2,
    const float* __restrict__ x,
    float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
    const int K, const int n0, const int n1, const int n2,
    const size_t row_bytes) {

    int row = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * ROWS_PER_WARP;
    if (row < n0) {
        matvec_rows<TYPE>(w0, x, y0, K, row, n0, row_bytes);
    } else if ((row -= n0) < n1) {
        matvec_rows<TYPE>(w1, x, y1, K, row, n1, row_bytes);
    } else if ((row -= n1) < n2) {
        matvec_rows<TYPE>(w2, x, y2, K, row, n2, row_bytes);
    }
}
// One block per output row, for decode against matrices too narrow to fill
// the GPU with row parallelism. The 9B's 32-row alpha/beta projections ran on
// a 2-block grid through the warp-per-rows kernel -- 36us each at essentially
// zero utilization, twice per linear layer, which was 15% of a decode step.
// Here 128 threads split the row's groups and a block reduction folds them;
// the reduction order differs from the warp kernel's, so this path is held to
// the selftest bound rather than bit equality, and dispatch only takes it for
// matrices the exact kernel serves badly.
template <int TYPE>
__global__ __launch_bounds__(128) void matvec_tallk_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const size_t row_bytes) {

    const int row = blockIdx.x;
    if (row >= n_rows) return;
    // A few tokens (a --parallel step, an MTP verify): one per blockIdx.y. The
    // row is small enough to come from L2 after the first.
    x += (size_t)blockIdx.y * K;
    y += (size_t)blockIdx.y * n_rows;

    const int groups = K / 32;
    const uint8_t* wr = w + (size_t)row * row_bytes;

    float acc = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        float vals[32];
        decode_group<TYPE>(wr, g, vals);
        const float4* xg = (const float4*)x + g * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 v = xg[i];
            acc = fmaf(vals[i * 4 + 0], v.x, acc);
            acc = fmaf(vals[i * 4 + 1], v.y, acc);
            acc = fmaf(vals[i * 4 + 2], v.z, acc);
            acc = fmaf(vals[i * 4 + 3], v.w, acc);
        }
    }

    __shared__ float red[4];
    acc = block_reduce_sum(acc, red);
    if (threadIdx.x == 0) {
        y[row] = acc;
    }
}

#define PREFILL_ROWS 4
// Tokens up to which a narrow matrix stays on matvec_tallk_kernel.
#define TALLK_TOKENS 16

// Integer-path prefill tile, separate from the f32 one because the register
// pressure differs: the dp4a kernel holds no decoded-float arrays, only the
// accumulators, so it can afford a different shape.
#define Q8_TOKENS_PER_BLOCK 16
#define Q8_PREFILL_ROWS 4

// Same product for TOKENS consecutive tokens and ROWS consecutive rows at once
// -- the prefill path. Every accumulator still walks its groups in the order
// the decode kernel uses, so results stay bit-identical.
template <int TYPE, int TOKENS, int ROWS>
__global__ __launch_bounds__(32 * ROWS_PER_BLOCK) void matvec_tiled_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const int n_tokens,
    const size_t row_bytes) {

    const int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * ROWS;
    if (row0 >= n_rows) return;

    const int tok0 = blockIdx.y * TOKENS;
    // Derived from blockIdx and threadIdx.y alone, so every lane agrees and the
    // exits below stay warp-uniform -- warp_reduce_sum needs a full warp.
    const int n_tok = min(TOKENS, n_tokens - tok0);
    const int n_row = min(ROWS, n_rows - row0);

    const int groups = K / 32;

    float acc[TOKENS][ROWS];
#pragma unroll
    for (int t = 0; t < TOKENS; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += 32) {
        float vals[ROWS][32];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= n_row) break;
            decode_group<TYPE>(w + (size_t)(row0 + r) * row_bytes, g, vals[r]);
        }

#pragma unroll
        for (int t = 0; t < TOKENS; ++t) {
            if (t >= n_tok) break;
            // 32 floats == 8 float4 loads; x is 16-byte aligned at every group.
            const float4* xg = (const float4*)(x + (size_t)(tok0 + t) * K) + g * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 v = xg[i];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (r >= n_row) break;
                    acc[t][r] = fmaf(vals[r][i * 4 + 0], v.x, acc[t][r]);
                    acc[t][r] = fmaf(vals[r][i * 4 + 1], v.y, acc[t][r]);
                    acc[t][r] = fmaf(vals[r][i * 4 + 2], v.z, acc[t][r]);
                    acc[t][r] = fmaf(vals[r][i * 4 + 3], v.w, acc[t][r]);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < TOKENS; ++t) {
        if (t >= n_tok) break;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= n_row) break;
            const float s = warp_reduce_sum(acc[t][r]);
            if (threadIdx.x == 0) {
                y[(size_t)(tok0 + t) * n_rows + row0 + r] = s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor-core path: fp16 tiles in shared memory, fp32 accumulation.
//
// Large-batch prefill is arithmetic bound, and the register-tiled kernels top
// out around 4 MACs per instruction (dp4a). The tensor cores do a 16x16x16
// matrix product per warp instruction, an order of magnitude more. Weights are
// dequantized tile-by-tile into shared fp16 through the same decode_group
// oracle every other kernel uses, so this path works for every weight type,
// and the activation is rounded once to fp16 -- an 11-bit mantissa, which is
// *less* loss than the 8-bit codes the dp4a path feeds on.
//
// Accumulation is fp32. Not bit-identical to the f32 reference (fp16 rounding
// of both operands); validated by the selftest bound and end-to-end
// perplexity, and AIRUN_MATVEC_F32=1 forces the exact path throughout.
// ---------------------------------------------------------------------------

#include <mma.h>

// Output tile per block: WMMA_TR rows x WMMA_TC tokens, 4 warps, each warp
// owning a 16-row slice across all WMMA_TC tokens. K advances in 64-wide
// slabs (two 32-element quant groups), staged through shared memory.
// Output tile per block: TR rows x TC tokens; K advances in TK-wide slabs
// (multiples of 32, i.e. whole quant groups), staged through one shared-memory
// buffer. Double buffering was tried and measured slower (983 -> 908 tok/s on
// the 9B): the second buffer pushes shared memory to 92 KiB, which caps the SM
// at one block and starves the L1 that the staging loads live off. The
// latency-hiding the pipeline was meant to buy, the extra resident warps of
// the leaner layout buy better.
//
// Tile sizes are template parameters because the right size is a function of
// the shared memory the architecture offers: an 87 KiB tile that suits
// Blackwell does not launch on Turing's 64 KiB limit, so dispatch picks a
// configuration per device at runtime.
#define WMMA_PAD 8

template <int TYPE, int TR, int TC, int TK>
__global__ __launch_bounds__(32 * TR / 16) void matvec_wmma_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const int n_tokens,
    const size_t row_bytes) {

    using namespace nvcuda::wmma;

    constexpr int WARPS = TR / 16;
    constexpr int THREADS = 32 * WARPS;
    constexpr int LD = TK + WMMA_PAD;

    // Dynamic so the tiles can exceed the 48 KiB static limit; the dispatch
    // raises the function's shared-memory ceiling before launching.
    extern __shared__ half wmma_smem[];
    half (*ws)[LD] = (half (*)[LD])wmma_smem;
    half (*xs)[LD] = (half (*)[LD])(wmma_smem + TR * LD);

    const int row0 = blockIdx.x * TR;
    const int tok0 = blockIdx.y * TC;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    fragment<accumulator, 16, 16, 16, float> acc[TC / 16];
#pragma unroll
    for (int i = 0; i < TC / 16; ++i) fill_fragment(acc[i], 0.0f);

    // Stages one TK-wide slab of weights and activations. Edge rows and
    // tokens stage zeros so the fragment math needs no bounds checks; the
    // fp16 rounding happens here, once per staged element.
    auto stage = [&](int k0) {
        for (int task = threadIdx.x; task < TR * (TK / 32); task += THREADS) {
            const int r = task / (TK / 32);
            const int gsub = task % (TK / 32);
            const int row = row0 + r;
            half* dst = &ws[r][gsub * 32];
            if (row < n_rows) {
                float vals[32];
                decode_group<TYPE>(w + (size_t)row * row_bytes,
                                   k0 / 32 + gsub, vals);
#pragma unroll
                for (int i = 0; i < 32; ++i) dst[i] = __float2half(vals[i]);
            } else {
#pragma unroll
                for (int i = 0; i < 32; ++i) dst[i] = __float2half(0.0f);
            }
        }
        // Token-major, so the b fragment reads it as a column-major k x token
        // matrix.
        for (int task = threadIdx.x; task < TC * (TK / 32); task += THREADS) {
            const int t = task / (TK / 32);
            const int gsub = task % (TK / 32);
            const int tok = tok0 + t;
            half* dst = &xs[t][gsub * 32];
            if (tok < n_tokens) {
                const float* src = x + (size_t)tok * K + k0 + gsub * 32;
#pragma unroll
                for (int i = 0; i < 32; ++i) dst[i] = __float2half(src[i]);
            } else {
#pragma unroll
                for (int i = 0; i < 32; ++i) dst[i] = __float2half(0.0f);
            }
        }
    };

    for (int k0 = 0; k0 < K; k0 += TK) {
        stage(k0);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            fragment<matrix_a, 16, 16, 16, half, row_major> a;
            load_matrix_sync(a, &ws[warp * 16][kk], LD);
#pragma unroll
            for (int j = 0; j < TC / 16; ++j) {
                fragment<matrix_b, 16, 16, 16, half, col_major> b;
                load_matrix_sync(b, &xs[j * 16][kk], LD);
                mma_sync(acc[j], a, b, acc[j]);
            }
        }
        __syncthreads();
    }

    // y[token][row]: element (m=row, n=token) lives at y[n * n_rows + m],
    // which is a column-major view with leading dimension n_rows. Edge tiles
    // spill into a staging buffer first so nothing writes out of bounds.
    const int rows_here = n_rows - row0;
    const int toks_here = n_tokens - tok0;
    if (rows_here >= TR && toks_here >= TC) {
#pragma unroll
        for (int j = 0; j < TC / 16; ++j) {
            store_matrix_sync(&y[(size_t)(tok0 + j * 16) * n_rows + row0 + warp * 16],
                              acc[j], n_rows, mem_col_major);
        }
    } else {
        // Reuse the weight tile's shared memory as a 16x16 float staging pad
        // per warp; fragments cannot be indexed directly by (m, n).
        float* pad = (float*)&ws[warp * 16][0];
#pragma unroll
        for (int j = 0; j < TC / 16; ++j) {
            store_matrix_sync(pad, acc[j], 16, mem_col_major);
            __syncwarp();
            for (int e = lane; e < 16 * 16; e += 32) {
                const int n = e / 16;
                const int m = e % 16;
                const int row = row0 + warp * 16 + m;
                const int tok = tok0 + j * 16 + n;
                if (row < n_rows && tok < n_tokens) {
                    y[(size_t)tok * n_rows + row] = pad[n * 16 + m];
                }
            }
            __syncwarp();
        }
    }
}

// The two launch configurations dispatch chooses between at runtime:
// Blackwell/Ampere-class devices take the tall double-buffered tile, Turing
// the one that fits its 64 KiB dynamic shared-memory ceiling.
#define WMMA_BIG_TR 256
#define WMMA_BIG_TC 96
#define WMMA_BIG_TK 64
#define WMMA_SMALL_TR 128
#define WMMA_SMALL_TC 64
#define WMMA_SMALL_TK 64
// For few rows and a long K (the hyper-connection down projection, 320 rows x
// 10240; the MoE router, 512 x 2560) the big tile made ~12 blocks for a 70-SM
// card, each walking all of K: the tensor cores sat mostly idle. A small tile
// spreads the same work over ~7x the blocks.
#define WMMA_MID_TR 64
#define WMMA_MID_TC 32
#define WMMA_MID_TK 64

// ---------------------------------------------------------------------------
// Integer (dp4a) path: quantized activations against quantized weights.
// See vecdotq.cuh for the trade being made and the validation story.
// ---------------------------------------------------------------------------

// f32 -> block_q8a, one thread per 32-element group. Memory bound and tiny
// next to any matvec that consumes it.
// One warp per group, a lane per element: loads and stores are coalesced and
// the max and sum are shuffles. A thread per group (the previous form) had the
// lanes of a warp 128 bytes apart and took 26 ms of a 512-token Qwen3-4B
// prefill for ~1.3 GB of traffic. Same arithmetic as before, so the CPU mirror
// in selftest is unchanged.
__global__ void quantize_q8a_kernel(
    const float* __restrict__ x,
    block_q8a* __restrict__ out,
    const int64_t n_groups) {

    const int64_t g = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (g >= n_groups) return;   // g is warp-uniform

    const float v = x[g * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    // Round-to-nearest-even, mirrored by round_ties_even in the CPU check.
    const int qi = __float2int_rn(v * id);
    int sum = qi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);

    out[g].qs[lane] = (int8_t)qi;
    if (lane == 0) out[g].ds = __floats2half2_rn(d, d * (float)sum);
}

// Decode: same warp layout as matvec_kernel, integer dots instead of FMAs.
template <int TYPE>
__global__ __launch_bounds__(32 * ROWS_PER_BLOCK) void matvec_q8_kernel(
    const uint8_t* __restrict__ w,
    const block_q8a* __restrict__ xq,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const size_t row_bytes) {

    const int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * ROWS_PER_WARP;
    if (row0 >= n_rows) return;

    const int n_row = min(ROWS_PER_WARP, n_rows - row0);
    const int groups = K / 32;

    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += 32) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if (r >= n_row) break;
            acc[r] += q8dot<TYPE>::dot(w + (size_t)(row0 + r) * row_bytes, g, xq);
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        if (r >= n_row) break;
        const float s = warp_reduce_sum(acc[r]);
        if (threadIdx.x == 0) {
            y[row0 + r] = s;
        }
    }
}

// Decode for the 4-bit types: a block of 128 threads covers the whole K of R
// rows, so every group of a short row is requested at once. With a warp per
// four rows (matvec_q8_kernel) a 5120-wide row is five groups per lane, and
// the loads of one iteration were too few bytes in flight to keep DRAM busy:
// the 27B NVFP4 decode ran its matvecs at 73% of bandwidth.
#define WIDE_THREADS 128
template <int TYPE, int R>
__global__ __launch_bounds__(WIDE_THREADS) void matvec_q8_wide_kernel(
    const uint8_t* __restrict__ w,
    const block_q8a* __restrict__ xq,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const size_t row_bytes) {

    const int row0 = blockIdx.x * R;
    const int groups = K / 32;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += WIDE_THREADS) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[r] += q8dot<TYPE>::dot(w + (size_t)min(row0 + r, n_rows - 1) * row_bytes, g, xq);
        }
    }

    __shared__ float red[R][WIDE_THREADS / 32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float s = warp_reduce_sum(acc[r]);
        if (lane == 0) red[r][warp] = s;
    }
    __syncthreads();
    if (threadIdx.x < R && row0 + threadIdx.x < n_rows) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < WIDE_THREADS / 32; ++i) s += red[threadIdx.x][i];
        y[row0 + threadIdx.x] = s;
    }
}

// matvec_q8_wide_kernel for 2..NT tokens (the speculative verify): each group
// is decoded once (mmq_decode) and spent against every token's activation.
template <int TYPE, int R, int NT>
__global__ __launch_bounds__(WIDE_THREADS) void matvec_q8_wide_multi_kernel(
    const uint8_t* __restrict__ w,
    const block_q8a* __restrict__ xq,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const int n_tokens,
    const size_t row_bytes) {

    using D = mmq_decode<TYPE>;
    const int row0 = blockIdx.x * R;
    const int groups = K / 32;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += WIDE_THREADS) {
        typename D::Raw raw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) D::load(w + (size_t)min(row0 + r, n_rows - 1) * row_bytes, g, raw[r]);
        uint32_t q[R][8];
        float s0[R], s1[R], m[R];
#pragma unroll
        for (int r = 0; r < R; ++r) D::decode(raw[r], g, q[r], s0[r], s1[r], m[r]);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= n_tokens) continue;
            const block_q8a* a = xq + (size_t)t * groups + g;
            int aw[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) aw[i] = ((const int*)a->qs)[i];
            const float2 ds = __half22float2(a->ds);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if constexpr (D::SPLIT) {
                    int d0 = 0, d1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) d0 = __dp4a((int)q[r][i], aw[i], d0);
#pragma unroll
                    for (int i = 4; i < 8; ++i) d1 = __dp4a((int)q[r][i], aw[i], d1);
                    acc[r][t] += (s0[r] * (float)d0 + s1[r] * (float)d1) * ds.x;
                } else {
                    int d = 0;
#pragma unroll
                    for (int i = 0; i < 8; ++i) d = __dp4a((int)q[r][i], aw[i], d);
                    float v = s0[r] * ds.x * (float)d;
                    if constexpr (D::HAS_MIN) v -= m[r] * ds.y;
                    acc[r][t] += v;
                }
            }
        }
    }

    __shared__ float red[R][NT][WIDE_THREADS / 32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float s = warp_reduce_sum(acc[r][t]);
            if (lane == 0) red[r][t][warp] = s;
        }
    __syncthreads();
    if (threadIdx.x < R * NT) {
        const int r = threadIdx.x / NT, t = threadIdx.x % NT;
        if (row0 + r < n_rows && t < n_tokens) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < WIDE_THREADS / 32; ++i) s += red[r][t][i];
            y[(size_t)t * n_rows + row0 + r] = s;
        }
    }
}

// A few tokens (speculative verify, 2..8): each weight group is loaded and
// decoded to its integer codes once -- mmq_decode, the same codes and scales
// the tensor-core kernel uses -- and spent against every token's activation.
// The tiled kernel below re-decoded the group per token, and for NVFP4, whose
// decode is all byte permutes, a 4-token verify cost twice a decode step.
template <int TYPE, int NT>
__global__ __launch_bounds__(32 * ROWS_PER_BLOCK) void matvec_q8_multi_kernel(
    const uint8_t* __restrict__ w,
    const block_q8a* __restrict__ xq,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const int n_tokens,
    const size_t row_bytes) {

    using D = mmq_decode<TYPE>;
    const int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * ROWS_PER_WARP;
    if (row0 >= n_rows) return;
    const int n_row = min(ROWS_PER_WARP, n_rows - row0);
    const int groups = K / 32;

    float acc[ROWS_PER_WARP][NT];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += 32) {
        // All rows' bytes requested before any is decoded (see q8_0_load),
        // then every row decoded before the tokens: each token's activation
        // is loaded once and spent against all rows.
        typename D::Raw raw[ROWS_PER_WARP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            D::load(w + (size_t)(row0 + min(r, n_row - 1)) * row_bytes, g, raw[r]);
        }
        uint32_t q[ROWS_PER_WARP][8];
        float s0[ROWS_PER_WARP], s1[ROWS_PER_WARP], m[ROWS_PER_WARP];
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) D::decode(raw[r], g, q[r], s0[r], s1[r], m[r]);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= n_tokens) continue;
            const block_q8a* a = xq + (size_t)t * groups + g;
            // 36-byte blocks: 4-byte alignment only.
            int aw[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) aw[i] = ((const int*)a->qs)[i];
            const float2 ds = __half22float2(a->ds);
#pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                if constexpr (D::SPLIT) {
                    int d0 = 0, d1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) d0 = __dp4a((int)q[r][i], aw[i], d0);
#pragma unroll
                    for (int i = 4; i < 8; ++i) d1 = __dp4a((int)q[r][i], aw[i], d1);
                    acc[r][t] += (s0[r] * (float)d0 + s1[r] * (float)d1) * ds.x;
                } else {
                    int d = 0;
#pragma unroll
                    for (int i = 0; i < 8; ++i) d = __dp4a((int)q[r][i], aw[i], d);
                    float v = s0[r] * ds.x * (float)d;
                    if constexpr (D::HAS_MIN) v -= m[r] * ds.y;
                    acc[r][t] += v;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        if (r >= n_row) break;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= n_tokens) break;
            const float s = warp_reduce_sum(acc[r][t]);
            if (threadIdx.x == 0) y[(size_t)t * n_rows + row0 + r] = s;
        }
    }
}

// Prefill: token x row tile as in matvec_tiled_kernel. The integer dot both
// shrinks the activation reads 4x and replaces 32 FMAs with 8 dp4a per group,
// which is what an instruction-bound kernel wants.
// No min-blocks occupancy hint here: it was tried (6 resident blocks, capping
// registers at 85) against the profiled 33% theoretical occupancy, and the
// spills it forced cost more than the extra warps bought -- prefill dropped
// about 9% on both test models. The tile size is the occupancy lever instead.
template <int TYPE, int TOKENS, int ROWS>
__global__ __launch_bounds__(32 * ROWS_PER_BLOCK) void matvec_q8_tiled_kernel(
    const uint8_t* __restrict__ w,
    const block_q8a* __restrict__ xq,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const int n_tokens,
    const size_t row_bytes) {

    const int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.y) * ROWS;
    if (row0 >= n_rows) return;

    const int tok0 = blockIdx.y * TOKENS;
    const int n_tok = min(TOKENS, n_tokens - tok0);
    const int n_row = min(ROWS, n_rows - row0);
    const int groups = K / 32;

    float acc[TOKENS][ROWS];
#pragma unroll
    for (int t = 0; t < TOKENS; ++t)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += 32) {
#pragma unroll
        for (int t = 0; t < TOKENS; ++t) {
            if (t >= n_tok) break;
            const block_q8a* at = xq + (size_t)(tok0 + t) * groups;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r >= n_row) break;
                acc[t][r] += q8dot<TYPE>::dot(
                    w + (size_t)(row0 + r) * row_bytes, g, at);
            }
        }
    }

#pragma unroll
    for (int t = 0; t < TOKENS; ++t) {
        if (t >= n_tok) break;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= n_row) break;
            const float s = warp_reduce_sum(acc[t][r]);
            if (threadIdx.x == 0) {
                y[(size_t)(tok0 + t) * n_rows + row0 + r] = s;
            }
        }
    }
}

// Expands a packed tensor to f32. Used by the converter and by the numerical
// self-test that checks the GPU decoders against the CPU reference.
template <int TYPE>
__global__ void dequantize_kernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    const int64_t n_groups) {

    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;

    float vals[32];
    decode_group<TYPE>(src, (int)g, vals);

    float* out = dst + g * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = vals[i];
}

// Dispatch: one case per supported type id, each instantiating the template so
// the decoder inlines with no runtime branch in the inner loop. `LAUNCH` is
// defined at each call site and consumed by `AIRUN_CASE`.
#define AIRUN_CASE(TYPE) case TYPE: LAUNCH(TYPE); break;

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

// BF16 / F16 weights against 1..8 tokens: each lane loads 16 bytes (8 weights)
// of a row per step, so a warp reads 512 contiguous bytes, and every weight
// serves all M tokens. The group-at-a-time decode kernel had each lane fetch a
// 64-byte group -- strided across the warp -- and re-read W once per token in
// the tiled path: the 248k x 5120 BF16 output head of a Qwen3.8-27B took 6 ms
// for one token (420 GB/s on an RTX 5070 Ti) and 13 ms for four.
#define HALF_MV_R 4
template <int TYPE, int M>
__global__ __launch_bounds__(128) void matvec_half_kernel(
    const uint8_t* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int K,
    const int n_rows,
    const size_t row_bytes) {

    constexpr int R = HALF_MV_R;
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * R;
    if (row0 >= n_rows) return;
    const int n_row = min(R, n_rows - row0);

    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

    for (int k0 = threadIdx.x * 8; k0 < K; k0 += 256) {
        // Every row's bytes requested before any arithmetic (see matvec_rows).
        uint4 q[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q[r] = *(const uint4*)(w + (size_t)(row0 + min(r, n_row - 1)) * row_bytes + (size_t)k0 * 2);
        }
        float xv[M][8];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float4* xp = (const float4*)(x + (size_t)m * K + k0);
            const float4 a = xp[0], b = xp[1];
            xv[m][0] = a.x; xv[m][1] = a.y; xv[m][2] = a.z; xv[m][3] = a.w;
            xv[m][4] = b.x; xv[m][5] = b.y; xv[m][6] = b.z; xv[m][7] = b.w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t u[4] = { q[r].x, q[r].y, q[r].z, q[r].w };
            float wv[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if constexpr (TYPE == AIRUN_TYPE_BF16) {
                    wv[2 * j + 0] = bf16_to_f32((uint16_t)(u[j] & 0xFFFF));
                    wv[2 * j + 1] = bf16_to_f32((uint16_t)(u[j] >> 16));
                } else {
                    wv[2 * j + 0] = __half2float(__ushort_as_half((unsigned short)(u[j] & 0xFFFF)));
                    wv[2 * j + 1] = __half2float(__ushort_as_half((unsigned short)(u[j] >> 16)));
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][m] = fmaf(wv[i], xv[m][i], acc[r][m]);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r >= n_row) break;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float s = warp_reduce_sum(acc[r][m]);
            if (threadIdx.x == 0) y[(size_t)m * n_rows + row0 + r] = s;
        }
    }
}

// AIRUN_NO_TALLK_TOKENS=1 sends narrow matrices with 2+ tokens back to the
// tiled kernel.
static bool tallk_tokens_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_TALLK_TOKENS");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// AIRUN_NO_HALF_MATVEC=1 keeps BF16/F16 decode on the group kernels.
static bool half_matvec_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_HALF_MATVEC");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// AIRUN_NO_WMMA=1 keeps large batches on the register-tiled kernels, so the
// tensor-core path can be measured against them without a rebuild.
static bool airun_force_no_wmma() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_WMMA");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// AIRUN_NO_Q8_WIDE=1 keeps NVFP4 decode on the warp-per-four-rows kernel.
static bool q8_wide_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_Q8_WIDE");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// AIRUN_NO_Q8_MULTI=1 sends 2..8-token batches back to the tiled kernel.
static bool q8_multi_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_Q8_MULTI");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// Smallest batch the integer tensor-core kernel takes; below it the dp4a tiles
// win because an MMQ tile of 64 tokens would be mostly padding.
#define MMQ_MIN_TOKENS 32

// 1 when the current device runs matvec_mmq_kernel: m16n8k32 integer MMA is
// sm_80+. AIRUN_NO_MMQ=1 turns it off to measure it against the fp16 path.
extern "C" int32_t airun_mmq_supported() {
    static int disabled = -1;
    if (disabled < 0) {
        const char* v = getenv("AIRUN_NO_MMQ");
        disabled = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    if (disabled) return 0;
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 0;
    return major >= 8 ? 1 : 0;
}

// AIRUN_NO_MMQ_ASYNC=1 keeps every type on the synchronous MMQ kernel.
static bool mmq_async_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_MMQ_ASYNC");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// The cp.async kernel needs whole, aligned chunks; anything else (Q6_K's
// 210-byte blocks, a K that is not a multiple of 128) takes the synchronous one.
// Split-K: how many K slices to cut the grid into, from how well the plain grid
// fills the GPU. `per_sm` blocks fit an SM at once, so the grid runs in waves of
// SMs * per_sm; a 16-block grid on 28 SMs leaves 12 idle, and 40 blocks run as
// a full wave plus a 12-block one. Splitting multiplies the blocks; each extra
// split costs a partial write and a read in the reduction, priced here at 10%.
// AIRUN_NO_MMQ_SPLIT=1 turns it off.
// A 128-token Qwen3-4B prefill measured 46 ms of MMQ without this, llama.cpp's
// stream-k 30 ms.
// SMs of the current device; per device, as the two cards of a split differ.
static int mmq_sm_count() {
    static int sms[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    if (sms[dev] == 0) {
        int n = 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        sms[dev] = n > 0 ? n : 1;
    }
    return sms[dev];
}

static int mmq_splits(int blocks, int per_sm, int chunks) {
    const int sms = mmq_sm_count();
    static int disabled = -1;
    if (disabled < 0) {
        const char* v = getenv("AIRUN_NO_MMQ_SPLIT");
        disabled = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    const double slots = (double)sms * per_sm;
    const double waves1 = blocks / slots;
    // Splitting every grid measured slower: a split halves each block's K
    // loop (more pipeline fill per MAC) and adds the reduction. Only grids
    // that fill the GPU badly are worth it.
    // 0.80: the 64-block k/v grids of a Qwen3-4B 512-token prefill (76%) do
    // split; at 0.75 they did not and ran 34% behind llama.cpp's stream-k.
    if (disabled || waves1 / std::ceil(waves1) >= 0.80) return 1;
    int best = 1;
    double best_score = 0.0;
    for (int s = 1; s <= 4 && chunks / s >= 2; ++s) {
        const double waves = blocks * s / slots;
        const double eff = waves / std::ceil(waves) - 0.10 * (s - 1);
        if (eff > best_score + 1e-9) { best_score = eff; best = s; }
    }
    return best;
}

// Per-device scratch for the split partials, grown on demand. Prefill is never
// captured into a CUDA graph (decode is), so allocating here is allowed.
static float* mmq_scratch(size_t floats) {
    static float* buf[16] = {};
    static size_t cap[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return nullptr;
    if (cap[dev] < floats) {
        if (buf[dev]) cudaFree(buf[dev]);
        buf[dev] = nullptr;
        cap[dev] = 0;
        if (cudaMalloc(&buf[dev], floats * sizeof(float)) != cudaSuccess) return nullptr;
        cap[dev] = floats;
    }
    return buf[dev];
}

// AIRUN_NO_MMQ_STREAMK=1 leaves the asynchronous kernel on whole-tile grids
// (with split-K where they fill the GPU badly).
static bool mmq_streamk_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_MMQ_STREAMK");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// Plans a stream-K launch of `kernel` over `tiles` tiles (see MmqStreamK):
// whole waves as whole tiles, the leftover tiles shared by one wave of blocks.
// False when the tiles already make whole waves, or the launch cannot use it:
// a stream being captured into a graph would freeze the flag epoch into it.
static bool mmq_streamk_plan(const void* kernel, size_t smem, int tiles, int tok_tiles,
                             size_t slot_floats, cudaStream_t stream, MmqStreamK& sk) {
    if (mmq_streamk_disabled()) return false;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        return false;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return false;
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 256, smem) != cudaSuccess ||
        per_sm < 1) {
        return false;
    }
    // At most one resident wave of stream-K blocks: a head spins until the
    // tails after it have run, so they must be able to be on the GPU at once.
    const int wave = mmq_sm_count() * per_sm;
    const int rest = tiles % wave;
    if (rest == 0) return false;

    static float* slots[16] = {};
    static size_t slots_cap[16] = {};
    static unsigned* flags[16] = {};
    static int flags_cap[16] = {};
    static unsigned epoch[16] = {};
    const size_t need = (size_t)wave * slot_floats;
    if (slots_cap[dev] < need) {
        if (slots[dev]) cudaFree(slots[dev]);
        slots[dev] = nullptr;
        slots_cap[dev] = 0;
        if (cudaMalloc(&slots[dev], need * sizeof(float)) != cudaSuccess) return false;
        slots_cap[dev] = need;
    }
    if (flags_cap[dev] < wave) {
        if (flags[dev]) cudaFree(flags[dev]);
        flags[dev] = nullptr;
        flags_cap[dev] = 0;
        if (cudaMalloc(&flags[dev], wave * sizeof(unsigned)) != cudaSuccess) return false;
        // Synchronous, so no launch sees the buffer before it is zeroed.
        if (cudaMemset(flags[dev], 0, wave * sizeof(unsigned)) != cudaSuccess) return false;
        flags_cap[dev] = wave;
    }
    if (++epoch[dev] == 0) epoch[dev] = 1;   // 0 is what a fresh flag holds
    sk.tiles = tiles;
    sk.dp = tiles - rest;
    sk.sk = wave;
    sk.tok_tiles = tok_tiles;
    sk.epoch = epoch[dev];
    sk.slots = slots[dev];
    sk.flags = flags[dev];
    return true;
}

// The cp.async kernel needs whole, aligned chunks; anything else (Q6_K's
// 210-byte blocks, a K that is not a multiple of 128) takes the synchronous one.
template <int TYPE, int TC>
static void launch_mmq_tc(const void* w, const void* xq, void* y, int k, int n_rows,
                          int n_tokens, uint64_t row_bytes, cudaStream_t stream) {
    bool async = false;
    if constexpr (mmq_async<TYPE>::OK) {
        async = row_bytes % 16 == 0 && (uintptr_t)w % 16 == 0 && (uintptr_t)xq % 16 == 0 &&
                (k / 32) % mmq_async<TYPE>::GROUP_MULTIPLE == 0 && !mmq_async_disabled();
    }
    const int tok_tiles = (n_tokens + TC - 1) / TC;
    const int tiles = tok_tiles * ((n_rows + MMQ_TR - 1) / MMQ_TR);
    const int chunks = (k / 32 + MMQ_KG - 1) / MMQ_KG;
    if constexpr (mmq_async<TYPE>::OK) {
        if (async) {
            constexpr size_t smem = mmq_async_smem<TYPE, TC>();
            const void* kernel = (const void*)matvec_mmq_async_kernel<TYPE, TC>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            MmqStreamK sk = {};
            if (mmq_streamk_plan(kernel, smem, tiles, tok_tiles, (size_t)MMQ_TR * TC, stream, sk)) {
                matvec_mmq_async_kernel<TYPE, TC><<<sk.dp + sk.sk, 256, smem, stream>>>(
                    (const uint8_t*)w, nullptr, (const block_q8a*)xq, (float*)y, nullptr, k,
                    n_rows, n_tokens, (size_t)row_bytes, sk);
                return;
            }
        }
    }
    int splits = mmq_splits(tiles, async ? 1 : 2, chunks);
    float* partial = nullptr;
    if (splits > 1) {
        partial = mmq_scratch((size_t)splits * n_rows * n_tokens);
        if (!partial) splits = 1;
    }
    const dim3 grid(tok_tiles, (n_rows + MMQ_TR - 1) / MMQ_TR, splits);
    if constexpr (mmq_async<TYPE>::OK) {
        if (async) {
            constexpr size_t smem = mmq_async_smem<TYPE, TC>();
            matvec_mmq_async_kernel<TYPE, TC><<<grid, 256, smem, stream>>>(
                (const uint8_t*)w, nullptr, (const block_q8a*)xq, (float*)y, partial, k, n_rows,
                n_tokens, (size_t)row_bytes, MmqStreamK{});
        }
    }
    if (!async) {
        constexpr int smem = mmq_sync_smem<TC>();
        cudaFuncSetAttribute(matvec_mmq_kernel<TYPE, TC>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        matvec_mmq_kernel<TYPE, TC><<<grid, 256, smem, stream>>>(
            (const uint8_t*)w, (const block_q8a*)xq, (float*)y, partial, k, n_rows, n_tokens,
            (size_t)row_bytes);
    }
    if (splits > 1) {
        const int64_t n = (int64_t)n_rows * n_tokens;
        mmq_reduce_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>((float*)y, partial, n,
                                                                             splits);
    }
}
// 128-token tiles from 128 tokens up: a wider tile does twice the MMAs per
// staged weight and per barrier; below that it is mostly padding. AIRUN_MMQ_TC=64 or =128 forces one.
static int mmq_forced_tc() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_MMQ_TC");
        cached = v ? atoi(v) : 0;
    }
    return cached;
}
template <int TYPE>
static void launch_mmq(const void* w, const void* xq, void* y, int k, int n_rows,
                       int n_tokens, uint64_t row_bytes, cudaStream_t stream) {
    // 99 KiB is the per-block opt-in on sm_86/89/120. Q5_K's 176-byte
    // superblocks make the 128-token asynchronous tile 106 KiB: it stays at 64.
    constexpr bool wide_fits = !mmq_async<TYPE>::OK || mmq_async_smem<TYPE, 128>() <= 99 * 1024;
    const int forced = mmq_forced_tc();
    // Measured on Qwen3-4B (best of 3): at 128 tokens the wide tile wins 13%
    // even as a single token tile (split-K fills the grid), ties at 192 and
    // loses 25% at 64.
    const bool wide = forced ? forced == 128 : n_tokens >= 128;
    if (wide_fits && wide) {
        launch_mmq_tc<TYPE, 128>(w, xq, y, k, n_rows, n_tokens, row_bytes, stream);
    } else {
        launch_mmq_tc<TYPE, 64>(w, xq, y, k, n_rows, n_tokens, row_bytes, stream);
    }
}
#ifndef AIRUN_ERR_BAD_TYPE
#define AIRUN_ERR_BAD_TYPE 1   // cudaErrorInvalidValue
#endif

// Gate and up projections plus SwiGLU in one launch (see matvec_mmq_async_kernel):
// y = silu(w_gate x) * (w_up x). Only where the asynchronous MMQ kernel runs;
// airun_mmq_glu_supported says whether this call qualifies, and the caller
// keeps the separate path otherwise.
template <int TYPE, int TC>
static void launch_mmq_glu_tc(const void* wg, const void* wu, const void* xq, void* y, int k,
                              int n_rows, int n_tokens, uint64_t row_bytes, cudaStream_t stream) {
    const int tok_tiles = (n_tokens + TC - 1) / TC;
    const int row_tiles = (n_rows + MMQ_TR / 2 - 1) / (MMQ_TR / 2);
    constexpr size_t smem = mmq_async_smem<TYPE, TC>();
    const void* kernel = (const void*)matvec_mmq_async_kernel<TYPE, TC, true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    // SwiGLU is not linear, so gate/up cannot split K into partial outputs; the
    // stream-K slots carry the raw gate and up sums, and the epilogue runs once
    // the head block has all of them.
    MmqStreamK sk = {};
    const bool streamk = mmq_streamk_plan(kernel, smem, tok_tiles * row_tiles, tok_tiles,
                                          (size_t)MMQ_TR * TC, stream, sk);
    const dim3 grid = streamk ? dim3(sk.dp + sk.sk) : dim3(tok_tiles, row_tiles);
    matvec_mmq_async_kernel<TYPE, TC, true><<<grid, 256, smem, stream>>>(
        (const uint8_t*)wg, (const uint8_t*)wu, (const block_q8a*)xq, (float*)y, nullptr, k,
        n_rows, n_tokens, (size_t)row_bytes, sk);
}

template <int TYPE>
static bool mmq_glu_ok(const void* wg, const void* wu, const void* xq, int k, uint64_t row_bytes,
                       int n_tokens) {
    if constexpr (!mmq_async<TYPE>::OK) {
        return false;
    } else {
        return n_tokens >= MMQ_MIN_TOKENS && airun_mmq_supported() && !mmq_async_disabled() &&
               row_bytes % 16 == 0 && (uintptr_t)wg % 16 == 0 && (uintptr_t)wu % 16 == 0 &&
               (uintptr_t)xq % 16 == 0 && (k / 32) % mmq_async<TYPE>::GROUP_MULTIPLE == 0;
    }
}

// AIRUN_NO_MMQ_GLU=1 keeps gate, up and SwiGLU as three launches.
static bool mmq_glu_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* v = getenv("AIRUN_NO_MMQ_GLU");
        cached = (v && v[0] && v[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

extern "C" int32_t airun_mmq_glu_supported(int32_t type_id, const void* wg, const void* wu,
                                           const void* xq, int32_t k, uint64_t row_bytes,
                                           int32_t n_tokens) {
    if (mmq_glu_disabled() || k % 32 != 0) return 0;
    switch (type_id) {
#define GLU_CASE(TYPE) case TYPE: return mmq_glu_ok<TYPE>(wg, wu, xq, k, row_bytes, n_tokens) ? 1 : 0;
        AIRUN_FOR_EACH_Q8_TYPE(GLU_CASE)
#undef GLU_CASE
        default: return 0;
    }
}

extern "C" airun_status airun_matvec_q8_glu(
    int32_t type_id, const void* wg, const void* wu, const void* xq, void* y,
    int32_t k, int32_t n_rows, int32_t n_tokens, uint64_t row_bytes, cudaStream_t stream) {
    if (!airun_mmq_glu_supported(type_id, wg, wu, xq, k, row_bytes, n_tokens)) return AIRUN_ERR_BAD_TYPE;
    const int forced = mmq_forced_tc();
    const bool wide = forced ? forced == 128 : n_tokens >= 128;
    switch (type_id) {
#define GLU_LAUNCH(TYPE)                                                                         \
    case TYPE: {                                                                                 \
        constexpr bool fits = mmq_async<TYPE>::OK && mmq_async_smem<TYPE, 128>() <= 99 * 1024;  \
        if (fits && wide) launch_mmq_glu_tc<TYPE, 128>(wg, wu, xq, y, k, n_rows, n_tokens, row_bytes, stream); \
        else launch_mmq_glu_tc<TYPE, 64>(wg, wu, xq, y, k, n_rows, n_tokens, row_bytes, stream); \
        break;                                                                                   \
    }
        GLU_LAUNCH(AIRUN_TYPE_Q8_0)
        GLU_LAUNCH(AIRUN_TYPE_Q4_K)
        GLU_LAUNCH(AIRUN_TYPE_Q5_K)
        GLU_LAUNCH(AIRUN_TYPE_NVFP4)
#undef GLU_LAUNCH
        default: return AIRUN_ERR_BAD_TYPE;
    }
    AIRUN_CHECK_LAUNCH();
}
extern "C" {
// cudaErrorInvalidValue, returned for a type this build has no kernel for;
// defined above, where the gate/up entry point first needs it.

airun_status airun_matvec(
    int32_t type_id,
    const void* w,
    const void* x,
    void* y,
    int32_t k,
    int32_t n_rows,
    int32_t n_tokens,
    uint64_t row_bytes,
    cudaStream_t stream) {

    if (k % 32 != 0) return AIRUN_ERR_BAD_TYPE;
    if (n_rows <= 0 || n_tokens <= 0) return AIRUN_OK;

    const dim3 block(32, ROWS_PER_BLOCK);
    const int grid_rows = (n_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    if ((n_tokens == 1 || (n_tokens <= TALLK_TOKENS && !tallk_tokens_disabled())) && n_rows < 512) {
        // Narrow decode matrices: one block per row (see matvec_tallk_kernel),
        // and per token. With 4 tokens the tiled kernel below ran 3 blocks:
        // 55us for a 48-row delta-net gate, against 5us for one token.
        const dim3 tgrid(n_rows, n_tokens);
        const dim3 tblock(128);

#define LAUNCH(TYPE)                                                           matvec_tallk_kernel<TYPE><<<tgrid, tblock, 0, stream>>>(                       (const uint8_t*)w, (const float*)x, (float*)y,                             k, n_rows, (size_t)row_bytes);

        switch (type_id) {
            AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    } else if ((type_id == AIRUN_TYPE_BF16 || type_id == AIRUN_TYPE_F16) && n_tokens <= 8
               && k % 256 == 0 && !half_matvec_disabled()) {
        const dim3 hblock(32, 4);
        const dim3 hgrid((n_rows + 4 * HALF_MV_R - 1) / (4 * HALF_MV_R));
#define HALF_LAUNCH(TYPE, M) matvec_half_kernel<TYPE, M><<<hgrid, hblock, 0, stream>>>( \
            (const uint8_t*)w, (const float*)x, (float*)y, k, n_rows, (size_t)row_bytes)
#define HALF_CASES(TYPE) switch (n_tokens) { \
            case 1: HALF_LAUNCH(TYPE, 1); break; case 2: HALF_LAUNCH(TYPE, 2); break; \
            case 3: HALF_LAUNCH(TYPE, 3); break; case 4: HALF_LAUNCH(TYPE, 4); break; \
            case 5: HALF_LAUNCH(TYPE, 5); break; case 6: HALF_LAUNCH(TYPE, 6); break; \
            case 7: HALF_LAUNCH(TYPE, 7); break; default: HALF_LAUNCH(TYPE, 8); break; }
        if (type_id == AIRUN_TYPE_BF16) { HALF_CASES(AIRUN_TYPE_BF16) } else { HALF_CASES(AIRUN_TYPE_F16) }
#undef HALF_CASES
#undef HALF_LAUNCH
    } else if (n_tokens == 1) {
        // Each warp covers ROWS_PER_WARP rows in this path.
        const int warps = (n_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

#define LAUNCH(TYPE)                                                       \
    matvec_kernel<TYPE><<<grid, block, 0, stream>>>(                       \
        (const uint8_t*)w, (const float*)x, (float*)y,                     \
        k, n_rows, (size_t)row_bytes);

        switch (type_id) {
            AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    } else if (n_tokens >= 64 && k % 64 == 0 && !airun_force_no_wmma()) {
        // Large batches go to the tensor cores. K is a multiple of 64 for
        // every projection in the supported models; anything else falls
        // through to the register-tiled kernel below. The tile is chosen by
        // what the device's shared memory allows.
        int dev = 0, cc_major = 8;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);

        int n_sm = 1;
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
        const long big_blocks = (long)((n_rows + WMMA_BIG_TR - 1) / WMMA_BIG_TR)
                              * ((n_tokens + WMMA_BIG_TC - 1) / WMMA_BIG_TC);
        if (cc_major >= 8 && big_blocks < 2L * n_sm) {
            const dim3 wblock(32 * WMMA_MID_TR / 16);
            const dim3 wgrid((n_rows + WMMA_MID_TR - 1) / WMMA_MID_TR,
                             (n_tokens + WMMA_MID_TC - 1) / WMMA_MID_TC);
            const size_t wsmem = (size_t)(WMMA_MID_TR + WMMA_MID_TC)
                               * (WMMA_MID_TK + WMMA_PAD) * sizeof(half);
#define LAUNCH(TYPE)                                                                          matvec_wmma_kernel<TYPE, WMMA_MID_TR, WMMA_MID_TC, WMMA_MID_TK>                       <<<wgrid, wblock, wsmem, stream>>>(                                               (const uint8_t*)w, (const float*)x, (float*)y,                                    k, n_rows, n_tokens, (size_t)row_bytes);
            switch (type_id) {
                AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
                default: return AIRUN_ERR_BAD_TYPE;
            }
#undef LAUNCH
        } else if (cc_major >= 8) {
            const dim3 wblock(32 * WMMA_BIG_TR / 16);
            const dim3 wgrid((n_rows + WMMA_BIG_TR - 1) / WMMA_BIG_TR,
                             (n_tokens + WMMA_BIG_TC - 1) / WMMA_BIG_TC);
            const size_t wsmem = (size_t)(WMMA_BIG_TR + WMMA_BIG_TC)
                               * (WMMA_BIG_TK + WMMA_PAD) * sizeof(half);

#define LAUNCH(TYPE)                                                              cudaFuncSetAttribute(                                                             matvec_wmma_kernel<TYPE, WMMA_BIG_TR, WMMA_BIG_TC, WMMA_BIG_TK>,              cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsmem);                 matvec_wmma_kernel<TYPE, WMMA_BIG_TR, WMMA_BIG_TC, WMMA_BIG_TK>                   <<<wgrid, wblock, wsmem, stream>>>(                                           (const uint8_t*)w, (const float*)x, (float*)y,                                k, n_rows, n_tokens, (size_t)row_bytes);

            switch (type_id) {
                AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
                default: return AIRUN_ERR_BAD_TYPE;
            }
#undef LAUNCH
        } else {
            const dim3 wblock(32 * WMMA_SMALL_TR / 16);
            const dim3 wgrid((n_rows + WMMA_SMALL_TR - 1) / WMMA_SMALL_TR,
                             (n_tokens + WMMA_SMALL_TC - 1) / WMMA_SMALL_TC);
            const size_t wsmem = (size_t)(WMMA_SMALL_TR + WMMA_SMALL_TC)
                               * (WMMA_SMALL_TK + WMMA_PAD) * sizeof(half);

#define LAUNCH(TYPE)                                                              cudaFuncSetAttribute(                                                             matvec_wmma_kernel<TYPE, WMMA_SMALL_TR, WMMA_SMALL_TC,                                           WMMA_SMALL_TK>,                                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsmem);                 matvec_wmma_kernel<TYPE, WMMA_SMALL_TR, WMMA_SMALL_TC, WMMA_SMALL_TK>             <<<wgrid, wblock, wsmem, stream>>>(                                           (const uint8_t*)w, (const float*)x, (float*)y,                                k, n_rows, n_tokens, (size_t)row_bytes);

            switch (type_id) {
                AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
                default: return AIRUN_ERR_BAD_TYPE;
            }
#undef LAUNCH
        }
    } else {
        // Each warp covers PREFILL_ROWS rows and TOKENS_PER_BLOCK tokens.
        const int warps = (n_rows + PREFILL_ROWS - 1) / PREFILL_ROWS;
        const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
                        (n_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK);

#define LAUNCH(TYPE)                                                          \
    matvec_tiled_kernel<TYPE, TOKENS_PER_BLOCK, PREFILL_ROWS>                 \
        <<<grid, block, 0, stream>>>(                                         \
        (const uint8_t*)w, (const float*)x, (float*)y,                        \
        k, n_rows, n_tokens, (size_t)row_bytes);

        switch (type_id) {
            AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    }

    AIRUN_CHECK_LAUNCH();
}

// Decode (one token): y = silu(w_gate x) * (w_up x). Same type and shape.
airun_status airun_matvec_glu(
    int32_t type_id, const void* wg, const void* wu, const void* x, void* y,
    int32_t k, int32_t n_rows, uint64_t row_bytes, cudaStream_t stream) {

    if (k % 32 != 0) return AIRUN_ERR_BAD_TYPE;
    if (n_rows <= 0) return AIRUN_OK;
    // Rows per warp: 2 and 4 measured the same on Qwen3-4B (93% of DRAM peak
    // under ncu); 1 fell to 85%, the x loads no longer amortized.
    constexpr int R = ROWS_PER_WARP;
    const dim3 block(32, ROWS_PER_BLOCK);
    const int warps = (n_rows + R - 1) / R;
    const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

#define LAUNCH(TYPE)                                                             \
    matvec_glu_kernel<TYPE, R><<<grid, block, 0, stream>>>(                      \
        (const uint8_t*)wg, (const uint8_t*)wu, (const float*)x, (float*)y,     \
        k, n_rows, (size_t)row_bytes);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_TYPE;
    }
#undef LAUNCH
    AIRUN_CHECK_LAUNCH();
}
// Decode (one token) against three matrices sharing K, type and row size.
airun_status airun_matvec3(
    int32_t type_id,
    const void* w0, const void* w1, const void* w2,
    const void* x,
    void* y0, void* y1, void* y2,
    int32_t k, int32_t n0, int32_t n1, int32_t n2,
    uint64_t row_bytes,
    cudaStream_t stream) {

    if (k % 32 != 0 || n0 % ROWS_PER_WARP != 0 || n1 % ROWS_PER_WARP != 0) return AIRUN_ERR_BAD_TYPE;
    const int rows = n0 + n1 + n2;
    if (rows <= 0) return AIRUN_OK;
    const dim3 block(32, ROWS_PER_BLOCK);
    const int warps = (rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

#define LAUNCH(TYPE)                                                             \
    matvec3_kernel<TYPE><<<grid, block, 0, stream>>>(                            \
        (const uint8_t*)w0, (const uint8_t*)w1, (const uint8_t*)w2,             \
        (const float*)x, (float*)y0, (float*)y1, (float*)y2, k, n0, n1, n2,     \
        (size_t)row_bytes);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_TYPE;
    }
#undef LAUNCH
    AIRUN_CHECK_LAUNCH();
}
// 1 when airun_matvec_q8 has a kernel for this type. The host uses this to
// decide whether to quantize the activation at all.
int32_t airun_matvec_has_q8(int32_t type_id) {
    switch (type_id) {
#define Q8CASE(TYPE) case TYPE:
        AIRUN_FOR_EACH_Q8_TYPE(Q8CASE)
#undef Q8CASE
        return 1;
        default:
            return 0;
    }
}

airun_status airun_quantize_act(
    const void* x,
    void* out,
    int64_t n_elements,
    cudaStream_t stream) {

    if (n_elements % 32 != 0) return AIRUN_ERR_BAD_TYPE;
    const int64_t n_groups = n_elements / 32;
    if (n_groups == 0) return AIRUN_OK;

    const int threads = 256;
    const int64_t blocks = (n_groups * 32 + threads - 1) / threads;
    quantize_q8a_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const float*)x, (block_q8a*)out, n_groups);

    AIRUN_CHECK_LAUNCH();
}

airun_status airun_matvec_q8(
    int32_t type_id,
    const void* w,
    const void* xq,
    void* y,
    int32_t k,
    int32_t n_rows,
    int32_t n_tokens,
    uint64_t row_bytes,
    cudaStream_t stream) {

    if (k % 32 != 0) return AIRUN_ERR_BAD_TYPE;
    if (n_rows <= 0 || n_tokens <= 0) return AIRUN_OK;

    const dim3 block(32, ROWS_PER_BLOCK);

    if (n_tokens >= MMQ_MIN_TOKENS && airun_mmq_supported()) {
#define LAUNCH(TYPE) launch_mmq<TYPE>(w, xq, y, k, n_rows, n_tokens, row_bytes, stream);

        switch (type_id) {
            AIRUN_FOR_EACH_Q8_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    } else if (n_tokens == 1 && type_id == AIRUN_TYPE_NVFP4 && !q8_wide_disabled()) {
        constexpr int R = 2;
        matvec_q8_wide_kernel<AIRUN_TYPE_NVFP4, R><<<(n_rows + R - 1) / R, WIDE_THREADS, 0, stream>>>(
            (const uint8_t*)w, (const block_q8a*)xq, (float*)y, k, n_rows, (size_t)row_bytes);
    } else if (n_tokens == 1 && !getenv("AIRUN_Q8_DECODE_MULTI")) {
        const int warps = (n_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

#define LAUNCH(TYPE)                                                       \
    matvec_q8_kernel<TYPE><<<grid, block, 0, stream>>>(                    \
        (const uint8_t*)w, (const block_q8a*)xq, (float*)y,                \
        k, n_rows, (size_t)row_bytes);

        switch (type_id) {
            AIRUN_FOR_EACH_Q8_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    } else if (n_tokens == 1 && getenv("AIRUN_Q8_DECODE_MULTI")) {
        const int warps = (n_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
#define LAUNCH(TYPE) matvec_q8_multi_kernel<TYPE, 1><<<grid, block, 0, stream>>>(                        (const uint8_t*)w, (const block_q8a*)xq, (float*)y, k, n_rows, n_tokens, (size_t)row_bytes);
        switch (type_id) {
            AIRUN_FOR_EACH_Q8_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    } else if (n_tokens <= 8 && type_id == AIRUN_TYPE_NVFP4 && !q8_wide_disabled()) {
        constexpr int R = 2;
        const int blocks = (n_rows + R - 1) / R;
        if (n_tokens <= 4) {
            matvec_q8_wide_multi_kernel<AIRUN_TYPE_NVFP4, R, 4><<<blocks, WIDE_THREADS, 0, stream>>>(
                (const uint8_t*)w, (const block_q8a*)xq, (float*)y, k, n_rows, n_tokens, (size_t)row_bytes);
        } else {
            matvec_q8_wide_multi_kernel<AIRUN_TYPE_NVFP4, R, 8><<<blocks, WIDE_THREADS, 0, stream>>>(
                (const uint8_t*)w, (const block_q8a*)xq, (float*)y, k, n_rows, n_tokens, (size_t)row_bytes);
        }
    } else if (n_tokens <= 8 && !q8_multi_disabled()) {
        const int warps = (n_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);

#define LAUNCH(TYPE)                                                                        if (n_tokens <= 2) matvec_q8_multi_kernel<TYPE, 2><<<grid, block, 0, stream>>>(             (const uint8_t*)w, (const block_q8a*)xq, (float*)y, k, n_rows, n_tokens, (size_t)row_bytes);     else if (n_tokens <= 4) matvec_q8_multi_kernel<TYPE, 4><<<grid, block, 0, stream>>>(         (const uint8_t*)w, (const block_q8a*)xq, (float*)y, k, n_rows, n_tokens, (size_t)row_bytes);     else matvec_q8_multi_kernel<TYPE, 8><<<grid, block, 0, stream>>>(                           (const uint8_t*)w, (const block_q8a*)xq, (float*)y, k, n_rows, n_tokens, (size_t)row_bytes);

        switch (type_id) {
            AIRUN_FOR_EACH_Q8_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    } else {
        const int warps = (n_rows + Q8_PREFILL_ROWS - 1) / Q8_PREFILL_ROWS;
        const dim3 grid((warps + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
                        (n_tokens + Q8_TOKENS_PER_BLOCK - 1) / Q8_TOKENS_PER_BLOCK);

#define LAUNCH(TYPE)                                                          \
    matvec_q8_tiled_kernel<TYPE, Q8_TOKENS_PER_BLOCK, Q8_PREFILL_ROWS>              \
        <<<grid, block, 0, stream>>>(                                         \
        (const uint8_t*)w, (const block_q8a*)xq, (float*)y,                   \
        k, n_rows, n_tokens, (size_t)row_bytes);

        switch (type_id) {
            AIRUN_FOR_EACH_Q8_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
    }

    AIRUN_CHECK_LAUNCH();
}

airun_status airun_dequantize(
    int32_t type_id,
    const void* src,
    void* dst,
    int64_t n_elements,
    cudaStream_t stream) {

    if (n_elements % 32 != 0) return AIRUN_ERR_BAD_TYPE;
    const int64_t n_groups = n_elements / 32;
    if (n_groups == 0) return AIRUN_OK;

    const int threads = 256;
    const int64_t blocks = (n_groups + threads - 1) / threads;

#define LAUNCH(TYPE)                                                       \
    dequantize_kernel<TYPE><<<(unsigned)blocks, threads, 0, stream>>>(     \
        (const uint8_t*)src, (float*)dst, n_groups);

    switch (type_id) {
        AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
        default: return AIRUN_ERR_BAD_TYPE;
    }
#undef LAUNCH

    AIRUN_CHECK_LAUNCH();
}

} // extern "C"
