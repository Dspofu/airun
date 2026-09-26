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

#include "dequant.cuh"
#include "vecdotq.cuh"

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

    // Warp-uniform: derived from blockIdx and threadIdx.y only, so the loops
    // below exit together and warp_reduce_sum always sees a full warp.
    const int n_row = min(ROWS_PER_WARP, n_rows - row0);
    const int groups = K / 32;

    float acc[ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) acc[r] = 0.0f;

    for (int g = threadIdx.x; g < groups; g += 32) {
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

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            if (r >= n_row) break;
            float vals[32];
            decode_group<TYPE>(w + (size_t)(row0 + r) * row_bytes, g, vals);
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

// ---------------------------------------------------------------------------
// Integer (dp4a) path: quantized activations against quantized weights.
// See vecdotq.cuh for the trade being made and the validation story.
// ---------------------------------------------------------------------------

// f32 -> block_q8a, one thread per 32-element group. Memory bound and tiny
// next to any matvec that consumes it.
__global__ void quantize_q8a_kernel(
    const float* __restrict__ x,
    block_q8a* __restrict__ out,
    const int64_t n_groups) {

    const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;

    float v[32];
    const float4* xg = (const float4*)x + g * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 f = xg[i];
        v[i * 4 + 0] = f.x;
        v[i * 4 + 1] = f.y;
        v[i * 4 + 2] = f.z;
        v[i * 4 + 3] = f.w;
    }

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));

    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    int sum = 0;
    int8_t q[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        // Round-to-nearest-even, mirrored by round_ties_even in the CPU check.
        const int qi = __float2int_rn(v[i] * id);
        q[i] = (int8_t)qi;
        sum += qi;
    }

#pragma unroll
    for (int i = 0; i < 32; ++i) out[g].qs[i] = q[i];
    out[g].ds = __floats2half2_rn(d, d * (float)sum);
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
    X(AIRUN_TYPE_MXFP4)         \
    X(AIRUN_TYPE_NVFP4)

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

extern "C" {

// cudaErrorInvalidValue, returned for a type this build has no kernel for.
#define AIRUN_ERR_BAD_TYPE 1

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

    if (n_tokens == 1 && n_rows < 512) {
        // Narrow decode matrices: one block per row (see matvec_tallk_kernel).
        const dim3 tgrid(n_rows);
        const dim3 tblock(128);

#define LAUNCH(TYPE)                                                           matvec_tallk_kernel<TYPE><<<tgrid, tblock, 0, stream>>>(                       (const uint8_t*)w, (const float*)x, (float*)y,                             k, n_rows, (size_t)row_bytes);

        switch (type_id) {
            AIRUN_FOR_EACH_TYPE(AIRUN_CASE)
            default: return AIRUN_ERR_BAD_TYPE;
        }
#undef LAUNCH
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

        if (cc_major >= 8) {
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
    const int64_t blocks = (n_groups + threads - 1) / threads;
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

    if (n_tokens == 1) {
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
