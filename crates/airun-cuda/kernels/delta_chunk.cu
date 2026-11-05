// Gated delta rule over a prompt in chunks of 64 tokens, on tensor cores.
//
// The recurrence of ssm.cu (S_t = d_t S_{t-1} + b_t k_t (v_t - d_t S_{t-1}^T k_t)^T,
// o_t = S_t^T q_t) does four operations per state element per token, in a
// chain the GPU cannot spread: on the 27B it was a fifth of the prefill, and
// reorganizing the same chain (staged inputs, warp-level reductions, other
// splits of dk) only made it slower. Within a chunk the recurrence has a
// closed form (the WY representation of DeltaNet, as in flash-linear-attention's
// chunk_gated_delta_rule). With G_i the running sum of log decays inside the
// chunk, g_i = exp(G_i), and S0 the state before it:
//
//   A_ij = b_i (k_i . k_j) g_i / g_j  for j < i, else 0;   T = (I + A)^-1
//   W = T diag(b g) K          U = T diag(b) V
//   E = U - W S0                         (each token's correction, times b)
//   o_i = g_i q_i^T S0 + sum_{j<=i} (q_i . k_j) g_i / g_j  E_j
//   S  = g_C S0 + sum_j (g_C / g_j) k_j E_j^T
//
// Everything but T's triangular solve is a matrix product, run as bf16 WMMA
// with fp32 accumulation; the state itself stays fp32. Two kernels: the
// first builds W, U, the decayed Q and K and the masked Q K^T for every
// (head, chunk) at once; the second walks the chunks in order per (head,
// 32 value columns), which is the only sequential part left.
//
// bf16 operands make this a few 1e-3 relative off the fp32 recurrence (the
// selftest's bound); decode and speculative verifies keep the recurrence.

#include "airun.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define AIRUN_NO_BF16_WMMA 1
#endif

#include <mma.h>
#include <cuda_bf16.h>
using namespace nvcuda;

#define DC_C 64      // tokens per chunk
#define DC_D 128     // head_k_dim = head_v_dim
#define DC_COLS 32   // value columns per block of the sequential pass
// Per (head, chunk) scratch, bf16 elements: W, U, Qg, Kd (C x D each), P (C x C).
#define DC_PER (4 * DC_C * DC_D + DC_C * DC_C)

// Shared-memory row strides, padded off multiples of 128 bytes: with 256-byte
// rows every row of a 16 x 16 fragment fell in the same banks.
#define LK (DC_D + 8)        // bf16 rows of D (K, Q)
#define LA (DC_C + 4)        // fp32 rows of C (A, T)
#define LT (DC_C + 8)        // bf16 rows of C (T)
#define LS (DC_COLS + 4)     // fp32 rows of the state slice and E
#define LSB (DC_COLS + 8)    // bf16 rows of the same

typedef __nv_bfloat16 bf16;

// --- pass 1: per (v head, chunk), fully parallel -----------------------------
__global__ __launch_bounds__(256) void delta_chunk_prep_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ decay, const float* __restrict__ beta,
    bf16* __restrict__ scratch, float* __restrict__ gam,
    const int n_tokens, const int n_k_heads, const int n_v_heads) {
#ifndef AIRUN_NO_BF16_WMMA
    const int h = blockIdx.x, c = blockIdx.y, n_chunks = gridDim.y;
    const int kh = h % n_k_heads;
    const int t0 = c * DC_C;
    const int nt = min(DC_C, n_tokens - t0);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

    extern __shared__ __align__(32) unsigned char dc_smem[];
    bf16* Kb = (bf16*)dc_smem;                         // [C][D] k, then scaled k, then scaled v
    bf16* Qb = Kb + DC_C * LK;                         // [C][D] q
    float* A = (float*)(Qb + DC_C * LK);               // [C][C] K K^T, then A
    float* M = A + DC_C * LA;                          // [C][C] Q K^T, then T
    bf16* Tb = (bf16*)(M + DC_C * LA);                 // [C][C]
    float* stage = (float*)(Tb + DC_C * LT);           // [8 warps][16][16]
    __shared__ float G[DC_C], Bt[DC_C];

    bf16* W  = scratch + (size_t)(h * n_chunks + c) * DC_PER;
    bf16* U  = W + DC_C * DC_D;
    bf16* Qg = U + DC_C * DC_D;
    bf16* Kd = Qg + DC_C * DC_D;
    bf16* P  = Kd + DC_C * DC_D;

    // Running log decay, two tokens a lane and a shuffle scan; tokens past
    // the end decay by nothing and add nothing.
    if (tid < 32) {
        float l[2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int i = 2 * lane + u;
            const bool ok = i < nt;
            l[u] = ok ? logf(fmaxf(decay[(size_t)(t0 + i) * n_v_heads + h], 1e-30f)) : 0.0f;
            Bt[i] = ok ? beta[(size_t)(t0 + i) * n_v_heads + h] : 0.0f;
        }
        float run = l[0] + l[1];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, run, o);
            if (lane >= o) run += n;
        }
        G[2 * lane + 1] = run;
        G[2 * lane] = run - l[1];
    }
    for (int i = tid; i < DC_C * DC_D; i += 256) {
        const int t = i / DC_D, d = i % DC_D;
        const size_t src = ((size_t)(t0 + t) * n_k_heads + kh) * DC_D + d;
        Kb[t * LK + d] = __float2bfloat16(t < nt ? k[src] : 0.0f);
        Qb[t * LK + d] = __float2bfloat16(t < nt ? q[src] : 0.0f);
    }
    __syncthreads();

    // K K^T and Q K^T: 16 tiles each, two per warp.
    for (int tile = warp; tile < 32; tile += 8) {
        const int which = tile / 16, ti = (tile % 16) / 4, tj = tile % 4;
        const bf16* X = which ? Qb : Kb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk < DC_D; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
            wmma::load_matrix_sync(a, X + ti * 16 * LK + kk, LK);
            wmma::load_matrix_sync(b, Kb + tj * 16 * LK + kk, LK);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync((which ? M : A) + ti * 16 * LA + tj * 16, acc, LA, wmma::mem_row_major);
    }
    __syncthreads();
    // A masked and scaled; P = masked Q K^T, to the scratch.
    for (int e = tid; e < DC_C * DC_C; e += 256) {
        const int i = e / DC_C, j = e % DC_C;
        const float r = __expf(G[i] - G[j]);
        A[i * LA + j] = j < i ? Bt[i] * A[i * LA + j] * r : 0.0f;
        P[e] = __float2bfloat16(j <= i ? M[i * LA + j] * r : 0.0f);
    }
    // Decayed q (g_i q_i) and k (g_C / g_j k_j), to the scratch.
    for (int i = tid; i < DC_C * DC_D; i += 256) {
        const int t = i / DC_D, d = i % DC_D;
        Qg[i] = __float2bfloat16(__bfloat162float(Qb[t * LK + d]) * __expf(G[t]));
        Kd[i] = __float2bfloat16(__bfloat162float(Kb[t * LK + d]) * __expf(G[DC_C - 1] - G[t]));
    }
    __syncthreads();

    // T = (I + A)^-1 in 16 x 16 blocks (A is strictly lower, T unit lower).
    // Diagonal blocks first, one warp each, by forward substitution (row r =
    // e_r - sum_{j<r} A_rj T_j, lane = column); then block row I from the
    // ones above it: T_IJ = -T_II sum_{K=J}^{I-1} A_IK T_KJ. A row-by-row
    // solve of the whole 64 x 64 was 64 block barriers and most of this pass.
    for (int e = tid; e < DC_C * LA; e += 256) M[e] = 0.0f;
    __syncthreads();
    if (warp < 4) {
        const int b0 = warp * 16;
        const int col = lane & 15;
        for (int r = 0; r < 16; ++r) {
            float s = 0.0f;
            for (int j = col; j < r; ++j) s = fmaf(A[(b0 + r) * LA + b0 + j], M[(b0 + j) * LA + b0 + col], s);
            if (lane < 16) M[(b0 + r) * LA + b0 + col] = (col == r ? 1.0f : 0.0f) - (col < r ? s : 0.0f);
            __syncwarp();
        }
    }
    __syncthreads();
    {
        float* X = stage;                       // [16][16]
        const int er = tid / 16, ec = tid % 16;
        for (int I = 1; I < 4; ++I) {
            for (int J = 0; J < I; ++J) {
                float x = 0.0f;
                for (int K = J; K < I; ++K)
                    for (int m = 0; m < 16; ++m)
                        x = fmaf(A[(I * 16 + er) * LA + K * 16 + m], M[(K * 16 + m) * LA + J * 16 + ec], x);
                X[er * 16 + ec] = x;
                __syncthreads();
                float t = 0.0f;
                for (int m = 0; m < 16; ++m) t = fmaf(M[(I * 16 + er) * LA + I * 16 + m], X[m * 16 + ec], t);
                M[(I * 16 + er) * LA + J * 16 + ec] = -t;
                __syncthreads();
            }
        }
    }
    for (int e = tid; e < DC_C * DC_C; e += 256) Tb[(e / DC_C) * LT + e % DC_C] = __float2bfloat16(M[(e / DC_C) * LA + e % DC_C]);
    // Kb := b_j g_j k_j for W.
    for (int i = tid; i < DC_C * DC_D; i += 256) {
        const int t = i / DC_D, d = i % DC_D;
        Kb[t * LK + d] = __float2bfloat16(__bfloat162float(Kb[t * LK + d]) * Bt[t] * __expf(G[t]));
    }
    __syncthreads();

    // W = T Kb, then (Kb := b_j v_j) U = T Kb: 32 tiles each, four per warp.
    float* st = stage + warp * 256;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            __syncthreads();
            for (int i = tid; i < DC_C * DC_D; i += 256) {
                const int t = i / DC_D, d = i % DC_D;
                Kb[t * LK + d] = __float2bfloat16(t < nt ? v[((size_t)(t0 + t) * n_v_heads + h) * DC_D + d] * Bt[t] : 0.0f);
            }
            __syncthreads();
        }
        bf16* dst = pass ? U : W;
        for (int tile = warp; tile < 32; tile += 8) {
            const int ti = tile / 8, tj = tile % 8;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < DC_C; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, Tb + ti * 16 * LT + kk, LT);
                wmma::load_matrix_sync(b, Kb + kk * LK + tj * 16, LK);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(st, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                dst[(ti * 16 + e / 16) * DC_D + tj * 16 + e % 16] = __float2bfloat16(st[e]);
            __syncwarp();
        }
    }
    if (tid == 0) gam[h * n_chunks + c] = __expf(G[DC_C - 1]);
#endif
}

// --- pass 2: per (v head, 32 value columns), the chunks in order --------------
__global__ __launch_bounds__(256) void delta_chunk_scan_kernel(
    float* __restrict__ state, const bf16* __restrict__ scratch, const float* __restrict__ gam,
    float* __restrict__ out, const int n_tokens, const int n_v_heads, const int keep_state) {
#ifndef AIRUN_NO_BF16_WMMA
    const int h = blockIdx.x, slice = blockIdx.y;
    const int n_chunks = (n_tokens + DC_C - 1) / DC_C;
    const int tid = threadIdx.x, warp = tid / 32;
    const int c0 = slice * DC_COLS;

    extern __shared__ __align__(32) unsigned char dc_smem[];
    float* S  = (float*)dc_smem;              // [D][32] fp32 state slice
    bf16* Sb  = (bf16*)(S + DC_D * LS);       // [D][32]
    float* E  = (float*)(Sb + DC_D * LSB);    // [C][32] corrections, then the output tile
    bf16* Eb  = (bf16*)(E + DC_C * LS);       // [C][32]

    float* Sg = state + (size_t)h * DC_D * DC_D;
    for (int e = tid; e < DC_D * DC_COLS; e += 256) S[(e / DC_COLS) * LS + e % DC_COLS] = Sg[(e / DC_COLS) * DC_D + c0 + e % DC_COLS];

    for (int c = 0; c < n_chunks; ++c) {
        const bf16* W  = scratch + (size_t)(h * n_chunks + c) * DC_PER;
        const bf16* U  = W + DC_C * DC_D;
        const bf16* Qg = U + DC_C * DC_D;
        const bf16* Kd = Qg + DC_C * DC_D;
        const bf16* P  = Kd + DC_C * DC_D;
        const int t0 = c * DC_C, nt = min(DC_C, n_tokens - t0);
        __syncthreads();
        for (int e = tid; e < DC_D * DC_COLS; e += 256)
            Sb[(e / DC_COLS) * LSB + e % DC_COLS] = __float2bfloat16(S[(e / DC_COLS) * LS + e % DC_COLS]);
        __syncthreads();

        // E = U - W S0: an 4 x 2 grid of tiles, one per warp.
        const int ti = warp / 2, tj = warp % 2;
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < DC_D; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, W + ti * 16 * DC_D + kk, DC_D);
                wmma::load_matrix_sync(b, Sb + kk * LSB + tj * 16, LSB);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(E + ti * 16 * LS + tj * 16, acc, LS, wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < DC_C * DC_COLS; e += 256) {
            const int r = e / DC_COLS, cc = e % DC_COLS;
            Eb[r * LSB + cc] = __float2bfloat16(__bfloat162float(U[r * DC_D + c0 + cc]) - E[r * LS + cc]);
        }
        __syncthreads();

        // O = Qg S0 + P E, staged in E (no longer needed in fp32).
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < DC_D; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, Qg + ti * 16 * DC_D + kk, DC_D);
                wmma::load_matrix_sync(b, Sb + kk * LSB + tj * 16, LSB);
                wmma::mma_sync(acc, a, b, acc);
            }
            for (int kk = 0; kk < DC_C; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, P + ti * 16 * DC_C + kk, DC_C);
                wmma::load_matrix_sync(b, Eb + kk * LSB + tj * 16, LSB);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(E + ti * 16 * LS + tj * 16, acc, LS, wmma::mem_row_major);
        }
        // S = g_C S0 (scaled before it seeds the accumulators below).
        const float gC = gam[h * n_chunks + c];
        for (int e = tid; e < DC_D * DC_COLS; e += 256) S[(e / DC_COLS) * LS + e % DC_COLS] *= gC;
        __syncthreads();
        for (int e = tid; e < nt * DC_COLS; e += 256)
            out[((size_t)(t0 + e / DC_COLS) * n_v_heads + h) * DC_D + c0 + e % DC_COLS] = E[(e / DC_COLS) * LS + e % DC_COLS];

        // S += Kd^T E: 8 x 2 tiles of the [D][32] slice, two per warp.
        for (int tile = warp; tile < 16; tile += 8) {
            const int si = tile / 2, sj = tile % 2;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, S + si * 16 * LS + sj * 16, LS, wmma::mem_row_major);
            for (int kk = 0; kk < DC_C; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b;
                wmma::load_matrix_sync(a, Kd + kk * DC_D + si * 16, DC_D);
                wmma::load_matrix_sync(b, Eb + kk * LSB + sj * 16, LSB);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(S + si * 16 * LS + sj * 16, acc, LS, wmma::mem_row_major);
        }
    }
    __syncthreads();
    if (keep_state)
        for (int e = tid; e < DC_D * DC_COLS; e += 256) Sg[(e / DC_COLS) * DC_D + c0 + e % DC_COLS] = S[(e / DC_COLS) * LS + e % DC_COLS];
#endif
}

extern "C" {

// Whether the chunked path takes this call: a prompt-sized batch, 128-wide
// heads, a GPU with bf16 tensor cores. AIRUN_NO_DELTA_CHUNK=1 keeps the
// recurrence everywhere.
int32_t airun_delta_chunk_ok(int32_t n_tokens, int32_t head_k_dim, int32_t head_v_dim, int32_t keep_state) {
    static int disabled = -1;
    if (disabled < 0) {
        const char* e = getenv("AIRUN_NO_DELTA_CHUNK");
        disabled = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (disabled || !keep_state || n_tokens < DC_C || head_k_dim != DC_D || head_v_dim != DC_D) return 0;
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 8;
}

airun_status airun_delta_chunk(void* state, const void* q, const void* k, const void* v, const void* decay,
                               const void* beta, void* out, int32_t n_tokens, int32_t n_k_heads,
                               int32_t n_v_heads, cudaStream_t stream) {
    // Scratch per device, grown as needed: prompts run outside graph capture.
    static void* buf[64] = {};
    static size_t cap[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return (airun_status)cudaErrorInvalidDevice;
    const int n_chunks = (n_tokens + DC_C - 1) / DC_C;
    const size_t need = (size_t)n_v_heads * n_chunks * (DC_PER * sizeof(bf16) + sizeof(float));
    if (need > cap[dev]) {
        if (buf[dev]) cudaFree(buf[dev]);
        buf[dev] = nullptr;
        cap[dev] = 0;
        if (cudaMalloc(&buf[dev], need) != cudaSuccess) return (airun_status)cudaErrorMemoryAllocation;
        cap[dev] = need;
    }
    bf16* scratch = (bf16*)buf[dev];
    float* gam = (float*)((unsigned char*)buf[dev] + (size_t)n_v_heads * n_chunks * DC_PER * sizeof(bf16));

    const size_t prep_smem = 2 * (size_t)DC_C * LK * 2 + 2 * (size_t)DC_C * LA * 4 + (size_t)DC_C * LT * 2
        + 8 * 256 * 4;
    cudaFuncSetAttribute(delta_chunk_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)prep_smem);
    delta_chunk_prep_kernel<<<dim3(n_v_heads, n_chunks), 256, prep_smem, stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)decay, (const float*)beta,
        scratch, gam, n_tokens, n_k_heads, n_v_heads);
    AIRUN_TRY(cudaGetLastError());
    const size_t scan_smem = (size_t)DC_D * (LS * 4 + LSB * 2) + (size_t)DC_C * (LS * 4 + LSB * 2);
    delta_chunk_scan_kernel<<<dim3(n_v_heads, DC_D / DC_COLS), 256, scan_smem, stream>>>(
        (float*)state, scratch, gam, (float*)out, n_tokens, n_v_heads, 1);
    AIRUN_CHECK_LAUNCH();
}

}
