// Integer dot products between quantized weights and quantized activations.
//
// The f32 matvec path dequantizes every weight to a float and pays one FMA per
// element. Keeping the weights in their integer codes and quantizing the
// activation once per forward pass lets `__dp4a` do four multiply-accumulates
// per instruction on data a quarter the size -- the same trade llama.cpp's
// `mmvq` path makes, and the reason its decode outruns a dequantizing kernel.
//
// This is NOT bit-identical to the f32 path: the activation is rounded to 8
// bits (plus a per-32 scale), which perturbs each product by about 1 part in
// 2^8 of the group's largest element. `airun selftest` checks this path against
// a CPU mirror of the same algorithm with the backward-error bound, and
// `airun perplexity` is the end-to-end guard. The f32 path stays available via
// AIRUN_MATVEC_F32=1 and remains the reference.
//
// The quantization itself (round-to-nearest-even against amax/127, sum folded
// into the second half) mirrors `quantize_q8a` in airun-cli's selftest -- the
// two must change together, like dequant.rs and dequant.cuh.

#pragma once

#include "dequant.cuh"

// One 32-element group of a quantized activation: the codes, then the scale
// `d` and the folded sum `s = d * sum(qs)` as a half2. 36 bytes, so an array
// of blocks keeps every field 4-byte aligned, which is all `__dp4a` loads need.
typedef struct {
    int8_t qs[32];
    half2  ds;
} block_q8a;

static_assert(sizeof(block_q8a) == 36, "block_q8a layout");

// ---------------------------------------------------------------------------
// Per-type integer dots. Each returns dot(W[row] group g, activation group g)
// as f32, reproducing the same value the f32 decoder pair would, up to the
// final float combine.
// ---------------------------------------------------------------------------

// w = d * q  ->  sum(w*a) = d * d_a * dot(q, qa)
__device__ __forceinline__ float vdot_q8_0(const uint8_t* wr, int g,
                                           const block_q8a* a) {
    const block_q8_0* b = (const block_q8_0*)wr + g;

    uint32_t w[8];
    load_32b_align2(b->qs, w); // 34-byte block: 2-byte alignment only
    const int* aw = (const int*)a[g].qs;

    int dot = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) dot = __dp4a((int)w[i], aw[i], dot);

    const float2 ds = __half22float2(a[g].ds);
    return __half2float(b->d) * ds.x * (float)dot;
}

// w = dl*qn - ml  ->  sum(w*a) = dl*d_a*dot(qn, qa) - ml*s_a
__device__ __forceinline__ float vdot_q4_K(const uint8_t* wr, int g,
                                           const block_q8a* a) {
    const int block = g / 8;
    const int sub   = g % 8;
    const block_q4_K* b = (const block_q4_K*)(wr + (size_t)block * sizeof(block_q4_K));

    float d, dmin;
    uint8_t sc, m;
    load_k_header(b, sub, &d, &dmin, &sc, &m);

    uint32_t w[8];
    load_32b_align16(b->qs + (sub / 2) * 32, w);
    const int shift = (sub & 1) ? 4 : 0;
    const int* aw = (const int*)a[g].qs;

    int dot = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dot = __dp4a((int)((w[i] >> shift) & 0x0F0F0F0Fu), aw[i], dot);
    }

    const float2 ds = __half22float2(a[g].ds);
    return d * (float)sc * ds.x * (float)dot - dmin * (float)m * ds.y;
}

// Q4_K plus the fifth bit pulled from qh.
__device__ __forceinline__ float vdot_q5_K(const uint8_t* wr, int g,
                                           const block_q8a* a) {
    const int block = g / 8;
    const int sub   = g % 8;
    const block_q5_K* b = (const block_q5_K*)(wr + (size_t)block * sizeof(block_q5_K));

    float d, dmin;
    uint8_t sc, m;
    load_k_header(b, sub, &d, &dmin, &sc, &m);

    uint32_t w[8], h[8];
    load_32b_align16(b->qs + (sub / 2) * 32, w);
    load_32b_align16(b->qh, h);
    const int shift = (sub & 1) ? 4 : 0;
    const int* aw = (const int*)a[g].qs;

    int dot = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t lo = (w[i] >> shift) & 0x0F0F0F0Fu;
        const uint32_t hi = ((h[i] >> sub) & 0x01010101u) << 4;
        dot = __dp4a((int)(lo | hi), aw[i], dot);
    }

    const float2 ds = __half22float2(a[g].ds);
    return d * (float)sc * ds.x * (float)dot - dmin * (float)m * ds.y;
}

// w = d * sc16 * (q - 32), where sc16 changes at the group's midpoint -- so two
// integer dots, one per 16-element half, each scaled by its own int8 scale.
__device__ __forceinline__ float vdot_q6_K(const uint8_t* wr, int g,
                                           const block_q8a* a) {
    const int block = g / 8;
    const int sub   = g % 8;
    const block_q6_K* b = (const block_q6_K*)(wr + (size_t)block * sizeof(block_q6_K));

    const int half_   = sub / 4;
    const int quarter = sub % 4;
    const uint8_t* ql = b->ql + half_ * 64 + (quarter & 1) * 32;
    const uint8_t* qh = b->qh + half_ * 32;
    const int8_t*  sc = b->scales + half_ * 8 + quarter * 2;
    const int shift   = quarter * 2;
    const int loshift = (quarter >= 2) ? 4 : 0;

    uint32_t wl[8], wh[8];
    load_32b_align2(ql, wl); // 210-byte block: 2-byte alignment only
    load_32b_align2(qh, wh);
    const int* aw = (const int*)a[g].qs;

    int dot0 = 0, dot1 = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t lo = (wl[i] >> loshift) & 0x0F0F0F0Fu;
        const uint32_t hi = ((wh[i] >> shift) & 0x03030303u) << 4;
        // (q - 32) per byte; the wrap-around subtraction produces exactly the
        // two's-complement pattern __dp4a reads as a signed byte.
        const int q = (int)__vsub4(lo | hi, 0x20202020u);
        if (i < 4) dot0 = __dp4a(q, aw[i], dot0);
        else       dot1 = __dp4a(q, aw[i], dot1);
    }

    const float2 ds = __half22float2(a[g].ds);
    const float d = __half2float(b->d);
    return d * ds.x * ((float)sc[0] * (float)dot0 + (float)sc[1] * (float)dot1);
}

// w = d_run * code for each run of 16 (two runs per group), codes the doubled
// E2M1 values; d_run already carries the half.
__device__ __forceinline__ float vdot_nvfp4(const uint8_t* wr, int g,
                                            const block_q8a* a) {
    const block_nvfp4* b = (const block_nvfp4*)wr + g / 2;
    const uint32_t* q = (const uint32_t*)(b->qs + (g & 1) * 16);
    const int* aw = (const int*)a[g].qs;

    float sum = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t lo0, hi0, lo1, hi1;
        fp4_codes8(q[h * 2 + 0], lo0, hi0);
        fp4_codes8(q[h * 2 + 1], lo1, hi1);
        int dot = __dp4a((int)lo0, aw[h * 4 + 0], 0);
        dot = __dp4a((int)lo1, aw[h * 4 + 1], dot);
        dot = __dp4a((int)hi0, aw[h * 4 + 2], dot);
        dot = __dp4a((int)hi1, aw[h * 4 + 3], dot);
        sum = fmaf(ue4m3_to_f32(b->d[(g & 1) * 2 + h]), (float)dot, sum);
    }
    return sum * __low2float(a[g].ds);
}

// Types with an integer path. Everything else (floats, the rarer quants) stays
// on the f32 kernels -- dispatch falls back per tensor, so a model mixing both
// still runs whole.
#define AIRUN_FOR_EACH_Q8_TYPE(X) \
    X(AIRUN_TYPE_Q8_0)            \
    X(AIRUN_TYPE_Q4_K)            \
    X(AIRUN_TYPE_Q5_K)            \
    X(AIRUN_TYPE_Q6_K)            \
    X(AIRUN_TYPE_NVFP4)

template <int TYPE> struct q8dot;
#define AIRUN_DEFINE_Q8DOT(TYPE, FN)                                          \
    template <> struct q8dot<TYPE> {                                          \
        __device__ static __forceinline__ float dot(                          \
            const uint8_t* wr, int g, const block_q8a* a) {                   \
            return FN(wr, g, a);                                              \
        }                                                                     \
    }

AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_Q8_0, vdot_q8_0);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_Q4_K, vdot_q4_K);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_Q5_K, vdot_q5_K);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_Q6_K, vdot_q6_K);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_NVFP4, vdot_nvfp4);
