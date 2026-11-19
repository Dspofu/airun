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

// ---------------------------------------------------------------------------
// The IQ types and Q2_0, for the MoE decode kernels. Qwen4 Flash-Next keeps
// its experts in these (IQ3_XXS/IQ2_S/IQ3_S gate and up, IQ4_NL/Q2_0 down),
// and their float decoders -- a codebook lookup and 32 FMAs per group --
// held the expert kernels at 15-35 GB/s on an RTX 3060.
// ---------------------------------------------------------------------------

// 16 bytes from a 2-byte-aligned address as four words, by the funnel shift
// of load_32b_align2: every byte read lies in [p - 2, p + 16).
__device__ __forceinline__ void load_16b_align2(const void* p, uint32_t* w) {
    const uintptr_t a = (uintptr_t)p;
    const uint32_t* base = (const uint32_t*)(a & ~(uintptr_t)3);
    const uint32_t sh = (uint32_t)(a & 3) * 8;   // 0 or 16
    uint32_t v[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = base[i];
    v[4] = sh ? (uint32_t)*(const uint16_t*)(base + 4) : 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = __funnelshift_r(v[i], v[i + 1], sh);
}

// A word from a 2-byte-aligned address.
__device__ __forceinline__ uint32_t load_u32_align2(const uint8_t* p) {
    const uint16_t* h = (const uint16_t*)p;
    return (uint32_t)h[0] | ((uint32_t)h[1] << 16);
}

// 0xFF in byte j where bit j of `s` (j < 4) is set: the four sign bits of a
// lattice point's half as a per-byte mask, so (v ^ m) - m negates those bytes.
__device__ __forceinline__ uint32_t sign_mask4(uint32_t s) {
    return (((s & 0x0Fu) * 0x00204081u) & 0x01010101u) * 0xFFu;
}

__device__ __forceinline__ int signed_codes4(uint32_t mag, uint32_t s) {
    const uint32_t m = sign_mask4(s);
    return (int)__vsub4(mag ^ m, m);
}

// w = d * kvalue[nibble]  ->  d * d_a * dot(kvalues, qa)
__device__ __forceinline__ float vdot_iq4_nl(const uint8_t* wr, int g, const block_q8a* a) {
    const block_iq4_nl* b = (const block_iq4_nl*)wr + g;
    uint32_t q[4];
    load_16b_align2(b->qs, q);   // an 18-byte block: 2-byte alignment only
    const int* aw = (const int*)a[g].qs;
    int dot = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo, hi;
        iq4_codes8(q[j], lo, hi);   // elements 4j..4j+3 and 16+4j..16+4j+3
        dot = __dp4a((int)lo, aw[j], dot);
        dot = __dp4a((int)hi, aw[4 + j], dot);
    }
    return __half2float(b->d) * __low2float(a[g].ds) * (float)dot;
}

// As IQ4_NL under a 6-bit scale per 32 of a 256 superblock.
__device__ __forceinline__ float vdot_iq4_xs(const uint8_t* wr, int g, const block_q8a* a) {
    const block_iq4_xs* b = (const block_iq4_xs*)(wr + (size_t)(g / 8) * sizeof(block_iq4_xs));
    const int sub = g % 8;
    const int ls = ((b->scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0F) | (((b->scales_h >> (2 * sub)) & 3) << 4);
    // A 136-byte block keeps its codes 8-byte aligned.
    const uint2 q01 = *(const uint2*)(b->qs + sub * 16);
    const uint2 q23 = *(const uint2*)(b->qs + sub * 16 + 8);
    const uint32_t q[4] = {q01.x, q01.y, q23.x, q23.y};
    const int* aw = (const int*)a[g].qs;
    int dot = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo, hi;
        iq4_codes8(q[j], lo, hi);
        dot = __dp4a((int)lo, aw[j], dot);
        dot = __dp4a((int)hi, aw[4 + j], dot);
    }
    return __half2float(b->d) * (float)(ls - 32) * __low2float(a[g].ds) * (float)dot;
}

// Four lattice points of 8 magnitudes (iq2s_grid, 10-bit index), a sign byte
// per point, a 4-bit scale per 16: two integer dots, one per scale.
__device__ __forceinline__ float vdot_iq2_s(const uint8_t* wr, int g, const block_q8a* a) {
    const uint8_t* b = wr + (size_t)(g / 8) * sizeof(block_iq2_s);   // 82 bytes: 2-byte aligned
    const int sub = g % 8;
    const uint32_t qs = load_u32_align2(b + 2 + 4 * sub);
    const uint32_t sg = load_u32_align2(b + 34 + 4 * sub);
    const uint32_t qh = b[66 + sub];
    const uint32_t sc = b[74 + sub];
    const int* aw = (const int*)a[g].qs;
    int dot0 = 0, dot1 = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int idx = (int)((qs >> (8 * l)) & 0xFF) | (int)((qh << (8 - 2 * l)) & 0x300);
        const uint64_t grid = iq2s_grid[idx];
        const uint32_t s = sg >> (8 * l);
        const int v0 = signed_codes4((uint32_t)grid, s);
        const int v1 = signed_codes4((uint32_t)(grid >> 32), s >> 4);
        if (l < 2) {
            dot0 = __dp4a(v0, aw[2 * l], dot0);
            dot0 = __dp4a(v1, aw[2 * l + 1], dot0);
        } else {
            dot1 = __dp4a(v0, aw[2 * l], dot1);
            dot1 = __dp4a(v1, aw[2 * l + 1], dot1);
        }
    }
    const float d = __half2float(*(const half*)b) * 0.25f * __low2float(a[g].ds);
    return d * ((0.5f + (float)(sc & 0x0F)) * (float)dot0 + (0.5f + (float)(sc >> 4)) * (float)dot1);
}

// Eight 4-magnitude points (iq3xxs_grid), four 7-bit sign codes and a 4-bit
// scale packed in one word per group.
__device__ __forceinline__ float vdot_iq3_xxs(const uint8_t* wr, int g, const block_q8a* a) {
    const uint8_t* b = wr + (size_t)(g / 8) * sizeof(block_iq3_xxs);   // 98 bytes: 2-byte aligned
    const int sub = g % 8;
    const uint32_t aux = load_u32_align2(b + 66 + 4 * sub);
    const int* aw = (const int*)a[g].qs;
    int dot = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t idx = *(const uint16_t*)(b + 2 + 8 * sub + 2 * l);
        const uint32_t s = ksigns_iq2xs((aux >> (7 * l)) & 127);
        dot = __dp4a(signed_codes4(iq3xxs_grid[idx & 0xFF], s), aw[2 * l], dot);
        dot = __dp4a(signed_codes4(iq3xxs_grid[idx >> 8], s >> 4), aw[2 * l + 1], dot);
    }
    const float db = __half2float(*(const half*)b) * (0.5f + (float)(aux >> 28)) * 0.5f;
    return db * __low2float(a[g].ds) * (float)dot;
}

// As IQ3_XXS with a 512-entry codebook (the ninth index bit in qh), explicit
// sign bytes and an odd 4-bit scale.
__device__ __forceinline__ float vdot_iq3_s(const uint8_t* wr, int g, const block_q8a* a) {
    const uint8_t* b = wr + (size_t)(g / 8) * sizeof(block_iq3_s);   // 110 bytes: 2-byte aligned
    const int sub = g % 8;
    const uint32_t h = b[66 + sub];
    const uint32_t sg = load_u32_align2(b + 74 + 4 * sub);
    const int sc = (b[106 + sub / 2] >> (4 * (sub % 2))) & 0x0F;
    const int* aw = (const int*)a[g].qs;
    int dot = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t idx = *(const uint16_t*)(b + 2 + 8 * sub + 2 * l);
        const uint32_t s = sg >> (8 * l);
        const uint32_t g1 = iq3s_grid[(idx & 0xFF) | ((h << (8 - 2 * l)) & 256)];
        const uint32_t g2 = iq3s_grid[(idx >> 8) | ((h << (7 - 2 * l)) & 256)];
        dot = __dp4a(signed_codes4(g1, s), aw[2 * l], dot);
        dot = __dp4a(signed_codes4(g2, s >> 4), aw[2 * l + 1], dot);
    }
    return __half2float(*(const half*)b) * (float)(1 + 2 * sc) * __low2float(a[g].ds) * (float)dot;
}

// Codes 0..3 meaning -1..+2 under one scale per 64. The -1 goes into the
// bytes (q - 1 is a signed byte), so the integer dot is exact; folding it in
// as d * s_a afterwards, as the K-quants' minimum is, would cancel two large
// terms in half precision wherever most codes are 1 (a zero weight).
__device__ __forceinline__ float vdot_q2_0(const uint8_t* wr, int g, const block_q8a* a) {
    const block_q2_0* b = (const block_q2_0*)wr + g / 2;   // 18 bytes: 2-byte aligned
    const uint8_t* q = b->qs + (g & 1) * 8;
    const int* aw = (const int*)a[g].qs;
    int dot = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w2 = *(const uint16_t*)(q + 2 * i);   // elements 8i..8i+7
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            // One byte's four codes, spread to a byte each, low bits first.
            const uint32_t c = (w2 >> (8 * h)) & 0xFF;
            const uint32_t v = (c | (c << 6) | (c << 12) | (c << 18)) & 0x03030303u;
            dot = __dp4a((int)__vsub4(v, 0x01010101u), aw[2 * i + h], dot);
        }
    }
    return __half2float(b->d) * __low2float(a[g].ds) * (float)dot;
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

// The MoE decode kernels' integer types: the above plus the IQ types and
// Q2_0, which have no tiled or tensor-core kernel (the dense paths keep them
// on the f32 kernels).
#define AIRUN_FOR_EACH_MOE_Q8_TYPE(X) \
    AIRUN_FOR_EACH_Q8_TYPE(X)         \
    X(AIRUN_TYPE_IQ4_NL)              \
    X(AIRUN_TYPE_IQ4_XS)              \
    X(AIRUN_TYPE_IQ2_S)               \
    X(AIRUN_TYPE_IQ3_XXS)             \
    X(AIRUN_TYPE_IQ3_S)               \
    X(AIRUN_TYPE_Q2_0)

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
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_IQ4_NL, vdot_iq4_nl);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_IQ4_XS, vdot_iq4_xs);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_IQ2_S, vdot_iq2_s);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_IQ3_XXS, vdot_iq3_xxs);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_IQ3_S, vdot_iq3_s);
AIRUN_DEFINE_Q8DOT(AIRUN_TYPE_Q2_0, vdot_q2_0);
