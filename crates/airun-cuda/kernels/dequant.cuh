// Block decoders for every GGUF quantization type airun computes on.
//
// Each type exposes `dequant_sub32(block_ptr, sub, out[32])`, which decodes the
// 32 consecutive logical values starting at element `sub * 32` of a superblock.
// Fixing the granularity at 32 lets one lane of a warp own one group regardless
// of whether the type packs 32 or 256 elements per block, so the matmul kernels
// need exactly one code path.
//
// The layouts and the arithmetic follow ggml's reference `dequantize_row_*`
// implementations; the Rust CPU decoders in airun-gguf are the cross-check.

#pragma once

#include "airun.cuh"

// ---------------------------------------------------------------------------
// Block layouts. Sizes are asserted against the on-disk format.
// ---------------------------------------------------------------------------

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct { half d; int8_t  qs[32]; } block_q8_0;
typedef struct { half d; uint8_t qs[16]; } block_q4_0;
typedef struct { half d; half m; uint8_t qs[16]; } block_q4_1;
typedef struct { half d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;
typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1;

typedef struct { uint8_t scales[16]; uint8_t qs[64]; half d; half dmin; } block_q2_K;
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; } block_q3_K;
typedef struct { half d; half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[128]; } block_q4_K;
typedef struct { half d; half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;

typedef struct { half d; uint8_t qs[16]; } block_iq4_nl;
typedef struct { half d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } block_iq4_xs;
typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;
typedef struct { uint8_t d[4]; uint8_t qs[32]; } block_nvfp4;

static_assert(sizeof(block_q8_0)   == 34,  "block_q8_0 layout");
static_assert(sizeof(block_q4_0)   == 18,  "block_q4_0 layout");
static_assert(sizeof(block_q4_1)   == 20,  "block_q4_1 layout");
static_assert(sizeof(block_q5_0)   == 22,  "block_q5_0 layout");
static_assert(sizeof(block_q5_1)   == 24,  "block_q5_1 layout");
static_assert(sizeof(block_q2_K)   == 84,  "block_q2_K layout");
static_assert(sizeof(block_q3_K)   == 110, "block_q3_K layout");
static_assert(sizeof(block_q4_K)   == 144, "block_q4_K layout");
static_assert(sizeof(block_q5_K)   == 176, "block_q5_K layout");
static_assert(sizeof(block_q6_K)   == 210, "block_q6_K layout");
static_assert(sizeof(block_iq4_nl) == 18,  "block_iq4_nl layout");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs layout");
static_assert(sizeof(block_mxfp4)  == 17,  "block_mxfp4 layout");
static_assert(sizeof(block_nvfp4)  == 36,  "block_nvfp4 layout");

// Non-linear codebook shared by IQ4_NL and IQ4_XS.
__device__ __constant__ static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// E2M1 codebook for MXFP4, scaled by two so it pairs with the half-valued
// E8M0 exponent below (this is how ggml avoids a division in the inner loop).
__device__ __constant__ static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// ---------------------------------------------------------------------------
// Wide payload loads
// ---------------------------------------------------------------------------
//
// The decoders below read 32 quantized values at a time. Fetching those a byte
// at a time costs 32 load instructions per lane per group, and decode turns out
// to be bound by issuing them rather than by bandwidth or arithmetic -- the
// difference between ~32% and ~60% of the card's memory throughput. Reading the
// same bytes in 16- or 128-bit chunks and unpacking in registers produces
// bit-identical values; only the load width changes.
//
// How wide is fixed by the block size, because the tensor base is 256-byte
// aligned (cudaMalloc) and rows and blocks tile it with no padding
// (`row_bytes = n_blocks * bytes_per_block`), so every block inherits the
// alignment of `bytes_per_block` itself:
//
//   F32 128, F16/BF16 64, Q4_K 144, Q5_K 176   -> 16-byte aligned
//   NVFP4 36                                   -> 4-byte aligned
//   Q8_0 34, Q6_K 210, Q4_0 18, Q5_0 22, ...   -> 2-byte aligned only
//
// A decoder must not use a wider load than its own row above allows.

// Do NOT fetch weights with `__ldcs`/streaming hints. It looks right on paper --
// gigabytes of single-use data evicting an `x` that every warp re-reads -- but
// measured, it costs about a third of decode (75 -> 45 tok/s on a Q4_K 9B) and
// a third of prefill. Neighbouring lanes do reuse weight bytes: with Q4_K the
// even and odd group of a pair read the same 32 bytes, and normal caching is
// what turns the second one into a hit.

/// 32 bytes from a 16-byte-aligned address, as eight little-endian words.
__device__ __forceinline__ void load_32b_align16(const void* p, uint32_t* w) {
    const uint4 a = ((const uint4*)p)[0];
    const uint4 b = ((const uint4*)p)[1];
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
    w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
}

/// Same 32 bytes when only 2-byte alignment is guaranteed (a 34-byte block is
/// never reliably 4-byte aligned). Eight aligned words from the 4-byte boundary
/// at or below `p`, the trailing halfword when `p` is off by 2, and a funnel
/// shift to realign: nine loads instead of sixteen halfword ones, which kept
/// the decode matvec at 87% of DRAM bandwidth where llama.cpp's reaches 94%.
/// The shift is an operand, not a branch -- neighbouring Q8_0 groups alternate
/// alignment, so a branch would diverge every warp. Every byte read lies in
/// [p - 2, p + 32): the two below belong to the same or the previous block, and
/// the allocation itself is 4-byte aligned, so nothing precedes it.
__device__ __forceinline__ void load_32b_align2(const void* p, uint32_t* w) {
    const uintptr_t a = (uintptr_t)p;
    const uint32_t* base = (const uint32_t*)(a & ~(uintptr_t)3);
    const uint32_t sh = (uint32_t)(a & 3) * 8;   // 0 or 16
    uint32_t v[9];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = base[i];
    v[8] = sh ? (uint32_t)*(const uint16_t*)(base + 8) : 0u;
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = __funnelshift_r(v[i], v[i + 1], sh);
}

// ---------------------------------------------------------------------------
// Scale helpers
// ---------------------------------------------------------------------------

/// `d`, `dmin` and the twelve packed scale bytes are exactly the first sixteen
/// bytes of a Q4_K or Q5_K superblock, so one 16-byte load covers all of them.
/// The block's eight groups each decode separately and would otherwise repeat
/// the halfword and byte reads eight times over.
///
/// The scale unpacking is `get_scale_min_k4` from the Rust reference, done on
/// registers instead of memory: the twelve scale bytes are words y, z and w of
/// the header, and byte `sub & 3` of each is the only one a single group needs.
__device__ __forceinline__ void load_k_header(const void* p, int sub,
                                              float* d, float* dmin,
                                              uint8_t* sc, uint8_t* m) {
    const uint4 h = *(const uint4*)p;
    *d    = __half2float(__ushort_as_half((unsigned short)(h.x & 0xFFFF)));
    *dmin = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));

    const int k = sub & 3;
    const uint32_t s0 = (h.y >> (8 * k)) & 0xFF; // scales[k]
    const uint32_t s4 = (h.z >> (8 * k)) & 0xFF; // scales[k + 4]
    const uint32_t s8 = (h.w >> (8 * k)) & 0xFF; // scales[k + 8]

    if (sub < 4) {
        *sc = (uint8_t)(s0 & 63);
        *m  = (uint8_t)(s4 & 63);
    } else {
        *sc = (uint8_t)((s8 & 0xF) | ((s0 >> 6) << 4));
        *m  = (uint8_t)((s8 >> 4)  | ((s4 >> 6) << 4));
    }
}

// Q3_K packs sixteen 6-bit scales into 12 bytes; unpack them all at once.
__device__ __forceinline__ void unpack_q3k_scales(const uint8_t* src, int8_t* out) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, src, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = ((aux[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = ((aux[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);
    memcpy(out, aux, 16);
}

// E8M0: an 8-bit exponent-only float. ggml folds a factor of one half into the
// conversion so the integer codebook above can stay whole-numbered.
__device__ __forceinline__ float ue4m3_to_f32(uint8_t x) {
    // 0x7F is ggml's NaN encoding; it and a zero byte both mean "this run of 16
    // is silent", which is the same float either way.
    if (x == 0 || x == 0x7F) return 0.0f;

    const uint32_t exp = (x >> 3) & 0x0F;
    const uint32_t man = x & 0x07;
    // Subnormal: no implicit leading one, so the mantissa alone scales 2^-10.
    if (exp == 0) return (float)man * (1.0f / 1024.0f);

    const uint32_t bits = ((exp + 119) << 23) | (man << 20);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

__device__ __forceinline__ float e8m0_to_f32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        // Denormal region: 2^-127 and 2^-126 cannot be built by the shift below.
        bits = 0x00200000u << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

// ---------------------------------------------------------------------------
// Decoders: 32 consecutive values from group `sub` of a superblock.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void dq_f32(const void* p, int sub, float* out) {
    const float* src = (const float*)p + sub * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = src[i];
}

__device__ __forceinline__ void dq_f16(const void* p, int sub, float* out) {
    const half* src = (const half*)p + sub * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = __half2float(src[i]);
}

__device__ __forceinline__ void dq_bf16(const void* p, int sub, float* out) {
    // 64 bytes per group off a 16-byte-aligned base.
    const uint4* src = (const uint4*)((const uint16_t*)p + sub * 32);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint4 v = src[k];
        const uint32_t q[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[k * 8 + 2 * j + 0] = bf16_to_f32((uint16_t)(q[j] & 0xFFFF));
            out[k * 8 + 2 * j + 1] = bf16_to_f32((uint16_t)(q[j] >> 16));
        }
    }
}

// A Q8_0 block as loaded, before any arithmetic: the loads and the decode are
// separate steps so a caller can issue several blocks' loads before it spends
// any of them (see matvec_rows). load_32b_align2's words, unshifted.
struct q8_0_raw {
    uint32_t v[9];
    uint32_t sh;
    unsigned short d;
};

__device__ __forceinline__ void q8_0_load(const void* p, int sub, q8_0_raw& r) {
    const block_q8_0* b = (const block_q8_0*)p + sub;
    r.d = __half_as_ushort(b->d);
    const uintptr_t a = (uintptr_t)b->qs;   // 34-byte block: 2-byte alignment only
    const uint32_t* base = (const uint32_t*)(a & ~(uintptr_t)3);
    r.sh = (uint32_t)(a & 3) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) r.v[i] = base[i];
    r.v[8] = r.sh ? (uint32_t)*(const uint16_t*)(base + 8) : 0u;
}

__device__ __forceinline__ void q8_0_decode(const q8_0_raw& r, float* out) {
    const float d = __half2float(__ushort_as_half(r.d));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t v = __funnelshift_r(r.v[i], r.v[i + 1], r.sh);
        out[4 * i + 0] = d * (float)(int8_t)(v      );
        out[4 * i + 1] = d * (float)(int8_t)(v >>  8);
        out[4 * i + 2] = d * (float)(int8_t)(v >> 16);
        out[4 * i + 3] = d * (float)(int8_t)(v >> 24);
    }
}

__device__ __forceinline__ void dq_q8_0(const void* p, int sub, float* out) {
    q8_0_raw r;
    q8_0_load(p, sub, r);
    q8_0_decode(r, out);
}

__device__ __forceinline__ void dq_q4_0(const void* p, int sub, float* out) {
    const block_q4_0* b = (const block_q4_0*)p + sub;
    const float d = __half2float(b->d);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i]      = d * (float)((int)(b->qs[i] & 0x0F) - 8);
        out[i + 16] = d * (float)((int)(b->qs[i] >> 4)   - 8);
    }
}

__device__ __forceinline__ void dq_q4_1(const void* p, int sub, float* out) {
    const block_q4_1* b = (const block_q4_1*)p + sub;
    const float d = __half2float(b->d);
    const float m = __half2float(b->m);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i]      = d * (float)(b->qs[i] & 0x0F) + m;
        out[i + 16] = d * (float)(b->qs[i] >> 4)   + m;
    }
}

__device__ __forceinline__ void dq_q5_0(const void* p, int sub, float* out) {
    const block_q5_0* b = (const block_q5_0*)p + sub;
    const float d = __half2float(b->d);
    uint32_t qh;
    memcpy(&qh, b->qh, sizeof(qh));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t hi0 = ((qh >> (i + 0))  << 4) & 0x10;
        const uint8_t hi1 = ((qh >> (i + 12)) & 0x10);
        out[i]      = d * (float)((int)((b->qs[i] & 0x0F) | hi0) - 16);
        out[i + 16] = d * (float)((int)((b->qs[i] >> 4)   | hi1) - 16);
    }
}

__device__ __forceinline__ void dq_q5_1(const void* p, int sub, float* out) {
    const block_q5_1* b = (const block_q5_1*)p + sub;
    const float d = __half2float(b->d);
    const float m = __half2float(b->m);
    uint32_t qh;
    memcpy(&qh, b->qh, sizeof(qh));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t hi0 = ((qh >> (i + 0))  << 4) & 0x10;
        const uint8_t hi1 = ((qh >> (i + 12)) & 0x10);
        out[i]      = d * (float)((b->qs[i] & 0x0F) | hi0) + m;
        out[i + 16] = d * (float)((b->qs[i] >> 4)   | hi1) + m;
    }
}

// --- K-quants: `sub` selects one of eight 32-element groups in the superblock.

__device__ __forceinline__ void dq_q4_K(const void* p, int sub, float* out) {
    const block_q4_K* b = (const block_q4_K*)p;

    float d, dmin;
    uint8_t sc, m;
    load_k_header(p, sub, &d, &dmin, &sc, &m);
    const float dl = d * (float)sc;
    const float ml = dmin * (float)m;

    // Groups pair up: the even group takes low nibbles, the odd one the highs
    // of the same 32 bytes.
    uint32_t w[8];
    load_32b_align16(b->qs + (sub / 2) * 32, w); // 144-byte block, qs at +16
    const int shift = (sub & 1) ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t v = w[i] >> shift;
        out[4 * i + 0] = dl * (float)((v      ) & 0x0F) - ml;
        out[4 * i + 1] = dl * (float)((v >>  8) & 0x0F) - ml;
        out[4 * i + 2] = dl * (float)((v >> 16) & 0x0F) - ml;
        out[4 * i + 3] = dl * (float)((v >> 24) & 0x0F) - ml;
    }
}

__device__ __forceinline__ void dq_q5_K(const void* p, int sub, float* out) {
    const block_q5_K* b = (const block_q5_K*)p;

    float d, dmin;
    uint8_t sc, m;
    load_k_header(p, sub, &d, &dmin, &sc, &m);
    const float dl = d * (float)sc;
    const float ml = dmin * (float)m;

    // 176-byte block: qs sits at +48 and qh at +16, both 16-byte aligned.
    uint32_t wq[8], wh[8];
    load_32b_align16(b->qs + (sub / 2) * 32, wq);
    load_32b_align16(b->qh, wh);

    const int shift = (sub & 1) ? 4 : 0;
    const uint32_t bit = 1u << sub; // fifth bit for this group, within each byte
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t v = wq[i] >> shift;
        const uint32_t h = wh[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int lo = (v >> (8 * j)) & 0x0F;
            const int hi = ((h >> (8 * j)) & bit) ? 16 : 0;
            out[4 * i + j] = dl * (float)(lo + hi) - ml;
        }
    }
}

__device__ __forceinline__ void dq_q6_K(const void* p, int sub, float* out) {
    const block_q6_K* b = (const block_q6_K*)p;
    const float d = __half2float(b->d);

    // The superblock splits into two 128-value halves; within a half, four
    // quarters interleave low/high nibbles with a 2-bit extension.
    const int half    = sub / 4;
    const int quarter = sub % 4;

    const uint8_t* ql = b->ql + half * 64 + (quarter & 1) * 32;
    const uint8_t* qh = b->qh + half * 32;
    const int8_t*  sc = b->scales + half * 8 + quarter * 2;
    const int shift   = quarter * 2;
    const bool high   = quarter >= 2;

    // 210-byte block, so 2-byte alignment is all this one gets.
    uint32_t wl[8], wh[8];
    load_32b_align2(ql, wl);
    load_32b_align2(qh, wh);

    const int loshift = high ? 4 : 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t v = wl[i] >> loshift;
        const uint32_t h = wh[i] >> shift;
        // out index is 4*i + j, so the scale index 4*i/16 is just i/4.
        const float s = d * (float)sc[i / 4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int lo = (v >> (8 * j)) & 0x0F;
            const int hi = ((h >> (8 * j)) & 3) << 4;
            out[4 * i + j] = s * (float)((lo | hi) - 32);
        }
    }
}

__device__ __forceinline__ void dq_q2_K(const void* p, int sub, float* out) {
    const block_q2_K* b = (const block_q2_K*)p;
    const float d    = __half2float(b->d);
    const float dmin = __half2float(b->dmin);

    // Two 16-value units per group.
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int unit  = sub * 2 + u;
        const int half  = unit / 8;
        const int within = unit % 8;
        const int j     = within / 2;
        const int part  = within % 2;

        const uint8_t* q = b->qs + half * 32 + part * 16;
        const int shift  = j * 2;
        const uint8_t s  = b->scales[unit];
        const float dl   = d * (float)(s & 0x0F);
        const float ml   = dmin * (float)(s >> 4);

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[u * 16 + i] = dl * (float)((q[i] >> shift) & 3) - ml;
        }
    }
}

__device__ __forceinline__ void dq_q3_K(const void* p, int sub, float* out) {
    const block_q3_K* b = (const block_q3_K*)p;
    const float d = __half2float(b->d);

    int8_t scales[16];
    unpack_q3k_scales(b->scales, scales);

#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int unit   = sub * 2 + u;
        const int half   = unit / 8;
        const int within = unit % 8;
        const int j      = within / 2;
        const int part   = within % 2;

        const uint8_t* q  = b->qs + half * 32 + part * 16;
        const uint8_t* hm = b->hmask + part * 16;
        const int shift   = j * 2;
        // The high-bit mask advances across the whole superblock, not per half.
        const uint8_t m   = (uint8_t)(1u << (half * 4 + j));
        const float dl    = d * (float)(scales[unit] - 32);

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int lo = (q[i] >> shift) & 3;
            const int hi = (hm[i] & m) ? 0 : 4; // note: bit set means *no* offset
            out[u * 16 + i] = dl * (float)(lo - hi);
        }
    }
}

__device__ __forceinline__ void dq_iq4_nl(const void* p, int sub, float* out) {
    const block_iq4_nl* b = (const block_iq4_nl*)p + sub;
    const float d = __half2float(b->d);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i]      = d * (float)kvalues_iq4nl[b->qs[i] & 0x0F];
        out[i + 16] = d * (float)kvalues_iq4nl[b->qs[i] >> 4];
    }
}

__device__ __forceinline__ void dq_iq4_xs(const void* p, int sub, float* out) {
    const block_iq4_xs* b = (const block_iq4_xs*)p;
    const float d = __half2float(b->d);

    // 6-bit scale split across a nibble array and a packed 2-bit high field.
    const int ls = ((b->scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0F)
                 | (((b->scales_h >> (2 * sub)) & 3) << 4);
    const float dl = d * (float)(ls - 32);

    const uint8_t* qs = b->qs + sub * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i]      = dl * (float)kvalues_iq4nl[qs[i] & 0x0F];
        out[i + 16] = dl * (float)kvalues_iq4nl[qs[i] >> 4];
    }
}

__device__ __forceinline__ void dq_mxfp4(const void* p, int sub, float* out) {
    const block_mxfp4* b = (const block_mxfp4*)p + sub;
    const float d = e8m0_to_f32_half(b->e);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i]      = d * (float)kvalues_mxfp4[b->qs[i] & 0x0F];
        out[i + 16] = d * (float)kvalues_mxfp4[b->qs[i] >> 4];
    }
}

// NVFP4 packs 64 elements as four independently scaled runs of 16, so one
// 32-element group spans two of them. The block is 36 bytes and therefore
// 4-byte aligned, which is exactly wide enough to pull each run's payload in
// two word loads instead of eight byte loads.
__device__ __forceinline__ void dq_nvfp4(const void* p, int sub, float* out) {
    const block_nvfp4* b = (const block_nvfp4*)p;

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int run = sub * 2 + half;
        const float d = ue4m3_to_f32(b->d[run]);

        const uint32_t* q = (const uint32_t*)(b->qs + run * 8);
        const uint32_t w0 = q[0];
        const uint32_t w1 = q[1];

        float* o = out + half * 16;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t byte0 = (w0 >> (8 * i)) & 0xFF;
            const uint32_t byte1 = (w1 >> (8 * i)) & 0xFF;
            o[i]         = d * (float)kvalues_mxfp4[byte0 & 0x0F];
            o[i + 8]     = d * (float)kvalues_mxfp4[byte0 >> 4];
            o[i + 4]     = d * (float)kvalues_mxfp4[byte1 & 0x0F];
            o[i + 12]    = d * (float)kvalues_mxfp4[byte1 >> 4];
        }
    }
}

// ---------------------------------------------------------------------------
// Type dispatch
// ---------------------------------------------------------------------------
//
// Type ids match the GGUF on-disk values, mirroring airun-gguf's table.

#define AIRUN_TYPE_F32     0
#define AIRUN_TYPE_F16     1
#define AIRUN_TYPE_Q4_0    2
#define AIRUN_TYPE_Q4_1    3
#define AIRUN_TYPE_Q5_0    6
#define AIRUN_TYPE_Q5_1    7
#define AIRUN_TYPE_Q8_0    8
#define AIRUN_TYPE_Q2_K   10
#define AIRUN_TYPE_Q3_K   11
#define AIRUN_TYPE_Q4_K   12
#define AIRUN_TYPE_Q5_K   13
#define AIRUN_TYPE_Q6_K   14
#define AIRUN_TYPE_IQ4_NL 20
#define AIRUN_TYPE_IQ4_XS 23
#define AIRUN_TYPE_BF16   30
#define AIRUN_TYPE_MXFP4  39
#define AIRUN_TYPE_NVFP4  40

// Compile-time traits so kernels can be templated on the type id: each
// instantiation inlines exactly one decoder with no branching in the inner loop.
template <int TYPE> struct qtraits;

#define AIRUN_DEFINE_TRAITS(TYPE, BLOCK_ELEMS, BLOCK_BYTES, FN)              \
    template <> struct qtraits<TYPE> {                                       \
        static constexpr int elems_per_block = BLOCK_ELEMS;                  \
        static constexpr int bytes_per_block = BLOCK_BYTES;                  \
        static constexpr int subs_per_block  = BLOCK_ELEMS / 32;             \
        __device__ static __forceinline__ void decode(const void* p, int sub, float* out) { \
            FN(p, sub, out);                                                 \
        }                                                                    \
    }

AIRUN_DEFINE_TRAITS(AIRUN_TYPE_F32,    32,   128, dq_f32);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_F16,    32,    64, dq_f16);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_BF16,   32,    64, dq_bf16);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q4_0,   32,    18, dq_q4_0);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q4_1,   32,    20, dq_q4_1);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q5_0,   32,    22, dq_q5_0);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q5_1,   32,    24, dq_q5_1);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q8_0,   32,    34, dq_q8_0);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_IQ4_NL, 32,    18, dq_iq4_nl);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_MXFP4,  32,    17, dq_mxfp4);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_NVFP4,  64,    36, dq_nvfp4);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q2_K,  256,    84, dq_q2_K);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q3_K,  256,   110, dq_q3_K);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q4_K,  256,   144, dq_q4_K);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q5_K,  256,   176, dq_q5_K);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_Q6_K,  256,   210, dq_q6_K);
AIRUN_DEFINE_TRAITS(AIRUN_TYPE_IQ4_XS,256,   136, dq_iq4_xs);

// Decodes group `group` (32 elements) of a row, where `row` points at the start
// of the row's packed bytes.
template <int TYPE>
__device__ __forceinline__ void decode_group(const uint8_t* row, int group, float* out) {
    using T = qtraits<TYPE>;
    const int block = group / T::subs_per_block;
    const int sub   = group % T::subs_per_block;
    T::decode(row + (size_t)block * T::bytes_per_block, sub, out);
}
