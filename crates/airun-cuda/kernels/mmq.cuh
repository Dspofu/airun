// Integer tensor-core matmul for prefill ("MMQ", after llama.cpp's name for
// the same idea).
//
// The fp16 WMMA kernel converts every weight and activation to half and runs
// the tensor cores at their fp16-with-fp32-accumulate rate, which on consumer
// Ampere is a quarter of the int8 rate; it was 69% of a Qwen3-4B prefill. Here
// the weights are decoded to their integer codes (exactly the codes vecdotq.cuh
// feeds __dp4a), the activations arrive as block_q8a, and mma.sync multiplies
// 8-bit integers with 32-bit accumulation. Scales are applied per 32-group in
// f32, so the arithmetic is the dp4a path's, reordered: selftest holds it to
// the same backward-error bound against the CPU mirror.
//
// What bounds this kernel is staging, not the MMAs: removing the MMAs did not
// change its time, removing the weight decode made it 40% faster. Measured and
// rejected: a 128-token tile (no change -- decode volume is not the cost), and a
// thread per 4-byte word for coalescing (half the speed: five times the
// instructions, while the 32 bytes a per-group decode touches share a sector L1
// serves anyway), and a register software pipeline (184 registers, one block
// per SM; capped at 128 it spilled -- slower both ways).

#pragma once

#include "vecdotq.cuh"

// Per weight type, the decode in two steps:
//   load(row, g, raw)          the group's bytes, global loads only
//   decode(raw, g, q, s0,s1,m) 32 integer codes (8 words; the codes vecdotq.cuh
//                              feeds dp4a) and w_i = s_half * q_i - m
template <int TYPE> struct mmq_decode;

/// Scale and min of group `sub` from a Q4_K/Q5_K header already in registers;
/// the same unpacking as load_k_header.
__device__ __forceinline__ void k_header_regs(const uint4 h, int sub, float& s, float& m) {
    const float d    = __half2float(__ushort_as_half((unsigned short)(h.x & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
    const int k = sub & 3;
    const uint32_t a = (h.y >> (8 * k)) & 0xFF;
    const uint32_t b = (h.z >> (8 * k)) & 0xFF;
    const uint32_t c = (h.w >> (8 * k)) & 0xFF;
    const uint32_t sc = sub < 4 ? (a & 63) : ((c & 0xF) | ((a >> 6) << 4));
    const uint32_t mn = sub < 4 ? (b & 63) : ((c >> 4) | ((b >> 6) << 4));
    s = d * (float)sc;
    m = dmin * (float)mn;
}

template <> struct mmq_decode<AIRUN_TYPE_Q8_0> {
    static constexpr bool SPLIT = false, HAS_MIN = false;
    // 34-byte blocks are only 2-byte aligned. Nine aligned words cover the
    // group's d and 32 codes whichever way it falls, and a funnel shift
    // realigns them: half the loads of 17 halfwords, which made Q8_0's decode
    // pass the biggest source of shared-memory instructions in the kernel.
    struct Raw { uint32_t w[9]; uint32_t sh; };
    __device__ static __forceinline__ void load(const uint8_t* wr, int g, Raw& r) {
        const uint8_t* qs = wr + (size_t)g * 34 + 2;
        const uintptr_t base = ((uintptr_t)qs - 2) & ~(uintptr_t)3;
        const uint32_t* p = (const uint32_t*)base;
#pragma unroll
        for (int i = 0; i < 9; ++i) r.w[i] = p[i];
        r.sh = 8u * (uint32_t)((uintptr_t)qs - base);   // 16 or 32
    }
    __device__ static __forceinline__ void decode(const Raw& r, int, uint32_t q[8],
                                                  float& s0, float& s1, float& m) {
#pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = __funnelshift_rc(r.w[i], r.w[i + 1], r.sh);
        s0 = s1 = __half2float(__ushort_as_half((unsigned short)(r.w[0] >> (r.sh - 16))));
        m = 0.0f;
    }
};
template <> struct mmq_decode<AIRUN_TYPE_Q4_K> {
    static constexpr bool SPLIT = false, HAS_MIN = true;
    struct Raw { uint4 hdr, q0, q1; };
    __device__ static __forceinline__ void load(const uint8_t* wr, int g, Raw& r) {
        const block_q4_K* b = (const block_q4_K*)(wr + (size_t)(g / 8) * sizeof(block_q4_K));
        const uint4* qs = (const uint4*)(b->qs + ((g % 8) / 2) * 32);
        r.hdr = *(const uint4*)b;
        r.q0 = qs[0];
        r.q1 = qs[1];
    }
    __device__ static __forceinline__ void decode(const Raw& r, int g, uint32_t q[8],
                                                  float& s0, float& s1, float& m) {
        const int sub = g % 8;
        const int shift = (sub & 1) ? 4 : 0;
        const uint32_t w[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = (w[i] >> shift) & 0x0F0F0F0Fu;
        k_header_regs(r.hdr, sub, s0, m);
        s1 = s0;
    }
};

template <> struct mmq_decode<AIRUN_TYPE_Q5_K> {
    static constexpr bool SPLIT = false, HAS_MIN = true;
    struct Raw { uint4 hdr, q0, q1, h0, h1; };
    __device__ static __forceinline__ void load(const uint8_t* wr, int g, Raw& r) {
        const block_q5_K* b = (const block_q5_K*)(wr + (size_t)(g / 8) * sizeof(block_q5_K));
        const uint4* qs = (const uint4*)(b->qs + ((g % 8) / 2) * 32);
        const uint4* qh = (const uint4*)b->qh;
        r.hdr = *(const uint4*)b;
        r.q0 = qs[0];
        r.q1 = qs[1];
        r.h0 = qh[0];
        r.h1 = qh[1];
    }
    __device__ static __forceinline__ void decode(const Raw& r, int g, uint32_t q[8],
                                                  float& s0, float& s1, float& m) {
        const int sub = g % 8;
        const int shift = (sub & 1) ? 4 : 0;
        const uint32_t w[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
        const uint32_t h[8] = {r.h0.x, r.h0.y, r.h0.z, r.h0.w, r.h1.x, r.h1.y, r.h1.z, r.h1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = ((w[i] >> shift) & 0x0F0F0F0Fu) | (((h[i] >> sub) & 0x01010101u) << 4);
        }
        k_header_regs(r.hdr, sub, s0, m);
        s1 = s0;
    }
};

// Q6_K changes its int8 scale at the group's midpoint, so it needs two k16
// products per group instead of one k32.
template <> struct mmq_decode<AIRUN_TYPE_Q6_K> {
    static constexpr bool SPLIT = true, HAS_MIN = false;
    struct Raw { uint16_t l[16], h[16], sc, d; };   // 210-byte blocks: 2-byte alignment
    __device__ static __forceinline__ void load(const uint8_t* wr, int g, Raw& r) {
        const int sub = g % 8;
        const block_q6_K* b = (const block_q6_K*)(wr + (size_t)(g / 8) * sizeof(block_q6_K));
        const int half_ = sub / 4;
        const int quarter = sub % 4;
        const uint16_t* ql = (const uint16_t*)(b->ql + half_ * 64 + (quarter & 1) * 32);
        const uint16_t* qh = (const uint16_t*)(b->qh + half_ * 32);
#pragma unroll
        for (int i = 0; i < 16; ++i) { r.l[i] = ql[i]; r.h[i] = qh[i]; }
        r.sc = *(const uint16_t*)(b->scales + half_ * 8 + quarter * 2);
        r.d = *(const uint16_t*)&b->d;
    }
    __device__ static __forceinline__ void decode(const Raw& r, int g, uint32_t q[8],
                                                  float& s0, float& s1, float& m) {
        const int quarter = (g % 8) % 4;
        const int shift = quarter * 2;
        const int loshift = (quarter >= 2) ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t wl = (uint32_t)r.l[2 * i] | ((uint32_t)r.l[2 * i + 1] << 16);
            const uint32_t wh = (uint32_t)r.h[2 * i] | ((uint32_t)r.h[2 * i + 1] << 16);
            const uint32_t lo = (wl >> loshift) & 0x0F0F0F0Fu;
            const uint32_t hi = ((wh >> shift) & 0x03030303u) << 4;
            q[i] = __vsub4(lo | hi, 0x20202020u);
        }
        const float d = __half2float(__ushort_as_half(r.d));
        s0 = d * (float)(int8_t)(r.sc & 0xFF);
        s1 = d * (float)(int8_t)(r.sc >> 8);
        m = 0.0f;
    }
};
// Integer results leave the MMA already as floats: the accumulator starts at
// the bit pattern of 1.5 * 2^23, whose ulp is 1, so adding an integer v to it
// gives the float 1.5 * 2^23 + v exactly for |v| < 2^22 -- far above the 516129
// a k32 int8 product can reach. One full-rate subtraction then replaces I2F,
// which consumer Ampere issues at 16 per clock per SM: ncu counted four per
// MMA per thread, and 20x as many FP32 instructions as MMAs overall.
#define MMQ_MAGIC_I 0x4B400000
__device__ __forceinline__ float mmq_f(int biased) { return __int_as_float(biased) - 12582912.0f; }

// The accumulator input is a separate operand, so passing the MMQ_MAGIC_I
// bias (see mmq_f) there costs nothing, where initializing `c` took four MOVs
// per MMA -- 8% of the kernel's samples.
__device__ __forceinline__ void mma_s8_k32(int c[4], const int a[4], int b0, int b1, int bias) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(bias));
#endif
}

__device__ __forceinline__ void mma_s8_k16(int c[4], int a0, int a1, int b0, int bias) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(bias));
#endif
}
// Block tile: MMQ_TR rows x TC tokens (a template parameter: 64 or 128),
// staged MMQ_KG groups (of 32) at a time. Eight warps split it 2 x 4, so each
// warp owns 64 rows x TC/4 tokens: four m16 tiles by TC/32 n8 tiles.
//
// Why these shapes, all measured with ncu on a Qwen3-4B prefill:
// - The warp tile is tall because the shared-memory instruction queue was the
//   top stall: 64 rows share each activation fragment four ways, and A comes
//   in by ldmatrix, ~1.75 shared loads per MMA against ~3.5 for 32 x 32.
// - 128 tokens per block for long prompts: once staging was asynchronous the
//   warps sat at the chunk barriers waiting for data, because a 64-token chunk
//   is too little work to cover the load of the next one. Twice the tokens is
//   twice the MMAs per staged weight byte and per barrier. Short prompts keep
//   64 so the grid still fills the GPU.
#define MMQ_TR 128
#define MMQ_KG 4
#define MMQ_WR 64
#define MMQ_MT (MMQ_WR / 16)
#define MMQ_WTASKS (MMQ_TR * MMQ_KG / 256)   // weight groups each thread decodes
// The int8 weight tile has 128-byte rows: MMQ_KG groups of 32 codes. Its
// 16-byte chunks are XOR-swizzled by row, so the eight rows an ldmatrix phase
// reads land in eight different chunks of the bank space without padding --
// the 16 bytes a row of padding cost per row are what let the 128-token
// asynchronous kernel fit in 99 KiB.
#define MMQ_LDW (MMQ_KG * 32)
__device__ __forceinline__ int mmq_swz(int r, int chunk) { return ((chunk ^ (r & 7)) << 4); }
// A staged activation row is MMQ_KG raw block_q8a of 36 bytes: 144 = 36 words,
// so the eight token rows a fragment load touches start 4 banks apart and the
// four lanes per row fill the gaps.
#define MMQ_XLD (MMQ_KG * 36)

// Scales, laid out for the fragment that reads them. A thread's C fragment
// covers rows r and r+8 and tokens t and t+1, so each pair is one vector load:
//   wsc[gg][rp] = (s_lo(r), s_lo(r+8), s_hi(r), s_hi(r+8))   rp: row pair
//   wmn[gg][rp] = (m(r), m(r+8))
//   xds[gg][t]  = (d, d*sum) of token t, so [t] and [t+1] are one float4
template <int TC>
struct MmqScales {
    float4 wsc[MMQ_KG][MMQ_TR / 2];
    float2 wmn[MMQ_KG][MMQ_TR / 2];
    float2 xds[MMQ_KG][TC];
};
__device__ __forceinline__ int mmq_row_pair(int r) { return (r / 16) * 8 + r % 8; }

__device__ __forceinline__ void ldsm_x4(int r[4], const void* p) {
#if __CUDA_ARCH__ >= 800
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}

// The MMAs of one staged chunk. `xs` holds each token's MMQ_KG activation
// blocks exactly as block_q8a lays them out: 32 codes, then (d, d*sum).
// HAS_MIN is false for types with no offset term (Q8_0, Q6_K), which then skip
// the multiply-add by the activation sum.
template <int TC, bool SPLIT, bool HAS_MIN>
__device__ __forceinline__ void mmq_chunk(
    const int8_t* ws, const MmqScales<TC>& S, const int8_t (*xs)[MMQ_XLD],
    float acc[MMQ_MT][TC / 32][4], int wr, int wc, int lane) {
    constexpr int NT = TC / 32;
    const int gid = lane >> 2;
    const int tig = lane & 3;
#pragma unroll
    for (int gg = 0; gg < MMQ_KG; ++gg) {
        int a[MMQ_MT][4];
        float4 sc[MMQ_MT];
        float2 mn[MMQ_MT];
#pragma unroll
        for (int mt = 0; mt < MMQ_MT; ++mt) {
            // ldmatrix: lanes 0-7 address rows 0-7, 8-15 rows 8-15, 16-31 the
            // same rows 16 bytes on -- exactly the four registers of A.
            const int r = wr + mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldsm_x4(a[mt], ws + r * MMQ_LDW + mmq_swz(r, 2 * gg + (lane >> 4)));
            const int rp = mmq_row_pair(wr + mt * 16) + gid;
            sc[mt] = S.wsc[gg][rp];
            if constexpr (HAS_MIN) mn[mt] = S.wmn[gg][rp];
        }
        // A software pipeline one MMA deep: MMA i is issued, then the scaling
        // of MMA i-1 runs while it computes. Consuming each result at once
        // stalled on its latency ("wait"); issuing the group's MMAs in a burst
        // and scaling afterwards left the tensor pipe busy 34% of the time
        // with "math pipe throttle" the top stall -- the MMAs queued up behind
        // each other, then the FP32 epilogue ran with the tensor pipe idle.
        constexpr int PARTS = SPLIT ? 2 : 1;
        int b[NT][2];
        float4 x[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int8_t* pb = &xs[wc + nt * 8 + gid][gg * 36 + tig * 4];
            b[nt][0] = *(const int*)pb;
            b[nt][1] = *(const int*)(pb + 16);
            // (d, d*sum) of tokens t + tig*2 and +1, which this thread's
            // outputs cover.
            x[nt] = *(const float4*)&S.xds[gg][wc + nt * 8 + tig * 2];
        }
        auto issue = [&](int i, int (&c)[PARTS][4]) {
            const int nt = i / MMQ_MT, mt = i % MMQ_MT;
            if constexpr (SPLIT) {
                mma_s8_k16(c[0], a[mt][0], a[mt][1], b[nt][0], MMQ_MAGIC_I);
                mma_s8_k16(c[PARTS - 1], a[mt][2], a[mt][3], b[nt][1], MMQ_MAGIC_I);
            } else {
                mma_s8_k32(c[0], a[mt], b[nt][0], b[nt][1], MMQ_MAGIC_I);
            }
        };
        auto scale = [&](int i, const int (&c)[PARTS][4]) {
            const int nt = i / MMQ_MT, mt = i % MMQ_MT;
            float* o = acc[mt][nt];
            const float4 xx = x[nt];
            const int* lo = c[0];
            if constexpr (SPLIT) {
                const int* hi = c[PARTS - 1];
                o[0] = fmaf(xx.x, fmaf(sc[mt].z, mmq_f(hi[0]), sc[mt].x * mmq_f(lo[0])), o[0]);
                o[1] = fmaf(xx.z, fmaf(sc[mt].z, mmq_f(hi[1]), sc[mt].x * mmq_f(lo[1])), o[1]);
                o[2] = fmaf(xx.x, fmaf(sc[mt].w, mmq_f(hi[2]), sc[mt].y * mmq_f(lo[2])), o[2]);
                o[3] = fmaf(xx.z, fmaf(sc[mt].w, mmq_f(hi[3]), sc[mt].y * mmq_f(lo[3])), o[3]);
            } else {
                o[0] = fmaf(xx.x * sc[mt].x, mmq_f(lo[0]), o[0]);
                o[1] = fmaf(xx.z * sc[mt].x, mmq_f(lo[1]), o[1]);
                o[2] = fmaf(xx.x * sc[mt].y, mmq_f(lo[2]), o[2]);
                o[3] = fmaf(xx.z * sc[mt].y, mmq_f(lo[3]), o[3]);
                if constexpr (HAS_MIN) {
                    o[0] = fmaf(-mn[mt].x, xx.y, o[0]);
                    o[1] = fmaf(-mn[mt].x, xx.w, o[1]);
                    o[2] = fmaf(-mn[mt].y, xx.y, o[2]);
                    o[3] = fmaf(-mn[mt].y, xx.w, o[3]);
                }
            }
        };
        constexpr int N = NT * MMQ_MT;
        int c[2][PARTS][4];
        issue(0, c[0]);
#pragma unroll
        for (int i = 1; i < N; ++i) {
            issue(i, c[i & 1]);
            scale(i - 1, c[(i - 1) & 1]);
        }
        scale(N - 1, c[(N - 1) & 1]);
    }
}

// Decodes MMQ_TR rows x MMQ_KG groups into the int8 tile and its scales.
// `src(r)` is where row r's bytes are (global memory, or the raw copy in
// shared memory) and `rel0` the group D::load counts from there.
template <typename D, int TC, typename Src>
__device__ __forceinline__ void mmq_decode_tile(
    int8_t* ws, MmqScales<TC>& S, Src src, int g0, int rel0,
    int rows_valid, int groups_valid, int tid, bool glu = false) {
#pragma unroll
    for (int i = 0; i < MMQ_WTASKS; ++i) {
        const int task = tid + i * 256;
        const int r = task / MMQ_KG;
        const int gg = task % MMQ_KG;
        uint32_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float s0 = 0.0f, s1 = 0.0f, m = 0.0f;
        // In a gate/up tile rows 64..127 repeat rows 0..63 of the other matrix.
        if ((glu ? (r & 63) : r) < rows_valid && gg < groups_valid) {
            typename D::Raw raw;
            D::load(src(r), g0 + gg - rel0, raw);
            D::decode(raw, g0 + gg, q, s0, s1, m);
        }
        *(uint4*)(ws + r * MMQ_LDW + mmq_swz(r, 2 * gg)) = make_uint4(q[0], q[1], q[2], q[3]);
        *(uint4*)(ws + r * MMQ_LDW + mmq_swz(r, 2 * gg + 1)) = make_uint4(q[4], q[5], q[6], q[7]);
        const int rp = mmq_row_pair(r), half = (r % 16) / 8;
        float* sc = (float*)&S.wsc[gg][rp];
        sc[half] = s0;
        sc[2 + half] = s1;
        ((float*)&S.wmn[gg][rp])[half] = m;
    }
}

// (d, d*sum) of every staged activation block as floats, for mmq_chunk's
// vector loads. Blocks past the edge were zeroed, so they give zeros.
template <int TC>
__device__ __forceinline__ void mmq_stage_xds(const int8_t (*xs)[MMQ_XLD], MmqScales<TC>& S, int tid) {
    for (int task = tid; task < TC * MMQ_KG; task += 256) {
        const int t = task / MMQ_KG;
        const int gg = task % MMQ_KG;
        S.xds[gg][t] = __half22float2(*(const half2*)&xs[t][gg * 36 + 32]);
    }
}

template <int TC>
__device__ __forceinline__ void mmq_store(float* y, float acc[MMQ_MT][TC / 32][4], int row0,
                                          int tok0, int wr, int wc, int lane, int n_rows,
                                          int n_tokens) {
    const int gid = lane >> 2;
    const int tig = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MMQ_MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < TC / 32; ++nt) {
            const int r = row0 + wr + mt * 16 + gid;
            const int t = tok0 + wc + nt * 8 + tig * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = r + (e >= 2 ? 8 : 0);
                const int tok = t + (e & 1);
                if (row < n_rows && tok < n_tokens) y[(size_t)tok * n_rows + row] = acc[mt][nt][e];
            }
        }
    }
}

// Split-K: blockIdx.z takes a contiguous share of the K chunks. With one split
// the block writes y; with several each writes its own partial slice, summed
// afterwards in a fixed order by mmq_reduce_kernel -- deterministic, unlike
// atomics, so a prefill gives the same logits every run.
__device__ __forceinline__ void mmq_split_range(int n_groups, int& begin, int& end) {
    const int chunks = (n_groups + MMQ_KG - 1) / MMQ_KG;
    begin = (int)((int64_t)chunks * blockIdx.z / gridDim.z) * MMQ_KG;
    end = min(n_groups, (int)((int64_t)chunks * (blockIdx.z + 1) / gridDim.z) * MMQ_KG);
}
__device__ __forceinline__ float* mmq_out(float* y, float* partial, int n_rows, int n_tokens) {
    return gridDim.z == 1 ? y : partial + (size_t)blockIdx.z * n_rows * n_tokens;
}

__global__ void mmq_reduce_kernel(float* __restrict__ y, const float* __restrict__ partial,
                                  const int64_t n, const int splits) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = partial[i];
    for (int z = 1; z < splits; ++z) v += partial[(size_t)z * n + i];
    y[i] = v;
}

// --- synchronous kernel: any alignment, any group count ---------------------
template <int TC>
constexpr int mmq_sync_smem() {
    return MMQ_TR * MMQ_LDW + TC * MMQ_XLD + (int)sizeof(MmqScales<TC>);
}

template <int TYPE, int TC>
// Two blocks per SM: at the compiler's natural register count only one fits,
// eight warps, too few to cover the staging latency.
__global__ __launch_bounds__(256, 2) void matvec_mmq_kernel(
    const uint8_t* __restrict__ w,
    const block_q8a* __restrict__ xq,
    float* __restrict__ y,
    float* __restrict__ partial,
    const int K,
    const int n_rows,
    const int n_tokens,
    const size_t row_bytes) {

    using D = mmq_decode<TYPE>;
    extern __shared__ __align__(16) unsigned char mmq_smem[];
    int8_t* ws = (int8_t*)mmq_smem;
    int8_t (*xs)[MMQ_XLD] = (int8_t (*)[MMQ_XLD])(mmq_smem + MMQ_TR * MMQ_LDW);
    MmqScales<TC>& S = *(MmqScales<TC>*)(mmq_smem + MMQ_TR * MMQ_LDW + TC * MMQ_XLD);

    // Tokens vary fastest across the grid, so the blocks resident together
    // share one weight tile: DRAM serves it once and L2 the rest. Rows-fastest
    // streamed the whole matrix from DRAM once per token tile. The grid in
    // airun_matvec_q8 must match this order.
    const int tok0 = blockIdx.x * TC;
    const int row0 = blockIdx.y * MMQ_TR;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wr = (warp % 2) * MMQ_WR;
    const int wc = (warp / 2) * (TC / 4);
    const int n_groups = K / 32;

    float acc[MMQ_MT][TC / 32][4] = {};
    auto src = [&](int r) { return w + (size_t)(row0 + r) * row_bytes; };
    int g_begin, g_end;
    mmq_split_range(n_groups, g_begin, g_end);

    for (int g0 = g_begin; g0 < g_end; g0 += MMQ_KG) {
        mmq_decode_tile<D, TC>(ws, S, src, g0, 0, n_rows - row0, n_groups - g0, tid);
        // Activations: a token's MMQ_KG blocks are contiguous, 9 words each.
        for (int task = tid; task < TC * MMQ_KG * 9; task += 256) {
            const int t = task / (MMQ_KG * 9);
            const int k = task % (MMQ_KG * 9);
            const bool ok = tok0 + t < n_tokens && g0 + k / 9 < n_groups;
            const int v = ok ? ((const int*)(xq + (size_t)(tok0 + t) * n_groups + g0))[k] : 0;
            ((int*)xs[t])[k] = v;
            if (k % 9 == 8) {
                S.xds[k / 9][t] = make_float2(
                    __half2float(__ushort_as_half((unsigned short)(v & 0xFFFF))),
                    __half2float(__ushort_as_half((unsigned short)((unsigned)v >> 16))));
            }
        }
        __syncthreads();
        mmq_chunk<TC, D::SPLIT, D::HAS_MIN>(ws, S, xs, acc, wr, wc, lane);
        __syncthreads();
    }
    mmq_store<TC>(mmq_out(y, partial, n_rows, n_tokens), acc, row0, tok0, wr, wc, lane, n_rows, n_tokens);
}

// --- asynchronous kernel -----------------------------------------------------
// The next chunk's raw bytes stream into shared memory with cp.async while the
// warps run the current chunk's MMAs, then decode from shared memory at ~30
// cycles instead of from L2 at ~500. Staging used to wait on every global load
// and was 60% of the kernel's time (clock64 inside the kernel); a register
// pipeline needed 184 registers and ran slower.
//
// Only for layouts whose chunks are aligned for cp.async and whole.
template <int TYPE> struct mmq_async {
    static constexpr bool OK = false;
    static constexpr int BYTES = 0, STRIDE = 0, COPY = 16, GROUP_MULTIPLE = 1;   // unused
};
// Q8_0 chunks start 8-byte aligned (136 bytes each). Copying from the 16-byte
// boundary at or below the start takes 144 bytes in nine 16-byte pieces instead
// of seventeen 8-byte ones; `skew` says where the chunk begins in the copy.
// Needs a group count divisible by 8, so the last copy ends inside the row.
template <> struct mmq_async<AIRUN_TYPE_Q8_0> {
    static constexpr bool OK = true;
    static constexpr int BYTES = 144, STRIDE = 144, COPY = 16, GROUP_MULTIPLE = 8;
    __device__ static __forceinline__ int base_group(int g0) { return g0; }
    __device__ static __forceinline__ size_t offset(int g0) { return ((size_t)g0 * 34) & ~(size_t)15; }
    __device__ static __forceinline__ int skew(int g0) { return (g0 * 34) & 15; }
};
template <> struct mmq_async<AIRUN_TYPE_Q4_K> {
    static constexpr bool OK = true;
    static constexpr int BYTES = 144, STRIDE = 144, COPY = 16, GROUP_MULTIPLE = MMQ_KG;
    __device__ static __forceinline__ int base_group(int g0) { return (g0 / 8) * 8; }
    __device__ static __forceinline__ size_t offset(int g0) { return (size_t)(g0 / 8) * 144; }
    __device__ static __forceinline__ int skew(int) { return 0; }
};
template <> struct mmq_async<AIRUN_TYPE_Q5_K> {
    static constexpr bool OK = true;
    static constexpr int BYTES = 176, STRIDE = 176, COPY = 16, GROUP_MULTIPLE = MMQ_KG;
    __device__ static __forceinline__ int base_group(int g0) { return (g0 / 8) * 8; }
    __device__ static __forceinline__ size_t offset(int g0) { return (size_t)(g0 / 8) * 176; }
    __device__ static __forceinline__ int skew(int) { return 0; }
};

template <int TYPE, int TC>
constexpr size_t mmq_async_smem() {
    return (size_t)MMQ_TR * MMQ_LDW + sizeof(MmqScales<TC>)
         + 2 * (size_t)MMQ_TR * mmq_async<TYPE>::STRIDE + 2 * (size_t)TC * MMQ_XLD;
}

template <int BYTES>
__device__ __forceinline__ void cp_async(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    if constexpr (BYTES == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem));
    }
}

// Stream-K: with too few tiles for whole waves (a 128-token prefill's 2560-row
// matrices are 20 tiles on 28 SMs; its gate/up launch is 5.4 waves), blocks
// [0, dp) each take one whole tile as usual and the last `sk` blocks share the
// remaining tiles' K chunks evenly, so the tail wave is as long as its work and
// not as one tile. A tile cut across blocks starts in block j (its "head") and
// continues at the start of blocks j+1.. ("tails"); each tail block leaves its
// accumulators in its slot and raises its flag, and the head block adds them
// in block order -- deterministic -- and writes the tile. Tails are at the start
// of a block's range and the head at its end, so the head rarely waits.
// Dependencies only point at higher blocks, which never wait on lower ones:
// that and the sk <= resident-block count the host chooses keep the spin-waits
// deadlock-free. flags hold the `epoch` of the launch that raised them, so they
// need no reset between launches.
struct MmqStreamK {
    int tiles;          // 0: plain grid (blockIdx.x tokens, .y rows, .z K split)
    int dp;
    int sk;
    int tok_tiles;      // token tiles per row tile; tile = row tile * tok_tiles + token tile
    unsigned epoch;
    float* slots;       // [sk][MMQ_TR * TC], thread-major: [e][256]
    unsigned* flags;    // [sk]
};

// GLU: the tile's rows 0..63 are rows of w (a gate projection) and rows
// 64..127 the same rows of w2 (up); the block writes silu(gate) * up for its 64
// rows, the SwiGLU the FFN applies next. Gate and up read the same activations,
// and this skips writing both products out and reading them back in a separate
// swiglu pass (6.7 ms of a 512-token Qwen3-4B prefill).
template <int TYPE, int TC, bool GLU = false>
__global__ __launch_bounds__(256, 1) void matvec_mmq_async_kernel(
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ w2,
    const block_q8a* __restrict__ xq,
    float* __restrict__ y,
    float* __restrict__ partial,
    const int K,
    const int n_rows,
    const int n_tokens,
    const size_t row_bytes,
    const MmqStreamK sk) {

    using D = mmq_decode<TYPE>;
    using A = mmq_async<TYPE>;
    extern __shared__ __align__(16) unsigned char mmq_smem[];
    int8_t* ws = (int8_t*)mmq_smem;
    MmqScales<TC>& S = *(MmqScales<TC>*)(mmq_smem + MMQ_TR * MMQ_LDW);
    unsigned char* wraw = mmq_smem + MMQ_TR * MMQ_LDW + sizeof(MmqScales<TC>);   // [2][TR][STRIDE]
    unsigned char* xraw = wraw + 2 * MMQ_TR * A::STRIDE;                          // [2][TC][XLD]

    constexpr int ROWS = GLU ? MMQ_TR / 2 : MMQ_TR;   // output rows per tile
    constexpr int E = MMQ_MT * (TC / 32) * 4;          // accumulators per thread
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wr = (warp % 2) * MMQ_WR;
    const int wc = (warp / 2) * (TC / 4);
    const int n_groups = K / 32;

    float acc[MMQ_MT][TC / 32][4];
    float* a = &acc[0][0][0];

    // Groups [g_begin, g_end) of the tile at (row0, tok0) into acc.
    auto run = [&](int row0, int tok0, int g_begin, int g_end) {
        const int rows_valid = min(ROWS, n_rows - row0);
        const int toks_valid = min(TC, n_tokens - tok0);

        // Every copy a thread makes has the same source row, destination and
        // piece in every chunk; only the K offset moves. So the addresses are
        // worked out once here. Recomputing them per chunk (a divide by 17 --
        // Q8_0 rows were 17 eight-byte pieces -- and a 64-bit multiply per copy)
        // was a third of the kernel's issue slots in ncu's per-instruction samples.
        // Rows and tokens past the edge are simply not copied: decode skips those
        // rows, and their activation rows are zeroed below.
        constexpr int WP = A::BYTES / A::COPY;                   // pieces per row
        constexpr int WT = (MMQ_TR * WP + 255) / 256;            // weight copies per thread
        constexpr int XP = MMQ_XLD / 16;                         // pieces per token
        constexpr int XT = (TC * XP + 255) / 256;                // activation copies per thread
        const uint8_t* wsrc_t[WT];
        int wdst_t[WT];
#pragma unroll
        for (int i = 0; i < WT; ++i) {
            const int task = tid + i * 256;
            const int r = task / WP, k = task % WP;
            const int rr = GLU ? (r & 63) : r;
            const bool ok = task < MMQ_TR * WP && rr < rows_valid;
            const uint8_t* mat = (GLU && r >= 64) ? w2 : w;
            wsrc_t[i] = ok ? mat + (size_t)(row0 + rr) * row_bytes + k * A::COPY : nullptr;
            wdst_t[i] = r * A::STRIDE + k * A::COPY;
        }
        const unsigned char* xsrc_t[XT];
        int xdst_t[XT];
#pragma unroll
        for (int i = 0; i < XT; ++i) {
            const int task = tid + i * 256;
            const int t = task / XP, k = task % XP;
            const bool ok = task < TC * XP && t < toks_valid;
            xsrc_t[i] = ok ? (const unsigned char*)(xq + (size_t)(tok0 + t) * n_groups) + k * 16 : nullptr;
            xdst_t[i] = t * MMQ_XLD + k * 16;
        }
        auto issue = [&](int g0, int stage) {
            unsigned char* wdst = wraw + stage * (MMQ_TR * A::STRIDE);
            const size_t woff = A::offset(g0);
#pragma unroll
            for (int i = 0; i < WT; ++i) {
                if (wsrc_t[i]) cp_async<A::COPY>(wdst + wdst_t[i], wsrc_t[i] + woff);
            }
            unsigned char* xdst = xraw + stage * (TC * MMQ_XLD);
            const size_t xoff = (size_t)g0 * sizeof(block_q8a);
#pragma unroll
            for (int i = 0; i < XT; ++i) {
                if (xsrc_t[i]) cp_async<16>(xdst + xdst_t[i], xsrc_t[i] + xoff);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };

        // The previous tile's last chunk and epilogue read these buffers.
        __syncthreads();
        if (toks_valid < TC) {
            for (int i = tid; i < 2 * TC * MMQ_XLD / 4; i += 256) {
                const int t = (i * 4 / MMQ_XLD) % TC;
                if (t >= toks_valid) ((int*)xraw)[i] = 0;
            }
        }
#pragma unroll
        for (int e = 0; e < E; ++e) a[e] = 0.0f;
        issue(g_begin, 0);
        int stage = 0;
        for (int g0 = g_begin; g0 < g_end; g0 += MMQ_KG, stage ^= 1) {
            asm volatile("cp.async.wait_group 0;\n" ::);
            // After this barrier every warp has finished the previous chunk's MMAs,
            // so the other stage is free: the next copy goes out here and needs no
            // barrier of its own at the end of the chunk (there were three per
            // chunk; ncu put ~10% of the samples on the waits after them).
            __syncthreads();
            if (g0 + MMQ_KG < g_end) issue(g0 + MMQ_KG, stage ^ 1);
            const unsigned char* wsrc = wraw + (size_t)stage * MMQ_TR * A::STRIDE;
            const int skew = A::skew(g0);
            auto src = [&](int r) { return wsrc + r * A::STRIDE + skew; };
            const int8_t (*xs)[MMQ_XLD] = (const int8_t (*)[MMQ_XLD])(xraw + (size_t)stage * TC * MMQ_XLD);
            mmq_decode_tile<D, TC>(ws, S, src, g0, A::base_group(g0), rows_valid, MMQ_KG, tid, GLU);
            mmq_stage_xds<TC>(xs, S, tid);
            __syncthreads();
            mmq_chunk<TC, D::SPLIT, D::HAS_MIN>(ws, S, xs, acc, wr, wc, lane);
        }
    };

    // Writes the finished tile at (row0, tok0) to `out`.
    auto finish = [&](float* out, int row0, int tok0) {
        if constexpr (GLU) {
            // Warps 2k (gate rows) and 2k+1 (up rows) hold the same (row, token)
            // elements lane for lane. The up warp hands its accumulators over
            // through the now idle copy buffers.
            float* ex = (float*)wraw + (size_t)(warp / 2) * 32 * E;
            __syncthreads();
            if (warp & 1) {
#pragma unroll
                for (int e = 0; e < E; ++e) ex[e * 32 + lane] = a[e];
            }
            __syncthreads();
            if (!(warp & 1)) {
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    const float g = a[e];
                    a[e] = (g / (1.0f + __expf(-g))) * ex[e * 32 + lane];   // as swiglu_kernel
                }
                mmq_store<TC>(out, acc, row0, tok0, 0, wc, lane, n_rows, n_tokens);
            }
        } else {
            mmq_store<TC>(out, acc, row0, tok0, wr, wc, lane, n_rows, n_tokens);
        }
    };

    if (sk.tiles == 0) {
        // Tokens fastest: see matvec_mmq_kernel.
        const int row0 = blockIdx.y * ROWS, tok0 = blockIdx.x * TC;
        int g_begin, g_end;
        mmq_split_range(n_groups, g_begin, g_end);
        run(row0, tok0, g_begin, g_end);
        finish(GLU ? y : mmq_out(y, partial, n_rows, n_tokens), row0, tok0);
        return;
    }

    // Unit u = tile * chunks + chunk; the async kernel's group counts are whole chunks.
    const int chunks = n_groups / MMQ_KG;
    const int64_t sk_units = (int64_t)(sk.tiles - sk.dp) * chunks;
    auto start = [&](int j) { return (int64_t)sk.dp * chunks + sk_units * j / sk.sk; };
    const int b = blockIdx.x;
    const int j = b - sk.dp;   // stream-K block index, if b >= dp
    int64_t u = b < sk.dp ? (int64_t)b * chunks : start(j);
    const int64_t u_end = b < sk.dp ? u + chunks : start(j + 1);
    while (u < u_end) {
        const int tile = (int)(u / chunks);
        const int kb = (int)(u % chunks);
        const int ke = (int)min((int64_t)chunks, kb + (u_end - u));
        const int row0 = (tile / sk.tok_tiles) * ROWS, tok0 = (tile % sk.tok_tiles) * TC;
        run(row0, tok0, kb * MMQ_KG, ke * MMQ_KG);
        if (kb > 0) {
            // A tail: only a block's first tile can start mid-K.
            float* slot = sk.slots + (size_t)j * 256 * E;
#pragma unroll
            for (int e = 0; e < E; ++e) __stcg(slot + e * 256 + tid, a[e]);
            __threadfence();
            __syncthreads();
            if (tid == 0) asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(sk.flags + j), "r"(sk.epoch) : "memory");
        } else {
            if (ke < chunks) {
                // A head: the rest of the tile is at the start of the next blocks.
                const int64_t tile_end = (int64_t)(tile + 1) * chunks;
                for (int jj = j + 1; jj < sk.sk && start(jj) < tile_end; ++jj) {
                    if (start(jj + 1) == start(jj)) continue;   // empty range, no tail
                    if (tid == 0) {
                        unsigned f;
                        do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(f) : "l"(sk.flags + jj) : "memory");
                        } while (f != sk.epoch);
                    }
                    __syncthreads();
                    const float* slot = sk.slots + (size_t)jj * 256 * E;
#pragma unroll
                    for (int e = 0; e < E; ++e) a[e] += __ldcg(slot + e * 256 + tid);
                }
            }
            finish(y, row0, tok0);
        }
        u += ke - kb;
    }
}
