// NVFP4 x NVFP4 matmul on Blackwell's FP4 tensor cores (sm_120a only).
//
// Built by build.rs as its own unit for compute_120a when sm_120 is among the
// targets; the instruction exists only in the arch-specific target. On any
// other build the kernels compile empty and airun_fp4_supported() says no, so
// RTX 20-40 keep the int8 MMQ path.
//
// The weights go in as stored. An NVFP4 block is 64 values in four runs of 16,
// each run 8 bytes whose byte j holds element j (low nibble) and element j + 8
// (high nibble), plus one UE4M3 scale per run. The m16n8k64 A fragment gives
// lane (g, t) the 32-bit words t and t + 4 of its row's 64 K values, and with
// scale_vec::4X one UE4M3 scale per 16 K. Word w of a block is half of run
// w / 2, so handing the MMA the block's word t (and t + 4) puts every run under
// its own scale; the order of K inside a run is a permutation the MMA cannot
// see, as long as the activation is packed with the same one -- which it is,
// because it is quantized into the very same block layout. The block's four
// scale bytes are the scale register as they are.
//
// The activation side is the precision cost (NVFP4 has 4 bits per value where
// the int8 path keeps 8); each token row also gets one f32 scale so its UE4M3
// run scales use their full range, applied to the output.

#include "../airun.cuh"
#include "../dequant.cuh"
#include <cuda_fp8.h>

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define AIRUN_FP4_MMA 1
#endif

// UE4M3 code of a positive scale, round-to-nearest-even and saturating at 448
// (code 0x7E): never 0x7F, which the hardware reads as NaN.
__device__ __forceinline__ uint8_t f32_to_ue4m3(float x) {
    if (!(x > 0.0f)) return 0;
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

// The UE4M3 value as the hardware reads it (ggml's ue4m3_to_f32 is half of it,
// to pair with doubled E2M1 codes; both describe the same weights).
__device__ __forceinline__ float ue4m3_hw(uint8_t c) {
    return 2.0f * ue4m3_to_f32(c);
}

// Nearest E2M1 code (sign in bit 3) of v, v already divided by its run scale.
__device__ __forceinline__ uint32_t f32_to_e2m1(float v) {
    const float a = fabsf(v);
    uint32_t m;
    if      (a < 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a < 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a < 2.5f)  m = 4;
    else if (a < 3.5f)  m = 5;
    else if (a < 5.0f)  m = 6;
    else                m = 7;
    return (v < 0.0f && m != 0) ? (m | 8u) : m;
}

// One block of 256 threads per token row: the row's amax sets its f32 scale,
// then each thread packs runs of 16 into NVFP4 blocks.
// With `up` non-null the row quantized is silu(x) * up -- the SwiGLU fused in,
// so the FFN's down projection reads its input straight from gate and up.
template <bool GLU>
__device__ __forceinline__ float act_at(const float* row, const float* uprow, int i) {
    if constexpr (GLU) {
        const float g = row[i];
        return g / (1.0f + __expf(-g)) * uprow[i];
    } else {
        return row[i];
    }
}

template <bool GLU>
__global__ __launch_bounds__(256) void quantize_act_nvfp4_kernel(
    const float* __restrict__ x, const float* __restrict__ up, block_nvfp4* __restrict__ out,
    float* __restrict__ row_scale, const int K) {
#ifdef AIRUN_FP4_MMA
    __shared__ float red[8];
    __shared__ float gs_sh;
    const int tok = blockIdx.x;
    const float* row = x + (size_t)tok * K;
    const float* uprow = GLU ? up + (size_t)tok * K : nullptr;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) amax = fmaxf(amax, fabsf(act_at<GLU>(row, uprow, i)));
    amax = block_reduce_max(amax, red);
    if (threadIdx.x == 0) {
        gs_sh = amax / (6.0f * 448.0f);
        row_scale[tok] = gs_sh;
    }
    __syncthreads();
    const float gs = gs_sh;
    const float inv_gs = gs > 0.0f ? 1.0f / gs : 0.0f;

    block_nvfp4* ob = out + (size_t)tok * (K / 64);
    for (int run = threadIdx.x; run < K / 16; run += blockDim.x) {
        float v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = act_at<GLU>(row, uprow, run * 16 + i);
        float sub = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) sub = fmaxf(sub, fabsf(v[i]));
        // One code, amax / 6 rounded. Trying the two codes either side for the
        // least squared error (llama.cpp's quantizer) cost 12% of the prefill and
        // moved a 27B's batched perplexity by -0.005 at 128 tokens and +0.004 at
        // 512: not a consistent gain.
        const int code = f32_to_ue4m3(sub / 6.0f * inv_gs);
        const float scale = ue4m3_hw((uint8_t)code) * gs;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        block_nvfp4* b = ob + run / 4;
        b->d[run % 4] = (uint8_t)code;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w0 |= (f32_to_e2m1(v[j] * inv) | (f32_to_e2m1(v[j + 8] * inv) << 4)) << (8 * j);
            w1 |= (f32_to_e2m1(v[j + 4] * inv) | (f32_to_e2m1(v[j + 12] * inv) << 4)) << (8 * j);
        }
        uint32_t* q = (uint32_t*)(b->qs + (run % 4) * 8);
        q[0] = w0;
        q[1] = w1;
    }
#endif
}

// The same quantization, laid out for the memory system: four lanes share a
// run, each loading one float4, so a warp reads 512 contiguous bytes; the first
// NV float4 of every thread stay in registers between the amax pass and the
// packing pass (all of a row up to K = 1024 * NV), the rest are read again.
// Bit-identical to quantize_act_nvfp4_kernel. Needs 16-byte aligned rows.
template <bool GLU>
__device__ __forceinline__ float4 act4_at(const float4* row, const float4* uprow, int f) {
    float4 v = row[f];
    if constexpr (GLU) {
        const float4 u = uprow[f];
        v.x = v.x / (1.0f + __expf(-v.x)) * u.x;
        v.y = v.y / (1.0f + __expf(-v.y)) * u.y;
        v.z = v.z / (1.0f + __expf(-v.z)) * u.z;
        v.w = v.w / (1.0f + __expf(-v.w)) * u.w;
    }
    return v;
}

__device__ __forceinline__ float max4(float4 v) {
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}

template <bool GLU, int NV>
__global__ __launch_bounds__(256) void quantize_act4_nvfp4_kernel(
    const float* __restrict__ x, const float* __restrict__ up, block_nvfp4* __restrict__ out,
    float* __restrict__ row_scale, const int K) {
#ifdef AIRUN_FP4_MMA
    __shared__ float red[8];
    __shared__ float gs_sh;
    const int tok = blockIdx.x;
    const int K4 = K / 4;
    const float4* row = (const float4*)(x + (size_t)tok * K);
    const float4* uprow = GLU ? (const float4*)(up + (size_t)tok * K) : nullptr;
    const int tid = threadIdx.x;

    float4 cache[NV];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int f = tid + i * 256;
        if (f < K4) {
            cache[i] = act4_at<GLU>(row, uprow, f);
            amax = fmaxf(amax, max4(cache[i]));
        }
    }
    for (int f = tid + NV * 256; f < K4; f += 256) amax = fmaxf(amax, max4(act4_at<GLU>(row, uprow, f)));
    amax = block_reduce_max(amax, red);
    if (tid == 0) {
        gs_sh = amax / (6.0f * 448.0f);
        row_scale[tok] = gs_sh;
    }
    __syncthreads();
    const float gs = gs_sh;
    const float inv_gs = gs > 0.0f ? 1.0f / gs : 0.0f;

    // Lane q of a run holds its elements 4q .. 4q+3. Word 0 of the run is
    // elements 0-3 (low nibbles) and 8-11 (high), word 1 is 4-7 and 12-15.
    const int q = tid & 3;
    const unsigned group = 0xFu << (tid & 28);
    block_nvfp4* ob = out + (size_t)tok * (K / 64);
    auto pack = [&](float4 v, int f) {
        float sub = max4(v);
        sub = fmaxf(sub, __shfl_xor_sync(group, sub, 1));
        sub = fmaxf(sub, __shfl_xor_sync(group, sub, 2));
        const int code = f32_to_ue4m3(sub / 6.0f * inv_gs);
        const float scale = ue4m3_hw((uint8_t)code) * gs;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        const int sh = (q >> 1) * 4;
        uint32_t p = (f32_to_e2m1(v.x * inv) << sh) | (f32_to_e2m1(v.y * inv) << (8 + sh)) |
                     (f32_to_e2m1(v.z * inv) << (16 + sh)) | (f32_to_e2m1(v.w * inv) << (24 + sh));
        p |= __shfl_xor_sync(group, p, 2);
        const int run = f >> 2;
        block_nvfp4* b = ob + run / 4;
        if (q < 2) ((uint32_t*)(b->qs + (run % 4) * 8))[q] = p;
        else if (q == 2) b->d[run % 4] = (uint8_t)code;
    };
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int f = tid + i * 256;
        if (f < K4) pack(cache[i], f);
    }
    for (int f = tid + NV * 256; f < K4; f += 256) pack(act4_at<GLU>(row, uprow, f), f);
#endif
}

__device__ __forceinline__ void mma_nvfp4(float c[4], const uint32_t a[4], const uint32_t b[2],
                                          uint32_t sa, uint32_t sb) {
#ifdef AIRUN_FP4_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
#endif
}

// Block tile 128 rows x 128 tokens, eight warps of 64 rows (four m16) x 32
// tokens (four n8). K moves four NVFP4 blocks (256 values) per stage: a row's
// four blocks are 144 contiguous, 16-byte aligned bytes in both operands, so a
// stage is nine cp.async copies per row, double-buffered while the warps run
// the previous stage's MMAs. Rows stay 144 bytes apart in shared memory: that
// is 36 words, and the eight rows a fragment load touches then fall in eight
// different groups of four banks -- no conflicts. Loading fragments straight
// from global memory instead ran at a seventh of the tensor-core rate.
#define FP4_KB 4                       // blocks per stage
#define FP4_ROW (FP4_KB * 36)          // bytes per row per stage
#define FP4_STAGE (2 * 128 * FP4_ROW)  // weights then activations

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}

// GLU: the tile's rows 0..63 are rows of w (an FFN gate) and 64..127 the same
// rows of w2 (up); the block writes silu(gate) * up for its 64 rows -- the
// SwiGLU in the epilogue. Writing gate and up out and reading both back for
// the down projection's quantization was ~6% of a 27B prefill.
template <bool GLU>
__global__ __launch_bounds__(256, 1) void matmul_nvfp4_kernel(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ w2, const size_t row_bytes,
    const block_nvfp4* __restrict__ xq, const float* __restrict__ xs,
    float* __restrict__ y, const int K, const int n_rows, const int n_tokens) {
#ifdef AIRUN_FP4_MMA
    extern __shared__ __align__(16) unsigned char fp4_smem[];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int brow = blockIdx.y * (GLU ? 64 : 128);
    const int btok = blockIdx.x * 128;
    const int wrow = (warp % 2) * 64;
    const int wtok = (warp / 2) * 32;
    const int nblk = K / 64;
    const size_t x_row_bytes = (size_t)nblk * 36;
    const int chunks = nblk / FP4_KB;

    // Copy tasks: 128 rows x 9 pieces of 16 bytes per operand, 4.5 per thread.
    // Rows past the edge copy the last valid row; their sums are never stored.
    auto issue = [&](int stage, int c) {
        unsigned char* base = fp4_smem + stage * FP4_STAGE;
        const size_t off = (size_t)c * FP4_ROW;
        for (int task = tid; task < 2 * 128 * 9; task += 256) {
            const int op = task / (128 * 9);
            const int r = (task / 9) % 128;
            const int p = task % 9;
            const uint8_t* src;
            if (op != 0) src = (const uint8_t*)xq + (size_t)min(btok + r, n_tokens - 1) * x_row_bytes;
            else if (GLU) src = (r < 64 ? w : w2) + (size_t)min(brow + (r & 63), n_rows - 1) * row_bytes;
            else src = w + (size_t)min(brow + r, n_rows - 1) * row_bytes;
            cp16(base + op * 128 * FP4_ROW + r * FP4_ROW + p * 16, src + off + p * 16);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[4][4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][n][i] = 0.0f;

    issue(0, 0);
    for (int c = 0; c < chunks; ++c) {
        const int stage = c & 1;
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        if (c + 1 < chunks) issue(stage ^ 1, c + 1);
        const uint32_t* ws = (const uint32_t*)(fp4_smem + stage * FP4_STAGE);
        const uint32_t* xsm = ws + 128 * FP4_ROW / 4;
#pragma unroll
        for (int j = 0; j < FP4_KB; ++j) {
            uint32_t a[4][4], sa[4], b[4][2], sb[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint32_t* r0 = ws + (wrow + m * 16 + g) * (FP4_ROW / 4) + j * 9;
                const uint32_t* r1 = r0 + 8 * (FP4_ROW / 4);
                a[m][0] = r0[1 + t];
                a[m][1] = r1[1 + t];
                a[m][2] = r0[5 + t];
                a[m][3] = r1[5 + t];
                // Each quad's even lanes give row g's scales, odd lanes row g + 8's.
                sa[m] = (lane & 1) ? r1[0] : r0[0];
            }
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const uint32_t* q = xsm + (wtok + n * 8 + g) * (FP4_ROW / 4) + j * 9;
                b[n][0] = q[1 + t];
                b[n][1] = q[5 + t];
                sb[n] = q[0];
            }
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int n = 0; n < 4; ++n) mma_nvfp4(acc[m][n], a[m], b[n], sa[m], sb[n]);
        }
    }

    if constexpr (GLU) {
        // The up warp of each pair (odd warps: rows 64..127) hands its sums to
        // the gate warp through shared memory, in the order both hold them.
        __syncthreads();   // the pipeline's buffers are free now
        float* xch = (float*)fp4_smem + (warp / 2) * (64 * 32);
        if (warp & 1) {
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int n = 0; n < 4; ++n)
#pragma unroll
                    for (int i = 0; i < 4; ++i) xch[((m * 4 + n) * 4 + i) * 32 + lane] = acc[m][n][i];
        }
        __syncthreads();
        if (warp & 1) return;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tok = btok + wtok + n * 8 + t * 2 + c;
                if (tok >= n_tokens) continue;
                const float s = xs[tok];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = brow + m * 16 + g + 8 * h;
                        const float gv = acc[m][n][h * 2 + c] * s;
                        const float uv = xch[((m * 4 + n) * 4 + h * 2 + c) * 32 + lane] * s;
                        if (r < n_rows) y[(size_t)tok * n_rows + r] = gv / (1.0f + __expf(-gv)) * uv;
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int tok = btok + wtok + n * 8 + t * 2 + c;
            if (tok >= n_tokens) continue;
            const float s = xs[tok];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = brow + wrow + m * 16 + g + 8 * h;
                    if (r < n_rows) y[(size_t)tok * n_rows + r] = acc[m][n][h * 2 + c] * s;
                }
            }
        }
    }
#endif
}

// ggml reads a 0x7F run scale as zero; the hardware reads it as NaN. Rewritten
// once at load so both readings agree.
__global__ void nvfp4_sanitize_kernel(block_nvfp4* b, const int64_t n_blocks) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) return;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (b[i].d[r] == 0x7F) b[i].d[r] = 0;
    }
}

extern "C" {

int32_t airun_fp4_supported(void) {
#if AIRUN_FP4_BUILT
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major == 12 ? 1 : 0;
#else
    return 0;
#endif
}

// Bytes of the quantized activation for n_tokens rows of K: the blocks, then
// one f32 scale per row.
uint64_t airun_nvfp4_act_bytes(int32_t n_tokens, int32_t k) {
    return (uint64_t)n_tokens * (k / 64) * sizeof(block_nvfp4) + (uint64_t)n_tokens * sizeof(float);
}

airun_status airun_quantize_act_nvfp4(const void* x, void* out, int32_t n_tokens, int32_t k,
                                      cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    if (k % 64 != 0) return (airun_status)cudaErrorInvalidValue;
    block_nvfp4* blocks = (block_nvfp4*)out;
    float* scales = (float*)((uint8_t*)out + (size_t)n_tokens * (k / 64) * sizeof(block_nvfp4));
    if ((uintptr_t)x % 16 == 0)
        quantize_act4_nvfp4_kernel<false, 20><<<n_tokens, 256, 0, stream>>>((const float*)x, nullptr, blocks,
                                                                            scales, k);
    else
        quantize_act_nvfp4_kernel<false><<<n_tokens, 256, 0, stream>>>((const float*)x, nullptr, blocks, scales, k);
    AIRUN_CHECK_LAUNCH();
}

// quantize of silu(gate) * up, for an FFN's down projection.
airun_status airun_quantize_glu_nvfp4(const void* gate, const void* up, void* out, int32_t n_tokens, int32_t k,
                                      cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    if (k % 64 != 0) return (airun_status)cudaErrorInvalidValue;
    block_nvfp4* blocks = (block_nvfp4*)out;
    float* scales = (float*)((uint8_t*)out + (size_t)n_tokens * (k / 64) * sizeof(block_nvfp4));
    if ((uintptr_t)gate % 16 == 0 && (uintptr_t)up % 16 == 0)
        quantize_act4_nvfp4_kernel<true, 20><<<n_tokens, 256, 0, stream>>>((const float*)gate, (const float*)up,
                                                                           blocks, scales, k);
    else
        quantize_act_nvfp4_kernel<true><<<n_tokens, 256, 0, stream>>>((const float*)gate, (const float*)up, blocks,
                                                                      scales, k);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_matmul_nvfp4(const void* w, uint64_t row_bytes, const void* xq, void* y,
                                int32_t k, int32_t n_rows, int32_t n_tokens, cudaStream_t stream) {
    if (n_tokens <= 0 || n_rows <= 0) return AIRUN_OK;
    // Stages are four blocks of 144 bytes, copied 16 at a time.
    if (k % 256 != 0 || row_bytes % 16 != 0 || (uintptr_t)w % 16 != 0 || (uintptr_t)xq % 16 != 0)
        return (airun_status)cudaErrorInvalidValue;
    const block_nvfp4* blocks = (const block_nvfp4*)xq;
    const float* scales = (const float*)((const uint8_t*)xq + (size_t)n_tokens * (k / 64) * sizeof(block_nvfp4));
    const dim3 grid((n_tokens + 127) / 128, (n_rows + 127) / 128);
    const int smem = 2 * FP4_STAGE;
    cudaFuncSetAttribute((const void*)matmul_nvfp4_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_nvfp4_kernel<false><<<grid, 256, smem, stream>>>((const uint8_t*)w, nullptr, (size_t)row_bytes, blocks,
                                                         scales, (float*)y, k, n_rows, n_tokens);
    AIRUN_CHECK_LAUNCH();
}

// y = silu(w x) * (w2 x): an FFN's gate, up and SwiGLU in one launch.
airun_status airun_matmul_nvfp4_glu(const void* w, const void* w2, uint64_t row_bytes, const void* xq, void* y,
                                    int32_t k, int32_t n_rows, int32_t n_tokens, cudaStream_t stream) {
    if (n_tokens <= 0 || n_rows <= 0) return AIRUN_OK;
    if (k % 256 != 0 || row_bytes % 16 != 0 || (uintptr_t)w % 16 != 0 || (uintptr_t)w2 % 16 != 0
        || (uintptr_t)xq % 16 != 0)
        return (airun_status)cudaErrorInvalidValue;
    const block_nvfp4* blocks = (const block_nvfp4*)xq;
    const float* scales = (const float*)((const uint8_t*)xq + (size_t)n_tokens * (k / 64) * sizeof(block_nvfp4));
    const dim3 grid((n_tokens + 127) / 128, (n_rows + 63) / 64);
    const int smem = 2 * FP4_STAGE;
    cudaFuncSetAttribute((const void*)matmul_nvfp4_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_nvfp4_kernel<true><<<grid, 256, smem, stream>>>((const uint8_t*)w, (const uint8_t*)w2, (size_t)row_bytes,
                                                        blocks, scales, (float*)y, k, n_rows, n_tokens);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_nvfp4_sanitize(void* w, int64_t n_blocks, cudaStream_t stream) {
    if (n_blocks <= 0) return AIRUN_OK;
    nvfp4_sanitize_kernel<<<(unsigned)((n_blocks + 255) / 256), 256, 0, stream>>>((block_nvfp4*)w, n_blocks);
    AIRUN_CHECK_LAUNCH();
}

} // extern "C"
