// Shared definitions for all airun kernels.
//
// Everything crossing the Rust boundary is plain C with fixed-width types, so
// the FFI surface does not shift when the CUDA toolkit does.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define AIRUN_WARP 32

// Status codes. Zero is success; anything else is a cudaError_t value, so the
// Rust side can render it with cudaGetErrorString.
typedef int32_t airun_status;
#define AIRUN_OK 0

// Wraps a kernel launch: reports a launch-time error immediately rather than
// letting it surface at the next unrelated synchronization point.
#define AIRUN_CHECK_LAUNCH()                    \
    do {                                        \
        cudaError_t _e = cudaGetLastError();    \
        if (_e != cudaSuccess) return (airun_status)_e; \
        return AIRUN_OK;                        \
    } while (0)

#define AIRUN_TRY(expr)                         \
    do {                                        \
        cudaError_t _e = (expr);                \
        if (_e != cudaSuccess) return (airun_status)_e; \
    } while (0)

// ---------------------------------------------------------------------------
// bfloat16
// ---------------------------------------------------------------------------
// We carry bf16 as a raw uint16_t rather than __nv_bfloat16 so the kernels build
// identically on every supported architecture; the conversion is a shift.

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

__device__ __forceinline__ uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    // Round to nearest even, matching what PyTorch writes.
    uint32_t rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    return (uint16_t)(rounded >> 16);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = AIRUN_WARP / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset, AIRUN_WARP);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = AIRUN_WARP / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset, AIRUN_WARP));
    }
    return v;
}

// Block-wide sum. `shared` must hold at least blockDim.x / 32 floats.
__device__ __forceinline__ float block_reduce_sum(float v, float* shared) {
    const int lane = threadIdx.x % AIRUN_WARP;
    const int warp = threadIdx.x / AIRUN_WARP;
    const int warps = (blockDim.x + AIRUN_WARP - 1) / AIRUN_WARP;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    v = (threadIdx.x < warps) ? shared[lane] : 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* shared) {
    const int lane = threadIdx.x % AIRUN_WARP;
    const int warp = threadIdx.x / AIRUN_WARP;
    const int warps = (blockDim.x + AIRUN_WARP - 1) / AIRUN_WARP;

    v = warp_reduce_max(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    v = (threadIdx.x < warps) ? shared[lane] : -INFINITY;
    if (warp == 0) v = warp_reduce_max(v);
    return v;
}

__device__ __forceinline__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// RoPE frequencies
// ---------------------------------------------------------------------------
// Plain RoPE scales the position by freq_scale (1/factor under linear
// scaling). YaRN (ext_factor != 0) keeps the fast dimensions extrapolated, the
// slow ones interpolated, ramps between them over [corr_lo, corr_hi], and
// scales cos/sin by mscale, as llama.cpp's rope_yarn. Mirrored by
// `RopeParams::cos_sin` in airun-cuda/src/ops.rs.
typedef struct {
    float base;
    float freq_scale;
    float ext_factor;
    float mscale;
    float corr_lo;
    float corr_hi;
    // Where the rotation angle's position comes from. By default it is the
    // token's KV-cache slot; after an image (whose tokens take max(rows, cols)
    // positions, not rows * cols) the text continues at slot - *shift. While
    // an image's own tokens run, mpos holds their (t, row, col) positions as
    // [3][n_tok], and each rotary pair takes one of them by the interleaved
    // M-RoPE sections (Qwen3-VL / Qwen3.5: ggml's GGML_ROPE_TYPE_IMROPE).
    const int32_t* shift;   // device scalar, or null for 0
    const int32_t* mpos;    // [3][n_tok], or null
    int32_t n_tok;
    int32_t sections[3];    // rotary pairs per axis (t, row, col)
} airun_rope_params;

// The position rotary pair `pair` of `token` turns by; `slot` is the token's
// KV-cache position.
__device__ __forceinline__ int rope_position(const airun_rope_params& p, int pair, int token, int slot) {
    if (p.mpos) {
        int axis = 0;
        if (pair % 3 == 1 && pair < 3 * p.sections[1]) axis = 1;
        else if (pair % 3 == 2 && pair < 3 * p.sections[2]) axis = 2;
        return p.mpos[axis * p.n_tok + token];
    }
    return p.shift ? slot - *p.shift : slot;
}

// The accurate sincosf, not __sincosf: at a million positions the angle of the
// fastest pair reaches 1e6 rad, where the intrinsic's range reduction fails.
__device__ __forceinline__ void rope_cos_sin(const airun_rope_params& p, int pair, int rope_dim,
                                             int pos, float* c, float* s) {
    const float inv_freq = powf(p.base, -2.0f * (float)pair / (float)rope_dim);
    float theta;
    if (p.ext_factor == 0.0f) {
        theta = ((float)pos * p.freq_scale) * inv_freq;
    } else {
        const float extrap = (float)pos * inv_freq;
        const float interp = p.freq_scale * extrap;
        const float y = ((float)pair - p.corr_lo) / fmaxf(0.001f, p.corr_hi - p.corr_lo);
        const float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * p.ext_factor;
        theta = interp * (1.0f - ramp) + extrap * ramp;
    }
    sincosf(theta, s, c);
    *c *= p.mscale;
    *s *= p.mscale;
}
