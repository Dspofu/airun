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
