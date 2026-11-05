// Vision encoder ops (Qwen3-VL-style ViT behind an mmproj file).
//
// The encoder's attention runs on the text model's attention kernel: its 72-wide
// heads are zero-padded to a multiple of 32 (zeros add nothing to q.k and the
// padded output columns are dropped), and every query is given the last
// position, so the causal kernel sees all keys -- the ViT's full attention.
// The ops here are the ones that kernel and the existing elementwise ones do
// not cover: LayerNorm with a bias, a per-column bias, and the split of the
// fused QKV projection with the encoder's 2D rotary embedding.

#include "dequant.cuh"

// LayerNorm over each row: (x - mean) * rsqrt(var + eps) * w + b. One block per
// row; the variance is the mean of squared deviations, computed in a second
// pass over the row (in L1) for accuracy.
__global__ void layernorm_kernel(float* __restrict__ out, const float* __restrict__ x,
                                 const float* __restrict__ w, const float* __restrict__ b,
                                 const int n, const float eps) {
    extern __shared__ float shared[];
    const float* xr = x + (size_t)blockIdx.x * n;
    float* outr = out + (size_t)blockIdx.x * n;
    __shared__ float mean_s, inv_s;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += xr[i];
    sum = block_reduce_sum(sum, shared);
    if (threadIdx.x == 0) mean_s = sum / (float)n;
    __syncthreads();
    const float mean = mean_s;

    float sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = xr[i] - mean;
        sq = fmaf(d, d, sq);
    }
    sq = block_reduce_sum(sq, shared);
    if (threadIdx.x == 0) inv_s = rsqrtf(sq / (float)n + eps);
    __syncthreads();
    const float inv = inv_s;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = (xr[i] - mean) * inv;
        outr[i] = fmaf(v, w ? w[i] : 1.0f, b ? b[i] : 0.0f);
    }
}

// y[r][i] += bias[i] for `rows` rows of n.
__global__ void add_bias_kernel(float* __restrict__ y, const float* __restrict__ bias,
                                const int64_t total, const int n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] += bias[i % n];
}

// The fused projection's output [n][3][H][hd] split into
//   q   [n][H][hdp]  f32, rotated
//   k,v [H][n][hdp]  f16, k rotated: the KV cache layout attention reads
// with hdp - hd zero columns. The rotation is ggml's GGML_ROPE_TYPE_VISION
// with four equal sections: pair j < hd/2 turns elements j and j + hd/2 by
//   pos * base^(-2 (j mod hd/4) / (hd/2)),   pos = y for j < hd/4, else x.
__global__ void vision_qkv_kernel(float* __restrict__ q_out, half* __restrict__ k_out, half* __restrict__ v_out,
                                  const float* __restrict__ qkv, const int32_t* __restrict__ pos_y,
                                  const int32_t* __restrict__ pos_x, const int n, const int n_head,
                                  const int hd, const int hdp, const float base) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;   // 0 .. hdp-1
    if (d >= hdp) return;
    const int width = n_head * hd;
    const float* src = qkv + (size_t)t * 3 * width + (size_t)h * hd;
    const size_t qi = ((size_t)t * n_head + h) * hdp + d;
    const size_t ki = ((size_t)h * n + t) * hdp + d;
    if (d >= hd) {
        q_out[qi] = 0.0f;
        k_out[ki] = __float2half(0.0f);
        v_out[ki] = __float2half(0.0f);
        return;
    }
    const int half_ = hd / 2, quarter = hd / 4;
    const int j = d < half_ ? d : d - half_;     // the pair this element belongs to
    const int p = j < quarter ? pos_y[t] : pos_x[t];
    const float theta = (float)p * powf(base, -2.0f * (float)(j % quarter) / (float)half_);
    float s, c;
    sincosf(theta, &s, &c);
    const int other = d < half_ ? d + half_ : d - half_;
    const float sign = d < half_ ? -1.0f : 1.0f;   // x0' = x0 c - x1 s, x1' = x1 c + x0 s
    const float q0 = src[d], q1 = src[other];
    const float k0 = src[width + d], k1 = src[width + other];
    q_out[qi] = q0 * c + sign * q1 * s;
    k_out[ki] = __float2half(k0 * c + sign * k1 * s);
    v_out[ki] = __float2half(src[2 * width + d]);
}

// [n][H][hdp] -> [n][H * hd], dropping the padding.
__global__ void vision_unpad_kernel(float* __restrict__ out, const float* __restrict__ in,
                                    const int64_t total, const int hd, const int hdp) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / hd;   // (token, head)
    const int d = (int)(i % hd);
    out[i] = in[row * hdp + d];
}

extern "C" {

airun_status airun_layernorm(void* out, const void* x, const void* w, const void* b, int32_t rows, int32_t n,
                             float eps, cudaStream_t stream) {
    if (rows <= 0) return AIRUN_OK;
    const int threads = n >= 1024 ? 1024 : 256;
    layernorm_kernel<<<rows, threads, 32 * sizeof(float), stream>>>(
        (float*)out, (const float*)x, (const float*)w, (const float*)b, n, eps);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_add_bias(void* y, const void* bias, int32_t rows, int32_t n, cudaStream_t stream) {
    const int64_t total = (int64_t)rows * n;
    if (total <= 0) return AIRUN_OK;
    const int threads = 256;
    add_bias_kernel<<<(unsigned)((total + threads - 1) / threads), threads, 0, stream>>>(
        (float*)y, (const float*)bias, total, n);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_vision_qkv(void* q, void* k, void* v, const void* qkv, const void* pos_y, const void* pos_x,
                              int32_t n, int32_t n_head, int32_t hd, int32_t hdp, float base, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    if (hd % 4 != 0 || hdp < hd || hdp > 1024) return (airun_status)cudaErrorInvalidValue;
    dim3 grid(n, n_head);
    vision_qkv_kernel<<<grid, hdp, 0, stream>>>((float*)q, (half*)k, (half*)v, (const float*)qkv,
        (const int32_t*)pos_y, (const int32_t*)pos_x, n, n_head, hd, hdp, base);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_vision_unpad(void* out, const void* in, int32_t rows, int32_t hd, int32_t hdp,
                                cudaStream_t stream) {
    const int64_t total = (int64_t)rows * hd;
    if (total <= 0) return AIRUN_OK;
    const int threads = 256;
    vision_unpad_kernel<<<(unsigned)((total + threads - 1) / threads), threads, 0, stream>>>(
        (float*)out, (const float*)in, total, hd, hdp);
    AIRUN_CHECK_LAUNCH();
}

}
