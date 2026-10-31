// Hyper-connections and per-layer n-gram embeddings (qwen4exp).
//
// The residual is `hc` parallel streams of width d, laid out [token][stream][d].
// A mix reads them through a grouped RMSNorm (one norm per stream, one gamma per
// stream) and averages them under a sigmoid gate into the block's input; a
// combine adds the block's output back into every stream under a per-stream
// weight. The arithmetic follows llama.cpp's qwen4exp graph
// (build_hc_mix / build_hc_combine / build_ple), op for op.

#include "airun.cuh"

// RMSNorm of every d-wide row; row r scales by gamma (r % group) of `w`.
__global__ void rmsnorm_grouped_kernel(
    float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ w,
    const int d, const int group, const float eps) {
    __shared__ float red[32];
    __shared__ float inv;
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * d;
    float* o = out + (size_t)row * d;
    const float* g = w + (size_t)(row % group) * d;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    ss = block_reduce_sum(ss, red);
    if (threadIdx.x == 0) inv = rsqrtf(ss / (float)d + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = xr[i] * inv * g[i];
}

// out[t][i] = mean over streams of xn * sigmoid(gate).
__global__ void hc_pre_kernel(
    float* __restrict__ out, const float* __restrict__ xn, const float* __restrict__ gate,
    const int n_tokens, const int hc, const int d) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tokens * d) return;
    const int t = idx / d, i = idx % d;
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const size_t at = ((size_t)t * hc + c) * d + i;
        acc += xn[at] * (1.0f / (1.0f + __expf(-gate[at])));
    }
    out[idx] = acc * (1.0f / (float)hc);
}

__global__ void silu_scale_kernel(float* __restrict__ x, const int n, const float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] * scale;
    x[i] = v / (1.0f + __expf(-v));
}

// x[t][c][i] += b[t][i] * 2 sigmoid(inject[t][c] / hc): a zero injection is a
// plain residual add into every stream.
__global__ void hc_post_kernel(
    float* __restrict__ x, const float* __restrict__ b, const float* __restrict__ inject,
    const int n_tokens, const int hc, const int d) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n_tokens * hc * d) return;
    const int i = idx % d;
    const size_t tc = idx / d;
    const int t = (int)(tc / hc);
    const float w = 2.0f / (1.0f + __expf(-inject[tc] / (float)hc));
    x[idx] += b[(size_t)t * d + i] * w;
}

// One block per (token, stream): s = <key, query> / sqrt(d), the gate is
// sigmoid(sign(s) sqrt(max(|s|, 1e-6))), and the stream's row of `gated` is
// the token's value times it.
__global__ void ple_gate_kernel(
    float* __restrict__ gated, const float* __restrict__ key, const float* __restrict__ query,
    const float* __restrict__ value, const int hc, const int d) {
    __shared__ float red[32];
    __shared__ float g_sh;
    const int tc = blockIdx.x;
    const int t = tc / hc;
    const float* k = key + (size_t)tc * d;
    const float* q = query + (size_t)tc * d;
    float s = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s = fmaf(k[i], q[i], s);
    s = block_reduce_sum(s, red);
    if (threadIdx.x == 0) {
        s *= rsqrtf((float)d);
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sg = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
        g_sh = 1.0f / (1.0f + __expf(-sg * mag));
    }
    __syncthreads();
    const float g = g_sh;
    const float* v = value + (size_t)t * d;
    float* o = gated + (size_t)tc * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = v[i] * g;
}

// The PLE's depthwise causal convolution, dilated: tap k of channel c reads
// (K-1-k)*dil tokens back, reaching into `hist` (the last (K-1)*dil normalized
// rows of earlier batches) before the batch starts. x += gated + silu(conv).
__device__ __forceinline__ float ple_at(const float* xn, const float* hist, int j, int c, int ch, int H) {
    return j >= 0 ? xn[(size_t)j * ch + c] : hist[(size_t)(H + j) * ch + c];
}

__global__ void ple_conv_kernel(
    float* __restrict__ x, const float* __restrict__ gated, const float* __restrict__ xn,
    const float* __restrict__ hist, const float* __restrict__ w,
    const int n_tokens, const int ch, const int K, const int dil) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n_tokens * ch) return;
    const int c = idx % ch;
    const int t = (int)(idx / ch);
    const int H = (K - 1) * dil;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += ple_at(xn, hist, t - (K - 1 - k) * dil, c, ch, H) * w[(size_t)c * K + k];
    x[idx] += gated[idx] + acc / (1.0f + __expf(-acc));
}

// The history after the batch: rows n-H .. n-1 of [hist | xn].
__global__ void ple_hist_kernel(
    float* __restrict__ out, const float* __restrict__ xn, const float* __restrict__ hist,
    const int n_tokens, const int ch, const int H) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * ch) return;
    const int c = idx % ch;
    const int r = (int)(idx / ch);
    out[idx] = ple_at(xn, hist, n_tokens - H + r, c, ch, H);
}

// The residual starts as hc copies of the embedding.
__global__ void hc_expand_kernel(float* __restrict__ x, const float* __restrict__ e, const int n_tokens,
                                 const int hc, const int d) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n_tokens * hc * d) return;
    x[idx] = e[(idx / ((size_t)hc * d)) * d + idx % d];
}

static int blocks_for(size_t n, int threads) { return (int)((n + threads - 1) / threads); }

extern "C" {

airun_status airun_rmsnorm_grouped(void* out, const void* x, const void* w, int32_t d, int32_t n_rows,
                                   int32_t group, float eps, cudaStream_t stream) {
    if (n_rows <= 0) return AIRUN_OK;
    rmsnorm_grouped_kernel<<<n_rows, 256, 0, stream>>>((float*)out, (const float*)x, (const float*)w, d, group, eps);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_hc_pre(void* out, const void* xn, const void* gate, int32_t n_tokens, int32_t hc, int32_t d,
                          cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    hc_pre_kernel<<<blocks_for((size_t)n_tokens * d, 256), 256, 0, stream>>>(
        (float*)out, (const float*)xn, (const float*)gate, n_tokens, hc, d);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_hc_expand(void* x, const void* e, int32_t n_tokens, int32_t hc, int32_t d,
                             cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    hc_expand_kernel<<<blocks_for((size_t)n_tokens * hc * d, 256), 256, 0, stream>>>(
        (float*)x, (const float*)e, n_tokens, hc, d);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_silu_scale(void* x, int32_t n, float scale, cudaStream_t stream) {
    if (n <= 0) return AIRUN_OK;
    silu_scale_kernel<<<blocks_for(n, 256), 256, 0, stream>>>((float*)x, n, scale);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_hc_post(void* x, const void* b, const void* inject, int32_t n_tokens, int32_t hc, int32_t d,
                           cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    hc_post_kernel<<<blocks_for((size_t)n_tokens * hc * d, 256), 256, 0, stream>>>(
        (float*)x, (const float*)b, (const float*)inject, n_tokens, hc, d);
    AIRUN_CHECK_LAUNCH();
}

airun_status airun_ple_gate(void* gated, const void* key, const void* query, const void* value,
                            int32_t n_tokens, int32_t hc, int32_t d, cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    ple_gate_kernel<<<n_tokens * hc, 256, 0, stream>>>(
        (float*)gated, (const float*)key, (const float*)query, (const float*)value, hc, d);
    AIRUN_CHECK_LAUNCH();
}

// `hist` is updated in place through `scratch` (H * ch floats).
airun_status airun_ple_conv(void* x, const void* gated, const void* xn, void* hist, void* scratch, const void* w,
                            int32_t n_tokens, int32_t ch, int32_t kernel, int32_t dil, cudaStream_t stream) {
    if (n_tokens <= 0) return AIRUN_OK;
    const int H = (kernel - 1) * dil;
    ple_conv_kernel<<<blocks_for((size_t)n_tokens * ch, 256), 256, 0, stream>>>(
        (float*)x, (const float*)gated, (const float*)xn, (const float*)hist, (const float*)w,
        n_tokens, ch, kernel, dil);
    if (H > 0) {
        ple_hist_kernel<<<blocks_for((size_t)H * ch, 256), 256, 0, stream>>>(
            (float*)scratch, (const float*)xn, (const float*)hist, n_tokens, ch, H);
        cudaMemcpyAsync(hist, scratch, (size_t)H * ch * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    AIRUN_CHECK_LAUNCH();
}

} // extern "C"
