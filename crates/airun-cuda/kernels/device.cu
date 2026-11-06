// Device discovery and memory shims.
//
// cudaDeviceProp changes shape between CUDA releases (CUDA 13 dropped
// memoryClockRate, for one), so airun never binds it from Rust. This file
// flattens the parts we care about into a struct we own.

#include "airun.cuh"

#include <string.h>

extern "C" {

typedef struct {
    int32_t  ordinal;
    char     name[256];
    int32_t  cc_major;
    int32_t  cc_minor;
    int32_t  sm_count;
    int32_t  warp_size;
    int32_t  max_threads_per_block;
    int32_t  max_threads_per_sm;
    int32_t  l2_cache_bytes;
    int32_t  mem_bus_width_bits;
    int32_t  pci_domain_id;
    int32_t  pci_bus_id;
    int32_t  pci_device_id;
    int32_t  integrated;
    int32_t  unified_addressing;
    int32_t  async_engine_count;
    int32_t  cooperative_launch;
    uint64_t total_mem;
    uint64_t free_mem;
    uint64_t shared_mem_per_block;
    uint64_t shared_mem_per_sm;
    uint8_t  uuid[16];
} AirunDeviceInfo;

airun_status airun_device_count(int32_t* out_count) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        *out_count = 0;
        return (airun_status)e;
    }
    *out_count = (int32_t)n;
    return AIRUN_OK;
}

airun_status airun_device_info(int32_t ordinal, AirunDeviceInfo* out) {
    cudaDeviceProp p;
    AIRUN_TRY(cudaGetDeviceProperties(&p, (int)ordinal));

    memset(out, 0, sizeof(*out));
    out->ordinal = ordinal;
    // cudaDeviceProp::name is a fixed 256-byte buffer; copy defensively anyway.
    strncpy(out->name, p.name, sizeof(out->name) - 1);
    out->cc_major              = p.major;
    out->cc_minor              = p.minor;
    out->sm_count              = p.multiProcessorCount;
    out->warp_size             = p.warpSize;
    out->max_threads_per_block = p.maxThreadsPerBlock;
    out->max_threads_per_sm    = p.maxThreadsPerMultiProcessor;
    out->l2_cache_bytes        = p.l2CacheSize;
    out->mem_bus_width_bits    = p.memoryBusWidth;
    out->pci_domain_id         = p.pciDomainID;
    out->pci_bus_id            = p.pciBusID;
    out->pci_device_id         = p.pciDeviceID;
    out->integrated            = p.integrated;
    out->unified_addressing    = p.unifiedAddressing;
    out->async_engine_count    = p.asyncEngineCount;
    out->cooperative_launch    = p.cooperativeLaunch;
    out->total_mem             = (uint64_t)p.totalGlobalMem;
    out->shared_mem_per_block  = (uint64_t)p.sharedMemPerBlock;
    out->shared_mem_per_sm     = (uint64_t)p.sharedMemPerMultiprocessor;
    memcpy(out->uuid, &p.uuid, sizeof(out->uuid));

    // Free memory is only meaningful with the device current, and it is the
    // number the VRAM planner actually budgets against.
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess &&
        cudaSetDevice((int)ordinal) == cudaSuccess) {
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
            out->free_mem  = (uint64_t)freeB;
            out->total_mem = (uint64_t)totalB;
        }
        cudaSetDevice(previous);
    }
    return AIRUN_OK;
}

airun_status airun_mem_info(int32_t ordinal, uint64_t* free_out, uint64_t* total_out) {
    int previous = 0;
    AIRUN_TRY(cudaGetDevice(&previous));
    AIRUN_TRY(cudaSetDevice((int)ordinal));
    size_t freeB = 0, totalB = 0;
    cudaError_t e = cudaMemGetInfo(&freeB, &totalB);
    cudaSetDevice(previous);
    if (e != cudaSuccess) return (airun_status)e;
    *free_out  = (uint64_t)freeB;
    *total_out = (uint64_t)totalB;
    return AIRUN_OK;
}

// Whether `from` can address `to`'s memory directly. Consumer cards under WDDM
// generally cannot, which forces staged copies through host memory.
airun_status airun_can_access_peer(int32_t from, int32_t to, int32_t* out) {
    int can = 0;
    AIRUN_TRY(cudaDeviceCanAccessPeer(&can, (int)from, (int)to));
    *out = (int32_t)can;
    return AIRUN_OK;
}

airun_status airun_enable_peer_access(int32_t from, int32_t to) {
    int previous = 0;
    AIRUN_TRY(cudaGetDevice(&previous));
    AIRUN_TRY(cudaSetDevice((int)from));
    cudaError_t e = cudaDeviceEnablePeerAccess((int)to, 0);
    cudaSetDevice(previous);
    // Already-enabled is a success as far as callers are concerned.
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return AIRUN_OK;
    }
    return (airun_status)e;
}

airun_status airun_set_device(int32_t ordinal) {
    return (airun_status)cudaSetDevice((int)ordinal);
}

airun_status airun_device_synchronize(void) {
    return (airun_status)cudaDeviceSynchronize();
}

// --- memory ---------------------------------------------------------------

airun_status airun_malloc(int32_t ordinal, uint64_t bytes, void** out) {
    AIRUN_TRY(cudaSetDevice((int)ordinal));
    void* ptr = nullptr;
    cudaError_t e = cudaMalloc(&ptr, (size_t)bytes);
    if (e != cudaSuccess) {
        *out = nullptr;
        return (airun_status)e;
    }
    *out = ptr;
    return AIRUN_OK;
}

airun_status airun_free(int32_t ordinal, void* ptr) {
    if (ptr == nullptr) return AIRUN_OK;
    AIRUN_TRY(cudaSetDevice((int)ordinal));
    return (airun_status)cudaFree(ptr);
}

// Page-locked host memory. Worth the allocation cost for the weight-upload path
// and for multi-GPU staging, where pageable memory would halve throughput.
// Mapped too, so a kernel can read a table that stays in RAM (the token
// embedding) a few rows at a time over PCIe.
airun_status airun_host_alloc(uint64_t bytes, void** out) {
    void* ptr = nullptr;
    cudaError_t e = cudaHostAlloc(&ptr, (size_t)bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (e != cudaSuccess) {
        *out = nullptr;
        return (airun_status)e;
    }
    *out = ptr;
    return AIRUN_OK;
}

// The address kernels on `ordinal` use for a mapped host allocation; the host
// pointer itself under unified addressing.
airun_status airun_host_device_ptr(int32_t ordinal, void* host, void** out) {
    AIRUN_TRY(cudaSetDevice((int)ordinal));
    return (airun_status)cudaHostGetDevicePointer(out, host, 0);
}

airun_status airun_host_free(void* ptr) {
    if (ptr == nullptr) return AIRUN_OK;
    return (airun_status)cudaFreeHost(ptr);
}

// cudaMemcpyDefault throughout: a KV cache kept in RAM (mapped pinned memory)
// is handed around as a device buffer, so a "device" side may be host memory,
// and the runtime works the direction out from the addresses.
airun_status airun_memcpy_h2d(void* dst, const void* src, uint64_t bytes, cudaStream_t stream) {
    return (airun_status)cudaMemcpyAsync(dst, src, (size_t)bytes, cudaMemcpyDefault, stream);
}

airun_status airun_memcpy_d2h(void* dst, const void* src, uint64_t bytes, cudaStream_t stream) {
    return (airun_status)cudaMemcpyAsync(dst, src, (size_t)bytes, cudaMemcpyDefault, stream);
}

airun_status airun_memcpy_d2d(void* dst, const void* src, uint64_t bytes, cudaStream_t stream) {
    return (airun_status)cudaMemcpyAsync(dst, src, (size_t)bytes, cudaMemcpyDefault, stream);
}

// `rows` runs of `width` bytes, `spitch` apart in src and `dpitch` apart in dst.
airun_status airun_memcpy2d_d2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                                uint64_t width, uint64_t rows, cudaStream_t stream) {
    return (airun_status)cudaMemcpy2DAsync(dst, (size_t)dpitch, src, (size_t)spitch, (size_t)width,
                                           (size_t)rows, cudaMemcpyDefault, stream);
}

// Cross-device copy. Falls back to a staged copy through host memory when the
// pair has no peer access, which is the common case on consumer Windows.
airun_status airun_memcpy_peer(void* dst, int32_t dst_dev,
                               const void* src, int32_t src_dev,
                               uint64_t bytes, cudaStream_t stream) {
    return (airun_status)cudaMemcpyPeerAsync(dst, (int)dst_dev, src, (int)src_dev,
                                             (size_t)bytes, stream);
}

airun_status airun_memset(void* dst, int32_t value, uint64_t bytes, cudaStream_t stream) {
    return (airun_status)cudaMemsetAsync(dst, value, (size_t)bytes, stream);
}

// --- streams and events ---------------------------------------------------

airun_status airun_stream_create(int32_t ordinal, cudaStream_t* out) {
    AIRUN_TRY(cudaSetDevice((int)ordinal));
    return (airun_status)cudaStreamCreateWithFlags(out, cudaStreamNonBlocking);
}

airun_status airun_stream_destroy(cudaStream_t stream) {
    if (stream == nullptr) return AIRUN_OK;
    return (airun_status)cudaStreamDestroy(stream);
}

airun_status airun_stream_synchronize(cudaStream_t stream) {
    return (airun_status)cudaStreamSynchronize(stream);
}

airun_status airun_event_create(cudaEvent_t* out) {
    return (airun_status)cudaEventCreate(out);
}

airun_status airun_event_destroy(cudaEvent_t ev) {
    if (ev == nullptr) return AIRUN_OK;
    return (airun_status)cudaEventDestroy(ev);
}

airun_status airun_event_record(cudaEvent_t ev, cudaStream_t stream) {
    return (airun_status)cudaEventRecord(ev, stream);
}

airun_status airun_event_synchronize(cudaEvent_t ev) {
    return (airun_status)cudaEventSynchronize(ev);
}

airun_status airun_event_elapsed_ms(cudaEvent_t start, cudaEvent_t end, float* out) {
    return (airun_status)cudaEventElapsedTime(out, start, end);
}

// Makes `stream` wait for `ev`, which may have been recorded on another device.
airun_status airun_stream_wait_event(cudaStream_t stream, cudaEvent_t ev) {
    return (airun_status)cudaStreamWaitEvent(stream, ev, 0);
}

// --- CUDA graphs -----------------------------------------------------------
//
// A decode step is ~300 tiny kernel launches, and the CPU-side cost of issuing
// them leaves the GPU idle between kernels -- measured at ~40% of a decode
// step on a 1.7B. Capturing the step once and replaying it moves the whole
// launch sequence into one call.

airun_status airun_capture_begin(cudaStream_t stream) {
    return (airun_status)cudaStreamBeginCapture(stream,
                                                cudaStreamCaptureModeThreadLocal);
}

// Ends capture and instantiates an executable graph. The intermediate graph
// object is destroyed either way; only the executable form is kept.
airun_status airun_capture_end(cudaStream_t stream, void** exec_out) {
    cudaGraph_t graph = nullptr;
    AIRUN_TRY(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) return (airun_status)err;
    *exec_out = (void*)exec;
    return AIRUN_OK;
}

airun_status airun_graph_launch(void* exec, cudaStream_t stream) {
    return (airun_status)cudaGraphLaunch((cudaGraphExec_t)exec, stream);
}

void airun_graph_destroy(void* exec) {
    if (exec) cudaGraphExecDestroy((cudaGraphExec_t)exec);
}

// --- diagnostics ----------------------------------------------------------

const char* airun_error_string(airun_status status) {
    return cudaGetErrorString((cudaError_t)status);
}

const char* airun_error_name(airun_status status) {
    return cudaGetErrorName((cudaError_t)status);
}

int32_t airun_runtime_version(void) {
    int v = 0;
    if (cudaRuntimeGetVersion(&v) != cudaSuccess) return 0;
    return (int32_t)v;
}

int32_t airun_driver_version(void) {
    int v = 0;
    if (cudaDriverGetVersion(&v) != cudaSuccess) return 0;
    return (int32_t)v;
}

// Clears a sticky error left by a previous failed launch so the caller can
// report the real cause instead of an unrelated follow-on error.
void airun_clear_error(void) {
    cudaGetLastError();
}

} // extern "C"
