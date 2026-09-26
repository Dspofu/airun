//! CUDA errors, translated into something a user can act on.

use crate::ffi;

pub type Result<T> = std::result::Result<T, CudaError>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA error {code} ({name}): {message}{}", hint_suffix(*code))]
    Runtime {
        code: i32,
        name: String,
        message: String,
    },

    #[error(
        "out of memory on GPU {device}: tried to allocate {}, but only {} of {} is free{}",
        crate::human_bytes(*requested),
        crate::human_bytes(*free),
        crate::human_bytes(*total),
        oom_hint(*requested, *free)
    )]
    OutOfMemory {
        device: i32,
        requested: u64,
        free: u64,
        total: u64,
    },

    #[error("no CUDA device found; airun needs an NVIDIA GPU with a working driver")]
    NoDevice,

    #[error("GPU {ordinal} does not exist (this machine has {available} CUDA device(s))")]
    BadOrdinal { ordinal: i32, available: i32 },

    #[error(
        "this airun binary was built without CUDA kernels; rebuild with the CUDA toolkit \
         installed (set CUDA_PATH) to enable GPU execution"
    )]
    NotBuiltWithCuda,
}

impl CudaError {
    pub fn from_status(status: ffi::Status) -> Self {
        // SAFETY: both entry points only read from static strings.
        let (name, message) = unsafe {
            (
                ffi::cstr(ffi::airun_error_name(status)),
                ffi::cstr(ffi::airun_error_string(status)),
            )
        };
        if name == "AIRUN_NO_CUDA" {
            return CudaError::NotBuiltWithCuda;
        }
        match status {
            // cudaErrorNoDevice / cudaErrorInsufficientDriver
            100 | 35 => CudaError::NoDevice,
            _ => CudaError::Runtime {
                code: status,
                name,
                message,
            },
        }
    }

    /// True when retrying with a smaller footprint could succeed. Drives the
    /// automatic fallbacks in the loader.
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            CudaError::OutOfMemory { .. } => true,
            // cudaErrorMemoryAllocation
            CudaError::Runtime { code, .. } => *code == 2,
            _ => false,
        }
    }
}

/// Actionable advice for the error codes users actually hit.
fn hint_suffix(code: i32) -> &'static str {
    match code {
        2 => "\n  hint: reduce --ctx-size, offload fewer layers with --gpu-layers, or use a smaller quantization",
        209 => "\n  hint: no kernel image for this GPU. Rebuild with AIRUN_CUDA_ARCHS set to your compute capability",
        222 => "\n  hint: the PTX was compiled by a newer toolkit than the installed driver supports; update the NVIDIA driver",
        304 | 719 => "\n  hint: a kernel crashed and the context is unusable. This is a bug in airun; please report the model and flags",
        700 => "\n  hint: illegal memory access inside a kernel. This is a bug in airun; please report the model and flags",
        702 => "\n  hint: a kernel exceeded the OS watchdog timeout. On Windows, raise the TDR delay or use a smaller batch",
        _ => "",
    }
}

fn oom_hint(requested: u64, free: u64) -> String {
    if requested > free {
        let short = requested - free;
        format!(
            "\n  hint: {} short. Lower --ctx-size, drop --gpu-layers, or spread the model with \
             --split-mode layer across more GPUs",
            crate::human_bytes(short)
        )
    } else {
        // Enough free memory in aggregate means fragmentation or another process
        // grabbing memory between the query and the allocation.
        "\n  hint: enough memory is free in aggregate, so this is fragmentation or another \
         process allocating concurrently. Close other GPU workloads and retry"
            .to_string()
    }
}
