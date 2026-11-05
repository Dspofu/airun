//! CUDA layer for airun: device discovery, memory ownership and the kernels.
//!
//! The kernels themselves live in `kernels/*.cu` and are compiled by `build.rs`
//! into a static library. Nothing here binds a CUDA header type directly --
//! every crossing goes through a C shim with fixed-width types, so a toolkit
//! upgrade cannot silently change a struct layout under us.
//!
//! When no CUDA toolkit is present at build time the crate still compiles, with
//! stub entry points that report [`CudaError::NotBuiltWithCuda`].

pub mod buffer;
pub mod device;
pub mod error;
pub mod ffi;
pub mod hc;
pub mod hybrid;
pub mod ops;
pub mod qsa;
pub mod stream;
pub mod vision;

pub use buffer::{DeviceBuffer, PinnedBuffer};
pub use ops::KernelType;
pub use device::{Device, Generation, devices, device_count};
pub use error::{CudaError, Result};
pub use stream::{Event, Graph, Stream, time_gpu};

/// Formats a byte count the way every airun diagnostic does.
/// Raw asynchronous copies on `stream`, for buffers handled as addresses
/// (mapped expert slots and the like). `bytes` must lie inside both sides.
pub fn copy_d2d(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, bytes: usize, stream: &Stream) -> Result<()> {
    // SAFETY: the caller guarantees both ranges are live device memory.
    ffi::check(unsafe { ffi::airun_memcpy_d2d(dst, src, bytes as u64, stream.raw()) })
}

pub fn copy_h2d(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, bytes: usize, stream: &Stream) -> Result<()> {
    // SAFETY: the caller guarantees `dst` is device memory and `src` pinned host memory, both live.
    ffi::check(unsafe { ffi::airun_memcpy_h2d(dst, src, bytes as u64, stream.raw()) })
}

pub fn copy_d2h(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, bytes: usize, stream: &Stream) -> Result<()> {
    // SAFETY: the caller guarantees `dst` is pinned host memory and `src` device memory, both live.
    ffi::check(unsafe { ffi::airun_memcpy_d2h(dst, src, bytes as u64, stream.raw()) })
}

/// `bytes` from `src` on device `src_dev` to `dst` on `dst_dev`, on `stream`
/// (staged through the host when the two GPUs have no peer access).
pub fn copy_peer(dst: *mut std::ffi::c_void, dst_dev: i32, src: *const std::ffi::c_void, src_dev: i32, bytes: usize,
    stream: &Stream) -> Result<()> {
    // SAFETY: the caller guarantees both ranges are live on their devices.
    ffi::check(unsafe { ffi::airun_memcpy_peer(dst, dst_dev, src, src_dev, bytes as u64, stream.raw()) })
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_reads_naturally() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.50 KiB");
        assert_eq!(human_bytes(16 * 1024 * 1024 * 1024), "16.00 GiB");
    }
}
