//! Raw bindings to the C shims in `kernels/`.
//!
//! Every symbol here is `extern "C"` and uses only fixed-width types, so nothing
//! in this module depends on the CUDA toolkit's own header layout.

use std::ffi::c_void;

/// Zero on success; otherwise a `cudaError_t` value.
pub type Status = i32;
pub const AIRUN_OK: Status = 0;

/// Opaque `cudaStream_t`.
pub type Stream = *mut c_void;
/// Opaque `cudaEvent_t`.
pub type Event = *mut c_void;

/// Mirrors `AirunDeviceInfo` in `kernels/device.cu`. Keep the two in sync.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeviceInfoRaw {
    pub ordinal: i32,
    pub name: [u8; 256],
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_sm: i32,
    pub l2_cache_bytes: i32,
    pub mem_bus_width_bits: i32,
    pub pci_domain_id: i32,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub integrated: i32,
    pub unified_addressing: i32,
    pub async_engine_count: i32,
    pub cooperative_launch: i32,
    pub total_mem: u64,
    pub free_mem: u64,
    pub shared_mem_per_block: u64,
    pub shared_mem_per_sm: u64,
    pub uuid: [u8; 16],
}

impl Default for DeviceInfoRaw {
    fn default() -> Self {
        // SAFETY: every field is a plain integer or byte array, so all-zero is a
        // valid (if meaningless) value. The struct is filled by the shim before use.
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(airun_cuda)]
unsafe extern "C" {
    pub fn airun_device_count(out_count: *mut i32) -> Status;
    pub fn airun_device_info(ordinal: i32, out: *mut DeviceInfoRaw) -> Status;
    pub fn airun_mem_info(ordinal: i32, free_out: *mut u64, total_out: *mut u64) -> Status;
    pub fn airun_can_access_peer(from: i32, to: i32, out: *mut i32) -> Status;
    pub fn airun_enable_peer_access(from: i32, to: i32) -> Status;
    pub fn airun_set_device(ordinal: i32) -> Status;
    pub fn airun_device_synchronize() -> Status;

    pub fn airun_malloc(ordinal: i32, bytes: u64, out: *mut *mut c_void) -> Status;
    pub fn airun_free(ordinal: i32, ptr: *mut c_void) -> Status;
    pub fn airun_host_alloc(bytes: u64, out: *mut *mut c_void) -> Status;
    pub fn airun_host_free(ptr: *mut c_void) -> Status;
    pub fn airun_host_device_ptr(ordinal: i32, host: *mut c_void, out: *mut *mut c_void) -> Status;

    pub fn airun_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: u64, stream: Stream) -> Status;
    pub fn airun_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: u64, stream: Stream) -> Status;
    pub fn airun_memcpy_d2d(dst: *mut c_void, src: *const c_void, bytes: u64, stream: Stream) -> Status;
    pub fn airun_memcpy2d_d2d(dst: *mut c_void, dpitch: u64, src: *const c_void, spitch: u64,
        width: u64, rows: u64, stream: Stream) -> Status;
    pub fn airun_memcpy_peer(
        dst: *mut c_void,
        dst_dev: i32,
        src: *const c_void,
        src_dev: i32,
        bytes: u64,
        stream: Stream,
    ) -> Status;
    pub fn airun_memset(dst: *mut c_void, value: i32, bytes: u64, stream: Stream) -> Status;

    pub fn airun_capture_begin(stream: Stream) -> Status;
    pub fn airun_capture_end(stream: Stream, exec_out: *mut *mut c_void) -> Status;
    pub fn airun_graph_launch(exec: *mut c_void, stream: Stream) -> Status;
    pub fn airun_graph_destroy(exec: *mut c_void);

    pub fn airun_stream_create(ordinal: i32, out: *mut Stream) -> Status;
    pub fn airun_stream_destroy(stream: Stream) -> Status;
    pub fn airun_stream_synchronize(stream: Stream) -> Status;
    pub fn airun_stream_wait_event(stream: Stream, ev: Event) -> Status;

    pub fn airun_event_create(out: *mut Event) -> Status;
    pub fn airun_event_destroy(ev: Event) -> Status;
    pub fn airun_event_record(ev: Event, stream: Stream) -> Status;
    pub fn airun_event_synchronize(ev: Event) -> Status;
    pub fn airun_event_elapsed_ms(start: Event, end: Event, out: *mut f32) -> Status;

    pub fn airun_error_string(status: Status) -> *const std::ffi::c_char;
    pub fn airun_error_name(status: Status) -> *const std::ffi::c_char;
    pub fn airun_runtime_version() -> i32;
    pub fn airun_driver_version() -> i32;
    pub fn airun_clear_error();
}

/// Stubs for builds without a CUDA toolkit, so the rest of the workspace still
/// compiles and can report the situation cleanly.
#[cfg(not(airun_cuda))]
mod stub {
    use super::*;

    /// `cudaErrorNoDevice`
    const NO_CUDA: Status = 100;

    macro_rules! stub_fns {
        ($( fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty = $val:expr; )*) => {
            $(
                #[unsafe(no_mangle)]
                pub unsafe extern "C" fn $name($(_: $ty),*) -> $ret { $val }
            )*
        };
    }

    stub_fns! {
        fn airun_device_count(a: *mut i32) -> Status = NO_CUDA;
        fn airun_device_info(a: i32, b: *mut DeviceInfoRaw) -> Status = NO_CUDA;
        fn airun_mem_info(a: i32, b: *mut u64, c: *mut u64) -> Status = NO_CUDA;
        fn airun_can_access_peer(a: i32, b: i32, c: *mut i32) -> Status = NO_CUDA;
        fn airun_enable_peer_access(a: i32, b: i32) -> Status = NO_CUDA;
        fn airun_set_device(a: i32) -> Status = NO_CUDA;
        fn airun_device_synchronize() -> Status = NO_CUDA;
        fn airun_malloc(a: i32, b: u64, c: *mut *mut c_void) -> Status = NO_CUDA;
        fn airun_free(a: i32, b: *mut c_void) -> Status = NO_CUDA;
        fn airun_host_alloc(a: u64, b: *mut *mut c_void) -> Status = NO_CUDA;
        fn airun_host_free(a: *mut c_void) -> Status = NO_CUDA;
        fn airun_host_device_ptr(a: i32, b: *mut c_void, c: *mut *mut c_void) -> Status = NO_CUDA;
        fn airun_memset(a: *mut c_void, b: i32, c: u64, d: Stream) -> Status = NO_CUDA;
        fn airun_capture_begin(a: Stream) -> Status = NO_CUDA;
        fn airun_capture_end(a: Stream, b: *mut *mut c_void) -> Status = NO_CUDA;
        fn airun_graph_launch(a: *mut c_void, b: Stream) -> Status = NO_CUDA;
        fn airun_stream_create(a: i32, b: *mut Stream) -> Status = NO_CUDA;
        fn airun_stream_destroy(a: Stream) -> Status = NO_CUDA;
        fn airun_stream_synchronize(a: Stream) -> Status = NO_CUDA;
        fn airun_stream_wait_event(a: Stream, b: Event) -> Status = NO_CUDA;
        fn airun_event_create(a: *mut Event) -> Status = NO_CUDA;
        fn airun_event_destroy(a: Event) -> Status = NO_CUDA;
        fn airun_event_record(a: Event, b: Stream) -> Status = NO_CUDA;
        fn airun_event_synchronize(a: Event) -> Status = NO_CUDA;
        fn airun_event_elapsed_ms(a: Event, b: Event, c: *mut f32) -> Status = NO_CUDA;
        fn airun_runtime_version() -> i32 = 0;
        fn airun_driver_version() -> i32 = 0;
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_memcpy_h2d(_: *mut c_void, _: *const c_void, _: u64, _: Stream) -> Status { NO_CUDA }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_memcpy_d2h(_: *mut c_void, _: *const c_void, _: u64, _: Stream) -> Status { NO_CUDA }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_memcpy_d2d(_: *mut c_void, _: *const c_void, _: u64, _: Stream) -> Status { NO_CUDA }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_memcpy2d_d2d(_: *mut c_void, _: u64, _: *const c_void, _: u64, _: u64, _: u64, _: Stream) -> Status { NO_CUDA }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_memcpy_peer(_: *mut c_void, _: i32, _: *const c_void, _: i32, _: u64, _: Stream) -> Status { NO_CUDA }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_graph_destroy(_: *mut c_void) {}
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_clear_error() {}

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_error_string(_: Status) -> *const std::ffi::c_char {
        c"airun was built without CUDA support".as_ptr()
    }
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn airun_error_name(_: Status) -> *const std::ffi::c_char {
        c"AIRUN_NO_CUDA".as_ptr()
    }
}

#[cfg(not(airun_cuda))]
pub use stub::*;

/// Turns a status code into a `Result`.
pub fn check(status: Status) -> crate::Result<()> {
    if status == AIRUN_OK {
        Ok(())
    } else {
        Err(crate::CudaError::from_status(status))
    }
}

/// Reads a NUL-terminated C string returned by the shim.
pub(crate) fn cstr(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        return "(null)".into();
    }
    // SAFETY: the shim only ever returns pointers to static string literals
    // owned by the CUDA runtime, which outlive this call.
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Reads a fixed-size, NUL-padded C char array.
pub(crate) fn fixed_str(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
