//! Owned device and pinned-host memory.

use std::ffi::c_void;
use std::marker::PhantomData;

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::stream::Stream;

/// A typed allocation in one GPU's memory.
///
/// Freed on drop, on the device it was allocated on. `DeviceBuffer` is `Send`
/// but not `Sync`: a buffer can be handed to another thread, but concurrent use
/// would need external synchronization against the stream it is queued on.
pub struct DeviceBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    device: i32,
    _marker: PhantomData<T>,
}

// SAFETY: the pointer is an opaque device address with no thread affinity; the
// owning device is recorded so drop targets the right context.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocates `len` elements of uninitialized device memory.
    ///
    /// On allocation failure the free/total figures are re-queried so the error
    /// can say how far short the request fell.
    pub fn new(device: i32, len: usize) -> Result<Self> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(len)
            .ok_or_else(|| CudaError::Runtime {
                code: -1,
                name: "AIRUN_SIZE_OVERFLOW".into(),
                message: format!("allocation of {len} elements overflows a usize"),
            })?;

        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-parameter; `bytes` is the exact size wanted.
        let status = unsafe { ffi::airun_malloc(device, bytes as u64, &mut ptr) };
        if status != ffi::AIRUN_OK {
            let err = CudaError::from_status(status);
            if err.is_out_of_memory() {
                let (free, total) = crate::device::mem_info(device).unwrap_or((0, 0));
                return Err(CudaError::OutOfMemory {
                    device,
                    requested: bytes as u64,
                    free,
                    total,
                });
            }
            return Err(err);
        }

        Ok(Self {
            ptr,
            len,
            device,
            _marker: PhantomData,
        })
    }

    /// Allocates and zeroes `len` elements.
    pub fn zeroed(device: i32, len: usize, stream: &Stream) -> Result<Self> {
        let buf = Self::new(device, len)?;
        buf.fill_zero(stream)?;
        Ok(buf)
    }

    /// Allocates and fills from a host slice.
    pub fn from_slice(device: i32, data: &[T], stream: &Stream) -> Result<Self> {
        let mut buf = Self::new(device, data.len())?;
        buf.copy_from_host(data, stream)?;
        stream.synchronize()?;
        Ok(buf)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Device pointer at an element offset, for kernels that consume a sub-range.
    ///
    /// # Panics
    /// If `offset` is past the end of the buffer.
    pub fn offset_ptr(&self, offset: usize) -> *mut c_void {
        assert!(
            offset <= self.len,
            "offset {offset} is past the end of a {}-element buffer",
            self.len
        );
        // SAFETY: bounds-checked above; the result stays within the allocation
        // (one-past-the-end is allowed and only ever passed as a length-0 range).
        unsafe { (self.ptr as *mut u8).add(offset * std::mem::size_of::<T>()) as *mut c_void }
    }

    /// Queues a host-to-device copy. The source must stay alive until the stream
    /// is synchronized.
    pub fn copy_from_host(&mut self, data: &[T], stream: &Stream) -> Result<()> {
        if data.len() > self.len {
            return Err(CudaError::Runtime {
                code: -1,
                name: "AIRUN_COPY_TOO_LARGE".into(),
                message: format!(
                    "cannot copy {} elements into a {}-element buffer",
                    data.len(),
                    self.len
                ),
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: source and destination are both valid for the byte count, which
        // is bounded by the destination length checked above.
        ffi::check(unsafe {
            ffi::airun_memcpy_h2d(
                self.ptr,
                data.as_ptr() as *const c_void,
                (data.len() * std::mem::size_of::<T>()) as u64,
                stream.raw(),
            )
        })
    }

    /// Queues a bounded upload into a subrange. Keep the source alive and
    /// unchanged until stream completion, as with `copy_from_host`.
    pub fn copy_from_host_at(&self, offset: usize, data: &[T], stream: &Stream) -> Result<()> {
        if offset > self.len || data.len() > self.len - offset {
            return Err(CudaError::Runtime { code: -1, name: "AIRUN_COPY_TOO_LARGE".into(),
                message: "upload exceeds buffer bounds; reduce the upload or enlarge the buffer".into() });
        }
        if data.is_empty() { return Ok(()); }
        // SAFETY: destination range is checked above; caller retains the source
        // through stream completion, matching the other asynchronous copies.
        ffi::check(unsafe { ffi::airun_memcpy_h2d(self.offset_ptr(offset),
            data.as_ptr().cast(), std::mem::size_of_val(data) as u64, stream.raw()) })
    }

    /// Copies the whole buffer back to the host, synchronizing before returning.
    pub fn to_vec(&self, stream: &Stream) -> Result<Vec<T>>
    where
        T: Default,
    {
        let mut out = vec![T::default(); self.len];
        self.copy_to_host(&mut out, stream)?;
        stream.synchronize()?;
        Ok(out)
    }

    /// Queues a device-to-host copy into `dst`.
    pub fn copy_to_host(&self, dst: &mut [T], stream: &Stream) -> Result<()> {
        let n = dst.len().min(self.len);
        if n == 0 {
            return Ok(());
        }
        // SAFETY: the byte count is clamped to the smaller of the two buffers.
        ffi::check(unsafe {
            ffi::airun_memcpy_d2h(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                (n * std::mem::size_of::<T>()) as u64,
                stream.raw(),
            )
        })
    }

    /// Copies from another buffer, which may live on a different device.
    pub fn copy_from_device(&mut self, src: &DeviceBuffer<T>, stream: &Stream) -> Result<()> {
        let n = src.len.min(self.len);
        if n == 0 {
            return Ok(());
        }
        let bytes = (n * std::mem::size_of::<T>()) as u64;
        // SAFETY: byte count clamped to the smaller allocation. The peer variant
        // handles the no-P2P case internally by staging through host memory.
        ffi::check(unsafe {
            if src.device == self.device {
                ffi::airun_memcpy_d2d(self.ptr, src.ptr, bytes, stream.raw())
            } else {
                ffi::airun_memcpy_peer(
                    self.ptr,
                    self.device,
                    src.ptr,
                    src.device,
                    bytes,
                    stream.raw(),
                )
            }
        })
    }

    pub fn fill_zero(&self, stream: &Stream) -> Result<()> {
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: writes exactly the allocation's byte length.
        ffi::check(unsafe { ffi::airun_memset(self.ptr, 0, self.bytes() as u64, stream.raw()) })
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: the pointer came from airun_malloc on this device and is freed
        // exactly once. A failure here is unrecoverable and not worth panicking
        // in a destructor over -- process teardown reclaims it regardless.
        unsafe {
            ffi::airun_free(self.device, self.ptr);
        }
    }
}

impl<T> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("device", &self.device)
            .field("len", &self.len)
            .field("bytes", &(self.len * std::mem::size_of::<T>()))
            .finish()
    }
}

/// Page-locked host memory.
///
/// Transfers from pinned memory run roughly twice as fast as from pageable
/// memory and can overlap with compute, which matters when streaming tens of
/// gigabytes of weights onto the GPU.
pub struct PinnedBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: pinned host memory is ordinary addressable memory with no affinity.
unsafe impl<T: Send> Send for PinnedBuffer<T> {}

impl<T: Copy + Default> PinnedBuffer<T> {
    pub fn new(len: usize) -> Result<Self> {
        let bytes = std::mem::size_of::<T>() * len;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-parameter.
        ffi::check(unsafe { ffi::airun_host_alloc(bytes as u64, &mut ptr) })?;
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the allocation holds `len` contiguous, initialized-or-Copy elements.
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: exclusive access via &mut self; the allocation holds `len` elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: allocated by airun_host_alloc, freed exactly once.
        unsafe {
            ffi::airun_host_free(self.ptr);
        }
    }
}
