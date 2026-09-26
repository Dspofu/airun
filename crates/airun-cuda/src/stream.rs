//! Streams and events.

use crate::error::Result;
use crate::ffi;

/// A CUDA stream bound to one device.
///
/// Every airun kernel takes a stream, so per-device work can be queued
/// independently and overlapped across GPUs.
pub struct Stream {
    raw: ffi::Stream,
    device: i32,
}

// SAFETY: a stream handle is an opaque pointer with no thread affinity; CUDA
// permits enqueueing from any thread.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new(device: i32) -> Result<Self> {
        let mut raw: ffi::Stream = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-parameter.
        ffi::check(unsafe { ffi::airun_stream_create(device, &mut raw) })?;
        Ok(Self { raw, device })
    }

    /// The default (null) stream. Useful for one-off setup work.
    pub fn null(device: i32) -> Self {
        Self {
            raw: std::ptr::null_mut(),
            device,
        }
    }

    pub fn raw(&self) -> ffi::Stream {
        self.raw
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    /// Blocks until every queued operation has finished.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `raw` is either a live stream from airun_stream_create or null,
        // both of which the runtime accepts.
        ffi::check(unsafe { ffi::airun_stream_synchronize(self.raw) })
    }

    /// Makes this stream wait for `event`, which may belong to another device.
    /// This is how cross-GPU handoffs stay ordered without a full sync.
    pub fn wait_for(&self, event: &Event) -> Result<()> {
        // SAFETY: both handles are live for the duration of the call.
        ffi::check(unsafe { ffi::airun_stream_wait_event(self.raw, event.raw) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        // SAFETY: created by airun_stream_create, destroyed exactly once.
        unsafe {
            ffi::airun_stream_destroy(self.raw);
        }
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("device", &self.device)
            .field("default", &self.raw.is_null())
            .finish()
    }
}

/// A CUDA event, used for cross-stream ordering and for kernel timing.
pub struct Event {
    raw: ffi::Event,
}

// SAFETY: opaque handle with no thread affinity.
unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut raw: ffi::Event = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-parameter.
        ffi::check(unsafe { ffi::airun_event_create(&mut raw) })?;
        Ok(Self { raw })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        // SAFETY: both handles are live for the duration of the call.
        ffi::check(unsafe { ffi::airun_event_record(self.raw, stream.raw()) })
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: `raw` is a live event.
        ffi::check(unsafe { ffi::airun_event_synchronize(self.raw) })
    }

    /// Milliseconds between two recorded events.
    ///
    /// This is GPU-side timing with roughly half-microsecond resolution, which
    /// is what the benchmark harness reports rather than wall-clock timing that
    /// would include launch overhead and host scheduling noise.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        let mut ms = 0f32;
        // SAFETY: both events are live and `ms` is a valid out-parameter.
        ffi::check(unsafe { ffi::airun_event_elapsed_ms(start.raw, end.raw, &mut ms) })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        // SAFETY: created by airun_event_create, destroyed exactly once.
        unsafe {
            ffi::airun_event_destroy(self.raw);
        }
    }
}

/// Times a closure's GPU work with events.
pub fn time_gpu<F, T>(stream: &Stream, f: F) -> Result<(T, f32)>
where
    F: FnOnce() -> Result<T>,
{
    let start = Event::new()?;
    let end = Event::new()?;
    start.record(stream)?;
    let out = f()?;
    end.record(stream)?;
    end.synchronize()?;
    let ms = Event::elapsed_ms(&start, &end)?;
    Ok((out, ms))
}

/// An executable CUDA graph: a recorded sequence of kernel launches and
/// copies replayed as one call.
///
/// A decode step is hundreds of micro-kernels, and the CPU cost of issuing
/// each one leaves the GPU idle between them. Capturing the step once and
/// replaying it collapses that overhead into a single launch. The capture
/// records *pointers*, so a replay is only valid while every buffer the step
/// touched is still alive at the same address -- the runtime owns that
/// invariant and drops the graph whenever the step's shape would change.
pub struct Graph {
    raw: *mut std::ffi::c_void,
}

// SAFETY: an executable graph is an opaque handle with no thread affinity.
unsafe impl Send for Graph {}

impl Graph {
    /// Starts recording everything subsequently queued on `stream`.
    pub fn begin_capture(stream: &Stream) -> Result<()> {
        // SAFETY: the stream handle is valid for this device.
        ffi::check(unsafe { ffi::airun_capture_begin(stream.raw()) })
    }

    /// Stops recording and returns the executable graph.
    pub fn end_capture(stream: &Stream) -> Result<Self> {
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-parameter; the stream is capturing.
        ffi::check(unsafe { ffi::airun_capture_end(stream.raw(), &mut raw) })?;
        Ok(Self { raw })
    }

    /// Replays the recorded work on `stream`.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        // SAFETY: `raw` came from a successful instantiation.
        ffi::check(unsafe { ffi::airun_graph_launch(self.raw, stream.raw()) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: destroyed exactly once; null is ignored by the shim.
        unsafe { ffi::airun_graph_destroy(self.raw) };
    }
}
