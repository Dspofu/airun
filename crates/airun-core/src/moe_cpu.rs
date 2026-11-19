//! A decode step's MoE experts that live in host memory, multiplied on the
//! CPU instead of copied to the GPU.
//!
//! With experts mapped in RAM, each decode layer copied the routed experts it
//! did not hold into VRAM before computing them. On Qwen4 Flash-Next on an
//! RTX 3060 (12 GB for a 78 GB model) half of the ten routed experts miss,
//! ~11 MB a layer: ~475 us at the 23 GB/s PCIe 4.0 x16 gives, 48 times a
//! token. The CPU reads the same bytes from DDR5 faster than that, and only
//! the activations cross PCIe.
//!
//! The GPU and this thread talk through a mailbox in mapped pinned memory, so
//! the decode stays one CUDA graph: after the router, `moe_cpu_post` writes the
//! input row, the listed experts' addresses (as the GPU's tables give them)
//! and their routing weights, then a sequence number. The leader thread here
//! spins on that number, multiplies (cpu::moe_experts, on the CPU pool) and
//! writes the weighted sum, then echoes the number. Meanwhile the GPU computes
//! the resident experts; `moe_cpu_wait` then spins until the echo and adds
//! the sum in. After 50 ms without an answer (the pool's threads descheduled
//! by other programs) it hands the request's experts back to the GPU, which
//! reads them from RAM in a pass after the wait, and the layers that follow
//! skip the CPU until it has caught up; the runtime reports how often.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use airun_cuda::{DeviceBuffer, PinnedBuffer};
use airun_gguf::GgmlType;

use crate::error::{CoreError, Result};

// Mailbox layout, in bytes; mirrored by the MBOX_* defines in moe.cu.
const SEQ: usize = 0;
const LAYER: usize = 4;
const N: usize = 8;
/// Set by the leader: a request it could not serve; 0 while all is well.
const ERR: usize = 12;
/// Counted by the GPU: requests (layers) it computed itself, the CPU late.
const LATE: usize = 16;
const ADDR: usize = 64;
const W: usize = 832;
const DONE: usize = 960;
const X: usize = 1024;
/// Experts one request can list.
pub const SLOTS: usize = 32;

/// How long the leader keeps spinning after its last request before it
/// parks; a decode step's layers are well under a millisecond apart.
const HOT_US: u64 = 50_000;

/// Where one expert tensor's host-side experts are: their allocation's host
/// address, the address the GPU's table uses for the same bytes, its length,
/// and the tensor's type and expert size.
#[derive(Clone, Copy, Debug)]
pub struct TensorRef {
    pub ty: GgmlType,
    pub host: usize,
    pub dev: u64,
    pub len: usize,
    pub stride: usize,
}

impl TensorRef {
    /// The expert at table address `addr`, if it lies in this allocation.
    fn expert(&self, addr: u64) -> Option<&'static [u8]> {
        let off = addr.checked_sub(self.dev)? as usize;
        if off.checked_add(self.stride)? > self.len {
            return None;
        }
        // SAFETY: the range lies inside the pinned allocation, which outlives
        // the leader thread (the model owns it and drops the runtime first).
        Some(unsafe { std::slice::from_raw_parts((self.host + off) as *const u8, self.stride) })
    }
}

/// A MoE layer's three expert tensors.
#[derive(Clone, Copy, Debug)]
pub struct LayerRef {
    pub gate: TensorRef,
    pub up: TensorRef,
    pub down: TensorRef,
}

struct Shared {
    mbox: usize,
    /// Per model layer (indexed by the layer number the GPU posts).
    layers: Vec<Option<LayerRef>>,
    n_embd: usize,
    ff: usize,
    quit: AtomicBool,
    /// Microseconds since `t0` until which the leader spins.
    hot_until: AtomicU64,
    t0: std::time::Instant,
    park: Mutex<()>,
    wake: Condvar,
}

impl Shared {
    fn now_us(&self) -> u64 {
        self.t0.elapsed().as_micros() as u64
    }

    fn word(&self, at: usize) -> &AtomicU32 {
        // SAFETY: `at` is one of the u32 fields of the mailbox, 4-byte aligned
        // inside the pinned allocation that outlives this thread.
        unsafe { &*((self.mbox + at) as *const AtomicU32) }
    }

    fn read<T: Copy>(&self, at: usize) -> T {
        // SAFETY: as `word`; the GPU wrote the field before the sequence
        // number this thread acquired.
        unsafe { std::ptr::read_volatile((self.mbox + at) as *const T) }
    }

    /// Serves the posted request into `y`: false when it names a layer or an
    /// expert this side does not know.
    fn serve(&self, y: &mut [f32], x: &mut Vec<f32>) -> bool {
        let layer = self.read::<i32>(LAYER);
        let n = (self.read::<i32>(N).max(0) as usize).min(SLOTS);
        let Some(lr) = usize::try_from(layer).ok().and_then(|l| self.layers.get(l)).and_then(|l| l.as_ref()) else {
            return false;
        };
        let mut set = Vec::with_capacity(n);
        for j in 0..n {
            let a = |t: usize| self.read::<u64>(ADDR + 8 * (3 * j + t));
            let (Some(g), Some(u), Some(d)) = (lr.gate.expert(a(0)), lr.up.expert(a(1)), lr.down.expert(a(2))) else {
                return false;
            };
            set.push((g, u, d, self.read::<f32>(W + 4 * j)));
        }
        x.clear();
        x.extend((0..self.n_embd).map(|i| self.read::<f32>(X + 4 * i)));
        crate::cpu::moe_experts(lr.gate.ty, lr.down.ty, &set, x, self.n_embd, self.ff, y).is_ok()
    }
}

fn leader(sh: Arc<Shared>) {
    let mut y = vec![0f32; sh.n_embd];
    let mut x = Vec::with_capacity(sh.n_embd);
    let mut last = sh.word(DONE).load(Ordering::Acquire);
    // AIRUN_MOE_CPU_STATS=1: every 480 requests, the mean time serving one
    // and the experts it listed.
    let stats = std::env::var("AIRUN_MOE_CPU_STATS").is_ok_and(|v| v.trim() == "1");
    let (mut served, mut busy_us, mut experts) = (0u64, 0u64, 0u64);
    loop {
        if sh.quit.load(Ordering::Relaxed) {
            return;
        }
        let s = sh.word(SEQ).load(Ordering::Acquire);
        if s != last {
            let t = sh.now_us();
            if stats {
                experts += sh.read::<i32>(N).max(0) as u64;
            }
            if !sh.serve(&mut y, &mut x) {
                y.iter_mut().for_each(|v| *v = 0.0);
                sh.word(ERR).store(s.max(1), Ordering::Relaxed);
            }
            let out = (sh.mbox + X + 4 * sh.n_embd) as *mut f32;
            for (i, &v) in y.iter().enumerate() {
                // SAFETY: the result row lies inside the mailbox; the GPU reads
                // it only after the echo below.
                unsafe { std::ptr::write_volatile(out.add(i), v) };
            }
            sh.word(DONE).store(s, Ordering::Release);
            last = s;
            let now = sh.now_us();
            sh.hot_until.store(now + HOT_US, Ordering::Relaxed);
            crate::cpu::keep_hot(HOT_US);
            if stats {
                served += 1;
                busy_us += now - t;
                if served % 480 == 0 {
                    eprintln!("moe cpu: {:.0} us a request, {:.2} experts each (last 480)", busy_us as f64 / 480.0,
                        experts as f64 / 480.0);
                    (busy_us, experts) = (0, 0);
                }
            }
            continue;
        }
        if sh.now_us() < sh.hot_until.load(Ordering::Relaxed) {
            std::hint::spin_loop();
            continue;
        }
        // Parked until the runtime kicks it before a decode step; the timeout
        // bounds what a missed kick costs.
        let g = sh.park.lock().unwrap();
        if sh.now_us() >= sh.hot_until.load(Ordering::Relaxed) {
            let _ = sh.wake.wait_timeout(g, std::time::Duration::from_millis(2)).unwrap();
        }
    }
}

/// One GPU's mailbox and the thread that serves it.
pub struct CpuExperts {
    /// Kept alive for the GPU kernels and the leader, which use its addresses.
    _mbox: PinnedBuffer<u8>,
    /// The mailbox's device address (an integer, so the runtime stays Send).
    mbox_dev: usize,
    /// The request counter the GPU kernels keep, and whether the current
    /// layer skipped the CPU and how many more will after a timeout (three
    /// u32 on the device).
    pub seq: DeviceBuffer<u32>,
    /// How many of a request's host-side experts the GPU keeps and copies
    /// (one i32 on the device, set by moe_cpu_post).
    pub n_copy: DeviceBuffer<i32>,
    /// The share of them the GPU keeps, in 256ths (AIRUN_MOE_GPU_SHARE).
    pub keep: u32,
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
    serves: Vec<bool>,
}

impl CpuExperts {
    /// A mailbox on `device` for the layers in `layers` (`None` where the
    /// layer is not served). `n_embd` and `ff` are the experts' shape.
    pub fn new(device: i32, layers: Vec<Option<LayerRef>>, n_embd: usize, ff: usize,
               stream: &airun_cuda::stream::Stream) -> Result<Self> {
        let bytes = X + 8 * n_embd;
        let mut mbox = PinnedBuffer::<u8>::new(bytes)?;
        mbox.as_mut_slice().fill(0);
        let mbox_dev = mbox.device_ptr(device)? as usize;
        let seq = DeviceBuffer::<u32>::zeroed(device, 3, stream)?;
        let n_copy = DeviceBuffer::<i32>::zeroed(device, 1, stream)?;
        stream.synchronize()?;
        let serves = layers.iter().map(Option::is_some).collect();
        let shared = Arc::new(Shared {
            mbox: mbox.as_raw() as usize,
            layers,
            n_embd,
            ff,
            quit: AtomicBool::new(false),
            hot_until: AtomicU64::new(0),
            t0: std::time::Instant::now(),
            park: Mutex::new(()),
            wake: Condvar::new(),
        });
        // Two thirds of the logical cores (with the leader), the rest left to
        // the thread driving the GPU and the system: on an i7-13700 (8 P-cores
        // with two threads, 8 E-cores) 16 gave 26.9 tok/s on Qwen4 Flash-Next
        // where 12 gave 23.9.
        crate::cpu::set_default_threads((std::thread::available_parallelism().map_or(8, |n| n.get()) * 2 / 3).max(2));
        let sh = Arc::clone(&shared);
        let thread = std::thread::Builder::new().name("airun-moe-cpu".into()).spawn(move || leader(sh))
            .map_err(|e| CoreError::Other(format!("cannot start the CPU expert thread ({e}); set AIRUN_MOE_CPU=0")))?;
        Ok(Self { _mbox: mbox, mbox_dev, seq, n_copy, keep: gpu_keep(), shared, thread: Some(thread), serves })
    }

    /// Whether model layer `layer`'s host-side experts go through here.
    pub fn serves(&self, layer: usize) -> bool {
        self.serves.get(layer).copied().unwrap_or(false)
    }

    /// The mailbox's device address, for the kernels.
    pub fn mbox_dev(&self) -> *mut std::ffi::c_void {
        self.mbox_dev as *mut std::ffi::c_void
    }

    /// Keeps the leader (and the CPU pool) spinning for the decode step about
    /// to be queued, so the first layer's request finds it awake.
    pub fn kick(&self) {
        let sh = &self.shared;
        sh.hot_until.store(sh.now_us() + HOT_US, Ordering::Relaxed);
        crate::cpu::keep_hot(HOT_US);
        let _g = sh.park.lock().unwrap();
        sh.wake.notify_all();
    }

    /// A request the leader could not serve, if any since the last call.
    pub fn take_error(&self) -> Option<u32> {
        let e = self.shared.word(ERR).swap(0, Ordering::Relaxed);
        (e != 0).then_some(e)
    }

    /// Layers the GPU computed itself since the last call, the CPU late.
    pub fn take_late(&self) -> u32 {
        self.shared.word(LATE).swap(0, Ordering::Relaxed)
    }

}


impl Drop for CpuExperts {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Relaxed);
        {
            let _g = self.shared.park.lock().unwrap();
            self.shared.wake.notify_all();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The share of a decode layer's host-side experts that is still copied to
/// the GPU over PCIe while the CPU computes the rest, so both ways into RAM
/// are busy: AIRUN_MOE_GPU_SHARE (0..1), in 256ths.
fn gpu_keep() -> u32 {
    let share = std::env::var("AIRUN_MOE_GPU_SHARE").ok().and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(0.0);
    (share.clamp(0.0, 1.0) * 256.0).round() as u32
}

/// AIRUN_MOE_CPU=0 keeps decode's host-side experts on the copy to VRAM.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AIRUN_MOE_CPU").as_deref() != Ok("0"))
}
