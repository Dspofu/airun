//! Matmuls on the CPU, for layers whose weights stay in RAM.
//!
//! A layer the GPUs cannot hold used to be copied into VRAM before it ran,
//! every token: on an RTX 5060 behind a PCIe 3.0 x1 link (~0.8 GB/s) the
//! Qwen3.8-27B's 47 such layers (9.4 GiB) took 11 s a token. The CPU reads
//! the same weights from DDR5 at tens of GB/s, so those layers' matrices are
//! multiplied here instead and only the activations (a few KiB a token) cross
//! the bus; everything else in the layer still runs on the GPU.
//!
//! The arithmetic follows the GPU's integer path: activations quantized to
//! int8 per 32 values with an f32 scale, weights decoded to small integers,
//! integer dot products scaled per block.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use airun_gguf::GgmlType;

use crate::error::{CoreError, Result};

// --- thread pool -------------------------------------------------------------

/// The work of one call: `f(chunk)` for chunks 0..n, taken one at a time.
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
}
// SAFETY: the closure is Sync and outlives the job: `Pool::run` returns only
// once every worker is done with it.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct Shared {
    job: Mutex<Option<Arc<JobState>>>,
    generation: AtomicU64,
    wake: Condvar,
    sleep: Mutex<()>,
}

struct JobState {
    job: Job,
    next: AtomicUsize,
    /// Workers still inside this job.
    busy: AtomicUsize,
}

/// Persistent threads that split a job into chunks dynamically (a fast core
/// takes more of them), spinning briefly between jobs: a decode step runs a
/// few hundred matmuls, too many to pay a wake-up from sleep for each.
pub struct Pool {
    shared: Arc<Shared>,
    workers: usize,
}

/// How long an idle worker spins before it sleeps: AIRUN_CPU_SPIN_US, else
/// 200 us. Longer is much worse: at 5 ms the 27B Q4_K_M with 7.5 GB in RAM
/// went from 3.0 to 0.9 tok/s, 23 spinning threads starving the thread that
/// drives the GPU and copies the activations.
fn spin() -> std::time::Duration {
    static SPIN: OnceLock<std::time::Duration> = OnceLock::new();
    *SPIN.get_or_init(|| std::time::Duration::from_micros(std::env::var("AIRUN_CPU_SPIN_US").ok()
        .and_then(|v| v.trim().parse().ok()).unwrap_or(200)))
}

/// Below this many weight bytes a matmul runs on the calling thread alone:
/// waking the pool costs more than the work (a 138 KB delta-net projection
/// took 205 us across the pool).
const SMALL_MATMUL: usize = 1 << 20;

/// `f(i)` for i in 0..n: on the pool, or here when the matmul is small.
fn run_sized(n: usize, bytes: usize, f: &(dyn Fn(usize) + Sync + '_)) {
    if bytes < SMALL_MATMUL {
        (0..n).for_each(f);
    } else {
        pool().run(n, f);
    }
}

impl Pool {
    fn new(threads: usize) -> Self {
        keep_full_speed();
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            generation: AtomicU64::new(0),
            wake: Condvar::new(),
            sleep: Mutex::new(()),
        });
        let workers = threads.saturating_sub(1);
        for id in 0..workers {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new().name(format!("airun-cpu-{id}")).spawn(move || worker(&shared))
                .expect("spawning a CPU matmul thread");
        }
        Self { shared, workers }
    }

    /// Calls `f(i)` for every i in 0..n across the threads and this one.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync + '_)) {
        if n == 0 {
            return;
        }
        if n == 1 || self.workers == 0 {
            (0..n).for_each(f);
            return;
        }
        // One job at a time: the CPU expert thread (moe_cpu.rs) and the thread
        // driving the GPU may both come here.
        static ONE: Mutex<()> = Mutex::new(());
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: only the lifetime is erased; this call returns only once
        // every worker has left the job.
        let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let state = Arc::new(JobState { job: Job { f, n }, next: AtomicUsize::new(0), busy: AtomicUsize::new(0) });
        *self.shared.job.lock().unwrap() = Some(Arc::clone(&state));
        self.shared.generation.fetch_add(1, Ordering::Release);
        {
            let _g = self.shared.sleep.lock().unwrap();
            self.shared.wake.notify_all();
        }
        take_chunks(&state);
        // Late workers find the job gone or its chunks taken; wait for those
        // already inside.
        *self.shared.job.lock().unwrap() = None;
        while state.busy.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }
}

fn take_chunks(state: &JobState) {
    // SAFETY: see Job.
    let f = unsafe { &*state.job.f };
    loop {
        let i = state.next.fetch_add(1, Ordering::Relaxed);
        if i >= state.job.n {
            break;
        }
        f(i);
    }
}

/// Microseconds on [`clock_us`] until which idle workers keep spinning.
static HOT_UNTIL_US: AtomicU64 = AtomicU64::new(0);

fn clock_us() -> u64 {
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// Keeps the pool's idle workers spinning for the next `us` microseconds
/// instead of sleeping after [`spin`]: the CPU experts of a decode step
/// (moe_cpu.rs) come a few hundred microseconds apart, layer after layer,
/// and a wake-up from sleep cost more than the work.
pub fn keep_hot(us: u64) {
    HOT_UNTIL_US.fetch_max(clock_us() + us, Ordering::Relaxed);
}

fn hot() -> bool {
    clock_us() < HOT_UNTIL_US.load(Ordering::Relaxed)
}

/// Opts the process out of Windows 11's power throttling (EcoQoS). A console
/// process without a foreground window is throttled a few seconds after it
/// starts: its threads go to the E-cores at a capped clock. On an i7-13700
/// a decode step's CPU experts (moe_cpu.rs) went from ~0.26 to ~3.6 ms a
/// layer once that kicked in, and plain reads of RAM from 67 to 9 GB/s
/// (tests/cpu_bench.rs, rounds and raw, with AIRUN_BENCH_WAIT_S=6); the host
/// side of every decode step (sampling, PLE reads, graph launches) ran at that
/// clock too. Called once at startup; a no-op elsewhere.
pub fn keep_full_speed() {
    #[cfg(windows)]
    {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            #[repr(C)]
            struct PowerThrottlingState {
                version: u32,
                control_mask: u32,
                state_mask: u32,
            }
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn GetCurrentProcess() -> *mut std::ffi::c_void;
                fn SetProcessInformation(process: *mut std::ffi::c_void, class: i32, info: *const std::ffi::c_void,
                                         size: u32) -> i32;
            }
            // PROCESS_POWER_THROTTLING_CURRENT_VERSION, the EXECUTION_SPEED
            // bit under control and cleared: never throttle execution speed.
            let state = PowerThrottlingState { version: 1, control_mask: 1, state_mask: 0 };
            // SAFETY: class 4 is ProcessPowerThrottling, whose information is a
            // PROCESS_POWER_THROTTLING_STATE of that size; the pseudo-handle
            // needs no closing. A failure (older Windows) leaves things as they were.
            unsafe {
                SetProcessInformation(GetCurrentProcess(), 4, &state as *const PowerThrottlingState as *const _,
                    std::mem::size_of::<PowerThrottlingState>() as u32);
            }
        });
    }
}

/// Sets flush-to-zero and denormals-are-zero for this thread's SSE/AVX
/// arithmetic. A subnormal operand or result costs a microcode assist on
/// x86, ~18x an ordinary multiply-add (tests/cpu_bench.rs, exp_cost); a
/// block's tiny scale times a tiny activation scale can reach one. Anything
/// that small is far below the int8 rounding already in the products.
pub fn flush_denormals() {
    #[cfg(target_arch = "x86_64")]
    {
        let mut csr: u32 = 0;
        // SAFETY: stmxcsr/ldmxcsr only read and write this thread's MXCSR;
        // bits 15 (FTZ) and 6 (DAZ) change no exception masks.
        unsafe {
            std::arch::asm!("stmxcsr [{p}]", p = in(reg) &mut csr, options(nostack));
            csr |= 0x8040;
            std::arch::asm!("ldmxcsr [{p}]", p = in(reg) &csr, options(nostack));
        }
    }
}

fn worker(shared: &Shared) {
    flush_denormals();
    let mut seen = 0u64;
    loop {
        // Spin for a while, then sleep until the next job.
        let start = std::time::Instant::now();
        while shared.generation.load(Ordering::Acquire) == seen {
            if start.elapsed() > spin() && !hot() {
                let g = shared.sleep.lock().unwrap();
                if shared.generation.load(Ordering::Acquire) == seen {
                    let _g = shared.wake.wait_timeout(g, std::time::Duration::from_millis(50)).unwrap();
                }
            } else {
                std::hint::spin_loop();
            }
        }
        seen = shared.generation.load(Ordering::Acquire);
        let state = {
            let job = shared.job.lock().unwrap();
            match job.as_ref() {
                Some(s) => {
                    s.busy.fetch_add(1, Ordering::AcqRel);
                    Arc::clone(s)
                }
                None => continue,
            }
        };
        take_chunks(&state);
        state.busy.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The process's CPU matmul threads: AIRUN_CPU_THREADS, else one per logical
/// core (on a 13700, 8 P-cores x 2 + 8 E-cores).
pub fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let all = std::thread::available_parallelism().map_or(4, |n| n.get());
        let n = std::env::var("AIRUN_CPU_THREADS").ok().and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| match DEFAULT_THREADS.load(Ordering::Relaxed) {
                0 => all,
                d => d.min(all),
            });
        Pool::new(n.max(1))
    })
}

/// The pool's size when AIRUN_CPU_THREADS is not set, if set before the pool
/// first runs (0: one per logical core).
static DEFAULT_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Sizes the pool for work that spins between short jobs (moe_cpu.rs): a
/// thread per logical core, all spinning, left none to the thread driving
/// the GPU or to the rest of the machine, and a preempted worker holds up
/// the whole job.
pub fn set_default_threads(n: usize) {
    DEFAULT_THREADS.store(n, Ordering::Relaxed);
}

/// Whether some layer's matmuls run here (set once at load); the runtime then
/// issues those passes without CUDA graphs, which cannot hold host work.
static IN_USE: AtomicBool = AtomicBool::new(false);

pub fn set_in_use() {
    IN_USE.store(true, Ordering::Relaxed);
}

pub fn in_use() -> bool {
    IN_USE.load(Ordering::Relaxed)
}

/// Whether the forward pass under way multiplies the RAM layers here: one of
/// at most AIRUN_CPU_MAX_TOKENS (64) tokens. A longer one (a prompt chunk)
/// copies them into VRAM as before: the CPU kernel is compute-bound at many
/// tokens, while the copy costs the same for any batch -- on PCIe 3.0 x1 the
/// 27B's 47 RAM layers break even at ~150 tokens.
static PASS: AtomicBool = AtomicBool::new(false);

pub fn set_pass(n_tokens: usize) {
    static MAX: OnceLock<usize> = OnceLock::new();
    let max = *MAX.get_or_init(|| std::env::var("AIRUN_CPU_MAX_TOKENS").ok()
        .and_then(|v| v.trim().parse().ok()).unwrap_or(64));
    PASS.store(in_use() && n_tokens <= max, Ordering::Relaxed);
}

pub fn pass_on_cpu() -> bool {
    PASS.load(Ordering::Relaxed) && !RESIDENT.load(Ordering::Relaxed)
}

/// Whether the layer running now is a RAM layer whose weights are already in
/// its GPU's staging VRAM (the last one a prompt chunk copied there): it then
/// runs on the GPU even in a CPU pass. The staging buffer would otherwise sit
/// idle while decoding.
static RESIDENT: AtomicBool = AtomicBool::new(false);

pub fn set_layer_resident(resident: bool) {
    RESIDENT.store(resident, Ordering::Relaxed);
}

// --- activations ---------------------------------------------------------------

/// `x` (n_tokens x k floats) as int8 per 32 values with one f32 scale each,
/// as the GPU's quantize_act does.
fn quantize(x: &[f32], q: &mut Vec<i8>, d: &mut Vec<f32>) {
    quantize_sums(x, q, d, None);
}

/// [`quantize`], plus each block's sum of int8 values in `sums`: what the
/// K-quants' per-block minimum multiplies.
fn quantize_sums(x: &[f32], q: &mut Vec<i8>, d: &mut Vec<f32>, mut sums: Option<&mut Vec<i32>>) {
    q.resize(x.len(), 0);
    d.resize(x.len() / 32, 0.0);
    if let Some(s) = sums.as_deref_mut() {
        s.resize(x.len() / 32, 0);
    }
    for (b, (xs, qs)) in x.chunks_exact(32).zip(q.chunks_exact_mut(32)).enumerate() {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = amax / 127.0;
        d[b] = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut sum = 0i32;
        for (o, v) in qs.iter_mut().zip(xs) {
            *o = (v * inv).round() as i8;
            sum += i32::from(*o);
        }
        if let Some(s) = sums.as_deref_mut() {
            s[b] = sum;
        }
    }
}

/// How far ahead of the block in use the AVX2 kernels prefetch, in bytes.
fn prefetch_distance() -> usize {
    static D: OnceLock<usize> = OnceLock::new();
    *D.get_or_init(|| std::env::var("AIRUN_CPU_PREFETCH").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024))
}

/// Whether the AVX2 integer kernels can run here.
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The ggml block types with an AVX2 integer kernel here, and their
/// elements per block.
fn int_kernel(ty: GgmlType) -> Option<usize> {
    match ty {
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => Some(256),
        GgmlType::IQ4_XS | GgmlType::IQ2_S | GgmlType::IQ3_XXS | GgmlType::IQ3_S => Some(256),
        GgmlType::Q8_0 | GgmlType::IQ4_NL => Some(32),
        GgmlType::Q2_0 => Some(64),
        _ => None,
    }
}


/// The eight signs of each 7-bit IQ sign code (the eighth is the parity, as
/// ksigns_iq2xs), as +1/-1 bytes for `_mm256_sign_epi8`.
fn ksigns64() -> &'static [u64; 128] {
    static T: OnceLock<[u64; 128]> = OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|i| {
        let s = airun_gguf::dequant::ksigns_iq2xs(i as u8);
        (0..8).fold(0u64, |acc, j| acc | (if s >> j & 1 == 1 { 0xFFu64 } else { 0x01 }) << (8 * j))
    }))
}

thread_local! {
    static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// AIRUN_CPU_STATS=1: every 1000 matmuls, the time spent in them and the
/// weight bytes they read, so a slow decode shows whether the CPU kernels or
/// the round trips around them take the time.
struct Stats(Option<(std::time::Instant, u64)>);

static STAT_NS: AtomicU64 = AtomicU64::new(0);
static STAT_BYTES: AtomicU64 = AtomicU64::new(0);
static STAT_CALLS: AtomicU64 = AtomicU64::new(0);
static STAT_SINCE: Mutex<Option<std::time::Instant>> = Mutex::new(None);

impl Stats {
    fn start(bytes: usize) -> Self {
        static ON: OnceLock<bool> = OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("AIRUN_CPU_STATS").is_ok_and(|v| v == "1"));
        Stats(on.then(|| (std::time::Instant::now(), bytes as u64)))
    }
}

impl Drop for Stats {
    fn drop(&mut self) {
        let Some((t, bytes)) = self.0 else { return };
        let ns = STAT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed) + t.elapsed().as_nanos() as u64;
        let b = STAT_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
        let calls = STAT_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        if calls % 1000 == 0 {
            let mut since = STAT_SINCE.lock().unwrap();
            let wall = since.map_or(0.0, |s| s.elapsed().as_secs_f64());
            *since = Some(std::time::Instant::now());
            eprintln!("cpu stats: {calls} matmuls, {:.0} ms in them ({:.1} GB/s), {:.0} ms wall since the last report",
                ns as f64 / 1e6, b as f64 / ns as f64, wall * 1e3);
            STAT_NS.store(0, Ordering::Relaxed);
            STAT_BYTES.store(0, Ordering::Relaxed);
        }
    }
}

struct Ptr<T>(*mut T);
// SAFETY: every chunk writes a disjoint set of output elements.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

/// `y[t][r] = sum_i w[r][i] * x[t][i]` for `rows` rows of `row_bytes` bytes of
/// type `ty` (`k` elements each) against `n` tokens, all on the CPU threads.
pub fn matvec(ty: GgmlType, w: &[u8], row_bytes: usize, k: usize, rows: usize, x: &[f32], n: usize,
              y: &mut [f32]) -> Result<()> {
    if x.len() < n * k || y.len() < n * rows || w.len() < rows * row_bytes {
        return Err(CoreError::Other("a CPU matmul got buffers smaller than its shape; this is a bug".into()));
    }
    let _stats = Stats::start(rows * row_bytes);
    let out = Ptr(y.as_mut_ptr());
    let out = &out;
    // Rows per chunk: enough work per grab, enough chunks to balance cores.
    const CHUNK: usize = 32;
    let chunks = rows.div_ceil(CHUNK);
    match ty {
        GgmlType::NVFP4 if k % 64 == 0 => {
            let mut q = Vec::new();
            let mut d = Vec::new();
            quantize(&x[..n * k], &mut q, &mut d);
            let (q, d) = (&q, &d);
            run_sized(chunks, rows * row_bytes, &|c| {
                for r in c * CHUNK..((c + 1) * CHUNK).min(rows) {
                    let wr = &w[r * row_bytes..(r + 1) * row_bytes];
                    nvfp4_row(wr, k, q, d, n, |t, v| {
                        // SAFETY: (t, r) belongs to this chunk's rows only.
                        unsafe { *out.0.add(t * rows + r) = v };
                    });
                }
            });
        }
        // K-quants and Q8_0, the bulk of a GGUF Q4_K_M/Q5_K_M/Q6_K/Q8_0:
        // decoded to small integers in registers and multiplied as int8, like
        // the NVFP4 kernel. The generic path below dequantized every row to
        // f32 first and ran ~30x slower (Qwen3.6-27B Q4_K_M, 7.5 GB in RAM,
        // on an i7-13700: 0.2 tok/s).
        _ if int_kernel(ty).is_some_and(|e| k % e == 0) && has_avx2() => {
            let mut q = Vec::new();
            let mut d = Vec::new();
            let mut sums = Vec::new();
            quantize_sums(&x[..n * k], &mut q, &mut d, Some(&mut sums));
            // Each activation block's scale times its int8 sum: what the
            // K-quants' minimums multiply.
            let dsum: Vec<f32> = d.iter().zip(&sums).map(|(&s, &v)| s * v as f32).collect();
            let (q, d, dsum) = (&q, &d, &dsum);
            run_sized(chunks, rows * row_bytes, &|c| {
                for r in c * CHUNK..((c + 1) * CHUNK).min(rows) {
                    let wr = &w[r * row_bytes..(r + 1) * row_bytes];
                    let mut t0 = 0;
                    while t0 < n {
                        let m = (n - t0).min(4);
                        #[cfg(target_arch = "x86_64")]
                        // SAFETY: has_avx2() checked AVX2 and FMA; the row and
                        // the activations hold k elements per token.
                        let v = unsafe {
                            match ty {
                                GgmlType::Q4_K => avx2::q4k_row4(wr, k, q, d, dsum, t0, m),
                                GgmlType::Q5_K => avx2::q5k_row4(wr, k, q, d, dsum, t0, m),
                                GgmlType::Q6_K => avx2::q6k_row4(wr, k, q, d, t0, m),
                                GgmlType::IQ4_NL => avx2::iq4nl_row4(wr, k, q, d, t0, m),
                                GgmlType::IQ4_XS => avx2::iq4xs_row4(wr, k, q, d, t0, m),
                                GgmlType::IQ2_S => avx2::iq2s_row4(wr, k, q, d, t0, m),
                                GgmlType::IQ3_XXS => avx2::iq3xxs_row4(wr, k, q, d, t0, m),
                                GgmlType::IQ3_S => avx2::iq3s_row4(wr, k, q, d, t0, m),
                                GgmlType::Q2_0 => avx2::q2_0_row4(wr, k, q, d, t0, m),
                                _ => avx2::q8_0_row4(wr, k, q, d, t0, m),
                            }
                        };
                        #[cfg(not(target_arch = "x86_64"))]
                        let v = [0f32; 4];
                        for (i, &val) in v.iter().enumerate().take(m) {
                            // SAFETY: (t, r) belongs to this chunk's rows only.
                            unsafe { *out.0.add((t0 + i) * rows + r) = val };
                        }
                        t0 += m;
                    }
                }
            });
        }
        _ => {
            let elems = ty.layout().map_or(1, |l| l.elements as usize);
            if k % elems != 0 {
                return Err(CoreError::Other(format!("no CPU matmul for {ty} rows of {k}; run with AIRUN_RAM_ON_GPU=1")));
            }
            let failed = AtomicBool::new(false);
            run_sized(chunks, rows * row_bytes, &|c| {
                ROW.with(|row| {
                    let mut row = row.borrow_mut();
                    row.resize(k, 0.0);
                    for r in c * CHUNK..((c + 1) * CHUNK).min(rows) {
                        let wr = &w[r * row_bytes..(r + 1) * row_bytes];
                        if airun_gguf::dequant::dequantize(ty, wr, k, &mut row).is_err() {
                            failed.store(true, Ordering::Relaxed);
                            return;
                        }
                        for t in 0..n {
                            let v = dot_f32(&row, &x[t * k..(t + 1) * k]);
                            // SAFETY: (t, r) belongs to this chunk's rows only.
                            unsafe { *out.0.add(t * rows + r) = v };
                        }
                    }
                });
            });
            if failed.load(Ordering::Relaxed) {
                return Err(CoreError::Other(format!("could not decode {ty} weights on the CPU; run with AIRUN_RAM_ON_GPU=1")));
            }
        }
    }
    Ok(())
}

/// One row of an integer-kernel type against one token's int8 activations.
/// The caller checked [`moe_expert_type`] and that `k` is a whole number of
/// blocks; the K-quants' minimums are not handled (no expert type has one).
fn int_row1(ty: GgmlType, w: &[u8], k: usize, q: &[i8], d: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: moe_expert_type checked AVX2 and FMA; `w` holds one row of k
    // elements, `q` and `d` one token of k.
    unsafe {
        let v = match ty {
            GgmlType::IQ4_NL => avx2::iq4nl_row4(w, k, q, d, 0, 1),
            GgmlType::IQ4_XS => avx2::iq4xs_row4(w, k, q, d, 0, 1),
            GgmlType::IQ2_S => avx2::iq2s_row4(w, k, q, d, 0, 1),
            GgmlType::IQ3_XXS => avx2::iq3xxs_row4(w, k, q, d, 0, 1),
            GgmlType::IQ3_S => avx2::iq3s_row4(w, k, q, d, 0, 1),
            GgmlType::Q2_0 => avx2::q2_0_row4(w, k, q, d, 0, 1),
            GgmlType::Q6_K => avx2::q6k_row4(w, k, q, d, 0, 1),
            _ => avx2::q8_0_row4(w, k, q, d, 0, 1),
        };
        v[0]
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ty, w, k, q, d);
        0.0
    }
}

/// Types [`moe_experts`] multiplies.
pub fn moe_expert_type(ty: GgmlType) -> bool {
    has_avx2() && matches!(ty, GgmlType::IQ4_NL | GgmlType::IQ4_XS | GgmlType::IQ2_S | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S | GgmlType::Q2_0 | GgmlType::Q6_K | GgmlType::Q8_0)
}

/// One token through routed experts held in RAM, on the CPU threads:
/// `y = sum_j w_j * down_j(silu(gate_j x) * up_j x)`. Expert j is
/// `(gate, up, down, w_j)`: gate and up `ff` rows of `k` (type `gty`), down
/// `k` rows of `ff` (type `dty`). All the experts' rows go to the pool as one
/// job per projection, so the threads stream every expert's weights at once;
/// one 640-row expert matrix alone is too little work to spread.
pub fn moe_experts(gty: GgmlType, dty: GgmlType, experts: &[(&[u8], &[u8], &[u8], f32)], x: &[f32], k: usize,
                   ff: usize, y: &mut [f32]) -> Result<()> {
    let (Some(ge), Some(de)) = (int_kernel(gty), int_kernel(dty)) else {
        return Err(CoreError::Other(format!("no CPU expert kernel for {gty}/{dty}; set AIRUN_MOE_CPU=0")));
    };
    if !moe_expert_type(gty) || !moe_expert_type(dty) || k % ge != 0 || ff % de != 0 || ff % 32 != 0
        || x.len() < k || y.len() < k {
        return Err(CoreError::Other(format!("CPU experts cannot take {gty}/{dty} at {k} x {ff}; set AIRUN_MOE_CPU=0")));
    }
    let row = |ty: GgmlType, n: usize| ty.layout().map_or(0, |l| n / l.elements as usize * l.bytes as usize);
    let (g_row, d_row) = (row(gty, k), row(dty, ff));
    if experts.iter().any(|e| e.0.len() < ff * g_row || e.1.len() < ff * g_row || e.2.len() < k * d_row) {
        return Err(CoreError::Other("a CPU expert is smaller than its shape; this is a bug".into()));
    }
    let n = experts.len();
    for v in y[..k].iter_mut() {
        *v = 0.0;
    }
    if n == 0 {
        return Ok(());
    }
    // The calling thread takes chunks too (the pool's workers already flush).
    flush_denormals();
    let _stats = Stats::start(n * (2 * ff * g_row + k * d_row));
    let (mut q, mut d) = (vec![0i8; k], vec![0f32; k / 32]);
    quantize_into(&x[..k], &mut q, &mut d);
    // One pool job: the gate/up chunks of every expert (GU rows each), then
    // the down chunks (DN rows of one expert each). The pool hands chunks out
    // in order, so the gate/up ones are all taken first; a down chunk waits
    // only for its own expert's h, which the thread finishing that expert's
    // last gate/up chunk quantizes and publishes. Two jobs with a barrier
    // between them left threads idle at the barrier and paid the dispatch
    // twice, a few hundred microseconds apart, 48 times a token.
    const GU: usize = 16;
    const DN: usize = 32;
    let per = ff / GU;
    let dper = k.div_ceil(DN);
    let mut h = vec![0f32; n * ff];
    let mut hq = vec![0i8; n * ff];
    let mut hd = vec![0f32; n * ff / 32];
    let mut part = vec![0f32; n * k];
    let left: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(per)).collect();
    let ready: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
    {
        let (hp, hqp, hdp, pp) = (Ptr(h.as_mut_ptr()), Ptr(hq.as_mut_ptr()), Ptr(hd.as_mut_ptr()), Ptr(part.as_mut_ptr()));
        let (hp, hqp, hdp, pp, q, d, left, ready) = (&hp, &hqp, &hdp, &pp, &q, &d, &left, &ready);
        pool().run(n * per + n * dper, &|c| {
            if c < n * per {
                let (j, r0) = (c / per, c % per * GU);
                let (gw, uw) = (experts[j].0, experts[j].1);
                for r in r0..r0 + GU {
                    let g = int_row1(gty, &gw[r * g_row..(r + 1) * g_row], k, q, d);
                    let u = int_row1(gty, &uw[r * g_row..(r + 1) * g_row], k, q, d);
                    // SAFETY: (j, r) belongs to this chunk only.
                    unsafe { *hp.0.add(j * ff + r) = g / (1.0 + (-g).exp()) * u };
                }
                if left[j].fetch_sub(1, Ordering::AcqRel) == 1 {
                    // SAFETY: every chunk of expert j has written its rows
                    // (the AcqRel count orders them before this), and only
                    // this thread touches hq/hd of j until `ready`.
                    unsafe {
                        let hj = std::slice::from_raw_parts(hp.0.add(j * ff), ff);
                        let qj = std::slice::from_raw_parts_mut(hqp.0.add(j * ff), ff);
                        let dj = std::slice::from_raw_parts_mut(hdp.0.add(j * ff / 32), ff / 32);
                        quantize_into(hj, qj, dj);
                    }
                    ready[j].store(true, Ordering::Release);
                }
            } else {
                let c2 = c - n * per;
                let (j, r0) = (c2 / dper, c2 % dper * DN);
                while !ready[j].load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                // SAFETY: published by `ready`; read-only from here on.
                let (qj, dj) = unsafe {
                    (std::slice::from_raw_parts(hqp.0.add(j * ff) as *const i8, ff),
                     std::slice::from_raw_parts(hdp.0.add(j * ff / 32) as *const f32, ff / 32))
                };
                let (dw, wj) = (experts[j].2, experts[j].3);
                for r in r0..(r0 + DN).min(k) {
                    let v = wj * int_row1(dty, &dw[r * d_row..(r + 1) * d_row], ff, qj, dj);
                    // SAFETY: (j, r) belongs to this chunk only.
                    unsafe { *pp.0.add(j * k + r) = v };
                }
            }
        });
    }
    for (r, out) in y[..k].iter_mut().enumerate() {
        *out = (0..n).map(|j| part[j * k + r]).sum();
    }
    Ok(())
}

/// [`quantize`] into slices the caller sized (x.len() values, x.len() / 32
/// scales). With AVX2 the rounding is to nearest even (the GPU's
/// __float2int_rn) instead of away from zero; they differ only on exact ties.
/// A layer's 2560-value input took ~13 us of scalar roundf calls per CPU
/// expert request, on the critical path.
fn quantize_into(x: &[f32], q: &mut [i8], d: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected; the slices hold whole groups of 32.
        unsafe { avx2::quantize32(x, q, d) };
        return;
    }
    for ((xs, qs), dv) in x.chunks_exact(32).zip(q.chunks_exact_mut(32)).zip(d.iter_mut()) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = amax / 127.0;
        *dv = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (o, v) in qs.iter_mut().zip(xs) {
            *o = (v * inv).round() as i8;
        }
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // Eight partial sums: auto-vectorizes and keeps the f32 error near the
    // GPU's tree reduction.
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] = x[i].mul_add(y[i], acc[i]);
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

// --- NVFP4 ----------------------------------------------------------------------

/// UE4M3 sub-block scales, all 256 codes.
fn ue4m3_table() -> &'static [f32; 256] {
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|i| airun_gguf::dequant::ue4m3_to_f32(i as u8)))
}

/// One NVFP4 row (64-element blocks of 4 UE4M3 scales and 32 bytes of
/// nibbles; run s of 16 takes bytes s*8.., low nibbles first) against every
/// token's int8 activations; `emit(t, value)` for each token.
fn nvfp4_row(w: &[u8], k: usize, q: &[i8], d: &[f32], n: usize, mut emit: impl FnMut(usize, f32)) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // Tokens four at a time: each block's weights decoded once for them.
            let mut t0 = 0;
            while t0 < n {
                let m = (n - t0).min(4);
                // SAFETY: AVX2 and FMA were just detected.
                let v = unsafe { avx2::nvfp4_row4(w, k, q, d, t0, m) };
                for (i, &val) in v.iter().enumerate().take(m) {
                    emit(t0 + i, val);
                }
                t0 += m;
            }
            return;
        }
    }
    let table = ue4m3_table();
    for t in 0..n {
        let (qt, dt) = (&q[t * k..(t + 1) * k], &d[t * k / 32..(t + 1) * k / 32]);
        let mut acc = 0f32;
        for (b, blk) in w.chunks_exact(36).take(k / 64).enumerate() {
            for s in 0..4 {
                let qs = &blk[4 + s * 8..4 + s * 8 + 8];
                let xs = &qt[b * 64 + s * 16..b * 64 + s * 16 + 16];
                let mut dot = 0i32;
                for i in 0..8 {
                    dot += i32::from(airun_gguf::dequant::KVALUES_MXFP4[usize::from(qs[i] & 0x0F)]) * i32::from(xs[i]);
                    dot += i32::from(airun_gguf::dequant::KVALUES_MXFP4[usize::from(qs[i] >> 4)]) * i32::from(xs[i + 8]);
                }
                acc += dot as f32 * table[usize::from(blk[s])] * dt[(b * 64 + s * 16) / 32];
            }
        }
        emit(t, acc);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx2 {
    use std::arch::x86_64::*;

    /// Calls `$f::<M>` with M = `$m` (1..=4) as a constant: with a run-time
    /// token count the per-token accumulators (`acc[i]`, `i < m`) lived on
    /// the stack, each FMA waiting on the previous one's store, and four
    /// tokens cost 1.7x one instead of reading the weights once for all.
    macro_rules! by_m {
        ($m:expr, $f:ident $(::<$($g:tt),*>)? ($($a:expr),*)) => {
            match $m {
                1 => $f::<$($($g,)*)? 1>($($a),*),
                2 => $f::<$($($g,)*)? 2>($($a),*),
                3 => $f::<$($($g,)*)? 3>($($a),*),
                _ => $f::<$($($g,)*)? 4>($($a),*),
            }
        };
    }


    /// Tokens t0..t0+m (m <= 4) of one NVFP4 row; see nvfp4_row.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn nvfp4_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, nvfp4_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn nvfp4_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let table = super::ue4m3_table();
        let lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                   0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        let low = _mm_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); M];
        let blocks = k / 64;
        for b in 0..blocks {
            let blk = w.as_ptr().add(b * 36);
            if b % 2 == 0 {
                prefetch(blk, 72);
            }
            let s = [table[*blk as usize], table[*blk.add(1) as usize], table[*blk.add(2) as usize],
                     table[*blk.add(3) as usize]];
            // Two halves of 32: runs (0,1) then (2,3), each one activation block.
            for h in 0..2 {
                let raw = _mm_loadu_si128(blk.add(4 + h * 16) as *const __m128i);
                let lo = _mm_and_si128(raw, low);
                let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), low);
                // Lane 0: run 2h (its 8 low nibbles, then its 8 high ones);
                // lane 1: run 2h + 1.
                let idx = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
                let wv = _mm256_shuffle_epi8(lut, idx);
                let aw = _mm256_sign_epi8(wv, wv);
                let scale = _mm256_setr_m128(_mm_set1_ps(s[2 * h]), _mm_set1_ps(s[2 * h + 1]));
                let at = b * 64 + h * 32;
                for i in 0..M {
                    let t = t0 + i;
                    let x = _mm256_loadu_si256(q.as_ptr().add(t * k + at) as *const __m256i);
                    let xs = _mm256_sign_epi8(x, wv);
                    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, xs), ones);
                    let dx = _mm256_set1_ps(*d.get_unchecked((t * k + at) / 32));
                    acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_mul_ps(scale, dx), acc[i]);
                }
            }
        }
        out4(&acc, &[0f32; M])
    }

    #[inline(always)]
    unsafe fn out4<const M: usize>(acc: &[__m256; M], sub: &[f32; M]) -> [f32; 4] {
        let mut out = [0f32; 4];
        for i in 0..M {
            out[i] = hsum(acc[i]) - sub[i];
        }
        out
    }

    /// Asks for the cache lines `PREFETCH` bytes past `p`, enough of them to
    /// cover `bytes` (one block's worth). Each thread streams its own rows,
    /// and the hardware prefetcher alone left the 1-token Q4_K matmul at 22.5
    /// GB/s of a ~27 GB/s DDR4 ceiling (i5-10400F): 26.3 with this. The
    /// address may run past the matrix; a prefetch never faults.
    #[inline(always)]
    unsafe fn prefetch(p: *const u8, bytes: usize) {
        let at = p.wrapping_add(super::prefetch_distance());
        let mut o = 0;
        while o < bytes {
            _mm_prefetch::<_MM_HINT_T0>(at.wrapping_add(o) as *const i8);
            o += 64;
        }
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        _mm_cvtss_f32(s1)
    }

    #[inline(always)]
    unsafe fn f16_at(p: *const u8) -> f32 {
        half::f16::from_bits(u16::from_le_bytes([*p, *p.add(1)])).to_f32()
    }

    /// 32 unsigned weights (0..63) against 32 int8 activations: 8 i32 sums,
    /// lanes 0-3 over bytes 0-15 and lanes 4-7 over bytes 16-31.
    #[inline(always)]
    unsafe fn dot_u8(w: __m256i, x: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1))
    }

    /// Same for signed weights (|w| <= 127).
    #[inline(always)]
    unsafe fn dot_i8(w: __m256i, x: __m256i) -> __m256i {
        dot_u8(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w))
    }

    #[inline(always)]
    unsafe fn xq(q: &[i8], at: usize) -> __m256i {
        _mm256_loadu_si256(q.as_ptr().add(at) as *const __m256i)
    }

    /// Byte-wise right shift of a vector of bytes by `s` (0..8): AVX2 shifts
    /// only 16-bit lanes, so the caller masks each byte's own bits after.
    #[inline(always)]
    unsafe fn srl(v: __m256i, s: usize) -> __m256i {
        _mm256_srl_epi16(v, _mm_cvtsi32_si128(s as i32))
    }

    /// Q6_K's bit pair `s` of each qh byte, as bits 4-5.
    #[inline(always)]
    unsafe fn high2(qh: __m256i, s: usize, three: __m256i) -> __m256i {
        _mm256_slli_epi16(_mm256_and_si256(srl(qh, s), three), 4)
    }

    /// The eight 6-bit scales and minimums of a Q4_K/Q5_K super-block
    /// (ggml's get_scale_min_k4).
    #[inline(always)]
    unsafe fn k4_scales(s: *const u8) -> ([f32; 8], [f32; 8]) {
        let b = |i: usize| *s.add(i);
        let mut sc = [0f32; 8];
        let mut mn = [0f32; 8];
        for j in 0..8 {
            let (a, m) = if j < 4 {
                (b(j) & 63, b(j + 4) & 63)
            } else {
                ((b(j + 4) & 0x0F) | ((b(j - 4) >> 6) << 4), (b(j + 4) >> 4) | ((b(j) >> 6) << 4))
            };
            sc[j] = f32::from(a);
            mn[j] = f32::from(m);
        }
        (sc, mn)
    }

    /// Eight vectors of eight i32 sums to one vector whose lane j is the sum
    /// of all of `p[j]`'s lanes.
    #[inline(always)]
    unsafe fn reduce8(p: &[__m256i; 8]) -> __m256i {
        let a = _mm256_hadd_epi32(_mm256_hadd_epi32(p[0], p[1]), _mm256_hadd_epi32(p[2], p[3]));
        let b = _mm256_hadd_epi32(_mm256_hadd_epi32(p[4], p[5]), _mm256_hadd_epi32(p[6], p[7]));
        // a: [p0..p3 over lanes 0-3 | p0..p3 over lanes 4-7], b likewise.
        _mm256_add_epi32(_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31))
    }

    /// Q4_K/Q5_K tokens t0..t0+M of one row: 256-element super-blocks of d,
    /// dmin (f16), 12 bytes of 6-bit scales and minimums, Q5_K's 32 bytes of
    /// fifth bits, and 128 bytes of nibbles (each 32 bytes: sub-block 2j in
    /// the low nibbles, 2j+1 in the high ones, their fifth bits bit 2j and
    /// 2j+1 of qh). value = d*sc*q - dmin*mn, so the minimum's term is
    /// dmin*mn times the activation block's sum (`dsum` = its scale times
    /// its int8 sum).
    ///
    /// Each token's eight sub-block dot products stay integers until the
    /// super-block's end, then one convert and two FMAs against the eight
    /// scales at once: converting and scaling every 32 values (scalar
    /// scale products, broadcasts, a scalar minimum chain) made four tokens
    /// compute-bound at 1.7x the time of one.
    #[inline(always)]
    unsafe fn k45_rows<const FIVE: bool, const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], dsum: &[f32],
                                                          t0: usize) -> [f32; 4] {
        let mask = _mm256_set1_epi8(0x0F);
        let one = _mm256_set1_epi8(1);
        let (bytes, nib) = if FIVE { (176, 48) } else { (144, 16) };
        let mut acc = [_mm256_setzero_ps(); M];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * bytes);
            prefetch(b, bytes);
            let (dd, dmin) = (f16_at(b), f16_at(b.add(2)));
            let (sc, mn) = k4_scales(b.add(4));
            let scv = _mm256_mul_ps(_mm256_set1_ps(dd), _mm256_loadu_ps(sc.as_ptr()));
            let mnv = _mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_loadu_ps(mn.as_ptr()));
            let qh = if FIVE { _mm256_loadu_si256(b.add(16) as *const __m256i) } else { _mm256_setzero_si256() };
            let mut wv = [_mm256_setzero_si256(); 8];
            for j in 0..4 {
                let raw = _mm256_loadu_si256(b.add(nib + j * 32) as *const __m256i);
                let mut lo = _mm256_and_si256(raw, mask);
                let mut hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask);
                if FIVE {
                    lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(srl(qh, 2 * j), one), 4));
                    hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(srl(qh, 2 * j + 1), one), 4));
                }
                wv[2 * j] = lo;
                wv[2 * j + 1] = hi;
            }
            for i in 0..M {
                let base = (t0 + i) * k + sb * 256;
                let mut p = [_mm256_setzero_si256(); 8];
                for s in 0..8 {
                    p[s] = dot_u8(wv[s], xq(q, base + s * 32));
                }
                let blk = base / 32;
                let scale = _mm256_mul_ps(scv, _mm256_loadu_ps(d.as_ptr().add(blk)));
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(reduce8(&p)), scale, acc[i]);
                acc[i] = _mm256_fnmadd_ps(mnv, _mm256_loadu_ps(dsum.as_ptr().add(blk)), acc[i]);
            }
        }
        out4(&acc, &[0f32; M])
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q4k_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], dsum: &[f32], t0: usize, m: usize)
        -> [f32; 4] {
        by_m!(m, k45_rows::<false>(w, k, q, d, dsum, t0))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q5k_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], dsum: &[f32], t0: usize, m: usize)
        -> [f32; 4] {
        by_m!(m, k45_rows::<true>(w, k, q, d, dsum, t0))
    }

    /// Q6_K: 210-byte super-blocks of 128 bytes of low nibbles (ql), 64 of
    /// high bit pairs (qh), 16 int8 scales (one per 16 values) and d (f16);
    /// value = d*sc*(q-32). In each 128-value half, ql[0..32]'s low nibbles
    /// are values 0..32, ql[32..64]'s 32..64, their high nibbles 64..96 and
    /// 96..128, with qh's bit pairs 0, 2, 4 and 6.
    ///
    /// The int8 scales go into the integer sum (`madd` by the scale instead
    /// of by 1: |q-32| * 127 * 2 * 127 fits i32 easily), so each 32 values
    /// carry only the activation's scale, applied once per super-block as in
    /// k45_rows.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q6k_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, q6k_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn q6k_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let mask = _mm256_set1_epi8(0x0F);
        let three = _mm256_set1_epi8(3);
        let bias = _mm256_set1_epi8(32);
        let mut acc = [_mm256_setzero_ps(); M];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * 210);
            prefetch(b, 210);
            let dd = _mm256_set1_ps(f16_at(b.add(208)));
            let sc = b.add(192) as *const i8;
            // Group g of 32 values (g = half*4 + 0..4) uses scales 2g, 2g+1.
            let mut wv = [_mm256_setzero_si256(); 8];
            let mut sv = [_mm256_setzero_si256(); 8];
            for half in 0..2 {
                let ql = b.add(half * 64);
                let la = _mm256_loadu_si256(ql as *const __m256i);
                let lb = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                let qh = _mm256_loadu_si256(b.add(128 + half * 32) as *const __m256i);
                let groups = [
                    _mm256_or_si256(_mm256_and_si256(la, mask), high2(qh, 0, three)),
                    _mm256_or_si256(_mm256_and_si256(lb, mask), high2(qh, 2, three)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(la, 4), mask), high2(qh, 4, three)),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), mask), high2(qh, 6, three)),
                ];
                for (g, &v) in groups.iter().enumerate() {
                    let at = half * 4 + g;
                    wv[at] = _mm256_sub_epi8(v, bias);
                    sv[at] = _mm256_setr_m128i(_mm_set1_epi16(i16::from(*sc.add(2 * at))),
                                               _mm_set1_epi16(i16::from(*sc.add(2 * at + 1))));
                }
            }
            for i in 0..M {
                let base = (t0 + i) * k + sb * 256;
                let mut p = [_mm256_setzero_si256(); 8];
                for g in 0..8 {
                    let (wg, x) = (wv[g], xq(q, base + g * 32));
                    p[g] = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(wg, wg), _mm256_sign_epi8(x, wg)),
                                             sv[g]);
                }
                let scale = _mm256_mul_ps(dd, _mm256_loadu_ps(d.as_ptr().add(base / 32)));
                acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(reduce8(&p)), scale, acc[i]);
            }
        }
        out4(&acc, &[0f32; M])
    }

    /// Q8_0: 34-byte blocks of d (f16) and 32 int8 values.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q8_0_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, q8_0_rows(w, k, q, d, t0))
    }

    /// int8 per 32 against amax / 127, as quantize_into.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn quantize32(x: &[f32], q: &mut [i8], d: &mut [f32]) {
        let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        for (b, (xs, dv)) in x.chunks_exact(32).zip(d.iter_mut()).enumerate() {
            let v = [_mm256_loadu_ps(xs.as_ptr()), _mm256_loadu_ps(xs.as_ptr().add(8)),
                     _mm256_loadu_ps(xs.as_ptr().add(16)), _mm256_loadu_ps(xs.as_ptr().add(24))];
            let m = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(v[0], abs), _mm256_and_ps(v[1], abs)),
                                  _mm256_max_ps(_mm256_and_ps(v[2], abs), _mm256_and_ps(v[3], abs)));
            let m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            let amax = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1)));
            let scale = amax / 127.0;
            *dv = scale;
            let inv = _mm256_set1_ps(if scale > 0.0 { 1.0 / scale } else { 0.0 });
            let i = [_mm256_cvtps_epi32(_mm256_mul_ps(v[0], inv)), _mm256_cvtps_epi32(_mm256_mul_ps(v[1], inv)),
                     _mm256_cvtps_epi32(_mm256_mul_ps(v[2], inv)), _mm256_cvtps_epi32(_mm256_mul_ps(v[3], inv))];
            // packs work per 128-bit lane; the permute puts the 32 bytes back
            // in element order.
            let p16a = _mm256_packs_epi32(i[0], i[1]);
            let p16b = _mm256_packs_epi32(i[2], i[3]);
            let p8 = _mm256_packs_epi16(p16a, p16b);
            let ordered = _mm256_permutevar8x32_epi32(p8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(q.as_mut_ptr().add(b * 32) as *mut __m256i, ordered);
        }
    }

    // --- the IQ types and Q2_0 ----------------------------------------------------
    //
    // Mirrors of the GPU's integer dots (vecdotq.cuh) for the types Qwen4
    // Flash-Next keeps its experts in. The codebook magnitudes are unsigned
    // bytes and the signs travel apart, so the signs go onto the activations
    // (`_mm256_sign_epi8`) and `maddubs` takes the magnitudes as they are.
    // Each sub-block's integer scale goes into the `madd` (instead of 1), as in
    // q6k_rows, so a super-block costs one convert and one FMA per token.

    /// IQ4_NL/IQ4_XS's codebook in both 128-bit lanes, for `_mm256_shuffle_epi8`.
    #[inline(always)]
    unsafe fn iq4_lut() -> __m256i {
        let l = _mm_loadu_si128(airun_gguf::dequant::KVALUES_IQ4NL.as_ptr() as *const __m128i);
        _mm256_set_m128i(l, l)
    }

    /// 32 IQ4 elements from 16 code bytes (element i in the low nibble of byte
    /// i, element i + 16 in its high one) as int8 codebook values, in order.
    #[inline(always)]
    unsafe fn iq4_values(p: *const u8, lut: __m256i) -> __m256i {
        let raw = _mm_loadu_si128(p as *const __m128i);
        let m = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(raw, m);
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m);
        _mm256_shuffle_epi8(lut, _mm256_set_m128i(hi, lo))
    }

    /// Four sign bytes (bit j of byte l: element 8l + j) as 32 +1/-1 bytes.
    #[inline(always)]
    unsafe fn expand_signs(s: u32) -> __m256i {
        let rep = _mm256_shuffle_epi8(_mm256_set1_epi32(s as i32),
            _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0));
        let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
        let neg = _mm256_cmpeq_epi8(_mm256_and_si256(rep, bits), bits);
        _mm256_or_si256(neg, _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn u32_at(p: *const u8) -> u32 {
        std::ptr::read_unaligned(p as *const u32)
    }

    /// One 32-element sub-block at `at` of every token: the `sg`-signed
    /// activations against unsigned magnitudes `wv`, each pair scaled by the
    /// integer `sv` in the madd, times `scale` and the activation's own scale,
    /// into `acc`. Called sub-block by sub-block as each is decoded: decoding
    /// a whole super-block first (24 vectors for 16 registers) spilled.
    #[inline(always)]
    unsafe fn lattice_sub<const M: usize>(wv: __m256i, sg: __m256i, sv: __m256i, scale: f32, q: &[i8], d: &[f32],
                                          k: usize, t0: usize, at: usize, acc: &mut [__m256; M]) {
        for i in 0..M {
            let base = (t0 + i) * k + at;
            let x = _mm256_sign_epi8(xq(q, base), sg);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(wv, x), sv);
            acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(scale * *d.get_unchecked(base / 32)), acc[i]);
        }
    }

    /// IQ4_NL: 18-byte blocks of d (f16) and 16 bytes of codebook nibbles.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn iq4nl_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, iq4nl_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn iq4nl_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let lut = iq4_lut();
        let mut acc = [_mm256_setzero_ps(); M];
        for blk in 0..k / 32 {
            let b = w.as_ptr().add(blk * 18);
            if blk % 2 == 0 {
                prefetch(b, 36);
            }
            let dw = f16_at(b);
            let wv = iq4_values(b.add(2), lut);
            for i in 0..M {
                let base = (t0 + i) * k + blk * 32;
                let p = _mm256_cvtepi32_ps(dot_i8(wv, xq(q, base)));
                acc[i] = _mm256_fmadd_ps(p, _mm256_set1_ps(dw * *d.get_unchecked(base / 32)), acc[i]);
            }
        }
        out4(&acc, &[0f32; M])
    }

    /// IQ4_XS: 136-byte super-blocks of d, 8 6-bit scales (low nibbles in
    /// scales_l, high bit pairs in scales_h) and 8 IQ4_NL payloads.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn iq4xs_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, iq4xs_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn iq4xs_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let lut = iq4_lut();
        let mut acc = [_mm256_setzero_ps(); M];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * 136);
            prefetch(b, 136);
            let dd = f16_at(b);
            let sh = u32::from(u16::from_le_bytes([*b.add(2), *b.add(3)]));
            for s in 0..8 {
                let lo = u32::from((*b.add(4 + s / 2) >> (4 * (s % 2))) & 0x0F);
                let ls = (lo | (((sh >> (2 * s)) & 3) << 4)) as i16 - 32;
                // Signed codes: their magnitudes, and their signs onto the activations.
                let wg = iq4_values(b.add(8 + 16 * s), lut);
                lattice_sub::<M>(_mm256_sign_epi8(wg, wg), wg, _mm256_set1_epi16(ls), dd, q, d, k, t0,
                    sb * 256 + s * 32, &mut acc);
            }
        }
        out4(&acc, &[0f32; M])
    }

    /// IQ2_S: 82-byte super-blocks of d, 32 index bytes, 32 sign bytes, 8
    /// bytes of index high bits and 8 scale bytes (a nibble per 16 elements);
    /// value = d * (0.5 + nibble) / 4 * grid magnitude * sign.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn iq2s_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, iq2s_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn iq2s_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let grid = airun_gguf::iq_grids::IQ2S_GRID.as_ptr() as *const i64;
        let hshift = _mm_setr_epi32(0, 2, 4, 6);
        let three = _mm_set1_epi32(3);
        let mut acc = [_mm256_setzero_ps(); M];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * 82);
            prefetch(b, 82);
            let (qs, signs, qh, scales) = (b.add(2), b.add(34), b.add(66), b.add(74));
            let dd = f16_at(b) * 0.125;
            for s in 0..8 {
                // Point l's 10-bit index: its qs byte, and bits 2l, 2l+1 of qh as 8, 9.
                let lo = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(u32_at(qs.add(4 * s)) as i32));
                let hi = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(i32::from(*qh.add(s))), hshift), three);
                let idx = _mm_or_si128(lo, _mm_slli_epi32(hi, 8));
                let sc = *scales.add(s);
                let sv = _mm256_setr_m128i(_mm_set1_epi16(i16::from(2 * (sc & 0x0F) + 1)),
                                           _mm_set1_epi16(i16::from(2 * (sc >> 4) + 1)));
                lattice_sub::<M>(_mm256_i32gather_epi64::<8>(grid, idx), expand_signs(u32_at(signs.add(4 * s))), sv,
                    dd, q, d, k, t0, sb * 256 + s * 32, &mut acc);
            }
        }
        out4(&acc, &[0f32; M])
    }

    /// IQ3_XXS: 98-byte super-blocks of d, 64 index bytes and a word per 32
    /// elements of four 7-bit sign codes and a 4-bit scale on top;
    /// value = d * (0.5 + scale) / 2 * grid magnitude * sign.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn iq3xxs_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, iq3xxs_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn iq3xxs_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let grid = airun_gguf::iq_grids::IQ3XXS_GRID.as_ptr() as *const i32;
        let signs = super::ksigns64().as_ptr() as *const i64;
        let cshift = _mm_setr_epi32(0, 7, 14, 21);
        let c127 = _mm_set1_epi32(127);
        let mut acc = [_mm256_setzero_ps(); M];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * 98);
            prefetch(b, 98);
            let (qs, sas) = (b.add(2), b.add(66));
            let dd = f16_at(b) * 0.25;
            for s in 0..8 {
                let idx = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs.add(8 * s) as *const __m128i));
                let aux = u32_at(sas.add(4 * s));
                let codes = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(aux as i32), cshift), c127);
                lattice_sub::<M>(_mm256_i32gather_epi32::<4>(grid, idx), _mm256_i32gather_epi64::<8>(signs, codes),
                    _mm256_set1_epi16((2 * (aux >> 28) + 1) as i16), dd, q, d, k, t0, sb * 256 + s * 32, &mut acc);
            }
        }
        out4(&acc, &[0f32; M])
    }

    /// IQ3_S: 110-byte super-blocks of d, 64 index bytes, 8 bytes of ninth
    /// index bits, 32 sign bytes and 4 bytes of 4-bit scales;
    /// value = d * (1 + 2 * scale) * grid magnitude * sign.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn iq3s_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, iq3s_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn iq3s_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let grid = airun_gguf::iq_grids::IQ3S_GRID.as_ptr() as *const i32;
        let hshift = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let one = _mm256_set1_epi32(1);
        let mut acc = [_mm256_setzero_ps(); M];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * 110);
            prefetch(b, 110);
            let (qs, qh, signs, scales) = (b.add(2), b.add(66), b.add(74), b.add(106));
            let dd = f16_at(b);
            for s in 0..8 {
                // Index i: its qs byte, and bit i of qh as the ninth bit.
                let lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qs.add(8 * s) as *const __m128i));
                let hi = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(i32::from(*qh.add(s))), hshift), one);
                let wv = _mm256_i32gather_epi32::<4>(grid, _mm256_or_si256(lo, _mm256_slli_epi32(hi, 8)));
                let sc = (*scales.add(s / 2) >> (4 * (s % 2))) & 0x0F;
                lattice_sub::<M>(wv, expand_signs(u32_at(signs.add(4 * s))), _mm256_set1_epi16(i16::from(1 + 2 * sc)),
                    dd, q, d, k, t0, sb * 256 + s * 32, &mut acc);
            }
        }
        out4(&acc, &[0f32; M])
    }

    /// Q2_0: 18-byte blocks of d (f16) and 64 2-bit codes meaning -1..+2, four
    /// to a byte, low bits first.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q2_0_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        by_m!(m, q2_0_rows(w, k, q, d, t0))
    }

    #[inline(always)]
    unsafe fn q2_0_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        // Byte e of a group's 32 takes code byte e / 4 (both lanes hold all
        // eight), then the bit pair e % 4 of it.
        let spread = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                      4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
        let bit0 = _mm256_set1_epi32(0x4010_0401);
        let bit1 = _mm256_set1_epi32(0x8020_0802u32 as i32);
        let one = _mm256_set1_epi8(1);
        let mut acc = [_mm256_setzero_ps(); M];
        for blk in 0..k / 64 {
            let b = w.as_ptr().add(blk * 18);
            if blk % 2 == 0 {
                prefetch(b, 36);
            }
            let dw = f16_at(b);
            for h in 0..2 {
                let codes = std::ptr::read_unaligned(b.add(2 + 8 * h) as *const i64);
                let rep = _mm256_shuffle_epi8(_mm256_set1_epi64x(codes), spread);
                let b0 = _mm256_cmpeq_epi8(_mm256_and_si256(rep, bit0), bit0);
                let b1 = _mm256_cmpeq_epi8(_mm256_and_si256(rep, bit1), bit1);
                // code = -b0 - 2 * b1 (b0, b1 are 0 or -1); the value is code - 1.
                let code = _mm256_sub_epi8(_mm256_sub_epi8(_mm256_setzero_si256(), b0), _mm256_add_epi8(b1, b1));
                let wv = _mm256_sub_epi8(code, one);
                for i in 0..M {
                    let base = (t0 + i) * k + blk * 64 + h * 32;
                    let p = _mm256_cvtepi32_ps(dot_i8(wv, xq(q, base)));
                    acc[i] = _mm256_fmadd_ps(p, _mm256_set1_ps(dw * *d.get_unchecked(base / 32)), acc[i]);
                }
            }
        }
        out4(&acc, &[0f32; M])
    }

    #[inline(always)]
    unsafe fn q8_0_rows<const M: usize>(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize) -> [f32; 4] {
        let mut acc = [_mm256_setzero_ps(); M];
        for blk in 0..k / 32 {
            let b = w.as_ptr().add(blk * 34);
            if blk % 2 == 0 {
                prefetch(b, 68);
            }
            let dw = f16_at(b);
            let wv = _mm256_loadu_si256(b.add(2) as *const __m256i);
            for i in 0..M {
                let base = (t0 + i) * k + blk * 32;
                let p = _mm256_cvtepi32_ps(dot_i8(wv, xq(q, base)));
                acc[i] = _mm256_fmadd_ps(p, _mm256_set1_ps(dw * *d.get_unchecked(base / 32)), acc[i]);
            }
        }
        out4(&acc, &[0f32; M])
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 NVFP4 kernel and the generic path (dequantize, f32 dot) agree
    /// to within the activation's int8 rounding.
    #[test]
    fn nvfp4_matches_dequantized_rows() {
        let (k, rows, n) = (256usize, 37usize, 5usize);
        let row_bytes = k / 64 * 36;
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut w = vec![0u8; rows * row_bytes];
        for (i, b) in w.iter_mut().enumerate() {
            // Scales in a sane UE4M3 range, random nibbles.
            *b = if i % 36 < 4 { 0x30 + (next() % 16) as u8 } else { next() as u8 };
        }
        let x: Vec<f32> = (0..n * k).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut y = vec![0f32; n * rows];
        matvec(GgmlType::NVFP4, &w, row_bytes, k, rows, &x, n, &mut y).unwrap();
        let mut row = vec![0f32; k];
        for r in 0..rows {
            airun_gguf::dequant::dequantize(GgmlType::NVFP4, &w[r * row_bytes..(r + 1) * row_bytes], k, &mut row)
                .unwrap();
            for t in 0..n {
                let exact: f32 = row.iter().zip(&x[t * k..(t + 1) * k]).map(|(a, b)| a * b).sum();
                let mag: f32 = row.iter().zip(&x[t * k..(t + 1) * k]).map(|(a, b)| (a * b).abs()).sum();
                let got = y[t * rows + r];
                assert!((got - exact).abs() <= mag * 0.02 + 1e-4, "row {r} token {t}: {got} vs {exact}");
            }
        }
    }

    /// Each AVX2 integer kernel agrees with dequantize + f32 dot, on random
    /// blocks with sane f16 scales.
    #[test]
    fn int_kernels_match_dequantized_rows() {
        if !has_avx2() {
            return;
        }
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let f16b = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
        for (ty, block, bytes) in [(GgmlType::Q4_K, 256usize, 144usize), (GgmlType::Q5_K, 256, 176),
                                   (GgmlType::Q6_K, 256, 210), (GgmlType::Q8_0, 32, 34), (GgmlType::IQ4_NL, 32, 18),
                                   (GgmlType::IQ4_XS, 256, 136), (GgmlType::IQ2_S, 256, 82),
                                   (GgmlType::IQ3_XXS, 256, 98), (GgmlType::IQ3_S, 256, 110),
                                   (GgmlType::Q2_0, 64, 18)] {
            let (k, rows) = (512usize, 21usize);
            let row_bytes = k / block * bytes;
            let mut w = vec![0u8; rows * row_bytes];
            for blk in w.chunks_exact_mut(bytes) {
                for b in blk.iter_mut() {
                    *b = next() as u8;
                }
                let s = f16b(0.01 + (next() % 100) as f32 / 10000.0);
                match ty {
                    GgmlType::Q4_K | GgmlType::Q5_K => {
                        blk[0..2].copy_from_slice(&s);
                        blk[2..4].copy_from_slice(&f16b(0.005));
                    }
                    GgmlType::Q6_K => blk[208..210].copy_from_slice(&s),
                    _ => {
                        blk[0..2].copy_from_slice(&s);
                        // ggml's Q8_0 never stores -128.
                        for b in &mut blk[2..] {
                            if *b == 0x80 {
                                *b = 0x81;
                            }
                        }
                    }
                }
            }
            for n in [1usize, 3, 4, 7] {
                let x: Vec<f32> = (0..n * k).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
                let mut y = vec![0f32; n * rows];
                matvec(ty, &w, row_bytes, k, rows, &x, n, &mut y).unwrap();
                let mut row = vec![0f32; k];
                for r in 0..rows {
                    airun_gguf::dequant::dequantize(ty, &w[r * row_bytes..(r + 1) * row_bytes], k, &mut row).unwrap();
                    for t in 0..n {
                        let xs = &x[t * k..(t + 1) * k];
                        let exact: f32 = row.iter().zip(xs).map(|(a, b)| a * b).sum();
                        let mag: f32 = row.iter().zip(xs).map(|(a, b)| (a * b).abs()).sum();
                        let got = y[t * rows + r];
                        assert!((got - exact).abs() <= mag * 0.02 + 1e-4,
                            "{ty} n {n} row {r} token {t}: {got} vs {exact}");
                    }
                }
            }
        }
    }

    /// moe_experts against dequantize + f32: gate/up, SwiGLU, down and the
    /// weighted sum, for Qwen4 Flash-Next's two expert type mixes.
    #[test]
    fn moe_experts_match_dequantized() {
        if !has_avx2() {
            return;
        }
        let mut seed = 0x51ED_2701_99AB_CD01u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // ff a whole number of Q2_0 blocks (64), as the model's 640.
        let (k, ff) = (512usize, 128usize);
        for (gty, dty) in [(GgmlType::IQ3_XXS, GgmlType::IQ4_NL), (GgmlType::IQ2_S, GgmlType::Q2_0),
                           (GgmlType::IQ3_S, GgmlType::IQ4_NL), (GgmlType::IQ4_XS, GgmlType::Q2_0)] {
            let bytes = |ty: GgmlType, n: usize| ty.layout().map(|l| n / l.elements as usize * l.bytes as usize).unwrap();
            let (g_row, d_row) = (bytes(gty, k), bytes(dty, ff));
            let mut mat = |ty: GgmlType, rows: usize, row: usize| -> Vec<u8> {
                let block = ty.layout().unwrap().bytes as usize;
                let mut w: Vec<u8> = (0..rows * row).map(|_| next() as u8).collect();
                for b in w.chunks_exact_mut(block) {
                    let s = half::f16::from_f32(0.002 + (next() % 100) as f32 / 20000.0).to_bits().to_le_bytes();
                    b[0..2].copy_from_slice(&s);
                }
                w
            };
            let experts: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, f32)> = (0..3).map(|j| {
                (mat(gty, ff, g_row), mat(gty, ff, g_row), mat(dty, k, d_row), 0.2 + 0.1 * j as f32)
            }).collect();
            let x: Vec<f32> = (0..k).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
            let set: Vec<(&[u8], &[u8], &[u8], f32)> =
                experts.iter().map(|e| (&e.0[..], &e.1[..], &e.2[..], e.3)).collect();
            let mut y = vec![0f32; k];
            moe_experts(gty, dty, &set, &x, k, ff, &mut y).unwrap();
            let row = |ty: GgmlType, w: &[u8], r: usize, n: usize, rb: usize| {
                let mut v = vec![0f32; n];
                airun_gguf::dequant::dequantize(ty, &w[r * rb..(r + 1) * rb], n, &mut v).unwrap();
                v
            };
            let mut exact = vec![0f32; k];
            let mut mag = vec![0f32; k];
            for e in &experts {
                let h: Vec<f32> = (0..ff).map(|r| {
                    let g: f32 = row(gty, &e.0, r, k, g_row).iter().zip(&x).map(|(a, b)| a * b).sum();
                    let u: f32 = row(gty, &e.1, r, k, g_row).iter().zip(&x).map(|(a, b)| a * b).sum();
                    g / (1.0 + (-g).exp()) * u
                }).collect();
                for r in 0..k {
                    let dr = row(dty, &e.2, r, ff, d_row);
                    exact[r] += e.3 * dr.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();
                    mag[r] += e.3 * dr.iter().zip(&h).map(|(a, b)| (a * b).abs()).sum::<f32>();
                }
            }
            let peak = mag.iter().fold(0f32, |m, v| m.max(*v));
            for r in 0..k {
                // Two int8 roundings (x, then h) on the way.
                assert!((y[r] - exact[r]).abs() <= 0.04 * mag[r] + 1e-3 * peak,
                    "{gty}/{dty} row {r}: {} vs {} (magnitude {})", y[r], exact[r], mag[r]);
            }
        }
    }

    #[test]
    fn f32_rows_are_exact_dots() {

        let (k, rows, n) = (96usize, 9usize, 3usize);
        let w: Vec<f32> = (0..rows * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.25).collect();
        let mut y = vec![0f32; n * rows];
        matvec(GgmlType::F32, &bytes, k * 4, k, rows, &x, n, &mut y).unwrap();
        for r in 0..rows {
            for t in 0..n {
                let e: f32 = (0..k).map(|i| w[r * k + i] * x[t * k + i]).sum();
                assert!((y[t * rows + r] - e).abs() < 1e-3);
            }
        }
    }
}
