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

fn worker(shared: &Shared) {
    let mut seen = 0u64;
    loop {
        // Spin for a while, then sleep until the next job.
        let start = std::time::Instant::now();
        while shared.generation.load(Ordering::Acquire) == seen {
            if start.elapsed() > spin() {
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
        let n = std::env::var("AIRUN_CPU_THREADS").ok().and_then(|v| v.trim().parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
        Pool::new(n.max(1))
    })
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
    PASS.load(Ordering::Relaxed)
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
        GgmlType::Q8_0 => Some(32),
        _ => None,
    }
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
            let (q, d, sums) = (&q, &d, &sums);
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
                                GgmlType::Q4_K => avx2::q4k_row4(wr, k, q, d, sums, t0, m),
                                GgmlType::Q5_K => avx2::q5k_row4(wr, k, q, d, sums, t0, m),
                                GgmlType::Q6_K => avx2::q6k_row4(wr, k, q, d, t0, m),
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

    /// Tokens t0..t0+m (m <= 4) of one NVFP4 row; see nvfp4_row.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn nvfp4_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        let table = super::ue4m3_table();
        let lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                   0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
        let low = _mm_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); 4];
        let blocks = k / 64;
        for b in 0..blocks {
            let blk = w.as_ptr().add(b * 36);
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
                for i in 0..m {
                    let t = t0 + i;
                    let x = _mm256_loadu_si256(q.as_ptr().add(t * k + at) as *const __m256i);
                    let xs = _mm256_sign_epi8(x, wv);
                    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, xs), ones);
                    let dx = _mm256_set1_ps(*d.get_unchecked((t * k + at) / 32));
                    acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_mul_ps(scale, dx), acc[i]);
                }
            }
        }
        let mut out = [0f32; 4];
        for i in 0..m {
            out[i] = hsum(acc[i]);
        }
        out
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

    /// Q4_K/Q5_K tokens t0..t0+m (m <= 4) of one row: 256-element
    /// super-blocks of d, dmin (f16), 12 bytes of 6-bit scales and minimums,
    /// Q5_K's 32 bytes of fifth bits, and 128 bytes of nibbles (each 32
    /// bytes: sub-block 2j in the low nibbles, 2j+1 in the high ones, their
    /// fifth bits bit 2j and 2j+1 of qh). value = d*sc*q - dmin*mn, so the
    /// minimum's term is dmin*mn times the activation block's sum.
    #[inline(always)]
    unsafe fn k45_row4<const FIVE: bool>(w: &[u8], k: usize, q: &[i8], d: &[f32], sums: &[i32], t0: usize,
                                         m: usize) -> [f32; 4] {
        let mask = _mm256_set1_epi8(0x0F);
        let one = _mm256_set1_epi8(1);
        let (bytes, nib) = if FIVE { (176, 48) } else { (144, 16) };
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut acc_min = [0f32; 4];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * bytes);
            let (dd, dmin) = (f16_at(b), f16_at(b.add(2)));
            let (sc, mn) = k4_scales(b.add(4));
            let qh = if FIVE { _mm256_loadu_si256(b.add(16) as *const __m256i) } else { _mm256_setzero_si256() };
            for j in 0..4 {
                let raw = _mm256_loadu_si256(b.add(nib + j * 32) as *const __m256i);
                let mut lo = _mm256_and_si256(raw, mask);
                let mut hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask);
                if FIVE {
                    lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(srl(qh, 2 * j), one), 4));
                    hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(srl(qh, 2 * j + 1), one), 4));
                }
                let (s0, s1) = (dd * sc[2 * j], dd * sc[2 * j + 1]);
                let (m0, m1) = (dmin * mn[2 * j], dmin * mn[2 * j + 1]);
                let a = sb * 256 + j * 64;
                for i in 0..m {
                    let base = (t0 + i) * k + a;
                    let blk = base / 32;
                    let (d0, d1) = (*d.get_unchecked(blk), *d.get_unchecked(blk + 1));
                    let p0 = _mm256_cvtepi32_ps(dot_u8(lo, xq(q, base)));
                    let p1 = _mm256_cvtepi32_ps(dot_u8(hi, xq(q, base + 32)));
                    acc[i] = _mm256_fmadd_ps(p0, _mm256_set1_ps(s0 * d0), acc[i]);
                    acc[i] = _mm256_fmadd_ps(p1, _mm256_set1_ps(s1 * d1), acc[i]);
                    acc_min[i] += m0 * d0 * *sums.get_unchecked(blk) as f32
                        + m1 * d1 * *sums.get_unchecked(blk + 1) as f32;
                }
            }
        }
        let mut out = [0f32; 4];
        for i in 0..m {
            out[i] = hsum(acc[i]) - acc_min[i];
        }
        out
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q4k_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], sums: &[i32], t0: usize, m: usize)
        -> [f32; 4] {
        k45_row4::<false>(w, k, q, d, sums, t0, m)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q5k_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], sums: &[i32], t0: usize, m: usize)
        -> [f32; 4] {
        k45_row4::<true>(w, k, q, d, sums, t0, m)
    }

    /// Q6_K: 210-byte super-blocks of 128 bytes of low nibbles (ql), 64 of
    /// high bit pairs (qh), 16 int8 scales (one per 16 values) and d (f16);
    /// value = d*sc*(q-32). In each 128-value half, ql[0..32]'s low nibbles
    /// are values 0..32, ql[32..64]'s 32..64, their high nibbles 64..96 and
    /// 96..128, with qh's bit pairs 0, 2, 4 and 6.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q6k_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        let mask = _mm256_set1_epi8(0x0F);
        let three = _mm256_set1_epi8(3);
        let bias = _mm256_set1_epi8(32);
        let mut acc = [_mm256_setzero_ps(); 4];
        for sb in 0..k / 256 {
            let b = w.as_ptr().add(sb * 210);
            let dd = f16_at(b.add(208));
            for half in 0..2 {
                let ql = b.add(half * 64);
                let sc = b.add(192 + half * 8) as *const i8;
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
                    let wv = _mm256_sub_epi8(v, bias);
                    let (s0, s1) = (dd * f32::from(*sc.add(2 * g)), dd * f32::from(*sc.add(2 * g + 1)));
                    let a = sb * 256 + half * 128 + g * 32;
                    for i in 0..m {
                        let base = (t0 + i) * k + a;
                        let dx = *d.get_unchecked(base / 32);
                        let p = _mm256_cvtepi32_ps(dot_i8(wv, xq(q, base)));
                        let scale = _mm256_setr_m128(_mm_set1_ps(s0 * dx), _mm_set1_ps(s1 * dx));
                        acc[i] = _mm256_fmadd_ps(p, scale, acc[i]);
                    }
                }
            }
        }
        let mut out = [0f32; 4];
        for i in 0..m {
            out[i] = hsum(acc[i]);
        }
        out
    }

    /// Q8_0: 34-byte blocks of d (f16) and 32 int8 values.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q8_0_row4(w: &[u8], k: usize, q: &[i8], d: &[f32], t0: usize, m: usize) -> [f32; 4] {
        let mut acc = [_mm256_setzero_ps(); 4];
        for blk in 0..k / 32 {
            let b = w.as_ptr().add(blk * 34);
            let dw = f16_at(b);
            let wv = _mm256_loadu_si256(b.add(2) as *const __m256i);
            for i in 0..m {
                let base = (t0 + i) * k + blk * 32;
                let p = _mm256_cvtepi32_ps(dot_i8(wv, xq(q, base)));
                acc[i] = _mm256_fmadd_ps(p, _mm256_set1_ps(dw * *d.get_unchecked(base / 32)), acc[i]);
            }
        }
        let mut out = [0f32; 4];
        for i in 0..m {
            out[i] = hsum(acc[i]);
        }
        out
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
                                   (GgmlType::Q6_K, 256, 210), (GgmlType::Q8_0, 32, 34)] {
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
