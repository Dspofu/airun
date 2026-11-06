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

const SPIN: std::time::Duration = std::time::Duration::from_micros(200);

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
            if start.elapsed() > SPIN {
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
    q.resize(x.len(), 0);
    d.resize(x.len() / 32, 0.0);
    for (b, (xs, qs)) in x.chunks_exact(32).zip(q.chunks_exact_mut(32)).enumerate() {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = amax / 127.0;
        d[b] = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (o, v) in qs.iter_mut().zip(xs) {
            *o = (v * inv).round() as i8;
        }
    }
}

thread_local! {
    static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
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
            pool().run(chunks, &|c| {
                for r in c * CHUNK..((c + 1) * CHUNK).min(rows) {
                    let wr = &w[r * row_bytes..(r + 1) * row_bytes];
                    nvfp4_row(wr, k, q, d, n, |t, v| {
                        // SAFETY: (t, r) belongs to this chunk's rows only.
                        unsafe { *out.0.add(t * rows + r) = v };
                    });
                }
            });
        }
        _ => {
            let elems = ty.layout().map_or(1, |l| l.elements as usize);
            if k % elems != 0 {
                return Err(CoreError::Other(format!("no CPU matmul for {ty} rows of {k}; run with AIRUN_RAM_ON_GPU=1")));
            }
            let failed = AtomicBool::new(false);
            pool().run(chunks, &|c| {
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
            let v = acc[i];
            let s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
            let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
            out[i] = _mm_cvtss_f32(s1);
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
