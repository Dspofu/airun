//! Times the CPU matmuls (cpu.rs) on the Qwen3.8-27B FFN shape, the weights
//! in pinned memory as the runtime keeps them. Ignored by default:
//!
//!     cargo test --release -p airun-core --test cpu_bench -- --ignored --nocapture

use airun_gguf::GgmlType;

/// The IQ types and Q2_0 in Qwen4 Flash-Next's expert shapes (gate/up 640 x
/// 2560, down 2560 x 640), over enough experts that the timing reads DRAM:
///
///     cargo test --release -p airun-core --test cpu_bench iq -- --ignored --nocapture
#[test]
#[ignore]
fn iq_expert_bandwidth() {
    for (ty, block, bytes, rows, k) in [(GgmlType::IQ3_XXS, 256usize, 98usize, 640usize, 2560usize),
        (GgmlType::IQ2_S, 256, 82, 640, 2560), (GgmlType::IQ3_S, 256, 110, 640, 2560),
        (GgmlType::IQ4_XS, 256, 136, 640, 2560), (GgmlType::IQ4_NL, 32, 18, 2560, 640),
        (GgmlType::Q2_0, 64, 18, 2560, 640)] {
        let row_bytes = k / block * bytes;
        let size = rows * row_bytes;
        let span: usize = std::env::var("AIRUN_BENCH_SPAN_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mats = if span > 0 { (span << 30) / size } else { 200 };
        let mut w = airun_cuda::PinnedBuffer::<u8>::new(mats * size).unwrap();
        let mut seed = 0x9E37_79B9u64;
        for (i, b) in w.as_mut_slice().iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *b = if i % bytes < 2 { [0x00, 0x20][i % 2] } else { seed as u8 };
        }
        let x: Vec<f32> = (0..k).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
        let mut y = vec![0f32; rows];
        // Best of three passes: one thread lands on a P- or an E-core.
        let mut s = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for i in 0..200 {
                let m = i * 7 % mats;
                airun_core::cpu::matvec(ty, &w.as_slice()[m * size..(m + 1) * size], row_bytes, k, rows, &x, 1,
                    &mut y).unwrap();
            }
            s = s.min(t.elapsed().as_secs_f64() / 200.0);
        }
        println!("{ty:>7} {rows} x {k}: {:6.1} us a matrix, {:5.1} GB/s of weights", s * 1e6, size as f64 / s / 1e9);
    }
}

/// A decode step's experts in RAM on one Qwen4 Flash-Next layer (gate/up
/// IQ3_XXS or IQ2_S, down IQ4_NL or Q2_0; 2560 wide, ff 640), `n` of them
/// per call, through cpu::moe_experts, each call on other experts:
///
///     cargo test --release -p airun-core --test cpu_bench moe -- --ignored --nocapture
#[test]
#[ignore]
fn moe_experts_bandwidth() {
    let (k, ff) = (2560usize, 640usize);
    for (gty, gb, dty, db) in [(GgmlType::IQ3_XXS, 98usize, GgmlType::IQ4_NL, 18usize),
                               (GgmlType::IQ2_S, 82, GgmlType::Q2_0, 18)] {
        let g_size = ff * k / 256 * gb;
        let d_size = k * ff / (if dty == GgmlType::Q2_0 { 64 } else { 32 }) * db;
        let per = 2 * g_size + d_size;
        // AIRUN_BENCH_SPAN_GB spreads the experts over that much pinned RAM,
        // as the runtime's ~40 GiB of them are (page tables then miss the cache).
        let span: usize = std::env::var("AIRUN_BENCH_SPAN_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let experts = if span > 0 { (span << 30) / per } else { 160 };
        // AIRUN_BENCH_PAGEABLE=1: ordinary memory instead of pinned.
        let pageable = std::env::var("AIRUN_BENCH_PAGEABLE").is_ok_and(|v| v == "1");
        let mut pinned = (!pageable).then(|| airun_cuda::PinnedBuffer::<u8>::new(experts * per).unwrap());
        let mut plain = if pageable { vec![0u8; experts * per] } else { Vec::new() };
        let w: &mut [u8] = match pinned.as_mut() { Some(p) => p.as_mut_slice(), None => &mut plain };
        let mut seed = 0x9E37_79B9u64;
        for (i, b) in w.iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *b = if i % 2 == 0 && seed % 7 == 0 { 0x20 } else { seed as u8 };
        }
        let w: &[u8] = w;
        let x: Vec<f32> = (0..k).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
        let mut y = vec![0f32; k];
        for n in [3usize, 5, 8] {
            let calls = 60;
            let mut busy = 0.0;
            for c in 0..calls {
                // AIRUN_BENCH_USE: cycle over only that many of the experts.
                let used = std::env::var("AIRUN_BENCH_USE").ok().and_then(|v| v.parse().ok()).unwrap_or(experts)
                    .min(experts);
                let set: Vec<(&[u8], &[u8], &[u8], f32)> = (0..n).map(|j| {
                    let e = (c * n + j) * 7 % used;
                    let b = &w[e * per..(e + 1) * per];
                    (&b[..g_size], &b[g_size..2 * g_size], &b[2 * g_size..], 0.1)
                }).collect();
                // The GPU's dense part runs between two layers.
                std::thread::sleep(std::time::Duration::from_micros(300));
                let t = std::time::Instant::now();
                airun_core::cpu::moe_experts(gty, dty, &set, &x, k, ff, &mut y).unwrap();
                busy += t.elapsed().as_secs_f64();
            }
            let s = busy / calls as f64;
            println!("{gty}/{dty}, {n} experts: {:6.1} us a call, {:5.1} GB/s", s * 1e6, (n * per) as f64 / s / 1e9);
        }
    }
}

/// f32::exp's cost by input range (the C runtime's error paths), and f32
/// arithmetic on subnormal values.
///
///     cargo test --release -p airun-core --test cpu_bench exp_cost -- --ignored --nocapture
#[test]
#[ignore]
fn exp_cost() {
    for (name, v) in [("0.5", 0.5f32), ("-30", -30.0), ("-95 (subnormal)", -95.0), ("-200 (underflow)", -200.0),
                      ("+200 (overflow)", 200.0), ("NaN", f32::NAN)] {
        let n = 200_000;
        let t = std::time::Instant::now();
        let mut s = 0f32;
        for i in 0..n {
            s += std::hint::black_box(v + (i % 3) as f32 * 1e-3).exp();
        }
        std::hint::black_box(s);
        println!("exp({name}): {:6.1} ns a call", t.elapsed().as_secs_f64() / n as f64 * 1e9);
    }
    for (name, v) in [("normal", 1e-3f32), ("subnormal", 1e-40)] {
        let n = 2_000_000;
        let t = std::time::Instant::now();
        let mut acc = 0f32;
        for i in 0..n {
            acc = std::hint::black_box(acc * 0.5 + v * (i % 3) as f32);
        }
        std::hint::black_box(acc);
        println!("multiply-add on {name} floats: {:6.2} ns", t.elapsed().as_secs_f64() / n as f64 * 1e9);
    }
}

/// The process's page fault count and working set (Windows; zeros elsewhere).
fn page_faults() -> (u64, u64) {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Counters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set: usize,
            working_set: usize,
            rest: [usize; 7],
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(process: *mut std::ffi::c_void, counters: *mut Counters, cb: u32) -> i32;
        }
        let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
        // SAFETY: `c` is a PROCESS_MEMORY_COUNTERS_EX with its size in `cb`.
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (u64::from(c.page_fault_count), c.working_set as u64)
    }
    #[cfg(not(windows))]
    (0, 0)
}

/// The same 5-expert call repeated in rounds over 2 GB of pinned RAM, one
/// line per round, to see whether the speed changes as the process runs:
///
///     cargo test --release -p airun-core --test cpu_bench rounds -- --ignored --nocapture
#[test]
#[ignore]
fn moe_experts_rounds() {
    let (k, ff) = (2560usize, 640usize);
    let (g_size, d_size) = (ff * k / 256 * 98, k * ff / 32 * 18);
    let per = 2 * g_size + d_size;
    let experts = std::env::var("AIRUN_BENCH_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or((2usize << 30) / per);
    let mut w = airun_cuda::PinnedBuffer::<u8>::new(experts * per).unwrap();
    // AIRUN_BENCH_RANDOM=1: random bytes everywhere, scales included (some
    // NaN, infinite or subnormal), as moe_experts_bandwidth fills them.
    let random = std::env::var("AIRUN_BENCH_RANDOM").is_ok_and(|v| v == "1");
    let mut seed = 0x9E37_79B9u64;
    for (i, b) in w.as_mut_slice().iter_mut().enumerate() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // Otherwise sane f16 scales (0x2000 is 2^-7) at every block's start, small codes elsewhere.
        *b = if random { if i % 2 == 0 && seed % 7 == 0 { 0x20 } else { seed as u8 } }
             else if i % 2 == 0 { 0x00 } else { 0x20 };
    }
    let w = w.as_slice();
    // AIRUN_BENCH_NO_THROTTLE=1: opt the process out of Windows' power
    // throttling (EcoQoS) first.
    #[cfg(windows)]
    if std::env::var("AIRUN_BENCH_NO_THROTTLE").is_ok_and(|v| v == "1") {
        #[repr(C)]
        struct State { version: u32, control: u32, state: u32 }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn SetProcessInformation(p: *mut std::ffi::c_void, class: i32, info: *const std::ffi::c_void, size: u32) -> i32;
        }
        let s = State { version: 1, control: 1, state: 0 };
        // SAFETY: ProcessPowerThrottling (4) takes a PROCESS_POWER_THROTTLING_STATE.
        let ok = unsafe { SetProcessInformation(GetCurrentProcess(), 4, &s as *const State as *const _, 12) };
        println!("power throttling opt-out: {}", if ok != 0 { "ok" } else { "failed" });
    }
    // AIRUN_BENCH_WAIT_S: idle that long first.
    if let Some(s) = std::env::var("AIRUN_BENCH_WAIT_S").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::sleep(std::time::Duration::from_secs(s));
    }
    let x: Vec<f32> = (0..k).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
    let mut y = vec![0f32; k];
    let t0 = std::time::Instant::now();
    for round in 0..12 {
        let mut busy = 0.0;
        for c in 0..40 {
            let set: Vec<(&[u8], &[u8], &[u8], f32)> = (0..5).map(|j| {
                let e = (c * 5 + j) * 7 % 160;
                let b = &w[e * per..(e + 1) * per];
                (&b[..g_size], &b[g_size..2 * g_size], &b[2 * g_size..], 0.1)
            }).collect();
            std::thread::sleep(std::time::Duration::from_micros(300));
            let t = std::time::Instant::now();
            airun_core::cpu::moe_experts(GgmlType::IQ3_XXS, GgmlType::IQ4_NL, &set, &x, k, ff, &mut y).unwrap();
            busy += t.elapsed().as_secs_f64();
        }
        let (faults, ws) = page_faults();
        println!("t = {:5.2} s: round {round}: {:7.1} us a call, page faults so far {faults}, working set {} MB",
            t0.elapsed().as_secs_f64(), busy / 40.0 * 1e6, ws >> 20);
    }
}

/// Plain reads (a sum of words, on the CPU pool) of `n` experts' worth of
/// bytes a call at random places in a pinned buffer of AIRUN_BENCH_SPAN_GB
/// (default 1): what the memory system gives that access pattern, without the
/// expert kernels.
///
///     cargo test --release -p airun-core --test cpu_bench raw -- --ignored --nocapture
#[test]
#[ignore]
fn raw_spread_reads() {
    let per = 2 * 627_200 + 921_600;
    let spans: Vec<usize> = std::env::var("AIRUN_BENCH_SPANS").ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect()).unwrap_or_else(|| vec![0, 1, 2, 8]);
    for span in spans {
        let experts = if span == 0 { 160 } else { (span << 30) / per };
        let mut w = airun_cuda::PinnedBuffer::<u64>::new(experts * per / 8).unwrap();
        for (i, v) in w.as_mut_slice().iter_mut().enumerate() {
            *v = i as u64;
        }
        let w = w.as_slice();
        if let Some(s) = std::env::var("AIRUN_BENCH_WAIT_S").ok().and_then(|v| v.parse::<u64>().ok()) {
            std::thread::sleep(std::time::Duration::from_secs(s));
        }
        let words = per / 8;
        for n in [1usize, 5] {
            let calls = 60;
            let mut busy = 0.0;
            for c in 0..calls {
                std::thread::sleep(std::time::Duration::from_micros(300));
                let t = std::time::Instant::now();
                let total = std::sync::atomic::AtomicU64::new(0);
                // 64 KiB chunks, the experts' bytes one after another.
                let chunk = 8192;
                let per_e = words.div_ceil(chunk);
                airun_core::cpu::pool().run(n * per_e, &|i| {
                    let e = ((c * n + i / per_e) * 7) % experts;
                    let lo = e * words + (i % per_e) * chunk;
                    let hi = (lo + chunk).min((e + 1) * words);
                    let s: u64 = w[lo..hi].iter().fold(0u64, |a, &v| a.wrapping_add(v));
                    total.fetch_add(s, std::sync::atomic::Ordering::Relaxed);
                });
                busy += t.elapsed().as_secs_f64();
                std::hint::black_box(total.load(std::sync::atomic::Ordering::Relaxed));
            }
            let s = busy / calls as f64;
            println!("span {:>5} MB, {n} experts a call: {:7.1} us, {:5.1} GB/s", experts * per >> 20, s * 1e6,
                (n * per) as f64 / s / 1e9);
        }
    }
}

#[test]
#[ignore]
fn cpu_matvec_bandwidth() {
    let (rows, k) = (17408usize, 5120usize);
    // (type, elements per block, bytes per block)
    for (ty, block, bytes) in [(GgmlType::NVFP4, 64usize, 36usize), (GgmlType::Q4_K, 256, 144),
                               (GgmlType::Q5_K, 256, 176), (GgmlType::Q6_K, 256, 210), (GgmlType::Q8_0, 32, 34)] {
        let row_bytes = k / block * bytes;
        // Several matrices, so the timing reads DRAM rather than the 30 MB L3.
        let mats = 6;
        let mut w = airun_cuda::PinnedBuffer::<u8>::new(mats * rows * row_bytes).unwrap();
        let mut seed = 0x9E37_79B9u64;
        for (i, b) in w.as_mut_slice().iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // A small positive f16/UE4M3 in the scale bytes, random elsewhere.
            *b = if i % bytes < 4 { 0x38 } else { seed as u8 };
        }
        let x: Vec<f32> = (0..16 * k).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
        let mut y = vec![0f32; 16 * rows];
        let size = rows * row_bytes;
        for n in [1usize, 4, 8] {
            let iters = 3 * mats;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let m = i % mats;
                airun_core::cpu::matvec(ty, &w.as_slice()[m * size..(m + 1) * size], row_bytes, k, rows,
                    &x, n, &mut y).unwrap();
            }
            let s = t.elapsed().as_secs_f64() / iters as f64;
            println!("{ty:>6} {n} tokens: {:7.2} ms a matrix, {:5.1} GB/s of weights", s * 1e3, size as f64 / s / 1e9);
        }
        // As in a decode step: the GPU runs between two CPU matmuls, long
        // enough for the pool's workers to fall asleep.
        let mut busy = 0.0;
        for i in 0..3 * mats {
            let m = i % mats;
            // Sleeping, as the thread waiting on the GPU does.
            std::thread::sleep(std::time::Duration::from_micros(500));
            let t = std::time::Instant::now();
            airun_core::cpu::matvec(ty, &w.as_slice()[m * size..(m + 1) * size], row_bytes, k, rows, &x, 1, &mut y)
                .unwrap();
            busy += t.elapsed().as_secs_f64();
        }
        let s = busy / (3 * mats) as f64;
        println!("{ty:>6} 1 token after 0.5 ms idle: {:7.2} ms a matrix, {:5.1} GB/s", s * 1e3, size as f64 / s / 1e9);
    }
}
