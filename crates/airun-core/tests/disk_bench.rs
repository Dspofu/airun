//! How fast weights left on disk (--disk-offload) come in, by read method.
//! Each method reads layers of its own, so every one starts from pages the
//! file cache has not seen in this run; use a model not read since boot (or
//! since the cache was emptied), on the disk you want to measure. Ignored:
//!
//!     $env:AIRUN_DISK_BENCH_FILE="D:\modelo.gguf"
//!     cargo test --release -p airun-core --test disk_bench -- --ignored --nocapture

use std::sync::atomic::{AtomicUsize, Ordering};

use airun_gguf::GgufFile;

/// (address, length) of each tensor of `blk.{l}.`.
fn layer_ranges(f: &GgufFile, l: usize) -> Vec<(usize, usize)> {
    let prefix = format!("blk.{l}.");
    f.tensors().iter().filter(|t| t.name.starts_with(&prefix))
        .map(|t| { let b = f.tensor_bytes(t).unwrap(); (b.as_ptr() as usize, b.len()) })
        .collect()
}

/// Splits ranges into `chunk`-byte jobs and runs `job` on them from
/// `threads` threads; returns GB/s.
fn run(ranges: &[(usize, usize)], threads: usize, chunk: usize, job: &(dyn Fn(usize, usize) + Sync)) -> f64 {
    let mut jobs = Vec::new();
    for &(a, n) in ranges {
        let mut o = 0;
        while o < n {
            jobs.push((a + o, chunk.min(n - o)));
            o += chunk;
        }
    }
    let bytes: usize = ranges.iter().map(|r| r.1).sum();
    let next = AtomicUsize::new(0);
    let t = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(a, n)) = jobs.get(i) else { break };
                job(a, n);
            });
        }
    });
    bytes as f64 / t.elapsed().as_secs_f64() / 1e9
}

fn touch(a: usize, n: usize) {
    let mut s = 0u8;
    let mut o = 0;
    while o < n {
        s = s.wrapping_add(unsafe { std::ptr::read_volatile((a + o) as *const u8) });
        o += 4096;
    }
    std::hint::black_box(s);
}

#[test]
#[ignore]
fn disk_read_methods() {
    let Ok(path) = std::env::var("AIRUN_DISK_BENCH_FILE") else {
        eprintln!("set AIRUN_DISK_BENCH_FILE to a .gguf on the disk to measure");
        return;
    };
    let f = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let n_layers = (0..10_000).take_while(|&l| f.tensor(&format!("blk.{l}.attn_norm.weight")).is_some()).count();
    // Two layers per case, from the end backwards, a different pair each.
    let mut layer = n_layers;
    let mut take = || -> (String, Vec<(usize, usize)>) {
        layer -= 2;
        let r: Vec<_> = (layer..layer + 2).flat_map(|l| layer_ranges(&f, l)).collect();
        let mb = r.iter().map(|x| x.1).sum::<usize>() >> 20;
        (format!("layers {}..{} ({mb} MiB)", layer, layer + 1), r)
    };

    let cases: Vec<(&str, usize, usize, Box<dyn Fn(usize, usize) + Sync>)> = vec![
        ("page faults, 1 thread", 1, 8 << 20, Box::new(touch)),
        ("page faults, 8 threads", 8, 8 << 20, Box::new(touch)),
        ("hint + touch, 4 threads, 8 MiB (the Prefetcher)", 4, 8 << 20, Box::new(airun_core::disk::fetch)),
        ("hint + touch, 8 threads, 8 MiB", 8, 8 << 20, Box::new(airun_core::disk::fetch)),
        ("hint + touch, 16 threads, 4 MiB", 16, 4 << 20, Box::new(airun_core::disk::fetch)),
        ("hint + touch, 1 thread, whole range", 1, usize::MAX, Box::new(airun_core::disk::fetch)),
    ];
    for (name, threads, chunk, job) in &cases {
        let (what, r) = take();
        let gbs = run(&r, *threads, *chunk, job.as_ref());
        eprintln!("{name:<50} {what:<28} {gbs:6.2} GB/s");
    }

    // Plain reads into a buffer through the file cache, and around it.
    #[cfg(windows)]
    {
        use std::os::windows::fs::{FileExt, OpenOptionsExt};
        const NO_BUFFERING: u32 = 0x2000_0000;
        for (name, flags, threads) in [("ReadFile, buffered, 8 threads", 0u32, 8usize),
                                       ("ReadFile, unbuffered, 8 threads", NO_BUFFERING, 8),
                                       ("ReadFile, unbuffered, 16 threads", NO_BUFFERING, 16)] {
            let (what, r) = take();
            // File offsets of the ranges: address minus the mapping's start.
            let map0 = f.tensor_bytes(&f.tensors()[0]).unwrap().as_ptr() as usize - f.tensors()[0].file_offset as usize;
            let file = std::fs::OpenOptions::new().read(true).custom_flags(flags).open(&path).unwrap();
            let chunk = 8usize << 20;
            let gbs = run(&r, threads, chunk, &|a, n| {
                // Sector-aligned, as unbuffered reads need.
                let off = ((a - map0) & !4095) as u64;
                let len = (n + 8192) & !4095;
                let mut buf = vec![0u8; len + 4096];
                let p = buf.as_ptr() as usize;
                let at = (4096 - p % 4096) % 4096;
                let _ = file.seek_read(&mut buf[at..at + len], off);
            });
            eprintln!("{name:<50} {what:<28} {gbs:6.2} GB/s");
        }
    }
}
