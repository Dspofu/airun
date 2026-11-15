//! Times an NVFP4 matmul in Qwen3.8-27B's FFN shape (17408 x 5120) for the
//! token counts a decode, a speculative verify and a batched verify of
//! several conversations feed it: the dp4a kernels (matvec_q8) against the
//! FP4 tensor cores (matmul_nvfp4). Ignored by default; needs a GPU:
//!
//!     cargo test --release -p airun-cuda --test matvec_bench -- --ignored --nocapture
//!
//! AIRUN_BENCH_DEVICE picks the GPU.

use airun_cuda::ops::{self, KernelType};
use airun_cuda::stream::Stream;
use airun_cuda::{DeviceBuffer, device};

const ROWS: usize = 17408;
const K: usize = 5120;

/// The integer kernels by weight type, in the same FFN shape: decode (1), a
/// speculative verify (5), a short prompt tail (16) and prefill batches (128,
/// 512). Rotates over several copies of the matrix so the one-token rows time
/// DRAM and not L2 (one Q4_K copy is 50 MB; the 5070 Ti's L2 is 48):
///
///     cargo test --release -p airun-cuda --test matvec_bench kquant -- --ignored --nocapture
#[test]
#[ignore]
fn kquant_matvec_by_tokens() {
    let dev: i32 = std::env::var("AIRUN_BENCH_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    device::set_device(dev).unwrap();
    let stream = Stream::new(dev).unwrap();
    const COPIES: usize = 6;
    let max_tok = 512;
    let act: Vec<f32> = (0..max_tok * K).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
    let act = DeviceBuffer::<f32>::from_slice(dev, &act, &stream).unwrap();
    let xq = DeviceBuffer::<u8>::new(dev, max_tok * K / 32 * 36 + 64).unwrap();
    let y = DeviceBuffer::<f32>::new(dev, max_tok * ROWS).unwrap();
    // (name, ggml id, bytes per 256 weights)
    for (name, id, block) in [("Q8_0", 8, 272usize), ("Q4_K", 12, 144), ("Q5_K", 13, 176), ("Q6_K", 14, 210),
        ("NVFP4", 40, 144)] {
        let row_bytes = K / 256 * block;
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut w = vec![0u8; ROWS * row_bytes];
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = x as u8;
        }
        let ws: Vec<DeviceBuffer<u8>> =
            (0..COPIES).map(|_| DeviceBuffer::<u8>::from_slice(dev, &w, &stream).unwrap()).collect();
        let gb = (ROWS * row_bytes) as f64 / 1e9;
        let mut line = format!("{name:>6}:");
        let tokens: Vec<usize> = std::env::var("AIRUN_BENCH_TOKENS").ok()
            .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 5, 16, 128, 512]);
        for n in tokens {
            ops::quantize_act(act.as_ptr(), xq.as_mut_ptr(), n * K, &stream).unwrap();
            let run = |i: usize| ops::matvec_q8(KernelType(id), ws[i % COPIES].as_ptr(), xq.as_ptr(), y.as_mut_ptr(),
                K, ROWS, n, row_bytes, &stream).unwrap();
            run(0);
            stream.synchronize().unwrap();
            let iters = if n >= 128 { 12 } else { 60 };
            let t = std::time::Instant::now();
            for i in 0..iters {
                run(i);
            }
            stream.synchronize().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
            line.push_str(&if n <= 16 {
                format!("  {n} tok {us:6.1} us ({:4.0} GB/s)", gb / us * 1e6)
            } else {
                // Integer multiply-adds per second, in units of 1e12.
                format!("  {n} tok {us:7.1} us ({:5.1} T MAC/s)", (ROWS * K * n) as f64 / us / 1e6)
            });
        }
        println!("{line}");
    }
}

#[test]
#[ignore]
fn nvfp4_matvec_by_tokens() {
    let dev: i32 = std::env::var("AIRUN_BENCH_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    device::set_device(dev).unwrap();
    let stream = Stream::new(dev).unwrap();
    let row_bytes = K / 64 * 36;
    // Random nibbles under a 1.0 (UE4M3 0x38) scale per run of 16.
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut w = vec![0u8; ROWS * row_bytes];
    for (i, b) in w.iter_mut().enumerate() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *b = if i % 36 < 4 { 0x38 } else { x as u8 };
    }
    let w = DeviceBuffer::<u8>::from_slice(dev, &w, &stream).unwrap();
    let max_tok = 16;
    let act: Vec<f32> = (0..max_tok * K).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
    let act = DeviceBuffer::<f32>::from_slice(dev, &act, &stream).unwrap();
    let xq = DeviceBuffer::<u8>::new(dev, max_tok * K / 32 * 36 + 64).unwrap();
    let x4 = DeviceBuffer::<u8>::new(dev, ops::nvfp4_act_bytes(max_tok, K)).unwrap();
    let y = DeviceBuffer::<f32>::new(dev, max_tok * ROWS).unwrap();
    let gb = (ROWS * row_bytes) as f64 / 1e9;
    let time = |f: &dyn Fn()| {
        f();
        stream.synchronize().unwrap();
        let iters = 50;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        stream.synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    };
    for n in [1usize, 2, 4, 5, 8, 12, 16] {
        ops::quantize_act(act.as_ptr(), xq.as_mut_ptr(), n * K, &stream).unwrap();
        let q8 = time(&|| ops::matvec_q8(KernelType(40), w.as_ptr(), xq.as_ptr(), y.as_mut_ptr(), K, ROWS, n,
            row_bytes, &stream).unwrap());
        let fp4 = if ops::fp4_supported() && n > 1 {
            ops::quantize_act_nvfp4(act.as_ptr(), x4.as_mut_ptr(), n, K, &stream).unwrap();
            format!("{:8.1} us", time(&|| ops::matmul_nvfp4(w.as_ptr(), row_bytes, x4.as_ptr(), y.as_mut_ptr(), K,
                ROWS, n, &stream).unwrap()))
        } else {
            "       -   ".into()
        };
        println!("{n:>2} tokens: dp4a {q8:8.1} us ({:5.0} GB/s of weights), fp4 tensor cores {fp4}", gb / q8 * 1e6);
    }
}
