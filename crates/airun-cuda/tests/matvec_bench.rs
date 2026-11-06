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
