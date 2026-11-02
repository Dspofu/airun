//! Times prefill attention alone: a 512-query chunk at the end of a long
//! cache, in Qwen3.8-27B's shape (24 query heads over 4 KV heads, head_dim
//! 256). Ignored by default; it needs a GPU and prints throughput:
//!
//!     cargo test --release -p airun-cuda --test attention_bench -- --ignored --nocapture

use airun_cuda::ops::{self, KvType};
use airun_cuda::stream::Stream;
use airun_cuda::{DeviceBuffer, device};

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const CHUNK: usize = 512;

fn rand_f16_bits(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let v = ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5;
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

/// One decode step's attention (a single query) over a long cache, in GB/s of
/// K and V read: decode attention is bandwidth-bound.
#[test]
#[ignore]
fn decode_attention_bandwidth() {
    // AIRUN_BENCH_DEVICE picks the GPU (the slower card of a split matters).
    let dev: i32 = std::env::var("AIRUN_BENCH_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    device::set_device(dev).unwrap();
    let stream = Stream::new(dev).unwrap();
    for &depth in &[8192usize, 32768, 131072] {
        let ctx = depth;
        let row = HEAD_DIM * 2;
        let kv_bytes = N_KV_HEADS * ctx * row;
        let k = DeviceBuffer::<u8>::from_slice(dev, &rand_f16_bits(kv_bytes / 2, 1), &stream).unwrap();
        let v = DeviceBuffer::<u8>::from_slice(dev, &rand_f16_bits(kv_bytes / 2, 2), &stream).unwrap();
        let q_host: Vec<f32> = (0..N_HEADS * HEAD_DIM).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
        let q = DeviceBuffer::<f32>::from_slice(dev, &q_host, &stream).unwrap();
        let out = DeviceBuffer::<f32>::new(dev, N_HEADS * HEAD_DIM).unwrap();
        let positions = DeviceBuffer::<i32>::from_slice(dev, &[(depth - 1) as i32], &stream).unwrap();
        let splits = ops::attention_splits(ctx);
        let partial = DeviceBuffer::<f32>::new(dev, N_HEADS * splits * (HEAD_DIM + 2)).unwrap();
        let run = || ops::attention_split(out.as_mut_ptr(), &partial, q.as_ptr(), k.as_ptr(), v.as_ptr(),
            positions.as_ptr(), N_HEADS, N_KV_HEADS, HEAD_DIM, ctx, 1.0 / (HEAD_DIM as f32).sqrt(), 0, splits,
            KvType::F16, &stream).unwrap();
        run();
        stream.synchronize().unwrap();
        let iters = 50;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            run();
        }
        stream.synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("decode depth {depth:>6}: {us:8.1} us per layer, {:6.1} GB/s of K+V ({splits} splits)",
            2.0 * kv_bytes as f64 / us / 1e3);
    }
}

#[test]
#[ignore]
fn prefill_attention_throughput() {
    device::set_device(0).unwrap();
    let stream = Stream::new(0).unwrap();
    for &depth in &[8192usize, 32768, 131072] {
        let ctx = depth;
        let row = HEAD_DIM * 2;
        let kv_bytes = N_KV_HEADS * ctx * row;
        let k = DeviceBuffer::<u8>::from_slice(0, &rand_f16_bits(kv_bytes / 2, 1), &stream).unwrap();
        let v = DeviceBuffer::<u8>::from_slice(0, &rand_f16_bits(kv_bytes / 2, 2), &stream).unwrap();
        let q_host: Vec<f32> = (0..CHUNK * N_HEADS * HEAD_DIM).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
        let q = DeviceBuffer::<f32>::from_slice(0, &q_host, &stream).unwrap();
        let out = DeviceBuffer::<f32>::new(0, CHUNK * N_HEADS * HEAD_DIM).unwrap();
        let pos: Vec<i32> = (0..CHUNK).map(|i| (depth - CHUNK + i) as i32).collect();
        let positions = DeviceBuffer::<i32>::from_slice(0, &pos, &stream).unwrap();
        let run = || ops::attention_masked(out.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(), positions.as_ptr(),
            CHUNK, N_HEADS, N_KV_HEADS, HEAD_DIM, ctx, 1.0 / (HEAD_DIM as f32).sqrt(), 0, KvType::F16, None, depth,
            &stream)
            .unwrap();
        run();
        stream.synchronize().unwrap();
        let iters = if depth >= 131072 { 5 } else { 20 };
        let t = std::time::Instant::now();
        for _ in 0..iters {
            run();
        }
        stream.synchronize().unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        // QK^T and PV: 2 matmuls of CHUNK x depth x HEAD_DIM per head (the
        // causal triangle inside the chunk is negligible at these depths).
        let flops = 4.0 * CHUNK as f64 * depth as f64 * HEAD_DIM as f64 * N_HEADS as f64;
        println!("depth {depth:>6}: {ms:8.3} ms per 512-token chunk, {:6.1} TFLOPS", flops / ms / 1e9);
    }
}
