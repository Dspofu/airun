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

/// A speculative verify's 4 queries over one long q4_0 cache: the split
/// kernel (attention_split_multi) against the prefill kernel it replaced and
/// four one-query decodes, per layer.
#[test]
#[ignore]
fn verify_attention_by_depth() {
    let dev: i32 = std::env::var("AIRUN_BENCH_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    device::set_device(dev).unwrap();
    let stream = Stream::new(dev).unwrap();
    let n = 4usize;
    let ctx = 32768usize;
    let kv = KvType::Q4_0;
    let row = kv.row_bytes(HEAD_DIM);
    let k = DeviceBuffer::<u8>::zeroed(dev, N_KV_HEADS * ctx * row, &stream).unwrap();
    let v = DeviceBuffer::<u8>::zeroed(dev, N_KV_HEADS * ctx * row, &stream).unwrap();
    let q_host: Vec<f32> = (0..n * N_HEADS * HEAD_DIM).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
    let q = DeviceBuffer::<f32>::from_slice(dev, &q_host, &stream).unwrap();
    let out = DeviceBuffer::<f32>::new(dev, n * N_HEADS * HEAD_DIM).unwrap();
    let splits = ops::attention_splits(ctx);
    let partial = DeviceBuffer::<f32>::new(dev, n * N_HEADS * splits * (HEAD_DIM + 2)).unwrap();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for &depth in &[512usize, 8192, 25000] {
        let pos: Vec<i32> = (0..n as i32).map(|i| depth as i32 - n as i32 + i).collect();
        let positions = DeviceBuffer::<i32>::from_slice(dev, &pos, &stream).unwrap();
        let time = |f: &dyn Fn()| {
            f();
            stream.synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..50 {
                f();
            }
            stream.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e6 / 50.0
        };
        let split = time(&|| ops::attention_split_multi(out.as_mut_ptr(), partial.as_mut_ptr(), q.as_ptr(), k.as_ptr(),
            v.as_ptr(), positions.as_ptr(), N_HEADS, N_KV_HEADS, HEAD_DIM, ctx, scale, 0, splits, kv, n, &stream).unwrap());
        let masked = time(&|| ops::attention_masked(out.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(),
            positions.as_ptr(), n, N_HEADS, N_KV_HEADS, HEAD_DIM, ctx, scale, 0, kv, None, depth, &stream).unwrap());
        let one = time(&|| ops::attention_split(out.as_mut_ptr(), &partial, q.as_ptr(), k.as_ptr(), v.as_ptr(),
            positions.as_ptr(), N_HEADS, N_KV_HEADS, HEAD_DIM, ctx, scale, 0, splits, kv, &stream).unwrap());
        println!("depth {depth:>6}: 4 queries split {split:8.1} us, prefill kernel {masked:8.1} us; 1 query {one:7.1} us");
    }
}

/// The first queries of a verify come out bit-identical whether 2 or 4 ride
/// in the block (a lone request verifies 4, the same one in a --parallel step
/// fewer).
#[test]
#[ignore]
fn verify_attention_ignores_the_drafts_count() {
    let dev: i32 = std::env::var("AIRUN_BENCH_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    device::set_device(dev).unwrap();
    let stream = Stream::new(dev).unwrap();
    let ctx = 4096usize;
    let kv = KvType::Q4_0;
    let row = kv.row_bytes(HEAD_DIM);
    let kb: Vec<u8> = (0..N_KV_HEADS * ctx * row).map(|i| if i % 18 < 2 { 0x20 } else { (i * 7 % 251) as u8 }).collect();
    let k = DeviceBuffer::<u8>::from_slice(dev, &kb, &stream).unwrap();
    let v = DeviceBuffer::<u8>::from_slice(dev, &kb, &stream).unwrap();
    let q_host: Vec<f32> = (0..4 * N_HEADS * HEAD_DIM).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
    let q = DeviceBuffer::<f32>::from_slice(dev, &q_host, &stream).unwrap();
    let splits = ops::attention_splits(ctx);
    let partial = DeviceBuffer::<f32>::new(dev, 4 * N_HEADS * splits * (HEAD_DIM + 2)).unwrap();
    let pos = DeviceBuffer::<i32>::from_slice(dev, &[3000, 3001, 3002, 3003], &stream).unwrap();
    let run = |n: usize| {
        let out = DeviceBuffer::<f32>::new(dev, n * N_HEADS * HEAD_DIM).unwrap();
        ops::attention_split_multi(out.as_mut_ptr(), partial.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(),
            pos.as_ptr(), N_HEADS, N_KV_HEADS, HEAD_DIM, ctx, 0.0625, 0, splits, kv, n, &stream).unwrap();
        out.to_vec(&stream).unwrap()
    };
    let (four, two, one) = (run(4), run(2), run(1));
    let w = 2 * N_HEADS * HEAD_DIM;
    let diff2 = four[..w].iter().zip(&two[..w]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let diff1 = four[..w / 2].iter().zip(&one[..w / 2]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    println!("rows 0-1, 4 vs 2 queries: {diff2} of {w} values differ; row 0, 4 vs 1: {diff1}");
}
