//! Times the CPU NVFP4 matmul (cpu.rs) on the Qwen3.8-27B FFN shape, the
//! weights in pinned memory as the runtime keeps them. Ignored by default:
//!
//!     cargo test --release -p airun-core --test cpu_bench -- --ignored --nocapture

use airun_gguf::GgmlType;

#[test]
#[ignore]
fn nvfp4_cpu_matvec_bandwidth() {
    let (rows, k) = (17408usize, 5120usize);
    let row_bytes = k / 64 * 36;
    // Several matrices, so the timing reads DRAM rather than the 30 MB L3.
    let mats = 8;
    let mut w = airun_cuda::PinnedBuffer::<u8>::new(mats * rows * row_bytes).unwrap();
    let mut seed = 0x9E37_79B9u64;
    for (i, b) in w.as_mut_slice().iter_mut().enumerate() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        *b = if i % 36 < 4 { 0x38 } else { seed as u8 };
    }
    let x: Vec<f32> = (0..16 * k).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.3).collect();
    let mut y = vec![0f32; 16 * rows];
    let bytes = rows * row_bytes;
    for n in [1usize, 2, 3, 4, 8] {
        let iters = 4 * mats;
        let t = std::time::Instant::now();
        for i in 0..iters {
            let m = i % mats;
            airun_core::cpu::matvec(GgmlType::NVFP4, &w.as_slice()[m * bytes..(m + 1) * bytes], row_bytes, k, rows,
                &x, n, &mut y).unwrap();
        }
        let s = t.elapsed().as_secs_f64() / iters as f64;
        println!("{n} tokens: {:7.2} ms a matrix, {:5.1} GB/s of weights", s * 1e3, bytes as f64 / s / 1e9);
    }
}
