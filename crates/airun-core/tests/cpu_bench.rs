//! Times the CPU matmuls (cpu.rs) on the Qwen3.8-27B FFN shape, the weights
//! in pinned memory as the runtime keeps them. Ignored by default:
//!
//!     cargo test --release -p airun-core --test cpu_bench -- --ignored --nocapture

use airun_gguf::GgmlType;

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
