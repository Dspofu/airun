//! Host -> GPU bandwidth over PCIe: the DMA engine (cudaMemcpyAsync from
//! pinned memory) against the expert stage copy (`moe_stage_copy`, kernels
//! reading mapped host memory), for the expert sizes and counts a MoE decode
//! step copies. Ignored by default; needs a GPU:
//!
//!     cargo test --release -p airun-cuda --test pcie_bench -- --ignored --nocapture
//!
//! AIRUN_BENCH_DEVICE picks the GPU; AIRUN_STAGE_STREAMS the copy's blocks
//! (0: AIRUN_STAGE_BLOCKS blocks per expert), read once per process.

use airun_cuda::hybrid;
use airun_cuda::stream::Stream;
use airun_cuda::{DeviceBuffer, PinnedBuffer, device};

/// One expert's gate (or up) projection on Qwen4 Flash-Next: 640 rows of 2560
/// IQ3_XXS values are 627,200 bytes.
const STRIDE: usize = 640 << 10;
const EXPERTS: usize = 64;

#[test]
#[ignore]
fn host_to_device_bandwidth() {
    let dev: i32 = std::env::var("AIRUN_BENCH_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    device::set_device(dev).unwrap();
    let stream = Stream::new(dev).unwrap();
    let mut host = PinnedBuffer::<u8>::new(EXPERTS * STRIDE).unwrap();
    for (i, b) in host.as_mut_slice().iter_mut().enumerate() {
        *b = (i * 31 % 251) as u8;
    }
    let dst = DeviceBuffer::<u8>::new(dev, EXPERTS * STRIDE).unwrap();
    let time = |f: &dyn Fn(), bytes: usize| {
        f();
        stream.synchronize().unwrap();
        let iters = 20;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        stream.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64() / iters as f64;
        (s * 1e6, bytes as f64 / s / 1e9)
    };

    for n in [1usize, 6, 10, 64] {
        let bytes = n * STRIDE;
        let (us, gbs) = time(&|| dst.copy_from_host_at(0, &host.as_slice()[..bytes], &stream).unwrap(), bytes);
        println!("DMA, one copy of {n:2} experts ({:5.1} MB): {us:8.1} us  {gbs:5.1} GB/s", bytes as f64 / 1e6);
        let (us, gbs) = time(&|| {
            for e in 0..n {
                dst.copy_from_host_at(e * STRIDE, &host.as_slice()[e * STRIDE..(e + 1) * STRIDE], &stream).unwrap();
            }
        }, bytes);
        println!("DMA, {n:2} copies of one expert:          {us:8.1} us  {gbs:5.1} GB/s");
    }

    let base = host.device_ptr(dev).unwrap() as u64;
    let tab: Vec<u64> = (0..EXPERTS).map(|e| base + (e * STRIDE) as u64).collect();
    let tab = DeviceBuffer::<u64>::from_slice(dev, &tab, &stream).unwrap();
    let list: Vec<i32> = (0..EXPERTS as i32).collect();
    let list = DeviceBuffer::<i32>::from_slice(dev, &list, &stream).unwrap();
    for n in [1usize, 2, 3, 6, 10, 20, 64] {
        let n_list = DeviceBuffer::<i32>::from_slice(dev, &[n as i32], &stream).unwrap();
        let (us, gbs) = time(&|| hybrid::moe_stage_copy(tab.as_ptr(), list.as_ptr(), n_list.as_ptr(),
            std::ptr::null(), dst.as_mut_ptr(), STRIDE, n, &stream).unwrap(), n * STRIDE);
        println!("stage copy kernel, {n:2} experts:             {us:8.1} us  {gbs:5.1} GB/s");
    }
    stream.synchronize().unwrap();
    drop(dst);
}
