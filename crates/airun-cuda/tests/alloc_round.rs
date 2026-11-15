//! How much VRAM cudaMalloc takes past what is asked (its granularity).
use airun_cuda::{device, DeviceBuffer};

#[test]
#[ignore]
fn alloc_rounding() {
    let Ok(devs) = device::devices() else { return };
    let Some(d) = devs.first() else { return };
    device::set_device(d.ordinal).unwrap();
    for &(bytes, n) in &[(45_711_360usize, 100usize), (20_480, 2000), (3_145_728 + 4096, 300), (700_000, 500)] {
        let (free0, _) = device::mem_info(d.ordinal).unwrap();
        let bufs: Vec<_> = (0..n).map(|_| DeviceBuffer::<u8>::new(d.ordinal, bytes).unwrap()).collect();
        let (free1, _) = device::mem_info(d.ordinal).unwrap();
        let took = free0 - free1;
        println!("{n} x {bytes} B: asked {:.1} MiB, took {:.1} MiB ({:.2} MiB over each)",
            (n * bytes) as f64 / 1048576.0, took as f64 / 1048576.0,
            (took as f64 - (n * bytes) as f64) / n as f64 / 1048576.0);
        drop(bufs);
    }
}
