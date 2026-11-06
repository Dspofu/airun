//! `airun gpu` -- report the CUDA devices airun can see and what it can do with them.

use airun_cuda::device::{self, Generation};
use anyhow::Result;
use clap::Args;

#[derive(Args)]
pub struct GpuArgs {
    /// Emit machine-readable JSON instead of a report.
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: GpuArgs) -> Result<()> {
    if !device::built_with_cuda() {
        anyhow::bail!(
            "this airun binary was built without CUDA kernels.\n\
             Install the CUDA toolkit (11.8 or newer), make sure CUDA_PATH points at it, \
             and rebuild with `cargo build --release`."
        );
    }

    let devices = device::devices()?;
    let (rt_major, rt_minor) = device::runtime_version();
    let (drv_major, drv_minor) = device::driver_version();

    if args.json {
        return emit_json(&devices, (rt_major, rt_minor), (drv_major, drv_minor));
    }

    println!(
        "CUDA runtime {rt_major}.{rt_minor}  |  driver supports up to {drv_major}.{drv_minor}"
    );

    if devices.is_empty() {
        println!("\nNo CUDA device found.");
        println!("  - Check that an NVIDIA GPU is present and `nvidia-smi` works.");
        println!("  - A driver older than the CUDA runtime above will also hide every device.");
        return Ok(());
    }

    // The ordinal/nvidia-smi mismatch bites everyone with mixed GPUs exactly
    // once, so say it up front rather than in a footnote.
    println!(
        "\n{} CUDA device(s). Ordinals below are CUDA's, which need not match nvidia-smi's order.\n",
        devices.len()
    );

    for h in crate::hide::hidden() {
        println!("hidden (--hide-gpu / AIRUN_HIDE_GPUS): {h}");
    }
    if !crate::hide::hidden().is_empty() {
        println!();
    }

    for d in &devices {
        let generation = d.generation();
        let status = if !generation.is_supported() {
            "  [UNSUPPORTED]"
        } else {
            ""
        };
        println!("GPU {} - {}{}", d.ordinal, d.name, status);
        println!(
            "  arch          {} (compute {}, {})",
            d.sm_arch(),
            d.compute_capability(),
            generation.name()
        );
        println!(
            "  memory        {} free of {}",
            airun_cuda::human_bytes(d.free_mem),
            airun_cuda::human_bytes(d.total_mem)
        );
        println!(
            "  compute       {} SMs, {} threads/SM, {}-bit bus, {} L2",
            d.sm_count,
            d.max_threads_per_sm,
            d.mem_bus_width_bits,
            airun_cuda::human_bytes(d.l2_cache_bytes as u64)
        );
        println!(
            "  features      bf16 {}, copy engines {}, unified addressing {}",
            yes_no(generation.has_native_bf16()),
            d.async_engine_count,
            yes_no(d.unified_addressing)
        );
        println!("  pci           {}", d.pci_bus_id());
        println!("  uuid          {}", d.uuid_string());

        if !generation.is_supported() {
            println!(
                "  note          airun needs compute 7.5 (RTX 20-series / Turing) or newer; \
                 this device is {}",
                generation.name()
            );
        }
        println!();
    }

    if devices.len() > 1 {
        println!("peer-to-peer access:");
        let mut any_p2p = false;
        for a in &devices {
            for b in &devices {
                if a.ordinal >= b.ordinal {
                    continue;
                }
                let ab = device::can_access_peer(a.ordinal, b.ordinal).unwrap_or(false);
                let ba = device::can_access_peer(b.ordinal, a.ordinal).unwrap_or(false);
                let both = ab && ba;
                any_p2p |= both;
                println!(
                    "  GPU {} <-> GPU {}   {}",
                    a.ordinal,
                    b.ordinal,
                    if both {
                        "direct"
                    } else {
                        "via host memory"
                    }
                );
            }
        }
        if !any_p2p {
            println!(
                "\n  No direct peer access. Cross-GPU tensors are staged through pinned host\n  \
                 memory, so a layer split that cuts between GPUs costs a PCIe round trip per\n  \
                 boundary. Prefer --split-mode layer (few boundaries) over tensor splitting."
            );
        }
    }

    let usable: Vec<_> = devices
        .iter()
        .filter(|d| d.generation().is_supported())
        .collect();
    let total_vram: u64 = usable.iter().map(|d| d.free_mem).sum();
    println!(
        "\nusable for airun: {} device(s), {} of free VRAM in aggregate",
        usable.len(),
        airun_cuda::human_bytes(total_vram)
    );

    Ok(())
}

fn emit_json(
    devices: &[airun_cuda::Device],
    runtime: (i32, i32),
    driver: (i32, i32),
) -> Result<()> {
    use serde_json::json;

    let list: Vec<_> = devices
        .iter()
        .map(|d| {
            json!({
                "ordinal": d.ordinal,
                "name": d.name,
                "compute_capability": d.compute_capability(),
                "sm_arch": d.sm_arch(),
                "generation": d.generation().name(),
                "supported": d.generation().is_supported(),
                "native_bf16": d.generation().has_native_bf16(),
                "total_mem": d.total_mem,
                "free_mem": d.free_mem,
                "sm_count": d.sm_count,
                "mem_bus_width_bits": d.mem_bus_width_bits,
                "l2_cache_bytes": d.l2_cache_bytes,
                "async_engine_count": d.async_engine_count,
                "pci_bus_id": d.pci_bus_id(),
                "uuid": d.uuid_string(),
                "perf_score": d.perf_score(),
            })
        })
        .collect();

    let mut peers = Vec::new();
    for a in devices {
        for b in devices {
            if a.ordinal >= b.ordinal {
                continue;
            }
            peers.push(json!({
                "a": a.ordinal,
                "b": b.ordinal,
                "direct": device::can_access_peer(a.ordinal, b.ordinal).unwrap_or(false)
                    && device::can_access_peer(b.ordinal, a.ordinal).unwrap_or(false),
            }));
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "cuda_runtime": format!("{}.{}", runtime.0, runtime.1),
            "cuda_driver_max": format!("{}.{}", driver.0, driver.1),
            "devices": list,
            "peer_access": peers,
        }))?
    );
    Ok(())
}

fn yes_no(v: bool) -> &'static str {
    if v { "yes" } else { "no" }
}

/// Devices airun will actually use, newest-capable first.
///
/// Unsupported generations are filtered out here so every downstream planner can
/// assume compute 7.5 or better.
pub fn usable_devices() -> Result<Vec<airun_cuda::Device>> {
    let all = device::devices()?;
    let usable: Vec<_> = all
        .into_iter()
        .filter(|d| d.generation().is_supported())
        .collect();
    if usable.is_empty() {
        anyhow::bail!(
            "no usable CUDA device: airun needs compute capability 7.5 or newer \
             (RTX 20-series / Turing and up). Run `airun gpu` to see what was detected."
        );
    }
    Ok(usable)
}

/// Silences the dead-code lint until the loader starts calling this.
const _: fn() -> Result<Vec<airun_cuda::Device>> = usable_devices;

/// Restates the supported floor for error messages elsewhere.
pub const MIN_GENERATION: Generation = Generation::Turing;
