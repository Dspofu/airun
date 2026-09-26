//! CUDA device discovery.

use crate::error::{CudaError, Result};
use crate::ffi;

/// NVIDIA GPU generations, derived from compute capability.
///
/// airun targets Turing and newer: that is the first generation with the
/// independent-thread-scheduling and `__shfl_sync` semantics the kernels assume,
/// and it lines up with the RTX 20-series floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Generation {
    /// Pre-Turing: Pascal, Volta and older. Not supported.
    Legacy,
    /// SM 7.5 -- RTX 20 / GTX 16.
    Turing,
    /// SM 8.0/8.6/8.7 -- A100, RTX 30.
    Ampere,
    /// SM 8.9 -- RTX 40, L40.
    AdaLovelace,
    /// SM 9.0 -- H100.
    Hopper,
    /// SM 10.x/12.x -- B200, RTX 50.
    Blackwell,
    /// Newer than anything this build knows about.
    Future,
}

impl Generation {
    pub fn from_compute_capability(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (m, _) if m < 7 => Generation::Legacy,
            (7, n) if n < 5 => Generation::Legacy, // Volta
            (7, _) => Generation::Turing,
            (8, 9) => Generation::AdaLovelace,
            (8, _) => Generation::Ampere,
            (9, _) => Generation::Hopper,
            (10 | 11 | 12, _) => Generation::Blackwell,
            _ => Generation::Future,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Generation::Legacy => "pre-Turing",
            Generation::Turing => "Turing",
            Generation::Ampere => "Ampere",
            Generation::AdaLovelace => "Ada Lovelace",
            Generation::Hopper => "Hopper",
            Generation::Blackwell => "Blackwell",
            Generation::Future => "newer than this build",
        }
    }

    pub fn is_supported(self) -> bool {
        self >= Generation::Turing
    }

    /// Whether the generation has hardware bf16 arithmetic. Turing does not, so
    /// bf16 weights are widened to f32 there instead of computed natively.
    pub fn has_native_bf16(self) -> bool {
        self >= Generation::Ampere
    }
}

/// A CUDA device as airun sees it.
#[derive(Debug, Clone)]
pub struct Device {
    /// CUDA ordinal. Note this is *not* the nvidia-smi index: CUDA orders by
    /// speed by default, so the two disagree on mixed-GPU machines.
    pub ordinal: i32,
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_sm: i32,
    pub l2_cache_bytes: i64,
    pub mem_bus_width_bits: i32,
    pub shared_mem_per_block: u64,
    pub shared_mem_per_sm: u64,
    pub total_mem: u64,
    pub free_mem: u64,
    pub async_engine_count: i32,
    pub unified_addressing: bool,
    pub integrated: bool,
    pub pci_domain_id: i32,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub uuid: [u8; 16],
}

impl Device {
    pub fn generation(&self) -> Generation {
        Generation::from_compute_capability(self.cc_major, self.cc_minor)
    }

    /// `sm_XY`, the form nvcc's `-arch` flag wants.
    pub fn sm_arch(&self) -> String {
        format!("sm_{}{}", self.cc_major, self.cc_minor)
    }

    pub fn compute_capability(&self) -> String {
        format!("{}.{}", self.cc_major, self.cc_minor)
    }

    /// Stable identifier across reboots and driver reorderings, unlike the
    /// ordinal. This is what `--gpus` accepts for unambiguous selection.
    pub fn pci_bus_id(&self) -> String {
        format!(
            "{:08X}:{:02X}:{:02X}.0",
            self.pci_domain_id, self.pci_bus_id, self.pci_device_id
        )
    }

    pub fn uuid_string(&self) -> String {
        let h: String = self.uuid.iter().map(|b| format!("{b:02x}")).collect();
        format!(
            "GPU-{}-{}-{}-{}-{}",
            &h[0..8],
            &h[8..12],
            &h[12..16],
            &h[16..20],
            &h[20..32]
        )
    }

    /// Rough capability score used to rank devices when splitting a model.
    ///
    /// Memory bandwidth dominates decode throughput, and bus width times SM
    /// count is the best proxy available without a clock query (CUDA 13 removed
    /// `memoryClockRate` from the properties struct).
    pub fn perf_score(&self) -> f64 {
        let sm = self.sm_count as f64;
        let bus = self.mem_bus_width_bits.max(64) as f64;
        sm * (bus / 128.0).sqrt()
    }

    /// Re-query free memory, which changes as other processes come and go.
    pub fn refresh_memory(&mut self) -> Result<()> {
        let (free, total) = mem_info(self.ordinal)?;
        self.free_mem = free;
        self.total_mem = total;
        Ok(())
    }
}

/// Number of visible CUDA devices. Returns 0 rather than erroring when the
/// driver reports no device, so callers can print a friendly message.
pub fn device_count() -> Result<i32> {
    let mut n = 0i32;
    // SAFETY: `n` is a valid, initialized out-parameter.
    let status = unsafe { ffi::airun_device_count(&mut n) };
    match status {
        ffi::AIRUN_OK => Ok(n),
        // cudaErrorNoDevice: an answer, not a failure.
        100 => Ok(0),
        other => Err(CudaError::from_status(other)),
    }
}

pub fn device(ordinal: i32) -> Result<Device> {
    let count = device_count()?;
    if ordinal < 0 || ordinal >= count {
        return Err(CudaError::BadOrdinal {
            ordinal,
            available: count,
        });
    }
    let mut raw = ffi::DeviceInfoRaw::default();
    // SAFETY: `raw` is a valid, fully initialized out-parameter of the right type.
    ffi::check(unsafe { ffi::airun_device_info(ordinal, &mut raw) })?;
    Ok(Device {
        ordinal: raw.ordinal,
        name: ffi::fixed_str(&raw.name),
        cc_major: raw.cc_major,
        cc_minor: raw.cc_minor,
        sm_count: raw.sm_count,
        warp_size: raw.warp_size,
        max_threads_per_block: raw.max_threads_per_block,
        max_threads_per_sm: raw.max_threads_per_sm,
        l2_cache_bytes: raw.l2_cache_bytes as i64,
        mem_bus_width_bits: raw.mem_bus_width_bits,
        shared_mem_per_block: raw.shared_mem_per_block,
        shared_mem_per_sm: raw.shared_mem_per_sm,
        total_mem: raw.total_mem,
        free_mem: raw.free_mem,
        async_engine_count: raw.async_engine_count,
        unified_addressing: raw.unified_addressing != 0,
        integrated: raw.integrated != 0,
        pci_domain_id: raw.pci_domain_id,
        pci_bus_id: raw.pci_bus_id,
        pci_device_id: raw.pci_device_id,
        uuid: raw.uuid,
    })
}

/// Every visible device, in CUDA ordinal order.
pub fn devices() -> Result<Vec<Device>> {
    let count = device_count()?;
    (0..count).map(device).collect()
}

pub fn mem_info(ordinal: i32) -> Result<(u64, u64)> {
    let (mut free, mut total) = (0u64, 0u64);
    // SAFETY: both out-parameters are valid and initialized.
    ffi::check(unsafe { ffi::airun_mem_info(ordinal, &mut free, &mut total) })?;
    Ok((free, total))
}

pub fn can_access_peer(from: i32, to: i32) -> Result<bool> {
    let mut can = 0i32;
    // SAFETY: `can` is a valid out-parameter.
    ffi::check(unsafe { ffi::airun_can_access_peer(from, to, &mut can) })?;
    Ok(can != 0)
}

/// Enables direct peer addressing where the hardware allows it.
///
/// Returns the pairs that were enabled. Failures are not errors: consumer cards
/// under WDDM routinely refuse, and the transfer path falls back to staging
/// through host memory.
pub fn enable_all_peer_access(devices: &[Device]) -> Vec<(i32, i32)> {
    let mut enabled = Vec::new();
    for a in devices {
        for b in devices {
            if a.ordinal == b.ordinal {
                continue;
            }
            if can_access_peer(a.ordinal, b.ordinal).unwrap_or(false) {
                // SAFETY: both ordinals came from a successful enumeration.
                if unsafe { ffi::airun_enable_peer_access(a.ordinal, b.ordinal) } == ffi::AIRUN_OK {
                    enabled.push((a.ordinal, b.ordinal));
                }
            }
        }
    }
    enabled
}

pub fn set_device(ordinal: i32) -> Result<()> {
    // SAFETY: plain integer argument; the shim validates the ordinal.
    ffi::check(unsafe { ffi::airun_set_device(ordinal) })
}

pub fn synchronize() -> Result<()> {
    // SAFETY: no arguments.
    ffi::check(unsafe { ffi::airun_device_synchronize() })
}

/// CUDA runtime version this binary was built against, as (major, minor).
pub fn runtime_version() -> (i32, i32) {
    // SAFETY: no arguments, returns a plain integer.
    let v = unsafe { ffi::airun_runtime_version() };
    (v / 1000, (v % 1000) / 10)
}

/// CUDA version the installed driver supports, as (major, minor).
pub fn driver_version() -> (i32, i32) {
    // SAFETY: no arguments, returns a plain integer.
    let v = unsafe { ffi::airun_driver_version() };
    (v / 1000, (v % 1000) / 10)
}

/// True when this binary carries compiled GPU kernels.
pub const fn built_with_cuda() -> bool {
    cfg!(airun_cuda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_capability_maps_to_generation() {
        assert_eq!(Generation::from_compute_capability(6, 1), Generation::Legacy);
        assert_eq!(Generation::from_compute_capability(7, 0), Generation::Legacy);
        assert_eq!(Generation::from_compute_capability(7, 5), Generation::Turing);
        assert_eq!(Generation::from_compute_capability(8, 6), Generation::Ampere);
        assert_eq!(
            Generation::from_compute_capability(8, 9),
            Generation::AdaLovelace
        );
        assert_eq!(Generation::from_compute_capability(9, 0), Generation::Hopper);
        assert_eq!(
            Generation::from_compute_capability(12, 0),
            Generation::Blackwell
        );
    }

    #[test]
    fn rtx20_is_the_supported_floor() {
        assert!(!Generation::from_compute_capability(7, 0).is_supported());
        assert!(Generation::from_compute_capability(7, 5).is_supported());
        assert!(!Generation::Turing.has_native_bf16());
        assert!(Generation::Ampere.has_native_bf16());
    }
}
