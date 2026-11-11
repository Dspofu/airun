//! Live GPU readings (utilization, temperature, power, clocks, PCIe traffic)
//! from NVML, the library `nvidia-smi` is built on.
//!
//! NVML ships with the driver (nvml.dll in System32 on Windows,
//! libnvidia-ml.so.1 on Linux), not with the CUDA toolkit, so it is opened at
//! run time: a machine without it still runs models, and only these readings
//! are missing. Nothing here is on the inference path.

use std::ffi::{c_char, c_void, CString};
use std::sync::OnceLock;

type Ret = i32;
type Dev = *mut c_void;
const SUCCESS: Ret = 0;

#[repr(C)]
#[derive(Default)]
struct Utilization {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
#[derive(Default)]
struct Memory {
    total: u64,
    free: u64,
    used: u64,
}

/// nvmlMemory_v2_t: `used` leaves out what the driver reserves for itself
/// (~300 MiB on a 16 GB card), as nvidia-smi shows it; v1 counts it.
#[repr(C)]
#[derive(Default)]
struct MemoryV2 {
    version: u32,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}
/// NVML_STRUCT_VERSION(Memory, 2).
const MEMORY_V2: u32 = std::mem::size_of::<MemoryV2>() as u32 | (2 << 24);

type FnVoid = unsafe extern "C" fn() -> Ret;
type FnHandle = unsafe extern "C" fn(*const c_char, *mut Dev) -> Ret;
type FnU32 = unsafe extern "C" fn(Dev, *mut u32) -> Ret;
type FnArgU32 = unsafe extern "C" fn(Dev, u32, *mut u32) -> Ret;
type FnI32 = unsafe extern "C" fn(Dev, *mut i32) -> Ret;
type FnUtil = unsafe extern "C" fn(Dev, *mut Utilization) -> Ret;
type FnMem = unsafe extern "C" fn(Dev, *mut Memory) -> Ret;
type FnMemV2 = unsafe extern "C" fn(Dev, *mut MemoryV2) -> Ret;
type FnStr = unsafe extern "C" fn(*mut c_char, u32) -> Ret;

/// The entry points used. Each is optional but the handle lookup: an older
/// driver may lack one, and NVML answers NOT_SUPPORTED for what a card does
/// not report (fan speed on a passively cooled card, power on some laptops).
struct Nvml {
    handle: FnHandle,
    utilization: Option<FnUtil>,
    memory: Option<FnMem>,
    memory_v2: Option<FnMemV2>,
    temperature: Option<FnArgU32>,
    power: Option<FnU32>,
    power_limit: Option<FnU32>,
    clock: Option<FnArgU32>,
    max_clock: Option<FnArgU32>,
    fan: Option<FnU32>,
    pcie_throughput: Option<FnArgU32>,
    pcie_gen: Option<FnU32>,
    pcie_width: Option<FnU32>,
    pstate: Option<FnI32>,
    driver: Option<FnStr>,
}

// SAFETY: plain function pointers into a library that is never unloaded;
// NVML's entry points are thread-safe.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

#[cfg(windows)]
mod os {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    pub fn open() -> Option<*mut c_void> {
        // System32 (DCH drivers) first, then the old NVSMI folder.
        for path in ["nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"] {
            let wide: Vec<u16> = path.encode_utf16().chain(Some(0)).collect();
            // SAFETY: `wide` is a NUL-terminated UTF-16 string.
            let h = unsafe { LoadLibraryW(wide.as_ptr()) };
            if !h.is_null() {
                return Some(h);
            }
        }
        None
    }

    pub fn symbol(lib: *mut c_void, name: &std::ffi::CStr) -> *mut c_void {
        // SAFETY: `lib` is a loaded module, `name` NUL-terminated.
        unsafe { GetProcAddress(lib, name.as_ptr().cast()) }
    }
}

#[cfg(not(windows))]
mod os {
    use std::ffi::{c_char, c_int, c_void};

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(lib: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;

    pub fn open() -> Option<*mut c_void> {
        for name in [c"libnvidia-ml.so.1", c"libnvidia-ml.so"] {
            // SAFETY: `name` is NUL-terminated.
            let h = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            if !h.is_null() {
                return Some(h);
            }
        }
        None
    }

    pub fn symbol(lib: *mut c_void, name: &std::ffi::CStr) -> *mut c_void {
        // SAFETY: `lib` is a dlopen handle, `name` NUL-terminated.
        unsafe { dlsym(lib, name.as_ptr()) }
    }
}

fn load() -> Option<Nvml> {
    let lib = os::open()?;
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let p = os::symbol(lib, $name);
            // SAFETY: NVML exports this symbol with exactly this C signature
            // (nvml.h); a null pointer means the driver predates it.
            (!p.is_null()).then(|| unsafe { std::mem::transmute::<*mut c_void, $ty>(p) })
        }};
    }
    let init = sym!(c"nvmlInit_v2", FnVoid)?;
    // SAFETY: no arguments; NVML reference-counts init, and this process
    // never calls shutdown, so the library stays initialized.
    if unsafe { init() } != SUCCESS {
        return None;
    }
    Some(Nvml {
        handle: sym!(c"nvmlDeviceGetHandleByPciBusId_v2", FnHandle)?,
        utilization: sym!(c"nvmlDeviceGetUtilizationRates", FnUtil),
        memory: sym!(c"nvmlDeviceGetMemoryInfo", FnMem),
        memory_v2: sym!(c"nvmlDeviceGetMemoryInfo_v2", FnMemV2),
        temperature: sym!(c"nvmlDeviceGetTemperature", FnArgU32),
        power: sym!(c"nvmlDeviceGetPowerUsage", FnU32),
        power_limit: sym!(c"nvmlDeviceGetEnforcedPowerLimit", FnU32),
        clock: sym!(c"nvmlDeviceGetClockInfo", FnArgU32),
        max_clock: sym!(c"nvmlDeviceGetMaxClockInfo", FnArgU32),
        fan: sym!(c"nvmlDeviceGetFanSpeed", FnU32),
        pcie_throughput: sym!(c"nvmlDeviceGetPcieThroughput", FnArgU32),
        pcie_gen: sym!(c"nvmlDeviceGetCurrPcieLinkGeneration", FnU32),
        pcie_width: sym!(c"nvmlDeviceGetCurrPcieLinkWidth", FnU32),
        pstate: sym!(c"nvmlDeviceGetPerformanceState", FnI32),
        driver: sym!(c"nvmlSystemGetDriverVersion", FnStr),
    })
}

fn nvml() -> Option<&'static Nvml> {
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    NVML.get_or_init(load).as_ref()
}

/// Whether NVML could be opened on this machine.
pub fn available() -> bool {
    nvml().is_some()
}

/// The display driver's version, e.g. "581.57".
pub fn driver_version() -> Option<String> {
    let f = nvml()?.driver?;
    let mut buf = [0 as c_char; 96];
    // SAFETY: `buf` holds as many bytes as passed.
    (unsafe { f(buf.as_mut_ptr(), buf.len() as u32) } == SUCCESS)
        .then(|| unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
}

/// One reading of one GPU. A field is None where the card or driver does not
/// report it.
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// Percent of the last sample period a kernel was running.
    pub util_gpu: Option<u32>,
    /// Percent of that period the memory was being read or written.
    pub util_mem: Option<u32>,
    pub mem_total: Option<u64>,
    pub mem_used: Option<u64>,
    pub temp_c: Option<u32>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub clock_sm_mhz: Option<u32>,
    pub clock_sm_max_mhz: Option<u32>,
    pub clock_mem_mhz: Option<u32>,
    pub fan_pct: Option<u32>,
    /// PCIe traffic in bytes per second, sampled by NVML over ~20 ms.
    pub pcie_tx: Option<u64>,
    pub pcie_rx: Option<u64>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    /// P-state, 0 (most performance) to 15 (least).
    pub pstate: Option<i32>,
}

/// Reads the GPU at `pci_bus_id` ([`crate::Device::pci_bus_id`]). None when
/// NVML is missing or does not know the card. Takes ~50 ms: the PCIe
/// counters are sampled.
pub fn gpu_stats(pci_bus_id: &str) -> Option<GpuStats> {
    let n = nvml()?;
    let id = CString::new(pci_bus_id).ok()?;
    let mut dev: Dev = std::ptr::null_mut();
    // SAFETY: `id` is NUL-terminated, `dev` a valid out-parameter.
    if unsafe { (n.handle)(id.as_ptr(), &mut dev) } != SUCCESS {
        return None;
    }
    // SAFETY (all calls below): `dev` is a live NVML handle and every
    // out-parameter is a valid, initialized local of the type NVML writes.
    let u32_of = |f: Option<FnU32>| f.and_then(|f| {
        let mut v = 0u32;
        (unsafe { f(dev, &mut v) } == SUCCESS).then_some(v)
    });
    let arg_of = |f: Option<FnArgU32>, arg: u32| f.and_then(|f| {
        let mut v = 0u32;
        (unsafe { f(dev, arg, &mut v) } == SUCCESS).then_some(v)
    });
    let util = n.utilization.and_then(|f| {
        let mut u = Utilization::default();
        (unsafe { f(dev, &mut u) } == SUCCESS).then_some(u)
    });
    let mem = n.memory_v2.and_then(|f| {
        let mut m = MemoryV2 { version: MEMORY_V2, ..Default::default() };
        (unsafe { f(dev, &mut m) } == SUCCESS).then_some(Memory { total: m.total, free: m.free, used: m.used })
    }).or_else(|| n.memory.and_then(|f| {
        let mut m = Memory::default();
        (unsafe { f(dev, &mut m) } == SUCCESS).then_some(m)
    }));
    let pstate = n.pstate.and_then(|f| {
        let mut v = 0i32;
        (unsafe { f(dev, &mut v) } == SUCCESS && (0..=15).contains(&v)).then_some(v)
    });
    // nvmlClockType_t: 0 graphics, 1 SM, 2 memory. nvmlPcieUtilCounter_t: 0
    // TX, 1 RX, in KB/s. NVML_TEMPERATURE_GPU = 0.
    Some(GpuStats {
        util_gpu: util.as_ref().map(|u| u.gpu),
        util_mem: util.as_ref().map(|u| u.memory),
        mem_total: mem.as_ref().map(|m| m.total),
        mem_used: mem.as_ref().map(|m| m.used),
        temp_c: arg_of(n.temperature, 0),
        power_w: u32_of(n.power).map(|mw| mw as f64 / 1000.0),
        power_limit_w: u32_of(n.power_limit).map(|mw| mw as f64 / 1000.0),
        clock_sm_mhz: arg_of(n.clock, 1),
        clock_sm_max_mhz: arg_of(n.max_clock, 1),
        clock_mem_mhz: arg_of(n.clock, 2),
        fan_pct: u32_of(n.fan),
        pcie_tx: arg_of(n.pcie_throughput, 0).map(|kb| kb as u64 * 1024),
        pcie_rx: arg_of(n.pcie_throughput, 1).map(|kb| kb as u64 * 1024),
        pcie_gen: u32_of(n.pcie_gen),
        pcie_width: u32_of(n.pcie_width),
        pstate,
    })
}
