//! Weights read from the model file on disk (`--disk-offload`).
//!
//! When the GPUs are full the planner keeps layers in pinned RAM (see
//! [`crate::plan::Shard::ram_layers`]). With `--disk-offload`, the layers that
//! do not fit the RAM either are never copied out of the GGUF: their tensors
//! are slices of the file's memory mapping, and the operating system pages
//! them in from the SSD as the CPU kernels (cpu.rs) read them, and drops them
//! again when other memory needs the room. Nothing is pinned for them, so they
//! cost neither RAM nor commit.
//!
//! Page faults alone arrive one at a time, at the SSD's queue-1 speed. A
//! [`Prefetcher`] keeps the next disk layers coming in the background, with
//! several threads and the OS's own read-ahead hint, while the current layer
//! computes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use airun_gguf::GgufFile;

/// Whether some layer's weights are read from the file (set once at load).
/// The runtime then issues passes without CUDA graphs: a copy from mapped
/// file pages cannot be captured.
static IN_USE: AtomicBool = AtomicBool::new(false);

pub fn set_in_use() {
    IN_USE.store(true, Ordering::Relaxed);
}

pub fn in_use() -> bool {
    IN_USE.load(Ordering::Relaxed)
}

/// Physical RAM the system could hand out now (Windows: available physical
/// memory, which includes the standby file cache; Linux: MemAvailable).
/// None where it cannot be read.
pub fn available_ram() -> Option<u64> {
    memory_status().map(|(_, avail)| avail)
}

/// Installed physical RAM.
pub fn total_ram() -> Option<u64> {
    memory_status().map(|(total, _)| total)
}

#[cfg(windows)]
fn memory_status() -> Option<(u64, u64)> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `s` is a properly sized and initialized MEMORYSTATUSEX.
    (unsafe { GlobalMemoryStatusEx(&mut s) } != 0).then_some((s.total_phys, s.avail_phys))
}

#[cfg(not(windows))]
fn memory_status() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(name))?;
        let kb: u64 = line[name.len()..].trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    };
    Some((field("MemTotal:")?, field("MemAvailable:")?))
}

/// This process's memory: (resident in RAM, committed). Committed counts
/// what the process allocated whether or not it is in RAM now (Windows:
/// private bytes; Linux: none, so None).
pub fn process_memory() -> Option<(u64, Option<u64>)> {
    process_memory_impl()
}

#[cfg(windows)]
fn process_memory_impl() -> Option<(u64, Option<u64>)> {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged_pool: usize,
        quota_paged_pool: usize,
        quota_peak_non_paged_pool: usize,
        quota_non_paged_pool: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(process: *mut std::ffi::c_void, counters: *mut Counters, cb: u32) -> i32;
    }
    let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
    // SAFETY: `c` is a PROCESS_MEMORY_COUNTERS_EX with its size in `cb`; the
    // pseudo-handle of the current process needs no closing.
    (unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0)
        .then_some((c.working_set as u64, Some(c.private_usage as u64)))
}

#[cfg(not(windows))]
fn process_memory_impl() -> Option<(u64, Option<u64>)> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = text.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line["VmRSS:".len()..].trim().trim_end_matches("kB").trim().parse().ok()?;
    Some((kb * 1024, None))
}

/// RAM kept free for the system and other programs when deciding what the
/// weights may pin, unless `--ram-reserve` says otherwise: a tenth of the
/// installed RAM, between 2 and 8 GiB.
pub fn default_ram_reserve() -> u64 {
    let tenth = total_ram().unwrap_or(16 << 30) / 10;
    tenth.clamp(2 << 30, 8 << 30)
}

/// Layers ahead of the one running whose pages are fetched in the background
/// (AIRUN_DISK_AHEAD, default 2). Each one in flight holds its bytes of file
/// cache, so a large value only evicts what it fetched.
fn ahead() -> usize {
    static AHEAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AHEAD.get_or_init(|| std::env::var("AIRUN_DISK_AHEAD").ok()
        .and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Bytes one prefetch job covers; the workers take jobs from a shared queue.
const CHUNK: usize = 8 << 20;

struct Queue {
    jobs: VecDeque<(usize, usize)>,
    stop: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    ready: Condvar,
}

/// Brings the file pages of the disk layers into memory ahead of use.
pub struct Prefetcher {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Disk layers in the order a forward pass runs them.
    order: Vec<usize>,
    /// Each one's mapped byte ranges (address, length), by position in `order`.
    ranges: Vec<Vec<(usize, usize)>>,
    /// Positions requested lately, so overlapping windows ask once.
    recent: Mutex<VecDeque<usize>>,
}

impl Prefetcher {
    /// `layers` holds each disk layer and its ranges inside `file`'s mapping,
    /// in run order. The workers keep `file` (and so the mapping) alive until
    /// they have stopped.
    pub fn new(file: &Arc<GgufFile>, layers: Vec<(usize, Vec<(usize, usize)>)>) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue { jobs: VecDeque::new(), stop: false }),
            ready: Condvar::new(),
        });
        let threads = std::env::var("AIRUN_DISK_THREADS").ok().and_then(|v| v.trim().parse().ok())
            .unwrap_or(4usize).max(1);
        let workers = (0..threads).map(|i| {
            let shared = Arc::clone(&shared);
            let file = Arc::clone(file);
            std::thread::Builder::new().name(format!("airun-disk-{i}")).spawn(move || {
                let _keep_mapping = file;
                worker(&shared);
            }).expect("cannot start a disk prefetch thread")
        }).collect();
        let (order, ranges) = layers.into_iter().unzip();
        Self { shared, workers, order, ranges, recent: Mutex::new(VecDeque::new()) }
    }

    /// Bytes of all disk layers.
    pub fn bytes(&self) -> u64 {
        self.ranges.iter().flatten().map(|&(_, len)| len as u64).sum()
    }

    /// Called right before `layer` runs: queues the disk layers after it
    /// (wrapping around to the first ones, which the next token needs), and
    /// it too when nothing fetched it yet (the first pass).
    pub fn before(&self, layer: usize) {
        let Some(at) = self.order.iter().position(|&l| l == layer) else { return };
        let n = self.order.len();
        let window = ahead().min(n);
        let mut recent = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        let mut wanted: Vec<usize> = Vec::new();
        if !recent.contains(&at) {
            wanted.push(at);
        }
        wanted.extend((1..=window).map(|d| (at + d) % n).filter(|p| *p != at && !recent.contains(p)));
        if wanted.is_empty() {
            return;
        }
        for &p in &wanted {
            recent.push_back(p);
        }
        while recent.len() > window.max(1) {
            recent.pop_front();
        }
        drop(recent);
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        // What is still queued was for layers now running or past.
        q.jobs.clear();
        for p in wanted {
            for &(addr, len) in &self.ranges[p] {
                let mut off = 0;
                while off < len {
                    let n = CHUNK.min(len - off);
                    q.jobs.push_back((addr + off, n));
                    off += n;
                }
            }
        }
        drop(q);
        self.shared.ready.notify_all();
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        {
            let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.stop = true;
            q.jobs.clear();
        }
        self.shared.ready.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: &Shared) {
    loop {
        let (addr, len) = {
            let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if q.stop {
                    return;
                }
                if let Some(job) = q.jobs.pop_front() {
                    break job;
                }
                q = shared.ready.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        fetch(addr, len);
    }
}

/// What one prefetch job does to a mapped range: the read-ahead hint, then
/// a read per page. Public for tests/disk_bench.rs.
pub fn fetch(addr: usize, len: usize) {
    hint(addr, len);
    touch(addr, len);
}

/// Reads one byte per page, which faults the range in (or finds it already
/// fetched by [`hint`]).
fn touch(addr: usize, len: usize) {
    let mut sum = 0u8;
    let mut off = 0;
    while off < len {
        // SAFETY: the range lies inside a live read-only mapping (the worker
        // holds the file), and a volatile read keeps the loop from being
        // optimized away.
        sum = sum.wrapping_add(unsafe { std::ptr::read_volatile((addr + off) as *const u8) });
        off += 4096;
    }
    std::hint::black_box(sum);
}

/// Asks the OS to read the range in with large I/Os.
#[cfg(windows)]
fn hint(addr: usize, len: usize) {
    #[repr(C)]
    struct RangeEntry {
        address: *mut std::ffi::c_void,
        bytes: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn PrefetchVirtualMemory(process: *mut std::ffi::c_void, entries: usize, ranges: *const RangeEntry,
                                 flags: u32) -> i32;
    }
    let range = RangeEntry { address: addr as *mut _, bytes: len };
    // SAFETY: a hint over part of a live mapping; failure only means no
    // read-ahead, and touch() faults the pages in anyway.
    unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) };
}

#[cfg(unix)]
fn hint(addr: usize, len: usize) {
    unsafe extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    const MADV_WILLNEED: i32 = 3;
    let page = 4096;
    let start = addr & !(page - 1);
    // SAFETY: a hint over part of a live mapping, page-aligned as madvise
    // requires; failure only means no read-ahead.
    unsafe { madvise(start as *mut _, len + (addr - start), MADV_WILLNEED) };
}

#[cfg(not(any(windows, unix)))]
fn hint(_addr: usize, _len: usize) {}
