//! A few persistent threads that decode rows of a memory-mapped table in
//! parallel.
//!
//! The PLE table (26.8 GiB on the Qwen4 preview) stays in the file and does
//! not fit the page cache next to everything else, so most rows a token needs
//! are a random read from the SSD. Read one after another, a 792-token prompt's
//! 12,672 rows kept the GPUs idle for 1.4 s; an NVMe drive serves dozens of
//! such reads at once, so the rows are spread over threads that fault their
//! pages in together. The threads persist because a decode step needs only 16
//! rows, too few to pay for spawning threads each time.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Decodes row `i` of a job into its output slot; false when the row is bad.
pub type RowFn = dyn Fn(usize) -> bool + Sync;

struct Job {
    f: *const RowFn,
    n: usize,
}
// SAFETY: the job's closure is Sync and outlives the job: `run` does not
// return until every worker has finished with it.
unsafe impl Send for Job {}

struct State {
    generation: u64,
    job: Option<Job>,
    busy: usize,
    failed: bool,
    quit: bool,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
}

pub struct GatherPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl GatherPool {
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { generation: 0, job: None, busy: 0, failed: false, quit: false }),
            start: Condvar::new(),
            done: Condvar::new(),
        });
        let n = threads.max(1);
        let workers = (0..n).map(|id| {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new().name(format!("airun-gather-{id}")).spawn(move || worker(&shared, id, n))
                .expect("spawning a gather thread")
        }).collect();
        Self { shared, workers }
    }

    /// Calls `f(i)` for every i in 0..n across the threads; false when any
    /// call returned false.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) -> bool + Sync + '_)) -> bool {
        if n == 0 {
            return true;
        }
        // SAFETY: only the lifetime is erased; this call outlives every use
        // of `f`, since it returns only once the workers are done with it.
        let f: *const RowFn = unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) -> bool + Sync + '_), *const RowFn>(f)
        };
        let mut st = self.shared.state.lock().unwrap();
        st.job = Some(Job { f, n });
        st.busy = self.workers.len();
        st.failed = false;
        st.generation += 1;
        self.shared.start.notify_all();
        while st.busy > 0 {
            st = self.shared.done.wait(st).unwrap();
        }
        st.job = None;
        !st.failed
    }
}

fn worker(shared: &Shared, id: usize, n_workers: usize) {
    let mut seen = 0u64;
    loop {
        let (f, n) = {
            let mut st = shared.state.lock().unwrap();
            while st.generation == seen && !st.quit {
                st = shared.start.wait(st).unwrap();
            }
            if st.quit {
                return;
            }
            seen = st.generation;
            let job = st.job.as_ref().expect("a new generation carries a job");
            (job.f, job.n)
        };
        // SAFETY: see Job.
        let f = unsafe { &*f };
        let mut ok = true;
        let mut i = id;
        while i < n {
            ok &= f(i);
            i += n_workers;
        }
        let mut st = shared.state.lock().unwrap();
        st.failed |= !ok;
        st.busy -= 1;
        if st.busy == 0 {
            shared.done.notify_all();
        }
    }
}

impl Drop for GatherPool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().quit = true;
        self.shared.start.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_index_runs_once_and_failures_report() {
        let pool = GatherPool::new(5);
        for n in [0usize, 1, 16, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            assert!(pool.run(n, &|i| { hits[i].fetch_add(1, Ordering::Relaxed); true }));
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
        assert!(!pool.run(10, &|i| i != 7));
        assert!(pool.run(3, &|_| true));
    }
}
