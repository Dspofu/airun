//! Long-lived reader threads for streamed MoE experts.
//!
//! A decode step reads a handful of experts in each of 40 layers. Spawning the
//! readers per layer cost more than the reads themselves on a warm page cache,
//! so the threads live as long as the runtime, each owning one file handle --
//! Windows serializes I/O issued through a single synchronous handle, so
//! sharing one would put the drive back at queue depth 1. A split model gives
//! every thread one handle per shard.

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

/// One positional read into memory the caller keeps alive until `read_all`
/// returns.
struct Job {
    file: usize,
    offset: u64,
    dst: *mut u8,
    len: usize,
}

// SAFETY: a Job only reaches a worker inside `read_all`, which holds the
// exclusive borrow of every destination and blocks until each worker reports
// back, so the pointer never outlives or aliases the slice it came from.
unsafe impl Send for Job {}

pub struct ExpertReader {
    lanes: Vec<Sender<Vec<Job>>>,
    done: Receiver<io::Result<()>>,
    workers: Vec<JoinHandle<()>>,
}

impl ExpertReader {
    pub fn new(paths: &[&Path], threads: usize) -> io::Result<Self> {
        let (done_tx, done) = mpsc::channel();
        let mut lanes = Vec::with_capacity(threads);
        let mut workers = Vec::with_capacity(threads);
        for i in 0..threads.max(1) {
            let files = paths.iter().map(File::open).collect::<io::Result<Vec<_>>>()?;
            let (tx, rx) = mpsc::channel::<Vec<Job>>();
            let done_tx = done_tx.clone();
            let worker = std::thread::Builder::new()
                .name(format!("airun-expert-reader-{i}"))
                .spawn(move || {
                    for batch in rx {
                        let result = batch.into_iter().try_for_each(|job| {
                            // SAFETY: see `Job`; read_all keeps the slice borrowed
                            // and unaliased until this batch's result is received.
                            let buf = unsafe { std::slice::from_raw_parts_mut(job.dst, job.len) };
                            let file = files.get(job.file).ok_or_else(|| io::Error::other("an expert names a shard that was not opened"))?;
                            read_exact_at(file, buf, job.offset)
                        });
                        if done_tx.send(result).is_err() {
                            break;
                        }
                    }
                })?;
            lanes.push(tx);
            workers.push(worker);
        }
        Ok(Self { lanes, done, workers })
    }

    /// Fills every `(shard, offset, buffer)` from the files, spreading them
    /// over the workers, and returns once all have finished.
    pub fn read_all(&self, reads: Vec<(usize, u64, &mut [u8])>) -> io::Result<()> {
        if reads.is_empty() {
            return Ok(());
        }
        let n = self.lanes.len().min(reads.len());
        let mut batches: Vec<Vec<Job>> = (0..n).map(|_| Vec::new()).collect();
        // The pieces are near-equal in size, so round-robin balances.
        for (i, (file, offset, buf)) in reads.into_iter().enumerate() {
            batches[i % n].push(Job { file, offset, dst: buf.as_mut_ptr(), len: buf.len() });
        }
        let mut sent = 0;
        let mut first_error = None;
        for (lane, batch) in self.lanes.iter().zip(batches) {
            match lane.send(batch) {
                Ok(()) => sent += 1,
                Err(_) => {
                    first_error.get_or_insert(io::Error::other("an expert reader thread exited"));
                    break;
                }
            }
        }
        // Every dispatched batch must report back before the borrows end, even
        // after an error, or a worker could still be writing into them.
        for _ in 0..sent {
            match self.done.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => { first_error.get_or_insert(e); }
                Err(_) => { first_error.get_or_insert(io::Error::other("an expert reader thread exited")); }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl Drop for ExpertReader {
    fn drop(&mut self) {
        self.lanes.clear();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(file, buf, offset)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(file, buf, offset)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_scattered_ranges_across_workers() {
        let dir = std::env::temp_dir().join(format!("airun-reader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        let data: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();

        let reader = ExpertReader::new(&[&path], 3).unwrap();
        let mut bufs = vec![vec![0u8; 5000]; 7];
        let offsets = [0u64, 12_345, 50_000, 95_000, 1, 77_777, 33_333];
        for _ in 0..3 {
            let reads = bufs.iter_mut().zip(offsets).map(|(b, o)| (0, o, b.as_mut_slice())).collect();
            reader.read_all(reads).unwrap();
            for (b, o) in bufs.iter().zip(offsets) {
                assert_eq!(b.as_slice(), &data[o as usize..o as usize + 5000]);
            }
        }
        // A read past the end fails without poisoning the pool.
        let mut tail = vec![0u8; 10];
        assert!(reader.read_all(vec![(0, 99_995, tail.as_mut_slice())]).is_err());
        let mut ok = vec![0u8; 10];
        reader.read_all(vec![(0, 10, ok.as_mut_slice())]).unwrap();
        assert_eq!(ok.as_slice(), &data[10..20]);

        drop(reader);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
