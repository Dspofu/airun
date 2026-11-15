//! A RAM tier for streamed MoE experts: VRAM slots, then this, then the file.
//!
//! With `--disk-offload` the experts that do not fit VRAM were all read from
//! the GGUF on every miss, because mapping them in pinned RAM was all or
//! nothing: Qwen4 Flash-Next needs 24.7 GiB there and a 32 GB machine had 19
//! free, so every miss went to the SSD and reads were 70% of a prompt. This
//! keeps as many of them as the RAM budget holds in pinned memory: the most
//! routed ones. A miss in VRAM that is here uploads straight from
//! it (no read, no copy into staging); one that is not is read from the file
//! into its block here, and uploaded from there.
//!
//! The memory is pinned in chunks as the tier fills, so loading does not wait
//! for gigabytes to be pinned, and a refused allocation just caps the tier.
//!
//! Experts differ in size by layer (the quantization mix), so blocks are
//! reused only by an expert of the same size. Once the budget is spent an
//! expert takes the block of the least routed one of its size, and only if it
//! has been routed more itself. Recency is the wrong rule here: a prompt batch
//! walks every expert of a layer, and with fewer blocks than experts a least
//! recently used tier evicts exactly what the next batch asks for first (Qwen4
//! Flash-Next, 512-token batches: 37 tok/s with it, the same as with no tier).

use std::collections::HashMap;

use airun_cuda::PinnedBuffer;

/// Bytes pinned at a time.
const CHUNK: usize = 256 << 20;

struct Entry {
    /// (tensor, expert); the tensor by the address of its VRAM slots.
    key: (usize, u32),
    ptr: *mut u8,
    /// The pass that last used the block.
    age: u64,
    /// Routings of the expert so far (ExpertRam::routed's count).
    score: f32,
}

pub struct ExpertRam {
    chunks: Vec<PinnedBuffer<u8>>,
    /// The chunk being handed out, its size and the bytes of it given.
    cur: *mut u8,
    cur_len: usize,
    used: usize,
    budget: usize,
    /// Bytes pinned so far.
    held: usize,
    entries: Vec<Entry>,
    map: HashMap<(usize, u32), usize>,
    /// Entry indices by block size.
    by_size: HashMap<usize, Vec<usize>>,
    /// Routings of every expert that missed VRAM, in the tier or not.
    routed: HashMap<(usize, u32), f32>,
    clock: u64,
    /// VRAM misses served from here, and read from the file.
    pub hits: u64,
    pub reads: u64,
    /// Misses the tier had no block for, which the caller read from the file
    /// into its staging buffer.
    pub refused: u64,
}

// SAFETY: the pointers in `entries` and `cur` point into `chunks`, which the tier owns
// and never frees or moves while it lives; nothing else holds them.
unsafe impl Send for ExpertRam {}

/// Where an expert's bytes are for an upload.
pub struct Block {
    pub ptr: *mut u8,
    /// False when the block was just assigned: the caller reads the file into
    /// it before uploading.
    pub filled: bool,
}

impl ExpertRam {
    pub fn new(budget: usize) -> Self {
        Self { chunks: Vec::new(), cur: std::ptr::null_mut(), cur_len: 0, used: 0, budget, held: 0, entries: Vec::new(), map: HashMap::new(),
            by_size: HashMap::new(), routed: HashMap::new(), clock: 0, hits: 0, reads: 0, refused: 0 }
    }

    /// Bytes pinned so far.
    pub fn held(&self) -> usize {
        self.held
    }

    /// Starts a pass: blocks handed out from now on stay put until the next
    /// call, so the uploads the caller queues from them read what it put
    /// there. The caller synchronizes its stream before the next pass.
    pub fn begin(&mut self) {
        self.clock += 1;
    }

    /// The block of `size` bytes holding expert `key`, routed `weight` more
    /// times now, or a block for it to be read into. None when the tier has
    /// no block for it -- no room left and no expert of that size routed less
    /// that this pass is not using: the caller stages the expert as it did
    /// before the tier existed.
    pub fn block(&mut self, key: (usize, u32), size: usize, weight: f32) -> Option<Block> {
        let score = {
            let s = self.routed.entry(key).or_insert(0.0);
            *s += weight;
            *s
        };
        if let Some(&i) = self.map.get(&key) {
            self.entries[i].age = self.clock;
            self.entries[i].score = score;
            self.hits += 1;
            return Some(Block { ptr: self.entries[i].ptr, filled: true });
        }
        let ptr = match self.grow(size) {
            Some(ptr) => {
                self.by_size.entry(size).or_default().push(self.entries.len());
                self.map.insert(key, self.entries.len());
                self.entries.push(Entry { key, ptr, age: self.clock, score });
                ptr
            }
            None => {
                let clock = self.clock;
                let entries = &self.entries;
                let i = *self.by_size.get(&size)?.iter()
                    .filter(|&&i| entries[i].age < clock && entries[i].score < score)
                    .min_by(|&&a, &&b| entries[a].score.total_cmp(&entries[b].score))?;
                if self.map.get(&self.entries[i].key) == Some(&i) {
                    self.map.remove(&self.entries[i].key);
                }
                self.map.insert(key, i);
                self.entries[i].key = key;
                self.entries[i].age = clock;
                self.entries[i].score = score;
                self.entries[i].ptr
            }
        };
        self.reads += 1;
        Some(Block { ptr, filled: false })
    }
    /// Drops an expert whose read into its block failed; the block is the
    /// first of its size to be reused.
    pub fn forget(&mut self, key: (usize, u32)) {
        if let Some(i) = self.map.remove(&key) {
            self.entries[i].age = 0;
            self.entries[i].score = -1.0;
        }
    }

    /// `size` fresh bytes, pinning another chunk when the last one is spent
    /// and the budget allows.
    fn grow(&mut self, size: usize) -> Option<*mut u8> {
        if self.cur_len - self.used < size {
            let want = CHUNK.max(size).min(self.budget.saturating_sub(self.held));
            if want < size {
                return None;
            }
            // Only pinned memory: the budget past what the system pins was
            // tried as ordinary memory and made things worse. Windows stops
            // pinning near half the RAM (13.75 GiB of 32); 5.6 GiB more of
            // plain chunks put the machine in the page file, and the warm-up
            // went from 23 to 92-115 s and generation from 13 to 7 tok/s.
            let base = match PinnedBuffer::<u8>::new(want) {
                Ok(mut c) => {
                    let base = c.as_mut_slice().as_mut_ptr();
                    self.chunks.push(c);
                    base
                }
                Err(_) => {
                    self.budget = self.held;
                    return None;
                }
            };
            self.held += want;
            self.cur = base;
            self.cur_len = want;
            self.used = 0;
        }
        // 16-byte steps keep every block aligned for the copy engine.
        let at = self.used;
        self.used += (size + 15) & !15;
        // SAFETY: at + size <= cur_len, checked above.
        Some(unsafe { self.cur.add(at) })
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Runs only where pinned memory can be allocated (a CUDA build with a GPU).
    fn tier(budget: usize) -> Option<ExpertRam> {
        PinnedBuffer::<u8>::new(16).ok().map(|_| ExpertRam::new(budget))
    }

    #[test]
    fn a_more_routed_expert_takes_the_block_of_the_least_routed_one() {
        let Some(mut t) = tier(3000) else { return };
        t.begin();
        let a = t.block((1, 0), 1000, 1.0).unwrap();
        let b = t.block((1, 1), 1000, 5.0).unwrap();
        assert!(!a.filled && !b.filled && a.ptr != b.ptr);
        // No room for a third, and both are in use by this pass.
        assert!(t.block((1, 2), 1000, 9.0).is_none());
        t.begin();
        // Routed 9 + 1 times by now, against 1 and 5: it takes the block of
        // the least routed, 0.
        let c = t.block((1, 2), 1000, 1.0).unwrap();
        assert!(!c.filled && c.ptr == a.ptr);
        t.begin();
        assert!(t.block((1, 1), 1000, 1.0).unwrap().filled);
        assert!(t.block((1, 2), 1000, 1.0).unwrap().filled);
        // 0 has been routed twice now, less than either: it stays out, and
        // keeps staying out however recently the others were used.
        assert!(t.block((1, 0), 1000, 1.0).is_none());
        t.begin();
        assert!(t.block((1, 0), 1000, 1.0).is_none());
        // A size with no block of its own finds none to reuse.
        assert!(t.block((2, 0), 990, 100.0).is_none());
        assert_eq!((t.hits, t.reads), (2, 3));
    }
}