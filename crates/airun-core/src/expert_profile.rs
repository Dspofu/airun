//! How often each MoE expert is routed to, kept per model across runs.
//!
//! When the experts do not all fit in VRAM, the ones that stay there decide
//! how much of every token crosses PCIe. Routing is far from uniform -- a few
//! dozen experts of a layer's 512 take most of its tokens -- but which ones is
//! only known by running the model. The runtime counts every routed expert
//! (the MoE's `moe_count`), adds the counts here, and saves them when it is
//! done; the next load puts each layer's most used experts in VRAM from the
//! start instead of the first `slots` by index, and the adaptive rebalance
//! only has to follow what changes. The profile learns the user's own
//! workload, not a fixed one shipped with the program.
//!
//! The file sits in the user's cache directory, named after the model file
//! and its size; `AIRUN_EXPERT_PROFILE=<path>` puts it elsewhere and
//! `AIRUN_EXPERT_PROFILE=off` neither reads nor writes one.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"AIRUNEXP";
const VERSION: u32 = 1;

/// Routing counts per (layer, expert), and where they are saved.
pub struct ExpertProfile {
    path: PathBuf,
    model_len: u64,
    n_layers: usize,
    n_experts: usize,
    counts: Vec<f64>,
    /// Whether anything was added since the last save.
    dirty: bool,
}

impl ExpertProfile {
    /// The profile of the model at `model` (its first file) of `model_len`
    /// bytes: the saved one when it matches the model, an empty one otherwise.
    /// None when profiles are turned off or no cache directory is known.
    pub fn open(model: &Path, model_len: u64, n_layers: usize, n_experts: usize) -> Option<Self> {
        let path = match std::env::var("AIRUN_EXPERT_PROFILE") {
            Ok(v) if v == "off" => return None,
            Ok(v) if !v.is_empty() => PathBuf::from(v),
            _ => {
                let name = model.file_stem()?.to_string_lossy().to_string();
                cache_dir()?.join("airun").join("profiles").join(format!("{name}-{model_len}.experts"))
            }
        };
        let mut p = Self { path, model_len, n_layers, n_experts, counts: vec![0.0; n_layers * n_experts], dirty: false };
        if let Some(c) = p.read() {
            p.counts = c;
        }
        Some(p)
    }

    fn read(&self) -> Option<Vec<f64>> {
        let mut f = std::fs::File::open(&self.path).ok()?;
        let mut head = [0u8; 8 + 4 + 4 + 4 + 8];
        f.read_exact(&mut head).ok()?;
        let u32_at = |o: usize| u32::from_le_bytes(head[o..o + 4].try_into().unwrap()) as usize;
        if &head[..8] != MAGIC || u32_at(8) != VERSION as usize || u32_at(12) != self.n_layers
            || u32_at(16) != self.n_experts || u64::from_le_bytes(head[20..28].try_into().unwrap()) != self.model_len {
            return None;
        }
        let mut body = vec![0u8; self.counts.len() * 8];
        f.read_exact(&mut body).ok()?;
        Some(body.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect())
    }

    /// Whether the profile holds any counts for layer `l`.
    pub fn knows(&self, l: usize) -> bool {
        l < self.n_layers && self.layer(l).iter().any(|&c| c > 0.0)
    }

    fn layer(&self, l: usize) -> &[f64] {
        &self.counts[l * self.n_experts..(l + 1) * self.n_experts]
    }

    /// Each of layer `l`'s experts' share of its routings, None when the
    /// profile has none for it.
    pub fn shares(&self, l: usize) -> Option<Vec<f32>> {
        if !self.knows(l) {
            return None;
        }
        let c = self.layer(l);
        let total: f64 = c.iter().sum();
        Some(c.iter().map(|&x| (x / total) as f32).collect())
    }

    /// Layer `l`'s experts, most routed first (ties by index).
    pub fn order(&self, l: usize) -> Vec<usize> {
        let c = self.layer(l);
        let mut order: Vec<usize> = (0..self.n_experts).collect();
        order.sort_by(|&a, &b| c[b].total_cmp(&c[a]).then(a.cmp(&b)));
        order
    }

    /// Adds one layer's routing counts.
    pub fn add(&mut self, l: usize, counts: &[u32]) {
        if l >= self.n_layers || counts.len() != self.n_experts {
            return;
        }
        for (c, &n) in self.counts[l * self.n_experts..(l + 1) * self.n_experts].iter_mut().zip(counts) {
            *c += f64::from(n);
        }
        self.dirty = self.dirty || counts.iter().any(|&n| n > 0);
    }

    /// Writes the profile when it changed, through a temporary file so an
    /// interrupted save never leaves a torn one behind.
    pub fn save(&mut self) -> std::io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension("experts.tmp");
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(MAGIC)?;
            f.write_all(&VERSION.to_le_bytes())?;
            f.write_all(&(self.n_layers as u32).to_le_bytes())?;
            f.write_all(&(self.n_experts as u32).to_le_bytes())?;
            f.write_all(&self.model_len.to_le_bytes())?;
            for c in &self.counts {
                f.write_all(&c.to_le_bytes())?;
            }
            f.flush()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        self.dirty = false;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The per-user cache directory: %LOCALAPPDATA% on Windows, $XDG_CACHE_HOME
/// or ~/.cache elsewhere.
fn cache_dir() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if cfg!(windows) {
        var("LOCALAPPDATA")
    } else {
        var("XDG_CACHE_HOME").or_else(|| var("HOME").map(|h| h.join(".cache")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_round_trips_and_orders_by_use() {
        let dir = std::env::temp_dir().join(format!("airun-profile-test-{}", std::process::id()));
        let path = dir.join("m.experts");
        let mut p = ExpertProfile { path: path.clone(), model_len: 7, n_layers: 2, n_experts: 4,
            counts: vec![0.0; 8], dirty: false };
        p.add(1, &[1, 5, 0, 5]);
        assert_eq!(p.order(1), vec![1, 3, 0, 2]);
        assert!(p.knows(1) && !p.knows(0));
        p.save().unwrap();
        let q = ExpertProfile { path: path.clone(), model_len: 7, n_layers: 2, n_experts: 4,
            counts: vec![0.0; 8], dirty: false };
        assert_eq!(q.read().unwrap(), p.counts);
        // Another model (size) or shape does not read it.
        let r = ExpertProfile { path, model_len: 8, n_layers: 2, n_experts: 4, counts: vec![0.0; 8], dirty: false };
        assert!(r.read().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
