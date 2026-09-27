//! LRU slots for exact MoE offload. A whole selection is pinned before eviction,
//! so a later miss cannot replace an expert the current token still needs.

use crate::error::{CoreError, Result};

pub struct ExpertSlots {
    experts: Vec<Option<i32>>,
    ages: Vec<u64>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
}

impl ExpertSlots {
    pub fn new(capacity: usize) -> Self {
        Self { experts: vec![None; capacity], ages: vec![0; capacity], clock: 0, hits: 0, misses: 0 }
    }

    pub fn select(&mut self, ids: &[i32], n_experts: usize) -> Result<(Vec<i32>, Vec<(usize, usize)>)> {
        if ids.iter().any(|&id| id < 0 || id as usize >= n_experts) {
            return Err(CoreError::Other("router produced an invalid expert id; run airun selftest".into()));
        }
        let mut unique = ids.to_vec();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() > self.experts.len() {
            return Err(CoreError::Other(format!("{} selected experts exceed {} cache slots; increase --expert-cache", unique.len(), self.experts.len())));
        }
        self.clock += 1;
        let mut uploads = Vec::new();
        let mut slots = Vec::with_capacity(ids.len());
        for &id in ids {
            let slot = if let Some(slot) = self.experts.iter().position(|&e| e == Some(id)) {
                self.hits += 1;
                slot
            } else {
                let slot = (0..self.experts.len())
                    .filter(|&s| self.experts[s].is_none_or(|e| !unique.contains(&e)))
                    .min_by_key(|&s| (self.experts[s].is_some(), self.ages[s]))
                    .expect("selection fits and only selected experts are pinned");
                self.experts[slot] = Some(id);
                self.misses += 1;
                uploads.push((slot, id as usize));
                slot
            };
            self.ages[slot] = self.clock;
            slots.push(slot as i32);
        }
        Ok((slots, uploads))
    }

    pub fn invalidate(&mut self) {
        self.experts.fill(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protects_hits_that_appear_after_misses() {
        let mut c = ExpertSlots::new(3);
        c.select(&[0, 1, 2], 8).unwrap();
        let (slots, uploads) = c.select(&[3, 0, 2], 8).unwrap();
        assert_eq!(slots, [1, 0, 2]);
        assert_eq!(uploads, [(1, 3)]);
        assert!(c.select(&[2, 3, 0], 8).unwrap().1.is_empty());
    }

    #[test]
    fn invalid_selection_does_not_change_cache() {
        let mut c = ExpertSlots::new(2);
        c.select(&[1, 2], 8).unwrap();
        assert!(c.select(&[-1], 8).is_err());
        assert!(c.select(&[8], 8).is_err());
        assert!(c.select(&[1, 2, 3], 8).is_err());
        assert!(c.select(&[1, 2, 1], 8).unwrap().1.is_empty());
        c.invalidate();
        assert_eq!(c.select(&[1, 2], 8).unwrap().1.len(), 2);
    }
}
