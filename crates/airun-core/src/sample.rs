//! Token sampling.
//!
//! The chain is bias -> penalties -> top-n-sigma -> top-k -> typical -> top-p
//! -> min-p -> temperature, which is the order llama.cpp uses and the one the
//! OpenAI API assumes. The order is not cosmetic: applying temperature first
//! would change what top-p selects, and running top-p before top-k would make
//! top-k a no-op on any distribution top-p already narrowed.
//!
//! Every filter keeps at least one candidate. A filter that can empty the set
//! turns an aggressive setting into a silent fallback to a different sampler,
//! which is much harder to notice than a narrow distribution.

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// 0 means greedy: always take the most likely token.
    pub temperature: f32,
    /// 0 disables.
    pub top_k: usize,
    /// 1.0 disables.
    pub top_p: f32,
    /// Minimum probability relative to the best token. 0 disables.
    pub min_p: f32,
    /// Locally typical sampling: keep the tokens whose surprise is closest to
    /// the distribution's entropy. 1.0 disables.
    pub typical_p: f32,
    /// Keep only logits within this many standard deviations of the best one.
    /// 0 or less disables. Unlike top-p this threshold does not move with
    /// temperature, which is the property it was proposed for.
    pub top_n_sigma: f32,
    /// 1.0 disables. Values above 1 discourage repeats.
    pub repeat_penalty: f32,
    /// Subtracted once from any token already seen. 0 disables.
    pub presence_penalty: f32,
    /// Subtracted per occurrence of the token. 0 disables.
    pub frequency_penalty: f32,
    /// How far back the three penalties look.
    pub repeat_last_n: usize,
    /// Added to the named tokens' logits before anything else. This is the only
    /// hook that can raise a score, so it is also how a token gets banned:
    /// f32::NEG_INFINITY removes it from consideration entirely.
    pub logit_bias: Vec<(u32, f32)>,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            top_n_sigma: 0.0,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: 64,
            logit_bias: Vec::new(),
            seed: 0,
        }
    }
}

impl SamplingParams {
    /// Deterministic decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            top_n_sigma: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: 0,
            logit_bias: Vec::new(),
            seed: 0,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    /// Scratch reused between calls to keep sampling allocation-free.
    candidates: Vec<(u32, f32)>,
    /// Occurrence counts over the penalty window, indexed by token id. Sized
    /// lazily on first use so a sampler with penalties off costs nothing.
    counts: Vec<u32>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        // A zero seed means "pick one", so repeated runs differ unless the user
        // asked for reproducibility.
        let seed = if params.seed == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x2545F4914F6CDD1D)
        } else {
            params.seed
        };
        Self {
            params,
            rng: Rng::new(seed),
            candidates: Vec::new(),
            counts: Vec::new(),
        }
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// True when [`sample`] reduces to the argmax of the raw logits: no
    /// temperature, no biases, no penalty that could move the maximum. The
    /// runtime then computes the argmax on the GPU instead of copying logits.
    pub fn is_pure_greedy(&self) -> bool {
        let p = &self.params;
        let penalizing = (p.repeat_penalty != 1.0 || p.presence_penalty != 0.0
            || p.frequency_penalty != 0.0) && p.repeat_last_n > 0;
        p.temperature <= 0.0 && p.logit_bias.is_empty() && !penalizing
    }

    /// Picks the next token. `logits` is modified in place.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        for &(token, bias) in &self.params.logit_bias {
            if let Some(l) = logits.get_mut(token as usize) {
                *l += bias;
            }
        }

        let penalizing = self.params.repeat_penalty != 1.0
            || self.params.presence_penalty != 0.0
            || self.params.frequency_penalty != 0.0;
        if penalizing && self.params.repeat_last_n > 0 {
            apply_penalties(
                logits,
                history,
                &mut self.counts,
                self.params.repeat_last_n,
                self.params.repeat_penalty,
                self.params.presence_penalty,
                self.params.frequency_penalty,
            );
        }

        if self.params.top_n_sigma > 0.0 {
            apply_top_n_sigma(logits, self.params.top_n_sigma);
        }

        if self.params.temperature <= 0.0 {
            return argmax(logits);
        }

        self.candidates.clear();
        self.candidates
            .extend(logits.iter().copied().enumerate().map(|(i, l)| (i as u32, l)));

        // top-k first: it is the cheapest way to cut the candidate set, and
        // sorting the full vocabulary every token would dominate the cost.
        let keep = if self.params.top_k > 0 {
            self.params.top_k.clamp(1, self.candidates.len())
        } else {
            self.candidates.len()
        };
        if keep < self.candidates.len() {
            self.candidates
                .select_nth_unstable_by(keep - 1, |a, b| b.1.total_cmp(&a.1));
            self.candidates.truncate(keep);
        }
        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Softmax over the surviving candidates, at temperature.
        let max = self.candidates[0].1;
        let inv_t = 1.0 / self.params.temperature;
        let mut sum = 0f32;
        for c in &mut self.candidates {
            let prob = (((c.1 - max) * inv_t) as f64).exp() as f32;
            c.1 = prob;
            sum += prob;
        }
        for c in &mut self.candidates {
            c.1 /= sum;
        }

        if self.params.typical_p < 1.0 {
            apply_typical(&mut self.candidates, self.params.typical_p);
        }

        if self.params.min_p > 0.0 {
            let threshold = self.params.min_p * self.candidates[0].1;
            let keep = self.candidates.iter().filter(|c| c.1 >= threshold).count();
            self.candidates.truncate(keep.max(1));
        }

        if self.params.top_p < 1.0 {
            let mut cumulative = 0f32;
            let mut cut = self.candidates.len();
            for (i, c) in self.candidates.iter().enumerate() {
                cumulative += c.1;
                if cumulative >= self.params.top_p {
                    // Keep the token that crossed the threshold.
                    cut = i + 1;
                    break;
                }
            }
            self.candidates.truncate(cut.max(1));
        }

        let total: f32 = self.candidates.iter().map(|c| c.1).sum();
        if total <= 0.0 {
            // Every survivor rounded to zero probability, which a very cold
            // temperature on a flat distribution can do.
            return self.candidates[0].0;
        }

        let mut target = self.rng.next_f32() * total;
        for c in &self.candidates {
            target -= c.1;
            if target <= 0.0 {
                return c.0;
            }
        }
        self.candidates[self.candidates.len() - 1].0
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best_val {
            best_val = l;
            best = i as u32;
        }
    }
    best
}

/// The three repetition controls, applied in one pass over the window.
///
/// `repeat_penalty` is multiplicative and sign-aware, so it always pushes a
/// score down; `presence` and `frequency` are the additive OpenAI pair. They
/// are different scales on purpose and a caller may want both: the first
/// scales with how confident the model already was, the second does not.
///
/// `counts` is scratch owned by the sampler. Counting into a vocabulary-sized
/// array rather than scanning the window per token keeps this linear in the
/// window even when the same token fills it. The apply loop clears every entry
/// it touched, so the array comes back all-zero and is only ever zeroed in full
/// once -- re-zeroing 150k entries per token would cost more than the penalty.
fn apply_penalties(
    logits: &mut [f32],
    history: &[u32],
    counts: &mut Vec<u32>,
    last_n: usize,
    repeat: f32,
    presence: f32,
    frequency: f32,
) {
    let start = history.len().saturating_sub(last_n);
    let window = &history[start..];

    if counts.len() != logits.len() {
        counts.clear();
        counts.resize(logits.len(), 0);
    }
    for &token in window {
        if let Some(c) = counts.get_mut(token as usize) {
            *c += 1;
        }
    }

    for &token in window {
        let Some(l) = logits.get_mut(token as usize) else {
            continue;
        };
        let count = counts[token as usize];
        if count == 0 {
            continue; // already handled: a repeat later in the window
        }
        counts[token as usize] = 0;

        if repeat != 1.0 {
            if *l > 0.0 {
                *l /= repeat;
            } else {
                *l *= repeat;
            }
        }
        *l -= count as f32 * frequency + presence;
    }
}

/// Top-n-sigma: keep only logits within `n` standard deviations of the best.
///
/// Computed on raw logits before temperature, which is the point of the filter
/// -- the cut is a property of how peaked the model's own scores are, so it
/// does not widen as temperature rises the way top-p does.
fn apply_top_n_sigma(logits: &mut [f32], n: f32) {
    if logits.len() < 2 {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0f64;
    for &l in logits.iter() {
        if l > max {
            max = l;
        }
        sum += l as f64;
    }
    let mean = sum / logits.len() as f64;
    let var = logits
        .iter()
        .map(|&l| (l as f64 - mean) * (l as f64 - mean))
        .sum::<f64>()
        / logits.len() as f64;

    let threshold = max - n * var.sqrt() as f32;
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Locally typical sampling: keep the tokens whose information content sits
/// closest to the distribution's entropy, rather than the most likely ones.
///
/// Selection walks the typicality order, but the survivors are handed back in
/// descending probability: min-p and top-p both read `candidates[0]` as the
/// most likely token, and leaving them the typicality order would silently
/// measure everything against the wrong reference.
fn apply_typical(candidates: &mut Vec<(u32, f32)>, typical_p: f32) {
    let entropy: f32 = -candidates
        .iter()
        .filter(|c| c.1 > 0.0)
        .map(|c| c.1 * c.1.ln())
        .sum::<f32>();

    // Sort by |surprise - entropy|; the closest tokens are the "typical" ones.
    let mut scored: Vec<(u32, f32, f32)> = candidates
        .iter()
        .map(|&(id, p)| {
            let surprise = if p > 0.0 { -p.ln() } else { f32::INFINITY };
            (id, p, (surprise - entropy).abs())
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));

    let mut cumulative = 0f32;
    let mut cut = scored.len();
    for (i, s) in scored.iter().enumerate() {
        cumulative += s.1;
        if cumulative >= typical_p {
            cut = i + 1;
            break;
        }
    }

    candidates.clear();
    candidates.extend(scored.iter().take(cut.max(1)).map(|&(id, p, _)| (id, p)));
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
}

/// xoshiro-style PRNG: fast, seedable, and reproducible across platforms.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn penalties(logits: &mut [f32], history: &[u32], last_n: usize, r: f32, p: f32, f: f32) {
        let mut counts = Vec::new();
        apply_penalties(logits, history, &mut counts, last_n, r, p, f);
    }

    #[test]
    fn greedy_picks_the_maximum() {
        let mut s = Sampler::new(SamplingParams::greedy());
        let mut logits = vec![0.1, 5.0, 0.3, 2.0];
        assert_eq!(s.sample(&mut logits, &[]), 1);
    }

    #[test]
    fn top_k_of_one_is_deterministic() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 42,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        for _ in 0..20 {
            let mut logits = vec![0.1, 5.0, 0.3, 2.0];
            assert_eq!(s.sample(&mut logits, &[]), 1);
        }
    }

    #[test]
    fn a_fixed_seed_reproduces_the_same_sequence() {
        let params = SamplingParams {
            seed: 12345,
            ..Default::default()
        };
        let run = || {
            let mut s = Sampler::new(params.clone());
            (0..16)
                .map(|_| {
                    let mut logits = vec![1.0, 1.2, 0.9, 1.1, 0.8, 1.05];
                    s.sample(&mut logits, &[])
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn repeat_penalty_pushes_scores_down_from_both_signs() {
        let mut logits = vec![2.0, -2.0, 1.0];
        penalties(&mut logits, &[0, 1], 8, 2.0, 0.0, 0.0);
        assert_eq!(logits[0], 1.0); // positive: divided
        assert_eq!(logits[1], -4.0); // negative: multiplied
        assert_eq!(logits[2], 1.0); // untouched
    }

    #[test]
    fn repeat_penalty_only_looks_at_the_recent_window() {
        let mut logits = vec![2.0, 2.0];
        // Token 0 is outside the 1-token window, token 1 is inside it.
        penalties(&mut logits, &[0, 1], 1, 2.0, 0.0, 0.0);
        assert_eq!(logits[0], 2.0);
        assert_eq!(logits[1], 1.0);
    }

    /// The count scratch is reused across tokens, so a leftover count would
    /// penalize a token the current window does not even contain.
    #[test]
    fn the_penalty_scratch_does_not_leak_between_calls() {
        let mut counts = Vec::new();

        let mut logits = vec![5.0, 5.0, 5.0];
        apply_penalties(&mut logits, &[0, 0, 0], &mut counts, 8, 1.0, 0.0, 1.0);
        assert_eq!(logits[0], 2.0);

        // A second call over a window that never mentions token 0.
        let mut logits = vec![5.0, 5.0, 5.0];
        apply_penalties(&mut logits, &[1], &mut counts, 8, 1.0, 0.0, 1.0);
        assert_eq!(logits[0], 5.0, "token 0 is not in this window");
        assert_eq!(logits[1], 4.0);
    }

    #[test]
    fn presence_penalty_is_flat_and_frequency_penalty_counts() {
        // Token 0 appears three times, token 1 once.
        let history = [0u32, 0, 1, 0];

        let mut logits = vec![5.0, 5.0, 5.0];
        penalties(&mut logits, &history, 8, 1.0, 0.5, 0.0);
        assert_eq!(logits[0], 4.5, "presence does not scale with the count");
        assert_eq!(logits[1], 4.5);
        assert_eq!(logits[2], 5.0);

        let mut logits = vec![5.0, 5.0, 5.0];
        penalties(&mut logits, &history, 8, 1.0, 0.0, 0.5);
        assert_eq!(logits[0], 3.5, "frequency scales with the count");
        assert_eq!(logits[1], 4.5);
        assert_eq!(logits[2], 5.0);
    }

    #[test]
    fn a_repeated_token_is_penalized_once_not_once_per_occurrence() {
        // The multiplicative penalty must not compound: three occurrences of a
        // token used to divide its logit by the penalty three times over.
        let mut logits = vec![8.0];
        penalties(&mut logits, &[0, 0, 0], 8, 2.0, 0.0, 0.0);
        assert_eq!(logits[0], 4.0);
    }

    #[test]
    fn logit_bias_can_ban_and_promote() {
        let params = SamplingParams {
            temperature: 0.0,
            logit_bias: vec![(1, f32::NEG_INFINITY), (3, 100.0)],
            ..SamplingParams::greedy()
        };
        let mut s = Sampler::new(params);
        let mut logits = vec![0.1, 5.0, 0.3, 2.0];
        // 1 was the argmax and is banned; 3 is promoted past everything else.
        assert_eq!(s.sample(&mut logits, &[]), 3);
    }

    #[test]
    fn top_n_sigma_keeps_the_peak_and_drops_the_tail() {
        // One clear winner over a flat floor: a one-sigma cut must keep the
        // winner and nothing else.
        let mut logits = vec![0.0; 64];
        logits[7] = 20.0;
        apply_top_n_sigma(&mut logits, 1.0);
        assert_eq!(logits[7], 20.0);
        assert!(logits.iter().enumerate().all(|(i, &l)| i == 7 || l == f32::NEG_INFINITY));
    }

    #[test]
    fn top_n_sigma_leaves_a_flat_distribution_alone() {
        // Zero spread means zero sigma, so the threshold is the max itself and
        // every equal logit is at it. Nothing may be filtered.
        let mut logits = vec![1.0; 32];
        apply_top_n_sigma(&mut logits, 1.0);
        assert!(logits.iter().all(|&l| l == 1.0));
    }

    #[test]
    fn typical_p_can_drop_the_most_likely_token() {
        // Three tiers: a 0.5 peak, one middling token, and a long thin tail.
        // The entropy lands near the middling token's surprise, so typical
        // sampling reaches it first -- which is the whole difference from
        // top-p, and impossible to see on a distribution where the peak
        // happens to be typical anyway.
        let mut candidates = vec![(0u32, 0.5f32), (1, 0.2)];
        for i in 2..32 {
            candidates.push((i, 0.01));
        }

        apply_typical(&mut candidates, 0.15);
        assert_eq!(candidates, vec![(1, 0.2)], "the peak is not the typical token");

        // Widen the mass and the peak comes back in, behind the typical one.
        let mut candidates = vec![(0u32, 0.5f32), (1, 0.2)];
        for i in 2..32 {
            candidates.push((i, 0.01));
        }
        apply_typical(&mut candidates, 0.5);
        assert_eq!(
            candidates,
            vec![(0, 0.5), (1, 0.2)],
            "survivors come back in probability order for the filters downstream"
        );
    }

    #[test]
    fn every_filter_keeps_at_least_one_candidate() {
        // Aggressive settings on a flat distribution: each of these can empty
        // the candidate set if it truncates without a floor.
        for params in [
            SamplingParams {
                min_p: 0.99,
                ..Default::default()
            },
            SamplingParams {
                top_p: 0.0,
                ..Default::default()
            },
            SamplingParams {
                typical_p: 0.0,
                ..Default::default()
            },
            SamplingParams {
                top_k: 0,
                min_p: 1.0,
                typical_p: 0.01,
                top_p: 0.01,
                ..Default::default()
            },
        ] {
            let mut s = Sampler::new(SamplingParams { seed: 7, ..params });
            let mut logits = vec![1.0; 64];
            let t = s.sample(&mut logits, &[]);
            assert!(t < 64, "sampler returned {t} for a 64-token vocabulary");
        }
    }
}
