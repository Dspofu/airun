//! Token sampling.
//!
//! The chain is grammar -> bias -> penalties -> DRY -> top-n-sigma -> top-k ->
//! typical -> min-p -> top-p -> XTC -> temperature (dynamic when asked), the
//! order llama.cpp uses and the one the OpenAI API assumes; mirostat, when on,
//! replaces everything after DRY, as in llama.cpp. The order is not cosmetic:
//! applying temperature first would change what top-p selects, and running
//! top-p before top-k would make top-k a no-op on any distribution top-p
//! already narrowed.
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
    /// Dynamic temperature: the temperature moves within +-range with the
    /// candidates' normalized entropy raised to `exponent`. 0 disables.
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    /// XTC ("exclude top choices"): with this probability, every candidate at
    /// or above `xtc_threshold` but the least likely of them is removed. 0
    /// disables.
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    /// DRY ("don't repeat yourself"): a token that would extend a sequence
    /// already seen in the last `dry_penalty_last_n` tokens (0: all of them)
    /// loses `multiplier * base^(length - allowed_length)` of logit once the
    /// repeat is `allowed_length` long. 0 disables. Tokens marked in
    /// `dry_breakers` (indexed by id) end any sequence.
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: usize,
    pub dry_breakers: std::sync::Arc<Vec<bool>>,
    /// Mirostat 1 or 2 (0 disables): adaptive truncation that holds the text's
    /// surprise near `tau`, learning at `eta`.
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
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
            ..Self::off()
        }
    }
}

impl SamplingParams {
    /// The samplers llama.cpp adds on top of the classic chain, all off, at
    /// llama.cpp's default settings.
    fn off() -> Self {
        Self {
            temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, typical_p: 1.0, top_n_sigma: 0.0,
            repeat_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0, repeat_last_n: 0,
            logit_bias: Vec::new(), seed: 0,
            dynatemp_range: 0.0, dynatemp_exponent: 1.0,
            xtc_probability: 0.0, xtc_threshold: 0.1,
            dry_multiplier: 0.0, dry_base: 1.75, dry_allowed_length: 2, dry_penalty_last_n: 0,
            dry_breakers: std::sync::Arc::new(Vec::new()),
            mirostat: 0, mirostat_tau: 5.0, mirostat_eta: 0.1,
        }
    }

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
            ..Self::off()
        }
    }
}

/// One sampled token's log-probability and the likeliest alternatives, from
/// the model's raw distribution (before any sampler), as llama.cpp reports
/// them for `n_probs` and `logprobs`.
#[derive(Debug, Clone)]
pub struct TokenProbs {
    pub token: u32,
    pub logprob: f32,
    /// Best first.
    pub top: Vec<(u32, f32)>,
}

/// Where a sampler records [`TokenProbs`], one per sampled token, for the
/// caller to read while generation runs.
pub type ProbsSink = std::sync::Arc<std::sync::Mutex<Vec<TokenProbs>>>;

/// A limit on a reply's reasoning, as llama.cpp's reasoning budget: once the
/// block has used `budget` tokens, or once `stop` is raised from outside (the
/// server's `/control`), the sampler stops sampling and writes `forced` (an
/// optional message, then the end tag) so the model moves on to its answer.
#[derive(Clone)]
pub struct ReasoningBudget {
    /// Token sequences that open and close a reasoning block.
    pub start: Vec<u32>,
    pub end: Vec<u32>,
    /// What closes the block when it is cut short.
    pub forced: Vec<u32>,
    /// Reasoning tokens allowed; None: unlimited.
    pub budget: Option<usize>,
    /// Raised to end the reasoning at the next token.
    pub stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The reply starts inside a block (the prompt opened it).
    pub opened: bool,
}

/// Where a [`ReasoningBudget`] stands.
struct BudgetState {
    rules: ReasoningBudget,
    inside: bool,
    used: usize,
    /// Next position in `forced`, while forcing.
    forcing: Option<usize>,
    /// The tokens this sampler produced, for matching the tags.
    recent: Vec<u32>,
}

impl BudgetState {
    /// The token to emit regardless of the logits, if the block is being closed.
    fn forced(&mut self) -> Option<u32> {
        if self.forcing.is_none() && self.inside {
            let out_of_budget = self.rules.budget.is_some_and(|b| self.used >= b);
            if out_of_budget || self.rules.stop.load(std::sync::atomic::Ordering::Relaxed) {
                self.forcing = Some(0);
            }
        }
        let i = self.forcing?;
        let t = *self.rules.forced.get(i)?;
        self.forcing = Some(i + 1);
        Some(t)
    }

    fn accept(&mut self, token: u32) {
        self.recent.push(token);
        if self.forcing.is_some_and(|i| i >= self.rules.forced.len()) {
            self.forcing = None;
            self.inside = false;
            return;
        }
        if self.forcing.is_some() {
            return;
        }
        if self.inside {
            self.used += 1;
            if !self.rules.end.is_empty() && self.recent.ends_with(&self.rules.end) {
                self.inside = false;
            }
        } else if !self.rules.start.is_empty() && self.recent.ends_with(&self.rules.start) {
            self.inside = true;
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    /// Only tokens this grammar allows are sampled.
    grammar: Option<crate::grammar::Grammar>,
    /// How many alternatives to record per token, and where.
    n_probs: usize,
    probs: Option<ProbsSink>,
    budget: Option<BudgetState>,
    rng: Rng,
    /// Scratch reused between calls to keep sampling allocation-free.
    candidates: Vec<(u32, f32)>,
    /// Occurrence counts over the penalty window, indexed by token id. Sized
    /// lazily on first use so a sampler with penalties off costs nothing.
    counts: Vec<u32>,
    /// Mirostat's running target surprise.
    mu: f32,
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
            grammar: None,
            n_probs: 0,
            probs: None,
            budget: None,
            candidates: Vec::new(),
            counts: Vec::new(),
            mu: 2.0 * 5.0,
        }
        .with_mu()
    }

    fn with_mu(mut self) -> Self {
        self.mu = 2.0 * self.params.mirostat_tau;
        self
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// Constrains sampling to `grammar`.
    pub fn with_grammar(mut self, grammar: crate::grammar::Grammar) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Records each sampled token's probabilities and its `n` likeliest
    /// alternatives into `sink`.
    pub fn with_probs(mut self, n: usize, sink: ProbsSink) -> Self {
        self.n_probs = n;
        self.probs = Some(sink);
        self
    }

    /// Limits the reply's reasoning (see [`ReasoningBudget`]).
    pub fn with_reasoning_budget(mut self, rules: ReasoningBudget) -> Self {
        let inside = rules.opened;
        self.budget = Some(BudgetState { rules, inside, used: 0, forcing: None, recent: Vec::new() });
        self
    }

    /// True when [`sample`] reduces to the argmax of the raw logits: no
    /// temperature, no biases, no penalty that could move the maximum. The
    /// runtime then computes the argmax on the GPU instead of copying logits.
    pub fn is_pure_greedy(&self) -> bool {
        let p = &self.params;
        let penalizing = (p.repeat_penalty != 1.0 || p.presence_penalty != 0.0
            || p.frequency_penalty != 0.0) && p.repeat_last_n > 0;
        p.temperature <= 0.0 && p.logit_bias.is_empty() && !penalizing && p.dry_multiplier <= 0.0
            && self.grammar.is_none() && self.probs.is_none() && self.budget.is_none()
    }

    /// Picks the next token. `logits` is modified in place.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        let raw = self.probs.is_some().then(|| logits.to_vec());
        let forced = self.budget.as_mut().and_then(BudgetState::forced);
        if forced.is_none() {
            if let Some(g) = &self.grammar {
                g.mask(logits);
            }
        }
        let token = match forced {
            Some(t) => t,
            None => self.pick(logits, history),
        };
        if let Some(b) = &mut self.budget {
            b.accept(token);
        }
        if let Some(g) = &mut self.grammar {
            g.accept(token);
        }
        if let (Some(raw), Some(sink)) = (raw, &self.probs) {
            sink.lock().unwrap().push(token_probs(&raw, token, self.n_probs));
        }
        token
    }

    fn pick(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
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

        if self.params.dry_multiplier > 0.0 {
            apply_dry(logits, history, &self.params);
        }

        if self.params.top_n_sigma > 0.0 {
            apply_top_n_sigma(logits, self.params.top_n_sigma);
        }

        if self.params.temperature <= 0.0 {
            return argmax(logits);
        }

        if self.params.mirostat != 0 {
            return self.sample_mirostat(logits);
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

        // Softmax over the surviving candidates, at temperature -- or at the
        // dynamic temperature their spread calls for.
        let max = self.candidates[0].1;
        let inv_t = 1.0 / self.dynamic_temperature(max);
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

        if self.params.xtc_probability > 0.0 && self.rng.next_f32() < self.params.xtc_probability {
            // Candidates are in descending probability: drop the leading ones
            // above the threshold but the last of them.
            let above = self.candidates.iter().take_while(|c| c.1 >= self.params.xtc_threshold).count();
            if above >= 2 {
                self.candidates.drain(..above - 1);
            }
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

/// `token`'s log-probability under softmax(`logits`) and the `n` best.
fn token_probs(logits: &[f32], token: u32, n: usize) -> TokenProbs {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|&l| ((l - max) as f64).exp()).sum::<f64>().ln() as f32;
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    let n = n.min(ids.len());
    let mut top = Vec::new();
    if n > 0 {
        ids.select_nth_unstable_by(n - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        ids.truncate(n);
        ids.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        top = ids.iter().map(|&i| (i, logits[i as usize] - lse)).collect();
    }
    let logprob = logits.get(token as usize).map_or(f32::NEG_INFINITY, |&l| l - lse);
    TokenProbs { token, logprob, top }
}

impl Sampler {
    /// The temperature for this step: the configured one, or with dynamic
    /// temperature the one between `t - range` and `t + range` that the
    /// candidates' normalized entropy (at temperature 1) calls for.
    fn dynamic_temperature(&self, max: f32) -> f32 {
        let (t, r) = (self.params.temperature, self.params.dynatemp_range);
        if r <= 0.0 || self.candidates.len() < 2 {
            return t;
        }
        let ps: Vec<f64> = self.candidates.iter().map(|c| ((c.1 - max) as f64).exp()).collect();
        let sum: f64 = ps.iter().sum();
        let entropy: f64 = -ps.iter().map(|p| p / sum).filter(|&p| p > 0.0).map(|p| p * p.ln()).sum::<f64>();
        let norm = (entropy / (self.candidates.len() as f64).ln()).clamp(0.0, 1.0);
        let (lo, hi) = ((t - r).max(0.0), t + r);
        (lo + (hi - lo) * norm.powf(self.params.dynatemp_exponent as f64) as f32).max(1e-6)
    }

    /// Mirostat 1 or 2 over the whole vocabulary at temperature, updating mu
    /// from the surprise of the token it picks.
    fn sample_mirostat(&mut self, logits: &[f32]) -> u32 {
        let inv_t = 1.0 / self.params.temperature;
        self.candidates.clear();
        self.candidates.extend(logits.iter().copied().enumerate().map(|(i, l)| (i as u32, l * inv_t)));
        let max = self.candidates.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f64;
        for c in &mut self.candidates {
            c.1 = ((c.1 - max) as f64).exp() as f32;
            sum += c.1 as f64;
        }
        for c in &mut self.candidates {
            c.1 = (c.1 as f64 / sum) as f32;
        }
        // Only the head of the distribution is ever kept, so it alone gets
        // sorted: sorting the whole vocabulary every token cost a fifth of
        // the decode speed.
        let by_prob = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
        let head = |c: &mut Vec<(u32, f32)>, k: usize| {
            let k = k.clamp(1, c.len());
            if k < c.len() {
                c.select_nth_unstable_by(k - 1, by_prob);
                c.truncate(k);
            }
            c.sort_unstable_by(by_prob);
        };
        let (tau, eta) = (self.params.mirostat_tau, self.params.mirostat_eta);
        if self.params.mirostat == 1 {
            // Estimate the Zipf exponent from the top 100 and pick k from mu.
            let n = self.candidates.len() as f64;
            let mut all = std::mem::take(&mut self.candidates);
            let mut top = all.clone();
            head(&mut top, 101);
            self.candidates = top;
            let m = 100.min(self.candidates.len() - 1);
            let (mut num, mut den) = (0f64, 0f64);
            for i in 0..m {
                let t_i = ((i + 2) as f64 / (i + 1) as f64).ln();
                let b_i = (self.candidates[i].1 as f64 / self.candidates[i + 1].1.max(f32::MIN_POSITIVE) as f64).ln();
                num += t_i * b_i;
                den += t_i * t_i;
            }
            let s_hat = if den > 0.0 { num / den } else { 1.0 };
            let eps = s_hat - 1.0;
            let k = ((eps * 2f64.powf(self.mu as f64)) / (1.0 - n.powf(-eps))).powf(1.0 / s_hat);
            let k = if k.is_finite() { k as usize } else { all.len() };
            head(&mut all, k);
            self.candidates = all;
        } else {
            // Surprise -log2(p) at most mu: p at least 2^-mu.
            let floor = 2f32.powf(-self.mu);
            let best = self.candidates.iter().copied().max_by(by_prob).expect("non-empty vocabulary");
            self.candidates.retain(|c| c.1 >= floor);
            if self.candidates.is_empty() {
                self.candidates.push(best);
            }
            self.candidates.sort_unstable_by(by_prob);
        }
        let total: f32 = self.candidates.iter().map(|c| c.1).sum();
        let mut target = self.rng.next_f32() * total;
        let mut pick = self.candidates[self.candidates.len() - 1];
        for c in &self.candidates {
            target -= c.1;
            if target <= 0.0 {
                pick = *c;
                break;
            }
        }
        let surprise = -(pick.1 / total).max(f32::MIN_POSITIVE).log2();
        self.mu -= eta * (surprise - tau);
        pick.0
    }
}

/// DRY: for every earlier position whose preceding tokens repeat the end of
/// the history, the token that followed it would extend the repeat; it loses
/// logit by how long that repeat already is. A breaker token stops the match.
fn apply_dry(logits: &mut [f32], history: &[u32], p: &SamplingParams) {
    let n = history.len();
    if n < 2 {
        return;
    }
    let window = if p.dry_penalty_last_n == 0 { n } else { p.dry_penalty_last_n.min(n) };
    let start = n - window;
    let breaks = |t: u32| p.dry_breakers.get(t as usize).copied().unwrap_or(false);
    let last = history[n - 1];
    if breaks(last) {
        return;
    }
    let mut penalty_len: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for j in start..n - 1 {
        if history[j] != last {
            continue;
        }
        // Length of the repeat ending at j against the one ending at n - 1.
        let mut len = 0;
        while len <= j - start && history[j - len] == history[n - 1 - len] && !breaks(history[j - len]) {
            len += 1;
            if j < len + start {
                break;
            }
        }
        let next = history[j + 1];
        if len >= p.dry_allowed_length && !breaks(next) {
            let e = penalty_len.entry(next).or_insert(0);
            *e = (*e).max(len);
        }
    }
    for (tok, len) in penalty_len {
        if let Some(l) = logits.get_mut(tok as usize) {
            *l -= p.dry_multiplier * p.dry_base.powf((len - p.dry_allowed_length) as f32);
        }
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
    fn dry_penalizes_the_token_that_would_extend_a_repeat() {
        // "1 2 3 ... 1 2" : 3 would repeat "1 2 3"; 4 would not.
        let history = [1u32, 2, 3, 9, 1, 2];
        let params = SamplingParams { dry_multiplier: 1.0, dry_allowed_length: 2, ..SamplingParams::greedy() };
        let mut logits = vec![0.0f32; 10];
        apply_dry(&mut logits, &history, &params);
        assert!(logits[3] < 0.0);
        assert_eq!(logits[4], 0.0);
        // A breaker inside the repeat ends it.
        let mut breakers = vec![false; 10];
        breakers[2] = true;
        let params = SamplingParams { dry_breakers: std::sync::Arc::new(breakers), ..params };
        let mut logits = vec![0.0f32; 10];
        apply_dry(&mut logits, &history, &params);
        assert_eq!(logits[3], 0.0);
    }

    fn budget(budget: Option<usize>, stop: bool) -> ReasoningBudget {
        ReasoningBudget {
            start: vec![90], end: vec![91], forced: vec![7, 91], budget,
            stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(stop)), opened: true,
        }
    }

    /// Greedy picks token 1 forever; the budget cuts the reasoning off with
    /// the forced message and end tag, then sampling resumes.
    #[test]
    fn a_reasoning_budget_forces_the_end_tag() {
        let mut s = Sampler::new(SamplingParams::greedy()).with_reasoning_budget(budget(Some(3), false));
        assert!(!s.is_pure_greedy());
        let mut logits = vec![0.0f32; 100];
        logits[1] = 5.0;
        let out: Vec<u32> = (0..7).map(|_| s.sample(&mut logits.clone(), &[])).collect();
        assert_eq!(out, vec![1, 1, 1, 7, 91, 1, 1]);
    }

    #[test]
    fn a_raised_stop_ends_the_reasoning_at_once() {
        let rules = budget(None, false);
        let stop = std::sync::Arc::clone(&rules.stop);
        let mut s = Sampler::new(SamplingParams::greedy()).with_reasoning_budget(rules);
        let mut logits = vec![0.0f32; 100];
        logits[1] = 5.0;
        assert_eq!(s.sample(&mut logits.clone(), &[]), 1);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let out: Vec<u32> = (0..3).map(|_| s.sample(&mut logits.clone(), &[])).collect();
        assert_eq!(out, vec![7, 91, 1]);
        // Outside the block, a stop that stays raised changes nothing.
        assert_eq!(s.sample(&mut logits.clone(), &[]), 1);
    }

    #[test]
    fn mirostat_samples_from_the_head_and_adapts() {
        for version in [1u8, 2] {
            let params = SamplingParams { temperature: 1.0, mirostat: version, seed: 9, ..SamplingParams::greedy() };
            let mut s = Sampler::new(params);
            let mu0 = s.mu;
            let base: Vec<f32> = (0..1000).map(|i| -(i as f32) * 0.05).collect();
            for _ in 0..20 {
                let mut logits = base.clone();
                let t = s.sample(&mut logits, &[]);
                assert!(t < 1000);
            }
            assert_ne!(s.mu, mu0, "mirostat {version} must learn");
        }
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
