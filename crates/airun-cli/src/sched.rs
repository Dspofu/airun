//! Several requests at once (serve --parallel N), as llama-server's slots.
//!
//! One thread owns the turns on the model. A request takes a free slot --
//! the one whose cache already holds the longest prefix of its prompt, so a
//! chat's next turn finds its own history -- and is prefilled there; then
//! every active request advances one token per step through
//! Runtime::decode_rows, which reads each weight once for all of them. The
//! requesting thread gets its tokens over a channel and does everything else
//! (stop sequences, streaming, the response) as it did with one request at a
//! time. The runtime lock is taken per step, so tokenize and the other small
//! endpoints keep working between steps.
//!
//! A request with nobody else running or waiting runs alone through
//! Runtime::generate on its slot, as without --parallel: the MTP head drafts,
//! greedy picks stay on the GPU, the decode graphs and the turn snapshot work.
//! The batched path reads every logit back and samples on the host, so a lone
//! request there lost up to 29% (27B with --draft 4: 38.6 -> 27.4 tok/s).
//! When another request arrives the lone one pauses after the token it is
//! handing out, and both go on in batched steps.
//!
//! With drafting on, a batched step speculates too (spec_step): every
//! request's drafts are verified in one pass and each takes 1 + accepted
//! tokens, with fewer drafts the more requests there are (Runtime::rows_draft).
//! 27B NVFP4, 5070 Ti + 5060, --draft 3: 2 requests 59 -> 81 tok/s summed,
//! 4 requests 106 -> 117.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use airun_core::runtime::{GenerationStats, Runtime, StopReason};
use airun_core::Sampler;
use tokio::sync::Mutex;

pub enum Event {
    /// A token and the text it completes (may be empty mid-character).
    Token(u32, String),
    Done(Result<GenerationStats, String>),
}

pub struct Job {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    pub sampler: Sampler,
    pub events: Sender<Event>,
    /// Raised by the requester to end the generation (stop sequence, client gone).
    pub cancel: Arc<AtomicBool>,
    /// Told after each batch of this job's prefill (`return_progress`).
    pub progress: Option<airun_core::runtime::PrefillHook>,
}

struct Running {
    slot: usize,
    job: Job,
    /// Prompt and generated tokens, for the sampler's penalties.
    history: Vec<u32>,
    decoder: airun_core::tokenizer::StreamDecoder,
    next: u32,
    last: u32,
    generated: usize,
    cached: usize,
    prefill_ms: f64,
    decode_start: Instant,
    drafted: usize,
    accepted: usize,
    /// The MTP head's drafts for the token after `next`, for the next
    /// batched speculative step (empty: none yet).
    drafts: Vec<u32>,
}

/// Starts the scheduler thread; the returned sender takes jobs.
pub fn spawn(runtime: Arc<Mutex<Runtime>>) -> Sender<Job> {
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    std::thread::Builder::new()
        .name("airun-scheduler".into())
        .spawn(move || run(runtime, rx))
        .expect("the scheduler thread starts");
    tx
}

fn run(runtime: Arc<Mutex<Runtime>>, rx: Receiver<Job>) {
    let slots = runtime.blocking_lock().n_slots();
    let mut active: Vec<Running> = Vec::new();
    let mut waiting: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    // When each slot last took a request (0: never), for admit's tie-break.
    let mut used = vec![0u64; slots];
    let mut last_step = Instant::now();
    let started = Instant::now();
    loop {
        // Take what has arrived; with nothing running, wait for a job.
        loop {
            match rx.try_recv() {
                Ok(j) => waiting.push_back(j),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if active.is_empty() && waiting.is_empty() {
                        return;
                    }
                    break;
                }
            }
        }
        if active.is_empty() && waiting.is_empty() {
            match rx.recv() {
                Ok(j) => waiting.push_back(j),
                Err(_) => return,
            }
        }

        // Alone: run it as without --parallel until someone else arrives.
        if active.len() + waiting.len() == 1 && !solo_disabled() {
            let r = match waiting.pop_front() {
                Some(job) if job.cancel.load(Ordering::Relaxed) => {
                    let _ = job.events.send(Event::Done(Err("cancelled".into())));
                    continue;
                }
                Some(job) => {
                    let rt = runtime.blocking_lock();
                    let slot = pick_slot(&rt, &job.tokens, &active, &mut used);
                    fresh(&rt, slot, job, 0, 0.0)
                }
                None => active.pop().expect("one is running"),
            };
            let mut rt = runtime.blocking_lock();
            if let Some(r) = solo(&mut rt, r, &rx, &mut waiting) {
                active.push(r);
            }
            continue;
        }

        // Admit one waiting job per step, so a long prompt delays the others'
        // tokens by one prefill at a time.
        if active.len() < slots {
            if let Some(job) = waiting.pop_front() {
                if job.cancel.load(Ordering::Relaxed) {
                    let _ = job.events.send(Event::Done(Err("cancelled".into())));
                } else {
                    let mut rt = runtime.blocking_lock();
                    match admit(&mut rt, job, &active, &mut used) {
                        Ok(Some(r)) => active.push(r),
                        Ok(None) => {}
                        Err((job, e)) => {
                            let _ = job.events.send(Event::Done(Err(e)));
                        }
                    }
                }
                continue;
            }
        }

        if active.is_empty() {
            continue;
        }
        // With drafting on, every running request verifies its drafts in one
        // pass: each gets 1 + accepted tokens a step instead of 1.
        {
            let mut rt = runtime.blocking_lock();
            let k = rt.rows_draft(active.len());
            let room = active.iter().all(|r| rt.slot_history(r.slot).len() + 2 * (rt.draft() + 1) <= rt.ctx());
            if room && k > 0 {
                let t0 = Instant::now();
                if let Err(e) = spec_step(&mut rt, &mut active, k) {
                    for r in active.drain(..) {
                        let _ = r.job.events.send(Event::Done(Err(e.to_string())));
                    }
                }
                if step_timing() {
                    eprintln!("sched spec step: {} rows, {:.2} ms", active.len(), t0.elapsed().as_secs_f64() * 1e3);
                }
                last_step = Instant::now();
                continue;
            }
        }
        // One token for every running request.
        for r in active.iter_mut() {
            r.drafts.clear();
        }
        let rows: Vec<(usize, u32)> = active.iter().map(|r| (r.slot, r.next)).collect();
        let mut rt = runtime.blocking_lock();
        let t0 = Instant::now();
        let vocab = rt.model.config.vocab_size;
        let ctx = rt.ctx();
        let (toks, forward_ms) = match rt.decode_rows(&rows) {
            Ok(logits) => {
                let f = t0.elapsed().as_secs_f64() * 1e3;
                (sample_rows(&mut active, logits, vocab), f)
            }
            Err(e) => {
                for r in active.drain(..) {
                    let _ = r.job.events.send(Event::Done(Err(e.to_string())));
                }
                continue;
            }
        };
        let t2 = Instant::now();
        let mut still = Vec::with_capacity(active.len());
        for (mut r, tok) in active.drain(..).zip(toks) {
            let full = rt.slot_history(r.slot).len() >= ctx;
            match emit(&rt, &mut r, tok, full) {
                Some(stop) => finish(&rt, r, stop),
                None => still.push(r),
            }
        }
        if step_timing() {
            eprintln!("sched step: {} rows, forward {:.2} ms, sample {:.2} ms, emit {:.2} ms, between steps {:.2} ms, at {:.3} s",
                rows.len(), forward_ms, (t2 - t0).as_secs_f64() * 1e3 - forward_ms, t2.elapsed().as_secs_f64() * 1e3,
                (t0 - last_step).as_secs_f64() * 1e3, t0.duration_since(started).as_secs_f64());
        }
        last_step = Instant::now();
        active = still;
    }
}

/// One batched speculative step: every request's newest token and drafts
/// through one verify, then each takes the drafts the main model agrees with
/// (or, sampling, those sample_draft keeps) and the model's own next token,
/// exactly as a lone generate does. A request whose head cannot draft
/// (prefilled without it) rides along with filler drafts: still exact, it
/// just takes one token.
fn spec_step(rt: &mut Runtime, active: &mut Vec<Running>, k: usize) -> anyhow::Result<()> {
    // Drafts from a step with fewer rows are longer: their first k still hold.
    for r in active.iter_mut() {
        r.drafts.truncate(k);
    }
    let need: Vec<(usize, u32)> = active.iter()
        .filter(|r| r.drafts.len() != k && rt.head_synced(r.slot)).map(|r| (r.slot, r.next)).collect();
    if !need.is_empty() {
        let drafted = rt.draft_rows(&need, k)?;
        for ((slot, _), d) in need.iter().zip(drafted) {
            if let Some(r) = active.iter_mut().find(|r| r.slot == *slot) {
                r.drafts = d;
            }
        }
    }
    for r in active.iter_mut() {
        if r.drafts.len() != k {
            r.drafts = vec![r.next; k];
        }
    }
    let rows: Vec<(usize, Vec<u32>)> = active.iter().map(|r| {
        let mut v = vec![r.next];
        v.extend_from_slice(&r.drafts);
        (r.slot, v)
    }).collect();
    let sampling = active.iter().any(|r| !r.job.sampler.is_pure_greedy());
    let vocab = rt.model.config.vocab_size;
    let tv = Instant::now();
    let verified = rt.verify_rows(&rows, sampling)?;
    let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
    let t = verified.t;
    let picks = verified.picks;
    // Per request: the tokens it takes, in order (accepted drafts, then the
    // model's next). Sampled requests on threads of their own, as sample_rows.
    let mut logits = verified.logits;
    let chosen: Vec<(usize, u32)> = {
        let rows_logits: Vec<Option<&mut [f32]>> = match logits.as_deref_mut() {
            Some(l) => l.chunks_mut(t * vocab).map(Some).collect(),
            None => (0..active.len()).map(|_| None).collect(),
        };
        std::thread::scope(|sc| {
            let handles: Vec<_> = active.iter_mut().zip(rows_logits).enumerate().map(|(b, (r, lg))| {
                let picks = &picks[b * t..(b + 1) * t];
                let drafts = &r.drafts;
                let sampler = &mut r.job.sampler;
                let hist = &r.history;
                sc.spawn(move || -> (usize, u32) {
                    match lg {
                        Some(lg) if !sampler.is_pure_greedy() => {
                            let mut h = hist.clone();
                            let mut a = 0;
                            while a < drafts.len() {
                                let row = &mut lg[a * vocab..(a + 1) * vocab];
                                let (tok, took) = sampler.sample_draft(row, &h, drafts[a]);
                                if !took {
                                    return (a, tok);
                                }
                                h.push(tok);
                                a += 1;
                            }
                            let row = &mut lg[a * vocab..(a + 1) * vocab];
                            (a, sampler.sample(row, &h))
                        }
                        _ => {
                            let mut a = 0;
                            while a < drafts.len() && drafts[a] == picks[a] {
                                a += 1;
                            }
                            (a, picks[a])
                        }
                    }
                })
            }).collect();
            handles.into_iter().map(|h| h.join().expect("a sampler thread does not panic")).collect()
        })
    };
    // Hand the tokens out; a request stops wherever a lone one would.
    let ctx = rt.ctx();
    let mut accept = Vec::with_capacity(active.len());
    let mut ended = Vec::with_capacity(active.len());
    for (r, &(a, next)) in active.iter_mut().zip(&chosen) {
        let real = r.drafts != vec![r.next; k] || rt.head_synced(r.slot);
        if real {
            r.drafted += k;
            r.accepted += a;
        }
        let base = rt.slot_history(r.slot).len();
        let mut committed = 1;
        let mut stop = None;
        let toks: Vec<u32> = r.drafts[..a].to_vec();
        for &tok in &toks {
            if let Some(s) = emit(rt, r, tok, base + committed >= ctx) {
                stop = Some(s);
                break;
            }
            committed += 1;
        }
        if stop.is_none() {
            stop = emit(rt, r, next, base + committed >= ctx);
        }
        accept.push((committed, next));
        ended.push(stop);
    }
    let ta = Instant::now();
    let drafts = rt.accept_rows(&accept, k)?;
    if step_timing() {
        eprintln!("  verify {verify_ms:.2} ms, accept {:.2} ms, accepted {:?}", ta.elapsed().as_secs_f64() * 1e3,
            accept.iter().map(|a| a.0).collect::<Vec<_>>());
    }
    let mut still = Vec::with_capacity(active.len());
    for ((mut r, stop), d) in active.drain(..).zip(ended).zip(drafts) {
        match stop {
            Some(stop) => finish(rt, r, stop),
            None => {
                r.drafts = if rt.head_synced(r.slot) { d } else { Vec::new() };
                still.push(r);
            }
        }
    }
    *active = still;
    Ok(())
}

/// AIRUN_SCHED_TIMING=1 prints where each batched step's time went.
fn step_timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AIRUN_SCHED_TIMING").is_ok_and(|v| v.trim() != "0"))
}

/// Each row's next token from its logits (`vocab` floats a row), the rows on
/// threads of their own: sampling a 248k vocabulary takes milliseconds, and
/// one after another the GPU waited for all of them every step.
fn sample_rows(active: &mut [Running], logits: &mut [f32], vocab: usize) -> Vec<u32> {
    if active.len() == 1 {
        let r = &mut active[0];
        return vec![r.job.sampler.sample(&mut logits[..vocab], &r.history)];
    }
    let work = active.iter_mut().zip(logits.chunks_mut(vocab))
        .map(|(r, row)| (&mut r.job.sampler, &r.history, row));
    std::thread::scope(|sc| {
        let handles: Vec<_> = work.map(|(s, h, row)| sc.spawn(move || s.sample(row, h))).collect();
        handles.into_iter().map(|h| h.join().expect("a sampler thread does not panic")).collect()
    })
}

/// Cached tokens a lone request may prefill again to get the MTP head back
/// (its cache missed them): a few hundred cost less than drafting gains over
/// a reply.
const REDO_FOR_DRAFTS: usize = 256;

/// AIRUN_NO_PARALLEL_SOLO=1 sends a lone request through the batched path too.
fn solo_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("AIRUN_NO_PARALLEL_SOLO").is_ok_and(|v| v.trim() != "0"))
}

/// The free slot a job goes to: the most reusable prefix first; on a tie
/// (often none at all) the least recently used, so an unrelated request does
/// not evict the conversation that will come back next while an idle slot
/// sits there.
fn pick_slot(rt: &Runtime, tokens: &[u32], active: &[Running], used: &mut [u64]) -> usize {
    let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
    let free = (0..used.len()).filter(|s| !busy.contains(s));
    let slot = free.max_by_key(|&s| (rt.slot_prefix(s, tokens), std::cmp::Reverse(used[s]),
        usize::MAX - s)).expect("a slot is free");
    used[slot] = used.iter().max().copied().unwrap_or(0) + 1;
    slot
}

/// A job's state before its first token.
fn fresh(rt: &Runtime, slot: usize, job: Job, cached: usize, prefill_ms: f64) -> Running {
    let history: Vec<u32> = job.tokens.iter().copied().filter(|&t| !airun_core::runtime::is_image_token(t)).collect();
    Running {
        slot, history, decoder: rt.tokenizer.stream_decoder(), next: 0, last: 0, generated: 0, cached,
        prefill_ms, decode_start: Instant::now(), drafted: 0, accepted: 0, drafts: Vec::new(), job,
    }
}

/// Runs `r` alone through Runtime::generate on its slot: from its prompt if
/// it has no token yet, else from the slot's history and the token it handed
/// out last. Returns it, paused, when another job arrives (now in `waiting`);
/// None once it has finished.
fn solo(rt: &mut Runtime, mut r: Running, rx: &Receiver<Job>,
        waiting: &mut std::collections::VecDeque<Job>) -> Option<Running> {
    let begun = r.generated == 0;
    if let Err(e) = rt.switch_slot(r.slot) {
        let _ = r.job.events.send(Event::Done(Err(e.to_string())));
        return None;
    }
    let prompt = if begun {
        r.job.tokens.clone()
    } else {
        let mut p = rt.slot_history(r.slot).to_vec();
        p.push(r.next);
        p
    };
    if begun {
        rt.set_prefill_hook(r.job.progress.take());
    }
    // A head that missed part of what the slot holds (another slot ran, or
    // this one decoded in batched steps) would make generate prefill the whole
    // prompt again: go on without drafts rather than redo a conversation,
    // unless what it would redo is short.
    let draft = rt.draft();
    if draft > 0 && !rt.spec_keeps_prefix(&prompt) && rt.reusable_prefix(&prompt) > REDO_FOR_DRAFTS {
        rt.set_draft(0);
    }
    let left = r.job.max_tokens.saturating_sub(r.generated);
    let (mut paused, mut gone) = (false, false);
    let result = {
        let Running { job, history, generated, last, next, decoder, .. } = &mut r;
        let Job { sampler, events, cancel, max_tokens, .. } = job;
        let max_tokens = *max_tokens;
        let mut on_token = |tok: u32, text: &str| -> bool {
            *generated += 1;
            history.push(tok);
            *last = tok;
            *next = tok;
            if events.send(Event::Token(tok, text.to_string())).is_err() || cancel.load(Ordering::Relaxed) {
                gone = true;
                return false;
            }
            if *generated < max_tokens {
                while let Ok(j) = rx.try_recv() {
                    waiting.push_back(j);
                    paused = true;
                }
            }
            !paused
        };
        rt.generate_with(&prompt, left, sampler, decoder, false, &mut on_token)
    };
    rt.set_draft(draft);
    rt.set_prefill_hook(None);
    let stats = match result {
        Ok(s) => s,
        Err(e) => {
            let _ = r.job.events.send(Event::Done(Err(e.to_string())));
            return None;
        }
    };
    if begun {
        r.cached = stats.cached_tokens;
        r.prefill_ms = stats.prefill_ms;
        r.decode_start = Instant::now() - std::time::Duration::from_secs_f64(stats.decode_ms / 1000.0);
    }
    r.drafted += stats.drafted;
    r.accepted += stats.accepted;
    if gone {
        finish(rt, r, StopReason::Cancelled);
        return None;
    }
    if paused && stats.stop == StopReason::Cancelled {
        if rt.slot_history(r.slot).len() >= rt.ctx() {
            finish(rt, r, StopReason::ContextFull);
            return None;
        }
        return Some(r);
    }
    finish(rt, r, stats.stop);
    None
}

/// Prefills a job into the best free slot and emits its first token.
fn admit(rt: &mut Runtime, job: Job, active: &[Running], used: &mut [u64])
    -> Result<Option<Running>, (Job, String)> {
    let slot = pick_slot(rt, &job.tokens, active, used);
    let start = Instant::now();
    let mut job = job;
    rt.set_prefill_hook(job.progress.take());
    let begun = rt.slot_begin(slot, &job.tokens);
    rt.set_prefill_hook(None);
    let (mut logits, cached) = match begun {
        Ok(v) => v,
        Err(e) => return Err((job, e.to_string())),
    };
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut r = fresh(rt, slot, job, cached, prefill_ms);
    let tok = r.job.sampler.sample(&mut logits, &r.history);
    let full = rt.slot_history(slot).len() >= rt.ctx();
    Ok(match emit(rt, &mut r, tok, full) {
        Some(stop) => {
            finish(rt, r, stop);
            None
        }
        None => Some(r),
    })
}

/// Hands `tok` to the requester; Some(reason) when the generation ends here.
fn emit(rt: &Runtime, r: &mut Running, tok: u32, ctx_full: bool) -> Option<StopReason> {
    if r.generated >= r.job.max_tokens {
        return Some(StopReason::MaxTokens);
    }
    if rt.tokenizer.is_eog(tok) {
        return Some(StopReason::EndOfGeneration);
    }
    if r.job.cancel.load(Ordering::Relaxed) {
        return Some(StopReason::Cancelled);
    }
    let text = r.decoder.push(&rt.tokenizer, tok);
    r.generated += 1;
    r.history.push(tok);
    r.last = tok;
    r.next = tok;
    if r.job.events.send(Event::Token(tok, text)).is_err() {
        return Some(StopReason::Cancelled);
    }
    if r.job.cancel.load(Ordering::Relaxed) {
        return Some(StopReason::Cancelled);
    }
    if r.generated >= r.job.max_tokens {
        return Some(StopReason::MaxTokens);
    }
    if ctx_full {
        return Some(StopReason::ContextFull);
    }
    None
}

fn finish(_rt: &Runtime, mut r: Running, stop: StopReason) {
    let tail = r.decoder.flush();
    if !tail.is_empty() {
        let _ = r.job.events.send(Event::Token(r.last, tail));
    }
    let stats = GenerationStats {
        prompt_tokens: r.job.tokens.len(),
        cached_tokens: r.cached,
        generated_tokens: r.generated,
        prefill_ms: r.prefill_ms,
        decode_ms: r.decode_start.elapsed().as_secs_f64() * 1000.0,
        stop,
        drafted: r.drafted,
        accepted: r.accepted,
    };
    let _ = r.job.events.send(Event::Done(Ok(stats)));
}
