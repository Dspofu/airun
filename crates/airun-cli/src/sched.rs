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

        // Admit one waiting job per step, so a long prompt delays the others'
        // tokens by one prefill at a time.
        if active.len() < slots {
            if let Some(job) = waiting.pop_front() {
                if job.cancel.load(Ordering::Relaxed) {
                    let _ = job.events.send(Event::Done(Err("cancelled".into())));
                } else {
                    let mut rt = runtime.blocking_lock();
                    match admit(&mut rt, job, &active, slots) {
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
        // One token for every running request.
        let rows: Vec<(usize, u32)> = active.iter().map(|r| (r.slot, r.next)).collect();
        let mut rt = runtime.blocking_lock();
        let logits = match rt.decode_rows(&rows) {
            Ok(l) => l,
            Err(e) => {
                for r in active.drain(..) {
                    let _ = r.job.events.send(Event::Done(Err(e.to_string())));
                }
                continue;
            }
        };
        let vocab = rt.model.config.vocab_size;
        let ctx = rt.ctx();
        let mut still = Vec::with_capacity(active.len());
        for (b, mut r) in active.drain(..).enumerate() {
            let row = &mut logits[b * vocab..(b + 1) * vocab].to_vec();
            let tok = r.job.sampler.sample(row, &r.history);
            let full = rt.slot_history(r.slot).len() >= ctx;
            match emit(&rt, &mut r, tok, full) {
                Some(stop) => finish(&rt, r, stop),
                None => still.push(r),
            }
        }
        active = still;
    }
}

/// Prefills a job into the best free slot and emits its first token.
fn admit(rt: &mut Runtime, job: Job, active: &[Running], slots: usize)
    -> Result<Option<Running>, (Job, String)> {
    let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
    let free = (0..slots).filter(|s| !busy.contains(s));
    let slot = free.max_by_key(|&s| (rt.slot_prefix(s, &job.tokens), usize::MAX - s)).expect("a slot is free");
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
    let history: Vec<u32> = job.tokens.iter().copied().filter(|&t| !airun_core::runtime::is_image_token(t)).collect();
    let mut r = Running {
        slot, history, decoder: rt.tokenizer.stream_decoder(), next: 0, last: 0, generated: 0, cached,
        prefill_ms, decode_start: Instant::now(), job,
    };
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
        drafted: 0,
        accepted: 0,
    };
    let _ = r.job.events.send(Event::Done(Ok(stats)));
}
