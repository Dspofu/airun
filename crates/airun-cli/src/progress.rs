//! The model-loading progress line on stderr.
//!
//! On a terminal it redraws one line:
//!   loading  [##########----------]  52%  33/67 layers  7.1/13.6 GiB  1.9 GiB/s  ~3s left
//! With the model split across GPUs each GPU gets its own line, kept on screen:
//!   GPU 0 RTX 5070 Ti, layers 0..51 [########################]  ok  6.4s
//!   GPU 1 RTX 5060, layers 52..63 [###################-----]  81%  13.3/16.4 GiB  1.9 GiB/s  ~2s
//! Elsewhere (a log, a pipe) it prints a line every 25% and one per GPU, so
//! logs stay short.
//! The loader reports steps (one per layer, plus the embedding, output and
//! tokenizer); bytes are the file's size times the fraction done, since the
//! layers are close to equal in size.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::Instant;

pub struct LoadBar {
    total_bytes: u64,
    start: Instant,
    tty: bool,
    last_draw: Option<Instant>,
    last_quarter: usize,
    stage: String,
    drawn: bool,
    last_step: (usize, usize),
    stage_start_secs: f64,
}

/// The loader's per-GPU upload stages start with "-> " (see loader::upload_labels).
fn is_gpu_stage(stage: &str) -> bool {
    stage.starts_with("-> ")
}

impl LoadBar {
    pub fn new(model: &Path) -> Self {
        let total_bytes = shard_bytes(model);
        Self {
            total_bytes,
            start: Instant::now(),
            tty: std::io::stderr().is_terminal(),
            last_draw: None,
            last_quarter: 0,
            stage: String::new(),
            drawn: false,
            last_step: (0, 1),
            stage_start_secs: 0.0,
        }
    }

    /// The loader's callback: `done` of `total` steps of `stage`.
    pub fn update(&mut self, stage: &str, done: usize, total: usize) {
        if total == 0 {
            return;
        }
        let stage_changed = stage != self.stage;
        if stage_changed {
            // A split load names each GPU's share as its own stage: the
            // finished GPU's line is drawn at its last step and stays on
            // screen, and the next GPU starts on the line below.
            if self.tty && self.drawn && is_gpu_stage(&self.stage) {
                let (old, (d, t)) = (self.stage.clone(), self.last_step);
                self.draw(&old, d, t, true);
                eprintln!();
                // The next GPU's share starts here (the first GPU's share
                // includes the embedding and output uploaded before it).
                self.stage_start_secs = self.start.elapsed().as_secs_f64();
            }
            self.stage = stage.to_string();
        }
        self.last_step = (done, total);
        let frac = (done as f64 / total as f64).clamp(0.0, 1.0);
        if !self.tty {
            let quarter = (frac * 4.0).floor() as usize;
            if quarter > self.last_quarter || (stage_changed && (done == 0 || is_gpu_stage(stage))) {
                self.last_quarter = quarter;
                eprintln!("{}: {:.0}%", stage.trim_start_matches("-> "), frac * 100.0);
            }
            return;
        }
        // Redraw at most every 50 ms, and always at the end of a stage.
        let now = Instant::now();
        if done < total && !stage_changed && self.last_draw.is_some_and(|t| now.duration_since(t).as_millis() < 50) {
            return;
        }
        self.last_draw = Some(now);
        self.draw(stage, done, total, false);
    }

    /// `finished`: a GPU whose share is all uploaded, drawn full with what it
    /// took, since the percentage and time left are the whole load's.
    fn draw(&mut self, stage: &str, done: usize, total: usize, finished: bool) {
        let frac = (done as f64 / total as f64).clamp(0.0, 1.0);
        let width = 24;
        let filled = if finished { width } else { (frac * width as f64).round() as usize };
        let bar: String = "#".repeat(filled) + &"-".repeat(width - filled);
        let gib = |b: f64| b / (1u64 << 30) as f64;
        let elapsed = self.start.elapsed().as_secs_f64().max(1e-3);
        let gpu = is_gpu_stage(stage);
        // Per-GPU stages carry the layer range, which replaces the step count;
        // the line stays short enough not to wrap (a wrapped line breaks \r).
        let mut line = if gpu && finished {
            // No byte count: it is the file size times the step fraction, which
            // misplaces the embedding and output (the plan printed after has
            // each GPU's real share).
            let line = format!("\r{} [{bar}]  ok  {:.1}s", stage.trim_start_matches("-> "),
                elapsed - self.stage_start_secs);
            let mut err = std::io::stderr();
            let _ = write!(err, "{line}\x1b[K");
            let _ = err.flush();
            return;
        } else if gpu {
            format!("\r{} [{bar}] {:>3.0}%", stage.trim_start_matches("-> "), frac * 100.0)
        } else {
            format!("\r{stage:<18} [{bar}] {:>3.0}%  {done}/{total}", frac * 100.0)
        };
        if (gpu || stage == "uploading weights") && self.total_bytes > 0 {
            let loaded = self.total_bytes as f64 * frac;
            let rate = loaded / elapsed;
            line += &format!("  {:.1}/{:.1} GiB  {:.1} GiB/s", gib(loaded), gib(self.total_bytes as f64), gib(rate));
            if frac > 0.02 && frac < 1.0 {
                line += &format!("  ~{:.0}s", elapsed * (1.0 - frac) / frac);
            }
        }
        let mut err = std::io::stderr();
        let _ = write!(err, "{line}\x1b[K");
        let _ = err.flush();
        self.drawn = true;
    }

    /// Ends the bar so what follows starts clean and reports the load time.
    /// On a terminal the line replaces the bar; piped it is its own line, so
    /// logs keep the timing too.
    pub fn finish(&mut self) {
        if self.tty && self.drawn {
            eprintln!("\r\x1b[Kloaded in {:.1}s", self.start.elapsed().as_secs_f64());
        } else if !self.tty {
            eprintln!("loaded in {:.1}s", self.start.elapsed().as_secs_f64());
        }
    }
}

/// Bytes of the model's files: the one given, plus its -0000N-of-0000M
/// siblings when it is the first shard of a split GGUF.
fn shard_bytes(model: &Path) -> u64 {
    let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let name = model.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some(pos) = name.find("-00001-of-") else { return size(model) };
    let count: usize = name[pos + 10..].trim_end_matches(".gguf").parse().unwrap_or(1);
    let dir = model.parent().unwrap_or(Path::new("."));
    (1..=count).map(|i| size(&dir.join(name.replacen("-00001-of-", &format!("-{i:05}-of-"), 1)))).sum()
}
