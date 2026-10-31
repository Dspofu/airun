//! End-to-end model loading, with automatic recovery from capacity failures.
//!
//! A first-time user's most common experience is an out-of-memory error with no
//! indication of what to change. The loader instead retries down a ladder of
//! progressively smaller configurations and reports exactly what it settled on,
//! so a model that *can* run does run.

use std::path::Path;

use airun_cuda::{Device, Stream, device};
use airun_gguf::GgufFile;

use crate::config::{Architecture, Config};
use crate::error::{CoreError, Result};
use crate::model::Model;
use crate::plan::{Placement, PlacementRequest, SplitMode, WeightSizes, plan};
use crate::runtime::Runtime;
use crate::tokenizer::Tokenizer;

/// What to load and how.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub ctx: usize,
    pub batch: usize,
    /// GPUs to use. Empty means every supported device.
    pub gpus: Vec<i32>,
    pub split: SplitMode,
    pub gpu_layers: Option<usize>,
    /// Resident slots per MoE tensor; absent keeps all experts on the GPU.
    pub expert_cache: Option<usize>,
    /// KV cache storage; quantized formats trade a little accuracy for
    /// 2x (q8_0) or 3.6x (q4_0) more context per byte.
    pub kv_type: airun_cuda::ops::KvType,
    pub reserve_bytes: u64,
    /// Allow the loader to shrink context/batch to make the model fit.
    pub allow_fallback: bool,
    /// Speculative decoding: tokens the model's MTP head drafts per step. 0
    /// leaves the head unloaded; ignored on files without one.
    pub draft: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            ctx: 4096,
            batch: 512,
            gpus: Vec::new(),
            split: SplitMode::Auto,
            gpu_layers: None,
            expert_cache: None,
            kv_type: Default::default(),
            reserve_bytes: 384 << 20,
            allow_fallback: true,
            draft: 0,
        }
    }
}

/// What the loader did, for reporting.
pub struct LoadReport {
    pub config: Config,
    pub placement: Placement,
    pub weight_bytes: u64,
    pub load_ms: f64,
    /// Adjustments made to get the model to fit.
    pub fallbacks: Vec<String>,
}

/// Opens, plans, uploads and wraps a model into a ready [`Runtime`].
pub fn load(
    path: &Path,
    opts: &LoadOptions,
    mut progress: impl FnMut(&str, usize, usize),
) -> Result<(Runtime, LoadReport)> {
    let start = std::time::Instant::now();

    let gguf = std::sync::Arc::new(GgufFile::open(path)?);
    let mut config = Config::from_gguf(&gguf)?;
    config.kv_type = opts.kv_type;

    if let Architecture::Unsupported(reason) = config.architecture {
        return Err(CoreError::UnsupportedArchitecture {
            arch: config.arch_name.clone(),
            reason: reason.to_string(),
        });
    }

    let devices = select_devices(&opts.gpus)?;
    let mut sizes = WeightSizes::from_gguf(&gguf, config.n_layers);
    if crate::model::embeds_on_host(&config) {
        sizes.embedding = 0;
    }
    // The n-gram embedding table stays in the file (see model::PleTable).
    if let Some(t) = config.ple.as_ref().and_then(|_| gguf.tensor("per_layer_token_embd.weight")) {
        sizes.other = sizes.other.saturating_sub(t.nbytes);
    }
    let load_mtp = opts.draft > 0 && crate::model::mtp_supported(&config);
    if load_mtp {
        // The head and its own KV cache sit with the output projection.
        sizes.output += sizes.per_layer.get(config.n_run_layers()).copied().unwrap_or(0)
            + config.kv_bytes_per_layer(opts.ctx);
    }
    let expert_cache = opts.expert_cache.or_else(|| auto_expert_cache(&gguf, &config, &sizes, &devices, opts));
    if let Some(slots) = expert_cache {
        let moe = config.moe.as_ref().ok_or_else(|| CoreError::Other(
            "--expert-cache requires a MoE model; omit it for dense models".into()))?;
        if slots < moe.n_experts_used || slots > moe.n_experts {
            return Err(CoreError::Other(format!(
                "--expert-cache must be between {} and {}; choose enough slots for every active expert",
                moe.n_experts_used, moe.n_experts)));
        }
        for l in 0..config.n_run_layers() {
            for kind in ["gate", "up", "down"] {
                if let Some(t) = gguf.tensor(&format!("blk.{l}.ffn_{kind}_exps.weight")) {
                    sizes.per_layer[l] -= t.nbytes - t.nbytes / moe.n_experts as u64 * slots as u64;
                }
            }
        }
    }

    // Ladder of retries. Each rung halves the memory the run needs; the first
    // that both plans and allocates wins.
    let mut fallbacks = Vec::new();
    let mut ctx = opts.ctx;
    let mut batch = opts.batch;
    let mut reserve = opts.reserve_bytes;

    loop {
        let req = PlacementRequest {
            ctx,
            batch,
            split: opts.split.clone(),
            gpu_layers: opts.gpu_layers,
            reserve_bytes: reserve,
            allow_ctx_shrink: opts.allow_fallback,
        };

        let mut placement = match plan(&config, &sizes, &devices, &req) {
            Ok(p) => {
                batch = p.batch;
                p
            }
            Err(e) if opts.allow_fallback && e.is_capacity_error() => {
                match next_rung(&mut ctx, &mut batch, &mut reserve) {
                    Some(note) => {
                        fallbacks.push(note);
                        continue;
                    }
                    None => return Err(e),
                }
            }
            Err(e) => return Err(e),
        };

        if let Some(slots) = expert_cache {
            let how = if opts.expert_cache.is_some() { "" } else { " (sized to fill the GPUs; --expert-cache overrides)" };
            placement.notes.push(format!("MoE expert cache: {slots} slots per layer{how}; misses read from the GGUF, weights unchanged"));
        }
        if config.indexer_top_k > 0 && placement.ctx > config.indexer_top_k {
            placement.notes.push(format!(
                "the model's sparse (QSA) attention is not implemented: past {} tokens airun attends to the whole \
                 context, where the reference attends to its best-scored {} -- output may drift on long inputs",
                config.indexer_top_k, config.indexer_top_k));
        }
        if opts.kv_type != airun_cuda::ops::KvType::F16 {
            placement.notes.push(format!("KV cache {}", opts.kv_type.name()));
        }

        if placement.ctx != ctx {
            fallbacks.push(format!(
                "context reduced from {ctx} to {} to fit available VRAM",
                placement.ctx
            ));
            ctx = placement.ctx;
        }

        // Streams have to exist before any allocation, since every buffer is
        // filled through one.
        let mut streams = Vec::new();
        for shard in &placement.shards {
            device::set_device(shard.device)?;
            streams.push((shard.device, Stream::new(shard.device)?));
        }

        let attempt = Model::load(&gguf, config.clone(), placement.clone(), &streams, expert_cache, load_mtp, |d, t| {
            progress("uploading weights", d, t)
        });

        let model = match attempt {
            Ok(m) => m,
            Err(e) if opts.allow_fallback && e.is_capacity_error() => {
                // The plan said it fits but the driver disagreed: fragmentation,
                // or another process took memory in between. Drop everything and
                // try a smaller rung.
                drop(streams);
                match next_rung(&mut ctx, &mut batch, &mut reserve) {
                    Some(note) => {
                        fallbacks.push(format!("{note} (allocation failed at the planned size)"));
                        continue;
                    }
                    None => return Err(e),
                }
            }
            Err(e) => return Err(e),
        };

        progress("building tokenizer", 0, 1);
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let weight_bytes = model.total_bytes;
        let placement_out = model.placement.clone();
        let mut runtime = Runtime::new(model, tokenizer, streams, batch)?;
        runtime.set_draft(opts.draft);

        return Ok((
            runtime,
            LoadReport {
                config,
                placement: placement_out,
                weight_bytes,
                load_ms: start.elapsed().as_secs_f64() * 1000.0,
                fallbacks,
            },
        ));
    }
}

/// Expert slots per layer for a MoE model too big for the GPUs: as many as
/// the free VRAM holds once everything else is placed, so the misses that
/// stream from the file are as few as memory allows. `None` when the model
/// fits whole, or is not MoE.
fn auto_expert_cache(gguf: &GgufFile, config: &Config, sizes: &WeightSizes, devices: &[Device],
    opts: &LoadOptions) -> Option<usize> {
    let moe = config.moe.as_ref()?;
    let used: Vec<&Device> = match &opts.split {
        SplitMode::Single(target) => devices.iter().filter(|d| d.ordinal == *target).collect(),
        _ => devices.iter().collect(),
    };
    let budget: u64 = used.iter().map(|d| d.free_mem.saturating_sub(opts.reserve_bytes)).sum();
    let n_layers = config.n_run_layers();
    let state: u64 = (0..n_layers).map(|l| match config.layer_kind(l) {
        crate::config::LayerKind::Attention => config.kv_bytes_per_layer(opts.ctx),
        crate::config::LayerKind::LinearAttention => config.ssm_bytes_per_layer(),
    }).sum();
    let scratch = used.len() as u64 * crate::plan::scratch_bytes(config, opts.batch, true);
    let needed = sizes.total() + state + scratch;
    if needed <= budget {
        return None;
    }
    // Bytes one slot costs across every layer, and what the experts cost now.
    let (mut per_slot, mut experts) = (0u64, 0u64);
    for l in 0..n_layers {
        for kind in ["gate", "up", "down"] {
            if let Some(t) = gguf.tensor(&format!("blk.{l}.ffn_{kind}_exps.weight")) {
                per_slot += t.nbytes / moe.n_experts as u64;
                experts += t.nbytes;
            }
        }
    }
    if per_slot == 0 {
        return None;
    }
    // A layer split never lands exactly on the budget; keep 6% of it spare.
    let room = (budget - budget / 16).checked_sub(needed - experts)?;
    let slots = (room / per_slot) as usize;
    (slots >= moe.n_experts_used).then_some(slots.min(moe.n_experts))
}

/// The next smaller configuration to try, or `None` when out of options.
///
/// Batch shrinks first because it costs only prefill speed; context shrinks
/// second because it changes what the model can do; the reserve is given up last
/// because eating into it risks a driver-level failure.
fn next_rung(ctx: &mut usize, batch: &mut usize, reserve: &mut u64) -> Option<String> {
    if *batch > 64 {
        *batch = (*batch / 2).max(64);
        return Some(format!("batch size reduced to {batch}"));
    }
    if *ctx > 512 {
        *ctx = (*ctx / 2).max(512);
        return Some(format!("context reduced to {ctx}"));
    }
    if *reserve > (128 << 20) {
        *reserve = 128 << 20;
        return Some("VRAM reserve reduced to 128 MiB".to_string());
    }
    None
}

/// Resolves the requested GPU ordinals, or every usable device.
fn select_devices(requested: &[i32]) -> Result<Vec<Device>> {
    let all = device::devices()?;
    if all.is_empty() {
        return Err(CoreError::NoDevices);
    }

    let usable: Vec<Device> = all
        .into_iter()
        .filter(|d| d.generation().is_supported())
        .collect();
    if usable.is_empty() {
        return Err(CoreError::NoDevices);
    }

    if requested.is_empty() {
        return Ok(usable);
    }

    let mut out = Vec::new();
    for &ord in requested {
        match usable.iter().find(|d| d.ordinal == ord) {
            Some(d) => out.push(d.clone()),
            None => {
                return Err(CoreError::Other(format!(
                    "GPU {ord} was requested with --gpus but is not usable. \
                     Run `airun gpu` to see the available ordinals."
                )));
            }
        }
    }
    Ok(out)
}

/// Formats messages into the ChatML layout Qwen and most chat models use.
///
/// The GGUF ships a Jinja template, which airun stores but does not evaluate;
/// ChatML covers the models the runtime supports today. Anything else should be
/// passed pre-formatted with `--raw`.
pub fn apply_chat_template(system: Option<&str>, user: &str, prefill: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str("<|im_start|>system\n");
        out.push_str(sys);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>user\n");
    out.push_str(user);
    out.push_str("<|im_end|>\n");
    out.push_str("<|im_start|>assistant\n");
    if let Some(p) = prefill {
        out.push_str(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_ladder_shrinks_batch_then_context_then_reserve() {
        let (mut ctx, mut batch, mut reserve) = (4096usize, 512usize, 384u64 << 20);

        assert!(next_rung(&mut ctx, &mut batch, &mut reserve).unwrap().contains("batch"));
        assert_eq!(batch, 256);
        assert_eq!(ctx, 4096, "context should not move while batch can shrink");

        while batch > 64 {
            next_rung(&mut ctx, &mut batch, &mut reserve);
        }
        assert!(next_rung(&mut ctx, &mut batch, &mut reserve).unwrap().contains("context"));
        assert_eq!(ctx, 2048);
    }

    #[test]
    fn fallback_ladder_terminates() {
        let (mut ctx, mut batch, mut reserve) = (4096usize, 512usize, 384u64 << 20);
        let mut steps = 0;
        while next_rung(&mut ctx, &mut batch, &mut reserve).is_some() {
            steps += 1;
            assert!(steps < 64, "ladder did not terminate");
        }
        assert_eq!(batch, 64);
        assert_eq!(ctx, 512);
    }

    #[test]
    fn chat_template_is_well_formed_chatml() {
        let s = apply_chat_template(Some("Be brief."), "Hello", None);
        assert!(s.starts_with("<|im_start|>system\nBe brief.<|im_end|>\n"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
        assert_eq!(s.matches("<|im_start|>").count(), 3);

        let no_sys = apply_chat_template(None, "Hi", None);
        assert!(no_sys.starts_with("<|im_start|>user\n"));
    }
}
