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
use crate::plan::{AllocMode, Placement, PlacementRequest, SplitMode, WeightSizes, plan};
use crate::runtime::Runtime;
use crate::tokenizer::Tokenizer;

/// What to load and how.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub ctx: usize,
    /// No context was asked for: use the largest the GPUs hold with every
    /// weight in VRAM (see auto_ctx), `ctx` being the floor.
    pub ctx_auto: bool,
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
    /// leaves the head unloaded; ignored on files without one. None decides
    /// by model: DEFAULT_DRAFT on dense models, off on MoE ones (see load).
    pub draft: Option<usize>,
    /// The MTP head in a file of its own (llama.cpp's `convert --mtp` writes
    /// `mtp-<name>.gguf` next to a model converted with `--no-mtp`). None
    /// looks for one next to the model when the model has no head.
    pub mtp_file: Option<std::path::PathBuf>,
    /// Conversations kept side by side (serve --parallel); their caches are
    /// planned for and allocated at load. Turns MTP off when above 1.
    pub slots: usize,
    /// Where weights may live: only VRAM, VRAM then RAM, or airun's choice.
    pub alloc: AllocMode,
    /// The GPU filled first; None picks the fastest.
    pub main_gpu: Option<i32>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            ctx: 4096,
            ctx_auto: false,
            batch: 512,
            gpus: Vec::new(),
            split: SplitMode::Auto,
            gpu_layers: None,
            expert_cache: None,
            kv_type: Default::default(),
            reserve_bytes: 384 << 20,
            allow_fallback: true,
            draft: Some(0),
            mtp_file: None,
            slots: 1,
            alloc: AllocMode::Auto,
            main_gpu: None,
        }
    }
}

/// Drafts per step when `--draft` is not given, on dense models with an MTP
/// head (Qwen3.8-27B: 26 -> 68 tok/s on an RTX 5060 Ti). 4 beat 3 in every
/// alternated sweep (2026-11-05, 2x 5060 Ti, 128k q4_0): split 43.2 -> 45.0
/// on prose and 47.6 -> 48.2 on code, one GPU 54.1 -> 55.0; 5 lost on prose.
pub const DEFAULT_DRAFT: usize = 4;

/// The same on MoE models, when every expert fits in VRAM. Fewer than on
/// dense models: each verified row wakes its own experts. Qwen3.8-35B-A3B
/// Q5_K_M on 2x RTX 5060 Ti (2026-11-06): no MTP 109.6, draft 2 137.6, draft 3
/// 128.2 tok/s on prose; 108.8 / 132.1 / 127.4 on code.
pub const MOE_DEFAULT_DRAFT: usize = 2;

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
    config.slots = opts.slots.max(1);

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
    // One MTP head keeps one sequence: with several slots it drafts for a
    // request running alone (see sched.rs), and the batched steps decode plainly.
    let draft_wanted = opts.draft != Some(0);
    let config_slots = opts.slots.max(1);
    let mut mtp_src = if draft_wanted { mtp_source(path, &gguf, &config, opts.mtp_file.as_deref())? } else { None };
    let moe_default = opts.draft.is_none() && config.moe.is_some();
    let mut draft = match opts.draft {
        Some(d) => d,
        None if config.moe.is_some() => MOE_DEFAULT_DRAFT,
        None => DEFAULT_DRAFT,
    };
    // On a MoE the head pays only with every expert in VRAM: the verify's
    // rows route to many more experts than one token does, and the ones in
    // RAM cross PCIe (35B-A3B on one RTX 5060 Ti, experts partly in RAM:
    // 44.7 -> 34.5 tok/s with draft 2). Same test plan_experts makes below,
    // counting the head and its cache.
    if moe_default && opts.alloc != AllocMode::Gpu && opts.expert_cache.is_none() {
        if let Some(src) = &mtp_src {
            let mut with_head = sizes.clone();
            with_head.output += crate::model::mtp_head_bytes(src, &config) + config.kv_bytes_per_layer(opts.ctx);
            if plan_experts(&gguf, &config, &with_head, &devices, opts).is_some() {
                eprintln!("note: the MTP head is off: this MoE model does not fit the GPUs whole, and \
                           drafting slows decoding when experts sit in RAM; --draft {MOE_DEFAULT_DRAFT} \
                           turns it on anyway");
                draft = 0;
            }
        }
    }
    // --expert-cache streams the misses from the file, which the head's wider
    // verify only multiplies.
    if moe_default && opts.expert_cache.is_some() {
        draft = 0;
    }
    mtp_src = mtp_src.filter(|_| draft > 0);
    let load_mtp = mtp_src.is_some();
    if draft > 0 && !load_mtp && opts.draft.is_some() {
        let reason = if config.nextn_predict_layers == 0 {
            "the file has no MTP head, and no mtp-*.gguf for it sits next to it (--mtp names one)"
        } else if config.nextn_predict_layers != 1 {
            "the file declares more than one MTP head"
        } else if !config.is_hybrid() {
            "MTP heads run on Qwen3.5/3.8 hybrid models only"
        } else {
            "the file declares an MTP head but does not carry its weights"
        };
        eprintln!("note: --draft {draft} is ignored: {reason}.");
    }
    let head_bytes = mtp_src.as_ref().map_or(0, |src| crate::model::mtp_head_bytes(src, &config));
    let auto;
    let opts = if opts.ctx_auto {
        let ctx = auto_ctx(&config, &sizes, &devices, opts, load_mtp.then_some(head_bytes));
        auto = LoadOptions { ctx, ..opts.clone() };
        &auto
    } else {
        opts
    };
    if load_mtp {
        // The head and its own KV cache sit with the output projection.
        sizes.output += head_bytes + config.kv_bytes_per_layer(opts.ctx);
    }
    let mut split = opts.split.clone();
    let mut device_order: Vec<i32> = Vec::new();
    let mut expert_note = None;
    let expert_slots: Option<Vec<usize>> = match opts.expert_cache {
        Some(slots) => {
            let moe = config.moe.as_ref().ok_or_else(|| CoreError::Other(
                "--expert-cache requires a MoE model; omit it for dense models".into()))?;
            if slots < moe.n_experts_used || slots > moe.n_experts {
                return Err(CoreError::Other(format!(
                    "--expert-cache must be between {} and {}; choose enough slots for every active expert",
                    moe.n_experts_used, moe.n_experts)));
            }
            expert_note = Some(format!("MoE expert cache: {slots} slots per layer"));
            Some(vec![slots; config.n_run_layers()])
        }
        // --alloc gpu: no expert goes to RAM; a MoE that does not fit then
        // gets a smaller context or the planner's error.
        None if opts.alloc == AllocMode::Gpu => None,
        None => match plan_experts(&gguf, &config, &sizes, &devices, opts) {
            Some(p) => {
                if let (Some(counts), SplitMode::Auto) = (p.counts, &opts.split) {
                    split = SplitMode::Manual(counts);
                    device_order = p.order;
                }
                expert_note = Some(p.note);
                Some(p.slots)
            }
            None => None,
        },
    };
    if let (Some(slots), Some(moe)) = (&expert_slots, config.moe.as_ref()) {
        for l in 0..config.n_run_layers() {
            for kind in ["gate", "up", "down"] {
                if let Some(t) = gguf.tensor(&format!("blk.{l}.ffn_{kind}_exps.weight")) {
                    sizes.per_layer[l] -= t.nbytes - t.nbytes / moe.n_experts as u64 * slots[l] as u64;
                }
            }
        }
    }

    // Which experts take the VRAM slots: the most routed ones of earlier runs.
    let mut profile = match (&expert_slots, config.moe.as_ref()) {
        (Some(_), Some(moe)) if crate::model::experts_mapped() =>
            crate::expert_profile::ExpertProfile::open(gguf.path(), gguf.file_len(), config.n_run_layers(), moe.n_experts),
        _ => None,
    };
    if let (Some(p), Some(note)) = (&profile, &mut expert_note) {
        let known = (0..config.n_run_layers()).filter(|&l| p.knows(l)).count();
        note.push_str(&if known > 0 {
            format!("; resident experts chosen by the routing profile {}", p.path().display())
        } else {
            format!("; no routing profile yet, it is learned from this run into {}", p.path().display())
        });
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
            split: split.clone(),
            gpu_layers: opts.gpu_layers,
            reserve_bytes: reserve,
            allow_ctx_shrink: opts.allow_fallback,
            alloc: opts.alloc,
            main_gpu: opts.main_gpu,
            device_order: device_order.clone(),
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

        if let Some(note) = &expert_note {
            let rest = if crate::model::experts_mapped() {
                "the rest in pinned RAM, read by the GPU"
            } else {
                "misses read from the GGUF"
            };
            placement.notes.push(format!("{note}; {rest}"));
        }
        if let Some(ix) = &config.indexer {
            if placement.ctx > ix.top_k && std::env::var("AIRUN_QSA").as_deref() == Ok("off") {
                placement.notes.push(format!(
                    "QSA sparse attention off (AIRUN_QSA=off): past {} tokens every query attends to the whole \
                     context, where the model was trained to see its best-scored {} -- output may drift",
                    ix.top_k, ix.top_k));
            }
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

        // Steps of Model::load: embedding, output norm, output, then one per
        // layer. On more than one GPU the stage names where the layers go, so
        // the load shows each GPU taking its share.
        let labels = upload_labels(&placement, config.n_run_layers());
        let attempt = Model::load(&gguf, config.clone(), placement.clone(), &streams, expert_slots.as_deref(), profile.as_ref(), mtp_src.as_ref(), |d, t| {
            let stage = labels.get(d.min(labels.len().saturating_sub(1))).map_or("uploading weights", |s| s.as_str());
            progress(stage, d, t)
        });

        let mut model = match attempt {
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
        model.expert_profile = profile.take().map(std::cell::RefCell::new);

        progress("building tokenizer", 0, 1);
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let weight_bytes = model.total_bytes;
        let placement_out = model.placement.clone();
        let mut runtime = Runtime::new(model, tokenizer, streams, batch)?;
        runtime.set_draft(draft);
        runtime.set_slots(config_slots)?;

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

/// Where the MTP head comes from: the model file when it carries one, else
/// `--mtp`'s file, else the one `mtp-*.gguf` next to the model that belongs
/// to it. A file of its own must be the same model: architecture, width,
/// vocabulary and trunk depth.
fn mtp_source(path: &Path, gguf: &std::sync::Arc<GgufFile>, config: &Config, explicit: Option<&Path>)
              -> Result<Option<std::sync::Arc<GgufFile>>> {
    if let Some(file) = explicit {
        let side = GgufFile::open(file).map_err(|e| CoreError::Other(format!(
            "--mtp {}: {e}. Pass the mtp-*.gguf written for this model (llama.cpp: convert_hf_to_gguf.py --mtp)",
            file.display())))?;
        return match mtp_mismatch(config, &side) {
            None => {
                eprintln!("MTP head: {}", file.display());
                Ok(Some(std::sync::Arc::new(side)))
            }
            Some(why) => Err(CoreError::Other(format!(
                "--mtp {} does not fit this model: {why}. Use the mtp-*.gguf converted from the same checkpoint",
                file.display()))),
        };
    }
    if crate::model::mtp_supported(gguf.as_ref(), config) {
        return Ok(Some(std::sync::Arc::clone(gguf)));
    }
    if !config.is_hybrid() {
        return Ok(None);
    }
    let Some(dir) = path.parent() else { return Ok(None) };
    let Ok(entries) = std::fs::read_dir(if dir.as_os_str().is_empty() { Path::new(".") } else { dir }) else {
        return Ok(None);
    };
    let mut found: Vec<(std::path::PathBuf, GgufFile)> = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
        if !name.starts_with("mtp-") || !name.ends_with(".gguf") || p == path {
            continue;
        }
        if let Ok(side) = GgufFile::open(&p) {
            if mtp_mismatch(config, &side).is_none() {
                found.push((p, side));
            }
        }
    }
    match found.len() {
        0 => Ok(None),
        1 => {
            let (p, side) = found.pop().unwrap();
            eprintln!("MTP head: {} (found next to the model; --mtp picks another, --draft 0 skips it)", p.display());
            Ok(Some(std::sync::Arc::new(side)))
        }
        _ => {
            let names: Vec<String> = found.iter()
                .map(|(p, _)| p.file_name().unwrap_or_default().to_string_lossy().into_owned()).collect();
            eprintln!("note: several MTP heads fit this model ({}); --mtp <file> picks one", names.join(", "));
            Ok(None)
        }
    }
}

/// Why `side` cannot be this model's MTP head, or None when it can.
fn mtp_mismatch(config: &Config, side: &GgufFile) -> Option<String> {
    let arch = side.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("?");
    if arch != config.arch_name {
        return Some(format!("it is a {arch} file and the model is {}", config.arch_name));
    }
    let c = match Config::from_gguf(side) {
        Ok(c) => c,
        Err(e) => return Some(format!("it is not a model file airun reads ({e})")),
    };
    if c.arch_name != config.arch_name {
        return Some(format!("it is a {} file and the model is {}", c.arch_name, config.arch_name));
    }
    if c.n_embd != config.n_embd || c.vocab_size != config.vocab_size {
        return Some(format!("width {} / vocabulary {} against the model's {} / {}",
            c.n_embd, c.vocab_size, config.n_embd, config.vocab_size));
    }
    if c.nextn_predict_layers != 1 {
        return Some(format!("it declares {} MTP heads, and airun runs exactly one", c.nextn_predict_layers));
    }
    if c.n_run_layers() != config.n_run_layers() {
        return Some(format!("its head follows {} layers and the model has {}", c.n_run_layers(), config.n_run_layers()));
    }
    if !crate::model::mtp_head_present(side, config) {
        return Some(format!("it has no blk.{}.nextn.eh_proj tensor", config.n_run_layers()));
    }
    None
}

/// The context to use when none was asked for: the largest the GPUs hold
/// with every weight still in VRAM, up to what the model was trained for.
///
/// The GPUs are the ones the model takes at the default context (`opts.ctx`):
/// a model that fits one card stays on it, since spreading it over a second
/// for context would slow every token. When the weights do not fit the
/// GPUs even then (a MoE whose experts go to RAM), the default stays: each
/// gigabyte of KV cache would push weights out of VRAM.
fn auto_ctx(config: &Config, sizes: &WeightSizes, devices: &[Device], opts: &LoadOptions,
            mtp_head: Option<u64>) -> usize {
    let base = opts.ctx;
    let sizes_at = |ctx: usize| {
        let mut s = sizes.clone();
        if let Some(head) = mtp_head {
            s.output += head + config.kv_bytes_per_layer(ctx);
        }
        s
    };
    let req_at = |ctx: usize, split: SplitMode| PlacementRequest {
        ctx,
        batch: opts.batch,
        split,
        gpu_layers: None,
        reserve_bytes: opts.reserve_bytes,
        allow_ctx_shrink: false,
        alloc: AllocMode::Gpu,
        main_gpu: opts.main_gpu,
        device_order: Vec::new(),
    };
    let Ok(at_base) = plan(config, &sizes_at(base), devices, &req_at(base, opts.split.clone())) else {
        return base;
    };
    // Plus any GPU as fast as those: across equal cards a split costs only
    // the handoff, each card running its share of the layers at the same
    // speed (two RTX 5060 Ti: the 27B goes from 4k to the context both hold).
    let slowest = devices.iter().filter(|d| at_base.devices().contains(&d.ordinal))
        .map(|d| d.perf_score()).fold(f64::INFINITY, f64::min);
    let used: Vec<Device> = devices.iter()
        .filter(|d| at_base.devices().contains(&d.ordinal) || d.perf_score() >= 0.9 * slowest)
        .cloned().collect();
    let split = match used.as_slice() {
        [one] => SplitMode::Single(one.ordinal),
        _ => opts.split.clone(),
    };
    // Same GPUs, and no smaller prefill batch than the default context got.
    let fits = |ctx: usize| plan(config, &sizes_at(ctx), &used, &req_at(ctx, split.clone()))
        .is_ok_and(|p| p.ctx == ctx && p.batch >= at_base.batch);
    let cap = config.train_ctx.max(base);
    let (mut lo, mut hi) = (base, cap);
    if fits(hi) {
        lo = hi;
    } else {
        while hi - lo > 256 {
            let mid = (lo + (hi - lo) / 2) / 256 * 256;
            if mid <= lo {
                break;
            }
            if fits(mid) { lo = mid } else { hi = mid }
        }
    }
    if lo > base {
        eprintln!("context: {lo} tokens, the most the GPUs hold with every weight in VRAM (-c sets it)");
    }
    lo
}

/// How a MoE model too big for the GPUs is spread: expert slots per layer,
/// and (when it decides the split) layers per device in the planner's order.
struct ExpertPlan {
    slots: Vec<usize>,
    counts: Option<Vec<f32>>,
    /// The device order `counts` is in, which the planner must follow.
    order: Vec<i32>,
    note: String,
}

/// Host-to-device bandwidth below which a GPU only takes layers whose experts
/// all fit in its VRAM: reading experts across a PCIe 3.0 x4 slot (measured
/// 1.6 GB/s on a 5060 Ti) cost more than the rest of the layer.
const SLOW_LINK_GBPS: f64 = 8.0;

/// Plans expert residency for a MoE model too big for the GPUs, or `None`
/// when it fits whole or is not MoE. GPUs with a slow host link get whole
/// layers, all experts resident; the rest of the layers go to the fast-link
/// GPUs with as many expert slots per layer as their VRAM holds, the other
/// experts left in host memory.
fn plan_experts(gguf: &GgufFile, config: &Config, sizes: &WeightSizes, devices: &[Device],
    opts: &LoadOptions) -> Option<ExpertPlan> {
    let moe = config.moe.as_ref()?;
    let ne = moe.n_experts as u64;
    let mut used: Vec<&Device> = match &opts.split {
        SplitMode::Single(target) => devices.iter().filter(|d| d.ordinal == *target).collect(),
        _ => devices.iter().collect(),
    };
    // The planner's order: the main GPU (by default the fastest) first, which
    // takes the first layers.
    used.sort_by(|a, b| b.perf_score().partial_cmp(&a.perf_score())
        .unwrap_or(std::cmp::Ordering::Equal).then(a.ordinal.cmp(&b.ordinal)));
    if let Some(i) = opts.main_gpu.and_then(|m| used.iter().position(|d| d.ordinal == m)) {
        let d = used.remove(i);
        used.insert(0, d);
    }
    let n_layers = config.n_run_layers();
    let state = |l: usize| match config.layer_kind(l) {
        crate::config::LayerKind::Attention => config.kv_bytes_per_layer(opts.ctx),
        crate::config::LayerKind::LinearAttention => config.ssm_bytes_per_layer(),
    };
    let experts: Vec<u64> = (0..n_layers).map(|l| ["gate", "up", "down"].iter()
        .filter_map(|k| gguf.tensor(&format!("blk.{l}.ffn_{k}_exps.weight")).map(|t| t.nbytes)).sum()).collect();
    let budget = |d: &Device| {
        let b = d.free_mem.saturating_sub(opts.reserve_bytes);
        b - b / 16 - crate::plan::scratch_bytes(config, opts.batch, true)
    };
    let total_budget: u64 = used.iter().map(|d| budget(d)).sum();
    let fixed = sizes.embedding + sizes.output + sizes.other;
    let full: u64 = (0..n_layers).map(|l| sizes.per_layer[l] + state(l)).sum::<u64>() + fixed;
    if full <= total_budget {
        return None;
    }

    let mut slow: Vec<bool> = if used.len() > 1 {
        used.iter().map(|d| measure_h2d_gbps(d.ordinal).is_some_and(|g| g < SLOW_LINK_GBPS)).collect()
    } else {
        vec![false]
    };
    // Slow-link GPUs take the last layers, so they go last in the order the
    // planner lays layers out in. On two identical cards the performance
    // order falls back to the ordinal, and a slow-link card at ordinal 0 used
    // to get the first layers with the last layers' expert counts: Qwen4 on
    // two RTX 5060 Ti failed to load, one card 7 GiB over.
    let mut paired: Vec<(&Device, bool)> = used.iter().copied().zip(slow.iter().copied()).collect();
    paired.sort_by_key(|&(_, s)| s);
    used = paired.iter().map(|&(d, _)| d).collect();
    slow = paired.iter().map(|&(_, s)| s).collect();
    // Whole layers for the slow-link GPUs, taken from the end, where the
    // planner puts its last devices; the output head sits with the last one.
    let mut counts = vec![0usize; used.len()];
    let mut end = n_layers;
    for di in (0..used.len()).rev() {
        if !slow[di] || slow.iter().all(|&s| s) {
            continue;
        }
        let mut room = budget(used[di]).saturating_sub(if di == used.len() - 1 { sizes.output + sizes.other } else { 0 });
        while end > 0 {
            let cost = sizes.per_layer[end - 1] + state(end - 1);
            if cost > room { break; }
            room -= cost;
            end -= 1;
            counts[di] += 1;
        }
    }
    // The fast GPUs share layers 0..end with uniform slots.
    let fast: Vec<usize> = (0..used.len()).filter(|&i| !(slow[i] && !slow.iter().all(|&s| s))).collect();
    let fast_budget: u64 = fast.iter().map(|&i| budget(used[i])).sum();
    let out_on_fast = counts.last().copied().unwrap_or(0) == 0;
    let fast_fixed = sizes.embedding + if out_on_fast { sizes.output + sizes.other } else { 0 };
    let base: u64 = (0..end).map(|l| sizes.per_layer[l] - experts[l] + state(l)).sum::<u64>() + fast_fixed;
    let per_slot: u64 = (0..end).map(|l| experts[l] / ne).sum();
    let slots_fast = if per_slot == 0 { moe.n_experts } else {
        ((fast_budget.checked_sub(base)? / per_slot) as usize).min(moe.n_experts)
    };
    if slots_fast < moe.n_experts_used {
        return None;
    }
    // Split the fast layers in proportion to the fast GPUs' budgets.
    let mut left = end;
    for (k, &i) in fast.iter().enumerate() {
        let c = if k + 1 == fast.len() { left } else {
            ((end as u64 * budget(used[i]) / fast_budget.max(1)) as usize).min(left)
        };
        counts[i] = c;
        left -= c;
    }
    let slots: Vec<usize> = (0..n_layers).map(|l| if l < end { slots_fast } else { moe.n_experts }).collect();
    let whole = n_layers - end;
    let note = if whole > 0 {
        format!("MoE experts: {slots_fast} of {} per layer in VRAM on layers 0..{}; layers {end}..{} whole on the \
                 GPU(s) with a slow host link", moe.n_experts, end.saturating_sub(1), n_layers - 1)
    } else {
        format!("MoE expert cache: {slots_fast} slots per layer (sized to fill the GPUs; --expert-cache overrides)")
    };
    Some(ExpertPlan {
        slots,
        counts: (whole > 0).then(|| counts.iter().map(|&c| c as f32).collect()),
        order: used.iter().map(|d| d.ordinal).collect(),
        note,
    })
}

/// Pinned host-to-device bandwidth of `ordinal`, from a few 64 MiB copies.
fn measure_h2d_gbps(ordinal: i32) -> Option<f64> {
    const BYTES: usize = 64 << 20;
    device::set_device(ordinal).ok()?;
    let stream = Stream::new(ordinal).ok()?;
    let host = airun_cuda::PinnedBuffer::<u8>::new(BYTES).ok()?;
    let dev = airun_cuda::DeviceBuffer::<u8>::new(ordinal, BYTES).ok()?;
    dev.copy_from_host_at(0, host.as_slice(), &stream).ok()?;
    stream.synchronize().ok()?;
    let t = std::time::Instant::now();
    for _ in 0..3 {
        dev.copy_from_host_at(0, host.as_slice(), &stream).ok()?;
    }
    stream.synchronize().ok()?;
    Some(3.0 * BYTES as f64 / t.elapsed().as_secs_f64() / 1e9)
}

/// The progress stage after each of Model::load's steps (index = steps done):
/// "uploading weights" on one GPU; on several, the embedding/output steps and
/// then, per layer, the GPU it goes to ("-> GPU 1 RTX 5060, layers 52..63"),
/// or RAM for layers kept there.
fn upload_labels(placement: &Placement, n_run: usize) -> Vec<String> {
    let total = n_run + 3;
    if placement.shards.len() < 2 {
        return vec!["uploading weights".to_string(); total + 1];
    }
    let label = |l: usize| -> String {
        let Some(s) = placement.shards.iter().find(|s| s.layers.contains(&l)) else {
            return "uploading weights".to_string();
        };
        let range = |v: &[usize]| match (v.first(), v.last()) {
            (Some(a), Some(b)) => format!("{a}..{b}"),
            _ => String::new(),
        };
        if s.ram_layers.contains(&l) {
            format!("-> RAM via GPU {}, layers {}", s.device, range(&s.ram_layers))
        } else {
            let vram: Vec<usize> = s.layers.iter().copied().filter(|x| !s.ram_layers.contains(x)).collect();
            let name = s.device_name.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ");
            format!("-> GPU {} {name}, layers {}", s.device, range(&vram))
        }
    };
    (0..=total)
        .map(|d| if d < 3 { "embedding + output".to_string() } else { label((d - 3).min(n_run.saturating_sub(1))) })
        .collect()
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
