//! VRAM planning: deciding what goes on which GPU before anything is allocated.
//!
//! The plan is computed from exact tensor sizes read out of the GGUF header, not
//! from an estimate, so a plan that says it fits does fit. Everything the planner
//! decides is reported in [`Placement::notes`] -- an automatic split that cannot
//! explain itself is impossible to debug when a model fails to load.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use airun_cuda::Device;
use airun_gguf::GgufFile;

use crate::config::Config;
use crate::error::{CoreError, Result};

/// How to spread a model across the selected GPUs.
#[derive(Debug, Clone, PartialEq)]
pub enum SplitMode {
    /// Pick the layout automatically: one GPU when the model fits on one,
    /// otherwise contiguous layer ranges sized by free VRAM.
    Auto,
    /// Contiguous layer ranges in the given proportions, one entry per GPU.
    Manual(Vec<f32>),
    /// Everything on one device.
    Single(i32),
}

/// Where weights may live (`--alloc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocMode {
    /// airun decides: as few GPUs as the model needs, the main one first, and
    /// RAM only for what no GPU holds.
    #[default]
    Auto,
    /// GPU memory only. A model that does not fit gets a smaller context, or
    /// an error saying what is missing; nothing goes to RAM.
    Gpu,
    /// The selected GPUs, then RAM for what they do not hold, keeping the
    /// requested context.
    Memory,
}

impl AllocMode {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "gpu" => Ok(Self::Gpu),
            "memory" | "ram" => Ok(Self::Memory),
            _ => Err(format!("unknown --alloc '{s}': use auto, gpu or memory")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpu => "gpu",
            Self::Memory => "memory",
        }
    }

    /// Whether weights may be kept in RAM when the GPUs are full.
    pub fn spills(self) -> bool {
        self != Self::Gpu
    }
}

/// What the caller wants; the planner turns this into a [`Placement`].
#[derive(Debug, Clone)]
pub struct PlacementRequest {
    pub ctx: usize,
    /// Largest number of tokens processed in one forward pass.
    pub batch: usize,
    pub split: SplitMode,
    /// Cap on layers placed on GPUs. `None` means all of them.
    pub gpu_layers: Option<usize>,
    /// Headroom left untouched on each GPU, for driver overhead and
    /// fragmentation. Windows WDDM in particular will not hand out the last
    /// few hundred MiB.
    pub reserve_bytes: u64,
    /// Permit shrinking the context to make the model fit.
    pub allow_ctx_shrink: bool,
    pub alloc: AllocMode,
    /// The GPU filled first and the one that runs layers kept in RAM. None:
    /// the fastest.
    pub main_gpu: Option<i32>,
}

impl Default for PlacementRequest {
    fn default() -> Self {
        Self {
            ctx: 4096,
            batch: 512,
            split: SplitMode::Auto,
            gpu_layers: None,
            reserve_bytes: 384 << 20,
            allow_ctx_shrink: true,
            alloc: AllocMode::Auto,
            main_gpu: None,
        }
    }
}

/// One GPU's share of the model.
#[derive(Debug, Clone)]
pub struct Shard {
    pub device: i32,
    pub device_name: String,
    pub layers: Vec<usize>,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub scratch_bytes: u64,
    pub budget_bytes: u64,
    /// Holds the token embedding table.
    pub hosts_embedding: bool,
    /// Holds the final norm and output projection.
    pub hosts_output: bool,
    /// Layers of this shard whose weights stay in pinned RAM and are copied
    /// into `staging_bytes` of VRAM right before they run. Their KV cache or
    /// recurrent state is on the GPU like any other layer's.
    pub ram_layers: Vec<usize>,
    pub ram_bytes: u64,
    pub staging_bytes: u64,
}

impl Shard {
    pub fn total_bytes(&self) -> u64 {
        self.weight_bytes + self.kv_bytes + self.scratch_bytes + self.staging_bytes
    }

    pub fn headroom(&self) -> i64 {
        self.budget_bytes as i64 - self.total_bytes() as i64
    }
}

/// The resolved layout.
#[derive(Debug, Clone)]
pub struct Placement {
    pub shards: Vec<Shard>,
    /// Context actually planned for, which may be below what was requested.
    pub ctx: usize,
    /// Prefill batch planned for, which may be below what was requested.
    pub batch: usize,
    pub notes: Vec<String>,
}

impl Placement {
    pub fn device_for_layer(&self, layer: usize) -> Option<i32> {
        self.shards
            .iter()
            .find(|s| s.layers.contains(&layer))
            .map(|s| s.device)
    }

    pub fn embedding_device(&self) -> i32 {
        self.shards
            .iter()
            .find(|s| s.hosts_embedding)
            .map(|s| s.device)
            .unwrap_or(0)
    }

    pub fn output_device(&self) -> i32 {
        self.shards
            .iter()
            .find(|s| s.hosts_output)
            .map(|s| s.device)
            .unwrap_or(0)
    }

    pub fn devices(&self) -> Vec<i32> {
        self.shards.iter().map(|s| s.device).collect()
    }

    /// Whether layer `l`'s weights stay in RAM.
    pub fn layer_in_ram(&self, l: usize) -> bool {
        self.shards.iter().any(|s| s.ram_layers.contains(&l))
    }

    /// VRAM staging for the layers in RAM on `device`.
    pub fn staging_bytes(&self, device: i32) -> u64 {
        self.shards.iter().filter(|s| s.device == device).map(|s| s.staging_bytes).sum()
    }

    /// Number of points in the forward pass where activations cross GPUs.
    pub fn crossings(&self) -> usize {
        self.shards.len().saturating_sub(1)
    }

    pub fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "memory plan (context {} tokens):", self.ctx);
        for s in &self.shards {
            let roles = match (s.hosts_embedding, s.hosts_output) {
                (true, true) => " +embed +output",
                (true, false) => " +embed",
                (false, true) => " +output",
                (false, false) => "",
            };
            let layers = if s.layers.is_empty() {
                "none".to_string()
            } else {
                format!(
                    "{}..{} ({})",
                    s.layers[0],
                    s.layers[s.layers.len() - 1],
                    s.layers.len()
                )
            };
            let _ = writeln!(
                out,
                "  GPU {} {:<26} layers {:<16}{}",
                s.device, s.device_name, layers, roles
            );
            let _ = writeln!(
                out,
                "      weights {:>10}  kv {:>10}  scratch {:>10}  =  {:>10} of {:>10} budget",
                airun_cuda::human_bytes(s.weight_bytes),
                airun_cuda::human_bytes(s.kv_bytes),
                airun_cuda::human_bytes(s.scratch_bytes),
                airun_cuda::human_bytes(s.total_bytes()),
                airun_cuda::human_bytes(s.budget_bytes),
            );
            if !s.ram_layers.is_empty() {
                let _ = writeln!(
                    out,
                    "      layers {}..{} ({}) in RAM: {} pinned, copied through {} of VRAM before they run",
                    s.ram_layers[0],
                    s.ram_layers[s.ram_layers.len() - 1],
                    s.ram_layers.len(),
                    airun_cuda::human_bytes(s.ram_bytes),
                    airun_cuda::human_bytes(s.staging_bytes),
                );
            }
        }
        for note in &self.notes {
            let _ = writeln!(out, "  note: {note}");
        }
        out
    }
}

/// Exact per-tensor byte counts, grouped the way the planner needs them.
#[derive(Debug, Clone)]
pub struct WeightSizes {
    /// Bytes of `blk.N.*` tensors, indexed by layer.
    pub per_layer: Vec<u64>,
    /// Token embedding table.
    pub embedding: u64,
    /// Final norm plus the output projection, when it is not tied.
    pub output: u64,
    /// Anything that matched no bucket, so nothing is silently unaccounted for.
    pub other: u64,
}

impl WeightSizes {
    pub fn from_gguf(gguf: &GgufFile, n_layers: usize) -> Self {
        let mut per_layer = vec![0u64; n_layers];
        let mut embedding = 0;
        let mut output = 0;
        let mut other = 0;

        for t in gguf.tensors() {
            if let Some(rest) = t.name.strip_prefix("blk.") {
                if let Some((idx, _)) = rest.split_once('.') {
                    if let Ok(layer) = idx.parse::<usize>() {
                        if layer < n_layers {
                            per_layer[layer] += t.nbytes;
                            continue;
                        }
                    }
                }
                other += t.nbytes;
            } else if t.name.starts_with("token_embd") {
                embedding += t.nbytes;
            } else if t.name.starts_with("output") {
                output += t.nbytes;
            } else {
                other += t.nbytes;
            }
        }

        Self {
            per_layer,
            embedding,
            output,
            other,
        }
    }

    pub fn total(&self) -> u64 {
        self.per_layer.iter().sum::<u64>() + self.embedding + self.output + self.other
    }
}

/// Activation scratch for a forward pass of `batch` tokens on one device.
///
/// Counts the largest set of buffers alive at once, then adds a quarter for
/// allocator slack. Being generous here is cheap; being wrong means an
/// out-of-memory error partway through a load.
pub fn scratch_bytes(cfg: &Config, batch: usize, hosts_output: bool) -> u64 {
    let f32_size = 4u64;
    let b = batch as u64;

    let residual = b * cfg.n_embd as u64; // x
    let normed = b * cfg.n_embd as u64; // normalized copy
    let q = b * cfg.q_dim() as u64;
    let k = b * cfg.kv_dim() as u64;
    let v = b * cfg.kv_dim() as u64;
    let attn_out = b * cfg.q_dim() as u64;
    let proj = b * cfg.n_embd as u64;
    // gate and up are alive simultaneously for the SwiGLU.
    let ffn = 2 * b * cfg.ffn_len as u64;

    let mut total = (residual + normed + q + k + v + attn_out + proj + ffn) * f32_size;

    // Hybrids need room for the packed qkv projection, its gate, and the
    // delta-net intermediates, none of which the dense path allocates.
    if let Some(ssm) = &cfg.ssm {
        // The packed projection, and its convolved copy (batches convolve out of place).
        let qkv = 2 * b * ssm.qkv_dim() as u64;
        let gate = b * ssm.inner_size as u64;
        let core = b * ssm.v_dim() as u64;
        let coeffs = 2 * b * ssm.n_v_heads as u64;
        total += (qkv + gate + core + coeffs) * f32_size;
    }
    // A gated attention output doubles the q projection.
    if cfg.attn_output_gate {
        total += b * cfg.q_dim() as u64 * 2 * f32_size;
    }
    // MoE holds one hidden vector per active expert per token.
    if let Some(moe) = &cfg.moe {
        let h = b * (moe.n_experts_used * moe.expert_ff) as u64;
        let routing = b * moe.n_experts as u64 + 2 * b * moe.n_experts_used as u64;
        let shared = 2 * b * moe.shared_expert_ff as u64;
        total += (h + routing + shared) * f32_size;
    }

    if hosts_output {
        // Logits dominate for a large vocabulary, but only one row is ever
        // needed: sampling looks at the last position.
        total += cfg.vocab_size as u64 * f32_size;
    }
    total += b * 4; // position ids
    // Bound split-attention scratch at its maximum, including long contexts.
    total += (cfg.n_head * 128 * (cfg.head_dim + 2) * 4) as u64;
    // Hyper-connections widen the residual to `count` streams, and a mix
    // holds the normalized streams and their gate; the PLE layer's device
    // also keeps its key, gated value and embedding rows.
    if let Some(h) = &cfg.hc {
        let wide = b * (h.count * cfg.n_embd) as u64;
        total += ((h.count as u64 - 1) * b * cfg.n_embd as u64 + 2 * wide) * f32_size;
        if let Some(p) = &cfg.ple {
            total += (2 * wide + b * cfg.n_embd as u64 + b * (p.n_heads() * p.head_dim) as u64) * f32_size;
        }
    }
    // Batched MoE keeps every (token, expert) pair's output for the combine.
    if let Some(moe) = &cfg.moe {
        total += b * (moe.n_experts_used * cfg.n_embd) as u64 * f32_size;
    }
    // The QSA indexer's queries and keys.
    if let Some(ix) = &cfg.indexer {
        total += b * ((ix.heads + 1) * ix.dim) as u64 * f32_size;
    }

    total + total / 4
}

/// Builds a placement, or explains precisely why the model will not fit.
pub fn plan(
    cfg: &Config,
    sizes: &WeightSizes,
    devices: &[Device],
    req: &PlacementRequest,
) -> Result<Placement> {
    if devices.is_empty() {
        return Err(CoreError::NoDevices);
    }

    let mut notes = Vec::new();
    let mut ctx = req.ctx.min(cfg.train_ctx.max(1));
    if ctx < req.ctx {
        notes.push(format!(
            "context clamped to {ctx}, the length this model was trained for"
        ));
    }

    // Devices are considered fastest-first so a single-GPU fit lands on the
    // best card, and a proportional split gives the fastest card the most work.
    let mut ordered: Vec<&Device> = devices.iter().collect();
    ordered.sort_by(|a, b| {
        b.perf_score()
            .partial_cmp(&a.perf_score())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.ordinal.cmp(&b.ordinal))
    });
    if let Some(main) = req.main_gpu {
        let i = ordered.iter().position(|d| d.ordinal == main).ok_or_else(|| CoreError::Other(format!(
            "--main-gpu {main} is not among the GPUs in use; list it in --gpus, or check `airun gpu`")))?;
        let d = ordered.remove(i);
        ordered.insert(0, d);
    }

    let budgets: BTreeMap<i32, u64> = ordered
        .iter()
        .map(|d| (d.ordinal, d.free_mem.saturating_sub(req.reserve_bytes)))
        .collect();

    // Only the layers the forward pass runs get placed. Multi-token-prediction
    // blocks sit at the end of `block_count` but are never executed, so giving
    // them VRAM would both waste it and hand the runtime a layer index that has
    // no loaded weights behind it.
    let n_layers = cfg.n_run_layers();
    let gpu_layers = req.gpu_layers.unwrap_or(n_layers).min(n_layers);
    if gpu_layers < n_layers {
        return Err(CoreError::PartialOffloadUnsupported {
            requested: gpu_layers,
            total: n_layers,
        });
    }

    // Shrink the context until the plan fits, if permitted. Each attempt is a
    // full placement, so what succeeds is exactly what will be allocated.
    loop {
        match try_place(cfg, sizes, &ordered, &budgets, req, ctx, notes.clone()) {
            Ok(p) => return Ok(p),
            Err(shortfall) => {
                let min_ctx = 512;
                if !req.allow_ctx_shrink || ctx <= min_ctx {
                    let total_budget: u64 = budgets.values().sum();
                    return Err(CoreError::DoesNotFit {
                        model: cfg.name.clone(),
                        needed: shortfall.needed,
                        available: total_budget,
                        ctx,
                        detail: shortfall.detail,
                    });
                }
                let next = (ctx / 2).max(min_ctx);
                notes.push(format!(
                    "context {ctx} did not fit ({}), retrying at {next}",
                    airun_cuda::human_bytes(shortfall.needed.saturating_sub(
                        budgets.values().sum::<u64>()
                    ))
                ));
                ctx = next;
            }
        }
    }
}

struct Shortfall {
    needed: u64,
    detail: String,
}

fn try_place(
    cfg: &Config,
    sizes: &WeightSizes,
    ordered: &[&Device],
    budgets: &BTreeMap<i32, u64>,
    req: &PlacementRequest,
    ctx: usize,
    mut notes: Vec<String>,
) -> std::result::Result<Placement, Shortfall> {
    // Per-layer state depends on what the layer is: softmax-attention layers
    // carry a KV cache that scales with context, linear-attention layers a
    // recurrent state that does not, and MTP layers are never run.
    let per_layer_state = |l: usize| -> u64 {
        if l >= cfg.n_run_layers() {
            return 0;
        }
        match cfg.layer_kind(l) {
            crate::config::LayerKind::Attention => cfg.kv_bytes_per_layer(ctx),
            crate::config::LayerKind::LinearAttention => cfg.ssm_bytes_per_layer(),
        }
    };

    let n_layers = cfg.n_run_layers();
    let layer_cost: Vec<u64> = (0..n_layers)
        .map(|l| sizes.per_layer.get(l).copied().unwrap_or(0) + per_layer_state(l))
        .collect();
    let fixed = sizes.embedding + sizes.output + sizes.other;
    let total_needed: u64 = layer_cost.iter().sum::<u64>() + fixed;
    let mut batch = req.batch;
    // Layers whose weights end up in RAM (see spill_to_ram).
    let mut ram: Vec<usize> = Vec::new();

    // Which devices to use, and how many layers each takes.
    let assignment: Vec<(i32, usize)> = match &req.split {
        SplitMode::Single(target) => {
            let dev = ordered
                .iter()
                .find(|d| d.ordinal == *target)
                .copied()
                .ok_or_else(|| Shortfall {
                    needed: total_needed,
                    detail: format!("GPU {target} was requested but is not available"),
                })?;
            let budget = budgets.get(&dev.ordinal).copied().unwrap_or(0);
            if req.alloc.spills() && cfg.moe.is_none() && total_needed + scratch_bytes(cfg, batch, true) > budget {
                match spill_to_ram(&[dev], &budgets, &layer_cost, sizes, cfg, batch, &per_layer_state) {
                    Some((a, r)) => {
                        notes.push(format!(
                            "GPU {} holds {} of {n_layers} layers; the other {} stay in RAM (--alloc {})",
                            dev.ordinal, n_layers - r.len(), r.len(), req.alloc.name()));
                        ram = r;
                        a
                    }
                    None => vec![(dev.ordinal, n_layers)],
                }
            } else {
                vec![(dev.ordinal, n_layers)]
            }
        }
        SplitMode::Manual(fractions) => {
            if fractions.len() != ordered.len() {
                return Err(Shortfall {
                    needed: total_needed,
                    detail: format!(
                        "--tensor-split has {} values but {} GPUs are selected",
                        fractions.len(),
                        ordered.len()
                    ),
                });
            }
            let sum: f32 = fractions.iter().sum();
            if sum <= 0.0 {
                return Err(Shortfall {
                    needed: total_needed,
                    detail: "--tensor-split values must add up to more than zero".into(),
                });
            }
            notes.push("layer split taken from --tensor-split".to_string());
            split_by_fraction(ordered, fractions, sum, n_layers)
        }
        SplitMode::Auto => {
            // One GPU is always better when the model fits: no activation ever
            // crosses PCIe, which matters most on consumer cards with no P2P.
            let fits = |d: &Device, b: usize| {
                budgets.get(&d.ordinal).copied().unwrap_or(0) >= total_needed + scratch_bytes(cfg, b, true)
            };
            // The fastest card with a smaller prefill batch beats a slower one
            // at the full batch: the batch only sets how many prompt tokens go
            // through per pass, while the card sets every decode step.
            let fastest = ordered.first().copied();
            let shrunk = fastest.and_then(|d| {
                let mut b = req.batch;
                while b > 128 && !fits(d, b) {
                    b = (b / 2).max(128);
                }
                (b < req.batch && fits(d, b)).then_some((d, b))
            });
            // With a main GPU named, "the fastest" means it.
            let single = if fastest.is_some_and(|d| fits(d, req.batch)) {
                fastest
            } else if req.main_gpu.is_some() {
                None
            } else if let Some((d, b)) = shrunk {
                batch = b;
                notes.push(format!("prefill batch reduced to {b} so the model stays on the fastest GPU"));
                Some(d)
            } else {
                ordered.iter().copied().find(|d| fits(d, req.batch))
            };
            if let Some(d) = single {
                notes.push(format!(
                    "whole model fits on GPU {} ({}), keeping it on one device to avoid \
                     cross-GPU transfers",
                    d.ordinal, d.name
                ));
                vec![(d.ordinal, n_layers)]
            } else {
                let caps: Vec<f32> = ordered
                    .iter()
                    .map(|d| budgets.get(&d.ordinal).copied().unwrap_or(0) as f32)
                    .collect();
                let sum: f32 = caps.iter().sum();
                if sum <= 0.0 {
                    return Err(Shortfall {
                        needed: total_needed,
                        detail: "no GPU has free memory after the reserve".into(),
                    });
                }
                // Fill the fastest GPU first, the next with what is left, and
                // so on. A decode step runs the shards one after another, so a
                // layer on a card with half the bandwidth costs twice as much
                // there; splitting in proportion to free VRAM (as before) put
                // half the 27B on a 5060 Ti that has half a 5070 Ti's
                // bandwidth. A pipelined prefill also wins: its pace is the
                // slower shard's, and the slow card now holds the least.
                match fill_fastest_first(ordered, &budgets, &layer_cost, sizes, cfg, batch,
                    req.alloc != AllocMode::Memory) {
                    Some(a) => {
                        notes.push(format!(
                            "model exceeds any single GPU; filling GPU {} ({}) first",
                            ordered[0].ordinal,
                            if req.main_gpu.is_some() { "--main-gpu" } else { "the fastest" }));
                        a
                    }
                    None if req.alloc.spills() && cfg.moe.is_none() => {
                        match spill_to_ram(ordered, &budgets, &layer_cost, sizes, cfg, batch, &per_layer_state) {
                            Some((a, r)) => {
                                notes.push(format!(
                                    "the GPUs hold {} of {n_layers} layers; the other {} stay in RAM and run on \
                                     GPU {} (--alloc {}; --alloc gpu keeps everything in VRAM)",
                                    n_layers - r.len(), r.len(), ordered[0].ordinal, req.alloc.name()));
                                ram = r;
                                a
                            }
                            None => return Err(Shortfall {
                                needed: total_needed,
                                detail: "even with the weights in RAM, the KV cache does not fit the GPUs".into(),
                            }),
                        }
                    }
                    None => {
                        notes.push(
                            "model exceeds any single GPU; splitting layers in proportion to free VRAM"
                                .to_string(),
                        );
                        split_by_fraction(ordered, &caps, sum, n_layers)
                    }
                }
            }
        }
    };

    // Materialise shards, then verify each against its budget.
    let mut shards = Vec::new();
    let mut next_layer = 0usize;
    let n_shards = assignment.len();

    for (i, (ordinal, count)) in assignment.iter().enumerate() {
        let device = ordered.iter().find(|d| d.ordinal == *ordinal).unwrap();
        let layers: Vec<usize> = (next_layer..next_layer + count).collect();
        next_layer += count;

        // The embedding sits with layer 0 and the output head with the last
        // layer, so neither adds a crossing.
        let hosts_embedding = i == 0;
        let hosts_output = i == n_shards - 1;

        let ram_layers: Vec<usize> = layers.iter().copied().filter(|l| ram.contains(l)).collect();
        let ram_bytes: u64 = ram_layers.iter().map(|&l| sizes.per_layer[l]).sum();
        let weight_bytes: u64 = layers.iter().map(|&l| sizes.per_layer[l]).sum::<u64>() - ram_bytes
            + if hosts_embedding { sizes.embedding } else { 0 }
            + if hosts_output {
                sizes.output + sizes.other
            } else {
                0
            };

        shards.push(Shard {
            device: *ordinal,
            device_name: device.name.clone(),
            layers: layers.clone(),
            weight_bytes,
            kv_bytes: layers.iter().map(|&l| per_layer_state(l)).sum(),
            scratch_bytes: scratch_bytes(cfg, batch, hosts_output),
            budget_bytes: budgets.get(ordinal).copied().unwrap_or(0),
            hosts_embedding,
            hosts_output,
            staging_bytes: staging_for(&ram_layers, sizes),
            ram_layers,
            ram_bytes,
        });
    }

    // A tied output head reads the embedding table, so both must be on the same
    // device. Splitting across GPUs would mean copying the table twice.
    if cfg.tied_embeddings && shards.len() > 1 {
        let embed_dev = shards[0].device;
        let out_dev = shards[shards.len() - 1].device;
        if embed_dev != out_dev {
            let last = shards.len() - 1;
            shards[last].weight_bytes += sizes.embedding;
            notes.push(format!(
                "output projection is tied to the embedding table, so a {} copy is \
                 mirrored onto GPU {out_dev}",
                airun_cuda::human_bytes(sizes.embedding)
            ));
        }
    }

    if let Some(bad) = shards.iter().find(|s| s.headroom() < 0) {
        return Err(Shortfall {
            needed: total_needed + shards.iter().map(|s| s.scratch_bytes).sum::<u64>(),
            detail: format!(
                "GPU {} is short by {} ({} needed, {} budget after reserve)",
                bad.device,
                airun_cuda::human_bytes(bad.headroom().unsigned_abs()),
                airun_cuda::human_bytes(bad.total_bytes()),
                airun_cuda::human_bytes(bad.budget_bytes),
            ),
        });
    }

    // Drop GPUs a proportional split left with nothing to do.
    shards.retain(|s| !s.layers.is_empty() || s.hosts_embedding || s.hosts_output);

    if shards.len() > 1 {
        notes.push(format!(
            "{} activation handoff(s) per forward pass between GPUs",
            shards.len() - 1
        ));
    }

    Ok(Placement { shards, ctx, batch, notes })
}

/// Contiguous layer runs that fill each GPU in `ordered` (fastest first) up
/// to its budget before the next takes any. With `fewest`, a GPU that can
/// hold everything left (with the output head) takes it and the rest are not
/// used: a third card that two already make unnecessary only adds a crossing
/// (`--alloc auto`). Without it every GPU takes at least one layer (`--alloc
/// memory`: the GPUs the user listed). None if the GPUs run out, so the
/// caller can fall back.
fn fill_fastest_first(
    ordered: &[&Device],
    budgets: &BTreeMap<i32, u64>,
    layer_cost: &[u64],
    sizes: &WeightSizes,
    cfg: &Config,
    batch: usize,
    fewest: bool,
) -> Option<Vec<(i32, usize)>> {
    let n_layers = layer_cost.len();
    let mut out = Vec::with_capacity(ordered.len());
    let mut next = 0usize;
    for (i, d) in ordered.iter().enumerate() {
        let budget = budgets.get(&d.ordinal).copied().unwrap_or(0);
        let embed = if i == 0 { sizes.embedding } else { 0 };
        let rest: u64 = layer_cost[next..].iter().sum();
        let last = i + 1 == ordered.len();
        if (fewest || last) && embed + rest + sizes.output + sizes.other + scratch_bytes(cfg, batch, true) <= budget {
            out.push((d.ordinal, n_layers - next));
            return Some(out);
        }
        if last {
            return None;
        }
        let mut used = embed + scratch_bytes(cfg, batch, false);
        let start = next;
        let room = if fewest { n_layers } else { n_layers - (ordered.len() - 1 - i) };
        while next < room && used + layer_cost[next] <= budget {
            used += layer_cost[next];
            next += 1;
        }
        out.push((d.ordinal, next - start));
    }
    None
}

/// VRAM to stage the largest of `ram_layers` (plus room for the per-tensor
/// alignment the loader adds).
fn staging_for(ram_layers: &[usize], sizes: &WeightSizes) -> u64 {
    ram_layers.iter().map(|&l| sizes.per_layer[l] + (64 << 10)).max().unwrap_or(0)
}

/// A layout for a model the GPUs cannot hold: every GPU but the main one
/// (`ordered[0]`) filled with whole layers from the end of the model, the
/// main GPU running the rest, of which those that do not fit keep their
/// weights in RAM. The main GPU runs them because it is filled first and is
/// expected to have the fastest host link (a slow slot reads RAM at a few
/// GB/s). Their KV caches stay in its VRAM, so the context is kept. Returns
/// the layer counts per GPU in order and the layers in RAM, or None when
/// even then the main GPU cannot hold its share.
fn spill_to_ram(
    ordered: &[&Device],
    budgets: &BTreeMap<i32, u64>,
    layer_cost: &[u64],
    sizes: &WeightSizes,
    cfg: &Config,
    batch: usize,
    state: &dyn Fn(usize) -> u64,
) -> Option<(Vec<(i32, usize)>, Vec<usize>)> {
    let n_layers = layer_cost.len();
    let mut counts = vec![0usize; ordered.len()];
    let mut end = n_layers;
    // The other GPUs, last first, take layers from the end.
    for i in (1..ordered.len()).rev() {
        let last = i + 1 == ordered.len();
        let budget = budgets.get(&ordered[i].ordinal).copied().unwrap_or(0);
        let mut used = scratch_bytes(cfg, batch, last) + if last { sizes.output + sizes.other } else { 0 };
        while end > 1 && used + layer_cost[end - 1] <= budget {
            used += layer_cost[end - 1];
            end -= 1;
            counts[i] += 1;
        }
    }
    // The main GPU runs layers 0..end; move weights to RAM from the end of
    // that range until its share fits.
    let only = ordered.len() == 1;
    let budget = budgets.get(&ordered[0].ordinal).copied().unwrap_or(0);
    let fixed = sizes.embedding + scratch_bytes(cfg, batch, only) + if only { sizes.output + sizes.other } else { 0 };
    let states: u64 = (0..end).map(state).sum();
    let mut resident: u64 = (0..end).map(|l| sizes.per_layer[l]).sum();
    let mut ram = Vec::new();
    let mut first_ram = end;
    loop {
        let staging = staging_for(&ram, sizes);
        if fixed + states + resident + staging <= budget {
            break;
        }
        if first_ram == 0 {
            return None;
        }
        first_ram -= 1;
        resident -= sizes.per_layer[first_ram];
        ram.insert(0, first_ram);
    }
    counts[0] = end;
    let mut out: Vec<(i32, usize)> = ordered.iter().zip(&counts).map(|(d, &c)| (d.ordinal, c)).collect();
    out.retain(|&(_, c)| c > 0);
    Some((out, ram))
}

/// Splits `n_layers` into contiguous runs proportional to `weights`.
///
/// Uses largest-remainder so the parts always sum to exactly `n_layers`, which
/// naive rounding does not guarantee.
fn split_by_fraction(
    ordered: &[&Device],
    weights: &[f32],
    sum: f32,
    n_layers: usize,
) -> Vec<(i32, usize)> {
    let exact: Vec<f64> = weights
        .iter()
        .map(|w| (*w as f64 / sum as f64) * n_layers as f64)
        .collect();

    let mut counts: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
    let mut remainder = n_layers - counts.iter().sum::<usize>();

    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = exact[a] - exact[a].floor();
        let fb = exact[b] - exact[b].floor();
        fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in order.iter() {
        if remainder == 0 {
            break;
        }
        counts[i] += 1;
        remainder -= 1;
    }

    ordered
        .iter()
        .zip(counts)
        .map(|(d, c)| (d.ordinal, c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(ordinal: i32, free_gb: u64, sms: i32, bus: i32) -> Device {
        Device {
            ordinal,
            name: format!("GPU{ordinal}"),
            cc_major: 12,
            cc_minor: 0,
            sm_count: sms,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_threads_per_sm: 1536,
            l2_cache_bytes: 0,
            mem_bus_width_bits: bus,
            shared_mem_per_block: 49152,
            shared_mem_per_sm: 102400,
            total_mem: free_gb << 30,
            free_mem: free_gb << 30,
            async_engine_count: 1,
            unified_addressing: true,
            integrated: false,
            pci_domain_id: 0,
            pci_bus_id: ordinal,
            pci_device_id: 0,
            uuid: [0; 16],
        }
    }

    fn config(n_layers: usize) -> Config {
        Config {
            architecture: crate::config::Architecture::Qwen3Dense,
            arch_name: "qwen3".into(),
            name: "test".into(),
            n_layers,
            n_embd: 2048,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 128,
            ffn_len: 6144,
            vocab_size: 151936,
            train_ctx: 40960,
            rms_eps: 1e-6,
            rope: airun_cuda::ops::RopeParams::plain(1e6, 1.0),
            rope_dim: 128,
            sliding_window: 0,
            tied_embeddings: true,
            has_qk_norm: true,
            nextn_predict_layers: 0,
            ssm: None,
            moe: None,
            full_attention_interval: 0,
            attn_output_gate: false,
            ffn_norm_name: "ffn_norm",
            kv_type: Default::default(),
            hc: None,
            ple: None,
            ssm_gate_sigmoid: false,
            indexer: None,
        }
    }

    fn sizes(n_layers: usize, per_layer_mb: u64) -> WeightSizes {
        WeightSizes {
            per_layer: vec![per_layer_mb << 20; n_layers],
            embedding: 600 << 20,
            output: 0,
            other: 1 << 20,
        }
    }

    #[test]
    fn a_model_that_fits_stays_on_one_gpu() {
        let cfg = config(28);
        let sz = sizes(28, 50); // 1.4 GiB of layers
        let devs = vec![device(0, 15, 70, 256), device(1, 15, 36, 128)];
        let p = plan(&cfg, &sz, &devs, &PlacementRequest::default()).unwrap();

        assert_eq!(p.shards.len(), 1, "should not split a model that fits");
        // The faster card wins: more SMs and a wider bus.
        assert_eq!(p.shards[0].device, 0);
        assert_eq!(p.shards[0].layers.len(), 28);
        assert_eq!(p.crossings(), 0);
    }

    /// Two cards with the same memory, one with half the SMs and bus: the
    /// fast one is filled first and the slow one takes only the rest. The
    /// split used to follow free memory, which here meant half and half.
    #[test]
    fn an_oversized_model_fills_the_fastest_gpu_first() {
        let cfg = config(40);
        let sz = sizes(40, 500); // ~19.5 GiB, over any single card
        let devs = vec![device(0, 15, 70, 256), device(1, 15, 36, 128)];
        let p = plan(&cfg, &sz, &devs, &PlacementRequest::default()).unwrap();

        assert_eq!(p.shards.len(), 2);
        assert_eq!(p.shards[0].device, 0, "the fast card holds the first layers");
        assert_eq!(p.shards.iter().map(|s| s.layers.len()).sum::<usize>(), 40);
        assert!(p.shards[0].layers.len() > p.shards[1].layers.len());
        // Full: it fits its budget, and one more layer would not.
        let fast = &p.shards[0];
        let layer = (500u64 << 20) + fast.kv_bytes / fast.layers.len() as u64;
        assert!(fast.total_bytes() <= fast.budget_bytes);
        assert!(fast.budget_bytes - fast.total_bytes() < layer,
            "the fast card has room for another layer: {:?}",
            p.shards.iter().map(|s| s.layers.len()).collect::<Vec<_>>());
    }

    fn layers(p: &Placement) -> Vec<usize> {
        p.shards.iter().map(|s| s.layers.len()).collect()
    }

    /// Two GPUs are enough: the third is left out instead of taking a layer
    /// and adding a crossing.
    #[test]
    fn auto_uses_only_the_gpus_it_needs() {
        let cfg = config(40);
        let sz = sizes(40, 500); // ~19.5 GiB
        let devs = vec![device(0, 15, 70, 256), device(1, 15, 36, 128), device(2, 15, 36, 128)];
        let p = plan(&cfg, &sz, &devs, &PlacementRequest::default()).unwrap();
        assert_eq!(p.shards.len(), 2, "{:?}", layers(&p));
        let memory = PlacementRequest { alloc: AllocMode::Memory, ..Default::default() };
        let p = plan(&cfg, &sz, &devs, &memory).unwrap();
        assert_eq!(p.shards.len(), 3, "--alloc memory uses every listed GPU: {:?}", layers(&p));
    }

    #[test]
    fn the_main_gpu_is_filled_first() {
        let cfg = config(40);
        let sz = sizes(40, 500);
        let devs = vec![device(0, 15, 70, 256), device(1, 15, 36, 128)];
        let req = PlacementRequest { main_gpu: Some(1), ..Default::default() };
        let p = plan(&cfg, &sz, &devs, &req).unwrap();
        assert_eq!(p.shards[0].device, 1);
        assert!(p.shards[0].layers.len() > p.shards[1].layers.len(), "{:?}", layers(&p));
        let bad = PlacementRequest { main_gpu: Some(7), ..Default::default() };
        assert!(plan(&cfg, &sz, &devs, &bad).is_err());
    }

    /// A model bigger than the GPU keeps its context and puts the layers that
    /// do not fit in RAM; --alloc gpu shrinks the context instead.
    #[test]
    fn what_no_gpu_holds_goes_to_ram_unless_gpu_only() {
        let cfg = config(40);
        let sz = sizes(40, 500); // ~19.5 GiB on one 15 GiB card
        let devs = vec![device(0, 15, 70, 256)];
        let req = PlacementRequest { ctx: 8192, ..Default::default() };
        let p = plan(&cfg, &sz, &devs, &req).unwrap();
        assert_eq!(p.ctx, 8192);
        let s = &p.shards[0];
        assert_eq!(s.layers.len(), 40);
        assert!(!s.ram_layers.is_empty() && s.ram_layers.len() < 40);
        // The RAM layers are the last ones, and the GPU still fits.
        assert_eq!(*s.ram_layers.last().unwrap(), 39);
        assert!(s.total_bytes() <= s.budget_bytes);
        assert!(s.staging_bytes >= 500 << 20);

        let gpu_only = PlacementRequest { ctx: 8192, alloc: AllocMode::Gpu, allow_ctx_shrink: false, ..Default::default() };
        assert!(plan(&cfg, &sz, &devs, &gpu_only).is_err(), "--alloc gpu must not spill");
    }

    #[test]
    fn an_oversized_model_splits_proportionally() {
        let cfg = config(40);
        let sz = sizes(40, 500); // ~19.5 GiB, over any single card
        let devs = vec![device(0, 15, 70, 256), device(1, 9, 36, 128)];
        let p = plan(&cfg, &sz, &devs, &PlacementRequest::default()).unwrap();

        assert_eq!(p.shards.len(), 2);
        assert_eq!(p.shards.iter().map(|s| s.layers.len()).sum::<usize>(), 40);
        // The 15 GiB card must take more layers than the 9 GiB one.
        assert!(p.shards[0].layers.len() > p.shards[1].layers.len());
        // Layers stay contiguous and in order.
        assert_eq!(p.shards[0].layers[0], 0);
        assert_eq!(
            p.shards[1].layers[0],
            p.shards[0].layers[p.shards[0].layers.len() - 1] + 1
        );
    }

    /// GPU memory only (--alloc gpu): the context shrinks instead of the
    /// weights going to RAM, which is what the other modes do.
    #[test]
    fn context_shrinks_rather_than_failing() {
        let cfg = config(28);
        // Layers alone nearly fill the card, so the KV cache forces a smaller ctx.
        let sz = sizes(28, 380);
        let devs = vec![device(0, 12, 70, 256)];
        let req = PlacementRequest {
            ctx: 32768,
            allow_ctx_shrink: true,
            alloc: AllocMode::Gpu,
            ..Default::default()
        };
        let p = plan(&cfg, &sz, &devs, &req).unwrap();
        assert!(p.ctx < 32768, "planner should have reduced the context");
        assert!(p.ctx >= 512);
        assert!(p.notes.iter().any(|n| n.contains("retrying at")));
    }

    #[test]
    fn a_model_that_cannot_fit_reports_the_shortfall() {
        let cfg = config(80);
        let sz = sizes(80, 1024); // 80 GiB
        let devs = vec![device(0, 15, 70, 256)];
        let req = PlacementRequest {
            allow_ctx_shrink: false,
            alloc: AllocMode::Gpu,
            ..Default::default()
        };
        let err = plan(&cfg, &sz, &devs, &req).unwrap_err();
        assert!(matches!(err, CoreError::DoesNotFit { .. }));
    }

    #[test]
    fn manual_split_is_respected() {
        let cfg = config(40);
        let sz = sizes(40, 300);
        let devs = vec![device(0, 15, 70, 256), device(1, 15, 36, 128)];
        let req = PlacementRequest {
            split: SplitMode::Manual(vec![0.75, 0.25]),
            ..Default::default()
        };
        let p = plan(&cfg, &sz, &devs, &req).unwrap();
        assert_eq!(p.shards[0].layers.len(), 30);
        assert_eq!(p.shards[1].layers.len(), 10);
    }

    #[test]
    fn largest_remainder_keeps_the_layer_count_exact() {
        let devs = vec![device(0, 10, 1, 1), device(1, 10, 1, 1), device(2, 10, 1, 1)];
        let refs: Vec<&Device> = devs.iter().collect();
        // 10 layers over three equal shares: 3.33 each, must still total 10.
        let split = split_by_fraction(&refs, &[1.0, 1.0, 1.0], 3.0, 10);
        assert_eq!(split.iter().map(|(_, c)| c).sum::<usize>(), 10);
    }
}
