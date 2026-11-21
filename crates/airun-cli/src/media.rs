//! Images in prompts: decoding, the vision encoder's placement, and the
//! prompt's token layout around each image.
//!
//! A prompt marks where each image goes with [`MARKER`] (llama.cpp's
//! `<__media__>`) where airun takes it so (`run --image`, /completion's
//! multimodal_data); images without a marker go before the text, as
//! llama-mtmd-cli puts them. Inside, each place is a per-process [`slot`],
//! so a chat that merely quotes the marker stays text. Each becomes `<|vision_start|>`, the image's
//! sentinel ids (see Runtime::register_image), `<|vision_end|>`.

use std::path::Path;

use airun_core::runtime::Runtime;
use airun_core::vision::{Rgb, VisionConfig, VisionModel};
use anyhow::{Context, Result, bail};

pub const MARKER: &str = "<__media__>";

/// Where an image goes once airun has found it: a marker no text contains by
/// chance, drawn once per process (as llama-server's now is). The public
/// [`MARKER`] stays plain text wherever it was not given as an image's place
/// -- a chat that quotes it (code, a log, airun's own docs) was refused with
/// "the prompt has 1 image markers and 0 images were given".
pub fn slot() -> &'static str {
    static SLOT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SLOT.get_or_init(|| {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
        format!("<__airun_image_{:016x}__>", h.finish())
    })
}

/// `text` with each public [`MARKER`] made an image's place, where the caller
/// takes MARKER to mean one (`run --image`, /completion's multimodal_data).
pub fn mark_slots(text: &str) -> String {
    text.replace(MARKER, slot())
}

/// Decodes a PNG, JPEG or WebP image to 8-bit RGB.
pub fn decode_image(bytes: &[u8]) -> Result<Rgb> {
    let img = image::load_from_memory(bytes)
        .context("the image could not be decoded; send a PNG, JPEG or WebP file")?
        .to_rgb8();
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        bail!("the image is empty (0 pixels wide or tall)");
    }
    Ok(Rgb { width: w as usize, height: h as usize, pixels: img.into_raw() })
}

pub fn read_image(path: &Path) -> Result<Rgb> {
    let bytes = std::fs::read(path).with_context(|| format!("could not read the image {}", path.display()))?;
    decode_image(&bytes).with_context(|| path.display().to_string())
}

/// Planned on top of the encoder's own VRAM (see vision_reserve).
const VISION_MARGIN: u64 = 320 << 20;

/// VRAM the encoder takes on the GPU it runs on: its weights plus the
/// scratch of the largest image it accepts. Added to every GPU's reserve
/// before the text model is placed, so the automatic context leaves room.
pub fn vision_reserve(mmproj: &Path, max_tokens: usize) -> Result<(u64, u64)> {
    let g = airun_gguf::GgufFile::open(mmproj)
        .with_context(|| format!("could not open the mmproj file {}", mmproj.display()))?;
    let mut cfg = VisionConfig::from_gguf(mmproj, &g)?;
    cfg.max_tokens = max_tokens;
    let scratch = VisionModel::scratch_bytes(&cfg, max_tokens);
    // + the FFN padding, + room for what loading takes past the plan (the
    // CUDA context and kernel modules: the text model's grow into the
    // default reserve and left the encoder ~250 MiB short on a 12 GB card).
    let need = |streamed: bool| VisionModel::weights_vram(&g, streamed) * 21 / 20 + scratch + VISION_MARGIN;
    Ok((need(false), need(true)))
}

/// Loads the encoder on the GPU with the most free memory after the text
/// model, checking the text model can take its tokens.
pub fn load_vision(mmproj: &Path, runtime: &Runtime, max_tokens: usize, streamed: bool) -> Result<VisionModel> {
    if !runtime.takes_images() {
        bail!("{} cannot take images: its RoPE has no M-RoPE sections (rope.dimension_sections). Use --mmproj \
               only with a Qwen3.5/3.8 model whose mmproj it is", runtime.model.config.arch_name);
    }
    let mut best: Option<(i32, u64)> = None;
    for d in runtime.model.placement.devices() {
        if let Ok((free, _)) = airun_cuda::device::mem_info(d) {
            if best.is_none_or(|(_, f)| free > f) {
                best = Some((d, free));
            }
        }
    }
    let (dev, free) = best.context("no GPU to run the vision encoder on")?;
    let (resident, stream) = vision_reserve(mmproj, max_tokens)?;
    let (resident, stream) = (resident - VISION_MARGIN, stream - VISION_MARGIN);
    // The plan kept room for the encoder, but on Windows (WDDM) the free
    // memory CUDA reports before loading can run past what is really there
    // (other programs' memory counts as free), so the text model may have
    // taken that room. Uploading the encoder's blocks per image needs much
    // less. Past what is free, an allocation can still succeed, but Windows
    // then moves part of the model to RAM and generation runs ~3x slower,
    // so the reported number decides.
    let streamed = streamed || (free < resident && stream < resident);
    let need = if streamed { stream } else { resident };
    if free < need {
        bail!("the vision encoder needs {} on GPU {dev} and {} is free after the model: lower --ctx-size \
               (or --expert-cache for MoE) to leave it room, or close programs that use the GPU (games, \
               browsers, Discord, overlays)", airun_cuda::human_bytes(need), airun_cuda::human_bytes(free));
    }
    let mut v = VisionModel::load(mmproj, dev, streamed)
        .with_context(|| format!("failed to load the vision encoder {}", mmproj.display()))?;
    v.cfg.max_tokens = max_tokens;
    if v.cfg.proj_dim != runtime.model.config.n_embd {
        bail!("{} projects images to {} values and the model is {} wide: it is another model's mmproj",
            mmproj.display(), v.cfg.proj_dim, runtime.model.config.n_embd);
    }
    Ok(v)
}

/// Encodes `img` and registers it with the runtime; returns its ids.
pub fn encode_image(vision: &VisionModel, runtime: &mut Runtime, img: &Rgb) -> Result<Vec<u32>> {
    let e = vision.encode(img).context("the vision encoder failed")?;
    Ok(runtime.register_image(e.embeds, e.nx, e.ny)?)
}

/// The prompt's tokens with each image place ([`slot`]) replaced by the next
/// image. Any [`MARKER`] left in the text is text.
pub fn tokenize(tk: &airun_core::tokenizer::Tokenizer, text: &str, images: &[Vec<u32>], add_special: bool) -> Result<Vec<u32>> {
    let parts: Vec<&str> = text.split(slot()).collect();
    if parts.len() - 1 != images.len() {
        bail!("the prompt has {} places for images ({MARKER}) and {} images were given; send one image per                {MARKER}, or leave the marker out and the images go before the text", parts.len() - 1, images.len());
    }
    let id = |s: &str| tk.id_of(s).with_context(|| format!("the model's vocabulary has no {s} token"));
    let (start, end) = if images.is_empty() { (0, 0) } else { (id("<|vision_start|>")?, id("<|vision_end|>")?) };
    let mut out = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        // Only the first piece carries the BOS the tokenizer may add.
        out.extend(tk.encode(part, add_special && i == 0));
        if let Some(ids) = images.get(i) {
            out.push(start);
            out.extend_from_slice(ids);
            out.push(end);
        }
    }
    Ok(out)
}
