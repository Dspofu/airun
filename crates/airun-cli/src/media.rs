//! Images in prompts: decoding, the vision encoder's placement, and the
//! prompt's token layout around each image.
//!
//! A prompt marks where each image goes with [`MARKER`] (llama.cpp's
//! `<__media__>`); images without a marker go before the text, as
//! llama-mtmd-cli puts them. Each becomes `<|vision_start|>`, the image's
//! sentinel ids (see Runtime::register_image), `<|vision_end|>`.

use std::path::Path;

use airun_core::runtime::Runtime;
use airun_core::vision::{Rgb, VisionConfig, VisionModel};
use anyhow::{Context, Result, bail};

pub const MARKER: &str = "<__media__>";

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

/// VRAM the encoder takes on the GPU it runs on: its weights plus the
/// scratch of the largest image it accepts. Added to every GPU's reserve
/// before the text model is placed, so the automatic context leaves room.
pub fn vision_reserve(mmproj: &Path, max_tokens: usize) -> Result<u64> {
    let g = airun_gguf::GgufFile::open(mmproj)
        .with_context(|| format!("could not open the mmproj file {}", mmproj.display()))?;
    let mut cfg = VisionConfig::from_gguf(mmproj, &g)?;
    cfg.max_tokens = max_tokens;
    let weights: u64 = g.tensors().iter().map(|t| t.nbytes).sum::<u64>() * 21 / 20; // + the FFN padding
    Ok(weights + VisionModel::scratch_bytes(&cfg, max_tokens))
}

/// Loads the encoder on the GPU with the most free memory after the text
/// model, checking the text model can take its tokens.
pub fn load_vision(mmproj: &Path, runtime: &Runtime, max_tokens: usize) -> Result<VisionModel> {
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
    let need = vision_reserve(mmproj, max_tokens)?;
    if free < need {
        bail!("the vision encoder needs {} on a GPU and the most free is {} (GPU {dev}); lower --ctx-size or \
               --image-max-tokens", airun_cuda::human_bytes(need), airun_cuda::human_bytes(free));
    }
    let mut v = VisionModel::load(mmproj, dev)
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

/// The prompt's tokens with each [`MARKER`] replaced by the next image.
pub fn tokenize(tk: &airun_core::tokenizer::Tokenizer, text: &str, images: &[Vec<u32>], add_special: bool) -> Result<Vec<u32>> {
    let parts: Vec<&str> = text.split(MARKER).collect();
    if parts.len() - 1 != images.len() {
        bail!("the prompt has {} image markers ({MARKER}) and {} images were given", parts.len() - 1, images.len());
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
