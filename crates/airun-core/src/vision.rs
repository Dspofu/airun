//! The vision encoder behind an mmproj file: a Qwen3-VL-style ViT
//! (`clip.projector_type = qwen3vl_merger`), as llama.cpp's
//! tools/mtmd/models/qwen3vl.cpp builds it.
//!
//! An image is resized so both sides are multiples of patch * merge (32),
//! within the token budget, cut into 16 x 16 patches taken in 2 x 2 merge
//! windows (so each window's four patches are consecutive), embedded,
//! offset by the learned position grid (bilinearly resized), and run through
//! the ViT blocks: LayerNorm, fused QKV with a 2D rotary embedding (row
//! angles on the first quarter pairs, column angles on the next), full
//! attention, LayerNorm, GELU FFN. A final LayerNorm, then the merger turns
//! each window's four patches (4 x 1152 values) into one token of the text
//! model's width through a two-layer GELU MLP.
//!
//! Everything runs on one GPU with the text model's kernels (BF16 matmuls,
//! the KV-cache attention with every query at the last position, GELU), plus
//! airun_cuda::vision for what they lack.

use std::path::Path;

use airun_cuda::{DeviceBuffer, Stream, device, ops, vision};
use airun_gguf::{GgmlType, GgufFile};

use crate::error::{CoreError, Result};

/// Encoder dimensions and preprocessing, from the mmproj metadata.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    pub patch: usize,
    pub merge: usize,
    /// Width of the tokens it hands the text model.
    pub proj_dim: usize,
    pub eps: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Side of the learned position grid.
    pub pos_side: usize,
    /// Merged tokens an image may take (llama.cpp: 8 .. 4096 for Qwen-VL).
    pub min_tokens: usize,
    pub max_tokens: usize,
}

/// The FFN width rounded up to what the tensor-core matmul takes (k % 64).
fn padded(n: usize) -> usize {
    n.div_ceil(64) * 64
}

struct Weight {
    data: DeviceBuffer<u8>,
    ty: ops::KernelType,
    rows: usize,
    k: usize,
    row_bytes: usize,
}

struct Layer {
    ln1_w: DeviceBuffer<f32>,
    ln1_b: DeviceBuffer<f32>,
    qkv: Weight,
    qkv_b: DeviceBuffer<f32>,
    out: Weight,
    out_b: DeviceBuffer<f32>,
    ln2_w: DeviceBuffer<f32>,
    ln2_b: DeviceBuffer<f32>,
    up: Weight,
    up_b: DeviceBuffer<f32>,
    down: Weight,
    down_b: DeviceBuffer<f32>,
}

/// Where the encoder's blocks (nearly all of its weights) are.
enum Layers {
    /// In VRAM, loaded once.
    Resident(Vec<Layer>),
    /// Left in the mmproj file and uploaded one at a time for each image
    /// (see [`VisionModel::load`]): the file's mapping, and its path for
    /// messages.
    Streamed(GgufFile, std::path::PathBuf),
}

pub struct VisionModel {
    pub cfg: VisionConfig,
    pub device: i32,
    stream: Stream,
    patch_w: Weight,
    patch_b: DeviceBuffer<f32>,
    /// [pos_side * pos_side][n_embd], row-major over the grid.
    pos_embd: Vec<f32>,
    layers: Layers,
    post_w: DeviceBuffer<f32>,
    post_b: DeviceBuffer<f32>,
    mm0: Weight,
    mm0_b: DeviceBuffer<f32>,
    mm2: Weight,
    mm2_b: DeviceBuffer<f32>,
    weight_bytes: u64,
}

/// An RGB image, 8 bits per channel, row-major.
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// What an image became: `nx * ny` tokens of the text model's width,
/// row-major over the merged grid.
pub struct Encoded {
    pub embeds: Vec<f32>,
    pub nx: usize,
    pub ny: usize,
}

fn bad(path: &Path, why: impl std::fmt::Display) -> CoreError {
    CoreError::Other(format!("{}: {why}", path.display()))
}

impl VisionConfig {
    pub fn from_gguf(path: &Path, g: &GgufFile) -> Result<Self> {
        let s = |k: &str| g.get(k).and_then(|v| v.as_str()).map(str::to_string);
        if s("general.architecture").as_deref() != Some("clip") {
            return Err(bad(path, "not an mmproj file (general.architecture is not \"clip\"); pass the model's \
                                  mmproj-*.gguf with --mmproj"));
        }
        let proj = s("clip.projector_type").unwrap_or_default();
        if proj != "qwen3vl_merger" {
            return Err(bad(path, format!(
                "its projector is \"{proj}\"; airun runs the Qwen3-VL / Qwen3.5 vision encoder (qwen3vl_merger) only")));
        }
        let u = |k: &str| g.get(k).and_then(|v| v.as_usize())
            .ok_or_else(|| bad(path, format!("the metadata has no {k}")));
        let f3 = |k: &str, d: [f32; 3]| g.get(k).and_then(|v| v.as_array()).and_then(|a| a.as_f32_vec())
            .and_then(|v| v.try_into().ok()).unwrap_or(d);
        if g.tensor("v.deepstack.0.fc1.weight").is_some()
            || g.tensors().iter().any(|t| t.name.contains("deepstack")) {
            return Err(bad(path, "it has deepstack layers (Qwen3-VL), which airun does not run yet; Qwen3.5/3.8 \
                                  mmproj files have none"));
        }
        let pos = g.tensor("v.position_embd.weight").ok_or_else(|| bad(path, "it has no v.position_embd.weight"))?;
        let n_pos = pos.dims.get(1).copied().unwrap_or(0) as usize;
        let pos_side = (n_pos as f64).sqrt() as usize;
        if pos_side * pos_side != n_pos {
            return Err(bad(path, format!("its position grid has {n_pos} entries, not a square")));
        }
        let cfg = Self {
            n_embd: u("clip.vision.embedding_length")?,
            n_head: u("clip.vision.attention.head_count")?,
            n_layer: u("clip.vision.block_count")?,
            n_ff: u("clip.vision.feed_forward_length")?,
            patch: u("clip.vision.patch_size")?,
            merge: g.get("clip.vision.spatial_merge_size").and_then(|v| v.as_usize()).unwrap_or(2),
            proj_dim: u("clip.vision.projection_dim")?,
            eps: g.get("clip.vision.attention.layer_norm_epsilon").and_then(|v| v.as_f32()).unwrap_or(1e-6),
            mean: f3("clip.vision.image_mean", [0.5; 3]),
            std: f3("clip.vision.image_std", [0.5; 3]),
            pos_side,
            min_tokens: 8,
            max_tokens: 4096,
        };
        if cfg.merge != 2 || cfg.n_embd % cfg.n_head != 0 || (cfg.n_embd / cfg.n_head) % 4 != 0 {
            return Err(bad(path, "its shape (merge size, heads) is not the Qwen-VL one airun implements"));
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// The size an image is resized to: llama.cpp's (and transformers')
    /// smart_resize, both sides multiples of patch * merge.
    pub fn target_size(&self, width: usize, height: usize) -> (usize, usize) {
        let align = (self.patch * self.merge) as f64;
        let area = align * align;
        let (min_px, max_px) = (self.min_tokens as f64 * area, self.max_tokens as f64 * area);
        let (w, h) = (width as f64, height as f64);
        let round = |x: f64| ((x / align).round() * align) as usize;
        let floor = |x: f64| ((x / align).floor() * align) as usize;
        let ceil = |x: f64| ((x / align).ceil() * align) as usize;
        let a = align as usize;
        let mut hb = round(h).max(a);
        let mut wb = round(w).max(a);
        if (hb * wb) as f64 > max_px {
            let beta = ((h * w) / max_px).sqrt();
            hb = floor(h / beta).max(a);
            wb = floor(w / beta).max(a);
        } else if ((hb * wb) as f64) < min_px {
            let beta = (min_px / (h * w)).sqrt();
            hb = ceil(h * beta);
            wb = ceil(w * beta);
        }
        (wb, hb)
    }
}

/// llama.cpp's bilinear resize (mtmd-image.cpp resize_bilinear), corners
/// aligned, truncating to 8 bits.
pub fn resize_bilinear(src: &Rgb, tw: usize, th: usize) -> Rgb {
    let (sw, sh) = (src.width, src.height);
    let xr = if tw > 1 { (sw - 1) as f32 / (tw - 1) as f32 } else { 0.0 };
    let yr = if th > 1 { (sh - 1) as f32 / (th - 1) as f32 } else { 0.0 };
    let mut out = vec![0u8; tw * th * 3];
    let px = |x: usize, y: usize, c: usize| src.pixels[(y * sw + x) * 3 + c] as f32;
    for y in 0..th {
        let fy = y as f32 * yr;
        let y0 = (fy as usize).min(sh - 1);
        let y1 = (y0 + 1).min(sh - 1);
        let yf = fy - y0 as f32;
        for x in 0..tw {
            let fx = x as f32 * xr;
            let x0 = (fx as usize).min(sw - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let xf = fx - x0 as f32;
            for c in 0..3 {
                let top = px(x0, y0, c) + (px(x1, y0, c) - px(x0, y0, c)) * xf;
                let bot = px(x0, y1, c) + (px(x1, y1, c) - px(x0, y1, c)) * xf;
                out[(y * tw + x) * 3 + c] = (top + (bot - top) * yf) as u8;
            }
        }
    }
    Rgb { width: tw, height: th, pixels: out }
}

impl VisionModel {
    /// Loads the encoder onto `device`. With `streamed`, its blocks stay in
    /// the file and each image uploads them one at a time, so the encoder
    /// holds one block's VRAM, not all 27: for a text model whose weights do
    /// not fit the GPUs, where every GiB it keeps free is a GiB of weights
    /// off the CPU on every token.
    pub fn load(path: &Path, device: i32, streamed: bool) -> Result<Self> {
        let g = GgufFile::open(path).map_err(|e| bad(path, format!("{e}; pass the model's mmproj-*.gguf")))?;
        let cfg = VisionConfig::from_gguf(path, &g)?;
        device::set_device(device)?;
        let stream = Stream::new(device)?;
        let mut bytes = 0u64;
        let up = Uploader { g: &g, path, device, stream: &stream };
        let (f32s, vec_dev, matrix) = (
            |name: &str| up.f32s(name),
            |v: &[f32], bytes: &mut u64| up.vec_dev(v, bytes),
            |name: &str, rows_to: Option<usize>, k_to: Option<usize>, bytes: &mut u64| up.matrix(name, rows_to, k_to, bytes),
        );

        let mut layers = Vec::new();
        if streamed {
            // Shapes and types are checked now, on the first block.
            let mut probe = 0u64;
            let first = up.layer(&cfg, 0, &mut probe)?;
            stream.synchronize()?;
            drop(first);
            bytes += probe;
        } else {
            for l in 0..cfg.n_layer {
                layers.push(up.layer(&cfg, l, &mut bytes)?);
            }
        }

        // The two temporal patch kernels see the same still image: their sum
        // is the one convolution. As a [n_embd][3 * patch * patch] F32 matrix,
        // input ordered (channel, row, col) like the ggml kernel.
        let mut pw = f32s("v.patch_embd.weight")?;
        if g.tensor("v.patch_embd.weight.1").is_some() {
            for (a, b) in pw.iter_mut().zip(f32s("v.patch_embd.weight.1")?) {
                *a += b;
            }
        }
        let kp = 3 * cfg.patch * cfg.patch;
        bytes += pw.len() as u64 * 4;
        let patch_w = Weight {
            data: DeviceBuffer::from_slice(device, bytemuck_f32(&pw), &stream)?,
            ty: ops::KernelType(GgmlType::F32.id() as i32),
            rows: cfg.n_embd,
            k: kp,
            row_bytes: kp * 4,
        };
        let model = Self {
            patch_b: vec_dev(&f32s("v.patch_embd.bias")?, &mut bytes)?,
            pos_embd: f32s("v.position_embd.weight")?,
            post_w: vec_dev(&f32s("v.post_ln.weight")?, &mut bytes)?,
            post_b: vec_dev(&f32s("v.post_ln.bias")?, &mut bytes)?,
            mm0: matrix("mm.0.weight", None, None, &mut bytes)?,
            mm0_b: vec_dev(&f32s("mm.0.bias")?, &mut bytes)?,
            mm2: matrix("mm.2.weight", None, None, &mut bytes)?,
            mm2_b: vec_dev(&f32s("mm.2.bias")?, &mut bytes)?,
            patch_w,
            layers: Layers::Resident(layers),
            weight_bytes: bytes,
            cfg,
            device,
            stream,
        };
        model.stream.synchronize()?;
        if model.mm2.rows != model.cfg.proj_dim {
            return Err(bad(path, format!("its projection is {} wide, the metadata says {}",
                model.mm2.rows, model.cfg.proj_dim)));
        }
        let mut model = model;
        if streamed {
            model.layers = Layers::Streamed(g, path.to_path_buf());
        }
        Ok(model)
    }

    /// Whether the blocks are uploaded for each image (see [`Self::load`]).
    pub fn streamed(&self) -> bool {
        matches!(self.layers, Layers::Streamed(..))
    }

    /// VRAM the encoder's weights take: all of them resident, or with the
    /// blocks streamed the rest plus the largest block.
    pub fn weights_vram(g: &GgufFile, streamed: bool) -> u64 {
        let all: u64 = g.tensors().iter().map(|t| t.nbytes).sum();
        if !streamed {
            return all;
        }
        let mut per_block = std::collections::HashMap::<&str, u64>::new();
        for t in g.tensors() {
            if let Some(rest) = t.name.strip_prefix("v.blk.") {
                *per_block.entry(rest.split('.').next().unwrap_or("")).or_default() += t.nbytes;
            }
        }
        let blocks: u64 = per_block.values().sum();
        all - blocks + per_block.values().copied().max().unwrap_or(0)
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Scratch VRAM an image of `tokens` merged tokens needs.
    pub fn scratch_bytes(cfg: &VisionConfig, tokens: usize) -> u64 {
        let n = (tokens * cfg.merge * cfg.merge) as u64;
        let hdp = cfg.head_dim().div_ceil(32) as u64 * 32;
        let h = cfg.n_head as u64;
        let floats = n * (3 * cfg.n_embd as u64 + 3 * cfg.n_embd as u64 + padded(cfg.n_ff) as u64
            + 2 * h * hdp + 3 * cfg.patch as u64 * cfg.patch as u64)
            + n / 4 * (4 * cfg.n_embd as u64 + cfg.proj_dim as u64);
        floats * 4 + n * h * hdp * 2 * 2 + (64 << 20)
    }

    /// Encodes one image into text-model tokens.
    pub fn encode(&self, img: &Rgb) -> Result<Encoded> {
        let c = &self.cfg;
        let (tw, th) = c.target_size(img.width, img.height);
        let img = if (tw, th) == (img.width, img.height) {
            Rgb { width: tw, height: th, pixels: img.pixels.clone() }
        } else {
            resize_bilinear(img, tw, th)
        };
        let (pw, ph) = (tw / c.patch, th / c.patch);
        let n = pw * ph;
        let e = c.n_embd;
        let kp = 3 * c.patch * c.patch;

        // Patches and positions in merge-window order.
        let mut patches = vec![0f32; n * kp];
        let mut pos_y = Vec::with_capacity(n);
        let mut pos_x = Vec::with_capacity(n);
        let mut order = Vec::with_capacity(n); // grid index of each patch
        for by in (0..ph).step_by(2) {
            for bx in (0..pw).step_by(2) {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (y, x) = (by + dy, bx + dx);
                        let row = pos_y.len();
                        for ch in 0..3 {
                            for ky in 0..c.patch {
                                for kx in 0..c.patch {
                                    let p = img.pixels[((y * c.patch + ky) * tw + x * c.patch + kx) * 3 + ch];
                                    patches[row * kp + (ch * c.patch + ky) * c.patch + kx] =
                                        (p as f32 / 255.0 - c.mean[ch]) / c.std[ch];
                                }
                            }
                        }
                        pos_y.push(y as i32);
                        pos_x.push(x as i32);
                        order.push(y * pw + x);
                    }
                }
            }
        }
        // The learned grid resized to (ph, pw), corners aligned, then taken
        // in the same order.
        let s = c.pos_side;
        let grid = |gy: usize, gx: usize, d: usize| self.pos_embd[(gy * s + gx) * e + d];
        let mut pos = vec![0f32; n * e];
        for (row, &cell) in order.iter().enumerate() {
            let (y, x) = (cell / pw, cell % pw);
            let fy = if ph > 1 { y as f32 * (s - 1) as f32 / (ph - 1) as f32 } else { 0.0 };
            let fx = if pw > 1 { x as f32 * (s - 1) as f32 / (pw - 1) as f32 } else { 0.0 };
            let (y0, x0) = ((fy as usize).min(s - 1), (fx as usize).min(s - 1));
            let (y1, x1) = ((y0 + 1).min(s - 1), (x0 + 1).min(s - 1));
            let (wy, wx) = (fy - y0 as f32, fx - x0 as f32);
            for d in 0..e {
                let top = grid(y0, x0, d) * (1.0 - wx) + grid(y0, x1, d) * wx;
                let bot = grid(y1, x0, d) * (1.0 - wx) + grid(y1, x1, d) * wx;
                pos[row * e + d] = top * (1.0 - wy) + bot * wy;
            }
        }

        device::set_device(self.device)?;
        let st = &self.stream;
        let dev = self.device;
        let hd = c.head_dim();
        let hdp = hd.div_ceil(32) * 32;
        let ffp = padded(c.n_ff);
        let x = DeviceBuffer::<f32>::from_slice(dev, &pos, st)?;      // residual stream
        let xb = DeviceBuffer::<f32>::new(dev, n * e)?;
        let big = DeviceBuffer::<f32>::new(dev, n * (3 * e).max(ffp))?;
        let attn = DeviceBuffer::<f32>::new(dev, n * c.n_head * hdp)?;
        let q = DeviceBuffer::<f32>::new(dev, n * c.n_head * hdp)?;
        let kc = DeviceBuffer::<u16>::new(dev, n * c.n_head * hdp)?;
        let vc = DeviceBuffer::<u16>::new(dev, n * c.n_head * hdp)?;
        let py = DeviceBuffer::<i32>::from_slice(dev, &pos_y, st)?;
        let px = DeviceBuffer::<i32>::from_slice(dev, &pos_x, st)?;
        let last = DeviceBuffer::<i32>::from_slice(dev, &vec![n as i32 - 1; n], st)?;
        {
            let pin = DeviceBuffer::<f32>::from_slice(dev, &patches, st)?;
            self.mm(&self.patch_w, pin.as_ptr(), xb.as_mut_ptr(), n)?;
            vision::add_bias(xb.as_mut_ptr(), self.patch_b.as_ptr(), n, e, st)?;
            ops::add(x.as_mut_ptr(), xb.as_ptr(), n * e, st)?;
            st.synchronize()?; // `pin` goes out of scope next
        }
        let scale = 1.0 / (hd as f32).sqrt();
        for li in 0..c.n_layer {
            // A streamed block is uploaded now and freed at the end of its turn.
            let uploaded;
            let l = match &self.layers {
                Layers::Resident(v) => &v[li],
                Layers::Streamed(g, path) => {
                    let up = Uploader { g, path, device: dev, stream: st };
                    uploaded = up.layer(c, li, &mut 0)?;
                    &uploaded
                }
            };
            vision::layernorm(xb.as_mut_ptr(), x.as_ptr(), l.ln1_w.as_ptr(), l.ln1_b.as_ptr(), n, e, c.eps, st)?;
            self.mm(&l.qkv, xb.as_ptr(), big.as_mut_ptr(), n)?;
            vision::add_bias(big.as_mut_ptr(), l.qkv_b.as_ptr(), n, 3 * e, st)?;
            vision::vision_qkv(q.as_mut_ptr(), kc.as_mut_ptr(), vc.as_mut_ptr(), big.as_ptr(), py.as_ptr(),
                px.as_ptr(), n, c.n_head, hd, hdp, 10_000.0, st)?;
            ops::attention(attn.as_mut_ptr(), q.as_ptr(), kc.as_ptr(), vc.as_ptr(), last.as_ptr(), n, c.n_head,
                c.n_head, hdp, n, scale, 0, ops::KvType::F16, st)?;
            vision::vision_unpad(q.as_mut_ptr(), attn.as_ptr(), n * c.n_head, hd, hdp, st)?;
            self.mm(&l.out, q.as_ptr(), xb.as_mut_ptr(), n)?;
            vision::add_bias(xb.as_mut_ptr(), l.out_b.as_ptr(), n, e, st)?;
            ops::add(x.as_mut_ptr(), xb.as_ptr(), n * e, st)?;

            vision::layernorm(xb.as_mut_ptr(), x.as_ptr(), l.ln2_w.as_ptr(), l.ln2_b.as_ptr(), n, e, c.eps, st)?;
            self.mm(&l.up, xb.as_ptr(), big.as_mut_ptr(), n)?;
            vision::add_bias(big.as_mut_ptr(), l.up_b.as_ptr(), n, ffp, st)?;
            ops::gelu(big.as_mut_ptr(), n * ffp, st)?;
            self.mm(&l.down, big.as_ptr(), xb.as_mut_ptr(), n)?;
            vision::add_bias(xb.as_mut_ptr(), l.down_b.as_ptr(), n, e, st)?;
            ops::add(x.as_mut_ptr(), xb.as_ptr(), n * e, st)?;
            if self.streamed() {
                st.synchronize()?; // the block's weights are freed next
            }
        }
        vision::layernorm(xb.as_mut_ptr(), x.as_ptr(), self.post_w.as_ptr(), self.post_b.as_ptr(), n, e, c.eps, st)?;
        // Four consecutive patches are one merge window: [n/4][4e] as laid out.
        let m = n / 4;
        self.mm(&self.mm0, xb.as_ptr(), big.as_mut_ptr(), m)?;
        vision::add_bias(big.as_mut_ptr(), self.mm0_b.as_ptr(), m, self.mm0.rows, st)?;
        ops::gelu(big.as_mut_ptr(), m * self.mm0.rows, st)?;
        let out = DeviceBuffer::<f32>::new(dev, m * c.proj_dim)?;
        self.mm(&self.mm2, big.as_ptr(), out.as_mut_ptr(), m)?;
        vision::add_bias(out.as_mut_ptr(), self.mm2_b.as_ptr(), m, c.proj_dim, st)?;
        let mut embeds = vec![0f32; m * c.proj_dim];
        out.copy_to_host(&mut embeds, st)?;
        st.synchronize()?;
        Ok(Encoded { embeds, nx: pw / 2, ny: ph / 2 })
    }

    fn mm(&self, w: &Weight, x: *const std::ffi::c_void, y: *mut std::ffi::c_void, n: usize) -> Result<()> {
        ops::matvec(w.ty, w.data.as_ptr(), x, y, w.k, w.rows, n, w.row_bytes, &self.stream)?;
        Ok(())
    }
}

/// Reads an mmproj file's tensors into VRAM.
struct Uploader<'a> {
    g: &'a GgufFile,
    path: &'a Path,
    device: i32,
    stream: &'a Stream,
}

impl Uploader<'_> {
    fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.g.tensor(name).ok_or_else(|| bad(self.path, format!("it has no {name}")))?;
        let n = t.dims.iter().product::<u64>() as usize;
        let mut out = vec![0f32; n];
        airun_gguf::dequant::dequantize(t.ty, self.g.tensor_bytes(t)?, n, &mut out)?;
        Ok(out)
    }

    fn vec_dev(&self, v: &[f32], bytes: &mut u64) -> Result<DeviceBuffer<f32>> {
        *bytes += v.len() as u64 * 4;
        Ok(DeviceBuffer::from_slice(self.device, v, self.stream)?)
    }

    fn vec_pad(&self, name: &str, to: usize, bytes: &mut u64) -> Result<DeviceBuffer<f32>> {
        let mut v = self.f32s(name)?;
        v.resize(to, 0.0);
        self.vec_dev(&v, bytes)
    }

    /// A matrix as stored (BF16/F16/F32/quantized), its rows padded to
    /// `rows_to` and each row's k to `k_to` with zeros (BF16/F16/F32 only).
    fn matrix(&self, name: &str, rows_to: Option<usize>, k_to: Option<usize>, bytes: &mut u64) -> Result<Weight> {
        let path = self.path;
        let t = self.g.tensor(name).ok_or_else(|| bad(path, format!("it has no {name}")))?;
        let (k, rows) = (t.dims[0] as usize, t.dims.get(1).copied().unwrap_or(1) as usize);
        let raw = self.g.tensor_bytes(t)?;
        let (rows_to, k_to) = (rows_to.unwrap_or(rows), k_to.unwrap_or(k));
        let padded;
        let (data, row_bytes): (&[u8], usize) = if rows_to == rows && k_to == k {
            (raw, raw.len() / rows)
        } else {
            let el = match t.ty {
                GgmlType::BF16 | GgmlType::F16 => 2,
                GgmlType::F32 => 4,
                other => return Err(bad(path, format!("{name} is {other}; padding it needs a float type"))),
            };
            let (src_rb, dst_rb) = (k * el, k_to * el);
            let mut out = vec![0u8; rows_to * dst_rb];
            for r in 0..rows {
                out[r * dst_rb..r * dst_rb + src_rb].copy_from_slice(&raw[r * src_rb..(r + 1) * src_rb]);
            }
            padded = out;
            (&padded, dst_rb)
        };
        *bytes += data.len() as u64;
        Ok(Weight {
            data: DeviceBuffer::from_slice(self.device, data, self.stream)?,
            ty: ops::KernelType(t.ty.id() as i32),
            rows: rows_to,
            k: k_to,
            row_bytes,
        })
    }

    /// Block `l`'s weights.
    fn layer(&self, cfg: &VisionConfig, l: usize, bytes: &mut u64) -> Result<Layer> {
        let ffp = padded(cfg.n_ff);
        let n = |s: &str| format!("v.blk.{l}.{s}");
        Ok(Layer {
            ln1_w: self.vec_dev(&self.f32s(&n("ln1.weight"))?, bytes)?,
            ln1_b: self.vec_dev(&self.f32s(&n("ln1.bias"))?, bytes)?,
            qkv: self.matrix(&n("attn_qkv.weight"), None, None, bytes)?,
            qkv_b: self.vec_dev(&self.f32s(&n("attn_qkv.bias"))?, bytes)?,
            out: self.matrix(&n("attn_out.weight"), None, None, bytes)?,
            out_b: self.vec_dev(&self.f32s(&n("attn_out.bias"))?, bytes)?,
            ln2_w: self.vec_dev(&self.f32s(&n("ln2.weight"))?, bytes)?,
            ln2_b: self.vec_dev(&self.f32s(&n("ln2.bias"))?, bytes)?,
            up: self.matrix(&n("ffn_up.weight"), Some(ffp), None, bytes)?,
            up_b: self.vec_pad(&n("ffn_up.bias"), ffp, bytes)?,
            down: self.matrix(&n("ffn_down.weight"), None, Some(ffp), bytes)?,
            down_b: self.vec_dev(&self.f32s(&n("ffn_down.bias"))?, bytes)?,
        })
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte pattern is a valid u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        VisionConfig { n_embd: 1152, n_head: 16, n_layer: 27, n_ff: 4304, patch: 16, merge: 2, proj_dim: 5120,
            eps: 1e-6, mean: [0.5; 3], std: [0.5; 3], pos_side: 48, min_tokens: 8, max_tokens: 4096 }
    }

    #[test]
    fn smart_resize_matches_llama_cpp() {
        let c = cfg();
        // Already aligned and within budget: unchanged.
        assert_eq!(c.target_size(768, 768), (768, 768));
        // Rounded to multiples of 32.
        assert_eq!(c.target_size(1000, 700), (992, 704));
        // Over 4096 tokens: scaled down, floored.
        let (w, h) = c.target_size(4000, 3000);
        assert!(w % 32 == 0 && h % 32 == 0 && (w / 32) * (h / 32) <= 4096);
        // Tiny: scaled up to at least 8 tokens.
        let (w, h) = c.target_size(20, 20);
        assert!((w / 32) * (h / 32) >= 8);
    }

    #[test]
    fn bilinear_resize_keeps_corners() {
        let src = Rgb { width: 2, height: 2, pixels: vec![0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255] };
        let out = resize_bilinear(&src, 3, 2);
        assert_eq!(&out.pixels[0..3], &[0, 0, 0]);
        assert_eq!(&out.pixels[6..9], &[255, 255, 255]);
        assert_eq!(out.pixels[3], 127);
    }
}
