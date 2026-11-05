//! `airun selftest` -- check every GPU kernel against the CPU reference.
//!
//! The CPU decoders in airun-gguf are the oracle. A quantization layout that is
//! subtly wrong produces text that looks almost right, which is far worse than a
//! crash, so this runs the two implementations against the same bytes and
//! compares numerically.

use airun_cuda::{DeviceBuffer, Stream, ops};
use airun_gguf::{GgmlType, dequant};
use anyhow::{Result, bail};
use clap::Args;

#[derive(Args)]
pub struct SelftestArgs {
    /// CUDA device to test on.
    #[arg(long, default_value_t = 0)]
    pub device: i32,

    /// Print per-case detail even when the case passes.
    #[arg(long)]
    pub verbose: bool,
}

/// Deterministic PRNG, so a failure reproduces exactly.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 13) as u8
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() % 20001) as f32 / 10000.0 - 1.0
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }
    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

struct Report {
    passed: usize,
    failed: usize,
    verbose: bool,
}

impl Report {
    fn case(&mut self, name: &str, detail: String, ok: bool) {
        if ok {
            self.passed += 1;
            if self.verbose {
                println!("  PASS  {name:<34} {detail}");
            }
        } else {
            self.failed += 1;
            println!("  FAIL  {name:<34} {detail}");
        }
    }
}

/// Largest absolute and relative deviation between two vectors.
///
/// Non-finite values are counted separately and reported as infinite error.
/// `f32::max` silently returns the other operand when one is NaN, so a naive
/// maximum over differences reads two NaN-filled arrays as a perfect match --
/// a comparison that cannot fail is worse than no comparison at all.
fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (abs, rel, bad) = compare_detailed(a, b);
    if bad > 0 {
        return (f32::INFINITY, f32::INFINITY);
    }
    (abs, rel)
}

/// Returns (max absolute error, max relative error, count of non-finite values).
fn compare_detailed(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut non_finite = 0usize;
    for (x, y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            non_finite += 1;
            continue;
        }
        let abs = (x - y).abs();
        max_abs = max_abs.max(abs);
        let scale = x.abs().max(y.abs());
        if scale > 1e-6 {
            max_rel = max_rel.max(abs / scale);
        }
    }
    (max_abs, max_rel, non_finite)
}

pub fn run(args: SelftestArgs) -> Result<()> {
    if !airun_cuda::device::built_with_cuda() {
        bail!("this airun binary was built without CUDA kernels; nothing to test");
    }
    let dev = airun_cuda::device::device(args.device)?;
    println!(
        "airun selftest on GPU {} - {} ({}, {})",
        dev.ordinal,
        dev.name,
        dev.sm_arch(),
        dev.generation().name()
    );

    airun_cuda::device::set_device(args.device)?;
    let stream = Stream::new(args.device)?;
    let mut report = Report {
        passed: 0,
        failed: 0,
        verbose: args.verbose,
    };

    println!("\n== block decoders (GPU vs CPU reference) ==");
    test_dequant(args.device, &stream, &mut report)?;

    println!("\n== matvec ==");
    test_matvec(args.device, &stream, &mut report)?;
    test_matvec_q8(args.device, &stream, &mut report)?;
    test_mmq_glu(args.device, &stream, &mut report)?;
    test_matmul_fp4(args.device, &stream, &mut report)?;
    test_matvec_batch(args.device, &stream, &mut report)?;

    println!("\n== elementwise and norms ==");
    test_rmsnorm(args.device, &stream, &mut report)?;
    test_swiglu(args.device, &stream, &mut report)?;
    test_rope(args.device, &stream, &mut report)?;

    println!("\n== attention ==");
    test_attention(args.device, &stream, &mut report)?;

    println!("\n== vision encoder and image positions ==");
    test_layernorm(args.device, &stream, &mut report)?;
    test_vision_attention(args.device, &stream, &mut report)?;
    test_mrope(args.device, &stream, &mut report)?;

    println!("\n== gated delta net (hybrid layers) ==");
    test_causal_conv1d(args.device, &stream, &mut report)?;
    test_l2norm(args.device, &stream, &mut report)?;
    test_delta_gates(args.device, &stream, &mut report)?;
    test_delta_rule(args.device, &stream, &mut report)?;
    test_delta_chunk(args.device, &stream, &mut report)?;
    test_delta_defer(args.device, &stream, &mut report)?;
    test_conv_oop(args.device, &stream, &mut report)?;
    test_delta_head_broadcast(args.device, &stream, &mut report)?;
    test_rmsnorm_gated(args.device, &stream, &mut report)?;
    test_split_head_gate(args.device, &stream, &mut report)?;

    println!("\n== mixture of experts ==");
    test_moe_router(args.device, &stream, &mut report)?;
    test_moe_ffn(args.device, &stream, &mut report)?;

    println!(
        "\n{} passed, {} failed",
        report.passed, report.failed
    );
    if report.failed > 0 {
        bail!("{} selftest case(s) failed", report.failed);
    }
    println!("all kernels agree with the CPU reference");
    Ok(())
}

/// Types with both a CPU decoder and a GPU kernel.
const TYPES: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::IQ4_NL,
    GgmlType::IQ4_XS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::Q2_0,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
];

/// Builds `n_blocks` of random packed data.
///
/// Two types need their scale field constrained. f16/bf16 built from arbitrary
/// bits can be inf or NaN, and MXFP4's E8M0 byte spans 2^-127..2^128 -- a
/// uniform draw there produces scales no quantizer would ever emit and overflows
/// any f32 accumulation downstream. The payload bits stay fully random in both
/// cases, so the decode path is still exercised end to end.
///
/// NVFP4 needs no such guard even though its scale is also an exponent field:
/// UE4M3 tops out at 224, so random bytes stay inside f32 and additionally hit
/// the zero and 0x7F encodings that a well-behaved quantizer rarely emits.
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    let layout = ty.layout().unwrap();
    let total = n_blocks * layout.bytes as usize;
    match ty {
        GgmlType::F32 => (0..total / 4)
            .flat_map(|_| rng.unit().to_le_bytes())
            .collect(),
        GgmlType::F16 => (0..total / 2)
            .flat_map(|_| half::f16::from_f32(rng.unit()).to_le_bytes())
            .collect(),
        GgmlType::BF16 => (0..total / 2)
            .flat_map(|_| {
                // Round-to-nearest-even down to bf16, then keep the top 16 bits.
                let bits = rng.unit().to_bits();
                let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
                ((rounded >> 16) as u16).to_le_bytes()
            })
            .collect(),
        GgmlType::MXFP4 => {
            let mut out = rng.bytes(total);
            // Byte 0 of each 17-byte block is the E8M0 exponent. 121..=133 spans
            // 2^-7..2^5 after the half-scaling, the range real weights occupy.
            for block in out.chunks_mut(17) {
                block[0] = 121 + (rng.byte() % 13);
            }
            out
        }
        _ => {
            let mut out = rng.bytes(total);
            // Random bits land on an f16 NaN or infinity about 3% of the time
            // (exponent all ones). A NaN scale poisons every value in the block
            // and then every dot product that touches it, so the scale fields
            // get plausible finite values while the payload stays fully random.
            let block_bytes = layout.bytes as usize;
            for block in out.chunks_mut(block_bytes) {
                for &off in f16_scale_offsets(ty) {
                    let scale = half::f16::from_f32(rng.unit() * 0.1);
                    block[off..off + 2].copy_from_slice(&scale.to_le_bytes());
                }
            }
            out
        }
    }
}

/// Byte offsets of the f16 scale fields within each block type.
///
/// These follow the struct layouts in `dequant.cuh`; a wrong offset here only
/// makes the test data less realistic, never incorrect.
fn f16_scale_offsets(ty: GgmlType) -> &'static [usize] {
    match ty {
        GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::IQ4_NL => &[0],
        // d then dmin / m, both at the front.
        GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        // IQ4_XS keeps d first, followed by packed integer scales; so do the
        // lattice types and Q2_0.
        GgmlType::IQ4_XS | GgmlType::IQ2_S | GgmlType::IQ3_XXS | GgmlType::IQ3_S | GgmlType::Q2_0 => &[0],
        GgmlType::Q2_K => &[80, 82],
        GgmlType::Q3_K => &[108],
        GgmlType::Q6_K => &[208],
        _ => &[],
    }
}

fn test_dequant(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N_BLOCKS: usize = 64;

    for &ty in TYPES {
        let layout = ty.layout().unwrap();
        let n_elements = N_BLOCKS * layout.elements as usize;

        let mut rng = Rng::new(0xA1_1000 + ty.id() as u64);
        let packed = random_blocks(&mut rng, ty, N_BLOCKS);

        let mut cpu = vec![0f32; n_elements];
        dequant::dequantize(ty, &packed, n_elements, &mut cpu)?;

        let d_src = DeviceBuffer::<u8>::from_slice(device, &packed, stream)?;
        let d_dst = DeviceBuffer::<f32>::new(device, n_elements)?;
        ops::dequantize(
            ops::KernelType(ty.id() as i32),
            d_src.as_ptr(),
            d_dst.as_mut_ptr(),
            n_elements,
            stream,
        )?;
        let gpu = d_dst.to_vec(stream)?;

        let (abs, rel) = compare(&cpu, &gpu);
        // Both sides compute in f32 from identical integers, so agreement should
        // be near-exact; the tolerance only covers fused-multiply-add reassociation.
        let ok = abs <= 1e-4 * scale_of(&cpu).max(1.0) && rel <= 1e-5;
        report.case(
            &format!("dequant {ty}"),
            format!("max_abs {abs:.3e}  max_rel {rel:.3e}  ({n_elements} elements)"),
            ok,
        );
    }
    Ok(())
}

fn scale_of(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

fn test_matvec(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const K: usize = 512;
    const N_ROWS: usize = 96;
    const N_TOKENS: usize = 3;

    for &ty in TYPES {
        let layout = ty.layout().unwrap();
        let elems = layout.elements as usize;
        let blocks_per_row = K / elems;
        let row_bytes = blocks_per_row * layout.bytes as usize;

        let mut rng = Rng::new(0xB2_2000 + ty.id() as u64);
        let mut packed = Vec::with_capacity(N_ROWS * row_bytes);
        for _ in 0..N_ROWS {
            packed.extend_from_slice(&random_blocks(&mut rng, ty, blocks_per_row));
        }
        let x = rng.floats(N_TOKENS * K);

        // CPU reference accumulates in f64. Some types (MXFP4's E8M0 exponent,
        // Q6_K's 8-bit scales) reach magnitudes of 1e30 from random bytes, and an
        // f32 reference would itself be the least accurate term in the comparison.
        let mut row = vec![0f32; K];
        let mut cpu = vec![0f32; N_TOKENS * N_ROWS];
        // Sum of |w_i * x_i| per output: the conditioning of that dot product.
        let mut magnitude = vec![0f64; N_TOKENS * N_ROWS];
        for r in 0..N_ROWS {
            dequant::dequantize(ty, &packed[r * row_bytes..(r + 1) * row_bytes], K, &mut row)?;
            for t in 0..N_TOKENS {
                let xt = &x[t * K..(t + 1) * K];
                let mut acc = 0f64;
                let mut mag = 0f64;
                for (w, v) in row.iter().zip(xt) {
                    let term = f64::from(*w) * f64::from(*v);
                    acc += term;
                    mag += term.abs();
                }
                cpu[t * N_ROWS + r] = acc as f32;
                magnitude[t * N_ROWS + r] = mag;
            }
        }

        let d_w = DeviceBuffer::<u8>::from_slice(device, &packed, stream)?;
        let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
        let d_y = DeviceBuffer::<f32>::new(device, N_TOKENS * N_ROWS)?;
        ops::matvec(
            ops::KernelType(ty.id() as i32),
            d_w.as_ptr(),
            d_x.as_ptr(),
            d_y.as_mut_ptr(),
            K,
            N_ROWS,
            N_TOKENS,
            row_bytes,
            stream,
        )?;
        let gpu = d_y.to_vec(stream)?;

        // Backward-error test. Comparing outputs relatively is meaningless for a
        // dot product: when terms cancel, the result can be arbitrarily small
        // next to the values that produced it, and any summation order gives a
        // different answer. The meaningful bound is against the sum of absolute
        // terms, scaled by f32 epsilon and the growth factor of a tree reduction
        // over K terms.
        const EPS_F32: f64 = 1.192_092_9e-7;
        let bound_factor = EPS_F32 * (K as f64).sqrt() * 4.0;

        let mut worst_ulps = 0f64;
        let mut worst_abs = 0f64;
        for i in 0..cpu.len() {
            let diff = (f64::from(gpu[i]) - f64::from(cpu[i])).abs();
            worst_abs = worst_abs.max(diff);
            // How many times over the acceptable error this output is.
            let allowed = magnitude[i] * bound_factor;
            if allowed > 0.0 {
                worst_ulps = worst_ulps.max(diff / allowed);
            }
        }

        report.case(
            &format!("matvec {ty}"),
            format!(
                "max_abs {worst_abs:.3e}  error/bound {worst_ulps:.2}x  (bound = {bound_factor:.1e} * sum|w*x|)"
            ),
            worst_ulps <= 1.0,
        );
    }
    Ok(())
}

/// The large-batch (tensor-core) matvec against the f32 CPU reference.
///
/// Dispatch sends 64+ tokens through the WMMA kernel, which rounds both
/// operands to fp16 before a fp32 accumulate, so the bound here is fp16's
/// epsilon, not f32's. A batch of 3 (the other matvec tests) never reaches
/// this kernel -- this case exists so the tensor-core path cannot silently
/// break while everything else stays green.
fn test_matvec_batch(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const K: usize = 512;
    const N_ROWS: usize = 80; // not a multiple of the 64-row tile: edge path runs
    const N_TOKENS: usize = 96; // >= 64, so the WMMA dispatch takes it

    for &ty in TYPES {
        let layout = ty.layout().unwrap();
        let elems = layout.elements as usize;
        let blocks_per_row = K / elems;
        let row_bytes = blocks_per_row * layout.bytes as usize;

        let mut rng = Rng::new(0xE5_5000 + ty.id() as u64);
        let mut packed = Vec::with_capacity(N_ROWS * row_bytes);
        for _ in 0..N_ROWS {
            packed.extend_from_slice(&random_blocks(&mut rng, ty, blocks_per_row));
        }
        let x = rng.floats(N_TOKENS * K);

        let mut row = vec![0f32; K];
        let mut cpu = vec![0f32; N_TOKENS * N_ROWS];
        let mut magnitude = vec![0f64; N_TOKENS * N_ROWS];
        for r in 0..N_ROWS {
            dequant::dequantize(ty, &packed[r * row_bytes..(r + 1) * row_bytes], K, &mut row)?;
            for t in 0..N_TOKENS {
                let xt = &x[t * K..(t + 1) * K];
                let mut acc = 0f64;
                let mut mag = 0f64;
                for (w, v) in row.iter().zip(xt) {
                    let term = f64::from(*w) * f64::from(*v);
                    acc += term;
                    mag += term.abs();
                }
                cpu[t * N_ROWS + r] = acc as f32;
                magnitude[t * N_ROWS + r] = mag;
            }
        }

        let d_w = DeviceBuffer::<u8>::from_slice(device, &packed, stream)?;
        let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
        let d_y = DeviceBuffer::<f32>::new(device, N_TOKENS * N_ROWS)?;
        ops::matvec(
            ops::KernelType(ty.id() as i32),
            d_w.as_ptr(),
            d_x.as_ptr(),
            d_y.as_mut_ptr(),
            K,
            N_ROWS,
            N_TOKENS,
            row_bytes,
            stream,
        )?;
        let gpu = d_y.to_vec(stream)?;

        // Both operands round to fp16 (relative error up to eps_f16/2 each);
        // the products then accumulate in f32.
        const EPS_F16: f64 = 9.765_625e-4;
        let bound_factor = EPS_F16 * 3.0;

        let mut worst_ulps = 0f64;
        let mut worst_abs = 0f64;
        for i in 0..cpu.len() {
            let diff = (f64::from(gpu[i]) - f64::from(cpu[i])).abs();
            worst_abs = worst_abs.max(diff);
            let allowed = magnitude[i] * bound_factor;
            if allowed > 0.0 {
                worst_ulps = worst_ulps.max(diff / allowed);
            }
        }

        report.case(
            &format!("matvec_batch {ty}"),
            format!("max_abs {worst_abs:.3e}  error/bound {worst_ulps:.2}x  (fp16 tensor-core path)"),
            worst_ulps <= 1.0,
        );
    }
    Ok(())
}

/// CPU mirror of `quantize_q8a_kernel` in kernels/matmul.cu -- the two must
/// change together. Returns each 32-group's codes dequantized back to f32
/// (`half(d) * q`), which is exactly the value the integer dot's scale algebra
/// resolves to, so the f64 reference dot below prices in every rounding the
/// GPU path makes except the final combine.
fn quantize_q8a_mirror(x: &[f32]) -> Vec<f32> {
    assert!(x.len() % 32 == 0);
    let mut out = Vec::with_capacity(x.len());
    for group in x.chunks_exact(32) {
        let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        // The GPU stores d as half and multiplies with the half-rounded value.
        let d_h = half::f16::from_f32(d).to_f32();
        for &v in group {
            let q = (v * id).round_ties_even();
            out.push(d_h * q);
        }
    }
    out
}

/// The integer (dp4a) matvec against the CPU mirror of its own algorithm.
///
/// Unlike `test_matvec` this is not a bit-for-bit check of a decoder: the
/// activation is quantized to 8 bits on both sides, so agreement within the
/// backward-error bound proves the GPU integer path computes the same product
/// the mirrored quantization defines.
fn test_matvec_q8(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    // 3 tokens take the dp4a tiles; 100 take the integer tensor-core kernel on
    // sm_80+. Neither 96 rows nor 100 tokens fill its 128 x 64 tile, and 544
    // is 17 groups where it stages 4 at a time, so every edge path runs.
    const N_ROWS: usize = 96;
    // (tokens, odd K): 100 tokens with K = 512 take the cp.async MMQ kernel for
    // Q8_0/Q4_K/Q5_K; K = 544 (Q8_0 only) is 17 groups, which it cannot chunk,
    // so the synchronous one runs.
    // 300 tokens take the 128-token tile, with a partial last tile.
    for (n_tokens, odd_k) in [(1usize, false), (3, false), (8, false), (12, false), (100, false), (100, true), (300, false)] {
    for &ty in TYPES {
        let kty = ops::KernelType(ty.id() as i32);
        if !ops::matvec_supports_q8(kty) {
            continue;
        }
        let layout = ty.layout().unwrap();
        let elems = layout.elements as usize;
        if odd_k && elems != 32 {
            continue;
        }
        let K: usize = if odd_k { 544 } else { 512 };
        let N_TOKENS = n_tokens;
        let blocks_per_row = K / elems;
        let row_bytes = blocks_per_row * layout.bytes as usize;

        let mut rng = Rng::new(0xD8_4000 + ty.id() as u64);
        let mut packed = Vec::with_capacity(N_ROWS * row_bytes);
        for _ in 0..N_ROWS {
            packed.extend_from_slice(&random_blocks(&mut rng, ty, blocks_per_row));
        }
        let x = rng.floats(N_TOKENS * K);
        let xq = quantize_q8a_mirror(&x);

        let mut row = vec![0f32; K];
        let mut cpu = vec![0f32; N_TOKENS * N_ROWS];
        let mut magnitude = vec![0f64; N_TOKENS * N_ROWS];
        for r in 0..N_ROWS {
            dequant::dequantize(ty, &packed[r * row_bytes..(r + 1) * row_bytes], K, &mut row)?;
            for t in 0..N_TOKENS {
                let at = &xq[t * K..(t + 1) * K];
                let mut acc = 0f64;
                let mut mag = 0f64;
                for (w, v) in row.iter().zip(at) {
                    let term = f64::from(*w) * f64::from(*v);
                    acc += term;
                    mag += term.abs();
                }
                cpu[t * N_ROWS + r] = acc as f32;
                magnitude[t * N_ROWS + r] = mag;
            }
        }

        let d_w = DeviceBuffer::<u8>::from_slice(device, &packed, stream)?;
        let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
        let d_xq = DeviceBuffer::<u8>::new(device, N_TOKENS * K / 32 * 36)?;
        let d_y = DeviceBuffer::<f32>::new(device, N_TOKENS * N_ROWS)?;
        ops::quantize_act(d_x.as_ptr(), d_xq.as_mut_ptr(), N_TOKENS * K, stream)?;
        ops::matvec_q8(
            kty,
            d_w.as_ptr(),
            d_xq.as_ptr(),
            d_y.as_mut_ptr(),
            K,
            N_ROWS,
            N_TOKENS,
            row_bytes,
            stream,
        )?;
        let gpu = d_y.to_vec(stream)?;

        // Same backward-error bound as test_matvec; the mirrored quantization
        // means the 8-bit rounding is already inside `cpu` and does not need
        // headroom here. The GPU folds the group sum through a half `s`, which
        // the mirror prices per element instead, so the bound gets one extra
        // half-epsilon of slack per term.
        const EPS_F32: f64 = 1.192_092_9e-7;
        const EPS_F16: f64 = 9.765_625e-4;
        let bound_factor = EPS_F32 * (K as f64).sqrt() * 4.0 + EPS_F16 * 2.0;

        let mut worst_ulps = 0f64;
        let mut worst_abs = 0f64;
        for i in 0..cpu.len() {
            let diff = (f64::from(gpu[i]) - f64::from(cpu[i])).abs();
            worst_abs = worst_abs.max(diff);
            let allowed = magnitude[i] * bound_factor;
            if allowed > 0.0 {
                worst_ulps = worst_ulps.max(diff / allowed);
            }
        }

        let path = if n_tokens >= 32 && ops::mmq_supported() { "mmq" } else { "dp4a" };
        report.case(
            &format!("matvec_q8 {ty}, {n_tokens} tokens, K {K} ({path})"),
            format!(
                "max_abs {worst_abs:.3e}  error/bound {worst_ulps:.2}x  (activation quantized both sides)"
            ),
            worst_ulps <= 1.0,
        );
    }
    }
    Ok(())
}

/// NVFP4 x NVFP4 on the FP4 tensor cores (Blackwell only). The activation is
/// quantized on the GPU and read back, so the reference multiplies exactly the
/// values the MMA saw, in f64; the quantization itself is checked separately
/// against its rounding bound. Rows and tokens that do not fill a 128 x 128
/// tile, and 0x7F run scales (ggml's zero, the hardware's NaN), included.
fn test_matmul_fp4(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    if !ops::fp4_supported() {
        return Ok(());
    }
    const ROWS: usize = 200;
    const K: usize = 512;
    for n_tokens in [100usize, 300] {
        let ty = GgmlType::NVFP4;
        let row_bytes = K / 64 * 36;
        let mut rng = Rng::new(0xF4F4_0000 + n_tokens as u64);
        let mut packed = Vec::with_capacity(ROWS * row_bytes);
        for _ in 0..ROWS {
            packed.extend_from_slice(&random_blocks(&mut rng, ty, K / 64));
        }
        let x = rng.floats(n_tokens * K);

        let d_w = DeviceBuffer::<u8>::from_slice(device, &packed, stream)?;
        ops::nvfp4_sanitize(d_w.as_mut_ptr(), ROWS * K / 64, stream)?;
        let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
        let d_xq = DeviceBuffer::<u8>::new(device, ops::nvfp4_act_bytes(n_tokens, K))?;
        let d_y = DeviceBuffer::<f32>::new(device, n_tokens * ROWS)?;
        ops::quantize_act_nvfp4(d_x.as_ptr(), d_xq.as_mut_ptr(), n_tokens, K, stream)?;
        ops::matmul_nvfp4(d_w.as_ptr(), row_bytes, d_xq.as_ptr(), d_y.as_mut_ptr(), K, ROWS, n_tokens, stream)?;
        let gpu = d_y.to_vec(stream)?;
        let xq = d_xq.to_vec(stream)?;

        // The activation as the MMA reads it: NVFP4 blocks times the row scale.
        let blocks = n_tokens * K / 64 * 36;
        let mut xd = vec![0f32; n_tokens * K];
        let mut q_over = 0usize;
        for t in 0..n_tokens {
            let s = f32::from_le_bytes(xq[blocks + 4 * t..blocks + 4 * t + 4].try_into().unwrap());
            dequant::dequantize(ty, &xq[t * K / 64 * 36..(t + 1) * K / 64 * 36], K, &mut xd[t * K..(t + 1) * K])?;
            for (i, v) in xd[t * K..(t + 1) * K].iter_mut().enumerate() {
                *v *= s;
                // Rounding to E2M1 under a UE4M3 run scale is off by at most a
                // quarter of the run's peak (the gap between 4 and 6, relative to 6,
                // plus the scale's own rounding).
                let run = &x[t * K + i / 16 * 16..t * K + i / 16 * 16 + 16];
                let peak = run.iter().fold(0f32, |m, v| m.max(v.abs()));
                if (x[t * K + i] - *v).abs() > 0.25 * peak + 1e-6 {
                    q_over += 1;
                }
            }
        }
        let mut row = vec![0f32; K];
        let (mut worst, mut worst_abs) = (0f64, 0f64);
        for r in 0..ROWS {
            dequant::dequantize(ty, &packed[r * row_bytes..(r + 1) * row_bytes], K, &mut row)?;
            for t in 0..n_tokens {
                let (mut acc, mut mag) = (0f64, 0f64);
                for (w, v) in row.iter().zip(&xd[t * K..(t + 1) * K]) {
                    let term = f64::from(*w) * f64::from(*v);
                    acc += term;
                    mag += term.abs();
                }
                let diff = (f64::from(gpu[t * ROWS + r]) - acc).abs();
                worst_abs = worst_abs.max(diff);
                if mag > 0.0 {
                    worst = worst.max(diff / (mag * 1.192_092_9e-7 * (K as f64).sqrt() * 4.0));
                }
            }
        }
        report.case(&format!("matmul NVFP4 x NVFP4 (FP4 tensor cores), {n_tokens} tokens"),
            format!("max_abs {worst_abs:.3e}  error/bound {worst:.2}x  {q_over} activations past the rounding bound"),
            worst <= 1.0 && q_over == 0);
    }

    // The vectorized quantizer runs on 16-byte aligned rows and the scalar one
    // on anything else; one float off alignment puts the same rows through
    // the scalar kernel, and the two must agree to the bit. K past the
    // vectorized kernel's register cache (20480) covers its reload tail.
    for k in [512usize, 21504] {
        let n_tokens = 7;
        let mut rng = Rng::new(0xF4F4_1000 + k as u64);
        let g = rng.floats(n_tokens * k + 1);
        let u = rng.floats(n_tokens * k + 1);
        let d_g = DeviceBuffer::<f32>::from_slice(device, &g, stream)?;
        let d_u = DeviceBuffer::<f32>::from_slice(device, &u, stream)?;
        let bytes = ops::nvfp4_act_bytes(n_tokens, k);
        let outs: Vec<DeviceBuffer<u8>> =
            (0..4).map(|_| DeviceBuffer::<u8>::new(device, bytes)).collect::<std::result::Result<_, _>>()?;
        // Aligned, then the same values one float in (the buffers hold g[1..]
        // at offset 1, so feed the aligned kernel a host copy shifted down).
        let d_g1 = DeviceBuffer::<f32>::from_slice(device, &g[1..], stream)?;
        let d_u1 = DeviceBuffer::<f32>::from_slice(device, &u[1..], stream)?;
        ops::quantize_act_nvfp4(d_g1.as_ptr(), outs[0].as_mut_ptr(), n_tokens, k, stream)?;
        ops::quantize_act_nvfp4(d_g.offset_ptr(1), outs[1].as_mut_ptr(), n_tokens, k, stream)?;
        ops::quantize_glu_nvfp4(d_g1.as_ptr(), d_u1.as_ptr(), outs[2].as_mut_ptr(), n_tokens, k, stream)?;
        ops::quantize_glu_nvfp4(d_g.offset_ptr(1), d_u.offset_ptr(1), outs[3].as_mut_ptr(), n_tokens, k, stream)?;
        let o: Vec<Vec<u8>> = outs.iter().map(|b| b.to_vec(stream)).collect::<std::result::Result<_, _>>()?;
        let diff = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
        let (dp, dg) = (diff(&o[0], &o[1]), diff(&o[2], &o[3]));
        report.case(&format!("NVFP4 activation quantizer, vectorized = scalar, K {k}"),
            format!("{dp} bytes differ (plain), {dg} (SwiGLU)"),
            dp == 0 && dg == 0);
    }
    Ok(())
}

/// The fused gate/up/SwiGLU MMQ launch against the unfused one: gate and up
/// through matvec_q8 (itself checked above against the CPU), SwiGLU applied on
/// the CPU. Same integer products and scaling, so they agree to rounding.
fn test_mmq_glu(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const K: usize = 512;
    const N_ROWS: usize = 96; // one full 64-row gate/up tile and a partial one
    for n_tokens in [100usize, 300] {
        for &ty in TYPES {
            let kty = ops::KernelType(ty.id() as i32);
            if !ops::matvec_supports_q8(kty) {
                continue;
            }
            let layout = ty.layout().unwrap();
            let row_bytes = K / layout.elements as usize * layout.bytes as usize;
            let mut rng = Rng::new(0x61_1000 + ty.id() as u64 + n_tokens as u64);
            let mut gate = Vec::new();
            let mut up = Vec::new();
            for _ in 0..N_ROWS {
                gate.extend_from_slice(&random_blocks(&mut rng, ty, K / layout.elements as usize));
                up.extend_from_slice(&random_blocks(&mut rng, ty, K / layout.elements as usize));
            }
            let x = rng.floats(n_tokens * K);
            let d_g = DeviceBuffer::<u8>::from_slice(device, &gate, stream)?;
            let d_u = DeviceBuffer::<u8>::from_slice(device, &up, stream)?;
            let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
            let d_xq = DeviceBuffer::<u8>::new(device, n_tokens * K / 32 * 36)?;
            ops::quantize_act(d_x.as_ptr(), d_xq.as_mut_ptr(), n_tokens * K, stream)?;
            if !ops::mmq_glu_supported(kty, d_g.as_ptr(), d_u.as_ptr(), d_xq.as_ptr(), K, row_bytes, n_tokens) {
                continue;
            }
            let d_yg = DeviceBuffer::<f32>::new(device, n_tokens * N_ROWS)?;
            let d_yu = DeviceBuffer::<f32>::new(device, n_tokens * N_ROWS)?;
            let d_y = DeviceBuffer::<f32>::new(device, n_tokens * N_ROWS)?;
            ops::matvec_q8(kty, d_g.as_ptr(), d_xq.as_ptr(), d_yg.as_mut_ptr(), K, N_ROWS, n_tokens, row_bytes, stream)?;
            ops::matvec_q8(kty, d_u.as_ptr(), d_xq.as_ptr(), d_yu.as_mut_ptr(), K, N_ROWS, n_tokens, row_bytes, stream)?;
            ops::matvec_q8_glu(kty, d_g.as_ptr(), d_u.as_ptr(), d_xq.as_ptr(), d_y.as_mut_ptr(), K, N_ROWS, n_tokens, row_bytes, stream)?;
            let (g, u, y) = (d_yg.to_vec(stream)?, d_yu.to_vec(stream)?, d_y.to_vec(stream)?);
            // The unfused reference may split K on a grid this small (the fused
            // launch never does), which sums the groups in another order; with
            // random weights the dot products cancel heavily, so the error is
            // measured against the scale of the products, not entry by entry.
            let scale = g.iter().zip(&u).map(|(a, b)| f64::from((a * b).abs())).fold(1.0, f64::max);
            let mut worst = 0f64;
            for i in 0..y.len() {
                let want = (g[i] / (1.0 + (-g[i]).exp())) * u[i];
                worst = worst.max(f64::from((y[i] - want).abs()) / scale);
            }
            report.case(&format!("mmq gate/up/SwiGLU fused {ty}, {n_tokens} tokens"),
                format!("max error {worst:.2e} of max |gate*up|, against unfused"), worst <= 1e-5);
        }
    }
    Ok(())
}
fn test_rmsnorm(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N: usize = 2048;
    const ROWS: usize = 5;
    const EPS: f32 = 1e-6;

    let mut rng = Rng::new(0xC3_3000);
    let x = rng.floats(ROWS * N);
    let w = rng.floats(N);

    let mut cpu = vec![0f32; ROWS * N];
    for r in 0..ROWS {
        let row = &x[r * N..(r + 1) * N];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / N as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        for i in 0..N {
            cpu[r * N + i] = row[i] * inv * w[i];
        }
    }

    let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
    let d_w = DeviceBuffer::<f32>::from_slice(device, &w, stream)?;
    let d_o = DeviceBuffer::<f32>::new(device, ROWS * N)?;
    ops::rmsnorm(
        d_o.as_mut_ptr(),
        d_x.as_ptr(),
        d_w.as_ptr(),
        N,
        ROWS,
        EPS,
        stream,
    )?;
    let gpu = d_o.to_vec(stream)?;

    let (abs, rel) = compare(&cpu, &gpu);
    report.case(
        "rmsnorm",
        format!("max_abs {abs:.3e}  max_rel {rel:.3e}"),
        rel <= 1e-4,
    );
    Ok(())
}

fn test_layernorm(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N: usize = 1152;
    const ROWS: usize = 7;
    const EPS: f32 = 1e-6;
    let mut rng = Rng::new(0x1A_7000);
    let x: Vec<f32> = rng.floats(ROWS * N).iter().map(|v| v * 3.0 + 0.5).collect();
    let w = rng.floats(N);
    let b = rng.floats(N);
    let mut cpu = vec![0f32; ROWS * N];
    for r in 0..ROWS {
        let row = &x[r * N..(r + 1) * N];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / N as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / N as f64;
        let inv = 1.0 / (var + EPS as f64).sqrt();
        for i in 0..N {
            cpu[r * N + i] = ((row[i] as f64 - mean) * inv) as f32 * w[i] + b[i];
        }
    }
    let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
    let d_w = DeviceBuffer::<f32>::from_slice(device, &w, stream)?;
    let d_b = DeviceBuffer::<f32>::from_slice(device, &b, stream)?;
    let d_o = DeviceBuffer::<f32>::new(device, ROWS * N)?;
    airun_cuda::vision::layernorm(d_o.as_mut_ptr(), d_x.as_ptr(), d_w.as_ptr(), d_b.as_ptr(), ROWS, N, EPS, stream)?;
    let gpu = d_o.to_vec(stream)?;
    let (abs, rel) = compare(&cpu, &gpu);
    report.case("layernorm (bias)", format!("max_abs {abs:.3e}  max_rel {rel:.3e}"), abs <= 1e-4);
    Ok(())
}

/// The ViT's attention as the encoder runs it: fused QKV split with the 2D
/// rotary embedding, heads padded to 32, the causal kernel with every query
/// at the last position (= full attention), padding dropped. Against plain
/// full attention on the CPU.
fn test_vision_attention(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N: usize = 24;
    const H: usize = 2;
    const HD: usize = 8;
    const HDP: usize = 32;
    const BASE: f32 = 10_000.0;
    let mut rng = Rng::new(0x71_5100);
    let qkv = rng.floats(N * 3 * H * HD);
    let pos_y: Vec<i32> = (0..N as i32).map(|i| (i / 4) % 3).collect();
    let pos_x: Vec<i32> = (0..N as i32).map(|i| i % 5).collect();
    let w = H * HD;
    let rot = |v: &[f32], t: usize| -> Vec<f32> {
        let (half, quarter) = (HD / 2, HD / 4);
        let mut o = v.to_vec();
        for j in 0..half {
            let p = if j < quarter { pos_y[t] } else { pos_x[t] } as f32;
            let theta = p * BASE.powf(-2.0 * (j % quarter) as f32 / half as f32);
            let (sn, c) = theta.sin_cos();
            o[j] = v[j] * c - v[j + half] * sn;
            o[j + half] = v[j] * sn + v[j + half] * c;
        }
        o
    };
    let scale = 1.0 / (HD as f32).sqrt();
    let mut cpu = vec![0f32; N * w];
    for h in 0..H {
        let q: Vec<Vec<f32>> = (0..N).map(|t| rot(&qkv[t * 3 * w + h * HD..][..HD], t)).collect();
        let k: Vec<Vec<f32>> = (0..N).map(|t| rot(&qkv[t * 3 * w + w + h * HD..][..HD], t)).collect();
        // The cache is f16; round K and V the way the kernel stores them.
        let f16 = |v: f32| half::f16::from_f32(v).to_f32();
        let v: Vec<&[f32]> = (0..N).map(|t| &qkv[t * 3 * w + 2 * w + h * HD..][..HD]).collect();
        for t in 0..N {
            let s: Vec<f32> = (0..N).map(|j| q[t].iter().zip(&k[j]).map(|(a, b)| a * f16(*b)).sum::<f32>() * scale)
                .collect();
            let m = s.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for d in 0..HD {
                cpu[t * w + h * HD + d] = (0..N).map(|j| e[j] / z * f16(v[j][d])).sum();
            }
        }
    }
    let d_qkv = DeviceBuffer::<f32>::from_slice(device, &qkv, stream)?;
    let d_py = DeviceBuffer::<i32>::from_slice(device, &pos_y, stream)?;
    let d_px = DeviceBuffer::<i32>::from_slice(device, &pos_x, stream)?;
    let d_last = DeviceBuffer::<i32>::from_slice(device, &vec![N as i32 - 1; N], stream)?;
    let d_q = DeviceBuffer::<f32>::new(device, N * H * HDP)?;
    let d_k = DeviceBuffer::<u16>::new(device, N * H * HDP)?;
    let d_v = DeviceBuffer::<u16>::new(device, N * H * HDP)?;
    let d_a = DeviceBuffer::<f32>::new(device, N * H * HDP)?;
    let d_o = DeviceBuffer::<f32>::new(device, N * w)?;
    airun_cuda::vision::vision_qkv(d_q.as_mut_ptr(), d_k.as_mut_ptr(), d_v.as_mut_ptr(), d_qkv.as_ptr(),
        d_py.as_ptr(), d_px.as_ptr(), N, H, HD, HDP, BASE, stream)?;
    ops::attention(d_a.as_mut_ptr(), d_q.as_ptr(), d_k.as_ptr(), d_v.as_ptr(), d_last.as_ptr(), N, H, H, HDP, N,
        scale, 0, ops::KvType::F16, stream)?;
    airun_cuda::vision::vision_unpad(d_o.as_mut_ptr(), d_a.as_ptr(), N * H, HD, HDP, stream)?;
    let gpu = d_o.to_vec(stream)?;
    let (abs, rel) = compare(&cpu, &gpu);
    report.case("vision attention (2D rope, full)", format!("max_abs {abs:.3e}  max_rel {rel:.3e}"), abs <= 2e-3);
    Ok(())
}

/// The text model's RoPE with an image's (t, row, col) positions spread over
/// the pairs by the interleaved M-RoPE sections, and with the shift text
/// after an image runs at.
fn test_mrope(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N: usize = 6;
    const HEADS: usize = 2;
    const HD: usize = 64;
    let sections = [11, 11, 10];
    let mut rng = Rng::new(0x3D_0E00);
    let x = rng.floats(N * HEADS * HD);
    let mut rp = ops::RopeParams::plain(10_000_000.0, 1.0);
    rp.sections = sections;
    let mpos: Vec<i32> = [vec![40; N], (0..N as i32).map(|i| 40 + i / 3).collect(), (0..N as i32).map(|i| 40 + i % 3)
        .collect()].concat();
    let slots: Vec<i32> = (100..100 + N as i32).collect();
    let axis = |pair: usize| if pair % 3 == 1 && pair < 3 * sections[1] as usize { 1 }
        else if pair % 3 == 2 && pair < 3 * sections[2] as usize { 2 } else { 0 };
    let cpu = |pos: &dyn Fn(usize, usize) -> i32| -> Vec<f32> {
        let mut o = x.clone();
        for t in 0..N {
            for h in 0..HEADS {
                let b = (t * HEADS + h) * HD;
                for i in 0..HD / 2 {
                    let (c, s) = rp.cos_sin(i, HD, pos(t, i));
                    let (x0, x1) = (x[b + i], x[b + i + HD / 2]);
                    o[b + i] = x0 * c - x1 * s;
                    o[b + i + HD / 2] = x0 * s + x1 * c;
                }
            }
        }
        o
    };
    let d_pos = DeviceBuffer::<i32>::from_slice(device, &slots, stream)?;
    let d_m = DeviceBuffer::<i32>::from_slice(device, &mpos, stream)?;
    let d_shift = DeviceBuffer::<i32>::from_slice(device, &[37], stream)?;
    let run = |p: ops::RopeParams| -> Result<Vec<f32>> {
        let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
        ops::rope(d_x.as_mut_ptr(), d_pos.as_ptr(), N, HEADS, HD, HD, p, false, stream)?;
        Ok(d_x.to_vec(stream)?)
    };
    let img = run(rp.positioned(d_shift.as_ptr() as u64, Some((d_m.as_ptr() as u64, N))))?;
    let want = cpu(&|t, i| mpos[axis(i) * N + t]);
    let (abs, rel) = compare(&want, &img);
    report.case("rope: image positions (M-RoPE)", format!("max_abs {abs:.3e}  max_rel {rel:.3e}"), abs <= 1e-4);
    let txt = run(rp.positioned(d_shift.as_ptr() as u64, None))?;
    let want = cpu(&|t, _| slots[t] - 37);
    let (abs, rel) = compare(&want, &txt);
    report.case("rope: text after an image (shift)", format!("max_abs {abs:.3e}  max_rel {rel:.3e}"), abs <= 1e-4);
    Ok(())
}

fn test_swiglu(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N: usize = 4096;
    let mut rng = Rng::new(0xD4_4000);
    let gate = rng.floats(N);
    let up = rng.floats(N);

    let cpu: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    let mut d_g = DeviceBuffer::<f32>::from_slice(device, &gate, stream)?;
    let d_u = DeviceBuffer::<f32>::from_slice(device, &up, stream)?;
    ops::swiglu(d_g.as_mut_ptr(), d_u.as_ptr(), N, stream)?;
    let gpu = d_g.to_vec(stream)?;
    let _ = &mut d_g;

    let (abs, rel) = compare(&cpu, &gpu);
    // __expf is a fast-math intrinsic, so a few ulp of drift is expected.
    report.case(
        "swiglu",
        format!("max_abs {abs:.3e}  max_rel {rel:.3e}"),
        rel <= 1e-4,
    );
    Ok(())
}

fn test_rope(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N_HEADS: usize = 6;
    const HEAD_DIM: usize = 128;

    // (name, rope_dim, params, positions). The YaRN case is Qwen3.8-27B's
    // 1M-context file: 64 of 256 dims rotated, base 1e7, factor 4 over 262144.
    let cases = [
        ("rope (neox)", 128, ops::RopeParams::plain(1_000_000.0, 1.0), vec![3, 10, 17, 24]),
        ("rope (neox, linear x4)", 128, ops::RopeParams::plain(1_000_000.0, 0.25), vec![3, 900, 70_001, 262_143]),
        ("rope (yarn x4, partial 64)", 64, ops::RopeParams::yarn(1e7, 4.0, 64, 262_144, 32.0, 1.0, 1.0),
            vec![3, 1000, 300_007, 1_048_575]),
    ];
    for (name, rope_dim, p, positions) in cases {
        let n_tokens = positions.len();
        let mut rng = Rng::new(0xE5_5000);
        let x = rng.floats(n_tokens * N_HEADS * HEAD_DIM);

        // The angle in f64 from the formulas themselves (llama.cpp's rope_yarn),
        // not from RopeParams::cos_sin, so the kernel and its Rust mirror cannot
        // agree on a shared mistake. f32 rounds the angle to ~|theta| * eps, so
        // that is what each element is allowed to be off by, plus a floor.
        let mut cpu = x.clone();
        let mut allowed = vec![5e-5f32; x.len()];
        let half = rope_dim / 2;
        for (t, &pos) in positions.iter().enumerate() {
            for h in 0..N_HEADS {
                let base = (t * N_HEADS + h) * HEAD_DIM;
                for i in 0..half {
                    let inv_freq = (p.base as f64).powf(-2.0 * i as f64 / rope_dim as f64);
                    let extrap = pos as f64 * inv_freq;
                    let interp = p.freq_scale as f64 * extrap;
                    let (theta, m) = if p.ext_factor == 0.0 {
                        (interp, 1.0)
                    } else {
                        let y = (i as f64 - p.corr_lo as f64) / (p.corr_hi as f64 - p.corr_lo as f64).max(0.001);
                        let ramp = (1.0 - y.clamp(0.0, 1.0)) * p.ext_factor as f64;
                        (interp * (1.0 - ramp) + extrap * ramp, p.mscale as f64)
                    };
                    let (sin_t, cos_t) = theta.sin_cos();
                    let x0 = x[base + i] as f64;
                    let x1 = x[base + i + half] as f64;
                    cpu[base + i] = (m * (x0 * cos_t - x1 * sin_t)) as f32;
                    cpu[base + i + half] = (m * (x0 * sin_t + x1 * cos_t)) as f32;
                    let tol = 5e-5 + 8.0 * f32::EPSILON as f64 * theta.abs() * (x0.abs() + x1.abs()) * m;
                    allowed[base + i] = tol as f32;
                    allowed[base + i + half] = tol as f32;
                }
            }
        }

        let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
        let d_p = DeviceBuffer::<i32>::from_slice(device, &positions, stream)?;
        ops::rope(d_x.as_mut_ptr(), d_p.as_ptr(), n_tokens, N_HEADS, HEAD_DIM, rope_dim, p, false, stream)?;
        let gpu = d_x.to_vec(stream)?;

        let (abs, rel) = compare(&cpu, &gpu);
        let over = cpu.iter().zip(&gpu).zip(&allowed).filter(|((c, g), a)| (*c - *g).abs() > **a).count();
        report.case(name, format!("max_abs {abs:.3e}  max_rel {rel:.3e}  {over} over the angle bound"), over == 0);
    }
    Ok(())
}

fn test_causal_conv1d(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const CHANNELS: usize = 256;
    const KERNEL: usize = 4;
    const N_TOKENS: usize = 12;

    let mut rng = Rng::new(0x11_0000);
    let x = rng.floats(N_TOKENS * CHANNELS);
    let w = rng.floats(CHANNELS * KERNEL);

    // CPU reference: each channel convolves with its own taps over the current
    // token and the previous three, with zeros before the sequence starts.
    let mut cpu = vec![0f32; N_TOKENS * CHANNELS];
    for c in 0..CHANNELS {
        for t in 0..N_TOKENS {
            let mut acc = 0f32;
            for i in 0..KERNEL {
                let src = t as isize - (KERNEL as isize - 1 - i as isize);
                let v = if src < 0 { 0.0 } else { x[src as usize * CHANNELS + c] };
                acc += w[c * KERNEL + i] * v;
            }
            cpu[t * CHANNELS + c] = acc / (1.0 + (-acc).exp());
        }
    }

    let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
    let d_w = DeviceBuffer::<f32>::from_slice(device, &w, stream)?;
    let d_state = DeviceBuffer::<f32>::zeroed(device, CHANNELS * (KERNEL - 1), stream)?;
    airun_cuda::hybrid::causal_conv1d(
        d_x.as_mut_ptr(),
        d_w.as_ptr(),
        d_state.as_mut_ptr(),
        N_TOKENS,
        CHANNELS,
        KERNEL,
        true,
        false,
        stream,
    )?;
    let gpu = d_x.to_vec(stream)?;

    let (abs, rel) = compare(&cpu, &gpu);
    report.case(
        "causal conv1d + silu",
        format!("max_abs {abs:.3e}  max_rel {rel:.3e}"),
        rel <= 1e-4,
    );

    // Feeding the same sequence one token at a time must give the same answer,
    // or decoding would diverge from prefill.
    let d_x2 = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
    let d_state2 = DeviceBuffer::<f32>::zeroed(device, CHANNELS * (KERNEL - 1), stream)?;
    for t in 0..N_TOKENS {
        airun_cuda::hybrid::causal_conv1d(
            d_x2.offset_ptr(t * CHANNELS),
            d_w.as_ptr(),
            d_state2.as_mut_ptr(),
            1,
            CHANNELS,
            KERNEL,
            true,
            false,
            stream,
        )?;
    }
    let stepwise = d_x2.to_vec(stream)?;
    let (sabs, _) = compare(&cpu, &stepwise);
    report.case(
        "conv1d streaming == batched",
        format!("max_abs {sabs:.3e}"),
        sabs <= 1e-5,
    );
    Ok(())
}

fn test_l2norm(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const ROWS: usize = 24;
    const DIM: usize = 128;

    let mut rng = Rng::new(0x12_0000);
    let x = rng.floats(ROWS * DIM);

    let mut cpu = x.clone();
    for r in 0..ROWS {
        let row = &mut cpu[r * DIM..(r + 1) * DIM];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        for v in row.iter_mut() {
            *v *= inv;
        }
    }

    let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
    airun_cuda::hybrid::l2norm(d_x.as_mut_ptr(), ROWS, DIM, 1e-6, stream)?;
    let gpu = d_x.to_vec(stream)?;

    let (abs, rel) = compare(&cpu, &gpu);
    report.case("l2norm", format!("max_abs {abs:.3e}  max_rel {rel:.3e}"), rel <= 1e-5);
    Ok(())
}

fn test_delta_gates(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N_TOKENS: usize = 7;
    const HEADS: usize = 32;

    let mut rng = Rng::new(0x13_0000);
    let a: Vec<f32> = (0..N_TOKENS * HEADS).map(|_| rng.unit() * 4.0).collect();
    let b: Vec<f32> = (0..N_TOKENS * HEADS).map(|_| rng.unit() * 4.0).collect();
    // The kernel receives the coefficient already in its negative form, which is
    // what the loader normalizes to. Real files hold values around -0.1.
    let a_neg: Vec<f32> = (0..HEADS).map(|_| -(0.005 + rng.unit().abs() * 0.15)).collect();
    let dt_bias: Vec<f32> = (0..HEADS).map(|_| rng.unit() * 2.0).collect();

    let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
    let mut cpu_decay = vec![0f32; N_TOKENS * HEADS];
    let mut cpu_beta = vec![0f32; N_TOKENS * HEADS];
    for i in 0..N_TOKENS * HEADS {
        let h = i % HEADS;
        let g = a_neg[h] * softplus(a[i] + dt_bias[h]);
        cpu_decay[i] = g.exp();
        cpu_beta[i] = 1.0 / (1.0 + (-b[i]).exp());
    }

    let d_a = DeviceBuffer::<f32>::from_slice(device, &a, stream)?;
    let d_b = DeviceBuffer::<f32>::from_slice(device, &b, stream)?;
    let d_alog = DeviceBuffer::<f32>::from_slice(device, &a_neg, stream)?;
    let d_dt = DeviceBuffer::<f32>::from_slice(device, &dt_bias, stream)?;
    let d_decay = DeviceBuffer::<f32>::new(device, N_TOKENS * HEADS)?;
    let d_beta = DeviceBuffer::<f32>::new(device, N_TOKENS * HEADS)?;

    airun_cuda::hybrid::delta_gates(
        d_a.as_ptr(),
        d_b.as_ptr(),
        d_alog.as_ptr(),
        d_dt.as_ptr(),
        d_decay.as_mut_ptr(),
        d_beta.as_mut_ptr(),
        N_TOKENS,
        HEADS,
        stream,
    )?;

    let (dabs, drel) = compare(&cpu_decay, &d_decay.to_vec(stream)?);
    let (babs, brel) = compare(&cpu_beta, &d_beta.to_vec(stream)?);
    // Decay must stay in (0, 1] or the recurrent state would grow without bound.
    let decay_gpu = d_decay.to_vec(stream)?;
    let in_range = decay_gpu.iter().all(|&d| d > 0.0 && d <= 1.0);
    report.case(
        "delta gates (decay, beta)",
        format!("decay {dabs:.3e}/{drel:.3e}  beta {babs:.3e}/{brel:.3e}  decay in (0,1]: {in_range}"),
        drel <= 1e-4 && brel <= 1e-4 && in_range,
    );
    Ok(())
}

fn test_delta_head_broadcast(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    let mut q = vec![0.0f32; 64]; q[0] = 1.0; q[32] = 2.0;
    let mut k = vec![0.0f32; 64]; k[0] = 1.0; k[32] = 1.0;
    let q = DeviceBuffer::from_slice(device, &q, stream)?;
    let k = DeviceBuffer::from_slice(device, &k, stream)?;
    let v = DeviceBuffer::from_slice(device, &vec![1.0f32; 128], stream)?;
    let coeff = DeviceBuffer::from_slice(device, &[1.0f32; 4], stream)?;
    let state = DeviceBuffer::<f32>::zeroed(device, 4 * 32 * 32, stream)?;
    let out = DeviceBuffer::<f32>::new(device, 128)?;
    airun_cuda::hybrid::delta_rule(state.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(),
        coeff.as_ptr(), coeff.as_ptr(), out.as_mut_ptr(), 1, 2, 4, 32, 32, stream)?;
    let got = out.to_vec(stream)?;
    let expected = [1.0f32, 2.0, 1.0, 2.0];
    report.case("delta-net GGUF head broadcast [0,1,0,1]",
        format!("first values {:?}", [got[0], got[32], got[64], got[96]]),
        got.chunks(32).zip(expected).all(|(row, e)| row.iter().all(|&x| (x-e).abs() < 1e-6)));
    Ok(())
}

/// The batch convolution (out of place, parallel over tokens) against the
/// in-place one: same outputs and same history, bit for bit, including runs
/// shorter than the history.
fn test_conv_oop(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const CH: usize = 300;
    const KER: usize = 4;
    let mut rng = Rng::new(0xC0_0E00);
    let w = rng.floats(CH * KER);
    let hist = rng.floats(CH * (KER - 1));
    let d_w = DeviceBuffer::<f32>::from_slice(device, &w, stream)?;
    let mut bad = 0usize;
    for n in [1usize, 2, 3, 9, 37] {
        let x = rng.floats(n * CH);
        let (xa, ha) = (DeviceBuffer::from_slice(device, &x, stream)?, DeviceBuffer::from_slice(device, &hist, stream)?);
        airun_cuda::hybrid::causal_conv1d(xa.as_mut_ptr(), d_w.as_ptr(), ha.as_mut_ptr(), n, CH, KER, true, false, stream)?;
        let (xb, hb) = (DeviceBuffer::from_slice(device, &x, stream)?, DeviceBuffer::from_slice(device, &hist, stream)?);
        let out = DeviceBuffer::<f32>::new(device, n * CH)?;
        airun_cuda::hybrid::causal_conv1d_oop(xb.as_ptr(), out.as_mut_ptr(), d_w.as_ptr(), hb.as_mut_ptr(), n, CH,
            KER, true, false, true, stream)?;
        let (a, b) = (xa.to_vec(stream)?, out.to_vec(stream)?);
        bad += a.iter().zip(&b).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
        bad += ha.to_vec(stream)?.iter().zip(&hb.to_vec(stream)?).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
    }
    report.case("causal conv1d, batch (out of place) vs in place", format!("{bad} values differ"), bad == 0);
    Ok(())
}

/// Speculative decoding's undo for delta-net layers (see airun-core spec.rs):
/// a verify over n tokens with keep_state off must leave state and conv
/// history untouched, and pushing the first m tokens afterwards must reach
/// the state, history and outputs of running only those m -- bit for bit,
/// since an accepted draft is meant to be indistinguishable from plain decode.
/// Head width 128 takes the register kernel the Qwen hybrids use.
fn test_delta_defer(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N: usize = 4;
    const M: usize = 2;
    const NK: usize = 2;
    const NV: usize = 4;
    const DK: usize = 128;
    const DV: usize = 128;
    const CH: usize = 96;
    const KER: usize = 4;
    let mut rng = Rng::new(0x5BEC_0001);
    let q = rng.floats(N * NK * DK);
    let k = rng.floats(N * NK * DK);
    let v = rng.floats(N * NV * DV);
    let decay: Vec<f32> = (0..N * NV).map(|_| 0.5 + rng.unit().abs() * 0.49).collect();
    let beta: Vec<f32> = (0..N * NV).map(|_| rng.unit().abs().max(0.01)).collect();
    let state0: Vec<f32> = rng.floats(NV * DK * DV).iter().map(|x| x * 0.1).collect();
    let raw = rng.floats(N * CH);
    let w = rng.floats(CH * KER);
    let hist0 = rng.floats(CH * (KER - 1));

    let up = |x: &[f32]| DeviceBuffer::<f32>::from_slice(device, x, stream);
    let (d_q, d_k, d_v, d_dec, d_beta) = (up(&q)?, up(&k)?, up(&v)?, up(&decay)?, up(&beta)?);
    let d_w = up(&w)?;
    let delta = |state: &DeviceBuffer<f32>, out: &DeviceBuffer<f32>, n: usize, keep: bool| {
        airun_cuda::hybrid::delta_rule_ex(state.as_mut_ptr(), d_q.as_ptr(), d_k.as_ptr(), d_v.as_ptr(),
            d_dec.as_ptr(), d_beta.as_ptr(), out.as_mut_ptr(), n, NK, NV, DK, DV, keep, stream)
    };
    let conv = |x: &DeviceBuffer<f32>, hist: &DeviceBuffer<f32>, n: usize, keep: bool| {
        airun_cuda::hybrid::causal_conv1d_ex(x.as_mut_ptr(), d_w.as_ptr(), hist.as_mut_ptr(), n, CH, KER,
            true, false, keep, stream)
    };

    // Reference: only the first M tokens, state kept.
    let (s_ref, o_ref) = (up(&state0)?, DeviceBuffer::<f32>::new(device, N * NV * DV)?);
    delta(&s_ref, &o_ref, M, true)?;
    let (x_ref, h_ref) = (up(&raw)?, up(&hist0)?);
    conv(&x_ref, &h_ref, M, true)?;

    // Verify over N without keeping, then commit M.
    let (s, o) = (up(&state0)?, DeviceBuffer::<f32>::new(device, N * NV * DV)?);
    delta(&s, &o, N, false)?;
    let untouched = s.to_vec(stream)? == state0;
    let (x, h) = (up(&raw)?, up(&hist0)?);
    conv(&x, &h, N, false)?;
    let hist_untouched = h.to_vec(stream)? == hist0;
    let same_out = o.to_vec(stream)?[..M * NV * DV] == o_ref.to_vec(stream)?[..M * NV * DV]
        && x.to_vec(stream)?[..M * CH] == x_ref.to_vec(stream)?[..M * CH];
    let scratch = DeviceBuffer::<f32>::new(device, N * NV * DV)?;
    delta(&s, &scratch, M, true)?;
    let x2 = up(&raw)?;
    conv(&x2, &h, M, true)?;
    let same_state = s.to_vec(stream)? == s_ref.to_vec(stream)? && h.to_vec(stream)? == h_ref.to_vec(stream)?;

    report.case("delta rule + conv: verify leaves state, commit replays it",
        format!("state untouched {untouched}, history untouched {hist_untouched}, outputs equal {same_out},                  committed state equal {same_state}"),
        untouched && hist_untouched && same_out && same_state);
    Ok(())
}

/// The chunked delta rule (delta_chunk.cu), which a prompt of 64+ tokens with
/// 128-wide heads takes, against the f64 recurrence: three full chunks and a
/// partial one, q and k L2-normalized as the model feeds them, carrying a
/// state in. bf16 tensor-core operands: held to 1% of the output's range.
fn test_delta_chunk(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N: usize = 211;
    const KH: usize = 2;
    const VH: usize = 4;
    const D: usize = 128;
    let mut rng = Rng::new(0xDC_0001);
    let norm = |x: Vec<f32>| -> Vec<f32> {
        x.chunks(D).flat_map(|r| {
            let n = r.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            r.iter().map(move |v| v / n).collect::<Vec<_>>()
        }).collect()
    };
    let q = norm(rng.floats(N * KH * D));
    let k = norm(rng.floats(N * KH * D));
    let v = rng.floats(N * VH * D);
    let decay: Vec<f32> = (0..N * VH).map(|_| 0.6 + rng.unit().abs() * 0.399).collect();
    let beta: Vec<f32> = (0..N * VH).map(|_| rng.unit().abs().max(0.01)).collect();
    let s0: Vec<f32> = rng.floats(VH * D * D).iter().map(|x| x * 0.1).collect();

    let mut st: Vec<f64> = s0.iter().map(|&x| x as f64).collect();
    let mut cpu = vec![0f32; N * VH * D];
    for t in 0..N {
        for vh in 0..VH {
            let kh = vh % KH;
            let s = &mut st[vh * D * D..(vh + 1) * D * D];
            let d = decay[t * VH + vh] as f64;
            let b = beta[t * VH + vh] as f64;
            let (qv, kv, vv) = (&q[(t * KH + kh) * D..][..D], &k[(t * KH + kh) * D..][..D], &v[(t * VH + vh) * D..][..D]);
            s.iter_mut().for_each(|x| *x *= d);
            let mut mem = vec![0f64; D];
            for i in 0..D {
                for j in 0..D {
                    mem[j] += s[i * D + j] * kv[i] as f64;
                }
            }
            for i in 0..D {
                for j in 0..D {
                    s[i * D + j] += kv[i] as f64 * (vv[j] as f64 - mem[j]) * b;
                }
            }
            for i in 0..D {
                for j in 0..D {
                    cpu[(t * VH + vh) * D + j] += (s[i * D + j] * qv[i] as f64) as f32;
                }
            }
        }
    }
    let up = |x: &[f32]| DeviceBuffer::<f32>::from_slice(device, x, stream);
    let (d_q, d_k, d_v, d_d, d_b, d_s) = (up(&q)?, up(&k)?, up(&v)?, up(&decay)?, up(&beta)?, up(&s0)?);
    let d_o = DeviceBuffer::<f32>::new(device, N * VH * D)?;
    airun_cuda::hybrid::delta_rule(d_s.as_mut_ptr(), d_q.as_ptr(), d_k.as_ptr(), d_v.as_ptr(), d_d.as_ptr(),
        d_b.as_ptr(), d_o.as_mut_ptr(), N, KH, VH, D, D, stream)?;
    let gpu = d_o.to_vec(stream)?;
    let gpu_s = d_s.to_vec(stream)?;
    let range = |x: &[f32]| x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
    let (abs, _) = compare(&cpu, &gpu);
    let cpu_s: Vec<f32> = st.iter().map(|&x| x as f32).collect();
    let (sabs, _) = compare(&cpu_s, &gpu_s);
    let (ro, rs) = (abs / range(&cpu), sabs / range(&cpu_s));
    report.case("delta rule, chunked (tensor cores)",
        format!("output err {ro:.2e} of range  state err {rs:.2e} of range"), ro <= 1e-2 && rs <= 1e-2);
    Ok(())
}

fn test_delta_rule(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N_TOKENS: usize = 9;
    const N_K_HEADS: usize = 4;
    const N_V_HEADS: usize = 8;
    const HEAD_K: usize = 64;
    const HEAD_V: usize = 64;

    let mut rng = Rng::new(0x14_0000);
    let q = rng.floats(N_TOKENS * N_K_HEADS * HEAD_K);
    let k = rng.floats(N_TOKENS * N_K_HEADS * HEAD_K);
    let v = rng.floats(N_TOKENS * N_V_HEADS * HEAD_V);
    // Decay in (0,1) and beta in (0,1), as the gate kernel produces.
    let decay: Vec<f32> = (0..N_TOKENS * N_V_HEADS)
        .map(|_| 0.5 + rng.unit().abs() * 0.49)
        .collect();
    let beta: Vec<f32> = (0..N_TOKENS * N_V_HEADS)
        .map(|_| rng.unit().abs().max(0.01))
        .collect();

    // CPU reference, transcribed from torch_recurrent_gated_delta_rule.
    let mut state = vec![0f64; N_V_HEADS * HEAD_K * HEAD_V];
    let mut cpu = vec![0f32; N_TOKENS * N_V_HEADS * HEAD_V];
    // GGUF broadcasts the complete K-head axis (0,1,0,1), not adjacent pairs.
    for t in 0..N_TOKENS {
        for vh in 0..N_V_HEADS {
            let kh = vh % N_K_HEADS;
            let s = &mut state[vh * HEAD_K * HEAD_V..(vh + 1) * HEAD_K * HEAD_V];
            let d = f64::from(decay[t * N_V_HEADS + vh]);
            let b = f64::from(beta[t * N_V_HEADS + vh]);
            let qv = &q[(t * N_K_HEADS + kh) * HEAD_K..][..HEAD_K];
            let kv = &k[(t * N_K_HEADS + kh) * HEAD_K..][..HEAD_K];
            let vv = &v[(t * N_V_HEADS + vh) * HEAD_V..][..HEAD_V];

            for x in s.iter_mut() {
                *x *= d;
            }
            let mut mem = vec![0f64; HEAD_V];
            for dk in 0..HEAD_K {
                for dv in 0..HEAD_V {
                    mem[dv] += s[dk * HEAD_V + dv] * f64::from(kv[dk]);
                }
            }
            for dk in 0..HEAD_K {
                for dv in 0..HEAD_V {
                    let delta = (f64::from(vv[dv]) - mem[dv]) * b;
                    s[dk * HEAD_V + dv] += f64::from(kv[dk]) * delta;
                }
            }
            let out = &mut cpu[(t * N_V_HEADS + vh) * HEAD_V..][..HEAD_V];
            for dk in 0..HEAD_K {
                for dv in 0..HEAD_V {
                    out[dv] += (s[dk * HEAD_V + dv] * f64::from(qv[dk])) as f32;
                }
            }
        }
    }

    let d_q = DeviceBuffer::<f32>::from_slice(device, &q, stream)?;
    let d_k = DeviceBuffer::<f32>::from_slice(device, &k, stream)?;
    let d_v = DeviceBuffer::<f32>::from_slice(device, &v, stream)?;
    let d_decay = DeviceBuffer::<f32>::from_slice(device, &decay, stream)?;
    let d_beta = DeviceBuffer::<f32>::from_slice(device, &beta, stream)?;
    let d_state = DeviceBuffer::<f32>::zeroed(device, N_V_HEADS * HEAD_K * HEAD_V, stream)?;
    let d_out = DeviceBuffer::<f32>::new(device, N_TOKENS * N_V_HEADS * HEAD_V)?;

    airun_cuda::hybrid::delta_rule(
        d_state.as_mut_ptr(),
        d_q.as_ptr(),
        d_k.as_ptr(),
        d_v.as_ptr(),
        d_decay.as_ptr(),
        d_beta.as_ptr(),
        d_out.as_mut_ptr(),
        N_TOKENS,
        N_K_HEADS,
        N_V_HEADS,
        HEAD_K,
        HEAD_V,
        stream,
    )?;
    let gpu = d_out.to_vec(stream)?;

    let (abs, rel) = compare(&cpu, &gpu);
    report.case(
        "delta rule (batched)",
        format!("max_abs {abs:.3e}  max_rel {rel:.3e}"),
        rel <= 1e-3,
    );

    // The recurrence must be identical whether tokens arrive together or one at
    // a time -- this is exactly the prefill/decode consistency that matters.
    let d_state2 = DeviceBuffer::<f32>::zeroed(device, N_V_HEADS * HEAD_K * HEAD_V, stream)?;
    let d_out2 = DeviceBuffer::<f32>::new(device, N_TOKENS * N_V_HEADS * HEAD_V)?;
    for t in 0..N_TOKENS {
        airun_cuda::hybrid::delta_rule(
            d_state2.as_mut_ptr(),
            d_q.offset_ptr(t * N_K_HEADS * HEAD_K),
            d_k.offset_ptr(t * N_K_HEADS * HEAD_K),
            d_v.offset_ptr(t * N_V_HEADS * HEAD_V),
            d_decay.offset_ptr(t * N_V_HEADS),
            d_beta.offset_ptr(t * N_V_HEADS),
            d_out2.offset_ptr(t * N_V_HEADS * HEAD_V),
            1,
            N_K_HEADS,
            N_V_HEADS,
            HEAD_K,
            HEAD_V,
            stream,
        )?;
    }
    let stepwise = d_out2.to_vec(stream)?;
    let (sabs, srel) = compare(&gpu, &stepwise);
    report.case(
        "delta rule streaming == batched",
        format!("max_abs {sabs:.3e}  max_rel {srel:.3e}"),
        sabs <= 1e-5,
    );
    Ok(())
}

fn test_rmsnorm_gated(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const ROWS: usize = 16;
    const N: usize = 128;
    const EPS: f32 = 1e-6;

    let mut rng = Rng::new(0x15_0000);
    let x = rng.floats(ROWS * N);
    let w = rng.floats(N);
    let gate = rng.floats(ROWS * N);

    let silu = |v: f32| v / (1.0 + (-v).exp());

    // The reference gates first, so the gate is part of what gets measured. The
    // two orderings produce different numbers from the same shapes, which is
    // why both are pinned here rather than trusted to the runtime.
    let reference = |gate_first: bool| -> Vec<f32> {
        let mut out = vec![0f32; ROWS * N];
        for r in 0..ROWS {
            let row: Vec<f32> = (0..N)
                .map(|i| {
                    let v = x[r * N + i];
                    if gate_first { v * silu(gate[r * N + i]) } else { v }
                })
                .collect();
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / N as f32 + EPS).sqrt();
            for i in 0..N {
                let v = row[i] * inv * w[i];
                out[r * N + i] = if gate_first { v } else { v * silu(gate[r * N + i]) };
            }
        }
        out
    };

    let d_w = DeviceBuffer::<f32>::from_slice(device, &w, stream)?;
    let d_g = DeviceBuffer::<f32>::from_slice(device, &gate, stream)?;
    for gate_first in [true, false] {
        let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
        airun_cuda::hybrid::rmsnorm_gated(
            d_x.as_mut_ptr(),
            d_w.as_ptr(),
            d_g.as_ptr(),
            N,
            ROWS,
            EPS,
            gate_first,
            false,
            stream,
        )?;
        let gpu = d_x.to_vec(stream)?;

        let (abs, rel) = compare(&reference(gate_first), &gpu);
        report.case(
            if gate_first {
                "rmsnorm gated (gate before norm, the reference order)"
            } else {
                "rmsnorm gated (gate after norm)"
            },
            format!("max_abs {abs:.3e}  max_rel {rel:.3e}"),
            rel <= 1e-4,
        );
    }

    // A guard against the two branches quietly collapsing into one: they must
    // disagree, or the flag is not doing anything.
    let (spread, _) = compare(&reference(true), &reference(false));
    report.case(
        "the two gate orderings are actually different",
        format!("max_abs {spread:.3e}"),
        spread > 1e-3,
    );
    Ok(())
}

fn test_split_head_gate(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N_TOKENS: usize = 3;
    const HEADS: usize = 4;
    const DIM: usize = 8;

    // Encode position so a wrong interleave is unmistakable.
    let src: Vec<f32> = (0..N_TOKENS * HEADS * 2 * DIM).map(|i| i as f32).collect();
    let d_src = DeviceBuffer::<f32>::from_slice(device, &src, stream)?;
    let d_val = DeviceBuffer::<f32>::new(device, N_TOKENS * HEADS * DIM)?;
    let d_gate = DeviceBuffer::<f32>::new(device, N_TOKENS * HEADS * DIM)?;

    airun_cuda::hybrid::split_head_gate(
        d_src.as_ptr(),
        d_val.as_mut_ptr(),
        d_gate.as_mut_ptr(),
        N_TOKENS,
        HEADS,
        DIM,
        true,
        stream,
    )?;
    let val = d_val.to_vec(stream)?;
    let gate = d_gate.to_vec(stream)?;

    // Each head's slice is [value(DIM), gate(DIM)].
    let mut ok = true;
    for t in 0..N_TOKENS {
        for h in 0..HEADS {
            for d in 0..DIM {
                let base = (t * HEADS + h) * 2 * DIM;
                ok &= val[(t * HEADS + h) * DIM + d] == src[base + d];
                ok &= gate[(t * HEADS + h) * DIM + d] == src[base + DIM + d];
            }
        }
    }
    report.case("split head/gate interleaved", format!("exact match: {ok}"), ok);

    // Blocked layout: all values, then all gates.
    airun_cuda::hybrid::split_head_gate(
        d_src.as_ptr(),
        d_val.as_mut_ptr(),
        d_gate.as_mut_ptr(),
        N_TOKENS,
        HEADS,
        DIM,
        false,
        stream,
    )?;
    let val_b = d_val.to_vec(stream)?;
    let gate_b = d_gate.to_vec(stream)?;
    let width = HEADS * DIM;
    let mut ok_b = true;
    for t in 0..N_TOKENS {
        for i in 0..width {
            ok_b &= val_b[t * width + i] == src[t * 2 * width + i];
            ok_b &= gate_b[t * width + i] == src[t * 2 * width + width + i];
        }
    }
    report.case("split head/gate blocked", format!("exact match: {ok_b}"), ok_b);
    Ok(())
}

fn test_moe_router(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N_TOKENS: usize = 5;
    const N_EXPERTS: usize = 64;
    const TOP_K: usize = 8;

    let mut rng = Rng::new(0x16_0000);
    let logits: Vec<f32> = (0..N_TOKENS * N_EXPERTS).map(|_| rng.unit() * 3.0).collect();

    let d_l = DeviceBuffer::<f32>::from_slice(device, &logits, stream)?;
    let d_ids = DeviceBuffer::<i32>::new(device, N_TOKENS * TOP_K)?;
    let d_w = DeviceBuffer::<f32>::new(device, N_TOKENS * TOP_K)?;
    airun_cuda::hybrid::moe_router(
        d_l.as_ptr(),
        d_ids.as_mut_ptr(),
        d_w.as_mut_ptr(),
        N_TOKENS,
        N_EXPERTS,
        TOP_K,
        true,
        stream,
    )?;
    let ids = d_ids.to_vec(stream)?;
    let weights = d_w.to_vec(stream)?;

    let mut ok = true;
    let mut worst_weight_err = 0f32;
    for t in 0..N_TOKENS {
        let row = &logits[t * N_EXPERTS..(t + 1) * N_EXPERTS];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut order: Vec<usize> = (0..N_EXPERTS).collect();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let expected: Vec<i32> = order[..TOP_K].iter().map(|&i| i as i32).collect();
        let got = &ids[t * TOP_K..(t + 1) * TOP_K];
        ok &= got == expected.as_slice();

        // Selected weights must be renormalized to sum to one.
        let total: f32 = weights[t * TOP_K..(t + 1) * TOP_K].iter().sum();
        worst_weight_err = worst_weight_err.max((total - 1.0).abs());

        let denom: f32 = expected.iter().map(|&e| probs[e as usize]).sum();
        for (j, &e) in expected.iter().enumerate() {
            let want = probs[e as usize] / denom;
            worst_weight_err = worst_weight_err.max((weights[t * TOP_K + j] - want).abs());
        }
    }

    report.case(
        "moe router (softmax + top-k)",
        format!("expert ids match: {ok}  weight err {worst_weight_err:.3e}"),
        ok && worst_weight_err <= 1e-5,
    );
    Ok(())
}

fn test_moe_ffn(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const N_EMBD: usize = 128;
    // 128 wide so the tensor-core grouped kernels (128-wide slices) take it.
    const FF: usize = 128;
    const N_EXPERTS: usize = 6;
    const TOP_K: usize = 2;
    const N_TOKENS: usize = 4;

    let ty = GgmlType::Q8_0;
    let layout = ty.layout().unwrap();
    let elems = layout.elements as usize;
    let mut rng = Rng::new(0x17_0000);

    // Expert stacks: [n_experts][rows][row_bytes], contiguous per expert.
    let gate_row_bytes = (N_EMBD / elems) * layout.bytes as usize;
    let down_row_bytes = (FF / elems) * layout.bytes as usize;
    let gate_stride = FF * gate_row_bytes;
    let down_stride = N_EMBD * down_row_bytes;

    let mut gate_w = Vec::new();
    let mut up_w = Vec::new();
    let mut down_w = Vec::new();
    for _ in 0..N_EXPERTS {
        gate_w.extend_from_slice(&random_blocks(&mut rng, ty, FF * N_EMBD / elems));
        up_w.extend_from_slice(&random_blocks(&mut rng, ty, FF * N_EMBD / elems));
        down_w.extend_from_slice(&random_blocks(&mut rng, ty, N_EMBD * FF / elems));
    }

    let x = rng.floats(N_TOKENS * N_EMBD);
    let ids: Vec<i32> = (0..N_TOKENS * TOP_K)
        .map(|i| ((i * 3 + 1) % N_EXPERTS) as i32)
        .collect();
    let weights: Vec<f32> = (0..N_TOKENS * TOP_K).map(|i| 0.3 + 0.1 * (i % 4) as f32).collect();

    // CPU reference.
    let decode_row = |packed: &[u8], base: usize, k: usize| -> Vec<f32> {
        let mut out = vec![0f32; k];
        dequant::dequantize(ty, &packed[base..], k, &mut out).unwrap();
        out
    };

    let mut cpu = vec![0f32; N_TOKENS * N_EMBD];
    for t in 0..N_TOKENS {
        let xt = &x[t * N_EMBD..(t + 1) * N_EMBD];
        for slot in 0..TOP_K {
            let e = ids[t * TOP_K + slot] as usize;
            let w = weights[t * TOP_K + slot];

            let mut h = vec![0f32; FF];
            for r in 0..FF {
                let g = decode_row(&gate_w, e * gate_stride + r * gate_row_bytes, N_EMBD);
                let u = decode_row(&up_w, e * gate_stride + r * gate_row_bytes, N_EMBD);
                let gd: f32 = g.iter().zip(xt).map(|(a, b)| a * b).sum();
                let ud: f32 = u.iter().zip(xt).map(|(a, b)| a * b).sum();
                h[r] = (gd / (1.0 + (-gd).exp())) * ud;
            }
            for r in 0..N_EMBD {
                let d = decode_row(&down_w, e * down_stride + r * down_row_bytes, FF);
                let dot: f32 = d.iter().zip(&h).map(|(a, b)| a * b).sum();
                cpu[t * N_EMBD + r] += w * dot;
            }
        }
    }

    let d_gate = DeviceBuffer::<u8>::from_slice(device, &gate_w, stream)?;
    let d_up = DeviceBuffer::<u8>::from_slice(device, &up_w, stream)?;
    let d_down = DeviceBuffer::<u8>::from_slice(device, &down_w, stream)?;
    let d_x = DeviceBuffer::<f32>::from_slice(device, &x, stream)?;
    let d_ids = DeviceBuffer::<i32>::from_slice(device, &ids, stream)?;
    let d_w = DeviceBuffer::<f32>::from_slice(device, &weights, stream)?;
    let d_h = DeviceBuffer::<f32>::new(device, N_TOKENS * TOP_K * FF)?;
    let d_y = DeviceBuffer::<f32>::zeroed(device, N_TOKENS * N_EMBD, stream)?;

    airun_cuda::hybrid::moe_gate_up(
        ops::KernelType(ty.id() as i32),
        d_gate.as_ptr(),
        d_up.as_ptr(),
        d_x.as_ptr(),
        d_ids.as_ptr(),
        d_h.as_mut_ptr(),
        N_EMBD,
        FF,
        N_TOKENS,
        TOP_K,
        gate_row_bytes,
        gate_stride,
        stream,
    )?;
    airun_cuda::hybrid::moe_down(
        ops::KernelType(ty.id() as i32),
        d_down.as_ptr(),
        d_h.as_ptr(),
        d_ids.as_ptr(),
        d_w.as_ptr(),
        d_y.as_mut_ptr(),
        FF,
        N_EMBD,
        N_TOKENS,
        TOP_K,
        down_row_bytes,
        down_stride,
        stream,
    )?;
    let gpu = d_y.to_vec(stream)?;

    let (abs, rel) = compare(&cpu, &gpu);
    // Two all-zero arrays would compare perfectly and prove nothing, so the
    // output magnitude is part of the assertion.
    let peak = scale_of(&cpu).max(scale_of(&gpu));
    report.case(
        "moe expert ffn (gate/up/down)",
        format!("max_abs {abs:.3e}  max_rel {rel:.3e}  output peak {peak:.3e}"),
        rel <= 1e-3 && peak > 1e-3,
    );

    // Routing must actually matter: sending every token to a different expert
    // has to change the answer, or the expert offset is being ignored.
    let shifted: Vec<i32> = ids.iter().map(|&e| (e + 1) % N_EXPERTS as i32).collect();
    let d_ids2 = DeviceBuffer::<i32>::from_slice(device, &shifted, stream)?;
    let d_y2 = DeviceBuffer::<f32>::zeroed(device, N_TOKENS * N_EMBD, stream)?;
    airun_cuda::hybrid::moe_gate_up(
        ops::KernelType(ty.id() as i32),
        d_gate.as_ptr(), d_up.as_ptr(), d_x.as_ptr(), d_ids2.as_ptr(), d_h.as_mut_ptr(),
        N_EMBD, FF, N_TOKENS, TOP_K, gate_row_bytes, gate_stride, stream,
    )?;
    airun_cuda::hybrid::moe_down(
        ops::KernelType(ty.id() as i32),
        d_down.as_ptr(), d_h.as_ptr(), d_ids2.as_ptr(), d_w.as_ptr(), d_y2.as_mut_ptr(),
        FF, N_EMBD, N_TOKENS, TOP_K, down_row_bytes, down_stride, stream,
    )?;
    let other = d_y2.to_vec(stream)?;
    let (diff, _) = compare(&gpu, &other);
    report.case(
        "moe expert selection changes output",
        format!("delta between expert sets {diff:.3e}"),
        diff > 1e-3,
    );

    // The expert-major batch path against the token-major kernels: expert 0
    // takes every token's first slot, more than one chunk of pairs, and the
    // other slots spread over the rest. 20 tokens take the f32 grouped
    // kernels, 80 the tensor-core ones.
    for n2 in [20usize, 80] {
    let x2 = rng.floats(n2 * N_EMBD);
    let ids2: Vec<i32> = (0..n2 * TOP_K)
        .map(|i| if i % TOP_K == 0 { 0 } else { ((i / TOP_K) % (N_EXPERTS - 1) + 1) as i32 })
        .collect();
    let w2: Vec<f32> = (0..n2 * TOP_K).map(|i| 0.2 + 0.05 * (i % 7) as f32).collect();
    let d_x2 = DeviceBuffer::<f32>::from_slice(device, &x2, stream)?;
    let d_ids2 = DeviceBuffer::<i32>::from_slice(device, &ids2, stream)?;
    let d_w2 = DeviceBuffer::<f32>::from_slice(device, &w2, stream)?;
    let d_h2 = DeviceBuffer::<f32>::new(device, n2 * TOP_K * FF)?;
    let kt = ops::KernelType(ty.id() as i32);
    let token_major = DeviceBuffer::<f32>::zeroed(device, n2 * N_EMBD, stream)?;
    airun_cuda::hybrid::moe_gate_up(kt, d_gate.as_ptr(), d_up.as_ptr(), d_x2.as_ptr(), d_ids2.as_ptr(),
        d_h2.as_mut_ptr(), N_EMBD, FF, n2, TOP_K, gate_row_bytes, gate_stride, stream)?;
    airun_cuda::hybrid::moe_down(kt, d_down.as_ptr(), d_h2.as_ptr(), d_ids2.as_ptr(), d_w2.as_ptr(),
        token_major.as_mut_ptr(), FF, N_EMBD, n2, TOP_K, down_row_bytes, down_stride, stream)?;
    let grouped = DeviceBuffer::<f32>::zeroed(device, n2 * N_EMBD, stream)?;
    let scratch = DeviceBuffer::<i32>::new(device, N_EXPERTS)?;
    let offsets = DeviceBuffer::<i32>::new(device, N_EXPERTS + 1)?;
    let perm = DeviceBuffer::<i32>::new(device, n2 * TOP_K)?;
    let ypair = DeviceBuffer::<f32>::new(device, n2 * TOP_K * N_EMBD)?;
    let xh = DeviceBuffer::<u16>::new(device, n2 * N_EMBD)?;
    airun_cuda::hybrid::moe_sort(d_ids2.as_ptr(), n2 * TOP_K, N_EXPERTS, scratch.as_mut_ptr(), offsets.as_mut_ptr(),
        perm.as_mut_ptr(), stream)?;
    let ew = |base: *const std::ffi::c_void, stride: usize, row_bytes: usize| airun_cuda::hybrid::ExpertWeights {
        ty: kt, tab: std::ptr::null(), base, stride, row_bytes, resident: std::ptr::null() };
    airun_cuda::hybrid::moe_grouped(ew(d_gate.as_ptr(), gate_stride, gate_row_bytes),
        ew(d_up.as_ptr(), gate_stride, gate_row_bytes), ew(d_down.as_ptr(), down_stride, down_row_bytes),
        d_x2.as_ptr(), perm.as_ptr(), offsets.as_ptr(), d_w2.as_ptr(), d_h2.as_mut_ptr(), ypair.as_mut_ptr(),
        grouped.as_mut_ptr(), N_EMBD, FF, n2, TOP_K, N_EXPERTS, xh.as_mut_ptr(), None, stream)?;
    let (a, b) = (token_major.to_vec(stream)?, grouped.to_vec(stream)?);
    // On sm_80+ the grouped kernels multiply in fp16 on the tensor cores
    // (f32 accumulation), so they agree with the f32 token-major kernels to
    // fp16 rounding; elsewhere only the order of the sums differs.
    let (abs, _) = compare(&a, &b);
    let peak = scale_of(&a);
    report.case(&format!("moe expert-major batch vs token-major, {n2} tokens"),
        format!("max_abs {abs:.3e}, peak {peak:.3e}"), abs <= peak * 2e-3 && peak > 1e-3);
    }
    Ok(())
}

fn test_attention(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    // Cover the shapes real models actually use. head_dim 256 with GQA 4x is
    // Qwen3.5's attention layer and sits exactly at the kernel's per-lane limit,
    // which is where an off-by-one in the register blocking would show up.
    let few = [40, 55, 70];
    attention_case(device, stream, report, 8, 2, 64, "head_dim 64, GQA 4x", &few)?;
    attention_case(device, stream, report, 16, 4, 256, "head_dim 256, GQA 4x", &few)?;
    attention_case(device, stream, report, 16, 2, 128, "head_dim 128, GQA 8x", &few)?;
    // 50 contiguous positions, as a prefill chunk arrives: the tensor-core
    // (flash) kernel on sm_80+, with a partial last query tile.
    let chunk: Vec<i32> = (20..70).collect();
    attention_case(device, stream, report, 8, 2, 64, "head_dim 64, 50-token prefill", &chunk)?;
    attention_case(device, stream, report, 16, 2, 128, "head_dim 128, 50-token prefill", &chunk)?;
    attention_case(device, stream, report, 16, 4, 256, "head_dim 256, 50-token prefill", &chunk)?;
    test_split_attention_long(device, stream, report)?;
    test_qsa(device, stream, report)?;
    test_fused_prologue(device, stream, report)?;
    test_argmax(device, stream, report)?;
    for kv in [ops::KvType::Q8_0, ops::KvType::Q4_0] {
        test_kv_quant(device, stream, report, kv)?;
    }
    Ok(())
}

/// GPU argmax against the sampler's rule: first maximum wins, NaN never does,
/// and index 0 when nothing beats -inf.
fn test_argmax(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    let cpu = |v: &[f32]| -> i32 {
        let (mut best, mut val) = (0i32, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > val { val = x; best = i as i32; }
        }
        best
    };
    let mut rng = Rng::new(0xA7_6000);
    let mut cases: Vec<Vec<f32>> = Vec::new();
    cases.push(rng.floats(151_936));
    let mut tie = rng.floats(100_000);
    tie[777] = 5.0; tie[50_001] = 5.0; tie[99_999] = 5.0;   // first of equal maxima
    cases.push(tie);
    let mut nan = rng.floats(4097);
    nan[10] = f32::NAN; nan[11] = 3.0;
    cases.push(nan);
    cases.push(vec![f32::NEG_INFINITY; 300]);
    cases.push(vec![-1.0]);
    // Maxima in the last element and in the first, where the two passes split.
    let mut last = rng.floats(151_936);
    last[151_935] = 9.0;
    cases.push(last);
    let mut first = rng.floats(70_000);
    first[0] = 9.0;
    cases.push(first);
    let out = DeviceBuffer::<i32>::new(device, 1)?;
    let scratch = DeviceBuffer::<i32>::new(device, ops::ARGMAX_SCRATCH)?;
    let mut bad = 0;
    for c in &cases {
        let d = DeviceBuffer::from_slice(device, c, stream)?;
        ops::argmax(d.as_ptr(), c.len(), out.as_mut_ptr(), scratch.as_mut_ptr(), std::ptr::null_mut(), stream)?;
        if out.to_vec(stream)?[0] != cpu(c) { bad += 1; }
    }
    report.case("argmax on the GPU matches the sampler", format!("{bad} of {} cases differ", cases.len()), bad == 0);

    // The chained decode graph's position bump: +1 per call, after the pick.
    let bump = DeviceBuffer::<i32>::from_slice(device, &[41], stream)?;
    let d = DeviceBuffer::from_slice(device, &cases[0], stream)?;
    ops::argmax(d.as_ptr(), cases[0].len(), out.as_mut_ptr(), scratch.as_mut_ptr(), bump.as_mut_ptr(), stream)?;
    let bumped = bump.to_vec(stream)?[0];
    report.case("argmax position bump", format!("41 -> {bumped}"), bumped == 42 && out.to_vec(stream)?[0] == cpu(&cases[0]));
    Ok(())
}
/// The fused attention prologue must reproduce rmsnorm, rope and kv_store run
/// one after another, bit for bit: same q, same cache bytes.
fn test_fused_prologue(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    const HEADS: usize = 8;
    const KV_HEADS: usize = 2;
    const DIM: usize = 128;
    const MAX_SEQ: usize = 64;
    const TOKENS: usize = 5;
    let mut rng = Rng::new(0x9E_0000);
    let q = rng.floats(TOKENS * HEADS * DIM);
    let k = rng.floats(TOKENS * KV_HEADS * DIM);
    let v = rng.floats(TOKENS * KV_HEADS * DIM);
    let qn: Vec<f32> = (0..DIM).map(|_| 1.0 + rng.unit() * 0.5).collect();
    let kn: Vec<f32> = (0..DIM).map(|_| 1.0 + rng.unit() * 0.5).collect();
    let pos: Vec<i32> = vec![3, 17, 40, 41, 63];
    let d_qn = DeviceBuffer::from_slice(device, &qn, stream)?;
    let d_kn = DeviceBuffer::from_slice(device, &kn, stream)?;
    let d_pos = DeviceBuffer::from_slice(device, &pos, stream)?;
    // (rope_dim, interleaved): full NeoX, partial (the hybrids' 64 of 128), GPT-J.
    for kv in [ops::KvType::F16, ops::KvType::Q8_0, ops::KvType::Q4_0] {
        for (rope_dim, inter) in [(DIM, false), (64, false), (DIM, true)] {
            let row = kv.row_bytes(DIM);
            let run = |fused: bool| -> Result<(Vec<f32>, Vec<u8>, Vec<u8>)> {
                let d_q = DeviceBuffer::from_slice(device, &q, stream)?;
                let d_k = DeviceBuffer::from_slice(device, &k, stream)?;
                let d_v = DeviceBuffer::from_slice(device, &v, stream)?;
                let kc = DeviceBuffer::<u8>::zeroed(device, KV_HEADS * MAX_SEQ * row, stream)?;
                let vc = DeviceBuffer::<u8>::zeroed(device, KV_HEADS * MAX_SEQ * row, stream)?;
                let rope = Some((rope_dim, ops::RopeParams::plain(1_000_000.0, 1.0), inter));
                if fused {
                    ops::qk_norm_rope_store(d_q.as_mut_ptr(), d_k.as_ptr(), d_v.as_ptr(),
                        d_qn.as_ptr(), d_kn.as_ptr(), kc.as_mut_ptr(), vc.as_mut_ptr(), d_pos.as_ptr(),
                        TOKENS, HEADS, KV_HEADS, DIM, MAX_SEQ, 1e-6, rope, kv, stream)?;
                } else {
                    ops::rmsnorm(d_q.as_mut_ptr(), d_q.as_ptr(), d_qn.as_ptr(), DIM, TOKENS * HEADS, 1e-6, stream)?;
                    ops::rmsnorm(d_k.as_mut_ptr(), d_k.as_ptr(), d_kn.as_ptr(), DIM, TOKENS * KV_HEADS, 1e-6, stream)?;
                    ops::rope(d_q.as_mut_ptr(), d_pos.as_ptr(), TOKENS, HEADS, DIM, rope_dim, ops::RopeParams::plain(1_000_000.0, 1.0), inter, stream)?;
                    ops::rope(d_k.as_mut_ptr(), d_pos.as_ptr(), TOKENS, KV_HEADS, DIM, rope_dim, ops::RopeParams::plain(1_000_000.0, 1.0), inter, stream)?;
                    ops::kv_store(kc.as_mut_ptr(), vc.as_mut_ptr(), d_k.as_ptr(), d_v.as_ptr(), d_pos.as_ptr(),
                        TOKENS, KV_HEADS, DIM, MAX_SEQ, kv, stream)?;
                }
                Ok((d_q.to_vec(stream)?, kc.to_vec(stream)?, vc.to_vec(stream)?))
            };
            let (qa, ka, va) = run(false)?;
            let (qb, kb, vb) = run(true)?;
            let q_diff = qa.iter().zip(&qb).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let c_diff = ka.iter().zip(&kb).filter(|(a, b)| a != b).count()
                + va.iter().zip(&vb).filter(|(a, b)| a != b).count();
            report.case(&format!("fused q/k norm+rope+kv store ({}, rope {rope_dim}{})", kv.name(),
                    if inter { ", interleaved" } else { "" }),
                format!("{q_diff} q values and {c_diff} cache bytes differ"), q_diff == 0 && c_diff == 0);
        }
    }
    Ok(())
}
/// A quantized KV cache is only as good as two mirrors: the GPU encoder must
/// write the bytes quantize.rs writes, and attention must read them the way
/// dequant.rs decodes them. The second check starts from CPU-encoded caches so
/// a bug in one side cannot be cancelled by the same bug in the other.
fn test_kv_quant(device: i32, stream: &Stream, report: &mut Report, kv: ops::KvType) -> Result<()> {
    const N_HEADS: usize = 8;
    const N_KV: usize = 2;
    const DIM: usize = 128;
    const MAX_SEQ: usize = 96;
    let ty = match kv { ops::KvType::Q8_0 => GgmlType::Q8_0, _ => GgmlType::Q4_0 };
    let row = kv.row_bytes(DIM);
    let mut rng = Rng::new(0x4B56_0000 + row as u64);

    // Encoder: three tokens, including an all-zero block, a negative extreme
    // and a tie in magnitude, which is where the Q4_0 scale choice can differ.
    let positions = [5i32, 40, 95];
    let mut k_new = rng.floats(3 * N_KV * DIM);
    let v_new = rng.floats(3 * N_KV * DIM);
    k_new[..32].fill(0.0);
    k_new[32] = -3.0;
    k_new[64] = 2.0;
    k_new[70] = -2.0;
    let d_kn = DeviceBuffer::from_slice(device, &k_new, stream)?;
    let d_vn = DeviceBuffer::from_slice(device, &v_new, stream)?;
    let d_pos = DeviceBuffer::from_slice(device, &positions, stream)?;
    let d_kc = DeviceBuffer::<u8>::zeroed(device, N_KV * MAX_SEQ * row, stream)?;
    let d_vc = DeviceBuffer::<u8>::zeroed(device, N_KV * MAX_SEQ * row, stream)?;
    ops::kv_store(d_kc.as_mut_ptr(), d_vc.as_mut_ptr(), d_kn.as_ptr(), d_vn.as_ptr(),
        d_pos.as_ptr(), 3, N_KV, DIM, MAX_SEQ, kv, stream)?;
    let (kc, vc) = (d_kc.to_vec(stream)?, d_vc.to_vec(stream)?);
    let mut mismatched = 0usize;
    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..N_KV {
            for (src, cache) in [(&k_new, &kc), (&v_new, &vc)] {
                let mut want = Vec::new();
                airun_gguf::quantize(ty, &src[(t * N_KV + h) * DIM..][..DIM], DIM, &mut want)?;
                let got = &cache[(h * MAX_SEQ + pos as usize) * row..][..row];
                mismatched += want.iter().zip(got).filter(|(a, b)| a != b).count();
            }
        }
    }
    report.case(&format!("kv_store {} matches quantize.rs", kv.name()),
        format!("{mismatched} mismatched bytes"), mismatched == 0);

    // Attention: caches encoded on the CPU, reference decoded by dequant.rs.
    let k_raw = rng.floats(N_KV * MAX_SEQ * DIM);
    let v_raw = rng.floats(N_KV * MAX_SEQ * DIM);
    let (mut kq, mut vq) = (Vec::new(), Vec::new());
    airun_gguf::quantize(ty, &k_raw, DIM, &mut kq)?;
    airun_gguf::quantize(ty, &v_raw, DIM, &mut vq)?;
    let (mut k, mut v) = (vec![0f32; k_raw.len()], vec![0f32; v_raw.len()]);
    dequant::dequantize(ty, &kq, k.len(), &mut k)?;
    dequant::dequantize(ty, &vq, v.len(), &mut v)?;

    let q = rng.floats(N_HEADS * DIM);
    let pos = 77usize;
    let scale = 1.0 / (DIM as f32).sqrt();
    let mut cpu = vec![0f32; N_HEADS * DIM];
    for h in 0..N_HEADS {
        let kvh = h / (N_HEADS / N_KV);
        let qv = &q[h * DIM..][..DIM];
        let scores: Vec<f32> = (0..=pos).map(|j| {
            qv.iter().zip(&k[(kvh * MAX_SEQ + j) * DIM..][..DIM]).map(|(a, b)| a * b).sum::<f32>() * scale
        }).collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (j, e) in exps.iter().enumerate() {
            for d in 0..DIM {
                cpu[h * DIM + d] += e / sum * v[(kvh * MAX_SEQ + j) * DIM + d];
            }
        }
    }

    let d_q = DeviceBuffer::from_slice(device, &q, stream)?;
    let d_k = DeviceBuffer::from_slice(device, &kq, stream)?;
    let d_v = DeviceBuffer::from_slice(device, &vq, stream)?;
    let d_p = DeviceBuffer::from_slice(device, &[pos as i32], stream)?;
    let d_o = DeviceBuffer::<f32>::new(device, N_HEADS * DIM)?;
    ops::attention(d_o.as_mut_ptr(), d_q.as_ptr(), d_k.as_ptr(), d_v.as_ptr(), d_p.as_ptr(),
        1, N_HEADS, N_KV, DIM, MAX_SEQ, scale, 0, kv, stream)?;
    let (abs, _) = compare(&cpu, &d_o.to_vec(stream)?);
    let partial = DeviceBuffer::<f32>::new(device, N_HEADS * 16 * (DIM + 2))?;
    ops::attention_split(d_o.as_mut_ptr(), &partial, d_q.as_ptr(), d_k.as_ptr(), d_v.as_ptr(),
        d_p.as_ptr(), N_HEADS, N_KV, DIM, MAX_SEQ, scale, 0, 16, kv, stream)?;
    let (split_abs, _) = compare(&cpu, &d_o.to_vec(stream)?);
    report.case(&format!("attention over {} cache vs dequant.rs", kv.name()),
        format!("max_abs {abs:.3e}  split {split_abs:.3e}"), abs <= 1e-3 && split_abs <= 1e-3);

    // A 20-token prefill chunk over the same cache: the flash kernel's path.
    const CHUNK: usize = 20;
    let qc = rng.floats(CHUNK * N_HEADS * DIM);
    let pos_c: Vec<i32> = (58..58 + CHUNK as i32).collect();
    let mut want = vec![0f32; CHUNK * N_HEADS * DIM];
    for (t, &p) in pos_c.iter().enumerate() {
        for h in 0..N_HEADS {
            let kvh = h / (N_HEADS / N_KV);
            let qv = &qc[(t * N_HEADS + h) * DIM..][..DIM];
            let scores: Vec<f32> = (0..=p as usize).map(|j| {
                qv.iter().zip(&k[(kvh * MAX_SEQ + j) * DIM..][..DIM]).map(|(a, b)| a * b).sum::<f32>() * scale
            }).collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                for d in 0..DIM {
                    want[(t * N_HEADS + h) * DIM + d] += e / sum * v[(kvh * MAX_SEQ + j) * DIM + d];
                }
            }
        }
    }
    let d_qc = DeviceBuffer::from_slice(device, &qc, stream)?;
    let d_pc = DeviceBuffer::from_slice(device, &pos_c, stream)?;
    let d_oc = DeviceBuffer::<f32>::new(device, CHUNK * N_HEADS * DIM)?;
    ops::attention(d_oc.as_mut_ptr(), d_qc.as_ptr(), d_k.as_ptr(), d_v.as_ptr(), d_pc.as_ptr(),
        CHUNK, N_HEADS, N_KV, DIM, MAX_SEQ, scale, 0, kv, stream)?;
    let (chunk_abs, _) = compare(&want, &d_oc.to_vec(stream)?);
    report.case(&format!("{CHUNK}-token prefill attention over {} cache", kv.name()),
        format!("max_abs {chunk_abs:.3e}"), chunk_abs <= 3e-3);
    Ok(())
}

/// QSA sparse attention (qwen4exp) against a CPU model of llama.cpp's graph:
/// the pooled block keys, the block scores, the selected cells (ties to the
/// lower block, a partly taken block keeping its first cells) and attention
/// restricted to them, through both the batch mask and the decode list.
fn test_qsa(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    use airun_cuda::qsa;
    const DIM: usize = 128;
    const HEADS: usize = 4;
    const R: usize = 4;
    const TOP_K: usize = 64;
    const WIDTH: usize = TOP_K + R - 1;
    const CTX: usize = 512;
    const ROPE: usize = 64;
    const EPS: f32 = 1e-6;
    let rope = ops::RopeParams::plain(1e7, 1.0);
    let mut rng = Rng::new(0x05A_0001);

    // Two calls as two prefill chunks: positions 0..300, then 300..420.
    let raw = rng.floats(CTX * DIM);
    let norm: Vec<f32> = rng.floats(DIM).iter().map(|x| 1.0 + 0.5 * x).collect();
    let calls = [(0usize, 300usize), (300, 120)];
    let ring_len = 300 + 64;
    let d_ring = DeviceBuffer::<half::f16>::zeroed(device, ring_len * DIM, stream)?;
    let d_pooled = DeviceBuffer::<f32>::zeroed(device, CTX / R * DIM, stream)?;
    let d_norm = DeviceBuffer::from_slice(device, &norm, stream)?;
    for &(p0, n) in &calls {
        let pos: Vec<i32> = (p0 as i32..(p0 + n) as i32).collect();
        let d_pos = DeviceBuffer::from_slice(device, &pos, stream)?;
        let d_k = DeviceBuffer::from_slice(device, &raw[p0 * DIM..(p0 + n) * DIM], stream)?;
        qsa::store(d_ring.as_mut_ptr(), d_k.as_ptr(), d_pos.as_ptr(), n, DIM, ring_len, stream)?;
        qsa::pool(d_pooled.as_mut_ptr(), d_ring.as_ptr(), d_norm.as_ptr(), d_pos.as_ptr(), n, DIM, R, ring_len,
            CTX / R, EPS, ROPE, rope, stream)?;
    }
    let blocks = 420 / R;
    let mut want_pooled = vec![0f32; blocks * DIM];
    for b in 0..blocks {
        let mean: Vec<f32> = (0..DIM).map(|i| {
            let s: f32 = (0..R).map(|j| half::f16::from_f32(raw[(b * R + j) * DIM + i]).to_f32()).sum();
            s * (1.0 / R as f32)
        }).collect();
        let inv = 1.0 / (mean.iter().map(|x| x * x).sum::<f32>() / DIM as f32 + EPS).sqrt();
        let x: Vec<f32> = (0..DIM).map(|i| mean[i] * inv * norm[i]).collect();
        let out = &mut want_pooled[b * DIM..][..DIM];
        out.copy_from_slice(&x);
        for p in 0..ROPE / 2 {
            let (c, sn) = rope.cos_sin(p, ROPE, (b * R) as i32);
            out[p] = x[p] * c - x[p + ROPE / 2] * sn;
            out[p + ROPE / 2] = x[p] * sn + x[p + ROPE / 2] * c;
        }
    }
    let pooled = d_pooled.to_vec(stream)?;
    let (pabs, _) = compare(&want_pooled, &pooled[..blocks * DIM]);
    report.case("qsa pooled block keys across two calls", format!("max_abs {pabs:.3e}"), pabs <= 5e-4);

    // Scores and selection for the second call's 120 queries, 64 per launch.
    // Every other query is zero, so all its blocks tie at a score of 0.
    let n = 120;
    let pos: Vec<i32> = (300..420).collect();
    let d_pos = DeviceBuffer::from_slice(device, &pos, stream)?;
    let mut q = rng.floats(n * HEADS * DIM);
    for t in (0..n).step_by(2) {
        for x in &mut q[t * HEADS * DIM..][..HEADS * DIM] { *x = 0.0; }
    }
    let d_q = DeviceBuffer::from_slice(device, &q, stream)?;
    let max_blocks = CTX / R;
    let words = CTX / 32;
    let d_scores = DeviceBuffer::<f32>::new(device, 64 * max_blocks)?;
    let d_mask = DeviceBuffer::<u32>::zeroed(device, n * words, stream)?;
    let mut score_err = 0f32;
    let mut sel_ok = true;
    let mut selected: Vec<Vec<usize>> = Vec::new();
    for t0 in (0..n).step_by(64) {
        let g = 64.min(n - t0);
        qsa::score(d_scores.as_mut_ptr(), max_blocks, d_q.as_ptr(), d_pooled.as_ptr(), d_pos.as_ptr(), t0, g,
            HEADS, DIM, R, WIDTH, max_blocks, stream)?;
        qsa::select(d_scores.as_ptr(), max_blocks, d_pos.as_ptr(), t0, g, R, WIDTH, std::ptr::null_mut(),
            d_mask.as_mut_ptr(), words, stream)?;
        let sc = d_scores.to_vec(stream)?;
        for qi in 0..g {
            let t = t0 + qi;
            let p = pos[t] as usize;
            let nb = (p + 1) / R;
            let s = |b: usize| sc[qi * max_blocks + b];
            for b in 0..nb {
                let mut want = 0f32;
                for h in 0..HEADS {
                    let d: f32 = (0..DIM).map(|i| q[(t * HEADS + h) * DIM + i] * pooled[b * DIM + i]).sum();
                    want += d.max(0.0);
                }
                score_err = score_err.max((want - s(b)).abs());
            }
            // The selection the GPU's own scores call for.
            let mut order: Vec<usize> = (0..nb).collect();
            order.sort_by(|&a, &b| s(b).partial_cmp(&s(a)).unwrap().then(a.cmp(&b)));
            let tail = nb * R;
            let mut cells: Vec<usize> = (tail..=p).collect();
            let mut budget = WIDTH - cells.len();
            for &b in &order {
                let take = budget.min(R);
                cells.extend(b * R..b * R + take);
                budget -= take;
                if budget == 0 { break; }
            }
            cells.sort_unstable();
            selected.push(cells);
        }
    }
    let mask = d_mask.to_vec(stream)?;
    for (t, cells) in selected.iter().enumerate() {
        let p = pos[t] as usize;
        let got: Vec<usize> = (0..=p).filter(|&j| (mask[t * words + j / 32] >> (j % 32)) & 1 == 1).collect();
        sel_ok &= &got == cells;
    }
    report.case("qsa block scores", format!("max_abs {score_err:.3e}"), score_err <= 1e-4);
    report.case("qsa selection: tail + best blocks, ties to the lower block", format!("{n} queries"), sel_ok);

    // Attention over the selected cells: head_dim 256, GQA 4x, as qwen4exp.
    const NH: usize = 8;
    const NKV: usize = 2;
    const HD: usize = 256;
    let k_half: Vec<half::f16> = rng.floats(NKV * CTX * HD).iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_half: Vec<half::f16> = rng.floats(NKV * CTX * HD).iter().map(|&x| half::f16::from_f32(x)).collect();
    let qa = rng.floats(n * NH * HD);
    let scale = 1.0 / (HD as f32).sqrt();
    let attend = |t: usize, cells: &[usize]| -> Vec<f32> {
        let mut out = vec![0f32; NH * HD];
        for h in 0..NH {
            let kvh = h / (NH / NKV);
            let qv = &qa[(t * NH + h) * HD..][..HD];
            let s: Vec<f32> = cells.iter().map(|&j| qv.iter().zip(&k_half[(kvh * CTX + j) * HD..][..HD])
                .map(|(a, b)| a * b.to_f32()).sum::<f32>() * scale).collect();
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            for (ci, &j) in cells.iter().enumerate() {
                for d in 0..HD {
                    out[h * HD + d] += e[ci] / sum * v_half[(kvh * CTX + j) * HD + d].to_f32();
                }
            }
        }
        out
    };
    let d_k = DeviceBuffer::from_slice(device, &k_half, stream)?;
    let d_v = DeviceBuffer::from_slice(device, &v_half, stream)?;
    let d_qa = DeviceBuffer::from_slice(device, &qa, stream)?;
    let d_o = DeviceBuffer::<f32>::new(device, n * NH * HD)?;
    ops::attention_masked(d_o.as_mut_ptr(), d_qa.as_ptr(), d_k.as_ptr(), d_v.as_ptr(), d_pos.as_ptr(), n, NH, NKV,
        HD, CTX, scale, 0, ops::KvType::F16, Some((d_mask.as_ptr(), words)), 0, stream)?;
    let got = d_o.to_vec(stream)?;
    let want: Vec<f32> = (0..n).flat_map(|t| attend(t, &selected[t])).collect();
    let (aabs, _) = compare(&want, &got);
    report.case("qsa prefill attention through the mask (flash)", format!("max_abs {aabs:.3e}"), aabs <= 3e-3);
    // The same, as a batch deep in a long context would run it: P.V summed in
    // fp16 per key tile.
    ops::attention_masked(d_o.as_mut_ptr(), d_qa.as_ptr(), d_k.as_ptr(), d_v.as_ptr(), d_pos.as_ptr(), n, NH, NKV,
        HD, CTX, scale, 0, ops::KvType::F16, Some((d_mask.as_ptr(), words)), 1 << 20, stream)?;
    let got = d_o.to_vec(stream)?;
    let (pabs, _) = compare(&want, &got);
    report.case("qsa prefill attention, fp16 P.V tiles (long context)", format!("max_abs {pabs:.3e}"), pabs <= 3e-3);
    // A single query through the mask: attention_kernel's path.
    ops::attention_masked(d_o.as_mut_ptr(), d_qa.offset_ptr(7 * NH * HD), d_k.as_ptr(), d_v.as_ptr(),
        d_pos.offset_ptr(7), 1, NH, NKV, HD, CTX, scale, 0, ops::KvType::F16,
        Some((d_mask.offset_ptr(7 * words) as *const std::ffi::c_void, words)), 0, stream)?;
    let mut one = vec![0f32; NH * HD];
    d_o.copy_to_host(&mut one, stream)?;
    stream.synchronize()?;
    let (oabs, _) = compare(&attend(7, &selected[7]), &one);
    report.case("qsa single-query attention through the mask", format!("max_abs {oabs:.3e}"), oabs <= 1e-3);

    // Decode: one query through the list (the split kernel); position 50 fits
    // the budget and must attend densely.
    let d_list = DeviceBuffer::<i32>::new(device, 1 + WIDTH)?;
    let partial = DeviceBuffer::<f32>::new(device, NH * 16 * (HD + 2))?;
    let mut dec_err = 0f32;
    let mut list_ok = true;
    for (t, p) in [(119usize, 419i32), (118, 418), (5, 50)] {
        let d_p1 = DeviceBuffer::from_slice(device, &[p], stream)?;
        qsa::score(d_scores.as_mut_ptr(), max_blocks, d_q.offset_ptr(t * HEADS * DIM), d_pooled.as_ptr(),
            d_p1.as_ptr(), 0, 1, HEADS, DIM, R, WIDTH, max_blocks, stream)?;
        qsa::select(d_scores.as_ptr(), max_blocks, d_p1.as_ptr(), 0, 1, R, WIDTH, d_list.as_mut_ptr(),
            std::ptr::null_mut(), words, stream)?;
        ops::attention_split_sel(d_o.as_mut_ptr(), &partial, d_qa.offset_ptr(t * NH * HD), d_k.as_ptr(),
            d_v.as_ptr(), d_p1.as_ptr(), NH, NKV, HD, CTX, scale, 0, 16, ops::KvType::F16, d_list.as_ptr(), stream)?;
        let mut out = vec![0f32; NH * HD];
        d_o.copy_to_host(&mut out, stream)?;
        stream.synchronize()?;
        let cells: Vec<usize> = if p as usize >= WIDTH { selected[t].clone() } else { (0..=p as usize).collect() };
        let (e, _) = compare(&attend(t, &cells), &out);
        dec_err = dec_err.max(e);
        // The list itself: [count - 1, cells ascending].
        let list = d_list.to_vec(stream)?;
        let count = list[0] as usize + 1;
        list_ok &= list[1..1 + count].iter().map(|&x| x as usize).collect::<Vec<_>>() == cells;
    }
    report.case("qsa decode attention through the cell list (split kernel)",
        format!("max_abs {dec_err:.3e}"), dec_err <= 1e-3 && list_ok);
    Ok(())
}

fn test_split_attention_long(device: i32, stream: &Stream, report: &mut Report) -> Result<()> {
    // Zero queries give uniform attention: the answer is an exact arithmetic
    // mean, independently of both implementations of the softmax reduction.
    let ctx = 212_992usize;
    let dim = 128;
    let q = DeviceBuffer::from_slice(device, &vec![0.0f32; 2 * dim], stream)?;
    let k = DeviceBuffer::<half::f16>::zeroed(device, ctx * dim, stream)?;
    let values: Vec<half::f16> = (0..ctx * dim).map(|i|
        half::f16::from_f32(((i / dim) % 17) as f32 - 8.0)).collect();
    let v = DeviceBuffer::from_slice(device, &values, stream)?;
    let scratch = DeviceBuffer::new(device, 2 * 128 * (dim + 2))?;
    let out = DeviceBuffer::<f32>::new(device, 2 * dim)?;
    let mut worst = 0.0f32;
    for pos in [0usize, 1, 17, 3072, ctx - 1] {
        let p = DeviceBuffer::from_slice(device, &[pos as i32], stream)?;
        for window in [0usize, 1, 37] {
            ops::attention_split(out.as_mut_ptr(), &scratch, q.as_ptr(), k.as_ptr(), v.as_ptr(),
                p.as_ptr(), 2, 1, dim, ctx, 1.0, window, 128, ops::KvType::F16, stream)?;
            let begin = if window == 0 { 0 } else { (pos + 1).saturating_sub(window) };
            let expected = (begin..=pos).map(|j| (j % 17) as f64 - 8.0).sum::<f64>()
                / (pos + 1 - begin) as f64;
            for x in out.to_vec(stream)? { worst = worst.max((x - expected as f32).abs()); }
        }
    }
    report.case("split attention 212992 context, empty splits and sliding windows",
        format!("max_abs {worst:.3e}"), worst.is_finite() && worst < 1e-4);
    Ok(())
}

fn attention_case(
    device: i32,
    stream: &Stream,
    report: &mut Report,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    label: &str,
    positions: &[i32],
) -> Result<()> {
    let (N_HEADS, N_KV_HEADS, HEAD_DIM) = (n_heads, n_kv_heads, head_dim);
    const MAX_SEQ: usize = 96;
    let N_TOKENS = positions.len();
    // The flash kernel rounds Q and the probabilities to fp16.
    let tolerance = if N_TOKENS >= 16 { 3e-3 } else { 1e-3 };

    let mut rng = Rng::new(0xF6_6000);
    let q = rng.floats(N_TOKENS * N_HEADS * HEAD_DIM);
    let positions = positions.to_vec();
    let n_kv = *positions.iter().max().unwrap() as usize + 1;

    // The cache is f16 on device, so the reference must round-trip through f16
    // too or the comparison would measure storage precision, not the kernel.
    let k_raw = rng.floats(N_KV_HEADS * MAX_SEQ * HEAD_DIM);
    let v_raw = rng.floats(N_KV_HEADS * MAX_SEQ * HEAD_DIM);
    let k_half: Vec<half::f16> = k_raw.iter().map(|&v| half::f16::from_f32(v)).collect();
    let v_half: Vec<half::f16> = v_raw.iter().map(|&v| half::f16::from_f32(v)).collect();
    let k: Vec<f32> = k_half.iter().map(|v| v.to_f32()).collect();
    let v: Vec<f32> = v_half.iter().map(|v| v.to_f32()).collect();

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut cpu = vec![0f32; N_TOKENS * N_HEADS * HEAD_DIM];
    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..N_HEADS {
            let kv_head = h / (N_HEADS / N_KV_HEADS);
            let qv = &q[(t * N_HEADS + h) * HEAD_DIM..][..HEAD_DIM];

            let mut scores = Vec::with_capacity(pos as usize + 1);
            for j in 0..=pos as usize {
                let kj = &k[(kv_head * MAX_SEQ + j) * HEAD_DIM..][..HEAD_DIM];
                let dot: f32 = qv.iter().zip(kj).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();

            let out = &mut cpu[(t * N_HEADS + h) * HEAD_DIM..][..HEAD_DIM];
            for (j, e) in exps.iter().enumerate() {
                let vj = &v[(kv_head * MAX_SEQ + j) * HEAD_DIM..][..HEAD_DIM];
                let w = e / sum;
                for d in 0..HEAD_DIM {
                    out[d] += w * vj[d];
                }
            }
        }
    }

    let d_q = DeviceBuffer::<f32>::from_slice(device, &q, stream)?;
    let d_k = DeviceBuffer::<half::f16>::from_slice(device, &k_half, stream)?;
    let d_v = DeviceBuffer::<half::f16>::from_slice(device, &v_half, stream)?;
    let d_p = DeviceBuffer::<i32>::from_slice(device, &positions, stream)?;
    let d_o = DeviceBuffer::<f32>::new(device, N_TOKENS * N_HEADS * HEAD_DIM)?;

    ops::attention(
        d_o.as_mut_ptr(),
        d_q.as_ptr(),
        d_k.as_ptr(),
        d_v.as_ptr(),
        d_p.as_ptr(),
        N_TOKENS,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        MAX_SEQ,
        scale,
        0,
        ops::KvType::F16,
        stream,
    )?;
    let gpu = d_o.to_vec(stream)?;

    let (abs, rel) = compare(&cpu, &gpu);
    let peak = scale_of(&cpu).max(scale_of(&gpu));
    report.case(
        &format!("attention {label}"),
        format!("max_abs {abs:.3e}  max_rel {rel:.3e}  peak {peak:.3e}  ({n_kv} keys)"),
        abs <= tolerance && peak > 1e-4,
    );

    // A window of 1 must reduce to attending only to the query's own position.
    let d_o2 = DeviceBuffer::<f32>::new(device, N_TOKENS * N_HEADS * HEAD_DIM)?;
    ops::attention(
        d_o2.as_mut_ptr(),
        d_q.as_ptr(),
        d_k.as_ptr(),
        d_v.as_ptr(),
        d_p.as_ptr(),
        N_TOKENS,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        MAX_SEQ,
        scale,
        1,
        ops::KvType::F16,
        stream,
    )?;
    let windowed = d_o2.to_vec(stream)?;
    let mut expect = vec![0f32; N_TOKENS * N_HEADS * HEAD_DIM];
    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..N_HEADS {
            let kv_head = h / (N_HEADS / N_KV_HEADS);
            let vj = &v[(kv_head * MAX_SEQ + pos as usize) * HEAD_DIM..][..HEAD_DIM];
            expect[(t * N_HEADS + h) * HEAD_DIM..][..HEAD_DIM].copy_from_slice(vj);
        }
    }
    let (wabs, wrel) = compare(&expect, &windowed);
    report.case(
        &format!("attention {label}, window = 1"),
        format!("max_abs {wabs:.3e}  max_rel {wrel:.3e}"),
        wabs <= 1e-4,
    );

    let partial = DeviceBuffer::<f32>::new(device, N_HEADS * 128 * (HEAD_DIM + 2))?;
    for splits in [4, 16, 128] {
        for t in 0..N_TOKENS.min(3) {
            for window in [0, 1] {
                ops::attention_split(d_o.as_mut_ptr(), &partial,
                    d_q.offset_ptr(t * N_HEADS * HEAD_DIM), d_k.as_ptr(), d_v.as_ptr(),
                    d_p.offset_ptr(t), N_HEADS, N_KV_HEADS, HEAD_DIM, MAX_SEQ,
                    scale, window, splits, ops::KvType::F16, stream)?;
                let mut actual = vec![0.0f32; N_HEADS * HEAD_DIM];
                d_o.copy_to_host(&mut actual, stream)?;
                stream.synchronize()?;
                let reference = if window == 0 { &cpu } else { &expect };
                let reference = &reference[t * N_HEADS * HEAD_DIM..(t + 1) * N_HEADS * HEAD_DIM];
                let (error, _) = compare(reference, &actual);
                report.case(&format!("split attention {label}, splits={splits}, token={t}, window={window}"),
                    format!("max_abs {error:.3e}"), error <= 1e-3 && actual.iter().all(|x| x.is_finite()));
            }
        }
    }

    Ok(())
}
