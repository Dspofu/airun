//! Quantizers: f32 rows into packed GGUF blocks.
//!
//! Every encoder here is the inverse of the decoder in [`crate::dequant`], and
//! the round-trip is checked by unit tests. Output is byte-format-compatible
//! with ggml, so a file airun produces loads in any GGUF reader.
//!
//! The simple types (Q4_0 through Q8_0) follow ggml's reference algorithm
//! exactly. The K-quants use a direct min/max fit per sub-block rather than
//! ggml's iterative error search: the format is identical and the file is
//! interchangeable, but ggml's own quantizer will find marginally lower error.
//! See `docs/QUANTIZATION.md`.

use half::f16;

use crate::error::{Error, Result};
use crate::ggml_type::GgmlType;

/// Quantizes `src` (a whole tensor, row-major) into `dst`.
///
/// `row_len` must be a multiple of the type's block size; quantization is
/// per-block and never crosses a row boundary.
pub fn quantize(ty: GgmlType, src: &[f32], row_len: usize, dst: &mut Vec<u8>) -> Result<()> {
    let layout = ty.layout().ok_or(Error::UnknownTensorType {
        tensor: "<buffer>".into(),
        type_id: ty.id(),
    })?;
    let block_elems = layout.elements as usize;

    if row_len % block_elems != 0 {
        return Err(Error::Corrupt(format!(
            "row length {row_len} is not a multiple of the {ty} block size {block_elems}"
        )));
    }
    if src.len() % row_len != 0 {
        return Err(Error::Corrupt(format!(
            "tensor of {} elements does not divide into rows of {row_len}",
            src.len()
        )));
    }

    dst.reserve(src.len() / block_elems * layout.bytes as usize);
    for block in src.chunks(block_elems) {
        encode_block(ty, block, dst)?;
    }
    Ok(())
}

/// Encodes one block, appending exactly `layout.bytes` bytes.
pub fn encode_block(ty: GgmlType, x: &[f32], dst: &mut Vec<u8>) -> Result<()> {
    let before = dst.len();
    match ty {
        GgmlType::F32 => {
            for &v in x {
                dst.extend_from_slice(&v.to_le_bytes());
            }
        }
        GgmlType::F16 => {
            for &v in x {
                dst.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        GgmlType::BF16 => {
            for &v in x {
                dst.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
            }
        }
        GgmlType::Q8_0 => encode_q8_0(x, dst),
        GgmlType::Q4_0 => encode_q4_0(x, dst),
        GgmlType::Q4_1 => encode_q4_1(x, dst),
        GgmlType::Q5_0 => encode_q5_0(x, dst),
        GgmlType::Q5_1 => encode_q5_1(x, dst),
        GgmlType::Q4_K => encode_q4_k(x, dst),
        GgmlType::Q5_K => encode_q5_k(x, dst),
        GgmlType::Q6_K => encode_q6_k(x, dst),
        GgmlType::NVFP4 => encode_nvfp4(x, dst),
        other => {
            return Err(Error::Corrupt(format!(
                "no quantizer for {other}; supported targets are F32, F16, BF16, \
                 Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K and NVFP4"
            )));
        }
    }
    debug_assert_eq!(
        dst.len() - before,
        ty.layout().unwrap().bytes as usize,
        "{ty} encoder wrote the wrong number of bytes"
    );
    Ok(())
}

/// Round-to-nearest-even down to bf16, matching what PyTorch writes.
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        // Preserve NaN-ness; the naive rounding below could clear the payload.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

// --- simple block types -----------------------------------------------------

fn encode_q8_0(x: &[f32], dst: &mut Vec<u8>) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    for &v in x {
        dst.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
    }
}

/// Q4_0 centres on 8 and takes its scale from the largest-magnitude value,
/// keeping the sign of that value so the extreme maps to code 0.
fn encode_q4_0(x: &[f32], dst: &mut Vec<u8>) {
    let mut amax = 0f32;
    let mut max = 0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    for j in 0..16 {
        let lo = ((x[j] * id + 8.5) as i32).clamp(0, 15) as u8;
        let hi = ((x[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
        dst.push(lo | (hi << 4));
    }
}

fn encode_q4_1(x: &[f32], dst: &mut Vec<u8>) {
    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let d = (max - min) / 15.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    dst.extend_from_slice(&f16::from_f32(min).to_le_bytes());
    for j in 0..16 {
        let lo = (((x[j] - min) * id + 0.5) as i32).clamp(0, 15) as u8;
        let hi = (((x[j + 16] - min) * id + 0.5) as i32).clamp(0, 15) as u8;
        dst.push(lo | (hi << 4));
    }
}

fn encode_q5_0(x: &[f32], dst: &mut Vec<u8>) {
    let mut amax = 0f32;
    let mut max = 0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let d = max / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
        let x0 = ((x[j] * id + 16.5) as i32).clamp(0, 31) as u32;
        let x1 = ((x[j + 16] * id + 16.5) as i32).clamp(0, 31) as u32;
        qs[j] = ((x0 & 0x0F) | ((x1 & 0x0F) << 4)) as u8;
        // The fifth bit of each value lives in the shared 32-bit field.
        qh |= ((x0 & 0x10) >> 4) << j;
        qh |= ((x1 & 0x10) >> 4) << (j + 16);
    }

    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    dst.extend_from_slice(&qh.to_le_bytes());
    dst.extend_from_slice(&qs);
}

fn encode_q5_1(x: &[f32], dst: &mut Vec<u8>) {
    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let d = (max - min) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
        let x0 = (((x[j] - min) * id + 0.5) as i32).clamp(0, 31) as u32;
        let x1 = (((x[j + 16] - min) * id + 0.5) as i32).clamp(0, 31) as u32;
        qs[j] = ((x0 & 0x0F) | ((x1 & 0x0F) << 4)) as u8;
        qh |= ((x0 & 0x10) >> 4) << j;
        qh |= ((x1 & 0x10) >> 4) << (j + 16);
    }

    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    dst.extend_from_slice(&f16::from_f32(min).to_le_bytes());
    dst.extend_from_slice(&qh.to_le_bytes());
    dst.extend_from_slice(&qs);
}

// --- K-quants ---------------------------------------------------------------

/// Packs eight 6-bit scale/min pairs into the 12-byte K-quant header.
/// Inverse of `get_scale_min_k4`.
fn pack_k_scales(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
    let mut q = [0u8; 12];
    for j in 0..4 {
        q[j] = scales[j] & 63;
        q[j + 4] = mins[j] & 63;
    }
    for j in 4..8 {
        // Low nibbles go in bytes 8..12; the high 2 bits go into the top of the
        // first eight bytes.
        q[j + 4] = (scales[j] & 0x0F) | ((mins[j] & 0x0F) << 4);
        q[j - 4] |= (scales[j] >> 4) << 6;
        q[j] |= (mins[j] >> 4) << 6;
    }
    q
}

/// Q4_K: 8 sub-blocks of 32, each with a 6-bit scale and 6-bit min.
fn encode_q4_k(x: &[f32], dst: &mut Vec<u8>) {
    let mut sub_scale = [0f32; 8];
    let mut sub_min = [0f32; 8];
    for s in 0..8 {
        let block = &x[s * 32..(s + 1) * 32];
        let min = block.iter().copied().fold(f32::INFINITY, f32::min);
        let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        sub_scale[s] = (max - min) / 15.0;
        // Stored as a positive magnitude; the decoder subtracts it.
        sub_min[s] = -min;
    }

    let max_scale = sub_scale.iter().copied().fold(0f32, f32::max);
    let max_min = sub_min.iter().copied().fold(0f32, f32::max);
    let d = max_scale / 63.0;
    let dmin = max_min / 63.0;
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let inv_dmin = if dmin != 0.0 { 1.0 / dmin } else { 0.0 };

    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for s in 0..8 {
        sc[s] = (sub_scale[s] * inv_d).round().clamp(0.0, 63.0) as u8;
        mn[s] = (sub_min[s] * inv_dmin).round().clamp(0.0, 63.0) as u8;
    }

    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    dst.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
    dst.extend_from_slice(&pack_k_scales(&sc, &mn));

    // Quantize against the scales as they will be reconstructed, not the ideal
    // ones, so the 6-bit rounding of the scale does not compound.
    let mut qs = [0u8; 128];
    for s in 0..8 {
        let scale = d * sc[s] as f32;
        let min = dmin * mn[s] as f32;
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        for i in 0..32 {
            let q = (((x[s * 32 + i] + min) * inv).round()).clamp(0.0, 15.0) as u8;
            // Even sub-blocks take low nibbles, odd ones the highs of the same bytes.
            let byte = (s / 2) * 32 + i;
            if s % 2 == 0 {
                qs[byte] |= q;
            } else {
                qs[byte] |= q << 4;
            }
        }
    }
    dst.extend_from_slice(&qs);
}

/// Q5_K: like Q4_K with a fifth bit per value in a shared field.
fn encode_q5_k(x: &[f32], dst: &mut Vec<u8>) {
    let mut sub_scale = [0f32; 8];
    let mut sub_min = [0f32; 8];
    for s in 0..8 {
        let block = &x[s * 32..(s + 1) * 32];
        let min = block.iter().copied().fold(f32::INFINITY, f32::min);
        let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        sub_scale[s] = (max - min) / 31.0;
        sub_min[s] = -min;
    }

    let d = sub_scale.iter().copied().fold(0f32, f32::max) / 63.0;
    let dmin = sub_min.iter().copied().fold(0f32, f32::max) / 63.0;
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let inv_dmin = if dmin != 0.0 { 1.0 / dmin } else { 0.0 };

    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for s in 0..8 {
        sc[s] = (sub_scale[s] * inv_d).round().clamp(0.0, 63.0) as u8;
        mn[s] = (sub_min[s] * inv_dmin).round().clamp(0.0, 63.0) as u8;
    }

    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    dst.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
    dst.extend_from_slice(&pack_k_scales(&sc, &mn));

    let mut qh = [0u8; 32];
    let mut qs = [0u8; 128];
    for s in 0..8 {
        let scale = d * sc[s] as f32;
        let min = dmin * mn[s] as f32;
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        for i in 0..32 {
            let q = (((x[s * 32 + i] + min) * inv).round()).clamp(0.0, 31.0) as u8;
            let byte = (s / 2) * 32 + i;
            if s % 2 == 0 {
                qs[byte] |= q & 0x0F;
            } else {
                qs[byte] |= (q & 0x0F) << 4;
            }
            if q & 0x10 != 0 {
                qh[i] |= 1 << s;
            }
        }
    }
    dst.extend_from_slice(&qh);
    dst.extend_from_slice(&qs);
}

/// Q6_K: 16 sub-blocks of 16, each with a signed 8-bit scale.
fn encode_q6_k(x: &[f32], dst: &mut Vec<u8>) {
    // Symmetric quantization: codes span -32..31 around zero.
    let mut sub_scale = [0f32; 16];
    for s in 0..16 {
        let amax = x[s * 16..(s + 1) * 16]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()));
        sub_scale[s] = amax / 32.0;
    }
    let max_scale = sub_scale.iter().copied().fold(0f32, f32::max);
    let d = max_scale / 127.0;
    let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut scales = [0i8; 16];
    for s in 0..16 {
        scales[s] = (sub_scale[s] * inv_d).round().clamp(0.0, 127.0) as i8;
    }

    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for s in 0..16 {
        let scale = d * scales[s] as f32;
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        for i in 0..16 {
            let idx = s * 16 + i;
            let q = ((x[idx] * inv).round().clamp(-32.0, 31.0) as i32 + 32) as u8;

            // Reproduce the decoder's interleave: the superblock splits into two
            // 128-value halves, each of four quarters.
            let sub = idx / 32; // 0..7, matching dq_q6_K's `sub`
            let within = idx % 32;
            let half = sub / 4;
            let quarter = sub % 4;

            let ql_index = half * 64 + (quarter % 2) * 32 + within;
            if quarter >= 2 {
                ql[ql_index] |= (q & 0x0F) << 4;
            } else {
                ql[ql_index] |= q & 0x0F;
            }
            qh[half * 32 + within] |= ((q >> 4) & 3) << (quarter * 2);
        }
    }

    dst.extend_from_slice(&ql);
    dst.extend_from_slice(&qh);
    for s in scales {
        dst.push(s as u8);
    }
    dst.extend_from_slice(&f16::from_f32(d).to_le_bytes());
}

/// Encodes a positive float as UE4M3, the inverse of
/// [`crate::dequant::ue4m3_to_f32`] up to that decoder's halving.
///
/// Follows ggml's rounding exactly: round-to-nearest on the mantissa, saturate
/// at 0x7E rather than emitting the 0x7F the decoder reads as zero.
fn f32_to_ue4m3(x: f32) -> u8 {
    if !(x > 0.0) {
        return 0;
    }
    let x = x.min(448.0);
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut exp = fp32_exp + 7;

    if exp <= 0 {
        // Below the smallest normal: fall back to the subnormal encoding, where
        // the mantissa alone carries the value.
        let man = (x * 512.0 + 0.5) as i32;
        return man.clamp(0, 7) as u8;
    }
    if exp >= 15 {
        return 0x7E;
    }

    let mut man = fp32_man + ((bits >> 19) & 1) as i32;
    if man > 7 {
        man = 0;
        exp += 1;
        if exp >= 15 {
            return 0x7E;
        }
    }
    ((exp << 3) | man) as u8
}

/// The FP4 code whose decoded value sits closest to `x` at scale `d`.
///
/// A linear scan over sixteen codes: the codebook is not monotone in the index
/// (the sign bit is the high bit), so a bisect would need the sorted order and
/// buys nothing at this size.
fn best_index_fp4(x: f32, d: f32) -> u8 {
    if d == 0.0 {
        return 0;
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &kv) in crate::dequant::KVALUES_MXFP4.iter().enumerate() {
        let err = (x - d * f32::from(kv)).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    best
}

/// NVFP4: 64 elements as four runs of 16, each with its own UE4M3 scale.
///
/// The scale targets `amax / 6` because the decoder halves it and the codebook
/// is doubled, which together put the largest code exactly on `amax`.
///
/// The block carries no per-tensor scale, so UE4M3's smallest step is a floor
/// on absolute magnitude: a run of 16 whose peak is under about 5.9e-3 encodes
/// as all zeros. Real weight rows sit well above that, but a tensor of unusually
/// small values would be silently erased -- see the unit test that pins it.
fn encode_nvfp4(x: &[f32], dst: &mut Vec<u8>) {
    let mut scales = [0u8; 4];
    let mut qs = [0u8; 32];

    for sub in 0..4 {
        let xb = &x[sub * 16..sub * 16 + 16];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));

        let ue = f32_to_ue4m3(amax / 6.0);
        scales[sub] = ue;
        let d = crate::dequant::ue4m3_to_f32(ue);

        for j in 0..8 {
            let lo = best_index_fp4(xb[j], d);
            let hi = best_index_fp4(xb[j + 8], d);
            qs[sub * 8 + j] = lo | (hi << 4);
        }
    }

    dst.extend_from_slice(&scales);
    dst.extend_from_slice(&qs);
}

/// Types [`quantize`] can produce.
pub fn can_encode(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::NVFP4
    )
}

/// Parses a quantization name like `Q4_K` or `q8_0`.
pub fn parse_type(name: &str) -> Option<GgmlType> {
    let upper = name.to_ascii_uppercase();
    let ty = match upper.as_str() {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "BF16" => GgmlType::BF16,
        "Q4_0" => GgmlType::Q4_0,
        "Q4_1" => GgmlType::Q4_1,
        "Q5_0" => GgmlType::Q5_0,
        "Q5_1" => GgmlType::Q5_1,
        "Q8_0" => GgmlType::Q8_0,
        // The _M / _S suffixes name a per-tensor mixing policy, not a block
        // format; both map to the same block type here.
        "Q4_K" | "Q4_K_M" | "Q4_K_S" => GgmlType::Q4_K,
        "Q5_K" | "Q5_K_M" | "Q5_K_S" => GgmlType::Q5_K,
        "Q6_K" => GgmlType::Q6_K,
        "NVFP4" => GgmlType::NVFP4,
        _ => return None,
    };
    Some(ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequant::decode_block;

    /// Smooth, deterministic test data resembling real weight distributions.
    fn sample(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 * 0.1 + seed;
                (t.sin() * 0.7 + (t * 0.37).cos() * 0.3) * 0.05
            })
            .collect()
    }

    /// Round-trip every encoder through its decoder and bound the error by what
    /// the format's bit width can represent.
    #[test]
    fn round_trip_error_is_within_the_format_budget() {
        // type, max RMS error as a fraction of the block's peak magnitude
        let cases = [
            (GgmlType::F32, 1e-7),
            (GgmlType::F16, 1e-3),
            (GgmlType::BF16, 1e-2),
            (GgmlType::Q8_0, 5e-3),
            (GgmlType::Q6_K, 1e-2),
            (GgmlType::Q5_K, 2e-2),
            (GgmlType::Q5_0, 3e-2),
            (GgmlType::Q5_1, 3e-2),
            (GgmlType::Q4_K, 4e-2),
            // FP4 spends its eight magnitude levels on dynamic range, so the
            // gaps at the top of the scale are wide; a sinusoid, whose values
            // pile up at the extremes, is the worst case for it.
            (GgmlType::NVFP4, 7e-2),
            (GgmlType::Q4_0, 6e-2),
            (GgmlType::Q4_1, 6e-2),
        ];

        for (ty, tolerance) in cases {
            let n = ty.layout().unwrap().elements as usize;
            let x = sample(n, ty.id() as f32);

            let mut packed = Vec::new();
            encode_block(ty, &x, &mut packed).unwrap();
            assert_eq!(
                packed.len(),
                ty.layout().unwrap().bytes as usize,
                "{ty} wrote the wrong block size"
            );

            let mut back = vec![0f32; n];
            decode_block(ty, &packed, &mut back).unwrap();

            let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let rms = (x
                .iter()
                .zip(&back)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / n as f64)
                .sqrt() as f32;

            assert!(
                rms <= tolerance * peak,
                "{ty}: RMS error {rms:.3e} exceeds {:.3e} (peak {peak:.3e})",
                tolerance * peak
            );
        }
    }

    /// NVFP4's reason to exist over MXFP4 is the per-16 scale: one loud run in
    /// a block must not cost the quiet runs their resolution.
    #[test]
    fn nvfp4_scales_each_sixteen_independently() {
        let mut x = vec![0f32; 64];
        for (i, v) in x.iter_mut().enumerate() {
            // Four runs spanning five orders of magnitude between them.
            let scale = [1.0f32, 100.0, 0.05, 1.0][i / 16];
            *v = scale * ((i % 16) as f32 / 15.0 - 0.5);
        }

        let mut packed = Vec::new();
        encode_block(GgmlType::NVFP4, &x, &mut packed).unwrap();
        let mut back = vec![0f32; 64];
        decode_block(GgmlType::NVFP4, &packed, &mut back).unwrap();

        for sub in 0..4 {
            let run = &x[sub * 16..sub * 16 + 16];
            let peak = run.iter().fold(0f32, |m, v| m.max(v.abs()));
            for (i, (&a, &b)) in run.iter().zip(&back[sub * 16..sub * 16 + 16]).enumerate() {
                // FP4's widest gap is between codes 8 and 12, so a fifth of the
                // run's peak is the worst any correct encoder can do. A shared
                // scale would miss by the whole peak, which this still catches.
                assert!(
                    (a - b).abs() <= 0.2 * peak,
                    "run {sub} element {i}: {a:e} decoded as {b:e}, peak {peak:e}"
                );
            }
        }
    }

    /// The block carries no per-tensor scale, so UE4M3's smallest step is a hard
    /// floor on absolute magnitude: a run quieter than that encodes as silence.
    /// Worth pinning down, because it is invisible in relative error -- a tensor
    /// of tiny weights round-trips to exactly zero with no warning anywhere.
    #[test]
    fn nvfp4_has_an_absolute_magnitude_floor() {
        let floor = 6.0 / 1024.0; // the smallest peak a sub-block can encode

        let audible = vec![floor * 1.5; 64];
        let mut packed = Vec::new();
        encode_block(GgmlType::NVFP4, &audible, &mut packed).unwrap();
        let mut back = vec![0f32; 64];
        decode_block(GgmlType::NVFP4, &packed, &mut back).unwrap();
        assert!(back.iter().any(|&v| v != 0.0), "above the floor, values survive");

        let inaudible = vec![floor * 0.4; 64];
        packed.clear();
        encode_block(GgmlType::NVFP4, &inaudible, &mut packed).unwrap();
        decode_block(GgmlType::NVFP4, &packed, &mut back).unwrap();
        assert!(back.iter().all(|&v| v == 0.0), "below the floor, the run is lost");
    }

    #[test]
    fn k_scale_packing_round_trips() {
        // Mirrors get_scale_min_k4 from the decoder.
        fn unpack(j: usize, q: &[u8; 12]) -> (u8, u8) {
            if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                (
                    (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
                    (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                )
            }
        }

        let scales = [63u8, 1, 40, 17, 62, 3, 55, 8];
        let mins = [0u8, 63, 12, 45, 7, 61, 30, 22];
        let packed = pack_k_scales(&scales, &mins);
        for j in 0..8 {
            let (s, m) = unpack(j, &packed);
            assert_eq!(s, scales[j], "scale {j} did not round-trip");
            assert_eq!(m, mins[j], "min {j} did not round-trip");
        }
    }

    #[test]
    fn a_constant_block_survives_every_type() {
        // Zero range is the degenerate case every scale computation must handle
        // without dividing by zero.
        for ty in [
            GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1,
            GgmlType::Q8_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K,
        ] {
            let n = ty.layout().unwrap().elements as usize;
            for value in [0.0f32, 0.25] {
                let x = vec![value; n];
                let mut packed = Vec::new();
                encode_block(ty, &x, &mut packed).unwrap();
                let mut back = vec![0f32; n];
                decode_block(ty, &packed, &mut back).unwrap();
                assert!(
                    back.iter().all(|v| v.is_finite()),
                    "{ty} produced a non-finite value for a constant block"
                );
            }
        }
    }

    #[test]
    fn quantize_rejects_a_misaligned_row() {
        let src = vec![0f32; 300];
        let mut dst = Vec::new();
        // 100 is not a multiple of the 256-element Q4_K block.
        assert!(quantize(GgmlType::Q4_K, &src, 100, &mut dst).is_err());
    }

    #[test]
    fn type_names_parse_case_insensitively() {
        assert_eq!(parse_type("q4_k_m"), Some(GgmlType::Q4_K));
        assert_eq!(parse_type("Q8_0"), Some(GgmlType::Q8_0));
        assert_eq!(parse_type("nonsense"), None);
    }
}
