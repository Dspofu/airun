//! CPU reference decoders for every quantization type airun computes on.
//!
//! These mirror the CUDA decoders in `airun-cuda/kernels/dequant.cuh` block for
//! block. They serve three purposes: they are the oracle the GPU kernels are
//! checked against by `airun selftest`, they let the converter round-trip a
//! quantization without a GPU, and they are the fallback when no device is
//! available.
//!
//! Correctness here is load-bearing for everything else, so the layouts follow
//! ggml's reference `dequantize_row_*` exactly rather than a clever rewrite.

use half::f16;

use crate::error::{Error, Result};
use crate::ggml_type::GgmlType;

/// Non-linear codebook shared by IQ4_NL and IQ4_XS.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// E2M1 codebook for MXFP4, doubled so it pairs with the half-valued E8M0
/// exponent (matching ggml, which avoids a division this way).
///
/// NVFP4 uses the same codebook against a UE4M3 scale that is likewise halved,
/// so the doubling cancels there too.
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

#[inline]
fn f16_at(bytes: &[u8], offset: usize) -> f32 {
    f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32()
}

#[inline]
fn bf16_from_le(lo: u8, hi: u8) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([lo, hi])) << 16)
}

/// E8M0: an exponent-only 8-bit float, evaluated at half its nominal value.
#[inline]
pub fn e8m0_to_f32_half(x: u8) -> f32 {
    // 0 and 1 land below the range the shift can express, so they are special-cased.
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        u32::from(x - 1) << 23
    };
    f32::from_bits(bits)
}

/// UE4M3: an unsigned 8-bit float (4-bit exponent, 3-bit mantissa, bias 7),
/// evaluated at half its nominal value so it pairs with the doubled FP4
/// codebook. This is NVFP4's per-16-element scale.
///
/// 0x7F is ggml's NaN encoding and decodes to zero, which is also what a zero
/// byte means -- a sub-block a quantizer found entirely zero.
#[inline]
pub fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = u32::from((x >> 3) & 0x0F);
    let man = u32::from(x & 0x07);
    if exp == 0 {
        // Subnormal: no implicit leading one, so the mantissa scales 2^-10
        // directly (2^-9 nominal, halved).
        return man as f32 / 1024.0;
    }
    // Normal: (1 + man/8) * 2^(exp-7) * 0.5. Assembling the f32 bits skips the
    // libm call this would otherwise take per sub-block.
    f32::from_bits(((exp + 119) << 23) | (man << 20))
}

/// Unpacks one of eight 6-bit scale/min pairs from a K-quant's 12-byte header.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Unpacks Q3_K's sixteen 6-bit scales from 12 bytes.
fn unpack_q3k_scales(src: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let mut aux = [0u32; 4];
    for (i, slot) in aux.iter_mut().enumerate().take(3) {
        *slot = u32::from_le_bytes([src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]]);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    let mut out = [0i8; 16];
    for i in 0..4 {
        let b = aux[i].to_le_bytes();
        for j in 0..4 {
            out[i * 4 + j] = b[j] as i8;
        }
    }
    out
}

/// Decodes a whole tensor into `out`, which must hold exactly `n_elements`.
pub fn dequantize(ty: GgmlType, src: &[u8], n_elements: usize, out: &mut [f32]) -> Result<()> {
    if out.len() != n_elements {
        return Err(Error::Corrupt(format!(
            "dequantize output has {} slots for {n_elements} elements",
            out.len()
        )));
    }
    let layout = ty.layout().ok_or(Error::UnknownTensorType {
        tensor: "<buffer>".into(),
        type_id: ty.id(),
    })?;
    let elems = layout.elements as usize;
    if n_elements % elems != 0 {
        return Err(Error::Corrupt(format!(
            "{n_elements} elements is not a multiple of the {ty} block size {elems}"
        )));
    }
    let n_blocks = n_elements / elems;
    let need = n_blocks * layout.bytes as usize;
    if src.len() < need {
        return Err(Error::Truncated {
            what: format!("{ty} tensor data"),
            need: need as u64,
            have: src.len() as u64,
        });
    }

    let bs = layout.bytes as usize;
    for b in 0..n_blocks {
        let block = &src[b * bs..(b + 1) * bs];
        let dst = &mut out[b * elems..(b + 1) * elems];
        decode_block(ty, block, dst)?;
    }
    Ok(())
}

/// Decodes one block. `dst` must be exactly one block's worth of elements.
pub fn decode_block(ty: GgmlType, b: &[u8], dst: &mut [f32]) -> Result<()> {
    match ty {
        GgmlType::F32 => {
            for (i, v) in dst.iter_mut().enumerate() {
                *v = f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
            }
        }
        GgmlType::F16 => {
            for (i, v) in dst.iter_mut().enumerate() {
                *v = f16_at(b, i * 2);
            }
        }
        GgmlType::BF16 => {
            for (i, v) in dst.iter_mut().enumerate() {
                *v = bf16_from_le(b[i * 2], b[i * 2 + 1]);
            }
        }
        GgmlType::F64 => {
            for (i, v) in dst.iter_mut().enumerate() {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&b[i * 8..i * 8 + 8]);
                *v = f64::from_le_bytes(buf) as f32;
            }
        }
        GgmlType::Q8_0 => {
            let d = f16_at(b, 0);
            for i in 0..32 {
                dst[i] = d * f32::from(b[2 + i] as i8);
            }
        }
        GgmlType::Q4_0 => {
            let d = f16_at(b, 0);
            let qs = &b[2..18];
            for i in 0..16 {
                dst[i] = d * (f32::from(qs[i] & 0x0F) - 8.0);
                dst[i + 16] = d * (f32::from(qs[i] >> 4) - 8.0);
            }
        }
        GgmlType::Q4_1 => {
            let d = f16_at(b, 0);
            let m = f16_at(b, 2);
            let qs = &b[4..20];
            for i in 0..16 {
                dst[i] = d * f32::from(qs[i] & 0x0F) + m;
                dst[i + 16] = d * f32::from(qs[i] >> 4) + m;
            }
        }
        GgmlType::Q5_0 => {
            let d = f16_at(b, 0);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            let qs = &b[6..22];
            for i in 0..16 {
                let hi0 = ((qh >> i) << 4) & 0x10;
                let hi1 = (qh >> (i + 12)) & 0x10;
                dst[i] = d * ((i32::from(qs[i] & 0x0F) | hi0 as i32) - 16) as f32;
                dst[i + 16] = d * ((i32::from(qs[i] >> 4) | hi1 as i32) - 16) as f32;
            }
        }
        GgmlType::Q5_1 => {
            let d = f16_at(b, 0);
            let m = f16_at(b, 2);
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let qs = &b[8..24];
            for i in 0..16 {
                let hi0 = ((qh >> i) << 4) & 0x10;
                let hi1 = (qh >> (i + 12)) & 0x10;
                dst[i] = d * (i32::from(qs[i] & 0x0F) | hi0 as i32) as f32 + m;
                dst[i + 16] = d * (i32::from(qs[i] >> 4) | hi1 as i32) as f32 + m;
            }
        }
        GgmlType::Q2_K => {
            let scales = &b[0..16];
            let qs = &b[16..80];
            let d = f16_at(b, 80);
            let dmin = f16_at(b, 82);
            // 16 units of 16 values each.
            for unit in 0..16 {
                let half = unit / 8;
                let within = unit % 8;
                let j = within / 2;
                let part = within % 2;
                let q = &qs[half * 32 + part * 16..];
                let shift = j * 2;
                let sc = scales[unit];
                let dl = d * f32::from(sc & 0x0F);
                let ml = dmin * f32::from(sc >> 4);
                for i in 0..16 {
                    dst[unit * 16 + i] = dl * f32::from((q[i] >> shift) & 3) - ml;
                }
            }
        }
        GgmlType::Q3_K => {
            let hmask = &b[0..32];
            let qs = &b[32..96];
            let scales = unpack_q3k_scales(&b[96..108]);
            let d = f16_at(b, 108);
            for unit in 0..16 {
                let half = unit / 8;
                let within = unit % 8;
                let j = within / 2;
                let part = within % 2;
                let q = &qs[half * 32 + part * 16..];
                let hm = &hmask[part * 16..];
                let shift = j * 2;
                // The high-bit mask walks the whole superblock, not each half.
                let m = 1u8 << (half * 4 + j);
                let dl = d * (i32::from(scales[unit]) - 32) as f32;
                for i in 0..16 {
                    let lo = i32::from((q[i] >> shift) & 3);
                    // A set bit means "no offset"; this inversion is deliberate.
                    let hi = if hm[i] & m != 0 { 0 } else { 4 };
                    dst[unit * 16 + i] = dl * (lo - hi) as f32;
                }
            }
        }
        GgmlType::Q4_K => {
            let d = f16_at(b, 0);
            let dmin = f16_at(b, 2);
            let scales = &b[4..16];
            let qs = &b[16..144];
            for sub in 0..8 {
                let (sc, m) = get_scale_min_k4(sub, scales);
                let dl = d * f32::from(sc);
                let ml = dmin * f32::from(m);
                let q = &qs[(sub / 2) * 32..];
                if sub % 2 == 0 {
                    for i in 0..32 {
                        dst[sub * 32 + i] = dl * f32::from(q[i] & 0x0F) - ml;
                    }
                } else {
                    for i in 0..32 {
                        dst[sub * 32 + i] = dl * f32::from(q[i] >> 4) - ml;
                    }
                }
            }
        }
        GgmlType::Q5_K => {
            let d = f16_at(b, 0);
            let dmin = f16_at(b, 2);
            let scales = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..176];
            for sub in 0..8 {
                let (sc, m) = get_scale_min_k4(sub, scales);
                let dl = d * f32::from(sc);
                let ml = dmin * f32::from(m);
                let q = &qs[(sub / 2) * 32..];
                let bit = 1u8 << sub;
                for i in 0..32 {
                    let base = if sub % 2 == 0 {
                        q[i] & 0x0F
                    } else {
                        q[i] >> 4
                    };
                    let hi = if qh[i] & bit != 0 { 16 } else { 0 };
                    dst[sub * 32 + i] = dl * (i32::from(base) + hi) as f32 - ml;
                }
            }
        }
        GgmlType::Q6_K => {
            let ql = &b[0..128];
            let qh = &b[128..192];
            let scales = &b[192..208];
            let d = f16_at(b, 208);
            for sub in 0..8 {
                let half = sub / 4;
                let quarter = sub % 4;
                let ql_base = half * 64 + (quarter % 2) * 32;
                let qh_base = half * 32;
                let sc_base = half * 8 + quarter * 2;
                let shift = quarter * 2;
                let high = quarter >= 2;
                for i in 0..32 {
                    let lo = if high {
                        ql[ql_base + i] >> 4
                    } else {
                        ql[ql_base + i] & 0x0F
                    };
                    let hi = ((qh[qh_base + i] >> shift) & 3) << 4;
                    let q = i32::from(lo | hi) - 32;
                    let sc = scales[sc_base + i / 16] as i8;
                    dst[sub * 32 + i] = d * f32::from(sc) * q as f32;
                }
            }
        }
        GgmlType::IQ4_NL => {
            let d = f16_at(b, 0);
            let qs = &b[2..18];
            for i in 0..16 {
                dst[i] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[i] & 0x0F)]);
                dst[i + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(qs[i] >> 4)]);
            }
        }
        GgmlType::IQ4_XS => {
            let d = f16_at(b, 0);
            let scales_h = u16::from_le_bytes([b[2], b[3]]);
            let scales_l = &b[4..8];
            let qs = &b[8..136];
            for sub in 0..8 {
                let ls = i32::from((scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0F)
                    | (i32::from((scales_h >> (2 * sub)) & 3) << 4);
                let dl = d * (ls - 32) as f32;
                let q = &qs[sub * 16..];
                for i in 0..16 {
                    dst[sub * 32 + i] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[i] & 0x0F)]);
                    dst[sub * 32 + i + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(q[i] >> 4)]);
                }
            }
        }
        GgmlType::MXFP4 => {
            let d = e8m0_to_f32_half(b[0]);
            let qs = &b[1..17];
            for i in 0..16 {
                dst[i] = d * f32::from(KVALUES_MXFP4[usize::from(qs[i] & 0x0F)]);
                dst[i + 16] = d * f32::from(KVALUES_MXFP4[usize::from(qs[i] >> 4)]);
            }
        }
        GgmlType::NVFP4 => {
            // 64 elements as four independently scaled runs of 16. Within a run
            // the low nibbles fill the first eight slots and the high nibbles
            // the next eight, the same split MXFP4 uses across its 32.
            let (scales, qs) = b.split_at(4);
            for sub in 0..4 {
                let d = ue4m3_to_f32(scales[sub]);
                let q = &qs[sub * 8..sub * 8 + 8];
                let out = &mut dst[sub * 16..sub * 16 + 16];
                for i in 0..8 {
                    out[i] = d * f32::from(KVALUES_MXFP4[usize::from(q[i] & 0x0F)]);
                    out[i + 8] = d * f32::from(KVALUES_MXFP4[usize::from(q[i] >> 4)]);
                }
            }
        }
        other => {
            return Err(Error::Corrupt(format!(
                "no CPU decoder for {other}; run `airun inspect` to see the model's type mix"
            )));
        }
    }
    Ok(())
}

/// Whether a CPU decoder exists for this type.
pub fn can_decode(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::F64
            | GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
            | GgmlType::MXFP4
            | GgmlType::NVFP4
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random block bytes, so failures reproduce.
    fn fill(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect()
    }

    /// Every decoder must consume exactly its declared block size and produce
    /// finite values for arbitrary input bytes.
    #[test]
    fn all_decoders_handle_arbitrary_bytes() {
        let types = [
            GgmlType::F32, GgmlType::F16, GgmlType::BF16,
            GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_0,
            GgmlType::Q2_K, GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K,
            GgmlType::IQ4_NL, GgmlType::IQ4_XS, GgmlType::MXFP4, GgmlType::NVFP4,
        ];
        for ty in types {
            let layout = ty.layout().unwrap();
            let bytes = fill(ty.id() as u64 + 7, layout.bytes as usize);
            let mut out = vec![0f32; layout.elements as usize];
            decode_block(ty, &bytes, &mut out)
                .unwrap_or_else(|e| panic!("{ty} failed to decode: {e}"));
            // F16/BF16 can legitimately produce NaN from random bits; the
            // quantized types cannot, because every value is scale times integer.
            if ty.is_quantized() {
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "{ty} produced a non-finite value from valid block bytes"
                );
            }
            assert_eq!(out.len(), layout.elements as usize);
        }
    }

    #[test]
    fn q8_0_round_trips_a_known_block() {
        // d = 2.0, quants 0..31 -> values 0, 2, 4, ...
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        for i in 0..32 {
            block[2 + i] = i as u8;
        }
        let mut out = vec![0f32; 32];
        decode_block(GgmlType::Q8_0, &block, &mut out).unwrap();
        for i in 0..32 {
            assert_eq!(out[i], 2.0 * i as f32);
        }
    }

    #[test]
    fn q4_0_centres_on_eight() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        // Low nibble 8 and high nibble 0 -> values 0 and -8.
        block[2] = 0x08;
        let mut out = vec![0f32; 32];
        decode_block(GgmlType::Q4_0, &block, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[16], -8.0);
    }

    #[test]
    fn e8m0_matches_powers_of_two() {
        // Exponent 128 is the identity point: 2^(128-127) / 2 == 1.0
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(129), 2.0);
        assert_eq!(e8m0_to_f32_half(127), 0.5);
    }

    #[test]
    fn dequantize_rejects_a_short_buffer() {
        let src = vec![0u8; 10];
        let mut out = vec![0f32; 32];
        let err = dequantize(GgmlType::Q8_0, &src, 32, &mut out).unwrap_err();
        assert!(matches!(err, Error::Truncated { .. }));
    }

    #[test]
    fn dequantize_rejects_a_partial_block() {
        let src = vec![0u8; 1024];
        let mut out = vec![0f32; 100];
        let err = dequantize(GgmlType::Q4_K, &src, 100, &mut out).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)));
    }
}
