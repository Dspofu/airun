//! ggml tensor type table.
//!
//! Variants deliberately keep ggml's own spelling (`Q4_K`, `IQ4_XS`) so that
//! grepping this file against upstream or against a `--tensors` dump lines up.
#![allow(non_camel_case_types)]
//!
//! Type IDs are the on-disk contract of GGUF. Upstream ggml has retired IDs over
//! time (the `Q4_0_4_4` family at 31..33, `IQ4_NL_4_4` at 36..38) and keeps adding
//! new ones at the tail. We therefore never treat an unrecognised ID as a parse
//! error: it round-trips through `Unknown(u32)` so `inspect` can still describe a
//! file produced by a newer toolchain than this build.

/// Layout of one quantization block: how many logical elements it encodes and
/// how many bytes it occupies on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    /// Elements per block. 1 for the plain scalar types.
    pub elements: u64,
    /// Bytes per block.
    pub bytes: u64,
}

impl BlockLayout {
    const fn new(elements: u64, bytes: u64) -> Self {
        Self { elements, bytes }
    }
}

macro_rules! ggml_types {
    ($( $variant:ident = $id:expr, $name:literal, $elems:expr, $bytes:expr, $support:ident ; )*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum GgmlType {
            $( $variant, )*
            /// A type ID this build does not know about.
            Unknown(u32),
        }

        impl GgmlType {
            pub fn from_id(id: u32) -> Self {
                match id {
                    $( $id => Self::$variant, )*
                    other => Self::Unknown(other),
                }
            }

            pub fn id(self) -> u32 {
                match self {
                    $( Self::$variant => $id, )*
                    Self::Unknown(id) => id,
                }
            }

            pub fn name(self) -> &'static str {
                match self {
                    $( Self::$variant => $name, )*
                    Self::Unknown(_) => "UNKNOWN",
                }
            }

            /// `None` for unknown types, whose block geometry we cannot guess.
            pub fn layout(self) -> Option<BlockLayout> {
                match self {
                    $( Self::$variant => Some(BlockLayout::new($elems, $bytes)), )*
                    Self::Unknown(_) => None,
                }
            }

            /// How far along this type is in the airun compute path.
            pub fn support(self) -> Support {
                match self {
                    $( Self::$variant => Support::$support, )*
                    Self::Unknown(_) => Support::Unknown,
                }
            }
        }
    };
}

/// How well airun handles a given tensor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Parsed, dequantized and usable as a compute input.
    Full,
    /// Parsed and dequantized on CPU, but no fused GPU kernel yet.
    Dequant,
    /// Block geometry known, so the file maps cleanly, but no dequantizer.
    Layout,
    /// Type ID not known to this build.
    Unknown,
}

impl Support {
    pub fn label(self) -> &'static str {
        match self {
            Support::Full => "full",
            Support::Dequant => "dequant-only",
            Support::Layout => "layout-only",
            Support::Unknown => "unknown",
        }
    }
}

// id, display name, elements/block, bytes/block, support level
ggml_types! {
    F32     =  0, "F32",     1,   4,   Full;
    F16     =  1, "F16",     1,   2,   Full;
    Q4_0    =  2, "Q4_0",    32,  18,  Full;
    Q4_1    =  3, "Q4_1",    32,  20,  Full;
    Q5_0    =  6, "Q5_0",    32,  22,  Full;
    Q5_1    =  7, "Q5_1",    32,  24,  Full;
    Q8_0    =  8, "Q8_0",    32,  34,  Full;
    Q8_1    =  9, "Q8_1",    32,  40,  Dequant;
    Q2_K    = 10, "Q2_K",    256, 84,  Full;
    Q3_K    = 11, "Q3_K",    256, 110, Full;
    Q4_K    = 12, "Q4_K",    256, 144, Full;
    Q5_K    = 13, "Q5_K",    256, 176, Full;
    Q6_K    = 14, "Q6_K",    256, 210, Full;
    Q8_K    = 15, "Q8_K",    256, 292, Dequant;
    IQ2_XXS = 16, "IQ2_XXS", 256, 66,  Layout;
    IQ2_XS  = 17, "IQ2_XS",  256, 74,  Layout;
    IQ3_XXS = 18, "IQ3_XXS", 256, 98,  Layout;
    IQ1_S   = 19, "IQ1_S",   256, 50,  Layout;
    IQ4_NL  = 20, "IQ4_NL",  32,  18,  Full;
    IQ3_S   = 21, "IQ3_S",   256, 110, Layout;
    IQ2_S   = 22, "IQ2_S",   256, 82,  Layout;
    IQ4_XS  = 23, "IQ4_XS",  256, 136, Full;
    I8      = 24, "I8",      1,   1,   Layout;
    I16     = 25, "I16",     1,   2,   Layout;
    I32     = 26, "I32",     1,   4,   Layout;
    I64     = 27, "I64",     1,   8,   Layout;
    F64     = 28, "F64",     1,   8,   Dequant;
    IQ1_M   = 29, "IQ1_M",   256, 56,  Layout;
    BF16    = 30, "BF16",    1,   2,   Full;
    TQ1_0   = 34, "TQ1_0",   256, 54,  Layout;
    TQ2_0   = 35, "TQ2_0",   256, 66,  Layout;
    MXFP4   = 39, "MXFP4",   32,  17,  Full;
    NVFP4   = 40, "NVFP4",   64,  36,  Full;
    Q1_0    = 41, "Q1_0",    128, 18,  Layout;
    Q2_0    = 42, "Q2_0",    64,  18,  Layout;
}

impl GgmlType {
    /// Bytes needed to store `n` elements of this type.
    ///
    /// Returns `None` for unknown types, and for element counts that do not fill
    /// a whole number of blocks (which would mean a corrupt tensor shape).
    pub fn nbytes(self, n_elements: u64) -> Option<u64> {
        let layout = self.layout()?;
        if n_elements % layout.elements != 0 {
            return None;
        }
        Some(n_elements / layout.elements * layout.bytes)
    }

    /// Bits per weight, the usual way quantization levels get compared.
    pub fn bits_per_weight(self) -> Option<f64> {
        let layout = self.layout()?;
        Some(layout.bytes as f64 * 8.0 / layout.elements as f64)
    }

    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// K-quants carry per-superblock scales and share a 256-element geometry.
    pub fn is_k_quant(self) -> bool {
        matches!(
            self,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K
        )
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "UNKNOWN({id})"),
            other => f.write_str(other.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_ids_round_trip() {
        for id in 0u32..64 {
            let t = GgmlType::from_id(id);
            assert_eq!(t.id(), id, "type id {id} did not round-trip");
        }
    }

    #[test]
    fn retired_ids_stay_unknown() {
        // 4/5 (Q4_2, Q4_3) and 31..33 (Q4_0_4_4 family) were removed upstream.
        for id in [4, 5, 31, 32, 33, 36, 37, 38] {
            assert!(matches!(GgmlType::from_id(id), GgmlType::Unknown(_)));
        }
    }

    #[test]
    fn byte_sizes_match_block_geometry() {
        assert_eq!(GgmlType::Q8_0.nbytes(32), Some(34));
        assert_eq!(GgmlType::Q4_K.nbytes(256), Some(144));
        assert_eq!(GgmlType::Q6_K.nbytes(512), Some(420));
        assert_eq!(GgmlType::F32.nbytes(10), Some(40));
        assert_eq!(GgmlType::MXFP4.nbytes(64), Some(34));
        assert_eq!(GgmlType::NVFP4.nbytes(128), Some(72));
        // NVFP4 packs 64 elements, so a 32-element tail is not a whole block.
        assert_eq!(GgmlType::NVFP4.nbytes(32), None);
        // A partial block means the shape is wrong for the type.
        assert_eq!(GgmlType::Q4_K.nbytes(100), None);
        assert_eq!(GgmlType::Unknown(99).nbytes(256), None);
    }

    #[test]
    fn bits_per_weight_is_sane() {
        assert_eq!(GgmlType::F16.bits_per_weight(), Some(16.0));
        assert_eq!(GgmlType::Q8_0.bits_per_weight(), Some(8.5));
        assert_eq!(GgmlType::Q4_K.bits_per_weight(), Some(4.5));
        assert_eq!(GgmlType::MXFP4.bits_per_weight(), Some(4.25));
        // Four extra scale bytes per 64 weights is what NVFP4 pays over MXFP4.
        assert_eq!(GgmlType::NVFP4.bits_per_weight(), Some(4.5));
    }
}
