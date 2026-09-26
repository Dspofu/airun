//! Reading safetensors checkpoints.
//!
//! The format is a u64 header length, a JSON header describing every tensor, and
//! then raw little-endian payloads. Files are memory-mapped, and a sharded
//! checkpoint (`model-00001-of-00005.safetensors`) is presented as one namespace.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};

use crate::error::{Error, Result};

/// Element types safetensors can declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "F64",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
        }
    }
}

#[derive(Debug, Clone)]
pub struct STensor {
    pub name: String,
    pub dtype: Dtype,
    /// Shape in maths order: slowest axis first, as PyTorch writes it.
    pub shape: Vec<u64>,
    /// Index into `SafeTensors::files`.
    pub file: usize,
    pub start: usize,
    pub end: usize,
}

impl STensor {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The same shape in GGUF's order: fastest axis first.
    pub fn gguf_dims(&self) -> Vec<u64> {
        self.shape.iter().rev().copied().collect()
    }
}

/// One or more safetensors files, presented as a single tensor namespace.
pub struct SafeTensors {
    files: Vec<Mmap>,
    paths: Vec<PathBuf>,
    tensors: BTreeMap<String, STensor>,
}

impl SafeTensors {
    /// Opens a single file, or every shard in a directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut shards: Vec<PathBuf> = if path.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(path)
                .map_err(|e| Error::Io {
                    path: path.to_path_buf(),
                    source: e,
                })?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .collect();
            // Shard order matters only for reproducibility of the tensor list.
            found.sort();
            found
        } else {
            vec![path.to_path_buf()]
        };

        if shards.is_empty() {
            return Err(Error::Corrupt(format!(
                "no .safetensors file found in {}",
                path.display()
            )));
        }
        shards.dedup();

        let mut files = Vec::new();
        let mut tensors = BTreeMap::new();

        for (idx, shard) in shards.iter().enumerate() {
            let file = File::open(shard).map_err(|e| Error::Io {
                path: shard.clone(),
                source: e,
            })?;
            // SAFETY: same caveat as the GGUF reader -- checkpoints are read-only
            // artifacts, and reading them fully into RAM is not viable.
            let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| Error::Io {
                path: shard.clone(),
                source: e,
            })?;

            parse_header(&mmap, idx, shard, &mut tensors)?;
            files.push(mmap);
        }

        Ok(Self {
            files,
            paths: shards,
            tensors,
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, STensor> {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&STensor> {
        self.tensors.get(name)
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn raw(&self, t: &STensor) -> Result<&[u8]> {
        self.files[t.file]
            .get(t.start..t.end)
            .ok_or_else(|| Error::Truncated {
                what: format!("tensor {:?}", t.name),
                need: t.end as u64,
                have: self.files[t.file].len() as u64,
            })
    }

    /// Reads a tensor as f32, widening whatever it was stored as.
    pub fn to_f32(&self, t: &STensor) -> Result<Vec<f32>> {
        let bytes = self.raw(t)?;
        let n = t.n_elements() as usize;
        let expected = n * t.dtype.size();
        if bytes.len() != expected {
            return Err(Error::Corrupt(format!(
                "tensor {:?} declares {} elements of {} ({expected} bytes) but spans {}",
                t.name,
                n,
                t.dtype.name(),
                bytes.len()
            )));
        }

        let mut out = vec![0f32; n];
        match t.dtype {
            Dtype::F32 => {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                }
            }
            Dtype::F16 => {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32();
                }
            }
            Dtype::BF16 => {
                for (i, v) in out.iter_mut().enumerate() {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    *v = f32::from_bits(u32::from(bits) << 16);
                }
            }
            Dtype::F64 => {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()) as f32;
                }
            }
            other => {
                return Err(Error::Corrupt(format!(
                    "tensor {:?} is {}; airun converts float tensors only",
                    t.name,
                    other.name()
                )));
            }
        }
        Ok(out)
    }
}

fn parse_header(
    mmap: &Mmap,
    file_index: usize,
    path: &Path,
    out: &mut BTreeMap<String, STensor>,
) -> Result<()> {
    if mmap.len() < 8 {
        return Err(Error::Truncated {
            what: format!("{} header length", path.display()),
            need: 8,
            have: mmap.len() as u64,
        });
    }
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_len;
    if data_start > mmap.len() {
        return Err(Error::Truncated {
            what: format!("{} JSON header", path.display()),
            need: data_start as u64,
            have: mmap.len() as u64,
        });
    }

    let json: serde_json::Value = serde_json::from_slice(&mmap[8..data_start])
        .map_err(|e| Error::Corrupt(format!("{}: invalid safetensors header: {e}", path.display())))?;

    let obj = json.as_object().ok_or_else(|| {
        Error::Corrupt(format!("{}: header is not a JSON object", path.display()))
    })?;

    for (name, spec) in obj {
        // Metadata is not a tensor.
        if name == "__metadata__" {
            continue;
        }
        let spec = spec.as_object().ok_or_else(|| {
            Error::Corrupt(format!("{}: entry {name:?} is not an object", path.display()))
        })?;

        let dtype_str = spec
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Corrupt(format!("tensor {name:?} has no dtype")))?;
        let dtype = Dtype::parse(dtype_str).ok_or_else(|| {
            Error::Corrupt(format!("tensor {name:?} has unsupported dtype {dtype_str:?}"))
        })?;

        let shape: Vec<u64> = spec
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Corrupt(format!("tensor {name:?} has no shape")))?
            .iter()
            .filter_map(|v| v.as_u64())
            .collect();

        let offsets = spec
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Corrupt(format!("tensor {name:?} has no data_offsets")))?;
        if offsets.len() != 2 {
            return Err(Error::Corrupt(format!(
                "tensor {name:?} has {} data_offsets, expected 2",
                offsets.len()
            )));
        }
        let rel_start = offsets[0].as_u64().unwrap_or(0) as usize;
        let rel_end = offsets[1].as_u64().unwrap_or(0) as usize;

        let start = data_start + rel_start;
        let end = data_start + rel_end;
        if end > mmap.len() || start > end {
            return Err(Error::Truncated {
                what: format!("tensor {name:?} in {}", path.display()),
                need: end as u64,
                have: mmap.len() as u64,
            });
        }

        out.insert(
            name.clone(),
            STensor {
                name: name.clone(),
                dtype,
                shape,
                file: file_index,
                start,
                end,
            },
        );
    }
    Ok(())
}
