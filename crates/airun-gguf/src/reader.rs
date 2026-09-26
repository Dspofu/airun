//! GGUF container parsing.
//!
//! The file is memory-mapped and the header decoded eagerly; tensor payloads are
//! only ever handed out as borrowed slices into the mapping, so opening a 20 GB
//! checkpoint costs no resident memory beyond its metadata.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};

use crate::error::{Error, Result};
use crate::ggml_type::GgmlType;
use crate::value::{Array, Value, ValueType};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Guards against a corrupt length field causing a huge allocation. No real
/// metadata string or array comes close to these.
const MAX_STRING_LEN: u64 = 64 << 20; // 64 MiB
const MAX_ARRAY_LEN: u64 = 64 << 20; // 64M elements

/// Where one tensor lives inside the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in ggml order: fastest-varying axis first.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the tensor-data section.
    pub offset: u64,
    /// Absolute offset in the file.
    pub file_offset: u64,
    /// Payload length in bytes.
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Row length, i.e. the fastest-varying dimension.
    pub fn row_size(&self) -> u64 {
        self.dims.first().copied().unwrap_or(0)
    }

    pub fn n_rows(&self) -> u64 {
        self.dims.iter().skip(1).product()
    }

    /// Shape in the conventional maths order (slowest axis first), which is how
    /// the same tensor is written in a safetensors checkpoint.
    pub fn logical_shape(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }
}

/// Metadata key/value store, ordered so `inspect` output is stable.
pub type Metadata = BTreeMap<String, Value>;

pub struct GgufFile {
    path: PathBuf,
    version: u32,
    alignment: u64,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    tensor_index: BTreeMap<String, usize>,
    data_offset: u64,
    file_len: u64,
    mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| Error::Io {
            path: path.clone(),
            source: e,
        })?;
        let file_len = file
            .metadata()
            .map_err(|e| Error::Io {
                path: path.clone(),
                source: e,
            })?
            .len();

        if file_len < 24 {
            return Err(Error::Truncated {
                what: "header".into(),
                need: 24,
                have: file_len,
            });
        }

        // SAFETY: the usual mmap caveat -- another process truncating or writing
        // the file under us is UB. Model files are read-only artifacts in
        // practice, and the alternative (reading 20 GB into RAM) is not viable.
        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| Error::Io {
            path: path.clone(),
            source: e,
        })?;

        let mut cur = Cursor::new(&mmap);

        let magic = cur.u32()?;
        if magic != GGUF_MAGIC {
            return Err(Error::BadMagic { found: magic });
        }
        let version = cur.u32()?;
        if !(1..=3).contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }

        // v1 counted with u32; v2+ switched to u64.
        let (tensor_count, kv_count) = if version == 1 {
            (cur.u32()? as u64, cur.u32()? as u64)
        } else {
            (cur.u64()?, cur.u64()?)
        };

        if tensor_count > 1 << 22 {
            return Err(Error::Corrupt(format!(
                "implausible tensor count {tensor_count}"
            )));
        }
        if kv_count > 1 << 20 {
            return Err(Error::Corrupt(format!(
                "implausible metadata count {kv_count}"
            )));
        }

        let mut metadata = Metadata::new();
        for i in 0..kv_count {
            let key = cur
                .string(version)
                .map_err(|e| e.context(format!("metadata key #{i}")))?;
            let value = cur
                .value(version)
                .map_err(|e| e.context(format!("metadata value for {key:?}")))?;
            // Later duplicates win, matching upstream loader behaviour.
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::Corrupt(format!(
                "general.alignment must be a power of two, got {alignment}"
            )));
        }

        let mut raw: Vec<(String, Vec<u64>, GgmlType, u64)> =
            Vec::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = cur
                .string(version)
                .map_err(|e| e.context(format!("tensor name #{i}")))?;
            let n_dims = cur.u32()?;
            if n_dims == 0 || n_dims > 4 {
                return Err(Error::Corrupt(format!(
                    "tensor {name:?} declares {n_dims} dimensions (expected 1..=4)"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(if version == 1 {
                    cur.u32()? as u64
                } else {
                    cur.u64()?
                });
            }
            let ty = GgmlType::from_id(cur.u32()?);
            let offset = cur.u64()?;
            raw.push((name, dims, ty, offset));
        }

        let data_offset = align_up(cur.pos() as u64, alignment);
        if data_offset > file_len {
            return Err(Error::Truncated {
                what: "tensor data section".into(),
                need: data_offset,
                have: file_len,
            });
        }

        let mut tensors = Vec::with_capacity(raw.len());
        let mut tensor_index = BTreeMap::new();
        for (name, dims, ty, offset) in raw {
            let n_elements: u64 = dims.iter().product();
            let nbytes = ty.nbytes(n_elements).ok_or_else(|| {
                if ty.layout().is_none() {
                    Error::UnknownTensorType {
                        tensor: name.clone(),
                        type_id: ty.id(),
                    }
                } else {
                    Error::Corrupt(format!(
                        "tensor {name:?} has {n_elements} elements, not a multiple of the \
                         {ty} block size"
                    ))
                }
            })?;

            let file_offset = data_offset
                .checked_add(offset)
                .ok_or_else(|| Error::Corrupt(format!("tensor {name:?} offset overflows")))?;
            let end = file_offset
                .checked_add(nbytes)
                .ok_or_else(|| Error::Corrupt(format!("tensor {name:?} extent overflows")))?;
            if end > file_len {
                return Err(Error::Truncated {
                    what: format!("tensor {name:?}"),
                    need: end,
                    have: file_len,
                });
            }

            tensor_index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                file_offset,
                nbytes,
            });
        }

        Ok(Self {
            path,
            version,
            alignment,
            metadata,
            tensors,
            tensor_index,
            data_offset,
            file_len,
            mmap,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// Look up a key that may be namespaced by architecture, e.g.
    /// `qwen3.block_count` given `suffix = "block_count"`.
    pub fn get_arch(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.{suffix}"))
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Raw bytes of a tensor, borrowed from the mapping.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = usize::try_from(info.file_offset)
            .map_err(|_| Error::Corrupt("tensor offset exceeds address space".into()))?;
        let len = usize::try_from(info.nbytes)
            .map_err(|_| Error::Corrupt("tensor length exceeds address space".into()))?;
        self.mmap
            .get(start..start + len)
            .ok_or_else(|| Error::Truncated {
                what: format!("tensor {:?}", info.name),
                need: info.file_offset + info.nbytes,
                have: self.file_len,
            })
    }

    pub fn tensor_bytes_by_name(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensor(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        self.tensor_bytes(info)
    }

    /// Total size of the tensor-data section.
    pub fn tensor_data_size(&self) -> u64 {
        self.file_len.saturating_sub(self.data_offset)
    }

    /// Count of tensors per quantization type, for `inspect` summaries.
    pub fn type_histogram(&self) -> BTreeMap<String, (usize, u64)> {
        let mut hist: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        for t in &self.tensors {
            let e = hist.entry(t.ty.to_string()).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.nbytes;
        }
        hist
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

/// Bounds-checked sequential reader over the mapped bytes.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| Error::Truncated {
            what: "header field".into(),
            need: u64::MAX,
            have: self.buf.len() as u64,
        })?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| Error::Truncated {
            what: "header field".into(),
            need: end as u64,
            have: self.buf.len() as u64,
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    fn string(&mut self, version: u32) -> Result<String> {
        let len = if version == 1 {
            self.u32()? as u64
        } else {
            self.u64()?
        };
        if len > MAX_STRING_LEN {
            return Err(Error::Corrupt(format!(
                "string length {len} exceeds the {MAX_STRING_LEN} byte sanity limit"
            )));
        }
        let bytes = self.take(len as usize)?;
        // Some converters emit raw byte-level BPE tokens that are not valid
        // UTF-8. Replacing is the only way to keep such files loadable.
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, version: u32) -> Result<Value> {
        let tag = self.u32()?;
        let ty = ValueType::from_id(tag).ok_or(Error::BadValueType(tag))?;
        Ok(match ty {
            ValueType::U8 => Value::U8(self.u8()?),
            ValueType::I8 => Value::I8(self.i8()?),
            ValueType::U16 => Value::U16(self.u16()?),
            ValueType::I16 => Value::I16(self.i16()?),
            ValueType::U32 => Value::U32(self.u32()?),
            ValueType::I32 => Value::I32(self.i32()?),
            ValueType::U64 => Value::U64(self.u64()?),
            ValueType::I64 => Value::I64(self.i64()?),
            ValueType::F32 => Value::F32(self.f32()?),
            ValueType::F64 => Value::F64(self.f64()?),
            ValueType::Bool => Value::Bool(self.bool()?),
            ValueType::String => Value::String(self.string(version)?),
            ValueType::Array => Value::Array(self.array(version)?),
        })
    }

    fn array(&mut self, version: u32) -> Result<Array> {
        let tag = self.u32()?;
        let ty = ValueType::from_id(tag).ok_or(Error::BadValueType(tag))?;
        let len = if version == 1 {
            self.u32()? as u64
        } else {
            self.u64()?
        };
        if len > MAX_ARRAY_LEN {
            return Err(Error::Corrupt(format!(
                "array length {len} exceeds the {MAX_ARRAY_LEN} element sanity limit"
            )));
        }
        let n = len as usize;

        // Reserve lazily via with_capacity on the element count only after the
        // sanity check above, so a corrupt header cannot drive a huge alloc.
        Ok(match ty {
            ValueType::U8 => Array::U8(self.take(n)?.to_vec()),
            ValueType::I8 => {
                Array::I8(self.take(n)?.iter().map(|&b| b as i8).collect())
            }
            ValueType::U16 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.u16()?);
                }
                Array::U16(v)
            }
            ValueType::I16 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.i16()?);
                }
                Array::I16(v)
            }
            ValueType::U32 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.u32()?);
                }
                Array::U32(v)
            }
            ValueType::I32 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.i32()?);
                }
                Array::I32(v)
            }
            ValueType::U64 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.u64()?);
                }
                Array::U64(v)
            }
            ValueType::I64 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.i64()?);
                }
                Array::I64(v)
            }
            ValueType::F32 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.f32()?);
                }
                Array::F32(v)
            }
            ValueType::F64 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.f64()?);
                }
                Array::F64(v)
            }
            ValueType::Bool => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.bool()?);
                }
                Array::Bool(v)
            }
            ValueType::String => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.string(version)?);
                }
                Array::String(v)
            }
            ValueType::Array => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(version)?);
                }
                Array::Nested(v)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_next_multiple() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn cursor_refuses_to_read_past_the_end() {
        let buf = [1u8, 2, 3];
        let mut c = Cursor::new(&buf);
        assert_eq!(c.u16().unwrap(), 0x0201);
        assert!(c.u32().is_err());
    }

    #[test]
    fn tensor_info_shape_helpers() {
        let t = TensorInfo {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![2048, 4096],
            ty: GgmlType::Q4_K,
            offset: 0,
            file_offset: 0,
            nbytes: 0,
        };
        assert_eq!(t.n_elements(), 2048 * 4096);
        assert_eq!(t.row_size(), 2048);
        assert_eq!(t.n_rows(), 4096);
        assert_eq!(t.logical_shape(), vec![4096, 2048]);
    }
}
