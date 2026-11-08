//! GGUF container parsing.
//!
//! The file is memory-mapped and the header decoded eagerly; tensor payloads are
//! only ever handed out as borrowed slices into the mapping, so opening a 20 GB
//! checkpoint costs no resident memory beyond its metadata.
//!
//! A model split by `llama-gguf-split` (`name-00001-of-00003.gguf`, ...) opens
//! from any of its shards: the first carries the metadata, every shard its own
//! tensors, and each tensor remembers which shard it lives in.

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

/// Takes a mapped range's pages out of the working set (see
/// [`GgufFile::release`]). On Windows, VirtualUnlock on pages that were never
/// locked does exactly that (and reports ERROR_NOT_LOCKED, which is expected).
/// Elsewhere the kernel counts mapped file pages as reclaimable cache, so
/// there is nothing to do.
#[cfg(windows)]
fn release_pages(bytes: &[u8]) {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualUnlock(address: *const std::ffi::c_void, size: usize) -> i32;
    }
    if !bytes.is_empty() {
        // SAFETY: the range is part of a live mapping; VirtualUnlock only
        // changes which of its pages are resident, never their contents.
        unsafe { VirtualUnlock(bytes.as_ptr().cast(), bytes.len()) };
    }
}

#[cfg(not(windows))]
fn release_pages(_bytes: &[u8]) {}

/// Where one tensor lives inside the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in ggml order: fastest-varying axis first.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the tensor-data section.
    pub offset: u64,
    /// Absolute offset in its shard's file.
    pub file_offset: u64,
    /// Payload length in bytes.
    pub nbytes: u64,
    /// Index of the shard file holding the payload; 0 for unsplit models.
    pub shard: usize,
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
    version: u32,
    alignment: u64,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    tensor_index: BTreeMap<String, usize>,
    shards: Vec<Shard>,
}

struct Shard {
    path: PathBuf,
    data_offset: u64,
    file_len: u64,
    mmap: Mmap,
}

/// One file's header, before shards are merged.
struct Parsed {
    shard: Shard,
    version: u32,
    alignment: u64,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
}

/// `name-00002-of-00003.gguf` -> ("name", 2, 3).
fn split_name(path: &Path) -> Option<(String, u32, u32)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (rest, count) = stem.rsplit_once("-of-")?;
    let (prefix, no) = rest.rsplit_once('-')?;
    if no.len() != 5 || count.len() != 5 {
        return None;
    }
    Some((prefix.to_string(), no.parse().ok()?, count.parse().ok()?))
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let first = Self::parse(path, 0)?;
        let count = first.metadata.get("split.count").and_then(|v| v.as_u64()).unwrap_or(1);
        if count <= 1 {
            return Self::assemble(vec![first]);
        }
        // Any shard opens the whole model: find the siblings by name.
        let (prefix, _, named_count) = split_name(path).ok_or_else(|| Error::Corrupt(format!(
            "{} is one of {count} split files but is not named like `model-00001-of-{count:05}.gguf`; \
             restore the original shard names", path.display())))?;
        if u64::from(named_count) != count {
            return Err(Error::Corrupt(format!(
                "{} says it is split in {count} but its name says {named_count}; restore the original shard names",
                path.display())));
        }
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut parts = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let p = dir.join(format!("{prefix}-{:05}-of-{count:05}.gguf", i + 1));
            let part = Self::parse(&p, i)?;
            let no = part.metadata.get("split.no").and_then(|v| v.as_u64());
            if no != Some(i as u64) {
                return Err(Error::Corrupt(format!(
                    "{} should be shard {i} of {count} but says {no:?}; the split files do not belong together",
                    p.display())));
            }
            parts.push(part);
        }
        Self::assemble(parts)
    }

    /// Merges parsed shards: metadata from the first, tensors from all.
    fn assemble(parts: Vec<Parsed>) -> Result<Self> {
        let mut parts = parts.into_iter();
        let first = parts.next().expect("at least one shard");
        let mut tensors = first.tensors;
        let mut shards = vec![first.shard];
        for part in parts {
            tensors.extend(part.tensors);
            shards.push(part.shard);
        }
        let mut tensor_index = BTreeMap::new();
        for (i, t) in tensors.iter().enumerate() {
            if tensor_index.insert(t.name.clone(), i).is_some() {
                return Err(Error::Corrupt(format!(
                    "tensor {:?} appears twice across the split files", t.name)));
            }
        }
        Ok(Self {
            version: first.version,
            alignment: first.alignment,
            metadata: first.metadata,
            tensors,
            tensor_index,
            shards,
        })
    }

    fn parse(path: &Path, shard: usize) -> Result<Parsed> {
        let path = path.to_path_buf();
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

            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                file_offset,
                nbytes,
                shard,
            });
        }

        Ok(Parsed {
            shard: Shard { path, data_offset, file_len, mmap },
            version,
            alignment,
            metadata,
            tensors,
        })
    }

    /// The opened model's first shard (the only file, unless split).
    pub fn path(&self) -> &Path {
        &self.shards[0].path
    }

    /// Every file of the model, in shard order.
    pub fn shard_paths(&self) -> Vec<&Path> {
        self.shards.iter().map(|s| s.path.as_path()).collect()
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
    /// Start of the first shard's tensor data.
    pub fn data_offset(&self) -> u64 {
        self.shards[0].data_offset
    }
    /// Bytes across all shards.
    pub fn file_len(&self) -> u64 {
        self.shards.iter().map(|s| s.file_len).sum()
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
        let shard = self.shards.get(info.shard).ok_or_else(|| Error::Corrupt(format!(
            "tensor {:?} names shard {} of {}", info.name, info.shard, self.shards.len())))?;
        shard
            .mmap
            .get(start..start + len)
            .ok_or_else(|| Error::Truncated {
                what: format!("tensor {:?}", info.name),
                need: info.file_offset + info.nbytes,
                have: shard.file_len,
            })
    }

    /// Drops the pages of every tensor whose name starts with `prefix` from
    /// this process's working set, once they are uploaded. The data stays
    /// valid (a later read faults it back in from the file); only the RAM the
    /// read held goes back to the system. Without it, loading a 20 GB model
    /// held 20 GB of RAM "in use" until the load ended, so on a machine with
    /// little free RAM other apps had to be closed to load it.
    pub fn release(&self, prefix: &str) {
        for info in self.tensors.iter().filter(|t| t.name.starts_with(prefix)) {
            if let Ok(bytes) = self.tensor_bytes(info) {
                release_pages(bytes);
            }
        }
    }

    pub fn tensor_bytes_by_name(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensor(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))?;
        self.tensor_bytes(info)
    }

    /// Total size of the tensor-data sections.
    pub fn tensor_data_size(&self) -> u64 {
        self.shards.iter().map(|s| s.file_len.saturating_sub(s.data_offset)).sum()
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
    fn split_names_parse() {
        assert_eq!(split_name(Path::new("a/b/m-x-00002-of-00003.gguf")), Some(("m-x".into(), 2, 3)));
        assert_eq!(split_name(Path::new("model.gguf")), None);
        assert_eq!(split_name(Path::new("m-2-of-3.gguf")), None);
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
            shard: 0,
        };
        assert_eq!(t.n_elements(), 2048 * 4096);
        assert_eq!(t.row_size(), 2048);
        assert_eq!(t.n_rows(), 4096);
        assert_eq!(t.logical_shape(), vec![4096, 2048]);
    }
}
