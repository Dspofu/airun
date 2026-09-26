//! Writing GGUF files.
//!
//! Tensor data is streamed to disk as it is produced, so converting a 20 GB
//! checkpoint never needs 20 GB of RAM. The header is written first with
//! placeholder offsets, then rewritten once the real offsets are known.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::ggml_type::GgmlType;
use crate::reader::{DEFAULT_ALIGNMENT, GGUF_MAGIC};
use crate::value::{Array, Value};

/// A tensor queued for writing.
struct PendingTensor {
    name: String,
    dims: Vec<u64>,
    ty: GgmlType,
    offset: u64,
    nbytes: u64,
}

/// Builds a GGUF file.
///
/// Metadata and the tensor list are declared up front, then payloads are written
/// in the same order they were declared.
pub struct GgufWriter {
    path: PathBuf,
    file: BufWriter<File>,
    metadata: Vec<(String, Value)>,
    tensors: Vec<PendingTensor>,
    alignment: u64,
    /// Running offset within the data section.
    data_cursor: u64,
    /// Absolute file offset where tensor data begins; set once the header is out.
    data_start: Option<u64>,
    written: usize,
}

impl GgufWriter {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        // Creating the destination directory is friendlier than failing an hour
        // into a conversion because the output path did not exist yet.
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| Error::Io {
                    path: parent.to_path_buf(),
                    source: e,
                })?;
            }
        }
        let file = File::create(&path).map_err(|e| Error::Io {
            path: path.clone(),
            source: e,
        })?;
        Ok(Self {
            path,
            file: BufWriter::with_capacity(8 << 20, file),
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
            data_cursor: 0,
            data_start: None,
            written: 0,
        })
    }

    pub fn set_metadata(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        // Last write wins, so callers can override a default they set earlier.
        if let Some(slot) = self.metadata.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.metadata.push((key, value));
        }
    }

    pub fn set_str(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.set_metadata(key, Value::String(value.into()));
    }

    pub fn set_u32(&mut self, key: impl Into<String>, value: u32) {
        self.set_metadata(key, Value::U32(value));
    }

    pub fn set_f32(&mut self, key: impl Into<String>, value: f32) {
        self.set_metadata(key, Value::F32(value));
    }

    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) {
        self.set_metadata(key, Value::Bool(value));
    }

    pub fn set_strings(&mut self, key: impl Into<String>, values: Vec<String>) {
        self.set_metadata(key, Value::Array(Array::String(values)));
    }

    pub fn set_i32_array(&mut self, key: impl Into<String>, values: Vec<i32>) {
        self.set_metadata(key, Value::Array(Array::I32(values)));
    }

    /// Declares a tensor. Payloads must later be written in declaration order.
    pub fn add_tensor(&mut self, name: impl Into<String>, dims: Vec<u64>, ty: GgmlType) -> Result<()> {
        let name = name.into();
        if self.data_start.is_some() {
            return Err(Error::Corrupt(format!(
                "tensor {name:?} was declared after the header was written"
            )));
        }
        let n_elements: u64 = dims.iter().product();
        let nbytes = ty.nbytes(n_elements).ok_or_else(|| {
            Error::Corrupt(format!(
                "tensor {name:?}: {n_elements} elements do not fill whole {ty} blocks"
            ))
        })?;

        self.tensors.push(PendingTensor {
            name,
            dims,
            ty,
            offset: self.data_cursor,
            nbytes,
        });
        // Every tensor starts on an alignment boundary.
        self.data_cursor = align_up(self.data_cursor + nbytes, self.alignment);
        Ok(())
    }

    /// Writes the header. Call once, after all tensors are declared.
    pub fn write_header(&mut self) -> Result<()> {
        if self.data_start.is_some() {
            return Err(Error::Corrupt("header written twice".into()));
        }

        self.raw(&GGUF_MAGIC.to_le_bytes())?;
        self.raw(&3u32.to_le_bytes())?; // version
        self.raw(&(self.tensors.len() as u64).to_le_bytes())?;
        self.raw(&(self.metadata.len() as u64).to_le_bytes())?;

        let metadata = std::mem::take(&mut self.metadata);
        for (key, value) in &metadata {
            self.write_string(key)?;
            self.write_value(value)?;
        }
        self.metadata = metadata;

        let tensors = std::mem::take(&mut self.tensors);
        for t in &tensors {
            self.write_string(&t.name)?;
            self.raw(&(t.dims.len() as u32).to_le_bytes())?;
            for d in &t.dims {
                self.raw(&d.to_le_bytes())?;
            }
            self.raw(&t.ty.id().to_le_bytes())?;
            self.raw(&t.offset.to_le_bytes())?;
        }
        self.tensors = tensors;

        let pos = self.file.stream_position().map_err(|e| self.io(e))?;
        let start = align_up(pos, self.alignment);
        self.pad_to(start)?;
        self.data_start = Some(start);
        Ok(())
    }

    /// Writes the next tensor's payload. Order must match declaration order.
    pub fn write_tensor_data(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let start = self
            .data_start
            .ok_or_else(|| Error::Corrupt("write_header must be called first".into()))?;

        let index = self.written;
        let expected = self.tensors.get(index).ok_or_else(|| {
            Error::Corrupt(format!(
                "tensor {name:?} written but only {} were declared",
                self.tensors.len()
            ))
        })?;

        if expected.name != name {
            return Err(Error::Corrupt(format!(
                "tensor payloads must be written in declaration order: expected \
                 {:?} but got {name:?}",
                expected.name
            )));
        }
        if data.len() as u64 != expected.nbytes {
            return Err(Error::Corrupt(format!(
                "tensor {name:?} is {} bytes but {} were declared",
                data.len(),
                expected.nbytes
            )));
        }

        self.pad_to(start + expected.offset)?;
        self.raw(data)?;
        self.written += 1;
        Ok(())
    }

    /// Flushes and closes, verifying that every declared tensor was written.
    pub fn finish(mut self) -> Result<PathBuf> {
        if self.written != self.tensors.len() {
            return Err(Error::Corrupt(format!(
                "only {} of {} declared tensors were written; the output would be \
                 truncated",
                self.written,
                self.tensors.len()
            )));
        }
        // Pad the tail so the last tensor's extent is a whole number of blocks.
        if let Some(start) = self.data_start {
            self.pad_to(start + self.data_cursor)?;
        }
        self.file.flush().map_err(|e| Error::Io {
            path: self.path.clone(),
            source: e,
        })?;
        Ok(self.path)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    // --- primitives ---

    fn io(&self, e: std::io::Error) -> Error {
        Error::Io {
            path: self.path.clone(),
            source: e,
        }
    }

    fn raw(&mut self, bytes: &[u8]) -> Result<()> {
        self.file.write_all(bytes).map_err(|e| Error::Io {
            path: self.path.clone(),
            source: e,
        })
    }

    fn pad_to(&mut self, target: u64) -> Result<()> {
        let pos = self.file.stream_position().map_err(|e| self.io(e))?;
        if pos > target {
            return Err(Error::Corrupt(format!(
                "cannot pad backwards from {pos} to {target}"
            )));
        }
        const ZEROS: [u8; 512] = [0; 512];
        let mut remaining = target - pos;
        while remaining > 0 {
            let n = remaining.min(ZEROS.len() as u64) as usize;
            self.raw(&ZEROS[..n])?;
            remaining -= n as u64;
        }
        Ok(())
    }

    fn write_string(&mut self, s: &str) -> Result<()> {
        self.raw(&(s.len() as u64).to_le_bytes())?;
        self.raw(s.as_bytes())
    }

    fn write_value(&mut self, v: &Value) -> Result<()> {
        match v {
            Value::U8(x) => { self.raw(&0u32.to_le_bytes())?; self.raw(&[*x]) }
            Value::I8(x) => { self.raw(&1u32.to_le_bytes())?; self.raw(&[*x as u8]) }
            Value::U16(x) => { self.raw(&2u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::I16(x) => { self.raw(&3u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::U32(x) => { self.raw(&4u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::I32(x) => { self.raw(&5u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::F32(x) => { self.raw(&6u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::Bool(x) => { self.raw(&7u32.to_le_bytes())?; self.raw(&[*x as u8]) }
            Value::String(s) => { self.raw(&8u32.to_le_bytes())?; self.write_string(s) }
            Value::U64(x) => { self.raw(&10u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::I64(x) => { self.raw(&11u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::F64(x) => { self.raw(&12u32.to_le_bytes())?; self.raw(&x.to_le_bytes()) }
            Value::Array(a) => {
                self.raw(&9u32.to_le_bytes())?;
                self.write_array(a)
            }
        }
    }

    fn write_array(&mut self, a: &Array) -> Result<()> {
        macro_rules! plain {
            ($tag:expr, $vec:expr, $conv:expr) => {{
                self.raw(&($tag as u32).to_le_bytes())?;
                self.raw(&($vec.len() as u64).to_le_bytes())?;
                for v in $vec {
                    #[allow(clippy::redundant_closure_call)]
                    self.raw(&$conv(v))?;
                }
                Ok(())
            }};
        }

        match a {
            Array::U8(v) => plain!(0, v, |x: &u8| [*x]),
            Array::I8(v) => plain!(1, v, |x: &i8| [*x as u8]),
            Array::U16(v) => plain!(2, v, |x: &u16| x.to_le_bytes()),
            Array::I16(v) => plain!(3, v, |x: &i16| x.to_le_bytes()),
            Array::U32(v) => plain!(4, v, |x: &u32| x.to_le_bytes()),
            Array::I32(v) => plain!(5, v, |x: &i32| x.to_le_bytes()),
            Array::F32(v) => plain!(6, v, |x: &f32| x.to_le_bytes()),
            Array::Bool(v) => plain!(7, v, |x: &bool| [*x as u8]),
            Array::U64(v) => plain!(10, v, |x: &u64| x.to_le_bytes()),
            Array::I64(v) => plain!(11, v, |x: &i64| x.to_le_bytes()),
            Array::F64(v) => plain!(12, v, |x: &f64| x.to_le_bytes()),
            Array::String(v) => {
                self.raw(&8u32.to_le_bytes())?;
                self.raw(&(v.len() as u64).to_le_bytes())?;
                for s in v {
                    self.write_string(s)?;
                }
                Ok(())
            }
            Array::Nested(_) => Err(Error::Corrupt(
                "nested metadata arrays are not written by airun".into(),
            )),
        }
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::GgufFile;

    #[test]
    fn a_written_file_reads_back_identically() {
        let dir = std::env::temp_dir().join("airun-writer-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.gguf");

        let mut w = GgufWriter::create(&path).unwrap();
        w.set_str("general.architecture", "qwen3");
        w.set_u32("qwen3.block_count", 28);
        w.set_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6);
        w.set_bool("tokenizer.ggml.add_bos_token", false);
        w.set_strings(
            "tokenizer.ggml.tokens",
            vec!["<pad>".into(), "hello".into(), "world".into()],
        );
        w.set_i32_array("tokenizer.ggml.token_type", vec![3, 1, 1]);

        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..32).map(|i| -(i as f32)).collect();
        w.add_tensor("alpha.weight", vec![64], GgmlType::F32).unwrap();
        w.add_tensor("beta.weight", vec![32], GgmlType::F32).unwrap();
        w.write_header().unwrap();

        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_tensor_data("alpha.weight", &a_bytes).unwrap();
        w.write_tensor_data("beta.weight", &b_bytes).unwrap();
        w.finish().unwrap();

        let f = GgufFile::open(&path).unwrap();
        assert_eq!(f.version(), 3);
        assert_eq!(f.architecture(), Some("qwen3"));
        assert_eq!(f.get("qwen3.block_count").unwrap().as_usize(), Some(28));
        assert_eq!(f.tensors().len(), 2);
        assert_eq!(
            f.get("tokenizer.ggml.tokens")
                .unwrap()
                .as_array()
                .unwrap()
                .as_strings()
                .unwrap(),
            &["<pad>".to_string(), "hello".into(), "world".into()]
        );

        let back = f.tensor_bytes_by_name("alpha.weight").unwrap();
        assert_eq!(back, a_bytes.as_slice());
        let back_b = f.tensor_bytes_by_name("beta.weight").unwrap();
        assert_eq!(back_b, b_bytes.as_slice());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writing_out_of_order_is_rejected() {
        let dir = std::env::temp_dir().join("airun-writer-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("order.gguf");

        let mut w = GgufWriter::create(&path).unwrap();
        w.set_str("general.architecture", "test");
        w.add_tensor("first", vec![32], GgmlType::F32).unwrap();
        w.add_tensor("second", vec![32], GgmlType::F32).unwrap();
        w.write_header().unwrap();

        let data = vec![0u8; 128];
        // "second" cannot be written before "first".
        assert!(w.write_tensor_data("second", &data).is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_incomplete_file_is_rejected_on_finish() {
        let dir = std::env::temp_dir().join("airun-writer-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("incomplete.gguf");

        let mut w = GgufWriter::create(&path).unwrap();
        w.set_str("general.architecture", "test");
        w.add_tensor("only", vec![32], GgmlType::F32).unwrap();
        w.write_header().unwrap();
        assert!(w.finish().is_err(), "finish must reject a missing payload");

        let _ = std::fs::remove_file(&path);
    }
}
