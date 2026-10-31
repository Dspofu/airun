//! Native GGUF support for airun.
//!
//! This crate owns the on-disk model format end to end: reading containers,
//! decoding every quantization block layout, and (later) writing new ones from
//! safetensors checkpoints. It has no dependency on any external inference
//! runtime -- the block decoders here are the reference implementation that the
//! CUDA kernels are validated against.

pub mod dequant;
pub mod error;
pub mod ggml_type;
mod iq_grids;
pub mod quantize;
pub mod reader;
pub mod safetensors;
pub mod value;
pub mod writer;

pub use dequant::{decode_block, dequantize};
pub use error::{Error, Result};
pub use ggml_type::{BlockLayout, GgmlType, Support};
pub use quantize::{encode_block, parse_type, quantize};
pub use reader::{GgufFile, Metadata, TensorInfo};
pub use safetensors::{Dtype, STensor, SafeTensors};
pub use value::{Array, Value, ValueType};
pub use writer::GgufWriter;
