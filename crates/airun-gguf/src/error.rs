//! Errors from GGUF parsing.
//!
//! Messages are written to be read by a user staring at a model that will not
//! load, so each one names the offending tensor or key and says what was
//! expected -- not just "invalid file".

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "not a GGUF file: magic is 0x{found:08X}, expected 0x46554747 \
         (a safetensors or PyTorch checkpoint needs `airun convert` first)"
    )]
    BadMagic { found: u32 },

    #[error("GGUF version {0} is not supported (this build reads v1 through v3)")]
    UnsupportedVersion(u32),

    #[error("metadata value type {0} is not a known GGUF type tag")]
    BadValueType(u32),

    #[error(
        "tensor {tensor:?} uses ggml type id {type_id}, which this build does not know; \
         the file was likely produced by a newer converter"
    )]
    UnknownTensorType { tensor: String, type_id: u32 },

    #[error("file is truncated: {what} needs {need} bytes but the file is {have}")]
    Truncated {
        what: String,
        need: u64,
        have: u64,
    },

    #[error("corrupt GGUF: {0}")]
    Corrupt(String),

    #[error("required tensor {0:?} is missing from the model")]
    MissingTensor(String),

    #[error("required metadata key {0:?} is missing from the model")]
    MissingKey(String),

    #[error("metadata key {key:?} is {found}, expected {expected}")]
    WrongKeyType {
        key: String,
        found: &'static str,
        expected: &'static str,
    },

    #[error("{context}: {source}")]
    Context {
        context: String,
        #[source]
        source: Box<Error>,
    },
}

impl Error {
    /// Attach a human-readable location to an error raised deeper in the parser.
    pub fn context(self, context: impl Into<String>) -> Self {
        Error::Context {
            context: context.into(),
            source: Box::new(self),
        }
    }
}
