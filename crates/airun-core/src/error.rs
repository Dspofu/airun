//! Errors from model loading and execution.
//!
//! Loading a model fails for a small number of recurring reasons, and each one
//! has a concrete fix. These messages name the fix rather than leaving the user
//! to guess, because "failed to load model" costs an hour of searching.

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error(transparent)]
    Gguf(#[from] airun_gguf::Error),

    #[error(transparent)]
    Cuda(#[from] airun_cuda::CudaError),

    #[error(
        "model metadata is missing {key:?}.\n  \
         This key is required to know the model's shape. If the file came from a \
         third-party converter, re-run `airun convert` on the original checkpoint."
    )]
    MissingConfig { key: String },

    #[error(
        "architecture {arch:?} is not supported yet: {reason}.\n  \
         Supported today: qwen3, qwen2 and llama-shaped dense decoders. \
         Run `airun inspect <model>` to see what the file declares."
    )]
    UnsupportedArchitecture { arch: String, reason: String },

    #[error("model {model:?} has a shape airun cannot run:\n{}", .problems.iter().map(|p| format!("  - {p}")).collect::<Vec<_>>().join("\n"))]
    UnsupportedShape { model: String, problems: Vec<String> },

    #[error(
        "model {model:?} needs {} but only {} of GPU memory is usable at context {ctx}.\n  \
         {detail}\n  \
         Try: a smaller --ctx-size, a more compact quantization, or adding a GPU with --gpus.",
        airun_cuda::human_bytes(*needed),
        airun_cuda::human_bytes(*available)
    )]
    DoesNotFit {
        model: String,
        needed: u64,
        available: u64,
        ctx: usize,
        detail: String,
    },

    #[error("no usable CUDA device; run `airun gpu` to see what was detected")]
    NoDevices,

    #[error(
        "--gpu-layers {requested} of {total} would leave layers on the CPU, which this \
         build cannot execute. Use --ctx-size to reduce memory instead, or add a GPU."
    )]
    PartialOffloadUnsupported { requested: usize, total: usize },

    #[error(
        "tensor {name:?} is missing from the model.\n  \
         The file declares architecture {arch:?}, which expects it. \
         The model may be a partial download or a split GGUF whose other parts are absent."
    )]
    MissingTensor { name: String, arch: String },

    #[error(
        "tensor {name:?} has shape {found:?} but {expected:?} was expected from the model's \
         own metadata. The file is inconsistent and cannot be loaded safely."
    )]
    ShapeMismatch {
        name: String,
        found: Vec<u64>,
        expected: Vec<u64>,
    },

    #[error(
        "tensor {name:?} is stored as {ty}, which has no GPU kernel in this build.\n  \
         Re-quantize with `airun quantize --type Q4_K` (or another supported type), \
         or run `airun inspect --tensors` to see the full type mix."
    )]
    UnsupportedTensorType { name: String, ty: String },

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("{0}")]
    Other(String),
}

impl CoreError {
    /// Whether retrying with a smaller memory footprint might succeed. Drives
    /// the automatic fallback ladder in the loader.
    pub fn is_capacity_error(&self) -> bool {
        match self {
            CoreError::DoesNotFit { .. } => true,
            CoreError::Cuda(e) => e.is_out_of_memory(),
            _ => false,
        }
    }
}
