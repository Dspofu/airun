//! The airun runtime: model configuration, memory planning and execution.

pub mod config;
pub mod error;
mod expert_cache;
mod gather_pool;
pub mod expert_profile;
pub mod grammar;
mod expert_reader;
pub mod loader;
pub mod model;
pub mod plan;
pub mod runtime;
pub mod sample;
pub mod tokenizer;

pub use config::{Architecture, Config};
pub use error::{CoreError, Result};
pub use loader::{LoadOptions, LoadReport, apply_chat_template, load};
pub use model::Model;
pub use plan::{Placement, PlacementRequest, Shard, SplitMode, WeightSizes};
pub use runtime::{GenerationStats, Runtime, StopReason};
pub use sample::{ProbsSink, Sampler, SamplingParams, TokenProbs};
pub use tokenizer::Tokenizer;
