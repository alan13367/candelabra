//! Candelabra - A high-level, framework-agnostic wrapper around candle-core
//! for desktop GUI applications.
//!
//! This crate provides:
//! - Async model downloads with progress reporting
//! - Automatic hardware detection with Metal/CUDA/CPU fallback
//! - Reusable model/tokenizer state for repeated inference runs
//!
//! # Example
//!
//! ```no_run
//! use candelabra::{download_model, load_tokenizer_from_repo, LlamaModel, InferenceConfig, run_inference};
//! use std::sync::{Arc, atomic::AtomicBool};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model_path = download_model(
//!         "bartowski/SmolLM2-360M-Instruct-GGUF",
//!         "SmolLM2-360M-Instruct-Q4_K_M.gguf",
//!     )?;
//!     let tokenizer = load_tokenizer_from_repo("HuggingFaceTB/SmolLM2-360M-Instruct")?;
//!     let mut model = LlamaModel::load(&model_path)?;
//!     let cancel_token = Arc::new(AtomicBool::new(false));
//!     let config = InferenceConfig::default();
//!
//!     let _result = run_inference(
//!         &mut model,
//!         &tokenizer,
//!         &config,
//!         cancel_token,
//!         |_| Ok(()),
//!     )?;
//!
//!     Ok(())
//! }
//! ```

mod config;
mod device;
mod download;
mod inference;
mod model;

pub use config::{InferenceConfig, InferenceResult};
pub use device::{DeviceType, get_best_device, get_device};
pub use download::{
    DownloadProgress, check_model_cached, download_model, download_model_with_channel,
    download_model_with_progress, download_tokenizer, download_tokenizer_with_channel,
    download_tokenizer_with_progress, load_tokenizer, load_tokenizer_from_repo,
};
pub use inference::{run_inference, run_inference_with_channel};
pub use model::LlamaModel;

/// Error type for all candelabra operations.
#[derive(Debug, thiserror::Error)]
pub enum CandelabraError {
    /// Download failed
    #[error("Download error: {0}")]
    Download(String),

    /// Model loading failed
    #[error("Model error: {0}")]
    Model(String),

    /// Inference failed
    #[error("Inference error: {0}")]
    Inference(String),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Operation was cancelled
    #[error("Operation cancelled")]
    Cancelled,

    /// I/O error
    #[error("I/O error: {0}")]
    Io(String),

    /// Device error
    #[error("Device error: {0}")]
    Device(String),
}

impl From<std::io::Error> for CandelabraError {
    fn from(e: std::io::Error) -> Self {
        CandelabraError::Io(e.to_string())
    }
}

impl From<tokenizers::Error> for CandelabraError {
    fn from(e: tokenizers::Error) -> Self {
        CandelabraError::Tokenizer(e.to_string())
    }
}

impl From<candle_core::Error> for CandelabraError {
    fn from(e: candle_core::Error) -> Self {
        CandelabraError::Inference(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, CandelabraError>;