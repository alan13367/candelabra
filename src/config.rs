//! Configuration and result types for inference.

use serde::{Deserialize, Serialize};

/// Configuration for inference runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// HuggingFace repo ID (e.g. "bartowski/SmolLM2-360M-Instruct-GGUF")
    pub model_id: String,
    /// GGUF filename within the repo
    pub filename: String,
    /// Prompt fed to the model for token generation
    pub prompt: String,
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = deterministic, higher = more random)
    pub temperature: f64,
    /// Optional wall-clock time limit in seconds (stops generation early if hit)
    pub max_duration_secs: Option<u64>,
}

impl Default for InferenceConfig {
    /// Returns a lightweight default config using a small quantized model
    /// suitable for quick benchmarking without requiring large downloads.
    fn default() -> Self {
        Self {
            model_id: "bartowski/SmolLM2-360M-Instruct-GGUF".to_string(),
            filename: "SmolLM2-360M-Instruct-Q4_K_M.gguf".to_string(),
            prompt: "Tell me a story about a helpful robot.".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            max_duration_secs: Some(10),
        }
    }
}

/// Result produced by a completed inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Tokens generated per second (excludes prompt pre-fill)
    pub tokens_per_second: f64,
    /// Total number of tokens generated
    pub total_tokens: usize,
    /// Wall-clock duration of the generation loop in milliseconds
    pub duration_ms: u64,
    /// The text that was generated
    pub generated_text: String,
    /// The compute device used for inference (e.g. "Metal GPU", "CUDA GPU", "CPU")
    pub device_used: String,
}