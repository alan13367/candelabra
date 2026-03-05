//! Model loading wrapper for GGUF format models.

use crate::device::{device_name, get_best_device};
use crate::{CandelabraError, DeviceType};
use candle_core::Tensor;
use candle_core::{quantized::gguf_file, DType, Device};
use candle_transformers::models::quantized_gemma3::ModelWeights as Gemma3Weights;
use candle_transformers::models::quantized_glm4::ModelWeights as Glm4Weights;
use candle_transformers::models::quantized_llama::ModelWeights as LlamaWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3Weights;
use std::path::Path;

/// Different quantized model architectures supported by candelabra
pub enum QuantizedWeights {
    Llama(LlamaWeights),
    Phi3(Phi3Weights),
    Qwen2(Qwen2Weights),
    Qwen3(Qwen3Weights),
    Gemma3(Gemma3Weights),
    Glm4(Glm4Weights),
}

impl QuantizedWeights {
    pub fn forward(
        &mut self,
        x: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor, candle_core::Error> {
        match self {
            Self::Llama(w) => w.forward(x, seqlen_offset),
            Self::Phi3(w) => w.forward(x, seqlen_offset),
            Self::Qwen2(w) => w.forward(x, seqlen_offset),
            Self::Qwen3(w) => w.forward(x, seqlen_offset),
            Self::Gemma3(w) => w.forward(x, seqlen_offset),
            Self::Glm4(w) => w.forward(x, seqlen_offset),
        }
    }
}

/// A loaded model ready for inference.
pub struct Model {
    /// The quantized model weights
    pub(crate) weights: QuantizedWeights,
    /// The compute device (Metal, CUDA, or CPU)
    pub(crate) device: Device,
    /// The type of device being used
    device_type: DeviceType,
}

impl Model {
    /// Load a GGUF model from the given path.
    ///
    /// This method automatically selects the best available device
    /// (Metal > CUDA > CPU) and loads the model weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    ///
    /// # Returns
    ///
    /// A loaded `Model` ready for inference.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CandelabraError> {
        let (device, device_type) = get_best_device();
        Self::load_with_device(path, device, device_type)
    }

    /// Load a GGUF model with a specific device.
    ///
    /// Use this when you want to control which device is used,
    /// such as for testing or when the user has selected a specific device.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    /// * `device` - The compute device to use
    /// * `device_type` - The type of device (for reporting purposes)
    ///
    /// # Returns
    ///
    /// A loaded `Model` ready for inference.
    pub fn load_with_device<P: AsRef<Path>>(
        path: P,
        device: Device,
        device_type: DeviceType,
    ) -> Result<Self, CandelabraError> {
        let path = path.as_ref();

        let mut file = std::fs::File::open(path)
            .map_err(|e| CandelabraError::Model(format!("Failed to open model file: {}", e)))?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| CandelabraError::Model(format!("Failed to read GGUF content: {}", e)))?;

        let architecture = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .ok_or_else(|| {
                CandelabraError::Model(
                    "Failed to find general.architecture in GGUF metadata".to_string(),
                )
            })?;

        // Note: Mistral and Gemma architectures use the LLaMA weights implementation under the hood
        // in candle-transformers, as well as many other models.
        let weights = match architecture.as_str() {
            "llama" | "mistral" | "gemma" | "gemma2" | "mixtral" => QuantizedWeights::Llama(
                LlamaWeights::from_gguf(content, &mut file, &device).map_err(|e| {
                    CandelabraError::Model(format!(
                        "Failed to load LLaMA/Mistral/Gemma weights: {}",
                        e
                    ))
                })?,
            ),
            "phi3" => QuantizedWeights::Phi3(
                Phi3Weights::from_gguf(false, content, &mut file, &device).map_err(|e| {
                    CandelabraError::Model(format!("Failed to load Phi3 weights: {}", e))
                })?,
            ),
            "qwen2" => QuantizedWeights::Qwen2(
                Qwen2Weights::from_gguf(content, &mut file, &device).map_err(|e| {
                    CandelabraError::Model(format!("Failed to load Qwen2 weights: {}", e))
                })?,
            ),
            "qwen3" => QuantizedWeights::Qwen3(
                Qwen3Weights::from_gguf(content, &mut file, &device).map_err(|e| {
                    CandelabraError::Model(format!("Failed to load Qwen3 weights: {}", e))
                })?,
            ),
            "gemma3" => QuantizedWeights::Gemma3(
                Gemma3Weights::from_gguf(content, &mut file, &device).map_err(|e| {
                    CandelabraError::Model(format!("Failed to load Gemma3 weights: {}", e))
                })?,
            ),
            "glm4" => QuantizedWeights::Glm4(
                Glm4Weights::from_gguf(content, &mut file, &device, DType::F32).map_err(|e| {
                    CandelabraError::Model(format!("Failed to load GLM4 weights: {}", e))
                })?,
            ),
            arch => {
                return Err(CandelabraError::Model(format!(
                    "Unsupported architecture: {}. Supported natively: llama, mistral, gemma, gemma2, mixtral, phi3, qwen2, qwen3, gemma3, glm4.",
                    arch
                )));
            }
        };

        Ok(Self {
            weights,
            device,
            device_type,
        })
    }

    /// Returns a human-readable name for the compute device in use.
    pub fn device_name(&self) -> String {
        device_name(&self.device)
    }

    /// Returns the device type selected for this model.
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Reset any internal state (KV cache, etc.).
    ///
    /// Currently a no-op but provided for future extensibility.
    pub fn reset(&mut self) {
        // Clear any internal KV cache or state here if needed in the future
    }
}
