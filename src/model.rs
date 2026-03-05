//! Model loading wrapper for GGUF format models.

use crate::device::{device_name, get_best_device};
use crate::{CandelabraError, DeviceType};
use candle_core::{Device, quantized::gguf_file};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;

/// A loaded LLaMA model ready for inference.
pub struct LlamaModel {
    /// The quantized model weights
    pub(crate) weights: ModelWeights,
    /// The compute device (Metal, CUDA, or CPU)
    pub(crate) device: Device,
    /// The type of device being used
    device_type: DeviceType,
}

impl LlamaModel {
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
    /// A loaded `LlamaModel` ready for inference.
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
    /// A loaded `LlamaModel` ready for inference.
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

        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| CandelabraError::Model(format!("Failed to load weights from GGUF: {}", e)))?;

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