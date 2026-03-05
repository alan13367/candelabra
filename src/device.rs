//! Hardware detection and device management with automatic fallback.

use crate::CandelabraError;
use candle_core::Device;
use serde::{Deserialize, Serialize};

/// The type of compute device available for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Apple Metal GPU (macOS)
    Metal,
    /// NVIDIA CUDA GPU
    Cuda,
    /// CPU fallback
    Cpu,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Metal => write!(f, "Metal GPU"),
            DeviceType::Cuda => write!(f, "CUDA GPU"),
            DeviceType::Cpu => write!(f, "CPU"),
        }
    }
}

/// Returns the best available compute device with automatic fallback.
///
/// Priority order:
/// 1. Metal (macOS) - if available and creation succeeds
/// 2. CUDA - if available and creation succeeds  
/// 3. CPU - always available as fallback
pub fn get_best_device() -> (Device, DeviceType) {
    #[cfg(target_os = "macos")]
    {
        if candle_core::utils::metal_is_available() {
            match Device::new_metal(0) {
                Ok(device) => {
                    return (device, DeviceType::Metal);
                }
                Err(e) => {
                    eprintln!("Metal device creation failed, falling back: {}", e);
                }
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        if candle_core::utils::cuda_is_available() {
            match Device::new_cuda(0) {
                Ok(device) => {
                    return (device, DeviceType::Cuda);
                }
                Err(e) => {
                    eprintln!("CUDA device creation failed, falling back: {}", e);
                }
            }
        }
    }

    (Device::Cpu, DeviceType::Cpu)
}

/// Returns the best available compute device with detailed error information.
///
/// Unlike `get_best_device`, this function returns an error if a preferred
/// device type is explicitly requested but unavailable.
pub fn get_device(preferred: Option<DeviceType>) -> Result<(Device, DeviceType), CandelabraError> {
    if let Some(device_type) = preferred {
        return match device_type {
            DeviceType::Metal => {
                #[cfg(target_os = "macos")]
                {
                    if candle_core::utils::metal_is_available() {
                        return Device::new_metal(0)
                            .map(|d| (d, DeviceType::Metal))
                            .map_err(|e| CandelabraError::Device(format!("Metal: {}", e)));
                    }
                }
                Err(CandelabraError::Device("Metal not available".to_string()))
            }
            DeviceType::Cuda => {
                #[cfg(not(target_os = "macos"))]
                {
                    if candle_core::utils::cuda_is_available() {
                        return Device::new_cuda(0)
                            .map(|d| (d, DeviceType::Cuda))
                            .map_err(|e| CandelabraError::Device(format!("CUDA: {}", e)));
                    }
                }
                Err(CandelabraError::Device("CUDA not available".to_string()))
            }
            DeviceType::Cpu => Ok((Device::Cpu, DeviceType::Cpu)),
        };
    }

    Ok(get_best_device())
}

/// Returns a human-readable name for a device.
pub fn device_name(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        #[cfg(not(target_os = "macos"))]
        Device::Cuda(_) => "CUDA GPU".to_string(),
        #[cfg(target_os = "macos")]
        Device::Metal(_) => "Metal GPU".to_string(),
        #[allow(unreachable_patterns)]
        _ => "Unknown".to_string(),
    }
}
