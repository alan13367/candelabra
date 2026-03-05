//! Token streaming inference with cancellation support.

use crate::{CandelabraError, InferenceConfig, InferenceResult, Model};
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;

const EOS_TOKEN_CANDIDATES: &[&str] = &["<|endoftext|>", "</s>", "<|eot_id|>", "<|end|>"];

/// Runs inference using reusable model and tokenizer state.
pub fn run_inference<F>(
    model: &mut Model,
    tokenizer: &Tokenizer,
    config: &InferenceConfig,
    cancel_token: Arc<AtomicBool>,
    mut on_token: F,
) -> Result<InferenceResult, CandelabraError>
where
    F: FnMut(String) -> Result<(), CandelabraError>,
{
    if cancel_token.load(Ordering::Relaxed) {
        return Err(CandelabraError::Cancelled);
    }

    let tokens = tokenizer
        .encode(config.prompt.clone(), true)
        .map_err(|e| CandelabraError::Tokenizer(format!("Encoding error: {}", e)))?;

    if config.max_tokens == 0 {
        return Ok(zero_result(model.device_name()));
    }

    let mut all_tokens = tokens.get_ids().to_vec();
    let mut generated_text = String::new();
    let mut logits_processor = LogitsProcessor::new(1337, Some(config.temperature), None);
    let eos_tokens = resolve_eos_token_ids(tokenizer);

    let input = Tensor::new(&all_tokens[..], &model.device)
        .map_err(|e| CandelabraError::Inference(format!("Prompt tensor creation error: {}", e)))?
        .unsqueeze(0)
        .map_err(|e| CandelabraError::Inference(format!("Prompt tensor unsqueeze error: {}", e)))?;

    let prompt_logits = model
        .weights
        .forward(&input, 0)
        .map_err(|e| CandelabraError::Inference(format!("Prompt forward pass error: {}", e)))?;
    let mut current_logits = prepare_logits(prompt_logits)?;

    // Measure generated-token throughput after prompt pre-fill.
    let start = Instant::now();
    let mut tokens_count = 0_usize;

    for _ in 0..config.max_tokens {
        if cancel_token.load(Ordering::Relaxed) {
            return Err(CandelabraError::Cancelled);
        }
        if hit_time_limit(start, config.max_duration_secs) {
            break;
        }

        let next_token = logits_processor
            .sample(&current_logits)
            .map_err(|e| CandelabraError::Inference(format!("Sampling error: {}", e)))?;

        all_tokens.push(next_token);
        tokens_count += 1;

        let token_text = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| CandelabraError::Tokenizer(format!("Decoding error: {}", e)))?;

        generated_text.push_str(&token_text);
        on_token(token_text)?;

        if eos_tokens.contains(&next_token) {
            break;
        }

        let input = Tensor::new(&[next_token], &model.device)
            .map_err(|e| CandelabraError::Inference(format!("Token tensor creation error: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| {
                CandelabraError::Inference(format!("Token tensor unsqueeze error: {}", e))
            })?;

        let generation_logits = model
            .weights
            .forward(&input, all_tokens.len() - 1)
            .map_err(|e| {
                CandelabraError::Inference(format!("Generation forward pass error: {}", e))
            })?;
        current_logits = prepare_logits(generation_logits)?;
    }

    let duration = start.elapsed();
    Ok(InferenceResult {
        tokens_per_second: calculate_tokens_per_second(tokens_count, duration),
        total_tokens: tokens_count,
        duration_ms: duration.as_millis() as u64,
        generated_text,
        device_used: model.device_name(),
    })
}

/// Runs inference and streams tokens over a Tokio channel.
pub fn run_inference_with_channel(
    model: &mut Model,
    tokenizer: &Tokenizer,
    config: &InferenceConfig,
    cancel_token: Arc<AtomicBool>,
    token_tx: Sender<String>,
) -> Result<InferenceResult, CandelabraError> {
    run_inference(model, tokenizer, config, cancel_token, move |token| {
        token_tx
            .blocking_send(token)
            .map_err(|_| CandelabraError::Cancelled)
    })
}

fn prepare_logits(logits: Tensor) -> Result<Tensor, CandelabraError> {
    let mut logits = logits
        .squeeze(0)
        .map_err(|e| CandelabraError::Inference(format!("Logits squeeze error: {}", e)))?;

    if logits.dims().len() > 1 {
        logits = logits
            .get(logits.dim(0)? - 1)
            .map_err(|e| CandelabraError::Inference(format!("Logits get error: {}", e)))?;
    }

    logits
        .clamp(-100.0, 100.0)
        .map_err(|e| CandelabraError::Inference(format!("Logits clamp error: {}", e)))
}

fn resolve_eos_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut eos_tokens = Vec::new();
    for candidate in EOS_TOKEN_CANDIDATES {
        if let Some(token_id) = tokenizer.token_to_id(candidate) {
            if !eos_tokens.contains(&token_id) {
                eos_tokens.push(token_id);
            }
        }
    }
    eos_tokens
}

fn hit_time_limit(start: Instant, max_duration_secs: Option<u64>) -> bool {
    match max_duration_secs {
        Some(limit) => start.elapsed().as_secs() >= limit,
        None => false,
    }
}

fn calculate_tokens_per_second(tokens_count: usize, duration: Duration) -> f64 {
    let elapsed = duration.as_secs_f64();
    if tokens_count == 0 || elapsed <= 0.0 {
        0.0
    } else {
        tokens_count as f64 / elapsed
    }
}

fn zero_result(device_used: String) -> InferenceResult {
    InferenceResult {
        tokens_per_second: 0.0,
        total_tokens: 0,
        duration_ms: 0,
        generated_text: String::new(),
        device_used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;

    fn build_test_tokenizer() -> Tokenizer {
        let vocab = HashMap::from([
            ("[UNK]".to_string(), 0),
            ("hello".to_string(), 1),
            ("</s>".to_string(), 2),
            ("<|endoftext|>".to_string(), 3),
        ]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("failed to build wordlevel tokenizer");
        Tokenizer::new(model)
    }

    #[test]
    fn zero_result_has_empty_metrics() {
        let result = zero_result("CPU".to_string());
        assert_eq!(result.total_tokens, 0);
        assert_eq!(result.duration_ms, 0);
        assert_eq!(result.tokens_per_second, 0.0);
        assert!(result.generated_text.is_empty());
    }

    #[test]
    fn calculate_tokens_per_second_handles_zero_duration() {
        assert_eq!(calculate_tokens_per_second(10, Duration::ZERO), 0.0);
        assert_eq!(calculate_tokens_per_second(0, Duration::from_secs(1)), 0.0);
    }

    #[test]
    fn resolves_common_eos_token_ids() {
        let tokenizer = build_test_tokenizer();
        let eos_tokens = resolve_eos_token_ids(&tokenizer);
        assert_eq!(eos_tokens, vec![3, 2]);
    }

    #[test]
    fn time_limit_zero_stops_immediately() {
        assert!(hit_time_limit(Instant::now(), Some(0)));
    }
}
