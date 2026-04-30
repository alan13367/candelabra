use candelabra::{download_model, load_tokenizer_from_repo, run_inference, InferenceConfig, Model};
use std::sync::{atomic::AtomicBool, Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Qwen2.5-0.5B-Instruct (Qwen2 architecture)...");
    let model_path = download_model(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )?;

    let tokenizer = load_tokenizer_from_repo("Qwen/Qwen2.5-0.5B-Instruct")?;

    let mut model = Model::load(&model_path)?;
    println!("Loaded model on device: {}", model.device_name());

    let cancel_token = Arc::new(AtomicBool::new(false));
    let config = InferenceConfig {
        prompt: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWrite a very short poem about algorithms.<|im_end|>\n<|im_start|>assistant\n".to_string(),
        max_tokens: 50,
        ..Default::default()
    };

    let result = run_inference(&mut model, &tokenizer, &config, cancel_token, |token| {
        print!("{}", token);
        use std::io::Write;
        let _ = std::io::stdout().flush();
        Ok(())
    })?;

    println!(
        "\nInference finished. Generated {} tokens in {} ms. Speed: {:.2} t/s",
        result.total_tokens, result.duration_ms, result.tokens_per_second
    );

    Ok(())
}
