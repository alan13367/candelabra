use candelabra::{download_model, load_tokenizer_from_repo, run_inference, InferenceConfig, Model};
use std::sync::{atomic::AtomicBool, Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Phi-3-mini-4k-instruct (Phi3 architecture)...");
    let model_path = download_model(
        "microsoft/Phi-3-mini-4k-instruct-gguf",
        "Phi-3-mini-4k-instruct-q4.gguf",
    )?;

    // Using an alternative repo that holds tokenizer.json if needed,
    // but microsoft/Phi-3-mini-4k-instruct has tokenizer.json
    let tokenizer = load_tokenizer_from_repo("microsoft/Phi-3-mini-4k-instruct")?;

    let mut model = Model::load(&model_path)?;
    println!("Loaded model on device: {}", model.device_name());

    let cancel_token = Arc::new(AtomicBool::new(false));
    let mut config = InferenceConfig::default();
    config.prompt =
        "<|user|>\nWrite a very short poem about a rust compiler.\n<|end|>\n<|assistant|>"
            .to_string();
    config.max_tokens = 50;

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
