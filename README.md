# candelabra

`candelabra` is a small Rust crate for desktop applications that want to run
quantized GGUF models (LLaMA, Qwen, Phi, Gemma, etc.) with
[`candle-core`](https://crates.io/crates/candle-core),
[`candle-transformers`](https://crates.io/crates/candle-transformers), and
[`hf-hub`](https://crates.io/crates/hf-hub).

It focuses on the pieces GUI apps usually need:

- Hugging Face downloads that respect the local `hf-hub` cache
- tokenizer loading helpers
- automatic Metal or CUDA fallback to CPU
- reusable loaded model state
- token streaming with cancellation support

## Current Scope

`candelabra` natively supports quantized GGUF checkpoints with dynamic architecture detection.
Supported architectures include:
- `llama` / `mistral` / `mixtral` / `gemma` / `gemma2`
- `phi3`
- `qwen2` (Qwen 2, Qwen 2.5, QwQ)
- `qwen3` / `qwen3moe`
- `gemma3`
- `glm4`

That means the crate is a good fit if you want a lightweight Rust API for local
desktop inference on models such as Qwen 2.5 or SmolLM GGUF variants. 
It abstracts away the `candle_transformers::models` paths into a single unified `Model` block.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
candelabra = "0.1"
```

## Quick Start

```rust,no_run
use candelabra::{
    download_model,
    load_tokenizer_from_repo,
    run_inference,
    InferenceConfig,
    Model,
};
use std::sync::{
    Arc,
    atomic::AtomicBool,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = download_model(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )?;
    let tokenizer = load_tokenizer_from_repo("Qwen/Qwen2.5-0.5B-Instruct")?;
    let mut model = Model::load(&model_path)?;
    let cancel_token = Arc::new(AtomicBool::new(false));

    let mut config = InferenceConfig::default();
    config.prompt = "<|im_start|>user\nTell me a story about a helpful robot.<|im_end|>\n<|im_start|>assistant\n".to_string();

    let result = run_inference(
        &mut model,
        &tokenizer,
        &config,
        cancel_token,
        |token| {
            print!("{token}");
            Ok(())
        },
    )?;

    println!("\n{:.2} tokens/s", result.tokens_per_second);
    Ok(())
}
```

## Main API

- `download_model()` downloads a model file through the local Hugging Face cache.
- `download_model_with_progress()` and `download_model_with_channel()` emit
  progress updates suitable for UI progress bars.
- `load_tokenizer_from_repo()` downloads and loads `tokenizer.json`.
- `Model::load()` loads a quantized GGUF model onto the best available
  device, dynamically instantiating the correct candle architecture base on metadata.
- `run_inference()` streams generated tokens through a callback.
- `run_inference_with_channel()` streams generated tokens over a Tokio channel.

## Platform Notes

- On macOS, the crate prefers Metal and falls back to CPU.
- On non-macOS platforms, the crate prefers CUDA and falls back to CPU.
- The public `device_used` string is intended to be easy to surface directly in
  desktop UIs.

## License

Licensed under either of these, at your option:

- Apache License, Version 2.0
- MIT license
