mod attention;
mod data;
mod generate;
mod model;
mod train;

use burn::backend::{Autodiff, Wgpu, wgpu::WgpuDevice};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use clap::{Parser, Subcommand};
use std::fs;

#[derive(Parser)]
#[command(name = "TinyGPT CLI")]
#[command(about = "Train or generate text with a Tiny GPT model", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model on the provided dataset
    Train,
    /// Generate text using a trained model
    Generate {
        /// The prompt to start generation
        #[arg(short, long, default_value = "The ")]
        prompt: String,
        /// Number of tokens to generate
        #[arg(short, long, default_value_t = 100)]
        length: usize,
        /// Sampling temperature
        #[arg(short, long, default_value_t = 0.8)]
        temperature: f32,
    },
}

fn main() {
    let cli = Cli::parse();

    // Setup Backend and Paths
    type MyBackend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    let model_path = "artifacts/lesson8/model";
    let train_path = "data/lesson8/train.txt";
    let val_path = "data/lesson8/val.txt";

    // Load Tokenizer (needed for both commands)
    let combined_text = format!(
        "{}{}",
        fs::read_to_string(train_path).expect("Train file missing"),
        fs::read_to_string(val_path).expect("Val file missing")
    );
    let tokenizer = data::Tokenizer::new(&combined_text);

    match cli.command {
        Commands::Train => {
            println!("Using device: {:?}", device);
            println!(
                "Tokenizer initialized with {} characters.",
                tokenizer.vocab_size()
            );
            println!("\n--- Starting Training ---");
            let trained_model = train::train::<MyBackend>(device.clone(), &tokenizer, train_path);

            // Save the model
            fs::create_dir_all("artifacts/lesson8").ok();
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            trained_model
                .save_file(model_path, &recorder)
                .expect("Failed to save model");
            println!("Model saved to {}", model_path);
        }
        Commands::Generate {
            prompt,
            length,
            temperature,
        } => {
            println!("\n--- Loading Model for Generation ---");
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

            let gen_config = model::TinyGPTConfig::new(
                tokenizer.vocab_size(),
                64,  // block_size
                4,   // n_layers
                128, // d_model
                4,   // n_heads
                0.0, // no dropout for gen
            );

            let mut gen_model = gen_config.init::<Wgpu>(&device);
            gen_model = gen_model
                .load_file(model_path, &recorder, &device)
                .expect("Failed to load model. Did you train it first?");

            println!("Prompt: \"{}\"", prompt);
            let generated = generate::generate::<Wgpu>(
                &gen_model,
                &tokenizer,
                &prompt,
                length,
                temperature,
                &device,
            );

            println!("\nResult:\n{}", generated);
        }
    }
}
