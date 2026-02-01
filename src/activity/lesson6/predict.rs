use crate::data::Normalizer;
use crate::model::NeuralModel;
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, backend::Backend},
};

const MODEL_PATH: &str = "artifacts/bike_model";
const NORMALIZER_PATH: &str = "artifacts/normalizer.json";

pub fn run_inference<B: Backend>(
    device: B::Device,
    input_values: [f32; 5], // [hr, temp, hum, windspeed, workingday]
) {
    // 1. Load the Normalizer
    let normalizer = match Normalizer::load(NORMALIZER_PATH) {
        Ok(n) => {
            println!("Loaded normalizer from: {}", NORMALIZER_PATH);
            n
        }
        Err(e) => {
            eprintln!(
                "Error: Could not load normalizer from '{}': {}",
                NORMALIZER_PATH, e
            );
            eprintln!("Please run 'train' first to create the model and normalizer.");
            return;
        }
    };

    // 2. Load the trained model
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model: NeuralModel<B> =
        match NeuralModel::new(&device).load_file(MODEL_PATH, &recorder, &device) {
            Ok(m) => {
                println!("Loaded model from: {}.mpk", MODEL_PATH);
                m
            }
            Err(e) => {
                eprintln!(
                    "Error: Could not load model from '{}.mpk': {}",
                    MODEL_PATH, e
                );
                eprintln!("Please run 'train' first to create the model and normalizer.");
                return;
            }
        };

    // 3. Normalize the input
    let normalized_input = normalizer.normalize(&input_values);
    println!("\nRaw input:        {:?}", input_values);
    println!("Normalized input: {:?}", normalized_input);

    // 4. Create tensor and run forward pass
    let input_tensor = Tensor::<B, 1>::from_floats(normalized_input, &device).reshape([1, 5]);
    let output = model.forward(input_tensor);

    // 5. Get the predicted value and denormalize
    // The output is in 0-1 range (normalized by target_max during training)
    let output_data: Vec<f32> = output.to_data().to_vec().unwrap();
    let normalized_prediction = output_data[0];
    let actual_prediction = normalizer.denormalize_target(normalized_prediction);

    println!("\n--- Prediction ---");
    println!("Normalized output: {:.4}", normalized_prediction);
    println!("Predicted bike count: {:.0}", actual_prediction);
}
