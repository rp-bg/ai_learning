use crate::data::{BikeBatcher, load_data};
use crate::model::{LinearModel, NeuralModel};
use burn::{
    data::dataloader::DataLoaderBuilder,
    data::dataset::Dataset,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::AutodiffBackend,
};

const MODEL_PATH: &str = "artifacts/bike_model";
const NORMALIZER_PATH: &str = "artifacts/normalizer.json";

pub fn run_training<B: AutodiffBackend>(
    device: B::Device,
    dataset_path: &str,
    epochs: usize,
    batch_size: usize,
    lr: f64,
) {
    println!("Loading data from {}...", dataset_path);
    let (dataset, normalizer) = load_data(dataset_path);
    println!("Dataset size: {}", dataset.len());
    println!("Normalizer: {:?}", normalizer);

    let batcher = BikeBatcher::<B>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(dataset);

    // --- Train Linear Model ---
    println!("\nTraining Linear Model...");
    let mut model_linear = LinearModel::<B>::new(&device);
    let mut optim_linear = AdamConfig::new().init();
    let mut last_loss_linear = None;

    for epoch in 1..=epochs {
        let mut batch_count = 0;
        let mut epoch_loss = None;

        for batch in dataloader.iter() {
            let outputs = model_linear.forward(batch.inputs);
            let loss = MseLoss::new().forward(outputs, batch.targets, Reduction::Mean);

            epoch_loss = Some(loss.clone());
            batch_count += 1;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model_linear);
            model_linear = optim_linear.step(lr, model_linear, grads_params);
        }
        if epoch % 5 == 0 || epoch == 1 {
            if let Some(loss) = &epoch_loss {
                println!(
                    "Epoch {} | Batches: {} | Last Loss: {}",
                    epoch, batch_count, loss
                );
            }
        }
        last_loss_linear = epoch_loss;
    }
    println!("Linear Model Final Loss: {:?}", last_loss_linear);

    // --- Train Neural Model ---
    println!("\nTraining Neural Model (ReLU)...");
    let mut model_neural = NeuralModel::<B>::new(&device);
    let mut optim_neural = AdamConfig::new().init();
    let mut last_loss_neural = None;

    for epoch in 1..=epochs {
        let mut batch_count = 0;
        let mut epoch_loss = None;

        for batch in dataloader.iter() {
            let outputs = model_neural.forward(batch.inputs);
            let loss = MseLoss::new().forward(outputs, batch.targets, Reduction::Mean);

            epoch_loss = Some(loss.clone());
            batch_count += 1;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model_neural);
            model_neural = optim_neural.step(lr, model_neural, grads_params);
        }
        if epoch % 5 == 0 || epoch == 1 {
            if let Some(loss) = &epoch_loss {
                println!(
                    "Epoch {} | Batches: {} | Last Loss: {}",
                    epoch, batch_count, loss
                );
            }
        }
        last_loss_neural = epoch_loss;
    }
    println!("Neural Model Final Loss: {:?}", last_loss_neural);

    // --- Save Model and Normalizer ---
    println!("\nSaving model and normalizer...");

    // Create artifacts directory if it doesn't exist
    std::fs::create_dir_all("artifacts").expect("Failed to create artifacts directory");

    // Save the Neural Model (the better one)
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model_neural
        .save_file(MODEL_PATH, &recorder)
        .expect("Failed to save model");
    println!("Model saved to: {}.mpk", MODEL_PATH);

    // Save the Normalizer
    normalizer
        .save(NORMALIZER_PATH)
        .expect("Failed to save normalizer");
    println!("Normalizer saved to: {}", NORMALIZER_PATH);

    println!("\nTraining Complete.");
}
