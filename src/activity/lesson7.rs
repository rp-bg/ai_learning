use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::Module,
    nn::loss::{MseLoss, Reduction},
    nn::{Linear, LinearConfig, Relu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
        cast::ToElement,
    },
};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;

const ARTIFACTS_DIR: &str = "artifacts/lesson7";
const MODEL_FILE: &str = "artifacts/lesson7/model";
const STATS_FILE: &str = "artifacts/lesson7/stats.json";

#[derive(Parser)]
#[command(name = "Lesson 7 Bike Predictor")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model
    Train,
    /// Predict bike demand
    Predict {
        /// Hour (0-23)
        #[arg(long)]
        hr: f32,
        /// Temperature (standardized)
        #[arg(long)]
        temp: f32,
        /// Humidity (standardized)
        #[arg(long)]
        hum: f32,
        /// Windspeed (standardized)
        #[arg(long)]
        windspeed: f32,
        /// Working day (0 or 1)
        #[arg(long)]
        workingday: f32,
    },
}

#[derive(Serialize, Deserialize)]
struct Stats {
    pub max_cnt: f32,
}

/*
    Lesson 7: Batched DataLoader and Training Loop

    In previous lessons, we trained on the entire dataset at once.
    For larger datasets, this is impossible (memory limits) or inefficient.
    "Mini-batching" involves splitting the data into small groups (batches).

    This lesson covers:
    1. Defining a Batcher: How to transform a list of items into a Tensor batch.
    2. Using DataLoader: Burn's utility for shuffling and batching data.
    3. The Batched Training Loop: Iterating over batches and updating weights.
*/

// --- 1. Data Structures ---

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BikeRecord {
    pub hr: f32,
    pub holiday: u8,
    pub workingday: f32,
    pub temp: f32,
    pub hum: f32,
    pub windspeed: f32,
    pub cnt: f32,
}

#[derive(Clone, Debug)]
pub struct BikeItem {
    pub features: [f32; 5],
    pub target: [f32; 1],
}

// --- 2. Batching Logic ---

/// The Batcher is responsible for taking a Vec of items from the dataset
/// and converting them into a single Batch object containing Tensors.
#[derive(Clone)]
pub struct BikeBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> BikeBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

/// A Batch contains the inputs and targets ready for the model.
#[derive(Clone, Debug)]
pub struct BikeBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Batcher<BikeItem, BikeBatch<B>> for BikeBatcher<B> {
    /// This method is called by the DataLoader to create a batch.
    fn batch(&self, items: Vec<BikeItem>) -> BikeBatch<B> {
        let batch_size = items.len();

        // 1. Flatten all features and targets into flat vectors
        let flat_inputs: Vec<f32> = items.iter().flat_map(|i| i.features).collect();
        let flat_targets: Vec<f32> = items.iter().flat_map(|i| i.target).collect();

        // 2. Create Tensors from the flat data and reshape into [BatchSize, Dim]
        let inputs = Tensor::<B, 1>::from_floats(flat_inputs.as_slice(), &self.device)
            .reshape([batch_size, 5]);

        let targets = Tensor::<B, 1>::from_floats(flat_targets.as_slice(), &self.device)
            .reshape([batch_size, 1]);

        BikeBatch { inputs, targets }
    }
}

// --- 3. Model Definition ---

#[derive(Module, Debug)]
pub struct BikeNN<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

impl<B: Backend> BikeNN<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(5, 16).init(device),
            activation: Relu::new(),
            linear2: LinearConfig::new(16, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

// --- 4. Data Loading Utility ---

fn prepare_data(path: &str) -> (InMemDataset<BikeItem>, f32) {
    let mut reader =
        csv::Reader::from_path(path).expect("Make sure hour.csv is in data/lesson7/ directory");

    let mut items = Vec::new();
    let mut max_cnt = 1.0;

    // First pass to find max_cnt for normalization
    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: BikeRecord = result.expect("Error parsing CSV");
        if record.cnt > max_cnt {
            max_cnt = record.cnt;
        }
        records.push(record);
    }

    // Second pass to create normalized items
    for r in records {
        items.push(BikeItem {
            features: [
                r.hr / 23.0,  // Scale hour to 0-1
                r.temp,       // Already normalized in dataset
                r.hum,        // Already normalized in dataset
                r.windspeed,  // Already normalized in dataset
                r.workingday, // 0 or 1
            ],
            target: [r.cnt / max_cnt], // Normalize target
        });
    }

    (InMemDataset::new(items), max_cnt)
}

// --- 5. Training Function ---

fn train<B: AutodiffBackend>(device: B::Device) {
    // Hyperparameters
    let batch_size = 32;
    let epochs = 10;
    let lr = 1e-3;

    // Setup Data
    let (dataset, max_cnt) = prepare_data("data/lesson7/hour.csv");
    let batcher = BikeBatcher::<B>::new(device.clone());

    // The DataLoader manages the batcher, dataset, shuffling, and worker threads.
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(42) // Constant seed for reproducibility
        .build(dataset);

    // Setup Model and Optimizer
    let mut model = BikeNN::<B>::new(&device);
    let mut optimizer = AdamConfig::new().init();

    println!("Starting training with batch size: {}", batch_size);

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in dataloader.iter() {
            let output = model.forward(batch.inputs);
            let loss = MseLoss::new().forward(output, batch.targets, Reduction::Mean);
            total_loss += loss.clone().into_scalar().to_f32();
            batch_count += 1;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads_params);
        }

        println!(
            "Epoch {}/{} - Average Loss: {:.6}",
            epoch,
            epochs,
            total_loss / batch_count as f32
        );
    }

    // --- Save Artifacts ---
    fs::create_dir_all(ARTIFACTS_DIR).ok();

    // Save Model
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(MODEL_FILE, &recorder)
        .expect("Failed to save model");

    // Save Stats
    let stats = Stats { max_cnt };
    let stats_json = serde_json::to_string(&stats).unwrap();
    fs::write(STATS_FILE, stats_json).expect("Failed to save stats");

    println!("Training complete. Model saved to: {}", MODEL_FILE);
}

fn predict<B: Backend>(
    device: B::Device,
    hr: f32,
    temp: f32,
    hum: f32,
    windspeed: f32,
    workingday: f32,
) {
    // 1. Setup Model and Load Weights
    let mut model = BikeNN::<B>::new(&device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model = model
        .load_file(MODEL_FILE, &recorder, &device)
        .expect("Failed to load model. Did you train first?");

    // 2. Load Stats
    let stats_json = fs::read_to_string(STATS_FILE).expect("Failed to read stats.json");
    let stats: Stats = serde_json::from_str(&stats_json).unwrap();

    // 3. Prepare Input
    let input_data = [hr / 23.0, temp, hum, windspeed, workingday];
    let input = Tensor::<B, 2>::from_floats([input_data], &device);

    // 4. Inference
    let output = model.forward(input);
    let normalized_demand = output.into_scalar().to_f32();
    let actual_demand = normalized_demand * stats.max_cnt;

    println!("\nPrediction Results:");
    println!(
        "Input: hr={}, temp={}, hum={}, wind={}, work={}",
        hr, temp, hum, windspeed, workingday
    );
    println!("Predicted Bike Demand: {:.0} units", actual_demand);
}

fn main() {
    let cli = Cli::parse();
    type MyBackend = Autodiff<Wgpu>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    match cli.command {
        Commands::Train => {
            train::<MyBackend>(device);
        }
        Commands::Predict {
            hr,
            temp,
            hum,
            windspeed,
            workingday,
        } => {
            predict::<MyBackend>(device, hr, temp, hum, windspeed, workingday);
        }
    }
}
