use burn::{
    data::{dataloader::batcher::Batcher, dataset::InMemDataset},
    tensor::{Tensor, backend::Backend},
};
use serde::{Deserialize, Serialize};

// The raw format from CSV - matches the full Bike Sharing Dataset schema
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BikeRecord {
    pub instant: u32,    // Record index
    pub dteday: String,  // Date
    pub season: u8,      // Season (1-4)
    pub yr: u8,          // Year (0=2011, 1=2012)
    pub mnth: u8,        // Month (1-12)
    pub hr: f32,         // Hour (0-23) - FEATURE
    pub holiday: u8,     // Holiday flag
    pub weekday: u8,     // Day of week (0-6)
    pub workingday: f32, // Working day flag - FEATURE
    pub weathersit: u8,  // Weather situation (1-4)
    pub temp: f32,       // Normalized temp - FEATURE
    pub atemp: f32,      // Normalized feeling temp
    pub hum: f32,        // Normalized humidity - FEATURE
    pub windspeed: f32,  // Normalized windspeed - FEATURE
    pub casual: u32,     // Casual users count
    pub registered: u32, // Registered users count
    pub cnt: f32,        // Total count - TARGET
}

// The processed item
#[derive(Clone, Debug)]
pub struct BikeItem {
    pub features: [f32; 5],
    pub target: [f32; 1],
}

// Normalization params - serializable for checkpointing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Normalizer {
    pub mean: [f32; 5],
    pub std: [f32; 5],
    pub target_max: f32, // to scale target 0-1
}

impl Normalizer {
    pub fn normalize(&self, features: &[f32; 5]) -> [f32; 5] {
        let mut n = [0.0; 5];
        for i in 0..5 {
            // If std is 0 (no variance), just center around mean without scaling
            // This prevents division by near-zero for constant features
            let divisor = if self.std[i] > 1e-6 { self.std[i] } else { 1.0 };
            n[i] = (features[i] - self.mean[i]) / divisor;
        }
        n
    }

    /// Denormalize a predicted count (0-1 range) back to original scale
    pub fn denormalize_target(&self, normalized: f32) -> f32 {
        normalized * self.target_max
    }

    /// Save normalizer to JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load normalizer from JSON file
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

pub fn load_data(path: &str) -> (InMemDataset<BikeItem>, Normalizer) {
    let mut reader =
        csv::Reader::from_path(path).expect("Could not open CSV file. Make sure hour.csv exists.");

    let mut records: Vec<BikeRecord> = Vec::new();
    for result in reader.deserialize() {
        let record: BikeRecord = result.expect("Error parsing CSV record");
        records.push(record);
    }

    // Calculate stats
    let count = records.len() as f32;
    let mut sum = [0.0; 5];
    let mut sq_sum = [0.0; 5];
    let mut target_max = 0.0_f32;

    for r in &records {
        let f = [r.hr, r.temp, r.hum, r.windspeed, r.workingday];
        for i in 0..5 {
            sum[i] += f[i];
            sq_sum[i] += f[i] * f[i];
        }
        if r.cnt > target_max {
            target_max = r.cnt
        };
    }

    let mut mean = [0.0; 5];
    let mut std = [0.0; 5];
    for i in 0..5 {
        mean[i] = sum[i] / count;
        std[i] = (sq_sum[i] / count - mean[i] * mean[i]).sqrt();
    }

    let normalizer = Normalizer {
        mean,
        std,
        target_max,
    };
    println!("Normalizer: {:?}", normalizer);

    // Create Items
    let items: Vec<BikeItem> = records
        .iter()
        .map(|r| {
            let f = [r.hr, r.temp, r.hum, r.windspeed, r.workingday];
            BikeItem {
                features: normalizer.normalize(&f),
                target: [r.cnt / target_max], // Scale target to 0-1 range for stability
            }
        })
        .collect();

    (InMemDataset::new(items), normalizer)
}

// Batcher for DataLoader
#[derive(Clone)]
pub struct BikeBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> BikeBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct BikeBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Batcher<BikeItem, BikeBatch<B>> for BikeBatcher<B> {
    fn batch(&self, items: Vec<BikeItem>) -> BikeBatch<B> {
        let inputs: Vec<[f32; 5]> = items.iter().map(|item| item.features).collect();
        // Flattening logic handles the [f32; 5] into the tensor

        let targets: Vec<[f32; 1]> = items.iter().map(|item| item.target).collect();

        // Burn's Tensor::from_floats expects a flattened list of floats if we use from_data or similar,
        // but let's use a simpler way: construct from 2D data

        // Actually for from_floats (1D slice), we need to ensure layout is correct.
        // Let's use `from_data` with shape

        let batch_size = items.len();

        let flat_inputs: Vec<f32> = inputs.iter().flatten().cloned().collect();
        let flat_targets: Vec<f32> = targets.iter().flatten().cloned().collect();

        let inputs = Tensor::<B, 1>::from_floats(flat_inputs.as_slice(), &self.device)
            .reshape([batch_size, 5]);

        let targets = Tensor::<B, 1>::from_floats(flat_targets.as_slice(), &self.device)
            .reshape([batch_size, 1]);

        BikeBatch { inputs, targets }
    }
}
