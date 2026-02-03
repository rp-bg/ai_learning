use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about = "Bike Sharing Prediction Tool", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train the models (Linear vs ReLU)
    Train {
        /// Path to the dataset (csv file)
        #[arg(short, long, default_value = "data/lesson7/hour.csv")]
        dataset: String,

        /// Number of epochs
        #[arg(short, long, default_value_t = 20)]
        epochs: usize,

        /// Batch size
        #[arg(short, long, default_value_t = 64)]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value_t = 1e-2)]
        lr: f64,
    },
    /// Make a prediction using the trained logic (dummy weights for now as we don't save model yet)
    Predict {
        #[arg(long)]
        hr: f32,
        #[arg(long)]
        temp: f32,
        #[arg(long)]
        hum: f32,
        #[arg(long)]
        windspeed: f32,
        #[arg(long)]
        workingday: f32,
    },
}
