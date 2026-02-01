mod cli;
mod data;
mod model;
mod predict;
mod train;

use burn::backend::{Autodiff, Wgpu};
use clap::Parser;
use cli::{Cli, Commands};

fn main() {
    let args = Cli::parse();

    // Use WGPU backend with Autodiff
    // We can fallback to NdArray if WGPU is not available, but for this lesson strict GPU is requested.
    // However, to make it runnable everywhere easily, we default to Wgpu,
    // but the burn-wgpu crate handles device selection.

    type MyBackend = Autodiff<Wgpu>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    match args.command {
        Commands::Train {
            dataset,
            epochs,
            batch_size,
            lr,
        } => {
            train::run_training::<MyBackend>(device, &dataset, epochs, batch_size, lr);
        }
        Commands::Predict {
            hr,
            temp,
            hum,
            windspeed,
            workingday,
        } => {
            predict::run_inference::<MyBackend>(device, [hr, temp, hum, windspeed, workingday]);
        }
    }
}
