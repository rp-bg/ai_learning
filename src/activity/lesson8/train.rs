use crate::data::{TextDataset, Tokenizer};
use crate::model::{TinyGPT, TinyGPTConfig};
use burn::data::dataset::Dataset;
use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    optim::Optimizer,
    tensor::{Int, Tensor, backend::AutodiffBackend},
};

pub fn train<B: AutodiffBackend>(
    device: B::Device,
    tokenizer: &Tokenizer,
    train_path: &str,
) -> TinyGPT<B> {
    // 1. Hyperparameters
    let block_size = 64;
    let batch_size = 32;
    let n_layers = 4;
    let d_model = 128;
    let n_heads = 4;
    let dropout = 0.1;
    let learning_rate = 1e-3;
    let n_iterations = 1000;

    // 2. Prepare Data
    let train_ds = TextDataset::new(train_path, tokenizer, block_size);

    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Train items: {}", train_ds.len());

    // 3. Initialize Model & Optimizer
    let config = TinyGPTConfig::new(
        tokenizer.vocab_size(),
        block_size,
        n_layers,
        d_model,
        n_heads,
        dropout,
    );
    let mut model = config.init::<B>(&device);
    let mut optim = AdamConfig::new().init::<B, TinyGPT<B>>();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    // 4. Training Loop
    println!("Training for {} iterations...", n_iterations);
    for i in 0..n_iterations {
        // Simple random batch sampling
        let (input, targets) = sample_batch::<B>(&train_ds, batch_size, &device);

        // Forward
        let logits = model.forward(input);

        // Loss calculation (flattening for CrossEntropy)
        let [b, t, v] = logits.dims();
        let logits_flat = logits.reshape([b * t, v]);
        let targets_flat = targets.reshape([b * t]);

        let loss = loss_fn.forward(logits_flat, targets_flat);

        if i % 100 == 0 {
            println!("Iter {}: Loss = {}", i, loss.clone().into_scalar());
        }

        // Backward & Step
        let grads = loss.backward();
        let grads = burn::optim::GradientsParams::from_grads(grads, &model);
        model = optim.step(learning_rate, model, grads);
    }

    model
}

fn sample_batch<B: AutodiffBackend>(
    dataset: &TextDataset,
    batch_size: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut inputs = Vec::with_capacity(batch_size);
    let mut targets = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let idx = rng.gen_range(0..dataset.len());
        let item = dataset.get(idx).unwrap();
        inputs.push(Tensor::<B, 1, Int>::from_ints(
            item.input.as_slice(),
            device,
        ));
        targets.push(Tensor::<B, 1, Int>::from_ints(
            item.target.as_slice(),
            device,
        ));
    }

    let input = Tensor::stack(inputs, 0);
    let target = Tensor::stack(targets, 0);

    (input, target)
}
