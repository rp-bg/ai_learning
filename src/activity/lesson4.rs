/**
 * memory_usage ≈ f(session_duration, api_calls)
 * learn this relationship but not as a hard coded forumula.
 * Instead:
 * the model learns weights
 * the model learns bias
 * Burn handles gradients
 * Add ReLU activation function
 */
use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
struct MemoryModel<B: Backend> {
    fc1: Linear<B>, // first layer
    relu: Relu,     // activation
    fc2: Linear<B>, // output layer
}

impl<B: Backend> MemoryModel<B> {
    fn new(device: &B::Device) -> Self {
        let fc1 = LinearConfig::new(2, 8).init(device); // 2 → 8
        let fc2 = LinearConfig::new(8, 1).init(device); // 8 → 1

        Self {
            fc1,
            relu: Relu::new(),
            fc2,
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        self.fc2.forward(x)
    }
}

fn main() {
    // Prepare training data
    type B = Autodiff<Wgpu>;
    let device = <B as Backend>::Device::default();

    // Input data
    let x = Tensor::<B, 2>::from_data(
        [
            [100.0, 10.0],
            [200.0, 20.0],
            [300.0, 30.0],
            [400.0, 40.0],
            [500.0, 50.0],
        ],
        &device,
    );

    // Target data
    let y = Tensor::<B, 2>::from_data([[150.0], [240.0], [355.0], [465.0], [585.0]], &device);

    let mut model = MemoryModel::<B>::new(&device);
    let mut optimizer = AdamConfig::new().init();

    // Training loop
    for epoch in 0..1000 {
        let predictions = model.forward(x.clone());

        let loss = (predictions - y.clone()).powf_scalar(2.0).mean();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        let lr = 1e-2;
        model = optimizer.step(lr, model, grads_params);

        if epoch % 100 == 0 {
            println!("Epoch {epoch}, Loss: {}", loss);
        }
    }

    let test_input = Tensor::<B, 2>::from_data([[150.0, 15.0]], &device);

    let prediction = model.forward(test_input);
    println!("Predicted memory usage: {}", prediction);
}
