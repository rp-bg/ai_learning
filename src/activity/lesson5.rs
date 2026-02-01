use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Tensor, backend::Backend},
};

/// A simple Neural Network module on CPU
/// Structure: Linear (Input -> Hidden) -> ReLU -> Linear (Hidden -> Output)
#[derive(Module, Debug)]
pub struct SimpleNN<B: Backend> {
    linear1: Linear<B>,
    activation: Relu,
    linear2: Linear<B>,
}

impl<B: Backend> SimpleNN<B> {
    /// Create a new model initialized on the given device
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Input layer: 2 input features -> 8 hidden units
            linear1: LinearConfig::new(2, 8).init(device),
            // Activation function: ReLU (Rectified Linear Unit)
            // Used to introduce non-linearity, allowing the model to learn complex patterns
            activation: Relu::new(),
            // Output layer: 8 hidden units -> 1 output value
            linear2: LinearConfig::new(8, 1).init(device),
        }
    }

    /// Forward pass through the network
    /// x -> Linear1 -> ReLU -> Linear2 -> output
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}

fn main() {
    // 1. Backend Configuration
    // We use NdArray (CPU) backend wrapped with Autodiff to enable automatic differentiation.
    // To use GPU, you would swap `NdArray` with `Wgpu`.
    type B = Autodiff<NdArray>;
    let device = Default::default(); // Uses CPU by default for NdArray

    // 2. Data Preparation (XOR-like problem)
    // Inputs: [0,0], [0,1], [1,0], [1,1]
    let x = Tensor::<B, 2>::from_data([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], &device);
    // Targets: [0], [1], [1], [0]
    let y = Tensor::<B, 2>::from_data([[0.0], [1.0], [1.0], [0.0]], &device);

    // 3. Model Initialization
    let mut model = SimpleNN::<B>::new(&device);

    // 4. Optimizer Setup
    // Adam is a popular optimization algorithm that adapts learning rates for each parameter.
    let mut optimizer = AdamConfig::new().init();

    // 5. Training Loop
    println!("Starting training on CPU...");
    for i in 0..2000 {
        // Forward pass: Compute predictions
        // Autograd Note: Operations on Tensors are recorded in a computation graph
        let output = model.forward(x.clone());

        // Compute Loss (Mean Squared Error)
        let loss = (output - y.clone()).powf_scalar(2.0).mean();

        // Print progress
        if i % 200 == 0 {
            println!("Epoch {}: Loss = {}", i, loss);
        }

        // Backward pass: Compute gradients
        // `loss.backward()` traverses the graph backwards to calculate gradients for all parameters.
        let grads = loss.backward();

        // Extract gradients for the specific model parameters
        let grads_params = GradientsParams::from_grads(grads, &model);

        // Update parameters using the optimizer
        // Returns the updated model
        model = optimizer.step(0.01, model, grads_params);
    }

    // 6. Verification
    // Test input
    let test_input =
        Tensor::<B, 2>::from_data([[0.0, 0.0], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]], &device);
    let final_output = model.forward(test_input);
    println!("Final predictions:\n{}", final_output);
    // Explanation for expected output
    // The model should output values close to 0 for [0,0] and [1,1], and values close to 1 for [0,1] and [1,0].
    // This is because the model has learned to approximate the XOR function.
    // The output is close to 0 for [0,0] and [1,1], and close to 1 for [0,1] and [1,0].
}
