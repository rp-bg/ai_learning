use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, backend::Backend},
};

// --- Linear Model ---
#[derive(Module, Debug)]
pub struct LinearModel<B: Backend> {
    layer: Linear<B>,
}

impl<B: Backend> LinearModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // 5 Inputs -> 1 Output (Count)
            layer: LinearConfig::new(5, 1).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.layer.forward(input)
    }
}

// --- Neural Model (ReLU) ---
#[derive(Module, Debug)]
pub struct NeuralModel<B: Backend> {
    fc1: Linear<B>,
    relu1: Relu,
    fc2: Linear<B>,
    relu2: Relu,
    output: Linear<B>,
}

impl<B: Backend> NeuralModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            // Input (5) -> Hidden (16)
            fc1: LinearConfig::new(5, 16).init(device),
            relu1: Relu::new(),
            // Hidden (16) -> Hidden (8)
            fc2: LinearConfig::new(16, 8).init(device),
            relu2: Relu::new(),
            // Hidden (8) -> Output (1)
            output: LinearConfig::new(8, 1).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.relu1.forward(x);
        let x = self.fc2.forward(x);
        let x = self.relu2.forward(x);
        self.output.forward(x)
    }
}
