use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Tensor;

/** Linear Regression - memory_usage ≈ w1 * session_duration
             + w2 * api_calls
             + b
 * train a model that predicts memory usage based on two inputs:
 * session_duration and api_calls
 *
*/

fn main() {
    println!("Lesson 2 Activity - Linear Regression");

    // Use Autodiff backend to enable gradient calculation
    type B = Autodiff<Wgpu>;
    let device = WgpuDevice::default();

    // Data - Training Set
    let session_duration_in_ms_set = [100.0, 200.0, 300.0, 400.0, 500.0];
    let api_calls_per_ms_set = [10.0, 20.0, 30.0, 40.0, 50.0];
    let memory_usage_in_bytes_set = [150.0, 240.0, 355.0, 465.0, 585.0];

    let session_duration_in_ms: Tensor<B, 1> =
        Tensor::from_data(session_duration_in_ms_set, &device);
    let api_calls_per_ms: Tensor<B, 1> = Tensor::from_data(api_calls_per_ms_set, &device);
    let memory_usage_in_bytes: Tensor<B, 1> = Tensor::from_data(memory_usage_in_bytes_set, &device);

    println!("Session Duration: {}", session_duration_in_ms);
    println!("API Calls: {}", api_calls_per_ms);
    println!("Memory Usage: {}", memory_usage_in_bytes);

    // Initial weights initialized typically to small random values or zeros,
    // but here we start with 0.0 or small values.
    // We expect w1 -> 1.0, w2 -> 0.0, b -> 0.0
    let mut w1 = Tensor::<B, 1>::from_data([0.1], &device).require_grad();
    let mut w2 = Tensor::<B, 1>::from_data([0.1], &device).require_grad();
    let mut b = Tensor::<B, 1>::from_data([0.0], &device).require_grad();

    let lr = 1e-6 as f32;
    let lr_tensor = Tensor::<B, 1>::from_data([lr], &device);
    let epochs = 1000;

    println!("Starting training...");

    for i in 0..epochs {
        let y_pred = w1.clone() * session_duration_in_ms.clone()
            + w2.clone() * api_calls_per_ms.clone()
            + b.clone();

        let loss = (y_pred - memory_usage_in_bytes.clone())
            .powf_scalar(2.0)
            .mean();

        if i % 100 == 0 {
            println!("Epoch {}: Loss {}", i, loss);
        }

        let grads = loss.backward();

        let w1_grad = w1.grad(&grads).unwrap();
        let w2_grad = w2.grad(&grads).unwrap();
        let b_grad = b.grad(&grads).unwrap();

        let w1_grad = Tensor::<B, 1>::from_data(w1_grad.into_data(), &device);
        let w2_grad = Tensor::<B, 1>::from_data(w2_grad.into_data(), &device);
        let b_grad = Tensor::<B, 1>::from_data(b_grad.into_data(), &device);

        w1 = (w1 - w1_grad * lr_tensor.clone()).detach().require_grad();
        w2 = (w2 - w2_grad * lr_tensor.clone()).detach().require_grad();
        b = (b - b_grad * lr_tensor.clone()).detach().require_grad();
    }

    println!("Training Complete");
    println!("w1: {}", w1);
    println!("w2: {}", w2);
    println!("b: {}", b);

    // Test
    // - session duration 150ms
    // - api calls 15
    // - expected memory usage 150 * 1 + 15 * 0 = 150 roughly
    // The previous test expectation (165) assumed w1=1, w2=1.
    // True relationship in data is Memory = Session. So w1=1, w2=0.
    // So for 150ms, memory should be 150.
    let test_pred = w1.clone() * Tensor::from_data([150.0], &device)
        + w2.clone() * Tensor::from_data([15.0], &device)
        + b.clone();
    println!(
        "Predicted Memory Usage for test (150ms, 15calls): {}",
        test_pred
    );
}
