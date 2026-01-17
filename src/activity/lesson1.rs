use ndarray::prelude::*;

/** Lesson 1 Activity - Linear Regression and Tensors */
fn main() {
    println!("Lesson 1 Activity - Linear Regression and Tensors");
    // Create a 2x3 matrix (2 rows, 3 columns)
    let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    println!("Matrix: {:#?}", matrix);

    // 2. Linear Regression (y = 2x + 1)
    let x = array![1.0f64, 2.0, 3.0];
    let y = array![3.0f64, 5.0, 7.0];
    println!("x: {:#?}", x);
    println!("y: {:#?}", y);

    // Initialize parameters (weights and bias) randomly
    let mut w = array![2.0];
    let mut b = array![1.0];
    println!("w: {:#?}", w);
    println!("b: {:#?}", b);

    // Forward pass
    let y_pred = &x * &w + &b;
    println!("y_pred: {:#?}", y_pred);

    // Calculate loss
    let loss = (&y_pred - &y).mapv(|a: f64| a.powi(2)).mean().unwrap();
    println!("loss: {:#?}", loss);

    // Backward pass
    // d_loss/d_w = 2 * (y_pred - y) * x
    // We average gradients over the batch to keep parameters same shape
    let diff = &y_pred - &y;
    let dw = (&diff * &x).mean().unwrap() * 2.0;
    let db = diff.mean().unwrap() * 2.0;

    println!("dw: {:#?}", dw);
    println!("db: {:#?}", db);

    // Update parameters
    w = &w - dw * 0.1; // Using small learning rate
    b = &b - db * 0.1;
    println!("w: {:#?}", w);
    println!("b: {:#?}", b);

    // Repeat for multiple epochs
    for epoch in 0..10 {
        let y_pred = &x * &w + &b;
        let loss = (&y_pred - &y).mapv(|a: f64| a.powi(2)).mean().unwrap();
        println!("Epoch {epoch}, Loss: {loss}");

        // Update loop
        let diff = &y_pred - &y;
        let dw = (&diff * &x).mean().unwrap() * 2.0;
        let db = diff.mean().unwrap() * 2.0;
        w = &w - dw * 0.01;
        b = &b - db * 0.01;
    }

    // Test
    let y_pred = &x * &w + &b;
    println!("y_pred: {:#?}", y_pred);
}
