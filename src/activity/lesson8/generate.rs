use crate::data::Tokenizer;
use crate::model::TinyGPT;
use burn::tensor::activation::softmax;
use burn::tensor::{Int, Tensor, backend::Backend};

pub fn generate<B: Backend>(
    model: &TinyGPT<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    device: &B::Device,
) -> String {
    let mut current_ids = tokenizer.encode(prompt);

    // We'll perform generation in an autoregressive loop
    for _ in 0..max_new_tokens {
        // 1. Ensure context fits within block_size
        let start = current_ids.len().saturating_sub(model.block_size);
        let input_slice = &current_ids[start..];

        // 2. Prepare input tensor [1, T]
        let input_tensor =
            Tensor::<B, 1, Int>::from_ints(input_slice, device).reshape([1, input_slice.len()]);

        // 3. Get logits [1, T, Vocab]
        let logits = model.forward(input_tensor);
        let [_, t, v] = logits.dims();

        // 4. Extract last token logits [1, Vocab]
        let last_logit = logits.slice([0..1, t - 1..t, 0..v]).reshape([v]);

        // 5. Apply temperature scaling
        let scaled_logits = last_logit / (temperature as f64);
        let probs = softmax(scaled_logits, 0);

        // 6. Sample next token (Greedy for simplicity in this version)
        // Note: For true diversity, one would use multinomial sampling.
        use burn::tensor::cast::ToElement;
        let next_token = probs.argmax(0).into_scalar().to_i64() as usize;

        current_ids.push(next_token);

        // If we generate a newline (optionally stop or just continue)
        // if next_token == tokenizer.encode("\n")[0] { break; }
    }

    tokenizer.decode(&current_ids)
}
