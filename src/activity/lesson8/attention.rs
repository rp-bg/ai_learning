use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Bool, Tensor, activation::softmax, backend::Backend},
};

#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    n_heads: usize,
    d_model: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        CausalSelfAttention {
            query: LinearConfig::new(self.d_model, self.d_model).init(device),
            key: LinearConfig::new(self.d_model, self.d_model).init(device),
            value: LinearConfig::new(self.d_model, self.d_model).init(device),
            output: LinearConfig::new(self.d_model, self.d_model).init(device),
            n_heads: self.n_heads,
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = x.dims();
        let d_head = d_model / self.n_heads;

        // 1. Projections: [Batch, Seq, d_model]
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        // 2. Split heads: [Batch, n_heads, Seq, d_head]
        let q = self.split_heads(q, batch_size, seq_len, d_head);
        let k = self.split_heads(k, batch_size, seq_len, d_head);
        let v = self.split_heads(v, batch_size, seq_len, d_head);

        // 3. Scaled dot-product: (Q @ K^T) / sqrt(d_head)
        // [Batch, n_heads, Seq, Seq]
        let mut scores = q.matmul(k.transpose()) / (d_head as f64).sqrt();

        // 4. Causal Mask: prevent attending to future tokens
        let mask = self.create_causal_mask(seq_len, &scores.device());
        scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        // 5. Softmax & Attention application
        let attn = softmax(scores, 3);
        let out = attn.matmul(v); // [Batch, n_heads, Seq, d_head]

        // 6. Merge heads & final projection
        let out = self.merge_heads(out, batch_size, seq_len);
        self.output.forward(out)
    }

    fn split_heads(
        &self,
        x: Tensor<B, 3>,
        batch_size: usize,
        seq_len: usize,
        d_head: usize,
    ) -> Tensor<B, 4> {
        x.reshape([batch_size, seq_len, self.n_heads, d_head])
            .swap_dims(1, 2)
    }

    fn merge_heads(&self, x: Tensor<B, 4>, batch_size: usize, seq_len: usize) -> Tensor<B, 3> {
        x.swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model])
    }

    fn create_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4, Bool> {
        // Create a 2D upper triangular mask: i > j
        // In Burn, we can use tril (lower triangular) or create it manually.
        // We want a mask where mask[i, j] is true if j > i (future tokens).

        let mask = Tensor::<B, 2>::ones([seq_len, seq_len], device)
            .tril(0) // Keep lower triangular (including diagonal)
            .equal_elem(0.0); // True where it was zero (upper triangular without diagonal)

        // Reshape to [1, 1, Seq, Seq] for broadcasting
        mask.reshape([1, 1, seq_len, seq_len])
    }
}
