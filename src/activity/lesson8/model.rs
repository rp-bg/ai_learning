use crate::attention::{CausalSelfAttention, CausalSelfAttentionConfig};
use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu},
    tensor::{Tensor, backend::Backend},
};

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    norm2: LayerNorm<B>,
    mlp_fc1: Linear<B>,
    mlp_fc2: Linear<B>,
    relu: Relu,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            norm1: LayerNormConfig::new(self.d_model).init(device),
            attn: CausalSelfAttentionConfig::new(self.d_model, self.n_heads, self.dropout)
                .init(device),
            norm2: LayerNormConfig::new(self.d_model).init(device),
            mlp_fc1: LinearConfig::new(self.d_model, 4 * self.d_model).init(device),
            mlp_fc2: LinearConfig::new(4 * self.d_model, self.d_model).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // 1. Attention with Residual (Pre-norm)
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.attn.forward(x);
        let x = x + residual;

        // 2. MLP with Residual (Pre-norm)
        let residual = x.clone();
        let x = self.norm2.forward(x);
        let x = self.mlp_fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.mlp_fc2.forward(x);
        let x = x + residual;

        x
    }
}

#[derive(Config, Debug)]
pub struct TinyGPTConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct TinyGPT<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    // We'll use a fixed number of layers for simplicity in this learning example,
    // or handle the Vec manually. In Burn 0.14, we can use a small fix for Vec.
    blocks: Vec<TransformerBlock<B>>,
    norm_f: LayerNorm<B>,
    lm_head: Linear<B>,
    pub block_size: usize,
}

impl TinyGPTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TinyGPT<B> {
        let block_config = TransformerBlockConfig::new(self.d_model, self.n_heads, self.dropout);

        let mut blocks = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            blocks.push(block_config.init(device));
        }

        TinyGPT {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.block_size, self.d_model).init(device),
            blocks,
            norm_f: LayerNormConfig::new(self.d_model).init(device),
            lm_head: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            block_size: self.block_size,
        }
    }
}

impl<B: Backend> TinyGPT<B> {
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input.dims();
        let device = &input.device();

        // 1. Token & Position Embeddings
        let tok_emb = self.token_embedding.forward(input); // [B, T, D]

        // Generate positions: [0, 1, ..., T-1]
        let pos = Tensor::<B, 1, burn::tensor::Int>::arange(0..seq_len as i64, device)
            .reshape([1, seq_len]);
        let pos_emb = self.position_embedding.forward(pos); // [1, T, D]

        let mut x = tok_emb + pos_emb;

        // 2. Transformer Blocks
        for block in self.blocks.iter() {
            x = block.forward(x);
        }

        // 3. Final Norm and Head
        let x = self.norm_f.forward(x);
        self.lm_head.forward(x) // [B, T, Vocab]
    }
}
