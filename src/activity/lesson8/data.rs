use burn::data::dataset::Dataset;
use std::collections::{HashMap, HashSet};
use std::fs;

pub struct Tokenizer {
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
    vocab_size: usize,
}

impl Tokenizer {
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<HashSet<_>>().into_iter().collect();
        chars.sort();

        let char_to_id: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let id_to_char: HashMap<usize, char> =
            chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();
        let vocab_size = chars.len();

        Self {
            char_to_id,
            id_to_char,
            vocab_size,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| *self.id_to_char.get(&id).unwrap_or(&'?'))
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

pub struct TextDataset {
    data: Vec<usize>,
    block_size: usize,
}

impl TextDataset {
    pub fn new(path: &str, tokenizer: &Tokenizer, block_size: usize) -> Self {
        let content = fs::read_to_string(path).expect("Could not read file");
        let data = tokenizer.encode(&content);
        Self { data, block_size }
    }
}

#[derive(Debug, Clone)]
pub struct TextItem {
    pub input: Vec<usize>,
    pub target: Vec<usize>,
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        if index + self.block_size >= self.data.len() {
            return None;
        }

        let input = self.data[index..index + self.block_size].to_vec();
        let target = self.data[index + 1..index + self.block_size + 1].to_vec();

        Some(TextItem { input, target })
    }

    fn len(&self) -> usize {
        if self.data.len() <= self.block_size {
            0
        } else {
            self.data.len() - self.block_size
        }
    }
}
