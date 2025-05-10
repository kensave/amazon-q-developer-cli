use std::path::PathBuf;

use candle_transformers::models::bert::{
    Config,
    HiddenAct,
};

/// Configuration for a text embedding model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Name of the model
    pub name: String,
    /// Hugging Face repository path
    pub repo_path: String,
    /// Model file name in the repository
    pub model_file: String,
    /// Tokenizer file name in the repository
    pub tokenizer_file: String,
    /// BERT configuration
    pub config: Config,
    /// Whether to normalize embeddings
    pub normalize_embeddings: bool,
    /// Default batch size for processing
    pub batch_size: usize,
}

impl ModelConfig {
    /// Get the local path for model files
    pub fn get_local_paths(&self) -> (PathBuf, PathBuf) {
        let model_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("amazon-q-models");

        let model_path = model_dir.join(format!("{}.safetensors", self.name));
        let tokenizer_path = model_dir.join(format!("{}-tokenizer.json", self.name));

        (model_path, tokenizer_path)
    }
}

/// Available pre-configured models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// all-MiniLM-L6-v2 (default)
    MiniLML6V2,
    /// all-MiniLM-L12-v2
    MiniLML12V2,
}

impl Default for ModelType {
    fn default() -> Self {
        Self::MiniLML6V2
    }
}

impl ModelType {
    /// Get the configuration for this model type
    pub fn get_config(&self) -> ModelConfig {
        match self {
            Self::MiniLML6V2 => ModelConfig {
                name: "all-MiniLM-L6-v2".to_string(),
                repo_path: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                model_file: "model.safetensors".to_string(),
                tokenizer_file: "tokenizer.json".to_string(),
                config: Config {
                    vocab_size: 30522,
                    hidden_size: 384,
                    num_hidden_layers: 6,
                    num_attention_heads: 12,
                    intermediate_size: 1536,
                    hidden_act: HiddenAct::GeluApproximate,
                    hidden_dropout_prob: 0.0,
                    max_position_embeddings: 512,
                    type_vocab_size: 2,
                    initializer_range: 0.02,
                    layer_norm_eps: 1e-12,
                    pad_token_id: 0,
                    position_embedding_type: Default::default(),
                    use_cache: true,
                    classifier_dropout: None,
                    model_type: Some("bert".to_string()),
                },
                normalize_embeddings: true,
                batch_size: 32,
            },
            Self::MiniLML12V2 => ModelConfig {
                name: "all-MiniLM-L12-v2".to_string(),
                repo_path: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
                model_file: "model.safetensors".to_string(),
                tokenizer_file: "tokenizer.json".to_string(),
                config: Config {
                    vocab_size: 30522,
                    hidden_size: 384,
                    num_hidden_layers: 12, // More layers than L6
                    num_attention_heads: 12,
                    intermediate_size: 1536,
                    hidden_act: HiddenAct::GeluApproximate,
                    hidden_dropout_prob: 0.0,
                    max_position_embeddings: 512,
                    type_vocab_size: 2,
                    initializer_range: 0.02,
                    layer_norm_eps: 1e-12,
                    pad_token_id: 0,
                    position_embedding_type: Default::default(),
                    use_cache: true,
                    classifier_dropout: None,
                    model_type: Some("bert".to_string()),
                },
                normalize_embeddings: true,
                batch_size: 32,
            },
        }
    }
}
