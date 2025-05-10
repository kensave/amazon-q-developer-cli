use fastembed::EmbeddingModel as FastEmbeddingModel;

/// Available pre-configured models for ONNX embedder
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxModelType {
    /// all-MiniLM-L6-v2-Q (default) - Quantized version of all-MiniLM-L6-v2
    MiniLML6V2Q,
    /// all-MiniLM-L12-v2-Q - Quantized version of all-MiniLM-L12-v2
    MiniLML12V2Q,
}

impl Default for OnnxModelType {
    fn default() -> Self {
        Self::MiniLML6V2Q
    }
}

impl OnnxModelType {
    /// Get the fastembed model type for this model
    pub fn get_fastembed_model(&self) -> FastEmbeddingModel {
        match self {
            Self::MiniLML6V2Q => FastEmbeddingModel::AllMiniLML6V2Q,
            Self::MiniLML12V2Q => FastEmbeddingModel::AllMiniLML12V2Q,
        }
    }

    /// Get the embedding dimension for this model
    pub fn get_embedding_dim(&self) -> usize {
        match self {
            Self::MiniLML6V2Q => 384,
            Self::MiniLML12V2Q => 384,
        }
    }

    /// Get the model name as a string
    pub fn get_name(&self) -> &'static str {
        match self {
            Self::MiniLML6V2Q => "all-MiniLM-L6-v2-Q",
            Self::MiniLML12V2Q => "all-MiniLM-L12-v2-Q",
        }
    }
}
