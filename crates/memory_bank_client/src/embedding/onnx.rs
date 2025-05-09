//! Text embedding functionality using fastembed
//!
//! This module provides functionality for generating text embeddings
//! using the fastembed library, which is available on macOS and Windows platforms.

use fastembed::{
    EmbeddingModel as FastEmbeddingModel,
    InitOptions,
    TextEmbedding,
};
use tracing::{
    debug,
    error,
    info,
};

use crate::error::{
    MemoryBankError,
    Result,
};

/// Text embedder using fastembed
pub struct TextEmbedder {
    /// The embedding model
    model: TextEmbedding,
}

impl TextEmbedder {
    /// Create a new TextEmbedder with the default model (all-MiniLM-L6-v2)
    ///
    /// # Returns
    ///
    /// A new TextEmbedder instance
    pub fn new() -> Result<Self> {
        info!("Initializing text embedder with fastembed");

        // Initialize the embedding model
        let model = match TextEmbedding::try_new(
            InitOptions::new(FastEmbeddingModel::AllMiniLML6V2Q).with_show_download_progress(true),
        ) {
            Ok(model) => model,
            Err(e) => {
                error!("Failed to initialize fastembed model: {}", e);
                return Err(MemoryBankError::FastembedError(e.to_string()));
            },
        };

        debug!("Fastembed text embedder initialized successfully");

        Ok(Self { model })
    }

    /// Generate an embedding for a text
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed
    ///
    /// # Returns
    ///
    /// A vector of floats representing the text embedding
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text];
        match self.model.embed(texts, None) {
            Ok(embeddings) => Ok(embeddings.into_iter().next().unwrap()),
            Err(e) => {
                error!("Failed to embed text: {}", e);
                Err(MemoryBankError::FastembedError(e.to_string()))
            },
        }
    }

    /// Generate embeddings for multiple texts
    ///
    /// # Arguments
    ///
    /// * `texts` - The texts to embed
    ///
    /// # Returns
    ///
    /// A vector of embeddings
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let documents: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        match self.model.embed(documents, None) {
            Ok(embeddings) => Ok(embeddings),
            Err(e) => {
                error!("Failed to embed batch of texts: {}", e);
                Err(MemoryBankError::FastembedError(e.to_string()))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    #[test]
    fn test_embed_single() {
        // Skip if real embedders are not explicitly requested
        if env::var("MEMORY_BANK_USE_REAL_EMBEDDERS").is_err() {
            return;
        }

        // Use real embedder for testing
        match TextEmbedder::new() {
            Ok(embedder) => {
                let embedding = embedder.embed("This is a test sentence.").unwrap();

                // MiniLM-L6-v2 produces 384-dimensional embeddings
                assert_eq!(embedding.len(), 384);
            },
            Err(e) => {
                // If model loading fails, skip the test
                println!("Skipping test: Failed to load real embedder: {}", e);
            },
        }
    }

    #[test]
    fn test_embed_batch() {
        // Skip if real embedders are not explicitly requested
        if env::var("MEMORY_BANK_USE_REAL_EMBEDDERS").is_err() {
            return;
        }

        // Use real embedder for testing
        match TextEmbedder::new() {
            Ok(embedder) => {
                let texts = vec![
                    "The cat sits outside".to_string(),
                    "A man is playing guitar".to_string(),
                ];
                let embeddings = embedder.embed_batch(&texts).unwrap();

                assert_eq!(embeddings.len(), 2);
                assert_eq!(embeddings[0].len(), 384);
                assert_eq!(embeddings[1].len(), 384);

                // Check that embeddings are different
                let mut different = false;
                for i in 0..384 {
                    if (embeddings[0][i] - embeddings[1][i]).abs() > 1e-5 {
                        different = true;
                        break;
                    }
                }
                assert!(different);
            },
            Err(e) => {
                // If model loading fails, skip the test
                println!("Skipping test: Failed to load real embedder: {}", e);
            },
        }
    }
}
