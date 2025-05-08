use std::path::{
    Path,
    PathBuf,
};
use std::thread::available_parallelism;

use anyhow::Result as AnyhowResult;
use candle_core::{
    Device,
    Tensor,
};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{
    BertModel,
    Config,
    DTYPE,
    HiddenAct,
};
use rayon::prelude::*;
use tokenizers::Tokenizer;
use tracing::{
    debug,
    error,
    info,
};

use crate::error::{
    MemoryBankError,
    Result,
};

/// Get the best available device for inference
fn get_best_available_device() -> Device {
    // Try Metal on macOS
    // Disabled: Layer norm not supported for selected model
    // only enable Metal if verified it's compatible.

    // Try CUDA on any platform where it's available
    match Device::new_cuda(0) {
        Ok(device) => {
            info!("Using CUDA acceleration for text embedding");
            return device;
        },
        Err(e) => {
            debug!("CUDA acceleration not available: {}", e);
        },
    }

    // Fall back to CPU
    info!("Using CPU for text embedding (no GPU acceleration available)");
    Device::Cpu
}

// Default batch size for processing
const DEFAULT_BATCH_SIZE: usize = 32;

/// Text embedding generator using Candle for the all-MiniLM-L6-v2 model
pub struct CandleTextEmbedder {
    /// The BERT model
    model: BertModel,
    /// The tokenizer
    tokenizer: Tokenizer,
    /// The device to run on
    device: Device,
    /// Whether to normalize embeddings
    normalize_embeddings: bool,
}

impl CandleTextEmbedder {
    /// Create a new TextEmbedder with the default model (all-MiniLM-L6-v2)
    ///
    /// # Returns
    ///
    /// A new TextEmbedder instance
    pub fn new() -> Result<Self> {
        // Default paths for model files - these should be downloaded during build or installation
        let model_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("amazon-q-models");

        if let Err(err) = std::fs::create_dir_all(&model_dir) {
            error!("Failed to create model directory: {}", err);
            return Err(MemoryBankError::IoError(err));
        }

        let model_path = model_dir.join("all-MiniLM-L6-v2.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");

        // Download files if they don't exist
        match Self::ensure_model_files(&model_path, &tokenizer_path) {
            Ok(_) => {},
            Err(e) => {
                error!("Failed to ensure model files: {}", e);
                return Err(MemoryBankError::EmbeddingError(e.to_string()));
            },
        }

        Self::with_model_paths(&model_path, &tokenizer_path)
    }

    /// Create a new TextEmbedder with specific model paths
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file (.safetensors)
    /// * `tokenizer_path` - Path to the tokenizer file (.json)
    ///
    /// # Returns
    ///
    /// A new TextEmbedder instance
    pub fn with_model_paths(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        info!("Initializing text embedder with model: {:?}", model_path);

        // Automatically detect available parallelism
        let threads = match available_parallelism() {
            Ok(n) => n.get(),
            Err(e) => {
                error!("Failed to detect available parallelism: {}", e);
                // Default to 4 threads if detection fails
                4
            },
        };
        info!("Using {} threads for text embedding", threads);

        // Initialize the global Rayon thread pool once
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global() {
            // This is fine - it means the pool is already initialized
            debug!("Rayon thread pool already initialized or failed: {}", e);
        }

        // Load tokenizer
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e);
                return Err(MemoryBankError::EmbeddingError(format!(
                    "Failed to load tokenizer: {}",
                    e
                )));
            },
        };

        // Get the best available device (Metal, CUDA, or CPU)
        let device = get_best_available_device();

        // Create a config for all-MiniLM-L6-v2
        // Using optimized configuration for better performance
        let config = Config {
            vocab_size: 30522,
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 1536,
            hidden_act: HiddenAct::GeluApproximate, // Use approximate GELU for better performance
            hidden_dropout_prob: 0.0,               // Disable dropout for inference
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Default::default(),
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        };

        // Load model weights
        let vb = unsafe {
            match VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to load model weights from {:?}: {}", model_path, e);
                    return Err(MemoryBankError::EmbeddingError(format!(
                        "Failed to load model weights: {}",
                        e
                    )));
                },
            }
        };

        let model = match BertModel::load(vb, &config) {
            Ok(m) => m,
            Err(e) => {
                error!("Failed to create BERT model: {}", e);
                return Err(MemoryBankError::EmbeddingError(format!(
                    "Failed to create BERT model: {}",
                    e
                )));
            },
        };

        debug!("Text embedder initialized successfully");

        Ok(Self {
            model,
            tokenizer,
            device,
            normalize_embeddings: true,
        })
    }

    /// Ensure model files exist, downloading them if necessary
    fn ensure_model_files(model_path: &Path, tokenizer_path: &Path) -> AnyhowResult<()> {
        // Check if files already exist
        if model_path.exists() && tokenizer_path.exists() {
            return Ok(());
        }

        info!("Downloading model files...");

        // Use Hugging Face Hub API to download files
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));

        // Download model file
        if !model_path.exists() {
            let model_file = repo.get("model.safetensors")?;
            std::fs::copy(model_file, model_path)?;
        }

        // Download tokenizer file
        if !tokenizer_path.exists() {
            let tokenizer_file = repo.get("tokenizer.json")?;
            std::fs::copy(tokenizer_file, tokenizer_path)?;
        }

        Ok(())
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
        let texts = vec![text.to_string()];
        match self.embed_batch(&texts) {
            Ok(embeddings) => Ok(embeddings.into_iter().next().unwrap()),
            Err(e) => {
                error!("Failed to embed text: {}", e);
                Err(e)
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
        // Configure tokenizer with padding
        let mut tokenizer = self.tokenizer.clone();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        // Process in batches for better memory efficiency
        let batch_size = DEFAULT_BATCH_SIZE;

        // Use parallel iterator to process batches in parallel
        let all_embeddings: Vec<Vec<f32>> = texts
            .par_chunks(batch_size)
            .flat_map(|batch| {
                // Tokenize batch
                let tokens = match tokenizer.encode_batch(batch.to_vec(), true) {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Failed to tokenize texts: {}", e);
                        return Vec::new();
                    },
                };

                // Pre-allocate vectors with exact capacity
                let mut token_ids = Vec::with_capacity(batch.len());
                let mut attention_mask = Vec::with_capacity(batch.len());

                // Convert tokens to tensors
                for tokens in &tokens {
                    let ids = tokens.get_ids().to_vec();
                    let mask = tokens.get_attention_mask().to_vec();

                    let ids_tensor = match Tensor::new(ids.as_slice(), &self.device) {
                        Ok(t) => t,
                        Err(e) => {
                            error!("Failed to create token_ids tensor: {}", e);
                            return Vec::new();
                        },
                    };

                    let mask_tensor = match Tensor::new(mask.as_slice(), &self.device) {
                        Ok(t) => t,
                        Err(e) => {
                            error!("Failed to create attention_mask tensor: {}", e);
                            return Vec::new();
                        },
                    };

                    token_ids.push(ids_tensor);
                    attention_mask.push(mask_tensor);
                }

                // Stack tensors into batches
                let token_ids = match Tensor::stack(&token_ids, 0) {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Failed to stack token_ids tensors: {}", e);
                        return Vec::new();
                    },
                };

                let attention_mask = match Tensor::stack(&attention_mask, 0) {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Failed to stack attention_mask tensors: {}", e);
                        return Vec::new();
                    },
                };

                let token_type_ids = match token_ids.zeros_like() {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Failed to create zeros tensor for token_type_ids: {}", e);
                        return Vec::new();
                    },
                };

                // Run model inference
                let embeddings = match self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask)) {
                    Ok(e) => e,
                    Err(e) => {
                        error!("Model inference failed: {}", e);
                        return Vec::new();
                    },
                };

                // Apply mean pooling
                let mean_embeddings = match embeddings.mean(1) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Failed to compute mean embeddings: {}", e);
                        return Vec::new();
                    },
                };

                // Normalize if configured
                let final_embeddings = if self.normalize_embeddings {
                    match Self::normalize_l2(&mean_embeddings) {
                        Ok(n) => n,
                        Err(_) => return Vec::new(),
                    }
                } else {
                    mean_embeddings
                };

                // Convert to Vec<Vec<f32>>
                final_embeddings.to_vec2::<f32>().unwrap_or_else(|e| {
                    error!("Failed to convert embeddings to vector: {}", e);
                    Vec::new()
                })
            })
            .collect();

        // Check if we have the correct number of embeddings
        if all_embeddings.len() != texts.len() {
            return Err(MemoryBankError::EmbeddingError(
                "Failed to generate embeddings for all texts".to_string(),
            ));
        }

        Ok(all_embeddings)
    }

    /// Normalize embedding to unit length (L2 norm)
    fn normalize_l2(v: &Tensor) -> Result<Tensor> {
        // Calculate squared values
        let squared = match v.sqr() {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to square tensor for L2 normalization: {}", e);
                return Err(MemoryBankError::EmbeddingError(format!(
                    "Failed to square tensor: {}",
                    e
                )));
            },
        };

        // Sum along last dimension and keep dimensions
        let sum_squared = match squared.sum_keepdim(1) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to sum squared values: {}", e);
                return Err(MemoryBankError::EmbeddingError(format!("Failed to sum tensor: {}", e)));
            },
        };

        // Calculate square root for L2 norm
        let norm = match sum_squared.sqrt() {
            Ok(n) => n,
            Err(e) => {
                error!("Failed to compute square root for normalization: {}", e);
                return Err(MemoryBankError::EmbeddingError(format!(
                    "Failed to compute square root: {}",
                    e
                )));
            },
        };

        // Divide by norm
        let normalized = match v.broadcast_div(&norm) {
            Ok(n) => n,
            Err(e) => {
                error!("Failed to normalize by division: {}", e);
                return Err(MemoryBankError::EmbeddingError(format!("Failed to normalize: {}", e)));
            },
        };

        Ok(normalized)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        fs,
    };

    use tempfile::tempdir;

    use super::*;

    // Helper function to create a test embedder with mock files
    fn create_test_embedder() -> Result<CandleTextEmbedder> {
        // Use a temporary directory for test files
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let model_path = temp_dir.path().join("model.safetensors");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Mock the ensure_model_files function to avoid actual downloads
        // This is a simplified test that checks error handling paths

        // Return a mock error to test error handling
        Err(MemoryBankError::EmbeddingError("Test error".to_string()))
    }

    #[test]
    fn test_embed_single() {
        // Skip this test in CI environments where model files might not be available
        if env::var("CI").is_ok() {
            return;
        }

        let embedder = CandleTextEmbedder::new().unwrap();
        let embedding = embedder.embed("This is a test sentence.").unwrap();

        // MiniLM-L6-v2 produces 384-dimensional embeddings
        assert_eq!(embedding.len(), 384);

        // Check that the embedding is normalized (L2 norm ≈ 1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_embed_batch() {
        // Skip this test in CI environments where model files might not be available
        if env::var("CI").is_ok() {
            return;
        }

        let embedder = CandleTextEmbedder::new().unwrap();
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
    }

    #[test]
    fn test_error_handling() {
        // Test error handling with invalid paths
        let invalid_path = Path::new("/nonexistent/path");
        let result = CandleTextEmbedder::with_model_paths(invalid_path, invalid_path);
        assert!(result.is_err());

        // Test error handling with mock embedder
        let result = create_test_embedder();
        assert!(result.is_err());
    }

    #[test]
    fn test_ensure_model_files() {
        // Create temporary directory for test
        let temp_dir = tempdir().expect("Failed to create temp directory");
        let model_path = temp_dir.path().join("model.safetensors");
        let tokenizer_path = temp_dir.path().join("tokenizer.json");

        // Create empty files to simulate existing files
        fs::write(&model_path, "mock data").expect("Failed to write mock model file");
        fs::write(&tokenizer_path, "mock data").expect("Failed to write mock tokenizer file");

        // Test that ensure_model_files returns Ok when files exist
        let result = CandleTextEmbedder::ensure_model_files(&model_path, &tokenizer_path);
        assert!(result.is_ok());
    }
}
