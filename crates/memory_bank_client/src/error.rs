use thiserror::Error;

/// Errors that can occur when using the memory bank client
#[derive(Error, Debug)]
pub enum MemoryBankError {
    /// IO error from the standard library
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization or deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Error generating text embeddings
    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    /// Error with the vector index
    #[error("Index error: {0}")]
    IndexError(String),

    /// Requested context was not found
    #[error("Context not found: {0}")]
    ContextNotFound(String),

    /// Invalid file or directory path
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// General operation failure
    #[error("Memory operation failed: {0}")]
    OperationFailed(String),

    /// Error from the fastembed library
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[error("Fastembed error: {0}")]
    FastembedError(String),

    /// Other miscellaneous errors
    #[error("Other error: {0}")]
    Other(String),
}

// Handle fastembed errors by converting them to FastembedError
#[cfg(any(target_os = "macos", target_os = "windows"))]
impl From<fastembed::Error> for MemoryBankError {
    fn from(err: fastembed::Error) -> Self {
        MemoryBankError::FastembedError(err.to_string())
    }
}

/// Result type for memory bank operations
pub type Result<T> = std::result::Result<T, MemoryBankError>;
