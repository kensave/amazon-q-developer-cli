use std::{
    fmt,
    io,
};

/// Result type for memory bank operations
pub type Result<T> = std::result::Result<T, MemoryBankError>;

/// Error types for memory bank operations
#[derive(Debug)]
pub enum MemoryBankError {
    /// I/O error
    IoError(io::Error),
    /// JSON serialization/deserialization error
    SerdeError(serde_json::Error),
    /// JSON serialization/deserialization error (string variant)
    SerializationError(String),
    /// Invalid path
    InvalidPath(String),
    /// Context not found
    ContextNotFound(String),
    /// Operation failed
    OperationFailed(String),
    /// Invalid argument
    InvalidArgument(String),
    /// Embedding error
    EmbeddingError(String),
    /// Fastembed error
    FastembedError(String),
}

impl fmt::Display for MemoryBankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryBankError::IoError(e) => write!(f, "I/O error: {}", e),
            MemoryBankError::SerdeError(e) => write!(f, "Serialization error: {}", e),
            MemoryBankError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            MemoryBankError::InvalidPath(path) => write!(f, "Invalid path: {}", path),
            MemoryBankError::ContextNotFound(id) => write!(f, "Context not found: {}", id),
            MemoryBankError::OperationFailed(msg) => write!(f, "Operation failed: {}", msg),
            MemoryBankError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            MemoryBankError::EmbeddingError(msg) => write!(f, "Embedding error: {}", msg),
            MemoryBankError::FastembedError(msg) => write!(f, "Fastembed error: {}", msg),
        }
    }
}

impl std::error::Error for MemoryBankError {}

impl From<io::Error> for MemoryBankError {
    fn from(error: io::Error) -> Self {
        MemoryBankError::IoError(error)
    }
}

impl From<serde_json::Error> for MemoryBankError {
    fn from(error: serde_json::Error) -> Self {
        MemoryBankError::SerdeError(error)
    }
}
