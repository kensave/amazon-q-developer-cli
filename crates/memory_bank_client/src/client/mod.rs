/// Factory for creating embedders
pub mod embedder_factory;
/// Client implementation for memory bank operations
mod implementation;
/// Semantic context implementation for memory operations
pub mod semantic_context;
/// Utility functions for memory bank operations
pub mod utils;

pub use implementation::MemoryBankClient;
pub use semantic_context::SemanticContext;
