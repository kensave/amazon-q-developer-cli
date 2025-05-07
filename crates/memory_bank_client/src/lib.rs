//! Memory Bank Client - A library for managing semantic memory contexts
//!
//! This crate provides functionality for creating, managing, and searching
//! semantic memory contexts. It uses vector embeddings to enable semantic search
//! across text and code.

#![warn(missing_docs)]

/// Client implementation for memory bank operations
pub mod client;
/// Error types for memory bank operations
pub mod error;
/// Vector index implementation
pub mod index;
/// File processing utilities
pub mod processing;
/// Data types for memory operations
pub mod types;

// Platform-specific embedding modules
/// Text embedding functionality using fastembed (macOS and Windows only)
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub mod embedding;
/// Text embedding functionality using candle (used on all platforms, default on Linux)
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub mod embedding_candle;

pub use client::MemoryBankClient;
pub use error::{
    MemoryBankError,
    Result,
};
pub use types::{
    DataPoint,
    FileType,
    MemoryContext,
    ProgressStatus,
    SearchResult,
};
