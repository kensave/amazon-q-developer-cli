mod trait_def;

mod candle;
/// Mock embedder for testing
#[cfg(test)]
pub mod mock;
#[cfg(any(target_os = "macos", target_os = "windows"))]
mod onnx;

pub use candle::CandleTextEmbedder;
#[cfg(test)]
pub use mock::MockTextEmbedder;
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub use onnx::TextEmbedder;
pub use trait_def::{
    EmbeddingType,
    TextEmbedderTrait,
};
