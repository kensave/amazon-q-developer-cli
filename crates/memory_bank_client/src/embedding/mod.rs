mod trait_def;

mod candle;
#[cfg(any(target_os = "macos", target_os = "windows"))]
mod onnx;

pub use candle::CandleTextEmbedder;
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub use onnx::TextEmbedder;
pub use trait_def::{
    EmbeddingType,
    TextEmbedderTrait,
};
