use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::{
    MemoryBankError,
    Result,
};
use crate::types::FileType;

/// Determine the file type based on extension
pub fn get_file_type(path: &Path) -> FileType {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("txt") => FileType::Text,
        Some("md" | "markdown") => FileType::Markdown,
        Some("json") => FileType::Json,
        // Code file extensions
        Some("rs" | "py" | "js" | "ts" | "java" | "c" | "cpp" | "h" | "hpp" | "go" | "rb" | "php" | "cs" | 
             "swift" | "kt" | "scala" | "sh" | "bash" | "html" | "css" | "sql") => FileType::Code,
        _ => FileType::Unknown,
    }
}

/// Process a file and extract its content
///
/// # Arguments
///
/// * `path` - Path to the file
///
/// # Returns
///
/// A vector of JSON objects representing the file content
pub fn process_file(path: &Path) -> Result<Vec<Value>> {
    if !path.exists() {
        return Err(MemoryBankError::InvalidPath(format!(
            "File does not exist: {}",
            path.display()
        )));
    }

    let file_type = get_file_type(path);
    let content = fs::read_to_string(path).map_err(|e| {
        MemoryBankError::IoError(std::io::Error::new(
            e.kind(),
            format!("Failed to read file {}: {}", path.display(), e),
        ))
    })?;

    match file_type {
        FileType::Text | FileType::Markdown | FileType::Code => {
            // For text-based files, create a single data point
            let mut metadata = serde_json::Map::new();
            metadata.insert("text".to_string(), Value::String(content.clone()));
            metadata.insert("path".to_string(), Value::String(path.to_string_lossy().to_string()));
            metadata.insert("file_type".to_string(), Value::String(format!("{:?}", file_type)));

            // For code files, add additional metadata
            if file_type == FileType::Code {
                metadata.insert(
                    "language".to_string(),
                    Value::String(
                        path.extension()
                            .and_then(|ext| ext.to_str())
                            .unwrap_or("unknown")
                            .to_string(),
                    ),
                );
            }

            Ok(vec![Value::Object(metadata)])
        },
        FileType::Json => {
            // For JSON files, parse the content
            let json: Value = serde_json::from_str(&content).map_err(MemoryBankError::SerializationError)?;

            match json {
                Value::Array(items) => {
                    // If it's an array, return each item
                    Ok(items)
                },
                _ => {
                    // Otherwise, return the whole object
                    Ok(vec![json])
                },
            }
        },
        FileType::Unknown => {
            // For unknown file types, just store the path
            let mut metadata = serde_json::Map::new();
            metadata.insert("path".to_string(), Value::String(path.to_string_lossy().to_string()));
            metadata.insert("file_type".to_string(), Value::String("Unknown".to_string()));

            Ok(vec![Value::Object(metadata)])
        },
    }
}

/// Process a directory and extract content from all files
///
/// # Arguments
///
/// * `dir_path` - Path to the directory
///
/// # Returns
///
/// A vector of JSON objects representing the content of all files
pub fn process_directory(dir_path: &Path) -> Result<Vec<Value>> {
    let mut results = Vec::new();

    for entry in walkdir::WalkDir::new(dir_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();

        // Skip hidden files
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|s| s.starts_with('.'))
        {
            continue;
        }

        // Process the file
        match process_file(path) {
            Ok(mut items) => results.append(&mut items),
            Err(_) => continue, // Skip files that fail to process
        }
    }

    Ok(results)
}
