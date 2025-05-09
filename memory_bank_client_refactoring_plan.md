# Memory Bank Client Refactoring Plan

## Implementation Checklist

1. **Error Handling**: Migrate to `thiserror` for better error definitions
2. **Logging**: Replace `eprintln!` with `tracing` for structured logging
3. **Type Aliases**: Add type aliases for common types
4. **Code Organization**: Break down large methods into smaller functions
5. **Documentation**: Improve documentation with examples
6. **Concurrency**: Update concurrency patterns if needed
7. **Configuration**: Improve configuration management
8. **Testing**: Ensure tests pass after each change

## Implementation Steps

### Step 1: Error Handling with thiserror

First, add thiserror to the dependencies:

```bash
cd /Volumes/workplace/QCLI/amazon-q-developer-cli
cargo add thiserror --package memory_bank_client
```

Then, update the error.rs file:

```rust
// crates/memory_bank_client/src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryBankError {
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    
    #[error(transparent)]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Context not found: {0}")]
    ContextNotFound(String),
    
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    
    #[error("Operation failed: {0}")]
    OperationFailed(String),
    
    #[error("Embedding error: {0}")]
    EmbeddingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, MemoryBankError>;
```

### Step 2: Add Logging with tracing

Add tracing to the dependencies:

```bash
cd /Volumes/workplace/QCLI/amazon-q-developer-cli
cargo add tracing --package memory_bank_client
```

Update client code to use tracing instead of eprintln:

```rust
// Replace eprintln statements with tracing
// For example:
// eprintln!("Failed to initialize memory bank configuration: {}", e);
// becomes:
tracing::error!("Failed to initialize memory bank configuration: {}", e);
```

### Step 3: Add Type Aliases

Add type aliases for common types:

```rust
// Add to the top of client.rs or in a separate types.rs file
pub type ContextId = String;
pub type SearchResults = Vec<SearchResult>;
pub type ContextMap = HashMap<ContextId, Arc<Mutex<SemanticContext>>>;
```

### Step 4: Break Down Large Methods

Refactor the `add_context_from_directory` method:

```rust
pub fn add_context_from_directory<F>(
    &mut self,
    dir_path: impl AsRef<Path>,
    name: &str,
    description: &str,
    persistent: bool,
    progress_callback: Option<F>,
) -> Result<ContextId>
where
    F: Fn(ProgressStatus) + Send + 'static,
{
    let dir_path = dir_path.as_ref();
    let id = Uuid::new_v4().to_string();
    
    // Create context directory
    let context_dir = self.create_context_directory(&id, persistent)?;
    
    // Count files and notify progress
    let file_count = self.count_files_in_directory(dir_path, &progress_callback)?;
    
    // Process files
    let items = self.process_directory_files(dir_path, file_count, &progress_callback)?;
    
    // Create and populate semantic context
    let semantic_context = self.create_semantic_context(&context_dir, &items, &progress_callback)?;
    
    // Save and store context
    self.save_and_store_context(&id, name, description, persistent, Some(dir_path.to_string_lossy().to_string()), semantic_context)?;
    
    Ok(id)
}

// Helper methods
fn create_context_directory(&self, id: &str, persistent: bool) -> Result<PathBuf> {
    let context_dir = if persistent {
        let context_dir = self.base_dir.join(id);
        fs::create_dir_all(&context_dir)?;
        context_dir
    } else {
        // For volatile contexts, use a temporary directory
        let temp_dir = std::env::temp_dir().join("memory_bank").join(id);
        fs::create_dir_all(&temp_dir)?;
        temp_dir
    };
    
    Ok(context_dir)
}

// Implement other helper methods...
```

### Step 5: Improve Documentation

Enhance the documentation for the MemoryBankClient struct:

```rust
/// Memory bank client for managing semantic memory
///
/// This client provides functionality for creating, managing, and searching
/// through semantic memory contexts. It supports both volatile (in-memory)
/// and persistent (on-disk) contexts.
///
/// # Examples
///
/// ```
/// use memory_bank_client::MemoryBankClient;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let client = MemoryBankClient::new_with_default_dir()?;
/// let context_id = client.add_context_from_text(
///     "This is a test text for semantic memory",
///     "Test Context",
///     "A test context",
///     false
/// )?;
/// # Ok(())
/// # }
/// ```
pub struct MemoryBankClient {
    // ...
}
```

### Step 6: Update Concurrency Patterns

Consider updating concurrency patterns if needed:

```rust
// Replace std::sync::Mutex with tokio::sync::Mutex for async contexts
use tokio::sync::Mutex;

// Use atomic types for simple flags
use std::sync::atomic::{AtomicBool, Ordering};

// Example usage
let is_modified = Arc::new(AtomicBool::new(false));
is_modified.store(true, Ordering::Relaxed);
```

### Step 7: Improve Configuration Management

Enhance configuration management:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    pub base_dir: PathBuf,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub default_results: usize,
    pub model_name: String,
    pub timeout: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_dir: dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".memory_bank"),
            chunk_size: 1000,
            chunk_overlap: 200,
            default_results: 5,
            model_name: "all-MiniLM-L6-v2".to_string(),
            timeout: 30000, // 30 seconds
        }
    }
}
```

## Testing Strategy

After each change:

1. Run unit tests: `cargo test -p memory_bank_client`
2. Verify compilation: `cargo build -p memory_bank_client`
3. Check for any regressions in functionality
4. Update any affected tests

## Progress Tracking

- [x] Step 1: Error Handling with thiserror
- [x] Step 2: Add Logging with tracing
- [x] Step 3: Add Type Aliases
- [x] Step 4: Break Down Large Methods
- [x] Step 5: Improve Documentation
- [x] Step 6: Update Concurrency Patterns
- [x] Step 7: Improve Configuration Management
