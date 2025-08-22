use serde::{Deserialize, Serialize};

/// Core resource operation types
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceOperation {
    Add { name: String, value: String },
    Remove { id: Option<String>, name: Option<String>, path: Option<String> },
    Show { expand: bool },
    Search { query: String, context_id: Option<String> },
    Clear { confirm: bool },
    Status,
    Cancel { operation_id: String },
}

/// Resource data returned from operations
#[derive(Debug, Clone)]
pub enum ResourceData {
    Success(String),
    PinnedResources(PinnedResourceData),
    IndexedResources(IndexedResourceData),
    Status(StatusData),
}

/// Pinned resources with context-specific metadata
#[derive(Debug, Clone)]
pub struct PinnedResourceData {
    pub agent_files: Vec<ContextPath>,
    pub session_files: Vec<ContextPath>,
    pub matched_files: Vec<MatchedFile>,
    pub total_tokens: usize,
    pub dropped_files: Option<Vec<(String, String)>>,
    pub context_files_max_size: usize,
}

/// Context path with match information
#[derive(Debug, Clone)]
pub struct ContextPath {
    pub path: String,
    pub match_count: usize,
}

/// Matched file with metadata
#[derive(Debug, Clone)]
pub struct MatchedFile {
    pub filename: String,
    pub content: String,
    pub tokens: usize,
    pub is_temporary: bool, // true for session, false for agent
}

/// Indexed resources with knowledge-specific metadata
#[derive(Debug, Clone)]
pub struct IndexedResourceData {
    pub items: Vec<ResourceItem>,
}

/// Individual resource item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceItem {
    pub id: String,
    pub name: String,
    pub content: Option<String>,
    pub metadata: ResourceMetadata,
}

/// Resource metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub size: usize,
    pub resource_type: String,
    pub token_count: Option<usize>,
}

/// Status information for background operations
#[derive(Debug, Clone, Serialize)]
pub struct StatusData {
    pub active_operations: Vec<OperationInfo>,
    pub total_items: usize,
    pub storage_info: StorageInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct OperationInfo {
    pub id: String,
    pub operation_type: String,
    pub status: String,
    pub progress: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StorageInfo {
    pub total_size: usize,
    pub item_count: usize,
}

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Table,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Table
    }
}
