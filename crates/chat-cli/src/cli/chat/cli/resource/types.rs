use serde::{Deserialize, Serialize};
use clap::ValueEnum;
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, ValueEnum)]
pub enum StorageType {
    Pinned,
    Indexed,
    All,
}

impl fmt::Display for StorageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for StorageType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pinned" => Ok(StorageType::Pinned),
            "indexed" => Ok(StorageType::Indexed),
            "all" => Ok(StorageType::All),
            _ => Err(format!("Invalid storage type: {}", s))
        }
    }
}

impl StorageType {
    pub fn as_str(&self) -> &'static str {
        match self {
            StorageType::Pinned => "pinned",
            StorageType::Indexed => "indexed",
            StorageType::All => "all"
        }
    }

    pub fn from_str_option(s: &str) -> Option<Self> {
        match s {
            "pinned" => Some(StorageType::Pinned),
            "indexed" => Some(StorageType::Indexed),
            "all" => Some(StorageType::All),
            _ => None
        }
    }
}

impl Default for StorageType {
    fn default() -> Self {
        StorageType::Pinned
    }
}

/// Core resource operation types
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceOperation {
    Add { 
        name: String, 
        value: String,
        include_patterns: Option<Vec<String>>,
        exclude_patterns: Option<Vec<String>>,
        index_type: Option<String>,
    },
    Update { path: String },
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
    PlainText
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::PlainText
    }
}
