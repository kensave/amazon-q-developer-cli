use std::path::PathBuf;
use std::sync::{
    Arc,
    LazyLock as Lazy,
};

use eyre::Result;
use semantic_search_client::KnowledgeContext;
use semantic_search_client::client::AsyncSemanticSearchClient;
use semantic_search_client::embedding::EmbeddingType;
use semantic_search_client::types::{
    AddContextRequest,
    SearchResult,
};
use tokio::sync::Mutex;

const CONVERSATION_HISTORY_NAME: &str = "Conversation History";
use uuid::Uuid;

use crate::os::Os;
use crate::util::directories;

/// Configuration for adding knowledge contexts
#[derive(Default)]
pub struct AddOptions {
    pub description: Option<String>,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub embedding_type: Option<String>,
}

impl AddOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create AddOptions with DB default patterns
    pub fn with_db_defaults(os: &crate::os::Os) -> Self {
        let default_include = os
            .database
            .settings
            .get(crate::database::settings::Setting::KnowledgeDefaultIncludePatterns)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let default_exclude = os
            .database
            .settings
            .get(crate::database::settings::Setting::KnowledgeDefaultExcludePatterns)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let default_embedding_type = os
            .database
            .settings
            .get(crate::database::settings::Setting::KnowledgeIndexType)
            .and_then(|v| v.as_str().map(|s| s.to_string()));

        Self {
            description: None,
            include_patterns: default_include,
            exclude_patterns: default_exclude,
            embedding_type: default_embedding_type,
        }
    }

    pub fn with_include_patterns(mut self, patterns: Vec<String>) -> Self {
        self.include_patterns = patterns;
        self
    }

    pub fn with_exclude_patterns(mut self, patterns: Vec<String>) -> Self {
        self.exclude_patterns = patterns;
        self
    }

    pub fn with_embedding_type(mut self, embedding_type: Option<String>) -> Self {
        self.embedding_type = embedding_type;
        self
    }

}

#[derive(Debug, Clone)]
pub struct KnowledgeEntry {
    pub context: KnowledgeContext,
}


/// Async knowledge store - manages agent and global clients
pub struct KnowledgeStore {
    agent_client: AsyncSemanticSearchClient,
}

impl KnowledgeStore {
    /// Get singleton instance with optional agent name
    pub async fn get_async_instance(os: &Os, agent_name: Option<&str>) -> Result<Arc<Mutex<Self>>, directories::DirectoryError> {
        static ASYNC_INSTANCE: Lazy<tokio::sync::OnceCell<Arc<Mutex<KnowledgeStore>>>> =
            Lazy::new(tokio::sync::OnceCell::new);

        if cfg!(test) {
            // For tests, create a new instance each time
            let store = Self::new_with_os_settings(os, agent_name).await
                .map_err(|_| directories::DirectoryError::Io(std::io::Error::new(std::io::ErrorKind::Other, "Failed to create store")))?;
            Ok(Arc::new(Mutex::new(store)))
        } else {
            Ok(ASYNC_INSTANCE
                .get_or_init(|| async {
                    // Check for migration before initializing the client
                    let global_dir = crate::util::directories::global_knowledge_dir(os)
                        .expect("Failed to get global directory");
                    
                    Self::migrate_legacy_knowledge_base(&global_dir).await;

                    let store = Self::new_with_os_settings(os, agent_name)
                        .await
                        .expect("Failed to create knowledge store");
                    Arc::new(Mutex::new(store))
                })
                .await
                .clone())
        }
    }

    /// Helper function to copy directory recursively
    fn copy_dir_recursive(src: &PathBuf, dst: &PathBuf) -> std::io::Result<bool> {
        let mut copied = false;
        if !dst.exists() {
            std::fs::create_dir_all(dst)?;
        }

        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());

            if src_path.is_dir() {
                if Self::copy_dir_recursive(&src_path, &dst_path)? {
                    copied = true;
                }
            } else {
                std::fs::copy(&src_path, &dst_path)?;
                copied = true;
            }
        }
        Ok(copied)
    }

    /// Migrate legacy knowledge base from old location if needed
    async fn migrate_legacy_knowledge_base(knowledge_dir: &PathBuf) -> bool {
        // Create global directory first
        std::fs::create_dir_all(&knowledge_dir).ok();

        let mut migrated = false;

        // Check both possible source locations
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let source_dirs = vec![
            home.join(".semantic_search"),
        ];

        // Migrate from legacy locations
        for src in source_dirs {
            if src.exists() {
                if let Ok(entries) = std::fs::read_dir(&src) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let src_path = entry.path();
                        let dst_path = knowledge_dir.join(&name);

                        if !dst_path.exists() {
                            if let Ok(true) = Self::copy_dir_recursive(&src_path, &dst_path) {
                                migrated = true;
                            }
                        }
                    }
                }
            }
        }

        // Migrate from knowledge_bases root to __global_knowledge__ (but avoid self-copy)
        let knowledge_bases_root = knowledge_dir.parent();
        if let Some(kb_root) = knowledge_bases_root {
            if kb_root.exists() && kb_root != knowledge_dir {
                if let Ok(entries) = std::fs::read_dir(kb_root) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        let src_path = entry.path();
                        
                        // Skip the __global_knowledge__ directory itself to avoid recursion
                        if name == "__global_knowledge__" {
                            continue;
                        }
                        
                        // Skip database files - only migrate knowledge contexts and metadata
                        let name_str = name.to_string_lossy();
                        if name_str == "models" || name_str.starts_with('.') {
                            continue;
                        }
                        
                        // Only migrate directories that look like UUIDs or agent names, and contexts.json
                        let dst_path = knowledge_dir.join(&name);
                        if !dst_path.exists() {
                            if src_path.is_dir() {
                                if let Ok(true) = Self::copy_dir_recursive(&src_path, &dst_path) {
                                    migrated = true;
                                }
                            } else if name_str == "contexts.json" {
                                if std::fs::copy(&src_path, &dst_path).is_ok() {
                                    migrated = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        migrated
    }



    /// Create SemanticSearchConfig from database settings with fallbacks to defaults
    fn create_config_from_db_settings(
        os: &crate::os::Os,
        base_dir: PathBuf,
    ) -> semantic_search_client::config::SemanticSearchConfig {
        use semantic_search_client::config::SemanticSearchConfig;
        use semantic_search_client::embedding::EmbeddingType;

        use crate::database::settings::Setting;

        // Create default config first
        let default_config = SemanticSearchConfig {
            base_dir: base_dir.clone(),
            ..Default::default()
        };

        // Override with DB settings if provided, otherwise use defaults
        let chunk_size = os
            .database
            .settings
            .get_int_or(Setting::KnowledgeChunkSize, default_config.chunk_size);
        let chunk_overlap = os
            .database
            .settings
            .get_int_or(Setting::KnowledgeChunkOverlap, default_config.chunk_overlap);
        let max_files = os
            .database
            .settings
            .get_int_or(Setting::KnowledgeMaxFiles, default_config.max_files);
        
        // Get embedding type from settings
        let embedding_type = os
            .database
            .settings
            .get_string(Setting::KnowledgeIndexType)
            .and_then(|s| EmbeddingType::from_str(&s))
            .unwrap_or_default();

        SemanticSearchConfig {
            chunk_size,
            chunk_overlap,
            max_files,
            embedding_type,
            base_dir,
            ..default_config
        }
    }

    /// Create instance with database settings from OS - for tests only
    async fn new_with_os_settings(os: &crate::os::Os, agent_name: Option<&str>) -> Result<Self> {
        // Always create global client
        // Always create agent client, use default if no name provided
        let agent_name = agent_name.unwrap_or("default");
        let agent_dir = crate::util::directories::agent_knowledge_dir(os, agent_name)?;
        let agent_config = Self::create_config_from_db_settings(os, agent_dir.clone());
        let agent_client = AsyncSemanticSearchClient::with_config(&agent_dir, agent_config)
            .await
            .map_err(|e| eyre::eyre!("Failed to create agent client at {}: {}", agent_dir.display(), e))?;

        let store = Self {
            agent_client,
        };
        // Auto-create "Conversation History" context if it doesn't exist
        store.ensure_conversation_context().await?;

        Ok(store)
    }

    /// Ensure "Conversation History" context exists (BM25 for fast text search)
    /// Ensure "Conversation History" context exists (BM25 for fast text search)
    async fn ensure_conversation_context(&self) -> Result<()> {
        let contexts = self.agent_client.get_contexts().await;
        let conversation_exists = contexts.iter().any(|c| c.name == CONVERSATION_HISTORY_NAME);
        if !conversation_exists {
            // Create conversation context as semantic (Best embedding for better search quality)
            let _ = self.agent_client.add_context_from_text(
                CONVERSATION_HISTORY_NAME,
                "Automatically created context for conversation history",
                Some(semantic_search_client::embedding::EmbeddingType::Best), // Semantic context
            ).await?;
        }
        Ok(())
    }

    /// Add conversation content to the "Conversation History" context
    pub async fn add_conversation_content(&self, content: &str) -> Result<()> {
        // Find the Conversation History context
        let contexts = self.agent_client.get_contexts().await;
        
        if let Some(conv_context) = contexts.iter().find(|c| c.name == CONVERSATION_HISTORY_NAME) {
            self.agent_client.add_text_to_context(&conv_context.id, content).await
                .map_err(|e| eyre::eyre!("Failed to add conversation content: {}", e))?;
        }
        Ok(())
    }

    /// Add content to an existing context
    pub async fn add_to_context(&self, context_id: &str, content: &str) -> Result<()> {
        self.agent_client.add_to_context(context_id, content).await
            .map_err(|e| eyre::eyre!("Failed to add content to context: {}", e))
    }

    /// Get all contexts from agent client
    pub async fn get_all(&self) -> Result<Vec<KnowledgeContext>, String> {
        Ok(self.agent_client.get_contexts().await)
    }

    /// Get all contexts from appropriate client (deprecated - use get_all instead)
    pub async fn get_all_for_scope(&self, is_global: bool) -> Result<Vec<KnowledgeEntry>, String> {
        let contexts = if is_global {
            vec![] // No global client anymore
        } else {
            self.agent_client.get_contexts().await
        };

        let entries = contexts.into_iter().map(|context| KnowledgeEntry {
            context,
        }).collect();

        Ok(entries)
    }

    /// Add context to specific scope (agent or global)
    pub async fn add(&mut self, name: &str, path_str: &str, options: AddOptions) -> Result<String, String> {
        let path_buf = std::path::PathBuf::from(path_str);
        let canonical_path = path_buf
            .canonicalize()
            .map_err(|_io_error| format!("❌ Path does not exist: {}", path_str))?;

        // Use provided description or generate default
        let description = options
            .description
            .unwrap_or_else(|| format!("Knowledge context for {}", name));

        // Create AddContextRequest with all options
        let request = AddContextRequest {
            path: canonical_path.clone(),
            name: name.to_string(),
            description: if !options.include_patterns.is_empty() || !options.exclude_patterns.is_empty() {
                let mut full_description = description;
                if !options.include_patterns.is_empty() {
                    full_description.push_str(&format!(" [Include: {}]", options.include_patterns.join(", ")));
                }
                if !options.exclude_patterns.is_empty() {
                    full_description.push_str(&format!(" [Exclude: {}]", options.exclude_patterns.join(", ")));
                }
                full_description
            } else {
                description
            },
            persistent: true,
            include_patterns: if options.include_patterns.is_empty() {
                None
            } else {
                Some(options.include_patterns.clone())
            },
            exclude_patterns: if options.exclude_patterns.is_empty() {
                None
            } else {
                Some(options.exclude_patterns.clone())
            },
            embedding_type: match options.embedding_type.as_ref() {
                Some(s) => match EmbeddingType::from_str(s) {
                    Some(et) => Some(et),
                    None => {
                        return Err(format!("Invalid embedding type '{}'. Valid options are: fast, best", s));
                    },
                },
                None => None,
            },
            id: None, // Let the system generate UUID
        };

        match self.agent_client.add_context(request).await {
            Ok((operation_id, _)) => {
                let mut message = format!(
                    "🚀 Started indexing '{}'\n📁 Path: {}\n🆔 Operation ID: {}",
                    name,
                    canonical_path.display(),
                    &operation_id.to_string()[..8]
                );
                if !options.include_patterns.is_empty() || !options.exclude_patterns.is_empty() {
                    message.push_str("\n📋 Pattern filtering applied:");
                    if !options.include_patterns.is_empty() {
                        message.push_str(&format!("\n   Include: {}", options.include_patterns.join(", ")));
                    }
                    if !options.exclude_patterns.is_empty() {
                        message.push_str(&format!("\n   Exclude: {}", options.exclude_patterns.join(", ")));
                    }
                    message.push_str("\n✅ Only matching files will be indexed");
                }
                Ok(message)
            },
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("Invalid include pattern") || error_msg.contains("Invalid exclude pattern") {
                    Err(error_msg)
                } else {
                    Err(format!("Failed to start indexing: {}", e))
                }
            },
        }
    }

    /// Search - delegates to async client
    pub async fn search(&self, query: &str, _context_id: Option<&str>) -> Result<Vec<SearchResult>, String> {
        let mut flattened = Vec::new();
        
        // Search agent client
        if let Ok(agent_results) = self.agent_client.search_all(query, None).await {
            for (_, context_results) in agent_results {
                flattened.extend(context_results);
            }
        }
        
        flattened.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        Ok(flattened)
    }

    /// Get status data - only from agent client
    pub async fn get_status_data(&self) -> Result<semantic_search_client::SystemStatus, String> {
        self.agent_client
            .get_status_data()
            .await
            .map_err(|e| e.to_string())
    }

    /// Cancel operation - only from agent client
    pub async fn cancel_operation(&mut self, operation_id: Option<&str>) -> Result<String, String> {
        if let Some(short_id) = operation_id {
            let available_ops = self.agent_client.list_operation_ids().await;
            if available_ops.is_empty() {
                return Ok("No active operations to cancel".to_string());
            }

            // Try to parse as full UUID first
            if let Ok(uuid) = Uuid::parse_str(short_id) {
                self.agent_client.cancel_operation(uuid).await.map_err(|e| e.to_string())
            } else {
                // Try to find by short ID (first 8 characters)
                if let Some(full_uuid) = self.agent_client.find_operation_by_short_id(short_id).await {
                    self.agent_client.cancel_operation(full_uuid).await.map_err(|e| e.to_string())
                } else {
                    let available_ops_str: Vec<String> = available_ops.iter().map(|id| id.to_string()[..8].to_string()).collect();
                    Err(format!(
                        "Operation '{}' not found. Available operations: {}",
                        short_id,
                        available_ops_str.join(", ")
                    ))
                }
            }
        } else {
            // Cancel most recent operation
            self.agent_client
                .cancel_most_recent_operation()
                .await
                .map_err(|e| e.to_string())
        }
    }

    /// Clear all contexts - only from agent client
    pub async fn clear(&mut self) -> Result<String, String> {
        match self.agent_client.clear_all().await {
            Ok((operation_id, _cancel_token)) => Ok(format!(
                "🚀 Started clearing all contexts in background.\n📊 Use 'knowledge status' to check progress.\n🆔 Operation ID: {}",
                &operation_id.to_string()[..8]
            )),
            Err(e) => Err(format!("Failed to start clear operation: {}", e)),
        }
    }

    /// Remove context by path - checks both agent and global clients
    pub async fn remove_by_path(&mut self, path: &str) -> Result<(), String> {
        // Try agent client first if available
        if let Some(context) = self.agent_client.get_context_by_path(path).await {
            self.agent_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err(format!("No context found with path '{}'", path))
        }
    }

    /// Remove context by name - checks agent client only
    pub async fn remove_by_name(&mut self, name: &str) -> Result<(), String> {
        if let Some(context) = self.agent_client.get_context_by_name(name).await {
            self.agent_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err(format!("No context found with name '{}'", name))
        }
    }

    /// Remove context by ID - tries agent client only
    pub async fn remove_by_id(&mut self, context_id: &str) -> Result<(), String> {
        self.agent_client.remove_context_by_id(context_id).await.map_err(|e| e.to_string())
    }

    /// Update context by path - checks agent client only
    pub async fn update_by_path(&mut self, path_str: &str) -> Result<String, String> {
        if let Some(context) = self.agent_client.get_context_by_path(path_str).await {
            // Remove the existing context first
            self.agent_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())?;

            // Then add it back with the same name and original patterns (agent scope)
            let options = AddOptions {
                description: None,
                include_patterns: context.include_patterns.clone(),
                exclude_patterns: context.exclude_patterns.clone(),
                embedding_type: None,
            };
            self.add(&context.name, path_str, options).await
        } else {
            Err(format!("No context found with path '{}'", path_str))
        }
    }

    /// Update context by name - checks agent client only
    pub async fn update_context_by_name(&mut self, name: &str, path_str: &str) -> Result<String, String> {
        if let Some(context) = self.agent_client.get_context_by_name(name).await {
            // Remove the existing context first
            self.agent_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())?;

            // Then add it back with the same name and original patterns (agent scope)
            let options = AddOptions {
                description: None,
                include_patterns: context.include_patterns.clone(),
                exclude_patterns: context.exclude_patterns.clone(),
                embedding_type: None,
            };
            self.add(name, path_str, options).await
        } else {
            Err(format!("Context with name '{}' not found", name))
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::os::Os;

    async fn create_test_os(temp_dir: &TempDir) -> Os {
        let os = Os::new().await.unwrap();
        // Override home directory to use temp directory
        unsafe {
            os.env.set_var("HOME", temp_dir.path().to_str().unwrap());
        }
        os
    }

    #[tokio::test]
    async fn test_create_config_from_db_settings() {
        let temp_dir = TempDir::new().unwrap();
        let os = create_test_os(&temp_dir).await;
        let base_dir = temp_dir.path().join("test_kb");

        // Test config creation with default settings
        let config = KnowledgeStore::create_config_from_db_settings(&os, base_dir.clone());

        // Should use defaults when no database settings exist
        assert_eq!(config.chunk_size, 512); // Default chunk size
        assert_eq!(config.chunk_overlap, 128); // Default chunk overlap
        assert_eq!(config.max_files, 10000); // Default max files
        assert_eq!(config.base_dir, base_dir);
    }

    #[tokio::test]
    async fn test_knowledge_bases_dir_structure() {
        let temp_dir = TempDir::new().unwrap();
        let os = create_test_os(&temp_dir).await;

        let base_dir = crate::util::directories::knowledge_bases_dir(&os).unwrap();

        // Verify directory structure
        assert!(base_dir.to_string_lossy().contains("knowledge_bases"));
    }

    #[tokio::test]
    async fn test_agent_knowledge_dir_creation() {
        let temp_dir = TempDir::new().unwrap();
        let os = create_test_os(&temp_dir).await;

        let agent_dir = crate::util::directories::agent_knowledge_dir(&os, "test-agent").unwrap();
        let global_dir = crate::util::directories::global_knowledge_dir(&os).unwrap();

        // Verify paths are different
        assert_ne!(agent_dir, global_dir);
        assert!(agent_dir.to_string_lossy().contains("test-agent"));
        assert!(global_dir.to_string_lossy().contains("global_knowledge"));
    }
}
