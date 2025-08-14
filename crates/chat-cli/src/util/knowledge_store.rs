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
use tracing::debug;
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

#[derive(Debug)]
pub enum KnowledgeError {
    ClientError(String),
}

impl std::fmt::Display for KnowledgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KnowledgeError::ClientError(msg) => write!(f, "Client error: {}", msg),
        }
    }
}

impl std::error::Error for KnowledgeError {}

#[derive(Debug, Clone)]
pub struct KnowledgeEntry {
    pub context: KnowledgeContext,
}


/// Async knowledge store - manages agent and global clients
pub struct KnowledgeStore {
    agent_client: Option<AsyncSemanticSearchClient>,
    global_client: AsyncSemanticSearchClient,
}

impl KnowledgeStore {
    /// Get singleton instance with agent-aware directory selection
    pub async fn get_async_instance_with_agent(
        os: &Os,
        agent_name: Option<&str>,
        is_global: bool,
    ) -> Result<Arc<Mutex<Self>>, directories::DirectoryError> {
        let knowledge_dir = if is_global {
            let global_dir = crate::util::directories::global_knowledge_dir(os)?;
            // Only migrate when accessing global context - existing knowledge becomes global
            Self::migrate_legacy_knowledge_base(&global_dir).await;
            global_dir
        } else if let Some(agent) = agent_name {
            crate::util::directories::agent_knowledge_dir(os, agent)?
        } else {
            // Fallback to original behavior
            crate::util::directories::knowledge_bases_dir(os)?
        };

        Ok(Self::get_async_instance_with_os_settings(os, knowledge_dir).await)
    }

    /// Get singleton instance with directory from OS (includes migration)
    pub async fn get_async_instance_with_os(os: &Os) -> Result<Arc<Mutex<Self>>, directories::DirectoryError> {
        let knowledge_dir = crate::util::directories::knowledge_bases_dir(os)?;
        Self::migrate_legacy_knowledge_base(&knowledge_dir).await;
        Ok(Self::get_async_instance_with_os_settings(os, knowledge_dir).await)
    }

    /// Migrate legacy knowledge base from old location if needed
    async fn migrate_legacy_knowledge_base(knowledge_dir: &PathBuf) {
        let old_flat_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".semantic_search");

        if old_flat_dir.exists() && !knowledge_dir.exists() {
            // Create parent directories first
            if let Some(parent) = knowledge_dir.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    debug!(
                        "Warning: Failed to create parent directories for knowledge base migration: {}",
                        e
                    );
                    return;
                }
            }

            // Attempt migration
            if let Err(e) = std::fs::rename(&old_flat_dir, knowledge_dir) {
                debug!(
                    "Warning: Failed to migrate legacy knowledge base from {} to {}: {}",
                    old_flat_dir.display(),
                    knowledge_dir.display(),
                    e
                );
            } else {
                println!(
                    "✅ Migrated knowledge base from {} to {}",
                    old_flat_dir.display(),
                    knowledge_dir.display()
                );
            }
        }
    }

    /// Get singleton instance with OS settings (primary method)
    pub async fn get_async_instance_with_os_settings(os: &crate::os::Os, base_dir: PathBuf) -> Arc<Mutex<Self>> {
        static ASYNC_INSTANCE: Lazy<tokio::sync::OnceCell<Arc<Mutex<KnowledgeStore>>>> =
            Lazy::new(tokio::sync::OnceCell::new);

        if cfg!(test) {
            Arc::new(Mutex::new(
                KnowledgeStore::new_with_os_settings(os, base_dir)
                    .await
                    .expect("Failed to create test async knowledge store"),
            ))
        } else {
            ASYNC_INSTANCE
                .get_or_init(|| async {
                    Arc::new(Mutex::new(
                        KnowledgeStore::new_with_os_settings(os, base_dir)
                            .await
                            .expect("Failed to create async knowledge store"),
                    ))
                })
                .await
                .clone()
        }
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

    /// Create instance with database settings from OS - creates both agent and global clients
    pub async fn new_with_os_settings(os: &crate::os::Os, base_dir: PathBuf) -> Result<Self> {
        // Extract agent name from base_dir if it's an agent directory
        let agent_name = if let Some(parent) = base_dir.parent() {
            if parent.file_name().and_then(|n| n.to_str()) == Some("knowledge_bases") {
                base_dir.file_name().and_then(|n| n.to_str()).map(|s| s.to_string())
            } else {
                None
            }
        } else {
            None
        };

        // Always create global client
        let global_dir = crate::util::directories::global_knowledge_dir(os)?;
        let global_config = Self::create_config_from_db_settings(os, global_dir.clone());
        let global_client = AsyncSemanticSearchClient::with_config(&global_dir, global_config)
            .await
            .map_err(|e| eyre::eyre!("Failed to create global client: {}", e))?;

        // Try to create agent client - first from extracted agent name, then try to find active agent
        let agent_client = if let Some(ref agent) = agent_name {
            let agent_dir = crate::util::directories::agent_knowledge_dir(os, agent)?;
            let agent_config = Self::create_config_from_db_settings(os, agent_dir.clone());
            AsyncSemanticSearchClient::with_config(&agent_dir, agent_config).await.ok()
        } else {
            None
        };

        Ok(Self {
            agent_client,
            global_client,
        })
    }

    /// Get all contexts from appropriate client
    pub async fn get_all_for_scope(&self, is_global: bool) -> Result<Vec<KnowledgeEntry>, String> {
        let contexts = if is_global {
            self.global_client.get_contexts().await
        } else if let Some(ref agent_client) = self.agent_client {
            agent_client.get_contexts().await
        } else {
            return Ok(Vec::new());
        };

        let entries = contexts.into_iter().map(|context| KnowledgeEntry {
            context,
        }).collect();

        Ok(entries)
    }

    /// Add context with flexible options - routes to global client by default
    /// Add context to agent scope by default (for function calling)
    pub async fn add(&mut self, name: &str, path_str: &str, options: AddOptions) -> Result<String, String> {
        self.add_with_scope(name, path_str, options, false).await
    }

    /// Add context to specific scope (agent or global)
    pub async fn add_with_scope(&mut self, name: &str, path_str: &str, options: AddOptions, is_global: bool) -> Result<String, String> {
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
        };

        let client = if is_global {
            &mut self.global_client
        } else if let Some(ref mut agent_client) = self.agent_client {
            agent_client
        } else {
            return Err("No agent context available for agent-specific knowledge".to_string());
        };

        match client.add_context(request).await {
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

    /// Get all contexts from both agent and global clients
    pub async fn get_all(&self) -> Result<Vec<KnowledgeContext>, KnowledgeError> {
        let mut all_contexts = Vec::new();
        
        // Get agent contexts if available
        if let Some(ref agent_client) = self.agent_client {
            let agent_contexts = agent_client.get_contexts().await;
            all_contexts.extend(agent_contexts);
        }
        
        // Get global contexts
        let global_contexts = self.global_client.get_contexts().await;
        all_contexts.extend(global_contexts);
        
        Ok(all_contexts)
    }

    /// Search - delegates to async client
    pub async fn search(&self, query: &str, _context_id: Option<&str>) -> Result<Vec<SearchResult>, KnowledgeError> {
        let mut flattened = Vec::new();
        
        // Search agent client if available
        if let Some(ref agent_client) = self.agent_client {
            if let Ok(agent_results) = agent_client.search_all(query, None).await {
                for (_, context_results) in agent_results {
                    flattened.extend(context_results);
                }
            }
        }
        
        // Search global client
        let global_results = self
            .global_client
            .search_all(query, None)
            .await
            .map_err(|e| KnowledgeError::ClientError(e.to_string()))?;

        for (_, context_results) in global_results {
            flattened.extend(context_results);
        }

        flattened.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        Ok(flattened)
    }

    /// Get status data - combines status from both agent and global clients
    pub async fn get_status_data(&self) -> Result<semantic_search_client::SystemStatus, String> {
        let mut global_status = self.global_client
            .get_status_data()
            .await
            .map_err(|e| format!("Failed to get global status data: {}", e))?;

        // If agent client exists, merge its operations with global status
        if let Some(ref agent_client) = self.agent_client {
            if let Ok(agent_status) = agent_client.get_status_data().await {
                // Merge operations from agent client
                global_status.operations.extend(agent_status.operations);
                // Update counts
                global_status.active_count += agent_status.active_count;
                global_status.waiting_count += agent_status.waiting_count;
            }
        }

        Ok(global_status)
    }

    /// Cancel operation - delegates to async client
    pub async fn cancel_operation(&mut self, operation_id: Option<&str>) -> Result<String, String> {
        if let Some(short_id) = operation_id {
            let available_ops = self.client.list_operation_ids().await;
            if available_ops.is_empty() {
                // This is fine.
                return Ok("No operations to cancel".to_string());
            }

            // Try to parse as full UUID first
            if let Ok(uuid) = Uuid::parse_str(short_id) {
                self.global_client.cancel_operation(uuid).await.map_err(|e| e.to_string())
            } else {
                // Try to find by short ID (first 8 characters)
                if let Some(full_uuid) = self.global_client.find_operation_by_short_id(short_id).await {
                    self.global_client.cancel_operation(full_uuid).await.map_err(|e| e.to_string())
                } else {
                    Err(format!(
                        "No operation found matching ID: {}\nAvailable operations:\n{}",
                        short_id,
                        available_ops.join("\n")
                    ))
                }
            }
        } else {
            // Cancel most recent operation (not all operations)
            self.global_client
                .cancel_most_recent_operation()
                .await
                .map_err(|e| e.to_string())
        }
    }

    /// Clear contexts from specific scope
    pub async fn clear_scope(&mut self, is_global: bool) -> Result<String, String> {
        if is_global {
            match self.global_client.clear_all_immediate().await {
                Ok(count) => Ok(format!("✅ Successfully cleared {} global knowledge base entries", count)),
                Err(e) => Err(format!("Failed to clear global knowledge: {}", e)),
            }
        } else if let Some(ref mut agent_client) = self.agent_client {
            match agent_client.clear_all_immediate().await {
                Ok(count) => Ok(format!("✅ Successfully cleared {} agent knowledge base entries", count)),
                Err(e) => Err(format!("Failed to clear agent knowledge: {}", e)),
            }
        } else {
            Err("No agent context available".to_string())
        }
    }

    /// Clear all contexts (background operation)
    pub async fn clear(&mut self) -> Result<String, String> {
        match self.global_client.clear_all().await {
            Ok((operation_id, _cancel_token)) => Ok(format!(
                "🚀 Started clearing all contexts in background.\n📊 Use 'knowledge status' to check progress.\n🆔 Operation ID: {}",
                &operation_id.to_string()[..8]
            )),
            Err(e) => Err(format!("Failed to start clear operation: {}", e)),
        }
    }

    /// Remove context by path from specific scope
    pub async fn remove_by_path_scope(&mut self, path: &str, is_global: bool) -> Result<(), String> {
        if is_global {
            if let Some(context) = self.global_client.get_context_by_path(path).await {
                self.global_client
                    .remove_context_by_id(&context.id)
                    .await
                    .map_err(|e| e.to_string())
            } else {
                Err(format!("No context found with path '{}' in global knowledge base", path))
            }
        } else if let Some(ref mut agent_client) = self.agent_client {
            if let Some(context) = agent_client.get_context_by_path(path).await {
                agent_client
                    .remove_context_by_id(&context.id)
                    .await
                    .map_err(|e| e.to_string())
            } else {
                Err(format!("No context found with path '{}' in agent knowledge base", path))
            }
        } else {
            Err("No agent context available".to_string())
        }
    }

    /// Remove context by name from specific scope
    pub async fn remove_by_name_scope(&mut self, name: &str, is_global: bool) -> Result<(), String> {
        if is_global {
            if let Some(context) = self.global_client.get_context_by_name(name).await {
                self.global_client
                    .remove_context_by_id(&context.id)
                    .await
                    .map_err(|e| e.to_string())
            } else {
                Err(format!("No context found with name '{}' in global knowledge base", name))
            }
        } else if let Some(ref mut agent_client) = self.agent_client {
            if let Some(context) = agent_client.get_context_by_name(name).await {
                agent_client
                    .remove_context_by_id(&context.id)
                    .await
                    .map_err(|e| e.to_string())
            } else {
                Err(format!("No context found with name '{}' in agent knowledge base", name))
            }
        } else {
            Err("No agent context available".to_string())
        }
    }

    /// Remove context by path - checks both agent and global clients
    pub async fn remove_by_path(&mut self, path: &str) -> Result<(), String> {
        // Try agent client first if available
        if let Some(ref mut agent_client) = self.agent_client {
            if let Some(context) = agent_client.get_context_by_path(path).await {
                return agent_client
                    .remove_context_by_id(&context.id)
                    .await
                    .map_err(|e| e.to_string());
            }
        }
        
        // Try global client
        if let Some(context) = self.global_client.get_context_by_path(path).await {
            self.global_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err(format!("No context found with path '{}'", path))
        }
    }

    /// Remove context by name - checks both agent and global clients
    pub async fn remove_by_name(&mut self, name: &str) -> Result<(), String> {
        // Try agent client first if available
        if let Some(ref mut agent_client) = self.agent_client {
            if let Some(context) = agent_client.get_context_by_name(name).await {
                return agent_client
                    .remove_context_by_id(&context.id)
                    .await
                    .map_err(|e| e.to_string());
            }
        }
        
        // Try global client
        if let Some(context) = self.global_client.get_context_by_name(name).await {
            self.global_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())
        } else {
            Err(format!("No context found with name '{}'", name))
        }
    }

    /// Remove context by ID - tries both agent and global clients
    pub async fn remove_by_id(&mut self, context_id: &str) -> Result<(), String> {
        // Try agent client first if available
        if let Some(ref mut agent_client) = self.agent_client {
            if agent_client.remove_context_by_id(context_id).await.is_ok() {
                return Ok(());
            }
        }
        
        // Try global client
        self.global_client
            .remove_context_by_id(context_id)
            .await
            .map_err(|e| e.to_string())
    }

    /// Update context by path - checks both agent and global clients
    pub async fn update_by_path(&mut self, path_str: &str) -> Result<String, String> {
        // Try agent client first if available
        if let Some(ref mut agent_client) = self.agent_client {
            if let Some(context) = agent_client.get_context_by_path(path_str).await {
                // Remove the existing context first
                agent_client
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
                return self.add_with_scope(&context.name, path_str, options, false).await;
            }
        }
        
        // Try global client
        if let Some(context) = self.global_client.get_context_by_path(path_str).await {
            // Remove the existing context first
            self.global_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())?;

            // Then add it back with the same name and original patterns (global scope)
            let options = AddOptions {
                description: None,
                include_patterns: context.include_patterns.clone(),
                exclude_patterns: context.exclude_patterns.clone(),
                embedding_type: None,
            };
            self.add_with_scope(&context.name, path_str, options, true).await
        } else {
            Err(format!("No context found with path '{}'", path_str))
        }
    }

    /// Update context by ID - finds context in both clients and preserves scope
    pub async fn update_context_by_id(&mut self, context_id: &str, path_str: &str) -> Result<String, String> {
        // Try agent client first if available
        if let Some(ref mut agent_client) = self.agent_client {
            let agent_contexts = agent_client.get_contexts().await;
            if let Some(context) = agent_contexts.iter().find(|c| c.id == context_id) {
                // Remove from agent client
                agent_client
                    .remove_context_by_id(context_id)
                    .await
                    .map_err(|e| e.to_string())?;

                // Add back to agent scope
                let options = AddOptions {
                    description: None,
                    include_patterns: context.include_patterns.clone(),
                    exclude_patterns: context.exclude_patterns.clone(),
                    embedding_type: None,
                };
                return self.add_with_scope(&context.name, path_str, options, false).await;
            }
        }
        
        // Try global client
        let global_contexts = self.global_client.get_contexts().await;
        if let Some(context) = global_contexts.iter().find(|c| c.id == context_id) {
            // Remove from global client
            self.global_client
                .remove_context_by_id(context_id)
                .await
                .map_err(|e| e.to_string())?;

            // Add back to global scope
            let options = AddOptions {
                description: None,
                include_patterns: context.include_patterns.clone(),
                exclude_patterns: context.exclude_patterns.clone(),
                embedding_type: None,
            };
            self.add_with_scope(&context.name, path_str, options, true).await
        } else {
            Err(format!("Context '{}' not found", context_id))
        }
    }

    /// Update context by name - checks both agent and global clients
    pub async fn update_context_by_name(&mut self, name: &str, path_str: &str) -> Result<String, String> {
        // Try agent client first if available
        if let Some(ref mut agent_client) = self.agent_client {
            if let Some(context) = agent_client.get_context_by_name(name).await {
                // Remove the existing context first
                agent_client
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
                return self.add_with_scope(name, path_str, options, false).await;
            }
        }
        
        // Try global client
        if let Some(context) = self.global_client.get_context_by_name(name).await {
            // Remove the existing context first
            self.global_client
                .remove_context_by_id(&context.id)
                .await
                .map_err(|e| e.to_string())?;

            // Then add it back with the same name and original patterns (global scope)
            let options = AddOptions {
                description: None,
                include_patterns: context.include_patterns.clone(),
                exclude_patterns: context.exclude_patterns.clone(),
                embedding_type: None,
            };
            self.add_with_scope(name, path_str, options, true).await
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
