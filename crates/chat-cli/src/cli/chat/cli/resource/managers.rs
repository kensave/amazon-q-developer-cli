use std::sync::Arc;
use tokio::sync::Mutex;
use eyre::Result;
use async_trait::async_trait;

use crate::cli::chat::context::ContextManager;
use crate::cli::chat::token_counter::TokenCounter;
use crate::util::knowledge_store::KnowledgeStore;
use crate::os::Os;

use super::types::{ResourceOperation, ResourceData, ResourceItem, ResourceMetadata, StatusData, OperationInfo, StorageInfo, IndexedResourceData};
use super::manager::ResourceManager;

/// Context resource manager - wraps ContextManager for session-scoped resources
pub struct ContextResourceManager<'a> {
    manager: &'a mut ContextManager,
    os: &'a Os,
}

impl<'a> ContextResourceManager<'a> {
    pub fn new(manager: &'a mut ContextManager, os: &'a Os) -> Self {
        Self { manager, os }
    }
}

#[async_trait]
impl ResourceManager for ContextResourceManager<'_> {
    async fn execute(&mut self, operation: ResourceOperation) -> Result<ResourceData> {
        match operation {
            ResourceOperation::Add { name: _, value } => {
                // Context uses paths - split comma-separated paths
                let paths: Vec<String> = value.split(',').map(|s| s.trim().to_string()).collect();
                self.manager.add_paths(self.os, paths, false).await
                    .map_err(|e| eyre::eyre!("Failed to add to context: {}", e))?;
                Ok(ResourceData::Success("Added to context".to_string()))
            }
            ResourceOperation::Remove { path: Some(path), .. } => {
                let paths = vec![path];
                self.manager.remove_paths(paths)
                    .map_err(|e| eyre::eyre!("Failed to remove from context: {}", e))?;
                Ok(ResourceData::Success("Removed from context".to_string()))
            }
            ResourceOperation::Show { expand: _ } => {
                // Get context files and convert to ResourceItems
                let context_files = self.manager.get_context_files(self.os).await
                    .map_err(|e| eyre::eyre!("Failed to get context files: {}", e))?;
                
                let items: Vec<ResourceItem> = context_files.into_iter().map(|(path, content)| {
                    let token_count = TokenCounter::count_tokens(&content);
                    ResourceItem {
                        id: path.clone(),
                        name: path,
                        content: Some(content),
                        metadata: ResourceMetadata {
                            created_at: chrono::Utc::now(),
                            updated_at: chrono::Utc::now(),
                            size: 0, // Context doesn't track size
                            resource_type: "pinned".to_string(),
                            token_count: Some(token_count),
                        },
                    }
                }).collect();
                
                Ok(ResourceData::IndexedResources(IndexedResourceData { items }))
            }
            ResourceOperation::Clear { .. } => {
                self.manager.clear();
                Ok(ResourceData::Success("Cleared context".to_string()))
            }
            _ => Err(eyre::eyre!("Operation not supported for context resources")),
        }
    }
    
    fn name(&self) -> &'static str {
        "context"
    }
    
    fn supports_operation(&self, operation: &ResourceOperation) -> bool {
        matches!(operation, 
            ResourceOperation::Add { .. } | 
            ResourceOperation::Remove { .. } | 
            ResourceOperation::Show { .. } | 
            ResourceOperation::Clear { .. }
        )
    }
}

/// Knowledge resource manager - wraps KnowledgeStore for persistent indexed resources
pub struct KnowledgeResourceManager {
    store: Arc<Mutex<KnowledgeStore>>,
}

impl KnowledgeResourceManager {
    pub async fn new(os: &Os, agent: Option<&crate::cli::agent::Agent>) -> Result<Self> {
        let store = KnowledgeStore::get_async_instance(os, agent).await
            .map_err(|e| eyre::eyre!("Failed to initialize knowledge store: {}", e))?;
        Ok(Self { store })
    }
}

#[async_trait]
impl ResourceManager for KnowledgeResourceManager {
    async fn execute(&mut self, operation: ResourceOperation) -> Result<ResourceData> {
        let mut store = self.store.lock().await;
        
        match operation {
            ResourceOperation::Add { name, value } => {
                let id = store.add(&name, &value, Default::default()).await
                    .map_err(|e| eyre::eyre!("Failed to add to knowledge: {}", e))?;
                Ok(ResourceData::Success(format!("Added '{}' with ID: {}", name, id)))
            }
            ResourceOperation::Remove { id: Some(id), .. } => {
                store.remove_by_id(&id).await
                    .map_err(|e| eyre::eyre!("Failed to remove by ID: {}", e))?;
                Ok(ResourceData::Success(format!("Removed resource with ID: {}", id)))
            }
            ResourceOperation::Remove { name: Some(name), .. } => {
                store.remove_by_name(&name).await
                    .map_err(|e| eyre::eyre!("Failed to remove by name: {}", e))?;
                Ok(ResourceData::Success(format!("Removed resource: {}", name)))
            }
            ResourceOperation::Search { query, context_id } => {
                let results = store.search(&query, context_id.as_deref()).await
                    .map_err(|e| eyre::eyre!("Search failed: {}", e))?;
                
                if results.is_empty() {
                    Ok(ResourceData::Success(format!("No results found for: {}", query)))
                } else {
                    let mut output = format!("Search results for \"{}\":\n\n", query);
                    for result in results {
                        if let Some(text) = result.text() {
                            output.push_str(&format!("{}\n\n", text));
                        }
                    }
                    Ok(ResourceData::Success(output))
                }
            }
            ResourceOperation::Show { .. } => {
                let contexts = store.get_all().await
                    .map_err(|e| eyre::eyre!("Failed to get knowledge entries: {}", e))?;
                
                let items: Vec<ResourceItem> = contexts.into_iter().map(|ctx| {
                    let token_count = TokenCounter::count_tokens(&ctx.description);
                    ResourceItem {
                        id: ctx.id.clone(),
                        name: ctx.name.clone(),
                        content: Some(ctx.description),
                        metadata: ResourceMetadata {
                            created_at: ctx.created_at,
                            updated_at: ctx.updated_at,
                            size: ctx.item_count,
                            resource_type: "indexed".to_string(),
                            token_count: Some(token_count),
                        },
                    }
                }).collect();
                
                Ok(ResourceData::IndexedResources(IndexedResourceData { items }))
            }
            ResourceOperation::Status => {
                let status = store.get_status_data().await
                    .map_err(|e| eyre::eyre!("Failed to get status: {}", e))?;
                
                let status_data = StatusData {
                    active_operations: status.operations.into_iter().map(|op| {
                        OperationInfo {
                            id: op.short_id,
                            operation_type: op.operation_type.display_name().to_string(),
                            status: if op.is_cancelled { "Cancelled".to_string() }
                                   else if op.is_failed { "Failed".to_string() }
                                   else if op.is_waiting { "Waiting".to_string() }
                                   else { "In Progress".to_string() },
                            progress: if op.total > 0 { 
                                Some(op.current as f32 / op.total as f32 * 100.0) 
                            } else { None },
                        }
                    }).collect(),
                    total_items: status.total_contexts,
                    storage_info: StorageInfo {
                        total_size: 0, // Not tracked by knowledge store
                        item_count: status.total_contexts,
                    },
                };
                
                Ok(ResourceData::Status(status_data))
            }
            ResourceOperation::Clear { .. } => {
                let result = store.clear().await
                    .unwrap_or_else(|e| format!("Failed to clear: {}", e));
                Ok(ResourceData::Success(result))
            }
            ResourceOperation::Cancel { operation_id } => {
                let result = store.cancel_operation(Some(&operation_id)).await
                    .unwrap_or_else(|e| format!("Failed to cancel: {}", e));
                Ok(ResourceData::Success(result))
            }
            _ => Err(eyre::eyre!("Operation not supported for knowledge resources")),
        }
    }
    
    fn name(&self) -> &'static str {
        "knowledge"
    }
    
    fn supports_operation(&self, operation: &ResourceOperation) -> bool {
        matches!(operation, 
            ResourceOperation::Add { .. } | 
            ResourceOperation::Remove { .. } | 
            ResourceOperation::Search { .. } | 
            ResourceOperation::Show { .. } | 
            ResourceOperation::Status | 
            ResourceOperation::Clear { .. } | 
            ResourceOperation::Cancel { .. }
        )
    }
}
