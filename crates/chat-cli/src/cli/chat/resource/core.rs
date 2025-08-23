use eyre::Result;

use crate::cli::chat::context::{ContextManager, ContextFilePath, calc_max_context_files_size};
use crate::cli::chat::token_counter::TokenCounter;
use crate::cli::chat::util::drop_matched_context_files;
use crate::util::knowledge_store::KnowledgeStore;
use crate::os::Os;

use super::types::{ResourceOperation, ResourceData, ResourceItem, ResourceMetadata, StatusData, OperationInfo, StorageInfo, IndexedResourceData, PinnedResourceData, ContextPath, MatchedFile};

/// Core resource operations - shareable business logic
pub struct ResourceCore;

impl ResourceCore {
    /// Invoke pinned resource operations
    pub async fn invoke_pinned(
        operation: ResourceOperation,
        context_manager: &mut ContextManager,
        os: &Os,
    ) -> Result<ResourceData> {
        match operation {
            ResourceOperation::Add { name: _, value, include_patterns: _, exclude_patterns: _, index_type: _ } => {
                let paths: Vec<String> = value.split(',').map(|s| s.trim().to_string()).collect();
                context_manager.add_paths(os, paths, false).await?;
                Ok(ResourceData::Success("Added to context".to_string()))
            }
            ResourceOperation::Remove { path: Some(path), .. } => {
                let paths = vec![path];
                context_manager.remove_paths(paths)?;
                Ok(ResourceData::Success("Removed from context".to_string()))
            }
            ResourceOperation::Show { expand: _ } => {
                Self::get_pinned_resources(context_manager, os).await
            }
            ResourceOperation::Clear { .. } => {
                context_manager.clear();
                Ok(ResourceData::Success("Cleared context".to_string()))
            }
            _ => Err(eyre::eyre!("Operation not supported for pinned resources")),
        }
    }

    /// Invoke indexed resource operations
    pub async fn invoke_indexed(
        operation: ResourceOperation,
        os: &Os,
        agent: Option<&crate::cli::agent::Agent>,
    ) -> Result<ResourceData> {
        let store = KnowledgeStore::get_async_instance(os, agent).await?;
        
        match operation {
            ResourceOperation::Add { name, value, include_patterns, exclude_patterns, index_type } => {
                let options = crate::util::knowledge_store::AddOptions::new()
                    .with_include_patterns(include_patterns.unwrap_or_default())
                    .with_exclude_patterns(exclude_patterns.unwrap_or_default())
                    .with_embedding_type(index_type);
                    
                let mut store_guard = store.lock().await;
                let id = store_guard.add(&name, &value, options).await
                    .map_err(|e| eyre::eyre!("{}", e))?;
                Ok(ResourceData::Success(format!("Added '{}' with ID: {}", name, id)))
            }
            ResourceOperation::Remove { id: Some(id), .. } => {
                let mut store_guard = store.lock().await;
                store_guard.remove_by_id(&id).await
                    .map_err(|e| eyre::eyre!("{}", e))?;
                Ok(ResourceData::Success(format!("Removed resource with ID: {}", id)))
            }
            ResourceOperation::Remove { name: Some(name), .. } => {
                let mut store_guard = store.lock().await;
                store_guard.remove_by_name(&name).await
                    .map_err(|e| eyre::eyre!("{}", e))?;
                Ok(ResourceData::Success(format!("Removed resource: {}", name)))
            }
            ResourceOperation::Update { path } => {
                let mut store_guard = store.lock().await;
                let message = store_guard.update_by_path(&path).await
                    .map_err(|e| eyre::eyre!("{}", e))?;
                Ok(ResourceData::Success(message))
            }
            ResourceOperation::Search { query, context_id } => {
                let store_guard = store.lock().await;
                let results = store_guard.search(&query, context_id.as_deref()).await
                    .map_err(|e| eyre::eyre!("{}", e))?;
                
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
                Self::get_indexed_resources(store).await
            }
            ResourceOperation::Status => {
                Self::get_status(store).await
            }
            ResourceOperation::Clear { .. } => {
                let mut store_guard = store.lock().await;
                let result = store_guard.clear_immediate().await
                    .unwrap_or_else(|e| format!("Failed to clear: {}", e));
                Ok(ResourceData::Success(result))
            }
            ResourceOperation::Cancel { operation_id } => {
                let mut store_guard = store.lock().await;
                let result = store_guard.cancel_operation(Some(&operation_id)).await
                    .unwrap_or_else(|e| format!("Failed to cancel: {}", e));
                Ok(ResourceData::Success(result))
            }
            _ => Err(eyre::eyre!("Operation not supported for indexed resources")),
        }
    }

    async fn get_pinned_resources(context_manager: &mut ContextManager, os: &Os) -> Result<ResourceData> {
        let mut profile_context_files = std::collections::HashSet::<(String, String, bool)>::new();
        let (agent_owned_list, session_owned_list) = context_manager
            .paths
            .iter()
            .partition::<Vec<_>, _>(|p| matches!(**p, ContextFilePath::Agent(_)));
        
        let mut agent_files = Vec::new();
        for path in &agent_owned_list {
            let context_files = context_manager
                .get_context_files_by_path(os, path.get_path_as_str())
                .await
                .unwrap_or_default();

            agent_files.push(ContextPath {
                path: path.get_path_as_str().to_string(),
                match_count: context_files.len(),
            });

            if !context_files.is_empty() {
                profile_context_files
                    .extend(context_files.into_iter().map(|(path, content)| (path, content, false)));
            }
        }
        
        let mut session_files = Vec::new();
        for path in &session_owned_list {
            let context_files = context_manager
                .get_context_files_by_path(os, path.get_path_as_str())
                .await
                .unwrap_or_default();

            session_files.push(ContextPath {
                path: path.get_path_as_str().to_string(),
                match_count: context_files.len(),
            });

            if !context_files.is_empty() {
                profile_context_files
                    .extend(context_files.into_iter().map(|(path, content)| (path, content, true)));
            }
        }
        
        let matched_files: Vec<MatchedFile> = profile_context_files.into_iter().map(|(path, content, is_temporary)| {
            let tokens = TokenCounter::count_tokens(&content);
            MatchedFile {
                filename: path,
                content,
                tokens,
                is_temporary,
            }
        }).collect();
        
        let total_tokens = matched_files.iter().map(|f| f.tokens).sum();
        let context_files_max_size = calc_max_context_files_size(None);
        let mut files_as_vec = matched_files
            .iter()
            .map(|f| (f.filename.clone(), f.content.clone()))
            .collect::<Vec<_>>();
        let dropped_files = drop_matched_context_files(&mut files_as_vec, context_files_max_size).ok();
        
        let pinned_data = PinnedResourceData {
            agent_files,
            session_files,
            matched_files,
            total_tokens,
            dropped_files,
            context_files_max_size,
        };
        
        Ok(ResourceData::PinnedResources(pinned_data))
    }

    async fn get_indexed_resources(store: std::sync::Arc<tokio::sync::Mutex<KnowledgeStore>>) -> Result<ResourceData> {
        let store_guard = store.lock().await;
        let contexts = store_guard.get_all().await
            .map_err(|e| eyre::eyre!("{}", e))?;
        let status = store_guard.get_status_data().await
            .map_err(|e| eyre::eyre!("{}", e))?;
        
        let mut items: Vec<ResourceItem> = contexts.into_iter().map(|ctx| {
            ResourceItem {
                id: ctx.id.clone(),
                name: format!("{} ({:?})", ctx.name, ctx.embedding_type),
                content: Some(ctx.description),
                metadata: ResourceMetadata {
                    created_at: ctx.created_at,
                    updated_at: ctx.updated_at,
                    size: ctx.item_count,
                    resource_type: "indexed".to_string(),
                    token_count: None,
                },
            }
        }).collect();
        
        // Add in-progress operations
        for op in status.operations {
            if !op.is_cancelled {
                let eta = if op.total > 0 && op.current > 0 {
                    let remaining = op.total - op.current;
                    let elapsed = op.started_at.elapsed().unwrap_or_default().as_secs();
                    let eta_secs = (elapsed * remaining) / op.current;
                    format!(" (ETA: {}s)", eta_secs)
                } else { String::new() };
                
                items.push(ResourceItem {
                    id: op.id.clone(),
                    name: format!("{} - indexing{}", op.message, eta),
                    content: None,
                    metadata: ResourceMetadata {
                        created_at: op.started_at.into(),
                        updated_at: op.started_at.into(),
                        size: op.current as usize,
                        resource_type: "indexing".to_string(),
                        token_count: None,
                    },
                });
            }
        }
        
        Ok(ResourceData::IndexedResources(IndexedResourceData { items }))
    }

    async fn get_status(store: std::sync::Arc<tokio::sync::Mutex<KnowledgeStore>>) -> Result<ResourceData> {
        let store_guard = store.lock().await;
        let status = store_guard.get_status_data().await
            .map_err(|e| eyre::eyre!("{}", e))?;
        
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
                total_size: 0,
                item_count: status.total_contexts,
            },
        };
        
        Ok(ResourceData::Status(status_data))
    }
}
