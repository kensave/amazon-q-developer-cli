use std::collections::HashMap;
use std::path::{
    Path,
    PathBuf,
};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{
    Mutex,
    RwLock,
};
use tracing::warn;

use super::{
    BM25Context,
    SemanticContext,
};
use crate::client::utils;
use crate::embedding::{
    EmbeddingType,
    TextEmbedderTrait,
};
use crate::error::{
    Result,
    SemanticSearchError,
};
use crate::types::*;

type VolatileContexts = Arc<RwLock<HashMap<ContextId, Arc<Mutex<SemanticContext>>>>>;
type BM25Contexts = Arc<RwLock<HashMap<ContextId, Arc<Mutex<BM25Context>>>>>;

const SEMANTIC_DATA_FILE: &str = "data.json";
const BM25_DATA_FILE: &str = "data.bm25.json";
const DEFAULT_BM25_SCORE: f64 = 100.0;

#[derive(Clone)]
/// Context manager for handling contexts
pub struct ContextManager {
    contexts: Arc<RwLock<HashMap<ContextId, KnowledgeContext>>>,
    volatile_contexts: VolatileContexts,
    bm25_contexts: BM25Contexts,
    base_dir: PathBuf,
}

impl ContextManager {
    /// Create new context manager
    pub async fn new(base_dir: &Path) -> Result<Self> {
        let contexts_file = base_dir.join("contexts.json");
        let persistent_contexts: HashMap<ContextId, KnowledgeContext> = utils::load_json_from_file(&contexts_file)?;

        Ok(Self {
            contexts: Arc::new(RwLock::new(persistent_contexts)),
            volatile_contexts: Arc::new(RwLock::new(HashMap::new())),
            bm25_contexts: Arc::new(RwLock::new(HashMap::new())),
            base_dir: base_dir.to_path_buf(),
        })
    }

    /// Get all contexts
    pub async fn get_contexts(&self) -> Vec<KnowledgeContext> {
        match tokio::time::timeout(Duration::from_secs(2), self.contexts.read()).await {
            Ok(contexts_guard) => contexts_guard.values().cloned().collect(),
            Err(_) => {
                if let Ok(contexts_guard) = self.contexts.try_read() {
                    contexts_guard.values().cloned().collect()
                } else {
                    warn!("Could not access contexts - heavy indexing in progress");
                    Vec::new()
                }
            },
        }
    }

    /// Search all contexts
    pub async fn search_all(
        &self,
        query_text: &str,
        effective_limit: usize,
        embedder: &dyn TextEmbedderTrait,
    ) -> Result<Vec<(ContextId, SearchResults)>> {
        let mut all_results = Vec::new();
        let contexts_metadata = self.contexts.read().await;

        for (context_id, context_meta) in contexts_metadata.iter() {
            if context_meta.embedding_type.is_bm25() {
                if let Some(results) = self.search_bm25_context(context_id, query_text, effective_limit).await {
                    all_results.push((context_id.clone(), results));
                }
            } else if let Some(results) = self
                .search_semantic_context(context_id, query_text, effective_limit, embedder)
                .await?
            {
                all_results.push((context_id.clone(), results));
            }
        }

        all_results.sort_by(|(_, a), (_, b)| {
            if a.is_empty() || b.is_empty() {
                return std::cmp::Ordering::Equal;
            }
            a[0].distance
                .partial_cmp(&b[0].distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(all_results)
    }

    /// Search in a specific context
    pub async fn search_context(
        &self,
        context_id: &str,
        query_text: &str,
        effective_limit: usize,
        embedder: &dyn TextEmbedderTrait,
    ) -> Result<Option<SearchResults>> {
        let contexts_metadata = self.contexts.read().await;
        let context_meta = contexts_metadata
            .get(context_id)
            .ok_or_else(|| SemanticSearchError::ContextNotFound(context_id.to_string()))?;

        if context_meta.embedding_type.is_bm25() {
            Ok(self.search_bm25_context(context_id, query_text, effective_limit).await)
        } else {
            self.search_semantic_context(context_id, query_text, effective_limit, embedder)
                .await
        }
    }

    async fn search_bm25_context(&self, context_id: &str, query_text: &str, limit: usize) -> Option<SearchResults> {
        let bm25_contexts = tokio::time::timeout(Duration::from_millis(100), self.bm25_contexts.read())
            .await
            .ok()?;
            
        let context_arc = bm25_contexts.get(context_id)?;
        let context = context_arc.try_lock().ok()?;

        let search_results = context.search(query_text, limit);
        let results: Vec<SearchResult> = search_results
            .into_iter()
            .filter_map(|(id, score)| {
                if let Some(data_point) = context.get_data_points().get(id) {
                    let vector = vec![0.0; 384];
                    let point = DataPoint {
                        id: data_point.id,
                        vector,
                        payload: data_point.payload.clone(),
                    };
                    Some(SearchResult::new(point, score))
                } else {
                    None
                }
            })
            .collect();

        if results.is_empty() { None } else { Some(results) }
    }

    async fn search_semantic_context(
        &self,
        context_id: &str,
        query_text: &str,
        limit: usize,
        embedder: &dyn TextEmbedderTrait,
    ) -> Result<Option<SearchResults>> {
        let query_vector = embedder.embed(query_text)?;
        let volatile_contexts = tokio::time::timeout(Duration::from_millis(100), self.volatile_contexts.read())
            .await
            .map_err(|_timeout| SemanticSearchError::OperationFailed("Timeout accessing contexts".to_string()))?;

        if let Some(context_arc) = volatile_contexts.get(context_id) {
            if let Ok(context_guard) = context_arc.try_lock() {
                match context_guard.search(&query_vector, limit) {
                    Ok(results) => Ok(if results.is_empty() { None } else { Some(results) }),
                    Err(e) => {
                        warn!("Failed to search context {}: {}", context_id, e);
                        Ok(None)
                    },
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Check if path exists or is being indexed
    pub async fn check_path_exists(
        &self,
        canonical_path: &Path,
        operation_manager: &crate::client::operation::OperationManager,
    ) -> Result<()> {
        // First check if there's already an ACTIVE indexing operation for this exact path
        if let Ok(operations) = operation_manager.get_active_operations().try_read() {
            for handle in operations.values() {
                if let crate::types::OperationType::Indexing { path, name } = &handle.operation_type {
                    if let Ok(operation_canonical) = PathBuf::from(path).canonicalize() {
                        if operation_canonical == *canonical_path {
                            if let Ok(progress) = handle.progress.try_lock() {
                                // Only block if the operation is truly active (not cancelled, failed, or completed)
                                let is_cancelled = progress.message.contains("cancelled");
                                let is_failed =
                                    progress.message.contains("failed") || progress.message.contains("error");
                                let is_completed = progress.message.contains("complete");

                                if !is_cancelled && !is_failed && !is_completed {
                                    return Err(crate::error::SemanticSearchError::InvalidArgument(format!(
                                        "Already indexing this path: {} (Operation: {})",
                                        path, name
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Then check if path already exists in knowledge base contexts
        if let Ok(contexts_guard) = self.contexts.try_read() {
            for context in contexts_guard.values() {
                if let Some(existing_path) = &context.source_path {
                    let existing_path_buf = PathBuf::from(existing_path);
                    if let Ok(existing_canonical) = existing_path_buf.canonicalize() {
                        if existing_canonical == *canonical_path {
                            return Err(crate::error::SemanticSearchError::InvalidArgument(format!(
                                "Path already exists in knowledge base: {} (Context: '{}')",
                                existing_path, context.name
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Load persistent contexts
    pub async fn load_persistent_contexts(&self) -> Result<()> {
        let context_ids: Vec<String> = {
            let contexts = self.contexts.read().await;
            contexts.keys().cloned().collect()
        };

        for id in context_ids {
            if let Err(e) = self.load_persistent_context(&id).await {
                tracing::error!("Failed to load persistent context {}: {}", id, e);
            }
        }

        Ok(())
    }

    async fn load_persistent_context(&self, context_id: &str) -> Result<()> {
        let embedding_type = self.get_context_embedding_type(context_id).await;
        let Some(embedding_type) = embedding_type else {
            return Ok(());
        };

        let context_dir = self.base_dir.join(context_id);
        if !context_dir.exists() {
            return Ok(());
        }

        if embedding_type.is_bm25() {
            self.load_bm25_context(context_id, &context_dir).await
        } else {
            self.load_semantic_context(context_id, &context_dir).await
        }
    }
    
    /// Simple token counting based on character count (similar to chat-cli)
    fn count_tokens(content: &str) -> usize {
        const TOKEN_TO_CHAR_RATIO: usize = 4; // Approximate ratio
        (content.len() / TOKEN_TO_CHAR_RATIO + 5) / 10 * 10
    }

    async fn update_context_counts(&self, context_id: &str, added_items: usize, added_tokens: usize) {
        let mut contexts = self.contexts.write().await;
        if let Some(context) = contexts.get_mut(context_id) {
            context.item_count += added_items;
            context.token_count += added_tokens;
            context.updated_at = chrono::Utc::now();
        }
    }

    async fn get_context_embedding_type(&self, context_id: &str) -> Option<EmbeddingType> {
        let contexts = self.contexts.read().await;
        contexts.get(context_id).map(|ctx| ctx.embedding_type)
    }

    async fn load_bm25_context(&self, context_id: &str, context_dir: &Path) -> Result<()> {
        // Check if already loaded
        {
            let bm25_contexts = self.bm25_contexts.read().await;
            if bm25_contexts.contains_key(context_id) {
                return Ok(());
            }
        }

        let data_file = context_dir.join(BM25_DATA_FILE);
        let bm25_context = BM25Context::new(data_file, DEFAULT_BM25_SCORE)?;
        
        // Update context metadata with actual counts from loaded data
        let actual_item_count = bm25_context.get_data_points().len();
        let actual_token_count: usize = bm25_context.get_data_points()
            .iter()
            .map(|dp| Self::count_tokens(&dp.content))
            .sum();
            
        // Update the context metadata with actual counts
        {
            let mut contexts = self.contexts.write().await;
            if let Some(context) = contexts.get_mut(context_id) {
                context.item_count = actual_item_count;
                context.token_count = actual_token_count;
            }
        }

        let mut bm25_contexts = self.bm25_contexts.write().await;
        bm25_contexts.insert(context_id.to_string(), Arc::new(Mutex::new(bm25_context)));
        Ok(())
    }

    async fn load_semantic_context(&self, context_id: &str, context_dir: &Path) -> Result<()> {
        // Check if already loaded
        {
            let volatile_contexts = self.volatile_contexts.read().await;
            if volatile_contexts.contains_key(context_id) {
                return Ok(());
            }
        }

        let data_file = context_dir.join(SEMANTIC_DATA_FILE);
        let semantic_context = SemanticContext::new(data_file)?;
        
        // Update context metadata with actual counts from loaded data
        let actual_item_count = semantic_context.get_data_points().len();
        let actual_token_count: usize = semantic_context.get_data_points()
            .iter()
            .map(|dp| {
                dp.payload.get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| Self::count_tokens(s))
                    .unwrap_or(0)
            })
            .sum();
            
        // Update the context metadata with actual counts
        {
            let mut contexts = self.contexts.write().await;
            if let Some(context) = contexts.get_mut(context_id) {
                context.item_count = actual_item_count;
                context.token_count = actual_token_count;
            }
        }

        let mut volatile_contexts = self.volatile_contexts.write().await;
        volatile_contexts.insert(context_id.to_string(), Arc::new(Mutex::new(semantic_context)));
        Ok(())
    }

    /// Clear all contexts immediately
    pub async fn clear_all_immediate(&self, base_dir: &Path) -> Result<usize> {
        let context_count = {
            let contexts = self.contexts.read().await;
            contexts.len()
        };

        {
            let mut contexts = self.contexts.write().await;
            contexts.clear();
        }

        {
            let mut volatile_contexts = self.volatile_contexts.write().await;
            volatile_contexts.clear();
        }

        if base_dir.exists() {
            std::fs::remove_dir_all(base_dir).map_err(SemanticSearchError::IoError)?;
            std::fs::create_dir_all(base_dir).map_err(SemanticSearchError::IoError)?;
        }

        Ok(context_count)
    }

    /// Remove context by ID
    pub async fn remove_context_by_id(&self, context_id: &str, base_dir: &Path) -> Result<()> {
        {
            let mut contexts = self.contexts.write().await;
            contexts.remove(context_id);
        }

        {
            let mut volatile_contexts = self.volatile_contexts.write().await;
            volatile_contexts.remove(context_id);
        }

        let context_dir = base_dir.join(context_id);
        if context_dir.exists() {
            tokio::fs::remove_dir_all(&context_dir).await.map_err(|e| {
                SemanticSearchError::OperationFailed(format!("Failed to remove context directory: {}", e))
            })?;
        }

        self.save_contexts_metadata(base_dir).await?;
        Ok(())
    }

    /// Get context by path
    pub async fn get_context_by_path(&self, path: &str) -> Option<KnowledgeContext> {
        let contexts = self.contexts.read().await;
        let canonical_input = PathBuf::from(path).canonicalize().ok();

        contexts
            .values()
            .find(|c| {
                if let Some(source_path) = &c.source_path {
                    if source_path == path {
                        return true;
                    }

                    if let Some(ref canonical_input) = canonical_input {
                        if let Ok(canonical_source) = PathBuf::from(source_path).canonicalize() {
                            return canonical_input == &canonical_source;
                        }
                    }

                    let normalized_source = source_path.replace('\\', "/");
                    let normalized_input = path.replace('\\', "/");
                    normalized_source == normalized_input
                } else {
                    false
                }
            })
            .cloned()
    }

    /// Get context by name
    pub async fn get_context_by_name(&self, name: &str) -> Option<KnowledgeContext> {
        let contexts = self.contexts.read().await;
        contexts.values().find(|c| c.name == name).cloned()
    }

    /// List context paths
    pub async fn list_context_paths(&self) -> Vec<String> {
        let contexts = self.contexts.read().await;
        contexts
            .values()
            .map(|c| format!("{} -> {}", c.name, c.source_path.as_deref().unwrap_or("None")))
            .collect()
    }

    /// Save contexts metadata
    pub async fn save_contexts_metadata(&self, base_dir: &Path) -> Result<()> {
        let contexts = self.contexts.read().await;
        let contexts_file = base_dir.join("contexts.json");

        let persistent_contexts: HashMap<String, KnowledgeContext> = contexts
            .iter()
            .filter(|(_, ctx)| ctx.persistent)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        utils::save_json_to_file(&contexts_file, &persistent_contexts)
            .map_err(|e| SemanticSearchError::OperationFailed(format!("Failed to save contexts metadata: {}", e)))
    }

    /// Get contexts reference
    pub fn get_contexts_ref(&self) -> &Arc<RwLock<HashMap<ContextId, KnowledgeContext>>> {
        &self.contexts
    }

    /// Get volatile contexts reference
    pub fn get_volatile_contexts_ref(&self) -> &VolatileContexts {
        &self.volatile_contexts
    }

    /// Get BM25 contexts reference
    pub fn get_bm25_contexts_ref(&self) -> &BM25Contexts {
        &self.bm25_contexts
    }

    /// Add text chunks to an existing context
    pub async fn add_text_chunks_to_context(&self, context_id: &str, chunks: Vec<String>, embedder: &dyn TextEmbedderTrait) -> std::result::Result<(), String> {
        // Find the context
        let contexts = self.get_contexts().await;
        let _context = contexts.iter().find(|c| c.id == context_id)
            .ok_or_else(|| format!("Context '{}' not found", context_id))?;

        // Check if it's semantic or BM25 context
        let embedding_type = self.get_context_embedding_type(&context_id).await;
        
        match embedding_type {
            Some(crate::embedding::EmbeddingType::Best) => {
                // Semantic context - generate embeddings and add
                self.add_chunks_to_semantic_context(context_id, chunks, embedding_type.unwrap(), embedder).await
            },
            Some(crate::embedding::EmbeddingType::Fast) | None => {
                // BM25 context - add directly
                self.add_chunks_to_bm25_context(context_id, chunks).await
            }
        }
    }

    async fn add_chunks_to_bm25_context(&self, context_id: &str, chunks: Vec<String>) -> std::result::Result<(), String> {
        if let Ok(bm25_contexts) = self.bm25_contexts.try_read() {
            if let Some(context_mutex) = bm25_contexts.get(context_id) {
                if let Ok(mut context) = context_mutex.try_lock() {
                    // Convert chunks to BM25DataPoints
                    let mut data_points = Vec::new();
                    let base_id = context.get_data_points().len(); // Start from current count to avoid ID conflicts
                    
                    for (i, chunk) in chunks.iter().enumerate() {
                        let mut metadata = std::collections::HashMap::new();
                        metadata.insert("timestamp".to_string(), serde_json::Value::String(chrono::Utc::now().to_rfc3339()));
                        metadata.insert("text".to_string(), serde_json::Value::String(chunk.clone())); // Add text field for compatibility
                        
                        let data_point_id = base_id + i;
                        
                        data_points.push(crate::types::BM25DataPoint {
                            id: data_point_id, // Use unique ID based on existing count
                            payload: metadata,
                            content: chunk.clone(),
                        });
                    }
                    
                    // Add to context
                    let data_points_len = data_points.len();
                    let total_tokens: usize = data_points.iter().map(|dp| Self::count_tokens(&dp.content)).sum();
                    
                    context.add_data_points(data_points)
                        .map_err(|e| format!("Failed to add data points: {}", e))?;
                    
                    // Save context to ensure persistence
                    if let Err(e) = context.save() {
                        eprintln!("Warning: Failed to save BM25 context: {}", e);
                    }
                    
                    // Update knowledge context item and token counts
                    self.update_context_counts(context_id, data_points_len, total_tokens).await;
                    
                    return Ok(());
                } else {
                    return Err(format!("BM25 context '{}' is locked", context_id));
                }
            }
        }
        Err(format!("BM25 context '{}' not found", context_id))
    }

    async fn add_chunks_to_semantic_context(
        &self, 
        context_id: &str, 
        chunks: Vec<String>, 
        _embedding_type: crate::embedding::EmbeddingType,
        embedder: &dyn TextEmbedderTrait
    ) -> std::result::Result<(), String> {
        let volatile_contexts = self.volatile_contexts.read().await;
        if let Some(context_mutex) = volatile_contexts.get(context_id) {
            if let Ok(mut context) = context_mutex.try_lock() {
                // Convert chunks to DataPoints with real embeddings
                let mut data_points = Vec::new();
                let base_id = context.get_data_points().len(); // Start from current count to avoid ID conflicts
                
                for (i, chunk) in chunks.iter().enumerate() {
                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert("timestamp".to_string(), serde_json::Value::String(chrono::Utc::now().to_rfc3339()));
                    metadata.insert("content".to_string(), serde_json::Value::String(chunk.clone()));
                    metadata.insert("text".to_string(), serde_json::Value::String(chunk.clone())); // Add text field for compatibility
                    
                    // Generate actual embeddings using the embedder
                    let vector = embedder.embed(chunk)
                        .map_err(|e| format!("Failed to generate embedding: {}", e))?;
                    
                    // Validate embedding dimension to prevent panic
                    if vector.is_empty() {
                        return Err(format!("Embedder returned empty vector for chunk: {}", chunk));
                    }
                    
                    let data_point_id = base_id + i;
                    data_points.push(crate::types::DataPoint {
                        id: data_point_id, // Use unique ID based on existing count
                        payload: metadata,
                        vector,
                    });
                }
                
                // Add to semantic context
                let data_points_len = data_points.len();
                let total_tokens: usize = data_points.iter().map(|dp| {
                    dp.payload.get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| Self::count_tokens(s))
                        .unwrap_or(0)
                }).sum();
                
                context.add_data_points(data_points)
                    .map_err(|e| format!("Failed to add data points: {}", e))?;
                
                // Save context to ensure persistence
                if let Err(e) = context.save() {
                    eprintln!("Warning: Failed to save semantic context: {}", e);
                }
                
                // Update knowledge context item and token counts
                self.update_context_counts(context_id, data_points_len, total_tokens).await;
                
                return Ok(());
            } else {
                return Err(format!("Semantic context '{}' is locked", context_id));
            }
        }
        Err(format!("Semantic context '{}' not found", context_id))
    }
}
