use clap::Subcommand;
use eyre::Result;
use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::cli::chat::cli::resource::StorageType;
use crate::os::Os;
use super::types::ResourceOperation;
use super::managers::{ContextResourceManager, KnowledgeResourceManager};
use super::manager::ResourceManager;

/// Unified resource management system
#[derive(Debug, PartialEq, Subcommand)]
pub enum ResourceNewCommand {
    /// Add content to resources
    Add {
        /// Name for the resource entry
        name: String,
        /// Content or path to add
        value: String,
        /// Storage type: "pinned" (session-scoped) or "indexed" (persistent, searchable)
        #[arg(long, default_value = "indexed")]
        r#type: StorageType,
        /// Include patterns (e.g., `**/*.ts`, `**/*.md`) - only for indexed storage
        #[arg(long, action = clap::ArgAction::Append)]
        include: Vec<String>,
        /// Exclude patterns (e.g., `node_modules/**`, `target/**`) - only for indexed storage
        #[arg(long, action = clap::ArgAction::Append)]
        exclude: Vec<String>,
        /// Index type to use (Fast, Best) - only for indexed storage
        #[arg(long)]
        index_type: Option<String>,
    },
    /// Remove content from resources
    Remove {
        /// ID of the entry to remove
        #[arg(long)]
        id: Option<String>,
        /// Name of the entry to remove
        #[arg(long)]
        name: Option<String>,
        /// Path of the entry to remove
        #[arg(long)]
        path: Option<String>,
        /// Storage type to remove from: "pinned", "indexed", or "all"
        #[arg(long, default_value = "all")]
        r#type: StorageType,
    },
    /// Show all resources
    Show {
        /// Show detailed content of each resource
        #[arg(long)]
        expand: bool,
        /// Storage type to show: "pinned", "indexed", or "all"
        #[arg(long, default_value = "all")]
        r#type: StorageType,
    },
    /// Update a resource by path (re-index for indexed storage)
    Update {
        /// Path to update
        path: String,
        /// Storage type: "pinned" or "indexed"
        #[arg(long, default_value = "indexed")]
        r#type: StorageType,
    },
    /// Clear resources
    Clear {
        /// Confirm the clear operation
        #[arg(long)]
        confirm: bool,
        /// Storage type to clear: "pinned", "indexed", or "all"
        #[arg(long, default_value = "all")]
        r#type: StorageType,
    },
    /// Cancel background operation
    Cancel {
        /// Operation ID to cancel
        operation_id: String,
    },
}

impl ResourceNewCommand {
    pub async fn execute(self, os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        match self {
            Self::Add { name, value, r#type, include, exclude, index_type } => {
                handle_add(os, session, name, value, r#type, include, exclude, index_type).await
            }
            Self::Remove { id, name, path, r#type } => {
                handle_remove(os, session, id, name, path, r#type).await
            }
            Self::Show { expand, r#type } => {
                handle_show(os, session, expand, r#type).await
            }
            Self::Update { path, r#type } => {
                handle_update(os, session, path, r#type).await
            }
            Self::Clear { confirm, r#type } => {
                handle_clear(os, session, confirm, r#type).await
            }
            Self::Cancel { operation_id } => {
                handle_cancel(os, session, operation_id).await
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Add { .. } => "add",
            Self::Remove { .. } => "remove",
            Self::Show { .. } => "show",
            Self::Update { .. } => "update",
            Self::Clear { .. } => "clear",
            Self::Cancel { .. } => "cancel",
        }
    }
}

// ============================================================================
// Command Handlers
// ============================================================================

async fn handle_add(
    os: &Os, 
    session: &mut ChatSession, 
    name: String, 
    value: String, 
    storage_type: StorageType,
    include: Vec<String>,
    exclude: Vec<String>,
    index_type: Option<String>
) -> Result<ChatState, ChatError> {
    validate_add_inputs(session, &storage_type, &include, &exclude, &index_type)?;
    
    let operation = ResourceOperation::Add { 
        name, 
        value, 
        include_patterns: (!include.is_empty()).then_some(include),
        exclude_patterns: (!exclude.is_empty()).then_some(exclude),
        index_type 
    };
    
    let data = execute_operation(os, session, operation, storage_type).await?;
    render_to_session(&data, session)?;
    
    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_update(os: &Os, session: &mut ChatSession, path: String, storage_type: StorageType) -> Result<ChatState, ChatError> {
    let operation = ResourceOperation::Update { path };
    let data = execute_operation(os, session, operation, storage_type).await?;
    render_to_session(&data, session)?;

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_remove(os: &Os, session: &mut ChatSession, id: Option<String>, name: Option<String>, path: Option<String>, storage_type: StorageType) -> Result<ChatState, ChatError> {
    for st in get_storage_types(storage_type) {
        handle_remove_by_type(os, session, id.clone(), name.clone(), path.clone(), st).await?;
    }

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_show(os: &Os, session: &mut ChatSession, expand: bool, storage_type: StorageType) -> Result<ChatState, ChatError> {
    for st in get_storage_types(storage_type) {
        handle_show_by_type(os, session, expand, st).await?;
    }

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_clear(os: &Os, session: &mut ChatSession, confirm: bool, storage_type: StorageType) -> Result<ChatState, ChatError> {
    for st in get_storage_types(storage_type) {
        handle_clear_by_type(os, session, confirm, st).await?;
    }

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_status(os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
    let operation = ResourceOperation::Status;
    let data = execute_operation(os, session, operation, StorageType::Indexed).await?;
    render_to_session(&data, session)?;

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_cancel(os: &Os, session: &mut ChatSession, operation_id: String) -> Result<ChatState, ChatError> {
    let operation = ResourceOperation::Cancel { operation_id };
    
    match execute_operation(os, session, operation, StorageType::Indexed).await {
        Ok(output) => println!("{:?}", output),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

// ============================================================================
// Helper Functions (by_type handlers)
// ============================================================================

async fn handle_remove_by_type(os: &Os, session: &mut ChatSession, id: Option<String>, name: Option<String>, path: Option<String>, storage_type: StorageType) -> Result<(), ChatError> {
    let operation = ResourceOperation::Remove { id, name, path };
    let data = execute_operation(os, session, operation, storage_type).await?;
    render_to_session(&data, session)?;
    Ok(())
}

async fn handle_show_by_type(os: &Os, session: &mut ChatSession, expand: bool, storage_type: StorageType) -> Result<(), ChatError> {
    let operation = ResourceOperation::Show { expand };
    let data = execute_operation(os, session, operation, storage_type).await?;
    render_to_session(&data, session)?;
    Ok(())
}

async fn handle_clear_by_type(os: &Os, session: &mut ChatSession, confirm: bool, storage_type: StorageType) -> Result<(), ChatError> {
    let operation = ResourceOperation::Clear { confirm };
    let data = execute_operation(os, session, operation, storage_type).await?;
    render_to_session(&data, session)?;
    Ok(())
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert any error to ChatError for consistent error handling
fn to_chat_error(e: impl std::fmt::Display) -> ChatError {
    ChatError::Custom(e.to_string().into())
}

/// Render resource data to session with colors and styling
fn render_to_session(data: &super::types::ResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
    use super::renderer::{CliRenderer, ResourceRenderer};
    let renderer = CliRenderer::new();
    renderer.render_with_session(data, session)
}

/// Get storage types to iterate over - returns both Pinned and Indexed for All, otherwise the single type
fn get_storage_types(storage_type: StorageType) -> Vec<StorageType> {
    match storage_type {
        StorageType::All => vec![StorageType::Pinned, StorageType::Indexed],
        _ => vec![storage_type],
    }
}

/// Create a context manager for pinned resources
fn get_context_manager<'a>(os: &'a Os, session: &'a mut ChatSession) -> Result<ContextResourceManager<'a>, eyre::Report> {
    let context_manager = session.conversation.context_manager.as_mut()
        .ok_or_else(|| eyre::eyre!("Context manager not available"))?;
    Ok(ContextResourceManager::new(context_manager, os))
}

/// Create a knowledge manager for indexed resources
async fn get_knowledge_manager(os: &Os, session: &ChatSession) -> Result<KnowledgeResourceManager, eyre::Report> {
    let agent = session.conversation.agents.get_active();
    KnowledgeResourceManager::new(os, agent).await
}

/// Execute operation on the appropriate manager based on storage type
async fn execute_operation(
    os: &Os, 
    session: &mut ChatSession, 
    operation: ResourceOperation, 
    storage_type: StorageType
) -> Result<super::types::ResourceData, ChatError> {
    match storage_type {
        StorageType::Pinned => {
            let mut manager = get_context_manager(os, session).map_err(to_chat_error)?;
            manager.execute(operation).await.map_err(to_chat_error)
        }
        StorageType::Indexed => {
            let mut manager = get_knowledge_manager(os, session).await.map_err(to_chat_error)?;
            manager.execute(operation).await.map_err(to_chat_error)
        }
        StorageType::All => {
            Err(ChatError::Custom("StorageType::All not supported for single operations".into()))
        }
    }
}

fn validate_add_inputs(
    session: &mut ChatSession,
    storage_type: &StorageType,
    include: &[String],
    exclude: &[String],
    index_type: &Option<String>
) -> Result<(), ChatError> {
    if *storage_type == StorageType::Pinned && (!include.is_empty() || !exclude.is_empty() || index_type.is_some()) {
        let warning_data = super::ResourceData::Success(
            "Warning: include/exclude patterns and index-type are ignored for pinned resources".to_string()
        );
        render_to_session(&warning_data, session)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_names() {
        assert_eq!(ResourceNewCommand::Add {
            name: "test".into(),
            value: "test".into(),
            r#type: StorageType::Indexed,
            include: vec![],
            exclude: vec![],
            index_type: None,
        }.name(), "add");

        assert_eq!(ResourceNewCommand::Show {
            expand: false,
            r#type: StorageType::All,
        }.name(), "show");

        assert_eq!(ResourceNewCommand::Update {
            path: "test".into(),
            r#type: StorageType::Indexed,
        }.name(), "update");
    }
}
