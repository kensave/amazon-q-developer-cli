use clap::Subcommand;
use eyre::Result;
use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::cli::chat::cli::resource::{ResourceData, StorageType};
use crate::os::Os;
use super::types::ResourceOperation;
use super::managers::{ContextResourceManager, KnowledgeResourceManager};
use super::renderer::{CliRenderer, ResourceRenderer};
use super::manager::ResourceHandler;

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
}

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
    
    let mut handler = get_handler(os, session, storage_type).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    
    let operation = ResourceOperation::Add { 
        name, 
        value, 
        include_patterns: (!include.is_empty()).then_some(include),
        exclude_patterns: (!exclude.is_empty()).then_some(exclude),
        index_type 
    };
    
    let data = handler.handle(operation).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    
    let renderer = CliRenderer::new();
    renderer.render_with_session(&data, session)?;
    
    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_update(os: &Os, session: &mut ChatSession, path: String, storage_type: StorageType) -> Result<ChatState, ChatError> {
    let mut handler = get_handler(os, session, storage_type).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    let operation = ResourceOperation::Update { path };

    let data = handler.handle(operation).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;

    let renderer = CliRenderer::new();
    renderer.render_with_session(&data, session)?;

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}
    session: &mut ChatSession,
    storage_type: &StorageType,
    include: &[String],
    exclude: &[String],
    index_type: &Option<String>
) -> Result<(), ChatError> {
    if *storage_type == StorageType::Pinned && (!include.is_empty() || !exclude.is_empty() || index_type.is_some()) {
        let renderer = CliRenderer::new();
        let warning_data = super::ResourceData::Success(
            "Warning: include/exclude patterns and index-type are ignored for pinned resources".to_string()
        );
        renderer.render_with_session(&warning_data, session)?;
    }
    Ok(())
}

async fn handle_remove(os: &Os, session: &mut ChatSession, id: Option<String>, name: Option<String>, path: Option<String>, storage_type: StorageType) -> Result<ChatState, ChatError> {
    match storage_type {
        StorageType::All => {
            handle_remove_by_type(os, session, id.clone(), name.clone(), path.clone(), StorageType::Pinned).await?;
            handle_remove_by_type(os, session, id, name, path, StorageType::Indexed).await?;
        }
        _ => {
            handle_remove_by_type(os, session, id, name, path, storage_type).await?;
        }
    }

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_remove_by_type(os: &Os, session: &mut ChatSession, id: Option<String>, name: Option<String>, path: Option<String>, storage_type: StorageType) -> Result<(), ChatError> {
    let mut handler = get_handler(os, session, storage_type).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    let operation = ResourceOperation::Remove { id, name, path };

    let data = handler.handle(operation).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;

    let renderer = CliRenderer::new();
    renderer.render_with_session(&data, session)?;
    Ok(())
}

async fn handle_show(os: &Os, session: &mut ChatSession, expand: bool, storage_type: StorageType) -> Result<ChatState, ChatError> {
    match storage_type {
        StorageType::All => {
            handle_show_by_type(os, session, expand, StorageType::Pinned).await?;
            handle_show_by_type(os, session, expand, StorageType::Indexed).await?;
        }
        _ => {
            handle_show_by_type(os, session, expand, storage_type).await?;
        }
    }

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_show_by_type(os: &Os, session: &mut ChatSession, expand: bool, storage_type: StorageType) -> Result<(), ChatError> {
    let mut handler = get_handler(os, session, storage_type).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    let operation = ResourceOperation::Show { expand };
    
    let data = handler.handle(operation).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    
    let renderer = CliRenderer::new();
    renderer.render_with_session(&data, session)?;
    
    Ok(())
}

async fn handle_status(os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
    let mut handler = get_handler(os, session, StorageType::Indexed).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    let operation = ResourceOperation::Status;

    let data = handler.handle(operation).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;

    let renderer = CliRenderer::new();
    renderer.render_with_session(&data, session)?;

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_clear(os: &Os, session: &mut ChatSession, confirm: bool, storage_type: StorageType) -> Result<ChatState, ChatError> {
    match storage_type {
        StorageType::All => {
            handle_clear_by_type(os, session, confirm, StorageType::Pinned).await?;
            handle_clear_by_type(os, session, confirm, StorageType::Indexed).await?;
        }
        _ => {
            handle_clear_by_type(os, session, confirm, storage_type).await?;
        }
    }

    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_clear_by_type(os: &Os, session: &mut ChatSession, confirm: bool, storage_type: StorageType) -> Result<(), ChatError> {
    let mut handler = get_handler(os, session, storage_type).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;
    let operation = ResourceOperation::Clear { confirm };

    let data = handler.handle(operation).await
        .map_err(|e| ChatError::Custom(e.to_string().into()))?;

    let renderer = CliRenderer::new();
    renderer.render_with_session(&data, session)?;
    Ok(())
}

async fn handle_cancel(os: &Os, session: &mut ChatSession, operation_id: String) -> Result<ChatState, ChatError> {
        // Cancel only applies to indexed resources (background operations)
        let mut handler = get_handler(os, session, StorageType::Indexed).await
            .map_err(|e| ChatError::Custom(e.to_string().into()))?;

        let resource_op = ResourceOperation::Cancel { operation_id };
        match handler.handle(resource_op).await {
            Ok(output) => println!("{:?}", output),
            Err(e) => eprintln!("Error: {}", e),
        }

        Ok(ChatState::PromptUser { skip_printing_tools: true })
}

/// Resource handler that delegates to different storage backends
enum Handler<'a> {
    Context(ResourceHandler<ContextResourceManager<'a>>),
    Knowledge(ResourceHandler<KnowledgeResourceManager>),
}

impl<'a> Handler<'a> {
    /// Handle a resource operation, returning the result or an error
    async fn handle(&mut self, operation: ResourceOperation) -> Result<ResourceData, eyre::Report> {
        match self {
            Handler::Context(handler) if handler.supports_operation(&operation) => {
                handler.handle(operation).await
            }
            Handler::Knowledge(handler) if handler.supports_operation(&operation) => {
                handler.handle(operation).await
            }
            _ => Err(eyre::eyre!("Operation not supported by this handler")),
        }
    }
}

/// Create a handler for the specified storage type
/// 
/// # Errors
/// Returns error if:
/// - Context manager is unavailable for Pinned storage
/// - Knowledge manager creation fails for Indexed storage  
/// - StorageType::All is requested (not supported)
async fn get_handler<'a>(
    os: &'a Os, 
    session: &'a mut ChatSession, 
    storage_type: StorageType
) -> Result<Handler<'a>, eyre::Report> {
    match storage_type {
        StorageType::Pinned => {
            let context_manager = session.conversation.context_manager.as_mut()
                .ok_or_else(|| eyre::eyre!("Context manager not available"))?;
            Ok(Handler::Context(ResourceHandler::new(
                ContextResourceManager::new(context_manager, os)
            )))
        }
        StorageType::Indexed => {
            let agent = session.conversation.agents.get_active();
            let knowledge_manager = KnowledgeResourceManager::new(os, agent).await?;
            Ok(Handler::Knowledge(ResourceHandler::new(knowledge_manager)))
        }
        StorageType::All => {
            Err(eyre::eyre!("StorageType::All not supported for single handler operations"))
        }
    }
}

impl ResourceNewCommand {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add { .. } => "add",
            Self::Remove { .. } => "remove",
            Self::Show { .. } => "show",
            Self::Status => "status",
            Self::Clear { .. } => "clear",
            Self::Cancel { .. } => "cancel",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_names() {
        assert_eq!(ResourceNewCommand::Add {
            name: "test".into(),
            value: "test".into(),
            r#type: "indexed".into()
        }.name(), "add");

        assert_eq!(ResourceNewCommand::Show {
            expand: false,
            r#type: "all".into()
        }.name(), "show");

        assert_eq!(ResourceNewCommand::Status.name(), "status");
    }
}
