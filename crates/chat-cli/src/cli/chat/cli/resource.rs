use clap::Subcommand;
use eyre::Result;
use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::cli::chat::resource::{StorageType, ResourceOperation, ResourceData};
use crate::cli::chat::resource::core::ResourceCore;
use crate::os::Os;

// Parameter structs to simplify function signatures
#[derive(Debug)]
struct AddParams {
    name: String,
    value: String,
    storage_type: StorageType,
    include: Vec<String>,
    exclude: Vec<String>,
    index_type: Option<String>,
}

#[derive(Debug)]
struct RemoveParams {
    id: Option<String>,
    name: Option<String>,
    path: Option<String>,
    storage_type: StorageType,
}

/// Unified resource management system
#[derive(Debug, PartialEq, Subcommand)]
pub enum ResourceCommand {
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

impl ResourceCommand {
    pub async fn execute(self, os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        match self {
            Self::Add { name, value, r#type, include, exclude, index_type } => {
                let params = AddParams { name, value, storage_type: r#type, include, exclude, index_type };
                handle_add(os, session, params).await
            }
            Self::Remove { id, name, path, r#type } => {
                let params = RemoveParams { id, name, path, storage_type: r#type };
                handle_remove(os, session, params).await
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

async fn handle_add(os: &Os, session: &mut ChatSession, params: AddParams) -> Result<ChatState, ChatError> {
    validate_add_inputs(session, &params.storage_type, &params.include, &params.exclude, &params.index_type)?;

    let operation = ResourceOperation::Add {
        name: params.name,
        value: params.value,
        include_patterns: (!params.include.is_empty()).then_some(params.include),
        exclude_patterns: (!params.exclude.is_empty()).then_some(params.exclude),
        index_type: params.index_type
    };

    let data = match params.storage_type {
        StorageType::Pinned => {
            let context_manager = session.conversation.context_manager.as_mut()
                .ok_or_else(|| ChatError::Custom("Context manager not available".into()))?;
            ResourceCore::invoke_pinned(operation, context_manager, os).await.map_err(to_chat_error)?
        }
        StorageType::Indexed => {
            let agent = session.conversation.agents.get_active();
            ResourceCore::invoke_indexed(operation, os, agent).await.map_err(to_chat_error)?
        }
        StorageType::All => {
            return Err(ChatError::Custom("StorageType::All not supported for add operation".into()));
        }
    };

    render_to_session(&data, session)?;
    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_update(os: &Os, session: &mut ChatSession, path: String, storage_type: StorageType) -> Result<ChatState, ChatError> {
    let operation = ResourceOperation::Update { path };

    let data = match storage_type {
        StorageType::Pinned => {
            return Err(ChatError::Custom("Update not supported for pinned resources".into()));
        }
        StorageType::Indexed => {
            let agent = session.conversation.agents.get_active();
            ResourceCore::invoke_indexed(operation, os, agent).await.map_err(to_chat_error)?
        }
        StorageType::All => {
            return Err(ChatError::Custom("StorageType::All not supported for update operation".into()));
        }
    };

    render_to_session(&data, session)?;
    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_remove(os: &Os, session: &mut ChatSession, params: RemoveParams) -> Result<ChatState, ChatError> {
    for st in get_storage_types(params.storage_type) {
        handle_remove_by_type(os, session, params.id.clone(), params.name.clone(), params.path.clone(), st).await?;
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

async fn handle_cancel(os: &Os, session: &mut ChatSession, operation_id: String) -> Result<ChatState, ChatError> {
    let operation = ResourceOperation::Cancel { operation_id };
    let agent = session.conversation.agents.get_active();

    let data = ResourceCore::invoke_indexed(operation, os, agent).await.map_err(to_chat_error)?;
    render_to_session(&data, session)?;
    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

// ============================================================================
// Helper Functions (by_type handlers)
// ============================================================================

/// Generic handler for operations by storage type - eliminates DRY violation
async fn handle_operation_by_type(
    operation: ResourceOperation,
    storage_type: StorageType,
    os: &Os,
    session: &mut ChatSession,
) -> Result<(), ChatError> {
    let data = invoke_by_storage_type(operation, storage_type, os, session).await?;
    render_to_session(&data, session)?;
    Ok(())
}

async fn handle_remove_by_type(os: &Os, session: &mut ChatSession, id: Option<String>, name: Option<String>, path: Option<String>, storage_type: StorageType) -> Result<(), ChatError> {
    handle_operation_by_type(ResourceOperation::Remove { id, name, path }, storage_type, os, session).await
}

async fn handle_show_by_type(os: &Os, session: &mut ChatSession, expand: bool, storage_type: StorageType) -> Result<(), ChatError> {
    handle_operation_by_type(ResourceOperation::Show { expand }, storage_type, os, session).await
}

async fn handle_clear_by_type(os: &Os, session: &mut ChatSession, confirm: bool, storage_type: StorageType) -> Result<(), ChatError> {
    handle_operation_by_type(ResourceOperation::Clear { confirm }, storage_type, os, session).await
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert any error to ChatError for consistent error handling
fn to_chat_error(e: impl std::fmt::Display) -> ChatError {
    ChatError::Custom(e.to_string().into())
}

/// Invoke operation by storage type - eliminates DRY violation
async fn invoke_by_storage_type(
    operation: ResourceOperation,
    storage_type: StorageType,
    os: &Os,
    session: &mut ChatSession,
) -> Result<ResourceData, ChatError> {
    match storage_type {
        StorageType::Pinned => {
            let context_manager = session.conversation.context_manager.as_mut()
                .ok_or_else(|| ChatError::Custom("Context manager not available".into()))?;
            ResourceCore::invoke_pinned(operation, context_manager, os).await.map_err(to_chat_error)
        }
        StorageType::Indexed => {
            let agent = session.conversation.agents.get_active();
            ResourceCore::invoke_indexed(operation, os, agent).await.map_err(to_chat_error)
        }
        StorageType::All => {
            Err(ChatError::Custom("StorageType::All should not reach single invocation".into()))
        }
    }
}

/// Render resource data to session with colors and styling
fn render_to_session(data: &ResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
    use crate::cli::chat::resource::renderer::{CliRenderer, ResourceRenderer};
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

fn validate_add_inputs(
    session: &mut ChatSession,
    storage_type: &StorageType,
    include: &[String],
    exclude: &[String],
    index_type: &Option<String>
) -> Result<(), ChatError> {
    if *storage_type == StorageType::Pinned && (!include.is_empty() || !exclude.is_empty() || index_type.is_some()) {
        let warning_data = ResourceData::Success(
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
        assert_eq!(ResourceCommand::Add {
            name: "test".into(),
            value: "test".into(),
            r#type: StorageType::Indexed,
            include: vec![],
            exclude: vec![],
            index_type: None,
        }.name(), "add");

        assert_eq!(ResourceCommand::Show {
            expand: false,
            r#type: StorageType::All,
        }.name(), "show");

        assert_eq!(ResourceCommand::Update {
            path: "test".into(),
            r#type: StorageType::Indexed,
        }.name(), "update");
    }
}
