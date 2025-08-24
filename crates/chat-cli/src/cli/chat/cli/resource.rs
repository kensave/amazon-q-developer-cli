use clap::Subcommand;
use eyre::Result;
use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::cli::chat::resource::{StorageType, ResourceOperation, ResourceData};
use crate::cli::chat::resource::core::ResourceCore;
use crate::cli::chat::resource::renderer::{CliRenderer, ResourceRenderer};
use crate::os::Os;
use crate::database::settings::Setting;

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
        #[arg(long)]
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
        #[arg(long)]
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
        #[arg(long)]
        r#type: StorageType,
    },
    /// Clear resources
    Clear {
        /// Confirm the clear operation
        #[arg(long)]
        confirm: bool,
        /// Storage type to clear: "pinned", "indexed", or "all"
        #[arg(long)]
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
        if !is_feature_enabled(os) {
            let info_data = ResourceData::Info(
                "Resource tool is disabled. Enable it with: q settings chat.enableKnowledge true\n💡 Your resource data is preserved and will be available when re-enabled.".to_string()
            );
            render_to_session(&info_data, session)?;
            return Ok(ChatState::PromptUser { skip_printing_tools: true });
        }

        match self {
            Self::Add { name, value, r#type, include, exclude, index_type } => {
                validate_add_inputs(session, &r#type, &include, &exclude, &index_type)?;
                let operation = ResourceOperation::Add {
                    name,
                    value,
                    include_patterns: (!include.is_empty()).then_some(include),
                    exclude_patterns: (!exclude.is_empty()).then_some(exclude),
                    index_type
                };
                handle_operation(operation, r#type, os, session).await
            }
            Self::Remove { id, name, path, r#type } => {
                for st in get_storage_types(r#type) {
                    let operation = ResourceOperation::Remove { id: id.clone(), name: name.clone(), path: path.clone() };
                    println!("Handling operation for {}", st);
                    handle_operation(operation, st, os, session).await?;
                }
                Ok(ChatState::PromptUser { skip_printing_tools: true })
            }
            Self::Show { expand, r#type } => {
                for st in get_storage_types(r#type) {
                    let operation = ResourceOperation::Show { expand };
                    handle_operation(operation, st, os, session).await?;
                }
                Ok(ChatState::PromptUser { skip_printing_tools: true })
            }
            Self::Update { path, r#type } => {
                let operation = ResourceOperation::Update { id: None, path: Some(path) };
                handle_operation(operation, r#type, os, session).await
            }
            Self::Clear { confirm, r#type } => {
                for st in get_storage_types(r#type) {
                    let operation = ResourceOperation::Clear { confirm };
                    handle_operation(operation, st, os, session).await?;
                }
                Ok(ChatState::PromptUser { skip_printing_tools: true })
            }
            Self::Cancel { operation_id } => {
                let operation = ResourceOperation::Cancel { operation_id };
                handle_operation(operation, StorageType::Indexed, os, session).await
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

async fn handle_operation(
    operation: ResourceOperation,
    storage_type: StorageType,
    os: &Os,
    session: &mut ChatSession,
) -> Result<ChatState, ChatError> {
    let data = match storage_type {
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
            return Err(ChatError::Custom("StorageType::All should not reach single invocation".into()));
        }
    };

    render_to_session(&data, session)?;
    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

fn to_chat_error(e: impl std::fmt::Display) -> ChatError {
    ChatError::Custom(e.to_string().into())
}

fn render_to_session(data: &ResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
    let renderer = CliRenderer::new();
    renderer.render_with_session(data, session)
}

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
        let warning_data = ResourceData::Info(
            "Warning: include/exclude patterns and index-type are ignored for pinned resources".to_string()
        );
        render_to_session(&warning_data, session)?;
    }
    Ok(())
}

fn is_feature_enabled(os: &Os) -> bool {
    os.database
        .settings
        .get_bool(Setting::EnabledKnowledge)
        .unwrap_or(false)
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
