// ABOUTME: Resource tool that provides unified access to both pinned and indexed resources
// ABOUTME: Uses shared ResourceCore for consistent behavior with CLI commands

use eyre::Result;
use serde::Deserialize;
use std::io::Write;
use crossterm::{queue, style::{self, Color}};

use crate::cli::agent::Agent;
use crate::cli::chat::tools::{InvokeOutput, OutputKind, PermissionEvalResult, sanitize_path_tool_arg};
use crate::cli::chat::resource::{ResourceOperation, StorageType, OutputFormat, ResourceData};
use crate::cli::chat::resource::core::ResourceCore;
use crate::cli::chat::resource::renderer::{ResourceRenderer, ToolRenderer};
use crate::cli::chat::ChatSession;
use crate::os::Os;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "command", rename_all = "lowercase")]
pub enum Resource {
    Add(ResourceAdd),
    Remove(ResourceRemove),
    Clear(ResourceClear),
    Search(ResourceSearch),
    Update(ResourceUpdate),
    Show(ResourceShow),
    Cancel(ResourceCancel),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceShow {
    /// Storage type to show: "pinned", "indexed", or "all" (default: "all")
    #[serde(default = "all", rename = "type")]
    pub r#type: StorageType,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceAdd {
    pub name: String,
    pub value: String,
    /// Storage type: "pinned" or "indexed" (default: "indexed")
    #[serde(default = "indexed", rename = "type")]
    pub r#type: StorageType,
    /// Index type: "fast" (lexical search, recommended for code bases) or "best" (semantic search, recommended for documentation)
    pub index_type: Option<String>,
    /// Include patterns (e.g., `**/*.ts`, `**/*.md`) - only for indexed storage
    pub include: Option<Vec<String>>,
    /// Exclude patterns (e.g., `**/node_modules/**`, `**/target/**`) - only for indexed storage
    pub exclude: Option<Vec<String>>,
}

fn all() -> StorageType {
    StorageType::All
}

fn indexed() -> StorageType {
    StorageType::Indexed
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceRemove {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub path: String,
    /// Storage type: "pinned", "all", or "all" (default: "all")
    #[serde(default = "all", rename = "type")]
    pub r#type: StorageType,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceClear {
    /// Storage type: "pinned", "indexed", or "all" (default: "all")
    #[serde(default = "all", rename = "type")]
    pub r#type: StorageType,
}



#[derive(Debug, Clone, Deserialize)]
pub struct ResourceSearch {
    pub query: String,
    pub id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceUpdate {
    pub path: Option<String>,
    pub id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceCancel {
    pub operation_id: String,
}

impl Resource {
    // Entry point - main public interface
    pub async fn invoke(&self, os: &Os, session: &mut ChatSession) -> Result<InvokeOutput> {
        let operation = self.to_operation();
        let storage_type = self.get_storage_type();

        let result = self.invoke_operation(operation, storage_type, os, session).await?;
        let renderer = ToolRenderer::new();
        Ok(InvokeOutput {
            output: OutputKind::Text(renderer.render(&result, OutputFormat::PlainText))
        })
    }

    // Public interface methods
    pub fn is_enabled(_os: &Os) -> bool {
        true
    }

    pub async fn validate(&mut self, os: &Os) -> Result<()> {
        match self {
            Resource::Add(add) => {
                if !matches!(add.r#type,  StorageType::Indexed |  StorageType::Pinned) {
                    eyre::bail!("type must be either 'pinned' or 'indexed'");
                }
                // Validate pinned content only supports files/glob patterns
                if add.r#type == StorageType::Pinned && add.value.contains('\n') {
                    eyre::bail!("Pinned storage ONLY supports file/directory paths or glob patterns, not text content");
                }
                // Validate index_type if provided
                if let Some(ref index_type) = add.index_type {
                    if !matches!(index_type.as_str(), "fast" | "best") {
                        eyre::bail!("index_type must be either 'fast' (lexical search for code) or 'best' (semantic search for docs)");
                    }
                    if add.r#type != StorageType::Indexed {
                        eyre::bail!("index_type can only be used with indexed storage");
                    }
                }
                // Validate patterns only work with indexed
                if (add.include.is_some() || add.exclude.is_some()) && add.r#type != StorageType::Indexed {
                    eyre::bail!("include/exclude patterns can only be used with indexed storage");
                }
                // Check if value is a path
                if !add.value.contains('\n') {
                    let path = sanitize_path_tool_arg(os, &add.value);
                    if !path.exists() {
                        eyre::bail!("Path does not exist: {}", add.value);
                    }
                }
            }
            Resource::Remove(remove) => {
                if !matches!(remove.r#type.as_str(), "pinned" | "indexed" | "all") {
                    eyre::bail!("type must be 'pinned', 'indexed', or 'all'");
                }
                if remove.id.is_empty() && remove.name.is_empty() && remove.path.is_empty() {
                    eyre::bail!("Please provide at least one of: id, name, or path");
                }
                if !remove.path.is_empty() {
                    let path = sanitize_path_tool_arg(os, &remove.path);
                    if !path.exists() {
                        eyre::bail!("Path does not exist: {}", remove.path);
                    }
                }
            }
            Resource::Clear(clear) => {
                if !matches!(clear.r#type.as_str(), "pinned" | "indexed" | "all") {
                    eyre::bail!("type must be 'pinned', 'indexed', or 'all'");
                }
            }
            Resource::Update(update) => {
                if update.path.is_none() && update.id.is_none() {
                    eyre::bail!("Either path or id is required for update operation");
                }
                if let Some(path) = &update.path {
                    let path_buf = sanitize_path_tool_arg(os, path);
                    if !path_buf.exists() {
                        eyre::bail!("Path does not exist: {}", path);
                    }
                }
            }
            Resource::Show(show) => {
                if !matches!(show.r#type.as_str(), "pinned" | "indexed" | "all") {
                    eyre::bail!("type must be 'pinned', 'indexed', or 'all'");
                }
            }
            Resource::Search(_) => {
                // No additional validation needed
            }
            Resource::Cancel(_) => {
                // No additional validation needed
            }
        }
        Ok(())
    }

    pub async fn queue_description(&self, _os: &Os, updates: &mut impl Write) -> Result<()> {
        match self {
            Resource::Add(add) => {
                queue!(updates,
                    style::Print("Adding to resources: "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(&add.name),
                    style::ResetColor,
                    style::Print(" as "),
                    style::SetForegroundColor(Color::Cyan),
                    style::Print(&add.r#type),
                    style::ResetColor,
                )?;
            }
            Resource::Remove(remove) => {
                let identifier = if !remove.name.is_empty() {
                    format!("name: {}", remove.name)
                } else if !remove.id.is_empty() {
                    format!("ID: {}", remove.id)
                } else if !remove.path.is_empty() {
                    format!("path: {}", remove.path)
                } else {
                    "unknown identifier".to_string()
                };
                queue!(updates,
                    style::Print("Removing from resources by "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(identifier),
                    style::ResetColor,
                )?;
            }
            Resource::Clear(_) => {
                queue!(updates,
                    style::Print("Clearing "),
                    style::SetForegroundColor(Color::Red),
                    style::Print("all"),
                    style::ResetColor,
                    style::Print(" resource entries"),
                )?;
            }
            Resource::Search(search) => {
                queue!(updates,
                    style::Print("Searching indexed resources for: "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(&search.query),
                    style::ResetColor,
                )?;
            }
            Resource::Update(update) => {
                let identifier = if let Some(path) = &update.path {
                    format!("path: {}", path)
                } else if let Some(id) = &update.id {
                    format!("id: {}", id)
                } else {
                    "unknown".to_string()
                };
                queue!(updates,
                    style::Print("Updating indexed resource with "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(&identifier),
                    style::ResetColor,
                )?;
            }
            Resource::Show(show) => {
                queue!(updates,
                    style::Print("Showing "),
                    style::SetForegroundColor(Color::Cyan),
                    style::Print(&show.r#type),
                    style::ResetColor,
                    style::Print(" resources with status"),
                )?;
            }
            Resource::Cancel(cancel) => {
                queue!(updates,
                    style::Print("Cancelling operation: "),
                    style::SetForegroundColor(Color::Yellow),
                    style::Print(&cancel.operation_id),
                    style::ResetColor,
                )?;
            }
        }
        Ok(())
    }

    pub fn permission_eval(&self, _agent: &Agent) -> PermissionEvalResult {
        PermissionEvalResult::Allow
    }

    // Private utility methods
    fn to_operation(&self) -> ResourceOperation {
        match self {
            Resource::Add(add) => ResourceOperation::Add {
                name: add.name.clone(),
                value: add.value.clone(),
                include_patterns: add.include.clone(),
                exclude_patterns: add.exclude.clone(),
                index_type: add.index_type.clone(),
            },
            Resource::Remove(remove) => ResourceOperation::Remove {
                id: if remove.id.is_empty() { None } else { Some(remove.id.clone()) },
                name: if remove.name.is_empty() { None } else { Some(remove.name.clone()) },
                path: if remove.path.is_empty() { None } else { Some(remove.path.clone()) },
            },
            Resource::Clear(_) => ResourceOperation::Clear { confirm: true },
            Resource::Search(search) => ResourceOperation::Search {
                query: search.query.clone(),
                context_id: search.id.clone(),
            },
            Resource::Update(update) => ResourceOperation::Update { 
                id: update.id.clone(), 
                path: update.path.clone() 
            },
            Resource::Show(_) => ResourceOperation::Show { expand: true },
            Resource::Cancel(cancel) => ResourceOperation::Cancel { operation_id: cancel.operation_id.clone() },
        }
    }

    fn get_storage_type(&self) -> StorageType {
        match self {
            Resource::Add(add) => add.r#type.clone(),
            Resource::Remove(remove) => remove.r#type.clone(),
            Resource::Clear(clear) => clear.r#type.clone(),
            Resource::Show(show) => show.r#type.clone(),
            _ => StorageType::Indexed, // Search, Update, Cancel only work with indexed
        }
    }

    fn get_context_manager(session: &mut ChatSession) -> Result<&mut crate::cli::chat::context::ContextManager> {
        session.conversation.context_manager.as_mut()
            .ok_or_else(|| eyre::eyre!("Context manager not available"))
    }

    async fn invoke_operation(
        &self,
        operation: ResourceOperation,
        storage_type: StorageType,
        os: &Os,
        session: &mut ChatSession
    ) -> Result<ResourceData> {
        match storage_type {
            StorageType::Pinned => {
                ResourceCore::invoke_pinned(operation, Self::get_context_manager(session)?, os).await
            }
            StorageType::Indexed => {
                ResourceCore::invoke_indexed(operation, os, session.conversation.agents.get_active()).await
            }
            StorageType::All => {
                self.execute_all_storage_types(operation, os, session).await
            }
        }
    }

    async fn execute_all_storage_types(
        &self,
        operation: ResourceOperation,
        os: &Os,
        session: &mut ChatSession
    ) -> Result<ResourceData> {
        let pinned_result = ResourceCore::invoke_pinned(
            operation.clone(),
            Self::get_context_manager(session)?,
            os
        ).await.ok();

        let indexed_result = ResourceCore::invoke_indexed(
            operation,
            os,
            session.conversation.agents.get_active()
        ).await.ok();

        match (pinned_result, indexed_result) {
            (Some(pinned), Some(indexed)) => {
                let renderer = ToolRenderer::new();
                let combined = format!(
                    "{}\n{}",
                    renderer.render(&pinned, OutputFormat::PlainText),
                    renderer.render(&indexed, OutputFormat::PlainText)
                );
                Ok(ResourceData::Success(combined))
            }
            (Some(result), None) | (None, Some(result)) => Ok(result),
            (None, None) => Ok(ResourceData::Success("No resources found".to_string())),
        }
    }
}
