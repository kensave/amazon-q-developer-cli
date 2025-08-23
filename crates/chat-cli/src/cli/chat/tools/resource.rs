// ABOUTME: Resource tool that provides unified access to both pinned and indexed resources
// ABOUTME: Uses shared ResourceCore for consistent behavior with CLI commands

use eyre::Result;
use serde::Deserialize;
use std::io::Write;
use crossterm::{queue, style::{self, Color}};

use crate::cli::agent::Agent;
use crate::cli::chat::tools::{InvokeOutput, OutputKind, PermissionEvalResult, sanitize_path_tool_arg};
use crate::cli::chat::resource::{ResourceOperation, StorageType, OutputFormat};
use crate::cli::chat::resource::core::ResourceCore;
use crate::cli::chat::resource::renderer::{ResourceRenderer, ToolRenderer};
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
    #[serde(default = "default_all_type", rename = "type")]
    pub r#type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceAdd {
    pub name: String,
    pub value: String,
    /// Storage type: "pinned" or "indexed" (default: "indexed")
    #[serde(default = "default_indexed_type", rename = "type")]
    pub r#type: String,
    /// Index type: "fast" (lexical search, recommended for code bases) or "best" (semantic search, recommended for documentation)
    pub index_type: Option<String>,
    /// Include patterns (e.g., `**/*.ts`, `**/*.md`) - only for indexed storage
    pub include: Option<Vec<String>>,
    /// Exclude patterns (e.g., `**/node_modules/**`, `**/target/**`) - only for indexed storage
    pub exclude: Option<Vec<String>>,
}

fn default_indexed_type() -> String {
    "indexed".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceRemove {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub path: String,
    /// Storage type: "pinned", "indexed", or "all" (default: "all")
    #[serde(default = "default_all_type", rename = "type")]
    pub r#type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceClear {
    /// Storage type: "pinned", "indexed", or "all" (default: "all")
    #[serde(default = "default_all_type", rename = "type")]
    pub r#type: String,
}

fn default_all_type() -> String {
    "all".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceSearch {
    pub query: String,
    pub resource_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceUpdate {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceCancel {
    pub operation_id: String,
}

impl Resource {
    pub fn is_enabled(_os: &Os) -> bool {
        true
    }

    pub async fn validate(&mut self, os: &Os) -> Result<()> {
        match self {
            Resource::Add(add) => {
                if !matches!(add.r#type.as_str(), "pinned" | "indexed") {
                    eyre::bail!("type must be either 'pinned' or 'indexed'");
                }
                // Validate pinned content only supports files/glob patterns
                if add.r#type == "pinned" && add.value.contains('\n') {
                    eyre::bail!("Pinned storage ONLY supports file/directory paths or glob patterns, not text content");
                }
                // Validate index_type if provided
                if let Some(ref index_type) = add.index_type {
                    if !matches!(index_type.as_str(), "fast" | "best") {
                        eyre::bail!("index_type must be either 'fast' (lexical search for code) or 'best' (semantic search for docs)");
                    }
                    if add.r#type != "indexed" {
                        eyre::bail!("index_type can only be used with indexed storage");
                    }
                }
                // Validate patterns only work with indexed
                if (add.include.is_some() || add.exclude.is_some()) && add.r#type != "indexed" {
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
                if update.path.is_empty() {
                    eyre::bail!("Path is required for update operation");
                }
                let path = sanitize_path_tool_arg(os, &update.path);
                if !path.exists() {
                    eyre::bail!("Path does not exist: {}", update.path);
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
                queue!(updates,
                    style::Print("Updating indexed resource with path: "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(&update.path),
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

    pub async fn invoke(&self, os: &Os, agent: Option<&Agent>) -> Result<InvokeOutput> {
        let renderer = ToolRenderer::new();

        let operation = match self {
            Resource::Add(add) => {
                ResourceOperation::Add {
                    name: add.name.clone(),
                    value: add.value.clone(),
                    include_patterns: add.include.clone(),
                    exclude_patterns: add.exclude.clone(),
                    index_type: add.index_type.clone(),
                }
            }
            Resource::Remove(remove) => {
                ResourceOperation::Remove {
                    id: if remove.id.is_empty() { None } else { Some(remove.id.clone()) },
                    name: if remove.name.is_empty() { None } else { Some(remove.name.clone()) },
                    path: if remove.path.is_empty() { None } else { Some(remove.path.clone()) },
                }
            }
            Resource::Clear(_clear) => {
                ResourceOperation::Clear { confirm: true }
            }
            Resource::Search(search) => {
                ResourceOperation::Search {
                    query: search.query.clone(),
                    context_id: search.resource_id.clone(),
                }
            }
            Resource::Update(update) => {
                ResourceOperation::Update {
                    path: update.path.clone(),
                }
            }
            Resource::Show(_show) => {
                ResourceOperation::Show { expand: true } // Show with status info
            }
            Resource::Cancel(cancel) => {
                ResourceOperation::Cancel { operation_id: cancel.operation_id.clone() }
            }
        };

        let storage_type = match self {
            Resource::Add(add) => match add.r#type.as_str() {
                "pinned" => StorageType::Pinned,
                "indexed" => StorageType::Indexed,
                _ => StorageType::Indexed, // default to indexed
            },
            Resource::Remove(remove) => match remove.r#type.as_str() {
                "pinned" => StorageType::Pinned,
                "indexed" => StorageType::Indexed,
                "all" => StorageType::All,
                _ => StorageType::All, // default to all
            },
            Resource::Clear(clear) => match clear.r#type.as_str() {
                "pinned" => StorageType::Pinned,
                "indexed" => StorageType::Indexed,
                "all" => StorageType::All,
                _ => StorageType::All, // default to all
            },
            Resource::Search(_) => StorageType::Indexed, // Search only indexed
            Resource::Update(_) => StorageType::Indexed, // Update only indexed
            Resource::Show(show) => match show.r#type.as_str() {
                "pinned" => StorageType::Pinned,
                "indexed" => StorageType::Indexed,
                "all" => StorageType::All,
                _ => StorageType::All, // default to all
            },
            Resource::Cancel(_) => StorageType::Indexed, // Cancel only indexed
        };

        let result = match storage_type {
            StorageType::Pinned => {
                // For tools, we don't have access to context_manager, so return error
                return Ok(InvokeOutput {
                    output: OutputKind::Text("Pinned resources not supported in tools".to_string()),
                });
            }
            StorageType::Indexed => {
                ResourceCore::invoke_indexed(operation, os, agent).await?
            }
            StorageType::All => {
                // For tools, just show indexed resources
                ResourceCore::invoke_indexed(ResourceOperation::Show { expand: false }, os, agent).await?
            }
        };
        let output = renderer.render(&result, OutputFormat::PlainText);

        Ok(InvokeOutput {
            output: OutputKind::Text(output),
        })
    }

    pub fn permission_eval(&self, _agent: &Agent) -> PermissionEvalResult {
        PermissionEvalResult::Allow
    }
}
