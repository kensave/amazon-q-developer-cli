// ABOUTME: Resource tool that provides unified access to both pinned and indexed resources
// ABOUTME: Replaces the context tool to provide a single interface for resource management

use std::io::Write;

use crossterm::{
    queue,
    style::{self, Color},
};
use eyre::Result;
use serde::Deserialize;

use crate::cli::agent::Agent;
use crate::cli::chat::tools::{
    sanitize_path_tool_arg,
    InvokeOutput,
    OutputKind,
    PermissionEvalResult,
};
use crate::os::Os;
use crate::util::knowledge_store::KnowledgeStore;

/// The Resource tool allows managing both pinned (always included) and indexed (retrieved on demand) resources.
/// It provides a unified interface for context management across chat sessions.
///
/// This feature can be enabled/disabled via settings:
/// - Knowledge base functionality requires the knowledge feature to be enabled
/// - Session context is always available when context management is enabled

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "command", rename_all = "lowercase")]
pub enum Resource {
    Add(ResourceAdd),
    Remove(ResourceRemove),
    Clear(ResourceClear),
    Search(ResourceSearch),
    Update(ResourceUpdate),
    Show,
    /// Show background operation status
    Status,
    /// Cancel a background operation
    Cancel(ResourceCancel),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceAdd {
    pub name: String,
    pub value: String,
    /// Storage type: "pinned" (always included) or "indexed" (retrieved on demand)
    #[serde(default = "default_storage_type")]
    pub storage_type: String,
}

fn default_storage_type() -> String {
    "pinned".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceRemove {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceClear {
    pub confirm: bool,
    /// Storage type to clear: "pinned", "indexed", or "all" (default)
    #[serde(default = "default_clear_type")]
    pub storage_type: String,
}

fn default_clear_type() -> String {
    "all".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceSearch {
    pub query: String,
    pub resource_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceUpdate {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResourceCancel {
    /// Operation ID to cancel, or "all" to cancel all operations
    pub operation_id: String,
}

impl Resource {
    /// Checks if the knowledge feature is enabled in settings
    pub fn is_enabled(_os: &Os) -> bool {
        // For now, always enabled since it's a unified tool
        true
    }

    pub async fn validate(&mut self, os: &Os) -> Result<()> {
        match self {
            Resource::Add(add) => {
                // Validate storage type
                if !matches!(add.storage_type.as_str(), "pinned" | "indexed") {
                    eyre::bail!("storage_type must be either 'pinned' or 'indexed'");
                }
                
                // Check if value is intended to be a path (doesn't contain newlines)
                if !add.value.contains('\n') {
                    let path = sanitize_path_tool_arg(os, &add.value);
                    if !path.exists() {
                        eyre::bail!("Path does not exist: {}", add.value);
                    }
                }
                Ok(())
            },
            Resource::Remove(remove) => {
                if remove.name.is_empty() && remove.resource_id.is_empty() && remove.path.is_empty() {
                    eyre::bail!("Please provide at least one of: name, resource_id, or path");
                }
                // If path is provided, validate it exists
                if !remove.path.is_empty() {
                    let path = sanitize_path_tool_arg(os, &remove.path);
                    if !path.exists() {
                        eyre::bail!("Path does not exist: {}", remove.path);
                    }
                }
                Ok(())
            },
            Resource::Update(update) => {
                // Require at least one identifier (resource_id or name)
                if update.resource_id.is_empty() && update.name.is_empty() && update.path.is_empty() {
                    eyre::bail!(
                        "Please provide either resource_id, name, or path to identify the resource entry to update"
                    );
                }
                
                // If path is provided, validate it exists
                if !update.path.is_empty() {
                    let path = sanitize_path_tool_arg(os, &update.path);
                    if !path.exists() {
                        eyre::bail!("Path does not exist: {}", update.path);
                    }
                }
                Ok(())
            },
            Resource::Clear(clear) => {
                if !clear.confirm {
                    eyre::bail!("Please confirm clearing resources by setting confirm=true");
                }
                // Validate storage type
                if !matches!(clear.storage_type.as_str(), "pinned" | "indexed" | "all") {
                    eyre::bail!("storage_type must be 'pinned', 'indexed', or 'all'");
                }
                Ok(())
            },
            Resource::Search(_) => Ok(()),
            Resource::Show => Ok(()),
            Resource::Status => Ok(()),
            Resource::Cancel(_) => Ok(()),
        }
    }

    pub async fn queue_description(&self, _os: &Os, updates: &mut impl Write) -> Result<()> {
        match self {
            Resource::Add(add) => {
                queue!(
                    updates,
                    style::Print("Adding to resources: "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(&add.name),
                    style::ResetColor,
                    style::Print(" as "),
                    style::SetForegroundColor(Color::Cyan),
                    style::Print(&add.storage_type),
                    style::ResetColor,
                )?;
            },
            Resource::Remove(remove) => {
                if !remove.name.is_empty() {
                    queue!(
                        updates,
                        style::Print("Removing from resources by name: "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(&remove.name),
                        style::ResetColor,
                    )?;
                } else if !remove.resource_id.is_empty() {
                    queue!(
                        updates,
                        style::Print("Removing from resources by ID: "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(&remove.resource_id),
                        style::ResetColor,
                    )?;
                } else if !remove.path.is_empty() {
                    queue!(
                        updates,
                        style::Print("Removing from resources by path: "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(&remove.path),
                        style::ResetColor,
                    )?;
                } else {
                    queue!(
                        updates,
                        style::Print("Removing from resources: "),
                        style::SetForegroundColor(Color::Yellow),
                        style::Print("No identifier provided"),
                        style::ResetColor,
                    )?;
                }
            },
            Resource::Update(update) => {
                queue!(updates, style::Print("Updating resource"),)?;

                if !update.resource_id.is_empty() {
                    queue!(
                        updates,
                        style::Print(" with ID: "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(&update.resource_id),
                        style::ResetColor,
                    )?;
                } else if !update.name.is_empty() {
                    queue!(
                        updates,
                        style::Print(" with name: "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(&update.name),
                        style::ResetColor,
                    )?;
                } else if !update.path.is_empty() {
                    queue!(
                        updates,
                        style::Print(" with path: "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(&update.path),
                        style::ResetColor,
                    )?;
                }
            },
            Resource::Clear(clear) => {
                queue!(
                    updates,
                    style::Print("Clearing "),
                    style::SetForegroundColor(Color::Red),
                    style::Print(&clear.storage_type),
                    style::ResetColor,
                    style::Print(" resource entries"),
                )?;
            },
            Resource::Search(search) => {
                queue!(
                    updates,
                    style::Print("Searching resources for: "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(&search.query),
                    style::ResetColor,
                )?;

                if let Some(resource_id) = &search.resource_id {
                    queue!(
                        updates,
                        style::Print(" in resource: "),
                        style::SetForegroundColor(Color::Green),
                        style::Print(resource_id),
                        style::ResetColor,
                    )?;
                } else {
                    queue!(updates, style::Print(" across all resources"),)?;
                }
            },
            Resource::Show => {
                queue!(updates, style::Print("Showing all resource entries"),)?;
            },
            Resource::Status => {
                queue!(updates, style::Print("Checking background operation status"),)?;
            },
            Resource::Cancel(cancel) => {
                queue!(
                    updates,
                    style::Print("Cancelling operation: "),
                    style::SetForegroundColor(Color::Yellow),
                    style::Print(&cancel.operation_id),
                    style::ResetColor,
                )?;
            },
        }
        Ok(())
    }

    pub async fn invoke(&self, os: &Os, _updates: &mut impl Write, agent: Option<&crate::cli::agent::Agent>) -> Result<InvokeOutput> {
        // Get agent name from the agent parameter
        let agent_name = agent.map(|a| a.name.as_str());

        // For indexed resources, we use the knowledge store
        // For pinned resources, we use the session context
        // This tool provides a unified interface to both

        let async_knowledge_store = KnowledgeStore::get_async_instance(os, agent_name)
            .await
            .map_err(|e| eyre::eyre!("Failed to access resources: {}", e))?;
        let mut store = async_knowledge_store.lock().await;

        let result = match self {
            Resource::Add(add) => {
                if add.storage_type == "indexed" {
                    // Add to knowledge store (indexed)
                    let path = sanitize_path_tool_arg(os, &add.value);
                    match store.add(&add.name, path.to_string_lossy().as_ref(), Default::default()).await {
                        Ok(resource_id) => format!(
                            "Added '{}' to indexed resources with ID: {}. Track active jobs in '/resource status'.",
                            add.name, resource_id
                        ),
                        Err(e) => format!("Failed to add to indexed resources: {}", e),
                    }
                } else {
                    // Add to session context (pinned) - this would need integration with context manager
                    format!("Adding to pinned resources is handled via /context add command. Use '/context add {}' instead.", add.value)
                }
            },
            Resource::Show => {
                // Show both pinned and indexed resources
                "Use '/resource show' slash command to see both pinned and indexed resources.".to_string()
            },
            Resource::Search(search) => {
                // Search only works on indexed resources
                let results = store.search(&search.query, search.resource_id.as_deref()).await;
                match results {
                    Ok(results) => {
                        if results.is_empty() {
                            "No matching indexed resources found.".to_string()
                        } else {
                            let mut output = format!("Found {} matching indexed resources:\n", results.len());
                            for result in results.iter().take(5) {
                                // Get full content from payload
                                let content = result.point.payload.get("content")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("No content available");
                                output.push_str(&format!("- {} (distance: {:.2})\n", content, result.distance));
                            }
                            if results.len() > 5 {
                                output.push_str(&format!("... and {} more results\n", results.len() - 5));
                            }
                            output
                        }
                    },
                    Err(e) => format!("Search failed: {}", e),
                }
            },
            _ => "This operation is best performed using the '/resource' slash command.".to_string(),
        };

        Ok(InvokeOutput {
            output: OutputKind::Text(result),
        })
    }

    pub fn eval_perm(&self, _agent: &Agent) -> PermissionEvalResult {
        // Resource tool is a core tool like thinking and gh_issue, always allow
        PermissionEvalResult::Allow
    }
}
