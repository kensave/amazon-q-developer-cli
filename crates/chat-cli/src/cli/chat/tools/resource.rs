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
        let async_knowledge_store = KnowledgeStore::get_async_instance(os, agent)
            .await
            .map_err(|e| eyre::eyre!("Failed to access resources: {}", e))?;
        let mut store = async_knowledge_store.lock().await;

        let result = match self {
            Resource::Add(add) => {
                // For path indexing, we'll show a progress message first
                let path = sanitize_path_tool_arg(os, &add.value);
                let value_to_use = if path.exists() {
                    path.to_string_lossy().to_string()
                } else {
                    // If it's not a valid path, use the original value (might be text content)
                    add.value.clone()
                };

                match store
                    .add(
                        &add.name,
                        &value_to_use,
                        crate::util::knowledge_store::AddOptions::with_db_defaults(os),
                    )
                    .await
                {
                    Ok(resource_id) => format!(
                        "Added '{}' to resources with ID: {}. Track active jobs in '/resource status' with provided id.",
                        add.name, resource_id
                    ),
                    Err(e) => format!("Failed to add to resources: {}", e),
                }
            },
            Resource::Remove(remove) => {
                if !remove.resource_id.is_empty() {
                    // Remove by ID
                    match store.remove_by_id(&remove.resource_id).await {
                        Ok(_) => format!("Removed resource with ID '{}' from resources", remove.resource_id),
                        Err(e) => format!("Failed to remove resource by ID: {}", e),
                    }
                } else if !remove.name.is_empty() {
                    // Remove by name
                    match store.remove_by_name(&remove.name).await {
                        Ok(_) => format!("Removed resource with name '{}' from resources", remove.name),
                        Err(e) => format!("Failed to remove resource by name: {}", e),
                    }
                } else if !remove.path.is_empty() {
                    // Remove by path
                    let sanitized_path = sanitize_path_tool_arg(os, &remove.path);
                    match store.remove_by_path(sanitized_path.to_string_lossy().as_ref()).await {
                        Ok(_) => format!("Removed resource with path '{}' from resources", remove.path),
                        Err(e) => format!("Failed to remove resource by path: {}", e),
                    }
                } else {
                    "Error: No identifier provided for removal. Please specify name, resource_id, or path.".to_string()
                }
            },
            Resource::Update(update) => {
                // Validate that we have a path and at least one identifier
                if update.path.is_empty() {
                    return Ok(InvokeOutput {
                        output: OutputKind::Text(
                            "Error: No path provided for update. Please specify a path to update with.".to_string(),
                        ),
                    });
                }

                // Sanitize the path
                let path = sanitize_path_tool_arg(os, &update.path);
                if !path.exists() {
                    return Ok(InvokeOutput {
                        output: OutputKind::Text(format!("Error: Path '{}' does not exist", update.path)),
                    });
                }

                let sanitized_path = path.to_string_lossy().to_string();

                // Choose the appropriate update method based on provided identifiers
                if !update.resource_id.is_empty() {
                    // Update by ID
                    match store.update_context_by_id(&update.resource_id, &sanitized_path).await {
                        Ok(_) => format!(
                            "Updated resource with ID '{}' using path '{}'. Track active jobs in '/resource status' with provided id.",
                            update.resource_id, update.path
                        ),
                        Err(e) => format!("Failed to update resource by ID: {}", e),
                    }
                } else if !update.name.is_empty() {
                    // Update by name
                    match store.update_context_by_name(&update.name, &sanitized_path).await {
                        Ok(_) => format!(
                            "Updated resource with name '{}' using path '{}'. Track active jobs in '/resource status' with provided id.",
                            update.name, update.path
                        ),
                        Err(e) => format!("Failed to update resource by name: {}", e),
                    }
                } else {
                    // Update by path (if no ID or name provided)
                    match store.update_by_path(&sanitized_path).await {
                        Ok(_) => format!(
                            "Updated resource with path '{}'. Track active jobs in '/resource status' with provided id.",
                            update.path
                        ),
                        Err(e) => format!("Failed to update resource by path: {}", e),
                    }
                }
            },
            Resource::Clear(_) => store
                .clear()
                .await
                .unwrap_or_else(|e| format!("Failed to clear resources: {}", e)),
            Resource::Search(search) => {
                let results = store.search(&search.query, search.resource_id.as_deref()).await;
                match results {
                    Ok(results) => {
                        if results.is_empty() {
                            format!("No matching entries found for query: \"{}\"", search.query)
                        } else {
                            let mut output = format!("Search results for \"{}\":\n\n", search.query);
                            for result in results {
                                if let Some(text) = result.text() {
                                    output.push_str(&format!("{}\n\n", text));
                                }
                            }
                            output
                        }
                    },
                    Err(e) => {
                        format!("Search failed: {}", e)
                    },
                }
            },
            Resource::Show => {
                let contexts = store.get_all().await;
                match contexts {
                    Ok(contexts) => {
                        if contexts.is_empty() {
                            "No resource entries found".to_string()
                        } else {
                            let mut output = String::from("Resource entries:\n");
                            for context in contexts {
                                output.push_str(&format!("- ID: {}\n  Name: {}\n  Description: {}\n  Persistent: {}\n  Created: {}\n  Last Updated: {}\n  Items: {}\n\n",
                                    context.id,
                                    context.name,
                                    context.description,
                                    context.persistent,
                                    context.created_at.format("%Y-%m-%d %H:%M:%S"),
                                    context.updated_at.format("%Y-%m-%d %H:%M:%S"),
                                    context.item_count
                                ));
                            }
                            output
                        }
                    },
                    Err(e) => format!("Failed to get resource entries: {}", e),
                }
            },
            Resource::Status => {
                match store.get_status_data().await {
                    Ok(status_data) => {
                        // Format the status data for display (same logic as knowledge command)
                        Self::format_status_display(&status_data)
                    },
                    Err(e) => format!("Failed to get status: {}", e),
                }
            },
            Resource::Cancel(cancel) => store
                .cancel_operation(Some(&cancel.operation_id))
                .await
                .unwrap_or_else(|e| format!("Failed to cancel operation: {}", e)),
        };

        Ok(InvokeOutput {
            output: OutputKind::Text(result),
        })
    }

    pub fn eval_perm(&self, _agent: &Agent) -> PermissionEvalResult {
        // Resource tool is a core tool like thinking and gh_issue, always allow
        PermissionEvalResult::Allow
    }

    /// Format status data for display (UI rendering responsibility)
    fn format_status_display(status: &semantic_search_client::SystemStatus) -> String {
        let mut status_lines = Vec::new();

        // Show context summary
        status_lines.push(format!(
            "Total contexts: {} ({} persistent, {} volatile)",
            status.total_contexts, status.persistent_contexts, status.volatile_contexts
        ));

        if status.operations.is_empty() {
            status_lines.push("No active operations".to_string());
            return status_lines.join("\n");
        }

        status_lines.push("Active Operations:".to_string());
        status_lines.push(format!(
            "Queue Status: {} active, {} waiting (max {} concurrent)",
            status.active_count, status.waiting_count, status.max_concurrent
        ));

        for op in &status.operations {
            let formatted_operation = Self::format_operation_display(op);
            status_lines.push(formatted_operation);
        }

        status_lines.join("\n")
    }

    /// Format a single operation for display (LLM-friendly data format)
    fn format_operation_display(op: &semantic_search_client::OperationStatus) -> String {
        let elapsed = op.started_at.elapsed().unwrap_or_default();

        let status_info = if op.is_cancelled {
            "Status: Cancelled".to_string()
        } else if op.is_failed {
            format!("Status: Failed - {}", op.message)
        } else if op.is_waiting {
            format!("Status: Waiting - {}", op.message)
        } else if op.total > 0 {
            let percentage = (op.current as f64 / op.total as f64 * 100.0) as u8;
            format!(
                "Status: In Progress - {}% ({}/{}) - {}",
                percentage, op.current, op.total, op.message
            )
        } else {
            format!("Status: In Progress - {}", op.message)
        };

        let operation_desc = op.operation_type.display_name();

        // Format with conditional elapsed time and ETA
        if op.is_cancelled || op.is_failed {
            format!(
                "Operation ID: {} | Type: {} | {}",
                op.short_id, operation_desc, status_info
            )
        } else {
            let mut time_info = format!("Elapsed: {}s", elapsed.as_secs());

            if let Some(eta) = op.eta {
                time_info.push_str(&format!(" | ETA: {}s", eta.as_secs()));
            }

            format!(
                "Operation ID: {} | Type: {} | {} | {}",
                op.short_id, operation_desc, status_info, time_info
            )
        }
    }
}
