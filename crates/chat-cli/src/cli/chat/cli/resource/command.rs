use clap::Subcommand;
use eyre::Result;

use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::cli::chat::token_counter::TokenCounter;
use crate::cli::chat::context::{calc_max_context_files_size, ContextFilePath};
use crate::cli::chat::util::drop_matched_context_files;
use crate::os::Os;
use super::types::ResourceOperation;
use super::managers::{ContextResourceManager, KnowledgeResourceManager};
use super::renderer::{CliRenderer, ResourceRenderer};
use super::manager::{ResourceHandler, ResourceManager};

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
        r#type: String,
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
        r#type: String,
    },
    /// Show all resources
    Show {
        /// Show detailed content of each resource
        #[arg(long)]
        expand: bool,
        /// Storage type to show: "pinned", "indexed", or "all"
        #[arg(long, default_value = "all")]
        r#type: String,
    },
    /// Show background operation status
    Status,
    /// Clear resources
    Clear {
        /// Confirm the clear operation
        #[arg(long)]
        confirm: bool,
        /// Storage type to clear: "pinned", "indexed", or "all"
        #[arg(long, default_value = "all")]
        r#type: String,
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
            Self::Add { name, value, r#type } => {
                handle_add(os, session, name, value, r#type).await
            }
            Self::Remove { id, name, path, r#type } => {
                handle_remove(os, session, id, name, path, r#type).await
            }
            Self::Show { expand, r#type } => {
                handle_show(os, session, expand, r#type).await
            }
            Self::Status => {
                handle_status(os, session).await
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

async fn handle_add(os: &Os, session: &mut ChatSession, name: String, value: String, storage_type: String) -> Result<ChatState, ChatError> {
        match storage_type.as_str() {
            "pinned" => {
                // Use context manager for pinned resources
                let Some(context_manager) = &mut session.conversation.context_manager else {
                    eprintln!("No context manager available");
                    return Ok(ChatState::PromptUser { skip_printing_tools: true });
                };
                
                let mut handler = ResourceHandler::new(
                    ContextResourceManager::new(context_manager, os),
                    CliRenderer::new()
                );
                
                let resource_op = ResourceOperation::Add { name, value };
                match handler.handle(resource_op, super::OutputFormat::Table).await {
                    Ok(_) => {
                        // Use CliRenderer to render directly to session.stderr with colors
                        let renderer = CliRenderer::new();
                        let success_data = super::ResourceData::Success("Added to pinned resources".to_string());
                        renderer.render_with_session(&success_data, session)?;
                    },
                    Err(e) => {
                        let renderer = CliRenderer::new();
                        let error_data = super::ResourceData::Success(format!("Error: {}", e));
                        renderer.render_with_session(&error_data, session)?;
                    },
                }
            }
            "indexed" => {
                // Use knowledge manager for indexed resources
                let agent = session.conversation.agents.get_active();
                let knowledge_manager = match KnowledgeResourceManager::new(os, agent).await {
                    Ok(manager) => manager,
                    Err(e) => {
                        eprintln!("Failed to initialize knowledge manager: {}", e);
                        return Ok(ChatState::PromptUser { skip_printing_tools: true });
                    }
                };
                
                let mut handler = ResourceHandler::new(
                    knowledge_manager,
                    CliRenderer::new()
                );
                
                let resource_op = ResourceOperation::Add { name, value };
                match handler.handle(resource_op, super::OutputFormat::Table).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            _ => {
                eprintln!("Invalid storage type '{}'. Use 'pinned' or 'indexed'", storage_type);
            }
        }
        
        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

async fn handle_remove(os: &Os, session: &mut ChatSession, id: Option<String>, name: Option<String>, path: Option<String>, storage_type: String) -> Result<ChatState, ChatError> {
        match storage_type.as_str() {
            "pinned" => {
                let Some(context_manager) = &mut session.conversation.context_manager else {
                    eprintln!("No context manager available");
                    return Ok(ChatState::PromptUser { skip_printing_tools: true });
                };
                
                let mut handler = ResourceHandler::new(
                    ContextResourceManager::new(context_manager, os),
                    CliRenderer::new()
                );
                
                let resource_op = ResourceOperation::Remove { id, name, path };
                match handler.handle(resource_op, super::OutputFormat::Table).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            "indexed" => {
                let agent = session.conversation.agents.get_active();
                let knowledge_manager = match KnowledgeResourceManager::new(os, agent).await {
                    Ok(manager) => manager,
                    Err(e) => {
                        eprintln!("Failed to initialize knowledge manager: {}", e);
                        return Ok(ChatState::PromptUser { skip_printing_tools: true });
                    }
                };
                
                let mut handler = ResourceHandler::new(
                    knowledge_manager,
                    CliRenderer::new()
                );
                
                let resource_op = ResourceOperation::Remove { id, name, path };
                match handler.handle(resource_op, super::OutputFormat::Table).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            "all" => {
                // Try both storage types
                println!("Removing from both pinned and indexed resources...");
                Box::pin(handle_remove(os, session, id.clone(), name.clone(), path.clone(), "pinned".to_string())).await?;
                Box::pin(handle_remove(os, session, id, name, path, "indexed".to_string())).await?;
            }
            _ => {
                eprintln!("Invalid storage type '{}'. Use 'pinned', 'indexed', or 'all'", storage_type);
            }
        }
        
        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

async fn handle_search(os: &Os, session: &mut ChatSession, query: String, context_id: Option<String>, storage_type: String) -> Result<ChatState, ChatError> {
        match storage_type.as_str() {
            "indexed" | "all" => {
                // Only indexed resources support search
                let agent = session.conversation.agents.get_active();
                let knowledge_manager = match KnowledgeResourceManager::new(os, agent).await {
                    Ok(manager) => manager,
                    Err(e) => {
                        eprintln!("Failed to initialize knowledge manager: {}", e);
                        return Ok(ChatState::PromptUser { skip_printing_tools: true });
                    }
                };
                
                let mut handler = ResourceHandler::new(
                    knowledge_manager,
                    CliRenderer::new()
                );
                
                let resource_op = ResourceOperation::Search { query, context_id };
                match handler.handle(resource_op, super::OutputFormat::Table).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            "pinned" => {
                eprintln!("Search is not supported for pinned resources. Use 'indexed' or 'all'");
            }
            _ => {
                eprintln!("Invalid storage type '{}'. Use 'indexed' or 'all' for search", storage_type);
            }
        }
        
        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

async fn handle_show(os: &Os, session: &mut ChatSession, expand: bool, storage_type: String) -> Result<ChatState, ChatError> {
    let renderer = CliRenderer::new();
    
    match storage_type.as_str() {
        "pinned" => {
            handle_show_pinned(os, session, &renderer).await?;
        }
        "indexed" => {
            handle_show_indexed(os, session, expand, &renderer).await?;
        }
        "all" => {
            Box::pin(handle_show(os, session, expand, "pinned".to_string())).await?;
            Box::pin(handle_show(os, session, expand, "indexed".to_string())).await?;
        }
        _ => {
            let error_data = super::ResourceData::Success(format!("Invalid storage type '{}'. Use 'pinned', 'indexed', or 'all'", storage_type));
            renderer.render_with_session(&error_data, session)?;
        }
    }
    
    Ok(ChatState::PromptUser { skip_printing_tools: true })
}

async fn handle_show_pinned(os: &Os, session: &mut ChatSession, renderer: &CliRenderer) -> Result<(), ChatError> {
    let Some(context_manager) = &mut session.conversation.context_manager else {
        let error_data = super::ResourceData::Success("No context manager available".to_string());
        renderer.render_with_session(&error_data, session)?;
        return Ok(());
    };
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
    
        agent_files.push(super::ContextPath {
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
    
        session_files.push(super::ContextPath {
            path: path.get_path_as_str().to_string(),
            match_count: context_files.len(),
        });
    
        if !context_files.is_empty() {
            profile_context_files
                .extend(context_files.into_iter().map(|(path, content)| (path, content, true)));
        }
    }
    let matched_files: Vec<super::MatchedFile> = profile_context_files.into_iter().map(|(path, content, is_temporary)| {
        let tokens = TokenCounter::count_tokens(&content);
        super::MatchedFile {
            filename: path,
            content,
            tokens,
            is_temporary,
        }
    }).collect();
    let total_tokens = matched_files.iter().map(|f| f.tokens).sum();
    let context_files_max_size = calc_max_context_files_size(session.conversation.model_info.as_ref());
    let mut files_as_vec = matched_files
        .iter()
        .map(|f| (f.filename.clone(), f.content.clone()))
        .collect::<Vec<_>>();
    let dropped_files = drop_matched_context_files(&mut files_as_vec, context_files_max_size).ok();
    let pinned_data = super::PinnedResourceData {
        agent_files,
        session_files,
        matched_files,
        total_tokens,
        dropped_files,
        context_files_max_size,
    };
    let data = super::ResourceData::PinnedResources(pinned_data);
    renderer.render_with_session(&data, session)?;
    Ok(())
}

async fn handle_show_indexed(os: &Os, session: &mut ChatSession, expand: bool, renderer: &CliRenderer) -> Result<(), ChatError> {
    let agent = session.conversation.agents.get_active();
    Ok(match super::managers::KnowledgeResourceManager::new(os, agent).await {
        Ok(mut manager) => {
            let resource_op = super::ResourceOperation::Show { expand };
            match manager.execute(resource_op).await {
                Ok(data) => {
                    renderer.render_with_session(&data, session)?;
                }
                Err(e) => {
                    let error_data = super::ResourceData::Success(format!("Error: {}", e));
                    renderer.render_with_session(&error_data, session)?;
                }
            }
        }
        Err(e) => {
            let error_data = super::ResourceData::Success(format!("Failed to initialize knowledge manager: {}", e));
            renderer.render_with_session(&error_data, session)?;
        }
    })
}

async fn handle_status(os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        // Status only applies to indexed resources (background operations)
        let agent = session.conversation.agents.get_active();
        let knowledge_manager = match KnowledgeResourceManager::new(os, agent).await {
            Ok(manager) => manager,
            Err(e) => {
                eprintln!("Failed to initialize knowledge manager: {}", e);
                return Ok(ChatState::PromptUser { skip_printing_tools: true });
            }
        };
        
        let mut handler = ResourceHandler::new(
            knowledge_manager,
            CliRenderer::new()
        );
        
        let resource_op = ResourceOperation::Status;
        match handler.handle(resource_op, super::OutputFormat::Table).await {
            Ok(output) => println!("{}", output),
            Err(e) => eprintln!("Error: {}", e),
        }
        
        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

async fn handle_clear(os: &Os, session: &mut ChatSession, confirm: bool, storage_type: String) -> Result<ChatState, ChatError> {
        match storage_type.as_str() {
            "pinned" => {
                let Some(context_manager) = &mut session.conversation.context_manager else {
                    eprintln!("No context manager available");
                    return Ok(ChatState::PromptUser { skip_printing_tools: true });
                };
                
                let mut handler = ResourceHandler::new(
                    ContextResourceManager::new(context_manager, os),
                    CliRenderer::new()
                );
                
                let resource_op = ResourceOperation::Clear { confirm };
                match handler.handle(resource_op, super::OutputFormat::Table).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            "indexed" => {
                let agent = session.conversation.agents.get_active();
                let knowledge_manager = match KnowledgeResourceManager::new(os, agent).await {
                    Ok(manager) => manager,
                    Err(e) => {
                        eprintln!("Failed to initialize knowledge manager: {}", e);
                        return Ok(ChatState::PromptUser { skip_printing_tools: true });
                    }
                };
                
                let mut handler = ResourceHandler::new(
                    knowledge_manager,
                    CliRenderer::new()
                );
                
                let resource_op = ResourceOperation::Clear { confirm };
                match handler.handle(resource_op, super::OutputFormat::Table).await {
                    Ok(output) => println!("{}", output),
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            "all" => {
                // Clear both storage types
                println!("Clearing both pinned and indexed resources...");
                Box::pin(handle_clear(os, session, confirm, "pinned".to_string())).await?;
                Box::pin(handle_clear(os, session, confirm, "indexed".to_string())).await?;
            }
            _ => {
                eprintln!("Invalid storage type '{}'. Use 'pinned', 'indexed', or 'all'", storage_type);
            }
        }
        
        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

async fn handle_cancel(os: &Os, session: &mut ChatSession, operation_id: String) -> Result<ChatState, ChatError> {
        // Cancel only applies to indexed resources (background operations)
        let agent = session.conversation.agents.get_active();
        let knowledge_manager = match KnowledgeResourceManager::new(os, agent).await {
            Ok(manager) => manager,
            Err(e) => {
                eprintln!("Failed to initialize knowledge manager: {}", e);
                return Ok(ChatState::PromptUser { skip_printing_tools: true });
            }
        };
        
        let mut handler = ResourceHandler::new(
            knowledge_manager,
            CliRenderer::new()
        );
        
        let resource_op = ResourceOperation::Cancel { operation_id };
        match handler.handle(resource_op, super::OutputFormat::Table).await {
            Ok(output) => println!("{}", output),
            Err(e) => eprintln!("Error: {}", e),
        }
        
        Ok(ChatState::PromptUser { skip_printing_tools: true })
}

impl ResourceNewCommand {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add { .. } => "add",
            Self::Remove { .. } => "remove",
            Self::Search { .. } => "search",
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
