// ABOUTME: Unified context system that merges knowledge and context functionality into a single interface
// ABOUTME: Provides backward compatibility while enabling new unified workflows for context management

use clap::Subcommand;
use crossterm::queue;
use crossterm::style::{
    self,
    Color,
};
use eyre::Result;

use crate::cli::chat::{
    ChatError,
    ChatSession,
    ChatState,
};
use crate::database::settings::Setting;
use crate::os::Os;
use crate::util::knowledge_store::KnowledgeStore;

use super::context::ContextSubcommand as OriginalContextSubcommand;
use super::knowledge::KnowledgeSubcommand;

/// Unified context management commands that merge knowledge and context functionality
#[derive(Clone, Debug, PartialEq, Eq, Subcommand)]
pub enum UnifiedContextSubcommand {
    /// Display session context and knowledge base entries
    Show {
        /// Print out each matched file's content, hook configurations, and last session summary
        #[arg(long)]
        expand: bool,
        /// Show only knowledge base entries (retrieved on demand)
        #[arg(long)]
        knowledge_only: bool,
        /// Show only session context entries (always included)
        #[arg(long)]
        session_only: bool,
    },
    /// Add files to session context (always included) or knowledge base (retrieved on demand)
    Add {
        /// Include even if matched files exceed size limits
        #[arg(short, long)]
        force: bool,
        /// Add to knowledge base for semantic retrieval
        #[arg(long)]
        knowledge: bool,
        /// Add to session context (default, always included)
        #[arg(long)]
        session_context: bool,
        /// Include patterns (e.g., `**/*.ts`, `**/*.md`)
        #[arg(long, action = clap::ArgAction::Append)]
        include: Vec<String>,
        /// Exclude patterns (e.g., `node_modules/**`, `target/**`)
        #[arg(long, action = clap::ArgAction::Append)]
        exclude: Vec<String>,
        /// Index type to use (Fast, Best) - only for knowledge entries
        #[arg(long)]
        index_type: Option<String>,
        #[arg(required = true)]
        paths: Vec<String>,
    },
    /// Add content to existing context
    AddItem {
        /// Context ID to add content to
        #[arg(long)]
        context_id: String,
        /// Content to add (text or file path)
        content: String,
    },
    /// Remove specified context or knowledge entries
    #[command(alias = "rm")]
    Remove {
        #[arg(required = true)]
        paths: Vec<String>,
    },
    /// Remove all session context and knowledge base entries
    Clear {
        /// Clear only knowledge base entries
        #[arg(long)]
        knowledge_only: bool,
        /// Clear only session context entries
        #[arg(long)]
        session_only: bool,
    },
    /// Show background operation status
    Status,
    /// Cancel a background operation
    Cancel {
        /// Operation ID to cancel (optional - cancels most recent if not provided)
        operation_id: Option<String>,
    },
    /// Update a knowledge entry
    Update {
        path: String,
    },
    /// Manage hooks (deprecated, redirects to agent config)
    #[command(hide = true)]
    Hooks,
}

impl UnifiedContextSubcommand {
    pub async fn execute(self, os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        match self {
            Self::Show { expand, knowledge_only, session_only } => {
                self.handle_unified_show(os, session, expand, knowledge_only, session_only).await
            },
            Self::Add { force, knowledge, session_context: _, ref include, ref exclude, ref index_type, ref paths } => {
                // Default to session context if neither flag is specified
                if knowledge {
                    self.handle_knowledge_add(os, session, paths.clone(), include.clone(), exclude.clone(), index_type.clone()).await
                } else {
                    // Default behavior: add to session context
                    self.handle_context_add(os, session, paths.clone(), force).await
                }
            },
            Self::AddItem { ref context_id, ref content } => {
                self.handle_add_item(os, session, context_id.clone(), content.clone()).await
            },
            Self::Remove { ref paths } => {
                self.handle_unified_remove(os, session, paths.clone()).await
            },
            Self::Clear { knowledge_only, session_only } => {
                self.handle_unified_clear(os, session, knowledge_only, session_only).await
            },
            Self::Status => {
                self.handle_status(os, session).await
            },
            Self::Cancel { ref operation_id } => {
                self.handle_cancel(os, session, operation_id.as_deref()).await
            },
            Self::Update { ref path } => {
                self.handle_update(os, session, path.clone()).await
            },
            Self::Hooks => {
                self.handle_hooks_redirect(session).await
            },
        }
    }

    async fn handle_unified_show(
        &self,
        os: &Os,
        session: &mut ChatSession,
        expand: bool,
        knowledge_only: bool,
        context_only: bool,
    ) -> Result<ChatState, ChatError> {
        // If specific filter requested, delegate to appropriate handler
        if knowledge_only {
            let knowledge_cmd = KnowledgeSubcommand::Show;
            return knowledge_cmd.execute(os, session).await;
        }
        
        if context_only {
            let context_cmd = OriginalContextSubcommand::Show { expand };
            return context_cmd.execute(os, session).await;
        }

        // Show unified view - both knowledge and context
        self.show_unified_entries(os, session, expand).await
    }

    async fn show_unified_entries(
        &self,
        os: &Os,
        session: &mut ChatSession,
        expand: bool,
    ) -> Result<ChatState, ChatError> {
        // First show context files
        queue!(
            session.stderr,
            style::SetAttribute(crossterm::style::Attribute::Bold),
            style::SetForegroundColor(Color::Magenta),
            style::Print("📁 Context files:\n"),
            style::SetAttribute(crossterm::style::Attribute::Reset),
        )?;

        // Delegate to original context show implementation
        let context_cmd = OriginalContextSubcommand::Show { expand };
        let context_result = context_cmd.execute(os, session).await?;

        // Then show knowledge entries if feature is enabled
        if Self::is_knowledge_enabled(os) {
            queue!(
                session.stderr,
                style::SetAttribute(crossterm::style::Attribute::Bold),
                style::SetForegroundColor(Color::Magenta),
                style::Print("📚 Knowledge Base:\n"),
                style::SetAttribute(crossterm::style::Attribute::Reset),
            )?;

            match self.show_knowledge_entries(os, session).await {
                Ok(_) => {},
                Err(e) => {
                    queue!(
                        session.stderr,
                        style::SetForegroundColor(Color::Red),
                        style::Print(format!("Error accessing knowledge base: {}\n", e)),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                }
            }
        }

        Ok(context_result)
    }

    async fn show_knowledge_entries(
        &self,
        os: &Os,
        session: &mut ChatSession,
    ) -> Result<(), std::io::Error> {
        // Get agent name from session context
        let agent_name = session
            .conversation
            .context_manager
            .as_ref()
            .map(|cm| cm.current_profile.clone());

        // Show agent-specific knowledge
        if let Some(ref agent) = agent_name {
            queue!(
                session.stderr,
                style::SetAttribute(crossterm::style::Attribute::Bold),
                style::SetForegroundColor(Color::Cyan),
                style::Print(format!("  👤 Agent ({}):\n", agent)),
                style::SetAttribute(crossterm::style::Attribute::Reset),
            )?;

            match KnowledgeStore::get_async_instance(os, Some(agent)).await {
                Ok(store) => {
                    let store = store.lock().await;
                    let entries = store.get_all_for_scope(false).await.unwrap_or_default();
                    if entries.is_empty() {
                        queue!(
                            session.stderr,
                            style::SetForegroundColor(Color::DarkGrey),
                            style::Print("        <none>\n"),
                            style::SetForegroundColor(Color::Reset)
                        )?;
                    } else {
                        Self::format_knowledge_entries_with_indent(session, &entries, "        ")?;
                    }
                },
                Err(_) => {
                    queue!(
                        session.stderr,
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print("      <none>\n"),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                },
            }
        }

        Ok(())
    }

    fn format_knowledge_entries_with_indent(
        session: &mut ChatSession,
        knowledge_entries: &[crate::util::knowledge_store::KnowledgeEntry],
        indent: &str,
    ) -> Result<(), std::io::Error> {
        for entry in knowledge_entries {
            let ctx = &entry.context;
            
            // Main entry line with name and ID
            queue!(
                session.stderr,
                style::Print(format!("{}📂 ", indent)),
                style::SetAttribute(style::Attribute::Bold),
                style::SetForegroundColor(Color::Grey),
                style::Print(&ctx.name),
                style::SetForegroundColor(Color::Green),
                style::Print(format!(" ({})", &ctx.id[..8])),
                style::SetAttribute(style::Attribute::Reset),
                style::SetForegroundColor(Color::Reset),
                style::Print("\n")
            )?;

            // Description line
            queue!(
                session.stderr,
                style::Print(format!("{}   ", indent)),
                style::SetForegroundColor(Color::Grey),
                style::Print(format!("{}\n", ctx.description)),
                style::SetForegroundColor(Color::Reset)
            )?;

            // Stats line
            queue!(
                session.stderr,
                style::Print(format!("{}   ", indent)),
                style::SetForegroundColor(Color::Green),
                style::Print(format!("{} items", ctx.item_count)),
                style::SetForegroundColor(Color::DarkGrey),
                style::Print(" • "),
                style::SetForegroundColor(Color::Cyan),
                style::Print(format!("{} tokens", ctx.token_count)),
                style::SetForegroundColor(Color::DarkGrey),
                style::Print(" • "),
                style::SetForegroundColor(Color::Blue),
                style::Print(ctx.embedding_type.description()),
                style::SetForegroundColor(Color::DarkGrey),
                style::Print(" • "),
                style::SetForegroundColor(Color::DarkGrey),
                style::Print(format!("{}", ctx.updated_at.format("%m/%d %H:%M"))),
                style::SetForegroundColor(Color::Reset),
                style::Print("\n\n")
            )?;
        }
        Ok(())
    }

    async fn handle_knowledge_add(
        &self,
        os: &Os,
        session: &mut ChatSession,
        paths: Vec<String>,
        include: Vec<String>,
        exclude: Vec<String>,
        index_type: Option<String>,
    ) -> Result<ChatState, ChatError> {
        if !Self::is_knowledge_enabled(os) {
            Self::write_knowledge_disabled_message(session)?;
            return Ok(ChatState::PromptUser { skip_printing_tools: true });
        }

        // For now, handle only the first path (knowledge add handles one path at a time)
        if let Some(path) = paths.first() {
            let knowledge_cmd = KnowledgeSubcommand::Add {
                path: path.clone(),
                include,
                exclude,
                index_type,
            };
            knowledge_cmd.execute(os, session).await
        } else {
            queue!(
                session.stderr,
                style::SetForegroundColor(Color::Red),
                style::Print("\nError: No paths provided\n\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
            Ok(ChatState::PromptUser { skip_printing_tools: true })
        }
    }

    async fn handle_context_add(
        &self,
        os: &Os,
        session: &mut ChatSession,
        paths: Vec<String>,
        force: bool,
    ) -> Result<ChatState, ChatError> {
        let context_cmd = OriginalContextSubcommand::Add { force, paths };
        context_cmd.execute(os, session).await
    }

    async fn handle_unified_remove(
        &self,
        os: &Os,
        session: &mut ChatSession,
        paths: Vec<String>,
    ) -> Result<ChatState, ChatError> {
        let mut removed_from_knowledge = false;
        let mut removed_from_context = false;

        // Try to remove from knowledge first if enabled
        if Self::is_knowledge_enabled(os) {
            for path in &paths {
                let knowledge_cmd = KnowledgeSubcommand::Remove {
                    path: path.clone(),
                };
                match knowledge_cmd.execute(os, session).await {
                    Ok(_) => removed_from_knowledge = true,
                    Err(_) => {}, // Continue to try context removal
                }
            }
        }

        // Try to remove from context
        let context_cmd = OriginalContextSubcommand::Remove { paths: paths.clone() };
        match context_cmd.execute(os, session).await {
            Ok(_) => removed_from_context = true,
            Err(_) => {},
        }

        if removed_from_knowledge || removed_from_context {
            queue!(
                session.stderr,
                style::SetForegroundColor(Color::Green),
                style::Print(format!("\nRemoved {} path(s) from ", paths.len())),
                style::Print(match (removed_from_knowledge, removed_from_context) {
                    (true, true) => "knowledge and context",
                    (true, false) => "knowledge",
                    (false, true) => "context",
                    (false, false) => unreachable!(),
                }),
                style::Print(".\n\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
        } else {
            queue!(
                session.stderr,
                style::SetForegroundColor(Color::Yellow),
                style::Print(format!("\nNo entries found to remove for: {}\n\n", paths.join(", "))),
                style::SetForegroundColor(Color::Reset)
            )?;
        }

        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

    async fn handle_unified_clear(
        &self,
        os: &Os,
        session: &mut ChatSession,
        knowledge_only: bool,
        context_only: bool,
    ) -> Result<ChatState, ChatError> {
        if knowledge_only {
            if Self::is_knowledge_enabled(os) {
                let knowledge_cmd = KnowledgeSubcommand::Clear;
                return knowledge_cmd.execute(os, session).await;
            } else {
                Self::write_knowledge_disabled_message(session)?;
                return Ok(ChatState::PromptUser { skip_printing_tools: true });
            }
        }

        if context_only {
            let context_cmd = OriginalContextSubcommand::Clear;
            return context_cmd.execute(os, session).await;
        }

        // Clear both knowledge and context
        let mut cleared_knowledge = false;
        let mut cleared_context = false;

        // Clear knowledge if enabled
        if Self::is_knowledge_enabled(os) {
            let knowledge_cmd = KnowledgeSubcommand::Clear;
            match knowledge_cmd.execute(os, session).await {
                Ok(_) => cleared_knowledge = true,
                Err(_) => {},
            }
        }

        // Clear context
        let context_cmd = OriginalContextSubcommand::Clear;
        match context_cmd.execute(os, session).await {
            Ok(_) => cleared_context = true,
            Err(_) => {},
        }

        if cleared_knowledge || cleared_context {
            queue!(
                session.stderr,
                style::SetForegroundColor(Color::Green),
                style::Print("\nCleared "),
                style::Print(match (cleared_knowledge, cleared_context) {
                    (true, true) => "knowledge and context",
                    (true, false) => "knowledge",
                    (false, true) => "context",
                    (false, false) => unreachable!(),
                }),
                style::Print(" entries.\n\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
        }

        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

    async fn handle_status(&self, os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        if Self::is_knowledge_enabled(os) {
            let knowledge_cmd = KnowledgeSubcommand::Status;
            knowledge_cmd.execute(os, session).await
        } else {
            Self::write_knowledge_disabled_message(session)?;
            Ok(ChatState::PromptUser { skip_printing_tools: true })
        }
    }

    async fn handle_cancel(&self, os: &Os, session: &mut ChatSession, operation_id: Option<&str>) -> Result<ChatState, ChatError> {
        if Self::is_knowledge_enabled(os) {
            let knowledge_cmd = KnowledgeSubcommand::Cancel {
                operation_id: operation_id.map(|s| s.to_string()),
            };
            knowledge_cmd.execute(os, session).await
        } else {
            Self::write_knowledge_disabled_message(session)?;
            Ok(ChatState::PromptUser { skip_printing_tools: true })
        }
    }

    async fn handle_update(&self, os: &Os, session: &mut ChatSession, path: String) -> Result<ChatState, ChatError> {
        if Self::is_knowledge_enabled(os) {
            let knowledge_cmd = KnowledgeSubcommand::Update { path };
            knowledge_cmd.execute(os, session).await
        } else {
            Self::write_knowledge_disabled_message(session)?;
            Ok(ChatState::PromptUser { skip_printing_tools: true })
        }
    }

    async fn handle_hooks_redirect(&self, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        use crossterm::{execute, style};
        
        execute!(
            session.stderr,
            style::SetForegroundColor(Color::Yellow),
            style::Print("The /context hooks command is deprecated.\n\nConfigure hooks directly with your agent instead: "),
            style::SetForegroundColor(Color::Green),
            style::Print("https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-agents.html"),
            style::SetForegroundColor(Color::Reset),
            style::Print("\n"),
        )?;

        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

    fn is_knowledge_enabled(os: &Os) -> bool {
        os.database
            .settings
            .get_bool(Setting::EnabledKnowledge)
            .unwrap_or(false)
    }

    fn write_knowledge_disabled_message(session: &mut ChatSession) -> Result<(), std::io::Error> {
        queue!(
            session.stderr,
            style::SetForegroundColor(Color::Red),
            style::Print("\nKnowledge features are disabled. Enable with: q settings chat.enableKnowledge true\n"),
            style::SetForegroundColor(Color::Yellow),
            style::Print("💡 Your knowledge base data is preserved and will be available when re-enabled.\n\n"),
            style::SetForegroundColor(Color::Reset)
        )
    }

    async fn handle_add_item(
        &self,
        os: &Os,
        session: &mut ChatSession,
        context_id: String,
        content: String,
    ) -> Result<ChatState, ChatError> {
        if Self::is_knowledge_enabled(os) {
            let agent_name = session.conversation.agents.get_active().map(|a| a.name.as_str());
            let knowledge_store = KnowledgeStore::get_async_instance(os, agent_name).await
                .map_err(|e| ChatError::Custom(format!("Failed to get knowledge store: {}", e).into()))?;
            
            let store = knowledge_store.lock().await;
            
            // Use the new add_to_context API
            match store.add_to_context(&context_id, &content).await {
                Ok(_) => {
                    queue!(
                        session.stderr,
                        style::SetForegroundColor(Color::Green),
                        style::Print(format!("✓ Added content to context '{}'\n", context_id)),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                },
                Err(e) => {
                    queue!(
                        session.stderr,
                        style::SetForegroundColor(Color::Red),
                        style::Print(format!("✗ Failed to add content to context '{}': {}\n", context_id, e)),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                }
            }
        } else {
            queue!(
                session.stderr,
                style::SetForegroundColor(Color::Yellow),
                style::Print("Knowledge feature is not enabled\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
        }

        Ok(ChatState::PromptUser { skip_printing_tools: true })
    }

    pub fn name(&self) -> &'static str {
        match self {
            UnifiedContextSubcommand::Show { .. } => "show",
            UnifiedContextSubcommand::Add { .. } => "add",
            UnifiedContextSubcommand::AddItem { .. } => "add-item",
            UnifiedContextSubcommand::Remove { .. } => "remove",
            UnifiedContextSubcommand::Clear { .. } => "clear",
            UnifiedContextSubcommand::Status => "status",
            UnifiedContextSubcommand::Cancel { .. } => "cancel",
            UnifiedContextSubcommand::Update { .. } => "update",
            UnifiedContextSubcommand::Hooks => "hooks",
        }
    }
}
