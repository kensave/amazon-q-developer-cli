// ABOUTME: Updated module declarations and SlashCommand implementation with unified context system
// ABOUTME: Maintains backward compatibility while integrating the new unified context interface

pub mod clear;
pub mod compact;
pub mod context;
pub mod editor;
pub mod hooks;
pub mod knowledge;
pub mod mcp;
pub mod model;
pub mod persist;
pub mod profile;
pub mod prompts;
pub mod subscribe;
pub mod tools;
pub mod unified_context;
pub mod usage;

use clap::Parser;
use clear::ClearArgs;
use compact::CompactArgs;
use context::ContextSubcommand as OriginalContextSubcommand;
use editor::EditorArgs;
use hooks::HooksArgs;
use knowledge::KnowledgeSubcommand;
use mcp::McpArgs;
use model::ModelArgs;
use persist::PersistSubcommand;
use profile::AgentSubcommand;
use prompts::PromptsArgs;
use tools::ToolsArgs;
use unified_context::UnifiedContextSubcommand;

use crate::cli::chat::cli::subscribe::SubscribeArgs;
use crate::cli::chat::cli::usage::UsageArgs;
use crate::cli::chat::consts::AGENT_MIGRATION_DOC_URL;
use crate::cli::chat::{
    ChatError,
    ChatSession,
    ChatState,
    EXTRA_HELP,
};
use crate::cli::issue;
use crate::os::Os;

/// q (Amazon Q Chat) - Enhanced with unified context system
#[derive(Debug, PartialEq, Parser)]
#[command(color = clap::ColorChoice::Always, term_width = 0, after_long_help = EXTRA_HELP)]
pub enum SlashCommand {
    /// Quit the application
    #[command(aliases = ["q", "exit"])]
    Quit,
    /// Clear the conversation history
    Clear(ClearArgs),
    /// Manage agents
    #[command(subcommand)]
    Agent(AgentSubcommand),
    #[command(hide = true)]
    Profile,
    /// Manage context files and knowledge base (unified interface)
    #[command(subcommand)]
    Context(UnifiedContextSubcommand),
    /// (Deprecated) Use /context instead - Legacy knowledge base management
    #[command(subcommand, hide = true)]
    Knowledge(KnowledgeSubcommand),
    /// Open $EDITOR (defaults to vi) to compose a prompt
    #[command(name = "editor")]
    PromptEditor(EditorArgs),
    /// Summarize the conversation to free up context space
    Compact(CompactArgs),
    /// View tools and permissions
    Tools(ToolsArgs),
    /// Create a new Github issue or make a feature request
    Issue(issue::IssueArgs),
    /// View and retrieve prompts
    Prompts(PromptsArgs),
    /// View context hooks
    Hooks(HooksArgs),
    /// Show current session's context window usage
    Usage(UsageArgs),
    /// See mcp server loaded
    Mcp(McpArgs),
    /// Select a model for the current conversation session
    Model(ModelArgs),
    /// Upgrade to a Q Developer Pro subscription for increased query limits
    Subscribe(SubscribeArgs),
    #[command(flatten)]
    Persist(PersistSubcommand),
}

impl SlashCommand {
    pub async fn execute(self, os: &mut Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        match self {
            Self::Quit => Ok(ChatState::Exit),
            Self::Clear(args) => args.execute(session).await,
            Self::Agent(subcommand) => subcommand.execute(os, session).await,
            Self::Profile => {
                use crossterm::{execute, style};
                execute!(
                    session.stderr,
                    style::SetForegroundColor(style::Color::Yellow),
                    style::Print("This command has been deprecated. Use"),
                    style::SetForegroundColor(style::Color::Cyan),
                    style::Print(" /agent "),
                    style::SetForegroundColor(style::Color::Yellow),
                    style::Print("instead.\nSee "),
                    style::Print(AGENT_MIGRATION_DOC_URL),
                    style::Print(" for more detail"),
                    style::Print("\n"),
                    style::ResetColor,
                )?;

                Ok(ChatState::PromptUser {
                    skip_printing_tools: true,
                })
            },
            Self::Context(subcommand) => {
                // Use the new unified context system
                subcommand.execute(os, session).await
            },
            Self::Knowledge(subcommand) => {
                // Show deprecation warning and redirect to unified context
                use crossterm::{queue, style};
                queue!(
                    session.stderr,
                    style::SetForegroundColor(style::Color::Yellow),
                    style::Print("⚠️  The /knowledge command is deprecated. Use /context instead.\n"),
                    style::SetForegroundColor(style::Color::Cyan),
                    style::Print("Migration examples:\n"),
                    style::Print("  /knowledge add file.md        → /context add --knowledge file.md\n"),
                    style::Print("  /knowledge show               → /context show --knowledge-only\n"),
                    style::Print("  /knowledge status             → /context status\n"),
                    style::Print("  /knowledge clear --global     → /context clear --global --knowledge-only\n\n"),
                    style::SetForegroundColor(style::Color::Reset)
                )?;

                // Still execute the command for backward compatibility
                subcommand.execute(os, session).await
            },
            Self::PromptEditor(args) => args.execute(os, session).await,
            Self::Compact(args) => args.execute(os, session).await,
            Self::Tools(args) => args.execute(os, session).await,
            Self::Issue(args) => args.execute(os, session).await,
            Self::Prompts(args) => args.execute(os, session).await,
            Self::Hooks(args) => args.execute(os, session).await,
            Self::Usage(args) => args.execute(os, session).await,
            Self::Mcp(args) => args.execute(os, session).await,
            Self::Model(args) => args.execute(os, session).await,
            Self::Subscribe(args) => args.execute(os, session).await,
            Self::Persist(subcommand) => subcommand.execute(os, session).await,
        }
    }

    pub fn command_name(&self) -> &'static str {
        match self {
            Self::Quit => "quit",
            Self::Clear(_) => "clear",
            Self::Agent(_) => "agent",
            Self::Profile => "profile",
            Self::Context(_) => "context",
            Self::Knowledge(_) => "knowledge",
            Self::PromptEditor(_) => "editor",
            Self::Compact(_) => "compact",
            Self::Tools(_) => "tools",
            Self::Issue(_) => "issue",
            Self::Prompts(_) => "prompts",
            Self::Hooks(_) => "hooks",
            Self::Usage(_) => "usage",
            Self::Mcp(_) => "mcp",
            Self::Model(_) => "model",
            Self::Subscribe(_) => "subscribe",
            Self::Persist(_) => "persist",
        }
    }

    pub fn subcommand_name(&self) -> Option<&'static str> {
        match self {
            Self::Context(subcommand) => Some(subcommand.name()),
            Self::Knowledge(subcommand) => Some(subcommand.name()),
            Self::Agent(subcommand) => Some(subcommand.name()),
            Self::Persist(subcommand) => Some(subcommand.name()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    #[command(name = "test")]
    struct TestCli {
        #[command(subcommand)]
        command: SlashCommand,
    }

    #[test]
    fn test_unified_context_commands() {
        // Test new unified context add with knowledge flag
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "add",
            "--knowledge",
            "--index-type=fast",
            "file.md"
        ]);
        assert!(result.is_ok());

        // Test unified context show with filters
        let result = TestCli::try_parse_from(&[
            "test", 
            "context",
            "show",
            "--knowledge-only"
        ]);
        assert!(result.is_ok());

        let result = TestCli::try_parse_from(&[
            "test", 
            "context",
            "show",
            "--context-only"
        ]);
        assert!(result.is_ok());

        // Test unified context clear with scope options
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "clear",
            "--global",
            "--knowledge-only"
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that old knowledge commands still parse
        let result = TestCli::try_parse_from(&[
            "test",
            "knowledge",
            "show"
        ]);
        assert!(result.is_ok());

        let result = TestCli::try_parse_from(&[
            "test",
            "knowledge",
            "add",
            "file.txt"
        ]);
        assert!(result.is_ok());

        // Test that old context commands still work
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "show"
        ]);
        assert!(result.is_ok());

        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "add",
            "pattern.rs"
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_command_identification() {
        let context_cmd = SlashCommand::Context(
            UnifiedContextSubcommand::Show {
                expand: false,
                knowledge_only: false,
                context_only: false,
            }
        );
        assert_eq!(context_cmd.command_name(), "context");
        assert_eq!(context_cmd.subcommand_name(), Some("show"));

        let knowledge_cmd = SlashCommand::Knowledge(KnowledgeSubcommand::Show);
        assert_eq!(knowledge_cmd.command_name(), "knowledge");
        assert_eq!(knowledge_cmd.subcommand_name(), Some("show"));
    }

    #[test]
    fn test_new_unified_features() {
        // Test status command through unified interface
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "status"
        ]);
        assert!(result.is_ok());

        // Test cancel command through unified interface
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "cancel",
            "operation-id-123"
        ]);
        assert!(result.is_ok());

        // Test update command through unified interface
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "update",
            "file.md"
        ]);
        assert!(result.is_ok());
    }
}
