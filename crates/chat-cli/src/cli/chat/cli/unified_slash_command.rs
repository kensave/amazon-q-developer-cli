// ABOUTME: Modified SlashCommand enum that integrates unified context system
// ABOUTME: Provides backward compatibility while enabling the new unified context interface

use clap::{Parser, Subcommand};
use eyre::Result;

use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::os::Os;

use super::agent::AgentSubcommand;
use super::clear::ClearArgs;
use super::compact::CompactArgs;
use super::context::ContextSubcommand;
use super::editor::EditorArgs;
use super::hooks::HooksArgs;
use super::issue;
use super::knowledge::KnowledgeSubcommand;
use super::mcp::McpArgs;
use super::model::ModelArgs;
use super::persist::PersistSubcommand;
use super::prompts::PromptsArgs;
use super::subscribe::SubscribeArgs;
use super::resource::ResourceSubcommand;
use super::tools::ToolsArgs;
use super::unified_context::UnifiedContextSubcommand;
use super::usage::UsageArgs;

const AGENT_MIGRATION_DOC_URL: &str = "https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-agents.html";
const EXTRA_HELP: &str = "Additional help text here";

/// Enhanced SlashCommand enum with unified context system
#[derive(Debug, PartialEq, Parser)]
#[command(color = clap::ColorChoice::Always, term_width = 0, after_long_help = EXTRA_HELP)]
pub enum UnifiedSlashCommand {
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
    /// Manage context files (original functionality)
    #[command(subcommand)]
    Context(ContextSubcommand),
    /// Manage resources (pinned and indexed)
    #[command(subcommand)]
    Resource(ResourceSubcommand),
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

impl UnifiedSlashCommand {
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
                // Use original context system
                subcommand.execute(os, session).await
            },
            Self::Resource(subcommand) => {
                // Use the new resource system
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
                    style::Print("Examples:\n"),
                    style::Print("  /context add --knowledge file.md\n"),
                    style::Print("  /context show --knowledge-only\n"),
                    style::Print("  /context status\n\n"),
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
            Self::Resource(_) => "resource",
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
            Self::Resource(subcommand) => Some(subcommand.name()),
            Self::Knowledge(subcommand) => Some(subcommand.name()),
            Self::Agent(subcommand) => Some(subcommand.name()),
            Self::Persist(subcommand) => Some(subcommand.name()),
            _ => None,
        }
    }
}

/// Compatibility wrapper to maintain existing API
pub type SlashCommand = UnifiedSlashCommand;

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    #[command(name = "test")]
    struct TestCli {
        #[command(subcommand)]
        command: UnifiedSlashCommand,
    }

    #[test]
    fn test_unified_context_parsing() {
        // Test new unified context commands
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "add",
            "--knowledge",
            "--index-type=fast",
            "file.md"
        ]);
        assert!(result.is_ok());

        let result = TestCli::try_parse_from(&[
            "test", 
            "context",
            "show",
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

        // Test that old context commands still parse
        let result = TestCli::try_parse_from(&[
            "test",
            "context",
            "show"
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_command_names() {
        let context_cmd = UnifiedSlashCommand::Context(
            UnifiedContextSubcommand::Show {
                expand: false,
                knowledge_only: false,
                context_only: false,
            }
        );
        assert_eq!(context_cmd.command_name(), "context");
        assert_eq!(context_cmd.subcommand_name(), Some("show"));
    }
}
