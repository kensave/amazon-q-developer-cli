use clap::{Subcommand, ValueEnum};
use crossterm::{queue, style};
use eyre::Result;

use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::os::Os;

use super::context::ContextSubcommand;
use super::knowledge::KnowledgeSubcommand;

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum StorageType {
    Indexed,
    Pinned,
}

#[derive(Clone, Debug, PartialEq, Eq, Subcommand)]
pub enum ResourceSubcommand {
    Show {
        #[arg(long)]
        expand: bool,
        #[arg(long, value_enum, ignore_case = true)]
        r#type: Option<StorageType>,
    },
    Add {
        #[arg(short, long)]
        force: bool,
        #[arg(long, value_enum, ignore_case = true)]
        r#type: Option<StorageType>,
        #[arg(required = true)]
        paths: Vec<String>,
    },
    #[command(alias = "rm")]
    Remove {
        #[arg(required = true)]
        paths: Vec<String>,
    },
    Clear {
        #[arg(long, value_enum, ignore_case = true)]
        r#type: Option<StorageType>,
    },
    Status,
}

impl ResourceSubcommand {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Show { .. } => "show",
            Self::Add { .. } => "add",
            Self::Remove { .. } => "remove",
            Self::Clear { .. } => "clear",
            Self::Status => "status",
        }
    }

    pub async fn execute(self, os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        match self {
            Self::Show { expand, r#type } => {
                match r#type {
                    Some(StorageType::Indexed) => KnowledgeSubcommand::Show.execute(os, session).await,
                    Some(StorageType::Pinned) => ContextSubcommand::Show { expand }.execute(os, session).await,
                    None => {
                        // Show both with clear separation
                        queue!(
                            session.stderr,
                            style::SetForegroundColor(style::Color::Cyan),
                            style::Print("📌 Pinned Resources (always included):\n"),
                            style::ResetColor
                        )?;
                        
                        ContextSubcommand::Show { expand }.execute(os, session).await?;
                        
                        // Context command ends with \n\n, so we don't need extra spacing
                        queue!(
                            session.stderr,
                            style::SetForegroundColor(style::Color::Green),
                            style::Print("🔍 Indexed Resources (retrieved on demand):\n"),
                            style::ResetColor
                        )?;
                        
                        KnowledgeSubcommand::Show.execute(os, session).await
                    }
                }
            },
            Self::Add { force, r#type, paths } => {
                match r#type.unwrap_or(StorageType::Pinned) {
                    StorageType::Indexed => {
                        KnowledgeSubcommand::Add { 
                            path: paths.join(" "), 
                            include: vec![], 
                            exclude: vec![], 
                            index_type: None 
                        }.execute(os, session).await
                    },
                    StorageType::Pinned => {
                        ContextSubcommand::Add { paths, force }.execute(os, session).await
                    },
                }
            },
            Self::Remove { paths } => {
                let _ = ContextSubcommand::Remove { paths: paths.clone() }.execute(os, session).await;
                // Remove first path from knowledge (knowledge only supports single path)
                if let Some(first_path) = paths.first() {
                    KnowledgeSubcommand::Remove { path: first_path.clone() }.execute(os, session).await
                } else {
                    Ok(ChatState::PromptUser { skip_printing_tools: true })
                }
            },
            Self::Clear { r#type } => {
                match r#type {
                    Some(StorageType::Indexed) => KnowledgeSubcommand::Clear.execute(os, session).await,
                    Some(StorageType::Pinned) => ContextSubcommand::Clear.execute(os, session).await,
                    None => {
                        ContextSubcommand::Clear.execute(os, session).await?;
                        KnowledgeSubcommand::Clear.execute(os, session).await
                    }
                }
            },
            Self::Status => KnowledgeSubcommand::Status.execute(os, session).await,
        }
    }
}
