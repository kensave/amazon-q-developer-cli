use clap::{Subcommand, ValueEnum};
use eyre::Result;

use crate::cli::chat::{ChatError, ChatSession, ChatState};
use crate::os::Os;

use super::context::ContextSubcommand as OriginalContextSubcommand;
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
                    Some(StorageType::Pinned) => OriginalContextSubcommand::Show { expand }.execute(os, session).await,
                    None => {
                        OriginalContextSubcommand::Show { expand }.execute(os, session).await?;
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
                        OriginalContextSubcommand::Add { paths, force }.execute(os, session).await
                    },
                }
            },
            Self::Remove { paths } => {
                let _ = OriginalContextSubcommand::Remove { paths: paths.clone() }.execute(os, session).await;
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
                    Some(StorageType::Pinned) => OriginalContextSubcommand::Clear.execute(os, session).await,
                    None => {
                        OriginalContextSubcommand::Clear.execute(os, session).await?;
                        KnowledgeSubcommand::Clear.execute(os, session).await
                    }
                }
            },
            Self::Status => KnowledgeSubcommand::Status.execute(os, session).await,
        }
    }
}
