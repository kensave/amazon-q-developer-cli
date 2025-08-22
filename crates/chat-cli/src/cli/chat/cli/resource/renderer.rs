use crossterm::{execute, queue, style::{self, Color}};

use crate::cli::chat::cli::ChatSession;
use super::types::{ResourceData, PinnedResourceData, IndexedResourceData, OutputFormat};

/// Trait for rendering resource data in different formats
pub trait ResourceRenderer {
    /// Render resource data in the specified format
    fn render(&self, data: &ResourceData, format: OutputFormat) -> String;

    /// Render to session with colors and styling (for CLI renderers)
    fn render_with_session(&self, data: &ResourceData, _: &mut ChatSession) -> Result<(), std::io::Error> {
        // Default implementation just prints the rendered string
        let output = self.render(data, OutputFormat::Table);
        println!("{}", output);
        Ok(())
    }
}

/// CLI renderer for colored terminal output
pub struct CliRenderer;

impl ResourceRenderer for CliRenderer {
    fn render(&self, data: &ResourceData, _format: OutputFormat) -> String {
        // Fallback string rendering for when we can't use session.stderr
        match data {
            ResourceData::Success(msg) => msg.clone(),
            ResourceData::PinnedResources(pinned_data) => {
                format!("{} pinned files", pinned_data.matched_files.len())
            }
            ResourceData::IndexedResources(indexed_data) => {
                format!("{} indexed resources", indexed_data.items.len())
            }
            ResourceData::Status(status) => {
                format!("Total: {} items, {} operations",
                    status.total_items,
                    status.active_operations.len()
                )
            }
        }
    }

    fn render_with_session(&self, data: &ResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
        match data {
            ResourceData::Success(msg) => {
                execute!(
                    session.stderr,
                    style::SetForegroundColor(Color::Green),
                    style::Print(format!("\n{}\n\n", msg)),
                    style::SetForegroundColor(Color::Reset)
                )?;
            }
            ResourceData::PinnedResources(pinned_data) => {
                self.render_pinned_resources(pinned_data, session)?;
            }
            ResourceData::IndexedResources(indexed_data) => {
                self.render_indexed_resources(indexed_data, session)?;
            }
            ResourceData::Status(status) => {
                execute!(
                    session.stderr,
                    style::SetForegroundColor(Color::Green),
                    style::Print(format!("Total items: {}\n", status.total_items)),
                    style::SetForegroundColor(Color::Reset)
                )?;

                if !status.active_operations.is_empty() {
                    for op in &status.active_operations {
                        execute!(
                            session.stderr,
                            style::Print("  "),
                            style::SetForegroundColor(Color::Blue),
                            style::Print(&op.id),
                            style::SetForegroundColor(Color::DarkGrey),
                            style::Print(" • "),
                            style::SetForegroundColor(Color::Yellow),
                            style::Print(&op.status),
                            style::SetForegroundColor(Color::Reset),
                            style::Print("\n")
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl CliRenderer {
    pub fn new() -> Self {
        Self
    }

    fn render_pinned_resources(&self, data: &PinnedResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
        execute!(
            session.stderr,
            style::SetAttribute(style::Attribute::Bold),
            style::SetForegroundColor(Color::Magenta),
            style::Print("📌 Pinned Resources:\n"),
            style::SetAttribute(style::Attribute::Reset),
            style::SetForegroundColor(Color::Reset)
        )?;

        // Agent section
        if let Some(context_manager) = &session.conversation.context_manager {
            execute!(
                session.stderr,
                style::SetAttribute(style::Attribute::Bold),
                style::SetForegroundColor(Color::Magenta),
                style::Print(format!("  👤 Agent ({}):\n", context_manager.current_profile)),
                style::SetAttribute(style::Attribute::Reset),
                style::SetForegroundColor(Color::Reset)
            )?;
        }

        if data.agent_files.is_empty() {
            execute!(
                session.stderr,
                style::SetForegroundColor(Color::DarkGrey),
                style::Print("        <none>\n\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
        } else {
            for path in &data.agent_files {
                execute!(
                    session.stderr,
                    style::Print(format!("        {} ", path.path)),
                    style::SetForegroundColor(Color::Green),
                    style::Print(format!("({} match{})", path.match_count, if path.match_count == 1 { "" } else { "es" })),
                    style::SetForegroundColor(Color::Reset)
                )?;
                
                // Only show tokens if there are matches
                if path.match_count > 0 {
                    let tokens: usize = data.matched_files
                        .iter()
                        .filter(|f| !f.is_temporary && f.filename.contains(&path.path))
                        .map(|f| f.tokens)
                        .sum();
                    
                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print(format!(" • ~{} tkns", tokens)),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                }
                
                execute!(session.stderr, style::Print("\n"))?;
            }
            execute!(session.stderr, style::Print("\n"))?;
        }

        // Session section
        execute!(
            session.stderr,
            style::SetAttribute(style::Attribute::Bold),
            style::SetForegroundColor(Color::Magenta),
            style::Print("  💬 Session (temporary):\n"),
            style::SetAttribute(style::Attribute::Reset),
            style::SetForegroundColor(Color::Reset)
        )?;

        if data.session_files.is_empty() {
            execute!(
                session.stderr,
                style::SetForegroundColor(Color::DarkGrey),
                style::Print("        <none>\n\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
        } else {
            for path in &data.session_files {
                execute!(
                    session.stderr,
                    style::Print(format!("        {} ", path.path)),
                    style::SetForegroundColor(Color::Green),
                    style::Print(format!("({} match{})", path.match_count, if path.match_count == 1 { "" } else { "es" })),
                    style::SetForegroundColor(Color::Reset)
                )?;
                
                // Only show tokens if there are matches
                if path.match_count > 0 {
                    let tokens: usize = data.matched_files
                        .iter()
                        .filter(|f| f.is_temporary && f.filename.contains(&path.path))
                        .map(|f| f.tokens)
                        .sum();
                    
                    execute!(
                        session.stderr,
                        style::SetForegroundColor(Color::DarkGrey),
                        style::Print(format!(" • ~{} tkns", tokens)),
                        style::SetForegroundColor(Color::Reset)
                    )?;
                }
                
                execute!(session.stderr, style::Print("\n"))?;
            }
            execute!(session.stderr, style::Print("\n"))?;
        }

        // Just show total tokens if any files matched
        if !data.matched_files.is_empty() {
            execute!(
                session.stderr,
                style::Print(format!("Total: ~{} tokens\n\n", data.total_tokens))
            )?;
        }

        Ok(())
    }

    fn render_indexed_resources(&self, data: &IndexedResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
        execute!(
            session.stderr,
            style::SetAttribute(style::Attribute::Bold),
            style::SetForegroundColor(Color::Magenta),
            style::Print("🔍 Indexed Resources:\n"),
            style::SetAttribute(style::Attribute::Reset),
            style::SetForegroundColor(Color::Reset)
        )?;

        if let Some(context_manager) = &session.conversation.context_manager {
            execute!(
                session.stderr,
                style::SetAttribute(style::Attribute::Bold),
                style::SetForegroundColor(Color::Magenta),
                style::Print(format!("  👤 Agent ({}):\n", context_manager.current_profile)),
                style::SetAttribute(style::Attribute::Reset),
                style::SetForegroundColor(Color::Reset)
            )?;
        }

        if data.items.is_empty() {
            execute!(
                session.stderr,
                style::SetForegroundColor(Color::DarkGrey),
                style::Print("        <none>\n\n"),
                style::SetForegroundColor(Color::Reset)
            )?;
        } else {
            for item in &data.items {
                // Main entry line with icon, name and ID
                queue!(
                    session.stderr,
                    style::Print(format!("        📂 ")),
                    style::SetAttribute(style::Attribute::Bold),
                    style::SetForegroundColor(Color::Grey),
                    style::Print(&item.name),
                    style::SetForegroundColor(Color::Green),
                    style::Print(format!(" ({})", &item.id[..8.min(item.id.len())])),
                    style::SetAttribute(style::Attribute::Reset),
                    style::SetForegroundColor(Color::Reset),
                    style::Print("\n")
                )?;

                // Description line
                if let Some(content) = &item.content {
                    if !content.is_empty() {
                        let description = if content.len() > 100 {
                            format!("{}...", &content[..100])
                        } else {
                            content.clone()
                        };
                        queue!(
                            session.stderr,
                            style::Print("           "),
                            style::SetForegroundColor(Color::Grey),
                            style::Print(format!("{}\n", description.lines().next().unwrap_or(""))),
                            style::SetForegroundColor(Color::Reset)
                        )?;
                    }
                }

                // Stats line - consistent with pinned format
                queue!(
                    session.stderr,
                    style::Print("           "),
                    style::SetForegroundColor(Color::Green),
                    style::Print(format!("{} items", item.metadata.size)),
                    style::SetForegroundColor(Color::DarkGrey),
                    style::Print(" • "),
                    style::SetForegroundColor(Color::Blue),
                    style::Print(&item.metadata.resource_type),
                    style::SetForegroundColor(Color::DarkGrey),
                    style::Print(" • "),
                    style::SetForegroundColor(Color::DarkGrey),
                    style::Print(format!("{}\n", item.metadata.updated_at.format("%m/%d %H:%M"))),
                    style::SetForegroundColor(Color::Reset)
                )?;
            }
            execute!(session.stderr, style::Print("\n"))?;
        }

        Ok(())
    }
}