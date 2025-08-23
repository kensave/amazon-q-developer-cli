use crossterm::{execute, style::{self, Color}};

use crate::cli::chat::cli::ChatSession;
use super::types::{ResourceData, PinnedResourceData, IndexedResourceData, OutputFormat};

// Styling helpers to eliminate DRY violations
fn print_colored(session: &mut ChatSession, text: &str, color: Color) -> Result<(), std::io::Error> {
    execute!(session.stderr, 
        style::SetForegroundColor(color),
        style::Print(text),
        style::SetForegroundColor(Color::Reset)
    )
}

fn print_bold_colored(session: &mut ChatSession, text: &str, color: Color) -> Result<(), std::io::Error> {
    execute!(session.stderr,
        style::SetAttribute(style::Attribute::Bold),
        style::SetForegroundColor(color),
        style::Print(text),
        style::SetAttribute(style::Attribute::Reset),
        style::SetForegroundColor(Color::Reset)
    )
}

/// Trait for rendering resource data in different formats
pub trait ResourceRenderer {
    /// Render resource data in the specified format
    fn render(&self, data: &ResourceData, format: OutputFormat) -> String;

    /// Render to session with colors and styling (for CLI renderers)
    fn render_with_session(&self, data: &ResourceData, _: &mut ChatSession) -> Result<(), std::io::Error> {
        // Default implementation just prints the rendered string
        let output = self.render(data, OutputFormat::PlainText);
        println!("{}", output);
        Ok(())
    }
}

/// CLI renderer for colored terminal output
pub struct CliRenderer;

impl ResourceRenderer for CliRenderer {
    fn render(&self, data: &ResourceData, _format: OutputFormat) -> String {
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
                print_colored(session, &format!("\n{}\n\n", msg), Color::Green)?;
            }
            ResourceData::PinnedResources(pinned_data) => {
                self.render_pinned_resources(pinned_data, session)?;
            }
            ResourceData::IndexedResources(indexed_data) => {
                self.render_indexed_resources(indexed_data, session)?;
            }
            ResourceData::Status(status) => {
                print_colored(session, &format!("Total items: {}\n", status.total_items), Color::Green)?;

                for op in &status.active_operations {
                    execute!(session.stderr, style::Print("  "))?;
                    print_colored(session, &op.id, Color::Blue)?;
                    print_colored(session, " • ", Color::DarkGrey)?;
                    print_colored(session, &op.status, Color::Yellow)?;
                    execute!(session.stderr, style::Print("\n"))?;
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
        print_bold_colored(session, "📌 Pinned Resources:\n", Color::Magenta)?;

        // Agent section
        if let Some(context_manager) = &session.conversation.context_manager {
            print_bold_colored(session, &format!("  👤 Agent ({}):\n", context_manager.current_profile), Color::Magenta)?;
        }

        if data.agent_files.is_empty() {
            print_colored(session, "        <none>\n\n", Color::DarkGrey)?;
        } else {
            for path in &data.agent_files {
                execute!(session.stderr, style::Print(format!("        {} ", path.path)))?;
                print_colored(session, &format!("({} match{})", path.match_count, if path.match_count == 1 { "" } else { "es" }), Color::Green)?;

                if path.match_count > 0 {
                    let tokens: usize = data.matched_files
                        .iter()
                        .filter(|f| !f.is_temporary && f.filename.contains(&path.path))
                        .map(|f| f.tokens)
                        .sum();
                    print_colored(session, &format!(" • ~{} tkns", tokens), Color::DarkGrey)?;
                }
                execute!(session.stderr, style::Print("\n"))?;
            }
            execute!(session.stderr, style::Print("\n"))?;
        }

        // Session section
        print_bold_colored(session, "  💬 Session (temporary):\n", Color::Magenta)?;

        if data.session_files.is_empty() {
            print_colored(session, "        <none>\n\n", Color::DarkGrey)?;
        } else {
            for path in &data.session_files {
                execute!(session.stderr, style::Print(format!("        {} ", path.path)))?;
                print_colored(session, &format!("({} match{})", path.match_count, if path.match_count == 1 { "" } else { "es" }), Color::Green)?;

                if path.match_count > 0 {
                    let tokens: usize = data.matched_files
                        .iter()
                        .filter(|f| f.is_temporary && f.filename.contains(&path.path))
                        .map(|f| f.tokens)
                        .sum();
                    print_colored(session, &format!(" • ~{} tkns", tokens), Color::DarkGrey)?;
                }
                execute!(session.stderr, style::Print("\n"))?;
            }
            execute!(session.stderr, style::Print("\n"))?;
        }

        if !data.matched_files.is_empty() {
            execute!(session.stderr, style::Print(format!("Total: ~{} tokens\n\n", data.total_tokens)))?;
        }

        // Show dropped files warning
        if let Some(dropped_files) = &data.dropped_files {
            if !dropped_files.is_empty() {
                print_colored(session, &format!(
                    "Total token count exceeds limit: {}. The following files will be automatically dropped when interacting with Q. Consider removing them.\n\n",
                    data.context_files_max_size
                ), Color::DarkYellow)?;
                
                let total_files = dropped_files.len();
                let truncated_dropped_files = &dropped_files[..std::cmp::min(10, total_files)];

                for (filename, content) in truncated_dropped_files {
                    let est_tokens = crate::cli::chat::token_counter::TokenCounter::count_tokens(content);
                    execute!(session.stderr, style::Print(format!("{} ", filename)))?;
                    print_colored(session, &format!("(~{} tkns)\n", est_tokens), Color::DarkGrey)?;
                }

                if total_files > 10 {
                    execute!(session.stderr, style::Print(format!("({} more files)\n", total_files - 10)))?;
                }
                execute!(session.stderr, style::Print("\n"))?;
            }
        }

        Ok(())
    }

    fn render_indexed_resources(&self, data: &IndexedResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
        print_bold_colored(session, "🔍 Indexed Resources:\n", Color::Magenta)?;

        if let Some(context_manager) = &session.conversation.context_manager {
            print_bold_colored(session, &format!("  👤 Agent ({}):\n", context_manager.current_profile), Color::Magenta)?;
        }

        if data.items.is_empty() {
            print_colored(session, "        <none>\n\n", Color::DarkGrey)?;
        } else {
            for item in &data.items {
                // Main entry line
                execute!(session.stderr, style::Print("        📂 "))?;
                execute!(session.stderr, style::SetAttribute(style::Attribute::Bold))?;
                print_colored(session, &item.name, Color::Grey)?;
                print_colored(session, &format!(" ({})", &item.id[..8.min(item.id.len())]), Color::Green)?;
                execute!(session.stderr, style::SetAttribute(style::Attribute::Reset), style::Print("\n"))?;

                // Description line
                if let Some(content) = &item.content {
                    if !content.is_empty() {
                        let description = if content.len() > 100 {
                            format!("{}...", &content[..100])
                        } else {
                            content.clone()
                        };
                        execute!(session.stderr, style::Print("           "))?;
                        print_colored(session, &format!("{}\n", description.lines().next().unwrap_or("")), Color::Grey)?;
                    }
                }

                // Stats line
                execute!(session.stderr, style::Print("           "))?;
                print_colored(session, &format!("{} items", item.metadata.size), Color::Green)?;
                print_colored(session, " • ", Color::DarkGrey)?;
                print_colored(session, &item.metadata.resource_type, Color::Blue)?;
                print_colored(session, " • ", Color::DarkGrey)?;
                print_colored(session, &format!("{}\n", item.metadata.updated_at.format("%m/%d %H:%M")), Color::DarkGrey)?;
            }
            execute!(session.stderr, style::Print("\n"))?;
        }

        Ok(())
    }
}
