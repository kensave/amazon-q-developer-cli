use crossterm::{execute, style::{self, Color}};

use crate::cli::chat::ChatSession;
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
            ResourceData::Info(msg) => msg.clone(),
            ResourceData::PinnedResources(pinned_data) => {
                format!("{} pinned files", pinned_data.matched_files.len())
            }
            ResourceData::IndexedResources(indexed_data) => {
                format!("{} indexed resources", indexed_data.items.len())
            }
        }
    }

    fn render_with_session(&self, data: &ResourceData, session: &mut ChatSession) -> Result<(), std::io::Error> {
        match data {
            ResourceData::Success(msg) => {
                print_colored(session, &format!("\n{}\n\n", msg), Color::Green)?;
            }
            ResourceData::Info(msg) => {
                print_colored(session, &format!("\n{}\n\n", msg), Color::Yellow)?;
            }
            ResourceData::PinnedResources(pinned_data) => {
                self.render_pinned_resources(pinned_data, session)?;
            }
            ResourceData::IndexedResources(indexed_data) => {
                self.render_indexed_resources(indexed_data, session)?;
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
                print_colored(session, &item.path.as_ref().unwrap_or(&item.name), Color::Grey)?;
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
                
                // Extract embedding type from name format: "Name (EmbeddingType)"
                let resource_type_display = if let Some(start) = item.name.rfind(" (") {
                    if let Some(end) = item.name[start..].find(")") {
                        let embedding_type = &item.name[start + 2..start + end];
                        format!("indexed ({})", embedding_type.to_lowercase())
                    } else {
                        "indexed".to_string()
                    }
                } else {
                    "indexed".to_string()
                };
                
                print_colored(session, &resource_type_display, Color::Blue)?;
                print_colored(session, " • ", Color::DarkGrey)?;
                print_colored(session, &format!("{}\n", item.metadata.updated_at.format("%m/%d %H:%M")), Color::DarkGrey)?;
            }
            execute!(session.stderr, style::Print("\n"))?;
        }

        Ok(())
    }
}
pub struct ToolRenderer;

impl ToolRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl ResourceRenderer for ToolRenderer {
    fn render(&self, data: &ResourceData, _format: OutputFormat) -> String {
        match data {
            ResourceData::Success(msg) => msg.clone(),
            ResourceData::Info(msg) => msg.clone(),
            ResourceData::PinnedResources(pinned) => {
                if pinned.session_files.is_empty() && pinned.agent_files.is_empty() {
                    "No pinned resources found.".to_string()
                } else {
                    let mut output = String::new();
                    output.push_str("📌 Pinned Resources:\n");
                    
                    // Agent files
                    if !pinned.agent_files.is_empty() {
                        output.push_str("👤 Agent files:\n");
                        for path in &pinned.agent_files {
                            output.push_str(&format!("  • {} ({} matches)\n", path.path, path.match_count));
                        }
                    }
                    
                    // Session files  
                    if !pinned.session_files.is_empty() {
                        output.push_str("💬 Session files:\n");
                        for path in &pinned.session_files {
                            output.push_str(&format!("  • {} ({} matches)\n", path.path, path.match_count));
                        }
                    }
                    
                    output.push_str(&format!("Total tokens: {}", pinned.total_tokens));
                    output
                }
            }
            ResourceData::IndexedResources(indexed) => {
                if indexed.items.is_empty() {
                    "No indexed resources found.".to_string()
                } else {
                    let mut output = format!("🔍 Indexed Resources ({}):\n", indexed.items.len());
                    for item in &indexed.items {
                        let status = match item.metadata.resource_type.as_str() {
                            "indexing" => " (indexing)",
                            _ => ""
                        };
                        output.push_str(&format!("• {} ({}){}\n", item.name, &item.id[..8.min(item.id.len())], status));
                        output.push_str(&format!("  Path: {}\n", item.name)); // Show full name/path for removal
                        output.push_str(&format!("  {} items • {} • {}\n", 
                            item.metadata.size, 
                            if item.metadata.resource_type == "indexed" {
                                // Extract embedding type from name
                                if let Some(start) = item.name.rfind("(Some(") {
                                    if let Some(end) = item.name[start..].find("))") {
                                        let embedding_type = &item.name[start + 6..start + end];
                                        format!("indexed ({})", embedding_type)
                                    } else {
                                        item.metadata.resource_type.clone()
                                    }
                                } else {
                                    item.metadata.resource_type.clone()
                                }
                            } else {
                                item.metadata.resource_type.clone()
                            },
                            item.metadata.updated_at.format("%m/%d %H:%M")
                        ));
                    }
                    output
                }
            }
        }
    }

    fn render_with_session(&self, _data: &ResourceData, _session: &mut ChatSession) -> Result<(), std::io::Error> {
        // Tools don't use session rendering, just return the string
        Ok(())
    }
}
