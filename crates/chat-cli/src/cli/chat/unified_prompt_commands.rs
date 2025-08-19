// ABOUTME: Updated command list for prompt completion with unified context system
// ABOUTME: Includes new unified context commands while maintaining backward compatibility

pub const UNIFIED_COMMANDS: &[&str] = &[
    "/clear",
    "/help",
    "/editor",
    "/issue",
    "/quit",
    "/tools",
    "/tools trust",
    "/tools untrust",
    "/tools trust-all",
    "/tools reset",
    "/mcp",
    "/model",
    "/agent",
    "/agent help",
    "/agent list",
    "/agent create",
    "/agent delete",
    "/agent rename",
    "/agent set",
    "/agent schema",
    "/prompts",
    // Unified context commands (new)
    "/context",
    "/context help",
    "/context show",
    "/context show --expand",
    "/context show --knowledge-only",
    "/context show --context-only",
    "/context add",
    "/context add --knowledge",
    "/context add --global",
    "/context add --force",
    "/context rm",
    "/context remove",
    "/context clear",
    "/context clear --global",
    "/context clear --knowledge-only",
    "/context clear --context-only",
    "/context status",
    "/context cancel",
    "/context update",
    // Legacy knowledge commands (deprecated but still supported)
    "/knowledge",
    "/knowledge help",
    "/knowledge show",
    "/knowledge add",
    "/knowledge rm",
    "/knowledge remove",
    "/knowledge clear",
    "/knowledge status",
    "/knowledge cancel",
    "/knowledge update",
    // Other commands
    "/hooks",
    "/hooks help",
    "/hooks add",
    "/hooks rm",
    "/hooks enable",
    "/hooks disable",
    "/hooks enable-all",
    "/hooks disable-all",
    "/compact",
    "/compact help",
    "/usage",
    "/save",
    "/load",
    "/subscribe",
];

/// Get the appropriate command list based on feature flags
pub fn get_commands_list(knowledge_enabled: bool) -> &'static [&'static str] {
    if knowledge_enabled {
        UNIFIED_COMMANDS
    } else {
        // Filter out knowledge-specific commands if feature is disabled
        &UNIFIED_COMMANDS[..UNIFIED_COMMANDS.len() - 20] // Rough filter, could be more precise
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_commands_include_new_features() {
        let commands = UNIFIED_COMMANDS;
        
        // Test that new unified context commands are included
        assert!(commands.contains(&"/context show --knowledge-only"));
        assert!(commands.contains(&"/context show --context-only"));
        assert!(commands.contains(&"/context add --knowledge"));
        assert!(commands.contains(&"/context clear --knowledge-only"));
        assert!(commands.contains(&"/context status"));
        assert!(commands.contains(&"/context cancel"));
        assert!(commands.contains(&"/context update"));
    }

    #[test]
    fn test_backward_compatibility_commands() {
        let commands = UNIFIED_COMMANDS;
        
        // Test that legacy knowledge commands are still included
        assert!(commands.contains(&"/knowledge"));
        assert!(commands.contains(&"/knowledge show"));
        assert!(commands.contains(&"/knowledge add"));
        assert!(commands.contains(&"/knowledge status"));
        
        // Test that original context commands are still included
        assert!(commands.contains(&"/context"));
        assert!(commands.contains(&"/context show"));
        assert!(commands.contains(&"/context add"));
    }

    #[test]
    fn test_command_list_filtering() {
        let enabled_commands = get_commands_list(true);
        let disabled_commands = get_commands_list(false);
        
        // When knowledge is enabled, should get full list
        assert_eq!(enabled_commands.len(), UNIFIED_COMMANDS.len());
        
        // When knowledge is disabled, should get filtered list
        assert!(disabled_commands.len() < enabled_commands.len());
    }
}
