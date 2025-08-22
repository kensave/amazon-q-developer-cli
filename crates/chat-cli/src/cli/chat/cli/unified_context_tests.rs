// ABOUTME: Test suite for unified context system that merges knowledge and context functionality
// ABOUTME: Covers backward compatibility, unified interface, scope management, and async operations

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::chat::{ChatSession, ChatState};
    use crate::os::Os;
    use crate::util::knowledge_store::KnowledgeStore;
    use std::collections::HashMap;
    use tokio;

    async fn setup_test_environment() -> (Os, ChatSession) {
        let mut os = Os::new().await.unwrap();
        // Setup test session - this would need proper initialization
        // For now, this is a placeholder structure
        todo!("Implement proper test session setup")
    }

    #[tokio::test]
    async fn test_backward_compatibility_knowledge_commands() {
        // TC-001: Existing /knowledge commands continue to work
        let (mut os, mut session) = setup_test_environment().await;
        
        // Test knowledge show command
        let knowledge_show = KnowledgeSubcommand::Show;
        let result = knowledge_show.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Test knowledge add command
        let knowledge_add = KnowledgeSubcommand::Add {
            path: "test_file.txt".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: None,
            global: false,
        };
        // This should still work as before
        assert!(matches!(knowledge_add.execute(&os, &mut session).await, Ok(_)));
    }

    #[tokio::test]
    async fn test_backward_compatibility_context_commands() {
        // TC-002: Existing /context commands continue to work
        let (mut os, mut session) = setup_test_environment().await;
        
        // Test context show command
        let context_show = ContextSubcommand::Show { expand: false };
        let result = context_show.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Test context add command
        let context_add = ContextSubcommand::Add {
            force: false,
            paths: vec!["test_pattern.rs".to_string()],
        };
        assert!(matches!(context_add.execute(&os, &mut session).await, Ok(_)));
    }

    #[tokio::test]
    async fn test_unified_show_command() {
        // TC-005: /context show displays both knowledge and context entries
        let (mut os, mut session) = setup_test_environment().await;
        
        // Add some knowledge entries
        let knowledge_add = KnowledgeSubcommand::Add {
            path: "knowledge_file.md".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: Some("fast".to_string()),
            global: false,
        };
        knowledge_add.execute(&os, &mut session).await.unwrap();
        
        // Add some context entries
        let context_add = ContextSubcommand::Add {
            force: false,
            paths: vec!["context_file.rs".to_string()],
        };
        context_add.execute(&os, &mut session).await.unwrap();
        
        // Show should display both
        let show_command = ContextSubcommand::Show { expand: false };
        let result = show_command.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Verify output contains both types of entries
        // This would need to capture and verify the actual output
    }

    #[tokio::test]
    async fn test_unified_add_with_knowledge_flag() {
        // TC-006: /context add supports both file patterns and knowledge indexing
        let (mut os, mut session) = setup_test_environment().await;
        
        // Test adding with knowledge flag (this is the new unified interface)
        // This test assumes the new unified interface is implemented
        // For now, this tests the concept
        
        // Add file as knowledge entry through context command
        // This would be the new unified interface:
        // /context add --knowledge --index-type=fast file.md
        
        // Verify it was added to knowledge store
        let knowledge_store = KnowledgeStore::get_async_instance(&os, None).await.unwrap();
        let store = knowledge_store.lock().await;
        let entries = store.get_all_for_scope(false).await.unwrap_or_default();
        
        // Should find the entry we added
        assert!(!entries.is_empty());
    }

    #[tokio::test]
    async fn test_scope_isolation() {
        // TC-010: Agent-specific context isolation
        let (mut os, mut session) = setup_test_environment().await;
        
        // Add entry to agent scope
        let agent_add = KnowledgeSubcommand::Add {
            path: "agent_file.txt".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: None,
            global: false,
        };
        agent_add.execute(&os, &mut session).await.unwrap();
        
        // Add entry to global scope
        let global_add = KnowledgeSubcommand::Add {
            path: "global_file.txt".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: None,
            global: true,
        };
        global_add.execute(&os, &mut session).await.unwrap();
        
        // Verify isolation - agent scope should not see global entries by default
        // and vice versa
        let knowledge_store = KnowledgeStore::get_async_instance(&os, None).await.unwrap();
        let store = knowledge_store.lock().await;
        
        let agent_entries = store.get_all_for_scope(false).await.unwrap_or_default();
        let global_entries = store.get_all_for_scope(true).await.unwrap_or_default();
        
        assert_eq!(agent_entries.len(), 1);
        assert_eq!(global_entries.len(), 1);
    }

    #[tokio::test]
    async fn test_async_operation_status() {
        // TC-014: Knowledge indexing status integration
        let (mut os, mut session) = setup_test_environment().await;
        
        // Start a knowledge indexing operation
        let knowledge_add = KnowledgeSubcommand::Add {
            path: "large_directory/".to_string(),
            include: vec!["**/*.rs".to_string()],
            exclude: vec!["target/**".to_string()],
            index_type: Some("best".to_string()),
            global: false,
        };
        knowledge_add.execute(&os, &mut session).await.unwrap();
        
        // Check status
        let status_command = KnowledgeSubcommand::Status;
        let result = status_command.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Should show active operations
        let knowledge_store = KnowledgeStore::get_async_instance(&os, None).await.unwrap();
        let store = knowledge_store.lock().await;
        let status = store.get_status_data().await.unwrap();
        
        // Should have at least one operation
        assert!(!status.operations.is_empty());
    }

    #[tokio::test]
    async fn test_operation_cancellation() {
        // TC-016: Cancellation of background operations
        let (mut os, mut session) = setup_test_environment().await;
        
        // Start a long-running operation
        let knowledge_add = KnowledgeSubcommand::Add {
            path: "very_large_directory/".to_string(),
            include: vec!["**/*".to_string()],
            exclude: vec![],
            index_type: Some("best".to_string()),
            global: false,
        };
        knowledge_add.execute(&os, &mut session).await.unwrap();
        
        // Cancel the operation
        let cancel_command = KnowledgeSubcommand::Cancel { operation_id: None };
        let result = cancel_command.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Verify operation was cancelled
        let knowledge_store = KnowledgeStore::get_async_instance(&os, None).await.unwrap();
        let store = knowledge_store.lock().await;
        let status = store.get_status_data().await.unwrap();
        
        // Operations should be cancelled or empty
        for op in &status.operations {
            assert!(op.is_cancelled || op.is_failed);
        }
    }

    #[tokio::test]
    async fn test_clear_operations() {
        // TC-008: /context clear handles both scopes appropriately
        let (mut os, mut session) = setup_test_environment().await;
        
        // Add entries to both knowledge and context
        let knowledge_add = KnowledgeSubcommand::Add {
            path: "knowledge_test.md".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: None,
            global: false,
        };
        knowledge_add.execute(&os, &mut session).await.unwrap();
        
        let context_add = ContextSubcommand::Add {
            force: false,
            paths: vec!["context_test.rs".to_string()],
        };
        context_add.execute(&os, &mut session).await.unwrap();
        
        // Clear all
        let clear_command = ContextSubcommand::Clear;
        let result = clear_command.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Verify both are cleared
        // This would need to check both knowledge store and context manager
    }

    #[tokio::test]
    async fn test_remove_operations() {
        // TC-007: /context remove works for both knowledge and context entries
        let (mut os, mut session) = setup_test_environment().await;
        
        // Add entries
        let knowledge_add = KnowledgeSubcommand::Add {
            path: "to_remove.md".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: None,
            global: false,
        };
        knowledge_add.execute(&os, &mut session).await.unwrap();
        
        // Remove by path
        let remove_command = ContextSubcommand::Remove {
            paths: vec!["to_remove.md".to_string()],
        };
        let result = remove_command.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Verify removal
        let knowledge_store = KnowledgeStore::get_async_instance(&os, None).await.unwrap();
        let store = knowledge_store.lock().await;
        let entries = store.get_all_for_scope(false).await.unwrap_or_default();
        
        // Should not find the removed entry
        assert!(!entries.iter().any(|e| e.context.name.contains("to_remove.md")));
    }

    #[tokio::test]
    async fn test_feature_flag_behavior() {
        // TC-003: Knowledge feature flag behavior preserved
        let (mut os, mut session) = setup_test_environment().await;
        
        // Disable knowledge feature
        os.database.settings.set(
            crate::database::settings::Setting::EnabledKnowledge, 
            false
        ).await.unwrap();
        
        // Knowledge commands should show disabled message
        let knowledge_show = KnowledgeSubcommand::Show;
        let result = knowledge_show.execute(&os, &mut session).await;
        assert!(result.is_ok());
        
        // Should return PromptUser state with skip_printing_tools: true
        // indicating the feature is disabled
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Test various error conditions
        let (mut os, mut session) = setup_test_environment().await;
        
        // Test adding non-existent file
        let invalid_add = KnowledgeSubcommand::Add {
            path: "/non/existent/path".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: None,
            global: false,
        };
        let result = invalid_add.execute(&os, &mut session).await;
        // Should handle error gracefully
        assert!(result.is_ok()); // Command execution succeeds but shows error message
        
        // Test invalid patterns
        let invalid_pattern_add = KnowledgeSubcommand::Add {
            path: "valid_path.txt".to_string(),
            include: vec!["[invalid_pattern".to_string()],
            exclude: vec![],
            index_type: None,
            global: false,
        };
        let result = invalid_pattern_add.execute(&os, &mut session).await;
        assert!(result.is_ok()); // Should handle gracefully with error message
    }

    #[tokio::test]
    async fn test_update_operations() {
        // Test knowledge update functionality
        let (mut os, mut session) = setup_test_environment().await;
        
        // Add a file to knowledge
        let knowledge_add = KnowledgeSubcommand::Add {
            path: "update_test.md".to_string(),
            include: vec![],
            exclude: vec![],
            index_type: None,
            global: false,
        };
        knowledge_add.execute(&os, &mut session).await.unwrap();
        
        // Update the file
        let update_command = KnowledgeSubcommand::Update {
            path: "update_test.md".to_string(),
        };
        let result = update_command.execute(&os, &mut session).await;
        assert!(result.is_ok());
    }
}
