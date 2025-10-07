// ABOUTME: Tests for continuation ID billing functionality
// ABOUTME: Ensures continuation IDs are generated and propagated correctly for billing consistency

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use uuid;

    use super::super::ConversationState;
    use crate::cli::agent::Agents;
    use crate::cli::chat::tool_manager::ToolManager;
    use crate::os::Os;

    #[tokio::test]
    async fn test_continuation_id_generation() {
        // Setup
        let os = Os::new().await.unwrap();
        let agents = Agents::default();
        let tool_manager = ToolManager::default();
        let mut conversation = ConversationState::new(
            "test-conversation",
            agents,
            HashMap::new(),
            tool_manager,
            None,
            &os,
            true,
        )
        .await;

        // Initially no continuation ID
        assert!(conversation.continuation_id().is_none());

        // Set user message should generate continuation ID
        conversation.set_next_user_message("test message".to_string()).await;

        let continuation_id = conversation.continuation_id();
        assert!(continuation_id.is_some());
        assert!(!continuation_id.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_continuation_id_uniqueness() {
        // Setup
        let os = Os::new().await.unwrap();
        let agents = Agents::default();
        let tool_manager = ToolManager::default();
        let mut conversation = ConversationState::new(
            "test-conversation",
            agents,
            HashMap::new(),
            tool_manager,
            None,
            &os,
            true,
        )
        .await;

        // Generate first continuation ID
        conversation.set_next_user_message("first message".to_string()).await;
        let first_id = conversation.continuation_id().unwrap().to_string();

        // Reset and generate second continuation ID
        conversation.reset_next_user_message();
        conversation.set_next_user_message("second message".to_string()).await;
        let second_id = conversation.continuation_id().unwrap().to_string();

        // Should be different
        assert_ne!(first_id, second_id);
    }

    #[tokio::test]
    async fn test_continuation_id_format() {
        // Setup
        let os = Os::new().await.unwrap();
        let agents = Agents::default();
        let tool_manager = ToolManager::default();
        let mut conversation = ConversationState::new(
            "test-conversation",
            agents,
            HashMap::new(),
            tool_manager,
            None,
            &os,
            true,
        )
        .await;

        // Generate continuation ID
        conversation.set_next_user_message("test message".to_string()).await;
        let continuation_id = conversation.continuation_id().unwrap();

        // Should be a valid UUID format (36 characters with hyphens)
        assert_eq!(continuation_id.len(), 36);
        assert!(continuation_id.chars().filter(|&c| c == '-').count() == 4);

        // Should be parseable as UUID
        assert!(uuid::Uuid::parse_str(continuation_id).is_ok());
    }

    #[tokio::test]
    async fn test_continuation_id_in_api_request() {
        // Setup
        let os = Os::new().await.unwrap();
        let agents = Agents::default();
        let tool_manager = ToolManager::default();
        let mut conversation = ConversationState::new(
            "test-conversation",
            agents,
            HashMap::new(),
            tool_manager,
            None,
            &os,
            true,
        )
        .await;

        // Set user message to generate continuation ID
        conversation.set_next_user_message("test message".to_string()).await;
        let expected_continuation_id = conversation.continuation_id().unwrap().to_string();

        // Create sendable conversation state (this is what gets sent to API)
        let mut stderr = std::io::stderr();
        let sendable_state = conversation
            .as_sendable_conversation_state(&os, &mut stderr, false)
            .await
            .unwrap();

        // Verify continuation ID is included in the API request
        assert_eq!(
            sendable_state.agent_continuation_id.as_deref(),
            Some(expected_continuation_id.as_str())
        );
    }
}
