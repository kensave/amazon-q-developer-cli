# Unified Context System - Implementation Plan

## Test Strategy

### Test Scenarios

#### 1. Backward Compatibility Tests
- **TC-001**: Existing `/knowledge` commands continue to work
- **TC-002**: Existing `/context` commands continue to work  
- **TC-003**: Knowledge feature flag behavior preserved
- **TC-004**: Context session behavior preserved

#### 2. Unified Interface Tests
- **TC-005**: `/context show` displays both knowledge and context entries
- **TC-006**: `/context add` supports both file patterns and knowledge indexing
- **TC-007**: `/context remove` works for both knowledge and context entries
- **TC-008**: `/context clear` handles both scopes appropriately
- **TC-009**: `/context status` shows unified progress information

#### 3. Scope Management Tests
- **TC-010**: Agent-specific context isolation
- **TC-011**: Global context sharing
- **TC-012**: Session-temporary context handling
- **TC-013**: Scope precedence and conflict resolution

#### 4. Async Processing Tests
- **TC-014**: Knowledge indexing status integration
- **TC-015**: Progress reporting in unified interface
- **TC-016**: Cancellation of background operations
- **TC-017**: Error handling during async operations

#### 5. Agent Configuration Tests
- **TC-018**: Agent schema knowledge configuration
- **TC-019**: Automatic directory indexing
- **TC-020**: Configuration validation and error handling

### Test Data Strategy
- Mock file systems with various file types
- Sample agent configurations with knowledge settings
- Test knowledge bases with different embedding types
- Context files with various patterns and sizes

## Implementation Architecture

### Unified Command Structure

```rust
pub enum ContextSubcommand {
    /// Display context and knowledge entries
    Show {
        #[arg(long)]
        expand: bool,
        #[arg(long)]
        knowledge_only: bool,
        #[arg(long)]
        context_only: bool,
    },
    /// Add files or directories to context/knowledge
    Add {
        #[arg(short, long)]
        force: bool,
        #[arg(long)]
        knowledge: bool,
        #[arg(long)]
        global: bool,
        #[arg(long, action = clap::ArgAction::Append)]
        include: Vec<String>,
        #[arg(long, action = clap::ArgAction::Append)]
        exclude: Vec<String>,
        #[arg(long)]
        index_type: Option<String>,
        #[arg(required = true)]
        paths: Vec<String>,
    },
    /// Remove context or knowledge entries
    Remove {
        #[arg(long)]
        global: bool,
        #[arg(required = true)]
        paths: Vec<String>,
    },
    /// Clear all context and knowledge entries
    Clear {
        #[arg(long)]
        global: bool,
        #[arg(long)]
        knowledge_only: bool,
        #[arg(long)]
        context_only: bool,
    },
    /// Show status of background operations
    Status,
    /// Cancel background operations
    Cancel {
        operation_id: Option<String>,
    },
    /// Update knowledge entries
    Update {
        path: String,
    },
    /// Manage hooks (deprecated, redirects to agent config)
    #[command(hide = true)]
    Hooks,
}
```

### Backend Integration Strategy

#### 1. Unified Context Manager
- Extend `ContextManager` to integrate with `KnowledgeStore`
- Add knowledge-aware methods to existing context operations
- Maintain separate storage but unified interface

#### 2. Command Routing Logic
```rust
impl ContextSubcommand {
    pub async fn execute(self, os: &Os, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        match self {
            Self::Show { expand, knowledge_only, context_only } => {
                self.handle_unified_show(os, session, expand, knowledge_only, context_only).await
            },
            Self::Add { knowledge, paths, .. } => {
                if knowledge {
                    self.handle_knowledge_add(os, session, ..).await
                } else {
                    self.handle_context_add(os, session, ..).await
                }
            },
            // ... other commands
        }
    }
}
```

#### 3. Status Integration
- Merge knowledge operation status with context information
- Unified progress reporting in show command
- Consistent error handling across both systems

### Migration Strategy

#### Phase 1: Extend Context Command
1. Add knowledge-related flags to existing context subcommands
2. Integrate knowledge operations behind feature flags
3. Maintain full backward compatibility

#### Phase 2: Unified Display
1. Modify `/context show` to display both types of entries
2. Add filtering options for knowledge-only or context-only views
3. Integrate status information from knowledge operations

#### Phase 3: Deprecate Knowledge Command
1. Mark `/knowledge` command as deprecated
2. Add migration warnings and guidance
3. Eventually remove knowledge command entirely

### Agent Configuration Schema

```json
{
  "knowledge": {
    "conversation": {
      "persistent": true,
      "description": "Store important information from this conversation here"
    },
    "local-documentation": {
      "source": "file://~/path/to/documentation/**.md",
      "description": "Project documentation",
      "embedding": "fast",
      "chunkingStrategy": {
        "type": "line",
        "chunkSize": 50
      }
    }
  }
}
```

## Implementation Steps

### Step 1: Extend Context Command Interface
- Add knowledge-related flags to context subcommands
- Implement feature detection and routing logic
- Add comprehensive error handling

### Step 2: Integrate Knowledge Backend
- Modify context command execution to call knowledge operations
- Implement unified show command with both data sources
- Add status integration for async operations

### Step 3: Implement Agent Configuration
- Add knowledge schema support to agent configuration
- Implement automatic directory indexing
- Add validation and error handling

### Step 4: Add Backward Compatibility Layer
- Ensure existing workflows continue to work
- Add deprecation warnings for knowledge command
- Implement migration guidance

### Step 5: Testing and Validation
- Comprehensive test suite covering all scenarios
- Performance testing with large knowledge bases
- User experience validation

## Risk Mitigation

### Technical Risks
- **Async/Sync Integration**: Careful handling of async knowledge operations in sync context flows
- **Performance**: Ensure unified operations don't degrade performance
- **Data Consistency**: Maintain consistency between knowledge and context data

### User Experience Risks
- **Complexity**: Hide complexity behind intuitive interface design
- **Migration**: Provide clear migration path and documentation
- **Feature Discovery**: Make new capabilities discoverable without overwhelming users

### Mitigation Strategies
- Extensive testing with real-world scenarios
- Gradual rollout with feature flags
- Comprehensive documentation and examples
- User feedback integration during development
