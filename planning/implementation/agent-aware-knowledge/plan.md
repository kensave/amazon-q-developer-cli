# Agent-Aware Knowledge Implementation Plan

## Test Strategy

### Test Scenarios
1. **Multi-Client Management Tests**
   - Test agent client creation only when agent context available
   - Test global client creation on-demand for --global operations
   - Test client lifecycle and resource management

2. **Directory Structure Tests**
   - Test agent-specific directory creation
   - Test global directory creation only when needed
   - Test directory resolution for different agents

3. **CLI Flag Tests**
   - Test `knowledge add <name> <path>` creates agent-specific knowledge (default)
   - Test `knowledge add <name> <path> --global` creates global knowledge
   - Test operations fail gracefully when no agent context and no --global flag

4. **Migration Tests**
   - Test migration of existing knowledge bases to current agent directory
   - Test backward compatibility after migration
   - Test no data loss during migration

5. **Agent Context Tests**
   - Test knowledge isolation between different agents
   - Test global knowledge accessible from all agents when --global used
   - Test search includes both agent and global knowledge when both clients exist
   - Test operations when agent context is unavailable

6. **Multi-Client Search Tests**
   - Test search across both agent and global clients
   - Test result merging and ranking from multiple clients
   - Test search when only one client is available

7. **Error Handling Tests**
   - Test operations without agent context and without --global flag
   - Test permission errors during directory creation
   - Test migration failure scenarios
   - Test client initialization failures

### Expected Test Behaviors
- All existing knowledge functionality continues to work
- New --global flag properly routes to global directory
- Agent-specific knowledge is isolated per agent
- Migration preserves all existing data
- Search results include both agent and global knowledge appropriately

## Implementation Plan

### Phase 1: Directory Structure Updates
- [ ] Update `directories.rs` with agent-aware and global directory functions
- [ ] Add agent-aware path resolution
- [ ] Maintain backward compatibility

### Phase 2: Multi-Client KnowledgeStore Architecture
- [ ] Refactor KnowledgeStore to manage multiple AsyncSemanticSearchClient instances
- [ ] Add agent client (created only when agent context available)
- [ ] Add global client (created on-demand for --global operations)
- [ ] Implement client routing logic based on operation type

### Phase 3: CLI Command Updates
- [ ] Add --global flag to knowledge add command
- [ ] Update CLI to pass agent context and global flag to KnowledgeStore
- [ ] Update all commands to route to appropriate client
- [ ] Update command help text and documentation

### Phase 4: Migration Implementation
- [ ] Implement migration logic for existing knowledge bases to current agent directory
- [ ] Add migration to KnowledgeStore initialization
- [ ] Ensure safe migration with rollback capability

### Phase 5: Search Integration
- [ ] Update search to query both agent and global clients when available
- [ ] Merge and rank results from multiple clients
- [ ] Ensure proper result attribution

### Phase 6: Integration and Testing
- [ ] Update knowledge tool integration
- [ ] Comprehensive testing of multi-client scenarios
- [ ] Test agent availability edge cases

## Architecture Decisions

### Multi-Client Architecture
- **Agent-specific client**: One AsyncSemanticSearchClient per agent directory
- **Global client**: One AsyncSemanticSearchClient for global_knowledge directory  
- **Client management**: KnowledgeStore manages multiple clients internally
- **Agent availability**: Only create agent-specific client if current agent is available
- **Default behavior**: Nothing is global by default - all operations are agent-specific unless --global flag is used

### Agent Identification
- Use `agent.name` from current agent context (same as context feature)
- Pass agent name through CLI commands to KnowledgeStore
- If no agent context available, only global operations allowed with --global flag

### Directory Structure
- `~/.aws/amazonq/knowledge_bases/global_knowledge/` for shared knowledge (only with --global)
- `~/.aws/amazonq/knowledge_bases/{agent_name}/` for agent-specific knowledge (default)
- Existing knowledge bases migrate to current agent's directory (not global)

### CLI Interface
- `knowledge add <name> <path>` - adds to current agent's knowledge (default)
- `knowledge add <name> <path> --global` - adds to global knowledge
- All operations default to agent-specific unless --global specified
- Search includes both agent and global knowledge when both clients exist

### Client Lifecycle
- Agent client: Created only when agent context is available
- Global client: Created on-demand when --global operations are used
- Lazy initialization of clients to avoid unnecessary resource usage

## Risk Mitigation
- Comprehensive backup strategy during migration
- Gradual rollout with feature flag support
- Extensive testing of edge cases
- Clear error messages and recovery instructions
