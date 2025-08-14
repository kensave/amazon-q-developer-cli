# Agent-Aware Knowledge Implementation Context

## Project Structure
- **Project Type**: Rust workspace with multiple crates
- **Main CLI Crate**: `crates/chat-cli`
- **Build System**: Cargo
- **Repository Root**: `/Volumes/workplace/QCLI/amazon-q-developer-cli`

## Requirements
- Implement agent-aware knowledge base functionality
- Create separate folders per agent in knowledge_bases directory
- Add global_knowledge folder for shared knowledge bases
- Support --global flag when adding knowledge bases
- Migrate existing knowledge bases to root directory structure
- Maintain backward compatibility

## Existing Documentation
- **CONTRIBUTING.md**: Standard GitHub contribution guidelines, requires feature requests before major changes
- **README.md**: Installation and development setup instructions
- **Missing DEVELOPMENT.md**: Should be created with Rust-specific build instructions

## Current Knowledge Implementation
Key files identified:
- `crates/chat-cli/src/util/knowledge_store.rs` - Core knowledge storage logic
- `crates/chat-cli/src/cli/chat/tools/knowledge.rs` - Knowledge tool implementation  
- `crates/chat-cli/src/cli/chat/cli/knowledge.rs` - Knowledge CLI commands
- `crates/chat-cli/src/util/directories.rs` - Directory utilities (knowledge_bases_dir function)
- `docs/knowledge-management.md` - Knowledge management documentation

**Current Knowledge Directory Structure:**
- Base directory: `~/.aws/amazonq/knowledge_bases/`
- All knowledge bases stored in single flat directory
- No agent separation currently implemented

## Current Context Implementation (Agent-Aware Pattern)
Key files identified:
- `crates/chat-cli/src/cli/chat/cli/context.rs` - Context CLI commands with agent awareness
- `crates/chat-cli/src/cli/chat/context.rs` - Context functionality with ContextManager
- `crates/chat-cli/src/cli/agent/mod.rs` - Agent struct with name field
- `crates/chat-cli/src/cli/agent/legacy/context.rs` - Legacy agent context

**Agent Awareness Pattern from Context:**
- ContextManager has `current_profile: String` field (agent.name)
- ContextFilePath enum distinguishes Agent vs Session paths
- Context CLI shows "👤 Agent ({agent_name})" and "💬 Session (temporary)" sections
- Agent-owned vs session-owned context files are tracked separately

## Implementation Paths
1. **Update directory structure in directories.rs**
   - Modify `knowledge_bases_dir()` to support agent-aware paths
   - Add `agent_knowledge_dir()` and `global_knowledge_dir()` functions

2. **Extend KnowledgeStore for agent awareness**
   - Add agent parameter to KnowledgeStore methods
   - Update directory resolution logic to use agent-specific or global paths
   - Maintain backward compatibility with existing knowledge bases

3. **Update Knowledge CLI commands**
   - Add `--global` flag to `add` command
   - Update all commands to work with current agent context
   - Add agent identification to CLI operations

4. **Implement migration logic**
   - Move existing knowledge bases to global_knowledge directory
   - Preserve existing functionality during migration

5. **Update knowledge tool integration**
   - Ensure knowledge tool works with agent-aware storage
   - Update search functionality to include both agent and global knowledge

## Target Directory Structure
```
~/.aws/amazonq/knowledge_bases/
├── global_knowledge/           # Shared across all agents
│   └── [existing knowledge bases moved here]
├── agent_1/                   # Agent-specific knowledge
│   └── [agent 1 knowledge bases]
└── agent_2/                   # Agent-specific knowledge
    └── [agent 2 knowledge bases]
```

## Dependencies
- Rust toolchain (stable + nightly for formatting)
- Cargo for build and test
- Standard Rust testing framework

## Patterns to Follow
- Follow existing CLI command patterns in chat-cli crate
- Use similar agent awareness patterns as context feature
- Maintain existing knowledge API compatibility where possible
