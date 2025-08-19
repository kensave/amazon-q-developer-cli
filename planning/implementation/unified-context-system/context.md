# Unified Context System - Implementation Context

## Project Structure

**Repository**: Amazon Q Developer CLI (Rust-based)
**Main Binary**: `chat_cli` in `crates/chat_cli/`
**Build System**: Cargo (Rust)
**Testing**: `cargo test`
**Linting**: `cargo clippy`
**Formatting**: `cargo +nightly fmt`

## Requirements Analysis

### Core Functionality to Merge
1. **Current /knowledge capabilities**:
   - Indexing with async processing
   - Embedding options (fast/best)
   - Multiple knowledge contexts
   - Global and local knowledge bases
   - Status tracking and progress indicators

2. **Current /context capabilities**:
   - Session management
   - File matching and inclusion
   - Temporary context handling
   - Rule-based context loading

### Unified System Goals
- Abstract complexity from users while maintaining full functionality
- Single command interface for both knowledge and context management
- Agent configuration schema support
- Backward compatibility with existing workflows
- Elegant UX that hides indexing complexity until needed

## Implementation Paths

### Primary Components to Modify
1. **CLI Command Structure**: Merge `/knowledge` and `/context` commands
2. **Backend Services**: Unify knowledge indexing and context management
3. **Configuration Schema**: Support agent-based knowledge configuration
4. **Status Management**: Integrate async processing status into unified interface

### Key Integration Points
- Command parsing and routing
- Knowledge base management
- Context session handling
- Configuration file processing
- Status and progress reporting

## Existing Documentation
- **CONTRIBUTING.md**: Standard contribution guidelines, requires feature coordination
- **README.md**: Build instructions using Cargo, standard Rust development workflow
- **No DEVELOPMENT.md found**: Will follow standard Rust practices

## Dependencies and Patterns
- Rust ecosystem with Cargo build system
- CLI framework (likely clap or similar)
- Async processing for knowledge indexing
- File system operations for context management
- Configuration file handling (likely TOML/JSON)

## Implementation Strategy
1. Analyze existing knowledge and context command implementations
2. Design unified command interface
3. Implement backward-compatible command routing
4. Integrate status and progress reporting
5. Add agent configuration schema support
