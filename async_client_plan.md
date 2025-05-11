# Async Semantic Search Client Implementation Plan

This document outlines the step-by-step plan for implementing an asynchronous wrapper around the existing `SemanticSearchClient` to enable background indexing and non-blocking operations.

## Ground Rules
- Don't change the logic of the existing synchronous client
- Don't delete existing methods
- Use "extract until you drop it clean" approach
- Ensure all tests pass after each step
- Update tests as needed to work with the new async functionality

## Implementation Instructions

If this session gets closed at any time, you can continue from where you left off by:
1. Checking which steps are marked as completed in this file
2. Running `cargo build` to see if there are any compilation errors
3. Running `cargo test -p semantic_search_client` to check if tests are passing
4. Continuing with the next uncompleted step

## Implementation Steps

### Step 1: Add Status Enum to MemoryContext ✅
- [x] Add `ContextStatus` enum with `Indexing`, `Ready`, and `Failed` variants
- [x] Update `MemoryContext` struct to include a `status` field
- [x] Update the constructor to accept status parameter
- [x] Update serialization/deserialization if needed
- [x] Update tests to accommodate the new field

### Step 2: Add Empty Context Constructor to SemanticContext ✅
- [x] Add `new_empty` method to `SemanticContext`
- [x] Ensure it creates a valid but empty context
- [x] Verify tests still pass

### Step 3: Add Dependencies to Cargo.toml ✅
- [x] Add tokio with appropriate features
- [x] Add async-trait
- [x] Make dependencies optional with a feature flag
- [x] Verify build succeeds with the new dependencies

### Step 4: Create AsyncSemanticSearchClient Wrapper ✅
- [x] Create a new struct `AsyncSemanticSearchClient` that wraps `SemanticSearchClient`
- [x] Implement basic constructor methods (`new`, `new_with_default_dir`)
- [x] Add necessary dependencies (tokio, async-trait)
- [x] Ensure the wrapper compiles without errors

### Step 5: Implement Basic Async Methods ✅
- [x] Implement `search_all` async method
- [x] Implement `search_context` async method
- [x] Implement `get_all_contexts` async method
- [x] Add tests for these basic methods

### Step 6: Implement Background Indexing Example ✅
- [x] Add example test showing how to use tokio::task::spawn_blocking for background indexing
- [x] Add test with progress tracking for larger datasets
- [x] Update TextEmbedderTrait to implement Sync for thread safety
- [x] Ensure all tests pass

## Progress Tracking

We'll mark each step as completed (✅) once it's implemented and all tests pass.
