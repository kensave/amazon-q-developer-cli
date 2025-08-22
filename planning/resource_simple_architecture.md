# Simplified Resource Architecture

## Core Insight
- **Operations**: add, remove, show, search, clear, status (not just fetching)
- **Managers**: Direct integration with ContextManager & KnowledgeStore
- **Composition**: 2-3 lines to interact with any resource type
- **Dual Use**: Same components for CLI and tool

## Simple Architecture

```rust
// Core operation types
enum ResourceOperation {
    Add { name: String, value: String },
    Remove { id: String },
    Show,
    Search { query: String },
    Clear,
    Status,
}

// Resource type with manager
trait ResourceManager {
    async fn execute(&mut self, op: ResourceOperation) -> Result<ResourceData>;
}

// Output rendering (keep this clean pattern)
trait ResourceRenderer {
    fn render(&self, data: &ResourceData, format: OutputFormat) -> String;
}

// Simple composition
struct ResourceHandler {
    manager: Box<dyn ResourceManager>,
    renderer: Box<dyn ResourceRenderer>,
}

impl ResourceHandler {
    async fn handle(&mut self, op: ResourceOperation, format: OutputFormat) -> Result<String> {
        let data = self.manager.execute(op).await?;
        Ok(self.renderer.render(&data, format))
    }
}
```

## Implementation Plan

### Step 1: Skeleton CLI Command
```rust
// crates/chat-cli/src/cli/chat/cli/resource_new.rs
#[derive(Subcommand)]
pub enum ResourceNewCommand {
    #[command(subcommand)]
    Context(ContextOp),
    #[command(subcommand)] 
    Knowledge(KnowledgeOp),
}

#[derive(Subcommand)]
pub enum ContextOp {
    Add { paths: Vec<String> },
    Remove { paths: Vec<String> },
    Show,
    Clear,
}

#[derive(Subcommand)]
pub enum KnowledgeOp {
    Add { name: String, value: String },
    Remove { id: String },
    Show,
    Search { query: String },
    Clear,
    Status,
}
```

### Step 2: Manager Adapters (2-3 lines each)
```rust
// Context manager adapter
struct ContextResourceManager<'a> {
    manager: &'a mut ContextManager,
}

impl ResourceManager for ContextResourceManager<'_> {
    async fn execute(&mut self, op: ResourceOperation) -> Result<ResourceData> {
        match op {
            ResourceOperation::Add { paths, .. } => {
                self.manager.add_paths(os, paths, false).await?;
                Ok(ResourceData::Success("Added to context".into()))
            }
            ResourceOperation::Show => {
                let contexts = self.manager.get_all_contexts();
                Ok(ResourceData::Items(contexts))
            }
            // ... other ops
        }
    }
}

// Knowledge manager adapter  
struct KnowledgeResourceManager {
    store: Arc<Mutex<KnowledgeStore>>,
}

impl ResourceManager for KnowledgeResourceManager {
    async fn execute(&mut self, op: ResourceOperation) -> Result<ResourceData> {
        let mut store = self.store.lock().await;
        match op {
            ResourceOperation::Add { name, value } => {
                let id = store.add(&name, &value, Default::default()).await?;
                Ok(ResourceData::Success(format!("Added with ID: {}", id)))
            }
            // ... other ops
        }
    }
}
```

### Step 3: Easy Composition
```rust
// 2-3 lines to use any resource type
pub async fn handle_context_command(cmd: ContextOp, session: &mut ChatSession) -> Result<()> {
    let mut handler = ResourceHandler::new(
        ContextResourceManager::new(&mut session.context_manager),
        TableRenderer::new()
    );
    let output = handler.handle(cmd.into(), OutputFormat::Table).await?;
    println!("{}", output);
    Ok(())
}

pub async fn handle_knowledge_command(cmd: KnowledgeOp, os: &Os) -> Result<()> {
    let mut handler = ResourceHandler::new(
        KnowledgeResourceManager::new(os).await?,
        JsonRenderer::new()
    );
    let output = handler.handle(cmd.into(), OutputFormat::Json).await?;
    println!("{}", output);
    Ok(())
}
```

### Step 4: Tool Integration
```rust
// Same managers work in tool context
impl Resource {
    pub async fn invoke(&self, os: &Os, agent: Option<&Agent>) -> Result<InvokeOutput> {
        let mut handler = match self.resource_type {
            ResourceType::Context => ResourceHandler::new(
                ContextResourceManager::from_session(/* session */),
                PlainRenderer::new()
            ),
            ResourceType::Knowledge => ResourceHandler::new(
                KnowledgeResourceManager::new(os).await?,
                PlainRenderer::new()
            ),
        };
        
        let output = handler.handle(self.operation.clone(), OutputFormat::Plain).await?;
        Ok(InvokeOutput { output: OutputKind::Text(output) })
    }
}
```

## File Structure (Minimal)
```
crates/chat-cli/src/cli/chat/cli/
├── resource_new.rs              # CLI command skeleton
└── resource/
    ├── mod.rs                   # Public API
    ├── types.rs                 # ResourceOperation, ResourceData
    ├── manager.rs               # ResourceManager trait + adapters
    ├── renderer.rs              # ResourceRenderer trait + impls
    └── handler.rs               # ResourceHandler composition
```

## Benefits
1. **Simple**: Each manager adapter is ~20 lines
2. **Reusable**: Same components for CLI and tool
3. **Extensible**: Add new resource types easily
4. **Testable**: Mock managers and renderers independently
5. **Clean**: Separation between data operations and presentation
