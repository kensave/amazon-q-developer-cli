use eyre::Result;
use async_trait::async_trait;

use super::types::{ResourceOperation, ResourceData};

/// Core trait for resource management operations
#[async_trait]
pub trait ResourceManager {
    /// Execute a resource operation and return the result
    async fn execute(&mut self, operation: ResourceOperation) -> Result<ResourceData>;
    
    /// Get the display name for this resource manager
    fn name(&self) -> &'static str;
    
    /// Check if this manager supports a specific operation
    fn supports_operation(&self, operation: &ResourceOperation) -> bool;
}

/// Simple composition struct that combines manager and renderer
pub struct ResourceHandler<M: ResourceManager, R: super::ResourceRenderer> {
    manager: M,
    renderer: R,
}

impl<M: ResourceManager, R: super::ResourceRenderer> ResourceHandler<M, R> {
    pub fn new(manager: M, renderer: R) -> Self {
        Self { manager, renderer }
    }
    
    /// Handle an operation with the specified output format
    pub async fn handle(
        &mut self, 
        operation: ResourceOperation, 
        format: super::OutputFormat
    ) -> Result<String> {
        let data = self.manager.execute(operation).await?;
        Ok(self.renderer.render(&data, format))
    }
}
