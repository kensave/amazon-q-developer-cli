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

#[async_trait]
impl ResourceManager for Box<dyn ResourceManager + Send> {
    async fn execute(&mut self, operation: ResourceOperation) -> Result<ResourceData> {
        (**self).execute(operation).await
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }

    fn supports_operation(&self, operation: &ResourceOperation) -> bool {
        (**self).supports_operation(operation)
    }
}

/// Simple composition struct that executes operations
pub struct ResourceHandler<M: ResourceManager> {
    manager: M,
}

impl<M: ResourceManager> ResourceHandler<M> {
    pub fn new(manager: M) -> Self {
        Self { manager }
    }

    /// Execute an operation and return raw data
    pub async fn handle(
        &mut self,
        operation: ResourceOperation
    ) -> Result<super::ResourceData> {
        self.manager.execute(operation).await
    }

    /// Check if this handler supports a specific operation
    pub fn supports_operation(&self, operation: &ResourceOperation) -> bool {
        self.manager.supports_operation(operation)
    }
}
