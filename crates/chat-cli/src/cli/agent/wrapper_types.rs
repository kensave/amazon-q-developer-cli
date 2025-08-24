use std::borrow::Borrow;
use std::ops::Deref;

use schemars::{
    JsonSchema,
    Schema,
    SchemaGenerator,
    json_schema,
};
use serde::{
    Deserialize,
    Serialize,
};

/// Subject of the tool name change. For tools in mcp servers, you would need to prefix them with
/// their server names
#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq, JsonSchema)]
pub struct OriginalToolName(String);

impl Deref for OriginalToolName {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Borrow<str> for OriginalToolName {
    fn borrow(&self) -> &str {
        self.0.as_str()
    }
}

pub fn alias_schema(generator: &mut SchemaGenerator) -> Schema {
    let key_schema = generator.subschema_for::<OriginalToolName>();
    let key_description = key_schema
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("Subject of the tool name change. For tools in mcp servers, you would need to prefix them with their server names");

    json_schema!({
        "type": "object",
        "additionalProperties": {
            "type": "string",
            "description": "The name to change to. For tools in mcp servers, you would need to exclude their server prefix"
        },
        "propertyNames": {
            "type": "string",
            "description": key_description
        }
    })
}

/// The name of the tool to be configured
#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq, JsonSchema)]
pub struct ToolSettingTarget(pub String);

impl Deref for ToolSettingTarget {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Borrow<str> for ToolSettingTarget {
    fn borrow(&self) -> &str {
        self.0.as_str()
    }
}

pub fn tool_settings_schema(generator: &mut SchemaGenerator) -> Schema {
    let key_schema = generator.subschema_for::<ToolSettingTarget>();
    let key_description = key_schema
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("The name of the tool to be configured");

    json_schema!({
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "description": "Settings for tools. Refer to our documentations to see how to configure them"
        },
        "propertyNames": {
            "type": "string",
            "description": key_description
        }
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ResourceType {
    Pinned,
    OnDemand,
    Indexed,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum IndexType {
    Fast,
    Best,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq, JsonSchema)]
pub struct ResourceDefinition {
    pub source: String,
    #[serde(rename = "type")]
    pub resource_type: ResourceType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_type: Option<IndexType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq, JsonSchema)]
#[serde(untagged)]
pub enum ResourcePath {
    Complex(ResourceDefinition),
    Simple(String),
}

impl ResourcePath {
    pub fn source(&self) -> &str {
        match self {
            ResourcePath::Complex(def) => &def.source,
            ResourcePath::Simple(s) => s,
        }
    }
}

impl Deref for ResourcePath {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.source()
    }
}

impl Borrow<str> for ResourcePath {
    fn borrow(&self) -> &str {
        self.source()
    }
}

impl From<&str> for ResourcePath {
    fn from(value: &str) -> Self {
        ResourcePath::Simple(value.to_string())
    }
}

impl From<String> for ResourcePath {
    fn from(value: String) -> Self {
        ResourcePath::Simple(value)
    }
}
