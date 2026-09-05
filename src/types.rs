use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    pub id: Option<Value>,
    pub method: Option<String>,
    pub params: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Value, code: i64, message: &str) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({"code": code, "message": message})),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: &'static str,
    pub method: String,
    pub params: Value,
}

/// Persisted session→conversation mapping stored in ~/.openab/agy-acp/sessions.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionStore {
    pub sessions: HashMap<String, StoredSession>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSession {
    pub conversation_id: Option<String>,
    /// Last step idx read from SQLite; used for delta extraction.
    #[serde(default)]
    pub last_step_idx: i64,
    /// Selected model ID for this session.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Absolute working directory supplied by the ACP lifecycle request.
    #[serde(default)]
    pub cwd: Option<String>,
}

pub struct Session {
    pub conversation_id: Option<String>,
    /// Last step idx read from SQLite.
    pub last_step_idx: i64,
    /// Selected model ID for this session.
    pub model_id: Option<String>,
    /// Absolute working directory used for this session's `agy` subprocesses.
    pub cwd: PathBuf,
}

#[derive(Debug)]
pub(crate) struct PromptExecution {
    pub id: Value,
    pub session_id: String,
    pub prompt_text: String,
    pub cwd: PathBuf,
    pub conversation_id: Option<String>,
    pub model_id: Option<String>,
    pub initial_step_idx: i64,
    pub conversations_dir: PathBuf,
    pub state_dir: PathBuf,
    pub skip_naration: bool,
}

#[cfg(test)]
pub struct ConversationDelta {
    pub text: Option<String>,
    pub max_step_idx: i64,
}

#[derive(Debug, Default)]
pub struct StreamingState {
    pub conversation_id: Option<String>,
    pub base_step_idx: i64,
    pub last_step_idx: i64,
    pub had_agent_text: bool,
    pub agent_text_lengths: HashMap<i64, usize>,
    pub thought_text_lengths: HashMap<i64, usize>,
    pub emitted_tool_steps: HashSet<i64>,
    pub last_title: Option<String>,
    pub skip_naration: bool,
    pub child_pid: Option<u32>,
}
