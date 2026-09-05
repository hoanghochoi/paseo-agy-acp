use fs2::FileExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

#[cfg(test)]
use crate::db::read_delta_from_db;
use crate::db::read_replay_updates_from_db;
use crate::runtime::PromptOutcome;
use crate::types::*;

fn validate_session_setup(params: &Value) -> Result<PathBuf, String> {
    let cwd = params
        .get("cwd")
        .and_then(Value::as_str)
        .filter(|cwd| !cwd.is_empty())
        .ok_or_else(|| "cwd must be a non-empty absolute directory".to_string())?;
    let path = PathBuf::from(cwd);
    if !path.is_absolute() || !path.is_dir() {
        return Err("cwd must be an existing absolute directory".to_string());
    }
    let servers = params
        .get("mcpServers")
        .and_then(Value::as_array)
        .ok_or_else(|| "mcpServers must be an array".to_string())?;
    if !servers.is_empty() {
        return Err("non-empty mcpServers are not supported by agy-acp".to_string());
    }
    Ok(path)
}

fn split_model_entry(entry: &str) -> (&str, &str) {
    let entry = entry.trim();
    match entry.split_once('\t') {
        Some((id, label)) if !id.trim().is_empty() && !label.trim().is_empty() => {
            (id.trim(), label.trim())
        }
        _ => (entry, entry),
    }
}

pub struct Adapter {
    pub sessions: HashMap<String, Session>,
    pub conversations_dir: PathBuf,
    pub state_file: PathBuf,
    pub available_models: Vec<String>,
    pub skip_naration: bool,
}

impl Adapter {
    pub const MODEL_CONFIG_ID: &'static str = "model";

    pub fn new() -> Self {
        Self::new_with_skip_naration(false)
    }

    pub fn new_with_skip_naration(skip_naration: bool) -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| "/tmp".to_string());
        let state_dir = PathBuf::from(&home).join(".openab/agy-acp");
        Self {
            sessions: HashMap::new(),
            conversations_dir: PathBuf::from(&home).join(".gemini/antigravity-cli/conversations"),
            state_file: state_dir.join("sessions.json"),
            available_models: Self::fetch_available_models(),
            skip_naration,
        }
    }

    /// Run `agy models` and parse the output into a list of model names.
    fn fetch_available_models() -> Vec<String> {
        std::process::Command::new("agy")
            .arg("models")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Build the ACP `models` JSON for a session, given its current model_id.
    pub fn session_models_json(&mut self, model_id: Option<&str>) -> Value {
        if self.available_models.is_empty() {
            self.available_models = Self::fetch_available_models();
        }
        let current = model_id
            .map(|model| split_model_entry(model).0)
            .or_else(|| {
                self.available_models
                    .first()
                    .map(|model| split_model_entry(model).0)
            })
            .unwrap_or("");
        let available: Vec<Value> = self
            .available_models
            .iter()
            .map(|model| {
                let (model_id, name) = split_model_entry(model);
                json!({
                    "modelId": model_id,
                    "name": name,
                })
            })
            .collect();
        json!({
            "currentModelId": current,
            "availableModels": available,
        })
    }

    /// Build the ACP session config option that Zed uses for its model selector.
    pub fn session_config_options_json(&mut self, model_id: Option<&str>) -> Value {
        if self.available_models.is_empty() {
            self.available_models = Self::fetch_available_models();
        }
        let current = model_id
            .map(|model| split_model_entry(model).0)
            .or_else(|| {
                self.available_models
                    .first()
                    .map(|model| split_model_entry(model).0)
            })
            .unwrap_or("");
        let options: Vec<Value> = self
            .available_models
            .iter()
            .map(|model| {
                let (value, name) = split_model_entry(model);
                json!({
                    "value": value,
                    "name": name,
                })
            })
            .collect();
        json!([{
            "id": Self::MODEL_CONFIG_ID,
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": current,
            "options": options,
        }])
    }

    pub fn session_config_result_json(
        &mut self,
        session_id: &str,
        model_id: Option<&str>,
    ) -> Value {
        json!({
            "sessionId": session_id,
            "models": self.session_models_json(model_id),
            "configOptions": self.session_config_options_json(model_id),
        })
    }

    /// Acquire exclusive lock on a dedicated lock file for read-write mutual exclusion.
    fn lock_state_file(&self) -> Option<fs::File> {
        if let Some(parent) = self.state_file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = self.state_file.with_extension("lock");
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .ok()?;
        lock_file.lock_exclusive().ok()?;
        Some(lock_file)
    }

    /// Load persisted session store (caller must hold lock).
    fn load_store_inner(&self) -> SessionStore {
        let Some(file) = fs::File::open(&self.state_file).ok() else {
            return SessionStore::default();
        };
        serde_json::from_reader(&file).unwrap_or_default()
    }

    /// Load persisted session store with lock.
    pub fn load_store(&self) -> SessionStore {
        let _lock = self.lock_state_file();
        self.load_store_inner()
    }

    /// Restore the complete persisted session, including unbound new sessions.
    pub fn restore_session(&self, session_id: &str) -> Option<StoredSession> {
        let store = self.load_store();
        store.sessions.get(session_id).cloned()
    }

    /// Persist a session binding (read-modify-write under single lock).
    pub fn persist_session(
        &self,
        session_id: &str,
        conversation_id: Option<&str>,
        last_step_idx: i64,
        model_id: Option<&str>,
    ) {
        let Some(_lock) = self.lock_state_file() else {
            return;
        };
        let mut store = self.load_store_inner();
        let cwd = self
            .sessions
            .get(session_id)
            .map(|session| session.cwd.to_string_lossy().to_string())
            .or_else(|| {
                store
                    .sessions
                    .get(session_id)
                    .and_then(|session| session.cwd.clone())
            });
        store.sessions.insert(
            session_id.to_string(),
            StoredSession {
                conversation_id: conversation_id.map(String::from),
                last_step_idx,
                model_id: model_id.map(String::from),
                cwd,
            },
        );
        let tmp = self.state_file.with_extension("tmp");
        if let Ok(file) = fs::File::create(&tmp) {
            if serde_json::to_writer_pretty(&file, &store).is_ok() {
                let _ = fs::rename(&tmp, &self.state_file);
            }
        }
    }

    pub fn read_replay_updates_from_db_inner(
        &self,
        conversation_id: &str,
    ) -> Option<(Vec<Value>, i64)> {
        read_replay_updates_from_db(&self.conversations_dir, conversation_id, self.skip_naration)
    }

    #[cfg(test)]
    fn read_delta_from_db_inner(
        &self,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<crate::types::ConversationDelta> {
        read_delta_from_db(&self.conversations_dir, conversation_id, after_step_idx)
    }

    #[cfg(test)]
    pub fn read_response_from_db(
        &self,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<(String, i64)> {
        self.read_delta_from_db_inner(conversation_id, after_step_idx)
            .and_then(|delta| delta.text.map(|text| (text, delta.max_step_idx)))
    }

    /// Filter out leading narration ("I will ...", "I'll ...") from response parts.
    #[cfg(test)]
    pub fn filter_narration(parts: &[String]) -> Option<String> {
        filter_narration(parts)
    }

    /// A part is considered narration if every non-empty line starts with "I will" or "I'll".
    #[cfg(test)]
    pub fn is_narration(text: &str) -> bool {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return false;
        }
        lines.iter().all(|l| {
            let line = l.trim_start();
            line.starts_with("I will") || line.starts_with("I'll") || line.starts_with("I’ll")
        })
    }

    fn evict_if_needed(&mut self) {
        const MAX_SESSIONS: usize = 64;
        while self.sessions.len() >= MAX_SESSIONS {
            if let Some(key) = self.sessions.keys().next().cloned() {
                self.sessions.remove(&key);
            }
        }
    }

    pub fn restore_session_state(
        &mut self,
        session_id: &str,
        cwd_override: Option<PathBuf>,
    ) -> bool {
        let Some(stored) = self.restore_session(session_id) else {
            return false;
        };
        let Some(cwd) = cwd_override.or_else(|| stored.cwd.as_deref().map(PathBuf::from)) else {
            return false;
        };
        if !self.sessions.contains_key(session_id) {
            self.evict_if_needed();
        }
        self.sessions.insert(
            session_id.to_string(),
            Session {
                conversation_id: stored.conversation_id,
                last_step_idx: stored.last_step_idx,
                model_id: stored.model_id,
                cwd,
            },
        );
        true
    }

    pub fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": 1,
                "agentInfo": { "name": "agy", "version": env!("CARGO_PKG_VERSION") },
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": { "resume": {} },
                },
                "authMethods": [],
            })),
            error: None,
        }
    }

    pub fn handle_session_new(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let cwd = match validate_session_setup(params) {
            Ok(cwd) => cwd,
            Err(message) => return JsonRpcResponse::error(id, -32602, &message),
        };
        let session_id = Uuid::new_v4().to_string();
        self.evict_if_needed();
        self.sessions.insert(
            session_id.clone(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
                cwd,
            },
        );
        self.persist_session(&session_id, None, -1, None);
        let result = self.session_config_result_json(&session_id, None);
        JsonRpcResponse::success(id, result)
    }

    pub fn handle_session_load(&mut self, id: Value, params: &Value) -> Vec<String> {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            })
            .unwrap()];
        }

        let cwd = match validate_session_setup(params) {
            Ok(cwd) => cwd,
            Err(message) => {
                return vec![
                    serde_json::to_string(&JsonRpcResponse::error(id, -32602, &message)).unwrap(),
                ]
            }
        };

        if let Some(session) = self.sessions.get_mut(session_id) {
            session.cwd = cwd.clone();
        } else if !self.restore_session_state(session_id, Some(cwd)) {
            return vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": format!("unknown sessionId: {session_id}"),
                })),
            })
            .unwrap()];
        }

        let (conversation_id, last_step_idx, model_id) = {
            let session = &self.sessions[session_id];
            (
                session.conversation_id.clone(),
                session.last_step_idx,
                session.model_id.clone(),
            )
        };
        self.persist_session(
            session_id,
            conversation_id.as_deref(),
            last_step_idx,
            model_id.as_deref(),
        );

        let mut output_lines: Vec<String> = Vec::new();

        let replay_conv_id = self
            .sessions
            .get(session_id)
            .and_then(|session| session.conversation_id.clone());
        if let Some(conv_id) = replay_conv_id {
            if let Some((updates, max_step_idx)) = self.read_replay_updates_from_db_inner(&conv_id)
            {
                for update in updates {
                    let notification = serde_json::to_string(&JsonRpcNotification {
                        jsonrpc: "2.0",
                        method: "session/update".to_string(),
                        params: json!({
                            "sessionId": session_id,
                            "update": update,
                        }),
                    })
                    .unwrap();
                    output_lines.push(notification);
                }
                if let Some(session) = self.sessions.get_mut(session_id) {
                    session.last_step_idx = max_step_idx;
                }
                let model_id = self
                    .sessions
                    .get(session_id)
                    .and_then(|s| s.model_id.clone());
                self.persist_session(
                    session_id,
                    Some(conv_id.as_str()),
                    max_step_idx,
                    model_id.as_deref(),
                );
            }
        }

        output_lines.push({
            let model_id = self
                .sessions
                .get(session_id)
                .and_then(|s| s.model_id.clone());
            let result = self.session_config_result_json(session_id, model_id.as_deref());
            serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(result),
                error: None,
            })
            .unwrap()
        });

        output_lines
    }

    pub fn handle_session_resume(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            };
        }

        let cwd = match validate_session_setup(params) {
            Ok(cwd) => cwd,
            Err(message) => return JsonRpcResponse::error(id, -32602, &message),
        };

        let found = if let Some(session) = self.sessions.get_mut(session_id) {
            session.cwd = cwd.clone();
            true
        } else {
            self.restore_session_state(session_id, Some(cwd))
        };
        if found {
            let (conversation_id, last_step_idx, model_id) = {
                let session = &self.sessions[session_id];
                (
                    session.conversation_id.clone(),
                    session.last_step_idx,
                    session.model_id.clone(),
                )
            };
            self.persist_session(
                session_id,
                conversation_id.as_deref(),
                last_step_idx,
                model_id.as_deref(),
            );
            let model_id = self
                .sessions
                .get(session_id)
                .and_then(|s| s.model_id.clone());
            let result = self.session_config_result_json(session_id, model_id.as_deref());
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(result),
                error: None,
            };
        }

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({
                "code": -32000,
                "message": format!("unknown sessionId: {session_id}"),
            })),
        }
    }

    pub fn handle_session_set_model(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let model_id = params.get("modelId").and_then(|v| v.as_str()).unwrap_or("");

        if session_id.is_empty() || model_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId or modelId"})),
            };
        }

        if !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id, None);
        }

        let Some(session) = self.sessions.get_mut(session_id) else {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": format!("unknown sessionId: {session_id}"),
                })),
            };
        };

        session.model_id = Some(model_id.to_string());
        let model_id_str = session.model_id.clone();
        let last_step_idx = session.last_step_idx;
        let conv_id = session.conversation_id.clone();

        self.persist_session(
            session_id,
            conv_id.as_deref(),
            last_step_idx,
            model_id_str.as_deref(),
        );

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({})),
            error: None,
        }
    }

    pub fn handle_session_set_config_option(
        &mut self,
        id: Value,
        params: &Value,
    ) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let config_id = params
            .get("configId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let model_id = params.get("value").and_then(|v| v.as_str()).unwrap_or("");

        if session_id.is_empty() || config_id.is_empty() || model_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(
                    json!({"code":-32602,"message":"missing sessionId, configId, or value"}),
                ),
            };
        }

        if config_id != Self::MODEL_CONFIG_ID {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32602,
                    "message": format!("unknown configId: {config_id}"),
                })),
            };
        }

        if !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id, None);
        }

        let Some(session) = self.sessions.get_mut(session_id) else {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": format!("unknown sessionId: {session_id}"),
                })),
            };
        };

        session.model_id = Some(model_id.to_string());
        let model_id_str = session.model_id.clone();
        let last_step_idx = session.last_step_idx;
        let conv_id = session.conversation_id.clone();

        self.persist_session(
            session_id,
            conv_id.as_deref(),
            last_step_idx,
            model_id_str.as_deref(),
        );

        let config_options = self.session_config_options_json(model_id_str.as_deref());
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "configOptions": config_options })),
            error: None,
        }
    }

    pub(crate) fn prepare_prompt(
        &mut self,
        id: Value,
        params: &Value,
    ) -> Result<PromptExecution, JsonRpcResponse> {
        let session_id = params
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or("");

        if !session_id.is_empty() && !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id, None);
        }

        let Some(session) = self.sessions.get(session_id) else {
            return Err(JsonRpcResponse::error(
                id,
                -32000,
                &format!("unknown sessionId: {session_id}"),
            ));
        };
        let cwd = session.cwd.clone();
        let conversation_id = session.conversation_id.clone();
        let model_id = session.model_id.clone();
        let initial_step_idx = session.last_step_idx;

        let Some(blocks) = params.get("prompt").and_then(Value::as_array) else {
            return Err(JsonRpcResponse::error(
                id,
                -32602,
                "prompt must be an array of text blocks",
            ));
        };
        let mut text_blocks = Vec::with_capacity(blocks.len());
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                return Err(JsonRpcResponse::error(
                    id,
                    -32602,
                    "only text prompt blocks are supported",
                ));
            }
            let Some(text) = block.get("text").and_then(Value::as_str) else {
                return Err(JsonRpcResponse::error(
                    id,
                    -32602,
                    "text prompt blocks must contain string text",
                ));
            };
            text_blocks.push(text);
        }
        let prompt_text = text_blocks.join("\n").trim().to_string();
        if prompt_text.is_empty() {
            return Err(JsonRpcResponse::error(
                id,
                -32602,
                "prompt text must not be empty",
            ));
        }

        let state_dir = self
            .state_file
            .parent()
            .unwrap_or_else(|| std::path::Path::new("/tmp"))
            .to_path_buf();
        let run_logs_dir = state_dir.join("run-logs");
        if let Err(error) = fs::create_dir_all(&run_logs_dir) {
            return Err(JsonRpcResponse::error(
                id,
                -32000,
                &format!("failed to create agy run-log directory: {error}"),
            ));
        }

        Ok(PromptExecution {
            id,
            session_id: session_id.to_string(),
            prompt_text,
            cwd,
            conversation_id,
            model_id,
            initial_step_idx,
            conversations_dir: self.conversations_dir.clone(),
            state_dir,
            skip_naration: self.skip_naration,
        })
    }

    pub(crate) fn apply_prompt_outcome(&mut self, outcome: &PromptOutcome) {
        let Some(session) = self.sessions.get_mut(&outcome.session_id) else {
            return;
        };

        if session.conversation_id.is_none() {
            session.conversation_id = outcome.conversation_id.clone();
        }
        if outcome.conversation_id.is_some() {
            session.last_step_idx = outcome.last_step_idx;
        }

        let model_id = session.model_id.clone();
        if outcome.conversation_id.is_some() {
            self.persist_session(
                &outcome.session_id,
                outcome.conversation_id.as_deref(),
                outcome.last_step_idx,
                model_id.as_deref(),
            );
        }
    }
}

/// Filter out leading narration ("I will ...", "I'll ...") from response parts.
pub fn filter_narration(parts: &[String]) -> Option<String> {
    let text = parts
        .iter()
        .filter(|part| !is_narration(part))
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

/// A part is considered narration if every non-empty line starts with "I will" or "I'll".
pub fn is_narration(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return false;
    }
    lines.iter().all(|l| {
        let line = l.trim_start();
        line.starts_with("I will") || line.starts_with("I'll") || line.starts_with("I’ll")
    })
}
