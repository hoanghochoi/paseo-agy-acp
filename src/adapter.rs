use fs2::FileExt;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[cfg(test)]
use crate::db::read_delta_from_db;
use crate::db::{read_replay_updates_from_db, ReplayReadError};
use crate::runtime::PromptOutcome;
use crate::types::*;

const PERSISTENCE_FAILURE_MESSAGE: &str = "failed to persist session state";
const MODEL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);
const MODEL_DISCOVERY_POLL_INTERVAL: Duration = Duration::from_millis(25);
pub(crate) const MAX_PROMPT_TEXT_BYTES: usize = 256 * 1024;
pub(crate) const MAX_MODEL_OUTPUT_BYTES: usize = 256 * 1024;
pub(crate) const MAX_EXTRA_ARGS_BYTES: usize = 64 * 1024;
pub(crate) const MAX_PERSISTED_STATE_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_PERSISTED_SESSIONS: usize = 1024;
pub(crate) const MAX_RESIDENT_SESSIONS: usize = 64;
const MAX_PERSISTED_SESSION_ID_BYTES: usize = 256;
const MAX_PERSISTED_CONVERSATION_ID_BYTES: usize = 128;
const MAX_PERSISTED_MODEL_ID_BYTES: usize = 256;
const MAX_PERSISTED_CWD_BYTES: usize = 16 * 1024;

fn persistence_error(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::error(id, -32603, PERSISTENCE_FAILURE_MESSAGE)
}

fn validate_session_id(params: &Value) -> Option<&str> {
    params
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|session_id| !session_id.trim().is_empty())
}

fn validate_prompt_text(params: &Value) -> Result<String, &'static str> {
    let Some(blocks) = params.get("prompt").and_then(Value::as_array) else {
        return Err("prompt must be an array of text blocks");
    };
    let mut text_blocks = Vec::with_capacity(blocks.len());
    for block in blocks {
        if block.get("type").and_then(Value::as_str) != Some("text") {
            return Err("only text prompt blocks are supported");
        }
        let Some(text) = block.get("text").and_then(Value::as_str) else {
            return Err("text prompt blocks must contain string text");
        };
        text_blocks.push(text);
    }
    let prompt_text = text_blocks.join("\n").trim().to_string();
    if prompt_text.is_empty() {
        return Err("prompt text must not be empty");
    }
    if prompt_text.len() > MAX_PROMPT_TEXT_BYTES {
        return Err("prompt text exceeds maximum size");
    }
    Ok(prompt_text)
}

fn session_from_stored(stored: StoredSession, cwd: PathBuf) -> Session {
    Session {
        conversation_id: stored.conversation_id,
        last_step_idx: stored.last_step_idx,
        model_id: stored.model_id,
        cwd,
    }
}

fn stored_session_from_session(session: &Session) -> StoredSession {
    StoredSession {
        conversation_id: session.conversation_id.clone(),
        last_step_idx: session.last_step_idx,
        model_id: session.model_id.clone(),
        cwd: Some(session.cwd.to_string_lossy().to_string()),
    }
}

fn conversation_bindings_conflict(left: Option<&str>, right: Option<&str>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if left != right)
}

fn apply_outcome_binding(
    conversation_id: &mut Option<String>,
    last_step_idx: &mut i64,
    outcome: &PromptOutcome,
) -> io::Result<()> {
    let Some(outcome_conversation_id) = outcome.conversation_id.as_deref() else {
        return Ok(());
    };

    match conversation_id.as_deref() {
        Some(current) if current != outcome_conversation_id => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "prompt outcome conversation conflicts with session state",
            ))
        }
        Some(_) => {}
        None => *conversation_id = Some(outcome_conversation_id.to_string()),
    }
    *last_step_idx = (*last_step_idx).max(outcome.last_step_idx);
    Ok(())
}

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

fn parse_available_models(output: &str) -> Vec<String> {
    output
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

pub(crate) fn parse_available_models_bounded(output: &[u8]) -> Vec<String> {
    if output.len() > MAX_MODEL_OUTPUT_BYTES {
        return Vec::new();
    }
    parse_available_models(&String::from_utf8_lossy(output))
}

pub(crate) fn parse_extra_args_bounded(extra: &str) -> Option<Vec<std::ffi::OsString>> {
    if extra.len() > MAX_EXTRA_ARGS_BYTES {
        return None;
    }

    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut token_started = false;

    for character in extra.chars() {
        if escaped {
            current.push(character);
            token_started = true;
            escaped = false;
            continue;
        }

        match quote {
            Some('\'') => {
                if character == '\'' {
                    quote = None;
                } else {
                    current.push(character);
                }
            }
            Some('"') => match character {
                '"' => quote = None,
                '\\' => escaped = true,
                _ => current.push(character),
            },
            Some(_) => unreachable!("extra-argument parser only uses quote delimiters"),
            None => match character {
                '\\' => escaped = true,
                '\'' | '"' => {
                    quote = Some(character);
                    token_started = true;
                }
                character if character.is_whitespace() => {
                    if token_started {
                        args.push(std::ffi::OsString::from(std::mem::take(&mut current)));
                        token_started = false;
                    }
                }
                _ => {
                    current.push(character);
                    token_started = true;
                }
            },
        }
    }

    if escaped || quote.is_some() {
        return None;
    }
    if token_started {
        args.push(std::ffi::OsString::from(current));
    }
    Some(args)
}

fn read_file_bounded(path: &std::path::Path, max_bytes: usize) -> Option<Vec<u8>> {
    let file = fs::File::open(path).ok()?;
    let mut output = Vec::new();
    file.take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut output)
        .ok()?;
    (output.len() <= max_bytes).then_some(output)
}

fn read_state_file_bounded(path: &std::path::Path) -> io::Result<Vec<u8>> {
    let file = fs::File::open(path)?;
    let mut output = Vec::new();
    file.take((MAX_PERSISTED_STATE_BYTES as u64).saturating_add(1))
        .read_to_end(&mut output)?;
    if output.len() > MAX_PERSISTED_STATE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "persisted session state exceeds maximum size",
        ));
    }
    Ok(output)
}

fn invalid_state(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn validate_store_bounds(store: &SessionStore) -> io::Result<()> {
    if store.sessions.len() > MAX_PERSISTED_SESSIONS {
        return Err(invalid_state("persisted session count exceeds maximum"));
    }
    for (session_id, session) in &store.sessions {
        if session_id.is_empty() || session_id.len() > MAX_PERSISTED_SESSION_ID_BYTES {
            return Err(invalid_state("persisted session ID exceeds maximum size"));
        }
        if let Some(conversation_id) = session.conversation_id.as_deref() {
            if conversation_id.len() > MAX_PERSISTED_CONVERSATION_ID_BYTES {
                return Err(invalid_state(
                    "persisted conversation ID exceeds maximum size",
                ));
            }
            if conversation_id.trim().is_empty()
                || conversation_id
                    .chars()
                    .any(|character| character.is_control() || matches!(character, '/' | '\\'))
            {
                return Err(invalid_state("persisted conversation ID is unsafe"));
            }
        }
        if session
            .model_id
            .as_deref()
            .is_some_and(|value| value.len() > MAX_PERSISTED_MODEL_ID_BYTES)
        {
            return Err(invalid_state("persisted model ID exceeds maximum size"));
        }
        if session
            .cwd
            .as_deref()
            .is_some_and(|value| value.len() > MAX_PERSISTED_CWD_BYTES)
        {
            return Err(invalid_state("persisted cwd exceeds maximum size"));
        }
        if session.last_step_idx < -1 {
            return Err(invalid_state("persisted step cursor is invalid"));
        }
    }
    Ok(())
}

pub struct Adapter {
    pub sessions: HashMap<String, Session>,
    pub conversations_dir: PathBuf,
    pub state_file: PathBuf,
    pub available_models: Vec<String>,
    pub skip_naration: bool,
    pub(crate) command: CommandSpec,
    pub(crate) session_access: HashMap<String, u64>,
    pub(crate) next_access: u64,
    pub(crate) active_sessions: HashSet<String>,
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
            command: CommandSpec::new("agy", Vec::new()),
            session_access: HashMap::new(),
            next_access: 0,
            active_sessions: HashSet::new(),
        }
    }

    /// Run `agy models` and parse the output into a list of model names.
    fn fetch_available_models() -> Vec<String> {
        let output_path =
            std::env::temp_dir().join(format!("agy-acp-models-{}.txt", Uuid::new_v4()));
        let output_file = match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output_path)
        {
            Ok(file) => file,
            Err(_) => return Vec::new(),
        };
        let mut child = match Command::new("agy")
            .arg("models")
            .stdout(Stdio::from(output_file))
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => {
                let _ = fs::remove_file(&output_path);
                return Vec::new();
            }
        };
        let deadline = Instant::now() + MODEL_DISCOVERY_TIMEOUT;

        let models = loop {
            match child.try_wait() {
                Ok(Some(status)) if status.success() => {
                    break read_file_bounded(&output_path, MAX_MODEL_OUTPUT_BYTES)
                        .map(|output| parse_available_models_bounded(&output))
                        .unwrap_or_default();
                }
                Ok(Some(_)) => break Vec::new(),
                Ok(None) if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Vec::new();
                }
                Ok(None) => {
                    std::thread::sleep(MODEL_DISCOVERY_POLL_INTERVAL);
                }
                Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Vec::new();
                }
            }
        };
        let _ = fs::remove_file(output_path);
        models
    }

    /// Build the ACP `models` JSON for a session, given its current model_id.
    pub fn session_models_json(&self, model_id: Option<&str>) -> Value {
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

    /// Build the ACP session config option that Paseo exposes for model selection.
    pub fn session_config_options_json(&self, model_id: Option<&str>) -> Value {
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

    pub fn session_config_result_json(&self, session_id: &str, model_id: Option<&str>) -> Value {
        json!({
            "sessionId": session_id,
            "models": self.session_models_json(model_id),
            "configOptions": self.session_config_options_json(model_id),
        })
    }

    /// Acquire exclusive lock on a dedicated lock file for read-write mutual exclusion.
    fn lock_state_file(&self) -> io::Result<fs::File> {
        if let Some(parent) = self.state_file.parent() {
            fs::create_dir_all(parent)?;
        }
        let lock_path = self.state_file.with_extension("lock");
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;
        lock_file.lock_exclusive()?;
        Ok(lock_file)
    }

    /// Load persisted session store (caller must hold lock).
    fn load_store_inner(&self) -> io::Result<SessionStore> {
        let bytes = match read_state_file_bounded(&self.state_file) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(SessionStore::default());
            }
            Err(error) => return Err(error),
        };
        let store: SessionStore = serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        validate_store_bounds(&store)?;
        Ok(store)
    }

    /// Load persisted session store with lock.
    pub fn load_store(&self) -> io::Result<SessionStore> {
        let _lock = self.lock_state_file()?;
        self.load_store_inner()
    }

    /// Restore the complete persisted session, including unbound new sessions.
    pub fn restore_session(&self, session_id: &str) -> io::Result<Option<StoredSession>> {
        let store = self.load_store()?;
        Ok(store.sessions.get(session_id).cloned())
    }

    fn write_store_inner(&self, store: &SessionStore) -> io::Result<()> {
        validate_store_bounds(store)?;
        let contents = serde_json::to_vec_pretty(store)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if contents.len() > MAX_PERSISTED_STATE_BYTES {
            return Err(invalid_state(
                "persisted session state exceeds maximum size",
            ));
        }
        let tmp = self.state_file.with_extension("tmp");
        let mut file = fs::File::create(&tmp)?;
        file.write_all(&contents)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, &self.state_file)?;
        #[cfg(unix)]
        if let Some(parent) = self.state_file.parent() {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(())
    }

    fn update_store<F>(&self, update: F) -> io::Result<()>
    where
        F: FnOnce(&mut SessionStore) -> io::Result<()>,
    {
        let _lock = self.lock_state_file()?;
        let mut store = self.load_store_inner()?;
        update(&mut store)?;
        self.write_store_inner(&store)
    }

    pub fn persist_session(&self, session_id: &str, session: &Session) -> io::Result<()> {
        let snapshot = stored_session_from_session(session);
        self.update_store(|store| {
            store.sessions.insert(session_id.to_string(), snapshot);
            Ok(())
        })
    }

    pub fn read_replay_updates_from_db_inner(
        &self,
        conversation_id: &str,
    ) -> Result<Option<(Vec<Value>, i64)>, ReplayReadError> {
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

    fn touch_session(&mut self, session_id: &str) {
        self.next_access = self.next_access.saturating_add(1);
        self.session_access
            .insert(session_id.to_string(), self.next_access);
    }

    pub(crate) fn mark_prompt_active(&mut self, session_id: &str) {
        self.active_sessions.insert(session_id.to_string());
    }

    pub(crate) fn mark_prompt_complete(&mut self, session_id: &str) {
        self.active_sessions.remove(session_id);
    }

    fn evict_if_needed(&mut self) {
        while self.sessions.len() >= MAX_RESIDENT_SESSIONS {
            let inactive_victim = self
                .sessions
                .keys()
                .filter(|key| !self.active_sessions.contains(*key))
                .min_by_key(|key| self.session_access.get(*key).copied().unwrap_or(0))
                .cloned();
            let victim = inactive_victim.or_else(|| {
                self.sessions
                    .keys()
                    .min_by_key(|key| self.session_access.get(*key).copied().unwrap_or(0))
                    .cloned()
            });
            let Some(victim) = victim else {
                break;
            };
            self.sessions.remove(&victim);
            self.session_access.remove(&victim);
        }
    }

    fn candidate_session(
        &mut self,
        session_id: &str,
        cwd_override: Option<PathBuf>,
    ) -> io::Result<Option<Session>> {
        if let Some(session) = self.sessions.get(session_id) {
            let mut candidate = session.clone();
            self.touch_session(session_id);
            if let Some(cwd) = cwd_override {
                candidate.cwd = cwd;
            }
            return Ok(Some(candidate));
        }

        let Some(stored) = self.restore_session(session_id)? else {
            return Ok(None);
        };
        let Some(cwd) = cwd_override.or_else(|| stored.cwd.as_deref().map(PathBuf::from)) else {
            return Ok(None);
        };
        Ok(Some(session_from_stored(stored, cwd)))
    }

    fn install_candidate(&mut self, session_id: &str, candidate: Session) {
        if !self.sessions.contains_key(session_id) {
            self.evict_if_needed();
        }
        self.sessions.insert(session_id.to_string(), candidate);
        self.touch_session(session_id);
    }

    pub fn restore_session_state(
        &mut self,
        session_id: &str,
        cwd_override: Option<PathBuf>,
    ) -> io::Result<bool> {
        let Some(candidate) = self.candidate_session(session_id, cwd_override)? else {
            return Ok(false);
        };
        self.install_candidate(session_id, candidate);
        Ok(true)
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
        let candidate = Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
            cwd,
        };
        if self.persist_session(&session_id, &candidate).is_err() {
            return persistence_error(id);
        }
        self.install_candidate(&session_id, candidate);
        let result = self.session_config_result_json(&session_id, None);
        JsonRpcResponse::success(id, result)
    }

    pub fn handle_session_load(&mut self, id: Value, params: &Value) -> Vec<String> {
        let Some(session_id) = validate_session_id(params) else {
            return vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(
                    json!({"code":-32602,"message":"sessionId must be a non-empty string"}),
                ),
            })
            .unwrap()];
        };

        let cwd = match validate_session_setup(params) {
            Ok(cwd) => cwd,
            Err(message) => {
                return vec![
                    serde_json::to_string(&JsonRpcResponse::error(id, -32602, &message)).unwrap(),
                ]
            }
        };

        let mut candidate = match self.candidate_session(session_id, Some(cwd)) {
            Ok(Some(candidate)) => candidate,
            Ok(None) => {
                return vec![serde_json::to_string(&JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(json!({
                        "code": -32000,
                        "message": format!("unknown sessionId: {session_id}"),
                    })),
                })
                .unwrap()]
            }
            Err(_) => {
                return vec![serde_json::to_string(&persistence_error(id)).unwrap()];
            }
        };

        let replay = match candidate.conversation_id.as_deref() {
            Some(conversation_id) => {
                match self.read_replay_updates_from_db_inner(conversation_id) {
                    Ok(replay) => replay,
                    Err(ReplayReadError::Unavailable) => None,
                    Err(ReplayReadError::BudgetExceeded) => {
                        return vec![serde_json::to_string(&JsonRpcResponse::error(
                            id,
                            -32000,
                            "conversation history exceeds replay budget",
                        ))
                        .unwrap()];
                    }
                }
            }
            None => None,
        };
        if let Some((_, max_step_idx)) = &replay {
            candidate.last_step_idx = *max_step_idx;
        }

        if self.persist_session(session_id, &candidate).is_err() {
            return vec![serde_json::to_string(&persistence_error(id)).unwrap()];
        }
        self.install_candidate(session_id, candidate.clone());

        let mut output_lines: Vec<String> = Vec::new();
        if let Some((updates, _)) = replay {
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
        }

        output_lines.push({
            let result = self.session_config_result_json(session_id, candidate.model_id.as_deref());
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
        let Some(session_id) = validate_session_id(params) else {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(
                    json!({"code":-32602,"message":"sessionId must be a non-empty string"}),
                ),
            };
        };

        let cwd = match validate_session_setup(params) {
            Ok(cwd) => cwd,
            Err(message) => return JsonRpcResponse::error(id, -32602, &message),
        };

        let candidate = match self.candidate_session(session_id, Some(cwd)) {
            Ok(Some(candidate)) => candidate,
            Ok(None) => {
                return JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(json!({
                        "code": -32000,
                        "message": format!("unknown sessionId: {session_id}"),
                    })),
                }
            }
            Err(_) => return persistence_error(id),
        };

        if self.persist_session(session_id, &candidate).is_err() {
            return persistence_error(id);
        }
        let result = self.session_config_result_json(session_id, candidate.model_id.as_deref());
        self.install_candidate(session_id, candidate);
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn handle_session_set_model(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = validate_session_id(params).unwrap_or("");
        let model_id = params.get("modelId").and_then(|v| v.as_str()).unwrap_or("");

        if session_id.is_empty() || model_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId or modelId"})),
            };
        }

        let mut candidate = match self.candidate_session(session_id, None) {
            Ok(Some(candidate)) => candidate,
            Ok(None) => {
                return JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(json!({
                        "code": -32000,
                        "message": format!("unknown sessionId: {session_id}"),
                    })),
                }
            }
            Err(_) => return persistence_error(id),
        };

        candidate.model_id = Some(model_id.to_string());
        if self.persist_session(session_id, &candidate).is_err() {
            return persistence_error(id);
        }
        self.install_candidate(session_id, candidate);

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
        let session_id = validate_session_id(params).unwrap_or("");
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

        let mut candidate = match self.candidate_session(session_id, None) {
            Ok(Some(candidate)) => candidate,
            Ok(None) => {
                return JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(json!({
                        "code": -32000,
                        "message": format!("unknown sessionId: {session_id}"),
                    })),
                }
            }
            Err(_) => return persistence_error(id),
        };

        candidate.model_id = Some(model_id.to_string());
        if self.persist_session(session_id, &candidate).is_err() {
            return persistence_error(id);
        }
        let model_id_str = candidate.model_id.clone();
        self.install_candidate(session_id, candidate);

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
        let Some(session_id) = validate_session_id(params) else {
            return Err(JsonRpcResponse::error(
                id,
                -32602,
                "sessionId must be a non-empty string",
            ));
        };
        let prompt_text = match validate_prompt_text(params) {
            Ok(prompt_text) => prompt_text,
            Err(message) => return Err(JsonRpcResponse::error(id, -32602, message)),
        };

        if !self.sessions.contains_key(session_id)
            && self.restore_session_state(session_id, None).is_err()
        {
            return Err(persistence_error(id));
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
            command: self.command.clone(),
        })
    }

    pub(crate) fn apply_prompt_outcome(&mut self, outcome: &PromptOutcome) -> io::Result<()> {
        if let Some(session) = self.sessions.get(&outcome.session_id) {
            let mut candidate = session.clone();
            apply_outcome_binding(
                &mut candidate.conversation_id,
                &mut candidate.last_step_idx,
                outcome,
            )?;
            self.update_store(|store| {
                let Some(stored) = store.sessions.get_mut(&outcome.session_id) else {
                    store.sessions.insert(
                        outcome.session_id.clone(),
                        stored_session_from_session(&candidate),
                    );
                    return Ok(());
                };

                if conversation_bindings_conflict(
                    stored.conversation_id.as_deref(),
                    candidate.conversation_id.as_deref(),
                ) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "persisted conversation conflicts with resident session",
                    ));
                }

                if outcome.conversation_id.is_some() {
                    apply_outcome_binding(
                        &mut stored.conversation_id,
                        &mut stored.last_step_idx,
                        outcome,
                    )?;
                    candidate.last_step_idx = candidate.last_step_idx.max(stored.last_step_idx);
                    *stored = stored_session_from_session(&candidate);
                }
                Ok(())
            })?;
            self.sessions.insert(outcome.session_id.clone(), candidate);
            return Ok(());
        }

        self.update_store(|store| {
            let stored = store.sessions.get_mut(&outcome.session_id).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "prompt session state is missing")
            })?;
            apply_outcome_binding(
                &mut stored.conversation_id,
                &mut stored.last_step_idx,
                outcome,
            )
        })
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
