# ACP P0 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `agy-acp` execute each session in its Paseo-provided workspace, emit JSON-RPC through one writer, remain cancellable under concurrent prompts, and fail closed on invalid input.

**Architecture:** Add a small protocol module for validated JSON-RPC envelopes, retain validated working-directory state per session, and split prompt preparation/execution/commit so the adapter lock is never held while `agy` runs. A bounded MPSC output sink serializes every response and notification, while an active-prompt registry enforces one turn per session and permits different sessions to run concurrently.

**Tech Stack:** Rust 2021, Tokio 1, serde/serde_json, rusqlite, fs2, UUID, Cargo test harness.

**Spec:** `docs/superpowers/specs/2026-09-05-acp-p0-hardening-design.md`

## Global Constraints

- Keep ACP wire protocol version `1` and preserve existing model/configuration response fields.
- Accept only an empty `mcpServers` array; reject non-empty lists before subprocess creation.
- Support text prompt blocks only and reject mixed/unsupported content without silently dropping data.
- Never hold the adapter mutex while waiting for `agy` or while blocking on output backpressure.
- Use exactly one stdout owner; no adapter, poller, or dispatcher branch writes stdout directly.
- Keep existing invocation-log conversation binding, timeout detection, stdout fallback, narration behavior, and fail-closed terminal checks.
- Preserve existing state files with backward-compatible `cwd` deserialization.
- Do not contact Gemini, push GitHub, change Paseo configuration, or implement Phase 2/3 features.
- Every runtime behavior change follows a witnessed RED → GREEN cycle before refactoring.

---

### Task 1: Validated JSON-RPC Envelope and Shared Responses

**Files:**
- Create: `src/protocol.rs`
- Modify: `src/main.rs`
- Modify: `src/types.rs`
- Modify: `src/tests.rs`

**Interfaces:**
- Produces: `protocol::parse_jsonrpc_line(&str) -> Result<IncomingMessage, JsonRpcResponse>`
- Produces: `IncomingMessage::{Request { id, call }, Notification { call }}`
- Produces: `RpcCall { method: String, params: Value }`
- Produces: `JsonRpcResponse::{success, error}`
- Consumes later: Task 5 dispatches only validated `IncomingMessage` values.

- [ ] **Step 1: Write failing parser and constructor tests**

Add imports and focused tests to `src/tests.rs`:

```rust
use crate::protocol::{parse_jsonrpc_line, IncomingMessage};
use crate::types::JsonRpcResponse;

#[test]
fn malformed_json_returns_parse_error() {
    let error = parse_jsonrpc_line("{").unwrap_err();
    assert_eq!(error.id, Value::Null);
    assert_eq!(error.error.unwrap()["code"], -32700);
}

#[test]
fn invalid_jsonrpc_envelope_returns_invalid_request() {
    let error = parse_jsonrpc_line(r#"{"jsonrpc":"1.0","id":7,"method":"initialize"}"#)
        .unwrap_err();
    assert_eq!(error.id, json!(7));
    assert_eq!(error.error.unwrap()["code"], -32600);
}

#[test]
fn parser_distinguishes_request_from_notification() {
    let request = parse_jsonrpc_line(
        r#"{"jsonrpc":"2.0","id":null,"method":"initialize","params":{}}"#,
    )
    .unwrap();
    assert!(matches!(request, IncomingMessage::Request { id: Value::Null, .. }));

    let notification = parse_jsonrpc_line(
        r#"{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s"}}"#,
    )
    .unwrap();
    assert!(matches!(notification, IncomingMessage::Notification { .. }));
}

#[test]
fn response_helpers_keep_jsonrpc_shape() {
    let response = JsonRpcResponse::error(json!(3), -32602, "Invalid params");
    assert_eq!(serde_json::to_value(response).unwrap(), json!({
        "jsonrpc": "2.0",
        "id": 3,
        "error": {"code": -32602, "message": "Invalid params"}
    }));
}
```

- [ ] **Step 2: Run Task 1 tests and witness RED**

Run:

```powershell
cargo test malformed_json_returns_parse_error -- --nocapture
cargo test invalid_jsonrpc_envelope_returns_invalid_request -- --nocapture
cargo test parser_distinguishes_request_from_notification -- --nocapture
cargo test response_helpers_keep_jsonrpc_shape -- --nocapture
```

Expected: compilation fails because `protocol`, `IncomingMessage`, and response constructors do not exist.

- [ ] **Step 3: Implement the minimal protocol module**

Create `src/protocol.rs` with these public crate interfaces and validation rules:

```rust
use serde_json::{json, Value};
use crate::types::JsonRpcResponse;

#[derive(Debug)]
pub(crate) struct RpcCall {
    pub method: String,
    pub params: Value,
}

#[derive(Debug)]
pub(crate) enum IncomingMessage {
    Request { id: Value, call: RpcCall },
    Notification { call: RpcCall },
}

pub(crate) fn parse_jsonrpc_line(line: &str) -> Result<IncomingMessage, JsonRpcResponse> {
    let value: Value = serde_json::from_str(line)
        .map_err(|_| JsonRpcResponse::error(Value::Null, -32700, "Parse error"))?;
    let object = value.as_object().ok_or_else(||
        JsonRpcResponse::error(Value::Null, -32600, "Invalid Request")
    )?;
    let candidate_id = object.get("id").cloned().unwrap_or(Value::Null);
    let valid_id = candidate_id.is_null() || candidate_id.is_string() || candidate_id.is_number();
    let response_id = valid_id.then_some(candidate_id.clone()).unwrap_or(Value::Null);
    if object.get("jsonrpc") != Some(&json!("2.0")) || !valid_id {
        return Err(JsonRpcResponse::error(response_id, -32600, "Invalid Request"));
    }
    let method = object.get("method").and_then(Value::as_str)
        .filter(|method| !method.is_empty())
        .ok_or_else(|| JsonRpcResponse::error(candidate_id.clone(), -32600, "Invalid Request"))?
        .to_string();
    let params = object.get("params").cloned().unwrap_or_else(|| json!({}));
    if !params.is_object() && !params.is_array() {
        return Err(JsonRpcResponse::error(candidate_id, -32602, "Invalid params"));
    }
    let call = RpcCall { method, params };
    if object.contains_key("id") {
        Ok(IncomingMessage::Request { id: candidate_id, call })
    } else {
        Ok(IncomingMessage::Notification { call })
    }
}
```

Add `JsonRpcResponse::success` and `JsonRpcResponse::error` in `src/types.rs`; add `mod protocol;` in `src/main.rs`. Do not switch dispatch yet.

- [ ] **Step 4: Run focused tests and the default suite GREEN**

Run:

```powershell
cargo test malformed_json_returns_parse_error -- --nocapture
cargo test --quiet
```

Expected: all four new tests pass; default suite has zero failures.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/protocol.rs src/types.rs src/main.rs src/tests.rs
git diff --cached --check
git commit -m "feat: validate JSON-RPC envelopes"
```

---

### Task 2: Per-Session Working Directory and Lifecycle Validation

**Files:**
- Modify: `src/types.rs`
- Modify: `src/adapter.rs`
- Modify: `src/main.rs`
- Modify: `src/tests.rs`
- Modify: `README.md`

**Interfaces:**
- Produces: `validate_session_setup(&Value) -> Result<PathBuf, String>` inside `adapter.rs`
- Produces for deterministic tests: `Adapter::new_for_test(root: &Path)`, using `root/state/sessions.json`, `root/conversations`, and a fixed model list without running `agy models`.
- Changes: `Adapter::handle_session_new(id, params)` now consumes lifecycle params.
- Changes: `Adapter::restore_session(session_id) -> Option<StoredSession>` returns entries even before a conversation is bound.
- Changes: `Adapter::restore_session_state(session_id, cwd_override)` can restore stored `cwd` or accept a validated lifecycle override.
- Data: `Session.cwd: PathBuf`; `StoredSession.cwd: Option<String>` with `#[serde(default)]`.
- Consumes later: `prepare_prompt` snapshots `Session.cwd` in Task 3.

- [ ] **Step 1: Add failing lifecycle and compatibility tests**

Add reusable test helpers:

```rust
fn fresh_test_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("agy-acp-{label}-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    root
}

fn session_setup_params(cwd: &std::path::Path) -> Value {
    json!({"cwd": cwd.to_string_lossy(), "mcpServers": []})
}

fn open_test_session(adapter: &mut Adapter, cwd: &std::path::Path) -> String {
    adapter
        .handle_session_new(json!(1), &session_setup_params(cwd))
        .result
        .unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string()
}
```

Add these complete tests:

```rust
#[test]
fn session_new_rejects_missing_or_relative_cwd() {
    let root = fresh_test_root("bad-cwd");
    let mut adapter = Adapter::new_for_test(&root);
    for params in [json!({"mcpServers": []}), json!({"cwd": "relative", "mcpServers": []})] {
        let response = adapter.handle_session_new(json!(1), &params);
        assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
    }
}

#[test]
fn session_new_rejects_non_empty_mcp_servers() {
    let root = fresh_test_root("mcp");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = Adapter::new_for_test(&root);
    let response = adapter.handle_session_new(
        json!(1),
        &json!({"cwd": cwd.to_string_lossy(), "mcpServers": [{"type": "stdio"}]}),
    );
    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn session_new_retains_and_persists_cwd() {
    let root = fresh_test_root("persist-cwd");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = Adapter::new_for_test(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    assert_eq!(adapter.sessions[&session_id].cwd, cwd);
    assert_eq!(
        adapter.load_store().sessions[&session_id].cwd.as_deref(),
        Some(cwd.to_string_lossy().as_ref()),
    );
}

#[test]
fn legacy_stored_session_without_cwd_loads_with_request_cwd() {
    let root = fresh_test_root("legacy-cwd");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = Adapter::new_for_test(&root);
    fs::create_dir_all(adapter.state_file.parent().unwrap()).unwrap();
    fs::write(
        &adapter.state_file,
        r#"{"sessions":{"legacy":{"conversation_id":"00000000-0000-4000-8000-000000000001","last_step_idx":4,"model_id":null}}}"#,
    )
    .unwrap();
    let mut params = session_setup_params(&cwd);
    params["sessionId"] = json!("legacy");
    let output = adapter.handle_session_load(json!(9), &params);
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response.get("result").is_some());
    assert_eq!(adapter.sessions["legacy"].cwd, cwd);
}
```

Implement `Adapter::new_for_test` with this deterministic layout, then replace existing `Adapter` struct literals in `src/tests.rs` with the constructor and override only the paths/models relevant to each test:

```rust
#[cfg(test)]
pub(crate) fn new_for_test(root: &std::path::Path) -> Self {
    Self {
        sessions: HashMap::new(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("state").join("sessions.json"),
        available_models: vec!["fake-model\tFake Model".to_string()],
        skip_naration: false,
    }
}
```

Update every existing `handle_session_new`, `handle_session_load`, and `handle_session_resume` test call to include `cwd` and `mcpServers`; tests that intentionally validate missing params remain unchanged. Add `cwd` to direct `Session` literals using that test's workspace directory.

- [ ] **Step 2: Run lifecycle tests and witness RED**

Run:

```powershell
cargo test session_new_rejects_missing_or_relative_cwd -- --nocapture
cargo test session_new_retains_and_persists_cwd -- --nocapture
cargo test legacy_stored_session_without_cwd_loads_with_request_cwd -- --nocapture
```

Expected: failures show that lifecycle params are ignored and `cwd` is absent from session state.

- [ ] **Step 3: Implement lifecycle validation and state compatibility**

Use this validator shape in `src/adapter.rs`:

```rust
fn validate_session_setup(params: &Value) -> Result<PathBuf, String> {
    let cwd = params.get("cwd").and_then(Value::as_str)
        .filter(|cwd| !cwd.is_empty())
        .ok_or_else(|| "cwd must be a non-empty absolute directory".to_string())?;
    let path = PathBuf::from(cwd);
    if !path.is_absolute() || !path.is_dir() {
        return Err("cwd must be an existing absolute directory".to_string());
    }
    let servers = params.get("mcpServers").and_then(Value::as_array)
        .ok_or_else(|| "mcpServers must be an array".to_string())?;
    if !servers.is_empty() {
        return Err("non-empty mcpServers are not supported by agy-acp".to_string());
    }
    Ok(path)
}
```

Store `cwd` in `Session`; serialize it in `persist_session`; accept missing stored `cwd` on read. Change `restore_session` to return the complete `StoredSession`, including entries whose `conversation_id` is still `None`. `session/new` persists immediately. Load/resume validate setup before reading history, apply the request `cwd`, and persist the refreshed context. Remove `Adapter.working_dir` after all call sites use session state.

Use `JsonRpcResponse::error(id, -32602, message)` for setup validation and preserve existing `-32000` unknown-session behavior.

- [ ] **Step 4: Update documentation and run GREEN**

Update README request examples to include absolute `cwd` and `mcpServers: []`; document the explicit rejection of non-empty MCP lists.

Run:

```powershell
cargo fmt -- --check
cargo test session_new_ -- --nocapture
cargo test session_load_ -- --nocapture
cargo test session_resume_ -- --nocapture
cargo test --quiet
```

Expected: lifecycle tests and the default suite pass with zero failures.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/types.rs src/adapter.rs src/main.rs src/tests.rs README.md
git diff --cached --check
git commit -m "feat: bind sessions to ACP working directories"
```

---

### Task 3: Fail-Closed Prompt Preparation

**Files:**
- Modify: `src/adapter.rs`
- Modify: `src/types.rs`
- Modify: `src/tests.rs`

**Interfaces:**
- Produces: `Adapter::prepare_prompt(id, params) -> Result<PromptExecution, JsonRpcResponse>`
- Produces: `Adapter::handle_prepared_prompt(execution, cancelled) -> Vec<String>` as a temporary compatibility boundary used until Task 4 adds the output sink.
- Data: `PromptExecution` owns request ID, session snapshot, validated text, paths, flags, and initial cursor.
- Consumes later: Tasks 4 and 5 replace the compatibility boundary with streamed output and lock-free execution.

- [ ] **Step 1: Write failing preparation tests**

Add tests using the Task 2 helpers:

```rust
#[test]
fn prepare_prompt_rejects_unknown_session_before_log_creation() {
    let root = fresh_test_root("unknown-prompt");
    let mut adapter = Adapter::new_for_test(&root);
    let error = adapter.prepare_prompt(
        json!(4),
        &json!({"sessionId": "missing", "prompt": [{"type": "text", "text": "hello"}]}),
    ).unwrap_err();
    assert_eq!(error.error.as_ref().unwrap()["code"], -32000);
    assert!(!adapter.state_file.parent().unwrap().join("run-logs").exists());
}

#[test]
fn prepare_prompt_rejects_empty_text() {
    let root = fresh_test_root("empty-prompt");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = Adapter::new_for_test(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    let error = adapter.prepare_prompt(
        json!(5),
        &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "  "}]}),
    ).unwrap_err();
    assert_eq!(error.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn prepare_prompt_rejects_unsupported_or_mixed_content() {
    let root = fresh_test_root("mixed-prompt");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = Adapter::new_for_test(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    let error = adapter.prepare_prompt(
        json!(6),
        &json!({"sessionId": session_id, "prompt": [
            {"type": "text", "text": "hello"},
            {"type": "image", "data": "AA==", "mimeType": "image/png"}
        ]}),
    ).unwrap_err();
    assert_eq!(error.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn prepare_prompt_snapshots_session_cwd_model_and_conversation() {
    let root = fresh_test_root("snapshot");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = Adapter::new_for_test(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    let session = adapter.sessions.get_mut(&session_id).unwrap();
    session.model_id = Some("fake-model".to_string());
    session.conversation_id = Some("00000000-0000-4000-8000-000000000002".to_string());
    let execution = adapter.prepare_prompt(
        json!(7),
        &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]}),
    ).unwrap();
    assert_eq!(execution.cwd, cwd);
    assert_eq!(execution.model_id.as_deref(), Some("fake-model"));
    assert_eq!(execution.conversation_id.as_deref(), Some("00000000-0000-4000-8000-000000000002"));
    assert_eq!(execution.prompt_text, "hello");
}
```

- [ ] **Step 2: Run preparation tests and witness RED**

Run:

```powershell
cargo test prepare_prompt_ -- --nocapture
```

Expected: compilation fails because `prepare_prompt` and `PromptExecution` do not exist.

- [ ] **Step 3: Extract validation and owned prompt preparation**

Implement owned structures with these fields:

```rust
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

```

`prepare_prompt` must finish all session/content validation before creating the run-log directory. Every block must have `type: "text"` and string `text`; join blocks with `\n`, trim once, and reject an empty result.

Make `handle_session_prompt` call `prepare_prompt` and immediately return its error response on failure. Move the existing process body behind `handle_prepared_prompt`, taking `PromptExecution` rather than raw params. Replace `self.working_dir` with `execution.cwd`; keep existing mutation/output behavior temporarily so this task changes validation and ownership only.

- [ ] **Step 4: Run preparation and regression tests GREEN**

Run:

```powershell
cargo fmt -- --check
cargo test prepare_prompt_ -- --nocapture
cargo test detects_untrustworthy_print_timeout -- --nocapture
cargo test parallel_invocation_logs_cannot_cross_bind -- --nocapture
cargo test --quiet
```

Expected: focused and default tests pass; `rg -n "working_dir" src` returns no adapter startup-directory dependency.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/adapter.rs src/types.rs src/tests.rs
git diff --cached --check
git commit -m "refactor: separate prompt execution from session state"
```

---

### Task 4: Bounded Single-Writer Output Sink

**Files:**
- Create: `src/output.rs`
- Modify: `src/main.rs`
- Modify: `src/adapter.rs`
- Modify: `src/tests.rs`

**Interfaces:**
- Produces: `output::channel() -> (OutputSender, Receiver<String>)` with capacity 256.
- Produces: `output::write_messages<W: AsyncWrite + Unpin>(writer, receiver) -> io::Result<()>`.
- Produces: `output::send_response(&OutputSender, JsonRpcResponse) -> Result<(), SendError<String>>`.
- Changes: `handle_session_prompt(id, params, cancelled, output)` prepares the request and delegates to `handle_prepared_prompt`.
- Changes: `handle_prepared_prompt` receives `OutputSender`; its polling thread uses `blocking_send` after delta extraction and returns one terminal `JsonRpcResponse`.
- Consumes later: Task 5 routes dispatcher responses through the same sender.

- [ ] **Step 1: Write the failing output serialization test**

```rust
#[tokio::test]
async fn output_writer_emits_complete_lines_in_channel_order() {
    use tokio::io::{duplex, AsyncReadExt};
    let (writer_side, mut reader_side) = duplex(4096);
    let (sender, receiver) = crate::output::channel();
    let writer = tokio::spawn(crate::output::write_messages(writer_side, receiver));
    sender.send("{\"id\":1}".to_string()).await.unwrap();
    sender.send("{\"method\":\"session/update\"}".to_string()).await.unwrap();
    drop(sender);
    writer.await.unwrap().unwrap();
    let mut output = String::new();
    reader_side.read_to_string(&mut output).await.unwrap();
    assert_eq!(output, "{\"id\":1}\n{\"method\":\"session/update\"}\n");
}
```

- [ ] **Step 2: Run the writer test and witness RED**

Run:

```powershell
cargo test output_writer_emits_complete_lines_in_channel_order -- --nocapture
```

Expected: compilation fails because `output` does not exist.

- [ ] **Step 3: Implement the bounded sink and migrate prompt updates**

Create `src/output.rs`:

```rust
use std::io;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use crate::types::JsonRpcResponse;

pub(crate) type OutputSender = mpsc::Sender<String>;

pub(crate) fn channel() -> (OutputSender, mpsc::Receiver<String>) {
    mpsc::channel(256)
}

pub(crate) async fn send_response(
    sender: &OutputSender,
    response: JsonRpcResponse,
) -> Result<(), mpsc::error::SendError<String>> {
    sender.send(serde_json::to_string(&response).expect("JSON-RPC response is serializable")).await
}

pub(crate) async fn write_messages<W>(
    mut writer: W,
    mut receiver: mpsc::Receiver<String>,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    while let Some(message) = receiver.recv().await {
        writer.write_all(message.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }
    Ok(())
}
```

Add `mod output;`. Pass a cloned `OutputSender` into `handle_prepared_prompt`. Polling threads call `blocking_send(line)` after `poll_streaming_delta` returns; async execution calls `send(line).await`. Return only the terminal response to the caller. Start the dedicated writer task in `main` and route synchronous handler lines through the same sender, while retaining the existing long prompt lock until Task 5. Remove all `io::stdout`, `writeln!(stdout, ...)`, and `stdout.flush()` calls from `adapter.rs`.

- [ ] **Step 4: Run GREEN and enforce the no-direct-writer invariant**

Run:

```powershell
cargo fmt -- --check
cargo test output_writer_emits_complete_lines_in_channel_order -- --nocapture
rg -n "io::stdout|writeln!\(stdout|stdout\.flush" src/adapter.rs src/streaming.rs
cargo test --quiet
```

Expected: writer test and default suite pass; `rg` exits 1 because no direct writer remains in adapter/streaming code.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/output.rs src/main.rs src/adapter.rs src/tests.rs
git diff --cached --check
git commit -m "refactor: serialize ACP output through one writer"
```

---

### Task 5: Concurrent Dispatch and Session-Scoped Cancellation

**Files:**
- Create: `src/runtime.rs`
- Modify: `src/main.rs`
- Modify: `src/tests.rs`

**Interfaces:**
- Produces: `ActivePrompts::{register, cancel, cancel_all, complete}`.
- Produces: `ActivePrompt { cancelled: Arc<AtomicBool> }` as an opaque registration token.
- Produces: `execute_prompt(execution, cancelled, output) -> PromptOutcome` with no `Adapter` borrow.
- Produces: `Adapter::apply_prompt_outcome(&PromptOutcome)` for the short post-execution state commit.
- Data: `PromptOutcome` owns session ID, optional conversation binding, last step index, and terminal response.
- Changes: main dispatch parses `IncomingMessage`, queues every synchronous response, prepares prompts under a short adapter lock, then runs executions in spawned tasks.
- Consumes: protocol parser from Task 1, prompt split from Task 3, output sink from Task 4.

- [ ] **Step 1: Write failing active-registry tests**

```rust
#[test]
fn active_prompts_allow_different_sessions_and_reject_duplicates() {
    let active = crate::runtime::ActivePrompts::default();
    let first = active.register("session-a").unwrap();
    assert!(active.register("session-a").is_err());
    assert!(active.register("session-b").is_ok());
    active.complete("session-a", &first);
    assert!(active.register("session-a").is_ok());
}

#[test]
fn cancel_targets_only_the_registered_session() {
    let active = crate::runtime::ActivePrompts::default();
    let first = active.register("session-a").unwrap();
    let second = active.register("session-b").unwrap();
    assert!(active.cancel("session-a"));
    assert!(first.is_cancelled());
    assert!(!second.is_cancelled());
}
```

- [ ] **Step 2: Run registry tests and witness RED**

Run:

```powershell
cargo test active_prompts_ -- --nocapture
cargo test cancel_targets_only -- --nocapture
```

Expected: compilation fails because `runtime::ActivePrompts` does not exist.

- [ ] **Step 3: Implement identity-safe active registrations**

Create `src/runtime.rs` with a registry holding `HashMap<String, Arc<AtomicBool>>`. `register` inserts only into a vacant entry. `complete` removes only when `Arc::ptr_eq` confirms the supplied token is still the registered token. `cancel` loads the matching token and stores `true` with `Ordering::SeqCst`; `cancel_all` snapshots every token under the lock, releases the lock, then marks each token cancelled.

Expose on the opaque token:

```rust
impl ActivePrompt {
    pub(crate) fn cancellation_flag(&self) -> Arc<AtomicBool>;
    #[cfg(test)]
    pub(crate) fn is_cancelled(&self) -> bool;
}
```

- [ ] **Step 4: Rewrite main dispatch without long adapter locks**

First extract the process body from `handle_prepared_prompt` into `execute_prompt`. It owns child spawn, pipe readers, polling, cancellation, final deltas, timeout/non-zero/no-conversation/no-assistant checks, and log cleanup without borrowing `Adapter`. `PromptOutcome` returns the terminal response plus discovered conversation/cursor. `apply_prompt_outcome` preserves existing binding semantics, including diagnostic recovery after a partial failed turn, and never creates a missing session.

Replace direct deserialization with `parse_jsonrpc_line`. Start one `write_messages(tokio::io::stdout(), receiver)` task. Every response is serialized and sent through the bounded sender.

For `session/prompt`, use this ownership order:

```rust
let request_id = id.clone();
let prepared = {
    let mut adapter = adapter.lock().await;
    adapter.prepare_prompt(request_id.clone(), &call.params)
};
match prepared {
    Err(response) => {
        let _ = send_response(&output, response).await;
    }
    Ok(execution) => {
        let session_id = execution.session_id.clone();
        match active.register(&session_id) {
            Err(()) => {
                let _ = send_response(
                    &output,
                    JsonRpcResponse::error(
                        request_id,
                        -32000,
                        "session already has an active prompt",
                    ),
                ).await;
            }
            Ok(registration) => {
                let cancelled = registration.cancellation_flag();
                let task_active = active.clone();
                let task_adapter = adapter.clone();
                let task_output = output.clone();
                let task_done = done.clone();
                tokio::spawn(async move {
                    let outcome = execute_prompt(execution, cancelled, task_output.clone()).await;
                    {
                        let mut adapter = task_adapter.lock().await;
                        adapter.apply_prompt_outcome(&outcome);
                    }
                    let _ = send_response(&task_output, outcome.response).await;
                    task_active.complete(&session_id, &registration);
                    let _ = task_done.send(());
                });
            }
        }
    }
}
```

Notifications: handle only `session/cancel` through `ActivePrompts::cancel`; emit no response. Requests using `session/cancel` retain the existing `{}` result for compatibility. Invalid parse/envelope results are queued immediately. Unknown notifications are ignored; unknown requests return `-32601`.

Wrap dispatch in `run_bridge(...) -> std::io::Result<()>`. Select on the writer task as well as input/completion channels; if the writer finishes early, cancel every active registration, wait for prompt cleanup, and return the writer error (or `BrokenPipe` if it closed without one). `main` prints that local diagnostic and exits with status 1. Background prompt tasks may ignore an individual send error because `run_bridge` owns the fatal writer decision. On stdin EOF, wait for active prompt completions, drop all output senders, await the writer task, and propagate its result.

- [ ] **Step 5: Run concurrency-focused and full default tests GREEN**

Run:

```powershell
cargo fmt -- --check
cargo test active_prompts_ -- --nocapture
cargo test cancel_targets_only -- --nocapture
cargo test malformed_json_returns_parse_error -- --nocapture
cargo test --quiet
```

Expected: all tests pass and `rg -n "adapter\.lock\(\)\.await" src/main.rs` shows no lock guard spanning `execute_prompt(...).await`.

- [ ] **Step 6: Commit Task 5**

```powershell
git add src/runtime.rs src/main.rs src/tests.rs
git diff --cached --check
git commit -m "feat: run ACP prompts concurrently by session"
```

---

### Task 6: Deterministic Fake-`agy` Integration and Final Documentation

**Files:**
- Modify: `src/adapter.rs`
- Modify: `src/tests.rs`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Produces: internal `CommandSpec { program: OsString, prefix_args: Vec<OsString> }` captured by `PromptExecution`.
- Produces: `CommandSpec::new(program, prefix_args)` and test-only `Adapter::set_command_for_test(CommandSpec)`.
- Production default: program `agy`, no prefix args.
- Test use: PowerShell script on Windows or POSIX shell script on Unix; no environment-level PATH mutation and no external API.

- [ ] **Step 1: Write failing deterministic orchestration tests**

Create `fake_agy_command(root: &Path) -> CommandSpec`. On Windows it writes this PowerShell script and returns `powershell.exe` with prefix args `-NoProfile`, `-NonInteractive`, `-File`, and the script path:

```powershell
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$logFile = $null
$prompt = $null
for ($i = 0; $i -lt $Rest.Count; $i++) {
    if ($Rest[$i] -eq '--log-file') { $logFile = $Rest[$i + 1] }
    if ($Rest[$i] -eq '-p') { $prompt = $Rest[$i + 1] }
}
Set-Content -LiteralPath $logFile -Value 'Created conversation 00000000-0000-4000-8000-000000000099'
if ($prompt -eq 'slow') { Start-Sleep -Milliseconds 3000 }
[Console]::Out.WriteLine('fake assistant response')
```

On Unix it writes a shell script and returns `/bin/sh` with the script path as its prefix arg:

```sh
log_file=''
prompt=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --log-file) shift; log_file="$1" ;;
    -p) shift; prompt="$1" ;;
  esac
  shift
done
printf '%s\n' 'Created conversation 00000000-0000-4000-8000-000000000099' > "$log_file"
[ "$prompt" = 'slow' ] && sleep 3
printf '%s\n' 'fake assistant response'
```

Write the platform helper with concrete `CommandSpec` values:

```rust
#[cfg(windows)]
fn fake_agy_command(root: &std::path::Path) -> CommandSpec {
    let script = root.join("fake-agy.ps1");
    fs::write(&script, WINDOWS_FAKE_AGY_SCRIPT).unwrap();
    CommandSpec::new(
        "powershell.exe",
        vec![
            "-NoProfile".into(),
            "-NonInteractive".into(),
            "-File".into(),
            script.into_os_string(),
        ],
    )
}

#[cfg(not(windows))]
fn fake_agy_command(root: &std::path::Path) -> CommandSpec {
    let script = root.join("fake-agy.sh");
    fs::write(&script, UNIX_FAKE_AGY_SCRIPT).unwrap();
    CommandSpec::new("/bin/sh", vec![script.into_os_string()])
}
```

Define `WINDOWS_FAKE_AGY_SCRIPT` and `UNIX_FAKE_AGY_SCRIPT` as raw string constants containing the exact scripts above.

Use this concrete harness for the async tests:

```rust
struct ConcurrentHarness {
    adapter: Adapter,
    output: crate::output::OutputSender,
    receiver: tokio::sync::mpsc::Receiver<String>,
}

impl ConcurrentHarness {
    fn new(label: &str) -> Self {
        let root = fresh_test_root(label);
        let mut adapter = Adapter::new_for_test(&root);
        adapter.set_command_for_test(fake_agy_command(&root));
        for session_id in ["session-a", "session-b"] {
            let cwd = root.join(session_id);
            fs::create_dir_all(&cwd).unwrap();
            adapter.sessions.insert(session_id.to_string(), crate::types::Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
                cwd,
            });
        }
        let (output, receiver) = crate::output::channel();
        Self { adapter, output, receiver }
    }

    fn prepare(&mut self, session_id: &str, text: &str) -> crate::adapter::PromptExecution {
        self.adapter.prepare_prompt(
            json!(Uuid::new_v4().to_string()),
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": text}]}),
        ).unwrap()
    }
}
```

Add these async tests:

```rust
#[tokio::test]
async fn prompts_for_two_sessions_execute_without_a_global_adapter_lock() {
    let mut harness = ConcurrentHarness::new("parallel");
    let slow = harness.prepare("session-a", "slow");
    let fast = harness.prepare("session-b", "fast");
    let slow_task = tokio::spawn(execute_prompt(
        slow,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = std::time::Instant::now();
    let fast_outcome = execute_prompt(
        fast,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    ).await;
    assert!(fast_outcome.response.error.is_none());
    assert!(started.elapsed() < Duration::from_millis(1500));
    assert!(slow_task.await.unwrap().response.error.is_none());
}

#[tokio::test]
async fn cancellation_finishes_a_slow_fake_child_without_waiting_for_timeout() {
    let mut harness = ConcurrentHarness::new("cancel");
    let execution = harness.prepare("session-a", "slow");
    let cancelled = Arc::new(AtomicBool::new(false));
    let task = tokio::spawn(execute_prompt(
        execution,
        cancelled.clone(),
        harness.output.clone(),
    ));
    tokio::time::sleep(Duration::from_millis(150)).await;
    cancelled.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(outcome.response.result.unwrap()["stopReason"], "cancelled");
}

#[tokio::test]
async fn concurrent_output_is_valid_newline_delimited_json() {
    let mut harness = ConcurrentHarness::new("jsonl");
    let first = harness.prepare("session-a", "fast");
    let second = harness.prepare("session-b", "fast");
    let one = execute_prompt(first, Arc::new(AtomicBool::new(false)), harness.output.clone());
    let two = execute_prompt(second, Arc::new(AtomicBool::new(false)), harness.output.clone());
    let _ = tokio::join!(one, two);
    drop(harness.output);
    while let Some(line) = harness.receiver.recv().await {
        let value: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["jsonrpc"], "2.0");
    }
}
```

- [ ] **Step 2: Run integration tests and witness RED**

Run:

```powershell
cargo test prompts_for_two_sessions_execute_without_a_global_adapter_lock -- --nocapture
cargo test cancellation_finishes_a_slow_fake_child_without_waiting_for_timeout -- --nocapture
cargo test concurrent_output_is_valid_newline_delimited_json -- --nocapture
```

Expected: compilation fails because command injection and the fake helper do not exist.

- [ ] **Step 3: Add minimal command injection and make integration tests GREEN**

Keep `CommandSpec` internal and initialized to `agy` in production constructors. `fetch_available_models` remains unchanged. Build the child command by applying prefix args first and existing ACP-generated args second. Do not add a public environment variable or CLI flag.

On Windows use `powershell.exe -NoProfile -NonInteractive -File <script>`; on Unix use `/bin/sh <script>`. Script contents and prompt strings contain no credentials or user data.

- [ ] **Step 4: Update project operating documentation**

Update README architecture/behavior sections and AGENTS project notes to state:

- ACP lifecycle requires `cwd` and `mcpServers`.
- Prompts are concurrent across sessions and serialized within a session.
- Output has one writer.
- Non-text prompt blocks and non-empty MCP server lists fail explicitly.
- The safe local ignored-test command is `cargo test -- --include-ignored --skip test_e2e_`.

- [ ] **Step 5: Run final non-live verification**

Use an isolated HOME and run exactly:

```powershell
$verifyHome = Join-Path $env:TEMP 'agy-acp-p0-final-home'
New-Item -ItemType Directory -Force -Path $verifyHome | Out-Null
$env:HOME = $verifyHome
cargo fmt -- --check
cargo build
cargo test
cargo test -- --include-ignored --skip test_e2e_
if (rg -n "io::stdout|writeln!\(stdout|stdout\.flush" src/adapter.rs src/streaming.rs) {
    throw 'direct stdout writer remains outside the output module'
}
git diff --check
git status --short --branch
```

Expected: format/build pass; default and filesystem-only ignored tests have zero failures; direct-writer `rg` returns no matches; diff check is clean. Do not run tests named `test_e2e_`.

- [ ] **Step 6: Rebuild release and verify the exact candidate binary without a live prompt**

Run:

```powershell
cargo build --release
$candidate = (Resolve-Path '.\target\release\agy-acp.exe').Path
$cwd = (Get-Location).Path
$inputLines = @(
    (@{jsonrpc='2.0'; id=1; method='initialize'; params=@{protocolVersion=1; clientCapabilities=@{}}} | ConvertTo-Json -Compress -Depth 8),
    (@{jsonrpc='2.0'; id=2; method='session/new'; params=@{cwd=$cwd; mcpServers=@()}} | ConvertTo-Json -Compress -Depth 8)
)
$outputLines = $inputLines | & $candidate
$messages = @($outputLines | ForEach-Object { $_ | ConvertFrom-Json })
if ($messages.Count -ne 2) { throw "expected 2 ACP responses, got $($messages.Count)" }
if ($messages[0].result.protocolVersion -ne 1) { throw 'initialize handshake failed' }
if (-not $messages[1].result.sessionId) { throw 'session/new handshake failed' }
```

Expected: the release build in this worktree succeeds and its exact binary passes initialize plus `session/new` using the same ACP lifecycle shape Paseo sends. Do not run the configured Paseo provider diagnostic here: Paseo still points to the stable binary under the main checkout, so that command would test the wrong artifact. Re-run Paseo diagnostic only after separately approved promotion updates that binary.

- [ ] **Step 7: Review diff and commit Task 6**

```powershell
git diff --check
git diff --stat 743d74e
git status --short
git add src/adapter.rs src/tests.rs README.md AGENTS.md
git diff --cached --check
git commit -m "test: verify concurrent ACP runtime"
```

Do not push. Report exact commits, verification counts, candidate-handshake result, the unchanged Paseo production binding, and the unverified live-Gemini boundary.
