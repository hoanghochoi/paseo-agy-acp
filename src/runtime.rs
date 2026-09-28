use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock,
};
use std::time::{Duration, SystemTime};

use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::adapter::Adapter;
use crate::mcp::RunConfig;
use crate::output::{self, OutputSender};
use crate::protocol::{parse_jsonrpc_line, IncomingMessage, RpcCall};
use crate::streaming::poll_streaming_delta;
use crate::types::{JsonRpcNotification, JsonRpcResponse, PromptExecution, StreamingState};
use crate::OVERSIZED_INPUT_SENTINEL;

pub(crate) const MAX_RUNTIME_ERROR_MESSAGE_LEN: usize = 256;
pub(crate) const MAX_CHILD_STDOUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_CHILD_STDOUT_DRAIN_TIME: Duration = Duration::from_secs(2);
const MIN_POLL_INTERVAL: Duration = Duration::from_millis(100);
const MAX_POLL_INTERVAL: Duration = Duration::from_millis(500);
pub(crate) const MAX_RETAINED_RUN_LOGS: usize = 64;
pub(crate) const MAX_RETAINED_RUN_LOG_BYTES: u64 = 16 * 1024 * 1024;

static ACTIVE_RUN_LOGS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();

fn active_run_logs() -> &'static Mutex<HashSet<PathBuf>> {
    ACTIVE_RUN_LOGS.get_or_init(|| Mutex::new(HashSet::new()))
}

pub(crate) fn register_active_run_log(path: &Path) {
    active_run_logs().lock().unwrap().insert(path.to_path_buf());
}

pub(crate) fn unregister_active_run_log(path: &Path) {
    active_run_logs().lock().unwrap().remove(path);
}

pub(crate) fn prune_retained_run_logs(run_logs_dir: &Path, keep: Option<&Path>) {
    let active = active_run_logs().lock().unwrap();
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(run_logs_dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("log")
            || active.contains(&path)
            || keep.is_some_and(|keep| keep == path)
        {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if Uuid::parse_str(stem).is_err() {
            continue;
        }
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        files.push((
            path,
            metadata.len(),
            metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        ));
    }

    files.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));

    let mut retained_bytes = 0u64;
    for (index, (path, size, _)) in files.into_iter().enumerate() {
        let within_count = index < MAX_RETAINED_RUN_LOGS;
        let within_bytes = retained_bytes.saturating_add(size) <= MAX_RETAINED_RUN_LOG_BYTES;
        // Always retain the newest file even when one individual diagnostic is
        // larger than the aggregate byte budget; older files remain bounded.
        if within_count && (within_bytes || index == 0) {
            retained_bytes = retained_bytes.saturating_add(size);
        } else {
            let _ = fs::remove_file(path);
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct LimitedChildOutput {
    pub(crate) bytes: Vec<u8>,
    pub(crate) truncated: bool,
}

pub(crate) async fn read_limited_child_output<R>(mut reader: R) -> LimitedChildOutput
where
    R: AsyncRead + Unpin,
{
    let mut output = LimitedChildOutput::default();
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                let remaining = MAX_CHILD_STDOUT_BYTES.saturating_sub(output.bytes.len());
                let retained = read.min(remaining);
                output.bytes.extend_from_slice(&buffer[..retained]);
                if retained < read {
                    output.truncated = true;
                }
            }
        }
    }
    output
}

async fn terminate_child_tree(child: &mut Child) -> io::Result<()> {
    #[cfg(windows)]
    if let Some(pid) = child.id() {
        let pid = pid.to_string();
        if let Ok(status) = Command::new("taskkill")
            .args(["/PID", &pid, "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
        {
            if status.success() {
                return Ok(());
            }
        }
    }
    child.kill().await
}

pub(crate) fn next_poll_delay(current: Duration, emitted_delta: bool) -> Duration {
    if emitted_delta {
        return MIN_POLL_INTERVAL;
    }
    current
        .checked_mul(2)
        .unwrap_or(MAX_POLL_INTERVAL)
        .max(MIN_POLL_INTERVAL)
        .min(MAX_POLL_INTERVAL)
}

fn run_log_reference(run_log_path: &Path) -> String {
    run_log_path
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".log"))
        .and_then(|stem| Uuid::parse_str(stem).ok())
        .map(|id| format!("run-logs/{id}.log"))
        .unwrap_or_else(|| "run-logs/unavailable.log".to_string())
}

pub(crate) fn retained_run_log_message(summary: &str, run_log_path: &Path) -> String {
    let reference = run_log_reference(run_log_path);
    let message = format!("{summary}; run log retained at {reference}");
    if message.len() <= MAX_RUNTIME_ERROR_MESSAGE_LEN {
        message
    } else {
        format!("agy runtime failure; run log retained at {reference}")
    }
}

#[derive(Clone, Default)]
pub(crate) struct ActivePrompts {
    registrations: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

pub(crate) struct ActivePrompt {
    cancelled: Arc<AtomicBool>,
}

pub(crate) struct PromptCompletion {
    active: ActivePrompts,
    session_id: String,
    registration: ActivePrompt,
    done: mpsc::UnboundedSender<()>,
}

pub(crate) struct PollerCompletion {
    completed: oneshot::Receiver<()>,
}

impl ActivePrompts {
    pub(crate) fn register(&self, session_id: &str) -> Result<ActivePrompt, ()> {
        let mut registrations = self.registrations.lock().unwrap();
        if registrations.contains_key(session_id) {
            return Err(());
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        registrations.insert(session_id.to_string(), Arc::clone(&cancelled));
        Ok(ActivePrompt { cancelled })
    }

    pub(crate) fn cancel(&self, session_id: &str) -> bool {
        let cancelled = self.registrations.lock().unwrap().get(session_id).cloned();
        if let Some(cancelled) = cancelled {
            cancelled.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub(crate) fn cancel_all(&self) {
        let registrations: Vec<_> = self
            .registrations
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect();
        for cancelled in registrations {
            cancelled.store(true, Ordering::SeqCst);
        }
    }

    pub(crate) fn complete(&self, session_id: &str, registration: &ActivePrompt) {
        let mut registrations = self.registrations.lock().unwrap();
        let is_current = registrations
            .get(session_id)
            .is_some_and(|current| Arc::ptr_eq(current, &registration.cancelled));
        if is_current {
            registrations.remove(session_id);
        }
    }
}

impl ActivePrompt {
    pub(crate) fn cancellation_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }

    #[cfg(test)]
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl PromptCompletion {
    pub(crate) fn new(
        active: ActivePrompts,
        session_id: String,
        registration: ActivePrompt,
        done: mpsc::UnboundedSender<()>,
    ) -> Self {
        Self {
            active,
            session_id,
            registration,
            done,
        }
    }
}

impl Drop for PromptCompletion {
    fn drop(&mut self) {
        self.active.complete(&self.session_id, &self.registration);
        let _ = self.done.send(());
    }
}

impl PollerCompletion {
    pub(crate) async fn wait(self) {
        let _ = self.completed.await;
    }
}

pub(crate) fn spawn_poller_thread<F>(poller: F) -> PollerCompletion
where
    F: FnOnce() + Send + 'static,
{
    let (completed, receiver) = oneshot::channel();
    std::thread::spawn(move || {
        poller();
        let _ = completed.send(());
    });
    PollerCompletion {
        completed: receiver,
    }
}

#[derive(Debug)]
pub(crate) struct PromptOutcome {
    pub(crate) session_id: String,
    pub(crate) conversation_id: Option<String>,
    pub(crate) last_step_idx: i64,
    pub(crate) response: JsonRpcResponse,
    pub(crate) run_log_path: PathBuf,
    pub(crate) remove_run_log_on_commit: bool,
}

pub(crate) fn finalize_prompt_outcome(
    adapter: &mut Adapter,
    outcome: PromptOutcome,
) -> JsonRpcResponse {
    let response_id = outcome.response.id.clone();
    let run_log_path = outcome.run_log_path.clone();
    let run_logs_dir = run_log_path.parent().map(Path::to_path_buf);
    if adapter.apply_prompt_outcome(&outcome).is_err() {
        unregister_active_run_log(&run_log_path);
        if let Some(run_logs_dir) = run_logs_dir.as_deref() {
            prune_retained_run_logs(run_logs_dir, Some(&run_log_path));
        }
        return JsonRpcResponse::error(response_id, -32603, "failed to persist session state");
    }
    if outcome.remove_run_log_on_commit {
        let _ = fs::remove_file(&run_log_path);
    }
    unregister_active_run_log(&run_log_path);
    if let Some(run_logs_dir) = run_logs_dir.as_deref() {
        let keep = outcome
            .response
            .error
            .as_ref()
            .map(|_| run_log_path.as_path());
        prune_retained_run_logs(run_logs_dir, keep);
    }
    outcome.response
}

pub(crate) async fn execute_prompt(
    execution: PromptExecution,
    cancelled: Arc<AtomicBool>,
    output: OutputSender,
) -> PromptOutcome {
    let PromptExecution {
        id,
        session_id,
        prompt_text,
        cwd,
        conversation_id,
        model_id,
        initial_step_idx,
        conversations_dir,
        state_dir,
        skip_naration,
        command,
        add_dir,
        mcp_config,
    } = execution;
    let run_logs_dir = state_dir.join("run-logs");
    let run_log_path = run_logs_dir.join(format!("{}.log", Uuid::new_v4()));
    register_active_run_log(&run_log_path);

    let mut args: Vec<OsString> = vec![
        "--add-dir".into(),
        cwd.as_os_str().to_os_string(),
        "--log-file".into(),
        run_log_path.as_os_str().to_os_string(),
        "--print-timeout".into(),
        std::env::var("AGY_PRINT_TIMEOUT")
            .unwrap_or_else(|_| "24h".to_string())
            .into(),
    ];
    if let Some(add_dir) = &add_dir {
        args.push("--add-dir".into());
        args.push(add_dir.as_os_str().to_os_string());
    }
    if let Ok(extra) = std::env::var("AGY_EXTRA_ARGS") {
        let Some(extra_args) = crate::adapter::parse_extra_args_bounded(&extra) else {
            return PromptOutcome {
                session_id,
                conversation_id,
                last_step_idx: initial_step_idx,
                response: JsonRpcResponse::error(
                    id,
                    -32602,
                    "AGY_EXTRA_ARGS is invalid or exceeds maximum size",
                ),
                run_log_path,
                remove_run_log_on_commit: false,
            };
        };
        args.extend(extra_args);
    }
    // Held until the run ends: `agy` may start an MCP server at any point of its turn.
    let _mcp_run = match mcp_config
        .as_ref()
        .map(|config| RunConfig::write(config, &state_dir))
    {
        None => None,
        Some(Ok(run)) => {
            args.push("--add-dir".into());
            args.push(run.dir().as_os_str().to_os_string());
            Some(run)
        }
        Some(Err(error)) => {
            return PromptOutcome {
                session_id,
                conversation_id,
                last_step_idx: initial_step_idx,
                response: JsonRpcResponse::error(
                    id,
                    -32000,
                    &format!("failed to write the session's MCP configuration: {error}"),
                ),
                run_log_path,
                remove_run_log_on_commit: false,
            };
        }
    };
    if let Some(conv_id) = &conversation_id {
        args.push("--conversation".into());
        args.push(conv_id.into());
    }
    if let Some(model_id) = &model_id {
        args.push("--model".into());
        args.push(model_id.into());
    }
    args.push("-p".into());
    args.push(prompt_text.into());

    let spawn_result = Command::new(&command.program)
        .args(&command.prefix_args)
        .args(&args)
        .current_dir(&cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = match spawn_result {
        Ok(child) => child,
        Err(error) => {
            return PromptOutcome {
                session_id,
                conversation_id,
                last_step_idx: initial_step_idx,
                response: JsonRpcResponse::error(
                    id,
                    -32000,
                    &format!("failed to run agy: {error}"),
                ),
                run_log_path,
                remove_run_log_on_commit: false,
            };
        }
    };

    let mut stdout = child.stdout.take();
    let mut stdout_reader = tokio::spawn(async move {
        if let Some(stdout) = stdout.take() {
            read_limited_child_output(stdout).await
        } else {
            LimitedChildOutput::default()
        }
    });

    let mut stderr = child.stderr.take();
    let mut stderr_reader = tokio::spawn(async move {
        if let Some(mut stderr) = stderr.take() {
            let _ = tokio::io::copy(&mut stderr, &mut tokio::io::sink()).await;
        }
    });

    let streaming_state = Arc::new(Mutex::new(StreamingState {
        conversation_id,
        base_step_idx: initial_step_idx,
        last_step_idx: initial_step_idx,
        had_agent_text: false,
        agent_text_lengths: HashMap::new(),
        thought_text_lengths: HashMap::new(),
        emitted_tool_steps: HashSet::new(),
        last_title: None,
        skip_naration,
        child_pid: child.id(),
    }));
    let stop_polling = Arc::new(AtomicBool::new(false));
    let poll_conversations_dir = conversations_dir.clone();
    let poll_run_log_path = run_log_path.clone();
    let poll_session_id = session_id.clone();
    let poll_state = Arc::clone(&streaming_state);
    let poll_stop = Arc::clone(&stop_polling);
    let poll_output = output.clone();

    let poller = spawn_poller_thread(move || {
        let mut poll_delay = MIN_POLL_INTERVAL;
        while !poll_stop.load(Ordering::SeqCst) {
            let mut emitted_delta = false;
            for line in poll_streaming_delta(
                &poll_conversations_dir,
                Some(&poll_run_log_path),
                &poll_session_id,
                &poll_state,
            ) {
                emitted_delta = true;
                if poll_output.blocking_send(line).is_err() {
                    return;
                }
            }
            poll_delay = next_poll_delay(poll_delay, emitted_delta);
            std::thread::sleep(poll_delay);
        }
    });

    let mut was_cancelled = false;
    let result = tokio::select! {
        result = child.wait() => result,
        _ = async {
            while !cancelled.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        } => {
            was_cancelled = true;
            let _ = terminate_child_tree(&mut child).await;
            child.wait().await
        }
    };
    let stdout_output =
        match tokio::time::timeout(MAX_CHILD_STDOUT_DRAIN_TIME, &mut stdout_reader).await {
            Ok(Ok(output)) => output,
            Ok(Err(_)) | Err(_) => {
                stdout_reader.abort();
                LimitedChildOutput {
                    bytes: Vec::new(),
                    truncated: true,
                }
            }
        };
    let stdout_text = String::from_utf8_lossy(&stdout_output.bytes)
        .trim()
        .to_string();
    let stdout_truncated = stdout_output.truncated;
    if tokio::time::timeout(MAX_CHILD_STDOUT_DRAIN_TIME, &mut stderr_reader)
        .await
        .is_err()
    {
        stderr_reader.abort();
    }
    stop_polling.store(true, Ordering::SeqCst);
    poller.wait().await;

    let mut final_lines = Vec::new();
    for attempt in 0..3 {
        final_lines.extend(poll_streaming_delta(
            &conversations_dir,
            Some(&run_log_path),
            &session_id,
            &streaming_state,
        ));
        if attempt < 2 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    let had_agent_text_before_stdout = streaming_state.lock().unwrap().had_agent_text;
    if !was_cancelled
        && !stdout_truncated
        && !had_agent_text_before_stdout
        && !stdout_text.is_empty()
    {
        final_lines.push(
            serde_json::to_string(&JsonRpcNotification {
                jsonrpc: "2.0",
                method: "session/update".to_string(),
                params: json!({
                    "sessionId": session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": { "type": "text", "text": stdout_text },
                    },
                }),
            })
            .unwrap(),
        );
        streaming_state.lock().unwrap().had_agent_text = true;
    }

    for line in final_lines {
        if output.send(line).await.is_err() {
            break;
        }
    }

    let state = streaming_state.lock().unwrap();
    let bound_conv_id = state.conversation_id.clone();
    let new_step_idx = state.last_step_idx;
    let had_agent_text = state.had_agent_text;
    drop(state);

    let run_timed_out = crate::db::agy_run_timed_out(&run_log_path);
    let success_response = JsonRpcResponse::success(
        id.clone(),
        json!({
            "stopReason": if was_cancelled { "cancelled" } else { "end_turn" }
        }),
    );
    let response = match result {
        Ok(status) => {
            if !was_cancelled && run_timed_out {
                let message = retained_run_log_message(
                    "agy print mode timed out before a trustworthy handback",
                    &run_log_path,
                );
                JsonRpcResponse::error(id, -32000, &message)
            } else if !was_cancelled && !status.success() {
                let status_summary = status
                    .code()
                    .map(|code| format!("agy exited with status code {code}"))
                    .unwrap_or_else(|| "agy exited without a status code".to_string());
                let message = retained_run_log_message(&status_summary, &run_log_path);
                eprintln!("[agy-acp] WARN: {message}");
                JsonRpcResponse::error(id, -32000, &message)
            } else if !was_cancelled && bound_conv_id.is_none() {
                let message = retained_run_log_message(
                    "agy completed but no conversation ID was found",
                    &run_log_path,
                );
                JsonRpcResponse::error(id, -32000, &message)
            } else if !was_cancelled && !had_agent_text {
                if stdout_truncated {
                    JsonRpcResponse::error(
                        id,
                        -32000,
                        "agy stdout exceeded the maximum buffered size",
                    )
                } else {
                    JsonRpcResponse::error(
                        id,
                        -32000,
                        "agy completed without an assistant response",
                    )
                }
            } else {
                success_response
            }
        }
        Err(error) => {
            JsonRpcResponse::error(id, -32000, &format!("failed to wait for agy: {error}"))
        }
    };

    let remove_run_log_on_commit = response.error.is_none();

    PromptOutcome {
        session_id,
        conversation_id: bound_conv_id,
        last_step_idx: new_step_idx,
        response,
        run_log_path,
        remove_run_log_on_commit,
    }
}

pub(crate) async fn run_bridge<W>(
    adapter: Adapter,
    mut input: mpsc::UnboundedReceiver<String>,
    writer: W,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin + Send + 'static,
{
    let adapter = Arc::new(tokio::sync::Mutex::new(adapter));
    let active = ActivePrompts::default();
    let (done, mut completions) = mpsc::unbounded_channel::<()>();
    let (output, receiver) = output::channel();
    let mut writer_task = tokio::spawn(output::write_messages(writer, receiver));
    let mut input_open = true;
    let mut pending_prompts = 0usize;

    loop {
        if !input_open && pending_prompts == 0 {
            break;
        }

        tokio::select! {
            writer_result = &mut writer_task => {
                return handle_writer_exit(
                    &active,
                    pending_prompts,
                    &mut completions,
                    writer_result,
                ).await;
            }
            completion = completions.recv(), if pending_prompts > 0 => {
                if completion.is_some() {
                    pending_prompts = pending_prompts.saturating_sub(1);
                }
            }
            next = input.recv(), if input_open => {
                let Some(line) = next else {
                    input_open = false;
                    continue;
                };
                if dispatch_line(
                    &line,
                    &adapter,
                    &active,
                    &done,
                    &output,
                ).await {
                    pending_prompts += 1;
                }
            }
        }
    }

    drop(done);
    drop(output);
    writer_result_to_io(writer_task.await, false)
}

async fn dispatch_line(
    line: &str,
    adapter: &Arc<tokio::sync::Mutex<Adapter>>,
    active: &ActivePrompts,
    done: &mpsc::UnboundedSender<()>,
    output: &OutputSender,
) -> bool {
    if line == OVERSIZED_INPUT_SENTINEL {
        let _ = output::send_response(
            output,
            JsonRpcResponse::error(Value::Null, -32600, "Request exceeds maximum line length"),
        )
        .await;
        return false;
    }
    match parse_jsonrpc_line(line) {
        Err(response) => {
            let _ = output::send_response(output, response).await;
            false
        }
        Ok(IncomingMessage::Notification { call }) => {
            if call.method == "session/cancel" {
                cancel_from_params(active, &call.params);
            }
            false
        }
        Ok(IncomingMessage::Request { id, call }) => {
            dispatch_request(id, call, adapter, active, done, output).await
        }
    }
}

async fn dispatch_request(
    id: Value,
    call: RpcCall,
    adapter: &Arc<tokio::sync::Mutex<Adapter>>,
    active: &ActivePrompts,
    done: &mpsc::UnboundedSender<()>,
    output: &OutputSender,
) -> bool {
    match call.method.as_str() {
        "initialize" => {
            let response = {
                let adapter = adapter.lock().await;
                adapter.handle_initialize(id)
            };
            let _ = output::send_response(output, response).await;
        }
        "session/new" => {
            let response = {
                let mut adapter = adapter.lock().await;
                adapter.handle_session_new(id, &call.params)
            };
            let _ = output::send_response(output, response).await;
        }
        "session/load" => {
            let messages = {
                let mut adapter = adapter.lock().await;
                adapter.handle_session_load(id, &call.params)
            };
            for message in messages {
                if output.send(message).await.is_err() {
                    break;
                }
            }
        }
        "session/resume" => {
            let response = {
                let mut adapter = adapter.lock().await;
                adapter.handle_session_resume(id, &call.params)
            };
            let _ = output::send_response(output, response).await;
        }
        "session/prompt" => {
            let request_id = id.clone();
            let prepared = {
                let mut adapter = adapter.lock().await;
                adapter.prepare_prompt(request_id.clone(), &call.params)
            };
            match prepared {
                Err(response) => {
                    let _ = output::send_response(output, response).await;
                }
                Ok(execution) => {
                    let session_id = execution.session_id.clone();
                    match active.register(&session_id) {
                        Err(()) => {
                            let _ = output::send_response(
                                output,
                                JsonRpcResponse::error(
                                    request_id,
                                    -32000,
                                    "session already has an active prompt",
                                ),
                            )
                            .await;
                        }
                        Ok(registration) => {
                            {
                                let mut adapter = adapter.lock().await;
                                adapter.mark_prompt_active(&session_id);
                            }
                            let cancelled = registration.cancellation_flag();
                            let task_adapter = Arc::clone(adapter);
                            let task_output = output.clone();
                            let completion = PromptCompletion::new(
                                active.clone(),
                                session_id.clone(),
                                registration,
                                done.clone(),
                            );
                            tokio::spawn(async move {
                                let _completion = completion;
                                let outcome =
                                    execute_prompt(execution, cancelled, task_output.clone()).await;
                                let response = {
                                    let mut adapter = task_adapter.lock().await;
                                    let response = finalize_prompt_outcome(&mut adapter, outcome);
                                    adapter.mark_prompt_complete(&session_id);
                                    response
                                };
                                let _ = output::send_response(&task_output, response).await;
                            });
                            return true;
                        }
                    }
                }
            }
        }
        "session/cancel" => {
            cancel_from_params(active, &call.params);
            let _ = output::send_response(output, JsonRpcResponse::success(id, json!({}))).await;
        }
        "session/set_model" | "session/setModel" => {
            let response = {
                let mut adapter = adapter.lock().await;
                adapter.handle_session_set_model(id, &call.params)
            };
            let _ = output::send_response(output, response).await;
        }
        "session/set_config_option" | "session/setConfigOption" => {
            let response = {
                let mut adapter = adapter.lock().await;
                adapter.handle_session_set_config_option(id, &call.params)
            };
            let _ = output::send_response(output, response).await;
        }
        method => {
            let _ = output::send_response(
                output,
                JsonRpcResponse::error(id, -32601, &format!("method not found: {method}")),
            )
            .await;
        }
    }
    false
}

fn cancel_from_params(active: &ActivePrompts, params: &Value) {
    if let Some(session_id) = params.get("sessionId").and_then(Value::as_str) {
        active.cancel(session_id);
    }
}

async fn drain_prompt_tasks(mut pending: usize, completions: &mut mpsc::UnboundedReceiver<()>) {
    while pending > 0 {
        if completions.recv().await.is_none() {
            break;
        }
        pending -= 1;
    }
}

pub(crate) async fn handle_writer_exit(
    active: &ActivePrompts,
    pending_prompts: usize,
    completions: &mut mpsc::UnboundedReceiver<()>,
    writer_result: Result<io::Result<()>, tokio::task::JoinError>,
) -> io::Result<()> {
    active.cancel_all();
    drain_prompt_tasks(pending_prompts, completions).await;
    writer_result_to_io(writer_result, true)
}

fn writer_result_to_io(
    result: Result<io::Result<()>, tokio::task::JoinError>,
    early: bool,
) -> io::Result<()> {
    match result {
        Ok(Ok(())) if early => Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "output writer closed before bridge shutdown",
        )),
        Ok(result) => result,
        Err(error) => Err(io::Error::other(format!(
            "output writer task failed: {error}"
        ))),
    }
}
