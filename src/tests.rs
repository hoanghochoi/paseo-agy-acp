use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

#[tokio::test]
async fn output_writer_emits_complete_lines_in_channel_order() {
    use tokio::io::{duplex, AsyncReadExt};

    let (writer_side, mut reader_side) = duplex(4096);
    let (sender, receiver) = crate::output::channel();
    let writer = tokio::spawn(crate::output::write_messages(writer_side, receiver));
    sender.send("{\"id\":1}".to_string()).await.unwrap();
    sender
        .send("{\"method\":\"session/update\"}".to_string())
        .await
        .unwrap();
    drop(sender);
    writer.await.unwrap().unwrap();
    let mut output = String::new();
    reader_side.read_to_string(&mut output).await.unwrap();
    assert_eq!(output, "{\"id\":1}\n{\"method\":\"session/update\"}\n");
}

use crate::adapter::{filter_narration, Adapter};
use crate::protobuf::{
    extract_text_from_step_payload, extract_thought_from_step_payload,
    extract_title_from_step_payload, extract_tool_name, extract_tool_update_from_step_payload,
    extract_user_text_from_step_payload, get_proto_field, is_tool_step_type, read_varint,
};
use crate::protocol::{parse_jsonrpc_line, IncomingMessage};
use crate::types::{CommandSpec, JsonRpcResponse, SessionStore, StoredSession};
use crate::Cli;
use clap::Parser;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::AsyncWrite;

const TEST_CONVERSATION_A: &str = "00000000-0000-4000-8000-000000000101";
const TEST_CONVERSATION_B: &str = "00000000-0000-4000-8000-000000000102";
const TEST_CONVERSATION_AFTER_ERROR: &str = "00000000-0000-4000-8000-000000000103";
const TEST_CONVERSATION_LATE: &str = "00000000-0000-4000-8000-000000000104";
const TEST_CONVERSATION_ABC: &str = "00000000-0000-4000-8000-000000000105";
const TEST_CONVERSATION_XYZ: &str = "00000000-0000-4000-8000-000000000106";
const TEST_CONVERSATION_NR: &str = "00000000-0000-4000-8000-000000000107";
const TEST_CONVERSATION_M1: &str = "00000000-0000-4000-8000-000000000108";
const TEST_CONVERSATION_LOAD: &str = "00000000-0000-4000-8000-000000000109";
const TEST_CONVERSATION_RESUME: &str = "00000000-0000-4000-8000-000000000110";

#[test]
fn oversized_protocol_line_is_rejected_without_forwarding_payload() {
    let oversized = format!("{}\n", "x".repeat(crate::MAX_INPUT_LINE_BYTES + 1));
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();

    crate::forward_input_lines(std::io::Cursor::new(oversized), sender).unwrap();

    assert_eq!(
        receiver.try_recv().unwrap(),
        crate::OVERSIZED_INPUT_SENTINEL
    );
    assert!(receiver.try_recv().is_err());
}

#[tokio::test]
async fn oversized_protocol_line_emits_bounded_invalid_request() {
    use tokio::io::{duplex, AsyncReadExt};

    let root = fresh_test_root("oversized-protocol-frame");
    let adapter = test_adapter(&root);
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    crate::forward_input_lines(
        std::io::Cursor::new(format!("{}\n", "x".repeat(crate::MAX_INPUT_LINE_BYTES + 1))),
        sender,
    )
    .unwrap();
    let (writer_side, mut reader_side) = duplex(4096);
    crate::runtime::run_bridge(adapter, receiver, writer_side)
        .await
        .unwrap();

    let mut output = String::new();
    reader_side.read_to_string(&mut output).await.unwrap();
    let response: Value = serde_json::from_str(output.trim()).unwrap();
    assert_eq!(response["error"]["code"], -32600);
    assert_eq!(
        response["error"]["message"],
        "Request exceeds maximum line length"
    );
    assert!(output.len() < 256);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn prompt_text_limit_rejects_oversized_input() {
    let root = fresh_test_root("oversized-prompt");
    let mut adapter = test_adapter(&root);
    let cwd = root.to_string_lossy().to_string();
    let session_id = open_test_session(&mut adapter, &root);
    let error = adapter
        .prepare_prompt(
            json!(1),
            &json!({
                "sessionId": session_id,
                "prompt": [{
                    "type": "text",
                    "text": "x".repeat(crate::adapter::MAX_PROMPT_TEXT_BYTES + 1)
                }],
                "cwd": cwd,
            }),
        )
        .unwrap_err();

    let error_value = error.error.unwrap();
    assert_eq!(error_value["code"], -32602);
    assert_eq!(error_value["message"], "prompt text exceeds maximum size");
    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn limited_child_stdout_caps_buffer_and_drains_reader() {
    let (mut writer, reader) = tokio::io::duplex(crate::runtime::MAX_CHILD_STDOUT_BYTES + 1);
    let payload = vec![b'x'; crate::runtime::MAX_CHILD_STDOUT_BYTES + 1];
    let writer_task = tokio::spawn(async move {
        tokio::io::AsyncWriteExt::write_all(&mut writer, &payload)
            .await
            .unwrap();
        tokio::io::AsyncWriteExt::shutdown(&mut writer)
            .await
            .unwrap();
    });

    let output = crate::runtime::read_limited_child_output(reader).await;
    writer_task.await.unwrap();
    assert_eq!(output.bytes.len(), crate::runtime::MAX_CHILD_STDOUT_BYTES);
    assert!(output.truncated);
}

#[test]
fn model_output_parser_rejects_oversized_output() {
    let output = "x".repeat(crate::adapter::MAX_MODEL_OUTPUT_BYTES + 1);
    assert!(crate::adapter::parse_available_models_bounded(output.as_bytes()).is_empty());
}

#[test]
fn model_output_parser_ignores_cli_progress_output() {
    let output = b"Fetching available models...\ngemini-fast\tFast Model\n";
    assert_eq!(
        crate::adapter::parse_available_models_bounded(output),
        vec!["gemini-fast\tFast Model"]
    );
}

#[test]
fn extra_args_parser_rejects_oversized_environment_input() {
    let output = "--flag ".repeat(crate::adapter::MAX_EXTRA_ARGS_BYTES / 7 + 1);
    assert!(crate::adapter::parse_extra_args_bounded(&output).is_none());
}

#[test]
fn extra_args_parser_supports_quotes_and_escaped_whitespace() {
    let args = crate::adapter::parse_extra_args_bounded(
        r#"--label "hello world" --path 'C:\Program Files\agy' escaped\ value """#,
    )
    .expect("quoted extra args should parse");
    let args: Vec<_> = args
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        args,
        vec![
            "--label",
            "hello world",
            "--path",
            "C:\\Program Files\\agy",
            "escaped value",
            "",
        ]
    );
}

#[test]
fn extra_args_parser_rejects_unterminated_quotes_and_escapes() {
    assert!(crate::adapter::parse_extra_args_bounded("--flag \"unterminated").is_none());
    assert!(crate::adapter::parse_extra_args_bounded("--flag trailing\\").is_none());
}

#[test]
fn invocation_log_scan_stays_bounded_and_keeps_prefix_marker() {
    let root = fresh_test_root("bounded-run-log");
    let path = root.join("run.log");
    let conversation_id = "00000000-0000-4000-8000-000000000031";
    let mut contents = format!("Created conversation {conversation_id}\n").into_bytes();
    contents.extend(std::iter::repeat_n(
        b'x',
        crate::db::MAX_INVOCATION_LOG_BYTES + 1,
    ));
    fs::write(&path, contents).unwrap();

    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&path).as_deref(),
        Some(conversation_id)
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn timeout_log_scan_keeps_bounded_tail_marker() {
    let root = fresh_test_root("bounded-timeout-log");
    let path = root.join("run.log");
    let mut contents = vec![b'x'; crate::db::MAX_INVOCATION_LOG_BYTES + 1];
    contents.extend_from_slice(b"Print mode: timed out\n");
    fs::write(&path, contents).unwrap();

    assert!(crate::db::agy_run_timed_out(&path));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn db_row_reader_bounds_rows_and_payloads() {
    let root = fresh_test_root("bounded-db-rows");
    let conversations_dir = root.join("conversations");
    fs::create_dir_all(&conversations_dir).unwrap();
    let conversation_id = "00000000-0000-4000-8000-000000000030";
    let db_path = conversations_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER NOT NULL, step_payload BLOB)",
    )
    .unwrap();
    for idx in 0..(crate::db::MAX_DB_ROWS_PER_READ as i64 + 3) {
        let payload = if idx == 0 {
            vec![b'x'; crate::db::MAX_STEP_PAYLOAD_BYTES + 1]
        } else {
            vec![idx as u8]
        };
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![idx, payload],
        )
        .unwrap();
    }
    drop(conn);

    let rows = crate::db::read_rows_from_db(&conversations_dir, conversation_id, -1).unwrap();
    assert_eq!(rows.len(), crate::db::MAX_DB_ROWS_PER_READ);
    assert!(rows[0].2.is_empty(), "oversized payload must be redacted");
    assert!(rows
        .iter()
        .all(|(_, _, payload)| { payload.len() <= crate::db::MAX_STEP_PAYLOAD_BYTES }));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn oversized_persisted_state_file_is_rejected() {
    let root = fresh_test_root("oversized-state-file");
    let adapter = test_adapter(&root);
    fs::create_dir_all(adapter.state_file.parent().unwrap()).unwrap();
    let oversized_cwd = "x".repeat(1024 * 1024);
    let state = json!({
        "sessions": {
            "session": {
                "conversation_id": null,
                "last_step_idx": -1,
                "model_id": null,
                "cwd": oversized_cwd,
            }
        }
    });
    fs::write(&adapter.state_file, serde_json::to_vec(&state).unwrap()).unwrap();

    let error = adapter.load_store().unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn persisted_state_session_count_is_bounded() {
    let root = fresh_test_root("state-session-count");
    let adapter = test_adapter(&root);
    fs::create_dir_all(adapter.state_file.parent().unwrap()).unwrap();
    let mut state = SessionStore::default();
    for index in 0..1025 {
        state.sessions.insert(
            format!("session-{index}"),
            StoredSession {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
                cwd: Some(root.to_string_lossy().to_string()),
            },
        );
    }
    fs::write(&adapter.state_file, serde_json::to_vec(&state).unwrap()).unwrap();

    let error = adapter.load_store().unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn persisted_conversation_id_rejects_path_injection() {
    let root = fresh_test_root("invalid-persisted-conversation");
    let adapter = test_adapter(&root);
    fs::create_dir_all(adapter.state_file.parent().unwrap()).unwrap();
    fs::write(
        &adapter.state_file,
        serde_json::to_vec(&json!({
            "sessions": {
                "session": {
                    "conversation_id": "..\\outside.db",
                    "last_step_idx": -1,
                    "model_id": null,
                    "cwd": root.to_string_lossy(),
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let error = adapter.load_store().unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn resident_eviction_is_lru_and_prefers_inactive_sessions() {
    let root = fresh_test_root("resident-lru");
    let mut adapter = test_adapter(&root);
    let mut session_ids = Vec::new();
    for index in 0..crate::adapter::MAX_RESIDENT_SESSIONS {
        let response = adapter.handle_session_new(json!(index), &session_setup_params(&root));
        session_ids.push(
            response.result.as_ref().unwrap()["sessionId"]
                .as_str()
                .unwrap()
                .to_string(),
        );
    }

    let first = session_ids[0].clone();
    let second = session_ids[1].clone();
    let third = session_ids[2].clone();
    let response =
        adapter.handle_session_resume(json!(100), &session_lifecycle_params(&first, &root));
    assert!(response.error.is_none());
    adapter.mark_prompt_active(&second);

    let inserted = adapter.handle_session_new(json!(101), &session_setup_params(&root));
    assert!(inserted.error.is_none());
    assert!(adapter.sessions.contains_key(&first));
    assert!(adapter.sessions.contains_key(&second));
    assert!(!adapter.sessions.contains_key(&third));

    let fourth = session_ids[3].clone();
    let inserted = adapter.handle_session_new(json!(102), &session_setup_params(&root));
    assert!(inserted.error.is_none());
    assert!(adapter.sessions.contains_key(&second));
    assert!(!adapter.sessions.contains_key(&fourth));
    adapter.mark_prompt_complete(&second);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn db_row_reader_caps_total_payload_bytes_per_page() {
    let root = fresh_test_root("bounded-db-bytes");
    let conversations_dir = root.join("conversations");
    fs::create_dir_all(&conversations_dir).unwrap();
    let conversation_id = "00000000-0000-4000-8000-000000000032";
    let db_path = conversations_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER NOT NULL, step_payload BLOB)",
    )
    .unwrap();
    for idx in 0..8i64 {
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![idx, vec![b'x'; 900_000]],
        )
        .unwrap();
    }
    drop(conn);

    let rows = crate::db::read_rows_from_db(&conversations_dir, conversation_id, -1).unwrap();
    let total_payload_bytes: usize = rows.iter().map(|(_, _, payload)| payload.len()).sum();
    assert!(
        total_payload_bytes <= 4 * 1024 * 1024,
        "page payload bytes were not bounded: {total_payload_bytes}"
    );
    assert!(rows.len() < 8, "byte budget should stop before all rows");
    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn session_load_replays_beyond_one_page_and_keeps_tail_cursor() {
    let root = fresh_test_root("replay-pagination");
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();
    let conversation_id = "00000000-0000-4000-8000-000000000033";
    let db_path = conv_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();
    for idx in 1..=260i64 {
        let (step_type, payload) = if idx % 2 == 0 {
            (15, make_assistant_payload(&format!("assistant-{idx}")))
        } else {
            (14, make_user_payload(&format!("user-{idx}")))
        };
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, ?2, ?3)",
            rusqlite::params![idx, step_type, payload],
        )
        .unwrap();
    }
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;
    adapter
        .persist_session(
            "sess-replay-tail",
            &crate::types::Session {
                conversation_id: Some(conversation_id.to_string()),
                last_step_idx: -1,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .unwrap();

    let output = adapter.handle_session_load(
        json!(1),
        &json!({
            "sessionId": "sess-replay-tail",
            "cwd": root.to_string_lossy(),
            "mcpServers": []
        }),
    );
    assert!(output.iter().any(|line| line.contains("assistant-260")));
    assert_eq!(adapter.sessions["sess-replay-tail"].last_step_idx, 260);
    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn oversized_replay_history_fails_closed_without_installing_session() {
    let root = fresh_test_root("replay-budget");
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();
    let conversation_id = "00000000-0000-4000-8000-000000000034";
    let db_path = conv_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();
    let large_text = "x".repeat(900_000);
    for idx in 1..=10i64 {
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![idx, make_assistant_payload(&large_text)],
        )
        .unwrap();
    }
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;
    adapter
        .persist_session(
            "sess-replay-budget",
            &crate::types::Session {
                conversation_id: Some(conversation_id.to_string()),
                last_step_idx: -1,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .unwrap();

    let output = adapter.handle_session_load(
        json!(1),
        &json!({
            "sessionId": "sess-replay-budget",
            "cwd": root.to_string_lossy(),
            "mcpServers": []
        }),
    );
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert_eq!(response["error"]["code"], -32000);
    assert_eq!(
        response["error"]["message"],
        "conversation history exceeds replay budget"
    );
    assert!(!adapter.sessions.contains_key("sess-replay-budget"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn active_prompts_allow_different_sessions_and_reject_duplicates() {
    // Break caught: replacing an in-flight registration for the same session.
    let active = crate::runtime::ActivePrompts::default();
    let first = active.register("session-a").unwrap();
    assert!(active.register("session-a").is_err());
    assert!(active.register("session-b").is_ok());
    active.complete("session-a", &first);
    assert!(active.register("session-a").is_ok());
}

#[test]
fn cancel_targets_only_the_registered_session() {
    // Break caught: a session/cancel request cancelling unrelated prompts.
    let active = crate::runtime::ActivePrompts::default();
    let first = active.register("session-a").unwrap();
    let second = active.register("session-b").unwrap();
    assert!(active.cancel("session-a"));
    assert!(first.is_cancelled());
    assert!(!second.is_cancelled());
}

#[test]
fn active_prompts_complete_uses_registration_identity() {
    // Break caught: stale task cleanup removing a newer registration.
    let active = crate::runtime::ActivePrompts::default();
    let stale = active.register("session-a").unwrap();
    active.complete("session-a", &stale);
    let current = active.register("session-a").unwrap();

    active.complete("session-a", &stale);

    assert!(active.register("session-a").is_err());
    active.complete("session-a", &current);
}

#[test]
fn active_prompts_cancel_all_marks_snapshot() {
    // Break caught: fatal bridge shutdown leaving one or more child prompts running.
    let active = crate::runtime::ActivePrompts::default();
    let first = active.register("session-a").unwrap();
    let second = active.register("session-b").unwrap();

    active.cancel_all();

    assert!(first.is_cancelled());
    assert!(second.is_cancelled());
}

#[test]
fn active_prompts_completion_guard_cleans_registry_on_drop() {
    // Break caught: a panic/abort path leaking the active registration and completion count.
    let active = crate::runtime::ActivePrompts::default();
    let registration = active.register("session-a").unwrap();
    let (done, mut completions) = tokio::sync::mpsc::unbounded_channel();

    drop(crate::runtime::PromptCompletion::new(
        active.clone(),
        "session-a".to_string(),
        registration,
        done,
    ));

    assert!(active.register("session-a").is_ok());
    assert_eq!(completions.try_recv(), Ok(()));
}

#[test]
fn adaptive_poll_delay_backs_off_when_idle_and_resets_on_delta() {
    assert_eq!(
        crate::runtime::next_poll_delay(Duration::from_millis(100), false),
        Duration::from_millis(200)
    );
    assert_eq!(
        crate::runtime::next_poll_delay(Duration::from_millis(200), false),
        Duration::from_millis(400)
    );
    assert_eq!(
        crate::runtime::next_poll_delay(Duration::from_millis(400), false),
        Duration::from_millis(500)
    );
    assert_eq!(
        crate::runtime::next_poll_delay(Duration::from_millis(500), false),
        Duration::from_millis(500)
    );
    assert_eq!(
        crate::runtime::next_poll_delay(Duration::from_millis(500), true),
        Duration::from_millis(100)
    );
}

struct BrokenPipeWriter;

impl AsyncWrite for BrokenPipeWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Poll::Ready(Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "test writer closed",
        )))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn bridge_returns_writer_failure() {
    // Break caught: stdout failure being logged and swallowed with a zero exit status.
    let root = fresh_test_root("writer-failure");
    let adapter = test_adapter(&root);
    let (input, receiver) = tokio::sync::mpsc::unbounded_channel();
    input
        .send(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#.to_string())
        .unwrap();
    drop(input);

    let error = crate::runtime::run_bridge(adapter, receiver, BrokenPipeWriter)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
}

#[tokio::test]
async fn blank_stdin_line_is_forwarded_to_the_bridge_as_a_parse_error() {
    // Break caught: stdin ingress silently dropping blank or whitespace-only protocol frames.
    use tokio::io::{duplex, AsyncReadExt};

    let root = fresh_test_root("blank-stdin-frame");
    let adapter = test_adapter(&root);
    let (input, receiver) = tokio::sync::mpsc::unbounded_channel();
    crate::forward_input_lines(std::io::Cursor::new("\n  \t\n"), input).unwrap();

    let (writer_side, mut reader_side) = duplex(4096);
    crate::runtime::run_bridge(adapter, receiver, writer_side)
        .await
        .unwrap();

    let mut output = String::new();
    reader_side.read_to_string(&mut output).await.unwrap();
    let responses: Vec<Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses.len(), 2);
    assert!(responses
        .iter()
        .all(|response| response["error"]["code"] == -32700));
}

#[tokio::test]
async fn writer_failure_cancels_active_prompt_and_waits_for_completion() {
    // Break caught: returning a writer error before cancelling and draining active prompts.
    let active = crate::runtime::ActivePrompts::default();
    let registration = active.register("session-a").unwrap();
    let cancelled = registration.cancellation_flag();
    let (done, mut completions) = tokio::sync::mpsc::unbounded_channel();
    let completion = crate::runtime::PromptCompletion::new(
        active.clone(),
        "session-a".to_string(),
        registration,
        done,
    );
    let prompt = tokio::spawn(async move {
        while !cancelled.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        drop(completion);
        true
    });
    let writer_result: Result<std::io::Result<()>, tokio::task::JoinError> = Ok(Err(
        std::io::Error::new(std::io::ErrorKind::BrokenPipe, "original writer failure"),
    ));

    let error = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        crate::runtime::handle_writer_exit(&active, 1, &mut completions, writer_result),
    )
    .await
    .expect("writer cleanup timed out")
    .unwrap_err();

    assert!(prompt.await.unwrap());
    assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
    assert_eq!(error.to_string(), "original writer failure");
    assert!(active.register("session-a").is_ok());
}

#[test]
fn poller_completion_wait_keeps_single_worker_runtime_schedulable() {
    // Break caught: synchronously joining a backpressured poller on the only Tokio worker.
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel(1);
    let runner = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let completed = runtime.block_on(async {
            let (output, mut receiver) = tokio::sync::mpsc::channel(1);
            let (saturated_tx, saturated_rx) = tokio::sync::oneshot::channel();
            let poller = crate::runtime::spawn_poller_thread(move || {
                output.blocking_send("first").unwrap();
                let _ = saturated_tx.send(());
                output.blocking_send("second").unwrap();
            });

            saturated_rx.await.unwrap();
            let waiter = tokio::spawn(poller.wait());
            tokio::time::sleep(Duration::from_millis(25)).await;
            let first = receiver.recv().await;
            let second = receiver.recv().await;
            tokio::time::timeout(Duration::from_millis(500), waiter)
                .await
                .expect("poller completion was not delivered")
                .unwrap();

            let (closed_output, closed_receiver) = tokio::sync::mpsc::channel(1);
            let (filled_tx, filled_rx) = tokio::sync::oneshot::channel();
            let (unblocked_tx, unblocked_rx) = tokio::sync::oneshot::channel();
            let closed_poller = crate::runtime::spawn_poller_thread(move || {
                closed_output.blocking_send("first").unwrap();
                let _ = filled_tx.send(());
                let unblocked = closed_output.blocking_send("second").is_err();
                let _ = unblocked_tx.send(unblocked);
            });
            filled_rx.await.unwrap();
            drop(closed_receiver);
            let unblocked = tokio::time::timeout(Duration::from_millis(500), unblocked_rx)
                .await
                .expect("closed receiver did not unblock poller")
                .unwrap();
            tokio::time::timeout(Duration::from_millis(500), closed_poller.wait())
                .await
                .expect("closed-channel poller completion was not delivered");

            first == Some("first") && second == Some("second") && unblocked
        });
        let _ = result_tx.send(completed);
    });

    let completed = result_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("single-worker runtime was blocked while awaiting poller completion");
    runner.join().unwrap();
    assert!(completed);
}

#[test]
fn apply_prompt_outcome_keeps_partial_failed_turn_progress() {
    // Break caught: returning an execution error before committing a discovered binding/cursor.
    let root = fresh_test_root("partial-outcome");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    let outcome = crate::runtime::PromptOutcome {
        session_id: session_id.clone(),
        conversation_id: Some(TEST_CONVERSATION_AFTER_ERROR.to_string()),
        last_step_idx: 9,
        response: JsonRpcResponse::error(json!(8), -32000, "terminal check failed"),
        run_log_path: root.join("partial-outcome.log"),
        remove_run_log_on_commit: false,
    };

    adapter
        .apply_prompt_outcome(&outcome)
        .expect("partial diagnostic progress should persist");

    let session = &adapter.sessions[&session_id];
    assert_eq!(
        session.conversation_id.as_deref(),
        Some(TEST_CONVERSATION_AFTER_ERROR)
    );
    assert_eq!(session.last_step_idx, 9);
}

#[test]
fn apply_prompt_outcome_never_recreates_a_missing_session() {
    // Break caught: a late prompt completion resurrecting a removed session.
    let root = fresh_test_root("missing-outcome");
    let mut adapter = test_adapter(&root);
    let outcome = crate::runtime::PromptOutcome {
        session_id: "removed-session".to_string(),
        conversation_id: Some(TEST_CONVERSATION_LATE.to_string()),
        last_step_idx: 3,
        response: JsonRpcResponse::success(json!(9), json!({"stopReason": "end_turn"})),
        run_log_path: root.join("missing-outcome.log"),
        remove_run_log_on_commit: false,
    };

    assert!(adapter.apply_prompt_outcome(&outcome).is_err());

    assert!(!adapter.sessions.contains_key("removed-session"));
    assert!(adapter
        .restore_session("removed-session")
        .expect("missing-session lookup should read a valid empty store")
        .is_none());
}

#[test]
fn outcome_persistence_failure_maps_terminal_error_and_retains_run_log() {
    // Break caught: sending end_turn and deleting diagnostics after state commit failed.
    let root = fresh_test_root("outcome-persistence-failure");
    let blocked_parent = root.join("blocked-state-parent");
    fs::write(&blocked_parent, b"not a directory").unwrap();
    let run_log_path = root.join("prompt.log");
    fs::write(&run_log_path, b"diagnostic run log").unwrap();
    let mut adapter = test_adapter(&root);
    adapter.state_file = blocked_parent.join("sessions.json");
    adapter.sessions.insert(
        "active-session".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: Some("preserved-model".to_string()),
            cwd: root.clone(),
        },
    );
    let before = adapter.sessions["active-session"].clone();
    let outcome = crate::runtime::PromptOutcome {
        session_id: "active-session".to_string(),
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 12,
        response: JsonRpcResponse::success(json!(10), json!({"stopReason": "end_turn"})),
        run_log_path: run_log_path.clone(),
        remove_run_log_on_commit: true,
    };

    let response = crate::runtime::finalize_prompt_outcome(&mut adapter, outcome);

    assert_persistence_error(&response);
    assert_eq!(adapter.sessions["active-session"], before);
    assert!(run_log_path.exists());
}

#[test]
fn late_outcome_updates_evicted_persisted_session_without_reinserting_it() {
    // Break caught: dropping an active session's late binding after arbitrary cache eviction.
    let root = fresh_test_root("evicted-outcome");
    let mut adapter = test_adapter(&root);
    let mut original_ids = Vec::new();
    let mut store = SessionStore::default();
    for index in 0..64 {
        let session_id = format!("persisted-session-{index}");
        adapter.sessions.insert(
            session_id.clone(),
            crate::types::Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: Some("preserved-model".to_string()),
                cwd: root.clone(),
            },
        );
        store.sessions.insert(
            session_id.clone(),
            StoredSession {
                conversation_id: None,
                last_step_idx: -1,
                model_id: Some("preserved-model".to_string()),
                cwd: Some(root.to_string_lossy().to_string()),
            },
        );
        original_ids.push(session_id);
    }
    fs::create_dir_all(adapter.state_file.parent().unwrap()).unwrap();
    fs::write(
        &adapter.state_file,
        serde_json::to_vec_pretty(&store).unwrap(),
    )
    .unwrap();
    assert_eq!(adapter.sessions.len(), 64);

    let inserted_id = open_test_session(&mut adapter, &root);
    assert!(adapter.sessions.contains_key(&inserted_id));
    let evicted: Vec<_> = original_ids
        .iter()
        .filter(|session_id| !adapter.sessions.contains_key(*session_id))
        .cloned()
        .collect();
    assert_eq!(evicted.len(), 1);
    let victim = &evicted[0];
    let before = adapter
        .restore_session(victim)
        .expect("evicted victim state should be readable")
        .expect("evicted victim should remain persisted");
    assert_eq!(before.cwd.as_deref(), Some(root.to_string_lossy().as_ref()));
    assert_eq!(before.model_id.as_deref(), Some("preserved-model"));
    let run_log_path = root.join("evicted-outcome.log");
    fs::write(&run_log_path, b"late outcome diagnostic").unwrap();

    let outcome = crate::runtime::PromptOutcome {
        session_id: victim.clone(),
        conversation_id: Some(TEST_CONVERSATION_LATE.to_string()),
        last_step_idx: 42,
        response: JsonRpcResponse::success(json!(12), json!({"stopReason": "end_turn"})),
        run_log_path: run_log_path.clone(),
        remove_run_log_on_commit: true,
    };
    let response = crate::runtime::finalize_prompt_outcome(&mut adapter, outcome);

    assert!(response.error.is_none());
    let after = adapter
        .restore_session(victim)
        .expect("updated victim state should be readable")
        .expect("updated victim should remain persisted");
    assert_eq!(
        after.conversation_id.as_deref(),
        Some(TEST_CONVERSATION_LATE)
    );
    assert_eq!(after.last_step_idx, 42);
    assert_eq!(after.cwd, before.cwd);
    assert_eq!(after.model_id, before.model_id);
    assert!(!adapter.sessions.contains_key(victim));
    assert!(!run_log_path.exists());
}

#[test]
fn resident_conversation_conflict_fails_without_mutation_and_retains_run_log() {
    // Break caught: combining resident conversation A with outcome B's cursor.
    let root = fresh_test_root("resident-conversation-conflict");
    let mut adapter = test_adapter(&root);
    let session_id = "resident-conflict";
    let session = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_A.to_string()),
        last_step_idx: 11,
        model_id: Some("preserved-model".to_string()),
        cwd: root.clone(),
    };
    adapter
        .persist_session(session_id, &session)
        .expect("resident conflict fixture should persist");
    adapter
        .sessions
        .insert(session_id.to_string(), session.clone());
    let stored_before = adapter
        .restore_session(session_id)
        .expect("resident conflict store should be readable")
        .expect("resident conflict row should exist");
    let run_log_path = root.join("resident-conflict.log");
    fs::write(&run_log_path, b"resident conflict diagnostic").unwrap();
    let outcome = crate::runtime::PromptOutcome {
        session_id: session_id.to_string(),
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 99,
        response: JsonRpcResponse::success(json!(13), json!({"stopReason": "end_turn"})),
        run_log_path: run_log_path.clone(),
        remove_run_log_on_commit: true,
    };

    let response = crate::runtime::finalize_prompt_outcome(&mut adapter, outcome);

    assert_persistence_error(&response);
    assert_eq!(adapter.sessions[session_id], session);
    assert_eq!(
        adapter
            .restore_session(session_id)
            .expect("resident conflict store should remain readable")
            .expect("resident conflict row should remain present"),
        stored_before
    );
    assert!(run_log_path.exists());
}

#[test]
fn evicted_conversation_conflict_fails_without_mutation_and_retains_run_log() {
    // Break caught: replacing persisted conversation A with evicted outcome B.
    let root = fresh_test_root("evicted-conversation-conflict");
    let mut adapter = test_adapter(&root);
    let session_id = "evicted-conflict";
    let session = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_A.to_string()),
        last_step_idx: 21,
        model_id: Some("preserved-model".to_string()),
        cwd: root.clone(),
    };
    adapter
        .persist_session(session_id, &session)
        .expect("evicted conflict fixture should persist");
    let stored_before = adapter
        .restore_session(session_id)
        .expect("evicted conflict store should be readable")
        .expect("evicted conflict row should exist");
    let run_log_path = root.join("evicted-conflict.log");
    fs::write(&run_log_path, b"evicted conflict diagnostic").unwrap();
    let outcome = crate::runtime::PromptOutcome {
        session_id: session_id.to_string(),
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 100,
        response: JsonRpcResponse::success(json!(14), json!({"stopReason": "end_turn"})),
        run_log_path: run_log_path.clone(),
        remove_run_log_on_commit: true,
    };

    let response = crate::runtime::finalize_prompt_outcome(&mut adapter, outcome);

    assert_persistence_error(&response);
    assert!(!adapter.sessions.contains_key(session_id));
    assert_eq!(
        adapter
            .restore_session(session_id)
            .expect("evicted conflict store should remain readable")
            .expect("evicted conflict row should remain present"),
        stored_before
    );
    assert!(run_log_path.exists());
}

#[test]
fn resident_matching_outcome_rejects_conflicting_persisted_binding() {
    // Break caught: a resident B/outcome B pair overwriting stale persisted conversation A.
    let root = fresh_test_root("persisted-conversation-conflict");
    let mut adapter = test_adapter(&root);
    let session_id = "persisted-conflict";
    let persisted = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_A.to_string()),
        last_step_idx: 30,
        model_id: Some("persisted-model".to_string()),
        cwd: root.clone(),
    };
    adapter
        .persist_session(session_id, &persisted)
        .expect("persisted conflict fixture should persist");
    let resident = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 31,
        model_id: Some("resident-model".to_string()),
        cwd: root.clone(),
    };
    adapter
        .sessions
        .insert(session_id.to_string(), resident.clone());
    let stored_before = adapter
        .restore_session(session_id)
        .expect("persisted conflict store should be readable")
        .expect("persisted conflict row should exist");
    let run_log_path = root.join("persisted-conflict.log");
    fs::write(&run_log_path, b"persisted conflict diagnostic").unwrap();
    let outcome = crate::runtime::PromptOutcome {
        session_id: session_id.to_string(),
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 32,
        response: JsonRpcResponse::success(json!(17), json!({"stopReason": "end_turn"})),
        run_log_path: run_log_path.clone(),
        remove_run_log_on_commit: true,
    };

    let response = crate::runtime::finalize_prompt_outcome(&mut adapter, outcome);

    assert_persistence_error(&response);
    assert_eq!(adapter.sessions[session_id], resident);
    assert_eq!(
        adapter
            .restore_session(session_id)
            .expect("persisted conflict store should remain readable")
            .expect("persisted conflict row should remain present"),
        stored_before
    );
    assert!(run_log_path.exists());
}

#[test]
fn prompt_outcome_cursor_is_monotonic_for_resident_and_evicted_sessions() {
    // Break caught: regressing a same-conversation cursor or failing to advance it.
    let root = fresh_test_root("monotonic-outcomes");
    let mut adapter = test_adapter(&root);
    let resident_id = "resident-monotonic";
    let resident = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 10,
        model_id: None,
        cwd: root.clone(),
    };
    adapter
        .persist_session(resident_id, &resident)
        .expect("resident monotonic fixture should persist");
    adapter.sessions.insert(resident_id.to_string(), resident);
    let resident_outcome = crate::runtime::PromptOutcome {
        session_id: resident_id.to_string(),
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 6,
        response: JsonRpcResponse::success(json!(15), json!({"stopReason": "end_turn"})),
        run_log_path: root.join("resident-monotonic.log"),
        remove_run_log_on_commit: false,
    };

    let resident_response = crate::runtime::finalize_prompt_outcome(&mut adapter, resident_outcome);

    assert!(resident_response.error.is_none());
    assert_eq!(adapter.sessions[resident_id].last_step_idx, 10);
    assert_eq!(
        adapter
            .restore_session(resident_id)
            .expect("resident monotonic store should be readable")
            .expect("resident monotonic row should exist")
            .last_step_idx,
        10
    );

    let evicted_id = "evicted-monotonic";
    let evicted = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 20,
        model_id: None,
        cwd: root.clone(),
    };
    adapter
        .persist_session(evicted_id, &evicted)
        .expect("evicted monotonic fixture should persist");
    let evicted_outcome = crate::runtime::PromptOutcome {
        session_id: evicted_id.to_string(),
        conversation_id: Some(TEST_CONVERSATION_B.to_string()),
        last_step_idx: 25,
        response: JsonRpcResponse::success(json!(16), json!({"stopReason": "end_turn"})),
        run_log_path: root.join("evicted-monotonic.log"),
        remove_run_log_on_commit: false,
    };

    let evicted_response = crate::runtime::finalize_prompt_outcome(&mut adapter, evicted_outcome);

    assert!(evicted_response.error.is_none());
    assert!(!adapter.sessions.contains_key(evicted_id));
    assert_eq!(
        adapter
            .restore_session(evicted_id)
            .expect("evicted monotonic store should be readable")
            .expect("evicted monotonic row should exist")
            .last_step_idx,
        25
    );
}

#[test]
fn outcome_without_conversation_preserves_resident_and_evicted_bindings() {
    // Retained contract: an unbound outcome must not alter an existing binding or cursor.
    let root = fresh_test_root("outcome-without-conversation");
    let mut adapter = test_adapter(&root);
    let resident_id = "resident-without-outcome-binding";
    let resident = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_A.to_string()),
        last_step_idx: 40,
        model_id: None,
        cwd: root.clone(),
    };
    adapter
        .persist_session(resident_id, &resident)
        .expect("resident no-binding fixture should persist");
    adapter
        .sessions
        .insert(resident_id.to_string(), resident.clone());
    let resident_outcome = crate::runtime::PromptOutcome {
        session_id: resident_id.to_string(),
        conversation_id: None,
        last_step_idx: 400,
        response: JsonRpcResponse::error(json!(18), -32000, "process failed"),
        run_log_path: root.join("resident-without-binding.log"),
        remove_run_log_on_commit: false,
    };

    adapter
        .apply_prompt_outcome(&resident_outcome)
        .expect("resident outcome without binding should commit");

    assert_eq!(adapter.sessions[resident_id], resident);
    assert_eq!(
        adapter
            .restore_session(resident_id)
            .expect("resident no-binding state should be readable")
            .expect("resident no-binding row should exist")
            .last_step_idx,
        40
    );

    let evicted_id = "evicted-without-outcome-binding";
    let evicted = crate::types::Session {
        conversation_id: Some(TEST_CONVERSATION_A.to_string()),
        last_step_idx: 50,
        model_id: None,
        cwd: root.clone(),
    };
    adapter
        .persist_session(evicted_id, &evicted)
        .expect("evicted no-binding fixture should persist");
    let evicted_outcome = crate::runtime::PromptOutcome {
        session_id: evicted_id.to_string(),
        conversation_id: None,
        last_step_idx: 500,
        response: JsonRpcResponse::error(json!(19), -32000, "process failed"),
        run_log_path: root.join("evicted-without-binding.log"),
        remove_run_log_on_commit: false,
    };

    adapter
        .apply_prompt_outcome(&evicted_outcome)
        .expect("evicted outcome without binding should commit");

    assert!(!adapter.sessions.contains_key(evicted_id));
    let stored = adapter
        .restore_session(evicted_id)
        .expect("evicted no-binding state should be readable")
        .expect("evicted no-binding row should exist");
    assert_eq!(stored.conversation_id.as_deref(), Some(TEST_CONVERSATION_A));
    assert_eq!(stored.last_step_idx, 50);
}

fn fresh_test_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("agy-acp-{label}-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn adapter_model_discovery_times_out_without_blocking_startup() {
    let root = fresh_test_root("models-timeout");
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    write_slow_models_stub(&bin_dir);

    let probe = launch_model_discovery_probe("tests::model_discovery_timeout_probe", &bin_dir);
    assert!(
        probe.success(),
        "model discovery timeout probe failed: {probe}"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn adapter_model_discovery_keeps_fast_model_output() {
    let root = fresh_test_root("models-success");
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    write_models_stub(
        &bin_dir,
        r#"println!("gemini-fast\tFast Model"); println!("gemini-plain");"#,
    );

    let probe = launch_model_discovery_probe("tests::model_discovery_success_probe", &bin_dir);
    assert!(
        probe.success(),
        "model discovery success probe failed: {probe}"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn adapter_model_discovery_rejects_oversized_model_output() {
    let root = fresh_test_root("models-oversized");
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    write_models_stub(
        &bin_dir,
        &format!(
            "print!(\"{{}}\", \"x\".repeat({}));",
            crate::adapter::MAX_MODEL_OUTPUT_BYTES + 1
        ),
    );

    let probe = launch_model_discovery_probe("tests::model_discovery_oversized_probe", &bin_dir);
    assert!(
        probe.success(),
        "oversized model discovery probe failed: {probe}"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn model_discovery_timeout_probe() {
    if std::env::var("AGY_ACP_MODEL_DISCOVERY_PROBE").as_deref() != Ok("1") {
        return;
    }

    let bin_dir = PathBuf::from(std::env::var_os("AGY_ACP_MODEL_DISCOVERY_BIN").unwrap());
    let parent_path = std::env::var_os("AGY_ACP_MODEL_DISCOVERY_PARENT_PATH").unwrap();
    let path = prepend_to_path(&bin_dir, &parent_path);
    std::env::set_var("PATH", path);

    let started = std::time::Instant::now();
    let adapter = Adapter::new();
    assert!(
        adapter.available_models.is_empty(),
        "timed-out discovery should preserve the empty-model fallback"
    );
    assert!(
        started.elapsed() < Duration::from_secs(6),
        "startup model discovery exceeded the timeout budget: {:?}",
        started.elapsed()
    );
}

#[test]
fn model_discovery_success_probe() {
    if std::env::var("AGY_ACP_MODEL_DISCOVERY_PROBE").as_deref() != Ok("1") {
        return;
    }

    let bin_dir = PathBuf::from(std::env::var_os("AGY_ACP_MODEL_DISCOVERY_BIN").unwrap());
    let parent_path = std::env::var_os("AGY_ACP_MODEL_DISCOVERY_PARENT_PATH").unwrap();
    let path = prepend_to_path(&bin_dir, &parent_path);
    std::env::set_var("PATH", path);

    let adapter = Adapter::new();
    assert_eq!(
        adapter.available_models,
        vec![
            "gemini-fast\tFast Model".to_string(),
            "gemini-plain".to_string()
        ]
    );
}

#[test]
fn model_discovery_oversized_probe() {
    if std::env::var("AGY_ACP_MODEL_DISCOVERY_PROBE").as_deref() != Ok("1") {
        return;
    }

    let bin_dir = PathBuf::from(std::env::var_os("AGY_ACP_MODEL_DISCOVERY_BIN").unwrap());
    let parent_path = std::env::var_os("AGY_ACP_MODEL_DISCOVERY_PARENT_PATH").unwrap();
    let path = prepend_to_path(&bin_dir, &parent_path);
    std::env::set_var("PATH", path);

    let adapter = Adapter::new();
    assert!(adapter.available_models.is_empty());
}

fn prepend_to_path(bin_dir: &std::path::Path, parent_path: &std::ffi::OsStr) -> std::ffi::OsString {
    let mut entries = vec![bin_dir.to_path_buf()];
    entries.extend(std::env::split_paths(parent_path));
    std::env::join_paths(entries).expect("test probe PATH entries should be valid")
}

fn launch_model_discovery_probe(
    test_name: &str,
    bin_dir: &std::path::Path,
) -> std::process::ExitStatus {
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let mut probe = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env("AGY_ACP_MODEL_DISCOVERY_PROBE", "1")
        .env("AGY_ACP_MODEL_DISCOVERY_BIN", bin_dir)
        .env("AGY_ACP_MODEL_DISCOVERY_PARENT_PATH", original_path)
        .env("AGY_MODEL_DISCOVERY_TIMEOUT_MS", "5000")
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(8);

    loop {
        match probe.try_wait() {
            Ok(Some(status)) => return status,
            Ok(None) if std::time::Instant::now() >= deadline => {
                let _ = probe.kill();
                let _ = probe.wait();
                panic!("model discovery probe exceeded its 8-second test deadline");
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(error) => {
                let _ = probe.kill();
                let _ = probe.wait();
                panic!("failed to poll model discovery probe: {error}");
            }
        }
    }
}

fn write_slow_models_stub(bin_dir: &std::path::Path) {
    write_models_stub(
        bin_dir,
        "std::thread::sleep(std::time::Duration::from_millis(7000));",
    );
}

fn write_models_stub(bin_dir: &std::path::Path, body: &str) {
    let source = bin_dir.join("agy-stub.rs");
    let binary = if cfg!(windows) {
        bin_dir.join("agy.exe")
    } else {
        bin_dir.join("agy")
    };
    fs::write(&source, format!("fn main() {{ {body} }}\n")).unwrap();
    let compile = std::process::Command::new("rustc")
        .args(["--edition=2021"])
        .arg(&source)
        .args(["-o"])
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "failed to build agy model-discovery stub: stdout={} stderr={}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );
}

#[cfg(windows)]
const WINDOWS_FAKE_AGY_SCRIPT: &str = r#"
$logFile = $null
$prompt = $null
$addDirs = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    if (($args[$i] -eq '--add-dir') -and (($i + 1) -lt $args.Count)) {
        $addDirs += $args[$i + 1]
    }
    if (($null -eq $logFile) -and ($args[$i] -eq '--log-file') -and (($i + 1) -lt $args.Count)) {
        $logFile = $args[$i + 1]
    }
    if (($args[$i] -eq '-p') -and (($i + 1) -lt $args.Count)) {
        $prompt = $args[$i + 1]
    }
}
Set-Content -LiteralPath $logFile -Value 'Created conversation 00000000-0000-4000-8000-000000000099' -Encoding utf8
foreach ($dir in $addDirs) {
    $config = Join-Path $dir '.agents\mcp_config.json'
    if (Test-Path -LiteralPath $config) { Copy-Item -LiteralPath $config -Destination ($logFile + '.mcp') }
}
if ($prompt -eq 'slow') { Start-Sleep -Milliseconds 3000 }
if ($prompt -eq 'tree') {
    $descendant = Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile', '-NonInteractive', '-Command', 'Start-Sleep -Seconds 30' -PassThru
    Set-Content -LiteralPath ($logFile + '.child') -Value $descendant.Id -Encoding ascii
    Start-Sleep -Seconds 30
}
if ($prompt -eq 'stderr-fail') {
    [Console]::Error.WriteLine('AGY_STDERR_SECRET_SENTINEL')
    exit 7
}
if ($prompt -eq 'fail') { exit 7 }
if ($prompt -eq 'huge') { [Console]::Out.Write('x' * 5000000); exit 0 }
[Console]::Out.WriteLine('fake assistant response')
"#;

#[cfg(not(windows))]
const UNIX_FAKE_AGY_SCRIPT: &str = r#"#!/bin/sh
log_file=''
prompt=''
add_dirs=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --add-dir)
      if [ "$#" -gt 1 ]; then add_dirs="$add_dirs
$2"; fi
      shift
      ;;
    --log-file)
      if [ -z "$log_file" ] && [ "$#" -gt 1 ]; then log_file="$2"; fi
      shift
      ;;
    -p)
      if [ "$#" -gt 1 ]; then prompt="$2"; fi
      shift
      ;;
  esac
  shift
done
printf '%s\n' 'Created conversation 00000000-0000-4000-8000-000000000099' > "$log_file"
printf '%s\n' "$add_dirs" | while IFS= read -r dir; do
  if [ -n "$dir" ] && [ -f "$dir/.agents/mcp_config.json" ]; then cp "$dir/.agents/mcp_config.json" "$log_file.mcp"; fi
done
if [ "$prompt" = 'slow' ]; then
  printf '%s\n' 'fake assistant response'
  exec sleep 3
fi
if [ "$prompt" = 'tree' ]; then
  (sleep 30) &
  echo $! > "$log_file.child"
  sleep 30
fi
if [ "$prompt" = 'stderr-fail' ]; then
  printf '%s\n' 'AGY_STDERR_SECRET_SENTINEL' >&2
  exit 7
fi
if [ "$prompt" = 'fail' ]; then exit 7; fi
if [ "$prompt" = 'huge' ]; then head -c 5000000 /dev/zero | tr '\\0' 'x'; exit 0; fi
printf '%s\n' 'fake assistant response'
"#;

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

struct ConcurrentHarness {
    root: PathBuf,
    adapter: Adapter,
    output: crate::output::OutputSender,
    receiver: tokio::sync::mpsc::Receiver<String>,
}

impl ConcurrentHarness {
    fn new(label: &str) -> Self {
        let root = fresh_test_root(label);
        let mut adapter = test_adapter(&root);
        adapter.command = fake_agy_command(&root);
        for session_id in ["session-a", "session-b"] {
            let cwd = root.join(session_id);
            fs::create_dir_all(&cwd).unwrap();
            adapter.sessions.insert(
                session_id.to_string(),
                crate::types::Session {
                    conversation_id: None,
                    last_step_idx: -1,
                    model_id: None,
                    cwd,
                },
            );
        }
        let (output, receiver) = crate::output::channel();
        Self {
            root,
            adapter,
            output,
            receiver,
        }
    }

    fn prepare(&mut self, session_id: &str, text: &str) -> crate::types::PromptExecution {
        self.adapter
            .prepare_prompt(
                json!(Uuid::new_v4().to_string()),
                &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": text}]}),
            )
            .unwrap()
    }
}

// PowerShell fake-child startup has been observed at 2.1–2.8s under parallel Windows load.
// Keep this bounded while allowing headroom above that measured range.
const FAKE_RUN_LOG_WAIT_TIMEOUT: Duration = Duration::from_secs(5);
// The fake child is a PowerShell process. Under a four-process live-E2E load it has
// the same 2.1–2.8s startup cost as run-log creation, so keep a finite bound with
// headroom rather than treating scheduler contention as a concurrency failure.
const FAKE_CONCURRENT_PROMPT_TIMEOUT: Duration = Duration::from_secs(5);

async fn wait_for_fake_run_log(root: &std::path::Path) {
    let run_logs = root.join("state").join("run-logs");
    tokio::time::timeout(FAKE_RUN_LOG_WAIT_TIMEOUT, async {
        loop {
            let started = fs::read_dir(&run_logs)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .any(|entry| {
                    fs::read_to_string(entry.path())
                        .is_ok_and(|contents| contents.contains("Created conversation "))
                });
            if started {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("fake agy did not create its invocation log");
}

#[tokio::test]
async fn fake_run_log_wait_tolerates_observed_powershell_startup_delay() {
    // Break caught: treating a Windows fake-child startup delay above two seconds as a failure.
    let root = fresh_test_root("delayed-fake-run-log");
    let run_logs = root.join("state").join("run-logs");
    fs::create_dir_all(&run_logs).unwrap();
    let run_log = run_logs.join("delayed.log");

    let delayed_writer = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(2100)).await;
        fs::write(
            run_log,
            "Created conversation 00000000-0000-4000-8000-000000000099\\n",
        )
        .unwrap();
    });

    wait_for_fake_run_log(&root).await;
    delayed_writer.await.unwrap();
}

#[tokio::test]
async fn prompt_executions_for_two_sessions_run_concurrently() {
    // Break caught: prompt execution hard-coding `agy`, preventing deterministic concurrency proof.
    let mut harness = ConcurrentHarness::new("parallel");
    let slow = harness.prepare("session-a", "slow");
    let fast = harness.prepare("session-b", "fast");
    let slow_task = tokio::spawn(crate::runtime::execute_prompt(
        slow,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    ));
    wait_for_fake_run_log(&harness.root).await;

    let fast_outcome = tokio::time::timeout(
        FAKE_CONCURRENT_PROMPT_TIMEOUT,
        crate::runtime::execute_prompt(
            fast,
            Arc::new(AtomicBool::new(false)),
            harness.output.clone(),
        ),
    )
    .await
    .expect("fast prompt did not complete while slow prompt was running");

    assert!(fast_outcome.response.error.is_none());
    assert!(!slow_task.is_finished());
    let slow_outcome = tokio::time::timeout(Duration::from_secs(6), slow_task)
        .await
        .expect("slow prompt timed out")
        .unwrap();
    assert!(slow_outcome.response.error.is_none());
}

#[tokio::test]
async fn cancellation_finishes_a_slow_fake_child_without_waiting_for_timeout() {
    // Break caught: cancellation leaving the fake child or its inherited pipes alive.
    let mut harness = ConcurrentHarness::new("cancel");
    let execution = harness.prepare("session-a", "slow");
    let cancelled = Arc::new(AtomicBool::new(false));
    let task = tokio::spawn(crate::runtime::execute_prompt(
        execution,
        cancelled.clone(),
        harness.output.clone(),
    ));
    wait_for_fake_run_log(&harness.root).await;

    cancelled.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("cancelled fake child did not finish promptly")
        .unwrap();

    assert_eq!(outcome.response.result.unwrap()["stopReason"], "cancelled");
}

#[tokio::test]
async fn oversized_child_stdout_fails_closed_without_buffering_all_output() {
    let mut harness = ConcurrentHarness::new("huge-stdout");
    let execution = harness.prepare("session-a", "huge");
    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        crate::runtime::execute_prompt(
            execution,
            Arc::new(AtomicBool::new(false)),
            harness.output.clone(),
        ),
    )
    .await
    .expect("huge fake stdout did not finish promptly");

    assert_eq!(
        outcome.response.error.unwrap()["message"],
        "agy stdout exceeded the maximum buffered size"
    );
}

#[cfg(windows)]
#[tokio::test]
async fn cancellation_terminates_windows_process_tree() {
    let mut harness = ConcurrentHarness::new("cancel-tree");
    let execution = harness.prepare("session-a", "tree");
    let cancelled = Arc::new(AtomicBool::new(false));
    let task = tokio::spawn(crate::runtime::execute_prompt(
        execution,
        cancelled.clone(),
        harness.output.clone(),
    ));
    wait_for_fake_run_log(&harness.root).await;

    let log_path = fs::read_dir(harness.root.join("state").join("run-logs"))
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("log"))
        .unwrap();
    let child_path = log_path.with_file_name(format!(
        "{}.child",
        log_path.file_name().unwrap().to_string_lossy()
    ));
    let child_pid = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(pid) = fs::read_to_string(&child_path) {
                if let Ok(pid) = pid.trim().parse::<u32>() {
                    break pid;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("fake descendant did not start");

    cancelled.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("process-tree cancellation did not finish")
        .unwrap();
    assert_eq!(outcome.response.result.unwrap()["stopReason"], "cancelled");

    tokio::time::sleep(Duration::from_millis(100)).await;
    let tasklist = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {child_pid}"), "/NH"])
        .output()
        .unwrap();
    let tasklist = String::from_utf8_lossy(&tasklist.stdout);
    assert!(
        !tasklist.contains(&child_pid.to_string()),
        "descendant survived: {tasklist}"
    );
}

#[tokio::test]
async fn concurrent_output_is_valid_newline_delimited_json() {
    // Break caught: concurrent prompt output interleaving partial JSON messages.
    let mut harness = ConcurrentHarness::new("jsonl");
    let first = harness.prepare("session-a", "fast");
    let second = harness.prepare("session-b", "fast");
    let one = crate::runtime::execute_prompt(
        first,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    );
    let two = crate::runtime::execute_prompt(
        second,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    );

    let (first_outcome, second_outcome) = tokio::join!(one, two);
    assert!(first_outcome.response.error.is_none());
    assert!(second_outcome.response.error.is_none());
    drop(harness.output);

    let mut line_count = 0;
    while let Some(line) = harness.receiver.recv().await {
        let value: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["jsonrpc"], "2.0");
        line_count += 1;
    }
    assert!(line_count >= 2);
}

#[tokio::test]
async fn successful_run_log_is_removed_only_after_state_commit() {
    // Break caught: deleting the run log inside execution before outcome persistence commits.
    let mut harness = ConcurrentHarness::new("run-log-commit-order");
    let execution = harness.prepare("session-a", "fast");
    let outcome = crate::runtime::execute_prompt(
        execution,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    )
    .await;
    let run_log_path = outcome.run_log_path.clone();

    assert!(outcome.response.error.is_none());
    assert!(run_log_path.exists());

    let response = crate::runtime::finalize_prompt_outcome(&mut harness.adapter, outcome);

    assert!(response.error.is_none());
    assert!(!run_log_path.exists());
}

#[tokio::test]
async fn failed_process_run_log_is_retained_after_state_commit() {
    // Break caught: deleting diagnostics for a child process that returned a terminal error.
    let mut harness = ConcurrentHarness::new("failed-process-run-log");
    let execution = harness.prepare("session-a", "fail");
    let outcome = crate::runtime::execute_prompt(
        execution,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    )
    .await;
    let run_log_path = outcome.run_log_path.clone();

    assert!(outcome.response.error.is_some());
    assert!(run_log_path.exists());

    let response = crate::runtime::finalize_prompt_outcome(&mut harness.adapter, outcome);

    assert!(response.error.is_some());
    assert!(run_log_path.exists());
}

#[test]
fn retained_run_log_message_uses_only_a_bounded_bridge_reference() {
    // Break caught: exposing an unbounded or sensitive parent path in runtime diagnostics.
    let controlled_name = "00000000-0000-4000-8000-000000000123.log";
    let untrusted_parent = format!("PARENT_PATH_SENTINEL_{}", "x".repeat(4096));
    let run_log_path = PathBuf::from(&untrusted_parent).join(controlled_name);

    let message = crate::runtime::retained_run_log_message(
        "agy completed but no conversation ID was found",
        &run_log_path,
    );

    assert!(message.len() <= crate::runtime::MAX_RUNTIME_ERROR_MESSAGE_LEN);
    assert!(!message.contains("PARENT_PATH_SENTINEL"));
    assert!(message.ends_with(&format!("run-logs/{controlled_name}")));

    let untrusted_name = format!("UNTRUSTED_FILE_SENTINEL_{}.log", "y".repeat(4096));
    let fallback = crate::runtime::retained_run_log_message(
        &"untrusted summary ".repeat(4096),
        &PathBuf::from(untrusted_parent).join(untrusted_name),
    );
    assert!(fallback.len() <= crate::runtime::MAX_RUNTIME_ERROR_MESSAGE_LEN);
    assert!(!fallback.contains("PARENT_PATH_SENTINEL"));
    assert!(!fallback.contains("UNTRUSTED_FILE_SENTINEL"));
    assert!(fallback.ends_with("run-logs/unavailable.log"));
}

#[test]
fn retained_run_logs_are_bounded_without_deleting_active_or_current() {
    let root = fresh_test_root("run-log-retention");
    let run_logs_dir = root.join("run-logs");
    fs::create_dir_all(&run_logs_dir).unwrap();
    let active_path = run_logs_dir.join(format!("{}.log", Uuid::new_v4()));
    let current_path = run_logs_dir.join(format!("{}.log", Uuid::new_v4()));
    fs::write(&active_path, b"active").unwrap();
    fs::write(&current_path, b"current").unwrap();
    crate::runtime::register_active_run_log(&active_path);

    for _ in 0..(crate::runtime::MAX_RETAINED_RUN_LOGS + 3) {
        let path = run_logs_dir.join(format!("{}.log", Uuid::new_v4()));
        fs::write(path, b"retained diagnostic").unwrap();
    }

    crate::runtime::prune_retained_run_logs(&run_logs_dir, Some(&current_path));
    crate::runtime::unregister_active_run_log(&active_path);

    assert!(active_path.exists());
    assert!(current_path.exists());
    let candidate_count = fs::read_dir(&run_logs_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path() != active_path && entry.path() != current_path)
        .count();
    assert!(candidate_count <= crate::runtime::MAX_RETAINED_RUN_LOGS);

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn failed_process_stderr_and_prompt_never_reach_protocol_output() {
    // Break caught: child stderr or the submitted prompt being reflected into ACP output.
    let mut harness = ConcurrentHarness::new("sanitized-child-error");
    let prompt = "stderr-fail";
    let execution = harness.prepare("session-a", prompt);
    let outcome = crate::runtime::execute_prompt(
        execution,
        Arc::new(AtomicBool::new(false)),
        harness.output.clone(),
    )
    .await;
    let run_log_path = outcome.run_log_path.clone();
    let response = crate::runtime::finalize_prompt_outcome(&mut harness.adapter, outcome);

    let response_json = serde_json::to_string(&response).unwrap();
    let error = response.error.as_ref().unwrap();
    let message = error["message"].as_str().unwrap();
    let expected_reference = format!(
        "run-logs/{}",
        run_log_path.file_name().unwrap().to_str().unwrap()
    );
    let mut protocol_output = String::new();
    while let Ok(line) = harness.receiver.try_recv() {
        protocol_output.push_str(&line);
    }

    assert_eq!(error["code"], -32000);
    assert!(message.contains("status"));
    assert!(message.contains("run log retained at"));
    assert!(message.ends_with(&expected_reference));
    assert!(message.len() <= crate::runtime::MAX_RUNTIME_ERROR_MESSAGE_LEN);
    assert!(!message.contains(&harness.root.to_string_lossy().to_string()));
    assert!(run_log_path.exists());
    for forbidden in ["AGY_STDERR_SECRET_SENTINEL", prompt] {
        assert!(!response_json.contains(forbidden));
        assert!(!message.contains(forbidden));
        assert!(!protocol_output.contains(forbidden));
    }
}

fn test_adapter(root: &std::path::Path) -> Adapter {
    Adapter {
        sessions: HashMap::new(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("state").join("sessions.json"),
        available_models: vec!["fake-model\tFake Model".to_string()],
        skip_naration: false,
        command: CommandSpec::new("agy", Vec::new()),
        session_access: HashMap::new(),
        next_access: 0,
        active_sessions: std::collections::HashSet::new(),
        mcp_configs: HashMap::new(),
    }
}

fn session_setup_params(cwd: &std::path::Path) -> Value {
    json!({"cwd": cwd.to_string_lossy(), "mcpServers": []})
}

fn session_lifecycle_params(session_id: &str, cwd: &std::path::Path) -> Value {
    json!({
        "sessionId": session_id,
        "cwd": cwd.to_string_lossy(),
        "mcpServers": [],
    })
}

fn assert_persistence_error(response: &JsonRpcResponse) {
    assert!(response.result.is_none());
    assert_eq!(response.error.as_ref().unwrap()["code"], -32603);
    assert_eq!(
        response.error.as_ref().unwrap()["message"],
        "failed to persist session state"
    );
}

fn assert_persistence_error_value(response: &Value) {
    assert!(response["result"].is_null());
    assert_eq!(response["error"]["code"], -32603);
    assert_eq!(
        response["error"]["message"],
        "failed to persist session state"
    );
}

fn write_corrupt_state(adapter: &Adapter) -> Vec<u8> {
    let bytes = b"{not valid session state".to_vec();
    fs::create_dir_all(adapter.state_file.parent().unwrap()).unwrap();
    fs::write(&adapter.state_file, &bytes).unwrap();
    bytes
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

#[test]
fn session_new_fails_closed_when_state_parent_is_a_file() {
    // Break caught: reporting a new session as successful after state creation failed.
    let root = fresh_test_root("new-state-parent-file");
    let blocked_parent = root.join("blocked-state-parent");
    fs::write(&blocked_parent, b"not a directory").unwrap();
    let mut adapter = test_adapter(&root);
    adapter.state_file = blocked_parent.join("sessions.json");

    let response = adapter.handle_session_new(json!(1), &session_setup_params(&root));

    assert_persistence_error(&response);
    assert!(adapter.sessions.is_empty());
}

#[test]
fn session_load_persistence_failure_preserves_memory_and_emits_no_replay() {
    // Break caught: mutating cwd/cursor or emitting replay before the candidate is durable.
    let root = fresh_test_root("load-corrupt-state");
    let old_cwd = root.join("old");
    let new_cwd = root.join("new");
    fs::create_dir_all(&old_cwd).unwrap();
    fs::create_dir_all(&new_cwd).unwrap();
    let mut adapter = test_adapter(&root);
    fs::create_dir_all(&adapter.conversations_dir).unwrap();
    let conversation_id = "00000000-0000-4000-8000-000000000013";
    let replay_db = adapter
        .conversations_dir
        .join(format!("{conversation_id}.db"));
    let connection = Connection::open(&replay_db).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL,
                step_payload BLOB NOT NULL
            )",
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
            rusqlite::params![9i64, make_user_payload("replayed prompt")],
        )
        .unwrap();
    drop(connection);
    adapter.sessions.insert(
        "load-session".to_string(),
        crate::types::Session {
            conversation_id: Some(conversation_id.to_string()),
            last_step_idx: 7,
            model_id: Some("old-model".to_string()),
            cwd: old_cwd.clone(),
        },
    );
    let corrupt_bytes = write_corrupt_state(&adapter);

    let output = adapter.handle_session_load(
        json!(2),
        &session_lifecycle_params("load-session", &new_cwd),
    );

    assert_eq!(output.len(), 1, "replay must not precede durable state");
    let response: Value = serde_json::from_str(&output[0]).unwrap();
    assert_persistence_error_value(&response);
    let session = &adapter.sessions["load-session"];
    assert_eq!(session.cwd, old_cwd);
    assert_eq!(session.last_step_idx, 7);
    assert_eq!(fs::read(&adapter.state_file).unwrap(), corrupt_bytes);
}

#[test]
fn session_resume_persistence_failure_preserves_memory() {
    // Break caught: changing the in-memory cwd before a resume snapshot is durable.
    let root = fresh_test_root("resume-corrupt-state");
    let old_cwd = root.join("old");
    let new_cwd = root.join("new");
    fs::create_dir_all(&old_cwd).unwrap();
    fs::create_dir_all(&new_cwd).unwrap();
    let mut adapter = test_adapter(&root);
    adapter.sessions.insert(
        "resume-session".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
            cwd: old_cwd.clone(),
        },
    );
    let corrupt_bytes = write_corrupt_state(&adapter);

    let response = adapter.handle_session_resume(
        json!(3),
        &session_lifecycle_params("resume-session", &new_cwd),
    );

    assert_persistence_error(&response);
    assert_eq!(adapter.sessions["resume-session"].cwd, old_cwd);
    assert_eq!(fs::read(&adapter.state_file).unwrap(), corrupt_bytes);
}

#[test]
fn session_set_model_persistence_failure_preserves_memory_and_corrupt_store() {
    // Break caught: overwriting corrupt state and exposing an uncommitted model in memory.
    let root = fresh_test_root("model-corrupt-state");
    let mut adapter = test_adapter(&root);
    adapter.sessions.insert(
        "model-session".to_string(),
        crate::types::Session {
            conversation_id: Some(TEST_CONVERSATION_B.to_string()),
            last_step_idx: 4,
            model_id: Some("old-model".to_string()),
            cwd: root.clone(),
        },
    );
    let corrupt_bytes = write_corrupt_state(&adapter);

    let response = adapter.handle_session_set_model(
        json!(4),
        &json!({"sessionId": "model-session", "modelId": "new-model"}),
    );

    assert_persistence_error(&response);
    assert_eq!(
        adapter.sessions["model-session"].model_id.as_deref(),
        Some("old-model")
    );
    assert_eq!(fs::read(&adapter.state_file).unwrap(), corrupt_bytes);
}

#[test]
fn session_set_config_persistence_failure_preserves_memory() {
    // Break caught: exposing an uncommitted config-option model in memory.
    let root = fresh_test_root("config-corrupt-state");
    let mut adapter = test_adapter(&root);
    adapter.sessions.insert(
        "config-session".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: Some("old-model".to_string()),
            cwd: root.clone(),
        },
    );
    write_corrupt_state(&adapter);

    let response = adapter.handle_session_set_config_option(
        json!(5),
        &json!({"sessionId": "config-session", "configId": "model", "value": "new-model"}),
    );

    assert_persistence_error(&response);
    assert_eq!(
        adapter.sessions["config-session"].model_id.as_deref(),
        Some("old-model")
    );
}

#[test]
fn corrupt_store_restore_is_internal_error_not_unknown_session() {
    // Break caught: treating an unreadable state store as an empty store/unknown session.
    let root = fresh_test_root("restore-corrupt-state");
    let mut adapter = test_adapter(&root);
    write_corrupt_state(&adapter);

    let response = adapter.handle_session_resume(
        json!(6),
        &session_lifecycle_params("persisted-session", &root),
    );

    assert_persistence_error(&response);
    assert!(adapter.sessions.is_empty());
}

#[test]
fn corrupt_state_load_returns_error_and_preserves_original_bytes() {
    // Break caught: parsing malformed JSON as an empty store that a later write can replace.
    let root = fresh_test_root("load-corrupt-state-api");
    let adapter = test_adapter(&root);
    let corrupt_bytes = write_corrupt_state(&adapter);

    let error = adapter
        .load_store()
        .expect_err("corrupt session state must not become an empty store");

    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(fs::read(&adapter.state_file).unwrap(), corrupt_bytes);
}

#[test]
fn prepare_prompt_rejects_unknown_session_before_log_creation() {
    // Break caught: invalid sessions creating invocation artifacts before rejection.
    let root = fresh_test_root("unknown-prompt");
    let mut adapter = test_adapter(&root);
    let error = adapter
        .prepare_prompt(
            json!(4),
            &json!({"sessionId": "missing", "prompt": [{"type": "text", "text": "hello"}]}),
        )
        .unwrap_err();

    assert_eq!(error.error.as_ref().unwrap()["code"], -32000);
    assert!(!adapter
        .state_file
        .parent()
        .unwrap()
        .join("run-logs")
        .exists());
}

#[test]
fn prepare_prompt_rejects_every_invalid_session_id_without_side_effects() {
    // Break caught: malformed session IDs reaching persistence, cache mutation, or run-log setup.
    let cases = [
        (
            "missing",
            json!({"prompt": [{"type": "text", "text": "hello"}]}),
        ),
        (
            "null",
            json!({"sessionId": null, "prompt": [{"type": "text", "text": "hello"}]}),
        ),
        (
            "number",
            json!({"sessionId": 42, "prompt": [{"type": "text", "text": "hello"}]}),
        ),
        (
            "empty",
            json!({"sessionId": "", "prompt": [{"type": "text", "text": "hello"}]}),
        ),
        (
            "whitespace",
            json!({"sessionId": "  \t", "prompt": [{"type": "text", "text": "hello"}]}),
        ),
    ];

    for (label, params) in cases {
        let root = fresh_test_root(&format!("invalid-session-id-{label}"));
        let blocked_parent = root.join("blocked-state-parent");
        fs::write(&blocked_parent, b"not a directory").unwrap();
        let mut adapter = test_adapter(&root);
        adapter.state_file = blocked_parent.join("sessions.json");
        let sessions_before = adapter.sessions.clone();

        let error = adapter.prepare_prompt(json!(20), &params).unwrap_err();

        assert_eq!(error.error.as_ref().unwrap()["code"], -32602, "{label}");
        assert_eq!(adapter.sessions, sessions_before, "{label}");
        assert!(!blocked_parent.join("sessions.lock").exists(), "{label}");
        assert!(!blocked_parent.join("run-logs").exists(), "{label}");
    }
}

#[test]
fn malformed_prompt_does_not_restore_a_persisted_nonresident_session() {
    // Break caught: restoring/evicting session cache state before validating every prompt block.
    let root = fresh_test_root("malformed-prompt-nonresident");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let session_id = "persisted-nonresident";
    let persisted = crate::types::Session {
        conversation_id: None,
        last_step_idx: -1,
        model_id: Some("fake-model".to_string()),
        cwd,
    };
    adapter.persist_session(session_id, &persisted).unwrap();
    let store_before = adapter.load_store().unwrap();
    assert!(!adapter.sessions.contains_key(session_id));

    let error = adapter
        .prepare_prompt(
            json!(21),
            &json!({"sessionId": session_id, "prompt": [
                {"type": "text", "text": "hello"},
                {"type": "image", "data": "AA==", "mimeType": "image/png"}
            ]}),
        )
        .unwrap_err();

    assert_eq!(error.error.as_ref().unwrap()["code"], -32602);
    assert!(!adapter.sessions.contains_key(session_id));
    assert_eq!(
        adapter.load_store().unwrap().sessions,
        store_before.sessions
    );
}

#[test]
fn prepare_prompt_rejects_empty_text() {
    // Break caught: spawning agy with a prompt that is empty after normalization.
    let root = fresh_test_root("empty-prompt");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    let error = adapter
        .prepare_prompt(
            json!(5),
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "  "}]}),
        )
        .unwrap_err();

    assert_eq!(error.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn prepare_prompt_rejects_unsupported_or_mixed_content() {
    // Break caught: silently dropping a non-text block while executing the remaining text.
    let root = fresh_test_root("mixed-prompt");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    let error = adapter
        .prepare_prompt(
            json!(6),
            &json!({"sessionId": session_id, "prompt": [
                {"type": "text", "text": "hello"},
                {"type": "image", "data": "AA==", "mimeType": "image/png"}
            ]}),
        )
        .unwrap_err();

    assert_eq!(error.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn prepare_prompt_rejects_non_string_text_content() {
    // Coverage: a text-shaped block must not stringify or silently drop structured data.
    let root = fresh_test_root("non-string-prompt");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let session_id = open_test_session(&mut adapter, &cwd);

    let error = adapter
        .prepare_prompt(
            json!(6),
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": 42}]}),
        )
        .unwrap_err();

    assert_eq!(error.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn prepare_prompt_snapshots_session_cwd_model_and_conversation() {
    // Break caught: prompt execution rereading mutable session fields after preparation.
    let root = fresh_test_root("snapshot");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    let session = adapter.sessions.get_mut(&session_id).unwrap();
    session.model_id = Some("fake-model".to_string());
    session.conversation_id = Some("00000000-0000-4000-8000-000000000002".to_string());

    let execution = adapter
        .prepare_prompt(
            json!(7),
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]}),
        )
        .unwrap();

    assert_eq!(execution.cwd, cwd);
    assert_eq!(execution.model_id.as_deref(), Some("fake-model"));
    assert_eq!(
        execution.conversation_id.as_deref(),
        Some("00000000-0000-4000-8000-000000000002")
    );
    assert_eq!(execution.prompt_text, "hello");
}

#[test]
fn session_new_rejects_missing_or_relative_cwd() {
    // Break caught: accepting a session whose workspace cannot be safely scoped.
    let root = fresh_test_root("bad-cwd");
    let mut adapter = test_adapter(&root);
    for params in [
        json!({"mcpServers": []}),
        json!({"cwd": "relative", "mcpServers": []}),
    ] {
        let response = adapter.handle_session_new(json!(1), &params);
        assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
    }
}

#[test]
fn session_new_rejects_empty_cwd() {
    // Coverage: an empty cwd must not fall back to the bridge process directory.
    let root = fresh_test_root("empty-cwd");
    let mut adapter = test_adapter(&root);
    let response = adapter.handle_session_new(json!(1), &json!({"cwd": "", "mcpServers": []}));

    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn session_new_rejects_absolute_existing_non_directory_cwd() {
    // Coverage: absolute path validation must distinguish files from directories.
    let root = fresh_test_root("file-cwd");
    let file = root.join("workspace-file");
    fs::write(&file, "not a directory").unwrap();
    let mut adapter = test_adapter(&root);
    let response = adapter.handle_session_new(
        json!(1),
        &json!({"cwd": file.to_string_lossy(), "mcpServers": []}),
    );

    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn session_new_rejects_missing_mcp_servers() {
    // Coverage: omitting mcpServers must not be treated as an empty list.
    let root = fresh_test_root("missing-mcp");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let response = adapter.handle_session_new(json!(1), &json!({"cwd": cwd.to_string_lossy()}));

    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
}

#[test]
fn session_new_rejects_non_array_mcp_servers() {
    // Coverage: object-shaped MCP configuration must fail instead of being ignored.
    let root = fresh_test_root("non-array-mcp");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let response = adapter.handle_session_new(
        json!(1),
        &json!({"cwd": cwd.to_string_lossy(), "mcpServers": {}}),
    );

    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
}

fn mcp_session_params(cwd: &std::path::Path) -> Value {
    json!({
        "cwd": cwd.to_string_lossy(),
        "mcpServers": [
            {
                "name": "desk",
                "command": "node",
                "args": ["desk.mjs"],
                "env": [{"name": "DESK_TOKEN", "value": "MCP_ENV_SECRET_SENTINEL"}],
            },
            {
                "type": "http",
                "name": "remote",
                "url": "https://example.com/mcp",
                "headers": [{"name": "Authorization", "value": "Bearer MCP_HEADER_SECRET_SENTINEL"}],
            },
        ],
    })
}

fn agy_mcp_config() -> Value {
    json!({
        "mcpServers": {
            "desk": {
                "command": "node",
                "args": ["desk.mjs"],
                "env": {"DESK_TOKEN": "MCP_ENV_SECRET_SENTINEL"},
            },
            "remote": {
                "serverUrl": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer MCP_HEADER_SECRET_SENTINEL"},
            },
        },
    })
}

#[test]
fn session_mcp_servers_are_held_in_agy_shape_in_memory_only_and_replaced_on_resume() {
    // Break caught: dropping a seat's MCP servers, or persisting their secrets in the session store.
    let root = fresh_test_root("mcp-held");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let response = adapter.handle_session_new(json!(1), &mcp_session_params(&cwd));
    let session_id = response.result.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    assert_eq!(adapter.mcp_configs[&session_id], agy_mcp_config());
    let stored = fs::read_to_string(&adapter.state_file).unwrap();
    assert!(!stored.contains("MCP_ENV_SECRET_SENTINEL"));
    assert!(!stored.contains("MCP_HEADER_SECRET_SENTINEL"));

    let resumed =
        adapter.handle_session_resume(json!(2), &session_lifecycle_params(&session_id, &cwd));
    assert!(resumed.error.is_none());
    assert!(!adapter.mcp_configs.contains_key(&session_id));
}

#[test]
fn session_new_rejects_mcp_servers_agy_cannot_run() {
    // Break caught: accepting a server agy would silently never start.
    let root = fresh_test_root("mcp-rejected");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    for servers in [
        json!([{"type": "sse", "name": "events", "url": "https://example.com/sse", "headers": []}]),
        json!([{"type": "stdio"}]),
        json!([{"name": "twice", "command": "a"}, {"name": "twice", "command": "b"}]),
        json!([{"name": "env", "command": "a", "env": [{"name": "X", "value": 1}]}]),
    ] {
        let response = adapter.handle_session_new(
            json!(1),
            &json!({"cwd": cwd.to_string_lossy(), "mcpServers": servers}),
        );
        assert_eq!(
            response.error.as_ref().unwrap()["code"],
            -32602,
            "{servers}"
        );
    }
    assert!(adapter.mcp_configs.is_empty());
}

#[tokio::test]
async fn prompt_run_hands_agy_the_session_mcp_config_through_an_added_dir_and_removes_it() {
    // Break caught: MCP servers accepted at session setup but never reaching agy, or their config outliving the run.
    let root = fresh_test_root("mcp-run");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    adapter.command = fake_agy_command(&root);
    let response = adapter.handle_session_new(json!(1), &mcp_session_params(&cwd));
    let session_id = response.result.unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();
    let execution = adapter
        .prepare_prompt(
            json!(2),
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}),
        )
        .unwrap();
    let (output, _receiver) = crate::output::channel();

    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        crate::runtime::execute_prompt(execution, Arc::new(AtomicBool::new(false)), output),
    )
    .await
    .expect("fake agy did not finish");

    let mut seen = outcome.run_log_path.clone().into_os_string();
    seen.push(".mcp");
    let seen: Value = serde_json::from_str(&fs::read_to_string(&seen).unwrap()).unwrap();
    assert_eq!(seen, agy_mcp_config());
    let left = fs::read_dir(root.join("state").join("mcp"))
        .unwrap()
        .count();
    assert_eq!(left, 0);
}

#[test]
fn session_new_retains_and_persists_cwd() {
    // Break caught: losing cwd between creation, in-memory execution, and restart.
    let root = fresh_test_root("persist-cwd");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let session_id = open_test_session(&mut adapter, &cwd);
    assert_eq!(adapter.sessions[&session_id].cwd, cwd);
    assert_eq!(
        adapter
            .load_store()
            .expect("new session state should be readable")
            .sessions[&session_id]
            .cwd
            .as_deref(),
        Some(cwd.to_string_lossy().as_ref()),
    );
}

#[test]
fn legacy_stored_session_without_cwd_loads_with_request_cwd() {
    // Break caught: rejecting pre-cwd state files instead of refreshing their context.
    let root = fresh_test_root("legacy-cwd");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
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

#[test]
fn session_load_restores_unbound_session_after_restart() {
    // Break caught: restore filters out persisted sessions until a conversation is bound.
    let root = fresh_test_root("load-unbound-restart");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let session_id = {
        let mut adapter = test_adapter(&root);
        open_test_session(&mut adapter, &cwd)
    };

    let mut restarted = test_adapter(&root);
    let output =
        restarted.handle_session_load(json!(2), &session_lifecycle_params(&session_id, &cwd));
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();

    assert!(response.get("result").is_some(), "response: {response}");
    assert_eq!(restarted.sessions[&session_id].conversation_id, None);
    assert_eq!(restarted.sessions[&session_id].cwd, cwd);
}

#[test]
fn session_resume_restores_unbound_session_after_restart() {
    // Break caught: resume treats a persisted unbound session as unknown after restart.
    let root = fresh_test_root("resume-unbound-restart");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let session_id = {
        let mut adapter = test_adapter(&root);
        open_test_session(&mut adapter, &cwd)
    };

    let mut restarted = test_adapter(&root);
    let response =
        restarted.handle_session_resume(json!(2), &session_lifecycle_params(&session_id, &cwd));

    assert!(response.error.is_none(), "error: {:?}", response.error);
    assert_eq!(restarted.sessions[&session_id].conversation_id, None);
    assert_eq!(restarted.sessions[&session_id].cwd, cwd);
}

#[test]
fn session_load_persists_refreshed_cwd() {
    // Break caught: load updates cwd only in memory and leaves stale persisted context.
    let root = fresh_test_root("load-refresh-cwd");
    let cwd_a = root.join("workspace-a");
    let cwd_b = root.join("workspace-b");
    fs::create_dir_all(&cwd_a).unwrap();
    fs::create_dir_all(&cwd_b).unwrap();
    let session_id = {
        let mut adapter = test_adapter(&root);
        open_test_session(&mut adapter, &cwd_a)
    };

    let mut restarted = test_adapter(&root);
    let output =
        restarted.handle_session_load(json!(2), &session_lifecycle_params(&session_id, &cwd_b));
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response.get("result").is_some(), "response: {response}");

    let reread = test_adapter(&root)
        .load_store()
        .expect("loaded session state should be readable");
    assert_eq!(
        reread.sessions[&session_id].cwd.as_deref(),
        Some(cwd_b.to_string_lossy().as_ref()),
    );
}

#[test]
fn session_resume_persists_refreshed_cwd() {
    // Break caught: resume updates cwd only in memory and leaves stale persisted context.
    let root = fresh_test_root("resume-refresh-cwd");
    let cwd_a = root.join("workspace-a");
    let cwd_b = root.join("workspace-b");
    fs::create_dir_all(&cwd_a).unwrap();
    fs::create_dir_all(&cwd_b).unwrap();
    let session_id = {
        let mut adapter = test_adapter(&root);
        open_test_session(&mut adapter, &cwd_a)
    };

    let mut restarted = test_adapter(&root);
    let response =
        restarted.handle_session_resume(json!(2), &session_lifecycle_params(&session_id, &cwd_b));
    assert!(response.error.is_none(), "error: {:?}", response.error);

    let reread = test_adapter(&root)
        .load_store()
        .expect("resumed session state should be readable");
    assert_eq!(
        reread.sessions[&session_id].cwd.as_deref(),
        Some(cwd_b.to_string_lossy().as_ref()),
    );
}

#[test]
fn malformed_json_returns_parse_error() {
    let error = parse_jsonrpc_line("{").unwrap_err();
    assert_eq!(error.id, Value::Null);
    assert_eq!(error.error.unwrap()["code"], -32700);
}

#[test]
fn invalid_jsonrpc_envelope_returns_invalid_request() {
    let error =
        parse_jsonrpc_line(r#"{"jsonrpc":"1.0","id":7,"method":"initialize"}"#).unwrap_err();
    assert_eq!(error.id, json!(7));
    assert_eq!(error.error.unwrap()["code"], -32600);
}

#[test]
fn parser_distinguishes_request_from_notification() {
    let request =
        parse_jsonrpc_line(r#"{"jsonrpc":"2.0","id":null,"method":"initialize","params":{}}"#)
            .unwrap();
    assert!(matches!(
        request,
        IncomingMessage::Request {
            id: Value::Null,
            ..
        }
    ));

    let notification = parse_jsonrpc_line(
        r#"{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s"}}"#,
    )
    .unwrap();
    assert!(matches!(notification, IncomingMessage::Notification { .. }));
}

#[test]
fn response_helpers_keep_jsonrpc_shape() {
    let response = JsonRpcResponse::error(json!(3), -32602, "Invalid params");
    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({
            "jsonrpc": "2.0",
            "id": 3,
            "error": {"code": -32602, "message": "Invalid params"}
        })
    );
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        if value < 128 {
            out.push(value as u8);
            break;
        }
        out.push(((value as u8) & 0x7F) | 0x80);
        value >>= 7;
    }
}

fn push_len_field(out: &mut Vec<u8>, field_number: u64, bytes: &[u8]) {
    push_varint(out, (field_number << 3) | 2);
    push_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_varint_field(out: &mut Vec<u8>, field_number: u64, value: u64) {
    push_varint(out, field_number << 3);
    push_varint(out, value);
}

fn make_assistant_payload(text: &str) -> Vec<u8> {
    let mut inner = Vec::new();
    push_len_field(&mut inner, 1, text.as_bytes());

    let mut outer = Vec::new();
    push_len_field(&mut outer, 20, &inner);
    outer
}

fn make_assistant_payload_with_thought(text: &str, thought: &str) -> Vec<u8> {
    let mut inner = Vec::new();
    push_len_field(&mut inner, 1, text.as_bytes());
    push_len_field(&mut inner, 3, thought.as_bytes());

    let mut outer = Vec::new();
    push_len_field(&mut outer, 20, &inner);
    outer
}

#[test]
fn test_parse_skip_naration_flag() {
    assert!(
        Cli::try_parse_from(["agy-acp", "--skip-naration"])
            .unwrap()
            .skip_naration
    );
    assert!(!Cli::try_parse_from(["agy-acp"]).unwrap().skip_naration);
    assert!(Cli::try_parse_from(["agy-acp", "--skip-narration"]).is_err());
}

fn make_user_payload(text: &str) -> Vec<u8> {
    let mut content = Vec::new();
    push_len_field(&mut content, 1, text.as_bytes());

    let mut prompt = Vec::new();
    push_len_field(&mut prompt, 2, text.as_bytes());
    push_len_field(&mut prompt, 3, &content);

    let mut outer = Vec::new();
    push_len_field(&mut outer, 19, &prompt);
    outer
}

fn make_title_payload(title: &str) -> Vec<u8> {
    let mut title_update = Vec::new();
    push_len_field(&mut title_update, 4, title.as_bytes());

    let mut outer = Vec::new();
    push_len_field(&mut outer, 30, &title_update);
    outer
}

#[test]
fn poller_advances_tail_cursor_without_replaying_or_missing_incremental_text() {
    let root = std::env::temp_dir().join(format!("agy-acp-stream-cursor-{}", Uuid::new_v4()));
    let conversations_dir = root.join("conversations");
    fs::create_dir_all(&conversations_dir).unwrap();

    let conversation_id = "00000000-0000-4000-8000-000000000020";
    let db_path = conversations_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL,
            step_payload BLOB
        )",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 23, ?2)",
        rusqlite::params![1i64, make_title_payload("Old title")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 23, ?2)",
        rusqlite::params![2i64, make_title_payload("Current title")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![3i64, make_assistant_payload("hello")],
    )
    .unwrap();
    drop(conn);

    let state = Arc::new(Mutex::new(crate::types::StreamingState {
        conversation_id: Some(conversation_id.to_string()),
        base_step_idx: -1,
        last_step_idx: -1,
        ..Default::default()
    }));

    let first = crate::streaming::poll_streaming_delta(&conversations_dir, None, "session", &state);
    assert_eq!(
        first.len(),
        3,
        "first poll should emit both titles and text"
    );
    assert_eq!(
        state.lock().unwrap().base_step_idx,
        2,
        "cursor should retain one-row overlap for in-place payload growth"
    );

    let second =
        crate::streaming::poll_streaming_delta(&conversations_dir, None, "session", &state);
    assert!(
        second.is_empty(),
        "a stable database must not replay previously emitted title updates"
    );

    let conn = Connection::open(&db_path).unwrap();
    conn.execute(
        "UPDATE steps SET step_payload = ?1 WHERE idx = 3",
        rusqlite::params![make_assistant_payload("hello world")],
    )
    .unwrap();
    let third = crate::streaming::poll_streaming_delta(&conversations_dir, None, "session", &state);
    let third_updates: Vec<Value> = third
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(third_updates.len(), 1);
    assert_eq!(
        third_updates[0]["params"]["update"]["sessionUpdate"],
        "agent_message_chunk"
    );
    assert_eq!(
        third_updates[0]["params"]["update"]["content"]["text"],
        " world"
    );

    let conn = Connection::open(&db_path).unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![4i64, make_assistant_payload("next")],
    )
    .unwrap();
    let fourth =
        crate::streaming::poll_streaming_delta(&conversations_dir, None, "session", &state);
    let fourth_updates: Vec<Value> = fourth
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(fourth_updates.len(), 1);
    assert_eq!(
        fourth_updates[0]["params"]["update"]["content"]["text"],
        "next"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn poller_prunes_stale_streaming_bookkeeping_after_tail_advance() {
    let root = std::env::temp_dir().join(format!("agy-acp-stream-memory-{}", Uuid::new_v4()));
    let conversations_dir = root.join("conversations");
    fs::create_dir_all(&conversations_dir).unwrap();

    let conversation_id = "00000000-0000-4000-8000-000000000021";
    let db_path = conversations_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL,
            step_payload BLOB
        )",
    )
    .unwrap();

    let rows = vec![
        (
            1i64,
            15i64,
            make_assistant_payload_with_thought("one", "thought one"),
        ),
        (
            2i64,
            8i64,
            make_tool_payload(
                "call-2",
                "view_file",
                r#"{"toolAction":"Read file 2","toolSummary":"Read file 2"}"#,
                "Read file 2",
                None,
            ),
        ),
        (
            3i64,
            15i64,
            make_assistant_payload_with_thought("three", "thought three"),
        ),
        (
            4i64,
            8i64,
            make_tool_payload(
                "call-4",
                "view_file",
                r#"{"toolAction":"Read file 4","toolSummary":"Read file 4"}"#,
                "Read file 4",
                None,
            ),
        ),
        (
            5i64,
            15i64,
            make_assistant_payload_with_thought("five", "thought five"),
        ),
        (
            6i64,
            8i64,
            make_tool_payload(
                "call-6",
                "view_file",
                r#"{"toolAction":"Read file 6","toolSummary":"Read file 6"}"#,
                "Read file 6",
                None,
            ),
        ),
        (
            7i64,
            15i64,
            make_assistant_payload_with_thought("seven", "thought seven"),
        ),
    ];
    for (idx, step_type, payload) in rows {
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, ?2, ?3)",
            rusqlite::params![idx, step_type, payload],
        )
        .unwrap();
    }
    drop(conn);

    let state = Arc::new(Mutex::new(crate::types::StreamingState {
        conversation_id: Some(conversation_id.to_string()),
        base_step_idx: -1,
        last_step_idx: -1,
        ..Default::default()
    }));

    let first = crate::streaming::poll_streaming_delta(&conversations_dir, None, "session", &state);
    assert!(
        first.iter().any(|line| line.contains("\"tool_call\"")),
        "fixture should emit tool updates before checking bookkeeping"
    );

    let guard = state.lock().unwrap();
    assert_eq!(guard.base_step_idx, 6);
    let mut agent_keys: Vec<_> = guard.agent_text_lengths.keys().copied().collect();
    agent_keys.sort_unstable();
    assert_eq!(
        agent_keys,
        vec![7],
        "only the current overlap row needs assistant text bookkeeping"
    );
    let mut thought_keys: Vec<_> = guard.thought_text_lengths.keys().copied().collect();
    thought_keys.sort_unstable();
    assert_eq!(
        thought_keys,
        vec![7],
        "only the current overlap row needs thought bookkeeping"
    );
    assert!(
        guard.emitted_tool_steps.is_empty(),
        "completed tool rows older than the overlap should not stay resident"
    );

    let _ = fs::remove_dir_all(root);
}

fn make_tool_payload(
    call_id: &str,
    tool_name: &str,
    input_json: &str,
    summary: &str,
    result_field: Option<(u64, Vec<u8>)>,
) -> Vec<u8> {
    let mut call = Vec::new();
    push_len_field(&mut call, 1, call_id.as_bytes());
    push_len_field(&mut call, 2, tool_name.as_bytes());
    push_len_field(&mut call, 3, input_json.as_bytes());
    push_len_field(&mut call, 9, tool_name.as_bytes());

    let mut tool = Vec::new();
    push_len_field(&mut tool, 4, &call);
    push_len_field(&mut tool, 30, summary.as_bytes());

    let mut outer = Vec::new();
    push_varint_field(&mut outer, 1, 7);
    push_len_field(&mut outer, 5, &tool);
    if let Some((field, result)) = result_field {
        push_len_field(&mut outer, field, &result);
    }
    outer
}

#[test]
fn test_extract_text_from_step_payload_field20_field1() {
    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(0x05);
    inner.extend_from_slice(b"hello");

    let mut blob = vec![0x08, 0x0F, 0xA2, 0x01, inner.len() as u8];
    blob.extend_from_slice(&inner);
    assert_eq!(
        extract_text_from_step_payload(&blob),
        Some("hello".to_string())
    );
}

#[test]
fn test_extract_text_returns_none_without_field20() {
    let blob = vec![0x08, 0x03];
    assert_eq!(extract_text_from_step_payload(&blob), None);
}

#[test]
fn test_extract_thought_from_step_payload_field20_field3() {
    let mut inner = Vec::new();
    push_len_field(&mut inner, 1, b"I will inspect the script.");
    push_len_field(
        &mut inner,
        3,
        b"**Analyzing Container Startup**\nI need to verify how the container reuse check works.",
    );

    let mut blob = Vec::new();
    push_varint_field(&mut blob, 1, 15);
    push_len_field(&mut blob, 20, &inner);

    assert_eq!(
        extract_thought_from_step_payload(&blob),
        Some(
            "**Analyzing Container Startup**\nI need to verify how the container reuse check works."
                .to_string()
        )
    );
}

#[test]
fn test_extract_user_text_from_step_payload_field19_field2() {
    let payload = make_user_payload("how are you?");
    assert_eq!(
        extract_user_text_from_step_payload(&payload),
        Some("how are you?".to_string())
    );
}

#[test]
fn test_extract_title_from_step_payload_field30_field4() {
    let payload = make_title_payload("Documenting Conversation Snapshot Function");
    assert_eq!(
        extract_title_from_step_payload(&payload),
        Some("Documenting Conversation Snapshot Function".to_string())
    );
}

#[test]
fn test_extract_title_ignores_empty_title() {
    assert_eq!(
        extract_title_from_step_payload(&make_title_payload("  ")),
        None
    );
}

#[test]
fn test_extract_text_multiline() {
    let text = b"Safe memory rules\nCompiler points out the flaws\nFast and fearless code";
    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(text.len() as u8);
    inner.extend_from_slice(text);

    let mut blob = vec![0x08, 0x01, 0xA2, 0x01, inner.len() as u8];
    blob.extend_from_slice(&inner);
    assert_eq!(
        extract_text_from_step_payload(&blob),
        Some(
            "Safe memory rules\nCompiler points out the flaws\nFast and fearless code".to_string()
        )
    );
}

#[test]
fn test_extract_tool_update_from_step_payload_json() {
    let payload = br#"
        grep_search
        {"Query":"prompt","SearchPath":"/tmp/project/src/main.rs","toolAction":"Finding prompt handling","toolSummary":"Grep prompt"}
    "#;

    let update = extract_tool_update_from_step_payload(19, 7, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "agy-19-7");
    assert_eq!(update["title"], "Grep prompt");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["status"], "completed");
    assert_eq!(update["rawInput"]["Query"], "prompt");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src/main.rs");
}

#[test]
/// Tests that when a tool payload lacks a JSON body (and thus has no `toolSummary`
/// or `toolAction`), the extractor falls back to using the extracted tool name
/// (e.g., `view_file`) as the update title.
fn test_extract_tool_update_uses_tool_name_fallback() {
    let payload = b"view_file";
    let update = extract_tool_update_from_step_payload(3, 8, payload).unwrap();
    assert_eq!(update["title"], "view_file");
    assert_eq!(update["kind"], "read");
}

#[test]
fn test_extract_tool_update_ignores_single_letter_noise() {
    let payload = b"P";
    assert_eq!(extract_tool_update_from_step_payload(4, 17, payload), None);
}

#[test]
fn test_extract_tool_update_ignores_generic_message_fallback() {
    let payload = b"Message";
    assert_eq!(extract_tool_update_from_step_payload(5, 17, payload), None);
}

#[test]
fn test_extract_tool_update_parses_first_balanced_json_object() {
    let payload = br#"
        abc123 view_file
        {"AbsolutePath":"/tmp/project/README.md","toolAction":"Reading README.md","toolSummary":"View README file"}
        trailing render blob {not json}
    "#;

    let update = extract_tool_update_from_step_payload(6, 8, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["title"], "View README file");
    assert_eq!(update["kind"], "read");
    assert_eq!(update["rawInput"]["AbsolutePath"], "/tmp/project/README.md");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/README.md");
}

#[test]
fn test_extract_tool_update_kind_prefers_tool_name_over_title() {
    let payload = br#"
        view_file
        {"AbsolutePath":"/tmp/project/flow_graph_write_node.go","toolSummary":"View flow_graph_write_node.go"}
    "#;

    let update = extract_tool_update_from_step_payload(7, 8, payload).unwrap();
    assert_eq!(update["title"], "View flow_graph_write_node.go");
    assert_eq!(update["kind"], "read");
}

#[test]
fn test_extract_tool_name_from_embedded_token() {
    assert_eq!(
        extract_tool_name("abc123\tview_file\n{...}"),
        Some("view_file".to_string())
    );
}

#[test]
fn test_extract_tool_update_from_pascal_case_edit_tool() {
    let payload = br#"
        Edit
        {"file_path":"/tmp/project/src/main.rs","old_string":"old","new_string":"new"}
    "#;

    let update = extract_tool_update_from_step_payload(9, 4, payload).unwrap();
    assert_eq!(update["title"], "Edit");
    assert_eq!(update["kind"], "edit");
    assert_eq!(update["rawInput"]["file_path"], "/tmp/project/src/main.rs");
}

#[test]
fn test_extract_tool_update_from_bash_tool() {
    let payload = br#"
        run_command
        {"CommandLine":"cargo test","Cwd":"/tmp/project","toolAction":"Running tests","toolSummary":"Run cargo test"}
    "#;

    let update = extract_tool_update_from_step_payload(10, 21, payload).unwrap();
    assert_eq!(update["title"], "Run cargo test");
    assert_eq!(update["kind"], "execute");
    assert_eq!(update["rawInput"]["CommandLine"], "cargo test");
}

#[test]
fn test_extract_tool_update_from_web_search_step() {
    let payload = br#"
        search_web
        {"query":"FIFA World Cup 2026 dates","toolAction":"Searching World Cup dates","toolSummary":"Search FIFA World Cup 2026 dates"}
    "#;

    assert!(is_tool_step_type(33));
    let update = extract_tool_update_from_step_payload(3, 33, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "agy-3-33");
    assert_eq!(update["title"], "Search FIFA World Cup 2026 dates");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["status"], "completed");
    assert_eq!(update["rawInput"]["query"], "FIFA World Cup 2026 dates");
}

#[test]
fn test_extract_tool_update_maps_reasoning_to_think_content() {
    let payload = br#"
        thinking
        {"thought":"Need to inspect the protocol before changing serialization.","toolSummary":"Reasoning"}
    "#;

    let update = extract_tool_update_from_step_payload(21, 17, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "agent_thought_chunk");
    assert_eq!(update["content"]["type"], "text");
    assert_eq!(
        update["content"]["text"],
        "Need to inspect the protocol before changing serialization."
    );
}

#[test]
fn test_extract_tool_update_from_structured_grep_payload() {
    let mut grep = Vec::new();
    push_len_field(&mut grep, 1, b"StepPayload");
    push_len_field(&mut grep, 2, b"src/*.rs");
    push_len_field(&mut grep, 3, b"src/protobuf.rs:1:message StepPayload");
    push_len_field(&mut grep, 10, b"rg StepPayload src");
    push_len_field(&mut grep, 11, b"file:///tmp/project");
    let payload = make_tool_payload(
        "0t0p5kn3",
        "grep_search",
        r#"{"SearchPath":"/tmp/project/src","toolAction":"Searching protobuf schema"}"#,
        "Proto search",
        Some((13, grep)),
    );

    let update = extract_tool_update_from_step_payload(22, 7, &payload).unwrap();
    assert_eq!(update["toolCallId"], "0t0p5kn3");
    assert_eq!(update["title"], "Proto search");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["rawInput"]["SearchPath"], "/tmp/project/src");
    assert_eq!(update["rawOutput"]["query"], "StepPayload");
    assert_eq!(
        update["rawOutput"]["textOutput"],
        "src/protobuf.rs:1:message StepPayload"
    );
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src");
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nsrc/protobuf.rs:1:message StepPayload\n```"
    );
}

#[test]
fn test_extract_tool_update_formats_structured_grep_hits_without_text_output() {
    let mut hit = Vec::new();
    push_len_field(&mut hit, 1, b"src/protobuf.rs");
    push_varint_field(&mut hit, 2, 42);
    push_len_field(
        &mut hit,
        3,
        b"fn parse_tool_result(blob: &[u8]) -> Option<Value> {",
    );

    let mut grep = Vec::new();
    push_len_field(&mut grep, 1, b"parse_tool_result");
    push_len_field(&mut grep, 4, &hit);
    let payload = make_tool_payload(
        "grep-hit-call",
        "grep_search",
        r#"{"SearchPath":"/tmp/project/src","toolAction":"Searching parser"}"#,
        "Parser search",
        Some((13, grep)),
    );

    let update = extract_tool_update_from_step_payload(26, 7, &payload).unwrap();
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nfield1: src/protobuf.rs | field2: 42 | field3: fn parse_tool_result(blob: &[u8]) -> Option<Value> {\n```"
    );
}

#[test]
fn test_extract_tool_update_from_structured_view_payload() {
    let mut view = Vec::new();
    push_len_field(&mut view, 1, b"file:///tmp/project/src/protobuf.rs");
    push_varint_field(&mut view, 2, 10);
    push_varint_field(&mut view, 3, 12);
    push_len_field(&mut view, 4, b"pub fn read_varint() {}\n```");
    push_varint_field(&mut view, 11, 13);
    push_varint_field(&mut view, 12, 200);
    let payload = make_tool_payload(
        "view-call",
        "view_file",
        "{}",
        "Viewing file",
        Some((14, view)),
    );

    let update = extract_tool_update_from_step_payload(23, 8, &payload).unwrap();
    assert_eq!(update["title"], "Viewing file");
    assert_eq!(update["kind"], "read");
    assert_eq!(
        update["rawOutput"]["fileUri"],
        "file:///tmp/project/src/protobuf.rs"
    );
    assert_eq!(update["rawOutput"]["startLine"], 10);
    assert_eq!(
        update["locations"][0]["path"],
        "file:///tmp/project/src/protobuf.rs"
    );
    assert_eq!(update["locations"][0]["line"], 10);
    assert_eq!(
        update["content"][0]["content"]["text"],
        "````\npub fn read_varint() {}\n```\n````"
    );
}

#[test]
fn test_extract_tool_update_from_structured_list_payload() {
    let mut entry = Vec::new();
    push_len_field(&mut entry, 1, b"src");
    push_varint_field(&mut entry, 2, 1);
    push_varint_field(&mut entry, 4, 0);

    let mut list = Vec::new();
    push_len_field(&mut list, 1, b"file:///tmp/project");
    push_len_field(&mut list, 3, &entry);
    let payload = make_tool_payload(
        "list-call",
        "list_dir",
        "{}",
        "Listing directory",
        Some((15, list)),
    );

    let update = extract_tool_update_from_step_payload(24, 9, &payload).unwrap();
    assert_eq!(update["title"], "Listing directory");
    assert_eq!(update["kind"], "read");
    assert_eq!(update["rawOutput"]["dirUri"], "file:///tmp/project");
    assert_eq!(update["rawOutput"]["entries"][0]["name"], "src");
    assert_eq!(update["rawOutput"]["entries"][0]["isDirectory"], true);
    assert_eq!(update["content"][0]["content"]["text"], "```\nsrc/\n```");
}

#[test]
fn test_extract_tool_update_formats_empty_structured_list_payload() {
    let mut list = Vec::new();
    push_len_field(&mut list, 1, b"file:///tmp/project");
    let payload = make_tool_payload(
        "empty-list-call",
        "list_dir",
        "{}",
        "Listing directory",
        Some((15, list)),
    );

    let update = extract_tool_update_from_step_payload(27, 9, &payload).unwrap();
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\n(empty directory)\n```"
    );
}

#[test]
fn test_extract_tool_update_from_structured_write_payload() {
    let mut write = Vec::new();
    push_len_field(&mut write, 26, b"Wrote 42 bytes");
    let payload = make_tool_payload(
        "write-call",
        "write_to_file",
        r#"{"AbsolutePath":"/tmp/project/src/main.rs"}"#,
        "Writing file",
        Some((10, write)),
    );

    let update = extract_tool_update_from_step_payload(25, 5, &payload).unwrap();
    assert_eq!(update["title"], "Writing file");
    assert_eq!(update["kind"], "edit");
    assert_eq!(update["rawOutput"]["summary"], "Wrote 42 bytes");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src/main.rs");
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nWrote 42 bytes\n```"
    );
}

#[test]
fn test_read_varint() {
    assert_eq!(read_varint(&[0x05]), Some((5, 1)));
    assert_eq!(read_varint(&[0xAC, 0x02]), Some((300, 2)));
    assert_eq!(read_varint(&[]), None);
}

#[test]
fn test_malformed_structured_grep_does_not_emit_partial_hits() {
    let mut hit = Vec::new();
    push_len_field(&mut hit, 1, b"src/protobuf.rs");

    let mut grep = Vec::new();
    push_len_field(&mut grep, 1, b"parse_tool_result");
    push_len_field(&mut grep, 4, &hit);
    // A second hit declares three bytes but only contains one. The malformed
    // payload must invalidate the whole repeated-field scan instead of
    // leaking the first, otherwise-valid hit.
    grep.extend_from_slice(&[0x22, 0x03, b'x']);

    let payload = make_tool_payload(
        "grep-malformed-call",
        "grep_search",
        r#"{"SearchPath":"/tmp/project/src","toolAction":"Searching parser"}"#,
        "Parser search",
        Some((13, grep)),
    );

    let update = extract_tool_update_from_step_payload(28, 7, &payload).unwrap();
    assert!(update["rawOutput"].get("hits").is_none());
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nNo matches\n```"
    );
}

#[test]
fn test_malformed_structured_list_does_not_emit_partial_entries() {
    let mut entry = Vec::new();
    push_len_field(&mut entry, 1, b"README.md");

    let mut list = Vec::new();
    push_len_field(&mut list, 1, b"file:///tmp/project");
    push_len_field(&mut list, 3, &entry);
    list.extend_from_slice(&[0x1A, 0x03, b'x']);

    let payload = make_tool_payload(
        "list-malformed-call",
        "list_directory",
        r#"{"dirUri":"file:///tmp/project"}"#,
        "List project",
        Some((15, list)),
    );

    let update = extract_tool_update_from_step_payload(29, 8, &payload).unwrap();
    assert!(update["rawOutput"].get("entries").is_none());
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\n(empty directory)\n```"
    );
}

#[test]
fn test_read_varint_rejects_u64_overflow() {
    assert_eq!(
        read_varint(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]),
        Some((u64::MAX, 10))
    );
    assert_eq!(
        read_varint(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02]),
        None
    );
    assert_eq!(
        read_varint(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]),
        None
    );
}

#[test]
fn test_proto_length_field_too_large_fails_closed() {
    let mut blob = vec![0xA2, 0x01];
    blob.extend([0xFF; 9]);
    blob.push(0x01);
    assert_eq!(extract_text_from_step_payload(&blob), None);
}

#[test]
fn test_proto_fixed_width_field_truncation_fails_closed() {
    assert_eq!(get_proto_field(&[0x09, 1, 2, 3], 1), None);
    assert_eq!(get_proto_field(&[0x0D, 1, 2, 3], 1), None);
}

#[test]
fn test_proto_nested_payload_truncation_fails_closed() {
    let blob = vec![0xA2, 0x01, 0x02, 0x0A, 0x01];
    assert_eq!(extract_text_from_step_payload(&blob), None);
}

#[test]
fn test_proto_trailing_truncation_invalidates_preceding_target() {
    let mut blob = make_assistant_payload("hello");
    blob.extend_from_slice(&[0x22, 0x03, b'x']);
    assert_eq!(extract_text_from_step_payload(&blob), None);
}

#[test]
fn test_proto_field_number_overflow_invalidates_preceding_target() {
    let mut blob = make_assistant_payload("hello");
    let invalid_tag = ((1u64 << 29) << 3) | 2;
    push_varint(&mut blob, invalid_tag);
    push_varint(&mut blob, 0);
    assert_eq!(extract_text_from_step_payload(&blob), None);
}

#[test]
fn test_initialize_advertises_load_session_support() {
    let adapter = Adapter::new();
    let response = adapter.handle_initialize(json!(1));
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("agentCapabilities"))
            .and_then(|c| c.get("loadSession"))
            .and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[test]
fn initialize_advertises_the_mcp_transports_agy_runs() {
    let adapter = Adapter::new();
    let response = adapter.handle_initialize(json!(1));
    assert_eq!(
        response.result.unwrap()["agentCapabilities"]["mcpCapabilities"],
        json!({"http": true, "sse": false})
    );
}

#[test]
fn test_initialize_advertises_resume_capability() {
    let adapter = Adapter::new();
    let response = adapter.handle_initialize(json!(1));
    assert!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("agentCapabilities"))
            .and_then(|c| c.get("sessionCapabilities"))
            .and_then(|sc| sc.get("resume"))
            .is_some(),
        "sessionCapabilities.resume should be present"
    );
}

#[test]
#[ignore]
fn test_session_load_restores_persisted_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-load-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = test_adapter(&root);
    adapter
        .persist_session(
            "sess-1",
            &crate::types::Session {
                conversation_id: Some(TEST_CONVERSATION_ABC.to_string()),
                last_step_idx: 5,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .expect("load fixture should persist");

    let output = adapter.handle_session_load(
        json!(7),
        &json!({"sessionId": "sess-1", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(
        adapter
            .sessions
            .get("sess-1")
            .and_then(|s| s.conversation_id.as_deref()),
        Some(TEST_CONVERSATION_ABC)
    );
    assert_eq!(
        adapter.sessions.get("sess-1").map(|s| s.last_step_idx),
        Some(5)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_load_rejects_unknown_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-missing-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = test_adapter(&root);

    let output = adapter.handle_session_load(
        json!(9),
        &json!({"sessionId": "missing", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response["result"].is_null());
    assert_eq!(
        response["error"]["message"].as_str(),
        Some("unknown sessionId: missing")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_load_replays_conversation_history() {
    let root = std::env::temp_dir().join(format!("agy-acp-load-replay-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let conversation_id = "00000000-0000-4000-8000-000000000022";
    let db_path = conv_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
        rusqlite::params![1i64, make_user_payload("hello")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![
            2i64,
            make_assistant_payload("I will inspect the workspace.")
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 8, ?2)",
        rusqlite::params![
            3i64,
            br#"view_file
            {"AbsolutePath":"/tmp/project/README.md","toolAction":"Reading README.md","toolSummary":"View README file"}
            trailing render blob {not json}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 5, ?2)",
        rusqlite::params![
            4i64,
            br#"replace_file_content
            {"AbsolutePath":"/tmp/project/README.md","toolAction":"Editing README.md","toolSummary":"Edit README file"}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 21, ?2)",
        rusqlite::params![
            5i64,
            br#"run_command
            {"CommandLine":"cargo test","Cwd":"/tmp/project","toolAction":"Running tests","toolSummary":"Run cargo test"}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 17, ?2)",
        rusqlite::params![6i64, br#"thinking
            {"thought":"I should verify the test output before summarizing.","toolSummary":"Analyzing test output"}"#
            .as_slice()],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![7i64, make_assistant_payload("hello from agent")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
        rusqlite::params![8i64, make_user_payload("how are you?")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![9i64, make_assistant_payload("second response")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 23, ?2)",
        rusqlite::params![10i64, make_title_payload("Replayed conversation")],
    )
    .unwrap();
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;
    adapter
        .persist_session(
            "sess-replay",
            &crate::types::Session {
                conversation_id: Some(conversation_id.to_string()),
                last_step_idx: 9,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .expect("replay fixture should persist");

    let output = adapter.handle_session_load(
        json!(1),
        &json!({"sessionId": "sess-replay", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );

    assert!(
        output.len() >= 2,
        "expected replay notification + response, got {}",
        output.len()
    );

    let updates: Vec<Value> = output[..output.len() - 1]
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(updates.iter().any(|notification| {
        notification["method"] == "session/update"
            && notification["params"]["update"]["sessionUpdate"] == "tool_call"
            && notification["params"]["update"]["title"] == "View README file"
            && notification["params"]["update"]["kind"] == "read"
    }));
    assert!(updates.iter().any(|notification| {
        notification["params"]["update"]["title"] == "Edit README file"
            && notification["params"]["update"]["kind"] == "edit"
    }));
    assert!(updates.iter().any(|notification| {
        notification["params"]["update"]["title"] == "Run cargo test"
            && notification["params"]["update"]["kind"] == "execute"
    }));
    assert!(updates.iter().any(|notification| {
        notification["params"]["update"]["sessionUpdate"] == "session_info_update"
            && notification["params"]["update"]["title"] == "Replayed conversation"
    }));
    let replay_kinds: Vec<_> = updates
        .iter()
        .map(|notification| {
            notification["params"]["update"]["sessionUpdate"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(
        replay_kinds,
        vec![
            "user_message_chunk",
            "agent_message_chunk",
            "tool_call",
            "tool_call",
            "tool_call",
            "agent_thought_chunk",
            "agent_message_chunk",
            "user_message_chunk",
            "agent_message_chunk",
            "session_info_update"
        ]
    );
    let message_updates: Vec<_> = updates
        .iter()
        .filter(|notification| {
            matches!(
                notification["params"]["update"]["sessionUpdate"].as_str(),
                Some("user_message_chunk")
                    | Some("agent_message_chunk")
                    | Some("agent_thought_chunk")
            )
        })
        .collect();
    let update_kinds: Vec<_> = message_updates
        .iter()
        .map(|notification| {
            notification["params"]["update"]["sessionUpdate"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(
        update_kinds,
        vec![
            "user_message_chunk",
            "agent_message_chunk",
            "agent_thought_chunk",
            "agent_message_chunk",
            "user_message_chunk",
            "agent_message_chunk"
        ]
    );
    let message_texts: Vec<_> = message_updates
        .iter()
        .map(|notification| {
            notification["params"]["update"]["content"]["text"]
                .as_str()
                .unwrap()
        })
        .collect();
    assert_eq!(
        message_texts,
        vec![
            "hello",
            "I will inspect the workspace.",
            "I should verify the test output before summarizing.",
            "hello from agent",
            "how are you?",
            "second response"
        ]
    );
    assert!(
        message_texts[1].contains("I will inspect"),
        "load replay should preserve narration shown in the live session"
    );

    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(
        response["result"]["sessionId"].as_str(),
        Some("sess-replay")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_resume_restores_persisted_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = test_adapter(&root);
    adapter
        .persist_session(
            "sess-r1",
            &crate::types::Session {
                conversation_id: Some(TEST_CONVERSATION_XYZ.to_string()),
                last_step_idx: 3,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .expect("resume fixture should persist");

    let response = adapter.handle_session_resume(
        json!(10),
        &json!({"sessionId": "sess-r1", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );
    assert!(response.error.is_none());
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("sessionId"))
            .and_then(|s| s.as_str()),
        Some("sess-r1")
    );
    assert_eq!(
        adapter
            .sessions
            .get("sess-r1")
            .and_then(|s| s.conversation_id.as_deref()),
        Some(TEST_CONVERSATION_XYZ)
    );
    assert_eq!(
        adapter.sessions.get("sess-r1").map(|s| s.last_step_idx),
        Some(3)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_resume_rejects_unknown_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-miss-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = test_adapter(&root);

    let response = adapter.handle_session_resume(
        json!(11),
        &json!({"sessionId": "nope", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );
    assert!(response.result.is_none());
    assert_eq!(
        response
            .error
            .as_ref()
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str()),
        Some("unknown sessionId: nope")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_session_resume_rejects_empty_session_id() {
    let mut adapter = Adapter::new();
    let response = adapter.handle_session_resume(json!(12), &json!({}));
    assert!(response.result.is_none());
    assert_eq!(
        response
            .error
            .as_ref()
            .and_then(|e| e.get("code"))
            .and_then(|c| c.as_i64()),
        Some(-32602)
    );
}

#[test]
fn test_session_resume_accepts_in_memory_session() {
    let root = fresh_test_root("resume-memory");
    let mut adapter = test_adapter(&root);
    adapter.sessions.insert(
        "sess-memory".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
            cwd: root.clone(),
        },
    );

    let response = adapter.handle_session_resume(
        json!(12),
        &json!({"sessionId": "sess-memory", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );

    assert!(response.error.is_none());
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("sessionId"))
            .and_then(|s| s.as_str()),
        Some("sess-memory")
    );
}

#[test]
fn test_session_load_accepts_in_memory_session_without_replay() {
    let root = fresh_test_root("load-memory");
    let mut adapter = test_adapter(&root);
    adapter.sessions.insert(
        "sess-memory-load".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
            cwd: root.clone(),
        },
    );

    let output = adapter.handle_session_load(
        json!(13),
        &json!({"sessionId": "sess-memory-load", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );

    assert_eq!(output.len(), 1);
    let response: Value = serde_json::from_str(&output[0]).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(response["result"]["sessionId"], "sess-memory-load");
}

#[test]
#[ignore]
fn test_session_resume_does_not_replay_history() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-noreplay-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = test_adapter(&root);
    adapter
        .persist_session(
            "sess-nr",
            &crate::types::Session {
                conversation_id: Some(TEST_CONVERSATION_NR.to_string()),
                last_step_idx: 10,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .expect("no-replay fixture should persist");

    let response = adapter.handle_session_resume(
        json!(13),
        &json!({"sessionId": "sess-nr", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );
    assert!(response.error.is_none());
    assert_eq!(
        response
            .result
            .as_ref()
            .and_then(|r| r.get("sessionId"))
            .and_then(|s| s.as_str()),
        Some("sess-nr")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_finds_conversation_from_invocation_specific_log() {
    let root = std::env::temp_dir().join(format!("agy-acp-log-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        b"I0722 printmode.go] conversationID=\"\"\nI0722 server.go] Created conversation 155c3eac-60e1-43c6-8f7c-eee98a34c2b8\n",
    )
    .unwrap();

    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&log_path),
        Some("155c3eac-60e1-43c6-8f7c-eee98a34c2b8".to_string())
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_rejects_invalid_conversation_id_in_invocation_log() {
    let root = std::env::temp_dir().join(format!("agy-acp-log-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        b"I0722 server.go] Created conversation not-a-uuid\n",
    )
    .unwrap();

    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&log_path),
        None
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_parallel_invocation_logs_cannot_cross_bind() {
    let root = std::env::temp_dir().join(format!("agy-acp-parallel-logs-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let first_log = root.join("first.log");
    let second_log = root.join("second.log");
    fs::write(
        &first_log,
        b"I0722 server.go] Created conversation 155c3eac-60e1-43c6-8f7c-eee98a34c2b8\n",
    )
    .unwrap();
    fs::write(
        &second_log,
        b"I0722 server.go] Created conversation 3f33f7fe-787a-489c-9fa9-477e6b4bf19b\n",
    )
    .unwrap();

    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&first_log).as_deref(),
        Some("155c3eac-60e1-43c6-8f7c-eee98a34c2b8")
    );
    assert_eq!(
        crate::db::find_created_conversation_id_in_log(&second_log).as_deref(),
        Some("3f33f7fe-787a-489c-9fa9-477e6b4bf19b")
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_detects_untrustworthy_print_timeout() {
    let root = std::env::temp_dir().join(format!("agy-acp-timeout-log-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let log_path = root.join("run.log");
    fs::write(
        &log_path,
        b"I0722 printmode.go] Print mode: timed out after 1496 polls (printed=162)\n",
    )
    .unwrap();

    assert!(crate::db::agy_run_timed_out(&log_path));
    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_persist_and_restore_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-state-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let adapter = test_adapter(&root);

    adapter
        .persist_session(
            "sess-1",
            &crate::types::Session {
                conversation_id: Some(TEST_CONVERSATION_ABC.to_string()),
                last_step_idx: 7,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .expect("session should persist");
    let restored = adapter
        .restore_session("sess-1")
        .expect("persisted state should be readable");
    assert_eq!(
        restored,
        Some(crate::types::StoredSession {
            conversation_id: Some(TEST_CONVERSATION_ABC.to_string()),
            last_step_idx: 7,
            model_id: None,
            cwd: Some(root.to_string_lossy().to_string()),
        })
    );

    let missing = adapter
        .restore_session("sess-unknown")
        .expect("missing-session lookup should read valid state");
    assert_eq!(missing, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_read_response_from_db() {
    let root = std::env::temp_dir().join(format!("agy-acp-sqlite-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let conversation_id = "00000000-0000-4000-8000-000000000010";
    let db_path = conv_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();

    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(11);
    inner.extend_from_slice(b"hello world");
    let mut payload = vec![0x08, 0x0F, 0xA2, 0x01, inner.len() as u8];
    payload.extend_from_slice(&inner);

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![1i64, payload],
    )
    .unwrap();

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
        rusqlite::params![2i64, vec![0x08u8, 0x0E]],
    )
    .unwrap();
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;

    let result = adapter.read_response_from_db(conversation_id, -1);
    assert_eq!(result, Some(("hello world".to_string(), 2)));

    let result = adapter.read_response_from_db(conversation_id, 1);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

fn prepare_auth() -> bool {
    if std::env::var("GEMINI_API_KEY")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        eprintln!("[e2e] Using GEMINI_API_KEY");
        return true;
    }
    if find_local_auth_root(configured_auth_roots()).is_some() {
        eprintln!("[e2e] Using local auth (keyring)");
        return true;
    }
    eprintln!("SKIP: No GEMINI_API_KEY and no local auth found");
    false
}

fn configured_auth_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for variable in ["HOME", "USERPROFILE"] {
        let Ok(value) = std::env::var(variable) else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let root = PathBuf::from(value);
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    roots
}

fn find_local_auth_root(roots: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    roots.into_iter().find(|root| {
        root.join(".gemini")
            .join("antigravity-cli")
            .join("settings.json")
            .is_file()
    })
}

fn release_binary_path(release_dir: &std::path::Path) -> PathBuf {
    release_dir.join(if cfg!(windows) {
        "agy-acp.exe"
    } else {
        "agy-acp"
    })
}

fn collect_agent_message_delta(response_text: &mut String, message: &Value) -> bool {
    let Some(update) = message.pointer("/params/update") else {
        return false;
    };
    if update["sessionUpdate"] != "agent_message_chunk" {
        return false;
    }
    let Some(text) = update["content"]["text"]
        .as_str()
        .filter(|text| !text.is_empty())
    else {
        return false;
    };

    response_text.push_str(text);
    true
}

#[test]
fn release_binary_path_uses_host_executable_suffix() {
    // Break caught: invoking the Unix release artifact name on Windows, where cargo emits agy-acp.exe.
    let release_dir = PathBuf::from("target").join("release");
    let expected = release_dir.join(if cfg!(windows) {
        "agy-acp.exe"
    } else {
        "agy-acp"
    });

    assert_eq!(release_binary_path(&release_dir), expected);
}

#[test]
fn local_auth_discovery_checks_a_later_environment_root() {
    // Break caught: checking HOME only and missing Windows local auth stored under USERPROFILE.
    let root = fresh_test_root("auth-roots");
    let home_root = root.join("home");
    let user_profile = root.join("user-profile");
    fs::create_dir_all(user_profile.join(".gemini").join("antigravity-cli")).unwrap();
    fs::write(
        user_profile
            .join(".gemini")
            .join("antigravity-cli")
            .join("settings.json"),
        "{}",
    )
    .unwrap();

    assert_eq!(
        find_local_auth_root(vec![home_root, user_profile.clone()]),
        Some(user_profile)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn e2e_message_collector_appends_only_assistant_message_deltas() {
    // Break caught: overwriting an assistant reply with a later thought or tool session update.
    let mut response_text = String::new();
    let updates = [
        json!({
            "method": "session/update",
            "params": {"update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "BAN"},
            }},
        }),
        json!({
            "method": "session/update",
            "params": {"update": {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "internal reasoning"},
            }},
        }),
        json!({
            "method": "session/update",
            "params": {"update": {
                "sessionUpdate": "tool_call",
                "title": "irrelevant tool update",
            }},
        }),
        json!({
            "method": "session/update",
            "params": {"update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "ANA"},
            }},
        }),
    ];

    for update in &updates {
        collect_agent_message_delta(&mut response_text, update);
    }

    assert_eq!(response_text, "BANANA");
}

#[test]
#[ignore]
fn test_e2e_agy_acp_full_round_trip() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    use std::time::Duration;

    if !prepare_auth() {
        return;
    }

    let agy_check = Command::new("agy").arg("--help").output();
    if agy_check.is_err() || !agy_check.unwrap().status.success() {
        eprintln!("SKIP: agy not found in PATH");
        return;
    }

    let binary = release_binary_path(&std::env::current_dir().unwrap().join("target/release"));
    if !binary.exists() {
        panic!("Run `cargo build --release` first");
    }

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn agy-acp");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    let mut send_and_recv = |msg: &str| -> String {
        writeln!(stdin, "{}", msg).unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        line
    };

    let resp = send_and_recv(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );
    let init: Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(init["result"]["protocolVersion"], 1);

    let session_new = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/new",
        "params": {
            "cwd": std::env::current_dir().unwrap().to_string_lossy(),
            "mcpServers": [],
        },
    })
    .to_string();
    let resp = send_and_recv(&session_new);
    let session: Value = serde_json::from_str(&resp).unwrap();
    let session_id = session["result"]["sessionId"].as_str().unwrap();
    assert!(!session_id.is_empty());

    let prompt_msg = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"Answer from this prompt only. Do not call any tools. Reply with exactly one word: PONG."}}]}}}}"#,
        session_id
    );
    writeln!(stdin, "{}", prompt_msg).unwrap();
    stdin.flush().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut got_message_update = false;
    let mut response_text = String::new();
    loop {
        if std::time::Instant::now() > deadline {
            panic!("Timed out waiting for agy-acp response");
        }
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.is_empty() {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let msg: Value = serde_json::from_str(line.trim()).unwrap();
        if msg.get("method") == Some(&json!("session/update")) {
            got_message_update |= collect_agent_message_delta(&mut response_text, &msg);
        }
        if msg.get("id") == Some(&json!(3)) {
            assert!(msg["error"].is_null(), "Got error: {}", msg["error"]);
            assert_eq!(msg["result"]["stopReason"], "end_turn");
            break;
        }
    }

    drop(stdin);
    let _ = child.wait();

    assert!(
        got_message_update,
        "Expected agent_message_chunk notification"
    );
    let lower = response_text.to_lowercase();
    assert!(
        lower.contains("pong"),
        "Expected 'PONG' in response, got: '{}'",
        response_text
    );
}

fn spawn_agy_acp() -> Option<(
    std::process::ChildStdin,
    std::io::BufReader<std::process::ChildStdout>,
    std::process::Child,
)> {
    use std::io::BufReader;
    use std::process::{Command, Stdio};

    if !prepare_auth() {
        return None;
    }
    let agy_check = Command::new("agy").arg("--help").output();
    if agy_check.is_err() || !agy_check.unwrap().status.success() {
        eprintln!("SKIP: agy not found in PATH");
        return None;
    }
    let binary = release_binary_path(&std::env::current_dir().unwrap().join("target/release"));
    if !binary.exists() {
        panic!("Run `cargo build --release` first");
    }

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn agy-acp");
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    Some((stdin, BufReader::new(stdout), child))
}

fn send_recv(
    stdin: &mut std::process::ChildStdin,
    reader: &mut std::io::BufReader<std::process::ChildStdout>,
    msg: &str,
) -> String {
    use std::io::{BufRead, Write};
    writeln!(stdin, "{}", msg).unwrap();
    stdin.flush().unwrap();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    line
}

fn send_prompt_wait(
    stdin: &mut std::process::ChildStdin,
    reader: &mut std::io::BufReader<std::process::ChildStdout>,
    id: u64,
    session_id: &str,
    text: &str,
) -> (Option<String>, Value) {
    use std::io::{BufRead, Write};
    use std::time::Duration;

    let msg = format!(
        r#"{{"jsonrpc":"2.0","id":{},"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"{}"}}]}}}}"#,
        id, session_id, text
    );
    writeln!(stdin, "{}", msg).unwrap();
    stdin.flush().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut notification_text = String::new();
    loop {
        if std::time::Instant::now() > deadline {
            panic!("Timed out");
        }
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.is_empty() {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let msg: Value = serde_json::from_str(line.trim()).unwrap();
        if msg.get("method") == Some(&json!("session/update")) {
            collect_agent_message_delta(&mut notification_text, &msg);
        }
        if msg.get("id") == Some(&json!(id)) {
            return (
                (!notification_text.is_empty()).then_some(notification_text),
                msg,
            );
        }
    }
}

#[test]
#[ignore]
fn test_e2e_multi_turn() {
    let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
        return;
    };

    send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );

    let session_new = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/new",
        "params": {
            "cwd": std::env::current_dir().unwrap().to_string_lossy(),
            "mcpServers": [],
        },
    })
    .to_string();
    let resp = send_recv(&mut stdin, &mut reader, &session_new);
    let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let (text1, resp1) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        3,
        &session_id,
        "Remember the word BANANA for this session. Answer from this prompt only, do not call any tools, and reply exactly: OK.",
    );
    assert!(resp1["error"].is_null(), "Turn 1 error: {}", resp1["error"]);
    assert!(text1.is_some());

    let (text2, resp2) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        4,
        &session_id,
        "Using only the existing conversation context, do not call any tools. What word did I ask you to remember? Reply with exactly that one word.",
    );
    assert!(resp2["error"].is_null(), "Turn 2 error: {}", resp2["error"]);
    let reply = text2.unwrap_or_default().to_lowercase();
    assert!(
        reply.contains("banana"),
        "Expected 'BANANA' in multi-turn reply, got: '{}'",
        reply
    );

    drop(stdin);
    let _ = child.wait();
}

#[test]
#[ignore]
fn test_e2e_session_load() {
    let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
        return;
    };

    send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );
    let session_new = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/new",
        "params": {
            "cwd": std::env::current_dir().unwrap().to_string_lossy(),
            "mcpServers": [],
        },
    })
    .to_string();
    let resp = send_recv(&mut stdin, &mut reader, &session_new);
    let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let (_text, resp1) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        3,
        &session_id,
        "Answer from this prompt only. Do not call any tools. Reply with exactly: FIRST_TURN",
    );
    assert!(
        resp1["error"].is_null(),
        "First turn error: {}",
        resp1["error"]
    );

    let (text2, resp2) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        4,
        &session_id,
        "Using only the existing conversation context, do not call any tools. Reply with exactly one word: SECOND",
    );
    assert!(
        resp2["error"].is_null(),
        "Second turn error: {}",
        resp2["error"]
    );
    assert!(text2.is_some(), "Expected response on continued session");

    drop(stdin);
    let _ = child.wait();
}

#[test]
#[ignore]
fn test_e2e_error_paths() {
    let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
        return;
    };

    send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );

    let session_load = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "session/load",
        "params": {
            "sessionId": "non-existent-session",
            "cwd": std::env::current_dir().unwrap().to_string_lossy(),
            "mcpServers": [],
        },
    })
    .to_string();
    let resp = send_recv(&mut stdin, &mut reader, &session_load);
    let val: Value = serde_json::from_str(&resp).unwrap();
    assert!(
        !val["error"].is_null(),
        "Expected error for unknown session"
    );

    let resp = send_recv(
        &mut stdin,
        &mut reader,
        r#"{"jsonrpc":"2.0","id":3,"method":"bogus/method","params":{}}"#,
    );
    let val: Value = serde_json::from_str(&resp).unwrap();
    assert!(!val["error"].is_null(), "Expected error for unknown method");

    drop(stdin);
    let _ = child.wait();
}

#[test]
#[ignore]
fn test_read_response_multi_step_no_skip_no_duplicate() {
    let root = std::env::temp_dir().join(format!("agy-acp-multi-step-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let conversation_id = "00000000-0000-4000-8000-000000000011";
    let db_path = conv_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 0,
            has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
            metadata BLOB,
            error_details BLOB,
            permissions BLOB,
            task_details BLOB,
            render_info BLOB,
            step_payload BLOB,
            step_format INTEGER NOT NULL DEFAULT 0
        )",
    )
    .unwrap();

    fn make_payload(text: &str) -> Vec<u8> {
        let text_bytes = text.as_bytes();
        let mut inner = vec![0x0A];
        let mut len = text_bytes.len();
        loop {
            if len < 128 {
                inner.push(len as u8);
                break;
            }
            inner.push((len as u8 & 0x7F) | 0x80);
            len >>= 7;
        }
        inner.extend_from_slice(text_bytes);

        let mut outer = vec![0xA2, 0x01];
        let mut ilen = inner.len();
        loop {
            if ilen < 128 {
                outer.push(ilen as u8);
                break;
            }
            outer.push((ilen as u8 & 0x7F) | 0x80);
            ilen >>= 7;
        }
        outer.extend(inner);
        outer
    }

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (1, 0, X'0801')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![2i64, make_payload("hello")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (3, 0, X'0802')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![4i64, make_payload("world")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![5i64, make_payload("line1\nline2\nline3")],
    )
    .unwrap();
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;

    let result = adapter.read_response_from_db(conversation_id, -1);
    assert_eq!(
        result,
        Some(("hello\nworld\nline1\nline2\nline3".to_string(), 5))
    );

    let result = adapter.read_response_from_db(conversation_id, 2);
    assert_eq!(result, Some(("world\nline1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db(conversation_id, 4);
    assert_eq!(result, Some(("line1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db(conversation_id, 5);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_read_response_missing_steps_table() {
    let root = std::env::temp_dir().join(format!("agy-acp-noschema-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let conversation_id = "00000000-0000-4000-8000-000000000012";
    let db_path = conv_dir.join(format!("{conversation_id}.db"));
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch("CREATE TABLE other (id INTEGER)")
        .unwrap();
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;

    let result = adapter.read_response_from_db(conversation_id, -1);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_read_rows_rejects_invalid_conversation_id_path_traversal() {
    let root = std::env::temp_dir().join(format!("agy-acp-path-guard-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    // If the raw ID were joined before validation, this database would be
    // reachable through `conversations/../escape.db`.
    let outside_path = root.join("escape.db");
    let conn = Connection::open(&outside_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE steps (
            idx INTEGER,
            step_type INTEGER,
            step_payload BLOB
        )",
    )
    .unwrap();
    drop(conn);

    assert!(crate::db::read_rows_from_db(&conv_dir, "../escape", -1).is_none());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_is_narration_true() {
    assert!(Adapter::is_narration("I will fetch the latest commits."));
    assert!(Adapter::is_narration("I'll fetch the latest commits."));
    assert!(Adapter::is_narration("I’ll fetch the latest commits."));
    assert!(Adapter::is_narration(
        "I will fetch the latest commits.\nI'll check the diff."
    ));
    assert!(Adapter::is_narration(
        "I will read the file.\n\nI will analyze the output."
    ));
}

#[test]
fn test_is_narration_false() {
    assert!(!Adapter::is_narration("Here is the result."));
    assert!(!Adapter::is_narration(
        "I will fetch the commits.\nHere is the result."
    ));
    assert!(!Adapter::is_narration(""));
}

#[test]
fn test_filter_narration_drops_all_narration() {
    let parts = vec![
        "I will fetch the latest commits.\nI will check the diff.".to_string(),
        "I will read the file.".to_string(),
        "The fix is confirmed! LGTM ✅".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result.as_deref(), Some("The fix is confirmed! LGTM ✅"));
}

#[test]
fn test_filter_narration_preserves_content_after_first_non_narration() {
    let parts = vec![
        "I will check things.".to_string(),
        "Here is my analysis.".to_string(),
        "I will also note this is fine.".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result.as_deref(), Some("Here is my analysis."));
}

#[test]
fn test_filter_narration_single_part_unchanged() {
    let parts = vec!["I will do something.".to_string()];
    let result = Adapter::filter_narration(&parts);
    assert_eq!(result, None);
}

#[test]
fn test_filter_narration_all_narration_drops_all() {
    let parts = vec![
        "I will fetch the file.".to_string(),
        "I'll check the output.".to_string(),
        "I will verify the fix.".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result, None);
}

#[test]
fn test_session_new_returns_models() {
    let root = fresh_test_root("new-models");
    let mut adapter = test_adapter(&root);
    let response = adapter.handle_session_new(json!(1), &session_setup_params(&root));
    let result = response.result.as_ref().unwrap();
    assert!(result.get("sessionId").is_some());
    let models = result.get("models").unwrap();
    assert!(models.get("currentModelId").is_some());
    assert!(models.get("availableModels").is_some());
    let config_options = result.get("configOptions").unwrap().as_array().unwrap();
    assert_eq!(config_options.len(), 1);
    assert_eq!(config_options[0]["id"].as_str(), Some("model"));
    assert_eq!(config_options[0]["category"].as_str(), Some("model"));
    assert_eq!(config_options[0]["type"].as_str(), Some("select"));
    assert!(config_options[0].get("currentValue").is_some());
    assert!(config_options[0].get("options").is_some());
}

#[test]
fn test_session_set_model() {
    let root = fresh_test_root("set-model");
    let mut adapter = test_adapter(&root);
    let new_resp = adapter.handle_session_new(json!(1), &session_setup_params(&root));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let set_resp = adapter.handle_session_set_model(
        json!(2),
        &json!({"sessionId": session_id, "modelId": "Gemini 3.5 Flash (High)"}),
    );
    assert!(set_resp.error.is_none());
    assert_eq!(
        adapter
            .sessions
            .get(&session_id)
            .unwrap()
            .model_id
            .as_deref(),
        Some("Gemini 3.5 Flash (High)")
    );
}

#[test]
fn test_session_set_model_missing_params() {
    let mut adapter = Adapter::new();
    let resp = adapter.handle_session_set_model(json!(1), &json!({}));
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32602));
}

#[test]
fn test_session_set_model_unknown_session() {
    let mut adapter = Adapter::new();
    let resp = adapter.handle_session_set_model(
        json!(1),
        &json!({"sessionId": "nonexistent", "modelId": "some-model"}),
    );
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32000));
}

#[test]
fn test_session_set_config_option_sets_model() {
    let root = fresh_test_root("set-config-model");
    let mut adapter = test_adapter(&root);
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let new_resp = adapter.handle_session_new(json!(1), &session_setup_params(&root));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let set_resp = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "model", "value": "Model B"}),
    );

    assert!(set_resp.error.is_none(), "error: {:?}", set_resp.error);
    assert_eq!(
        adapter
            .sessions
            .get(&session_id)
            .unwrap()
            .model_id
            .as_deref(),
        Some("Model B")
    );
    let config_options = set_resp.result.as_ref().unwrap()["configOptions"]
        .as_array()
        .unwrap();
    assert_eq!(config_options[0]["currentValue"].as_str(), Some("Model B"));
}

#[test]
fn test_session_set_config_option_rejects_unknown_config() {
    let root = fresh_test_root("reject-config");
    let mut adapter = test_adapter(&root);
    let new_resp = adapter.handle_session_new(json!(1), &session_setup_params(&root));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "not-model", "value": "Model B"}),
    );

    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32602));
}

#[test]
#[ignore]
fn test_session_set_model_persists() {
    let root = std::env::temp_dir().join(format!("agy-acp-model-persist-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = test_adapter(&root);

    adapter
        .persist_session(
            "sess-m1",
            &crate::types::Session {
                conversation_id: Some(TEST_CONVERSATION_M1.to_string()),
                last_step_idx: 0,
                model_id: None,
                cwd: root.clone(),
            },
        )
        .expect("model fixture should persist");

    assert!(adapter
        .restore_session_state("sess-m1", Some(root.clone()))
        .expect("model fixture should restore"));
    adapter.handle_session_set_model(
        json!(1),
        &json!({"sessionId": "sess-m1", "modelId": "Claude Opus 4.6 (Thinking)"}),
    );

    let adapter2 = test_adapter(&root);
    let restored = adapter2
        .restore_session("sess-m1")
        .expect("updated model state should be readable");
    assert_eq!(
        restored,
        Some(crate::types::StoredSession {
            conversation_id: Some(TEST_CONVERSATION_M1.to_string()),
            last_step_idx: 0,
            model_id: Some("Claude Opus 4.6 (Thinking)".to_string()),
            cwd: Some(root.to_string_lossy().to_string()),
        })
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_session_load_returns_models() {
    let root = fresh_test_root("load-models");
    let mut adapter = test_adapter(&root);
    adapter.sessions.insert(
        "test-load".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: Some("Gemini 3.1 Pro (High)".to_string()),
            cwd: root.clone(),
        },
    );
    adapter
        .persist_session(
            "test-load",
            &crate::types::Session {
                conversation_id: Some(TEST_CONVERSATION_LOAD.to_string()),
                last_step_idx: -1,
                model_id: Some("Gemini 3.1 Pro (High)".to_string()),
                cwd: root.clone(),
            },
        )
        .expect("load model fixture should persist");
    adapter.sessions.clear();

    let output = adapter.handle_session_load(
        json!(1),
        &json!({"sessionId": "test-load", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );
    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(
        response["error"].is_null(),
        "error: {:?}",
        response["error"]
    );
    let models = response["result"]["models"].as_object().unwrap();
    assert_eq!(
        models["currentModelId"].as_str(),
        Some("Gemini 3.1 Pro (High)")
    );
    assert_eq!(
        response["result"]["configOptions"][0]["currentValue"].as_str(),
        Some("Gemini 3.1 Pro (High)")
    );
}

#[test]
fn test_session_resume_returns_models() {
    let root = fresh_test_root("resume-models");
    let mut adapter = test_adapter(&root);
    adapter
        .persist_session(
            "test-resume",
            &crate::types::Session {
                conversation_id: Some(TEST_CONVERSATION_RESUME.to_string()),
                last_step_idx: -1,
                model_id: Some("GPT-OSS 120B (Medium)".to_string()),
                cwd: root.clone(),
            },
        )
        .expect("resume model fixture should persist");
    adapter.sessions.clear();

    let response = adapter.handle_session_resume(
        json!(1),
        &json!({"sessionId": "test-resume", "cwd": root.to_string_lossy(), "mcpServers": []}),
    );
    assert!(response.error.is_none(), "error: {:?}", response.error);
    let models = response.result.as_ref().unwrap()["models"]
        .as_object()
        .unwrap();
    assert_eq!(
        models["currentModelId"].as_str(),
        Some("GPT-OSS 120B (Medium)")
    );
    assert_eq!(
        response.result.as_ref().unwrap()["configOptions"][0]["currentValue"].as_str(),
        Some("GPT-OSS 120B (Medium)")
    );
}

#[test]
fn test_session_models_json_default() {
    let adapter = Adapter::new();
    let models = adapter.session_models_json(None);
    let current = models["currentModelId"].as_str().unwrap();
    if adapter.available_models.is_empty() {
        assert_eq!(current, "");
    } else {
        assert!(!current.is_empty());
        assert!(!current.contains('\t'));
    }
}

#[test]
fn model_responses_read_the_initialized_cache_without_mutating_adapter() {
    // Break caught: lifecycle rendering lazily spawning `agy models` under the adapter lock.
    let root = fresh_test_root("models-cache-only");
    let mut initializing_adapter = test_adapter(&root);
    initializing_adapter.available_models.clear();
    let adapter = initializing_adapter;

    let models = adapter.session_models_json(None);
    let config_options = adapter.session_config_options_json(None);

    assert_eq!(models["currentModelId"], "");
    assert_eq!(models["availableModels"], json!([]));
    assert_eq!(config_options[0]["currentValue"], "");
    assert_eq!(config_options[0]["options"], json!([]));
}

#[test]
fn test_session_models_json_with_model() {
    let mut adapter = Adapter::new();
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let models = adapter.session_models_json(Some("Model B"));
    assert_eq!(models["currentModelId"].as_str(), Some("Model B"));
    let available = models["availableModels"].as_array().unwrap();
    assert_eq!(available.len(), 2);
    assert_eq!(available[0]["modelId"].as_str(), Some("Model A"));
    assert_eq!(available[1]["modelId"].as_str(), Some("Model B"));
}

#[test]
fn test_session_models_json_splits_agy_model_id_and_label() {
    let root = fresh_test_root("model-label");
    let mut adapter = test_adapter(&root);
    adapter.available_models = vec!["gemini-3.8-flash-high\tGemini 3.8 Flash (High)".to_string()];

    let models = adapter.session_models_json(None);

    assert_eq!(
        models["currentModelId"].as_str(),
        Some("gemini-3.8-flash-high")
    );
    assert_eq!(
        models["availableModels"][0]["modelId"].as_str(),
        Some("gemini-3.8-flash-high")
    );
    assert_eq!(
        models["availableModels"][0]["name"].as_str(),
        Some("Gemini 3.8 Flash (High)")
    );
}

#[test]
fn test_session_config_options_json_with_model() {
    let mut adapter = Adapter::new();
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let config_options = adapter.session_config_options_json(Some("Model B"));
    assert_eq!(config_options[0]["id"].as_str(), Some("model"));
    assert_eq!(config_options[0]["category"].as_str(), Some("model"));
    assert_eq!(config_options[0]["type"].as_str(), Some("select"));
    assert_eq!(config_options[0]["currentValue"].as_str(), Some("Model B"));
    let options = config_options[0]["options"].as_array().unwrap();
    assert_eq!(options.len(), 2);
    assert_eq!(options[0]["value"].as_str(), Some("Model A"));
    assert_eq!(options[1]["value"].as_str(), Some("Model B"));
}

#[test]
fn test_session_config_options_split_agy_model_id_and_label() {
    let root = fresh_test_root("config-label");
    let mut adapter = test_adapter(&root);
    adapter.available_models = vec!["gemini-3.8-flash-high\tGemini 3.8 Flash (High)".to_string()];

    let config_options = adapter.session_config_options_json(None);

    assert_eq!(
        config_options[0]["currentValue"].as_str(),
        Some("gemini-3.8-flash-high")
    );
    assert_eq!(
        config_options[0]["options"][0]["value"].as_str(),
        Some("gemini-3.8-flash-high")
    );
    assert_eq!(
        config_options[0]["options"][0]["name"].as_str(),
        Some("Gemini 3.8 Flash (High)")
    );
}
