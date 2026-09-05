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
    extract_user_text_from_step_payload, is_tool_step_type, read_varint,
};
use crate::protocol::{parse_jsonrpc_line, IncomingMessage};
use crate::types::{CommandSpec, JsonRpcResponse, SessionStore, StoredSession};
use crate::Cli;
use clap::Parser;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::AsyncWrite;

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
        conversation_id: Some("conversation-after-error".to_string()),
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
        Some("conversation-after-error")
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
        conversation_id: Some("late-conversation".to_string()),
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
        conversation_id: Some("new-conversation".to_string()),
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
        conversation_id: Some("late-conversation".to_string()),
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
    assert_eq!(after.conversation_id.as_deref(), Some("late-conversation"));
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
        conversation_id: Some("conversation-a".to_string()),
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
        conversation_id: Some("conversation-b".to_string()),
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
        conversation_id: Some("conversation-a".to_string()),
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
        conversation_id: Some("conversation-b".to_string()),
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
        conversation_id: Some("conversation-a".to_string()),
        last_step_idx: 30,
        model_id: Some("persisted-model".to_string()),
        cwd: root.clone(),
    };
    adapter
        .persist_session(session_id, &persisted)
        .expect("persisted conflict fixture should persist");
    let resident = crate::types::Session {
        conversation_id: Some("conversation-b".to_string()),
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
        conversation_id: Some("conversation-b".to_string()),
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
        conversation_id: Some("conversation-b".to_string()),
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
        conversation_id: Some("conversation-b".to_string()),
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
        conversation_id: Some("conversation-b".to_string()),
        last_step_idx: 20,
        model_id: None,
        cwd: root.clone(),
    };
    adapter
        .persist_session(evicted_id, &evicted)
        .expect("evicted monotonic fixture should persist");
    let evicted_outcome = crate::runtime::PromptOutcome {
        session_id: evicted_id.to_string(),
        conversation_id: Some("conversation-b".to_string()),
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
        conversation_id: Some("conversation-a".to_string()),
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
        conversation_id: Some("conversation-a".to_string()),
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
    assert_eq!(stored.conversation_id.as_deref(), Some("conversation-a"));
    assert_eq!(stored.last_step_idx, 50);
}

fn fresh_test_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("agy-acp-{label}-{}", Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    root
}

#[cfg(windows)]
const WINDOWS_FAKE_AGY_SCRIPT: &str = r#"
$logFile = $null
$prompt = $null
for ($i = 0; $i -lt $args.Count; $i++) {
    if (($null -eq $logFile) -and ($args[$i] -eq '--log-file') -and (($i + 1) -lt $args.Count)) {
        $logFile = $args[$i + 1]
    }
    if (($args[$i] -eq '-p') -and (($i + 1) -lt $args.Count)) {
        $prompt = $args[$i + 1]
    }
}
Set-Content -LiteralPath $logFile -Value 'Created conversation 00000000-0000-4000-8000-000000000099' -Encoding utf8
if ($prompt -eq 'slow') { Start-Sleep -Milliseconds 3000 }
if ($prompt -eq 'stderr-fail') {
    [Console]::Error.WriteLine('AGY_STDERR_SECRET_SENTINEL')
    exit 7
}
if ($prompt -eq 'fail') { exit 7 }
[Console]::Out.WriteLine('fake assistant response')
"#;

#[cfg(not(windows))]
const UNIX_FAKE_AGY_SCRIPT: &str = r#"#!/bin/sh
log_file=''
prompt=''
while [ "$#" -gt 0 ]; do
  case "$1" in
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
if [ "$prompt" = 'slow' ]; then
  printf '%s\n' 'fake assistant response'
  exec sleep 3
fi
if [ "$prompt" = 'stderr-fail' ]; then
  printf '%s\n' 'AGY_STDERR_SECRET_SENTINEL' >&2
  exit 7
fi
if [ "$prompt" = 'fail' ]; then exit 7; fi
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

async fn wait_for_fake_run_log(root: &std::path::Path) {
    let run_logs = root.join("state").join("run-logs");
    tokio::time::timeout(Duration::from_secs(2), async {
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
        Duration::from_secs(2),
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
    let outcome = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .expect("cancelled fake child did not finish promptly")
        .unwrap();

    assert_eq!(outcome.response.result.unwrap()["stopReason"], "cancelled");
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
    let mut protocol_output = String::new();
    while let Ok(line) = harness.receiver.try_recv() {
        protocol_output.push_str(&line);
    }

    assert_eq!(error["code"], -32000);
    assert!(message.contains("status"));
    assert!(message.contains("run log retained at"));
    assert!(message.contains(&run_log_path.to_string_lossy().to_string()));
    assert!(message.len() <= 512);
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
    let replay_db = adapter.conversations_dir.join("conversation-load.db");
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
            conversation_id: Some("conversation-load".to_string()),
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
            conversation_id: Some("conversation-model".to_string()),
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

#[test]
fn session_new_rejects_non_empty_mcp_servers() {
    // Break caught: silently claiming support for MCP servers that are never forwarded.
    let root = fresh_test_root("mcp");
    let cwd = root.join("workspace");
    fs::create_dir_all(&cwd).unwrap();
    let mut adapter = test_adapter(&root);
    let response = adapter.handle_session_new(
        json!(1),
        &json!({"cwd": cwd.to_string_lossy(), "mcpServers": [{"type": "stdio"}]}),
    );
    assert_eq!(response.error.as_ref().unwrap()["code"], -32602);
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
                conversation_id: Some("conv-abc".to_string()),
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
        Some("conv-abc")
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

    let db_path = conv_dir.join("conv-replay.db");
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
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;
    adapter
        .persist_session(
            "sess-replay",
            &crate::types::Session {
                conversation_id: Some("conv-replay".to_string()),
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
            "agent_message_chunk"
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
                conversation_id: Some("conv-xyz".to_string()),
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
        Some("conv-xyz")
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
                conversation_id: Some("conv-nr".to_string()),
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
                conversation_id: Some("conv-abc".to_string()),
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
            conversation_id: Some("conv-abc".to_string()),
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

    let db_path = conv_dir.join("test-conv.db");
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

    let result = adapter.read_response_from_db("test-conv", -1);
    assert_eq!(result, Some(("hello world".to_string(), 2)));

    let result = adapter.read_response_from_db("test-conv", 1);
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
    let home = std::env::var("HOME").unwrap_or_default();
    let settings = format!("{}/.gemini/antigravity-cli/settings.json", home);
    if std::path::Path::new(&settings).exists() {
        eprintln!("[e2e] Using local auth (keyring)");
        return true;
    }
    eprintln!("SKIP: No GEMINI_API_KEY and no local auth found");
    false
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

    let binary = std::env::current_dir()
        .unwrap()
        .join("target/release/agy-acp");
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
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"Reply with exactly one word: PONG"}}]}}}}"#,
        session_id
    );
    writeln!(stdin, "{}", prompt_msg).unwrap();
    stdin.flush().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    let mut got_notification = false;
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
            got_notification = true;
            response_text = msg["params"]["update"]["content"]["text"]
                .as_str()
                .unwrap_or("")
                .to_string();
        }
        if msg.get("id") == Some(&json!(3)) {
            assert!(msg["error"].is_null(), "Got error: {}", msg["error"]);
            assert_eq!(msg["result"]["stopReason"], "end_turn");
            break;
        }
    }

    drop(stdin);
    let _ = child.wait();

    assert!(got_notification, "Expected session/update notification");
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
    let binary = std::env::current_dir()
        .unwrap()
        .join("target/release/agy-acp");
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
    let mut notification_text: Option<String> = None;
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
            notification_text = msg["params"]["update"]["content"]["text"]
                .as_str()
                .map(String::from);
        }
        if msg.get("id") == Some(&json!(id)) {
            return (notification_text, msg);
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
        "Remember this word: BANANA. Reply OK.",
    );
    assert!(resp1["error"].is_null(), "Turn 1 error: {}", resp1["error"]);
    assert!(text1.is_some());

    let (text2, resp2) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        4,
        &session_id,
        "What word did I ask you to remember? Reply with just that word.",
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
        "Reply with exactly: FIRST_TURN",
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
        "Reply with exactly one word: SECOND",
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

    let db_path = conv_dir.join("multi.db");
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

    let result = adapter.read_response_from_db("multi", -1);
    assert_eq!(
        result,
        Some(("hello\nworld\nline1\nline2\nline3".to_string(), 5))
    );

    let result = adapter.read_response_from_db("multi", 2);
    assert_eq!(result, Some(("world\nline1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db("multi", 4);
    assert_eq!(result, Some(("line1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db("multi", 5);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_read_response_missing_steps_table() {
    let root = std::env::temp_dir().join(format!("agy-acp-noschema-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let db_path = conv_dir.join("empty.db");
    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch("CREATE TABLE other (id INTEGER)")
        .unwrap();
    drop(conn);

    let mut adapter = test_adapter(&root);
    adapter.conversations_dir = conv_dir;

    let result = adapter.read_response_from_db("empty", -1);
    assert_eq!(result, None);

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
                conversation_id: Some("conv-m1".to_string()),
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
            conversation_id: Some("conv-m1".to_string()),
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
                conversation_id: Some("conv-load".to_string()),
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
                conversation_id: Some("conv-resume".to_string()),
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
