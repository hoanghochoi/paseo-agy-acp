use rusqlite::Connection;
use serde_json::Value;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use uuid::Uuid;

use crate::adapter::filter_narration;
use crate::protobuf::{
    extract_text_from_step_payload, extract_thought_from_step_payload,
    extract_title_from_step_payload, extract_tool_update_from_step_payload,
    extract_user_text_from_step_payload, is_tool_step_type, message_chunk_update,
};

/// Bound one SQLite read so a long-running turn cannot materialize an
/// unbounded tail of history in one poll.
pub(crate) const MAX_DB_ROWS_PER_READ: usize = 256;
/// Bound the total payload materialized by one SQLite page in addition to the
/// row count bound. A single page must not turn 256 individually-valid rows
/// into an unbounded aggregate allocation.
pub(crate) const MAX_DB_BYTES_PER_READ: usize = 4 * 1024 * 1024;
/// Oversized step payloads are represented as empty payloads and still advance
/// the cursor, preventing repeated allocation and retry of malformed data.
pub(crate) const MAX_STEP_PAYLOAD_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_INVOCATION_LOG_BYTES: usize = 256 * 1024;
pub(crate) const MAX_REPLAY_INPUT_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const MAX_REPLAY_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayReadError {
    Unavailable,
    BudgetExceeded,
}

fn read_log_prefix(path: &Path) -> Option<Vec<u8>> {
    let file = fs::File::open(path).ok()?;
    let mut output = Vec::new();
    file.take(MAX_INVOCATION_LOG_BYTES as u64)
        .read_to_end(&mut output)
        .ok()?;
    Some(output)
}

fn read_log_tail(path: &Path) -> Option<Vec<u8>> {
    let mut file = fs::File::open(path).ok()?;
    let start = file
        .metadata()
        .ok()?
        .len()
        .saturating_sub(MAX_INVOCATION_LOG_BYTES as u64);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut output = Vec::new();
    file.read_to_end(&mut output).ok()?;
    Some(output)
}

#[cfg(test)]
use crate::types::ConversationDelta;

/// Read the conversation ID created by one specific `agy` invocation.
///
/// The caller gives every invocation its own `--log-file`, so unlike a global
/// directory diff this remains deterministic when several adapters start at
/// the same time.
pub fn find_created_conversation_id_in_log(log_path: &Path) -> Option<String> {
    const MARKER: &str = "Created conversation ";
    let contents = String::from_utf8_lossy(&read_log_prefix(log_path)?).into_owned();
    contents.lines().find_map(|line| {
        let candidate = line.split_once(MARKER)?.1.split_whitespace().next()?;
        Uuid::parse_str(candidate).ok().map(|id| id.to_string())
    })
}

pub fn agy_run_timed_out(log_path: &Path) -> bool {
    String::from_utf8_lossy(&read_log_tail(log_path).unwrap_or_default())
        .contains("Print mode: timed out")
}

pub fn find_conversation_id_by_pid(_pid: u32, _conversations_dir: &Path) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let fd_dir = format!("/proc/{}/fd", _pid);
        if let Ok(entries) = std::fs::read_dir(fd_dir) {
            let conversations_dir_canonical = _conversations_dir
                .canonicalize()
                .unwrap_or_else(|_| _conversations_dir.to_path_buf());
            for entry in entries.filter_map(|e| e.ok()) {
                if let Ok(target) = std::fs::read_link(entry.path()) {
                    if target.extension().map(|ext| ext == "db").unwrap_or(false) {
                        if let Some(parent) = target.parent() {
                            let target_parent = parent
                                .canonicalize()
                                .unwrap_or_else(|_| parent.to_path_buf());
                            if target_parent == conversations_dir_canonical {
                                if let Some(stem) = target.file_stem() {
                                    return Some(stem.to_string_lossy().to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("lsof")
            .args(&["-p", &_pid.to_string(), "-Fn"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let conversations_dir_canonical = _conversations_dir
                    .canonicalize()
                    .unwrap_or_else(|_| _conversations_dir.to_path_buf());
                for line in stdout.lines() {
                    if line.starts_with('n') {
                        let path_str = &line[1..];
                        let path = Path::new(path_str);
                        if path.extension().map(|ext| ext == "db").unwrap_or(false) {
                            if let Some(parent) = path.parent() {
                                let target_parent = parent
                                    .canonicalize()
                                    .unwrap_or_else(|_| parent.to_path_buf());
                                if target_parent == conversations_dir_canonical {
                                    if let Some(stem) = path.file_stem() {
                                        return Some(stem.to_string_lossy().to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

pub fn read_rows_from_db(
    conversations_dir: &Path,
    conversation_id: &str,
    after_step_idx: i64,
) -> Option<Vec<(i64, i64, Vec<u8>)>> {
    // Conversation IDs originate from agy logs/state, but they are still
    // untrusted path components. Parse and normalize the UUID before joining
    // it to the conversations directory so traversal or absolute paths can
    // never escape the intended database root.
    let conversation_uuid = Uuid::parse_str(conversation_id).ok()?;
    let db_path = conversations_dir.join(format!("{conversation_uuid}.db"));
    let conn = Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;

    let table_exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='steps'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(false);
    if !table_exists {
        eprintln!(
            "[agy-acp] WARN: steps table not found in {}.db — schema changed?",
            conversation_id
        );
        return None;
    }

    let mut stmt = conn
        .prepare(
            "SELECT idx, step_type,
                    CASE WHEN step_payload IS NULL OR length(step_payload) > ?2
                         THEN NULL ELSE step_payload END
             FROM steps WHERE idx > ?1 ORDER BY idx LIMIT ?3",
        )
        .ok()?;
    let mut rows = stmt
        .query(rusqlite::params![
            after_step_idx,
            MAX_STEP_PAYLOAD_BYTES as i64,
            MAX_DB_ROWS_PER_READ as i64
        ])
        .ok()?;
    let mut collected = Vec::new();
    let mut payload_bytes = 0usize;
    while let Some(row) = rows.next().ok()? {
        let idx = row.get(0).ok()?;
        let step_type = row.get(1).ok()?;
        let payload = row.get::<_, Option<Vec<u8>>>(2).ok()?.unwrap_or_default();
        let next_payload_bytes = payload_bytes.saturating_add(payload.len());
        if !collected.is_empty() && next_payload_bytes > MAX_DB_BYTES_PER_READ {
            break;
        }
        payload_bytes = next_payload_bytes;
        collected.push((idx, step_type, payload));
        if collected.len() >= MAX_DB_ROWS_PER_READ {
            break;
        }
    }
    let rows = collected;
    Some(rows)
}

pub fn read_replay_updates_from_db(
    conversations_dir: &Path,
    conversation_id: &str,
    skip_naration: bool,
) -> Result<Option<(Vec<Value>, i64)>, ReplayReadError> {
    let mut max_idx = -1;
    let mut updates = Vec::new();
    let mut pending_agent_parts = Vec::new();
    let mut pending_thought_parts = Vec::new();
    let mut after_step_idx = -1;
    let mut input_bytes = 0usize;
    let mut output_bytes = 0usize;
    let mut saw_rows = false;

    loop {
        let rows = read_rows_from_db(conversations_dir, conversation_id, after_step_idx)
            .ok_or(ReplayReadError::Unavailable)?;
        if rows.is_empty() {
            break;
        }
        saw_rows = true;

        for (idx, step_type, payload) in &rows {
            input_bytes = input_bytes.saturating_add(payload.len());
            if input_bytes > MAX_REPLAY_INPUT_BYTES {
                return Err(ReplayReadError::BudgetExceeded);
            }
            max_idx = max_idx.max(*idx);
            if *step_type == 14 {
                flush_agent_message(
                    &mut pending_agent_parts,
                    &mut updates,
                    &mut output_bytes,
                    skip_naration,
                )?;
                flush_thought_message(&mut pending_thought_parts, &mut updates, &mut output_bytes)?;
                if let Some(text) = extract_user_text_from_step_payload(payload) {
                    push_replay_update(
                        &mut updates,
                        &mut output_bytes,
                        message_chunk_update("user_message_chunk", text),
                    )?;
                }
            } else if *step_type == 15 {
                if let Some(text) = extract_text_from_step_payload(payload) {
                    if !text.is_empty() {
                        pending_agent_parts.push(text);
                    }
                }
                if let Some(text) = extract_thought_from_step_payload(payload) {
                    pending_thought_parts.push(text);
                }
            } else if is_tool_step_type(*step_type) {
                flush_agent_message(
                    &mut pending_agent_parts,
                    &mut updates,
                    &mut output_bytes,
                    skip_naration,
                )?;
                flush_thought_message(&mut pending_thought_parts, &mut updates, &mut output_bytes)?;
                if let Some(update) =
                    extract_tool_update_from_step_payload(*idx, *step_type, payload)
                {
                    push_replay_update(&mut updates, &mut output_bytes, update)?;
                }
            } else if *step_type == 23 {
                flush_agent_message(
                    &mut pending_agent_parts,
                    &mut updates,
                    &mut output_bytes,
                    skip_naration,
                )?;
                flush_thought_message(&mut pending_thought_parts, &mut updates, &mut output_bytes)?;
                if let Some(title) = extract_title_from_step_payload(payload) {
                    push_replay_update(
                        &mut updates,
                        &mut output_bytes,
                        serde_json::json!({
                            "sessionUpdate": "session_info_update",
                            "title": title,
                        }),
                    )?;
                }
            }
        }

        let page_last_idx = rows
            .last()
            .map(|(idx, _, _)| *idx)
            .unwrap_or(after_step_idx);
        if page_last_idx <= after_step_idx {
            return Err(ReplayReadError::Unavailable);
        }
        after_step_idx = page_last_idx;
    }

    flush_agent_message(
        &mut pending_agent_parts,
        &mut updates,
        &mut output_bytes,
        skip_naration,
    )?;
    flush_thought_message(&mut pending_thought_parts, &mut updates, &mut output_bytes)?;

    if !saw_rows {
        Ok(None)
    } else {
        Ok(Some((updates, max_idx)))
    }
}

fn push_replay_update(
    updates: &mut Vec<Value>,
    output_bytes: &mut usize,
    update: Value,
) -> Result<(), ReplayReadError> {
    let update_bytes = serde_json::to_vec(&update)
        .map_err(|_| ReplayReadError::Unavailable)?
        .len();
    let next_output_bytes = output_bytes.saturating_add(update_bytes);
    if next_output_bytes > MAX_REPLAY_OUTPUT_BYTES {
        return Err(ReplayReadError::BudgetExceeded);
    }
    *output_bytes = next_output_bytes;
    updates.push(update);
    Ok(())
}

fn flush_agent_message(
    parts: &mut Vec<String>,
    updates: &mut Vec<Value>,
    output_bytes: &mut usize,
    skip_naration: bool,
) -> Result<(), ReplayReadError> {
    if parts.is_empty() {
        return Ok(());
    }
    let text = if skip_naration {
        filter_narration(parts)
    } else {
        Some(parts.join("\n"))
    };
    parts.clear();
    if let Some(text) = text {
        if !text.is_empty() {
            push_replay_update(
                updates,
                output_bytes,
                message_chunk_update("agent_message_chunk", text),
            )?;
        }
    }
    Ok(())
}

fn flush_thought_message(
    parts: &mut Vec<String>,
    updates: &mut Vec<Value>,
    output_bytes: &mut usize,
) -> Result<(), ReplayReadError> {
    if parts.is_empty() {
        return Ok(());
    }
    let text = parts.join("\n");
    parts.clear();
    if !text.is_empty() {
        push_replay_update(
            updates,
            output_bytes,
            message_chunk_update("agent_thought_chunk", text),
        )?;
    }
    Ok(())
}

#[cfg(test)]
pub fn read_delta_from_db(
    conversations_dir: &Path,
    conversation_id: &str,
    after_step_idx: i64,
) -> Option<ConversationDelta> {
    let rows = read_rows_from_db(conversations_dir, conversation_id, after_step_idx)?;

    let mut max_idx = after_step_idx;
    let mut response_parts: Vec<String> = Vec::new();
    for (idx, step_type, payload) in &rows {
        max_idx = max_idx.max(*idx);
        if *step_type == 15 {
            if let Some(text) = extract_text_from_step_payload(payload) {
                if !text.is_empty() {
                    response_parts.push(text);
                }
            }
        }
    }
    if response_parts.is_empty() {
        let response_rows: Vec<_> = rows
            .iter()
            .filter(|(_, step_type, _)| *step_type == 15)
            .collect();
        if !response_rows.is_empty() {
            let payload_sizes: Vec<usize> = response_rows.iter().map(|(_, _, p)| p.len()).collect();
            eprintln!(
                "[agy-acp] WARN: {} new response steps found (payload sizes: {:?}) but none had extractable text \
                 (field 20.1 missing — schema change?)",
                response_rows.len(), payload_sizes
            );
        }
        return None;
    }
    let text = if response_parts.is_empty() {
        None
    } else {
        Some(response_parts.join("\n"))
    };
    Some(ConversationDelta {
        text,
        max_step_idx: max_idx,
    })
}
