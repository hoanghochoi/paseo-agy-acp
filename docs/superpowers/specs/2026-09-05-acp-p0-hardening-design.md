# ACP P0 Hardening Design

**Status:** Awaiting written-spec review

**Date:** 2026-09-05

## Goal

Make `agy-acp` safe and predictable when Paseo runs more than one workspace or session. The bridge must execute `agy` in the ACP session's requested working directory, keep JSON-RPC output structurally valid under concurrency, remain responsive to cancellation, and reject invalid input before spawning a child process.

## Baseline

The current adapter passes 54 default tests and its Paseo provider diagnostic. It already provides ACP v1 initialization, session new/load/resume/prompt/cancel, model configuration, SQLite streaming, invocation-specific conversation binding, and fail-closed turn completion.

The P0 risks are:

1. `session/new`, `session/load`, and `session/resume` do not retain their request `cwd`; every child uses the bridge process's startup directory.
2. A single `Mutex<Adapter>` is held for the complete lifetime of `agy`, serializing unrelated sessions and allowing an earlier blocking request to delay a later cancel notification.
3. The main loop and polling thread both write to stdout directly.
4. Malformed JSON is ignored, unknown sessions can reach prompt execution, and unsupported prompt blocks are silently discarded.

The ACP v1 JSON schema is the protocol authority for this phase:

- <https://raw.githubusercontent.com/agentclientprotocol/agent-client-protocol/main/schema/v1/schema.json>
- <https://www.jsonrpc.org/specification>

## Scope

### Included

- Validate and retain `cwd` for `session/new`, `session/load`, and `session/resume`.
- Require `mcpServers` to be an array. Accept an empty array; reject a non-empty array until the adapter can actually connect those servers.
- Persist a session's `cwd` with backward-compatible deserialization of existing state.
- Reject missing, empty, relative, nonexistent, or non-directory `cwd` values.
- Reject an empty or unknown `sessionId` before spawning `agy`.
- Accept text prompt blocks only. Reject the entire request when any unsupported content block is present; never silently omit user content.
- Reject empty effective prompt text.
- Route every JSON-RPC response and notification through one bounded output channel and one stdout writer.
- Hold the adapter mutex only while reading or updating shared session state, never while waiting for `agy`.
- Permit prompts in different sessions to run concurrently.
- Permit at most one active prompt per session and return a clear implementation-defined error for a second prompt.
- Preserve `session/cancel` behavior and make it responsive while prompts are running.
- Return standard JSON-RPC parse, invalid-request, invalid-params, method-not-found, and internal/runtime errors.
- Add deterministic tests that do not contact Gemini.

### Excluded

- `session/list`, `session/close`, `session/delete`, and `$/cancel_request`.
- Forwarding non-empty ACP MCP server definitions into `agy`.
- Image, audio, embedded-resource, and resource-link prompt conversion.
- Changes to protobuf decoding, SQLite polling frequency, tool lifecycle status, model discovery, log retention, or `AGY_EXTRA_ARGS` parsing.
- ACP v2 adoption.
- GitHub push, release, or Paseo production configuration changes.

## Architecture

### 1. Protocol envelope and errors

Input remains newline-delimited JSON-RPC 2.0. Parsing is split into two stages:

1. Parse the line as JSON. A syntax failure emits error `-32700` with `id: null`.
2. Validate the JSON-RPC envelope: it must be an object with `jsonrpc: "2.0"`, a non-empty string `method`, and an optional string, number, or null `id`. An invalid envelope emits `-32600` with the usable request ID or `null`.

Valid notifications receive no response. `session/cancel` remains the only notification with behavior in this phase. Unknown notifications are ignored as required by JSON-RPC compatibility rules. Unknown request methods return `-32601`.

Handlers use shared constructors for success and error responses so every path emits the same envelope shape. Error mapping is:

| Condition | Code |
|---|---:|
| Malformed JSON | `-32700` |
| Invalid JSON-RPC envelope | `-32600` |
| Unknown request method | `-32601` |
| Missing or malformed request parameters | `-32602` |
| Unknown session, session busy, child-process failure, timeout, or invalid upstream result | `-32000` |
| Unexpected bridge failure | `-32603` |

No error message includes prompt content, environment values, or unredacted subprocess output.

### 2. Session context

`Session` gains a `cwd: PathBuf`. `StoredSession` gains a backward-compatible optional serialized `cwd` field using `#[serde(default)]`.

Lifecycle behavior is:

- `session/new`: validate `cwd` and `mcpServers`, create the session, and persist it immediately.
- `session/load` and `session/resume`: validate the request `cwd` and `mcpServers`; restore the conversation state; use the request `cwd` as the current execution context; persist the refreshed context.
- Legacy stored sessions without `cwd` remain loadable because load/resume supply and persist the current request `cwd`.
- `session/prompt`: resolve an in-memory session or restore a known persisted session. If neither exists, return `-32000` without creating logs or starting `agy`.

Working-directory validation requires an absolute path whose metadata identifies a directory. The bridge retains the validated absolute path as supplied rather than canonicalizing it; this preserves Windows drive/UNC spelling and legitimate symlink semantics. Child execution uses that session path for both `Command::current_dir` and the required `agy --add-dir` argument. A restored legacy session with no stored `cwd` cannot execute a prompt until `session/load` or `session/resume` supplies one.

### 3. Prompt preparation and execution

Prompt handling is split into three phases:

1. `prepare_prompt` runs while the adapter is locked. It validates the session and content, snapshots the session's conversation ID, model ID, `cwd`, state/log paths, and streaming cursor into an owned `PromptExecution` value.
2. `execute_prompt` owns the child process and streaming poller and runs without an adapter lock. It sends update notifications through the shared output sender and returns a `PromptOutcome` containing the final response data and updated conversation cursor.
3. `apply_prompt_outcome` briefly locks the adapter to update/persist session state. It only applies an outcome to the matching session.

This boundary ensures child-process lifetime cannot block unrelated lifecycle/configuration requests.

### 4. Active-prompt registry and cancellation

The process keeps an active registry keyed by `sessionId`. Each entry owns the cancellation flag for exactly one prompt.

- Registration happens before the prompt task is spawned.
- A duplicate prompt for the same session returns `-32000` and does not replace the original cancellation flag.
- Different session IDs may execute concurrently.
- `session/cancel` only flips the matching flag and remains a notification without a response.
- Cleanup removes an entry only if it still belongs to the completing prompt, preventing stale completion from removing a newer entry.

The existing child-kill behavior is retained: cancellation kills and waits for the owned child before final cleanup.

### 5. Single output writer

A bounded Tokio MPSC channel with capacity 256 is the only path to stdout.

- The main dispatcher, prompt tasks, and polling threads receive cloned senders.
- The polling thread uses `blocking_send` only after releasing its streaming-state lock.
- A dedicated async writer owns buffered stdout, writes one serialized message plus one newline, and flushes after every message for interactive delivery.
- A prompt joins its poller and queues all final deltas before queuing its terminal response. This preserves update-before-response order within that prompt.
- Responses and notifications from different sessions may interleave as complete JSON lines, which is valid JSON-RPC concurrency.
- A closed output channel is treated as process-fatal because the bridge can no longer satisfy the protocol.

No adapter or polling function may call `io::stdout()` directly after this change.

## Data compatibility

Existing `sessions.json` files remain readable because `StoredSession.cwd` is optional during deserialization. New or refreshed sessions write canonical `cwd` values. No state file is deleted or automatically migrated in bulk.

The external ACP protocol version remains `1`. Existing model/configuration response fields remain unchanged.

## Security and failure behavior

- Paths are supplied only as discrete `Command` arguments; no shell is introduced.
- An invalid path or unsupported non-empty MCP server list fails before subprocess creation.
- Unsupported content fails closed so user-provided context cannot be silently lost.
- Runtime errors continue retaining the invocation log locally for diagnosis, but response messages expose only the existing local log path and summarized failure reason.
- Concurrency never permits two active prompts to mutate the same session cursor.

## Verification strategy

Implementation follows red-green-refactor. Each behavior first receives a focused failing test whose failure proves the old behavior.

Required automated coverage:

1. JSON parse error and invalid envelope responses.
2. Lifecycle rejection for missing/relative/non-directory `cwd` and malformed/non-empty `mcpServers`.
3. `session/new`, load, and resume retain the request `cwd`; legacy state without `cwd` remains readable.
4. Prompt rejects empty/unknown session IDs, empty text, and unsupported content before spawn preparation.
5. Two sessions can hold independent execution snapshots while duplicate prompt registration for one session is rejected.
6. Every streaming line and terminal response passes through the output channel; no direct stdout writer remains in adapter code.
7. Cancellation remains observable while another session is running.
8. Existing conversation binding, timeout, narration, model, load, and resume tests continue to pass.

Final local verification commands are:

```powershell
cargo fmt -- --check
cargo build
cargo test
cargo test -- --include-ignored --skip test_e2e_
```

The ignored live Gemini E2E tests are not run without separate authorization because they transmit prompt content externally. The `--skip test_e2e_` filter runs ignored filesystem coverage without invoking those live model tests.

## Rollback

The stable baseline is local commit `5207341` on `main`; worktree infrastructure is commit `743d74e`. All implementation occurs on `feat/acp-p0-hardening` in `.worktrees/p0-hardening`. Rollback consists of discarding that feature worktree/branch after explicit approval; `main` remains untouched by runtime changes.

## Acceptance criteria

The phase is acceptable only when:

- Paseo-provided `cwd` determines every spawned `agy` process's working directory.
- Invalid requests cannot spawn `agy`.
- A long prompt does not block cancellation or operations on another session.
- Concurrent activity produces only complete, parseable JSON-RPC lines.
- One session never has more than one active prompt.
- All non-live verification commands pass, and the remaining unverified live-E2E boundary is reported explicitly.
