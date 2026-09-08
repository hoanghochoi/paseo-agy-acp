# agy-acp

Single Rust crate. ACP (Agent Client Protocol) stdio adapter for Google Antigravity CLI (`agy`). Bridges `agy` into Paseo's workspace-scoped ACP/JSON-RPC provider flow.

## Commands

```bash
cargo build                    # debug build
cargo build --release          # release build (required for e2e tests)
cargo test                     # unit tests only (fast, no I/O)
cargo test -- --include-ignored  # all tests including filesystem I/O tests
cargo test e2e -- --ignored --nocapture  # e2e only (needs agy binary + auth)
cargo fmt -- --check           # formatting gate
cargo clippy --all-targets -- -D warnings  # lint gate
```

GitHub Actions runs the format, Clippy, unit/I/O test, and release-build gates on Ubuntu and Windows. Authenticated E2E remains a local/Paseo gate because it requires an `agy` installation and user authentication.

## Architecture

- `main.rs` — stdin/stdout JSON-RPC loop. Reads lines, dispatches to adapter methods, writes responses.
- `adapter.rs` — core logic: session lifecycle, spawning `agy` subprocess, state persistence. `Adapter::new()` reads `HOME` for state/conv dirs.
- `db.rs` — reads agy's SQLite conversation DBs (read-only). Table: `steps` with columns `idx`, `step_type`, `step_payload`.
- `protobuf.rs` — hand-rolled protobuf varint/field extraction (no prost/protobuf dependency). Extracts text from `step_payload` field 20 → sub-field 1.
- `streaming.rs` — polls SQLite every 500ms during `session/prompt`, emits incremental `session/update` notifications to stdout.
- `types.rs` — JSON-RPC types, `SessionStore` for persistence, `StreamingState`.

## Key paths

| Path | Purpose |
|---|---|
| `~/.openab/agy-acp/sessions.json` | Persisted session→conversation mapping (with `.lock` file for mutual exclusion) |
| `~/.gemini/antigravity-cli/conversations/*.db` | agy's SQLite conversation databases |

## Test tiers

1. **Unit tests** (`cargo test`) — protobuf parsing, narration filtering, JSON-RPC response shape. No filesystem or network I/O.
2. **Ignored I/O tests** (`-- --include-ignored`) — session persist/restore, SQLite read, conversation snapshot. Create temp dirs in `$TMPDIR`.
3. **E2E tests** (`e2e -- --ignored`) — spawn the release binary, send JSON-RPC over stdin, verify responses. Requires:
   - `agy` in `PATH` (install from `google-antigravity/antigravity-cli` releases)
   - Auth via `GEMINI_API_KEY` env var or macOS Keychain (`~/.gemini/antigravity-cli/settings.json`)
   - `cargo build --release` must have been run first

## Environment variables

| Var | Effect |
|---|---|
| `AGY_EXTRA_ARGS` | Space-separated extra args passed to every `agy` invocation |
| `AGY_PRINT_TIMEOUT` | Override the bridge-owned `agy --print-timeout` (default `24h`) |
| `AGY_MODEL_DISCOVERY_TIMEOUT_MS` | Override the `agy models` discovery deadline in milliseconds (default `30000`, capped at `120000`) |
| `GEMINI_API_KEY` | API key for e2e tests and CI |

## Quirks

- `rusqlite` uses `bundled` feature — no system SQLite dependency needed.
- SQLite reads use `SQLITE_OPEN_READ_ONLY | SQLITE_OPEN_NO_MUTEX` — single-threaded access assumed per conversation DB.
- State persistence uses write-to-tmp-then-rename pattern under an exclusive file lock (`fs2`).
- Streaming notifications and responses share one bounded output channel and writer, so concurrent prompts cannot interleave bytes on stdout.
- `handle_session_load` returns a `Vec<String>` (multiple notifications + final response), not a single response like other methods.
- Conversation binding: every `agy` invocation gets a unique `--log-file`; the adapter reads that invocation's `Created conversation <uuid>` record (or its own child PID's open DB as a fallback). Never infer ownership from a process-global directory diff, because parallel invocations can otherwise cross-bind.
- Final-answer recovery: SQLite provides streaming/tool/history updates, while captured `agy --print` stdout is the fallback when SQLite emitted no assistant text. A successful process with no conversation ID or no assistant response fails closed instead of returning an empty `end_turn`.
- Turn completion: the bridge raises `agy --print-timeout` to 24 hours by default and treats an observed print timeout or any non-zero `agy` exit as a failed ACP turn, even if partial updates were streamed. It never converts either condition into `end_turn`.
- `fetch_available_models()` runs `agy models` synchronously during `Adapter::new()` with a 30-second default deadline (override with `AGY_MODEL_DISCOVERY_TIMEOUT_MS`, capped at 120 seconds). If `agy` isn't installed, times out, or exits unsuccessfully, the models list is empty (no error).
- `session/cancel` marks the active prompt cancelled; on Windows the adapter terminates the child process tree before returning the cancelled response.
- Both `session/set_model` and `session/setConfigOption` are accepted for model selection.
- Runtime input and output are bounded: JSON-RPC frames (1 MiB), prompt text (256 KiB), `AGY_EXTRA_ARGS` (64 KiB), model discovery output (256 KiB), child stdout (4 MiB), SQLite step payloads (1 MiB), and SQLite rows per poll (256). Oversized data fails closed or advances the cursor without retaining the payload.
- Persisted session state is bounded to 1 MiB and 1024 sessions; malformed, oversized, or invalid-cursor state fails closed before installation.
- Persisted conversation IDs are bounded and reject control/path-separator characters; SQLite access still requires a parsed UUID before joining the conversation database path.
- SQLite replay pages are bounded to 256 rows and 4 MiB per read, with an 8 MiB total replay budget. `session/load` paginates to the latest cursor; histories exceeding the budget return an explicit error instead of partial replay.
- The resident cache uses a deterministic LRU-style eviction policy at 64 sessions and prefers evicting inactive sessions; retained failure logs are pruned to 64 files/16 MiB while active and current diagnostics are protected.
