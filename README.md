# agy-acp

An [Agent Client Protocol (ACP)](https://agentclientprotocol.com) stdio adapter for [Google Antigravity CLI](https://github.com/google-antigravity/antigravity-cli) (`agy`). It is optimized for Paseo-managed workspaces and agents, while remaining compatible with other ACP hosts.

## How It Works

`agy-acp` speaks JSON-RPC over stdin/stdout (the ACP transport). When Paseo sends a prompt for a workspace session, `agy-acp` spawns `agy` in that workspace, gives the invocation a unique log file so its conversation ID can be bound safely under parallel execution, streams SQLite-backed updates as incremental `session/update` notifications, and persists session state across restarts. Captured `agy --print` stdout supplies the final-answer fallback; missing conversation identity or assistant output is reported as an error rather than a successful empty turn.

```
Paseo workspace/agent (ACP host)  <--stdin/stdout JSON-RPC-->  agy-acp  <--subprocess-->  agy  <--API-->  Gemini
```

## Prerequisites

- A stable **Rust** toolchain with Cargo
- **`agy`** installed and in your `PATH` — install from [google-antigravity/antigravity-cli releases](https://github.com/google-antigravity/antigravity-cli)
- **Authentication** — either set `GEMINI_API_KEY` or configure auth via `~/.gemini/antigravity-cli/settings.json`

## Build & Install

```bash
cargo build --release
```

The binary is at `target/release/agy-acp`. Copy it somewhere in your `PATH`:

```bash
cp target/release/agy-acp /usr/local/bin/
```

For Paseo, prefer configuring the absolute release-binary path so the daemon does not depend on an interactive shell `PATH`:

```text
Windows:  C:\path\to\agy-acp\target\release\agy-acp.exe
Unix:     /path/to/agy-acp/target/release/agy-acp
```

## Use with Paseo

Configure the Paseo ACP provider/agent to launch the release binary above directly; no shell wrapper is required. Paseo supplies the active workspace directory in each ACP lifecycle request, so the adapter runs `agy` in the same workspace and keeps session state bound to that workspace.

Before starting a Paseo session:

```bash
cargo build --release
agy --version
```

The Paseo daemon account must be able to find `agy` and its authentication. Use either `GEMINI_API_KEY` or the local Antigravity CLI settings/keyring available to that account.

### Model Selection

`agy-acp` queries available models by running `agy models` at startup. Paseo can switch models through the ACP config options (`session/set_model` or `session/setConfigOption`). Discovery has a 30-second default deadline, configurable with `AGY_MODEL_DISCOVERY_TIMEOUT_MS`, and falls back to an empty model list when `agy` is unavailable.

### Passing Extra Arguments

For an opt-in primary AGY agent with a bounded inspection tool catalog, see
[the SLP PEER profile](docs/slp-peer-profile.md). It uses the existing
`AGY_EXTRA_ARGS` launch configuration without changing default ACP behavior.

Set `AGY_EXTRA_ARGS` in the Paseo provider environment to pass additional arguments to every `agy` invocation. Values support single/double quotes and backslash escapes without invoking a shell:

```text
AGY_EXTRA_ARGS=--some-flag value
```

Keep ACP stdout dedicated to protocol messages. Paseo's managed terminal/agent diagnostics and the bounded local invocation logs are the supported debugging surfaces.

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key for Gemini (passed through to `agy`) |
| `AGY_EXTRA_ARGS` | Space-separated extra args passed to every `agy` invocation |
| `AGY_PRINT_TIMEOUT` | Maximum time `agy --print` may wait for a trustworthy completed turn (default `24h`) |
| `AGY_MODEL_DISCOVERY_TIMEOUT_MS` | Maximum time to wait for `agy models` (default `30000`, capped at `120000`) |

## Session Persistence

Sessions are persisted to `~/.openab/agy-acp/sessions.json`, including the absolute working directory supplied by Paseo. When Paseo resumes a session, `agy-acp` restores the conversation binding and replays the message history from `agy`'s SQLite conversation databases (`~/.gemini/antigravity-cli/conversations/*.db`).

ACP lifecycle requests must include an existing absolute `cwd` and an empty `mcpServers` array. For example:

```json
{"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/absolute/path/to/project","mcpServers":[]}}
{"jsonrpc":"2.0","id":2,"method":"session/load","params":{"sessionId":"SESSION_ID","cwd":"/absolute/path/to/project","mcpServers":[]}}
{"jsonrpc":"2.0","id":3,"method":"session/resume","params":{"sessionId":"SESSION_ID","cwd":"/absolute/path/to/project","mcpServers":[]}}
```

Non-empty `mcpServers` lists are rejected because `agy-acp` does not currently forward external MCP server definitions to `agy`.

## Runtime Guarantees

- ACP `session/new`, `session/load`, and `session/resume` lifecycle requests require both an existing absolute `cwd` and an `mcpServers` array. Only an empty `mcpServers` array is currently supported; missing, malformed, or non-empty values fail explicitly.
- Prompt executions can run concurrently across different sessions. At most one prompt per session may be active; overlapping requests for the same session are rejected rather than queued or serialized.
- All JSON-RPC responses and `session/update` notifications flow through one stdout writer, preserving complete newline-delimited JSON messages under concurrent activity.
- Prompt content must be a non-empty array of text blocks with string `text` values. Unsupported, mixed, or malformed content fails explicitly instead of being silently dropped.
- Input and subprocess buffers are bounded: JSON-RPC frames are limited to 1 MiB, prompt text to 256 KiB, `AGY_EXTRA_ARGS` to 64 KiB, model discovery output to 256 KiB, child stdout to 4 MiB, and each SQLite step payload to 1 MiB. Oversized frames fail with an invalid-request response; oversized runtime data fails closed without unbounded allocation.
- SQLite polling reads at most 256 steps and 4 MiB of payload per pass, and advances over oversized payloads, so a long conversation cannot force one poll to materialize its entire history. Session replay paginates across pages but fails closed above an 8 MiB history budget rather than returning a partial cursor. Invocation log scans use bounded prefix/tail windows.
- Persisted `sessions.json` is capped at 1 MiB and 1024 sessions; oversized or malformed state is rejected before it can enter the in-memory session cache.
- `AGY_EXTRA_ARGS` supports single/double quotes and backslash escapes without invoking a shell; malformed quoting fails closed. Resident sessions use deterministic LRU-style eviction at 64 entries and prefer inactive sessions.
- Persisted conversation IDs reject control/path-separator characters before state installation; SQLite database access still requires a valid UUID. Retained failure logs are bounded to 64 files/16 MiB while active/current logs are protected.
- Cancelling a prompt terminates the full child process tree on Windows and suppresses the helper command's output from the ACP stdout stream.

## Local Verification

Run the default unit suite with `cargo test`. To include filesystem-backed ignored tests without contacting Gemini or running authenticated end-to-end coverage, use:

```bash
cargo test -- --include-ignored --skip test_e2e_
```

GitHub Actions runs the formatting, Clippy, unit/I/O test, and release-build gates on Ubuntu and Windows. Authenticated E2E coverage remains a local/Paseo gate because it requires an `agy` installation and user authentication.

## ACP host compatibility

Paseo is the primary integration path for this repository. Other ACP hosts can launch the same stdio binary when they provide an existing absolute `cwd` and an empty `mcpServers` array.

For Paseo debugging, capture the managed provider terminal and inspect the bounded invocation-log reference returned by the adapter. Do not write diagnostics to stdout: stdout is reserved for newline-delimited ACP JSON-RPC.

## License

MIT
