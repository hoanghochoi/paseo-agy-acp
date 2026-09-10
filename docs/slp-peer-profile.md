# Bounded AGY primary-agent profile

The logical role is **PEER**, independent of engine. The checked-in
`.agents/agents/slp-peer.md` is an AGY primary agent, not an AGY child subagent.
It is opt-in; the default ACP adapter behavior is unchanged.

## Organizational source of truth

The Markdown body is synchronized verbatim from the existing Paseo
`docs/slp-r1/roles/PEER.md` contract. Do not independently edit the organizational
instructions in the generated profile. From this repository:

```text
python scripts/sync-peer-profile.py <paseo-repository>/docs/slp-r1/roles/PEER.md
python scripts/sync-peer-profile.py <paseo-repository>/docs/slp-r1/roles/PEER.md --check
```

The sync script reads the upstream contract without modifying Paseo or DCM.
Implementation, investigation and falsification are assignments for the same
PEER role; no additional Reviewer organizational role is defined.

## Supported launch mechanism

AGY Markdown frontmatter supports `mainAgent: true`, `subagent: false`, and an
explicit `tools` list. `inheritCustomizations: false` disables ambient skills,
rules, plugins, subagents and MCP inheritance for this custom agent. Sources:
[custom-agent schema](https://antigravity.google/docs/subagents),
[primary-agent selection](https://www.antigravity.google/docs/cli/commands/agents),
and [customization inheritance changelog](https://antigravity.google/changelog).

The explicit allowlist is `view_file`, `list_dir`, `find_by_name`, `grep_search`.
AGY 1.2.0 also registers `manage_task` for background-task management. The
resolved model catalog does **not** contain `define_subagent`, `invoke_subagent`,
`manage_subagents`, shell execution, file writes, browser, network or MCP tools.
This is a model-tool ceiling, not an OS sandbox or general process isolation.

The existing bounded/quoted `AGY_EXTRA_ARGS` parser already supports selection;
no Rust or ACP protocol changes are necessary. Configure this environment value
only on the `slp-peer-agy` Paseo provider (replace the absolute path):

```text
AGY_EXTRA_ARGS=--add-dir "C:/path/to/agy-acp" --agent slp-peer --disable-slash-commands
```

Keep `supportsMcpServers: false`. Back up and hash operator configuration before
merging this one environment entry. Reload config, then create a **new** session.
Do not migrate an existing unrestricted AGY conversation into this seat.

The profile repository must be an AGY workspace directory: merely setting the
process working directory was insufficient in the initial direct smoke.
`--add-dir` makes the definition discoverable, including when ACP receives a
different workspace. `--disable-slash-commands` keeps `/agents` and other slash
commands from expanding in print mode. AGY has no separate config-root flag in
the inspected CLI help; this workspace definition avoids global customization.

**Operational prerequisite:** keep this profile at the configured absolute root
and avoid conflicting custom-agent names. AGY itself falls back to its default
agent when a selected name is missing. The generic adapter does not validate
AGY's customization registry. Missing/moved profiles require stopping use of
the seat and re-running the runtime catalog check, not assuming the ceiling.

## Runtime verification

Acceptance was performed on installed AGY **1.2.0**, selected by the operator
after discovering an update from 1.1.28. A separate local 1.1.28 binary provided
preliminary corroboration; it is not the configured runtime.

Use a fresh harmless conversation and `--output-format stream-json` for direct
smoke. Verify actual read/search calls in the conversation transcript. Then
request `Use invoke_subagent to create another agent` without permitting an
alternative delegation path. Inspect the retained model request declarations:

```text
python scripts/inspect-agy-catalog.py <conversation-uuid>.db --require-peer
python scripts/test-agy-catalog.py
```

The SQLite reader opens only the named database in read-only mode and emits
tool names, metadata hashes and subtrajectory counts, not prompts/credentials.
It checks observed internal protobuf declarations: `gen_metadata` request field
1, repeated tool field 8, name/description/JSON schema fields 1/2/3. This is a
version-sensitive diagnostic, not a supported configuration API. Schema
changes or missing full request evidence fail validation.

`stream-json init.tools` lists a broader engine registry even when a custom
agent is selected; do not confuse that list with the resolved model request
catalog. Older generation rows can retain usage only; the reader reports those
as unavailable evidence, never as empty catalogs. Snapshot after each turn when
per-turn catalog evidence is needed. Model prose alone is not acceptance proof.

Repeat through ACP (initialize, new, prompt, streaming, model/config switching,
restart/load/resume, cancel, exit) and through a new `slp-peer-agy` Paseo session.
Check the exact native conversation database for both paths. Preserve failed
probes separately from successful evidence. Re-verify after AGY/profile changes.
