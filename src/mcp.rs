use serde_json::{json, Map, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

const MAX_MCP_CONFIG_BYTES: usize = 1024 * 1024;
const MAX_MCP_SERVERS: usize = 128;
/// Longer than the default `--print-timeout`, so a folder is only pruned once its run can no longer be alive.
const STALE_RUN_CONFIG_AGE: Duration = Duration::from_secs(48 * 60 * 60);

/// Converts ACP `mcpServers` into Antigravity's `{ "mcpServers": { name: config } }`; `None` when there are none.
pub(crate) fn from_acp(params: &Value) -> Result<Option<Value>, String> {
    let servers = params
        .get("mcpServers")
        .and_then(Value::as_array)
        .ok_or_else(|| "mcpServers must be an array".to_string())?;
    if servers.is_empty() {
        return Ok(None);
    }
    if servers.len() > MAX_MCP_SERVERS {
        return Err("mcpServers exceeds the maximum server count".to_string());
    }

    let mut converted = Map::new();
    for server in servers {
        let name = server
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| "each MCP server must have a non-empty name".to_string())?;
        if converted.contains_key(name) {
            return Err("mcpServers contains duplicate server names".to_string());
        }
        let config = match server.get("type") {
            None => stdio_server(server)?,
            Some(Value::String(transport)) if transport == "stdio" => stdio_server(server)?,
            Some(Value::String(transport)) if transport == "http" => http_server(server)?,
            Some(Value::String(_)) => {
                return Err("MCP server transport is not supported by agy-acp".to_string())
            }
            Some(_) => return Err("MCP server type must be a string".to_string()),
        };
        converted.insert(name.to_string(), config);
    }

    let config = json!({ "mcpServers": converted });
    let size = serde_json::to_vec(&config)
        .map_err(|_| "mcpServers could not be encoded".to_string())?
        .len();
    if size > MAX_MCP_CONFIG_BYTES {
        return Err("mcpServers exceeds the maximum configuration size".to_string());
    }
    Ok(Some(config))
}

fn stdio_server(server: &Value) -> Result<Value, String> {
    let command = server
        .get("command")
        .and_then(Value::as_str)
        .filter(|command| !command.trim().is_empty())
        .ok_or_else(|| "stdio MCP server must have a command".to_string())?;
    let args = string_array(server.get("args"), "stdio MCP server args")?;
    let env = name_values(server.get("env"), "stdio MCP server env", false)?;
    Ok(json!({ "command": command, "args": args, "env": env }))
}

fn http_server(server: &Value) -> Result<Value, String> {
    let url = server
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.trim().is_empty())
        .ok_or_else(|| "http MCP server must have a URL".to_string())?;
    let headers = name_values(server.get("headers"), "http MCP server headers", true)?;
    Ok(json!({ "serverUrl": url, "headers": headers }))
}

/// ACP's `[{name, value}]` as an object; header names compare without case, as HTTP does.
fn name_values(
    value: Option<&Value>,
    what: &str,
    ignore_case: bool,
) -> Result<Map<String, Value>, String> {
    let entries = match value {
        None => return Ok(Map::new()),
        Some(Value::Array(entries)) => entries,
        Some(_) => return Err(format!("{what} must be an array")),
    };
    let mut map = Map::new();
    for entry in entries {
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| format!("{what} entries need a name"))?;
        let value = entry
            .get("value")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{what} entries need a string value"))?;
        let duplicate = if ignore_case {
            map.keys().any(|present| present.eq_ignore_ascii_case(name))
        } else {
            map.contains_key(name)
        };
        if duplicate {
            return Err(format!("{what} has duplicate names"));
        }
        map.insert(name.to_string(), Value::String(value.to_string()));
    }
    Ok(map)
}

fn string_array(value: Option<&Value>, what: &str) -> Result<Vec<String>, String> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("{what} must contain only strings"))
            })
            .collect(),
        Some(_) => Err(format!("{what} must be an array")),
    }
}

/// One run's private workspace folder holding `.agents/mcp_config.json`, which `agy` reads from every `--add-dir`;
/// the global config and the session's own directory are never touched. Removed when dropped.
#[derive(Debug)]
pub(crate) struct RunConfig {
    dir: PathBuf,
}

impl RunConfig {
    pub(crate) fn write(config: &Value, state_dir: &Path) -> io::Result<Self> {
        let root = state_dir.join("mcp");
        prune_stale(&root);
        let dir = root.join(Uuid::new_v4().to_string());
        let agents = dir.join(".agents");
        fs::create_dir_all(&agents)?;
        let run = Self { dir };
        let mut bytes = serde_json::to_vec_pretty(config)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        bytes.push(b'\n');
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(agents.join("mcp_config.json"))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(run)
    }

    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for RunConfig {
    fn drop(&mut self) {
        // Best effort: a folder left behind is pruned by a later run once it is stale.
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn prune_stale(root: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        let named_by_us = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| Uuid::parse_str(name).is_ok());
        let stale = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > STALE_RUN_CONFIG_AGE);
        if named_by_us && stale {
            let _ = fs::remove_dir_all(path);
        }
    }
}
