mod adapter;
mod db;
mod mcp;
mod output;
mod protobuf;
mod protocol;
mod runtime;
mod streaming;
#[allow(dead_code)]
mod types;

#[cfg(test)]
mod tests;

use std::io::{self, BufRead};

use adapter::Adapter;
use clap::Parser;
use tokio::sync::mpsc;

/// Maximum number of bytes accepted for one newline-delimited JSON-RPC frame.
///
/// The reader still consumes an oversized frame in full so the next frame is
/// not desynchronised, but forwards only a bounded sentinel to the async loop.
pub(crate) const MAX_INPUT_LINE_BYTES: usize = 1024 * 1024;
pub(crate) const OVERSIZED_INPUT_SENTINEL: &str = "\0agy-acp/input-too-large";

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Skip pure narration messages from agy, such as "I will ...".
    #[arg(long = "skip-naration", default_value_t = false)]
    skip_naration: bool,
}

pub(crate) fn forward_input_lines<R: BufRead>(
    mut reader: R,
    input_tx: mpsc::UnboundedSender<String>,
) -> io::Result<()> {
    loop {
        let mut bytes = Vec::new();
        let mut overflowed = false;
        let mut reached_eof = false;

        loop {
            let chunk = reader.fill_buf()?;
            if chunk.is_empty() {
                reached_eof = true;
                break;
            }

            let newline = chunk.iter().position(|byte| *byte == b'\n');
            let consumed = newline.map_or(chunk.len(), |index| index + 1);
            if !overflowed {
                let remaining = MAX_INPUT_LINE_BYTES
                    .saturating_add(1)
                    .saturating_sub(bytes.len());
                if consumed > remaining {
                    bytes.extend_from_slice(&chunk[..remaining]);
                    overflowed = true;
                } else {
                    bytes.extend_from_slice(&chunk[..consumed]);
                }
            }
            reader.consume(consumed);

            if newline.is_some() {
                break;
            }
        }

        if reached_eof && bytes.is_empty() {
            break;
        }

        let content_len = bytes.strip_suffix(b"\n").map_or(bytes.len(), |content| {
            content
                .strip_suffix(b"\r")
                .map_or(content.len(), |content| content.len())
        });
        if overflowed || content_len > MAX_INPUT_LINE_BYTES {
            if input_tx.send(OVERSIZED_INPUT_SENTINEL.to_string()).is_err() {
                break;
            }
        } else {
            let mut line = String::from_utf8(bytes)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "stdin is not UTF-8"))?;
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            if input_tx.send(line).is_err() {
                break;
            }
        }

        if reached_eof {
            break;
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let adapter = if cli.skip_naration {
        Adapter::new_with_skip_naration(true)
    } else {
        Adapter::new()
    };

    let (input_tx, input_rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        if let Err(error) = forward_input_lines(stdin.lock(), input_tx) {
            eprintln!("[agy-acp] stdin failed: {error}");
        }
    });

    if let Err(error) = runtime::run_bridge(adapter, input_rx, tokio::io::stdout()).await {
        eprintln!("[agy-acp] bridge failed: {error}");
        std::process::exit(1);
    }
}
