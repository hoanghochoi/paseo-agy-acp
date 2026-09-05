mod adapter;
mod db;
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

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Skip pure narration messages from agy, such as "I will ...".
    #[arg(long = "skip-naration", default_value_t = false)]
    skip_naration: bool,
}

pub(crate) fn forward_input_lines<R: BufRead>(
    reader: R,
    input_tx: mpsc::UnboundedSender<String>,
) -> io::Result<()> {
    for line in reader.lines() {
        let line = line?;
        if input_tx.send(line).is_err() {
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
