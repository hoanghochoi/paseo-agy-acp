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
        for line in stdin.lock().lines() {
            match line {
                Ok(line) if !line.trim().is_empty() => {
                    if input_tx.send(line).is_err() {
                        break;
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    eprintln!("[agy-acp] stdin failed: {error}");
                    break;
                }
            }
        }
    });

    if let Err(error) = runtime::run_bridge(adapter, input_rx, tokio::io::stdout()).await {
        eprintln!("[agy-acp] bridge failed: {error}");
        std::process::exit(1);
    }
}
