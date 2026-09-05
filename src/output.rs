use std::io;

use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;

use crate::types::JsonRpcResponse;

pub(crate) type OutputSender = mpsc::Sender<String>;

pub(crate) fn channel() -> (OutputSender, mpsc::Receiver<String>) {
    mpsc::channel(256)
}

pub(crate) async fn send_response(
    sender: &OutputSender,
    response: JsonRpcResponse,
) -> Result<(), mpsc::error::SendError<String>> {
    sender
        .send(serde_json::to_string(&response).expect("JSON-RPC response is serializable"))
        .await
}

pub(crate) async fn write_messages<W>(
    mut writer: W,
    mut receiver: mpsc::Receiver<String>,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    while let Some(message) = receiver.recv().await {
        writer.write_all(message.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }
    Ok(())
}
