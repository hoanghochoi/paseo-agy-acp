use serde_json::{json, Value};

use crate::types::JsonRpcResponse;

#[derive(Debug)]
pub(crate) struct RpcCall {
    pub method: String,
    pub params: Value,
}

#[derive(Debug)]
pub(crate) enum IncomingMessage {
    Request { id: Value, call: RpcCall },
    Notification { call: RpcCall },
}

pub(crate) fn parse_jsonrpc_line(line: &str) -> Result<IncomingMessage, JsonRpcResponse> {
    let value: Value = serde_json::from_str(line)
        .map_err(|_| JsonRpcResponse::error(Value::Null, -32700, "Parse error"))?;
    let object = value
        .as_object()
        .ok_or_else(|| JsonRpcResponse::error(Value::Null, -32600, "Invalid Request"))?;
    let candidate_id = object.get("id").cloned().unwrap_or(Value::Null);
    let valid_id = candidate_id.is_null() || candidate_id.is_string() || candidate_id.is_number();
    let response_id = valid_id
        .then_some(candidate_id.clone())
        .unwrap_or(Value::Null);
    if object.get("jsonrpc") != Some(&json!("2.0")) || !valid_id {
        return Err(JsonRpcResponse::error(
            response_id,
            -32600,
            "Invalid Request",
        ));
    }
    let method = object
        .get("method")
        .and_then(Value::as_str)
        .filter(|method| !method.is_empty())
        .ok_or_else(|| {
            JsonRpcResponse::error(candidate_id.clone(), -32600, "Invalid Request")
        })?
        .to_string();
    let params = object.get("params").cloned().unwrap_or_else(|| json!({}));
    if !params.is_object() && !params.is_array() {
        return Err(JsonRpcResponse::error(
            candidate_id,
            -32602,
            "Invalid params",
        ));
    }
    let call = RpcCall { method, params };
    if object.contains_key("id") {
        Ok(IncomingMessage::Request {
            id: candidate_id,
            call,
        })
    } else {
        Ok(IncomingMessage::Notification { call })
    }
}
