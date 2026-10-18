from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from llmtestlab.client import free_port


class FakeOpenAIServer:
    def __init__(self) -> None:
        self.host = "127.0.0.1"
        self.port = free_port(self.host)
        self.server = ThreadingHTTPServer((self.host, self.port), FakeOpenAIHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> FakeOpenAIServer:
        self.thread.start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.send_json({"ok": True})

    def do_POST(self) -> None:
        body = self.read_json()
        if self.path.endswith("/chat/completions"):
            self.handle_chat(body)
        elif self.path.endswith("/responses"):
            self.handle_responses(body)
        else:
            self.send_json({"ok": True, "path": self.path})

    def handle_chat(self, body: dict[str, Any]) -> None:
        raw_messages = body.get("messages")
        messages = raw_messages if isinstance(raw_messages, list) else []
        if any(isinstance(message, dict) and message.get("role") == "tool" for message in messages):
            payload = {
                "id": "chatcmpl-final",
                "object": "chat.completion",
                "model": body.get("model", "fake-model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"answer":"sunny"}'},
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        else:
            payload = {
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "model": body.get("model", "fake-model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": json.dumps({"city": "Paris"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        self.send_json(payload)

    def handle_responses(self, body: dict[str, Any]) -> None:
        input_value = body.get("input")
        has_tool_output = isinstance(input_value, list) and any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in input_value
        )
        if has_tool_output:
            payload = {
                "id": "resp_final",
                "object": "response",
                "model": body.get("model", "fake-model"),
                "output_text": '{"answer":"sunny"}',
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": '{"answer":"sunny"}'}],
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            }
        else:
            payload = {
                "id": "resp_tool",
                "object": "response",
                "model": body.get("model", "fake-model"),
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_weather",
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    }
                ],
                "usage": {"input_tokens": 6, "output_tokens": 4, "total_tokens": 10},
            }
        self.send_json(payload)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}

    def send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return
