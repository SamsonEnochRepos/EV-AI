"""Test doubles: a fake Ollama HTTP transport that replays scripted responses."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx


def ndjson(*items: dict[str, Any]) -> bytes:
    return ("\n".join(json.dumps(i) for i in items) + "\n").encode()


def text_reply(text: str, parts: int = 3) -> bytes:
    n = max(1, len(text) // parts)
    chunks = [text[i:i + n] for i in range(0, len(text), n)]
    lines = [{"message": {"role": "assistant", "content": c}, "done": False} for c in chunks]
    lines.append({"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 10,
                  "eval_duration": 500_000_000, "prompt_eval_count": 20})
    return ndjson(*lines)


def tool_reply(name: str, args: dict[str, Any]) -> bytes:
    return ndjson(
        {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    )


class FakeOllama:
    """handler(request_json, call_index) -> bytes ndjson. Records every request."""

    def __init__(self, replies: list[bytes] | Callable[[dict[str, Any], int], bytes]):
        self.replies = replies
        self.requests: list[dict[str, Any]] = []
        self.models = ["qwen2.5:3b", "llama3.2:latest"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        i = len(self.requests) - 1
        data = self.replies(body, i) if callable(self.replies) else self.replies[min(i, len(self.replies) - 1)]
        return httpx.Response(200, content=data)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)
