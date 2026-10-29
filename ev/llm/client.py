"""Ollama client (localhost only). Streaming chat with tool-call support."""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ev.core.config import Config
from ev.core.log import get_logger

log = get_logger("llm")


class LLMError(RuntimeError):
    pass


class OllamaDown(LLMError):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatChunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    CPU_FALLBACK_S = 300      # how long to stay on the CPU after the GPU ran out of memory

    def __init__(self, cfg: Config, transport: httpx.BaseTransport | None = None):
        c = cfg["llm"]
        host = c["host"]
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise LLMError(f"refusing non-loopback Ollama host: {host}")
        self.base = f"http://{host}:{c['port']}"
        self.model = c["model"]
        self.fallback = c.get("fallback_model")
        self.opts = {
            "num_ctx": c["num_ctx"],
            "temperature": c["temperature"],
            "num_predict": c["num_predict"],
        }
        self.keep_alive = c["keep_alive"]
        self.think = c.get("think", False)
        self._cpu_until = 0.0     # monotonic time until which we run on CPU (set after a GPU out-of-memory error)
        self.http = httpx.Client(
            base_url=self.base, timeout=c["request_timeout_s"], transport=transport
        )

    def close(self) -> None:
        self.http.close()

    def available_models(self) -> list[str]:
        try:
            r = self.http.get("/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaDown(str(e)) from e
        return [m["name"] for m in r.json().get("models", [])]

    def is_up(self) -> bool:
        try:
            self.available_models()
            return True
        except LLMError:
            return False

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> Iterator[ChatChunk]:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,
            "keep_alive": keep_alive if keep_alive is not None else self.keep_alive,
            "options": {**self.opts, **(options or {})},
        }
        if tools:
            body["tools"] = tools
        if time.monotonic() < self._cpu_until:
            body["options"]["num_gpu"] = 0
        started = False
        try:
            for ch in self._stream(body):
                started = True
                yield ch
            return
        except LLMError as e:
            # A game or browser can eat the 4 GB of VRAM; slower CPU inference beats no answer at all.
            if (started or isinstance(e, OllamaDown) or "out of memory" not in str(e).lower()
                    or body["options"].get("num_gpu") == 0):
                raise
        log.warning("model did not fit in GPU memory; using the CPU for the next %d s", self.CPU_FALLBACK_S)
        self._cpu_until = time.monotonic() + self.CPU_FALLBACK_S
        body["options"]["num_gpu"] = 0
        yield from self._stream(body)

    def _stream(self, body: dict[str, Any]) -> Iterator[ChatChunk]:
        try:
            with self.http.stream("POST", "/api/chat", json=body) as r:
                if r.status_code != 200:
                    r.read()
                    raise LLMError(f"ollama {r.status_code}: {r.text[:200]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("error"):
                        raise LLMError(f"ollama: {str(d['error'])[:200]}")
                    msg = d.get("message", {})
                    calls = [
                        ToolCall(tc["function"]["name"], _args(tc["function"].get("arguments")))
                        for tc in msg.get("tool_calls", []) or []
                    ]
                    stats = {}
                    if d.get("done"):
                        stats = {k: d[k] for k in (
                            "total_duration", "load_duration", "prompt_eval_count",
                            "prompt_eval_duration", "eval_count", "eval_duration",
                        ) if k in d}
                    yield ChatChunk(msg.get("content", "") or "", calls, bool(d.get("done")), stats)
        except httpx.ConnectError as e:
            raise OllamaDown(str(e)) from e
        except httpx.HTTPError as e:
            raise LLMError(str(e)) from e

    def chat(self, messages: list[dict[str, Any]], **kw: Any) -> ChatChunk:
        text, calls, stats = [], [], {}
        for ch in self.chat_stream(messages, **kw):
            text.append(ch.text)
            calls += ch.tool_calls
            if ch.done:
                stats = ch.stats
        return ChatChunk("".join(text), calls, True, stats)

    def unload(self, model: str) -> None:
        try:
            self.http.post("/api/generate", json={"model": model, "keep_alive": 0})
        except httpx.HTTPError:
            log.warning("unload of %s failed", model)

    def warm_up(self) -> None:
        self.chat([{"role": "user", "content": "hi"}], options={"num_predict": 1})


def _args(a: Any) -> dict[str, Any]:
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
