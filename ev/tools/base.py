"""Tool framework: schema, validation, timeout, structured results."""
from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ev.core.log import get_logger

log = get_logger("tools")


@dataclass
class ToolResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    untrusted: bool = False  # true => message is external text and must be wrapped as data
    source: str = ""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON-schema "properties" mapping
    func: Callable[..., ToolResult]
    required: tuple[str, ...] = ()
    timeout_s: float = 10.0

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.required),
                },
            },
        }


_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str, "integer": int, "number": (int, float), "boolean": bool,
}


def validate_args(tool: Tool, args: Any) -> tuple[dict[str, Any] | None, str]:
    """Strict validation. Unknown keys are dropped-and-rejected; types coerced only int->float."""
    if not isinstance(args, dict):
        return None, "arguments must be an object"
    for r in tool.required:
        if r not in args:
            return None, f"missing argument: {r}"
    clean: dict[str, Any] = {}
    for k, v in args.items():
        spec = tool.parameters.get(k)
        if spec is None:
            return None, f"unexpected argument: {k}"
        t = _TYPES.get(spec.get("type", "string"), str)
        if spec.get("type") == "integer" and isinstance(v, float) and v.is_integer():
            v = int(v)
        if isinstance(v, bool) and spec.get("type") != "boolean":
            return None, f"bad type for {k}"
        if not isinstance(v, t):
            return None, f"bad type for {k}"
        if "enum" in spec and v not in spec["enum"]:
            return None, f"{k} must be one of {spec['enum']}"
        if isinstance(v, str) and len(v) > spec.get("maxLength", 2000):
            return None, f"{k} too long"
        if isinstance(v, int | float) and not isinstance(v, bool):
            if "minimum" in spec and v < spec["minimum"]:
                return None, f"{k} below minimum"
            if "maximum" in spec and v > spec["maximum"]:
                return None, f"{k} above maximum"
        clean[k] = v
    return clean, ""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool")
        self._schemas: list[dict[str, Any]] | None = None  # cached (schema caching optimisation)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._schemas = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        if only is not None:
            return [t.schema() for n, t in self._tools.items() if n in only]
        if self._schemas is None:
            self._schemas = [t.schema() for t in self._tools.values()]
        return self._schemas

    def run(self, tool: Tool, args: dict[str, Any]) -> ToolResult:
        fut = self._pool.submit(tool.func, **args)
        try:
            return fut.result(timeout=tool.timeout_s)
        except concurrent.futures.TimeoutError:
            return ToolResult(False, f"{tool.name} timed out")
        except Exception as e:
            log.exception("tool %s failed", tool.name)
            return ToolResult(False, f"{tool.name} failed: {type(e).__name__}")
