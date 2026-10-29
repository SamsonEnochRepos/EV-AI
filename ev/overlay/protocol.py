"""Overlay IPC: newline-delimited JSON over a loopback TCP socket, token-authenticated.

The overlay process is the server. It binds 127.0.0.1 on an ephemeral port, receives the
per-session token on its stdin from the parent, and prints the port on stdout. The first
line from any client must carry the token (constant-time compared); otherwise it is dropped.
"""
from __future__ import annotations

import hmac
import json
import secrets
import socket
from typing import Any

MAX_LINE = 4096
AUTH_TIMEOUT_S = 2.0
LOOPBACK = "127.0.0.1"


def new_token() -> str:
    return secrets.token_urlsafe(24)


def encode(msg: dict[str, Any]) -> bytes:
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode()


def decode(line: bytes) -> dict[str, Any] | None:
    if len(line) > MAX_LINE:
        return None
    try:
        d = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    return d if isinstance(d, dict) else None


def token_ok(expected: str, got: object) -> bool:
    return isinstance(got, str) and hmac.compare_digest(expected.encode(), got.encode())


def is_loopback(addr: str) -> bool:
    return addr in ("127.0.0.1", "::1")


def bind_loopback() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((LOOPBACK, 0))
    s.listen(2)
    return s


def valid_message(m: dict[str, Any]) -> bool:
    t = m.get("t")
    if t == "state":
        return isinstance(m.get("v"), str) and len(m["v"]) < 24
    if t == "level":
        return isinstance(m.get("v"), int | float) and not isinstance(m.get("v"), bool)
    if t == "quit":
        return True
    return False
