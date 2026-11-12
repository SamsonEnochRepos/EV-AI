"""A loopback-only HTTP endpoint that coding-agent hooks POST their approval requests to.

Separate from the dashboard on purpose (a different port, a different token, no cookies, no page, no way to run anything).
* Binds 127.0.0.1 only; every request must come from loopback with a Host of exactly that address and port.
* `Authorization: Bearer <token>` is required, compared in constant time. The token lives in the data folder.
* JSON body only, at most 1 MB. Anything else is refused before it is parsed.
* The response is the hook's decision, or 204 (no decision) so the agent shows its own prompt.
"""
from __future__ import annotations

import hmac
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ev.core.log import get_logger

from .supervisor import AgentSupervisor

log = get_logger("agents.listener")

MAX_BODY = 1_000_000
LOOPBACK = "127.0.0.1"


def load_or_create_token(path: Path) -> str:
    """The shared secret between EV and the hook config. Created once; delete the file to rotate it."""
    try:
        t = path.read_text(encoding="utf-8").strip()
        if len(t) >= 32:
            return t
    except OSError:
        pass
    t = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(t, encoding="utf-8")
    return t


class AgentListener:
    def __init__(self, supervisor: AgentSupervisor, token: str, port: int):
        self.supervisor = supervisor
        self.token = token
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "EV"
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:       # noqa: A002  (the stdlib's own name)
                pass

            def _send(self, code: int, body: dict[str, Any] | None = None) -> None:
                data = json.dumps(body).encode() if body is not None else b""
                self.send_response(code)
                if body is not None:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                if code >= 400:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def do_POST(self) -> None:                                     # noqa: N802
                if self.client_address[0] != LOOPBACK or self.headers.get("Host") not in (f"{LOOPBACK}:{owner.port}", f"localhost:{owner.port}"):
                    return self._send(403)
                got = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                if not hmac.compare_digest(owner.token.encode(), got.encode()):
                    return self._send(401)
                parts = self.path.strip("/").split("/")
                if len(parts) != 3 or parts[:2] != ["v1", "permission"] or parts[2] not in ("claude", "codex"):
                    return self._send(404)
                if "json" not in (self.headers.get("Content-Type") or ""):
                    return self._send(415)
                try:
                    n = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    return self._send(400)
                if not 0 < n <= MAX_BODY:
                    return self._send(413 if n > MAX_BODY else 400)
                try:
                    payload = json.loads(self.rfile.read(n))
                except ValueError:
                    return self._send(400)
                if not isinstance(payload, dict):
                    return self._send(400)
                try:
                    out = owner.supervisor.decide(parts[2], payload)
                except Exception:                                          # noqa: BLE001  (never take the agent down with us)
                    log.exception("deciding an agent request failed")
                    out = None
                self._send(200 if out else 204, out)

            def do_GET(self) -> None:                                      # noqa: N802
                self._send(405)

        class Quiet(ThreadingHTTPServer):
            def handle_error(self, request: Any, client_address: Any) -> None:       # a dropped connection is not news
                log.debug("agent listener: connection from %s dropped", client_address)

        try:
            self._server = Quiet((LOOPBACK, self.port), Handler)
        except OSError as e:
            log.error("agent listener could not bind %s:%s (%s); agent approvals are off", LOOPBACK, self.port, e)
            return False
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])          # a test may ask for port 0
        self._thread = threading.Thread(target=self._server.serve_forever, name="agent-listener", daemon=True)
        self._thread.start()
        log.info("agent approvals listening on %s:%s", LOOPBACK, self.port)
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
