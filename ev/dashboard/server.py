"""The dashboard's web server: stdlib only, 127.0.0.1 only, one browser at a time is the expected use.

How it is locked down (see docs/SECURITY.md):
* Binds the loopback address; any other peer address is dropped.
* EV opens the browser at /?t=<one-time token>. The token works once and for two minutes; the server trades it for an
  HttpOnly, SameSite=Strict session cookie and redirects, so the token never stays in the address bar or the history.
  Every other request needs that cookie.
* The Host header must be exactly this server's (stops DNS rebinding). A POST also needs an Origin of this server, a JSON
  content type and a body under 64 KB (stops cross-site requests even if a cookie were somehow sent).
* The page is static, served with a strict Content-Security-Policy (no inline script, no external anything), and it puts
  every piece of text into the page as text, never as markup.
* There is no route that writes policy.toml or any file other than ev.local.toml (through the allow-list in settings_schema)
  and the skills folder (through SkillStore).
"""
from __future__ import annotations

import hmac
import http.cookies
import json
import queue
import secrets
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from ev.core.log import get_logger
from ev.dashboard.hub import Hub

log = get_logger("dashboard")

STATIC = Path(__file__).parent / "static"
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/app.css": ("app.css", "text/css; charset=utf-8"), "/orb.js": ("orb.js", "text/javascript; charset=utf-8")}
REPLAY = ("state", "turn", "tool", "proactive")      # what a page that has just connected is caught up on (never audio levels)
MAX_BODY = 64 * 1024
TOKEN_TTL_S = 120
COOKIE = "ev_dash"
HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; "
                               "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin", "X-Frame-Options": "DENY",
}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """A browser that hangs up mid-response is normal; log it at debug instead of printing a traceback to the console."""
        log.debug("connection from %s ended badly", client_address, exc_info=True)


class DashboardError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Backend(Protocol):
    """What the server asks of EV. `AppBackend` (backend.py) is the real one; tests use a fake."""

    def status(self) -> dict[str, Any]: ...
    def control(self, action: str) -> dict[str, Any]: ...
    def settings(self) -> dict[str, Any]: ...
    def apply_settings(self, changes: dict[str, Any]) -> dict[str, Any]: ...
    def voices(self) -> dict[str, Any]: ...
    def set_voice(self, engine: str, voice: str, speed: float, preview: bool, save: bool) -> dict[str, Any]: ...
    def models(self) -> dict[str, Any]: ...
    def set_model(self, model: str, save: bool) -> dict[str, Any]: ...
    def snooze(self, minutes: float) -> dict[str, Any]: ...
    def skills(self) -> dict[str, Any]: ...
    def run_skill(self, name: str) -> dict[str, Any]: ...
    def save_skill(self, spec: dict[str, Any]) -> dict[str, Any]: ...
    def delete_skill(self, name: str) -> dict[str, Any]: ...
    def facts(self) -> dict[str, Any]: ...
    def search(self, query: str, when: str) -> dict[str, Any]: ...
    def forget(self, key: str) -> dict[str, Any]: ...
    def keep(self, ident: str) -> dict[str, Any]: ...
    def policy_text(self) -> dict[str, Any]: ...


class DashboardServer:
    def __init__(self, backend: Backend, hub: Hub, *, port: int = 0, clock: Callable[[], float] = time.monotonic):
        self.backend, self.hub, self.clock = backend, hub, clock
        self._want_port = port
        self._tokens: dict[str, float] = {}
        self._sessions: set[str] = set()
        self._lock = threading.Lock()
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.port = 0

    # ---- lifecycle
    def start(self) -> int:
        server = self

        class Handler(_Handler):
            owner = server

        self._httpd = _Server(("127.0.0.1", self._want_port), Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="dashboard", daemon=True)
        self._thread.start()
        log.info("dashboard listening on 127.0.0.1:%d", self.port)
        return self.port

    def stop(self) -> None:
        self.stopping.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # ---- sign-in
    def login_url(self) -> str:
        """A fresh one-time URL. It is what EV opens in the browser; it is worthless after one use or two minutes."""
        token = secrets.token_urlsafe(24)
        with self._lock:
            now = self.clock()
            self._tokens = {t: exp for t, exp in self._tokens.items() if exp > now}
            self._tokens[token] = now + TOKEN_TTL_S
        return f"http://127.0.0.1:{self.port}/?t={token}"

    def _redeem(self, token: str) -> str | None:
        with self._lock:
            for t, exp in list(self._tokens.items()):
                if hmac.compare_digest(t, token):
                    del self._tokens[t]
                    if exp > self.clock():
                        session = secrets.token_urlsafe(32)
                        self._sessions.add(session)
                        return session
            return None

    def _has_session(self, cookie: str) -> bool:
        with self._lock:
            return any(hmac.compare_digest(s, cookie) for s in self._sessions)

    def allowed_hosts(self) -> set[str]:
        return {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}


class _Handler(BaseHTTPRequestHandler):
    owner: DashboardServer
    server_version = "EV"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:         # EV's own log, not stderr
        log.debug("http " + fmt, *args)

    # ---- plumbing
    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in {**HEADERS, **(extra or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

    def _deny(self, status: int, why: str) -> None:
        self._json(status, {"error": why})

    def _session_ok(self) -> bool:
        raw = self.headers.get("Cookie", "")
        try:
            jar = http.cookies.SimpleCookie(raw)
        except http.cookies.CookieError:
            return False
        morsel = jar.get(COOKIE)
        return bool(morsel and morsel.value and self.owner._has_session(morsel.value))

    def _common_checks(self) -> bool:
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            self._deny(403, "loopback only")
            return False
        if (self.headers.get("Host") or "").lower() not in self.owner.allowed_hosts():
            self._deny(403, "bad host")
            return False
        return True

    # ---- GET
    def do_GET(self) -> None:                                    # noqa: N802
        if not self._common_checks():
            return
        url = urlsplit(self.path)
        if url.path == "/" and "t" in parse_qs(url.query):
            session = self.owner._redeem(parse_qs(url.query)["t"][0])
            if session is None:
                return self._deny(403, "that sign-in link has been used or has expired; ask EV to open the dashboard again")
            self.send_response(302)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", f"{COOKIE}={session}; HttpOnly; SameSite=Strict; Path=/")
            for k, v in HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if not self._session_ok():
            return self._deny(401, "not signed in; ask EV to open the dashboard")
        if url.path in FILES:
            name, ctype = FILES[url.path]
            try:
                return self._send(200, (STATIC / name).read_bytes(), ctype)
            except OSError:
                return self._deny(404, "missing")
        if url.path == "/api/events":
            return self._events()
        routes: dict[str, Callable[[], dict[str, Any]]] = {
            "/api/status": self.owner.backend.status, "/api/settings": self.owner.backend.settings,
            "/api/voices": self.owner.backend.voices, "/api/models": self.owner.backend.models,
            "/api/skills": self.owner.backend.skills, "/api/facts": self.owner.backend.facts,
            "/api/policy": self.owner.backend.policy_text}
        if url.path in routes:
            return self._call(routes[url.path])
        if url.path == "/api/search":
            q = parse_qs(url.query)
            return self._call(lambda: self.owner.backend.search(q.get("q", [""])[0][:200], q.get("when", [""])[0][:30]))
        self._deny(404, "no such page")

    def _call(self, fn: Callable[[], dict[str, Any]]) -> None:
        try:
            self._json(200, fn())
        except DashboardError as e:
            self._deny(e.status, str(e))
        except Exception:                                        # noqa: BLE001
            log.exception("dashboard request failed")
            self._deny(500, "something went wrong; the log has the details")

    def _events(self) -> None:
        q = self.owner.hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        for k, v in HEADERS.items():
            self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(b": connected\n\n")
            past = sorted((i for kind, items in self.owner.hub.recent().items() if kind in REPLAY for i in items), key=lambda i: i["at"])
            for item in past:        # a reload shows the recent feed, not an empty page (the page ignores any it already has)
                self.wfile.write(f"event: {item['kind']}\ndata: {json.dumps(item, default=str)}\n\n".encode())
            self.wfile.flush()
            while not self.owner.stopping.is_set():
                try:
                    item = q.get(timeout=10)
                    self.wfile.write(f"event: {item['kind']}\ndata: {json.dumps(item, default=str)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (OSError, ValueError):
            pass
        finally:
            self.owner.hub.unsubscribe(q)

    # ---- POST
    def do_POST(self) -> None:                                   # noqa: N802
        if not self._common_checks():
            return
        if not self._session_ok():
            return self._deny(401, "not signed in; ask EV to open the dashboard")
        origins = {f"http://{h}" for h in self.owner.allowed_hosts()}
        if (self.headers.get("Origin") or "") not in origins:
            return self._deny(403, "bad origin")
        if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return self._deny(403, "cross-site request")
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._deny(415, "JSON only")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._deny(400, "bad length")
        if not 0 <= length <= MAX_BODY:
            return self._deny(413, "too big")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            return self._deny(400, "bad JSON")
        if not isinstance(body, dict):
            return self._deny(400, "expected an object")
        b = self.owner.backend
        path = urlsplit(self.path).path
        posts: dict[str, Callable[[], dict[str, Any]]] = {
            "/api/control": lambda: b.control(str(body.get("action", ""))),
            "/api/settings": lambda: b.apply_settings(body.get("changes", {}) if isinstance(body.get("changes"), dict) else {}),
            "/api/voice": lambda: b.set_voice(str(body.get("engine", "")), str(body.get("voice", "")), _num(body.get("speed", 1.0)),
                                              bool(body.get("preview", False)), bool(body.get("save", False))),
            "/api/model": lambda: b.set_model(str(body.get("model", "")), bool(body.get("save", False))),
            "/api/snooze": lambda: b.snooze(_num(body.get("minutes", 0))),
            "/api/skills/run": lambda: b.run_skill(str(body.get("name", ""))),
            "/api/skills/save": lambda: b.save_skill(body),
            "/api/skills/delete": lambda: b.delete_skill(str(body.get("name", ""))),
            "/api/forget": lambda: b.forget(str(body.get("key", ""))),
            "/api/keep": lambda: b.keep(str(body.get("id", ""))),
        }
        if path not in posts:
            return self._deny(404, "no such action")
        self._call(posts[path])


def _num(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise DashboardError("expected a number")
    return float(v)
