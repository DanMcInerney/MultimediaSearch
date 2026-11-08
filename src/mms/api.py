"""The daemon's HTTP API (DESIGN.md §11): a server on 127.0.0.1 and a port the OS picks, `daemon.json` with the token, and
the dispatch of `POST /v1/<op>` to `ops` and `jobs`. Calling it is client.py's."""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mms import ops
from mms.client import FILE
from mms.config import iso
from mms.errors import ApiError
from mms.jobs import Jobs

log = logging.getLogger("mms.api")

API_VERSION = 1
BODY_MAX = 32 << 20  # a body larger than this is refused


class Api:
    def __init__(self, env: ops.Env, jobs: Jobs, started: float):
        self.env, self.jobs, self.started = env, jobs, started
        self.token = secrets.token_hex(32)
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.file = env.home / FILE

    def start(self) -> int:
        """Bind, serve, then write `daemon.json`: the file appears once the API answers. Returns the port."""
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def do_POST(self):
                api.handle(self)

            def do_GET(self):
                api.handle(self)

            do_PUT = do_DELETE = do_PATCH = do_GET

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, name="api", daemon=True)
        self.thread.start()
        info = {"api": API_VERSION, "pid": os.getpid(), "port": port, "token": self.token, "started": iso(self.started)}
        part = self.file.with_name(FILE + ".part")
        part.write_text(json.dumps(info), encoding="utf-8")
        os.replace(part, self.file)
        return port

    def stop(self) -> None:
        """Remove `daemon.json` and close the server."""
        self.file.unlink(missing_ok=True)
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(5)

    # one request

    def handle(self, request: BaseHTTPRequestHandler) -> None:
        try:
            length = int(request.headers.get("Content-Length") or 0)
            body = request.rfile.read(length) if 0 <= length <= BODY_MAX else b""
            if not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {self.token}"):
                raise ApiError("unauthorized", "no token, or not this daemon's", hint="read daemon.json again")
            if length > BODY_MAX:
                raise ApiError("bad_request", f"the body is over {BODY_MAX >> 20} MB")
            path = request.path.split("?")[0]
            if request.command != "POST" or not path.startswith("/v1/") or not path[4:]:
                raise ApiError("not_found", f"no op {path!r}: POST /v1/<op>", field="op")
            try:
                parsed = json.loads(body.decode("utf-8")) if body else {}
            except ValueError:
                raise ApiError("bad_request", "the body is JSON") from None
            op = path[4:]
            result = self.jobs.call(op, parsed) if op.startswith("jobs/") else ops.call(self.env, op, parsed)
            self._send(request, 200, result)
        except ApiError as error:
            self._send(request, error.status, {"error": error.body()})
        except Exception as error:
            log.error("%s %s failed:\n%s", request.command, request.path, traceback.format_exc())
            self._send(request, 500, {"error": {"code": "internal", "message": f"{type(error).__name__}: {error}"}})

    @staticmethod
    def _send(request: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request.send_response(status)
        request.send_header("Content-Type", "application/json; charset=utf-8")
        request.send_header("Content-Length", str(len(data)))
        request.end_headers()
        request.wfile.write(data)
