"""The daemon API's client (DESIGN.md §11) against a real daemon: discovery, calls, errors, and starting a daemon."""

from __future__ import annotations

import http.server
import json
import os
import socket
import sqlite3
import threading
import time

import pytest

from mms import client
from mms.errors import ApiError
from mms.store import Store


def test_call_returns_the_result_and_sends_the_token_and_the_caller(home, memory):
    caller = {"session": "s-0123456789ab", "project": "id:demo1234", "project_name": "demo", "project_root": "/x",
              "client": "test"}
    assert client.call(home, "remember", {"note": "the kettle"}, caller=caller)["stored"][0]["type"] == "record"
    store = Store(home)
    try:
        (row,) = store.records(sources=["agent"])
    finally:
        store.close()
    assert (row.session, row.project) == ("s-0123456789ab", "id:demo1234"), "the daemon had the caller"


def test_connect_finds_a_running_daemon_and_nothing_else(home, memory):
    found = client.connect(home)
    assert found == client.read(home) and found.pid == os.getpid(), "the in-process daemon's"
    times = []
    for _ in range(5):
        began = time.perf_counter()
        client.connect(home)
        times.append(time.perf_counter() - began)
    assert sorted(times)[2] < 0.1, f"a healthy daemon is one HTTP call, not the 0.3 s lock probe: {times}"
    (home / "daemon.json").unlink()
    assert client.connect(home) is None and client.read(home) is None


def test_a_stale_file_is_no_daemon_whether_the_port_refuses_or_the_token_is_wrong(home, memory):
    good = json.loads((home / "daemon.json").read_text())
    (home / "daemon.json").write_text(json.dumps({**good, "token": "0" * 64}))
    with pytest.raises(client.DaemonUnavailable, match="stale"):
        client.call(home, "status")
    assert client.connect(home) is None
    with socket.socket() as closed:  # a port nobody listens on
        closed.bind(("127.0.0.1", 0))
        nobody = closed.getsockname()[1]
    (home / "daemon.json").write_text(json.dumps({**good, "port": nobody}))
    with pytest.raises(client.DaemonUnavailable, match="refused"):
        client.call(home, "status")
    assert client.connect(home) is None


def test_no_daemon_file_is_unavailable_and_says_where_it_looked(home):
    with pytest.raises(client.DaemonUnavailable, match="daemon.json") as caught:
        client.call(home, "status")
    assert caught.value.body()["code"] == "daemon_unavailable" and caught.value.body()["hint"]
    assert isinstance(caught.value, ApiError) and caught.value.status is None, "one error type, and nothing was sent"


def test_an_error_keeps_its_code_status_field_and_hint(home, memory):
    with pytest.raises(ApiError) as caught:
        client.call(home, "find", {"bogus": 1})
    error = caught.value
    assert (error.code, error.status, error.field) == ("bad_request", 400, "bogus") and not isinstance(error, client.DaemonUnavailable)
    assert error.body() == {"code": "bad_request", "message": error.message, "field": "bogus"}
    with pytest.raises(ApiError) as caught:
        client.call(home, "look", {"source": "nothing", "t": "now"})
    error = caught.value
    assert (error.code, error.status, error.field) == ("not_found", 404, "source") and "holds nothing" in error.hint


def test_a_reply_that_is_not_the_api_is_a_failure_not_a_result(home):
    class Wrong(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = b"<html>not json</html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Wrong)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (home / "daemon.json").write_text(json.dumps({"api": 1, "pid": os.getpid(), "port": server.server_address[1],
                                                  "token": "t" * 64}))
    try:
        with pytest.raises(client.DaemonFailed, match="not JSON"):
            client.call(home, "status")
    finally:
        server.shutdown()


def test_ensure_leaves_a_running_daemon_alone(home, memory):
    before = client.read(home)
    assert client.ensure(home) == before and client.read(home) == before, "no second daemon, no new address"


def test_ensure_starts_a_daemon_that_answers_for_the_home(home, detached):
    found = client.ensure(home)
    assert found.pid != os.getpid() and client.call(home, "status")["home"] == str(home)
    assert client.ensure(home) == found, "the second call starts nothing"


def test_ensure_fails_loudly_with_the_daemons_log(home):
    db = sqlite3.connect(home / "index.sqlite")  # an index of another version: the daemon refuses to start
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
    db.commit()
    db.close()
    with pytest.raises(client.DaemonUnavailable, match=r"exited 1:\nmms: the daemon exited; daemon.log ends:\n.*another version"):
        client.ensure(home)


def test_ensure_gives_up_after_its_wait(home):
    started = time.monotonic()
    with pytest.raises(client.DaemonUnavailable, match="did not return within 0.01 s"):
        client.ensure(home, wait_s=0.01)
    assert time.monotonic() - started < 20 and not (home / "daemon.json").exists()


class Foreign(http.server.BaseHTTPRequestHandler):
    """Another program's server, on a port a dead daemon's `daemon.json` still names."""

    body = b"<html>not the API</html>"

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):
        pass


def stale_file_of(home, handler):
    """A `daemon.json` no daemon holds the lock for, naming a port another server answers on."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    (home / "daemon.json").write_text(json.dumps({"api": 1, "pid": 1, "port": server.server_address[1], "token": "t" * 64}))
    return server


@pytest.mark.parametrize("body", [Foreign.body, b'{"looks": "like the API"}'], ids=["html", "json"])
def test_a_free_lock_is_no_daemon_and_its_stale_file_goes_whatever_answers_on_its_port(home, body):
    server = stale_file_of(home, type("Other", (Foreign,), {"body": body}))
    try:
        assert client.connect(home) is None
        assert not (home / "daemon.json").exists(), "the stale file is deleted, under the lock"
        assert client.is_running(home) is False
    finally:
        server.shutdown()


def test_ensure_starts_a_daemon_although_a_stale_file_names_a_port_another_server_uses(home, detached):
    """Judged by HTTP, this `daemon.json` made every `ensure` fail for good: the reply was not the API's."""
    server = stale_file_of(home, Foreign)
    try:
        found = client.ensure(home)
        assert found.pid != os.getpid() and client.read(home) == found and client.call(home, "status")["running"]
    finally:
        server.shutdown()
