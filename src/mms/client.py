"""The daemon API's client (DESIGN.md §11): finds the daemon through `daemon.json`, makes one call, and starts a daemon
when none runs. It is the one way into the daemon: the MCP adapter, the CLI and `mms start --detach` all call it.

It keeps no state: every call reads `daemon.json` again, so a daemon that restarted is found at the next call. Whether
a daemon is alive is the lock's to say (`daemon.lock`, which the operating system drops when a daemon dies); HTTP says
only whether a live one is ready.
"""

from __future__ import annotations

import http.client
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from mms.errors import ApiError
from mms.lock import HomeLock

FILE = "daemon.json"  # the daemon's address, which it writes and this reads
LOCK = "daemon.lock"  # held by a daemon for as long as it lives
CONNECT_TIMEOUT_S = 5.0  # `connect` waits this long for a daemon's `status` before it asks the lock
PROBE_WAIT_S = 0.3  # a probe waits this long: it holds the lock for microseconds, so a lock still busy is a daemon's
START_WAIT_S = 30.0  # `mms start --detach` waits this long for the daemon to answer
CALL_TIMEOUT_S = 60.0
TAIL_LINES = 20  # of a failed start's output, in the error


class DaemonUnavailable(ApiError):
    """Nothing reached a daemon: no `daemon.json`, a refused connection, or a token that is not this daemon's. The
    request was not processed, so asking again after `ensure` is safe."""

    def __init__(self, message: str):
        super().__init__("daemon_unavailable", message, hint="the owner can start it with `mms start --detach`")


class DaemonFailed(ApiError):
    """A daemon was reached but no usable answer came back (no reply in time, a dropped connection, a reply that is
    not the API's). The request may have been processed, so it is not asked again."""

    def __init__(self, message: str):
        super().__init__("daemon_failed", message)


@dataclass(frozen=True)
class Endpoint:
    port: int
    token: str
    pid: int


def read(home: Path) -> Endpoint | None:
    """What `daemon.json` says, or None when there is no such file (no daemon, or one that stopped cleanly)."""
    file = Path(home) / FILE
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
        return Endpoint(int(raw["port"]), str(raw["token"]), int(raw["pid"]))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:  # a file the daemon writes whole, so this is a bug
        raise DaemonFailed(f"{file} is not a daemon's address file: {error!r}") from error


def is_running(home: Path) -> bool:
    """Whether a daemon holds this home's lock (a daemon that is still starting does)."""
    return _probe(Path(home), free=lambda: None)


def _probe(home: Path, free) -> bool:
    """Take the lock to look: True when a daemon holds it. When it is free, `free()` runs while the probe holds it, so
    no daemon starts in between."""
    if not home.is_dir():
        free()
        return False
    probe = HomeLock(home / LOCK)
    if not probe.acquire(PROBE_WAIT_S):
        return True
    try:
        free()
    finally:
        probe.release()
    return False


def connect(home: Path) -> Endpoint | None:
    """The endpoint of a daemon that is running and ready, or None.

    A daemon is alive when `daemon.json` names a port that answers `status` as that daemon (its token, its pid): that
    costs one call, and no lock. Whatever else happens (no file, refused, reset, timeout, 401, a reply that is not
    this daemon's) the lock decides. A free lock means no daemon, and the `daemon.json` left by one that died is deleted
    under it. A held lock is a daemon that is starting or wedged: not ready, and the caller waits (`start_detached`).
    """
    home = Path(home)
    try:
        endpoint = read(home)
        if endpoint is not None:
            reply = call(home, "status", {}, timeout_s=CONNECT_TIMEOUT_S)
            if reply.get("running") is True and reply.get("pid") == endpoint.pid:
                return endpoint
    except (DaemonUnavailable, DaemonFailed):
        pass
    _probe(home, free=lambda: (home / FILE).unlink(missing_ok=True))
    return None


def call(home: Path, op: str, params: dict | None = None, *, caller: dict | None = None,
         timeout_s: float = CALL_TIMEOUT_S) -> dict:
    """POST one op and return its result. Raises `ApiError` for an error the daemon answered with, `DaemonUnavailable`
    when none was reached and `DaemonFailed` when none answered usefully."""
    endpoint = read(home)
    if endpoint is None:
        raise DaemonUnavailable(f"no daemon is running for {home} ({Path(home) / FILE} is missing)")
    body = dict(params or {})
    if caller is not None:
        body["caller"] = caller
    connection = http.client.HTTPConnection("127.0.0.1", endpoint.port, timeout=timeout_s)
    try:
        connection.request("POST", f"/v1/{op}", json.dumps(body).encode("utf-8"),
                           {"Authorization": f"Bearer {endpoint.token}", "Content-Type": "application/json"})
        reply = connection.getresponse()
        status, raw = reply.status, reply.read()
    except ConnectionRefusedError as error:
        raise DaemonUnavailable(f"the daemon in {Path(home) / FILE} (port {endpoint.port}) refused the "
                                "connection: the file is stale") from error
    except TimeoutError as error:
        raise DaemonFailed(f"{op}: no reply within {timeout_s:g} s") from error
    except (OSError, http.client.HTTPException) as error:
        raise DaemonFailed(f"{op}: {error!r}") from error
    finally:
        connection.close()
    try:
        payload = json.loads(raw)
    except ValueError as error:
        raise DaemonFailed(f"{op}: HTTP {status} with a reply that is not JSON: {raw[:200]!r}") from error
    if status == 200 and isinstance(payload, dict):
        return payload
    error = payload.get("error") if isinstance(payload, dict) else None
    if not (isinstance(error, dict) and isinstance(error.get("code"), str)):
        raise DaemonFailed(f"{op}: HTTP {status} with a reply that is not the API's: {raw[:200]!r}")
    if status == 401:
        raise DaemonUnavailable(f"the daemon refused the token in {Path(home) / FILE}: the file is stale")
    raise ApiError(error["code"], str(error.get("message", "")), error.get("field"), error.get("hint"))


def wait_until_answering(home: Path, timeout_s: float, alive=lambda: True) -> dict | None:
    """The daemon's `status` once it answers, polling until `timeout_s` or until `alive()` is false."""
    end = time.monotonic() + timeout_s
    while alive():
        try:
            return call(home, "status", timeout_s=5)
        except (DaemonUnavailable, DaemonFailed):
            pass
        if time.monotonic() > end:
            return None
        time.sleep(0.1)
    return None


def ensure(home: Path, *, wait_s: float = START_WAIT_S) -> Endpoint:
    """The running daemon's endpoint, starting one with `mms start --detach` (which returns once it answers) if none
    runs."""
    found = connect(home)
    if found is not None:
        return found
    command = [sys.executable, "-m", "mms.cli", "--home", str(home), "start", "--detach"]
    shown = " ".join(command)
    with tempfile.TemporaryFile() as output:  # a file, not a pipe: a detached daemon must not hold our pipe open
        try:
            done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                  timeout=wait_s)
        except subprocess.TimeoutExpired as error:
            raise DaemonUnavailable(f"`{shown}` did not return within {wait_s:g} s") from error
        except OSError as error:
            raise DaemonUnavailable(f"could not run `{shown}`: {error!r}") from error
        output.seek(0)
        tail = "\n".join(output.read().decode("utf-8", "replace").splitlines()[-TAIL_LINES:])
    if done.returncode != 0:
        raise DaemonUnavailable(f"`{shown}` exited {done.returncode}:\n{tail}")
    found = connect(home)
    if found is None:
        raise DaemonUnavailable(f"`{shown}` exited 0 but no daemon answers for {home}:\n{tail}")
    return found
