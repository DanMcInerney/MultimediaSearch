"""How a decider's backend reaches its model (DESIGN.md §13.3): `Worker`, a process mms starts that holds the model and
speaks the frames of `worker/wire.py` over its stdin and stdout, and `Served`, a model the owner serves on the loopback
interface over HTTP. A model that fails to fit what it is asked raises `ValueError`; one that cannot answer raises
`ApiError("decider")`.
"""

from __future__ import annotations

import collections
import http.client
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from mms import children, config
from mms.errors import ApiError
from mms.worker import wire

LOAD_S, DECIDE_S = 900.0, 120.0  # seconds a load and a decision may take before the worker is stopped
QUIT_S = 10.0  # seconds a worker has to exit after `quit`, before it is killed
STDERR_LINES = 40  # lines of a worker's stderr kept to say why it died
REFUSED_S = 3.0  # seconds a server may refuse connections before `Served.load` gives up
POLL_S = 1.0  # between `Served.load`'s asks
SERVE = Path(__file__).with_name("worker") / "serve.py"
CHILD_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
             "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "DO_NOT_TRACK": "1",
             "VLLM_NO_USAGE_STATS": "1"}  # a worker never goes to the network
# What a worker takes of the daemon's environment: the system's, the user's folders, CUDA and the caches. Nothing else, as
# the model's code runs there: no token, key or proxy (matched in upper case, as Windows has no other).
ENV_NAMES = frozenset({
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
    "HOMEDRIVE", "HOMEPATH", "USERNAME", "USER", "LOGNAME", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "LANG",
    "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS"})
ENV_PREFIXES = ("LC_", "CUDA_", "NVIDIA_", "PYTORCH_")


def child_env() -> dict:
    """The environment of a worker: the allowed part of the daemon's, and `CHILD_ENV`."""
    kept = {name: value for name, value in os.environ.items()
            if name.upper() in ENV_NAMES or name.upper().startswith(ENV_PREFIXES)}
    return {**kept, **CHILD_ENV}


class Worker:
    """A worker process holding one model: started by `load`, ended by `unload` or by a failure. After a failure the
    caller loads it again (through the backend's own `load`, which checks what the worker reports). Calls are one at a
    time. `python` is the env's interpreter and `family_file` the `worker/<family>.py` it runs; `options` go to that
    file's `load`."""

    serve = SERVE  # the child's script

    def __init__(self, python: Path, family_file: Path, *, name: str, folder: Path, options: dict | None = None,
                 load_s: float = LOAD_S, decide_s: float = DECIDE_S):
        self.python, self.family_file = Path(python), Path(family_file)
        self.name, self.folder, self.options = name, Path(folder), dict(options or {})
        self.load_s, self.decide_s = load_s, decide_s
        self.facts: dict = {}  # what the family's `load` reported
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._frames: queue.Queue | None = None
        self._tail: collections.deque = collections.deque(maxlen=STDERR_LINES)
        self._threads: list[threading.Thread] = []
        self._loaded = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def load(self) -> dict:
        """Start the worker and load the model; the family's facts."""
        with self._lock:
            return self._load()

    def decide(self, state, questions: dict, media: list) -> dict:
        """The model's answers to a window. A worker that is not loaded (it failed, or was never loaded) is an error."""
        descriptors, blobs = wire.encode_media(media)
        with self._lock:
            if not self._loaded:
                raise ApiError("decider", f"{self.name}'s worker is not loaded")
            try:
                reply = self._call({"op": "decide", "state": state, "questions": questions, "media": descriptors},
                                   blobs, self.decide_s, "decided")
            except wire.WireError as error:  # too big to send: nothing was written, and the worker is as it was
                raise ValueError(str(error)) from None
            return reply["answers"]

    def stats(self) -> dict:
        """The worker's GPU memory in bytes (`max_memory_allocated`, `memory_allocated`), None where it has no torch."""
        with self._lock:
            if self._proc is None:
                return {"max_memory_allocated": None, "memory_allocated": None}
            reply = self._call({"op": "stats"}, [], self.decide_s, "counted its memory")
            return {key: value for key, value in reply.items() if key != "ok"}

    def unload(self) -> None:
        """End the worker: `quit`, then a kill after QUIT_S. The OS frees all its memory."""
        free = self._lock.acquire(timeout=QUIT_S)
        try:
            proc = self._proc
            if proc is None:
                return
            if not free:  # a call is stuck on it: killing the worker ends that call, which then cleans up
                proc.kill()
                return
            try:
                wire.write_frame(proc.stdin, {"op": "quit"})
                proc.wait(QUIT_S)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._finish()
        finally:
            if free:
                self._lock.release()

    # the process

    def _load(self) -> dict:
        if self._proc is None:
            self._start()
        try:
            reply = self._call({"op": "load", "folder": str(self.folder), "options": self.options}, [], self.load_s,
                               "loaded")
        except BaseException:
            self._finish()  # a worker that could not load holds nothing worth keeping
            raise
        self._loaded, self.facts = True, reply["facts"]
        return self.facts

    def _start(self) -> None:
        try:
            proc = children.popen([str(self.python), "-I", "-X", "utf8", str(self.serve), "--family-file",
                                   str(self.family_file)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=child_env())
        except OSError as error:
            raise ApiError("decider", f"{self.name}'s worker could not start: {error}") from None
        frames, tail = queue.Queue(), collections.deque(maxlen=STDERR_LINES)
        threads = [threading.Thread(target=_read_frames, args=(proc.stdout, frames), daemon=True,
                                    name=f"{self.name}-frames"),
                   threading.Thread(target=_copy_stderr, args=(proc.stderr, tail, self.name), daemon=True,
                                    name=f"{self.name}-stderr")]
        for thread in threads:
            thread.start()
        self._proc, self._frames, self._tail, self._threads = proc, frames, tail, threads

    def _call(self, header: dict, blobs: list, limit: float, doing: str) -> dict:
        """Send one frame and wait for its reply, for at most `limit` seconds. Any failure but an error reply of the
        model ends the worker."""
        proc, frames = self._proc, self._frames
        settle, state = threading.Lock(), {"taken": False, "killed": False}

        def stop() -> None:
            with settle:
                if state["taken"]:  # the reply is in hand: it is not thrown away
                    return
                state["killed"] = True
                proc.kill()

        timer = threading.Timer(limit, stop)
        timer.daemon = True
        timer.start()
        try:
            try:
                wire.write_frame(proc.stdin, header, blobs)
            except OSError:
                pass  # the worker is gone: its end is reported from the stream below
            try:
                frame = frames.get(timeout=limit + 5)
            except queue.Empty:
                stop()
                frame = None
            with settle:
                state["taken"] = True
        finally:
            timer.cancel()
        if state["killed"] and not isinstance(frame, tuple):  # a reply that came before the kill still counts
            self._finish()
            raise ApiError("decider", f"{self.name}'s worker gave no answer in {limit:g} s, so it was stopped")
        if frame is None:
            code, tail = self._finish()
            raise ApiError("decider", f"{self.name}'s worker exited with code {code} while it {doing}: "
                                      + ("\n".join(tail) or "(it wrote nothing to stderr)"))
        if isinstance(frame, wire.WireError):
            self._finish()
            if frame.found is not None:
                raise ApiError("decider", f"{self.name}'s worker wrote something that is not a frame: "
                                          f"{frame.found[:16]!r}")
            raise ApiError("decider", f"{self.name}'s worker broke the frame format: {frame}")
        reply = frame[0]
        if reply.get("ok") is not True:
            if reply.get("error") == "unfit":
                raise ValueError(reply.get("message", ""))
            raise ApiError("decider", f"{self.name}: {reply.get('type')}: {reply.get('message')}")
        return reply

    def _finish(self) -> tuple[int | None, list[str]]:
        """Make sure the worker is gone and forget it; its exit code and the last lines of its stderr."""
        proc, threads, tail = self._proc, self._threads, self._tail
        self._proc, self._frames, self._threads, self._loaded = None, None, [], False
        if proc is None:
            return None, list(tail)
        try:
            proc.wait(1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for thread in threads:  # the pipes end with the process, and the last lines are in `tail` once they have
            thread.join(2)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if not any(thread.is_alive() for thread in threads):
                try:
                    stream.close()
                except OSError:
                    pass
        return proc.returncode, list(tail)


def _read_frames(stream, frames: queue.Queue) -> None:
    """Put each frame the worker writes on `frames`, then a `WireError` if its bytes are not frames, then None."""
    try:
        while (frame := wire.read_frame(stream)) is not None:
            frames.put(frame)
    except wire.WireError as error:
        frames.put(error)
    except (OSError, ValueError):  # the pipe was closed under the read
        pass
    frames.put(None)


def _copy_stderr(stream, tail: collections.deque, name: str) -> None:
    """Keep the worker's last lines of stderr, and copy each to the daemon's own (the log, when it is detached)."""
    try:
        for raw in stream:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            tail.append(line)
            if sys.stderr is not None:
                try:
                    print(f"{name} worker: {line}", file=sys.stderr, flush=True)
                except (OSError, ValueError):
                    pass
    except (OSError, ValueError):
        pass


class Served:
    """A model the owner serves on the loopback interface (H2O). The requests go straight to the host `config.loopback`
    names, so no proxy variable applies and no redirect is followed. `model_id`, when given, is what `/health` must
    name as the model, and `hint` is what `load` tells the caller to do when nothing answers. The server and its memory
    are the owner's: `unload` does nothing."""

    def __init__(self, url: str, *, name: str, model_id: str | None = None, hint: str | None = None,
                 load_s: float = LOAD_S, decide_s: float = DECIDE_S):
        self.url, self.name, self.model_id, self.hint = url, name, model_id, hint
        self.load_s, self.decide_s = load_s, decide_s
        self.host, self.port = config.loopback(url)

    def load(self) -> dict:
        """Wait until `/health` answers 200 (a server that is warming says 503), for at most `load_s`; its JSON."""
        end, refused = time.monotonic() + self.load_s, None
        while True:
            try:
                status, location, text = self._request("GET", "/health", None, 5.0)
            except ConnectionRefusedError:
                refused = refused or time.monotonic()
                if time.monotonic() - refused >= REFUSED_S:
                    raise ApiError("decider", f"{self.name}: nothing answers at {self.url}",
                                   hint=self.hint) from None
            except (ConnectionResetError, ConnectionAbortedError):  # reset or hung up on: it may still be starting
                refused = None
            except (OSError, http.client.HTTPException) as error:  # a server that is broken says so now, not at the deadline
                raise ApiError("decider", f"{self.name}'s server at {self.url} failed: {type(error).__name__}: {error}",
                               hint="is it the model's server?") from None
            else:
                refused = None
                if status == 200:
                    return self._health(text)
                if status != 503:
                    raise self._failure(status, location, text)
            if time.monotonic() + POLL_S > end:
                raise ApiError("decider", f"{self.name}'s server at {self.url} was not ready in {self.load_s:g} s")
            time.sleep(POLL_S)

    def probe(self, timeout_s: float) -> int | None:
        """The status `GET /health` gives, or None when the server does not answer."""
        try:
            return self._request("GET", "/health", None, timeout_s)[0]
        except (OSError, http.client.HTTPException):
            return None

    def post(self, path: str, body: dict) -> dict:
        """POST a JSON body and return the JSON reply. 400 and 422 mean the input does not fit the model."""
        try:
            status, location, text = self._request("POST", path, body, self.decide_s)
        except TimeoutError:
            raise ApiError("decider", f"{self.name}'s server gave no answer in {self.decide_s:g} s") from None
        except (OSError, http.client.HTTPException) as error:
            raise ApiError("decider", f"{self.name}'s server at {self.url} failed: {type(error).__name__}: {error}",
                           hint="is the server running?") from None
        if status == 200:
            try:
                return json.loads(text)
            except ValueError:
                raise ApiError("decider", f"{self.name}'s server answered something that is not JSON: "
                                          f"{text[:200]!r}") from None
        if status in (400, 422):
            raise ValueError(_detail(text))
        raise self._failure(status, location, text)

    def unload(self) -> None:
        pass

    def _request(self, method: str, path: str, body: dict | None, timeout: float) -> tuple[int, str | None, str]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            headers = {} if body is None else {"Content-Type": "application/json"}
            connection.request(method, path, None if body is None else json.dumps(body).encode("utf-8"), headers)
            response = connection.getresponse()
            return response.status, response.getheader("Location"), response.read().decode("utf-8", "replace")
        finally:
            connection.close()

    def _health(self, text: str) -> dict:
        try:
            health = json.loads(text)
        except ValueError:
            raise ApiError("decider", f"{self.name}'s server at {self.url} answered /health with something that is "
                                      f"not JSON: {text[:200]!r}") from None
        served = health.get("model") if isinstance(health, dict) else None
        if self.model_id is not None and served not in (None, self.model_id):
            raise ApiError("decider", f"the server at {self.url} serves {served}, not {self.model_id}")
        return health

    def _failure(self, status: int, location: str | None, text: str) -> ApiError:
        if 300 <= status < 400:
            return ApiError("decider", f"{self.name}'s server at {self.url} redirected to {location}, which mms does "
                                       "not follow")
        return ApiError("decider", f"{self.name}'s server answered {status}: {_detail(text)}")


def _detail(text: str) -> str:
    """The reason in an error body: its `detail` (or `error`, `message`) when it is JSON, else the text itself."""
    try:
        body = json.loads(text)
    except ValueError:
        return text.strip()[:500]
    if isinstance(body, dict):
        for key in ("detail", "error", "message"):
            if key in body:
                body = body[key]
                break
    return (body if isinstance(body, str) else json.dumps(body))[:500]
