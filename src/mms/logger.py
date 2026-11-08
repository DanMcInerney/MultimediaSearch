"""The program log client (DESIGN.md §8.1): `mms.log(stream, data, t=...)` and `frame=`, written to the spool of
§8.2, which the daemon ingests.

`log` copies what it is given, queues it and returns; one background thread writes the batches. Nothing it does ever
waits on the disk or the daemon, or raises for them. A call raises only for a bad stream name, data that is not a JSON
object, or a frame of the wrong shape or type, at the call. Numpy and PIL are imported by the first frame, not before.
"""

from __future__ import annotations

import atexit
import errno
import json
import math
import numbers
import os
import secrets
import shutil
import sys
import threading
import time
import weakref
from datetime import datetime, timezone
from pathlib import Path

from mms import config
from mms import project as projects

QUEUE_MAX_BYTES = 64 * 2**20  # queued data and frame bytes; `log` drops what does not fit
BATCH_S = 0.2  # a batch is written this long after its first line,
BATCH_LINES = 1000  # or at this many lines,
BATCH_FRAME_BYTES = 32 * 2**20  # or at this many bytes of frames
SPOOL_MAX_BYTES = 2 * 2**30  # the thread stops writing while the spool holds more (the daemon is down)
SPOOL_MEASURE_S = 10.0  # the spool's size is measured this often, and counted from what this logger writes between
FLOOR_CHECK_S = 60.0  # `[memory] free_gb` and the free space are read at start and this often
JPEG_QUALITY = 90
CLOSE_WAIT_S = 5.0
DROPS = ("queue_full", "spool_full", "floor", "refused")


class Logger:
    """One program run's logging: a run folder per stream in `home/spool/<stream>/` (§8.2). `project` is the project's
    ID, else the one found from the working directory when the thread first writes."""

    def __init__(self, home: Path | str | None = None, project: str | None = None):
        self.home = Path(home).expanduser() if home else config.default_home()
        self._project = project
        self._project_known = project is not None
        self._project_error: str | None = None  # why the project could not be found, while it cannot
        self._cwd = Path.cwd()
        self._cond = threading.Condition()
        self._queue: list[tuple] = []  # (stream, wall, t, payload: JSON bytes | frame array, size)
        self._queue_bytes = 0  # queued and being written
        self._lines = 0  # queued lines, and bytes of queued frames, since the thread last took the queue
        self._frame_bytes = 0
        self._oldest = 0.0
        self._inflight = 0
        self._flush = False
        self._closing = False
        self._thread: threading.Thread | None = None
        self._written = 0
        self._dropped = dict.fromkeys(DROPS, 0)
        self._write_errors = 0  # lines lost to a write error that was not a full disk
        self._last_error: str | None = None
        self._dropped_by_stream: dict[str, int] = {}
        self._announced: set[str] = set()
        self._runs: dict[str, _Run] = {}
        self._floor_gb: float | None = None
        self._below_floor = False
        self._floor_read = -math.inf
        self._spool_bytes = 0
        self._spool_read = -math.inf

    def log(self, stream: str, data: dict | None = None, *, frame=None, t: float | None = None) -> None:
        """Queue one record (`data`, a JSON object) or one frame (a uint8 array, HxWx3 RGB or HxW grey) of `stream`,
        stamped with the wall clock now, and `t` when the program has a clock of its own. Returns at once."""
        if not isinstance(stream, str) or not config.NAME.match(stream):
            raise ValueError(f"a stream name is letters, digits, '_' or '-', up to 64 characters, not {stream!r}")
        if (data is None) == (frame is None):
            raise ValueError("log takes data (a JSON object) or frame=, one of the two")
        if t is not None:
            if isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(t):
                raise ValueError(f"t is a finite number or None, not {t!r}")
            t = float(t)
        if frame is None:
            if not isinstance(data, dict):
                raise ValueError(f"data is a JSON object (a dict), not {type(data).__name__}")
            try:  # now, so a later change to `data` is not seen
                payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as error:  # UnicodeEncodeError is a ValueError
                raise ValueError(f"data is not JSON: {error}") from error
            size = len(payload)
        else:
            payload = _frame_copy(frame)
            size = payload.nbytes
        wall = time.time()
        with self._cond:
            if self._closing:
                raise RuntimeError("this Logger is closed")
            if self._queue_bytes + size > QUEUE_MAX_BYTES:
                announce = self._count("queue_full", stream, 1)
            else:
                announce = None
                if self._thread is None:
                    self._start()
                if not self._queue:
                    self._oldest = time.monotonic()
                    self._cond.notify()
                self._queue.append((stream, wall, t, payload, size))
                self._queue_bytes += size
                self._lines += 1
                if frame is not None:
                    self._frame_bytes += size
                if self._lines == BATCH_LINES or self._frame_bytes >= BATCH_FRAME_BYTES:
                    self._cond.notify()
        if announce:
            print(announce, file=sys.stderr)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """True once everything queued has been written to the spool or dropped (and counted); False after
        `timeout_s`."""
        end = time.monotonic() + timeout_s
        with self._cond:
            if self._queue:
                self._flush = True
                self._cond.notify_all()
            while self._queue or self._inflight:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    def stats(self) -> dict:
        with self._cond:
            return {"queued": len(self._queue) + self._inflight, "written": self._written,
                    "dropped": dict(self._dropped), "write_errors": self._write_errors, "last_error": self._last_error}

    def close(self, timeout_s: float = CLOSE_WAIT_S) -> bool:
        """Write what is queued, then each run's `end.json`, and stop. True when that finished within `timeout_s`.
        Logging to a closed Logger raises."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    # -- the writer thread ------------------------------------------------------------------------------------------

    def _start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mms-log", daemon=True)
        self._thread.start()
        atexit.register(_close_at_exit, weakref.ref(self))

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closing:
                    self._cond.wait()
                if not self._queue:
                    break  # closing, and nothing left
                while not (self._flush or self._closing or self._lines >= BATCH_LINES
                           or self._frame_bytes >= BATCH_FRAME_BYTES):
                    left = self._oldest + BATCH_S - time.monotonic()
                    if left <= 0:
                        break
                    self._cond.wait(left)
                taken, self._queue = self._queue, []
                self._lines = self._frame_bytes = 0
                self._flush = False
                self._inflight = len(taken)
            try:
                self._write(taken)
            finally:
                with self._cond:
                    self._inflight = 0
                    self._queue_bytes -= sum(item[4] for item in taken)
                    self._cond.notify_all()
        self._end()

    def _write(self, taken: list[tuple]) -> None:
        by_stream: dict[str, list[tuple]] = {}
        for item in taken:
            by_stream.setdefault(item[0], []).append(item)
        for stream, items in by_stream.items():
            for batch in _batches(items):
                self._write_batch(stream, batch)

    def _write_batch(self, stream: str, items: list[tuple]) -> None:
        try:
            reason = self._refusal(stream)
            if reason is None:
                self._write_files(stream, items)
                return
            kind, detail = reason
        except Exception as error:  # the disk or an encoder failed; the logging program goes on
            if _disk_full(error):
                kind, detail = "spool_full", f"the disk is full, so the spool for {stream!r} cannot be written: {error!r}"
            else:
                with self._cond:
                    self._write_errors += len(items)
                    self._last_error = f"could not write the spool for {stream!r}: {error!r}"
                    announce = None if "write_error" in self._announced else self._last_error
                    self._announced.add("write_error")
                if announce:
                    print(f"mms: {announce}; later ones are counted in mms.stats()", file=sys.stderr)
                return
        with self._cond:
            announce = self._count(kind, stream, len(items), detail)
        if announce:
            print(announce, file=sys.stderr)

    def _refusal(self, stream: str) -> tuple[str, str] | None:
        """Why this stream's batch is not written, as (drop kind, detail), or None to write it."""
        if (self.home / "spool" / stream / "refused.json").exists():
            return "refused", f"the daemon refused stream {stream!r} (spool/{stream}/refused.json)"
        if not self._project_known:
            try:
                found = projects.find(self._cwd)
            except Exception as error:  # a project that cannot be told must not be logged as ambient
                self._project_error = f"{type(error).__name__}: {error}"
                return "refused", f"the project of {self._cwd} cannot be found ({self._project_error}); fix it or pass project="
            self._project, self._project_error, self._project_known = (found.id if found else None), None, True
        now = time.monotonic()
        if now - self._floor_read >= FLOOR_CHECK_S:
            self._floor_read = now
            self._read_floor()
        if self._below_floor:
            return "floor", f"free space on {self.home} is below [memory] free_gb ({self._floor_gb:g} GB)"
        if now - self._spool_read >= SPOOL_MEASURE_S:
            self._spool_read = now
            self._spool_bytes = _tree_bytes(self.home / "spool")
        if self._spool_bytes > SPOOL_MAX_BYTES:
            return "spool_full", f"the spool holds {self._spool_bytes / 2**30:.1f} GB: is the daemon running?"
        return None

    def _read_floor(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        try:
            self._floor_gb = config.load(self.home).free_gb
        except (ValueError, OSError) as error:
            if "config" not in self._announced:
                self._announced.add("config")
                print(f"mms: cannot read free_gb from {self.home / 'mms.toml'}: {error}", file=sys.stderr)
        self._below_floor = (self._floor_gb is not None
                             and shutil.disk_usage(self.home).free < self._floor_gb * 2**30)

    def _write_files(self, stream: str, items: list[tuple]) -> None:
        run = self._runs.get(stream)
        if run is None:
            run = self._runs[stream] = _Run(self.home / "spool" / stream, stream)
        run.ensure(self._project)
        run.seq += 1
        prefix = f"{run.seq:08d}"
        lines, size = [], 0
        for index, (_, wall, t, payload, _) in enumerate(items):
            head = f'{{"wall":{wall!r},"t":{"null" if t is None else repr(t)},'.encode()
            if isinstance(payload, bytes):
                lines.append(head + b'"data":' + payload + b"}")
            else:
                name = f"{prefix}-{index:03d}.jpg"
                size += _jpeg(payload, run.folder / name)
                lines.append(head + f'"frame":"{name}","w":{payload.shape[1]},"h":{payload.shape[0]}}}'.encode())
        text = b"\n".join(lines) + b"\n"
        part = run.folder / f"{prefix}.jsonl.part"
        part.write_bytes(text)
        os.replace(part, run.folder / f"{prefix}.jsonl")
        self._spool_bytes += len(text) + size
        with self._cond:
            self._written += len(items)

    def _end(self) -> None:
        for stream, run in self._runs.items():
            try:
                run.ensure(self._project)
                run.write("end.json", {"ended": time.time(), "dropped": self._dropped_by_stream.get(stream, 0)})
            except OSError as error:
                print(f"mms: could not close the run {run.folder}: {error!r}", file=sys.stderr)

    def _count(self, kind: str, stream: str, n: int, detail: str = "") -> str | None:
        """Count `n` dropped lines (the caller holds the lock); the message to print the first time of each kind."""
        self._dropped[kind] += n
        self._dropped_by_stream[stream] = self._dropped_by_stream.get(stream, 0) + n
        if kind in self._announced:
            return None
        self._announced.add(kind)
        return f"mms: dropping log lines ({kind}); later ones are counted in mms.stats(). {detail}".rstrip()


def _disk_full(error: BaseException) -> bool:
    """Whether a write failed because the disk is full (ENOSPC, or Windows' ERROR_DISK_FULL and ERROR_HANDLE_DISK_FULL)."""
    return isinstance(error, OSError) and (error.errno == errno.ENOSPC or getattr(error, "winerror", None) in (112, 39))


class _Run:
    """The spool folder of one stream's run (§8.2): `run.json` first, then batches, and `end.json` at the close."""

    def __init__(self, stream_folder: Path, stream: str):
        self.stream = stream
        self.started = time.time()
        stamp = datetime.fromtimestamp(self.started, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.name = f"{stamp}-{os.getpid()}-{secrets.token_hex(2)}"
        self.folder = stream_folder / self.name
        self.seq = 0

    def ensure(self, project_id: str | None) -> None:
        """Make the folder and `run.json`, again if the daemon removed them."""
        if not (self.folder / "run.json").exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            self.write("run.json", {"v": 1, "stream": self.stream, "run": self.name, "pid": os.getpid(),
                                    "project": project_id, "started": self.started})

    def write(self, name: str, value: dict) -> None:
        part = self.folder / f"{name}.part"
        part.write_text(json.dumps(value), encoding="utf-8")
        os.replace(part, self.folder / name)


def _batches(items: list[tuple]):
    """`items` in order, cut into batches of at most BATCH_LINES lines and BATCH_FRAME_BYTES of frames."""
    batch, frames = [], 0
    for item in items:
        frame = item[4] if not isinstance(item[3], bytes) else 0
        if batch and (len(batch) >= BATCH_LINES or frames + frame > BATCH_FRAME_BYTES):
            yield batch
            batch, frames = [], 0
        batch.append(item)
        frames += frame
    if batch:
        yield batch


def _frame_copy(frame):
    import numpy as np  # a program that passes a frame has numpy

    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError(f"a frame is a numpy uint8 array, not {getattr(frame, 'dtype', type(frame).__name__)}")
    if not (frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 3)) or 0 in frame.shape:
        raise ValueError(f"a frame is HxWx3 (RGB) or HxW (grey), not shape {frame.shape}")
    return frame.copy()


def _jpeg(frame, path: Path) -> int:
    from PIL import Image

    Image.fromarray(frame).save(path, "JPEG", quality=JPEG_QUALITY)
    return path.stat().st_size


def _tree_bytes(folder: Path) -> int:
    total = 0
    for root, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:  # the daemon removed it meanwhile
                pass
    return total


def _close_at_exit(ref: "weakref.ref[Logger]") -> None:
    logger = ref()
    if logger is not None and not logger.close():
        print(f"mms: the log was not fully written to {logger.home / 'spool'} within {CLOSE_WAIT_S:g} s",
              file=sys.stderr)


_shared: Logger | None = None
_shared_lock = threading.Lock()


def _logger() -> Logger:
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = Logger()
    return _shared


def log(stream: str, data: dict | None = None, *, frame=None, t: float | None = None) -> None:
    """`Logger.log` on the Logger the module functions share (its home is `MMS_HOME` at the first call)."""
    _logger().log(stream, data, frame=frame, t=t)


def flush(timeout_s: float = 5.0) -> bool:
    return True if _shared is None else _shared.flush(timeout_s)


def stats() -> dict:
    if _shared is None:
        return {"queued": 0, "written": 0, "dropped": dict.fromkeys(DROPS, 0), "write_errors": 0, "last_error": None}
    return _shared.stats()


def close() -> None:
    """Close the shared Logger. The next `log` starts a new run."""
    global _shared
    with _shared_lock:
        logger, _shared = _shared, None
    if logger is not None:
        logger.close()
