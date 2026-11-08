"""Folder, log, command and push sources, and the spool (DESIGN.md §7).

`Ingest.step(now)` runs from the daemon's loop: it rescans each watched path every `rescan_s`, keeps one process per
approved command source, and steps the spool (spool.py). Where a watch has got to is kept in the store's `cursors`, so a
restarted daemon carries on. Push's write path is `push_note` and `push_path`, which the `remember` op calls.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from mms import children, consent, media
from mms.config import NAME, Config, Source, reads_the_home
from mms.errors import ApiError
from mms.spool import Spool
from mms.store import Store

MB = 1_000_000
BACKOFF_S = (1.0, 60.0)  # a command's restart delay: first, and most; it doubles per quick exit
LONG_RUN_S = 30.0  # a command that ran this long starts again with the first delay
STOP_WAIT_S = 3.0
TAIL_CHUNK = 1 << 20  # a tail reads this many bytes at a time, and a line longer than it is cut there
TAIL_PASS = 16  # chunks of one file in one pass of the loop; a file still behind is read again at once
FOLDER_MAX_FILES = 1000  # the most files `remember` takes from one folder
STEP_FILES = 500  # files of a folder or log source looked at in one step of the loop; a bigger tree takes several
STEP_BYTES = 256 * MB  # and the most of them copied in one step
RESCAN_S = 5.0  # a folder or log source looks again this often unless its table says `rescan_s`
FLOOR_RETRY_S = 5.0  # a source whose write would pass the floor looks again after this


def _noop(_: str) -> None:
    pass


def record_time(value, default: float) -> float:
    """The time a record's time field names (epoch seconds or ISO 8601; no zone is local), else `default`."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().timestamp()
        except ValueError:
            pass
    return default


def line_record(text: str, arrived: float, time_field: str | None = None) -> tuple[float, str]:
    """A line as a record: (t, JSON text). A JSON object is kept as it is, and its `time_field`, when the source names
    one, sets the time; any other line becomes {"line": text}. Without a `time_field` a record's time is `arrived`."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return arrived, json.dumps({"line": text}, ensure_ascii=False)
    when = record_time(data.get(time_field), arrived) if time_field else arrived
    return when, json.dumps(data, ensure_ascii=False)


def _same(a: Source, b: Source) -> bool:
    """Whether two tables ask for the same work: a cap or a project does not."""
    return (a.kind, a.options) == (b.kind, b.options)


class _Walk:
    """A pass over a folder or log source that takes several steps: the files still to look at, and what it has seen."""

    def __init__(self, paths, known: dict, options: dict):
        self.paths, self.known, self.options = paths, known, options  # `options` are the source's, when it began
        self.seen: set[str] = set()
        self.skipped = 0
        self.last_error: str | None = None


class Ingest:
    def __init__(self, store: Store, config: Config, log=_noop):
        self.store = store
        self.config = config
        self.log = log
        self.next_scan: dict[str, float] = {}
        self.missing: set[str] = set()  # watches whose path is not there, as last seen
        self.watches: dict[str, Source] = {}
        self.refused: dict[str, str] = {}  # watches whose path is the MultimediaSearch home, or holds it, and why
        self.walks: dict[str, _Walk] = {}
        self.errors: dict[str, str] = {}  # the last thing that went wrong in a source, shown in `status` until it does not
        self.commands: dict[str, _Command] = {}
        self.spool = Spool(store, config, log)
        self.apply(config)

    def apply(self, config: Config) -> None:
        """A changed config (DESIGN.md §4.2): removed, changed and no longer approved sources stop, new ones start at
        the next step, and the rest are left alone."""
        self.config = config
        every = {name: source for name, source in config.sources.items() if source.kind in ("folder", "log")}
        self.refused = {name: f"{source.get('path')} is, holds or is inside the MultimediaSearch home, so it is not scanned"
                        for name, source in every.items() if reads_the_home(config.home, source.get("path"))}
        self.watches = {name: source for name, source in every.items() if name not in self.refused}
        for name in list(self.next_scan.keys() - self.watches.keys()):
            del self.next_scan[name]
        for name in list(self.walks):
            if name not in self.watches or self.walks[name].options != self.watches[name].options:
                del self.walks[name]  # a source that went, or changed what it reads, starts its pass again
        self.missing &= self.watches.keys()
        for name in list(self.errors.keys() - config.sources.keys()):
            del self.errors[name]
        wanted = {name: source for name, source in config.sources.items()
                  if source.kind == "command" and consent.approved(config.home, name, source.what())}
        for name in list(self.commands):
            if name not in wanted or not _same(self.commands[name].source, wanted[name]):
                self.commands.pop(name).stop(self.store)
        for name, source in wanted.items():
            if name in self.commands:
                self.commands[name].source = source  # a new project or cap counts from the next line
            else:
                self.commands[name] = _Command(source, self.log)
        self.spool.apply(config)

    def step(self, now: float) -> None:
        for source in self.watches.values():
            if now >= self.next_scan.get(source.name, 0):
                self.next_scan[source.name] = now + float(source.get("rescan_s", RESCAN_S))
                try:
                    if self.scan(source, now):  # a tail or a big tree with more to read does not wait for the rescan
                        self.next_scan[source.name] = now
                except ApiError as error:  # a write that would pass the floor or the cap: look again soon
                    self.next_scan[source.name] = now + FLOOR_RETRY_S
                    self._fail(source, error.message)
                except Exception as error:  # one bad source must not stop the others
                    self._fail(source, f"{type(error).__name__}: {error}")
        for one in list(self.commands.values()):
            try:
                one.step(now, self.store)
                self.errors.pop(one.source.name, None)
            except Exception as error:
                self._fail(one.source, f"{type(error).__name__}: {error}")
        self.spool.step(now)

    def _fail(self, source: Source, message: str) -> None:
        """Keep what went wrong for `status`, and log it when it is new."""
        if self.errors.get(source.name) != message:
            self.log(f"{source.kind} {source.name}: {message}")
        self.errors[source.name] = message

    def stop(self) -> None:
        for one in list(self.commands.values()):
            one.stop(self.store)
        self.spool.stop()

    def status(self) -> dict:
        """Per source ingest owns: its state, and what is wrong with it."""
        out = {}
        for name, source in self.config.sources.items():
            if name in self.refused:
                out[name] = {"state": "error", "error": self.refused[name]}
            elif source.kind in ("folder", "log"):
                out[name] = {"state": "missing" if name in self.missing else "running"}
                if name in self.errors:
                    out[name]["error"] = self.errors[name]
            elif source.kind == "command":
                one = self.commands.get(name)
                if one is None:
                    out[name] = {"state": "needs_approval"}
                else:
                    out[name] = {"state": "running" if one.proc is not None else "waiting"}
                    if (one.error and one.proc is None) or name in self.errors:
                        out[name]["error"] = self.errors.get(name) or one.error
            elif source.kind in ("push", "program"):
                out[name] = {"state": "idle"}
                if name in self.spool.errors:
                    out[name]["error"] = self.spool.errors[name]
        return out

    # folder and log

    @staticmethod
    def _paths(base: Path, source: Source):
        """The files a source watches, as (name, path), found as they are asked for."""
        if base.is_file():
            yield base.name, base
        elif base.is_dir():
            for path in base.glob(source.get("glob", "**/*")):
                if path.is_file():
                    yield path.relative_to(base).as_posix(), path

    def scan(self, source: Source, now: float) -> bool:
        """One step of a pass over a watched path: new versions or new lines, and when the pass ends, what vanished. A
        step looks at `STEP_FILES` files and copies `STEP_BYTES`, so a huge tree takes many steps and never holds the
        loop or the janitor. True when there is more to read now. A path that is not there (an unmounted drive, an
        offline share) is not scanned, or every file in it would look deleted, and be copied again when it came back."""
        base = Path(source.get("path")).expanduser()
        if not base.exists():
            if source.name not in self.missing:
                self.missing.add(source.name)
                self.log(f"{source.kind} {source.name}: {base} is not there, so it is not scanned")
            self.walks.pop(source.name, None)
            return False
        if source.name in self.missing:
            self.missing.discard(source.name)
            self.log(f"{source.kind} {source.name}: {base} is back")
        tail = source.kind == "log"
        walk = self.walks.get(source.name)
        if walk is None:
            walk = self.walks[source.name] = _Walk(
                self._paths(base, source), {cursor["name"]: cursor for cursor in self.store.cursors(source.name)},
                source.options)
        behind, looked, copied = False, 0, 0
        for name, path in walk.paths:
            walk.seen.add(name)
            try:
                if tail:
                    behind |= self._tail(source, name, path, walk.known.get(name), now)
                else:
                    copied += self._version(source, name, path, walk.known.get(name), now)
            except OSError as error:  # open in another program, or gone since the listing: the next pass tries again
                walk.skipped += 1
                walk.last_error = f"{name}: {error}"
            looked += 1
            if looked >= STEP_FILES or copied >= STEP_BYTES:
                return True
        del self.walks[source.name]
        for name in walk.known.keys() - walk.seen:
            if not tail and self.store.end_version(source.name, name, now):
                self.log(f"folder {source.name}: {name} is gone")
            self.store.drop_cursor(source.name, name)
        if walk.skipped:
            self._fail(source, f"{walk.skipped} files could not be read and are tried again: {walk.last_error}")
        else:
            self.errors.pop(source.name, None)
        return behind

    def _version(self, source: Source, name: str, path: Path, cursor: dict | None, now: float) -> int:
        """A changed file as a version, or as a recording at its real time when it holds video, sound or a picture (a
        file too big to copy stays a version, in place); returns the bytes it copied."""
        stat = path.stat()
        if cursor and cursor["size"] == stat.st_size and cursor["mtime"] == stat.st_mtime:
            return 0
        if stat.st_size > float(source.get("copy_max_mb", 50)) * MB:  # no copy, no hash: one row that follows the file
            self.store.index_in_place(source.name, name, path, now, stat.st_size, project=source.project)
            self.store.set_cursor(source.name, name, size=stat.st_size, mtime=stat.st_mtime)
            self.log(f"folder {source.name}: {name} changed (indexed in place)")
            return 0
        facts = media.of_file(path)
        recording = media.is_recording(facts["media"])
        if recording and 0 <= now - stat.st_mtime < 2 * float(source.get("rescan_s", RESCAN_S)):
            return 0  # perhaps still being written: no cursor, so the next pass looks again (a future mtime is settled)
        if recording:
            added = "stored" in store_recording(self.store, source.name, path, name, None, facts, now, strict=False,
                                                project=source.project)
        else:
            added = self.store.add_version(source.name, name, now, src=path, project=source.project) is not None
        self.store.set_cursor(source.name, name, size=stat.st_size, mtime=stat.st_mtime)
        if added:
            self.log(f"folder {source.name}: {name} changed")
        return stat.st_size

    def _tail(self, source: Source, name: str, path: Path, cursor: dict | None, now: float) -> bool:
        """New complete lines as records, a bounded read at a time. Each chunk's records and the cursor commit together.
        True when the file has more to read after this pass."""
        offset = cursor["offset"] if cursor else 0
        size = path.stat().st_size
        if size < offset:  # it shrank: a new file, or truncated
            offset = 0
        if size == offset:
            if not cursor or cursor["offset"] != offset:
                self.store.set_cursor(source.name, name, offset=offset, size=size)
            return False
        time_field = source.get("time_field")
        with open(path, "rb") as file:
            wide = _utf16(file.read(2))  # PowerShell 5.1 writes its logs as UTF-16 with a BOM
            if wide:
                offset = max(offset, 2)  # past the BOM
                offset += offset % 2  # the old tail saved an odd offset, between a newline and its NUL
            for _ in range(TAIL_PASS):
                file.seek(offset)
                data = file.read(min(TAIL_CHUNK, size - offset))
                end = _line_end(data, wide) if wide else data.rfind(b"\n") + 1  # only complete lines
                if not end and len(data) == TAIL_CHUNK:  # a line longer than a chunk: cut it, or the tail would stick
                    end = len(data)
                chunk = data[:end].decode(wide, errors="replace").encode() if wide else data[:end]
                rows = []
                for raw in chunk.split(b"\n"):
                    text = media.decode_line(raw)
                    if text:
                        rows.append(line_record(text, now, time_field))
                offset += end
                self.store.add_records(source.name, rows, cursor=(name, {"offset": offset, "size": size}),
                                       project=source.project)
                if not end or offset >= size:
                    return False
        return True


def _utf16(start: bytes) -> str | None:
    """The UTF-16 codec a file's first two bytes name (its BOM), else None."""
    return {b"\xff\xfe": "utf-16-le", b"\xfe\xff": "utf-16-be"}.get(start)


def _line_end(data: bytes, codec: str) -> int:
    """Where the last complete line of UTF-16 `data` ends: after the last newline that falls on a character."""
    newline = b"\n\x00" if codec == "utf-16-le" else b"\x00\n"
    end = data.rfind(newline)
    while end > 0 and end % 2:  # a byte pair across two characters is no newline
        end = data.rfind(newline, 0, end + 1)
    return end + 2 if end >= 0 else 0


class _Command:
    """One command source's process: a thread reads its stdout lines into a queue, and `step` turns them into records."""

    def __init__(self, source: Source, log):
        self.source = source
        self.log = log
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.lines: queue.Queue = queue.Queue()
        self.started = 0.0
        self.next_start = 0.0
        self.backoff = BACKOFF_S[0]
        self.error: str | None = None

    def step(self, now: float, store: Store) -> None:
        self.drain(store)
        if self.proc is None:
            if now >= self.next_start:
                self.start(now)
        elif self.proc.poll() is not None and not self.thread.is_alive() and self.lines.empty():
            ran = now - self.started
            self.backoff = BACKOFF_S[0] if ran >= LONG_RUN_S else min(self.backoff * 2, BACKOFF_S[1])
            self.next_start = now + self.backoff
            self.log(f"command {self.source.name}: exited {self.proc.returncode}, again in {self.backoff:g} s")
            self.proc = None

    def start(self, now: float) -> None:
        command = self.source.get("command")
        self.started = now
        try:
            self.proc = children.popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1"))
        except OSError as error:
            self.error = f"cannot start {command[0]}: {error}"
            self.log(f"command {self.source.name}: {self.error}")
            self.next_start = now + self.backoff
            self.backoff = min(self.backoff * 2, BACKOFF_S[1])
            return
        self.error = None
        self.thread = threading.Thread(target=self._read, args=(self.proc,), daemon=True)
        self.thread.start()
        self.log(f"command {self.source.name}: started pid {self.proc.pid}")

    def _read(self, proc: subprocess.Popen) -> None:
        for raw in proc.stdout:
            self.lines.put((time.time(), media.decode_line(raw)))

    def drain(self, store: Store) -> None:
        rows = []
        time_field = self.source.get("time_field")
        while True:
            try:
                arrived, text = self.lines.get_nowait()
            except queue.Empty:
                break
            if text:
                rows.append(line_record(text, arrived, time_field))
        store.add_records(self.source.name, rows, project=self.source.project)

    def stop(self, store: Store) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self.thread is not None:
            self.thread.join(STOP_WAIT_S)
        if proc is not None and proc.stdout:
            proc.stdout.close()
        self.drain(store)
        self.proc = None


# push: what `remember` writes (DESIGN.md §7)

def check_push(config: Config, source: str) -> None:
    """A push may go to any named source, but not to one the owner set up to be something else."""
    if not NAME.match(source):
        raise ApiError("bad_request", f"a source's name is letters, digits, '_' or '-', up to 64 characters, "
                                      f"not {source!r}", field="source")
    configured = config.sources.get(source)
    if configured is not None and configured.kind != "push":
        raise ApiError("wrong_kind", f"source {source!r} is a {configured.kind} source; remember into a source of "
                                     f"kind push", field="source")


def push_note(store: Store, source: str, text: str, name: str | None, t: float, **who) -> dict:
    """A note as a record `{"text": ..., "name"?: ...}` at `t`; `who` is project, session and expires."""
    data = {"text": text} if name is None else {"text": text, "name": name}
    made = store.add_record(source, t, json.dumps(data, ensure_ascii=False), **who)
    return {"stored": [{"type": "record", "id": made, "t0": t}], "unchanged": []}


def push_path(store: Store, config: Config, source: str, path: Path, name: str | None, t: float | None,
              **who) -> dict:
    """A file as a version, a folder as a version of each file under it, a video, sound or picture as a recording at
    its real time."""
    if path.is_dir():
        files = [file for file in sorted(path.rglob("*")) if file.is_file()]
        if len(files) > FOLDER_MAX_FILES:
            raise ApiError("too_big", f"{path} holds {len(files)} files; remember takes at most {FOLDER_MAX_FILES}",
                           field="path")
        prefix = name or path.name
        named = [(file, f"{prefix}/{file.relative_to(path).as_posix()}") for file in files]
    elif path.is_file():
        named = [(path, name or path.name)]
    else:
        raise ApiError("not_found", f"no such file or folder: {path}", field="path")
    for file, _ in named:
        if file.stat().st_size > config.remember_max_mb * MB:
            raise ApiError("too_big", f"{file} is {file.stat().st_size / MB:.0f} MB; remember takes at most "
                                      f"{config.remember_max_mb:g} MB (remember_max_mb)", field="path")
    now = time.time()
    out: dict = {"stored": [], "unchanged": []}
    for file, label in named:
        facts = media.of_file(file)
        if media.is_recording(facts["media"]):
            result = store_recording(store, source, file, label, t, facts, now, **who)
        else:
            added = store.add_version(source, label, now if t is None else t, src=file, **who)
            if added is None:
                result = {"unchanged": {"name": label, "id": store.current_version(source, label).id}}
            else:
                result = {"stored": {"type": "item", "id": added, "kind": "file", "name": label,
                                     "t0": now if t is None else t}}
        for key in ("stored", "unchanged"):
            if key in result:
                out[key].append(result[key])
    return out


def store_recording(store: Store, source: str, file: Path, name: str, given: float | None, facts: dict, now: float,
                    strict: bool = True, **who) -> dict:
    """A video, a sound or a picture copied into the source and indexed at its real time (DESIGN.md §7): a segment over
    its length, a picture at one instant. The result is `{"stored": ...}` or `{"unchanged": ...}`, the entry carrying
    `time_from` (`recorded_at`, kept in the item's `meta`); an `unchanged` entry is the stored item's own, in the same form.
    A video or sound PyAV finds nothing in is `bad_request`, or with `strict` off a recording of no length, which the
    reader reports when it scans it. The name's current version, if any, ends at `now`: the name holds a recording now."""
    kind = facts["media"].split("/")[0]
    length = 0.0
    if kind != "image":
        length = media.duration(file)
        if length is None:
            if strict:
                raise ApiError("bad_request", f"{file} is not media this machine can read", field="path")
            length = 0.0
    start, time_from = recorded_at(file, facts["media"], given, length, now)
    added, new = store.add_recording(source, kind, name, file, start, start + length, facts["media"],
                                     sound=facts.get("sound", False), time_from=time_from, **who)
    store.end_version(source, name, now)
    if new:
        return {"stored": _entry(added, kind, name, start, start + length, time_from)}
    held = store.item(added)
    return {"unchanged": _entry(added, held.kind, name, held.t0, held.t1, held.meta.get("time_from"))}


def _entry(item_id: int, kind: str, name: str, t0: float, t1: float, time_from: str | None) -> dict:
    return {"type": "item", "id": item_id, "kind": kind, "name": name, "t0": t0, "t1": t1,
            **({"time_from": time_from} if time_from else {})}


def recorded_at(file: Path, media_type: str, given: float | None, length: float, now: float) -> tuple[float, str]:
    """When a recording starts, and where that comes from (`time_from`): the `t` given, else the file's own time
    (`metadata`: a picture's EXIF, a video's or sound's `creation_time`), else its mtime less its length (`mtime`), else
    now less its length (`now`, so it ends now). A time is believed only if the recording lies within [1970, now]: not
    after now (a camera's clock set wrong, a bad copy), and not before the epoch, which has no file name or ISO time here."""
    if given is not None:
        return given, "given"
    own = media.recorded(file, media_type)
    if own is not None and 0 <= own and own + length <= now:
        return own, "metadata"
    modified = file.stat().st_mtime
    if 0 <= modified - length and modified <= now:
        return modified - length, "mtime"
    return now - length, "now"
