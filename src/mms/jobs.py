"""Jobs (DESIGN.md §10, §13): a job runs a script in its own thread, with the decider handed in through the GPU queue.

`Jobs.call` serves `jobs/start`, `jobs/get` and `jobs/cancel`. A script is `scripts/<name>.py` in the home, else the
shipped one, loaded afresh for each job so an edit counts from the next one. It defines `run(job)` and may define
`cli(argv)`, which turns `mms <name> ...` flags into `params`. `load_script` is the one loader, for the daemon and the CLI.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import logging
import threading
import time
import traceback
import uuid
from pathlib import Path

from mms import ops, reader
from mms.config import NAME, iso
from mms.errors import ApiError, Cancelled

log = logging.getLogger("mms.jobs")

SHIPPED = Path(__file__).parent / "scripts"
WAIT_DEFAULT_S = 40.0
WAIT_MAX_S = 50.0
CANCEL_WAIT_S = 10.0  # a cancel waits this long for the job to stop, which takes at most one window
KEEP_S = 3600.0  # an ended job is kept this long, in memory only
KEEP_MAX = 200
UNREADABLE_SHOWN = 5  # items that could not be read are named in the warnings up to this many
UNREAD_SHOWN = 5  # items the decider does not read, or reads in part, are named up to this many for each group
ENDED = ("done", "failed", "cancelled")
_loads = itertools.count(1)


def script_path(home: Path, name: str) -> Path | None:
    """The home's copy of a script, else the shipped one."""
    for folder in (home / "scripts", SHIPPED):
        path = folder / f"{name}.py"
        if path.is_file():
            return path
    return None


def script_names(home: Path) -> list[str]:
    return sorted({file.stem for folder in (home / "scripts", SHIPPED) if folder.is_dir()
                   for file in folder.glob("*.py") if not file.stem.startswith("_")})


def load_script(home: Path, name: str):
    """A script's module, read afresh. `ValueError` names the scripts there are when there is none by this name."""
    path = script_path(home, name) if NAME.match(name) else None
    if path is None:
        raise ValueError(f"no script {name!r}; there are: {', '.join(script_names(home)) or 'none'}")
    spec = importlib.util.spec_from_file_location(f"mms_script_{name}_{next(_loads)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Record:
    """A job as the daemon keeps it: its state, and what the script has reported."""

    def __init__(self, script: str, caller: ops.Caller, params: dict):
        self.id = f"j-{uuid.uuid4().hex[:8]}"
        self.script, self.caller, self.params = script, caller, params
        self.state = "queued"
        self.started = time.time()
        self.began: float | None = None  # when it left the queue, on the monotonic clock
        self.ended: float | None = None
        self.error: dict | None = None
        self.result: dict = {}
        self.cancelled = threading.Event()
        self.thread: threading.Thread | None = None


class _Rolling:
    """One warning about items a scan reports, each once however often it is reported: counted by group with the first
    few ids, and rewritten in its one place in the job's warnings as more arrive."""

    def __init__(self):
        self.seen: set[int] = set()
        self.groups: dict[str, tuple[int, list[int]]] = {}  # group -> (count, first ids), in the order first seen
        self.at: int | None = None  # where the warning sits

    def note(self, item_id: int, group: str) -> bool:
        """Count an item under its group; False if it was counted before."""
        if item_id in self.seen:
            return False
        self.seen.add(item_id)
        count, ids = self.groups.get(group, (0, []))
        if len(ids) < UNREAD_SHOWN:
            ids.append(item_id)
        self.groups[group] = (count + 1, ids)
        return True

    def listed(self) -> str:
        """The groups, each with its count and first ids: `pdf (items 1, 2 and 7 more), zip (item 9)`."""
        out = []
        for group, (count, ids) in self.groups.items():
            more = f" and {count - UNREAD_SHOWN} more" if count > UNREAD_SHOWN else ""
            out.append(f"{group} ({'item' if count == 1 else 'items'} {', '.join(map(str, ids))}{more})")
        return ", ".join(out)

    def put(self, warnings: list[str], line: str) -> None:
        """Write the warning: at the end the first time, in its place after."""
        if self.at is None:
            self.at = len(warnings)
            warnings.append(line)
        else:
            warnings[self.at] = line


class Job:
    """What a script is handed (DESIGN.md §13.1)."""

    Cancelled = Cancelled

    def __init__(self, record: _Record, jobs: "Jobs"):
        self._record, self._jobs = record, jobs
        self.id, self.params, self.caller = record.id, record.params, record.caller
        self.home, self.store, self.now = jobs.env.home, jobs.env.store, time.time()
        self._about = None  # the decider's About, read once for the job's whole life
        self._unreadable_count = 0
        self._more = _Rolling()  # the "and N more items unreadable" warning (it uses only the place)
        self._not_read = _Rolling()  # the items it does not read, by type
        self._look_shows: dict[str, bool] = {}  # ... and whether look shows that type (else it gives a copy)
        self._in_part = {"sound": _Rolling(), "frames": _Rolling()}  # the video items it reads in part, by source

    @property
    def cancelled(self) -> bool:
        return self._record.cancelled.is_set()

    def about(self):
        """What the decider reads (decider.About), from the GPU queue without loading the model; kept for the job."""
        if self._about is None:
            self._about = self._jobs.gpu.about()
        return self._about

    def check(self, questions: dict) -> None:
        """`ValueError` naming what in `questions` the decider cannot take, before any window is read; loads nothing."""
        self._jobs.gpu.check(questions)

    def profile(self, overrides=None) -> dict:
        """The reader's profile for this decider: `reader.PROFILE`, the decider's own, then `overrides`."""
        return reader.profile_of(overrides, self.about())

    def windows(self, since, until, sources=None, scope="default", newest_first=True, profile=None):
        """`reader.windows` with the caller's scope applied (§5.2) and cut for the decider's `about`. `profile`
        overrides the keys of `reader.PROFILE` it names. An item that cannot be read is counted in the job's warnings
        and the scan goes on, and the items the decider does not read are named in one warning."""
        return reader.windows(self.store, since, until, sources, scope=self.caller.scope(scope), about=self.about(),
                              newest_first=newest_first, profile=profile, on_unreadable=self._unreadable,
                              on_unread=self._unread, on_partial=self._partly)

    def _unreadable(self, item, error: Exception) -> None:
        """Count an item that could not be read: the first few by name, then one line that counts the rest."""
        with self._jobs.cv:
            self._unreadable_count += 1
            warnings = self._record.result.setdefault("warnings", [])
            if self._unreadable_count <= UNREADABLE_SHOWN:
                warnings.append(f"item {item.id} unreadable: {type(error).__name__}: {error}")
            else:
                self._more.put(warnings, f"and {self._unreadable_count - UNREADABLE_SHOWN} more items unreadable")
            self._jobs.cv.notify_all()

    def _unread(self, item) -> None:
        """Note an item the decider does not read, once, in one warning that is rewritten as more arrive: the count, then
        each type with its count and the first few ids. It wakes no waiter (there can be tens of thousands of items): the
        next update or the end of the job carries it."""
        with self._jobs.cv:
            kind = item.meta.get("media") or (f"{item.kind} segments" if item.kind != "file" else "an unknown type")
            if not self._not_read.note(item.id, kind):
                return
            self._look_shows[kind] = reader.form(item) is not None
            total = len(self._not_read.seen)
            one = total == 1
            it, each = ("it", "it") if one else ("them", "each")
            if all(self._look_shows.values()):
                look = f"look shows {it}"
            elif not any(self._look_shows.values()):
                look = f"look gives a copy of {each}"
            else:
                look = "look shows a picture, sound or video and gives a copy of any other file"
            line = (f"{total} {'item was' if one else 'items were'} not read, since "
                    f"{self.about().name} does not read {'its type' if one else 'their types'}: "
                    f"{self._not_read.listed()}; find lists {it} and {look}")
            self._not_read.put(self._record.result.setdefault("warnings", []), line)

    def _partly(self, item, what: str) -> None:
        """Note a video item the decider reads in part, once, in a rolling warning like the not-read one: `what` is the
        part that is lost, "sound" (the decider does not hear) or "frames" (it does not see). It wakes no waiter."""
        with self._jobs.cv:
            rolling = self._in_part[what]
            if not rolling.note(item.id, item.source):
                return
            total = len(rolling.seen)
            one = total == 1
            lost, verb, sense = (("the sound of", "was not heard", "hear") if what == "sound"
                                 else ("the frames of", "were not seen", "see"))
            line = (f"{lost} {total} video {'item' if one else 'items'} {verb}, since {self.about().name} does not "
                    f"{sense}: {rolling.listed()}; find lists {'it' if one else 'them'}")
            rolling.put(self._record.result.setdefault("warnings", []), line)

    def estimate(self, since, until, sources=None, scope="default", profile=None) -> dict:
        """How many windows of each kind `windows` would give, from the index, without decoding."""
        return reader.estimate(self.store, since, until, sources, self.caller.scope(scope), profile,
                               about=self.about())

    def searched(self, since, until, sources=None, scope="default") -> dict:
        return ops.searched(self._jobs.env, self.caller, since, until, sources, scope)

    def decide(self, state, questions: dict, media: list | None = None) -> dict:
        """The decider's answers for one window, through the GPU queue; `Cancelled` once the job is cancelled."""
        if self.cancelled:
            raise Cancelled()
        return self._jobs.gpu.decide(state, questions, media or [], self._record.cancelled.is_set)

    def decider(self) -> dict:
        return self._jobs.gpu.status()

    def update(self, **fields) -> None:
        """Merge fields into the job's `result`, visible to waiters at once."""
        with self._jobs.cv:
            self._record.result.update(copy.deepcopy(fields))
            self._jobs.cv.notify_all()

    def warn(self, text: str) -> None:
        with self._jobs.cv:
            self._record.result.setdefault("warnings", []).append(text)
            self._jobs.cv.notify_all()

    def label(self, t0: float, t1: float, text: str, *, source: str | None = None, item: int | None = None,
              meta: dict | None = None) -> int:
        """A label by the script's name, with the caller's project and session."""
        return self.store.label(t0, t1, text, self._record.script, source=source, item=item, meta=meta,
                                project=self.caller.project, session=self.caller.session)


class Jobs:
    def __init__(self, env: ops.Env, gpu):
        self.env, self.gpu = env, gpu
        self.cv = threading.Condition()
        self.records: dict[str, _Record] = {}
        self.waiting: list[_Record] = []
        self.running = 0
        self.stopping = False

    # the API

    def call(self, op: str, body) -> dict:
        if not isinstance(body, dict):
            raise ApiError("bad_request", "the body is a JSON object")
        body = dict(body)
        caller = ops.parse_caller(body.pop("caller", None))
        self.env.store.note_caller(caller.asdict())
        if op == "jobs/start":
            return self._start(caller, body)
        if op == "jobs/get":
            p = ops.parse("jobs/get", body, {"job": ops._text(blank=False), "wait_s": ops._number(atleast=0)},
                          required=("job",))
            record = self._record(p["job"])
            return self._wait(record, self._wait_s(p))
        if op == "jobs/cancel":
            p = ops.parse("jobs/cancel", body, {"job": ops._text(blank=False)}, required=("job",))
            record = self._record(p["job"])
            self._cancel(record)
            return self._wait(record, CANCEL_WAIT_S)
        raise ApiError("not_found", f"no op {op!r}", field="op")

    @staticmethod
    def _wait_s(p: dict) -> float:
        wait = p.get("wait_s", WAIT_DEFAULT_S)
        if wait > WAIT_MAX_S:
            raise ApiError("bad_request", f"wait_s is a number from 0 to {WAIT_MAX_S:g}", field="wait_s")
        return wait

    def _record(self, job: str) -> _Record:
        with self.cv:
            record = self.records.get(job)
        if record is None:
            raise ApiError("not_found", f"no job {job!r}; a restarted daemon forgets its jobs, so ask again",
                           field="job")
        return record

    def _start(self, caller: ops.Caller, body: dict) -> dict:
        p = ops.parse("jobs/start", body, {"script": ops._name, "params": _object, "wait_s": ops._number(atleast=0)},
                      required=("script",))
        wait = self._wait_s(p)
        if script_path(self.env.home, p["script"]) is None:
            raise ApiError("not_found", f"no script {p['script']!r}; there are: "
                                        f"{', '.join(script_names(self.env.home))}", field="script")
        record = _Record(p["script"], caller, p.get("params", {}))
        with self.cv:
            if self.stopping:
                raise ApiError("internal", "the daemon is stopping")
            self.records[record.id] = record
            self.waiting.append(record)
        record.thread = threading.Thread(target=self._run, args=(record,), name=f"job-{record.id}", daemon=True)
        record.thread.start()
        return self._wait(record, wait)

    def _view(self, record: _Record) -> dict:
        """A job as the API gives it; the caller holds `cv`."""
        return {"job": record.id, "script": record.script, "state": record.state, "started": iso(record.started),
                "ended": iso(record.ended), "error": copy.deepcopy(record.error),
                "result": copy.deepcopy(record.result)}

    def _wait(self, record: _Record, wait_s: float) -> dict:
        """The job once it ends, or after `wait_s`, with what it has reported so far."""
        end = time.monotonic() + wait_s
        with self.cv:
            while record.state not in ENDED:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self.cv.wait(left)
            return self._view(record)

    def _cancel(self, record: _Record) -> None:
        with self.cv:
            record.cancelled.set()
            self.cv.notify_all()

    # running

    def _run(self, record: _Record) -> None:
        with self.cv:
            while not record.cancelled.is_set() and not (self.waiting[0] is record
                                                          and self.running < self.env.config.jobs):
                self.cv.wait(0.5)
            self.waiting.remove(record)
            if record.cancelled.is_set():
                self._end(record, "cancelled")
                return
            self.running += 1
            record.state, record.began = "running", time.monotonic()
            self.cv.notify_all()
        state, error = "done", None
        try:
            load_script(self.env.home, record.script).run(Job(record, self))
        except Cancelled:
            pass
        except ValueError as problem:
            state, error = "failed", {"code": "bad_request", "message": str(problem)}
        except ApiError as problem:
            state, error = "failed", problem.body()
        except Exception as problem:
            log.error("job %s (%s) failed:\n%s", record.id, record.script, traceback.format_exc())
            state, error = "failed", {"code": "internal", "message": f"{type(problem).__name__}: {problem}"}
        with self.cv:
            self.running -= 1
            if record.cancelled.is_set() and state == "done":
                state = "cancelled"
            self._end(record, state, error)

    def _end(self, record: _Record, state: str, error: dict | None = None) -> None:
        """The caller holds `cv`."""
        record.state, record.error, record.ended = state, error, time.time()
        self.cv.notify_all()

    # the loop's chores

    def sweep(self, now: float | None = None) -> None:
        """Cancel a job past `job_max_s`, and forget ended jobs after an hour (at most 200 are kept)."""
        now = time.time() if now is None else now
        limit = self.env.config.job_max_s
        with self.cv:
            for record in self.records.values():
                if (record.state == "running" and time.monotonic() - record.began > limit
                        and not record.cancelled.is_set()):
                    record.cancelled.set()
                    record.result.setdefault("warnings", []).append(f"cancelled: it ran past job_max_s ({limit:g} s)")
            ended = sorted((r for r in self.records.values() if r.state in ENDED), key=lambda r: r.ended)
            excess = len(ended) - KEEP_MAX
            for i, record in enumerate(ended):
                if i < excess or now - record.ended > KEEP_S:
                    del self.records[record.id]
            self.cv.notify_all()

    def apply(self) -> None:
        """A new `[memory] jobs` lets waiting jobs start."""
        with self.cv:
            self.cv.notify_all()

    def counts(self) -> dict:
        with self.cv:
            return {"running": self.running, "queued": len(self.waiting)}

    def stop(self) -> None:
        """Refuse new jobs, cancel the rest, and wait for them."""
        with self.cv:
            self.stopping = True
            threads = [r.thread for r in self.records.values() if r.state not in ENDED and r.thread]
            for record in self.records.values():
                record.cancelled.set()
            self.cv.notify_all()
        for thread in threads:
            thread.join(CANCEL_WAIT_S)


def _object(value):
    if not isinstance(value, dict):
        raise ValueError("is an object")
    return value
