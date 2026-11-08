"""Ingesting the spool (DESIGN.md §8.3): the batches a program's `mms.log` wrote under `spool/` become records, clock
rows and video segments. The files follow §8.2, which is the contract with the client; nothing here is shared with it.

`Spool.step(now)` runs from `Ingest.step`. A run's records commit with their clock row and cursor in one transaction, so
a crash never repeats one. Its frames wait in the spool until they span `segment_s` (or stop for 10 s, or the run ends),
then one ffmpeg child encodes them into a segment, which is indexed with the frames cursor in one transaction.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from mms import capture, children
from mms.config import NAME, Config, ConfigError, ensure_program, ffmpeg_exe
from mms.errors import ApiError
from mms.store import Store

SCAN_S = 0.5  # how often the spool's folders are listed
IDLE_FRAMES_S = 10.0  # frames that stopped coming this long ago are encoded
IDLE_RUN_S = 86400.0  # a run whose cursor has not moved this long, with nothing unread, is removed
BATCH = re.compile(r"^(\d{8})\.jsonl$")
FRAME = re.compile(r"^(\d{8})-(\d{3})\.jpg$")
BATCHES_PER_STEP = 200  # a daemon that was down reads its backlog over many steps, not one
ENCODE_RETRY_S = 10.0
STOP_WAIT_S = 10.0
STREAM_KEEP = {"max_gb": 5, "when_full": "thin"}  # the cap a new program stream gets (DESIGN.md §4.4)


def _noop(_: str) -> None:
    pass


def cleanup_parts(store: Store) -> None:
    """At startup: an encode a crash cut short left a `.part`; its frames are still in the spool."""
    for part in store.streams.glob("*/*/*.mkv.part"):
        part.unlink(missing_ok=True)


@dataclass
class Frame:
    pos: int  # seq * 1000 + i: where the frame sits in the run, which the frames cursor counts
    wall: float
    t: float | None
    path: Path
    w: int
    h: int


@dataclass
class Encode:
    process: subprocess.Popen
    frames: list[Frame]
    durations: list[float]
    part: Path
    target: Path
    errors: Path


@dataclass
class RunState:
    stream: str
    run: str
    folder: Path
    project: str | None
    records_done: int
    frames_done: int  # the frames cursor: the position of the last frame encoded
    progress_at: float = 0.0  # when the records cursor last moved, kept in the cursor's `mtime`
    frames_seen: int = 0  # the last batch whose frames were listed
    pending: list[Frame] = field(default_factory=list)
    encode: Encode | None = None
    retry_at: float = 0.0


def parse_batch(text: str) -> list[dict]:
    """The lines of a batch, each checked against §8.2: a record `{wall, t, data}` or a frame `{wall, t, frame, w, h}`."""
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            line = json.loads(raw)
        except ValueError as error:
            raise ValueError(f"line {number}: {error}") from None
        wall, t = (line.get("wall"), line.get("t")) if isinstance(line, dict) else (None, None)
        number_ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
        if not number_ok(wall) or not (t is None or number_ok(t)):
            raise ValueError(f"line {number}: a line is an object with a number `wall` and a number or null `t`")
        if isinstance(line.get("data"), dict) and "frame" not in line:
            pass
        elif (isinstance(line.get("frame"), str) and FRAME.match(line["frame"]) and "data" not in line
              and all(isinstance(line.get(k), int) and line[k] > 0 for k in ("w", "h"))):
            pass
        else:
            raise ValueError(f"line {number}: a line has `data` (an object) or `frame` (<seq>-<i>.jpg) with `w` and `h`")
        lines.append(line)
    return lines


class Spool:
    def __init__(self, store: Store, config: Config, log=_noop):
        self.store = store
        self.config = config
        self.log = log
        self.root = store.home / "spool"
        self.ffmpeg = ffmpeg_exe(config)
        self.runs: dict[tuple[str, str], RunState] = {}
        self.table: dict[str, bool] = {}  # streams whose table is settled: True is a program stream, False refused
        self.stream_retry: dict[str, float] = {}
        self.next_scan = 0.0
        self.bad_logged: set[str] = set()
        self.errors: dict[str, str] = {}  # per stream, what went wrong last, for `status`, until a pass is clean

    def apply(self, config: Config) -> None:
        self.config = config
        self.ffmpeg = ffmpeg_exe(config)
        for stream, program in list(self.table.items()):
            source = config.sources.get(stream)
            if not program and (source is None or source.kind == "program"):  # the clash is gone: let the client write
                (self.root / stream / "refused.json").unlink(missing_ok=True)
                del self.table[stream]
            elif program and (source is None or source.kind != "program"):  # the table went: settle the stream again
                del self.table[stream]

    # the loop

    def step(self, now: float) -> None:
        for state in self.runs.values():
            if state.encode is not None:
                self._poll_encode(state)
        if now < self.next_scan or not self.root.is_dir():
            return
        self.next_scan = now + SCAN_S
        budget = BATCHES_PER_STEP
        for stream_dir in sorted(self.root.iterdir()):
            if not stream_dir.is_dir() or stream_dir.name.startswith("_") or not NAME.match(stream_dir.name):
                continue
            if not self._settle_table(stream_dir.name, now):
                continue
            for run_dir in sorted(stream_dir.iterdir()):
                if run_dir.is_dir():
                    try:
                        budget -= self._step_run(stream_dir.name, run_dir, now, budget)
                        self.errors.pop(stream_dir.name, None)
                    except Exception as error:  # one bad run must not stop the others
                        self._fail(stream_dir.name, f"{run_dir.name}: {type(error).__name__}: {error}")

    def _fail(self, stream: str, message: str) -> None:
        """Keep what went wrong for `status`, and log it when it is new."""
        if self.errors.get(stream) != message:
            self.log(f"spool {stream}: {message}")
        self.errors[stream] = message

    def stop(self) -> None:
        """Let each encode finish (10 s, then kill it), committing what finished."""
        end = time.monotonic() + STOP_WAIT_S
        for state in self.runs.values():
            if state.encode is None:
                continue
            try:
                state.encode.process.wait(max(0.0, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                state.encode.process.kill()
                state.encode.process.wait()
            self._poll_encode(state)

    # a stream's table

    def _settle_table(self, stream: str, now: float) -> bool:
        """Whether the stream may be ingested: it has a program table (added with a cap at its first log), or the owner's
        table of another kind refuses it, with `refused.json` for the client."""
        if stream in self.table:
            return self.table[stream]
        if now < self.stream_retry.get(stream, 0):
            return False
        source = self.config.sources.get(stream)
        try:
            kind = source.kind if source else ensure_program(self.store.home, stream, STREAM_KEEP)
        except ConfigError as error:
            self.stream_retry[stream] = now + 5
            self._fail(stream, f"waiting, mms.toml does not parse: {error}")
            return False
        if kind != "program":
            reason = f"source {stream!r} is a {kind} source, not a program stream"
            (self.root / stream / "refused.json").write_text(json.dumps({"reason": reason}), encoding="utf-8")
            self.log(f"spool {stream}: refused: {reason}")
        self.table[stream] = kind == "program"
        return self.table[stream]

    # a run

    def _step_run(self, stream: str, folder: Path, now: float, budget: int) -> int:
        """One pass over a run's folder; returns the batches it read."""
        key = (stream, folder.name)
        state = self.runs.get(key)
        if state is None:
            state = self._open(stream, folder)
            if state is None:
                return 0
            self.runs[key] = state
        read = 0
        names = sorted(name for name in os.listdir(folder) if BATCH.match(name))
        for name in names:
            seq = int(name[:8])
            if seq <= state.records_done and seq <= state.frames_seen:
                continue
            if read >= budget:
                break
            read += 1
            self._read_batch(state, folder / name, seq)
        ended = (folder / "end.json").exists()
        if state.encode is None and state.pending and now >= state.retry_at:
            self._maybe_encode(state, now, ended and not [n for n in names if int(n[:8]) > state.frames_seen])
        self._cleanup(state, folder, ended, now)
        return read

    def _open(self, stream: str, folder: Path) -> RunState | None:
        try:
            meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None  # the client writes run.json first, but the folder may be seen between the two
        except ValueError as error:
            if str(folder) not in self.bad_logged:
                self.bad_logged.add(str(folder))
                self.log(f"spool {stream}/{folder.name}: run.json does not parse, run left alone: {error}")
            return None
        project = meta.get("project") if isinstance(meta, dict) else None
        records = self.store.cursor(stream, folder.name)
        frames = self.store.cursor(stream, f"{folder.name}:frames")
        began = records["mtime"] if records and records["mtime"] else time.time()
        if records is None or not records["mtime"]:  # a run's first sight starts its idle clock, kept across restarts
            self.store.set_cursor(stream, folder.name, offset=records["offset"] if records else 0, mtime=began)
        return RunState(stream, folder.name, folder, project if isinstance(project, str) else None,
                        records["offset"] if records else 0, frames["offset"] if frames else 0, progress_at=began)

    def _read_batch(self, state: RunState, path: Path, seq: int) -> None:
        try:
            lines = parse_batch(path.read_text(encoding="utf-8"))
            frames = []
            for line in lines:
                if "frame" in line:
                    _, i = FRAME.match(line["frame"]).groups()
                    frame_path = state.folder / line["frame"]
                    if not frame_path.is_file():
                        raise ValueError(f"frame {line['frame']} is not in the spool")
                    frames.append(Frame(seq * 1000 + int(i), line["wall"], line["t"], frame_path, line["w"], line["h"]))
        except (ValueError, OSError) as error:
            self._bad(state, path, seq, error)
            return
        if seq > state.frames_seen:
            state.frames_seen = seq
            state.pending += [frame for frame in frames if frame.pos > state.frames_done]
        if seq > state.records_done:
            rows = [(line["wall"], json.dumps(line["data"], ensure_ascii=False)) for line in lines if "data" in line]
            first = next((line for line in lines if line["t"] is not None), None)
            clock = (state.run, first["wall"], first["t"]) if first else None
            state.progress_at = time.time()
            self.store.add_records(state.stream, rows, cursor=(state.run, {"offset": seq, "mtime": state.progress_at}),
                                   project=state.project, clock=clock)
            state.records_done = seq

    def _bad(self, state: RunState, path: Path, seq: int, error: Exception) -> None:
        """A batch that does not parse moves to `spool/_bad/` with its error beside it; the run goes on after it."""
        bad = self.root / "_bad"
        bad.mkdir(exist_ok=True)
        target = bad / f"{state.stream}-{state.run}-{path.name}"
        shutil.move(str(path), target)
        target.with_name(target.name + ".error").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        for frame in state.folder.glob(f"{seq:08d}-*.jpg"):
            frame.unlink(missing_ok=True)
        self.log(f"spool {state.stream}/{state.run}: batch {path.name} moved to _bad: {error}")
        state.progress_at = time.time()
        self.store.set_cursor(state.stream, state.run, offset=max(seq, state.records_done), mtime=state.progress_at)
        state.records_done = max(seq, state.records_done)
        state.frames_seen = max(seq, state.frames_seen)

    # frames into segments

    def _maybe_encode(self, state: RunState, now: float, ended: bool) -> None:
        first = state.pending[0]
        source = self.config.sources.get(state.stream)
        segment_s = source.get("segment_s", 60) if source else 60
        # a segment holds frames of one size within `segment_s` of its first: a frame of another size, or one past that,
        # starts the next, so the frames before it go now, and a backlog after an outage is cut into segments as well
        group = state.pending[:next((i for i, f in enumerate(state.pending)
                                     if (f.w, f.h) != (first.w, first.h) or f.wall - first.wall >= segment_s),
                                    len(state.pending))]
        due = len(group) < len(state.pending) or now - group[-1].wall >= IDLE_FRAMES_S or ended
        if not due:
            return
        try:
            self.store.reserve(sum(frame.path.stat().st_size for frame in group), state.stream)
        except ApiError as error:  # the encode would pass the floor or the cap: the frames wait in the spool
            state.retry_at = now + ENCODE_RETRY_S
            self._fail(state.stream, error.message)
            return
        gaps = [b.wall - a.wall for a, b in zip(group, group[1:])]
        durations = gaps + [statistics.median(gaps) if gaps else 1.0]
        name = f"{group[0].pos // 1000:08d}-{group[0].pos % 1000:03d}.mkv"
        target = self.store.streams / state.stream / state.run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        listing = target.with_name(target.stem + ".ffconcat")
        lines = ["ffconcat version 1.0"]
        for frame, length in zip(group, durations):
            quoted = frame.path.resolve().as_posix().replace("'", "'\\''")
            # option framerate makes the jpeg demuxer's timebase 1 ms, or the frame times snap to 1/25 s
            lines += [f"file '{quoted}'", "option framerate 1000", f"duration {max(length, 0.001):.6f}"]
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        args = [self.ffmpeg, "-y", "-hide_banner", "-nostats", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", str(listing), "-fps_mode", "passthrough", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                *capture.CAMERA, "-pix_fmt", "yuv420p", *capture.keyframes(segment_s), "-f", "matroska", str(part)]
        errors = target.with_name(target.stem + ".err")
        with open(errors, "wb") as err:
            process = children.popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err)
        state.encode = Encode(process, group, durations, part, target, errors)

    def _poll_encode(self, state: RunState) -> None:
        job = state.encode
        if job is None or job.process.poll() is None:
            return
        state.encode = None
        listing = job.target.with_name(job.target.stem + ".ffconcat")
        try:
            if job.process.returncode != 0:
                message = job.errors.read_text(encoding="utf-8", errors="replace").strip()[-400:]
                self.log(f"spool {state.stream}/{state.run}: encoding {len(job.frames)} frames failed "
                         f"({job.process.returncode}), again in {ENCODE_RETRY_S:g} s: {message}")
                job.part.unlink(missing_ok=True)
                state.retry_at = time.time() + ENCODE_RETRY_S
                return
            os.replace(job.part, job.target)
            first, last = job.frames[0], job.frames[-1]
            with self.store.tx():
                self.store.add_item(state.stream, "video", first.wall, last.wall + job.durations[-1],
                                    self.store.relative(job.target), job.target.stat().st_size,
                                    meta={"run": state.run, "frames": len(job.frames)}, project=state.project)
                self.store.set_cursor(state.stream, f"{state.run}:frames", offset=last.pos)
            state.frames_done = last.pos
            state.pending = [frame for frame in state.pending if frame.pos > last.pos]
        finally:
            listing.unlink(missing_ok=True)
            job.errors.unlink(missing_ok=True)

    # cleanup

    def _cleanup(self, state: RunState, folder: Path, ended: bool, now: float) -> None:
        """Delete batches and frames both cursors are past; remove a finished run, or one idle for a day."""
        first_waiting = state.pending[0].pos // 1000 if state.pending else None
        names = sorted(name for name in os.listdir(folder) if BATCH.match(name))
        unread = any(int(name[:8]) > state.records_done or int(name[:8]) > state.frames_seen for name in names)
        for name in list(names):
            seq = int(name[:8])
            if seq <= state.records_done and seq <= state.frames_seen and (first_waiting is None or seq < first_waiting):
                (folder / name).unlink(missing_ok=True)
                for frame in folder.glob(f"{seq:08d}-*.jpg"):
                    frame.unlink(missing_ok=True)
                names.remove(name)
        if state.encode is not None:
            return
        idle = not unread and now - state.progress_at > IDLE_RUN_S  # by its cursor, not by the files' times
        if (ended or idle) and not names and not state.pending:
            if idle and not ended:
                self.log(f"spool {state.stream}/{state.run}: no end and no batch for a day: removed")
            shutil.rmtree(folder, ignore_errors=True)
            self.store.drop_cursor(state.stream, state.run)
            self.store.drop_cursor(state.stream, f"{state.run}:frames")
            del self.runs[(state.stream, state.run)]
