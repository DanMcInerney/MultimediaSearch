"""ffmpeg sources (DESIGN.md §6): one ffmpeg per source, writing segments into a folder per launch, whose list is
tailed into the index. The kinds screen, microphone and camera name a device and expand to an ffmpeg input
(`opens`); `ffmpeg` takes the input as written. All wait for the owner's approval (consent.py). The daemon's loop calls
`step`; `reconcile` runs once at startup; `apply` takes a changed config."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from mms import children, consent
from mms.config import Config, Source, ffmpeg_exe, stamp
from mms.media import duration, has_sound
from mms.store import Store

log = logging.getLogger("mms.capture")

KINDS = ("screen", "microphone", "camera", "ffmpeg")
PRESET_OF = {"screen": "screen", "microphone": "audio", "camera": "camera"}
EXTENSIONS = {"video": ".mkv", "audio": ".mka"}
STOP_WAIT_S = 10  # how long ffmpeg gets to close its segment after `q`
CAMERA = ("-c:v", "libx264", "-crf", "28", "-tune", "zerolatency")  # the camera preset's codec, which the spool uses too


def keyframes(segment_s: float) -> list[str]:
    """The key frame interval of the video presets: `min(10, segment_s)` s, since a segment ends only on a key frame."""
    return ["-force_key_frames", f"expr:gte(t,n_forced*{min(10, segment_s):g})"]


def opens(source: Source, platform: str | None = None) -> list[str]:
    """The ffmpeg input a source opens: the named device of a screen, microphone or camera expanded for the platform
    (DESIGN.md §4.1), or an ffmpeg source's own `input`."""
    if source.kind == "ffmpeg":
        return list(source.options["input"])
    windows = (platform or sys.platform) == "win32"
    device = source.device()
    fps = source.get("fps")
    if source.kind == "screen":
        rate = f"{fps if fps is not None else 2:g}"
        if windows:
            return ["-f", "lavfi", "-i", f"ddagrab=output_idx={device}:framerate={rate}"]
        return ["-f", "x11grab", "-framerate", rate, "-i", device]
    rate_args = [] if fps is None else ["-framerate", f"{fps:g}"]
    if source.kind == "microphone":
        return ["-f", "dshow", "-i", f"audio={device}"] if windows else ["-f", "pulse", "-i", device]
    if source.kind == "camera":
        return ["-f", "dshow", *rate_args, "-i", f"video={device}"] if windows else ["-f", "v4l2", *rate_args, "-i", device]
    raise ValueError(f"source {source.name!r}: a {source.kind} source opens no ffmpeg input")


def command(source: Source, ffmpeg: str = "ffmpeg") -> tuple[list[str], str]:
    """The ffmpeg command line for a source, and its media: `video` or `audio`. It writes `%06d.<ext>` and `list.csv`
    in the working directory. `encode`, when given, replaces everything the preset adds between input and output."""
    given = opens(source)
    segment_s = source.get("segment_s", 60)
    preset = PRESET_OF.get(source.kind) or (None if source.get("preset", "none") == "none" else source.get("preset"))
    encode = source.get("encode")
    media = source.get("media") or ("audio" if preset == "audio" or "-vn" in (encode or []) else "video")
    if encode is None:
        if preset == "screen":
            max_width = source.get("max_width", 1920)
            # a wider frame is scaled down (even height, same aspect) before mpdecimate compares it; the bgra of a
            # real grab leaves as yuv420p, not yuv444p
            filters = f"scale='min(iw,{max_width})':-2,format=yuv420p,mpdecimate=max=119"
            if any(arg.startswith("ddagrab") for arg in given):
                filters = "hwdownload,format=bgra," + filters
            encode = ["-vf", filters, "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "30", "-tune", "zerolatency",
                      *keyframes(segment_s)]
        elif preset == "camera":
            encode = [*CAMERA, *keyframes(segment_s)]
        else:
            encode = ["-af", "aresample=async=1", "-c:a", "libopus", "-b:a", "24k"]
    args = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "warning", *given, *encode,
            "-f", "segment", "-segment_time", f"{segment_s:g}", "-segment_format", "matroska",
            "-reset_timestamps", "1", "-segment_list", "list.csv", "-segment_list_type", "csv",
            f"%06d{EXTENSIONS[media]}"]
    return args, media


# the devices `source_set` may name

DSHOW = re.compile(r'"([^"]+)" \((audio|video)\)')


def parse_dshow(listing: str) -> dict:
    """The audio and video devices in ffmpeg's `-list_devices true -f dshow -i dummy` output."""
    found: dict[str, list] = {"audio": [], "video": []}
    for name, kind in DSHOW.findall(listing):
        if {"device": name} not in found[kind]:
            found[kind].append({"device": name})
    return found


def displays() -> list[dict]:
    """The displays: Windows' monitors as ddagrab numbers them, else the X display."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        sizes: list[dict] = []
        callback = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                      wintypes.LPARAM)

        def seen(_monitor, _dc, rect, _data):
            box = rect.contents
            sizes.append({"device": str(len(sizes)), "width": box.right - box.left, "height": box.bottom - box.top})
            return 1

        ctypes.windll.user32.EnumDisplayMonitors(None, None, callback(seen), 0)
        return sizes
    import os
    return [{"device": os.environ["DISPLAY"]}] if os.environ.get("DISPLAY") else []


def devices(ffmpeg: str, platform: str | None = None) -> dict:
    """What `source_set` may name: displays, microphones and cameras. A listing that fails is given as `error`, not
    left empty."""
    out: dict = {"displays": [], "microphones": [], "cameras": []}
    try:
        out["displays"] = displays()
        if (platform or sys.platform) == "win32":
            listing = children.run([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
            found = parse_dshow(listing.stdout + listing.stderr)
            out["microphones"], out["cameras"] = found["audio"], found["video"]
        else:
            out["cameras"] = [{"device": str(path)} for path in sorted(Path("/dev").glob("video*"))]
            pulse = children.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, timeout=10)
            out["microphones"] = [{"device": line.split("\t")[1]} for line in pulse.stdout.splitlines()
                                  if "\t" in line and ".monitor" not in line]
    except (OSError, subprocess.SubprocessError) as error:
        out["error"] = f"{type(error).__name__}: {error}"
    return out


# list lines and segments

def read_list(folder: Path) -> list[tuple[str, float, float]]:
    """The complete lines of a launch's `list.csv` as (file, start, end); a half-written last line is left."""
    try:
        raw = (folder / "list.csv").read_bytes()
    except FileNotFoundError:
        return []
    lines = []
    for line in raw[:raw.rfind(b"\n") + 1].decode("utf-8", "replace").splitlines():
        parts = line.strip().rsplit(",", 2)
        try:
            lines.append((parts[0], float(parts[1]), float(parts[2])))
        except (IndexError, ValueError):
            continue
    return lines


def segment_meta(file: Path) -> dict | None:
    """What a closed segment records of itself: `sound` when a video segment has an audio stream (the fact
    `media.of_file` gives a file), which ffmpeg's own arguments cannot tell for an `ffmpeg` source. The reader needs it
    to hand a sound-only decider a video's soundtrack. One PyAV open, about 1 to 6 ms."""
    return {"sound": True} if file.suffix == EXTENSIONS["video"] and has_sound(file) else None


def index_lines(store: Store, launch: dict, project: str | None = None) -> None:
    """Add an item for each list line of the launch not yet indexed. `launch` is its row, kept up to date here."""
    folder = store.home / launch["folder"]
    lines = read_list(folder)
    new = lines[launch["lines"]:]
    if not new:
        return
    first = launch["first_start"] if launch["first_start"] is not None else new[0][1]
    found = []
    for name, start, end in new:  # probed before the transaction opens, which holds the database's one writer
        try:
            found.append((name, start, end, (folder / name).stat().st_size, segment_meta(folder / name)))
        except FileNotFoundError:
            log.warning("%s: segment %s is gone; not indexed", launch["folder"], name)
    with store.tx():
        for name, start, end, size, meta in found:
            store.add_item(launch["source"], "audio" if name.endswith(".mka") else "video",
                           launch["started"] + (start - first), launch["started"] + (end - first),
                           store.relative(folder / name), size, meta=meta, project=project)
        store.update_launch(launch["id"], lines=len(lines), first_start=first)
    launch.update(lines=len(lines), first_start=first)


def index_orphans(store: Store, launch: dict, project: str | None = None) -> float:
    """Settle the media files no list line names, the segment a crash cut short: index each after the last line's end
    if PyAV can read it, else delete it. Returns where the launch's indexed time ends."""
    folder = store.home / launch["folder"]
    lines = read_list(folder)
    end_t = launch["started"]
    if lines:
        end_t += lines[-1][2] - (launch["first_start"] if launch["first_start"] is not None else lines[0][1])
    named = {name for name, _, _ in lines}
    suffixes = tuple(EXTENSIONS.values())
    for file in sorted(folder.iterdir()):
        if file.suffix not in suffixes or file.name in named:
            continue
        path = store.relative(file)
        if store.has_path(path):
            continue
        length = duration(file)
        if length is None:
            try:
                file.unlink()
            except OSError as error:
                log.warning("%s: cannot delete the unreadable segment %s: %s", launch["folder"], file.name, error)
            continue
        store.add_item(launch["source"], "audio" if file.suffix == ".mka" else "video", end_t, end_t + length, path,
                       file.stat().st_size, meta=segment_meta(file), project=project)
        end_t += length
    return end_t


def settle(store: Store, launch: dict, project: str | None = None) -> float:
    """Index everything a launch's folder holds, once its ffmpeg is gone. Returns where its indexed time ends."""
    index_lines(store, launch, project)
    end_t = index_orphans(store, launch, project)
    folder = store.home / launch["folder"]
    if not any(file.suffix in EXTENSIONS.values() for file in folder.iterdir()):
        shutil.rmtree(folder, ignore_errors=True)  # a launch that never wrote a segment is not kept
    return end_t


class Run:
    """One ffmpeg launch."""

    def __init__(self, launch: dict, process: subprocess.Popen, source: Source):
        self.launch = launch
        self.process = process
        self.source = source  # as it was when this ffmpeg started
        self.errors: deque[str] = deque(maxlen=20)  # ffmpeg's last stderr lines
        self.listed = -1  # the size of list.csv when last read
        self.thread = threading.Thread(target=self._drain, name=f"ffmpeg-{launch['source']}", daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        for raw in self.process.stderr:
            self.errors.append(raw.decode("utf-8", "replace").rstrip())

    def tail(self, store: Store) -> None:
        try:
            size = (store.home / self.launch["folder"] / "list.csv").stat().st_size
        except FileNotFoundError:
            return
        if size != self.listed:
            index_lines(store, self.launch, self.source.project)
            self.listed = size

    def ask_stop(self) -> None:
        try:
            self.process.stdin.write(b"q\n")
            self.process.stdin.close()
        except (OSError, ValueError):
            pass  # already gone


class Capture:
    def __init__(self, store: Store, config: Config):
        self.store = store
        self.config = config
        self.ffmpeg = ffmpeg_exe(config)
        self.sources: dict[str, Source] = {}
        self.approved: dict[str, bool] = {}
        self.runs: dict[str, Run] = {}
        self.fails: dict[str, int] = {}  # short runs in a row, for the backoff
        self.restarts: dict[str, int] = {}
        self.retry_at: dict[str, float] = {}
        self.error: dict[str, str] = {}
        self.paused = False
        self.stopped = False
        self.apply(config)

    def apply(self, config: Config) -> None:
        """A changed config (DESIGN.md §4.2): the ffmpegs of removed, changed and no longer approved sources stop, new
        ones start at the next step, and the rest are left alone. Only what ffmpeg runs counts as a change: a cap or a
        project does not restart a source."""
        wanted = {name: source for name, source in config.sources.items() if source.kind in KINDS}
        consent.prune(config.home, {name for name, source in config.sources.items() if source.what() is not None})
        approved = {name: consent.approved(config.home, name, source.what()) for name, source in wanted.items()}
        ffmpeg = ffmpeg_exe(config)

        def same(name: str, run: Run) -> bool:
            source = wanted.get(name)
            return (source is not None and approved[name] and ffmpeg == self.ffmpeg
                    and (source.kind, source.options) == (run.source.kind, run.source.options))

        leaving = [name for name, run in self.runs.items() if not same(name, run)]
        self._end([self.runs.pop(name) for name in leaving])
        for name, source in wanted.items():
            old = self.sources.get(name)
            if (old is None or (old.kind, old.options) != (source.kind, source.options)
                    or approved[name] != self.approved.get(name)):
                self.retry_at.pop(name, None)  # a changed or newly approved source starts at once
                self.error.pop(name, None)
            self.fails.setdefault(name, 0)
            self.restarts.setdefault(name, 0)
        self.config, self.ffmpeg, self.sources, self.approved = config, ffmpeg, wanted, approved

    def reconcile(self) -> None:
        """At startup, before any ffmpeg runs: index list lines never indexed, and settle crash-cut segments. Every
        ffmpeg of an earlier daemon died with it (children.py), so a launch's folder is quiet."""
        for launch in self.store.launches():
            if launch["ended"] is not None:
                continue
            source = self.config.sources.get(launch["source"])
            if (self.store.home / launch["folder"]).is_dir():
                ended = settle(self.store, launch, source.project if source else None)
            else:
                ended = launch["started"]
            self.store.update_launch(launch["id"], ended=ended)

    def step(self, now: float) -> None:
        """Start, restart and tail: called by the daemon's loop about four times a second."""
        if self.stopped:
            return
        for name in self.sources:
            run = self.runs.get(name)
            if run is not None:
                run.tail(self.store)
                if run.process.poll() is not None:
                    self._exited(name, run, now)
                elif int(run.launch["started"] // 86400) != int(now // 86400):
                    log.info("%s: a new UTC day: restarting ffmpeg", name)
                    self._end([run])
                    del self.runs[name]
            if (name not in self.runs and self.approved[name] and not self.paused
                    and now >= self.retry_at.get(name, 0)):
                self._start(name, now)

    def pause(self, paused: bool) -> None:
        """The free-space floor: stop every ffmpeg while paused, and start them again after."""
        if paused == self.paused:
            return
        self.paused = paused
        if paused:
            log.warning("capture paused")
            self._end(list(self.runs.values()))
            self.runs.clear()
        else:
            log.info("capture resumed")

    def stop(self) -> None:
        """Write `q` to every ffmpeg, wait, and index their last segments."""
        self.stopped = True
        self._end(list(self.runs.values()))
        self.runs.clear()

    def status(self) -> dict:
        """Per source: `running`, `paused`, `waiting` (between ffmpegs), `needs_approval` or `stopped`; its launch,
        segment count and ffmpeg's pid."""
        out = {}
        for name in list(self.sources):
            run = self.runs.get(name)
            state = ("stopped" if self.stopped else "needs_approval" if not self.approved[name]
                     else "paused" if self.paused else "running" if run else "waiting")
            entry = {"state": state, "launch": run.launch["folder"] if run else None,
                     "segments": run.launch["lines"] if run else 0, "restarts": self.restarts[name],
                     "pid": run.process.pid if run else None}
            if name in self.error and state == "waiting":
                entry["error"] = self.error[name]
            out[name] = entry
        return out

    # starting and ending launches

    def _start(self, name: str, now: float) -> None:
        source = self.sources[name]
        args, _ = command(source, self.ffmpeg)
        started = time.time()
        folder = self.store.streams / name / stamp(started)
        number = 1
        while folder.exists():
            number += 1
            folder = folder.with_name(f"{stamp(started)}-{number}")
        folder.mkdir(parents=True)
        launch = {"source": name, "folder": self.store.relative(folder), "started": started, "ended": None,
                  "lines": 0, "first_start": None}
        launch["id"] = self.store.add_launch(name, launch["folder"], started)
        try:
            process = children.popen(args, cwd=folder, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        except OSError as error:
            log.error("%s: cannot start ffmpeg: %s", name, error)
            self.error[name] = str(error)
            self.store.update_launch(launch["id"], ended=started)
            shutil.rmtree(folder, ignore_errors=True)
            self._back_off(name, now, 0)
            return
        self.runs[name] = Run(launch, process, source)
        self.error.pop(name, None)
        log.info("%s: ffmpeg started in %s", name, launch["folder"])

    def _exited(self, name: str, run: Run, now: float) -> None:
        """ffmpeg exited that was not asked to: index what it left, and start again after a backoff."""
        run.thread.join(2)
        self._finish(run)
        del self.runs[name]
        self.restarts[name] += 1
        self.error[name] = " | ".join(run.errors) or f"exited with {run.process.returncode}"
        log.warning("%s: ffmpeg exited with %s: %s", name, run.process.returncode, self.error[name])
        self._back_off(name, now, time.time() - run.launch["started"])

    def _back_off(self, name: str, now: float, lived: float) -> None:
        self.fails[name] = 1 if lived >= 60 else self.fails[name] + 1
        self.retry_at[name] = now + min(60, 2 ** (self.fails[name] - 1))

    def _end(self, runs: list[Run]) -> None:
        """Ask each ffmpeg to stop, so its open segment closes; kill what is still running after a wait."""
        for run in runs:
            run.ask_stop()
        deadline = time.monotonic() + STOP_WAIT_S
        for run in runs:
            try:
                run.process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                log.warning("%s: ffmpeg did not stop; killing it", run.launch["source"])
                run.process.kill()
                run.process.wait()
            run.thread.join(2)
            self._finish(run)

    def _finish(self, run: Run) -> None:
        settle(self.store, run.launch, run.source.project)
        self.store.update_launch(run.launch["id"], ended=time.time())
