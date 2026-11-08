"""`mms start` (DESIGN.md §10): one daemon per home. It takes the lock, reads the config (a bad file gives no sources and a
`config_error`), opens the store, runs the janitor's pass and capture's reconcile, starts the API and writes `daemon.json`,
then loops: it reloads `mms.toml` and `consent.json` when they change, steps capture and ingest, runs the janitor
and sweeps jobs. It stops when `mms stop` writes `daemon.stop`, or on Ctrl+C. Its children die with it (children.py).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from mms import api, children, client, consent, ops
from mms import config as configuration
from mms.config import Config, ConfigError, iso
from mms.consent import ConsentError
from mms.lock import HomeLock
from mms.store import IndexVersionError, Store

STOP, LOG = "daemon.stop", "daemon.log"
REPEAT_S = 60.0  # a step that keeps failing the same way is logged this often
START_WAIT_S = 2.0  # a starting daemon waits this long for the lock: a probe or a daemon that is stopping holds it
DETACH_WAIT_S = 30.0  # `mms start --detach` waits this long for the daemon to answer


class LogFile:
    """A detached daemon's stdout and stderr: `daemon.log`, opened for each write, so the janitor can cut it."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def write(self, text: str) -> int:
        with self.lock, open(self.path, "a", encoding="utf-8", errors="replace") as out:
            out.write(text)
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def log(message: str) -> None:
    print(f"{iso(time.time())} {message}", flush=True)


def _stamp(home: Path) -> tuple:
    out = []
    for name in (configuration.FILE, consent.FILE):
        try:
            stat = (home / name).stat()
            out.append((stat.st_size, stat.st_mtime_ns))
        except OSError:
            out.append(None)
    return tuple(out)


class Daemon:
    """`parts` may name a `capture`, `ingest` or `janitor` to use in place of the real ones, for tests."""

    def __init__(self, home: Path | str | None = None, *, detached: bool = False, **parts):
        self.home = Path(home).expanduser() if home else configuration.default_home()
        self.detached = detached
        self.parts = parts
        self.stopping = False
        self.failures: dict[str, tuple[str, float]] = {}
        self.pid = os.getpid()
        self.started = 0.0
        self.paused = False

    # what the ops ask of a running daemon (ops.Env.live)

    def floor(self) -> dict:
        return {"breached": self.janitor.breached, "free_gb": self.janitor.free_gb,
                "floor_gb": self.env.config.free_gb}

    def source_states(self) -> dict:
        states = {**self.ingest.status(), **self.capture.status()}
        if self.paused:
            for name, entry in states.items():
                if entry["state"] in ("running", "waiting"):
                    entry["state"] = "paused"
        return states

    def decider(self) -> dict:
        return self.gpu.status()

    def job_counts(self) -> dict:
        return self.jobs.counts()

    # running

    def run(self) -> int:
        """Run until stopped; returns the exit code."""
        self.home.mkdir(parents=True, exist_ok=True)
        if self.detached:
            sys.stdout = sys.stderr = LogFile(self.home / LOG)
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
        lock = HomeLock(self.home / client.LOCK)
        if not lock.acquire(START_WAIT_S):
            endpoint = client.read(self.home)
            print(f"mms: a daemon is already running in {self.home}" + (f" (pid {endpoint.pid})" if endpoint else ""),
                  file=sys.stderr, flush=True)
            return 1
        try:
            try:
                children.contain()
            except OSError as error:
                log(f"children may outlive the daemon: {error}")
            try:
                self.loop()
            except IndexVersionError as error:
                print(f"mms: {error}", file=sys.stderr, flush=True)
                return 2
        finally:
            for name in (STOP, client.FILE):
                try:
                    (self.home / name).unlink()
                except OSError:
                    pass
            lock.release()
        return 0

    def loop(self) -> None:
        (self.home / STOP).unlink(missing_ok=True)  # one left over from a daemon that died
        self.catch_signals()
        try:
            config, config_error = configuration.load(self.home), None
            consent.load(self.home)
        except (ConfigError, ConsentError) as error:
            config, config_error = Config(home=self.home), str(error)
            log(f"{error}: running with no sources until it is fixed")
        self.stamps = _stamp(self.home)
        store = Store(self.home)
        from mms.capture import Capture
        from mms.gpu import Gpu
        from mms.ingest import Ingest
        from mms.janitor import Janitor
        from mms.jobs import Jobs
        from mms.spool import cleanup_parts
        self.env = ops.Env(self.home, store, config, config_error, live=self)
        self.janitor = self.parts.get("janitor") or Janitor(store, config, log)
        self.ingest = self.parts.get("ingest") or Ingest(store, config, log)
        self.capture = self.parts.get("capture") or Capture(store, config)
        self.gpu = Gpu(config)
        self.jobs = Jobs(self.env, self.gpu)
        self.started = time.time()
        server = api.Api(self.env, self.jobs, self.started)
        log(f"mms daemon started in {self.home} (pid {self.pid}), {len(config.sources)} sources")
        try:
            self.paused = bool(self.janitor.step(self.started))
            self.capture.reconcile()
            cleanup_parts(store)
            if self.paused:
                self.hold(True)
            server.start()
            next_janitor = self.started + config.janitor_s
            while not self.stopping and not (self.home / STOP).exists():
                now = time.time()
                config = self.reload(config)
                self.guard("capture", self.capture.step, now)
                if not self.paused:
                    self.guard("ingest", self.ingest.step, now)
                if now >= next_janitor:
                    next_janitor = now + config.janitor_s
                    breached = self.guard("janitor", self.janitor.step, now)
                    if breached is not None and breached != self.paused:
                        self.paused = breached
                        self.hold(self.paused)
                self.guard("jobs", self.jobs.sweep, now)
                time.sleep(config.loop_s)
        except KeyboardInterrupt:
            pass
        finally:
            log("mms daemon stopping")
            self.guard("jobs stop", self.jobs.stop)
            self.guard("api stop", server.stop)
            for name, part in (("capture", self.capture), ("ingest", self.ingest), ("janitor", self.janitor),
                               ("gpu", self.gpu)):
                self.guard(f"{name} stop", part.stop)
            store.close()
            log("mms daemon stopped")

    def reload(self, config: Config) -> Config:
        """When `mms.toml` or `consent.json` changed, read them again and hand the new config to each part. A file
        that does not parse is refused: the last good config stays, and `status` gives the error."""
        stamps = _stamp(self.home)
        if stamps == self.stamps:
            return config
        self.stamps = stamps
        try:
            new = configuration.load(self.home)
            consent.load(self.home)
        except (ConfigError, ConsentError) as error:
            self.env.config_error = str(error)
            log(f"{error}: keeping the last good config")
            return config
        self.env.config, self.env.config_error = new, None
        for name, part in (("capture", self.capture), ("ingest", self.ingest), ("janitor", self.janitor),
                           ("gpu", self.gpu)):
            self.guard(f"{name} apply", part.apply, new)
        self.jobs.apply()
        log(f"config reloaded: {len(new.sources)} sources")
        return new

    def hold(self, paused: bool) -> None:
        """The floor: capture pauses and ingest stops (its commands end, and its watches pick up where their cursors
        are) while it is breached, and both start again after."""
        self.guard("capture pause", self.capture.pause, paused)
        if paused:
            self.guard("ingest stop", self.ingest.stop)

    def guard(self, name: str, step, *args):
        """Run one step; a failure is logged, and the loop goes on."""
        try:
            return step(*args)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            last, when = self.failures.get(name, (None, 0.0))
            if message != last or time.time() - when > REPEAT_S:
                self.failures[name] = (message, time.time())
                log(f"{name} failed: {traceback.format_exc().strip()}")
            return None

    def catch_signals(self) -> None:
        """Ctrl+C, Ctrl+Break and a terminate end the loop, so everything is stopped in order."""
        if threading.current_thread() is not threading.main_thread():
            return

        def stop(*_):
            self.stopping = True

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), stop)


# `mms start --detach`

def start_detached(home: Path) -> int:
    """Start the daemon in the background and return once it answers (exit code 0), or 1 with the log's tail. A daemon
    that is running or starting counts: this waits for that one."""
    home.mkdir(parents=True, exist_ok=True)
    alive = None if client.is_running(home) else _spawn(home)
    answer = client.wait_until_answering(
        home, DETACH_WAIT_S, alive=lambda: alive is None or alive() or client.is_running(home))
    if answer is not None:
        print(f"daemon running (pid {answer['pid']})", flush=True)
        return 0
    try:
        tail = "\n".join((home / LOG).read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
    except OSError:
        tail = "(no daemon.log)"
    died = alive is not None and not alive()
    print(f"mms: the daemon {'exited' if died else f'did not answer within {DETACH_WAIT_S:g} s'}; daemon.log ends:\n{tail}",
          file=sys.stderr)
    return 1


def _spawn(home: Path):
    """Start the daemon as nobody's descendant, so that it outlives whatever started it, a job object included (the MCP
    SDK's client kills the adapter's whole tree, and a breakaway flag is accepted but does nothing). Windows: through WMI,
    whose new process belongs to WmiPrvSE and has none of our environment, hence `--home` on the command line and the
    windowless interpreter. Elsewhere: a new session. Returns a function that says whether the daemon's process lives."""
    command = ["-m", "mms.cli", "--home", str(home), "start", "--detached"]
    if os.name != "nt":
        process = subprocess.Popen([sys.executable, *command], start_new_session=True, cwd=home, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return lambda: process.poll() is None
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    line = subprocess.list2cmdline([str(pythonw if pythonw.exists() else sys.executable), *command])
    script = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
              "-Arguments @{CommandLine = $env:MMS_COMMAND; CurrentDirectory = $env:MMS_CWD}; "
              "if ($r.ReturnValue -ne 0) { [Console]::Error.WriteLine('Win32_Process.Create returned ' + $r.ReturnValue); exit 1 }; "
              "[Console]::Out.WriteLine($r.ProcessId)")
    done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
                          env=dict(os.environ, MMS_COMMAND=line, MMS_CWD=str(home)), stdin=subprocess.DEVNULL, timeout=60)
    if done.returncode != 0:
        raise OSError(f"could not start the daemon through WMI: {done.stderr.strip() or done.stdout.strip()}")
    pid = int(done.stdout.split()[-1])
    return lambda: _alive(pid)


def _alive(pid: int) -> bool:
    """Whether a Windows process has not exited."""
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
    kernel.CloseHandle(ctypes.c_void_p(handle))
    return code.value == 259  # STILL_ACTIVE
