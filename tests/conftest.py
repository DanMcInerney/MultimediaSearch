"""Shared fixtures (TESTBED.md): a home per test, the `mms` CLI as a process, the daemon as a process or in a thread, the
API as a test calls it, and a few recordings made with ffmpeg."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from mms import config as configuration
from mms import consent
from mms.config import ffmpeg_exe

from corpus import build as build_corpus

ROOT = Path(__file__).resolve().parents[1]
FAKE = f"python:{ROOT / 'tests' / 'fake_decider.py'}"
DECIDER = f'[decider]\nmodel = "{FAKE.replace(chr(92), "/")}"\n'
OTHER = f"python:{ROOT / 'tests' / 'other_decider.py'}"  # a decider whose facts are not d1's
OTHER_DECIDER = f'[decider]\nmodel = "{OTHER.replace(chr(92), "/")}"\n'


def span(found: dict) -> tuple[float, float]:
    """A sighting's (or a candidate's) start and end as epoch seconds: `t` is both, else `t0` and `t1`."""
    return (configuration.parse_when(found.get("t") or found["t0"]),
            configuration.parse_when(found.get("t") or found["t1"]))


@pytest.fixture(autouse=True)
def no_hugging_face(monkeypatch):
    """A test never reaches the hub: importing huggingface_hub in the test's process fails (a test that fakes the hub puts its
    own module in its place), and a process a test starts gets HF_HUB_OFFLINE (`env`)."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)


@pytest.fixture(autouse=True)
def no_real_uv(monkeypatch):
    """`mms model use` and `prepare` build an env with uv, which downloads gigabytes: a test that gets that far runs a script
    that fails instead. A test that wants to see uv's calls puts a recording one in its place (test_model_cli's `fake_uv`)."""
    from mms import prepare

    monkeypatch.setattr(prepare, "uv_command", lambda: [sys.executable, "-c", "import sys; sys.exit('no real uv in tests')"])


@pytest.fixture
def home(tmp_path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    return ffmpeg_exe()


def write_config(home: Path, text: str, approve: bool = True) -> None:
    """Write `mms.toml`, and approve every source in it that needs it (as `mms approve` does once the owner has
    typed the name), unless told not to."""
    (home / "mms.toml").write_text(text, encoding="utf-8")
    if approve:
        for name, source in configuration.load(home).sources.items():
            if source.what() is not None:
                consent.record(home, name, source.what())


def env(home: Path) -> dict:
    out = dict(os.environ, MMS_HOME=str(home), PYTHONUTF8="1", HF_HUB_OFFLINE="1")  # a test never downloads
    out.pop("PYTHONPATH", None)
    return out


D1_KEYS = {"d1_attention": "set", "d1_set_size": 16, "d1_vision_max_pixels": 1638400, "d1_numbers": True, "d1_codes": 2}
D1_3B_CONFIG = {"model_type": "lfm2_vl", "architectures": ["Lfm2VlForConditionalGeneration"],
                "auto_map": {"AutoModel": "modeling_d1.D1Model"},
                "text_config": {"model_type": "lfm2", "hidden_size": 2048, "num_hidden_layers": 30}}
D3_FORMAT = {"format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3"}
MODEL_FOLDERS = ("d1-Omni-2B", "d1-2B", "d1-3B", "d3-edge", "d3-mini", "d3-flash", "h2o-lightning-4b",
                 "pplx-decider-flash-v1.2", "unclaimed")


def model_folder(parent: Path, name: str) -> Path:
    """`parent/name`, holding the few files a backend's `claims` reads, with the values the real downloads have, and an
    empty model.safetensors (never real weights). `name` is one of MODEL_FOLDERS; "unclaimed" has a config.json no backend
    reads and no weights."""
    folder = Path(parent) / name
    folder.mkdir(parents=True, exist_ok=True)

    def write(file: str, data) -> None:
        path = folder / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    if name in ("d1-Omni-2B", "d1-2B"):
        write("config.json", {"model_type": "lfm3_omni" if name == "d1-Omni-2B" else "lfm3_vl", **D1_KEYS})
    elif name == "d1-3B":
        write("config.json", D1_3B_CONFIG)
    elif name in ("d3-edge", "d3-mini", "d3-flash"):
        write("config.json", {"model_type": "qwen3_5"})
        write("decision_config.json", {**D3_FORMAT, "base_model": "Qwen/Qwen3.5-9B" if name == "d3-flash" else "Qwen/Qwen3.5-4B"})
        write("MODEL_MANIFEST.json", {"schema": "d3-package-manifest/1", "model_name": name, "format_id": D3_FORMAT["format_id"]})
    elif name == "h2o-lightning-4b":
        write("config.json", {"model_type": "qwen3_5"})
        write("serve_config.json", {"model": "h2oai/h2o-lightning-4b", "base_model": "Qwen/Qwen3.5-4B"})
        write("h2o_lightning_shim.py", "")
    elif name == "pplx-decider-flash-v1.2":
        write("config.json", {"model_type": "qwen3_5"})
        write("joint_config.json", {"version": 3, "template": "joint-question-inline-readout-v1", "marker": "<|fim_suffix|>",
                                    "readout_offset": 0, "max_questions": None, "max_length": None})
        write("decision_config.json", {"format_version": 2, "base_model": "Qwen/Qwen3.5-4B", "max_pixels": 1638400})
        write("source/src/pplx_decider/__init__.py", "")
    elif name == "unclaimed":
        write("config.json", {"model_type": "gpt2"})
    else:
        raise ValueError(f"model_folder knows {', '.join(MODEL_FOLDERS)}, not {name!r}")
    if name != "unclaimed":
        write("model.safetensors", "")
    return folder


def prepared_env(parent: Path, family: str, requirements) -> Path:
    """`parent/family`, an env that `mms model prepare` has made for these requirements: only its mms-env.json (no python).
    `parent` is the envs folder. (Not `env`: that name is the process environment, above.)"""
    folder = Path(parent) / family
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mms-env.json").write_text(json.dumps({"family": family, "requirements": list(requirements), "versions": {},
                                                    "prepared": "2026-11-01T00:00:00+00:00"}), encoding="utf-8")
    return folder


def mms(home: Path, *args: str, check: bool = True, timeout: float = 120, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run the CLI as the agent would, in its own process, in `cwd` (the home, which is in no project, unless given)."""
    done = subprocess.run([sys.executable, "-m", "mms.cli", *args], env=env(home), capture_output=True,
                          text=True, encoding="utf-8", timeout=timeout, cwd=cwd or home)
    if check and done.returncode != 0:
        raise AssertionError(f"mms {' '.join(args)} exited {done.returncode}\n{done.stdout}\n{done.stderr}")
    return done


def mms_json(home: Path, *args: str, **kwargs):
    return json.loads(mms(home, *args, "--json", **kwargs).stdout)


class ApiFailure(Exception):
    """An error the API gave: its status and its `error` object."""

    def __init__(self, status: int, error: dict):
        super().__init__(f"{status} {error}")
        self.status, self.error = status, error
        self.code, self.field = error["code"], error.get("field")


class Api:
    """The daemon's API as a test calls it: `daemon.json` is read at each call, so a restarted daemon is followed."""

    def __init__(self, home: Path):
        self.home = home

    def info(self) -> dict:
        return json.loads((self.home / "daemon.json").read_text(encoding="utf-8"))

    def raw(self, op: str, body=None, token: str | None = "", method: str = "POST") -> tuple[int, dict]:
        info = self.info()
        connection = http.client.HTTPConnection("127.0.0.1", info["port"], timeout=60)
        headers = {} if token is None else {"Authorization": f"Bearer {info['token'] if token == '' else token}"}
        try:
            connection.request(method, f"/v1/{op}", json.dumps(body if body is not None else {}).encode(), headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    def call(self, op: str, **body) -> dict:
        status, data = self.raw(op, body)
        if status != 200:
            raise ApiFailure(status, data["error"])
        return data

    def failure(self, op: str, **body) -> ApiFailure:
        """The error a call must give."""
        try:
            self.call(op, **body)
        except ApiFailure as error:
            return error
        raise AssertionError(f"{op} {body} did not fail")


class Daemon:
    """`mms start` in its own process; `stop()` asks it to stop with `mms stop` and waits."""

    def __init__(self, home: Path):
        self.home = home
        self.log = home / "daemon.test.log"
        self.process = None
        self.api = Api(home)

    def start(self, ready: bool = True) -> "Daemon":
        self._out = open(self.log, "ab")
        self.process = subprocess.Popen([sys.executable, "-m", "mms.cli", "start"], env=env(self.home),
                                        stdout=self._out, stderr=subprocess.STDOUT)
        if ready:
            self.wait_ready()
        return self

    def wait_ready(self) -> None:
        def answers():
            if self.process.poll() is not None:
                raise AssertionError(f"the daemon exited {self.process.returncode}\n{self.text()}")
            try:
                return self.api.call("status")
            except (OSError, ValueError, ApiFailure):
                return None
        wait_for(answers, timeout=60, what="the daemon's API", detail=self.text)

    def stop(self, timeout: float = 30) -> int:
        if self.process is None or self.process.poll() is not None:
            return self.process.returncode if self.process else 0
        mms(self.home, "stop", check=False)
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._out.close()
        return self.process.returncode

    def kill(self) -> None:
        self.process.kill()
        self.process.wait()
        self._out.close()

    def text(self) -> str:
        return self.log.read_text(encoding="utf-8", errors="replace") if self.log.exists() else ""


class Served:
    """A daemon in a thread of the test process: the same code and API as `mms start`, without an interpreter to start.
    Used where a test needs many daemons, such as the scans of `ask`."""

    def __init__(self, home: Path, **parts):
        from mms.daemon import Daemon as Real
        self.home = home
        self.daemon = Real(home, **parts)
        self.thread = threading.Thread(target=self.daemon.run, name="served-daemon", daemon=True)
        self.api = Api(home)

    def start(self) -> "Served":
        self.thread.start()
        wait_for(lambda: (self.home / "daemon.json").exists() and self._answers(), timeout=60, what="the API")
        return self

    def _answers(self) -> bool:
        try:
            self.api.call("status")
            return True
        except (OSError, ValueError, ApiFailure):
            return False

    def stop(self) -> None:
        if self.thread.is_alive():
            (self.home / "daemon.stop").write_text("stop")
            self.thread.join(60)
            assert not self.thread.is_alive(), "the daemon did not stop"


@pytest.fixture
def daemon(home):
    running: list[Daemon] = []

    def start(ready: bool = True) -> Daemon:
        one = Daemon(home).start(ready)
        running.append(one)
        return one

    yield start
    for one in running:
        if one.process and one.process.poll() is None:
            one.stop()


@pytest.fixture
def served(home):
    running: list[Served] = []

    def start(**parts) -> Served:
        one = Served(home, **parts).start()
        running.append(one)
        return one

    yield start
    for one in running:
        one.stop()


def fake_ffmpeg(folder) -> str:
    """An `ffmpeg` that lists two dshow devices, so no test asks the real machine."""
    lines = ['[dshow @ 0] "Fake Mic" (audio)', '[dshow @ 0] "Fake Cam" (video)']
    if os.name == "nt":
        path = folder / "ffmpeg.cmd"
        path.write_text("@echo off\n" + "".join(f"echo {line} 1>&2\n" for line in lines), encoding="utf-8")
    else:
        path = folder / "ffmpeg.sh"
        path.write_text("#!/bin/sh\n" + "".join(f"echo '{line}' >&2\n" for line in lines), encoding="utf-8")
        path.chmod(0o755)
    return path.as_posix()


@pytest.fixture
def memory(home, served, tmp_path):
    """A running daemon (in this process) with the fake decider, no free-space floor, and a fake `ffmpeg` that lists
    devices, so that no test asks the real machine."""
    write_config(home, f'[memory]\nloop_s = 0.1\nfree_gb = 0\nffmpeg = "{fake_ffmpeg(tmp_path)}"\n' + DECIDER)
    return served()


@pytest.fixture
def detached(home):
    """For a test that lets an adapter or the CLI start the daemon in the background: it is stopped at the end."""
    yield home
    mms(home, "stop", check=False)


@pytest.fixture(scope="session")
def red_clip(tmp_path_factory, ffmpeg) -> Path:
    """Six seconds of red, 160x90 at 5 frames a second: the fake decider's `red` hits every window of it."""
    path = tmp_path_factory.mktemp("clips") / "red.mkv"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=160x90:r=5:d=6", "-c:v",
                    "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> dict:
    """The files of `tests/corpus.py`, made once per worker in its own folder and only read: a test that watches or
    changes files copies them into its own `tmp_path`."""
    return build_corpus(tmp_path_factory.mktemp("corpus"))


PEAK = """
import sys
from pathlib import Path

import av, numpy, PIL.Image
from mms import decider_d1, reader
from mms.store import Scope, Store

def peak():
    if sys.platform != "win32":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters(cb=ctypes.sizeof(Counters))
    psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize

store = Store(Path(sys.argv[1]))
before = peak()
"""


def peak_memory(code: str, home: Path) -> int:
    """Bytes of memory `code` grows the peak by, run in its own process after the imports and a `store` over `home` (and
    `reader`, `Scope`, `Path`) are in: the peak working set on Windows, `ru_maxrss` elsewhere. The code asserts what it
    checks; a failure raises with its output."""
    done = subprocess.run([sys.executable, "-c", PEAK + textwrap.dedent(code) + "\nprint(peak() - before)", str(home)],
                          env=env(home), capture_output=True, text=True, encoding="utf-8", timeout=300)
    if done.returncode != 0:
        raise AssertionError(f"the code exited {done.returncode}\n{done.stdout}\n{done.stderr}")
    return int(done.stdout.splitlines()[-1])


def wait_for(check, timeout: float = 20, every: float = 0.1, what: str = "the condition", detail=None):
    """Poll `check` until it returns something truthy, and return that. On a timeout the message adds `detail()`,
    such as a daemon's log."""
    end = time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if time.monotonic() > end:
            raise AssertionError(f"timed out after {timeout:g} s waiting for {what}"
                                 + (f"\n{detail()}" if detail else ""))
        time.sleep(every)
