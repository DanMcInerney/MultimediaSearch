"""The worker wire and the two transports (TESTBED.md, "No GPU"; DESIGN.md §13.3): `worker/wire.py`'s frames, a real
`Worker` process running `tests/fake_family.py` through `worker/serve.py`, and `Served` against a stdlib HTTP server.
No model, no GPU: the fake family echoes what reached it, so a test can check it was lossless."""

from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import statistics
import struct
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT, write_config, wait_for
from corpus import T0, scan
from fake_decider import ABOUT as FAKE_ABOUT
from PIL import Image

from mms import decider, transport
from mms.errors import ApiError
from mms.gpu import Gpu
from mms.transport import Served, Worker
from mms.worker import serve, wire

FAMILY = ROOT / "tests" / "fake_family.py"
WORKER = ROOT / "src" / "mms" / "worker"
MODEL_ID = "h2oai/h2o-lightning-4b"


def alive(pid: int) -> bool:
    if os.name == "nt":
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
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone(pid: int, timeout: float = 10) -> bool:
    wait_for(lambda: not alive(pid), timeout=timeout, what=f"process {pid} to end")
    return True


def sha(data) -> str:
    return hashlib.sha256(data).hexdigest()


def noise(width: int, height: int, seed: int = 1) -> Image.Image:
    data = np.random.default_rng(seed).integers(0, 256, width * height * 3, dtype=np.uint8).tobytes()
    return Image.frombytes("RGB", (width, height), data)


def ask(text: str = "nothing") -> dict:
    return {"q": {"type": "noul", "instructions": text}}


@pytest.fixture
def spawned(monkeypatch):
    """The processes `Worker` starts, and an end for any a test leaves running."""
    procs = []
    real = transport.children.popen

    def spy(*args, **kwargs):
        procs.append(real(*args, **kwargs))
        return procs[-1]

    monkeypatch.setattr(transport.children, "popen", spy)
    yield procs
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.fixture
def make(tmp_path, spawned):
    """A `Worker` over `fake_family` and a folder whose fake.json holds `settings`; unloaded at the end of the test."""
    made = []

    def build(settings: dict | None = None, *, load_s: float = 60, decide_s: float = 60, cls=Worker, options=None,
              python=sys.executable) -> Worker:
        folder = tmp_path / f"model{len(made)}"
        folder.mkdir()
        (folder / "fake.json").write_text(json.dumps(settings or {}), encoding="utf-8")
        made.append(cls(python, FAMILY, name="fam", folder=folder, options=options, load_s=load_s, decide_s=decide_s))
        return made[-1]

    yield build
    for worker in made:
        worker.unload()


# -- the frames -------------------------------------------------------------------------------------------------------

def reader(data: bytes) -> io.BufferedReader:
    return io.BufferedReader(io.BytesIO(data))


def frame(header: dict, *blobs: bytes) -> bytes:
    out = io.BytesIO()
    wire.write_frame(out, header, blobs)
    return out.getvalue()


def test_a_frame_is_the_magic_a_length_a_json_header_and_the_blobs():
    data = frame({"op": "decide", "state": "é"}, b"abc", b"", b"\x00\xff")
    text = json.dumps({"op": "decide", "state": "é", "blobs": [3, 0, 2]}, separators=(",", ":")).encode("utf-8")
    assert data == b"MSW1" + struct.pack("<I", len(text)) + text + b"abc" + b"\x00\xff"
    stream = reader(data + frame({"op": "quit"}))
    assert wire.read_frame(stream) == ({"op": "decide", "state": "é", "blobs": [3, 0, 2]}, [b"abc", b"", b"\x00\xff"])
    assert wire.read_frame(stream) == ({"op": "quit"}, [])
    assert wire.read_frame(stream) is None, "a stream that ends between frames ends cleanly"


def test_bytes_that_are_not_a_frame_or_are_over_the_limits_are_named(monkeypatch):
    with pytest.raises(wire.WireError, match="not a frame") as raised:
        wire.read_frame(reader(b"Loading weights... 0%\r\n"))
    assert raised.value.found == b"Loading weights..."[:16] and len(raised.value.found) == 16
    with pytest.raises(wire.WireError, match="not a frame"):
        wire.read_frame(reader(b"Hi"))  # a stray write shorter than a prefix is named too, not waited for
    with pytest.raises(wire.WireError, match=r"a header of 16777217 bytes is over the limit of 16777216"):
        wire.read_frame(reader(wire.MAGIC + struct.pack("<I", wire.MAX_HEADER + 1)))
    with pytest.raises(wire.WireError, match="short inside a frame"):
        wire.read_frame(reader(frame({"op": "x"}, b"abcd")[:-1]))
    with pytest.raises(wire.WireError, match="short inside a frame"):
        wire.read_frame(reader(wire.MAGIC + struct.pack("<I", 100) + b"{}"))
    with pytest.raises(wire.WireError, match="not JSON"):
        wire.read_frame(reader(wire.MAGIC + struct.pack("<I", 3) + b"{x}"))
    for bad in (b"[]", b'{"blobs": 3}', b'{"blobs": [-1]}', b'{"blobs": [true]}'):
        with pytest.raises(wire.WireError, match="list of blob sizes"):
            wire.read_frame(reader(wire.MAGIC + struct.pack("<I", len(bad)) + bad))

    monkeypatch.setattr(wire, "MAX_HEADER", 40)
    with pytest.raises(wire.WireError, match="over the limit of 40"):
        wire.write_frame(io.BytesIO(), {"op": "decide", "state": "x" * 100})
    monkeypatch.setattr(wire, "MAX_BLOBS", 10)
    written = io.BytesIO()
    with pytest.raises(wire.WireError, match="blobs of 12 bytes are over the limit of 10"):
        wire.write_frame(written, {"op": "decide"}, [b"x" * 6, b"y" * 6])
    assert written.getvalue() == b"", "a frame over the limits is refused before any of it is written"
    header = b'{"blobs":[6,6]}'
    with pytest.raises(wire.WireError, match="blobs of 12 bytes are over the limit of 10"):
        wire.read_frame(reader(wire.MAGIC + struct.pack("<I", len(header)) + header))


def test_media_descriptors_rebuild_what_was_sent():
    audio = np.array([0.5, -0.25, np.nan], dtype="<f4")
    media = [Image.new("RGB", (3, 2), (1, 2, 3)), {"type": "audio", "audio": audio, "sampling_rate": 16000},
             {"type": "file", "media": None, "name": "n.bin", "data": b"\x00\x01"}]
    descriptors, blobs = wire.encode_media(media)
    assert descriptors == [{"kind": "image", "mode": "RGB", "size": [3, 2], "blob": 0},
                           {"kind": "audio", "rate": 16000, "samples": 3, "blob": 1},
                           {"kind": "file", "media": None, "name": "n.bin", "blob": 2}]
    picture, sound, file = wire.decode_media(descriptors, blobs)
    assert picture.mode == "RGB" and picture.size == (3, 2) and picture.getpixel((2, 1)) == (1, 2, 3)
    assert sound["type"] == "audio" and sound["sampling_rate"] == 16000 and sound["audio"].dtype == np.dtype("<f4")
    assert sound["audio"].tobytes() == audio.tobytes() and sound["audio"].flags.writeable
    assert file == {"type": "file", "media": None, "name": "n.bin", "data": b"\x00\x01"}
    for odd in (Image.new("RGBA", (2, 2)), "text", {"type": "video"}):
        with pytest.raises(ValueError):
            wire.encode_media([odd])
    with pytest.raises(ValueError, match="is not 5 bytes"):
        wire.decode_media([{"kind": "image", "mode": "RGB", "size": [2, 2], "blob": 0}], [b"12345"])
    with pytest.raises(ValueError, match="is not 3 bytes"):
        wire.decode_media([{"kind": "audio", "rate": 16000, "samples": 2, "blob": 0}], [b"123"])


# -- the worker process -----------------------------------------------------------------------------------------------

def test_a_worker_hands_back_every_byte_it_was_given(make):
    worker = make(options={"warm": True})
    facts = worker.load()
    assert facts["options"] == {"warm": True} and facts["isolated"] == 1, "its options arrive, and it runs under -I"
    assert facts["stdout_is_stderr"] and worker.facts == facts and facts["pid"] != os.getpid()
    pid = worker.pid  # on Windows a venv's python.exe starts the real interpreter, so this is not facts["pid"]

    small, big = noise(640, 360, 1), noise(1280, 1280, 2)  # 1,638,400 pixels, d1's cap
    bits = np.random.default_rng(3).integers(0, 2 ** 32, 100_000, dtype=np.uint32).view("<f4")
    special = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-45, -1e-45, 3.4028235e38], dtype="<f4")
    audio = np.concatenate([bits, special])  # random bit patterns, so signalling NaNs and denormals too
    data = np.random.default_rng(4).bytes(5_000_000)

    def sound(samples):
        return {"type": "audio", "audio": samples, "sampling_rate": 16000}

    def file(blob):
        return {"type": "file", "media": "application/pdf", "name": "a.pdf", "data": blob}

    picture = lambda image: {"kind": "image", "size": list(image.size), "mode": "RGB", "sha256": sha(image.tobytes())}  # noqa: E731
    assert worker.decide(None, ask(), [small])["__echo__"] == [picture(small)]
    assert worker.decide(None, ask(), [big])["__echo__"] == [picture(big)]
    assert worker.decide("state", ask(), [sound(audio)])["__echo__"] == [
        {"kind": "audio", "dtype": "float32", "rate": 16000, "count": audio.size, "sha256": sha(audio.tobytes())}]
    assert worker.decide(None, ask(), [file(data)])["__echo__"] == [
        {"kind": "file", "name": "a.pdf", "media": "application/pdf", "length": len(data), "sha256": sha(data)}]
    both = worker.decide(None, ask(), [file(b""), small, sound(special)])["__echo__"]  # all at once, and an empty file
    assert [e["kind"] for e in both] == ["file", "image", "audio"]
    assert both[0]["sha256"] == sha(b"") and both[1] == picture(small) and both[2]["sha256"] == sha(special.tobytes())
    assert worker.decide(None, ask(), [sound(special.astype(">f4"))])["__echo__"][0]["sha256"] == sha(special.tobytes()), \
        "sound is sent little-endian float32 whatever its dtype"
    assert worker.pid == pid, "one process served all of it"


def test_the_workers_answers_follow_the_questions_and_its_state(make):
    worker = make()
    worker.load()
    questions = {"kettle": {"type": "noul", "instructions": "contains:kettle"},
                 "tl": {"type": "noul", "instructions": "corner:red"},
                 "mean": {"type": "noul", "instructions": "is it red?"}}
    red, blue = Image.new("RGB", (64, 36), (255, 0, 0)), Image.new("RGB", (64, 36), (0, 0, 255))
    answers = worker.decide("the kettle is on", questions, [red])
    assert {n: a["noul"] for n, a in answers.items() if n != "__echo__"} == {"kettle": 0.95, "tl": 0.95, "mean": 0.95}
    answers = worker.decide("", questions, [blue])
    assert {n: a["noul"] for n, a in answers.items() if n != "__echo__"} == {"kettle": 0.05, "tl": 0.05, "mean": 0.05}
    assert worker.stats() == {"max_memory_allocated": None, "memory_allocated": None}, "no torch, no GPU memory"


def test_a_worker_prints_to_stderr_whatever_a_library_writes_to_stdout(make, capsys):
    worker = make({"print_on_load": True})
    worker.load()
    assert worker.decide(None, ask("contains:x"), [])["q"]["noul"] == 0.05
    err = capsys.readouterr().err
    assert "fam worker: a library says hello on stdout" in err, "copied to the daemon's own stderr"
    assert "fam worker: MSW1 not a frame, written to fd 1 from C" in err and "and again, unflushed" in err


def test_a_valueerror_in_the_model_is_a_valueerror_here_and_the_worker_serves_on(make):
    worker = make()
    worker.load()
    pid = worker.pid
    with pytest.raises(ValueError, match="q: this question does not fit") as raised:
        worker.decide(None, ask("raise:value"), [])
    assert not isinstance(raised.value, ApiError)
    with pytest.raises(ApiError, match=r"^fam: RuntimeError: q: the model failed$") as failed:
        worker.decide(None, ask("raise:runtime"), [])
    assert failed.value.code == "decider"
    assert worker.pid == pid and worker.decide(None, ask(), [])["q"]["noul"] == 0.05, "it served on after both"
    with pytest.raises(ValueError, match="sent as RGB, not RGBA"):
        worker.decide(None, ask(), [Image.new("RGBA", (2, 2))])
    assert worker.pid == pid, "a medium that cannot be sent never reaches it"


def test_a_decision_too_big_to_send_is_a_valueerror_and_leaves_the_worker_alone(make, monkeypatch):
    worker = make()
    worker.load()
    pid = worker.pid
    monkeypatch.setattr(wire, "MAX_BLOBS", 1000)
    with pytest.raises(ValueError, match="over the limit of 1000"):
        worker.decide(None, ask(), [Image.new("RGB", (100, 100))])
    monkeypatch.undo()
    assert worker.pid == pid and worker.decide(None, ask(), [])["q"]["noul"] == 0.05


def test_a_worker_that_crashes_is_an_error_with_its_exit_code_and_only_a_load_starts_another(make, capsys, spawned):
    worker = make({"crash_on": "boom"})
    real = worker.load()["pid"]
    first = worker.pid
    with pytest.raises(ApiError) as raised:
        worker.decide(None, ask("boom"), [])
    assert raised.value.code == "decider"
    assert str(raised.value) == "fam's worker exited with code 3 while it decided: crashing on boom"
    assert worker.pid is None and gone(first) and gone(real)
    assert "fam worker: crashing on boom" in capsys.readouterr().err
    with pytest.raises(ApiError, match=r"^fam's worker is not loaded$") as raised:
        worker.decide(None, ask(), [])
    assert raised.value.code == "decider" and worker.pid is None and len(spawned) == 1, \
        "a decide never starts a worker: the backend's load does, and checks what it reports"
    worker.load()
    assert worker.decide(None, ask(), [])["q"]["noul"] == 0.05 and worker.pid not in (None, first)


def test_a_worker_that_never_loaded_is_not_loaded_and_starts_nothing(make, spawned):
    worker = make()
    with pytest.raises(ApiError, match=r"^fam's worker is not loaded$"):
        worker.decide(None, ask(), [])
    assert spawned == [] and worker.pid is None


def test_a_load_that_raises_valueerror_is_a_failed_load_not_an_unfit_question(make):
    worker = make()
    (worker.folder / "fake.json").write_text("{not json", encoding="utf-8")  # json.loads raises a ValueError
    with pytest.raises(ApiError, match=r"^fam: JSONDecodeError: ") as raised:
        worker.load()
    assert raised.value.code == "decider" and not isinstance(raised.value, ValueError)
    assert worker.pid is None, "a worker that could not load holds nothing worth keeping"


def test_a_worker_that_exits_while_loading_says_so_and_a_missing_python_says_it_could_not_start(make, tmp_path):
    script = tmp_path / "exits.py"
    script.write_text("import sys\nprint('no GPU here', file=sys.stderr)\nsys.exit(5)\n", encoding="utf-8")

    class Exits(Worker):
        serve = script

    with pytest.raises(ApiError) as raised:
        make(cls=Exits).load()
    assert str(raised.value) == "fam's worker exited with code 5 while it loaded: no GPU here"
    with pytest.raises(ApiError, match=r"^fam's worker could not start: ") as raised:
        make(python=tmp_path / "no-such-python.exe").load()
    assert raised.value.code == "decider"
    lost = Worker(sys.executable, tmp_path / "no_family.py", name="fam", folder=tmp_path)
    with pytest.raises(ApiError, match=r"^fam: FileNotFoundError: "):
        lost.load()


def test_stray_bytes_and_an_oversized_header_from_a_worker_are_named(make, tmp_path, spawned):
    for name, code, expected in (
            ("junk", "import os, time\nos.write(1, b'Loading weights... done\\n')\ntime.sleep(60)\n",
             r"^fam's worker wrote something that is not a frame: b'Loading weights\.'$"),
            ("huge", "import os, struct, time\nos.write(1, b'MSW1' + struct.pack('<I', 17 << 20))\ntime.sleep(60)\n",
             r"^fam's worker broke the frame format: a header of 17825792 bytes is over the limit of 16777216$"),
            ("cut", "import os, struct\nos.write(1, b'MSW1' + struct.pack('<I', 100) + b'{}')\n",
             r"^fam's worker broke the frame format: the stream ended \d+ bytes short inside a frame$")):
        script = tmp_path / f"{name}.py"
        script.write_text(code, encoding="utf-8")
        before = len(spawned)

        class Odd(Worker):
            serve = script

        started = time.monotonic()
        with pytest.raises(ApiError, match=expected):
            make(cls=Odd).load()
        assert time.monotonic() - started < 10, "named at once, not after a deadline"
        assert spawned[before].poll() is not None, "the worker is stopped"


def test_the_worker_exits_on_a_broken_frame_and_on_the_end_of_its_input():
    serve = [sys.executable, "-I", "-X", "utf8", str(WORKER / "serve.py"), "--family-file", str(FAMILY)]

    def run(data: bytes):
        done = subprocess.run(serve, input=data, capture_output=True, timeout=60)
        return done.returncode, done.stdout, done.stderr.decode("utf-8", "replace")

    code, out, err = run(wire.MAGIC + struct.pack("<I", wire.MAX_HEADER + 1))
    assert code == 2 and out == b"" and "over the limit" in err
    code, out, err = run(b"hello\n")
    assert code == 2 and out == b"" and "not a frame" in err
    assert run(b"")[:2] == (0, b""), "the parent closed its end"
    code, out, err = run(frame({"op": "quit"}))
    assert code == 0 and out == b""

    # decide before load, an op it does not know, then load, stats and quit: an error reply each time, and it serves on
    folder = ROOT / "tests"  # fake.json is not here: load fails
    code, out, err = run(frame({"op": "decide", "state": None, "questions": {}}) + frame({"op": "nope"})
                         + frame({"op": "load", "folder": str(folder)}) + frame({"op": "stats"})
                         + frame({"op": "quit"}))
    replies = []
    stream = reader(out)
    while (got := wire.read_frame(stream)) is not None:
        replies.append(got[0])
    assert code == 0 and [r["ok"] for r in replies] == [False, False, False, True]
    assert [(r.get("error"), r.get("type")) for r in replies[:3]] == [
        ("failed", "RuntimeError"), ("failed", "RuntimeError"), ("failed", "FileNotFoundError")]
    assert replies[0]["message"] == "decide came before load" and "nope" in replies[1]["message"]
    assert "Traceback" in replies[2]["trace"] and "Traceback" in err
    assert replies[3] == {"ok": True, "max_memory_allocated": None, "memory_allocated": None}


def test_a_worker_that_hangs_is_stopped_at_its_deadline(make, spawned):
    worker = make({"hang_on": "stuck"}, decide_s=1)
    real = worker.load()["pid"]
    pid = worker.pid
    started = time.monotonic()
    with pytest.raises(ApiError) as raised:
        worker.decide(None, ask("stuck"), [])
    assert str(raised.value) == "fam's worker gave no answer in 1 s, so it was stopped" and raised.value.code == "decider"
    assert 0.9 < time.monotonic() - started < 8
    assert worker.pid is None and gone(pid) and gone(real), "the process is gone, and the interpreter it started"
    with pytest.raises(ApiError, match="not loaded"):
        worker.decide(None, ask(), [])
    worker.load()
    assert worker.decide(None, ask(), [])["q"]["noul"] == 0.05 and worker.pid != pid, "and a load starts another"


def test_a_reply_taken_as_the_deadline_passes_is_kept_and_the_worker_is_not_killed(make, monkeypatch):
    class Late:
        """A timer whose deadline passes in the moment between taking the reply and cancelling the timer."""

        def __init__(self, interval, function):
            self.function, self.daemon = function, False

        def start(self):
            pass

        def cancel(self):
            self.function()

    monkeypatch.setattr(transport.threading, "Timer", Late)
    worker = make()
    real = worker.load()["pid"]
    pid = worker.pid
    assert worker.decide(None, ask(), [])["q"]["noul"] == 0.05, "the answer that came is not thrown away"
    assert worker.pid == pid and alive(pid) and alive(real), "and the timer did not kill the worker after it"
    assert worker.decide(None, ask(), [])["q"]["noul"] == 0.05 and worker.stats()["memory_allocated"] is None


def test_a_load_past_its_deadline_is_stopped(make, spawned):
    worker = make({"delay_s": 60}, load_s=1)
    started = time.monotonic()
    with pytest.raises(ApiError) as raised:
        worker.load()
    assert str(raised.value) == "fam's worker gave no answer in 1 s, so it was stopped"
    assert time.monotonic() - started < 8
    assert worker.pid is None and spawned[0].poll() is not None
    gone(spawned[0].pid)


def test_a_worker_gets_the_allowed_environment_and_no_secret(tmp_path, monkeypatch, spawned):
    family = tmp_path / "env_family.py"
    family.write_text("import os\n\ndef load(folder, options):\n    return None, {'env': dict(os.environ)}\n",
                      encoding="utf-8")
    secrets = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "OPENAI_API_KEY",
               "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "GITHUB_TOKEN", "SOME_SERVICE_KEY",
               "MMS_HOME", "PYTHONPATH", "VIRTUAL_ENV")
    for name in secrets:
        monkeypatch.setenv(name, "secret")
    kept = {"CUDA_VISIBLE_DEVICES": "0", "NVIDIA_VISIBLE_DEVICES": "all", "PYTORCH_CUDA_ALLOC_CONF": "x", "LC_ALL": "C",
            "HF_HOME": str(tmp_path / "hf"), "OMP_NUM_THREADS": "2"}
    for name, value in kept.items():
        monkeypatch.setenv(name, value)
    worker = Worker(sys.executable, family, name="env", folder=tmp_path)
    try:
        seen = {name.upper(): value for name, value in worker.load()["env"].items()}
    finally:
        worker.unload()
    assert not [name for name in secrets if name.upper() in seen], "no token, key, proxy or path of the daemon's own"
    assert all(seen.get(name) == value for name, value in kept.items())
    assert all(seen.get(name) == value for name, value in transport.CHILD_ENV.items()), "mms's own settings are there"
    assert "PATH" in seen and (os.name != "nt" or {"SYSTEMROOT", "TEMP"} <= set(seen)), "and what Python and CUDA need"
    allowed = {name.upper() for name in transport.child_env()}
    assert {name.upper() for name in os.environ} - allowed >= {name.upper() for name in secrets}


def test_unload_ends_the_worker_and_the_next_call_loads_a_new_one(make):
    worker = make()
    worker.unload()  # nothing running: nothing to do
    real = worker.load()["pid"]
    pid = worker.pid
    started = time.monotonic()
    worker.unload()
    assert time.monotonic() - started < 10 and worker.pid is None and gone(pid) and gone(real)
    with pytest.raises(ApiError, match="not loaded"):
        worker.decide(None, ask(), [])
    worker.load()
    assert worker.decide(None, ask(), [])["q"]["noul"] == 0.05 and worker.pid not in (None, pid)


def test_every_worker_file_imports_with_the_standard_library_alone():
    files = sorted(WORKER.glob("*.py"))
    assert {f.name for f in files} >= {"__init__.py", "serve.py", "wire.py"}
    probe = textwrap.dedent("""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("mms_worker_probe", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        heavy = [m for m in ("mms", "numpy", "PIL", "torch", "transformers") if m in sys.modules]
        assert not heavy, f"imported {heavy}"
    """)
    for path in files:
        done = subprocess.run([sys.executable, "-I", "-S", "-c", probe, str(path)], capture_output=True, text=True,
                              encoding="utf-8", timeout=60)
        assert done.returncode == 0, f"{path.name}: {done.stdout}{done.stderr}"
        text = path.read_text(encoding="utf-8")
        assert not any(line.lstrip().startswith(("import mms", "from mms")) for line in text.splitlines()), \
            f"{path.name} imports from mms, which a foreign env does not have"


def test_check_env_prints_the_installed_versions_as_json():
    done = subprocess.run([sys.executable, "-I", str(WORKER / "serve.py"), "--check-env"], capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    versions = json.loads(done.stdout)
    assert versions["numpy"] and versions["pillow"] and all(isinstance(v, str) for v in versions.values())


def test_installed_keeps_the_first_of_two_installs_of_a_package_as_an_import_would(tmp_path, monkeypatch):
    for place, version in (("first", "1.0"), ("second", "2.0")):
        info = tmp_path / place / "Dup_Pkg-1.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: Dup_Pkg\nVersion: {version}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path / "first"), str(tmp_path / "second"), *sys.path])
    assert serve.installed()["dup-pkg"] == "1.0"


def test_the_wire_is_fast_enough_that_it_does_not_show(make):
    worker = make({"fast": True})  # the family echoes sizes only, so what is timed is the wire and the decode
    worker.load()
    rows = {}
    for label, picture in (("no picture", None), ("640x360", noise(640, 360)), ("1.6 MP", noise(1280, 1280))):
        media = [picture] if picture else []
        worker.decide(None, ask(), media)  # the first call imports numpy and PIL in the worker
        times = []
        for _ in range(20):
            started = time.perf_counter()
            worker.decide(None, ask(), media)
            times.append((time.perf_counter() - started) * 1000)
        rows[label] = statistics.median(times)
    print("\nworker round trip, median of 20: " + ", ".join(f"{label} {ms:.1f} ms" for label, ms in rows.items()))
    for label, ms in rows.items():
        assert ms < 50, f"{label}: {ms:.1f} ms a round trip"


def test_a_daemon_killed_outright_takes_its_worker_with_it(home, tmp_path, daemon, red_clip):
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "fake.json").write_text("{}", encoding="utf-8")
    decider_file = tmp_path / "via_worker.py"
    decider_file.write_text(textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from mms.decider import About
        from mms.transport import Worker

        ABOUT = About(name="via-worker")
        home = None
        worker = Worker(sys.executable, Path({str(FAMILY)!r}), name="via", folder=Path({str(folder)!r}))

        def load():
            facts = worker.load()  # on Windows worker.pid is a venv's launcher, and facts["pid"] the real interpreter
            (home / "worker.pid").write_text(f"{{worker.pid}} {{facts['pid']}}")

        def decide(state, questions, media):
            answers = worker.decide(state, questions, media)
            answers.pop("__echo__")
            return answers

        def unload():
            worker.unload()
    """), encoding="utf-8")
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + f'[decider]\nmodel = "python:{decider_file.as_posix()}"\n')
    running = daemon()
    running.api.call("remember", path=str(red_clip), source="clips", t=T0)
    scan(running.api, T0 + 100, question="Is it red?")
    pids = [int(pid) for pid in wait_for(lambda: (home / "worker.pid").exists() and (home / "worker.pid").read_text(),
                                         what="the daemon's decider to start its worker", detail=running.text).split()]
    assert len(pids) == 2 and all(alive(pid) for pid in pids) and running.process.poll() is None
    running.kill()
    for pid in pids:
        assert gone(pid, timeout=15), f"process {pid} (the worker, and the interpreter it started) ended with the daemon"


# -- a model the owner serves -----------------------------------------------------------------------------------------

class Fake:
    """A stdlib HTTP server on 127.0.0.1: `health` is the list of (status, body) it answers `/health` with in turn (the last
    repeats), `replies` maps a path to (status, body, headers, pause_s) for a POST, and `seen` records every request."""

    def __init__(self):
        self.health = [(200, json.dumps({"status": "ok", "model": MODEL_ID}))]
        self.replies: dict = {}
        self.seen: list = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body, headers=None):
                data = body.encode("utf-8")
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.seen.append(("GET", self.path, None))
                if self.path != "/health":
                    return self.reply(404, "{}")
                status, body = fake.health.pop(0) if len(fake.health) > 1 else fake.health[0]
                self.reply(status, body)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fake.seen.append(("POST", self.path, json.loads(raw), self.headers.get("Content-Type")))
                status, body, headers, pause = fake.replies.get(self.path, (404, "{}", {}, 0))
                time.sleep(pause)
                self.reply(status, body, headers)

        class Quiet(ThreadingHTTPServer):
            def handle_error(self, request, client_address):  # a client that gave up is not news
                pass

        self.server = Quiet(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(transport, "POLL_S", 0.05)
    server = Fake()
    yield server
    server.close()


def served_model(fake: Fake, **kwargs) -> Served:
    return Served(fake.url, name="h2o", model_id=MODEL_ID, **kwargs)


def test_load_waits_while_the_server_warms_and_checks_the_model(fake):
    fake.health = [(503, '{"status": "warming"}')] * 2 + [(200, json.dumps({"status": "ok", "model": MODEL_ID}))]
    health = served_model(fake, load_s=30).load()
    assert health == {"status": "ok", "model": MODEL_ID}
    assert [path for _, path, *_ in fake.seen] == ["/health"] * 3, "503, 503, then 200"
    served_model(fake).unload()  # the server and its memory are the owner's: nothing to do

    fake.health = [(200, json.dumps({"status": "ok"}))]
    assert served_model(fake).load() == {"status": "ok"}, "a server that names no model is taken as it is"
    fake.health = [(200, json.dumps({"status": "ok", "model": "someone/else"}))]
    with pytest.raises(ApiError, match=rf"serves someone/else, not {MODEL_ID}") as raised:
        served_model(fake).load()
    assert raised.value.code == "decider"
    fake.health = [(200, "<html>")]
    with pytest.raises(ApiError, match="not JSON"):
        served_model(fake).load()
    fake.health = [(503, "{}")]
    with pytest.raises(ApiError, match=r"was not ready in 0\.3 s"):
        served_model(fake, load_s=0.3).load()
    fake.health = [(500, '{"detail": "engine died"}')]
    with pytest.raises(ApiError, match="answered 500: engine died"):
        served_model(fake).load()


def test_a_refused_connection_says_how_to_start_the_server(monkeypatch):
    monkeypatch.setattr(transport, "REFUSED_S", 0.3)
    monkeypatch.setattr(transport, "POLL_S", 0.05)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    started = time.monotonic()
    with pytest.raises(ApiError, match=rf"^h2o: nothing answers at http://127\.0\.0\.1:{port}$") as raised:
        Served(f"http://127.0.0.1:{port}", name="h2o", model_id=MODEL_ID, hint="start the server", load_s=60).load()
    assert raised.value.hint == "start the server" and raised.value.code == "decider"
    assert time.monotonic() - started < 10
    with pytest.raises(ApiError, match="failed: ConnectionRefusedError") as raised:
        Served(f"http://127.0.0.1:{port}", name="h2o").post("/v1/systemone", {})
    assert raised.value.hint
    with pytest.raises(ValueError, match="loopback address"):
        Served("http://192.168.1.5:8741", name="h2o")


def test_a_probe_gives_the_status_of_health_and_none_when_nothing_answers(fake):
    fake.health = [(200, "{}"), (503, "{}"), (500, "{}")]
    assert [served_model(fake).probe(1.0) for _ in range(3)] == [200, 503, 500], "any status is an answer"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert Served(f"http://127.0.0.1:{port}", name="h2o").probe(1.0) is None
    assert Served(raw_server(b"this is not http\r\n\r\n"), name="h2o").probe(1.0) is None, "not HTTP: no answer"


def raw_server(*replies: bytes | None) -> str:
    """A server on 127.0.0.1 that answers its connections in turn with these raw bytes (None: hangs up without a reply)."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)

    def serve() -> None:
        with listener:
            for reply in replies:
                connection, _ = listener.accept()
                with connection:
                    connection.recv(4096)
                    if reply is not None:
                        connection.sendall(reply)

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{listener.getsockname()[1]}"


def test_a_server_that_does_not_speak_http_fails_at_once_naming_its_url_and_a_hangup_is_retried(monkeypatch):
    monkeypatch.setattr(transport, "POLL_S", 0.05)
    url = raw_server(b"this is not http\r\n\r\n")
    started = time.monotonic()
    with pytest.raises(ApiError, match=rf"^h2o's server at {url} failed: BadStatusLine") as raised:
        Served(url, name="h2o", model_id=MODEL_ID, load_s=60).load()
    assert raised.value.code == "decider" and time.monotonic() - started < 5, "not retried until the deadline"

    body = json.dumps({"status": "ok", "model": MODEL_ID}).encode()
    ok = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body
    url = raw_server(None, None, ok)  # hung up on twice (it may be starting), then ready
    assert Served(url, name="h2o", model_id=MODEL_ID, load_s=60).load()["status"] == "ok"


def test_a_post_maps_the_statuses_of_the_server(fake):
    model = served_model(fake)
    body = {"state": "", "questions": {"q": {"type": "noul", "instructions": "x"}}, "images": ["data:image/png;base64,AA"]}
    fake.replies["/v1/systemone"] = (200, '{"answers": {"q": {"noul": 0.5}}}', {}, 0)
    assert model.post("/v1/systemone", body) == {"answers": {"q": {"noul": 0.5}}}
    assert fake.seen[-1] == ("POST", "/v1/systemone", body, "application/json")

    for status in (400, 422):
        fake.replies["/v1/systemone"] = (status, '{"detail": "at most 4 images"}', {}, 0)
        with pytest.raises(ValueError, match="^at most 4 images$") as raised:
            model.post("/v1/systemone", body)
        assert not isinstance(raised.value, ApiError)
    fake.replies["/v1/systemone"] = (422, '{"detail": [{"loc": ["body", "images"], "msg": "too many"}]}', {}, 0)
    with pytest.raises(ValueError, match="too many"):
        model.post("/v1/systemone", body)
    for status in (502, 503, 500):
        fake.replies["/v1/systemone"] = (status, "upstream is down", {}, 0)
        with pytest.raises(ApiError, match=rf"^h2o's server answered {status}: upstream is down$") as raised:
            model.post("/v1/systemone", body)
        assert raised.value.code == "decider"
    fake.replies["/v1/systemone"] = (200, "not json", {}, 0)
    with pytest.raises(ApiError, match="something that is not JSON"):
        model.post("/v1/systemone", body)


def test_a_post_that_gets_no_answer_in_time_is_an_error(fake):
    fake.replies["/v1/systemone"] = (200, "{}", {}, 1.5)
    started = time.monotonic()
    with pytest.raises(ApiError, match=r"^h2o's server gave no answer in 0\.3 s$"):
        served_model(fake, decide_s=0.3).post("/v1/systemone", {})
    assert time.monotonic() - started < 1.4


def test_a_redirect_is_refused_and_not_followed(fake):
    fake.replies["/v1/systemone"] = (302, "", {"Location": f"{fake.url}/elsewhere"}, 0)
    with pytest.raises(ApiError, match=rf"redirected to {fake.url}/elsewhere, which mms does not follow") as raised:
        served_model(fake).post("/v1/systemone", {"state": "secret"})
    assert raised.value.code == "decider"
    assert [path for _, path, *_ in fake.seen] == ["/v1/systemone"], "the target was never asked"
    fake.health = [(301, "")]
    with pytest.raises(ApiError, match="redirected to None"):
        served_model(fake).load()


def test_a_proxy_in_the_environment_is_not_used(fake, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # nothing listens on the discard port
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    fake.replies["/v1/systemone"] = (200, '{"ok": true}', {}, 0)
    model = served_model(fake)
    assert model.load()["status"] == "ok" and model.post("/v1/systemone", {}) == {"ok": True}


def test_localhost_and_the_ipv6_loopback_connect_where_config_says():
    local = Served("http://localhost:8741", name="h2o")
    assert (local.host, local.port) == ("127.0.0.1", 8741)
    assert Served("http://[::1]:8741", name="h2o").host == "::1"


# -- the whole daemon over a worker -------------------------------------------------------------------------------------

class FamilyBackend:
    """A backend for a folder with a fake.json, which runs a real `Worker` over `fake_family`: what a family's backend
    does in `decider_*.py`, with d1's facts."""

    abouts = (FAKE_ABOUT,)
    reads_folder = "the test backend reads a folder with a fake.json"
    made: list = []

    @staticmethod
    def claims(folder: Path):
        return FAKE_ABOUT if (Path(folder) / "fake.json").is_file() else None

    @staticmethod
    def check(about, questions: dict) -> None:
        pass  # it takes every question

    def __init__(self, folder, *, url=None, envs=None):
        self.about, self.folder, self.pids, self.real = FAKE_ABOUT, Path(folder), [], []
        self.worker = Worker(sys.executable, FAMILY, name="fam", folder=Path(folder), load_s=60, decide_s=60)
        FamilyBackend.made.append(self)

    def load(self) -> None:
        facts = self.worker.load()
        self.pids.append(self.worker.pid)
        self.real.append(facts["pid"])  # the interpreter itself: on Windows the worker's pid is a venv's launcher

    def decide(self, state, questions, media=None) -> dict:
        answers = self.worker.decide(state, questions, media or [])
        answers.pop("__echo__")
        return answers

    def unload(self) -> None:
        self.worker.unload()


def family_folder(tmp_path: Path, name: str, **settings) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "fake.json").write_text(json.dumps(settings), encoding="utf-8")
    return folder


def switch(home: Path, folder: Path, idle_s: float = 1) -> None:
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + f'[decider]\nmodel = "{folder.as_posix()}"\nidle_s = {idle_s}\n')


def test_the_daemon_asks_through_a_worker_unloads_it_when_idle_and_ends_it_on_a_switch(home, tmp_path, served, red_clip,
                                                                                       monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", (f"{__name__}:FamilyBackend",))
    monkeypatch.setattr(FamilyBackend, "made", [])
    first = family_folder(tmp_path, "first")
    second = family_folder(tmp_path, "second", pause_s=0.8)  # slow enough to switch in the middle of a scan

    def running(folder: Path) -> list:
        return [b for b in FamilyBackend.made if b.folder == folder and b.worker.pid is not None]

    switch(home, first)
    daemon = served()
    api = daemon.api
    api.call("remember", path=str(red_clip), source="clips", t=T0)
    result = scan(api, T0 + 100, question="Is it red?")
    assert result["sightings"] and all(s["p"] == 0.95 for s in result["sightings"]), result
    (backend,) = running(first)
    (pid,), (real,) = backend.pids, backend.real
    assert alive(pid) and alive(real) and api.call("status")["decider"]["state"] == "ready"
    wait_for(lambda: api.call("status")["decider"]["state"] == "unloaded", what="the model to unload when idle")
    # the status says unloaded once the decider is swapped out; the worker is ended just after, outside the lock
    wait_for(lambda: backend.worker.pid is None and gone(pid) and gone(real), what="the worker process to end after idle_s")

    switch(home, second, idle_s=60)
    wait_for(lambda: api.call("status")["decider"]["model"] == second.as_posix(), what="the switch to be read")
    job = api.call("jobs/start", script="ask", params={"question": "Is it red?", "since": T0 - 100, "until": T0 + 100,
                                                       "limit": 50, "still_s": 0}, wait_s=1)  # a window a second: it is still going when the model is switched
    (slow,) = wait_for(lambda: running(second), what="the scan to start the second model's worker",
                       detail=lambda: f"{job} {api.call('status')['decider']} {api.call('jobs/get', job=job['job'])}")
    old, old_real = slow.worker.pid, slow.real[-1]
    assert job["state"] == "running", "the scan is still going when the model is switched"
    switch(home, first, idle_s=60)  # between two windows the daemon ends the old worker and loads the new model
    assert gone(old, timeout=30) and gone(old_real)
    wait_for(lambda: slow.worker.pid is None, what="the old worker to be forgotten after it ended")
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=20)
    assert job["state"] == "done" and job["result"]["sightings"], job
    (current,) = running(first)
    last, last_real = current.worker.pid, current.real[-1]
    assert last not in (None, old, pid) and alive(last) and alive(last_real), "the scan went on in a new worker"

    daemon.stop()
    assert current.worker.pid is None and gone(last) and gone(last_real), "the daemon's stop ended the worker"


def gpu_over(tmp_path: Path, model: str, idle_s: float = 60) -> Gpu:
    return Gpu(SimpleNamespace(home=tmp_path, decider_model=model, decider_url=None, decider_envs=None,
                               decider_idle_s=idle_s))


def test_a_failed_decide_lets_go_of_the_decider_and_the_next_request_loads_it_again_once(tmp_path, monkeypatch, spawned):
    monkeypatch.setattr(decider, "BACKENDS", (f"{__name__}:FamilyBackend",))
    monkeypatch.setattr(FamilyBackend, "made", [])
    gpu = gpu_over(tmp_path, str(family_folder(tmp_path, "m", crash_on="boom")))
    try:
        assert gpu.decide(None, ask(), [])["q"]["noul"] == 0.05
        (first,) = FamilyBackend.made
        assert first.pids and len(first.pids) == 1
        with pytest.raises(ValueError, match="does not fit"):
            gpu.decide(None, ask("raise:value"), [])
        assert FamilyBackend.made == [first] and gpu.status()["state"] == "ready", "a question that does not fit drops nothing"
        with pytest.raises(ApiError, match="exited with code 3 while it decided") as raised:
            gpu.decide(None, ask("boom"), [])
        assert raised.value.code == "decider"
        assert gpu.status()["state"] == "unloaded" and first.worker.pid is None, "the decider was let go and ended"
        for _ in range(3):  # the windows after it
            assert gpu.decide(None, ask(), [])["q"]["noul"] == 0.05
        _, second = FamilyBackend.made
        assert [len(b.pids) for b in FamilyBackend.made] == [1, 1], "one load of a new decider for the failure, none per window"
        assert gpu.status()["state"] == "ready" and second.worker.pid is not None
    finally:
        gpu.stop()
    assert all(b.worker.pid is None for b in FamilyBackend.made)


class SlowUnload:
    """A decider whose `unload` takes 2 s, and whose `decide` fails with `decider` when it is asked to."""

    about = FAKE_ABOUT

    def __init__(self):
        self.unloading, self.unloaded = threading.Event(), threading.Event()

    def decide(self, state, questions, media=None):
        if "fail" in questions:
            raise ApiError("decider", "the model broke")
        return {name: {"type": "noul", "noul": 0.95} for name in questions}

    def unload(self):
        self.unloading.set()
        time.sleep(2)
        self.unloaded.set()


def test_status_and_apply_do_not_wait_for_a_slow_unload_when_idle_or_after_a_failure(tmp_path, monkeypatch):
    slow = SlowUnload()
    monkeypatch.setattr(decider, "load_decider", lambda *args, **kwargs: slow)
    gpu = gpu_over(tmp_path, "model", idle_s=0.2)
    try:
        assert gpu.decide(None, ask(), [])["q"]["noul"] == 0.95
        wait_for(slow.unloading.is_set, what="the idle model to be unloaded")
        started = time.monotonic()
        status = gpu.status()
        gpu.apply(SimpleNamespace(decider_model="model", decider_url=None, decider_envs=None, decider_idle_s=0.2))
        assert time.monotonic() - started < 0.5 and not slow.unloaded.is_set(), "it was asked during the unload"
        assert status["state"] == "unloaded"
        wait_for(slow.unloaded.is_set, what="the unload to end")

        slow.unloading.clear()
        slow.unloaded.clear()
        gpu.apply(SimpleNamespace(decider_model="model", decider_url=None, decider_envs=None, decider_idle_s=60))
        started = time.monotonic()
        with pytest.raises(ApiError, match="the model broke"):
            gpu.decide(None, {"fail": {"type": "noul", "instructions": "x"}}, [])
        assert time.monotonic() - started < 1, "the failed request is answered before its decider is ended"
        wait_for(slow.unloading.is_set, what="the failed decider to be unloaded")
        started = time.monotonic()
        assert gpu.status()["state"] == "unloaded" and time.monotonic() - started < 0.5 and not slow.unloaded.is_set()
    finally:
        gpu.stop()
    assert slow.unloaded.is_set()
