"""The worker process (DESIGN.md §13.3): it holds one model and answers the daemon's frames on stdin and stdout.

    python -I -X utf8 serve.py --family-file <worker/d1.py>     serve frames until `quit` or the end of stdin
    python -I serve.py --check-env                              print the installed packages as JSON, and exit

The family file defines `load(folder, options) -> (model, facts)` and `decide(model, state, questions, media) -> dict`.
Standard library only at the top, and nothing from `mms`, so a foreign env's Python runs it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path


def canonical(name: str) -> str:
    """A package's name as `installed` keys it: lower case, with runs of `-`, `_` and `.` as one dash."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed() -> dict:
    """Every installed distribution's version, by its canonical name, from the metadata alone."""
    from importlib import metadata

    versions = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            versions.setdefault(canonical(name), dist.version)  # the first on sys.path is the one imported
    return versions


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no Python file at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def take_streams():
    """The frames' own stdin and stdout: private copies of fds 0 and 1. Fd 1 then points at stderr and fd 0 at the null
    device, so whatever a library prints or reads, from Python or C, cannot touch the frames."""
    frames_in, frames_out = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")
    if os.name == "nt":
        import msvcrt

        msvcrt.setmode(frames_in.fileno(), os.O_BINARY)
        msvcrt.setmode(frames_out.fileno(), os.O_BINARY)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return frames_in, frames_out


def _memory() -> dict:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {"max_memory_allocated": None, "memory_allocated": None}
    return {"max_memory_allocated": int(torch.cuda.max_memory_allocated()),
            "memory_allocated": int(torch.cuda.memory_allocated())}


def serve(family_file: Path) -> int:
    """Answer frames until `quit` (exit 0), the end of stdin (0) or a broken frame (2)."""
    frames_in, frames_out = take_streams()
    wire = _import("mms_worker_wire", Path(__file__).with_name("wire.py"))
    family, model, loaded = None, None, False

    def send(reply: dict) -> None:
        try:
            wire.write_frame(frames_out, reply)
        except (TypeError, ValueError) as error:  # the reply cannot be sent: nothing of it was written
            wire.write_frame(frames_out, {"ok": False, "error": "failed", "type": type(error).__name__,
                                          "message": f"the reply cannot be sent as JSON: {error}", "trace": ""})

    while True:
        try:
            frame = wire.read_frame(frames_in)
        except wire.WireError as error:
            print(f"worker: {error}", file=sys.stderr)
            return 2
        if frame is None:
            return 0
        header, blobs = frame
        op = header.get("op")
        if op == "quit":
            return 0
        started = time.perf_counter()
        try:
            if op == "load":
                family = family or _import("mms_worker_family", family_file)
                model, facts = family.load(header["folder"], header.get("options") or {})
                loaded = True
                reply = {"ok": True, "facts": facts, "seconds": time.perf_counter() - started}
            elif op == "decide":
                if not loaded:
                    raise RuntimeError("decide came before load")
                media = wire.decode_media(header.get("media") or [], blobs)
                answers = family.decide(model, header.get("state"), header["questions"], media)
                reply = {"ok": True, "answers": answers, "seconds": time.perf_counter() - started}
            elif op == "stats":
                reply = {"ok": True, **_memory()}
            else:
                raise RuntimeError(f"no op {op!r}: load, decide, stats and quit")
        except Exception as error:
            unfit = op == "decide" and isinstance(error, (ValueError, TypeError))  # what is asked does not fit the model
            if not unfit:  # (a load that raises ValueError is a failed load)
                traceback.print_exc()
            reply = {"ok": False, "error": "unfit" if unfit else "failed", "type": type(error).__name__,
                     "message": str(error), "trace": traceback.format_exc()}
        send(reply)


def main(argv: list[str]) -> int:
    if argv == ["--check-env"]:
        print(json.dumps(installed(), sort_keys=True))
        return 0
    if len(argv) != 2 or argv[0] != "--family-file":
        print("usage: serve.py --family-file FILE | --check-env", file=sys.stderr)
        return 2
    return serve(Path(argv[1]))


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # a loaded model's teardown can be slow, and the OS frees its memory
