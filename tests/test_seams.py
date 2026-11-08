"""A program's `mms.log` and the daemon together (TESTBED.md item 10): records and numpy frames from a real
`Logger` in its own process, through the spool and the daemon, with the daemon running, stopped and started again."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from conftest import env, wait_for, write_config
from PIL import Image

from mms.config import parse_when

PROGRAM = """
import json, sys, time
import numpy as np
import mms

start = time.time()
colours = {"a": (230, 20, 20), "b": (20, 230, 20), "c": (20, 20, 230)}
for line in sys.stdin:  # one phase a line
    phase = line.strip()
    for n in range(5):
        sim = time.time() - start
        mms.log("prog", {"phase": phase, "n": n, "sim": sim}, t=sim)
    for n in range(3):
        mms.log("cam", frame=np.full((48, 64, 3), colours[phase], np.uint8), t=time.time() - start)
        print("frame", phase, repr(time.time()), flush=True)  # inside this frame's 50 ms
        time.sleep(0.05)
    assert mms.flush(10)
    print("done", json.dumps(mms.stats()["dropped"]), flush=True)
"""
CHANNEL = {"a": 0, "b": 1, "c": 2}  # which of R, G, B is high in a phase's frames


def test_a_program_logs_through_a_daemon_that_stops_and_starts_and_every_record_and_frame_arrives(home, tmp_path, served):
    program = tmp_path / "program.py"
    program.write_text(PROGRAM, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    process = subprocess.Popen([sys.executable, "-u", str(program)], env=env(home), cwd=work, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, text=True, encoding="utf-8")
    frames: list[tuple[str, float]] = []
    dropped: list[dict] = []

    def phase(name: str) -> None:
        process.stdin.write(name + "\n")
        process.stdin.flush()
        while (words := process.stdout.readline().split(" ", 1)) and words[0] == "frame":
            which, wall = words[1].split()
            frames.append((which, float(wall)))
        assert words[0] == "done", "the program ended early"
        dropped.append(json.loads(words[1]))

    begun = time.time()
    try:
        running = served()
        phase("a")
        wait_for(lambda: len(records(running)) == 5, what="phase a's records, ingested while the daemon ran")
        running.stop()
        assert not (home / "daemon.json").exists()
        phase("b")  # nobody ingests: it waits in the spool
        assert list((home / "spool" / "prog").glob("*/*.jsonl")), "the spool holds what the daemon has not read"
        again = served()
        phase("c")
        process.stdin.close()
        assert process.wait(30) == 0
    finally:
        if process.poll() is None:
            process.kill()
    ended = time.time()

    found = wait_for(lambda: len(rows := records(again)) == 15 and rows, timeout=60, what="all 15 records")
    assert all(drops == dict.fromkeys(drops, 0) for drops in dropped), dropped
    texts = [json.loads(row["text"]) for row in found]
    assert sorted((text["phase"], text["n"]) for text in texts) == [(p, n) for p in "abc" for n in range(5)], "each once"
    for row, text in zip(found, texts):
        assert begun - 1 <= parse_when(row["t0"]) <= ended + 1, "the row's time is the call's wall time"
        assert row["run"] and abs(row["tp"] - text["sim"]) < 1.0, "find gives the run and the program's own time"
    assert len({row["run"] for row in found}) == 1

    def settled() -> bool:
        items = again.api.call("find", sources=["cam"], types=["item"])["results"]
        return bool(items) and not list((home / "spool").glob("*/*/run.json"))

    wait_for(settled, timeout=60, what="the frames to be encoded and the spool to be emptied",
             detail=lambda: str(list((home / "spool").rglob("*"))))
    assert len(frames) == 9
    for name, wall in frames:  # the picture the program sent at that time
        shown = again.api.call("look", source="cam", t=wall + 0.015)  # frames last 50 ms, and their times are rounded to the ms
        assert shown["kind"] == "image" and shown["source"] == "cam"
        with Image.open(Path(shown["image"]["path"])) as picture:
            mean = np.asarray(picture.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
        assert mean[CHANNEL[name]] > 150 and sorted(mean)[1] < 80, (name, wall, mean)
    status = again.api.call("status")
    cam = next(source for source in status["sources"] if source["name"] == "cam")
    assert cam["kind"] == "program" and cam["items"] >= 1, "the stream's first log added its table"
    assert parse_when(again.api.call("find", sources=["cam"], types=["item"])["results"][0]["t0"]) >= begun - 1


def records(running) -> list[dict]:
    return running.api.call("find", sources=["prog"], types=["record"], limit=200)["results"]
