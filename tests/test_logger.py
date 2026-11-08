"""The program log client (DESIGN.md §8, TESTBED.md item 10): what it writes into the spool, against §8.2.

The daemon that ingests the spool is tested with it in test_spool.py and test_seams.py.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import wait_for
from PIL import Image

from mms import logger as logger_module
from mms.logger import Logger

RUN = re.compile(r"^\d{8}T\d{6}Z-\d+-[0-9a-f]{4}$")
NO_DROPS = {"queue_full": 0, "spool_full": 0, "floor": 0, "refused": 0}


def runs(home: Path, stream: str) -> list[Path]:
    folder = home / "spool" / stream
    return sorted(path for path in folder.glob("*") if path.is_dir()) if folder.exists() else []


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def batches(home: Path, stream: str) -> list[Path]:
    return [batch for run in runs(home, stream) for batch in sorted(run.glob("*.jsonl"))]


@pytest.fixture
def slow_batches(monkeypatch):
    """No batch is written until a flush, so the queue's state is exact."""
    monkeypatch.setattr(logger_module, "BATCH_S", 60.0)


def test_records_and_frames_are_written_as_the_spool_contract_says(home):
    logger = Logger(home, project="id:demo")
    before = time.time()
    logger.log("nervelet", {"event": "fell", "height_m": 0.12, "name": "café"}, t=12.5)
    logger.log("nervelet", {"event": "up"})
    data = {"n": 1}
    logger.log("nervelet", data)
    data["n"] = 2  # a later change is not seen
    picture = np.zeros((30, 40, 3), np.uint8)
    picture[..., 0] = 250  # red
    logger.log("nervelet", frame=picture, t=3)
    picture[:] = 0  # nor is a later change to the frame
    logger.log("nervelet", frame=np.full((20, 10), 128, np.uint8))
    assert logger.flush()
    after = time.time()

    [run] = runs(home, "nervelet")
    assert RUN.match(run.name)
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["v"] == 1 and info["stream"] == "nervelet" and info["run"] == run.name
    assert info["project"] == "id:demo" and isinstance(info["pid"], int) and before - 1 <= info["started"] <= after
    assert sorted(path.name for path in run.iterdir()) == [
        "00000001-003.jpg", "00000001-004.jpg", "00000001.jsonl", "run.json"]  # no .part left

    first, second, third, color, grey = lines(run / "00000001.jsonl")
    assert first["data"] == {"event": "fell", "height_m": 0.12, "name": "café"} and first["t"] == 12.5
    assert second["data"] == {"event": "up"} and second["t"] is None
    assert third["data"] == {"n": 1}
    assert first["wall"] <= second["wall"] <= third["wall"] and before <= first["wall"] <= after
    assert set(first) == {"wall", "t", "data"}
    assert color["frame"] == "00000001-003.jpg" and (color["w"], color["h"]) == (40, 30) and color["t"] == 3.0
    assert grey["frame"] == "00000001-004.jpg" and (grey["w"], grey["h"]) == (10, 20) and grey["t"] is None
    assert set(color) == {"wall", "t", "frame", "w", "h"}
    with Image.open(run / color["frame"]) as image:
        assert image.format == "JPEG" and image.size == (40, 30)
        red, green, blue = image.convert("RGB").getpixel((20, 15))
        assert red > 200 and green < 60 and blue < 60  # the copy, not the zeroed array
    with Image.open(run / grey["frame"]) as image:
        assert image.size == (10, 20)

    logger.log("nervelet", {"again": True})  # the next batch of the run takes the next number
    assert logger.flush()
    assert lines(run / "00000002.jsonl")[0]["data"] == {"again": True}
    assert logger.stats() == {"queued": 0, "written": 6, "dropped": NO_DROPS, "write_errors": 0, "last_error": None}
    logger.close()


def test_a_batch_is_written_by_the_thread_alone_after_its_delay_and_at_1000_lines(home, monkeypatch):
    logger = Logger(home, project="id:demo")
    logger.log("early", {"n": 0})
    wait_for(lambda: batches(home, "early"), what="the 0.2 s batch")
    monkeypatch.setattr(logger_module, "BATCH_S", 60.0)
    for number in range(logger_module.BATCH_LINES):  # the thread now waits on a long delay, until the 1000th line
        logger.log("full", {"n": number})
    wait_for(lambda: batches(home, "full"), what="the 1000-line batch")
    assert len(lines(batches(home, "full")[0])) == 1000
    logger.close()


def test_streams_have_their_own_runs_and_close_ends_each_with_its_drops(home):
    logger = Logger(home, project="id:demo")
    logger.log("one", {"a": 1})
    logger.log("two", {"b": 1})
    assert logger.close()
    for stream in ("one", "two"):
        [run] = runs(home, stream)
        end = json.loads((run / "end.json").read_text(encoding="utf-8"))
        assert end["dropped"] == 0 and end["ended"] >= json.loads((run / "run.json").read_text())["started"]
        assert len(lines(run / "00000001.jsonl")) == 1
    with pytest.raises(RuntimeError, match="closed"):
        logger.log("one", {"a": 2})


def test_ten_thousand_calls_take_under_half_a_second(home, capsys):
    logger = Logger(home, project="id:demo")
    record = {"event": "step", "pos": [1.0, 2.0, 3.0], "ok": True}
    start = time.perf_counter()
    for number in range(10_000):
        logger.log("busy", record, t=number * 0.01)
    took = time.perf_counter() - start
    with capsys.disabled():
        print(f"\n10,000 mms.log calls: {took * 1000:.0f} ms ({took / 10_000 * 1e6:.0f} us each)")
    assert took < 0.5
    assert logger.flush(timeout_s=20)
    assert sum(len(lines(path)) for path in batches(home, "busy")) == 10_000 and logger.stats()["written"] == 10_000
    logger.close()


def test_a_full_queue_drops_and_counts_without_blocking_or_raising(home, monkeypatch, slow_batches, capsys):
    record = {"x": "a" * 100}
    size = len(json.dumps(record, separators=(",", ":")))
    monkeypatch.setattr(logger_module, "QUEUE_MAX_BYTES", 10 * size + 5)
    logger = Logger(home, project="id:demo")
    start = time.perf_counter()
    for _ in range(25):
        logger.log("big", record)
    assert time.perf_counter() - start < 0.5
    stats = logger.stats()
    assert stats["queued"] == 10 and stats["dropped"]["queue_full"] == 15
    assert capsys.readouterr().err.count("queue_full") == 1  # the first of the kind is printed, later ones counted
    assert logger.flush()
    assert logger.stats()["written"] == 10
    logger.log("big", record)  # room again
    assert logger.stats()["dropped"]["queue_full"] == 15
    logger.close()
    [run] = runs(home, "big")
    assert json.loads((run / "end.json").read_text())["dropped"] == 15


def test_a_full_spool_stops_writing_a_stream_and_counts(home, monkeypatch, capsys):
    monkeypatch.setattr(logger_module, "SPOOL_MAX_BYTES", 1000)
    (home / "spool" / "other").mkdir(parents=True)
    (home / "spool" / "other" / "backlog.bin").write_bytes(b"x" * 2000)  # the daemon is not taking batches
    logger = Logger(home, project="id:demo")
    for number in range(5):
        logger.log("late", {"n": number})
    assert logger.flush()
    assert logger.stats()["written"] == 0 and logger.stats()["dropped"]["spool_full"] == 5
    assert runs(home, "late") == []
    assert "spool_full" in capsys.readouterr().err
    logger.close()


def test_a_spool_that_fills_while_logging_stops_the_next_batch(home, monkeypatch):
    monkeypatch.setattr(logger_module, "SPOOL_MAX_BYTES", 3000)
    logger = Logger(home, project="id:demo")
    for _ in range(20):
        logger.log("grow", {"x": "a" * 200})
        assert logger.flush()  # a batch each, 220 bytes or so
    stats = logger.stats()
    assert 0 < stats["written"] < 20 and stats["written"] + stats["dropped"]["spool_full"] == 20
    logger.close()


def test_a_refused_stream_is_dropped_before_and_after_the_daemon_refuses_it(home):
    logger = Logger(home, project="id:demo")
    logger.log("pick", {"n": 1})
    assert logger.flush() and logger.stats()["written"] == 1
    (home / "spool" / "pick" / "refused.json").write_text(json.dumps({"reason": "a folder source has that name"}))
    for number in range(3):
        logger.log("pick", {"n": number})
    logger.log("fine", {"n": 1})
    assert logger.flush()
    stats = logger.stats()
    assert stats["written"] == 2 and stats["dropped"]["refused"] == 3  # "fine" still goes through
    (home / "spool" / "named").mkdir(parents=True)
    (home / "spool" / "named" / "refused.json").write_text("{}")
    logger.log("named", {"n": 1})
    assert logger.flush() and logger.stats()["dropped"]["refused"] == 4 and runs(home, "named") == []
    logger.close()


def test_below_the_floor_the_client_drops_and_counts(home):
    (home / "mms.toml").write_text("[memory]\nfree_gb = 1000000000\n")  # more than any disk has
    logger = Logger(home, project="id:demo")
    logger.log("low", {"n": 1})
    logger.log("low", {"n": 2})
    assert logger.flush()
    assert logger.stats()["dropped"]["floor"] == 2 and runs(home, "low") == []
    logger.close()


def test_a_bad_call_raises_at_the_call_and_nothing_is_queued(home):
    logger = Logger(home, project="id:demo")
    grey = np.zeros((2, 2), np.uint8)
    bad = [
        dict(stream="has space", data={"a": 1}),
        dict(stream="", data={"a": 1}),
        dict(stream="x" * 65, data={"a": 1}),
        dict(stream="a/b", data={"a": 1}),
        dict(stream="s", data=["not", "an", "object"]),
        dict(stream="s", data="text"),
        dict(stream="s", data={"a": object()}),
        dict(stream="s", data={"a": float("nan")}),
        dict(stream="s", data={"a": "\ud800"}),  # a lone surrogate cannot be UTF-8
        dict(stream="s"),
        dict(stream="s", data={"a": 1}, frame=grey),
        dict(stream="s", data={"a": 1}, t="soon"),
        dict(stream="s", data={"a": 1}, t=float("inf")),
        dict(stream="s", frame=np.zeros((4, 4, 3), np.float32)),
        dict(stream="s", frame=np.zeros((4, 4, 4), np.uint8)),
        dict(stream="s", frame=np.zeros(4, np.uint8)),
        dict(stream="s", frame=np.zeros((0, 4), np.uint8)),
        dict(stream="s", frame=[[0, 0], [0, 0]]),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            logger.log(kwargs.pop("stream"), kwargs.pop("data", None), **kwargs)
    assert logger.stats() == {"queued": 0, "written": 0, "dropped": NO_DROPS, "write_errors": 0, "last_error": None}
    logger.close()


def test_a_write_error_that_is_not_a_full_disk_is_counted_apart_and_never_raised(tmp_path, capsys):
    not_a_folder = tmp_path / "file"
    not_a_folder.write_text("in the way")
    logger = Logger(not_a_folder, project="id:demo")
    logger.log("s", {"n": 1})
    logger.log("s", {"n": 2})
    assert logger.flush()
    stats = logger.stats()
    assert stats["written"] == 0 and stats["dropped"] == NO_DROPS, "not the spool being full"
    assert stats["write_errors"] == 2 and "could not write the spool" in stats["last_error"]
    assert "could not write the spool" in capsys.readouterr().err
    logger.close()


def test_a_full_disk_is_the_spool_being_full(home, monkeypatch, capsys):
    import errno

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    logger = Logger(home, project="id:demo")
    monkeypatch.setattr(logger, "_write_files", full)
    logger.log("s", {"n": 1})
    assert logger.flush()
    stats = logger.stats()
    assert stats["dropped"]["spool_full"] == 1 and stats["write_errors"] == 0 and stats["last_error"] is None
    assert "spool_full" in capsys.readouterr().err
    logger.close()
