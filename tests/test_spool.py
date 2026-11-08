"""Program logs, the daemon's half (TESTBED.md item 10): spool files written exactly as DESIGN.md §8.2 gives them become
records, clock rows and video segments whose frame times match the `wall` given to the millisecond."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import wait_for, write_config
from PIL import Image

from mms import config as configuration
from mms import reader
from mms.spool import Spool
from mms.store import Store

RUN = "20261031T120000Z-1234-abcd"
COLOURS = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20), (20, 220, 220), (220, 20, 220)]


def run_folder(home: Path, stream: str, run: str = RUN, project: str | None = None, started: float | None = None) -> Path:
    folder = home / "spool" / stream / run
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"v": 1, "stream": stream, "run": run, "pid": 1234, "project": project,
                                                 "started": started or time.time()}), encoding="utf-8")
    return folder


def write_batch(folder: Path, seq: int, records=(), frames=()) -> None:
    """A batch as the client writes it: the frames first, then the lines as .part, renamed. `records` are
    (wall, t, data) and `frames` are (wall, t, colour index, (w, h))."""
    lines = [{"wall": wall, "t": t, "data": data} for wall, t, data in records]
    for i, (wall, t, colour, (w, h)) in enumerate(frames):
        name = f"{seq:08d}-{i:03d}.jpg"
        Image.new("RGB", (w, h), COLOURS[colour % len(COLOURS)]).save(folder / name, quality=90)
        lines.append({"wall": wall, "t": t, "frame": name, "w": w, "h": h})
    part = folder / f"{seq:08d}.jsonl.part"
    part.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    os.replace(part, folder / f"{seq:08d}.jsonl")


def end(folder: Path) -> None:
    (folder / "end.json").write_text(json.dumps({"ended": time.time(), "dropped": 0}), encoding="utf-8")


def config(home: Path, extra: str = "") -> None:
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + extra)


def dominant(path: str) -> int:
    with Image.open(path) as picture:
        mean = np.asarray(picture.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    return min(range(len(COLOURS)), key=lambda i: float(np.abs(mean - np.array(COLOURS[i])).sum()))


def test_records_become_rows_a_clock_and_the_new_streams_table_with_the_runs_project(home, served):
    config(home)
    mine = "git:0123456789abcdef"
    folder = run_folder(home, "nervelet", project=mine)
    base = time.time() - 100
    write_batch(folder, 1, records=[(base + 0.0, 5.0, {"event": "step", "n": 1}), (base + 0.5, 5.5, {"event": "step", "n": 2})])
    write_batch(folder, 2, records=[(base + 10.0, 15.0, {"event": "fell", "height_m": 0.12})])
    write_batch(folder, 3, records=[(base + 20.0, None, {"event": "no sim time"})])
    api = served().api
    store = Store(home)
    wait_for(lambda: len(store.records(sources=["nervelet"])) == 4, what="the records", timeout=30)
    rows = store.records(sources=["nervelet"])
    assert [r.t for r in rows] == pytest.approx([base, base + 0.5, base + 10, base + 20]), "a record's time is the wall time"
    assert {r.project for r in rows} == {mine} and json.loads(rows[2].data) == {"event": "fell", "height_m": 0.12}
    assert [tuple(r) for r in store.db().execute("SELECT run, wall, t FROM clock ORDER BY wall")] == [
        (RUN, pytest.approx(base), 5.0), (RUN, pytest.approx(base + 10), 15.0)], \
        "one clock row a batch: its first line's wall and t (none for a batch with no t)"
    table = configuration.load(home).sources["nervelet"]
    assert table.kind == "program" and table.keep == configuration.Keep(5.0, "thin", None), "added with a cap"
    assert "kind" in (home / "mms.toml").read_text(encoding="utf-8")
    run, tp = store.program_time("nervelet", base + 5)
    assert run == RUN and tp == pytest.approx(10.0), "between the clock rows"
    assert store.program_time("nervelet", base + 12)[1] == pytest.approx(17.0), "the last row carried on at the wall's rate"
    assert store.wall_time("nervelet", RUN, 10.0) == pytest.approx(base + 5)
    assert store.program_time("nobody", base) is None and store.wall_time("nervelet", "no-run", 1.0) is None
    found = api.call("find", sources=["nervelet"], since=base - 1, until=base + 30, scope="all")["results"]
    fell = next(r for r in found if "fell" in r["text"])
    assert fell["run"] == RUN and fell["tp"] == pytest.approx(15.0, abs=0.01), "find gives run and tp"
    seen = api.call("look", source="nervelet", run=RUN, tp=15.0)
    assert seen["kind"] == "text" and "fell" in seen["text"] and seen["run"] == RUN and seen["tp"] == 15.0
    assert [folder.name for folder in (home / "spool" / "nervelet").iterdir()] == [RUN], "the run is not ended"
    assert not list((home / "spool" / "nervelet" / RUN).glob("*.jsonl")), "read batches are deleted"


def test_frames_become_a_segment_whose_frame_times_match_their_wall_times_to_the_millisecond(home, served):
    config(home)
    folder = run_folder(home, "front_cam")
    base = time.time() - 100
    walls = [base + w for w in (0.0, 0.137, 0.337, 0.648, 0.6985, 1.4685)]
    write_batch(folder, 1, frames=[(w, 1.0 + i, i, (64, 48)) for i, w in enumerate(walls[:3])])
    write_batch(folder, 2, frames=[(w, 1.0 + i, i + 3, (64, 48)) for i, w in enumerate(walls[3:])])
    end(folder)
    api = served().api
    store = Store(home)
    item = wait_for(lambda: store.items_of("front_cam"), what="the segment", timeout=30)[0]
    assert item.kind == "video" and item.meta == {"run": RUN, "frames": 6} and item.path.startswith("streams/front_cam/")
    assert item.t0 == pytest.approx(walls[0], abs=0.0005)
    gaps = [b - a for a, b in zip(walls, walls[1:])]
    last = sorted(gaps)[len(gaps) // 2]  # the last frame takes the median gap
    assert item.t1 == pytest.approx(walls[-1] + last, abs=0.002)
    seconds = [sec for _, sec in reader._times(store.path_of(item))]
    assert [item.t0 + sec for sec in seconds] == pytest.approx(walls, abs=0.0015), "frame times match the wall to the ms"
    for index, wall in enumerate(walls):
        shown = api.call("look", source="front_cam", t=wall + 0.001)  # pts are whole milliseconds
        assert shown["item"] == item.id and dominant(shown["image"]["path"]) == index, index
    wait_for(lambda: not (home / "spool" / "front_cam" / RUN).exists(), what="the finished run to be removed")
    assert store.cursors("front_cam") == [], "the finished run is removed with its cursors"


def test_a_frame_of_another_size_starts_a_new_segment_and_waiting_frames_wait_for_their_span(home, served):
    config(home, '[sources.cam]\nkind = "program"\nsegment_s = 60\n')
    folder = run_folder(home, "cam")
    base = time.time() - 100
    write_batch(folder, 1, frames=[(base, None, 0, (64, 48)), (base + 1, None, 1, (64, 48)), (base + 2, None, 2, (80, 60)),
                                   (base + 3, None, 3, (80, 60))])
    served()
    store = Store(home)
    items = wait_for(lambda: len(store.items_of("cam")) == 2 and store.items_of("cam"), what="two segments", timeout=30)
    assert [(i.meta["frames"], i.t0) for i in items] == [(2, pytest.approx(base)), (2, pytest.approx(base + 2))]
    sizes = [reader._decode(store.path_of(i), [reader._times(store.path_of(i))[0]]) for i in items]
    assert [next(s)[1].size for s in sizes] == [(64, 48), (80, 60)]


def test_frames_that_are_recent_wait_until_they_span_segment_s_or_the_run_ends(home, served):
    config(home, '[sources.cam]\nkind = "program"\nsegment_s = 2\n')
    folder = run_folder(home, "cam")
    now = time.time()
    write_batch(folder, 1, frames=[(now - 0.4, None, 0, (64, 48)), (now - 0.2, None, 1, (64, 48))])
    served()
    store = Store(home)
    time.sleep(1.5)
    assert store.items_of("cam") == [] and store.records() == [], "0.2 s of frames is not a segment, and 10 s have not passed"
    write_batch(folder, 2, frames=[(now + 0.1, None, 2, (64, 48)), (now + 2.4, None, 3, (64, 48))])
    item = wait_for(lambda: store.items_of("cam"), what="a segment once a frame is past segment_s", timeout=30)[0]
    assert item.meta["frames"] == 3 and item.t0 == pytest.approx(now - 0.4), "the frame 2.4 s on starts the next"
    write_batch(folder, 3, frames=[(now + 2.6, None, 4, (64, 48))])
    time.sleep(1)
    assert len(store.items_of("cam")) == 1
    end(folder)
    wait_for(lambda: len(store.items_of("cam")) == 2, what="the rest once the run ended", timeout=30)
    assert store.items_of("cam")[1].meta["frames"] == 2


def test_a_crash_between_an_encode_and_its_commit_encodes_the_frames_again_and_repeats_no_record(home, monkeypatch):
    config(home, '[sources.cam]\nkind = "program"\nsegment_s = 60\n')
    store = Store(home)
    folder = run_folder(home, "cam")
    base = time.time() - 100
    write_batch(folder, 1, records=[(base, 1.0, {"n": 1})], frames=[(base, 1.0, 0, (64, 48)), (base + 1, 2.0, 1, (64, 48))])
    write_batch(folder, 2, records=[(base + 2, 3.0, {"n": 2})], frames=[(base + 2, 3.0, 2, (64, 48))])
    end(folder)

    def drive(spool: Spool, done) -> None:
        end_at = time.monotonic() + 30
        while not done():
            spool.next_scan = 0
            spool.step(time.time())
            assert time.monotonic() < end_at, "the spool did not settle"
            time.sleep(0.05)

    first = Spool(store, configuration.load(home))
    real = store.add_item
    monkeypatch.setattr(store, "add_item", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the daemon died here")))
    with pytest.raises(RuntimeError):
        drive(first, lambda: False)
    monkeypatch.setattr(store, "add_item", real)
    segment = next((store.streams / "cam" / RUN).glob("*.mkv"))
    assert segment.exists() and store.items_of("cam") == [], "the file was renamed, and nothing was committed"
    assert len(store.records(sources=["cam"])) == 2, "the records were committed with their cursor"
    assert store.cursor("cam", f"{RUN}:frames") is None

    second = Spool(store, configuration.load(home))  # a new process: nothing is remembered
    drive(second, lambda: store.items_of("cam") and not (home / "spool" / "cam" / RUN).exists())
    (item,) = store.items_of("cam")
    assert item.meta["frames"] == 3 and store.path_of(item) == segment, "the same frames, in one segment, in the same file"
    assert [json.loads(r.data)["n"] for r in store.records(sources=["cam"])] == [1, 2], "no record twice"
    assert len(list((store.streams / "cam" / RUN).glob("*"))) == 1 and store.cursors("cam") == []


def test_a_batch_that_does_not_parse_moves_to_bad_and_the_run_goes_on(home, served):
    config(home)
    folder = run_folder(home, "prog")
    base = time.time() - 100
    write_batch(folder, 1, records=[(base, None, {"n": 1})])
    (folder / "00000002.jsonl").write_text('{"wall": 1.0, "data": {"n": 2}}\nthis is not json\n', encoding="utf-8")
    (folder / "00000003.jsonl").write_text('{"wall": "late", "t": null, "data": {}}\n', encoding="utf-8")
    write_batch(folder, 4, records=[(base + 1, None, {"n": 4})])
    served()
    store = Store(home)
    wait_for(lambda: len(store.records(sources=["prog"])) == 2, what="the good batches", timeout=30)
    assert [json.loads(r.data)["n"] for r in store.records(sources=["prog"])] == [1, 4], "nothing of a bad batch is kept"
    bad = home / "spool" / "_bad"
    assert sorted(p.name for p in bad.iterdir()) == [f"prog-{RUN}-00000002.jsonl", f"prog-{RUN}-00000002.jsonl.error",
                                                      f"prog-{RUN}-00000003.jsonl", f"prog-{RUN}-00000003.jsonl.error"]
    assert "line 2" in (bad / f"prog-{RUN}-00000002.jsonl.error").read_text(encoding="utf-8")
    assert "wall" in (bad / f"prog-{RUN}-00000003.jsonl.error").read_text(encoding="utf-8")


def test_a_stream_named_like_a_folder_source_is_refused_and_left_alone(home, served, tmp_path):
    config(home, f'[sources.notes]\nkind = "folder"\npath = "{tmp_path.as_posix()}"\n')
    folder = run_folder(home, "notes")
    write_batch(folder, 1, records=[(time.time() - 5, None, {"n": 1})])
    api = served().api
    refused = home / "spool" / "notes" / "refused.json"
    wait_for(refused.exists, what="refused.json", timeout=30)
    assert "folder source" in json.loads(refused.read_text(encoding="utf-8"))["reason"]
    time.sleep(0.8)
    assert Store(home).records() == [] and (folder / "00000001.jsonl").exists(), "the run is left alone"
    assert configuration.load(home).sources["notes"].kind == "folder", "the owner's table is untouched"
    api.call("source_set", name="notes", remove=True)
    wait_for(lambda: not refused.exists(), what="refused.json to go once the clash did", timeout=30)
    wait_for(lambda: Store(home).records(sources=["notes"]), what="the run to be read", timeout=30)


def test_a_restarted_daemon_reads_each_record_once_and_clears_the_parts_a_crash_left(home, served):
    config(home)
    folder = run_folder(home, "prog")
    base = time.time() - 100
    write_batch(folder, 1, records=[(base, 1.0, {"n": 1}), (base + 1, 2.0, {"n": 2})])
    first = served()
    store = Store(home)
    wait_for(lambda: len(store.records(sources=["prog"])) == 2, what="the first batch", timeout=30)
    first.stop()
    (store.streams / "prog" / RUN).mkdir(parents=True, exist_ok=True)
    part = store.streams / "prog" / RUN / "00000000-000.mkv.part"
    part.write_bytes(b"half an encode")
    write_batch(folder, 2, records=[(base + 2, 3.0, {"n": 3})])
    # a batch the daemon was in the middle of reading when it died: the cursor is behind the file
    store.set_cursor("prog", RUN, offset=0)
    store.db().execute("DELETE FROM records WHERE source = 'prog'")
    served()
    wait_for(lambda: len(store.records(sources=["prog"])) == 1, what="the second batch", timeout=30)
    assert not part.exists(), "a .part is deleted at start"
    assert [json.loads(r.data)["n"] for r in store.records(sources=["prog"])] == [3], "batches behind the cursor stay read"


def drive(spool: Spool, steps: int = 3) -> None:
    for _ in range(steps):
        spool.next_scan = 0
        spool.step(time.time())


def test_a_backlog_after_an_outage_is_cut_into_segments_of_segment_s(home, served):
    config(home, '[sources.cam]\nkind = "program"\nsegment_s = 2\n')
    folder = run_folder(home, "cam")
    base = time.time() - 600  # the daemon was down: 13 frames, half a second apart, ten minutes ago
    write_batch(folder, 1, frames=[(base + 0.5 * i, None, i, (64, 48)) for i in range(13)])
    end(folder)
    served()
    store = Store(home)
    items = wait_for(lambda: len(store.items_of("cam")) == 4 and store.items_of("cam"), what="four segments", timeout=30)
    assert [item.meta["frames"] for item in items] == [4, 4, 4, 1], "not one segment of the whole backlog"
    assert all(item.t1 - item.t0 <= 2.6 for item in items)
    assert items[1].t0 == pytest.approx(base + 2.0)


def test_a_run_is_idle_by_its_cursor_and_never_by_its_files_or_with_anything_unread(home, monkeypatch):
    from mms import spool as spool_module
    config(home, '[sources.cam]\nkind = "program"\n')
    store = Store(home)
    old = time.time() - 2 * 86400
    stale = run_folder(home, "cam", "stale-run")  # no end, no batch, and the cursor has not moved in two days
    store.set_cursor("cam", "stale-run", offset=0, mtime=old)
    aged = run_folder(home, "cam", "aged-files-run")  # its files are old, its cursor just began
    for file in aged.iterdir():
        os.utime(file, (old, old))
    os.utime(aged, (old, old))
    backlog = run_folder(home, "cam", "backlog-run")  # the cursor is stale, but there is a lot unread
    store.set_cursor("cam", "backlog-run", offset=0, mtime=old)
    for seq in range(1, 6):
        write_batch(backlog, seq, records=[(time.time(), None, {"n": seq})])
    monkeypatch.setattr(spool_module, "BATCHES_PER_STEP", 1)
    spool = Spool(store, configuration.load(home))
    drive(spool, 1)
    assert not stale.exists() and store.cursor("cam", "stale-run") is None, "idle by its cursor, and nothing unread"
    assert aged.exists(), "old files do not make a run idle"
    assert backlog.exists(), "unread batches are never idle, however stale the cursor"
    drive(spool, 8)
    assert len(store.records(sources=["cam"])) == 5 and backlog.exists(), "read, and its cursor moved just now"
    assert store.cursor("cam", "backlog-run")["mtime"] > time.time() - 30
    assert aged.exists()


def test_a_removed_program_table_is_made_again_at_the_next_log_not_ingested_without_one(home, served):
    config(home)
    folder = run_folder(home, "nervelet")
    write_batch(folder, 1, records=[(time.time() - 5, None, {"n": 1})])
    api = served().api
    store = Store(home)
    wait_for(lambda: "nervelet" in configuration.load(home).sources and store.records(sources=["nervelet"]),
             what="the first log, and its table", timeout=30)
    api.call("source_set", name="nervelet", remove=True)
    wait_for(lambda: "nervelet" not in configuration.load(home).sources and
             api.call("status")["sources"] and not any(s["name"] == "nervelet" and s["kind"] for s in api.call("status")["sources"]),
             what="the table to go", timeout=30)
    second = run_folder(home, "nervelet", "20261031T130000Z-9-aaaa")
    write_batch(second, 1, records=[(time.time() - 4, None, {"n": 2})])
    wait_for(lambda: len(store.records(sources=["nervelet"])) == 2, what="the second run", timeout=30)
    assert "nervelet" in configuration.load(home).sources, "its table is back, with a cap, so the janitor has one to keep"
    assert configuration.load(home).sources["nervelet"].keep.max_gb == 5


def test_a_program_table_changed_to_another_kind_refuses_the_stream_at_once(home, served, tmp_path):
    config(home)
    folder = run_folder(home, "nervelet")
    write_batch(folder, 1, records=[(time.time() - 5, None, {"n": 1})])
    served()
    store = Store(home)
    wait_for(lambda: store.records(sources=["nervelet"]), what="the first log", timeout=30)
    (home / "mms.toml").write_text(
        f'[memory]\nloop_s = 0.1\nfree_gb = 0\n[sources.nervelet]\nkind = "folder"\npath = "{tmp_path.as_posix()}"\n', encoding="utf-8")
    refused = home / "spool" / "nervelet" / "refused.json"
    second = run_folder(home, "nervelet", "20261031T130000Z-9-aaaa")
    write_batch(second, 1, records=[(time.time() - 4, None, {"n": 2})])
    wait_for(refused.exists, what="the stream to be refused", timeout=30)
    assert len(store.records(sources=["nervelet"])) == 1
