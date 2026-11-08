"""Folder, log and command sources (TESTBED.md item 2): the daemon as a process, real files, real child processes."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from conftest import mms, mms_json, wait_for, write_config

from mms.store import Store


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def config(home: Path, *sources: str) -> None:
    write_config(home, "[memory]\nloop_s = 0.1\njanitor_s = 1\nfree_gb = 0\n\n" + "\n".join(sources))


def folder(name: str, path: Path, glob: str = "**/*.md", extra: str = "") -> str:
    return f'[sources.{name}]\nkind = "folder"\npath = "{path.as_posix()}"\nglob = "{glob}"\nrescan_s = 0.2\n{extra}\n'


def log(name: str, path: Path, glob: str = "*.log", extra: str = "") -> str:
    return f'[sources.{name}]\nkind = "log"\npath = "{path.as_posix()}"\nglob = "{glob}"\nrescan_s = 0.2\n{extra}\n'


def results(home: Path, *args: str) -> list[dict]:
    return mms_json(home, "find", *args)["results"]


def test_versions_follow_a_file(home, store, daemon, tmp_path):
    notes = tmp_path / "notes"
    (notes / "sub").mkdir(parents=True)
    config(home, folder("notes", notes))
    note = notes / "sub" / "a.md"
    note.write_bytes(b"one\n")
    (notes / "skipped.txt").write_text("not matched by the glob")
    daemon()

    first = wait_for(lambda: store.current_version("notes", "sub/a.md"), what="the first version")
    assert abs(first.t0 - time.time()) < 5
    assert store.read_bytes(first) == b"one\n"
    assert first.path.endswith(".zst"), "text is stored compressed"

    between = time.time()
    note.write_bytes(b"two\n")
    wait_for(lambda: len(store.items_of("notes")) == 2, what="a changed file to add a version")
    note.write_bytes(b"two\n")  # saved again with the same content: no version
    note.write_bytes(b"three\n")
    wait_for(lambda: len(store.items_of("notes")) == 3, what="the third version")
    one, two, three = store.items_of("notes")
    assert [store.read_bytes(v) for v in (one, two, three)] == [b"one\n", b"two\n", b"three\n"]
    assert one.t1 == two.t0 and two.t1 == three.t0, "a new version closes the last"
    assert three.t1 is None
    assert {item.name for item in store.items_of("notes")} == {"sub/a.md"}

    then = results(home, "--as-of", repr(between), "--source", "notes")
    assert [entry["id"] for entry in then] == [one.id], "as of a time before the change, the old version"
    assert [entry["id"] for entry in results(home, "--as-of", "now")] == [three.id]

    note.unlink()
    wait_for(lambda: store.item(three.id).t1 is not None, what="a deleted file to end its version")
    assert results(home, "--as-of", "now") == []
    assert [entry["id"] for entry in results(home, "--as-of", repr(between))] == [one.id]


def test_a_file_that_comes_back_is_a_new_version(home, store, daemon, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    config(home, folder("notes", notes))
    note = notes / "a.md"
    note.write_text("same")
    daemon()
    first = wait_for(lambda: store.current_version("notes", "a.md"), what="the version")
    note.unlink()
    wait_for(lambda: store.item(first.id).t1 is not None, what="its end")
    note.write_text("same")
    wait_for(lambda: len(store.items_of("notes")) == 2, what="a version for the file that came back")


def test_a_restarted_daemon_carries_on(home, store, daemon, tmp_path):
    notes, logs = tmp_path / "notes", tmp_path / "logs"
    notes.mkdir()
    logs.mkdir()
    config(home, folder("notes", notes), log("app", logs))
    (notes / "a.md").write_text("one")
    (logs / "x.log").write_bytes(b"first\n")
    first = daemon()
    wait_for(lambda: store.current_version("notes", "a.md") and len(store.records()) == 1, what="the first pass")
    first.stop()

    (notes / "b.md").write_text("two")
    (logs / "x.log").write_bytes(b"first\nsecond\n")
    daemon()
    wait_for(lambda: store.current_version("notes", "b.md") and len(store.records()) == 2, what="the second pass")
    assert len(store.items_of("notes")) == 2, "the unchanged file was not seen again"
    assert [json.loads(r.data)["line"] for r in store.records()] == ["first", "second"], "no line twice"


def test_a_big_file_is_indexed_in_place(home, store, daemon, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    config(home, folder("notes", notes, glob="*", extra="copy_max_mb = 0.001"))
    (notes / "small.md").write_text("small")
    big = notes / "big.md"
    big.write_text("big " * 1000)
    daemon()
    wait_for(lambda: len(store.items_of("notes")) == 2, what="both files")
    by_name = {item.name: item for item in store.items_of("notes")}
    assert not by_name["small.md"].in_place
    assert by_name["big.md"].in_place and Path(by_name["big.md"].path) == big.resolve()
    assert not list((store.files / "notes").glob("*big.md*")), "a big file is never copied"
    assert store.read_bytes(by_name["big.md"]) == big.read_bytes()


def test_a_log_makes_a_record_of_each_line_and_reads_a_time_only_from_a_named_field(home, store, daemon, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    config(home, log("app", logs, "app.log", 'time_field = "stamp"'), log("plain", logs, "plain.log"))
    app, plain = logs / "app.log", logs / "plain.log"
    stamped = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc).timestamp()
    app.write_bytes(b'{"stamp": "2026-10-30T12:00:00Z", "msg": "first alpha"}\nplain beta line\npartial')
    plain.write_bytes(b'{"stamp": "2026-10-30T12:00:00Z", "t": 1000.5, "msg": "named nothing"}\n')
    daemon()
    wait_for(lambda: len(store.records(sources=["app"])) == 2 and store.records(sources=["plain"]),
             what="two complete lines")
    first, second = store.records(sources=["app"])
    assert first.t == stamped and json.loads(first.data)["msg"] == "first alpha", "the named field sets the time"
    assert abs(second.t - time.time()) < 10 and json.loads(second.data) == {"line": "plain beta line"}
    (unnamed,) = store.records(sources=["plain"])
    assert abs(unnamed.t - time.time()) < 10, "a `t` or `stamp` field that is not named is not read"

    with open(app, "ab") as file:
        file.write(b" gamma\n")  # the partial line is complete now
    wait_for(lambda: len(store.records(sources=["app"])) == 3, what="the partial line, once complete")
    assert json.loads(store.records(sources=["app"])[-1].data) == {"line": "partial gamma"}

    app.write_bytes(b"fresh delta\n")  # shorter than before: read from the start
    wait_for(lambda: len(store.records(sources=["app"])) == 4, what="the line of a file that shrank")
    assert json.loads(store.records(sources=["app"])[-1].data) == {"line": "fresh delta"}

    found = results(home, "--text", "beta", "--source", "app")
    assert [entry["type"] for entry in found] == ["record"] and "plain beta line" in found[0]["text"]
    assert results(home, "--text", "no-such-word") == []


CHILD = """
import json, os, sys, time
print(json.dumps({"t": 1000.5, "temp": 21.5, "pid": os.getpid()}), flush=True)
print("not json cafe", flush=True)
print(json.dumps({"temp": 22, "pid": os.getpid()}), flush=True)
time.sleep(0.2)
"""


def command_source(name: str, script: Path, extra: str = "") -> str:
    return f'[sources.{name}]\nkind = "command"\ncommand = ["{Path(sys.executable).as_posix()}", "{script.as_posix()}"]\n{extra}\n'


def test_command_lines_become_records_and_the_process_comes_back(home, store, daemon, tmp_path):
    script = tmp_path / "read_temp.py"
    script.write_text(CHILD)
    config(home, command_source("thermometer", script, 'time_field = "t"'), command_source("blind", script))
    daemon()
    wait_for(lambda: len(store.records(sources=["thermometer"])) >= 6 and len(store.records(sources=["blind"])) >= 3,
             what="the lines of two runs", timeout=30)
    records = store.records(sources=["thermometer"])
    stamped = [r for r in records if r.t == 1000.5]
    assert stamped and json.loads(stamped[0].data)["temp"] == 21.5, "the named field sets the time"
    unstamped = [r for r in records if json.loads(r.data).get("temp") == 22]
    assert abs(unstamped[0].t - time.time()) < 15, "otherwise the arrival time"
    assert all(r.t != 1000.5 for r in store.records(sources=["blind"])), "a `t` that is not named is not read"
    assert len({json.loads(r.data).get("pid") for r in records} - {None}) >= 2, "restarted after it exited"
    assert [entry["id"] for entry in results(home, "--text", "cafe")][0] in {r.id for r in records} | {
        r.id for r in store.records(sources=["blind"])}
    assert any(json.loads(r.data) == {"line": "not json cafe"} for r in records)


def test_stopping_the_daemon_stops_its_commands(home, store, daemon, tmp_path):
    script, beat = tmp_path / "beat.py", tmp_path / "beat.txt"
    script.write_text('import sys, time\nprint("up", flush=True)\nfor _ in range(600):\n'
                      '    open(sys.argv[1], "a").write("x")\n    time.sleep(0.05)\n')
    config(home, f'[sources.beat]\nkind = "command"\ncommand = ["{Path(sys.executable).as_posix()}", '
                 f'"{script.as_posix()}", "{beat.as_posix()}"]\n')
    running = daemon()
    wait_for(lambda: beat.exists() and beat.stat().st_size > 3 and store.records(), what="the process to run")
    assert running.stop() == 0
    size = beat.stat().st_size
    time.sleep(0.5)
    assert beat.stat().st_size == size, "the process was still running after the daemon stopped"
    assert json.loads(store.records()[0].data) == {"line": "up"}


def test_an_in_place_file_has_one_row_that_follows_it(home, store, daemon, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    config(home, folder("notes", notes, glob="*", extra="copy_max_mb = 0.001"))
    big = notes / "big.md"
    big.write_text("old " * 1000)
    daemon()
    first = wait_for(lambda: store.current_version("notes", "big.md"), what="the file indexed")
    assert first.in_place and first.sha256 is None, "nothing is hashed"
    before = time.time()

    big.write_text("new and longer " * 1000)
    row = wait_for(lambda: store.items_of("notes")[0].bytes == big.stat().st_size and store.items_of("notes")[0],
                   what="the row to follow the file")
    assert len(store.items_of("notes")) == 1 and row.id == first.id, "a change updates the row, adds none"
    assert row.t0 > before and row.t1 is None
    assert results(home, "--as-of", repr(before), "--source", "notes") == [], \
        "before the change there is no history, not today's content under an old time"
    assert [entry["id"] for entry in results(home, "--as-of", "now")] == [row.id]

    big.unlink()
    wait_for(lambda: store.item(row.id).t1 is not None, what="the end of a vanished file")
    assert store.item(row.id).state == "gone"
    assert store.select(sources=["notes"]) == [] and results(home, "--as-of", "now") == []
    big.write_text("a different file at the same path " * 100)
    wait_for(lambda: store.current_version("notes", "big.md"), what="the file back")
    assert len(store.items_of("notes")) == 2
    assert [item.id for item in store.select(sources=["notes"])] == [store.current_version("notes", "big.md").id], \
        "the old row does not read the new content"


def test_a_tail_commits_records_and_cursor_together(home, store):
    rows = [(1.0, json.dumps({"line": "a"})), (2.0, json.dumps({"line": "b"}))]
    with pytest.raises(ValueError):
        store.add_records("app", rows, cursor=("x.log", {"bogus": 1}))
    assert store.records() == [] and store.cursor("app", "x.log") is None, "all or none"
    store.add_records("app", rows, cursor=("x.log", {"offset": 4, "size": 4}))
    assert len(store.records()) == 2 and store.cursor("app", "x.log")["offset"] == 4


def test_a_tail_reads_in_bounded_chunks_and_only_whole_lines_advance(home, store, tmp_path, monkeypatch):
    from mms import config as configuration
    from mms import ingest
    logs = tmp_path / "logs"
    logs.mkdir()
    config(home, log("app", logs))
    monkeypatch.setattr(ingest, "TAIL_CHUNK", 64)
    monkeypatch.setattr(ingest, "TAIL_PASS", 3)
    watcher = ingest.Ingest(store, configuration.load(home))
    source = watcher.config.sources["app"]
    lines = [f"line {i:03d} " + "x" * 20 for i in range(40)]
    width = len(lines[0]) + 1  # 30 bytes with the newline, so a chunk of 64 ends mid-line
    file = logs / "big.log"
    file.write_bytes(("\n".join(lines) + "\n").encode())

    assert watcher.scan(source, time.time()) is True, "three chunks of 64 bytes leave most of the file behind"
    cursor = store.cursor("app", "big.log")
    assert cursor["offset"] <= 3 * 64 and cursor["offset"] % width == 0, "the cursor sits at the end of a whole line"
    assert len(store.records()) == cursor["offset"] // width
    passes = 1
    while watcher.scan(source, time.time()):
        passes += 1
        assert passes < 20
    assert [json.loads(r.data)["line"] for r in store.records()] == lines, "every line once, in order"
    assert watcher.scan(source, time.time()) is False

    file.write_bytes(file.read_bytes() + b"q" * 200)  # a line with no end, longer than a chunk, is cut and not stuck
    while watcher.scan(source, time.time()):
        pass
    cut = [json.loads(r.data)["line"] for r in store.records()][40:]
    assert "".join(cut) == "q" * 192 and store.cursor("app", "big.log")["offset"] == file.stat().st_size - 8, \
        "whole chunks of the long line were taken; the last 8 bytes wait for a newline"


def test_concurrent_remembers_leave_one_current_version(home, detached, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    first = tmp_path / "x.md"
    first.write_text("version one\n")
    mms(home, "remember", "--path", str(first), "--name", "n.md", "--source", "agent")
    contents = {}
    for i in range(4):
        contents[i] = tmp_path / f"v{i}.md"
        contents[i].write_text(f"version {i}\n" * 2000)  # a few ms of hashing and compressing each
    with ThreadPoolExecutor(4) as pool:
        done = list(pool.map(lambda i: mms(home, "remember", "--path", str(contents[i]), "--name", "n.md",
                                          "--source", "agent"), contents))
    assert all(d.returncode == 0 for d in done)
    store = Store(home)
    versions = store.items_of("agent")
    assert len(versions) == 5 and len([v for v in versions if v.t1 is None]) == 1, "exactly one open version"
    assert all(a.t1 == b.t0 for a, b in zip(versions, versions[1:])), "each version closes where the next begins"
    assert all(hashlib.sha256(store.read_bytes(v)).hexdigest() == v.sha256 for v in versions), \
        "what is stored is what was hashed"
    assert not list((store.files / "agent").glob("*.part")), "no partial file is left"
    store.close()


def test_a_watched_path_that_goes_away_is_not_scanned_and_comes_back_unchanged(home, store, daemon, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    config(home, folder("notes", notes))
    (notes / "a.md").write_text("alpha")
    (notes / "b.md").write_text("beta")
    running = daemon()
    wait_for(lambda: len(store.items_of("notes")) == 2, what="both files")
    kept = [item.id for item in store.items_of("notes")]
    copies = sorted(p.name for p in (store.files / "notes").iterdir())

    away = tmp_path / "unmounted"
    notes.rename(away)  # an unmounted drive, an offline share
    wait_for(lambda: "is not there" in running.text(), what="the watch to notice")
    assert next(s for s in running.api.call("status")["sources"] if s["name"] == "notes")["state"] == "missing"
    time.sleep(1)  # several rescans
    assert [item.t1 for item in store.items_of("notes")] == [None, None], "nothing ended"
    assert len(store.cursors("notes")) == 2, "the cursors stay"

    away.rename(notes)
    wait_for(lambda: "is back" in running.text(), what="the watch to see it again")
    time.sleep(1)
    assert [item.id for item in store.items_of("notes")] == kept and all(i.t1 is None for i in store.items_of("notes"))
    assert sorted(p.name for p in (store.files / "notes").iterdir()) == copies, "nothing was copied again"


def test_rows_take_the_project_of_their_source(home, store, daemon, tmp_path):
    notes, logs = tmp_path / "notes", tmp_path / "logs"
    notes.mkdir()
    logs.mkdir()
    mine = "git:0123456789abcdef"
    config(home, folder("notes", notes, extra=f'project = "{mine}"'), log("app", logs, extra=f'project = "{mine}"'),
           folder("ambient", notes))
    (notes / "a.md").write_text("one")
    (logs / "x.log").write_bytes(b"line\n")
    daemon()
    wait_for(lambda: store.items_of("notes") and store.records(sources=["app"]) and store.items_of("ambient"),
             what="rows")
    assert store.items_of("notes")[0].project == mine and store.records(sources=["app"])[0].project == mine
    assert store.items_of("ambient")[0].project is None


def test_a_big_tree_is_scanned_a_budget_at_a_time_and_what_vanished_is_found_when_the_pass_ends(home, store, tmp_path,
                                                                                                    monkeypatch):
    from mms import config as configuration
    from mms import ingest
    notes = tmp_path / "notes"
    notes.mkdir()
    for i in range(100):
        (notes / f"{i:03d}.md").write_text(f"note {i}")
    config(home, folder("notes", notes, glob="*.md"))
    monkeypatch.setattr(ingest, "STEP_FILES", 20)
    watcher = ingest.Ingest(store, configuration.load(home))
    source = watcher.config.sources["notes"]
    steps, counts = 0, []
    while True:
        steps += 1
        more = watcher.scan(source, time.time())
        counts.append(len(store.items_of("notes")))
        if steps == 1:
            (notes / "000.md").unlink()  # gone while the pass is under way
        if not more:
            break
        assert steps < 20
    assert steps == 6 and counts == [20, 40, 60, 80, 100, 100], "twenty files a step; the sixth finds the end"
    assert store.current_version("notes", "000.md") is not None, "not ended before the pass is over"
    while watcher.scan(source, time.time()):  # the next pass sees it is gone, when it has looked at them all
        assert store.current_version("notes", "000.md") is not None
    assert store.current_version("notes", "000.md") is None and len(store.items_of("notes")) == 100


def test_a_step_copies_at_most_its_byte_budget(home, store, tmp_path, monkeypatch):
    from mms import config as configuration
    from mms import ingest
    notes = tmp_path / "notes"
    notes.mkdir()
    for i in range(6):
        (notes / f"{i}.bin").write_bytes(bytes([i]) * 1000)
    config(home, folder("notes", notes, glob="*.bin"))
    monkeypatch.setattr(ingest, "STEP_BYTES", 2500)
    watcher = ingest.Ingest(store, configuration.load(home))
    source = watcher.config.sources["notes"]
    assert watcher.scan(source, time.time()) is True and len(store.items_of("notes")) == 3, "the third file passes 2,500"
    assert watcher.scan(source, time.time()) is True and len(store.items_of("notes")) == 6
    assert watcher.scan(source, time.time()) is False, "the pass ends"


def test_a_floor_that_a_write_would_pass_is_shown_in_status_and_the_source_goes_on_when_there_is_room(
        home, store, served, tmp_path, monkeypatch):
    from test_remember_look import fake_free
    notes = tmp_path / "notes"
    notes.mkdir()
    for i in range(4):
        (notes / f"{i}.bin").write_bytes(b"\x00" + bytes([i + 1]) * 999_999)  # a NUL makes it binary: stored as it is
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 20\njanitor_s = 600\n" + folder("notes", notes, glob="*.bin"))
    fake_free(monkeypatch, home, 20, 2_500_000)
    api = served().api
    wait_for(lambda: next(s for s in api.call("status")["sources"] if s["name"] == "notes").get("error"),
             what="the floor to show in status")
    state = next(s for s in api.call("status")["sources"] if s["name"] == "notes")
    assert "floor" in state["error"] and len(store.items_of("notes")) <= 2
    fake_free(monkeypatch, home, 20, 50_000_000)
    wait_for(lambda: len(store.items_of("notes")) == 4, what="the rest, once there is room")
    wait_for(lambda: "error" not in next(s for s in api.call("status")["sources"] if s["name"] == "notes"),
             what="the error to clear")
