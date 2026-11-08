"""Retention (TESTBED.md item 3): the janitor inside the daemon process. Items are written into the
store before the daemon starts, so a pass has something to cap; sources are `push` so no loop touches them."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import pytest
from conftest import Api, mms, mms_json, wait_for, write_config

from mms import config as configuration
from mms.daemon import STOP, Daemon
from mms.janitor import Janitor, crowded
from mms.store import Store

DAY = 86400


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def config(home: Path, *sources: str, memory: str = "") -> None:
    write_config(home, f"[memory]\nloop_s = 0.1\njanitor_s = 0.3\n{memory}\n" + "\n".join(sources))


def source(name: str, keep: str = "") -> str:
    return f'[sources.{name}]\nkind = "push"\n{"keep = { " + keep + " }" if keep else ""}\n'


def fill(store: Store, name: str, count: int, size: int = 1000, *, end: float | None = None, step: float = 60,
         kind: str = "video", launch: str = "fake") -> list[int]:
    """`count` segments of `size` bytes, `step` seconds each, the last ending at `end` (default now)."""
    end = time.time() if end is None else end
    folder = store.streams / name / launch
    folder.mkdir(parents=True, exist_ok=True)
    ids = []
    for i in range(count):
        path = folder / f"{i:06d}.mkv"
        path.write_bytes(b"x" * size)
        t0 = end - (count - i) * step
        ids.append(store.add_item(name, kind, t0, t0 + step, store.relative(path), size))
    return ids


def left(store: Store, name: str) -> list[int]:
    return [item.id for item in store.items_of(name)]


def test_crowded_takes_the_most_crowded_for_its_age():
    # a dense old cluster, then sparse newer rows
    times = [0, 1, 2, 3, 4, 100, 200, 300, 400, 500]
    rows = [(i, t, 10, True) for i, t in enumerate(times)]
    chosen = crowded(rows, 500, over=25)
    assert len(chosen) == 3 and chosen[0] in (1, 2, 3)
    assert set(chosen) <= {1, 2, 3, 4}, "the cluster thins before the sparse rows do"
    everything = crowded(rows, 500, over=10**9)
    assert 0 not in everything and 9 not in everything, "the oldest and the newest are never taken"
    protected = [(i, t, 10, i not in (1, 2)) for i, t in enumerate(times)]
    assert not {1, 2} & set(crowded(protected, 500, over=10**9)), "rows that are not deletable stay"


def test_drop_oldest_deletes_the_oldest_down_to_the_cap(home, store, daemon):
    config(home, source("cam", 'max_gb = 0.00001, when_full = "drop_oldest"'))  # 10 kB
    ids = fill(store, "cam", 20)  # 20 kB
    daemon()
    wait_for(lambda: len(left(store, "cam")) <= 11, what="the source to drop to its cap")
    assert left(store, "cam") == ids[-11:], "the newest ten, and the newest of all, which the cap does not count"
    assert len(list((store.streams / "cam" / "fake").glob("*.mkv"))) == 11, "the others' files are unlinked"
    assert "janitor cam: deleted 9 items" in daemon_log(home)


def daemon_log(home: Path) -> str:
    return (home / "daemon.test.log").read_text(encoding="utf-8", errors="replace")


def test_thin_keeps_history_everywhere_but_sparser_with_age(home, store, daemon):
    config(home, source("cam", 'max_gb = 0.00001, when_full = "thin"'))
    ids = fill(store, "cam", 40, step=60)
    daemon()
    wait_for(lambda: len(left(store, "cam")) <= 11, what="the source to thin to its cap")
    kept = left(store, "cam")
    assert 9 <= len(kept) <= 11
    assert kept[0] == ids[0] and kept[-1] == ids[-1], "the oldest and the newest stay"
    positions = [ids.index(i) for i in kept]
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert gaps[0] >= gaps[-1] and max(gaps) > 1, f"older history is sparser: gaps {gaps}"
    assert sum(1 for p in positions if p >= 36) >= 2, "the newest minutes stay at full rate"


def test_keep_in_place_and_the_newest_survive_a_cap(home, store, daemon, tmp_path):
    config(home, source("cam", 'max_gb = 0, when_full = "drop_oldest"'))
    ids = fill(store, "cam", 8)
    outside = tmp_path / "mine.bin"
    outside.write_bytes(b"y" * 5000)
    placed = store.add_item("cam", "file", time.time() - 3000, None, str(outside), 5000, name="mine.bin",
                            in_place=True)
    first = store.item(ids[0])
    store.label(first.t0, first.t1 - 1, "important", "agent", source="cam", keep=True)  # keeps the oldest
    daemon()
    wait_for(lambda: len(left(store, "cam")) == 3, what="everything else to be dropped")
    assert sorted(left(store, "cam")) == sorted([ids[0], placed, ids[-1]])
    assert outside.read_bytes() == b"y" * 5000, "the owner's own file is never touched"
    assert store.path_of(store.item(ids[0])).exists() and store.path_of(store.item(ids[-1])).exists()


def test_what_a_cap_counts_is_what_it_may_delete(home, store, daemon, tmp_path):
    """Kept items, in-place files and the newest item are outside the cap: recording on after a kept day thins the
    new segments to the cap and leaves the kept day alone."""
    config(home, source("cam", 'max_gb = 0.00001, when_full = "thin"'))  # 10 kB
    now = time.time()
    day = fill(store, "cam", 12, end=now - 86400, step=60, launch="day")  # 12 kB, yesterday
    store.label(now - 86400 - 720, now - 86400, "an important day", "agent", source="cam", keep=True)
    outside = tmp_path / "mine.bin"
    outside.write_bytes(b"y" * 5000)
    store.add_item("cam", "file", now - 90000, None, str(outside), 5000, name="mine.bin", in_place=True)
    new = fill(store, "cam", 25, end=now, step=2)  # 25 kB: 24 of them count
    assert store.source_bytes() == {"cam": 24000}
    daemon()
    wait_for(lambda: store.source_bytes()["cam"] <= 10000, what="the new segments to thin to the cap")
    time.sleep(1)  # more passes: nothing else goes
    assert set(day) <= set(left(store, "cam")), "the kept day is untouched"
    assert new[-1] in left(store, "cam"), "the newest stays"
    assert sum(1 for i in left(store, "cam") if i in new) == 11, "14 of the 24 went: 14 kB over the cap"
    assert outside.read_bytes() == b"y" * 5000


def test_drop_oldest_is_one_cutoff_through_items_and_records(home, store, daemon):
    """Items and records go by time together: every one older than the cutoff, and none that is newer."""
    config(home, source("mix", 'max_gb = 0.0001, when_full = "drop_oldest"'))  # 100 kB
    now = time.time()
    old = [(now - 3000 + i * 0.1, '{"n": %d}' % i) for i in range(700)]  # 700 records of 221 each: 155 kB
    store.add_records("mix", old)
    folder = store.streams / "mix" / "fake"
    folder.mkdir(parents=True)
    starts = {}
    for i in range(10):  # 10 segments of 8 kB, from the middle of the old records on
        path = folder / f"{i:06d}.mkv"
        path.write_bytes(b"x" * 8000)
        t0 = now - 3000 + 35 + i * 6
        starts[store.add_item("mix", "video", t0, t0 + 6, store.relative(path), 8000)] = t0
    new = [(now - 1000 + i, '{"n": %d}' % (1000 + i)) for i in range(100)]
    store.add_records("mix", new)
    assert store.source_bytes()["mix"] > 100_000 + 8 * 8000
    daemon()
    wait_for(lambda: store.source_bytes()["mix"] <= 100_000, what="the source to drop to its cap")
    time.sleep(1)  # more passes: nothing else goes
    items_left = {item.id: item.t0 for item in store.items_of("mix")}
    records_left = {r.t for r in store.records(sources=["mix"])}
    deleted = [t for i, t in starts.items() if i not in items_left] + [t for t, _ in old + new if t not in records_left]
    survivors = list(items_left.values()) + list(records_left)
    assert any(t in records_left for t, _ in old) and any(i not in items_left for i in starts), "it reached both"
    assert max(deleted) < min(survivors), "what went is everything before one time"
    assert store.source_bytes()["mix"] > 100_000 - 8000, "and no more than the cap asks"


def test_max_days_deletes_what_ended_before_it(home, store, daemon):
    config(home, source("cam", "max_days = 1"), source("dead", "max_days = 1"), source("notes", "max_days = 1"))
    now = time.time()
    fill(store, "cam", 3, end=now - 3 * DAY)  # old
    fresh = fill(store, "cam", 2, end=now - 100)
    old = fill(store, "dead", 4, end=now - 5 * DAY)  # all old: the newest stays
    store.add_records("cam", [(now - 3 * DAY, '{"n": "old"}'), (now - 10, '{"n": "recent"}')])
    # versions: a current one never ages out, a closed one does
    kept = store.add_item("notes", "file", now - 6 * DAY, None, "files/notes/a", 1, name="a.md")
    closed = store.add_item("notes", "file", now - 6 * DAY, now - 5.5 * DAY, "files/notes/b", 1, name="b.md")
    newest = store.add_item("notes", "file", now - 4 * DAY, None, "files/notes/c", 1, name="c.md")
    daemon()
    wait_for(lambda: len(left(store, "dead")) == 1 and closed not in left(store, "notes"), what="the old to go")
    assert left(store, "cam") == fresh
    assert left(store, "dead") == [old[-1]]
    assert [r.data for r in store.records(sources=["cam"])] == ['{"n": "recent"}']
    assert left(store, "notes") == [kept, newest]


def test_a_cap_drops_records_oldest_first(home, store, daemon):
    config(home, source("agent", "max_gb = 0.000005"))  # 5 kB
    now = time.time()
    store.add_records("agent", [(now - 1000 + i, '{"n": %d, "pad": "%s"}' % (i, "p" * 80)) for i in range(200)])
    daemon()
    wait_for(lambda: store.source_bytes().get("agent", 0) <= 5000, what="records to drop to the cap")
    numbers = [int(r.data.split(",")[0][6:]) for r in store.records(sources=["agent"])]
    assert numbers == list(range(200 - len(numbers), 200)), "the newest records are what is left"
    assert 8 <= len(numbers) <= 11, "a record of about 100 bytes costs about 500 against the cap"


def test_a_record_costs_its_utf8_bytes_and_the_indexes(home, store):
    """The cost is the text's bytes (not characters) times 3, plus 200 a row: measured with FTS5 at steady state."""
    text = '{"text": "café → 日本語"}'
    assert len(text.encode("utf-8")) > len(text)
    store.add_record("agent", time.time(), text)
    assert store.source_bytes() == {"agent": 3 * len(text.encode("utf-8")) + 200}
    assert store.status()["agent"]["record_bytes"] == 3 * len(text.encode("utf-8")) + 200


def test_the_floor_is_reported_while_free_space_is_below_it(home, daemon):
    config(home, source("agent"), memory="free_gb = 1000000000")  # more than any disk
    daemon()
    status = wait_for(lambda: (s := mms_json(home, "status"))["running"] and s["floor"]["breached"] and s,
                      what="the floor to show in mms status")
    assert status["floor"]["free_gb"] > 0 and status["floor"]["floor_gb"] == 1e9
    assert "floor 1e+09 GB breached, capture and ingest paused" in daemon_log(home)


Usage = namedtuple("Usage", "total used free")


def fake_free(monkeypatch, gb: float) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, int(gb * 1e9)))


def writers(log: Path) -> list[str]:
    """Sources that write by themselves: a tailed file, and a command that prints a line every 0.1 s."""
    ticker = json.dumps([sys.executable, "-c", "import time\nwhile True:\n print('tick', flush=True)\n time.sleep(0.1)"])
    return [f'[sources.tailed]\nkind = "log"\npath = "{log.as_posix()}"\nrescan_s = 0.2\n',
            f'[sources.ticks]\nkind = "command"\ncommand = {ticker}\n', source("agent")]


def test_below_the_floor_no_writer_writes(home, store, daemon, tmp_path):
    """Ingest is paused with capture, and `remember` refuses, for a note and for a file."""
    log = tmp_path / "app.log"
    log.write_text('{"n": 1}\n{"n": 2}\n')
    config(home, *writers(log), memory="free_gb = 1000000000")  # more than any disk
    daemon()
    wait_for(lambda: (s := mms_json(home, "status"))["running"] and s["floor"]["breached"] and s, what="the floor")
    time.sleep(1.5)  # many loops
    assert store.records() == [], "neither the tail nor the command wrote"
    for args in (("a note",), ("--path", str(log))):
        refused = mms(home, "remember", *args, check=False)
        assert refused.returncode == 2 and "floor" in refused.stderr
    assert daemon_api(home).failure("remember", note="a note").code == "floor"
    assert store.records() == [] and store.items_of("agent") == []


def daemon_api(home: Path) -> Api:
    return Api(home)


def test_ingest_starts_again_when_the_floor_clears(home, store, monkeypatch, tmp_path):
    log = tmp_path / "app.log"
    log.write_text('{"n": 1}\n')
    config(home, *writers(log), memory="free_gb = 20")
    fake_free(monkeypatch, 5)
    daemon = Daemon(home)
    thread = threading.Thread(target=daemon.run)
    thread.start()
    try:
        time.sleep(1.2)
        assert store.records() == []
        fake_free(monkeypatch, 50)
        wait_for(lambda: store.records(sources=["tailed"]) and len(store.records(sources=["ticks"])) >= 2,
                 what="the tail and the command to write again")
        assert [r.data for r in store.records(sources=["tailed"])] == ['{"n": 1}']
    finally:
        (home / STOP).write_text("stop")
        thread.join(30)
    assert not thread.is_alive()


def test_the_floor_clears_one_gb_above_it(home, store, monkeypatch):
    config(home, source("agent"), memory="free_gb = 20")
    janitor = Janitor(store, configuration.load(home))
    fake_free(monkeypatch, 30)
    assert janitor.step(time.time()) is False
    fake_free(monkeypatch, 19.9)
    assert janitor.step(time.time()) is True
    fake_free(monkeypatch, 20.5)
    assert janitor.step(time.time()) is True, "still breached until 1 GB above the floor"
    fake_free(monkeypatch, 21.0)
    assert janitor.step(time.time()) is False
    assert janitor.free_gb == pytest.approx(21.0)


class Part:
    """A capture or ingest that records what the daemon does to it."""

    def __init__(self, calls: list, name: str):
        self.calls, self.name = calls, name

    def reconcile(self):
        self.calls.append(f"{self.name}.reconcile")

    def apply(self, config):
        self.calls.append(f"{self.name}.apply")

    def step(self, now):
        if f"{self.name}.step" not in self.calls:
            self.calls.append(f"{self.name}.step")

    def pause(self, flag):
        self.calls.append(f"{self.name}.pause({flag})")

    def stop(self):
        self.calls.append(f"{self.name}.stop")

    def status(self):
        return {"parts": self.name}


def test_the_daemon_passes_the_floor_to_capture(home, monkeypatch):
    config(home, source("agent"), memory="free_gb = 20")
    calls: list = []
    daemon = Daemon(home, capture=Part(calls, "capture"), ingest=Part(calls, "ingest"))
    fake_free(monkeypatch, 5)  # below the floor from the start
    thread = threading.Thread(target=daemon.run)
    thread.start()
    try:
        wait_for(lambda: "capture.step" in calls, what="the loop")
        assert calls[:2] == ["capture.reconcile", "capture.pause(True)"], "the janitor's pass comes before reconcile"
        fake_free(monkeypatch, 50)
        wait_for(lambda: "capture.pause(False)" in calls, what="capture to resume")
        fake_free(monkeypatch, 5)
        wait_for(lambda: calls.count("capture.pause(True)") == 2, what="capture to pause again")
    finally:
        (home / STOP).write_text("stop")
        thread.join(30)
    assert not thread.is_alive()
    assert calls[-2:] == ["capture.stop", "ingest.stop"], "capture stops first"
    assert not (home / STOP).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to unlink an open file")
def test_a_file_in_use_is_deleted_on_a_later_pass(home, store):
    config(home, source("cam", "max_gb = 0"))
    ids = fill(store, "cam", 3)
    janitor = Janitor(store, configuration.load(home))
    reader = open(store.path_of(store.item(ids[0])), "rb")
    try:
        janitor.step(time.time())
        assert ids[0] in left(store, "cam"), "its row stays while a reader has the file open"
        assert left(store, "cam") == [ids[0], ids[-1]], "the others went"
    finally:
        reader.close()
    janitor.step(time.time())
    assert left(store, "cam") == [ids[-1]], "retried once it was closed; the newest stays"


def test_a_programs_clock_rows_count_against_its_cap_and_go_with_drop_oldest(home, store, daemon):
    config(home, source("prog", 'max_gb = 0.000005, when_full = "drop_oldest"'))  # 5 kB
    now = time.time()
    with store.tx() as db:
        db.executemany("INSERT INTO clock (source, run, wall, t) VALUES ('prog', 'r', ?, ?)",
                       [(now - 2000 + i, float(i)) for i in range(100)])  # 100 rows: 10 kB
    assert store.source_bytes() == {"prog": 100 * 100}
    daemon()
    wait_for(lambda: store.source_bytes().get("prog", 0) <= 5000, what="the clock rows to drop to the cap")
    left = [row[0] for row in store.db().execute("SELECT wall FROM clock WHERE source = 'prog' ORDER BY wall")]
    assert 45 <= len(left) <= 50 and left == [now - 2000 + i for i in range(100 - len(left), 100)], \
        "the oldest rows went, by time, and nothing else"


def test_a_remembered_note_and_file_go_after_expires_unless_kept(home, store, served, tmp_path):
    config(home, source("agent"))
    api = served().api
    now = time.time()
    names = ("one", "two", "three")
    for name in names:
        (tmp_path / f"{name}.txt").write_text(f"file {name}\n")
    ids = {name: api.call("remember", path=str(tmp_path / f"{name}.txt"), expires="2s", name=name,
                          t=now - 5000 if name == "three" else None)["stored"][0]["id"] for name in names}
    api.call("remember", note="soon gone", expires="2s")
    api.call("remember", note="stays")
    api.call("keep", source="agent", since=now - 5001, until=now - 4999)  # only the third file
    files = {name: store.path_of(store.item(i)) for name, i in ids.items()}
    wait_for(lambda: store.item(ids["one"]) is None and store.item(ids["two"]) is None
             and [json.loads(r.data)["text"] for r in store.records(sources=["agent"])] == ["stays"],
             timeout=20, what="the expired note and files to go")
    assert store.item(ids["three"]) is not None and files["three"].exists(), "kept"
    assert not files["one"].exists() and not files["two"].exists(), "the files went with their rows"


def test_looks_files_go_after_a_day(home, daemon):
    config(home, source("agent"))
    looks = home / "looks"
    looks.mkdir()
    old, fresh = looks / "old.png", looks / "fresh.png"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    longago = time.time() - 2 * DAY
    os.utime(old, (longago, longago))
    bad = home / "spool" / "_bad"
    bad.mkdir(parents=True)
    (bad / "old.jsonl").write_bytes(b"x")
    os.utime(bad / "old.jsonl", (longago, longago))
    daemon()
    wait_for(lambda: not old.exists() and not (bad / "old.jsonl").exists(), what="the old files to go")
    assert fresh.exists()


def test_what_may_be_kept_is_limited_by_the_sources_cap(home, store, served):
    config(home, source("cam", "max_gb = 0.000025"), source("free"))  # 25 kB
    now = time.time()
    fill(store, "cam", 10, size=4000, end=now - 1000, step=60)  # 10 segments of 4 kB
    fill(store, "free", 10, size=4000, end=now - 1000, step=60)
    start = lambda n: now - 1000 - (10 - n) * 60 + 1  # noqa: E731  inside segment n
    assert store.set_keep("cam", start(0), start(4)) == 5  # 20 kB kept already, as an earlier day's marks
    api = served().api

    def kept() -> int:
        return len([i for i in store.items_of("cam") if i.keep])

    assert api.call("keep", source="cam", since=start(5), until=start(5)) == {"items": 1, "keep": True}  # 24 kB
    over = api.failure("label", text="too much", since=start(6), until=start(6), source="cam", keep=True)
    assert (over.status, over.code, over.field) == (413, "too_big", "keep")
    assert "24,000" in over.error["message"] and "4,000" in over.error["message"] and "25,000" in over.error["message"], \
        "the message gives the numbers"
    assert kept() == 6 and not [label for label in store.labels() if label.text == "too much"], "marks nothing, labels nothing"
    assert api.failure("keep", source="cam", since=start(6), until=start(8)).code == "too_big"
    assert api.call("keep", source="cam", since=start(0), until=start(9), clear=True) == {"items": 10, "keep": False}
    assert kept() == 0, "clearing is never refused"
    assert api.call("label", text="five", since=start(0), until=start(4), source="cam", keep=True)["kept"] == 5
    assert api.call("keep", source="free", since=start(0), until=start(9))["items"] == 10, "no max_gb, no limit"
    assert api.call("label", text="all of it", since=start(0), until=start(9), source="free", keep=True)["kept"] == 10


def test_the_stray_files_a_crash_left_go_by_a_rule_and_a_copy_under_way_stays(home, daemon):
    config(home, source("agent"))
    long_ago = time.time() - 7200
    strays = [home / "streams" / "cam" / "run" / "000.mkv.part", home / "streams" / "cam" / "run" / "000.ffconcat",
              home / "streams" / "cam" / "run" / "000.err", home / "files" / "agent" / ".1-2-name.part",
              home / "spool" / "log" / "run" / "00000001.jsonl.part"]
    fresh = [home / "streams" / "cam" / "run" / "001.ffconcat", home / "files" / "agent" / ".3-4-new.part"]
    kept = [home / "streams" / "cam" / "run" / "000.mkv", home / "files" / "agent" / "file.txt.zst"]
    for file in (*strays, *fresh, *kept):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"x")
    for file in strays + kept[:1]:
        os.utime(file, (long_ago, long_ago))
    daemon()
    wait_for(lambda: not any(file.exists() for file in strays), what="the stray files to go")
    assert all(file.exists() for file in fresh), "a copy or an encode under way is not a stray"
    assert all(file.exists() for file in kept), "only the three suffixes are strays, however old the file"


def test_a_stored_file_is_not_taken_for_a_stray_by_its_name_alone(home, store, daemon):
    """A stored name ends with the owner's own file's name, which may end `.part`, `.err` or `.ffconcat`: under `files/` only the
    writer's own dot-prefixed `.part` is a stray, and a stored file goes by its source's caps and ages, with its row."""
    config(home, source("agent"))
    long_ago = time.time() - 7200
    folder = home / "files" / "agent"
    folder.mkdir(parents=True)
    stored = [folder / f"20261101T000000Z-0123abcd-{name}" for name in ("download.part", "build.err", "concat.ffconcat")]
    stray = folder / ".5-6-download.part"
    dotted = folder / ".5-6-build.err"  # the writer never makes this one
    for file in (*stored, stray, dotted):
        file.write_bytes(b"x")
        os.utime(file, (long_ago, long_ago))
    for file in stored:
        store.add_item("agent", "file", long_ago, None, store.relative(file), 1, name=file.name.split("-", 2)[2])
    daemon()
    wait_for(lambda: not stray.exists(), what="the writer's partial copy to go")
    assert all(file.exists() for file in (*stored, dotted)), "only a dot-prefixed .part is a stray under files/"
    assert len(store.items_of("agent")) == 3 and all((home / item.path).exists() for item in store.items_of("agent"))


def test_a_label_goes_with_its_item_and_a_label_of_no_item_stays(home, store, daemon):
    config(home, source("cam", "max_gb = 0.000004"))  # 4 kB
    ids = fill(store, "cam", 10, size=1000, end=time.time() - 100, step=60)
    on_old = store.label(0, 1, "on the oldest", "ask", source="cam", item=ids[0])
    on_new = store.label(0, 1, "on the newest", "ask", source="cam", item=ids[-1])
    note = store.label(0, 1, "no item: a note", "agent", source="cam")
    elsewhere = store.label(0, 1, "no source", "agent")
    daemon()
    wait_for(lambda: ids[0] not in left(store, "cam"), what="the oldest segment to be dropped by the cap")
    labels = {label.id for label in store.labels(-1, 2)}
    assert on_old not in labels, "the label of a deleted item went with it"
    assert {on_new, note, elsewhere} <= labels, "the newest stays, and a label of no item is a note"


def test_two_sessions_cannot_together_keep_more_than_max_gb(home, store, served):
    """Each keep passes the limit alone and the two together do not: the check and the mark are one transaction, so
    exactly one of a pair wins."""
    import threading
    config(home, *[source(f"s{i}", "max_gb = 0.000005") for i in range(25)])  # 5 kB each
    now = time.time()
    for i in range(25):
        fill(store, f"s{i}", 2, size=4000, end=now - 1000, step=60)
    api = served().api
    outcomes = []

    def keep(name: str, n: int, barrier: threading.Barrier) -> None:
        barrier.wait()
        try:
            api.call("keep", source=name, since=now - 1000 - (2 - n) * 60 + 1, until=now - 1000 - (2 - n) * 60 + 2)
            outcomes.append((name, True))
        except Exception as error:
            outcomes.append((name, getattr(error, "code", repr(error))))

    for i in range(25):
        barrier = threading.Barrier(2)
        pair = [threading.Thread(target=keep, args=(f"s{i}", n, barrier)) for n in (0, 1)]
        for thread in pair:
            thread.start()
        for thread in pair:
            thread.join()
    for i in range(25):
        kept = sum(item.bytes for item in store.items_of(f"s{i}") if item.keep)
        assert kept <= 5000, f"s{i} keeps {kept} bytes"
    assert sorted(set(code for _, code in outcomes if code is not True)) == ["too_big"]
    assert sum(1 for _, ok in outcomes if ok is True) == 25
