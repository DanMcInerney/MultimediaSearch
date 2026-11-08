"""The `mms` command: the daemon's lock and stop, status, find, look, label, keep, and scripts, with the project of its folder."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import DECIDER, env, mms, mms_json, wait_for, write_config
from test_project import make_repo

from mms import project
from mms.client import connect, is_running
from mms.store import Store

HOUR = 3600


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def running(home: Path) -> bool:
    return mms_json(home, "status")["running"]


def test_start_refuses_a_second_daemon_and_stop_cleans_up(home, daemon):
    write_config(home, '[memory]\nloop_s = 0.1\n[sources.agent]\nkind = "push"\n')
    first = daemon()
    status = mms_json(home, "status")
    assert status["running"] and status["pid"] > 0 and status["floor"]["breached"] is False
    assert [s["name"] for s in status["sources"]] == ["agent"] and status["sources"][0]["state"] == "idle"

    second = mms(home, "start", check=False)
    assert second.returncode == 1 and "already running" in second.stderr and f"pid {status['pid']}" in second.stderr
    assert first.process.poll() is None, "the first daemon is untouched"

    stopped = mms(home, "stop")
    assert "stopped" in stopped.stdout
    assert first.process.wait(10) == 0
    assert not (home / "daemon.stop").exists() and not (home / "daemon.json").exists()
    assert "mms daemon stopped" in first.text()
    assert running(home) is False
    assert "no daemon is running" in mms(home, "stop").stdout
    assert not (home / "daemon.stop").exists(), "a stop with no daemon leaves nothing to stop the next one"

    again = daemon()  # the home starts again after a clean stop
    assert running(home)
    again.stop()


def test_probes_never_cost_a_starting_daemon_its_lock(home, daemon):
    """`mms status` and `mms stop` take the lock to look: that must not make a daemon that is starting give up."""
    write_config(home, "[memory]\nloop_s = 0.1\n")
    errors: list[str] = []
    for _ in range(5):
        one = daemon(ready=False)
        end = time.monotonic() + 30

        def hammer():  # from the launch until the daemon has the lock and its API is up
            while time.monotonic() < end and one.process.poll() is None and not (home / "daemon.json").exists():
                try:
                    is_running(home)
                    connect(home)  # which also takes the lock, and removes a daemon.json no daemon holds
                except Exception as error:
                    errors.append(repr(error))

        probes = [threading.Thread(target=hammer) for _ in range(3)]
        for probe in probes:
            probe.start()
        for probe in probes:
            probe.join()
        one.wait_ready()
        assert one.process.poll() is None, f"a daemon exited at start:\n{one.text()}"
        assert not errors, errors[:3]
        assert is_running(home)
        one.stop()
        assert not is_running(home)


def test_a_stale_stop_file_does_not_stop_the_next_daemon(home, daemon):
    write_config(home, "[memory]\nloop_s = 0.1\n")
    (home / "daemon.stop").write_text("stop")
    one = daemon()
    time.sleep(0.5)
    assert one.process.poll() is None
    one.stop()


def test_ctrl_c_stops_the_daemon_cleanly(home):
    write_config(home, "[memory]\nloop_s = 0.1\n")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with open(home / "out.log", "wb") as out:
        process = subprocess.Popen([sys.executable, "-m", "mms.cli", "start"], env=env(home), stdout=out,
                                   stderr=subprocess.STDOUT, creationflags=flags)
        try:
            wait_for(lambda: running(home), what="the daemon",
                     detail=lambda: (home / "out.log").read_text(encoding="utf-8", errors="replace"))
            time.sleep(0.3)  # the handlers are in by the first loop
            process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            assert process.wait(30) == 0
        finally:
            if process.poll() is None:
                process.kill()
    assert "mms daemon stopped" in (home / "out.log").read_text(encoding="utf-8")
    assert not (home / "daemon.json").exists()


def test_an_index_of_another_version_is_refused_with_a_message(home):
    import sqlite3
    db = sqlite3.connect(home / "index.sqlite")
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
    db.commit()
    db.close()
    for args in (("status",), ("find",), ("start",)):
        done = mms(home, *args, check=False)
        assert done.returncode == 2 and "another version" in done.stderr and "Traceback" not in done.stderr, args


def test_a_bad_config_without_a_daemon_is_a_clear_error(home):
    write_config(home, '[sources.x]\nkind = "telepathy"\n', approve=False)
    done = mms(home, "status", check=False)
    assert done.returncode == 2 and "telepathy" in done.stderr and "Traceback" not in done.stderr


OPEN_AT = """
import sys, threading, time
from mms.store import Store
home, start = sys.argv[1], float(sys.argv[2])
time.sleep(max(0.0, start - time.time()))
errors = []
def open_it():
    try:
        Store(home).close()
    except Exception as error:
        errors.append(repr(error))
threads = [threading.Thread(target=open_it) for _ in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(*errors)
sys.exit(1 if errors else 0)
"""


def test_a_new_home_opens_from_many_processes_and_threads_at_once(tmp_path):
    """Switching a new index to WAL needs an exclusive lock the busy timeout does not wait for: `Store` serializes it."""
    for n in range(5):
        home = tmp_path / f"fresh{n}"
        start = time.time() + 1.0  # every process opens at the same moment
        processes = [subprocess.Popen([sys.executable, "-c", OPEN_AT, str(home), str(start)], env=env(home),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(4)]
        for process in processes:
            out, _ = process.communicate(timeout=60)
            assert process.returncode == 0, out
        store = Store(home)
        try:
            tables = {row[0] for row in store.db().execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            assert {"items", "labels", "clock", "projects", "sessions"} <= tables
            assert store.db().execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert store.db().execute("PRAGMA user_version").fetchone()[0] == 2
        finally:
            store.close()


def test_status_without_a_daemon_shows_what_the_store_and_config_hold(home, store):
    write_config(home, '[sources.agent]\nkind = "push"\nkeep = { max_gb = 1, when_full = "thin", max_days = 30 }\n'
                       '[sources.mic]\nkind = "microphone"\ndevice = "Some Mic"\n', approve=False)
    store.add_records("agent", [(time.time(), '{"text": "hello"}')])
    status = mms_json(home, "status")
    assert status["running"] is False and status["pid"] is None and status["started"] is None
    assert status["decider"] == {"model": None, "state": None, "error": None, "waiting": None}
    assert status["jobs"] == {"running": None, "queued": None}
    by_name = {s["name"]: s for s in status["sources"]}
    assert by_name["agent"]["records"] == 1 and by_name["agent"]["keep"] == {"max_gb": 1, "when_full": "thin", "max_days": 30}
    assert by_name["mic"]["state"] == "needs_approval" and by_name["mic"]["approve"] == "mms approve mic"
    table = mms(home, "status").stdout
    assert "not running" in table and "agent" in table and "1 GB thin 30 d" in table and "run: mms approve mic" in table


def populate(store: Store, now: float) -> dict:
    """A little of everything, hours apart, newest last."""
    ids = {}
    ids["old_segment"] = store.add_item("cam", "video", now - 5 * HOUR, now - 5 * HOUR + 60, "streams/cam/a.mkv", 10)
    ids["new_segment"] = store.add_item("cam", "video", now - HOUR, now - HOUR + 60, "streams/cam/b.mkv", 10)
    ids["note"] = store.add_item("notes", "file", now - 3 * HOUR, None, "files/notes/x", 5, name="plan.md")
    ids["old_record"] = store.add_record("log", now - 4 * HOUR, '{"line": "disk full on alpha"}')
    ids["new_record"] = store.add_record("log", now - 2 * HOUR, '{"line": "all well on beta"}')
    ids["label"] = store.label(now - 4.5 * HOUR, now - 3.5 * HOUR, "saw alpha crash", "ask", source="cam")
    return ids


def test_find_selects_by_time_source_and_text(home, store, memory):
    now = time.time()
    ids = populate(store, now)

    everything = mms_json(home, "find")["results"]
    assert [(e["type"], e["id"]) for e in everything] == [  # newest first, by start
        ("item", ids["new_segment"]), ("record", ids["new_record"]), ("item", ids["note"]),
        ("record", ids["old_record"]), ("label", ids["label"]), ("item", ids["old_segment"])]

    def ids_of(*args):
        return {(e["type"], e["id"]) for e in mms_json(home, "find", *args)["results"]}

    assert ids_of("--since", "2.5h") == {("item", ids["new_segment"]), ("record", ids["new_record"])}
    assert ids_of("--until", "3.8h") == {("item", ids["old_segment"]), ("record", ids["old_record"]),
                                         ("label", ids["label"])}
    window = ids_of("--since", "4.2h", "--until", "2.5h")
    assert window == {("record", ids["old_record"]), ("item", ids["note"]), ("label", ids["label"])}
    assert {t for t, _ in ids_of("--source", "log")} == {"record"}
    assert {t for t, _ in ids_of("--source", "log", "--source", "notes")} == {"record", "item"}
    assert ids_of("--text", "alpha") == {("record", ids["old_record"]), ("label", ids["label"])}
    assert ids_of("--text", "alpha", "--source", "log") == {("record", ids["old_record"])}
    assert ids_of("--text", "plan") == {("item", ids["note"])}, "a version is found by its name"
    assert ids_of("--text", "alpha", "--type", "label") == {("label", ids["label"])}
    page = mms_json(home, "find", "--limit", "2")
    assert len(page["results"]) == 2 and page["next"]
    rest = mms_json(home, "find", "--limit", "9", "--after", page["next"])
    assert [(e["type"], e["id"]) for e in page["results"] + rest["results"]] == [(e["type"], e["id"]) for e in everything]
    assert rest["next"] is None and "--after" in mms(home, "find", "--limit", "2").stdout
    assert ids_of("--text", "nothing-like-this") == set()
    assert mms(home, "find", "--since", "tomorrow-ish", check=False).returncode == 2
    rows = mms_json(home, "find", "--source", "log")["results"]
    assert rows[0]["look"] == {"source": "log", "t": rows[0]["t0"]}, "a handle for look"

    lines = mms(home, "find", "--source", "log").stdout.splitlines()
    assert len(lines) == 2 and "beta" in lines[0] and "alpha" in lines[1] and lines[0].startswith("20")


def test_look_prints_the_path_of_the_file_it_writes_or_the_text(home, store, tmp_path, memory):
    source = tmp_path / "plan.md"
    source.write_bytes(b"# the plan\n")
    mms(home, "remember", "--path", str(source), "--name", "plan.md", "--t", "2h")
    (item,) = store.items_of("agent")
    assert mms(home, "look", str(item.id)).stdout == "# the plan\n\n"
    shown = mms_json(home, "look", str(item.id))
    assert shown["kind"] == "text" and shown["item"] == item.id and shown["text"] == "# the plan\n"
    assert mms(home, "look", "9999", check=False).returncode == 2
    assert mms(home, "look", check=False).returncode == 2
    from PIL import Image
    target = store.streams / "cam" / "x.png"
    target.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (200, 10, 10)).save(target)
    picture = store.add_version("pics", "x.png", time.time() - 100, src=target)
    printed = Path(mms(home, "look", str(picture)).stdout.strip())
    assert printed.parent == home / "looks" and Image.open(printed).size == (32, 24)


def test_look_over_a_range_prints_the_path_of_the_sheet(home, store, memory):
    from PIL import Image
    for name, colour, age in (("a.png", (200, 10, 10), 1800), ("b.png", (10, 10, 200), 600)):
        target = store.streams / "cam" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), colour).save(target)
        store.add_version("pics", name, time.time() - age, src=target)
    printed = Path(mms(home, "look", "--source", "pics", "--since", "1h", "--until", "now").stdout.strip())
    assert printed.parent == home / "looks" and printed.suffix == ".png" and Image.open(printed).size == (1568, 588)
    shown = mms_json(home, "look", "--source", "pics", "--since", "1h", "--until", "now")
    assert shown["kind"] == "sheet" and [tile["n"] for tile in shown["tiles"]] == [1, 2] and Path(shown["image"]["path"]).is_file()
    assert mms(home, "look", "--source", "pics", "--since", "1h", check=False).returncode == 2, "until is needed"
    assert mms(home, "look", "--source", "pics", "--since", "1h", "--until", "now", "--t", "now", check=False).returncode == 2


def test_look_at_a_file_that_is_none_of_those_prints_the_path_of_a_copy(home, store, tmp_path, memory):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
    mms(home, "remember", "--path", str(source), "--name", "report.pdf", "--t", "2h")
    (item,) = store.items_of("agent")
    ran = mms(home, "look", str(item.id))
    copy = Path(ran.stdout.strip())
    assert copy.parent == home / "looks" and copy.name.endswith("-report.pdf") and copy.read_bytes() == source.read_bytes()
    assert "report.pdf" in ran.stderr and "application/pdf" in ran.stderr
    shown = mms_json(home, "look", str(item.id))
    assert shown["kind"] == "file" and shown["file"]["name"] == "report.pdf" and shown["file"]["media"] == "application/pdf"
    assert shown["file"]["bytes"] == source.stat().st_size and Path(shown["file"]["path"]).parent == home / "looks"


def test_label_with_keep_marks_what_it_overlaps(home, store, memory):
    now = time.time()
    ids = populate(store, now)
    made = mms_json(home, "label", "the crash", "--since", "5.5h", "--until", "4.5h", "--source", "cam", "--keep")
    assert made["label"] and made["kept"] == 1
    keep = {item.id: item.keep for item in store.items_of("cam") + store.items_of("notes")}
    assert keep == {ids["old_segment"]: 1, ids["new_segment"]: 0, ids["note"]: 0}, "only the overlapped item, of cam"
    labels = [e for e in mms_json(home, "find", "--text", "crash")["results"] if e["type"] == "label"]
    assert {e["text"] for e in labels} == {"saw alpha crash", "the crash"}
    assert next(e for e in labels if e["text"] == "the crash")["by"] == "agent"

    mms(home, "label", "just a note", "--since", "1h")  # no --keep: nothing marked
    assert [i.keep for i in store.items_of("cam")] == [1, 0]
    assert mms(home, "label", "backwards", "--since", "1h", "--until", "2h", check=False).returncode == 2


def test_keep_needs_a_source_and_can_be_cleared(home, store, memory):
    now = time.time()
    ids = populate(store, now)
    refused = mms(home, "label", "the crash", "--since", "6h", "--keep", check=False)
    assert refused.returncode == 2 and "source" in refused.stderr
    assert [i.keep for i in store.items_of("cam") + store.items_of("notes")] == [0, 0, 0], "nothing was marked"
    assert [e for e in mms_json(home, "find", "--text", "crash")["results"] if e["text"] == "the crash"] == [], "and no label made"
    assert mms(home, "keep", "--since", "6h", check=False).returncode == 2, "mms keep names its source too"

    kept = mms_json(home, "keep", "--source", "cam", "--since", "6h", "--until", "30m")
    assert kept == {"items": 2, "keep": True}
    assert [i.keep for i in store.items_of("cam")] == [1, 1] and store.item(ids["note"]).keep == 0

    cleared = mms_json(home, "keep", "--source", "cam", "--since", "5.5h", "--until", "4.5h", "--clear")
    assert cleared == {"items": 1, "keep": False}
    assert [i.keep for i in store.items_of("cam")] == [0, 1], "only the range's item let go"
    assert "let go 0 items" in mms(home, "keep", "--source", "cam", "--since", "3h", "--until", "2h", "--clear").stdout


def test_remember_prints_what_it_stored(home, tmp_path, memory):
    assert mms_json(home, "remember", "the build needs python 3.12")["stored"][0]["type"] == "record"
    picture = tmp_path / "x.png"
    picture.write_bytes(b"\x89PNG not really")
    assert mms(home, "remember", "--path", str(picture)).stdout.startswith("item ")
    assert "unchanged" in mms(home, "remember", "--path", str(picture)).stdout
    assert mms(home, "remember", check=False).returncode == 2, "a note or --path is needed"
    assert mms(home, "remember", "a", "--path", str(picture), check=False).returncode == 2


def test_run_and_ask_start_a_job_through_the_daemon_starting_it_when_none_runs(home):
    scripts = home / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "echo.py").write_text(
        "import argparse\n"
        "def cli(argv):\n    p = argparse.ArgumentParser(prog='echo')\n    p.add_argument('word')\n    p.add_argument('--times', type=int, default=1)\n"
        "    return vars(p.parse_args(argv))\n"
        "def run(job):\n    job.update(said=job.params['word'] * job.params['times'], project=job.caller.project)\n"
        "def note(result):\n    return 'echoed ' + result['said']\n", encoding="utf-8")
    (scripts / "noflags.py").write_text("def run(job):\n    pass\n", encoding="utf-8")
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + DECIDER)
    Store(home).add_record("thermo", time.time() - 60, json.dumps({"note": "the kettle is on"}))
    try:
        done = mms(home, "run", "echo", "ab", "--times", "2")  # no daemon runs: it is started
        assert json.loads(done.stdout) == {"said": "abab", "project": None} and "echoed abab" in done.stderr
        repo = make_repo(home.parent / "work" / "thing")
        there = mms(home, "run", "echo", "x", cwd=repo)
        assert json.loads(there.stdout)["project"] == project.find(repo).id, "a job carries its folder's project"
        assert mms_json(home, "status")["running"] is True, "and left running"
        asked = mms(home, "ask", "contains:kettle", "--since", "2h", "--source", "thermo", "--limit", "1")
        result = json.loads(asked.stdout)
        assert len(result["sightings"]) == 1 and result["sightings"][0]["source"] == "thermo"
        assert "scanned 1 windows" in asked.stderr
        assert mms(home, "ask", "q", "--limit", "99", check=False).returncode == 2
        bad = mms(home, "ask", "--since", "2h", check=False)
        assert bad.returncode == 2 and "bad_request" in bad.stderr
        assert mms(home, "run", "nope", check=False).returncode == 2
        missing = mms(home, "run", "nope", check=False)
        assert "no script 'nope'" in missing.stderr and "echo" in missing.stderr
        assert mms(home, "run", "noflags", check=False).returncode == 2
        assert mms(home, "run", check=False).returncode == 2
        assert mms(home, "run", "echo", "--bogus", check=False).returncode == 2
    finally:
        mms(home, "stop", check=False)


def test_the_clis_writes_and_reads_are_of_the_project_of_its_folder(home, tmp_path, memory):
    repo, plain = make_repo(tmp_path / "work" / "thing"), tmp_path / "plain"
    plain.mkdir()
    mms(home, "remember", "the kettle in the repo", cwd=repo)
    mms(home, "remember", "the kettle in no project", cwd=plain)

    def seen(*args, cwd):
        return {row["text"]: row["project"] for row in mms_json(home, "find", "--text", "kettle", *args, cwd=cwd)["results"]}

    assert seen(cwd=repo) == {'{"text": "the kettle in the repo"}': "thing", '{"text": "the kettle in no project"}': None}
    assert seen(cwd=plain) == {'{"text": "the kettle in no project"}': None}, "another folder reads the ambient rows only"
    assert len(seen("--scope", "all", cwd=plain)) == 2
    assert mms_json(home, "find", "--text", "kettle", cwd=repo)["searched"]["project"] == {
        "id": project.find(repo).id, "name": "thing"}
    row = mms_json(home, "find", "--text", "repo", "--scope", "all", cwd=plain)["results"][0]
    shown = mms_json(home, "look", "--source", row["source"], "--t", row["t0"], cwd=plain)
    assert "in the repo" in shown["text"], "a handle shows what it names, whatever its project"
    assert mms(home, "look", "--source", "agent", "--t", row["t0"], "--scope", "all", check=False).returncode == 2, \
        "look takes no scope"
    marked = tmp_path / "marked"
    marked.mkdir()
    (marked / ".mms-project").write_text("not valid!\n")
    broken = mms(home, "find", cwd=marked, check=False)
    assert broken.returncode == 2 and "mms-project" in broken.stderr and "Traceback" not in broken.stderr
    assert mms(home, "status", cwd=marked, check=False).returncode == 0, "status needs no project"


def test_the_cli_is_not_a_second_writer_remember_label_and_keep_go_through_the_daemon(home, detached):
    from mms import client

    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    assert client.connect(home) is None and not (home / "index.sqlite").exists()
    assert mms_json(home, "remember", "the kettle is on")["stored"]  # no daemon runs: this one starts it
    started = client.connect(home)
    assert started is not None and started.pid != os.getpid(), "the daemon the CLI started has the store"
    assert mms_json(home, "label", "saw it", "--since", "1h")["label"]
    assert mms_json(home, "keep", "--source", "agent", "--since", "1h")["items"] == 0
    assert client.connect(home) == started, "the same daemon took all three"
    assert [row["type"] for row in mms_json(home, "find", "--text", "kettle")["results"]] == ["record"]
