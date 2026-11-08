"""The daemon's API (TESTBED.md item 6): `daemon.json`, the token, each op's shape and each documented error, and
`mms start --detach`."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from conftest import Api, env, fake_ffmpeg, mms, wait_for, write_config

from mms import ops
from mms.store import Store

HOUR = 3600


def test_daemon_json_appears_once_the_api_answers_and_goes_at_a_clean_stop(home, daemon):
    write_config(home, "[memory]\nloop_s = 0.1\n")
    one = daemon(ready=False)
    info = wait_for(lambda: (home / "daemon.json").exists() and Api(home).info(), what="daemon.json", detail=one.text)
    assert set(info) == {"api", "pid", "port", "token", "started"} and info["api"] == 1
    assert re.fullmatch(r"[0-9a-f]{64}", info["token"]) and info["pid"] > 0 and info["port"] > 0
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d", info["started"])
    assert one.api.call("status")["running"] is True, "the file is there only once the API answers"
    assert not list(home.glob("daemon.json.part"))
    assert one.stop() == 0
    assert not (home / "daemon.json").exists()


def test_a_call_without_the_token_is_unauthorized(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\n")
    api = served().api
    for token in (None, "wrong", "0" * 64):
        status, body = api.raw("status", {}, token=token)
        assert status == 401 and body["error"]["code"] == "unauthorized" and body["error"]["hint"]
    assert api.call("status")["running"] is True


def test_there_is_no_approve_op_and_only_post_to_v1_is_served(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\n")
    api = served().api
    for op in ("approve", "consent", "jobs", "nothing/here"):
        status, body = api.raw(op, {})
        assert status == 404 and body["error"]["code"] == "not_found", op
    status, body = api.raw("status", {}, method="GET")
    assert status == 404 and body["error"]["code"] == "not_found"


def test_status_gives_its_documented_shape(home, served, tmp_path):
    write_config(home, "[memory]\nloop_s = 0.1\n" + f'[sources.notes]\nkind = "folder"\npath = "{tmp_path.as_posix()}"\n'
                 'keep = { max_gb = 5, when_full = "thin" }\n[sources.mic]\nkind = "microphone"\ndevice = "Some Mic"\n',
                 approve=False)
    status = served().api.call("status")
    assert set(status) == {"home", "running", "pid", "started", "config_error", "floor", "decider", "jobs",
                           "sources"}
    assert status["running"] is True and status["config_error"] is None
    assert set(status["floor"]) == {"breached", "free_gb", "floor_gb"} and status["floor"]["breached"] is False
    assert status["decider"] == {"model": None, "state": "unloaded", "error": None, "waiting": 0}
    assert status["jobs"] == {"running": 0, "queued": 0}
    by_name = {s["name"]: s for s in status["sources"]}
    assert by_name["notes"]["kind"] == "folder" and by_name["notes"]["keep"] == {
        "max_gb": 5, "when_full": "thin", "max_days": None}
    assert {"state", "project", "bytes", "items", "records", "newest"} <= set(by_name["notes"])
    assert by_name["mic"]["state"] == "needs_approval" and by_name["mic"]["approve"] == "mms approve mic"
    assert "devices" not in status


def test_every_error_comes_back_with_its_code_and_field(home, served, tmp_path, monkeypatch):
    write_config(home, "[memory]\nloop_s = 0.1\nremember_max_mb = 0.001\n"
                       '[sources.cam]\nkind = "ffmpeg"\ninput = ["-i", "x"]\npreset = "camera"\n'
                       '[sources.capped]\nkind = "push"\nkeep = { max_gb = 0.000001 }\n')
    api = served().api
    store = Store(home)
    assert (e := api.failure("find", nope=1)).status == 400 and (e.code, e.field) == ("bad_request", "nope")
    assert (e := api.failure("find", limit=500)).code == "bad_request" and e.field == "limit"
    assert (e := api.failure("find", since="yesterday-ish")).field == "since"
    assert (e := api.failure("find", caller={"session": "x"})).field == "caller"
    assert (e := api.failure("nothing")).status == 404 and (e.code, e.field) == ("not_found", "op")
    assert (e := api.failure("look", item=999)).status == 404 and e.field == "item"
    assert (e := api.failure("jobs/get", job="j-00000000")).status == 404 and e.field == "job"
    assert (e := api.failure("jobs/start", script="nope")).status == 404 and e.field == "script"
    assert (e := api.failure("source_set", name="cam", kind="screen", max_gb=1)).status == 409
    assert (e.code, e.field) == ("wrong_kind", "name")
    store.add_record("capped", time.time(), "x" * 10000)
    assert (e := api.failure("source_set", name="capped", kind="program", max_gb=1)).code == "wrong_kind"
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 5000)
    assert (e := api.failure("remember", path=str(big))).status == 413 and (e.code, e.field) == ("too_big", "path")
    assert (e := api.failure("remember", note="x", source="cam")).code == "wrong_kind"
    job = api.call("jobs/start", script="ask", params={"question": "Is it red?", "since": "1h"})
    assert job["state"] == "failed" and job["error"]["code"] == "decider", "no model is configured"

    def boom(env, caller, body):
        raise RuntimeError("a bug")

    monkeypatch.setitem(ops.OPS, "find", boom)
    assert (e := api.failure("find")).status == 500 and e.code == "internal"
    store.close()


def test_each_op_gives_its_shape(home, served, tmp_path):
    write_config(home, f'[memory]\nloop_s = 0.1\nffmpeg = "{fake_ffmpeg(tmp_path)}"\n[sources.agent]\nkind = "push"\n')
    api = served().api
    now = time.time()
    note = api.call("remember", note="the kettle is on", t=now - 60, name="kettle")
    assert note["source"] == "agent" and note["unchanged"] == [] and note["expires"] is None
    assert "time_from" not in note and note["stored"][0]["type"] == "record" and "t0" in note["stored"][0]
    made = api.call("label", text="saw the kettle", since=now - 120, until=now - 30, source="agent")
    assert set(made) == {"label", "kept"} and made["kept"] == 0
    kept = api.call("keep", source="agent", since=now - 3600)
    assert kept == {"items": 0, "keep": True}
    found = api.call("find", text="kettle")
    assert set(found) == {"results", "next", "searched"} and found["next"] is None
    assert {r["type"] for r in found["results"]} == {"record", "label"}
    searched = found["searched"]
    assert set(searched) == {"now", "since", "until", "scope", "project", "sources"}
    assert searched["scope"] == "default" and searched["project"] is None and "agent" in searched["sources"]
    record = next(r for r in found["results"] if r["type"] == "record")
    assert record["source"] == "agent" and "kettle" in record["text"] and record["look"]["source"] == "agent"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d", record["t0"])
    shown = api.call("look", **record["look"])
    assert shown["kind"] == "text" and "kettle" in shown["text"] and shown["more"] is False
    assert shown["labels"][0]["text"] == "saw the kettle" and shown["item"] is None
    assert api.call("find", text="kettle", types=["label"])["results"][0]["by"] == "agent"
    assert isinstance(api.call("find", limit=1)["next"], str), "a cursor, while there is more"
    status = api.call("status", devices=True)
    assert set(status["devices"]) >= {"displays", "microphones", "cameras"}
    if os.name == "nt":
        assert status["devices"]["microphones"] == [{"device": "Fake Mic"}]
        assert status["devices"]["cameras"] == [{"device": "Fake Cam"}]


def test_mms_start_detach_returns_once_the_daemon_answers_and_it_outlives_the_shell(home):
    write_config(home, "[memory]\nloop_s = 0.1\n")
    # as the adapter runs it: MMS_HOME in the environment, no --home, its output to a file
    with open(home.parent / "adapter-start.log", "wb") as out:
        started = subprocess.run([sys.executable, "-m", "mms.cli", "start", "--detach"], env=env(home),
                                 stdout=out, stderr=subprocess.STDOUT, timeout=90)
    started.stderr = (home.parent / "adapter-start.log").read_text(encoding="utf-8", errors="replace")
    try:
        assert started.returncode == 0, started.stderr
        api = Api(home)
        assert api.call("status")["running"] is True, "it answers after the command that started it has gone"
        again = subprocess.run([sys.executable, "-m", "mms.cli", "--home", str(home), "start", "--detach"],
                               env=env(home), capture_output=True, text=True, timeout=90)
        assert again.returncode == 0 and f"pid {api.info()['pid']}" in again.stdout, "a running daemon counts"
        assert (home / "daemon.log").exists() and "mms daemon started" in (home / "daemon.log").read_text(encoding="utf-8")
        assert f"running (pid {api.info()['pid']})" in mms(home, "status").stdout
    finally:
        assert "stopped" in mms(home, "stop").stdout
    assert not (home / "daemon.json").exists()
    assert "mms daemon stopped" in (home / "daemon.log").read_text(encoding="utf-8")


def test_find_pages_by_a_cursor_through_ties_and_sub_millisecond_times_without_a_gap_or_a_repeat(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    store = Store(home)
    base = 1_700_000_000.0
    store.add_records("log", [(base, json.dumps({"n": i})) for i in range(7)])  # seven rows at one time
    store.add_records("log", [(base + 1 + i * 1e-5, json.dumps({"n": 100 + i})) for i in range(5)])  # ten microseconds apart
    for i in range(3):
        store.label(base, base + 5, f"label {i}", "agent", source="log")
        store.add_item("log", "video", base, base + 60, f"streams/log/{i}.mkv", 10)
    api = served().api
    whole = api.call("find", since=base - 1, until=base + 10, limit=200)
    assert whole["next"] is None and len(whole["results"]) == 7 + 5 + 3 + 3
    seen, after, pages = [], None, 0
    while True:
        page = api.call("find", since=base - 1, until=base + 10, limit=2, **({"after": after} if after else {}))
        seen += [(r["type"], r["id"]) for r in page["results"]]
        pages += 1
        after = page["next"]
        if after is None:
            break
        assert isinstance(after, str) and len(page["results"]) == 2
    assert pages == 9 and seen == [(r["type"], r["id"]) for r in whole["results"]], "every row once, in the same order"
    assert len(set(seen)) == len(seen)
    assert [t for _, t in [(r["id"], r["t0"]) for r in whole["results"]]] == sorted(
        (r["t0"] for r in whole["results"]), reverse=True), "newest first"
    error = api.failure("find", after="not a cursor")
    assert (error.status, error.code, error.field) == (400, "bad_request", "after")
    assert api.failure("find", after=5).field == "after"


def test_a_find_text_without_words_is_a_bad_request_not_a_500_and_a_blank_one_is_no_text(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    store = Store(home)
    store.add_record("log", time.time() - 5, json.dumps({"line": "the kettle is on"}))
    api = served().api
    for text in ('"', '""', "*", "()", "---", "!!"):
        error = api.failure("find", text=text)
        assert (error.status, error.code, error.field) == (400, "bad_request", "text"), text
    assert len(api.call("find", text="   ")["results"]) == 1 and len(api.call("find", text="")["results"]) == 1
    assert len(api.call("find", text="kettle!")["results"]) == 1 and len(api.call("find", text='"kettle" on')["results"]) == 1
    assert api.call("find", text="nothing-like-this")["results"] == []


def test_a_caller_already_noted_costs_no_write(home):
    store = Store(home)
    caller = {"project": "git:0123456789abcdef", "project_name": "mine", "project_root": "/x", "session": "s-0123456789ab",
              "client": "t"}
    store.note_caller(caller)
    written = store.db().total_changes
    for _ in range(5):
        store.note_caller(caller)
        store.note_caller({"project": None, "session": None})
    assert store.db().total_changes == written
    store.note_caller({**caller, "session": "s-ba9876543210"})  # a new session is written once
    assert store.db().total_changes > written
