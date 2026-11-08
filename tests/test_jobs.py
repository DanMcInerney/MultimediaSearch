"""Jobs and the GPU queue (TESTBED.md item 7): waiting and collecting, taking turns, cancelling, bad parameters, the time
limit, a home copy of a script, and a restart. The scans run `ask` over a still red video with the fake decider."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import DECIDER, FAKE, span, wait_for, write_config

from mms import jobs
from mms.config import ffmpeg_exe
from mms.store import Store

T0 = 1_699_999_980.0
LONG = 60


@pytest.fixture(scope="module")
def red(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("jobs") / "red60.mkv"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=red:s=64x36:r=1:d={LONG}",
                    "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def put(store: Store, source: str, file: Path, t0: float = T0) -> None:
    target = store.streams / source / "launch" / f"{int(t0)}.mkv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, target)
    store.add_item(source, "video", t0, t0 + LONG, store.relative(target), target.stat().st_size)


def params(question: str = "Is it red? slow:0.05", **more) -> dict:
    """An ask of a window a second (`still_s` 0), which a slow model takes its turns over."""
    return {"question": question, "since": T0 - 10, "until": T0 + 100, "limit": 1, "still_s": 0, **more}


def config(home: Path, extra: str = "") -> None:
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + extra + DECIDER)


def finish(api, job: dict) -> dict:
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    return job


def calls(home: Path) -> list[dict]:
    file = home / "fake-decider.jsonl"
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()] if file.exists() else []


def test_a_short_wait_returns_a_running_job_with_what_it_has_and_get_collects_the_rest(home, served, red):
    config(home)
    put(Store(home), "cam", red)
    api = served().api
    began = time.monotonic()
    job = api.call("jobs/start", script="ask", params=params(), wait_s=0.5)
    assert time.monotonic() - began < 5 and job["state"] == "running" and job["job"].startswith("j-")
    assert job["ended"] is None and job["error"] is None and job["script"] == "ask"
    first = job["result"]
    assert 0 < first["progress"]["windows"] < LONG and first["sightings"] and first["stopped"] is None, \
        "a sighting so far, while it is still growing"
    seen = first["sightings"][0]
    assert api.call("status")["jobs"]["running"] == 1
    done = finish(api, api.call("jobs/get", job=job["job"], wait_s=0))
    assert done["state"] == "done" and done["ended"] is not None and done["result"]["stopped"] == "range_end"
    assert done["result"]["progress"]["windows"] == LONG
    assert span(done["result"]["sightings"][0])[1] >= span(seen)[1], "the sighting grew while it ran"
    assert api.call("status")["jobs"] == {"running": 0, "queued": 0}
    assert api.call("jobs/get", job=job["job"], wait_s=0)["result"] == done["result"], "an ended job is kept"


def test_two_slow_jobs_take_turns_window_by_window(home, served, red):
    config(home)
    store = Store(home)
    put(store, "one", red)
    put(store, "two", red)
    api = served().api
    a = api.call("jobs/start", script="ask", params=params("Is it red? slow:0.05 one", sources=["one"], max_s=60), wait_s=0)
    b = api.call("jobs/start", script="ask", params=params("Is it red? slow:0.05 two", sources=["two"], max_s=60), wait_s=0)
    assert finish(api, a)["state"] == "done" and finish(api, b)["state"] == "done"
    order = [next(iter(call["questions"].values()))["instructions"].split()[-1] for call in calls(home)]
    assert sorted(set(order)) == ["one", "two"] and len(order) == 2 * LONG
    middle = order[4:-4]  # the first and last few may run alone
    assert sum(1 for x, y in zip(middle, middle[1:]) if x == y) <= 2, f"they alternate: {''.join(o[0] for o in order)}"
    assert abs(order.index("two") - order.index("one")) <= 2 and order[:8].count("one") <= 5


def test_cancel_stops_a_job_within_one_window(home, served, red):
    config(home)
    put(Store(home), "cam", red)
    api = served().api
    job = api.call("jobs/start", script="ask", params=params("Is it red? slow:0.3"), wait_s=0)
    wait_for(lambda: len(calls(home)) >= 2, what="the scan to be under way")
    began = time.monotonic()
    stopped = api.call("jobs/cancel", job=job["job"])
    assert time.monotonic() - began < 2 and stopped["state"] == "cancelled" and stopped["ended"] is not None
    assert stopped["result"]["stopped"] == "cancelled" and stopped["result"]["progress"]["windows"] < 10
    seen = len(calls(home))
    time.sleep(0.8)
    assert len(calls(home)) == seen, "no window was read after the cancel"
    assert api.call("status")["jobs"]["running"] == 0
    assert Store(home).labels(by="ask"), "the sighting found so far is labelled"
    assert api.call("jobs/cancel", job=job["job"])["state"] == "cancelled", "cancelling again changes nothing"


def test_bad_parameters_come_back_as_a_failed_job_in_the_same_call(home, served):
    config(home)
    api = served().api
    began = time.monotonic()
    job = api.call("jobs/start", script="ask", params={"question": "Is it red?", "limit": 99}, wait_s=40)
    assert time.monotonic() - began < 5, "it did not wait for the 40 s"
    assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and "limit" in job["error"]["message"]
    assert api.call("jobs/start", script="ask", params={"nope": 1}, wait_s=40)["error"]["code"] == "bad_request"
    assert api.call("jobs/start", script="ask", params={}, wait_s=40)["error"]["message"].startswith("ask needs")
    assert api.failure("jobs/start", script="ask", params=[]).field == "params"
    assert api.failure("jobs/start", script="ask", wait_s=51).field == "wait_s"
    assert api.failure("jobs/get", job="j-0", wait_s=51).code in ("bad_request", "not_found")
    assert api.failure("jobs/start", script="../ask").code == "bad_request"


def test_a_job_past_job_max_s_is_cancelled(home, served, red):
    config(home, "job_max_s = 1\n")
    put(Store(home), "cam", red)
    api = served().api
    job = api.call("jobs/start", script="ask", params=params("Is it red? slow:0.2", max_s=100), wait_s=0)
    ended = finish(api, job)
    assert ended["state"] == "cancelled" and ended["result"]["progress"]["windows"] < 20
    assert any("job_max_s" in w for w in ended["result"]["warnings"])


def test_a_home_copy_of_a_script_is_used_from_the_next_job(home, served):
    config(home)
    api = served().api
    first = api.call("jobs/start", script="ask", params={"question": "Is it red?"}, wait_s=40)
    assert "sightings" in first["result"]
    (home / "scripts").mkdir(exist_ok=True)
    (home / "scripts" / "ask.py").write_text("def run(job):\n    job.update(mine=job.params)\n", encoding="utf-8")
    second = api.call("jobs/start", script="ask", params={"x": 1}, wait_s=40)
    assert second["state"] == "done" and second["result"] == {"mine": {"x": 1}}
    (home / "scripts" / "ask.py").write_text("def run(job):\n    job.update(mine='again')\n", encoding="utf-8")
    assert api.call("jobs/start", script="ask", wait_s=40)["result"] == {"mine": "again"}, "read afresh for each job"
    assert api.call("jobs/start", script="ask", params={}, wait_s=40)["state"] == "done"


def test_a_scripts_own_errors_fail_the_job_with_a_code(home, served):
    config(home)
    (home / "scripts").mkdir(exist_ok=True)
    (home / "scripts" / "mine.py").write_text("def run(job):\n    if job.params.get('bad'):\n        raise ValueError('bad thing')\n"
                                              "    raise RuntimeError('a bug')\n", encoding="utf-8")
    (home / "scripts" / "broken.py").write_text("import nothing_like_this\n", encoding="utf-8")
    api = served().api
    bad = api.call("jobs/start", script="mine", params={"bad": True}, wait_s=40)
    assert bad["state"] == "failed" and bad["error"] == {"code": "bad_request", "message": "bad thing"}
    bug = api.call("jobs/start", script="mine", wait_s=40)
    assert bug["error"]["code"] == "internal" and "a bug" in bug["error"]["message"]
    assert api.call("jobs/start", script="broken", wait_s=40)["error"]["code"] == "internal"
    assert api.call("status")["jobs"] == {"running": 0, "queued": 0}


def test_jobs_beyond_the_limit_wait_their_turn(home, served, red):
    config(home, "jobs = 1\n")
    put(Store(home), "cam", red)
    api = served().api
    first = api.call("jobs/start", script="ask", params=params("Is it red? slow:0.1"), wait_s=0)
    second = api.call("jobs/start", script="ask", params=params("Is it red? slow:0.1"), wait_s=0)
    assert second["state"] == "queued" and api.call("status")["jobs"] == {"running": 1, "queued": 1}
    gone = api.call("jobs/cancel", job=second["job"])
    assert gone["state"] == "cancelled" and gone["result"] == {}, "a queued job cancels without running"
    api.call("jobs/cancel", job=first["job"])


def test_after_a_restart_the_daemon_has_forgotten_its_jobs(home, served):
    config(home)
    first = served()
    job = first.api.call("jobs/start", script="ask", params={"question": "Is it red?"}, wait_s=40)
    assert first.api.call("jobs/get", job=job["job"])["state"] == "done"
    first.stop()
    again = served()
    error = again.api.failure("jobs/get", job=job["job"])
    assert (error.status, error.code, error.field) == (404, "not_found", "job")


def test_the_gpu_queue_loads_on_demand_unloads_when_idle_and_tries_again_after_a_failed_load(home, served, red):
    config(home, "")
    (home / "mms.toml").write_text("[memory]\nloop_s = 0.1\nfree_gb = 0\n"
                                        f'[decider]\nmodel = "{FAKE.replace(chr(92), "/")}"\nidle_s = 0.5\n', encoding="utf-8")
    api = served().api
    assert api.call("status")["decider"]["state"] == "unloaded"
    job = api.call("jobs/start", script="ask", params={"question": "Is it red?"}, wait_s=40)
    assert job["state"] == "done" and api.call("status")["decider"]["state"] in ("ready", "unloaded")
    wait_for(lambda: api.call("status")["decider"]["state"] == "unloaded", what="the model to unload when idle")
    broken = "python:" + (home / "nothing.py").as_posix()
    (home / "mms.toml").write_text(f'[memory]\nloop_s = 0.1\nfree_gb = 0\n[decider]\nmodel = "{broken}"\n', encoding="utf-8")
    put(Store(home), "cam", red)
    wait_for(lambda: api.call("status")["decider"]["model"] == broken, what="the new model to be read")
    failed = api.call("jobs/start", script="ask", params=params(), wait_s=40)
    assert failed["state"] == "failed" and failed["error"]["code"] == "decider" and "could not load" in failed["error"]["message"]
    status = api.call("status")["decider"]
    assert status["state"] == "failed" and status["error"]
    config(home)
    wait_for(lambda: api.call("status")["decider"]["model"] == FAKE.replace(chr(92), "/"), what="the fixed model to be read")
    assert api.call("jobs/start", script="ask", params=params("Is it red?"), wait_s=40)["state"] in ("running", "done")


def fake_file(folder: Path, name: str, marker: str, load_s: float = 0.0, pause_s: float = 0.0) -> str:
    """A decider file that sleeps `load_s` as it loads and `pause_s` in each call, and appends a line to
    `<marker>.jsonl` in the home for each call."""
    path = folder / f"{name}.py"
    path.write_text(
        "import json, time\nfrom mms.decider import About\nABOUT = About(name='test')\nhome = None\n"
        "def load():\n"
        f"    time.sleep({load_s})\n"
        "def decide(state, questions, media):\n"
        f"    open(home / '{marker}.jsonl', 'a').write('x' + chr(10))\n"
        f"    time.sleep({pause_s})\n"
        "    return {name: {'type': 'noul', 'noul': 0.95} for name in questions}\n", encoding="utf-8")
    return "python:" + path.as_posix()


def test_cancelling_while_d1_loads_reports_cancelled_at_once(home, served, red, tmp_path):
    slow = fake_file(tmp_path, "slow_load", "slow", load_s=5)
    write_config(home, f'[memory]\nloop_s = 0.1\nfree_gb = 0\n[decider]\nmodel = "{slow}"\n')
    put(Store(home), "cam", red)
    api = served().api
    job = api.call("jobs/start", script="ask", params=params("Is it red?"), wait_s=0)
    wait_for(lambda: api.call("status")["decider"]["state"] == "loading", what="d1 to be loading")
    began = time.monotonic()
    stopped = api.call("jobs/cancel", job=job["job"])
    assert time.monotonic() - began < 3, "it did not wait for the 5 s load"
    assert stopped["state"] == "cancelled" and stopped["result"]["stopped"] == "cancelled"
    assert api.call("status")["jobs"]["running"] == 0
    wait_for(lambda: api.call("status")["decider"]["state"] == "ready", what="the load to finish for nobody", timeout=30)


def test_a_model_changed_in_the_config_takes_effect_between_windows_with_jobs_still_waiting(home, served, red, tmp_path):
    first = fake_file(tmp_path, "model_a", "a", pause_s=0.05)
    second = fake_file(tmp_path, "model_b", "b", pause_s=0.05)

    def model(spec: str) -> None:
        (home / "mms.toml").write_text(f'[memory]\nloop_s = 0.1\nfree_gb = 0\n[decider]\nmodel = "{spec}"\n', encoding="utf-8")

    model(first)
    put(Store(home), "cam", red)
    api = served().api
    job = api.call("jobs/start", script="ask", params=params("Is it red?", max_s=60), wait_s=0)
    log = lambda name: len((home / f"{name}.jsonl").read_text()) // 2 if (home / f"{name}.jsonl").exists() else 0  # noqa: E731
    wait_for(lambda: log("a") >= 5, what="the first model to be asked")
    model(second)
    wait_for(lambda: log("b") >= 3, what="the new model to be asked, with the job still running")
    assert api.call("jobs/get", job=job["job"], wait_s=0)["state"] == "running"
    seen = log("a")
    time.sleep(0.5)
    assert log("a") == seen, "the old model was not asked again"
    assert api.call("status")["decider"]["model"] == second
    assert finish(api, job)["state"] == "done"
    assert log("a") + log("b") == LONG, "every window was read, by one model or the other"


class CountingCv(threading.Condition):
    woken = 0

    def notify_all(self):
        self.woken += 1
        super().notify_all()


def unread_job(name: str = "d1-Omni-2B"):
    cv = CountingCv()
    owner = SimpleNamespace(env=SimpleNamespace(home=None, store=None), cv=cv,
                            gpu=SimpleNamespace(about=lambda: SimpleNamespace(name=name)))
    record = jobs._Record("s", None, {})
    return jobs.Job(record, owner), record, cv


def unread(id: int, media: str | None):
    return SimpleNamespace(id=id, kind="file", meta={"media": media} if media else {})


def segment(id: int, kind: str):
    """A captured segment: it has no `meta.media`."""
    return SimpleNamespace(id=id, kind=kind, meta={})


def test_the_not_read_warning_calls_a_segment_by_its_kind_and_says_what_look_gives():
    job, record, _ = unread_job("eyes")
    job._unread(segment(3, "audio"))
    assert record.result["warnings"] == ["1 item was not read, since eyes does not read its type: audio segments "
                                         "(item 3); find lists it and look shows it"]
    job._unread(unread(4, "audio/wav"))
    job._unread(segment(5, "audio"))
    job._unread(unread(6, "image/heic"))
    assert record.result["warnings"] == [
        "4 items were not read, since eyes does not read their types: audio segments (items 3, 5), audio/wav (item 4), "
        "image/heic (item 6); find lists them and look shows them"], "every group is media: look shows each"
    job._unread(unread(7, "application/pdf"))
    assert record.result["warnings"][0].endswith(
        "application/pdf (item 7); find lists them and look shows a picture, sound or video and gives a copy of any "
        "other file"), "a mix says both"
    assert len(record.result["warnings"]) == 1

    job, record, _ = unread_job("ears")
    job._unread(segment(8, "video"))
    job._unread(segment(9, "video"))
    assert record.result["warnings"] == ["2 items were not read, since ears does not read their types: video segments "
                                         "(items 8, 9); find lists them and look shows them"]
    job, record, _ = unread_job("docs")
    job._unread(unread(1, None))
    assert record.result["warnings"] == ["1 item was not read, since docs does not read its type: an unknown type "
                                         "(item 1); find lists it and look gives a copy of it"]


def test_the_not_read_warning_names_each_item_once_by_type_and_wakes_no_waiter():
    job, record, cv = unread_job()
    job._unread(unread(14, "application/pdf"))
    assert record.result["warnings"] == ["1 item was not read, since d1-Omni-2B does not read its type: application/pdf "
                                         "(item 14); find lists it and look gives a copy of it"]
    job._unread(unread(14, "application/pdf"))  # a script that scans twice reports the item twice
    assert len(record.result["warnings"]) == 1 and record.result["warnings"][0].startswith("1 item was not read")
    for item in (unread(15, "application/pdf"), unread(16, "application/zip"), unread(17, None)):
        job._unread(item)
    assert record.result["warnings"] == [
        "4 items were not read, since d1-Omni-2B does not read their types: application/pdf (items 14, 15), "
        "application/zip (item 16), an unknown type (item 17); find lists them and look gives a copy of each"]
    assert cv.woken == 0, "an unread item wakes nobody"

    job, record, cv = unread_job()
    for n in range(1, 20_001):
        job._unread(unread(n, "application/pdf"))
        job._unread(unread(n, "application/pdf"))
    assert record.result["warnings"] == [
        "20000 items were not read, since d1-Omni-2B does not read their types: application/pdf "
        "(items 1, 2, 3, 4, 5 and 19995 more); find lists them and look gives a copy of each"]
    assert len(job._not_read.groups["application/pdf"][1]) == 5 and cv.woken == 0, \
        "five ids are kept for a type, however many items"


def test_a_scan_twice_over_an_unread_file_has_one_warning_in_the_final_result(home, served, corpus):
    config(home)
    (home / "scripts").mkdir(exist_ok=True)
    (home / "scripts" / "twice.py").write_text(
        "def run(job):\n    for _ in range(2):\n        list(job.windows(job.params['since'], job.params['until'], scope='all'))\n",
        encoding="utf-8")
    api = served().api
    api.call("remember", path=str(corpus["documents/invoice.pdf"].path), t=T0, source="agent")
    job = api.call("jobs/start", script="twice", params={"since": T0 - 10, "until": T0 + 10}, wait_s=40)
    assert job["state"] == "done", job
    assert len(job["result"]["warnings"]) == 1 and job["result"]["warnings"][0].startswith(
        "1 item was not read, since d1-Omni-2B does not read its type: application/pdf (item "), job["result"]
    assert job["result"]["warnings"][0].endswith("); find lists it and look gives a copy of it")
