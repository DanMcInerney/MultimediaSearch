"""Consent (TESTBED.md item 4, the daemon's half): nothing that opens a device or runs a program starts until the owner
approved it, an approval is bound to what the source opens or runs, and no call of the API approves."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from conftest import ROOT, env, mms, wait_for, write_config

from mms import config as configuration
from mms import consent
from mms.store import Store

TONE = '[sources.tone]\nkind = "ffmpeg"\npreset = "audio"\ninput = ["-re", "-f", "lavfi", "-i", "sine=frequency=440"]\nsegment_s = 1\n'


def source(toml: str) -> configuration.Source:
    return next(iter(configuration.load_text("x", toml).sources.values()))


def states(api) -> dict:
    return {s["name"]: s for s in api.call("status")["sources"]}


def test_an_approval_binds_a_name_to_what_it_opens_or_runs(tmp_path):
    mic = source('[sources.m]\nkind = "microphone"\ndevice = "Microphone (Realtek(R) Audio)"\nkeep = { max_gb = 1 }')
    assert mic.what() == "microphone: Microphone (Realtek(R) Audio)"
    assert not consent.approved(tmp_path, "m", mic.what())
    consent.record(tmp_path, "m", mic.what())
    entry = consent.load(tmp_path)["m"]
    assert entry["fingerprint"] == hashlib.sha256(mic.what().encode()).hexdigest()
    assert entry["what"] == mic.what() and entry["approved"]
    assert consent.approved(tmp_path, "m", mic.what())
    assert not consent.approved(tmp_path, "other", mic.what()), "the name is part of it"
    same = source('[sources.m]\nkind = "microphone"\ndevice = "Microphone (Realtek(R) Audio)"\nsegment_s = 5\n'
                  'keep = { max_gb = 9, when_full = "thin" }')
    assert consent.approved(tmp_path, "m", same.what()), "a cap or another option needs no new approval"
    assert not consent.approved(tmp_path, "m", source('[sources.m]\nkind = "microphone"\ndevice = "Another"').what())
    screen = source('[sources.s]\nkind = "screen"\ndevice = "0"\nfps = 2')
    assert screen.what() == source('[sources.s]\nkind = "screen"\ndevice = "0"\nfps = 9').what(), "fps is not what it opens"
    assert screen.what() != source('[sources.s]\nkind = "screen"\ndevice = "1"').what()
    assert source('[sources.c]\nkind = "camera"\ndevice = "x"').what() != source('[sources.c]\nkind = "microphone"\ndevice = "x"').what()
    ffmpeg = source('[sources.f]\nkind = "ffmpeg"\npreset = "audio"\ninput = ["-i", "a"]')
    assert ffmpeg.what() != source('[sources.f]\nkind = "ffmpeg"\npreset = "audio"\ninput = ["-i", "b"]').what()
    assert ffmpeg.what() == source('[sources.f]\nkind = "ffmpeg"\npreset = "camera"\ninput = ["-i", "a"]').what(), "the preset is not input"
    run = source('[sources.j]\nkind = "command"\ncommand = ["python", "a.py"]')
    assert run.what() != source('[sources.j]\nkind = "command"\ncommand = ["python", "b.py"]').what()
    assert source('[sources.n]\nkind = "folder"\npath = "x"').what() is None and source('[sources.p]\nkind = "push"').what() is None
    assert consent.revoke(tmp_path, "m") is True and consent.revoke(tmp_path, "m") is False
    assert not consent.approved(tmp_path, "m", mic.what())


def test_an_unapproved_source_starts_nothing_and_once_approved_starts_with_no_restart(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + TONE.replace("440", "4420"), approve=False)
    api = served().api
    pid = api.call("status")["pid"]
    tone = states(api)["tone"]
    assert tone["state"] == "needs_approval" and tone["approve"] == "mms approve tone"
    time.sleep(0.8)
    assert not (home / "streams" / "tone").exists(), "nothing started"
    consent.record(home, "tone", configuration.load(home).sources["tone"].what())
    wait_for(lambda: states(api)["tone"]["state"] == "running", what="the approved source to start")
    assert api.call("status")["pid"] == pid, "no restart"
    wait_for(lambda: Store(home).select(sources=["tone"]), what="a segment")


def test_changing_what_it_runs_needs_a_new_approval_and_changing_a_cap_does_not(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + TONE.replace("440", "4410"))
    api = served().api
    wait_for(lambda: states(api)["tone"]["state"] == "running", what="the source to run")
    pid = wait_for(lambda: subprocess_pids("sine=frequency=4410"), what="ffmpeg")
    api.call("source_set", name="x", kind="program", max_gb=1)  # an unrelated write of the file
    text = (home / "mms.toml").read_text(encoding="utf-8")
    (home / "mms.toml").write_text(text.replace('segment_s = 1\n', 'segment_s = 1\nkeep = { max_gb = 7 }\n'), encoding="utf-8")
    wait_for(lambda: states(api)["tone"]["keep"]["max_gb"] == 7, what="the cap to be read")
    assert states(api)["tone"]["state"] == "running" and subprocess_pids("sine=frequency=4410") == pid, \
        "a cap changed, and ffmpeg was left alone"
    (home / "mms.toml").write_text(text.replace("frequency=4410", "frequency=8800"), encoding="utf-8")
    wait_for(lambda: states(api)["tone"]["state"] == "needs_approval", what="the changed input to need approval")
    wait_for(lambda: not subprocess_pids("sine=frequency=4410"), what="the old ffmpeg to stop")
    assert not subprocess_pids("sine=frequency=8800"), "the new input did not start"


def subprocess_pids(marker: str) -> list[int]:
    from test_capture import pids_of
    return pids_of(marker)


def test_a_command_waits_for_approval_too(home, served, tmp_path):
    beat = tmp_path / "beat.txt"
    script = tmp_path / "beat.py"
    script.write_text('import sys, time\nwhile True:\n    open(sys.argv[1], "a").write("x")\n    time.sleep(0.05)\n')
    command = json.dumps([sys.executable, str(script), str(beat)])
    write_config(home, f"[memory]\nloop_s = 0.1\nfree_gb = 0\n[sources.beat]\nkind = \"command\"\ncommand = {command}\n",
                 approve=False)
    api = served().api
    assert states(api)["beat"]["state"] == "needs_approval"
    time.sleep(0.6)
    assert not beat.exists(), "nothing ran"
    consent.record(home, "beat", configuration.load(home).sources["beat"].what())
    wait_for(lambda: beat.exists() and beat.stat().st_size > 3, what="the approved command to run")
    mms(home, "approve", "beat", "--revoke")
    wait_for(lambda: states(api)["beat"]["state"] == "needs_approval", what="the revoked command to stop")
    size = beat.stat().st_size
    time.sleep(0.6)
    assert beat.stat().st_size == size, "it stopped running"


def test_mms_approve_refuses_without_a_terminal_and_revoke_stops_the_source(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + TONE.replace("440", "4420"), approve=False)
    api = served().api
    refused = subprocess.run([sys.executable, "-m", "mms.cli", "approve", "tone"], env=env(home), input="tone\n",
                             capture_output=True, text=True, timeout=60)
    assert refused.returncode == 2 and "terminal" in refused.stderr
    assert not (home / "consent.json").exists() and states(api)["tone"]["state"] == "needs_approval"
    assert mms(home, "approve", "nothing", check=False).returncode == 2
    assert mms(home, "approve", "tone", "--revoke").stdout.strip() == "tone had no approval"
    consent.record(home, "tone", configuration.load(home).sources["tone"].what())  # what `mms approve` does after the name
    wait_for(lambda: states(api)["tone"]["state"] == "running", what="the approved source")
    assert "approval removed" in mms(home, "approve", "tone", "--revoke").stdout, "revoke needs no terminal"
    wait_for(lambda: states(api)["tone"]["state"] == "needs_approval", what="the revoked source to stop")
    wait_for(lambda: not subprocess_pids("sine=frequency=4420"), what="its ffmpeg to end")
    assert "tone" not in consent.load(home)
    pushed = write_push(home)
    assert mms(home, "approve", pushed, check=False).returncode == 2, "a source that opens nothing needs no approval"


def write_push(home: Path) -> str:
    with open(home / "mms.toml", "a", encoding="utf-8") as out:
        out.write('\n[sources.agent]\nkind = "push"\n')
    return "agent"


def test_no_call_of_the_api_approves(home, served, tmp_path):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + TONE.replace("tone", "t1"), approve=False)
    api = served().api
    digest = lambda: hashlib.sha256((home / "consent.json").read_bytes()).hexdigest() if (home / "consent.json").exists() else None  # noqa: E731
    seen = digest()
    for op in ("approve", "consent", "approve/tone"):
        status, body = api.raw(op, {"source": "t1"})
        assert status == 404 and body["error"]["code"] == "not_found", op
    for field in ("approved", "approve", "consent"):
        assert api.failure("source_set", name="t2", kind="microphone", device="m", max_gb=1, **{field: True}).code == "bad_request"
    api.call("source_set", name="t2", kind="microphone", device="m", max_gb=1)
    api.call("source_set", name="t3", kind="program", max_gb=1)
    api.call("source_set", name="t3", remove=True)
    api.call("source_set", name="t3", kind="program", max_gb=1)
    api.call("remember", note="n")
    api.call("label", text="l", since=time.time() - 5)
    api.call("keep", source="t3", since=time.time() - 5)
    api.call("find")
    api.call("status", devices=False)
    job = api.call("jobs/start", script="ask", params={"question": "q"}, wait_s=5)
    assert job["state"] in ("failed", "done")
    assert digest() == seen == None, "consent.json is unchanged, and was never made"
    assert not (ROOT / "consent.json").exists()
