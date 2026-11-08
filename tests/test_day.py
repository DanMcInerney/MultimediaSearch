"""A synthetic day (`tests/day.py`) loaded into a daemon with the fake decider, and the questions an owner asks of it
(TESTBED.md): what text said, which sounds were heard and when, what the pictures and videos showed, upright and on time.
Each test loads the day into its own daemon; the files are made once for the module and only read."""

from __future__ import annotations

import json

import day
import pytest
from conftest import DECIDER, span, write_config
from corpus import check_unread, not_read_warnings

from mms.config import parse_when
from mms.store import Store

DAY = day.FIXED_DAY
NOT_TEXT = set("\x00﻿�\r")  # what a decoded text must not hold
pytestmark = pytest.mark.xdist_group("day")  # one worker makes the day's files once


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("day")
    return folder, day.build(folder, DAY, speech=False)


@pytest.fixture
def api(home, served, files):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 50\n" + DECIDER)
    one = served().api
    day.load(one, *files)
    return one


def when(clock: str) -> float:
    return day.at(DAY, clock)


def scan(api, **params) -> dict:
    """The shipped `ask` as a job over the whole day, and its result once it is done."""
    params.setdefault("since", DAY)
    params.setdefault("until", DAY + 86400)
    job = api.call("jobs/start", script="ask", params=params, wait_s=50)
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    assert job["state"] == "done", job
    return job["result"]


def found(api, text: str, **params) -> list[dict]:
    return api.call("find", text=text, **params)["results"]


def decider_log(home) -> list[dict]:
    file = home / "fake-decider.jsonl"
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()] if file.exists() else []


def test_a_days_text_is_recalled(home, api):
    gate = scan(api, question="contains:4821", sources=["mail"])["sightings"]
    assert [s["source"] for s in gate] == ["mail"] and span(gate[0])[0] == pytest.approx(when("08:05"), abs=1)
    assert "gate code is 4821" in api.call("look", **gate[0]["look"])["text"]

    # a note rewritten in the day: the newest version says 30, the older 45, and both are read decoded (UTF-8 and UTF-16)
    def versions(question: str, **params) -> list[dict]:
        return [s for s in scan(api, question=question, sources=["notes"], **params)["sightings"] if "item" in s["look"]]

    (newest,) = versions("contains:days", limit=1)
    assert span(newest)[0] == pytest.approx(when("17:30"), abs=1)
    (older,) = versions("contains:45")
    assert span(older)[0] == pytest.approx(when("09:00"), abs=1)
    for sighting, days in ((newest, "30"), (older, "45")):
        text = api.call("look", **sighting["look"])["text"]
        assert f"keep logs for {days} days" in text and not NOT_TEXT & set(text)

    (dentist,) = found(api, "dentist")
    assert "dentist moved to Thursday 9:30" in dentist["text"]
    assert len(found(api, "ERR-503", types=["record"])) == 5 and len(found(api, "ERR-429", types=["record"])) == 2
    (deployed,) = found(api, "deploy finished")
    assert span(deployed)[0] == pytest.approx(when(day.DEPLOYED), abs=1)

    assert found(api, "wifi") == []
    assert scan(api, question="contains:wifi", sources=["mail", "notes", day.LOG_SOURCE])["sightings"] == []
    for call in decider_log(home):
        assert call["text"]["nul"] == call["text"]["fffd"] == call["text"]["cr"] == call["text"]["esc"] == 0, call


def test_a_days_sounds_are_recalled(home, api):
    bells = scan(api, question="loud", sources=["doorbell"])["sightings"]
    assert [span(s)[0] for s in bells] == pytest.approx([when(c) for c in reversed(day.DOORBELLS)], abs=10)
    assert [s["kind"] for s in bells] == ["audio"] * 3, "newest first, a sighting for each ring"

    (ring,) = scan(api, question="loud", sources=["meeting"])["sightings"]
    ringing = when(day.MEETING_RING)
    assert ring["kind"] == "audio", "the call's soundtrack, not its picture"
    assert ringing - 5 <= span(ring)[0] <= ringing <= span(ring)[1]
    shown = api.call("look", **ring["look"])
    assert shown["image"]["width"] > 0 and shown["audio"]["rms"] > 0.05, "a look at the call gives its frame and its sound"

    voices = scan(api, question="loud", sources=["phone"])["sightings"]
    assert [span(s)[0] for s in voices] == pytest.approx([when("18:00"), when("09:12")], abs=10)
    assert {h["rate"] for call in decider_log(home) for h in call["heard"]} == {16000}


def test_a_days_pictures_and_videos_are_recalled(home, api):
    (error,) = scan(api, question="colour:red", sources=["screen"], oldest_first=True, limit=1)["sightings"]
    assert span(error)[0] == pytest.approx(when(day.ERROR_SHOWN[0]), abs=1)

    upright = sorted(scan(api, question="corner:red", sources=["phone"])["sightings"], key=span)
    assert [s["kind"] for s in upright] == ["video", "image"], "the parking video and the whiteboard photo, both upright"
    assert span(upright[0])[0] == pytest.approx(when("08:30"), abs=2)
    assert span(upright[1])[0] == pytest.approx(when("10:45"), abs=1)
    sizes = {tuple(size) for call in decider_log(home) for size, looks in zip(call["pictures"], call["looks"])
             if looks["corners"][0] == "red"}
    assert any(w > h for w, h in sizes) and any(h > w for w, h in sizes), f"a landscape and a portrait picture: {sizes}"

    (editor,) = scan(api, question="colour:blue", sources=["screen"], oldest_first=True, limit=1)["sightings"]
    assert span(editor)[0] == pytest.approx(when("15:00"), abs=1) and "capped" not in editor

    (dark,) = scan(api, question="colour:black", sources=["garage"], limit=1)["sightings"]
    assert span(dark)[0] == pytest.approx(when("21:00"), abs=1)
    (lit,) = scan(api, question="colour:white", sources=["garage"])["sightings"]
    assert span(lit)[0] == pytest.approx(when("20:00"), abs=1)


def test_a_days_pdf_is_recorded_found_and_named_as_not_read(home, api):
    store = Store(home)
    try:
        (item,) = store.items_of("downloads")
    finally:
        store.close()
    assert (item.kind, item.meta["media"], item.name) == ("file", "application/pdf", "invoice.pdf")
    rows = found(api, "invoice")
    assert [row["id"] for row in rows] == [item.id] and parse_when(rows[0]["t0"]) == pytest.approx(when("11:50"), abs=1), \
        "found by its name, at its time"

    result = scan(api, question="contains:342.50", sources=["downloads", "mail"])  # the mail is read, so the log is not empty
    assert result["sightings"] == [], "d1 does not read a PDF, so its total is not found"
    (warning,) = not_read_warnings(result)
    assert result["warnings"] == [warning], result["warnings"]
    named = check_unread(warning, "d1-Omni-2B", {"application/pdf": {item.id}})
    assert named.count == 1 and named.tail == "find lists it and look gives a copy of it", warning
    calls = decider_log(home)
    assert calls and all(call["files"] == [] for call in calls), "the mail was read, and no file reached the decider"
    shown = api.call("look", item=item.id)
    assert shown["kind"] == "file" and shown["file"]["media"] == "application/pdf" and shown["file"]["name"] == "invoice.pdf"
