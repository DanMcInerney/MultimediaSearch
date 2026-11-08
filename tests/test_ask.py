"""`ask` as a job (TESTBED.md item 8): segments made with ffmpeg and indexed at known times, text versions, records and
remembered recordings, asked about through the daemon's API with the fake decider, which hits on content a test knows."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import DECIDER, ROOT, model_folder, prepared_env, span, write_config
from PIL import Image

from mms import decider_d1, decider_d3, jobs, reader
from mms.config import ffmpeg_exe, parse_when
from mms.decider import About
from mms.store import Scope, Store

EVERYTHING = Scope(all=True)
T0 = 1_699_999_980.0  # the first segment's start, a multiple of 60 s so the clock's slots line up; later ones are 100 s apart
SECONDS = 9  # a video segment: blue 0-3, red 3-6, blue 6-9
TONE = (8, 11)  # an audio segment of 19 s: silence, a tone from 8 to 11 s, silence
LONG = 60  # a still red video segment of 60 s, one frame a second


def ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("media")
    video, audio = folder / "colours.mkv", folder / "tone.mka"
    colour = lambda name: ["-f", "lavfi", "-i", f"color=c={name}:s=160x90:r=5:d=3"]  # noqa: E731
    ffmpeg(*colour("blue"), *colour("red"), *colour("blue"), "-filter_complex", "concat=n=3:v=1:a=0",
           "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(video))
    quiet = lambda s: ["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=mono:d={s}"]  # noqa: E731
    tone = ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={TONE[1] - TONE[0]},volume=4"]
    ffmpeg(*quiet(TONE[0]), *tone, *quiet(19 - TONE[1]), "-filter_complex", "concat=n=3:v=0:a=1",
           "-c:a", "libopus", "-b:a", "24k", str(audio))
    still = folder / "red60.mkv"
    ffmpeg("-f", "lavfi", "-i", f"color=c=red:s=64x36:r=1:d={LONG}", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p",
           str(still))
    big = folder / "hd.mkv"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=1920x1080:r=2:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(big))
    talk = folder / "red20-tone.mkv"  # a still red picture of 20 s with a tone for its soundtrack
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x36:r=1:d=20", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=20,volume=4",
           "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "24k", str(talk))
    return {"video": video, "audio": audio, "red60": still, "hd": big, "talk": talk}


@pytest.fixture
def store(home) -> Store:
    one = Store(home)
    yield one
    one.close()


@pytest.fixture
def api(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + DECIDER)
    return served().api


def index(store: Store, source: str, kind: str, file: Path, t0: float, length: float, **who) -> int:
    """Put a copy of a media file in the home as a segment of `source` starting at `t0`."""
    target = store.streams / source / "launch" / f"{int(t0)}{file.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, target)
    return store.add_item(source, kind, t0, t0 + length, store.relative(target), target.stat().st_size, **who)


def scan(api, **params) -> dict:
    """Run the shipped script as a job over the history (made at T0), and return its result once it is done."""
    params.setdefault("since", T0 - 100)
    params.setdefault("until", T0 + 1000)
    job = api.call("jobs/start", script="ask", params=params, wait_s=50)
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    assert job["state"] == "done", job
    return job["result"]


# A decider with other facts than d1's: a `python:` file that answers as `fake_decider.py` does (loaded by its path, since a
# daemon in a subprocess has no tests/ on sys.path), under facts the test names. `known=True` takes the About of a backend
# (`decider.known_abouts()`) by `label` instead.
FACADE = """\
import dataclasses
import importlib.util

{about}
home = None
_fake = None


def decide(state, questions, media):
    global _fake
    if _fake is None:
        found = importlib.util.spec_from_file_location("fake_decider_of_facade", {path!r})
        _fake = importlib.util.module_from_spec(found)
        found.loader.exec_module(_fake)
    _fake.home = home
    return _fake.decide(state, questions, media)
"""
D1_ABOUT = "from mms.decider_d1 import ABOUT as D1\n\nABOUT = dataclasses.replace(D1, name={label!r}, **{facts!r})"
KNOWN_ABOUT = "from mms.decider import known_abouts\n\nABOUT = next(a for a in known_abouts() if a.name == {label!r})"


def facade(folder: Path, label: str, *, known: bool = False, **facts) -> str:
    """The `python:` model of a decider called `label`: d1's facts changed by `facts`, or (known) the About of that name."""
    about = (KNOWN_ABOUT if known else D1_ABOUT).format(label=label, facts=facts)
    path = folder / f"facade_{label}.py"
    path.write_text(FACADE.format(about=about, path=str(ROOT / "tests" / "fake_decider.py")), encoding="utf-8")
    return f"python:{path.as_posix()}"


def api_under(home: Path, served, spec: str):
    """The API of a daemon whose decider is the model `spec`."""
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + f'[decider]\nmodel = "{spec}"\n')
    return served().api


def decider_log(home: Path) -> list[dict]:
    file = home / "fake-decider.jsonl"
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()] if file.exists() else []


def test_video_sighting_is_the_red_seconds_and_its_handle_shows_the_frame(home, api, store, media):
    item = index(store, "cam", "video", media["video"], T0, SECONDS)
    result = scan(api, question="Is it red?", sources=["cam"])
    (s,) = result["sightings"]
    assert set(s) == {"source", "t0", "t1", "p", "kind", "look"}, "no said for a yes/no, no capped or project when unset"
    assert s["source"] == "cam" and s["kind"] == "video"
    assert span(s) == (pytest.approx(T0 + 3, abs=0.3), pytest.approx(T0 + 5, abs=0.3))
    assert s["p"] == 0.95
    assert s["look"]["item"] == item and 2.7 <= s["look"]["at"] <= 5.3
    assert result["stopped"] == "range_end" and result["searched"]["sources"] == ["cam"]
    assert result["progress"]["windows"] == 3 and result["progress"]["of_about"] == 9, "a window for each colour, not a second"
    shown = api.call("look", **s["look"])
    with Image.open(shown["image"]["path"]) as picture:
        r, g, b = np.asarray(picture.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    assert r > 200 and g < 60 and b < 60, "the handle gives the frame the sighting is about"


def test_a_choice_sighting_says_its_pick_and_a_candidate_what_the_model_picked_instead(home, api, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    result = scan(api, question="What colour is it? red", answers=["red", "blue"], merge_s=0.5)
    (s,) = result["sightings"]
    assert s["said"] == "red" and s["p"] == 0.9, "the pick alone: no question, no p in the text"
    assert result["candidates"] and {c["said"] for c in result["candidates"]} == {"none"}, "the blue seconds picked none"
    assert store.labels(by="ask")[0].text.startswith("What colour is it? red: red (p 0.90), "), "the label keeps the question"


def test_a_score_says_its_level(home, api, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    red = {"type": "score", "instructions": "How red is it?", "criteria": ["no", "a bit", "very"], "hit": 2}
    result = scan(api, questions={"red": red}, merge_s=0.5)
    (s,) = result["sightings"]
    assert s["said"] == "level 2" and s["p"] == 1.0
    assert {c["said"] for c in result["candidates"]} == {"level 0"}
    assert store.labels(by="ask")[0].text.startswith("red: level 2 (p 1.00), ")


def test_two_questions_that_must_hit_keep_the_named_form_with_each_ones_p(home, api, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    both = {"red": {"type": "noul", "instructions": "red?", "hit": 0.9},
            "level": {"type": "score", "instructions": "red?", "criteria": ["no", "a bit", "very"], "hit": 2}}
    result = scan(api, questions=both, merge_s=0.5)
    (s,) = result["sightings"]
    assert s["said"] == "red: yes (p 0.95); level: level 2 (p 1.00)" and s["p"] == 0.95
    assert store.labels(by="ask")[0].text.startswith(s["said"] + ", "), "the label is the same text, with the time"
    assert {c["said"] for c in result["candidates"]} == {"red: below 0.9 (p 0.05); level: level 0 (p 0.00)"}
    held = scan(api, questions={**both, "red": {**both["red"], "hit": 0.99}})
    assert held["sightings"] == []
    assert held["candidates"][0]["said"] == "red: below 0.99 (p 0.95); level: level 2 (p 1.00)", "each p says which one held it back"
    assert held["candidates"][0]["p"] == 0.95


def test_a_moment_is_one_t_and_a_span_is_t0_and_t1(home, api, store, media):
    store.add_version("notes", "n.md", T0, data=b"the kettle\n")
    index(store, "mic", "audio", media["audio"], T0 + 100, 19)
    (s,) = scan(api, question="contains:kettle", sources=["notes"])["sightings"]
    assert "t" in s and not {"t0", "t1"} & set(s), "a text version is one instant"
    (heard,) = scan(api, question="Is it loud?", sources=["mic"])["sightings"]
    assert {"t0", "t1"} <= set(heard) and "t" not in heard and span(heard)[1] > span(heard)[0], "a sound spans its windows"
    candidates = scan(api, question="contains:kettle", sources=["mic"])["candidates"]
    assert candidates, "the sound's windows that did not hit"
    assert all({"t0", "t1"} <= set(c) and "t" not in c and span(c)[1] > span(c)[0] for c in candidates), "a sound's window spans"
    (note,) = scan(api, question="contains:nothing", sources=["notes"])["candidates"]
    assert "t" in note and not {"t0", "t1"} & set(note)


def test_labels_are_written_one_per_sighting_by_ask(home, api, store, media):
    item = index(store, "cam", "video", media["video"], T0, SECONDS)
    index(store, "cam", "video", media["video"], T0 + 100, SECONDS)
    result = scan(api, question="Is it red?")
    labels = store.labels(by="ask")
    assert len(result["sightings"]) == len(labels) == 2
    first = max(labels, key=lambda label: label.t0)
    assert first.source == "cam" and first.t0 == pytest.approx(span(result["sightings"][0])[0], abs=0.01)
    assert first.item is not None and first.item != item and "yes" in first.text
    assert first.meta["kind"] == "video" and first.meta["model"] == str(api.call("status")["decider"]["model"])


def test_limit_one_is_the_newest_and_oldest_first_flips_it(home, api, store, media):
    first = index(store, "cam", "video", media["video"], T0, SECONDS)
    last = index(store, "cam", "video", media["video"], T0 + 100, SECONDS)
    items = lambda result: [s["look"]["item"] for s in result["sightings"]]  # noqa: E731
    assert items(scan(api, question="Is it red?", limit=1)) == [last]
    assert items(scan(api, question="Is it red?", limit=5)) == [last, first]
    assert items(scan(api, question="Is it red?", limit=1, oldest_first=True)) == [first]


def test_frames_read_backwards_in_blocks_match_forwards(home, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    every = {"every_s": 0, "still_s": 0}  # every frame, as the blocks are of frames
    forward = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=False, profile=every, scope=EVERYTHING))
    backward = list(reader.windows(store, about=decider_d1.ABOUT, profile=every, scope=EVERYTHING))
    assert len(forward) == 45  # 9 s at 5 frames a second: five blocks of frames, each read from its own keyframe
    assert [w.at for w in backward] == [w.at for w in reversed(forward)]
    assert [w.at for w in forward] == pytest.approx([i / 5 for i in range(45)])
    assert all(a.media.tobytes() == b.media.tobytes() for a, b in zip(backward, reversed(forward)))
    reds = [w.at for w in forward if w.media.getpixel((0, 0))[0] > 150]
    assert reds[0] == pytest.approx(3.0) and reds[-1] == pytest.approx(5.8)


def test_since_and_until_cut_the_range(home, api, store, media):
    first = index(store, "cam", "video", media["video"], T0, SECONDS)
    index(store, "cam", "video", media["video"], T0 + 100, SECONDS)
    items = lambda result: [s["look"]["item"] for s in result["sightings"]]  # noqa: E731
    assert items(scan(api, question="Is it red?", until=T0 + 50)) == [first]
    found = scan(api, question="Is it red?", since=T0 + 4.5, until=T0 + 50)
    assert items(found) == [first] and span(found["sightings"][0])[0] == pytest.approx(T0 + 4.6, abs=0.05), \
        "the first frame in the range, not the next slot"


def test_frames_at_a_ranges_start_are_not_used_up_by_the_frames_before_it(home, store, media):
    index(store, "cam", "video", media["red60"], T0, LONG)
    seconds = lambda **kw: [w.at for w in reader.windows(store, about=decider_d1.ABOUT, newest_first=False, scope=EVERYTHING, **kw)]  # noqa: E731
    every = lambda s: {"every_s": s, "still_s": 0}  # noqa: E731  # the slots, not the runs of still frames
    assert seconds(since=T0 + 31, profile=every(30)) == pytest.approx([31])  # was none: slot 0 was spent at 0 s
    assert seconds(since=T0 + 10, profile=every(60)) == pytest.approx([10])
    assert seconds(since=T0 + 10, until=T0 + 20, profile=every(4)) == pytest.approx([10, 12, 16])
    assert seconds(since=T0 + 10, profile=every(30)) == pytest.approx([10, 30])  # slots are of the clock
    assert len(seconds(profile=every(1))) == LONG


def test_nothing_matching_is_no_sightings_and_up_to_five_candidates_best_first(home, api, store):
    for i, grade in enumerate([0.1, 0.5, 0.3, 0.7, 0.2, 0.6, 0.4]):  # seven moments, a minute apart
        store.add_version("notes", f"n{i}.md", T0 + 60 * i, data=f"nothing here grade:{grade}\n".encode())
    result = scan(api, question="contains:kettle")
    assert result["sightings"] == [] and store.labels() == [], "candidates are never labelled"
    assert [c["p"] for c in result["candidates"]] == [0.7, 0.6, 0.5, 0.4, 0.3], "the best five, best first"
    assert all(set(c) == {"source", "t", "p", "kind", "look"} and c["kind"] == "text" and c["source"] == "notes"
               for c in result["candidates"]), "a yes/no candidate is below the hit by definition: no said"
    best = result["candidates"][0]
    assert parse_when(best["t"]) == pytest.approx(T0 + 180, abs=0.001) and "grade:0.7" in api.call("look", **best["look"])["text"]
    assert len(scan(api, question="contains:kettle", candidates=7)["candidates"]) == 7
    assert [c["p"] for c in scan(api, question="contains:kettle", candidates=2)["candidates"]] == [0.7, 0.6]


def test_candidates_of_sound_and_pictures_are_windows_that_did_not_count(home, api, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    index(store, "mic", "audio", media["audio"], T0, 19)
    result = scan(api, question="contains:kettle")
    assert result["sightings"] == [] and store.labels() == []
    assert sorted(c["source"] for c in result["candidates"]) == ["cam", "mic"], "a moment is a source within merge_s: one each"
    assert all(c["p"] == 0.05 and "said" not in c for c in result["candidates"])
    assert len(scan(api, question="contains:kettle", merge_s=0, still_s=0)["candidates"]) == 5, "windows a second apart are moments"
    assert scan(api, question="Is it red?", sources=["mic"])["sightings"] == [], "sound is never a red picture"


def test_candidates_is_a_whole_number_from_0_to_20(home, api, store):
    store.add_version("notes", "n.md", T0, data=b"nothing here\n")
    assert scan(api, question="contains:kettle", candidates=0)["candidates"] == []
    assert len(scan(api, question="contains:kettle", candidates=20)["candidates"]) == 1
    for bad in (21, 1.5, -1, True, "5"):
        job = api.call("jobs/start", script="ask", params={"question": "contains:kettle", "candidates": bad}, wait_s=40)
        assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and "candidates" in job["error"]["message"], bad


def test_candidates_are_distinct_moments(home, api, store):
    store.add_version("notes", "a.md", T0, data=b"nothing grade:0.3\n")
    store.add_version("notes", "b.md", T0 + 5, data=b"nothing grade:0.6\n")  # within merge_s of a.md's: the better one is kept
    store.add_version("notes", "c.md", T0 + 60, data=b"nothing grade:0.2\n")  # within merge_s of the sighting below
    store.add_version("notes", "d.md", T0 + 66, data=b"the kettle\n")
    store.add_version("other", "e.md", T0 + 5, data=b"nothing grade:0.4\n")  # the same time, another source
    result = scan(api, question="contains:kettle")
    (s,) = result["sightings"]
    assert parse_when(s["t"]) == pytest.approx(T0 + 66, abs=0.001)
    assert [(label.source, round(label.t0)) for label in store.labels()] == [("notes", round(T0 + 66))], "only the sighting"
    assert [(c["source"], c["p"]) for c in result["candidates"]] == [("notes", 0.6), ("other", 0.4)]
    far = scan(api, question="contains:kettle", merge_s=1)
    assert [c["p"] for c in far["candidates"]] == [0.6, 0.4, 0.3, 0.2], "with merge_s 1 each of them is a moment"


def test_candidates_of_equal_p_come_in_scan_order_newest_first(home, api, store):
    for i in range(3):
        store.add_version("notes", f"n{i}.md", T0 + 100 * i, data=b"nothing grade:0.7\n")
    got = [parse_when(c["t"]) for c in scan(api, question="contains:kettle")["candidates"]]
    assert got == pytest.approx([T0 + 200, T0 + 100, T0], abs=0.001)
    oldest = [parse_when(c["t"]) for c in scan(api, question="contains:kettle", oldest_first=True)["candidates"]]
    assert oldest == pytest.approx([T0, T0 + 100, T0 + 200], abs=0.001)


def test_a_candidate_keeps_its_distance_from_the_whole_span_of_a_sighting(home, api, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    result = scan(api, question="Is it red?", merge_s=1, still_s=0)  # a sighting that grows over a window a second
    (s,) = result["sightings"]
    start, end = span(s)
    assert end - start >= 2, "a sighting of several seconds"
    times = [parse_when(c["t"]) for c in result["candidates"]]
    assert times, "the blue seconds either side of it"
    assert all(max(0.0, max(t, start) - min(t, end)) > 1 for t in times), "from its end as well as its start"
    assert not any(end < t <= end + 1 for t in times)


def test_a_sighting_that_flickers_does_not_fill_the_pool_and_crowd_out_other_moments(home, api, store):
    pad = "z" * 1100  # two records fill a window of 3,000 characters: a window a second apart, hits and misses alternating
    notes = ["the kettle" if k % 2 else "nothing grade:0.7" for k in range(500)]
    store.add_records("thermo", [(T0 + i, json.dumps({"n": i, "note": notes[i // 2], "pad": pad})) for i in range(1000)])
    store.add_version("other", "o.md", T0 + 1500, data=b"nothing grade:0.5\n")
    result = scan(api, question="contains:kettle", until=T0 + 2000, merge_s=4, max_sighting_s=0, limit=5)
    assert result["progress"]["windows"] == 501 and len(result["sightings"]) == 1
    assert [(c["source"], c["p"]) for c in result["candidates"]] == [("other", 0.5)], \
        "the misses between the hits belong to the sighting"


def test_the_windows_of_one_file_version_are_one_moment_and_do_not_fill_the_pool(home, api, store):
    paragraphs = [f"grade:0.6 {'x' * 4980}" for _ in range(250)]  # a window each, all at the version's time
    store.add_version("notes", "long.md", T0, data="\n\n".join(paragraphs).encode())
    store.add_version("other", "o.md", T0 + 500, data=b"nothing grade:0.5\n")
    result = scan(api, question="contains:kettle")
    assert result["progress"]["windows"] == 251
    assert [(c["source"], c["p"]) for c in result["candidates"]] == [("notes", 0.6), ("other", 0.5)]


def test_equal_p_at_the_edge_of_the_pool_keeps_the_ones_scanned_first(home, api, store):
    for i in range(201):  # one more moment than the pool holds, all alike
        store.add_version("notes", f"n{i}.md", T0 + 60 * i, data=b"nothing grade:0.3\n")
    result = scan(api, question="contains:kettle", until=T0 + 20000)
    assert result["progress"]["windows"] == 201
    assert [parse_when(c["t"]) for c in result["candidates"]] == pytest.approx([T0 + 60 * i for i in range(200, 195, -1)],
                                                                               abs=0.001), "the newest five, scanned first"


def test_audio_sighting_covers_the_tone_and_quiet_windows_are_skipped(home, api, store, media):
    item = index(store, "mic", "audio", media["audio"], T0, 19)
    loud = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=False, scope=EVERYTHING))
    assert [w.at for w in loud] == [5.0, 7.5, 10.0]  # the other four windows of 19 s are silent
    assert all(w.kind == "audio" and w.media["sampling_rate"] == 16000 for w in loud)
    (s,) = scan(api, question="Is it loud?")["sightings"]
    assert s["look"]["item"] == item and s["kind"] == "audio"
    start, end = span(s)
    assert start <= T0 + TONE[0] and end >= T0 + TONE[1]
    assert end - start <= 10 and s["look"]["at"] in (5.0, 7.5, 10.0)
    sound = api.call("look", **s["look"])
    assert sound["kind"] == "audio" and sound["audio"]["rms"] > 0.1 and sound["audio"]["seconds"] == 5.0


def test_text_versions_are_asked_by_their_content(home, api, store):
    old = store.add_version("notes", "todo.md", T0, data=b"buy milk\n\nfix the kettle\n")
    store.add_version("notes", "todo.md", T0 + 60, data=b"buy milk\n")
    store.add_version("notes", "other.md", T0 + 30, data=b"nothing here\n")
    (s,) = scan(api, question="contains:kettle", context="the owner's todo list")["sightings"]
    assert s["look"] == {"item": old, "at": 0} and s["kind"] == "text"
    assert "t0" not in s and "t1" not in s and parse_when(s["t"]) == pytest.approx(T0, abs=0.001), "one instant is one t"
    assert [x["look"]["item"] for x in scan(api, question="contains:kettle", since=T0 + 1)["sightings"]] == [old], \
        "the first version was still the file until T0 + 60"
    assert scan(api, question="contains:kettle", since=T0 + 61)["sightings"] == []
    shown = api.call("look", **s["look"])
    assert shown["kind"] == "text" and "fix the kettle" in shown["text"]


def test_ask_reads_a_version_saved_before_the_range_and_unchanged_in_it(home, api, store):
    monday, tuesday = T0, T0 + 86400
    saved = store.add_version("notes", "retention.md", monday, data=b"the retention rule: keep 30 days\n")
    store.add_version("notes", "retention.md", tuesday + 3600, data=b"the retention rule: keep 7 days\n")
    store.add_version("notes", "other.md", tuesday, data=b"nothing here\n")
    found = scan(api, question="contains:retention", since=tuesday, until=tuesday + 600)
    assert [s["look"]["item"] for s in found["sightings"]] == [saved], "saved on Monday, unchanged all Tuesday morning"
    later = scan(api, question="contains:retention", since=tuesday + 7200, until=tuesday + 9000)
    assert later["sightings"][0]["look"]["item"] != saved
    changed = api.call("find", since=tuesday, until=tuesday + 600)
    assert [(r["type"], r["name"]) for r in changed["results"]] == [("item", "other.md")], \
        "find is what changed in the range"


# best_first: the notes below are 100 s apart, so each is a sighting of its own at a `hit` of 0.6 (a note is a sighting at the
# p its `grade:` gives), and what the scan must return is worked out from these plants alone.
GRADES = [0.65, 0.9, 0.7, 0.85, 0.62, 0.75]  # a note per 100 s from T0, oldest first
BEST = dict(question="contains:kettle", hit=0.6)


def plant(store: Store, grades: list[float], source: str = "notes") -> None:
    for i, grade in enumerate(grades):
        store.add_version(source, f"n{i}.md", T0 + 100 * i, data=f"nothing grade:{grade}\n".encode())


def test_best_first_gives_the_limit_highest_p_best_first_where_the_default_gives_the_first_found_in_scan_order(home, api, store):
    plant(store, GRADES)
    best = scan(api, **BEST, limit=3, best_first=True)
    assert [s["p"] for s in best["sightings"]] == [0.9, 0.85, 0.75], "the three highest, best first, whatever their time"
    assert [span(s)[0] for s in best["sightings"]] == pytest.approx([T0 + 100, T0 + 300, T0 + 500], abs=0.001)
    assert best["stopped"] == "range_end" and best["progress"]["windows"] == 6, "the scan read past three sightings"
    assert sorted(label.t0 for label in store.labels(by="ask")) == pytest.approx([T0 + 100, T0 + 300, T0 + 500], abs=0.001), \
        "one label for each sighting returned, none for the three it left out"
    default = scan(api, **BEST, limit=3)
    assert [s["p"] for s in default["sightings"]] == [0.75, 0.62, 0.85], "the first three found, newest first"
    assert default["stopped"] == "limit" and default["progress"]["windows"] == 3
    assert [s["p"] for s in scan(api, **BEST, limit=3, oldest_first=True)["sightings"]] == [0.65, 0.9, 0.7]
    assert [s["p"] for s in scan(api, **BEST, limit=10, best_first=True)["sightings"]] == sorted(GRADES, reverse=True)


def test_best_first_ties_go_to_the_sighting_scanned_first(home, api, store):
    plant(store, [0.7, 0.7, 0.7])
    times = lambda **kw: [span(s)[0] for s in scan(api, **BEST, limit=2, best_first=True, **kw)["sightings"]]  # noqa: E731
    assert times() == pytest.approx([T0 + 200, T0 + 100], abs=0.001), "newest first: the newest two"
    assert times(oldest_first=True) == pytest.approx([T0, T0 + 100], abs=0.001), "oldest first: the oldest two"


def test_best_first_makes_its_sightings_as_the_default_does(home, api, store):
    store.add_version("notes", "a.md", T0, data=b"nothing grade:0.7\n")
    store.add_version("notes", "b.md", T0 + 5, data=b"nothing grade:0.8\n")  # within merge_s: one sighting, at its best window
    store.add_version("notes", "c.md", T0 + 100, data=b"nothing grade:0.75\n")
    store.add_version("notes", "d.md", T0 + 200, data=b"nothing grade:0.3\n")  # no hit
    result = scan(api, **BEST, limit=5, best_first=True)
    assert [s["p"] for s in result["sightings"]] == [0.8, 0.75]
    assert span(result["sightings"][0]) == pytest.approx((T0, T0 + 5), abs=0.001)
    assert [c["p"] for c in result["candidates"]] == [0.3]


def test_a_candidate_leaves_out_the_windows_near_a_sighting_best_first_let_go(home, api, store):
    store.add_version("notes", "far.md", T0 + 500, data=b"nothing grade:0.4\n")  # a moment of its own
    store.add_version("notes", "x.md", T0 + 200, data=b"nothing grade:0.9\n")  # the sighting that stays
    store.add_version("notes", "near.md", T0 + 105, data=b"nothing grade:0.5\n")  # 5 s from the one let go
    store.add_version("notes", "b.md", T0 + 100, data=b"nothing grade:0.7\n")  # closed by the next window, then let go (limit 1)
    store.add_version("notes", "a.md", T0, data=b"nothing grade:0.8\n")  # still open at the end, and not returned either
    result = scan(api, **BEST, limit=1, best_first=True, candidates=5)
    assert [s["p"] for s in result["sightings"]] == [0.9] and len(store.labels(by="ask")) == 1
    assert [c["p"] for c in result["candidates"]] == [0.4], "0.5 is a lesser window of the 0.7 moment, which was sighted"


def test_a_candidate_is_not_crowded_out_by_the_many_windows_beside_the_sightings_best_first_let_go(home, api, store):
    """250 sightings (0.9) with a 0.7 note 5 s after each, which is beside a sighting and so no candidate, and three notes of 0.3
    of their own, the newest. Only the best sighting is kept (limit 1): the 0.7 notes must not fill the pool of candidates
    and push the three out of it, wherever the sightings they stand beside went."""
    for i in range(250):
        store.add_version("notes", f"hit{i}.md", T0 + 100 * i, data=b"nothing grade:0.9\n")
        store.add_version("notes", f"near{i}.md", T0 + 100 * i + 5, data=b"nothing grade:0.7\n")
    far = [T0 + 100 * (260 + 10 * k) for k in range(3)]
    for k, t in enumerate(far):
        store.add_version("notes", f"far{k}.md", t, data=b"nothing grade:0.3\n")
    result = scan(api, question="contains:kettle", hit=0.8, limit=1, best_first=True, candidates=5, since=T0 - 100,
                  until=T0 + 100 * 300)
    assert result["stopped"] == "range_end" and result["progress"]["windows"] == 503
    (best,) = result["sightings"]
    assert best["p"] == 0.9 and span(best)[0] == pytest.approx(T0 + 100 * 249, abs=0.001), "the newest of the 250 equal sightings"
    assert [c["p"] for c in result["candidates"]] == [0.3] * 3
    assert sorted(span(c)[0] for c in result["candidates"]) == pytest.approx(far, abs=0.001)


def test_best_first_stops_at_max_s_and_ranks_what_it_scanned(home, api, store):
    plant(store, GRADES + [0.8] * 4)  # the 0.8 notes are the newest, so they are read first
    result = scan(api, question="contains:kettle slow:0.3", hit=0.6, limit=2, best_first=True, max_s=1)
    n = result["progress"]["windows"]
    assert result["stopped"] == "max_s" and 1 <= n < 10, "0.3 s a window against 1 s"
    scanned = ([0.8] * 4 + GRADES[::-1])[:n]  # newest first
    assert [s["p"] for s in result["sightings"]] == sorted(scanned, reverse=True)[:2]


def test_a_running_best_first_scan_reports_the_best_sightings_so_far(home, api, store):
    plant(store, GRADES)
    params = {"question": "contains:kettle slow:0.2", "hit": 0.6, "limit": 2, "best_first": True, "since": T0 - 100,
              "until": T0 + 1000}
    job, mid = api.call("jobs/start", script="ask", params=params, wait_s=0), 0
    while True:
        result = job.get("result") or {}
        n = (result.get("progress") or {}).get("windows", 0)
        scanned = GRADES[::-1][:n]  # newest first
        assert [s["p"] for s in result.get("sightings", [])] == sorted(scanned, reverse=True)[:2], (n, result)
        mid += 0 < n < len(GRADES)
        if job["state"] not in ("queued", "running"):
            break
        job = api.call("jobs/get", job=job["job"], wait_s=0.1)
    assert job["state"] == "done" and mid, "the sightings were best first while the scan was running, not only at its end"


def test_best_first_keeps_no_more_than_the_limit_of_closed_sightings_alive(home):
    """40 sightings of one source, each window an object a test can see freed: while the scan runs only the closed ones it
    keeps (the limit), the open one and the window just read are alive, and the result is the planted highest three."""
    import gc
    import weakref

    class Window:
        kind, item, at, media, project, source = "text", None, None, None, None, "s"

        def __init__(self, i: int, p: float):
            self.t0 = self.t1 = T0 + 100.0 * i
            self.text = f"{p}"

    grades = [0.7 + 0.005 * ((i * 7) % 40) for i in range(40)]  # distinct, and every one a hit at 0.6
    refs, alive, reported, labels = [], [], {}, []

    def read_windows(*args, **kw):
        for i in reversed(range(40)):  # newest first
            w = Window(i, grades[i])
            refs.append(weakref.ref(w))
            yield w
            del w
            gc.collect()
            alive.append(sum(r() is not None for r in refs))

    class Job:
        Cancelled = type("Cancelled", (Exception,), {})
        cancelled = False
        now = T0 + 5000
        params = {"question": "contains:kettle", "hit": 0.6, "limit": 3, "best_first": True, "since": T0 - 100,
                  "until": T0 + 5000}
        store = SimpleNamespace(project_name=lambda project: None, has_clock=lambda source: False)
        about = staticmethod(lambda: decider_d1.ABOUT)
        check = staticmethod(lambda questions: None)
        profile = staticmethod(lambda overrides=None: reader.profile_of(overrides or {}, decider_d1.ABOUT))
        estimate = staticmethod(lambda *a, **kw: {"text": 40, "audio": 0})
        searched = staticmethod(lambda *a, **kw: {})
        decider = staticmethod(lambda: {"state": "ready", "model": "m"})
        warn = staticmethod(lambda text: None)
        update = staticmethod(lambda **fields: reported.update(fields))
        label = staticmethod(lambda *a, **kw: labels.append(a))
        windows = staticmethod(read_windows)

        @staticmethod
        def decide(state, questions, media):
            return {name: {"type": "noul", "noul": float(state)} for name in questions}

    jobs.load_script(home, "ask").run(Job())
    assert [s["p"] for s in reported["sightings"]] == pytest.approx(sorted(grades, reverse=True)[:3]) and len(labels) == 3
    assert max(alive) <= 3 + 2, alive  # three closed, the open one, and the window just read


def test_best_first_is_true_or_false(home, api, store):
    store.add_version("notes", "n.md", T0, data=b"nothing here\n")
    for bad in ("true", "yes", 1, 0, None, [True]):
        job = api.call("jobs/start", script="ask", params={"question": "contains:kettle", "best_first": bad}, wait_s=40)
        assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and "best_first" in job["error"]["message"], bad
    assert scan(api, question="contains:kettle", best_first=False)["stopped"] == "range_end"


def test_the_estimate_warning_says_best_first_ranks_only_what_is_scanned_before_max_s(home, api, store, media):
    index(store, "cam", "video", media["red60"], T0, LONG)

    def warnings(**params) -> list[str]:
        return scan(api, question="Is it red?", max_s=1, sources=["cam"], **params)["warnings"]

    assert any("max_s" in w and "best_first ranks only what is scanned before max_s" in w for w in warnings(best_first=True))
    assert not any("best_first" in w for w in warnings()), "60 windows at 0.1 s are more than 1 s: the plain warning only"
    assert not any("best_first" in w for w in warnings(best_first=False))


def test_a_long_run_is_cut_at_the_cap_and_marked(home, api, store, media):
    item = index(store, "cam", "video", media["red60"], T0, LONG)
    whole = scan(api, question="Is it red?", limit=1, still_s=0)  # a window a second, which the cap cuts
    (s,) = whole["sightings"]
    assert "capped" not in s and span(s)[1] - span(s)[0] >= 58 and whole["progress"]["windows"] == 60
    cut = scan(api, question="Is it red?", limit=1, max_sighting_s=20, still_s=0)
    (s,) = cut["sightings"]
    assert s["capped"] is True and s["look"]["item"] == item and span(s)[1] - span(s)[0] <= 20
    assert span(s)[1] >= T0 + 58, "the newest 20 s: last seen"
    assert cut["progress"]["windows"] == 22, "it stopped at the cap, not after the whole run"
    assert cut["stopped"] == "limit" and store.labels(by="ask")[0].meta["capped"] is True
    assert any(label.text.startswith("Is it red?: yes (p 0.95), ") and label.text.endswith(" (capped)") for label in store.labels(by="ask"))
    more = scan(api, question="Is it red?", limit=5, max_sighting_s=20, still_s=0)
    assert [s.get("capped", False) for s in more["sightings"]] == [True, True, False]
    assert "capped" not in scan(api, question="Is it red?", max_sighting_s=0, still_s=0)["sightings"][0]


def test_a_still_clip_that_hits_is_one_sighting_that_spans_it_read_in_a_few_windows(home, api, store, media):
    item = index(store, "cam", "video", media["red60"], T0, LONG)
    result = scan(api, question="Is it red?")
    (s,) = result["sightings"]
    assert span(s) == (pytest.approx(T0, abs=0.01), pytest.approx(T0 + 59, abs=0.01)), "the whole of the still stretch"
    assert s["look"] == {"item": item, "at": 59.0}, "the key frame of the newest run"
    assert result["progress"]["windows"] == 2 and result["progress"]["of_about"] == 60, "30 s a window, against the estimate's most"
    every = scan(api, question="Is it red?", still_s=0)
    assert every["progress"]["windows"] == 60 and span(every["sightings"][0]) == span(s), "the same sighting, a window a second"
    oldest = scan(api, question="Is it red?", oldest_first=True)
    assert span(oldest["sightings"][0]) == span(s) and oldest["sightings"][0]["look"]["at"] == 0.0 and oldest["progress"]["windows"] == 2
    assert scan(api, question="Is it red?", still_s=10)["progress"]["windows"] == 6, "10 s a window"
    assert scan(api, question="Is it red?", still_s=1000)["progress"]["windows"] == 1, "a segment is at least one window"


def test_still_s_is_a_number_of_at_least_0(home, api, store, media):
    index(store, "cam", "video", media["red60"], T0, LONG)
    for bad in (-1, "30", True, None):
        job = api.call("jobs/start", script="ask", params={"question": "Is it red?", "still_s": bad}, wait_s=40)
        assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and "still_s" in job["error"]["message"], bad
    assert scan(api, question="Is it red?", still_s=0.5)["progress"]["windows"] == 60


def test_runs_of_still_frames_come_in_order_across_items_newest_first_and_oldest_first(home, store, media):
    first = index(store, "cam", "video", media["red60"], T0, LONG)
    second = index(store, "cam", "video", media["video"], T0 + 100, SECONDS)  # blue 0-3, red 3-6, blue 6-9
    newest = list(reader.windows(store, about=decider_d1.ABOUT, scope=EVERYTHING))
    oldest = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=False, scope=EVERYTHING))
    assert [w.item.id for w in newest] == [second] * 3 + [first] * 2 and [w.item.id for w in oldest] == [first] * 2 + [second] * 3
    assert [w.at for w in newest] == pytest.approx([8.0, 5.0, 2.0, 59.0, 29.0], abs=1e-3), "each run's first frame in scan order"
    assert [w.at for w in oldest] == pytest.approx([0.0, 30.0, 0.0, 3.0, 6.0], abs=1e-3)
    assert [w.t0 - T0 for w in newest] == pytest.approx([106, 103, 100, 30, 0], abs=0.01), "each over the span of its frames"
    assert [w.t1 - T0 for w in newest] == pytest.approx([108, 105, 102, 59, 29], abs=0.01)
    assert [w.t1 for w in newest] == sorted((w.t1 for w in newest), reverse=True)
    assert [w.t0 for w in oldest] == sorted(w.t0 for w in oldest)


def test_still_s_is_cut_to_max_sighting_s_so_no_window_outgrows_the_cap(home, api, store, media):
    index(store, "cam", "video", media["red60"], T0, LONG)
    capped = scan(api, question="Is it red?", limit=5, max_sighting_s=20)
    assert capped["progress"]["windows"] == 3, "20 s a window, not 30"
    assert [s.get("capped", False) for s in capped["sightings"]] == [True, True, False]
    assert all(span(s)[1] - span(s)[0] <= 20 for s in capped["sightings"]), [span(s) for s in capped["sightings"]]
    assert [round(span(s)[0] - T0) for s in capped["sightings"]] == [40, 20, 0], "newest first, each a window's span"
    assert scan(api, question="Is it red?", limit=5, max_sighting_s=20, still_s=5)["progress"]["windows"] == 12, "a smaller still_s stays"
    assert scan(api, question="Is it red?", max_sighting_s=0)["progress"]["windows"] == 2, "no cap, no cut"


@pytest.mark.parametrize("newest_first", [True, False], ids=["backwards", "forwards"])
def test_a_videos_still_runs_and_its_soundtrack_come_in_order_together(home, store, media, newest_first):
    item = index(store, "cam", "video", media["talk"], T0, 20, meta={"sound": True})
    found = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=newest_first, profile={"still_s": 8}, scope=EVERYTHING))
    assert all(w.item.id == item for w in found)
    kinds = Counter(w.kind for w in found)
    assert kinds["video"] == 3 and kinds["audio"] >= 7, "three runs of frames, and a sound window every 2.5 s"
    keys = [w.t1 if newest_first else w.t0 for w in found]
    assert keys == sorted(keys, reverse=newest_first), "merged by the end going newest first, by the start going oldest first"
    assert [w.at for w in found if w.kind == "video"] == pytest.approx([19.0, 11.0, 3.0] if newest_first else [0.0, 8.0, 16.0], abs=1e-3)


def test_a_sightings_program_time_is_that_of_the_picture_its_handle_shows(home):
    """A run scanned newest first starts at its newest frame, which is the picture; its t0 is the oldest frame's."""
    ask = jobs.load_script(home, "ask")
    seen = []
    store = SimpleNamespace(project_name=lambda project: None, has_clock=lambda source: True,
                            program_time=lambda source, t: (seen.append(t), ("r1", 5.0))[1])
    segment = SimpleNamespace(id=7, kind="video", t0=T0, project=None)
    version = SimpleNamespace(id=8, kind="file", t0=T0 + 100, project=None)
    windows = [reader.Window(source="prog", item=segment, t0=T0, t1=T0 + 29, at=29.0, kind="video"),
               reader.Window(source="prog", item=version, t0=T0 + 100, t1=T0 + 100, at=12.0, kind="video"),
               reader.Window(source="prog", records=[], t0=T0 + 200, t1=T0 + 210, kind="records")]
    for w in windows:
        shown = ask._public(SimpleNamespace(store=store), {"source": "prog", "t0": w.t0, "t1": w.t1, "p": 0.9, "said": None,
                                                           "capped": False, "window": w})
        assert shown["run"] == "r1" and shown["tp"] == 5.0
    assert seen == [T0 + 29, T0 + 100, T0 + 200], "a segment's frame at `at`, else the window's start"


def test_long_text_is_cut_at_paragraphs_and_the_hit_is_in_its_window(home, store):
    paragraphs = [f"paragraph {i} " + "x" * 1900 for i in range(10)]  # 1,900+ characters each
    store.add_version("notes", "long.txt", T0, data="\n\n".join(paragraphs).encode())
    found = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=False, scope=EVERYTHING))
    assert 3 <= len(found) <= 5
    assert all(w.kind == "text" and len(w.text) <= 6000 for w in found)
    assert "".join(w.text for w in found) == "\n\n".join(paragraphs)
    assert all(w.text.endswith("\n\n") for w in found[:-1])  # cut after a paragraph break
    assert list(reversed(found)) == list(reader.windows(store, about=decider_d1.ABOUT, scope=EVERYTHING))  # newest first reads the end first


def test_records_are_asked_in_windows_of_3000_characters_and_the_sighting_spans_the_window(home, api, store):
    pad = "z" * 1100  # two records fill a window of 3,000 characters
    rows = [(T0 + 10 * i, json.dumps({"n": i, "note": "the kettle is on" if i == 4 else "nothing", "pad": pad}))
            for i in range(6)]
    store.add_records("thermo", rows)
    runs = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=False, scope=EVERYTHING))
    assert [(w.t0, w.t1) for w in runs] == [(T0, T0 + 10), (T0 + 20, T0 + 30), (T0 + 40, T0 + 50)]
    assert all(w.kind == "records" and w.item is None and len(w.records) == 2 and len(w.text) <= 3000 for w in runs)
    result = scan(api, question="contains:kettle", sources=["thermo"])
    (s,) = result["sightings"]
    assert span(s) == (pytest.approx(T0 + 40, abs=0.001), pytest.approx(T0 + 50, abs=0.001))
    assert s["look"]["source"] == "thermo" and s["kind"] == "records"
    assert store.labels(by="ask")[0].item is None
    assert all(call["state_len"] <= 3000 for call in decider_log(home)), "d1 read at most 3,000 characters of records"


def test_a_choice_from_answers_gets_a_none_option_that_never_counts(home, api, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    result = scan(api, question="What colour is it? red", answers=["red", "blue"])
    (s,) = result["sightings"]
    assert span(s)[0] == pytest.approx(T0 + 3, abs=0.3), "the blue seconds picked none: no hit"
    assert s["said"] == "red", "a choice's said is its pick"
    options = {tuple(q["options"]) for call in decider_log(home) for q in call["questions"].values()}
    assert options == {("red", "blue", "none")}
    named = scan(api, question="What colour is it? red", answers=["red", "none"])
    assert named["sightings"] and {tuple(q["options"]) for call in decider_log(home)[-3:]
                                   for q in call["questions"].values()} == {("red", "none")}, "an answer named none is kept"


def test_a_questions_object_with_several_questions_needs_every_hit(home, api, store, media):
    index(store, "cam", "video", media["video"], T0, SECONDS)
    both = {"red": {"type": "noul", "instructions": "red?", "hit": 0.9},
            "level": {"type": "score", "instructions": "red?", "criteria": ["no", "a bit", "very"], "hit": 2}}
    assert len(scan(api, questions=both)["sightings"]) == 1
    clash = {"red": {"type": "noul", "instructions": "red?", "hit": True},
             "kettle": {"type": "noul", "instructions": "contains:kettle", "hit": True}}
    assert scan(api, questions=clash)["sightings"] == []
    assert all(len(call["questions"]) == 2 for call in decider_log(home)), "every question in one call per window"


def test_the_decider_gets_pictures_within_640x360_and_never_more_than_the_state_it_needs(home, api, store, media):
    index(store, "hd", "video", media["hd"], T0, 2)
    index(store, "cam", "video", media["video"], T0 + 100, SECONDS)
    store.add_version("pics", "wide.png", T0 + 200, data=_png(1280, 720))
    store.add_version("pics", "tall.png", T0 + 210, data=_png(400, 1000))
    store.add_version("pics", "small.png", T0 + 220, data=_png(100, 50))
    scan(api, question="Is it red?", every_s=0)
    sizes = {tuple(size) for call in decider_log(home) for size in call["pictures"]}
    assert (640, 360) in sizes and (160, 90) in sizes and (100, 50) in sizes, "never upscaled"
    assert all(w * h <= 230_400 for w, h in sizes)
    assert (400, 1000) not in sizes
    tall = next(size for size in sizes if size[1] > size[0])
    assert abs(tall[0] / tall[1] - 0.4) < 0.01, "aspect kept"
    assert all(len(call["pictures"]) <= 1 for call in decider_log(home))


def _png(width: int, height: int) -> bytes:
    import io
    out = io.BytesIO()
    Image.new("RGB", (width, height), (200, 10, 10)).save(out, format="PNG")
    return out.getvalue()


def test_a_remembered_recording_is_asked_at_its_real_time(home, api, media):
    stored = api.call("remember", path=str(media["video"]), t=T0 + 500, source="clips")
    assert stored["stored"][0]["time_from"] == "given"
    result = scan(api, question="Is it red?", sources=["clips"])
    (s,) = result["sightings"]
    assert span(s)[0] == pytest.approx(T0 + 503, abs=0.3) and s["look"]["item"] == stored["stored"][0]["id"]
    heard = api.call("remember", path=str(media["audio"]), t=T0 + 600, source="clips")
    (s,) = scan(api, question="Is it loud?", sources=["clips"])["sightings"]
    assert span(s)[0] <= T0 + 600 + TONE[0] and s["look"]["item"] == heard["stored"][0]["id"]


def test_a_video_version_is_read_as_media_at_the_versions_time(home, api, store, media):
    version = store.add_version("clips", "colours.mkv", T0, data=media["video"].read_bytes())
    windows = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=False, scope=EVERYTHING))
    assert len(windows) == 3 and all(w.kind == "video" and w.t0 == w.t1 == T0 for w in windows), "a run of a colour, all at t0"
    assert [w.at for w in windows] == pytest.approx([0.0, 3.0, 6.0], abs=0.3)
    every = list(reader.windows(store, about=decider_d1.ABOUT, newest_first=False, profile={"still_s": 0}, scope=EVERYTHING))
    assert len(every) == 9 and all(w.t0 == w.t1 == T0 for w in every)
    assert [w.at for w in every] == pytest.approx([float(i) for i in range(9)], abs=0.3)
    result = scan(api, question="Is it red?", since=T0 + 5, until=T0 + 6)
    assert [s["look"]["item"] for s in result["sightings"]] == [version], "saved before the range, unchanged in it"


def test_the_lints(home, api, store, media):
    index(store, "cam", "video", media["red60"], T0, LONG)
    index(store, "mic", "audio", media["audio"], T0, 19)

    def warned(text: str, **params) -> bool:
        return any(text in w for w in scan(api, **{"question": "Is it red?", **params}).get("warnings", []))

    assert warned("time word", question="Is it red yesterday?")
    assert warned("time word", question="Is it red?", context="the screen at 3pm")
    assert warned("joined", question="Is it red or blue?")
    assert not warned("joined") and not warned("time word")
    job = api.call("jobs/start", script="ask", params={"question": "Is it red?"}, wait_s=50)  # no since
    assert any("no since" in w for w in job["result"]["warnings"])
    many = {f"q{i}": {"type": "noul", "instructions": f"red {i}?", "hit": 0.5} for i in range(17)}
    assert any("17 questions" in w for w in scan(api, questions=many)["warnings"])
    text = {"t": {"type": "text", "instructions": "What does it say?", "hit": True}}
    assert any("text, date or number" in w for w in scan(api, questions=text, sources=["cam"])["warnings"])
    slow = scan(api, question="Is it red?", max_s=1, sources=["cam"])
    assert any("max_s" in w and "narrow" in w for w in slow["warnings"]), "60 windows at 0.1 s are more than 1 s"
    assert not any("max_s" in w for w in scan(api, question="Is it red?", sources=["mic"])["warnings"])
    assert warned("speech", question="What was said?", sources=["mic"])


def test_d1s_cost_model_is_unchanged(home):
    """`window_seconds` under d1's About is the formula ask had before the facts moved to the model, within 1e-4 s."""
    ask = jobs.load_script(home, "ask")

    def before(kind: str, profile: dict) -> float:
        if kind in ("video", "image"):
            pixels = min(profile["max_px"] or 1_638_400, 1_638_400)
            return 0.053 + 0.00032 * max(0.0, pixels / 1047 - 220) + 0.010
        if kind == "audio":
            return 0.05
        return 0.044 / 2000 * profile["text_chars" if kind == "text" else "records_chars"]

    for max_px in (230_400, 921_600, 0, 5_000_000):  # the last two are the model's cap
        profile = reader.profile_of({"max_px": max_px}, decider_d1.ABOUT)
        for kind in ("video", "image"):
            assert ask.window_seconds(kind, profile, decider_d1.ABOUT) == pytest.approx(before(kind, profile), abs=1e-4)
    profile = {**reader.PROFILE, "text_chars": 6000}
    for kind in ("audio", "text"):
        assert ask.window_seconds(kind, profile, decider_d1.ABOUT) == pytest.approx(before(kind, profile), abs=1e-4)
    assert ask.window_seconds("text", profile, decider_d1.ABOUT) == pytest.approx(0.132, abs=1e-4), "6000 characters"


def test_a_pictures_cost_is_the_cost_at_the_models_cap_for_max_px_0_and_for_more(home):
    """The cost keeps rising past the cap here, so an estimate made at the asked size and not the cap would show."""
    ask = jobs.load_script(home, "ask")
    about = About(name="x", max_px=500_000, picture_s=((230_400, 0.05), (2_000_000, 1.0)))
    at_cap = about.picture_seconds(500_000) + ask.DECODE_S
    assert at_cap > about.picture_seconds(230_400) + ask.DECODE_S
    for asked in (0, 500_000, 2_000_000):
        for kind in ("video", "image"):
            profile = reader.profile_of({"max_px": asked}, about)
            assert ask.window_seconds(kind, profile, about) == pytest.approx(at_cap), (asked, kind)
    smaller = reader.profile_of({"max_px": 300_000}, about)
    assert ask.window_seconds("video", smaller, about) == pytest.approx(about.picture_seconds(300_000) + ask.DECODE_S)
    uncapped = dataclasses.replace(about, max_px=0)
    assert ask.window_seconds("video", reader.profile_of({"max_px": 0}, uncapped), uncapped) ==         pytest.approx(about.picture_seconds(2_000_000) + ask.DECODE_S), "a model with no cap: 0 is as stored, the dearest point"


def test_max_s_stops_a_slow_scan_with_the_windows_it_has_not_read(home, api, store):
    for i in range(10):
        store.add_version("notes", f"n{i}.md", T0 + 60 * i, data=b"nothing here\n")
    result = scan(api, question="contains:kettle slow:0.3", max_s=1, until=T0 + 1000)
    assert result["stopped"] == "max_s" and 1 <= result["progress"]["windows"] < 10, "0.3 s a window against 1 s"


def test_max_s_does_not_count_the_time_d1_takes_to_load(home, api, store):
    for i in range(10):
        store.add_version("notes", f"n{i}.md", T0 + 60 * i, data=b"nothing here\n")
    result = scan(api, question="contains:kettle load:1.5", max_s=1, until=T0 + 1000)  # the first answer takes 1.5 s
    assert result["stopped"] == "range_end" and result["progress"]["windows"] == 10
    assert result["progress"]["seconds"] >= 1.5, "progress.seconds is still the job's wall time"


def test_the_default_scope_reads_ambient_rows_and_the_callers_project_not_another_projects(home, api, store, media):
    mine, theirs = "git:aaaaaaaaaaaaaaaa", "git:bbbbbbbbbbbbbbbb"
    ambient = index(store, "cam", "video", media["video"], T0, SECONDS)
    own = index(store, "cam2", "video", media["video"], T0 + 100, SECONDS, project=mine)
    other = index(store, "cam3", "video", media["video"], T0 + 200, SECONDS, project=theirs)
    caller = {"project": mine, "project_name": "mine", "project_root": "/x", "session": "s-0123456789ab", "client": "t"}

    def run(**params):
        job = api.call("jobs/start", script="ask", wait_s=50, caller=caller,
                       params={"question": "Is it red?", "since": T0 - 100, "until": T0 + 1000, "limit": 10, **params})
        while job["state"] in ("queued", "running"):
            job = api.call("jobs/get", job=job["job"], wait_s=50)
        assert job["state"] == "done", job
        return job["result"]

    default = run()
    assert {s["look"]["item"] for s in default["sightings"]} == {ambient, own}
    assert default["searched"]["scope"] == "default" and default["searched"]["project"] == {"id": mine, "name": "mine"}
    assert "cam3" not in default["searched"]["sources"] and "cam2" in default["searched"]["sources"]
    assert {s.get("project") for s in default["sightings"]} == {None, "mine"}, "project only when there is one"
    assert all(("project" in s) == (s["look"]["item"] == own) for s in default["sightings"])
    everything = run(scope="all")
    assert {s["look"]["item"] for s in everything["sightings"]} == {ambient, own, other}
    assert everything["searched"]["scope"] == "all" and "cam3" in everything["searched"]["sources"]
    assert {label.project for label in store.labels(by="ask")} >= {mine}
    assert [s["look"]["item"] for s in run(sources=["cam3"])["sightings"]] == [], "narrowing never widens the scope"


def test_an_unreadable_item_is_counted_in_the_warnings_and_the_scan_goes_on(home, api, store, media):
    for i in range(8):  # segments whose files retention (or anything else) took
        store.add_item("lost", "video", T0 + i, T0 + i + 5, f"streams/lost/{i}.mkv", 10)
    good = index(store, "cam", "video", media["video"], T0 + 100, SECONDS)
    result = scan(api, question="Is it red?", since=T0 - 10, until=T0 + 200)
    assert [s["look"]["item"] for s in result["sightings"]] == [good], "the readable item was still read"
    unreadable = [w for w in result["warnings"] if "unreadable" in w]
    assert len([w for w in unreadable if w.startswith("item ")]) == 5 and all("item " in w for w in unreadable[:5])
    assert unreadable[5] == "and 3 more items unreadable", "five named, the rest counted"
    assert any("FileNotFoundError" in w for w in unreadable[:5]), "and why"


def test_a_job_can_override_the_readers_profile_for_its_own_windows(home, api, store, media):
    index(store, "hd", "video", media["hd"], T0, 2)
    (home / "scripts").mkdir(exist_ok=True)
    (home / "scripts" / "sizes.py").write_text(
        "def run(job):\n"
        "    w = next(job.windows(job.params['since'], job.params['until'], profile=job.params.get('profile')))\n"
        "    job.update(size=list(w.media.size))\n", encoding="utf-8")

    def size(**more) -> list[int]:
        job = api.call("jobs/start", script="sizes", params={"since": T0 - 1, "until": T0 + 10, **more}, wait_s=40)
        assert job["state"] == "done", job
        return job["result"]["size"]

    cap = list(reader.fit(1920, 1080, decider_d1.ABOUT.max_px))
    assert size() == [640, 360], "the reader's own profile"
    assert size(profile={"max_px": 0}) == cap != [1920, 1080], "0 is the model's cap, not as stored"
    assert size(profile={"max_px": 4 * decider_d1.ABOUT.max_px}) == cap, "above the cap is the cap"
    assert size(profile={"max_px": 57_600}) == [320, 180]
    job = api.call("jobs/start", script="sizes", params={"since": T0 - 1, "until": T0 + 10, "profile": {"nope": 1}}, wait_s=40)
    assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and "nope" in job["error"]["message"]


def test_a_yes_no_hit_of_true_counts_at_the_models_own_hit_and_the_asker_can_still_choose(home, tmp_path, served, store):
    store.add_version("notes", "low.md", T0, data=b"nothing grade:0.5\n")
    store.add_version("notes", "high.md", T0 + 60, data=b"nothing grade:0.7\n")
    api = api_under(home, served, facade(tmp_path, "loose", hit=0.6))
    result = scan(api, question="contains:kettle")
    (s,) = result["sightings"]
    assert s["p"] == 0.7 and "said" not in s, "a 0.7 window hits at the model's 0.6"
    assert store.labels(by="ask")[0].text.startswith("contains:kettle: yes (p 0.70), "), "the label keeps the whole answer"
    assert [c["p"] for c in result["candidates"]] == [0.5]
    stricter = scan(api, question="contains:kettle", hit=0.8)
    assert stricter["sightings"] == [] and [c["p"] for c in stricter["candidates"]] == [0.7, 0.5]
    asked = {"k": {"type": "noul", "instructions": "contains:kettle", "hit": True}}
    assert [x["p"] for x in scan(api, questions=asked)["sightings"]] == [0.7], "true is the model's hit in questions too"


def test_a_model_with_the_default_hit_of_d1_counts_at_0_8(home, api, store):
    store.add_version("notes", "n.md", T0, data=b"nothing grade:0.7\n")
    result = scan(api, question="contains:kettle")
    assert result["sightings"] == [] and [c["p"] for c in result["candidates"]] == [0.7], "0.7 is below the hit of 0.8"


def test_a_yes_no_hit_sent_as_text_is_the_threshold_it_names(home, api, store):
    store.add_version("notes", "n.md", T0, data=b"nothing grade:0.7\n")
    (s,) = scan(api, question="contains:kettle", hit="0.6")["sightings"]  # as an agent sent "hit": "0.5" in the benchmark
    assert s["p"] == 0.7, "\"0.6\" is 0.6, below the window's 0.7 (d1's own 0.8 would find nothing)"
    assert scan(api, question="contains:kettle", hit="0.75")["sightings"] == [], "\"0.75\" is 0.75, above the window's 0.7"


NARROW = dict(types=("noul", "choice", "score"), question_keys=frozenset({"criteria"}))


@pytest.mark.parametrize("facts,questions,words", [
    (NARROW, {"x": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "none": None}, "multiple": True,
                    "hit": "a"}},
     ["question 'x' has multiple, which narrow does not take; it takes criteria (besides type, instructions and hit)"]),
    (dict(question_keys=frozenset()), {"x": {"type": "noul", "instructions": "Is it red?",
                                             "criteria": {"true": "red", "false": "not red"}, "hit": True}},
     ["question 'x' has criteria, which narrow does not take; it takes only type, instructions and hit"]),
    (NARROW, {"x": {"type": "number", "instructions": "How many?", "hit": True}},
     ["question 'x' is a number, which narrow does not answer; it answers noul, choice, score"]),
    (NARROW, {"x": {"type": "noul", "instructions": "Is it red?", "hit": True, "legend": "red"}},
     ["question 'x' has legend, which a question does not take; a question takes type, instructions, hit, criteria, "
      "multiple, range, step, unit"]),
    ({}, {"x": {"type": "noul", "instructions": "Is it red?", "hit": True, "unit": "x", "weight": 2}},
     ["question 'x' has weight, which a question does not take"]),
])
def test_a_question_the_model_does_not_take_fails_the_job_before_a_window_is_scanned(home, tmp_path, served, store, facts,
                                                                                     questions, words):
    store.add_version("notes", "n.md", T0, data=b"the kettle\n")
    api = api_under(home, served, facade(tmp_path, "narrow", **facts))
    job = api.call("jobs/start", script="ask", params={"questions": questions, "since": T0 - 10, "until": T0 + 10}, wait_s=40)
    assert job["state"] == "failed" and job["error"]["code"] == "bad_request", job
    assert all(text in job["error"]["message"] for text in words), job["error"]["message"]
    assert decider_log(home) == [], "no window was scanned"
    assert api.call("status")["decider"]["state"] == "unloaded", "and the model was never loaded"


@pytest.mark.parametrize("name,questions,words", [
    ("d3-mini", {"x": {"type": "choice", "instructions": "Which lamp is on?", "criteria": {"": "none", "a": None}, "hit": "a"}},
     ["question 'x'", "empty option name"]),
    ("d3-mini", {"x": {"type": "score", "instructions": "How tidy?", "criteria": ["messy", "messy", "tidy"], "hit": 2}},
     ["question 'x'", "distinct and not empty"]),
    ("h2o-lightning-4b", {"x": {"type": "box", "instructions": "the red mug", "criteria": {"a": None}, "hit": True}},
     ["question 'x' has criteria", "h2o-lightning-4b does not take for a box question"]),
])
def test_a_question_a_real_model_refuses_fails_the_job_before_a_window_is_scanned(home, tmp_path, served, store, name,
                                                                                  questions, words):
    store.add_version("notes", "n.md", T0, data=b"the kettle\n")
    folder = model_folder(tmp_path / "models", name)
    prepared_env(tmp_path / "models" / "envs", "d3", decider_d3.D3.requirements)
    api = api_under(home, served, folder.as_posix())
    job = api.call("jobs/start", script="ask", params={"questions": questions, "since": T0 - 10, "until": T0 + 10}, wait_s=40)
    assert job["state"] == "failed" and job["error"]["code"] == "bad_request", job
    assert all(text in job["error"]["message"] for text in words), job["error"]["message"]
    assert not job["result"].get("progress", {}).get("windows"), "no window was scanned"
    assert api.call("status")["decider"]["state"] == "unloaded", "and the model was never loaded"


def test_a_key_or_type_outside_the_model_is_also_refused_in_the_question_form(home, tmp_path, served, store):
    store.add_version("notes", "n.md", T0, data=b"the kettle\n")
    api = api_under(home, served, facade(tmp_path, "narrow", **NARROW))
    job = api.call("jobs/start", script="ask", params={"question": "contains:kettle", "answers": ["a", "b"], "since": T0 - 10},
                   wait_s=40)
    assert job["state"] == "done" and job["error"] is None, "a choice from answers has only criteria"
    for question, text in (({"type": "text", "instructions": "x", "hit": True}, "is a text"),
                           ({"type": "noul", "instructions": "x", "hit": True, "range": [0, 1]}, "has range")):
        job = api.call("jobs/start", script="ask", params={"questions": {"q": question}, "since": T0 - 10}, wait_s=40)
        assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and text in job["error"]["message"], job


def test_a_window_costs_a_pass_for_each_questions_per_read_questions(home):
    ask = jobs.load_script(home, "ask")
    one = dataclasses.replace(decider_d1.ABOUT, questions_per_read=1)
    profile = reader.profile_of(None, one)
    for kind in ("audio", "text", "records", "file"):
        assert ask.window_seconds(kind, profile, one, 2) == pytest.approx(2 * ask.window_seconds(kind, profile, one))
        assert ask.window_seconds(kind, profile, one, 5) == pytest.approx(5 * ask.window_seconds(kind, profile, one))
    for kind in ("video", "image"):  # the decoding is once, the model's reading is per pass
        model = ask.window_seconds(kind, profile, one) - ask.DECODE_S
        assert ask.window_seconds(kind, profile, one, 2) - ask.DECODE_S == pytest.approx(2 * model)
    sixteen = decider_d1.ABOUT
    assert sixteen.questions_per_read == 16
    for questions, passes in ((1, 1), (16, 1), (17, 2), (32, 2), (33, 3)):
        assert ask.window_seconds("audio", profile, sixteen, questions) == pytest.approx(passes * sixteen.sound_s)
    anything = dataclasses.replace(sixteen, questions_per_read=0)
    assert ask.window_seconds("audio", profile, anything, 100) == anything.sound_s, "0 is any number in one pass"


def lint_of(home: Path, about, questions: int, counts: dict, **params) -> list[str]:
    ask = jobs.load_script(home, "ask")
    asked = {f"q{i}": {"type": "noul", "instructions": f"red {i}?", "hit": True} for i in range(questions)}
    plan = ask.Plan({"questions": asked, "since": T0 - 10, "until": T0, **params}, T0)
    return ask.lint(plan, {"video": 0, "audio": 0, "image": 0, "file": 0, "text": 0, "records": 0, **counts},
                    reader.profile_of(None, about), about)


def test_the_estimate_counts_every_pass_and_the_lint_says_a_model_reads_one_question_a_pass(home):
    one = dataclasses.replace(decider_d1.ABOUT, name="solo", questions_per_read=1, sound_s=0.05)
    assert not [w for w in lint_of(home, one, 1, {"audio": 100}, max_s=6) if "max_s" in w], "100 x 0.05 s is 5 s"
    (warning,) = [w for w in lint_of(home, one, 2, {"audio": 100}, max_s=6) if "max_s" in w]
    assert warning.startswith("about 100 windows, about 10 s against max_s of 6 s: narrow"), "two questions, twice the time"
    (many,) = [w for w in lint_of(home, one, 3, {}) if "questions" in w]
    assert many == "3 questions: solo reads a window once for each question, so this takes 3 times as long; ask fewer"
    assert not [w for w in lint_of(home, one, 1, {}) if "questions" in w]
    sixteen = lint_of(home, decider_d1.ABOUT, 17, {})
    assert ("17 questions: d1-Omni-2B reads the window again for each 16, so this takes several times as long; ask fewer"
            in sixteen)


def test_the_estimate_warning_says_when_the_models_costs_are_assumed(home):
    assumed = dataclasses.replace(decider_d1.ABOUT, measured=frozenset({"sound_s"}))
    measured = dataclasses.replace(decider_d1.ABOUT, measured=frozenset({"picture_s", "sound_s", "char_s", "hit"}))
    counts = {"video": 100_000}
    (said,) = [w for w in lint_of(home, assumed, 1, counts) if "max_s" in w]
    assert said.endswith("or raise every_s (its costs are assumed)"), said
    (said,) = [w for w in lint_of(home, measured, 1, counts) if "max_s" in w]
    assert said.endswith("or raise every_s"), said
