"""A decider that reads less than d1 (TESTBED.md, "No GPU"; DESIGN.md §13.3), run end to end through the daemon and `ask`:
an eyes-only one (`reads={"video", "image"}`) and an ears-only one (`reads={"audio"}`), each a `python:` file written
into a tmp folder that is otherwise `fake_decider.py`. The store holds what a home does: a mic segment (sound alone), a
screen segment (video, no sound), a camera segment (video with sound, registered as capture registers it), a picture, a
text and a PDF. Each test reads what a decider was handed from its log, what the job says it left unread or read in part
(a camera's soundtrack an eyes-only decider cannot hear, or its frames an ears-only one cannot see), and whether
`progress.of_about` (the reader's estimate) agrees with the windows read."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import ROOT, Served, write_config
from corpus import EVERYTHING, GAP, OUT, T0, X264, run, scan

from mms import decider, reader
from mms.capture import segment_meta
from mms.store import Store

BASE = "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 50\n"
SEGMENT_S = 2  # a sound shorter than the 5 s audio window is one window, which the estimate counts as one too
SOURCES = ["mic", "screen", "cam", "pics", "notes"]  # the media and the text; "docs" holds the PDF
PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"

# A decider file: the facts of the narrow decider, and `fake_decider.py` (loaded by its path, since a daemon in a
# subprocess has no tests/ on sys.path) for the answers and the log it keeps in the home.
FILE = """\
import importlib.util

from mms.decider import About

ABOUT = About(name={name!r}, reads={reads!r})
home = None
_fake = None


def decide(state, questions, media):
    global _fake
    if _fake is None:
        found = importlib.util.spec_from_file_location("fake_decider_of_{name}", {path!r})
        _fake = importlib.util.module_from_spec(found)
        found.loader.exec_module(_fake)
    _fake.home = home
    return _fake.decide(state, questions, media)
"""


def decider_file(folder: Path, name: str, reads: set) -> str:
    """The `python:` model of a decider that reads `reads` of the media window kinds."""
    path = folder / f"{name}.py"
    path.write_text(FILE.format(name=name, reads=reads, path=str(ROOT / "tests" / "fake_decider.py")), encoding="utf-8")
    return f"python:{path.as_posix()}"


@pytest.fixture
def filled(home, tmp_path):
    """The store, and the id of each item by what it is."""
    from PIL import Image
    tone = ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={SEGMENT_S}"]
    red = ["-f", "lavfi", "-i", f"color=c=red:s=160x90:r=5:d={SEGMENT_S}"]
    opus = ["-c:a", "libopus", "-b:a", "24k"]
    clips = tmp_path / "clips"
    clips.mkdir()
    run(*tone, *opus, clips / "mic.mka")
    run(*red, *OUT, *X264, clips / "screen.mkv")
    run(*red, *tone, *OUT, *X264, *opus, clips / "cam.mkv")
    picture = clips / "card.png"
    Image.new("RGB", (160, 90), (200, 10, 10)).save(picture)

    store = Store(home)
    ids = {}
    for n, (source, kind, file) in enumerate([("mic", "audio", "mic.mka"), ("screen", "video", "screen.mkv"),
                                              ("cam", "video", "cam.mkv")]):
        t0 = T0 + n * GAP
        target = store.streams / source / "launch" / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((clips / file).read_bytes())
        ids[source] = store.add_item(source, kind, t0, t0 + SEGMENT_S, store.relative(target), target.stat().st_size,
                                     meta=segment_meta(target))  # no media type, as a captured segment has none
    ids["pics"] = store.add_version("pics", "card.png", T0 + 3 * GAP, data=picture.read_bytes())
    ids["notes"] = store.add_version("notes", "note.txt", T0 + 4 * GAP, data=b"the kettle is on\n")
    ids["docs"] = store.add_version("docs", "invoice.pdf", T0 + 5 * GAP, data=PDF)
    store.close()
    return ids


class Run:
    """One daemon under one decider, over the home `filled` made; `ask()` scans it all, with the window kinds it read."""

    def __init__(self, home: Path, spec: str):
        self.home, self.spec = home, spec
        (home / "fake-decider.jsonl").unlink(missing_ok=True)
        write_config(home, BASE + f'[decider]\nmodel = "{spec}"\n')
        self.served = Served(home).start()

    def ask(self, sources=None) -> dict:
        params = {"question": "Is there anything red?", "limit": 50, "scope": "all", "still_s": 0}  # a window a frame, as the estimate counts
        if sources is not None:
            params["sources"] = sources
        return scan(self.served.api, T0 + GAP * 6, **params)

    def calls(self) -> list[dict]:
        path = self.home / "fake-decider.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def stop(self) -> None:
        self.served.stop()


@pytest.fixture
def under(home):
    runs: list[Run] = []

    def start(spec: str) -> Run:
        while runs:
            runs.pop().stop()  # one daemon at a time, on the one home
        runs.append(Run(home, spec))
        return runs[-1]

    yield start
    for one in runs:
        one.stop()


def not_read(result: dict) -> str:
    """The one warning that says what was not read (and none other is a surprise)."""
    (warning,) = [w for w in result["warnings"] if "not read" in w]
    return warning


def partial(result: dict, what: str) -> list[str]:
    """The warnings that say a video was read in part: `what` is "sound" (not heard) or "frames" (not seen)."""
    return [w for w in result["warnings"] if w.startswith(f"the {what} of ")]


def groups(warning: str) -> dict[str, set[int]]:
    """The groups the not-read warning names, each with the ids it lists."""
    body = warning.split(": ", 1)[1].rsplit("; ", 1)[0]
    return {group: {int(n) for n in re.findall(r"\d+", ids)}
            for group, ids in re.findall(r"(?:^|, )([^,()]+?) \(items? ([\d, ]+?)(?: and \d+ more)?\)", body)}


def estimated(home: Path, spec: str) -> dict:
    """What the reader's estimate says of the home's whole range under the decider a model spec makes."""
    store = Store(home)
    try:
        return reader.estimate(store, None, None, None, EVERYTHING, about=decider.load_decider(spec).about)
    finally:
        store.close()


def test_an_eyes_only_decider_is_given_frames_and_the_picture_and_never_a_sound(home, tmp_path, filled, under):
    spec = decider_file(tmp_path, "eyes", {"video", "image"})
    one = under(spec)
    result = one.ask(SOURCES)

    # the mic segment is the one thing it leaves, called what it is; look shows it
    warning = not_read(result)
    assert warning == (f"1 item was not read, since eyes does not read its type: audio segments (item {filled['mic']}); "
                       "find lists it and look shows it"), warning
    assert partial(result, "sound") == [
        f"the sound of 1 video item was not heard, since eyes does not hear: cam (item {filled['cam']}); find lists it"]
    assert len(result["warnings"]) == 2, f"nothing else was said: {result['warnings']}"

    calls = one.calls()
    assert all(call["sounds"] == [] for call in calls), "eyes is never handed a sound, not the camera segment's soundtrack"
    assert sum(len(call["pictures"]) for call in calls) == 2 * SEGMENT_S + 1, \
        "a frame a second of the screen segment and of the camera segment, and the picture"
    assert {tuple(size) for call in calls for size in call["pictures"]} == {(160, 90)}
    assert sum(1 for call in calls if not call["pictures"]) == 1, "and the text, which every decider reads"

    estimate = estimated(home, spec)
    assert estimate == {"video": 2 * SEGMENT_S, "audio": 0, "image": 1, "file": 0, "text": 1, "records": 0}, estimate
    assert result["progress"]["windows"] == result["progress"]["of_about"] == sum(estimate.values()) == len(calls)


def test_an_ears_only_decider_hears_the_mic_and_the_camera_soundtrack_and_sees_nothing(home, tmp_path, filled, under):
    spec = decider_file(tmp_path, "ears", {"audio"})
    one = under(spec)
    result = one.ask(SOURCES)

    # what it cannot hear is the screen segment (no sound) and the picture; both are shown by look
    warning = not_read(result)
    assert warning.startswith("2 items were not read, since ears does not read their types: "), warning
    assert warning.endswith("; find lists them and look shows them"), warning
    assert groups(warning) == {"video segments": {filled["screen"]}, "image/png": {filled["pics"]}}, warning
    assert partial(result, "frames") == [
        f"the frames of 1 video item were not seen, since ears does not see: cam (item {filled['cam']}); find lists it"]
    assert len(result["warnings"]) == 2, f"nothing else was said: {result['warnings']}"

    calls = one.calls()
    assert all(call["pictures"] == [] for call in calls), "ears is never handed a picture or a frame"
    heard = [heard for call in calls for heard in call["heard"]]
    assert len(heard) == 2, "the mic segment's sound, and the camera segment's soundtrack (it has meta.sound)"
    assert {sound["rate"] for sound in heard} == {16000} and all(sound["rms"] > 0.01 for sound in heard)
    assert sum(1 for call in calls if not call["sounds"]) == 1, "and the text, which every decider reads"

    estimate = estimated(home, spec)
    assert estimate == {"video": 0, "audio": 2, "image": 0, "file": 0, "text": 1, "records": 0}, estimate
    assert result["progress"]["windows"] == result["progress"]["of_about"] == sum(estimate.values()) == len(calls)


def test_a_file_among_what_is_not_read_changes_what_the_warning_says_look_does(home, tmp_path, filled, under):
    for name, reads, left in (("eyes", {"video", "image"}, {"audio segments": {"mic"}}),
                              ("ears", {"audio"}, {"video segments": {"screen"}, "image/png": {"pics"}})):
        one = under(decider_file(tmp_path, name, reads))
        warning = not_read(one.ask())  # every source now, the PDF's too
        total = sum(len(keys) for keys in left.values()) + 1
        assert warning.startswith(f"{total} items were not read, since {name} does not read their types: "), warning
        assert groups(warning) == {**{group: {filled[key] for key in keys} for group, keys in left.items()},
                                   "application/pdf": {filled["docs"]}}, warning
        assert warning.endswith("; find lists them and look shows a picture, sound or video and gives a copy of any other "
                                "file"), "a mix of media and a file says both"


def test_an_ears_only_decider_leaves_a_segment_whose_sound_was_never_recorded(home, tmp_path, filled, under):
    """What `meta.sound` is for: a video segment registered without it is taken as having no sound."""
    store = Store(home)
    cam = store.item(filled["cam"])
    with store.tx() as db:
        db.execute("UPDATE items SET meta = NULL WHERE id = ?", (cam.id,))
    store.close()
    one = under(decider_file(tmp_path, "ears", {"audio"}))
    warning = not_read(one.ask(SOURCES))
    assert groups(warning) == {"video segments": {filled["screen"], filled["cam"]}, "image/png": {filled["pics"]}}, warning
    assert sum(len(call["heard"]) for call in one.calls()) == 1, "only the mic segment's sound"


def test_a_soundtrack_the_decider_cannot_hear_is_one_rolling_warning_and_a_decider_that_reads_both_says_nothing(
        home, tmp_path, filled, under):
    store = Store(home)
    cam = store.item(filled["cam"])
    more = []
    for n in range(7):  # seven more camera segments with a soundtrack, so eight in all
        target = store.streams / "cam" / "launch" / f"more{n}.mkv"
        target.write_bytes((store.streams / "cam" / "launch" / "cam.mkv").read_bytes())
        more.append(store.add_item("cam", "video", T0 + 2 * GAP + 10 * (n + 1), T0 + 2 * GAP + 10 * (n + 1) + SEGMENT_S,
                                   store.relative(target), target.stat().st_size, meta=segment_meta(target)))
    store.close()
    assert cam.meta.get("sound") and len(more) == 7

    result = under(decider_file(tmp_path, "eyes", {"video", "image"})).ask(["cam"])
    (warning,) = partial(result, "sound")
    assert warning.startswith("the sound of 8 video items was not heard, since eyes does not hear: cam (items "), warning
    assert warning.endswith(" and 3 more); find lists them"), "five are named and the rest counted"
    named = {int(n) for n in re.findall(r"\d+", warning.split("(items ", 1)[1].split(" and ")[0])}
    assert len(named) == 5 and named <= {filled["cam"], *more}, warning
    assert not [w for w in result["warnings"] if "not read" in w], "the camera is read, for its frames"

    for reads in ({"video", "image", "audio"}, {"audio"}):
        both = under(decider_file(tmp_path, "both" if "video" in reads else "ears", reads)).ask(["cam"])
        assert not partial(both, "sound"), "it hears the soundtrack"
        assert bool(partial(both, "frames")) == ("video" not in reads), both["warnings"]

    quiet = under(decider_file(tmp_path, "eyes", {"video", "image"})).ask(["screen"])
    assert not partial(quiet, "sound"), "a video with no soundtrack loses no sound"


def test_the_reader_reports_a_video_read_in_part_once_per_item_before_its_first_window(home, tmp_path, filled):
    store = Store(home)
    try:
        def reports(reads: set) -> tuple[list, list]:
            seen: list = []
            about = decider.load_decider(decider_file(tmp_path, "narrow", reads)).about
            windows = reader.windows(store, scope=EVERYTHING, about=about, newest_first=False,
                                     on_partial=lambda item, what: seen.append((item.id, what)))
            next(windows)
            before = list(seen)
            list(windows)
            return before, seen

        before, seen = reports({"video", "image"})
        assert before == seen == [(filled["cam"], "sound")], "reported before the first window, and not again"
        before, seen = reports({"audio"})
        assert before == seen == [(filled["cam"], "frames")]
        assert reports({"video", "image", "audio"})[1] == [], "nothing is lost when it reads both"
    finally:
        store.close()
