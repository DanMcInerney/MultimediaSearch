"""The shortcut probe (`benchmarks/shortcuts.py`): on a small haystack of its own, written with `haystack_media`'s writers into a temporary
folder, it finds the shortcuts that single a needle out (a colour filter, frame differencing, a band filter, the file's size) and
passes a haystack whose distractors share each needle's colour, motion and pitch; its floors, report and ffmpeg check are pinned.
A screen state is one candidate however many frames the owner's acts and the clock give it (its measures are the median of its frames).
The last test runs it on `haystack.plan`'s `tiny` scale (the harder haystack, `haystack.VERSION` 6)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import haystack_media as hm
import numpy as np
import pytest
import shortcuts
from PIL import Image, ImageDraw

NEEDLE, DISTRACTOR = hm.NEEDLE_RGB, hm.DISTRACTOR_RGB
SHADE_ERROR = {"kind": "shade", "hue": "red", "rgb": list(NEEDLE["error"])}
SHADE_SIGN = {"kind": "shade", "hue": "green", "rgb": list(NEEDLE["sign"])}
SHADE_PARCEL = {"kind": "shade", "hue": "brown", "rgb": list(NEEDLE["parcel"])}
PITCH_ALARM = {"kind": "pitch", "hz": 3200.0, "band": [2000, 5000]}
PITCH_DOORBELL = {"kind": "pitch", "hz": hm.DOORBELL_PAIRS[0][0], "band": [400, 1000]}
DAY = date(2026, 10, 28)


@dataclass(frozen=True)
class File:
    """A file of a plan, as `haystack.File` has it."""

    path: str
    source: str
    t: float
    kind: str
    spec: dict
    name: str | None = None


@dataclass(frozen=True)
class Plan:
    """A plan, as far as the probe reads one."""

    files: tuple
    questions: tuple
    days: tuple = (DAY,)
    scale: str = "tiny"
    seed: int = 1


def at(hour: int, minute: int = 0, second: float = 0) -> float:
    return datetime(DAY.year, DAY.month, DAY.day).timestamp() + hour * 3600 + minute * 60 + second  # (hour 24 is the next midnight)


class Haystack:
    """Files written into `folder` and the questions about them."""

    def __init__(self, folder: Path):
        self.folder, self.work, self.files, self.questions = folder, folder.parent / f"{folder.name}-work", [], []

    def add(self, f: File) -> File:
        self.files.append(f)
        return f

    def screen(self, hour: int, minute: int, states: list[tuple], acts: dict[int, list] | None = None) -> File:
        """A segment of `states`, `(app, overlay, accent)`, 30 s each (the owner's acts `acts[k]`, `[[tick, "move", x, y] | [tick, "type", n] | [tick, "scroll", lines], ..]`, in state k)."""
        spec = {"length_s": 30 * len(states), "states": [{"from": 30 * k, "to": 30 * (k + 1), "app": app, "overlay": overlay, "accent": accent,
                                                          "seed": 1000 * hour + 10 * minute + k, "acts": (acts or {}).get(k, [])}
                                                         for k, (app, overlay, accent) in enumerate(states)]}
        path = f"screen/screen-{DAY}_{hour:02d}-{minute:02d}-00.mkv"
        hm.write_screen(self.folder / path, spec, at(hour, minute), self.work)
        return self.add(File(path, "screen", at(hour, minute), "screen", spec))

    def photo(self, hour: int, minute: int, spec: dict, noise: int = 0) -> File:
        path = f"phone/IMG_{DAY:%Y%m%d}_{hour:02d}{minute:02d}00.jpg"
        image = hm.photo(spec)
        if noise:
            pixels = np.asarray(image, dtype=np.int16)
            pixels = pixels + np.random.default_rng(1).integers(-noise, noise + 1, pixels.shape)
            image = Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
        (self.folder / path).parent.mkdir(parents=True, exist_ok=True)
        image.save(self.folder / path, quality=90)
        return self.add(File(path, "phone", at(hour, minute), "photo", spec))

    def porch(self, hour: int, minute: int, spec: dict, crate: bool = False) -> File:
        """A porch clip. With `crate`, a big brown crate (the courier's box colour) sits in the area a parcel can be left in from the 3rd second:
        a brown thing that is no parcel, which the generator's own distractors may or may not supply (its door was brown, and is now blue)."""
        path = f"porch/porch-{DAY}_{hour:02d}-{minute:02d}-00.mp4"
        stamp = f"{DAY} {hour:02d}:{minute:02d}:00"
        if crate:
            x0, y0, x1, y1 = pixels(getattr(hm, "BOX_AREA", hm.BOX), *hm.PORCH)
            frames = []
            for second in range(int(spec["length_s"])):
                image = hm.porch_frame(spec, second, hm._stamp_text(stamp, second))
                if second >= 3:
                    ImageDraw.Draw(image).rectangle([x0 + 6, y0 + 6, x1 - 6, y1 - 6], fill=DISTRACTOR["parcel"]["carried"])
                frames.append((second, image))
            hm._write_video(self.folder / path, frames, spec["length_s"], hm.PORCH, hm.PORCH_CRF, self.work)
        else:
            hm.write_porch(self.folder / path, spec, stamp, self.work)
        return self.add(File(path, "porch", at(hour, minute), "porch", spec))

    def mic(self, hour: int, minute: int, sounds: list[dict], length_s: int = 120) -> File:
        spec = {"length_s": length_s, "seed": 10 * hour + minute, "bed_dbfs": -50, "tv": [], "sounds": sounds}
        path = f"mic/mic-{DAY}_{hour:02d}-{minute:02d}-00.ogg"
        hm.write_mic(self.folder / path, spec, self.work)
        return self.add(File(path, "mic", at(hour, minute), "mic", spec))

    def ask(self, qid: str, source: str, since: float, until: float, moments: list, probe: dict, level: str = "part", absent: bool = False,
            last: bool = False) -> dict:
        question = {"qid": qid, "template": qid.split("-", 1)[1].rsplit("-", 1)[0], "absent": absent, "source": source, "since": since, "until": until,
                    "level": level, "last": last, "truth": {"value": None if absent else "x", "moments": moments, "others": []}, "probe": probe}
        self.questions.append(question)
        return question

    @property
    def plan(self) -> Plan:
        return Plan(tuple(self.files), tuple(self.questions))


def error_state(code: str = "E-4417") -> tuple:
    return "editor", {"kind": "error", "rgb": list(NEEDLE["error"]), "code": code, "msg": "disk full"}, None


def sound(kind: str, at_: float, dur: float, **extra) -> dict:
    return {"kind": kind, "at": at_, "dur": dur, "seed": int(at_), **extra}


ALARM = sound("alarm", 20.0, hm.ALARM_DUR, hz=3200.0)
DOORBELL = sound("doorbell", 20.0, hm.CHIME_DUR, notes=list(hm.DOORBELL_PAIRS[0]))


def parcel_clip(seed: int = 7) -> dict:
    """A courier drops a box at 8 s and it stays on the step to the end of the clip."""
    return {"length_s": 20, "seed": seed, "parcel": [8, 20], "events": [{"kind": "courier", "from": 2, "to": 14, "drop_at": 8}]}


@pytest.fixture(scope="module")
def good(tmp_path_factory) -> Haystack:
    """A haystack whose distractors share each needle's colour, motion and pitch."""
    h = Haystack(tmp_path_factory.mktemp("good"))
    first = h.screen(9, 0, [("editor", None, None), ("browser", None, {"kind": "banner"}), ("terminal", None, None), ("chat", None, {"kind": "toast"}),
                            ("sheet", None, {"kind": "cells"}), error_state(), ("chat", {"kind": "confirm"}, None), ("slides", None, None),
                            ("browser", None, None), ("terminal", None, None)])
    h.screen(10, 0, [(app, None, None) for app in ("editor", "chat", "sheet", "slides", "terminal", "browser")])
    h.ask("video-screen-error-1", "screen", at(8), at(12), [[first.t + 150, first.t + 180]], SHADE_ERROR)
    h.ask("video-screen-error-2", "screen", at(10), at(11), [], SHADE_ERROR, level="day", absent=True)
    sign = {"scene": "sign", "rgb": list(NEEDLE["sign"]), "word": "GATE", "number": "47", "seed": 1}
    sign_photo = h.photo(8, 0, sign)
    for k, spec in enumerate([{"scene": "car", "rgb": list(DISTRACTOR["sign"]["car"]), "seed": 2}, {"scene": "plant", "rgb": None, "seed": 3},
                              {"scene": "street", "rgb": list(DISTRACTOR["sign"]["shirt"]), "seed": 4},
                              {"scene": "sign", "rgb": list(DISTRACTOR["sign"]["blue"]), "word": "ROOM", "number": "208", "seed": 5},
                              {"scene": "sign", "rgb": list(DISTRACTOR["sign"]["yellow"]), "word": "BAY", "number": "12", "seed": 6},
                              {"scene": "room", "rgb": list(DISTRACTOR["sign"]["door"]), "seed": 7}, {"scene": "food", "rgb": None, "seed": 8}], 1):
        h.photo(8, 5 * k, spec)
    h.ask("image-photo-sign-1", "phone", at(6), at(12), [[sign_photo.t, sign_photo.t]], SHADE_SIGN)
    first_clip = h.porch(15, 0, parcel_clip())
    h.porch(15, 30, {"length_s": 16, "seed": 7, "parcel": None, "events": [{"kind": "walker", "from": 2, "to": 10}, {"kind": "light", "from": 5, "to": 9},
                                                                            {"kind": "cat", "from": 8, "to": 14}]}, crate=True)
    h.porch(16, 0, {"length_s": 14, "seed": 7, "parcel": None, "events": [{"kind": "car", "from": 2, "to": 10, "rgb": [30, 60, 200]}]})
    h.ask("video-porch-parcel-1", "porch", at(12), at(18), [[first_clip.t + 8, first_clip.t + 20]], SHADE_PARCEL)
    alarm_file = h.mic(7, 0, [sound("microwave", 50.0, 2.0), sound("oven", 70.0, 2.5), sound("truck", 90.0, 3.0), sound("watch", 105.0, 1.0), ALARM])
    bell_file = h.mic(8, 0, [DOORBELL, sound("ding", 50.0, 0.8, hz=987.77), sound("ring", 70.0, 2.4, hz=440.0), sound("bark", 100.0, 1.2)])
    h.mic(18, 0, [sound("microwave", 30.0, 2.0), sound("oven", 60.0, 2.5)])
    h.ask("audio-mic-alarm-1", "mic", at(6), at(12), [[alarm_file.t + 20, alarm_file.t + 20 + hm.ALARM_DUR]], PITCH_ALARM)
    h.ask("audio-mic-doorbell-1", "mic", at(6), at(12), [[bell_file.t + 20, bell_file.t + 20 + hm.CHIME_DUR]], PITCH_DOORBELL)
    h.ask("audio-mic-alarm-2", "mic", at(0), at(24), [[alarm_file.t + 20, alarm_file.t + 20 + hm.ALARM_DUR]], PITCH_ALARM, level="week", last=True)
    h.questions.append({"qid": "text-note-code-1", "template": "note-code", "absent": False, "source": "notes", "since": at(6), "until": at(12),
                        "level": "part", "truth": {"value": "1234", "moments": [[at(9), at(9)]]}, "probe": None})
    return h


@pytest.fixture(scope="module")
def good_rows(good) -> dict[str, dict]:
    start = time.perf_counter()
    rows = shortcuts.probe(good.plan, good.folder)
    print(f"\nthe probe of {len(good.files)} files and {len(rows)} questions: {time.perf_counter() - start:.1f} s")
    return {row["qid"]: row for row in rows}


@pytest.fixture(scope="module")
def bad(tmp_path_factory) -> Haystack:
    """A haystack where each shortcut singles the needle out: nothing else red on the screen, nothing else green in the photos, a still
    porch, a lone alarm among sounds of another pitch."""
    h = Haystack(tmp_path_factory.mktemp("bad"))
    first = h.screen(9, 0, [(app, None, None) for app in ("editor", "chat", "sheet")] + [error_state()] + [("terminal", None, None), ("slides", None, None)])
    h.ask("video-screen-error-1", "screen", at(8), at(12), [[first.t + 90, first.t + 120]], SHADE_ERROR)
    sign_photo = h.photo(8, 0, {"scene": "sign", "rgb": list(NEEDLE["sign"]), "word": "GATE", "number": "47", "seed": 1})
    for k, spec in enumerate([{"scene": "room", "rgb": None, "seed": 2}, {"scene": "sign", "rgb": list(DISTRACTOR["sign"]["blue"]), "word": "ROOM", "number": "208", "seed": 3},
                              {"scene": "sign", "rgb": list(DISTRACTOR["sign"]["yellow"]), "word": "BAY", "number": "12", "seed": 4}], 1):
        h.photo(8, 10 * k, spec)
    h.ask("image-photo-sign-1", "phone", at(6), at(12), [[sign_photo.t, sign_photo.t]], SHADE_SIGN)
    still = {"length_s": 20, "seed": 7, "events": []}
    clip = h.porch(15, 0, {**still, "parcel": [8, 20]})
    h.porch(15, 30, {**still, "parcel": None})
    h.ask("video-porch-parcel-1", "porch", at(12), at(18), [[clip.t + 8, clip.t + 20]], SHADE_PARCEL)
    alone = h.mic(7, 0, [ALARM, sound("bark", 60.0, 1.2), sound("steps", 90.0, 3.0)])
    h.ask("audio-mic-alarm-1", "mic", at(6), at(12), [[alone.t + 20, alone.t + 20 + hm.ALARM_DUR]], PITCH_ALARM)
    return h


@pytest.fixture(scope="module")
def decoys(tmp_path_factory) -> Haystack:
    """A haystack where a filter tuned to the needle finds more: a second error dialog and a second green sign in the needle's own shade,
    and a second alarm within 8% of its pitch."""
    h = Haystack(tmp_path_factory.mktemp("decoys"))
    first = h.screen(9, 0, [("editor", None, None), error_state("E-4417"), ("browser", None, None), error_state("E-5230"), ("terminal", None, None)])
    h.ask("video-screen-error-1", "screen", at(8), at(12), [[first.t + 30, first.t + 60]], SHADE_ERROR)
    needle = h.photo(8, 0, {"scene": "sign", "rgb": list(NEEDLE["sign"]), "word": "GATE", "number": "47", "seed": 1})
    h.photo(8, 10, {"scene": "sign", "rgb": list(NEEDLE["sign"]), "word": "BAY", "number": "12", "seed": 9})
    h.photo(8, 20, {"scene": "plant", "rgb": None, "seed": 3})
    h.ask("image-photo-sign-1", "phone", at(6), at(12), [[needle.t, needle.t]], SHADE_SIGN)
    mic = h.mic(7, 0, [ALARM, sound("alarm", 60.0, hm.ALARM_DUR, hz=3300.0), sound("microwave", 90.0, 2.0)])
    h.ask("audio-mic-alarm-1", "mic", at(6), at(12), [[mic.t + 20, mic.t + 20 + hm.ALARM_DUR]], PITCH_ALARM)
    return h


# -- the hue bands -----------------------------------------------------------------------------------------------------------

def swatch(rgb) -> np.ndarray:
    return np.full((1, 9, 16, 3), rgb, np.uint8)


def test_each_needle_is_in_the_band_of_its_colours_name_and_so_are_the_distractors_that_share_the_name():
    in_band = lambda rgb, hue: float(shortcuts.band_share(swatch(rgb), hue)[0]) == 1.0  # noqa: E731
    assert in_band(NEEDLE["error"], "red") and in_band(NEEDLE["car"], "red")
    assert in_band(NEEDLE["sign"], "green") and in_band(NEEDLE["parcel"], "brown")
    shared = {"red": [DISTRACTOR["error"][k] for k in ("banner", "cells", "badge", "toast", "confirm")] + [DISTRACTOR["car"][k] for k in ("door", "umbrella", "tomato")],
              "green": [DISTRACTOR["sign"][k] for k in ("plant", "door", "shirt", "car")],
              "brown": [DISTRACTOR["parcel"][k] for k in ("carried", "dog", "courier")]}
    for hue, colours in shared.items():
        for rgb in colours:
            assert in_band(rgb, hue), f"{rgb} is not {hue} to a filter for {hue}"
    for hue, other in (("red", DISTRACTOR["error"]["warning"]), ("red", DISTRACTOR["error"]["info"]), ("red", DISTRACTOR["error"]["save"]),
                       ("green", DISTRACTOR["sign"]["yellow"]), ("green", DISTRACTOR["sign"]["blue"]), ("green", DISTRACTOR["sign"]["white"]),
                       ("red", DISTRACTOR["car"]["blue"]), ("red", DISTRACTOR["car"]["white"]), ("brown", (205, 200, 190)), ("brown", (60, 90, 70))):
        assert not in_band(other, hue), f"{other} is {hue} to a filter for {hue}"


def test_a_share_is_the_part_of_the_pixels_in_the_band():
    frame = np.zeros((2, 10, 10, 3), np.uint8)
    frame[0, :, :4] = NEEDLE["sign"]
    frame[1, :5, :] = NEEDLE["error"]
    assert shortcuts.band_share(frame, "green").tolist() == [0.4, 0.0]
    assert shortcuts.band_share(frame, "red").tolist() == [0.0, 0.5]
    assert shortcuts.band_share(frame, "brown").tolist() == [0.0, 0.0]


# -- a haystack where no shortcut singles the needle out --------------------------------------------------------------------

def test_a_haystack_of_distractors_that_share_each_needles_colour_motion_and_pitch_passes(good, good_rows):
    rows = list(good_rows.values())
    assert [r["qid"] for r in rows] == [q["qid"] for q in good.questions if q["probe"]], "a row for each media question, and none for text"
    assert shortcuts.check(rows, "tiny") == []
    error, sign, parcel = good_rows["video-screen-error-1"], good_rows["image-photo-sign-1"], good_rows["video-porch-parcel-1"]
    assert (error["candidates"], error["frames"], error["views"], error["unit"]) == (16, 16, 1, "screen states"), "both segments of the morning"
    assert error["colour_shortlist"] == 4 and 0.04 < error["colour_needle"] < 0.1, "the banner, the toast, the cells and the confirm dialog"
    assert error["needles"] == 1 and error["colour_rank"] >= 2, "the red banner has more red than the dialog's header"
    assert (sign["candidates"], sign["views"]) == (8, 1) and sign["colour_shortlist"] >= 1
    assert (parcel["candidates"], parcel["unit"], parcel["frames"], parcel["views"]) == (3, "porch clips", 50, 2)
    assert parcel["views_per_clip"] == 1 and error["views_per_clip"] is None and sign["views_per_clip"] is None, "a montage of a frame a clip, for the porch only"
    assert parcel["needles"] == 12 and parcel["colour_shortlist"] > 0 and parcel["change_shortlist"] >= 1 and parcel["change_needle"] > 0


def test_a_screen_state_with_a_moving_cursor_typing_and_scrolling_is_one_candidate_whose_colour_is_the_median_of_its_frames(tmp_path):
    states = [(app, None, None) for app in ("editor", "chat", "sheet")] + [error_state()] + [("terminal", None, None), ("slides", None, None)]
    still, moving = Haystack(tmp_path / "still"), Haystack(tmp_path / "moving")
    busy = [[2 * sec, "move", 480 + 20 * sec, 220 + 5 * sec] for sec in range(2, 28)] + [[2 * sec + 1, "type", sec] for sec in range(2, 28, 4)]
    for h, acts in ((still, None), (moving, {3: sorted(busy), 1: [[10, "move", 10, 10], [12, "move", 800, 500], [13, "scroll", 1]]})):
        first = h.screen(9, 0, states, acts)
        h.ask("video-screen-error-1", "screen", at(8), at(12), [[first.t + 90, first.t + 120]], SHADE_ERROR)
    a, b = (shortcuts.probe(h.plan, h.folder)[0] for h in (still, moving))
    assert len(hm.screen_frames(moving.files[0].spec, moving.files[0].t)) == 6 + 26 + 7 + 3, "a frame for each state and each act (the clock changes only where a state starts)"
    assert (a["candidates"], a["frames"]) == (b["candidates"], b["frames"]) == (6, 6), "a state is a candidate, whatever its frames"
    assert (a["recorded_frames"], b["recorded_frames"], a["recorded_views"], b["recorded_views"]) == (6, 42, 1, 2), "the frames the files arm would read are counted beside"
    assert b["needles"] == 1 and b["colour_needle"] == pytest.approx(a["colour_needle"], rel=0.05), "a cursor over the dialog does not change what the state holds"
    assert b["colour_shortlist"] == a["colour_shortlist"] and b["size_files"] == a["size_files"]
    assert hm.screen_frames(still.files[0].spec, still.files[0].t) != hm.screen_frames(moving.files[0].spec, moving.files[0].t)
    wrong = File(moving.files[0].path, "screen", moving.files[0].t, "screen", still.files[0].spec)  # a spec that places other frames than the file has
    with pytest.raises(ValueError, match="frames were decoded and its spec has"):
        shortcuts.probe(Plan((wrong,), tuple(moving.questions)), moving.folder)


def test_the_band_shortlist_holds_the_beepers_for_the_alarm_and_the_dings_for_the_doorbell(good_rows):
    alarm, bell = good_rows["audio-mic-alarm-1"], good_rows["audio-mic-doorbell-1"]
    assert (alarm["candidates"], alarm["unit"], alarm["frames"], alarm["views"]) == (9, "sounds", None, None)
    assert alarm["band_shortlist"] == 4, "the microwave, the oven, the truck and the watch beep in the alarm's band"
    assert abs(alarm["band_needle_hz"] - 3200) < 10 and 0.25 < alarm["band_needle_rms"] < 0.5
    assert abs(bell["band_needle_hz"] - hm.DOORBELL_PAIRS[0][0]) < 5
    assert 1 <= bell["band_shortlist"] <= 3, "a ding, a phone ring and a bark sound in the doorbell's band"


def test_an_absent_question_has_the_candidates_and_the_views_and_nothing_to_find(good_rows):
    absent = good_rows["video-screen-error-2"]
    assert (absent["absent"], absent["candidates"], absent["views"], absent["needles"]) == (True, 6, 1, 0)
    assert absent["colour_needle"] is None and absent["colour_shortlist"] is None and absent["size_z"] is None and absent["problems"] == []


def test_a_last_question_counts_what_comes_after_the_last_needle(good_rows):
    day, last = good_rows["audio-mic-alarm-1"], good_rows["audio-mic-alarm-2"]
    assert last["last"] and last["level"] == "week"
    assert last["scope"][0] == datetime.fromtimestamp(at(7, 0, 20 + hm.ALARM_DUR)).strftime("%Y-%m-%d %H:%M:%S"), "from where the needle ends"
    assert last["candidates"] == 10, "the four sounds after the alarm in its file, the four of the next, and the two of the evening"
    assert last["band_shortlist"] == 6, "after the alarm: its file's four beepers and the evening's two"
    assert last["needles"] == day["needles"] == 1 and last["band_needle_rms"] == day["band_needle_rms"]


def test_a_size_among_a_few_files_is_never_an_outlier(good_rows):
    for row in good_rows.values():
        assert not row["size_extreme"], row["qid"]
    assert good_rows["video-screen-error-1"]["size_files"] == 2 and good_rows["audio-mic-alarm-1"]["size_files"] == 2


# -- each shortcut that singles a needle out fails --------------------------------------------------------------------------

def test_a_colour_filter_a_frame_difference_and_a_band_filter_that_single_the_needle_out_are_failures(bad):
    rows = shortcuts.probe(bad.plan, bad.folder)
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["video-screen-error-1"]["colour_shortlist"] == 0 and by_qid["image-photo-sign-1"]["colour_shortlist"] == 0
    assert by_qid["video-porch-parcel-1"]["change_shortlist"] == 0 and by_qid["audio-mic-alarm-1"]["band_shortlist"] == 0
    assert (by_qid["video-screen-error-1"]["colour_rank"], by_qid["video-porch-parcel-1"]["change_rank"], by_qid["audio-mic-alarm-1"]["band_rank"]) == (1, 1, 1)
    assert (by_qid["video-screen-error-1"]["region_broad_shortlist"], by_qid["video-screen-error-1"]["region_broad_rank"]) == (0, 1)
    failures = shortcuts.check(rows, "tiny")
    assert len(failures) >= 5, failures
    assert "a red filter leaves 0 other frames" in " ".join(f for f in failures if "video-screen-error-1" in f)
    assert "cropping the dialog's area and keeping red leaves 0 other frames" in " ".join(f for f in failures if "video-screen-error-1" in f)
    assert "a green filter leaves 0 other frames" in " ".join(f for f in failures if "image-photo-sign-1" in f)
    assert "frame differencing leaves 0 other frames" in " ".join(f for f in failures if "video-porch-parcel-1" in f)
    assert "band filter of 2000 to 5000 Hz leaves 0 other sounds" in " ".join(f for f in failures if "audio-mic-alarm-1" in f)
    assert all("singles out the needle" in f for f in failures)


def test_a_needle_file_that_is_the_largest_of_its_scope_by_a_wide_margin_is_a_failure(tmp_path):
    h = Haystack(tmp_path / "sizes")
    sign = {"scene": "sign", "rgb": list(NEEDLE["sign"]), "word": "GATE", "number": "47", "seed": 1}
    needle = h.photo(8, 0, sign, noise=60)
    for k in range(1, 15):
        h.photo(8, 3 * k, {"scene": ("room", "food", "plant", "street")[k % 4], "rgb": None, "seed": k})
    h.ask("image-photo-sign-1", "phone", at(6), at(12), [[needle.t, needle.t]], SHADE_SIGN)
    (row,) = shortcuts.probe(h.plan, h.folder)
    assert row["size_extreme"] and row["size_rank"] == "largest" and row["size_z"] > 2.5 and row["size_files"] == 15
    assert any("the largest of 15 in scope" in f and "size gives it away" in f for f in shortcuts.check([row], "tiny"))
    quiet = Haystack(tmp_path / "plain")  # the needle among signs of other colours, which are as small a file as it is
    quiet.photo(8, 0, sign)
    for k, colour in enumerate(("yellow", "white", "blue", "grey"), 1):
        quiet.photo(8, 3 * k, {"scene": "sign", "rgb": list(DISTRACTOR["sign"][colour]), "word": "ROOM", "number": str(200 + k), "seed": k})
    for k in range(5, 15):
        quiet.photo(8, 3 * k, {"scene": ("room", "food", "plant", "street")[k % 4], "rgb": None, "seed": k})
    quiet.ask("image-photo-sign-1", "phone", at(6), at(12), [[at(8), at(8)]], SHADE_SIGN)
    (plain,) = shortcuts.probe(quiet.plan, quiet.folder)
    assert not plain["size_extreme"] and abs(plain["size_z"]) < 2.5


def test_a_needle_the_plan_puts_nowhere_and_a_probe_of_the_wrong_kind_are_failures(good):
    wrong_time = {**good.questions[0], "qid": "video-screen-error-9", "truth": {"value": "x", "moments": [[at(3), at(3, 0, 30)]], "others": []}}
    wrong_kind = {**good.questions[0], "qid": "video-screen-error-8", "probe": PITCH_ALARM}
    wrong_hue = {**good.questions[0], "qid": "video-screen-error-7", "probe": {"kind": "shade", "hue": "mauve", "rgb": [1, 2, 3]}}
    present = {**good.questions[0], "qid": "video-screen-error-6", "truth": {"value": "x", "moments": [], "others": []}}
    plan = Plan(tuple(good.files), (wrong_time, wrong_kind, wrong_hue, present))
    rows = shortcuts.probe(plan, good.folder)
    failures = shortcuts.check(rows, "tiny")
    assert any("video-screen-error-9" in f and "no screen state of the plan is at the needle's moment" in f for f in failures)
    assert any("video-screen-error-8" in f and "has a shade probe, not 'pitch'" in f for f in failures)
    assert any("video-screen-error-7" in f and "'mauve' is none of red, green, brown" in f for f in failures)
    assert any("video-screen-error-6" in f and "no absent one and has no moment" in f for f in failures)


# -- the tuned filters (report only) ---------------------------------------------------------------------------------------

def test_a_tuned_colour_keeps_the_needles_own_hue_saturation_and_value_and_wraps_round_red():
    h, s, v = shortcuts.needle_hsv(NEEDLE["error"])
    assert (round(h, 1), round(s, 2), round(v, 2)) == (356.8, 0.84, 0.69) and shortcuts.TUNE == (10.0, 0.10, 0.10)
    jitter = np.random.default_rng(3).integers(-3, 4, (10, 3))  # what the encoder does to a shade
    frame = np.zeros((1, 10, 10, 3), np.uint8)
    frame[0, 0:4] = (np.array(NEEDLE["error"]) + jitter[:4, None, :]).clip(0, 255)  # the needle: 40%
    frame[0, 4:6] = DISTRACTOR["error"]["banner"]  # a red of the same name with another value: 20%
    frame[0, 6:8] = (176, 36, 28)  # the needle's saturation and value, hue 3 degrees, across the wrap from 357: 20%
    frame[0, 8:] = 30
    assert shortcuts.tuned_share(frame, NEEDLE["error"]).tolist() == [0.6]
    assert shortcuts.band_share(frame, "red").tolist() == [0.8], "the broad filter for red keeps the banner too"
    frame[0, 6:8] = (176, 90, 28)  # hue 16 degrees: out
    assert shortcuts.tuned_share(frame, NEEDLE["error"]).tolist() == [0.4]


def test_a_tuned_colour_tells_the_sign_the_parcel_and_the_car_from_the_distractors_of_their_names():
    for needle, others in (("sign", ("plant", "door", "shirt", "car")), ("parcel", ("carried", "dog", "door", "courier", "planter")),
                           ("car", ("door", "umbrella", "tomato")), ("error", ("banner", "cells", "badge", "toast", "confirm"))):
        for name in others:
            assert shortcuts.tuned_share(swatch(DISTRACTOR[needle][name]), NEEDLE[needle])[0] == 0.0, f"{name} is not the {needle}'s shade"
        assert shortcuts.tuned_share(swatch(NEEDLE[needle]), NEEDLE[needle])[0] == 1.0


def test_a_region_is_where_the_error_dialog_and_the_parcel_are_drawn_and_a_photos_needle_has_none(monkeypatch):
    for name in ("DIALOG_AREA", "BOX_AREA"):  # (the area is the fixed place when the module names no wider one)
        monkeypatch.delattr(hm, name, raising=False)
    x0, y0, x1, y1 = shortcuts.region_of(NEEDLE["error"])
    assert (x1 - x0, y1 - y0) == pytest.approx(hm.DIALOG) and (x0 + x1, y0 + y1) == pytest.approx((1, 1)), "the dialog's box, centred"
    assert shortcuts.region_of(NEEDLE["parcel"]) == hm.BOX
    assert shortcuts.region_of(NEEDLE["sign"]) is None and shortcuts.region_of(NEEDLE["car"]) is None
    assert shortcuts.region_share(swatch(NEEDLE["sign"]), NEEDLE["sign"]) is None
    inside, outside = np.zeros((2, 90, 160, 3), np.uint8), np.zeros((2, 90, 160, 3), np.uint8)
    inside[0, 29:61, 44:116] = NEEDLE["error"]  # in the dialog's box
    inside[1, 0:20] = NEEDLE["error"]  # in a banner's place
    shares = shortcuts.region_share(inside, NEEDLE["error"])
    assert shares[0] > 0.95 and shares[1] == 0.0
    assert shortcuts.tuned_share(inside, NEEDLE["error"])[1] > 0.1, "the colour is in the second frame, but not in the region"
    outside[:, 40:60, 25:55] = NEEDLE["parcel"]
    assert shortcuts.region_share(outside, NEEDLE["parcel"]).min() > 0.5


def pixels(area, width: int = 160, height: int = 90) -> tuple[int, int, int, int]:
    """An area in fractions as the pixels `(x0, y0, x1, y1)` of a frame."""
    return round(area[0] * width), round(area[1] * height), round(area[2] * width), round(area[3] * height)


def test_a_broad_filter_on_the_area_counts_a_red_patch_inside_it_and_not_a_banner_outside_it():
    x0, y0, x1, y1 = pixels(shortcuts.area_of(NEEDLE["error"]))
    assert y0 > 4, "there is room above the area for a banner"
    frames = np.zeros((4, 90, 160, 3), np.uint8)
    frames[0, 0:y0 - 3] = DISTRACTOR["error"]["banner"]  # a red banner above the area
    frames[1, y0:y1, x0:x1] = DISTRACTOR["error"]["confirm"]  # a red dialog of another shade in it
    frames[2, y0:y1, x0:x1] = NEEDLE["error"]  # the needle in it
    frames[3, y0:y1, x0:x1] = DISTRACTOR["error"]["warning"]  # an orange one: not red
    assert shortcuts.area_share(frames, NEEDLE["error"], "red").tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0], abs=0.03)
    assert shortcuts.band_share(frames, "red")[0] > 0.05, "the main colour shortlist counts the banner"
    box = pixels(shortcuts.area_of(NEEDLE["parcel"]))
    parcel = np.zeros((3, 90, 160, 3), np.uint8)
    parcel[0, box[1]:box[3], box[0]:box[2]] = NEEDLE["parcel"]
    parcel[1, box[1]:box[3], box[0]:box[2]] = DISTRACTOR["parcel"]["carried"]  # a brown crate, which is no parcel's shade, is brown too
    parcel[2, box[3] + 2:] = DISTRACTOR["parcel"]["dog"]  # brown, but below the area
    assert shortcuts.area_share(parcel, NEEDLE["parcel"], "brown").tolist() == pytest.approx([1.0, 1.0, 0.0], abs=0.03)
    assert shortcuts.area_share(swatch(NEEDLE["sign"]), NEEDLE["sign"], "green") is None, "a photo's needle has no area"


def test_the_area_is_the_union_the_module_names_and_the_one_fixed_place_when_it_names_none(monkeypatch):
    for name in ("DIALOG_AREA", "BOX_AREA"):
        monkeypatch.delattr(hm, name, raising=False)
    assert shortcuts.area_of(NEEDLE["error"]) == shortcuts.region_of(NEEDLE["error"]) and shortcuts.area_of(NEEDLE["parcel"]) == hm.BOX
    monkeypatch.setattr(hm, "DIALOG_AREA", (0.1, 0.2, 0.9, 0.7), raising=False)
    monkeypatch.setattr(hm, "BOX_AREA", [0.05, 0.4, 0.5, 0.75], raising=False)
    assert shortcuts.area_of(NEEDLE["error"]) == (0.1, 0.2, 0.9, 0.7) and shortcuts.area_of(NEEDLE["parcel"]) == (0.05, 0.4, 0.5, 0.75)
    assert shortcuts.region_of(NEEDLE["error"]) != (0.1, 0.2, 0.9, 0.7), "the fixed place stays as it was"
    frames = np.zeros((1, 90, 160, 3), np.uint8)
    frames[0, 18:63, 16:144] = NEEDLE["error"]  # fills the wider area ...
    frames[0, 29:61, 44:116] = 0  # ... but not the centred dialog's box
    assert shortcuts.area_share(frames, NEEDLE["error"], "red")[0] > 0.5
    monkeypatch.delattr(hm, "DIALOG_AREA")
    assert shortcuts.area_share(frames, NEEDLE["error"], "red")[0] == 0.0
    assert shortcuts.area_of(NEEDLE["sign"]) is None


def test_cropping_the_area_is_held_to_the_colour_shortlists_floors_and_the_rank_rule():
    cases = {"dialog": (SHADE_ERROR, "screen", "screen states", "red"), "parcel": (SHADE_PARCEL, "porch", "porch clips", "brown")}
    for scale in ("pilot", "standard"):
        for level, floor in SHORTLIST_FLOORS[scale].items():
            for name, (probe_, source, unit, hue) in cases.items():
                base = dict(level=level, source=source, unit=unit, probe=probe_, region_broad_needle=0.06, region_broad_rank=40)
                assert shortcuts.check([row(**base, region_broad_shortlist=floor)], scale) == [], (scale, level, name)
                (failure,) = shortcuts.check([row(**base, region_broad_shortlist=floor - 1)], scale)
                assert failure == (f"video-screen-error-1 ({level}, {source}): cropping the {name}'s area and keeping {hue} leaves {floor - 1} other frames "
                                   f"at half the needle's 6.0%, {floor} needed: it singles out the needle")
    for scale in ("pilot", "standard"):
        for level in ("day", "week"):
            for name, (probe_, source, unit, hue) in cases.items():
                base = dict(level=level, source=source, unit=unit, probe=probe_, region_broad_needle=0.06, region_broad_shortlist=100)
                (failure,) = shortcuts.check([row(**base, region_broad_rank=1)], scale)
                assert failure == (f"video-screen-error-1 ({level}, {source}): sorting by the share of {hue} inside the {name}'s area puts the needle first "
                                   f"(6.0%, ahead of every other frame): one view finds it")
                assert shortcuts.check([row(**base, region_broad_rank=2)], scale) == []
        part = dict(level="part", region_broad_needle=0.06, region_broad_shortlist=100, region_broad_rank=1)
        assert shortcuts.check([row(**part)], scale) == [], "part level is exempt from the rank rule"
    tiny = dict(region_broad_needle=0.06, region_broad_shortlist=1, region_broad_rank=1)
    for level in ("part", "day", "week"):
        assert shortcuts.check([row(level=level, **tiny)], "tiny") == []
    assert len(shortcuts.check([row(**{**tiny, "region_broad_shortlist": 0})], "tiny")) == 1, "tiny needs one other frame"
    assert shortcuts.check([row(absent=True, colour_needle=None, colour_shortlist=None, colour_rank=None)], "standard") == []
    assert shortcuts.check([row(problems=["x"])], "tiny") == ["video-screen-error-1 (part, screen): x"]


def test_the_tuned_filters_leave_no_other_frame_or_sound_in_a_haystack_of_distractors_of_the_needles_names(good_rows):
    for qid in ("video-screen-error-1", "image-photo-sign-1", "video-porch-parcel-1"):
        r = good_rows[qid]
        assert r["tuned_colour_needle"] > 0 and (r["tuned_colour_shortlist"], r["tuned_colour_rank"]) == (0, 1), qid
        assert r["tuned_colour_hsv"] == [round(x, 3) for x in shortcuts.needle_hsv(r["probe"]["rgb"])]
    for qid in ("video-screen-error-1", "video-porch-parcel-1"):
        r = good_rows[qid]
        assert r["region_colour_needle"] > r["tuned_colour_needle"] and (r["region_colour_shortlist"], r["region_colour_rank"]) == (0, 1), qid
    assert good_rows["image-photo-sign-1"]["region_colour_needle"] is None and good_rows["image-photo-sign-1"]["region_broad_needle"] is None
    for qid, hz in (("audio-mic-alarm-1", 3200), ("audio-mic-alarm-2", 3200), ("audio-mic-doorbell-1", hm.DOORBELL_PAIRS[0][0])):
        r = good_rows[qid]
        low, high = r["tuned_band"]
        assert low < hz < high and high - low == pytest.approx(0.16 * r["band_needle_hz"], abs=2), qid
        assert (r["tuned_band_shortlist"], r["tuned_band_rank"]) == (0, 1), "the broad band keeps the beepers, the narrow one only the needle"
    assert good_rows["audio-mic-alarm-1"]["band_shortlist"] == 4
    absent = good_rows["video-screen-error-2"]
    assert absent["tuned_colour_needle"] is None and absent["region_colour_needle"] is None and absent["tuned_band"] is None


def test_a_broad_filter_on_the_area_keeps_the_dialogs_that_are_red_in_it_and_not_the_banner_and_the_toast_outside_it(good_rows):
    error, parcel = good_rows["video-screen-error-1"], good_rows["video-porch-parcel-1"]
    assert error["region_area"] == [round(x, 3) for x in shortcuts.area_of(NEEDLE["error"])] and parcel["region_area"] == [round(x, 3) for x in shortcuts.area_of(NEEDLE["parcel"])]
    assert error["region_broad_needle"] > 0.05, "the dialog's header is part of the area"
    assert 1 <= error["region_broad_shortlist"] < error["colour_shortlist"], "the confirm dialog is in the area; the banner and the toast are not"
    assert error["region_broad_rank"] >= 1 and parcel["region_broad_shortlist"] >= 1
    assert good_rows["video-screen-error-2"]["region_broad_needle"] is None, "an absent question has no needle"
    assert good_rows["audio-mic-alarm-1"]["region_broad_needle"] is None


def test_a_second_dialog_sign_or_alarm_in_the_needles_own_shade_or_pitch_is_in_the_tuned_shortlist(decoys):
    by_qid = {r["qid"]: r for r in shortcuts.probe(decoys.plan, decoys.folder)}
    error, sign, alarm = by_qid["video-screen-error-1"], by_qid["image-photo-sign-1"], by_qid["audio-mic-alarm-1"]
    assert error["tuned_colour_shortlist"] == 1 and error["region_colour_shortlist"] == 1 and error["tuned_colour_rank"] in (1, 2)
    assert sign["tuned_colour_shortlist"] == 1 and sign["tuned_colour_rank"] in (1, 2)
    assert alarm["tuned_band_shortlist"] == 1 and alarm["tuned_band_rank"] in (1, 2), "the alarm at 3,300 Hz; the microwave at 2,450 is out of the 8%"
    assert alarm["band_shortlist"] == 2 and alarm["candidates"] == 3
    assert shortcuts.check(list(by_qid.values()), "tiny") == [], "report only: none of it is a floor"


def test_the_tuned_filters_are_report_only_and_check_does_not_read_them():
    tuned = dict(level="week", tuned_colour_hsv=[357.0, 0.84, 0.69], tuned_colour_needle=0.03, tuned_colour_shortlist=0, tuned_colour_rank=1,
                 region_colour_needle=0.2, region_colour_shortlist=0, region_colour_rank=1, tuned_band=[2944, 3456], tuned_band_shortlist=0, tuned_band_rank=1)
    for scale in ("tiny", "pilot", "standard"):
        assert shortcuts.check([row(**tuned)], scale) == []


# -- the floors -------------------------------------------------------------------------------------------------------------

def row(**kw) -> dict:
    return {"qid": "video-screen-error-1", "level": "part", "source": "screen", "unit": "screen states", "absent": False, "last": False,
            "candidates": 5000, "frames": 5000, "views": 200, "recorded_frames": 90000, "recorded_views": 3600, "probe": SHADE_ERROR, "colour_needle": 0.06, "colour_shortlist": 100, "colour_rank": 40,
            "change_needle": None, "change_shortlist": None, "change_rank": None, "band_needle_rms": None, "band_needle_hz": None, "band_shortlist": None,
            "band_rank": None, "region_area": None, "region_broad_needle": None, "region_broad_shortlist": None, "region_broad_rank": None,
            "size_z": 0.5, "size_rank": None,
            "size_extreme": False, "size_files": 78, "problems": [], **kw}


STANDARD_FLOORS = {("part", "screen"): 250, ("part", "porch"): 8, ("part", "phone"): 15, ("part", "mic"): 5, ("day", "screen"): 600, ("day", "porch"): 25,
                   ("day", "phone"): 50, ("day", "mic"): 15, ("week", "screen"): 1500, ("week", "porch"): 60, ("week", "phone"): 150, ("week", "mic"): 30}
PILOT_FLOORS = {**STANDARD_FLOORS, ("week", "screen"): 1300, ("week", "porch"): 50, ("week", "phone"): 90}  # about a day follows the pilot's last needle
SHORTLIST_FLOORS = {"standard": {"part": 5, "day": 15, "week": 30}, "pilot": {"part": 5, "day": 15, "week": 25}}
BAND_FLOORS = {"part": 3, "day": 8, "week": 15}


@pytest.mark.parametrize("scale, floors", [("standard", STANDARD_FLOORS), ("pilot", PILOT_FLOORS)])
def test_standard_has_the_floors_of_the_plan_and_pilot_a_lower_week_by_level_and_source(scale, floors):
    for (level, source), need in floors.items():
        unit = shortcuts.SOURCES[source][1]
        assert shortcuts.check([row(level=level, source=source, unit=unit, candidates=need)], scale) == [], (level, source)
        (failure,) = shortcuts.check([row(level=level, source=source, unit=unit, candidates=need - 1)], scale)
        assert f"{need - 1} {unit} in scope, {need} needed" in failure
    colour = SHORTLIST_FLOORS[scale]
    for level in colour:
        assert shortcuts.check([row(level=level, colour_shortlist=colour[level])], scale) == []
        assert "singles out the needle" in shortcuts.check([row(level=level, colour_shortlist=colour[level] - 1)], scale)[0]
        porch = dict(level=level, source="porch", unit="porch clips", change_needle=9.0)
        assert shortcuts.check([row(**porch, change_shortlist=colour[level])], scale) == []
        assert "frame differencing" in shortcuts.check([row(**porch, change_shortlist=colour[level] - 1)], scale)[0]
        mic = dict(level=level, source="mic", unit="sounds", colour_needle=None, colour_shortlist=None, probe=PITCH_ALARM, band_needle_rms=0.3, band_needle_hz=3200.0)
        assert shortcuts.check([row(**mic, band_shortlist=BAND_FLOORS[level])], scale) == []
        assert "band filter" in shortcuts.check([row(**mic, band_shortlist=BAND_FLOORS[level] - 1)], scale)[0]


def test_the_pilots_week_floors_are_under_what_its_haystack_measures_and_only_the_week_differs():
    assert shortcuts.FLOORS["standard"] == {"candidates": shortcuts.CANDIDATES, "shortlist": shortcuts.SHORTLIST, "band": shortcuts.BAND}
    standard, pilot = shortcuts.FLOORS["standard"], shortcuts.FLOORS["pilot"]
    assert {(level, source): n for level, by_source in pilot["candidates"].items() for source, n in by_source.items()} == PILOT_FLOORS
    assert pilot["shortlist"] == SHORTLIST_FLOORS["pilot"] and pilot["band"] == standard["band"]
    for level in ("part", "day"):
        assert pilot["candidates"][level] == standard["candidates"][level] and pilot["shortlist"][level] == standard["shortlist"][level]
    week = pilot["candidates"]["week"]
    assert week["screen"] <= 1409 and week["porch"] <= 53 and week["phone"] <= 101 and pilot["shortlist"]["week"] <= 30, "what the pilot measured after its last needle"


def test_a_needle_that_a_filter_ranks_first_fails_at_day_and_week_on_pilot_and_standard_and_nowhere_else():
    colour = dict(colour_needle=0.06, colour_shortlist=40)
    porch = dict(source="porch", unit="porch clips", change_needle=7.3, change_shortlist=40)
    mic = dict(source="mic", unit="sounds", colour_needle=None, colour_shortlist=None, colour_rank=None, probe=PITCH_ALARM, band_needle_rms=0.34,
               band_needle_hz=3200.0, band_shortlist=40)
    cases = {"colour": (colour, "colour_rank", "sorting by the share of red puts the needle first (6.0%, ahead of every other frame): one view finds it"),
             "change": (porch, "change_rank", "sorting by the change from the day's median frame puts the needle first (7.3, ahead of every other frame): one view finds it"),
             "band": (mic, "band_rank", "sorting the sounds of 2000 to 5000 Hz by level puts the needle first (0.34, ahead of every other sound in the band): one check finds it")}
    for name, (base, rank, sentence) in cases.items():
        for scale in ("pilot", "standard"):
            for level in ("day", "week"):
                (failure,) = shortcuts.check([row(**base, level=level, **{rank: 1})], scale)
                assert failure == f"video-screen-error-1 ({level}, {base.get('source', 'screen')}): {sentence}", (name, scale, level)
                assert shortcuts.check([row(**base, level=level, **{rank: 2})], scale) == [], f"rank 2 passes: {name} {scale} {level}"
            assert shortcuts.check([row(**base, level="part", **{rank: 1})], scale) == [], f"part level is exempt: {name} {scale}"
        for level in ("part", "day", "week"):
            assert shortcuts.check([row(**base, level=level, **{rank: 1})], "tiny") == [], f"tiny is exempt: {name} {level}"
    both = shortcuts.check([row(**porch, level="week", colour_rank=1, change_rank=1)], "standard")
    assert len(both) == 2 and "share of red" in both[0] and "change from the day's median" in both[1], "each shortcut that puts the needle first fails on its own"
    absent = row(absent=True, colour_needle=None, colour_shortlist=None, colour_rank=None, level="day")
    assert shortcuts.check([absent], "standard") == [], "an absent question has no rank"


def test_tiny_needs_one_of_each_and_nothing_more():
    assert shortcuts.check([row(candidates=1, colour_shortlist=1)], "tiny") == []
    assert len(shortcuts.check([row(candidates=0, colour_shortlist=0)], "tiny")) == 2
    assert len(shortcuts.check([row(candidates=5, colour_shortlist=4)], "standard")) == 2, "four others and five states: both under the part's floors"


def test_a_row_without_a_needle_is_held_to_its_candidates_only_and_a_size_outlier_and_a_problem_fail_at_every_scale():
    assert shortcuts.check([row(absent=True, colour_needle=None, colour_shortlist=None)], "standard") == []
    assert len(shortcuts.check([row(absent=True, colour_needle=None, colour_shortlist=None, candidates=10)], "standard")) == 1
    for scale in ("tiny", "pilot", "standard"):
        (failure,) = shortcuts.check([row(size_extreme=True, size_rank="smallest", size_z=-3.1, size_files=78)], scale)
        assert "the smallest of 78 in scope (z -3.1)" in failure
        assert shortcuts.check([row(problems=["the needle is nowhere"])], scale) == ["video-screen-error-1 (part, screen): the needle is nowhere"]
    (failure,) = shortcuts.check([row(level="time")], "standard")
    assert "no floors" in failure
    with pytest.raises(ValueError, match="tiny, pilot or standard"):
        shortcuts.check([], "huge")


# -- ffmpeg -----------------------------------------------------------------------------------------------------------------

def test_the_files_arms_ffmpeg_has_no_ocr_filter():
    names = shortcuts.filters()
    assert len(names) > 100 and {"scale", "tile", "showinfo", "silencedetect", "highpass"} <= set(names), "the filters the smoke's files arm used are listed"
    assert "ocr" not in names and shortcuts.ocr_failures(names) == []


def test_an_ffmpeg_with_an_ocr_filter_or_a_listing_that_was_not_read_is_a_failure():
    (failure,) = shortcuts.ocr_failures(["scale", "ocr", "tile"], "ffmpeg.exe")
    assert "ffmpeg.exe has the ocr filter" in failure
    assert "not read" in shortcuts.ocr_failures([])[0]


# -- the report -------------------------------------------------------------------------------------------------------------

def test_the_report_is_json_and_markdown_in_the_run_folder_and_never_the_data_folder(good, good_rows, tmp_path):
    rows = list(good_rows.values())
    failures = shortcuts.check(rows, "standard")
    assert failures, "a haystack of a few files is far under the floors of a standard one"
    meta = {"scale": "standard", "seed": 3, "seconds": 1.5, "ffmpeg": "ffmpeg.exe", "ocr": False}
    json_path, md_path = shortcuts.write_report(rows, failures, tmp_path / "run", good.folder, meta)
    assert (json_path.name, md_path.name) == ("shortcuts.json", "shortcuts.md") and json_path.parent == (tmp_path / "run").resolve()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["failures"] == failures and [r["qid"] for r in saved["rows"]] == [r["qid"] for r in rows] and saved["scale"] == "standard"
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("# Shortcut probe: standard, seed 3") and f"FAILED: {len(failures)} failures" in text and "## Failures" in text
    for r in rows:
        assert r["qid"] in text
    assert "## Brute-force views" in text and "| part | " in text and "porch, a frame a clip" in text
    assert "colour in the needle's area: needle / others / rank" in text
    assert "## Tuned filters (report only)" in text and "Not floors, and not in `check`" in text and "tuned band: Hz / others / rank" in text
    tuned = text.split("## Tuned filters (report only)")[1].split("## Brute-force views")[0]
    assert "video-screen-error-1" in tuned and "audio-mic-alarm-1" in tuned and "(absent)" not in tuned and "text-note" not in tuned
    with pytest.raises(ValueError, match="never in the data folder"):
        shortcuts.write_report(rows, [], good.folder / "report", good.folder, meta)
    with pytest.raises(ValueError, match="never in the data folder"):
        shortcuts.write_report(rows, [], good.folder, good.folder, meta)
    assert "No failures." in shortcuts.markdown(rows, [], meta)
    assert not (good.folder / "shortcuts.json").exists()


def test_run_probes_checks_the_ffmpeg_and_writes_the_report(good, tmp_path):
    out = tmp_path / "run"
    assert shortcuts.run(good.plan, good.folder, out) == []
    saved = json.loads((out / "shortcuts.json").read_text(encoding="utf-8"))
    assert {"tuned_colour_shortlist", "region_colour_rank", "tuned_band", "tuned_band_shortlist", "views_per_clip", "region_area", "region_broad_needle",
         "region_broad_shortlist", "region_broad_rank"} <= set(saved["rows"][0])
    assert saved["failures"] == [] and saved["ocr"] is False and saved["scale"] == "tiny" and saved["days"] == [DAY.isoformat()] and len(saved["rows"]) == 7
    assert "No failures." in (out / "shortcuts.md").read_text(encoding="utf-8")
    standard = Plan(good.plan.files, good.plan.questions, scale="standard")
    assert shortcuts.run(standard, good.folder, tmp_path / "standard")
    assert "FAILED" in (tmp_path / "standard" / "shortcuts.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="never in the data folder"):
        shortcuts.run(good.plan, good.folder, good.folder / "run")


# -- the harder haystack ----------------------------------------------------------------------------------------------------

def test_the_probe_passes_the_tiny_haystack(tmp_path):
    import haystack

    assert haystack.VERSION >= 3, "the probe reads the questions' `probe`, which the harder haystack (VERSION 3) has"
    plan = haystack.plan(1, "tiny", haystack.days_before(date(2026, 10, 30), haystack.SCALES["tiny"].days))
    haystack.write(plan, tmp_path / "data")
    start = time.perf_counter()
    failures = shortcuts.run(plan, tmp_path / "data", tmp_path / "run")
    seconds = time.perf_counter() - start
    print(f"\nthe probe of the tiny haystack: {seconds:.1f} s (the target: 10 s)")
    assert failures == []
    saved = json.loads((tmp_path / "run" / "shortcuts.json").read_text(encoding="utf-8"))
    for r in saved["rows"]:
        if not r["absent"]:
            print(f"  {r['qid']} ({r['level']}): area {r['region_broad_shortlist']}/{r['region_broad_rank']}, tuned colour {r['tuned_colour_shortlist']}/{r['tuned_colour_rank']}, region {r['region_colour_shortlist']}/{r['region_colour_rank']}, "
                  f"tuned band {r['tuned_band_shortlist']}/{r['tuned_band_rank']} (others/rank), views {r['views']}, a frame a clip {r['views_per_clip']}")
    media = [q for q in plan.questions if q.get("probe")]
    assert media and [r["qid"] for r in saved["rows"]] == [q["qid"] for q in media]
    assert all(r["colour_shortlist"] or r["change_shortlist"] or r["band_shortlist"] for r in saved["rows"] if not r["absent"]), "a shortlisted distractor for every needle"
    limit = 120 if os.environ.get("PYTEST_XDIST_WORKER") else 30  # the target is 10 s alone; the suite runs 4 workers at once
    assert seconds <= limit, f"the probe takes {seconds:.1f} s, the target is 10 (this run allows {limit})"
