"""The drawings and sounds of the benchmark haystack (`benchmarks/haystack_media.py`): the same spec is the same pixels, samples and
file bytes; each needle's shade survives the encoder (JPEG for a photo, x264 at the capture's crf for the screen and the porch) at 3%
or more of the picture while every distractor of its source has under 0.5% of any needle's shade; the needles meet their area floors;
the alarm's and the doorbell's pitches are the fake decider's `pitch:<hz>` for their own windows and for no distractor's; a VFR screen
segment has the frames `screen_frames` lists (a state's start, an act: a cursor move, typing or a scroll, a minute of the clock, at most
one every half second, one at least every 60 s), with the cursor and the clock drawn, and the reader's change gate sees them; a frame
drawn from a state's layers (the card, scrolled and typed on, with the decorations pasted over through their mask) is the state's picture
exactly when nothing has happened; and the writers keep to a throughput target (screen 20 ms a frame, porch 15 ms a frame, microphone
1 s per 15 minutes of sound), which the tests print (`-s`)."""

from __future__ import annotations

import io
import os
import random
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import haystack_media as hm
import numpy as np
import pytest
from day import RATE
from fake_decider import PITCH_RMS, PITCH_SHARE, pitch_share, shade_share
from PIL import Image

from mms import media, reader

NEEDLE = hm.NEEDLE_RGB
DISTRACTOR = hm.DISTRACTOR_RGB
NEEDLE_SHARE = 0.03  # a needle's share of its shade, after the encoder
DISTRACTOR_SHARE = 0.005  # a distractor's, of any needle's shade of its source
MOST_SHARE = 0.5  # a needle's shade is never most of its frame: no needle is its frame's dominant colour
TARGET_S = {"screen frame": 0.020, "porch frame": 0.015, "mic 15 minutes": 1.0}  # the most seconds a unit may take
# a single process (`-n 0`) is held to the targets; under xdist (`-n 4` runs four workers, and another suite or a GPU run may share the machine) the
# bounds are SLACK times as loose, so that the test fails for a writer that got several times slower and never for a busy machine
START = datetime(2026, 10, 28, 10, 0, 0).timestamp()  # a screen segment starts on a minute: a 30 s state's start is the only frame its clock adds
SLACK = 4.0 if os.environ.get("PYTEST_XDIST_WORKER") else 1.0


def jpeg(image: Image.Image, quality: int = 90) -> Image.Image:
    """`image` as a phone saves it (a JPEG) and the reader hands it to a decider: scaled down, bicubic, to the reader's `max_px`."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    back = Image.open(buffer).convert("RGB")
    size = reader.fit(*back.size, reader.PROFILE["max_px"])
    return back.resize(size, Image.Resampling.BICUBIC) if size != back.size else back


def decode(path: Path) -> list[tuple[float, Image.Image]]:
    """Every frame of a video as the reader hands it to a decider: (seconds into the file, picture at the reader's `max_px`), in order."""
    return list(reader._decode(path, reader._times(path), reader.PROFILE["max_px"]))


def same(a: Image.Image, b: Image.Image) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def decoded(path: Path) -> np.ndarray:
    """A sound file's samples as the reader decodes them (mono float32 at RATE)."""
    return reader._span(path, 0, 1e9, RATE)[1]


def windows(samples: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """The reader's windows of a sound (`audio_s` every `audio_hop_s`) that are not under its `quiet_rms`: (from, to, samples)."""
    size, hop = reader.PROFILE["audio_s"], reader.PROFILE["audio_hop_s"]
    length = len(samples) / RATE
    starts = [k * hop for k in range(int(max(length - size, 0) // hop) + 1)]
    out = []
    for a in starts:
        piece = samples[int(round(a * RATE)):int(round((a + size) * RATE))]
        if float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)))) >= reader.PROFILE["quiet_rms"]:
            out.append((a, a + size, piece))
    return out


def heard(piece: np.ndarray, hz: float) -> bool:
    """The fake decider's `pitch:<hz>` for one window."""
    rms = float(np.sqrt(np.mean(np.square(piece, dtype=np.float64))))
    return rms >= PITCH_RMS and pitch_share(piece, RATE, hz) >= PITCH_SHARE


# -- the constants ----------------------------------------------------------------------------------------------------------

def test_the_shades_and_pitches_keep_their_distance():
    shades = list(NEEDLE.values())
    assert all(hm.distance(a, b) >= hm.SHADE_GAP for i, a in enumerate(shades) for b in shades[i + 1:]), "one shade for each needle"
    assert hm.SHADE_GAP == 2 * hm.SHADE_TOL == 60
    for source, kinds in hm.SOURCE_NEEDLES.items():
        for group in kinds:  # a source's distractors are clear of every needle of the source
            for name, rgb in DISTRACTOR[group].items():
                for kind in kinds:
                    assert hm.distance(rgb, NEEDLE[kind]) >= hm.SHADE_GAP, f"{source}: {group}/{name} is near the {kind} shade"
    assert hm.ALARM_HZ == (3000, 3400) and hm.PITCH_GAP == 0.08 and len(hm.DOORBELL_PAIRS) == 4
    notes = [n for pair in hm.DOORBELL_PAIRS for n in pair]
    for kind, pitches in hm.BEEPER_HZ.items():
        for hz in pitches:
            assert 2400 <= hz <= 2750 or 3750 <= hz <= 4300, f"a {kind} beeps at {hz}"
            assert all(hm.gap(hz, a) >= hm.PITCH_GAP for a in hm.ALARM_HZ), f"{kind} {hz} is near the alarm"
            assert all(hm.gap(hz, n) >= hm.PITCH_GAP for n in notes), f"{kind} {hz} is near a chime note"
    assert all(400 <= hz <= 1000 for hz in hm.TIMER_HZ), "a timer is in the chime's band"
    for hz in (*hm.TIMER_HZ, *hm.DING_HZ, *hm.RING_HZ, *(hz * 480 / 440 for hz in hm.RING_HZ)):  # timers, dings and phone rings, a ring's second tone too
        assert all(hm.gap(hz, n) >= hm.PITCH_GAP for n in notes), f"{hz} is near a chime note"
        assert all(not (a / (1 + hm.PITCH_GAP) < hz < b * (1 + hm.PITCH_GAP)) for a, b in [hm.ALARM_HZ]), f"{hz} is near the alarm"


def test_a_colour_near_a_needles_shade_is_refused():
    near_car = tuple(c + 5 for c in NEEDLE["car"][:2]) + (NEEDLE["car"][2],)
    with pytest.raises(ValueError, match="car needle's shade"):
        hm.photo({"scene": "object", "rgb": near_car, "seed": 1})
    with pytest.raises(ValueError, match="car needle's shade"):
        hm.photo({"scene": "car", "rgb": near_car, "seed": 1})  # a red car is the needle's shade exactly, or no red at all
    with pytest.raises(ValueError, match="error needle's shade"):
        hm.screen_state({"app": "editor", "overlay": {"kind": "warning", "rgb": list(NEEDLE["error"]), "code": "W-1"}, "seed": 1})
    with pytest.raises(ValueError, match="error needle's shade"):
        hm.screen_state({"app": "editor", "overlay": {"kind": "error", "rgb": [200, 30, 30], "code": "E-1"}, "seed": 1})
    with pytest.raises(ValueError, match="error needle's shade"):
        hm.screen_state({"app": "browser", "accent": {"kind": "banner", "rgb": [180, 30, 40]}, "seed": 1})
    with pytest.raises(ValueError, match="parcel needle's shade"):
        hm.porch_frame({"length_s": 5, "seed": 1, "events": [{"kind": "dog", "from": 0, "to": 5, "rgb": [195, 140, 90]}], "parcel": None}, 1, "t")


# -- determinism ------------------------------------------------------------------------------------------------------------

PHOTO_SPECS = [{"scene": "sign", "rgb": NEEDLE["sign"], "word": "GATE", "number": "47", "seed": 7},
               {"scene": "sign", "rgb": DISTRACTOR["sign"]["blue"], "word": "ROOM", "number": "318", "box": [0.3, 0.1, 0.7, 0.95], "seed": 8},
               {"scene": "car", "rgb": NEEDLE["car"], "seed": 9}] + [
    {"scene": scene, "rgb": None, "seed": 10 + k} for k, scene in enumerate(("object", "room", "food", "plant", "street"))]


def test_a_spec_is_the_same_pixels_and_samples_every_time():
    for spec in PHOTO_SPECS:
        assert same(hm.photo(spec), hm.photo(dict(spec))), spec
        others = (hm.photo({**spec, "seed": spec["seed"] + k}) for k in range(1, 8))
        assert any(not same(hm.photo(spec), other) for other in others), f"the seed makes no difference to {spec['scene']}"
    state = {"app": "browser", "overlay": {"kind": "confirm"}, "accent": {"kind": "banner"}, "seed": 4}
    assert same(hm.screen_state(state), hm.screen_state(dict(state))) and not same(hm.screen_state(state), hm.screen_state({**state, "seed": 5}))
    clip = {"length_s": 10, "seed": 2, "events": [{"kind": "walker", "from": 0, "to": 8}], "parcel": [3, 8]}
    assert same(hm.porch_frame(clip, 4, "t"), hm.porch_frame(dict(clip), 4, "t"))
    assert not same(hm.porch_frame(clip, 4, "t"), hm.porch_frame({**clip, "seed": 3}, 4, "t"))
    spec = mic_spec(300)
    assert np.array_equal(hm.mic_samples(spec), hm.mic_samples(dict(spec)))
    assert not np.array_equal(hm.mic_samples(spec), hm.mic_samples({**spec, "seed": 99}))
    for sound in spec["sounds"]:
        assert np.array_equal(hm.sound(sound), hm.sound(dict(sound))), sound["kind"]


def test_a_file_is_the_same_bytes_every_time(tmp_path):
    screen = {"length_s": 90, "states": [{"from": 0, "to": 45, "app": "chat", "overlay": None, "accent": None, "seed": 1, "cursor": [100, 100],
                                          "acts": [[6, "move", 140, 120], [7, "type", 2], [8, "move", 200, 90], [40, "scroll", 1]]},
                                         {"from": 45, "to": 90, "app": "editor", "overlay": {"kind": "info", "msg": "Done"}, "accent": None, "seed": 2,
                                          "cursor": [200, 90], "acts": [[20, "type", 3], [21, "type", 6], [30, "move", 300, 310]]}]}
    porch = {"length_s": 6, "seed": 3, "events": [{"kind": "cat", "from": 0, "to": 5}], "parcel": None}
    mic = {"length_s": 20, "seed": 4, "bed_dbfs": -50, "tv": [[2, 10]], "sounds": [{"kind": "bark", "at": 12.0, "dur": 1.0, "seed": 1}]}
    for name, write in (("a.mkv", lambda p, w: hm.write_screen(p, screen, START + 17, w)), ("b.mp4", lambda p, w: hm.write_porch(p, porch, "2026-10-28 15:35:44", w)),
                        ("c.ogg", lambda p, w: hm.write_mic(p, mic, w))):
        paths = []
        for k in range(2):
            path = tmp_path / f"out{k}" / name
            write(path, tmp_path / f"work{k}")
            paths.append(path)
        assert paths[0].read_bytes() == paths[1].read_bytes(), name
    assert not list((tmp_path / "work0").iterdir()), "a finished file is moved out of the working folder"


# -- photos -----------------------------------------------------------------------------------------------------------------

def photo_catalogue() -> list[tuple[str, dict, str | None]]:
    """(label, spec, the needle it is or None): every scene with every colour its source's distractors have."""
    out: list[tuple[str, dict, str | None]] = []
    for seed in (1, 2, 3):
        word = ("GATE", "BAY", "LEVEL")[seed - 1]
        out.append((f"green sign {seed}", {"scene": "sign", "rgb": NEEDLE["sign"], "word": word, "number": str(40 + 7 * seed), "seed": seed}, "sign"))
        out.append((f"red car {seed}", {"scene": "car", "rgb": NEEDLE["car"], "seed": seed}, "car"))
    for colour in ("yellow", "white", "blue", "grey"):
        out.append((f"{colour} sign", {"scene": "sign", "rgb": DISTRACTOR["sign"][colour], "word": "ROOM", "number": "208", "seed": 20}, None))
        out.append((f"{colour} car", {"scene": "car", "rgb": DISTRACTOR["car"][colour], "seed": 21}, None))
    out.append(("green car", {"scene": "car", "rgb": DISTRACTOR["sign"]["car"], "seed": 22}, None))
    for scene in ("object", "room", "food", "plant", "street"):
        out.append((f"{scene}, its own colours", {"scene": scene, "rgb": None, "seed": 30}, None))
    for scene, name in (("object", "shirt"), ("room", "door"), ("plant", "plant"), ("street", "shirt"), ("food", "plant")):  # green things that are no sign
        out.append((f"{scene} with a green {name}", {"scene": scene, "rgb": DISTRACTOR["sign"][name], "seed": 31}, None))
    for scene, name in (("object", "umbrella"), ("room", "door"), ("food", "tomato"), ("street", "umbrella")):  # red things that are no car
        out.append((f"{scene} with a red {name}", {"scene": scene, "rgb": DISTRACTOR["car"][name], "seed": 32}, None))
    return out


def test_a_needle_photo_keeps_its_shade_through_jpeg_and_no_distractor_has_it():
    seen = set()
    for label, spec, needle in photo_catalogue():
        image = hm.photo(spec)
        assert image.size == hm.PHOTO
        shown = jpeg(image)  # (what the phone saves: a share that holds after it held before, since the encoder only blurs edges)
        for kind in ("sign", "car"):
            share = shade_share(shown, NEEDLE[kind])
            if kind == needle:
                assert NEEDLE_SHARE <= share < MOST_SHARE, f"{label} as a jpeg: {share:.1%} of the {kind} shade, not {NEEDLE_SHARE:.0%} to {MOST_SHARE:.0%}"
                seen.add(kind)
            else:
                assert share < DISTRACTOR_SHARE, f"{label} as a jpeg: {share:.2%} of the {kind} shade, a distractor's must be under {DISTRACTOR_SHARE:.1%}"
    assert seen == {"sign", "car"}


def test_a_needle_sign_is_a_fifth_to_a_third_of_the_photo_and_a_red_car_a_third():
    for seed in range(300):
        x0, y0, x1, y1 = hm.sign_box({"seed": seed})
        assert 0.20 <= (x1 - x0) * (y1 - y0) <= 0.35 and 0 <= x0 and x1 <= 1 and 0 <= y0 and y1 <= 1, f"seed {seed}: the sign's box {(x0, y0, x1, y1)}"
        assert hm.sign_box({"seed": seed}) == hm.sign_box({"seed": seed})
    for seed in range(8):
        x0, y0, x1, y1 = hm.sign_box({"seed": seed})
        sign = hm.photo({"scene": "sign", "rgb": NEEDLE["sign"], "word": "PLATFORM", "number": "318", "seed": seed})
        assert shade_share(jpeg(sign), NEEDLE["sign"]) >= 0.6 * (x1 - x0) * (y1 - y0), f"seed {seed}: the sign's own colour is most of its plate"
    box = [0.30, 0.10, 0.70, 0.95]  # a box in the spec is the sign's
    arr = np.asarray(hm.photo({"scene": "sign", "rgb": NEEDLE["sign"], "word": "GATE", "number": "47", "box": box, "seed": 1}), dtype=np.int32)
    near = (np.square(arr - np.array(NEEDLE["sign"])).sum(axis=2) <= hm.SHADE_TOL ** 2)
    ys, xs = np.nonzero(near)
    assert abs(xs.min() / 1280 - 0.30) < 0.02 and abs(xs.max() / 1280 - 0.70) < 0.02 and abs(ys.min() / 720 - 0.10) < 0.02
    for seed in range(10):
        car = hm.photo({"scene": "car", "rgb": NEEDLE["car"], "seed": seed})
        assert shade_share(car, NEEDLE["car"]) >= 0.30 and shade_share(jpeg(car), NEEDLE["car"]) >= 0.30, f"seed {seed}: the car's body is under 30% of the photo"


# -- the screen -------------------------------------------------------------------------------------------------------------

def error_overlay(n: int) -> dict:
    return {"kind": "error", "rgb": list(NEEDLE["error"]), "code": f"E-{4400 + n}", "msg": ("disk full", "out of memory", "connection lost")[n % 3]}


def screen_catalogue() -> list[tuple[str, dict, bool]]:
    """(label, state, whether it is the error needle): each app with the error and with each other overlay, and with each accent."""
    out = []
    for n, app in enumerate(hm.APPS):
        out.append((f"{app}", {"app": app, "overlay": None, "accent": None, "seed": n}, False))
        out.append((f"{app} with an error", {"app": app, "overlay": error_overlay(n), "accent": None, "seed": 10 + n}, True))
        for kind in ("warning", "info", "save", "confirm") if n % 2 == 0 else ("warning", "confirm"):
            overlay = {"kind": kind, "code": f"W-{1200 + n}" if kind == "warning" else None}
            out.append((f"{app} with a {kind}", {"app": app, "overlay": overlay, "accent": None, "seed": 20 + n}, False))
    for n, app in enumerate(("browser", "sheet")):
        for kind in hm.ACCENTS:
            out.append((f"{app} with a {kind}", {"app": app, "overlay": None, "accent": {"kind": kind}, "seed": 40 + n}, False))
    out.append(("an error over a banner", {"app": "browser", "overlay": error_overlay(1), "accent": {"kind": "banner"}, "seed": 50}, True))
    out.append(("a save prompt over cells", {"app": "sheet", "overlay": {"kind": "save"}, "accent": {"kind": "cells"}, "seed": 51}, False))
    return out


@pytest.fixture(scope="module")
def screen_file(tmp_path_factory):
    """The catalogue as one VFR segment of states of 30 s, with no cursor moves, from START: its path, its spec, and the frames as decoded
    (a frame for each state, since the clock changes only where one starts)."""
    states = [{**state, "from": 30 * k, "to": 30 * (k + 1)} for k, (_, state, _) in enumerate(screen_catalogue())]
    spec = {"length_s": 30 * len(states), "states": states}
    path = tmp_path_factory.mktemp("screen") / "screen.mkv"
    hm.write_screen(path, spec, START, path.parent / "work")
    return path, spec, decode(path)


def test_an_error_dialog_keeps_its_shade_through_x264_and_no_other_state_has_it(screen_file):
    path, spec, frames = screen_file
    catalogue = screen_catalogue()
    assert len(frames) == len(catalogue)
    for (label, state, needle), (_, frame) in zip(catalogue, frames):
        share = shade_share(frame, NEEDLE["error"])
        if needle:
            assert NEEDLE_SHARE <= share < MOST_SHARE, f"{label} encoded: {share:.1%} of the error shade, not {NEEDLE_SHARE:.0%} to {MOST_SHARE:.0%}"
        else:
            assert share < DISTRACTOR_SHARE, f"{label} encoded: {share:.2%} of the error shade, a distractor's must be under {DISTRACTOR_SHARE:.1%}"


def bbox_of(image: Image.Image, rgb) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), inclusive, of the pixels that are exactly `rgb`."""
    ys, xs = np.nonzero((np.asarray(image) == np.array(rgb)).all(axis=2))
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_an_error_dialog_is_centred_a_sixth_of_the_screen_with_a_readable_title_and_so_are_its_distractors():
    w, h = hm.SCREEN
    headers = {"error": NEEDLE["error"], **{kind: DISTRACTOR["error"][kind] for kind in ("warning", "info", "save", "confirm")}}
    boxes = {}
    for kind, rgb in headers.items():
        overlay = {"kind": kind, "rgb": list(rgb) if kind == "error" else None, "code": "E-4417", "msg": "disk full"}
        state = {"app": "editor", "overlay": overlay, "seed": 5}
        x0, y0, x1, y1 = bbox_of(hm.screen_state(state).crop((0, 0, w, h // 2)), rgb)  # (a confirm's button is its header's red, below the middle)
        boxes[kind] = (x0, x1, y0)
        assert abs((x0 + x1 + 1) / 2 - w / 2) <= 1 and abs(x1 + 1 - x0 - hm.DIALOG[0] * w) <= 2, f"the {kind} dialog is 45% wide and centred"
        assert abs((y1 + 1 - y0) - 0.38 * hm.DIALOG[1] * h) <= 2, f"the {kind} dialog's header is 38% of its height"
        assert abs(y0 + hm.DIALOG[1] * h / 2 - h / 2) <= 2, f"the {kind} dialog is centred"
    assert len(set(boxes.values())) == 1, "the dialogs have one place and one size"
    assert hm.DIALOG[0] * hm.DIALOG[1] >= 0.15, "the error dialog is 15% of the screen or more"
    # the dialog is what changes: its box and its shadow, and nothing of the card behind
    state = {"app": "terminal", "overlay": error_overlay(0), "seed": 6}
    with_dialog, without = (np.asarray(hm.screen_state(s), dtype=np.int16) for s in (state, {**state, "overlay": None}))
    changed = (with_dialog != without).any(axis=2)
    ys, xs = np.nonzero(changed)
    assert (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) >= 0.15 * w * h
    assert xs.min() >= 0.275 * w - 2 and xs.max() <= 0.725 * w + 16 and ys.min() >= 0.325 * h - 2 and ys.max() <= 0.675 * h + 18
    # the title: its letters, scaled to 640x360, are about 20 px tall (the capital letters' height is 70% of that): readable there, about 8 px in a contact sheet 5 frames across
    image = hm.screen_state({"app": "editor", "overlay": error_overlay(0), "seed": 7})
    x0, y0, x1, y1 = bbox_of(image, NEEDLE["error"])
    header = np.asarray(image.crop((x0, y0, x1 + 1, y1 + 1)))
    white = (header > 240).all(axis=2)
    rows = np.nonzero(white.any(axis=1))[0]
    capitals = (rows.max() - rows.min() + 1) / 2  # (14 px measured)
    assert 12 <= capitals <= 16, f"the error's capitals are {capitals:.0f} px tall at 640x360, not 12 to 16"
    assert 20 <= hm.TITLE_PX / 2 <= 24, "the title is 20 to 24 px at 640x360: readable there, small in a contact sheet"


def test_a_dialog_opens_where_it_is_told_within_the_places_of_its_area():
    """`at` is a dialog's centre: the error's header and every other kind's are there at every corner of the places a dialog can be, the whole
    dialog is inside DIALOG_AREA, and a place outside DIALOG_AT is refused."""
    w, h = hm.SCREEN
    ax0, ay0, ax1, ay1 = hm.DIALOG_AREA
    assert 0.1 < ax0 < 0.2 and 0.8 < ax1 < 0.9 and 0.2 < ay0 < 0.3 and 0.7 < ay1 < 0.8, "the middle part of the screen"
    kinds = {"error": NEEDLE["error"], **{kind: DISTRACTOR["error"][kind] for kind in ("warning", "info", "save", "confirm")}}
    for cx in hm.DIALOG_AT[0]:
        for cy in hm.DIALOG_AT[1]:
            for kind, rgb in kinds.items():
                overlay = {"kind": kind, "rgb": list(rgb) if kind == "error" else None, "code": "E-4417", "msg": "disk full", "at": [cx, cy]}
                image = hm.screen_state({"app": "editor", "overlay": overlay, "seed": 2})
                x0, y0, x1, y1 = bbox_of(image.crop((0, 0, w, h // 2)), rgb)  # (a confirm's button is its header's red, in the header's width)
                assert abs((x0 + x1 + 1) / 2 - w * cx) <= 1.5 and abs(x1 + 1 - x0 - hm.DIALOG[0] * w) <= 2, f"{kind} at {cx}, {cy}: the header"
                top = y0  # the header is the dialog's top: the dialog is DIALOG tall from there
                assert abs(top + hm.DIALOG[1] * h / 2 - h * cy) <= 2, f"{kind} at {cx}, {cy}: the dialog's centre"
                assert ax0 * w - 2 <= x0 and x1 + 1 <= ax1 * w + 2 and ay0 * h - 2 <= top and top + hm.DIALOG[1] * h <= ay1 * h + 2, f"{kind} is outside DIALOG_AREA"
    centred = hm.screen_state({"app": "editor", "overlay": error_overlay(0), "seed": 2})
    assert same(centred, hm.screen_state({"app": "editor", "overlay": {**error_overlay(0), "at": None}, "seed": 2})) and \
        same(centred, hm.screen_state({"app": "editor", "overlay": {**error_overlay(0), "at": [0.5, 0.5]}, "seed": 2})), "no `at` is the middle"
    for bad in ([0.2, 0.5], [0.5, 0.9], [0.7, 0.5]):
        with pytest.raises(ValueError, match="outside the places of DIALOG_AT"):
            hm.screen_state({"app": "editor", "overlay": {**error_overlay(0), "at": bad}, "seed": 2})
    r = random.Random(4)
    places = [hm.dialog_at(r) for _ in range(200)]
    assert all(hm.DIALOG_AT[0][0] <= x <= hm.DIALOG_AT[0][1] and hm.DIALOG_AT[1][0] <= y <= hm.DIALOG_AT[1][1] for x, y in places) and len(set(map(tuple, places))) > 190
    assert hm.dialog_at(random.Random(4)) == places[0], "deterministic from the generator"


def test_a_vfr_screen_segment_with_no_moves_decodes_to_one_frame_per_state_at_the_states_time(screen_file):
    path, spec, frames = screen_file
    states = spec["states"]
    assert len(frames) == len(states) == len(hm.screen_frames(spec, START)), "one frame per state"
    assert [round(sec, 3) for sec, _ in frames] == [float(s["from"]) for s in states]
    assert media.duration(path) == spec["length_s"], "the file lasts as long as the segment"
    assert len({int(s["from"] // reader.PROFILE["every_s"]) for s in states}) == len(states), "each state is a window of its own"
    assert all(image.size == (hm.SCREEN[0] // 2, hm.SCREEN[1] // 2) for _, image in frames), "the reader scales a screen to 640x360"
    # the reader's own count of a still stretch: it slots a clip's frames by `every_s`, and a state is one frame, however long it lasts
    assert [pts for pts, _ in reader._times(path)] == [round(s["from"] * 1000) for s in states]


def test_a_screen_segment_of_one_state_or_a_short_last_one_writes_and_decodes(tmp_path):
    """The recorder cuts a block at a state change, so a segment can be one state (as short as 10 s, as long as a state and its tail,
    68 s) or the short last one of a block: each is a file with the frames `screen_frames` lists (a state's start, and the clock's minute
    of the 68 s one), the first at 0, that lasts the segment."""
    shapes = {"a state alone, the shortest": (10, [(0, 10)]), "a state alone, the longest": (68, [(0, 68)]),
              "a short last segment": (25, [(0, 10), (10, 25)]), "an error alone": (31, [(0, 31)])}
    for k, (label, (length, spans)) in enumerate(shapes.items()):
        error = label == "an error alone"
        states = [{"from": a, "to": b, "app": hm.APPS[(k + i) % len(hm.APPS)], "overlay": error_overlay(k) if error else None, "accent": None,
                   "seed": 300 + 10 * k + i} for i, (a, b) in enumerate(spans)]
        path = tmp_path / f"segment-{k}.mkv"
        spec = {"length_s": length, "states": states}
        hm.write_screen(path, spec, START, tmp_path / "work")
        frames = decode(path)
        wanted = [frame.second for frame in hm.screen_frames(spec, START)]
        assert {a for a, _ in spans} <= set(wanted) and wanted == sorted(wanted), label
        assert [round(sec, 3) for sec, _ in frames] == [float(sec) for sec in wanted], label
        assert media.duration(path) == length, f"{label}: the file lasts as long as the segment"
        assert [pts for pts, _ in reader._times(path)] == [round(sec * 1000) for sec in wanted], label
        shares = [shade_share(image, NEEDLE["error"]) for _, image in frames]
        assert all(NEEDLE_SHARE <= share < MOST_SHARE for share in shares) if error else all(share < DISTRACTOR_SHARE for share in shares), f"{label}: {shares}"


def states_of(*spans: tuple[int, int, list]) -> list[dict]:
    return [{"from": a, "to": b, "app": hm.APPS[k % len(hm.APPS)], "overlay": None, "accent": None, "seed": k, "cursor": [50 * k, 60],
             "acts": acts} for k, (a, b, acts) in enumerate(spans)]


def test_a_segments_frames_are_at_half_seconds_the_states_starts_the_acts_the_clocks_minutes_and_a_floor_of_60_s():
    acts = [[6, "move", 10, 10], [7, "type", 2], [8, "type", 5], [40, "scroll", 1]]  # at 3, 3.5, 4 and 20 s
    spec = {"length_s": 100, "states": states_of((0, 40, acts), (40, 100, [[31, "move", 90, 90]]))}
    start = datetime(2026, 10, 28, 14, 59, 40).timestamp()  # the clock changes at 20 and 80 s
    frames = hm.screen_frames(spec, start)
    assert [f.second for f in frames] == [0, 3, 3.5, 4, 20, 40, 55.5, 80]
    assert [(f.state, f.cursor, f.typed, f.lines, f.clock, f.date) for f in frames] == [
        (0, (0, 60), 0, 0, "14:59", "2026-10-28"), (0, (10, 10), 0, 0, "14:59", "2026-10-28"), (0, (10, 10), 2, 0, "14:59", "2026-10-28"),
        (0, (10, 10), 5, 0, "14:59", "2026-10-28"), (0, (10, 10), 0, 1, "15:00", "2026-10-28"), (1, (50, 60), 0, 0, "15:00", "2026-10-28"),
        (1, (90, 90), 0, 0, "15:00", "2026-10-28"), (1, (90, 90), 0, 0, "15:01", "2026-10-28")], \
        "a state's start shows the cursor where the state names it and an empty typing line; a move, a type and a scroll change one thing; a scroll empties the line"
    assert hm.screen_frames(spec, start) == frames, "a function of its spec"
    midnight = datetime(2026, 10, 28, 23, 59, 50).timestamp()
    assert (hm.screen_frames({"length_s": 30, "states": states_of((0, 30, []))}, midnight)[-1].clock,
            hm.screen_frames({"length_s": 30, "states": states_of((0, 30, []))}, midnight)[-1].date) == ("00:00", "2026-10-29")


def test_a_frame_is_kept_at_least_every_60_s_and_at_most_one_a_tick():
    assert hm.with_floor([0], 200) == [0, 60, 120, 180]
    assert hm.with_floor([0, 61, 62], 70) == [0, 60, 61, 62], "a gap over 60 s gets a frame 60 s after the last"
    assert hm.with_floor([0, 60, 90], 150) == [0, 60, 90] and hm.with_floor([5], 6) == [5]
    assert hm.with_floor([0, 0.5, 61], 70) == [0, 0.5, 60.5, 61], "in half seconds too"
    spec = {"length_s": 130, "states": states_of((0, 70, [[4, "move", 5, 5]]), (70, 130, []))}
    frames = hm.screen_frames(spec, START + 31)  # the minutes are at 29 s and 89 s
    seconds = [f.second for f in frames]
    assert seconds == [0, 2, 29, 70, 89]
    assert all(b - a <= hm.FLOOR_S for a, b in zip(seconds, seconds[1:])) and 130 - seconds[-1] <= hm.FLOOR_S
    busy = {"length_s": 12, "states": states_of((0, 12, [[tick, "move", 10 * tick, 10] for tick in range(1, 24)]))}
    assert [f.second for f in hm.screen_frames(busy, START)] == [tick / 2 for tick in range(24)], "an act every half second is a frame every half second, never two"
    assert hm.TICK_S == 0.5


def test_the_file_has_the_frame_count_and_the_seconds_of_its_list(tmp_path):
    spec = {"length_s": 75, "states": states_of((0, 25, [[4, "move", 300, 200], [5, "type", 3], [6, "type", 5], [18, "scroll", 2]]),
                                                (25, 75, [[10, "move", 600, 300], [11, "type", 2]]))}
    path = tmp_path / "s.mkv"
    hm.write_screen(path, spec, START + 40, tmp_path / "w")
    frames = hm.screen_frames(spec, START + 40)
    assert [f.second for f in frames] == [0, 2, 2.5, 3, 9, 20, 25, 30, 30.5], "starts, acts and the clock's minute at 20 s"
    times = reader._times(path)
    assert [round(sec, 3) for _, sec in times] == [float(f.second) for f in frames] and media.duration(path) == spec["length_s"]
    assert [pts for pts, _ in times] == [round(f.second * 1000) for f in frames], "at the millisecond"


def test_the_cursor_is_a_white_arrow_with_a_black_outline_and_the_clock_is_at_the_right_of_the_taskbar():
    base = hm.screen_state({"app": "editor", "overlay": None, "accent": None, "seed": 3})
    layers = hm.screen_layers({"app": "editor", "overlay": None, "accent": None, "seed": 3})
    w, h = hm.SCREEN
    frame = hm.ScreenFrame(0, 0, (400, 300), "14:05", "2026-10-28")
    image = hm.screen_frame(layers, frame)
    a, b = np.asarray(base, dtype=np.int16), np.asarray(image, dtype=np.int16)
    changed = np.argwhere(np.abs(a - b).sum(axis=2) > 0)
    arrow = changed[changed[:, 0] < h - hm.TASKBAR_H]
    assert (arrow[:, 1].min(), arrow[:, 0].min()) == (400, 300), "the arrow's tip is the cursor"
    assert (arrow[:, 1].max() - 400 + 1, arrow[:, 0].max() - 300 + 1) == (12, 19), "12 by 19 px"
    patch = b[300:319, 400:412]
    assert (patch == 255).all(axis=2).any() and (patch == 0).all(axis=2).any(), "white inside, black outline"
    clock = changed[changed[:, 0] >= h - hm.TASKBAR_H]
    assert clock[:, 1].min() > w - 160 and clock[:, 0].min() >= h - hm.TASKBAR_H, "the time and the date at the right of the taskbar"
    assert (b[h - hm.TASKBAR_H + 4:, : w // 2 - 200] == np.asarray(hm.TASKBAR_RGB)).all(), "dark strip"
    again = hm.screen_frame(layers, hm.ScreenFrame(0, 0, (400, 300), "14:06", "2026-10-28"))
    assert not same(image, again) and same(image, hm.screen_frame(layers, frame)), "a minute of the clock is a different picture"
    assert same(layers.scrolled(0), hm.screen_layers({"app": "editor", "overlay": None, "accent": None, "seed": 3}).card), "the card is not drawn on"


def test_the_taskbar_is_clear_of_the_dialogs_and_of_every_needle_shade():
    y1 = hm.DIALOG_AREA[3]
    assert y1 * hm.SCREEN[1] + 14 < hm.SCREEN[1] - hm.TASKBAR_H, "a dialog and its shadow end above the taskbar"
    for rgb in (hm.TASKBAR_RGB, hm.TASKBAR_LINE, hm.TASKBAR_ICON, hm.TASKBAR_LOGO, hm.TASKBAR_INK, (255, 255, 255), (0, 0, 0)):
        for kind in hm.SOURCE_NEEDLES["screen"]:
            assert hm.distance(rgb, NEEDLE[kind]) >= hm.SHADE_GAP, (rgb, kind)


LAYER_ACCENTS = (None, {"kind": "banner"}, {"kind": "cells"}, {"kind": "badge"}, {"kind": "toast"})
LAYER_OVERLAYS = (None, {**error_overlay(0), "at": [0.4, 0.55]}, {"kind": "save"}, {"kind": "confirm", "at": [0.6, 0.45]})


def test_the_layers_make_screen_state_exactly_for_every_app_with_and_without_an_accent_and_an_overlay():
    """The card with the decorations pasted over it through the mask is `screen_state`, pixel for pixel: the mask is exact (found on a
    black and a white background, so a dialog's white body over a white page is in it) and a decoration is opaque."""
    for app in hm.APPS:
        for accent in LAYER_ACCENTS:
            for overlay in LAYER_OVERLAYS:
                state = {"app": app, "overlay": overlay, "accent": accent, "seed": 3}
                layers = hm.screen_layers(state)
                composed = layers.scrolled(0).copy()
                composed.paste(layers.deco, mask=layers.mask)
                assert same(composed, hm.screen_state(state)), f"{app} with {accent} and {overlay and overlay['kind']}"
                frame = hm.ScreenFrame(0, 0, (700, 300), "10:00", "2026-10-28")  # (and a frame with nothing happened: the pasted regions, the clock, the cursor)
                expected = hm.screen_state(state)
                hm.stamp(expected, frame)
                assert same(hm.screen_frame(layers, frame), expected), f"{app} with {accent} and {overlay and overlay['kind']}: the frame is not the state"
                assert set(np.unique(np.asarray(layers.mask))) <= {0, 255}, "a pixel is a decoration's or the card's"
    state = {"app": "chat", "overlay": {"kind": "info"}, "accent": None, "seed": 3}  # a white dialog over a white page is not see-through
    layers = hm.screen_layers(state)
    w, h = hm.SCREEN
    mask = np.asarray(layers.mask)
    assert (mask[h // 2, w // 2] == 255) and not mask[10, 10] and mask[h - 5, 10] == 255, "the dialog's middle and the taskbar are the decorations'"
    assert mask[int(h * 0.45):int(h * 0.55), int(w * 0.4):int(w * 0.6)].all(), "a dialog's body is all in it, whatever the card shows there"
    again = hm.screen_layers(state)
    assert same(layers.card, again.card) and same(layers.deco, again.deco) and np.array_equal(np.asarray(layers.mask), np.asarray(again.mask))
    assert layers.text == again.text and layers.text != hm.screen_layers({**state, "seed": 4}).text, "the text typed is from the state's seed"


def test_the_typing_lines_fit_their_apps_clear_of_the_taskbar_and_of_the_edge_of_what_they_are_typed_in_with_an_ink_that_is_no_needles():
    for app, typing in hm.TYPING.items():
        x0, y0, x1, y1 = typing.box
        assert (y1 - y0) % typing.line_h == 0 and 0 <= x0 < x1 <= hm.SCREEN[0] and 0 <= y0 < y1 <= hm.SCREEN[1] - hm.TASKBAR_H, f"{app}: the box is whole lines above the taskbar"
        top, bottom = hm._rows(app)
        rx0, ry0, rx1, ry1 = typing.rect
        assert typing.cap >= 30 and ry0 <= top and bottom <= ry1 < hm.SCREEN[1] - hm.TASKBAR_H, f"{app}: the line holds the text's whole height, clear of the taskbar"
        font = hm.font(typing.size)
        for seed in range(30):
            text = hm.typing_text(seed)
            for lines in (0, 1, 2, 7, 13):
                shown = hm.typed_text(text, lines, typing.cap, typing.prefix)
                assert typing.at[0] + font.getlength(shown) + 2 * typing.stroke <= rx1, f"{app}: {typing.cap} characters do not fit the line (seed {seed}, {lines} lines)"
                assert len(shown.replace(" ", "")) == typing.cap + len(typing.prefix.replace(" ", ""))
        # the edge of the line is the background of what it is typed in (an input box, the address bar, the formula bar, the page): the fill stays inside it
        layers = hm.screen_layers({"app": app, "overlay": None, "accent": None, "seed": 5})
        card = np.asarray(layers.card)
        edge = np.concatenate([card[ry0, rx0:rx1 + 1], card[ry1, rx0:rx1 + 1], card[ry0:ry1 + 1, rx0], card[ry0:ry1 + 1, rx1]])
        assert (edge == np.array(layers.bg)).all(), f"{app}: the typing line's fill reaches past the background it is typed on"
        for rgb in typing.ink or ():
            for kind in hm.SOURCE_NEEDLES["screen"]:
                assert hm.distance(rgb, NEEDLE[kind]) >= hm.SHADE_GAP, f"{app}'s ink {rgb} is near the {kind} shade"


def test_typed_text_counts_visible_characters_so_that_every_character_typed_changes_the_picture():
    text = "ab cd  ef"
    assert hm.typed_text(text, 0, 1) == "a" and hm.typed_text(text, 0, 2) == "ab" and hm.typed_text(text, 0, 3) == "ab c"
    assert hm.typed_text(text, 0, 6) == "ab cd  ef" and hm.typed_text(text, 0, 50) == text and hm.typed_text(text, 0, 2, "- ") == "- ab"
    long = hm.typing_text(3)
    assert long == hm.typing_text(3) and long != hm.typing_text(4)
    for n in range(1, 60):
        shown, before = hm.typed_text(long, 2, n), hm.typed_text(long, 2, n - 1)
        assert len(shown.replace(" ", "")) == n and shown.startswith(before) and not shown.endswith(" ")


def test_the_typing_line_is_capped_where_a_decoration_covers_it_and_the_cap_is_what_shows():
    """`typing_limit` is the x that a decoration over the text's rows leaves visible, never more than the mask of the layers leaves; and
    `typing_cap` is the most visible characters whose text ends before it."""
    capped = 0
    for app, typing in hm.TYPING.items():
        top, bottom = hm._rows(app)
        for accent in LAYER_ACCENTS:
            for overlay in LAYER_OVERLAYS:
                state = {"app": app, "overlay": overlay, "accent": accent, "seed": 3}
                mask = np.asarray(hm.screen_layers(state).mask)[top:bottom, typing.at[0]:] == 255
                columns = np.flatnonzero(mask.any(axis=0))
                seen = min(typing.rect[2], typing.at[0] + int(columns[0])) if len(columns) else typing.rect[2]
                limit = hm.typing_limit(state)
                label = f"{app} with {accent and accent['kind']} and {overlay and overlay['kind']}"
                assert limit <= seen, f"{label}: the line shows to {limit}, the mask leaves {seen}"
                if overlay is None:
                    assert limit == seen, f"{label}: {limit} against {seen}"
                font = hm.font(typing.size)
                for lines in (0, 1, 4):
                    cap = hm.typing_cap(state, lines)
                    text = hm.typing_text(3)
                    assert 0 <= cap <= typing.cap, label
                    fits = lambda n: typing.at[0] + font.getlength(hm.typed_text(text, lines, n, typing.prefix)) + 2 * typing.stroke <= limit  # noqa: E731
                    assert cap == 0 or fits(cap), f"{label}: {cap} characters reach under the decoration"
                    assert cap == typing.cap or not fits(cap + 1), f"{label}: {cap + 1} characters would still show"
                capped += hm.typing_cap(state, 0) < typing.cap
    assert capped >= 8, "a toast over the editor's, terminal's and slides' lines, and a dialog over some, cut them short"
    toast = {"app": "editor", "overlay": None, "accent": {"kind": "toast"}, "seed": 3}
    assert 0 < hm.typing_cap(toast, 0) < hm.TYPING["editor"].cap and hm.typing_cap({**toast, "accent": None}, 0) == hm.TYPING["editor"].cap
    assert hm.typing_cap({**toast, "app": "browser"}, 0) == hm.TYPING["browser"].cap, "a toast is clear of the address bar"
    hidden = {"app": "slides", "overlay": {"kind": "save", "at": [0.38, 0.6]}, "accent": None, "seed": 3}  # the dialog covers the slide line's start
    assert hm.typing_cap(hidden, 0) == 0


def test_a_typing_frame_changes_a_small_area_a_scroll_frame_a_large_one_and_the_decorations_never_move():
    """For every app, with and without a dialog, an accent and the taskbar over it: typing changes only the typing line (under 1% of the
    pixels), a scroll changes the card's text (2% or more, at least 4 times as much), and the decoration pixels are the same in all."""
    w, h = hm.SCREEN
    quiet = dict(cursor=(3, 3), clock="10:00", date="2026-10-28")  # (the cursor is clear of every decoration)
    for app in hm.APPS:
        for decorated in (False, True):
            state = {"app": app, "overlay": {"kind": "save", "at": [0.5, 0.5]} if decorated else None,
                     "accent": {"kind": "toast"} if decorated else None, "seed": 4}
            layers = hm.screen_layers(state)
            frame = lambda typed=0, lines=0: hm.screen_frame(layers, hm.ScreenFrame(0, 0, typed=typed, lines=lines, **quiet))  # noqa: E731
            rest = np.asarray(frame(), dtype=np.int16)
            typed = np.asarray(frame(typed=3), dtype=np.int16)
            scrolled = np.asarray(frame(lines=2), dtype=np.int16)
            more = np.asarray(frame(typed=hm.TYPING[app].cap), dtype=np.int16)
            small, large, full = ((a != rest).any(axis=2) for a in (typed, scrolled, more))
            label = f"{app}{' with decorations' if decorated else ''}"
            assert 0 < small.mean() < 0.01 and full.mean() < 0.02 and small.sum() < full.sum(), f"{label}: typing changes {small.mean():.2%}"
            assert large.mean() >= 0.02 and large.sum() >= 4 * small.sum(), f"{label}: a scroll changes {large.mean():.2%}"
            ys, xs = np.nonzero(full)
            typing = hm.TYPING[app]
            on_line = xs.min() >= typing.rect[0] and xs.max() <= typing.rect[2] and ys.min() >= typing.rect[1] and ys.max() <= typing.rect[3]
            assert on_line, f"{label}: typing stays inside its line"
            layers.text = "gjpqy|Q(" * 80  # (the tallest and the lowest letters)
            tall = (np.asarray(frame(typed=hm.TYPING[app].cap), dtype=np.int16) != rest).any(axis=2)
            ys, xs = np.nonzero(tall)
            assert ys.min() >= typing.rect[1] and ys.max() <= typing.rect[3], f"{label}: tall and low letters stay inside the line ({ys.min()} to {ys.max()})"
            layers.text = hm.typing_text(4)
            ys, xs = np.nonzero(large)
            assert xs.min() >= typing.box[0] and xs.max() < typing.box[2] and ys.min() >= typing.box[1] and ys.max() < typing.box[3], f"{label}: a scroll moves its box only"
            mask = np.asarray(layers.mask) == 255
            assert (typed[mask] == rest[mask]).all() and (scrolled[mask] == rest[mask]).all() and (more[mask] == rest[mask]).all(), f"{label}: the decorations stay"
            assert mask.sum() >= w * hm.TASKBAR_H, "the taskbar is one of them"
            # a scroll only moves the card's own pixels: no colour is new (the rolled box wraps)
            pack = lambda a: (a[..., 0].astype(np.int32) << 16) | (a[..., 1].astype(np.int32) << 8) | a[..., 2]  # noqa: E731
            card = set(np.unique(pack(np.asarray(layers.card, dtype=np.int16))).tolist()) | {0xFFFFFF, 0}  # (and the cursor's)
            assert set(np.unique(pack(scrolled)[~mask]).tolist()) <= card, f"{label}: a scroll brings a new colour"
    # typed text: the same characters every time, and a scroll starts the line from other text
    layers = hm.screen_layers({"app": "editor", "overlay": None, "accent": None, "seed": 4})
    a, b = (layers.typed(hm.ScreenFrame(0, 0, (0, 0), "", "", 5, lines)) for lines in (0, 1))
    assert len(a.replace(" ", "")) == len(b.replace(" ", "")) == 5 and a != b and layers.text.startswith(a)


def test_a_cursor_on_the_error_dialogs_header_leaves_its_shade(tmp_path):
    spec_state = {"app": "editor", "overlay": {**error_overlay(0), "at": [0.5, 0.5]}, "accent": None, "seed": 9}
    base = hm.screen_layers(spec_state)
    w, h = hm.SCREEN
    top = h * (0.5 - hm.DIALOG[1] / 2)
    for place in ((w // 2, round(top + 20)), (w // 2 - 200, round(top + 60)), (w // 2 + 250, round(top + 10))):
        image = hm.screen_frame(base, hm.ScreenFrame(0, 0, place, "10:00", "2026-10-28"))
        share = shade_share(image.resize(reader.fit(w, h, reader.PROFILE["max_px"]), Image.Resampling.BICUBIC), NEEDLE["error"])
        assert NEEDLE_SHARE <= share < MOST_SHARE, f"the cursor at {place}: {share:.1%} of the error shade"
    spec = {"length_s": 20, "states": [{**spec_state, "from": 0, "to": 20, "cursor": [w // 2, round(top + 20)],
                                                         "acts": [[4, "move", w // 2 + 40, round(top + 40)], [5, "type", 3], [6, "scroll", 1]]}]}
    hm.write_screen(tmp_path / "e.mkv", spec, START, tmp_path / "w")
    assert all(NEEDLE_SHARE <= shade_share(image, NEEDLE["error"]) < MOST_SHARE for _, image in decode(tmp_path / "e.mkv")), "and after x264"


def windows_of(path: Path, start: float, length: int) -> list:
    """The reader's windows of a written segment at its real profile (a frame a second, a run of frames that look the same one window)."""
    item = SimpleNamespace(kind="video", t0=start, t1=start + length, source="screen", id=1)
    return list(reader._video(SimpleNamespace(path_of=lambda _: path), item, dict(reader.PROFILE), None, None, False))


def test_the_cursor_typing_and_scrolling_split_the_readers_runs_of_still_frames(tmp_path):
    """A segment of 3 still states: with no acts it is a window a state (the clock's minute at a state's start joins none); with the
    owner's cursor moving, typing and scrolling it is more, and never more than the seconds that have a frame (the reader takes the first
    frame of a second, and a segment has up to 2 a second)."""
    still = states_of((0, 30, []), (30, 60, []), (60, 90, []))
    moving = states_of((0, 30, [[2 * s, "move", 100 + 60 * s, 200 + 20 * s] for s in (2, 3, 4, 10, 11)]),
                       (30, 60, [[2 * s, "type", 3 * s] for s in (1, 2, 3, 8)] + [[40, "scroll", 2]]),
                       (60, 90, sorted([[2 * s, "scroll", s - 3] for s in (4, 5, 6, 7)] + [[9, "type", 2]])))
    for name, states in (("still", still), ("moving", moving)):
        spec = {"length_s": 90, "states": states}
        path = tmp_path / f"{name}.mkv"
        hm.write_screen(path, spec, START, tmp_path / "w")
        windows, frames = windows_of(path, START, 90), hm.screen_frames(spec, START)
        if name == "still":
            assert len(windows) == len(states)
        else:
            seconds = {int(f.second) for f in frames}
            assert len(frames) > len(seconds), "a second can have 2 frames"
            assert len(states) < len(windows) <= len(seconds), (len(windows), len(seconds))
            assert len(windows) >= len(states) + 8, "each move, line typed and scroll is a window of its own"


def test_a_minute_of_the_clock_alone_does_not_split_a_run_but_the_gates_30_s_does(tmp_path):
    """The clock's digits are a few pixels of a cell: its minute joins the run of a still state (15 s in), and `still_s` ends the run at
    30 s, so a still state costs a window at its start and one at the first frame 30 s or more on, the clock's next minute (75 s in), in
    every app."""
    for k, app in enumerate(hm.APPS):
        spec = {"length_s": 90, "states": [{"from": 0, "to": 90, "app": app, "overlay": None, "accent": None, "seed": 3}]}
        path = tmp_path / f"c{k}.mkv"
        hm.write_screen(path, spec, START + 45, tmp_path / "w")  # the clock changes at 15 s and 75 s
        assert [f.second for f in hm.screen_frames(spec, START + 45)] == [0, 15, 75]
        assert [round(w.at) for w in windows_of(path, START + 45, 90)] == [0, 75], app


# -- the porch --------------------------------------------------------------------------------------------------------------

CLIP = {"length_s": 24, "seed": 11, "parcel": [10, 20], "events": [
    {"kind": "walker", "from": 0, "to": 4}, {"kind": "light", "from": 3, "to": 8}, {"kind": "courier", "from": 6, "to": 14, "drop_at": 10},
    {"kind": "dog", "from": 12, "to": 17}, {"kind": "cat", "from": 14, "to": 18, "rgb": list(DISTRACTOR["parcel"]["cat"])},
    {"kind": "owner_pickup", "from": 18, "to": 23}, {"kind": "car", "from": 20, "to": 23, "rgb": [240, 20, 20]}]}


@pytest.fixture(scope="module")
def porch_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("porch") / "porch.mp4"
    hm.write_porch(path, CLIP, "2026-10-28 15:35:44", path.parent / "work")
    return path, decode(path)


def test_the_parcel_is_a_box_on_the_step_from_the_drop_to_the_pickup_and_nothing_else_has_its_shade(porch_file):
    path, frames = porch_file
    assert [round(sec, 3) for sec, _ in frames] == [float(s) for s in range(CLIP["length_s"])], "one frame a second"
    assert media.duration(path) == CLIP["length_s"] and all(image.size == hm.PORCH for _, image in frames)
    for (sec, frame) in frames:
        on = CLIP["parcel"][0] <= sec < CLIP["parcel"][1]
        for stage, shown in (("drawn", hm.porch_frame(CLIP, sec, "t")), ("encoded", frame)):
            share = shade_share(shown, NEEDLE["parcel"])
            if on:
                assert NEEDLE_SHARE <= share < MOST_SHARE, f"second {sec:g} {stage}: {share:.1%} of the parcel shade, not {NEEDLE_SHARE:.0%} to {MOST_SHARE:.0%}"
            else:  # the courier's box, the dog, the cat, the owner, the door and the mat are no parcel
                assert share < DISTRACTOR_SHARE, f"second {sec:g} {stage}: {share:.2%} of the parcel shade, a distractor's must be under {DISTRACTOR_SHARE:.1%}"


def test_the_parcel_is_6_percent_of_the_frame_or_more_with_a_label_and_light_does_not_touch_it():
    with_parcel = np.asarray(hm.porch_frame({**CLIP, "events": []}, 12, "t"), dtype=np.int16)
    without = np.asarray(hm.porch_frame({**CLIP, "events": [], "parcel": None}, 12, "t"), dtype=np.int16)
    changed = (with_parcel != without).any(axis=2)
    assert changed.mean() >= 0.06, f"the parcel is {changed.mean():.1%} of the frame, under 6%"
    box = with_parcel[changed.any(axis=1)][:, changed.any(axis=0)]  # its pixels: the cardboard, the tape and the label
    assert (box > 240).all(axis=2).any(), "a white label"
    dim = {**CLIP, "events": [{"kind": "light", "from": 0, "to": 24}]}
    dimmed, plain = hm.porch_frame(dim, 12, "t"), hm.porch_frame({**CLIP, "events": []}, 12, "t")
    assert shade_share(dimmed, NEEDLE["parcel"]) == shade_share(plain, NEEDLE["parcel"])
    assert not same(dimmed, plain), "the light changes the scene"


def test_a_distractor_courier_keeps_his_box_to_the_door_and_back_out_and_the_step_stays_empty():
    """A courier with `drop_at` None carries the box to the door and out again, in his hands in every frame he is in the picture, and
    nothing is ever on the step (the parcel's shade is not in the frame, nor the carried box's in the step's place): no one could say a
    parcel was handed over. A courier who does drop it has his hands empty after, and the parcel on the step."""
    w, h = hm.PORCH
    carried = np.array(DISTRACTOR["parcel"]["carried"], dtype=np.int16)
    x0, y0, x1, y1 = hm._px(hm.PORCH, *hm.BOX)
    step = (x0 - 2, y0 - hm.TOP_STRIP - 2, x1 + 14, y1 + 2)  # the box on the step, its top face and its edges

    def hands(frame) -> np.ndarray:
        return (np.asarray(frame, dtype=np.int16) == carried).all(axis=2)

    keeps = {"length_s": 13, "seed": 5, "parcel": None, "events": [{"kind": "courier", "from": 1, "to": 9, "drop_at": None}]}
    seen = []
    for second in range(13):
        frame = hm.porch_frame(keeps, second, "t")
        mask = hands(frame)
        assert shade_share(frame, NEEDLE["parcel"]) == 0, f"second {second}: the parcel's shade is in the picture"
        assert not mask[step[1]:step[3], step[0]:step[2]].any(), f"second {second}: the courier's box is on the step"
        assert not (np.asarray(frame.crop(step), dtype=np.int16) == np.array(NEEDLE["parcel"])).all(axis=2).any()
        if 1 < second < 9:  # in the picture: the box in his hands, coming in (up to the door at 5) and going out
            assert mask.sum() > 800, f"second {second}: the box is not in his hands"
            xs = np.nonzero(mask.any(axis=0))[0]
            seen.append((second, int(xs.mean())))
        else:  # not there yet, or gone (at 1 and 9 he is off the right edge)
            assert not mask.any(), f"second {second}: a box and no courier"
    at_door = dict(seen)[5]
    assert 280 <= at_door <= 350, f"at the door at second 5, at x {at_door}"
    coming, going = [x for sec, x in seen if sec < 5], [x for sec, x in seen if sec > 5]
    assert coming == sorted(coming, reverse=True) and going == sorted(going) and max(coming + going) < w, "in from the right, and out to the right"
    assert at_door < min(coming) and at_door < min(going), "the door is the nearest he comes"

    gives = {"length_s": 13, "seed": 5, "parcel": [5, 13], "events": [{"kind": "courier", "from": 1, "to": 9, "drop_at": 5}]}
    for second in range(13):
        frame = hm.porch_frame(gives, second, "t")
        if second > 5:
            assert not hands(frame).any(), f"second {second}: he walks out with the box he left"
        assert (shade_share(frame, NEEDLE["parcel"]) >= NEEDLE_SHARE) is (second >= 5), second


def test_a_parcel_is_where_its_box_says_within_the_places_of_its_area_and_a_visitor_has_more_brown_there():
    """`box` is where the parcel's face is (any of the places of BOX_AREA, the whole box and its top face inside it, on the step); a box
    outside is refused. In BOX_AREA a brown filter finds more in the visitor's coat than in a parcel at any place: the area holds no brown
    door, so a crop of it keeping brown is not a way to the parcel."""
    import shortcuts  # (its hue bands: what an agent's filter for "brown" keeps)

    w, h = hm.PORCH
    ax0, ay0, ax1, ay1 = hm.BOX_AREA
    cut = lambda image: np.asarray(image.crop((int(ax0 * w), int(ay0 * h), int(np.ceil(ax1 * w)), int(np.ceil(ay1 * h)))))[None]  # noqa: E731
    r = random.Random(9)
    places = [hm.parcel_box(r) for _ in range(120)]
    assert hm.parcel_box(random.Random(9)) == places[0] and len({tuple(p) for p in places}) > 110
    corners = [[x0, round(y1 - hm.BOX_H, 3), round(x0 + hm.BOX_W, 3), y1] for x0 in hm.BOX_X0 for y1 in hm.BOX_Y1]
    brown = {}
    for box in (*places[:6], *corners, list(hm.BOX)):
        frame = hm.porch_frame({"length_s": 6, "seed": 2, "events": [], "parcel": [0, 6], "box": box}, 1, "t")
        x0, y0, x1, y1 = bbox_of(frame, NEEDLE["parcel"])
        fx0, fy0, fx1, fy1 = hm._px(hm.PORCH, *box)
        assert abs(x0 - fx0) <= 3 and abs(y1 - fy1) <= 3 and abs(x1 - fx1) <= 3, f"the parcel is not where {box} says"
        assert shade_share(frame, NEEDLE["parcel"]) >= 0.04, f"the parcel at {box} is under 4% of the frame in its own shade"
        assert ax0 * w - 1 <= fx0 and fx1 + 13 <= ax1 * w + 1 and ay0 * h - 1 <= fy0 - hm.TOP_STRIP and fy1 <= ay1 * h + 1, f"{box} is outside BOX_AREA"
        assert fy1 <= 256, "on the step"
        brown[tuple(box)] = float(shortcuts.band_share(cut(frame), "brown")[0])
    for bad in ([0.01, 0.43, 0.26, 0.69], [0.13, 0.30, 0.38, 0.56], [0.13, 0.43, 0.30, 0.69]):
        with pytest.raises(ValueError, match="not one of the places of BOX_AREA"):
            hm.porch_frame({"length_s": 6, "seed": 2, "events": [], "parcel": [0, 6], "box": bad}, 1, "t")
    empty = shortcuts.band_share(cut(hm.porch_frame({"length_s": 6, "seed": 2, "events": [], "parcel": None}, 1, "t")), "brown")[0]
    assert empty < 0.05, f"nothing is brown in the area when there is no parcel ({empty:.1%}): the door is no brown"
    assert 0.3 < min(brown.values()) and max(brown.values()) < 0.6, brown
    for scale in (2.9, 3.05, 3.2):
        for x in (110, 150, 200):
            for bag in (False, True):
                visitor = {"kind": "visitor", "from": 0, "to": 8, "x": x, "scale": scale, "bag": bag}
                frame = hm.porch_frame({"length_s": 8, "seed": 2, "events": [visitor], "parcel": None}, 3, "t")
                share = float(shortcuts.band_share(cut(frame), "brown")[0])
                assert share > max(brown.values()) + 0.02, f"a visitor at {x}, scale {scale}, bag {bag} has {share:.1%} brown in the area, a parcel up to {max(brown.values()):.1%}"
                assert shade_share(frame, NEEDLE["parcel"]) < DISTRACTOR_SHARE, "and none of the parcel's shade"
    with pytest.raises(ValueError, match="a porch event is a courier, walker, visitor"):
        hm.porch_frame({"length_s": 8, "seed": 2, "events": [{"kind": "ghost", "from": 0, "to": 8}], "parcel": None}, 3, "t")


def test_the_camera_stamp_runs_a_second_a_frame_and_is_all_that_changes_in_a_still_scene():
    still = {"length_s": 4, "seed": 5, "events": [], "parcel": None}
    assert hm._stamp_text("2026-10-28 23:59:58", 3) == "2026-10-29 00:00:01" and hm._stamp_text("not a time", 3) == "not a time"
    a, b = hm.porch_frame(still, 0, "2026-10-28 23:59:58"), hm.porch_frame(still, 1, "2026-10-28 23:59:59")
    ys = np.nonzero((np.asarray(a, dtype=np.int16) != np.asarray(b, dtype=np.int16)).any(axis=(1, 2)))[0]
    assert ys.min() >= hm.PORCH[1] - 26, "the stamp strip is the only difference between two seconds of a still scene"


# -- sound ------------------------------------------------------------------------------------------------------------------

KINDS = ("doorbell", "alarm", "microwave", "oven", "watch", "truck", "timer", "ding", "ring", "kettle", "slam", "steps", "vacuum", "bark")
DURATIONS = {"doorbell": hm.CHIME_DUR, "alarm": hm.ALARM_DUR, "microwave": 2.0, "oven": 2.5, "watch": 1.0, "truck": 3.0, "timer": 2.8, "ding": 0.8, "ring": 2.4,
             "kettle": 4.0, "slam": 0.15, "steps": 3.0, "vacuum": 6.0, "bark": 1.2}


def mic_spec(length_s: float = 300, alarm_hz: float = 3200.0, pair=hm.DOORBELL_PAIRS[0], tv: bool = True, loud: bool = False) -> dict:
    """Every kind of sound, each alone: 13 s from the next, and a minute of TV after them. (`length_s` is at least what they take.) With
    `loud`, the beepers and the timer are at `LOUD_GAIN`."""
    sounds, at = [], 6.0
    for k, kind in enumerate(KINDS):
        extra = {"notes": list(pair)} if kind == "doorbell" else {"hz": alarm_hz} if kind == "alarm" else {}
        if loud and kind in (*hm.BEEPER_HZ, "timer"):
            extra["gain"] = hm.LOUD_GAIN
        sounds.append({"kind": kind, "at": at, "dur": DURATIONS[kind], "seed": 100 + k, **extra})
        at += DURATIONS[kind] + 13
    return {"length_s": max(length_s, at + 80), "seed": 3, "bed_dbfs": -50, "tv": [[at + 5, at + 65]] if tv else [], "sounds": sounds}


def spans(spec: dict, kind: str) -> list[tuple[float, float]]:
    return [(s["at"], s["at"] + s["dur"]) for s in spec["sounds"] if s["kind"] == kind]


def overlapping(window: tuple[float, float, np.ndarray], spans_: list[tuple[float, float]]) -> bool:
    return any(window[0] < b and window[1] > a for a, b in spans_)


def check_pitch_shares(spec: dict, samples: np.ndarray, what: str) -> None:
    """The doorbell's first note and the alarm's pitch are heard in a window that overlaps the sound and in none that does not; no
    pitch of a distractor (a beeper, a ding, a phone ring, the chime's second note) is heard in a window of a needle, and neither
    needle's pitch is heard in a window of a distractor."""
    wins = windows(samples)
    first, second = spec["sounds"][0]["notes"]
    needles = {"doorbell": (first, second), "alarm": (spec["sounds"][1]["hz"],)}  # the pitches each is heard at: the chime's two notes
    others = sorted({*(hz for pitches in hm.BEEPER_HZ.values() for hz in pitches), *hm.TIMER_HZ, *hm.DING_HZ, *hm.RING_HZ,
                     *(hz * 480 / 440 for hz in hm.RING_HZ)})
    for kind, pitches in needles.items():
        mine = spans(spec, kind)
        in_a_window = [heard(piece, pitches[0]) for a, b, piece in wins if overlapping((a, b, piece), mine)]
        assert in_a_window and any(in_a_window), f"{what}: the {kind} at {pitches[0]:g} Hz is heard in no window"
        for a, b, piece in wins:
            if not overlapping((a, b, piece), mine):
                for hz in pitches:
                    assert not heard(piece, hz), f"{what}: the window {a:g}-{b:g} s is heard as the {kind}'s {hz:g} Hz and has no {kind}"
            else:
                for other in others:
                    assert not heard(piece, other), f"{what}: the {kind}'s window {a:g}-{b:g} s is heard as {other:g} Hz"


def test_the_alarm_and_the_doorbell_have_pitches_no_distractor_has_before_and_after_opus(tmp_path):
    spec = mic_spec()
    samples = hm.mic_samples(spec)
    assert samples.dtype == np.float32 and len(samples) == round(spec["length_s"] * RATE)
    check_pitch_shares(spec, samples, "the samples")
    path = tmp_path / "mic.ogg"
    hm.write_mic(path, spec, tmp_path / "work")
    check_pitch_shares(spec, decoded(path)[:len(samples)], "opus at 24k")


def test_the_alarm_at_each_end_of_its_range_and_each_doorbell_pair_are_told_from_the_distractors():
    specs = [(mic_spec(alarm_hz=float(hz), tv=False), f"an alarm at {hz} Hz") for hz in hm.ALARM_HZ]
    specs += [(mic_spec(pair=pair, tv=False), f"a doorbell of {pair}") for pair in hm.DOORBELL_PAIRS]
    for spec, what in specs:
        check_pitch_shares(spec, hm.mic_samples(spec), what)


def peak_rms(samples: np.ndarray, seconds: float = 0.5) -> float:
    """A sound's level: the highest RMS over windows of `seconds`, a half-window apart (what the shortcut probe sorts the band by)."""
    x = samples.astype(np.float64)
    size = int(seconds * RATE)
    if len(x) <= size:
        return float(np.sqrt(np.mean(x * x)))
    cumulative = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, len(x) - size + 1, size // 2)
    return float(np.sqrt(((cumulative[starts + size] - cumulative[starts]) / size).max()))


def test_a_gain_scales_a_sound_and_one_that_would_clip_is_refused():
    base = {"kind": "oven", "at": 0, "dur": 2.5, "seed": 1}
    assert np.array_equal(hm.sound(base), hm.sound({**base, "gain": 1.0})), "no gain is a gain of 1"
    for gain in (0.5, 1.5, hm.LOUD_GAIN):
        scaled = hm.sound({**base, "gain": gain})
        assert scaled.dtype == np.float32 and np.allclose(scaled, hm.sound(base) * gain, atol=1e-6), gain
        assert float(np.abs(scaled).max()) <= hm.MAX_PEAK
    assert np.array_equal(hm.sound({**base, "gain": 2.0}), hm.sound({**base, "gain": 2.0})), "the same spec is the same samples"
    with pytest.raises(ValueError, match="oven at gain 3 peaks at 1.05, over 0.98: it would clip"):
        hm.sound({**base, "gain": 3.0})
    for bad in (0, -1):
        with pytest.raises(ValueError, match="gain is above 0"):
            hm.sound({**base, "gain": bad})
    chime = {"kind": "doorbell", "at": 0, "dur": hm.CHIME_DUR, "seed": 1}
    assert np.array_equal(hm.sound(chime), hm.sound({**chime, "gain": 1.0})) and hm.LOUD_GAIN == 2.0


def test_a_loud_beeper_and_a_loud_timer_are_louder_than_the_needles_in_their_bands_at_pitches_of_their_own():
    """A sort by loudness in the needle's band puts a beeper at `LOUD_GAIN` above the alarm (at either end of its range) and a timer
    above every chime; their dominant pitch is in the needle's band, they are not the needle's pitch, and a segment that holds them still
    tells the needle by pitch before and after opus."""
    alarms = [peak_rms(hm.sound({"kind": "alarm", "at": 0, "dur": hm.ALARM_DUR, "hz": float(hz), "seed": 1})) for hz in hm.ALARM_HZ]
    chimes = [peak_rms(hm.sound({"kind": "doorbell", "at": 0, "dur": hm.CHIME_DUR, "notes": list(pair), "seed": 1})) for pair in hm.DOORBELL_PAIRS]
    for kind in ("oven", "truck"):
        for seed in range(4):  # (the pitch is the kind's own, by the seed)
            loud = hm.sound({"kind": kind, "at": 0, "dur": DURATIONS[kind], "seed": seed, "gain": hm.LOUD_GAIN})
            assert peak_rms(loud) > 1.15 * max(alarms), f"a loud {kind} ({peak_rms(loud):.2f}) is louder than the alarm ({max(alarms):.2f})"
            spectrum = np.abs(np.fft.rfft(loud))
            assert 2000 <= spectrum.argmax() * RATE / len(loud) <= 5000, f"a {kind} is in the alarm's band"
    for seed in range(4):
        loud = hm.sound({"kind": "timer", "at": 0, "dur": DURATIONS["timer"], "seed": seed, "gain": hm.LOUD_GAIN})
        assert peak_rms(loud) > 1.15 * max(chimes), f"a loud timer ({peak_rms(loud):.2f}) is louder than a chime ({max(chimes):.2f})"
        spectrum = np.abs(np.fft.rfft(loud))
        assert 400 <= spectrum.argmax() * RATE / len(loud) <= 1000, "a timer is in the chime's band"
    spec = mic_spec(loud=True)
    assert {s["gain"] for s in spec["sounds"] if "gain" in s} == {hm.LOUD_GAIN} and sum("gain" in s for s in spec["sounds"]) == 5
    check_pitch_shares(spec, hm.mic_samples(spec), "loud samples")


def test_the_bed_is_under_the_readers_quiet_floor_so_a_quiet_room_costs_no_window(tmp_path):
    bed = {"length_s": 120, "seed": 8, "bed_dbfs": -50, "tv": [], "sounds": []}
    samples = hm.mic_samples(bed)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    assert 20 * np.log10(rms) == pytest.approx(-50, abs=0.2) and rms < reader.PROFILE["quiet_rms"]
    assert windows(samples) == []
    path = tmp_path / "bed.ogg"
    hm.write_mic(path, bed, tmp_path / "work")
    assert windows(decoded(path)) == [], "after opus too"
    assert -52 < 20 * np.log10(reader.PROFILE["quiet_rms"]) < -47, "the floor is about -48 dBFS"


def test_a_needle_is_loud_and_every_sound_has_its_length():
    bed = 10 ** (hm.BED_DBFS / 20)
    for kind in KINDS:
        samples = hm.sound({"kind": kind, "at": 0, "dur": DURATIONS[kind], "seed": 1})
        assert len(samples) == round(DURATIONS[kind] * RATE) and samples.dtype == np.float32 and float(np.abs(samples).max()) < 1, kind
        if kind in ("doorbell", "alarm"):
            rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
            assert float(np.abs(samples).max()) >= 0.3, f"the {kind}'s amplitude"
            assert 20 * np.log10(rms / bed) >= 20, f"the {kind} is {20 * np.log10(rms / bed):.0f} dB over the bed"


def test_an_alarm_is_three_beeps_of_half_a_second_half_a_second_apart_and_a_doorbell_two_decaying_notes():
    alarm = hm.sound({"kind": "alarm", "at": 0, "dur": hm.ALARM_DUR, "hz": 3100.0, "seed": 1})
    assert len(alarm) == round(hm.ALARM_DUR * RATE)
    slots = [np.abs(alarm[int(k * 0.5 * RATE):int((k + 1) * 0.5 * RATE)]) for k in range(5)]
    assert [float(s.max()) >= 0.3 for s in slots] == [True, False, True, False, True], "on, off, on, off, on"
    assert not slots[1].any() and not slots[3].any(), "the off beats are silent"
    spectrum = np.abs(np.fft.rfft(alarm[:int(0.5 * RATE)])) ** 2
    freqs = np.fft.rfftfreq(int(0.5 * RATE), 1 / RATE)
    assert abs(freqs[spectrum.argmax()] - 3100) < 5, "the fundamental"
    assert spectrum[np.abs(freqs - 6200) < 30].sum() > 0.01 * spectrum.sum(), "a second harmonic"
    bell = hm.sound({"kind": "doorbell", "at": 0, "dur": hm.CHIME_DUR, "notes": [783.99, 659.26], "seed": 1})
    half = len(bell) // 2
    for piece, hz in ((bell[:half], 783.99), (bell[half:], 659.26)):
        freqs = np.fft.rfftfreq(len(piece), 1 / RATE)
        assert abs(freqs[np.abs(np.fft.rfft(piece)).argmax()] - hz) < 3, "each note at its pitch"
        assert np.abs(piece[:int(0.1 * RATE)]).max() > 3 * np.abs(piece[-int(0.1 * RATE):]).max(), "a note decays"
    assert np.abs(bell[:half]).max() > np.abs(bell[half:]).max(), "the first note is the louder"


def test_a_pitch_too_near_a_needle_is_refused():
    chime = {"kind": "doorbell", "at": 1.0, "dur": hm.CHIME_DUR, "notes": [659.26, 523.25], "seed": 1}
    base = {"length_s": 30, "seed": 1, "bed_dbfs": -50, "tv": []}
    hm.mic_samples({**base, "sounds": [chime, {"kind": "ding", "at": 10.0, "dur": 0.8, "hz": 987.77, "seed": 2}]})
    for bad in ({"kind": "ding", "hz": 680.0}, {"kind": "ring", "hz": 500.0}, {"kind": "microwave", "hz": 535.0}):
        with pytest.raises(ValueError, match="within 8%"):
            hm.mic_samples({**base, "sounds": [chime, {"at": 10.0, "dur": 2.0, "seed": 2, **bad}]})
    alarm = {"kind": "alarm", "at": 1.0, "dur": hm.ALARM_DUR, "hz": 3200.0, "seed": 1}
    with pytest.raises(ValueError, match="the alarm's pitch"):
        hm.mic_samples({**base, "sounds": [alarm, {"kind": "watch", "at": 10.0, "dur": 1.0, "hz": 3300.0, "seed": 2}]})
    hm.mic_samples({**base, "sounds": [alarm, {"kind": "watch", "at": 10.0, "dur": 1.0, "seed": 2}]})  # a pitch of its own: no clash
    with pytest.raises(ValueError, match="a sound is"):
        hm.sound({"kind": "banjo", "dur": 1.0})


def test_a_tv_stretch_is_loud_enough_to_read_and_is_nobodys_pitch():
    spec = {"length_s": 120, "seed": 6, "bed_dbfs": -50, "tv": [[10, 100]], "sounds": []}
    samples = hm.mic_samples(spec)
    wins = windows(samples)
    assert len(wins) >= 30 and all(10 - 5 <= a and b <= 100 + 5 for a, b, _ in wins), "only the TV is above the floor"
    for a, b, piece in wins:
        for hz in (*(n for pair in hm.DOORBELL_PAIRS for n in pair), hm.ALARM_HZ[0], 3200, hm.ALARM_HZ[1]):
            assert not heard(piece, hz), f"the TV at {a:g}-{b:g} s is heard as {hz:g} Hz"
    rms = float(np.sqrt(np.mean(np.square(samples[10 * RATE:100 * RATE], dtype=np.float64))))
    assert 0.01 < rms < 0.1, f"the TV's RMS {rms:.3f}"


# -- throughput -------------------------------------------------------------------------------------------------------------

def timed(label: str, units: int, unit: str, run, tries: int = 4) -> float:
    """Seconds a unit of `run`: the first of up to `tries` runs that is within the target, else the best; printed."""
    best = float("inf")
    for _ in range(tries):
        best = min(best, _clock(run) / units)
        if best <= TARGET_S[label] * SLACK:
            break
    shown = f"{best * 1000:.1f} ms" if TARGET_S[label] < 0.1 else f"{best:.2f} s"
    target = f"{TARGET_S[label] * 1000:.0f} ms" if TARGET_S[label] < 0.1 else f"{TARGET_S[label]:g} s"
    print(f"\n{label}: {shown} a {unit} (the target: {target}{f', held to {SLACK:g} times it under xdist' if SLACK > 1 else ''})")
    return best


def _clock(run) -> float:
    start = time.perf_counter()
    run()
    return time.perf_counter() - start


def busy_acts(k: int) -> list[list]:
    """A busy state of 30 s: an act on 40 of its 60 ticks, moves, typing and scrolling."""
    acts, typed, lines = [], 0, 0
    for tick in range(1, 60, 3):
        for t, kind in ((tick, "move"), (tick + 1, "type" if k % 3 else "scroll")):
            if kind == "move":
                acts.append([t, "move", 100 + 10 * k + 7 * t, 100 + 5 * t])
            elif kind == "type":
                typed = typed % 40 + 2
                acts.append([t, "type", typed])
            else:
                lines += 1
                acts.append([t, "scroll", lines])
    return acts


def test_the_writers_keep_to_their_throughput_target(tmp_path):
    apps = list(hm.APPS)
    states = [{"from": 30 * k, "to": 30 * (k + 1), "app": apps[k % 6], "overlay": error_overlay(k) if k % 5 == 4 else None,
               "accent": {"kind": "toast"} if k % 7 == 3 else None, "seed": k,
               "cursor": [100 + 10 * k, 100], "acts": busy_acts(k)} for k in range(20)]
    screen = {"length_s": 600, "states": states}
    frames = len(hm.screen_frames(screen, START))
    per_frame_screen = timed("screen frame", frames, "frame", lambda: hm.write_screen(tmp_path / "s.mkv", screen, START, tmp_path / "w"))
    clip = {**CLIP, "length_s": 20}
    per_frame = timed("porch frame", 20, "frame", lambda: hm.write_porch(tmp_path / "p.mp4", clip, "2026-10-28 15:35:44", tmp_path / "w"))
    mic = {"length_s": 900, "seed": 9, "bed_dbfs": -50, "tv": [[100, 200]], "sounds": [
        {"kind": "doorbell", "at": 300.0, "dur": hm.CHIME_DUR, "notes": list(hm.DOORBELL_PAIRS[1]), "seed": 1},
        {"kind": "alarm", "at": 600.0, "dur": hm.ALARM_DUR, "hz": 3300.0, "seed": 2}, {"kind": "kettle", "at": 450.0, "dur": 4.0, "seed": 3}]}
    per_15_minutes = timed("mic 15 minutes", 1, "segment", lambda: hm.write_mic(tmp_path / "m.ogg", mic, tmp_path / "w"))
    assert per_frame_screen <= TARGET_S["screen frame"] * SLACK, f"a screen frame takes {per_frame_screen * 1000:.1f} ms, the target is 20 (x{SLACK:g})"
    assert per_frame <= TARGET_S["porch frame"] * SLACK, f"a porch frame takes {per_frame * 1000:.1f} ms, the target is 15 (x{SLACK:g})"
    assert per_15_minutes <= TARGET_S["mic 15 minutes"] * SLACK, f"15 minutes of sound take {per_15_minutes:.2f} s, the target is 1 (x{SLACK:g})"
