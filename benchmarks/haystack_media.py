"""The drawings and sounds of the harder benchmark haystack (TESTBED.md, benchmark; `benchmarks/haystack.py` plans, this draws):

    photo(spec) -> Image              a phone photo, 1280x720: sign, car, object, room, food, plant or street
    screen_state(state) -> Image      one state of the screen at its start, 1280x720: an app card, an accent, an overlay (a dialog), the taskbar
    screen_layers(state)              the same in layers: the card, and the decorations over it with their exact mask
    screen_frames(spec, start)        the frames of a screen segment: when each is (half seconds), and the state, cursor, typing, scroll and clock it shows
    screen_frame(layers, frame) -> Image   a frame: the card scrolled and typed on, the decorations over it, the taskbar clock and the cursor
    porch_frame(spec, second, stamp)  one second of the front door camera, 640x360: a scene and its events
    sound(spec), mic_samples(spec)    one sound, and a microphone segment (bed, TV, sounds), float32 mono at RATE
    write_screen / write_porch / write_mic (path, spec, ..., work)   the files: VFR x264, x264 at 1 frame a second, opus

Everything is a function of its spec (`seed` and the rest): the same spec is the same pixels and samples, and nothing reads a
clock or a global random stream. The specs are plain JSON-able dicts, the shapes `benchmarks/haystack.py` makes (times are seconds
from the file's start):
- photo: `{"scene", "rgb": [..]|None, "word", "number", "box": [x0, y0, x1, y1] (fractions)|None, "seed"}`. `rgb` is the sign's
  fill, the car's body, or the colour of the scene's subject (an umbrella, a door, tomatoes, leaves, a shirt);
- screen segment: `{"length_s", "states": [{"from", "to", "app", "overlay": None|{"kind", "rgb", "code", "msg", "at"?},
  "accent": None|{"kind", "rgb"}, "seed", "cursor"?: [x, y], "acts"?: [[tick, "move", x, y] | [tick, "type", n] | [tick, "scroll", lines], ..]}]}`:
  the cursor's place at the state's start (CURSOR_HOME without one) and what the owner does after it, at `tick` half seconds from the
  state's start (from 1, increasing): a `move` of the cursor, a `type` (`n` characters are on the app's typing line, TYPING) or a `scroll`
  (`lines` lines scrolled since the state's start; a scroll empties the typing line). The plan also names the state's `level` of activity
  (idle, light or busy: the planner's, not drawn). Its frames are `screen_frames`';
- porch clip: `{"length_s", "events": [{"kind", "from", "to", "rgb"?, "drop_at"?, "x"?, "scale"?, "bag"?}], "parcel": [from, to]|None, "box"?, "seed"}`;
- mic segment: `{"length_s", "seed", "bed_dbfs", "tv": [[from, to], ..], "sounds": [{"kind", "at", "dur", "hz"?, "notes"?, "gain"?, "seed"}]}`.

A needle is one exact shade (`NEEDLE_RGB`): the fake decider's `shade:R,G,B` says yes when 2% of a picture's pixels are within
`SHADE_TOL` of it. Every distractor of the needle's source has the needle's named colour and a shade at least `SHADE_GAP` from
it (`DISTRACTOR_RGB`), so a colour filter gives a shortlist and not the answer; the module refuses a colour that is closer.
The reader hands a decider pictures of at most 640x360 (its `max_px`), so the needles are sized for that: a dialog's title is 20 px
tall there, readable to the decider but about 8 px in a contact sheet 5 frames across, so that an agent must open or crop a frame to tell
an error from the red "Confirm" and the red-headed warnings, as with a real dialog. The shades keep 3% or more of the picture through
that scaling and the encoder (`tests/test_haystack_media.py`).
Sound is the same with pitch: the alarm's pitch is in `ALARM_HZ`, the doorbell's notes are a pair of `DOORBELL_PAIRS`, and every
other sound's pitch is at least `PITCH_GAP` from both, which `mic_samples` checks for the sounds of its own segment. A bed of
-50 dBFS is under the reader's `quiet_rms` (-48 dBFS), so a window of a quiet room costs the reader its decoding and nothing more.

A needle does not always stand in one place, so that a crop of one place finds it: a dialog opens at `overlay["at"]`, its centre (`dialog_at`
draws one; DIALOG_AT says where, in the middle of the screen), and a parcel is left at `box` (`parcel_box` draws one, on the step by the door).
`DIALOG_AREA` and `BOX_AREA` are the unions of every place they can take, as fractions of the frame. The distractors of the parcel's colour are
in those places too: a `visitor` (close to the camera, in a brown coat, with a carrier bag or without) has more brown in `BOX_AREA` than a
parcel has, and the door of the porch is not brown (a brown door would put brown in the parcel's place in every frame); the dialogs of the
error's red are a confirm (its header and its button), and a warning given a red header.

For the planner: a needle photo is a `sign` with `rgb` NEEDLE_RGB["sign"] (its `box` 22 to 34% of the photo, or none: `sign_box` makes
one) or a `car` with `rgb` NEEDLE_RGB["car"]; every other photo's colour is one of DISTRACTOR_RGB's. An error state has an `overlay` of kind
`error`, `rgb` NEEDLE_RGB["error"] and a `code` such as `E-4417`; its distractors are the other overlay kinds and the accents, whose `rgb`
may be left out. A parcel is the clip's `parcel: [drop_at, pickup]`, with the courier's `drop_at` the same second; a courier whose `drop_at`
is None (a distractor) carries his box to the door and back out and leaves nothing on the step. A doorbell has
`notes` (a pair of DOORBELL_PAIRS, the first is the pitch to ask for) and `dur` CHIME_DUR; an alarm has `hz` (of ALARM_HZ) and `dur`
ALARM_DUR; a distractor sound names no `hz` (`pick_hz` takes its kind's own, by its `seed`) and may name a `gain` (1 by default; a
beeper or a timer at 2 is louder than the alarm and the chime, which a sort by loudness in the needle's band must not take for the needle).

The screen follows the real recorder (DESIGN.md §6: ddagrab at about 2 fps with the cursor drawn, then `mpdecimate=max=119` and
`-fps_mode vfr`): a frame whenever the picture changes, at most one every half second (a tick, TICK_S), and at least one every 60 s. mms's
reader takes the first frame of each second, so a segment's windows are at most its seconds with a frame, and the files arm sees every
frame. A frame is a state's start, an act (a cursor move, typing or a scroll) or a minute change of the taskbar's clock; `screen_frames`
is the one place that says where they are, and the writer, the checks, the window counts and the probe all use it. Each frame differs from
the one before it, as the recorder's `mpdecimate` makes them: a typed character is a visible (not space) one (`typed_text`), and the plan
types no more than the typing line shows where a decoration covers its end (`typing_cap`, from `_decorations`).
A state is drawn in layers (`screen_layers`): the app card, and the decorations that stand over it and never move, the accent, the dialog
and the taskbar (TASKBAR_H px along the bottom, below every dialog and needle: dark, with the time and the date at the right as Windows shows
them), with an exact mask found by drawing them on a black and on a white background, so a dialog's white body over a white page is
opaque. A frame is the card with its content box rolled up by the lines scrolled (wrapping: the lines that leave at the top come back at the
bottom, so a scroll brings no new colour), the typed text on the app's typing line (TYPING: a box, a line height, a line that holds the
text's whole height, a size, an ink from the app's own palette, and a bold face a little larger than the card's text), the decorations
pasted over both, then the clock and the cursor (`stamp`). Layers made with no scroll or typing are `screen_state(state)` exactly. A frame
costs a copy, a roll, a short text and a masked paste.
The cursor is a white arrow with a black outline, 12x19 px at 1280x720 (about 6x9.5 at the reader's 640x360; on the dev laptop, 3840x2400 at
200% with pointer size 2, the owner's own pointer is 0.67 to 1 times that at 640 px). The reader's change gate (a cell of 16 px at 640x360
moving by more than 6) is sensitive to its size: at this size 87 to 99% of moves are a change, at 0.8 times it 19 to 89%, at 0.67 times
0 to 78%. `benchmarks/haystack.py` plans what the owner does (`state_acts`: idle, light and busy states drawn as a chain, in runs of active
seconds with 1 or 2 acts each, in the mix of kinds of the app), and its constants are calibrated against two real hours of a screen
recording (TESTBED.md): the typed text is larger and bolder than the card's so that a frame of typing changes about as much of the frame
at 640x360 as a real one's frame (0.05 to 0.3%, the card's own size gives 0.025%), and any typed character is a change to the gate
(98% of the seconds with a frame are windows, 71 to 86% in the real hours).
Key frames are at a file's start and at scene cuts (x264 puts one at each state change, 1 to 4 a file), and none is forced every 10 s, so a
segment is cut at a state's start. Not modelled: the recorder forces a key frame every 10 s (DESIGN.md §6), so its cuts land mid-state; a scroll wraps instead of bringing new lines; there are no caret blinks (A-B-A returns are
0%, 2 to 5% in the real hours).

The files are written as the capture writes them (TESTBED.md): the screen at crf 30 and the camera at crf 28 with x264's
`zerolatency` tune, the microphone as opus at 24 kbit/s; in this process through PyAV (video) and a pipe to ffmpeg (sound, never a
WAV: 42 h of it is 4.8 GB), one thread each, so that three files written at once use three cores and the bytes do not depend on
the machine's core count."""

from __future__ import annotations

import bisect
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np
from PIL import Image, ImageChops, ImageDraw

from mms.config import ffmpeg_exe

# day and fake_decider are the tests' own, and a script run from benchmarks/ has only benchmarks/ on its path
TESTS_DIR = str(Path(__file__).resolve().parents[1] / "tests")
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)  # last, so no file of the tests' shadows a module of the standard library or the env

from day import RATE, font  # noqa: E402
from fake_decider import SHADE_TOL  # noqa: E402

PHOTO, SCREEN, PORCH = (1280, 720), (1280, 720), (640, 360)
SCREEN_CRF, PORCH_CRF = 30, 28  # capture.py's screen preset and camera preset
SHADE_GAP = 2 * SHADE_TOL  # a distractor's colour is at least this far from its source's needle shades (SHADE_TOL: the fake decider's)
PITCH_GAP = 0.08  # a sound's pitch is at least this share of the lower pitch away from a needle's
BED_DBFS = -50.0
ALARM_HZ = (3000, 3400)  # a house's smoke alarm beeps at a pitch of this range
DOORBELL_PAIRS = ((659.26, 523.25), (783.99, 659.26), (587.33, 392.00), (698.46, 523.25))  # a house's chime: first and second note
BEEPER_HZ = {"microwave": (2450.0, 3800.0), "oven": (2600.0, 4000.0), "watch": (2700.0, 4250.0), "truck": (2520.0, 3900.0)}
TIMER_HZ = (880.0, 940.0)  # a kitchen timer's, in the chime's band (400 to 1,000 Hz) but 8% or more from every note of a chime
DING_HZ = (987.77, 1174.66, 1567.98, 2093.0)  # a notification's, and a TV jingle's, pitches
RING_HZ = (440.0, 1250.0, 1900.0)  # a phone ring is two tones: this pitch and 480/440 of it (440 and 480 Hz)
ALARM_DUR, CHIME_DUR = 2.5, 1.8  # seconds: three beeps (0.5 s on, 0.5 s off), and a two-note chime
LOUD_GAIN = 2.0  # a beeper (oven, truck) or a timer at this gain has a higher level than the alarm (0.34) or any chime (0.25), whatever its pitch
MAX_PEAK = 0.98  # a sound with a `gain` may not peak above this: a clipped sound is not the pitch it was made at
ALARM_ON_S, ALARM_OFF_S = 0.5, 0.5
NEEDLE_RGB = {"error": (176, 28, 36), "sign": (20, 170, 80), "car": (240, 20, 20), "parcel": (190, 142, 88)}
DISTRACTOR_RGB = {  # by the needle they share a named colour or a look with: what a colour filter shortlists
    "error": {"warning": (240, 130, 20), "info": (40, 100, 200), "save": (110, 115, 125), "confirm": (232, 80, 72),
              "banner": (225, 85, 85), "cells": (240, 120, 120), "badge": (255, 59, 48), "toast": (225, 75, 60)},
    "sign": {"yellow": (250, 215, 0), "white": (245, 245, 245), "blue": (20, 60, 200), "grey": (130, 130, 135),
             "plant": (80, 140, 60), "door": (50, 110, 60), "shirt": (110, 190, 120), "car": (90, 150, 40)},
    "car": {"blue": (30, 60, 200), "white": (245, 245, 245), "grey": (130, 130, 135), "yellow": (240, 210, 20),
            "door": (150, 30, 30), "umbrella": (200, 60, 70), "tomato": (205, 70, 50)},
    "parcel": {"carried": (120, 88, 52), "dog": (110, 75, 40), "door": (55, 75, 115), "mat": (100, 90, 70),
               "planter": (160, 80, 55), "cat": (225, 125, 35), "courier": (110, 75, 40)},
}
SOURCE_NEEDLES = {"phone": ("sign", "car"), "screen": ("error",), "porch": ("parcel",)}  # the needles each source's colours are cleared of
DIALOG = (0.45, 0.35)  # a dialog's share of the screen's width and height: 15.75% of its area
DIALOG_AT = ((0.38, 0.62), (0.40, 0.60))  # where a dialog's centre may be, as fractions of the screen: the range of x and of y (in the middle part)
DIALOG_AREA = (DIALOG_AT[0][0] - DIALOG[0] / 2, DIALOG_AT[1][0] - DIALOG[1] / 2, DIALOG_AT[0][1] + DIALOG[0] / 2, DIALOG_AT[1][1] + DIALOG[1] / 2)
# ^ every place a dialog can be, (x0, y0, x1, y1) as fractions of the screen: the union of its positions (its shadow, grey, is left out)
TASKBAR_H = 40  # px: the taskbar along the bottom of the screen, under the dialogs (DIALOG_AREA ends at 78% of the height) and every needle
TASKBAR_RGB, TASKBAR_LINE, TASKBAR_ICON, TASKBAR_LOGO, TASKBAR_INK = (32, 32, 40), (64, 64, 72), (96, 96, 108), (0, 120, 215), (235, 235, 240)
CURSOR_ARROW = ((0, 0), (0, 17), (4, 13), (7, 18), (9, 17), (6, 12), (11, 12))  # the pointer's outline from its tip: 12 px wide, 19 tall
CURSOR_HOME = (640, 360)  # where the cursor is when a state names no place
FLOOR_S = 60  # the recorder keeps a frame at least this often (mpdecimate max=119 at 2 fps)
TITLE_PX = 40  # a dialog's title, in 1280x720 pixels: 20 after the scaling to 640x360, 8 in a contact sheet 5 frames across


def distance(a, b) -> float:
    return math.dist(a, b)


def gap(a: float, b: float) -> float:
    """How far apart two pitches are, as a share of the lower."""
    return abs(a - b) / min(a, b)


def _check_constants() -> None:
    """The module's own constants meet its rules (a wrong one fails when the file is imported, never in a run)."""
    for source, kinds in SOURCE_NEEDLES.items():
        for kind in kinds:
            for group in kinds:
                for name, rgb in DISTRACTOR_RGB[group].items():
                    assert distance(rgb, NEEDLE_RGB[kind]) >= SHADE_GAP, f"the {source} distractor {name} {rgb} is near the {kind} shade"
    for (a, ka), (b, kb) in ((x, y) for x in NEEDLE_RGB.items() for y in NEEDLE_RGB.items() if x[0] < y[0]):
        assert distance(ka, kb) >= SHADE_GAP, f"the {a} and {b} shades are within {SHADE_GAP}"
    for rgb in SAFE:  # photos and the browser's picture
        assert all(distance(rgb, NEEDLE_RGB[k]) >= SHADE_GAP for k in ("sign", "car", "error")), f"SAFE {rgb} is near a needle's shade"
    for rgb in (TASKBAR_RGB, TASKBAR_LINE, TASKBAR_ICON, TASKBAR_LOGO, TASKBAR_INK, (255, 255, 255), (0, 0, 0)):  # the taskbar, its clock, the cursor
        assert distance(rgb, NEEDLE_RGB["error"]) >= SHADE_GAP, f"the taskbar or cursor colour {rgb} is near the error shade"
    for app, typing in TYPING.items():  # the text typed (a slide's ink is `_ink`'s)
        for rgb in typing.ink or ((255, 255, 255), (20, 20, 24)):
            _clear_of_needles(rgb, f"the {app}'s typing ink", "screen")
    for rgb in (*WALLS, *JACKETS):  # the porch
        assert distance(rgb, NEEDLE_RGB["parcel"]) >= SHADE_GAP, f"the porch colour {rgb} is near the parcel's shade"
    notes = [n for pair in DOORBELL_PAIRS for n in pair]
    lone = {**{f"{k} {i}": hz for k, pitches in BEEPER_HZ.items() for i, hz in enumerate(pitches)},
            **{f"timer {hz}": hz for hz in TIMER_HZ}, **{f"ding {hz}": hz for hz in DING_HZ}, **{f"ring {hz}": hz for hz in RING_HZ},
            **{f"ring {hz} (+9%)": hz * 480 / 440 for hz in RING_HZ}}
    for name, hz in lone.items():
        assert all(gap(hz, note) >= PITCH_GAP for note in notes), f"{name} is within {PITCH_GAP:.0%} of a chime note"
        near_alarm = ALARM_HZ[0] / (1 + PITCH_GAP) < hz < ALARM_HZ[1] * (1 + PITCH_GAP)
        assert not near_alarm, f"{name} is within {PITCH_GAP:.0%} of the alarm"
    for note in notes:
        assert gap(note, ALARM_HZ[0]) >= PITCH_GAP, f"a chime note {note} is near the alarm"


def _rng(seed, salt: str = "") -> random.Random:
    return random.Random(f"{seed}-{salt}")


def _rgb(value) -> tuple[int, int, int]:
    return tuple(int(c) for c in value)  # type: ignore[return-value]


def _clear_of_needles(rgb, what: str, source: str, allow: str | None = None) -> None:
    """`rgb` is no shade of a needle of `source` (nor within SHADE_GAP of one), except the one of `allow`, which it may be exactly."""
    for kind in SOURCE_NEEDLES[source]:
        shade = NEEDLE_RGB[kind]
        if distance(rgb, shade) < SHADE_GAP and not (kind == allow and tuple(rgb) == shade):
            raise ValueError(f"{what} {tuple(rgb)} is within {SHADE_GAP} of the {kind} needle's shade {shade}")


def _ink(fill) -> tuple[int, int, int]:
    """Letters for a sign or button of `fill`: white on a dark one, near black on a light one."""
    return (20, 20, 24) if 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2] > 150 else (255, 255, 255)


def _gradient(size: tuple[int, int], top, bottom) -> Image.Image:
    w, h = size
    rows = np.linspace(top, bottom, h, dtype=np.float32)[:, None, :]
    return Image.fromarray(np.broadcast_to(rows, (h, w, 3)).astype(np.uint8), "RGB")


def _fit(draw: ImageDraw.ImageDraw, xy, text: str, fill, size: int, width: float, anchor: str = "mm") -> None:
    """`text` at `xy`, in `size` pixels or the largest below it that is `width` wide at most."""
    while size > 12 and draw.textlength(text, font=font(size)) > width:
        size -= 2
    draw.text(xy, text, fill=fill, font=font(size), anchor=anchor)


def dialog_at(r: random.Random) -> list[float]:
    """Where a dialog opens: the centre of one of the places of DIALOG_AREA, as fractions of the screen, from `r`."""
    return [round(r.uniform(*DIALOG_AT[0]), 3), round(r.uniform(*DIALOG_AT[1]), 3)]


def _px(size: tuple[int, int], *fractions: float) -> tuple[int, ...]:
    """Pixel corners from fractions of the picture's width and height: (x0, y0, x1, y1, ...)."""
    return tuple(int(round(f * size[i % 2])) for i, f in enumerate(fractions))


# -- photos -----------------------------------------------------------------------------------------------------------

SIGN_BOX = (0.22, 0.34)  # a needle sign's share of the photo
CAR_BOX = (0.04, 0.18, 0.96, 0.86)  # a car's box: its body is about 34% of the photo
OBJECTS = ("umbrella", "bag", "shirt")
SAFE = ((70, 90, 140), (150, 120, 90), (110, 110, 115), (235, 228, 210), (222, 228, 238), (238, 232, 240), (240, 215, 150), (90, 70, 60),
        (120, 90, 70), (180, 150, 110), (60, 60, 70), (200, 180, 120), (130, 100, 150), (70, 130, 150))  # no needle's shade, nor near one


def photo(spec: dict) -> Image.Image:
    """A phone photo of `spec["scene"]`: a `sign` (fill `rgb`, `word`, `number`, in `box`; 22 to 34% of the photo for a needle), a
    `car` (body `rgb`, in `box`; the body of a red one is 30% or more of the photo), or an `object`, `room`, `food`, `plant` or `street`
    with `rgb` the colour of its subject. A sign's fill and a car's body may be the needle's own shade; no other colour may be
    within SHADE_GAP of a needle's."""
    scene = spec["scene"]
    seed = spec.get("seed", 0)
    rgb = spec.get("rgb")
    if scene == "sign":
        return _sign_photo(spec, _rgb(rgb or NEEDLE_RGB["sign"]))
    if scene == "car":
        return _car_photo(spec, _rgb(rgb or NEEDLE_RGB["car"]))
    if rgb is not None:
        _clear_of_needles(rgb, f"the {scene} photo's colour", "phone")
    return {"object": _object_photo, "room": _room_photo, "food": _food_photo, "plant": _plant_photo,
            "street": _street_photo}[scene](seed, None if rgb is None else _rgb(rgb))


def _backdrop(r: random.Random) -> Image.Image:
    """A wall, a fence or a sky: muted, and no needle's shade."""
    kind = r.choice(("wall", "sky", "fence"))
    if kind == "sky":
        return _gradient(PHOTO, (150, 185, 225), (215, 228, 240))
    image = _gradient(PHOTO, *r.choice((((196, 188, 176), (170, 162, 150)), ((200, 205, 210), (176, 182, 190)), ((215, 205, 185), (190, 178, 158)))))
    if kind == "fence":
        draw = ImageDraw.Draw(image)
        for x in range(0, PHOTO[0], 64):
            draw.rectangle([x, 0, x + 6, PHOTO[1]], fill=(150, 142, 130))
    return image


def sign_box(spec: dict) -> tuple[float, ...]:
    """A sign's box, `(x0, y0, x1, y1)` as fractions of the photo: the spec's own, or one its seed makes, 22 to 34% of the photo."""
    if spec.get("box"):
        return tuple(spec["box"])
    r = _rng(spec.get("seed", 0), "sign-box")
    height = r.uniform(0.62, 0.85)
    width = r.uniform(*SIGN_BOX) / height
    x0, y0 = r.uniform(0.04, 0.96 - width), r.uniform(0.03, 0.95 - height)
    return x0, y0, x0 + width, y0 + height


def _sign_photo(spec: dict, fill: tuple[int, int, int]) -> Image.Image:
    _clear_of_needles(fill, "the sign's fill", "phone", allow="sign")
    image = _backdrop(_rng(spec.get("seed", 0), "sign"))
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = _px(PHOTO, *sign_box(spec))
    ink = _ink(fill)
    if y1 < PHOTO[1] - 4:
        draw.rectangle([(x0 + x1) // 2 - 12, y1, (x0 + x1) // 2 + 12, PHOTO[1]], fill=(95, 95, 100))
    draw.rounded_rectangle([x0 + 10, y0 + 10, x1 + 10, y1 + 10], radius=28, fill=(60, 60, 64))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=28, fill=fill, outline=ink, width=10)
    cx, height = (x0 + x1) // 2, y1 - y0
    _fit(draw, (cx, y0 + height * 0.24), str(spec.get("word", "GATE")), ink, int(height * 0.18), (x1 - x0) * 0.8)
    _fit(draw, (cx, y0 + height * 0.64), str(spec.get("number", "47")), ink, int(height * 0.46), (x1 - x0) * 0.8)
    return image


def _car_photo(spec: dict, body: tuple[int, int, int]) -> Image.Image:
    _clear_of_needles(body, "the car's body", "phone", allow="car")
    r = _rng(spec.get("seed", 0), "car")
    image = _backdrop(r)
    draw = ImageDraw.Draw(image)
    draw.rectangle(_px(PHOTO, 0, 0.86, 1, 1), fill=(88, 88, 94))
    x0, y0, x1, y1 = _px(PHOTO, *(spec.get("box") or CAR_BOX))
    w, h = x1 - x0, y1 - y0
    draw.ellipse([x0 - 10, y1 - h // 14, x1 + 10, y1 + h // 10], fill=(70, 70, 76))
    draw.rounded_rectangle([x0, y0 + h * 38 // 100, x1, y0 + h * 88 // 100], radius=h // 10, fill=body)
    cabin = [(12, 40), (26, 0), (70, 0), (86, 40)]  # (percent of the width, percent of the height) from the car's top left
    draw.polygon([(x0 + w * px // 100, y0 + h * py // 100) for px, py in cabin], fill=body)
    window = (35, 50, 70)
    for pane in ([(17, 38), (28, 8), (48, 8), (48, 38)], [(52, 38), (52, 8), (67, 8), (80, 38)]):  # the two windows
        draw.polygon([(x0 + w * px // 100, y0 + h * py // 100) for px, py in pane], fill=window)
    for x in (0.2, 0.8):
        cx, cy, radius = x0 + int(x * w), y0 + h * 88 // 100, h * 14 // 100
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(20, 20, 20))
        draw.ellipse([cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2], fill=(170, 170, 175))
    draw.rectangle([x1 - w * 6 // 100, y0 + h * 56 // 100, x1 - w * 2 // 100, y0 + h * 63 // 100], fill=(250, 240, 190))
    draw.rectangle([x0 + w * 45 // 100, y0 + h * 54 // 100, x0 + w * 55 // 100, y0 + h * 57 // 100], fill=(60, 60, 60))
    if r.random() < 0.5:  # a plate
        draw.rectangle([x0 + w * 40 // 100, y0 + h * 72 // 100, x0 + w * 60 // 100, y0 + h * 80 // 100], fill=(240, 240, 235))
    return image


def _object_photo(seed: int, rgb) -> Image.Image:
    r = _rng(seed, "object")
    image = _backdrop(r)
    draw = ImageDraw.Draw(image)
    colour = rgb or r.choice(SAFE)
    what = r.choice(OBJECTS)
    w, h = PHOTO
    draw.rectangle(_px(PHOTO, 0, 0.84, 1, 1), fill=r.choice(((150, 110, 75), (120, 90, 70), (180, 150, 110))))
    cx = r.randint(w * 35 // 100, w * 65 // 100)
    if what == "umbrella":
        draw.pieslice([cx - 300, h * 8 // 100, cx + 300, h * 8 // 100 + 600], 180, 360, fill=colour)
        draw.line([cx, h * 8 // 100 + 300, cx, h * 80 // 100], fill=(60, 60, 64), width=14)
        draw.arc([cx - 50, h * 70 // 100, cx + 10, h * 90 // 100], 0, 180, fill=(60, 60, 64), width=14)
        for k in (-200, -100, 100, 200):
            draw.line([cx, h * 8 // 100, cx + k, h * 8 // 100 + 300 - abs(k) // 6], fill=(40, 40, 44), width=4)
    elif what == "bag":
        draw.arc([cx - 130, h * 12 // 100, cx + 130, h * 52 // 100], 180, 360, fill=(60, 60, 64), width=16)
        draw.rounded_rectangle([cx - 240, h * 32 // 100, cx + 240, h * 82 // 100], radius=36, fill=colour)
        draw.rectangle([cx - 240, h * 44 // 100, cx + 240, h * 47 // 100], fill=(240, 240, 240))
    else:
        outline = [(-120, 12), (-330, 30), (-260, 48), (-180, 40), (-180, 82), (180, 82), (180, 40), (260, 48), (330, 30), (120, 12)]  # (dx, percent of the height)
        draw.polygon([(cx + dx, h * py // 100) for dx, py in outline], fill=colour)
        draw.ellipse([cx - 70, h * 7 // 100, cx + 70, h * 22 // 100], fill=(225, 185, 150))
    return image


def _room_photo(seed: int, rgb) -> Image.Image:
    """A room: a sofa, a window, a picture, a lamp and a door (of `rgb`, or wood)."""
    r = _rng(seed, "room")
    image = Image.new("RGB", PHOTO, r.choice(((235, 228, 210), (222, 228, 238), (238, 232, 240), (236, 236, 236))))
    draw = ImageDraw.Draw(image)
    draw.rectangle(_px(PHOTO, 0, 0.68, 1, 1), fill=r.choice(((150, 110, 75), (120, 90, 70), (180, 150, 110))))
    draw.rectangle(_px(PHOTO, 0.74, 0.06, 0.92, 0.70), fill=rgb or (150, 110, 75), outline=(90, 70, 60), width=6)
    draw.ellipse(_px(PHOTO, 0.755, 0.38, 0.775, 0.43), fill=(230, 215, 150))
    draw.rounded_rectangle(_px(PHOTO, 0.06, 0.45, 0.58, 0.78), radius=40, fill=r.choice(((70, 90, 140), (110, 110, 115), (150, 120, 90))))
    draw.rectangle(_px(PHOTO, 0.10, 0.10, 0.34, 0.34), fill=(90, 70, 60))
    draw.rectangle(_px(PHOTO, 0.11, 0.12, 0.33, 0.32), fill=(205, 195, 175))
    draw.rectangle(_px(PHOTO, 0.40, 0.14, 0.62, 0.48), fill=(205, 228, 250), outline=(250, 250, 250), width=12)
    draw.ellipse(_px(PHOTO, 0.64, 0.50, 0.68, 0.62), fill=(240, 215, 150))
    return image


def _food_photo(seed: int, rgb) -> Image.Image:
    """A plate on a table: round items of `rgb` (tomatoes) among others."""
    r = _rng(seed, "food")
    image = Image.new("RGB", PHOTO, r.choice(((160, 110, 70), (190, 160, 120), (120, 90, 60))))
    draw = ImageDraw.Draw(image)
    draw.ellipse(_px(PHOTO, 0.25, 0.06, 0.75, 0.94), fill=(250, 250, 250), outline=(225, 225, 225), width=14)
    palette = ((225, 150, 40), (130, 85, 40), (240, 225, 150), (250, 200, 90))
    for _ in range(r.randint(4, 7)):
        x, y, radius = r.randint(420, 860), r.randint(160, 560), r.randint(40, 85)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=rgb or r.choice(palette) if r.random() < 0.6 else r.choice(palette))
    for _ in range(6):
        x, y = r.randint(450, 830), r.randint(200, 520)
        draw.ellipse([x - 14, y - 14, x + 14, y + 14], fill=(70, 150, 70) if rgb is None else (240, 225, 150))
    return image


def _plant_photo(seed: int, rgb) -> Image.Image:
    """A potted plant, its leaves `rgb` (a green that is not the sign's)."""
    r = _rng(seed, "plant")
    leaf = rgb or DISTRACTOR_RGB["sign"]["plant"]
    image = Image.new("RGB", PHOTO, (238, 234, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle(_px(PHOTO, 0, 0.86, 1, 1), fill=(165, 130, 95))
    draw.polygon([_px(PHOTO, 0.40, 0.60), _px(PHOTO, 0.60, 0.60), _px(PHOTO, 0.56, 0.93), _px(PHOTO, 0.44, 0.93)], fill=(190, 100, 60))
    for _ in range(r.randint(8, 12)):
        x, y = r.randint(420, 860), r.randint(80, 420)
        shade = tuple(max(0, min(255, c + r.randint(-12, 12))) for c in leaf)
        draw.ellipse([x - 90, y - 34, x + 90, y + 34], fill=shade)
    return image


def _street_photo(seed: int, rgb) -> Image.Image:
    """A street: houses, a road, parked cars in other colours, a few small signs (never the sign's green) and passers-by, one
    of them in `rgb` (a green shirt)."""
    r = _rng(seed, "street")
    w, h = PHOTO
    image = _gradient(PHOTO, (170, 200, 230), (225, 232, 240))
    draw = ImageDraw.Draw(image)
    for k in range(5):
        x0 = k * w // 5
        wall = r.choice(((210, 190, 150), (130, 150, 175), (190, 190, 195), (200, 170, 150)))
        draw.rectangle([x0, h * r.randint(12, 30) // 100, x0 + w // 5 - 6, h * 66 // 100], fill=wall)
        for wx in range(x0 + 24, x0 + w // 5 - 40, 62):
            draw.rectangle([wx, h * 38 // 100, wx + 34, h * 38 // 100 + 54], fill=(80, 100, 130))
    draw.rectangle(_px(PHOTO, 0, 0.66, 1, 0.74), fill=(175, 175, 178))
    draw.rectangle(_px(PHOTO, 0, 0.74, 1, 1), fill=(85, 85, 90))
    for k in range(4):
        draw.rectangle(_px(PHOTO, 0.06 + k * 0.25, 0.88, 0.14 + k * 0.25, 0.90), fill=(235, 235, 235))
    for _ in range(r.randint(1, 2)):  # a parked car, in a colour that is no needle's
        x = r.randint(60, 900)
        body = _rgb(r.choice((DISTRACTOR_RGB["car"]["blue"], DISTRACTOR_RGB["car"]["grey"], DISTRACTOR_RGB["car"]["white"],
                              DISTRACTOR_RGB["sign"]["car"])))
        draw.rounded_rectangle([x, h * 70 // 100, x + 260, h * 80 // 100], radius=14, fill=body)
        draw.polygon([(x + 40, h * 70 // 100), (x + 70, h * 63 // 100), (x + 190, h * 63 // 100), (x + 220, h * 70 // 100)], fill=body)
        for cx in (x + 55, x + 205):
            draw.ellipse([cx - 22, h * 77 // 100, cx + 22, h * 77 // 100 + 44], fill=(20, 20, 20))
    for k in range(r.randint(2, 4)):  # small signs: a plate on a post
        x, y = r.randint(40, w - 120), r.randint(h * 40 // 100, h * 60 // 100)
        fill = _rgb(r.choice([DISTRACTOR_RGB["sign"][c] for c in ("yellow", "white", "blue", "grey")]))
        draw.rectangle([x + 18, y, x + 24, y + 150], fill=(90, 90, 95))
        draw.rounded_rectangle([x, y, x + 44, y + 40], radius=5, fill=fill, outline=_ink(fill), width=3)
    for k in range(r.randint(1, 3)):  # a passer-by
        x = r.randint(80, w - 120)
        shirt = rgb if (k == 0 and rgb) else r.choice(SAFE)
        draw.rectangle([x, h * 66 // 100, x + 36, h * 72 // 100 + 64], fill=(50, 60, 100))
        draw.rectangle([x - 4, h * 58 // 100, x + 40, h * 66 // 100 + 24], fill=shirt)
        draw.ellipse([x + 4, h * 52 // 100, x + 32, h * 59 // 100], fill=(225, 185, 150))
    return image


# -- the screen -------------------------------------------------------------------------------------------------------

APPS = ("editor", "browser", "terminal", "chat", "sheet", "slides")
ACCENTS = ("banner", "cells", "badge", "toast")
WORDS = ("main", "config", "parse", "render", "index", "value", "return", "import", "result", "total", "update", "queue",
         "stream", "handler", "buffer", "token", "client", "export", "merge", "cache")
FILES = ("app.py", "util.py", "server.py", "cli.py", "models.py", "tests.py", "README.md", "setup.cfg", "notes.txt")
CODE_RGB = ((86, 156, 214), (78, 201, 176), (206, 145, 120), (197, 134, 192), (220, 220, 170), (181, 206, 168))
THEMES = ((30, 50, 110), (20, 110, 120), (50, 50, 58), (240, 232, 215), (60, 40, 110), (30, 90, 60))


def _line(r: random.Random, lo: int = 2, hi: int = 6) -> str:
    return " ".join(r.choice(WORDS) for _ in range(r.randint(lo, hi)))


def screen_state(state: dict) -> Image.Image:
    """One screen state, 1280x720: the app card (`editor`, `browser`, `terminal`, `chat`, `sheet` or `slides`), then its accent
    (`banner`, `cells`, `badge` or `toast`, in a red that is not the error's), then its overlay: a dialog 45% by 35% of the screen,
    centred on the overlay's `at` (a place of DIALOG_AREA; the middle without one), with a header in the kind's colour. An `error`'s
    header is the error needle's shade, titled `ERROR <code>` in 40 px letters (as is every kind's title; the header stays 38% of the
    dialog's height); a warning is orange, an info blue, a save prompt grey and a confirm ("Delete 14 files?") another red. The
    taskbar is along the bottom, with no clock (`screen_frame` adds it, and the cursor)."""
    r = _rng(state.get("seed", 0), "screen")
    image = _APPS[state["app"]](r)
    _decorate(image, state, r)
    return image


def _decorate(image: Image.Image, state: dict, r: random.Random) -> None:
    """What stands on top of the app card and never moves: the state's accent, its overlay, the taskbar."""
    if accent := state.get("accent"):
        _accent(image, accent, r)
    if overlay := state.get("overlay"):
        _dialog(image, overlay, r)
    _taskbar(image)


def _chrome(draw: ImageDraw.ImageDraw, title: str, fill=(225, 228, 235), ink=(40, 40, 48)) -> None:
    draw.rectangle([0, 0, SCREEN[0], 40], fill=fill)
    draw.text((20, 6), title, fill=ink, font=font(24))
    for k in range(3):
        draw.ellipse([SCREEN[0] - 120 + k * 34, 12, SCREEN[0] - 102 + k * 34, 30], fill=(150, 152, 160))


def _editor(r: random.Random) -> Image.Image:
    dark = r.random() < 0.7
    bg, fg, side, bar = (((30, 30, 36), (212, 212, 212), (37, 37, 43), (52, 52, 60)) if dark
                         else ((252, 252, 252), (40, 40, 46), (240, 240, 243), (225, 228, 235)))
    image = Image.new("RGB", SCREEN, bg)
    draw = ImageDraw.Draw(image)
    name = r.choice(FILES)
    _chrome(draw, f"{name} - Visual Studio Code", bar, fg)
    draw.rectangle([0, 40, 56, SCREEN[1] - 28], fill=bar)
    draw.rectangle([56, 40, 300, SCREEN[1] - 28], fill=side)
    for k, file in enumerate(r.sample(FILES, 7)):
        draw.text((76, 62 + k * 38), file, fill=fg, font=font(24))
    draw.rectangle([300, 40, SCREEN[0], 80], fill=side)
    draw.text((320, 48), name, fill=fg, font=font(24))
    for k in range(15):
        y = 100 + k * 34
        draw.text((316, y), str(k + 1), fill=(120, 120, 130), font=font(24))
        draw.text((380 + 40 * r.randint(0, 3), y), _line(r), fill=r.choice(CODE_RGB) if dark else (40, 70, 130), font=font(26))
    draw.rectangle([0, SCREEN[1] - 28, SCREEN[0], SCREEN[1]], fill=(0, 122, 204))
    draw.text((16, SCREEN[1] - 26), "main  0 problems  Ln 12, Col 8  UTF-8", fill="white", font=font(20))
    return image


def _browser(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, SCREEN[0], 92], fill=(225, 228, 235))
    draw.rounded_rectangle([20, 6, 280, 40], radius=10, fill=(250, 250, 250))
    draw.text((36, 10), r.choice(("Docs", "News", "Shop", "Mail")), fill=(50, 50, 60), font=font(22))
    draw.rounded_rectangle([160, 44, SCREEN[0] - 160, 88], radius=18, fill="white")  # (the address that is typed at)
    draw.text((190, 54), f"https://example.com/{r.choice(('docs', 'news', 'shop', 'weather'))}", fill=(60, 60, 70), font=font(24))
    draw.text((80, 130), _line(r).title(), fill=(30, 30, 40), font=font(54))
    block = r.choice(SAFE)
    draw.rectangle([80, 220, 480, 440], fill=block)
    for k in range(7):
        draw.rounded_rectangle([520, 230 + k * 40, 520 + r.randint(400, 640), 254 + k * 40], radius=8, fill=(210, 210, 215))
    for k in range(4):
        draw.text((80, 470 + k * 44), _line(r, 5, 9), fill=(60, 60, 70), font=font(28))
    return image


def _terminal(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (14, 14, 16))
    draw = ImageDraw.Draw(image)
    _chrome(draw, "Terminal", (45, 45, 52), (220, 220, 225))
    for k in range(14):
        y = 56 + k * 40
        if k % 4 == 0:
            draw.text((30, y), "$ ", fill=(80, 220, 110), font=font(28))
            draw.text((70, y), _line(r, 1, 4), fill=(235, 235, 235), font=font(28))
        else:
            draw.text((30, y), _line(r), fill=r.choice(((200, 200, 205), (110, 190, 255), (230, 200, 90))), font=font(28))
    draw.text((30, 56 + 14 * 40), "$ ", fill=(80, 220, 110), font=font(28))  # the prompt that is typed at
    return image


def _chat(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (232, 232, 236))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 280, SCREEN[1]], fill=(60, 40, 90))
    for k, channel in enumerate(r.sample(WORDS, 8)):
        draw.text((24, 30 + k * 48), f"# {channel}", fill=(230, 225, 240), font=font(26))
    for k in range(6):
        mine = k % 2
        x0 = 640 if mine else 320
        draw.rounded_rectangle([x0, 40 + k * 100, x0 + 560, 110 + k * 100], radius=24, fill=(70, 140, 240) if mine else "white")
        draw.text((x0 + 24, 60 + k * 100), _line(r, 3, 6), fill="white" if mine else (40, 40, 50), font=font(28))
    draw.rounded_rectangle([320, SCREEN[1] - 104, SCREEN[0] - 40, SCREEN[1] - 56], radius=14, fill="white")  # the message that is typed in
    return image


def _sheet(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (252, 252, 252))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, SCREEN[0], 40], fill=(225, 235, 225))
    draw.text((20, 6), f"{r.choice(('budget', 'sales', 'plan', 'tracker'))}.xlsx", fill=(30, 60, 40), font=font(24))
    draw.text((14, 50), "fx", fill=(120, 120, 130), font=font(24))  # the formula bar (40 to 84) is typed in
    draw.rectangle([0, 84, SCREEN[0], 124], fill=(240, 240, 242))
    for x in range(0, SCREEN[0], 160):
        draw.line([x, 84, x, SCREEN[1]], fill=(200, 200, 205), width=2)
        draw.text((x + 70, 92), "ABCDEFGH"[x // 160], fill=(80, 80, 90), font=font(24))
    for y in range(124, SCREEN[1], 40):
        draw.line([0, y, SCREEN[0], y], fill=(200, 200, 205), width=2)
        for x in range(0, SCREEN[0], 160):
            draw.text((x + 10, y + 4), f"{r.randint(1, 999)}.{r.randint(10, 99)}", fill=(30, 30, 40), font=font(24))
    draw.rectangle([160, 164, 480, 244], outline=(30, 110, 220), width=4)
    return image


def _slides(r: random.Random) -> Image.Image:
    image = Image.new("RGB", SCREEN, (236, 236, 240))
    draw = ImageDraw.Draw(image)
    _chrome(draw, "Slides", (225, 228, 235))
    theme = r.choice(THEMES)
    for k in range(4):
        draw.rectangle([24, 70 + k * 150, 220, 190 + k * 150], fill=theme if k == 0 else (250, 250, 252), outline=(160, 160, 170), width=2)
    draw.rectangle([260, 70, SCREEN[0] - 40, SCREEN[1] - 40], fill=theme)
    ink = _ink(theme)
    draw.text((320, 120), _line(r, 2, 3).title(), fill=ink, font=font(72))
    for k in range(4):
        draw.text((340, 270 + k * 70), "- " + _line(r, 3, 6), fill=ink, font=font(38))
    return image


_APPS = {"editor": _editor, "browser": _browser, "terminal": _terminal, "chat": _chat, "sheet": _sheet, "slides": _slides}


def _accent(image: Image.Image, accent: dict, r: random.Random) -> None:
    """A red that is not the error's: an alert banner, a block of red cells, a badge on the title bar or a toast."""
    kind = accent["kind"]
    rgb = _rgb(accent.get("rgb") or DISTRACTOR_RGB["error"][kind])
    if distance(rgb, NEEDLE_RGB["error"]) < SHADE_GAP:
        raise ValueError(f"the {kind} accent {rgb} is within {SHADE_GAP} of the error needle's shade {NEEDLE_RGB['error']}")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    if kind == "banner":
        draw.rectangle([0, 96, w, 226], fill=rgb)
        words = r.choice(("Limited offer: 50% off today", "Service alert: delays expected", "Sale ends tonight"))
        draw.text((80, 130), words, fill="white", font=font(48))
    elif kind == "cells":
        x0, y0 = 160 * r.randint(1, 4), 40 * r.randint(3, 9)
        draw.rectangle([x0, y0, x0 + 320, y0 + 120], fill=rgb)
        for k in range(3):
            draw.text((x0 + 12, y0 + 6 + k * 40), f"{r.randint(1, 999)}.{r.randint(10, 99)}", fill=(30, 30, 40), font=font(24))
    elif kind == "badge":
        draw.ellipse([w - 70, 4, w - 30, 38], fill=rgb)
        draw.text((w - 50, 21), str(r.randint(1, 9)), fill="white", font=font(22), anchor="mm")
    else:
        draw.rounded_rectangle([w - 440, h - 140, w - 24, h - 40], radius=14, fill=rgb)
        draw.text((w - 420, h - 112), r.choice(("Sync failed, retrying", "Upload paused", "Low disk warning")), fill="white", font=font(32))


def _dialog_box(overlay: dict) -> tuple[int, int, int, int]:
    """A dialog's left, top, width and height (its shadow is 12 px to the right and 14 below), where `overlay["at"]` puts it."""
    w, h = SCREEN
    dw, dh = round(w * DIALOG[0]), round(h * DIALOG[1])
    x0, y0 = (w - dw) // 2, (h - dh) // 2
    if overlay.get("at"):
        cx, cy = overlay["at"]
        if not (DIALOG_AT[0][0] <= cx <= DIALOG_AT[0][1] and DIALOG_AT[1][0] <= cy <= DIALOG_AT[1][1]):
            raise ValueError(f"a {overlay['kind']} dialog opens at {overlay['at']}, outside the places of DIALOG_AT {DIALOG_AT}")
        x0, y0 = round(w * cx - dw / 2), round(h * cy - dh / 2)
    return x0, y0, dw, dh


def _dialog(image: Image.Image, overlay: dict, r: random.Random) -> None:
    kind = overlay["kind"]
    want = NEEDLE_RGB["error"] if kind == "error" else DISTRACTOR_RGB["error"][kind]
    head = _rgb(overlay.get("rgb") or want)
    if kind == "error" and head != NEEDLE_RGB["error"]:
        raise ValueError(f"an error dialog's header is the error needle's shade {NEEDLE_RGB['error']}, not {head}")
    if kind != "error" and distance(head, NEEDLE_RGB["error"]) < SHADE_GAP:
        raise ValueError(f"a {kind} dialog's header {head} is within {SHADE_GAP} of the error needle's shade {NEEDLE_RGB['error']}")
    draw = ImageDraw.Draw(image)
    x0, y0, dw, dh = _dialog_box(overlay)
    top = round(dh * 0.38)
    code = overlay.get("code")
    title = {"error": f"ERROR {code}" if code else "ERROR", "warning": f"WARNING {code}" if code else "WARNING", "info": "INFO",
             "save": "Unsaved changes", "confirm": "Confirm"}[kind]
    draw.rectangle([x0 + 12, y0 + 14, x0 + dw + 12, y0 + dh + 14], fill=(70, 70, 76))
    draw.rectangle([x0, y0, x0 + dw, y0 + dh], fill=(248, 248, 250), outline=(150, 150, 155), width=2)
    draw.rectangle([x0, y0, x0 + dw, y0 + top], fill=head)
    _fit(draw, (x0 + 28, y0 + top // 2), title, "white", TITLE_PX, dw - 56, anchor="lm")
    message = overlay.get("msg") or {"error": "The operation failed", "warning": "Check before you go on", "info": "Done",
                                     "save": "Save changes to report.docx?", "confirm": "Delete 14 files?"}[kind]
    _fit(draw, (x0 + 28, y0 + top + 50), str(message), (30, 30, 36), 38, dw - 56, anchor="lm")
    buttons = {"error": (("OK", (225, 225, 230)),), "warning": (("OK", (225, 225, 230)),), "info": (("OK", (225, 225, 230)),),
               "save": (("Cancel", (225, 225, 230)), ("Save", (60, 110, 220))),
               "confirm": (("Cancel", (225, 225, 230)), ("Delete", _rgb(DISTRACTOR_RGB["error"]["confirm"])))}[kind]
    x = x0 + dw - 24
    for label, fill in reversed(buttons):
        bw = 150
        draw.rounded_rectangle([x - bw, y0 + dh - 76, x, y0 + dh - 24], radius=10, fill=fill)
        draw.text((x - bw // 2, y0 + dh - 50), label, fill=_ink(fill), font=font(30), anchor="mm")
        x -= bw + 16


# -- the screen's frames: the layers, the typing, the taskbar's clock and the cursor -----------------------------------------------

TICK_S = 0.5  # a frame is at a tick: the recorder's 2 a second


class Typing(NamedTuple):
    """Where an app scrolls and is typed in (1280x720 px). `box` is what a scroll moves (up `line_h` px a line, wrapping), `rect` the
    typing line, filled with the card's background where text is typed, `at` where its text starts (after `prefix`), in `size` px
    letters, `cap` the most visible (not space) characters it holds, `ink` the letters' colour on a light and on a dark background (None: the slide's own)."""

    box: tuple[int, int, int, int]
    line_h: int
    rect: tuple[int, int, int, int]
    at: tuple[int, int]
    size: int
    cap: int
    ink: tuple | None
    prefix: str = ""
    stroke: int = 0


TYPING = {  # (the rect holds the text's whole height, descenders too, and is clear of the edge of what it is typed in)
    "editor": Typing((304, 96, 1280, 606), 34, (372, 610, 1272, 656), (380, 608), 36, 40, ((40, 40, 46), (212, 212, 212)), "", 1),
    "browser": Typing((60, 100, 1240, 672), 44, (186, 47, 1100, 85), (190, 43), 34, 44, ((60, 60, 70), (235, 235, 235)), "", 1),
    "terminal": Typing((0, 52, 1280, 612), 40, (66, 617, 1272, 665), (70, 614), 38, 52, ((20, 20, 24), (235, 235, 235)), "", 1),
    "chat": Typing((292, 0, 1280, 612), 34, (340, 618, 1224, 662), (346, 613), 38, 37, ((40, 40, 50), (235, 235, 235)), "", 1),
    "sheet": Typing((0, 124, 1280, 644), 40, (56, 42, 1272, 82), (60, 39), 34, 58, ((30, 30, 40), (235, 235, 235)), "", 1),
    "slides": Typing((260, 70, 1240, 560), 70, (336, 555, 1236, 610), (340, 550), 46, 30, None, "- ", 1),
}


class ScreenFrame(NamedTuple):
    """One frame of a screen segment: its second (from the segment's start, a multiple of TICK_S), the index of the state it is in, the
    cursor's tip, the taskbar's clock (`14:05`) and date (`2026-10-28`: ISO, since `01/11` reads as 11 January on a US machine and
    a week answer names its day), the characters typed on the app's typing line and the lines scrolled since the state's start."""

    second: float
    state: int
    cursor: tuple[int, int]
    clock: str
    date: str
    typed: int = 0
    lines: int = 0


def _taskbar(image: Image.Image) -> None:
    """The taskbar: a dark strip along the bottom with a logo and some icons in the middle; the clock is the frame's."""
    w, h = image.size
    top = h - TASKBAR_H
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, top, w, h], fill=TASKBAR_RGB)
    draw.line([0, top, w, top], fill=TASKBAR_LINE, width=1)
    x = w // 2 - 130
    for dx, dy in ((0, 0), (11, 0), (0, 11), (11, 11)):
        draw.rectangle([x + dx, top + 9 + dy, x + dx + 9, top + 9 + dy + 9], fill=TASKBAR_LOGO)
    for k in range(5):
        draw.rounded_rectangle([x + 48 + k * 44, top + 8, x + 48 + k * 44 + 24, top + 32], radius=6, fill=TASKBAR_ICON)


@lru_cache(maxsize=16)
def _cached_font(size: int):
    return font(size)


def typing_text(seed: int) -> str:
    """The text a state's owner types, from its seed: words with spaces between."""
    r = _rng(seed, "typing")
    return " ".join(r.choice(WORDS) for _ in range(120))


def typed_text(text: str, lines: int, n: int, prefix: str = "") -> str:
    """What is on the typing line when `n` characters are typed after `lines` lines have been scrolled: `prefix`, then `text` from a place
    that the lines give, up to its n-th visible (not space) character, so that every character typed changes the picture."""
    start, seen = lines * 41 % 300, 0
    for i in range(start, len(text)):
        seen += text[i] != " "
        if seen >= n:
            return prefix + text[start:i + 1]
    return prefix + text[start:]


def _decorations(state: dict) -> list[tuple[int, int, int, int]]:
    """Where the decorations of `state` are, (x0, y0, x1, y1) each, so far as they can hide a typing line: the taskbar, a toast, a banner, a
    badge, the box a block of red cells is in, and the dialog with its shadow."""
    w, h = SCREEN
    out = [(0, h - TASKBAR_H, w, h)]
    kind = (state.get("accent") or {}).get("kind")
    if kind == "toast":
        out.append((w - 440, h - 140, w - 24, h - 40))
    elif kind == "banner":
        out.append((0, 96, w, 226))
    elif kind == "badge":
        out.append((w - 70, 4, w - 30, 38))
    elif kind == "cells":
        out.append((160, 120, 1120 + 160, 480))  # (x0 is 160 to 640, y0 120 to 360)
    if overlay := state.get("overlay"):
        x0, y0, dw, dh = _dialog_box(overlay)
        out.append((x0, y0, x0 + dw + 12, y0 + dh + 14))
    return out


@lru_cache(maxsize=8)
def _rows(app: str) -> tuple[int, int]:
    """The first and last row of the app's typed text, whatever the letters (WORDS' ascenders and descenders, and the stroke)."""
    typing = TYPING[app]
    top, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(typing.at, " ".join(WORDS), font=_cached_font(typing.size), stroke_width=typing.stroke)[1::2]
    return top, bottom


def typing_limit(state: dict) -> int:
    """The x up to which the typing line of `state` shows: where the first decoration over the text's rows begins, else the line's end."""
    typing = TYPING[state["app"]]
    top, bottom = _rows(state["app"])
    limit = typing.rect[2]
    for x0, y0, x1, y1 in _decorations(state):
        if y0 < bottom and y1 > top and x1 > typing.at[0]:
            limit = min(limit, max(x0, typing.at[0]))
    return limit


def typing_cap(state: dict, lines: int) -> int:
    """The most visible characters the typing line of `state` can show after `lines` lines have been scrolled: the app's `cap`, or fewer
    where a decoration (a toast over the end of the line) hides the rest: a character typed under one would change nothing."""
    typing = TYPING[state["app"]]
    limit = typing_limit(state)
    if limit >= typing.rect[2]:
        return typing.cap
    text, font = typing_text(state.get("seed", 0)), _cached_font(typing.size)
    for n in range(1, typing.cap + 1):
        if typing.at[0] + font.getlength(typed_text(text, lines, n, typing.prefix)) + 2 * typing.stroke > limit:
            return n - 1
    return typing.cap


class ScreenLayers:
    """A state's picture in layers, so that a frame is a copy, a roll, a short text and a masked paste: the app card (`card`); what stands
    on top of it and never moves (`deco`, the accent, the overlay and the taskbar, drawn as `screen_state` draws them) with its exact
    `mask` (255 where a decoration pixel is, found by drawing the decorations on an all black and an all white background: a pixel that is
    the same on both is theirs, whatever the card shows there; a dialog's white body over a white page is in the mask); and the typing
    line's `bg` and `ink`, with the `text` the owner types (from the state's seed)."""

    def __init__(self, state: dict):
        seed = state.get("seed", 0)
        r = _rng(seed, "screen")
        self.card = _APPS[state["app"]](r)
        after = r.getstate()
        drawn = []
        for sentinel in ((0, 0, 0), (255, 255, 255)):
            image = Image.new("RGB", SCREEN, sentinel)
            r.setstate(after)
            _decorate(image, state, r)
            drawn.append(image)
        same = (np.asarray(drawn[0]) == np.asarray(drawn[1])).all(axis=2)
        self.deco = drawn[0]
        self.mask = Image.fromarray(same.astype(np.uint8) * 255, "L")
        rows = np.flatnonzero(same.any(axis=1))
        self.regions = []  # the decorations by bands of rows, each pasted by its own box: (xy, deco, mask), cheaper than the whole frame
        for band in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
            y0, y1 = int(band[0]), int(band[-1]) + 1
            cols = np.flatnonzero(same[y0:y1].any(axis=0))
            box = (int(cols[0]), y0, int(cols[-1]) + 1, y1)
            self.regions.append((box[:2], self.deco.crop(box), self.mask.crop(box)))
        self.typing = TYPING[state["app"]]
        self.bg = self.card.getpixel((self.typing.rect[2] - 6, self.typing.rect[1] + 3))
        light = 0.299 * self.bg[0] + 0.587 * self.bg[1] + 0.114 * self.bg[2] > 150
        self.ink = _ink(self.bg) if self.typing.ink is None else self.typing.ink[0 if light else 1]
        _clear_of_needles(self.ink, "the typing ink", "screen")
        self.text = typing_text(seed)
        self._rolled: tuple[int, Image.Image] = (0, self.card)

    def scrolled(self, lines: int) -> Image.Image:
        """The card with its scroll box rolled up by `lines` lines (wrapping: the lines that leave at the top come back at the bottom)."""
        if self._rolled[0] != lines:
            box = self.typing.box
            shift = lines * self.typing.line_h % (box[3] - box[1])
            image = self.card.copy()
            if shift:
                image.paste(ImageChops.offset(self.card.crop(box), 0, -shift), box[:2])
            self._rolled = (lines, image)
        return self._rolled[1]

    def typed(self, frame: ScreenFrame) -> str:
        """The text on the typing line of `frame`: its first `frame.typed` visible characters (`typed_text`)."""
        return typed_text(self.text, frame.lines, frame.typed, self.typing.prefix)


def screen_layers(state: dict) -> ScreenLayers:
    """`state` in layers, which `screen_frame` draws a frame from; with no scroll or typing they make `screen_state(state)` exactly."""
    return ScreenLayers(state)


def screen_frames(spec: dict, start: float) -> list[ScreenFrame]:
    """The frames of the screen segment `spec` that starts at `start` (local epoch, a whole second): what the recorder keeps, which is
    a frame whenever the screen changes and at least one every FLOOR_S, at most 2 a second (a tick is TICK_S). The reader reads the first of
    each second. A frame is at the start of each state (the cursor where it is), each of the states' `acts` (at `acts[k][0]` ticks from the
    state's start: a cursor `move`, a `type` or a `scroll`), each minute change of the taskbar's clock, then wherever more than FLOOR_S passes
    without one; in time order, all before `length_s`. The frame shows the cursor of the last `move`, the characters typed by the last
    `type` (none after a `scroll`, which starts the typing line afresh) and the lines scrolled by the last `scroll`. This is the one place
    that says where a frame is."""
    states, length = spec["states"], int(math.ceil(spec["length_s"]))
    starts = [s["from"] for s in states]
    marks = {2 * a for a in starts} | {2 * s["from"] + act[0] for s in states for act in s.get("acts", ())}
    marks |= {2 * sec for sec in range(1, length) if (int(start) + sec) % 60 == 0}
    out = []
    k, i, cursor, typed, lines = -1, 0, CURSOR_HOME, 0, 0
    for sec in with_floor([m * TICK_S for m in sorted(m for m in marks if 0 <= m < 2 * length)], length):
        now = bisect.bisect_right(starts, sec) - 1
        state = states[now]
        if now != k:
            k, i, cursor, typed, lines = now, 0, tuple(state.get("cursor", CURSOR_HOME)), 0, 0
        acts = state.get("acts", ())
        while i < len(acts) and state["from"] + acts[i][0] * TICK_S <= sec:
            kind, value = acts[i][1], acts[i][2:]
            if kind == "move":
                cursor = (value[0], value[1])
            elif kind == "type":
                typed = value[0]
            else:
                lines, typed = value[0], 0
            i += 1
        when = datetime.fromtimestamp(int(start) + int(sec))
        out.append(ScreenFrame(sec, k, (int(cursor[0]), int(cursor[1])), when.strftime("%H:%M"), when.strftime("%Y-%m-%d"), typed, lines))
    return out


def with_floor(times: list[float], length: float) -> list[float]:
    """`times` (sorted, seconds, before `length`) with a time added wherever more than FLOOR_S passes with no frame, from one frame to
    the next and to `length`: the recorder's `mpdecimate max=119`."""
    out: list[float] = []
    for sec in (*times, length):
        while out and sec - out[-1] > FLOOR_S:
            out.append(out[-1] + FLOOR_S)
        if sec < length:
            out.append(sec)
    return out


def screen_frame(layers: ScreenLayers, frame: ScreenFrame) -> Image.Image:
    """The state's `layers` as `frame` shows it: the card scrolled by the frame's lines, its typed text on the typing line, the
    decorations (accent, dialog, taskbar) pasted over both through their mask, then the taskbar's clock and date at the right and the
    cursor (a white arrow with a black outline, its tip at the frame's `cursor`) over everything."""
    image = layers.scrolled(frame.lines).copy()
    draw = ImageDraw.Draw(image)
    if frame.typed:
        typing = layers.typing
        draw.rectangle(typing.rect, fill=layers.bg)
        draw.text(typing.at, layers.typed(frame), fill=layers.ink, font=_cached_font(typing.size), stroke_width=typing.stroke, stroke_fill=layers.ink)
    for xy, deco, mask in layers.regions:
        image.paste(deco, xy, mask)
    stamp(image, frame)
    return image


def stamp(image: Image.Image, frame: ScreenFrame) -> None:
    """Draw the taskbar's clock and date at the right of `image` and the frame's cursor over everything."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.text((w - 16, h - TASKBAR_H + 3), frame.clock, fill=TASKBAR_INK, font=_cached_font(17), anchor="ra")
    draw.text((w - 16, h - TASKBAR_H + 21), frame.date, fill=TASKBAR_INK, font=_cached_font(15), anchor="ra")
    x, y = frame.cursor
    draw.polygon([(x + dx, y + dy) for dx, dy in CURSOR_ARROW], fill=(255, 255, 255), outline=(0, 0, 0))


# -- the porch --------------------------------------------------------------------------------------------------------

BOX = (0.13, 0.43, 0.38, 0.69)  # the parcel's front face on the step, as fractions of the frame: 160x95 px, 6.6% of it
BOX_W, BOX_H = 0.25, 0.26  # its size
TOP_STRIP = 26  # px: its lighter top face
BOX_X0, BOX_Y1 = (0.04, 0.20), (0.67, 0.70)  # where it may be: the range of its left edge and of its bottom edge, on the step by the door
BOX_AREA = (BOX_X0[0], BOX_Y1[0] - BOX_H - TOP_STRIP / 360, BOX_X0[1] + BOX_W + 12 / 640, BOX_Y1[1])
# ^ every place it can be (x0, y0, x1, y1) as fractions of the frame: the union of its positions, face and top face
WALLS = ((205, 200, 190), (196, 198, 200), (210, 205, 195), (190, 196, 190))
JACKETS = ((40, 60, 120), (60, 60, 70), (200, 60, 70), (230, 230, 235), (60, 120, 90), (120, 60, 140), (240, 200, 40))


def parcel_box(r: random.Random) -> list[float]:
    """Where a parcel is left on the step: the face's `[x0, y0, x1, y1]` as fractions of the frame, one of the places of BOX_AREA, from `r`.
    A parcel stays where it was left, so the planner gives every clip that shows it the same one (`box` in the clip's spec)."""
    x0, y1 = round(r.uniform(*BOX_X0), 3), round(r.uniform(*BOX_Y1), 3)
    return [x0, round(y1 - BOX_H, 3), round(x0 + BOX_W, 3), y1]


def porch_frame(spec: dict, second: float, stamp: str) -> Image.Image:
    """Second `second` of the clip `spec`: the door, the step and the path, the clip's own props (a doormat, a planter, a bin), the
    events that are on at that second, then (so that no light touches it) the parcel on the step, a cardboard box with tape and a
    label in the parcel needle's shade, and the camera's time `stamp` along the bottom. A `courier` walks in carrying a box (in
    the shadow, a darker brown), puts it down at `drop_at` (the clip's `parcel` then shows it on the step) and walks out; with
    `"drop_at": None` he never puts it down: he carries it to the door and back out, and the step is empty in every frame (a
    distractor: nothing is delivered). A `visitor` (`x`, `scale` 3, `bag`) stands close to the camera in a brown coat, a third of the
    frame or more: more brown, in the parcel's place, than a parcel has. A `walker`, `cat`, `dog` or `car` crosses; an
    `owner_pickup` is the owner at the door, lifting the parcel at the `parcel`'s end; a `light` dims or brightens the scene."""
    image = _porch_base(spec.get("seed", 0)).copy()
    draw = ImageDraw.Draw(image)
    w, h = PORCH
    parcel = spec.get("parcel")
    for event in spec.get("events", ()):
        if event["kind"] == "light" and event["from"] <= second < event["to"]:
            rgb = _rgb(event.get("rgb") or (25, 30, 55))
            _clear_of_needles(rgb, "a light event's tint", "porch")
            image = Image.blend(image, Image.new("RGB", PORCH, rgb), 0.3)
            draw = ImageDraw.Draw(image)
    for event in spec.get("events", ()):
        if event["from"] <= second <= event["to"] and event["kind"] != "light":
            _event(draw, event, second, parcel, _rng(spec.get("seed", 0), f"{event['kind']}-{event['from']}"))
    if parcel and parcel[0] <= second < parcel[1]:
        _parcel(draw, spec.get("box"))
    draw.rectangle([0, h - 26, w, h], fill=(0, 0, 0))
    draw.text((12, h - 24), stamp, fill="white", font=font(18))
    return image


@lru_cache(maxsize=16)
def _porch_base(seed: int) -> Image.Image:
    """The door, step, path and props of a clip: a function of its seed."""
    r = _rng(seed, "porch")
    w, h = PORCH
    image = Image.new("RGB", PORCH, r.choice(WALLS))
    draw = ImageDraw.Draw(image)
    door = _rgb(DISTRACTOR_RGB["parcel"]["door"])
    draw.rectangle([36, 0, 236, 216], fill=(90, 90, 95))
    draw.rectangle([42, 6, 230, 216], fill=door)
    for x0, y0, x1, y1 in ((56, 20, 128, 100), (146, 20, 218, 100), (56, 116, 128, 200), (146, 116, 218, 200)):
        draw.rectangle([x0, y0, x1, y1], outline=(30, 45, 75), width=3)
    draw.ellipse([204, 118, 214, 128], fill=(215, 205, 160))
    draw.rectangle([0, 216, 300, 256], fill=(170, 168, 160))
    draw.rectangle([0, 256, 300, 276], fill=(140, 138, 130))
    draw.rectangle([0, 276, w, h], fill=(150, 148, 140))
    for x in (60, 190, 330, 470):
        draw.line([x, 276, x - 30, h], fill=(120, 118, 110), width=3)
    draw.rectangle([300, 216, w, 276], fill=(80, 125, 70))
    if r.random() < 0.8:  # a doormat
        draw.rectangle([70, 240, 200, 252], fill=_rgb(DISTRACTOR_RGB["parcel"]["mat"]))
    if r.random() < 0.7:  # a planter
        draw.polygon([(250, 196), (292, 196), (286, 250), (256, 250)], fill=_rgb(DISTRACTOR_RGB["parcel"]["planter"]))
        draw.ellipse([240, 150, 302, 206], fill=(60, 130, 60))
    if r.random() < 0.6:  # a bin
        draw.rectangle([330, 150, 392, 262], fill=(60, 90, 70))
        draw.rectangle([326, 142, 396, 156], fill=(45, 70, 55))
    return image


def _person(draw: ImageDraw.ImageDraw, x: int, feet: int, jacket, scale: float = 1.0, carrying=None) -> None:
    """A person standing at `x` with feet at `feet`: head, jacket, trousers; `carrying` is the colour of a box held in front."""
    body, head = int(46 * scale), int(12 * scale)
    draw.rectangle([x - body // 2, feet - int(46 * scale), x - 2, feet], fill=(45, 50, 70))
    draw.rectangle([x + 2, feet - int(46 * scale), x + body // 2, feet], fill=(45, 50, 70))
    draw.rectangle([x - body // 2, feet - int(100 * scale), x + body // 2, feet - int(44 * scale)], fill=jacket)
    draw.ellipse([x - head, feet - int(130 * scale), x + head, feet - int(104 * scale)], fill=(225, 185, 150))
    if carrying:
        draw.rectangle([x - int(36 * scale), feet - int(88 * scale), x + int(10 * scale), feet - int(58 * scale)], fill=carrying)
        draw.rectangle([x - int(36 * scale), feet - int(88 * scale), x + int(10 * scale), feet - int(82 * scale)], fill=(160, 130, 95))


def _event(draw: ImageDraw.ImageDraw, event: dict, second: float, parcel, r: random.Random) -> None:
    kind, a, b = event["kind"], event["from"], event["to"]
    w, h = PORCH
    u = (second - a) / max(b - a, 1e-9)
    rgb = event.get("rgb")
    if rgb is not None:
        _clear_of_needles(rgb, f"a {kind} event's colour", "porch")
        rgb = _rgb(rgb)
    flip = r.random() < 0.5
    across = (1 - u) if flip else u
    if kind == "courier":
        jacket = rgb or _rgb(DISTRACTOR_RGB["parcel"]["courier"])
        carried = _rgb(DISTRACTOR_RGB["parcel"]["carried"])
        if "drop_at" in event and event["drop_at"] is None:  # he keeps the box: in to the door by the middle of his time, and out with it
            x = int(w + 40 + (330 - w - 40) * (1 - abs(2 * u - 1)))
            _person(draw, x, 300, jacket, carrying=carried)
            return
        drop = event.get("drop_at", (a + b) / 2)
        if second < drop:
            x = int(w + 40 + (330 - w - 40) * (second - a) / max(drop - a, 1e-9))
            _person(draw, x, 300, jacket, carrying=carried)
        else:
            x = int(330 + (w + 40 - 330) * (second - drop) / max(b - drop, 1e-9))
            _person(draw, x, 300, jacket)
    elif kind == "walker":
        _person(draw, int(-30 + (w + 60) * across), 330, rgb or r.choice(JACKETS), 0.85)
    elif kind == "visitor":  # close to the camera, in a brown coat, waiting at the door (with a carrier bag, or without): more brown than any parcel
        scale = float(event.get("scale", 3.0))
        _person(draw, int(event.get("x", 190) + 5 * math.sin(second)), int(130 * scale + 14), rgb or _rgb(DISTRACTOR_RGB["parcel"]["courier"]), scale,
                carrying=_rgb(DISTRACTOR_RGB["parcel"]["carried"]) if event.get("bag") else None)
    elif kind == "cat":
        x = int(-60 + (w + 120) * across)
        fur = rgb or r.choice(((60, 60, 65), (235, 235, 230), (140, 140, 145), _rgb(DISTRACTOR_RGB["parcel"]["cat"])))
        draw.ellipse([x, 296, x + 56, 328], fill=fur)
        draw.polygon([(x + 38, 300), (x + 42, 284), (x + 50, 300)], fill=fur)
        draw.line([x, 310, x - 22, 296], fill=fur, width=5)
    elif kind == "dog":
        x = int(-90 + (w + 180) * across)
        fur = rgb or _rgb(DISTRACTOR_RGB["parcel"]["dog"])
        draw.ellipse([x, 286, x + 90, 322], fill=fur)
        draw.ellipse([x + 70 if not flip else x - 22, 276, x + 112 if not flip else x + 20, 306], fill=fur)
        for lx in (x + 8, x + 66):
            draw.rectangle([lx, 314, lx + 10, 340], fill=fur)
    elif kind == "car":
        x = int(-170 + (w + 340) * across)
        body = rgb or r.choice(((30, 60, 200), (245, 245, 245), (130, 130, 135), (240, 20, 20), (240, 210, 20)))
        draw.rounded_rectangle([x, 290, x + 160, 330], radius=10, fill=body)
        draw.polygon([(x + 28, 290), (x + 46, 266), (x + 118, 266), (x + 136, 290)], fill=body)
        for cx in (x + 36, x + 124):
            draw.ellipse([cx - 14, 316, cx + 14, 344], fill=(20, 20, 20))
    elif kind == "owner_pickup":
        jacket = rgb or r.choice(JACKETS)
        lifted = bool(parcel) and second >= parcel[1]
        _person(draw, 300, 262, jacket, 1.1, carrying=_rgb(DISTRACTOR_RGB["parcel"]["carried"]) if lifted else None)
    else:
        raise ValueError(f"a porch event is a courier, walker, visitor, cat, dog, car, owner_pickup or light, not {kind!r}")


def _parcel(draw: ImageDraw.ImageDraw, box=None) -> None:
    """The parcel on the step: its front face in the needle's shade with a strip of tape and a label, its top face lighter. `box` is
    where (BOX by default; a `ValueError` when it is not one of the places of BOX_AREA)."""
    if box is not None:
        bx0, by0, bx1, by1 = box
        if not (BOX_X0[0] <= bx0 <= BOX_X0[1] and BOX_Y1[0] <= by1 <= BOX_Y1[1] and abs(bx1 - bx0 - BOX_W) < 1e-6 and abs(by1 - by0 - BOX_H) < 1e-6):
            raise ValueError(f"a parcel's box {list(box)} is not one of the places of BOX_AREA: left edge in {BOX_X0}, bottom edge in {BOX_Y1}, {BOX_W}x{BOX_H}")
    x0, y0, x1, y1 = _px(PORCH, *(box or BOX))
    draw.polygon([(x0, y0), (x0 + 12, y0 - TOP_STRIP), (x1 + 12, y0 - TOP_STRIP), (x1, y0)], fill=(212, 170, 120))
    draw.rectangle([x0, y0, x1, y1], fill=NEEDLE_RGB["parcel"], outline=(150, 108, 62), width=2)
    mid = (x0 + x1) // 2
    draw.rectangle([mid - 7, y0 - TOP_STRIP + 2, mid + 7, y1], fill=(232, 214, 168))
    draw.rectangle([x0 + 14, y1 - 42, x0 + 62, y1 - 12], fill=(246, 246, 246))
    draw.line([x0 + 20, y1 - 32, x0 + 56, y1 - 32], fill=(60, 60, 60), width=2)
    draw.line([x0 + 20, y1 - 22, x0 + 46, y1 - 22], fill=(60, 60, 60), width=2)


# -- sound ------------------------------------------------------------------------------------------------------------

def _times(seconds: float) -> np.ndarray:
    return np.arange(int(round(seconds * RATE))) / RATE


def _ramp(n: int, edge_s: float = 0.01) -> np.ndarray:
    """An envelope of 1 with `edge_s` ramps in and out (so a tone does not click)."""
    edge = min(int(edge_s * RATE), n // 2)
    env = np.ones(n)
    if edge:
        env[:edge] = np.linspace(0, 1, edge)
        env[n - edge:] = np.linspace(1, 0, edge)
    return env


def _partials(hz: float, t: np.ndarray, amps: tuple[float, ...]) -> np.ndarray:
    """A tone of `hz` with partials (the k-th at k * hz with amplitude `amps[k - 1]`), the ones below the Nyquist rate only."""
    out = np.zeros(len(t))
    for k, amp in enumerate(amps, 1):
        if k * hz < RATE * 0.45:
            out += amp * np.sin(2 * np.pi * k * hz * t)
    return out


def _chime(notes, dur: float) -> np.ndarray:
    """A doorbell: its first note, then its second halfway through `dur`, each decaying in 0.35 s with two partials. The first is
    louder (amplitude 0.6, the second 0.45)."""
    out = np.zeros(int(round(dur * RATE)))
    for k, (hz, amp) in enumerate(zip(notes, (0.6, 0.45))):
        start = int(round(k * dur / 2 * RATE))
        t = _times(dur - k * dur / 2)
        piece = amp * np.exp(-t / 0.35) * _ramp(len(t), 0.004) * _partials(hz, t, (1.0, 0.3, 0.1))
        out[start:start + len(piece)] += piece[:len(out) - start]
    return out


def _beeps(hz: float, dur: float, on: float, off: float, amps: tuple[float, ...], per: int = 0, pause: float = 0.0) -> np.ndarray:
    """Beeps of `on` seconds `off` apart through `dur`, in groups of `per` with a `pause` after each (no groups when 0)."""
    out = np.zeros(int(round(dur * RATE)))
    at, count = 0.0, 0
    while at + on <= dur + 1e-9:
        t = _times(on)
        i = int(round(at * RATE))
        out[i:i + len(t)] += (_ramp(len(t)) * _partials(hz, t, amps))[:len(out) - i]
        at += on + off
        count += 1
        if per and count % per == 0:
            at += pause
    return out


def pick_hz(kind: str, seed: int) -> float:
    """A pitch for a sound that names none: from its kind's own (`BEEPER_HZ`, `TIMER_HZ`, `DING_HZ`, `RING_HZ`), which are all at least
    PITCH_GAP from every chime note and from the whole of ALARM_HZ."""
    pool = {**BEEPER_HZ, "timer": TIMER_HZ, "ding": DING_HZ, "ring": RING_HZ}[kind]
    return float(_rng(seed, f"hz-{kind}").choice(pool))


def sound(spec: dict) -> np.ndarray:
    """The samples (float32, mono, at RATE) of one sound of a microphone segment, `spec["dur"]` seconds long, times `spec["gain"]`
    (default 1: a louder or quieter copy of the same sound; a `ValueError` when the result would peak above MAX_PEAK, since the
    segment's file would clip it). The needles: `doorbell` (two notes, `notes`, default the first of DOORBELL_PAIRS; louder than 0.45)
    and `alarm` (three beeps of 0.5 s, 0.5 s apart, repeated every 4.5 s, at `hz`, default 3200, with a second partial; amplitude
    0.45). The distractors: `microwave`, `oven`, `watch` and `truck` (a pure beeper, its pitch from BEEPER_HZ unless `hz` says),
    `timer` (a kitchen timer, 0.4 s on and 0.3 s off, TIMER_HZ), `ding` (a notification, DING_HZ), `ring` (a phone ring, RING_HZ and
    480/440 of it), `kettle`, `slam`, `steps`, `vacuum` and `bark`. A beeper or a timer with a `gain` of 2 is louder than either
    needle, at its own pitch: a sort by loudness in the needle's band does not put the needle first."""
    out = _sound(spec)
    gain = float(spec.get("gain", 1.0))
    if gain == 1.0:
        return out
    if not gain > 0:
        raise ValueError(f"a sound's gain is above 0, not {gain:g}")
    out = (out * np.float32(gain)).astype(np.float32)
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > MAX_PEAK:
        raise ValueError(f"a {spec['kind']} at gain {gain:g} peaks at {peak:.2f}, over {MAX_PEAK}: it would clip")
    return out


def _sound(spec: dict) -> np.ndarray:
    kind, dur = spec["kind"], float(spec["dur"])
    seed = spec.get("seed", 0)
    rng = np.random.default_rng(int.from_bytes(f"{seed}-{kind}".encode(), "little") % (1 << 63))
    t = _times(dur)
    n = len(t)
    if kind == "doorbell":
        return _chime(spec.get("notes") or DOORBELL_PAIRS[0], dur).astype(np.float32)
    if kind == "alarm":
        return _beeps(float(spec.get("hz", 3200)), dur, ALARM_ON_S, ALARM_OFF_S, (0.45, 0.18), per=3, pause=1.5).astype(np.float32)
    if kind in BEEPER_HZ:
        hz = float(spec.get("hz") or pick_hz(kind, seed))
        on, off = {"microwave": (0.25, 0.2), "oven": (0.9, 0.5), "watch": (0.1, 0.15), "truck": (0.4, 0.4)}[kind]
        return _beeps(hz, dur, on, off, (0.35,)).astype(np.float32)
    if kind == "timer":
        return _beeps(float(spec.get("hz") or pick_hz("timer", seed)), dur, 0.4, 0.3, (0.35,)).astype(np.float32)
    if kind == "ding":
        hz = float(spec.get("hz") or pick_hz("ding", seed))
        return (0.4 * np.exp(-t / 0.18) * _ramp(n, 0.004) * _partials(hz, t, (1.0, 0.25))).astype(np.float32)
    if kind == "ring":
        hz = float(spec.get("hz") or pick_hz("ring", seed))
        burst = 0.5 + 0.5 * np.sin(2 * np.pi * 20 * t)
        gate = ((t % 1.2) < 0.4).astype(float) * _ramp(n, 0.01)
        return (0.25 * gate * burst * (np.sin(2 * np.pi * hz * t) + np.sin(2 * np.pi * hz * 480 / 440 * t))).astype(np.float32)
    if kind == "kettle":  # a whistle that rises from 2,000 to 2,400 Hz, over hiss
        phase = 2 * np.pi * np.cumsum(2000 + 400 * t / dur) / RATE
        return (0.3 * np.sin(phase) * np.minimum(1, np.minimum(t, dur - t) / 0.05) + 0.02 * rng.standard_normal(n)).astype(np.float32)
    if kind == "slam":
        return (rng.uniform(-1, 1, n) * 0.6 * np.exp(-t / 0.04)).astype(np.float32)
    if kind == "steps":  # a low thump every half second
        out = np.zeros(n)
        for k in range(int(dur / 0.5)):
            i = int(k * 0.5 * RATE)
            burst = np.convolve(rng.standard_normal(int(0.08 * RATE)), np.ones(12) / 12, "same") * np.exp(-_times(0.08) / 0.025)
            out[i:i + len(burst)] += 0.5 * burst[:n - i]
        return out.astype(np.float32)
    if kind == "vacuum":  # a roar of smoothed noise, with a hum at 120 Hz
        noise = np.convolve(rng.standard_normal(n), np.ones(3) / 3, "same")
        return (0.06 * noise * _ramp(n, 0.3) + 0.03 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    if kind == "bark":  # two or three barks of 0.18 s around 450 Hz, with many partials and noise
        out = np.zeros(n)
        for k in range(min(3, 2 + seed % 2)):
            i = int(k * 0.35 * RATE)
            bt = _times(0.18)
            f0 = 450 - 80 * bt / 0.18
            tone_ = sum(a * np.sin(2 * np.pi * np.cumsum(f0 * m) / RATE) for m, a in zip(range(1, 7), (1, 0.7, 0.5, 0.4, 0.3, 0.2)))
            part = 0.2 * np.exp(-bt / 0.08) * (tone_ + 0.5 * rng.standard_normal(len(bt))) * _ramp(len(bt), 0.005)
            out[i:i + len(bt)] += part[:max(0, n - i)]
        return out.astype(np.float32)
    raise ValueError(f"a sound is a doorbell, alarm, microwave, oven, watch, truck, timer, ding, ring, kettle, slam, steps, vacuum or bark, not {kind!r}")


def _tv(seconds: float, seed) -> np.ndarray:
    """A TV stretch: speech-like noise (about 4 kHz wide, a syllable every 0.25 s) at about -32 dBFS, and every 40 s or so a
    jingle of three notes from DING_HZ."""
    rng = np.random.default_rng(int.from_bytes(f"{seed}-tv".encode(), "little") % (1 << 63))
    n = int(round(seconds * RATE))
    half = rng.standard_normal(n // 2 + 2, dtype=np.float32)
    out = np.empty(n, np.float32)  # twice the length, a sample between each two: lowpassed to about 4 kHz
    out[0::2] = half[:(n + 1) // 2]
    out[1::2] = (half[:n // 2] + half[1:n // 2 + 1]) / 2
    k = np.arange(n // 40 + 1) / (RATE / 40)  # the syllables, as an envelope of 400 steps a second
    out *= np.repeat((0.045 * (0.35 + 0.65 * np.abs(np.sin(np.pi * 4 * k + 0.8 * np.sin(2 * np.pi * 0.3 * k))))).astype(np.float32), 40)[:n]
    at = rng.uniform(8, 25)
    while at + 1.5 < seconds:
        for j, hz in enumerate(rng.choice(DING_HZ, 3)):
            i, jt = int((at + 0.3 * j) * RATE), _times(0.3)
            out[i:i + len(jt)] += (0.12 * np.exp(-jt / 0.15) * _ramp(len(jt), 0.004) * _partials(float(hz), jt, (1.0, 0.2)))[:max(0, n - i)]
        at += rng.uniform(30, 50)
    return out


def check_pitches(sounds) -> None:
    """The sounds of one segment keep their pitches apart: a ding, a ring, a timer and a beeper are at least PITCH_GAP from the
    chime's notes, a ring's upper tone included, and from the alarm's `hz`, whichever the segment holds. (A `ValueError` names the
    pair.)"""
    needles = [(f"the doorbell's note {hz}", hz) for s in sounds if s["kind"] == "doorbell" for hz in (s.get("notes") or DOORBELL_PAIRS[0])]
    needles += [("the alarm's pitch", float(s.get("hz", 3200))) for s in sounds if s["kind"] == "alarm"]
    for s in sounds:
        if s["kind"] in (*BEEPER_HZ, "timer", "ding", "ring"):
            hz = float(s.get("hz") or pick_hz(s["kind"], s.get("seed", 0)))
            for what, needle in needles:
                if min(gap(hz * m, needle) for m in ((1, 480 / 440) if s["kind"] == "ring" else (1,))) < PITCH_GAP:
                    raise ValueError(f"a {s['kind']} at {hz:g} Hz (at {s['at']:g} s) is within {PITCH_GAP:.0%} of {what} ({needle:g} Hz)")


def mic_samples(spec: dict) -> np.ndarray:
    """A microphone segment: a bed of white noise at `bed_dbfs` (-50 by default), each `tv` stretch, then each of `sounds` at its
    `at`, cut where the segment ends. float32 in about -1..1, `length_s` long, the same for the same spec."""
    check_pitches(spec["sounds"])
    n = int(round(spec["length_s"] * RATE))
    rng = np.random.default_rng(int.from_bytes(f"{spec['seed']}-bed".encode(), "little") % (1 << 63))
    out = rng.standard_normal(n, dtype=np.float32)
    out *= np.float32(10 ** (spec.get("bed_dbfs", BED_DBFS) / 20))
    for a, b in spec.get("tv", ()):
        i = int(round(a * RATE))
        piece = _tv(min(b, spec["length_s"]) - a, f"{spec['seed']}-{a:g}")
        out[i:i + len(piece)] += piece[:n - i]
    for s in spec["sounds"]:
        i = int(round(s["at"] * RATE))
        piece = sound(s)
        out[i:i + len(piece)] += piece[:max(0, n - i)]
    return out


# -- files ------------------------------------------------------------------------------------------------------------

def _finish(part: Path, path: Path) -> None:
    """Put a finished file from the working folder in its place, so a data folder never holds half a file. It is copied, not
    moved: on Windows a moved file keeps the working folder's owner-only permissions (Python makes temporary folders so), and
    Codex's sandbox then cannot open it, while a new file takes the permissions of the folder it is made in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    shutil.copyfile(part, temp)
    os.replace(temp, path)
    part.unlink()


def _write_video(path: Path, frames, length_s: float, size: tuple[int, int], crf: int, work: Path) -> None:
    """`frames`, `(second, image)` in order, as x264 at `crf` (preset veryfast, tune zerolatency, one thread), each at its own
    time. The container is the path's (`.mkv` or `.mp4`). A frame says it lasts 40 ms (as capture's own files do: ffmpeg's demuxer
    reads a frame's decode time from the frame rate and the duration, and a long duration on a 25 fps stream shifts the stream's
    start, and with it every time the reader computes), and the last lasts until `length_s`, which is the file's duration."""
    Path(work).mkdir(parents=True, exist_ok=True)
    part = Path(work) / Path(path).name
    packets = []
    with av.open(str(part), "w", options={"fflags": "+bitexact"}) as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        stream.codec_context.options = {"crf": str(crf), "preset": "veryfast", "tune": "zerolatency", "threads": "1"}
        for second, image in frames:
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = round(second * 1000), Fraction(1, 1000)
            packets += stream.encode(frame)
        packets += stream.encode(None)
        for i, packet in enumerate(packets):
            packet.duration = 40 if i + 1 < len(packets) else max(round(length_s * 1000) - packet.pts, 40)
            container.mux(packet)
    _finish(part, Path(path))


def write_screen(path: Path, spec: dict, start: float, work: Path) -> None:
    """A screen segment that starts at `start` as a VFR video: a frame at each time of `screen_frames`, drawn from the state's layers (made
    once for the state) with the frame's scroll, typing, clock and cursor."""
    def frames():
        k, layers = -1, None
        for frame in screen_frames(spec, start):
            if frame.state != k:
                k, layers = frame.state, screen_layers(spec["states"][frame.state])
            yield frame.second, screen_frame(layers, frame)

    _write_video(path, frames(), spec["length_s"], SCREEN, SCREEN_CRF, work)


def _stamp_text(stamp: str, second: float) -> str:
    """The camera's time at `second` of a clip that starts at `stamp` (`2026-10-28 15:35:44`; any other text is kept)."""
    try:
        return (datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=second)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stamp


def write_porch(path: Path, spec: dict, stamp: str, work: Path) -> None:
    """A porch clip, one frame a second, its stamp running from `stamp`."""
    frames = ((s, porch_frame(spec, s, _stamp_text(stamp, s))) for s in range(int(spec["length_s"])))
    _write_video(path, frames, spec["length_s"], PORCH, PORCH_CRF, work)


def write_mic(path: Path, spec: dict, work: Path) -> None:
    """A mic segment as opus at 24 kbit/s, its samples piped to ffmpeg as 16-bit PCM (never a WAV on disk). `compression_level 0` and
    60 ms frames make the encode about 6 times faster than ffmpeg's defaults, at the size and the sound the reader needs."""
    samples = mic_samples(spec)
    np.clip(samples, -1, 1, out=samples)
    samples *= 32767
    Path(work).mkdir(parents=True, exist_ok=True)
    part = Path(work) / Path(path).name
    done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0", "-c:a", "libopus",
                           "-b:a", "24k", "-compression_level", "0", "-frame_duration", "60", "-fflags", "+bitexact", str(part)],
                          input=samples.astype("<i2").tobytes(), capture_output=True)
    if done.returncode:
        raise RuntimeError(f"ffmpeg could not encode {Path(path).name}: {done.stderr.decode(errors='replace').strip()[-400:]}")
    _finish(part, Path(path))


_check_constants()
