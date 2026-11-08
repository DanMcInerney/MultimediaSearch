"""A synthetic week of an owner's phone, screen, front door camera, hallway microphone, notes, config and logs (TESTBED.md,
benchmark): the ground truth of the mms benchmark. Needles are things a question asks for (a code in a note, a sign's number,
an error on screen, a doorbell), hidden among distractors that must never match a needle's question. A needle is told apart by
meaning (an error dialog among red banners and a red "Delete?" dialog, a parcel among a brown dog and the courier's own box, a
smoke alarm among beepers and chimes) and, for the fake decider of the offline tests, by an exact shade or pitch that is its alone
(`benchmarks/haystack_media.py` draws and plays everything).

    plan = haystack.plan(seed, "tiny", haystack.days_before(date.today(), 2))   # pure: no files, no clock
    haystack.write(plan, folder)         # the files (PIL, numpy, PyAV, ffmpeg), each checked as it is made; never the truth
    haystack.load(api, folder, plan)     # remembers each file (the phone's at its own time), adds the log source, waits for its records
    haystack.questions_for(plan.questions, about)   # the questions a model reads (TESTBED.md: no document or speech ones)
    haystack.windows_in(plan, "screen", since, until)   # the reader windows a source holds in a range (the decider's estimate)

`plan` is a function of (seed, scale, days): the same inputs give the same files, lines and questions, and nothing is
written until `write`. It checks itself (`verify`): the scope of every question holds what its level says, counted from the
files' specs and not from the questions. The truth stays in the plan's questions and is never on disk. A question is a plain dict:
- `qid` (`image-photo-sign-1`: type, template, n), `type`, `template`, `absent` and `source` (the mms source, and the folder);
- `level`: how vague its time hint is, `part` (a part of a day: `PARTS`), `day`, `week` (no hint: the prompt gives the span, and the
  question asks for the last one, or, for a count, all of them) or `time` (an exact time: `config-asof`); `last`: a `week` question
  that asks for the last needle (a count asks for all of them);
- `since` and `until`, the range it names (local epoch), `hint` (the day and part of the day it words, with hours; "" for `week`)
  and `question`; an absent question has its template's exact words, so an agent cannot tell it from a needle question;
- `truth`: `{"value", "moments", "others"}`, the answer, the spans `[t0, t1]` it is on record in (None and none when absent) and,
  for a `week` question, the spans of the template's other needles (all earlier: nothing of the template comes after the last);
- `grade`: what `benchmarks/grade.py` needs (`kind`, `truth`, `time`, `tol_min`, `wrong`, `ignore`, `date`: the day a `week` question's
  time must be on); the earlier needles are a `week` question's `wrong` answers;
- `oracle`: a yes/no for a real model over the source and range, or None; `fake`: the question the fake decider is asked
  (`question`, `oldest_first`, `limit`) and what its sightings must show (`check`: `overlap` each moment is found and every sighting
  is at one; `none` no sighting in the range; `count` the answer's number of them; `look` one version whose text holds the value;
  `last` the newest sighting is at the last needle and every one is at some needle);
- `probe`: what the shortcut probe (`benchmarks/shortcuts.py`) measures: `{"kind": "shade", "hue", "rgb"}`, `{"kind": "pitch", "hz",
  "band"}` or None (text and records).

| Template      | Source (type)   | The needle                                          | The question asks                         |
| note-code     | notes (text)    | a note with a 4- or 6-digit code                    | the code, for a thing noted in a slot     |
| config-asof   | config (text)   | the version of app.toml current at a time           | retention_days at that time               |
| log-count     | logs (records)  | ERR-7xx-9xx k times in a slot, and on other days    | how many times payments logged it         |
| log-last      | logs (records)  | web deploys in a slot, started and finished         | when the last one finished                |
| photo-sign    | phone (image)   | a green sign (22-34% of the photo): word, number    | the sign's number                         |
| photo-count   | phone (image)   | k photos of a red car (30% or more) on one day      | how many (on a day, or all week)          |
| screen-error  | screen (video)  | a dialog, red header "ERROR E-xxxx", 20-40 s        | its code, and when it first appeared      |
| porch-parcel  | porch (video)   | a cardboard box left on the step by a courier       | when it was left                          |
| mic-doorbell  | mic (audio)     | a two-note chime (the house's pair)                 | when it rang                              |
| mic-alarm     | mic (audio)     | 3 beeps of 0.5 s at the house's alarm pitch         | when the smoke alarm beeped               |
| absent-*      | each type       | none: the same words, about what is not there       | the answer is "not there"                 |

Three scales (`SCALES`; a day is what `Scale` says, 6.5 h of screen and 6 h of microphone at the harder ones):
- `tiny`: 2 days, one 20-minute screen block a day, 6 porch clips, 12 photos and 30 minutes of microphone a day, 19 questions, one
  needle for a template (its part question and its week question ask about it; day 2 holds none). The offline test and the smoke.
- `pilot`: 3 days at the density below, 31 questions. For the four templates with a week question (sign, error, parcel, alarm) the
  last needle is on day 2, in the first part of it that has data, and the part and day questions ask about one needle on day 1: a
  week of 3 days has no room for two before the last one, and a late last needle would leave less than 2 days to scan.
- `standard`: 7 days, 31 questions (24 for a model that does not hear). Per day: about 790 screen states and 30,000 frames (VFR segments cut as
  the recorder cuts them, each ending at the first state at or after each minute from its block's start: 60 s on average, 10 to about 120 s,
  390 files; 09:00-11:30, 13:30-16:30, 20:00-21:00, a new state every 10 to 60 s, and a frame, at most one every half second, for each state,
  each act of the owner (a cursor move, typing, a scroll) and each minute of the taskbar's clock: `screen_frames`; the reader reads about 16,000
  of those seconds), 30 porch clips of 10 to 30 s (about 480 frames), 60 photos
  and 3 clips (each carries its own time, EXIF or `creation_time`, and is remembered without one), 6 h of microphone (07:00-09:00, 17:00-21:00) as 15 minute segments with about 25 sounds and 30 minutes of TV, and
  the controls unchanged (15 notes, 2 versions of app.toml, 2,000 log lines).
Distractors are planted in every part of a day before the random ones (`Scale.mix`), so each scope a question names holds some of every
class: red dialogs, banners and toasts among the screen's states (a banner is bigger than the error's header), green and red things,
signs and cars of other colours and facades and hedges bigger than a sign or a car among the photos, a brown dog, a light change and
the courier's own box and a visitor in a brown coat close to the camera among the porch's clips, beepers, timers, dings, rings and barks among the microphone's sounds (a beeper and a timer in each part are louder than the alarm and the chime). Dialogs (the error's and
the others', red ones among them) open in different places of the middle of the screen, a parcel is left in different places of the step (the same
in every clip that shows it), and the distractors of the error's red and of the parcel's brown are in those places too (`DIALOG_AREA`, `BOX_AREA`
of `haystack_media`): a crop of one place keeping red or brown does not find the needle first. A needle's photo has no
fence behind it, and a clip that shows the parcel has no light or car, so that no needle's file is the biggest of its scope by luck.
A part question names a part with enough to look through (`Scale.part_floor`: 250 screen states, 8 porch clips, 15 photos, 5 sounds), so
no question names the evening's screen block (an hour) or the microphone's one hour of the afternoon.
Times are local, made with `datetime(y, m, d, h, mi, s)` and never by adding seconds across days, and no event is before 06:00
(daylight saving changes happen at night). `write` asserts its own checks (a needle's shade or pitch is the fake decider's after
encoding, no distractor has it, a needle's code is in its own file only), so a wrong fixture fails here and never in the reader."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import grade
import haystack_media as hm
import numpy as np
from haystack_media import DISTRACTOR_RGB, NEEDLE_RGB
from PIL import ExifTags, Image, ImageDraw

from mms import reader

# day, fake_decider and (in `load`) conftest are the tests' own, and a script run from benchmarks/ has only benchmarks/ on its path
TESTS_DIR = str(Path(__file__).resolve().parents[1] / "tests")
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)  # last, so no file of the tests' shadows a module of the standard library or the env

from day import RATE, font, portrait_video  # noqa: E402
from fake_decider import PITCH_RMS, PITCH_SHARE, pitch_share, shade_share  # noqa: E402

VERSION = 6  # bump when plan() changes what it makes (6: the screen's acts, TESTBED.md)
TYPES = ("text", "records", "image", "video", "audio")
TEMPLATES = {"text": ("note-code", "config-asof"), "records": ("log-count", "log-last"),
             "image": ("photo-sign", "photo-count"), "video": ("screen-error", "porch-parcel"),
             "audio": ("mic-doorbell", "mic-alarm")}
ABSENT = {"text": ("absent-note",), "records": ("absent-log",), "image": ("absent-sign",),
          "video": ("absent-error", "absent-parcel"), "audio": ("absent-doorbell",)}
ABSENT_OF = {"note-code": "absent-note", "log-count": "absent-log", "photo-sign": "absent-sign", "screen-error": "absent-error",
             "porch-parcel": "absent-parcel", "mic-doorbell": "absent-doorbell"}
BASE_OF = {absent: template for template, absent in ABSENT_OF.items()}  # the template an absent one asks in the words of
INFO = {"note-code": ("text", "notes"), "config-asof": ("text", "config"), "log-count": ("records", "logs"),
        "log-last": ("records", "logs"), "photo-sign": ("image", "phone"), "photo-count": ("image", "phone"),
        "screen-error": ("video", "screen"), "porch-parcel": ("video", "porch"), "mic-doorbell": ("audio", "mic"),
        "mic-alarm": ("audio", "mic")}
LEVELS = ("part", "day", "week", "time")
PARTS = {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 24)}
SOURCES = {"screen": "my screen recordings", "phone": "photos and short videos from my phone",
           "porch": "the front door camera's motion clips", "mic": "the hallway microphone", "notes": "my notes",
           "config": "the versions of app.toml", "logs": "the payments, web and auth services' logs"}
LOG_SOURCE = "logs"
PHONE = "phone"  # the source whose files carry their own time
SERVICES = ("payments", "web", "auth")
TOL_S = 10.0  # a sighting overlaps a moment when it comes within this many seconds of it
MIC_GAP_S = 75  # a room's sound is this far from a needle's: the answer's tolerance is a minute, so it must not be mistaken for it
WORKERS = 3  # files written at once: ffmpeg and x264 run outside the interpreter for much of each
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
DAY_START, DAY_END = 7 * 3600, 23 * 3600  # photos, clips, notes and porch clips are at these times of day
NEED_SHARE, OTHER_SHARE, MOST_SHARE = 0.03, 0.005, 0.5  # a needle's share of its shade after encoding; any other picture's; and no more than
MEAN_STATE_S = 30.0  # a screen state lasts 10 to 60 s, 30 on average
BANDS = {"alarm": [2000, 5000], "doorbell": [400, 1000]}  # what a band filter for the sound's class keeps (the probe's `band`)
HOUR = 3600
FENCE_RGB = (150, 142, 130)  # the colour of a fence's posts in a photo's backdrop (haystack_media draws them at the top row of the photo)


@dataclass(frozen=True)
class Scale:
    """What a week holds: per day unless said. `asks` is each template's levels, `(needle levels, absent levels)`; `extra` the
    earlier needles of a template beyond the ones its questions need; `shared`: None, `last` (the part question and the week
    question ask about one needle: tiny has one needle for a template) or `day` (the part and day questions of a template that
    has a week question ask about one, so that its last needle can be early on a short week); `early`: the last needle is in the
    first part of its day that has data. `screen` and `mic` are the blocks of each day (`(start_s, end_s)`; the day after the
    last has the first's again), recorded as segments: the screen's of about `screen_s` (each ends at the first state change at or
    after a multiple of `screen_s` from its block's start: the recorder cuts at the first key frame at or after each `screen_s`, and
    here key frames are at a file's start and at scene cuts, which are the state changes; the recorder also forces one every 10 s, so
    its cuts land mid-state: not modelled), the microphone's of 900 s. `mix` is what every part of a day (`slot`) or day
    (`day`) holds of each source's distractors (planted before the random ones), `part_floor` the least candidates (states, clips,
    photos, sounds) a part of a day needs for a question to name it."""

    days: int
    asks: dict
    extra: int
    shared: str | None
    early: bool
    photos: int
    clips: int
    cars: tuple[int, int]
    other_cars: tuple[int, int]
    other_min: int
    screen: tuple
    screen_s: int
    porch: int
    mic: tuple
    mic_extra: float
    tv_s: int
    notes: int
    config: int
    log_lines: int
    mix: dict
    part_floor: dict


TINY_ASKS = {"note-code": (("part",), ("part",)), "config-asof": (("time",), ()), "log-count": (("part",), ("day",)),
             "log-last": (("part",), ()), "photo-sign": (("part", "week"), ("part",)), "photo-count": (("day",), ()),
             "screen-error": (("part", "week"), ("part",)), "porch-parcel": (("part", "week"), ()),
             "mic-doorbell": (("part",), ("part",)), "mic-alarm": (("part", "week"), ())}
FULL_ASKS = {"note-code": (("part", "day"), ("part",)), "config-asof": (("time",), ()),
             "log-count": (("part", "day"), ("day",)), "log-last": (("part",), ()),
             "photo-sign": (("part", "day", "week"), ("part", "day")), "photo-count": (("day", "week"), ()),
             "screen-error": (("part", "day", "week"), ("part", "day")), "porch-parcel": (("part", "day", "week"), ("day",)),
             "mic-doorbell": (("part", "day"), ("part", "day")), "mic-alarm": (("part", "day", "week"), ())}
TINY_MIX = {"screen": {"slot": {"banner": 1, "accent": 2, "confirm": 1, "alert": 1, "warning": 1, "info": 1, "save": 1}, "day": {}},
            "phone": {"slot": {"greenwall": 1, "sign": 1}, "day": {"redwall": 1, "car": 1}},
            "porch": {"slot": {"dog": 1, "light": 1, "visitor": 1}, "day": {"car": 1}},
            "mic": {"slot": {"beeper": 1, "band": 1, "loud": 1}, "day": {}}}
FULL_MIX = {"screen": {"slot": {"banner": 1, "accent": 4, "confirm": 3, "alert": 3, "warning": 1, "info": 1, "save": 1}, "day": {}},
            "phone": {"slot": {"greenwall": 3, "green": 3, "redwall": 5, "red": 2, "sign": 2, "car": 2}, "day": {}},
            "porch": {"slot": {"dog": 2, "light": 2, "car": 2, "courier": 1, "visitor": 2}, "day": {}},
            "mic": {"slot": {"beeper": 3, "band": 3, "loud": 1}, "day": {}}}
BLOCKS = ((9 * HOUR, 9 * HOUR + 9000), (13 * HOUR + 1800, 16 * HOUR + 1800), (20 * HOUR, 21 * HOUR))  # 09:00-11:30, 13:30-16:30, 20:00-21:00
MIC_BLOCKS = ((7 * HOUR, 9 * HOUR), (17 * HOUR, 21 * HOUR))
FULL = dict(asks=FULL_ASKS, photos=60, clips=3, screen=(BLOCKS,), screen_s=60, porch=30, mic=(MIC_BLOCKS,), mic_extra=0.25,
            tv_s=1800, notes=15, config=2, log_lines=2000, mix=FULL_MIX, cars=(2, 4), other_cars=(0, 2), other_min=2,
            part_floor={"screen": 250, "porch": 8, "phone": 15, "mic": 5})
# The owner's activity on the screen, calibrated against two real hours (TESTBED.md: per hour, over 150 plan hours, 2,360 seconds with a frame
# and 1.86 frames a second with a frame; over three encoded hours, 2,329 gated windows, 10% of the minutes with 2 or fewer gated windows, 34% of
# the frame pairs changing under 0.05%, median 0.084%, 90th percentile 7.6%). A state has a level, drawn as a chain over its block
# (the next state's level depends on the last's, so idle comes in stretches of a minute or more): `idle` has no acts (reading, thinking,
# away), `light` and `busy` have runs of active seconds, each with 1 or 2 acts (on the second's two ticks).
ACTIVITY = ("idle", "light", "busy")
ACTIVITY_START = (1, 2, 6)  # the odds of a block's first level
ACTIVITY_CHAIN = {"idle": (89, 4, 7), "light": (4, 50, 46), "busy": (18, 130, 852)}  # the odds of the next level, by the level before it
ACTIVE_SHARE = {"light": (0.2, 0.55), "busy": (0.72, 0.92)}  # the share of a state's seconds that are active, drawn for each state
ACTIVE_RUN_S = {"light": (2, 6), "busy": (6, 20)}  # ... in runs of this many seconds
TWO_ACTS = {"light": 0.7, "busy": 0.92}  # the odds that an active second has 2 acts and not 1
ACT_KINDS = ("move", "type", "scroll")
KIND_ODDS = {"editor": (1, 8, 1.5), "chat": (1, 9, 0.3), "slides": (2, 8, 1), "sheet": (2, 8, 1), "terminal": (1, 6, 3), "browser": (5, 1, 3)}  # by app
TYPE_CHARS = ((1, 2, 3), (3, 4, 3))  # a `type` adds this many characters, with these odds
SCROLL_LINES = ((1, 2, 3), (5, 3, 2))  # a `scroll` that is not an enter moves this many lines, with these odds
MOVE_PX = (15, 140, 90)  # a cursor move: at least this far, at most this far, and a spread of this much (px)
SCALES = {"tiny": Scale(days=2, asks=TINY_ASKS, extra=0, shared="last", early=False, photos=12, clips=1,
                        cars=(2, 2), other_cars=(1, 1), other_min=1,
                        screen=(((14 * HOUR, 14 * HOUR + 1200),), ((10 * HOUR, 10 * HOUR + 1200),)), screen_s=60, porch=6,
                        mic=(((7 * HOUR, 7 * HOUR + 900), (17 * HOUR, 17 * HOUR + 900)),), mic_extra=0.5, tv_s=240,
                        notes=4, config=1, log_lines=150, mix=TINY_MIX,
                        part_floor={"screen": 1, "porch": 1, "phone": 1, "mic": 1}),
          "pilot": Scale(days=3, extra=0, shared="day", early=True, **FULL),
          "standard": Scale(days=7, extra=1, shared=None, early=False, **FULL)}


@dataclass(frozen=True)
class File:
    path: str  # relative to the data folder, forward slashes; its first part is the source
    source: str
    t: float  # local epoch: when it was taken or starts
    name: str | None  # remember's name: "app.toml" for config versions, else None (the file name)
    kind: str  # "photo" | "clip" | "screen" | "porch" | "mic" | "note" | "config"
    spec: dict  # what to draw or play (kind-specific, JSON-able; the writer's business)


@dataclass(frozen=True)
class Plan:
    seed: int
    scale: str
    days: tuple[date, ...]
    files: tuple[File, ...]  # oldest first
    logs: dict[str, tuple[dict, ...]]  # service -> its lines ({"ts", "service", "level", "msg", "code"?}), in time order
    questions: tuple[dict, ...]  # in TYPES then template order


def days_before(today: date, n: int) -> list[date]:
    """The `n` days ending yesterday, oldest first."""
    return [today - timedelta(days=n - i) for i in range(n)]


def at(day: date, seconds: int) -> float:
    """The local epoch of `seconds` into `day` (86,400 is the next midnight), made from the clock and not by adding seconds."""
    if seconds >= 86400:
        day, seconds = day + timedelta(days=1), seconds - 86400
    return datetime(day.year, day.month, day.day, seconds // 3600, seconds // 60 % 60, seconds % 60).timestamp()


def clock(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def date_text(day: date) -> str:
    """`Tuesday 28 October`, from fixed English names (not `strftime`, which follows the locale)."""
    return f"{WEEKDAYS[day.weekday()]} {day.day} {MONTHS[day.month - 1]}"


def part_of(minute: int) -> str:
    return next(part for part, (lo, hi) in PARTS.items() if lo * 60 <= minute < hi * 60)


def slot_key(slot: tuple[int, str]) -> tuple[int, int]:
    """Where a (day, part) slot is in time."""
    return slot[0], list(PARTS).index(slot[1])


def clip_range(part: str) -> tuple[int, int]:
    """The seconds of the day a photo, clip or note of `part` can be at: the part, inside DAY_START and DAY_END."""
    lo, hi = PARTS[part]
    return max(lo * HOUR, DAY_START), min(hi * HOUR, DAY_END)


PART_WEIGHTS = tuple(clip_range(part)[1] - clip_range(part)[0] for part in PARTS)


def share_out(total: int, weights: Sequence[float]) -> list[int]:
    """`total` split in proportion to `weights` in whole numbers (the largest remainders get the last ones)."""
    raw = [total * w / sum(weights) for w in weights]
    out = [int(x) for x in raw]
    for k in sorted(range(len(raw)), key=lambda k: (-(raw[k] - out[k]), k))[:total - sum(out)]:
        out[k] += 1
    return out


def spaced(r: random.Random, lo: int, hi: int, n: int, gap: int, avoid: Sequence[int] = (), what: str = "") -> list[int]:
    """`n` whole numbers in [lo, hi), each at least `gap` from the others and from `avoid`, in order."""
    out: list[int] = []
    for _ in range(n):
        for _ in range(2000):
            x = r.randrange(lo, hi)
            if all(abs(x - y) >= gap for y in (*avoid, *out)):
                out.append(x)
                break
        else:
            raise ValueError(f"{what}: no room for {n} times {gap} apart from {lo} to {hi}")
    return sorted(out)


# -- where a template's needles go --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Spot:
    """Where a template's needles and absences are, as (day, part) slots (`absent_day` is a day): `last` the template's last
    needle (for a `week` question: nothing of the template comes after it, and a day or more of haystack does), `part` the one
    its part-level questions ask about (its part holds no other), `day` the one its day-level questions ask about (its day holds
    no other), `extras` earlier needles, `absent_part` a part and `absent_day` a day that hold none."""

    last: tuple[int, str] | None = None
    part: tuple[int, str] | None = None
    day: tuple[int, str] | None = None
    extras: tuple[tuple[int, str], ...] = ()
    absent_part: tuple[int, str] | None = None
    absent_day: int | None = None

    @property
    def needles(self) -> list[tuple[int, str]]:
        """The needles' slots, oldest first (there is one needle to a slot)."""
        return sorted({s for s in (self.last, self.part, self.day, *self.extras) if s is not None}, key=slot_key)


def last_days(n: int) -> range:
    """The day (0-based) a week's last needle is on: three full days of haystack follow it in a week of 5 or more (day 4 of 7: the
    probe's floors for a scan from the newest, 150 photos among them, want that), two in the 4 days of a short week and one in a
    shorter (day 2 of 3, day 1 of 2)."""
    return range(n - 4, n - 3) if n >= 5 else range(max(0, n - 2), max(1, n - 1))


def arrange(r: random.Random, what: str, n_days: int, levels: Sequence[str], absent: Sequence[str], avail: set, eligible: set,
            extra: int = 0, shared: str | None = None, early: bool = False) -> Spot:
    """Sample where a template's needles go so that each of its questions has the scope its level says (a `ValueError` names
    what cannot be met): a `part` question's part and a `day` question's day hold exactly one needle; a `week` question's answer
    is the last needle, on one of `last_days`, with nothing of the template after it and `extra` earlier needles besides; an
    absent question names a part (of `eligible`) or a day that holds none. `avail` are the slots the source has data in and
    `eligible` those that have enough to name in a question. `shared` and `early`: see `Scale`."""
    want_last, want_part, want_day = "week" in levels, "part" in levels, "day" in levels
    extra = extra if want_last else 0  # earlier needles only make a week question's answer the last of several
    for _ in range(5000):
        last = day_slot = part = None
        if want_last:
            day = r.choice(list(last_days(n_days)))
            options = sorted((s for s in avail if s[0] == day), key=slot_key)
            if not options:
                continue
            last = options[0] if early else r.choice(options)
        before = [s for s in avail if last is None or slot_key(s) < slot_key(last)]
        if want_day:
            days = sorted({s[0] for s in before if last is None or s[0] < last[0]})
            if not days:
                continue
            day = r.choice(days)
            options = sorted(s for s in (eligible if shared == "day" and want_last else avail) if s[0] == day)
            if not options:
                continue
            day_slot = r.choice(options)
        if want_part:
            if shared == "last" and last is not None:
                part = last if last in eligible else None
            elif shared == "day" and want_last and day_slot is not None:
                part = day_slot
            else:
                options = sorted(s for s in eligible if (last is None or slot_key(s) < slot_key(last))
                                 and (day_slot is None or s[0] != day_slot[0]))
                part = r.choice(options) if options else None
            if part is None:
                continue
        used = {s for s in (last, part, day_slot) if s is not None}
        pool = sorted(s for s in before if s not in used and (day_slot is None or s[0] != day_slot[0]))
        if len(pool) < extra:
            continue
        extras = tuple(sorted(r.sample(pool, extra), key=slot_key))
        taken = used | set(extras)
        a_part = a_day = None
        if "part" in absent:
            free = sorted(s for s in eligible if s not in taken)
            if not free:
                continue
            a_part = r.choice(free)
        if "day" in absent:
            free_days = [d for d in range(n_days) if all(s[0] != d for s in taken)]
            if not free_days:
                continue
            a_day = r.choice(free_days)
        return Spot(last, part, day_slot, extras, a_part, a_day)
    raise ValueError(f"{what}: no way to place needles for {tuple(levels)} (absent {tuple(absent)}, {extra} extra, shared {shared}, "
                     f"early {early}) in {n_days} days with data in {len(avail)} slots, {len(eligible)} of them big enough to name")


# -- the questions' words ---------------------------------------------------------------------------------------------

WORDS = {"note-code": "What is the {thing} that I noted {hint}?",
         "config-asof": "What was retention_days set to in app.toml {hint}?",
         "log-count": "How many times did the payments service log {code} {hint}?",
         "log-last": "At what time did the last deploy of the web service finish {hint}?",
         "photo-sign": "What number was on the green sign that I took a photo of {hint}?",
         "photo-count": "How many photos of a red car did I take {hint}?",
         "screen-error": "What was the error code on the red error screen {hint}, and at what time did it first appear?",
         "porch-parcel": "At what time was a parcel left at the front door {hint}?",
         "mic-doorbell": "At what time did the doorbell ring {hint}?",
         "mic-alarm": "At what time did the smoke alarm beep {hint}?"}
WEEK_WORDS = {  # no hint: the prompt names the span. A time answer must name the day too (`grade.date`)
    "photo-sign": "What number was on the last green sign that I took a photo of?",
    "photo-count": "How many photos of a red car did I take?",
    "screen-error": "What was the error code on the red error screen the last time one appeared, and on what day and at what time "
                    "did it first appear?",
    "porch-parcel": "On what day and at what time was a parcel last left at the front door?",
    "mic-alarm": "On what day and at what time did the smoke alarm last beep?"}
ORACLES = {"note-code": "Does the text mention {topic}?", "config-asof": None,
           "log-count": "Does a record have the error code {code}?",
           "log-last": "Does a record say that a deploy of the web service finished?",
           "photo-sign": "Is there a green sign in the picture?", "photo-count": "Is there a red car in the picture?",
           "screen-error": "Is an error dialog shown on the screen?", "porch-parcel": "Is there a parcel on the step by the door?",
           "mic-doorbell": "Is a doorbell ringing?", "mic-alarm": "Is a smoke alarm beeping?"}
TOPICS = (("code of the spare key box", "the spare key box", "Spare key box: it now opens with {code}.", 4),
          ("combination of my bike lock", "a bike lock", "Bike lock: the new combination is {code}.", 4),
          ("code of my gym locker", "a gym locker", "Gym locker: the code is {code} this month.", 4),
          ("PIN of the parcel locker", "a parcel locker", "Parcel locker: pick up with PIN {code}.", 6))
ERRORS = ("disk full", "out of memory", "connection lost", "license expired", "checksum mismatch", "database locked",
          "backup failed")
WARNINGS = ("low battery", "update available", "slow network", "certificate expires soon", "high temperature")
SIGN_WORDS = ("GATE", "PLATFORM", "ROOM", "BAY", "LEVEL")
OTHER_SIGNS = ("yellow", "white", "blue", "grey")
OTHER_CARS = ("blue", "white", "grey", "yellow")


def grade_of(kind: str, truth: str | None = None, time: str | None = None, tol_min: float | None = None,
             wrong: Sequence[str] = (), ignore: Sequence[str] = (), date_: str | None = None) -> dict:
    return {"kind": kind, "truth": truth, "time": time, "tol_min": tol_min, "wrong": list(wrong), "ignore": list(ignore),
            "date": date_}


def shade_of(kind: str) -> str:
    """The fake decider's question for a needle's shade (`shade:R,G,B`)."""
    return "shade:" + ",".join(str(c) for c in NEEDLE_RGB[kind])


def apart(clocks: Sequence[str], truth: str, minutes: float = 10) -> list[str]:
    """The clock times of `clocks` that are not within `minutes` of `truth`: the earlier needles of a week question that an answer
    must not give, with the true time's own neighbours left out (a right answer is within the tolerance of it)."""
    return [c for c in clocks if abs(grade.at(c) - grade.at(truth)) > minutes]


REASONS = ("gateway timeout", "card declined", "insufficient funds", "fraud check failed", "issuer unavailable")
PATHS = ("orders", "users", "items", "cart", "health", "search", "checkout")
NAMES = ("ana", "ben", "chloe", "dev", "eli", "fay", "gus", "hana", "ivan", "june")


def background_line(r: random.Random, service: str) -> tuple[str, str, str | None]:
    x, order = r.random(), r.randrange(10000, 99999)
    if service == "payments":
        if x < 0.82:
            return "info", f"payment ok order={order} amount={r.randrange(5, 500)}.{r.randrange(100):02d} EUR", None
        if x < 0.92:
            return "warning", f"card retry order={order} attempt={r.randrange(1, 4)}", None
        return "error", f"{r.choice(REASONS)} order={order}", f"ERR-{r.randrange(100, 700)}"
    path = f"/api/{r.choice(PATHS)}"
    if service == "web":
        if x < 0.88:
            return "info", f"request ok path={path} {r.choice((200, 200, 200, 304))} in {r.randrange(5, 650)} ms", None
        if x < 0.96:
            return "warning", f"slow request path={path} in {r.randrange(400, 650)} ms", None
        return "error", f"upstream error path={path} 502", f"ERR-{r.randrange(100, 700)}"
    user = r.choice(NAMES)
    if x < 0.85:
        return "info", r.choice((f"login ok user={user}", f"token refreshed user={user}")), None
    if x < 0.96:
        return "warning", f"login failed user={user} reason=bad credentials", None
    return "error", f"account locked user={user}", f"ERR-{r.randrange(100, 700)}"


GROCERIES = ("milk", "bread", "eggs", "apples", "rice", "pasta", "coffee", "tomatoes", "cheese", "oranges", "oats", "butter")
PEOPLE = ("Priya", "Marcus", "Elena", "Tom", "Sofia", "Daniel", "Aiko", "Omar")
BOOKS = (("The Left Hand of Darkness", "Le Guin"), ("Piranesi", "Clarke"), ("Station Eleven", "Mandel"),
         ("The Dispossessed", "Le Guin"), ("Klara and the Sun", "Ishiguro"))
IDEAS = ("a small script that renames photos by date", "a shelf above the desk", "a weekly review on Sunday evening",
         "moving the plants to the window", "a list of the trips for next year", "a plan for the garden beds")


def distractor_note(r: random.Random) -> str:
    """A note of the owner's life, with small numbers only (none is a needle's code) and none of a needle's topics."""
    kind = r.choice(("groceries", "meeting", "reminder", "book", "idea"))
    if kind == "groceries":
        a, b, c = r.sample(GROCERIES, 3)
        return f"Groceries: {a}, {b} and {c}. Get {r.randrange(2, 9)} of the first."
    if kind == "meeting":
        return f"Meeting with {r.choice(PEOPLE)} at {r.randrange(9, 18)}:{r.choice(('00', '15', '30', '45'))}, room {r.randrange(2, 30)}."
    if kind == "reminder":
        return f"Reminder: call {r.choice(PEOPLE)} about the {r.choice(('invoice', 'holiday', 'dentist', 'garden'))}, before {r.randrange(10, 29)}."
    if kind == "book":
        title, author = r.choice(BOOKS)
        return f"Book: {title} by {author}, now on page {r.randrange(20, 400)}."
    return f"Idea: {r.choice(IDEAS)}."


def cut(r: random.Random, lo: int, hi: int) -> list[tuple[int, int]]:
    """[lo, hi) cut into states of 10 to 60 s, 30 s on average (a tail under 10 s joins the last state)."""
    out: list[tuple[int, int]] = []
    t = lo
    while t < hi:
        d = int(r.triangular(10, 60, 20))
        if hi - (t + d) < 10:
            d = hi - t
        out.append((t, t + d))
        t += d
    return out


def segments(starts: Sequence[int], lo: int, hi: int, step: int) -> list[tuple[int, int]]:
    """The segments `(start, end)` the recorder's segment muxer makes of a block [lo, hi) whose states start at `starts` (in order, the
    first at `lo`): a segment ends only on a key frame, which here is a state's start (x264's scene cut puts one at each state change; the
    real recorder also forces one every 10 s, so its cuts land mid-state: not modelled), so the n-th segment ends where the first state that starts at or after `lo + n * step`
    begins (a long state can make one segment end where the next would). The last ends at `hi`. A segment starts on a state's start,
    so none repeats a still frame at its start."""
    cuts = [starts[0]]
    for t in starts[1:]:
        if t >= lo + step * len(cuts):
            cuts.append(t)
    return list(zip(cuts, [*cuts[1:], hi]))


def next_level(r: random.Random, last: str | None) -> str:
    """The activity level of the state after one of level `last` (None: the first of a block)."""
    return r.choices(ACTIVITY, ACTIVITY_START if last is None else ACTIVITY_CHAIN[last])[0]


def state_acts(state: dict, at: tuple[int, int], level: str) -> tuple[list[list], tuple[int, int]]:
    """The owner's acts over `state` (its `app`, `seed`, `from`, `to`, and the accent and overlay that can hide its typing line), which
    starts with the cursor at `at`, at activity `level`, from the state's own seed: none when idle; else an active share of its seconds
    (ACTIVE_SHARE, drawn here) in runs (ACTIVE_RUN_S), each active second with 1 or 2 acts (TWO_ACTS) on its two ticks. An act is a cursor
    `move` to a place near the last (MOVE_PX), a `type` of 1 to 3 visible characters or a `scroll`, in the odds of the app (KIND_ODDS); a
    full typing line (`hm.typing_cap`: the app's, or less where a decoration hides its end) is followed by a scroll of 1 line (enter), and
    any scroll starts the typing line afresh. A line that is hidden from its start is not typed. Returns
    `[[tick, "move", x, y] | [tick, "type", n] | [tick, "scroll", lines], ..]` (ticks are half seconds from the state's start, from 1, in
    order; `n` the visible characters on the typing line, `lines` the lines scrolled since the state's start) and where the cursor ends."""
    if level == "idle":
        return [], at
    seconds, app = state["to"] - state["from"], state["app"]
    r = hm._rng(state["seed"], "acts")
    share = r.uniform(*ACTIVE_SHARE[level])
    gap = lambda run: round(r.uniform(0.5, 1.5) * run * (1 - share) / share)  # noqa: E731  (idle seconds that give the share)
    acts: list[list] = []
    (x, y), typed, lines, sec = at, 0, 0, gap(3)
    cap = hm.typing_cap(state, 0)
    while sec < seconds:
        run = r.randint(*ACTIVE_RUN_S[level])
        for _ in range(min(run, seconds - sec)):
            ticks = (2 * sec, 2 * sec + 1) if r.random() < TWO_ACTS[level] else (2 * sec + r.randint(0, 1),)
            for tick in (t for t in ticks if t >= 1):
                odds = KIND_ODDS[app] if cap else (KIND_ODDS[app][0], 0, KIND_ODDS[app][2])
                kind = "scroll" if typed >= cap > 0 else r.choices(ACT_KINDS, odds)[0]
                if kind == "move":
                    for _ in range(20):
                        nx = min(max(round(x + r.gauss(0, MOVE_PX[2])), 0), hm.SCREEN[0] - 1)
                        ny = min(max(round(y + r.gauss(0, MOVE_PX[2])), 0), hm.SCREEN[1] - 1)
                        if MOVE_PX[0] <= math.dist((x, y), (nx, ny)) <= MOVE_PX[1]:
                            break
                    else:  # (cornered: any other place)
                        nx, ny = r.randrange(hm.SCREEN[0]), r.randrange(hm.SCREEN[1])
                    x, y = nx, ny
                    acts.append([tick, "move", x, y])
                elif kind == "type":
                    typed = min(cap, typed + r.choices(*TYPE_CHARS)[0])
                    acts.append([tick, "type", typed])
                else:
                    lines += 1 if typed >= cap > 0 else r.choices(*SCROLL_LINES)[0]
                    typed = 0
                    cap = hm.typing_cap(state, lines)
                    acts.append([tick, "scroll", lines])
            sec += 1
        sec += gap(run)
    return acts, (x, y)


def clip_length(r: random.Random, least: int = 10) -> int:
    """A porch clip's seconds: 10 to 30, 16 on average, at least `least`."""
    while (length := int(r.triangular(10, 30, 10))) < least:
        pass
    return length


# -- the plan ---------------------------------------------------------------------------------------------------------

class _Week:
    """The plan being made: a random stream of its own for each source (so a change to one leaves the rest as they were),
    the files, log lines and questions so far, and every text of the week, which no needle's code may be found in."""

    def __init__(self, seed: int, scale: str, days: Sequence[date]):
        self.seed, self.scale, self.sc, self.days = seed, scale, SCALES[scale], tuple(days)
        self.files: list[File] = []
        self.lines: dict[str, list[dict]] = {service: [] for service in SERVICES}
        self.questions: list[dict] = []
        self.texts: list[str] = []  # every text of the week so far
        self.taken: set[int] = set()  # the numbers that are a code, an error's or a sign's
        r = self.rng("house")  # the house's smoke alarm and doorbell: one pitch and one chime for the whole week
        self.alarm_hz = float(r.randrange(3050, 3351, 10))
        self.chime = [float(n) for n in r.choice(hm.DOORBELL_PAIRS)]

    def rng(self, name: str) -> random.Random:
        return random.Random(f"haystack-{VERSION}-{self.seed}-{self.scale}-{name}")

    @property
    def slots(self) -> list[tuple[int, str]]:
        return [(i, part) for i in range(len(self.days)) for part in PARTS]

    def span(self, slot: tuple[int, str]) -> tuple[float, float]:
        i, part = slot
        return at(self.days[i], PARTS[part][0] * HOUR), at(self.days[i], PARTS[part][1] * HOUR)

    def hint(self, slot: tuple[int, str]) -> str:
        i, part = slot
        lo, hi = PARTS[part]
        return f"on {date_text(self.days[i])}, in the {part} ({lo:02d}:00 to {hi:02d}:00)"

    def scope(self, level: str, where) -> tuple[float, float, str]:
        """The range and the hint of a question: `where` is its slot (`part`), its day (`day`), nothing (`week`) or the
        (since, until, hint) itself (`time`)."""
        if level == "part":
            return (*self.span(where), self.hint(where))
        if level == "day":
            return at(self.days[where], 0), at(self.days[where], 86400), f"on {date_text(self.days[where])}"
        if level == "week":
            return at(self.days[0], 0), at(self.days[-1], 86400), ""
        return where

    def pick(self, r: random.Random, population: Sequence, n: int, what: str) -> list:
        if n > len(population):
            raise ValueError(f"{what} needs {n} places and the week has {len(population)}")
        return r.sample(list(population), n)

    def fresh(self, r: random.Random, lo: int, hi: int, in_text: bool = True) -> int:
        """A number in [lo, hi) that is no other code or sign's number, and (`in_text`) is no number of any text so far."""
        blob = "\n".join(self.texts) if in_text else ""
        for _ in range(20_000):
            x = r.randrange(lo, hi)
            if x in self.taken or (in_text and (2020 <= x <= 2030 or re.search(rf"(?<!\d){x}(?!\d)", blob))):
                continue
            self.taken.add(x)
            return x
        raise ValueError(f"no number is left in [{lo}, {hi})")

    def eligible(self, source: str, avail: set) -> set:
        """The slots of `avail` with enough candidates (states, clips, photos, sounds) for a question to name them: at least the
        source's floor, with a tenth to spare for what is random (a screen state lasts 30 s on average, a segment of the
        microphone holds a sound or so)."""
        sc = self.sc
        floor = sc.part_floor[source]
        out = set()
        for slot in avail:
            d, part = slot
            k = list(PARTS).index(part)
            if source == "phone":
                have = share_out(sc.photos, PART_WEIGHTS)[k]
            elif source == "porch":
                have = share_out(sc.porch, PART_WEIGHTS)[k]
            elif source == "screen":
                lo, hi = PARTS[part][0] * HOUR, PARTS[part][1] * HOUR
                have = sum(max(0, min(b, hi) - max(a, lo)) for a, b in sc.screen[d % len(sc.screen)]) / MEAN_STATE_S
            else:  # about a sound to a segment
                lo, hi = PARTS[part][0] * HOUR, PARTS[part][1] * HOUR
                have = sum(max(0, min(b, hi) - max(a, lo)) for a, b in sc.mic[d % len(sc.mic)]) / 900
            if have >= floor * (1.1 if floor > 1 else 1):
                out.add(slot)
        return out

    def spot(self, template: str, avail: set, eligible: set | None = None) -> Spot:
        levels, absent = self.sc.asks[template]
        return arrange(self.rng(f"arrange-{template}"), template, len(self.days), levels, absent, avail,
                       avail if eligible is None else eligible, self.sc.extra, self.sc.shared, self.sc.early)

    def ask(self, template: str, level: str, where, *, thing: dict, value: str | None, moments: Sequence[Sequence[float]],
            grade_spec: dict, fake: tuple[str, bool, int, str], probe: dict | None = None, others: Sequence[Sequence[float]] = (),
            absent: bool = False, last: bool = False) -> None:
        """One question record (see the module's docstring). `thing` fills the template's words (`thing`, `topic`, `code`); an
        absent question is the template's own words about something that is not there."""
        type_, source = INFO[template]
        name = ABSENT_OF[template] if absent else template
        n = 1 + sum(1 for q in self.questions if q["template"] == name)
        since, until, hint = self.scope(level, where)
        words = {"hint": hint, **thing}
        question = (WEEK_WORDS[template] if level == "week" else WORDS[template]).format(**words)
        oracle = ORACLES[template]
        # an answer may repeat the numbers and times of the question (a code, the hours of the part of the day), and no others
        ignore = [*grade_spec["ignore"], *(x for x in grade.asked(question) if x not in grade_spec["ignore"])]
        self.questions.append({
            "qid": f"{type_}-{name}-{n}", "type": type_, "template": name, "absent": absent, "level": level, "last": last,
            "source": source, "since": since, "until": until, "hint": hint, "question": question,
            "truth": {"value": value, "moments": [[float(a), float(b)] for a, b in moments],
                      "others": [[float(a), float(b)] for a, b in others]},
            "grade": {**grade_spec, "ignore": ignore}, "oracle": oracle.format(**words) if oracle else None,
            "fake": {"question": fake[0], "oldest_first": fake[1], "limit": fake[2], "check": fake[3]}, "probe": probe})

    def emit(self, template: str, spot: Spot, placed: dict, args, absent_args) -> None:
        """The template's questions at the levels its scale asks. `args(level, needle, earlier)` is a question's keywords for
        the needle `placed` holds at the level's slot (`earlier`: the other needles, for a `week` question: all before it),
        `absent_args()` an absent one's."""
        levels, absent = self.sc.asks[template]
        for level in levels:
            if level == "time":
                continue
            slot = {"part": spot.part, "day": spot.day, "week": spot.last}[level]
            where = {"part": slot, "day": slot[0], "week": None}[level]
            earlier = [placed[s] for s in sorted(placed, key=slot_key) if s != slot] if level == "week" else []
            self.ask(template, level, where, last=level == "week" and template != "photo-count", **args(level, placed[slot], earlier))
        for level in absent:
            self.ask(template, level, spot.absent_part if level == "part" else spot.absent_day, absent=True, **absent_args())

    # -- text: config, notes

    def config(self) -> None:
        r, sc = self.rng("config"), self.sc
        n = len(self.days)
        times = [(i, m) for i in range(n) for m in spaced(r, 8 * 60, 20 * 60, sc.config, 120, what="app.toml versions")]
        values = self.pick(r, range(10, 100), len(times), "app.toml versions")
        for (i, minute), value in zip(times, values):
            text = f"[storage]\nretention_days = {value}\nmax_upload_mb = 25\n\n[server]\nport = 8080\n"
            self.texts.append(text)
            day = self.days[i]
            self.files.append(File(f"config/app.toml.{day:%Y-%m-%d}_{minute // 60:02d}-{minute % 60:02d}", "config",
                                   at(day, minute * 60), "app.toml", "config", {"text": text, "retention_days": value}))
        when = [at(self.days[i], m * 60) for i, m in times]
        used: set[tuple[int, str]] = set()  # the slots of the times asked about: one question to a slot
        for k in self.pick(r, range(len(times) - 1), 1, "config-asof (versions with a next one)"):
            candidates = [(i, minute) for i in range(n) for minute in range(6 * 60, 23 * 60 + 50)
                          if when[k] + 600 <= at(self.days[i], minute * 60) <= when[k + 1] - 600
                          and (i, part_of(minute)) not in used]
            if not candidates:
                raise ValueError("config-asof: no time is 10 minutes after a version and 10 minutes before the next")
            i, minute = r.choice(candidates)
            used.add((i, part_of(minute)))
            t = at(self.days[i], minute * 60)
            neighbours = [str(values[j]) for j in (k - 1, k + 1) if 0 <= j < len(values)]
            self.ask("config-asof", "time", (t, t + 1, f"at {minute // 60:02d}:{minute % 60:02d} on {date_text(self.days[i])}"),
                     thing={}, value=str(values[k]), moments=[[t, t + 1]], grade_spec=grade_of("number", str(values[k]), wrong=neighbours),
                     fake=("contains:retention_days", False, 50, "look"))

    def notes(self) -> None:
        r, sc = self.rng("notes"), self.sc
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("note-code", everywhere)
        topics = dict(zip(spot.needles, self.pick(r, TOPICS, len(spot.needles), "note-code (topics)")))
        minutes: dict[int, list[int]] = {i: [] for i in range(n)}
        entries: dict[tuple[int, str], int] = {}
        for slot in spot.needles:
            lo, hi = clip_range(slot[1])
            (minute,) = spaced(r, lo // 60, hi // 60, 1, 1, minutes[slot[0]], "note-code")
            minutes[slot[0]].append(minute)
            entries[slot] = minute
        for i in range(n):  # the distractors first: a needle's code must be in none of them
            need = sum(1 for slot in entries if slot[0] == i)
            for minute in spaced(r, DAY_START // 60, DAY_END // 60, max(0, sc.notes - need), 1, minutes[i], "notes"):
                self.add_note(i, minute, distractor_note(r))
        placed = {}
        for slot, minute in entries.items():
            thing, topic, sentence, digits = topics[slot]
            code = str(self.fresh(r, 10 ** (digits - 1), 10 ** digits))
            self.add_note(slot[0], minute, sentence.format(code=code), needle=True)
            placed[slot] = {"thing": thing, "topic": topic, "code": code, "t": at(self.days[slot[0]], minute * 60)}

        def args(level, needle, earlier):
            return {"thing": {"thing": needle["thing"], "topic": needle["topic"]}, "value": needle["code"],
                    "moments": [[needle["t"]] * 2], "grade_spec": grade_of("number", needle["code"]),
                    "fake": (f"contains:{needle['code']}", False, 50, "overlap")}

        def absent_args():
            return {"thing": {"thing": "Wi-Fi password", "topic": "a Wi-Fi password"}, "value": None, "moments": [],
                    "grade_spec": grade_of("absent"), "fake": ("contains:password", False, 50, "none")}

        self.emit("note-code", spot, placed, args, absent_args)

    def add_note(self, i: int, minute: int, text: str, needle: bool = False) -> None:
        day = self.days[i]
        self.texts.append(text)
        self.files.append(File(f"notes/note-{day:%Y-%m-%d}_{minute // 60:02d}-{minute % 60:02d}.md", "notes", at(day, minute * 60),
                               None, "note", {"text": text + "\n", "needle": needle}))

    # -- records: logs

    def logs_background(self) -> None:
        """Every service's ordinary lines: info, warnings and errors with codes ERR-100 to ERR-699. No number in them is a
        needle's code (ERR-700 to ERR-999, which is why a line has no bare number from 700 up)."""
        r, sc = self.rng("logs"), self.sc
        for i in range(len(self.days)):
            for service, share in zip(SERVICES, (0.4, 0.4, 0.2)):
                for secs in sorted(r.randrange(6 * HOUR, 86400) for _ in range(round(sc.log_lines * share))):
                    level, msg, code = background_line(r, service)
                    self.add_line(service, i, secs, level, msg, code)

    def add_line(self, service: str, i: int, secs: int, level: str, msg: str, code: str | None = None) -> None:
        day = self.days[i]
        line = {"ts": datetime(day.year, day.month, day.day, secs // 3600, secs // 60 % 60, secs % 60).isoformat(timespec="seconds"),
                "service": service, "level": level, "msg": msg, **({"code": code} if code else {})}
        self.lines[service].append(line)
        self.texts.append(json.dumps(line))

    def logs_needles(self) -> None:
        r = self.rng("log-needles")
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("log-count", everywhere)
        placed = {}
        for slot in spot.needles:
            i, part = slot
            k = r.randint(2, 6)
            code = f"ERR-{self.fresh(r, 700, 1000)}"
            lo, hi = (PARTS[part][0] * HOUR, PARTS[part][1] * HOUR) if slot == spot.part else (6 * HOUR, 86400)
            times = spaced(r, lo, hi, k, 600, what="log-count")
            others = [j for j in range(n) if j != i]
            if not others:
                raise ValueError("log-count: the week has no other day for the code to be logged on")
            for secs in times:
                self.add_line("payments", i, secs, "error", f"{r.choice(REASONS)} order={r.randrange(10000, 99999)}", code)
            for _ in range(r.randint(1, 3)):
                self.add_line("payments", r.choice(others), r.randrange(6 * HOUR, 86400), "error",
                              f"{r.choice(REASONS)} order={r.randrange(10000, 99999)}", code)
            placed[slot] = {"k": k, "code": code, "moments": [[at(self.days[i], s)] * 2 for s in times]}

        def args(level, needle, earlier):
            return {"thing": {"code": needle["code"]}, "value": str(needle["k"]), "moments": needle["moments"],
                    "grade_spec": grade_of("count", str(needle["k"]), ignore=[needle["code"][4:]]),
                    "fake": (f"contains:{needle['code']}", False, 50, "overlap")}

        def absent_args():
            code = f"ERR-{self.fresh(r, 700, 1000)}"
            return {"thing": {"code": code}, "value": None, "moments": [], "grade_spec": grade_of("absent"),
                    "fake": (f"contains:{code}", False, 50, "none")}

        self.emit("log-count", spot, placed, args, absent_args)
        last = self.spot("log-last", everywhere)
        builds: set[str] = set()
        for slot in last.needles:
            i, part = slot
            finishes = spaced(r, PARTS[part][0] * HOUR + 8 * 60, PARTS[part][1] * HOUR, r.randint(2, 4), 1200, what="log-last")
            ids = []
            for finish in finishes:
                build = self.new_build(r, builds)
                ids.append(build)
                self.add_line("web", i, finish - r.randint(180, 480), "info", f"deploy started build {build}")
                self.add_line("web", i, finish, "info", f"deploy finished build {build}")
            self.ask("log-last", "part", slot, thing={}, value=clock(finishes[-1]),
                     moments=[[at(self.days[i], s)] * 2 for s in finishes],
                     grade_spec=grade_of("time", clock(finishes[-1]), clock(finishes[-1]), 1, [clock(s) for s in finishes[:-1]]),
                     fake=(f"contains:{ids[-1]}", False, 50, "overlap"))
        free = [s for s in self.slots if s not in last.needles]  # the web service's other deploys, in no needle's slot
        for i in range(n):
            here = [s for s in free if s[0] == i]
            for slot in r.sample(here, min(len(here), r.randint(1, 2))):
                (finish,) = spaced(r, PARTS[slot[1]][0] * HOUR + 8 * 60, PARTS[slot[1]][1] * HOUR, 1, 1, what="web deploys")
                build = self.new_build(r, builds)
                self.add_line("web", i, finish - r.randint(180, 480), "info", f"deploy started build {build}")
                self.add_line("web", i, finish, "info", f"deploy finished build {build}")

    def new_build(self, r: random.Random, used: set[str]) -> str:
        while (build := f"{r.randrange(1 << 24):06x}") in used or build.isdigit():
            pass
        used.add(build)
        return build

    # -- image: the phone

    def photo_spec(self, r: random.Random, kind: str) -> dict:
        """A photo of a kind: a `sign` of another colour, `green` or `red` things that are no sign or car and a good part of the
        photo (a green car, a red bag or umbrella), `greensmall` or `redsmall` ones that are not (a door, a tomato), a `greenwall` or
        `redwall` (a hedge or a facade, most of the photo: more of the needle's colour than the needle has), a `car` of another
        colour, or a `plain` scene."""
        seed = r.randrange(1 << 30)
        green, red = DISTRACTOR_RGB["sign"], DISTRACTOR_RGB["car"]
        if kind == "sign":
            colour = r.choice(OTHER_SIGNS)
            return {"scene": "sign", "rgb": list(green[colour]), "word": r.choice(SIGN_WORDS),
                    "number": str(self.fresh(r, 10, 1000, False)), "box": None, "seed": seed}
        if kind == "car":
            return {"scene": "car", "rgb": list(red[r.choice(OTHER_CARS)]), "seed": seed}
        if kind == "greenwall":
            return {"scene": "wall", "rgb": list(r.choice((green["plant"], green["door"]))), "seed": seed}
        if kind == "redwall":
            return {"scene": "wall", "rgb": list(r.choice((red["umbrella"], red["tomato"]))), "seed": seed}
        if kind in ("green", "greensmall"):  # (a green car is a third of the photo, a green door a tenth)
            scene = "car" if kind == "green" else r.choices(("room", "plant", "street", "object"), (2, 1, 1, 1))[0]
            rgb = {"car": green["car"], "object": r.choice((green["shirt"], green["plant"], green["door"])), "room": green["door"],
                   "plant": green["plant"], "street": green["shirt"]}[scene]
            return {"scene": scene, "rgb": list(rgb), "seed": seed}
        if kind in ("red", "redsmall"):
            scene = r.choices(("object",) if kind == "red" else ("room", "food", "street"), (1,) if kind == "red" else (2, 1, 1))[0]
            rgb = {"object": r.choice((red["umbrella"], red["door"], red["tomato"])), "room": red["door"], "food": red["tomato"],
                   "street": red["umbrella"]}[scene]
            return {"scene": scene, "rgb": list(rgb), "seed": seed}
        return {"scene": r.choice(("object", "room", "food", "plant", "street")), "rgb": None, "seed": seed}

    def needle_seed(self, r: random.Random, scene: str) -> int:
        """A seed for a needle's photo (`sign` or `car`) whose backdrop is not a fence. A fence's vertical lines make a JPEG 15 KB
        bigger than a wall's or a sky's, so a third of the signs and cars would be the biggest files of their day: the needle must not
        be one by luck (the probe's size check), and it is no smaller than the rest either."""
        while True:
            seed = r.randrange(1 << 30)
            if tuple(hm._backdrop(hm._rng(seed, scene)).getpixel((3, 0))) != FENCE_RGB:
                return seed

    def phone(self) -> None:
        r, sc = self.rng("phone"), self.sc
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("photo-sign", everywhere, self.eligible("phone", everywhere))
        car_levels = sc.asks["photo-count"][0]
        car_day = r.randrange(n)  # the day a count question asks about; the other days hold red cars too
        cars = {d: r.randint(*sc.other_cars) for d in range(n)}
        cars[car_day] = r.randint(*sc.cars)
        others = [d for d in range(n) if d != car_day]
        while sum(cars[d] for d in others) < sc.other_min:
            cars[r.choice(others)] += 1
        shots: dict[tuple[int, str], list[dict]] = {}  # slot -> its photos: {"t": second of the day, "spec": ..}
        busy: dict[int, list[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for part, count in zip(PARTS, share_out(sc.photos, PART_WEIGHTS)):
                times = spaced(r, *clip_range(part), count, 120, busy[i], f"photos of {part}")
                busy[i] += times
                shots[i, part] = [{"t": t, "spec": None} for t in times]

        def free(slot: tuple[int, str]) -> list[dict]:
            return [p for p in shots[slot] if p["spec"] is None]

        def take(pool: list[dict], what: str) -> dict:
            if not pool:
                raise ValueError(f"phone: no free photo for {what}: the scale's photos a day ({sc.photos}) hold less than its needles and distractors")
            return r.choice(pool)

        placed = {}
        for slot in spot.needles:
            p = take(free(slot), "a green sign")
            number = str(self.fresh(r, 10, 1000, False))
            p["spec"] = {"scene": "sign", "rgb": list(NEEDLE_RGB["sign"]), "word": r.choice(SIGN_WORDS), "number": number,
                         "box": None, "seed": self.needle_seed(r, "sign")}
            placed[slot] = {"slot": slot, "number": number, "t": at(self.days[slot[0]], p["t"])}
        for slot in self.slots:
            for kind, count in sc.mix["phone"]["slot"].items():
                for _ in range(count):
                    take(free(slot), kind)["spec"] = self.photo_spec(r, kind)
        for i in range(n):
            for kind, count in sc.mix["phone"]["day"].items():
                for _ in range(count):
                    take([p for part in PARTS for p in free((i, part))], kind)["spec"] = self.photo_spec(r, kind)
        for i in range(n):  # the day's red cars, 5 minutes apart
            pool = [p for part in PARTS for p in free((i, part))]
            for _ in range(500):
                chosen = r.sample(pool, cars[i]) if len(pool) >= cars[i] else None
                if chosen is not None and all(abs(a["t"] - b["t"]) >= 300 for k, a in enumerate(chosen) for b in chosen[:k]):
                    break
            else:
                raise ValueError(f"phone: no room for {cars[i]} red cars 5 minutes apart on day {i + 1}")
            for p in chosen:
                p["spec"] = {"scene": "car", "rgb": list(NEEDLE_RGB["car"]), "seed": self.needle_seed(r, "car")}
        for ps in shots.values():  # the rest: ordinary photos, some of them of the needles' colours
            for p in ps:
                if p["spec"] is None:
                    p["spec"] = self.photo_spec(r, r.choices(("plain", "sign", "greensmall", "redsmall", "car"), (55, 10, 10, 10, 15))[0])
        for (i, part), ps in sorted(shots.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[i]
            for p in ps:
                self.files.append(File(f"phone/IMG_{day:%Y%m%d}_{clock(p['t']).replace(':', '')}.jpg", "phone", at(day, p["t"]), None,
                                       "photo", p["spec"]))
        for i in range(n):
            day = self.days[i]
            for secs in spaced(r, DAY_START, DAY_END, sc.clips, 120, busy[i], "phone clips"):
                busy[i].append(secs)
                self.files.append(File(f"phone/VID_{day:%Y%m%d}_{clock(secs).replace(':', '')}.mp4", "phone", at(day, secs), None,
                                       "clip", {"scene": r.choice(("street", "dog", "shop")), "seed": r.randrange(1 << 30)}))

        def wrong_signs(since: float, until: float) -> list[str]:
            """The numbers of the signs in other colours in a range: what a wrong answer of a model would be."""
            return [f.spec["number"] for f in self.files if f.kind == "photo" and f.spec["scene"] == "sign"
                    and tuple(f.spec["rgb"]) != NEEDLE_RGB["sign"] and since <= f.t < until]

        def args(level, needle, earlier):
            since, until, _ = self.scope(level, {"part": needle["slot"], "day": needle["slot"][0], "week": None}[level])
            wrong = [e["number"] for e in earlier] if level == "week" else wrong_signs(since, until)
            return {"thing": {}, "value": needle["number"], "moments": [[needle["t"]] * 2],
                    "others": [[e["t"]] * 2 for e in earlier], "grade_spec": grade_of("number", needle["number"], wrong=wrong),
                    "fake": (shade_of("sign"), False, 50, "last" if level == "week" else "overlap"),
                    "probe": {"kind": "shade", "hue": "green", "rgb": list(NEEDLE_RGB["sign"])}}

        def absent_args():
            return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (shade_of("sign"), False, 50, "none"),
                    "probe": {"kind": "shade", "hue": "green", "rgb": list(NEEDLE_RGB["sign"])}}

        self.emit("photo-sign", spot, placed, args, absent_args)
        for level in car_levels:  # the red cars: a day's, or the week's
            days = [car_day] if level == "day" else list(range(n))
            moments = [[at(self.days[i], p["t"])] * 2 for i in days for part in PARTS for p in shots[i, part]
                       if p["spec"]["scene"] == "car" and tuple(p["spec"]["rgb"]) == NEEDLE_RGB["car"]]
            self.ask("photo-count", level, car_day if level == "day" else None, thing={}, value=str(len(moments)),
                     moments=moments, grade_spec=grade_of("count", str(len(moments))), fake=(shade_of("car"), False, 50, "count"),
                     probe={"kind": "shade", "hue": "red", "rgb": list(NEEDLE_RGB["car"])})

    # -- video: the screen and the porch

    def screen(self) -> None:
        """Each day's screen blocks as the recorder writes them (DESIGN.md §4.1, §6): a block is planned whole as states of 10 to 60 s,
        each with an activity `level` (`next_level`: a chain over the block, from a stream of its own) and the owner's acts at that level
        (`state_acts`, from the state's own seed: the cursor's place at the start, which is where the last state left it, and its moves,
        typing and scrolling), and then cut into segments of about `screen_s` (`segments`), each starting on a state's start and holding
        each state once. Its frames are `hm.screen_frames`': a state's start, an act, a minute of the clock, at most 2 a second. The error
        is a state of 20 to 40 s at least 20 s from its block's edges, with the activity of any other state."""
        r, sc = self.rng("screen"), self.sc
        blocks: dict[tuple[int, str], list[dict]] = {}  # slot -> its blocks: {"start", "end", "error", "states", "segments"} (seconds of the day)
        for d in range(len(self.days)):
            for lo, hi in sc.screen[d % len(sc.screen)]:
                part = part_of(lo // 60)
                if part_of((hi - 1) // 60) != part:  # (so is every segment: none leaves its block)
                    raise ValueError(f"screen: the block at {clock(lo)} is in two parts of the day")
                blocks.setdefault((d, part), []).append({"start": lo, "end": hi, "error": None, "states": [], "segments": []})
        avail = set(blocks)
        spot = self.spot("screen-error", avail, self.eligible("screen", avail))
        for slot in spot.needles:
            block = r.choice(blocks[slot])
            span = r.randint(20, 40)
            a = r.randint(block["start"] + 20, block["end"] - span - 20)
            block["error"] = {"from": a, "to": a + span, "code": f"E-{self.fresh(r, 1000, 10000)}", "msg": r.choice(ERRORS)}
        levels, cursor = self.rng("screen-activity"), hm.CURSOR_HOME  # the owner's cursor goes on from state to state
        for here in blocks.values():
            for block in here:
                err = block["error"]
                if err:
                    spans = [*((a, b, None) for a, b in cut(r, block["start"], err["from"])), (err["from"], err["to"], err),
                             *((a, b, None) for a, b in cut(r, err["to"], block["end"]))]
                else:
                    spans = [(a, b, None) for a, b in cut(r, block["start"], block["end"])]
                prev = None
                for a, b, e in spans:
                    app = r.choice([x for x in hm.APPS if x != prev])
                    prev = app
                    overlay = {"kind": "error", "rgb": list(NEEDLE_RGB["error"]), "code": e["code"], "msg": e["msg"], "at": hm.dialog_at(r)} if e else None
                    block["states"].append({"from": a, "to": b, "app": app, "overlay": overlay, "accent": None, "seed": r.randrange(1 << 30)})
            self.decorate(r, [s for block in here for s in block["states"]])
            for block in here:  # then the owner's acts (the decorations can make a state a browser's), a chain of levels over the block
                level = None
                for s in block["states"]:
                    level = next_level(levels, level)
                    s["level"], s["cursor"] = level, list(cursor)
                    s["acts"], cursor = state_acts(s, cursor, level)
            for block in here:  # then the segments, whose states are timed from their own start
                for lo, hi in segments([s["from"] for s in block["states"]], block["start"], block["end"], sc.screen_s):
                    block["segments"].append({"start": lo, "length": hi - lo, "states": [{**s, "from": s["from"] - lo, "to": s["to"] - lo}
                                                                                        for s in block["states"] if lo <= s["from"] < hi]})
        for slot, here in sorted(blocks.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[slot[0]]
            for block in here:
                for seg in block["segments"]:
                    self.files.append(File(f"screen/screen-{day:%Y-%m-%d}_{clock(seg['start']).replace(':', '-')}.mkv", "screen",
                                           at(day, seg["start"]), None, "screen", {"length_s": seg["length"], "states": seg["states"]}))
        placed = {}
        for slot in spot.needles:  # (the error is one state, so it is whole in one segment: `verify` reads it back from the files)
            err = next(b["error"] for b in blocks[slot] if b["error"])
            day = self.days[slot[0]]
            placed[slot] = {"day": slot[0], "code": err["code"], "clock": clock(err["from"]), "moments": [[at(day, err["from"]), at(day, err["to"])]]}

        def args(level, needle, earlier):
            week = level == "week"
            value = f"{needle['code']} at {needle['clock']}" + (f" on {date_text(self.days[needle['day']])}" if week else "")
            return {"thing": {}, "value": value, "moments": needle["moments"], "others": [m for e in earlier for m in e["moments"]],
                    "grade_spec": grade_of("code_time", needle["code"], needle["clock"], 1, date_=self.days[needle["day"]].isoformat() if week else None),
                    "fake": (shade_of("error"), False, 50, "last" if week else "overlap"),
                    "probe": {"kind": "shade", "hue": "red", "rgb": list(NEEDLE_RGB["error"])}}

        def absent_args():
            return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (shade_of("error"), False, 50, "none"),
                    "probe": {"kind": "shade", "hue": "red", "rgb": list(NEEDLE_RGB["error"])}}

        self.emit("screen-error", spot, placed, args, absent_args)

    def dialog(self, r: random.Random, kind: str) -> dict:
        """An overlay that is no error, opened somewhere in the middle of the screen as an error is: a warning (orange, with its
        code), an `alert` (a warning with a red header, which a red filter takes for an error), a save prompt, an info box or a red
        "Delete?" confirm."""
        at = hm.dialog_at(r)
        if kind in ("warning", "alert"):
            red = {"rgb": list(DISTRACTOR_RGB["error"]["confirm"])} if kind == "alert" else {}
            return {"kind": "warning", **red, "code": f"W-{self.fresh(r, 1000, 10000, False)}", "msg": r.choice(WARNINGS), "at": at}
        return {"kind": kind, "at": at}

    def decorate(self, r: random.Random, states: list[dict]) -> None:
        """The distractors of a part of a day (its states): the scale's planted ones (dialogs of each kind, red ones among them, red banners and
        other red accents), then about 15% of the states with a red accent and 2% with a dialog."""
        mix = self.sc.mix["screen"]["slot"]
        free = [s for s in states if s["overlay"] is None]
        r.shuffle(free)
        for kind in ("confirm", "alert", "warning", "info", "save"):
            for _ in range(mix.get(kind, 0)):
                if not free:
                    raise ValueError("screen: a part of a day has fewer states than the scale's dialogs")
                free.pop()["overlay"] = self.dialog(r, kind)
        bare = [s for s in states if s["accent"] is None]
        r.shuffle(bare)
        for _ in range(mix.get("banner", 0)):  # a banner is a web page's: a red band bigger than the error's header
            s = bare.pop()
            s["app"], s["accent"] = "browser", {"kind": "banner"}
        for _ in range(mix.get("accent", 0)):
            s = bare.pop()
            s["accent"] = {"kind": {"browser": "banner", "sheet": "cells"}.get(s["app"], "toast")}
        for s in states:
            if s["accent"] is None and r.random() < 0.15:
                s["accent"] = {"kind": r.choice({"browser": ("banner", "toast", "badge"), "sheet": ("cells", "badge", "toast")}.get(s["app"], ("toast", "badge")))}
            if s["overlay"] is None and r.random() < 0.02:
                s["overlay"] = self.dialog(r, r.choice(("confirm", "warning", "info", "save", "save")))

    def event(self, r: random.Random, kind: str, length: int, window: tuple[int, int] | None = None) -> dict:
        """A distractor of the porch camera over some seconds of a clip of `length` (within the seconds `window` when there is one): a
        `walker`, `cat`, `dog`, `car`, a `visitor` close to the camera in a brown coat (more brown, in the place a parcel is left, than a
        parcel has), a `light` change, or the `courier` bringing a box up to the door and taking it away again (`drop_at` None: he never
        puts it down, and the step stays empty: nothing is delivered)."""
        lo, hi = window or (0, length)
        a = r.randint(lo, max(lo, hi - 8))
        b = min(hi - 1, a + r.randint(4, 10))
        if b <= a:
            a, b = max(lo, hi - 6), hi - 1
        if b <= a:
            raise ValueError(f"a {kind} needs 2 seconds or more of a clip, and the window {lo} to {hi} has fewer")
        if kind == "courier":
            return {"kind": "courier", "from": a, "to": b, "drop_at": None}
        if kind == "visitor":  # a brown coat close to the camera, at the door (x), with a carrier bag or without
            return {"kind": "visitor", "from": a, "to": b, "x": r.randint(110, 200), "scale": round(r.uniform(2.9, 3.2), 2), "bag": r.random() < 0.5}
        return {"kind": kind, "from": a, "to": b}

    @staticmethod
    def parcel_free(clip: dict) -> tuple[int, int]:
        """The seconds `(from, to)` of a clip that show no parcel: all of it, the seconds before the drop, or those after the pickup."""
        parcel = clip["parcel"]
        return (0, clip["length"]) if not parcel else (0, parcel[0]) if parcel[0] > 0 else (parcel[1], clip["length"])

    def has_person(self, clip: dict) -> bool:
        return any(e["kind"] in ("courier", "owner_pickup") for e in clip["events"])

    def porch(self) -> None:
        """Clips of 10 to 30 s all day, one frame a second. On the days of the needles a courier leaves a parcel in a clip, the
        clips of the next 20 to 120 minutes show it, and the owner takes it in in the last: all in one part of the day (so that
        no part holds a parcel that was left in another)."""
        r, sc = self.rng("porch"), self.sc
        n = len(self.days)
        everywhere = set(self.slots)
        spot = self.spot("porch-parcel", everywhere, self.eligible("porch", everywhere))
        clips: dict[tuple[int, str], list[dict]] = {}
        fixed: dict[int, list[int]] = {i: [] for i in range(n)}
        placed = {}
        for slot in spot.needles:  # the clip a parcel is left in, and the clip it is taken in
            i, part = slot
            lo, hi = clip_range(part)
            for _ in range(2000):
                stay = r.randint(20, 120) * 60
                first, second = r.randint(14, 20), r.randint(12, 20)  # (a middling length: a needle's file is neither the biggest nor the smallest)
                drop_at, pick = r.randint(4, 8), r.randint(4, 10)
                top = hi - 30 - second - drop_at - stay + pick
                if top > lo:
                    break
            else:
                raise ValueError(f"porch-parcel: no room for a parcel to stay up to 2 hours in the {part}")
            t = r.randrange(lo, top)
            back = t + drop_at + stay - pick
            box = hm.parcel_box(r)  # where it is left on the step: it stays there from the drop to the pickup
            clips.setdefault(slot, []).extend([
                {"t": t, "length": first, "events": [{"kind": "courier", "from": 0, "to": min(first - 1, drop_at + 6), "drop_at": drop_at}],
                 "parcel": [drop_at, first], "box": box},
                {"t": back, "length": second, "events": [{"kind": "owner_pickup", "from": max(0, pick - 3), "to": min(second - 1, pick + 3)}],
                 "parcel": [0, pick], "box": box}])
            fixed[i] += [t, back]
            placed[slot] = {"day": i, "clock": clock(t + drop_at), "t": t, "back": back, "drop_at": drop_at, "box": box}
        for i in range(n):
            for part, count in zip(PARTS, share_out(sc.porch, PART_WEIGHTS)):
                made = clips.setdefault((i, part), [])
                if len(made) > count:
                    raise ValueError(f"porch: the {part} of day {i + 1} has {count} clips, a parcel needs {len(made)} of them")
                lo, hi = clip_range(part)
                for t in spaced(r, lo, hi - 30, count - len(made), 120, fixed[i], f"porch clips of the {part}"):
                    made.append({"t": t, "length": clip_length(r), "events": [], "parcel": None})
                    fixed[i].append(t)
            day_clips = sorted((c for part in PARTS for c in clips[i, part]), key=lambda c: c["t"])
            for slot, needle in placed.items():  # the parcel is on the step in the clips between the two
                if slot[0] == i:
                    for c in day_clips:
                        if needle["t"] < c["t"] < needle["back"] and c["parcel"] is None:
                            c["parcel"], c["box"] = [0, c["length"]], needle["box"]
        for slot, cs in clips.items():
            plant = [kind for kind, count in sc.mix["porch"]["slot"].items() for _ in range(count)]
            for kind in plant:  # (in a clip that shows the parcel, on the seconds that do not: a scope of only the needle's own clips must still
                #                   hold the distractors' frames, and not have them hidden among the parcel's)
                usable = [c for c in cs if (free := self.parcel_free(c))[1] - free[0] >= 4]
                if not usable:
                    raise ValueError(f"porch: no clip of {slot} has 4 seconds with no parcel for a {kind}")
                calm = [c for c in usable if not self.has_person(c)]
                c = r.choice([c for c in calm if c["parcel"] is None] or calm or usable)
                c["events"].append(self.event(r, kind, c["length"], self.parcel_free(c)))
        for i in range(n):
            plant = [kind for kind, count in sc.mix["porch"]["day"].items() for _ in range(count)]
            day_clips = [c for part in PARTS for c in clips[i, part]]
            for kind in plant:
                c = r.choice(day_clips)
                c["events"].append(self.event(r, kind, c["length"]))
        for slot, cs in sorted(clips.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[slot[0]]
            for c in sorted(cs, key=lambda c: c["t"]):
                if not c["events"] or (r.random() < 0.15 and not self.has_person(c)):
                    kinds, weights = ("walker", "car", "cat", "dog", "light", "courier"), (45, 15, 12, 8, 10, 5)
                    if c["parcel"]:  # a clip that shows the parcel has nothing that makes its file bigger than the others': its size gives it away
                        kinds, weights = kinds[:3:2], weights[:3:2]
                    c["events"].append(self.event(r, r.choices(kinds, weights)[0], c["length"]))
                spec = {"length_s": c["length"], "events": c["events"], "parcel": c["parcel"], "box": c.get("box"), "seed": r.randrange(1 << 30)}
                self.files.append(File(f"porch/porch-{day:%Y-%m-%d}_{clock(c['t']).replace(':', '-')}.mp4", "porch", at(day, c["t"]),
                                       None, "porch", spec))
        for slot, needle in placed.items():  # where the parcel is on the step: its clips, in the files as they were made
            day = self.days[needle["day"]]
            needle["moments"] = [[at(day, c["t"]) + c["parcel"][0], at(day, c["t"]) + c["parcel"][1]]
                                 for part in PARTS for c in sorted(clips[needle["day"], part], key=lambda c: c["t"])
                                 if c["parcel"] and needle["t"] <= c["t"] <= needle["back"]]

        def args(level, needle, earlier):
            week = level == "week"
            return {"thing": {}, "value": needle["clock"], "moments": needle["moments"], "others": [m for e in earlier for m in e["moments"]],
                    "grade_spec": grade_of("time", needle["clock"], needle["clock"], 2, apart([e["clock"] for e in earlier], needle["clock"]) if week else (),
                                           date_=self.days[needle["day"]].isoformat() if week else None),
                    "fake": (shade_of("parcel"), False, 50, "last" if week else "overlap"),
                    "probe": {"kind": "shade", "hue": "brown", "rgb": list(NEEDLE_RGB["parcel"])}}

        def absent_args():
            return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (shade_of("parcel"), False, 50, "none"),
                    "probe": {"kind": "shade", "hue": "brown", "rgb": list(NEEDLE_RGB["parcel"])}}

        self.emit("porch-parcel", spot, placed, args, absent_args)

    # -- audio: the hallway microphone

    def mic(self) -> None:
        r, sc = self.rng("mic"), self.sc
        n = len(self.days)
        segs: dict[tuple[int, str], list[dict]] = {}
        for d in range(n):
            for lo, hi in sc.mic[d % len(sc.mic)]:
                for start in range(lo, hi, 900):
                    segs.setdefault((d, part_of(start // 60)), []).append(
                        {"day": d, "start": start, "length": 900, "sounds": [], "tv": [], "seed": r.randrange(1 << 30)})
        avail = set(segs)
        eligible = self.eligible("mic", avail)
        spots = {"mic-doorbell": self.spot("mic-doorbell", avail, eligible), "mic-alarm": self.spot("mic-alarm", avail, eligible)}
        needles: dict[str, dict] = {}
        for template, kind, dur in (("mic-doorbell", "doorbell", hm.CHIME_DUR), ("mic-alarm", "alarm", hm.ALARM_DUR)):
            needles[template] = {}
            for slot in spots[template].needles:
                extra = {"notes": list(self.chime)} if kind == "doorbell" else {"hz": self.alarm_hz}
                seg, sound = self.place_sound(r, segs[slot], kind, dur, extra, margin=100)
                needles[template][slot] = {"day": slot[0], "start": seg["start"], "sound": sound}
        mix = sc.mix["mic"]["slot"]
        beepers, band = ("microwave", "oven", "watch", "truck"), ("ding", "ding", "ding", "ring", "bark")
        for slot, ss in segs.items():
            for kind in [r.choice(beepers) for _ in range(mix.get("beeper", 0))] + [r.choice(band) for _ in range(mix.get("band", 0))]:
                extra = {"hz": 987.77} if kind == "ding" else {"hz": 440.0} if kind == "ring" else {}
                self.place_sound(r, ss, kind, self.sound_dur(r, kind), extra)
            for _ in range(mix.get("loud", 0)):  # louder than the needles, each in its band: a beeper in the alarm's, a timer in the chime's
                kind = r.choice(("oven", "truck"))
                self.place_sound(r, ss, kind, self.sound_dur(r, kind), {"gain": hm.LOUD_GAIN})
                self.place_sound(r, ss, "timer", self.sound_dur(r, "timer"), {"gain": hm.LOUD_GAIN})
            for _ in range(sum(r.random() < sc.mic_extra for _ in ss)):
                kind = r.choice(("kettle", "slam", "steps", "vacuum", "ding", "ring", "bark", "microwave", "watch"))
                self.place_sound(r, ss, kind, self.sound_dur(r, kind), {})
        for d in range(n):  # TV: stretches of 2 to 8 minutes, up to the day's minutes, never within a minute of a needle
            today = [seg for slot, ss in segs.items() if slot[0] == d for seg in ss]
            total = 0
            for _ in range(2000):
                if total >= sc.tv_s:
                    break
                seg = r.choice(today)
                length = r.randint(120, 480)
                a = r.randint(0, seg["length"] - length)
                needles_here = [s for s in seg["sounds"] if s["kind"] in ("doorbell", "alarm")]
                if all(a + length + MIC_GAP_S <= s["at"] or a >= s["at"] + s["dur"] + MIC_GAP_S for s in needles_here) and \
                        all(a + length <= t0 or a >= t1 for t0, t1 in seg["tv"]):
                    seg["tv"].append([a, a + length])
                    total += length
            else:
                raise ValueError(f"mic: no room for {sc.tv_s} s of TV on day {d + 1}")
        for slot, ss in sorted(segs.items(), key=lambda kv: slot_key(kv[0])):
            day = self.days[slot[0]]
            for seg in sorted(ss, key=lambda s: s["start"]):
                spec = {"length_s": seg["length"], "seed": seg["seed"], "bed_dbfs": hm.BED_DBFS, "tv": sorted(seg["tv"]),
                        "sounds": sorted(seg["sounds"], key=lambda s: s["at"])}
                self.files.append(File(f"mic/mic-{day:%Y-%m-%d}_{clock(seg['start']).replace(':', '-')}.ogg", "mic", at(day, seg["start"]),
                                       None, "mic", spec))
        for template, kind in (("mic-doorbell", "doorbell"), ("mic-alarm", "alarm")):
            hz = self.chime[0] if kind == "doorbell" else self.alarm_hz
            probe = {"kind": "pitch", "hz": hz, "band": BANDS[kind]}
            placed = {}
            for slot, needle in needles[template].items():
                s, day = needle["sound"], self.days[needle["day"]]
                heard = int(s["at"])
                placed[slot] = {"day": needle["day"], "clock": clock(needle["start"] + heard),
                                "moments": [[at(day, needle["start"]) + s["at"], at(day, needle["start"]) + s["at"] + s["dur"]]]}

            def args(level, needle, earlier, hz=hz, probe=probe):
                week = level == "week"
                return {"thing": {}, "value": needle["clock"], "moments": needle["moments"], "others": [m for e in earlier for m in e["moments"]],
                        "grade_spec": grade_of("time", needle["clock"], needle["clock"], 1, apart([e["clock"] for e in earlier], needle["clock"]) if week else (),
                                               date_=self.days[needle["day"]].isoformat() if week else None),
                        "fake": (f"pitch:{hz:g}", False, 50, "last" if week else "overlap"), "probe": probe}

            def absent_args(hz=hz, probe=probe):
                return {"thing": {}, "value": None, "moments": [], "grade_spec": grade_of("absent"), "fake": (f"pitch:{hz:g}", False, 50, "none"),
                        "probe": probe}

            self.emit(template, spots[template], placed, args, absent_args)

    def sound_dur(self, r: random.Random, kind: str) -> float:
        return {"microwave": 2.0, "oven": 2.5, "watch": 1.0, "truck": 3.0, "timer": 2.8, "ding": 0.8, "ring": 2.4, "slam": 0.15, "bark": 1.2,
                "kettle": round(r.uniform(3, 5), 1), "steps": 0.5 * r.randint(4, 8), "vacuum": float(r.randint(5, 10))}[kind]

    def place_sound(self, r: random.Random, segments: list[dict], kind: str, dur: float, extra: dict, margin: int = 2) -> tuple[dict, dict]:
        """A sound of `kind` somewhere in one of `segments`: 2 s from the others' and a needle's MIC_GAP_S, and from the segment's
        ends by `margin`. Returns the segment and the sound's spec."""
        for _ in range(500):
            seg = r.choice(segments)
            at_ = round(r.uniform(margin, seg["length"] - dur - margin), 1)
            ok = True
            for s in seg["sounds"]:
                gap = MIC_GAP_S if "doorbell" in (kind, s["kind"]) or "alarm" in (kind, s["kind"]) else 2
                if not (at_ + dur + gap <= s["at"] or at_ >= s["at"] + s["dur"] + gap):
                    ok = False
                    break
            if ok:
                sound = {"kind": kind, "at": at_, "dur": dur, "seed": r.randrange(1 << 30), **extra}
                seg["sounds"].append(sound)
                return seg, sound
        raise ValueError(f"mic: no room for a {kind} in {len(segments)} segments")


def plan(seed: int, scale: str, days: Sequence[date]) -> Plan:
    """The week for `seed` over `days` (oldest first, one per day of the scale): its files, its log lines and its questions.
    Pure: nothing is written and no clock is read. A `ValueError` names a template whose needs the week cannot meet; the
    plan checks itself (`verify`)."""
    if scale not in SCALES:
        raise ValueError(f"a scale is one of {', '.join(SCALES)}, not {scale!r}")
    days = tuple(days)
    if len(days) != SCALES[scale].days:
        raise ValueError(f"the {scale} scale is {SCALES[scale].days} days, not {len(days)}")
    if any(later <= earlier for earlier, later in zip(days, days[1:])):
        raise ValueError("the days are in order, oldest first, none twice")
    week = _Week(seed, scale, days)
    week.logs_background()
    week.config()
    week.notes()
    week.logs_needles()
    week.phone()
    week.screen()
    week.porch()
    week.mic()
    order = {template: (TYPES.index(type_), k) for type_ in TYPES for k, template in enumerate((*TEMPLATES[type_], *ABSENT[type_]))}
    made = Plan(seed, scale, days, tuple(sorted(week.files, key=lambda f: (f.t, f.path))),
                {service: tuple(sorted(lines, key=lambda line: line["ts"])) for service, lines in week.lines.items()},
                tuple(sorted(week.questions, key=lambda q: order[q["template"]])))
    verify(made)
    return made


def readable(about) -> set[str]:
    """The kinds of data a model is asked about: text and records always (the questions are `extracts_from`), and the pictures,
    video and sound it reads."""
    return {"text", "records"} | (set(about.reads) & {"image", "video", "audio"})


def questions_for(questions: Sequence[dict], about) -> list[dict]:
    kinds = readable(about)
    return [q for q in questions if q["type"] in kinds]


# -- what the files hold ------------------------------------------------------------------------------------------------------

def needle_moments(plan: Plan, template: str) -> list[list[tuple[float, float]]]:
    """Each needle of `template` as the spans of time its files show it in, read from the files' specs and not from the questions,
    oldest first: a sign or a car one photo, an error its state, a parcel its clips (the one it is left in, the ones between and the
    one it is taken in), a chime or an alarm its sound, a note with a code the note."""
    out: list[list[tuple[float, float]]] = []
    for f in plan.files:
        spec = f.spec
        if template == "note-code" and f.kind == "note" and spec.get("needle"):
            out.append([(f.t, f.t)])
        elif template == "photo-sign" and f.kind == "photo" and spec["scene"] == "sign" and tuple(spec["rgb"]) == NEEDLE_RGB["sign"]:
            out.append([(f.t, f.t)])
        elif template == "photo-count" and f.kind == "photo" and spec["scene"] == "car" and tuple(spec["rgb"]) == NEEDLE_RGB["car"]:
            out.append([(f.t, f.t)])
        elif template == "screen-error" and f.kind == "screen":
            out += [[(f.t + s["from"], f.t + s["to"])] for s in spec["states"] if (s["overlay"] or {}).get("kind") == "error"]
        elif template == "porch-parcel" and f.kind == "porch" and spec["parcel"]:
            a, b = spec["parcel"]
            if a > 0:  # the clip the parcel is left in
                out.append([(f.t + a, f.t + b)])
            elif out:  # the clips it is on the step in, and the one it is taken in
                out[-1].append((f.t + a, f.t + b))
            else:
                raise AssertionError(f"{f.path}: a parcel on the step that no clip left there")
        elif template in ("mic-doorbell", "mic-alarm") and f.kind == "mic":
            kind = template.split("-")[1]
            out += [[(f.t + s["at"], f.t + s["at"] + s["dur"])] for s in spec["sounds"] if s["kind"] == kind]
    return sorted(out)


def same_spans(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> bool:
    return len(a) == len(b) and all(math.isclose(x, y, abs_tol=1e-6) for u, v in zip(a, b) for x, y in zip(u, v))


def verify(plan: Plan) -> None:
    """Each media (and note) question's scope holds what its level says, counted from the files' specs: a `part` or `day` question's
    scope holds exactly its template's one needle (its moments are that needle's and it has no others); a `week` question's
    answer is the template's last needle, its `others` are all the earlier ones, and a full day of haystack or more follows it; an
    absent question's scope holds none; a count is the number of needles in its scope. An `AssertionError` names the question."""
    found: dict[str, list] = {}
    for q in plan.questions:
        base = BASE_OF.get(q["template"], q["template"])
        if q["level"] == "time" or base not in ("note-code", "photo-sign", "photo-count", "screen-error", "porch-parcel", "mic-doorbell", "mic-alarm"):
            continue
        needles = found.setdefault(base, needle_moments(plan, base))
        inside = [m for m in needles if q["since"] <= m[0][0] < q["until"]]
        who, truth = f"{q['qid']} ({q['level']})", q["truth"]
        if q["absent"]:
            if inside:
                raise AssertionError(f"{who}: an absent question's scope holds {len(inside)} needles")
        elif base == "photo-count":
            if len(inside) != int(truth["value"]) or not same_spans([s for m in inside for s in m], truth["moments"]):
                raise AssertionError(f"{who}: the scope holds {len(inside)} red cars, the answer is {truth['value']}")
        elif q["level"] in ("part", "day"):
            if len(inside) != 1 or not same_spans(inside[0], truth["moments"]) or truth["others"]:
                raise AssertionError(f"{who}: the scope holds {len(inside)} needles, not exactly the question's own")
        else:
            last = needles[-1]
            if not q["last"] or not same_spans(last, truth["moments"]) or not same_spans([s for m in needles[:-1] for s in m], truth["others"]):
                raise AssertionError(f"{who}: the answer is not the template's last needle, or its others are not all the earlier ones")
            if len(needles) < 2 and plan.scale != "tiny":
                raise AssertionError(f"{who}: the template has no earlier needle for a wrong answer")
            if q["until"] - max(b for _, b in last) < 86400:
                raise AssertionError(f"{who}: less than a day of haystack follows the last needle")
            if any(not q["since"] <= m[0][0] < q["until"] for m in needles):
                raise AssertionError(f"{who}: a needle is outside the week")


def windows_in(plan: Plan, source: str, since: float, until: float) -> int:
    """How many windows of the reader (at most: `still_s` joins frames that look the same) a source holds in [since, until) for a
    model that reads everything but documents and speech (what the decider estimate of `benchmarks/bench.py` multiplies by a model's
    seconds a window; a count from the plan, no file read): a second of the screen that has a frame (`hm.screen_frames`: a state's start,
    an act, a minute of the clock; the reader takes the first frame of each second, and there are up to 2), a second of a porch clip or a
    phone clip each one, a sound's 5 s windows every 2.5 s that the sound or the TV reaches, every photo in the range (a picture is an instant) and every note up to `until` (a version is current from its time: an ask
    reads them all), the current version of the config and those in the range, and the records' windows by the reader's own size."""
    n = 0
    if source in ("screen", "porch", "phone", "mic"):
        for f in plan.files:
            if f.source != source:
                continue
            if f.kind == "screen":
                if f.t < until and f.t + f.spec["length_s"] > since:  # (a file outside the range has no frame in it)
                    n += len({math.floor(f.t + frame.second) for frame in hm.screen_frames(f.spec, f.t) if since <= f.t + frame.second < until})
            elif f.kind == "porch":
                n += sum(since <= f.t + s < until for s in range(int(f.spec["length_s"])))
            elif f.kind == "clip":
                n += sum(since <= f.t + s < until for s in range(3))
            elif f.kind == "photo":
                n += since <= f.t < until
            elif f.kind == "mic" and f.t < until and f.t + f.spec["length_s"] > since:
                loud = [(s["at"], s["at"] + s["dur"]) for s in f.spec["sounds"]] + [tuple(span) for span in f.spec["tv"]]
                for a, b in audio_windows(f.spec["length_s"]):
                    if since <= f.t + a < until and any(a < y and b > x for x, y in loud):
                        n += 1
    elif source == "notes":
        n = sum(1 for f in plan.files if f.kind == "note" and f.t < until)
    elif source == "config":
        versions = [f.t for f in plan.files if f.kind == "config"]
        n = sum(since <= t < until for t in versions) + (1 if any(t < since for t in versions) else 0)
    elif source == LOG_SOURCE:
        lines = [line for service in plan.logs.values() for line in service
                 if since <= datetime.fromisoformat(line["ts"]).timestamp() < until]
        n = math.ceil((sum(len(json.dumps(line)) for line in lines) + 26 * len(lines)) / reader.PROFILE["records_chars"]) if lines else 0
    else:
        raise ValueError(f"a source is one of {', '.join(SOURCES)}, not {source!r}")
    return n


# -- pictures ---------------------------------------------------------------------------------------------------------

CLIP = (540, 960)


def box(size: tuple[int, int], *fractions: float) -> tuple[int, ...]:
    """Pixel corners from fractions of the picture's width and height: (x0, y0, x1, y1, ...)."""
    return tuple(int(f * size[i % 2]) for i, f in enumerate(fractions))


def wall_photo(spec: dict) -> Image.Image:
    """A facade or a hedge across 40 to 70% of the photo (`rgb`: a red wall, a green hedge), with its windows and door: more of a
    needle's named colour than the needle itself has (a sign is a quarter of the photo, a red car a third), and no needle's shade."""
    hm._clear_of_needles(spec["rgb"], "the wall's colour", "phone")
    r = random.Random(spec["seed"])
    w, h = hm.PHOTO
    image = hm._gradient(hm.PHOTO, (150, 185, 225), (215, 228, 240))
    draw = ImageDraw.Draw(image)
    width, height = r.uniform(0.75, 0.96), r.uniform(0.62, 0.86)
    left, top = r.uniform(0.02, 0.98 - width), r.uniform(0.04, 0.96 - height)
    x0, y0, x1, y1 = int(w * left), int(h * top), int(w * (left + width)), int(h * (top + height))
    draw.rectangle([x0, y0, x1, y1], fill=tuple(spec["rgb"]), outline=(90, 80, 75), width=6)
    columns = r.randint(3, 5)
    for row in range(2):
        for k in range(columns):
            wx = x0 + int((x1 - x0) * (0.06 + 0.88 * k / columns))
            wy = y0 + int((y1 - y0) * (0.08 + 0.36 * row))
            draw.rectangle([wx, wy, wx + int((x1 - x0) * 0.06), wy + int((y1 - y0) * 0.2)], fill=(205, 228, 250), outline=(245, 245, 245), width=5)
    mid = (x0 + x1) // 2
    draw.rectangle([mid - int((x1 - x0) * 0.06), y1 - int((y1 - y0) * 0.3), mid + int((x1 - x0) * 0.06), y1], fill=(90, 70, 60))
    return image


def photo_image(spec: dict) -> Image.Image:
    return wall_photo(spec) if spec["scene"] == "wall" else hm.photo(spec)


def stamp(image: Image.Image, text: str, bar: int = 60) -> Image.Image:
    """The camera's time stamp, along the bottom: it makes every clip different, and `remember` keeps a recording once."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.rectangle([0, h - bar, w, h], fill=(0, 0, 0))
    draw.text((16, h - bar + bar // 6), text, fill="white", font=font(bar * 2 // 3))
    return image


def clip_card(scene: str, seed: int, when: str) -> Image.Image:
    """A portrait frame of a phone clip: a street, a dog or a shop front."""
    r = random.Random(seed)
    image = Image.new("RGB", CLIP, (170, 200, 230))
    draw = ImageDraw.Draw(image)
    w, h = CLIP
    if scene == "street":
        draw.rectangle(box(CLIP, 0, .65, 1, 1), fill=(85, 85, 90))
        for x in (0, .35, .7):
            draw.rectangle(box(CLIP, x, .15, x + .3, .65), fill=r.choice(((210, 190, 150), (130, 150, 175), (190, 190, 195))))
        for k in range(4):
            draw.rectangle(box(CLIP, .06 + k * .25, .78, .14 + k * .25, .80), fill=(235, 235, 235))
        draw.rectangle(box(CLIP, .45, .55, .53, .75), fill=(30, 50, 110))
        draw.ellipse(box(CLIP, .45, .50, .53, .56), fill=(220, 180, 150))
    elif scene == "dog":
        draw.rectangle(box(CLIP, 0, 0, 1, .65), fill=(225, 215, 195))
        draw.rectangle(box(CLIP, 0, .65, 1, 1), fill=(175, 150, 120))
        draw.ellipse(box(CLIP, .15, .45, .85, .75), fill=(140, 100, 60))
        draw.ellipse(box(CLIP, .55, .28, .90, .52), fill=(140, 100, 60))
        draw.ellipse(box(CLIP, .55, .26, .66, .40), fill=(100, 70, 40))
        draw.ellipse(box(CLIP, .74, .34, .78, .37), fill=(20, 20, 20))
        for x in (.2, .35, .6, .72):
            draw.rectangle(box(CLIP, x, .70, x + .06, .86), fill=(120, 85, 50))
    else:
        draw.rectangle(box(CLIP, 0, 0, 1, 1), fill=(60, 70, 100))
        for k in range(8):
            draw.rectangle(box(CLIP, k / 8, .12, (k + 1) / 8, .30), fill=(30, 80, 190) if k % 2 else (245, 245, 245))
        draw.rectangle(box(CLIP, .08, .36, .92, .66), fill=(240, 230, 170))
        draw.rectangle(box(CLIP, .38, .70, .62, .96), fill=(110, 80, 50))
        draw.text((w // 2, h * 6 // 100), r.choice(("OPEN", "CAFE", "BOOKS")), fill="white", font=font(70), anchor="mm")
    return stamp(image, when)


# -- sound ------------------------------------------------------------------------------------------------------------

def audio_windows(length: float) -> list[tuple[float, float]]:
    """The windows the reader cuts a recording into: `audio_s` every `audio_hop_s`, the last one set back to the end."""
    size, hop = reader.PROFILE["audio_s"], reader.PROFILE["audio_hop_s"]
    starts = [k * hop for k in range(int(max(length - size, 0) // hop) + 1)]
    if starts[-1] + size < length - 1e-6:
        starts.append(length - size)
    return [(s, s + size) for s in starts]


# -- write ------------------------------------------------------------------------------------------------------------

def check_shades(image: Image.Image, source: str, present: str | None, where: str) -> None:
    """The picture (as a decider reads it) has each needle shade of `source` in the share the fake decider needs: the one that is
    `present` NEED_SHARE or more (and under MOST_SHARE: no needle is its frame's dominant colour), no other OTHER_SHARE."""
    for kind in hm.SOURCE_NEEDLES[source]:
        share = shade_share(image, NEEDLE_RGB[kind])
        if kind == present:
            if not NEED_SHARE <= share < MOST_SHARE:
                raise AssertionError(f"{where}: {share:.1%} of the {kind} shade, not {NEED_SHARE:.0%} to {MOST_SHARE:.0%}")
        elif share >= OTHER_SHARE:
            raise AssertionError(f"{where}: {share:.2%} of the {kind} shade, which only a needle has")


def photo_needle(spec: dict) -> str | None:
    """The needle a photo spec is: `sign`, `car`, or None."""
    if spec["scene"] in ("sign", "car") and tuple(spec["rgb"] or ()) == NEEDLE_RGB[spec["scene"]]:
        return spec["scene"]
    return None


def decoded(path: Path, wanted: set[int] | None = None) -> dict[int, Image.Image]:
    """The frames of a video as the reader hands them to a decider (scaled to its `max_px`), by index (frame k is the k-th of the file); the
    `wanted` ones only."""
    times = reader._times(path)
    pick = [(pts, k) for k, (pts, _) in enumerate(times) if wanted is None or k in wanted]
    return {k: image for k, image in reader._decode(path, pick, reader.PROFILE["max_px"])}


def check_screen(f: File, path: Path) -> None:
    """The file has the frames `hm.screen_frames` lists, at their seconds; in the first frame of every state with a dialog or an accent and
    of every 4th, and in all of an error's (the cursor wherever it went), the needle's shade is where the state says; and a frame with
    the cursor on the header of an error dialog still has it."""
    states, frames = f.spec["states"], hm.screen_frames(f.spec, f.t)
    times = reader._times(path)
    if len(times) != len(frames):
        raise AssertionError(f"{f.path}: {len(times)} frames, {len(frames)} in its spec ({len(states)} states)")
    if any(not math.isclose(seconds, frame.second, abs_tol=1e-3) for (_, seconds), frame in zip(times, frames)):
        raise AssertionError(f"{f.path}: its frames are not at the seconds of its spec")
    firsts = {frame.state: i for i, frame in reversed(list(enumerate(frames)))}
    pick = set()
    for i, frame in enumerate(frames):
        s = states[frame.state]
        if (s["overlay"] or {}).get("kind") == "error" or (firsts[frame.state] == i and (s["overlay"] or s["accent"] or frame.state % 4 == 0)):
            pick.add(i)
    for i, image in decoded(path, pick).items():
        s = states[frames[i].state]
        check_shades(image, "screen", "error" if (s["overlay"] or {}).get("kind") == "error" else None,
                     f"{f.path}: the frame at {frames[i].second} s, in the state at {s['from']} s")
    for s in states:
        if (s["overlay"] or {}).get("kind") == "error":  # the cursor on the dialog's header, as a frame at the size a decider reads
            w, h = hm.SCREEN
            cx, cy = s["overlay"].get("at") or (0.5, 0.5)
            on_header = hm.ScreenFrame(0, 0, (round(w * cx), round(h * (cy - hm.DIALOG[1] / 2 + 0.05))), "00:00", "2026-01-01")
            image = hm.screen_frame(hm.screen_layers(s), on_header).resize(reader.fit(w, h, reader.PROFILE["max_px"]), Image.Resampling.BICUBIC)
            check_shades(image, "screen", "error", f"{f.path}: the error with the cursor on its header")
            break


def check_porch(f: File, path: Path) -> None:
    length, parcel = int(f.spec["length_s"]), f.spec["parcel"]
    if len(reader._times(path)) != length:
        raise AssertionError(f"{f.path}: {len(reader._times(path))} frames, a clip of {length} s")
    pick = set(range(length)) if parcel or any(e["kind"] in ("courier", "owner_pickup") for e in f.spec["events"]) else set(range(0, length, 4))
    for k, image in decoded(path, pick).items():
        check_shades(image, "porch", "parcel" if parcel and parcel[0] <= k < parcel[1] else None, f"{f.path}: second {k}")


def check_sounds(f: File, samples: np.ndarray) -> None:
    """Over the reader's windows of a file with a needle (decoded from it), the needle is heard in a window that overlaps it (the fake
    decider's `pitch:<hz>`: the doorbell's first note, the alarm's pitch), and no window that overlaps none of its kind hears the
    pitch (of either note of the chime)."""
    needles: dict[tuple[str, tuple[float, ...]], list[tuple[float, float]]] = {}
    for s in f.spec["sounds"]:
        if s["kind"] in ("doorbell", "alarm"):
            pitches = tuple(s["notes"]) if s["kind"] == "doorbell" else (float(s["hz"]),)
            needles.setdefault((s["kind"], pitches), []).append((s["at"], s["at"] + s["dur"]))
    heard = {(key, span): False for key, spans in needles.items() for span in spans}
    for a, b in audio_windows(f.spec["length_s"]):
        piece = samples[int(round(a * RATE)):int(round(b * RATE))]
        if float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)))) < PITCH_RMS:
            continue
        for key, spans in needles.items():
            kind, pitches = key
            overlapping = [span for span in spans if a < span[1] and b > span[0]]
            if not overlapping:
                for hz in pitches:
                    if pitch_share(piece, RATE, hz) >= PITCH_SHARE:
                        raise AssertionError(f"{f.path}: the window {a:g} to {b:g} s sounds like {hz:g} Hz and overlaps no such {kind}")
            elif pitch_share(piece, RATE, pitches[0]) >= PITCH_SHARE:
                for span in overlapping:
                    heard[key, span] = True
    for (key, span), yes in heard.items():
        if not yes:
            raise AssertionError(f"{f.path}: the {key[0]} at {span[0]:g} s ({key[1][0]:g} Hz) is heard in no window")


def exif_of(t: float) -> Image.Exif:
    """The EXIF of a phone's photo taken at `t` (local epoch): DateTimeOriginal as the wall clock then, and OffsetTimeOriginal, the local
    offset at that moment (`+01:00`), as a phone writes them. Both are what `mms remember` reads a picture's time from."""
    when = datetime.fromtimestamp(t)
    zone = when.astimezone().strftime("%z")
    exif = Image.Exif()
    inner = exif.get_ifd(ExifTags.IFD.Exif)
    inner[ExifTags.Base.DateTimeOriginal] = when.strftime("%Y:%m:%d %H:%M:%S")
    inner[ExifTags.Base.OffsetTimeOriginal] = f"{zone[:3]}:{zone[3:]}"
    return exif


def write_file(f: File, folder: Path) -> None:
    """One file of the week, checked as it is made; its mtime is its time. A photo says its time in its EXIF and a phone clip in its
    `creation_time` (UTC), as a phone's files do."""
    path = folder / f.path
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = f.spec
    stamped = datetime.fromtimestamp(f.t).strftime("%Y-%m-%d %H:%M:%S")
    with tempfile.TemporaryDirectory(prefix="haystack-") as work_dir:
        work = Path(work_dir)
        if f.kind == "photo":
            photo_image(spec).save(path, quality=90, exif=exif_of(f.t))
            with Image.open(path) as saved:  # as the reader hands it to a decider: the JPEG decoded at 1/2
                saved.draft("RGB", (640, 360))
                check_shades(saved.convert("RGB"), "phone", photo_needle(spec), f.path)
        elif f.kind == "clip":
            card = clip_card(spec["scene"], spec["seed"], stamped)
            check_shades(card, "phone", None, f.path)
            made = f"creation_time={datetime.fromtimestamp(f.t, timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
            portrait_video(path, card, work, seconds=3, tags=("-metadata", made))
        elif f.kind == "screen":
            hm.write_screen(path, spec, f.t, work)
            check_screen(f, path)
        elif f.kind == "porch":
            hm.write_porch(path, spec, stamped, work)
            check_porch(f, path)
        elif f.kind == "mic":
            hm.write_mic(path, spec, work)
            if any(s["kind"] in ("doorbell", "alarm") for s in spec["sounds"]):
                check_sounds(f, reader._span(path, 0, 1e9, RATE)[1])
        else:  # note, config
            path.write_bytes(spec["text"].encode("utf-8"))
    os.utime(path, (f.t, f.t))


def count(token: str, text: str) -> int:
    """How often `token` is in `text` with no digit beside it."""
    return len(re.findall(rf"(?<!\d){re.escape(token)}(?!\d)", text))


def check_text(plan: Plan, folder: Path) -> None:
    """Each needle's code or value is in its own file or lines and nowhere else, read back from what was written."""
    def read(*parts: str) -> str:
        return folder.joinpath(*parts).read_text(encoding="utf-8")

    notes = {f.path: read(f.path) for f in plan.files if f.kind == "note"}
    configs = {f.path: read(f.path) for f in plan.files if f.kind == "config"}
    log_text = "\n".join(read("logs", f"{service}.jsonl") for service in plan.logs)
    everything = "\n".join([*notes.values(), *configs.values(), log_text])
    for q in plan.questions:
        if q["template"] == "note-code":
            code = q["truth"]["value"]
            if count(code, everything) != 1 or sum(code in text for text in notes.values()) != 1:
                raise AssertionError(f"{q['qid']}: the code {code} is in {count(code, everything)} places, not in one note only")
        elif q["template"] == "absent-note" and re.search(r"wi-?fi|password", "\n".join(notes.values()), re.I):
            raise AssertionError(f"{q['qid']}: a note mentions a Wi-Fi password")
        elif q["template"] in ("log-count", "absent-log"):
            code = q["fake"]["question"].split(":", 1)[1]
            lines = sum(code in line for line in log_text.splitlines())
            if count(code[4:], everything) != lines:  # no other line, note or version has the code's digits as a number
                raise AssertionError(f"{q['qid']}: the digits of {code} are in {count(code[4:], everything)} places, the code in {lines} lines")
            others = lines - len(q["truth"]["moments"])  # the lines on other days
            if (lines if q["absent"] else not 1 <= others <= 3):
                raise AssertionError(f"{q['qid']}: {code} is on {lines} lines, {others} of them outside the slot")
    for value in (f.spec["retention_days"] for f in plan.files if f.kind == "config"):
        if sum(text.count(f"retention_days = {value}\n") for text in configs.values()) != 1:
            raise AssertionError(f"retention_days = {value} is not in exactly one version of app.toml")


def write(plan: Plan, folder: Path) -> None:
    """Write every file and log of the plan under `folder`, set their mtimes, and assert the plan's own checks (each needle is
    what the fake decider sees after encoding, no distractor is, each needle's code is in its own file only, no two recordings are
    the same bytes, which `remember` would keep once: photos too, which are recordings now). Nothing of the truth is written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        for future in [pool.submit(write_file, f, folder) for f in plan.files]:
            future.result()
    (folder / "logs").mkdir(exist_ok=True)
    for service, lines in plan.logs.items():
        path = folder / "logs" / f"{service}.jsonl"
        path.write_bytes("".join(json.dumps(line) + "\n" for line in lines).encode("utf-8"))
        if lines:
            last = datetime.fromisoformat(lines[-1]["ts"]).timestamp()
            os.utime(path, (last, last))
    check_text(plan, folder)
    seen: dict[tuple[str, str], str] = {}
    for f in plan.files:
        if f.kind in ("photo", "clip", "screen", "porch", "mic"):
            key = (f.source, hashlib.sha256((folder / f.path).read_bytes()).hexdigest())
            if key in seen:
                raise AssertionError(f"{f.path} has the bytes of {seen[key]}: remember would keep one")
            seen[key] = f.path


LOAD_WORKERS = 3  # sources remembered at once: the daemon serves `remember` calls together (2.6 times faster than one at a time)


def load(api, folder: Path, plan: Plan, wait: float = 120) -> None:
    """Remember every file (no caller, so every row is ambient), add the log source, and wait until it holds every line. Each is
    given its time `t`, except the phone's photos and clips: those carry their own (EXIF, `creation_time`), as a camera roll does, and
    must come back stored at it, `time_from` `metadata`. A source's files go oldest first in one thread (the versions of app.toml need
    the order), the sources `LOAD_WORKERS` at a time, the biggest first."""
    from conftest import wait_for  # the tests' own helper

    def remember(files: list[File]) -> None:
        for f in files:
            given = {} if f.source == PHONE else {"t": f.t}
            out = api.call("remember", path=str(Path(folder) / f.path), source=f.source, **given, **({"name": f.name} if f.name else {}))
            if len(out["stored"]) != 1 or out["unchanged"]:
                raise AssertionError(f"{f.path} was not stored once: {out}")
            if f.source == PHONE:
                entry = out["stored"][0]
                if entry.get("time_from") != "metadata" or abs(datetime.fromisoformat(entry["t0"]).timestamp() - f.t) > 1:
                    raise AssertionError(f"{f.path} was stored at {entry['t0']} from {entry.get('time_from')}, not at its own time {f.t} from its metadata")

    by_source: dict[str, list[File]] = {}
    for f in plan.files:
        by_source.setdefault(f.source, []).append(f)
    with ThreadPoolExecutor(LOAD_WORKERS) as pool:
        for future in [pool.submit(remember, files) for files in sorted(by_source.values(), key=len, reverse=True)]:
            future.result()
    api.call("source_set", name=LOG_SOURCE, kind="log", path=str(Path(folder) / "logs"), glob="*.jsonl", time_field="ts", max_gb=1)
    lines = sum(len(v) for v in plan.logs.values())

    def records() -> bool:
        found = next((s for s in api.call("status")["sources"] if s["name"] == LOG_SOURCE), None)
        return found is not None and found["records"] >= lines

    wait_for(records, timeout=wait, what=f"the {lines} records of the log source")
