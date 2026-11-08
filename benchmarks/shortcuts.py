"""The shortcut probe of the benchmark haystack (TESTBED.md, benchmark): offline, from the plan and the files `haystack.write` made, it
checks that a cheap shortcut does not pick out a needle, so that the files arm cannot skip the work an `ask` does.

    rows = shortcuts.probe(plan, folder)             # one row per media question, from the data folder (read only)
    failures = shortcuts.check(rows, plan.scale)     # what a floor of the scale is not met by, each as a sentence
    failures = shortcuts.run(plan, folder, out)      # both, plus the ffmpeg check, and shortcuts.json and shortcuts.md in `out`

For each media question (a question whose `probe` is `{"kind": "shade", "hue", "rgb"}` or `{"kind": "pitch", "hz", "band"}`; text and
records have none) it measures, over the question's scope (`since` to `until`; for a `last` question, from the end of the template's
last needle, which is what a scan from the newest has to read before it hits one), what the shortcuts the smoke's files arm used
give an agent:
- candidates: the screen states, porch clips, photos or sounds of the source in scope (`FLOORS` has the least of each, by scale);
- colour shortlist: the other frames or photos whose share of pixels in the needle's hue band (HSV, as a filter for "red", "green" or
  "brown" would cut it, at 160x90) is at least half the needle's (the median over the needle's frames);
- change shortlist (porch): the other frames whose mean difference from the median frame of their day is at least half the needle's;
- band shortlist (sound): the other sounds whose dominant pitch is in the question's band with a level (the peak RMS over half
  seconds) of at least a quarter of the needle's. Each sound is synthesised alone from the plan's spec (`haystack_media.sound`), so
  hours of sound are never decoded. (A TV stretch is no sound of the spec's `sounds`, and is not counted);
- size: the z-score of the needle's file size among the source's files in scope, and whether it is the largest or smallest with
  |z| > 2.5 (`ls -S` would find it);
- colour in the needle's area (screen error and porch parcel): the same hue band, counted only inside the area where the needle can be
  drawn (`area_of`: `haystack_media.DIALOG_AREA` or `BOX_AREA`, the union of the positions it can take, else the one place `DIALOG` and
  `BOX` give), which an agent can crop from the start, with no needle found first: its shortlist is held to the colour shortlist's floors;
- brute-force views: the pictures in scope over 25, the 5x5 montages that look at all of them (`views`; for the screen a picture is a
  state, and `recorded_frames` and `recorded_views` count the frames the recorder keeps of those states, the files arm's brute force; for the porch, whose needle
  stays on the step, `views_per_clip` is a montage of one frame a clip).
Each shortlist has a `rank` beside it: 1 plus the others with more than the needle's value (1: sorting by the filter's measure puts
the needle first). On `pilot` and `standard` a rank of 1 at day or week level fails a question, as the floors on the shortlists do (the area's too). An
absent question has no needle, so it has the candidates and the views only.
Report only, never a floor and not in `check`: the tuned filters of an agent that has found one needle and lists the rest. `tuned_colour_*`
keeps the needle's own hue, saturation and value within `TUNE`; `region_colour_*` is the same colour inside that area (`area_of`); `tuned_band_*` keeps the sounds within `NARROW` of the needle's dominant pitch. Each has the
needle's value, the others it keeps at half of it (a quarter of the level for sound), and the needle's rank. A frame is placed from the file's spec (a porch frame at its second, a
screen frame by `haystack_media.screen_frames`, which is what the writer put there): a file with another number of frames is a `ValueError`. A screen state is
one candidate however many frames the owner's acts (cursor moves, typing, scrolling) and the clock give it (the floors count states), and its
measures are the median of its frames; a dialog, an accent and the taskbar never move, so the median holds what the state shows. The files arm
sees every frame, up to 2 a second (`recorded_frames`, `recorded_views`).
The files arm's ffmpeg is checked too: it must have no `ocr` filter (`filters`, `ocr_failures`)."""

from __future__ import annotations

import colorsys
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import haystack_media as hm
import numpy as np
from PIL import Image

# day is the tests' own, and a script run from benchmarks/ has only benchmarks/ on its path
TESTS_DIR = str(Path(__file__).resolve().parents[1] / "tests")
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)  # last, so no file of the tests' shadows a module of the standard library or the env

from day import RATE  # noqa: E402

WORKERS = 3  # files read at once
PICTURE = (160, 90)  # width and height frames and photos are read at
VIEW = 25  # pictures in one 5x5 montage
HALF = 0.5  # a colour or change shortlist holds the others with this share of the needle's value, or more
BAND_RMS = 0.25  # a band shortlist holds the other sounds with this share of the needle's level, or more
SIZE_Z = 2.5  # the needle's file is a shortcut when it is the largest or smallest of the scope with a z-score beyond this
SLACK_S = 0.5  # a picture is the needle's when it is within this of its moment; a moment that is a point is this long
WINDOW_S = 0.5  # the sound level is the peak RMS over windows of this length
TUNE = (10.0, 0.10, 0.10)  # a tuned colour filter keeps the needle's own hue (degrees), saturation and value, plus or minus this much
NARROW = 0.08  # a tuned band filter keeps this share either side of the needle's dominant pitch
NO_OCR = ("ocr",)  # the filters the files arm's ffmpeg must not have: they read text off a picture
LEVELS = ("part", "day", "week")
SOURCES = {"screen": ("screen", "screen states"), "porch": ("porch", "porch clips"), "phone": ("photo", "photos"), "mic": ("mic", "sounds")}  # source: file kind, the unit of its candidates
CANDIDATES = {"part": {"screen": 250, "porch": 8, "phone": 15, "mic": 5},
              "day": {"screen": 600, "porch": 25, "phone": 50, "mic": 15},
              "week": {"screen": 1500, "porch": 60, "phone": 150, "mic": 30}}  # `standard`: the least candidates of a scope, by level and source
SHORTLIST = {"part": 5, "day": 15, "week": 30}  # `standard`: the least other frames in a colour or change shortlist
BAND = {"part": 3, "day": 8, "week": 15}  # `standard`: the least other sounds in a band shortlist
FLOORS = {  # by scale; `tiny` needs one of each. The pilot keeps about a day after its last needle, so its week level is lower:
    "standard": {"candidates": CANDIDATES, "shortlist": SHORTLIST, "band": BAND},  # the plan's
    "pilot": {"candidates": {**CANDIDATES, "week": {**CANDIDATES["week"], "screen": 1300, "porch": 50, "phone": 90}},  # (measured: 1,409 to 1,595, 53 to 56, 101 to 118)
              "shortlist": {**SHORTLIST, "week": 25}, "band": BAND}}  # (a photo shortlist of 30 to 37)
RANK_LEVELS = ("day", "week")  # the levels at which a needle that a filter ranks first is a failure (on `pilot` and `standard`)
BANDS = {  # what a filter for the colour's name keeps: hue in degrees (lo, hi; wraps when lo > hi), saturation and value ranges
    "red": ((345, 15), (0.40, 1.0), (0.35, 1.0)),
    "green": ((75, 165), (0.35, 1.0), (0.30, 1.0)),
    "brown": ((15, 45), (0.35, 0.85), (0.25, 0.85))}


# -- the pictures -----------------------------------------------------------------------------------------------------------

def _hsv(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue (degrees), saturation and value (0 to 1) of each pixel of `frames` (n, height, width, 3) uint8."""
    x = frames.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    value, low = x.max(axis=-1), x.min(axis=-1)
    span = value - low
    safe = np.where(span > 0, span, 1.0)
    hue = np.where(value == r, ((g - b) / safe) % 6, np.where(value == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60.0
    return np.where(span > 0, hue, 0.0), np.where(value > 0, span / np.where(value > 0, value, 1.0), 0.0), value


def _band(h: np.ndarray, s: np.ndarray, v: np.ndarray, hue: str) -> np.ndarray:
    (h0, h1), (s0, s1), (v0, v1) = BANDS[hue]
    in_hue = (h >= h0) & (h <= h1) if h0 <= h1 else (h >= h0) | (h <= h1)
    return in_hue & (s >= s0) & (s <= s1) & (v >= v0) & (v <= v1)


def band_share(frames: np.ndarray, hue: str) -> np.ndarray:
    """The share of each frame's pixels (n, height, width, 3 uint8) that a filter for the colour `hue` ("red", "green" or "brown") keeps."""
    return _band(*_hsv(frames), hue).mean(axis=(1, 2))


def needle_hsv(rgb) -> tuple[float, float, float]:
    """Hue (degrees), saturation and value (0 to 1) of the colour `rgb`."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
    return h * 360.0, s, v


def _tuned(h: np.ndarray, s: np.ndarray, v: np.ndarray, rgb) -> np.ndarray:
    th, ts, tv = needle_hsv(rgb)
    return (np.abs((h - th + 180.0) % 360.0 - 180.0) <= TUNE[0]) & (np.abs(s - ts) <= TUNE[1]) & (np.abs(v - tv) <= TUNE[2])


def tuned_share(frames: np.ndarray, rgb) -> np.ndarray:
    """The share of each frame's pixels (n, height, width, 3 uint8) within TUNE of the colour `rgb` in hue, saturation and value: a filter
    tuned to a needle that has been found."""
    return _tuned(*_hsv(frames), rgb).mean(axis=(1, 2))


def region_of(rgb) -> tuple[float, float, float, float] | None:
    """Where the needle of the shade `rgb` is drawn when it is always in one place, as fractions of the picture `(x0, y0, x1, y1)`: the
    error dialog's box, the parcel's front face; none for a photo's needle, which is anywhere."""
    rgb = tuple(int(c) for c in rgb)
    if rgb == tuple(hm.NEEDLE_RGB["error"]):
        w, h = hm.DIALOG
        return (1 - w) / 2, (1 - h) / 2, (1 + w) / 2, (1 + h) / 2
    if rgb == tuple(hm.NEEDLE_RGB["parcel"]):
        return tuple(hm.BOX)
    return None


def area_of(rgb) -> tuple[float, float, float, float] | None:
    """The area where the needle of the shade `rgb` can appear, as fractions of the picture `(x0, y0, x1, y1)`: `haystack_media.DIALOG_AREA`
    or `BOX_AREA` (the union of every position a needle can take) when the module has them, else `region_of` (the one place it is drawn)."""
    fixed = region_of(rgb)
    if fixed is None:
        return None
    named = getattr(hm, "DIALOG_AREA" if tuple(int(c) for c in rgb) == tuple(hm.NEEDLE_RGB["error"]) else "BOX_AREA", None)
    return tuple(named) if named else fixed


def area_name(rgb) -> str:
    """What the area of the shade `rgb` is the area of, for a sentence."""
    return "dialog" if tuple(int(c) for c in rgb) == tuple(hm.NEEDLE_RGB["error"]) else "parcel"


def region_share(frames: np.ndarray, rgb) -> np.ndarray | None:
    """`tuned_share`, but of the pixels in `area_of(rgb)` only; None where the shade has no area."""
    area = area_of(rgb)
    if area is None:
        return None
    return _region(_tuned(*_hsv(frames), rgb), area)


def area_share(frames: np.ndarray, rgb, hue: str) -> np.ndarray | None:
    """The share of the pixels in `area_of(rgb)` that a filter for the colour `hue` keeps (the broad band, as `band_share`): the
    crop-and-filter that needs no needle found first. None where the shade has no area."""
    area = area_of(rgb)
    if area is None:
        return None
    return _region(_band(*_hsv(frames), hue), area)


def _region(mask: np.ndarray, region) -> np.ndarray:
    height, width = mask.shape[1:]
    x0, y0, x1, y1 = round(region[0] * width), round(region[1] * height), round(region[2] * width), round(region[3] * height)
    return mask[:, y0:y1, x0:x1].mean(axis=(1, 2))


def _key(rgb) -> str:
    return ",".join(str(int(c)) for c in rgb)


def _measures(frames: np.ndarray, targets) -> dict[str, np.ndarray]:
    """Per frame: the share in each hue band (by the hue's name) and, for each needle `(shade, hue)` of `targets`, the share within TUNE of
    the shade (`tuned:R,G,B`), where the shade has an area the share of that area alone (`region:R,G,B`), and the share of the area that
    the hue's broad band keeps (`region_broad:<hue>:R,G,B`)."""
    h, s, v = _hsv(frames)
    out = {hue: _band(h, s, v, hue).mean(axis=(1, 2)) for hue in BANDS}
    for rgb, hue in targets:
        key, area = _key(rgb), area_of(rgb)
        if f"tuned:{key}" not in out:
            mask = _tuned(h, s, v, rgb)
            out[f"tuned:{key}"] = mask.mean(axis=(1, 2))
            if area is not None:
                out[f"region:{key}"] = _region(mask, area)
        if area is not None and hue in BANDS:
            out[f"region_broad:{hue}:{key}"] = _region(_band(h, s, v, hue), area)
    return out


def _video(path: Path) -> np.ndarray:
    """Every frame of a video at PICTURE, RGB: (frames, height, width, 3) uint8."""
    with av.open(str(path)) as container:
        frames = [f.to_ndarray(width=PICTURE[0], height=PICTURE[1], format="rgb24", interpolation="AREA") for f in container.decode(container.streams.video[0])]
    return np.stack(frames) if frames else np.zeros((0, PICTURE[1], PICTURE[0], 3), np.uint8)


def _photo(path: Path) -> np.ndarray:
    """A photo at PICTURE, RGB: (1, height, width, 3) uint8. (A JPEG of 1280x720 is decoded at 1/8, which is that size.)"""
    with Image.open(path) as image:
        image.draft("RGB", PICTURE)
        image = image.convert("RGB")
        if image.size != PICTURE:
            image = image.resize(PICTURE, Image.Resampling.BOX)
        return np.asarray(image)[None]


def _read(f, folder: Path, keep: bool, targets=()):
    """(the second of the file each picture of `f` is at, measure -> its value for each (see `_measures`), the frames when `keep`, the
    recorded frames each picture stands for: 1, but a screen state's are as many as the recorder keeps of it)."""
    path = folder / f.path
    if f.kind == "photo":
        frames, seconds = _photo(path), [0.0]
    elif f.kind == "screen":  # a state is a picture: each measure is the median of the frames of the state (the cursor and the clock move, the state does not)
        states = f.spec.get("states")
        if states is None:
            raise ValueError(f"{f.path}: the spec of a screen file has `states`, to place its frames")
        frames, shown = _video(path), hm.screen_frames(f.spec, f.t)
        if len(frames) != len(shown):
            raise ValueError(f"{f.path}: {len(frames)} frames were decoded and its spec has {len(shown)}")
        cuts = np.flatnonzero(np.diff([frame.state for frame in shown])) + 1
        by_state = {key: np.array([np.median(part) for part in np.split(value, cuts)]) for key, value in _measures(frames, targets).items()}
        return [float(s["from"]) for s in states], by_state, None, np.bincount([frame.state for frame in shown], minlength=len(states))
    else:
        frames = _video(path)
        seconds = [float(k) for k in range(int(f.spec["length_s"]))]
        if len(frames) != len(seconds):
            raise ValueError(f"{f.path}: {len(frames)} frames were decoded and its spec has {len(seconds)}")
    return seconds, _measures(frames, targets), frames if keep else None, np.ones(len(seconds), np.int64)


def _change(frames: np.ndarray) -> np.ndarray:
    """The mean absolute difference (0 to 255) of each frame from the median frame of `frames` (n, height, width, 3 uint8)."""
    median = np.empty(frames.shape[1:], np.float32)
    for y in range(0, frames.shape[1], 10):  # in strips, so that the median's working copy stays small
        median[y:y + 10] = np.median(frames[:, y:y + 10], axis=0)
    return np.abs(frames.astype(np.int16) - median.astype(np.int16)).mean(axis=(1, 2, 3))


@dataclass
class Items:
    """The pictures or sounds of one source, oldest first. `t` is where each starts (local epoch) and `end` where it ends (a
    picture's end is its start), `file` the index of its file in `files`. Pictures have `share` (see `_measures`: a hue's name, or
    `tuned:R,G,B`, or `region:R,G,B` -> the share of each picture) and, for the porch, `change`; sounds have `rms` and `hz`."""

    files: list
    t: np.ndarray
    end: np.ndarray
    file: np.ndarray
    share: dict[str, np.ndarray] | None = None
    change: np.ndarray | None = None
    rms: np.ndarray | None = None
    hz: np.ndarray | None = None
    recorded: np.ndarray | None = None  # pictures: the frames of the recording each stands for (a screen state's, else 1)


def _pictures(files: list, folder: Path, pool: ThreadPoolExecutor, porch: bool, targets=()) -> Items:
    """The pictures of `files` (one source's, oldest first). A porch day is read whole: its pictures are measured against its median frame.
    `targets` are the needle shades the tuned measures are taken for."""
    ts, fs, shares, changes, recorded = [], [], {}, [], []
    days: dict = {}
    for i, f in enumerate(files):
        days.setdefault(datetime.fromtimestamp(f.t).date() if porch else None, []).append(i)
    for indices in days.values():
        got = list(pool.map(lambda i: _read(files[i], folder, porch, targets), indices))
        for i, (seconds, share, _, n) in zip(indices, got):
            ts += [files[i].t + s for s in seconds]
            fs += [i] * len(seconds)
            recorded.append(n)
            for key, value in share.items():
                shares.setdefault(key, []).append(value)
        if porch:
            changes.append(_change(np.concatenate([frames for _, _, frames, _ in got])))
    t = np.array(ts, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    out = lambda parts: np.concatenate(parts)[order] if parts else np.zeros(0)  # noqa: E731
    return Items(files, t[order], t[order], np.array(fs, dtype=np.int64)[order], {key: out(parts) for key, parts in shares.items()},
                 out(changes) if porch else None, recorded=out(recorded))


# -- the sounds -------------------------------------------------------------------------------------------------------------

def event_level(spec: dict) -> tuple[float, float]:
    """(peak RMS over windows of WINDOW_S, dominant pitch in Hz) of one sound of a mic segment's `sounds`, synthesised alone. The
    dominant pitch is the strongest bin of the whole sound's spectrum (as the fake decider's `pitch_share` takes it), DC left out."""
    x = hm.sound(spec).astype(np.float64)
    size = max(1, int(WINDOW_S * RATE))
    if len(x) <= size:
        rms = math.sqrt(float(np.mean(x * x))) if len(x) else 0.0
    else:
        cumulative = np.concatenate([[0.0], np.cumsum(x * x)])
        starts = np.arange(0, len(x) - size + 1, size // 2)
        rms = math.sqrt(float(((cumulative[starts + size] - cumulative[starts]) / size).max()))
    if len(x) < 2:
        return rms, 0.0
    spectrum = np.abs(np.fft.rfft(x))
    spectrum[0] = 0.0
    return rms, float(spectrum.argmax() * RATE / len(x))


def _sounds(files: list, pool: ThreadPoolExecutor) -> Items:
    """The sounds (`sounds` of each mic segment's spec) of `files`, oldest first."""
    def one(f):
        return [(f.t + s["at"], f.t + s["at"] + s["dur"], *event_level(s)) for s in f.spec.get("sounds", ())]

    ts, ends, fs, rms, hz = [], [], [], [], []
    for i, events in enumerate(pool.map(one, files)):
        for t, end, level, pitch in events:
            ts.append(t), ends.append(end), fs.append(i), rms.append(level), hz.append(pitch)
    order = np.argsort(ts, kind="stable")
    array = lambda values, dtype=np.float64: np.array(values, dtype=dtype)[order]  # noqa: E731
    return Items(files, array(ts), array(ends), array(fs, np.int64), rms=array(rms), hz=array(hz))


# -- the rows ---------------------------------------------------------------------------------------------------------------

def _level(q: dict) -> str:
    """The question's `level`, or (for a plan that has none) what its range makes it."""
    if q.get("level"):
        return q["level"]
    span = float(q["until"]) - float(q["since"])
    return "part" if span <= 12 * 3600 + 1 else "day" if span <= 24 * 3600 + 1 else "week"


def _clock(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _after(a: float, b: float) -> float:
    """Where a moment `[a, b]` ends: `b`, or a little after `a` when it is a point."""
    return b if b > a else a + SLACK_S


def _row(q: dict, items: Items, sizes: dict[str, int]) -> dict:
    source, level, probe = q["source"], _level(q), q["probe"]
    unit = SOURCES[source][1]
    moments = [(float(a), float(b)) for a, b in (q.get("truth") or {}).get("moments") or ()]
    absent, last = bool(q.get("absent")), bool(q.get("last"))
    problems: list[str] = []
    if probe.get("kind") != ("pitch" if source == "mic" else "shade"):
        problems.append(f"the {source} source has a {'pitch' if source == 'mic' else 'shade'} probe, not {probe.get('kind')!r}")
    lo, hi = float(q["since"]), float(q["until"])
    if last and moments:  # from the newest, a scan reads what comes after the last needle before it hits one
        lo = max(lo, max(_after(a, b) for a, b in moments))
    needle = np.zeros(len(items.t), bool)
    for a, b in moments:
        needle |= (items.end > a - SLACK_S) & (items.t < _after(a, b))
    scope = (items.t >= lo) & (items.t < hi)
    others = scope & ~needle
    if moments and not needle.any():
        problems.append(f"no {unit.rstrip('s')} of the plan is at the needle's moment {_clock(moments[0][0])}")
    if not absent and not moments:
        problems.append("the question is no absent one and has no moment")
    pictures = items.share is not None
    row: dict = {"qid": q["qid"], "template": q.get("template"), "level": level, "source": source, "absent": absent, "last": last,
                 "scope": [_clock(lo), _clock(hi)], "unit": unit, "needles": int(needle.sum()),
                 "frames": int(scope.sum()) if pictures else None,
                 "candidates": len({int(i) for i in items.file[scope]}) if source == "porch" else int(scope.sum()),
                 "views": math.ceil(int(scope.sum()) / VIEW) if pictures else None,
                 "recorded_frames": int(items.recorded[scope].sum()) if source == "screen" else None,
                 "recorded_views": math.ceil(int(items.recorded[scope].sum()) / VIEW) if source == "screen" else None,
                 "views_per_clip": math.ceil(len({int(i) for i in items.file[scope]}) / VIEW) if source == "porch" else None,
                 "probe": probe, "colour_needle": None, "colour_shortlist": None, "colour_rank": None, "change_needle": None, "change_shortlist": None,
                 "change_rank": None, "band_needle_rms": None, "band_needle_hz": None, "band_shortlist": None, "band_rank": None,
                 "tuned_colour_hsv": None, "tuned_colour_needle": None, "tuned_colour_shortlist": None, "tuned_colour_rank": None,
                 "region_colour_needle": None, "region_colour_shortlist": None, "region_colour_rank": None,
                 "region_area": None, "region_broad_needle": None, "region_broad_shortlist": None, "region_broad_rank": None,
                 "tuned_band": None, "tuned_band_shortlist": None, "tuned_band_rank": None,
                 "size_z": None, "size_rank": None, "size_extreme": False, "size_files": None}
    if needle.any() and not absent:
        if pictures and probe.get("kind") == "shade":
            hue = probe.get("hue")
            if hue not in BANDS:
                problems.append(f"the hue {hue!r} is none of {', '.join(BANDS)}")
            else:
                share = items.share[hue]
                row["colour_needle"] = float(np.median(share[needle]))
                if row["colour_needle"] <= 0:
                    problems.append(f"the needle has no pixel in the {hue} band: a {hue} filter would not find it")
                else:
                    row["colour_shortlist"] = int((share[others] >= HALF * row["colour_needle"]).sum())
                    row["colour_rank"] = 1 + int((share[others] > row["colour_needle"]).sum())
            if probe.get("rgb") and hue in BANDS and area_of(probe["rgb"]) is not None:  # a broad filter on the area a needle can be in: no needle found first
                row["region_area"] = [round(x, 3) for x in area_of(probe["rgb"])]
                share = items.share.get(f"region_broad:{hue}:{_key(probe['rgb'])}")
                if share is not None:
                    row["region_broad_needle"] = float(np.median(share[needle]))
                    if row["region_broad_needle"] <= 0:
                        problems.append(f"the needle has no pixel in the {hue} band inside its area: cropping the {area_name(probe['rgb'])}'s area would not find it")
                    else:
                        row["region_broad_shortlist"] = int((share[others] >= HALF * row["region_broad_needle"]).sum())
                        row["region_broad_rank"] = 1 + int((share[others] > row["region_broad_needle"]).sum())
            if probe.get("rgb"):  # report only: filters tuned to the needle itself
                key = _key(probe["rgb"])
                row["tuned_colour_hsv"] = [round(x, 3) for x in needle_hsv(probe["rgb"])]
                for name, prefix in (("tuned_colour", "tuned"), ("region_colour", "region")):
                    share = items.share.get(f"{prefix}:{key}")
                    if share is not None:
                        row[f"{name}_needle"] = float(np.median(share[needle]))
                        if row[f"{name}_needle"] > 0:
                            row[f"{name}_shortlist"] = int((share[others] >= HALF * row[f"{name}_needle"]).sum())
                            row[f"{name}_rank"] = 1 + int((share[others] > row[f"{name}_needle"]).sum())
            if items.change is not None:
                row["change_needle"] = float(np.median(items.change[needle]))
                if row["change_needle"] <= 0:
                    problems.append("the needle's frames do not differ from the day's median frame")
                else:
                    row["change_shortlist"] = int((items.change[others] >= HALF * row["change_needle"]).sum())
                    row["change_rank"] = 1 + int((items.change[others] > row["change_needle"]).sum())
        elif not pictures and probe.get("kind") == "pitch":
            lo_hz, hi_hz = (float(x) for x in probe["band"])
            row["band_needle_rms"], row["band_needle_hz"] = float(np.median(items.rms[needle])), float(np.median(items.hz[needle]))
            if not lo_hz <= row["band_needle_hz"] <= hi_hz:
                problems.append(f"the needle's dominant pitch {row['band_needle_hz']:.0f} Hz is outside its band {lo_hz:g} to {hi_hz:g} Hz")
            in_band = (items.hz >= lo_hz) & (items.hz <= hi_hz) & others
            row["band_shortlist"] = int((in_band & (items.rms >= BAND_RMS * row["band_needle_rms"])).sum())
            row["band_rank"] = 1 + int((in_band & (items.rms > row["band_needle_rms"])).sum())
            near = row["band_needle_hz"] * (1 - NARROW), row["band_needle_hz"] * (1 + NARROW)  # report only: a filter tuned to the needle's pitch
            narrow = (items.hz >= near[0]) & (items.hz <= near[1]) & others
            row["tuned_band"] = [round(near[0]), round(near[1])]
            row["tuned_band_shortlist"] = int((narrow & (items.rms >= BAND_RMS * row["band_needle_rms"])).sum())
            row["tuned_band_rank"] = 1 + int((narrow & (items.rms > row["band_needle_rms"])).sum())
        in_scope = {i for i, f in enumerate(items.files) if f.t < hi and f.t + float((f.spec or {}).get("length_s", 0)) > lo}
        mine = {int(i) for i in items.file[needle]}
        names = [items.files[i].path for i in sorted(in_scope | mine)]
        if len(names) >= 2:
            values = np.array([sizes[name] for name in names], dtype=np.float64)
            z = (values - values.mean()) / values.std() if values.std() > 0 else np.zeros(len(values))
            worst = max((names.index(items.files[i].path) for i in mine), key=lambda k: abs(z[k]))
            rank = "largest" if values[worst] == values.max() else "smallest" if values[worst] == values.min() else None
            row.update(size_z=float(z[worst]), size_rank=rank, size_extreme=bool(rank and abs(z[worst]) > SIZE_Z), size_files=len(names))
    row["problems"] = problems
    return row


def probe(plan, folder, *, workers: int = WORKERS) -> list[dict]:
    """One row per media question of `plan` (those with a `probe`), measured from the files under `folder`, which is only read.
    The plan is read for `files` (`path`, `source`, `t`, `kind`, `spec`) and `questions`. See the module's docstring for the fields."""
    folder = Path(folder)
    questions = [q for q in plan.questions if q.get("probe")]
    for q in questions:
        if q["source"] not in SOURCES:
            raise ValueError(f"{q['qid']}: the source {q['source']!r} is none of {', '.join(SOURCES)}")
    index: dict[str, Items] = {}
    sizes: dict[str, int] = {}
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for source in sorted({q["source"] for q in questions}):
            kind = SOURCES[source][0]
            files = [f for f in plan.files if f.source == source and f.kind == kind]
            wanted = [(float(q["since"]), float(q["until"])) for q in questions if q["source"] == source]
            hit = [f for f in files if any(f.t < b and f.t + float((f.spec or {}).get("length_s", 0)) > a for a, b in wanted)]
            if source == "porch":  # a day's median frame is of all its frames
                days = {datetime.fromtimestamp(f.t).date() for f in hit}
                hit = [f for f in files if datetime.fromtimestamp(f.t).date() in days]
            for f in files:
                sizes[f.path] = (folder / f.path).stat().st_size
            targets = sorted({(tuple(int(c) for c in q["probe"]["rgb"]), str(q["probe"].get("hue"))) for q in questions
                              if q["source"] == source and q["probe"].get("kind") == "shade" and q["probe"].get("rgb")})
            index[source] = _sounds(hit, pool) if source == "mic" else _pictures(hit, folder, pool, source == "porch", targets)
    return [_row(q, index[q["source"]], sizes) for q in questions]


# -- the floors -------------------------------------------------------------------------------------------------------------

def check(rows: list[dict], scale: str) -> list[str]:
    """What the floors of `scale` are not met by, as sentences (none: the haystack passes). `standard` and `pilot` have the floors of
    `FLOORS`; `tiny` needs one of each. On `standard` and `pilot` a needle that a filter ranks first (a `rank` of 1) at day or week level
    is a failure too, since sorting by the filter's measure and one look finds it; part level and `tiny` are exempt. A row's own
    problems (a needle the plan puts nowhere) and a needle file that is the largest or smallest of its scope by more than SIZE_Z are
    failures at every scale."""
    if scale not in ("tiny", "pilot", "standard"):
        raise ValueError(f"a scale is tiny, pilot or standard, not {scale!r}")
    failures: list[str] = []
    for row in rows:
        who = f"{row['qid']} ({row['level']}, {row['source']})"
        failures += [f"{who}: {problem}" for problem in row["problems"]]
        ranked = scale != "tiny" and row["level"] in RANK_LEVELS
        if scale == "tiny":
            need, shortlist, band = 1, 1, 1
        elif row["level"] in LEVELS:
            floors = FLOORS[scale]
            need, shortlist, band = floors["candidates"][row["level"]][row["source"]], floors["shortlist"][row["level"]], floors["band"][row["level"]]
        else:
            failures.append(f"{who}: the level {row['level']!r} has no floors; a media question is part, day or week")
            continue
        if row["candidates"] < need:
            failures.append(f"{who}: {row['candidates']} {row['unit']} in scope, {need} needed: brute force reads them all")
        if row["colour_shortlist"] is not None and row["colour_shortlist"] < shortlist:
            failures.append(f"{who}: a {row['probe']['hue']} filter leaves {row['colour_shortlist']} other frames at half the needle's {row['colour_needle']:.1%}, "
                            f"{shortlist} needed: it singles out the needle")
        if row.get("region_broad_shortlist") is not None and row["region_broad_shortlist"] < shortlist:
            failures.append(f"{who}: cropping the {area_name(row['probe']['rgb'])}'s area and keeping {row['probe']['hue']} leaves {row['region_broad_shortlist']} other frames "
                            f"at half the needle's {row['region_broad_needle']:.1%}, {shortlist} needed: it singles out the needle")
        if row["change_shortlist"] is not None and row["change_shortlist"] < shortlist:
            failures.append(f"{who}: frame differencing leaves {row['change_shortlist']} other frames at half the needle's change {row['change_needle']:.1f}, "
                            f"{shortlist} needed: it singles out the needle")
        if row["band_shortlist"] is not None and row["band_shortlist"] < band:
            low, high = row["probe"]["band"]
            failures.append(f"{who}: a band filter of {low:g} to {high:g} Hz leaves {row['band_shortlist']} other sounds at a quarter of the needle's level, "
                            f"{band} needed: it singles out the needle")
        if ranked and row["colour_rank"] == 1:
            failures.append(f"{who}: sorting by the share of {row['probe']['hue']} puts the needle first ({row['colour_needle']:.1%}, ahead of every other frame): "
                            f"one view finds it")
        if ranked and row.get("region_broad_rank") == 1:
            failures.append(f"{who}: sorting by the share of {row['probe']['hue']} inside the {area_name(row['probe']['rgb'])}'s area puts the needle first "
                            f"({row['region_broad_needle']:.1%}, ahead of every other frame): one view finds it")
        if ranked and row["change_rank"] == 1:
            failures.append(f"{who}: sorting by the change from the day's median frame puts the needle first ({row['change_needle']:.1f}, ahead of every other "
                            f"frame): one view finds it")
        if ranked and row["band_rank"] == 1:
            low, high = row["probe"]["band"]
            failures.append(f"{who}: sorting the sounds of {low:g} to {high:g} Hz by level puts the needle first ({row['band_needle_rms']:.2f}, ahead of every "
                            f"other sound in the band): one check finds it")
        if row["size_extreme"]:
            failures.append(f"{who}: the needle's file is the {row['size_rank']} of {row['size_files']} in scope (z {row['size_z']:.1f}): its size gives it away")
    return failures


# -- ffmpeg -----------------------------------------------------------------------------------------------------------------

def filters(ffmpeg: str | Path | None = None) -> list[str]:
    """The names of the filters of `ffmpeg` (`ffmpeg -hide_banner -filters`): by default imageio-ffmpeg's binary, the one the files
    arm gets (`bench.ensure_bin` copies it)."""
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    done = subprocess.run([str(ffmpeg), "-hide_banner", "-filters"], capture_output=True, timeout=60)
    if done.returncode:
        raise RuntimeError(f"{ffmpeg} -filters failed: {done.stderr.decode(errors='replace').strip()[-300:]}")
    names = []
    for line in done.stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and len(parts[0]) == 3 and set(parts[0]) <= set("TSC.") and "->" in parts[2]:
            names.append(parts[1])
    return names


def ocr_failures(names: list[str], ffmpeg: str | Path = "ffmpeg") -> list[str]:
    """A sentence for each filter of `names` that reads text off a picture (`NO_OCR`), and one when `names` is empty (it was not read)."""
    if not names:
        return [f"{ffmpeg} lists no filters: its output was not read"]
    return [f"{ffmpeg} has the {name} filter: the files arm could read the screen with it" for name in NO_OCR if name in names]


# -- the report -------------------------------------------------------------------------------------------------------------

def _cell(value, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def markdown(rows: list[dict], failures: list[str], meta: dict) -> str:
    """The report as Markdown: what ran, the failures, one line per media question, and the brute-force views per level."""
    lines = [f"# Shortcut probe: {meta.get('scale')}, seed {meta.get('seed')}", "",
             f"{len(rows)} media questions in {meta.get('seconds', 0):.1f} s; ffmpeg {meta.get('ffmpeg')} "
             f"({'has an ocr filter' if meta.get('ocr') else 'no ocr filter'}). "
             f"{'FAILED: ' + str(len(failures)) + ' failures.' if failures else 'No failures.'}", ""]
    if failures:
        lines += ["## Failures", "", *[f"- {failure}" for failure in failures], ""]
    lines += ["## Questions", "",
              f"The needle's value, then how many other frames or sounds a shortcut keeps (colour, colour in the needle's area and change: {HALF:.0%} of the needle's or more; band: in the "
              f"question's band at {BAND_RMS:.0%} of its level or more; the area is where a dialog or a parcel can be drawn, cropped and filtered with the broad "
              f"colour, which needs no needle found first), then the needle's rank by that measure (1: the filter puts it first, which "
              "fails a day or week question on pilot and standard). Size z is the needle's file among the source's files in scope.", "",
              "| qid | level | in scope | views | colour: needle / others / rank | colour in the needle's area: needle / others / rank | change: needle / others / rank "
              "| band: needle level / Hz / others / rank | size z |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        colour = "-" if r["colour_needle"] is None else f"{r['colour_needle']:.1%} / {_cell(r['colour_shortlist'])} / {_cell(r['colour_rank'])}"
        area = ("-" if r["region_broad_needle"] is None else
                f"{r['region_broad_needle']:.1%} / {_cell(r['region_broad_shortlist'])} / {_cell(r['region_broad_rank'])}")
        change = "-" if r["change_needle"] is None else f"{r['change_needle']:.1f} / {_cell(r['change_shortlist'])} / {_cell(r['change_rank'])}"
        band = ("-" if r["band_needle_rms"] is None else
                f"{r['band_needle_rms']:.2f} / {r['band_needle_hz']:.0f} / {_cell(r['band_shortlist'])} / {_cell(r['band_rank'])}")
        size = "-" if r["size_z"] is None else f"{r['size_z']:+.1f}{' (' + r['size_rank'] + ')' if r['size_extreme'] else ''}"
        lines.append(f"| {r['qid']}{' (absent)' if r['absent'] else ''}{' (last)' if r['last'] else ''} | {r['level']} | {r['candidates']} {r['unit']}{', ' + str(r['frames']) + ' frames' if r['source'] == 'porch' else ', ' + str(r['recorded_frames']) + ' frames' if r['source'] == 'screen' else ''} "
                     f"| {_cell(r['views'])} | {colour} | {area} | {change} | {band} | {size} |")
    tuned = [r for r in rows if r["tuned_colour_needle"] is not None or r["tuned_band"] is not None]
    if tuned:
        lines += ["", "## Tuned filters (report only)", "",
                  f"Not floors, and not in `check`: what a filter tuned to the needle itself leaves, for an agent that has found one needle (colour: its own hue, "
                  f"saturation and value within {TUNE[0]:g} degrees, {TUNE[1]:g} and {TUNE[2]:g}; in its area: the same colour inside the area the needle can be "
                  f"drawn in, for the error dialog and the parcel; band: {NARROW:.0%} either side of its pitch). Needle / others / rank, as above.", "",
                  "| qid | level | tuned colour: needle / others / rank | tuned colour in the needle's area | tuned band: Hz / others / rank |",
                  "| --- | --- | --- | --- | --- |"]
        for r in tuned:
            colour = "-" if r["tuned_colour_needle"] is None else f"{r['tuned_colour_needle']:.1%} / {_cell(r['tuned_colour_shortlist'])} / {_cell(r['tuned_colour_rank'])}"
            region = "-" if r["region_colour_needle"] is None else f"{r['region_colour_needle']:.1%} / {_cell(r['region_colour_shortlist'])} / {_cell(r['region_colour_rank'])}"
            band = "-" if r["tuned_band"] is None else f"{r['tuned_band'][0]} to {r['tuned_band'][1]} / {_cell(r['tuned_band_shortlist'])} / {_cell(r['tuned_band_rank'])}"
            lines.append(f"| {r['qid']} | {r['level']} | {colour} | {region} | {band} |")
    lines += ["", "## Brute-force views", "",
              "Pictures in scope over 25: the 5x5 montages that look at all of them (sound has none). The least and the most of the questions of a level. "
              "The screen's first column counts its states, the second the frames the recorder keeps of them (the files arm's brute force). "
              "The porch has a second column: a montage of one frame a clip, which finds a parcel that stays on the step.", "",
              "| level | screen, a view a state | screen, every frame | porch, every frame | porch, a frame a clip | phone |", "| --- | --- | --- | --- | --- | --- |"]
    for level in (*LEVELS, "time"):
        cells = []
        for source, key in (("screen", "views"), ("screen", "recorded_views"), ("porch", "views"), ("porch", "views_per_clip"), ("phone", "views")):
            views = [r[key] for r in rows if r["level"] == level and r["source"] == source]
            cells.append("-" if not views else str(views[0]) if min(views) == max(views) else f"{min(views)} to {max(views)}")
        if any(c != "-" for c in cells):
            lines.append(f"| {level} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(rows: list[dict], failures: list[str], out: Path, folder: Path, meta: dict) -> tuple[Path, Path]:
    """`shortcuts.json` and `shortcuts.md` in `out`, the run folder. Never in `folder`, the data (a `ValueError`)."""
    out, folder = Path(out).resolve(), Path(folder).resolve()
    if out == folder or folder in out.parents:
        raise ValueError(f"the probe's report goes in the run folder, never in the data folder {folder}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "shortcuts.json").write_text(json.dumps({**meta, "failures": failures, "rows": rows}, indent=1), encoding="utf-8")
    (out / "shortcuts.md").write_text(markdown(rows, failures, meta), encoding="utf-8")
    return out / "shortcuts.json", out / "shortcuts.md"


def run(plan, folder, out, *, ffmpeg: str | Path | None = None, seed: int | None = None) -> list[str]:
    """`probe` and `check` the plan's media questions over the data in `folder`, check that `ffmpeg` (default: the files arm's) has no
    `ocr` filter, write the report into `out` and return the failures. `build` raises on them unless it was told `--no-probe`."""
    start = time.perf_counter()
    rows = probe(plan, folder)
    names = filters(ffmpeg)
    ffmpeg_name = Path(ffmpeg).name if ffmpeg else "imageio-ffmpeg"
    failures = [*check(rows, plan.scale), *ocr_failures(names, ffmpeg_name)]
    meta = {"scale": plan.scale, "seed": seed if seed is not None else getattr(plan, "seed", None), "days": [d.isoformat() for d in getattr(plan, "days", ())],
            "ffmpeg": ffmpeg_name, "ocr": any(name in names for name in NO_OCR), "seconds": round(time.perf_counter() - start, 1)}
    write_report(rows, failures, out, folder, meta)
    return failures
