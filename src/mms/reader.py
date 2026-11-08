"""Windows of the index for scripts (DESIGN.md §5.3, §16): what a model reads, cut from the stored items and records at
query time. The store keeps segments of about 60 s and whole files; this module cuts them as a profile says, so a new
model changes a script's profile and never the store.

    for window in windows(store, since, until, sources=["cam"], scope=scope, about=about):
        ...window.media / window.text...

`about` (decider.About) says what the model reads, so the reader knows no model: its media window kinds, the file types
it takes whole, its audio rate and its profile. What each kind of file reaches a model as:
- a picture: upright by its EXIF orientation, 16-bit and float scaled to 8 bits, transparency flattened onto white (onto
  dark grey when what is drawn is light), a JPEG decoded at 1/2 to 1/8 when that still covers the box, and refused above
  `PICTURE_MAX_PX`;
- an animation (GIF, APNG, WebP) a window per `every_s` of its own clock, a multi-page TIFF a window per page, one
  second apart; an MPO its first picture only;
- a video: frames turned by the display matrix, a run of frames that look the same one window (`still_s`), and its
  soundtrack as `audio` windows of the same item, merged by time;
- a sound: mono at the model's rate (16 kHz for `look`), decoded `AUDIO_BLOCK_S` at a time from a seek, so a long file
  is never in memory whole;
- text: decoded by `media.decode_text` (UTF-8, UTF-16 or cp1252), refused above `TEXT_MAX_BYTES`;
- a version or picture of a type the model takes whole (`about.files`): one `file` window with its bytes, refused above
  `FILE_MAX_BYTES`;
- anything else is not read: `windows` reports the item to `on_unread`. A video read for its frames whose soundtrack is
  not (or the reverse) is read in part: `windows` reports the item to `on_partial`.

`look` writes one moment of an item as files (a PNG and its inline JPEG, or a WAV), or copies a file nothing reads, for
the `look` op; `sheet` writes up to nine moments as one picture.
"""

from __future__ import annotations

import bisect
import heapq
import io
import itertools
import math
import os
import re
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from mms import media
from mms.config import iso
from mms.store import SPANS, Item, Record, Scope, Store, file_name

PROFILE = {
    "every_s": 1.0,        # video: one frame per this many seconds (0: every frame)
    "still_s": 30.0,       # video: frames that look the same as the one that began a window join it, up to this long (0: off)
    "audio_s": 5.0,        # audio: the length of a window, and ...
    "audio_hop_s": 2.5,    # ... the step between windows
    "quiet_rms": 0.004,    # audio: a window whose RMS (samples in -1..1) is below this is skipped, about -48 dBFS
    "text_chars": 6000,    # a text version: about this many characters to a window
    "records_chars": 3000,  # a run of records: about this many characters to a window
    "max_px": 230_400,     # pictures are cut to at most this many pixels (640x360); 0: the model's cap (as stored for a model with no cap)
}
RATE = 16000  # the sample rate of `look`'s WAV, and the default of `_span` and `_cut`
BLOCK = 10  # frames decoded at a time when a segment is read backwards
DECODE_THREADS = min(os.cpu_count() or 1, 8)  # frame threads per decode: 4 to 16 were alike on a 1408x1408 clip, and more crowd capture
AUDIO_BLOCK_S = 60.0  # sound is decoded this much at a time, from one seek
PICTURE_MAX_PX = 64_000_000  # a picture of more pixels is not decoded (a JPEG is first decoded at 1/2 to 1/8)
TEXT_MAX_BYTES = 64_000_000  # a text version larger than this is not read
FILE_MAX_BYTES = 64_000_000  # a file larger than this is not handed to a decider whole, nor copied by `look`
SLACK_S = 5.0  # a segment may decode a little longer than its indexed end
LOOK_PAGE = 20_000  # characters of text `look` gives at a time
LOOK_INLINE_PX = 1568  # the long side of the picture `look` gives inline
LOOK_AUDIO_S = 5.0
SHEET_TILE_PX = LOOK_INLINE_PX ** 2  # a sheet's pictures are decoded as large as a tile alone could be, then cut to its cell
DARK_GREY = (32, 32, 32)  # behind a sheet's pictures, and under a light picture's transparency
PAST_END_S = 1.0  # `look` at more than this past the end of an item is refused
ORIENTATION = 0x0112  # the EXIF tag
# Two pictures look the same when their mean colour in cells of STILL_CELL_PX px differs by at most STILL_LEVEL (of 255) in
# every cell and channel. A thing of contrast 100 moves a cell's mean by more than 6 from about 9x9 px wherever it falls
# (about 5x5 inside one cell): roughly the smallest thing a model reading 640x360 resolves, about one of its vision
# patches. Sensor and codec noise move a cell's mean by well under 1, since a cell averages 256 pixels; a change of light
# moves every cell. So the test errs toward "changed": judging a frame again costs about 53 ms, and skipping one can cost
# a sighting. Not grey, which cannot tell a red (199, 28, 29) from a dark olive (79, 78, 53).
STILL_CELL_PX = 16
STILL_LEVEL = 6
Image.MAX_IMAGE_PIXELS = None  # Pillow's bomb check raises a plain Exception above 179 MP; `_open` checks the size


class TooBig(ValueError):
    """A file `look` would copy is over `FILE_MAX_BYTES`."""


@dataclass(kw_only=True)
class Window:
    """One thing to read. `kind` is video (a frame), audio (a sound, or the soundtrack of a video item), image (a
    picture, or a frame of an animation or a page of a TIFF), text (a part of a version), records (a run of them) or
    file (a whole version); `media` is a PIL image, `{"type": "audio", "audio": float32 mono, "sampling_rate": hz}` or
    `{"type": "file", "media": the media type, "name": the file's name, "data": its bytes}`, `text` the text of a
    text or records window. `at` is seconds into the item, None for text, records, files and a single picture, which
    have no item position (records have no item at all)."""

    source: str
    item: Item | None = None
    records: list[Record] = field(default_factory=list)
    t0: float
    t1: float
    at: float | None = None
    kind: str
    media: object = None
    text: str = ""

    @property
    def project(self) -> str | None:
        return self.item.project if self.item else (self.records[0].project if self.records else None)


def profile_of(profile: dict | None, about) -> dict:
    """The profile a scan uses: the defaults of `PROFILE`, then the model's (`about.profile`), then `profile`, the ask's. A
    model with a picture cap (`about.max_px`) is never given more: `max_px` is cut to it, and 0 is the cap."""
    wrong = set(about.profile) - set(PROFILE)
    if wrong:
        raise ValueError(f"{about.name}'s About.profile has {sorted(wrong)}; a profile has {', '.join(PROFILE)}")
    out = {**PROFILE, **about.profile, **(profile or {})}
    unknown = set(out) - set(PROFILE)
    if unknown:
        raise ValueError(f"a profile has {', '.join(PROFILE)}, not {sorted(unknown)}")
    if (out["every_s"] < 0 or out["still_s"] < 0 or out["audio_s"] <= 0 or out["audio_hop_s"] <= 0 or out["text_chars"] < 1
            or out["records_chars"] < 1 or out["max_px"] < 0):
        raise ValueError("a profile's every_s, still_s and max_px are at least 0; its audio_s, audio_hop_s, text_chars and "
                         "records_chars are above 0")
    if about.max_px:
        out["max_px"] = min(out["max_px"] or about.max_px, about.max_px)
    return out


def size_text(n: int) -> str:
    """A size for a message: bytes below 1 KB, KB below 1 MB, else MB (decimal, one decimal at most)."""
    for unit, size in (("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.1f}".removesuffix(".0") + f" {unit}"
    return f"{n} byte{'' if n == 1 else 's'}"


def fit(width: int, height: int, max_px: int) -> tuple[int, int]:
    """The size a picture is scaled to for a budget of `max_px` pixels: aspect kept, never up, 0 for as stored."""
    if not max_px or width * height <= max_px:
        return width, height
    scale = math.sqrt(max_px / (width * height))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    while w * h > max_px:
        if w >= h:
            w -= 1
        else:
            h -= 1
    return w, h


class Region(NamedTuple):
    """The part of a picture `look` shows: fractions of it from its top left, each from 0 to 1, with `top` above `bottom` and
    `left` before `right`. `ops` checks that, so a `Region` is always one."""

    top: float
    left: float
    bottom: float
    right: float

    def crop(self, image: Image.Image) -> Image.Image:
        """The part of `image` inside the region, at the image's own pixels: its edges rounded to whole pixels, so that the
        part is at least one pixel wide and high."""
        width, height = image.size
        left, top = min(round(self.left * width), width - 1), min(round(self.top * height), height - 1)
        right, bottom = max(round(self.right * width), left + 1), max(round(self.bottom * height), top + 1)
        return image.crop((left, top, right, bottom))


UNREADABLE = (OSError, ValueError, av.error.FFmpegError)  # what a file that is gone or cut short raises


def windows(store: Store, since: float | None = None, until: float | None = None, sources=None, *, scope: Scope, about,
            newest_first: bool = True, profile: dict | None = None, on_unreadable=None, on_unread=None,
            on_partial=None):
    """Windows over the items and records of a range (the rule of DESIGN.md §5.3) that `scope` sees, newest first by
    `t1` (or oldest first by `t0`) across all sources. Each item is decoded only when its turn comes, so stopping early
    costs nothing for the rest. `profile` overrides the keys of `PROFILE` it names. An item that cannot be read raises
    to the caller, or with `on_unreadable(item, error)` is reported there and the scan goes on to the next item. An item
    that `about` does not read (`_read_as` gives None) is reported to `on_unread(item)`, every such item before the first
    window (the items are all looked at when the scan starts). A video item with a soundtrack (`meta.sound`) that `about`
    reads only in part is reported to `on_partial(item, "sound")` when its frames are read and its sound is not, and
    `on_partial(item, "frames")` when its sound is read and its frames are not, likewise before the first window."""
    profile = profile_of(profile, about)
    sign = -1 if newest_first else 1
    order = itertools.count()
    heap: list = []  # (sign * time, n, the window or None while only its unit's bound is known, the unit's iterator)
    owners: dict[int, Item] = {}

    def advance(unit, n):
        """The next window of a unit, or None at its end, or when its item cannot be read and `on_unreadable` took it."""
        try:
            return next(unit, None)
        except UNREADABLE as error:
            if on_unreadable is None:
                raise
            on_unreadable(owners[n], error)
            return None

    for item in store.select(since, until, sources, newest_first=newest_first, scope=scope):
        form = _read_as(item, about)
        if form is None:
            if on_unread is not None:
                on_unread(item)
            continue
        if on_partial is not None and form == "video" and item.meta.get("sound"):
            if "audio" not in about.reads:
                on_partial(item, "sound")
            elif "video" not in about.reads:
                on_partial(item, "frames")
        bound = (max(item.t1 or item.t0, item.t0) + SLACK_S) if newest_first and item.kind in SPANS else item.t0
        unit = _unit(store, item, form, profile, since, until, newest_first, about)
        owners[n := next(order)] = item
        heapq.heappush(heap, (sign * bound, n, None, unit))
    for window in _runs(store, since, until, sources, profile["records_chars"], scope):
        heapq.heappush(heap, (sign * (window.t1 if newest_first else window.t0), next(order), window, iter(())))
    while heap:
        _, n, window, unit = heapq.heappop(heap)
        if window is None:
            window = advance(unit, n)
            if window is not None:
                heapq.heappush(heap, (sign * (window.t1 if newest_first else window.t0), n, window, unit))
            continue
        yield window
        after = advance(unit, n)
        if after is not None:
            heapq.heappush(heap, (sign * (after.t1 if newest_first else after.t0), n, after, unit))


def estimate(store: Store, since: float | None, until: float | None, sources, scope: Scope,
             profile: dict | None = None, *, about) -> dict:
    """How many windows of each kind a range holds, counted from the index without decoding (DESIGN.md §16): a video's
    length over `every_s` (which over-counts still stretches), a sound's over the hop, a text by characters
    over the window size, and records the same. A video or audio version counts as one window, and so does a picture
    (an animation's or a TIFF's others are not counted) and a file taken whole. The soundtrack of a video counts when its
    meta has `sound` (a captured segment's has it, probed when it was indexed). Frames count only when `about` reads video, and
    sound only when it reads audio."""
    profile = profile_of(profile, about)
    counts = {"video": 0, "audio": 0, "image": 0, "file": 0, "text": 0, "records": 0}
    lo = -math.inf if since is None else since
    hi = math.inf if until is None else until
    for item in store.select(since, until, sources, scope=scope):
        form = _read_as(item, about)
        if form is None:
            continue
        seconds = max(min(item.t1 or item.t0, hi) - max(item.t0, lo), 0.0)
        hops = math.ceil(seconds / profile["audio_hop_s"])
        sound = form == "video" and item.meta.get("sound") and "audio" in about.reads
        if form == "file":
            counts["file"] += 1
        elif item.kind == "file" and form in ("video", "audio"):
            counts[form] += 1 if form == "audio" or "video" in about.reads else 0
            counts["audio"] += 1 if sound else 0
        elif form == "video":
            frames = math.ceil(seconds / profile["every_s"]) if profile["every_s"] else math.ceil(seconds * 2)
            counts["video"] += frames if "video" in about.reads else 0
            counts["audio"] += hops if sound else 0
        elif form == "audio":
            counts["audio"] += hops
        elif form == "image":
            counts["image"] += 1
        else:
            counts["text"] += max(1, math.ceil(item.meta.get("size", item.bytes) / profile["text_chars"]))
    count, chars = store.record_size(since, until, sources, scope)
    if count:
        counts["records"] = math.ceil((chars + 26 * count) / profile["records_chars"])
    return counts


def form(item: Item) -> str | None:
    """What an item is read as: video, audio, image or text, or None for a file nothing reads. A recording is its kind;
    a file is what its recorded media type says (`meta.media`, from its content), text before a picture so that an SVG
    is read as text."""
    if item.kind != "file":
        return item.kind
    kind = item.meta.get("media") or ""
    if kind.startswith("video/"):
        return "video"
    if kind.startswith("audio/"):
        return "audio"
    if media.is_text(kind):
        return "text"
    return "image" if kind.startswith("image/") else None


def _read_as(item: Item, about) -> str | None:
    """What a decider is handed of an item, by what `about` declares: `file` for a version or picture of a type it takes whole;
    else the item's `form` if it reads that (text is always read; a video item is read for its frames if it reads
    video, and for its soundtrack if it reads audio and the item has sound); else None, which is unread."""
    if item.kind in ("file", "image") and about.takes(item.meta.get("media")):
        return "file"
    what = form(item)
    if what == "text":
        return what
    if what == "video":
        return what if "video" in about.reads or ("audio" in about.reads and item.meta.get("sound")) else None
    return what if what in about.reads else None


def _unit(store, item, form, profile, since, until, newest_first, about):
    """The windows of one item, in order. A file that cannot be read raises."""
    if form == "file":
        yield _file(store, item)
    elif form == "video":
        parts = []
        if "video" in about.reads:
            parts.append(_video(store, item, profile, since, until, newest_first))
        if "audio" in about.reads:
            parts.append(_sound(store, item, profile, since, until, newest_first, about.audio_rate))
        yield from heapq.merge(*parts, key=lambda window: (window.t1 if newest_first else window.t0, window.at),
                               reverse=newest_first)
    elif form == "audio":
        yield from _sound(store, item, profile, since, until, newest_first, about.audio_rate)
    elif form == "image":
        yield from _stills(store, item, profile, newest_first)
    else:
        parts = chunks(_text(store, item), profile["text_chars"])
        for part in (reversed(parts) if newest_first else parts):
            yield Window(source=item.source, item=item, t0=item.t0, t1=item.t0, kind="text", text=part)


def _file(store: Store, item: Item) -> Window:
    """A version or a picture as one window, whole; a `ValueError` says it is over the limit."""
    size = item.meta.get("size", item.bytes)
    if size > FILE_MAX_BYTES:
        raise ValueError(f"a file of {size_text(size)} is over the reader's limit of {size_text(FILE_MAX_BYTES)}")
    return Window(source=item.source, item=item, t0=item.t0, t1=item.t0, kind="file",
                  media={"type": "file", "media": item.meta.get("media"), "name": item.name,
                         "data": store.read_bytes(item)})


def _text(store: Store, item: Item) -> str:
    """A text version decoded; a `ValueError` says it is over the limit."""
    size = item.meta.get("size", item.bytes)
    if size > TEXT_MAX_BYTES:
        raise ValueError(f"a text of {size / 1e6:.0f} MB is over the reader's limit of {TEXT_MAX_BYTES // 1_000_000} "
                         "MB; watch it as a log source")
    return media.decode_text(store.read_bytes(item))


# pictures

def _open(store: Store, item: Item, max_px: int) -> Image.Image:
    """A picture, opened and sized up but not decoded. A JPEG is set to decode at 1/2 to 1/8 when that still covers the
    box of `max_px` (with no box, when it is over `PICTURE_MAX_PX`); a picture still over `PICTURE_MAX_PX` is refused."""
    if item.meta.get("media") == "image/heic":
        raise ValueError("a HEIC picture: reading it needs pi-heif, which is not installed")
    image = Image.open(io.BytesIO(store.read_bytes(item)))
    if image.format in ("JPEG", "MPO"):
        image.draft("RGB", fit(*image.size, max_px or PICTURE_MAX_PX // 4))
    width, height = image.size
    if width * height > PICTURE_MAX_PX:
        raise ValueError(f"a {width}×{height} picture is {width * height / 1e6:.0f} MP; the reader decodes at most "
                         f"{PICTURE_MAX_PX // 1_000_000} MP")
    return image


def _clock(image: Image.Image) -> tuple[list[float], float]:
    """Where each frame of a picture starts and where it ends, in seconds: an animation's own clock (a missing or
    under-20 ms duration counts 100 ms, as browsers play it), a multi-page TIFF's pages one second apart, and one frame
    at 0 for anything else (an MPO's other picture is a gain map or a second view)."""
    frames = getattr(image, "n_frames", 1)
    if frames == 1 or image.format not in ("GIF", "PNG", "WEBP", "TIFF"):
        return [0.0], 0.0
    if image.format == "TIFF":
        return [float(i) for i in range(frames)], float(frames)
    starts, end = [], 0.0
    for index in range(frames):
        image.seek(index)
        if image.format == "WEBP":
            image.load()  # a WebP frame has its duration only once loaded
        starts.append(end)
        duration = image.info.get("duration") or 0
        end += (duration if duration >= 20 else 100) / 1000
    return starts, end


def _upright(image: Image.Image, max_px: int) -> Image.Image:
    """The current frame of an opened picture as RGB, upright by its EXIF orientation, 16-bit and float scaled to 8 bits
    by their range, transparency flattened onto white (onto dark grey when what is opaque is light, say a white icon),
    and scaled down to `max_px` pixels."""
    if image.getexif().get(ORIENTATION, 1) != 1:
        image = ImageOps.exif_transpose(image)
    if image.mode.startswith("I;16"):
        image = Image.fromarray((np.asarray(image) // 257).astype(np.uint8))
    elif image.mode in ("I", "F"):
        values = np.asarray(image, dtype=np.float64)
        finite = values[np.isfinite(values)]  # NaN and infinity are no part of the range
        low, high = (finite.min(), finite.max()) if finite.size else (0.0, 0.0)
        values = np.nan_to_num(np.clip(values, low, high), nan=low)
        image = Image.fromarray(((values - low) * (255 / (high - low)) if high > low
                                 else np.full(values.shape, 128.0)).astype(np.uint8))
    if "transparency" in image.info or image.mode in ("LA", "PA", "La", "RGBa"):
        image = image.convert("RGBA")
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] < 255:
        solid = np.asarray(image.getchannel("A")) >= 128
        light = solid.any() and np.asarray(image.convert("L"))[solid].mean() >= 128
        ground = Image.new("RGBA", image.size, (*DARK_GREY, 255) if light else (255, 255, 255, 255))
        image = Image.alpha_composite(ground, image)
    image = image.convert("RGB")
    box = fit(*image.size, max_px)
    return image.resize(box, Image.Resampling.BICUBIC) if box != image.size else image


def _stills(store, item, profile, newest_first):
    """A picture is one window. An animation is a window per `every_s` of its clock, the first frame of each slot, and a
    multi-page TIFF a window per page; `at` is the frame's time, and every window is at the item's `t0`."""
    image = _open(store, item, profile["max_px"])
    starts, _ = _clock(image)
    if len(starts) == 1:
        yield Window(source=item.source, item=item, t0=item.t0, t1=item.t0, kind="image",
                     media=_upright(image, profile["max_px"]))
        return
    every, slots, picks = profile["every_s"], set(), []
    for index, start in enumerate(starts):
        slot = int(start // every) if every > 0 and image.format != "TIFF" else index  # every page is a window
        if slot not in slots:
            slots.add(slot)
            picks.append((index, start))
    for index, start in (reversed(picks) if newest_first else picks):
        image.seek(index)
        yield Window(source=item.source, item=item, t0=item.t0, t1=item.t0, at=start, kind="image",
                     media=_upright(image, profile["max_px"]))


def _within(t: float, since: float | None, until: float | None) -> bool:
    return (since is None or t >= since) and (until is None or t < until)


def _cells(image: Image.Image) -> tuple[tuple[int, int], np.ndarray]:
    """A picture's size and its mean colour in cells of `STILL_CELL_PX` pixels: what `_same` compares."""
    return image.size, np.asarray(image.reduce(STILL_CELL_PX), dtype=np.int16)


def _same(a, b) -> bool:
    """Whether two pictures, as `_cells` gives them, look the same to a model: of one size, with no cell's mean in any
    channel more than `STILL_LEVEL` apart."""
    return a[0] == b[0] and int(np.abs(a[1] - b[1]).max()) <= STILL_LEVEL


# video

def _seconds(stream, pts: int) -> float:
    return float((pts - (stream.start_time or 0)) * stream.time_base)


def _video_stream(container):
    """The first video stream, set to decode with frame threads. PyAV's default is slice threads, which help only a stream
    encoded in slices (a 1408x1408 camera clip decodes 3 to 7 times faster with frame threads). The frames are the same
    ones with the same pts, each coming out a few frames later, and the end of the file drains the last."""
    if not container.streams.video:
        raise ValueError("no video stream")
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    stream.thread_count = DECODE_THREADS
    return stream


def _times(path: Path) -> list[tuple[int, float]]:
    """Every video frame's `(pts, seconds into the file)`, in order."""
    with av.open(str(path)) as container:
        stream = _video_stream(container)
        pts = sorted({packet.pts for packet in container.demux(stream) if packet.pts is not None})
        return [(p, _seconds(stream, p)) for p in pts]


def _decode(path: Path, wanted: list[tuple[int, float]], max_px: int = 0):
    """`(seconds, PIL image)` for the frames with these pts, from one seek and one pass. With `max_px` a frame is
    scaled down to the budget in the decoder's own scaler (bicubic), not decoded whole and resized.
    A frame is turned by its display matrix (a phone's portrait video is coded sideways), the box taken from the
    frame's own size."""
    want = dict(wanted)
    with av.open(str(path)) as container:
        stream = _video_stream(container)
        first = min(want)
        if first > (stream.start_time or 0):
            container.seek(first, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts in want:
                box = fit(frame.width, frame.height, max_px)
                image = (frame.to_image(width=box[0], height=box[1], interpolation="BICUBIC")
                         if box != (frame.width, frame.height) else frame.to_image())
                yield want.pop(frame.pts), image.rotate(frame.rotation, expand=True) if frame.rotation else image
                if not want:
                    return


def _video(store, item, profile, since, until, newest_first):
    """One frame per `every_s`: the first frame in the range of each slot, the slots being `every_s` of the clock, not
    of the segment. A screen segment holds only the frames that changed, so a slot with no frame is skipped, and a
    still screen gives one window and not one per second. A frame that looks the same (`_same`) as the one that began
    the run, and is less than `still_s` from it, joins the run: the run is one window, its first frame in scan order
    (the latest when the scan goes newest first) over the span of all its frames, and a run never crosses items. A
    video version (a file) is read the same way, with every window at the version's `t0`."""
    path = store.path_of(item)
    version = item.kind == "file"
    every, still, slots, marks = profile["every_s"], profile["still_s"], set(), []
    for pts, sec in _times(path):
        clock = sec if version else item.t0 + sec
        if not version and not _within(clock, since, until):
            continue
        slot = int(clock // every) if every > 0 else pts
        if slot not in slots:
            slots.add(slot)
            marks.append((pts, sec))
    blocks = [marks[i:i + BLOCK] for i in range(0, len(marks), BLOCK)]  # read backwards a block at a time
    run = None  # the window of the run being read, and the cells of its first picture
    for block in (reversed(blocks) if newest_first else blocks):
        try:
            frames = list(_decode(path, block, profile["max_px"]))
        except UNREADABLE:
            if run is not None:
                yield run[0]  # the run held so far is not lost with the item
            raise
        for sec, image in (reversed(frames) if newest_first else frames):
            t = item.t0 if version else item.t0 + sec
            cells = _cells(image) if still else None
            if run is not None and abs(sec - run[0].at) < still and _same(run[1], cells):
                run[0].t0, run[0].t1 = min(run[0].t0, t), max(run[0].t1, t)
                continue
            if run is not None:
                yield run[0]
                run = None
            window = Window(source=item.source, item=item, t0=t, t1=t, at=sec, kind="video", media=image)
            if still:
                run = window, cells  # held for the frames that look the same to join
            else:
                yield window
    if run is not None:
        yield run[0]


# audio

def _span(path: Path, t0: float, t1: float, rate: int = RATE) -> tuple[float, np.ndarray]:
    """The first audio stream from `t0` to a little past `t1` (seconds), as mono float32 in -1..1 at `rate`, from one
    seek, with the time of its first sample; empty when the file has no sound. The resampler is made again whenever a
    frame's format, layout or rate differs from the one before (joined files change rate)."""
    out, first, setup, resampler = [], None, None, None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return 0.0, np.zeros(0, np.float32)
        stream = container.streams.audio[0]
        start = stream.start_time or 0
        if t0 > 0:
            container.seek(int(t0 / stream.time_base) + start, stream=stream, backward=True)
        for frame in container.decode(stream):
            at = float((frame.pts - start) * stream.time_base) if frame.pts is not None else None
            if first is None:
                first = at or 0.0
            if setup != (frame.format.name, frame.layout.name, frame.sample_rate):
                if resampler is not None:
                    out += [part.to_ndarray().reshape(-1) for part in resampler.resample(None)]
                setup = frame.format.name, frame.layout.name, frame.sample_rate
                resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            out += [part.to_ndarray().reshape(-1) for part in resampler.resample(frame)]
            if at is not None and at > t1:
                break
        if resampler is not None:
            out += [part.to_ndarray().reshape(-1) for part in resampler.resample(None)]
    return first or 0.0, np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def _cut(first: float, samples: np.ndarray, start: float, size: float, last: bool = False,
         rate: int = RATE) -> np.ndarray:
    """`size` seconds of a `_span` from `start` (a start before its first sample is taken from it), shorter or empty
    where the sound ends. With `last` the window is set back to end where the sound does."""
    count = int(round(size * rate))
    low = max(int(round((start - first) * rate)), 0)
    return samples[-count:] if last and low + count > len(samples) else samples[low:low + count]


def _sound(store, item, profile, since, until, newest_first, rate):
    """Windows of `audio_s` every `audio_hop_s`, the last one set back to end where the sound does; quiet ones are
    skipped. A sound version (a file) has every window at the version's `t0`. The sound is decoded `AUDIO_BLOCK_S` at a
    time, block k from one seek to its start, `audio_s` past its end, so a window is the same whichever way the scan
    goes. An item with no sound (a screen's video) has no windows."""
    path = store.path_of(item)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return
    length = media.duration(path, sound=True)  # the sound's own, which a video's picture can outlast
    if not length:
        return
    version = item.kind == "file"
    hop, size = profile["audio_hop_s"], profile["audio_s"]
    starts = [i * hop for i in range(int(max(length - size, 0) // hop) + 1)]
    if starts[-1] + size < length - 1e-6:
        starts.append(length - size)
    final = starts[-1]
    starts = [s for s in starts if version or _within(item.t0 + s, since, until)]
    blocks = [list(group) for _, group in itertools.groupby(starts, key=lambda s: int(s // AUDIO_BLOCK_S))]
    for block in (reversed(blocks) if newest_first else blocks):
        k = int(block[0] // AUDIO_BLOCK_S)
        first, samples = _span(path, k * AUDIO_BLOCK_S, (k + 1) * AUDIO_BLOCK_S + size, rate)
        for start in (reversed(block) if newest_first else block):
            piece = _cut(first, samples, start, size, last=start == final, rate=rate)
            if len(piece) == 0 or float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)))) < profile["quiet_rms"]:
                continue
            t0, t1 = (item.t0, item.t0) if version else (item.t0 + start, item.t0 + start + len(piece) / rate)
            yield Window(source=item.source, item=item, t0=t0, t1=t1, at=start, kind="audio",
                         media={"type": "audio", "audio": piece, "sampling_rate": rate})


# text and records

def chunks(text: str, limit: int) -> list[str]:
    """Parts of about `limit` characters, cut at paragraph breaks, else line breaks, else (for a line longer than
    `limit`) anywhere."""
    parts, current = [], ""
    for paragraph in re.split(r"(?<=\n\n)", text):
        for line in (paragraph.splitlines(keepends=True) if len(paragraph) > limit else [paragraph]):
            for piece in ([line[i:i + limit] for i in range(0, len(line), limit)] if len(line) > limit else [line]):
                if current and len(current) + len(piece) > limit:
                    parts.append(current)
                    current = ""
                current += piece
    if current:
        parts.append(current)
    return parts


def _runs(store, since, until, sources, limit: int, scope: Scope):
    """Windows of records: each source's records in time order, a run of them to a window up to `limit` characters (a
    longer record is a window of its own)."""
    by_source: dict[str, list[Record]] = {}
    for record in store.records(since, until, sources, scope=scope):
        by_source.setdefault(record.source, []).append(record)
    for source, rows in by_source.items():
        run, lines = [], []
        for record in rows:
            line = f"{iso(record.t)} {record.data}"
            if run and sum(map(len, lines)) + len(lines) + len(line) > limit:
                yield _run(source, run, lines)
                run, lines = [], []
            run.append(record)
            lines.append(line)
        if run:
            yield _run(source, run, lines)


def _run(source, run, lines) -> Window:
    return Window(source=source, records=run, t0=run[0].t, t1=run[-1].t, kind="records", text="\n".join(lines))


# look

def page(text: str, offset: int, size: int = LOOK_PAGE) -> dict:
    """A page of text for `look`: at most `size` characters from `offset`."""
    end = offset + size
    out = {"text": text[offset:end], "more": end < len(text)}
    if out["more"]:
        out["next_offset"] = end
    return out


def look(store: Store, item: Item, at: float, folder: Path, offset: int = 0, size: int = LOOK_PAGE,
         region: Region | None = None) -> dict:
    """One moment of an item (DESIGN.md §11): a video frame or a picture as a full-size PNG in `folder` with a JPEG file
    of at most 1568 px beside it (`inline` is its path), 5 s of sound from `at` as a 16 kHz mono WAV there, or a page of
    `size` characters of a version's text. A video with sound adds `audio`, its 5 s from `at`. An animation's or a
    multi-page TIFF's `at` picks the frame. Any other file is copied whole into `folder` (`file` has its path, media
    type, name and size); `TooBig` says it is over `FILE_MAX_BYTES`. A `ValueError` says `at` is past the item's end.
    With a `region` (a video frame or a picture only) the PNG is that part cut from the picture as it is decoded whole
    (a JPEG of over 64 MP at the reduced scale it is drafted to), and the JPEG is it enlarged, or shrunk, to `LOOK_INLINE_PX`
    on its long side."""
    what = form(item)
    path = store.path_of(item)
    if what == "video":
        times = _times(path)
        if not times:
            raise ValueError(f"item {item.id} has no frames")
        end = max(times[-1][1], (item.t1 - item.t0) if item.kind != "file" and item.t1 else 0.0)
        if at > end + PAST_END_S:
            raise ValueError(f"item {item.id} is {end:.1f} s long, so it has nothing at {at:g} s")
        image = _frame(path, _frame_at(times, at), 0, region)
        if image is None:
            raise ValueError(f"item {item.id} has no frame at {at:g} s")
        made = {"kind": "image", "image": _picture(image, folder, zoom=region is not None), "at": at}
        heard = _heard(path, at, folder)
        return {**made, "audio": heard} if heard else made
    if what == "image":
        image = _open(store, item, 0)
        starts, end = _clock(image)
        if len(starts) > 1 and at > end + PAST_END_S:
            raise ValueError(f"item {item.id} is {end:.1f} s long, so it has nothing at {at:g} s")
        shown = _page_image(image, starts, _page_at(starts, at), 0, region)
        return {"kind": "image", "image": _picture(shown, folder, zoom=region is not None),
                "at": at if len(starts) > 1 else None}
    if what == "audio":
        heard = _heard(path, at, folder)
        if heard is None:
            raise ValueError(f"item {item.id} has no sound at {at:g} s")
        return {"kind": "audio", "audio": heard, "at": at}
    if what == "text":
        return {"kind": "text", **page(_text(store, item), offset, size), "at": None}
    length = item.meta.get("size", item.bytes)
    if length > FILE_MAX_BYTES:
        raise TooBig(f"item {item.id} is {size_text(length)}; look copies at most {size_text(FILE_MAX_BYTES)}")
    data = store.read_bytes(item)
    target = folder / f"{uuid.uuid4().hex[:12]}-{file_name(item.name, 'file')}"
    target.write_bytes(data)
    return {"kind": "file", "at": None, "file": {"path": str(target), "name": item.name, "bytes": len(data),
                                                 "media": item.meta.get("media") or "application/octet-stream"}}


def _heard(path: Path, at: float, folder: Path) -> dict | None:
    """5 s of an item's sound from `at` as a WAV in `folder`, or None when it has none there."""
    first, samples = _span(path, at, at + LOOK_AUDIO_S)
    piece = _cut(first, samples, at, LOOK_AUDIO_S)
    if len(piece) == 0:
        return None
    target = folder / f"{uuid.uuid4().hex[:12]}.wav"
    with wave.open(str(target), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes((np.clip(piece, -1, 1) * 32767).astype("<i2").tobytes())
    rms = float(np.sqrt(np.mean(np.square(piece, dtype=np.float64))))
    return {"path": str(target), "seconds": round(len(piece) / RATE, 3), "rms": round(rms, 5)}


def _frame_at(times: list[tuple[int, float]], at: float) -> tuple[int, float]:
    """The frame of a video at or before `at` seconds (its first when `at` is before it): `(pts, seconds)`."""
    return times[max(bisect.bisect([sec for _, sec in times], at) - 1, 0)]


def _page_at(starts: list[float], at: float) -> int:
    """The frame of a picture (`_clock`'s starts) at or before `at` seconds: 0 for a picture with one frame."""
    return max(bisect.bisect(starts, at) - 1, 0)


def _frame(path: Path, frame: tuple[int, float], max_px: int, region: Region | None = None) -> Image.Image | None:
    """The picture of one video frame (`(pts, seconds)`), from its own seek (one GOP), cut to `max_px` and then to `region`;
    None when it does not come out."""
    image = next((image for _, image in _decode(path, [frame], max_px)), None)
    return image if image is None or region is None else region.crop(image)


def _page_image(image: Image.Image, starts: list[float], page: int, max_px: int,
                region: Region | None = None) -> Image.Image:
    """Frame `page` of an opened picture (`_clock`'s starts), upright and cut to `max_px` and then to `region`."""
    if len(starts) > 1:
        image.seek(page)
    shown = _upright(image, max_px)
    return shown if region is None else region.crop(shown)


def _shots(store: Store, item: Item, ats: list[float], max_px: int,
           region: Region | None = None) -> list[Image.Image | None]:
    """What `look` shows of an item at each of `ats`, cut to `max_px` and then to `region`: a video's frame at or before it,
    an animation's or a TIFF's page; None for a frame that does not come out. Each distinct one is decoded once, a video
    frame from its own seek, so a long video costs its picks and not the time between them."""
    if form(item) == "video":
        path = store.path_of(item)
        times = _times(path)
        if not times:
            raise ValueError(f"item {item.id} has no frames")
        frames = [_frame_at(times, at) for at in ats]
        got = {frame: _frame(path, frame, max_px, region) for frame in set(frames)}
        return [got[frame] for frame in frames]
    image = _open(store, item, max_px)
    starts, _ = _clock(image)
    pages = [_page_at(starts, at) for at in ats]
    got = {page: _page_image(image, starts, page, max_px, region) for page in sorted(set(pages))}
    return [got[page] for page in pages]


class Pick(NamedTuple):
    """One moment of a sheet: its `slot` among the sheet's times, its `time`, the `item` that holds it and `at` seconds into it."""

    slot: int
    time: float
    item: Item
    at: float


def sheet(store: Store, picks: list[Pick], folder: Path, region: Region | None = None) -> dict:
    """Moments as one picture (DESIGN.md §11), for the `look` op. `picks` are in time order, each the picture `look` gives at
    its `at` of its item. A pick that looks the same (`_same`) as the first of the tile before it, and is in the next slot
    after that tile's last, joins that tile, which then spans their times. A pick that cannot be read is left out like a
    gap (all of them failing raises the first error). The tiles go row-wise in a grid as square as can be, each picture
    fitted inside its cell on dark grey, with its number in its bottom left corner, the whole sheet `LOOK_INLINE_PX` on
    its long side. Written as `_picture` writes: `{"image": ..., "tiles": [{"t0", "t1", "item", "at"}, ...]}`, a tile's
    item and `at` being its first pick's. With a `region` each tile is that part of its picture, cut from the picture as a plain
    one-moment `look` decodes it (not cut to a tile's size first), and `_same` compares those parts: only what moves inside
    the region makes tiles."""
    where: dict[int, list[int]] = {}
    for k, pick in enumerate(picks):
        where.setdefault(pick.item.id, []).append(k)
    images: list = [None] * len(picks)
    errors: list[Exception] = []
    for ks in where.values():
        try:
            shots = _shots(store, picks[ks[0]].item, [picks[k].at for k in ks],
                           SHEET_TILE_PX if region is None else 0, region)
        except UNREADABLE as error:  # an item that cannot be read is a gap
            errors.append(error)
            continue
        for k, image in zip(ks, shots):
            images[k] = image
    tiles: list[dict] = []
    for pick, image in zip(picks, images):
        if image is None:
            continue
        cells = _cells(image)
        if tiles and pick.slot == tiles[-1]["slot"] + 1 and _same(tiles[-1]["cells"], cells):
            tiles[-1].update(t1=pick.time, slot=pick.slot)
        else:
            tiles.append({"t0": pick.time, "t1": pick.time, "item": pick.item.id, "at": pick.at, "slot": pick.slot,
                          "image": image, "cells": cells})
    if not tiles:
        raise errors[0] if errors else ValueError("no frame of the range could be read")
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    shape = tiles[0]["image"].width / tiles[0]["image"].height  # a cell has the first picture's shape
    unit = LOOK_INLINE_PX / max(cols * shape, rows)  # a cell's height in pixels
    width, height = round(cols * shape * unit), round(rows * unit)
    xs, ys = [round(i * width / cols) for i in range(cols + 1)], [round(i * height / rows) for i in range(rows + 1)]
    canvas = Image.new("RGB", (width, height), DARK_GREY)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(16, round(unit / 7)))
    for k, tile in enumerate(tiles):
        row, col = divmod(k, cols)
        cell = xs[col + 1] - xs[col], ys[row + 1] - ys[row]
        image = ImageOps.contain(tile["image"], cell, Image.Resampling.LANCZOS)
        x, y = xs[col] + (cell[0] - image.width) // 2, ys[row] + (cell[1] - image.height) // 2
        canvas.paste(image, (x, y))
        _burn(draw, font, str(k + 1), x, y + image.height)
    return {"image": _picture(canvas, folder),
            "tiles": [{key: tile[key] for key in ("t0", "t1", "item", "at")} for tile in tiles]}


def _burn(draw: ImageDraw.ImageDraw, font, text: str, x: int, bottom: int) -> None:
    """`text` in white on a black box in the bottom left corner of a picture whose left edge is `x` and bottom `bottom`."""
    left, top, right, base = draw.textbbox((0, 0), text, font=font)
    pad = max(2, round(font.size / 6))
    high = base - top + 2 * pad
    draw.rectangle((x, bottom - high, x + right - left + 2 * pad - 1, bottom - 1), fill="black")
    draw.text((x + pad - left, bottom - high + pad - top), text, font=font, fill="white")


def _picture(image: Image.Image, folder: Path, zoom: bool = False) -> dict:
    """`image` as a PNG in `folder` at its own size and a JPEG beside it of at most `LOOK_INLINE_PX` on the long side: with
    `zoom` (a region's crop) one of exactly that, enlarged when the image is smaller."""
    name = uuid.uuid4().hex[:12]
    target, inline = folder / f"{name}.png", folder / f"{name}.jpg"
    image.save(target, format="PNG")
    small = image.copy()
    if zoom and max(image.size) < LOOK_INLINE_PX:
        scale = LOOK_INLINE_PX / max(image.size)
        small = small.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.LANCZOS)
    else:
        small.thumbnail((LOOK_INLINE_PX, LOOK_INLINE_PX), Image.Resampling.LANCZOS)
    small.convert("RGB").save(inline, format="JPEG", quality=85)
    return {"path": str(target), "inline": str(inline), "width": image.width, "height": image.height}
