"""A picture is a recording of one instant (DESIGN.md §5.1, §7): remembered or found in a folder it is an `image` item at
the time it was taken, by its EXIF, else its file's mtime, else when it was stored; it is read by the ranges that hold
that instant and no others, and `ask` tells a camera roll's pictures apart."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from conftest import DECIDER, span, wait_for, write_config
from PIL import ExifTags, Image

from mms import config as configuration
from mms import ingest, media
from mms.config import ffmpeg_exe, parse_when
from mms.store import Store

T0 = 1_699_999_980.0
RED, GREEN, BLUE = (220, 20, 20), (20, 200, 20), (20, 20, 220)
BASE = ExifTags.Base


def taken(text: str, offset: str | None = None) -> float:
    """The epoch time an EXIF time means: in this machine's zone, or at the offset."""
    moment = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    if offset is None:
        return moment.timestamp()
    zone = timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6]))
    return moment.replace(tzinfo=timezone(-zone if offset[0] == "-" else zone)).timestamp()


def jpeg(path: Path, rgb=RED, original: str | None = None, digitized: str | None = None, offset: str | None = None,
         digitized_offset: str | None = None, mtime: float | None = None) -> Path:
    """A 64x36 JPEG with the EXIF times named, and the mtime if one is given."""
    exif = Image.Exif()
    tags = exif.get_ifd(ExifTags.IFD.Exif)
    for tag, value in ((BASE.DateTimeOriginal, original), (BASE.DateTimeDigitized, digitized),
                       (BASE.OffsetTimeOriginal, offset), (BASE.OffsetTimeDigitized, digitized_offset)):
        if value is not None:
            tags[tag] = value
    Image.new("RGB", (64, 36), rgb).save(path, format="JPEG", exif=exif)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def png(path: Path, rgb=RED) -> Path:
    Image.new("RGB", (64, 36), rgb).save(path, format="PNG")
    return path


@pytest.fixture
def api(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + DECIDER)
    return served().api


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def stored(out: dict) -> dict:
    (entry,) = out["stored"]
    return entry


# the time of a picture

def test_a_picture_is_an_image_at_its_exif_time_in_this_zone_or_at_its_offset(api, store, tmp_path):
    local = stored(api.call("remember", path=str(jpeg(tmp_path / "a.jpg", original="2019:07:04 15:30:00"))))
    assert local["kind"] == "image" and local["time_from"] == "metadata" and local["t0"] == local["t1"]
    assert parse_when(local["t0"]) == pytest.approx(taken("2019:07:04 15:30:00"), abs=0.001)
    row = store.item(local["id"])
    assert row.kind == "image" and row.t0 == row.t1 == taken("2019:07:04 15:30:00") and row.meta == {"media": "image/jpeg", "time_from": "metadata"}
    assert row.path.startswith("files/agent/") and (store.home / row.path).is_file() and row.in_place == 0

    zoned = stored(api.call("remember", path=str(jpeg(tmp_path / "b.jpg", original="2019:07:04 15:30:00", offset="+02:00"))))
    assert parse_when(zoned["t0"]) == pytest.approx(datetime(2019, 7, 4, 13, 30, tzinfo=timezone.utc).timestamp(), abs=0.001)
    west = stored(api.call("remember", path=str(jpeg(tmp_path / "c.jpg", original="2019:07:04 15:30:00", offset="-05:30"))))
    assert parse_when(west["t0"]) == pytest.approx(datetime(2019, 7, 4, 21, 0, tzinfo=timezone.utc).timestamp(), abs=0.001)

    digitized = stored(api.call("remember", path=str(jpeg(tmp_path / "d.jpg", digitized="2018:01:02 03:04:05"))))
    assert digitized["time_from"] == "metadata" and parse_when(digitized["t0"]) == pytest.approx(taken("2018:01:02 03:04:05"), abs=0.001)
    both = stored(api.call("remember", path=str(jpeg(tmp_path / "e.jpg", original="2019:07:04 15:30:00",
                                                     digitized="2018:01:02 03:04:05", digitized_offset="+09:00"))))
    assert parse_when(both["t0"]) == pytest.approx(taken("2019:07:04 15:30:00"), abs=0.001), "the original comes first"
    other = stored(api.call("remember", path=str(jpeg(tmp_path / "f.jpg", original="0000:00:00 00:00:00",
                                                      digitized="2018:01:02 03:04:05", digitized_offset="+09:00"))))
    assert parse_when(other["t0"]) == pytest.approx(taken("2018:01:02 03:04:05", "+09:00"), abs=0.001), "an offset goes with its own time"


def test_without_a_usable_exif_time_a_picture_is_at_its_mtime(api, tmp_path):
    mtime = T0 + 5000
    plain = stored(api.call("remember", path=str(jpeg(tmp_path / "plain.jpg", mtime=mtime))))
    assert plain["time_from"] == "mtime" and parse_when(plain["t0"]) == pytest.approx(mtime, abs=0.001)
    odd = stored(api.call("remember", path=str(jpeg(tmp_path / "odd.jpg", original="2019:07:04 15:30:00", offset="+25:00",
                                                    mtime=mtime + 20))))
    assert odd["time_from"] == "metadata" and parse_when(odd["t0"]) == pytest.approx(taken("2019:07:04 15:30:00"), abs=0.001), \
        "an offset that is no offset is ignored"
    for n, bad in enumerate(("", "0000:00:00 00:00:00", "   ", "last tuesday", "2019-07-04 15:30:00", "2019:13:45 25:61:61")):
        one = stored(api.call("remember", path=str(jpeg(tmp_path / f"bad{n}.jpg", original=bad, mtime=mtime + n))))
        assert one["time_from"] == "mtime" and parse_when(one["t0"]) == pytest.approx(mtime + n, abs=0.001), repr(bad)
    # a PNG, a damaged JPEG and a HEIC (no plugin reads its EXIF) are stored too
    only = png(tmp_path / "shot.png")
    os.utime(only, (mtime + 10, mtime + 10))
    cut = jpeg(tmp_path / "cut.jpg", original="2019:07:04 15:30:00")
    cut.write_bytes(cut.read_bytes()[:150])
    os.utime(cut, (mtime + 11, mtime + 11))
    heic = tmp_path / "IMG_1.heic"
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 40)
    os.utime(heic, (mtime + 12, mtime + 12))
    for n, path in enumerate((only, cut, heic), 10):
        one = stored(api.call("remember", path=str(path)))
        assert one["kind"] == "image" and one["time_from"] == "mtime" and parse_when(one["t0"]) == pytest.approx(mtime + n, abs=0.001), path


def test_a_time_after_now_is_not_believed(api, tmp_path):
    now = time.time()
    future = datetime.fromtimestamp(now + 40 * 86400).strftime("%Y:%m:%d %H:%M:%S")
    mtime = T0 + 7000
    one = stored(api.call("remember", path=str(jpeg(tmp_path / "clock.jpg", original=future, mtime=mtime))))
    assert one["time_from"] == "mtime" and parse_when(one["t0"]) == pytest.approx(mtime, abs=0.001), "a camera clock set wrong"
    both = stored(api.call("remember", path=str(jpeg(tmp_path / "copy.jpg", rgb=GREEN, original=future, mtime=now + 86400))))
    assert both["time_from"] == "now" and parse_when(both["t0"]) == pytest.approx(time.time(), abs=10)
    given = stored(api.call("remember", path=str(jpeg(tmp_path / "told.jpg", rgb=BLUE, original="2019:07:04 15:30:00", mtime=mtime)),
                            t=T0))
    assert given["time_from"] == "given" and parse_when(given["t0"]) == pytest.approx(T0, abs=0.001), "the caller's t wins"


def test_a_recording_in_the_future_is_believed_no_more_than_a_picture(api, tmp_path):
    ffmpeg = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x36:r=5:d=3", "-c:v", "libx264",
              "-pix_fmt", "yuv420p"]
    now = time.time()
    tagged, plain = tmp_path / "tagged.mp4", tmp_path / "plain.mp4"
    subprocess.run([*ffmpeg, "-metadata", "creation_time=2099-01-01T00:00:00Z", str(tagged)], check=True)
    subprocess.run([*ffmpeg, str(plain)], check=True)
    os.utime(tagged, (T0 + 100, T0 + 100))
    os.utime(plain, (now + 86400, now + 86400))
    one = stored(api.call("remember", path=str(tagged)))
    assert one["time_from"] == "mtime" and parse_when(one["t1"]) == pytest.approx(T0 + 100, abs=0.3)
    two = stored(api.call("remember", path=str(plain)))
    assert two["time_from"] == "now" and parse_when(two["t1"]) == pytest.approx(time.time(), abs=10), "it ends now"


def test_the_same_picture_again_is_unchanged_and_an_svg_stays_a_text_version(api, store, tmp_path):
    path = jpeg(tmp_path / "a.jpg", original="2019:07:04 15:30:00")
    first = stored(api.call("remember", path=str(path)))
    again = api.call("remember", path=str(path), t=T0)
    assert again["stored"] == [] and again["unchanged"] == [{**first, "name": "a.jpg"}],         "the stored item's own time and where it came from, not this call's t"
    copy = tmp_path / "copy.jpg"
    shutil.copy(path, copy)
    assert api.call("remember", path=str(copy))["unchanged"][0]["id"] == first["id"], "by content, whatever its name"
    assert [i.kind for i in store.items_of("agent")] == ["image"]
    svg = tmp_path / "logo.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>kettle</text></svg>', encoding="utf-8")
    made = api.call("remember", path=str(svg), t=T0)
    entry = stored(made)
    assert entry["kind"] == "file" and "time_from" not in entry and "time_from" not in made
    row = store.item(entry["id"])
    assert row.kind == "file" and row.meta["media"] == "image/svg+xml" and row.t0 == T0
    assert media.is_picture("image/png") and media.is_picture("image/heic") and not media.is_picture("image/svg+xml")


# the range

def test_a_picture_is_in_the_ranges_that_hold_its_instant_and_in_no_other(api, store, tmp_path):
    old = stored(api.call("remember", path=str(jpeg(tmp_path / "old.jpg", original="2019:07:04 15:30:00"))))
    recent = stored(api.call("remember", path=str(jpeg(tmp_path / "new.jpg", rgb=GREEN, original="2024:03:05 10:00:00"))))
    t_old, t_new = taken("2019:07:04 15:30:00"), taken("2024:03:05 10:00:00")

    def found(since: float, until: float) -> list[int]:
        return [row["id"] for row in api.call("find", since=since, until=until, sources=["agent"])["results"]]

    assert found(t_new - 3600, t_new + 3600) == [recent["id"]], "a 2019 picture is not in a range in 2024"
    assert found(t_old - 3600, t_old + 3600) == [old["id"]]
    assert found(t_old - 3600, t_new + 3600) == [recent["id"], old["id"]]
    assert found(t_old + 1, t_new - 1) == [] and found(t_old - 1, t_old + 1) == [old["id"]]
    assert [i.id for i in store.select(t_new - 60, t_new + 60, ["agent"])] == [recent["id"]]
    row = api.call("find", since=t_old - 60, until=t_old + 60)["results"][0]
    assert (row["kind"], row["media"], row["t0"], row["t1"]) == ("image", "image/jpeg", old["t0"], old["t0"])
    assert row["look"] == {"item": old["id"], "at": 0}


# a camera roll: pictures taken at different times are different sightings

def roll(folder: Path) -> list[float]:
    """Four red photos an hour apart in 2024, and a green one, none with the same mtime as its capture time."""
    folder.mkdir()
    times = []
    for n in range(4):
        text = f"2024:03:05 1{n}:00:00"
        jpeg(folder / f"IMG_{n}.jpg", original=text, mtime=T0 + n)
        times.append(taken(text))
    jpeg(folder / "IMG_9.jpg", rgb=GREEN, original="2024:03:05 14:00:00", mtime=T0 + 9)
    return times


def asked(api, since: float, until: float, **params) -> dict:
    job = api.call("jobs/start", script="ask", params={"question": "Is it red?", "since": since, "until": until, **params}, wait_s=50)
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    assert job["state"] == "done", job
    return job["result"]


def test_a_remembered_folder_of_pictures_is_one_sighting_each(api, tmp_path):
    times = roll(tmp_path / "roll")
    out = api.call("remember", path=str(tmp_path / "roll"), source="phone")
    assert sorted(e["time_from"] for e in out["stored"]) == ["metadata"] * 5
    result = asked(api, times[0] - 600, times[-1] + 7200, sources=["phone"])
    sightings = sorted(span(s)[0] for s in result["sightings"])
    assert sightings == pytest.approx(times, abs=0.001), "four red pictures, four sightings, each at its capture time"
    assert all(s["look"]["at"] == 0 and s["kind"] == "image" for s in result["sightings"])
    narrow = asked(api, times[1] - 60, times[1] + 60, sources=["phone"])
    assert [span(s)[0] for s in narrow["sightings"]] == pytest.approx([times[1]], abs=0.001), "since and until narrow pictures"
    assert asked(api, taken("2019:01:01 00:00:00"), taken("2019:12:31 00:00:00"), sources=["phone"])["sightings"] == []


def test_a_folder_source_of_pictures_is_one_sighting_each(home, served, tmp_path):
    times = roll(tmp_path / "roll")
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + DECIDER +
                 f'[sources.roll]\nkind = "folder"\npath = "{(tmp_path / "roll").as_posix()}"\nrescan_s = 0.2\n')
    api = served().api
    store = Store(home)
    wait_for(lambda: len(store.items_of("roll")) == 5, what="the five pictures")
    assert sorted(i.t0 for i in store.items_of("roll")) == pytest.approx(sorted([*times, taken("2024:03:05 14:00:00")]), abs=0.001)
    result = asked(api, times[0] - 600, times[-1] + 7200, sources=["roll"])
    assert sorted(span(s)[0] for s in result["sightings"]) == pytest.approx(times, abs=0.001)
    store.close()


# a folder source

def folder_watcher(home: Path, store: Store, folder: Path, extra: str = "") -> tuple[ingest.Ingest, configuration.Source]:
    write_config(home, "[memory]\nfree_gb = 0\n" + f'[sources.roll]\nkind = "folder"\npath = "{folder.as_posix()}"\n'
                                                   f'glob = "*"\nrescan_s = 0.2\n{extra}\n')
    config = configuration.load(home)
    return ingest.Ingest(store, config), config.sources["roll"]


def test_a_folder_copies_a_picture_as_a_still_and_a_big_one_stays_a_version_in_place(home, store, tmp_path):
    folder = tmp_path / "roll"
    folder.mkdir()
    jpeg(folder / "small.jpg", original="2019:07:04 15:30:00", mtime=T0)
    big = jpeg(folder / "big.jpg", rgb=GREEN, original="2019:07:04 15:31:00", mtime=T0)
    big.write_bytes(big.read_bytes() + b"\x00" * 3000)
    notes = folder / "notes.md"
    notes.write_text("the kettle")
    watcher_, source = folder_watcher(home, store, folder, "copy_max_mb = 0.003")
    now = time.time()
    while watcher_.scan(source, now):
        pass
    items = {i.name: i for i in store.items_of("roll")}
    small = items["small.jpg"]
    assert (small.kind, small.t0, small.t1, small.in_place) == ("image", taken("2019:07:04 15:30:00"), taken("2019:07:04 15:30:00"), 0)
    assert small.path.startswith("files/roll/") and small.meta == {"media": "image/jpeg", "time_from": "metadata"}
    assert (items["big.jpg"].kind, items["big.jpg"].in_place, items["big.jpg"].t0) == ("file", 1, now), "too big to copy: a version"
    assert items["notes.md"].kind == "file" and items["notes.md"].t0 == now
    assert store.current_version("roll", "big.jpg").id == items["big.jpg"].id and store.current_version("roll", "small.jpg") is None
    ids = [i.id for i in store.items_of("roll")]
    while watcher_.scan(source, now + 10):
        pass
    assert [i.id for i in store.items_of("roll")] == ids, "nothing twice"
    small_file = folder / "small.jpg"
    small_file.unlink()
    while watcher_.scan(source, now + 20):
        pass
    assert store.item(small.id).state == "ok" and store.item(small.id).t1 == small.t0, "a picture whose file vanishes stays"


def test_a_folder_copies_a_video_at_its_creation_time_or_its_mtime_less_its_length(home, store, tmp_path):
    folder = tmp_path / "roll"
    folder.mkdir()
    make = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x36:r=5:d=3", "-c:v", "libx264",
            "-pix_fmt", "yuv420p"]
    subprocess.run([*make, "-metadata", "creation_time=2026-10-30T12:00:00Z", str(folder / "tagged.mp4")], check=True)
    subprocess.run([*make, str(folder / "plain.mp4")], check=True)
    os.utime(folder / "plain.mp4", (T0 + 1000, T0 + 1000))
    (folder / "broken.mp4").write_bytes(b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00not a recording")
    watcher_, source = folder_watcher(home, store, folder)
    while watcher_.scan(source, time.time() + 60):  # the files are settled by then
        pass
    items = {i.name: i for i in store.items_of("roll")}
    tagged, plain = items["tagged.mp4"], items["plain.mp4"]
    assert tagged.kind == plain.kind == "video" and tagged.in_place == 0
    assert tagged.t0 == pytest.approx(parse_when("2026-10-30T12:00:00Z"), abs=0.001) and tagged.t1 - tagged.t0 == pytest.approx(3, abs=0.6)
    assert plain.t1 == pytest.approx(T0 + 1000, abs=0.01) and plain.t1 - plain.t0 == pytest.approx(3, abs=0.6)
    assert items["broken.mp4"].kind == "video" and items["broken.mp4"].t1 == items["broken.mp4"].t0, "kept, and reported when read"


# look

def test_look_at_a_picture_by_item_and_by_source_and_the_time_read_back_from_a_result(api, store, tmp_path):
    t = T0 + 0.4567
    red = stored(api.call("remember", path=str(jpeg(tmp_path / "red.jpg", RED)), t=t, source="roll"))
    green = stored(api.call("remember", path=str(jpeg(tmp_path / "green.jpg", GREEN)), t=T0 + 100, source="roll"))
    by_item = api.call("look", item=red["id"])
    assert by_item["kind"] == "image" and by_item["item"] == red["id"] and by_item["at"] is None and by_item["source"] == "roll"
    assert by_item["t"] == configuration.iso(t), "the still's own time"
    assert api.call("look", item=red["id"], at=0)["t"] == by_item["t"]
    row = api.call("find", sources=["roll"], until=T0 + 50)["results"][0]
    assert row["t0"] == by_item["t"] and parse_when(row["t0"]) < t, "cut to the millisecond"
    by_time = api.call("look", source="roll", t=row["t0"])
    assert by_time["item"] == red["id"] and by_time["at"] is None, "a picture within the millisecond is at it"
    far = api.call("look", source="roll", t=T0 + 50)
    assert far["item"] == red["id"] and far["t"] == by_item["t"], "the latest picture at or before t, shown at its own time"
    assert api.call("look", source="roll", t=T0 + 150)["item"] == green["id"]
    nothing = api.failure("look", source="roll", t=T0 - 1)
    assert (nothing.code, nothing.field) == ("not_found", "source") and "nearest" in nothing.error["hint"]
    assert api.call("look", moments=[{"source": "roll", "t": T0 + 50}])["moments"][0]["item"] == red["id"]


def test_a_picture_competes_with_the_versions_current_at_t_and_the_newest_wins(api, store, tmp_path):
    store.add_version("roll", "note.md", T0 + 100, data=b"the kettle\n")
    before = stored(api.call("remember", path=str(jpeg(tmp_path / "a.jpg")), t=T0 + 50, source="roll"))
    assert api.call("look", source="roll", t=T0 + 120)["kind"] == "text", "the version is newer"
    after = stored(api.call("remember", path=str(jpeg(tmp_path / "b.jpg", GREEN)), t=T0 + 150, source="roll"))
    assert api.call("look", source="roll", t=T0 + 120)["kind"] == "text"
    shown = api.call("look", source="roll", t=T0 + 200)
    assert shown["kind"] == "image" and shown["item"] == after["id"], "the picture is newer than the version"
    assert api.call("look", source="roll", t=T0 + 60)["item"] == before["id"]
    assert api.call("look", source="roll", t=T0 + 200)["t"] == configuration.iso(T0 + 150), "at the picture's own time"


def test_a_sheet_of_pictures_leaves_out_one_before_since(api, store, tmp_path):
    ids = [stored(api.call("remember", path=str(jpeg(tmp_path / f"{n}.jpg", rgb)), t=T0 + n * 100, source="roll"))["id"]
           for n, rgb in enumerate((RED, GREEN, BLUE))]
    sheet = api.call("look", source="roll", since=T0 + 50, until=T0 + 250)  # 9 times 25 s apart: 50, 75 are before the second picture
    assert [tile["look"] for tile in sheet["tiles"]] == [{"item": ids[1], "at": 0.0}, {"item": ids[2], "at": 0.0}]
    first, second = sheet["tiles"]
    assert (first["t"], second["t"]) == (configuration.iso(T0 + 100), configuration.iso(T0 + 200)), "when each photo was taken"
    all_three = api.call("look", source="roll", since=T0, until=T0 + 200)
    assert [tile["look"]["item"] for tile in all_three["tiles"]] == ids
    after = api.failure("look", source="roll", since=T0 + 210, until=T0 + 400)  # the last picture is before the range
    assert (after.status, after.code, after.field) == (404, "not_found", "source")


# a file still being written, a name that changes type, a time out of range

def test_a_media_file_that_may_still_be_written_waits_for_the_next_pass(home, store, tmp_path):
    folder = tmp_path / "roll"
    folder.mkdir()
    fresh = jpeg(folder / "fresh.jpg", original="2019:07:04 15:30:00")
    note = folder / "note.md"
    note.write_text("the kettle")
    watcher_, source = folder_watcher(home, store, folder)  # rescan_s = 0.2: a file waits 0.4 s after its last write
    mtime = fresh.stat().st_mtime
    assert watcher_.scan(source, mtime + 0.1) is False
    assert [i.name for i in store.items_of("roll")] == ["note.md"], "the picture is not copied yet; the text version is"
    assert store.cursor("roll", "fresh.jpg") is None, "and no cursor: the next pass looks again"
    watcher_.scan(source, mtime + 0.5)
    watcher_.scan(source, mtime + 5)
    assert [i.name for i in store.items_of("roll")].count("fresh.jpg") == 1, "copied once, when settled"
    assert store.cursor("roll", "fresh.jpg") is not None
    jpeg(folder / "ahead.jpg", rgb=GREEN, original="2019:07:05 15:30:00", mtime=time.time() + 86400)
    watcher_.scan(source, time.time())
    assert any(i.name == "ahead.jpg" for i in store.items_of("roll")), "a future mtime is settled, not held back forever"


def test_a_folder_picture_with_no_exif_is_at_its_mtime(home, store, tmp_path):
    folder = tmp_path / "roll"
    folder.mkdir()
    jpeg(folder / "plain.jpg", mtime=T0 + 123)
    watcher_, source = folder_watcher(home, store, folder)
    while watcher_.scan(source, time.time()):
        pass
    (item,) = store.items_of("roll")
    assert item.kind == "image" and item.t0 == item.t1 == pytest.approx(T0 + 123, abs=0.001) and item.meta["time_from"] == "mtime"


def test_a_name_that_becomes_a_picture_ends_its_old_version(home, store, tmp_path):
    folder = tmp_path / "roll"
    folder.mkdir()
    scan = folder / "scan.png"
    scan.write_bytes(b"\x00" + b"x" * 4000)  # binary, and over copy_max_mb = 0.003: indexed in place
    os.utime(scan, (T0, T0))
    watcher_, source = folder_watcher(home, store, folder, "copy_max_mb = 0.003")
    now = time.time()
    while watcher_.scan(source, now):
        pass
    (old,) = store.items_of("roll")
    assert old.kind == "file" and old.in_place == 1 and old.t1 is None
    jpeg(scan, original="2019:07:04 15:30:00", mtime=T0 + 10)  # the same name, now a small picture
    while watcher_.scan(source, now + 60):
        pass
    new, old = store.items_of("roll")  # by time: the picture of 2019, then the version
    assert old.kind == "file" and (new.kind, new.name) == ("image", "scan.png") and new.t0 == taken("2019:07:04 15:30:00")
    assert old.t1 == now + 60 and old.state == "gone", "the in-place version ended: nothing reads the live path for an old time"
    assert store.current_version("roll", "scan.png") is None


def test_remembering_a_picture_under_the_name_of_a_text_version_ends_the_version(api, store, tmp_path):
    notes = tmp_path / "plan"
    notes.write_text("the plan\n")
    api.call("remember", path=str(notes), source="roll")
    assert store.current_version("roll", "plan") is not None
    picture = jpeg(tmp_path / "other.jpg", original="2019:07:04 15:30:00")
    out = stored(api.call("remember", path=str(picture), name="plan", source="roll"))
    assert out["kind"] == "image" and store.current_version("roll", "plan") is None
    (old,) = [i for i in store.items_of("roll") if i.kind == "file"]
    assert old.t1 is not None


def test_a_recording_must_lie_within_the_epoch_and_now(api, tmp_path):
    now = time.time()
    early = stored(api.call("remember", path=str(jpeg(tmp_path / "old.jpg", original="1960:01:02 03:04:05", offset="+01:00",
                                                      mtime=T0 + 3))))
    assert early["time_from"] == "mtime" and parse_when(early["t0"]) == pytest.approx(T0 + 3, abs=0.001), "before 1970 is not believed"
    make = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x36:r=5:d=3", "-c:v", "libx264",
            "-pix_fmt", "yuv420p"]
    video = tmp_path / "ends_ahead.mp4"
    soon = datetime.fromtimestamp(now + 1, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # starts now, ends 3 s from now
    subprocess.run([*make, "-metadata", f"creation_time={soon}", str(video)], check=True)
    os.utime(video, (T0 + 500, T0 + 500))
    one = stored(api.call("remember", path=str(video)))
    assert one["time_from"] == "mtime" and parse_when(one["t1"]) == pytest.approx(T0 + 500, abs=0.3), "its end is in the future"
