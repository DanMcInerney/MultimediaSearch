"""remember and look (TESTBED.md item 9): a note, a file, a folder and a recording stored at their real times, and one
moment of what is stored shown back, each through the daemon's API."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import mms, write_config
from PIL import Image

from mms.config import ffmpeg_exe, parse_when
from mms.store import Store

T0 = 1_699_999_980.0
PROJECT = "git:0123456789abcdef"
PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0), (255, 128, 0)]


def ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("media")
    colour = lambda name: ["-f", "lavfi", "-i", f"color=c={name}:s=160x90:r=5:d=3"]  # noqa: E731
    video = folder / "colours.mkv"  # blue 0-3, red 3-6, blue 6-9
    ffmpeg(*colour("blue"), *colour("red"), *colour("blue"), "-filter_complex", "concat=n=3:v=1:a=0", "-c:v", "libx264",
           "-g", "10", "-pix_fmt", "yuv420p", str(video))
    tagged = folder / "tagged.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x36:r=5:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-metadata",
           "creation_time=2026-10-30T12:00:00Z", str(tagged))
    audio = folder / "tone.mka"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4,volume=4", "-c:a", "libopus", str(audio))
    wide = folder / "wide.mkv"
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=1920x1080:r=2:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(wide))
    nine = lambda size: [arg for r, g, b in PALETTE for arg in ("-f", "lavfi", "-i", f"color=c=0x{r:02X}{g:02X}{b:02X}:s={size}:r=5:d=1")] + [  # noqa: E731
        "-filter_complex", f"concat=n={len(PALETTE)}:v=1:a=0", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p"]
    changing, portrait, still = folder / "nine.mkv", folder / "portrait.mkv", folder / "still.mkv"  # a colour a second, for 9 s
    ffmpeg(*nine("160x90"), str(changing))
    ffmpeg(*nine("90x160"), str(portrait))
    ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=160x90:r=5:d=9", "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(still))
    return {"video": video, "tagged": tagged, "audio": audio, "wide": wide, "nine": changing, "portrait": portrait, "still": still}


@pytest.fixture
def api(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 1\n")
    return served().api


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def color(path: str) -> tuple[float, float, float]:
    with Image.open(path) as picture:
        r, g, b = np.asarray(picture.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    return r, g, b


# remember

def test_a_note_is_stored_at_its_time_with_its_name_and_expiry(home, api, store):
    out = api.call("remember", note="the build needs python 3.12", t=T0, name="build", expires="30d")
    assert "time_from" not in out and out["unchanged"] == [] and out["source"] == "agent"
    (stored,) = out["stored"]
    assert stored["type"] == "record" and parse_when(stored["t0"]) == pytest.approx(T0, abs=0.001) and "t1" not in stored
    assert parse_when(out["expires"]) == pytest.approx(time.time() + 30 * 86400, abs=60)
    (record,) = store.records(sources=["agent"])
    assert record.t == T0 and json.loads(record.data) == {"text": "the build needs python 3.12", "name": "build"}
    assert record.id == stored["id"] and record.expires == pytest.approx(time.time() + 30 * 86400, abs=60)
    again = api.call("remember", note="the build needs python 3.12", t=T0, name="build")
    assert again["stored"][0]["id"] != stored["id"], "a note is stored again"
    now = api.call("remember", note="no time given")
    assert parse_when(now["stored"][0]["t0"]) == pytest.approx(time.time(), abs=10)
    assert len(api.call("find", text="python", until=T0 + 10, since=T0 - 10)["results"]) == 2


def test_a_file_again_is_unchanged_and_a_changed_one_is_a_new_version(home, api, store, tmp_path):
    note = tmp_path / "plan.md"
    note.write_bytes(b"# the plan\n")
    first = api.call("remember", path=str(note), t=T0)
    (stored,) = first["stored"]
    assert stored["type"] == "item" and stored["kind"] == "file" and stored["name"] == "plan.md" and first["unchanged"] == []
    again = api.call("remember", path=str(note), t=T0 + 10)
    assert again["stored"] == [] and again["unchanged"] == [{"name": "plan.md", "id": stored["id"]}]
    note.write_bytes(b"# the plan, changed\n")
    changed = api.call("remember", path=str(note), t=T0 + 20, name="plan-v2.md")
    assert changed["stored"][0]["name"] == "plan-v2.md"
    note.write_bytes(b"# the plan, once more\n")
    api.call("remember", path=str(note), t=T0 + 30)
    old, new = (i for i in store.items_of("agent") if i.name == "plan.md")
    assert old.t1 == new.t0 == T0 + 30 and new.t1 is None and store.read_bytes(old) == b"# the plan\n"
    assert api.call("find", as_of=T0 + 15, text="plan")["results"][0]["id"] == old.id


def test_a_folder_is_a_version_of_each_file_under_it(home, api, store, tmp_path):
    folder = tmp_path / "notes"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.md").write_text("alpha")
    (folder / "sub" / "b.md").write_text("beta")
    out = api.call("remember", path=str(folder), t=T0)
    assert sorted(s["name"] for s in out["stored"]) == ["notes/a.md", "notes/sub/b.md"]
    again = api.call("remember", path=str(folder), t=T0 + 1)
    assert again["stored"] == [] and sorted(u["name"] for u in again["unchanged"]) == ["notes/a.md", "notes/sub/b.md"]
    (folder / "a.md").write_text("alpha two")
    part = api.call("remember", path=str(folder), t=T0 + 2)
    assert [s["name"] for s in part["stored"]] == ["notes/a.md"] and [u["name"] for u in part["unchanged"]] == ["notes/sub/b.md"]


def test_a_recording_is_a_segment_at_its_given_time_else_its_metadatas_else_its_mtime(home, api, store, media, tmp_path):
    given = api.call("remember", path=str(media["video"]), t=T0 + 500, source="clips")
    (item,) = given["stored"]
    assert item["time_from"] == "given" and item["kind"] == "video" and item["name"] == "colours.mkv"
    assert parse_when(item["t0"]) == pytest.approx(T0 + 500, abs=0.001) and parse_when(item["t1"]) == pytest.approx(T0 + 509, abs=0.6)
    row = store.item(item["id"])
    assert row.kind == "video" and row.t0 == T0 + 500 and row.path.startswith("files/clips/") and row.project is None

    tagged = api.call("remember", path=str(media["tagged"]), source="clips")
    stamp = parse_when("2026-10-30T12:00:00Z")
    assert tagged["stored"][0]["time_from"] == "metadata" and parse_when(tagged["stored"][0]["t0"]) == pytest.approx(stamp, abs=0.001)

    plain = tmp_path / "plain.mka"
    shutil.copy(media["audio"], plain)
    mtime = T0 + 1000
    os.utime(plain, (mtime, mtime))
    guessed = api.call("remember", path=str(plain), source="clips")
    (heard,) = guessed["stored"]
    assert heard["time_from"] == "mtime" and heard["kind"] == "audio"
    assert parse_when(heard["t1"]) == pytest.approx(mtime, abs=0.2) and parse_when(heard["t0"]) == pytest.approx(mtime - 4, abs=0.3), \
        "the file ended when it was last written"

    again = api.call("remember", path=str(media["video"]), t=T0 + 900, source="clips")
    assert again["stored"] == [] and again["unchanged"] == [item], "by its SHA-256: the stored item, not this call's t"
    assert [i.id for i in store.items_of("clips") if i.name == "colours.mkv"] == [item["id"]]
    copy = tmp_path / "renamed.mkv"
    shutil.copy(media["video"], copy)
    assert api.call("remember", path=str(copy), source="clips")["unchanged"][0]["id"] == item["id"], "the same recording, by content"


def test_remember_refuses_what_it_cannot_take(home, api, tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 1_100_000)
    too_big = api.failure("remember", path=str(big))
    assert (too_big.status, too_big.code, too_big.field) == (413, "too_big", "path") and "remember_max_mb" in too_big.error["message"]
    crowd = tmp_path / "crowd"
    crowd.mkdir()
    for i in range(1001):
        (crowd / f"{i}.txt").write_text("x")
    assert api.failure("remember", path=str(crowd)).code == "too_big"
    assert api.failure("remember", path=str(tmp_path / "none.txt")).code == "not_found"
    assert api.failure("remember").field == "note" and api.failure("remember", note="a", path=str(big)).field == "note"
    assert api.failure("remember", note="x" * 100_001).field == "note"
    assert api.failure("remember", path="relative.txt").field == "path"
    assert api.failure("remember", note="a", expires="yesterday-ish").field == "expires"
    assert api.failure("remember", note="a", expires="2020-01-01T00:00:00Z").field == "expires"
    assert api.failure("remember", note="a", t="when").field == "t"
    assert api.failure("remember", note="a", source="bad name").field == "source"
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00not a recording")  # an MP4 header, then junk
    assert api.failure("remember", path=str(junk)).code == "bad_request"
    assert api.failure("remember", note="a", nope=1).field == "nope"


def test_below_the_floor_remember_is_refused(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 1000000000\n")
    api = served().api
    refused = api.failure("remember", note="a note")
    assert (refused.status, refused.code) == (507, "floor") and refused.error["hint"]
    elsewhere = home.parent / "elsewhere.txt"
    elsewhere.write_text("x")
    assert api.failure("remember", path=str(elsewhere)).code == "floor"
    assert mms(home, "remember", "a note", check=False).returncode == 2
    assert Store(home).records() == []


def test_remember_into_a_source_that_is_something_else_is_refused(home, served, tmp_path):
    write_config(home, f'[memory]\nloop_s = 0.1\nfree_gb = 0\n[sources.notes]\nkind = "folder"\npath = "{tmp_path.as_posix()}"\n'
                       '[sources.mine]\nkind = "push"\nkeep = { max_gb = 1 }\n')
    api = served().api
    error = api.failure("remember", note="hello", source="notes")
    assert (error.status, error.code, error.field) == (409, "wrong_kind", "source")
    assert api.call("remember", note="hello", source="mine")["source"] == "mine"


def test_a_note_belongs_to_the_callers_project(home, api, store):
    caller = {"project": PROJECT, "project_name": "mine", "project_root": "/x", "session": "s-0123456789ab", "client": "t"}
    stored = api.call("remember", note="secret kettle", caller=caller)["stored"][0]
    row = store.records(sources=["agent"])[0]
    assert (row.project, row.session) == (PROJECT, "s-0123456789ab") and row.id == stored["id"]
    assert api.call("find", text="kettle")["results"] == [], "another caller sees ambient rows only"
    mine = api.call("find", text="kettle", caller=caller)
    assert mine["results"][0]["project"] == "mine" and mine["searched"]["project"] == {"id": PROJECT, "name": "mine"}
    assert api.call("find", text="kettle", scope="all")["results"][0]["project"] == "mine"
    assert store.db().execute("SELECT name, root FROM projects WHERE id = ?", (PROJECT,)).fetchone()[:] == ("mine", "/x")
    assert store.db().execute("SELECT client, project FROM sessions").fetchone()[:] == ("t", PROJECT)


# look

def put_video(store: Store, media: dict, source: str = "cam", t0: float = T0, name: str = "video", **who) -> int:
    target = store.streams / source / "launch" / f"{int(t0)}.mkv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(media[name], target)
    return store.add_item(source, "video", t0, t0 + 9, store.relative(target), target.stat().st_size, **who)


def test_look_at_an_item_and_at_gives_the_frame_by_its_colour_with_the_labels_over_it(home, api, store, media):
    item = put_video(store, media)
    for i in range(12):
        store.label(T0 + 1, T0 + 5, f"label {i}", "agent", source="cam")
    store.label(T0 + 7, T0 + 8, "elsewhere", "agent", source="cam")
    red = api.call("look", item=item, at=4.2)
    assert red["kind"] == "image" and red["item"] == item and red["source"] == "cam" and red["at"] == 4.2
    assert parse_when(red["t"]) == pytest.approx(T0 + 4.2, abs=0.001)
    r, g, b = color(red["image"]["path"])
    assert r > 200 and g < 60 and b < 60
    assert red["image"]["width"] == 160 and red["image"]["height"] == 90
    assert Path(red["image"]["inline"]).suffix == ".jpg" and Path(red["image"]["inline"]).is_file()
    assert Image.open(red["image"]["inline"]).size == (160, 90)
    assert len(red["labels"]) == 10 and red["labels"][0].keys() == {"t0", "t1", "by", "text"}, "up to 10 labels over the moment"
    assert all("elsewhere" != label["text"] for label in red["labels"])
    blue = api.call("look", item=item, at=1)
    r, g, b = color(blue["image"]["path"])
    assert b > 200 and r < 60
    start = api.call("look", item=item)
    assert start["at"] == 0 and color(start["image"]["path"])[2] > 200, "no `at` is the start"
    assert Path(red["image"]["path"]).parent == home / "looks"
    assert api.failure("look", item=item, at=30).field == "at" and api.failure("look", item=item, at=30).code == "bad_request"


def test_look_gives_a_full_size_png_and_an_inline_jpeg_of_at_most_1568_px(home, api, store, media):
    item = put_video(store, media, name="wide")
    shown = api.call("look", item=item, at=0.5)["image"]
    assert (shown["width"], shown["height"]) == (1920, 1080)
    assert Image.open(shown["path"]).size == (1920, 1080) and Image.open(shown["inline"]).size == (1568, 882)
    assert Image.open(shown["inline"]).format == "JPEG" and Image.open(shown["path"]).format == "PNG"


def test_look_at_a_sound_gives_a_five_second_wav_and_its_loudness(home, api, store, media):
    target = store.streams / "mic" / "launch" / "t.mka"
    target.parent.mkdir(parents=True)
    shutil.copy(media["audio"], target)
    item = store.add_item("mic", "audio", T0, T0 + 4, store.relative(target), target.stat().st_size)
    heard = api.call("look", item=item, at=0)
    assert heard["kind"] == "audio" and heard["audio"]["seconds"] == 4.0 and heard["audio"]["rms"] > 0.1
    import wave
    with wave.open(heard["audio"]["path"]) as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 16000)
    assert api.call("look", item=item, at=2)["audio"]["seconds"] == 2.0
    assert api.failure("look", item=item, at=9).field == "at"


def test_look_at_a_text_pages_through_it(home, api, store):
    text = "".join(f"line {i:05d}\n" for i in range(2500))  # 27,500 characters
    version = store.add_version("notes", "long.txt", T0, data=text.encode())
    first = api.call("look", item=version)
    assert first["kind"] == "text" and first["at"] is None and len(first["text"]) == 20_000 and first["more"] is True
    assert first["next_offset"] == 20_000
    second = api.call("look", item=version, offset=first["next_offset"])
    assert second["more"] is False and "next_offset" not in second and first["text"] + second["text"] == text


def test_look_at_a_source_and_a_time_finds_the_segment_the_version_or_the_records(home, api, store, media):
    segment = put_video(store, media, t0=T0)
    version = store.add_version("notes", "todo.md", T0 + 100, data=b"fix the kettle\n")
    store.add_version("notes", "todo.md", T0 + 200, data=b"all done\n")
    store.add_records("log", [(T0 + 300, json.dumps({"line": "disk full"})), (T0 + 310, json.dumps({"line": "disk ok"})),
                              (T0 + 500, json.dumps({"line": "much later"}))])
    shown = api.call("look", source="cam", t=T0 + 4.2)
    assert shown["item"] == segment and color(shown["image"]["path"])[0] > 200 and shown["at"] == pytest.approx(4.2, abs=0.001)
    old = api.call("look", source="notes", t=T0 + 150)
    assert old["item"] == version and "kettle" in old["text"], "the version current then"
    assert "all done" in api.call("look", source="notes", t=T0 + 250)["text"]
    around = api.call("look", source="log", t=T0 + 305)
    assert around["kind"] == "text" and around["item"] is None and "disk full" in around["text"] and "disk ok" in around["text"]
    assert "much later" not in around["text"], "only the records within 30 s"
    nothing = api.failure("look", source="cam", t=T0 + 5000)
    assert (nothing.status, nothing.code, nothing.field) == (404, "not_found", "source")
    assert f"#{segment}" in nothing.error["hint"] and "nearest" in nothing.error["hint"]
    assert api.failure("look", source="never-seen", t=T0).code == "not_found"
    assert api.failure("look", source="cam").field == "t" and api.failure("look").field == "source"
    assert api.failure("look", item=segment, source="cam").code == "bad_request"
    assert api.failure("look", source="log", run="r").field == "tp"
    assert api.failure("look", item="x").field == "item" and api.failure("look", item=segment, at=-1).field == "at"


def test_look_moments_runs_each_alone_in_order_and_a_bad_one_is_its_own_error(home, api, store, media):
    red = put_video(store, media, "cam", T0)
    blue = put_video(store, media, "other", T0 + 100)
    handles = [{"item": red, "at": 4.2}, {"item": blue, "at": 1}, {"item": 999}]
    got = api.call("look", moments=handles)["moments"]
    assert len(got) == 3
    for entry, handle in zip(got[:2], handles):
        alone = api.call("look", **handle)
        assert {k: v for k, v in entry.items() if k != "image"} == {k: v for k, v in alone.items() if k != "image"}
        assert Image.open(entry["image"]["inline"]).size == Image.open(alone["image"]["inline"]).size
    assert color(got[0]["image"]["path"])[0] > 200 and color(got[1]["image"]["path"])[2] > 200
    assert got[2] == {"error": {"code": "not_found", "message": "no item 999", "field": "item"}}
    one = api.call("look", moments=[{"source": "cam", "t": T0 + 4.2}])["moments"]
    assert len(one) == 1 and one[0]["item"] == red
    assert api.call("look", moments=[{"item": 1, "source": "cam"}])["moments"][0]["error"]["code"] == "bad_request"


def test_look_moments_share_one_page_of_text_between_them_and_a_single_look_has_it_whole(home, api, store):
    from mms.reader import LOOK_PAGE

    text = "".join(f"line {i:05d}\n" for i in range(2500))  # 27,500 characters
    versions = [store.add_version("notes", f"long{i}.txt", T0 + i, data=text.encode()) for i in range(3)]
    store.add_records("log", [(T0 + 300 + i * 0.1, json.dumps({"line": f"record {i:05d} " + "x" * 90})) for i in range(600)])
    for count, moments in ((3, [{"item": version} for version in versions]),
                           (2, [{"item": versions[0]}, {"source": "log", "t": T0 + 330}])):
        size = LOOK_PAGE // count
        for entry in api.call("look", moments=moments)["moments"]:
            assert len(entry["text"]) == size and entry["more"] is True and entry["next_offset"] == size, count
    second = api.call("look", item=versions[0], offset=LOOK_PAGE // 3)  # reading on is a single look
    assert second["text"] == text[LOOK_PAGE // 3:LOOK_PAGE // 3 + LOOK_PAGE]
    whole = api.call("look", moments=[{"item": versions[0]}, {"item": versions[1], "offset": 26_000}])["moments"]
    assert whole[1]["more"] is False and "next_offset" not in whole[1] and whole[1]["text"] == text[26_000:]
    assert len(api.call("look", item=versions[0])["text"]) == LOOK_PAGE
    assert len(api.call("look", source="log", t=T0 + 330)["text"]) == LOOK_PAGE


def test_a_malformed_look_moments_is_bad_request_naming_moments(home, api, store, media):
    item = put_video(store, media)
    for moments in ("x", [], [{"item": item}] * 9, [{"item": item}, "x"], [{"item": item}, None]):
        failure = api.failure("look", moments=moments)
        assert (failure.status, failure.code, failure.field) == (400, "bad_request", "moments"), moments
    assert api.failure("look", moments=[{"item": item}], item=item).field == "moments"
    assert api.failure("look", moments=[{"item": item}], at=1).field == "moments"
    assert len(api.call("look", moments=[{"item": item}] * 8)["moments"]) == 8


def test_look_shows_what_its_handle_names_whatever_the_project(home, api, store, media):
    """Scope is a relevance filter for `find` and `ask`; a handle names exactly one moment, so `look` has no scope."""
    caller = {"project": PROJECT, "project_name": "mine", "project_root": "/x", "session": "s-0123456789ab", "client": "t"}
    item = put_video(store, media, source="theirs", project="git:bbbbbbbbbbbbbbbb")
    assert api.call("look", source="theirs", t=T0 + 4, caller=caller)["item"] == item
    assert api.call("look", item=item, at=1, caller=caller)["item"] == item
    assert api.failure("look", source="theirs", t=T0 + 4, scope="all").field == "scope", "and takes no scope"


# a sheet: `look` over a range

def put_sound(store: Store, media: dict, source: str = "mic") -> int:
    target = store.streams / source / "launch" / "t.mka"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(media["audio"], target)
    return store.add_item(source, "audio", T0, T0 + 4, store.relative(target), target.stat().st_size)


def png(rgb: tuple[int, int, int]) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (64, 36), rgb).save(out, format="PNG")
    return out.getvalue()


def sheet_pixels(sheet: dict) -> Image.Image:
    with Image.open(sheet["image"]["path"]) as picture:
        return picture.convert("RGB").copy()


def centres(picture: Image.Image, cols: int, rows: int) -> list[tuple[int, int, int]]:
    """The pixel at the middle of each cell of a grid, row by row."""
    w, h = picture.size
    return [picture.getpixel((int((c + 0.5) * w / cols), int((r + 0.5) * h / rows))) for r in range(rows) for c in range(cols)]


def near(found, wanted, tolerance: int = 40) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(found, wanted))


def test_a_sheet_of_a_changing_clip_is_nine_numbered_tiles_row_wise(home, api, store, media):
    item = put_video(store, media, name="nine")
    store.label(T0 + 1, T0 + 5, "the colours", "agent", source="cam")
    sheet = api.call("look", source="cam", since=T0 + 0.1, until=T0 + 8.1)
    assert sheet.keys() == {"kind", "source", "since", "until", "labels", "image", "tiles"}
    assert sheet["kind"] == "sheet" and sheet["source"] == "cam"
    assert parse_when(sheet["since"]) == pytest.approx(T0 + 0.1, abs=0.001) and parse_when(sheet["until"]) == pytest.approx(T0 + 8.1, abs=0.001)
    assert [(label["by"], label["text"]) for label in sheet["labels"]] == [("agent", "the colours")]
    tiles = sheet["tiles"]
    assert [tile["n"] for tile in tiles] == list(range(1, 10)) and all(tile.keys() == {"n", "t", "look"} for tile in tiles)
    assert [parse_when(tile["t"]) for tile in tiles] == pytest.approx([T0 + 0.1 + k for k in range(9)], abs=0.001), "9 times, ends included"
    assert [tile["look"] for tile in tiles] == [{"item": item, "at": pytest.approx(0.1 + k, abs=0.001)} for k in range(9)]
    shown = sheet["image"]
    assert (shown["width"], shown["height"]) == (1568, 882) and Path(shown["path"]).parent == home / "looks"
    assert Image.open(shown["path"]).format == "PNG" and Image.open(shown["inline"]).format == "JPEG"
    assert Image.open(shown["path"]).size == Image.open(shown["inline"]).size == (1568, 882), "the sheet is its own full size"
    picture = sheet_pixels(sheet)
    assert all(near(found, wanted) for found, wanted in zip(centres(picture, 3, 3), PALETTE)), "row-wise, in time order"
    for k in range(9):  # a white digit on a black box in the bottom left corner of each picture
        x0, y1 = round((k % 3) * 1568 / 3), round((k // 3 + 1) * 294)
        box = [picture.getpixel((x, y)) for x in range(x0 + 1, x0 + 30) for y in range(y1 - 40, y1 - 3)]
        assert sum(max(pixel) < 40 for pixel in box) > 20 and sum(min(pixel) > 215 for pixel in box) > 10, k
        assert picture.getpixel((x0 + 1, y1 - 2)) == (0, 0, 0), k
    again = api.call("look", item=item, at=0.1 + 3)
    assert color(again["image"]["path"])[0] > 200 and near(centres(picture, 3, 3)[3], PALETTE[3]), "a tile's handle is its frame"


def test_a_still_range_is_one_tile_that_spans_it(home, api, store, media):
    item = put_video(store, media, name="still")
    sheet = api.call("look", source="cam", since=T0, until=T0 + 8)
    (tile,) = sheet["tiles"]
    assert tile.keys() == {"n", "t0", "t1", "look"} and tile["n"] == 1
    assert (parse_when(tile["t0"]), parse_when(tile["t1"])) == (pytest.approx(T0, abs=0.001), pytest.approx(T0 + 8, abs=0.001))
    assert tile["look"] == {"item": item, "at": 0.0}
    assert (sheet["image"]["width"], sheet["image"]["height"]) == (1568, 882), "one tile is the whole sheet"
    assert near(centres(sheet_pixels(sheet), 1, 1)[0], (128, 128, 128))


def test_a_range_with_a_gap_has_fewer_tiles(home, api, store, media):
    first = put_video(store, media, "cam", T0, name="nine")
    last = put_video(store, media, "cam", T0 + 100, name="still")
    sheet = api.call("look", source="cam", since=T0, until=T0 + 108)  # 9 times 13.5 s apart: the first and the last hold something
    assert [tile["n"] for tile in sheet["tiles"]] == [1, 2] and [tile["look"]["item"] for tile in sheet["tiles"]] == [first, last]
    assert [parse_when(tile["t"]) for tile in sheet["tiles"]] == pytest.approx([T0, T0 + 108], abs=0.001)
    assert (sheet["image"]["width"], sheet["image"]["height"]) == (1568, 441), "two tiles side by side"
    red, grey = centres(sheet_pixels(sheet), 2, 1)
    assert near(red, PALETTE[0]) and near(grey, (128, 128, 128))


def test_pictures_that_look_the_same_either_side_of_a_gap_are_two_tiles(home, api, store, media):
    first = put_video(store, media, "cam", T0, name="still")
    last = put_video(store, media, "cam", T0 + 100, name="still")
    sheet = api.call("look", source="cam", since=T0, until=T0 + 108)  # the times between hold nothing: the camera was off
    assert [(tile["n"], tile["look"]["item"]) for tile in sheet["tiles"]] == [(1, first), (2, last)]
    assert all("t" in tile for tile in sheet["tiles"]), "each one time: the unrecorded stretch is not a tile's"


def test_a_sheet_leaves_out_what_it_cannot_read_like_a_gap_and_fails_only_when_nothing_can_be_read(home, api, store, media):
    first = put_video(store, media, "cam", T0, name="still")
    store.add_item("cam", "video", T0 + 50, T0 + 59, "streams/cam/gone.mkv", 10)  # its file is not there
    last = put_video(store, media, "cam", T0 + 100, name="still")
    sheet = api.call("look", source="cam", since=T0, until=T0 + 108)  # the slots 0, 4 and 8: the middle one is lost
    assert [(tile["n"], tile["look"]["item"]) for tile in sheet["tiles"]] == [(1, first), (2, last)], "and not merged across it"
    store.add_item("lost", "video", T0, T0 + 9, "streams/lost/0.mkv", 10)
    nothing = api.failure("look", source="lost", since=T0, until=T0 + 8)
    assert (nothing.status, nothing.code, nothing.field) == (404, "not_found", "source") and "gone" in nothing.error["message"]
    junk = store.streams / "junk" / "launch" / "0.mkv"
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"this is not a video at all" * 20)
    store.add_item("junk", "video", T0, T0 + 9, store.relative(junk), junk.stat().st_size)
    broken = api.failure("look", source="junk", since=T0, until=T0 + 8)
    assert (broken.status, broken.code, broken.field) == (400, "bad_request", "source")


def test_a_sheet_of_pictures_has_a_tile_for_each_one_current_in_the_range_and_spans_its_times(home, api, store):
    red = store.add_version("pics", "a.png", T0, data=png((220, 20, 20)))
    blue = store.add_version("pics", "b.png", T0 + 100, data=png((20, 20, 220)))
    sheet = api.call("look", source="pics", since=T0 + 10, until=T0 + 190)  # 22.5 s apart: the first four see a.png, the rest b.png
    first, second = sheet["tiles"]
    assert (first["n"], second["n"]) == (1, 2) and first["look"] == {"item": red, "at": 0.0} and second["look"] == {"item": blue, "at": 0.0}
    assert [parse_when(first[k]) for k in ("t0", "t1")] == pytest.approx([T0 + 10, T0 + 77.5], abs=0.001)
    assert [parse_when(second[k]) for k in ("t0", "t1")] == pytest.approx([T0 + 100, T0 + 190], abs=0.001)
    left, right = centres(sheet_pixels(sheet), 2, 1)
    assert near(left, (220, 20, 20)) and near(right, (20, 20, 220))
    assert (sheet["image"]["width"], sheet["image"]["height"]) == (1568, 441)


def test_a_sheet_of_a_source_with_only_sound_or_records_is_not_found_with_a_hint(home, api, store, media):
    sound = put_sound(store, media)
    store.add_records("log", [(T0 + 1, json.dumps({"line": "disk full"}))])
    for source in ("mic", "log"):
        failure = api.failure("look", source=source, since=T0, until=T0 + 4)
        assert (failure.status, failure.code, failure.field) == (404, "not_found", "source"), source
        assert "no video or pictures" in failure.error["message"] and "nearest" in failure.error["hint"], source
    assert f"#{sound}" in api.failure("look", source="mic", since=T0, until=T0 + 4).error["hint"]
    assert api.failure("look", source="never-seen", since=T0, until=T0 + 4).code == "not_found"


def test_a_sheets_arguments_are_checked(home, api, store, media):
    item = put_video(store, media, name="nine")
    sheet = {"source": "cam", "since": T0, "until": T0 + 8}
    assert api.call("look", **sheet)["kind"] == "sheet"
    for extra in ({"item": item}, {"at": 1}, {"t": T0}, {"run": "r"}, {"tp": 1.5}, {"offset": 0}):
        failure = api.failure("look", **sheet, **extra)
        assert (failure.status, failure.code, failure.field) == (400, "bad_request", next(iter(extra))), extra
    assert api.failure("look", source="cam", since=T0).field == "until" and api.failure("look", source="cam", until=T0).field == "since"
    assert api.failure("look", since=T0, until=T0 + 8).field == "source"
    assert api.failure("look", **{**sheet, "until": T0}).field == "until", "until is after since"
    assert api.failure("look", **{**sheet, "until": T0 - 1}).field == "until"
    assert api.failure("look", **{**sheet, "since": "whenever"}).field == "since" and api.failure("look", **{**sheet, "until": []}).field == "until"
    assert api.failure("look", item=item, since=T0, until=T0 + 8).field == "item"


def test_a_sheet_can_be_one_of_the_moments(home, api, store, media):
    item = put_video(store, media, name="nine")
    put_sound(store, media)
    asked = {"source": "cam", "since": T0 + 0.1, "until": T0 + 8.1}
    got = api.call("look", moments=[asked, {"item": item, "at": 1}, {"source": "mic", "since": T0, "until": T0 + 4}])["moments"]
    alone = api.call("look", **asked)
    assert got[0]["kind"] == "sheet" and {k: v for k, v in got[0].items() if k != "image"} == {k: v for k, v in alone.items() if k != "image"}
    assert Image.open(got[0]["image"]["inline"]).size == Image.open(alone["image"]["inline"]).size == (1568, 882)
    assert got[1]["kind"] == "image" and got[2]["error"]["code"] == "not_found", "the sheet that found nothing is its own error"
    assert api.failure("look", moments=[asked], since=T0).field == "moments"


def test_a_portrait_video_fits_its_cells_and_a_picture_of_another_shape_is_centred_in_its_own(home, api, store, media):
    put_video(store, media, "phone", name="portrait")
    sheet = api.call("look", source="phone", since=T0 + 0.1, until=T0 + 8.1)
    assert len(sheet["tiles"]) == 9
    width, height = sheet["image"]["width"], sheet["image"]["height"]
    assert height == 1568 and abs(width - 882) <= 1 and Image.open(sheet["image"]["path"]).size == (width, height), "a tall sheet"
    picture = sheet_pixels(sheet)
    assert all(near(found, wanted) for found, wanted in zip(centres(picture, 3, 3), PALETTE))
    put_video(store, media, "mixed", T0, name="nine")
    put_video(store, media, "mixed", T0 + 100, name="portrait")
    mixed = sheet_pixels(api.call("look", source="mixed", since=T0 + 0.1, until=T0 + 108.1))  # a landscape cell, then a portrait picture
    assert mixed.size == (1568, 441)
    assert mixed.getpixel((784 + 10, 220)) == (32, 32, 32), "dark grey beside the portrait picture"
    assert near(mixed.getpixel((784 + 392, 220)), PALETTE[8]) and near(mixed.getpixel((392, 220)), PALETTE[0])
    assert mixed.getpixel((784 + 268 + 1, 441 - 2)) == (0, 0, 0), "the number is in the picture's corner, not the cell's"


# the one gate every write passes

def fake_free(monkeypatch, home: Path, floor_gb: float, spare_bytes: int) -> None:
    """A disk with `spare_bytes` to give above the floor, which what the home holds uses up."""
    import shutil
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")

    def disk_usage(path):
        held = sum(file.stat().st_size for file in home.rglob("*") if file.is_file())
        return usage(10**13, held, int(floor_gb * 1e9) + spare_bytes - held)

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)


def test_a_folder_remember_checks_the_floor_for_each_file_not_once(home, served, tmp_path, monkeypatch):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 20\nremember_max_mb = 5\n")
    folder = tmp_path / "big"
    folder.mkdir()
    for i in range(5):
        (folder / f"{i}.bin").write_bytes(b"\x00" + bytes([i + 1]) * 999_999)  # a NUL makes it binary: stored as it is
    fake_free(monkeypatch, home, 20, 2_500_000)  # room for two files above the floor
    api = served().api
    refused = api.failure("remember", path=str(folder))
    assert (refused.status, refused.code) == (507, "floor") and "floor" in refused.error["message"]
    assert refused.error["hint"], "the caller is told whom to ask"
    stored = Store(home).items_of("agent")
    assert 1 <= len(stored) <= 2, "the files that fit were stored, and the rest were not: the floor held"
    assert sum(item.bytes for item in stored) <= 2_500_000
    again = api.failure("remember", path=str(folder))
    assert again.code == "floor", "asking again changes nothing until there is room"


def test_a_write_bigger_than_the_sources_whole_cap_is_too_big(home, served, tmp_path):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n[sources.small]\nkind = \"push\"\nkeep = { max_gb = 0.0000005 }\n")
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 1000)
    api = served().api
    error = api.failure("remember", path=str(big), source="small")
    assert (error.status, error.code) == (413, "too_big") and "cap" in error.error["message"]
    assert api.call("remember", note="short", source="small")["stored"], "a note that fits is stored"
    assert Store(home).items_of("small") == []
