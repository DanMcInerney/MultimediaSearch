"""`look` with a region (DESIGN.md §11): the part of a picture or video frame an agent asks to see, as fractions of it from the top
left. One moment shows that part cut from the picture at its own pixels and its inline picture enlarged; a sheet makes each
tile that part of its frame, and tiles differ only by what moves inside it. The part is cut from the picture as a plain
`look` decodes it: whole, and a JPEG over 64 MP at the half size it is shown at. Each test plants what the result must show
(a square of a known colour in a known corner, a 1 or 2 px dot, motion on one side) and goes through the daemon's API."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import write_config
from PIL import Image
from test_remember_look import PALETTE, T0, centres, near, put_sound, put_video, sheet_pixels

from mms.config import ffmpeg_exe
from mms.store import Store

EDGES = ("top", "left", "bottom", "right")


def region(top: float, left: float, bottom: float, right: float) -> dict:
    return {"top": top, "left": left, "bottom": bottom, "right": right}


def ffmpeg(*args: str) -> None:
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], check=True)


def lavfi(graph: str, path: Path, *encode: str) -> Path:
    ffmpeg("-f", "lavfi", "-i", graph, "-c:v", "libx264", *(encode or ("-pix_fmt", "yuv420p")), str(path))
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("region")
    # gray 320x180 with a red 40x30 square whose top left is at (250, 120): x 0.78125 to 0.90625, y 0.6667 to 0.8333
    square = lavfi("color=c=gray:s=320x180:r=5:d=3,drawbox=x=250:y=120:w=40:h=30:color=red:t=fill", folder / "square.mkv",
                   "-crf", "12", "-pix_fmt", "yuv420p")
    # gray 640x360 where a 100x100 box at (20, 20), in the left half, shows a colour of the palette in each second and
    # nothing else changes
    boxes = ",".join(f"drawbox=x=20:y=20:w=100:h=100:color=0x{r:02X}{g:02X}{b:02X}:t=fill:enable='between(t,{i},{i + 1})'"
                     for i, (r, g, b) in enumerate(PALETTE))
    left_moves = lavfi(f"color=c=gray:s=640x360:r=5:d=9,{boxes}", folder / "left.mkv", "-g", "10", "-crf", "12", "-pix_fmt", "yuv420p")
    # a 4K frame, black with one white pixel at (1920, 1080), lossless so that the pixel is in the file
    dot = lavfi("color=c=black:s=3840x2160:r=1:d=2,drawbox=x=1920:y=1080:w=1:h=1:color=white:t=fill", folder / "dot.mkv",
                "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv444p")
    audio = folder / "tone.mka"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4,volume=4", "-c:a", "libopus", str(audio))
    # the square's picture for 6 s with a 440 Hz tone for its soundtrack
    talk = folder / "talk.mkv"
    ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=320x180:r=5:d=6,drawbox=x=250:y=120:w=40:h=30:color=red:t=fill",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6,volume=4",
           "-c:v", "libx264", "-g", "10", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "24k", str(talk))
    return {"square": square, "left": left_moves, "dot": dot, "audio": audio, "talk": talk}


@pytest.fixture
def api(home, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 1\n")
    return served().api


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def picture(size: tuple[int, int], colour=(0, 0, 0), *, boxes=(), format="PNG", exif: dict | None = None, **save) -> bytes:
    """A picture of one colour with filled boxes `(x0, y0, x1, y1, colour)` over it."""
    image = Image.new("RGB", size, colour)
    for x0, y0, x1, y1, shade in boxes:
        image.paste(shade, (x0, y0, x1, y1))
    if exif:
        tags = Image.Exif()
        tags.update(exif)
        save["exif"] = tags
    out = io.BytesIO()
    image.save(out, format=format, **save)
    return out.getvalue()


def shown(path: str) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=int)


def share(pixels: np.ndarray, rgb: tuple[int, int, int], tolerance: int = 60) -> float:
    """The fraction of the pixels within `tolerance` of a colour on every channel."""
    return float((np.abs(pixels - np.array(rgb)).max(axis=-1) <= tolerance).mean())


RED = (255, 0, 0)
GRAY = (128, 128, 128)


# one moment

def test_a_region_around_the_square_shows_it_enlarged_to_the_inline_size_and_its_png_is_the_crop(home, api, store, media):
    item = put_video(store, media, name="square", t0=T0)
    got = api.call("look", item=item, at=1, region=region(120 / 180, 250 / 320, 150 / 180, 290 / 320))
    assert got["kind"] == "image" and got["item"] == item and got["region"] == region(120 / 180, 250 / 320, 150 / 180, 290 / 320)
    assert (got["image"]["width"], got["image"]["height"]) == (40, 30), "the size of the crop"
    assert shown(got["image"]["path"]).shape == (30, 40, 3)
    assert shown(got["image"]["inline"]).shape == (1176, 1568, 3), "the crop enlarged so that its long side is 1568 px"
    assert share(shown(got["image"]["inline"]), RED) > 0.9 and share(shown(got["image"]["path"]), RED) > 0.9
    by_time = api.call("look", source="cam", t=T0 + 1, region=region(120 / 180, 250 / 320, 150 / 180, 290 / 320))
    assert (by_time["item"], by_time["image"]["width"], by_time["image"]["height"]) == (item, 40, 30)


def test_a_region_elsewhere_has_none_of_the_square_and_a_region_over_a_part_of_it_has_that_part(home, api, store, media):
    item = put_video(store, media, name="square")
    away = api.call("look", item=item, region=region(0, 0, 0.5, 0.5))
    assert (away["image"]["width"], away["image"]["height"]) == (160, 90)
    assert share(shown(away["image"]["inline"]), GRAY, 25) > 0.99 and share(shown(away["image"]["inline"]), RED) == 0
    half = api.call("look", item=item, region=region(0, 270 / 320, 1, 1))  # the right edge: the square's right half, 20 px
    pixels = shown(half["image"]["path"])
    assert pixels.shape == (180, 50, 3)
    assert share(pixels[120:150, :20], RED) > 0.9 and share(pixels[:, 20:], RED) == 0, "red in its 20 px, nothing right of it"


def test_a_region_is_enlarged_only_to_the_inline_size_and_without_one_nothing_changes(home, api, store, media):
    item = put_video(store, media, name="square")
    whole = api.call("look", item=item, region=region(0, 0, 1, 1))
    assert (whole["image"]["width"], whole["image"]["height"]) == (320, 180)
    assert Image.open(whole["image"]["inline"]).size == (1568, 882), "a region's inline picture is always 1568 px on its long side"
    plain = api.call("look", item=item)
    assert "region" not in plain and Image.open(plain["image"]["inline"]).size == (320, 180), "no region, no enlarging"
    assert plain.keys() == whole.keys() - {"region"}


def test_a_region_of_a_picture_larger_than_the_inline_size_is_shrunk_for_the_inline_picture(home, api, store):
    big = store.add_version("pics", "big.png", T0, data=picture((2000, 1000), boxes=[(0, 0, 100, 100, RED)]))
    got = api.call("look", item=big, region=region(0, 0, 1, 1))
    assert (got["image"]["width"], got["image"]["height"]) == (2000, 1000) and Image.open(got["image"]["path"]).size == (2000, 1000)
    assert Image.open(got["image"]["inline"]).size == (1568, 784)
    half = api.call("look", item=big, region=region(0, 0, 0.5, 0.5))  # 1000x500: its inline picture is enlarged to 1568
    assert (half["image"]["width"], half["image"]["height"]) == (1000, 500) and Image.open(half["image"]["inline"]).size == (1568, 784)
    assert share(shown(half["image"]["path"])[:100, :100], RED) == 1.0


def test_a_region_is_of_the_picture_as_shown_upright_and_not_as_stored(home, api, store):
    # stored 200x100 with a red 40x40 square at its top left, orientation 6: shown as 100x200 with the square at its top right
    data = picture((200, 100), (0, 0, 255), boxes=[(0, 0, 40, 40, RED)], exif={0x0112: 6})
    item = store.add_version("pics", "phone.png", T0, data=data)
    corner = api.call("look", item=item, region=region(0, 0.6, 0.2, 1))
    assert (corner["image"]["width"], corner["image"]["height"]) == (40, 40) and share(shown(corner["image"]["path"]), RED) == 1.0
    other = api.call("look", item=item, region=region(0, 0, 0.2, 0.4))
    assert share(shown(other["image"]["path"]), (0, 0, 255)) == 1.0


# full resolution

def test_the_crop_is_cut_from_the_full_picture_so_a_two_pixel_detail_survives(home, api, store):
    dot = picture((2000, 1000), boxes=[(1000, 500, 1002, 502, (255, 255, 255))])
    item = store.add_version("pics", "dot.png", T0, data=dot)
    got = api.call("look", item=item, region=region(0.49, 0.49, 0.51, 0.51))  # x 980 to 1020, y 490 to 510
    pixels = shown(got["image"]["path"])
    assert pixels.shape == (20, 40, 3)
    assert (pixels == 255).all(axis=-1).sum() == 4 and (pixels[10:12, 20:22] == 255).all(), "the dot is as bright and as big as stored"


def test_a_large_jpeg_is_cropped_from_its_full_size(home, api, store):
    data = picture((6000, 4000), boxes=[(3000, 2000, 3002, 2002, (255, 255, 255))], format="JPEG", quality=100, subsampling=0)
    item = store.add_version("pics", "wide.jpg", T0, data=data)
    got = api.call("look", item=item, region=region(0.49, 0.49, 0.51, 0.51))
    pixels = shown(got["image"]["path"])
    assert pixels.shape == (80, 120, 3), "x 2940 to 3060 and y 1960 to 2040 of the 6000x4000 picture"
    assert pixels.max() > 200


def test_a_jpeg_over_the_decode_limit_is_cropped_from_the_halved_copy_a_plain_look_shows(home, api, store):
    data = picture((9000, 7800), boxes=[(4500, 3900, 4502, 3902, (255, 255, 255))], format="JPEG", quality=100)  # 70 MP, over 64
    item = store.add_version("pics", "huge.jpg", T0, data=data)
    assert Image.open(api.call("look", item=item)["image"]["path"]).size == (4500, 3900), "looked at whole, a decoder halves it"
    # of the 4500x3900 it is shown as: x 1800 to 2700 and y 1560 to 2340
    got = api.call("look", item=item, region=region(0.4, 0.4, 0.6, 0.6))
    pixels = shown(got["image"]["path"])
    assert pixels.shape == (780, 900, 3), "the region's fractions of the halved picture, with no refusal"
    assert pixels[385:396, 445:456].max() > 150, "the dot, at (2250, 1950) of the halved picture, is a pixel of the crop"


def test_a_videos_sound_is_the_same_with_a_region_as_without_one(home, api, store, media):
    item = put_video(store, media, name="talk")
    square = region(120 / 180, 250 / 320, 150 / 180, 290 / 320)
    plain = api.call("look", item=item, at=0.5)
    cut = api.call("look", item=item, at=0.5, region=square)
    assert plain["audio"]["seconds"] == 5.0 and plain["audio"]["rms"] > 0.1, "5 s of the tone, from 0.5 s to the end"
    assert "audio" in cut and (cut["audio"]["seconds"], cut["audio"]["rms"]) == (plain["audio"]["seconds"], plain["audio"]["rms"])
    assert (plain["image"]["width"], plain["image"]["height"]) == (320, 180)
    assert (cut["image"]["width"], cut["image"]["height"]) == (40, 30) and share(shown(cut["image"]["path"]), RED) > 0.9


def test_a_video_frame_crop_is_in_the_pixels_of_the_frame_decoded_at_full_size(home, api, store, media):
    item = put_video(store, media, name="dot")
    got = api.call("look", item=item, at=0, region=region(0.49, 0.49, 0.51, 0.51))
    pixels = shown(got["image"]["path"])
    assert pixels.shape == (44, 76, 3), "x 1882 to 1958 and y 1058 to 1102 of 3840x2160"
    assert pixels.max() > 230 and np.unravel_index(pixels[..., 0].argmax(), pixels.shape[:2]) == (1080 - 1058, 1920 - 1882)


def test_a_sheet_decodes_the_frames_at_full_size_so_that_a_one_pixel_detail_reaches_its_tile(home, api, store, media):
    put_video(store, media, name="dot", t0=T0)
    sheet = api.call("look", source="cam", since=T0, until=T0 + 1.9, region=region(0.49, 0.49, 0.51, 0.51))
    assert len(sheet["tiles"]) == 1 and sheet["region"] == region(0.49, 0.49, 0.51, 0.51)
    assert (sheet["image"]["width"], sheet["image"]["height"]) == (1568, 908), "one tile of the crop (76x44) fills the sheet"
    assert sheet_pixels(sheet).getextrema()[0][1] > 230, "the pixel, enlarged: from a copy decoded at 2.4 MP it would be a dim blur"


# a sheet

def test_a_sheets_tiles_are_the_region_and_only_motion_inside_it_makes_tiles(home, api, store, media):
    put_video(store, media, name="left", t0=T0)
    ask = {"source": "cam", "since": T0 + 0.1, "until": T0 + 8.1}  # nine times, one in each second of the palette
    assert len(api.call("look", **ask)["tiles"]) == 9
    moving = api.call("look", **ask, region=region(0, 0, 1, 0.5))  # the left half, which holds the box
    still = api.call("look", **ask, region=region(0, 0.5, 1, 1))  # the right half, where nothing moves
    assert (len(moving["tiles"]), len(still["tiles"])) == (9, 1), "motion in the region makes tiles, motion outside it makes none"
    assert moving["region"] == region(0, 0, 1, 0.5) and still["region"] == region(0, 0.5, 1, 1)
    (only,) = still["tiles"]
    assert only["n"] == 1 and only["look"]["item"] == moving["tiles"][0]["look"]["item"] and only["t0"] < only["t1"]
    assert abs(still["image"]["width"] - 1568 * 8 / 9) <= 1 and still["image"]["height"] == 1568, "the right half, 320x360, whole"
    assert abs(moving["image"]["width"] - 1568 * 8 / 9) <= 1 and moving["image"]["height"] == 1568, "nine cells of the left half's shape"
    picture = sheet_pixels(moving)
    w, h = picture.size
    for k, wanted in enumerate(PALETTE):  # the box, at 0.2 of the way across and down its cell, shows the second's colour
        row, col = divmod(k, 3)
        assert near(picture.getpixel((int((col + 0.2) * w / 3), int((row + 0.2) * h / 3))), wanted), k


def test_a_sheet_of_pictures_that_differ_only_outside_the_region_is_one_tile(home, api, store):
    side = lambda colour: picture((640, 360), GRAY, boxes=[(320, 0, 640, 360, colour)])  # noqa: E731  the right half is the colour
    store.add_version("pics", "a.png", T0, data=side((220, 20, 20)))
    store.add_version("pics", "b.png", T0 + 100, data=side((20, 20, 220)))
    ask = {"source": "pics", "since": T0 + 10, "until": T0 + 190}
    assert len(api.call("look", **ask)["tiles"]) == 2
    assert len(api.call("look", **ask, region=region(0, 0, 1, 0.5))["tiles"]) == 1, "the left halves are the same grey"
    right = api.call("look", **ask, region=region(0, 0.5, 1, 1))
    assert len(right["tiles"]) == 2
    first, second = centres(sheet_pixels(right), 2, 1)
    assert near(first, (220, 20, 20)) and near(second, (20, 20, 220))


# what is refused

BAD_SHAPES = [
    pytest.param({"top": 0, "left": 0, "bottom": 1}, id="a missing edge"),
    pytest.param({"top": 0, "left": 0, "bottom": 1, "right": 1, "middle": 0.5}, id="an edge that is none"),
    pytest.param({}, id="empty"),
    pytest.param({"top": -0.1, "left": 0, "bottom": 1, "right": 1}, id="below 0"),
    pytest.param({"top": 0, "left": 0, "bottom": 1, "right": 1.1}, id="above 1"),
    pytest.param({"top": "0.1", "left": 0, "bottom": 1, "right": 1}, id="an edge as text"),
    pytest.param({"top": True, "left": 0, "bottom": 1, "right": 1}, id="a boolean"),
    pytest.param({"top": 0, "left": 0, "bottom": 1, "right": [1]}, id="a list"),
    pytest.param({"top": 0.5, "left": 0, "bottom": 0.5, "right": 1}, id="top is bottom"),
    pytest.param({"top": 0.6, "left": 0, "bottom": 0.4, "right": 1}, id="top below bottom"),
    pytest.param({"top": 0, "left": 0.5, "bottom": 1, "right": 0.5}, id="left is right"),
    pytest.param({"top": 0, "left": 0.7, "bottom": 1, "right": 0.2}, id="left after right"),
    pytest.param([0, 0, 1, 1], id="a list of the edges"),
    pytest.param("0,0,1,1", id="text instead of an object"),
    pytest.param(0.5, id="a number"),
    pytest.param(True, id="true"),
]


@pytest.mark.parametrize("bad", BAD_SHAPES)
def test_a_region_of_another_shape_is_refused_naming_region_and_saying_the_shape(home, api, store, media, bad):
    item = put_video(store, media, name="square")
    for handle in ({"item": item}, {"source": "cam", "t": T0 + 1}, {"source": "cam", "since": T0, "until": T0 + 2}):
        failure = api.failure("look", **handle, region=bad)
        assert (failure.status, failure.code, failure.field) == (400, "bad_request", "region"), handle
        assert "top, left, bottom and right" in failure.error["message"] and "0 to 1" in failure.error["message"], handle


def test_a_region_of_exactly_the_whole_or_a_sliver_is_taken(home, api, store, media):
    item = put_video(store, media, name="square")
    for edges in ((0, 0, 1, 1), (0, 0, 0.001, 0.001), (0.999, 0.999, 1, 1), (0, 0.4999, 1, 0.5)):
        got = api.call("look", item=item, region=region(*edges))
        assert got["image"]["width"] >= 1 and got["image"]["height"] >= 1, edges
        assert max(Image.open(got["image"]["inline"]).size) == 1568, edges
    assert api.call("look", item=item, region=region(0, 0.4999, 1, 0.5))["image"]["width"] == 1, "a sliver is at least a pixel across"


def test_region_is_for_pictures_and_video_and_a_sound_text_file_or_records_refuse_it(home, api, store, media):
    sound = put_sound(store, media)
    note = store.add_version("notes", "todo.md", T0, data=b"fix the kettle\n")
    pdf = store.add_version("docs", "paper.pdf", T0, data=b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n")
    store.add_records("log", [(T0 + 1, '{"line": "disk full"}')])
    assert api.call("look", item=pdf)["kind"] == "file" and api.call("look", item=note)["kind"] == "text"
    for handle in ({"item": sound}, {"item": note}, {"item": pdf}, {"source": "mic", "t": T0 + 1}, {"source": "notes", "t": T0 + 1},
                   {"source": "log", "t": T0 + 1}):
        failure = api.failure("look", **handle, region=region(0, 0, 1, 1))
        assert (failure.status, failure.code, failure.field) == (400, "bad_request", "region"), handle
        assert "pictures and video" in failure.error["message"], handle
    nothing = api.failure("look", source="log", t=T0 + 5000, region=region(0, 0, 1, 1))
    assert (nothing.code, nothing.field) == ("not_found", "source"), "a time with nothing is not found, whatever is asked of it"


def test_in_moments_a_bad_region_or_one_on_a_sound_fails_its_own_entry_and_the_others_run(home, api, store, media):
    video = put_video(store, media, name="square")
    sound = put_sound(store, media)
    square = region(120 / 180, 250 / 320, 150 / 180, 290 / 320)
    got = api.call("look", moments=[{"item": video, "at": 1, "region": square}, {"item": video, "region": {"top": 0}},
                                    {"item": sound, "region": region(0, 0, 1, 1)}, {"item": video, "at": 2}])["moments"]
    assert len(got) == 4
    assert got[0]["region"] == square and (got[0]["image"]["width"], got[0]["image"]["height"]) == (40, 30)
    assert share(shown(got[0]["image"]["inline"]), RED) > 0.9
    assert (got[1]["error"]["code"], got[1]["error"]["field"]) == ("bad_request", "region")
    assert (got[2]["error"]["code"], got[2]["error"]["field"]) == ("bad_request", "region")
    assert "region" not in got[3] and (got[3]["image"]["width"], got[3]["image"]["height"]) == (320, 180), "no region: the whole frame"
    alone = api.call("look", item=video, at=1, region=square)
    assert {k: v for k, v in got[0].items() if k != "image"} == {k: v for k, v in alone.items() if k != "image"}


def test_a_sheet_can_be_one_of_the_moments_with_its_own_region(home, api, store, media):
    put_video(store, media, name="left", t0=T0)
    ask = {"source": "cam", "since": T0 + 0.1, "until": T0 + 8.1}
    got = api.call("look", moments=[{**ask, "region": region(0, 0.5, 1, 1)}, {**ask, "region": region(0, 0, 1, 0.5)}, ask])["moments"]
    assert [len(entry["tiles"]) for entry in got] == [1, 9, 9]
    assert got[0]["region"] == region(0, 0.5, 1, 1) and "region" not in got[2]
