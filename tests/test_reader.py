"""The reader's video decoding (DESIGN.md §16): a video is decoded with frame threads, and the frames the reader hands over, their
pixels and their times, are those of PyAV's default decoder, however a segment is read: forwards at a fine and a coarse `every_s`,
backwards, from a seek, a screen recording of variable frame rate, and the single frame `look` asks for. Frames that look the same
are one window (`still_s`). CPU only, on clips made with the bundled ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import av
import pytest
from av.codec.context import ThreadType
from PIL import Image

from mms import decider_d1, reader

LENGTH_S = 13  # a camera clip: 10 frames a second, a key frame every 4 s, B-frames
SCREEN_STATES = [0.0, 2.3, 4.1, 6.2, 9.7, 11.0]  # a screen recording: a frame where the screen changed, and no other


def plain_stream(container):
    """The video stream as PyAV opens it, which is how the reader decoded before it set frame threads."""
    return container.streams.video[0]


def make(ffmpeg: str, path: Path, *args: str) -> Path:
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *args, str(path)], check=True)
    return path


def reordered(path: Path) -> bool:
    """Whether the file's frames are shown in another order than they are coded in (it has B-frames)."""
    with av.open(str(path)) as container:
        pts = [p.pts for p in container.demux(container.streams.video[0]) if p.pts is not None]
    return pts != sorted(pts)


@pytest.fixture(scope="module")
def camera(tmp_path_factory, ffmpeg) -> Path:
    path = make(ffmpeg, tmp_path_factory.mktemp("reader") / "camera.mp4", "-f", "lavfi", "-i",
                f"testsrc2=s=320x180:r=10:d={LENGTH_S}", "-c:v", "libx264", "-preset", "veryfast", "-bf", "3", "-g", "40",
                "-keyint_min", "40", "-sc_threshold", "0", "-pix_fmt", "yuv420p")
    assert reordered(path) and len(reader._times(path)) == LENGTH_S * 10
    return path


@pytest.fixture(scope="module")
def screen(tmp_path_factory, ffmpeg) -> Path:
    """A box that changes colour and place at each state's time, mpdecimate keeping one frame per state, as capture writes."""
    boxes = ",".join(f"drawbox=x={10 + 20 * i}:y={8 * i + 4}:w=40:h=30:color={colour}:t=fill:"
                     f"enable='gte(t,{start - 0.05})*lt(t,{end - 0.05})'"
                     for i, (colour, start, end) in enumerate(zip(("red", "blue", "green", "yellow", "cyan"), SCREEN_STATES[1:],
                                                                  SCREEN_STATES[2:] + [LENGTH_S])))
    path = make(ffmpeg, tmp_path_factory.mktemp("reader") / "screen.mkv", "-f", "lavfi", "-i",
                f"color=c=gray:s=160x90:r=10:d={LENGTH_S},{boxes}", "-vf", "mpdecimate", "-fps_mode", "vfr", "-c:v", "libx264",
                "-crf", "18", "-g", "30", "-pix_fmt", "yuv420p")
    assert [round(sec, 2) for _, sec in reader._times(path)] == SCREEN_STATES
    return path


def read(path: Path, monkeypatch, threads: bool, every_s: float, newest_first: bool, max_px: int = 0, since=None, until=None,
         kind: str = "file", still_s: float = reader.PROFILE["still_s"]) -> list[tuple]:
    """The windows `reader._video` gives for a clip, as (at, t0, t1, size, pixels), with or without the reader's frame threads."""
    store = SimpleNamespace(path_of=lambda item: path)
    item = SimpleNamespace(kind=kind, t0=1000.0, t1=1000.0 + LENGTH_S, source="cam", id=1)
    profile = {**reader.PROFILE, "every_s": every_s, "max_px": max_px, "still_s": still_s}
    with monkeypatch.context() as patch:
        if not threads:
            patch.setattr(reader, "_video_stream", plain_stream)
        return [(w.at, w.t0, w.t1, w.media.size, w.media.tobytes())
                for w in reader._video(store, item, profile, since, until, newest_first)]


def both(path: Path, monkeypatch, **how) -> list[tuple]:
    """The windows with the reader's frame threads, which must be those of PyAV's default decoder, to the pixel and the time."""
    threaded, default = read(path, monkeypatch, True, **how), read(path, monkeypatch, False, **how)
    assert threaded == default
    assert len({pixels for *_, pixels in threaded}) == len(threaded), "two windows hold the same picture"
    return threaded


def test_the_stream_is_set_to_decode_with_frame_threads(camera, ffmpeg, tmp_path):
    with av.open(str(camera)) as container:
        stream = reader._video_stream(container)
        assert stream.thread_type == ThreadType.AUTO and stream.thread_count == reader.DECODE_THREADS
        assert 1 <= reader.DECODE_THREADS <= 8
        assert sum(1 for _ in container.decode(stream)) == LENGTH_S * 10, "a frame is lost at the end of the file"
    sound = make(ffmpeg, tmp_path / "sound.wav", "-f", "lavfi", "-i", "sine=d=1")
    with av.open(str(sound)) as container, pytest.raises(ValueError, match="no video stream"):
        reader._video_stream(container)


@pytest.mark.parametrize("newest_first", [False, True], ids=["forwards", "backwards"])
@pytest.mark.parametrize("every_s", [1.0, 3.0])
def test_a_camera_clip_has_the_same_windows_at_a_fine_and_a_coarse_step(camera, monkeypatch, every_s, newest_first):
    got = both(camera, monkeypatch, every_s=every_s, newest_first=newest_first, max_px=20_000)  # a 320x180 frame is scaled down
    expected = [float(s) for s in range(0, LENGTH_S, int(every_s))][::-1 if newest_first else 1]
    assert [at for at, *_ in got] == pytest.approx(expected, abs=1e-3)
    assert {size for _, _, _, size, _ in got} == {reader.fit(320, 180, 20_000)}


def test_every_frame_of_a_camera_clip_comes_once_in_order_and_backwards_is_forwards_reversed(camera, monkeypatch):
    forwards = both(camera, monkeypatch, every_s=0, newest_first=False)
    assert [at for at, *_ in forwards] == pytest.approx([i / 10 for i in range(LENGTH_S * 10)], abs=1e-3)
    assert both(camera, monkeypatch, every_s=0, newest_first=True) == forwards[::-1]


@pytest.mark.parametrize("newest_first", [False, True], ids=["forwards", "backwards"])
def test_a_range_inside_a_clip_is_read_from_a_seek(camera, monkeypatch, newest_first):
    """5 to 10 s of a clip whose key frames are at 0, 4, 8 and 12 s: each block seeks back to a key frame."""
    got = both(camera, monkeypatch, every_s=1.0, newest_first=newest_first, since=1005.0, until=1010.0, kind="video")
    expected = [5.0, 6.0, 7.0, 8.0, 9.0][::-1 if newest_first else 1]
    assert [at for at, *_ in got] == pytest.approx(expected, abs=1e-3)
    assert [t0 for _, t0, *_ in got] == pytest.approx([1000.0 + at for at in expected], abs=1e-3)


@pytest.mark.parametrize("newest_first", [False, True], ids=["forwards", "backwards"])
def test_a_screen_recording_gives_one_window_per_state_at_its_own_time(screen, monkeypatch, newest_first):
    for every_s in (1.0, 5.0):
        got = both(screen, monkeypatch, every_s=every_s, newest_first=newest_first)
        slots = {int(state // every_s): state for state in reversed(SCREEN_STATES)}  # the first state of each slot
        expected = sorted(slots.values(), reverse=newest_first)
        assert [at for at, *_ in got] == pytest.approx(expected, abs=2e-3), every_s


def test_a_single_frame_is_the_same_wherever_it_is_in_the_clip(camera, screen, monkeypatch):
    """What `look` asks for: one frame, the last of a file only coming out when the decoder is drained."""
    for path in (camera, screen):
        times = reader._times(path)
        for pick in ([times[0]], [times[len(times) // 2]], [times[-1]], times[-3:], times[::7]):
            threaded = [(sec, image.tobytes()) for sec, image in reader._decode(path, pick, 0)]
            with monkeypatch.context() as patch:
                patch.setattr(reader, "_video_stream", plain_stream)
                default = [(sec, image.tobytes()) for sec, image in reader._decode(path, pick, 0)]
            assert threaded == default and [sec for sec, _ in threaded] == pytest.approx([sec for _, sec in pick], abs=1e-9)


def test_a_decoder_given_up_early_is_closed_and_lets_go_of_the_file(camera, tmp_path):
    copy = tmp_path / "copy.mp4"
    copy.write_bytes(camera.read_bytes())
    frames = reader._decode(copy, reader._times(copy), 0)
    assert next(frames)[0] == 0.0
    frames.close()
    copy.unlink()  # Windows refuses to delete a file that is still open
    assert not copy.exists()


# frames that look the same are one window

def colours(*args: str, size: str = "640x360", seconds: int = 20) -> list[str]:
    """ffmpeg arguments for a clip of one frame a second: a lavfi `color` source with `args` as its filters."""
    return ["-f", "lavfi", "-i", f"color=c=0x404040:s={size}:r=1:d={seconds}" + "".join("," + arg for arg in args), "-c:v", "libx264",
            "-g", "10", "-pix_fmt", "yuv420p"]


@pytest.fixture(scope="module")
def clips(tmp_path_factory, ffmpeg) -> dict[str, Path]:
    """Twenty seconds each at 640x360 and a frame a second: nothing moves, an 8x8 square appears at 10 s (off the 16 px cells),
    noise on every frame (a pixel's standard deviation of about 3), a change of brightness at 10 s, a drift of about a level a second, and red where there was dark olive,
    which are the same grey (a luma of 79 and 75)."""
    folder = tmp_path_factory.mktemp("still")
    two = lambda name: ["-f", "lavfi", "-i", f"color=c={name}:s=160x90:r=1:d=10"]  # noqa: E731
    return {
        "still": make(ffmpeg, folder / "still.mp4", *colours()),
        "square": make(ffmpeg, folder / "square.mp4", *colours("drawbox=x=300:y=172:w=8:h=8:color=white:t=fill:enable='gte(t,10)'")),
        "noisy": make(ffmpeg, folder / "noisy.mp4", *colours("noise=alls=3:allf=t")),
        "bright": make(ffmpeg, folder / "bright.mp4", *colours("eq=brightness=0.1:enable='gte(t,10)'")),
        "drift": make(ffmpeg, folder / "drift.mp4", *colours("eq=brightness='t/250':eval=frame")),
        "olive": make(ffmpeg, folder / "olive.mp4", *two("0x4F4E35"), *two("0xC71C1D"), "-filter_complex", "concat=n=2:v=1:a=0",
                      "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p"),
    }


def runs(path: Path, monkeypatch, newest_first: bool, **how) -> list[tuple[float, float, float]]:
    """The windows of a clip of a segment at 1000 s, as (at, t0 - 1000, t1 - 1000), in the order `reader._video` gives them."""
    how = {"every_s": 1.0, "kind": "video", **how}
    return [(at, t0 - 1000, t1 - 1000) for at, t0, t1, *_ in read(path, monkeypatch, True, newest_first=newest_first, **how)]


@pytest.mark.parametrize("newest_first", [False, True], ids=["forwards", "backwards"])
def test_a_still_clip_is_a_window_for_each_still_s_that_spans_its_frames(clips, monkeypatch, newest_first):
    """Twenty seconds of one picture at a limit of 8 s: runs of 8, 8 and 4 frames, each at its first frame in scan order."""
    got = runs(clips["still"], monkeypatch, newest_first, still_s=8)
    assert got == ([(19.0, 12.0, 19.0), (11.0, 4.0, 11.0), (3.0, 0.0, 3.0)] if newest_first else
                   [(0.0, 0.0, 7.0), (8.0, 8.0, 15.0), (16.0, 16.0, 19.0)])
    assert runs(clips["still"], monkeypatch, newest_first) == ([(19.0, 0.0, 19.0)] if newest_first else [(0.0, 0.0, 19.0)]), "the 30 s default"


@pytest.mark.parametrize("newest_first", [False, True], ids=["forwards", "backwards"])
def test_still_s_0_is_a_window_for_each_frame_and_the_limit_counts_from_the_runs_first_frame(clips, monkeypatch, newest_first):
    every = runs(clips["still"], monkeypatch, newest_first, still_s=0)
    assert [t0 for _, t0, _ in every] == [float(i) for i in range(20)][::-1 if newest_first else 1]
    assert all(t0 == t1 for _, t0, t1 in every), "a window of one frame"
    assert len(runs(clips["still"], monkeypatch, newest_first, still_s=1)) == 20, "a frame is at least 1 s from the run's first"
    assert len(runs(clips["still"], monkeypatch, newest_first, still_s=1.5)) == 10


def test_a_run_never_crosses_items_and_a_versions_windows_stay_at_its_t0(clips, monkeypatch):
    assert runs(clips["still"], monkeypatch, True, since=1005.0, until=1012.0) == [(11.0, 5.0, 11.0)], "a range's frames only"
    version = read(clips["still"], monkeypatch, True, every_s=1.0, newest_first=False, still_s=8)
    assert [(at, t0, t1) for at, t0, t1, *_ in version] == [(0.0, 1000.0, 1000.0), (8.0, 1000.0, 1000.0), (16.0, 1000.0, 1000.0)]


@pytest.mark.parametrize("newest_first", [False, True], ids=["forwards", "backwards"])
def test_a_small_square_appearing_is_a_window_where_it_appears(clips, monkeypatch, newest_first):
    got = runs(clips["square"], monkeypatch, newest_first)
    assert got == ([(19.0, 10.0, 19.0), (9.0, 0.0, 9.0)] if newest_first else [(0.0, 0.0, 9.0), (10.0, 10.0, 19.0)])


def test_mild_noise_stays_one_run_and_a_change_of_light_or_of_colour_starts_a_new_one(clips, monkeypatch):
    assert runs(clips["noisy"], monkeypatch, False) == [(0.0, 0.0, 19.0)], "a cell's mean moves by well under a level"
    assert runs(clips["bright"], monkeypatch, False) == [(0.0, 0.0, 9.0), (10.0, 10.0, 19.0)]
    assert runs(clips["olive"], monkeypatch, False) == [(0.0, 0.0, 9.0), (10.0, 10.0, 19.0)], "the same grey, another colour"


def test_a_slow_drift_adds_up_because_a_frame_is_judged_against_the_first_of_its_run(clips, monkeypatch):
    got = runs(clips["drift"], monkeypatch, False)
    assert 2 <= len(got) <= 4 and got[0][1] == 0.0 and got[-1][2] == 19.0
    assert all(b[1] > a[2] and b[1] - a[2] == 1.0 for a, b in zip(got, got[1:])), "runs follow each other"


def test_the_picture_is_the_same_only_at_one_size_and_within_the_level_in_every_cell_and_channel():
    grey = Image.new("RGB", (640, 360), (100, 100, 100))
    cells = reader._cells
    assert reader._same(cells(grey), cells(Image.new("RGB", (640, 360), (100 + reader.STILL_LEVEL, 100, 94))))
    assert not reader._same(cells(grey), cells(Image.new("RGB", (640, 360), (100, 100 + reader.STILL_LEVEL + 1, 100))))
    assert not reader._same(cells(grey), cells(Image.new("RGB", (320, 180), (100, 100, 100)))), "another size"
    spot = grey.copy()
    spot.paste((255, 255, 255), (300, 172, 308, 180))  # an 8x8 square over four cells
    assert not reader._same(cells(grey), cells(spot))


def test_a_thing_of_contrast_100_is_a_change_from_about_9x9_px_wherever_it_falls():
    """Cells are 16 px; four meet at (320, 176). A patch of n x n px adds n*n*100/256 to the mean of the cell it fills, and the
    most of one cell it can fill is what counts: 4x4 px moves a cell by 6.25, which rounds to 6, and 6 is still the same."""
    grey = Image.new("RGB", (640, 360), (100, 100, 100))

    def changed(n: int, x: int, y: int) -> bool:
        patch = grey.copy()
        patch.paste((200, 200, 200), (x, y, x + n, y + n))
        return not reader._same(reader._cells(grey), reader._cells(patch))

    assert not changed(4, 318, 174) and not changed(8, 316, 172), "4x4 at a corner; 8x8 centred on one, 4x4 in each of four cells"
    assert not changed(4, 322, 178), "4x4 wholly inside a cell"
    assert changed(5, 322, 178), "5x5 wholly inside a cell"
    assert all(changed(9, 315 + dx, 171 + dy) for dx in (0, 1) for dy in (0, 1)), "9x9 at a corner, however it falls"


def test_a_run_held_when_the_next_block_cannot_be_decoded_is_given_before_the_error(clips, monkeypatch):
    real = reader._decode
    blocks = []

    def second_block_fails(path, wanted, max_px=0):
        blocks.append(len(wanted))
        if len(blocks) == 2:
            raise OSError("the file ends here")
        yield from real(path, wanted, max_px)

    store = SimpleNamespace(path_of=lambda item: clips["still"])
    item = SimpleNamespace(kind="video", t0=1000.0, t1=1020.0, source="cam", id=1)
    monkeypatch.setattr(reader, "_decode", second_block_fails)
    for still_s, expected in ((30, [(0.0, 1000.0, 1009.0)]), (0, [(float(i), 1000.0 + i, 1000.0 + i) for i in range(10)])):
        blocks.clear()
        got = []
        with pytest.raises(OSError, match="ends here"):
            for w in reader._video(store, item, {**reader.PROFILE, "every_s": 1.0, "still_s": still_s}, None, None, False):
                got.append((w.at, w.t0, w.t1))
        assert got == expected, "the run of the first block, or each of its windows, whole"


def counting_decoder(monkeypatch) -> list[int]:
    """`av.open` made to count the frames the decoders of `reader` hand out: the one number in the list."""
    real, counted = av.open, [0]

    class Counted:
        def __init__(self, container):
            self.container = container

        def __enter__(self):
            self.container.__enter__()
            return self

        def __exit__(self, *error):
            return self.container.__exit__(*error)

        def __getattr__(self, name):
            return getattr(self.container, name)

        def decode(self, *args, **kwargs):
            for frame in self.container.decode(*args, **kwargs):
                counted[0] += 1
                yield frame

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Counted(real(*args, **kwargs)))
    return counted


def test_a_sheet_over_a_long_item_decodes_only_the_frames_it_shows(ffmpeg, tmp_path, monkeypatch):
    """A phone's hour is one item: 600 frames here, a key frame a second. Nine pictures cost nine seeks, not the time between."""
    path = make(ffmpeg, tmp_path / "long.mkv", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=10:d=60", "-c:v", "libx264", "-g", "10",
                "-bf", "0", "-pix_fmt", "yuv420p")
    item = SimpleNamespace(kind="video", t0=1000.0, t1=1060.0, source="phone", id=1, meta={})
    picks = [reader.Pick(k, 1000.0 + 7.5 * k, item, 7.5 * k) for k in range(9)]
    counted = counting_decoder(monkeypatch)
    made = reader.sheet(SimpleNamespace(path_of=lambda _: path), picks, tmp_path)
    assert len(made["tiles"]) == 9
    assert 9 <= counted[0] <= 9 * 11, f"{counted[0]} frames decoded for nine pictures of a 600 frame video"


def test_a_profile_takes_still_s_from_0_up():
    assert reader.profile_of({"still_s": 0}, decider_d1.ABOUT)["still_s"] == 0
    assert reader.profile_of(None, decider_d1.ABOUT)["still_s"] == reader.PROFILE["still_s"] == 30
    with pytest.raises(ValueError, match="still_s"):
        reader.profile_of({"still_s": -1}, decider_d1.ABOUT)
