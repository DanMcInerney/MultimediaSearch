"""ffmpeg sources (DESIGN.md §6, TESTBED.md item 1): `Capture` driven directly, with ffmpeg's own synthetic inputs in real
time and segments of 1 s, and what the named kinds expand to."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import av
import pytest
from conftest import wait_for, write_config

from mms import children
from mms import config as configs
from mms.capture import Capture, command, index_orphans, opens, parse_dshow, read_list
from mms.media import has_sound
from mms.store import Store

SCREEN = '''
[sources.screen]
kind = "ffmpeg"
preset = "screen"
input = ["-re", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=4"]
segment_s = 1
'''
TONE = '''
[sources.tone]
kind = "ffmpeg"
preset = "audio"
input = ["-re", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
segment_s = 1
'''


def setup(home, toml):
    write_config(home, toml)
    config = configs.load(home)
    store = Store(home)
    return store, config, Capture(store, config)


@pytest.fixture
def capture(home):
    """`make(toml)` gives a (store, capture); every capture is stopped at the end, so no ffmpeg outlives its test."""
    made = []

    def make(toml):
        store, _, cap = setup(home, toml)
        made.append(cap)
        return store, cap

    yield make
    for cap in made:
        cap.stop()


def run_until(cap, check, timeout=15, what="the condition"):
    """Step the capture, as the daemon's loop does, until `check()` is truthy."""
    def poll():
        cap.step(time.time())
        return check()
    return wait_for(poll, timeout=timeout, every=0.1, what=what)


def listed(folder) -> int:
    return len(read_list(folder))


def test_segments_are_indexed_in_utc(home, capture):
    store, cap = capture(SCREEN)
    before = time.time()
    run_until(cap, lambda: len(store.select(sources=["screen"])) >= 3, what="three segments")
    after = time.time()
    items = store.select(sources=["screen"])
    launch = store.launches("screen")[0]
    assert cap.status()["screen"]["state"] == "running"
    assert cap.status()["screen"]["segments"] >= 3
    assert before - 1 <= launch["started"] <= after
    assert launch["folder"].startswith("streams/screen/") and launch["lines"] >= 3
    assert launch["folder"].split("/")[-1] == configs.stamp(launch["started"])
    for item in items:
        file = store.path_of(item)
        assert item.kind == "video" and item.path.endswith(".mkv") and not item.in_place
        assert item.bytes == file.stat().st_size > 0
        # a segment's file was last written as it closed: its end time is that wall-clock moment
        assert abs(item.t1 - file.stat().st_mtime) < 2
        assert 0.5 < item.t1 - item.t0 < 2
    for a, b in zip(items, items[1:]):
        assert b.t0 == pytest.approx(a.t1, abs=0.01)
    assert items[0].t0 == pytest.approx(launch["started"], abs=0.01)


def test_stop_closes_the_open_segment(home, capture):
    store, cap = capture(SCREEN)
    run_until(cap, lambda: len(store.select(sources=["screen"])) >= 2, what="two segments")
    time.sleep(0.5)  # into the next segment
    folder = store.home / cap.runs["screen"].launch["folder"]
    indexed = len(store.select(sources=["screen"]))
    cap.stop()
    items = store.select(sources=["screen"])
    assert len(items) > indexed
    assert len(list(folder.glob("*.mkv"))) == len(items) == listed(folder)
    last = items[-1]
    with av.open(str(store.path_of(last))) as container:
        assert sum(1 for _ in container.demux(video=0)) > 0
    assert last.t1 - last.t0 < 1.5
    assert store.launches("screen")[0]["ended"] is not None
    assert cap.status()["screen"]["state"] == "stopped"
    assert not cap.runs


def test_killed_capture_is_reconciled(home):
    store, _, dead = setup(home, SCREEN)
    dead.step(time.time())
    process = dead.runs["screen"].process
    folder = store.home / dead.runs["screen"].launch["folder"]
    try:
        # nobody steps, so nothing is indexed while ffmpeg writes its list
        wait_for(lambda: listed(folder) >= 2, what="two list lines")
        time.sleep(0.3)
        process.kill()
        process.wait()
    finally:
        process.kill()
    assert store.select(sources=["screen"]) == []
    named = {name for name, _, _ in read_list(folder)}
    # what a crash can also leave: a segment cut short, and one with nothing in it
    first = sorted(folder.glob("*.mkv"))[0]
    shutil.copy(first, folder / "000090.mkv")
    (folder / "000091.mkv").write_bytes(b"not a media file")

    store2 = Store(home)
    Capture(store2, configs.load(home)).reconcile()
    items = store2.select(sources=["screen"])
    launch = store2.launches("screen")[0]
    lines = read_list(folder)
    assert launch["ended"] is not None and launch["lines"] == len(lines) >= 2
    assert {store2.path_of(item).name for item in items} >= named | {"000090.mkv"}
    assert not (folder / "000091.mkv").exists()
    assert {file.name for file in folder.glob("*.mkv")} == {store2.path_of(item).name for item in items}
    by_name = {store2.path_of(item).name: item for item in items}
    last_end = launch["started"] + lines[-1][2] - launch["first_start"]
    assert by_name["000090.mkv"].t0 == pytest.approx(last_end, abs=0.01)
    assert by_name["000090.mkv"].t1 - by_name["000090.mkv"].t0 == pytest.approx(
        by_name["000000.mkv"].t1 - by_name["000000.mkv"].t0, abs=0.2)
    assert by_name["000000.mkv"].t0 == pytest.approx(launch["started"], abs=0.01)

    Capture(store2, configs.load(home)).reconcile()  # a second pass finds nothing to do
    assert len(store2.select(sources=["screen"])) == len(items)


def alive(pid: int) -> bool:
    """Whether a process with this pid is running."""
    if os.name == "nt":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in listing.split()
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def pids_of(marker: str) -> list[int]:
    """The ffmpeg processes whose command line holds `marker`."""
    if os.name == "nt":
        listing = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  f"(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{marker}*' "
                                  f"-and $_.Name -like '*ffmpeg*' }}).ProcessId"], capture_output=True, text=True).stdout
        return [int(line) for line in listing.split()]
    return [int(pid) for pid in subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True).stdout.split()]


CONSOLE = ("import ctypes; k = ctypes.windll.kernel32; ids = (ctypes.c_uint * 64)(); "
           "n = k.GetConsoleProcessList(ids, 64); print(k.GetConsoleWindow() or 0, *ids[:n])")


@pytest.mark.skipif(os.name != "nt", reason="a console window is Windows'")
@pytest.mark.parametrize("start", ["popen", "run"])
def test_a_child_of_the_daemon_opens_no_console_window(start):
    """The detached daemon runs under pythonw, with no console, so a child sharing it would open a window of its own.
    Each child gets a console of its own with no window, shared with no parent."""
    probe = [sys.executable, "-I", "-c", CONSOLE]
    if start == "popen":
        out = children.popen(probe, stdout=subprocess.PIPE, text=True).communicate(timeout=60)[0]
    else:
        out = children.run(probe, capture_output=True, text=True, timeout=60).stdout
    window, *sharing = [int(word) for word in out.split()]
    assert window == 0, "the child's console has no window"
    assert sharing and os.getpid() not in sharing, f"the child shares its parent's console: {sharing}"


def test_killing_the_daemon_takes_its_children_with_it(home, daemon):
    """The daemon dies hard: its ffmpeg and command die with it, nothing is written into the launch folder after, and
    the next start indexes every segment of it."""
    held = json.dumps([sys.executable, "-c", "import os, time\nprint(os.getpid(), flush=True)\ntime.sleep(120)"])
    bounded = SCREEN.replace('"-re", "-f"', '"-re", "-t", "30", "-f"')  # a stray ffmpeg ends by itself
    write_config(home, f'[memory]\nloop_s = 0.1\n{bounded}\n[sources.held]\nkind = "command"\ncommand = {held}\n')
    store = Store(home)
    first = daemon()
    wait_for(lambda: len(store.select(sources=["screen"])) >= 3, what="three segments", detail=first.text)
    ffmpeg_pids = wait_for(lambda: pids_of("testsrc2=size=320x180"), what="ffmpeg")
    exec_pid = int(json.loads(wait_for(lambda: store.records(sources=["held"]), what="the command")[0].data)["line"])
    folder = store.home / store.launches("screen")[0]["folder"]
    os.kill(first.api.info()["pid"], signal.SIGTERM if os.name == "nt" else signal.SIGKILL)  # not a clean stop
    first.process.wait(10)
    wait_for(lambda: not any(alive(pid) for pid in [*ffmpeg_pids, exec_pid]), timeout=10, what="the children to die")
    count = len(list(folder.glob("*.mkv")))
    time.sleep(2.5)
    assert len(list(folder.glob("*.mkv"))) == count, "ffmpeg wrote on after the daemon died"

    again = daemon()
    wait_for(lambda: store.launches("screen")[0]["ended"] is not None, what="the next start to settle the launch")
    files = {file.name for file in folder.glob("*.mkv")}
    indexed = {store.path_of(item).name for item in store.select(sources=["screen"]) if store.path_of(item).parent == folder}
    assert files == indexed and len(files) >= 3, "every segment is indexed, the one cut short too"
    assert {name for name, _, _ in read_list(folder)} <= indexed
    again.stop()


def test_audio_source(home, capture):
    store, cap = capture(TONE)
    run_until(cap, lambda: len(store.select(sources=["tone"])) >= 2, what="two audio segments")
    cap.stop()
    items = store.select(sources=["tone"])
    for item in items:
        assert item.kind == "audio" and item.path.endswith(".mka")
    with av.open(str(store.path_of(items[0]))) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.name == "opus"
        assert sum(float(p.duration * p.stream.time_base) for p in container.demux(stream) if p.pts is not None
                   and p.duration) == pytest.approx(items[0].t1 - items[0].t0, abs=0.2)
    assert items[0].t0 == pytest.approx(store.launches("tone")[0]["started"], abs=0.01)


AV = '''
[sources.av]
kind = "ffmpeg"
input = ["-re", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=4", "-re", "-f", "lavfi", "-i", "sine=frequency=440"]
encode = ["-c:v", "libx264", "-g", "4", "-c:a", "libopus", "-b:a", "24k"]
segment_s = 1
'''


def test_a_video_segment_records_whether_it_has_sound(home, capture):
    """A reader that hears only (an audio decider) takes a video's soundtrack by `meta.sound`, and `estimate` counts it
    by that: a segment with an audio stream records it, as `of_file` does for a file; a screen's does not, nor does an
    audio segment."""
    store, cap = capture(SCREEN + TONE + AV)
    run_until(cap, lambda: all(len(store.select(sources=[name])) >= 2 for name in ("screen", "tone", "av")),
              what="two segments of each source")
    cap.stop()
    assert {item.meta.get("sound") for item in store.select(sources=["av"])} == {True}
    assert {item.kind for item in store.select(sources=["av"])} == {"video"}
    for name in ("screen", "tone"):
        assert {item.meta.get("sound") for item in store.select(sources=[name])} == {None}, name


def test_a_segment_found_after_a_crash_records_its_sound_too(home, ffmpeg):
    store = Store(home)
    folder = home / "streams" / "av" / "run"
    folder.mkdir(parents=True)
    run = [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=4", "-t", "1"]
    subprocess.run([*run, "-f", "lavfi", "-i", "sine=frequency=440", "-c:v", "libx264", "-c:a", "libopus", "-shortest",
                    str(folder / "000000.mkv")], check=True)
    subprocess.run([*run, "-c:v", "libx264", str(folder / "000001.mkv")], check=True)
    launch = {"source": "av", "folder": "streams/av/run", "started": 1000.0, "first_start": None}
    index_orphans(store, launch)
    by_name = {store.path_of(item).name: item for item in store.select(sources=["av"])}
    assert by_name["000000.mkv"].meta == {"sound": True} and by_name["000001.mkv"].meta == {}
    assert has_sound(folder / "000000.mkv") and not has_sound(folder / "000001.mkv")
    assert has_sound(folder / "gone.mkv") is False, "a file PyAV cannot open has no sound to report"


def test_pause_and_resume_make_a_new_launch(home, capture):
    store, cap = capture(TONE)
    run_until(cap, lambda: len(store.select(sources=["tone"])) >= 1, what="a segment")
    cap.pause(True)
    assert cap.status()["tone"]["state"] == "paused" and not cap.runs
    assert store.launches("tone")[0]["ended"] is not None
    cap.step(time.time())
    assert not cap.runs
    time.sleep(1.1)  # the next launch's folder is named by the second
    cap.pause(False)
    run_until(cap, lambda: "tone" in cap.runs, what="a new ffmpeg")
    assert len(store.launches("tone")) == 2


def test_restarts_after_an_unasked_exit_and_at_midnight(home, capture):
    store, cap = capture(TONE)
    run_until(cap, lambda: len(store.select(sources=["tone"])) >= 1, what="a segment")
    cap.runs["tone"].process.kill()
    cap.runs["tone"].process.wait()
    cap.step(time.time())
    assert cap.status()["tone"]["state"] == "waiting" and cap.status()["tone"]["restarts"] == 1
    first = store.launches("tone")[0]
    assert first["ended"] is not None
    run_until(cap, lambda: "tone" in cap.runs, timeout=5, what="a restart after the backoff")
    assert len(store.launches("tone")) == 2
    cap.step(time.time() + 86400)  # the next UTC day
    assert len(store.launches("tone")) == 3 and "tone" in cap.runs
    assert store.launches("tone")[1]["ended"] is not None


def test_a_missing_ffmpeg_backs_off_and_leaves_nothing(home):
    write_config(home, TONE.replace("[sources.tone]", '[memory]\nffmpeg = "no-such-ffmpeg.exe"\n\n[sources.tone]'))
    store = Store(home)
    cap = Capture(store, configs.load(home))
    cap.step(100.0)
    cap.step(100.5)
    assert cap.status()["tone"]["state"] == "waiting" and cap.status()["tone"]["error"]
    assert len(store.launches("tone")) == 1  # the second step waited
    assert not list((home / "streams" / "tone").iterdir())


def source(toml: str, home="x") -> configs.Source:
    return next(iter(configs.load_text(home, toml).sources.values()))


def test_commands():
    ddagrab = source('[sources.s]\nkind = "ffmpeg"\npreset = "screen"\ninput = ["-f", "lavfi", "-i", "ddagrab=framerate=2"]')
    args, media = command(ddagrab, "ffmpeg")
    assert media == "video" and "hwdownload,format=bgra,scale='min(iw,1920)':-2,format=yuv420p,mpdecimate=max=119" in args
    assert args[args.index("-segment_time") + 1] == "60" and args[-1] == "%06d.mkv"
    assert args[args.index("-force_key_frames") + 1] == "expr:gte(t,n_forced*10)"
    assert args[args.index("-segment_list_type") + 1] == "csv" and "-nostdin" not in args

    other = source('[sources.s]\nkind = "ffmpeg"\npreset = "screen"\ninput = ["-f", "gdigrab", "-i", "desktop"]\nsegment_s = 4')
    args, _ = command(other, "ffmpeg")
    assert "hwdownload" not in " ".join(args) and "scale='min(iw,1920)':-2,format=yuv420p,mpdecimate=max=119" in args
    assert args[args.index("-segment_time") + 1] == "4"
    assert args[args.index("-force_key_frames") + 1] == "expr:gte(t,n_forced*4)"

    wide = source('[sources.s]\nkind = "ffmpeg"\npreset = "screen"\ninput = ["-i", "x"]\nmax_width = 1280')
    assert "scale='min(iw,1280)':-2,format=yuv420p,mpdecimate=max=119" in command(wide, "ffmpeg")[0]

    args, media = command(source('[sources.s]\nkind = "ffmpeg"\npreset = "audio"\ninput = ["-f", "dshow", "-i", "audio=Mic"]'),
                          "ffmpeg")
    assert media == "audio" and args[-1] == "%06d.mka" and "libopus" in args

    args, media = command(source('[sources.s]\nkind = "ffmpeg"\npreset = "camera"\nencode = ["-c", "copy"]\n'
                                 'input = ["-f", "v4l2", "-i", "/dev/video0"]'), "ffmpeg")
    assert media == "video" and "libx264" not in args and ["-c", "copy"] == args[args.index("-c"):args.index("-c") + 2]

    args, media = command(source('[sources.s]\nkind = "ffmpeg"\nencode = ["-vn", "-c:a", "aac"]\ninput = ["-i", "x"]'), "ffmpeg")
    assert media == "audio"


def test_named_kinds_expand_to_ffmpeg_inputs_and_the_presets_of_their_media():
    screen = source('[sources.s]\nkind = "screen"\ndevice = "1"\nfps = 4')
    assert opens(screen, "win32") == ["-f", "lavfi", "-i", "ddagrab=output_idx=1:framerate=4"]
    assert opens(screen, "linux") == ["-f", "x11grab", "-framerate", "4", "-i", "1"]
    assert opens(source('[sources.s]\nkind = "screen"'), "win32")[-1] == "ddagrab=output_idx=0:framerate=2"
    mic = source('[sources.m]\nkind = "microphone"\ndevice = "Microphone (Realtek(R) Audio)"')
    assert opens(mic, "win32") == ["-f", "dshow", "-i", "audio=Microphone (Realtek(R) Audio)"]
    assert opens(mic, "linux") == ["-f", "pulse", "-i", "Microphone (Realtek(R) Audio)"]
    camera = source('[sources.c]\nkind = "camera"\ndevice = "HD Webcam"\nfps = 15')
    assert opens(camera, "win32") == ["-f", "dshow", "-framerate", "15", "-i", "video=HD Webcam"]
    assert opens(source('[sources.c]\nkind = "camera"\ndevice = "/dev/video2"'), "linux") == ["-f", "v4l2", "-i", "/dev/video2"]
    args, media = command(screen, "ffmpeg")  # a named kind takes its kind's preset
    assert media == "video" and any("mpdecimate" in arg for arg in args)
    args, media = command(mic, "ffmpeg")
    assert media == "audio" and args[-1] == "%06d.mka" and "libopus" in args
    args, media = command(camera, "ffmpeg")
    assert media == "video" and "libx264" in args and not any("mpdecimate" in arg for arg in args)


def test_a_dshow_listing_is_read_for_its_audio_and_video_devices():
    listing = ('[dshow @ 0000] DirectShow video devices (some may be both video and audio devices)\n'
               '[dshow @ 0000]  "HD Webcam" (video)\n[dshow @ 0000]     Alternative name "@device_pnp_x"\n'
               '[dshow @ 0000] DirectShow audio devices\n[dshow @ 0000]  "Microphone (Realtek(R) Audio)" (audio)\n'
               '[dshow @ 0000]  "Stereo Mix" (audio)\n')
    assert parse_dshow(listing) == {"audio": [{"device": "Microphone (Realtek(R) Audio)"}, {"device": "Stereo Mix"}],
                                    "video": [{"device": "HD Webcam"}]}


def test_screen_preset_scales_wide_frames_to_yuv420p(tmp_path, ffmpeg):
    """A frame wider than `max_width` is scaled down to it, aspect kept, in yuv420p; a narrower one keeps its size."""
    def record(size, extra=""):
        folder = tmp_path / size
        folder.mkdir()
        args, _ = command(source(f'[sources.s]\nkind = "ffmpeg"\npreset = "screen"\nsegment_s = 1\n{extra}\n'
                                 f'input = ["-t", "1", "-f", "lavfi", "-i", "testsrc2=size={size}:rate=2"]'), ffmpeg)
        subprocess.run(args, cwd=folder, check=True, capture_output=True, timeout=60)
        with av.open(str(next(folder.glob("*.mkv")))) as container:
            codec = container.streams.video[0].codec_context
            return codec.width, codec.height, codec.pix_fmt

    assert record("3840x2160") == (1920, 1080, "yuv420p")
    assert record("3840x2400", "max_width = 1280") == (1280, 800, "yuv420p")
    assert record("640x360") == (640, 360, "yuv420p")
