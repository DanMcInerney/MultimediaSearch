"""A synthetic day of an owner's laptop and phone (TESTBED.md): the files as the devices write them, each remembered at
its time, so a test asks the questions an owner would. `tests/test_day.py` asks with the fake decider (the sounds are
beeps and rings, no voices); `tests/test_codex.py` asks Codex, over the same day built for "yesterday" with Windows
speech (SAPI) for the voices.

    events = build(folder, day, speech)   # writes the files, and says what happened when
    load(api, folder, events)             # remembers each at its time, adds the log source, waits for its records

`day` is local midnight. Everything is at most 5 minutes long, so a whole-day scan is a few hundred windows.

| Time        | Source   | What                                                                                     |
| 08:05       | mail     | `gate.eml`, a version: "the new gate code is 4821"                                       |
| 08:30       | phone    | `IMG_0042.MOV`, portrait (display matrix), a green sign "PARKING LEVEL 3", corner red    |
| 09:00       | notes    | `notes.txt` v1, UTF-16LE with a BOM: "keep logs for 45 days"                             |
| 09:12       | phone    | `voice_memo.m4a` (speech, or a beep)                                                     |
| 10:00-10:05 | screen   | `screen.mkv`, variable frame rate; 10:00:20-10:00:50 a red panel "ERROR E-4417: disk full" |
| 10:45       | phone    | `IMG_2044.jpg`, EXIF 6, whiteboard "SHIP 2.1 FRI", corner red                            |
| 11:20 13:05 16:40 | doorbell | `ring-1120.wav` ..., 10 s each, the ding-dong 2-4 s in                                |
| 11:50       | downloads | `invoice.pdf`, a PDF with a text layer: "Invoice 2026-118", "Total 342.50 EUR"            |
| 12:00-12:08 | server   | `server.jsonl`, a log source: ERR-503 x5, ERR-429 x2, "deploy finished" at 12:07:31      |
| 14:00-14:05 | meeting  | `meeting.mp4`, a call: speech (or silence), a phone ring at 14:03:10                     |
| 15:00       | screen   | `screen-1500.mkv`, a blue editor, title bar "VS Code: app.ts"                            |
| 16:00       | notes    | a note: "dentist moved to Thursday 9:30"                                                 |
| 17:30       | notes    | `notes.txt` v2, UTF-8 with a BOM: "keep logs for 30 days"                                |
| 18:00       | phone    | `PTT-<date>-WA0007.opus`, WhatsApp voice note: "Sam here, bring the blue folder"         |
| 20:00 21:00 | garage   | `garage-2000.jpg` (light on, bright) and `garage-2100.jpg` (off, dark)                   |

Pictures carry their ground truth the way the corpus does: the top-left quarter of an upright picture is red, so a
picture shown sideways is not `corner:red`. The build checks each rotated file against ffmpeg's and Pillow's own idea of
upright, so a wrong fixture fails here and never in the reader.
"""

from __future__ import annotations

import email.utils
import json
import os
import subprocess
import wave
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from corpus import text_pdf
from PIL import Image, ImageDraw, ImageFont, ImageOps

from mms.config import ffmpeg_exe

RATE = 16000
LOG_SOURCE = "server"
FIXED_DAY = datetime(2026, 10, 31).timestamp()  # tier A's day: a fixed date
DOORBELLS = ("11:20", "13:05", "16:40")
RING_AT = {"11:20": 2.0, "13:05": 3.0, "16:40": 4.0}  # seconds into each doorbell file where the ding starts (the files differ: `remember` skips a recording it holds)
MEETING_RING = "14:03:10"
ERROR_SHOWN = ("10:00:20", "10:00:50")
DEPLOYED = "12:07:31"
RED, GREEN, BLUE = (220, 20, 20), (20, 150, 50), (0, 0, 200)


def yesterday() -> float:
    """Local midnight before the run."""
    return datetime.combine(date.today() - timedelta(days=1), time.min).timestamp()


def at(day: float, clock: str) -> float:
    """The epoch time of `clock` ("14:03" or "14:03:10", local) on the day that starts at `day`."""
    parts = [int(p) for p in clock.split(":")]
    d = datetime.fromtimestamp(day)
    return datetime(d.year, d.month, d.day, *parts).timestamp()


@dataclass(frozen=True)
class Event:
    """One thing that happened: `what` is `note`, `file` (a version), `recording` (an audio or video item) or `log` (the
    file the log source reads). `key` names it for a test."""

    key: str
    t: float
    source: str
    what: str
    path: Path | None = None
    note: str | None = None


def find(events: list[Event], key: str) -> Event:
    return next(e for e in events if e.key == key)


# -- sound ------------------------------------------------------------------------------------------------------------

class SpeechUnavailable(RuntimeError):
    """Windows speech (SAPI) cannot make a voice here; the message says why."""


SAPI = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "try { $s.SelectVoice($env:MMS_VOICE) } catch { }; $s.Rate = 0; "
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
        "$s.SetOutputToWaveFile($env:MMS_OUT, $f); $s.Speak($env:MMS_SAY); $s.Dispose()")


def speak(text: str, out: Path, voice: str = "Microsoft Zira Desktop") -> None:
    """`text` spoken by a Windows voice into `out`, a 16 kHz mono WAV (the default voice when `voice` is not installed)."""
    try:
        done = subprocess.run(["powershell", "-NoProfile", "-Command", SAPI], capture_output=True, text=True,
                              timeout=120, env=dict(os.environ, MMS_VOICE=voice, MMS_OUT=str(out), MMS_SAY=text))
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SpeechUnavailable(f"Windows speech (SAPI) cannot run here: {error}") from None
    if done.returncode or not out.exists() or out.stat().st_size < 1000:
        raise SpeechUnavailable(f"Windows speech (SAPI) made no sound: {(done.stderr or done.stdout).strip()[:200]}")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def write_wav(path: Path, samples: np.ndarray, rate: int = RATE) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def tone(hz: float, seconds: float, amp: float = 0.5, decay_s: float | None = None) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    wave_ = amp * np.sin(2 * np.pi * hz * t)
    return (wave_ * np.exp(-t / decay_s) if decay_s else wave_).astype(np.float32)


def beep() -> np.ndarray:
    """1 kHz at amplitude 0.5 for 0.4 s: a 5 s window holding it has RMS about 0.1."""
    return tone(1000, 0.4)


def ding_dong() -> np.ndarray:
    """A doorbell: 659 Hz then 523 Hz, 0.6 s each, decaying."""
    return np.concatenate([tone(659, 0.6, 0.6, 0.3), tone(523, 0.6, 0.6, 0.3)])


def us_ring() -> np.ndarray:
    """A US phone ring: 440 + 480 Hz for 2 s."""
    return tone(440, 2, 0.25) + tone(480, 2, 0.25)


def quiet(seconds: float, dbfs: float = -50.0, seed: int = 1) -> np.ndarray:
    """Room noise at `dbfs` RMS."""
    noise = np.random.default_rng(seed).normal(size=int(seconds * RATE))
    return (noise / noise.std() * 10 ** (dbfs / 20)).astype(np.float32)


def place(seconds: float, *parts: tuple[float, np.ndarray]) -> np.ndarray:
    """Silence of `seconds` with each `(offset_s, samples)` laid in."""
    out = np.zeros(int(seconds * RATE), dtype=np.float32)
    for offset, samples in parts:
        start = int(offset * RATE)
        out[start:start + len(samples)] += samples[:len(out) - start]
    return out


# -- pictures ---------------------------------------------------------------------------------------------------------

def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def whiteboard(text: str | None, size: tuple[int, int] = (1280, 720)) -> Image.Image:
    """An upright whiteboard photo: marker `text` (or none) on white, and a red panel exactly over the top-left quarter."""
    w, h = size
    image = Image.new("RGB", size, (245, 245, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, w // 2 - 1, h // 2 - 1], fill=RED)
    if text:
        draw.text((w // 4, h * 5 // 8), text, fill=(15, 15, 40), font=font(h // 6))
    return image


def same(a: Image.Image, b: Image.Image, limit: float = 25.0) -> bool:
    """Whether two pictures show the same thing: their sizes match and their 48x48 thumbnails differ by under `limit`
    levels on average."""
    def thumb(image: Image.Image) -> np.ndarray:
        return np.asarray(image.convert("RGB").resize((48, 48)), dtype=float)
    return a.size == b.size and float(np.abs(thumb(a) - thumb(b)).mean()) < limit


def sideways(path: Path, upright: Image.Image, orientation: int = 6) -> None:
    """Save `upright` as a phone does: pixels turned, and EXIF `orientation` saying how to turn them back. Checked
    against Pillow's own `exif_transpose`."""
    turn = {6: Image.Transpose.ROTATE_90, 8: Image.Transpose.ROTATE_270, 3: Image.Transpose.ROTATE_180}[orientation]
    exif = Image.Exif()
    exif[0x0112] = orientation
    upright.transpose(turn).save(path, quality=92, exif=exif)
    with Image.open(path) as stored:
        back = ImageOps.exif_transpose(stored).convert("RGB")
    assert same(back, upright), f"{path.name}: EXIF {orientation} does not turn back to the upright picture"


def parking_card(size: tuple[int, int] = (360, 640)) -> Image.Image:
    """A portrait phone frame: a red panel in the top-left quarter, a green sign in the bottom half."""
    w, h = size
    image = Image.new("RGB", size, (60, 60, 70))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, w // 2 - 1, h // 2 - 1], fill=RED)
    draw.rectangle([0, h // 2, w - 1, h - 1], fill=GREEN)
    draw.text((w // 10, h * 5 // 8), "PARKING\nLEVEL 3", fill="white", font=font(w // 5))
    return image


def arrow_card(size: tuple[int, int] = (360, 640)) -> Image.Image:
    """A portrait frame with a big white arrow pointing up."""
    w, h = size
    image = Image.new("RGB", size, (20, 40, 120))
    draw = ImageDraw.Draw(image)
    draw.polygon([(w // 2, h // 6), (w * 7 // 8, h // 2), (w // 8, h // 2)], fill="white")
    draw.rectangle([w * 3 // 8, h // 2, w * 5 // 8, h * 5 // 6], fill="white")
    return image


def screen_card(text: str | None, size: tuple[int, int] = (1280, 720)) -> Image.Image:
    """A screen: a light desktop with a thin task bar, or `text` in white on a full red panel (an error)."""
    w, h = size
    if text:
        image = Image.new("RGB", size, RED)
        ImageDraw.Draw(image).text((w // 12, h * 2 // 5), text, fill="white", font=font(h // 9))
        return image
    image = Image.new("RGB", size, (235, 235, 235))
    ImageDraw.Draw(image).rectangle([0, h - 30, w, h], fill=(40, 40, 50))
    return image


def editor_card(size: tuple[int, int] = (1280, 720)) -> Image.Image:
    """A blue editor with the title bar `VS Code: app.ts`."""
    w, h = size
    image = Image.new("RGB", size, BLUE)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, w, 48], fill=(40, 60, 230))
    draw.text((16, 6), "VS Code: app.ts", fill="white", font=font(32))
    for i, line in enumerate(("export function main() {", "  const port = 8080;", "  listen(port);", "}")):
        draw.text((40, 90 + i * 40), line, fill=(230, 230, 255), font=font(28))
    return image


def garage_card(light_on: bool, size: tuple[int, int] = (640, 360)) -> Image.Image:
    """The garage camera: a bright room under a lit lamp, or a dark one."""
    w, h = size
    image = Image.new("RGB", size, (250, 248, 235) if light_on else (14, 14, 16))
    draw = ImageDraw.Draw(image)
    draw.rectangle([w // 8, h // 4, w * 7 // 8, h - 20], outline=(150, 150, 150) if light_on else (45, 45, 50), width=4)
    draw.ellipse([w // 2 - 20, 20, w // 2 + 20, 60], fill=(255, 225, 90) if light_on else (55, 55, 60))
    draw.text((w // 8 + 12, h // 4 + 12), "GARAGE", fill=(90, 90, 90) if light_on else (60, 60, 66), font=font(30))
    return image


# -- video and sound files --------------------------------------------------------------------------------------------

def ff(*args: str) -> None:
    done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"ffmpeg {' '.join(args)[:200]} failed: {done.stderr.strip()[-400:]}")


def upright_frame(path: Path, folder: Path) -> Image.Image:
    """The first frame of `path` as ffmpeg itself turns it (it applies the display matrix)."""
    ff("-i", str(path), "-frames:v", "1", str(folder / "first.png"))
    with Image.open(folder / "first.png") as frame:
        return frame.convert("RGB")


def portrait_video(path: Path, upright: Image.Image, work: Path, *, rotation: int = -90, codec: str = "libx264",
                   pix_fmt: str = "yuv420p", seconds: int = 2, fps: int = 5, tags: tuple[str, ...] = ()) -> None:
    """A phone's portrait clip: the frames coded sideways and a display matrix of `rotation` degrees, so a player turns
    them upright. `tags` (ffmpeg output options, e.g. `-metadata creation_time=...`) are on both passes: the copy keeps no
    creation time of its own. Checked against ffmpeg's own decode, which applies the matrix."""
    turn = Image.Transpose.ROTATE_90 if rotation == -90 else Image.Transpose.ROTATE_270
    upright.transpose(turn).save(work / "coded.png")
    ff("-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", str(work / "coded.png"), "-c:v", codec,
       *(["-x265-params", "log-level=error"] if codec == "libx265" else []), "-pix_fmt", pix_fmt, *tags,
       str(work / ("coded" + path.suffix)))
    ff("-display_rotation", str(rotation), "-i", str(work / ("coded" + path.suffix)), "-c", "copy", *tags, str(path))  # (a copy drops creation_time)
    first = upright_frame(path, work)
    assert same(first, upright, 40), f"{path.name}: a display matrix of {rotation} does not turn the frames upright"


def vfr_video(path: Path, work: Path, cards: list[tuple[Image.Image, float]]) -> None:
    """A screen recording that stores a frame only when the screen changes: each card is on for its seconds."""
    lines = ["ffconcat version 1.0"]
    for i, (card, seconds) in enumerate(cards):
        card.save(work / f"card{i}.png")
        lines += [f"file card{i}.png", f"duration {seconds}"]
    lines.append(f"file card{len(cards) - 1}.png")
    (work / "cards.ffconcat").write_text("\n".join(lines) + "\n")
    ff("-f", "concat", "-safe", "0", "-i", str(work / "cards.ffconcat"), "-fps_mode", "vfr", "-c:v", "libx264",
       "-preset", "veryfast", "-tune", "stillimage", "-pix_fmt", "yuv420p", str(path))


def still_with_sound(path: Path, card: Image.Image, samples: np.ndarray, work: Path) -> None:
    """A call recording: a still picture at 1 frame a second and `samples` as AAC."""
    card.save(work / "call.png")
    write_wav(work / "call.wav", samples)
    seconds = len(samples) / RATE
    ff("-loop", "1", "-framerate", "1", "-t", f"{seconds:g}", "-i", str(work / "call.png"), "-i", str(work / "call.wav"),
       "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-c:a", "aac",
       "-b:a", "48k", "-shortest", str(path))
    (work / "call.wav").unlink()


def encode_sound(path: Path, samples: np.ndarray, work: Path, *codec: str) -> None:
    write_wav(work / "sound.wav", samples)
    ff("-i", str(work / "sound.wav"), *codec, str(path))
    (work / "sound.wav").unlink()


def voice(text: str, speech: bool, work: Path, voice_name: str, offset: float = 0.5) -> np.ndarray:
    """What a recording of `text` holds: the spoken words (speech), or a beep in the same place."""
    if not speech:
        return place(offset + 1.5, (offset, beep()))
    speak(text, work / "say.wav", voice_name)
    spoken = read_wav(work / "say.wav")
    (work / "say.wav").unlink()
    return place(offset + len(spoken) / RATE + 0.5, (offset, spoken))


# -- the day ----------------------------------------------------------------------------------------------------------

def build(folder: Path, day: float, speech: bool) -> list[Event]:
    """Write the day's files under `folder`, and return what happened when, oldest first. `speech` makes the voices
    with Windows speech (SpeechUnavailable when it cannot); otherwise a beep stands in for each voice and the call is
    silent but for its ring."""
    folder.mkdir(parents=True, exist_ok=True)
    work = folder / "_work"
    work.mkdir(exist_ok=True)
    events: list[Event] = []

    def add(key: str, clock: str, source: str, what: str, path: Path | None = None, note: str | None = None) -> Path:
        events.append(Event(key, at(day, clock), source, what, path, note))
        return path

    def path(*parts: str) -> Path:
        target = folder.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    date_text = datetime.fromtimestamp(day)
    sent = email.utils.format_datetime(datetime.fromtimestamp(at(day, "08:05")).astimezone())
    gate = path("mail", "gate.eml")
    gate.write_bytes(("From: Sam <sam@example.com>\r\nTo: me@example.com\r\nSubject: The gate\r\n"
                      f"Date: {sent}\r\nMessage-ID: <gate-1@example.com>\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
                      "Hi,\r\n\r\nThe new gate code is 4821. It works from today.\r\n\r\nSam\r\n").encode())
    add("gate", "08:05", "mail", "file", gate)

    parking = path("phone", "IMG_0042.MOV")
    portrait_video(parking, parking_card(), work, codec="libx265", pix_fmt="yuv420p10le",
                   tags=("-tag:v", "hvc1", "-metadata", f"creation_time={date_text:%Y-%m-%d}T08:30:00Z"))
    add("parking", "08:30", "phone", "recording", parking)

    path("notes", "v1", "notes.txt").write_bytes("keep logs for 45 days\r\n".encode("utf-16"))
    add("notes-1", "09:00", "notes", "file", folder / "notes" / "v1" / "notes.txt")

    memo = path("phone", "voice_memo.m4a")
    encode_sound(memo, voice("Reminder to self: call the plumber about the kitchen tap.", speech, work,
                             "Microsoft Zira Desktop", 1.0), work, "-c:a", "aac")
    add("memo", "09:12", "phone", "recording", memo)

    screen = path("screen", "screen.mkv")
    vfr_video(screen, work, [(screen_card(None), 20), (screen_card("ERROR E-4417: disk full"), 30),
                             (screen_card(None), 250)])
    add("screen", "10:00", "screen", "recording", screen)

    board = path("phone", "IMG_2044.jpg")
    sideways(board, whiteboard("SHIP 2.1 FRI"), 6)
    add("whiteboard", "10:45", "phone", "file", board)

    for clock in DOORBELLS:
        ring = path("doorbell", f"ring-{clock.replace(':', '')}.wav")
        write_wav(ring, place(10, (RING_AT[clock], ding_dong())))
        add(f"doorbell-{clock.replace(':', '')}", clock, "doorbell", "recording", ring)

    invoice = path("downloads", "invoice.pdf")
    invoice.write_bytes(text_pdf(["Invoice 2026-118", "Total 342.50 EUR"]))
    add("invoice", "11:50", "downloads", "file", invoice)

    lines = [(0, "info", "server started"), (65, "error", "upstream timeout", "ERR-503"),
             (140, "error", "rate limited", "ERR-429"), (180, "error", "upstream timeout", "ERR-503"),
             (240, "error", "upstream timeout", "ERR-503"), (280, "error", "rate limited", "ERR-429"),
             (330, "error", "upstream timeout", "ERR-503"), (370, "error", "upstream timeout", "ERR-503"),
             (451, "info", "deploy finished"), (470, "info", "health check ok")]
    start = at(day, "12:00:00")
    log = path("server", "server.jsonl")
    log.write_text("".join(json.dumps({"ts": datetime.fromtimestamp(start + s).isoformat(timespec="seconds"),
                                       "level": level, "msg": msg, **({"code": code[0]} if code else {})}) + "\n"
                           for s, level, msg, *code in lines), encoding="utf-8")
    add("server", "12:00", LOG_SOURCE, "log", log)

    meeting = path("meeting", "meeting.mp4")
    talk = (voice("Okay everyone, let's start with the release schedule. The deploy is planned for Friday.", True, work,
                  "Microsoft Zira Desktop", 20.0) if speech else np.zeros(1, dtype=np.float32))
    call = place(300, (0.0, talk), (at(day, MEETING_RING) - at(day, "14:00"), us_ring()))
    card = Image.new("RGB", (320, 180), (110, 110, 115))
    ImageDraw.Draw(card).text((20, 70), "Team call", fill="white", font=font(36))
    still_with_sound(meeting, card, call, work)
    add("meeting", "14:00", "meeting", "recording", meeting)

    editor = path("screen", "screen-1500.mkv")
    vfr_video(editor, work, [(editor_card(), 60)])
    add("editor", "15:00", "screen", "recording", editor)

    add("dentist", "16:00", "notes", "note", note="dentist moved to Thursday 9:30")

    path("notes", "v2", "notes.txt").write_bytes(b"\xef\xbb\xbf" + "keep logs for 30 days\r\n".encode())
    add("notes-2", "17:30", "notes", "file", folder / "notes" / "v2" / "notes.txt")

    note = path("phone", f"PTT-{date_text:%Y%m%d}-WA0007.opus")
    encode_sound(note, voice("Sam here, bring the blue folder", speech, work, "Microsoft David Desktop", 0.5), work,
                 "-c:a", "libopus", "-ar", "16000", "-ac", "1")
    add("voice-note", "18:00", "phone", "recording", note)

    for clock, on in (("20:00", True), ("21:00", False)):
        still = path("garage", f"garage-{clock.replace(':', '')}.jpg")
        garage_card(on).save(still, quality=92)
        add(f"garage-{clock.replace(':', '')}", clock, "garage", "file", still)

    for leftover in work.iterdir():
        leftover.unlink()
    work.rmdir()
    return sorted(events, key=lambda e: e.t)


def load(api, folder: Path, events: list[Event], wait: float = 60) -> None:
    """Add the log source, remember every other event at its time, and wait until the log's records are all in."""
    from conftest import wait_for  # the tests' own helper

    lines = 0
    for event in events:
        if event.what == "log":
            lines = len(event.path.read_text(encoding="utf-8").splitlines())
            api.call("source_set", name=event.source, kind="log", path=str(event.path.parent), glob="*.jsonl",
                     time_field="ts", max_gb=1)
        elif event.what == "note":
            api.call("remember", note=event.note, t=event.t, source=event.source)
        else:
            api.call("remember", path=str(event.path), t=event.t, source=event.source)

    def records() -> bool:
        found = next((s for s in api.call("status")["sources"] if s["name"] == LOG_SOURCE), None)
        return found is not None and found["records"] >= lines

    wait_for(records, timeout=wait, what=f"the {lines} records of the log source")
