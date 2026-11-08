"""What a file is, from its bytes and never its name (DESIGN.md §7), and how text and log lines are decoded.

`of_file` and `of_bytes` give `{"media": type}`, and `"sound": True` for a video with an audio stream. The order is: a
known signature; for a container, PyAV's streams; git's text rule (a BOM, or UTF-16's NULs, else no NUL in the first
8,000 bytes once runs of 32 or more, the padding a crash leaves in a log, are dropped); else `application/octet-stream`.
A weak signature (a bare ISO box name at byte 4, MPEG-TS's sync bytes: words and fixed-width text can start so) counts
only when PyAV finds a stream; a strong one (`ftyp`, EBML, OggS, RIFF) keeps the container's type for a broken file.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HEAD = 8000  # the bytes looked at to tell text from binary, as git does
PADDING = re.compile(rb"\x00{32,}")
# What a terminal does not show: CSI sequences, OSC strings, and two-character escapes
TERMINAL = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
BOMS = ((b"\xff\xfe\x00\x00", "utf-32-le"), (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))

OCTET = "application/octet-stream"
PREFIXES = (  # (bytes at 0, media type or container)
    (b"\xff\xd8\xff", "image/jpeg"), (b"\x89PNG\r\n\x1a\n", "image/png"), (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"), (b"II*\x00", "image/tiff"), (b"MM\x00*", "image/tiff"),
    (b"\x00\x00\x01\x00", "image/vnd.microsoft.icon"), (b"\xff\x0a", "image/jxl"),
    (b"\x00\x00\x00\x0cJXL \r\n\x87\n", "image/jxl"), (b"fLaC", "audio/flac"), (b"#!AMR", "audio/amr"),
    (b"caff\x00\x01", "audio/x-caf"), (b"\xff\xf1", "audio/aac"), (b"\xff\xf9", "audio/aac"), (b"%PDF", "application/pdf"),
    (b"\xd0\xcf\x11\xe0", "application/x-ole-storage"), (b"\x1f\x8b", "application/gzip"),
    (b"(\xb5/\xfd", "application/zstd"), (b"7z\xbc\xaf'\x1c", "application/x-7z-compressed"),
    (b"Rar!\x1a\x07", "application/vnd.rar"), (b"\x7fELF", "application/x-elf"),
    (b"SQLite format 3\x00", "application/vnd.sqlite3"),
    (b"\x1a\x45\xdf\xa3", "matroska"), (b"OggS", "ogg"), (b"0&\xb2u\x8ef\xcf\x11", "asf"), (b"FLV\x01", "flv"),
    (b"\x00\x00\x01\xba", "ps"))
RIFF = {b"WEBP": "image/webp", b"WAVE": "audio/wav", b"AVI ": "avi"}
HEIC = {b"heic", b"heix", b"hevc", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"}
AVIF = {b"avif", b"avis"}
BRANDS = HEIC | AVIF
BOXES = {b"moov", b"mdat", b"wide", b"free"}  # what an ISO base media file (MP4, MOV, M4A, 3GP) may open with
TS_PACKET = 188
DIB_SIZES = (12, 40, 52, 56, 64, 108, 124)
OFFICE = {"word/": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
          "xl/": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          "ppt/": "application/vnd.openxmlformats-officedocument.presentationml.presentation"}
# Containers that keep a video type when PyAV cannot open them: (video, sound alone)
VIDEO_FIRST = {"avi": ("video/x-msvideo", "audio/x-msvideo"), "flv": ("video/x-flv", "audio/x-flv"),
               "ps": ("video/mpeg", "audio/mpeg")}
WEAK = {"box", "ts"}  # the containers whose signature text can have


def of_file(path: Path | str) -> dict:
    with open(path, "rb") as file:
        return _of(file, str(path))


def of_bytes(data: bytes) -> dict:
    file = io.BytesIO(data)
    return _of(file, file)


def is_text(media: str) -> bool:
    """Whether a media type is read as text (an SVG is, for its words)."""
    return media.startswith("text/") or media == "image/svg+xml"


def is_picture(media: str) -> bool:
    """Whether a media type is a picture, which is not read as text (an SVG is not)."""
    return media.startswith("image/") and not is_text(media)


def is_recording(media: str) -> bool:
    """Whether a media type is kept as a recording, with a time of its own: video, sound or a picture."""
    return media.startswith(("video/", "audio/")) or is_picture(media)


def _of(file, target) -> dict:
    """`target` is what PyAV opens: the file's path, or the bytes in a file."""
    head = file.read(HEAD)
    found = _signature(head, file)
    if found and "/" in found:
        return {"media": found}
    if found:
        opened = _container(found, head, target)
        if opened:
            return opened
    try:
        text = decode_text(head).lstrip()
    except ValueError:
        return {"media": OCTET}
    svg = text.startswith("<svg") or (text.startswith("<?xml") and "<svg" in text)
    return {"media": "image/svg+xml" if svg else "text/plain"}


def _signature(head: bytes, file) -> str | None:
    """The media type for known leading bytes, or a container's name (no "/") to open; None when none is known."""
    for prefix, found in PREFIXES:
        if head.startswith(prefix):
            return found
    if head[:4] == b"RIFF" and head[8:12] in RIFF:
        return RIFF[head[8:12]]
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "audio/aiff"
    if head[:2] == b"BM" and int.from_bytes(head[14:18], "little") in DIB_SIZES:
        return "image/bmp"
    if head[4:8] == b"ftyp" and head[8:12] in BRANDS:
        compatible = head[16:int.from_bytes(head[:4], "big")]
        return "image/avif" if head[8:12] in AVIF or b"avif" in compatible or b"avis" in compatible else "image/heic"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[4:8] in BOXES:
        return "box"
    if head.startswith(b"ID3") and len(head) >= 10 and head[3] < 5 and all(byte < 0x80 for byte in head[6:10]):
        file.seek(_tag_end(head))  # an ID3 tag comes before an MP3, a FLAC or an AAC: look past it
        return _signature(file.read(64), file) or "audio/mpeg"
    if len(head) > 2 * TS_PACKET and all(head[i] == 0x47 for i in range(0, len(head), TS_PACKET)):
        return "ts"
    if len(head) > 3 and head[0] == 0xff and head[1] & 0xe0 == 0xe0 and head[1] >> 3 & 3 != 1 \
            and head[1] >> 1 & 3 != 0 and head[2] >> 4 != 0xf and head[2] >> 2 & 3 != 3 \
            and head[:2] != b"\xff\xfe":  # that is a UTF-16 BOM: Layer I with a CRC is not worth the clash
        return "audio/mpeg"
    if head.startswith(b"MZ") and head[int.from_bytes(head[0x3c:0x40], "little"):][:4] == b"PE\x00\x00":
        return "application/vnd.microsoft.portable-executable"
    if head.startswith(b"PK\x03\x04"):
        return _zip(file)
    return None


def _tag_end(head: bytes) -> int:
    """Where an ID3 tag ends: its 10-byte header and the syncsafe size in it."""
    size = 0
    for byte in head[6:10]:
        size = size << 7 | byte
    return 10 + size


def _zip(file) -> str:
    """A ZIP's media type by what it holds: an Office file, an ODF or EPUB (its `mimetype` entry), else a plain zip."""
    file.seek(0)
    try:
        with zipfile.ZipFile(file) as archive:
            names = archive.namelist()
            for folder, media in OFFICE.items():
                if any(name.startswith(folder) for name in names):
                    return media
            if "mimetype" in names:
                return archive.read("mimetype").decode("ascii", errors="replace").strip() or "application/zip"
    except zipfile.BadZipFile:
        pass
    return "application/zip"


def _container(kind: str, head: bytes, target) -> dict | None:
    """A container is a video unless its only picture is cover art, and audio when it has sound and no picture. One
    PyAV cannot open keeps the container's default, so a broken recording is still a video, which the reader warns
    about; but a weak signature with no stream found is None: that file is not media."""
    video, audio, unopened = _types(kind, head)
    picture, sound = _streams(target, "mpegts" if kind == "ts" else None)  # else ffmpeg reads text as a tty video
    if kind == "ogg" and picture != "theora":
        picture = None
    if picture:
        return {"media": video, "sound": True} if sound else {"media": video}
    if kind in WEAK and not sound:
        return None
    return {"media": audio if sound else unopened}


def _types(kind: str, head: bytes) -> tuple[str, str, str]:
    """A container's media type for video, for sound alone, and for a file that cannot be opened."""
    if kind in ("mp4", "box"):
        brand = head[8:12] if head[4:8] == b"ftyp" else b""
        video = "video/quicktime" if brand == b"qt  " else "video/3gpp" if brand[:2] == b"3g" else "video/mp4"
        audio = "audio/3gpp" if brand[:2] == b"3g" else "audio/mp4"
        return video, audio, audio if brand == b"M4A " else video
    if kind == "matroska":
        sub = "webm" if b"webm" in head[:64] else "x-matroska"
        return f"video/{sub}", f"audio/{sub}", f"video/{sub}"
    if kind == "ogg":
        return "video/ogg", "audio/ogg", "audio/ogg"
    if kind == "asf":
        return "video/x-ms-wmv", "audio/x-ms-wma", "video/x-ms-asf"
    if kind == "ts":
        return "video/mp2t", "audio/mp2t", "video/mp2t"
    video, audio = VIDEO_FIRST[kind]
    return video, audio, video


def _streams(target, format: str | None = None) -> tuple[str | None, bool]:
    """(the codec of the first video stream that is not cover art, whether there is an audio stream); nothing when
    PyAV cannot open it (as `format` if one is given, else as it finds the file to be)."""
    import av
    try:
        if hasattr(target, "seek"):
            target.seek(0)
        with av.open(target, format=format) as container:
            picture = next((stream.codec_context.name for stream in container.streams.video
                            if not stream.disposition & av.stream.Disposition.attached_pic), None)
            return picture, bool(container.streams.audio)
    except (av.FFmpegError, OSError, ValueError):
        return None, False


def duration(path: Path, sound: bool = False) -> float | None:
    """How long a media file plays, from its packets (with `sound`, from its first audio stream's alone); None when
    PyAV finds nothing in it."""
    import av
    first = last = None
    try:
        with av.open(str(path)) as container:
            for packet in container.demux(*(container.streams.audio[:1] if sound else ())):
                if packet.pts is None or packet.stream.time_base is None:
                    continue
                begin = float(packet.pts * packet.stream.time_base)
                end = begin + float((packet.duration or 0) * packet.stream.time_base)
                first = begin if first is None else min(first, begin)
                last = end if last is None else max(last, end)
    except (av.FFmpegError, OSError, ValueError):
        pass  # a file cut short ends in an error: what was read counts
    return None if first is None else max(last - first, 0.0)


EXIF_TIMES = ((0x9003, 0x9011), (0x9004, 0x9012))  # DateTimeOriginal, then DateTimeDigitized, each with its offset tag
OFFSET = re.compile(r"([+-])([01]\d|2[0-3]):?([0-5]\d)")  # a valid UTC offset; any other is no offset


def recorded(path: Path, media: str) -> float | None:
    """The time a file says it was made: a picture's EXIF DateTimeOriginal, else DateTimeDigitized (with its offset tag
    when it has one, else in this machine's zone); a video's or sound's `creation_time` tag. None when it has none, it is
    blank or malformed, or the file cannot be read (a HEIC without a plugin, a truncated file)."""
    if is_picture(media):
        return _exif_time(path)
    if media.startswith(("video/", "audio/")):
        return _creation_time(path)
    return None


def _exif_time(path: Path) -> float | None:
    from PIL import ExifTags, Image
    try:
        with Image.open(path) as image:
            exif = image.getexif().get_ifd(ExifTags.IFD.Exif)
    except Exception:  # a damaged picture can raise anything while its tags are read; it is still a picture
        return None
    for time_tag, offset_tag in EXIF_TIMES:
        text = _tag_text(exif.get(time_tag))
        try:
            moment = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
        except ValueError:  # blank, zeroed ("0000:00:00 00:00:00") or malformed
            continue
        zone = OFFSET.fullmatch(_tag_text(exif.get(offset_tag)))
        if zone:
            moment = moment.replace(tzinfo=timezone(
                (1 if zone[1] == "+" else -1) * timedelta(hours=int(zone[2]), minutes=int(zone[3]))))
        try:
            return moment.timestamp()
        except (OverflowError, OSError, ValueError):  # a year the platform has no clock for
            continue
    return None


def _tag_text(value) -> str:
    """An EXIF text tag as a string, without the NULs and spaces cameras pad it with."""
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    return value.strip("\x00 ") if isinstance(value, str) else ""


def _creation_time(path: Path) -> float | None:
    import av
    try:
        with av.open(str(path)) as container:
            tag = container.metadata.get("creation_time") or next(
                (s.metadata["creation_time"] for s in container.streams if "creation_time" in s.metadata), None)
    except (av.FFmpegError, OSError):  # a file PyAV cannot open has no length either, which the caller reports
        return None
    try:
        moment = datetime.fromisoformat(tag.replace("Z", "+00:00")) if tag else None
    except ValueError:
        return None
    return None if moment is None else (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()


def has_sound(path: Path) -> bool:
    """Whether PyAV finds an audio stream in a media file, the fact `of_file` gives a file as `sound`; False when it
    cannot open it (the reader reports such a file when it scans it). About 1 to 6 ms for a 60 s segment."""
    return _streams(str(path))[1]


def decode_text(data: bytes) -> str:
    """Text as the file's bytes say: a BOM's codec, UTF-16 by its NULs, else UTF-8 with U+FFFD for a bad byte, unless
    over half its non-ASCII characters are bad bytes (a legacy file): then cp1252; newlines are `\\n`. Raises
    ValueError("binary: ...") for what git's rule says is not text."""
    for bom, codec in BOMS:
        if data.startswith(bom):
            text = data[len(bom):].decode(codec, errors="replace")
            break
    else:
        codec = _bare_utf16(data[:HEAD])
        if codec:
            text = data.decode(codec, errors="replace")
        else:
            if b"\x00" in PADDING.sub(b"", data[:HEAD]):
                raise ValueError("binary: NUL bytes among the first 8,000")
            data = PADDING.sub(b"", data)
            text = data.decode("utf-8-sig", errors="replace")
            bad = text.count("\ufffd")
            if bad and 2 * bad > len(text) - len(text.encode("ascii", "ignore")):
                text = data.decode("cp1252", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _bare_utf16(head: bytes) -> str | None:
    """UTF-16 with no BOM, told by its NULs: at least 40% of one byte of each pair, under 5% of the other's."""
    pairs = len(head) // 2
    even, odd = head[0::2].count(0), head[1::2].count(0)
    if odd >= 0.4 * pairs and even < 0.05 * pairs:
        return "utf-16-le"
    if even >= 0.4 * pairs and odd < 0.05 * pairs:
        return "utf-16-be"
    return None


def decode_line(raw: bytes) -> str:
    """A log or command line as a terminal shows it: UTF-8, else cp1252; escape sequences removed; after a lone `\\r`
    only what follows it (a progress bar's last state); stripped."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return TERMINAL.sub("", text.rstrip("\r\n")).rpartition("\r")[2].strip()
