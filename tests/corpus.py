"""Files as phones, cameras, browsers, Office and Windows write them (TESTBED.md, "Real files"), made offline and
deterministically (ffmpeg with one thread and bitexact output, Pillow, numpy, the standard library) in a few seconds, under
8 MB. `build(folder)` writes them in `pictures/`, `sound/`, `video/`, `text/`, `other/`, `documents/` and `bad/` and returns an `Entry`
for each, keyed by its path under `folder` (`pictures/IMG_2041.jpg`):

- `media`: what mms must record for it (`meta.media`), whatever its name;
- `kind`: what `remember` and a folder source make of it, `video`, `audio` (a recording), `image` (a picture, at an instant)
  or `file` (a version);
- `read`: what the reader hands d1: `video`, `audio`, `image` or `text`, or None for a file nothing reads;
- `stub` (a property, `signal["stub"]`): the file is only as much of its type as mms sniffs, not one a real reader opens (an
  OLE file, a 7z, an HEIC, a PE header, a SQLite header, a PDF with no page and the one-part Office zips of `other/`). It
  holds no words, and a test hands it to a decider to check the plumbing (the bytes arrive intact), never its content;
- `signal`: what d1 must get. `corner` is the colour of the picture's top-left quarter when upright (the card: red,
  green, bottom-left blue, yellow), `ink` that what is drawn shows, `size` the upright size, `frames` the `at` of each
  window an animation, a TIFF or a variable-rate video gives, `beep_s` where the 1 kHz beep is (6.0 s of 12 s, silence
  elsewhere), `sound` that a video has a soundtrack, `tone_s` that the soundtrack is that long a tone, `contains` words every text holds, `warn` what the warning for a
  broken file says, and `words` the words a reader that takes a whole document finds in it.

Pictures: EXIF-rotated and mirrored JPEGs, WebP, AVIF, CMYK, ICC, BMP, ICO, TIFF, GIF, 16-bit and float (one with a
NaN) and 1-bit scans, transparent PNGs, animations, pages, a gain-map MPO, banners, a 1x1 pixel. Sound, each with the beep: AAC, ALAC,
Opus, Vorbis, a pipe-written WebM, AMR, 3GP, an MP4 with no video, MP3 (VBR, with cover art, joined across a rate
change), FLAC, WAV in four codings, 5.1, WMA, AIFF, an audio-only AVI and FLV. Video: display-matrix rotations, HEVC 10-bit MOV, a variable-rate
screen recording, AV1, odd-size VP9 4:4:4, full-range, MJPEG AVI, MPEG-TS, WMV, a pipe-written WebM, a meeting with a
soundtrack, a 20 s video with 4 s of sound, a size change. Text and other files: UTF-8, BOMs, UTF-16 with and without
one, cp1252, CR-only lines, NUL padding, no extension, a long line, text that starts like a container (`free`, `wide`,
`caff`, rows of MPEG-TS's sync byte), a PNG named .txt, text named .mp3, and PDF, Office, archive, program and database
files that nothing reads. `documents/` is what a home folder holds besides media, under 200 KB: PDFs (with a text layer,
a scan, a locked one, and a text file that is a PDF), Word, Excel, PowerPoint, ODF and EPUB files, OLE files, RTF,
iCalendar, vCard and a mail with a JPEG attached, MIDI, tar, bz2 and xz, a SQLite database and a 7z. Broken files in `bad/`,
among them an MKV with only subtitles.

The documents that are not stubs are valid for a real reader of their type, and the build follows each as that reader does
(`doctext.py`): a PDF has a valid xref, an Office package has a content type for every part and the relationships that lead
from `_rels/.rels` to each part (a deck has its master, layout and theme), an ODF file a manifest of every member and
`office:version`, an EPUB 3 a package document with its identifier, language and time and a spine, a database opens in
SQLite, the archives in `tarfile`, `bz2` and `lzma`. The words of a document are found only by that path, so words in a part
no reader reaches fail the build. A build is deterministic (a PDF is dated by a fixed time, a zip stamped by `STAMP`), and
`build_documents` makes the documents part alone.

The build checks its own ground truth with tools other than mms's (ffmpeg's autorotate, Pillow's `exif_transpose`, the
files' bytes), so a wrong fixture fails the build and never the reader."""

from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import posixpath
import random
import re
import shutil
import sqlite3
import subprocess
import tarfile
import wave
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree as ET

import doctext
import imageio_ffmpeg
import numpy as np
from fake_decider import corners
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

from mms import reader
from mms.media import is_picture
from mms.store import Scope

CARD = (160, 90)
BEEP_AT, BEEP_S, LENGTH_S = 6.0, 0.4, 12
TONE_S, LONG_S = 4, 20  # a video of LONG_S seconds with TONE_S seconds of sound
QUIET = ["-hide_banner", "-loglevel", "error", "-y", "-nostdin"]
OUT = ["-threads", "1", "-fflags", "+bitexact", "-flags", "+bitexact"]
X264 = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
WORDS = ["café", "4821"]
BODY = "Gate code 4821 at the café on Rue Cler.\nThe second line is plain.\nThe third line has ünï in it.\n"


@dataclass(frozen=True)
class Entry:
    name: str
    path: Path
    media: str
    kind: str
    read: str | None
    signal: dict = field(default_factory=dict)

    @property
    def stub(self) -> bool:
        """Only as much of its type as mms sniffs, not a file a real reader opens: a test hands it to a decider to check that
        the bytes arrive intact, and never asserts what it holds."""
        return bool(self.signal.get("stub"))


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(*args, stdout=None) -> None:
    done = subprocess.run([ffmpeg_exe(), *QUIET, *map(str, args)], stdout=stdout or subprocess.DEVNULL,
                          stderr=subprocess.PIPE, check=False)
    if done.returncode:
        raise RuntimeError(f"ffmpeg {' '.join(map(str, args))}\n{done.stderr.decode('utf-8', 'replace')}")


def pipe(*args, to: Path) -> None:
    """A WebM written to a pipe, as a browser's MediaRecorder does: no duration, no cues."""
    with open(to, "wb") as out:
        run(*args, "pipe:1", stdout=out)


def card(width: int = 160, height: int = 90) -> Image.Image:
    """The upright picture every orientation is checked against: red top-left, green top-right, blue bottom-left, yellow
    bottom-right."""
    image = Image.new("RGB", (width, height))
    draw, x, y = ImageDraw.Draw(image), width // 2, height // 2
    for box, colour in (((0, 0, x - 1, y - 1), "red"), ((x, 0, width, y - 1), "lime"), ((0, y, x - 1, height), "blue"),
                        ((x, y, width, height), "yellow")):
        draw.rectangle(box, fill=colour)
    return image


def words(text: str, background=(0, 0, 0, 0), size: int = 28, mode: str = "RGBA") -> Image.Image:
    """Black words in the built-in font, large enough for 1% of the picture to be ink, on `background`, which by default
    is transparent with a hidden colour of black, as a logo editor leaves it."""
    image = Image.new("RGBA", (160, 90), background)
    ImageDraw.Draw(image).text((6, 20), text, fill=(0, 0, 0, 255), font=ImageFont.load_default(size=size))
    return image.convert(mode)


def ink_scan(dark, light, dtype) -> np.ndarray:
    """Dark words on a light page as an array of `dtype` (a scanner's 16 bits, or floats)."""
    page = words("SCAN", background=(255, 255, 255, 255)).convert("L")
    return np.where(np.asarray(page) < 128, dark, light).astype(dtype)


def beep_wav(path: Path) -> None:
    """12 s of silence at 48 kHz with a 1 kHz beep of amplitude 0.5 from 6.0 to 6.4 s."""
    rate = 48000
    t = np.arange(rate * LENGTH_S) / rate
    sound = np.where((t >= BEEP_AT) & (t < BEEP_AT + BEEP_S), 0.5 * np.sin(2 * np.pi * 1000 * t), 0.0)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes((sound * 32767).astype("<i2").tobytes())


STAMP = (2026, 10, 31, 8, 0, 0)  # the time inside every file made here, so that a build is the same each time
PDF_TIME = datetime(2026, 10, 31, 8, 0, 0)  # a PDF's /CreationDate and /ModDate
OFFICE = "application/vnd.openxmlformats-officedocument."
ODF = "application/vnd.oasis.opendocument."


def pdf(stream: bytes, trailer: bytes = b"") -> bytes:
    """A one-page PDF whose content stream is `stream` (FlateDecode), with valid xref and startxref offsets. `trailer`
    adds entries to the trailer dictionary."""
    objects = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
               b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 120]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
               b"<</Length %d/Filter/FlateDecode>>stream\n" % len(stream) + stream + b"\nendstream",
               b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer<</Size %d/Root 1 0 R%s>>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, trailer, xref)
    return bytes(out)


def text_pdf(lines: list[str]) -> bytes:
    """A PDF with a text layer: each of `lines` (Latin-1) is shown by a `Tj` operator in a deflated content stream."""
    show = b"".join(b"(%s) Tj 0 -18 Td\n" % re.sub(rb"([\\()])", rb"\\\1", line.encode("latin-1")) for line in lines)
    return pdf(zlib.compress(b"BT /F1 14 Tf 20 90 Td\n" + show + b"ET", 9))


def locked_pdf() -> bytes:
    """A stand-in for an encrypted PDF: a trailer with an `/Encrypt` dictionary and a content stream of seeded noise,
    which no tool inflates."""
    noise = random.Random(11)
    hexed = lambda n: noise.randbytes(n).hex().encode()  # noqa: E731
    return pdf(noise.randbytes(120), b"/ID[<%s><%s>]/Encrypt<</Filter/Standard/V 1/R 2/O<%s>/U<%s>/P -4>>"
               % (hexed(16), hexed(16), hexed(32), hexed(32)))


def zipped(members: dict[str, str]) -> bytes:
    """A zip of `members` in order, each stamped `STAMP`; `mimetype` is stored, as an ODF file and an EPUB need, the rest
    deflated."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            info.create_system, info.external_attr = 3, 0o644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def archive(files: dict[str, bytes]) -> bytes:
    """A tar (USTAR) of `files`, each owned by no one and stamped alike."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 1_793_000_000, 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def mail() -> bytes:
    """A message with a JPEG attached as base64, its Date, Message-ID and boundary fixed."""
    note = EmailMessage()
    note["From"], note["To"], note["Subject"] = "sam@example.test", "owner@example.test", "The gate"
    note["Date"] = format_datetime(datetime(2026, 10, 31, 8, 5, tzinfo=timezone.utc))
    note["Message-ID"] = "<corpus-gate-1@example.test>"
    note.set_content("The new gate code is 4821. It works from today.\n")
    photo = io.BytesIO()
    card().save(photo, "JPEG", quality=85)
    note.add_attachment(photo.getvalue(), maintype="image", subtype="jpeg", filename="gate.jpg")
    note.set_boundary("corpus-boundary")
    return bytes(note)


XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006"
CORE = "application/vnd.openxmlformats-package.core-properties+xml"
NS_R, NS_A = f'xmlns:r="{REL}"', 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
NS_P = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
NS_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
NS_S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


def ooxml(main: str, parts: dict[str, tuple[str, str]], links: dict[str, list[tuple[str, str]]]) -> dict[str, str]:
    """The members of an Office package, as Word, Excel and PowerPoint write one: `parts` are its XML parts as `{name:
    (content type, xml)}`, each given an Override in `[Content_Types].xml`, and `links` the relationships of a part (its
    name, or "" for the package) as `[(type, target)]`, numbered rId1, rId2 ... in order and written in the part's
    `_rels` folder. The package's own are the `main` part and the core properties."""
    parts = {**parts, "docProps/core.xml": (CORE, f'<cp:coreProperties xmlns:cp="{PKG}/metadata/core-properties" '
                                                  f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>corpus</dc:title>'
                                                  f'</cp:coreProperties>')}
    links = {"": [(f"{REL}/officeDocument", main), (f"{PKG}/relationships/metadata/core-properties", "docProps/core.xml")],
             **links}
    members = {"[Content_Types].xml": XML + f'<Types xmlns="{PKG}/content-types"><Default Extension="rels" ContentType='
                                            f'"application/vnd.openxmlformats-package.relationships+xml"/><Default '
                                            f'Extension="xml" ContentType="application/xml"/>'
                                            + "".join(f'<Override PartName="/{name}" ContentType="{kind}"/>'
                                                      for name, (kind, _) in parts.items()) + "</Types>"}
    for source, targets in links.items():
        folder, name = posixpath.split(source)
        members[posixpath.join(folder, "_rels", name + ".rels")] = XML + f'<Relationships xmlns="{PKG}/relationships">' + "".join(
            f'<Relationship Id="rId{n}" Type="{kind}" Target="{target}"/>' for n, (kind, target) in enumerate(targets, 1)
        ) + "</Relationships>"
    return {**members, **{name: XML + xml for name, (_, xml) in parts.items()}}


def docx_members(days: int = 30) -> dict[str, str]:
    """Minutes as Word writes them: a heading, and a paragraph in two runs (Word splits the text of one it has edited)."""
    return ooxml("word/document.xml", {"word/document.xml": (
        OFFICE + "wordprocessingml.document.main+xml",
        f'<w:document {NS_W}><w:body><w:p><w:r><w:t>Retention rule</w:t></w:r></w:p><w:p><w:r><w:t xml:space="preserve">Keep '
        f'logs for </w:t></w:r><w:r><w:t>{days} days</w:t></w:r></w:p></w:body></w:document>')}, {})


def xlsx_members() -> dict[str, str]:
    """A workbook with one sheet, as Excel writes it: the sheet is `<sheet r:id>` in the workbook, whose relationships
    give the worksheet and the shared strings; A1 is a shared string, B1 an inline one and C1 a number."""
    return ooxml("xl/workbook.xml", {
        "xl/workbook.xml": (OFFICE + "spreadsheetml.sheet.main+xml",
                            f'<workbook {NS_S} {NS_R}><sheets><sheet name="Budget" sheetId="1" r:id="rId1"/></sheets></workbook>'),
        "xl/worksheets/sheet1.xml": (OFFICE + "spreadsheetml.worksheet+xml",
                                     f'<worksheet {NS_S}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" '
                                     f't="inlineStr"><is><t>Gate code</t></is></c><c r="C1"><v>4821</v></c></row></sheetData>'
                                     f'</worksheet>'),
        "xl/sharedStrings.xml": (OFFICE + "spreadsheetml.sharedStrings+xml",
                                 f'<sst {NS_S} count="1" uniqueCount="1"><si><t>café</t></si></sst>')},
        {"xl/workbook.xml": [(f"{REL}/worksheet", "worksheets/sheet1.xml"), (f"{REL}/sharedStrings", "sharedStrings.xml")]})


def pptx_members() -> dict[str, str]:
    """A one-slide deck with what PowerPoint needs to open it: the presentation (master, slide list, sizes), a master, a
    blank layout, a theme and the slide, each related to the next and given a content type."""
    ns = f"{NS_A} {NS_R} {NS_P}"
    tree = '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>'
    fill = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    line = f'<a:ln w="6350" cap="flat" cmpd="sng" algn="ctr">{fill}<a:prstDash val="solid"/></a:ln>'
    colours = "".join(f'<a:{name}><a:srgbClr val="{value}"/></a:{name}>' for name, value in (
        ("dk2", "44546A"), ("lt2", "E7E6E6"), ("accent1", "4472C4"), ("accent2", "ED7D31"), ("accent3", "A5A5A5"),
        ("accent4", "FFC000"), ("accent5", "5B9BD5"), ("accent6", "70AD47"), ("hlink", "0563C1"), ("folHlink", "954F72")))
    fonts = '<a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/>'
    theme = (f'<a:theme {NS_A} name="Office Theme"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:sysClr val='
             f'"windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>{colours}'
             f'</a:clrScheme><a:fontScheme name="Office"><a:majorFont>{fonts}</a:majorFont><a:minorFont>{fonts}</a:minorFont>'
             f'</a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst>{fill * 3}</a:fillStyleLst><a:lnStyleLst>{line * 3}'
             f'</a:lnStyleLst><a:effectStyleLst>{"<a:effectStyle><a:effectLst/></a:effectStyle>" * 3}</a:effectStyleLst>'
             f'<a:bgFillStyleLst>{fill * 3}</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>')
    slide = ('<p:sp><p:nvSpPr><p:cNvPr id="2" name="Text"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off '
             'x="457200" y="457200"/><a:ext cx="8229600" cy="1143000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
             '</p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>Ship 2.1</a:t></a:r></a:p><a:p>'
             '<a:r><a:rPr lang="en-US"/><a:t>Friday</a:t></a:r></a:p></p:txBody></p:sp>')
    pres = OFFICE + "presentationml."
    return ooxml("ppt/presentation.xml", {
        "ppt/presentation.xml": (pres + "presentation.main+xml",
                                 f'<p:presentation {ns}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/>'
                                 f'</p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst><p:sldSz '
                                 f'cx="9144000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'),
        "ppt/slideMasters/slideMaster1.xml": (pres + "slideMaster+xml",
                                              f'<p:sldMaster {ns}><p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/>'
                                              f'</p:bgRef></p:bg>{tree}</p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" '
                                              f'bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" '
                                              f'accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" '
                                              f'folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" '
                                              f'r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/>'
                                              f'<p:otherStyle/></p:txStyles></p:sldMaster>'),
        "ppt/slideLayouts/slideLayout1.xml": (pres + "slideLayout+xml",
                                              f'<p:sldLayout {ns} type="blank" preserve="1"><p:cSld name="Blank">{tree}'
                                              f'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>'
                                              f'</p:sldLayout>'),
        "ppt/slides/slide1.xml": (pres + "slide+xml",
                                  f'<p:sld {ns}><p:cSld>{tree}{slide}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/>'
                                  f'</p:clrMapOvr></p:sld>'),
        "ppt/theme/theme1.xml": (OFFICE + "theme+xml", theme)}, {
        "ppt/presentation.xml": [(f"{REL}/slideMaster", "slideMasters/slideMaster1.xml"), (f"{REL}/slide", "slides/slide1.xml"),
                                 (f"{REL}/theme", "theme/theme1.xml")],
        "ppt/slideMasters/slideMaster1.xml": [(f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                                              (f"{REL}/theme", "../theme/theme1.xml")],
        "ppt/slideLayouts/slideLayout1.xml": [(f"{REL}/slideMaster", "../slideMasters/slideMaster1.xml")],
        "ppt/slides/slide1.xml": [(f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml")]})


ODF_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:'
          'text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/"')
ODT_BODY = "<text:h text:outline-level=\"1\">Appointments</text:h><text:p>Dentist moved to Thursday at half past nine</text:p>"
ODS_BODY = ('<table:table table:name="Budget"><table:table-column/><table:table-row><table:table-cell office:value-type="string">'
            '<text:p>Gate code</text:p></table:table-cell><table:table-cell office:value-type="float" office:value="4821">'
            '<text:p>4821</text:p></table:table-cell></table:table-row></table:table>')


def odf_members(kind: str, body: str) -> dict[str, str]:
    """An OpenDocument file (`kind` is "text" or "spreadsheet"): the stored `mimetype` first, `content.xml` with `body`
    inside `office:<kind>`, `styles.xml` and `meta.xml`, and a manifest that lists `/` (with the media type) and every
    member but the `mimetype` and itself, as LibreOffice writes it."""
    media = ODF + kind

    def document(root: str, inner: str = "") -> str:
        return XML + f'<office:{root} {ODF_NS} office:version="1.2">{inner}</office:{root}>'

    entry = lambda path, what: f'<manifest:file-entry manifest:full-path="{path}" manifest:media-type="{what}"/>'  # noqa: E731
    return {
        "mimetype": media,
        "content.xml": document("document-content", f"<office:body><office:{kind}>{body}</office:{kind}></office:body>"),
        "styles.xml": document("document-styles"),
        "meta.xml": document("document-meta", "<office:meta><dc:title>corpus</dc:title></office:meta>"),
        "META-INF/manifest.xml": XML + '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" '
                                       'manifest:version="1.2"><manifest:file-entry manifest:full-path="/" manifest:version='
                                       f'"1.2" manifest:media-type="{media}"/>{entry("content.xml", "text/xml")}'
                                       f'{entry("styles.xml", "text/xml")}{entry("meta.xml", "text/xml")}</manifest:manifest>'}


def epub_members() -> dict[str, str]:
    """An EPUB 3 of one chapter: the stored `mimetype` first, the container, the package document (an identifier the
    package names, a title, a language, the time it was modified, a manifest with the chapter and the navigation document,
    and a spine of the chapter) and the two XHTML documents."""
    page = XML + '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>{}</title></head>' \
                 '<body>{}</body></html>'
    return {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": XML + '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                                        '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type='
                                        '"application/oebps-package+xml"/></rootfiles></container>',
        "OEBPS/content.opf": XML + '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">'
                                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="bookid">'
                                   'urn:uuid:5d1c9a3e-0a4b-4f5e-8d62-3f0c2b7a9e11</dc:identifier><dc:title>The Lighthouse'
                                   '</dc:title><dc:language>en</dc:language><meta property="dcterms:modified">'
                                   '2026-10-31T08:00:00Z</meta></metadata><manifest><item id="nav" href="nav.xhtml" '
                                   'media-type="application/xhtml+xml" properties="nav"/><item id="c1" href="chapter1.xhtml" '
                                   'media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/></spine></package>',
        "OEBPS/nav.xhtml": page.format("Contents", '<nav epub:type="toc"><ol><li><a href="chapter1.xhtml">Chapter 1</a></li>'
                                                   '</ol></nav>'),
        "OEBPS/chapter1.xhtml": page.format("Chapter 1", "<h1>Chapter 1</h1><p>The lighthouse keeper climbed the stairs at "
                                                         "dusk.</p>")}


class Builder:
    def __init__(self, folder: Path, parts: tuple = ("pictures", "sound", "video", "text", "other", "documents", "bad")):
        self.folder = folder
        self.work = folder / "_work"
        self.entries: dict[str, Entry] = {}
        self.tasks: list = []
        for part in parts:
            (folder / part).mkdir(parents=True)
        self.work.mkdir()
        self.beep = self.work / "beep.wav"
        if "sound" in parts or "video" in parts:
            beep_wav(self.beep)

    def add(self, rel: str, media: str, read: str | None, **signal) -> Path:
        path = self.folder / rel
        kind = ("video" if media.startswith("video/") else "audio" if media.startswith("audio/")
                else "image" if is_picture(media) else "file")
        self.entries[rel] = Entry(path.name, path, media, kind, read, signal)
        return path

    def task(self, function, *args, **kwargs) -> None:
        self.tasks.append((function, args, kwargs))

    def write(self, rel: str, data: bytes, media: str, read: str | None, **signal) -> None:
        self.add(rel, media, read, **signal).write_bytes(data)

    def run(self) -> None:
        with ThreadPoolExecutor(2) as pool:
            for future in [pool.submit(function, *args, **kwargs) for function, args, kwargs in self.tasks]:
                future.result()
        for rel, entry in self.entries.items():
            if not entry.path.is_file():
                raise RuntimeError(f"the corpus did not make {rel}")
        shutil.rmtree(self.work)

    # pictures

    def pictures(self) -> None:
        upright = card()
        P = lambda name, media, **signal: self.add(f"pictures/{name}", media, "image", **signal)  # noqa: E731
        turned = {"IMG_2041.jpg": (6, Image.Transpose.ROTATE_90), "IMG_2042.jpg": (8, Image.Transpose.ROTATE_270),
                  "IMG_2043.jpg": (3, Image.Transpose.ROTATE_180), "selfie.jpg": (2, Image.Transpose.FLIP_LEFT_RIGHT)}
        for name, (orientation, how) in turned.items():  # a phone's JPEG is stored as the sensor saw it, with a tag
            exif = Image.Exif()
            exif[0x0112] = orientation
            upright.transpose(how).save(P(name, "image/jpeg", corner="red", size=CARD, exif=orientation), quality=90,
                                        exif=exif)
        upright.save(P("photo.webp", "image/webp", corner="red", size=CARD), quality=90)
        upright.save(P("photo.avif", "image/avif", corner="red", size=CARD), quality=80)
        upright.save(P("progressive.jpg", "image/jpeg", corner="red", size=CARD), quality=90, progressive=True)
        upright.convert("CMYK").save(P("cmyk.jpg", "image/jpeg", corner="red", size=CARD), quality=90)
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        upright.save(P("icc.jpg", "image/jpeg", corner="red", size=CARD), quality=90, icc_profile=profile)
        upright.save(P("paint.bmp", "image/bmp", corner="red", size=CARD))
        icon = card(64, 64)
        icon.save(P("favicon.ico", "image/vnd.microsoft.icon", corner="red", size=(64, 64)), sizes=[(64, 64)])
        upright.save(P("scan.tif", "image/tiff", corner="red", size=CARD), compression="tiff_lzw")
        upright.save(P("still.gif", "image/gif", corner="red", size=CARD))
        upright.save(self.work / "card.png")
        self.add("pictures/rgb48.png", "image/png", "image", corner="red", size=CARD)
        self.task(run, "-i", self.work / "card.png", *OUT, "-pix_fmt", "rgb48be", self.folder / "pictures/rgb48.png")
        card(1920, 1080).convert("RGBA").save(P("screenshot.png", "image/png", corner="red", size=(1920, 1080)))
        card(4000, 500).save(P("banner.png", "image/png", corner="red", size=(4000, 500)))
        Image.new("RGB", (1, 1), "red").save(P("pixel.png", "image/png", size=(1, 1)))
        words("LOGO").save(P("logo.png", "image/png", ink=True, alpha=True, size=CARD))
        glyph = Image.new("P", (160, 90), 0)  # a white glyph on a palette with one transparent entry
        glyph.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
        ImageDraw.Draw(glyph).ellipse((20, 5, 140, 85), fill=1)
        glyph.save(P("icon_white.png", "image/png", ink=True, alpha=True, size=CARD), transparency=0)
        sticker = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(sticker).ellipse((40, 40, 472, 472), fill=(30, 30, 30, 255))
        ImageDraw.Draw(sticker).ellipse((160, 160, 352, 352), fill=(255, 255, 255, 255))
        sticker.save(P("sticker.webp", "image/webp", ink=True, alpha=True, size=(512, 512)), lossless=True)
        words("LA", mode="LA").save(P("la.png", "image/png", ink=True, alpha=True, size=CARD))
        Image.fromarray(ink_scan(6000, 56000, np.uint16)).save(P("scan16.png", "image/png", ink=True, size=CARD))
        Image.fromarray(ink_scan(0.1, 0.9, np.float32)).save(P("float.tif", "image/tiff", ink=True, size=CARD))
        depth = ink_scan(0.1, 0.9, np.float32)
        depth[0, 0] = np.nan  # a depth map's hole
        Image.fromarray(depth).save(P("depth.tif", "image/tiff", ink=True, size=CARD))
        fax = words("FAX", background=(255, 255, 255, 255)).convert("L").convert("1", dither=Image.Dither.NONE)
        fax.save(P("fax.tif", "image/tiff", ink=True, size=CARD), compression="group4")
        buffer = io.BytesIO()  # a JPEG with a gain map: the picture, then a greyscale second image
        upright.save(buffer, "MPO", save_all=True, append_images=[Image.new("L", (80, 45), 128)])
        P("IMG_HDR.jpg", "image/jpeg", corner="red", size=CARD, frames=[0.0]).write_bytes(buffer.getvalue())
        white = Image.new("RGB", CARD, "white")
        for name, media in (("reaction.gif", "image/gif"), ("anim.webp", "image/webp"), ("anim.png", "image/png")):
            white.save(P(name, media, corner="red", size=CARD, frames=[0.0, 1.2]), save_all=True,
                       append_images=[white, upright], duration=[500, 700, 900], loop=0)
        pages = P("pages.tif", "image/tiff", corner="red", size=CARD, frames=[0.0, 1.0, 2.0])
        blank = [Image.new("RGB", CARD, "white") for _ in range(2)]  # fresh: the saves above leave GIF options on their images
        blank[0].save(pages, save_all=True, append_images=[blank[1], card()], compression="tiff_lzw")
        self.write("pictures/diagram.svg", '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="90">'
                   '<text x="8" y="40">café 4821</text></svg>\n'.encode(), "image/svg+xml", "text", contains=WORDS)

    # sound

    def sounds(self) -> None:
        beep = self.beep
        aac = ["-c:a", "aac", "-b:a", "64k"]
        amr = ["-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1"]
        jobs = {
            "voice_memo.m4a": ("audio/mp4", [*aac, "-ac", "1"]),
            "alac.m4a": ("audio/mp4", ["-c:a", "alac", "-ar", "16000"]),
            "PTT-20261031-WA0003.opus": ("audio/ogg", ["-c:a", "libopus", "-ar", "16000", "-ac", "1", "-b:a", "16k"]),
            "tg_voice.ogg": ("audio/ogg", ["-c:a", "libopus", "-b:a", "24k"]),
            "vorbis.ogg": ("audio/ogg", ["-c:a", "libvorbis", "-q:a", "3"]),
            "call.amr": ("audio/amr", amr),
            "call.3gp": ("audio/3gpp", amr),
            "audio_only.mp4": ("audio/mp4", aac),
            "podcast.mp3": ("audio/mpeg", ["-c:a", "libmp3lame", "-q:a", "4", "-ac", "2", "-ar", "44100"]),
            "lossless.flac": ("audio/flac", ["-c:a", "flac", "-ar", "16000"]),
            "phone_8k.wav": ("audio/wav", ["-ar", "8000"]),
            "mulaw.wav": ("audio/wav", ["-c:a", "pcm_mulaw", "-ar", "8000"]),
            "studio.wav": ("audio/wav", ["-c:a", "pcm_s24le", "-ac", "2", "-ar", "22050"]),
            "float.wav": ("audio/wav", ["-c:a", "pcm_f32le", "-ar", "16000"]),
            "surround.m4a": ("audio/mp4", [*aac, "-ac", "6"]),
            "voice.wma": ("audio/x-ms-wma", ["-c:a", "wmav2"]),
            "mac.aiff": ("audio/aiff", ["-c:a", "pcm_s16be", "-ar", "16000"]),
            "audio_only.avi": ("audio/x-msvideo", ["-c:a", "pcm_s16le", "-ar", "16000"]),
            "audio_only.flv": ("audio/x-flv", ["-c:a", "libmp3lame", "-ar", "44100", "-ac", "1"]),
        }
        for name, (media, args) in jobs.items():
            self.task(run, "-i", beep, *OUT, *args, self.add(f"sound/{name}", media, "audio", beep_s=BEEP_AT))
        recording = self.add("sound/recording.webm", "audio/webm", "audio", beep_s=BEEP_AT)
        self.task(pipe, "-i", beep, *OUT, "-c:a", "libopus", "-b:a", "24k", "-f", "webm", to=recording)
        self.task(self.cover_art, beep, self.add("sound/song.mp3", "audio/mpeg", "audio", beep_s=BEEP_AT))
        self.task(self.rate_change, beep, self.add("sound/ratechange.mp3", "audio/mpeg", "audio", beep_s=BEEP_AT))

    def cover_art(self, beep: Path, target: Path) -> None:
        cover = self.work / "cover.png"
        Image.new("RGB", (32, 32), "red").save(cover)
        run("-i", beep, "-i", cover, *OUT, "-map", "0:a", "-map", "1", "-c:a", "libmp3lame", "-q:a", "4", "-c:v", "png",
            "-disposition:v", "attached_pic", "-id3v2_version", "3", target)

    def rate_change(self, beep: Path, target: Path) -> None:
        """Two MP3s joined by a tool that copies packets: 44.1 kHz for 6 s, then 22.05 kHz."""
        first, second, names = self.work / "rate1.mp3", self.work / "rate2.mp3", self.work / "rate.txt"
        mp3 = ["-c:a", "libmp3lame", "-q:a", "4", "-write_xing", "0", "-ac", "1"]
        run("-i", beep, *OUT, "-t", BEEP_AT, *mp3, "-ar", "44100", first)
        run("-ss", BEEP_AT, "-i", beep, *OUT, *mp3, "-ar", "22050", second)
        names.write_text(f"file '{first.name}'\nfile '{second.name}'\n")
        run("-f", "concat", "-safe", "0", "-i", names, *OUT, "-c", "copy", target)

    # video

    def videos(self) -> None:
        beep = self.beep
        upright = card()
        portrait = card(90, 160)
        odd = card(161, 91)
        V = lambda name, media, **signal: self.add(f"video/{name}", media, "video", **signal)  # noqa: E731
        for name, degrees, shown, how in (("rot90.mp4", 90, portrait, -90), ("rot270.mp4", -90, portrait, 90),
                                          ("rot180.mp4", 180, upright, 180)):
            self.task(self.turned, V(name, "video/mp4", corner="red", size=shown.size), shown, how, degrees, X264)
        self.task(self.turned, V("IMG_0042.MOV", "video/quicktime", corner="red", size=portrait.size), portrait, -90, 90,
                  ["-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-x265-params",
                   "log-level=error:frame-threads=1:pools=1"])
        white, red = self.work / "white.png", self.work / "red.png"
        Image.new("RGB", CARD, "white").save(white)
        Image.new("RGB", CARD, "red").save(red)
        self.task(self.screen, white, red, V("screen_vfr.mkv", "video/x-matroska", colour="red", size=CARD, at=20.0))
        picture = self.work / "picture.png"
        upright.save(picture)
        odd.save(self.work / "odd.png")
        still = ["-loop", "1", "-framerate", "5", "-i", picture, "-t", "2"]
        for name, media, size, input_, codec in (
                ("av1.webm", "video/webm", CARD, still, ["-c:v", "libaom-av1", "-cpu-used", "8", "-crf", "32", "-b:v", "0"]),
                ("vp9_444.webm", "video/webm", odd.size, ["-loop", "1", "-framerate", "5", "-i", self.work / "odd.png",
                                                          "-t", "2"],
                 ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv444p", "-deadline", "good", "-cpu-used", "8", "-crf", "32",
                  "-b:v", "0"]),
                ("webcam_yuvj.mp4", "video/mp4", CARD, still, ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt",
                                                                "yuvj420p"]),
                ("webcam.avi", "video/x-msvideo", CARD, still, ["-c:v", "mjpeg", "-q:v", "3", "-pix_fmt", "yuvj420p"]),
                ("dashcam.ts", "video/mp2t", CARD, still, [*X264, "-f", "mpegts"])):
            self.task(run, *input_, *OUT, *codec, V(name, media, corner="red", size=size))
        long = ["-loop", "1", "-framerate", "5", "-i", picture, "-i", beep, "-t", LENGTH_S]
        self.task(run, *long, *OUT, "-c:v", "wmv2", "-b:v", "1M", "-c:a", "wmav2",
                  V("clip.wmv", "video/x-ms-wmv", corner="red", size=CARD, beep_s=BEEP_AT, sound=True))
        self.task(pipe, *long, *OUT, "-c:v", "libvpx", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "300k",
                  "-c:a", "libopus", "-b:a", "24k", "-f", "webm",
                  to=V("mediarec.webm", "video/webm", corner="red", size=CARD, beep_s=BEEP_AT, sound=True))
        self.task(run, "-f", "lavfi", "-i", f"color=c=gray:s=160x90:r=1:d={LENGTH_S}", "-i", beep, *OUT, *X264,
                  "-tune", "stillimage", "-c:a", "aac", "-b:a", "64k", "-shortest",
                  V("meeting.mp4", "video/mp4", colour="grey", size=CARD, beep_s=BEEP_AT, sound=True))
        tone = f"sine=frequency=1000:duration={TONE_S}"  # the sound ends long before the picture does
        self.task(run, "-f", "lavfi", "-i", f"color=c=gray:s=160x90:r=1:d={LONG_S}", "-f", "lavfi", "-i", tone, *OUT,
                  *X264, "-c:a", "aac", "-b:a", "64k",
                  V("short_sound.mp4", "video/mp4", colour="grey", size=CARD, sound=True, tone_s=TONE_S))
        self.task(self.size_change, V("sizechange.webm", "video/webm", sizes=[CARD, (320, 180)]))

    def turned(self, target: Path, shown: Image.Image, how: int, degrees: int, codec: list) -> None:
        """A phone video: the pictures are stored turned (`how` degrees) and a display matrix turns them back. Checked
        against ffmpeg's own autorotated decode."""
        coded, first = self.work / f"{target.name}.coded.png", self.work / f"{target.name}.first.png"
        stored = self.work / f"stored-{target.name}"
        shown.rotate(how, expand=True).save(coded)
        run("-loop", "1", "-framerate", "5", "-i", coded, "-t", "2", *OUT, *codec, stored)
        run("-display_rotation", degrees, "-i", stored, *OUT, "-metadata", "creation_time=2026-10-31T08:30:00Z", "-c",
            "copy", target)  # the time is there as a phone writes it
        run("-i", target, "-frames:v", "1", first)
        with Image.open(first) as seen:
            if seen.size != shown.size or corners(seen) != corners(shown):
                raise RuntimeError(f"{target.name}: ffmpeg autorotate gives {seen.size} {corners(seen)}, not "
                                   f"{shown.size} {corners(shown)}")

    def screen(self, white: Path, red: Path, target: Path) -> None:
        """A screen recording like mms's own: variable rate, 30 s white with the red panel from 20.0 to 20.5 s."""
        names = self.work / "screen.ffconcat"
        names.write_text(f"ffconcat version 1.0\nfile {white.name}\nduration 20\nfile {red.name}\nduration 0.5\n"
                         f"file {white.name}\nduration 9.5\nfile {white.name}\n")
        run("-f", "concat", "-safe", "0", "-i", names, *OUT, "-vf", "mpdecimate", "-fps_mode", "vfr", *X264, target)

    def size_change(self, target: Path) -> None:
        """A call recording whose size changes mid-stream, and (in `bad/`) the first half of a WebM."""
        parts = []
        for size in (CARD, (320, 180)):
            picture, part = self.work / f"size{size[0]}.png", self.work / f"size{size[0]}.webm"
            card(*size).save(picture)
            run("-loop", "1", "-framerate", "5", "-i", picture, "-t", "1", *OUT, "-c:v", "libvpx-vp9", "-deadline",
                "good", "-cpu-used", "8", "-crf", "32", "-b:v", "0", part)
            parts.append(part)
        names = self.work / "size.txt"
        names.write_text("".join(f"file '{part.name}'\n" for part in parts))
        run("-f", "concat", "-safe", "0", "-i", names, *OUT, "-c", "copy", target)
        data = parts[0].read_bytes()
        self.entries["bad/cut.webm"].path.write_bytes(data[:len(data) // 2])

    # text and other files

    def texts(self) -> None:
        T = lambda name, data, **signal: self.write(f"text/{name}", data, "text/plain", "text",  # noqa: E731
                                                    **{"contains": WORDS, **signal})
        crlf = BODY.replace("\n", "\r\n")
        bom16 = b"\xff\xfe" + crlf.encode("utf-16-le")  # PowerShell 5.1 `>` and `Out-File`
        T("notes.md", BODY.encode())
        T("excel_utf8.csv", "name,code\r\ncafé,4821\r\n".encode("utf-8-sig"))
        T("powershell.txt", bom16)
        T("be.txt", b"\xfe\xff" + crlf.encode("utf-16-be"))
        T("nobom16.txt", crlf.encode("utf-16-le"))
        T("export.reg", b"\xff\xfe" + ('Windows Registry Editor Version 5.00\r\n\r\n[HKEY_CURRENT_USER\\Software\\Cafe]\r\n'
                                       '"Gate"="café 4821"\r\n').encode("utf-16-le"))
        T("legacy.csv", "name,code\r\ncafé,4821\r\n".encode("cp1252"))
        for name in ("README", "Dockerfile", ".env"):
            T(name, BODY.encode())
        T("app.ts", "const city: string = 'café'; // gate 4821\n".encode())
        T("talk.srt", "1\n00:00:01,000 --> 00:00:03,000\nThe café gate code is 4821\n".encode())
        T("crash.log", (BODY + "\0" * 4096 + "after the crash: café 4821 recovered\nlast line\n").encode(),
          after="after the crash")
        T("oldmac.txt", BODY.replace("\n", "\r").encode(), lines=3)
        T("events.jsonl", "".join(json.dumps({"event": "café", "code": 4821}, ensure_ascii=False) + "\n"
                                  for _ in range(3)).encode())
        T("notebook.ipynb", json.dumps({"cells": [{"cell_type": "markdown", "source": ["café 4821"]}]},
                                       ensure_ascii=False).encode())
        T("page.html", "<html><body><p>café 4821</p></body></html>\n".encode())
        T("script.ps1", b"\xef\xbb\xbf" + 'Write-Output "café 4821"\r\n'.encode())
        # text whose bytes 4-7 spell a box name, or that starts like a container
        T("free.txt", "I'm free on Thursday at the café, gate 4821.\n".encode())
        T("wide.txt", "The wide shot of the café, gate 4821, is ready.\n".encode())
        T("free.js", "var free=1; // café 4821\n".encode())
        T("caffeine.txt", "caffeine intake: 2 cups at the café, 4821\n".encode())
        row = b"G" + "ate code 4821 at the café".encode().ljust(186) + b"\n"  # 188 bytes, as an MPEG-TS packet is
        T("rows.txt", row * 6)  # fixed-width rows, each starting with MPEG-TS's sync byte
        line = json.dumps({"note": "café 4821", "pad": ""}, ensure_ascii=False)
        T("min.json", (line[:-2] + "x" * (50_000 - len(line)) + line[-2:]).encode(), chars=50_000)

    def others(self) -> None:
        pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n"
        data = random.Random(7).randbytes(4096)
        if b"\0" not in data:
            raise RuntimeError("data.bin must hold a NUL, so that no text rule takes it for text")
        card().save(self.add("other/notes.txt", "image/png", "image", corner="red", size=CARD), format="PNG")
        self.write("other/garbage.mp3", BODY.encode(), "text/plain", "text", contains=WORDS)
        self.write("other/data.bin", data, "application/octet-stream", None)
        self.write("other/archive.json.gz", gzip.compress(BODY.encode(), mtime=0), "application/gzip", None)
        self.write("other/report.pdf", pdf, "application/pdf", None, stub=True)
        office = "application/vnd.openxmlformats-officedocument."
        for name, media, members in (
                ("book.xlsx", office + "spreadsheetml.sheet", ["[Content_Types].xml", "xl/workbook.xml"]),
                ("letter.docx", office + "wordprocessingml.document", ["[Content_Types].xml", "word/document.xml"]),
                ("plain.zip", "application/zip", ["notes.txt"])):
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as zipped:
                for member in members:
                    zipped.writestr(zipfile.ZipInfo(member, (2026, 10, 31, 8, 0, 0)), "<x>café 4821</x>")
            self.write(f"other/{name}", archive.getvalue(), media, None, stub=name != "plain.zip")  # an Office file of one XML part
        header = bytearray(1024)  # a DOS header whose e_lfanew (at 0x3c) points at the PE signature
        header[:2], header[0x3C:0x40], header[0x40:0x44] = b"MZ", (0x40).to_bytes(4, "little"), b"PE\0\0"
        self.write("other/app.exe", bytes(header), "application/vnd.microsoft.portable-executable", None, stub=True)
        self.write("other/places.sqlite", b"SQLite format 3\0" + bytes(4080), "application/vnd.sqlite3", None, stub=True)

    def documents(self) -> None:
        """What a home folder holds besides pictures, sound and notes. `signal["words"]` are the words a reader that takes
        the whole file finds in it (`check_documents` finds them again by following the file as a reader does, with the
        standard library); no decider reads any of these files today, so `read` is None, but for the plain-text ones (RTF,
        iCalendar, vCard, a mail), which are text. A `stub` is only as much of its type as mms sniffs, and holds no words."""
        def D(name: str, media: str, words=(), *, data: bytes, read: str | None = None, **signal) -> None:
            self.write(f"documents/{name}", data, media, read, **({"words": list(words)} if words else {}), **signal)

        invoice = text_pdf(["Invoice 2026-118", "Total 342.50 EUR"])
        D("invoice.pdf", "application/pdf", ["invoice", "342.50"], data=invoice)
        D("invoice-copy.txt", "application/pdf", ["invoice", "342.50"], data=invoice)  # the same bytes under a text name
        card_scan = io.BytesIO()
        words("4821", background=(255, 255, 255, 255), mode="RGB").save(
            card_scan, format="PDF", creationDate=PDF_TIME.timetuple(), modDate=PDF_TIME.timetuple())  # or it is dated now
        D("scan.pdf", "application/pdf", data=card_scan.getvalue(), pages=1)  # a picture of "4821": no text layer
        D("locked.pdf", "application/pdf", data=locked_pdf(), encrypted=True)
        D("minutes.docx", OFFICE + "wordprocessingml.document", ["retention", "30"], data=zipped(docx_members()))
        D("budget.xlsx", OFFICE + "spreadsheetml.sheet", ["café", "4821"], data=zipped(xlsx_members()))
        D("deck.pptx", OFFICE + "presentationml.presentation", ["ship", "friday"], data=zipped(pptx_members()))
        D("notes.odt", ODF + "text", ["dentist"], data=zipped(odf_members("text", ODT_BODY)))
        D("sheet.ods", ODF + "spreadsheet", ["4821"], data=zipped(odf_members("spreadsheet", ODS_BODY)))
        D("book.epub", "application/epub+zip", ["lighthouse"], data=zipped(epub_members()))
        for name, label in (("legacy.doc", b"WordDocument"), ("meeting.msg", b"__substg1.0_0037001F")):
            D(name, "application/x-ole-storage", stub=True, data=bytes.fromhex("D0CF11E0A1B11AE1") + label.ljust(504, b"\0"))
        D("letter.rtf", "text/plain", read="text", contains=["4821"], data=(
            r"{\rtf1\ansi\deff0{\fonttbl{\f0 Arial;}}\f0\fs24 Gate code 4821 at the caf\'e9 on Rue Cler.\par "
            r"The second line is plain.\par}").encode("ascii") + b"\n")
        D("dentist.ics", "text/plain", read="text", contains=["Dentist"], data=(
            "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//MultimediaSearch//corpus//EN\r\nBEGIN:VEVENT\r\nUID:dentist-1@example.test\r\n"
            "DTSTAMP:20261031T080000Z\r\nDTSTART:20261105T093000Z\r\nDTEND:20261105T103000Z\r\nSUMMARY:Dentist\r\n"
            "END:VEVENT\r\nEND:VCALENDAR\r\n").encode())
        D("sam.vcf", "text/plain", read="text", contains=WORDS, data=(
            "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Sam Rivera\r\nN:Rivera;Sam;;;\r\nTEL;TYPE=CELL:+33 6 12 34 56 78\r\n"
            "NOTE:Gate code 4821 at the café on Rue Cler.\r\nEND:VCARD\r\n").encode())
        D("gate.eml", "text/plain", read="text", contains=["4821"], data=mail())
        D("song.mid", "application/octet-stream", data=b"MThd\0\0\0\6\0\0\0\1\0\x60" + b"MTrk\0\0\0\x0c"
          + bytes.fromhex("00903c40" "60803c00" "00ff2f00"))
        tarred = archive({"server.log": b"".join(b"2026-10-31 12:00:%02d INFO request served\n" % n for n in range(50))})
        D("backup.tar", "application/octet-stream", data=tarred)
        D("logs.tar.bz2", "application/octet-stream", data=bz2.compress(tarred, 9))
        D("dump.xz", "application/octet-stream", data=lzma.compress(BODY.encode(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC32))
        database = self.folder / "documents" / "history.sqlite"
        self.add("documents/history.sqlite", "application/vnd.sqlite3", None, tables=["visits"])
        with sqlite3.connect(database) as db:
            db.execute("CREATE TABLE visits (url TEXT, title TEXT, visited TEXT)")
            db.executemany("INSERT INTO visits VALUES (?, ?, ?)",
                           [("https://example.test/gate", "The gate", "2026-10-31T08:05:00Z"),
                            ("https://example.test/menu", "Café menu", "2026-10-31T12:30:00Z")])
        db.close()  # `with` commits and does not close
        D("photos.7z", "application/x-7z-compressed", stub=True, data=b"7z\xbc\xaf\x27\x1c\0\4" + bytes(40))  # zero CRCs

    def broken(self) -> None:
        noise = np.random.default_rng(1).integers(0, 256, (90, 160, 3), dtype=np.uint8)
        jpeg = io.BytesIO()
        Image.fromarray(noise).save(jpeg, "JPEG", quality=90)
        self.write("bad/truncated.jpg", jpeg.getvalue()[:len(jpeg.getvalue()) // 2], "image/jpeg", "image",
                   warn="OSError: image file is truncated")
        # no HEIC can be made offline here, so this is the smallest file that sniffs as one: an ftyp box of brand heic
        self.write("bad/IMG_0001.HEIC", b"\0\0\0\x18ftypheic\0\0\0\0mif1heic" + bytes(64), "image/heic", "image",
                   warn="pi-heif", stub=True)
        self.write("bad/empty.txt", b"", "text/plain", None)
        self.task(self.subtitles_only, self.add("bad/subs.mkv", "video/x-matroska", "video", warn="no video stream"))
        self.add("bad/cut.webm", "video/webm", "video")  # written with sizechange.webm, whose first part it halves
        self.task(self.no_moov, self.add("bad/no_moov.mp4", "video/mp4", "video", warn="unreadable"))

    def subtitles_only(self, target: Path) -> None:
        """An MKV that PyAV opens with neither a picture nor a sound."""
        subs = self.work / "subs.srt"
        subs.write_text("1\n00:00:01,000 --> 00:00:03,000\nThe café gate code is 4821\n", encoding="utf-8")
        run("-i", subs, *OUT, "-c:s", "srt", target)

    def no_moov(self, target: Path) -> None:
        """A phone video cut off before its index: an MP4 made without faststart ends with its `moov` box."""
        picture, whole = self.work / "whole.png", self.work / "whole.mp4"
        card().save(picture)
        run("-loop", "1", "-framerate", "5", "-i", picture, "-t", "2", *OUT, *X264, whole)
        data = whole.read_bytes()
        target.write_bytes(data[:data.rindex(b"moov") - 4])


def check(entries: dict[str, Entry]) -> None:
    """What the files hold, by tools other than mms's: Pillow's upright picture, the EXIF tag, transparency, the BOMs and
    where the beep is."""
    for rel, entry in entries.items():
        signal = entry.signal
        if entry.read == "image" and "warn" not in signal:
            with Image.open(entry.path) as opened:
                if "exif" in signal and opened.getexif().get(0x0112) != signal["exif"]:
                    raise RuntimeError(f"{rel} has no EXIF orientation {signal['exif']}")
                if len(signal.get("frames", ())) > 1:
                    opened.seek(opened.n_frames - 1)  # the card is the last frame
                seen = ImageOps.exif_transpose(opened).convert("RGB")
                transparent = opened.convert("RGBA").getchannel("A").getextrema()[0] == 0
            if seen.size != signal["size"] or ("corner" in signal and corners(seen)[0] != signal["corner"]):
                raise RuntimeError(f"{rel}: Pillow's upright picture is {seen.size} {corners(seen)}, not "
                                   f"{signal['size']} {signal.get('corner')}")
            if signal.get("alpha") and not transparent:
                raise RuntimeError(f"{rel} has no transparent pixel")
        if entry.stub and (signal.get("words") or signal.get("contains")):
            raise RuntimeError(f"{rel} is a stub and holds no words, yet it claims {signal.get('words') or signal.get('contains')}")
        if entry.name in ("powershell.txt", "export.reg") and not entry.path.read_bytes().startswith(b"\xff\xfe"):
            raise RuntimeError(f"{rel} has no UTF-16 BOM")
    with wave.open(str(entries["sound/phone_8k.wav"].path)) as file:
        sound = np.frombuffer(file.readframes(file.getnframes()), "<i2")
        if not BEEP_AT <= int(np.argmax(np.abs(sound))) / file.getframerate() < BEEP_AT + BEEP_S:
            raise RuntimeError("the beep is not at 6.0 s")


def zip_problems(archive: zipfile.ZipFile) -> list[str]:
    """What is wrong with a zip as a container: a bad CRC, or XML that is not well-formed."""
    if archive.testzip() is not None:
        return ["a broken zip"]
    problems = []
    for name in archive.namelist():
        if name.endswith((".xml", ".rels", ".opf", ".xhtml")):
            try:
                ET.fromstring(archive.read(name))
            except ET.ParseError as error:
                problems.append(f"{name} is not well-formed XML: {error}")
    return problems


def ooxml_problems(archive: zipfile.ZipFile, media: str) -> list[str]:
    """What keeps Word, Excel or PowerPoint (or a library) from following an Office package: a part with no content type,
    an override for no part, a relationship to nothing or with a repeated id, a part no relationship reaches from
    `_rels/.rels` (a reader never sees it), a relationships part of no part, or a main part of another content type."""
    names, problems = archive.namelist(), []
    parts = [name for name in names if name != "[Content_Types].xml" and not name.endswith(".rels")]
    for name in names:
        if name != "[Content_Types].xml" and doctext.content_type(archive, name) is None:
            problems.append(f"{name} has no content type")
    for element in doctext.part(archive, "[Content_Types].xml"):
        if doctext.local(element.tag) == "Override" and element.get("PartName", "")[1:] not in names:
            problems.append(f"the content type override for {element.get('PartName')} has no part")
    reached, queue = set(), [""]
    while queue:
        source = queue.pop()
        ids: set[str] = set()
        for link in doctext.relationships(archive, source):
            if link["Id"] in ids:
                problems.append(f"{source or 'the package'} has the relationship id {link['Id']} twice")
            ids.add(link["Id"])
            if link.get("TargetMode") == "External":
                continue
            if link["Target"] not in names:
                problems.append(f"{source or 'the package'} relates to {link['Target']}, which is not in the package")
            elif link["Target"] not in reached:
                reached.add(link["Target"])
                queue.append(link["Target"])
    problems += [f"{name} is related to by no part, so a reader never reaches it" for name in parts if name not in reached]
    for name in names:
        owner = posixpath.join(posixpath.dirname(posixpath.dirname(name)), posixpath.basename(name)[:-5])
        if name.endswith(".rels") and name != "_rels/.rels" and owner not in names:
            problems.append(f"{name} holds the relationships of {owner}, which is not there")
    mains = doctext.related(archive, "", "officeDocument")
    wanted = OFFICE + media.removeprefix(OFFICE) + ".main+xml"
    if len(mains) != 1 or doctext.content_type(archive, mains[0]) != wanted:
        problems.append(f"its office document is {mains}, not one part of the content type {wanted}")
    return problems


def odf_problems(archive: zipfile.ZipFile, media: str) -> list[str]:
    """What keeps LibreOffice from opening an OpenDocument file: a first member that is not the stored `mimetype`, a manifest
    that does not give `/` the media type or does not list every member (but the `mimetype` and the manifest itself), or a
    `content.xml` with no `office:version`."""
    first, problems = archive.infolist()[0], []
    if (first.filename, first.compress_type) != ("mimetype", zipfile.ZIP_STORED) or archive.read("mimetype").decode() != media:
        problems.append(f"its first member is {first.filename}, not a stored mimetype of {media}")
    manifest = doctext.part(archive, "META-INF/manifest.xml")
    listed = {entry.get(doctext.MANIFEST + "full-path"): entry.get(doctext.MANIFEST + "media-type") for entry in manifest}
    if listed.get("/") != media or manifest.get(doctext.MANIFEST + "version") is None:
        problems.append(f"the manifest does not give / the media type {media}, or has no manifest:version")
    names = archive.namelist()
    problems += [f"{name} is not in the manifest" for name in names
                 if name not in ("mimetype", "META-INF/manifest.xml") and name not in listed]
    problems += [f"the manifest lists {path}, which is not in the file" for path in listed if path != "/" and path not in names]
    if doctext.part(archive, "content.xml").get(doctext.OFFICE + "version") is None:
        problems.append("content.xml has no office:version")
    return problems


def epub_problems(archive: zipfile.ZipFile, media: str) -> list[str]:
    """What keeps a reader from opening an EPUB 3: a first member that is not the stored `mimetype`, a package document
    without the identifier it names, a title, a language or a modified time, a manifest with a file missing, an item the
    manifest lacks, no navigation document, or a spine item the manifest lacks."""
    first, names, problems = archive.infolist()[0], archive.namelist(), []
    if (first.filename, first.compress_type) != ("mimetype", zipfile.ZIP_STORED) or archive.read("mimetype").decode() != media:
        problems.append(f"its first member is {first.filename}, not a stored mimetype of {media}")
    rootfile = doctext.part(archive, "META-INF/container.xml").find(f".//{doctext.CONTAINER}rootfile")
    package = rootfile.get("full-path") if rootfile is not None else None
    if package not in names:
        return problems + [f"the container names {package}, which is not in the file"]
    opf = doctext.part(archive, package)
    named = opf.get("unique-identifier")
    for what, found in (("no identifier its unique-identifier names", [e for e in opf.iter(doctext.DC + "identifier") if e.get("id") == named]),
                        ("no title", list(opf.iter(doctext.DC + "title"))), ("no language", list(opf.iter(doctext.DC + "language")))):
        if not any((e.text or "").strip() for e in found):
            problems.append(f"the package document has {what}")
    modified = [e.text or "" for e in opf.iter(doctext.OPF + "meta") if e.get("property") == "dcterms:modified"]
    if not any(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", text) for text in modified):
        problems.append("the package document has no dcterms:modified of the form CCYY-MM-DDThh:mm:ssZ")
    items = {item.get("id"): item for item in opf.iter(doctext.OPF + "item")}
    members = {doctext.resolve(package, item.get("href")): item.get("id") for item in items.values()}
    problems += [f"the manifest lists {member}, which is not in the file" for member in members if member not in names]
    problems += [f"{name} is not in the manifest" for name in names
                 if name not in ("mimetype", "META-INF/container.xml", package) and name not in members]
    if not any("nav" in (item.get("properties") or "").split() for item in items.values()):
        problems.append("the manifest has no navigation document (properties=\"nav\")")
    spine = [itemref.get("idref") for itemref in opf.iter(doctext.OPF + "itemref")]
    problems += [f"the spine names {idref}, which the manifest does not" for idref in spine if idref not in items]
    if not spine:
        problems.append("the spine is empty")
    return problems


def check_package(name: str, media: str, data: bytes) -> None:
    """A zip document as a reader needs it: well-formed, with its parts reachable the way its type is followed (Office:
    relationships and content types; OpenDocument: the manifest; EPUB: the container, package document and spine)."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        problems = zip_problems(archive)
        if not problems:
            problems = (epub_problems if media == "application/epub+zip" else odf_problems if media.startswith(ODF)
                        else ooxml_problems)(archive, media)
    if problems:
        raise RuntimeError(f"{name}: " + "; ".join(problems))


def check_document(name: str, media: str, data: bytes, words=()) -> None:
    """One document: an Office, ODF or EPUB package is one a reader can follow, and each of `words` is in the text the
    file shows when it is followed (a part no path reaches shows nothing)."""
    if data.startswith(b"PK\x03\x04") and (media.startswith((OFFICE, ODF)) or media == "application/epub+zip"):
        check_package(name, media, data)
    if words:
        try:
            text = doctext.text_of(data)
        except doctext.Unreadable as error:
            raise RuntimeError(f"{name} cannot be followed to its words: {error}") from None
        if missing := [word for word in words if word.casefold() not in text.casefold()]:
            raise RuntimeError(f"{name} does not show {missing}: {text[:120]!r}")


def check_documents(entries: dict[str, Entry]) -> None:
    """What the documents hold, by tools other than mms's: the words of each in the text a reader finds by following the
    file (`check_document`, with the standard library), the page and the picture of the scan, the encryption of the locked
    file, a real database, and archives that open."""
    total = 0
    for rel, entry in entries.items():
        if not rel.startswith("documents/"):
            continue
        data = entry.path.read_bytes()
        total += len(data)
        check_document(entry.name, entry.media, data, entry.signal.get("words", ()))
        if (plain := entry.signal.get("contains")) and not entry.signal.get("words"):  # a plain-text document, by its characters
            if missing := [word for word in plain if word.casefold() not in data.decode("utf-8").casefold()]:
                raise RuntimeError(f"{rel} does not hold {missing}")
    documents = {Path(rel).name: entry for rel, entry in entries.items() if rel.startswith("documents/")}
    scan, locked = documents["scan.pdf"].path.read_bytes(), documents["locked.pdf"].path.read_bytes()
    if len(re.findall(rb"/Type /Page\b", scan)) != documents["scan.pdf"].signal["pages"] \
            or b"/Subtype /Image" not in scan or b"Tj" in scan:
        raise RuntimeError("scan.pdf is not one page holding a picture and no text")
    stream = re.findall(rb"stream\n(.*?)\nendstream", locked, re.S)[0]
    try:
        zlib.decompress(stream)
    except zlib.error:
        pass
    else:
        raise RuntimeError("locked.pdf's stream can be inflated")
    if b"/Encrypt" not in locked.rpartition(b"trailer")[2]:
        raise RuntimeError("locked.pdf's trailer has no /Encrypt")
    if documents["invoice-copy.txt"].path.read_bytes() != documents["invoice.pdf"].path.read_bytes():
        raise RuntimeError("invoice-copy.txt is not invoice.pdf's bytes")
    with sqlite3.connect(f"file:{documents['history.sqlite'].path.as_posix()}?mode=ro", uri=True) as db:
        if db.execute("SELECT COUNT(*) FROM visits").fetchone() != (2,):
            raise RuntimeError("history.sqlite does not hold its two visits")
    db.close()
    if tarfile.open(documents["backup.tar"].path).getnames() != ["server.log"] \
            or tarfile.open(documents["logs.tar.bz2"].path, "r:bz2").getnames() != ["server.log"] \
            or lzma.decompress(documents["dump.xz"].path.read_bytes()) != BODY.encode() \
            or not documents["song.mid"].path.read_bytes().startswith(b"MThd"):
        raise RuntimeError("an archive, the XZ file or the MIDI file does not open as what it is")
    if total > 200_000:
        raise RuntimeError(f"the documents are {total:,} bytes, over 200,000")


def build_documents(folder: Path) -> dict[str, Entry]:
    """Only the `documents/` part, made and checked as `build` does: a test makes it twice, to see the bytes are the same."""
    builder = Builder(folder, ("documents",))
    builder.documents()
    builder.run()
    check_documents(builder.entries)
    return builder.entries


def build(folder: Path) -> dict[str, Entry]:
    """Make the corpus in the empty `folder`; the entries, keyed by path under it."""
    builder = Builder(folder)
    builder.pictures()
    builder.sounds()
    builder.videos()
    builder.texts()
    builder.others()
    builder.documents()
    builder.broken()
    builder.run()
    check(builder.entries)
    check_documents(builder.entries)
    return builder.entries


# what the tests that use the corpus share

T0 = 1_699_999_980.0  # a multiple of 60 s; the n-th file of a test is remembered at T0 + n * GAP
GAP = 1000.0  # far apart, so that each file is a sighting of its own
EVERYTHING = Scope(all=True)


def remember_each(api, corpus: dict[str, Entry], rels, source: str = "agent") -> dict[str, float]:
    """Each file remembered at a time of its own, T0 + n * GAP; its name is the file's, so these are unique."""
    names = [corpus[rel].name for rel in rels]
    assert len(set(names)) == len(names), "two files of the corpus share a name"
    times = {}
    for n, rel in enumerate(rels):
        times[rel] = T0 + n * GAP
        api.call("remember", path=str(corpus[rel].path), t=times[rel], source=source)
    return times


def read_windows(store, about, newest_first: bool = False, profile: dict | None = None):
    """Every window by the name of its item, in the order read, and the items that could not be read, as `about` (a
    decider's facts) has them cut."""
    windows: dict[str, list] = {}
    unreadable: dict[str, str] = {}

    def failed(item, error):
        unreadable[item.name] = f"{type(error).__name__}: {error}"

    for window in reader.windows(store, about=about, scope=EVERYTHING, newest_first=newest_first, profile=profile,
                                 on_unreadable=failed):
        if window.item:
            windows.setdefault(window.item.name, []).append(window)
    return windows, unreadable


def scan(api, until: float, **params) -> dict:
    """Run the shipped `ask` as a job over [T0 - 100, until] and return its result once it is done."""
    params.setdefault("since", T0 - 100)
    params["until"] = until
    job = api.call("jobs/start", script="ask", params=params, wait_s=50)
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    assert job["state"] == "done", job
    return job["result"]


def names_hit(store, result) -> set[str]:
    return {store.item(s["look"]["item"]).name for s in result["sightings"]}


# the warning for items the decider does not read (jobs.py words it)

class Unread(NamedTuple):
    count: int  # items the warning counts, in all
    decider: str  # the name the warning gives the decider
    groups: dict[str, list[int]]  # each group (a media type, or "audio segments") with the item ids the warning names
    more: dict[str, int]  # the items of each group beyond those named ("and N more")
    tail: str  # what the agent is told to do: "find lists them and look gives a copy of each"


UNREAD = re.compile(r"(\d+) items? (was|were) not read, since (.+?) does not read (its|their) types?: (.+); (find lists .+)")
GROUP = re.compile(r"(.+?) \((items?) (\d+(?:, \d+)*)(?: and (\d+) more)?\)(?:, |\Z)")


def parse_unread(warning: str) -> Unread:
    """Parse a not-read warning, "N item(s) was/were not read, since NAME does not read its/their type(s): GROUP (item a, b),
    GROUP (items c, d and N more); TAIL", and fail on any part of it that is not so worded: the grammar of its numbers, a
    group's counts adding up to N, and at most 5 ids named for a group that has more."""
    found = UNREAD.fullmatch(warning)
    assert found, f"not a not-read warning: {warning!r}"
    count, verb, decider, pronoun, body, tail = found.groups()
    assert (verb, pronoun) == (("was", "its") if int(count) == 1 else ("were", "their")), f"the grammar of {count} items: {warning!r}"
    groups, more, end = {}, {}, 0
    for group in GROUP.finditer(body):
        assert group.start() == end, f"a group of the warning is not worded as one: {body[end:]!r}"
        end = group.end()
        name, noun, ids, rest = group.groups()
        groups[name], more[name] = [int(n) for n in ids.split(", ")], int(rest or 0)
        assert (noun == "item") == (len(groups[name]) + more[name] == 1), f"{name}: {noun} for {len(groups[name]) + more[name]}"
        assert not more[name] or len(groups[name]) == 5, f"{name} names {len(groups[name])} ids and then says 'and more'"
    assert groups and end == len(body), f"the groups of the warning do not end it: {body!r}"
    assert sum(map(len, groups.values())) + sum(more.values()) == int(count), f"the groups do not add up to {count}: {warning!r}"
    return Unread(int(count), decider, groups, more, tail)


def not_read_warnings(result: dict) -> list[str]:
    """The not-read warnings of an ask's result (there is at most one)."""
    return [warning for warning in result.get("warnings", []) if " not read, since " in warning]


def check_unread(warning: str, decider: str, expected: dict[str, set[int]]) -> Unread:
    """Parse the not-read `warning` and check that it is `decider`'s, that its groups are `expected`'s keys with the ids
    named in each among the item ids expected (all of them, but for a group that says "and N more", which counts the rest),
    and that it counts exactly the items expected."""
    got = parse_unread(warning)
    assert got.decider == decider, f"the warning names {got.decider}, not {decider}: {warning}"
    assert set(got.groups) == set(expected), f"the warning groups {sorted(got.groups)}, not {sorted(expected)}: {warning}"
    assert got.count == sum(map(len, expected.values())), f"the warning counts {got.count} items: {warning}"
    for group, ids in got.groups.items():
        assert set(ids) <= expected[group] and len(ids) + got.more[group] == len(expected[group]), \
            f"{group}: the warning names {ids} and {got.more[group]} more, not {sorted(expected[group])}: {warning}"
    return got
