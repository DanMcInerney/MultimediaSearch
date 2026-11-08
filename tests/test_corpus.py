"""Files as phones and programs write them (TESTBED.md item 15): the corpus of `tests/corpus.py` goes in the two ways a file
reaches mms, `remember` and a watched folder, and each test checks what mms recorded, what the reader hands d1 (the windows
over the same home) and what the fake decider is given through an `ask` job. The corpus says what each file must give."""

from __future__ import annotations

import json
import shutil
import textwrap
import time
import wave
from pathlib import Path

import doctext
import numpy as np
import pytest
from conftest import DECIDER, peak_memory, span, wait_for, write_config
from corpus import (BEEP_AT, BEEP_S, BODY, CARD, EVERYTHING, GAP, NS_W, ODF, ODS_BODY, ODT_BODY, OFFICE, REL, T0, TONE_S, Unread,
                    build_documents, check_document, check_unread, docx_members, epub_members, names_hit, not_read_warnings,
                    odf_members, ooxml, parse_unread, pptx_members, remember_each, scan, xlsx_members, zipped)
from corpus import read_windows as windows_of
from fake_decider import colour, corners, has_ink
from PIL import Image, ImageDraw

from mms import decider_d1, media, reader
from mms.store import Store

CONFIG = "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 50\n" + DECIDER
MB = 1_000_000


@pytest.fixture
def api(home, served):
    write_config(home, CONFIG)
    return served().api


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def folder_of(corpus, part: str) -> Path:
    return next(entry.path.parent for rel, entry in corpus.items() if rel.startswith(part + "/"))


def read_windows(store, newest_first: bool = False, profile: dict | None = None):
    """Every window by the name of its item, in the order read, and the items that could not be read (under d1's facts)."""
    return windows_of(store, decider_d1.ABOUT, newest_first, profile)


def decider_log(home: Path) -> list[dict]:
    file = home / "fake-decider.jsonl"
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()] if file.exists() else []


def beeps(windows) -> list[float]:
    """Seconds into the file of the loudest sample of each window that holds some of the beep."""
    found = []
    for w in windows:
        audio = np.abs(w.media["audio"])
        if audio.max() > 0.2:
            found.append(w.at + int(audio.argmax()) / reader.RATE)
    return found


def check_beep(name: str, windows, problems: list[str]) -> None:
    """Every window 80,000 samples of 16 kHz mono, and the beep in some of them at 6.0-6.4 s (60 ms either way)."""
    for w in windows:
        audio = w.media["audio"]
        if w.media["sampling_rate"] != 16000 or audio.ndim != 1 or len(audio) != 80_000:
            problems.append(f"{name}: a window of {len(audio)} samples at {w.media['sampling_rate']} Hz, {audio.ndim}-D")
    times = beeps(windows)
    if not times or not all(BEEP_AT - 0.06 <= t <= BEEP_AT + BEEP_S + 0.06 for t in times):
        problems.append(f"{name}: the beep (6.0-6.4 s) is heard at {[round(t, 3) for t in times]}")


# what is recorded

def test_each_file_is_recorded_as_what_its_content_is(home, served, corpus, tmp_path):
    good = [rel for rel in corpus if not rel.startswith("bad/")]
    copy = tmp_path / "copy"
    shutil.copytree(folder_of(corpus, "pictures").parent, copy, ignore=shutil.ignore_patterns("bad", "_work"))
    write_config(home, CONFIG + f'[sources.dir]\nkind = "folder"\nrescan_s = 0.2\npath = "{copy.as_posix()}"\n')
    api = served().api
    store = Store(home)
    for part in ("pictures", "sound", "video", "text", "other", "documents"):
        api.call("remember", path=str(folder_of(corpus, part)), t=T0)

    def versions():
        return {item.name: item for item in store.items_of("dir") if item.state == "ok"}

    wait_for(lambda: len(versions()) == len(good), timeout=60, what="the folder source's items",
             detail=lambda: f"{len(versions())} of {len(good)}")
    problems = []
    for rel in good:
        entry = corpus[rel]
        sound = entry.signal.get("sound", False)
        for how, item, kind in (("remember", next((i for i in store.items_of("agent") if i.name == rel), None), entry.kind),
                                ("folder", versions().get(rel), entry.kind)):
            if item is None:
                problems.append(f"{rel}: not stored by {how}")
            elif (item.kind, item.meta.get("media"), item.meta.get("sound", False)) != (kind, entry.media, sound):
                problems.append(f"{rel} by {how}: {(item.kind, item.meta.get('media'), item.meta.get('sound', False))}, "
                                f"not {(kind, entry.media, sound)}")
    assert not problems, "\n".join(problems)
    again = api.call("remember", path=str(corpus["video/IMG_0042.MOV"].path), source="phone")
    assert again["stored"][0]["time_from"] == "metadata", "the recording's own creation_time gives its time"
    store.close()


def test_what_is_not_text_or_media_is_kept_and_not_read(home, api, store, corpus):
    rels = [rel for rel, entry in corpus.items() if entry.read is None]
    assert {corpus[rel].name for rel in rels} >= {"data.bin", "report.pdf", "book.xlsx", "letter.docx", "plain.zip",
                                                  "app.exe", "places.sqlite", "archive.json.gz", "empty.txt"}
    remember_each(api, corpus, rels)
    items = {item.name: item for item in store.items_of("agent")}
    problems = []
    for rel in rels:
        entry = corpus[rel]
        item = items[entry.name]
        if (item.kind, item.meta.get("media")) != (entry.kind, entry.media):
            problems.append(f"{rel}: {(item.kind, item.meta.get('media'))}, not {(entry.kind, entry.media)}")
        found = [row["id"] for row in api.call("find", text=entry.name.split(".")[0])["results"]]
        if item.id not in found:
            problems.append(f"{rel} is not found by its name")
    windows, unreadable = read_windows(store)
    problems += [f"{name} is read: {len(ws)} windows" for name, ws in windows.items()]
    problems += [f"{name} is unreadable: {error}" for name, error in unreadable.items()]
    result = scan(api, T0 + GAP * len(rels), question="contains:café")
    assert not problems, "\n".join(problems)
    unread = [rel for rel in rels if corpus[rel].name != "empty.txt"]  # an empty text is read, and has no window
    (warning,) = not_read_warnings(result)
    assert result["warnings"] == [warning] and result["sightings"] == [], result
    expected: dict[str, set[int]] = {}
    for rel in unread:
        expected.setdefault(corpus[rel].media, set()).add(items[corpus[rel].name].id)
    check_unread(warning, "d1-Omni-2B", expected)


# the documents, and what the build checks of them

def test_a_stub_is_marked_and_holds_no_words(corpus):
    stubs = sorted(rel for rel, entry in corpus.items() if entry.stub)
    assert stubs == ["bad/IMG_0001.HEIC", "documents/legacy.doc", "documents/meeting.msg", "documents/photos.7z",
                     "other/app.exe", "other/book.xlsx", "other/letter.docx", "other/places.sqlite", "other/report.pdf"], \
        "a file that is only as much of its type as mms sniffs is marked, and nothing else is"
    for rel in stubs:
        assert corpus[rel].signal["stub"] is True and not {"words", "contains"} & set(corpus[rel].signal), \
            f"{rel} is a stub, so it claims no words: a test may check its plumbing, never its content"
    assert all(not entry.stub for rel, entry in corpus.items() if rel.startswith("documents/") and "words" in entry.signal)


def test_the_documents_are_the_same_bytes_each_build(tmp_path):
    first, second = build_documents(tmp_path / "one"), build_documents(tmp_path / "two")
    assert first.keys() == second.keys() and len(first) == 22
    differing = [rel for rel in first if first[rel].path.read_bytes() != second[rel].path.read_bytes()]
    assert not differing, f"{differing} differ between two builds"
    scan_pdf = first["documents/scan.pdf"].path.read_bytes()  # Pillow dates a PDF by the clock unless it is told the time
    assert b"/CreationDate (D:20261031080000Z)" in scan_pdf and b"/ModDate (D:20261031080000Z)" in scan_pdf
    assert sum(entry.path.stat().st_size for entry in first.values()) < 200_000


WORD, SHEET, DECK = (OFFICE + "wordprocessingml.document", OFFICE + "spreadsheetml.sheet", OFFICE + "presentationml.presentation")
ODT, ODS, EPUB = ODF + "text", ODF + "spreadsheet", "application/epub+zip"


def edited(members: dict[str, str], member: str, old: str, new: str) -> dict[str, str]:
    assert old in members[member], f"{old!r} is not in {member}"
    return {**members, member: members[member].replace(old, new)}


def test_the_build_check_follows_a_package_as_a_reader_does():
    docx, xlsx, pptx = docx_members(), xlsx_members(), pptx_members()
    odt, ods = odf_members("text", ODT_BODY), odf_members("spreadsheet", ODS_BODY)
    epub = epub_members()
    for name, media, members, words in (("minutes.docx", WORD, docx, ["retention", "30"]), ("budget.xlsx", SHEET, xlsx, ["café", "4821"]),
                                        ("deck.pptx", DECK, pptx, ["ship", "friday"]), ("notes.odt", ODT, odt, ["dentist"]),
                                        ("sheet.ods", ODS, ods, ["4821"]), ("book.epub", EPUB, epub, ["lighthouse"])):
        check_document(name, media, zipped(members), words)  # each good package is followed to its words

    notes = f'<w:document {NS_W}><w:body><w:p><w:r><w:t>{{}}</w:t></w:r></w:p></w:body></w:document>'
    main = OFFICE + "wordprocessingml.document.main+xml"
    orphan = ooxml("word/document.xml", {"word/document.xml": (main, notes.format("Notes")),
                                         "word/orphan.xml": (main, notes.format("Retention rule 30 days"))}, {})
    strings = f'<Relationship Id="rId2" Type="{REL}/sharedStrings" Target="sharedStrings.xml"/>'
    no_slide = edited(pptx, "ppt/presentation.xml", '<p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst>', "<p:sldIdLst/>")
    bad = [
        ("orphan.docx", WORD, orphan, ["retention"], "word/orphan.xml is related to by no part"),
        ("no-type.docx", WORD, edited(docx, "[Content_Types].xml", f'<Override PartName="/word/document.xml" ContentType="{main}"/>',
                                      ""), ["retention"], "content type"),
        ("stray.docx", WORD, {**docx, "word/blob.bin": "x"}, ["retention"], "word/blob.bin has no content type"),
        ("wrong-id.xlsx", SHEET, edited(xlsx, "xl/workbook.xml", 'r:id="rId1"', 'r:id="rId7"'), ["4821"], "has no relationship 'rId7'"),
        ("no-strings.xlsx", SHEET, edited(xlsx, "xl/_rels/workbook.xml.rels", strings, ""), ["café"],
         "xl/sharedStrings.xml is related to by no part"),
        ("no-slide.pptx", DECK, no_slide, ["ship"], "does not show ['ship']"),
        ("outside-cell.ods", ODS, odf_members("spreadsheet", "<text:p>4821</text:p>"), ["4821"], "does not show ['4821']"),
        ("outside-text.odt", ODT, edited(odf_members("text", "<text:p>Hello</text:p>"), "content.xml", "</office:text>",
                                         "</office:text><text:p>Dentist</text:p>"), ["dentist"], "does not show ['dentist']"),
        ("unlisted.odt", ODT, edited(odt, "META-INF/manifest.xml",
                                     '<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>', ""),
         ["dentist"], "styles.xml is not in the manifest"),
        ("late-mimetype.ods", ODS, {**{k: v for k, v in ods.items() if k != "mimetype"}, "mimetype": ODS}, ["4821"],
         "its first member is content.xml"),
        ("no-version.odt", ODT, edited(odt, "content.xml", ' office:version="1.2"', ""), ["dentist"], "content.xml has no office:version"),
        ("off-spine.epub", EPUB, edited(epub, "OEBPS/content.opf", '<spine><itemref idref="c1"/></spine>',
                                        '<spine><itemref idref="nav"/></spine>'), ["lighthouse"], "does not show ['lighthouse']"),
        ("no-id.epub", EPUB, edited(epub, "OEBPS/content.opf", 'unique-identifier="bookid"', 'unique-identifier="isbn"'),
         ["lighthouse"], "no identifier its unique-identifier names"),
        ("no-nav.epub", EPUB, edited(epub, "OEBPS/content.opf", ' properties="nav"', ""), ["lighthouse"], "no navigation document"),
        ("no-time.epub", EPUB, edited(epub, "OEBPS/content.opf", '<meta property="dcterms:modified">2026-10-31T08:00:00Z</meta>', ""),
         ["lighthouse"], "dcterms:modified"),
        ("stray.epub", EPUB, {**epub, "OEBPS/extra.xhtml": "<html/>"}, ["lighthouse"], "OEBPS/extra.xhtml is not in the manifest"),
    ]
    for name, media, members, words, says in bad:
        with pytest.raises(RuntimeError) as raised:
            check_document(name, media, zipped(members), words)
        assert says in str(raised.value), f"{name}: {raised.value}"
    assert "retention" not in doctext.text_of(zipped(orphan)).casefold(), "words in a part no relationship reaches are not shown"
    assert "ship" not in doctext.text_of(zipped(no_slide)).casefold(), "nor are the words of a slide the deck does not list"


def test_the_not_read_warning_is_parsed_and_misworded_ones_are_refused():
    one = ("1 item was not read, since d1-Omni-2B does not read its type: application/pdf (item 7); find lists it and look gives "
           "a copy of it")
    assert parse_unread(one) == Unread(1, "d1-Omni-2B", {"application/pdf": [7]}, {"application/pdf": 0},
                                       "find lists it and look gives a copy of it")
    many = ("9 items were not read, since other does not read their types: application/pdf (items 1, 2, 3, 4, 5 and 2 more), "
            "audio segments (items 8, 9); find lists them and look shows them")
    got = parse_unread(many)
    assert got.count == 9 and got.decider == "other" and got.groups == {"application/pdf": [1, 2, 3, 4, 5], "audio segments": [8, 9]}
    assert got.more == {"application/pdf": 2, "audio segments": 0} and got.tail == "find lists them and look shows them"
    assert check_unread(many, "other", {"application/pdf": {1, 2, 3, 4, 5, 20, 21}, "audio segments": {8, 9}}) == got
    for wrong in (one.replace("1 item was", "1 item were"), one.replace("its type", "their types"), one.replace("(item 7)", "(items 7)"),
                  many.replace("9 items", "8 items"), many.replace("items 1, 2, 3, 4, 5 and 2 more", "items 1, 2 and 2 more"),
                  many.replace("(items 8, 9)", "(item 8, 9)"), many.replace("; find", ", find"), "nothing was read"):
        with pytest.raises(AssertionError):
            parse_unread(wrong)
    for decider, expected in (("d1-Omni-2B", {"application/pdf": {1, 2, 3, 4, 5, 20, 21}, "audio segments": {8, 9}}),
                              ("other", {"application/pdf": {1, 2, 3, 4, 5, 20}, "audio segments": {8, 9}}),
                              ("other", {"application/pdf": {1, 2, 3, 4, 5, 20, 21}}),
                              ("other", {"application/pdf": {1, 2, 3, 4, 6, 20, 21}, "audio segments": {8, 9}})):
        with pytest.raises(AssertionError):
            check_unread(many, decider, expected)


# what the reader hands d1

def test_pictures_reach_d1_upright_flattened_and_within_budget(home, api, store, corpus):
    rels = [rel for rel, entry in corpus.items() if entry.read == "image" and not rel.startswith("bad/")]
    remember_each(api, corpus, rels)
    windows, unreadable = read_windows(store)
    problems = [f"{name} is unreadable: {error}" for name, error in unreadable.items()]
    for rel in rels:
        entry = corpus[rel]
        signal = entry.signal
        if len(signal.get("frames", ())) > 1:  # an animation or pages: the next test
            continue
        got = windows.get(entry.name, [])
        if [w.kind for w in got] != ["image"]:
            problems.append(f"{rel}: windows of {[w.kind for w in got]}, not one image")
            continue
        picture = got[0].media
        width, height = signal["size"]
        pixels = picture.size[0] * picture.size[1]
        if pixels > reader.PROFILE["max_px"]:
            problems.append(f"{rel}: {picture.size} is {pixels} px, over the budget")
        if abs(picture.size[0] / picture.size[1] / (width / height) - 1) > 0.01:
            problems.append(f"{rel}: {picture.size} is not the aspect of {signal['size']}")
        if width * height <= reader.PROFILE["max_px"] and picture.size != signal["size"]:
            problems.append(f"{rel}: {picture.size}, not its own size {signal['size']}")
        if "corner" in signal and corners(picture)[0] != signal["corner"]:
            problems.append(f"{rel}: the top-left is {corners(picture)[0]}, not {signal['corner']}: {corners(picture)}")
        if signal.get("ink") and not has_ink(picture):
            problems.append(f"{rel}: no ink shows (mean colour {colour(picture)})")
    assert not problems, "\n".join(problems)
    end = T0 + GAP * len(rels)
    red = {corpus[rel].name for rel in rels if corpus[rel].signal.get("corner") == "red"}
    assert names_hit(store, scan(api, end, question="corner:red", limit=50)) == red
    inked = {corpus[rel].name for rel in rels if corpus[rel].signal.get("ink") or "corner" in corpus[rel].signal}
    blank = home.parent / "blank"  # controls: nothing drawn, so nothing shows
    blank.mkdir()
    Image.new("RGB", CARD, "white").save(blank / "white.png")
    Image.new("RGBA", CARD, (0, 0, 0, 0)).save(blank / "clear.png")
    for n, control in enumerate(sorted(blank.iterdir())):
        api.call("remember", path=str(control), t=end + GAP * n)
    hit = names_hit(store, scan(api, end + 3 * GAP, question="ink", limit=50))
    assert hit == inked, f"missing {sorted(inked - hit)}, extra {sorted(hit - inked)}"


def test_animations_and_pages_are_read_frame_by_frame(home, api, store, corpus):
    rels = [rel for rel, entry in corpus.items() if len(entry.signal.get("frames", ())) > 1]
    assert len(rels) == 4
    remember_each(api, corpus, rels)
    forward, unreadable = read_windows(store)
    backward, _ = read_windows(store, newest_first=True)
    problems = [f"{name} is unreadable: {error}" for name, error in unreadable.items()]
    for rel in rels:
        entry = corpus[rel]
        got, back = forward.get(entry.name, []), backward.get(entry.name, [])
        if [w.at for w in got] != pytest.approx(entry.signal["frames"]):
            problems.append(f"{rel}: windows at {[w.at for w in got]}, not {entry.signal['frames']}")
            continue
        if [w.at for w in back] != [w.at for w in reversed(got)]:
            problems.append(f"{rel}: newest first is {[w.at for w in back]}")
        if corners(got[-1].media)[0] != "red" or any(corners(w.media)[0] != "white" for w in got[:-1]):
            problems.append(f"{rel}: the frames' top-left corners are {[corners(w.media)[0] for w in got]}")
        if len({(w.t0, w.t1) for w in got}) != 1 or got[0].t0 != got[0].item.t0:
            problems.append(f"{rel}: its windows are not all at the version's time")
    assert not problems, "\n".join(problems)
    thinned, _ = read_windows(store, profile={"every_s": 2})
    assert [w.at for w in thinned["pages.tif"]] == [0.0, 1.0, 2.0], "every page of a TIFF is a window, whatever every_s"


def test_text_reaches_d1_decoded(home, api, store, corpus):
    rels = [rel for rel, entry in corpus.items() if entry.read == "text" and not rel.startswith("bad/")]
    remember_each(api, corpus, rels)
    windows, unreadable = read_windows(store)
    problems = [f"{name} is unreadable: {error}" for name, error in unreadable.items()]
    for rel in rels:
        entry = corpus[rel]
        signal = entry.signal
        got = windows.get(entry.name, [])
        text = "".join(w.text for w in got)
        for word in (*signal["contains"], *([signal["after"]] if "after" in signal else [])):
            if word not in text:
                problems.append(f"{rel}: {word!r} is not in {text[:80]!r}")
        for bad in ("�", "\0", "\r", "﻿"):
            if bad in text:
                problems.append(f"{rel}: the text holds {bad!r}: {text[:80]!r}")
        if "lines" in signal and len(text.strip().splitlines()) != signal["lines"]:
            problems.append(f"{rel}: {len(text.strip().splitlines())} lines, not {signal['lines']}")
        if "chars" in signal and (max(map(len, (w.text for w in got)), default=0) > 6000
                                  or len(text) != signal["chars"]):
            problems.append(f"{rel}: {[len(w.text) for w in got]} characters in its windows, not 6,000 at most of "
                            f"{signal['chars']}")
    assert not problems, "\n".join(problems)
    result = scan(api, T0 + GAP * len(rels), question="contains:café", limit=50)
    cafe = {corpus[rel].name for rel in rels if "café" in corpus[rel].signal["contains"]}  # an RTF spells é as an escape
    assert names_hit(store, result) == cafe
    states = [entry["text"] for entry in decider_log(home)]
    assert states and all(s["nul"] == s["fffd"] == s["cr"] == s["esc"] == 0 for s in states), states[:3]


def test_a_few_bad_bytes_leave_utf8_as_it_is():
    body = BODY.encode()
    stray = body.replace(b"plain", b"pl\xe9in")
    marked = BODY.replace("plain", "pl\ufffdin")
    cases = {"a stray byte": (stray, marked), "a character cut at the end": (body + "é".encode()[:1], BODY + "\ufffd"),
             "a BOM and a stray byte": (b"\xef\xbb\xbf" + stray, marked)}
    for how, (data, text) in cases.items():
        assert media.decode_text(data) == text, how
    assert media.decode_text("café".encode("cp1252")) == "café", "a legacy file is still cp1252"


def test_sound_of_every_format_is_16k_mono_at_the_right_time(home, api, store, corpus, monkeypatch):
    monkeypatch.setattr(reader, "AUDIO_BLOCK_S", 4.0)  # every 12 s file is read in blocks, each from a seek
    rels = [rel for rel, entry in corpus.items() if rel.startswith("sound/")]
    times = remember_each(api, corpus, rels)
    quiet = {"quiet_rms": 0}
    forward, unreadable = read_windows(store, False, quiet)
    backward, _ = read_windows(store, True, quiet)
    problems = [f"{name} is unreadable: {error}" for name, error in unreadable.items()]
    for rel in rels:
        entry = corpus[rel]
        got, back = forward.get(entry.name, []), backward.get(entry.name, [])
        if not got:
            continue
        check_beep(rel, got, problems)
        if len(back) != len(got) or not all(np.array_equal(a.media["audio"], b.media["audio"])
                                            for a, b in zip(back, reversed(got))):
            problems.append(f"{rel}: newest first is not oldest first reversed")
    assert not problems, "\n".join(problems)
    result = scan(api, T0 + GAP * len(rels), question="loud", limit=50)
    spans = {store.item(s["look"]["item"]).name: span(s) for s in result["sightings"]}
    assert set(spans) == {corpus[rel].name for rel in rels}, "one sighting for each file"
    for rel in rels:
        t0, t1 = spans[corpus[rel].name]
        assert t0 <= times[rel] + 6.2 <= t1, f"{rel}: the sighting {t0 - times[rel]:.1f}-{t1 - times[rel]:.1f} s misses the beep"


def test_a_videos_frames_are_upright_and_its_soundtrack_is_heard(home, api, store, corpus):
    rels = [rel for rel, entry in corpus.items() if rel.startswith("video/")]
    remember_each(api, corpus, rels)
    remember_each(api, {"meeting": corpus["video/meeting.mp4"]}, ["meeting"], source="meet")
    windows, unreadable = read_windows(store, profile={"quiet_rms": 0, "still_s": 0})  # every sound window and frame, to count them
    problems = [f"{name} is unreadable: {error}" for name, error in unreadable.items()]
    for rel in rels:
        entry = corpus[rel]
        signal = entry.signal
        frames = [w for w in windows.get(entry.name, []) if w.kind == "video" and w.source == "agent"]
        sounds = [w for w in windows.get(entry.name, []) if w.kind == "audio" and w.source == "agent"]
        if not frames:
            problems.append(f"{rel}: no frames")
            continue
        sizes = {w.media.size for w in frames}
        if sizes != set(signal.get("sizes", [signal.get("size")])):
            problems.append(f"{rel}: frames of {sorted(sizes)}, not {signal.get('sizes', signal.get('size'))}")
        for w in frames:
            if "corner" in signal and corners(w.media)[0] != signal["corner"]:
                problems.append(f"{rel}: the top-left at {w.at} s is {corners(w.media)[0]}: {corners(w.media)}")
                break
            if "colour" in signal and signal["colour"] != "red" and colour(w.media) != signal["colour"]:
                problems.append(f"{rel}: the colour at {w.at} s is {colour(w.media)}, not {signal['colour']}")
                break
        ours = [w for w in windows[entry.name] if w.source == "agent"]
        if [w.t0 for w in ours] != sorted(w.t0 for w in ours):
            problems.append(f"{rel}: oldest first, its windows are not in order of t0: "
                            f"{[(w.kind, w.t0 - w.item.t0) for w in ours]}")
        if signal.get("sound"):
            if any(w.item.id != frames[0].item.id for w in sounds):
                problems.append(f"{rel}: a sound window of another item")
            if "beep_s" in signal:
                check_beep(rel, sounds, problems)
        elif sounds:
            problems.append(f"{rel}: {len(sounds)} sound windows, but it has no soundtrack")
    assert not problems, "\n".join(problems)
    counts = reader.estimate(store, None, None, ["meet"], EVERYTHING, about=decider_d1.ABOUT)
    read = {kind: len([w for w in windows["meeting.mp4"] if w.source == "meet" and w.kind == kind])
            for kind in ("video", "audio")}
    for kind, got in read.items():  # it may count one window too many, never too few
        assert got > 0 and got <= counts[kind] <= got + 1, f"{kind}: {counts[kind]} estimated, {got} read"


def test_a_soundtrack_shorter_than_the_picture_is_heard_only_where_it_plays(home, api, store, corpus):
    times = remember_each(api, corpus, ["video/short_sound.mp4"])
    windows, unreadable = read_windows(store)
    assert not unreadable, unreadable
    sounds = [w for w in windows["short_sound.mp4"] if w.kind == "audio"]
    assert sounds and max(w.t1 - w.item.t0 for w in sounds) <= TONE_S + 0.1, [(w.at, w.t1 - w.item.t0) for w in sounds]
    assert not any(np.array_equal(a.media["audio"], b.media["audio"]) for a, b in zip(sounds, sounds[1:])), \
        "a window of sound is repeated"
    (sighting,) = scan(api, T0 + GAP, question="loud")["sightings"]
    start = times["video/short_sound.mp4"]
    t0, t1 = span(sighting)[0] - start, span(sighting)[1] - start
    assert 0 <= t0 and t1 <= TONE_S + 0.1, f"loud at {t0:.1f}-{t1:.1f} s, but the sound ends at {TONE_S} s"


def test_a_screen_recordings_variable_frame_rate_keeps_its_times(home, api, store, corpus):
    times = remember_each(api, corpus, ["video/screen_vfr.mkv"])
    result = scan(api, T0 + GAP, question="colour:red", limit=1, oldest_first=True)
    (sighting,) = result["sightings"]
    assert span(sighting)[0] == pytest.approx(times["video/screen_vfr.mkv"] + 20.0, abs=0.6)


def test_broken_files_are_warned_and_the_scan_goes_on(home, served, corpus, tmp_path):
    watched = tmp_path / "watched"
    shutil.copytree(folder_of(corpus, "bad"), watched)
    shutil.copy(corpus["pictures/IMG_2041.jpg"].path, watched / "card.jpg")
    write_config(home, CONFIG + f'[sources.dir]\nkind = "folder"\nrescan_s = 0.2\npath = "{watched.as_posix()}"\n')
    api = served().api
    store = Store(home)
    wait_for(lambda: len([i for i in store.items_of("dir") if i.state == "ok"]) == len(list(watched.iterdir())),
             timeout=60, what="the folder source's versions")
    ids = {item.name: item.id for item in store.items_of("dir")}
    result = scan(api, time.time() + 60, since=time.time() - 3600, question="corner:red")
    warnings = "\n".join(result.get("warnings", []))
    problems = []
    for name, says in (("truncated.jpg", "OSError: image file is truncated"), ("no_moov.mp4", "unreadable"),
                       ("IMG_0001.HEIC", "pi-heif"), ("subs.mkv", "no video stream")):
        if f"item {ids[name]} unreadable" not in warnings or says not in warnings:
            problems.append(f"{name}: no warning with {says!r}")
    if f"item {ids['empty.txt']} " in warnings:
        problems.append("empty.txt is warned about")
    assert not problems, "\n".join(problems) + f"\nwarnings:\n{warnings}"
    assert names_hit(store, result) == {"card.jpg"}, "the card still hits"
    assert api.failure("remember", path=str(corpus["bad/no_moov.mp4"].path)).code == "bad_request"
    store.close()


def test_logs_are_read_as_a_terminal_shows_them(home, served, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_config(home, CONFIG + f'[sources.logs]\nkind = "log"\nrescan_s = 0.2\npath = "{logs.as_posix()}"\n')
    api = served().api
    store = Store(home)
    (logs / "server.log").write_bytes(
        b"\x1b[31mERROR\x1b[0m ERR-503 upstream timeout\n\x1b[31mERROR\x1b[0m ERR-503 upstream timeout\n"
        b"\x1b]0;build\x07build started\n"
        b"downloading 10%\rdownloading 50%\rdownloading 100%\n"
        b"caf\xe9 au lait (cp1252)\n" + "café au lait (utf-8)\n".encode())
    lines = lambda: [json.loads(r.data)["line"] for r in store.records(sources=["logs"])]  # noqa: E731
    wait_for(lambda: len(lines()) >= 6, what="the server log's records", detail=lambda: str(lines()))
    got = lines()
    assert not any("\x1b" in line for line in got), got
    assert sorted(got) == sorted(["ERROR ERR-503 upstream timeout"] * 2 + ["build started", "downloading 100%",
                                                                           "café au lait (cp1252)", "café au lait (utf-8)"])
    assert len(api.call("find", text="ERR-503")["results"]) == 2
    ps = logs / "ps.log"  # PowerShell 5.1 `>>`: UTF-16LE with a BOM and CRLF, then appended without one
    ps.write_bytes(b"\xff\xfe" + "first line café\r\nsecond line 4821\r\n".encode("utf-16-le"))
    wait_for(lambda: "second line 4821" in lines(), what="the first append", detail=lambda: str(lines()))
    with open(ps, "ab") as file:
        file.write("third line\r\nfourth line\r\n".encode("utf-16-le"))
    wait_for(lambda: "fourth line" in lines(), what="the second append", detail=lambda: str(lines()))
    assert lines()[6:] == ["first line café", "second line 4821", "third line", "fourth line"]
    size = ps.stat().st_size  # the old tail saved the offset after a newline's first byte, which is odd in UTF-16
    store.set_cursor("logs", "ps.log", offset=size - 1, size=size)
    with open(ps, "ab") as file:
        file.write("fifth line\r\nsixth line\r\n".encode("utf-16-le"))
    wait_for(lambda: "sixth line" in lines(), what="the append after an odd cursor", detail=lambda: str(lines()))
    assert lines()[10:] == ["fifth line", "sixth line"]
    assert not any("\0" in line for line in lines())
    store.close()


# big files

def test_reading_big_files_keeps_memory_bounded(home, served, store, tmp_path):
    watched = tmp_path / "watched"
    watched.mkdir()
    write_config(home, CONFIG + f'[sources.dir]\nkind = "folder"\nrescan_s = 0.2\npath = "{watched.as_posix()}"\ncopy_max_mb = 1\n')
    api = served().api
    photo = Image.new("RGB", (8000, 6000), "white")  # a 48 MP phone photo
    draw = ImageDraw.Draw(photo)
    for x in range(0, 8000, 400):
        draw.rectangle([x, 0, x + 200, 6000], fill=(200, 30, 30))
    photo.save(tmp_path / "photo.jpg", quality=80)
    del photo
    Image.new("1", (9000, 8000)).save(tmp_path / "scan.png")  # 72 MP, mostly nothing
    with wave.open(str(tmp_path / "memo.wav"), "wb") as memo:  # 20 minutes at 16 kHz mono: 38 MB
        memo.setnchannels(1)
        memo.setsampwidth(2)
        memo.setframerate(16000)
        second = (3000 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype("<i2").tobytes()
        for _ in range(1200):
            memo.writeframes(second)
    for name, source in (("photo.jpg", "photo"), ("scan.png", "scan"), ("memo.wav", "memo")):
        api.call("remember", path=str(tmp_path / name), t=T0, source=source)
    line = b"2026-10-31 12:00:00 INFO the server is up and the log is long\n"
    (tmp_path / "big.log").write_bytes(line * (65 * MB // len(line)))
    (tmp_path / "big.log").replace(watched / "big.log")  # whole at once, or the scan indexes a half-written file too
    wait_for(lambda: any(i.state == "ok" and i.meta.get("size", 0) >= 65 * MB - len(line) for i in store.items_of("dir")),
             timeout=60, what="the big log, indexed in place")

    def run(source: str, code: str) -> int:
        head = f"""
            errors = []
            windows = reader.windows(store, about=decider_d1.ABOUT, sources=["{source}"], scope=Scope(all=True),
                                     on_unreadable=lambda item, error: errors.append(str(error)))
            """
        return peak_memory(textwrap.dedent(head) + textwrap.dedent(code), home)

    grown = run("photo", """
        got = list(windows)
        assert not errors and len(got) == 1, (errors, len(got))
        assert got[0].media.size[0] * got[0].media.size[1] <= 230_400, got[0].media.size
        """)
    assert grown < 60 * MB, f"the 48 MP photo grew the peak by {grown / MB:.0f} MB (a full decode is 144 MB)"
    grown = run("scan", """
        got = list(windows)
        assert not got and len(errors) == 1, (errors, len(got))
        assert "72 MP" in errors[0] and "64 MP" in errors[0], errors
        """)
    assert grown < 40 * MB, f"the 72 MP scan grew the peak by {grown / MB:.0f} MB: it was decoded"
    grown = run("memo", """
        import itertools
        got = list(itertools.islice(windows, 10))
        assert not errors and len(got) == 10, (errors, len(got))
        assert [w.at for w in got] == [1195.0 - 2.5 * i for i in range(10)], [w.at for w in got]
        assert all(len(w.media["audio"]) == 80_000 for w in got)
        """)
    assert grown < 40 * MB, f"the first 10 windows of a 20 minute memo grew the peak by {grown / MB:.0f} MB"
    grown = run("dir", """
        got = list(windows)
        assert not got and len(errors) == 1, (errors, len(got))
        assert "64 MB" in errors[0] and "log source" in errors[0], errors
        """)
    assert grown < 40 * MB, f"the 65 MB text grew the peak by {grown / MB:.0f} MB: it was read"
    (big,) = [i for i in store.items_of("dir") if i.name == "big.log"]
    (looked,) = api.call("look", moments=[{"item": big.id}])["moments"]
    assert "64 MB" in looked["error"]["message"] and "log source" in looked["error"]["message"], looked


# look

def test_look_shows_each_kind_upright(home, api, store, corpus):
    rels = ["pictures/IMG_2041.jpg", "video/rot90.mp4", "pictures/reaction.gif", "video/meeting.mp4",
            "text/powershell.txt", "sound/voice_memo.m4a"]
    remember_each(api, corpus, rels)
    items = {item.name: item.id for item in store.items_of("agent")}
    ids = [items[corpus[rel].name] for rel in rels]
    moments = [{"item": ids[0]}, {"item": ids[1]}, {"item": ids[2], "at": 1.2}, {"item": ids[3], "at": 6},
               {"item": ids[4]}, {"item": ids[5], "at": 5}]
    got = api.call("look", moments=moments)["moments"]
    assert all("error" not in entry for entry in got), got

    def upright(entry) -> tuple[tuple[int, int], str]:
        with Image.open(entry["image"]["path"]) as seen:
            return seen.size, corners(seen)[0]

    assert upright(got[0]) == (CARD, "red")
    assert upright(got[1]) == ((90, 160), "red")
    assert upright(got[2]) == (CARD, "red") and got[2]["at"] == 1.2
    with Image.open(got[3]["image"]["path"]) as seen:
        assert colour(seen) == "grey"
    assert got[3]["audio"]["rms"] > 0.05, "a video with sound shows its sound as well"
    assert "café" in got[4]["text"]
    assert got[5]["kind"] == "audio" and got[5]["audio"]["rms"] > 0.05
