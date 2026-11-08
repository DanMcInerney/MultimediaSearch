"""The documents of a home folder through mms end to end (TESTBED.md, "Real files"): PDFs, Office and ODF files, EPUBs, mail,
OLE files, archives and databases are recorded, typed by their bytes, findable, and, when a decider does not read them,
named to the agent. When one does (`other_decider.py` declares PDFs, Office, ODF, EPUB and HEIC), each arrives whole and
`ask`'s sightings, candidates and labels and `look` work on it. These tests are the harness for a future decider that
reads documents: its `ABOUT` lists the types, and the corpus says what each must give. A corpus `stub` (an OLE file, a 7z,
an HEIC) is handed over like any file, and only its bytes are checked, never what it holds."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import other_decider
import pytest
from conftest import DECIDER, OTHER, OTHER_DECIDER, wait_for, write_config
from corpus import (GAP, OFFICE, T0, check_unread, docx_members, names_hit, not_read_warnings, read_windows, remember_each, scan,
                    zipped)
from mcp import types
from mcp_session import drive

from mms import decider_d1, reader
from mms.store import Store, file_name

CONFIG = "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 50\n"
ABOUT = other_decider.ABOUT


@pytest.fixture
def store(home):
    one = Store(home)
    yield one
    one.close()


def started(home, served, decider: str):
    write_config(home, CONFIG + decider)
    return served().api


def documents(corpus) -> list[str]:
    return [rel for rel in corpus if rel.startswith("documents/")]


def log_of(home: Path, file: str) -> list[dict]:
    path = home / file
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def unread_by_media(corpus, ids: dict[str, int], rels) -> dict[str, set[int]]:
    """The media type of each of `rels` with the ids of the items of that type: what a not-read warning must name."""
    expected: dict[str, set[int]] = {}
    for rel in rels:
        expected.setdefault(corpus[rel].media, set()).add(ids[corpus[rel].name])
    return expected


# -- under d1: recorded, findable, named, and never handed over ---------------------------------------------------------

def test_under_d1_each_document_is_named_and_none_reaches_it(home, served, store, corpus):
    api = started(home, served, DECIDER)
    rels = documents(corpus)
    remember_each(api, corpus, rels)
    result = scan(api, T0 + GAP * len(rels), question="contains:4821", limit=50)
    ids = {item.name: item.id for item in store.items_of("agent")}
    unread = [rel for rel in rels if corpus[rel].read is None]
    text = [rel for rel in rels if corpus[rel].read == "text"]
    assert len(unread) + len(text) == len(rels) and len(text) == 4

    # one warning, with each type of document and its items
    assert result["warnings"] == not_read_warnings(result) and len(result["warnings"]) == 1, \
        f"only the not-read warning, nothing skipped or unreadable: {result['warnings']}"
    (warning,) = result["warnings"]
    named = check_unread(warning, "d1-Omni-2B", unread_by_media(corpus, ids, unread))
    assert named.count == len(unread) and named.tail.endswith("look gives a copy of each"), warning

    # what d1 was given is the text of the four plain-text documents, and nothing else
    windows, unreadable = read_windows(store, decider_d1.ABOUT)
    assert not unreadable, unreadable
    assert set(windows) == {corpus[rel].name for rel in text}
    assert all(w.kind == "text" for ws in windows.values() for w in ws), "RTF, iCalendar, vCard and mail are text"
    calls = log_of(home, "fake-decider.jsonl")
    assert all(call["files"] == [] for call in calls), f"d1 was handed a file: {[c['files'] for c in calls if c['files']]}"
    assert len(calls) == sum(map(len, windows.values())) and all(not c["pictures"] and not c["sounds"] for c in calls), \
        f"d1 was given {len(calls)} windows, and only the text windows are {sum(map(len, windows.values()))}"
    rtf = "".join(w.text for w in windows["letter.rtf"])
    assert "Gate code 4821" in rtf and "\r" not in rtf
    assert names_hit(store, result) == {corpus[rel].name for rel in text if "4821" in corpus[rel].signal["contains"]}, \
        "contains:4821 finds the RTF, the vCard and the mail (gate.eml) by their words"
    assert "gate.eml" in names_hit(store, result)


def test_one_unread_document_is_worded_in_the_singular(home, served, store, corpus):
    """The one test of a file warning word for word; every other test parses it (`corpus.parse_unread`)."""
    api = started(home, served, DECIDER)
    remember_each(api, corpus, ["documents/invoice.pdf"])
    result = scan(api, T0 + GAP, question="contains:342.50")
    (item,) = store.items_of("agent")
    assert result["warnings"] == [f"1 item was not read, since d1-Omni-2B does not read its type: application/pdf "
                                  f"(item {item.id}); find lists it and look gives a copy of it"]


# -- under a decider that reads documents ---------------------------------------------------------------------------------

def test_a_decider_that_reads_documents_gets_each_whole(home, served, store, corpus):
    api = started(home, served, OTHER_DECIDER)
    rels = [*documents(corpus), "bad/IMG_0001.HEIC"]
    times = remember_each(api, corpus, rels)
    result = scan(api, T0 + GAP * len(rels), question="contains:nothing-says-this", limit=50, candidates=0)
    declared = [rel for rel in rels if ABOUT.takes(corpus[rel].media)]
    left = [rel for rel in rels if rel not in declared and corpus[rel].read is None]
    assert len(declared) == 11 and len(left) == 8, (declared, left)  # 4 PDFs, 6 Office, ODF and EPUB, HEIC; 8 no one reads yet

    windows, unreadable = read_windows(store, ABOUT)
    assert not unreadable, unreadable
    assert [rel for rel in declared if corpus[rel].stub] == ["bad/IMG_0001.HEIC"], \
        "the HEIC is a stub: it is handed over whole and its bytes are checked, never what it shows"
    problems = []
    for rel in declared:
        entry = corpus[rel]
        got = windows.get(entry.name, [])
        if [w.kind for w in got] != ["file"]:
            problems.append(f"{rel}: windows of {[w.kind for w in got]}, not one file")
            continue
        (w,) = got
        if w.media != {"type": "file", "media": entry.media, "name": entry.name, "data": entry.path.read_bytes()}:
            problems.append(f"{rel}: its window holds {w.media['media']} {w.media['name']} of {len(w.media['data'])} bytes")
        if not (w.t0 == w.t1 == w.item.t0 == times[rel]) or w.at is not None or w.source != "agent":
            problems.append(f"{rel}: at {w.t0}-{w.t1} (item {w.item.t0}, remembered at {times[rel]}), at={w.at}")
    assert not problems, "\n".join(problems)

    # (what the daemon's decider was handed, by hash, is checked against a store of every kind in test_deciders)
    # HEIC is handed whole, and is not unreadable; the rest are named
    assert "unreadable" not in "".join(result.get("warnings", [])) and "pi-heif" not in "".join(result.get("warnings", []))
    (warning,) = not_read_warnings(result)
    ids = {item.name: item.id for item in store.items_of("agent")}
    check_unread(warning, "other", unread_by_media(corpus, ids, left))


def test_a_document_hit_is_a_sighting(home, served, store, corpus):
    api = started(home, served, OTHER_DECIDER)
    rels = documents(corpus)
    remember_each(api, corpus, rels)
    end = T0 + GAP * len(rels)
    result = scan(api, end, question="contains:342.50", limit=50, candidates=20)

    assert names_hit(store, result) == {"invoice.pdf", "invoice-copy.txt"}
    assert [s["kind"] for s in result["sightings"]] == ["file", "file"] and result["stopped"] == "range_end"
    for sighting in result["sightings"]:
        shown = api.call("look", **sighting["look"])
        assert shown["kind"] == "file" and Path(shown["file"]["path"]).read_bytes() == corpus[
            f"documents/{shown['file']['name']}"].path.read_bytes(), "the handle gives the file the sighting is about"
    labels = store.labels(by="ask")
    assert sorted(label.item for label in labels) == sorted(s["look"]["item"] for s in result["sightings"]) \
        and all(label.meta["kind"] == "file" for label in labels), [(label.item, label.meta) for label in labels]
    assert {label.meta["model"] for label in labels} == {OTHER.replace(chr(92), "/")}, "each label names the decider that said so"

    candidates = {store.item(c["look"]["item"]).name: c for c in result["candidates"]}
    assert {"scan.pdf", "locked.pdf"} <= set(candidates), \
        f"a PDF with no text layer and a locked one come back as candidates: {sorted(candidates)}"
    assert not set(candidates) & {"invoice.pdf", "invoice-copy.txt"} and candidates["scan.pdf"]["kind"] == "file"
    assert len(store.labels(by="ask")) == 2, "candidates are never labelled"

    # every document is findable by the words it holds, as the corpus says, and by nothing else
    words = sorted({word.casefold() for entry in corpus.values() for word in entry.signal.get("words", []) if len(word) >= 4})
    assert {"342.50", "lighthouse", "4821", "café", "dentist", "retention"} <= set(words)
    problems = []
    for word in words:
        wanted = {corpus[rel].name for rel in rels if (ABOUT.takes(corpus[rel].media) and word in
                                                       [w.casefold() for w in corpus[rel].signal.get("words", [])])
                  or word in [w.casefold() for w in corpus[rel].signal.get("contains", [])]}
        got = names_hit(store, scan(api, end, question=f"contains:{word}", limit=50))
        if got != wanted:
            problems.append(f"contains:{word} found {sorted(got)}, not {sorted(wanted)}")
    assert not problems, "\n".join(problems)


def test_an_edited_document_in_a_watched_folder_is_two_versions_each_read_whole(home, served, store, corpus, tmp_path):
    watched = tmp_path / "watched"
    watched.mkdir()
    original = corpus["documents/minutes.docx"].path.read_bytes()
    edited = zipped(docx_members(45))  # "Keep logs for 45 days", where the original says 30
    (watched / "minutes.docx").write_bytes(original)
    api = started(home, served, OTHER_DECIDER + f'[sources.dir]\nkind = "folder"\nrescan_s = 0.2\npath = "{watched.as_posix()}"\n')

    def versions():
        return sorted((item for item in store.items_of("dir") if item.name == "minutes.docx"), key=lambda item: item.t0)

    wait_for(lambda: len(versions()) == 1, timeout=60, what="the first version", detail=lambda: str(versions()))
    (watched / "minutes.docx").write_bytes(edited)
    wait_for(lambda: len(versions()) == 2, timeout=60, what="the edited version", detail=lambda: str(versions()))
    first, second = versions()
    assert first.t0 < second.t0 and {first.kind, second.kind} == {"file"}
    assert {first.meta["media"], second.meta["media"]} == {OFFICE + "wordprocessingml.document"}

    # a file window for each version, at that version's own time and holding that version's bytes
    windows, unreadable = read_windows(store, ABOUT)
    assert not unreadable, unreadable
    got = windows["minutes.docx"]
    assert [(w.kind, w.item.id, w.t0, w.t1, w.at, w.source) for w in got] == [("file", first.id, first.t0, first.t0, None, "dir"),
                                                                             ("file", second.id, second.t0, second.t0, None, "dir")]
    assert [w.media["data"] for w in got] == [original, edited]

    # both hit on a word they share: two sightings (merge_s 0 keeps versions seconds apart apart) and a label each
    span = {"since": time.time() - 3600, "merge_s": 0, "limit": 10, "candidates": 0}
    both = scan(api, time.time() + 60, question="contains:retention", **span)
    assert [s["kind"] for s in both["sightings"]] == ["file", "file"]
    assert sorted(s["look"]["item"] for s in both["sightings"]) == sorted([first.id, second.id])
    labels = store.labels(by="ask")
    assert sorted(label.item for label in labels) == sorted([first.id, second.id]) and {label.meta["kind"] for label in labels} == {"file"}
    for sighting, want in zip(sorted(both["sightings"], key=lambda s: s["look"]["item"]), (original, edited)):
        assert Path(api.call("look", **sighting["look"])["file"]["path"]).read_bytes() == want

    # and on the word they differ in, only that version
    for word, version in (("45", second), ("30", first)):
        (only,) = scan(api, time.time() + 60, question=f"contains:{word}", **span)["sightings"]
        assert only["look"]["item"] == version.id, f"contains:{word} is the {'edited' if version is second else 'original'} version"


# -- look ---------------------------------------------------------------------------------------------------------------

def test_look_gives_a_copy_of_a_file(home, served, store, corpus, tmp_path, monkeypatch):
    api = started(home, served, DECIDER)
    rels = ["documents/invoice.pdf", "documents/invoice-copy.txt", "documents/budget.xlsx", "documents/song.mid"]
    remember_each(api, corpus, rels)
    ids = {item.name: item.id for item in store.items_of("agent")}
    for rel in rels:
        entry = corpus[rel]
        shown = api.call("look", item=ids[entry.name])
        assert shown["kind"] == "file" and shown["at"] is None and shown["source"] == "agent", shown
        file = shown["file"]
        path = Path(file["path"])
        assert path.parent == home / "looks" and re.fullmatch(rf"[0-9a-f]{{12}}-{re.escape(entry.name)}", path.name), file
        assert path.read_bytes() == entry.path.read_bytes() and path != store.path_of(store.item(ids[entry.name])), \
            f"{rel}: a copy of the file"
        assert file == {"path": str(path), "name": entry.name, "bytes": entry.path.stat().st_size, "media": entry.media}, file
    assert api.call("look", item=ids["invoice-copy.txt"])["file"]["media"] == "application/pdf", "typed by its bytes, not its name"

    # over MCP: one text block with the path, and no picture
    pdf = ids["invoice.pdf"]

    async def steps(client):
        return await client.call_tool("look", {"item": pdf})

    result = drive(steps, home, tmp_path)
    assert not result.is_error and len(result.content) == 1 and isinstance(result.content[0], types.TextContent), result.content
    over_mcp = json.loads(result.content[0].text)
    assert over_mcp["kind"] == "file" and Path(over_mcp["file"]["path"]).read_bytes() == corpus[rels[0]].path.read_bytes()

    # a file over the limit is not copied: too_big on the item, alone or among moments
    monkeypatch.setattr(reader, "FILE_MAX_BYTES", 100)
    error = api.failure("look", item=pdf)
    assert error.code == "too_big" and error.field == "item", error
    assert "is 602 bytes; look copies at most 100 bytes" in error.error["message"], error
    got = api.call("look", moments=[{"item": pdf}, {"item": ids["song.mid"]}])["moments"]
    assert got[0]["error"]["code"] == "too_big" and got[0]["error"]["field"] == "item"
    assert got[1]["kind"] == "file", "a file under the limit is still copied"
    refused = drive(steps, home, tmp_path)
    assert refused.is_error and json.loads(refused.content[-1].text)["error"]["code"] == "too_big"
    before = len(list((home / "looks").iterdir()))
    api.failure("look", item=pdf)
    assert len(list((home / "looks").iterdir())) == before, "nothing is copied for a refused look"


def test_look_mixes_files_and_a_picture_over_mcp(home, served, store, corpus, tmp_path):
    api = started(home, served, DECIDER)
    rels = ["documents/invoice.pdf", "pictures/IMG_2041.jpg", "documents/minutes.docx"]
    remember_each(api, corpus, rels)
    ids = {item.name: item.id for item in store.items_of("agent")}

    async def steps(client):
        return await client.call_tool("look", {"moments": [{"item": ids[corpus[rel].name]} for rel in rels]})

    result = drive(steps, home, tmp_path)
    assert not result.is_error, result.content
    assert [type(block) for block in result.content] == [types.TextContent, types.ImageContent, types.TextContent,
                                                         types.TextContent], "a picture gets one image block, before its text"
    first, picture, second = (json.loads(block.text) for block in result.content if isinstance(block, types.TextContent))
    assert (first["kind"], picture["kind"], second["kind"]) == ("file", "image", "file")
    for text, rel in ((first, rels[0]), (second, rels[2])):
        entry = corpus[rel]
        assert "image" not in text and text["file"]["name"] == entry.name and text["file"]["media"] == entry.media, text
        assert Path(text["file"]["path"]).read_bytes() == entry.path.read_bytes(), "the file's path is in the text"
    image = result.content[1]
    assert image.mime_type == "image/jpeg" and Path(picture["image"]["path"]).is_file()
    assert base64.b64decode(image.data)[:2] == b"\xff\xd8", "the image block is the JPEG"


def test_a_file_over_the_limit_is_unreadable_not_silent(home, served, store, corpus, monkeypatch):
    monkeypatch.setattr(reader, "FILE_MAX_BYTES", 100)
    api = started(home, served, OTHER_DECIDER)
    rels = ["documents/invoice.pdf", "documents/minutes.docx", "bad/IMG_0001.HEIC"]  # 602, 1,304 and 88 bytes
    remember_each(api, corpus, rels)
    result = scan(api, T0 + GAP * len(rels), question="contains:342.50")
    ids = {item.name: item.id for item in store.items_of("agent")}
    warnings = result.get("warnings", [])
    for name in ("invoice.pdf", "minutes.docx"):
        assert any(w.startswith(f"item {ids[name]} unreadable: ValueError: a file of ") and "is over the reader's limit of"
                   in w for w in warnings), f"{name} is not named as unreadable: {warnings}"
    assert not any(f"item {ids['IMG_0001.HEIC']} " in w for w in warnings), f"the 88 byte HEIC is read: {warnings}"
    assert not not_read_warnings(result), "an over-limit file is unreadable, not unread"
    assert any(w == f"item {ids['invoice.pdf']} unreadable: ValueError: a file of 602 bytes is over the reader's limit of "
                    f"100 bytes" for w in warnings), f"a size under 1 MB reads in bytes or KB, not 0 MB: {warnings}"
    assert any(re.search(r"a file of \d\.\d KB is over", w) for w in warnings), warnings
    assert result["stopped"] == "range_end" and result["sightings"] == [], "the scan goes on past them"
    assert [f["name"] for call in log_of(home, "other-decider.jsonl") for f in call["files"]] == ["IMG_0001.HEIC"]


def test_a_size_in_a_message_reads_in_the_unit_it_is_near():
    assert [reader.size_text(n) for n in (0, 1, 602, 999, 1000, 1184, 1500, 999_000, 1_000_000, 64_000_000, 65_400_000)] == \
        ["0 bytes", "1 byte", "602 bytes", "999 bytes", "1 KB", "1.2 KB", "1.5 KB", "999 KB", "1 MB", "64 MB", "65.4 MB"]


def test_look_copies_a_file_whose_name_windows_cannot_hold(home, served, store, corpus):
    data = corpus["documents/invoice.pdf"].path.read_bytes()
    names = ["report?.pdf", "notes:v2.pdf", "plain.pdf", "a<b>|c*.pdf", "x" * 300 + ".pdf", "dir/inner.pdf"]
    ids = {name: store.add_version("docs", name, T0 + n * GAP, data=data + name.encode()) for n, name in enumerate(names)}
    api = started(home, served, DECIDER)
    for name in names:
        shown = api.call("look", item=ids[name])
        path = Path(shown["file"]["path"])
        assert path.parent == home / "looks" and re.fullmatch(r"[0-9a-f]{12}-[A-Za-z0-9._-]{1,80}", path.name), \
            f"{name[:20]!r} is copied as {path.name!r}"
        assert path.read_bytes() == data + name.encode() and shown["file"]["name"] == name, "the item's own name is kept in `file`"
        assert path.name[13:] == file_name(name, "file"), "named by the store's rule"
    assert len(list((home / "looks").iterdir())) == len(names), "no stream or stray file beside the copies"
    assert [file_name(name, "file") for name in ("report?.pdf", "", None, "???", "x" * 100)] == \
        ["report_.pdf", "file", "file", "___", "x" * 80]
