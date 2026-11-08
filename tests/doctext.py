"""The text a document shows, found the way a real reader finds it (TESTBED.md, "Real files"), with the standard library.
The corpus builds its documents with this to check its ground truth, and `other_decider.py` extracts text with it, so the
test decider reads a document as a decider with a parser would, and no differently from the build's own check.

A reader follows the package, and a part it never reaches shows nothing:

- a PDF: the strings the `Tj` operators of its inflatable content streams show (a stream that is a picture, or is
  encrypted, shows none);
- Word, Excel and PowerPoint: `[Content_Types].xml` and `_rels/.rels` give the main part (its `officeDocument`
  relationship) and its content type; Word shows that part's `w:t` runs; Excel follows the workbook's `<sheet r:id>` to
  each sheet through the workbook's relationships and shows its cells (a shared string by index, an inline string, a
  number); PowerPoint follows `sldIdLst` to each slide and shows its `a:t` runs;
- ODT and ODS: `META-INF/manifest.xml` must list `content.xml`; an ODT shows the `text:p` and `text:h` inside
  `office:text`, an ODS the `text:p` inside the `table:table-cell`s of its `table:table`s;
- EPUB: `META-INF/container.xml` gives the package document, whose spine gives the chapters through its manifest, and each
  chapter's `body` text is shown (the navigation document, if the spine leaves it out, is not).

What a part holds that no path reaches is not shown. `Unreadable` is what a reader says of a package it cannot follow."""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
import zlib
from xml.etree import ElementTree as ET

PACKAGE = "{http://schemas.openxmlformats.org/package/2006/relationships}"
TYPES = "{http://schemas.openxmlformats.org/package/2006/content-types}"
RELS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
SLIDE = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
OFFICE = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"
TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
MANIFEST = "{urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}"
CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"
OPF = "{http://www.idpf.org/2007/opf}"
DC = "{http://purl.org/dc/elements/1.1/}"
XHTML = "{http://www.w3.org/1999/xhtml}"
OFFICE_MAIN = {"wordprocessingml.document.main+xml": "word", "spreadsheetml.sheet.main+xml": "sheet",
               "presentationml.presentation.main+xml": "deck"}
STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
SHOWN = re.compile(rb"\(((?:\\.|[^\\)])*)\)\s*Tj")


class Unreadable(ValueError):
    """A document a reader cannot follow to its words."""


def local(tag: str) -> str:
    return tag.rpartition("}")[2]


def part(archive: zipfile.ZipFile, name: str) -> ET.Element:
    """The XML of a member, parsed."""
    try:
        data = archive.read(name)
    except KeyError:
        raise Unreadable(f"{name} is not in the package") from None
    try:
        return ET.fromstring(data)
    except ET.ParseError as error:
        raise Unreadable(f"{name} is not well-formed XML: {error}") from None


def resolve(source: str, target: str) -> str:
    """The member a relationship's `target` names: from the package's root when it starts with `/`, else from the folder of
    `source` (the part that holds the relationship; "" for the package)."""
    if target.startswith("/"):
        return target[1:]
    return posixpath.normpath(posixpath.join(posixpath.dirname(source), target))


def rels_name(source: str) -> str:
    """The member that holds the relationships of the part `source` ("" for the package: `_rels/.rels`)."""
    folder, name = posixpath.split(source)
    return posixpath.join(folder, "_rels", name + ".rels")


def relationships(archive: zipfile.ZipFile, source: str) -> list[dict]:
    """The relationships of `source`, each as `{"Id", "Type", "Target", "TargetMode"?}` with Target resolved to a member
    (an external one is left as it is); empty when the part has no relationships part."""
    name = rels_name(source)
    if name not in archive.namelist():
        return []
    out = []
    for element in part(archive, name):
        if local(element.tag) == "Relationship":
            entry = dict(element.attrib)
            if entry.get("TargetMode") != "External":
                entry["Target"] = resolve(source, entry["Target"])
            out.append(entry)
    return out


def related(archive: zipfile.ZipFile, source: str, kind: str) -> list[str]:
    """The members `source` relates to by a relationship whose type ends in `/kind`."""
    return [entry["Target"] for entry in relationships(archive, source) if entry["Type"].endswith("/" + kind)]


def content_type(archive: zipfile.ZipFile, name: str) -> str | None:
    """What `[Content_Types].xml` says of a member: its Override, else the Default for its extension."""
    types = part(archive, "[Content_Types].xml")
    for element in types:
        if local(element.tag) == "Override" and element.get("PartName") == "/" + name:
            return element.get("ContentType")
    base = posixpath.basename(name)  # ".rels" is the extension rels of a file with no name
    extension = base.rpartition(".")[2].casefold() if "." in base else ""
    for element in types:
        if local(element.tag) == "Default" and (element.get("Extension") or "").casefold() == extension:
            return element.get("ContentType")
    return None


def runs(root: ET.Element, paragraph: str, run: str) -> list[str]:
    """One line for each `paragraph` element under `root`: the text of the `run` elements in it."""
    return ["".join(t.text or "" for t in p.iter(run)) for p in root.iter(paragraph)]


def word_text(archive: zipfile.ZipFile, main: str) -> list[str]:
    return runs(part(archive, main), WORD + "p", WORD + "t")


def sheet_text(archive: zipfile.ZipFile, main: str) -> list[str]:
    """The cells of each sheet the workbook lists, a row to a line."""
    ids = {entry["Id"]: entry["Target"] for entry in relationships(archive, main)}
    strings: list[str] = []
    for target in related(archive, main, "sharedStrings"):
        strings = ["".join(t.text or "" for t in si.iter(SHEET + "t")) for si in part(archive, target).iter(SHEET + "si")]
    lines = []
    for sheet in part(archive, main).iter(SHEET + "sheet"):
        target = ids.get(sheet.get(RELS + "id"))
        if target is None:
            raise Unreadable(f"sheet {sheet.get('name')!r} of {main} has no relationship {sheet.get(RELS + 'id')!r}")
        for row in part(archive, target).iter(SHEET + "row"):
            cells = []
            for cell in row.iter(SHEET + "c"):
                value = cell.findtext(SHEET + "v")
                if cell.get("t") == "s" and value is not None:
                    if int(value) >= len(strings):
                        raise Unreadable(f"cell {cell.get('r')} is shared string {value}, and the workbook has {len(strings)}")
                    cells.append(strings[int(value)])
                elif cell.get("t") == "inlineStr":
                    cells.append("".join(t.text or "" for t in cell.iter(SHEET + "t")))
                elif value is not None:
                    cells.append(value)
            lines.append(" ".join(cells))
    return lines


def deck_text(archive: zipfile.ZipFile, main: str) -> list[str]:
    """The text of each slide the presentation's `sldIdLst` lists, in its order."""
    ids = {entry["Id"]: entry["Target"] for entry in relationships(archive, main)}
    lines = []
    for slide in part(archive, main).iter(SLIDE + "sldId"):
        target = ids.get(slide.get(RELS + "id"))
        if target is None:
            raise Unreadable(f"a slide of {main} has no relationship {slide.get(RELS + 'id')!r}")
        lines += runs(part(archive, target), DRAWING + "p", DRAWING + "t")
    return lines


def ooxml_text(archive: zipfile.ZipFile) -> list[str]:
    mains = related(archive, "", "officeDocument")
    if len(mains) != 1:
        raise Unreadable(f"_rels/.rels names {len(mains)} office documents, not one")
    main = mains[0]
    kind = (content_type(archive, main) or "").rpartition("officedocument.")[2]
    if OFFICE_MAIN.get(kind) is None:
        raise Unreadable(f"{main} is of content type {content_type(archive, main)!r}, which is no Word, Excel or PowerPoint part")
    return {"word": word_text, "sheet": sheet_text, "deck": deck_text}[OFFICE_MAIN[kind]](archive, main)


def odf_text(archive: zipfile.ZipFile, kind: str) -> list[str]:
    listed = {entry.get(MANIFEST + "full-path") for entry in part(archive, "META-INF/manifest.xml")}
    if "content.xml" not in listed:
        raise Unreadable("the manifest does not list content.xml")
    body = part(archive, "content.xml").find(OFFICE + "body")
    if body is None:
        raise Unreadable("content.xml has no office:body")
    if kind.endswith(".text"):
        scope = body.find(OFFICE + "text")
        if scope is None:
            raise Unreadable("office:body has no office:text")
        return ["".join(p.itertext()) for p in scope.iter() if p.tag in (TEXT + "p", TEXT + "h")]
    if kind.endswith(".spreadsheet"):
        scope = body.find(OFFICE + "spreadsheet")
        if scope is None:
            raise Unreadable("office:body has no office:spreadsheet")
        return [" ".join("".join(p.itertext()) for cell in row.iter(TABLE + "table-cell") for p in cell.iter(TEXT + "p"))
                for table in scope.iter(TABLE + "table") for row in table.iter(TABLE + "table-row")]
    raise Unreadable(f"{kind} is an OpenDocument type this reader does not take")


def epub_text(archive: zipfile.ZipFile) -> list[str]:
    rootfile = part(archive, "META-INF/container.xml").find(f".//{CONTAINER}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise Unreadable("META-INF/container.xml names no package document")
    package = rootfile.get("full-path")
    opf = part(archive, package)
    hrefs = {item.get("id"): item.get("href") for item in opf.iter(OPF + "item")}
    lines = []
    for itemref in opf.iter(OPF + "itemref"):
        href = hrefs.get(itemref.get("idref"))
        if href is None:
            raise Unreadable(f"the spine of {package} names {itemref.get('idref')!r}, which its manifest does not")
        body = part(archive, resolve(package, href)).find(XHTML + "body")
        if body is None:
            raise Unreadable(f"{href} has no body")
        lines.append(" ".join(piece.strip() for piece in body.itertext() if piece.strip()))
    return lines


def package_text(data: bytes) -> str:
    """The text of an Office, OpenDocument or EPUB file, by following its package."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise Unreadable(f"not a zip: {error}") from None
    with archive:
        if "mimetype" in archive.namelist():
            kind = archive.read("mimetype").decode("ascii", "replace")
            if kind == "application/epub+zip":
                return "\n".join(epub_text(archive))
            if kind.startswith("application/vnd.oasis.opendocument."):
                return "\n".join(odf_text(archive, kind))
            raise Unreadable(f"its mimetype is {kind!r}")
        if "[Content_Types].xml" in archive.namelist():
            return "\n".join(ooxml_text(archive))
    raise Unreadable("a zip that is no Office, OpenDocument or EPUB package")


def pdf_text(data: bytes) -> str:
    """The strings a PDF's inflatable streams show, one per line; a stream that is a JPEG, or encrypted, gives none."""
    lines = []
    for raw in STREAM.findall(data):
        try:
            content = zlib.decompress(raw)
        except zlib.error:
            continue
        lines += [re.sub(rb"\\(.)", rb"\1", shown).decode("latin-1") for shown in SHOWN.findall(content)]
    return "\n".join(lines)


def text_of(data: bytes) -> str:
    """The text a document shows: a PDF's, or a package's."""
    return pdf_text(data) if data.startswith(b"%PDF") else package_text(data)
