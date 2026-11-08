"""The index (DESIGN.md §5): its schema, the home's folders, and every read and write of `index.sqlite`.

Each thread gets its own connection, in WAL mode, so the capture loop, the ingest loop, the janitor, the API's threads
and any number of readers (the CLI, scripts) work side by side. Writes that touch more than one row run in one
`BEGIN IMMEDIATE`. Every read an agent can reach takes a `Scope` (§5.2).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import base64
import re
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from mms import media as sniff
from mms.config import Config, stamp
from mms.errors import ApiError
from mms.lock import HomeLock

SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, kind TEXT NOT NULL, t0 REAL NOT NULL, t1 REAL,
    path TEXT NOT NULL, bytes INTEGER NOT NULL DEFAULT 0, sha256 TEXT, name TEXT,
    keep INTEGER NOT NULL DEFAULT 0, in_place INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'ok', meta TEXT,
    project TEXT, session TEXT, expires REAL);
CREATE INDEX IF NOT EXISTS items_t0 ON items(t0);
CREATE INDEX IF NOT EXISTS items_source_t0 ON items(source, t0);
CREATE INDEX IF NOT EXISTS items_name ON items(source, name, t0);
CREATE INDEX IF NOT EXISTS items_path ON items(path);
CREATE INDEX IF NOT EXISTS items_expires ON items(expires) WHERE expires IS NOT NULL;
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, t REAL NOT NULL, data TEXT NOT NULL, project TEXT, session TEXT,
    expires REAL);
CREATE INDEX IF NOT EXISTS records_t ON records(t);
CREATE INDEX IF NOT EXISTS records_source_t ON records(source, t);
CREATE INDEX IF NOT EXISTS records_expires ON records(expires) WHERE expires IS NOT NULL;
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY, source TEXT, item INTEGER, t0 REAL NOT NULL, t1 REAL NOT NULL, by TEXT NOT NULL,
    text TEXT NOT NULL, meta TEXT, made REAL NOT NULL, project TEXT, session TEXT);
CREATE INDEX IF NOT EXISTS labels_t0 ON labels(t0);
CREATE TABLE IF NOT EXISTS clock (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, run TEXT NOT NULL, wall REAL NOT NULL, t REAL NOT NULL);
CREATE INDEX IF NOT EXISTS clock_wall ON clock(source, wall);
CREATE INDEX IF NOT EXISTS clock_run_t ON clock(source, run, t);
CREATE TABLE IF NOT EXISTS cursors (
    source TEXT NOT NULL, name TEXT NOT NULL, offset INTEGER NOT NULL DEFAULT 0, size INTEGER, mtime REAL,
    sha256 TEXT, PRIMARY KEY (source, name));
CREATE TABLE IF NOT EXISTS launches (
    id INTEGER PRIMARY KEY, source TEXT NOT NULL, folder TEXT NOT NULL, started REAL NOT NULL, ended REAL,
    lines INTEGER NOT NULL DEFAULT 0, first_start REAL);
CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT, root TEXT, seen REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, project TEXT, client TEXT, started REAL NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(data, content='records', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS labels_fts USING fts5(text, content='labels', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
    INSERT INTO records_fts(rowid, data) VALUES (new.id, new.data); END;
CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
    INSERT INTO records_fts(records_fts, rowid, data) VALUES ('delete', old.id, old.data); END;
CREATE TRIGGER IF NOT EXISTS labels_ai AFTER INSERT ON labels BEGIN
    INSERT INTO labels_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS labels_ad AFTER DELETE ON labels BEGIN
    INSERT INTO labels_fts(labels_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
"""

KINDS = ("video", "audio", "image", "file")
SPANS = ("video", "audio")  # the kinds that cover a stretch of time; an image is one instant, a file a version
STILL_SLACK_S = 0.001  # a time read back from a result is cut to the millisecond, so a picture this much after it is at it
INDEX_LOCK = "index.lock"  # taken while the index is created: never deleted, like daemon.lock
INDEX_WAIT_S = 10.0  # how long a process waits its turn to create the index
FOREVER = 1e18  # a version's end while it is current, for comparisons
CLOCK_COST = 100  # what a clock row costs a cap, in bytes (DESIGN.md §9)
GB = 1_000_000_000
RANK = {"label": 0, "record": 1, "item": 2}  # the order of rows that share a time, for find's cursor
# What a cap counts and may delete (DESIGN.md §9), for a `FROM items` query: an item that is not kept, not in place and
# not its source's newest. Thinning, dropping, ageing and the byte count all use this one rule.
RETAINABLE = ("keep = 0 AND in_place = 0 AND id != (SELECT n.id FROM items n WHERE n.source = items.source "
              "ORDER BY n.t0 DESC, n.id DESC LIMIT 1)")
# What a record costs a cap, in SQL: its UTF-8 text times 3, plus 200 bytes a row, for the table, its two indexes and
# the FTS5 index, as measured once a capped table has settled (DESIGN.md §9)
RECORD_COST = "(LENGTH(CAST(data AS BLOB)) * 3 + 200)"
# An item overlaps [lo, hi]: a segment that does, or a version that was current at some time in it (DESIGN.md §5.3).
# Its arguments are hi, lo, hi, lo.
OVERLAP = ("((kind != 'file' AND t0 <= ? AND COALESCE(t1, t0) >= ?) "
           f"OR (kind = 'file' AND t0 <= ? AND COALESCE(t1, {FOREVER!r}) >= ?))")


@dataclass(frozen=True)
class Limits:
    """What a write may not pass: the free-space floor, and each capped source's `max_gb` in bytes."""

    floor_bytes: float
    caps: dict[str, int]


class IndexVersionError(ValueError):
    """The index is from another version of MultimediaSearch."""


@dataclass(frozen=True)
class Scope:
    """Which rows a read may see (DESIGN.md §5.2): the ambient rows (no project) and the caller's project, or with
    `all` every row. A caller with no project sees the ambient rows only."""

    project: str | None = None
    all: bool = False

    def clause(self, column: str = "project") -> tuple[str, list]:
        """` AND ...` for a query, and its arguments."""
        if self.all:
            return "", []
        if self.project is None:
            return f" AND {column} IS NULL", []
        return f" AND ({column} IS NULL OR {column} = ?)", [self.project]


@dataclass
class Item:
    id: int
    source: str
    kind: str
    t0: float
    t1: float | None
    path: str
    bytes: int
    sha256: str | None = None
    name: str | None = None
    keep: int = 0
    in_place: int = 0
    state: str = "ok"
    meta: dict = field(default_factory=dict)
    project: str | None = None
    session: str | None = None
    expires: float | None = None


@dataclass
class Record:
    id: int
    source: str
    t: float
    data: str
    project: str | None = None
    session: str | None = None
    expires: float | None = None


@dataclass
class Label:
    id: int
    source: str | None
    item: int | None
    t0: float
    t1: float
    by: str
    text: str
    meta: dict
    made: float
    project: str | None = None
    session: str | None = None


def _item(row) -> Item:
    return Item(id=row["id"], source=row["source"], kind=row["kind"], t0=row["t0"], t1=row["t1"], path=row["path"],
                bytes=row["bytes"], sha256=row["sha256"], name=row["name"], keep=row["keep"],
                in_place=row["in_place"], state=row["state"], meta=json.loads(row["meta"]) if row["meta"] else {},
                project=row["project"], session=row["session"], expires=row["expires"])


def _record(row) -> Record:
    return Record(row["id"], row["source"], row["t"], row["data"], row["project"], row["session"], row["expires"])


def _label(row) -> Label:
    return Label(id=row["id"], source=row["source"], item=row["item"], t0=row["t0"], t1=row["t1"], by=row["by"],
                 text=row["text"], meta=json.loads(row["meta"]) if row["meta"] else {}, made=row["made"],
                 project=row["project"], session=row["session"])


def file_name(name: str | None, default: str) -> str:
    """The part of a stored file's path that comes from an item's name: the last component, with every character outside
    A-Z a-z 0-9 . _ - turned to `_` (so Windows can hold it), at most 80 long, `default` when nothing is left."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "").name)[:80] or default


def text_words(text: str | None) -> list[str]:
    """The words a `find` text is searched for: its runs of letters and digits. Punctuation is no word."""
    return re.findall(r"\w+", text or "")


def item_matches(item: Item, words: list[str]) -> bool:
    """Whether a version's name holds every word (all items, with no words)."""
    return not words or bool(item.name and all(word.lower() in item.name.lower() for word in words))


def cursor_key(row: dict) -> tuple[float, int, int]:
    """Where a row sits in `find`'s order: newest first by time, then type, then id."""
    return row["t0"], RANK[row["type"]], row["id"]


def encode_cursor(row: dict) -> str:
    """An opaque string for `find` to hand back as `after`: the row's exact time, type and id."""
    return base64.urlsafe_b64encode(json.dumps(cursor_key(row)).encode()).decode().rstrip("=")


def decode_cursor(text: str) -> tuple[float, int, int]:
    """The key an `encode_cursor` string holds; `ValueError` for any other."""
    try:
        t0, rank, ident = json.loads(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)))
        if not (isinstance(t0, (int, float)) and rank in RANK.values() and isinstance(ident, int)):
            raise ValueError
        return float(t0), rank, ident
    except (ValueError, TypeError, UnicodeDecodeError):
        raise ValueError("is not a cursor this daemon gave") from None


def _after_sql(time_column: str, id_column: str, rank: int, after) -> tuple[str, list]:
    """` AND ...` for the rows of one table that come after the cursor in `find`'s order."""
    if after is None:
        return "", []
    t0, after_rank, after_id = after
    if rank < after_rank:
        return f" AND {time_column} <= ?", [t0]
    if rank == after_rank:
        return f" AND ({time_column} < ? OR ({time_column} = ? AND {id_column} < ?))", [t0, t0, after_id]
    return f" AND {time_column} < ?", [t0]


def _in(column: str, values) -> tuple[str, list]:
    """` AND column IN (?, ...)` for a non-empty list, else nothing."""
    if not values:
        return "", []
    values = list(values)
    return f" AND {column} IN ({', '.join('?' * len(values))})", values


class Store:
    def __init__(self, home: Path | str):
        self.home = Path(home)
        self.streams = self.home / "streams"
        self.files = self.home / "files"
        for folder in (self.home, self.streams, self.files, self.home / "scripts"):
            folder.mkdir(parents=True, exist_ok=True)
        self.file = self.home / "index.sqlite"
        self._local = threading.local()
        self.limits: Limits | None = None  # none until a config is set: nothing is gated
        self._seen: set[tuple] = set()  # callers already written down, so a call costs no write
        self._create_index()

    def _create_index(self) -> None:
        """Switch the file to WAL and make the tables. Any process may open a new home at once (the daemon, `mms status`,
        `mms remember`), and the switch needs an exclusive lock that the busy timeout does not wait for, so they take
        turns on `index.lock`. WAL stays in the file, so only a new home pays for the switch. An index of another schema
        version is refused: there is no migration (DESIGN.md §5.1)."""
        lock = HomeLock(self.home / INDEX_LOCK)
        if not lock.acquire(INDEX_WAIT_S):
            raise sqlite3.OperationalError(f"another process kept {self.home / INDEX_LOCK} for {INDEX_WAIT_S} s")
        try:
            db = self.db()
            made = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'items'").fetchone()
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if made and version != SCHEMA_VERSION:
                raise IndexVersionError(f"{self.file} is from another version of MultimediaSearch (schema {version}, this one reads "
                                 f"{SCHEMA_VERSION}); move the home aside and start a new one")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        finally:
            lock.release()

    # connections and transactions

    def db(self) -> sqlite3.Connection:
        db = getattr(self._local, "db", None)
        if db is None:
            db = sqlite3.connect(self.file, timeout=30, isolation_level=None, check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=NORMAL")
            self._local.db = db
        return db

    @contextmanager
    def tx(self):
        db = self.db()
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
        except BaseException:
            db.execute("ROLLBACK")
            raise
        db.execute("COMMIT")

    def close(self) -> None:
        db = getattr(self._local, "db", None)
        if db is not None:
            db.close()
            self._local.db = None

    # paths

    def path_of(self, item: Item) -> Path:
        """An item's file: the owner's own path when indexed in place, else under the home."""
        return Path(item.path) if item.in_place else self.home / item.path

    def relative(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.home.resolve()).as_posix()

    def read_bytes(self, item: Item) -> bytes:
        """A version's content, decompressed when it was stored with zstd."""
        data = self.path_of(item).read_bytes()
        if item.path.endswith(".zst") and not item.in_place:
            import zstandard
            data = zstandard.ZstdDecompressor().decompress(data, max_output_size=1 << 31)
        return data

    # may I write

    def set_limits(self, config: Config) -> None:
        """The floor and the caps `reserve` checks, from the config in force."""
        self.limits = Limits(config.free_gb * GB, {name: int(source.keep.max_gb * GB)
                                                    for name, source in config.sources.items()
                                                    if source.keep.max_gb is not None})

    def reserve(self, n: int, source: str | None = None) -> None:
        """The one gate every write passes: `n` more bytes may not take free space below the floor (`floor`), and one
        write may not be larger than its source's whole cap (`too_big`), since the cap would delete it at once."""
        limits = self.limits
        if limits is None:
            return
        free = shutil.disk_usage(self.home).free
        if free - n < limits.floor_bytes:
            raise ApiError("floor", f"free space is {free / GB:.1f} GB and this write is {n / GB:.2f} GB: it would go "
                                    f"below the floor of {limits.floor_bytes / GB:g} GB", hint="ask the owner")
        cap = limits.caps.get(source)
        if cap is not None and n > cap:
            raise ApiError("too_big", f"this write is {n:,} bytes, more than the whole cap of source {source!r} "
                                      f"({cap:,} bytes)", hint="ask the owner to raise max_gb")

    # who is calling

    def note_caller(self, caller: dict) -> None:
        """Record a caller's project (its name and root) and session (its client) the first time they are seen."""
        project, session = caller.get("project"), caller.get("session")
        key = (project, session)
        if (project is None and session is None) or key in self._seen:
            return
        with self.tx() as db:
            if project is not None:
                db.execute("INSERT OR IGNORE INTO projects (id, name, root, seen) VALUES (?, ?, ?, ?)",
                           (project, caller.get("project_name"), caller.get("project_root"), time.time()))
            if session is not None:
                db.execute("INSERT OR IGNORE INTO sessions (id, project, client, started) VALUES (?, ?, ?, ?)",
                           (session, project, caller.get("client"), time.time()))
        self._seen.add(key)

    def project_name(self, project: str | None) -> str | None:
        if project is None:
            return None
        row = self.db().execute("SELECT name FROM projects WHERE id = ?", (project,)).fetchone()
        return row[0] if row and row[0] else project

    # writes

    def add_item(self, source: str, kind: str, t0: float, t1: float | None, path: str, bytes: int, *,
                 sha256: str | None = None, name: str | None = None, meta: dict | None = None, keep: int = 0,
                 in_place: bool = False, project: str | None = None, session: str | None = None,
                 expires: float | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"an item's kind is one of {', '.join(KINDS)}, not {kind!r}")
        cursor = self.db().execute(
            "INSERT INTO items (source, kind, t0, t1, path, bytes, sha256, name, keep, in_place, meta, project, session, "
            "expires) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source, kind, t0, t1, path, bytes, sha256, name, keep, int(in_place), json.dumps(meta) if meta else None,
             project, session, expires))
        return cursor.lastrowid

    def add_records(self, source: str, rows, cursor: tuple[str, dict] | None = None, *, project: str | None = None,
                    session: str | None = None, expires: float | None = None,
                    clock: tuple[str, float, float] | None = None) -> int:
        """Records from `(t, data)` pairs, where data is JSON text. Returns how many were added. `cursor` is
        `(name, fields)` for `set_cursor`, and `clock` is `(run, wall, t)` for a clock row; both are set in the same
        transaction, so a tail never re-reads what it committed."""
        rows = [(source, float(t), data, project, session, expires) for t, data in rows]
        if not rows and cursor is None and clock is None:
            return 0
        self.reserve(sum(len(row[2].encode("utf-8")) for row in rows), source)
        with self.tx() as db:
            db.executemany("INSERT INTO records (source, t, data, project, session, expires) VALUES (?, ?, ?, ?, ?, ?)",
                           rows)
            if clock is not None:
                db.execute("INSERT INTO clock (source, run, wall, t) VALUES (?, ?, ?, ?)", (source, *clock))
            if cursor is not None:
                self._put_cursor(db, source, cursor[0], cursor[1])
        return len(rows)

    def add_record(self, source: str, t: float, data: str, *, project: str | None = None, session: str | None = None,
                   expires: float | None = None) -> int:
        """One record, as `add_records` adds them; returns its id."""
        self.reserve(len(data.encode("utf-8")), source)
        with self.tx() as db:
            return db.execute(
                "INSERT INTO records (source, t, data, project, session, expires) VALUES (?, ?, ?, ?, ?, ?)",
                (source, float(t), data, project, session, expires)).lastrowid

    def current_version(self, source: str, name: str) -> Item | None:
        row = self.db().execute(
            "SELECT * FROM items WHERE source = ? AND name = ? AND kind = 'file' AND t1 IS NULL "
            "ORDER BY t0 DESC LIMIT 1", (source, name)).fetchone()
        return _item(row) if row else None

    def _copy_in(self, source: str, name: str, data: bytes | None, src: Path | None, compress: bool):
        """Stream content into a partial file under `files/<source>/` while hashing it: `(partial, sha256, size, base)`."""
        self.reserve(len(data) if src is None else Path(src).stat().st_size, source)
        base = file_name(name, "item")
        folder = self.files / source
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / f".{os.getpid()}-{threading.get_ident()}-{base}.part"
        digest, size = hashlib.sha256(), 0
        try:
            with (open(src, "rb") if src is not None else io.BytesIO(data)) as reader, open(partial, "wb") as out:
                if compress:
                    import zstandard
                    sink = zstandard.ZstdCompressor(level=10).stream_writer(out, closefd=False)
                else:
                    sink = out
                while chunk := reader.read(1 << 20):
                    digest.update(chunk)
                    size += len(chunk)
                    sink.write(chunk)
                if compress:
                    sink.close()
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return partial, digest.hexdigest(), size, base

    def add_version(self, source: str, name: str, t: float, *, data: bytes | None = None, src: Path | None = None,
                    project: str | None = None, session: str | None = None,
                    expires: float | None = None) -> int | None:
        """A new version of `name`, from `data` or the file `src`; None when its content equals the current version's.
        What it is comes from its bytes (media.py). It closes the current version at `t`. The content is read once,
        streamed into the home while it is hashed (text compressed with zstd), and the current version is checked and
        replaced in one transaction, so the bytes stored are the bytes hashed and two writers never leave two open
        versions."""
        if (data is None) == (src is None):
            raise ValueError("a version comes from data or from a file, not both or neither")
        facts = sniff.of_bytes(data) if src is None else sniff.of_file(src)
        text = sniff.is_text(facts["media"])
        partial, sha, size, base = self._copy_in(source, name, data, src, text)
        try:
            target = self.files / source / f"{stamp(t)}-{sha[:8]}-{base}{'.zst' if text else ''}"
            with self.tx() as db:
                current = self.current_version(source, name)
                if current is not None and current.sha256 == sha:
                    return None
                if current is not None:
                    t = max(t, current.t0)  # a writer that started first but committed second still comes after
                self._close(db, current, t)
                added = db.execute(
                    "INSERT INTO items (source, kind, t0, t1, path, bytes, sha256, name, in_place, meta, project, "
                    "session, expires) VALUES (?, 'file', ?, NULL, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
                    (source, t, self.relative(target), partial.stat().st_size, sha, name,
                     json.dumps({**facts, "size": size}), project, session, expires)).lastrowid
                os.replace(partial, target)
                return added
        finally:
            partial.unlink(missing_ok=True)

    def add_recording(self, source: str, kind: str, name: str, src: Path, t0: float, t1: float, media: str, *,
                      sound: bool = False, time_from: str | None = None, project: str | None = None, session: str | None = None,
                      expires: float | None = None) -> tuple[int, bool]:
        """A recording copied into `files/<source>/` and indexed as a segment of `kind` (video or audio) over
        [t0, t1), or a picture (`image`) at the instant t0 = t1: `(id, True)`, or `(id, False)` of the one the source
        already holds with the same SHA-256. `media` is what its caller found it to be, and `sound` says a video has an
        audio stream. `time_from` (where `t0` came from, DESIGN.md §7) is kept in `meta`."""
        meta = {"media": media, **({"sound": True} if sound else {}), **({"time_from": time_from} if time_from else {})}
        partial, sha, size, base = self._copy_in(source, name, None, src, False)
        try:
            target = self.files / source / f"{stamp(t0)}-{sha[:8]}-{base}"
            with self.tx() as db:
                same = db.execute("SELECT id FROM items WHERE source = ? AND sha256 = ? AND kind != 'file'",
                                  (source, sha)).fetchone()
                if same:
                    return same[0], False
                added = db.execute(
                    "INSERT INTO items (source, kind, t0, t1, path, bytes, sha256, name, in_place, meta, project, "
                    "session, expires) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
                    (source, kind, t0, t1, self.relative(target), size, sha, name,
                     json.dumps(meta), project, session, expires)).lastrowid
                os.replace(partial, target)
                return added, True
        finally:
            partial.unlink(missing_ok=True)

    def index_in_place(self, source: str, name: str, path: Path, t: float, size: int, *,
                       project: str | None = None) -> int:
        """`name` is a file too big to copy, indexed where it is. It has no history: one row, with `t0` the last change
        seen (the caller compares size and mtime; nothing is hashed). A copied version it follows is closed at `t`."""
        path = Path(path).resolve()
        meta = json.dumps({**sniff.of_file(path), "size": size})
        with self.tx() as db:
            current = self.current_version(source, name)
            if current is not None and current.in_place and current.path == str(path):
                db.execute("UPDATE items SET t0 = ?, bytes = ?, meta = ? WHERE id = ?", (t, size, meta, current.id))
                return current.id
            self._close(db, current, t)
            return db.execute(
                "INSERT INTO items (source, kind, t0, t1, path, bytes, sha256, name, in_place, meta, project) "
                "VALUES (?, 'file', ?, NULL, ?, ?, NULL, ?, 1, ?, ?)",
                (source, t, str(path), size, name, meta, project)).lastrowid

    @staticmethod
    def _close(db, current: Item | None, t: float) -> None:
        """End a current version at `t`. One in place cannot be read as it was, so it goes `gone` with it, and nothing
        reads the live path for an old time."""
        if current is not None:
            db.execute("UPDATE items SET t1 = ?, state = CASE WHEN in_place = 1 THEN 'gone' ELSE state END "
                       "WHERE id = ?", (t, current.id))

    def end_version(self, source: str, name: str, t: float) -> bool:
        """The file behind `name` vanished at `t`: its current version ends there."""
        with self.tx() as db:
            current = self.current_version(source, name)
            self._close(db, current, t)
        return current is not None

    def label(self, t0: float, t1: float, text: str, by: str, *, source: str | None = None, item: int | None = None,
              keep: bool = False, meta: dict | None = None, made: float | None = None, project: str | None = None,
              session: str | None = None, max_gb: float | None = None) -> int:
        """Text on a time range. With `keep`, the items of `source` it overlaps, and `item`, are kept for good; keep
        needs one of the two. `max_gb` is the source's cap: the check that the kept bytes stay under it and the mark
        are one transaction (DESIGN.md §5.1), so two callers cannot together pass it."""
        if keep and source is None and item is None:
            raise ValueError("keep names the source whose items it keeps (or an item)")
        with self.tx() as db:
            if keep and source is not None:
                self._check_keep(db, source, t0, t1, max_gb)
            cursor = db.execute(
                "INSERT INTO labels (source, item, t0, t1, by, text, meta, made, project, session) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source, item, t0, t1, by, text, json.dumps(meta) if meta else None, made or time.time(), project,
                 session))
            if keep and source is not None:
                db.execute(f"UPDATE items SET keep = 1 WHERE source = ? AND {OVERLAP}", (source, t1, t0, t1, t0))
            if keep and item is not None:
                db.execute("UPDATE items SET keep = 1 WHERE id = ?", (item,))
            return cursor.lastrowid

    def set_keep(self, source: str, t0: float, t1: float, keep: bool = True, max_gb: float | None = None) -> int:
        """Keep, or with `keep=False` let go, the items of a source that overlap a range. Returns how many. Keeping
        checks `max_gb` (the source's cap) in the same transaction as the mark; letting go is never refused."""
        with self.tx() as db:
            if keep:
                self._check_keep(db, source, t0, t1, max_gb)
            return db.execute(f"UPDATE items SET keep = ? WHERE source = ? AND {OVERLAP}",
                              (int(keep), source, t1, t0, t1, t0)).rowcount

    @staticmethod
    def _check_keep(db, source: str, t0: float, t1: float, max_gb: float | None) -> None:
        if max_gb is None:
            return
        kept = db.execute("SELECT COALESCE(SUM(bytes), 0) FROM items WHERE source = ? AND keep = 1 AND in_place = 0",
                          (source,)).fetchone()[0]
        added = db.execute(f"SELECT COALESCE(SUM(bytes), 0) FROM items WHERE source = ? AND keep = 0 AND in_place = 0 "
                           f"AND {OVERLAP}", (source, t1, t0, t1, t0)).fetchone()[0]
        limit = int(max_gb * GB)
        if kept + added > limit:
            raise ApiError("too_big", f"source {source!r} keeps {kept:,} bytes; keeping this range adds {added:,}, past "
                                      f"its max_gb of {max_gb:g} ({limit:,} bytes)", field="keep",
                           hint="ask the owner to raise max_gb, or keep less")

    def count_kept(self, source: str, t0: float, t1: float) -> int:
        """How many of the source's items a range overlaps that are kept."""
        return self.db().execute(f"SELECT COUNT(*) FROM items WHERE source = ? AND keep = 1 AND {OVERLAP}",
                                 (source, t1, t0, t1, t0)).fetchone()[0]

    def delete_item(self, item_id: int) -> bool:
        """Unlink an item's file, then delete its row. False when the file could not be unlinked (a reader has it open
        on Windows): the row stays for the next try. The owner's in-place files are never unlinked, and a file another
        item still names is kept."""
        item = self.item(item_id)
        if item is None:
            return True
        shared = self.db().execute("SELECT COUNT(*) FROM items WHERE path = ? AND id != ?",
                                   (item.path, item.id)).fetchone()[0]
        if not item.in_place and not shared:
            try:
                self.path_of(item).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return False
        with self.tx() as db:  # a label on an item goes with it; a label of no item is a note, and stays
            db.execute("DELETE FROM labels WHERE item = ?", (item.id,))
            db.execute("DELETE FROM items WHERE id = ?", (item.id,))
        return True

    def delete_records(self, ids) -> int:
        ids = list(ids)
        if not ids:
            return 0
        with self.tx() as db:
            db.executemany("DELETE FROM records WHERE id = ?", [(i,) for i in ids])
        return len(ids)

    # reads

    def item(self, item_id: int) -> Item | None:
        row = self.db().execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return _item(row) if row else None

    def has_path(self, path: str) -> bool:
        """Whether an item is stored at `path` (relative to the home)."""
        return self.db().execute("SELECT 1 FROM items WHERE path = ?", (path,)).fetchone() is not None

    def select(self, since: float | None = None, until: float | None = None, sources=None, kinds=None,
               newest_first: bool = False, changed: bool = False, scope: Scope | None = None) -> list[Item]:
        """Items in a range by event time (DESIGN.md §5.3): segments that overlap it, and the versions that were
        current at some time in it (as of). With `changed`, only the versions whose `t0` falls in it. `scope` None is
        every row, for the daemon's own loops."""
        lo = -FOREVER if since is None else since
        hi = FOREVER if until is None else until
        src, src_args = _in("source", sources)
        kind, kind_args = _in("kind", kinds)
        shown, shown_args = scope.clause() if scope else ("", [])
        order = "DESC" if newest_first else "ASC"
        version = "t0 >= ? AND t0 < ?" if changed else f"t0 < ? AND COALESCE(t1, {FOREVER!r}) > ?"
        rows = self.db().execute(
            f"SELECT * FROM items WHERE state = 'ok' AND ("
            f"(kind != 'file' AND t0 < ? AND COALESCE(t1, t0) >= ?) OR (kind = 'file' AND {version}))"
            f"{src}{kind}{shown} ORDER BY t0 {order}, id {order}",
            [hi, lo, *([lo, hi] if changed else [hi, lo]), *src_args, *kind_args, *shown_args]).fetchall()
        return [_item(row) for row in rows]

    def as_of(self, t: float, sources=None, scope: Scope | None = None) -> list[Item]:
        """The version of each name that was current at `t`."""
        src, args = _in("source", sources)
        shown, shown_args = scope.clause() if scope else ("", [])
        rows = self.db().execute(
            f"SELECT * FROM items WHERE kind = 'file' AND state = 'ok' AND t0 <= ? AND COALESCE(t1, ?) > ?"
            f"{src}{shown} ORDER BY source, name", [t, FOREVER, t, *args, *shown_args]).fetchall()
        return [_item(row) for row in rows]

    def latest_still(self, source: str, t: float, since: float | None = None,
                     scope: Scope | None = None) -> Item | None:
        """The source's latest picture at or before `t` (and not before `since`), else None."""
        shown, shown_args = scope.clause() if scope else ("", [])
        row = self.db().execute(
            f"SELECT * FROM items WHERE source = ? AND kind = 'image' AND state = 'ok' AND t0 < ? AND t0 >= ?{shown} "
            f"ORDER BY t0 DESC, id DESC LIMIT 1",
            [source, t + STILL_SLACK_S, -FOREVER if since is None else since, *shown_args]).fetchone()
        return _item(row) if row else None

    def records(self, since: float | None = None, until: float | None = None, sources=None,
                newest_first: bool = False, limit: int | None = None, scope: Scope | None = None) -> list[Record]:
        lo = -FOREVER if since is None else since
        hi = FOREVER if until is None else until
        src, args = _in("source", sources)
        shown, shown_args = scope.clause() if scope else ("", [])
        order = "DESC" if newest_first else "ASC"
        cap = f" LIMIT {int(limit)}" if limit else ""
        rows = self.db().execute(
            f"SELECT * FROM records WHERE t >= ? AND t < ?{src}{shown} ORDER BY t {order}, id {order}{cap}",
            [lo, hi, *args, *shown_args]).fetchall()
        return [_record(row) for row in rows]

    def record_size(self, since: float | None, until: float | None, sources, scope: Scope) -> tuple[int, int]:
        """How many records a range holds and their characters, for an estimate."""
        src, src_args = _in("source", sources)
        shown, shown_args = scope.clause()
        row = self.db().execute(
            f"SELECT COUNT(*), COALESCE(SUM(LENGTH(data)), 0) FROM records WHERE t >= ? AND t < ?{src}{shown}",
            [-FOREVER if since is None else since, FOREVER if until is None else until, *src_args,
             *shown_args]).fetchone()
        return row[0], row[1]

    def labels(self, since: float | None = None, until: float | None = None, sources=None, by=None,
               scope: Scope | None = None, limit: int | None = None) -> list[Label]:
        lo = -FOREVER if since is None else since
        hi = FOREVER if until is None else until
        src, src_args = _in("source", sources)
        who, by_args = _in("by", [by] if isinstance(by, str) else by)
        shown, shown_args = scope.clause() if scope else ("", [])
        cap = f" LIMIT {int(limit)}" if limit else ""
        rows = self.db().execute(
            f"SELECT * FROM labels WHERE t0 < ? AND t1 >= ?{src}{who}{shown} ORDER BY t0 DESC, id DESC{cap}",
            [hi, lo, *src_args, *by_args, *shown_args]).fetchall()
        return [_label(row) for row in rows]

    def find(self, since: float | None = None, until: float | None = None, sources=None, text: str | None = None,
             limit: int = 100, scope: Scope | None = None, types=("item", "record", "label"),
             after: tuple[float, int, int] | None = None) -> list[dict]:
        """What a range holds, newest first by `(t0, type, id)`, as plain dicts with a `type` of item, record or label.
        With `text`, only what matches it: FTS5 over records and labels, and the names of versions. `after` is the key
        of the last row of the page before (`cursor_key`): the rows that come after it in that order."""
        lo = -FOREVER if since is None else since
        hi = FOREVER if until is None else until
        src, src_args = _in("source", sources)
        words = text_words(text)
        query = " ".join(f'"{word}"' for word in words)
        db = self.db()
        found: list[dict] = []
        if "record" in types:
            skip, skip_args = _after_sql("r.t", "r.id", RANK["record"], after)
            shown, shown_args = scope.clause("r.project") if scope else ("", [])
            join = "records_fts f JOIN records r ON r.id = f.rowid WHERE records_fts MATCH ? AND" if words else \
                "records r WHERE"
            rows = db.execute(
                f"SELECT r.* FROM {join} r.t >= ? AND r.t < ?{src.replace('source', 'r.source')}{shown}{skip} "
                f"ORDER BY r.t DESC, r.id DESC LIMIT ?",
                [*([query] if words else []), lo, hi, *src_args, *shown_args, *skip_args, limit]).fetchall()
            found += [_record_dict(_record(row)) for row in rows]
        if "label" in types:
            skip, skip_args = _after_sql("l.t0", "l.id", RANK["label"], after)
            shown, shown_args = scope.clause("l.project") if scope else ("", [])
            join = "labels_fts f JOIN labels l ON l.id = f.rowid WHERE labels_fts MATCH ? AND" if words else \
                "labels l WHERE"
            rows = db.execute(
                f"SELECT l.* FROM {join} l.t0 < ? AND l.t1 >= ?{src.replace('source', 'l.source')}{shown}{skip} "
                f"ORDER BY l.t0 DESC, l.id DESC LIMIT ?",
                [*([query] if words else []), hi, lo, *src_args, *shown_args, *skip_args, limit]).fetchall()
            found += [_label_dict(_label(row)) for row in rows]
        if "item" in types:
            items = self.select(since, until, sources, newest_first=True, changed=True, scope=scope)
            found += [item_dict(item) for item in items if item_matches(item, words)
                      and (after is None or cursor_key({"t0": item.t0, "type": "item", "id": item.id}) < after)][:limit]
        found.sort(key=cursor_key, reverse=True)
        return found[:limit]

    def source_names(self, scope: Scope) -> list[str]:
        """Every source with an item or a record the scope can see."""
        shown, args = scope.clause()
        rows = self.db().execute(f"SELECT source FROM items WHERE state = 'ok'{shown} UNION "
                                 f"SELECT source FROM records WHERE 1 = 1{shown}", [*args, *args]).fetchall()
        return sorted(row[0] for row in rows)

    def nearest(self, source: str, t: float) -> tuple[tuple[int, float] | None, float | None]:
        """What a source holds nearest `t`, whatever the scope: the nearest ok item's (id, t0) and the nearest record's t,
        None for one it has none of."""
        db = self.db()
        item = db.execute("SELECT id, t0 FROM items WHERE source = ? AND state = 'ok' ORDER BY ABS(t0 - ?) LIMIT 1",
                          (source, t)).fetchone()
        record = db.execute("SELECT t FROM records WHERE source = ? ORDER BY ABS(t - ?) LIMIT 1", (source, t)).fetchone()
        return (item["id"], item["t0"]) if item else None, record["t"] if record else None

    def items_of(self, source: str) -> list[Item]:
        """Every item of a source, oldest first."""
        rows = self.db().execute("SELECT * FROM items WHERE source = ? ORDER BY t0, id", (source,)).fetchall()
        return [_item(row) for row in rows]

    def retainable_items(self, source: str, before: float = FOREVER) -> list[Item]:
        """The items of a source that caps count and may delete (RETAINABLE), oldest first, starting before `before`."""
        rows = self.db().execute(f"SELECT * FROM items WHERE source = ? AND t0 < ? AND {RETAINABLE} ORDER BY t0, id",
                                 (source, before)).fetchall()
        return [_item(row) for row in rows]

    def expired_items(self, source: str, cutoff: float) -> list[Item]:
        """Retainable items that ended before `cutoff`, oldest first: a segment at its end, a version when a newer one
        or the file's vanishing closed it. A current version has not ended."""
        rows = self.db().execute(
            f"SELECT * FROM items WHERE source = ? AND ((kind != 'file' AND COALESCE(t1, t0) < ?) "
            f"OR (kind = 'file' AND t1 IS NOT NULL AND t1 < ?)) AND {RETAINABLE} ORDER BY t0, id",
            (source, cutoff, cutoff)).fetchall()
        return [_item(row) for row in rows]

    def holds_before(self, source: str, cutoff: float) -> bool:
        """Whether `max_days` would delete anything of the source: an item that ended, or a record, before `cutoff`."""
        return bool(self.expired_items(source, cutoff)) or bool(self.db().execute(
            "SELECT 1 FROM records WHERE source = ? AND t < ? LIMIT 1", (source, cutoff)).fetchone())

    def expired_by_date(self, now: float) -> tuple[list[Item], list[int]]:
        """Items and record ids past their `expires`; an item with `keep` is exempt."""
        db = self.db()
        items = [_item(row) for row in db.execute(
            "SELECT * FROM items WHERE expires IS NOT NULL AND expires <= ? AND keep = 0 ORDER BY id", (now,))]
        records = [row[0] for row in db.execute(
            "SELECT id FROM records WHERE expires IS NOT NULL AND expires <= ?", (now,))]
        return items, records

    def cap_cutoff(self, source: str, over: int) -> float:
        """The time before which a source's retainable items, records and clock rows, together, cost at least `over`
        bytes: a running sum by time. When all of them cost less, FOREVER."""
        row = self.db().execute(
            f"WITH rows AS (SELECT t0 AS t, bytes AS cost FROM items WHERE source = ? AND {RETAINABLE} "
            f"UNION ALL SELECT t, {RECORD_COST} FROM records WHERE source = ? "
            f"UNION ALL SELECT wall, {CLOCK_COST} FROM clock WHERE source = ?), "
            f"run AS (SELECT t, SUM(cost) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING) AS total FROM rows) "
            f"SELECT MIN(t) FROM run WHERE total >= ?", (source, source, source, over)).fetchone()
        return FOREVER if row[0] is None else math.nextafter(row[0], math.inf)

    def delete_records_before(self, source: str, cutoff: float) -> int:
        """Delete a source's records and clock rows before `cutoff`; returns how many records."""
        with self.tx() as db:
            db.execute("DELETE FROM clock WHERE source = ? AND wall < ?", (source, cutoff))
            return db.execute("DELETE FROM records WHERE source = ? AND t < ?", (source, cutoff)).rowcount

    def source_bytes(self) -> dict[str, int]:
        """What each source costs against its cap: its retainable items' files, its records (RECORD_COST) and its
        clock rows (CLOCK_COST)."""
        totals: dict[str, int] = {}
        for row in self.db().execute(f"SELECT source, SUM(bytes) FROM items WHERE {RETAINABLE} GROUP BY source"):
            totals[row[0]] = totals.get(row[0], 0) + (row[1] or 0)
        for row in self.db().execute(f"SELECT source, SUM({RECORD_COST}) FROM records GROUP BY source"):
            totals[row[0]] = totals.get(row[0], 0) + (row[1] or 0)
        for row in self.db().execute(f"SELECT source, COUNT(*) * {CLOCK_COST} FROM clock GROUP BY source"):
            totals[row[0]] = totals.get(row[0], 0) + row[1]
        return totals

    # the clock of a program's runs (DESIGN.md §5.4)

    def has_clock(self, source: str) -> bool:
        return self.db().execute("SELECT 1 FROM clock WHERE source = ? LIMIT 1", (source,)).fetchone() is not None

    def program_time(self, source: str, wall: float) -> tuple[str, float] | None:
        """The run a wall time belongs to and the program's own time then: between the clock rows around `wall` by
        interpolation, else the nearest row carried on at the wall clock's rate. Two runs at once are told apart only by
        the nearer clock row."""
        db = self.db()
        before = db.execute("SELECT run, wall FROM clock WHERE source = ? AND wall <= ? ORDER BY wall DESC LIMIT 1",
                            (source, wall)).fetchone()
        after = db.execute("SELECT run, wall FROM clock WHERE source = ? AND wall > ? ORDER BY wall LIMIT 1",
                           (source, wall)).fetchone()
        if before is None and after is None:
            return None
        if before is None or (after is not None and after["wall"] - wall < wall - before["wall"]):
            run = after["run"]
        else:
            run = before["run"]
        before = db.execute("SELECT wall, t FROM clock WHERE source = ? AND run = ? AND wall <= ? "
                            "ORDER BY wall DESC LIMIT 1", (source, run, wall)).fetchone()
        after = db.execute("SELECT wall, t FROM clock WHERE source = ? AND run = ? AND wall > ? ORDER BY wall LIMIT 1",
                           (source, run, wall)).fetchone()
        return run, _along(wall, before, after, "wall", "t")

    def wall_time(self, source: str, run: str, tp: float) -> float | None:
        """The inverse of `program_time` for one run: the wall time at the program's own time `tp`."""
        db = self.db()
        before = db.execute("SELECT wall, t FROM clock WHERE source = ? AND run = ? AND t <= ? ORDER BY t DESC LIMIT 1",
                            (source, run, tp)).fetchone()
        after = db.execute("SELECT wall, t FROM clock WHERE source = ? AND run = ? AND t > ? ORDER BY t LIMIT 1",
                           (source, run, tp)).fetchone()
        if before is None and after is None:
            return None
        return _along(tp, before, after, "t", "wall")

    # the loops' own state

    def cursor(self, source: str, name: str) -> dict | None:
        row = self.db().execute("SELECT * FROM cursors WHERE source = ? AND name = ?", (source, name)).fetchone()
        return dict(row) if row else None

    def cursors(self, source: str) -> list[dict]:
        return [dict(row) for row in self.db().execute("SELECT * FROM cursors WHERE source = ?", (source,))]

    def set_cursor(self, source: str, name: str, **fields) -> None:
        self._put_cursor(self.db(), source, name, fields)

    @staticmethod
    def _put_cursor(db, source: str, name: str, fields: dict) -> None:
        allowed = {"offset", "size", "mtime", "sha256"}
        if set(fields) - allowed:
            raise ValueError(f"a cursor holds {sorted(allowed)}")
        db.execute("INSERT OR IGNORE INTO cursors (source, name) VALUES (?, ?)", (source, name))
        if fields:
            sets = ", ".join(f"{key} = ?" for key in fields)
            db.execute(f"UPDATE cursors SET {sets} WHERE source = ? AND name = ?", [*fields.values(), source, name])

    def drop_cursor(self, source: str, name: str) -> None:
        self.db().execute("DELETE FROM cursors WHERE source = ? AND name = ?", (source, name))

    def add_launch(self, source: str, folder: str, started: float) -> int:
        return self.db().execute("INSERT INTO launches (source, folder, started) VALUES (?, ?, ?)",
                                 (source, folder, started)).lastrowid

    def update_launch(self, launch_id: int, **fields) -> None:
        allowed = {"ended", "lines", "first_start"}
        if set(fields) - allowed:
            raise ValueError(f"a launch holds {sorted(allowed)}")
        sets = ", ".join(f"{key} = ?" for key in fields)
        self.db().execute(f"UPDATE launches SET {sets} WHERE id = ?", [*fields.values(), launch_id])

    def launches(self, source: str | None = None) -> list[dict]:
        where, args = ("WHERE source = ?", [source]) if source else ("", [])
        return [dict(row) for row in self.db().execute(f"SELECT * FROM launches {where} ORDER BY id", args)]

    def status(self) -> dict:
        """Per source: items, records, the bytes they hold (a record at its RECORD_COST, a clock row at CLOCK_COST) and
        the newest time recorded."""
        out: dict[str, dict] = {}
        for row in self.db().execute("SELECT source, COUNT(*), SUM(bytes), MAX(COALESCE(t1, t0)) FROM items "
                                     "WHERE in_place = 0 GROUP BY source"):
            out.setdefault(row[0], {}).update(items=row[1], item_bytes=row[2] or 0, newest=row[3])
        for row in self.db().execute(
                f"SELECT source, COUNT(*), SUM({RECORD_COST}), MAX(t) FROM records GROUP BY source"):
            entry = out.setdefault(row[0], {})
            entry.update(records=row[1], record_bytes=row[2] or 0,
                         newest=max(filter(None, [entry.get("newest"), row[3]]), default=None))
        for row in self.db().execute(f"SELECT source, COUNT(*) * {CLOCK_COST} FROM clock GROUP BY source"):
            out.setdefault(row[0], {})["clock_bytes"] = row[1]
        return out


def _along(x: float, before, after, key: str, other: str) -> float:
    """`other` at `x` (measured in `key`) between two clock rows by interpolation, else carried on from the nearest at
    one second of `other` per second of `key`."""
    if before is not None and after is not None and after[key] > before[key]:
        return before[other] + (x - before[key]) * (after[other] - before[other]) / (after[key] - before[key])
    if before is not None:
        return before[other] + (x - before[key])
    return after[other] - (after[key] - x)


def item_dict(item: Item) -> dict:
    return {"type": "item", "id": item.id, "source": item.source, "kind": item.kind, "t0": item.t0, "t1": item.t1,
            "path": item.path, "bytes": item.bytes, "name": item.name, "keep": item.keep, "in_place": item.in_place,
            "meta": item.meta, "project": item.project}


def _record_dict(record: Record) -> dict:
    return {"type": "record", "id": record.id, "source": record.source, "t0": record.t, "t1": record.t,
            "data": record.data, "project": record.project}


def _label_dict(label: Label) -> dict:
    return {"type": "label", "id": label.id, "source": label.source, "item": label.item, "t0": label.t0,
            "t1": label.t1, "by": label.by, "text": label.text, "meta": label.meta, "project": label.project}
