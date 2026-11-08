"""Retention (DESIGN.md §9): the free-space floor, each source's cap and age limit, expiry, housekeeping and deletion of
every file and growing table under the home.

`Janitor.step(now)` is one pass and returns whether the floor is breached. It deletes whole items and rows, and only
retainable ones (`store.RETAINABLE`): never an item with `keep`, a file indexed in place, or the newest item of a
source. A file a reader has open cannot be unlinked on Windows: its row stays, and the next pass tries again.
"""

from __future__ import annotations

import heapq
import os
import shutil

from mms.config import Config, Source
from mms.store import GB, Store

RESUME_GB = 1  # capture resumes this far above the floor
DAY_S = 86400.0
STRAY_S = 3600.0  # a `.part`, `.ffconcat` or `.err` file this old is what a crash left, not a copy or an encode under way
STRAY_SUFFIXES = (".part", ".ffconcat", ".err")
STRAY_FOLDERS = ("streams", "files", "spool")
LOG_MAX_BYTES = 10_000_000  # daemon.log is cut here: one older file is kept, daemon.log.1
THIN_GRACE_S = 60.0  # added to an item's age when thinning, so the newest minute stays at full rate


def _noop(_: str) -> None:
    pass


def crowded(rows: list[tuple], newest: float, over: int) -> list[int]:
    """The ids to delete to free `over` bytes, from `rows` ordered oldest first as (id, t, bytes, deletable). It takes
    the row whose removal leaves the smallest gap between its neighbours, against its age plus THIN_GRACE_S, and so
    on, with the gaps closing as rows go. Rows that are not deletable stand in the gaps, and the oldest and the newest
    are never taken."""
    n = len(rows)
    before, after = list(range(-1, n - 1)), list(range(1, n + 1))
    version = [0] * n

    def score(i: int) -> float:
        return (rows[after[i]][1] - rows[before[i]][1]) / (newest - rows[i][1] + THIN_GRACE_S)

    heap = [(score(i), i, 0) for i in range(1, n - 1) if rows[i][3]]
    heapq.heapify(heap)

    def push(i: int) -> None:
        if rows[i][3] and 0 < i < n - 1 and before[i] >= 0 and after[i] < n:
            heapq.heappush(heap, (score(i), i, version[i]))

    chosen, freed = [], 0
    while heap and freed < over:
        _, i, seen = heapq.heappop(heap)
        if seen != version[i]:  # its neighbours changed since: it was pushed again
            continue
        chosen.append(rows[i][0])
        freed += rows[i][2]
        p, q = before[i], after[i]
        after[p], before[q] = q, p
        version[i] = -1
        for j in (p, q):
            version[j] += 1
            push(j)
    return chosen


class Janitor:
    def __init__(self, store: Store, config: Config, log=_noop):
        self.store = store
        self.config = config
        self.log = log
        self.breached = False
        self.free_gb: float | None = None

    def apply(self, config: Config) -> None:
        self.config = config

    def step(self, now: float) -> bool:
        """One pass: the floor, each source's age limit and cap, expiry and housekeeping. Returns whether the floor is
        breached."""
        self.check_floor()
        for source in self.config.sources.values():
            if source.keep.max_days is not None:
                self.expire(source, now - source.keep.max_days * 86400)
        sizes = self.store.source_bytes()
        for source in self.config.sources.values():
            if source.keep.max_gb is not None:
                over = sizes.get(source.name, 0) - int(source.keep.max_gb * GB)
                if over > 0:
                    self.shrink(source, over)
        self.expire_dates(now)
        self.housekeep(now)
        return self.breached

    def stop(self) -> None:
        pass

    # the floor

    def check_floor(self) -> bool:
        """Below `free_gb` the floor is breached, until free space is RESUME_GB above it."""
        free = shutil.disk_usage(self.store.home).free
        floor = self.config.free_gb * GB
        self.free_gb = free / GB
        was = self.breached
        if free < floor:
            self.breached = True
        elif free >= floor + RESUME_GB * GB:
            self.breached = False
        if self.breached != was:
            self.log(f"free space {self.free_gb:.1f} GB: floor {self.config.free_gb:g} GB "
                     + ("breached, capture and ingest paused" if self.breached
                        else "cleared, capture and ingest resume"))
        return self.breached

    # ages and caps

    def expire(self, source: Source, cutoff: float) -> None:
        gone = stuck = 0
        for item in self.store.expired_items(source.name, cutoff):
            if self.store.delete_item(item.id):
                gone += 1
            else:
                stuck += 1
        records = self.store.delete_records_before(source.name, cutoff)
        self.report(source, "older than max_days", gone, records, stuck)

    def shrink(self, source: Source, over: int) -> None:
        """Delete until `over` bytes are freed, by the source's `when_full`."""
        gone = stuck = 0
        if source.keep.when_full == "thin":
            freed, gone, stuck = self.thin(source, over)
            over -= freed
        records = 0
        if over > 0:  # drop_oldest, and what thinning could not reach
            more, records, more_stuck = self.drop_oldest(source, over)
            gone, stuck = gone + more, stuck + more_stuck
        self.report(source, f"over max_gb {source.keep.max_gb:g}", gone, records, stuck)

    def thin(self, source: Source, over: int) -> tuple[int, int, int]:
        """Thin the segments: returns the bytes freed, the items deleted and the deletes to retry."""
        segments = [item for item in self.store.items_of(source.name) if item.kind != "file"]
        if len(segments) < 3:
            return 0, 0, 0
        retainable = {item.id for item in self.store.retainable_items(source.name)}
        rows = [(item.id, item.t0, item.bytes, item.id in retainable) for item in segments]
        sizes = {item.id: item.bytes for item in segments}
        freed = gone = stuck = 0
        for item_id in crowded(rows, segments[-1].t0, over):
            if self.store.delete_item(item_id):
                freed += sizes[item_id]
                gone += 1
            else:
                stuck += 1
        return freed, gone, stuck

    def drop_oldest(self, source: Source, over: int) -> tuple[int, int, int]:
        """Delete the retainable items and the records before the one time that frees `over` bytes: returns the items
        and records deleted and the deletes to retry."""
        cutoff = self.store.cap_cutoff(source.name, over)
        gone = stuck = 0
        for item in self.store.retainable_items(source.name, before=cutoff):
            if self.store.delete_item(item.id):
                gone += 1
            else:
                stuck += 1
        return gone, self.store.delete_records_before(source.name, cutoff), stuck

    def expire_dates(self, now: float) -> None:
        """Items and records past the `expires` that `remember` set, unless kept."""
        items, records = self.store.expired_by_date(now)
        gone = stuck = 0
        for item in items:
            if self.store.delete_item(item.id):
                gone += 1
            else:
                stuck += 1
        deleted = self.store.delete_records(records)
        if gone or deleted:
            self.log(f"janitor: deleted {gone} items and {deleted} records past their expiry")
        if stuck:
            self.log(f"janitor: {stuck} expired files in use, tried again next pass")

    def housekeep(self, now: float) -> None:
        """`looks/` and `spool/_bad/` files older than a day, the stray files of a crash, and `daemon.log` over 10 MB."""
        home = self.store.home
        stuck = 0
        for folder in (home / "looks", home / "spool" / "_bad"):
            if folder.is_dir():
                stuck += sum(1 for entry in folder.iterdir() if self._remove_if_older(entry, now, DAY_S))
        for name in STRAY_FOLDERS:
            if (home / name).is_dir():
                stuck += sum(1 for entry in (home / name).rglob("*")
                             if entry.suffix in STRAY_SUFFIXES and self._remove_if_older(entry, now, STRAY_S))
        if stuck:
            self.log(f"janitor: {stuck} old files in use, tried again next pass")
        log = home / "daemon.log"
        try:
            if log.stat().st_size > LOG_MAX_BYTES:
                os.replace(log, home / "daemon.log.1")
        except FileNotFoundError:
            pass
        except OSError as error:
            self.log(f"janitor: cannot cut daemon.log: {error}")

    @staticmethod
    def _remove_if_older(entry, now: float, age: float) -> bool:
        """Delete a file older than `age`; True when it is in use (a reader has it open on Windows) and stays."""
        try:
            if entry.is_file() and now - entry.stat().st_mtime > age:
                entry.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return True
        return False

    def report(self, source: Source, why: str, items: int, records: int, stuck: int) -> None:
        if items or records:
            self.log(f"janitor {source.name}: deleted {items} items and {records} records, {why}")
        if stuck:
            self.log(f"janitor {source.name}: {stuck} files in use, tried again next pass")
