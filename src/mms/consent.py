"""The owner's approvals (DESIGN.md §4.3): `consent.json` binds a source's name to a fingerprint of what it opens or
runs. `mms approve` is the only writer, once the owner has typed the name; nothing reachable through the API writes it.

    {"<source>": {"fingerprint": "<sha256>", "what": "microphone: Microphone (Realtek(R) Audio)", "approved": "<ISO>"}}

A leaf: capture and ingest pass `approved` what a source opens or runs (`Source.what()`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from mms.config import iso
from mms.lock import HomeLock

FILE = "consent.json"
LOCK = "consent.lock"  # every read and write holds it: Windows refuses a read while os.replace swaps the file, and
LOCK_WAIT_S = 10.0     # `mms approve` and the daemon's prune would otherwise lose each other's writes


class ConsentError(ValueError):
    """`consent.json` does not parse."""


def fingerprint(what: str) -> str:
    return hashlib.sha256(what.encode("utf-8")).hexdigest()


@contextmanager
def _held(home: Path | str):
    lock = HomeLock(Path(home) / LOCK)
    if not lock.acquire(LOCK_WAIT_S):
        raise TimeoutError(f"{Path(home) / LOCK} stayed busy for {LOCK_WAIT_S:g} s")
    try:
        yield
    finally:
        lock.release()


def load(home: Path | str) -> dict:
    if not Path(home).is_dir():
        return {}
    with _held(home):
        return _read(home)


def _read(home: Path | str) -> dict:
    file = Path(home) / FILE
    if not file.exists():
        return {}
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ConsentError(f"{file} does not parse: {error}") from None
    if not isinstance(data, dict) or not all(isinstance(entry, dict) for entry in data.values()):
        raise ConsentError(f"{file} is an object of source name to approval")
    return data


def approved(home: Path | str, name: str, what: str) -> bool:
    """Whether the owner approved this name for exactly this: a change to what it opens or runs needs a new approval."""
    entry = load(home).get(name)
    return entry is not None and entry.get("fingerprint") == fingerprint(what)


def _save(home: Path | str, data: dict) -> None:
    file = Path(home) / FILE
    part = file.with_name(FILE + ".part")
    part.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(part, file)


def record(home: Path | str, name: str, what: str) -> None:
    """Approve `name` for `what`. `mms approve` calls this once the owner has typed the name; tests call it directly."""
    with _held(home):
        data = _read(home)
        data[name] = {"fingerprint": fingerprint(what), "what": what, "approved": iso(time.time())}
        _save(home, data)


def prune(home: Path | str, names) -> list[str]:
    """Drop the approvals of names that are not in `names` (the sources that need one): a removed source takes its
    approval with it, so a later source of that name starts unapproved. Returns the names dropped."""
    with _held(home):
        data = _read(home)
        gone = [name for name in data if name not in names]
        if gone:
            _save(home, {name: entry for name, entry in data.items() if name in names})
    return gone


def revoke(home: Path | str, name: str) -> bool:
    """Remove an approval; False when there was none."""
    with _held(home):
        data = _read(home)
        if data.pop(name, None) is None:
            return False
        _save(home, data)
    return True
