"""The project a program or agent works in (DESIGN.md §12.2): an ID that holds through renames, moves and clones.

It runs at most three `git` calls and caches nothing, so a folder that gains a remote or a `.mms-project` file is
seen at the next call.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MARKER = ".mms-project"
MARKER_ID = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
GIT_TIMEOUT_S = 10


@dataclass(frozen=True)
class Project:
    id: str  # `id:<value>`, `git:<16 hex>` or `remote:<16 hex>`; never a path
    name: str  # the git top level's folder name, else the folder's
    root: str  # for display only


def folder() -> Path:
    """The folder an agent works in: `CLAUDE_PROJECT_DIR` if set, else the working directory."""
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path.cwd())


def find(where: Path | str) -> Project | None:
    """The project of folder `where`, or None when it has none: a folder outside git with no marker, a repository
    with no commits, or a shallow clone with no `origin`. A bad `.mms-project`, or a git that fails for any other
    reason (an untrusted folder, say), raises."""
    where = Path(where).resolve()
    top, shallow = _repository(where)
    marked = _marker(where, top)
    root = top or where
    name = root.name or str(root)
    if marked is not None:
        return Project(f"id:{marked}", name, str(root))
    if top is None:
        return None
    if shallow:  # its root commit moves with its depth
        url = _git(top, "config", "--get", "remote.origin.url")
        if url.returncode == 1 and not url.stderr.strip():
            return None  # git's answer for a key that is not set
        return Project(f"remote:{_hex16(normalize_url(_checked(url, top)))}", name, str(root))
    listed = _git(top, "rev-list", "--max-parents=0", "HEAD")
    if listed.returncode != 0 and "HEAD" in listed.stderr and "unknown revision" in listed.stderr:
        return None  # no commits yet
    roots = _checked(listed, top).split()
    return Project(f"git:{min(roots)[:16]}", name, str(root))


def normalize_url(url: str) -> str:
    """A remote URL as `host/path`: lower case, no scheme, user or trailing `.git` or `/`, so
    `git@github.com:a/b.git` and `https://github.com/a/b` match."""
    text = url.strip().lower()
    if "://" in text:
        text = text.split("://", 1)[1]
        authority, slash, path = text.partition("/")
        text = authority.rpartition("@")[2] + slash + path
    else:  # scp-like: [user@]host:path
        text = text.rpartition("@")[2].replace(":", "/", 1)
    text = text.rstrip("/")
    while text.endswith(".git"):
        text = text[: -len(".git")].rstrip("/")
    return text


def _hex16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _repository(where: Path) -> tuple[Path | None, bool]:
    """(the git top level, whether the clone is shallow), or (None, False) outside a repository."""
    try:
        done = _git(where, "rev-parse", "--show-toplevel", "--is-shallow-repository")
    except FileNotFoundError:
        return None, False  # no git on this machine: no folder is in a repository
    if done.returncode != 0 and "not a git repository" in done.stderr:
        return None, False
    top, shallow = _checked(done, where).splitlines()
    return Path(top).resolve(), shallow == "true"


def _marker(where: Path, top: Path | None) -> str | None:
    """The ID in the nearest `.mms-project` from `where` up to the git top level (outside git, `where` only)."""
    folders = [where]
    while top is not None and folders[-1] != top and top in folders[-1].parents:
        folders.append(folders[-1].parent)
    for folder_ in folders:
        file = folder_ / MARKER
        if file.is_file():
            value = file.read_text(encoding="utf-8").strip()
            if not MARKER_ID.match(value):
                raise ValueError(f"{file}: the project ID is 4-64 letters, digits, '_' or '-', not {value!r}")
            return value
    return None


def _git(where: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(where), *args], capture_output=True, text=True, encoding="utf-8",
                          timeout=GIT_TIMEOUT_S, stdin=subprocess.DEVNULL)


def _checked(done: subprocess.CompletedProcess, where: Path) -> str:
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(done.args[3:])} in {where} failed: {done.stderr.strip() or done.returncode}")
    return done.stdout.strip()
