"""Downloading a model (DESIGN.md §13.3): `mms model use` and `prepare` call `download` for a model that is not in its folder.
This is the one module that imports huggingface_hub, and only when a download runs. Everything goes into the model's folder
(Xet off, so nothing lands under ~/.cache), with no token and no telemetry. The commit is the model's own (`About.commit`),
asked for and downloaded by its hash, never a branch, so what runs on the machine is what was checked, and huggingface_hub
writes no ref under ~/.cache either. A folder is only downloaded into when it is empty, holds a download that stopped, or
holds this model unfinished: anything else is somebody's, and is refused.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

from mms import decider

# huggingface_hub reads these when it is first imported; Xet would cache under ~/.cache, not in the folder
HUB_ENV = {"HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "DO_NOT_TRACK": "1",
           "HF_HUB_DISABLE_XET": "1"}
SPARE_BYTES = 1_000_000_000  # what a download leaves free on the drive
PARTIAL = Path(".cache") / "huggingface"  # where huggingface_hub keeps the state of a download into a folder
# huggingface_hub writes each file under a name of its own (`*.incomplete`) and moves it into place when it is whole, so a
# file cut off part-way is never continued: it starts again
RETRY = "the same command downloads what is still missing, and a file cut off part-way starts again"


class FetchError(Exception):
    """A download that failed or cannot start, with why; `mms model` exits 1 on it."""


def unfinished(folder: Path) -> str | None:
    """Why the weights are not all there (a download that stopped), or None. A file the Hub's record of the repo lists that
    is missing or not its size, a shard the index lists that is not on disk, or no weights."""
    listed = _listed(folder)
    short = sorted(name for name, size in listed.items()
                   if not (folder / name).is_file() or (folder / name).stat().st_size != size)
    if short:
        return f"{len(short)} of {len(listed)} files missing or cut short, first {short[0]}"
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
        missing = sorted(shard for shard in shards if not (folder / shard).is_file())
        if missing:
            return f"{len(missing)} of {len(shards)} weight shards missing, first {missing[0]}"
    elif not any(folder.glob("*.safetensors")):
        return "no weights (*.safetensors) yet"
    return None


def ready(about, folder: Path) -> bool:
    """Whether `folder` holds all of the model `about`: a backend claims it as that, and the weights are there."""
    return _claimed_name(folder) == about.name and unfinished(folder) is None


def download(about, folder: Path, keep_free: int = 0) -> None:
    """Download `about`'s repo, at `about.commit`, into `folder`, then check that what arrived is that model, whole.
    `keep_free` is the bytes the drive must still have free after it (mms's floor, `[memory] free_gb`, where the home is on
    this drive)."""
    if not about.repo:
        raise FetchError(f"{about.name} names no repo: put its folder at {folder}")
    if not _may_download_into(about, folder):
        raise FetchError(f"{folder} holds something that is not {about.name}, so nothing is downloaded into it: move it "
                         "away, or give another --dir")
    near = _nearest(folder)
    if near is None:
        raise FetchError(f"{folder} is on a drive that is not there: give another --dir")
    total, licence = hub_info(about.repo, about.commit)
    free = shutil.disk_usage(near).free
    left = free - max(total - _held(folder), 0)
    has = f"{about.repo} is {_gb(total)} and {Path(os.path.abspath(folder)).anchor} has {_gb(free)} free: "
    if left < SPARE_BYTES:
        raise FetchError(has + "free some space or give another --dir")
    if left < keep_free:
        raise FetchError(has + f"downloading it would leave {_gb(left)} free, under [memory] free_gb "
                         f"({keep_free / 1e9:g} GB), which pauses capture: free some space or give another --dir")
    print(f"downloading {about.repo} at {about.commit[:7]} ({_gb(total)}, licence "
          f"{licence or 'not stated: read its LICENSE'}) into {folder}; if it stops, {RETRY}", flush=True)
    began = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    for stray in (folder / PARTIAL / "download").rglob("*.incomplete"):  # a killed download leaves its file behind
        stray.unlink()
    hub_download(about.repo, folder, about.commit)
    claimed = _claimed_name(folder) == about.name
    why = unfinished(folder)
    if not claimed or why:
        raise FetchError(f"{about.repo} was downloaded into {folder}, but {why or f'no backend claims it as {about.name}'}")
    print(f"downloaded {about.name} in {(time.monotonic() - began) / 60:.1f} min")


def same_drive(a: Path, b: Path) -> bool:
    """Whether two paths, which may not be made yet, are on the same drive."""
    near_a, near_b = _nearest(a), _nearest(b)
    return near_a is not None and near_b is not None and os.stat(near_a).st_dev == os.stat(near_b).st_dev


def hub_info(repo: str, commit: str) -> tuple[int, str | None]:
    """(the bytes of the files of `repo` at `commit`, the name of its licence or None) from the Hub."""
    hub = _hub()
    try:
        info = hub.HfApi().model_info(repo, revision=commit, files_metadata=True, token=False)
    except Exception as error:  # the hub raises its own kinds, and so does the network
        raise _failed(repo, error) from None
    if info.sha != commit:
        raise FetchError(f"the download of {repo} failed: the Hub answered with commit {info.sha or 'none'}, not the "
                         f"{commit} that mms pins")
    card = info.card_data
    licence = (getattr(card, "license_name", None) or getattr(card, "license", None)) if card else None
    return sum(file.size or 0 for file in info.siblings or ()), licence


def hub_download(repo: str, folder: Path, commit: str) -> None:
    """Download the repo's files at `commit` into `folder`; a file that is already there whole is left."""
    hub = _hub()
    try:
        hub.snapshot_download(repo, local_dir=str(folder), revision=commit, token=False)
    except Exception as error:
        raise _failed(repo, error) from None


def _hub():
    """huggingface_hub, with its switches set first. It reads them at its first import, so one imported before them
    (telemetry or Xet still on) is refused."""
    os.environ.update(HUB_ENV)
    import huggingface_hub
    from huggingface_hub import constants
    if not (constants.HF_HUB_DISABLE_TELEMETRY and constants.HF_HUB_DISABLE_XET):
        raise RuntimeError("huggingface_hub was imported before mms turned off its telemetry and Xet, so mms does not use it: "
                           "run `mms model` in a process that has not imported it")
    logging.getLogger("huggingface_hub.utils._http").addFilter(_no_token_advice)  # added once, however often this runs
    return huggingface_hub


def _no_token_advice(record: logging.LogRecord) -> bool:
    """Drop the Hub's advice to set HF_TOKEN: mms sends no token on purpose, so setting one would change nothing."""
    return "HF_TOKEN" not in record.getMessage()


def _failed(repo: str, error: Exception) -> FetchError:
    return FetchError(f"the download of {repo} failed: {type(error).__name__}: {error}; {RETRY}")


def _listed(folder: Path) -> dict[str, int]:
    """The files the Hub says the repo has, with their sizes, from the newest `trees/<commit>.json` that huggingface_hub
    wrote into `folder/.cache/huggingface` when the download began; empty when there is none. A record it cannot read is
    left out, as the hub itself does."""
    records = sorted((folder / PARTIAL / "trees").glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not records:
        return {}
    try:
        files = json.loads(records[-1].read_text(encoding="utf-8"))["files"]
        return {name: int(info["size"]) for name, info in files.items()}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _claimed_name(folder: Path) -> str | None:
    """The name of the model a backend claims `folder` as, or None."""
    found = decider.claim(folder)
    return None if found is None else found[1].name


def _may_download_into(about, folder: Path) -> bool:
    """Whether nothing is in the way: a folder some backend claims must be this model; any other is missing or empty, or
    holds a download that stopped."""
    if not folder.exists():
        return True
    if not folder.is_dir():
        return False
    if (name := _claimed_name(folder)) is not None:
        return name == about.name
    return not any(folder.iterdir()) or (folder / PARTIAL).is_dir()


def _held(folder: Path) -> int:
    """Bytes already in `folder`, which a download does not fetch again: not a partial file, which starts again."""
    if not folder.is_dir():
        return 0
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file() and path.suffix != ".incomplete")


def _nearest(path: Path) -> Path | None:
    """The nearest of `path` and its parents that exists, or None when not even its drive is there."""
    near = Path(os.path.abspath(path))
    while not near.exists():
        if near.parent == near:
            return None
        near = near.parent
    return near


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"
