"""A stand-in for the two functions of `mms.fetch` that reach Hugging Face. They record what they are asked, and a
download leaves the files of a model folder (`conftest.model_folder`: never real weights) and the Hub's record of them, so no
test downloads anything. The drive is said to have plenty of room (a test of the space check says otherwise)."""

from __future__ import annotations

import json
import types

from conftest import model_folder

from mms import fetch


class FakeHub:
    """`fetch.hub_info` and `fetch.hub_download`, replaced for the test, and the free space `fetch` reads. `calls` is what
    they were asked, in order: `("info", repo, commit)` and `("download", repo, folder, commit)`, the commit a model's own
    `About` pins."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple] = []
        self.size = 6_300_000_000  # what the repo is said to weigh
        self.licence = "lfm1.0"
        self.free = 10**12  # what the drive is said to have free
        self.arrive = self.write_model  # what a download leaves in the folder: a function of the folder, or None for nothing
        self.listed: dict | None = None  # the files and sizes the Hub's record lists; None: those the download left
        self.fail: Exception | None = None  # raised by the download, after the Hub's state is written
        monkeypatch.setattr(fetch, "hub_info", self.info)
        monkeypatch.setattr(fetch, "hub_download", self.download)
        monkeypatch.setattr(fetch.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=self.free))

    def info(self, repo: str, commit: str) -> tuple:
        self.calls.append(("info", repo, commit))
        return self.size, self.licence

    def download(self, repo: str, folder, commit: str) -> None:
        self.calls.append(("download", repo, folder, commit))
        state = folder / fetch.PARTIAL / "download"  # where the Hub keeps the state of a download into a folder
        state.mkdir(parents=True, exist_ok=True)
        (state / "config.json.metadata").write_text("", encoding="utf-8")
        if self.fail is not None:
            raise self.fail
        if self.arrive is not None:
            self.arrive(folder)
        self.write_record(folder, self.listed)

    @staticmethod
    def write_model(folder) -> None:
        """The model whose name the folder has, as conftest writes it. A partial file the folder held stays: only the hub's
        own download, which the fake is not, takes it away."""
        model_folder(folder.parent, folder.name)

    @staticmethod
    def write_record(folder, listed: dict | None = None) -> None:
        """The record huggingface_hub keeps of the repo's files, `.cache/huggingface/trees/<commit>.json`: the files given
        and their sizes, else those now in the folder (a download that finished)."""
        if listed is None:
            listed = {path.relative_to(folder).as_posix(): path.stat().st_size for path in folder.rglob("*")
                      if path.is_file() and path.relative_to(folder).parts[0] != ".cache" and path.suffix != ".incomplete"}
        trees = folder / fetch.PARTIAL / "trees"
        trees.mkdir(parents=True, exist_ok=True)
        files = {name: {"size": size, "blob_id": "0" * 40} for name, size in listed.items()}
        (trees / ("0" * 40 + ".json")).write_text(json.dumps({"format_version": 1, "files": files}), encoding="utf-8")
