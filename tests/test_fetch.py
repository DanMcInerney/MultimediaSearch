"""`mms.fetch` (DESIGN.md §13.3): what is a finished model folder, which folders a download goes into, the checks before
and after it, and the switches under which the Hub is reached. The Hub is never reached: `hub_info` and `hub_download` are
replaced by `FakeHub`, and `_hub` is tried on a fake `huggingface_hub` module."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import types
from pathlib import Path

import pytest
from conftest import model_folder
from fake_hub import FakeHub

from mms import decider, decider_d1_3b, fetch

ABOUT = decider_d1_3b.ABOUT
REPO = "LiquidAI/d1-3B"
COMMIT = "bee5e87" + "9" * 33  # what the fake Hub says the repo is at
HTTP_LOG = "huggingface_hub.utils._http"  # the logger of the Hub's warnings


# unfinished

def folder_with(tmp_path: Path, *files: str) -> Path:
    folder = tmp_path / "weights"
    folder.mkdir()
    for name in files:
        (folder / name).write_text("", encoding="utf-8")
    return folder


def test_a_folder_with_no_weights_is_unfinished(tmp_path):
    assert fetch.unfinished(folder_with(tmp_path, "config.json")) == "no weights (*.safetensors) yet"


def test_a_folder_with_a_safetensors_file_and_no_index_is_finished(tmp_path):
    assert fetch.unfinished(folder_with(tmp_path, "model.safetensors")) is None


def test_a_shard_the_index_lists_that_is_not_there_is_missing(tmp_path):
    folder = folder_with(tmp_path, "model-00001-of-00002.safetensors")
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}), encoding="utf-8")
    assert fetch.unfinished(folder) == "1 of 2 weight shards missing, first model-00002-of-00002.safetensors"
    (folder / "model-00002-of-00002.safetensors").write_text("", encoding="utf-8")
    assert fetch.unfinished(folder) is None


def test_a_stray_incomplete_file_says_nothing_about_the_weights(tmp_path):
    """The hub names each partial file for the one process that wrote it, so a killed download leaves it behind for good."""
    folder = folder_with(tmp_path, "model.safetensors")
    state = folder / ".cache" / "huggingface" / "download"
    state.mkdir(parents=True)
    (state / "model.safetensors.9c1e.0f2a7b3d.incomplete").write_text("half", encoding="utf-8")
    assert fetch.unfinished(folder) is None


def hub_record(folder: Path, files: dict, name: str = "a" * 40) -> Path:
    """The record huggingface_hub writes when a download begins: `files` is {name: size}."""
    trees = folder / ".cache" / "huggingface" / "trees"
    trees.mkdir(parents=True, exist_ok=True)
    record = trees / f"{name}.json"
    listed = {name: {"size": size, "blob_id": "0"} for name, size in files.items()}
    record.write_text(json.dumps({"format_version": 1, "files": listed}), encoding="utf-8")
    return record


def test_a_file_the_hubs_record_lists_that_is_missing_or_the_wrong_size_is_unfinished(tmp_path):
    folder = folder_with(tmp_path, "model.safetensors", "config.json")
    (folder / "model.safetensors").write_bytes(b"x" * 10)
    (folder / "config.json").write_bytes(b"{}")
    hub_record(folder, {"model.safetensors": 10, "config.json": 2})
    assert fetch.unfinished(folder) is None, "every file the record lists is there at its size"
    (folder / "sub").mkdir()
    hub_record(folder, {"model.safetensors": 10, "config.json": 2, "sub/tokenizer.json": 7})
    assert fetch.unfinished(folder) == "1 of 3 files missing or cut short, first sub/tokenizer.json"
    (folder / "sub" / "tokenizer.json").write_bytes(b"x" * 3)
    assert fetch.unfinished(folder) == "1 of 3 files missing or cut short, first sub/tokenizer.json", "cut short is not whole"
    (folder / "sub" / "tokenizer.json").write_bytes(b"x" * 7)
    assert fetch.unfinished(folder) is None
    hub_record(folder, {"model.safetensors": 11, "config.json": 3})
    assert fetch.unfinished(folder) == "2 of 2 files missing or cut short, first config.json"


def test_only_the_newest_record_counts_and_one_that_cannot_be_read_is_left_out(tmp_path):
    folder = folder_with(tmp_path, "model.safetensors")
    old = hub_record(folder, {"model.safetensors": 0, "gone.bin": 5}, "1" * 40)
    assert fetch.unfinished(folder) is not None
    new = hub_record(folder, {"model.safetensors": 0}, "2" * 40)
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert fetch.unfinished(folder) is None, "the listing of the latest commit"
    new.write_text("{not json", encoding="utf-8")
    os.utime(new, (3_000_000, 3_000_000))
    assert fetch.unfinished(folder) is None, "the hub ignores a record it cannot read, and so does this"


# ready

def test_a_model_is_ready_when_it_is_claimed_as_that_model_and_all_there(tmp_path):
    folder = tmp_path / "d1-3B"
    assert not fetch.ready(ABOUT, folder), "a folder that is not there"
    model_folder(tmp_path, "d1-3B")
    assert fetch.ready(ABOUT, folder)
    (folder / "model.safetensors").unlink()
    assert not fetch.ready(ABOUT, folder), "claimed, but no weights"
    (folder / "model.safetensors").write_text("", encoding="utf-8")
    assert not fetch.ready(ABOUT, model_folder(tmp_path, "d1-2B")), "claimed as another model"
    assert not fetch.ready(ABOUT, model_folder(tmp_path, "unclaimed")), "claimed by nobody"


# download

@pytest.fixture
def hub(monkeypatch) -> FakeHub:
    return FakeHub(monkeypatch)


def test_a_download_says_the_size_and_licence_before_it_starts_and_asks_the_hub_for_that_repo_only(tmp_path, hub, capsys):
    folder = tmp_path / "models" / "d1-3B"
    said = []

    def arrive(where):
        said.append(capsys.readouterr().out)  # what was printed by the time the download ran
        FakeHub.write_model(where)

    hub.arrive = arrive
    fetch.download(ABOUT, folder)
    assert said == [f"downloading {REPO} at 0000000 (6.3 GB, licence lfm1.0) into {folder}; if it stops, the same command "
                    "downloads what is still missing, and a file cut off part-way starts again\n"]
    assert hub.calls == [("info", REPO), ("download", REPO, folder, hub.commit)]
    assert re.fullmatch(r"downloaded d1-3B in \d+\.\d min\n", capsys.readouterr().out)
    assert fetch.ready(ABOUT, folder)


def test_a_licence_the_hub_does_not_state_is_said_so(tmp_path, hub, capsys):
    hub.licence = None
    fetch.download(ABOUT, tmp_path / "d1-3B")
    assert " (6.3 GB, licence not stated: read its LICENSE) " in capsys.readouterr().out


def test_the_commit_the_size_was_checked_at_is_the_one_downloaded_and_the_line_names_it_short(tmp_path, hub, capsys):
    hub.commit = "bee5e87" + "1" * 33
    folder = tmp_path / "d1-3B"
    fetch.download(ABOUT, folder)
    assert hub.calls == [("info", REPO), ("download", REPO, folder, hub.commit)]
    said = capsys.readouterr().out
    assert f"downloading {REPO} at bee5e87 (" in said and hub.commit not in said


def test_an_empty_folder_and_a_download_that_stopped_are_downloaded_into(tmp_path, hub):
    empty = tmp_path / "empty" / "d1-3B"
    empty.mkdir(parents=True)
    fetch.download(ABOUT, empty)
    assert fetch.ready(ABOUT, empty)

    stopped = tmp_path / "stopped" / "d1-3B"
    state = stopped / ".cache" / "huggingface" / "download"
    state.mkdir(parents=True)
    (stopped / "README.md").write_text("arrived before the stop", encoding="utf-8")
    fetch.download(ABOUT, stopped)
    assert [call[0] for call in hub.calls] == ["info", "download", "info", "download"]
    assert fetch.ready(ABOUT, stopped)


def test_a_partial_file_a_killed_download_left_behind_is_removed_before_the_next_one_starts(tmp_path, hub):
    """The hub would never continue it (its name is for the process that died), and it would count as bytes already there."""
    folder = tmp_path / "d1-3B"
    state = folder / ".cache" / "huggingface" / "download" / "sub"
    state.mkdir(parents=True)
    strays = [state / "model.safetensors.9c1e.0f2a7b3d.incomplete", state.parent / "x.json.1.2.incomplete"]
    for stray in strays:
        stray.write_bytes(b"x" * 600)
    (state.parent / "prompt.py.metadata").write_text("kept", encoding="utf-8")
    seen = []

    def arrive(where):
        seen.append(sorted(where.rglob("*.incomplete")))  # what the hub finds when it starts
        FakeHub.write_model(where)

    hub.arrive = arrive
    fetch.download(ABOUT, folder)
    assert seen == [[]], "the stray files are gone before the download"
    assert (state.parent / "prompt.py.metadata").read_text(encoding="utf-8") == "kept", "only partial files go"
    assert fetch.ready(ABOUT, folder)


def test_a_stray_partial_file_is_not_counted_as_bytes_already_there(tmp_path, hub, monkeypatch):
    monkeypatch.setattr(fetch, "SPARE_BYTES", 100)
    hub.size = 1000
    folder = tmp_path / "d1-3B"
    state = folder / ".cache" / "huggingface" / "download"
    state.mkdir(parents=True)
    (state / "model.safetensors.1.2.incomplete").write_bytes(b"x" * 600)
    hub.free = 1099  # 1000 are needed, and 100 spare
    with pytest.raises(fetch.FetchError, match="free some space"):
        fetch.download(ABOUT, folder)
    hub.free = 1100
    fetch.download(ABOUT, folder)
    assert fetch.ready(ABOUT, folder)


def test_this_model_unfinished_is_resumed(tmp_path, hub):
    folder = model_folder(tmp_path, "d1-3B")
    (folder / "model.safetensors").unlink()
    assert not fetch.ready(ABOUT, folder)
    fetch.download(ABOUT, folder)
    assert hub.calls == [("info", REPO), ("download", REPO, folder, hub.commit)]
    assert fetch.ready(ABOUT, folder)


@pytest.mark.parametrize("what", ["a file of the owner's", "another model", "another model downloaded by the hub",
                                  "a file where the folder should be"])
def test_a_folder_that_holds_anything_else_is_refused_and_the_hub_is_not_asked(tmp_path, hub, what):
    folder = tmp_path / "d1-3B"
    if what == "a file of the owner's":
        folder.mkdir()
        (folder / "notes.txt").write_text("mine", encoding="utf-8")
    elif what.startswith("another model"):
        model_folder(tmp_path / "elsewhere", "d1-2B").rename(folder)
        if what.endswith("hub"):  # `hf download --local-dir` leaves this folder in every folder it fills
            (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
    else:
        folder.write_text("not a folder", encoding="utf-8")
    with pytest.raises(fetch.FetchError, match=re.escape(
            f"{folder} holds something that is not d1-3B, so nothing is downloaded into it: move it away, or give another "
            "--dir")):
        fetch.download(ABOUT, folder)
    assert hub.calls == []


def test_a_model_with_no_repo_is_refused(tmp_path, hub):
    with pytest.raises(fetch.FetchError, match=r"nowhere names no repo: put its folder at .*nowhere"):
        fetch.download(decider.About(name="nowhere"), tmp_path / "nowhere")
    assert hub.calls == []


def test_too_little_free_space_is_refused_after_asking_the_size_and_before_downloading(tmp_path, hub):
    hub.free = 6_000_000_000
    with pytest.raises(fetch.FetchError, match=re.escape(f"{REPO} is 6.3 GB and ") + r".* has 6\.0 GB free: free some space "
                                               "or give another --dir"):
        fetch.download(ABOUT, tmp_path / "models" / "d1-3B")
    assert hub.calls == [("info", REPO)], "the size is asked, nothing is downloaded"
    assert not (tmp_path / "models").exists(), "nothing is made either"


def test_the_space_needed_is_the_size_less_what_is_there_plus_a_spare(tmp_path, hub, monkeypatch):
    monkeypatch.setattr(fetch, "SPARE_BYTES", 100)
    hub.size = 1000
    folder = tmp_path / "d1-3B"
    state = folder / ".cache" / "huggingface"
    state.mkdir(parents=True)
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"x" * 600)  # a whole file counts as there
    hub.free = 499  # 1000 - 600 + 100 = 500 are needed
    with pytest.raises(fetch.FetchError, match="free some space"):
        fetch.download(ABOUT, folder)
    assert [call[0] for call in hub.calls] == ["info"]
    hub.free = 500
    fetch.download(ABOUT, folder)
    assert [call[0] for call in hub.calls] == ["info", "info", "download"]


def test_the_space_is_read_where_the_folder_will_be(tmp_path, hub, monkeypatch):
    asked = []

    def disk_usage(path):
        asked.append(Path(path))
        return types.SimpleNamespace(free=10**12)

    monkeypatch.setattr(fetch.shutil, "disk_usage", disk_usage)
    fetch.download(ABOUT, tmp_path / "not" / "there" / "yet" / "d1-3B")
    assert [Path(path) for path in asked] == [tmp_path], "the nearest folder that exists"


def missing_drive() -> str | None:
    """A drive letter nothing is mounted on, found now; None where there are no drive letters (or none is free)."""
    if os.name != "nt":
        return None
    return next((letter for letter in "ZYXWVUTSRQPONMLKJIHGFEDCBA" if not Path(f"{letter}:/").exists()), None)


def test_a_folder_on_a_drive_that_is_not_there_is_refused_before_the_hub_is_asked(tmp_path, hub):
    letter = missing_drive()
    if letter is None:
        pytest.skip("no drive letters")
    folder = Path(f"{letter}:/models/d1-3B")
    assert fetch._nearest(folder) is None, "the search ends at the root, and does not loop"
    with pytest.raises(fetch.FetchError, match=re.escape(f"{folder} is on a drive that is not there: give another --dir")):
        fetch.download(ABOUT, folder)
    assert hub.calls == []


def test_the_floor_of_mms_is_kept_free_when_it_is_asked_for(tmp_path, hub):
    """A download that would leave less free than `[memory] free_gb` would pause capture."""
    hub.free = 26_000_000_000  # the repo is 6.3 GB: 19.7 GB would be left
    folder = tmp_path / "d1-3B"
    fetch.download(ABOUT, folder)  # no floor asked for: room enough
    shutil.rmtree(folder)
    hub.calls.clear()
    with pytest.raises(fetch.FetchError, match=re.escape(
            f"{REPO} is 6.3 GB and ") + r".* has 26\.0 GB free: downloading it would leave 19\.7 GB free, under "
            r"\[memory\] free_gb \(20 GB\), which pauses capture: free some space or give another --dir"):
        fetch.download(ABOUT, folder, keep_free=20_000_000_000)
    assert hub.calls == [("info", REPO)] and not folder.exists()
    hub.free = 26_300_000_000  # exactly the floor is left
    fetch.download(ABOUT, folder, keep_free=20_000_000_000)
    assert fetch.ready(ABOUT, folder)


def test_two_paths_are_on_the_same_drive_by_their_nearest_folder_that_exists(tmp_path):
    assert fetch.same_drive(tmp_path / "a" / "b", tmp_path / "c")
    letter = missing_drive()
    if letter is not None:
        assert not fetch.same_drive(Path(f"{letter}:/models"), tmp_path), "a drive that is not there is not this one"


def test_a_download_that_leaves_nothing_a_backend_claims_is_an_error(tmp_path, hub):
    def junk(folder):
        (folder / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
        (folder / "model.safetensors").write_text("", encoding="utf-8")

    hub.arrive = junk
    folder = tmp_path / "d1-3B"
    with pytest.raises(fetch.FetchError, match=re.escape(f"{REPO} was downloaded into {folder}, but no backend claims it as "
                                                         "d1-3B")):
        fetch.download(ABOUT, folder)


def test_a_download_that_leaves_a_file_of_the_hubs_record_short_says_why(tmp_path, hub):
    hub.listed = {"model.safetensors": 5}  # the weights are said to be 5 bytes, and arrive as none
    with pytest.raises(fetch.FetchError, match=r"was downloaded into .*, but 1 of 1 files missing or cut short, "
                                               r"first model\.safetensors"):
        fetch.download(ABOUT, tmp_path / "d1-3B")


def test_a_download_that_leaves_a_file_of_the_hubs_record_missing_is_not_ready_and_is_downloaded_into_again(tmp_path, hub):
    folder = model_folder(tmp_path, "d1-3B")
    FakeHub.write_record(folder, {"config.json": (folder / "config.json").stat().st_size, "model.safetensors": 0,
                                  "tokenizer.json": 100})
    assert not fetch.ready(ABOUT, folder) and "tokenizer.json" in fetch.unfinished(folder)
    fetch.download(ABOUT, folder)
    assert hub.calls == [("info", REPO), ("download", REPO, folder, hub.commit)]
    assert fetch.ready(ABOUT, folder), "the fake hub's download writes the record of what it left"


def test_a_failed_download_is_the_hubs_error_and_the_folder_stays_for_the_next_try(tmp_path, hub):
    hub.fail = fetch.FetchError(f"the download of {REPO} failed: OSError: no network; {fetch.RETRY}")
    folder = tmp_path / "d1-3B"
    with pytest.raises(fetch.FetchError, match="no network"):
        fetch.download(ABOUT, folder)
    assert (folder / ".cache" / "huggingface").is_dir()
    hub.fail = None
    fetch.download(ABOUT, folder)  # the state the first try left is a download that stopped
    assert fetch.ready(ABOUT, folder)


# the hub

KEYS = tuple(fetch.HUB_ENV)


def fake_module(monkeypatch, quiet: bool = True, **parts) -> list:
    """A `huggingface_hub` in sys.modules that records each call with the switches in force when it was made. The switches
    are set to "0" first, so that the test sees them turned on and its environment is restored afterwards. `sha` is the
    commit the repo is said to be at (None: none given), and `arrive` is called with the folder a download is into. The
    Hub's logger has no filters for the test, and its own again after."""
    for key in KEYS:
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(logging.getLogger(HTTP_LOG), "filters", [])
    seen: list = []

    def switches() -> dict:
        return {key: os.environ.get(key) for key in KEYS}

    class HfApi:
        def model_info(self, repo, **kwargs):
            seen.append(("model_info", repo, kwargs, switches()))
            if "info_error" in parts:
                raise parts["info_error"]
            sizes = [3_000, None, 4_000]  # a file whose size the hub did not give
            return types.SimpleNamespace(
                siblings=[types.SimpleNamespace(size=size) for size in sizes], sha=parts.get("sha", COMMIT),
                card_data=parts.get("card", types.SimpleNamespace(license_name="lfm1.0", license="other")))

    def snapshot_download(repo, **kwargs):
        seen.append(("snapshot_download", repo, kwargs, switches()))
        if "download_error" in parts:
            raise parts["download_error"]
        if "arrive" in parts:
            parts["arrive"](Path(kwargs["local_dir"]))
        return kwargs["local_dir"]

    module = types.ModuleType("huggingface_hub")
    module.constants = types.SimpleNamespace(HF_HUB_DISABLE_TELEMETRY=quiet, HF_HUB_DISABLE_XET=quiet)
    module.HfApi, module.snapshot_download = HfApi, snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return seen


def test_the_hub_is_asked_for_the_size_licence_and_commit_with_no_token_and_the_switches_on(monkeypatch):
    seen = fake_module(monkeypatch)
    assert fetch.hub_info(REPO) == (7_000, "lfm1.0", COMMIT), "the sizes the hub gives, a missing one as 0"
    [(call, repo, kwargs, switches)] = seen
    assert (call, repo, kwargs) == ("model_info", REPO, {"files_metadata": True, "token": False})
    assert switches == dict.fromkeys(KEYS, "1"), "every switch is on when the hub is first used"


def test_a_licence_is_its_name_else_its_id_else_none(monkeypatch):
    fake_module(monkeypatch, card=types.SimpleNamespace(license_name=None, license="apache-2.0"))
    assert fetch.hub_info(REPO)[1] == "apache-2.0"
    fake_module(monkeypatch, card=None)
    assert fetch.hub_info(REPO)[1] is None


def test_a_download_is_of_the_commit_given_into_the_folder_given_with_no_token_and_the_switches_on(monkeypatch, tmp_path):
    seen = fake_module(monkeypatch)
    fetch.hub_download(REPO, tmp_path / "d1-3B", COMMIT)
    [(call, repo, kwargs, switches)] = seen
    assert (call, repo, kwargs) == ("snapshot_download", REPO, {"local_dir": str(tmp_path / "d1-3B"), "revision": COMMIT,
                                                                 "token": False})
    assert switches == dict.fromkeys(KEYS, "1")


def test_a_download_asks_the_hub_for_the_commit_the_size_was_checked_at_and_never_for_a_branch(monkeypatch, tmp_path):
    seen = fake_module(monkeypatch, arrive=FakeHub.write_model)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=10**12))
    fetch.download(ABOUT, tmp_path / "d1-3B")
    [(info, _, _, _), (download, _, kwargs, _)] = seen
    assert (info, download) == ("model_info", "snapshot_download")
    assert kwargs["revision"] == COMMIT and kwargs["revision"] != "main"


@pytest.mark.parametrize("sha", [None, ""])
def test_a_hub_that_gives_no_commit_is_an_error_and_nothing_is_downloaded(monkeypatch, tmp_path, sha):
    seen = fake_module(monkeypatch, sha=sha)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=10**12))
    folder = tmp_path / "d1-3B"
    with pytest.raises(fetch.FetchError, match=re.escape(
            f"the download of {REPO} failed: the Hub gave no commit for it; {fetch.RETRY}")):
        fetch.download(ABOUT, folder)
    assert [call[0] for call in seen] == ["model_info"] and not folder.exists()


def test_a_hub_imported_before_the_switches_were_set_is_not_used(monkeypatch):
    seen = fake_module(monkeypatch, quiet=False)
    with pytest.raises(RuntimeError, match="imported before mms turned off its telemetry and Xet"):
        fetch.hub_info(REPO)
    with pytest.raises(RuntimeError, match="imported before"):
        fetch.hub_download(REPO, Path("x"), COMMIT)
    assert seen == []


def test_what_the_hub_raises_is_a_fetch_error_naming_the_repo(monkeypatch, tmp_path):
    fake_module(monkeypatch, info_error=ConnectionError("no route"), download_error=TimeoutError("read timed out"))
    with pytest.raises(fetch.FetchError, match=re.escape(
            f"the download of {REPO} failed: ConnectionError: no route; {fetch.RETRY}")):
        fetch.hub_info(REPO)
    with pytest.raises(fetch.FetchError, match=re.escape(
            f"the download of {REPO} failed: TimeoutError: read timed out; {fetch.RETRY}")):
        fetch.hub_download(REPO, tmp_path, COMMIT)


class Shown(logging.Handler):
    """The messages that get past the filters of the logger it is on."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_the_hubs_advice_to_set_a_token_is_not_shown_and_its_other_warnings_are(monkeypatch):
    """mms sends no token on purpose, so the advice would change nothing."""
    fake_module(monkeypatch)
    fetch._hub()
    logger, shown = logging.getLogger(HTTP_LOG), Shown()
    monkeypatch.setattr(logger, "handlers", [shown])
    logger.warning("Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher "
                   "rate limits and faster downloads.")
    logger.warning("Rate limited. Waiting 4s before retry [Retry 1/5].")
    assert shown.messages == ["Rate limited. Waiting 4s before retry [Retry 1/5]."]


def test_the_filter_is_on_the_hubs_logger_once_however_often_the_hub_is_used(monkeypatch):
    fake_module(monkeypatch)
    fetch._hub()
    fetch._hub()
    assert logging.getLogger(HTTP_LOG).filters == [fetch._no_token_advice]


def test_with_the_real_hub_blocked_a_call_that_reaches_for_it_is_an_import_error():
    """conftest blocks the real one in every test: a call that reaches for it fails, and downloads nothing."""
    with pytest.raises(ImportError):
        fetch.hub_info(REPO)
