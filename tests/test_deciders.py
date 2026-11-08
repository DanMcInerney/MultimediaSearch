"""A decider is swapped by its facts alone (TESTBED.md, "No GPU"; DESIGN.md §13.3): `About` says what a model reads, and
mms cuts what it hands the model from that. These tests prove it three ways: the loader picks a backend by what claims a
folder and a decider declares its own facts; the reader hands each `About` what it declares and says what it left out;
and the same ask, over the same store, run end to end under two deciders with different facts (`fake_decider.py`, which
has d1's, and `other_decider.py`), gives each what it declares, with nothing outside the decider's file changed."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import other_decider
import pytest
from conftest import FAKE, MODEL_FOLDERS, OTHER, Served, env, model_folder, prepared_env, write_config
from corpus import (EVERYTHING, GAP, OUT, T0, X264, check_unread, names_hit, not_read_warnings, remember_each, run,
                    scan)
from fake_decider import ABOUT as FAKE_ABOUT

from mms import (decider, decider_d1, decider_d1_3b, decider_d3, decider_h2o, decider_pplx, jobs, modelenv, native,
                      reader)
from mms.decider import About
from mms.errors import ApiError
from mms.gpu import Gpu
from mms.store import Store

BASE = "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 50\n"
OMNI = {"model_type": "lfm3_omni", "d1_attention": "sdpa", "d1_set_size": 16, "d1_vision_max_pixels": 1638400, "d1_numbers": [0, 1], "d1_codes": [0, 1]}


def folder_with(tmp_path: Path, name: str, config: dict | None) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


class Toy:
    """A backend for a folder with a toy.json, to be tried before d1's."""

    about = About(name="toy")
    abouts = (about,)
    reads_folder = "toy reads a folder with a toy.json"

    @staticmethod
    def claims(folder: Path) -> About | None:
        return Toy.about if (Path(folder) / "toy.json").is_file() else None

    @staticmethod
    def check(about: About, questions: dict) -> None:
        pass

    def __init__(self, folder, *, url=None, envs=None):
        self.folder, self.url, self.envs = Path(folder), url, envs

    def decide(self, state, questions, media=None):
        return {}


# -- the loader -------------------------------------------------------------------------------------------------------

def test_a_model_folder_is_read_by_the_backend_that_claims_it(home, tmp_path, monkeypatch):
    omni = folder_with(tmp_path, "d1-Omni-2B", OMNI)
    loaded = decider.load_decider(str(omni), home)
    assert isinstance(loaded, decider_d1.D1) and loaded.about is decider_d1.ABOUT and loaded.worker.pid is None, \
        "claimed by d1's backend, with nothing loaded"
    # reading its facts needs neither torch nor the GPU: the interpreter that does it never imports torch
    code = ("import sys; from mms import decider; d = decider.load_decider(sys.argv[1]); "
            "assert d.about.name == 'd1-Omni-2B'; assert 'torch' not in sys.modules, 'torch was imported'")
    done = subprocess.run([sys.executable, "-c", code, str(omni)], env=env(home), capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr

    refused = {"empty": "{}", "not-json": "{", "a-list": "[]", "no-config": None}
    for name, text in refused.items():
        folder = folder_with(tmp_path, name, None)
        if text is not None:
            (folder / "config.json").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="no decider reads the model") as raised:
            decider.load_decider(str(folder), home)
        assert "d1-Omni-2B" in str(raised.value) and "python:<file.py>" in str(raised.value), \
            f"{name}: the error says what is read: {raised.value}"
    three = decider.load_decider(str(model_folder(tmp_path, "d1-3B")), home)
    assert isinstance(three, decider_d1_3b.D1_3B) and three.about is decider_d1_3b.ABOUT and three.worker.pid is None, \
        "claimed by d1-3B's own backend, not d1's"
    with pytest.raises(ValueError, match="there is no model folder at") as raised:
        decider.load_decider(str(tmp_path / "missing"), home)
    assert "no decider reads" not in str(raised.value), "a folder that is not there is not a model that is unsupported"
    with pytest.raises(ValueError, match="there is no model folder at"):
        decider.load_decider(str(omni / "config.json"), home)  # a file is no folder either

    for key, wrong, right in (("d1_vision_max_pixels", 1003520, 1638400), ("d1_set_size", 8, 16)):
        folder = folder_with(tmp_path, f"drift-{key}", {**OMNI, key: wrong})
        with pytest.raises(ValueError) as raised:
            decider.load_decider(str(folder), home)
        assert key in str(raised.value) and str(wrong) in str(raised.value) and str(right) in str(raised.value), \
            f"weights that differ from ABOUT are refused, naming both numbers: {raised.value}"

    # a new model is a module and a BACKENDS entry: the first whose `claims` is true makes the decider
    monkeypatch.setattr(decider, "BACKENDS", (f"{__name__}:Toy", *decider.BACKENDS))
    toy = folder_with(tmp_path, "toy", {})
    (toy / "toy.json").write_text("{}", encoding="utf-8")
    assert isinstance(decider.load_decider(str(toy), home), Toy)
    assert isinstance(decider.load_decider(str(omni), home), decider_d1.D1), "a folder the toy does not claim goes on to d1"
    with pytest.raises(ValueError) as raised:
        decider.load_decider(str(folder_with(tmp_path, "unclaimed", {})), home)
    assert Toy.reads_folder in str(raised.value) and decider_d1.D1.reads_folder in str(raised.value), \
        f"the error says what every backend reads: {raised.value}"


LIFECYCLE = """
from mms.decider import About
ABOUT = About(name="lifecycle")
home = None
def note(word):
    with open(home / "events.txt", "a") as out:
        out.write(word + "\\n")
def load(): note("load")
def unload(): note("unload")
def decide(state, questions, media):
    note("decide")
    return {name: {"type": "noul", "noul": 0.95} for name in questions}
"""


def write_py(folder: Path, name: str, body: str) -> str:
    path = folder / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path.as_posix()


def test_a_python_decider_declares_its_facts(home, tmp_path):
    for name, body, says in (
            ("no_about", "def decide(state, questions, media): return {}\n", "defines no ABOUT"),
            ("not_about", "ABOUT = 'd1'\ndef decide(state, questions, media): return {}\n", "defines no ABOUT"),
            ("no_decide", "from mms.decider import About\nABOUT = About(name='x')\n", "defines no decide"),
            ("not_callable", "from mms.decider import About\nABOUT = About(name='x')\ndecide = 3\n", "defines no decide")):
        with pytest.raises(ValueError, match=says):
            decider.load_decider("python:" + write_py(tmp_path, name, body), home)
    loaded = decider.load_decider("python:" + write_py(tmp_path, "ok", "from mms.decider import About\n"
                                                       "ABOUT = About(name='ok')\ndef decide(s, q, m): return {'home': home}\n"),
                                  home)
    assert loaded.about.name == "ok" and loaded.decide(None, {}) == {"home": home}, "the loader sets `home` in the file"
    loaded.load()
    loaded.unload()  # a file without them is still a decider

    # the facts are read without loading or deciding
    decider.load_decider("python:" + write_py(tmp_path, "lifecycle", LIFECYCLE), home).about
    assert not (home / "events.txt").exists(), "reading `about` calls neither load nor decide"


def test_the_queue_calls_load_and_unload_around_the_deciding(home, tmp_path, served):
    write_config(home, BASE + f'[decider]\nmodel = "python:{write_py(tmp_path, "lifecycle", LIFECYCLE)}"\n')
    running = served()
    api = running.api
    api.call("remember", note="a kettle", t=T0, source="notes")
    job = api.call("jobs/start", script="ask", params={"question": "Is it?", "since": T0 - 10, "until": T0 + 10}, wait_s=40)
    assert job["state"] == "done" and len(job["result"]["sightings"]) == 1, job
    assert (home / "events.txt").read_text().split() == ["load", "decide"], \
        "the model loads once, for the first window, and not for `about`"
    running.stop()
    assert (home / "events.txt").read_text().split() == ["load", "decide", "unload"]


def test_a_decider_that_declares_no_facts_fails_the_job_with_the_reason(home, tmp_path, served):
    broken = write_py(tmp_path, "broken", "def decide(state, questions, media): return {}\n")
    write_config(home, BASE + f'[decider]\nmodel = "python:{broken}"\n')
    api = served().api
    job = api.call("jobs/start", script="ask", params={"question": "Is it?", "since": T0 - 10, "until": T0 + 10}, wait_s=40)
    assert job["state"] == "failed" and job["error"]["code"] == "decider", job
    assert "the decider could not load" in job["error"]["message"] and "defines no ABOUT" in job["error"]["message"], job
    assert api.call("status")["decider"]["state"] == "failed"


GOOD = "from mms.decider import About\nABOUT = About(name='good')\ndef decide(s, q, m): return {n: {'type': 'noul', 'noul': 0.9} for n in q}\n"
NO_ABOUT = "def decide(state, questions, media): return {}\n"


def test_the_status_keeps_no_error_of_another_model(tmp_path):
    config = lambda model, url=None, envs=None: SimpleNamespace(  # noqa: E731
        home=tmp_path, decider_model=model, decider_url=url, decider_envs=envs, decider_idle_s=1)
    first, good = "python:" + write_py(tmp_path, "first", NO_ABOUT), "python:" + write_py(tmp_path, "good", GOOD)
    gpu = Gpu(config(first))
    with pytest.raises(ApiError, match="defines no ABOUT"):
        gpu.about()
    assert gpu.status()["state"] == "failed" and "defines no ABOUT" in gpu.status()["error"]

    gpu.apply(config(good))  # another model: the old one's error is not its
    assert gpu.status() == {"model": good, "state": "unloaded", "error": None, "waiting": 0}
    assert gpu.about().name == "good" and gpu.status()["state"] == "unloaded"

    gpu.apply(config(first))  # back to the broken one, then the file is mended under the same model string
    with pytest.raises(ApiError):
        gpu.about()
    assert gpu.status()["state"] == "failed"
    write_py(tmp_path, "first", GOOD)
    assert gpu.about().name == "good"
    assert gpu.status()["state"] == "unloaded" and gpu.status()["error"] is None, \
        "a model that can be made again is not failed, before its first real load"

    gpu.apply(config(str(tmp_path / "missing")))  # a folder that is not there says so
    with pytest.raises(ApiError, match="there is no model folder at"):
        gpu.about()
    assert "there is no model folder" in gpu.status()["error"]
    gpu.stop()


class Boom:
    """A backend that claims a folder with a boom.json, and cannot be constructed."""

    about = About(name="boom", hit=0.6, notes="boom.md")
    abouts = (about,)
    reads_folder = "boom reads a folder with a boom.json"
    made = []

    @staticmethod
    def claims(folder: Path) -> About | None:
        return Boom.about if (Path(folder) / "boom.json").is_file() else None

    @staticmethod
    def check(about: About, questions: dict) -> None:
        pass

    def __init__(self, folder, *, url=None, envs=None):
        Boom.made.append(folder)
        raise RuntimeError("constructing this needs the GPU")


def test_a_backend_is_made_with_the_url_and_envs_and_python_and_d1_refuse_a_url(home, tmp_path, monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", (f"{__name__}:Toy", *decider.BACKENDS))
    toy = folder_with(tmp_path, "toy", {})
    (toy / "toy.json").write_text("{}", encoding="utf-8")
    made = decider.load_decider(str(toy), home, url="http://127.0.0.1:8741", envs="C:/envs")
    assert (made.url, made.envs) == ("http://127.0.0.1:8741", "C:/envs")
    made = decider.load_decider(str(toy), home)
    assert (made.url, made.envs) == (None, None)

    good = write_py(tmp_path, "good", GOOD)
    with pytest.raises(ValueError, match=r"python: decider takes no \[decider\] url"):
        decider.load_decider("python:" + good, home, url="http://127.0.0.1:8741")
    assert decider.load_decider("python:" + good, home, envs="C:/envs").about.name == "good", \
        "envs is for the folder backends, and a python: decider has no use for it"
    omni = model_folder(tmp_path, "d1-Omni-2B")
    with pytest.raises(ValueError, match=r"not served.*url.*127\.0\.0\.1:8741"):
        decider.load_decider(str(omni), home, url="http://127.0.0.1:8741")
    assert isinstance(decider.load_decider(str(omni), home, envs="C:/envs"), decider_d1.D1)


def test_the_facts_of_a_folder_are_read_without_making_its_backend(home, tmp_path, monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", (f"{__name__}:Boom", *decider.BACKENDS))
    boom = folder_with(tmp_path, "boom", {})
    (boom / "boom.json").write_text("{}", encoding="utf-8")
    Boom.made.clear()
    assert decider.about_of(str(boom)) is Boom.about and Boom.made == [], "about_of constructs nothing"
    with pytest.raises(RuntimeError, match="needs the GPU"):
        decider.load_decider(str(boom), home)
    assert len(Boom.made) == 1
    omni = model_folder(tmp_path, "d1-Omni-2B")
    assert decider.about_of(str(omni)) is decider_d1.ABOUT
    assert decider.known_abouts() == [Boom.about, decider_d1.ABOUT, decider_d1.ABOUT_2B, *decider_d1_3b.D1_3B.abouts,
                                      *decider_d3.D3.abouts, *decider_pplx.Pplx.abouts, *decider_h2o.H2o.abouts], \
        "every backend's abouts, in BACKENDS order"
    assert len(decider_d3.D3.abouts) == 3
    for nothing in (None, "", FAKE, str(tmp_path / "missing"), str(omni / "config.json"),
                    str(model_folder(tmp_path, "unclaimed"))):
        assert decider.about_of(nothing) is None, nothing
    # a Gpu makes the backend to read its facts, so it fails with the backend's own error
    gpu = Gpu(SimpleNamespace(home=home, decider_model=str(boom), decider_url=None, decider_envs=None, decider_idle_s=1))
    with pytest.raises(ApiError, match="needs the GPU"):
        gpu.about()


class Picky:
    """A backend for a folder with a picky.json, whose static `check` refuses a question named "bad"."""

    about = About(name="picky")
    abouts = (about,)
    reads_folder = "picky reads a folder with a picky.json"
    made = []
    checked = []

    @staticmethod
    def claims(folder: Path) -> About | None:
        return Picky.about if (Path(folder) / "picky.json").is_file() else None

    @staticmethod
    def check(about: About, questions: dict) -> None:
        Picky.checked.append((about, questions))
        if "bad" in questions:
            raise ValueError(f"{about.name} cannot take the question 'bad'")

    def __init__(self, folder, *, url=None, envs=None):
        Picky.made.append(folder)
        raise RuntimeError("constructing this needs the GPU")


CHECKS = "from mms.decider import About\nABOUT = About(name='checks')\ndef decide(s, q, m): return {}\n" \
         "def check(questions):\n    if 'bad' in questions:\n        raise ValueError('checks cannot take bad')\n"


def test_questions_are_checked_by_the_claiming_backend_without_making_it(home, tmp_path, monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", (f"{__name__}:Picky", *decider.BACKENDS))
    picky = folder_with(tmp_path, "picky", {})
    (picky / "picky.json").write_text("{}", encoding="utf-8")
    Picky.made.clear()
    Picky.checked.clear()
    good, bad = {"q": {"type": "noul", "instructions": "x"}}, {"bad": {"type": "noul", "instructions": "x"}}
    assert decider.check_of(str(picky), good) is None
    assert Picky.checked == [(Picky.about, good)], "the backend's check is handed its About and the questions"
    with pytest.raises(ValueError, match="picky cannot take the question 'bad'"):
        decider.check_of(str(picky), bad)
    assert Picky.made == [], "nothing was constructed"

    gpu = Gpu(SimpleNamespace(home=home, decider_model=str(picky), decider_url=None, decider_envs=None, decider_idle_s=1))
    gpu.check(good)
    with pytest.raises(ValueError, match="cannot take"):
        gpu.check(bad)
    job = SimpleNamespace(_jobs=SimpleNamespace(gpu=gpu))
    with pytest.raises(ValueError, match="cannot take"):
        jobs.Job.check(job, bad)  # a Job reaches the Gpu as it does for `about`
    assert gpu.status()["state"] == "unloaded" and Picky.made == []

    before = len(Picky.checked)
    for nothing in (None, "", str(tmp_path / "missing"), str(model_folder(tmp_path, "unclaimed"))):
        assert decider.check_of(nothing, bad) is None, nothing
    assert len(Picky.checked) == before
    gpu.apply(SimpleNamespace(decider_model=None, decider_url=None, decider_envs=None, decider_idle_s=1))
    gpu.check(bad)  # no model is configured: `about` and the first window say so
    for call in (gpu.about, lambda: gpu.decide(None, good, [])):
        with pytest.raises(ApiError, match=r"no model is configured: the owner runs `mms model use NAME` "
                                           r"\(`mms model list` shows them\)") as raised:
            call()
        assert raised.value.code == "decider" and raised.value.hint == "ask the owner"

    ok = write_py(tmp_path, "checks", CHECKS)
    with pytest.raises(ValueError, match="checks cannot take bad"):
        decider.check_of("python:" + ok, bad)
    assert decider.check_of("python:" + ok, good) is None
    assert decider.check_of("python:" + write_py(tmp_path, "good", GOOD), bad) is None, "a file without a check takes all"
    with pytest.raises(ApiError, match="the decider could not load: ValueError: .*defines no ABOUT"):
        decider.check_of("python:" + write_py(tmp_path, "no_about", NO_ABOUT), good)


def test_the_queue_reloads_and_reads_facts_again_when_the_url_or_envs_change(tmp_path, monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", (f"{__name__}:Toy",))
    toy = folder_with(tmp_path, "toy", {})
    (toy / "toy.json").write_text("{}", encoding="utf-8")
    config = lambda url=None, envs=None: SimpleNamespace(  # noqa: E731
        home=tmp_path, decider_model=str(toy), decider_url=url, decider_envs=envs, decider_idle_s=1)
    made = []
    real = decider.load_decider
    monkeypatch.setattr(decider, "load_decider", lambda *args, **kwargs: made.append(kwargs) or real(*args, **kwargs))
    gpu = Gpu(config())
    gpu.about()
    gpu.about()
    assert made == [{"url": None, "envs": None}], "the facts are kept for one (model, url, envs)"
    gpu.apply(config())
    assert not gpu._reload, "the same model, url and envs is no change"
    gpu.apply(config(url="http://127.0.0.1:8741"))
    assert gpu._reload and (gpu.url, gpu.envs) == ("http://127.0.0.1:8741", None)
    gpu.about()
    assert made[-1] == {"url": "http://127.0.0.1:8741", "envs": None}, "a new url is read for again"
    gpu._reload = False
    gpu.apply(config(url="http://127.0.0.1:8741", envs="C:/envs"))
    assert gpu._reload and gpu.envs == "C:/envs"
    gpu.about()
    assert made[-1] == {"url": "http://127.0.0.1:8741", "envs": "C:/envs"}
    gpu._reload = False
    gpu.apply(config(url="http://127.0.0.1:8741", envs="C:/envs"))
    assert not gpu._reload
    gpu.apply(config(envs="C:/envs"))  # the url removed
    assert gpu._reload and gpu.url is None
    gpu.stop()


def test_an_env_is_found_beside_the_model_or_where_envs_says_and_is_prepared_for_exact_requirements(tmp_path):
    folder = tmp_path / "models" / "d3-edge"
    assert modelenv.folder(None, folder, "d3") == tmp_path / "models" / "envs" / "d3"
    assert modelenv.folder(str(tmp_path / "elsewhere"), folder, "d3") == tmp_path / "elsewhere" / "d3"
    assert modelenv.python(tmp_path / "e") == tmp_path / "e" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    pins = ("torch==2.11.0", "transformers==5.17.0")
    env_folder = tmp_path / "envs" / "d3"
    assert "not there" in modelenv.marker_problem(env_folder, pins), "no marker: not prepared"
    assert prepared_env(tmp_path / "envs", "d3", pins) == env_folder
    assert modelenv.marker_problem(env_folder, pins) is None
    assert "other requirements" in modelenv.marker_problem(env_folder, (*pins, "einops")), "mms was updated since"
    assert "other requirements" in modelenv.marker_problem(env_folder, pins[::-1]), "the pins are an ordered list"
    (env_folder / "mms-env.json").write_text("{", encoding="utf-8")
    assert "cannot be read" in modelenv.marker_problem(env_folder, pins)
    (env_folder / "mms-env.json").write_text("[]", encoding="utf-8")
    assert "other requirements" in modelenv.marker_problem(env_folder, pins)


def test_mms_python_is_python_even_under_the_detached_daemons_pythonw(monkeypatch):
    monkeypatch.setattr(modelenv.sys, "executable", "C:/mms/.venv/Scripts/pythonw.exe")
    assert modelenv.mms_python() == Path("C:/mms/.venv/Scripts/python.exe")
    monkeypatch.setattr(modelenv.sys, "executable", "/usr/bin/python3")
    assert modelenv.mms_python() == Path("/usr/bin/python3")


def test_the_median_level_is_d1s_readout_and_nothing_else_is_passed_off_as_it():
    level = native.median_level
    assert level({"0": 0.2, "1": 0.2, "2": 0.6}) == 2 and level({"0": 0.5, "1": 0.5}) == 0, "the first level that reaches half"
    assert level({"0": 0.4, "1": 0.2, "2": 0.4}) == 1 and level({"0": 1.0}) == 0
    assert level({"0": 2, "1": 2, "2": 6}) == 2, "d1 normalises, so does this"
    assert isinstance(level({"0": 0.2, "1": 0.8}), float)
    for bad in (None, {}, [0.2, 0.8], {"1": 0.5, "2": 0.5}, {"0": 0.5, "2": 0.5}, {0: 0.5, 1: 0.5}, {"a": 1.0},
                {"0": 0.5, "1": None}, {"0": True, "1": 0.5}, {"0": math.nan, "1": 0.5}, {"0": -0.1, "1": 1.1},
                {"0": 0.0, "1": 0.0}):
        assert level(bad) is None, bad


def test_the_helpers_write_the_files_each_claim_reads(tmp_path):
    for name in MODEL_FOLDERS:
        folder = model_folder(tmp_path, name)
        assert folder == tmp_path / name and (folder / "config.json").is_file()

    def read(folder: str, file: str) -> dict:
        return json.loads((tmp_path / folder / file).read_text(encoding="utf-8"))

    assert read("d1-2B", "config.json")["model_type"] == "lfm3_vl"
    assert read("d1-3B", "config.json")["auto_map"] == {"AutoModel": "modeling_d1.D1Model"}
    assert read("d1-3B", "config.json")["text_config"]["hidden_size"] == 2048
    assert read("d1-Omni-2B", "config.json")["model_type"] == "lfm3_omni"
    assert read("d3-mini", "MODEL_MANIFEST.json")["model_name"] == "d3-mini"
    assert read("d3-flash", "decision_config.json")["format_id"] == "d3-code-readout-v1"
    assert read("h2o-lightning-4b", "serve_config.json")["model"] == "h2oai/h2o-lightning-4b"
    assert (tmp_path / "h2o-lightning-4b" / "h2o_lightning_shim.py").is_file()
    assert read("pplx-decider-flash-v1.2", "joint_config.json")["version"] == 3
    assert read("pplx-decider-flash-v1.2", "decision_config.json")["max_pixels"] == 1638400
    assert (tmp_path / "pplx-decider-flash-v1.2" / "source" / "src" / "pplx_decider" / "__init__.py").is_file()
    assert decider.about_of(str(tmp_path / "unclaimed")) is None
    for name in MODEL_FOLDERS:  # an empty file stands for the weights, so a folder reads as downloaded and whole
        assert (tmp_path / name / "model.safetensors").exists() == (name != "unclaimed"), name
        if name != "unclaimed":
            assert (tmp_path / name / "model.safetensors").stat().st_size == 0, "never real weights"
    with pytest.raises(ValueError, match="model_folder knows"):
        model_folder(tmp_path, "gpt-2")


def test_every_known_model_names_its_download_and_no_name_repeats():
    abouts = decider.known_abouts()
    assert len(abouts) == 8, [about.name for about in abouts]
    for about in abouts:
        assert decider.REPO.fullmatch(about.repo) and about.repo_gb > 0, f"{about.name}: {about.repo!r}, {about.repo_gb}"
        assert re.fullmatch(r"[0-9a-f]{40}", about.commit), f"{about.name} pins no commit of {about.repo}: {about.commit!r}"
    names = [about.name.lower() for about in abouts]
    assert len(set(names)) == len(names), "a name is one model's, whatever its case"
    assert {about.name: (about.repo, about.repo_gb) for about in abouts} == {
        "d1-Omni-2B": ("LiquidAI/d1-Omni-2B", 4.4), "d1-2B": ("LiquidAI/d1-2B", 3.9), "d1-3B": ("LiquidAI/d1-3B", 6.3),
        "d3-edge": ("vllm-sr/d3-edge", 1.2), "d3-mini": ("vllm-sr/d3-mini", 9.1), "d3-flash": ("vllm-sr/d3-flash", 16.8),
        "pplx-decider-flash-v1.2": ("perplexity-ai/pplx-decider-flash-v1.2", 9.1),
        "h2o-lightning-4b": ("h2oai/h2o-lightning-4b", 9.4)}
    assert {about.name: about.commit for about in abouts} == {
        "d1-Omni-2B": "c8d48c08dada0191c6b362820f055cfb49c99426", "d1-3B": "051bcc464b01b9f92942b364d9586b0ef5912432",
        "d1-2B": "10e7671d44e4b37700020b1eed3f594c5f2af0b4", "d3-edge": "c454ee39b862248d70abd7f84195d2f54ffe6d43",
        "d3-mini": "df99bf4ca4418bc6908ee68e9b7479af1f3b8800", "d3-flash": "13dc1cc8ef23caefa765907e1d764751e5629f7d",
        "pplx-decider-flash-v1.2": "9ff7cf9838e59df461c276b7699d56b244d2af58",
        "h2o-lightning-4b": "659fc412de8486dbc2477b0d1cb4121e97e72471"}, "the commits that were tested, and run on the machine"
    assert decider.known("D1-3b") == (decider_d1_3b.D1_3B, decider_d1_3b.ABOUT)
    assert decider.known("d3-FLASH") == (decider_d3.D3, decider_d3.FLASH)
    assert decider.known("h2o-lightning-4b") == (decider_h2o.H2o, decider_h2o.ABOUT)
    for unknown in ("d1", "d1-3", "d1-3B ", "", "python:x.py"):
        assert decider.known(unknown) is None, unknown


# -- the facts --------------------------------------------------------------------------------------------------------

BAD_FACTS = [
    ("name", {"name": ""}), ("name", {"name": "  "}), ("name", {"name": 3}),
    ("reads", {"reads": {"smell"}}), ("reads", {"reads": "video"}), ("reads", {"reads": {"text"}}),
    ("types", {"types": ("poem",)}), ("types", {"types": ()}),
    ("extracts_from", {"extracts_from": {"nose"}}),
    ("files", {"files": ("",)}), ("files", {"files": (3,)}),
    ("audio_rate", {"audio_rate": 0}), ("audio_rate", {"audio_rate": 16000.5}), ("audio_rate", {"audio_rate": True}),
    ("max_px", {"max_px": -1}), ("questions_per_read", {"questions_per_read": -1}),
    ("picture_s", {"picture_s": ()}), ("picture_s", {"picture_s": ((200, 1.0), (100, 1.0))}),
    ("picture_s", {"picture_s": ((100, 1.0), (100, 2.0))}), ("picture_s", {"picture_s": ((100, -1.0),)}),
    ("picture_s", {"picture_s": ((100,),)}),
    ("sound_s", {"sound_s": -1}), ("char_s", {"char_s": -0.1}), ("file_s", {"file_s": -1}),
    ("sound_s", {"sound_s": math.nan}), ("char_s", {"char_s": math.inf}), ("file_s", {"file_s": -math.inf}),
    ("picture_s", {"picture_s": ((100, math.nan),)}), ("picture_s", {"picture_s": ((math.inf, 1.0),)}),
    ("picture_s", {"picture_s": ((100, 1.0), (200, math.inf))}),
    ("speech", {"speech": 1}), ("speech", {"speech": "yes"}), ("speech", {"speech": None}),
    ("extracts_from", {"extracts_from": ()}), ("extracts_from", {"extracts_from": set(), "types": ("noul", "number")}),
    ("profile", {"profile": [("max_px", 1)]}),
    ("hit", {"hit": 0}), ("hit", {"hit": 1.5}), ("hit", {"hit": -0.1}), ("hit", {"hit": True}), ("hit", {"hit": "0.8"}),
    ("hit", {"hit": math.nan}),
    ("question_keys", {"question_keys": {"colour"}}), ("question_keys", {"question_keys": "criteria"}),
    ("question_keys", {"question_keys": 3}),
    ("measured", {"measured": {"speed"}}), ("measured", {"measured": "hit"}),
    ("notes", {"notes": "models/d1.md"}), ("notes", {"notes": "..\\d1.md"}), ("notes", {"notes": ".."}),
    ("notes", {"notes": " d1.md"}), ("notes", {"notes": 3}), ("notes", {"notes": None}),
    ("repo", {"repo": "x"}), ("repo", {"repo": "a/b/c"}), ("repo", {"repo": 3}), ("repo", {"repo": None}),
    ("repo", {"repo": "a/b\n"}), ("repo", {"repo": " a/b"}), ("repo", {"repo": ".a/b"}), ("repo", {"repo": "a/"}),
    ("repo", {"repo": "a//b"}), ("repo", {"repo": "https://huggingface.co/a/b"}),
    ("commit", {"repo": "a/b"}), ("commit", {"commit": "0" * 40}), ("commit", {"repo": "a/b", "commit": "main"}),
    ("commit", {"repo": "a/b", "commit": "0" * 39}), ("commit", {"repo": "a/b", "commit": "0" * 41}),
    ("commit", {"repo": "a/b", "commit": "A" * 40}), ("commit", {"repo": "a/b", "commit": "0" * 39 + "g"}),
    ("commit", {"repo": "a/b", "commit": 3}), ("commit", {"repo": "a/b", "commit": None}),
    ("commit", {"repo": "a/b", "commit": "0" * 40 + "\n"}),
    ("repo_gb", {"repo_gb": -1}), ("repo_gb", {"repo_gb": math.inf}), ("repo_gb", {"repo_gb": True}),
    ("repo_gb", {"repo_gb": math.nan}), ("repo_gb", {"repo_gb": "6.3"}),
]


@pytest.mark.parametrize("field,facts", BAD_FACTS, ids=[f"{field}-{i}" for i, (field, _) in enumerate(BAD_FACTS)])
def test_about_refuses_what_is_not_a_fact(field, facts):
    with pytest.raises(ValueError, match=rf"About\.{field}\b"):
        About(**{"name": "x", **facts})


def test_about_has_the_new_facts_with_defaults_and_stays_hashable():
    about = About(name="x")
    assert (about.hit, about.question_keys, about.notes, about.measured) == (
        0.8, frozenset(decider.QUESTION_KEYS), "", frozenset()), "d1's own defaults, and nothing measured"
    assert (about.repo, about.repo_gb) == ("", 0.0), "a decider mms does not download names no repo"
    assert (about.commit, About(name="c", repo="o/n", commit="ab" * 20).commit) == ("", "ab" * 20), "a repo is pinned to a commit"
    assert About(name="r", repo="owner/name.v1_2-3", commit="0" * 40, repo_gb=0).repo == "owner/name.v1_2-3"
    assert About(name="r", repo="o/n", commit="0" * 40, repo_gb=6.3).repo_gb == 6.3 and About(name="r", repo_gb=2).repo_gb == 2
    given = About(name="y", hit=1, question_keys=["criteria"], notes="d1.md", measured=("hit", "picture_s"))
    assert given.hit == 1 and given.question_keys == frozenset({"criteria"}) and given.notes == "d1.md"
    assert given.measured == frozenset({"hit", "picture_s"})
    assert About(name="z", question_keys=()).question_keys == frozenset(), "a model that takes no key besides type"
    hash((given.hit, given.question_keys, given.notes, given.measured))  # the new facts are all hashable
    assert set(decider.MEASURABLE) == {"picture_s", "sound_s", "char_s", "file_s", "hit"}


def test_about_is_frozen_and_takes_media_by_exact_type_or_prefix():
    profile = {"max_px": 1000}
    about = About(name="x", reads=["image"], files=["application/pdf", "image/", "application/vnd.oasis."], profile=profile)
    profile["max_px"] = 5
    assert about.profile == {"max_px": 1000} and about.reads == frozenset({"image"}) and about.files[0] == "application/pdf"
    with pytest.raises(dataclasses.FrozenInstanceError):
        about.name = "y"
    taken = ["application/pdf", "image/heic", "image/png", "application/vnd.oasis.opendocument.text"]
    refused = ["application/pdfx", "application/x-pdf", "audio/wav", "application/vnd.oasis", None, "", "text/plain"]
    assert [about.takes(media) for media in taken] == [True] * 4
    assert [about.takes(media) for media in refused] == [False] * len(refused)
    assert not About(name="y").takes("application/pdf"), "a model that lists no file takes none"
    with pytest.raises(ValueError, match=r"z's About\.profile has \['colour'\]; a profile has"):  # the reader checks the keys
        reader.profile_of(None, About(name="z", profile={"colour": 1}))
    with pytest.raises(ValueError, match="a profile has") as raised:
        reader.profile_of({"colour": 1}, About(name="z"))
    assert "About.profile" not in str(raised.value) and "z's" not in str(raised.value), \
        "an asker's own bad key does not blame the decider"
    assert About(name="q", extracts_from=(), types=("noul", "choice", "score")).extracts_from == frozenset(), \
        "a decider that answers none of the extracting types needs no kind to extract from"


def test_about_profile_cannot_be_written_by_a_script():
    about = About(name="x", profile={"max_px": 1000})
    with pytest.raises(TypeError):
        about.profile["max_px"] = 5
    with pytest.raises(TypeError):
        about.profile["every_s"] = 5
    assert not hasattr(about.profile, "update") and about.profile == {"max_px": 1000}
    assert decider_d1.ABOUT.profile == {} and other_decider.ABOUT.profile["max_px"] == 102_400
    used = reader.profile_of(None, about)
    assert type(used) is dict and used["max_px"] == 1000 and used is not reader.profile_of(None, about), \
        "a scan's own profile is a plain dict a script may change"
    used["max_px"] = 7
    assert reader.profile_of(None, about)["max_px"] == 1000
    assert dataclasses.replace(about, name="y").profile == {"max_px": 1000}, "a copy of an About takes its read-only profile"
    with pytest.raises(ValueError, match=r"About\.profile"):
        About(name="x", profile=[("max_px", 1)])


def test_a_scans_max_px_is_cut_to_the_models_cap_and_0_is_the_cap():
    cap = About(name="capped", max_px=500_000)
    assert reader.profile_of(None, cap)["max_px"] == reader.PROFILE["max_px"] == 230_400, "the default is under the cap"
    assert reader.profile_of({"max_px": 300_000}, cap)["max_px"] == 300_000
    assert reader.profile_of({"max_px": 500_000}, cap)["max_px"] == 500_000
    assert reader.profile_of({"max_px": 2_000_000}, cap)["max_px"] == 500_000, "above the cap is the cap"
    assert reader.profile_of({"max_px": 0}, cap)["max_px"] == 500_000, "0 is the cap, not as stored"
    assert reader.profile_of(None, About(name="small", max_px=100_000))["max_px"] == 100_000, "the default is cut too"
    assert reader.profile_of(None, About(name="zero", max_px=500_000, profile={"max_px": 0}))["max_px"] == 500_000
    uncapped = About(name="uncapped")
    assert reader.profile_of({"max_px": 0}, uncapped)["max_px"] == 0, "a model with no cap reads 0 as stored"
    assert reader.profile_of({"max_px": 2_000_000}, uncapped)["max_px"] == 2_000_000
    with pytest.raises(ValueError, match="max_px"):
        reader.profile_of({"max_px": -1}, cap)
    assert reader.profile_of({"max_px": 0}, decider_d1.ABOUT)["max_px"] == decider_d1.ABOUT.max_px == 1_638_400
    assert reader.profile_of({"max_px": 0}, decider_d1_3b.ABOUT)["max_px"] == 1_048_576


def test_picture_seconds_matches_d1s_two_measured_sizes():
    about = decider_d1.ABOUT
    (low, fast), (high, slow) = about.picture_s
    assert (low, fast, high, slow) == (230_400, 0.053, 1_638_400, 0.4834), "640x360 and 1.6 MP, as measured"
    assert about.picture_seconds(low) == pytest.approx(fast) and about.picture_seconds(high) == pytest.approx(slow)
    assert about.picture_seconds((low + high) / 2) == pytest.approx((fast + slow) / 2)
    assert about.picture_seconds(0) == fast and about.picture_seconds(1) == fast, "flat below the first point"
    assert about.picture_seconds(100_000_000) == slow, "flat above the last"
    assert fast < about.picture_seconds(921_600) < slow
    assert About(name="one", picture_s=((1000, 0.3),)).picture_seconds(5) == 0.3
    assert About(name="one", picture_s=((1000, 0.3),)).picture_seconds(10**9) == 0.3


def test_the_two_test_deciders_have_facts_that_differ():
    """`other` is no copy of d1: every fact mms reads differs, bar the media it reads, which the reader tests vary."""
    assert FAKE_ABOUT is decider_d1.ABOUT, "the fake decider is d1's facts"
    same = [f.name for f in dataclasses.fields(About)
            if getattr(other_decider.ABOUT, f.name) == getattr(FAKE_ABOUT, f.name)]
    assert same == ["reads"], f"other and d1 share {same}"


# -- the reader cuts for the facts --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    """Six seconds each: a tone (every audio window is loud), a red video with the tone for its soundtrack, and a blue one
    with no sound."""
    folder = tmp_path_factory.mktemp("clips")
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6"]
    made = {"tone": folder / "tone.wav", "talk": folder / "talk.mp4", "screen": folder / "screen.mkv"}
    run(*tone, "-ac", "1", made["tone"])
    run("-f", "lavfi", "-i", "color=c=red:s=160x90:r=5:d=6", *tone, *OUT, *X264, "-c:a", "aac", made["talk"])
    run("-f", "lavfi", "-i", "color=c=blue:s=160x90:r=5:d=6", *OUT, *X264, made["screen"])
    return made


def indexed(store: Store, source: str, kind: str, file: Path, t0: float, **who) -> int:
    """A copy of a media file in the home as a segment of `source` that starts at `t0` and is six seconds long."""
    target = store.streams / source / "launch" / f"{int(t0)}{file.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(file.read_bytes())
    return store.add_item(source, kind, t0, t0 + 6, store.relative(target), target.stat().st_size, **who)


@pytest.fixture
def filled(home, corpus, clips):
    """A store of a picture, a sound, a video with sound, a video without, a PDF and a text, and their item ids."""
    store = Store(home)
    ids = {"picture": store.add_version("docs", "card.jpg", T0, data=corpus["pictures/progressive.jpg"].path.read_bytes()),
           "sound": indexed(store, "mic", "audio", clips["tone"], T0 + 100),
           "talk": indexed(store, "meet", "video", clips["talk"], T0 + 200, meta={"sound": True}),
           "screen": indexed(store, "screen", "video", clips["screen"], T0 + 300),
           "pdf": store.add_version("docs", "invoice.pdf", T0 + 400, data=corpus["documents/invoice.pdf"].path.read_bytes()),
           "text": store.add_version("docs", "note.txt", T0 + 500, data=b"the kettle is on\n")}
    yield store, ids
    store.close()


def cut(store: Store, about: About, **options):
    """Every window `reader.windows` gives under `about`, and the ids of the items it reported as not read."""
    unread, unreadable = [], []
    found = list(reader.windows(store, about=about, scope=EVERYTHING, newest_first=False, on_unread=lambda i: unread.append(i.id),
                                on_unreadable=lambda i, e: unreadable.append((i.id, str(e))), **options))
    assert not unreadable, unreadable
    return found, unread


def counted(found) -> dict[str, int]:
    return {kind: Counter(w.kind for w in found).get(kind, 0) for kind in ("video", "audio", "image", "file", "text", "records")}


def test_the_reader_hands_each_decider_what_it_declares_and_names_the_rest(filled):
    store, ids = filled
    eyes, ears = About(name="eyes", reads={"video", "image"}), About(name="ears", reads={"audio"})
    docs = About(name="docs", reads=set(), files=("application/pdf",))
    for about, windows, unread in (
            (eyes, {"video": 12, "audio": 0, "image": 1, "file": 0, "text": 1, "records": 0}, {"sound", "pdf"}),
            (ears, {"video": 0, "audio": 4, "image": 0, "file": 0, "text": 1, "records": 0}, {"picture", "screen", "pdf"}),
            (docs, {"video": 0, "audio": 0, "image": 0, "file": 1, "text": 1, "records": 0},
             {"picture", "sound", "talk", "screen"}),
            (decider_d1.ABOUT, {"video": 12, "audio": 4, "image": 1, "file": 0, "text": 1, "records": 0}, {"pdf"})):
        found, left = cut(store, about, profile={"still_s": 0})  # a window a frame the step picks, as the estimate counts
        assert counted(found) == windows, f"{about.name}: {counted(found)}"
        assert set(left) == {ids[name] for name in unread}, \
            f"{about.name} was not given {sorted(unread)}, but reported {sorted(set(left))} of ids {ids}"
        assert len(left) == len(set(left)), f"{about.name} reported an item twice"
        estimate = reader.estimate(store, None, None, None, EVERYTHING, about=about)
        assert {kind: n for kind, n in estimate.items() if kind != "audio"} == \
            {kind: n for kind, n in windows.items() if kind != "audio"}, f"{about.name}: estimated {estimate}, read {windows}"
        # a 6 s sound over a 2.5 s hop is counted in 3 windows and read in 2 (the last is set back to end with the sound)
        assert windows["audio"] <= estimate["audio"] <= windows["audio"] + 2, f"{about.name}: {estimate} for {windows}"
        kinds = {}
        for w in found:
            kinds.setdefault(w.item.id, set()).add(w.kind)
        if about is eyes:
            assert kinds[ids["talk"]] == {"video"}, "a video's soundtrack is not read by a decider that does not hear"
        if about is ears:
            assert kinds[ids["talk"]] == {"audio"}, "a video with sound is read for its soundtrack alone"
        if about is decider_d1.ABOUT:
            assert kinds[ids["talk"]] == {"video", "audio"}, "d1 reads both"

    (window,) = [w for w in cut(store, docs)[0] if w.kind == "file"]
    pdf = store.item(ids["pdf"])
    assert window.media == {"type": "file", "media": "application/pdf", "name": "invoice.pdf",
                            "data": store.path_of(pdf).read_bytes()}
    assert window.t0 == window.t1 == pdf.t0 == T0 + 400 and window.at is None and not window.text and window.item.id == pdf.id


def test_sound_is_cut_at_the_rate_and_in_the_windows_a_decider_declares(filled):
    store, ids = filled
    slow = About(name="slow", reads={"audio"}, audio_rate=8000, profile={"audio_s": 2.0, "audio_hop_s": 1.0})
    found, _ = cut(store, slow)
    assert {w.item.id for w in found if w.kind == "audio"} == {ids["sound"], ids["talk"]}, "the sound and the soundtrack"
    for w in found:
        if w.kind == "audio":
            assert w.media["sampling_rate"] == 8000 and len(w.media["audio"]) == 16_000 == 2.0 * 8000, \
                f"a window of {len(w.media['audio'])} samples at {w.media['sampling_rate']} Hz"
    counts = reader.estimate(store, None, None, None, EVERYTHING, about=slow)
    read = counted(found)["audio"]
    assert 10 <= read <= counts["audio"] <= read + 2 and counts["audio"] >= 12, \
        f"a 6 s sound at a 1 s hop is counted as 6 windows, and read as 5 or 6: {read} read, {counts['audio']} counted"
    # the model's profile applies, and an ask's own override beats it
    assert reader.profile_of(None, slow)["audio_s"] == 2.0 and reader.profile_of({"audio_s": 4.0}, slow)["audio_s"] == 4.0
    assert reader.profile_of(None, slow)["every_s"] == reader.PROFILE["every_s"], "what it leaves out is mms's"
    longer, _ = cut(store, slow, profile={"audio_s": 4.0})
    assert {len(w.media["audio"]) for w in longer if w.kind == "audio"} == {32_000}, "the ask's audio_s beats the model's"
    d1_found, _ = cut(store, decider_d1.ABOUT)
    assert {(w.media["sampling_rate"], len(w.media["audio"])) for w in d1_found if w.kind == "audio"} == {(16000, 80_000)}


# -- the same ask under two deciders --------------------------------------------------------------------------------------

DOCUMENTS = ["invoice.pdf", "invoice-copy.txt", "scan.pdf", "locked.pdf", "minutes.docx", "budget.xlsx", "deck.pptx",
             "notes.odt", "sheet.ods", "book.epub", "legacy.doc", "meeting.msg", "letter.rtf", "dentist.ics", "sam.vcf",
             "gate.eml", "song.mid", "backup.tar", "logs.tar.bz2", "dump.xz", "history.sqlite", "photos.7z"]
RELS = ["pictures/screenshot.png", "sound/voice_memo.m4a", "text/min.json", "bad/IMG_0001.HEIC",
        *[f"documents/{name}" for name in DOCUMENTS]]


class World:
    def __init__(self, home: Path, api, store: Store, times: dict, spec: str):
        self.home, self.api, self.store, self.times, self.spec = home, api, store, times, spec
        self.end = T0 + GAP * len(times)

    def ask(self, **params) -> dict:
        return scan(self.api, self.end, limit=50, candidates=20, **params)

    def log(self, file: str) -> list[dict]:
        path = self.home / file
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@pytest.fixture
def worlds(tmp_path, corpus):
    """`start(name, spec, rels)`: a daemon of its own under the decider `spec`, holding `rels` of the corpus."""
    running: list[tuple[Served, Store]] = []

    def start(name: str, spec: str, rels=RELS) -> World:
        spec = spec.replace(chr(92), "/")  # the path of a TOML string
        home = tmp_path / name
        home.mkdir()
        write_config(home, BASE + f'[decider]\nmodel = "{spec}"\n')
        served = Served(home).start()
        store = Store(home)
        running.append((served, store))
        return World(home, served.api, store, remember_each(served.api, corpus, rels), spec)

    yield start
    for served, store in running:
        store.close()
        served.stop()


def unread_media(world: World, corpus, unread) -> dict[str, set[int]]:
    """The documents of RELS that `unread(rel)` says a decider does not read, by media type, with the ids of their items."""
    ids = {item.name: item.id for item in world.store.items_of("agent")}
    expected: dict[str, set[int]] = {}
    for rel in RELS:
        if rel.startswith("documents/") and unread(rel):
            expected.setdefault(corpus[rel].media, set()).add(ids[corpus[rel].name])
    return expected


def test_the_same_ask_under_two_deciders_gets_what_each_declares(worlds, corpus):
    d1, other = worlds("d1", FAKE), worlds("other", OTHER)
    under_d1, under_other = d1.ask(question="contains:342.50"), other.ask(question="contains:342.50")
    taken = [rel for rel in RELS if rel.startswith("documents/") and other_decider.ABOUT.takes(corpus[rel].media)]
    taken.append("bad/IMG_0001.HEIC")

    # d1 is given no file, and the warning names every document it does not read
    warnings_d1 = "\n".join(under_d1.get("warnings", []))
    assert under_d1["sightings"] == [], f"a PDF's words never reach d1: {under_d1['sightings']}"
    (unread_d1,) = not_read_warnings(under_d1)
    check_unread(unread_d1, "d1-Omni-2B", unread_media(d1, corpus, lambda rel: corpus[rel].read is None))
    assert "skipped" not in warnings_d1, f"d1 was handed a window it could not take: {warnings_d1}"
    assert "pi-heif" in warnings_d1, "an HEIC is an image to d1, which cannot open it"
    log_d1 = d1.log("fake-decider.jsonl")
    assert max(max((w * h for w, h in call["pictures"]), default=0) for call in log_d1) == 640 * 360
    assert {(call["heard"][0]["rate"], call["sounds"][0]) for call in log_d1 if call["heard"]} == {(16000, 80_000)}
    assert max(call["state_len"] for call in log_d1) > 1000, "d1 reads text 6,000 characters at a time"

    # other is given what it declares, as it declares it
    warnings_other = "\n".join(under_other.get("warnings", []))
    (unread_other,) = not_read_warnings(under_other)
    check_unread(unread_other, "other", unread_media(other, corpus, lambda rel: corpus[rel].read is None
                                                     and not other_decider.ABOUT.takes(corpus[rel].media)))
    assert "pi-heif" not in warnings_other, f"other takes HEIC whole: {warnings_other}"
    assert "unreadable" not in warnings_other and "skipped" not in warnings_other, warnings_other
    log_other = other.log("other-decider.jsonl")
    sounds = [sound for call in log_other for sound in call["sounds"]]
    assert sounds and all(sound == {"len": 16_000, "rate": 8000} for sound in sounds), Counter(map(json.dumps, sounds))
    pictures = [w * h for call in log_other for w, h in call["pictures"]]
    assert pictures and max(pictures) <= 102_400 and max(pictures) > 80_000, f"pictures of {sorted(set(pictures))} px"
    assert 0 < max(call["state_len"] for call in log_other) <= 1100, "text arrives about 1,000 characters at a time"
    given = Counter((f["name"], f["media"], f["sha256"], f["bytes"]) for call in log_other for f in call["files"])
    expected = Counter((corpus[rel].name, corpus[rel].media, hashlib.sha256(corpus[rel].path.read_bytes()).hexdigest(),
                        corpus[rel].path.stat().st_size) for rel in taken)
    assert given == expected, f"each declared file arrives once and whole: missing {expected - given}, extra {given - expected}"
    assert d1.log("other-decider.jsonl") == [] and other.log("fake-decider.jsonl") == [], "each ran its own decider"

    # what each found (the sightings and their labels are test_documents's)
    assert names_hit(other.store, under_other) == {"invoice.pdf", "invoice-copy.txt"}
    assert d1.store.labels(by="ask") == [], "nothing was found under d1"


def lints(world: World, **params) -> str:
    return "\n".join(scan(world.api, world.end, **params).get("warnings", []))


def test_a_narrower_decider_is_refused_before_the_scan_and_linted_by_its_own_facts(worlds):
    rels = ["pictures/IMG_2041.jpg", "sound/voice_memo.m4a"]
    d1, other = worlds("d1", FAKE, rels), worlds("other", OTHER, rels)
    five = {f"q{n}": {"type": "noul", "instructions": "contains:kettle", "hit": True} for n in range(5)}
    assert "5 questions: other reads the window again for each 4, so this takes several times as long" in lints(
        other, questions=five), "other reads 4 questions in a pass"
    assert "reads the window again" not in lints(d1, questions=five), "d1 reads 16"

    said = "What was said in the memo?"
    assert "speech words over audio: d1-Omni-2B hears sounds better than speech" in lints(d1, question=said)
    assert "speech words" not in lints(other, question=said), "other understands speech"

    # a type it does not answer is refused at once, naming the decider and what it answers; d1 answers it
    box = {"b": {"type": "box", "instructions": "Is there a mug?", "hit": True}}
    scanned = len(other.log("other-decider.jsonl"))
    job = other.api.call("jobs/start", script="ask", params={"questions": box, "since": T0 - 100}, wait_s=40)
    assert job["state"] == "failed" and job["error"]["code"] == "bad_request", job
    assert "question 'b' is a box, which other does not answer; it answers noul, choice, score" in job["error"]["message"]
    assert len(other.log("other-decider.jsonl")) == scanned, "nothing was scanned"
    assert scan(d1.api, d1.end, questions=box)["progress"]["windows"] > 0, "d1 answers a box"


def test_lint_reads_what_a_decider_extracts_from_its_facts(home):
    ask = jobs.load_script(home, "ask")
    text = {"t": {"type": "text", "instructions": "What does the sign say?", "hit": True}}
    pictures = {"video": 0, "audio": 0, "image": 3, "file": 0, "text": 0, "records": 0}
    sounds = {**pictures, "image": 0, "audio": 4}

    def warned(about: About, counts: dict) -> str:
        plan = ask.Plan({"questions": text, "since": T0 - 10, "until": T0}, T0)
        return "\n".join(ask.lint(plan, counts, reader.profile_of(None, about), about))

    reads_pictures = dataclasses.replace(other_decider.ABOUT, types=decider.TYPES)  # a decider that reads text off a picture
    assert "a text, date or number with no range, and d1-Omni-2B reads those only from records and text; " \
           "it will find nothing in image" in warned(decider_d1.ABOUT, pictures)
    assert "no range" not in warned(reads_pictures, pictures), "its facts say it reads text from pictures"
    assert "it will find nothing in audio" in warned(reads_pictures, sounds), "but not from sound"
    assert "reads those only from file and image and records and text and video" in warned(reads_pictures, sounds)


def test_a_file_windows_cost_is_the_deciders_file_s_and_max_s_lints_a_range_of_them(home, worlds):
    ask = jobs.load_script(home, "ask")
    other_facts, d1_facts = other_decider.ABOUT, decider_d1.ABOUT
    profile = reader.profile_of(None, other_facts)
    assert ask.window_seconds("file", profile, other_facts) == other_facts.file_s == 0.2
    assert ask.window_seconds("file", profile, d1_facts) == d1_facts.file_s != other_facts.file_s
    slow = dataclasses.replace(other_facts, file_s=70.0)
    assert ask.window_seconds("file", profile, slow) == 70.0, "a file window costs what its decider's facts say"

    plan = ask.Plan({"questions": {"q": {"type": "noul", "instructions": "Is there a mug?", "hit": True}}, "since": T0 - 10,
                     "until": T0}, T0)
    ten = {"video": 0, "audio": 0, "image": 0, "file": 10, "text": 0, "records": 0}
    def warned(plan, counts: dict, about: About) -> str:
        return "\n".join(ask.lint(plan, counts, reader.profile_of(None, about), about))

    assert "about 10 windows, about 12 min against max_s of 600 s" in warned(plan, ten, slow)
    assert "against max_s" not in warned(plan, ten, other_facts), "ten files at 0.2 s are 2 s"
    # the estimate is in seconds under about 90 s, so a short one does not read "about 0 min"
    short = dataclasses.replace(other_facts, file_s=4.5)
    tight = ask.Plan({"questions": {"q": {"type": "noul", "instructions": "Is there a mug?", "hit": True}}, "since": T0 - 10,
                      "until": T0, "max_s": 10}, T0)
    assert "about 10 windows, about 45 s against max_s of 10 s: narrow" in warned(tight, ten, short)
    edge = dataclasses.replace(other_facts, file_s=8.9)
    assert "about 89 s against max_s of 10 s" in warned(tight, ten, edge)
    minute = dataclasses.replace(other_facts, file_s=9.0)
    assert "about 2 min against max_s of 10 s" in warned(tight, ten, minute), "90 s and over is in minutes"

    # end to end: every document of the corpus, under other, which reads 11 of them whole
    rels = [rel for rel in RELS if rel.startswith("documents/")] + ["bad/IMG_0001.HEIC"]
    other = worlds("other", OTHER, rels)
    counts = reader.estimate(other.store, None, None, None, EVERYTHING, about=other_facts)
    windows = sum(counts.values())
    assert counts["file"] == 11 and counts["text"] > 0, counts  # the four text documents are cut at 1,000 characters
    seconds = sum(ask.window_seconds(kind, profile, other_facts) * n for kind, n in counts.items())
    assert 2.0 < seconds < 3.0, f"11 files at 0.2 s and the texts: {seconds} s"
    assert f"about {windows} windows, about {seconds:.0f} s against max_s of 1 s: narrow the range" in lints(
        other, question="contains:342.50", max_s=1), "the range of file windows is more than max_s"
    assert "against max_s" not in lints(other, question="contains:342.50", max_s=3)
    # the same store under d1's facts has no file windows to count, and the same max_s is enough
    d1_counts = reader.estimate(other.store, None, None, None, EVERYTHING, about=d1_facts)
    assert d1_counts["file"] == 0
    one_second = ask.Plan({"questions": {"q": {"type": "noul", "instructions": "Is there a mug?", "hit": True}},
                           "since": T0 - 10, "until": T0, "max_s": 1}, T0)
    assert "against max_s" not in warned(one_second, d1_counts, d1_facts)
