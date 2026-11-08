"""Perplexity's backend and its worker half (TESTBED.md, "No GPU"; DESIGN.md §13.3): how a folder is claimed, the facts, the
env check, and the two translations, from canned answers in the shapes of `pplx_decider.protocol.answer`. `worker/pplx.py`
runs in this process against a fake `pplx_decider` package in a temporary folder, so its `sys.path` insert and import are
the real ones. No model, no GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from conftest import MODEL_FOLDERS, ROOT, model_folder, prepared_env
from PIL import Image
from test_family_d3 import Worker, ask, load_worker, verdict  # noqa: F401  helpers, and the `ask` fixture

from mms import decider, modelenv
from mms.decider_pplx import ABOUT, Pplx

WORKER = ROOT / "src" / "mms" / "worker"


def backend(tmp_path: Path) -> Pplx:
    folder = model_folder(tmp_path / "models", "pplx-decider-flash-v1.2")
    prepared_env(tmp_path / "models" / "envs", "pplx", Pplx.requirements)
    return Pplx(folder)


# pplx_decider.protocol.answer: a choice's confidence is (pmax - 1/n) / (1 - 1/n), a score has a legend
NOUL = {"type": "noul", "noul": 0.91}
CHOICE = {"type": "choice", "choice": "cup", "probabilities": {"mug": 0.1, "cup": 0.7, "none": 0.2}, "confidence": 0.55}
SCORE = {"type": "score", "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6}, "legend": {"0": "no", "1": "some", "2": "all"},
         "score": 1.6, "confidence": 0.4}
QUESTIONS = {"seen": {"type": "noul", "instructions": "Is a mug in view?"},
             "which": {"type": "choice", "instructions": "Which?", "criteria": {"mug": None, "cup": "a cup", "none": None}},
             "amount": {"type": "score", "instructions": "How much?", "criteria": ["no", "some", "all"]}}


# -- claims and facts -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODEL_FOLDERS)
def test_a_pplx_folder_is_claimed_by_its_protocol_and_package_and_no_other_folder_is(tmp_path, name):
    assert Pplx.claims(model_folder(tmp_path, name)) is (ABOUT if name == "pplx-decider-flash-v1.2" else None), name


def test_a_folder_without_protocol_version_3_or_its_package_is_not_claimed(tmp_path):
    folder = model_folder(tmp_path, "pplx-decider-flash-v1.2")
    joint = json.loads((folder / "joint_config.json").read_text(encoding="utf-8"))
    for changed in ({"version": 2}, {"template": "other"}):
        (folder / "joint_config.json").write_text(json.dumps({**joint, **changed}), encoding="utf-8")
        assert Pplx.claims(folder) is None, changed
    (folder / "joint_config.json").write_text(json.dumps(joint), encoding="utf-8")
    assert Pplx.claims(folder) is ABOUT
    (folder / "source" / "src" / "pplx_decider" / "__init__.py").unlink()
    assert Pplx.claims(folder) is None, "without its package the folder cannot be run"
    (folder / "joint_config.json").write_text("[]", encoding="utf-8")
    assert Pplx.claims(folder) is None
    (folder / "joint_config.json").unlink()
    assert Pplx.claims(folder) is None and Pplx.claims(folder / "missing") is None


def test_the_loader_finds_pplx_by_what_claims_it(home, tmp_path, monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", ("mms.decider_pplx:Pplx",))
    folder = model_folder(tmp_path / "models", "pplx-decider-flash-v1.2")
    assert decider.about_of(str(folder)) is ABOUT
    with pytest.raises(ValueError, match="not prepared.*mms model prepare pplx-decider-flash-v1.2"):
        decider.load_decider(str(folder), home)
    prepared_env(tmp_path / "models" / "envs", "pplx", Pplx.requirements)
    assert decider.load_decider(str(folder), home).about is ABOUT
    with pytest.raises(ValueError, match="no decider reads the model.*joint_config.json"):
        decider.load_decider(str(model_folder(tmp_path / "models", "d3-edge")), home)


def test_pplxs_facts_are_the_plans_and_its_costs_are_measured():
    assert Pplx.abouts == (ABOUT,)
    assert ABOUT.name == "pplx-decider-flash-v1.2" and ABOUT.reads == {"video", "image"}
    assert ABOUT.types == ("noul", "choice", "score") and ABOUT.question_keys == {"criteria"}
    assert ABOUT.extracts_from == frozenset() and ABOUT.speech is False
    assert (ABOUT.max_px, ABOUT.questions_per_read, ABOUT.hit, ABOUT.notes) == (1_638_400, 1, 0.8, "pplx.md")
    assert (ABOUT.picture_s, ABOUT.char_s) == (((230_400, 0.276), (1_638_400, 0.602)), 0.116 / 2000)
    assert ABOUT.measured == {"picture_s", "char_s"}, "its costs were timed here; its hit is assumed"
    assert (ABOUT.repo, ABOUT.repo_gb) == ("perplexity-ai/pplx-decider-flash-v1.2", 9.1), "where `mms model use` downloads it from"
    assert Pplx.family == "pplx" and Pplx.served is False and Pplx.default_url is None and Pplx.start == ""
    assert Pplx.torch_backend == "cu130"
    assert Pplx.requirements == ("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors==0.8.0",
                                 "pillow==12.3.0", "huggingface-hub==1.31.0"), "the card's six exact pins"


# -- the env and the weights ------------------------------------------------------------------------------------------

def test_a_missing_or_stale_env_marker_is_refused_naming_the_prepare_command(tmp_path):
    folder = model_folder(tmp_path / "models", "pplx-decider-flash-v1.2")
    with pytest.raises(ValueError, match=r"pplx-decider-flash-v1.2's env \(.*envs.pplx\) is not prepared.*not there"
                                         r".*mms model prepare pplx-decider-flash-v1.2"):
        Pplx(folder)
    prepared_env(tmp_path / "models" / "envs", "pplx", ("torch==2.13.0",))
    with pytest.raises(ValueError, match=r"is not prepared.*other requirements.*mms model prepare pplx-decider"):
        Pplx(folder)


def test_the_env_is_where_envs_says_the_worker_runs_in_it_and_a_url_is_refused(tmp_path):
    folder = model_folder(tmp_path / "models", "pplx-decider-flash-v1.2")
    moved = prepared_env(tmp_path / "elsewhere", "pplx", Pplx.requirements)
    made = Pplx(folder, envs=str(tmp_path / "elsewhere"))
    assert made.worker.python == modelenv.python(moved) and made.worker.family_file == WORKER / "pplx.py"
    assert made.worker.name == ABOUT.name and made.worker.folder == folder and made.worker.pid is None
    with pytest.raises(ValueError, match="not served.*url"):
        Pplx(folder, url="http://127.0.0.1:8741", envs=str(tmp_path / "elsewhere"))


def test_weights_with_another_picture_cap_than_the_facts_say_are_refused(tmp_path):
    folder = model_folder(tmp_path / "models", "pplx-decider-flash-v1.2")
    prepared_env(tmp_path / "models" / "envs", "pplx", Pplx.requirements)
    config = folder / "decision_config.json"
    recorded = json.loads(config.read_text(encoding="utf-8"))
    config.write_text(json.dumps({**recorded, "max_pixels": 262_144}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"decision_config.json says max_pixels is 262144, but decider_pplx.ABOUT says "
                                         r"max_px is 1638400"):
        Pplx(folder)
    config.write_text(json.dumps({k: v for k, v in recorded.items() if k != "max_pixels"}), encoding="utf-8")
    with pytest.raises(ValueError, match="max_pixels is None"):
        Pplx(folder)


def test_load_checks_the_loaded_model_reads_pictures_at_the_cap_the_facts_say(tmp_path):
    made = backend(tmp_path)
    made.worker = Worker(facts={"max_pixels": 1_638_400})
    made.load()
    made.worker = Worker(facts={"max_pixels": 262_144})
    with pytest.raises(RuntimeError, match="max_pixels 262144, but ABOUT says 1638400"):
        made.load()
    assert made.worker.unloaded == 1
    made.unload()
    assert made.worker.unloaded == 2


# -- the questions and answers ----------------------------------------------------------------------------------------

def test_pplx_is_asked_the_type_the_instructions_and_the_criteria_and_nothing_else(tmp_path):
    made = backend(tmp_path)
    native = made.to_model(made.about, QUESTIONS)
    assert native == QUESTIONS and native is not QUESTIONS
    for extra in ({"multiple": True}, {"range": [0, 1]}, {"unit": "s"}, {"step": 1}, {"hit": 0.8}):
        with pytest.raises(ValueError, match=rf"question 'q' has {next(iter(extra))}, which {ABOUT.name} does not take for a "
                                             "noul question; it takes criteria, instructions, type"):
            made.to_model(made.about, {"q": {"type": "noul", "instructions": "x", **extra}})
    for kind in ("text", "number", "date", "box", "boxes"):
        with pytest.raises(ValueError, match=rf"answers noul, choice, score, not '{kind}'"):
            made.to_model(made.about, {"q": {"type": kind, "instructions": "x"}})


def test_a_question_pplx_cannot_take_is_refused_before_its_worker_is_asked_and_decide_translates(tmp_path):
    made = backend(tmp_path)
    made.worker = Worker({"seen": NOUL, "which": CHOICE, "amount": SCORE})
    with pytest.raises(ValueError, match="does not take"):
        made.decide(None, {"q": {"type": "noul", "instructions": "x", "unit": "s"}})
    assert made.worker.asked == []
    picture = Image.new("RGB", (4, 2))
    got = made.decide(None, QUESTIONS, [picture])
    assert made.worker.asked == [(None, QUESTIONS, [picture])]
    assert got["seen"] == NOUL and "confidence" not in got["which"] and got["amount"]["score"] == 2.0


@pytest.mark.parametrize("question,words", [
    # what the protocol drops without a word (noul reads criteria.get("true") and .get("false")) is refused
    ({"type": "noul", "instructions": "x", "criteria": {"true": "lit", "dark": "off"}}, "criteria describes true and false only"),
    ({"type": "noul", "instructions": "x", "criteria": ["lit", "dark"]}, "criteria describes true and false only"),
    # Encoder.validate: a choice of 1 to 255 options, a score of at least two levels
    ({"type": "choice", "instructions": "x", "criteria": {str(n): None for n in range(256)}}, "1 to 255 options"),
    ({"type": "choice", "instructions": "x"}, "a choice has criteria"),
    ({"type": "score", "instructions": "x", "criteria": ["only"]}, "at least 2 levels"),
])
def test_a_question_pplx_refuses_is_refused_by_check_naming_it_and_before_its_worker_is_asked(tmp_path, question, words):
    made = backend(tmp_path)
    made.worker = Worker()
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        Pplx.check(made.about, {"ok": QUESTIONS["seen"], "q": question})
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        made.decide(None, {"q": question})
    assert made.worker.asked == []
    assert Pplx.check(made.about, QUESTIONS) is None
    same = {"q": {"type": "score", "instructions": "x", "criteria": ["same", "same"]}}
    assert Pplx.check(made.about, same) is None, "it does not need a score's levels apart"
    assert Pplx.install == ""


def test_an_error_entry_of_pplx_is_a_valueerror_naming_it_like_the_other_models(tmp_path):
    made = backend(tmp_path)
    answers = {"seen": {"type": "noul", "error": "invalid_question", "message": "too long"}}
    with pytest.raises(ValueError, match=rf"{ABOUT.name}: question 'seen': invalid_question: too long"):
        made.from_model(answers, {"seen": QUESTIONS["seen"]})
    made.worker = Worker(answers)
    with pytest.raises(ValueError, match="invalid_question: too long"):
        made.decide(None, {"seen": QUESTIONS["seen"]})


def test_pplxs_answers_read_and_judge_in_mms_terms(ask, tmp_path):
    made = backend(tmp_path)
    got = made.from_model({"seen": NOUL, "which": CHOICE, "amount": SCORE}, QUESTIONS)
    assert "confidence" not in got["which"] and got["which"]["probabilities"] == CHOICE["probabilities"]
    assert got["amount"]["score"] == 2.0 and SCORE["score"] == 1.6, "the median level, not the expected one"
    assert verdict(ask, QUESTIONS["seen"], True, ABOUT, got["seen"])[:2] == (True, 0.91)
    assert verdict(ask, QUESTIONS["seen"], 0.95, ABOUT, got["seen"])[:2] == (False, 0.91)
    assert verdict(ask, QUESTIONS["which"], "cup", ABOUT, got["which"]) == (True, 0.7, "cup"), \
        "p is the pick's probability, not the confidence of 0.55"
    assert verdict(ask, QUESTIONS["which"], ["mug", "none"], ABOUT, got["which"]) == (False, 0.2, "cup")
    assert verdict(ask, QUESTIONS["amount"], 2, ABOUT, got["amount"]) == (True, 1.0, "level 2")
    odd = made.from_model({"amount": {**SCORE, "probabilities": {"low": 0.4, "high": 0.6}}}, {"amount": QUESTIONS["amount"]})
    assert odd["amount"]["score"] is None
    with pytest.raises(ask.Unreadable, match="its score is None"):
        verdict(ask, QUESTIONS["amount"], 2, ABOUT, odd["amount"])
    with pytest.raises(ValueError, match="gave no answer to question 'seen'"):
        made.from_model({}, {"seen": QUESTIONS["seen"]})


# -- worker/pplx.py ---------------------------------------------------------------------------------------------------

PACKAGE = '''
CALLS, LOADS = [], []


class RequestError(ValueError):
    pass


class Decider:
    max_pixels = 1_638_400
    temperature = 1.554

    @classmethod
    def from_pretrained(cls, path, *, device="cuda", fast=False):
        LOADS.append({"path": path, "device": device, "fast": fast})
        return cls()

    def decide(self, request):
        CALLS.append(request)
        if request["state"] == "refuse":
            raise RequestError("Question 'q': type must be 'choice', 'noul' or 'score'")
        return {"answers": {name: {"type": "noul", "noul": 0.9} for name in request["questions"]},
                "usage": {"input_tokens": 1, "output_tokens": 0, "sequences": 1}}
'''


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """A model folder whose `source/src/pplx_decider` is the fake package; sys.path and sys.modules are put back."""
    root = tmp_path / "pplx"
    (root / "source" / "src" / "pplx_decider").mkdir(parents=True)
    (root / "source" / "src" / "pplx_decider" / "__init__.py").write_text(PACKAGE, encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert "pplx_decider" not in sys.modules
    yield root
    sys.modules.pop("pplx_decider", None)


def test_the_pplx_worker_imports_the_package_from_the_folders_source_and_loads_the_reference_path_on_the_gpu(folder):
    worker = load_worker("pplx")
    model, facts = worker.load(str(folder), {})
    package = sys.modules["pplx_decider"]
    assert Path(package.__file__) == folder / "source" / "src" / "pplx_decider" / "__init__.py"
    assert sys.path[0] == str(folder / "source" / "src")
    assert package.LOADS == [{"path": folder, "device": "cuda", "fast": False}], "fast is off, as the card's exact mode"
    assert facts == {"max_pixels": 1_638_400, "temperature": 1.554}
    text, picture = package.CALLS
    assert text == {"state": "warm-up", "questions": worker.WARM_TEXT}
    assert picture["state"] == "" and len(picture["images"]) == 1 and picture["images"][0].size == (640, 360)


def test_the_pplx_worker_makes_one_call_for_each_question_with_pictures_as_pil_and_a_none_state_as_empty(folder):
    worker = load_worker("pplx")
    model, _ = worker.load(str(folder), {})
    package = sys.modules["pplx_decider"]
    package.CALLS.clear()
    pictures = [Image.new("RGB", (8, 4), "red")]
    questions = {"a": {"type": "noul", "instructions": "x"},
                 "b": {"type": "choice", "instructions": "y", "criteria": {"p": None, "q": None}}}
    answers = worker.decide(model, None, questions, pictures)
    assert answers == {"a": {"type": "noul", "noul": 0.9}, "b": {"type": "noul", "noul": 0.9}}
    assert package.CALLS == [{"state": "", "questions": {"a": questions["a"]}, "images": pictures},
                             {"state": "", "questions": {"b": questions["b"]}, "images": pictures}]
    assert all(isinstance(image, Image.Image) for call in package.CALLS for image in call["images"])
    package.CALLS.clear()
    worker.decide(model, "a note", {"a": questions["a"]}, [])
    assert package.CALLS == [{"state": "a note", "questions": {"a": questions["a"]}}], "no pictures, no images key"
    assert worker.decide(model, None, {}, []) == {}


def test_a_request_the_model_refuses_is_a_valueerror_and_a_sound_is_not_sent(folder):
    worker = load_worker("pplx")
    model, _ = worker.load(str(folder), {})
    with pytest.raises(ValueError, match="type must be"):
        worker.decide(model, "refuse", {"q": {"type": "noul", "instructions": "x"}}, [])
    before = len(sys.modules["pplx_decider"].CALLS)
    with pytest.raises(ValueError, match="reads pictures, not audio"):
        worker.decide(model, None, {"q": {"type": "noul", "instructions": "x"}},
                      [{"type": "audio", "audio": [0.0], "sampling_rate": 16000}])
    assert len(sys.modules["pplx_decider"].CALLS) == before
