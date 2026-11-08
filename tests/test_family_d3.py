"""d3's backend and its worker half (TESTBED.md, "No GPU"; DESIGN.md §13.3): how a folder is claimed, the facts, the env check,
and the two translations (mms's questions to d3's, d3's answers to mms's), from canned answers in the shapes of
`d3_runtime.product_answer`. `worker/d3.py` runs in this process against a fake `transformers`. No model, no GPU."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest
from conftest import MODEL_FOLDERS, ROOT, model_folder, prepared_env
from PIL import Image

from mms import decider, jobs, modelenv
from mms.decider import About
from mms.decider_d3 import EDGE, FLASH, MINI, D3

VARIANTS = {"d3-edge": EDGE, "d3-mini": MINI, "d3-flash": FLASH}
WORKER = ROOT / "src" / "mms" / "worker"


@pytest.fixture
def ask(home):
    return jobs.load_script(home, "ask")


def verdict(ask, question: dict, hit, about: About, answer) -> tuple[bool, float, str]:
    """What `ask` makes of one answer to one question: whether it hits, its p, and what it said (the answer is read by
    `ask.read` first, so one that is not in its type's shape raises `Unreadable`)."""
    questions, hits = ask.build({"questions": {"q": {"instructions": "x", **question, "hit": hit}}})
    ask.read("q", questions["q"], answer)
    hit, p, label, _ = ask.judge(questions, ask.rules_of(questions, hits, about), {"q": answer})
    return hit, p, re.fullmatch(r"q: (.*) \(p \d\.\d\d\)", label).group(1)


def backend(tmp_path: Path, name: str = "d3-flash") -> D3:
    folder = model_folder(tmp_path / "models", name)
    prepared_env(tmp_path / "models" / "envs", "d3", D3.requirements)
    return D3(folder)


def load_worker(name: str):
    spec = importlib.util.spec_from_file_location(f"mms_worker_{name}_test", WORKER / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# d3's answers as d3_runtime.product_answer gives them
NOUL = {"type": "noul", "noul": 0.93}
CHOICE = {"type": "choice", "choice": "cup", "probabilities": {"mug": 0.1, "cup": 0.7, "none": 0.2}, "confidence": 0.43}
SCORE = {"type": "score", "score": 1.6, "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6}, "confidence": 0.1,
         "legend": {"0": "no", "1": "some", "2": "all"}}
QUESTIONS = {"seen": {"type": "noul", "instructions": "Is a mug in view?"},
             "which": {"type": "choice", "instructions": "Which?", "criteria": {"mug": None, "cup": "a cup", "none": None}},
             "amount": {"type": "score", "instructions": "How much?", "criteria": ["no", "some", "all"]}}


# -- claims and facts -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODEL_FOLDERS)
def test_a_d3_folder_is_claimed_by_its_manifest_and_no_other_folder_is(tmp_path, name):
    got = D3.claims(model_folder(tmp_path, name))
    assert got is VARIANTS.get(name), f"{name}: {got}"


def test_a_folder_d3_cannot_name_is_not_claimed(tmp_path):
    folder = model_folder(tmp_path, "d3-mini")
    manifest = json.loads((folder / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    for changed in ({"model_name": "d3-huge"}, {"model_name": None}, {"model_name": ["d3-mini"]}):
        (folder / "MODEL_MANIFEST.json").write_text(json.dumps({**manifest, **changed}), encoding="utf-8")
        assert D3.claims(folder) is None, changed
    (folder / "MODEL_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert D3.claims(folder) is MINI
    decision = json.loads((folder / "decision_config.json").read_text(encoding="utf-8"))
    (folder / "decision_config.json").write_text(json.dumps({**decision, "format_id": "d3-code-readout-v2"}), encoding="utf-8")
    assert D3.claims(folder) is None, "another format is not read"
    (folder / "decision_config.json").write_text("not json", encoding="utf-8")
    assert D3.claims(folder) is None
    (folder / "decision_config.json").unlink()
    assert D3.claims(folder) is None
    (folder / "decision_config.json").write_text(json.dumps(decision), encoding="utf-8")
    (folder / "MODEL_MANIFEST.json").unlink()
    assert D3.claims(folder) is None, "the manifest names the variant: without it the folder is not d3's"
    assert D3.claims(folder / "missing") is None


def test_the_loader_finds_d3_by_what_claims_it_and_says_what_it_reads_when_nothing_does(home, tmp_path, monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", ("mms.decider_d3:D3",))
    folder = model_folder(tmp_path / "models", "d3-edge")
    assert decider.about_of(str(folder)) is EDGE
    with pytest.raises(ValueError, match="not prepared"):
        decider.load_decider(str(folder), home)
    prepared_env(tmp_path / "models" / "envs", "d3", D3.requirements)
    assert decider.load_decider(str(folder), home).about is EDGE
    other = model_folder(tmp_path / "models", "d1-2B")
    with pytest.raises(ValueError, match="no decider reads the model.*decision_config.json.*d3-edge, d3-mini or d3-flash"):
        decider.load_decider(str(other), home)


def test_d3s_facts_are_the_plans_and_its_costs_are_measured():
    assert [a.name for a in D3.abouts] == ["d3-edge", "d3-mini", "d3-flash"]
    for about in D3.abouts:
        assert about.reads == {"video", "image"}, "d3 cannot hear"
        assert about.types == ("noul", "choice", "score") and about.question_keys == {"criteria"}
        assert about.extracts_from == frozenset() and about.speech is False and about.files == ()
        assert (about.max_px, about.questions_per_read, about.hit, about.notes) == (1_638_400, 1, 0.8, "d3.md")
        assert about.repo == f"vllm-sr/{about.name}", "where `mms model use` downloads it from"
    assert [a.repo_gb for a in D3.abouts] == [1.2, 9.1, 16.8]
    assert all(a.measured == {"picture_s", "char_s"} for a in D3.abouts), "the costs were timed here; every hit is assumed"
    assert [(a.picture_s, a.char_s) for a in D3.abouts] == [
        (((230_400, 0.041), (1_638_400, 0.153)), 0.042 / 2000), (((230_400, 0.163), (1_638_400, 0.573)), 0.135 / 2000),
        (((230_400, 0.184), (1_638_400, 1.005)), 0.225 / 2000)]
    assert D3.family == "d3" and D3.served is False and D3.default_url is None and D3.start == ""
    assert D3.torch_backend == "cu128" and D3.requirements[0] == "torch==2.11.0"
    assert "transformers==5.17.0" in D3.requirements
    assert not any("flash-linear-attention" in pin for pin in D3.requirements), "it needs triton, which Windows lacks"


# -- the env ----------------------------------------------------------------------------------------------------------

def test_a_missing_or_stale_env_marker_is_refused_naming_the_prepare_command(tmp_path):
    folder = model_folder(tmp_path / "models", "d3-flash")
    with pytest.raises(ValueError, match=r"d3-flash's env \(.*envs.d3\) is not prepared.*not there.*mms model prepare d3-flash"):
        D3(folder)
    prepared_env(tmp_path / "models" / "envs", "d3", ("torch==2.10.0",))
    with pytest.raises(ValueError, match=r"is not prepared.*other requirements.*mms model prepare d3-flash"):
        D3(folder)
    marker = tmp_path / "models" / "envs" / "d3" / "mms-env.json"
    marker.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cannot be read.*mms model prepare d3-flash"):
        D3(folder)


def test_the_env_is_beside_the_model_or_where_envs_says_and_the_worker_runs_in_it(tmp_path):
    folder = model_folder(tmp_path / "models", "d3-mini")
    beside = prepared_env(tmp_path / "models" / "envs", "d3", D3.requirements)
    made = D3(folder)
    assert made.worker.python == modelenv.python(beside) and made.worker.family_file == WORKER / "d3.py"
    assert made.worker.name == "d3-mini" and made.worker.folder == folder and made.worker.pid is None, "nothing is started"
    moved = prepared_env(tmp_path / "elsewhere", "d3", D3.requirements)
    assert D3(folder, envs=str(tmp_path / "elsewhere")).worker.python == modelenv.python(moved)
    with pytest.raises(ValueError, match="not prepared"):
        D3(folder, envs=str(tmp_path / "nowhere"))
    with pytest.raises(ValueError, match="not served.*url"):
        D3(folder, url="http://127.0.0.1:8741")
    with pytest.raises(ValueError, match="is not a d3 model"):
        D3(model_folder(tmp_path / "models", "unclaimed"))


# -- the questions ----------------------------------------------------------------------------------------------------

def test_d3_is_asked_the_type_the_instructions_and_the_criteria_and_nothing_else(tmp_path):
    made = backend(tmp_path)
    native = made.to_model(made.about, QUESTIONS)
    assert native == QUESTIONS and native is not QUESTIONS and native["which"] is not QUESTIONS["which"]
    noul = {"seen": {"type": "noul", "instructions": "x", "criteria": {"true": "a mug", "false": "no mug"}}}
    assert made.to_model(made.about, noul) == noul


@pytest.mark.parametrize("extra", [{"multiple": True}, {"range": [0, 10]}, {"unit": "beeps"}, {"step": 1}, {"hit": 0.8},
                                   {"colour": "red"}])
def test_a_key_d3_does_not_take_is_refused_by_name(tmp_path, extra):
    made = backend(tmp_path)
    question = {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}, **extra}}
    with pytest.raises(ValueError, match=rf"question 'q' has {next(iter(extra))}, which d3-flash does not take for a choice question; "
                                         "it takes criteria, instructions, type"):
        made.to_model(made.about, question)


@pytest.mark.parametrize("kind", ["text", "number", "date", "box", "boxes"])
def test_a_type_d3_does_not_answer_is_refused_naming_the_types_it_does(tmp_path, kind):
    made = backend(tmp_path)
    with pytest.raises(ValueError, match=rf"d3-flash answers noul, choice, score, not '{kind}'"):
        made.to_model(made.about, {"q": {"type": kind, "instructions": "x"}})


class Worker:
    """A worker that records what it is asked and answers `answers`."""

    def __init__(self, answers=None, facts=None):
        self.asked, self.answers, self.facts, self.unloaded = [], answers, facts or {}, 0

    def load(self):
        return self.facts

    def decide(self, state, questions, media):
        self.asked.append((state, questions, media))
        return self.answers

    def unload(self):
        self.unloaded += 1


def test_a_question_d3_cannot_take_is_refused_before_its_worker_is_asked(tmp_path):
    made = backend(tmp_path)
    made.worker = Worker()
    with pytest.raises(ValueError, match="does not take"):
        made.decide(None, {"q": {"type": "noul", "instructions": "x", "multiple": True}})
    assert made.worker.asked == []


ODD_SCORE = ["same", "same", "other"]
REFUSED = [  # d3_runtime.normalize_question's rules that mms's own question format lets through
    ({"type": "noul", "instructions": "x", "criteria": {"true": "lit", "dark": "off"}}, "criteria describes true and false only"),
    ({"type": "noul", "instructions": "x", "criteria": ["lit", "dark"]}, "criteria describes true and false only"),
    ({"type": "score", "instructions": "x", "criteria": ODD_SCORE}, "distinct and not empty"),
    ({"type": "score", "instructions": "x", "criteria": ["low", ""]}, "distinct and not empty"),
    ({"type": "score", "instructions": "x", "criteria": [str(n) for n in range(11)]}, "2 to 10 levels"),
    ({"type": "score", "instructions": "x", "criteria": ["only"]}, "at least 2 levels"),
    ({"type": "choice", "instructions": "x", "criteria": {"": "nothing", "a": None}}, "empty option name"),
    ({"type": "choice", "instructions": "x", "criteria": {str(n): None for n in range(256)}}, "1 to 255 options"),
    ({"type": "choice", "instructions": "x"}, "a choice has criteria"),
]


@pytest.mark.parametrize("question,words", REFUSED)
def test_a_question_d3_refuses_is_refused_by_check_naming_it_and_before_its_worker_is_asked(tmp_path, question, words):
    made = backend(tmp_path)
    made.worker = Worker()
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        D3.check(made.about, {"ok": QUESTIONS["seen"], "q": question})
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        made.decide(None, {"q": question})
    assert made.worker.asked == [], "d3 would answer each window with invalid_question: nothing is asked"


def test_check_passes_every_question_d3_takes_and_a_yes_no_with_true_and_false_described(tmp_path):
    made = backend(tmp_path)
    assert D3.check(made.about, QUESTIONS) is None
    assert D3.check(made.about, {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "lit"}}}) is None
    assert D3.check(made.about, {"q": {"type": "score", "instructions": "x", "criteria": [str(n) for n in range(10)]}}) is None
    with pytest.raises(ValueError, match="does not take for a noul question"):
        D3.check(made.about, {"q": {"type": "noul", "instructions": "x", "multiple": True}})


def test_decide_hands_d3_its_questions_state_and_pictures_and_gives_back_mms_answers(tmp_path):
    made = backend(tmp_path)
    picture = Image.new("RGB", (4, 2))
    made.worker = Worker({"seen": NOUL, "which": CHOICE, "amount": SCORE})
    got = made.decide(None, QUESTIONS, [picture])
    assert made.worker.asked == [(None, QUESTIONS, [picture])], "state None stays None"
    assert got["seen"] == NOUL and got["which"] == {key: v for key, v in CHOICE.items() if key != "confidence"}
    assert got["amount"]["score"] == 2.0 and got["amount"]["probabilities"] == SCORE["probabilities"]
    made.decide("text", {"seen": QUESTIONS["seen"]})
    assert made.worker.asked[-1] == ("text", {"seen": QUESTIONS["seen"]}, [])


def test_load_checks_the_weights_are_the_variant_the_folder_says(tmp_path):
    made = backend(tmp_path)
    made.worker = Worker(facts={"model_name": "d3-flash"})
    made.load()
    assert made.worker.unloaded == 0
    made.worker = Worker(facts={"model_name": "d3-mini"})
    with pytest.raises(RuntimeError, match="loaded as 'd3-mini', but it is d3-flash"):
        made.load()
    assert made.worker.unloaded == 1, "a worker that holds the wrong model is ended"
    made.unload()
    assert made.worker.unloaded == 2


# -- the answers ------------------------------------------------------------------------------------------------------

def test_d3s_answers_read_and_judge_in_mms_terms(ask, tmp_path):
    made = backend(tmp_path)
    got = made.from_model({"seen": NOUL, "which": CHOICE, "amount": SCORE}, QUESTIONS)
    assert "confidence" not in got["which"] and got["which"]["probabilities"] == CHOICE["probabilities"]
    assert got["amount"]["score"] == 2.0, "the median level of 0.2, 0.2, 0.6; d3's own 1.6 is the expected level"
    assert SCORE["score"] == 1.6 and "score" in SCORE and SCORE["confidence"] == 0.1, "the answer d3 gave is left as it was"
    about = made.about
    assert verdict(ask, QUESTIONS["seen"], True, about, got["seen"])[:2] == (True, 0.93)
    assert verdict(ask, QUESTIONS["seen"], 0.95, about, got["seen"])[:2] == (False, 0.93)
    hit, p, said = verdict(ask, QUESTIONS["which"], "cup", about, got["which"])
    assert (hit, p, said) == (True, 0.7, "cup"), "p is the pick's probability, not d3's entropy confidence of 0.43"
    hit, p, said = verdict(ask, QUESTIONS["which"], "mug", about, got["which"])
    assert (hit, p, said) == (False, 0.1, "cup")
    hit, p, said = verdict(ask, QUESTIONS["amount"], 2, about, got["amount"])
    assert (hit, p, said) == (True, 1.0, "level 2"), "level 2 counts, which the expected level 1.6 would not"
    assert verdict(ask, QUESTIONS["amount"], 3, about, got["amount"])[0] is False


def test_a_score_without_the_levels_d3_names_cannot_be_read_and_is_never_passed_off_as_the_expected_level(ask, tmp_path):
    made = backend(tmp_path)
    odd = {**SCORE, "probabilities": {"a": 0.5, "b": 0.5}}
    got = made.from_model({"amount": odd}, {"amount": QUESTIONS["amount"]})
    assert got["amount"]["score"] is None
    with pytest.raises(ask.Unreadable, match="its score is None"):
        verdict(ask, QUESTIONS["amount"], 2, made.about, got["amount"])


def test_d3_has_nothing_to_install_by_hand():
    assert D3.install == ""


def test_an_error_d3_gives_for_a_question_is_a_valueerror_naming_it(tmp_path):
    made = backend(tmp_path)
    for error in ("invalid_question", "max_length_exceeded", "invalid_model_output"):
        answers = {"seen": {"type": "noul", "error": error, "message": "too long"}}
        with pytest.raises(ValueError, match=rf"d3-flash: question 'seen': {error}: too long"):
            made.from_model(answers, {"seen": QUESTIONS["seen"]})
    with pytest.raises(ValueError, match="gave no answer to question 'seen'"):
        made.from_model({}, {"seen": QUESTIONS["seen"]})
    with pytest.raises(ValueError, match="gave no answer to question 'seen'"):
        made.from_model({"seen": "yes"}, {"seen": QUESTIONS["seen"]})


# -- worker/d3.py -----------------------------------------------------------------------------------------------------

class FakeD3Model:
    """What `AutoModel.from_pretrained(..., trust_remote_code=True)` gives: `system_one` takes keywords only."""

    model_name = "d3-flash"

    def __init__(self, error=None):
        self.calls, self.error = [], error

    def system_one(self, *, state, questions, images=None, videos=None):
        self.calls.append({"state": state, "questions": questions, "images": images, "videos": videos})
        answers = {name: {"type": "noul", "noul": 0.5} for name in questions}
        if self.error:
            answers = {name: {"type": "noul", "error": self.error, "message": "bad"} for name in questions}
        return {"model": "d3-flash", "answers": answers, "usage": {}}


@pytest.fixture
def transformers(monkeypatch):
    """A fake `transformers` whose `AutoModel.from_pretrained` records its call and gives a `FakeD3Model`."""
    made = {}

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            made["call"] = (path, kwargs)
            made["model"] = FakeD3Model(made.get("error"))
            return made["model"]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))
    return made


def test_the_d3_worker_loads_the_remote_code_on_the_gpu_and_warms_it_with_a_text_and_a_picture(transformers, tmp_path):
    worker = load_worker("d3")
    model, facts = worker.load(str(tmp_path), {})
    # an index: d3 gives the device to torch.cuda.set_device, which raises on a bare "cuda" in torch 2.11
    assert transformers["call"] == (str(tmp_path), {"trust_remote_code": True, "device": "cuda:0"})
    assert facts == {"model_name": "d3-flash"} and model is transformers["model"]
    text, picture = model.calls
    assert text["state"] == "warm-up" and text["images"] is None and text["videos"] is None
    assert text["questions"] == worker.WARM_TEXT
    assert picture["state"] is None and len(picture["images"]) == 1 and picture["images"][0].size == (640, 360)


def test_a_warm_up_d3_cannot_answer_fails_the_load_with_its_error(transformers, tmp_path):
    transformers["error"] = "invalid_question"
    with pytest.raises(RuntimeError, match="warm-up 'seen': invalid_question: bad"):
        load_worker("d3").load(str(tmp_path), {})


def test_the_d3_worker_passes_pictures_as_pil_and_a_none_state_as_none_and_gives_back_the_native_answers():
    worker, model = load_worker("d3"), FakeD3Model()
    pictures = [Image.new("RGB", (8, 4), "red"), Image.new("RGB", (2, 2), "blue")]
    questions = {"a": {"type": "noul", "instructions": "x"}, "b": {"type": "noul", "instructions": "y"}}
    answers = worker.decide(model, None, questions, pictures)
    assert answers == {"a": {"type": "noul", "noul": 0.5}, "b": {"type": "noul", "noul": 0.5}}
    (call,) = model.calls
    assert call["state"] is None and call["questions"] == questions and call["videos"] is None
    assert call["images"] == pictures and all(isinstance(image, Image.Image) for image in call["images"])
    worker.decide(model, "a note", questions, [])
    assert model.calls[-1]["state"] == "a note" and model.calls[-1]["images"] is None
    with pytest.raises(ValueError, match="d3 reads pictures, not audio"):
        worker.decide(model, None, questions, [{"type": "audio", "audio": [0.0], "sampling_rate": 16000}])
    assert len(model.calls) == 2


def test_d3s_answers_with_an_error_come_back_unchanged_from_the_worker_for_the_backend_to_refuse():
    worker, model = load_worker("d3"), FakeD3Model(error="max_length_exceeded")
    answers = worker.decide(model, "long", {"q": {"type": "noul", "instructions": "x"}}, [])
    assert answers == {"q": {"type": "noul", "error": "max_length_exceeded", "message": "bad"}}
