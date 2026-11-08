"""d1-3B as a decider family (TESTBED.md, "No GPU"; DESIGN.md §13.3): `decider_d1_3b`'s claim, facts and translations, and
`worker/d1_3b.py`, the half that holds the model. The model, torch and transformers are fakes, so nothing here needs a GPU
or weights; the real d1-3B is in `test_gpu.py`."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import D1_3B_CONFIG, MODEL_FOLDERS, ROOT, model_folder
from PIL import Image
from test_family_d3 import Worker, ask, load_worker, verdict  # noqa: F401  helpers, and the `ask` fixture

from mms import decider, decider_d1, decider_d1_3b, modelenv, transport
from mms.decider_d1_3b import ABOUT, D1_3B

WORKER_D1_3B = ROOT / "src" / "mms" / "worker" / "d1_3b.py"
CAP = 1_048_576  # runner.VISION_MAX_PIXELS in the model's code

# d1-3B's answers as api.answer gives them
NOUL = {"type": "noul", "noul": 0.93}
CHOICE = {"type": "choice", "choice": "cup", "confidence": 0.7, "probabilities": {"mug": 0.1, "cup": 0.7, "none": 0.2}}
SCORE = {"type": "score", "score": 1.6, "confidence": 0.6, "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6},
         "legend": {"0": "no", "1": "some", "2": "all"}}
QUESTIONS = {"seen": {"type": "noul", "instructions": "Is a mug in view?"},
             "which": {"type": "choice", "instructions": "Which?", "criteria": {"mug": None, "cup": "a cup", "none": None}},
             "amount": {"type": "score", "instructions": "How much?", "criteria": ["no", "some", "all"]}}


def backend(tmp_path: Path) -> D1_3B:
    return D1_3B(model_folder(tmp_path / "models", "d1-3B"))


# -- claims and facts --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODEL_FOLDERS)
def test_only_a_d1_3b_folder_is_claimed_by_d1_3b(tmp_path, name):
    got = D1_3B.claims(model_folder(tmp_path, name))
    assert got is (ABOUT if name == "d1-3B" else None), f"{name}: {got}"


def test_a_d1_3b_folder_is_found_by_the_loader_and_is_not_d1s(tmp_path):
    folder = model_folder(tmp_path, "d1-3B")
    assert decider_d1.D1.claims(folder) is None, "it has none of d1's keys"
    assert decider.claim(folder) == (D1_3B, ABOUT) and decider.claimed(folder) == (D1_3B, ABOUT)
    assert decider.about_of(str(folder)) is ABOUT and ABOUT in decider.known_abouts()
    made = decider.load_decider(str(folder))
    assert isinstance(made, D1_3B) and made.about is ABOUT and made.folder == folder and made.worker.pid is None
    for name in ("d1-Omni-2B", "d1-2B"):
        assert decider.about_of(str(model_folder(tmp_path, name))) is not ABOUT, f"{name} is not d1-3B"


def test_a_folder_that_is_not_the_3b_is_not_claimed_and_the_loader_says_what_d1_3b_reads(tmp_path):
    text = D1_3B_CONFIG["text_config"]
    cases = {
        "another-type": {**D1_3B_CONFIG, "model_type": "lfm3_vl"}, "no-type": {k: v for k, v in D1_3B_CONFIG.items() if k != "model_type"},
        "type-not-text": {**D1_3B_CONFIG, "model_type": 7},
        "no-auto-map": {k: v for k, v in D1_3B_CONFIG.items() if k != "auto_map"},
        "other-auto-map": {**D1_3B_CONFIG, "auto_map": {"AutoModel": "modeling_d3.D3Model"}},
        "auto-map-without-automodel": {**D1_3B_CONFIG, "auto_map": {"AutoConfig": "modeling_d1.D1Model"}},
        "auto-map-a-list": {**D1_3B_CONFIG, "auto_map": ["modeling_d1.D1Model"]},
        "no-text-config": {k: v for k, v in D1_3B_CONFIG.items() if k != "text_config"},
        "text-config-a-list": {**D1_3B_CONFIG, "text_config": [2048, 30]},
        "another-width": {**D1_3B_CONFIG, "text_config": {**text, "hidden_size": 1536}},
        "another-depth": {**D1_3B_CONFIG, "text_config": {**text, "num_hidden_layers": 16}},
        "text-sizes": {**D1_3B_CONFIG, "text_config": {"model_type": "lfm2"}},
        "width-a-string": {**D1_3B_CONFIG, "text_config": {**text, "hidden_size": "2048"}},
        "width-a-float": {**D1_3B_CONFIG, "text_config": {**text, "hidden_size": 2048.0}},
        "depth-a-bool": {**D1_3B_CONFIG, "text_config": {**text, "num_hidden_layers": True}},
        "depth-a-list": {**D1_3B_CONFIG, "text_config": {**text, "num_hidden_layers": [30]}},
        "depth-none": {**D1_3B_CONFIG, "text_config": {**text, "num_hidden_layers": None}},
        "not-json": "{", "a-list": "[]", "empty": "{}",
    }
    for name, config in cases.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(config if isinstance(config, str) else json.dumps(config), encoding="utf-8")
        assert D1_3B.claims(folder) is None and decider.about_of(str(folder)) is None, name
        with pytest.raises(ValueError, match="no decider reads the model") as raised:
            decider.load_decider(str(folder))
        assert "d1-3B" in str(raised.value) and "lfm2_vl" in str(raised.value) and "modeling_d1.D1Model" in str(raised.value), \
            f"{name}: the error names what is read: {raised.value}"
        with pytest.raises(ValueError, match="is not a d1-3B model"):
            D1_3B(folder)
    bare = tmp_path / "no-config"
    bare.mkdir()
    assert D1_3B.claims(bare) is None and D1_3B.claims(tmp_path / "missing") is None


def test_the_facts_of_d1_3b():
    assert ABOUT.name == "d1-3B" and ABOUT.reads == {"video", "image"} and ABOUT.files == ()
    assert ABOUT.types == ("noul", "choice", "score") and ABOUT.question_keys == {"criteria"}
    assert ABOUT.extracts_from == frozenset() and ABOUT.speech is False
    assert ABOUT.max_px == CAP == 1024 * 1024, "runner.VISION_MAX_PIXELS"
    assert ABOUT.questions_per_read == 0, "the state is read once and each question is a branch of the same pass"
    assert ABOUT.picture_s == ((230_400, 0.051), (CAP, 0.177)) and ABOUT.char_s == 0.050 / 2000
    assert ABOUT.hit == 0.8 and ABOUT.measured == {"picture_s", "char_s"}, "the costs are measured, the hit assumed"
    assert ABOUT.notes == "d1_3b.md" and (ABOUT.repo, ABOUT.repo_gb) == ("LiquidAI/d1-3B", 6.3)
    assert ABOUT.profile == {}
    assert decider_d1_3b.VARIANTS == {(2048, 30): ABOUT}
    assert decider.known("d1-3b") == (D1_3B, ABOUT)


def test_what_a_backend_has(tmp_path):
    assert D1_3B.family == "d1_3b" and D1_3B.abouts == (ABOUT,)
    assert D1_3B.requirements is decider_d1.D1.requirements and D1_3B.install == decider_d1.D1.install
    assert D1_3B.torch_backend is None and D1_3B.served is False and D1_3B.in_mms_env is True
    assert D1_3B.default_url is None and D1_3B.start == ""
    assert "d1-3B" in D1_3B.reads_folder and "lfm2_vl" in D1_3B.reads_folder
    assert decider_d1_3b.WORKER.name == "d1_3b.py" and decider_d1_3b.WORKER.is_file()
    made = D1_3B(model_folder(tmp_path, "d1-3B"), envs="C:/envs")
    assert isinstance(made.worker, transport.Worker) and made.worker.pid is None, "constructing starts nothing"
    assert made.worker.python == modelenv.mms_python() and made.worker.family_file == decider_d1_3b.WORKER
    assert made.worker.name == "d1-3B" and made.worker.folder == made.folder and made.worker.options == {}


def test_d1_3b_is_not_served_so_a_url_is_refused(tmp_path):
    folder = model_folder(tmp_path, "d1-3B")
    with pytest.raises(ValueError, match=r"not served.*127\.0\.0\.1:8741"):
        decider.load_decider(str(folder), url="http://127.0.0.1:8741")


def test_the_backend_imports_no_heavy_library(tmp_path):
    folder = model_folder(tmp_path, "d1-3B")
    code = ("import sys; from mms import decider; d = decider.load_decider(sys.argv[1]); "
            "assert d.about.name == 'd1-3B'; "
            "assert not {'torch', 'transformers'} & set(sys.modules), 'a heavy library was imported'")
    done = subprocess.run([sys.executable, "-c", code, str(folder)], capture_output=True, text=True, encoding="utf-8",
                          timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


# -- the questions -----------------------------------------------------------------------------------------------------

def test_d1_3b_is_asked_the_type_the_instructions_and_the_criteria_and_nothing_else(tmp_path):
    made = backend(tmp_path)
    asked = made.to_model(made.about, QUESTIONS)
    assert asked == QUESTIONS and asked is not QUESTIONS and asked["which"] is not QUESTIONS["which"]
    noul = {"seen": {"type": "noul", "instructions": "x", "criteria": {"true": "a mug", "false": "no mug"}}}
    assert made.to_model(made.about, noul) == noul
    assert D1_3B.check(made.about, QUESTIONS) is None


@pytest.mark.parametrize("extra", [{"multiple": True}, {"range": [0, 10]}, {"unit": "beeps"}, {"step": 1}, {"hit": 0.8},
                                   {"colour": "red"}])
def test_a_key_d1_3b_does_not_take_is_refused_by_name(tmp_path, extra):
    made = backend(tmp_path)
    question = {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}, **extra}}
    with pytest.raises(ValueError, match=rf"question 'q' has {next(iter(extra))}, which d1-3B does not take for a choice "
                                         "question; it takes criteria, instructions, type"):
        D1_3B.check(made.about, question)


@pytest.mark.parametrize("kind", ["text", "number", "date", "box", "boxes"])
def test_a_type_d1_3b_does_not_answer_is_refused_naming_the_types_it_does(tmp_path, kind):
    made = backend(tmp_path)
    with pytest.raises(ValueError, match=rf"d1-3B answers noul, choice, score, not '{kind}'"):
        D1_3B.check(made.about, {"q": {"type": kind, "instructions": "x"}})


REFUSED = [
    ({"type": "score", "instructions": "x", "criteria": [str(n) for n in range(11)]}, "d1-3B takes 2 to 10 levels in a score, not 11"),
    ({"type": "score", "instructions": "x", "criteria": ["only"]}, "at least 2 levels"),
    ({"type": "noul", "instructions": "x", "criteria": {"true": "lit", "dark": "off"}}, "criteria describes true and false only"),
    ({"type": "choice", "instructions": "x", "criteria": {str(n): None for n in range(256)}}, "1 to 255 options"),
    ({"type": "choice", "instructions": "x"}, "a choice has criteria"),
]


@pytest.mark.parametrize("question,words", REFUSED)
def test_a_question_d1_3b_refuses_is_refused_by_check_naming_it_and_before_its_worker_is_asked(tmp_path, question, words):
    made = backend(tmp_path)
    made.worker = Worker()
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        D1_3B.check(made.about, {"ok": QUESTIONS["seen"], "q": question})
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        made.decide(None, {"q": question})
    assert made.worker.asked == [], "nothing is asked of the model"


def test_a_score_of_ten_levels_is_taken(tmp_path):
    made = backend(tmp_path)
    ten = {"q": {"type": "score", "instructions": "x", "criteria": [str(n) for n in range(10)]}}
    assert D1_3B.check(made.about, ten) is None and made.to_model(made.about, ten) == ten


# -- the answers -------------------------------------------------------------------------------------------------------

def test_decide_hands_the_worker_its_questions_state_and_pictures_and_gives_back_mms_answers(tmp_path):
    made = backend(tmp_path)
    picture = Image.new("RGB", (4, 2))
    made.worker = Worker({"seen": NOUL, "which": CHOICE, "amount": SCORE})
    got = made.decide(None, QUESTIONS, [picture])
    assert made.worker.asked == [(None, QUESTIONS, [picture])], "state None stays None"
    assert got["seen"] == NOUL and got["which"] == {key: v for key, v in CHOICE.items() if key != "confidence"}
    assert got["amount"]["score"] == 2.0 and got["amount"]["probabilities"] == SCORE["probabilities"]
    made.decide("text", {"seen": QUESTIONS["seen"]})
    assert made.worker.asked[-1] == ("text", {"seen": QUESTIONS["seen"]}, [])


def test_d1_3bs_answers_read_and_judge_in_mms_terms(ask, tmp_path):
    made = backend(tmp_path)
    made.worker = Worker({"seen": NOUL, "which": CHOICE, "amount": SCORE})
    got = made.decide(None, QUESTIONS, [])
    assert "confidence" not in got["which"] and got["which"]["probabilities"] == CHOICE["probabilities"]
    assert got["amount"]["score"] == 2.0, "the median level of 0.2, 0.2, 0.6; its own 1.6 is the expected level"
    assert SCORE["score"] == 1.6 and SCORE["confidence"] == 0.6 and CHOICE["confidence"] == 0.7, "what it gave is left as it was"
    about = made.about
    assert verdict(ask, QUESTIONS["seen"], True, about, got["seen"])[:2] == (True, 0.93)
    assert verdict(ask, QUESTIONS["seen"], 0.95, about, got["seen"])[:2] == (False, 0.93)
    assert verdict(ask, QUESTIONS["which"], "cup", about, got["which"]) == (True, 0.7, "cup")
    assert verdict(ask, QUESTIONS["which"], "mug", about, got["which"]) == (False, 0.1, "cup")
    assert verdict(ask, QUESTIONS["amount"], 2, about, got["amount"]) == (True, 1.0, "level 2"), \
        "level 2 counts, which the expected level 1.6 would not"
    assert verdict(ask, QUESTIONS["amount"], 3, about, got["amount"])[0] is False


def test_a_score_without_the_levels_cannot_be_read_and_is_never_passed_off_as_the_expected_level(ask, tmp_path):
    made = backend(tmp_path)
    made.worker = Worker({"amount": {**SCORE, "probabilities": {"a": 0.5, "b": 0.5}}})
    got = made.decide(None, {"amount": QUESTIONS["amount"]}, [])
    assert got["amount"]["score"] is None
    with pytest.raises(ask.Unreadable, match="its score is None"):
        verdict(ask, QUESTIONS["amount"], 2, made.about, got["amount"])


def test_an_error_entry_or_a_missing_answer_is_a_valueerror_naming_the_question(tmp_path):
    made = backend(tmp_path)
    made.worker = Worker({"seen": {"type": "noul", "error": "invalid_question", "message": "too long"}})
    with pytest.raises(ValueError, match=r"d1-3B: question 'seen': invalid_question: too long"):
        made.decide(None, {"seen": QUESTIONS["seen"]}, [])
    made.worker = Worker({})
    with pytest.raises(ValueError, match="gave no answer to question 'seen'"):
        made.decide(None, {"seen": QUESTIONS["seen"]}, [])


def test_load_checks_the_code_caps_pictures_where_the_facts_say(tmp_path):
    made = backend(tmp_path)
    made.worker = Worker(facts={"torch": "x", "transformers": "y", "max_pixels": CAP})
    made.load()
    assert made.worker.unloaded == 0
    made.worker = Worker(facts={"max_pixels": 1_638_400})
    with pytest.raises(RuntimeError) as raised:
        made.load()
    message = str(raised.value)
    assert "1638400" in message and str(CAP) in message and "VISION_MAX_PIXELS" in message and str(made.folder) in message
    assert made.worker.unloaded == 1, "a worker that holds code of another cap is ended"
    made.worker = Worker(facts={})
    with pytest.raises(RuntimeError, match="None pixels"):
        made.load()
    made.unload()
    assert made.worker.unloaded == 2


# -- worker/d1_3b.py, against a fake model -----------------------------------------------------------------------------

class FakeModel:
    """What `AutoModel.from_pretrained` returns: records each `system_one`, which takes `images` as d1-3B's does."""

    def __init__(self):
        self.calls, self.device, self.evaluated = [], None, False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def system_one(self, state, questions, images=None):
        self.calls.append((state, questions, images))
        return {"answers": {name: {"noul": 0.9} for name in questions}, "usage": {"input_tokens": 3, "output_tokens": 0}}


class FakeMediaModel(FakeModel):
    """A model whose `system_one` takes `media`, as d1-Omni-2B's does: not d1-3B's code."""

    def system_one(self, state, questions, media=None):
        raise AssertionError("never called")


def fake_libraries(monkeypatch, cap=CAP, hybrid=True) -> SimpleNamespace:
    """A fake torch (with `no_grad` that counts how deep it is), a fake transformers whose AutoModel makes a FakeModel, and
    fake modules of the model's own code: `runner` with `VISION_MAX_PIXELS` (`cap`, none for None), `hybrid` with `_fused`
    (unless `hybrid` is false), and an `attention`. Whatever of the model's code an earlier call left is removed."""
    for name in [name for name in sys.modules if name.startswith("transformers_modules")]:
        monkeypatch.delitem(sys.modules, name)
    state = SimpleNamespace(depth=0, entered=0, loaded=[], model=FakeModel(), found={})

    @contextlib.contextmanager
    def no_grad():
        state.depth += 1
        state.entered += 1
        try:
            yield
        finally:
            state.depth -= 1

    torch = types.ModuleType("torch")
    torch.__version__, torch.bfloat16, torch.no_grad = "0.0-fake", "bf16", no_grad
    torch.cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "0.0-fake"

    class AutoModel:
        @staticmethod
        def from_pretrained(folder, **kwargs):
            state.loaded.append((folder, kwargs))
            return state.model, state.found

    transformers.AutoModel = AutoModel
    modules = {"torch": torch, "transformers": transformers}
    state.runner = types.ModuleType("transformers_modules.fake.runner")
    if cap is not None:
        state.runner.VISION_MAX_PIXELS = cap
        modules["transformers_modules.fake.runner"] = state.runner
    state.attention = types.ModuleType("transformers_modules.fake.attention")
    state.attention._fused = lambda q: True
    modules["transformers_modules.fake.attention"] = state.attention
    state.hybrid = types.ModuleType("transformers_modules.fake.hybrid")
    state.hybrid._fused = lambda q: True
    if hybrid:
        modules["transformers_modules.fake.hybrid"] = state.hybrid
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def kinds(images) -> list[str]:
    return ["image"] * len(images or [])


def test_decide_passes_the_pictures_as_images_and_returns_the_answers(monkeypatch):
    libs = fake_libraries(monkeypatch)
    worker = load_worker("d1_3b")
    state, questions = "Kitchen log.", {"q": {"type": "noul", "instructions": "Is the kettle boiling?"},
                                        "r": {"type": "score", "instructions": "How hot?", "criteria": ["a", "b"]}}
    picture, other = Image.new("RGB", (4, 4)), Image.new("RGB", (2, 2))

    assert worker.decide(libs.model, state, questions, [picture, other]) == {"q": {"noul": 0.9}, "r": {"noul": 0.9}}, \
        "only the model's answers come back, not its usage"
    (seen_state, seen_questions, seen_images), = libs.model.calls
    assert seen_state == state and seen_questions is questions
    assert seen_images == [picture, other] and seen_images[0] is picture
    assert libs.entered == 1 and libs.depth == 0, "the call ran under torch.no_grad"

    worker.decide(libs.model, None, questions, [])
    worker.decide(libs.model, None, questions, None)
    assert libs.model.calls[1][2] is None and libs.model.calls[2][2] is None, "no pictures is None, as d1-3B takes it"
    assert libs.model.calls[1][0] is None, "state None stays None"


def test_a_sound_or_a_file_is_refused_before_the_model_is_called(monkeypatch):
    libs = fake_libraries(monkeypatch)
    worker = load_worker("d1_3b")
    questions = {"q": {"type": "noul", "instructions": "x"}}
    with pytest.raises(ValueError, match="d1-3B reads pictures, not audio"):
        worker.decide(libs.model, None, questions, [{"type": "audio", "audio": [0.0], "sampling_rate": 16000}])
    with pytest.raises(ValueError, match="d1-3B reads pictures, not file"):
        worker.decide(libs.model, None, questions, [Image.new("RGB", (2, 2)), {"type": "file", "media": "application/pdf",
                                                                               "name": "a.pdf", "data": b""}])
    assert libs.model.calls == []


def test_a_loaded_model_is_on_the_gpu_checked_and_warmed_with_a_text_and_a_picture(monkeypatch, tmp_path):
    libs = fake_libraries(monkeypatch)
    folder = model_folder(tmp_path, "d1-3B")
    model, facts = load_worker("d1_3b").load(str(folder), {})
    assert model is libs.model and model.device == "cuda" and model.evaluated
    assert facts == {"torch": "0.0-fake", "transformers": "0.0-fake", "max_pixels": CAP}, \
        "the picture cap is the loaded code's, for the backend to check"
    (loaded, kwargs), = libs.loaded
    assert loaded == str(folder) and kwargs == {"trust_remote_code": True, "dtype": "bf16", "output_loading_info": True}
    assert [kinds(images) for _, _, images in model.calls] == [[], ["image"]], "a text, then a picture, and no sound"
    assert model.calls[0][0] == {"warm": 1} and model.calls[1][0] is None
    assert model.calls[1][2][0].size == (640, 360)
    assert libs.attention._fused(None) is True, "only the hybrid module is patched"
    if sys.platform == "win32":
        assert libs.hybrid._fused(None) is False, "torch's Windows wheels have no flash attention: explicit products"
    else:
        assert libs.hybrid._fused(None) is True


@pytest.mark.skipif(sys.platform != "win32", reason="the patch is for torch's Windows wheels")
def test_on_windows_every_hybrid_module_is_patched_and_a_missing_one_is_an_error(monkeypatch, tmp_path):
    folder = str(model_folder(tmp_path, "d1-3B"))
    libs = fake_libraries(monkeypatch)
    second = types.ModuleType("transformers_modules.other.hybrid")
    second._fused = lambda q: True
    unrelated = types.ModuleType("somewhere.hybrid")
    unrelated._fused = lambda q: True
    monkeypatch.setitem(sys.modules, "transformers_modules.other.hybrid", second)
    monkeypatch.setitem(sys.modules, "somewhere.hybrid", unrelated)
    load_worker("d1_3b").load(folder, {})
    assert libs.hybrid._fused(None) is False and second._fused(None) is False
    assert unrelated._fused(None) is True, "only the model's own code"

    fake_libraries(monkeypatch, hybrid=False)
    with pytest.raises(RuntimeError, match="no d1-3B hybrid module to patch"):
        load_worker("d1_3b").load(folder, {})


def test_a_load_says_what_is_wrong_with_the_weights_the_code_or_the_machine(monkeypatch, tmp_path):
    folder = str(model_folder(tmp_path, "d1-3B"))
    libs = fake_libraries(monkeypatch)
    libs.found = {"missing_keys": ["a.b", "c"], "unexpected_keys": [], "mismatched_keys": [("w", 1, 2)]}
    with pytest.raises(RuntimeError, match=r"do not fit the code: 2 missing \(a\.b, c\); 1 mismatched \(w\)"):
        load_worker("d1_3b").load(folder, {})

    libs = fake_libraries(monkeypatch)
    libs.model = FakeMediaModel()
    with pytest.raises(RuntimeError, match="system_one takes no `images`"):
        load_worker("d1_3b").load(folder, {})

    libs = fake_libraries(monkeypatch)
    sys.modules["torch"].cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="needs a CUDA GPU"):
        load_worker("d1_3b").load(folder, {})

    fake_libraries(monkeypatch)
    with pytest.raises(FileNotFoundError, match="no d1-3B model in"):
        load_worker("d1_3b").load(str(tmp_path / "nothing"), {})


def test_the_picture_cap_of_the_loaded_code_is_found_or_the_load_fails(monkeypatch, tmp_path):
    folder = str(model_folder(tmp_path, "d1-3B"))
    fake_libraries(monkeypatch, cap=None)
    with pytest.raises(RuntimeError, match="expected one VISION_MAX_PIXELS in d1-3B's runner module, found none"):
        load_worker("d1_3b").load(folder, {})

    fake_libraries(monkeypatch)
    another = types.ModuleType("transformers_modules.other.runner")
    another.VISION_MAX_PIXELS = 1_638_400
    monkeypatch.setitem(sys.modules, "transformers_modules.other.runner", another)
    with pytest.raises(RuntimeError, match=rf"found \[{CAP}, 1638400\]"):
        load_worker("d1_3b").load(folder, {})

    libs = fake_libraries(monkeypatch)
    libs.runner.VISION_MAX_PIXELS = 640 * 360
    _, facts = load_worker("d1_3b").load(folder, {})
    assert facts["max_pixels"] == 640 * 360, "the worker reports the number; the backend checks it"


def test_the_worker_file_imports_with_the_standard_library_alone():
    done = subprocess.run([sys.executable, "-I", "-S", "-c", textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("d1_3b_worker", {str(WORKER_D1_3B)!r})
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        assert callable(module.load) and callable(module.decide)
        assert not {{"torch", "transformers", "numpy", "PIL", "mms"}} & set(sys.modules), sorted(sys.modules)
    """)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


# -- a round trip through a real Worker, with worker/d1_3b.py's real functions over a fake model -----------------------

FAMILY = '''
import contextlib, importlib.util, sys, types
from types import SimpleNamespace

class Model:
    def __init__(self):
        self.history = []
    def to(self, device): return self
    def eval(self): return self
    def system_one(self, state, questions, images=None):
        seen = [list(image.size) for image in images or []]
        answers = {name: {"type": q["type"], "noul": 0.75, "state": state, "images": seen} for name, q in questions.items()}
        if "history" in questions:
            answers["history"]["history"] = list(self.history)
        self.history.append(len(seen))
        return {"answers": answers}

torch = types.ModuleType("torch")
torch.__version__, torch.bfloat16 = "fake", "bf16"
torch.no_grad = contextlib.nullcontext
torch.cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)
transformers = types.ModuleType("transformers")
transformers.__version__ = "fake"
transformers.AutoModel = SimpleNamespace(from_pretrained=lambda folder, **kw: (Model(), {}))
runner = types.ModuleType("transformers_modules.fake.runner")
runner.VISION_MAX_PIXELS = CAP
hybrid = types.ModuleType("transformers_modules.fake.hybrid")
hybrid._fused = lambda q: True
sys.modules.update({"torch": torch, "transformers": transformers, "transformers_modules.fake.runner": runner,
                    "transformers_modules.fake.hybrid": hybrid})

spec = importlib.util.spec_from_file_location("d1_3b_under_test", WORKER_D1_3B)
d1_3b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1_3b)
load, decide = d1_3b.load, d1_3b.decide
'''


def family_file(tmp_path: Path, cap: int = CAP) -> Path:
    path = tmp_path / "family.py"
    path.write_text(FAMILY.replace("WORKER_D1_3B", repr(str(WORKER_D1_3B))).replace("CAP", str(cap)), encoding="utf-8")
    return path


def test_a_d1_3b_round_trip_through_a_real_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(decider_d1_3b, "WORKER", family_file(tmp_path))  # the child half is worker/d1_3b.py over a fake model
    d1_3b = decider.load_decider(str(model_folder(tmp_path, "d1-3B")))
    try:
        d1_3b.load()
        assert d1_3b.worker.facts == {"torch": "fake", "transformers": "fake", "max_pixels": CAP}
        ask = {"history": {"type": "noul", "instructions": "x"}, "q": {"type": "noul", "instructions": "Is it?"}}
        answers = d1_3b.decide("a state", ask, [Image.new("RGB", (6, 4)), Image.new("RGB", (3, 5))])
        assert answers["history"]["history"] == [0, 1], "what the warm-ups sent: a text, then a picture"
        assert answers["q"] == {"type": "noul", "noul": 0.75, "state": "a state", "images": [[6, 4], [3, 5]]}, \
            "the pictures reached the model as sent"
        assert d1_3b.decide(None, {"q": ask["q"]}, [])["q"]["images"] == [], "a window with no pictures"
        with pytest.raises(ValueError, match="reads pictures, not audio"):
            d1_3b.decide(None, {"q": ask["q"]}, [{"type": "audio", "audio": np.zeros(8, "<f4"), "sampling_rate": 16000}])
        with pytest.raises(ValueError, match="2 to 10 levels"):
            d1_3b.decide(None, {"q": {"type": "score", "instructions": "x", "criteria": list("abcdefghijk")}}, [])
        assert d1_3b.worker.pid is not None
    finally:
        d1_3b.unload()
    assert d1_3b.worker.pid is None, "unload ended the worker"


def test_a_worker_whose_code_caps_pictures_elsewhere_is_refused_naming_both_numbers_and_is_gone(monkeypatch, tmp_path):
    monkeypatch.setattr(decider_d1_3b, "WORKER", family_file(tmp_path, cap=1_638_400))
    d1_3b = decider.load_decider(str(model_folder(tmp_path, "d1-3B")))
    with pytest.raises(RuntimeError) as raised:
        d1_3b.load()
    assert "1638400" in str(raised.value) and str(CAP) in str(raised.value) and "max_px" in str(raised.value)
    assert d1_3b.worker.pid is None, "the worker was ended"
