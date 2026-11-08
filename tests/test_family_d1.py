"""d1 and d1-2B as a decider family (TESTBED.md, "No GPU"; DESIGN.md §13.3): `decider_d1`'s claims, facts and checks, and
`worker/d1.py`, the half that holds the model. The model, torch and transformers are fakes, so nothing here needs a GPU or
weights; the real d1 is in `test_gpu.py`."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ROOT, model_folder
from PIL import Image

from mms import decider, decider_d1, modelenv, transport

WORKER_D1 = ROOT / "src" / "mms" / "worker" / "d1.py"
D1_CONFIG = {"model_type": "lfm3_omni", "d1_attention": "set", "d1_set_size": 16, "d1_vision_max_pixels": 1638400,
             "d1_numbers": True, "d1_codes": 2}


def worker_d1():
    """worker/d1.py as the worker child loads it: by its path."""
    spec = importlib.util.spec_from_file_location("mms_worker_d1_under_test", WORKER_D1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- claims and facts --------------------------------------------------------------------------------------------------

def test_each_d1_folder_is_claimed_as_its_own_model(tmp_path):
    omni, small = model_folder(tmp_path, "d1-Omni-2B"), model_folder(tmp_path, "d1-2B")
    assert decider_d1.D1.claims(omni) is decider_d1.ABOUT and decider_d1.D1.claims(small) is decider_d1.ABOUT_2B
    assert "audio" in decider_d1.ABOUT.reads and "audio" not in decider_d1.ABOUT_2B.reads, \
        "d1-2B has the same keys as d1-Omni-2B and no audio: it must not be handed sound"
    assert decider_d1.ABOUT_2B.reads == {"video", "image"} and decider_d1.ABOUT_2B.name == "d1-2B"
    assert decider.about_of(str(omni)) is decider_d1.ABOUT and decider.about_of(str(small)) is decider_d1.ABOUT_2B
    assert isinstance(decider.load_decider(str(small)), decider_d1.D1)
    assert decider.load_decider(str(small)).about is decider_d1.ABOUT_2B
    assert decider.known_abouts()[:2] == [decider_d1.ABOUT, decider_d1.ABOUT_2B], "d1's two variants come first, in order"


def test_the_facts_of_both_models(tmp_path):
    omni, small = decider_d1.ABOUT, decider_d1.ABOUT_2B
    assert (omni.hit, small.hit) == (0.8, 0.8) and omni.notes == small.notes == "d1.md"
    assert [(a.repo, a.repo_gb) for a in (omni, small)] == [("LiquidAI/d1-Omni-2B", 4.4), ("LiquidAI/d1-2B", 3.9)],         "where `mms model use` downloads them from"
    assert omni.measured == {"picture_s", "sound_s", "char_s", "hit"} and small.measured == {"picture_s", "char_s"}, \
        "d1-2B's costs were measured here, and only d1-Omni-2B's hit"
    for about in (omni, small):
        assert about.max_px == 1_638_400 and about.questions_per_read == 16 and about.audio_rate == 16000
        assert about.types == decider.TYPES and about.question_keys == set(decider.QUESTION_KEYS)
        assert about.extracts_from == {"text", "records"} and about.speech is False
    assert small.picture_s == ((230_400, 0.053), (1_638_400, 0.487)) and small.char_s == 0.047 / 2000, \
        "d1-2B's costs, as measured on 1 Nov"


def test_what_a_backend_has(tmp_path):
    backend = decider_d1.D1
    assert backend.family == "d1" and backend.abouts == (decider_d1.ABOUT, decider_d1.ABOUT_2B)
    assert backend.requirements == ("torch", "torchvision", "transformers==5.21.0") and backend.torch_backend is None
    assert backend.served is False and backend.in_mms_env is True and backend.default_url is None and backend.start == ""
    assert backend.install == "uv sync --group d1"
    assert "d1-Omni-2B" in backend.reads_folder and "d1-2B" in backend.reads_folder and "model_type" in backend.reads_folder
    assert decider_d1.WORKER.name == "d1.py" and decider_d1.WORKER.is_file()
    made = decider_d1.D1(model_folder(tmp_path, "d1-2B"), envs="C:/envs")
    assert isinstance(made.worker, transport.Worker) and made.worker.pid is None, "constructing starts nothing"
    assert made.worker.python == modelenv.mms_python() and made.worker.family_file == decider_d1.WORKER
    assert made.worker.name == "d1-2B" and made.worker.folder == made.folder
    assert made.worker.options == {"warm_sound": False}
    assert decider_d1.D1(model_folder(tmp_path / "other", "d1-Omni-2B")).worker.options == {"warm_sound": True}


def test_other_model_types_and_other_folders_are_not_claimed(tmp_path):
    keys = {key: value for key, value in D1_CONFIG.items() if key != "model_type"}
    cases = {"other-type": {**keys, "model_type": "gpt2"}, "no-type": keys, "type-not-text": {**keys, "model_type": 7},
             "no-keys": {"model_type": "lfm3_omni"}, "one-key-short": {**D1_CONFIG, "d1_codes": None},
             "not-json": "{", "a-list": "[]", "empty": "{}"}
    del cases["one-key-short"]["d1_codes"]
    for name, text in cases.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(text if isinstance(text, str) else json.dumps(text), encoding="utf-8")
        assert decider_d1.D1.claims(folder) is None and decider.about_of(str(folder)) is None, name
        with pytest.raises(ValueError, match="no decider reads the model") as raised:
            decider.load_decider(str(folder))
        assert "d1-Omni-2B" in str(raised.value) and "d1-2B" in str(raised.value) and "model_type" in str(raised.value), \
            f"{name}: the error names what is read: {raised.value}"
        with pytest.raises(ValueError, match="is not a d1 model"):
            decider_d1.D1(folder)
    bare = tmp_path / "no-config"
    bare.mkdir()
    assert decider_d1.D1.claims(bare) is None, "no config.json"


@pytest.mark.parametrize("name", ["d1-Omni-2B", "d1-2B"])
def test_weights_that_differ_from_the_facts_are_refused_naming_both_numbers(tmp_path, name):
    for key, wrong, right in (("d1_set_size", 8, 16), ("d1_set_size", 4, 16), ("d1_vision_max_pixels", 1003520, 1638400)):
        folder = model_folder(tmp_path / f"{key}-{wrong}", name)
        recorded = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        (folder / "config.json").write_text(json.dumps({**recorded, key: wrong}), encoding="utf-8")
        with pytest.raises(ValueError) as raised:
            decider.load_decider(str(folder))
        assert key in str(raised.value) and str(wrong) in str(raised.value) and str(right) in str(raised.value) \
            and name in str(raised.value), f"{name} {key}: {raised.value}"


def test_d1_is_not_served_so_a_url_is_refused(tmp_path):
    folder = model_folder(tmp_path, "d1-Omni-2B")
    with pytest.raises(ValueError, match=r"not served.*127\.0\.0\.1:8741"):
        decider.load_decider(str(folder), url="http://127.0.0.1:8741")


def test_the_backend_imports_no_heavy_library(tmp_path):
    folder = model_folder(tmp_path, "d1-Omni-2B")
    code = ("import sys; from mms import decider; d = decider.load_decider(sys.argv[1]); "
            "assert d.about.name == 'd1-Omni-2B'; "
            "assert not {'torch', 'transformers'} & set(sys.modules), 'a heavy library was imported'")
    done = subprocess.run([sys.executable, "-c", code, str(folder)], capture_output=True, text=True, encoding="utf-8",
                          timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_worker_file_imports_with_the_standard_library_alone():
    done = subprocess.run([sys.executable, "-I", "-S", "-c", textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("d1_worker", {str(WORKER_D1)!r})
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        assert callable(module.load) and callable(module.decide)
        assert not {{"torch", "transformers", "numpy", "PIL", "mms"}} & set(sys.modules), sorted(sys.modules)
    """)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


# -- worker/d1.py, against a fake model ---------------------------------------------------------------------------------

class FakeModel:
    """What `AutoModel.from_pretrained` returns: records each `system_one`, and holds the engine `_check_config` reads."""

    def __init__(self):
        self.calls, self.device, self.evaluated = [], None, False
        self.engine = SimpleNamespace(calibration=object(), attention="set", set_size=16, max_pixels=1638400,
                                      tokenizer=SimpleNamespace(d1_numbers=True, d1_codes=2))

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def system_one(self, state, questions, media=None):
        self.calls.append((state, questions, media))
        return {"answers": {name: {"noul": 0.9} for name in questions}, "not_answers": 1}


def fake_libraries(monkeypatch) -> SimpleNamespace:
    """A fake torch (with `no_grad` that counts how deep it is) and a fake transformers whose AutoModel makes a FakeModel."""
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
    attention = types.ModuleType("transformers_modules.fake.attention")
    attention._fused = lambda q: True
    state.attention = attention
    for name, module in (("torch", torch), ("transformers", transformers),
                         ("transformers_modules.fake.attention", attention)):
        monkeypatch.setitem(sys.modules, name, module)
    return state


def kinds(media) -> list[str]:
    return ["audio" if isinstance(m, dict) else "image" for m in media or []]


def test_decide_passes_the_window_through_and_returns_the_answers(monkeypatch):
    libs = fake_libraries(monkeypatch)
    state, questions = "Kitchen log.", {"q": {"type": "noul", "instructions": "Is the kettle boiling?"},
                                        "r": {"type": "score", "instructions": "How hot?", "criteria": ["a", "b"]}}
    picture = Image.new("RGB", (4, 4))

    answers = worker_d1().decide(libs.model, state, questions, [picture])
    assert answers == {"q": {"noul": 0.9}, "r": {"noul": 0.9}}, "only the model's answers come back, not its other keys"
    (seen_state, seen_questions, seen_media), = libs.model.calls
    assert seen_state == state and seen_questions is questions and seen_media == [picture] and seen_media[0] is picture
    assert libs.entered == 1 and libs.depth == 0, "the call ran under torch.no_grad"

    worker_d1().decide(libs.model, None, questions, [])
    worker_d1().decide(libs.model, None, questions, None)
    assert libs.model.calls[1][2] is None and libs.model.calls[2][2] is None, "no media is None, as d1 takes it"
    assert libs.model.calls[1][0] is None


def test_a_loaded_model_is_on_the_gpu_checked_and_warmed_with_sound_only_when_it_hears(monkeypatch, tmp_path):
    for name, hears in (("d1-Omni-2B", True), ("d1-2B", False)):
        libs = fake_libraries(monkeypatch)
        folder = model_folder(tmp_path, name)
        model, facts = worker_d1().load(str(folder), {"warm_sound": hears})
        assert model is libs.model and model.device == "cuda" and model.evaluated
        assert facts == {"torch": "0.0-fake", "transformers": "0.0-fake"}
        (loaded, kwargs), = libs.loaded
        assert loaded == str(folder) and kwargs["trust_remote_code"] is True and kwargs["dtype"] == "bf16" \
            and kwargs["output_loading_info"] is True
        assert [kinds(media) for _, _, media in model.calls] == ([[], ["image"], ["audio"]] if hears else [[], ["image"]]), \
            f"{name}: the warm-ups"
        sounds = [m for _, _, media in model.calls for m in media or [] if isinstance(m, dict)]
        assert len(sounds) == (1 if hears else 0)
        for sound in sounds:
            assert sound["sampling_rate"] == 16000 and sound["audio"].dtype == np.float32 and sound["audio"].size == 16000
        if sys.platform == "win32":
            assert libs.attention._fused(None) is False, "torch's Windows wheels: d1 takes its masked-SDPA path"


def test_a_load_says_what_is_wrong_with_the_weights_or_the_machine(monkeypatch, tmp_path):
    folder = str(model_folder(tmp_path, "d1-Omni-2B"))
    libs = fake_libraries(monkeypatch)
    libs.found = {"missing_keys": ["a.b", "c"], "unexpected_keys": [], "mismatched_keys": [("w", 1, 2)]}
    with pytest.raises(RuntimeError, match=r"do not fit the code: 2 missing \(a\.b, c\); 1 mismatched \(w\)"):
        worker_d1().load(folder, {"warm_sound": True})

    libs = fake_libraries(monkeypatch)
    libs.model.engine.set_size = 4
    with pytest.raises(RuntimeError, match=r"fell to defaults.*d1_set_size is 16 in config.json but the engine holds 4"):
        worker_d1().load(folder, {"warm_sound": True})

    libs = fake_libraries(monkeypatch)
    sys.modules["torch"].cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="needs a CUDA GPU"):
        worker_d1().load(folder, {"warm_sound": True})

    fake_libraries(monkeypatch)
    with pytest.raises(FileNotFoundError, match="no d1 model in"):
        worker_d1().load(str(tmp_path / "nothing"), {"warm_sound": True})


# -- a round trip through a real Worker, with worker/d1.py's real functions over a fake model ---------------------------

FAMILY = '''
import contextlib, importlib.util, sys, types
from types import SimpleNamespace

class Model:
    def __init__(self):
        self.history = []
        self.engine = SimpleNamespace(calibration=object(), attention="set", set_size=16, max_pixels=1638400,
                                      tokenizer=SimpleNamespace(d1_numbers=True, d1_codes=2))
    def to(self, device): return self
    def eval(self): return self
    def system_one(self, state, questions, media=None):
        seen = [["audio", int(m["audio"].size), m["sampling_rate"]] if isinstance(m, dict) else ["image", *m.size]
                for m in media or []]
        answers = {name: {"noul": 0.75, "state": state, "media": seen, "type": q["type"]} for name, q in questions.items()}
        if "history" in questions:
            answers["history"]["history"] = list(self.history)
        self.history.append([item[0] for item in seen])
        return {"answers": answers}

torch = types.ModuleType("torch")
torch.__version__, torch.bfloat16 = "fake", "bf16"
torch.no_grad = contextlib.nullcontext
torch.cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None)
transformers = types.ModuleType("transformers")
transformers.__version__ = "fake"
transformers.AutoModel = SimpleNamespace(from_pretrained=lambda folder, **kw: (Model(), {}))
attention = types.ModuleType("transformers_modules.fake.attention")
attention._fused = lambda q: True
sys.modules.update({"torch": torch, "transformers": transformers, "transformers_modules.fake.attention": attention})

spec = importlib.util.spec_from_file_location("d1_under_test", WORKER_D1)
d1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1)
load, decide = d1.load, d1.decide
'''.replace("WORKER_D1", repr(str(WORKER_D1)))


def test_a_d1_round_trip_through_a_real_worker(monkeypatch, tmp_path):
    family = tmp_path / "family.py"
    family.write_text(FAMILY, encoding="utf-8")
    monkeypatch.setattr(decider_d1, "WORKER", family)  # the child half is worker/d1.py over a fake model
    wave = (np.arange(8000, dtype=np.float32) / 8000).astype("<f4")
    for name, warmed in (("d1-Omni-2B", [[], ["image"], ["audio"]]), ("d1-2B", [[], ["image"]])):
        d1 = decider.load_decider(str(model_folder(tmp_path, name)))
        try:
            d1.load()
            assert d1.worker.facts == {"torch": "fake", "transformers": "fake"}
            ask = {"history": {"type": "noul", "instructions": "x"}, "q": {"type": "noul", "instructions": "Is it?"}}
            answers = d1.decide("a state", ask, [Image.new("RGB", (6, 4)), {"type": "audio", "audio": wave,
                                                                          "sampling_rate": 8000}])
            assert answers["history"]["history"] == warmed, f"{name}: what the warm-ups sent"
            assert answers["q"] == {"noul": 0.75, "state": "a state", "type": "noul",
                                    "media": [["image", 6, 4], ["audio", 8000, 8000]]}, "the window reached the model as sent"
            assert d1.decide(None, {"q": ask["q"]}, [])["q"]["media"] == [], "a window with no media"
            with pytest.raises(ValueError):  # a medium the wire does not send is a ValueError
                d1.decide(None, {"q": ask["q"]}, [Image.new("L", (2, 2))])
            assert d1.worker.pid is not None
        finally:
            d1.unload()
        assert d1.worker.pid is None, "unload ended the worker"
