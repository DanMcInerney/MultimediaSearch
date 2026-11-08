"""H2O-Lightning-4B's backend (TESTBED.md, "No GPU"; DESIGN.md §13.3): how a folder is claimed, the facts, the translations
from canned answers in the card's shapes (and `box.py`'s), and the requests it makes of a server: against `fake_shim.py`, a
stdlib server with no model, it checks the health wait, lossless PNG pictures, the limit of 4 and the model check. The last
test runs the whole daemon over the fake server. Nothing here needs the weights, WSL or a GPU."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from conftest import MODEL_FOLDERS, model_folder, write_config
from corpus import T0, scan
from fake_shim import MAX_IMAGES, MODEL_ID, FakeShim
from PIL import Image
from test_family_d3 import ask, verdict  # noqa: F401  the helper, and the `ask` fixture

from mms import config, decider, transport
from mms.decider_h2o import ABOUT, H2o, data_uri
from mms.errors import ApiError


def noise(width: int, height: int, seed: int = 1) -> Image.Image:
    data = np.random.default_rng(seed).integers(0, 256, width * height * 3, dtype=np.uint8).tobytes()
    return Image.frombytes("RGB", (width, height), data)


def pixels(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


@pytest.fixture
def fast(monkeypatch):
    """The wait for the server's health polled in milliseconds, not seconds."""
    monkeypatch.setattr(transport, "POLL_S", 0.01)


@pytest.fixture
def shim():
    with FakeShim() as running:
        yield running


def backend(tmp_path: Path, shim: FakeShim | None = None) -> H2o:
    folder = tmp_path / "models" / "h2o-lightning-4b"
    return H2o(folder if folder.exists() else model_folder(tmp_path / "models", folder.name), url=shim.url if shim else None)


# the card's answers (README) and box.py's
NOUL = {"type": "noul", "noul": 0.932}
CHOICE = {"type": "choice", "choice": "six", "confidence": 0.730,
          "probabilities": {"five": 0.130, "four": 0.043, "six": 0.798, "three": 0.029}}
SCORE = {"type": "score", "legend": {"0": "Unreadable", "1": "Partly readable", "2": "Fully readable"}, "score": 1.818,
         "confidence": 0.866, "probabilities": {"0": 0.048, "1": 0.086, "2": 0.866}}
BOX = {"type": "box", "box": {"left": 0.1, "top": 0.2, "right": 0.6, "bottom": 0.7}, "confidence": 0.8, "found": True,
       "present": 0.9}
NO_BOX = {"type": "box", "box": {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0}, "confidence": 0.0, "found": False,
          "present": 0.05}
QUESTIONS = {"seen": {"type": "noul", "instructions": "Is a mug in view?"},
             "count": {"type": "choice", "instructions": "How many?",
                       "criteria": {"three": None, "four": None, "five": None, "six": None}},
             "legible": {"type": "score", "instructions": "Legible?", "criteria": ["no", "partly", "fully"]},
             "where": {"type": "box", "instructions": "the mug on the left"}}


# -- claims and facts -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", MODEL_FOLDERS)
def test_an_h2o_folder_is_claimed_by_its_serve_config_and_shim_and_no_other_folder_is(tmp_path, name):
    assert H2o.claims(model_folder(tmp_path, name)) is (ABOUT if name == "h2o-lightning-4b" else None), name


def test_a_folder_for_another_model_or_without_its_shim_is_not_claimed(tmp_path):
    folder = model_folder(tmp_path, "h2o-lightning-4b")
    served = json.loads((folder / "serve_config.json").read_text(encoding="utf-8"))
    (folder / "serve_config.json").write_text(json.dumps({**served, "model": "h2oai/h2o-other"}), encoding="utf-8")
    assert H2o.claims(folder) is None
    (folder / "serve_config.json").write_text(json.dumps(served), encoding="utf-8")
    assert H2o.claims(folder) is ABOUT
    (folder / "h2o_lightning_shim.py").unlink()
    assert H2o.claims(folder) is None, "without its shim there is nothing to serve"
    (folder / "serve_config.json").write_text("[]", encoding="utf-8")
    assert H2o.claims(folder) is None
    (folder / "serve_config.json").unlink()
    assert H2o.claims(folder) is None and H2o.claims(folder / "missing") is None


def test_the_loader_finds_h2o_by_what_claims_it_with_no_env_to_prepare(home, tmp_path, monkeypatch):
    monkeypatch.setattr(decider, "BACKENDS", ("mms.decider_h2o:H2o",))
    folder = model_folder(tmp_path, "h2o-lightning-4b")
    assert decider.about_of(str(folder)) is ABOUT
    made = decider.load_decider(str(folder), home)
    assert made.about is ABOUT and made.url == H2o.default_url
    assert decider.load_decider(str(folder), home, url="http://localhost:9001", envs="C:/nowhere").url == "http://localhost:9001"
    with pytest.raises(ValueError, match="no decider reads the model.*serve_config.json"):
        decider.load_decider(str(model_folder(tmp_path, "d3-edge")), home)


def test_h2os_facts_are_the_plans_and_its_costs_are_measured():
    assert H2o.abouts == (ABOUT,)
    assert ABOUT.name == "h2o-lightning-4b" and ABOUT.reads == {"video", "image"}
    assert ABOUT.types == ("noul", "choice", "score", "box") and ABOUT.question_keys == {"criteria"}
    assert ABOUT.extracts_from == frozenset() and ABOUT.speech is False
    assert (ABOUT.max_px, ABOUT.questions_per_read, ABOUT.hit, ABOUT.notes) == (1_638_400, 1, 0.8, "h2o.md")
    assert (ABOUT.picture_s, ABOUT.char_s) == (((230_400, 0.081), (1_638_400, 0.113)), 0.087 / 2000)
    assert ABOUT.measured == {"picture_s", "char_s"}
    assert (ABOUT.repo, ABOUT.repo_gb) == ("h2oai/h2o-lightning-4b", 9.4), "where `mms model use` downloads it from"
    assert (H2o.family, H2o.served, H2o.in_mms_env, H2o.requirements, H2o.torch_backend) == ("h2o", True, False, (), None)
    assert H2o.default_url == "http://127.0.0.1:8741" and config.loopback(H2o.default_url) == ("127.0.0.1", 8741)


def test_the_start_command_sets_variables_serve_sh_and_the_shim_read(tmp_path):
    start = H2o.start
    assert start.startswith("source ~/h2o-venv/bin/activate && cd {folder} && "), "{folder} is filled in by `mms model prepare`"
    assert start.count("{") == start.count("}") == 1, "{folder} is its one field"
    assert start.format(folder="/mnt/c/models/h2o-lightning-4b").startswith(
        "source ~/h2o-venv/bin/activate && cd /mnt/c/models/h2o-lightning-4b && ")
    assert start.endswith(" bash serve.sh")
    settings = dict(part.split("=", 1) for part in re.findall(r"\b[A-Z_]+=\S+", start))
    assert settings == {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1", "LIGHTNING_IMAGE_URLS": "off",
                        "GPU_UTIL": "0.6", "MAX_LEN": "16384", "TWO_WAY": "0", "VLLM_ARGS": "--max-num-seqs=16"}
    models = Path("C:/models/h2o-lightning-4b")
    if (models / "serve.sh").is_file():  # the owner's download: the names are the ones its script and shim read
        text = (models / "serve.sh").read_text(encoding="utf-8") + (models / "h2o_lightning_shim.py").read_text(encoding="utf-8")
        for name in ("GPU_UTIL", "MAX_LEN", "TWO_WAY", "LIGHTNING_IMAGE_URLS", "VLLM_ARGS"):
            assert name in text, name


# -- the url ----------------------------------------------------------------------------------------------------------

def test_the_server_is_the_default_url_or_the_one_given_and_must_be_loopback(tmp_path):
    folder = model_folder(tmp_path, "h2o-lightning-4b")
    assert H2o(folder).url == "http://127.0.0.1:8741" and H2o(folder).server.port == 8741
    assert H2o(folder).server.hint == "start H2O's serve.sh in WSL (DESIGN.md §13.3)", "the server is told what to say"
    assert H2o(folder, url="http://[::1]:9000").server.host == "::1"
    assert H2o(folder, envs="C:/envs").url == H2o.default_url, "it has no env"
    with pytest.raises(ValueError, match="loopback address"):
        H2o(folder, url="http://192.168.1.5:8741")


# -- the questions and answers ----------------------------------------------------------------------------------------

def test_h2o_is_asked_the_type_the_instructions_and_the_criteria_and_a_box_only_its_instructions(tmp_path):
    made = backend(tmp_path)
    assert made.to_model(made.about, QUESTIONS) == QUESTIONS
    for extra in ({"multiple": True}, {"range": [0, 1]}, {"unit": "s"}, {"step": 1}, {"hit": 0.8}):
        with pytest.raises(ValueError, match=rf"question 'q' has {next(iter(extra))}, which h2o-lightning-4b does not take "
                                             "for a noul question; it takes criteria, instructions, type"):
            made.to_model(made.about, {"q": {"type": "noul", "instructions": "x", **extra}})
    with pytest.raises(ValueError, match="question 'q' has criteria, which h2o-lightning-4b does not take for a box "
                                         "question; it takes instructions, type"):
        made.to_model(made.about, {"q": {"type": "box", "instructions": "the mug", "criteria": {"a": None}}})
    for kind in ("text", "number", "date", "boxes"):
        with pytest.raises(ValueError, match=rf"answers noul, choice, score, box, not '{kind}'"):
            made.to_model(made.about, {"q": {"type": kind, "instructions": "x"}})


@pytest.mark.parametrize("question,words", [
    ({"type": "box", "instructions": "the mug", "criteria": {"a": None}}, "criteria, which h2o-lightning-4b does not take for a box"),
    ({"type": "noul", "instructions": "x", "criteria": {"true": "lit", "dark": "off"}}, "criteria describes true and false only"),
    ({"type": "choice", "instructions": "x", "criteria": {str(n): None for n in range(256)}}, "1 to 255 options"),
    ({"type": "score", "instructions": "x", "criteria": ["only"]}, "at least 2 levels"),
])
def test_a_question_h2o_refuses_is_refused_by_check_naming_it_and_before_the_server_is_asked(tmp_path, shim, fast, question, words):
    made = backend(tmp_path, shim)
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        H2o.check(made.about, {"ok": QUESTIONS["seen"], "q": question})
    with pytest.raises(ValueError, match=rf"question 'q'.*{words}"):
        made.decide(None, {"q": question})
    assert shim.requests == [], "nothing was sent"
    assert H2o.check(made.about, QUESTIONS) is None


def test_h2o_is_set_up_in_wsl_with_the_pins_that_worked_before_it_is_started():
    assert H2o.install.startswith("sudo apt install -y python3-venv python3-dev && "), "Triton needs Python.h to start"
    assert " && python3 -m venv ~/h2o-venv && ~/h2o-venv/bin/pip install --no-cache-dir " in H2o.install
    for pin in ('"vllm==0.30.0"', '"torchcodec==0.17.0"', '"transformers==5.20.2"'):
        assert pin in H2o.install
    assert "~/h2o-venv/bin/activate" in H2o.start, "start runs in the venv install makes"


def test_h2os_answers_read_and_judge_in_mms_terms(ask, tmp_path):
    made = backend(tmp_path)
    got = made.from_model({"seen": NOUL, "count": CHOICE, "legible": SCORE, "where": BOX}, QUESTIONS)
    assert got["seen"] == NOUL and "confidence" not in got["count"]
    assert got["legible"]["score"] == 2.0 and SCORE["score"] == 1.818
    assert verdict(ask, QUESTIONS["seen"], True, ABOUT, got["seen"])[:2] == (True, 0.932)
    assert verdict(ask, QUESTIONS["count"], "six", ABOUT, got["count"]) == (True, 0.798, "six"), \
        "p is the pick's probability, not the normalized maximum of 0.73"
    assert verdict(ask, QUESTIONS["count"], "five", ABOUT, got["count"]) == (False, 0.130, "six")
    assert verdict(ask, QUESTIONS["legible"], 2, ABOUT, got["legible"]) == (True, 1.0, "level 2")
    assert verdict(ask, QUESTIONS["legible"], 3, ABOUT, got["legible"])[0] is False


def test_a_box_answer_is_the_box_when_one_was_found_and_found_is_the_probability_it_is_in_view(ask, tmp_path):
    made = backend(tmp_path)
    where = {"where": QUESTIONS["where"]}
    got = made.from_model({"where": BOX}, where)["where"]
    assert got == {"type": "box", "box": BOX["box"], "found": 0.9, "confidence": 0.8}
    assert verdict(ask, where["where"], True, ABOUT, got) == (True, 0.9, "at top 0.20, left 0.10, bottom 0.70, right 0.60")
    none = made.from_model({"where": NO_BOX}, where)["where"]
    assert none == {"type": "box", "box": None, "found": 0.05, "confidence": 0.0}, "the whole image is not a box"
    assert verdict(ask, where["where"], True, ABOUT, none) == (False, 0.05, "none")
    unseen = {**NO_BOX, "present": 0.7}  # a box not located, though the thing may be in view: it still has no box
    assert verdict(ask, where["where"], True, ABOUT, made.from_model({"where": unseen}, where)["where"])[0] is False
    bare = {key: value for key, value in BOX.items() if key != "present"}  # without the presence read: its own evidence
    assert made.from_model({"where": bare}, where)["where"]["found"] == 0.8
    assert made.from_model({"where": {key: v for key, v in NO_BOX.items() if key != "present"}}, where)["where"]["found"] == 0.0
    with pytest.raises(ValueError, match="its box answer has found 'yes', not true or false"):
        made.from_model({"where": {**BOX, "found": "yes"}}, where)
    with pytest.raises(ValueError, match="gave no answer to question 'where'"):
        made.from_model({}, where)


def test_an_answer_with_an_error_is_a_valueerror_naming_it(tmp_path):
    with pytest.raises(ValueError, match="h2o-lightning-4b: question 'seen': too_long: nope"):
        backend(tmp_path).from_model({"seen": {"type": "noul", "error": "too_long", "message": "nope"}},
                                     {"seen": QUESTIONS["seen"]})


# -- the server -------------------------------------------------------------------------------------------------------

def test_load_waits_while_the_server_warms_up_and_a_wrong_model_is_refused(tmp_path, fast):
    with FakeShim(warming=2) as warming:
        backend(tmp_path, warming).load()
        assert warming.health_asks == 3, "503 twice, then 200"
    with FakeShim(model="h2oai/other") as wrong:
        with pytest.raises(ApiError, match=rf"serves h2oai/other, not {MODEL_ID}"):
            backend(tmp_path, wrong).load()


def test_pictures_go_as_lossless_png_data_uris_and_a_none_state_as_empty(tmp_path, shim):
    made = backend(tmp_path, shim)
    pictures = [noise(640, 360, 1), noise(97, 53, 2), Image.new("RGB", (64, 36), (200, 10, 10))]
    made.decide(None, {"seen": QUESTIONS["seen"]}, pictures)
    (request,) = shim.requests
    assert request["state"] == "" and request["questions"] == {"seen": QUESTIONS["seen"]}
    assert request["images"] == [{"size": list(p.size), "sha256": pixels(p)} for p in pictures], "PNG loses nothing"
    uri = data_uri(pictures[0])
    assert uri.startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as sent:
        assert sent.format == "PNG" and sent.size == (640, 360) and pixels(sent.convert("RGB")) == pixels(pictures[0])
    made.decide("a note", {"seen": QUESTIONS["seen"]})
    assert shim.requests[-1] == {"state": "a note", "questions": {"seen": QUESTIONS["seen"]}, "images": []}


def test_the_server_answers_in_the_cards_shapes_and_mms_reads_each(ask, tmp_path, shim):
    made = backend(tmp_path, shim)
    red = Image.new("RGB", (64, 36), (220, 20, 20))
    questions = {"seen": {"type": "noul", "instructions": "Is it red?"},
                 "count": {"type": "choice", "instructions": "red?", "criteria": {"a": None, "b": None, "none": None}},
                 "legible": {"type": "score", "instructions": "How red?", "criteria": ["no", "some", "very"]},
                 "where": {"type": "box", "instructions": "the red thing"}}
    got = made.decide(None, questions, [red])
    assert verdict(ask, questions["seen"], True, ABOUT, got["seen"])[:2] == (True, 0.95)
    assert verdict(ask, questions["count"], "a", ABOUT, got["count"])[0] is True
    assert verdict(ask, questions["legible"], 2, ABOUT, got["legible"])[0] is True
    assert verdict(ask, questions["where"], True, ABOUT, got["where"])[0] is True
    blue = made.decide(None, questions, [Image.new("RGB", (64, 36), (20, 20, 220))])
    assert verdict(ask, questions["seen"], True, ABOUT, blue["seen"])[0] is False
    assert verdict(ask, questions["legible"], 1, ABOUT, blue["legible"])[0] is False
    assert verdict(ask, questions["where"], True, ABOUT, blue["where"])[0] is False


def test_more_than_four_pictures_are_refused_by_the_server_as_a_valueerror(tmp_path, shim):
    made = backend(tmp_path, shim)
    ask_one = {"seen": QUESTIONS["seen"]}
    made.decide(None, ask_one, [noise(8, 8, n) for n in range(MAX_IMAGES)])
    with pytest.raises(ValueError, match=f"{MAX_IMAGES + 1} images; this model reads at most {MAX_IMAGES}"):
        made.decide(None, ask_one, [noise(8, 8, n) for n in range(MAX_IMAGES + 1)])
    assert len(shim.requests) == 1, "the refused request was not answered"


def test_a_sound_or_a_file_is_not_sent_and_a_server_that_is_gone_is_a_decider_error(tmp_path):
    with FakeShim() as shim:
        made = backend(tmp_path, shim)
        with pytest.raises(ValueError, match="reads pictures, not audio"):
            made.decide(None, {"seen": QUESTIONS["seen"]}, [{"type": "audio", "audio": [0.0], "sampling_rate": 16000}])
        assert shim.requests == []
    with pytest.raises(ApiError, match="h2o-lightning-4b's server at .* failed"):
        made.decide(None, {"seen": QUESTIONS["seen"]})
    made.unload()  # the owner's server and memory: nothing to end


# -- the whole daemon over the fake server ----------------------------------------------------------------------------

def test_the_daemon_asks_over_a_picture_and_a_video_through_the_server_and_finds_sightings(home, tmp_path, served, red_clip,
                                                                                         fast):
    folder = model_folder(tmp_path / "models", "h2o-lightning-4b")
    red = tmp_path / "red.png"
    Image.new("RGB", (64, 36), (220, 20, 20)).save(red)
    with FakeShim(warming=1) as shim:
        write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n"
                           f'[decider]\nmodel = "{folder.as_posix()}"\nurl = "{shim.url}"\nidle_s = 60\n')
        api = served().api
        api.call("remember", path=str(red), source="pictures", t=T0)
        api.call("remember", path=str(red_clip), source="clips", t=T0 + 50)
        result = scan(api, T0 + 100, question="Is it red?")
        assert len(result["sightings"]) == 2 and all(s["p"] == 0.95 for s in result["sightings"]), result
        assert shim.health_asks == 2, "one 503, then 200"
        assert shim.requests and all(len(r["images"]) == 1 and isinstance(r["state"], str) for r in shim.requests)
        sizes = {tuple(image["size"]) for r in shim.requests for image in r["images"]}
        assert (64, 36) in sizes and (160, 90) in sizes, "the picture as it is, and the video's frames"
        assert api.call("status")["decider"]["state"] == "ready"
