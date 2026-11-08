"""The real models on the GPU (TESTBED.md, GPU tests): one test module run for every model folder in `MMS_MODELS`
(required: with it unset no model is tested) that a backend claims, with one model loaded at a time. Each must answer a
picture, a text and (where it hears) beeps on the right side of 0.5, answer every type it takes in a shape `ask.read` parses,
and give its memory back when it is unloaded. d1-Omni-2B must also answer bit-identically through its worker and in a plain
subprocess that runs the same child code (the old in-process path is gone).
Opt-in, alone on the GPU: `uv run pytest -m gpu -n 0 -s tests/test_gpu.py -k d1-Omni-2B` (the timings print with -s).

A model that is not ready is skipped, with what to do: weights still downloading, its env not prepared (`mms model
prepare <name>`), or H2O's server not answering (serve.sh in WSL). Models that no backend claims are not tested."""

from __future__ import annotations

import http.client
import json
import os
import pickle
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mms import config, decider, decider_d1, modelenv, transport
from mms.fetch import unfinished
from mms.scripts import ask

MODELS = Path(os.environ["MMS_MODELS"]) if os.environ.get("MMS_MODELS") else None
UNLOAD_SLACK_MB = 300  # what a freed GPU may still hold of what the model took
pytestmark = [pytest.mark.gpu, pytest.mark.xdist_group("gpu")]  # one worker, so one copy of the model


# -- which models, and whether each is ready ----------------------------------------------------------------------------

def claimed_folders() -> list[Path]:
    """The subfolders of MODELS that a backend claims, by their config files alone."""
    if MODELS is None or not MODELS.is_dir():
        return []
    return sorted(folder for folder in MODELS.iterdir() if folder.is_dir() and decider.about_of(str(folder)) is not None)


def backend_of(folder: Path):
    return decider.claim(folder)[0]


def serving(url: str) -> bool:
    """Whether a model server answers 200 at /health."""
    try:
        connection = http.client.HTTPConnection(*config.loopback(url), timeout=5)
        try:
            connection.request("GET", "/health")
            return connection.getresponse().status == 200
        finally:
            connection.close()
    except (OSError, http.client.HTTPException):
        return False


def vram_used_mb() -> int | None:
    """The GPU's used memory in MiB by nvidia-smi, or None where there is none."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    done = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                          text=True, timeout=30)
    return int(done.stdout.split()[0]) if done.returncode == 0 and done.stdout.split() else None


@dataclass
class Loaded:
    """A model that is loaded: its backend, and what was measured while loading it."""

    folder: Path
    backend: type
    decider: object
    load_s: float
    vram_before_mb: int | None

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def about(self):
        return self.decider.about

    @property
    def asserts(self) -> bool:
        """Whether this model's answers are held to the checks: d1-Omni-2B's are; the others' are bake-off data."""
        return self.about is decider_d1.ABOUT

    def decide(self, state, questions: dict, media=None) -> dict:
        return self.decider.decide(state, questions, media or [])


@pytest.fixture(scope="module", params=[pytest.param(folder, id=folder.name) for folder in claimed_folders()])
def folder(request) -> Path:
    return request.param


@pytest.fixture(scope="module")
def made(folder):
    """The model's decider, constructed (cheap: its facts and checks) but not loaded. Skipped when its env is not prepared;
    d1's is mms's own, so a lack there fails."""
    backend = backend_of(folder)
    problem = modelenv.problem(backend, folder, None)
    if problem and backend.in_mms_env:
        pytest.fail(problem)
    if problem:
        pytest.skip(f"{folder.name}'s env is not prepared ({problem}): run mms model prepare {folder.name}")
    return decider.load_decider(str(folder))


@pytest.fixture(scope="module")
def model(folder, made):
    """The model loaded, with how long that took, and unloaded at the end. Skipped when the weights are not all there or
    its server is not up."""
    backend = type(made)
    if backend.served:
        if not serving(backend.default_url):
            pytest.skip(f"{folder.name} is not serving at {backend.default_url}: start serve.sh in WSL (DESIGN.md §13.3)")
    elif (why := unfinished(folder)) is not None:
        pytest.skip(f"{folder.name}'s weights are not all there: {why}")
    before = vram_used_mb()
    began = time.perf_counter()
    made.load()
    loaded = Loaded(folder, backend, made, time.perf_counter() - began, before)
    print(f"\n{loaded.name} load and warm: {loaded.load_s:.1f} s")
    yield loaded
    made.unload()


# -- the inputs ---------------------------------------------------------------------------------------------------------

def yes(model: Loaded, instructions: str, state=None, media=None) -> float:
    answer = model.decide(state, {"q": {"type": "noul", "instructions": instructions}}, media)["q"]
    return answer["noul"]


def picture(colour: str | None, size=(640, 360)) -> Image.Image:
    """White, with a square of `colour` in the middle (or nothing)."""
    image = Image.new("RGB", size, "white")
    if colour:
        w, h = size
        ImageDraw.Draw(image).rectangle([w // 2 - h // 6, h // 3, w // 2 + h // 6, 2 * h // 3], fill=colour)
    return image


def tone(seconds: float, hz: float | None = 440.0) -> dict:
    n = int(seconds * 16000)
    wave = 0.5 * np.sin(2 * np.pi * (hz or 0) * np.arange(n) / 16000) if hz else np.zeros(n)
    return {"type": "audio", "audio": wave.astype(np.float32), "sampling_rate": 16000}


def beeps(count: int = 3, seconds: float = 2.0) -> dict:
    """`count` 0.2 s beeps at 1 kHz, as in d1's README. A steady sine is no sound event to d1: on 31 Oct it put a 2 s
    440 Hz tone at 0.39-0.50 on "a beep?", "any sound?" and "silent?" alike, while it heard and counted beeps."""
    wave = np.zeros(int(seconds * 16000), dtype=np.float32)
    for i in range(count):
        start = int((0.2 + 0.6 * i) * 16000)
        wave[start:start + 3200] = tone(0.2, 1000)["audio"]
    return {"type": "audio", "audio": wave, "sampling_rate": 16000}


def hears(model: Loaded) -> None:
    if "audio" not in model.about.reads:
        pytest.skip(f"{model.about.name} does not hear")


KETTLE = "Kitchen log. 07:15 the kettle is boiling. 07:20 someone put out the bins."

# one question of each type, with what it is asked about: (state, question, a picture's colour or None for no picture)
TYPED = {
    "noul": (None, {"type": "noul", "instructions": "Is there a red square in the picture?"}, "red"),
    "choice": ("The kettle is boiling.", {"type": "choice", "instructions": "What is boiling?",
                                          "criteria": {"kettle": None, "soup": None}}, None),
    "score": ("The kettle is boiling.", {"type": "score", "instructions": "How hot is it?",
                                         "criteria": ["cold", "warm", "hot"]}, None),
    "text": ("The dentist appointment is on Thursday at 9:30.",
             {"type": "text", "instructions": "When is the dentist appointment?"}, None),
    "number": ("Three mugs and two plates are on the table.",
               {"type": "number", "instructions": "How many mugs are on the table?"}, None),
    "date": ("The invoice is due on 5 November 2026.", {"type": "date", "instructions": "When is the invoice due?"}, None),
    "box": (None, {"type": "box", "instructions": "Where is the red square?"}, "red"),
    "boxes": (None, {"type": "boxes", "instructions": "Where are the squares?"}, "red"),
}


# -- the checks ---------------------------------------------------------------------------------------------------------

def test_the_real_model_matches_the_backends_facts(folder, made):
    """The CPU part of the check that a backend's About is the weights' own numbers: the folder is claimed by the backend
    that made `made`, whose constructor has run its own checks, and for d1 config.json agrees with ABOUT. Loads nothing."""
    about = decider.about_of(str(folder))
    assert about == made.about, "the facts a spec gives without a backend are the facts of the backend made from it"
    if made.family != "d1":
        return
    recorded = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert recorded["d1_vision_max_pixels"] == about.max_px, "d1's own picture cap"
    assert recorded["d1_set_size"] == about.questions_per_read, "d1 reads this many questions in a pass"
    assert ("audio" in about.reads) == (recorded["model_type"] == "lfm3_omni"), "only d1-Omni hears"
    assert (about.measured == {"picture_s", "sound_s", "char_s", "hit"}) == (about is decider_d1.ABOUT)


def test_load_time(model):
    print(f"\n{model.name} load and warm (a fresh worker, torch and the weights): {model.load_s:.1f} s")
    assert model.load_s < transport.LOAD_S


def test_a_red_square_is_seen_and_its_absence_too(model):
    question = "Is there a red square in the picture?"
    assert yes(model, question, media=[picture("red")]) > 0.5
    assert yes(model, question, media=[picture(None)]) < 0.5
    assert yes(model, question, media=[picture("blue")]) < 0.5


def test_beeps_are_heard_and_counted_and_silence_is_not(model):
    hears(model)
    assert yes(model, "Is there a beep?", media=[beeps()]) > 0.5
    assert yes(model, "Is there a beep?", media=[tone(2, None)]) < 0.5
    counted = model.decide(None, {"n": {"type": "number", "instructions": "How many beeps are there?", "range": [0, 10],
                                        "step": 1}}, [beeps()])["n"]["number"]
    assert counted == 3


def test_a_text_is_read(model):
    assert yes(model, "Is the kettle boiling?", KETTLE) > 0.5
    assert yes(model, "Is the dishwasher running?", KETTLE) < 0.5


def test_choice_and_score_answers_have_their_shapes(model):
    answers = model.decide("The kettle is boiling.", {
        "choice": {"type": "choice", "instructions": "What is boiling?", "criteria": {"kettle": None, "soup": None}},
        "score": {"type": "score", "instructions": "How hot is it?", "criteria": ["cold", "warm", "hot"]},
    })
    assert answers["choice"]["choice"] == "kettle" and set(answers["choice"]["probabilities"]) == {"kettle", "soup"}
    assert answers["score"]["score"] >= 1


def test_every_type_the_model_takes_is_answered_in_a_shape_ask_reads(model):
    """Each type it takes, asked alone, comes back as something `ask.read` parses (what ask does with every answer)."""
    for kind in model.about.types:
        state, question, colour = TYPED[kind]
        answers = model.decide(state, {"q": question}, [picture(colour)] if colour else [])
        read = ask.read("q", question, answers["q"])
        print(f"{model.name} {kind}: {answers['q']} -> {read}")
        if kind == "noul":
            assert read["p"] > 0.5
        elif kind == "choice":
            assert read["picked"] == ["kettle"]
        elif kind == "box":
            assert read["edges"] and read["found"] > 0.5, f"the red square is not boxed: {answers['q']}"


def test_a_box_is_drawn_round_the_square(model):
    if "box" not in model.about.types:
        pytest.skip(f"{model.about.name} answers no box")
    answer = model.decide(None, {"q": {"type": "box", "instructions": "Where is the red square?"}}, [picture("red")])["q"]
    box = answer["box"]
    print(f"{model.name} box: {answer}")
    assert box is not None and {"top", "bottom", "left", "right"} <= set(box)
    assert box["left"] < box["right"] and box["top"] < box["bottom"]


def timed(model: Loaded, state, media, questions: int = 1, repeat: int = 5) -> list[float]:
    asked = {f"q{i}": {"type": "noul", "instructions": text} for i, text in enumerate((
        "Is there a kettle, a red square or a tone?", "Is the light on?", "Is there a person?", "Is there a door?"))}
    asked = dict(list(asked.items())[:questions])
    times = []
    for _ in range(repeat):
        began = time.perf_counter()
        model.decide(state, asked, media)
        times.append(time.perf_counter() - began)
    return times


def test_timings(model):
    """Seconds a decision takes, as the median of five warm calls (printed, and asserted only to be sane). A case that
    fails is reported with the rest, then fails the test, so the other timings are not lost."""
    text = ("The kettle is boiling in the kitchen. " * 60)[:2000]
    cases = {}
    for width, height, label in ((640, 360, "640x360"), (1280, 720, "1280x720"), (1920, 1080, "1080p")):
        for questions in (1, 4):
            cases[f"{label} frame, {questions} question{'s' * (questions > 1)}"] = (None, [picture("red", (width, height))],
                                                                                    questions)
    cases["2,000 characters"] = (text, [], 1)
    if "audio" in model.about.reads:
        cases["5 s sound"] = (None, [tone(5)], 1)
    medians, failed = {}, {}
    for name, (state, media, questions) in cases.items():
        try:
            times = timed(model, state, media, questions)
        except Exception as error:  # printed with the rest
            failed[name] = f"{type(error).__name__}: {error}"
            print(f"{model.name} {name}: FAILED {failed[name]}")
            continue
        medians[name] = statistics.median(times)
        print(f"{model.name} {name}: {medians[name] * 1000:.0f} ms (min {min(times) * 1000:.0f}, max {max(times) * 1000:.0f})")
    about = model.about
    if "640x360 frame, 1 question" in medians and "1080p frame, 1 question" in medians:
        print(f"{model.name} About numbers to set from these: picture_s=((230_400, {medians['640x360 frame, 1 question']:.3f}), "
              f"(1_638_400, {medians['1080p frame, 1 question']:.3f})), char_s={medians.get('2,000 characters', 0) / 2000:.2e}, "
              f"sound_s={medians.get('5 s sound', 0):.3f}; now picture_s={about.picture_s}, char_s={about.char_s:.2e}, "
              f"sound_s={about.sound_s}")
        small = medians["640x360 frame, 1 question"]
        if model.asserts:  # DESIGN.md §13.3: a frame through the worker against the 53 ms measured in process
            print(f"{model.name} 640x360 through the worker: {small * 1000:.0f} ms against {about.picture_seconds(230_400) * 1000:.0f} "
                  f"ms in process ({small / about.picture_seconds(230_400) - 1:+.1%}; the aim is +5% or less)")
    assert not failed, failed
    assert all(median < 10 for median in medians.values()), medians


def test_peak_memory(model):
    """The worker's peak GPU memory, from torch inside it (a served model has no worker of ours to ask)."""
    worker = getattr(model.decider, "worker", None)
    if worker is None:
        pytest.skip(f"{model.name} is served: its memory is the server's")
    stats = worker.stats()
    peak = stats["max_memory_allocated"]
    print(f"\n{model.name} peak GPU memory in the worker: {peak / 2**30:.2f} GiB (now {stats['memory_allocated'] / 2**30:.2f})")
    assert peak and peak > 0


# the same child code in this process's worker and in a plain process

def identity_cases() -> list[tuple]:
    """(state, questions, media) for the synthetic cases of this module, the same on both sides of the check."""
    def one(text: str, kind: str = "noul", **more) -> dict:
        return {"q": {"type": kind, "instructions": text, **more}}

    square = "Is there a red square in the picture?"
    four = {f"q{i}": {"type": "noul", "instructions": text} for i, text in enumerate((
        square, "Is the light on?", "Is there a person?", "Is there a blue square in the picture?"))}
    return [
        (None, one(square), [picture("red")]), (None, one(square), [picture(None)]), (None, one(square), [picture("blue")]),
        (None, four, [picture("red", (1280, 720))]),
        (None, one("Where is the red square?", "box"), [picture("red")]),
        (KETTLE, one("Is the kettle boiling?"), []), (KETTLE, one("Is the dishwasher running?"), []),
        (KETTLE, {"choice": {"type": "choice", "instructions": "What is boiling?", "criteria": {"kettle": None, "soup": None}},
                  "score": {"type": "score", "instructions": "How hot is it?", "criteria": ["cold", "warm", "hot"]}}, []),
        (None, one("Is there a beep?"), [beeps()]), (None, one("Is there a beep?"), [tone(2, None)]),
        (None, one("How many beeps are there?", "number", range=[0, 10], step=1), [beeps()]),
    ]


IN_PROCESS = '''
import importlib.util, json, pickle, sys

spec = importlib.util.spec_from_file_location("d1_in_process", sys.argv[1])
d1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1)
with open(sys.argv[3], "rb") as file:
    cases = pickle.load(file)
model, facts = d1.load(sys.argv[2], {"warm_sound": True})
with open(sys.argv[4], "w", encoding="utf-8") as file:
    json.dump([d1.decide(model, state, questions, media) for state, questions, media in cases], file)
'''


def test_d1_answers_are_bit_identical_in_a_plain_process_and_through_the_worker(model, tmp_path):
    """The old in-process path is gone (DESIGN.md §13.3), so this compares the worker with a plain subprocess that runs the
    same child code: `worker/d1.py`'s `load` and `decide`, loaded by path with no wire, must give exactly the answers the
    worker gives for the same inputs. It shows the wire changes nothing, not that the worker equals the old in-process d1.
    (Both copies of the model are resident for this test.)"""
    if not model.asserts:
        pytest.skip("the check is for d1-Omni-2B")
    cases = identity_cases()
    through = [model.decide(*case) for case in cases]
    (tmp_path / "cases.pkl").write_bytes(pickle.dumps(cases))
    done = subprocess.run([str(modelenv.mms_python()), "-X", "utf8", "-c", IN_PROCESS, str(decider_d1.WORKER),
                           str(model.folder), str(tmp_path / "cases.pkl"), str(tmp_path / "answers.json")],
                          capture_output=True, text=True, encoding="utf-8", timeout=900)
    assert done.returncode == 0, f"the plain process exited {done.returncode}\n{done.stdout}\n{done.stderr[-3000:]}"
    plain = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    differ = [i for i, (a, b) in enumerate(zip(through, plain)) if a != b]
    for i in differ:
        print(f"case {i}: through the worker {through[i]}\n         in a plain process {plain[i]}")
    assert len(through) == len(plain) == len(cases) and not differ, \
        f"cases {differ} differ between the worker and a plain process"


def test_unloading_gives_the_gpu_its_memory_back(model):
    """Ends the worker and checks nvidia-smi's used memory is back within 300 MB of what it was before the load. Last: the
    model is gone after it."""
    if model.backend.served:
        pytest.skip(f"{model.name} is served: unloading leaves its server and memory to the owner")
    if model.vram_before_mb is None:
        pytest.skip("nvidia-smi gave no memory figure")
    model.decider.unload()
    end = time.monotonic() + 20
    used = vram_used_mb()
    while used is not None and used > model.vram_before_mb + UNLOAD_SLACK_MB and time.monotonic() < end:
        time.sleep(1)  # the driver gives the memory back a moment after the process ends
        used = vram_used_mb()
    assert used is not None, "nvidia-smi gave no memory figure after the unload"
    print(f"\n{model.name} GPU memory used: {model.vram_before_mb} MiB before the load, {used} MiB after the unload")
    assert used <= model.vram_before_mb + UNLOAD_SLACK_MB
