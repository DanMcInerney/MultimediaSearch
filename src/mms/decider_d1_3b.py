"""d1-3B as a decider (DESIGN.md §10, §13.3): Liquid AI's d1 on LFM2.5-VL-3B, run through transformers from its own code,
offline, in bf16 on the GPU, in a worker process of mms's own env (`worker/d1_3b.py` is the half that holds the model). It
reads pictures and video frames, hears nothing, and answers noul, choice and score. Everything about it on this side is in
this file: its facts (`ABOUT`), how a model folder is recognised (`D1_3B.claims`), and how mms's questions become its and its
answers mms's. Nothing heavy is imported here.
"""

from __future__ import annotations

import json
from pathlib import Path

from mms import modelenv, native, transport
from mms.decider import About
from mms.decider_d1 import D1

WORKER = Path(__file__).with_name("worker") / "d1_3b.py"
SCORE_LEVELS = (2, 10)  # the readout is defined for 2 to 10 levels (prompt.readout_ids, README)

# Its code (runner.py) reads a picture at up to 1,048,576 pixels (VISION_MAX_PIXELS), and the worker checks the loaded code
# for that number. It reads the state once and every question as a branch of one tree (hybrid.py), so there is no set size:
# any number of questions is one pass.
# The cost of a window, measured (tests/test_gpu.py, 2 Nov, through the worker, the median of 5 warm calls with one yes/no
# question): a 640x360 picture 51 ms, 1280x720 153 ms, a 1080p one (cut to the cap) 177 ms, 2,000 characters 50 ms. Four
# questions cost about one (65 ms at 640x360). Its code applies no calibration (modeling_d1 passes none), so the hit is
# assumed.
ABOUT = About(name="d1-3B", reads={"video", "image"}, types=("noul", "choice", "score"), extracts_from=(), speech=False,
              question_keys={"criteria"}, max_px=1_048_576, questions_per_read=0,
              picture_s=((230_400, 0.051), (1_048_576, 0.177)), char_s=0.050 / 2000, hit=0.8, notes="d1_3b.md",
              measured={"picture_s", "char_s"}, repo="LiquidAI/d1-3B", commit="051bcc464b01b9f92942b364d9586b0ef5912432",
              repo_gb=6.3)
VARIANTS = {(2048, 30): ABOUT}  # config.json text_config (hidden_size, num_hidden_layers) -> the model's facts


class D1_3B:
    """d1-3B on the GPU, in a worker process: started and warmed by `load`, so a script's first call is a normal one, and
    ended by `unload`, which frees all its memory."""

    family = "d1_3b"
    abouts = (ABOUT,)
    reads_folder = ("decider_d1_3b reads a d1-3B folder (a config.json with model_type lfm2_vl, auto_map AutoModel "
                    "modeling_d1.D1Model, and LFM2.5-VL-3B's text_config)")
    requirements = D1.requirements  # it runs in d1's env: mms's own
    torch_backend = None
    served = False
    in_mms_env = True
    default_url = None
    install = D1.install
    start = ""  # d1-3B is started by mms, not by the owner

    @staticmethod
    def claims(folder: Path) -> About | None:
        """The About of the d1-3B model in `folder`: its config.json is an LFM2-VL one whose AutoModel is d1's, and whose
        text model has LFM2.5-VL-3B's size. None when it is not one."""
        try:
            recorded = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(recorded, dict) or recorded.get("model_type") != "lfm2_vl":
            return None
        auto, text = recorded.get("auto_map"), recorded.get("text_config")
        if not isinstance(auto, dict) or auto.get("AutoModel") != "modeling_d1.D1Model" or not isinstance(text, dict):
            return None
        size = (text.get("hidden_size"), text.get("num_hidden_layers"))
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in size):
            return None
        return VARIANTS.get(size)

    @staticmethod
    def check(about: About, questions: dict) -> None:
        """Refuse, before anything is scanned, a question it cannot take (ValueError, naming it): `to_model`'s."""
        D1_3B.to_model(about, questions)

    @staticmethod
    def to_model(about: About, questions: dict) -> dict:
        """Its questions: the shared format's (`native.to_model`), and a score of at most 10 levels."""
        asked = native.to_model(about, questions)
        for name, question in asked.items():
            if question["type"] == "score" and len(question["criteria"]) > SCORE_LEVELS[1]:
                raise ValueError(f"question {name!r}: {about.name} takes {SCORE_LEVELS[0]} to {SCORE_LEVELS[1]} levels "
                                 f"in a score, not {len(question['criteria'])}")
        return asked

    def __init__(self, folder: str | Path, *, url: str | None = None, envs: str | None = None):
        if url is not None:
            raise ValueError(f"d1-3B is not served, but [decider] url is {url!r}: remove it")
        self.folder = Path(folder).expanduser()
        self.about = self.claims(self.folder)
        if self.about is None:
            raise ValueError(f"{self.folder} is not a d1-3B model: {self.reads_folder}")
        self.worker = transport.Worker(modelenv.mms_python(), WORKER, name=self.about.name, folder=self.folder, options={})

    def load(self) -> None:
        """Start the worker: it reads the weights onto the GPU, checks them and warms up. The picture cap it reports is the
        code's, and must be the one `ABOUT` says."""
        facts = self.worker.load()
        if facts.get("max_pixels") != self.about.max_px:
            self.worker.unload()
            raise RuntimeError(f"{self.folder}'s code caps pictures at {facts.get('max_pixels')!r} pixels "
                               f"(runner.VISION_MAX_PIXELS), but decider_d1_3b's facts say max_px is {self.about.max_px}: "
                               "update them to the code's number, measured again")

    def decide(self, state, questions: dict, media: list | None = None) -> dict:
        asked = self.to_model(self.about, questions)  # before the worker starts: a question it cannot take loads nothing
        return native.from_model(self.about, self.worker.decide(state, asked, media or []), questions)

    def unload(self) -> None:
        """End the worker, which frees the GPU."""
        self.worker.unload()
