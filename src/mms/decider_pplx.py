"""Perplexity's pplx-decider-flash-v1.2 as a decider (DESIGN.md §13.3), run in a worker process in its own env (the card's
six pins, built by `mms model prepare`) from the `pplx_decider` package the folder ships. Everything about it is in this file
and `worker/pplx.py`: its facts (`ABOUT`), how a model folder is recognised (`Pplx.claims`), and how mms's questions become
its and its answers mms's. Nothing heavy is imported here.
"""

from __future__ import annotations

import json
from pathlib import Path

from mms import modelenv, native
from mms.decider import About
from mms.transport import Worker

TEMPLATE = "joint-question-inline-readout-v1"  # joint_config.json's template (protocol version 3)
FAMILY_FILE = Path(__file__).with_name("worker") / "pplx.py"
# The cost of a window, measured (tests/test_gpu.py, 1 Nov, through the worker, the median of 5 warm calls with one
# yes/no question, on transformers' reference kernels as d3's): a 640x360 picture 276 ms, a 1080p one 602 ms, 2,000
# characters 116 ms. The hit is assumed. The model reads one question a request (the card), so each question is a read of
# the window.
ABOUT = About(name="pplx-decider-flash-v1.2", reads={"video", "image"}, types=("noul", "choice", "score"), extracts_from=(),
              speech=False,
              max_px=1_638_400,  # decision_config.json's max_pixels
              questions_per_read=1, picture_s=((230_400, 0.276), (1_638_400, 0.602)), char_s=0.116 / 2000,
              question_keys={"criteria"}, notes="pplx.md", measured={"picture_s", "char_s"},
              repo="perplexity-ai/pplx-decider-flash-v1.2", repo_gb=9.1)


class Pplx:
    """Perplexity's decider on the GPU, in a worker process: started at the first window, and ended by `unload`."""

    about = ABOUT
    family = "pplx"  # the env's folder name under the envs folder
    abouts = (ABOUT,)
    reads_folder = ("decider_pplx reads a pplx-decider folder (a joint_config.json of version 3 with template "
                    "joint-question-inline-readout-v1, and source/src/pplx_decider)")
    requirements = ("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors==0.8.0", "pillow==12.3.0",
                    "huggingface-hub==1.31.0")  # the card's requirements.txt; its fast path (flash-linear-attention) is off
    torch_backend = "cu130"  # torch 2.14 has no cu128 wheel on Windows; cu130 needs a driver of R580 or newer
    served = False
    in_mms_env = False
    default_url = None
    install = ""  # nothing by hand: mms model prepare builds its env
    start = ""

    @staticmethod
    def claims(folder: Path) -> About | None:
        """The About of the Perplexity model in `folder`: a joint_config.json of protocol version 3, and the package that
        reads it. None when it is not one."""
        folder = Path(folder)
        try:
            joint = json.loads((folder / "joint_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (not isinstance(joint, dict) or joint.get("version") != 3 or joint.get("template") != TEMPLATE
                or not (folder / "source" / "src" / "pplx_decider" / "__init__.py").is_file()):
            return None
        return ABOUT

    def __init__(self, folder: str | Path, *, url: str | None = None, envs: str | None = None):
        if url is not None:
            raise ValueError(f"Perplexity's decider is not served, but [decider] url is {url!r}: remove it")
        self.folder = Path(folder).expanduser()
        python = modelenv.ready_python(self, self.folder, envs, ABOUT.name)
        # ABOUT is static, so the weights must agree with it: read decision_config.json now, on the CPU
        config = self.folder / "decision_config.json"
        recorded = json.loads(config.read_text(encoding="utf-8")).get("max_pixels")
        if recorded != ABOUT.max_px:
            raise ValueError(f"{config} says max_pixels is {recorded!r}, but decider_pplx.ABOUT says max_px is "
                             f"{ABOUT.max_px}: update ABOUT to the weights' numbers, measured again")
        self.worker = Worker(python, FAMILY_FILE, name=ABOUT.name, folder=self.folder)

    def load(self) -> None:
        facts = self.worker.load()
        if facts.get("max_pixels") != ABOUT.max_px:  # the loaded model reads pictures at another size than ABOUT says
            self.worker.unload()
            raise RuntimeError(f"the model loaded with max_pixels {facts.get('max_pixels')!r}, but ABOUT says "
                               f"{ABOUT.max_px}")

    def decide(self, state, questions: dict, media: list | None = None) -> dict:
        asked = self.to_model(self.about, questions)  # before the worker starts: a question it cannot take loads nothing
        return self.from_model(self.worker.decide(state, asked, media or []), questions)

    def unload(self) -> None:
        """End the worker, which frees the GPU."""
        self.worker.unload()

    @staticmethod
    def check(about: About, questions: dict) -> None:
        """Refuse, before anything is scanned, a question it cannot take (ValueError, naming it): `to_model`'s."""
        Pplx.to_model(about, questions)

    @staticmethod
    def to_model(about: About, questions: dict) -> dict:
        """Its questions: the shared format's (`native.to_model`). Its protocol (`pplx_decider.protocol.Encoder.validate`)
        takes a choice of 1 to 255 options and a score of at least two levels, as `native` checks; it takes no more."""
        return native.to_model(about, questions)

    def from_model(self, answers: dict, questions: dict) -> dict:
        """mms's answers from its (`native.from_model`)."""
        return native.from_model(self.about, answers, questions)
