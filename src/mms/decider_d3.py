"""d3 as a decider (DESIGN.md §13.3): d3-edge, d3-mini and d3-flash, vllm-sr's code-readout decision models, run in a worker
process in their own env (transformers 5.17, built by `mms model prepare`) from their own remote code. Everything about d3
is in this file and `worker/d3.py`: its facts (`EDGE`, `MINI`, `FLASH`), how a model folder is recognised (`D3.claims`),
and how mms's questions become d3's and its answers mms's. Nothing heavy is imported here.
"""

from __future__ import annotations

import json
from pathlib import Path

from mms import modelenv, native
from mms.decider import About
from mms.transport import Worker

FORMAT_ID = "d3-code-readout-v1"  # decision_config.json's format_id
FAMILY_FILE = Path(__file__).with_name("worker") / "d3.py"
SCORE_LEVELS = (2, 10)  # d3_runtime.SCORE_LEVELS
# The cost of a window, measured (tests/test_gpu.py, 1 Nov, through the worker, the median of 5 warm calls with one yes/no
# question), for edge, mini and flash: a 640x360 picture 41, 163 and 184 ms, a 1080p one 153, 573 and 1,005 ms, 2,000
# characters 42, 135 and 225 ms. They ran on transformers' reference kernels: the cards' optional fast ones
# (flash-linear-attention, causal-conv1d) need Triton or a CUDA build, with no official Windows wheels. d3-flash peaked at
# 17.5 GiB of the 24. Every hit is assumed.
# d3 reads the window again for each question (a forward each), and cannot hear: it takes pictures and the video's frames.
TIMED = {"picture_s", "char_s"}


def _about(name: str, picture_s: tuple, char_s: float, repo_gb: float, commit: str) -> About:
    return About(name=name, reads={"video", "image"}, types=("noul", "choice", "score"), extracts_from=(), speech=False,
                 max_px=1_638_400,  # a picture is read at up to 1.6 MP (modeling_d3's system_one)
                 questions_per_read=1, picture_s=picture_s, char_s=char_s, question_keys={"criteria"}, notes="d3.md",
                 measured=TIMED, repo=f"vllm-sr/{name}", commit=commit, repo_gb=repo_gb)


EDGE = _about("d3-edge", ((230_400, 0.041), (1_638_400, 0.153)), 0.042 / 2000, 1.2,
              "c454ee39b862248d70abd7f84195d2f54ffe6d43")
MINI = _about("d3-mini", ((230_400, 0.163), (1_638_400, 0.573)), 0.135 / 2000, 9.1,
              "df99bf4ca4418bc6908ee68e9b7479af1f3b8800")
FLASH = _about("d3-flash", ((230_400, 0.184), (1_638_400, 1.005)), 0.225 / 2000, 16.8,
               "13dc1cc8ef23caefa765907e1d764751e5629f7d")
BY_NAME = {about.name: about for about in (EDGE, MINI, FLASH)}


class D3:
    """d3 on the GPU, in a worker process: started at the first window, and ended by `unload`."""

    family = "d3"  # the env's folder name under the envs folder
    abouts = (EDGE, MINI, FLASH)
    reads_folder = ("decider_d3 reads a d3 folder (a decision_config.json with format_id d3-code-readout-v1, and a "
                    "MODEL_MANIFEST.json naming d3-edge, d3-mini or d3-flash)")
    requirements = ("torch==2.11.0", "torchvision==0.26.0", "transformers==5.17.0", "accelerate>=1.10", "safetensors>=0.6",
                    "huggingface_hub>=1.0", "einops", "pillow>=10", "opencv-python-headless", "numpy")
    # no flash-linear-attention: it needs triton, which Windows lacks, and d3 then runs Qwen3.5's linear layers in PyTorch
    torch_backend = "cu128"
    served = False
    in_mms_env = False
    default_url = None
    install = ""  # nothing by hand: mms model prepare builds its env
    start = ""

    @staticmethod
    def claims(folder: Path) -> About | None:
        """The About of the d3 model in `folder`: its decision_config.json is d3's format and its manifest names the
        variant. None when it is not one, or the variant is not known."""
        try:
            decision = json.loads((Path(folder) / "decision_config.json").read_text(encoding="utf-8"))
            manifest = json.loads((Path(folder) / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(decision, dict) or decision.get("format_id") != FORMAT_ID or not isinstance(manifest, dict):
            return None
        name = manifest.get("model_name")
        return BY_NAME.get(name) if isinstance(name, str) else None

    def __init__(self, folder: str | Path, *, url: str | None = None, envs: str | None = None):
        if url is not None:
            raise ValueError(f"d3 is not served, but [decider] url is {url!r}: remove it")
        self.folder = Path(folder).expanduser()
        self.about = self.claims(self.folder)
        if self.about is None:
            raise ValueError(f"{self.folder} is not a d3 model: {self.reads_folder}")
        self.worker = Worker(modelenv.ready_python(self, self.folder, envs, self.about.name), FAMILY_FILE,
                             name=self.about.name, folder=self.folder)

    def load(self) -> None:
        facts = self.worker.load()
        if facts.get("model_name") != self.about.name:  # the weights are another variant than the folder's manifest says
            self.worker.unload()
            raise RuntimeError(f"{self.folder} loaded as {facts.get('model_name')!r}, but it is {self.about.name}")

    def decide(self, state, questions: dict, media: list | None = None) -> dict:
        asked = self.to_model(self.about, questions)  # before the worker starts: a question d3 cannot take loads nothing
        return self.from_model(self.worker.decide(state, asked, media or []), questions)

    def unload(self) -> None:
        """End the worker, which frees the GPU."""
        self.worker.unload()

    @staticmethod
    def check(about: About, questions: dict) -> None:
        """Refuse, before anything is scanned, a question d3 cannot take (ValueError, naming it): `to_model`'s."""
        D3.to_model(about, questions)

    @staticmethod
    def to_model(about: About, questions: dict) -> dict:
        """d3's questions: the shared format's (`native.to_model`) and d3's own rules, from `d3_runtime.normalize_question`:
        a choice's option names are not empty, and a score has 2 to 10 levels, distinct and not empty. d3 answers any other
        question with an `invalid_question` error, which would fail every window and not the scan."""
        asked = native.to_model(about, questions)
        for name, question in asked.items():
            criteria = question.get("criteria")
            if question["type"] == "choice" and not all(criteria):
                raise ValueError(f"question {name!r}: d3 does not take an empty option name in a choice")
            if question["type"] == "score":
                levels = [level if isinstance(level, str) else json.dumps(level, ensure_ascii=False) for level in criteria]
                if len(levels) > SCORE_LEVELS[1] or not all(levels) or len(set(levels)) != len(levels):
                    raise ValueError(f"question {name!r}: d3 takes {SCORE_LEVELS[0]} to {SCORE_LEVELS[1]} levels in a score, "
                                     f"distinct and not empty, lowest first, not {criteria!r}")
        return asked

    def from_model(self, answers: dict, questions: dict) -> dict:
        """mms's answers from d3's (`native.from_model`)."""
        return native.from_model(self.about, answers, questions)
