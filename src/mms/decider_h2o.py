"""H2O-Lightning-4B as a decider (DESIGN.md §13.3): a server the owner runs (serve.sh in WSL, vLLM and the model's shim) on
the loopback interface, which mms reaches over HTTP. Everything about it is in this file: its facts (`ABOUT`), how a model
folder is recognised (`H2o.claims`), and how mms's questions, pictures and answers go to and from `/v1/systemone`. The
server and its GPU memory are the owner's: mms loads nothing and unloads nothing.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from mms import native
from mms.decider import About
from mms.errors import ApiError
from mms.transport import Served

MODEL_ID = "h2oai/h2o-lightning-4b"  # serve_config.json's model, and what the server's /health names
# The cost of a window, measured (tests/test_gpu.py, 2 Nov, through the server, the median of 5 warm calls with one
# yes/no question): a 640x360 picture 81 ms, a 1080p one 113 ms (cut to its 1.6 MP), 2,000 characters 87 ms. The hit is
# assumed. The shim makes a read for each question.
ABOUT = About(name="h2o-lightning-4b", reads={"video", "image"}, types=("noul", "choice", "score", "box"),
              extracts_from=(), speech=False,
              max_px=1_638_400,  # serve_config.json image.max_pixels
              questions_per_read=1, picture_s=((230_400, 0.081), (1_638_400, 0.113)), char_s=0.087 / 2000,
              question_keys={"criteria"}, notes="h2o.md", measured={"picture_s", "char_s"}, repo=MODEL_ID, repo_gb=9.4)


class H2o:
    """H2O-Lightning-4B through the owner's server. It is started with `start`, in WSL, and stopped by the owner."""

    about = ABOUT
    family = "h2o"
    abouts = (ABOUT,)
    reads_folder = "decider_h2o reads an H2O-Lightning-4B folder (a serve_config.json for h2oai/h2o-lightning-4b)"
    requirements = ()  # its env is the owner's WSL one (vllm 0.30.0, torchcodec 0.17.0, transformers 5.20.2): mms builds none
    torch_backend = None
    served = True
    in_mms_env = False
    default_url = "http://127.0.0.1:8741"
    # Once, in WSL Ubuntu: Python's headers (Triton compiles a helper against them as vLLM starts, which otherwise fails
    # with "Python.h: No such file or directory"), and the venv the server runs in. These pins gave torch 2.13.0+cu130
    # with CUDA available.
    install = ('sudo apt install -y python3-venv python3-dev && python3 -m venv ~/h2o-venv && '
               '~/h2o-venv/bin/pip install --no-cache-dir "vllm==0.30.0" "torchcodec==0.17.0" "transformers==5.20.2"')
    # One-way reading on stock vLLM: no patched files. 60 % of the GPU and a 16,384-token context leave 24 GB room for the
    # desktop. At most 16 reads at once: mms sends one window at a time, and vLLM's default of 256 needs more of the model's
    # Mamba cache than 60 % holds (it refused to start: 247 blocks, 2 Nov). Nothing goes off the machine (no usage
    # statistics, no hub, no image links). Run it in WSL Ubuntu. {folder}: the model folder as WSL sees it; `mms model
    # prepare` fills it in.
    start = ("source ~/h2o-venv/bin/activate && cd {folder} && VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 "
             "HF_HUB_OFFLINE=1 LIGHTNING_IMAGE_URLS=off GPU_UTIL=0.6 MAX_LEN=16384 TWO_WAY=0 VLLM_ARGS=--max-num-seqs=16 "
             "bash serve.sh")

    @staticmethod
    def claims(folder: Path) -> About | None:
        """The About of the H2O model in `folder`: a serve_config.json for it, beside its shim. None when it is not one."""
        folder = Path(folder)
        try:
            served = json.loads((folder / "serve_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (not isinstance(served, dict) or served.get("model") != MODEL_ID
                or not (folder / "h2o_lightning_shim.py").is_file()):
            return None
        return ABOUT

    def __init__(self, folder: str | Path, *, url: str | None = None, envs: str | None = None):
        self.folder = Path(folder).expanduser()
        self.url = url or self.default_url
        self.server = Served(self.url, name=ABOUT.name, model_id=MODEL_ID,
                             hint="start H2O's serve.sh in WSL (DESIGN.md §13.3)")

    def load(self) -> None:
        """Wait for the server to be ready: it says so at /health, and it names the model it serves."""
        self.server.load()

    def decide(self, state, questions: dict, media: list | None = None) -> dict:
        asked = self.to_model(self.about, questions)
        body = {"state": "" if state is None else state, "questions": asked}
        if media:
            body["images"] = [data_uri(medium) for medium in media]  # the server refuses more than 4 (serve_config.json image.max_images)
        reply = self.server.post("/v1/systemone", body)
        if not isinstance(reply, dict) or not isinstance(reply.get("answers"), dict):
            raise ApiError("decider", f"{ABOUT.name}'s server answered no answers: {str(reply)[:200]}")
        return self.from_model(reply["answers"], questions)

    def unload(self) -> None:
        self.server.unload()

    @staticmethod
    def check(about: About, questions: dict) -> None:
        """Refuse, before anything is scanned, a question it cannot take (ValueError, naming it): `to_model`'s."""
        H2o.to_model(about, questions)

    @staticmethod
    def to_model(about: About, questions: dict) -> dict:
        """Its questions: the shared format's (`native.to_model`), and a box question with instructions alone, which name
        the thing to find: the shim takes no criteria for it."""
        return native.to_model(about, questions,
                               takes=lambda question: set() if question["type"] == "box" else about.question_keys)

    def from_model(self, answers: dict, questions: dict) -> dict:
        """mms's answers from its (`native.from_model`), and a box in mms's shape: the box only when one was found, and
        `found` the probability that the thing is in view."""
        return native.from_model(self.about, answers, questions, convert={"box": self._box})

    @staticmethod
    def _box(name: str, answer: dict) -> dict:
        located = answer.get("found")
        if not isinstance(located, bool):
            raise ValueError(f"{ABOUT.name}: question {name!r}: its box answer has found {located!r}, not true or false")
        present = answer.get("present")  # P(the thing is in view), when the server reads it; else its own evidence
        confidence = answer.get("confidence")
        return {"type": "box", "box": answer.get("box") if located else None,
                "found": present if present is not None else (confidence if located else 0.0), "confidence": confidence}


def data_uri(image) -> str:
    """A picture as a PNG data URI, which is lossless; compress_level 1 keeps the encoding fast."""
    if isinstance(image, dict):
        raise ValueError(f"{ABOUT.name} reads pictures, not {image.get('type')}")
    buffer = io.BytesIO()
    image.save(buffer, "PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
