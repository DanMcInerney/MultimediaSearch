"""d1 as a decider (DESIGN.md §10, §13.3): d1-Omni-2B and d1-2B, run through transformers from their own code, offline,
in bf16 on the GPU, in a worker process of mms's own env (`worker/d1.py` is the half that holds the model). Everything about
d1 on this side is in this file: its facts (`ABOUT`, `ABOUT_2B`), how a model folder is recognised (`D1.claims`), and how
the worker is started. Nothing heavy is imported here.
"""

from __future__ import annotations

import json
from pathlib import Path

from mms import modelenv, transport
from mms.decider import MEDIA_KINDS, TYPES, About

# the keys a d1 model's config.json holds (the worker checks the loaded model against their values)
D1_KEYS = ("d1_attention", "d1_set_size", "d1_vision_max_pixels", "d1_numbers", "d1_codes")
WORKER = Path(__file__).with_name("worker") / "d1.py"

# The cost of a window. These are assumptions to change with the weights.
# Measured (tests/test_gpu.py::test_timings, 2026-10-31, the median of 5 warm calls with one yes/no question): a 640x360
# picture 53 ms, a 1080p one (d1 caps it at 1.6 MP) 489 ms, a 5 s tone 45 ms, 2,000 characters 44 ms. Measured by
# a separate GPU probe with 4 questions: 1280x720 293 ms (DESIGN.md §16).
# The picture's two points are the two measured sizes, 220 and 1,590 tokens at 1,047 pixels a token; the cost between
# them is extrapolated linearly, which 1280x720 fits. The text cost is assumed linear in characters from the one
# measurement, at 2,000; none above it exists. The hit is 0.8 on d1-Omni-2B's own calibration (temperature 3.38).
ABOUT = About(name="d1-Omni-2B", reads=MEDIA_KINDS, files=(), types=TYPES, extracts_from={"text", "records"},
              speech=False, audio_rate=16000,
              max_px=1_638_400,  # d1's own cap (config.json d1_vision_max_pixels)
              questions_per_read=16,  # d1 reads the window again for each 16 questions (d1_set_size)
              picture_s=((230_400, 0.053), (1_638_400, 0.4834)),
              sound_s=0.05, char_s=0.044 / 2000, hit=0.8, notes="d1.md",
              measured={"picture_s", "sound_s", "char_s", "hit"}, repo="LiquidAI/d1-Omni-2B", repo_gb=4.4, profile={})
# d1-2B has d1-Omni-2B's config keys and no audio tower. Measured (tests/test_gpu.py, 1 Nov, through the worker, the
# median of 5 warm calls with one yes/no question): a 640x360 picture 53 ms, a 1080p one 487 ms, 2,000 characters 47 ms.
# Its hit is assumed to be d1-Omni-2B's (its calibration temperature is 1.34, so 0.8 is not the same gap there).
ABOUT_2B = About(name="d1-2B", reads={"video", "image"}, files=(), types=TYPES, extracts_from={"text", "records"},
                 speech=False, max_px=1_638_400, questions_per_read=16,
                 picture_s=((230_400, 0.053), (1_638_400, 0.487)),
                 sound_s=0.05, char_s=0.047 / 2000, hit=0.8, notes="d1.md", measured={"picture_s", "char_s"},
                 repo="LiquidAI/d1-2B", repo_gb=3.9, profile={})
VARIANTS = {"lfm3_omni": ABOUT, "lfm3_vl": ABOUT_2B}  # config.json model_type -> the model's facts


class D1:
    """d1 on the GPU, in a worker process: started and warmed by `load`, so a script's first call is a normal one, and
    ended by `unload`, which frees all its memory."""

    family = "d1"
    abouts = (ABOUT, ABOUT_2B)
    reads_folder = ("decider_d1 reads a d1-Omni-2B or d1-2B folder (a config.json with the d1 keys, d1_vision_max_pixels "
                    "and the others, and a model_type of lfm3_omni or lfm3_vl)")
    requirements = ("torch", "torchvision", "transformers==5.21.0")  # pyproject.toml's `d1` group: mms's own env holds them
    torch_backend = None
    served = False
    in_mms_env = True
    default_url = None
    install = "uv sync --group d1"  # what the owner runs in mms's project when its env lacks them
    start = ""  # d1 is started by mms, not by the owner

    @staticmethod
    def claims(folder: Path) -> About | None:
        """The About of the d1 model in `folder`: its config.json holds every d1 key, and its model_type says which
        d1 (both have the same keys). None when it is not one."""
        try:
            recorded = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(recorded, dict) or not all(key in recorded for key in D1_KEYS):
            return None
        model_type = recorded.get("model_type")
        return VARIANTS.get(model_type) if isinstance(model_type, str) else None

    @staticmethod
    def check(about: About, questions: dict) -> None:
        """d1 takes every question in mms's format (ask has refused the others): nothing more to refuse."""

    def __init__(self, folder: str | Path, *, url: str | None = None, envs: str | None = None):
        if url is not None:
            raise ValueError(f"d1 is not served, but [decider] url is {url!r}: remove it")
        self.folder = Path(folder).expanduser()
        self.about = self.claims(self.folder)
        if self.about is None:
            raise ValueError(f"{self.folder} is not a d1 model: {self.reads_folder}")
        # the facts are static, so the weights must agree with them: read config.json now, on the CPU
        recorded = json.loads((self.folder / "config.json").read_text(encoding="utf-8"))
        for key, fact, value in (("d1_vision_max_pixels", "max_px", self.about.max_px),
                                 ("d1_set_size", "questions_per_read", self.about.questions_per_read)):
            if recorded.get(key) != value:
                raise ValueError(f"{self.folder / 'config.json'} says {key} is {recorded.get(key)!r}, but "
                                 f"decider_d1's facts for {self.about.name} say {fact} is {value}: update them to "
                                 "the weights' numbers, measured again")
        self.worker = transport.Worker(modelenv.mms_python(), WORKER, name=self.about.name, folder=self.folder,
                                       options={"warm_sound": "audio" in self.about.reads})

    def load(self) -> None:
        """Start the worker: it reads the weights onto the GPU, checks d1's config and warms up."""
        self.worker.load()

    def decide(self, state, questions: dict, media: list | None = None) -> dict:
        return self.worker.decide(state, questions, media or [])

    def unload(self) -> None:
        """End the worker, which frees the GPU."""
        self.worker.unload()
