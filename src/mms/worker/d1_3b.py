"""d1-3B's half of its worker (DESIGN.md §13.3): `worker/serve.py` loads this file by path and calls `load` and `decide`.
d1-3B runs through transformers from its own code, offline, in bf16 on the GPU. Standard library only at the top (torch and
transformers are imported inside the functions), and nothing from `mms`, so a plain `python -I -S` imports it.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

WARM_TEXT = {"seen": {"type": "noul", "instructions": "Is the light on?"}}
WARM_PICTURE = {"seen": {"type": "noul", "instructions": "Is there a red ball in the picture?"}}


def load(folder: str, options: dict) -> tuple:
    """Import torch and transformers and read the weights onto the GPU, check them against the code, and warm up with a
    text and a picture, so the first real call is a normal one. Returns the model and its facts, which include the picture
    cap of the loaded code (`max_pixels`)."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    import torch
    import transformers
    from PIL import Image, ImageDraw
    from transformers import AutoModel

    folder = Path(folder)
    if not torch.cuda.is_available():
        raise RuntimeError("d1-3B needs a CUDA GPU, and torch sees none")
    if not (folder / "config.json").exists():
        raise FileNotFoundError(f"no d1-3B model in {folder}")
    model, found = AutoModel.from_pretrained(str(folder), trust_remote_code=True, dtype=torch.bfloat16,
                                             output_loading_info=True)
    _check_loaded(found)
    model = model.to("cuda").eval()
    if "images" not in inspect.signature(model.system_one).parameters:
        raise RuntimeError("d1-3B's system_one takes no `images`: its code is not the one decider_d1_3b was written for")
    caps = {module.VISION_MAX_PIXELS for module in _code("runner", "VISION_MAX_PIXELS")}
    if len(caps) != 1:
        raise RuntimeError(f"expected one VISION_MAX_PIXELS in d1-3B's runner module, found {sorted(caps) or 'none'}")
    if sys.platform == "win32":  # torch's Windows wheels have no flash attention: d1-3B takes its explicit-product path
        patched = _code("hybrid", "_fused")
        if not patched:
            raise RuntimeError("found no d1-3B hybrid module to patch for Windows' torch")
        for module in patched:
            module._fused = lambda q: False
    picture = Image.new("RGB", (640, 360), (200, 200, 190))
    ImageDraw.Draw(picture).ellipse([380, 180, 444, 244], fill=(220, 30, 30))
    decide(model, {"warm": 1}, WARM_TEXT, [])
    decide(model, None, WARM_PICTURE, [picture])
    torch.cuda.synchronize()
    return model, {"torch": torch.__version__, "transformers": transformers.__version__, "max_pixels": caps.pop()}


def decide(model, state, questions: dict, media: list | None) -> dict:
    """d1-3B's answers to a window, as it gives them. Its pictures are PIL images, and `state` None stays None."""
    pictures = []
    for medium in media or []:
        if isinstance(medium, dict):
            raise ValueError(f"d1-3B reads pictures, not {medium.get('type')}")
        pictures.append(medium)
    import torch

    with torch.no_grad():
        return model.system_one(state, questions, images=pictures or None)["answers"]


def _code(suffix: str, name: str) -> list:
    """The loaded modules of the model's own code (`transformers_modules.<folder>.<suffix>`) that have `name`."""
    return [module for key, module in list(sys.modules.items())
            if key.startswith("transformers_modules") and key.endswith("." + suffix) and hasattr(module, name)]


def _check_loaded(found: dict) -> None:
    """Refuse weights that do not fit the code: a missing, unexpected or mismatched key leaves d1-3B half-initialised and
    running with no error. `found` is transformers' loading info (a mismatched entry is a tuple led by the key)."""
    wrong = {kind: sorted(str(key[0] if isinstance(key, tuple) else key) for key in found.get(f"{kind}_keys") or ())
             for kind in ("missing", "unexpected", "mismatched")}
    wrong = {kind: names for kind, names in wrong.items() if names}
    if wrong:
        raise RuntimeError("the weights do not fit the code: " + "; ".join(
            f"{len(names)} {kind} ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})"
            for kind, names in wrong.items()))
