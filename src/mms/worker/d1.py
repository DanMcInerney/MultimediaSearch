"""d1's half of its worker (DESIGN.md §13.3): `worker/serve.py` loads this file by path and calls `load` and `decide`. d1-Omni-2B
and d1-2B run through transformers from their own code, offline, in bf16 on the GPU. Standard library only at the top (torch
and transformers are imported inside the functions), and nothing from `mms`, so a plain `python -I -S` imports it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# config.json key -> where the loaded engine holds it
CONFIG_KEYS = {"d1_attention": "attention", "d1_set_size": "set_size", "d1_vision_max_pixels": "max_pixels",
               "d1_numbers": "tokenizer.d1_numbers", "d1_codes": "tokenizer.d1_codes"}
WARM_TEXT = {"seen": {"type": "noul", "instructions": "Is the light on?"}}
WARM_PICTURE = {"seen": {"type": "noul", "instructions": "Is there a red ball in the picture?"}}
WARM_SOUND = {"seen": {"type": "noul", "instructions": "Is there a tone?"}}


def load(folder: str, options: dict) -> tuple:
    """Import torch and transformers and read the weights onto the GPU, check d1's config, and warm up with one call of
    each kind of media, so the first real call is a normal one. `options["warm_sound"]` is whether the model hears (d1-2B
    does not). Returns the model and its facts."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import torch
    import transformers
    from PIL import Image, ImageDraw
    from transformers import AutoModel

    folder = Path(folder)
    if not torch.cuda.is_available():
        raise RuntimeError("d1 needs a CUDA GPU, and torch sees none")
    if not (folder / "config.json").exists():
        raise FileNotFoundError(f"no d1 model in {folder}")
    model, found = AutoModel.from_pretrained(str(folder), trust_remote_code=True, dtype=torch.bfloat16,
                                             output_loading_info=True)
    _check_loaded(found)
    model = model.to("cuda").eval()
    if sys.platform == "win32":  # torch's Windows wheels have no flash attention: d1 takes its masked-SDPA path
        patched = [m for name, m in sys.modules.items() if name.startswith("transformers_modules")
                   and name.endswith(".attention") and hasattr(m, "_fused")]
        if not patched:
            raise RuntimeError("found no d1 attention module to patch for Windows' torch")
        for module in patched:
            module._fused = lambda q: False
    _check_config(model, folder)
    picture = Image.new("RGB", (640, 360), (200, 200, 190))
    ImageDraw.Draw(picture).ellipse([380, 180, 444, 244], fill=(220, 30, 30))
    decide(model, {"warm": 1}, WARM_TEXT, [])
    decide(model, None, WARM_PICTURE, [picture])
    if options["warm_sound"]:
        tone = np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000).astype(np.float32)
        decide(model, None, WARM_SOUND, [{"type": "audio", "audio": tone, "sampling_rate": 16000}])
    torch.cuda.synchronize()
    return model, {"torch": torch.__version__, "transformers": transformers.__version__}


def decide(model, state, questions: dict, media: list | None) -> dict:
    """d1's answers to a window, as it gives them."""
    import torch

    with torch.no_grad():
        return model.system_one(state, questions, media=media or None)["answers"]


def _check_loaded(found: dict) -> None:
    """Refuse weights that do not fit the code: a missing, unexpected or mismatched key leaves d1 half-initialised and
    running with no error. `found` is transformers' loading info (a mismatched entry is a tuple led by the key)."""
    wrong = {kind: sorted(str(key[0] if isinstance(key, tuple) else key) for key in found.get(f"{kind}_keys") or ())
             for kind in ("missing", "unexpected", "mismatched")}
    wrong = {kind: names for kind, names in wrong.items() if names}
    if wrong:
        raise RuntimeError("the weights do not fit the code: " + "; ".join(
            f"{len(names)} {kind} ({', '.join(names[:3])}{', ...' if len(names) > 3 else ''})"
            for kind, names in wrong.items()))


def _check_config(model, folder: Path) -> None:
    """The engine must hold what config.json records for each `d1_*` key: a library that dropped them leaves d1 on
    causal attention, set size 4, code readout and a smaller picture cap, with no error."""
    recorded = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    engine = model.engine
    if engine.calibration is None:
        raise RuntimeError("d1 loaded without its calibration")
    wrong, present = [], 0
    for key, where in CONFIG_KEYS.items():
        if key not in recorded:
            continue
        present += 1
        held = engine
        for part in where.split("."):
            held = getattr(held, part, None)
        if held != recorded[key]:
            wrong.append(f"{key} is {recorded[key]!r} in config.json but the engine holds {held!r}")
    if not present:
        raise RuntimeError("config.json holds none of d1's keys " + ", ".join(CONFIG_KEYS))
    if wrong:
        raise RuntimeError("d1's config keys fell to defaults (transformers dropped them): " + "; ".join(wrong))
