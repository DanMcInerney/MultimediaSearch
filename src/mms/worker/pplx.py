"""Perplexity's half of the worker (DESIGN.md §13.3): it loads the model with the `pplx_decider` package the folder ships in
`source/src` and answers a window's questions, one request each. Standard library only at the top, and nothing from
`mms`: it runs in Perplexity's env.
"""

import sys
from pathlib import Path

WARM_TEXT = {"seen": {"type": "noul", "instructions": "Is the light on?"}}
WARM_PICTURE = {"seen": {"type": "noul", "instructions": "Is there a red ball in the picture?"}}


def load(folder, options):
    """The model on the GPU by the reference path (`fast` off: no packed kernels, the card's exact computation), warmed
    with one text and one 640x360 picture."""
    from PIL import Image

    root = Path(folder)
    sys.path.insert(0, str(root / "source" / "src"))
    from pplx_decider import Decider

    model = Decider.from_pretrained(root, device="cuda")
    picture = Image.new("RGB", (640, 360), (200, 200, 190))
    decide(model, "warm-up", WARM_TEXT, [])
    decide(model, None, WARM_PICTURE, [picture])
    return model, {"max_pixels": model.max_pixels, "temperature": model.temperature}


def decide(model, state, questions, media):
    """The native answers, `{name: answer}`, from one `decide` call for each question: the card's bit-stable mode, since a
    question read beside others sees them. `state` None is sent as "", which the model's request needs; its pictures are
    PIL images. A request it refuses (`RequestError`, `LengthError`) is a ValueError."""
    pictures = []
    for medium in media:
        if isinstance(medium, dict):
            raise ValueError(f"Perplexity's decider reads pictures, not {medium.get('type')}")
        pictures.append(medium)
    answers = {}
    for name, question in questions.items():
        request = {"state": "" if state is None else state, "questions": {name: question}}
        if pictures:
            request["images"] = pictures
        answers.update(model.decide(request)["answers"])
    return answers
