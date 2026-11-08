"""d3's half of the worker (DESIGN.md §13.3): it loads a d3 folder through its own remote code and answers a window's
questions from `system_one`. Standard library only at the top, and nothing from `mms`: it runs in d3's env.
"""

WARM_TEXT = {"seen": {"type": "noul", "instructions": "Is the light on?"}}
WARM_PICTURE = {"seen": {"type": "noul", "instructions": "Is there a red ball in the picture?"}}


def load(folder, options):
    """d3 on the GPU, offline, warmed with one text and one 640x360 picture, so a window's first call is a normal one."""
    from PIL import Image
    from transformers import AutoModel

    # "cuda:0", not "cuda": d3 gives the device to torch.cuda.set_device, which takes an index
    model = AutoModel.from_pretrained(str(folder), trust_remote_code=True, device="cuda:0")
    picture = Image.new("RGB", (640, 360), (200, 200, 190))
    for state, questions, media in (("warm-up", WARM_TEXT, []), (None, WARM_PICTURE, [picture])):
        for name, answer in decide(model, state, questions, media).items():
            if "error" in answer:
                raise RuntimeError(f"d3 could not answer its warm-up {name!r}: {answer['error']}: {answer.get('message')}")
    return model, {"model_name": model.model_name}


def decide(model, state, questions, media):
    """d3's native answers, `{name: answer}`. A question it cannot take is an answer with an `error`, not an exception;
    its pictures are PIL images, and `state` None stays None."""
    pictures = []
    for medium in media:
        if isinstance(medium, dict):
            raise ValueError(f"d3 reads pictures, not {medium.get('type')}")
        pictures.append(medium)
    return model.system_one(state=state, questions=questions, images=pictures or None)["answers"]
