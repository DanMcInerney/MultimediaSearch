"""A worker's family file for tests (TESTBED.md, "No GPU"; DESIGN.md §13.3): what `worker/serve.py` loads by path, in place
of `worker/d1.py`. Nothing here needs a GPU, a model or any package beyond numpy and PIL, which `serve.py` itself needs
to decode a window's media.

`load(folder, options)` reads `fake.json` in the folder (all keys optional):
- `delay_s`: sleep that long while loading; `pause_s`: sleep that long in each `decide`;
- `fast`: echo only the sizes, not the hashes of the bytes (for timing the wire alone);
- `print_on_load`: print to stdout, and `os.write(1, ...)`, as a library does while it loads;
- `crash_on`, `hang_on`: a word; a `decide` whose state or instructions hold it exits at once with code 3, or sleeps for
  an hour.

`decide` answers each question as a noul. A question that says `raise:value` raises `ValueError`, `raise:runtime` a
`RuntimeError`. Otherwise its instructions say what counts (as `fake_decider` does): `corner:red` is yes when a picture's
top-left quarter is mostly that colour, `contains:kettle` when the state holds the word, `red` when a picture's mean
colour is mostly red; anything else is no. A yes is 0.95 and a no 0.05. What reached the model is echoed under
`__echo__`: per picture its size and the sha256 of its bytes; per sound its dtype, rate, count and the sha256 of its
bytes; per file its name, media type, length and the sha256 of its bytes. A crash writes a line to stderr first."""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

CRASH_CODE = 3
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "white": (255, 255, 255), "black": (0, 0, 0)}


def load(folder, options):
    settings = json.loads((Path(folder) / "fake.json").read_text(encoding="utf-8"))
    if settings.get("print_on_load"):
        print("a library says hello on stdout")
        os.write(1, b"MSW1 not a frame, written to fd 1 from C\n")
        sys.stdout.write("and again, unflushed\n")
    time.sleep(settings.get("delay_s", 0))
    return settings, {"pid": os.getpid(), "isolated": sys.flags.isolated, "folder": str(folder), "options": options,
                      "stdout_is_stderr": sys.stdout is sys.stderr}


def _colour(pixels) -> str:
    import numpy as np

    mean = pixels.reshape(-1, 3).mean(axis=0)
    return min(COLOURS, key=lambda name: float(np.linalg.norm(mean - COLOURS[name])))


def _echo(medium, fast: bool) -> dict:
    def sha(data) -> str | None:
        return None if fast else hashlib.sha256(data).hexdigest()

    if isinstance(medium, dict) and medium.get("type") == "audio":
        audio = medium["audio"]
        return {"kind": "audio", "dtype": str(audio.dtype), "rate": medium["sampling_rate"], "count": int(audio.size),
                "sha256": sha(audio.tobytes())}
    if isinstance(medium, dict) and medium.get("type") == "file":
        return {"kind": "file", "name": medium["name"], "media": medium["media"], "length": len(medium["data"]),
                "sha256": sha(medium["data"])}
    return {"kind": "image", "size": list(medium.size), "mode": medium.mode, "sha256": sha(medium.tobytes())}


def _yes(instructions: str, state, media: list) -> bool:
    import numpy as np

    text = instructions.lower()
    pictures = [m for m in media if not isinstance(m, dict)]
    if word := re.search(r"contains:(\S+)", text):
        return word.group(1) in str(state or "").lower()
    if corner := re.search(r"corner:(\w+)", text):
        return any(_colour(np.asarray(p, dtype=float)[:p.size[1] // 2, :p.size[0] // 2]) == corner.group(1)
                   for p in pictures)
    if "red" in text:
        return any(_colour(np.asarray(p, dtype=float)) == "red" for p in pictures)
    return False


def decide(model, state, questions, media):
    text = " ".join([str(state or "")] + [str(q["instructions"]) for q in questions.values()])
    if model.get("crash_on") and model["crash_on"] in text:
        print(f"crashing on {model['crash_on']}", file=sys.stderr, flush=True)
        os._exit(CRASH_CODE)
    if model.get("hang_on") and model["hang_on"] in text:
        time.sleep(3600)
    time.sleep(model.get("pause_s", 0))
    answers = {}
    for name, question in questions.items():
        instructions = str(question["instructions"])
        if "raise:value" in instructions:
            raise ValueError(f"{name}: this question does not fit")
        if "raise:runtime" in instructions:
            raise RuntimeError(f"{name}: the model failed")
        answers[name] = {"type": "noul", "noul": 0.95 if _yes(instructions, state, media) else 0.05}
    answers["__echo__"] = [_echo(m, bool(model.get("fast"))) for m in media]
    return answers
