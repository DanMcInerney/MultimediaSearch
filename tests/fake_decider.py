"""A decider for tests (TESTBED.md): it answers from content a test knows, in d1's answer shapes. Its facts (`ABOUT`)
are d1's, so what it is handed is what d1 would be handed: it raises a `ValueError` for a file medium, which d1 takes
none of, and for sound that is not at `ABOUT.audio_rate`.

What a question asks for is its instructions, so a test picks which content counts by how it words the question:
- `contains:kettle`: yes when the state holds the word;
- `corner:red`: yes when a picture's top-left quarter is that colour (the picture is upright);
- `colour:red`: yes when a picture's mean colour is that colour;
- `shade:200,30,30`: yes when at least 2% of a picture's pixels are within 30 (RGB distance) of that shade, wherever they
  are (a part of the frame, after JPEG or video coding);
- `ink` (a whole word): yes when a picture has at least 1% of its pixels darker than 64 and at least 1% brighter than
  192 (what is drawn shows against its backdrop);
- `pitch:659`: yes when a sound has RMS of at least 0.01 and at least 30% of its energy within 3% of that frequency in Hz;
- `loud`: yes when a sound's RMS is above 0.05;
- `red`: yes when a picture's mean colour is mostly red;
- anything else: yes when there is a red picture or a loud sound.

A yes is P(yes) 0.95, a no 0.05, or the number after `grade:` when the state holds one (`grade:0.4`: a graded no).
A choice answers its first option for yes and its last for no (the `none` that `ask` adds, when it adds one), a score
its top level for yes and 0 for no. Every type answers in d1's real shape: text, number and date
read their value from the state (the first line, the last number, the first ISO date) and are None for no; a ranged
number answers the top of its range for yes, with an interval and no `found`; box and boxes find one box for yes. A
`slow:<s>` word in a question makes the call sleep that long; a `load:<s>` word does so on the first call that carries it
only, as d1 does while it loads. Each call is appended to `fake-decider.jsonl` in the home
(the loader sets `home`): what a test checks d1 would have read. Per picture it records its size and `looks` (the colour
of each quarter, top-left first, its mean colour, whether it has ink); per sound its length and `heard` (RMS, sample rate,
the second of its loudest sample); per file medium its name and media type (it takes none, and raises after logging);
and `text`, the state's counts of NUL, U+FFFD, ESC and CR characters and its first 60
characters. Colours are named by the nearest of red, green, blue, yellow, white, black and grey (`other` beyond 100 in
RGB); a picture's quarters are split at half its width and half its height.
"""

import json
import re
import time

import numpy as np

from mms.decider_d1 import ABOUT

home = None  # set by the loader
loaded = False  # whether a `load:<s>` word has slept yet
LOUD_RMS = 0.05
PITCH_RMS = 0.01  # the RMS a sound needs for `pitch:<hz>`, so that room noise alone is never a pitch
PITCH_SHARE = 0.3  # the share of a sound's spectral energy that must lie within PITCH_BAND of the pitch
PITCH_BAND = 0.03  # of the pitch, either side
COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
           "white": (255, 255, 255), "black": (0, 0, 0), "grey": (128, 128, 128)}
OTHER_FROM = 100  # an RGB distance from every named colour beyond which a colour is `other`
SHADE_TOL = 30  # the RGB distance within which a pixel is a `shade:R,G,B`
SHADE_SHARE = 0.02  # the share of a picture's pixels that must be that shade for `shade:R,G,B`
INK_SHARE = 0.01  # the share of a picture's pixels that must be dark, and that must be bright, for it to have ink


def _red(image) -> bool:
    r, g, b = np.asarray(image.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    return r > 120 and r > 2 * g and r > 2 * b


def colour_name(rgb) -> str:
    """The named colour nearest to `rgb`, or `other`."""
    name, distance = min(((n, float(np.linalg.norm(np.subtract(rgb, c)))) for n, c in COLOURS.items()),
                         key=lambda pair: pair[1])
    return name if distance <= OTHER_FROM else "other"


def _colour_of(pixels) -> str:
    return colour_name(pixels.reshape(-1, 3).mean(axis=0)) if pixels.size else "other"


def corners(image) -> list[str]:
    """The colour names of a picture's quarters: top-left, top-right, bottom-left, bottom-right."""
    pixels = np.asarray(image.convert("RGB"), dtype=float)
    y, x = pixels.shape[0] // 2, pixels.shape[1] // 2
    return [_colour_of(q) for q in (pixels[:y, :x], pixels[:y, x:], pixels[y:, :x], pixels[y:, x:])]


def colour(image) -> str:
    """The colour name of a picture's mean colour."""
    return _colour_of(np.asarray(image.convert("RGB"), dtype=float))


def shade_share(image, rgb) -> float:
    """The share of a picture's pixels within SHADE_TOL (Euclidean RGB distance) of the colour `rgb`."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.int32).reshape(-1, 3)
    if not len(pixels):
        return 0.0
    return float((np.square(pixels - np.asarray(rgb, dtype=np.int32)).sum(axis=1) <= SHADE_TOL ** 2).mean())


def has_ink(image) -> bool:
    """Whether a picture has dark (below 64) and bright (above 192) pixels, each at least INK_SHARE of it."""
    grey = np.asarray(image.convert("L"))
    return bool((grey < 64).mean() >= INK_SHARE and (grey > 192).mean() >= INK_SHARE)


def _rms(clip: dict) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(clip["audio"], dtype=np.float64)))))


def _loud(clip: dict) -> bool:
    return _rms(clip) > LOUD_RMS


def pitch_share(samples: np.ndarray, rate: int, hz: float) -> float:
    """The share of a sound's spectral energy (|rfft|², the DC bin left out) within 3% of `hz`; 0.0 for silence."""
    samples = np.asarray(samples, dtype=np.float64)
    if len(samples) < 2:
        return 0.0
    energy = np.abs(np.fft.rfft(samples)) ** 2
    energy[0] = 0.0
    total = energy.sum()
    if total <= 0:
        return 0.0
    near = np.abs(np.fft.rfftfreq(len(samples), 1 / rate) - hz) <= PITCH_BAND * hz
    return float(energy[near].sum() / total)


def _pitched(clip: dict, hz: float) -> bool:
    return _rms(clip) >= PITCH_RMS and pitch_share(clip["audio"], clip["sampling_rate"], hz) >= PITCH_SHARE


def _yes(instructions: str, state, media: list) -> bool:
    text = str(instructions).lower()
    word = re.search(r"contains:(\S+)", text)
    pictures = [m for m in media if not isinstance(m, dict)]
    sounds = [m for m in media if isinstance(m, dict) and m.get("type") == "audio"]
    if word:
        return word.group(1) in str(state or "").lower()
    if corner := re.search(r"corner:(\w+)", text):
        return any(corners(p)[0] == corner.group(1) for p in pictures)
    if shade := re.search(r"colour:(\w+)", text):
        return any(colour(p) == shade.group(1) for p in pictures)
    if tint := re.search(r"shade:(\d+),(\d+),(\d+)", text):
        rgb = tuple(int(v) for v in tint.groups())
        return any(shade_share(p, rgb) >= SHADE_SHARE for p in pictures)
    if re.search(r"\bink\b", text):
        return any(has_ink(p) for p in pictures)
    if pitch := re.search(r"pitch:(\d+(?:\.\d+)?)", text):
        return any(_pitched(s, float(pitch.group(1))) for s in sounds)
    if "loud" in text:
        return any(_loud(s) for s in sounds)
    if "red" in text:
        return any(_red(p) for p in pictures)
    return any(_red(p) for p in pictures) or any(_loud(s) for s in sounds)


def _log(state, questions: dict, media: list) -> None:
    if home is None:
        return
    text = str(state or "")
    entry = {"questions": {name: {"type": q["type"], "instructions": q["instructions"],
                                  "options": list(q["criteria"]) if isinstance(q.get("criteria"), (dict, list)) else None}
                           for name, q in questions.items()},
             "pictures": [list(m.size) for m in media if not isinstance(m, dict)],
             "looks": [{"corners": corners(m), "colour": colour(m), "ink": has_ink(m)}
                       for m in media if not isinstance(m, dict)],
             "files": [{"name": m["name"], "media": m["media"]} for m in media if isinstance(m, dict) and m.get("type") == "file"],
             "sounds": [len(m["audio"]) for m in media if isinstance(m, dict) and m.get("type") == "audio"],
             "heard": [{"rms": float(np.sqrt(np.mean(np.square(m["audio"], dtype=np.float64)))),
                        "rate": m["sampling_rate"], "peak_s": int(np.argmax(np.abs(m["audio"]))) / m["sampling_rate"]}
                       for m in media if isinstance(m, dict) and m.get("type") == "audio"],
             "state_len": len(state) if state else 0,
             "text": {"nul": text.count("\0"), "fffd": text.count("�"), "esc": text.count("\x1b"),
                      "cr": text.count("\r"), "head": text[:60]},
             "at": time.time()}
    with open(home / "fake-decider.jsonl", "a", encoding="utf-8") as out:
        out.write(json.dumps(entry) + "\n")


def _answer(question: dict, yes: bool, state) -> dict:
    """d1's answer to one question in d1's own shape for its type."""
    text = str(state or "")
    graded = re.search(r"grade:(\d?\.\d+)", text)
    p = 0.95 if yes else float(graded.group(1)) if graded else 0.05
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": p}
    if kind == "choice":
        options = list(question["criteria"])
        if question.get("multiple"):
            chosen = options[:1] if yes else []
            return {"type": "choice", "choices": chosen, "probabilities": {o: (0.9 if o in chosen else 0.1) for o in options}}
        chosen = options[0] if yes else options[-1]
        probabilities = {o: (0.9 if o == chosen else 0.1 / max(len(options) - 1, 1)) for o in options}
        return {"type": "choice", "choice": chosen, "confidence": probabilities[chosen], "probabilities": probabilities}
    if kind == "score":
        levels = question["criteria"]
        top = len(levels) - 1
        level = top if yes else 0
        return {"type": "score", "score": float(level), "confidence": 0.9,
                "probabilities": {str(i): (0.9 if i == level else 0.0) for i in range(len(levels))},
                "legend": {str(i): text for i, text in enumerate(levels)}}
    if kind == "box":
        return {"type": "box", "box": {"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 0.5} if yes else None,
                "found": p, "confidence": 0.9 if yes else 0.0}
    if kind == "boxes":
        box = {"box": {"top": 0.0, "bottom": 0.5, "left": 0.0, "right": 0.5}, "found": p, "confidence": 0.9}
        return {"type": "boxes", "boxes": [box] if yes else [], "count": 1 if yes else 0, "at_least": False,
                "presence": [p]}
    if kind == "number" and question.get("range"):  # a ranged number: no `found`, and an interval
        low, high = question["range"]
        number, interval = (high, [(low + high) / 2, high]) if yes else (low, [low, low])
        return {"type": "number", "number": number, "interval": interval, "unit": question.get("unit")}
    if kind in ("text", "number", "date"):  # read from the state: the first number, date or line in it
        found = {"text": re.search(r"\S.*", text), "number": next(iter(reversed(list(re.finditer(r"-?\d+(?:\.\d+)?", text)))), None),
                 "date": re.search(r"\d{4}-\d\d-\d\d", text)}[kind]
        if not yes or found is None:
            return {"type": kind, kind: None, "span": None, "found": 0.05, **({"text": None} if kind == "number" else {})}
        raw = found.group(0)
        value = (float(raw) if "." in raw else int(raw)) if kind == "number" else raw
        return {"type": kind, kind: value, "span": [found.start(), found.end()], "found": p,
                **({"text": raw, "unit": question.get("unit")} if kind == "number" else {})}
    raise ValueError(f"the fake decider answers noul, choice, score, text, number, date, box and boxes questions, "
                     f"not {kind!r}")


def _seconds(word: str, questions: dict) -> list[float]:
    """The numbers after `word:` in the questions' instructions."""
    pattern = rf"\b{word}:(\d+(?:\.\d+)?)"
    return [float(m) for q in questions.values() for m in re.findall(pattern, q["instructions"])]


def decide(state, questions: dict, media: list) -> dict:
    global loaded
    _log(state, questions, media)  # first, so that a call it then refuses is in the log too
    for medium in media:
        if isinstance(medium, dict) and medium.get("type") == "file":
            raise ValueError("the fake decider reads pictures and sound, as d1 does, not a file")
        if isinstance(medium, dict) and medium.get("type") == "audio" and medium["sampling_rate"] != ABOUT.audio_rate:
            raise ValueError(f"the fake decider is given sound at {ABOUT.audio_rate} Hz, not {medium.get('sampling_rate')}")
    slow, load = _seconds("slow", questions), _seconds("load", questions)
    if slow:
        time.sleep(max(slow))
    if load and not loaded:
        loaded = True
        time.sleep(max(load))
    return {name: _answer(question, _yes(question["instructions"], state, media), state)
            for name, question in questions.items()}
