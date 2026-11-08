"""The answer format d3, Perplexity's decider, H2O-Lightning and d1-3B share (DESIGN.md §13.3), and mms's questions and
answers in it.
`to_model` is mms's questions as these models take them, or a ValueError naming the first one they cannot; `from_model` is
their answers as mms reads them. A model's own rules and its extras (H2O's box) are its backend's. Nothing heavy is imported.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from mms.decider import About

MAX_OPTIONS = 255  # the most options of a choice: d3, Perplexity and H2O read 255 answer codes; d1-3B's is not verified
NOUL_KEYS = {"true", "false"}


def to_model(about: About, questions: dict, takes: Callable[[dict], set] | None = None) -> dict:
    """The models' questions: a type `about` answers, with its instructions and criteria. Any other key is refused, and so
    is a yes/no criteria that describes more than true and false, which the models would leave out of the question without a
    word. `takes(question)` is the keys besides type and instructions that one question may carry (default
    `about.question_keys`)."""
    native = {}
    for name, question in questions.items():
        if question.get("type") not in about.types:
            raise ValueError(f"question {name!r}: {about.name} answers {', '.join(about.types)}, not "
                             f"{question.get('type')!r}")
        allowed = {"type", "instructions"} | (about.question_keys if takes is None else takes(question))
        extra = sorted(set(question) - allowed)
        if extra:
            raise ValueError(f"question {name!r} has {', '.join(extra)}, which {about.name} does not take for a "
                             f"{question['type']} question; it takes {', '.join(sorted(allowed))}")
        criteria, kind = question.get("criteria"), question["type"]
        if kind == "noul" and criteria is not None and (not isinstance(criteria, dict) or set(criteria) - NOUL_KEYS):
            raise ValueError(f"question {name!r}: a yes/no's criteria describes true and false only, as "
                             f'{{"true": "...", "false": "..."}}, not {criteria!r}')
        if kind == "choice" and not (isinstance(criteria, dict) and 1 <= len(criteria) <= MAX_OPTIONS):
            raise ValueError(f"question {name!r}: a choice has criteria, {{option: what it means}}, of 1 to {MAX_OPTIONS} "
                             f"options")
        if kind == "score" and not (isinstance(criteria, list) and len(criteria) >= 2):
            raise ValueError(f"question {name!r}: a score has criteria, a list of at least 2 levels, lowest first")
        native[name] = dict(question)
    return native


def from_model(about: About, answers: dict, questions: dict, convert: dict[str, Callable[[str, dict], dict]] | None = None
               ) -> dict:
    """mms's answers from the model's: a choice without its `confidence` (it is not the pick's p, and `probabilities` has it),
    a score at its median level (the model's own is the expected level), and an `error` entry a ValueError. `convert[type]`
    makes the answer of a type of the model's own, from its name and the answer."""
    out = {}
    for name, question in questions.items():
        answer = answers.get(name)
        if not isinstance(answer, dict):
            raise ValueError(f"{about.name} gave no answer to question {name!r}")
        if "error" in answer:
            raise ValueError(f"{about.name}: question {name!r}: {answer['error']}: {answer.get('message')}")
        if question["type"] == "choice":
            answer = {key: value for key, value in answer.items() if key != "confidence"}
        elif question["type"] == "score":
            answer = {**answer, "score": median_level(answer.get("probabilities"))}
        elif convert and question["type"] in convert:
            answer = convert[question["type"]](name, answer)
        out[name] = answer
    return out


def median_level(probabilities) -> float | None:
    """The first level whose cumulative probability reaches 0.5, from `probabilities` keyed "0".."n-1" (d1's own readout).
    None for any other keys or values: a model's expected level is never passed off as this."""
    if not isinstance(probabilities, Mapping) or not probabilities:
        return None
    if set(probabilities) != {str(level) for level in range(len(probabilities))}:
        return None
    levels = [probabilities[str(level)] for level in range(len(probabilities))]
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in levels):
        return None
    total = sum(levels)
    if total <= 0:
        return None
    mass = 0.0
    for level, p in enumerate(levels):
        mass += p / total  # normalised, as d1 does
        if mass >= 0.5:
            return float(level)
    return float(len(levels) - 1)
