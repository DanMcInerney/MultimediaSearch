"""Ask the memory (DESIGN.md §13.2): walk what was recorded, newest first, put typed questions about each window to the
decider, and report the sightings, one label on the best window of each, and the candidates: the best windows that
did not hit, one per moment, which are never labelled.

    mms ask "Is there a red mug?" --answers yes --context "the kitchen camera" --since 2h --source porch --limit 1
    mms ask --questions asks/mug.json --since 1d

This file is shipped: copy it into the home's `scripts/` and change it (the hit rules, the lint, the state, the cost model).
It defines `run(job)`, which the daemon's job runner calls, and `cli(argv)`, which turns the command line into `params`.

Questions come from `question` (yes/no, or a choice with `answers`) or from `questions`, a dict of question name to question,
each a typed question (DESIGN.md §13.3). A question may carry `hit`, which the decider never sees:
    noul                    a P(yes) threshold (true is the model's own, `about.hit`)
    choice                  the option, or the list of options, that count
    score                   the least level that counts
    number                  the least value that counts (a ranged number); true: counts when a value is found
    box, boxes              true: counts when one is found; `said` gives its edges, fractions of the picture from its top left
    text, date              true: counts when a value is found
A window is a hit when every question with a `hit` hits. Hits of one source within `merge_s` of each other are one
sighting, labelled once on its best window. A sighting grows to at most `max_sighting_s` (600; 0 for no cap): a run that
goes on is cut there, so "last seen" does not read back through all of it, and the cut sighting says `capped`. A window may
stand for a run of video frames that look the same (the reader's `still_s`), so a sighting spans those too. A candidate
is a window that did not hit, best `p` first; windows of one source within `merge_s` of a better candidate or of a sighting
are the same moment and are left out. They come from the windows scanned, so a scan stopped early has seen only part.
Sightings come in scan order and the scan stops at `limit` of them. With `best_first` it reads the whole range (or to `max_s`)
and gives the `limit` sightings with the highest `p`, best first, each labelled once; a candidate still leaves out a
window near any sighting, those that did not make the `limit` too (they are let go as the scan goes, with the candidates beside
them, so they do not fill the pool).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time

from mms.config import iso, parse_when, when
from mms.decider import QUESTION_KEYS

PARAMS = ("question", "answers", "hit", "context", "questions", "since", "until", "sources", "scope", "limit",
          "candidates", "oldest_first", "best_first", "every_s", "still_s", "max_px", "merge_s", "max_sighting_s", "max_s")
LIMIT_MAX = 50  # sightings
FOUND = 0.5  # a typed answer hits at this `found`
CONTEXT_MAX = 4000
DEFAULT_SINCE_S = 86400
CANDIDATE_POOL = 200  # the most moments kept while scanning, the best non-hit of each; the candidates are chosen from them
UPDATE_S = 0.25  # how often the progress is reported while nothing else changes

# The cost of a window, for `progress.of_about` and the lint: the model's own costs come from its About (`window_seconds`),
# and this is the cost mms adds to a picture. It is this script's assumption, to change.
DECODE_S = 0.010  # a frame decoded straight to size, measured on the CPU (DESIGN.md §16: 9 ms at 1080p)

TIME_WORDS = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|this (morning|afternoon|evening|week|month)|last (night|week|month|year)|"
    r"(on )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|at \d{1,2}(:\d\d)?\s?(am|pm)?|\d{1,2}\s?(am|pm)|"
    r"\d{4}-\d\d-\d\d|(january|february|march|april|may|june|july|august|september|october|november|december) \d)\b",
    re.IGNORECASE)
JOINED = re.compile(r"\b(and|or)\b", re.IGNORECASE)
SPEECH = re.compile(r"\b(said|says|say|talk|talks|talking|spoke|speak|speaks|speaking|speech|conversation)\b", re.IGNORECASE)


# The answers, one parser for each type (DESIGN.md §13.3). A parser gives what its type's answer says, or raises
# `Unreadable`: an answer that is not in its type's shape is a counted warning, never a window skipped without a word.

class Unreadable(ValueError):
    """An answer that is not in the shape its question's type has."""


def _number(answer: dict, key: str, low: float | None = None, high: float | None = None) -> float:
    value = answer.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise Unreadable(f"its {key} is {value!r}, not a number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise Unreadable(f"its {key} is {value!r}, outside {low} to {high}")
    return float(value)


def _value(answer: dict, key: str, kind) -> object | None:
    value = answer.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, kind)):
        raise Unreadable(f"its {key} is {value!r}")
    return value


def _noul(a: dict) -> dict:
    return {"p": _number(a, "noul", 0.0, 1.0)}


def _choice(a: dict) -> dict:
    picked = a["choices"] if "choices" in a else [a.get("choice")]
    odds = a.get("probabilities", {})
    if not isinstance(picked, list) or not all(isinstance(c, str) for c in picked) or not isinstance(odds, dict):
        raise Unreadable(f"its choice is {picked!r}")
    if "choices" not in a and "confidence" in a:
        odds = {**odds, picked[0]: odds.get(picked[0], _number(a, "confidence", 0.0, 1.0))}
    return {"picked": picked, "odds": odds}


def _score(a: dict) -> dict:
    return {"level": _number(a, "score", 0.0)}


def _found(a: dict, key: str, kind) -> dict:
    """text and date, and a number read from text: a value, which is None whatever `found` says when it is not there."""
    return {"value": _value(a, key, kind), "found": _number(a, "found", 0.0, 1.0), "unit": a.get("unit")}


def _number_answer(a: dict) -> dict:
    if "interval" in a:  # a ranged number has no `found`: it is an estimate
        interval = a["interval"]
        if not (isinstance(interval, (list, tuple)) and len(interval) == 2 and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in interval)):
            raise Unreadable(f"its interval is {interval!r}")
        value = _value(a, "number", (int, float))
        return {"value": value, "found": 0.0 if value is None else 1.0, "interval": list(interval), "unit": a.get("unit")}
    return _found(a, "number", (int, float))


EDGES = ("top", "left", "bottom", "right")  # the order `said` gives a box's edges in


def _edges(box) -> str:
    """A box's edges, fractions of the picture from its top left: `top 0.15, left 0.06, bottom 0.24, right 0.27`. Each is
    from 0 to 1, and top is not below bottom nor left right of right."""
    if not (isinstance(box, dict) and set(EDGES) <= set(box)):
        raise Unreadable(f"its box is {box!r}")
    edges = {edge: _number(box, edge, 0.0, 1.0) for edge in EDGES}
    if edges["top"] > edges["bottom"] or edges["left"] > edges["right"]:
        raise Unreadable(f"its box is {box!r}, with an edge past the opposite one")
    return ", ".join(f"{edge} {value:.2f}" for edge, value in edges.items())


def _box(a: dict) -> dict:
    box = a.get("box")
    return {"edges": None if box is None else _edges(box), "found": _number(a, "found", 0.0, 1.0)}


def _boxes(a: dict) -> dict:
    boxes = a.get("boxes")
    if not isinstance(boxes, list) or not all(isinstance(b, dict) for b in boxes):
        raise Unreadable(f"its boxes are {boxes!r}")
    more = a.get("at_least", False)  # the count is a floor: the model stopped at its limit
    if not isinstance(more, bool):
        raise Unreadable(f"its at_least is {more!r}")
    return {"count": int(_number(a, "count", 0.0)), "at_least": more, "edges": [_edges(b.get("box")) for b in boxes],
            "found": max((_number(b, "found", 0.0, 1.0) for b in boxes), default=0.0)}


PARSERS = {"noul": _noul, "choice": _choice, "score": _score, "text": lambda a: _found(a, "text", str),
           "number": _number_answer, "date": lambda a: _found(a, "date", str), "box": _box, "boxes": _boxes}


def read(name: str, question: dict, answer) -> dict:
    """What the answer to one question says, parsed for its type; `Unreadable` names the question."""
    kind = question.get("type")
    try:
        if not isinstance(answer, dict):
            raise Unreadable(f"it is {type(answer).__name__}, not an object")
        if answer.get("type") != kind:
            raise Unreadable(f"it is a {answer.get('type')!r} answer to a {kind!r} question")
        return PARSERS[kind](answer)
    except Unreadable as error:
        raise Unreadable(f"question {name!r}: {error}") from None


# what counts as a hit

def _threshold(name: str, hit, low: float, high: float) -> float:
    """A number `hit`. Text that is a number ("0.5") is read as that number: the tool's schema takes text for a choice's
    option, so an agent may send a threshold as text, and `mms ask --hit` passes its text on as it is."""
    value = math.nan  # refused below unless `hit` is a finite number, or text that is one
    if not isinstance(hit, bool) and isinstance(hit, (int, float, str)):
        try:
            value = float(hit)
        except (ValueError, OverflowError):
            pass
    if not (math.isfinite(value) and low <= value <= high):
        what = f"a number from {low:g} to {high:g}" if math.isfinite(low) and math.isfinite(high) else "a finite number"
        raise ValueError(f"question {name!r}: hit is {what} here, not {hit!r}")
    return value


def _rule(name: str, question: dict, hit, about):
    """What counts as a hit for one question: a function of its parsed answer giving `(hit, p, what it said)`, where `p`
    is how far the answer is toward counting (a candidate's `p` is the weakest question's). A yes/no `hit` of true
    counts at the model's own threshold, `about.hit`."""
    kind = question.get("type")
    if kind == "noul":
        level = about.hit if hit is True else _threshold(name, hit, 0.0, 1.0)
        return lambda a: (a["p"] >= level, a["p"], "yes" if a["p"] >= level else f"below {level:g}")
    if kind == "choice":
        options = [hit] if isinstance(hit, str) else hit
        criteria = question.get("criteria")
        unknown = isinstance(criteria, dict) and isinstance(options, list) and set(options) - set(criteria)
        if not isinstance(options, list) or not options or unknown:
            raise ValueError(f"question {name!r}: hit is an option, or a list of options, of its criteria")

        def choice(a):
            chosen = [c for c in a["picked"] if c in options]
            if chosen:
                return True, max(a["odds"].get(c, 1.0) for c in chosen), ", ".join(chosen)
            return False, max((a["odds"].get(c, 0.0) for c in options), default=0.0), ", ".join(a["picked"])

        return choice
    if kind == "score":
        least = _threshold(name, hit, 0, 1000)
        return lambda a: (a["level"] >= least, min(a["level"] / least, 1.0) if least > 0 else 1.0, f"level {a['level']:g}")
    if kind == "number":
        if isinstance(hit, bool):
            if hit is not True or question.get("range"):
                raise ValueError(f"question {name!r}: a ranged number's hit is the least value that counts")
            return lambda a: (a["value"] is not None and a["found"] >= FOUND, a["found"], _said_number(a))
        least = _threshold(name, hit, float("-inf"), float("inf"))

        def number(a):
            value = a["value"]
            if value is None:
                return False, 0.0, "no number"
            return value >= least, (1.0 if value >= least or least <= 0 else value / least), _said_number(a)

        return number
    if kind == "box":
        return lambda a: (a["edges"] is not None and a["found"] >= FOUND, a["found"],
                          "none" if a["edges"] is None else f"at {a['edges']}")
    if kind == "boxes":
        return lambda a: (a["count"] > 0, a["found"], _said_boxes(a))
    if hit is not True:  # text, date
        raise ValueError(f"question {name!r}: hit is true for a {kind} question: it counts when a value is found")
    return lambda a: (a["value"] is not None and a["found"] >= FOUND, a["found"],
                      "no value" if a["value"] is None else f"{a['value']}")


def _said_number(a: dict) -> str:
    """The number a question read, with its interval and unit: `3 (2 to 4) beeps`."""
    value = a["value"]
    if value is None:
        return "no number"
    text = f"{value:g}"
    if a.get("interval"):
        text += f" ({a['interval'][0]:g} to {a['interval'][1]:g})"
    return text + (f" {a['unit']}" if a.get("unit") else "")


def _said_boxes(a: dict) -> str:
    """The count of boxes (`at least 16 found` when it is a floor), then each one's edges: `2 found: top 0.10, left 0.20,
    bottom 0.30, right 0.40; top ...`."""
    found = f"{a['count']} found"
    return ("at least " if a["at_least"] else "") + found + (": " + "; ".join(a["edges"]) if a["edges"] else "")


def _check(name: str, question: dict) -> None:
    """One item of `questions` must be a typed question (DESIGN.md §13.3); what every decider refuses, or would show the
    model wrongly (a one-sided yes/no), is refused here."""
    kind = question.get("type")
    if kind is None:
        raise ValueError(f"question {name!r} has no type: each item of questions is a typed question (DESIGN.md §13.3), as "
                         f'{{"type": "noul", "instructions": "Is a mug in view?", "hit": 0.8}}; to find any of several '
                         f"things in one scan, leave out questions and give question, answers (the things) and hit (all of them)")
    if not isinstance(kind, str) or kind not in PARSERS:
        raise ValueError(f"question {name!r}: type is one of {', '.join(PARSERS)}; not {kind!r}")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError(f"question {name!r} needs instructions: the question's text (instructions, not question)")
    criteria = question.get("criteria")
    if kind == "noul" and criteria is not None and not (isinstance(criteria, dict) and set(criteria) == {"true", "false"}
                                                       and all(isinstance(v, str) and v.strip() for v in criteria.values())):
        raise ValueError(f"question {name!r}: a yes/no's criteria says what both answers mean: "
                         f'{{"true": "...", "false": "..."}}, or leave criteria out')
    if kind == "choice" and not (isinstance(criteria, dict) and criteria and all(
            isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in criteria.items())):
        raise ValueError(f"question {name!r}: a choice has criteria, {{option: what it means}}; answers goes with question, "
                         f"not with questions")
    if kind == "score" and not (isinstance(criteria, list) and 2 <= len(criteria) <= 10 and all(
            isinstance(c, str) for c in criteria)):
        raise ValueError(f"question {name!r}: a score has criteria, a list of 2 to 10 levels, lowest first")


def build(params: dict) -> tuple[dict, dict]:
    """The questions as the decider takes them, and the `hit` of each question that has one (`rules_of` makes the rules)."""
    raw: dict = {}
    if "questions" in params:
        raw = params["questions"]
        if not isinstance(raw, dict) or not all(isinstance(q, dict) for q in raw.values()):
            raise ValueError("questions is an object of question name to question, each a typed question (DESIGN.md §13.3)")
        raw = {name: dict(question) for name, question in raw.items()}
        for name, question in raw.items():
            _check(name, question)
    if "question" in params:
        question, answers = params["question"], params.get("answers")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question is text")
        name = question
        while name in raw:
            name += " "
        if answers is not None:
            if not isinstance(answers, list) or len(answers) < 2 or not all(isinstance(a, str) and a for a in answers):
                raise ValueError("answers is a list of two or more options")
            criteria = {a: None for a in answers}
            hit = params.get("hit", answers[0])
            if "none" not in criteria:
                if "none" in ([hit] if isinstance(hit, str) else hit if isinstance(hit, list) else []):
                    raise ValueError("hit names none, which is the 'none of these' option and never counts")
                criteria["none"] = "none of these"  # without one d1 names a wrong option (DESIGN.md §16)
            raw[name] = {"type": "choice", "instructions": question, "criteria": criteria, "hit": hit}
        else:
            raw[name] = {"type": "noul", "instructions": question, "hit": params.get("hit", True)}
    elif "answers" in params or "hit" in params:
        raise ValueError("hit goes inside each question of questions, and answers goes with question" if "questions" in params
                         else "answers and hit go with a question")
    if not raw:
        raise ValueError("ask needs a question, or questions")
    questions, hits = {}, {}
    for name, question in raw.items():
        hit = question.pop("hit", None)
        questions[name] = question
        if hit is not None and hit is not False:
            hits[name] = hit
    if not hits:
        raise ValueError("no question has a `hit`, so nothing could be found: say what counts as one")
    return questions, hits


def rules_of(questions: dict, hits: dict, about) -> dict:
    """The hit rule of each question that has a `hit`, with the defaults of the model `about` describes."""
    return {name: _rule(name, questions[name], hit, about) for name, hit in hits.items()}


def judge(questions: dict, rules: dict, answers: dict) -> tuple[bool, float, str, str | None]:
    """Whether every rule hits, the weakest p, the text of what every question said (`name: answer (p 0.95); ...`, which a
    label keeps, since a later `find` looks for the question's words) and the `said` of a result: what the answers add
    beyond p. With one question that is its answer alone, and nothing for a yes/no, whose answer is the sighting or the
    candidate itself; with more it is the label's text, whose names and p's say which question held a window back.
    `Unreadable` for an answer not in its shape."""
    found = []
    for name, rule in rules.items():
        if name not in answers:
            raise Unreadable(f"question {name!r}: there is no answer to it")
        found.append((name, *rule(read(name, questions[name], answers[name]))))
    label = "; ".join(f"{name}: {said} (p {p:.2f})" for name, _, p, said in found)
    hit, p = all(hit for _, hit, _, _ in found), min(p for _, _, p, _ in found)
    if len(found) > 1:
        return hit, p, label, label
    name, _, _, said = found[0]
    return hit, p, label, None if questions[name]["type"] == "noul" else said


# the parameters

class Plan:
    """The params, checked once."""

    def __init__(self, params: dict, now: float):
        unknown = sorted(set(params) - set(PARAMS))
        if unknown:
            raise ValueError(f"ask takes {', '.join(PARAMS)}, not {unknown}")
        self.questions, self.hits = build(params)
        context = params.get("context")
        if context is not None and (not isinstance(context, str) or len(context) > CONTEXT_MAX):
            raise ValueError(f"context is text of at most {CONTEXT_MAX:,} characters")
        self.context = context
        self.warnings: list[str] = []
        self.until = parse_when(params.get("until", "now"), now)
        if "since" in params:
            self.since = parse_when(params["since"], now)
        else:
            self.since = now - DEFAULT_SINCE_S
            self.warnings.append("no since: the last 24 h were searched; put the range in since and until")
        if self.until < self.since:
            raise ValueError("until is before since")
        self.sources = params.get("sources")
        if self.sources is not None and not (isinstance(self.sources, list) and all(isinstance(s, str) for s in self.sources)):
            raise ValueError("sources is a list of source names")
        self.scope = params.get("scope", "default")
        if self.scope not in ("default", "all"):
            raise ValueError("scope is default or all")
        self.limit = _count(params, "limit", 10, 1, LIMIT_MAX, integer=True)
        self.candidates = _count(params, "candidates", 5, 0, 20, integer=True)
        self.max_s = _count(params, "max_s", 600, 1, 3600)
        self.profile = {}  # what this ask changes of the reader's profile (reader.PROFILE has the defaults)
        if "every_s" in params:
            self.profile["every_s"] = _count(params, "every_s", None, 0)
        if "still_s" in params:
            self.profile["still_s"] = _count(params, "still_s", None, 0)
        if "max_px" in params:
            self.profile["max_px"] = _count(params, "max_px", None, 0, integer=True)
        self.merge_s = _count(params, "merge_s", 10, 0)
        self.max_sighting_s = _count(params, "max_sighting_s", 600, 0)
        self.oldest_first = params.get("oldest_first", False)
        if not isinstance(self.oldest_first, bool):
            raise ValueError("oldest_first is true or false")
        self.best_first = params.get("best_first", False)
        if not isinstance(self.best_first, bool):
            raise ValueError("best_first is true or false")


def _count(params: dict, key: str, default, low, high=None, integer: bool = False):
    value = params.get(key, default)
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or value < low or (high is not None and value > high)
            or (integer and int(value) != value)):
        raise ValueError(f"{key} is {'a whole number' if integer else 'a number'} from {low:g}"
                         + (f" to {high:g}" if high is not None else " up"))
    return int(value) if integer else value


def _fits(plan: Plan, about) -> None:
    """Refuse a question the decider does not take, before anything is scanned: a key no question has, a type it does not
    answer, or a key it does not take."""
    shown = ", ".join(("type", "instructions", "hit", *QUESTION_KEYS))
    for name, question in plan.questions.items():
        unknown = sorted(set(question) - {"type", "instructions", *QUESTION_KEYS})
        if unknown:
            raise ValueError(f"question {name!r} has {', '.join(unknown)}, which a question does not take; a question "
                             f"takes {shown}")
        if question["type"] not in about.types:
            raise ValueError(f"question {name!r} is a {question['type']}, which {about.name} does not answer; it "
                             f"answers {', '.join(about.types)}")
        refused = [key for key in QUESTION_KEYS if key in question and key not in about.question_keys]
        if refused:
            takes = ", ".join(key for key in QUESTION_KEYS if key in about.question_keys)
            raise ValueError(f"question {name!r} has {', '.join(refused)}, which {about.name} does not take; "
                             + (f"it takes {takes} (besides type, instructions and hit)" if takes
                                else "it takes only type, instructions and hit"))


def window_seconds(kind: str, profile: dict, about, questions: int = 1) -> float:
    """What one window of this kind costs the decider, with `profile` (`job.profile`: this ask's, cut to the model's cap) and
    its `about`, for `questions` questions: the model reads the window once for each `questions_per_read` of them (0: any
    number). A picture of 0 pixels is as stored, which only a model with no cap reads."""
    passes = math.ceil(questions / about.questions_per_read) if about.questions_per_read else 1
    if kind in ("video", "image"):
        return about.picture_seconds(profile["max_px"] or math.inf) * passes + DECODE_S
    if kind == "audio":
        return about.sound_s * passes
    if kind == "file":
        return about.file_s * passes
    return about.char_s * profile["text_chars" if kind == "text" else "records_chars"] * passes


def lint(plan: Plan, counts: dict, profile: dict, about) -> list[str]:
    """Warnings at the start; the rules are this script's, to change."""
    out = list(plan.warnings)
    words = " ".join(q.get("instructions", "") for q in plan.questions.values())
    if TIME_WORDS.search(words) or TIME_WORDS.search(plan.context or ""):
        out.append("a time word in the question or context: put the time in since and until, not in the question")
    if any(JOINED.search(q.get("instructions", "")) for q in plan.questions.values()):
        out.append("two things joined by 'and' or 'or': ask about one thing, so each answer means one thing")
    if about.questions_per_read == 1 and len(plan.questions) > 1:
        out.append(f"{len(plan.questions)} questions: {about.name} reads a window once for each question, so this takes "
                   f"{len(plan.questions)} times as long; ask fewer")
    elif about.questions_per_read and len(plan.questions) > about.questions_per_read:
        out.append(f"{len(plan.questions)} questions: {about.name} reads the window again for each "
                   f"{about.questions_per_read}, so this takes several times as long; ask fewer")
    kinds = [kind for kind in ("video", "image", "audio", "file") if counts.get(kind) and kind not in about.extracts_from]
    unread = [name for name, q in plan.questions.items() if q.get("type") in ("text", "date")
              or (q.get("type") == "number" and not q.get("range"))]
    if unread and kinds:
        out.append(f"question {unread[0]!r} is a text, date or number with no range, and {about.name} reads those only "
                   f"from {' and '.join(sorted(about.extracts_from))}"
                   f"; it will find nothing in {', '.join(kinds)}")
    if not about.speech and counts["audio"] and SPEECH.search(words):
        out.append(f"speech words over audio: {about.name} hears sounds better than speech; ask about a sound, not what "
                   f"was said")
    seconds = sum(window_seconds(kind, profile, about, len(plan.questions)) * n for kind, n in counts.items())
    if seconds > plan.max_s:
        estimate = f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min"
        out.append(f"about {sum(counts.values()):,} windows, about {estimate} against max_s of "
                   f"{plan.max_s:g} s: narrow the range or the sources, or raise every_s"
                   + ("" if "picture_s" in about.measured else " (its costs are assumed)")
                   + ("; best_first ranks only what is scanned before max_s" if plan.best_first else ""))
    return out


# the scan

def _handle(w) -> dict:
    if w.item is not None:
        return {"item": w.item.id, "at": round(w.at or 0.0, 3)}
    return {"source": w.source, "t": iso(w.t0)}


def _sighting(w, p: float, label: str, said: str | None, seq: int) -> dict:
    return {"source": w.source, "t0": w.t0, "t1": w.t1, "p": p, "label": label, "said": said, "window": w, "capped": False,
            "seq": seq}


def _gap(a0: float, a1: float, b0: float, b1: float) -> float:
    """The seconds between two time spans; 0 when they overlap."""
    return max(0.0, max(a0, b0) - min(a1, b1))


def _rank(s: dict) -> tuple[float, int]:
    """`best_first`'s order, highest first: the best `p`, then the sighting scanned first."""
    return s["p"], -s["seq"]


def _candidates(pool: dict, sightings: list[dict], plan: Plan) -> list[dict]:
    """The best non-hits, one per moment, best first (ties in scan order): a window is left out when it is within
    `merge_s` of a better candidate or of a sighting of its source. The pool's entries are
    `(p, -n, source, t0, t1, said, kind, look)`."""
    taken = [(s["source"], s["t0"], s["t1"]) for s in sightings]
    out = []
    for p, _, source, t0, t1, said, kind, look in sorted(pool.values(), key=lambda e: (-e[0], -e[1])):
        if len(out) >= plan.candidates:
            break
        if any(s == source and _gap(a0, a1, t0, t1) <= plan.merge_s for s, a0, a1 in taken):
            continue
        taken.append((source, t0, t1))
        out.append({"source": source, **when(t0, t1), "p": round(p, 3), **({} if said is None else {"said": said}),
                    "kind": kind, "look": look})
    return out


def _public(job, s: dict) -> dict:
    """A sighting as the result gives it: `said`, `capped` and `project` only when there is something to say."""
    w = s["window"]
    project = job.store.project_name(w.project)
    out = {"source": s["source"], **when(s["t0"], s["t1"]), "p": round(s["p"], 3),
           **({} if s["said"] is None else {"said": s["said"]}), "kind": w.kind,
           **({"capped": True} if s["capped"] else {}), **({} if project is None else {"project": project}),
           "look": _handle(w)}
    if job.store.has_clock(s["source"]):
        found = job.store.program_time(s["source"], _shown_at(w))
        if found:
            out.update(run=found[0], tp=found[1])
    return out


def _shown_at(w) -> float:
    """The time of what a window's handle shows: the frame or sound at `at` of a segment, else the window's start (a
    picture is an instant, and a version has no position in time)."""
    return w.item.t0 + w.at if w.item is not None and w.item.kind in ("video", "audio") and w.at is not None else w.t0


def _clock(t0: float, t1: float) -> str:
    span = when(t0, t1)
    return span["t"] if "t" in span else f"{span['t0']} to {span['t1']}"


def run(job) -> None:
    started = time.monotonic()
    scanning = None  # when the decider first answered: max_s counts from there, so loading it is not counted
    plan = Plan(job.params, job.now)
    about = job.about()
    _fits(plan, about)
    job.check(plan.questions)  # what the model itself refuses (ValueError: bad_request), before a window is scanned
    rules = rules_of(plan.questions, plan.hits, about)
    profile = job.profile(plan.profile)
    if plan.max_sighting_s and profile["still_s"] > plan.max_sighting_s:  # no window outgrows the cap
        plan.profile["still_s"] = profile["still_s"] = plan.max_sighting_s
    counts = job.estimate(plan.since, plan.until, plan.sources, plan.scope, plan.profile)
    warnings = lint(plan, counts, profile, about)
    searched = job.searched(plan.since, plan.until, plan.sources, plan.scope)
    expected = sum(counts.values())
    job.update(sightings=[], candidates=[], searched=searched, stopped=None, warnings=[],
               progress={"windows": 0, "of_about": expected, "seconds": 0.0, "decider": job.decider()["state"]})
    for text in warnings:
        job.warn(text)
    newest = not plan.oldest_first
    open_: dict[str, dict] = {}  # per source, the sighting being built
    done: list[dict] = []  # closed sightings; best_first keeps only the best `limit` of them
    order = _rank if plan.best_first else _order(newest)  # the order of the sightings, used with reverse=True
    pool: dict = {}  # the best non-hit of each source and merge_s of the clock, as `_candidates` takes it
    stats = {"scanned": 0, "stopped": "range_end"}
    skipped = {"unfit": [0, None], "unreadable": [0, None]}  # windows left out, and the first reason for each
    last_report = 0.0
    changed = False

    def close(source: str) -> None:
        done.append(open_.pop(source))
        if plan.best_first and len(done) > plan.limit:  # a closed sighting holds its decoded picture: keep no more than the limit
            done.sort(key=order, reverse=True)
            worst = done.pop()
            # windows come in time order, so no later non-hit of its source lands near it: the pool's are the ones to clear
            span = (worst["t0"], worst["t1"])
            for key in [k for k, e in pool.items() if e[2] == worst["source"] and _gap(e[3], e[4], *span) <= plan.merge_s]:
                del pool[key]

    def report(force: bool = False) -> None:
        nonlocal last_report, changed
        if not force and not changed and time.monotonic() - last_report < UPDATE_S:
            return
        last_report, changed = time.monotonic(), False
        sightings = done + list(open_.values())
        shown = sorted(sightings, key=order, reverse=True)[:plan.limit]
        job.update(sightings=[_public(job, s) for s in shown], candidates=_candidates(pool, sightings, plan),
                   progress={"windows": stats["scanned"], "of_about": max(expected, stats["scanned"]),
                             "seconds": round(time.monotonic() - started, 1), "decider": job.decider()["state"]})

    try:
        for w in job.windows(plan.since, plan.until, plan.sources, plan.scope, newest_first=newest, profile=plan.profile):
            for source, o in list(open_.items()):  # windows come in time order, so one this far off ends a sighting
                if (o["t0"] - w.t1 if newest else w.t0 - o["t1"]) > plan.merge_s:
                    close(source)
            if len(done) >= plan.limit and not plan.best_first:
                stats["stopped"] = "limit"
                break
            if scanning is not None and time.monotonic() - scanning >= plan.max_s:
                stats["stopped"] = "max_s"
                break
            if job.cancelled:
                stats["stopped"] = "cancelled"
                break
            state = "\n\n".join(part for part in (plan.context, w.text) if part) or None
            try:
                answers = job.decide(state, plan.questions, [w.media] if w.media is not None else [])
            except (ValueError, TypeError) as error:  # a window the questions do not fit, as a box question over sound
                skipped["unfit"][0] += 1
                skipped["unfit"][1] = skipped["unfit"][1] or f"a {w.kind} window: {type(error).__name__}: {error}"
                continue
            finally:
                if scanning is None:
                    scanning = time.monotonic()
            stats["scanned"] += 1
            try:
                hit, p, label, said = judge(plan.questions, rules, answers)
            except Unreadable as error:
                skipped["unreadable"][0] += 1
                skipped["unreadable"][1] = skipped["unreadable"][1] or f"a {w.kind} window: {error}"
                continue
            if not hit:
                o = open_.get(w.source)  # a sighting only grows, so a window near its own source's is never a candidate
                if plan.candidates and not (o is not None and _gap(o["t0"], o["t1"], w.t0, w.t1) <= plan.merge_s):
                    key = (w.source, w.t0 // plan.merge_s if plan.merge_s else w.t0)
                    entry = (p, -stats["scanned"], w.source, w.t0, w.t1, said, w.kind, _handle(w))
                    if key not in pool or entry > pool[key]:
                        pool[key] = entry
                        if len(pool) > CANDIDATE_POOL:
                            del pool[min(pool, key=pool.get)]
                report()
                continue
            o = open_.get(w.source)
            if o is not None and plan.max_sighting_s and max(o["t1"], w.t1) - min(o["t0"], w.t0) > plan.max_sighting_s:
                o["capped"] = True  # this run goes on: close the sighting here and let the next window start another
                close(w.source)
                o = None
            if o is None:
                open_[w.source] = _sighting(w, p, label, said, stats["scanned"])
            else:
                o["t0"], o["t1"] = min(o["t0"], w.t0), max(o["t1"], w.t1)
                if p > o["p"]:
                    o.update(p=p, label=label, said=said, window=w)
            changed = True
            report()
    except job.Cancelled:
        stats["stopped"] = "cancelled"
    for reason, (n, first) in skipped.items():
        if n:
            job.warn(f"{n} windows were skipped, so the scan did not see them: "
                     + ("they did not fit the questions" if reason == "unfit" else f"{about.name}'s answer could not be read")
                     + f"; the first was {first}")
    done += open_.values()
    open_.clear()
    done.sort(key=order, reverse=True)
    found = done[:plan.limit]
    model = job.decider()["model"]
    for s in found:  # one label on the best window of each sighting
        w = s["window"]
        job.label(s["t0"], s["t1"], f"{s['label']}, {_clock(s['t0'], s['t1'])}" + (" (capped)" if s["capped"] else ""),
                  source=s["source"], item=w.item.id if w.item else None,
                  meta={"p": round(s["p"], 3), "at": w.at, "kind": w.kind, "model": model, "capped": s["capped"]})
    job.update(stopped=stats["stopped"])
    report(force=True)


def _order(newest: bool):
    return (lambda s: (s["t1"], s["t0"])) if newest else (lambda s: (-s["t0"], -s["t1"]))


def note(result: dict) -> str:
    """The line `mms ask` prints to stderr on how the scan went."""
    progress = result.get("progress", {})
    text = f"scanned {progress.get('windows', 0)} windows in {progress.get('seconds', 0)} s"
    if result.get("stopped"):
        text += f", stopped at {result['stopped']}"
    return text + "".join(f"\nwarning: {warning}" for warning in result.get("warnings", []))


# the command line

def cli(argv: list[str]) -> dict:
    """`mms ask ...` flags as `params`."""
    parser = argparse.ArgumentParser(prog="mms ask", description="Ask the memory typed questions about what was recorded.")
    parser.add_argument("question", nargs="?", help="one thing that could be seen or heard; yes/no unless --answers")
    parser.add_argument("--answers", action="append", metavar="A,B", help="the options of a choice question (two or more)")
    parser.add_argument("--hit", help="what counts: a P(yes) from 0 to 1 (default: the model's own, which the ask tool's "
                                      "description gives), or the options that count (default the first of --answers)")
    parser.add_argument("--context", help="state text given with each window: what the source shows")
    parser.add_argument("--questions", metavar="FILE", help="a JSON file of typed questions (DESIGN.md §13.3), each with a hit")
    parser.add_argument("--since", help="from: 'now', an age such as 2h or 7d, epoch seconds, or ISO 8601 (default 24 h ago)")
    parser.add_argument("--until", help="to: as --since (default now)")
    parser.add_argument("--source", action="append", dest="sources", help="only this source (repeatable)")
    parser.add_argument("--scope", choices=["default", "all"])
    parser.add_argument("--limit", type=int, help=f"sightings to find, 1 to {LIMIT_MAX} (default 10)")
    parser.add_argument("--candidates", type=int, help="the best windows that did not count to give as well, 0-20 (default 5)")
    parser.add_argument("--every-s", type=float, help="video: seconds between frames (default 1; 0 for every frame)")
    parser.add_argument("--still-s", type=float, help="video: a frame that looks the same as the one that began a window joins it, "
                                                      "for at most this many seconds (default 30; 0 reads every frame the step picks)")
    parser.add_argument("--max-px", type=int, help="the most pixels of a picture the model sees (default 230400, 640x360, "
                                                   "unless the model sets its own; never above the model's own cap, "
                                                   "which 0 gives): raise it for small text")
    parser.add_argument("--merge-s", type=float, help="hits this close are one sighting (default 10)")
    parser.add_argument("--max-sighting-s", type=float, help="a sighting is cut here and marked capped (default 600; 0 for no cap)")
    parser.add_argument("--max-s", type=float, help="stop scanning after this many seconds, counted from the model's first answer (default 600)")
    parser.add_argument("--oldest-first", action="store_true", default=None, help="scan from the oldest instead of the newest")
    parser.add_argument("--best-first", action="store_true", default=None,
                        help="read the whole range and give the --limit sightings with the highest p, best first")
    args = parser.parse_args(argv)
    params: dict = {}
    if args.question is not None:
        params["question"] = args.question
    if args.answers:
        params["answers"] = [part.strip() for text in args.answers for part in text.split(",") if part.strip()]
    if args.hit is not None:  # text, read by the question's own rule: "0.5" is a threshold, "2" may be an option
        params["hit"] = [part.strip() for part in args.hit.split(",") if part.strip()]
        if len(params["hit"]) == 1:
            params["hit"] = params["hit"][0]
    if args.questions:
        try:
            params["questions"] = json.loads(open(args.questions, encoding="utf-8").read())
        except (OSError, ValueError) as error:
            parser.error(f"cannot read --questions {args.questions}: {error}")
    for key in ("context", "since", "until", "sources", "scope", "limit", "candidates", "every_s", "still_s", "max_px", "merge_s",
                "max_sighting_s", "max_s"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    if args.oldest_first:
        params["oldest_first"] = True
    if args.best_first:
        params["best_first"] = True
    return params
