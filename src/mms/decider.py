"""The decider (DESIGN.md §10, §13.3): any object with `about` (an `About`, what mms must know of it), `decide(state,
questions, media)`, which answers typed questions about a window as `{name: answer}`, and optional `load()` and
`unload()`. `load_decider` makes one from a `[decider] model`; `gpu.py` is the daemon's queue in front of it.

A model is a folder that one of `BACKENDS` claims, or `python:<file.py>`, a file that defines `ABOUT` and `decide`.
Nothing heavy is imported or loaded until the first window is decided.
"""

from __future__ import annotations

import bisect
import importlib
import importlib.util
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from mms.errors import ApiError

TYPES = ("noul", "choice", "score", "text", "number", "date", "box", "boxes")  # the question types of ask's format
MEDIA_KINDS = frozenset({"video", "image", "audio"})  # window kinds besides text and records, which are always read
WINDOW_KINDS = MEDIA_KINDS | {"text", "records", "file"}
QUESTION_KEYS = ("criteria", "multiple", "range", "step", "unit")  # what a question may carry besides type, instructions, hit
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")  # a Hugging Face repo, owner/name
MEASURABLE = ("picture_s", "sound_s", "char_s", "file_s", "hit")  # the About facts that are timed or tried on the developer's laptop
BACKENDS = ("mms.decider_d1:D1", "mms.decider_d1_3b:D1_3B", "mms.decider_d3:D3",
            "mms.decider_pplx:Pplx", "mms.decider_h2o:H2o")  # folder backends, tried in order: the first whose
# `claims(folder)` is an About (a backend also has `abouts`, every About it can claim, and `reads_folder`, one phrase for the
# error when none claims)


@dataclass(frozen=True, kw_only=True)
class About:
    """What mms must know of a decider before loading it (DESIGN.md §13.3)."""

    name: str  # for messages: the model's name
    reads: frozenset = MEDIA_KINDS  # the media window kinds it takes
    files: tuple = ()  # media types of versions it takes whole; an entry ending "/" or "." is a prefix
    types: tuple = TYPES  # the question types it answers
    extracts_from: frozenset = frozenset({"text", "records"})  # kinds it reads a text, date or unranged number from
    speech: bool = True  # whether it understands what is said
    audio_rate: int = 16000  # Hz of the sound it is handed
    max_px: int = 0  # its own picture cap, 0 for none; `reader.profile_of` cuts a scan's `max_px` to it, and 0 is the cap
    questions_per_read: int = 0  # questions it reads in one pass, 0 for any number
    picture_s: tuple = ((230_400, 0.05),)  # (pixels, seconds) points; linear between, flat outside
    sound_s: float = 0.05
    char_s: float = 0.00002  # seconds per character of state
    file_s: float = 0.5  # a whole-file window
    hit: float = 0.8  # a yes/no question's threshold when its `hit` is true or left out
    question_keys: frozenset = frozenset(QUESTION_KEYS)  # the keys a question may carry besides type and instructions
    notes: str = ""  # a file name in skills/mms-questions/models/, or "" for none
    repo: str = ""  # the Hugging Face repo (owner/name) its folder is downloaded from; empty for none
    repo_gb: float = 0.0  # the repo's size in GB, for `mms model list` to show before a download
    measured: frozenset = frozenset()  # which of MEASURABLE were measured on the developer's laptop; the rest are assumed
    profile: Mapping = field(default_factory=dict)  # overrides of reader.PROFILE (the reader checks the keys); kept read-only

    def __post_init__(self):
        def put(name, value):
            object.__setattr__(self, name, value)

        def listed(name, value, allowed=None):
            if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
                raise ValueError(f"About.{name} is a collection, not {value!r}")
            value = tuple(value)
            if allowed is not None and set(value) - set(allowed):
                raise ValueError(f"About.{name} has {sorted(map(str, set(value) - set(allowed)))}; it is among "
                                 f"{', '.join(sorted(allowed))}")
            return value

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"About.name is a non-empty text, not {self.name!r}")
        put("reads", frozenset(listed("reads", self.reads, MEDIA_KINDS)))
        put("types", listed("types", self.types, TYPES))
        if not self.types:
            raise ValueError("About.types names at least one question type")
        put("extracts_from", frozenset(listed("extracts_from", self.extracts_from, WINDOW_KINDS)))
        if not self.extracts_from and {"text", "date", "number"} & set(self.types):
            raise ValueError("About.extracts_from names at least one kind when About.types has text, date or number: "
                             "it is where those are read from")
        if not isinstance(self.speech, bool):
            raise ValueError(f"About.speech is true or false, not {self.speech!r}")
        put("files", listed("files", self.files))
        if not all(isinstance(entry, str) and entry for entry in self.files):
            raise ValueError(f"About.files is media types, not {self.files!r}")
        if isinstance(self.audio_rate, bool) or not isinstance(self.audio_rate, int) or self.audio_rate <= 0:
            raise ValueError(f"About.audio_rate is a whole number of Hz above 0, not {self.audio_rate!r}")
        for name in ("max_px", "questions_per_read"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"About.{name} is a whole number from 0, not {value!r}")
        for name in ("sound_s", "char_s", "file_s", "repo_gb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"About.{name} is a number from 0, not {value!r}")
        if isinstance(self.hit, bool) or not isinstance(self.hit, (int, float)) or not 0 < self.hit <= 1:
            raise ValueError(f"About.hit is a number above 0 and at most 1, not {self.hit!r}")
        put("question_keys", frozenset(listed("question_keys", self.question_keys, QUESTION_KEYS)))
        if (not isinstance(self.notes, str) or self.notes in (".", "..") or "/" in self.notes or "\\" in self.notes
                or self.notes != self.notes.strip()):
            raise ValueError(f"About.notes is a plain file name, or empty for none, not {self.notes!r}")
        if not isinstance(self.repo, str) or (self.repo and not REPO.fullmatch(self.repo)):
            raise ValueError(f"About.repo is a Hugging Face repo, owner/name, or empty for none, not {self.repo!r}")
        put("measured", frozenset(listed("measured", self.measured, MEASURABLE)))
        points = [tuple(point) if isinstance(point, (list, tuple)) else point for point in listed("picture_s", self.picture_s)]
        if (not points or any(not isinstance(p, tuple) or len(p) != 2
                              or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                                         for x in p)
                              for p in points)
                or any(second < 0 for _, second in points) or any(a[0] >= b[0] for a, b in zip(points, points[1:]))):
            raise ValueError("About.picture_s is (pixels, seconds) points, at least one, with increasing pixels and no "
                             f"negative seconds, not {self.picture_s!r}")
        put("picture_s", tuple(points))
        if not isinstance(self.profile, Mapping):
            raise ValueError(f"About.profile is a dict of reader.PROFILE keys, not {self.profile!r}")
        put("profile", MappingProxyType(dict(self.profile)))  # a script reads it, and cannot change it for every job

    def takes(self, media: str | None) -> bool:
        """Whether a version of this media type is taken whole: an exact entry of `files`, or a prefix entry."""
        return bool(media) and any(media == entry or (entry[-1] in "/." and media.startswith(entry))
                                   for entry in self.files)

    def picture_seconds(self, pixels: float) -> float:
        """What one picture of this many pixels costs: `picture_s` interpolated linearly, flat outside its points."""
        points = self.picture_s
        if pixels <= points[0][0]:
            return points[0][1]
        at = bisect.bisect_left([x for x, _ in points], pixels)
        if at >= len(points):
            return points[-1][1]
        (x0, y0), (x1, y1) = points[at - 1], points[at]
        return y0 + (y1 - y0) * (pixels - x0) / (x1 - x0)


def backends() -> list:
    return [getattr(importlib.import_module(entry.partition(":")[0]), entry.partition(":")[2]) for entry in BACKENDS]


def claim(folder: Path) -> tuple | None:
    """The backend class that claims a model folder, and its About; None for a missing folder or no claim."""
    if not folder.is_dir():
        return None
    for backend in backends():
        about = backend.claims(folder)
        if about is not None:
            return backend, about
    return None


def claimed(folder: Path) -> tuple:
    """`claim`, or a `ValueError` saying the folder is missing or no backend reads it."""
    if not folder.is_dir():
        raise ValueError(f"there is no model folder at {folder}")
    found = claim(folder)
    if found is None:
        raise ValueError(f"no decider reads the model in {folder}: " + "; ".join(b.reads_folder for b in backends())
                         + "; a python:<file.py> can be any decider")
    return found


def load_decider(spec: str, home: Path | None = None, *, url: str | None = None, envs: str | None = None):
    """The decider for a model folder (the first of `BACKENDS` that claims it) or for `python:<file.py>`, which is told
    the `home` it serves. `url` is where the owner serves the model (a backend that is not served refuses it, as does
    `python:`), and `envs` is where the backends' envs are."""
    if spec.startswith("python:"):
        if url is not None:
            raise ValueError(f"a python: decider takes no [decider] url, but it is {url!r}: remove it")
        return PythonFile(spec[len("python:"):], home)
    folder = Path(spec).expanduser()
    backend, _ = claimed(folder)
    return backend(folder, url=url, envs=envs)


def about_of(spec: str | None) -> About | None:
    """The facts of the model a spec names, read from its folder alone: nothing is constructed, loaded or started. None
    for no spec, a `python:` file, a missing folder, or a folder no backend claims."""
    if not spec or spec.startswith("python:"):
        return None
    found = claim(Path(spec).expanduser())
    return None if found is None else found[1]


def check_of(spec: str | None, questions: dict) -> None:
    """Raise `ValueError` naming what in `questions` the model a spec names cannot take, before any window is read. Like
    `about_of` it constructs and loads nothing: it is the claiming backend's static `check(about, questions)`, or for a
    `python:` file its `check(questions)` if it defines one. No spec, or a folder no backend claims, checks nothing (the
    first window fails with the reason)."""
    if not spec:
        return
    if spec.startswith("python:"):
        try:
            file = PythonFile(spec[len("python:"):])
        except Exception as error:
            raise ApiError("decider", f"the decider could not load: {type(error).__name__}: {error}") from None
        file.check(questions)
        return
    found = claim(Path(spec).expanduser())
    if found is not None:
        found[0].check(found[1], questions)


def known_abouts() -> list[About]:
    """Every About every backend can claim, in `BACKENDS` order."""
    return [about for backend in backends() for about in backend.abouts]


def known(name: str) -> tuple | None:
    """The backend class and About of the model called `name` (in any case) among every backend's; None if none is."""
    for backend in backends():
        for about in backend.abouts:
            if about.name.lower() == name.lower():
                return backend, about
    return None


class PythonFile:
    """A decider in a Python file: it defines `ABOUT` (an `About`) and `decide(state, questions, media)`, and may define
    `load()` and `unload()`. The loader sets `home` in it. Its top level stays cheap, since reading `about` runs it."""

    def __init__(self, file: str, home: Path | None = None):
        path = Path(file).expanduser()
        found = importlib.util.spec_from_file_location(f"mms_decider_{path.stem}", path)
        if found is None or found.loader is None:
            raise ValueError(f"no python decider at {file!r}")
        module = importlib.util.module_from_spec(found)
        found.loader.exec_module(module)
        module.home = home
        if not isinstance(getattr(module, "ABOUT", None), About):
            raise ValueError(f"{file} defines no ABOUT, a mms.decider.About")
        if not callable(getattr(module, "decide", None)):
            raise ValueError(f"{file} defines no decide(state, questions, media)")
        self._module = module
        self.about = module.ABOUT

    def load(self) -> None:
        if callable(getattr(self._module, "load", None)):
            self._module.load()

    def unload(self) -> None:
        if callable(getattr(self._module, "unload", None)):
            self._module.unload()

    def decide(self, state, questions: dict, media: list | None = None) -> dict:
        return self._module.decide(state, questions, media or [])

    def check(self, questions: dict) -> None:
        """The file's `check(questions)`, which raises `ValueError` for a question it cannot take; none is no check."""
        if callable(getattr(self._module, "check", None)):
            self._module.check(questions)
