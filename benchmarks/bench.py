"""The benchmark: does Claude Code or Codex answer an owner's questions better with mms than without it, for each decision
model, on the kinds of data that model reads (TESTBED.md, Benchmark). It runs the agents of `benchmarks/claude_run.py` and
`benchmarks/codex_run.py` over the synthetic week of `benchmarks/haystack.py`, and reports the paired comparison.

    python benchmarks/bench.py build   [--scale NAME] [--seed N] [--out DIR] [--temp DIR] [--no-probe]   # a scale of haystack.SCALES
    python benchmarks/bench.py decider RUN [--models NAME,...] [--models-dir DIR]                     # the GPU, alone
    python benchmarks/bench.py agents  RUN [--agents claude,codex] [--models NAME,...|none] [--jobs N] [--subset all|pilot]
                                      [--controls MODEL|all] [--anyway] [--dry-run]
    python benchmarks/bench.py report  RUN [RUN ...]

`build` writes the week (read-only, in the temp folder), loads it into a snapshot home under `~/mms-bench/<run>/` and runs the
shortcut probe (`benchmarks/shortcuts.py`: no colour filter, frame difference, band filter or file size may single out a needle) over it,
writing `shortcuts.json` and `shortcuts.md` in the run folder; a probe that fails makes `build` exit 1 with the run kept, and
`--no-probe` skips it (the run and its report say so). `decider`
asks each model each needle's yes/no over its source and range, with no agent: a `last` question for 5 sightings, newest
first, the others for up to 20 (counts 50) and 5 candidates, each allowed 1,800 s. `agents` runs each question as one exec: the
files arm once per agent (the agent reads the files with its own tools), the mms arm once per agent and model, in its own copy
of the snapshot; a model that takes over 0.15 s a window runs one exec at a time (parallel execs would share its GPU queue),
and only `--controls` (d1-3B) is also asked the text and records questions, which do not depend on the model. `--subset pilot`
runs the media questions and 3 controls. Each exec is a working folder, a result file and an exec folder of its own, so a command that stops (the
plan's quota, Ctrl+C) is continued by running it again: a question with a done result is not asked twice (an exec that
escaped, timed out or was capped is a wrong answer; `setup`, `limit`, `auth` and `error` ones are asked again). Each queue runs
in ascending `until`. In an mms home the questions of file-version sources (AS_OF) run one at a time, so do two about one
template's needle, and an exec that may need over 300 GPU seconds (`gpu_estimate`) runs alone, when the others are done. What an
exec leaves in the home (its labels, the items it kept, its project and session rows, its running jobs) is taken out when it
ends (`scrub_home`), so each exec starts from the snapshot's state. A run whose probe failed is refused unless `--anyway`.
`--dry-run` writes
the first exec's files per agent and arm, prints the command lines and runs nothing. `report` pairs the arms by question, and
pools runs (another seed adds new needles): accuracy with and without mms, by how vague the question's time hint is (`level`:
part of a day, day, week), how each arm worked (asks, windows, GPU seconds, views), the cost of a correct answer, the decider
tier's verdict for each template, a summary per model and one row per question. It never writes to `~/.claude`, `~/.codex`
or `~/.mms`.

The run folder (`<run>` is `%Y%m%d-%H%M-<scale>-s<seed>`):

    ~/mms-bench/<run>/run.json, data-manifest.json, models.json, shortcuts.json, shortcuts.md, report.md, report.json
                     homes/snapshot, homes/<agent>-<model>, homes/decider-<model>
                     results/<agent>/files|mms-<model>/<qid>.json, results/decider/<model>/<qid>.json
                     execs/<agent>/files|mms-<model>/<qid>/        the exec's own files (prompt, argv, events, tools, final)
    <temp>/mmsb-<tag>-data/   the haystack files, read-only; mmsb-<tag>-bin/ the ffmpeg the files arm has on PATH;
                     mmsb-<tag>-w<k>/  working slot k, an exec's cwd, emptied before and after each exec
                     (<tag> is random, in run.json: the run's name holds its seed and scale)

The files arm is kept away from the answers: the data and the working folders are in the temp folder, the results in
`~/mms-bench`, every tool input is checked against `Run.deny()` (the run, the results, the repo, the owner's agent setup, the other
slots, and the words `haystack`, `truth` and `plan(`), and the data is checked after each exec.

`Run`'s `prompt`, `view`, `estimate`, `needle` and `extra` are the dataset's own parts: another dataset (`benchmarks/egolife.py`)
subclasses `Run` and reuses `run_arm`, `copy_home`, `record` and the statistics. Its arms are `blind` (no tools, no recordings),
`files` and `mms` (`ARMS`)."""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import http.client
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import statistics
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from contextlib import closing
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as clock
from pathlib import Path
from typing import ClassVar

import claude_run
import codex_run
import grade

from mms import config, decider, fetch, modelenv, reader
from mms.decider import About
from mms.project import MARKER
from mms.scripts import ask as ask_script

# haystack needs the tests' day and fake_decider, and conftest has the daemon helpers: a script run from benchmarks/ has only
# benchmarks/ on its path
TESTS_DIR = str(Path(__file__).resolve().parents[1] / "tests")
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)  # last, so no file of the tests' shadows a module of the standard library or the env

import haystack  # noqa: E402
import shortcuts  # noqa: E402
from conftest import Api, Daemon, wait_for, write_config  # noqa: E402

OUT = Path.home() / "mms-bench"
MODELS_DIR = Path(os.environ.get("MMS_MODELS") or config.default_home() / "models").expanduser()  # where `mms install` puts models
JOBS, JOBS_MAX = 3, 4  # execs at once; more at once have crashed the GPU driver
STOP_AT = 95.0  # percent of the week's plan use at which an agent stops
EXEC_MIN = 2.5  # minutes an exec takes, for the estimate printed before a run
BLIND_MIN = 1.0  # ... and a blind one, which runs no tool
SNAPSHOT_CONFIG = "[memory]\nfree_gb = 0\nremember_max_mb = 500\n\n[decider]\nidle_s = 3600\n"
# `mms-bench` is matched as a path part (the repo's worktree `codex-mms-benchmarks` is not one): before a slash, a quote or a space
DENY_WORDS = (".claude", ".codex", ".mms", "/mms-bench/", '/mms-bench"', "/mms-bench ", "/mms-bench'")
# what no command or path of the files arm may hold (an `in:` entry is checked in what the agent runs, not in what it reads): the
# generator of the week and its truth, as code reaches them (a mention in prose is no escape). An agent that runs
# `haystack.plan(seed, scale)` from the repo has every answer. The quotes are as `codex_run.reached` reads a command: a `"` after
# a backslash is `/"`.
SOURCE_WORDS = tuple(codex_run.IN_ONLY + word for word in (
    "haystack.py", "import haystack", "from haystack", "haystack.plan", "['truth']", '[/"truth/"]', "get('truth'", 'get(/"truth/"'))
AS_OF = ("notes", "config")  # versions stay current: an ask over a later slot reads and labels an earlier one (a photo is an instant)
MEDIA = ("image", "video", "audio")  # the kinds of data that depend on the model: the rest is text and records
DONE_KINDS = ("escaped", "timeout", "capped")  # problems that are a wrong answer: the exec ran and did not answer
STOP_KINDS = ("setup", "limit", "auth")  # problems after which the agent stops at once: nothing would count
ERRORS_STOP = 3  # `error:` results in a row after which the agent stops
COPY_SKIPS = ("daemon.json", "daemon.log*", "daemon.test.log", "daemon.stop", "looks", "fake-decider.jsonl")
NOT_RESULTS = ("warm", "ambient-labels")  # json files in a results folder that are not one question's
ORACLE_PROJECT, WARM_PROJECT = "id:bench-oracle", "id:bench-warm"  # so the labels of these asks are in no default scope
AGENTS = ("claude", "codex")
ARMS = ("blind", "files", "mms")  # blind: no tools and no recordings; files: the agent's own tools on the files; mms: mms's tools
MEDIA_FOLDERS = ("files", "streams")  # what a home keeps of the recordings: a copy of a home links them instead of copying
LEVELS = ("part", "day", "week", "time")  # how vague a question's time hint is; `time` is an exact time (config-asof)
CROSS_LEVELS = LEVELS[:3]  # the levels the crossover looks at, from the least vague
LEAD, HALF = 0.20, 0.5  # the crossover level: mms leads by this share of questions, or costs this share of files' cost, or less
NEAR_P = 0.5  # a needle's best p from which a miss is a near miss
ORACLE_LIMIT, COUNT_LIMIT, ORACLE_CANDIDATES, ORACLE_MAX_S = 20, 50, 5, 1800  # an oracle ask's sightings, a count's, candidates, seconds
LAST_LIMIT = 5  # a `last` ask's sightings, newest first: as many as one `look` checks (up to 8 moments)
SLOW_S = 0.15  # a model that takes more seconds a window runs one exec at a time: parallel execs share its GPU queue
CONTROLS_MODEL = "d1-3B"  # the model whose mms arm is also asked the text and records questions: they do not depend on the model
CONTROL_TYPES = ("text", "records")
SUBSETS = ("all", "pilot")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")  # a file Claude's `Read` shows as a picture
SOURCE_KINDS = {"mic": "audio", "logs": "records", "notes": "text", "config": "text"}  # the window kind a source costs; else video
MIN_DISCORDANT = 10  # pairs right in one arm only (b + c) from which a difference gets an interval: below it only McNemar speaks
MIN_LEVEL_N = 10  # paired questions a level needs before its crossover is told
MIN_CORRECT = 5  # correct answers an arm needs before its cost per correct answer is told
NOT_WORSE = 0.10  # the cost branch of the crossover lets mms be this much less accurate, when the interval's low end is above it
HEAVY_S = 300.0  # an mms exec that may need more GPU seconds than this runs alone on the GPU
TIMEOUT_S = codex_run.TIMEOUT_S  # an exec's time (Claude's is the same)
ASK_SHARE = 0.75  # the share of an exec's time its ask can have: the agent's own turns take the rest, so an ask estimated past it cannot finish
JOB_WAIT_S = 60.0  # in all, an exec that ended waits this long for its ask jobs (and the daemon's) to end
OTHERS_WAIT_S = 10.0  # ... but only this long when other execs are running, as it cannot tell their jobs from its orphans


@dataclass(frozen=True)
class Run:
    folder: Path
    seed: int
    scale: str
    days: tuple[date, ...]
    data: Path
    temp: Path
    tag: str = ""  # an opaque name for the run's folders in the temp folder (the run's name says its seed and scale)
    timeout_s: float = TIMEOUT_S  # the seconds an exec may take: an ask estimated past ASK_SHARE of it is the model's speed, not a miss

    @property
    def ask_limit_s(self) -> float:
        """The GPU seconds an ask may need and still finish inside the exec: ASK_SHARE of its time."""
        return ASK_SHARE * self.timeout_s

    @classmethod
    def open(cls, folder: Path) -> "Run":
        """The run in `folder`, from its run.json; refused when it is another dataset's (a run from before datasets is a
        haystack's), or when the haystack's generator has changed since."""
        folder = Path(folder)
        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if (dataset := meta.get("dataset", "haystack")) != "haystack":
            raise ValueError(f"{folder} is an {dataset} run: use benchmarks/{dataset}.py")
        if meta["generator"] != haystack.VERSION:
            raise ValueError(f"{folder} was made by haystack generator {meta['generator']}, this is {haystack.VERSION}: "
                             "its questions are not the plan's now; build a new run")
        run = cls(folder, meta["seed"], meta["scale"], tuple(date.fromisoformat(d) for d in meta["days"]),
                  Path(meta["data"]), Path(meta["temp"]), meta.get("tag", ""))
        plan = run.plan()
        if meta.get("plan") != plan_hash(plan):
            raise ValueError(f"{folder}'s questions are not those its run.json recorded: the haystack changed without a "
                             "VERSION bump, so its results would be graded against other questions; build a new run")
        if meta.get("files") not in (None, files_hash(plan)):
            raise ValueError(f"{folder}'s files are not those its run.json recorded: the haystack changed without a VERSION "
                             "bump, so the week on disk is not the plan's; build a new run")
        return run

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def key(self) -> str:
        """What the run's folders in the temp folder are named by: its tag, or for a run from before tags, its name."""
        return self.tag or self.name

    @property
    def bin(self) -> Path:
        return self.temp / f"mmsb-{self.key}-bin"

    @property
    def manifest(self) -> Path:
        return self.folder / "data-manifest.json"

    def slot(self, k: int) -> Path:
        return self.temp / f"mmsb-{self.key}-w{k}"

    def plan(self) -> haystack.Plan:
        return haystack.plan(self.seed, self.scale, self.days)

    # The dataset's own parts, which `run_arm` and the commands call: another dataset's run overrides them.

    def prompt(self, q: dict, arm: str, data: Path | None, now: float | None = None) -> str:
        """What an agent is asked: the time, how the sources are reached (the only difference between the arms; `data` is the
        folder `view` gave), the question and how to answer."""
        t = datetime.fromtimestamp(time.time() if now is None else now)
        stamp = f"{haystack.WEEKDAYS[t.weekday()]} {t.day:02d} {haystack.MONTHS[t.month - 1]} {t.year}, {t:%H:%M}"
        week = f"from {haystack.date_text(self.days[0])} to {haystack.date_text(self.days[-1])}"
        sources = ", ".join(f"{name} ({what})" for name, what in haystack.SOURCES.items())
        if arm == "mms":
            how = f"mms has recorded these sources of mine {week}."
            end = "Answer from what mms holds. End with one line ANSWER: <answer>, or ANSWER: unknown when mms does not hold it."
        elif arm == "files":
            how = (f"The files of these sources, {week}, are in {data}, one folder per source; each file is "
                   "named by when it was recorded, and the logs are JSON lines with a ts field. Your working folder is yours "
                   "to write in. ffmpeg is on PATH (there is no ffprobe).")
            end = "Answer from these files. End with one line ANSWER: <answer>, or ANSWER: unknown when the files do not hold it."
        else:
            raise ValueError(f"arm is mms or files, not {arm!r}")
        return f"It is now {stamp}. {how} The sources are: {sources}. {q['question']} {end}"

    def view(self, q: dict, arm: str, k: int, home: Path | None, build: bool = True) -> Path | None:
        """The folder the agent reads the recordings in, made ready for question `q` in slot `k` (after the exec's home was
        scrubbed, before the agent starts): the frozen data for the files arm, None for the others. With `build` False it only
        names the folder, makes nothing and touches no home (a dry run)."""
        return self.data if arm == "files" else None

    def estimate(self, plan: haystack.Plan, model: "Model", q: dict) -> float:
        """The GPU seconds an mms exec may need for one ask over the question's whole range."""
        return gpu_estimate(plan, model, q)

    def needle(self, q: dict) -> str:
        """What two questions share when they ask about one needle: they never run at once in an arm."""
        return base_template(q)

    def extra(self, e: "Exec", out: dict) -> dict:
        """Fields for the exec's result record, from the exec and what the agent returned."""
        return {}

    # The commands (`run_agents`, `run_decider`, `show_agents`, `show_decider`, `dry_run`) call these too, so another dataset's
    # run overrides them and keeps the loops.

    arms: ClassVar[tuple[str, ...]] = ("files",)  # the arms with no model, in the order an agent runs them; the mms arm follows

    def check_paths(self) -> None:
        """Raises when a folder of the files arm holds a string it refuses to reach."""
        check_paths(self)

    def ensure_data(self) -> None:
        """Makes sure what the arms read is there (the haystack writes it again when the temp folder lost it)."""
        ensure_data(self)

    def questions(self, plan: haystack.Plan, subset: str, models: Sequence["Model"], only: Sequence[str] = ()) -> list[dict]:
        """The questions `agents` runs, of the `subset` and `only` those qids."""
        return selected(plan.questions, subset, models, only)

    def warm_params(self, plan: haystack.Plan) -> dict:
        """The ask that loads a model before its execs: one window of the first photo (the first, so no earlier
        one is read)."""
        t = first_photo(plan)
        return {"question": "Is anything in view?", "sources": ["phone"], "since": t, "until": t + 1, "limit": 1, "candidates": 0}

    def mms_jobs(self, model: "Model", jobs: int) -> int:
        """The execs at once in a model's mms arm."""
        return arm_jobs(model, jobs)

    def oracle(self, api, q: dict, model: "Model", ask_with: str = "oracle") -> dict:
        """The decider tier's record for one question."""
        return oracle(api, q, model, ask_with)

    def release(self) -> None:
        """Called when `run_agents` ends, however it ends: puts away what the run made in the temp folder."""

    def snapshot(self) -> Path:
        return self.folder / "homes" / "snapshot"

    def deny(self, own: Path | None = None) -> tuple[str, ...]:
        """What no tool input of the files arm may hold, lower case with forward slashes: this run's folder, the results
        folder, the repo, the words of the owner's agent setup, the code of the generator (`SOURCE_WORDS`) and the working
        slots other than `own`, by absolute path, by name (a relative path) and by glob (an `in:` entry: a listing of the temp
        folder may name them, a command may not open them)."""
        others = [k for k in range(JOBS_MAX) if own is None or self.slot(k) != own]
        base = f"mmsb-{self.key}-w"
        slots = [codex_run.IN_ONLY + word for k in others for word in (str(self.slot(k)), f"{base}{k}", f"*-w{k}", f"?-w{k}")]
        slots += [codex_run.IN_ONLY + word for word in (f"{base}*", f"{base}?", f"{base}[", "mmsb-*", "mmsb-?", "mmsb-[")]
        words = (str(self.folder), str(OUT), str(codex_run.ROOT), *DENY_WORDS, *SOURCE_WORDS, *slots)
        return tuple(dict.fromkeys(re.sub(r"[\\/]+", "/", w.lower()) for w in words))


@dataclass(frozen=True)
class Model:
    name: str
    spec: str  # the `[decider] model`: a model folder, or python:<file>
    about: About
    url: str | None = None  # where a served model (H2O's) is served

    @property
    def slug(self) -> str:
        return slug(self.name)


@dataclass(frozen=True)
class Exec:
    """One exec as an agent gets it: its prompt, its own folders and what it may not reach."""

    agent: str
    arm: str  # "blind" | "files" | "mms"
    model: Model | None  # None in the blind and files arms
    question: dict
    prompt: str
    run_dir: Path
    work: Path
    home: Path | None  # the mms arm's home
    data: Path | None  # the folder the agent reads the recordings in (`Run.view`): the files arm's
    bin_dir: Path
    deny: tuple[str, ...]
    seed: int


Agent = Callable[[Exec], dict]  # an agent's result: the common exec result of claude_run and codex_run


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


def plan_hash(plan: haystack.Plan) -> str:
    """The sha256 of the plan's questions (JSON, sorted keys): a run whose plan is not the one it was built with is refused."""
    return hashlib.sha256(json.dumps(list(plan.questions), sort_keys=True).encode("utf-8")).hexdigest()


def files_hash(plan: haystack.Plan) -> str:
    """The sha256 of the plan's files (path, source, time, name, kind and spec) and log lines: a distractor that changed with no
    change to a question, and no VERSION bump, would otherwise go unseen. A run's `run.json` holds it as `files`."""
    body = {"files": [[f.path, f.source, f.t, f.name, f.kind, f.spec] for f in plan.files], "logs": plan.logs}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def kind_of(problem: str | None) -> str | None:
    """`timeout` for `timeout: no answer in 900 s`; None for no problem."""
    return problem.split(":", 1)[0] if problem else None


def arm_dir(arm: str, model: Model | None) -> str:
    """The arm's folder in results and execs: `blind`, `files` or `mms-<model>` (only the mms arm has a model)."""
    if arm not in ARMS or (arm == "mms") != (model is not None):
        raise ValueError(f"arm is one of {', '.join(ARMS)}, and only mms has a model: not {arm!r} with {model and model.name!r}")
    return f"mms-{model.slug}" if model else arm


def level_of(q: dict) -> str:
    """How vague the question's time hint is: `part`, `day`, `week` or `time`. A question with no `level` (a plan from before
    the levels) is `part`."""
    return q.get("level") or "part"


def family(name: str) -> str:
    """A model's family, from its name: `d1` for d1-3B, d1-2B and d1-Omni-2B, then d3, pplx and h2o; a test decider is its own.
    (The backends' own `family` is the name of their env, and d1-3B has one of its own.)"""
    return re.split(r"[-_.]", name, maxsplit=1)[0].lower()


def window_s(about: About, kind: str = "video") -> float:
    """What mms estimates one window of `kind` costs this model at the reader's default profile (`ask.window_seconds`)."""
    return ask_script.window_seconds(kind, reader.profile_of(None, about), about)


def arm_jobs(model: Model | None, jobs: int) -> int:
    """The execs at once in an arm: one for a model that takes over SLOW_S a window (parallel execs share one GPU queue, so
    its timeouts would be the queue's fault), else `jobs`."""
    return 1 if model is not None and window_s(model.about) > SLOW_S else jobs


def asked_types(model: Model, controls: str | None = None) -> list[str]:
    """The kinds of data the mms arm of `model` is asked about: what it reads, and of those the text and records only when
    `controls` is None (every model) or names it."""
    kinds = haystack.readable(model.about)
    return sorted(kinds if controls is None or model.name == controls else kinds & set(MEDIA))


def arm_questions(questions: Sequence[dict], model: Model, controls: str | None = None) -> list[dict]:
    """The questions of `questions` the mms arm of `model` is asked (see `asked_types`)."""
    kinds = asked_types(model, controls)
    return [q for q in questions if q["type"] in kinds]


def pick_controls(questions: Sequence[dict]) -> list[dict]:
    """Three controls from the text and records questions, by their fields: a needle of each, and an absent one. Of each the
    one with an oracle, then the least vague hint, then the qid; so any scale has them."""
    pool = [q for q in questions if q["type"] in CONTROL_TYPES]

    def first(of: list[dict]) -> list[dict]:
        return [min(of, key=lambda q: (q["oracle"] is None, LEVELS.index(level_of(q)), q["qid"]))] if of else []

    return [*first([q for q in pool if q["type"] == "text" and not q["absent"]]),
            *first([q for q in pool if q["type"] == "records" and not q["absent"]]), *first([q for q in pool if q["absent"]])]


def pilot_questions(questions: Sequence[dict], models: Sequence[Model]) -> list[dict]:
    """The pilot: the media questions of the kinds some model of the run reads (pictures and video when there is none), and
    3 controls, in the plan's order."""
    heard = {k for m in models for k in m.about.reads} & set(MEDIA) or {"image", "video"}
    keep = {q["qid"] for q in questions if q["type"] in heard} | {q["qid"] for q in pick_controls(questions)}
    return [q for q in questions if q["qid"] in keep]


def selected(questions: Sequence[dict], subset: str, models: Sequence[Model], only: Sequence[str] = ()) -> list[dict]:
    """The questions `agents` runs: all of them, or the pilot's; with `only`, just those qids of them (a smoke)."""
    if subset not in SUBSETS:
        raise ValueError(f"a subset is one of {', '.join(SUBSETS)}, not {subset!r}")
    picked = list(questions) if subset == "all" else pilot_questions(questions, models)
    if only:
        if unknown := sorted(set(only) - {q["qid"] for q in picked}):
            raise ValueError(f"--only names questions that are not in the {subset} questions: {', '.join(unknown)}")
        picked = [q for q in picked if q["qid"] in set(only)]
    return picked


def write_json(path: Path, data) -> None:
    """`data` written to `path` whole or not at all (a `.part` file, then a rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(part, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@functools.cache
def kernel():
    """kernel32 with the signatures `unlink` uses (Windows)."""
    from ctypes import wintypes

    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.CreateFileW.restype = wintypes.HANDLE
    lib.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    lib.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    lib.CloseHandle.argtypes = [wintypes.HANDLE]
    return lib


def unlink(path: Path) -> None:
    """Delete a file, or a hardlink to one, without touching its attributes. On Windows a link to a read-only file cannot be
    deleted until the attribute is cleared, and clearing it clears it on every link (the snapshot's media, the run's data, the
    download): this deletes the name alone."""
    if os.name != "nt":
        os.unlink(path)
        return
    lib = kernel()
    handle = lib.CreateFileW(str(path), 0x00010000, 7, None, 3, 0x00200000, None)  # DELETE; share all; open the link itself
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        flags = ctypes.c_uint32(0x1 | 0x2 | 0x10)  # FileDispositionInfoEx: delete, POSIX semantics, ignore the read-only attribute
        if not lib.SetFileInformationByHandle(handle, 21, ctypes.byref(flags), 4):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        lib.CloseHandle(handle)


def remove_tree(path: Path) -> None:
    """`path` and all below it removed, a read-only file (the data, a file an exec copied) made writable first, except a file
    with other links (a hardlink an agent made, a home's media): that is deleted as a name (`unlink`), so the other links keep
    their attributes. A directory junction (an agent can make one with `mklink /J`) is removed as the link it is: `rmtree` refuses
    one it is given, and a junction met inside a tree is removed by it without going through."""
    if path.is_junction():
        os.rmdir(path)
        return

    def writable(function, target, error) -> None:
        if function is os.unlink and os.lstat(target).st_nlink > 1:
            unlink(Path(target))
            return
        os.chmod(target, stat.S_IWRITE)
        function(target)

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, onexc=writable)
    elif path.lstat().st_nlink > 1:
        unlink(path)
    else:
        os.chmod(path, stat.S_IWRITE)
        path.unlink()


def empty(folder: Path) -> None:
    """`folder` made, with nothing in it."""
    folder.mkdir(parents=True, exist_ok=True)
    for child in folder.iterdir():
        remove_tree(child)


# -- the data and the homes ---------------------------------------------------------------------------------------------

def commit_of() -> str:
    """The repo's short commit, `+dirty` when tracked files differ from it."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(codex_run.ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def build(out: Path, scale: str, seed: int, *, temp: Path | None = None, days: Sequence[date] | None = None,
          probe: bool = True) -> Run:
    """Write the week, freeze it, copy ffmpeg for the files arm, load a snapshot home, and run the shortcut probe over the
    data (`probe=False` skips it). `run.json` comes last and holds the probe's outcome (`probe_of`), so a folder without it is a
    build that failed; a probe that fails leaves a complete run, flagged, for the caller to refuse."""
    if scale not in haystack.SCALES:
        raise ValueError(f"a scale is one of {', '.join(haystack.SCALES)}, not {scale!r}")
    temp = Path(temp or tempfile.gettempdir())
    days = tuple(days) if days is not None else tuple(haystack.days_before(date.today(), haystack.SCALES[scale].days))
    plan = haystack.plan(seed, scale, days)  # a wrong number of days fails here, before anything is written
    name = f"{datetime.now():%Y%m%d-%H%M}-{scale}-s{seed}"
    folder = Path(out) / name
    if folder.exists():
        raise ValueError(f"{folder} exists: a run is one a minute for a scale and seed; wait, or remove it")
    folder.mkdir(parents=True)
    tag = secrets.token_hex(5)  # the temp folders are named by this, not by the run: the name says the seed and the scale
    run = Run(folder, seed, scale, days, temp / f"mmsb-{tag}-data", temp, tag)
    if run.data.exists():
        remove_tree(run.data)
    haystack.write(plan, run.data)
    freeze(run.data, run.manifest)
    ensure_bin(run)
    home = run.snapshot()
    home.mkdir(parents=True)
    write_config(home, SNAPSHOT_CONFIG)
    daemon = Daemon(home).start()
    try:
        haystack.load(daemon.api, run.data, plan)
    finally:
        daemon.stop()
    wait_for(lambda: not (home / "daemon.json").exists(), what="the snapshot's daemon to stop")
    outcome = run_probe(run, plan, probe)
    meta = {"seed": seed, "scale": scale, "days": [d.isoformat() for d in days], "data": str(run.data), "temp": str(temp),
            "generator": haystack.VERSION, "plan": plan_hash(plan), "files": files_hash(plan), "created": datetime.now().isoformat(timespec="seconds"),
            "mms_commit": commit_of(), "probe": outcome, "tag": tag}
    (folder / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return run


def bin_exe(run: Run) -> Path:
    """The ffmpeg the files arm has first on its PATH (`ensure_bin` copies it)."""
    return run.bin / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def run_probe(run: Run, plan: haystack.Plan, probe: bool = True) -> dict:
    """The shortcut probe over the run's data, which `shortcuts.run` reads and reports on in the run folder (`shortcuts.json`,
    `shortcuts.md`), with the files arm's ffmpeg: `{"status": "passed" | "failed" | "skipped", "failures": [...]}`."""
    if not probe:
        return {"status": "skipped", "failures": []}
    failures = list(shortcuts.run(plan, run.data, run.folder, ffmpeg=bin_exe(run), seed=run.seed))
    return {"status": "failed" if failures else "passed", "failures": failures}


def probe_of(folder: Path) -> dict | None:
    """The probe's outcome that a run's run.json records, None for a run built before the probe."""
    return read_json(Path(folder) / "run.json").get("probe")


def probe_line(name: str, outcome: dict | None) -> str | None:
    """What to say of a run whose probe did not pass, else None."""
    if outcome is None:
        return f"{name} has no shortcut probe record (it was built before the probe), so nothing says a shortcut cannot find its needles"
    if outcome["status"] == "skipped":
        return f"{name} was built with --no-probe: nothing says a colour filter, frame difference, band filter or file size cannot find its needles"
    if outcome["status"] == "failed":
        return (f"{name} failed the shortcut probe ({len(outcome['failures'])} failures, in its shortcuts.md): a shortcut may find "
                "its needles, so its files arm may be doing less work than an ask")
    return None


def ensure_bin(run: Run) -> None:
    """imageio-ffmpeg's binary copied to the folder that goes first on the files arm's PATH, named `ffmpeg` (no ffprobe)."""
    import imageio_ffmpeg

    target = bin_exe(run)
    if not target.exists():
        run.bin.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), target)


def ensure_data(run: Run) -> None:
    """The data written again from the plan when the temp folder lost it (Windows cleans it), with a fresh manifest."""
    saved = read_json(run.manifest)
    if not run.data.is_dir() or any(not (run.data / rel).is_file() for rel in saved):
        print(f"the data of {run.name} is gone from {run.temp}: writing it again from the plan", flush=True)
        if run.data.exists():
            remove_tree(run.data)
        haystack.write(run.plan(), run.data)
        freeze(run.data, run.manifest)
    ensure_bin(run)


def stats_of(data: Path) -> dict[str, list[int]]:
    found = {}
    for path in sorted(data.rglob("*")):
        if path.is_file():
            info = path.stat()
            found[path.relative_to(data).as_posix()] = [info.st_size, info.st_mtime_ns]
    return found


def inherit_permissions(folder: Path) -> None:
    """On Windows, every file and folder under `folder` given back the permissions it inherits, and nothing of its own: a file
    with owner-only permissions (one moved out of a Python temporary folder) cannot be opened by Codex's sandbox, which reads
    as the CodexSandboxUsers group. A no-op elsewhere."""
    if os.name != "nt":
        return
    done = subprocess.run(["icacls", str(folder), "/reset", "/T", "/Q"], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"icacls /reset failed on {folder}: {(done.stderr or done.stdout).strip()}")


def own_permissions(folder: Path) -> list[str]:
    """On Windows, the files under `folder` with a permission of their own (not inherited), as paths relative to it, in one
    PowerShell `Get-Acl` pass; [] elsewhere."""
    if os.name != "nt":
        return []
    quoted = str(folder).replace("'", "''")
    # Stop, because a Get-Acl that fails would otherwise read as a file with nothing of its own. No PSModulePath, because a child of
    # PowerShell 7 (GitHub's runners) inherits one whose Get-Acl Windows PowerShell cannot load. Each path is cut against the root
    # PowerShell reports, because it expands a short (8.3) name in the folder it is given.
    script = ("$ErrorActionPreference = 'Stop'; "
              f"$root = (Get-Item -LiteralPath '{quoted}').FullName; "
              "Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object { @((Get-Acl -LiteralPath $_.FullName).Access"
              " | Where-Object { -not $_.IsInherited }).Count -gt 0 } | ForEach-Object { $_.FullName.Substring($root.TrimEnd('\\').Length + 1) }")
    env = {name: value for name, value in os.environ.items() if name.upper() != "PSMODULEPATH"}
    done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, env=env)
    if done.returncode:
        raise RuntimeError(f"Get-Acl failed on {folder}: {done.stderr.strip()}")
    return [Path(line.strip()).as_posix() for line in done.stdout.splitlines() if line.strip()]


def freeze(data: Path, manifest: Path) -> None:
    """Every file of `data` with only the permissions it inherits (so Codex's sandbox can read it) and read-only, and `manifest`
    (size and mtime of each) written. Raises when a file still has permissions of its own."""
    inherit_permissions(data)
    if mine := own_permissions(data):
        raise RuntimeError(f"{len(mine)} data files keep permissions of their own, which Codex's sandbox may not read: "
                           f"{', '.join(mine[:5])}")
    for path in data.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)
    manifest.write_text(json.dumps(stats_of(data)), encoding="utf-8")


def changed(data: Path, manifest: Path) -> str | None:
    """The first file of `data` that is gone, changed or added since `freeze` (`kind: path`), or None."""
    saved, now = read_json(manifest), stats_of(data)
    for rel, was in saved.items():
        if rel not in now:
            return f"gone: {rel}"
        if now[rel] != was:
            return f"changed: {rel}"
    return next((f"added: {rel}" for rel in now if rel not in saved), None)


def copy_home(snapshot: Path, target: Path, model: Model) -> Path:
    """A home for one (agent, model): the snapshot's files, with `model` as its decider. A `target` that exists (a run that
    is continued) is used as it is. The snapshot must have no daemon, so its index is whole. The media (MEDIA_FOLDERS) are
    hardlinked, not copied, so a home costs its index alone: the snapshot's are made read-only first, and a link shares that,
    so no home can write one. The daemon never rewrites a media file, and the janitor deletes nothing here (the snapshot's
    config sets no cap or age). The snapshot and the target must be on one volume."""
    if (snapshot / "daemon.json").exists():
        raise RuntimeError(f"the snapshot {snapshot} has a daemon running: stop it (mms stop) before copying")
    if target.exists():
        return target
    part = target.with_name(target.name + ".part")
    if part.exists():
        remove_tree(part)
    for folder in MEDIA_FOLDERS:
        for path in (snapshot / folder).rglob("*"):
            if path.is_file():
                os.chmod(path, stat.S_IREAD)

    def link_media(src: str, dst: str) -> None:
        (os.link if Path(src).relative_to(snapshot).parts[0] in MEDIA_FOLDERS else shutil.copy2)(src, dst)

    shutil.copytree(snapshot, part, ignore=shutil.ignore_patterns(*COPY_SKIPS), copy_function=link_media)
    config.set_decider(part, model=model.spec, url=model.url)
    os.replace(part, target)
    return target


def serving(url: str) -> bool:
    """Whether a model server answers 200 at /health."""
    try:
        connection = http.client.HTTPConnection(*config.loopback(url), timeout=5)
        try:
            connection.request("GET", "/health")
            return connection.getresponse().status == 200
        finally:
            connection.close()
    except (OSError, http.client.HTTPException):
        return False


def resolve_models(names: list[str], models_dir: Path) -> tuple[list[Model], list[str]]:
    """The models of `names` that are ready to run, and a line for each that is not, saying why."""
    ready: list[Model] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for name in names:
        found = decider.known(name)
        if found is None:
            skipped.append(f"{name}: not a model mms knows ({', '.join(a.name for a in decider.known_abouts())})")
            continue
        backend, about = found
        if about.name in seen:
            continue
        seen.add(about.name)
        folder = Path(models_dir).expanduser().resolve() / about.name
        claimed = decider.claim(folder)
        if claimed is None or claimed[1] != about:
            skipped.append(f"{about.name}: not downloaded: mms model use {about.name}")
        elif problem := modelenv.problem(backend, folder, None):
            skipped.append(f"{about.name}: its env is not ready: {problem}")
        elif backend.served and not serving(backend.default_url):
            skipped.append(f"{about.name}: not serving at {backend.default_url}: start serve.sh in WSL (DESIGN.md, 13.3)")
        elif not backend.served and (why := fetch.unfinished(folder)) is not None:
            skipped.append(f"{about.name}: its weights are not all there: {why}")
        else:
            ready.append(Model(about.name, folder.as_posix(), about, backend.default_url if backend.served else None))
    return ready, skipped


def needs_gpu(model: Model) -> bool:
    """Whether running `model` loads weights on this machine's GPU: not a test decider, not a server the owner runs."""
    return not model.spec.startswith("python:") and model.url is None


def gpu_busy() -> str | None:
    """What nvidia-smi lists as running on the GPU, or None when nothing does. The GPU runs alone (on Windows with a display
    the list can be empty while a window uses it: look at the machine too)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi is not on PATH, so whether the GPU is free for a real model cannot be told")
    done = subprocess.run([exe, "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True,
                          text=True, timeout=30)
    if done.returncode:
        raise RuntimeError(f"nvidia-smi exited {done.returncode}: {done.stderr.strip()[-300:]}")
    return "; ".join(line.strip() for line in done.stdout.splitlines() if line.strip()) or None


def check_gpu(model: Model) -> None:
    if needs_gpu(model) and (busy := gpu_busy()):
        raise RuntimeError(f"the GPU is busy, and it runs alone: {busy}")


def gpu_lost() -> str | None:
    """Why the GPU cannot be read any more, else None: after a driver crash nvidia-smi says "GPU is lost", mms's model worker
    cannot load, and an agent answers from `find` and `look` alone (2026-11-04: 18 mms results so before anything noticed)."""
    try:
        gpu_busy()
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        return str(e)
    return None


def note_model(run: Run, model: Model, asked: list[str] | None = None) -> None:
    """models.json: the kinds of data each model of this run reads (`readable`, what the decider tier asks) and the ones its
    mms arm is asked about (`asked`, from `asked_types`: set by `agents`; missing, the report takes `readable`), so the report
    knows a model that mms does not list (a test decider) and its questions."""
    file = run.folder / "models.json"
    known = read_json(file) if file.exists() else {}
    known.setdefault(model.name, {}).update(slug=model.slug, readable=sorted(haystack.readable(model.about)))
    if asked is not None:
        known[model.name]["asked"] = asked
    write_json(file, known)


# -- asking -------------------------------------------------------------------------------------------------------------

def ask(api, params: dict, project: str) -> tuple[dict, float]:
    """One `ask` as a job, collected until it ends: its result and the wall seconds."""
    began = time.monotonic()
    job = api.call("jobs/start", script="ask", params=params, wait_s=50, caller={"project": project})
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    if job["state"] != "done":
        raise RuntimeError(f"the ask {params['question']!r} {job['state']}: {job['error']}")
    return job["result"], round(time.monotonic() - began, 2)


def first_photo(plan: haystack.Plan) -> float:
    return min(f.t for f in plan.files if f.kind == "photo")


def warm(api, params: dict) -> float:
    """Seconds of the one-window ask `Run.warm_params` gives, so the model is loaded before the execs; the load is in the number."""
    _, seconds = ask(api, params, WARM_PROJECT)
    return seconds


def span_of(found: dict) -> tuple[float, float]:
    """The start and end of a sighting or candidate of an ask result: `t` when it is one instant, else `t0` and `t1`."""
    return config.parse_when(found.get("t") or found["t0"]), config.parse_when(found.get("t") or found["t1"])


def meets(a0: float, a1: float, moments: list, tol: float = haystack.TOL_S) -> bool:
    """Whether the span a0..a1 comes within `tol` seconds of a moment."""
    return any(a0 <= m1 + tol and a1 >= m0 - tol for m0, m1 in moments)


def touched(spans: list[tuple[float, float]], moments: list, tol: float = haystack.TOL_S) -> list[int]:
    """The indexes of the moments some span comes within `tol` seconds of."""
    return [i for i, (m0, m1) in enumerate(moments) if any(a <= m1 + tol and b >= m0 - tol for a, b in spans)]


def needle_p(moments: list, sightings: list[dict], candidates: list[dict], tol: float = haystack.TOL_S) -> float | None:
    """The needle's p: the best p of a sighting or a candidate within `tol` seconds of a moment of the truth, or None when
    the ask named none (the needle was neither a hit nor among its best windows)."""
    ps = [s["p"] for s in sightings if meets(*span_of(s), moments, tol)]
    ps += [c["p"] for c in candidates if meets(t := span_of(c)[0], t, moments, tol)]
    return max(ps) if ps else None


def oracle_params(q: dict, text: str) -> dict:
    """What an oracle ask passes: a `last` question asks for the newest LAST_LIMIT sightings (the agent checks each one, newest
    first, so a false sighting before the needle costs a look, not the answer), a count for up to COUNT_LIMIT, the rest for up
    to ORACLE_LIMIT, each with ORACLE_CANDIDATES candidates, and every one may run to ORACLE_MAX_S, so a slow model is not cut
    at 600."""
    last = bool(q.get("last"))
    counting = q["template"] == "photo-count" or q["fake"]["check"] == "count"
    limit = LAST_LIMIT if last else COUNT_LIMIT if counting else ORACLE_LIMIT
    return {"question": text, "sources": [q["source"]], "since": q["since"], "until": q["until"], "limit": limit,
            "candidates": ORACLE_CANDIDATES, "max_s": ORACLE_MAX_S,
            **({"oldest_first": True} if q["fake"]["oldest_first"] and not last else {})}


def oracle(api, question: dict, model: Model, ask_with: str = "oracle") -> dict:
    """One oracle ask: the needle's yes/no (the fake's own words with `ask_with="fake"`, for the offline test) over its source
    and range, scored against the moments of the plan. Only a sighting that overlaps the range counts, since an earlier note
    or version is read too. A sighting of another needle of the template (`truth.others`) is no false sighting. The record holds
    the needle's p (`needle_p`), which of the truth's moments the sightings found, and its `rank`: the place of the first
    in-range sighting that touches the truth, newest first (by end), or None when none does."""
    if ask_with not in ("oracle", "fake"):
        raise ValueError(f"ask_with is oracle or fake, not {ask_with!r}")
    q = question
    text = q["oracle"] if ask_with == "oracle" else q["fake"]["question"]
    params = oracle_params(q, text)
    result, wall = ask(api, params, ORACLE_PROJECT)
    moments, others = q["truth"]["moments"], q["truth"].get("others") or []
    spans = [span_of(s) for s in result["sightings"]]
    inside = [(a, b) for a, b in spans if a < q["until"] and b >= q["since"]]
    false = sum(1 for a, b in inside if not meets(a, b, moments) and not meets(a, b, others))
    found = top5 = p = rank = None
    got: list[int] = []
    if not q["absent"]:
        got = touched(inside, moments)
        found = bool(got)
        newest_first = sorted(inside, key=lambda s: s[1], reverse=True)
        rank = next((i for i, (a, b) in enumerate(newest_first, 1) if meets(a, b, moments)), None)
        top5 = found or any(meets(t, t, moments) for t in (span_of(c)[0] for c in result["candidates"]))
        p = needle_p(moments, result["sightings"], result["candidates"])
    progress = result["progress"]
    return {"qid": q["qid"], "type": q["type"], "template": q["template"], "level": level_of(q), "last": bool(q.get("last")),
            "absent": q["absent"], "model": model.name, "question": text, "source": q["source"], "since": q["since"],
            "until": q["until"], "limit": params["limit"], "found": found, "top5": top5, "p": p, "rank": rank,
            "moments": len(moments), "moments_found": got, "others": len(others), "others_found": len(touched(inside, others)),
            "in_range": len(inside), "false": false, "windows": progress["windows"], "seconds": progress["seconds"],
            "wall_s": wall, "stopped": result.get("stopped"), "warnings": result.get("warnings", []),
            "sightings": result["sightings"], "candidates": result["candidates"]}


def run_decider(run: Run, models: list[Model], ask_with: str = "oracle", qids: Sequence[str] | None = None) -> None:
    """The decider tier: each model, in a copy of the snapshot of its own, asked each oracle question of the types it reads (only
    those of `qids`, when given: a test's). A question with a record is not asked again."""
    plan = run.plan()
    for model in models:
        note_model(run, model)
        mine = [q for q in haystack.questions_for(plan.questions, model.about)
                if q["oracle"] is not None and (qids is None or q["qid"] in qids)]
        folder = run.folder / "results" / "decider" / model.slug
        todo = [q for q in mine if not (folder / f"{q['qid']}.json").exists()]
        if not todo:
            continue
        check_gpu(model)
        home = copy_home(run.snapshot(), run.folder / "homes" / f"decider-{model.slug}", model)
        daemon = Daemon(home).start()
        try:
            first = warm(daemon.api, run.warm_params(plan))
            if not (folder / "warm.json").exists():
                write_json(folder / "warm.json", {"model": model.name, "seconds": first, "at": now_text()})
            for q in todo:
                record = run.oracle(daemon.api, q, model, ask_with)
                write_json(folder / f"{q['qid']}.json", record)
                print(f"{model.name} {q['qid']}: found={record['found']}" + (f" false={record['false']}" if "false" in record else "")
                      + f" {record['windows']} windows in {record['seconds']:g} s", flush=True)
        finally:
            daemon.stop()


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def result_path(run: Run, agent: str, arm: str, model: Model | None, qid: str) -> Path:
    return run.folder / "results" / agent / arm_dir(arm, model) / f"{qid}.json"


def load_result(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def is_done(result: dict | None) -> bool:
    """A result that is not asked again: it finished, or it escaped, timed out or was capped (a wrong answer). A `setup`,
    `limit`, `auth` or `error` problem did not test the question: the next command asks it again."""
    return result is not None and (result["problem"] is None or kind_of(result["problem"]) in DONE_KINDS)


def result_jsons(result) -> list[dict]:
    """Every JSON object a tool's result holds: mms's tools answer with one in a text block, and a `look` of several moments with
    one for each (Codex's `result` is `{"content": [...]}`, Claude's the list of blocks or the text). [] when there is none."""
    if isinstance(result, dict):
        if isinstance(result.get("structured_content"), dict):
            return [result["structured_content"]]
        result = result.get("content")
    if isinstance(result, str):
        texts = [result]
    elif isinstance(result, (list, tuple)):
        texts = [b.get("text") if isinstance(b, dict) else b for b in result]
    else:
        return []
    out = []
    for text in texts:
        if isinstance(text, str):
            try:
                got = json.loads(text)
            except ValueError:
                continue
            if isinstance(got, dict):
                out.append(got)
    return out


def result_json(result) -> dict | None:
    """The first JSON object a tool's result holds (`result_jsons`), else None."""
    found = result_jsons(result)
    return found[0] if found else None


def is_view(tool: dict) -> bool:
    """Whether a call shows the agent a picture: Codex's `view_image`, or Claude's `Read` of a picture file."""
    args = tool.get("arguments")
    if tool.get("tool") == "view_image":
        return True
    path = (args.get("file_path") or args.get("path")) if isinstance(args, dict) else None
    return tool.get("tool") == "Read" and isinstance(path, str) and path.lower().endswith(IMAGE_SUFFIXES)


def work_of(tools: Sequence[dict]) -> dict:
    """How an exec worked, from its calls (tools.json): the ask jobs it ran, counted by the last result seen of each (the
    `progress` it gave: `windows`, GPU `seconds`; how it `stopped`, `running` if it had not), the pictures it viewed and the
    shell (ffmpeg) commands it ran. A tool that gives no result, or one that is not JSON, counts as none."""
    jobs: dict[str, dict] = {}
    views = shells = 0
    for t in tools:
        args = t.get("arguments") if isinstance(t.get("arguments"), dict) else {}
        if t.get("tool") == "ask":
            got = result_json(t.get("result"))
            if got is not None and (job := got.get("job") or args.get("job")):
                jobs[str(job)] = got
        elif is_view(t):
            views += 1
        elif t.get("tool") in ("shell", "Bash"):
            shells += 1
    progress = [(j.get("progress") if isinstance(j.get("progress"), dict) else {}) for j in jobs.values()]
    return {"jobs": len(jobs), "windows": sum(p.get("windows") or 0 for p in progress),
            "seconds": round(sum(p.get("seconds") or 0 for p in progress), 2),
            "stopped": dict(Counter(j.get("stopped") or "running" for j in jobs.values())), "views": views, "shells": shells}


def work_from_file(path: Path) -> dict | None:
    """`work_of` a tools.json that a result's exec folder holds (for a result made before it held `work`), None when it is
    gone or is not a list of calls."""
    try:
        tools = read_json(path)
    except (OSError, ValueError):
        return None
    return work_of([t for t in tools if isinstance(t, dict)]) if isinstance(tools, list) else None


def record(e: Exec, out: dict) -> dict:
    """The result record of one exec (no truth is stored: the report grades from the plan)."""
    q, problem, answer = e.question, out.get("problem"), out.get("answer")
    tools = out.get("tools") or []
    return {"qid": q["qid"], "type": q["type"], "template": q["template"], "level": level_of(q), "last": bool(q.get("last")),
            "absent": q["absent"], "seed": e.seed,
            "agent": e.agent, "arm": e.arm, "model": e.model.name if e.model else None, "answer": answer,
            "passed": problem is None and grade.grade(q["grade"], answer),
            "abstained": grade.abstained(answer, q["grade"]["ignore"]), "problem": problem, "seconds": out["seconds"],
            "turns": out.get("turns"), "denials": out.get("denials"), "calls": len(tools),
            "tools": dict(Counter(t["tool"] for t in tools)), "work": work_of(tools),
            "scope_all": sum(1 for t in tools if isinstance(t.get("arguments"), dict) and t["arguments"].get("scope") == "all"),
            "tokens": out.get("tokens"), "cost_usd": out.get("cost_usd"), "used_percent": out.get("used_percent"),
            "agent_model": out.get("model"), "agent_effort": out.get("effort"), "exec": "/".join(e.run_dir.parts[-4:]), "at": now_text()}


class Halt:
    """Why an agent (or one arm of it) stops: the first reason is kept and printed."""

    def __init__(self) -> None:
        self.reason: str | None = None
        self.errors = 0  # `error:` results in a row
        self._lock = threading.Lock()

    def set(self, reason: str) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
                print(f"stopping: {reason}", flush=True)

    def after(self, agent: str, rec: dict) -> None:
        """Count a result: the agent stops on its reason, or on ERRORS_STOP errors in a row."""
        with self._lock:
            self.errors = self.errors + 1 if kind_of(rec["problem"]) == "error" else 0
            streak = self.errors
        reason = stop_reason(agent, rec)
        if reason is None and streak >= ERRORS_STOP:
            reason = f"{agent}: {streak} errors in a row, the last {rec['problem']}"
        if reason:
            self.set(reason)


def stop_reason(agent: str, rec: dict) -> str | None:
    """Why an agent stops after an exec: its session was not the one asked for, its plan is out or its login is, or its use of
    the week is at STOP_AT (for Codex also when it cannot be read: it is read from the exec's own rollout)."""
    problem, used = rec["problem"] or "", rec["used_percent"]
    if kind_of(problem) in STOP_KINDS:
        return f"{agent}: {problem}"
    if agent == "codex" and used is None:
        return "codex: the plan's weekly use cannot be read, so the quota cannot be watched"
    if used is not None and used >= STOP_AT:
        return f"{agent}: the plan is at {used:g}% of its use; the run stops at {STOP_AT:g}%"
    return None


def gpu_estimate(plan: haystack.Plan, model: Model, q: dict) -> float:
    """The GPU seconds an mms exec may need for one ask over its question's whole range: the windows the haystack counts there
    times the model's seconds a window (as `decider_minutes` does). 0 while the haystack counts none."""
    count = getattr(haystack, "windows_in", None)
    if count is None:
        return 0.0
    return count(plan, q["source"], q["since"], q["until"]) * window_s(model.about, SOURCE_KINDS.get(q["source"], "video"))


def exec_project(q: dict) -> str:
    """The project an exec's asks and labels carry: its working folder's `.mms-project` holds `bench-<qid>`."""
    return f"id:bench-{q['qid']}"


def base_template(q: dict) -> str:
    """The template whose needle a question asks about: an absent question asks in its template's words."""
    return getattr(haystack, "BASE_OF", {}).get(q["template"], q["template"])


def kept_items(home: Path) -> set[int]:
    """The items of an mms home that are kept (a label with `keep` marks them)."""
    with closing(sqlite3.connect(home / "index.sqlite", timeout=30)) as db:
        return {row[0] for row in db.execute("SELECT id FROM items WHERE keep = 1")}


def open_jobs(tools: Sequence[dict]) -> list[str]:
    """The ask jobs that an exec's calls started and left running or queued (by the last result seen of each)."""
    last: dict[str, str] = {}
    for t in tools:
        got = result_json(t.get("result")) if t.get("tool") == "ask" else None
        args = t.get("arguments") if isinstance(t.get("arguments"), dict) else {}
        if got is not None and (job := got.get("job") or args.get("job")):
            last[str(job)] = str(got.get("state"))
    return [job for job, state in last.items() if state in ("queued", "running")]


def cancel_open(api, tools: Sequence[dict]) -> list[str]:
    """Cancel the ask jobs an exec left running, so that none goes on using the GPU or labels after the exec has ended (a cancelled
    ask still labels what it found, which `scrub_home` then takes out). A job already gone is passed over. The jobs it cancels."""
    out = []
    for job in open_jobs(tools):
        try:
            api.call("jobs/cancel", job=job)
        except Exception:  # noqa: BLE001 - the daemon says it is gone or ended: nothing to cancel
            continue
        out.append(job)
    return out


def settle(api, jobs: Sequence[str], limit_s: float | None = None) -> list[str]:
    """Wait for each of `jobs` to end (`jobs/get`), `limit_s` in all; the ids that had not ended by then. A job the daemon does
    not know (it ended long ago, or the daemon is gone) counts as ended."""
    until = time.monotonic() + (JOB_WAIT_S if limit_s is None else limit_s)
    unended = []
    for job in jobs:
        while True:
            left = until - time.monotonic()
            try:
                got = api.call("jobs/get", job=job, wait_s=max(0.0, min(left, 10.0)))
            except Exception:  # noqa: BLE001
                break
            if got.get("state") in ("done", "failed", "cancelled"):
                break
            if left <= 0:
                unended.append(job)
                break
    return unended


def wait_idle(api, allowed: int, limit_s: float) -> int | None:
    """Wait until the daemon's jobs (running and queued, from `status`) are `allowed` or fewer, `limit_s` at most; how many it
    still had over that, 0 when it came down, None when the count cannot be read. A job whose id was never logged, because its exec
    timed out inside the call that started it, is only in this count."""
    until = time.monotonic() + limit_s
    while True:
        try:
            counts = api.call("status").get("jobs") or {}
        except Exception:  # noqa: BLE001
            return None
        if counts.get("running") is None:
            return None
        over = counts["running"] + (counts.get("queued") or 0) - allowed
        if over <= 0:
            return 0
        if time.monotonic() >= until:
            return over
        time.sleep(0.25)


def quiesce(api, tools: Sequence[dict], others: int = 0, limit_s: float | None = None) -> dict:
    """What an mms exec leaves running when it ends, dealt with before its labels are scrubbed and its template released: the jobs
    its log shows open are cancelled and waited for (an ask labels what it found when it stops, a cancel included, so a label may
    land after the cancel), then the daemon's job count is waited down to `others`, one for each exec still running (a job whose id
    the log never held is only in the count; with others running that is a guess, so the wait is OTHERS_WAIT_S). Returns the
    cancelled jobs, those that had not ended (`unended`), the jobs still over the count (`orphans`, None when unreadable) and the
    seconds waited."""
    limit_s = JOB_WAIT_S if limit_s is None else limit_s
    began = time.monotonic()
    cancelled = cancel_open(api, tools)
    unended = settle(api, cancelled, limit_s)
    limit = max(0.0, (limit_s if not others else min(limit_s, OTHERS_WAIT_S)) - (time.monotonic() - began))
    orphans = wait_idle(api, others, limit)
    return {"cancelled": cancelled, "unended": unended, "orphans": orphans, "settle_s": round(time.monotonic() - began, 1)}


def scrub_home(home: Path, kept: set[int], *, projects: Sequence[str] = (), prefix: str | None = None) -> dict:
    """Take out of an mms home what the execs of `projects` (or of every project that starts with `prefix`) left in it, so that
    the next exec starts from the snapshot's state. That is every piece of state an exec can write there: the labels of its asks
    (`ask` labels each sighting with the caller's project, when the job ends) and of the `label` tool, the items that `label`
    kept (put back to `kept`, the snapshot's), and the project and session rows its calls made. What stays is no state of the
    exec's: the `looks/` pictures (named by what they show), the daemon's job records (random ids, in memory) and its log.
    Nothing else keeps an answer: ask has no cache, and no agent has `remember`. Done in the index, which the daemon reads at
    each call. It is idempotent, and run again at the start of each exec for the execs that ended, so that the label of a job
    that outlived its exec is taken out too."""
    where, arg = (("{} LIKE ?", prefix + "%") if prefix is not None else ("{} IN (SELECT value FROM json_each(?))", json.dumps(list(projects))))
    with closing(sqlite3.connect(home / "index.sqlite", timeout=30, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        labels = db.execute(f"DELETE FROM labels WHERE {where.format('project')}", (arg,)).rowcount
        db.execute(f"DELETE FROM sessions WHERE {where.format('project')}", (arg,))
        db.execute(f"DELETE FROM projects WHERE {where.format('id')}", (arg,))
        unkept = db.execute("UPDATE items SET keep = 0 WHERE keep = 1 AND id NOT IN (SELECT value FROM json_each(?))",
                            (json.dumps(sorted(kept)),)).rowcount
        db.execute("COMMIT")
    return {"labels": labels, "unkept": unkept}


def run_arm(run: Run, name: str, agent: Agent, arm: str, model: Model | None, questions: Sequence[dict], home: Path | None,
            jobs: int, halt: Halt) -> None:
    """The pending questions of one arm (`arm` is one of ARMS; `model` is the mms arm's alone) in ascending `until`, `jobs` at
    a time, each in a working slot. Each exec is given the folder `run.view` makes ready, the prompt of `run.prompt`, and its
    result holds what `run.extra` adds. In an mms home:
    - the questions of AS_OF sources run one at a time (the others stay parallel), so a later question's ask has labelled no
      earlier note or version before that one's own question ran;
    - two questions about one template's needle never run at once, and each exec's state is taken out of the home when it ends
      (`scrub_home`), so no exec can see another's labels, in `find` with scope all or in `look`;
    - an exec that may need over HEAVY_S GPU seconds (`run.estimate`) runs alone and holds the GPU: nothing else of the arm starts
      beside it (a week's ask over the screen is tens of thousands of frames, and parallel ones would pass the exec's timeout in the
      queue), and it starts when the lighter ones have finished.
    Each result records the estimate (and whether it is past the exec's timeout), the seconds the question was held back by a rule
    while first in line with a slot free (`held_s`), and whether it ran alone.
    When an mms exec ends its jobs are cancelled and waited for, and the daemon is waited down to the execs still running, before
    its labels are scrubbed and its template released (`quiesce`); one that holds the GPU starts when the daemon is idle. Two
    questions about one template's needle also never run at once in the files arm."""
    pending = sorted((q for q in questions if not is_done(load_result(result_path(run, name, arm, model, q["qid"])))),
                     key=lambda q: (q["until"], q["qid"]))
    if not pending:
        return
    arm_halt = Halt()  # the files arm alone stops when the data changed
    free = list(range(jobs))  # the working slots not in use
    running: dict = {}  # future -> (question, slot)
    plan = run.plan() if arm == "mms" else None
    estimate_s = {q["qid"]: run.estimate(plan, model, q) for q in pending} if arm == "mms" else {}
    kept = kept_items(run.snapshot()) if arm == "mms" else set()  # not the live home: an interrupted exec's keeps are not the baseline
    ended: set[str] = set()  # the projects of the execs that ended
    held: dict[str, float] = {}  # qid -> when it was first in line with a slot free and a rule kept it from starting
    lock = threading.Lock()

    def heavy(q: dict) -> bool:
        return estimate_s.get(q["qid"], 0.0) >= HEAVY_S

    def one(q: dict, k: int) -> None:
        since = held.pop(q["qid"], None)
        waited = 0.0 if since is None else time.monotonic() - since
        busy_at_start = wait_idle(Api(home), 0, JOB_WAIT_S) if arm == "mms" and heavy(q) else None  # it holds the GPU: it starts idle
        work = run.slot(k)
        empty(work)
        if arm == "mms":
            (work / MARKER).write_text(f"bench-{q['qid']}", encoding="utf-8")
        run_dir = run.folder / "execs" / name / arm_dir(arm, model) / q["qid"]
        if run_dir.exists():
            remove_tree(run_dir)
        if arm == "mms":
            with lock:
                gone = sorted(ended)
            scrub_home(home, kept, projects=gone)  # a job of an exec that ended may have labelled since
        data = run.view(q, arm, k, home)
        e = Exec(name, arm, model, q, run.prompt(q, arm, data), run_dir, work, home, data, run.bin, run.deny(work), run.seed)
        out = None
        try:
            out = agent(e)
            rec = {**record(e, out), **run.extra(e, out)}
        finally:
            quiet = quiesce(Api(home), (out or {}).get("tools") or [], max(0, len(running) - 1)) if arm == "mms" else {}
            left = scrub_home(home, kept, projects=[exec_project(q)]) if arm == "mms" else None
            with lock:
                ended.add(exec_project(q))
            try:
                empty(work)  # what the exec wrote is gone before another exec can reach it
            except OSError:
                pass  # the next exec in this slot empties it again before it starts
        rec.update(held_s=round(waited, 1), **({
            "gpu_estimate_s": round(estimate_s[q["qid"]], 1), "exclusive": heavy(q), "beyond_timeout": estimate_s[q["qid"]] > run.ask_limit_s,
            "busy_at_start": busy_at_start, "left": left, **quiet} if arm == "mms" else {}))
        if arm == "files" and (bad := changed(run.data, run.manifest)):
            rec.update(problem=f"escaped: the data changed: {bad}", passed=False)
            arm_halt.set(f"{name}'s files arm changed the data ({bad})")
        if arm == "mms" and needs_gpu(model) and (lost := gpu_lost()):  # a setup problem: asked again, and the agent stops
            rec.update(problem=f"setup: the GPU was lost, so mms's model may not have read: {lost}", passed=False)
        write_json(result_path(run, name, arm, model, q["qid"]), rec)
        print(f"{name} {arm_dir(arm, model)} {q['qid']}: {'pass' if rec['passed'] else 'FAIL'} {rec['seconds']:g} s"
              f"{'  ' + rec['problem'] if rec['problem'] else ''}", flush=True)
        halt.after(name, rec)

    def next_question() -> dict | None:
        busy = [q for q, _ in running.values()]
        templates = {run.needle(q) for q in busy}
        if arm != "mms":
            return next((q for q in pending if run.needle(q) not in templates), None)
        if any(heavy(q) for q in busy):
            return None  # an exec holds the GPU
        serial = any(q["source"] in AS_OF for q in busy)
        return next((q for q in pending if not (serial and q["source"] in AS_OF) and run.needle(q) not in templates
                     and not (heavy(q) and busy)), None)

    try:
        with ThreadPoolExecutor(jobs) as pool:
            try:
                while True:
                    while free and not (halt.reason or arm_halt.reason) and (q := next_question()) is not None:
                        pending.remove(q)
                        k = free.pop(0)
                        running[pool.submit(one, q, k)] = (q, k)
                    if free and pending and not (halt.reason or arm_halt.reason):
                        held.setdefault(pending[0]["qid"], time.monotonic())  # first in line, a slot free, and a rule says wait
                    if not running:
                        return
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        free.append(running.pop(future)[1])
                        future.result()
            except BaseException:  # an error, or Ctrl+C: nothing more starts, the running execs end
                arm_halt.set("stopped by an error or an interrupt")
                raise
    finally:
        if arm == "mms" and ended:
            scrub_home(home, kept, projects=sorted(ended))  # the labels of jobs that outlived their execs


def ambient_labels(run: Run, api) -> list[dict]:
    """The labels `ask` made with no project (its labels carry the caller's project), over the week. A pinned project makes
    none; any means an exec's ask had no project, so another question's default scope could hold its answer."""
    since = datetime.combine(run.days[0], clock.min).timestamp()
    until = datetime.combine(run.days[-1] + timedelta(days=1), clock.min).timestamp()
    out: list[dict] = []
    after = None
    while True:
        page = api.call("find", types=["label"], scope="all", since=since, until=until, limit=200,
                        **({"after": after} if after else {}))
        out += [r for r in page["results"] if r.get("by") == "ask" and r.get("project") is None]
        after = page["next"]
        if after is None:
            return out


def check_paths(run: Run) -> None:
    """The files arm's folders hold none of its deny strings, else every command naming them would be refused."""
    deny = run.deny(run.slot(0))
    for folder in (run.data, run.bin, run.slot(0)):
        if hit := codex_run.reached(str(folder), deny):
            raise ValueError(f"{folder} holds {hit!r}, which the files arm refuses to reach: build with --temp elsewhere")


def run_agents(run: Run, agents: dict[str, Agent], models: list[Model], jobs: int = JOBS, *, controls: str | None = None,
               subset: str = "all", only: Sequence[str] = ()) -> None:
    """The arms with no model (`run.arms`) of every agent over every question of the subset, then the mms arm of each model, for
    each agent: the questions `arm_questions` gives it (the text and records only in the arm of `controls`, or of every model when
    it is None), `jobs` at a time (`run.mms_jobs`: one for a model slower than SLOW_S a window). An agent that stops (its plan is
    out, or at STOP_AT) leaves its remaining execs for the next command."""
    if not 1 <= jobs <= JOBS_MAX:
        raise ValueError(f"jobs is 1 to {JOBS_MAX}, not {jobs}: more at once have crashed the GPU driver")
    run.check_paths()
    run.ensure_data()
    plan = run.plan()
    questions = run.questions(plan, subset, models, only)
    halts = {name: Halt() for name in agents}
    if agents and (bad := changed(run.data, run.manifest)):
        raise RuntimeError(f"the data changed before the run ({bad}): build a new run")
    try:
        for name, agent in agents.items():
            for arm in run.arms:
                run_arm(run, name, agent, arm, None, questions, None, jobs, halts[name])
        for model in models:
            note_model(run, model, asked_types(model, controls))
            mine = arm_questions(questions, model, controls)
            for name, agent in agents.items():
                if halts[name].reason or all(is_done(load_result(result_path(run, name, "mms", model, q["qid"]))) for q in mine):
                    continue
                check_gpu(model)
                home = copy_home(run.snapshot(), run.folder / "homes" / f"{name}-{model.slug}", model)
                daemon = Daemon(home).start()
                try:
                    warm(daemon.api, run.warm_params(plan))
                    scrub_home(home, kept_items(run.snapshot()), prefix="id:bench-")  # the warm ask's label, and an interrupted exec's
                    run_arm(run, name, agent, "mms", model, mine, home, run.mms_jobs(model, jobs), halts[name])
                    labels = ambient_labels(run, daemon.api)
                finally:
                    daemon.stop()
                file = run.folder / "results" / name / arm_dir("mms", model) / "ambient-labels.json"
                if labels:
                    write_json(file, labels)
                    print(f"{name} {model.name}: {len(labels)} ask labels have no project: see {file}", flush=True)
                elif file.exists():
                    file.unlink()
    finally:
        run.release()


def real_agents(timeout: float = TIMEOUT_S) -> dict[str, Agent]:
    """Claude Code and Codex, each adapted from an `Exec` to its own function, each exec allowed `timeout` seconds. Running one
    spends the owner's plan."""
    def claude(e: Exec) -> dict:
        return claude_run.ask_claude(e.prompt, e.run_dir, e.work, arm=e.arm, home=e.home, data=e.data,
                                     notes=(e.model.about.notes or None) if e.model else None, timeout=timeout, deny=e.deny,
                                     bin_dir=e.bin_dir)

    def codex(e: Exec) -> dict:
        return codex_run.ask_codex(e.prompt, e.home, e.run_dir, timeout=timeout, arm=e.arm, work=e.work, deny=e.deny,
                                   bin_dir=e.bin_dir, notes=e.model.about.notes if e.model else "")

    return {"claude": claude, "codex": codex}


# -- statistics ---------------------------------------------------------------------------------------------------------

def mcnemar(b: int, c: int) -> float:
    """McNemar's exact two-sided p: b pairs right only with mms and c right only without, under a fair coin."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson score interval (95%) of k out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def newcombe(a: int, b: int, c: int, d: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95% hybrid score interval of Newcombe (1998, method 10) for the difference of two proportions measured on the same
    pairs: `a` pairs right in both arms, `b` right with mms only, `c` right without only, `d` right in neither. The difference is
    (b - c) / n, and the correlation of the arms narrows the Wilson intervals' sum (with the paper's correction toward 0)."""
    n = a + b + c + d
    if n == 0:
        raise ValueError("no pairs")
    p1, p2 = (a + b) / n, (a + c) / n
    (l1, u1), (l2, u2) = wilson(a + b, n, z), wilson(a + c, n, z)
    spread = (a + b) * (c + d) * (a + c) * (b + d)
    cross = a * d - b * c
    cross = cross - n / 2 if cross > n / 2 else 0.0 if cross >= 0 else cross
    phi = cross / math.sqrt(spread) if spread else 0.0
    low = (p1 - p2) - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    high = (p1 - p2) + math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return low, high


def pair_counts(rows: list[tuple]) -> tuple[int, int, int, int]:
    """(a, b, c, d) of paired rows `(question, files result, mms result)`: right in both, right with mms only, right without only,
    right in neither."""
    pairs = [(bool(m["passed"]), bool(f["passed"])) for _, f, m in rows]
    return (sum(1 for x, y in pairs if x and y), sum(1 for x, y in pairs if x and not y),
            sum(1 for x, y in pairs if y and not x), sum(1 for x, y in pairs if not x and not y))


def interval(a: int, b: int, c: int, d: int) -> list[float] | None:
    """The interval of the difference when there are MIN_DISCORDANT pairs or more that are right in one arm only, else None: with
    fewer, an interval that excludes 0 is no sign (McNemar's p says as much)."""
    return list(newcombe(a, b, c, d)) if b + c >= MIN_DISCORDANT else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def ordered(names) -> list[str]:
    """Model names in the order mms lists them, then any other (a test decider) by name."""
    known = [a.name for a in decider.known_abouts()]
    return sorted(set(names), key=lambda n: (known.index(n) if n in known else len(known), n))


def token_total(r: dict) -> int | None:
    """What a result spent in tokens: input and output, and for Claude the cache it read and wrote too (its `input` leaves the
    cache out; Codex's already holds its cached tokens). None for a result that gives none."""
    t = r.get("tokens")
    if not t:
        return None
    cache = (t.get("cache_read") or 0) + (t.get("cache_write") or 0) if r.get("agent") == "claude" else 0
    return t["input"] + t["output"] + cache


def side(rows: list[dict]) -> dict:
    """The figures of one arm's results over the paired questions."""
    absent = [r for r in rows if r["absent"]]
    needles = [r for r in rows if not r["absent"]]
    tokens = [r["tokens"] for r in rows if r.get("tokens")]
    totals = [t for r in rows if (t := token_total(r)) is not None]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    return {"correct": sum(r["passed"] for r in rows), "accuracy": sum(r["passed"] for r in rows) / len(rows) if rows else None,
            "n": len(rows), "absent": len(absent),
            "absent_answered": sum(1 for r in absent if r["answer"] is not None and not r["abstained"]),
            "needles": len(needles), "abstained": sum(1 for r in needles if r["abstained"]),
            "median_s": median([r["seconds"] for r in rows]), "median_calls": median([r["calls"] for r in rows]),
            "seconds": sum(r["seconds"] for r in rows), "timeouts": sum(1 for r in rows if kind_of(r["problem"]) == "timeout"),
            "tokens": {k: sum(t[k] for t in tokens) for k in ("input", "output", "cache_read", "cache_write")} if tokens else None,
            "tokens_mean": mean(totals), "tokens_all": sum(totals) if totals else None, "tokens_n": len(totals),
            "cost_usd": sum(costs) if costs else None}


def results_of(run: Run) -> dict[str, dict[str, dict[str, dict]]]:
    """Every result record of the run: agent, then arm folder, then qid. A record made before results held `work` gets it from
    its exec's tools.json, or None when that is gone."""
    out: dict = {}
    root = run.folder / "results"

    def load(file: Path) -> dict:
        rec = read_json(file)
        if "work" not in rec:
            rec["work"] = work_from_file(run.folder / rec["exec"] / "tools.json") if rec.get("exec") else None
        return rec

    for agent in sorted(p for p in root.glob("*") if p.is_dir() and p.name != "decider"):
        out[agent.name] = {arm.name: {f.stem: load(f) for f in sorted(arm.glob("*.json")) if f.stem not in NOT_RESULTS}
                           for arm in sorted(agent.iterdir()) if arm.is_dir()}
    return out


def paired(rows: list[tuple]) -> dict:
    """The comparison over paired questions `(question, files result, mms result)`: both arms' figures, the difference (the mean
    of the paired differences) with Newcombe's interval when `interval` gives one, and b (right only with mms), c (right only
    without) with McNemar's exact p. The p is what a verdict rests on; b and c are always told."""
    a, b, c, d = pair_counts(rows)
    files, mms = side([f for _, f, _ in rows]), side([m for _, _, m in rows])
    return {"n": len(rows), "files": files, "mms": mms, "diff": (b - c) / len(rows) if rows else None,
            "ci": interval(a, b, c, d) if rows else None, "b": b, "c": c, "p": mcnemar(b, c) if rows else None}


def lean(rows: list[tuple]) -> dict:
    """`paired` for a level: the same difference, interval and counts, with each arm's figures. A files result is counted once
    however many mms results it is paired with (pooled over models, it is paired with one for each), so `n_files` can be below
    `n`; that is for display, the difference is the mean over the pairs."""
    files = list({id(f): f for _, f, _ in rows}.values())
    a, b, c, d = pair_counts(rows)
    return {"n": len(rows), "n_files": len(files), "files": side(files), "mms": side([m for _, _, m in rows]),
            "diff": (b - c) / len(rows) if rows else None, "ci": interval(a, b, c, d) if rows else None, "b": b, "c": c,
            "p": mcnemar(b, c) if rows else None}


def by_level(rows: list[tuple]) -> dict[str, dict]:
    """`lean` of the paired rows at each level of the question's hint that has any, from the least vague."""
    return {level: lean(of) for level in LEVELS if (of := [r for r in rows if level_of(r[0]) == level])}


def cost_ratios(c: dict) -> tuple[float | None, float | None]:
    """mms's seconds and tokens per correct answer as a share of files', for a level of `by_level`; None where an arm has fewer
    than MIN_CORRECT correct answers (a cost per correct answer of two or three is no figure) or no tokens."""
    f, m = c["files"], c["mms"]
    if min(f["correct"], m["correct"]) < MIN_CORRECT:
        return None, None
    seconds = ratio(m["seconds"] / m["correct"], f["seconds"] / f["correct"])
    tokens = (ratio(m["tokens_all"] / m["correct"], f["tokens_all"] / f["correct"])
              if f["tokens_all"] and m["tokens_all"] and f["tokens_n"] == f["n"] and m["tokens_n"] == m["n"] else None)
    return seconds, tokens


def crossover(levels: dict[str, dict], min_n: int | None = None, *,
              order: Sequence[str] = CROSS_LEVELS) -> dict | None:
    """The first of part, day and week (or of `order`) that has `min_n` paired questions (MIN_LEVEL_N) and where mms leads by LEAD, or costs HALF
    as much per correct answer or less (seconds, or tokens) without being less accurate (a difference of 0 or more, or an
    interval whose low end is above -NOT_WORSE). Its `by` says which of "accuracy" and "cost" held, with mms's lead and its cost
    per correct answer as a share of files'; None when no level qualifies."""
    min_n = MIN_LEVEL_N if min_n is None else min_n
    for level in order:
        c = levels.get(level)
        if not c or c["n"] < max(1, min_n):
            continue
        lead, by = c["diff"], []
        seconds, tokens = cost_ratios(c)
        if lead is not None and lead >= LEAD - 1e-9:
            by.append("accuracy")
        not_worse = lead is not None and (lead >= 0 or (c.get("ci") is not None and c["ci"][0] > -NOT_WORSE))
        if not_worse and any(x is not None and x <= HALF for x in (seconds, tokens)):
            by.append("cost")
        if by:
            return {"level": level, "by": by, "lead": lead, "seconds": seconds, "tokens": tokens}
    return None


def thin_levels(levels: dict[str, dict], min_n: int | None = None, *,
                order: Sequence[str] = CROSS_LEVELS) -> dict[str, int]:
    """The levels of part, day and week (or of `order`) that have questions but fewer than `min_n` pairs (MIN_LEVEL_N), with how many."""
    min_n = MIN_LEVEL_N if min_n is None else min_n
    return {level: levels[level]["n"] for level in order
            if level in levels and 0 < levels[level]["n"] < max(1, min_n)}


def worked(rs: list[dict]) -> dict:
    """How an arm worked over its results: the share that called `ask`, asks, finds and looks per exec (from the tool counts every
    result has), and from `work` (`known` of the results have it) the windows and GPU seconds of the ask jobs per exec, the jobs
    stopped at `max_s`, and the pictures viewed and shell commands per exec."""
    tools = [r.get("tools") or {} for r in rs]
    got = [r["work"] for r in rs if r.get("work")]
    return {"n": len(rs), "ask_execs": sum(1 for t in tools if t.get("ask")),
            **{k: mean([t.get(name, 0) for t in tools]) for k, name in (("asks", "ask"), ("finds", "find"), ("looks", "look"))},
            "known": len(got), "windows": mean([w["windows"] for w in got]), "gpu_s": mean([w["seconds"] for w in got]),
            "estimate_s": mean([r["gpu_estimate_s"] for r in rs if r.get("gpu_estimate_s") is not None]),
            "held_s": mean([r["held_s"] for r in rs if r.get("held_s") is not None]),
            "alone": sum(1 for r in rs if r.get("exclusive")), "beyond": sum(1 for r in rs if r.get("beyond_timeout")),
            "max_s": sum(w["stopped"].get("max_s", 0) for w in got), "views": mean([w["views"] for w in got]),
            "shells": mean([w["shells"] for w in got])}


def per_correct(rs: list[dict]) -> dict:
    """What a correct answer cost an arm: its results' seconds, tokens (k) and GPU seconds, each divided by the correct ones
    (`correct` is k: the report tells a cost only from MIN_CORRECT). The tokens are told only when every result gives them and the
    GPU seconds only when every result has `work` (else they would be too few)."""
    k = sum(r["passed"] for r in rs)
    totals = [t for r in rs if (t := token_total(r)) is not None]
    got = [r["work"] for r in rs if r.get("work")]
    return {"correct": k, "n": len(rs), "s": sum(r["seconds"] for r in rs) / k if k else None,
            "tokens_k": sum(totals) / 1000 / k if k and totals and len(totals) == len(rs) else None,
            "gpu_s": sum(w["seconds"] for w in got) / k if k and got and len(got) == len(rs) else None}


def compare(agent: str, info: dict, loaded: list[tuple]) -> dict | None:
    """One model's figures for one agent, pooled over the runs: of the questions its mms arm is asked (`asked`, else what it
    reads), those with a done result in both arms are paired (an escaped, timed-out or capped one is a wrong answer); one
    with a result to run again (`setup`, `limit`, `auth`, `error`) is left out, and one with none is pending. `all` is every
    paired question, `media` the pictures, video and sound, the part that depends on the model; `levels` and `crossover` are
    over the media by how vague the hint is, `work` and `cost` how each arm worked and what a correct answer cost it on them.
    `rows` are the paired media questions, for `numbers` to pool. None when the agent has no mms result for the model."""
    rows, asked, left_out, seen = [], 0, 0, False
    kinds: dict[str, Counter] = {"files": Counter(), "mms": Counter()}
    types_asked = info.get("asked") or info["readable"]
    for plan, results in loaded:
        arms = {"files": results.get(agent, {}).get("files", {}), "mms": results.get(agent, {}).get(f"mms-{info['slug']}", {})}
        seen = seen or bool(arms["mms"])
        for q in plan.questions:
            if q["type"] not in types_asked:
                continue
            asked += 1
            f, m = arms["files"].get(q["qid"]), arms["mms"].get(q["qid"])
            for arm, r in (("files", f), ("mms", m)):
                if r is not None and r["problem"]:
                    kinds[arm][kind_of(r["problem"])] += 1
            if is_done(f) and is_done(m):
                rows.append((q, f, m))
            elif (f is not None and not is_done(f)) or (m is not None and not is_done(m)):
                left_out += 1
    if not seen:
        return None
    types = {}
    for type_ in haystack.TYPES:
        of = [(f, m) for q, f, m in rows if q["type"] == type_]
        if of:
            types[type_] = {"n": len(of), "files": sum(f["passed"] for f, _ in of), "mms": sum(m["passed"] for _, m in of)}
    media = [r for r in rows if r[0]["type"] in MEDIA]
    levels = by_level(media)
    return {"questions": asked, "left_out": left_out, "pending": asked - len(rows) - left_out,
            "kinds": {arm: dict(counts) for arm, counts in kinds.items()}, "all": paired(rows), "media": paired(media),
            "levels": levels, "crossover": crossover(levels), "thin": thin_levels(levels),
            "work": {"files": worked([f for _, f, _ in media]), "mms": worked([m for _, _, m in media])},
            "cost": {"files": per_correct([f for _, f, _ in media]), "mms": per_correct([m for _, _, m in media])},
            "scope_all": sum(m["scope_all"] for _, _, m in rows), "types": types, "rows": media}


# the wordings of a decider tier verdict on a needle question that a model did not find
SUSPECT = "haystack suspect"
CALIBRATION = "below its model's hit (calibration)"
MISS = "this model's miss"
STOPPED = "stopped on a false sighting before the needle"
CUT = "cut at max_s before the needle"
ONE_MODEL = "one model ran: the agent arm's look, or a second family, tells a miss from a haystack fault"
ONE_FAMILY = "one family ran: the agent arm's look, or a second family, tells a miss from a haystack fault"
OUTCOMES = ("found", "near", "stopped", "cut", "missed")


def outcome(r: dict) -> str:
    """How an oracle record scored its needle: `found` (a sighting); `near` (not found, but among the top 5 candidates or with
    p of NEAR_P or more: the scan reached it); `stopped` (the ask met its limit of sightings, all false or of earlier needles,
    before it reached the needle: a `last` ask stops after LAST_LIMIT sightings, newest first, before it reaches the needle);
    `cut` (stopped at `max_s`, so the needle may not have been reached); else `missed`, a scan that went over it and saw
    nothing."""
    if r["found"]:
        return "found"
    if r["top5"] or (r.get("p") or 0) >= NEAR_P:
        return "near"
    if r.get("stopped") == "limit" and r.get("false"):
        return "stopped"
    return "cut" if r.get("stopped") == "max_s" else "missed"


def verdicts(rs: list[dict]) -> dict[str, str]:
    """For one needle question and the records of each model asked it: the verdict on each model that did not find it.
    - A near miss is below its model's hit (calibration).
    - A scan that ended before the needle (`stopped`, `cut`) says nothing of the model's eyes or the haystack, and is told as it
      is; it does not count as a family that ran.
    - A miss is the model's when any model found the needle, or a model of another family had it in its top 5.
    - Only when nobody found it and models of two or more families scanned it and none had it in the top 5 is the haystack
      suspect. With one family the decider tier cannot tell (the agent arm's look can)."""
    fair = [r for r in rs if outcome(r) in ("found", "near", "missed")]
    families = {family(r["model"]) for r in fair}
    found = any(outcome(r) == "found" for r in rs)
    out: dict[str, str] = {}
    for r in rs:
        got = outcome(r)
        if got == "found":
            continue
        if got in ("near", "stopped", "cut"):
            out[r["model"]] = {"near": CALIBRATION, "stopped": STOPPED, "cut": CUT}[got]
        elif found or any(outcome(x) == "near" for x in rs if family(x["model"]) != family(r["model"])):
            out[r["model"]] = MISS
        elif len(families) >= 2:
            out[r["model"]] = SUSPECT
        else:
            out[r["model"]] = ONE_MODEL if len(fair) == 1 else ONE_FAMILY
    return out


def decider_figures(runs: list[Run], plans: list[haystack.Plan]) -> dict:
    """The decider tier: per model and type, from the oracle records of all runs (a record made before the levels and the
    needle's p gets them from the plan), and per template who found it, nearly missed it or missed it, with the verdicts."""
    records: dict[str, list[dict]] = {}
    firsts: dict[str, list[float]] = {}
    for run, plan in zip(runs, plans):
        questions = {q["qid"]: q for q in plan.questions}
        for folder in sorted((run.folder / "results" / "decider").glob("*")):
            for file in sorted(folder.glob("*.json")):
                r = read_json(file)
                if file.stem == "warm":
                    firsts.setdefault(r["model"], []).append(r["seconds"])
                    continue
                q = questions.get(r["qid"])
                r = {**r, "run": run.name, "level": r.get("level") or level_of(q or {})}
                if "p" not in r:
                    r["p"] = None if r["absent"] or q is None else needle_p(q["truth"]["moments"], r["sightings"], r["candidates"])
                records.setdefault(r["model"], []).append(r)

    def near(r: dict) -> bool:
        return bool(r["found"]) or (r.get("p") or 0) >= NEAR_P

    def recall(of: list[dict]) -> dict:
        needles = [r for r in of if not r["absent"]]
        return {"asked": len(of), "needles": len(needles), "found": sum(bool(r["found"]) for r in needles),
                "p50": sum(near(r) for r in needles), "top5": sum(bool(r["top5"]) for r in needles),
                "false": sum(r["false"] for r in of),
                "s_per_window": median([r["seconds"] / r["windows"] for r in of if r["windows"]])}

    models = {}
    for name in ordered(records):
        types = {}
        for type_ in haystack.TYPES:
            of = [r for r in records[name] if r["type"] == type_]
            if of:
                absent = [r for r in of if r["absent"]]
                types[type_] = {**recall(of), "absent": len(absent), "absent_clean": sum(1 for r in absent if r["false"] == 0)}
        models[name] = {"family": family(name), "first_s": median(firsts.get(name, [])), "types": types,
                        "media": recall([r for r in records[name] if r["type"] in MEDIA])}
    asked: dict[tuple, list[dict]] = {}  # (run, qid) -> the records of each model asked that needle question
    for rs in records.values():
        for r in rs:
            if not r["absent"]:
                asked.setdefault((r["run"], r["qid"]), []).append(r)
    templates: dict[str, dict] = {}
    for _, rs in sorted(asked.items()):
        t = templates.setdefault(rs[0]["template"], {"questions": 0, "models": {}, "verdicts": {}})
        t["questions"] += 1
        said = verdicts(rs)
        for r in rs:
            m = t["models"].setdefault(r["model"], {"asked": 0, **{o: 0 for o in OUTCOMES}, "ps": []})
            m["asked"] += 1
            m[outcome(r)] += 1
            if r.get("p") is not None:
                m["ps"].append(r["p"])
            if r["model"] in said:
                v = t["verdicts"].setdefault(said[r["model"]], {"models": [], "levels": []})
                v["models"] = ordered({*v["models"], r["model"]})
                v["levels"] = sorted({*v["levels"], r["level"]}, key=lambda x: LEVELS.index(x) if x in LEVELS else len(LEVELS))
    for t in templates.values():
        for m in t["models"].values():
            m["p"] = median(m.pop("ps"))
        t["models"] = {n: t["models"][n] for n in ordered(t["models"])}
    found = {r["template"] for rs in records.values() for r in rs if r["found"]}
    return {"models": models, "templates": dict(sorted(templates.items())), "missed": sorted(set(templates) - found),
            "asks": sum(len(rs) for rs in records.values())}


def cell_of(r: dict | None) -> dict | None:
    """What the appendix keeps of one exec's result: whether it passed, its seconds and problem kind, the pictures it viewed
    (None when the result does not say) and the asks it made."""
    if r is None:
        return None
    return {"passed": bool(r["passed"]), "seconds": r["seconds"], "problem": kind_of(r["problem"]), "beyond": bool(r.get("beyond_timeout")),
            "views": (r.get("work") or {}).get("views"), "asks": (r.get("tools") or {}).get("ask", 0)}


def appendix_rows(runs: list[Run], loaded: list[tuple], infos: dict[str, dict]) -> dict[str, list[dict]]:
    """Per agent, a row for each question some arm ran: qid, level, type, the files arm's cell and each model's mms cell."""
    out: dict[str, list[dict]] = {}
    for run, (plan, results) in zip(runs, loaded):
        for agent, arms in results.items():
            for q in plan.questions:
                files = arms.get("files", {}).get(q["qid"])
                mms = {name: cell_of(arms[f"mms-{info['slug']}"].get(q["qid"])) for name, info in infos.items()
                      if f"mms-{info['slug']}" in arms and q["qid"] in arms[f"mms-{info['slug']}"]}
                if files is not None or mms:
                    out.setdefault(agent, []).append({"seed": run.seed, "qid": q["qid"], "level": level_of(q), "type": q["type"],
                                                      "files": cell_of(files), "mms": mms})
    return out


def agent_models(loaded: list[tuple]) -> dict[str, dict[str, int]]:
    """Per agent, how many of its results each model gave, with the effort when the agent records one (Codex), so results of
    two settings pooled by mistake show."""
    out: dict[str, Counter] = {}
    for _, results in loaded:
        for agent, arms in results.items():
            for rs in arms.values():
                for r in rs.values():
                    name = str(r.get("agent_model") or "unknown") + (f" at {r['agent_effort']} effort" if r.get("agent_effort") else "")
                    out.setdefault(agent, Counter())[name] += 1
    return {agent: dict(sorted(c.items())) for agent, c in sorted(out.items())}


def numbers(runs: list[Run]) -> dict:
    """The report's figures over `runs`, pooled (report.json; `markdown` renders them): per agent and model the paired
    comparison, by hint level and per type, how each arm worked and what a correct answer cost, the same by level pooled over
    the models, the decider tier, one row per question, and the counts that always print."""
    first: dict[tuple, str] = {}
    for run in runs:
        if (run.seed, run.scale) in first:
            raise ValueError(f"{run.name} and {first[run.seed, run.scale]} are the same seed ({run.seed}) and scale "
                             f"({run.scale}), so the same questions (the plan does not depend on the days): pooling them "
                             "would count each question twice")
        first[run.seed, run.scale] = run.name
    loaded = [(run.plan(), results_of(run)) for run in runs]
    infos: dict[str, dict] = {}
    for run in runs:
        file = run.folder / "models.json"
        infos.update(read_json(file) if file.exists() else {})
    agents: dict[str, dict] = {}
    pooled: dict[str, dict] = {}
    for agent in sorted({a for _, results in loaded for a in results}):
        rows: list[tuple] = []
        for name in ordered(infos):
            if (row := compare(agent, infos[name], loaded)) is not None:
                rows += row.pop("rows")
                agents.setdefault(agent, {})[name] = row
        if agent in agents:
            levels = by_level(rows)
            pooled[agent] = {"levels": levels, "crossover": crossover(levels), "thin": thin_levels(levels)}
    every = [r for _, results in loaded for arms in results.values() for rs in arms.values() for r in rs.values()]
    problems = Counter(kind_of(r["problem"]) for r in every if r["problem"])
    ambient = sum(len(read_json(f)) for run in runs for f in (run.folder / "results").glob("*/mms-*/ambient-labels.json"))
    return {"runs": [{"name": run.name, "seed": run.seed, "scale": run.scale, "days": len(run.days),
                      "commit": read_json(run.folder / "run.json")["mms_commit"], "probe": probe_of(run.folder)} for run in runs],
            "agents": agents, "agent_models": agent_models(loaded), "pooled": pooled,
            "decider": decider_figures(runs, [plan for plan, _ in loaded]),
            "appendix": {a: rows for a, rows in appendix_rows(runs, loaded, infos).items() if a in agents},
            "totals": {"results": len(every), "problems": dict(problems),
                       "to_rerun": sum(1 for r in every if not is_done(r)), "ambient_labels": ambient}}


# -- the report ---------------------------------------------------------------------------------------------------------

def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|",
                      *("| " + " | ".join(row) + " |" for row in rows)])


def share(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def number(x: float | None, spec: str = ".1f") -> str:
    return "-" if x is None else format(x, spec)


def chance(p: float | None) -> str:
    return "-" if p is None else "<0.001" if p < 0.001 else f"{p:.3f}"


def kilo(arm: dict) -> str:
    """An arm's tokens in thousands (see `token_total`)."""
    return "-" if arm["tokens_all"] is None else f"{arm['tokens_all'] / 1000:.0f}"


def diff_cell(c: dict) -> str:
    """A difference in points with its interval; with fewer than MIN_DISCORDANT pairs right in one arm only, no interval, and
    b and c instead."""
    if c["diff"] is None:
        return "-"
    if c["ci"] is None:
        return f"{pts(c['diff'])} (too few discordant pairs: b {c['b']}, c {c['c']})"
    return f"{pts(c['diff'])} [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]"


def both(files, mms, spec: str = ".1f") -> str:
    return f"{number(files, spec)} / {number(mms, spec)}"


def kinds_text(counts: dict) -> str:
    return ", ".join(f"{kind} {n}" for kind, n in sorted(counts.items())) or "none"


def pts(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:+.0f} pts"


def tokens_k(arm: dict) -> float | None:
    """A side's mean tokens per exec, in thousands."""
    return None if arm["tokens_mean"] is None else arm["tokens_mean"] / 1000


def known_of(w: dict, key: str, spec: str = ".1f") -> str:
    """A per-exec mean of `worked`, with how many of the results it comes from when that is not all of them."""
    if w[key] is None:
        return "-"
    return format(w[key], spec) + ("" if w["known"] == w["n"] else f" ({w['known']}/{w['n']})")


def crossover_text(c: dict | None, thin: dict | None = None) -> str:
    """The crossover of a model or of all: the level and which branch held (accuracy, cost per correct answer, or both), or why
    there is none, naming the levels that have too few pairs."""
    if c is None:
        few = ", ".join(f"{level} ({n})" for level, n in (thin or {}).items())
        return (f"none: at no level with {MIN_LEVEL_N} or more paired questions does mms lead by 20 points, or cost half as much "
                "per correct answer without being less accurate" + (f"; too few paired questions at {few}" if few else ""))
    bits = [f"accuracy {pts(c['lead'])}"]
    if "cost" in c["by"]:
        bits += [f"{c[k]:.2f}x the {k} per correct answer" for k in ("seconds", "tokens") if c[k] is not None]
    return f"`{c['level']}` by {' and '.join(c['by'])} ({', '.join(bits)})"


def level_lines(figures: dict) -> list[str]:
    """Per agent, accuracy with and without mms, seconds, tokens and timeouts at each level of the hint, per model and pooled over
    the models, and the crossover sentence for each."""
    out: list[str] = []
    for agent, models in figures["agents"].items():
        pooled = figures["pooled"][agent]
        groups = [(name, m["levels"], m["crossover"], m["thin"]) for name, m in models.items()]
        if len(models) > 1:
            groups.append(("all models", pooled["levels"], pooled["crossover"], pooled["thin"]))
        rows, notes = [], []
        for label, levels, cross, thin in groups:
            for level, c in levels.items():
                f, w = c["files"], c["mms"]
                rows.append([label, level, str(c["n"]), share(f["correct"], c["n_files"]), share(w["correct"], c["n"]),
                             diff_cell(c), f"{c['b']} / {c['c']}", chance(c["p"]), both(f["median_s"], w["median_s"]),
                             both(tokens_k(f), tokens_k(w), ".0f"), f"{f['timeouts']} / {w['timeouts']}"])
            notes.append(f"- {label}: " + (crossover_text(cross, thin) if levels else "no media question paired"))
        out += ["", f"## {agent}: by how vague the time hint is (the media questions, paired)", "",
                table(["model", "level", "paired", "without mms", "with mms", "difference (95% CI)", "b / c", "McNemar p",
                       "median s (files / mms)", "tokens k per exec (files / mms)", "timed out (files / mms)"], rows),
                "", "Crossover, the first of part, day and week with at least 10 paired questions where mms leads by 20 points "
                "or more, or costs half as much per correct answer or less (seconds, or tokens) without being less accurate:",
                "", *notes]
    return out


def worked_lines(figures: dict) -> list[str]:
    """How each arm worked, per agent and model, per exec on the paired media questions."""
    rows = []
    for agent, models in figures["agents"].items():
        for name, m in models.items():
            f, w = m["work"]["files"], m["work"]["mms"]
            rows.append([agent, name, "files", str(f["n"]), "-", "-", "-", "-", "-", "-", "-", "-", known_of(f, "views"),
                         known_of(f, "shells")])
            rows.append([agent, name, "mms", str(w["n"]), share(w["ask_execs"], w["n"]),
                         " / ".join(number(w[k]) for k in ("asks", "finds", "looks")), known_of(w, "windows"),
                         known_of(w, "gpu_s"), str(w["max_s"]) if w["known"] else "-", number(w["estimate_s"]),
                         f"{number(w['held_s'])} ({w['alone']} alone)" if w["held_s"] is not None else "-", str(w["beyond"]), "-", "-"])
    return ["", "## How each arm worked (the media questions, per exec)", "",
            table(["agent", "model", "arm", "execs", "called ask", "asks / finds / looks", "ask windows", "ask GPU s",
                   "asks stopped at max_s", "GPU s estimated", "held s (run alone)", "estimated past the timeout", "pictures viewed", "shell commands"], rows)]


def cost_cell(c: dict, key: str, spec: str = ".1f") -> str:
    """A cost per correct answer, told only from MIN_CORRECT correct answers (k is beside it)."""
    return number(c[key], spec) if c["correct"] >= MIN_CORRECT else "-"


def cost_lines(figures: dict) -> list[str]:
    """What a correct answer costs each arm, per agent and model, on the paired media questions. Fewer than MIN_CORRECT correct
    answers (k) make no figure."""
    rows = []
    for agent, models in figures["agents"].items():
        for name, m in models.items():
            for arm in ("files", "mms"):
                c = m["cost"][arm]
                rows.append([agent, name, arm, f"{c['correct']}/{c['n']}", cost_cell(c, "s"), cost_cell(c, "tokens_k"),
                             cost_cell(c, "gpu_s") if arm == "mms" else "-"])
    return ["", "## Cost per correct answer (the media questions)", "",
            table(["agent", "model", "arm", "correct (k)", "seconds", "tokens k", "GPU seconds"], rows), "",
            f"A cost per correct answer is told from {MIN_CORRECT} correct answers (k) in the arm; below that it is `-`."]


def verdict_lines(tier: dict) -> list[str]:
    """Per template: who found its needles, who nearly missed them (in the top 5 or with p of 0.5 or more), who missed them,
    whose scan ended before the needle (stopped on a false sighting, or cut at max_s), the needle's p, and the verdict on what a
    miss says."""
    rows = []
    for template, t in tier["templates"].items():
        models = t["models"]
        verdict = ("; ".join(f"{text}: {', '.join(v['models'])} ({', '.join(v['levels'])})" for text, v in t["verdicts"].items())
                   or "found by every model asked")
        rows.append([template, str(t["questions"]),
                     ", ".join(f"{n} {m['found']}/{m['asked']}" for n, m in models.items() if m["found"]) or "-",
                     ", ".join(f"{n} {m['near']}" for n, m in models.items() if m["near"]) or "-",
                     ", ".join(f"{n} {m['missed']}" for n, m in models.items() if m["missed"]) or "-",
                     ", ".join(f"{n} {m['stopped'] + m['cut']}" for n, m in models.items() if m["stopped"] + m["cut"]) or "-",
                     ", ".join(f"{n} {m['p']:.2f}" for n, m in models.items() if m["p"] is not None) or "-", verdict])
    return ["", table(["template", "needle questions", "found by", "nearly missed by (top 5 or p >= 0.5)", "missed by",
                       "scan ended before the needle by (false sighting, max_s)", "needle p (median)", "verdict"], rows)]


def summary_lines(figures: dict) -> list[str]:
    """One row per model: the decider tier's media figures beside each agent's mms accuracy and cost on the media."""
    tier, agents = figures["decider"]["models"], figures["agents"]
    names = ordered({*tier, *(n for models in agents.values() for n in models)})
    if not names:
        return []
    header = ["model", "family", "recall (at its hit)", "recall at p >= 0.5", "top 5", "false sightings per ask",
              "median s per window"]
    for agent in agents:
        header += [f"{agent}: mms accuracy", f"{agent}: s / k tokens / GPU s per correct"]
    rows = []
    for name in names:
        d = tier.get(name)
        m = d["media"] if d else None
        row = [name, family(name), *((share(m["found"], m["needles"]), share(m["p50"], m["needles"]),
                                     share(m["top5"], m["needles"]), number(m["false"] / m["asked"], ".2f") if m["asked"] else "-",
                                     number(m["s_per_window"], ".3f")) if m else ("-",) * 5)]
        for models in agents.values():
            a = models.get(name)
            c = a["cost"]["mms"] if a else None
            row += [share(a["media"]["mms"]["correct"], a["media"]["n"]) if a else "-",
                    f"{cost_cell(c, 's')} / {cost_cell(c, 'tokens_k')} / {cost_cell(c, 'gpu_s')} (k={c['correct']})" if c else "-"]
        rows.append(row)
    return ["", "## Summary per model (the media questions)", "", table(header, rows)]


def cell_text(c: dict | None, what: str) -> str:
    """One arm's result of a question in the appendix: pass or FAIL (and the kind of problem), seconds, then the views (`v`) or
    asks (`a`) it made."""
    if c is None:
        return "-"
    n = c["views"] if what == "views" else c["asks"]
    problem = f" {c['problem']}" if c["problem"] else ""
    flag = " (est. past timeout)" if c.get("beyond") else ""
    return f"{'pass' if c['passed'] else 'FAIL'}{problem} {c['seconds']:.0f} s {'v' if what == 'views' else 'a'}{'-' if n is None else n}{flag}"


def appendix_lines(figures: dict) -> list[str]:
    """One row per question and agent, to read the failures: qid, level, type, the files arm's pass, seconds and pictures viewed
    (`v`), and each model's pass, seconds and asks (`a`)."""
    out: list[str] = []
    many = len(figures["runs"]) > 1
    for agent, rows in figures["appendix"].items():
        names = list(figures["agents"][agent])
        body = [[*([str(r["seed"])] if many else []), r["qid"], r["level"], r["type"], cell_text(r["files"], "views"),
                 *(cell_text(r["mms"].get(n), "asks") for n in names)] for r in rows]
        out += ["", f"## Appendix: {agent}, one row per question", "",
                table([*(["seed"] if many else []), "qid", "level", "type", "files", *(f"mms {n}" for n in names)], body)]
    return out


def markdown(figures: dict) -> str:
    """report.md: the tables of `numbers`, and the lines that always print."""
    out = ["# mms benchmark: with mms and without", ""]
    runs = figures["runs"]
    out.append("Runs pooled: " + "; ".join(f"{r['name']} (seed {r['seed']}, {r['scale']}, {r['days']} days, commit "
                                            f"{r['commit']})" for r in runs) + ".")
    if figures.get("agent_models"):
        out.append("Agents: " + "; ".join(f"{agent} " + ", ".join(f"{name} ({n} results)" for name, n in names.items())
                                          for agent, names in figures["agent_models"].items()) + ".")
    for agent, models in figures["agents"].items():
        out += ["", f"## {agent}: accuracy with and without mms, on each model's questions", ""]
        rows = []
        for name, m in models.items():
            for label, c in (("all", m["all"]), ("media", m["media"])):
                diff = diff_cell(c)
                f, w = c["files"], c["mms"]
                headline = label == "all"  # the counts of what was left out and what mms was asked are of the whole model
                rows.append([name, label, f"{c['n']}/{m['questions']}" if headline else str(c["n"]),
                             share(f["correct"], c["n"]), share(w["correct"], c["n"]), diff, f"{c['b']} / {c['c']}",
                             chance(c["p"]), f"{f['absent_answered']}/{f['absent']} / {w['absent_answered']}/{w['absent']}",
                             f"{f['abstained']}/{f['needles']} / {w['abstained']}/{w['needles']}",
                             both(f["median_s"], w["median_s"]), both(f["median_calls"], w["median_calls"], ".0f"),
                             f"{f['timeouts']} / {w['timeouts']}", f"{kilo(f)} / {kilo(w)}", both(f["cost_usd"], w["cost_usd"], ".2f"),
                             str(m["left_out"]) if headline else "-", str(m["scope_all"]) if headline else "-",
                             f"{kinds_text(m['kinds']['files'])} / {kinds_text(m['kinds']['mms'])}" if headline else "-"])
        out.append(table(["model", "questions", "paired", "without mms", "with mms", "difference (95% CI)", "b / c",
                          "McNemar p", "wrong on absent (files / mms)", "abstained on needles (files / mms)",
                          "median s (files / mms)", "median calls (files / mms)", "timed out (files / mms)",
                          "tokens k (files / mms)", "cost $ (files / mms)", "left out", "scope all", "problems (files / mms)"], rows))
    if figures["agents"]:
        out += level_lines(figures) + worked_lines(figures) + cost_lines(figures)
        out += ["", "## By data type: without mms -> with mms, right/paired", ""]
        rows = [[agent, name, *(f"{c['files']}/{c['n']} -> {c['mms']}/{c['n']}" if (c := m["types"].get(type_)) else "-"
                                for type_ in haystack.TYPES)]
                for agent, models in figures["agents"].items() for name, m in models.items()]
        out.append(table(["agent", "model", *haystack.TYPES], rows))
    else:
        out += ["", "No agent results yet."]
    out += ["", "## The decider tier (each model's yes/no over each needle's source and range, no agent)", ""]
    tier = figures["decider"]
    if tier["asks"]:
        rows = []
        for name, m in tier["models"].items():
            for type_, t in m["types"].items():
                rows.append([name, type_, share(t["found"], t["needles"]), share(t["p50"], t["needles"]),
                             share(t["top5"], t["needles"]), share(t["absent_clean"], t["absent"]),
                             number(t["false"] / t["asked"], ".2f"), number(t["s_per_window"], ".3f"), number(m["first_s"])])
        out.append(table(["model", "type", "recall (at its hit)", "recall at p >= 0.5", "top 5",
                          "absent with no false sighting", "false sightings per ask", "median s per window", "first ask s"],
                         rows))
        out += ["", ("Templates no model found: " + ", ".join(tier["missed"]) + ". The verdicts below say whether that is the "
                     "haystack or the models.") if tier["missed"] else "Every template was found by some model."]
        out += verdict_lines(tier)
    else:
        out.append("No decider results yet.")
    out += summary_lines(figures)
    t = figures["totals"]
    rows_k = sum(len(models) for models in figures["agents"].values())
    low = sum(1 for models in figures["agents"].values() for m in models.values() for c in (m["all"], m["media"])
              if c["p"] is not None and c["p"] < 0.05)
    out += ["", "## Read with these", "",
            "- Without mms the screen, the microphone and the cameras would not be recorded at all: the files arm reads a week "
            "that exists only because the bench wrote it.",
            "- The screen changes like a real one while the owner is active: typing (small changes), scrolling (large ones), a cursor that "
            "moves and a clock, up to 2 frames a second, with idle and busy stretches. Its totals and the size of its changes are calibrated "
            "against two real hours of a screen recording (TESTBED.md). A caret's blinks are not modelled, and a scroll wraps instead of "
            "bringing new lines; the reader's change gate turns 98% of the haystack's seconds with a frame into windows, where it turns "
            "71 to 86% of a real screen's.",
            "- `look` costs about 25 times more per frame than an ffmpeg montage: it returns separate pictures, 8 to a call, "
            "where a 5x5 montage is about 1.5k tokens. An agent that opens frames with `look` pays for it; one that tiles "
            "them does not.",
            f"- With about 20 paired questions, p < 0.05 needs at least 6 discordant pairs all one way (McNemar exact: "
            f"mcnemar(6, 0) = {mcnemar(6, 0):.3f}, mcnemar(5, 0) = {mcnemar(5, 0):.4f}). A run with another seed adds new "
            "needles, and `report` pools runs. By hint level there are a few questions each: pool seeds before reading it.",
            "- The text and records questions do not depend on the model, so only one model's mms arm (d1-3B) is asked them: "
            "the other models' rows hold the media questions.",
            "- The decider tier's recall at p >= 0.5 counts a needle whose best p within 10 s of it, as a sighting or a "
            "candidate, is 0.5 or more: where it is above the recall at the model's hit, that hit may be set too high.",
            "- Tokens are the input, the output and for Claude the cache it read and wrote (Codex's input already holds its cached "
            "tokens): a size of work, not a bill, since a cached token costs much less. Cost is Claude's list-price estimate; "
            "Codex reports none.",
            f"- The difference is told in points with Newcombe's hybrid score interval for paired data, only from {MIN_DISCORDANT} "
            "pairs right in one arm only (b + c); below that the interval would exclude 0 where McNemar's exact p does not, so "
            "b and c and the p are given alone. A verdict rests on McNemar's p. A timed-out exec is a wrong answer and is "
            "counted on its own too (the `timed out` columns).",
            f"- An mms exec whose ask may need more GPU seconds than {ASK_SHARE:.0%} of the exec's time ({ASK_SHARE * TIMEOUT_S:g} s of "
            f"{TIMEOUT_S:g}: the agent's own turns take the rest; the `held s` and `estimated past the timeout` columns, and the appendix) "
            "may time out because of the model's speed alone: that is not a miss of its eyes, and `agents` names those questions before it runs.",
            "- The rows are not independent: the part and day questions of a template share a needle on the `pilot` scale, a "
            "template's levels share its week, and pooled seeds repeat templates. Read a level or a model with a few dozen "
            "questions as a sign to follow up, not as a finding.",
            f"- Multiple comparisons: this report has k = {rows_k} agent-model rows, each tested on all its questions and on "
            f"the media alone ({2 * rows_k} p values). At p < 0.05 about {0.05 * rows_k:.2f} of k rows would fall below it by "
            f"chance alone; {low} of the {2 * rows_k} p values here are.",
            f"- Ask labels with no project (an exec whose questions could leak into another's scope): {t['ambient_labels']}. "
            f"Problems by kind: {kinds_text(t['problems'])}. Results to run again: {t['to_rerun']} of {t['results']}.",
            "- Paired: a question counts when both arms have a done result. An escaped, timed-out or capped exec is a wrong "
            "answer; a `setup`, `limit`, `auth` or `error` result did not test the question, is run again by the next "
            "command, and is left out (the `left out` column).",
            f"- Runs pooled: {len(runs)}; seeds {', '.join(str(r['seed']) for r in runs)}."]
    out[-1:-1] = [f"- Shortcut probe: {line}." for r in runs if (line := probe_line(r["name"], r["probe"]))]
    out += appendix_lines(figures)
    return "\n".join(out) + "\n"


# -- the command line ---------------------------------------------------------------------------------------------------

def resolve_run(text: str) -> Run:
    """A run from its folder, or its name under the default out folder."""
    path = Path(text)
    return Run.open(path if path.is_dir() else OUT / text)


def pending_count(run: Run, agent: str, arm: str, model: Model | None, questions: Sequence[dict]) -> tuple[int, int]:
    todo = sum(1 for q in questions if not is_done(load_result(result_path(run, agent, arm, model, q["qid"]))))
    return todo, len(questions)


def estimate(arms: Sequence[tuple], jobs: int) -> str:
    """About how long the execs take: `arms` is (execs, execs at once) for each arm, at EXEC_MIN minutes an exec, or (execs, execs
    at once, minutes an exec) for one with its own figure (the blind arm: BLIND_MIN)."""
    hours = sum(n * (minutes[0] if minutes else EXEC_MIN) / at_once / 60 for n, at_once, *minutes in arms)
    blind = f" ({BLIND_MIN:g} for a blind one)" if any(a[2:] == (BLIND_MIN,) for a in arms) else ""
    return f"about {hours:.1f} h at {EXEC_MIN:g} min an exec{blind} over {jobs} jobs"


def arm_label(arm: str, model: Model | None) -> str:
    return f"mms {model.name}" if model else arm


def show_agents(run: Run, names: list[str], models: list[Model], skipped: list[str], jobs: int, controls: str | None = None,
                subset: str = "all", only: Sequence[str] = ()) -> None:
    """What `agents` will run, and how long that takes: it asks nothing."""
    plan = run.plan()
    questions = run.questions(plan, subset, models, only)
    total, arms = 0, []
    print(f"run {run.name}: {run.scale}, seed {run.seed}, {len(run.days)} days, the {subset} questions"
          + (f", only {', '.join(only)}" if only else ""))
    if line := probe_line(run.name, probe_of(run.folder)):
        print(f"warning: {line}")
    for name in names:
        for arm, model in (*((a, None) for a in run.arms), *(("mms", m) for m in models)):
            mine = questions if model is None else arm_questions(questions, model, controls)
            todo, of = pending_count(run, name, arm, model, mine)
            total += todo
            arms.append((todo, jobs if model is None else run.mms_jobs(model, jobs), *((BLIND_MIN,) if arm == "blind" else ())))
            print(f"  {name:6} {arm_label(arm, model):24} {todo} to run of {of}")
    for line in skipped:
        print(f"  skipped {line}")
    for model in models:
        if arm_jobs(model, jobs) < jobs:
            print(f"  {model.name}: {window_s(model.about):.2f} s a window, above {SLOW_S:g} s: one exec at a time, so no timeout "
                  "is the GPU queue's")
        elif run.mms_jobs(model, jobs) < jobs:
            print(f"  {model.name}: this run's mms execs go one at a time")
        past = [(q["qid"], run.estimate(plan, model, q)) for q in arm_questions(questions, model, controls)]
        if past := [(qid, seconds) for qid, seconds in past if seconds > run.ask_limit_s]:
            print(f"  {model.name}: an ask over the whole range of {len(past)} questions may need more GPU seconds than the {run.ask_limit_s:g} s "
                  f"an exec leaves its ask ({ASK_SHARE:.0%} of its {run.timeout_s:g} s: the agent's own turns take the rest), so a timeout there "
                  "is its speed, not a miss: " + ", ".join(f"{qid} ({seconds:.0f} s)" for qid, seconds in past))
    print(f"{total} execs to run, {estimate(arms, jobs)}" + ("; the GPU is used by each model's daemon" if models else ""))


def decider_minutes(plan: haystack.Plan, model: Model, questions: Sequence[dict], estimate_=None) -> float | None:
    """Minutes the oracle asks of `questions` take `model`: the windows `haystack.windows_in` counts in each ask's source and
    range, at the model's seconds a window (`ask.window_seconds`) for the source's kind of window (`gpu_estimate`, or the
    `estimate_(plan, model, q)` of a run). None while the haystack has no `windows_in`."""
    if getattr(haystack, "windows_in", None) is None:
        return None
    return sum((estimate_ or gpu_estimate)(plan, model, q) for q in questions) / 60


def minutes_text(minutes: float) -> str:
    return "under a minute" if minutes < 1 else f"about {minutes:.0f} min"


def show_decider(run: Run, models: list[Model], skipped: list[str]) -> None:
    plan = run.plan()
    total, minutes = 0, None
    print(f"run {run.name}: {run.scale}, seed {run.seed}, {len(run.days)} days")
    if line := probe_line(run.name, probe_of(run.folder)):
        print(f"warning: {line}")
    for model in models:
        mine = [q for q in haystack.questions_for(plan.questions, model.about) if q["oracle"] is not None]
        folder = run.folder / "results" / "decider" / model.slug
        todo = [q for q in mine if not (folder / f"{q['qid']}.json").exists()]
        total += len(todo)
        est = decider_minutes(plan, model, todo, run.estimate)
        if est is not None:
            minutes = (minutes or 0.0) + est
        print(f"  {model.name:26} {len(todo)} oracle asks to run of {len(mine)}"
              + (f", {minutes_text(est)}" if est is not None else ""))
    for line in skipped:
        print(f"  skipped {line}")
    print(f"{total} asks to run, "
          + (f"{minutes_text(minutes)} in all (each range's windows at the model's seconds a window from its About; loading a "
             "model is extra)" if minutes is not None else "each at most about a minute"))


def argv_lines(argv: list[str]) -> list[str]:
    """A command line as lines: each option with its values, quoted as a shell would need."""
    lines: list[str] = []
    for arg in argv:
        text = subprocess.list2cmdline([arg])
        if not lines or (arg.startswith("-") and len(arg) > 1):
            lines.append(text)
        else:
            lines[-1] += " " + text
    return lines


def dry_run(run: Run, names: list[str], models: list[Model], controls: str | None = None, subset: str = "all",
            only: Sequence[str] = ()) -> None:
    """The first question of each agent and arm prepared in a folder of its own under `dry-run` (an exec folder may hold a run's
    record), its command line and environment printed, nothing run. Codex's catalog comes from `codex debug models`, which only
    reads."""
    questions = run.questions(run.plan(), subset, models, only)
    run.check_paths()
    for name in names:
        found = shutil.which(name)
        command = claude_run.claude_command() if name == "claude" and found else [found or name]
        for arm, model in (*((a, None) for a in run.arms), *(("mms", m) for m in models[:1])):
            mine = questions if model is None else arm_questions(questions, model, controls)
            if not mine:
                print(f"\n== {name} {arm_dir(arm, model)}: no question to ask it")
                continue
            q = mine[0]
            run_dir, work = run.folder / "dry-run" / name / arm_dir(arm, model) / q["qid"], run.slot(0)
            home = None if model is None else run.folder / "homes" / f"{name}-{model.slug}"
            data = run.view(q, arm, 0, home, build=False)
            notes = model.about.notes if model else None
            if name == "claude":
                argv = claude_run.prepare(run_dir, arm, work, home=home, data=data, notes=notes or None, command=command)
                child = claude_run.env_of(run.bin, arm, work)
                env = [f"drop {', '.join(claude_run.ENV_DROP)}", *(f"set {k}={v}" for k, v in claude_run.ENV_SET.items()),
                       *(f"set {k}={child[k]}" for k in ("PATH", "MMS_HOME", "CLAUDE_CODE_GIT_BASH_PATH")
                         if arm == "files" and k in child)]
            else:
                argv = codex_run.prepare(run_dir, codex_run.model(), codex_run.effort(), "default", home, command=command,
                                         arm=arm, work=work, notes=notes if notes is not None else codex_run.NOTES.name,
                                         bin_dir=run.bin)
                env = ["Codex reads its own login (~/.codex/auth.json is never read or copied)",
                       *(["its shell's PATH and MMS_HOME are the shell_environment_policy settings above"]
                         if arm == "files" else [])]
            text = run.prompt(q, arm, data)
            (run_dir / "prompt.txt").write_text(text, encoding="utf-8")
            print(f"\n== {name} {arm_dir(arm, model)}, question {q['qid']}\n   exec folder {run_dir}\n   working folder {work}")
            print("   command:")
            for line in argv_lines(argv):
                print("     " + line)
            print("   environment:" + (f" {run.bin} first on PATH, the data is read-only" if arm == "files" else
                                          " mms's tools only" if arm == "mms" else " no tools"))
            for line in env:
                print(f"     {line}")
            print(f"   prompt ({run_dir / 'prompt.txt'}): {text}")
    print("\nnothing was run")


def model_names(text: str | None) -> list[str]:
    if text is None:
        return [a.name for a in decider.known_abouts()]
    return [] if text.strip().lower() == "none" else [n.strip() for n in text.split(",") if n.strip()]


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench.py", description="The mms benchmark: with mms and without, per decision model.")
    sub = parser.add_subparsers(dest="command", required=True)
    build_ = sub.add_parser("build", help="write the week and load a snapshot home")
    build_.add_argument("--scale", choices=tuple(haystack.SCALES), default="standard")
    build_.add_argument("--seed", type=int, default=1)
    build_.add_argument("--out", type=Path, default=OUT, help="where runs go (default ~/mms-bench)")
    build_.add_argument("--temp", type=Path, default=None, help="where the data and working folders go (default: the temp folder)")
    build_.add_argument("--no-probe", action="store_true",
                        help="do not run the shortcut probe over the data (the run and its report say so)")
    for command, help_ in (("decider", "each model's yes/no over each needle (the GPU, alone)"),
                           ("agents", "the execs of Claude and Codex, with and without mms (the plans' quota)")):
        one = sub.add_parser(command, help=help_)
        one.add_argument("run", help="a run folder, or its name under ~/mms-bench")
        one.add_argument("--models", help="model names, comma separated (default all eight)"
                         + ("; none runs the files arm alone" if command == "agents" else ""))
        one.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="where the model folders are (MMS_MODELS)")
        if command == "agents":
            one.add_argument("--agents", default=",".join(AGENTS))
            one.add_argument("--jobs", type=int, default=JOBS, help=f"execs at once, at most {JOBS_MAX}")
            one.add_argument("--subset", choices=SUBSETS, default="all",
                             help="pilot: the media questions and 3 controls (text and records)")
            one.add_argument("--controls", default=CONTROLS_MODEL,
                             help=f"the one model whose mms arm is also asked the text and records questions (default "
                                  f"{CONTROLS_MODEL}); all asks every model's arm")
            one.add_argument("--only", default="", help="these qids of the subset only, comma separated (a smoke)")
            one.add_argument("--anyway", action="store_true", help="run a run whose shortcut probe failed (it is refused otherwise)")
            one.add_argument("--dry-run", action="store_true", help="write the first exec's files per agent and arm, run nothing")
    report = sub.add_parser("report", help="the paired comparison, pooled over runs; written in the first run's folder")
    report.add_argument("runs", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parser_of()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            run = build(args.out, args.scale, args.seed, temp=args.temp, probe=not args.no_probe)
            print(run.folder)
            outcome = probe_of(run.folder)
            if outcome["status"] == "failed":
                for failure in outcome["failures"]:
                    print(f"probe: {failure}", file=sys.stderr)
                print(f"error: the shortcut probe failed ({len(outcome['failures'])}): the run is kept in {run.folder} (see "
                      "shortcuts.md); change the haystack and build another", file=sys.stderr)
                return 1
            if outcome["status"] == "skipped":
                print("the shortcut probe was skipped (--no-probe): the report will say so")
        elif args.command == "report":
            runs = [resolve_run(r) for r in args.runs]
            figures = numbers(runs)
            text = markdown(figures)
            (runs[0].folder / "report.md").write_text(text, encoding="utf-8")
            (runs[0].folder / "report.json").write_text(json.dumps(figures, indent=1), encoding="utf-8")
            print(text)
            print(f"written to {runs[0].folder / 'report.md'} and report.json")
        else:
            if args.command == "agents":
                controls = None if args.controls.strip().lower() == "all" else args.controls.strip()
                names = [n.strip() for n in args.agents.split(",") if n.strip()]
                if bad := [n for n in names if n not in AGENTS]:
                    parser.error(f"--agents is some of {', '.join(AGENTS)}, not {', '.join(bad)}")
                if not 1 <= args.jobs <= JOBS_MAX:
                    parser.error(f"--jobs is 1 to {JOBS_MAX}: more at once have crashed the GPU driver")
                only = [q.strip() for q in args.only.split(",") if q.strip()]
            run = resolve_run(args.run)
            models, skipped = resolve_models(model_names(args.models), args.models_dir)
            if args.command == "decider":
                show_decider(run, models, skipped)
                run_decider(run, models)
            else:
                if args.dry_run:
                    for line in skipped:
                        print(f"skipped {line}")
                    dry_run(run, names, models, controls, args.subset, only)
                elif (outcome := probe_of(run.folder)) and outcome["status"] == "failed" and not args.anyway:
                    print(f"error: {run.name} failed the shortcut probe ({len(outcome['failures'])} failures, in its shortcuts.md): "
                          "a shortcut may find its needles, so its numbers would not say what an ask is worth. Build another "
                          "run, or pass --anyway", file=sys.stderr)
                    return 1
                else:
                    show_agents(run, names, models, skipped, args.jobs, controls, args.subset, only)
                    if "claude" in names:
                        claude_run.check_login()
                    real = real_agents()
                    run_agents(run, {n: real[n] for n in names}, models, args.jobs, controls=controls, subset=args.subset,
                               only=only)
    except KeyboardInterrupt:
        print("interrupted: run the same command to continue", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, AssertionError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
