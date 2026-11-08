"""EgoLifeQA through the mms bench (TESTBED.md, Benchmark, EgoLifeQA): the public benchmark of the first-person days of EgoLife
(Jake's glasses, 30 s clips, and the transcript of what was said around him), asked of Claude Code and Codex three ways: without
the recordings (`blind`, the floor), with the files and their own tools (`files`), and with mms (`mms`, a decision model's reads).
It reuses `benchmarks/bench.py`'s exec loop, homes, agents and statistics; this file is the dataset behind them.

    python benchmarks/egolife.py build   --source DIR --out DIR --temp DIR [--seed N]   # DIR: the download, <data>/egolife
    python benchmarks/egolife.py decider RUN [--models NAME,...] [--models-dir DIR]     # the GPU, alone
    python benchmarks/egolife.py agents  RUN [--agents claude,codex] [--models NAME,...] [--jobs N] [--subset all|pilot]
                                        [--only QID,...] [--timeout S] [--dry-run]
    python benchmarks/egolife.py report  RUN

The download is `A1_JAKE/DAYn/DAYn_A1_JAKE_HHMMSSFF.mp4`, `EgoLifeQA/EgoLifeQA_A1_JAKE.json` and
`EgoLifeCap/Transcript/A1_JAKE/DAYn/*.srt`. Third-party data: it is read here and never run. The run folder and the temp folder
must be on the download's volume (the clips are hardlinked, not copied) and not on a small disk: mms's snapshot holds a second
copy of the clips (about 42 GB for 3 days), so `--out` and `--temp` have no default.

Which questions: for each of four ages (the query time less the latest target: under 2 h, 2-6 h, 6-24 h, over 24 h), 10 of the
questions that need no sound, chosen by the rarest categories first, and 10 that need sound, asked from the transcript alone
(`select`). Every question is asked as of its horizon H, the end of the last whole clip that ended by the query: no arm may
read a clip or a transcript line past H (the mms arm's index is cut to H before each exec, the files arm gets a folder of the
clips and lines up to H, `set_horizon`, `files_view`). What is lost is the seconds between H and the question: under 30, except
after a gap in the recording.

Times: a `plain` time is `day * 86400 + seconds into the day`, with no zone, which is what the data names; the plan turns them into
local epochs with `local`. So the questions chosen for a seed do not depend on the day the run is built on.

The run folder is `<out>/%Y%m%d-%H%M-egolife-s<seed>`: run.json, clips.json (the clip table), data/ (the clips as hardlinks under
neutral names, and the transcript as JSON lines), homes/, results/, execs/, report.md. The oracles of the decider tier are
`benchmarks/egolife_oracles.json`, written by hand by `ORACLE_RULES`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import secrets
import sqlite3
import stat
import subprocess
import sys
import time
import types
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import bench
import claude_run
import codex_run
import grade
import haystack

from mms import config, media, reader
from mms.scripts import ask as ask_script

VERSION = 1  # bump when the plan changes what it makes
PERSON, DAYS = "A1_JAKE", 3
DAY_S = 86400
QA_FILE, CLIP_FOLDER, SRT_FOLDER = f"EgoLifeQA/EgoLifeQA_{PERSON}.json", PERSON, f"EgoLifeCap/Transcript/{PERSON}"
AGES = (("0-2h", 7200), ("2-6h", 21600), ("6-24h", DAY_S), ("24h+", math.inf))  # the bins of a question's age, with their ends
AGE_NAMES = tuple(name for name, _ in AGES)
PER_AGE, SPEECH, PILOT_PER_AGE, PILOT_SPEECH = 10, 10, 2, 2  # questions per age bin and for speech; the pilot's share of each
TOL_S, PAD_S = 30.0, 300.0  # a sighting within this of a target counts; an oracle's range is a target +- this, cut at H
NEAR_S = 30.0  # a speech question is asked when a transcript line is within this of a target
VIDEO, TEXT = "glasses", "transcript"  # the mms sources
TEXT_ASK = "Does the text talk about "  # how a speech question's oracle starts (rule 6): its words are no borrowing from an option
CATEGORIES = ("EventRecall", "EntityLog", "RelationMap", "TaskMaster", "HabitInsight")
PEOPLE = ("Alice", "Choiszt", "Jake", "Katrina", "Lucia", "Nicous", "Shure", "Tasha", "Violet")  # who the transcript names
ORACLES = Path(__file__).with_name("egolife_oracles.json")
ORACLE_VERSION = 1
TIMEOUT_S = 900.0  # an exec's time; raised for both arms only on the pilot's evidence
AUDIT_S = 0.001  # a time in a tool result is past H when it is more than this after it (a result rounds to the millisecond)
CLIP_NAME = re.compile(rf"DAY(\d+)_{PERSON}_(\d{{8}})\.mp4")
SRT_NAME = re.compile(rf"{PERSON}_DAY(\d+)_(\d{{8}})\.srt")
SRT_TIME = re.compile(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})")
SPEAKER = re.compile(r"\s*([^:：]{1,20})[:：]\s*(.*)", re.S)
ORACLE_RULES = """\
1. Write it from the item's `question` and `keywords` only. Read `reason` to learn what the evidence looks like (an action, an
   object), never who, where or which.
2. Never use the words of the right option, or of any option that alone would settle the answer.
3. Ask about one thing that one window can show, in the present tense: "Is someone using a screwdriver?", "Is a black marker lying
   on a table?".
4. For `glasses`, name no person: d1 cannot tell people by name, and the agent tells who from a look and the transcript.
5. No time words: the range carries the time.
6. A `need_audio` item uses `transcript`: "Does the text talk about X?", with X the question's subject, not its answer.
7. Never the trigger: it is at the query time, after H."""


class HorizonLeak(RuntimeError):
    """A tool result of the mms arm held a recording from after the question's horizon: the harness is wrong, not the answer."""


# -- the download ---------------------------------------------------------------------------------------------------------------

def plain(day: int, seconds: float) -> float:
    """A plain time: `seconds` into day number `day` (1 is the first), as seconds from the start of day 0."""
    return day * DAY_S + seconds


def clock_of(text: str) -> float:
    """Seconds into the day of EgoLife's `HHMMSSFF`, where FF is hundredths of a second: 11210217 is 11:21:02.17."""
    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(f"a time is HHMMSSFF (8 digits), not {text!r}")
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:6]) + int(text[6:]) / 100


def day_of(text: str) -> int:
    """The day number of `DAY1` (the data also writes `Day1`)."""
    found = re.fullmatch(r"day(\d+)", text.strip().lower())
    if found is None:
        raise ValueError(f"a day is DAY and a number, not {text!r}")
    return int(found.group(1))


@dataclass(frozen=True)
class Clip:
    rel: str  # in the download: A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4
    day: int
    t0: float  # plain, from the name
    length: float  # seconds, as mms.media.duration gives it (the end mms stores)

    @property
    def t1(self) -> float:
        return self.t0 + self.length


def clip_table(source: Path, days: int = DAYS) -> list[Clip]:
    """The clips of days 1..`days` in time order, with their durations (read from the packets, as `remember` does)."""
    names = [(day, name) for day in range(1, days + 1) for name in sorted(os.listdir(Path(source) / CLIP_FOLDER / f"DAY{day}"))]
    paths = [Path(source) / CLIP_FOLDER / f"DAY{day}" / name for day, name in names]
    with ThreadPoolExecutor(3) as pool:
        lengths = list(pool.map(media.duration, paths))
    out = []
    for (day, name), path, length in zip(names, paths, lengths):
        found = CLIP_NAME.fullmatch(name)
        if found is None or int(found.group(1)) != day:
            raise ValueError(f"{path} is not a clip of DAY{day}: DAY{day}_{PERSON}_HHMMSSFF.mp4")
        if length is None:
            raise ValueError(f"{path} is not media this machine can read")
        out.append(Clip(path.relative_to(source).as_posix(), day, plain(day, clock_of(found.group(2))), length))
    out.sort(key=lambda c: c.t0)
    return out


def read_items(source: Path) -> list[dict]:
    """The questions of the download, one plain dict each: `id`, `category`, `audio` (needs sound), `need_name`, `last`, `query` (a
    plain time), `targets` (plain times, in order: the evidence), `question`, `options` (A to D), `answer`, `keywords`, `reason`.
    `Day1` is DAY1; a target is `time` or `time_list`. `targets` is empty for the few questions whose time holds a range or
    several moments in one string (`11153417DAY1_11181201`): they have no one moment to look at."""
    out, seen = [], set()
    for raw in json.loads((Path(source) / QA_FILE).read_text(encoding="utf-8")):
        id_ = str(raw["ID"])
        if id_ in seen:
            raise ValueError(f"the question ID {id_} is twice in {QA_FILE}")
        seen.add(id_)
        if raw["type"] not in CATEGORIES:
            raise ValueError(f"question {id_} is of the type {raw['type']!r}, not one of {', '.join(CATEGORIES)}")
        if raw["answer"] not in ("A", "B", "C", "D"):
            raise ValueError(f"question {id_}'s answer is {raw['answer']!r}, not A, B, C or D")
        target = raw["target_time"]
        times, day = target["time_list"] if "time_list" in target else [target["time"]], day_of(target["date"])
        out.append({"id": id_, "category": raw["type"], "audio": bool(raw["need_audio"]), "need_name": bool(raw["need_name"]),
                    "last": bool(raw["last_time"]),
                    "query": plain(day_of(raw["query_time"]["date"]), clock_of(raw["query_time"]["time"])),
                    "targets": sorted(plain(day, clock_of(t)) for t in times) if all(re.fullmatch(r"\d{8}", t) for t in times) else [],
                    "question": raw["question"], "options": {k: raw[f"choice_{k.lower()}"] for k in "ABCD"},
                    "answer": raw["answer"], "keywords": raw["keywords"], "reason": raw["reason"]})
    return out


def spoken(row: str) -> tuple[str, str]:
    """A transcript line as `(speaker, words)`: `Jake: Okay.` is `("Jake", "Okay.")`; a line with no speaker has the speaker ""."""
    found = SPEAKER.fullmatch(row)
    return (found.group(1).strip(), found.group(2).strip()) if found else ("", row.strip())


def read_transcripts(source: Path, days: int = DAYS) -> list[dict]:
    """The spoken lines of days 1..`days`, by their end: `start`, `end` (plain), `speaker`, `zh` (the first line of an entry,
    as written) and `en` (its second line, the English: "" for an entry of one line). A SRT file is an hour, its times offsets from
    the hour in its name; a line with no `Speaker:` has the speaker ""."""
    out = []
    for day in range(1, days + 1):
        folder = Path(source) / SRT_FOLDER / f"DAY{day}"
        for name in sorted(os.listdir(folder)):
            found = SRT_NAME.fullmatch(name)
            if found is None or int(found.group(1)) != day:
                raise ValueError(f"{folder / name} is not a transcript of DAY{day}: {PERSON}_DAY{day}_HHMMSSFF.srt")
            base = plain(day, clock_of(found.group(2)))
            text = (folder / name).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
            for block in re.split(r"\n\s*\n", text.strip()):
                rows = block.split("\n")
                when = SRT_TIME.fullmatch(rows[1].strip()) if len(rows) in (3, 4) and rows[0].strip().isdigit() else None
                if when is None:
                    raise ValueError(f"{folder / name}: an entry that is not a number, a time range and one or two lines: {block[:80]!r}")
                h0, m0, s0, ms0, h1, m1, s1, ms1 = (int(v) for v in when.groups())
                speaker, zh = spoken(rows[2])
                en = spoken(rows[3])[1] if len(rows) == 4 else ""
                out.append({"start": base + h0 * 3600 + m0 * 60 + s0 + ms0 / 1000, "end": base + h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
                            "speaker": speaker, "zh": zh, "en": en})
    out.sort(key=lambda line: (line["end"], line["start"]))
    return out


# -- which questions --------------------------------------------------------------------------------------------------------------

def horizon(clips: Sequence[Clip], t: float) -> Clip | None:
    """The last whole clip at or before plain time `t` (the one whose end is the latest that is not after it): the question's
    horizon H is its end. None when no clip had ended by `t`."""
    ended = [c for c in clips if c.t1 <= t]
    return max(ended, key=lambda c: c.t1) if ended else None


def age_of(item: dict) -> str:
    """The age bin of a question: its query time less its latest target."""
    age = item["query"] - max(item["targets"])
    return next(name for name, end in AGES if age < end)


def pools(items: Sequence[dict], clips: Sequence[Clip], lines: Sequence[dict], days: int = DAYS) -> tuple[list[dict], list[dict], Counter]:
    """The questions the bench can ask, `(video, speech, dropped)`. A question is asked when its query is on days 1..`days`, a clip
    had ended by then, and each of its targets is inside a clip that ended by the horizon; a speech question (needs sound) also
    when a transcript line that ended by the horizon is within NEAR_S of a target. `dropped` counts the others by why."""
    video, speech, dropped = [], [], Counter()
    for item in sorted(items, key=lambda i: int(i["id"])):
        if item["query"] >= plain(days + 1, 0):
            dropped["its query is after the last day"] += 1
            continue
        if not item["targets"]:
            dropped["its time is a range or a list in one string"] += 1
            continue
        top = horizon(clips, item["query"])
        if top is None:
            dropped["no clip had ended by its query"] += 1
            continue
        held = [c for c in clips if c.t1 <= top.t1]
        if not all(any(c.t0 <= t <= c.t1 for c in held) for t in item["targets"]):
            dropped["a target is outside the clips (a gap) or after the horizon"] += 1
            continue
        if item["audio"] and not any(line["start"] <= t + NEAR_S and line["end"] >= t - NEAR_S and line["end"] <= top.t1
                                     for t in item["targets"] for line in lines):
            dropped["speech: no transcript line near a target"] += 1
            continue
        (speech if item["audio"] else video).append(item)
    return video, speech, dropped


def rotate(queues: Sequence[Sequence], n: int) -> list:
    """`n` items taken one from each queue in turn, the first queue first; a queue that runs out is skipped."""
    queues, out = [list(q) for q in queues], []
    while len(out) < n and any(queues):
        for queue in queues:
            if queue and len(out) < n:
                out.append(queue.pop(0))
    return out


def by_category(items: Sequence[dict]) -> list[list[dict]]:
    """The items split by category, the rarest category first (ties by name), each in the order given."""
    groups = {c: [i for i in items if i["category"] == c] for c in sorted({i["category"] for i in items})}
    return [groups[c] for c in sorted(groups, key=lambda c: (len(groups[c]), c))]


def binned(items: Sequence[dict]) -> dict[str, list[dict]]:
    return {name: [i for i in items if age_of(i) == name] for name in AGE_NAMES}


def select(items: Sequence[dict], clips: Sequence[Clip], lines: Sequence[dict], seed: int, per_age: int = PER_AGE,
           speech: int = SPEECH, days: int = DAYS) -> list[dict]:
    """The questions of a seed: `per_age` video questions for each age bin, then `speech` speech ones, each item with its `age`
    and `pilot` (the first PILOT_PER_AGE of its bin, or PILOT_SPEECH of the speech ones). A bin is shuffled by the seed and its
    questions taken from the categories in turn, the rarest first; the speech ones from the bins in turn, the rarest first, and
    within a bin from the categories in turn. A pool too short for its share is a `ValueError` naming it."""
    video, heard, _ = pools(items, clips, lines, days)
    chosen = []
    for name, of in binned(video).items():
        if len(of) < per_age:
            raise ValueError(f"the {name} age bin holds {len(of)} questions the bench can ask, fewer than {per_age}")
        random.Random(f"{seed}-{name}").shuffle(of)
        chosen += [{**i, "age": name, "pilot": rank < PILOT_PER_AGE} for rank, i in enumerate(rotate(by_category(of), per_age))]
    queues = {}
    for name, of in binned(heard).items():
        random.Random(f"{seed}-speech-{name}").shuffle(of)
        queues[name] = rotate(by_category(of), len(of))
    if sum(len(q) for q in queues.values()) < speech:
        raise ValueError(f"the speech pool holds {sum(len(q) for q in queues.values())} questions, fewer than {speech}")
    order = sorted(queues, key=lambda n: (len(queues[n]), AGE_NAMES.index(n)))
    chosen += [{**i, "age": age_of(i), "pilot": rank < PILOT_SPEECH} for rank, i in enumerate(rotate([queues[n] for n in order], speech))]
    return chosen


# -- the plan ---------------------------------------------------------------------------------------------------------------------

def day_date(days: Sequence[date], day: int) -> date:
    """The date of day number `day` (1 is the first of `days`)."""
    if not 1 <= day <= len(days):
        raise ValueError(f"day {day} is not one of the run's {len(days)} days")
    return days[day - 1]


def local(days: Sequence[date], t: float) -> float:
    """The local epoch of plain time `t`, day number `d` being `days[d - 1]` (hundredths kept: `datetime` takes microseconds)."""
    cs = round(t * 100)
    day, cs = divmod(cs, DAY_S * 100)
    seconds, hundredths = divmod(cs, 100)
    d = day_date(days, day)
    return datetime(d.year, d.month, d.day, seconds // 3600, seconds // 60 % 60, seconds % 60, hundredths * 10000).timestamp()


def iso_of(days: Sequence[date], t: float) -> str:
    """Plain time `t` as a local ISO time to the millisecond, with no zone (mms reads it as local)."""
    day, ms = divmod(round(t * 1000), DAY_S * 1000)
    return (datetime.combine(day_date(days, day), datetime.min.time()) + timedelta(milliseconds=ms)).isoformat(timespec="milliseconds")


def clip_path(days: Sequence[date], c: Clip) -> str:
    """Where the run's data holds a clip, and the name `remember` gives it: neutral (no person, dataset or day number)."""
    s = round(c.t0 * 100) % (DAY_S * 100) // 100
    return f"{VIDEO}/{day_date(days, c.day).isoformat()}/{s // 3600:02d}.{s // 60 % 60:02d}.{s % 60:02d}.mp4"


def clip_end(days: Sequence[date], c: Clip) -> float:
    """A clip's end as mms stores it: the local epoch it is remembered at, plus its length (the same sum, so H compares exactly)."""
    return local(days, c.t0) + c.length


def log_lines(days: Sequence[date], lines: Sequence[dict]) -> dict[str, tuple[dict, ...]]:
    """The transcript as the run holds it: a date to its lines, each `start`, `end` (local ISO), `speaker`, `en` and `zh`."""
    out: dict[str, list[dict]] = {d.isoformat(): [] for d in days}
    for line in lines:
        day = int(line["end"] // DAY_S)
        out[day_date(days, day).isoformat()].append({"start": iso_of(days, line["start"]), "end": iso_of(days, line["end"]),
                                               "speaker": line["speaker"], "en": line["en"], "zh": line["zh"]})
    return {d: tuple(v) for d, v in out.items()}


def load_oracles(path: Path | None = None) -> dict[str, dict]:
    """The oracles file: question ID to `{"source", "question"}`."""
    data = json.loads(Path(path or ORACLES).read_text(encoding="utf-8"))
    if data.get("version") != ORACLE_VERSION:
        raise ValueError(f"{path or ORACLES} is version {data.get('version')}, this is {ORACLE_VERSION}")
    return data["items"]


def check_oracle(item: dict, oracle: dict) -> None:
    """The rules of ORACLE_RULES that a machine can read, applied to the oracle of `item`: its source is the question's, it asks
    one thing, with no time word, no 'and' or 'or', no option's words (a word of 4 letters or more of an option is allowed only
    when the question has it too) and (for the glasses) no person. A `ValueError` names the ID."""
    id_, text = item["id"], oracle.get("question", "")
    wanted = TEXT if item["audio"] else VIDEO
    if oracle.get("source") != wanted:
        raise ValueError(f"the oracle of question {id_} has the source {oracle.get('source')!r}, not {wanted!r}")
    if not isinstance(text, str) or not text.endswith("?"):
        raise ValueError(f"the oracle of question {id_} is not a question: {text!r}")
    if ask_script.TIME_WORDS.search(text):
        raise ValueError(f"the oracle of question {id_} has a time word (the range carries the time): {text!r}")
    if ask_script.JOINED.search(text):
        raise ValueError(f"the oracle of question {id_} joins two things with 'and' or 'or': {text!r}")
    if not item["audio"] and (named := [p for p in PEOPLE if re.search(rf"\b{p}\b", text, re.I)]):
        raise ValueError(f"the oracle of question {id_} names {named[0]} (the glasses cannot tell people by name): {text!r}")
    if leaked := [o for o in item["options"].values() if o.strip() and re.search(rf"\b{re.escape(o.strip())}\b", text, re.I)]:
        raise ValueError(f"the oracle of question {id_} holds an option's words ({leaked[0]!r}): {text!r}")
    asked = set(re.findall(r"[a-z]+", item["question"].lower()))
    if borrowed := sorted({w for o in item["options"].values() for w in re.findall(r"[a-z]{4,}", o.lower())}
                          & (set(re.findall(r"[a-z]+", text.removeprefix(TEXT_ASK).lower())) - asked)):
        raise ValueError(f"the oracle of question {id_} uses {borrowed[0]!r}, a word of an option that the question does not use: "
                         f"{text!r}")


def plan(source: Path, seed: int, days: Sequence[date], clips: Sequence[Clip], oracles: dict[str, dict], per_age: int = PER_AGE,
         speech: int = SPEECH) -> haystack.Plan:
    """The run's plan, a pure function of the download, the seed, the days, the clip table and the oracles: one `File` per clip
    (kind `clip`, spec `{"src", "length"}`), the transcript as logs (a date to its lines), and a question for each chosen item
    (the keys `bench.record` and `run_arm` read, and `options`, `age`, `oracle`, `oracle_source`, `pilot`). A chosen question with
    no oracle is refused by its ID."""
    days = tuple(days)
    items, lines = read_items(source), read_transcripts(source, len(days))
    chosen = select(items, clips, lines, seed, per_age, speech, len(days))
    if missing := [i["id"] for i in chosen if i["id"] not in oracles]:
        raise ValueError(f"no oracle for question {', '.join(missing)}: write it in {ORACLES.name} by ORACLE_RULES")
    for item in chosen:
        check_oracle(item, oracles[item["id"]])
    paths = [clip_path(days, c) for c in clips]
    if dup := [p for p, n in Counter(paths).items() if n > 1]:
        raise ValueError(f"two clips would be named {dup[0]}")
    files = tuple(haystack.File(p, VIDEO, local(days, c.t0), p.split("/", 1)[1].replace("/", " "), "clip",
                                {"src": c.rel, "length": c.length}) for p, c in zip(paths, clips))
    questions = []
    for item in chosen:
        top = horizon(clips, item["query"])
        oracle = oracles[item["id"]]
        questions.append({
            "qid": f"ego-{int(item['id']):03d}", "id": item["id"], "type": "records" if item["audio"] else "video",
            "template": item["category"], "level": item["age"], "last": item["last"], "absent": False, "need_name": item["need_name"],
            "source": TEXT if item["audio"] else VIDEO, "day": int(item["query"] // DAY_S), "query": local(days, item["query"]),
            "since": local(days, clips[0].t0), "until": clip_end(days, top), "question": item["question"], "options": item["options"],
            "grade": {"kind": "choice", "truth": item["answer"], "ignore": []},
            "truth": {"moments": [[local(days, t)] * 2 for t in item["targets"]], "others": []},
            "oracle": oracle["question"], "oracle_source": oracle["source"], "pilot": item["pilot"]})
    return haystack.Plan(seed, "egolife", days, files, log_lines(days, lines), tuple(questions))


# -- the horizon ------------------------------------------------------------------------------------------------------------------

def set_horizon(home: Path, h: float) -> dict:
    """Cut the home's index to the horizon `h` (a local epoch), in one transaction on the live index (the daemon reads it at each
    call, as `bench.scrub_home` relies on): a clip that ends after `h` is `gone`, one that ends by it is `ok`, and the transcript
    records after `h` move to a side table (`bench_hidden_records`) and come back, with their ids, when `h` moves later. Returns how
    many clips and records are shown and hidden."""
    with closing(sqlite3.connect(home / "index.sqlite", timeout=30, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS bench_hidden_records AS SELECT * FROM records WHERE 0")
        db.execute("UPDATE items SET state = CASE WHEN t1 <= ? THEN 'ok' ELSE 'gone' END WHERE source = ? AND state IN ('ok', 'gone')",
                   (h, VIDEO))
        db.execute("INSERT INTO bench_hidden_records SELECT * FROM records WHERE source = ? AND t > ?", (TEXT, h))
        db.execute("DELETE FROM records WHERE source = ? AND t > ?", (TEXT, h))  # records_ad takes them out of the search index
        db.execute("INSERT INTO records SELECT * FROM bench_hidden_records WHERE t <= ?", (h,))  # records_ai puts them back
        db.execute("DELETE FROM bench_hidden_records WHERE t <= ?", (h,))
        shown = db.execute("SELECT COUNT(*) FROM items WHERE source = ? AND state = 'ok'", (VIDEO,)).fetchone()[0]
        hidden = db.execute("SELECT COUNT(*) FROM items WHERE source = ? AND state = 'gone'", (VIDEO,)).fetchone()[0]
        records = db.execute("SELECT COUNT(*) FROM records WHERE source = ?", (TEXT,)).fetchone()[0]
        hidden_records = db.execute("SELECT COUNT(*) FROM bench_hidden_records").fetchone()[0]
        db.execute("COMMIT")
    return {"clips": shown, "clips_hidden": hidden, "records": records, "records_hidden": hidden_records}


def files_view(run: "EgoRun", k: int, h: float) -> Path:
    """Slot `k`'s folder of the recordings up to the horizon `h`, made new: `glasses/<date>/HH.MM.SS.mp4` for each clip that ended
    by `h` (hardlinks to the run's data, so the agent cannot write a copy) and `transcript/<date>.jsonl` with the lines that ended
    by `h` (read-only), for a date that has any. Nothing later is there to read, not even the name of a later day's file."""
    view = run.view_dir(k)
    bench.empty(view)
    for c in run.clips():
        if clip_end(run.days, c) <= h:
            path = clip_path(run.days, c)
            (view / path).parent.mkdir(parents=True, exist_ok=True)
            os.link(run.data / path, view / path)
    for file in sorted((run.data / TEXT).glob("*.jsonl")):
        kept = [line for line in file.read_text(encoding="utf-8").splitlines()
                if datetime.fromisoformat(json.loads(line)["end"]).timestamp() <= h]
        if kept:
            target = view / TEXT / file.name
            target.parent.mkdir(exist_ok=True)
            target.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            os.chmod(target, stat.S_IREAD)
    return view


def release_views(run: "EgoRun") -> None:
    """The slots' folders of recordings removed (they hold links only). One that cannot be (a child of a killed exec still holds
    a file) is told and left, so it does not hide the error that ended the run."""
    for k in range(bench.JOBS_MAX):
        if run.view_dir(k).exists():
            try:
                bench.remove_tree(run.view_dir(k))
            except OSError as error:
                print(f"could not remove {run.view_dir(k)}: {error}", flush=True)


# -- the run ----------------------------------------------------------------------------------------------------------------------

def oracle_ranges(q: dict) -> list[tuple[float, float]]:
    """What the decider tier asks over: each target +- PAD_S, cut at the horizon, overlapping ranges merged."""
    spans = sorted((m - PAD_S, min(m + PAD_S, q["until"])) for m, _ in q["truth"]["moments"])
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def seconds_in(files: Sequence[haystack.File], a: float, b: float) -> int:
    """The most windows the reader gives over [a, b) in glasses clips: a frame a second of each clip's overlap with it (a still
    clip joins them, which a moving camera hardly does)."""
    return sum(math.ceil(max(0.0, min(f.t + f.spec["length"], b) - max(f.t, a))) for f in files)


def tool_times(tool: dict, home: Path | None) -> list[float]:
    """The times of recordings in an mms tool's results (every JSON block of them: a `look` of several moments has one each): an
    ask's sightings and candidates, a find's rows (not its labels, which the agent made itself and mms dates by its own clock), the
    end of each item a look returned (from the home's index), and for a sheet the end of each item its tiles were cut from. A
    sheet's range and tile times are the times the agent asked for, as an ask's `until` is, so they are not counted."""
    name, times, items = tool.get("tool"), [], []
    when = lambda text: config.parse_when(text) if text else None  # noqa: E731
    for got in bench.result_jsons(tool.get("result")):
        if name == "ask":
            times += [when(s.get(k)) for s in got.get("sightings") or [] for k in ("t", "t0", "t1")]
            times += [when(c.get(k)) for c in got.get("candidates") or [] for k in ("t", "t0", "t1")]
        elif name == "find":
            times += [when(r.get(k)) for r in got.get("results") or [] if r.get("type") != "label" for k in ("t0", "t1")]
        elif name == "look":
            for m in [m for m in got.get("moments") or [got] if isinstance(m, dict)]:
                items += [m["item"]] if isinstance(m.get("item"), int) else []
                for tile in [x for x in m.get("tiles") or [] if isinstance(x, dict)]:
                    items += [tile["look"]["item"]] if isinstance(tile.get("look"), dict) else []
    if items and home is not None:
        with closing(sqlite3.connect(f"file:{home / 'index.sqlite'}?mode=ro", uri=True, timeout=30)) as db:
            times += [row[0] for i in items for row in db.execute("SELECT t1 FROM items WHERE id = ?", (i,))]
    return [t for t in times if t is not None]


def accepted_asks(tools: Sequence[dict]) -> list[dict]:
    """The arguments of the asks mms took (their result holds a `job`): an ask with `question` or `questions`, not a call that
    collects or cancels a job, and not one mms refused."""
    return [t["arguments"] for t in tools if t.get("tool") == "ask" and isinstance(t.get("arguments"), dict)
            and ("question" in t["arguments"] or "questions" in t["arguments"])
            and any("job" in got for got in bench.result_jsons(t.get("result")))]


def audit(tools: Sequence[dict], h: float, home: Path | None) -> dict:
    """What an mms exec's calls held past the horizon `h`: `after_horizon`, the distinct recording times after it in the results
    (any is a leak of the harness), and of the asks mms took (`asks`) those with an `until` past it (`asks_past_horizon`: the
    agent's confusion about mms's clock, which reads days later than the question, and harmless)."""
    late = {t for tool in tools for t in tool_times(tool, home) if t > h + AUDIT_S}
    starts = accepted_asks(tools)
    past = [a for a in starts if a.get("until") is None or config.parse_when(a["until"]) > h + AUDIT_S]
    return {"after_horizon": len(late), "asks": len(starts), "asks_past_horizon": len(past)}


SEEN_TIME = re.compile(r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}")  # a transcript line's start or end, as the files arm's lines write it
SEEN_CLIP = re.compile(r"glasses[/\\]+\d{4}-\d\d-\d\d[/\\]+\d\d\.\d\d\.\d\d\.mp4")  # a clip's path in the files arm (`\\` escaped too)
MISSING = re.compile(r"no such file|does not exist|cannot find|not found|filenotfound", re.I)  # a line that reports a missing file


def command_outputs(events: Sequence[dict]) -> list[str]:
    """What the files arm's commands and tools printed, from the exec's full events (tools.json cuts a shell's output): a Codex
    command's output, and the content of each tool result of Claude."""
    out = []
    for e in events:
        item = e.get("item") or {}
        if item.get("type") == "command_execution":
            out.append(str(item.get("aggregated_output") or ""))
        elif e.get("type") == "user":
            for block in claude_run.blocks_of(e):
                content = block.get("content") if block.get("type") == "tool_result" else None
                out.append(content if isinstance(content, str) else
                           "\n".join(str(x.get("text") or "") for x in content if isinstance(x, dict)) if isinstance(content, list) else "")
    return out


def seen_past_horizon(texts: Sequence[str], run: "EgoRun", h: float) -> list[str]:
    """The recordings past the horizon `h` that `texts` show: a start or an end of a transcript line that is after it (a time of
    the transcript, not any time: a file's mtime or a computed one is the agent's own), or the name of a clip that ended after
    it. The files arm's view holds neither, so an output that shows one read it somewhere else, without printing a path that
    `deny` would have stopped. A line that reports a missing file shows a clip name the agent made up (a guess at the clip of
    "now"), and its names are not counted."""
    ends = {clip_path(run.days, c): clip_end(run.days, c) for c in run.clips()}
    late = set()
    for file in (run.data / TEXT).glob("*.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            late |= {t for t in (row["start"], row["end"]) if datetime.fromisoformat(t).timestamp() > h}
    seen = set()
    for text in texts:
        for line in text.splitlines():
            seen.update(t for t in SEEN_TIME.findall(line) if t in late)
            if not MISSING.search(line):
                names = (re.sub(r"[/\\]+", "/", found) for found in SEEN_CLIP.findall(line))
                seen.update(name for name in names if ends.get(name, 0.0) > h)
    return sorted(seen)


@dataclass(frozen=True)
class EgoRun(bench.Run):
    """A run of EgoLifeQA: the `bench.Run` seam with this dataset's plan, prompts, views, estimate and audit."""

    arms: ClassVar[tuple[str, ...]] = ("blind", "files")  # before the mms arm

    source: Path = Path()  # the download
    decode_s: float = 0.0  # seconds the reader decodes a glasses window in, measured at build
    per_age: int = PER_AGE
    speech: int = SPEECH

    @classmethod
    def open(cls, folder: Path) -> "EgoRun":
        """The run in `folder`; refused when it is not an egolife run, when VERSION has changed, or when the questions or files
        are not those its run.json recorded (the download or the oracles file changed since the build)."""
        folder = Path(folder)
        meta = bench.read_json(folder / "run.json")
        if (dataset := meta.get("dataset", "haystack")) != "egolife":
            raise ValueError(f"{folder} is a {dataset} run: use benchmarks/bench.py")
        if meta["generator"] != VERSION:
            raise ValueError(f"{folder} was made by egolife version {meta['generator']}, this is {VERSION}: its questions are "
                             "not the plan's now; build a new run")
        run = cls(folder, meta["seed"], meta["scale"], tuple(date.fromisoformat(d) for d in meta["days"]), Path(meta["data"]),
                  Path(meta["temp"]), meta["tag"], timeout_s=meta["timeout_s"], source=Path(meta["source"]),
                  decode_s=meta["decode_s"], per_age=meta["per_age"], speech=meta["speech"])
        made = run.plan()
        if meta["plan"] != bench.plan_hash(made):
            raise ValueError(f"{folder}'s questions are not those its run.json recorded: the oracles file or the download "
                             "changed, so its results would be graded against other questions; build a new run")
        if meta["files"] != bench.files_hash(made):
            raise ValueError(f"{folder}'s files are not those its run.json recorded: the download changed; build a new run")
        return run

    def clips(self) -> list[Clip]:
        return [Clip(**c) for c in bench.read_json(self.folder / "clips.json")]

    def plan(self) -> haystack.Plan:
        return plan(self.source, self.seed, self.days, self.clips(), load_oracles(), self.per_age, self.speech)

    def view_dir(self, k: int) -> Path:
        return self.temp / f"mmsb-{self.key}-d{k}"

    def prompt(self, q: dict, arm: str, data: Path | None, now: float | None = None) -> str:
        """What an agent is asked: the scenario's time (the question's own, so `now` is not used), how the recordings are reached
        (the only difference between the arms), the question, its options and how to answer."""
        t = datetime.fromtimestamp(q["query"])
        head = (f"It is now {haystack.date_text(t.date())} {t.year}, {t:%H:%M} (day {q['day']} of these recordings; day 1 was "
                f"{haystack.date_text(self.days[0])}). I am Jake.")
        if arm == "blind":
            how = "You cannot see my recordings."
        elif arm == "files":
            how = (f"The recordings of my days so far are in {data}: glasses/<date>/ holds the video of the camera on my glasses as "
                   "30 s clips (1408x1408, 20 fps), each named by the time it starts; transcript/<date>.jsonl holds what was said "
                   "around me, one JSON line per spoken line with its start, end, speaker and the words in English and Chinese. "
                   "Your working folder is yours to write in. ffmpeg is on PATH (there is no ffprobe).")
        elif arm == "mms":
            how = (f"mms has recorded these sources of mine from day 1 to now: {VIDEO} (the video of the camera on my glasses, 30 s "
                   f"clips at 20 fps) and {TEXT} (what was said around me: a record per spoken line, with its speaker and the "
                   "words in English and Chinese). mms's clock reads a later date than now: give since and until as dates and "
                   "times, not as ages or \"now\".")
        else:
            raise ValueError(f"arm is one of {', '.join(bench.ARMS)}, not {arm!r}")
        options = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
        end = "If what you can find does not settle it, give your best guess. End with one line ANSWER: <A, B, C or D>."
        return f"{head} {how}\n\n{q['question']}\n{options}\n\n{end}"

    def view(self, q: dict, arm: str, k: int, home: Path | None, build: bool = True) -> Path | None:
        """Cut what the arm can read to the question's horizon: slot `k`'s folder of recordings for the files arm, the mms home's
        index for the mms arm (one exec at a time: the horizon is the home's), nothing for the blind arm. Without `build` only
        the files arm's folder is named."""
        if arm == "files":
            return files_view(self, k, q["until"]) if build else self.view_dir(k)
        if arm == "mms" and build:
            set_horizon(home, q["until"])
        return None

    def ensure_data(self) -> None:
        bench.ensure_bin(self)  # the data is in the run's folder; the ffmpeg is in the temp folder

    def questions(self, plan: haystack.Plan, subset: str, models: Sequence[bench.Model], only: Sequence[str] = ()) -> list[dict]:
        """All the plan's questions, or the pilot's (`pilot` marks the first of each age bin and of the speech ones); with `only`,
        just those qids of them (a smoke)."""
        if subset not in bench.SUBSETS:
            raise ValueError(f"a subset is one of {', '.join(bench.SUBSETS)}, not {subset!r}")
        picked = [q for q in plan.questions if subset == "all" or q["pilot"]]
        if unknown := sorted(set(only) - {q["qid"] for q in picked}):
            raise ValueError(f"--only names questions that are not in the {subset} questions: {', '.join(unknown)}")
        return [q for q in picked if not only or q["qid"] in set(only)]

    def warm_params(self, plan: haystack.Plan) -> dict:
        t = plan.files[0].t  # the first clip: one window of it
        return {"question": "Is anything in view?", "sources": [VIDEO], "since": t, "until": t + 1, "limit": 1, "candidates": 0}

    def mms_jobs(self, model: bench.Model, jobs: int) -> int:
        return 1  # the horizon cuts the whole home, so its execs go one at a time

    def oracle(self, api, q: dict, model: bench.Model, ask_with: str = "oracle") -> dict:
        if ask_with != "oracle":
            raise ValueError(f"ask_with is oracle, not {ask_with!r}: the fake decider's words are the test's oracles")
        return oracle_record(api, q, model)

    def release(self) -> None:
        release_views(self)

    def estimate(self, plan: haystack.Plan, model: bench.Model, q: dict) -> float:
        """The GPU seconds of the cheapest ask that could find the evidence, the decider's: its ranges (a target +- PAD_S) at the
        model's cost a window. A glasses window costs what the reader's decode really takes here (`decode_s`), not mms's guess."""
        ranges = oracle_ranges(q)
        if q["source"] == VIDEO:
            files = [f for f in plan.files if f.kind == "clip"]
            each = bench.window_s(model.about) - ask_script.DECODE_S + self.decode_s
            return sum(seconds_in(files, a, b) for a, b in ranges) * each
        lines = [line for rows in plan.logs.values() for line in rows]
        chars = sum(len(json.dumps(line)) + 26 for line in lines
                    for a, b in ranges if a <= datetime.fromisoformat(line["end"]).timestamp() < b)
        return math.ceil(chars / reader.PROFILE["records_chars"]) * bench.window_s(model.about, "records")

    def needle(self, q: dict) -> str:
        return q["qid"]

    def extra(self, e: bench.Exec, out: dict) -> dict:
        """The exec's timeout; for an mms exec the horizon audit (`audit`), where a recording past the horizon in a result is a
        `HorizonLeak`, which ends the whole command (it propagates out of `run_arm` and `run_agents`) and leaves the question to
        run again; for a files exec the recordings past the horizon its outputs show (`seen_past_horizon`), which end it as
        `escaped` like the other tripwires."""
        if e.arm == "files":
            seen = seen_past_horizon(command_outputs(out.get("events") or []), self, e.question["until"])
            late = {"problem": f"escaped: its output showed recordings past the horizon: {', '.join(seen[:5])}"
                               + (f" and {len(seen) - 5} more" if len(seen) > 5 else ""), "passed": False}
            return {"timeout_s": self.timeout_s, "seen_past_horizon": seen, **(late if seen and out.get("problem") is None else {})}
        if e.arm != "mms":
            return {"timeout_s": self.timeout_s}
        found = audit(out.get("tools") or [], e.question["until"], e.home)
        if found["after_horizon"]:
            raise HorizonLeak(f"{e.question['qid']}: {found['after_horizon']} recording times after the horizon in {e.agent}'s "
                              "tool results: the index was not cut to it")
        return {"timeout_s": self.timeout_s, **found}

    def deny(self, own: Path | None = None) -> tuple[str, ...]:
        """`bench.Run.deny` (its run folder holds the data) and the download's words (`egolife` only in a command: the
        transcript says it once), and the other slots' folders of recordings (by path, by name, by glob, and the `glasses`
        and `transcript` folders in them, which a command's output may name too)."""
        others = [k for k in range(bench.JOBS_MAX) if own is None or self.slot(k) != own]
        base = f"mmsb-{self.key}-d"
        words = ("bench-data", PERSON.lower(), codex_run.IN_ONLY + "egolife",
                 *(f"{base}{k}/{sub}" for k in others for sub in (VIDEO, TEXT)),
                 *(codex_run.IN_ONLY + w for k in others for w in (str(self.view_dir(k)), f"{base}{k}", f"*-d{k}", f"?-d{k}")))
        return tuple(dict.fromkeys((*super().deny(own), *(re.sub(r"[\\/]+", "/", w.lower()) for w in words))))

    def check_paths(self) -> None:
        """The files arm's folders hold none of its deny strings, else every command naming them would be refused."""
        deny = self.deny(self.slot(0))
        for folder in (self.view_dir(0), self.bin, self.slot(0)):
            if hit := codex_run.reached(str(folder), deny):
                raise ValueError(f"{folder} holds {hit!r}, which the files arm refuses to reach: use another --temp")


def measure_decode(run: EgoRun) -> float:
    """Seconds one glasses window takes the reader to decode, at its default profile, over three of the run's clips (the first,
    a middle one and the last)."""
    clips = run.clips()
    store = types.SimpleNamespace(path_of=lambda item: item.path)
    spent, windows = 0.0, 0
    for c in (clips[0], clips[len(clips) // 2], clips[-1]):
        item = types.SimpleNamespace(kind="video", source=VIDEO, t0=local(run.days, c.t0), path=run.data / clip_path(run.days, c))
        began = time.perf_counter()
        windows += sum(1 for _ in reader._video(store, item, dict(reader.PROFILE), None, None, False))
        spent += time.perf_counter() - began
    return spent / windows


def check_snapshot(run: EgoRun, made: haystack.Plan) -> None:
    """The snapshot holds each clip as a video item at its `t0` and `t1` (the clip table's) and each transcript line as a record
    at its end, and none at the time it arrived (what a time field that did not parse falls back to)."""
    with closing(sqlite3.connect(run.snapshot() / "index.sqlite", timeout=30)) as db:
        items = {name: (t0, t1, state) for name, t0, t1, state in
                 db.execute("SELECT name, t0, t1, state FROM items WHERE source = ? AND kind = 'video'", (VIDEO,))}
        stamps = sorted(t for (t,) in db.execute("SELECT t FROM records WHERE source = ?", (TEXT,)))
    if len(items) != len(made.files):
        raise RuntimeError(f"the snapshot holds {len(items)} clips, the plan {len(made.files)}")
    for f in made.files:
        t0, t1, state = items[f.name]
        if state != "ok" or abs(t0 - f.t) > 1e-6 or abs(t1 - (f.t + f.spec["length"])) > 1e-6:
            raise RuntimeError(f"the snapshot's clip {f.name} is {state} at {t0}..{t1}, not ok at {f.t}..{f.t + f.spec['length']}")
    want = sorted(datetime.fromisoformat(line["end"]).timestamp() for rows in made.logs.values() for line in rows)
    if len(stamps) != len(want) or any(abs(a - b) > 1e-6 for a, b in zip(stamps, want)):
        raise RuntimeError(f"the snapshot's {len(stamps)} transcript records are not the plan's {len(want)} lines at their ends")


def utc_offset(day: date) -> timedelta | None:
    """The local UTC offset at the start of `day`."""
    return datetime(day.year, day.month, day.day).astimezone().utcoffset()


def crosses_a_change(days: Sequence[date], offset=None) -> bool:
    """Whether the local UTC offset changes between the start of the first of `days` and the end of the last (the clocks change)."""
    return len({(offset or utc_offset)(d) for d in (*days, days[-1] + timedelta(days=1))}) > 1


def default_days(today: date, n: int = DAYS, offset=None) -> tuple[date, ...]:
    """The latest `n` consecutive days before `today` that cross no change of the local UTC offset: elapsed time across such a
    night is an hour different in mms (which keeps offsets) and in the files arm (which keeps naive local times)."""
    last = today - timedelta(days=1)
    while crosses_a_change(days := tuple(last - timedelta(days=n - 1 - i) for i in range(n)), offset):
        last -= timedelta(days=1)
    return days


def check_days(days: Sequence[date], offset=None) -> None:
    """`days` are consecutive, oldest first, and cross no change of the local UTC offset."""
    if any(b - a != timedelta(days=1) for a, b in zip(days, days[1:])):
        raise ValueError(f"the days are consecutive, oldest first: {', '.join(d.isoformat() for d in days)}")
    if crosses_a_change(days, offset):
        raise ValueError(f"{days[0]} to {days[-1]} cross a change of the local UTC offset (the clocks change): elapsed time across "
                         "that night would differ by an hour between mms and the files arm; use days that do not")


def check_transcript(run: EgoRun, made: haystack.Plan) -> None:
    """No transcript line holds a word the files arm refuses to see in an output (`deny`, the words that are not command-only):
    a line the agent reads and prints would end its exec as `escaped`."""
    deny = run.deny(run.slot(0))
    for day, rows in made.logs.items():
        for line in rows:
            if hit := codex_run.reached(line, deny, output=True):
                raise ValueError(f"the transcript line ending {line['end']} holds {hit!r}, which the files arm ends an exec for "
                                 f"printing: {line['en'] or line['zh']!r}")


def build(out: Path, source: Path, seed: int, *, temp: Path, days: Sequence[date] | None = None, per_age: int = PER_AGE,
          speech: int = SPEECH) -> EgoRun:
    """Choose the questions, link the clips and write the transcript into the run's data (frozen), load the snapshot home (every
    clip at its time, the transcript as a log source with `end` as its time), check it, and measure the decode. `run.json` comes
    last, so a folder without it is a build that failed. `per_age` and `speech` are the questions of each age bin and for speech."""
    source, out, temp = Path(source), Path(out), Path(temp)
    if days is None:
        days = default_days(date.today())
    days = tuple(days)
    check_days(days)
    clips = clip_table(source, len(days))
    made = plan(source, seed, days, clips, load_oracles(), per_age, speech)
    folder = out / f"{datetime.now():%Y%m%d-%H%M}-egolife-s{seed}"
    if folder.exists():
        raise ValueError(f"{folder} exists: a run is one a minute for a seed; wait, or remove it")
    run = EgoRun(folder, seed, "egolife", days, folder / "data", temp, secrets.token_hex(5), source=source, per_age=per_age,
                 speech=speech)
    check_transcript(run, made)
    folder.mkdir(parents=True)
    bench.write_json(folder / "clips.json", [dataclasses.asdict(c) for c in clips])
    for f in made.files:
        (run.data / f.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source / f.spec["src"], run.data / f.path)
        except OSError as error:
            raise ValueError(f"cannot link {source / f.spec['src']} into {run.data}: {error}; --out must be on the download's "
                             "volume") from None
    (run.data / TEXT).mkdir(parents=True, exist_ok=True)
    for day, rows in made.logs.items():
        (run.data / TEXT / f"{day}.jsonl").write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    bench.freeze(run.data, run.manifest)
    bench.ensure_bin(run)
    home = run.snapshot()
    home.mkdir(parents=True)
    bench.write_config(home, bench.SNAPSHOT_CONFIG)
    daemon = bench.Daemon(home).start()
    try:
        for n, f in enumerate(made.files, 1):
            stored = daemon.api.call("remember", path=str(run.data / f.path), t=f.t, source=VIDEO, name=f.name)
            if len(stored["stored"]) != 1 or stored["unchanged"]:
                raise RuntimeError(f"{f.path} was not stored once: {stored}")
            if n % 500 == 0:
                print(f"remembered {n} of {len(made.files)} clips", flush=True)
        daemon.api.call("source_set", name=TEXT, kind="log", path=str(run.data / TEXT), glob="*.jsonl", time_field="end", max_gb=1)
        total = sum(len(rows) for rows in made.logs.values())
        bench.wait_for(lambda: next((s["records"] for s in daemon.api.call("status")["sources"] if s["name"] == TEXT), 0) >= total,
                       timeout=600, what=f"the {total} transcript records")
    finally:
        daemon.stop()
    bench.wait_for(lambda: not (home / "daemon.json").exists(), what="the snapshot's daemon to stop")
    check_snapshot(run, made)
    video, heard, dropped = pools(read_items(source), clips, read_transcripts(source, len(days)), len(days))
    meta = {"dataset": "egolife", "generator": VERSION, "source": str(source), "seed": seed, "scale": "egolife",
            "days": [d.isoformat() for d in days], "data": str(run.data), "temp": str(temp), "tag": run.tag,
            "plan": bench.plan_hash(made), "files": bench.files_hash(made), "decode_s": round(measure_decode(run), 4),
            "timeout_s": TIMEOUT_S, "per_age": per_age, "speech": speech,
            "pools": {"video": len(video), "speech": len(heard), "dropped": dict(dropped)},
            "created": bench.now_text(), "mms_commit": bench.commit_of(), "probe": {"status": "none"}}
    bench.write_json(folder / "run.json", meta)
    return EgoRun.open(folder)


# -- the agents -------------------------------------------------------------------------------------------------------------------

def set_timeout(run: EgoRun, seconds: float) -> EgoRun:
    """The run with another exec time, kept in its run.json (each result records the time it had)."""
    meta = bench.read_json(run.folder / "run.json")
    bench.write_json(run.folder / "run.json", {**meta, "timeout_s": seconds})
    return dataclasses.replace(run, timeout_s=seconds)


# -- the decider tier -------------------------------------------------------------------------------------------------------------

def oracle_record(api, q: dict, model: bench.Model) -> dict:
    """One question's oracle: the yes/no asked over each range of `oracle_ranges` (up to ORACLE_LIMIT sightings and
    ORACLE_CANDIDATES candidates, each ask allowed ORACLE_MAX_S), scored against the targets with TOL_S. `found`: a sighting
    within TOL_S of a target; `top5`: found, or a candidate that near; `p`: the best p of either; `away`: sightings farther from
    every target (the event may recur, so they are told beside the recall and not as a fault)."""
    moments = q["truth"]["moments"]
    results = []
    for a, b in oracle_ranges(q):
        params = {"question": q["oracle"], "sources": [q["oracle_source"]], "since": a, "until": b, "limit": bench.ORACLE_LIMIT,
                  "candidates": bench.ORACLE_CANDIDATES, "max_s": bench.ORACLE_MAX_S}
        results.append(bench.ask(api, params, bench.ORACLE_PROJECT))
    sightings = [s for r, _ in results for s in r["sightings"]]
    candidates = [c for r, _ in results for c in r["candidates"]]
    spans = [bench.span_of(s) for s in sightings]
    got = bench.touched(spans, moments, TOL_S)
    stops = [r.get("stopped") for r, _ in results]
    return {"qid": q["qid"], "type": q["type"], "template": q["template"], "level": q["level"], "model": model.name,
            "question": q["oracle"], "source": q["oracle_source"], "ranges": [list(r) for r in oracle_ranges(q)],
            "found": bool(got), "top5": bool(got) or any(bench.meets(t, t, moments, TOL_S) for t in
                                                        (bench.span_of(c)[0] for c in candidates)),
            "p": bench.needle_p(moments, sightings, candidates, tol=TOL_S), "moments": len(moments), "moments_found": got,
            "away": sum(1 for a, b in spans if not bench.meets(a, b, moments, TOL_S)),
            "windows": sum(r["progress"]["windows"] for r, _ in results),
            "seconds": round(sum(r["progress"]["seconds"] for r, _ in results), 2), "wall_s": round(sum(w for _, w in results), 2),
            "stopped": "max_s" if "max_s" in stops else next((s for s in stops if s), None),
            "warnings": sorted({w for r, _ in results for w in r.get("warnings", [])}), "sightings": sightings,
            "candidates": candidates}


# -- the report -------------------------------------------------------------------------------------------------------------------

SPEECH_LABEL = {"records": "speech"}  # a question that needs sound is a records question to bench (the transcript is a log source)
P_LEVELS = (0.5, 0.3)  # the p from which a needle is counted as nearly found, besides the model's own hit


def duel(base: Sequence[dict], other: Sequence[dict], questions: Sequence[dict]) -> dict:
    """`bench.paired` for two arms' results of the same questions: `base` and `other` figures, the difference `other - base`
    (the mean of the paired differences) with Newcombe's interval from MIN_DISCORDANT pairs, b (right only in `other`), c (right
    only in `base`) and McNemar's p."""
    got = bench.paired(list(zip(questions, base, other)))
    return {"n": got["n"], "base": got["files"], "other": got["mms"], **{k: got[k] for k in ("diff", "ci", "b", "c", "p")}}


def unparsed(rs: Sequence[dict]) -> int:
    """The results with no problem whose answer has no letter A to D (they count as wrong)."""
    return sum(1 for r in rs if r["problem"] is None and grade.choice(r["answer"]) is None)


def arm_figures(rs: Sequence[dict]) -> dict:
    return {**bench.side(list(rs)), "unparsed": unparsed(rs), "wilson": list(bench.wilson(sum(r["passed"] for r in rs), len(rs)))}


def horizon_figures(questions: Sequence[dict], mms: Sequence[dict]) -> dict:
    """How much of each question's recording the horizon cut off (the seconds from H to the query), and what the mms results'
    audits say: recording times past H in their tool results, and asks started past it."""
    gaps = [q["query"] - q["until"] for q in questions]
    audited = [r for r in mms if "after_horizon" in r]
    return {"median_s": bench.median(gaps), "max_s": max(gaps) if gaps else None, "past_30s": sum(1 for g in gaps if g > 30),
            "audited": len(audited), "after_horizon": sum(r["after_horizon"] for r in audited),
            "asks": sum(r["asks"] for r in audited), "asks_past_horizon": sum(r["asks_past_horizon"] for r in audited)}


def group_figures(rows: list[tuple]) -> dict:
    """The figures of triples `(question, blind, files, mms)` of one set of questions: each arm, and the three pairings."""
    qs, blind, files, mms = ([r[i] for r in rows] for i in range(4))
    return {"n": len(rows), "arms": {"blind": arm_figures(blind), "files": arm_figures(files), "mms": arm_figures(mms)},
            "mms_files": duel(files, mms, qs), "files_blind": duel(blind, files, qs), "mms_blind": duel(blind, mms, qs)}


def agent_figures(plan_: haystack.Plan, arms: dict, slug: str) -> dict | None:
    """One agent with one model: the questions with a done result in all three arms (a result to run again, or none, leaves a
    question out), as video and speech; the video ones by age and by category; how each arm worked; the crossover."""
    qs = {q["qid"]: q for q in plan_.questions}
    blind, files, mms = arms.get("blind", {}), arms.get("files", {}), arms.get(f"mms-{slug}", {})
    if not mms:
        return None
    rows = [(q, blind[qid], files[qid], mms[qid]) for qid, q in qs.items()
            if all(bench.is_done(a.get(qid)) for a in (blind, files, mms))]
    video, speech = [r for r in rows if r[0]["type"] == "video"], [r for r in rows if r[0]["type"] == "records"]
    ages = {name: group_figures(of) for name in AGE_NAMES if (of := [r for r in video if r[0]["level"] == name])}
    levels = {name: bench.lean([(q, f, m) for q, _, f, m in [r for r in video if r[0]["level"] == name]]) for name in ages}
    return {"questions": len(qs), "paired": len(rows), "video": group_figures(video) if video else None,
            "speech": group_figures(speech) if speech else None, "ages": ages,
            "crossover": bench.crossover(levels, order=AGE_NAMES), "thin": bench.thin_levels(levels, order=AGE_NAMES),
            "categories": {c: {"n": len(of), **{a: sum(r[i]["passed"] for r in of) for a, i in (("blind", 1), ("files", 2), ("mms", 3))}}
                           for c in CATEGORIES if (of := [r for r in video if r[0]["template"] == c])},
            "work": {"files": bench.worked([r[2] for r in video]), "mms": bench.worked([r[3] for r in video])},
            "cost": {a: bench.per_correct([r[i] for r in video]) for a, i in (("blind", 1), ("files", 2), ("mms", 3))},
            "horizon": horizon_figures([r[0] for r in rows], [r[3] for r in rows]),
            "appendix": [{"qid": q["qid"], "level": q["level"], "template": q["template"], "type": SPEECH_LABEL.get(q["type"], q["type"]),
                          **{a: bench.cell_of(r) for a, r in (("blind", b), ("files", f), ("mms", m))}} for q, b, f, m in rows]}


def decider_recall(records: Sequence[dict]) -> dict:
    """A model's oracle records as recall: at its hit (`found`), at p of 0.5 and 0.3, in the top 5; the median needle p, the
    sightings away from every target, and the median seconds a window."""
    return {"asked": len(records), "found": sum(bool(r["found"]) for r in records),
            **{f"p{int(100 * x)}": sum((r["p"] or 0) >= x for r in records) for x in P_LEVELS},
            "top5": sum(bool(r["top5"]) for r in records), "p": bench.median([r["p"] for r in records if r["p"] is not None]),
            "away": sum(r["away"] for r in records), "windows": sum(r["windows"] for r in records),
            "s_per_window": bench.median([r["seconds"] / r["windows"] for r in records if r["windows"]]),
            "stopped": sum(1 for r in records if r["stopped"] == "max_s")}


def decider_figures(run: EgoRun) -> dict:
    """The decider tier per model: recall overall, as video and speech, and by category."""
    out = {}
    for folder in sorted((run.folder / "results" / "decider").glob("*")):
        records = [bench.read_json(f) for f in sorted(folder.glob("*.json")) if f.stem != "warm"]
        warm_file = folder / "warm.json"
        if records:
            out[records[0]["model"]] = {
                "first_s": bench.read_json(warm_file)["seconds"] if warm_file.exists() else None, "all": decider_recall(records),
                "types": {SPEECH_LABEL.get(t, t): decider_recall(of) for t in ("video", "records")
                          if (of := [r for r in records if r["type"] == t])},
                "categories": {c: decider_recall(of) for c in CATEGORIES if (of := [r for r in records if r["template"] == c])}}
    return out


def numbers(run: EgoRun) -> dict:
    """The report's figures (report.json; `markdown` renders them): per agent and model the three arms over the video questions
    and over the speech ones apart, by age and by category, how each arm worked and what a correct answer cost, the horizon's
    audit, and the decider tier."""
    plan_ = run.plan()
    meta = bench.read_json(run.folder / "run.json")
    results = bench.results_of(run)
    models_file = run.folder / "models.json"
    infos = bench.read_json(models_file) if models_file.exists() else {}
    questions = plan_.questions
    agents = {agent: {name: got for name in bench.ordered(infos) if (got := agent_figures(plan_, arms, infos[name]["slug"]))}
              for agent, arms in results.items()}
    every = [r for arms in results.values() for rs in arms.values() for r in rs.values()]
    return {"run": {"name": run.name, "seed": run.seed, "days": len(run.days), "commit": meta["mms_commit"],
                    "timeout_s": run.timeout_s, "decode_s": run.decode_s, "pools": meta["pools"],
                    "video": [sum(1 for q in questions if q["type"] == "video" and q["level"] == a) for a in AGE_NAMES],
                    "speech": sum(1 for q in questions if q["type"] == "records"),
                    "categories": {c: sum(1 for q in questions if q["template"] == c) for c in CATEGORIES},
                    "horizon": horizon_figures(questions, [])},
            "agents": {a: models for a, models in agents.items() if models}, "decider": decider_figures(run),
            "totals": {"results": len(every), "problems": dict(Counter(bench.kind_of(r["problem"]) for r in every if r["problem"])),
                       "to_rerun": sum(1 for r in every if not bench.is_done(r))}}


def right(k: int, n: int, ci: tuple[float, float] | None = None) -> str:
    """`k/n (60%)`, with the Wilson interval when given."""
    if not n:
        return "-"
    return f"{k}/{n} ({100 * k / n:.0f}%" + (f", {100 * ci[0]:.0f}-{100 * ci[1]:.0f}" if ci else "") + ")"


def diff_text(d: dict) -> str:
    """A difference in points, with its interval when there are pairs enough for one (b and c are in a column of their own)."""
    if d["diff"] is None:
        return "-"
    return bench.pts(d["diff"]) + ("" if d["ci"] is None else f" [{100 * d['ci'][0]:+.0f}, {100 * d['ci'][1]:+.0f}]")


def duel_row(label: str, d: dict) -> list[str]:
    return [label, str(d["n"]), diff_text(d), f"{d['b']} / {d['c']}", bench.chance(d["p"])]


def arm_table(g: dict) -> list[str]:
    rows = []
    for arm in bench.ARMS:
        a = g["arms"][arm]
        rows.append([arm, right(a["correct"], a["n"], a["wilson"]), bench.number(a["median_s"]), bench.number(a["median_calls"], ".0f"),
                     bench.kilo(a), str(a["timeouts"]), str(a["unparsed"])])
    return [bench.table(["arm", "right (95% interval)", "median s", "median calls", "tokens k", "timed out", "no letter"], rows), "",
            bench.table(["pair", "n", "difference (95% CI)", "b / c", "McNemar p"],
                        [duel_row("mms vs files", g["mms_files"]), duel_row("files vs blind", g["files_blind"]),
                         duel_row("mms vs blind", g["mms_blind"])]), "",
            "A difference is the second arm less the first, in points; b is the questions right only in the second, c only in the first."]


def age_lines(m: dict) -> list[str]:
    rows = []
    for name, g in m["ages"].items():
        a = g["arms"]
        d = g["mms_files"]
        rows.append([name, str(g["n"]), *(right(a[x]["correct"], g["n"]) for x in bench.ARMS), diff_text(d), f"{d['b']} / {d['c']}",
                     bench.chance(d["p"]), bench.both(a["files"]["median_s"], a["mms"]["median_s"]),
                     bench.both(bench.tokens_k(a["files"]), bench.tokens_k(a["mms"]), ".0f")])
    return [bench.table(["age", "n", "blind (the floor)", "files", "mms", "mms vs files", "b / c", "p", "median s (files / mms)",
                         "tokens k (files / mms)"], rows), "",
            f"The youngest age at which mms leads files: {bench.crossover_text(m['crossover'], m['thin'])} (n = 10 per age: a sign, "
            "not a finding)."]


def work_lines(m: dict) -> list[str]:
    rows = []
    for arm in ("files", "mms"):
        w, c = m["work"][arm], m["cost"][arm]
        rows.append([arm, bench.share(w["ask_execs"], w["n"]) if arm == "mms" else "-",
                     " / ".join(bench.number(w[k]) for k in ("asks", "finds", "looks")) if arm == "mms" else "-",
                     bench.known_of(w, "windows") if arm == "mms" else "-", bench.known_of(w, "gpu_s") if arm == "mms" else "-",
                     str(w["max_s"]) if arm == "mms" else "-", bench.known_of(w, "views") if arm == "files" else "-",
                     bench.known_of(w, "shells") if arm == "files" else "-", f"{c['correct']}/{c['n']}", bench.cost_cell(c, "s"),
                     bench.cost_cell(c, "tokens_k"), bench.cost_cell(c, "gpu_s") if arm == "mms" else "-"])
    return [bench.table(["arm", "called ask", "asks / finds / looks", "ask windows", "ask GPU s", "asks stopped at max_s", "pictures viewed",
                         "shell commands", "correct", "s per correct", "tokens k per correct", "GPU s per correct"], rows)]


def decider_lines(tier: dict) -> list[str]:
    if not tier:
        return ["No decider results yet."]
    rows = []
    for name, d in tier.items():
        for label, r in [("all", d["all"]), *d["types"].items(), *d["categories"].items()]:
            rows.append([name, label, str(r["asked"]), bench.share(r["found"], r["asked"]), bench.share(r["p50"], r["asked"]),
                         bench.share(r["p30"], r["asked"]), bench.share(r["top5"], r["asked"]), bench.number(r["p"], ".2f"),
                         str(r["away"]), bench.number(r["s_per_window"], ".3f"), str(r["stopped"])])
    return [bench.table(["model", "set", "asked", "recall (at its hit)", "p >= 0.5", "p >= 0.3", "top 5", "median needle p",
                         "sightings away from every target", "median s per window", "stopped at max_s"], rows), "",
            "Recall and the top 5 are read first. A sighting away from every target is told beside them and is no fault: the "
            "event may recur."]


def horizon_rule(h: dict) -> str:
    return (f"Every arm saw the clips that ended by each question's horizon H (median {bench.number(h['median_s'], '.0f')} s before "
            f"the question, at most {bench.number(h['max_s'], '.0f')} s; {h['past_30s']} questions more than 30 s, which fell after a "
            "gap) and the transcript lines that ended by H.")


def audit_line(h: dict) -> str:
    return (f"Recording times after H in mms tool results: {h['after_horizon']} in {h['audited']} audited execs; asks started past H "
            f"(the agent's confusion about mms's clock, harmless): {h['asks_past_horizon']}/{h['asks']}.")


def markdown(figures: dict) -> str:
    """report.md: the tables of `numbers`."""
    r = figures["run"]
    out = ["# EgoLifeQA through the mms bench", "",
           f"Run {r['name']} (seed {r['seed']}, {r['days']} days, mms commit {r['commit']}). {sum(r['video'])} video questions "
           f"({', '.join(f'{n} aged {a}' for a, n in zip(AGE_NAMES, r['video']))}; by category "
           f"{', '.join(f'{c} {n}' for c, n in r['categories'].items())}) and {r['speech']} speech questions, asked as of each "
           f"question's horizon. The pool before choosing: {r['pools']['video']} video and {r['pools']['speech']} speech; dropped: "
           f"{', '.join(f'{n} {why}' for why, n in r['pools']['dropped'].items())}. Each exec had {r['timeout_s']:g} s. "
           f"A glasses window decodes in {r['decode_s'] * 1000:.0f} ms here (mms assumes {ask_script.DECODE_S * 1000:.0f}).",
           "", horizon_rule(r["horizon"]), "",
           "No shortcut probe: a public dataset, so the blind arm is the floor (what the agent knows without the recordings)."]
    for agent, models in figures["agents"].items():
        for name, m in models.items():
            video = m["video"]
            out += ["", f"## {agent} with {name}: {video['n'] if video else 0} video questions of {m['questions']} (paired: done in "
                        "all three arms)", "", *(arm_table(video) if video else ["None paired yet."]), "",
                    f"### {agent} with {name}: by age (mms vs files; blind is the floor)",
                    "", *age_lines(m), "", f"### {agent} with {name}: by category (right / n)", "",
                    bench.table(["category", "n", *bench.ARMS], [[c, str(v["n"]), *(str(v[a]) for a in bench.ARMS)]
                                                                for c, v in m["categories"].items()]),
                    "", "TaskMaster has few questions in each age: read it as a count, not a rate.", "",
                    f"### {agent} with {name}: the {m['speech']['n'] if m['speech'] else 0} speech questions, apart", ""]
            out += (arm_table(m["speech"]) + ["", "Answered from the transcripts alone, since d1-3B does not hear."]
                    if m["speech"] else ["None paired yet."])
            out += ["", f"### {agent} with {name}: how each arm worked, and what a correct answer cost (the video questions)", "",
                    *work_lines(m)]
            out += ["", f"### {agent} with {name}: the horizon, from the mms results", "", audit_line(m["horizon"])]
    if not figures["agents"]:
        out += ["", "No agent results yet."]
    out += ["", "## The decider tier (each model's yes/no over each question's evidence, no agent)", "", *decider_lines(figures["decider"])]
    t = figures["totals"]
    out += ["", "## Read with these", "",
            "- The published scores (GPT-4o 36.2, Gemini-1.5-Pro 36.9 and EGAgent 57.5 in the paper) are on all 500 items and 7 days "
            "without tools: a scale, not a comparison.",
            "- Chance is 25%. With 40 paired questions p < 0.05 needs about 6 or more discordant pairs all one way; by age, 10 "
            "questions each, only a sign.",
            "- EgoLife has been public since 2025: the blind arm measures what the models know of it.",
            "- The files and mms arms both get EgoLife's transcripts (a Chinese line and its English); mms reads them as records.",
            f"- A glasses window decodes in {r['decode_s'] * 1000:.0f} ms here and mms assumes {ask_script.DECODE_S * 1000:.0f} ms, so "
            "its estimates and lint are low for these clips; asks stopped at `max_s` and timed-out execs are counted above.",
            "- Names are neutral in the files and mms arms: the clips are `glasses/<date>/HH.MM.SS.mp4`, with no person or dataset name.",
            "- A question is asked as of its horizon, so the seconds before it (above) are lost to every arm.",
            f"- Problems by kind: {bench.kinds_text(t['problems'])}. Results to run again: {t['to_rerun']} of {t['results']}."]
    for agent, models in figures["agents"].items():
        for name, m in models.items():
            body = [[x["qid"], x["level"], x["template"], x["type"], *(bench.cell_text(x[a], "asks") for a in bench.ARMS)]
                    for x in m["appendix"]]
            out += ["", f"## Appendix: {agent} with {name}, one row per question", "",
                    bench.table(["qid", "age", "category", "type", *bench.ARMS], body)]
    return "\n".join(out) + "\n"


# -- the command line -------------------------------------------------------------------------------------------------------------

def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egolife.py", description="EgoLifeQA through the mms bench: blind, files and mms.")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("build", help="choose the questions, link the clips and load a snapshot home")
    one.add_argument("--source", type=Path, required=True, help="the download (A1_JAKE, EgoLifeQA, EgoLifeCap)")
    one.add_argument("--seed", type=int, default=1)
    one.add_argument("--out", type=Path, required=True, help="where runs go: on the download's volume, with room for 42 GB")
    one.add_argument("--temp", type=Path, required=True, help="where the slots' folders go: on the same volume")
    for command, help_ in (("decider", "each model's yes/no over each question's evidence (the GPU, alone)"),
                           ("agents", "the execs of Claude and Codex: blind, files and mms (the plans' quota)")):
        one = sub.add_parser(command, help=help_)
        one.add_argument("run", type=Path, help="a run folder")
        one.add_argument("--models", default=bench.CONTROLS_MODEL, help="model names, comma separated")
        one.add_argument("--models-dir", type=Path, default=bench.MODELS_DIR, help="where the model folders are (MMS_MODELS)")
        if command == "agents":
            one.add_argument("--agents", default=",".join(bench.AGENTS))
            one.add_argument("--jobs", type=int, default=bench.JOBS, help=f"execs at once in the blind and files arms, at most "
                                                                          f"{bench.JOBS_MAX} (the mms arm runs one at a time)")
            one.add_argument("--subset", choices=bench.SUBSETS, default="all", help="pilot: the first 2 of each age and of the speech ones")
            one.add_argument("--only", default="", help="these qids of the subset only, comma separated (a smoke)")
            one.add_argument("--timeout", type=float, help="seconds an exec may take (kept in the run); default the run's")
            one.add_argument("--dry-run", action="store_true", help="write the first exec's files per agent and arm, run nothing")
    sub.add_parser("report", help="the three arms compared; written in the run's folder").add_argument("run", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parser_of()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            print(build(args.out, args.source, args.seed, temp=args.temp).folder)
        elif args.command == "report":
            run = EgoRun.open(args.run)
            figures = numbers(run)
            text = markdown(figures)
            (run.folder / "report.md").write_text(text, encoding="utf-8")
            (run.folder / "report.json").write_text(json.dumps(figures, indent=1), encoding="utf-8")
            print(text)
        else:
            run = EgoRun.open(args.run)
            models, skipped = bench.resolve_models(bench.model_names(args.models), args.models_dir)
            if args.command == "decider":
                bench.show_decider(run, models, skipped)
                bench.run_decider(run, models)
            else:
                names = [n.strip() for n in args.agents.split(",") if n.strip()]
                if bad := [n for n in names if n not in bench.AGENTS]:
                    parser.error(f"--agents is some of {', '.join(bench.AGENTS)}, not {', '.join(bad)}")
                only = [q.strip() for q in args.only.split(",") if q.strip()]
                if args.timeout is not None:
                    run = dataclasses.replace(run, timeout_s=args.timeout) if args.dry_run else set_timeout(run, args.timeout)
                if args.dry_run:
                    for line in skipped:
                        print(f"skipped {line}")
                    bench.dry_run(run, names, models, None, args.subset, only)
                else:
                    bench.show_agents(run, names, models, skipped, args.jobs, None, args.subset, only)
                    if "claude" in names:
                        claude_run.check_login()
                    real = bench.real_agents(timeout=run.timeout_s)
                    bench.run_agents(run, {n: real[n] for n in names}, models, args.jobs, subset=args.subset, only=only)
    except KeyboardInterrupt:
        print("interrupted: run the same command to continue", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, AssertionError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
