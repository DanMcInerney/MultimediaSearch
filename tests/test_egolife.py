"""EgoLifeQA through the mms bench (`benchmarks/egolife.py`), offline: a two-day fake download (tiny clips made with ffmpeg, SRT files with
the real file's oddities, a QA file of about twenty items) is read, selected from, built into a run, cut to each question's horizon
(the mms home's index and the files arm's folder), asked with the fake decider and with stub agents, and reported. No real agent or
model runs. The download and the run are built once for the module and worker; the tests that read them share them."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

import bench
import codex_run
import egolife
import pytest
from conftest import FAKE, Api, Daemon

from mms import config, decider_d1

DAYS = (date(2026, 10, 29), date(2026, 10, 30))
FAKE_MODEL = bench.Model("fake", FAKE, decider_d1.ABOUT)
OPTIONS = {"A": "Tasha", "B": "Alice", "C": "Shure", "D": "Lucia"}
SHADE = "Is shade:200,30,30 in view?"  # the fake decider's own words for the red box of the evidence clip
WORD = "Does the text have contains:screwdriver ?"
pytestmark = pytest.mark.xdist_group("egolife")

# Clips: a block starts with a short clip (18 s) at an unaligned time, then 30 s clips on :00 and :30; each day has two blocks and
# a gap between them. The evidence clip of day 1 (11:10:30 to 11:11:00) shows a red frame from 10 s to 20 s into it.
CLIPS = {1: [("11094208", 18), ("11100000", 30), ("11103000", 30), ("11110000", 30), ("11113000", 30), ("11120000", 30),
             ("12301225", 18), ("12303000", 30), ("12310000", 30), ("12313000", 30), ("12320000", 30), ("12323000", 30)],
         2: [("10050125", 18), ("10053000", 30), ("10060000", 30), ("10063000", 30), ("10070000", 30), ("10073000", 30),
             ("15195000", 18), ("15200000", 30), ("15203000", 30), ("15210000", 30), ("15213000", 30), ("15220000", 30)]}
EVIDENCE = "11103000"
# (offset from the hour, text lines): two-line entries with the real file's spacing, a one-line entry and one with no speaker
SRT = {(1, 11): [("09:47,100", "09:50,766", ["Jake: 好然后一个秒表", "Jake: Okay, then we need a stopwatch."]),
                 ("09:54,700", "09:55,700", ["Shure:  时间戳呗", "Shure: Mark a timestamp, right?"]),
                 ("10:44,000", "10:47,000", ["Alice:我用螺丝刀拧螺丝", "Alice:I use the screwdriver to tighten screws."]),
                 ("10:46,000", "10:49,500", ["Tasha: 好的", "Tasha: OK."]),
                 ("10:50,000", "10:51,000", ["Lucia: 要稍微使点力气"]),
                 ("12:25,000", "12:35,000", ["Jake: 再来一个", "Jake: One more."]),
                 ("12:40,000", "12:41,000", ["Jake: 快好了", "Jake: Almost done."])],
       (1, 12): [("30:40,000", "30:42,000", ["Katrina: 看这个", "Katrina: Look at this."]),
                 ("31:14,000", "31:16,000", ["Shure: 我们晚上做烧烤", "Shure: We will grill tonight."]),
                 ("34:00,000", "34:02,000", ["Jake: 收工", "Jake: Done for now."])],
       (2, 10): [("05:40,000", "05:42,000", ["Jake: 今天开始", "Jake: Let's start today."])],
       (2, 15): [("20:10,000", "20:12,000", ["Alice: 看", "Alice: Look."]), ("21:00,000", "21:01,000", ["小薇;哪个"])]}


def pl(day: int, text: str) -> float:
    """The plain time of `HH:MM:SS.ff` on a day."""
    h, m, s = text.split(":")
    return egolife.plain(day, int(h) * 3600 + int(m) * 60 + float(s))


def ep(day: int, text: str) -> float:
    return egolife.local(DAYS, pl(day, text))


def raw(id_: str, category: str, query: tuple, targets: list[tuple], *, audio: bool = False, listed: bool = False,
        date_text: str | None = None, time_text: str | None = None) -> dict:
    """One item of the QA file. `query` and each target are (day, `HH:MM:SS.ff`); a list of targets is a `time_list`."""
    def code(text: str) -> str:
        return text.replace(":", "").replace(".", "")

    day = targets[0][0]
    when = {"time_list": [code(t) for _, t in targets]} if listed else {"time": time_text or code(targets[0][1])}
    out = {"ID": id_, "query_time": {"date": f"DAY{query[0]}", "time": code(query[1])}, "type": category, "need_audio": audio,
           "need_name": False, "last_time": False, "trigger": "the table is full", "question": f"Question {id_}?",
           "answer": "B", "target_time": {"date": date_text or f"DAY{day}", **when}, "keywords": "use tool",
           "reason": "Saw someone with a tool"}
    out.update({f"choice_{k.lower()}": v for k, v in OPTIONS.items()})
    return out


ITEMS = [
    raw("1", "EntityLog", (1, "11:12:10.17"), [(1, "11:10:45.00")]),  # 0-2h; H 11:12:00
    raw("18", "EventRecall", (1, "11:10:50.00"), [(1, "11:10:05.00")]),  # 0-2h; H 11:10:30, before the evidence clip ends
    raw("2", "RelationMap", (1, "11:11:40.00"), [(1, "11:09:55.00")], date_text="Day1"),  # H 11:11:30; the date as "Day1"
    raw("12", "RelationMap", (1, "12:30:20.00"), [(1, "11:11:10.00")]),  # in the short clip after the gap: H 11:12:30
    raw("15", "RelationMap", (1, "11:12:20.00"), [(1, "11:09:50.00"), (1, "11:11:20.00")], listed=True),  # two targets, one range
    raw("17", "RelationMap", (1, "11:11:50.00"), [(1, "11:10:05.00")]),
    raw("3", "EntityLog", (1, "14:00:00.00"), [(1, "11:10:20.00")]),  # 2-6h; H 12:33:00
    raw("6", "HabitInsight", (1, "15:30:00.00"), [(1, "11:10:40.00"), (1, "12:30:45.00")], listed=True),  # two ranges
    raw("4", "RelationMap", (2, "09:00:00.00"), [(1, "11:10:50.00")]),  # 6-24h
    raw("20", "TaskMaster", (2, "09:00:00.00"), [(1, "11:11:15.00")]),
    raw("5", "HabitInsight", (2, "15:21:05.00"), [(1, "11:11:10.00")]),  # 24h+; H 15:21:00, the end of a clip
    raw("21", "EventRecall", (2, "15:21:00.00"), [(1, "12:30:50.00")]),
    raw("8", "TaskMaster", (1, "12:40:00.00"), [(1, "11:10:46.00")], audio=True),  # speech, 0-2h
    raw("9", "EventRecall", (2, "15:21:00.00"), [(1, "12:31:15.00")], audio=True),  # speech, 24h+
    raw("10", "EntityLog", (1, "13:00:00.00"), [(1, "11:30:00.00")]),  # dropped: the target is in the gap
    raw("11", "RelationMap", (4, "10:00:00.00"), [(4, "09:00:00.00")]),  # dropped: after the last day
    raw("13", "HabitInsight", (1, "13:00:00.00"), [(1, "11:15:34.17")], time_text="11153417DAY1_11181201"),  # dropped: a range
    raw("14", "TaskMaster", (2, "10:07:55.00"), [(2, "10:06:40.00")], audio=True)]  # dropped: no line within 30 s of the target
ORACLES = {i: {"source": egolife.VIDEO, "question": SHADE} for i in ("1", "18", "2", "12", "15", "17", "3", "6", "4", "20", "5", "21")}
ORACLES |= {i: {"source": egolife.TEXT, "question": WORD} for i in ("8", "9")}


def make_clip(ffmpeg: str, path: Path, seconds: int, n: int, evidence: bool) -> None:
    """A 64x64 clip of 2 frames a second with a silent AAC track; the colour is its own (remember keeps one copy of a file)."""
    video = f"color=c=0x{80:02x}{80:02x}{40 + 5 * n:02x}:s=64x64:r=2:d={seconds}"
    if evidence:
        video += ",drawbox=x=0:y=0:w=iw:h=ih:color=0xC81E1E:t=fill:enable='between(t,10,20)'"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "4", "-c:a", "aac", "-shortest",
                    str(path)], check=True)


@pytest.fixture(scope="module")
def source(tmp_path_factory, ffmpeg) -> Path:
    """The fake download, made once: `A1_JAKE/DAYn/*.mp4`, `EgoLifeQA/*.json` and `EgoLifeCap/Transcript/A1_JAKE/DAYn/*.srt`."""
    root = tmp_path_factory.mktemp("download")
    jobs = []
    for day, clips in CLIPS.items():
        (root / "A1_JAKE" / f"DAY{day}").mkdir(parents=True)
        jobs += [(root / "A1_JAKE" / f"DAY{day}" / f"DAY{day}_A1_JAKE_{name}.mp4", seconds, name == EVIDENCE and day == 1)
                 for name, seconds in clips]
    with ThreadPoolExecutor(4) as pool:
        for future in [pool.submit(make_clip, ffmpeg, path, seconds, n, evidence) for n, (path, seconds, evidence) in enumerate(jobs)]:
            future.result()
    (root / "EgoLifeQA").mkdir()
    (root / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json").write_text(json.dumps(ITEMS, ensure_ascii=False), encoding="utf-8")
    for (day, hour), entries in SRT.items():
        folder = root / "EgoLifeCap" / "Transcript" / "A1_JAKE" / f"DAY{day}"
        folder.mkdir(parents=True, exist_ok=True)
        blocks = [f"{n}\n00:{a} --> 00:{b}\n" + "\n".join(lines) for n, (a, b, lines) in enumerate(entries, 1)]
        (folder / f"A1_JAKE_DAY{day}_{hour:02d}000000.srt").write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module", autouse=True)
def oracle_file(tmp_path_factory):
    """The oracles file the module's runs read: `egolife.ORACLES` points at it."""
    path = tmp_path_factory.mktemp("oracles") / "oracles.json"
    path.write_text(json.dumps({"version": egolife.ORACLE_VERSION, "items": ORACLES}), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(egolife, "ORACLES", path)
        yield path


@pytest.fixture(scope="module")
def table(source) -> list[egolife.Clip]:
    return egolife.clip_table(source, 2)


@pytest.fixture(scope="module")
def built(tmp_path_factory, source) -> egolife.EgoRun:
    """A run of the fake download: 2 questions of each age and 2 of speech (deterministic: see ITEMS)."""
    root = tmp_path_factory.mktemp("run")
    return egolife.build(root / "out", source, 1, temp=root / "temp", days=DAYS, per_age=2, speech=2)


def clip_ends(run: egolife.EgoRun) -> list[float]:
    return [egolife.clip_end(run.days, c) for c in run.clips()]


def view_files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


# -- 1. parsing -------------------------------------------------------------------------------------------------------------------

def test_times_are_hundredths_and_days_are_numbers():
    assert egolife.clock_of("11210217") == pytest.approx(11 * 3600 + 21 * 60 + 2.17)
    assert egolife.day_of("DAY3") == 3 and egolife.day_of("Day1") == 1
    assert egolife.plain(2, 10.5) == 2 * 86400 + 10.5
    for bad in ("1121021", "11153417DAY1_11181201"):
        with pytest.raises(ValueError, match="HHMMSSFF"):
            egolife.clock_of(bad)
    with pytest.raises(ValueError, match="DAY and a number"):
        egolife.day_of("Tuesday")
    # a plain time is a local epoch of the day it names, kept to the hundredth, and a local ISO time to the millisecond
    t = ep(2, "10:05:01.25")
    assert datetime.fromtimestamp(t) == datetime(2026, 10, 30, 10, 5, 1, 250000)
    assert egolife.iso_of(DAYS, pl(2, "10:05:01.25")) == "2026-10-30T10:05:01.250"
    assert egolife.iso_of(DAYS, pl(1, "11:09:47.10")) == "2026-10-29T11:09:47.100"


def test_the_items_are_read_as_the_data_gives_them(source):
    items = {i["id"]: i for i in egolife.read_items(source)}
    assert len(items) == len(ITEMS)
    one = items["1"]
    assert one["query"] == pytest.approx(pl(1, "11:12:10.17")) and one["targets"] == [pytest.approx(pl(1, "11:10:45.00"))]
    assert (one["category"], one["audio"], one["answer"], one["options"]) == ("EntityLog", False, "B", OPTIONS)
    assert items["2"]["targets"] == [pytest.approx(pl(1, "11:09:55.00"))], "`Day1` is DAY1"
    assert items["6"]["targets"] == [pytest.approx(pl(1, "11:10:40.00")), pytest.approx(pl(1, "12:30:45.00"))], "a time_list, in order"
    assert items["8"]["audio"] is True and items["11"]["query"] > egolife.plain(4, 0)
    assert items["13"]["targets"] == [], "a range in one string is no moment"


def test_the_transcript_is_read_from_srt_entries(source):
    lines = egolife.read_transcripts(source, 2)
    assert len(lines) == 13 and [x["end"] for x in lines] == sorted(x["end"] for x in lines)
    by_en = {x["en"]: x for x in lines}
    first = by_en["Okay, then we need a stopwatch."]
    assert (first["speaker"], first["zh"]) == ("Jake", "好然后一个秒表")
    assert first["start"] == pytest.approx(pl(1, "11:09:47.10")) and first["end"] == pytest.approx(pl(1, "11:09:50.766")), \
        "times are offsets from the hour of the file's name"
    assert (by_en["Mark a timestamp, right?"]["speaker"], by_en["Mark a timestamp, right?"]["zh"]) == ("Shure", "时间戳呗"), "two spaces"
    glued = by_en["I use the screwdriver to tighten screws."]
    assert (glued["speaker"], glued["zh"]) == ("Alice", "我用螺丝刀拧螺丝"), "no space after the colon"
    assert pl(1, "11:10:44.00") <= by_en["OK."]["start"] <= glued["end"], "entries overlap"
    one = next(x for x in lines if x["zh"] == "要稍微使点力气")
    assert (one["speaker"], one["en"]) == ("Lucia", ""), "an entry of one line has no English"
    bare = next(x for x in lines if x["zh"] == "小薇;哪个")
    assert (bare["speaker"], bare["en"]) == ("", "")
    assert [x["end"] for x in egolife.read_transcripts(source, 1)] == [x["end"] for x in lines if x["end"] < egolife.plain(2, 0)]
    assert egolife.log_lines(DAYS, lines)["2026-10-29"][0]["start"] == "2026-10-29T11:09:47.100"


def test_the_clip_table_has_the_clips_in_time_order(table):
    assert len(table) == 24 and [c.t0 for c in table] == sorted(c.t0 for c in table)
    first = table[0]
    assert first.rel == "A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4" and first.day == 1
    assert first.t0 == pytest.approx(pl(1, "11:09:42.08")) and 17.9 < first.length < 18.2, "an 18 s clip at a name in hundredths"
    assert 29.9 < table[1].length < 30.2 and table[1].t0 == pytest.approx(pl(1, "11:10:00.00"))
    assert table[-1].day == 2 and table[-1].t0 == pytest.approx(pl(2, "15:22:00.00")) and table[-1].t1 == table[-1].t0 + table[-1].length


# -- 2. the horizon ---------------------------------------------------------------------------------------------------------------

def test_the_horizon_is_the_end_of_the_last_whole_clip(table):
    end = lambda day, text: egolife.horizon(table, pl(day, text))  # noqa: E731
    assert end(1, "11:12:10.17").t1 == pytest.approx(pl(1, "11:12:00.00"), abs=0.2)
    assert end(1, "11:12:10.17").t0 == pytest.approx(pl(1, "11:11:30.00"))
    assert end(1, "12:40:00.00").t1 == pytest.approx(pl(1, "12:33:00.00"), abs=0.2), "after a block, its last clip"
    # a query in the short first clip after a gap gets the clip before the gap
    assert end(1, "12:30:20.00").t1 == pytest.approx(pl(1, "11:12:30.00"), abs=0.2)
    # the clip that ended exactly at the query counts
    clip = egolife.horizon(table, table[3].t1)
    assert clip == table[3]
    assert egolife.horizon(table, table[3].t1 - 0.001) == table[2]
    assert egolife.horizon(table, pl(1, "11:00:00.00")) is None, "no clip had ended"
    assert egolife.horizon(table, pl(2, "23:00:00.00")) == table[-1]


def test_every_question_is_asked_as_of_a_clip_end_after_its_targets(built, table):
    plan = built.plan()
    ends = clip_ends(built)
    assert len(plan.questions) == 10 and {q["qid"] for q in plan.questions} == {f"ego-{i:03d}" for i in (1, 18, 3, 6, 4, 20, 5, 21, 8, 9)}
    for q in plan.questions:
        assert q["until"] in ends, "H is exactly the end mms stores for a clip"
        assert q["until"] <= q["query"] and all(m[0] <= q["until"] for m in q["truth"]["moments"])
        assert q["since"] == plan.files[0].t
    by = {q["qid"]: q for q in plan.questions}
    assert by["ego-001"]["until"] == pytest.approx(ep(1, "11:12:00.00"), abs=0.2)
    assert by["ego-018"]["until"] == pytest.approx(ep(1, "11:10:30.00"), abs=0.2), "the evidence clip has not ended by this question"
    assert by["ego-005"]["until"] == pytest.approx(ep(2, "15:21:00.00"), abs=0.2) and by["ego-005"]["until"] < by["ego-005"]["query"]


# -- 3. the selection -------------------------------------------------------------------------------------------------------------

IDS = itertools.count(1000)


def synthetic(bin_: str, category: str, n: int) -> list[dict]:
    """`n` items of an age bin and category, with targets inside the first block of day 1."""
    query = {"0-2h": pl(1, "11:11:50.00"), "2-6h": pl(1, "14:00:00.00"), "6-24h": pl(2, "09:00:00.00"), "24h+": pl(2, "15:21:00.00")}[bin_]
    return [{"id": str(next(IDS)), "category": category, "audio": False, "need_name": False, "last": False, "query": query,
             "targets": [pl(1, "11:10:05.00")], "question": "?", "options": OPTIONS, "answer": "B", "keywords": "", "reason": ""}
            for k in range(n)]


def test_the_pools_count_what_is_dropped_and_why(source, table):
    items, lines = egolife.read_items(source), egolife.read_transcripts(source, 2)
    video, speech, dropped = egolife.pools(items, table, lines, 2)
    assert {i["id"] for i in video} == {"1", "18", "2", "12", "15", "17", "3", "6", "4", "20", "5", "21"}
    assert {i["id"] for i in speech} == {"8", "9"}
    assert dict(dropped) == {"its query is after the last day": 1, "its time is a range or a list in one string": 1,
                             "a target is outside the clips (a gap) or after the horizon": 1,
                             "speech: no transcript line near a target": 1}
    assert [egolife.age_of(i) for i in video if i["id"] in ("1", "3", "4", "5")] == ["0-2h", "2-6h", "6-24h", "24h+"]


def test_select_takes_the_bins_in_order_and_the_speech_apart(source, table):
    items, lines = egolife.read_items(source), egolife.read_transcripts(source, 2)
    one = egolife.select(items, table, lines, 1, per_age=1, speech=1, days=2)
    assert [(i["id"], i["age"]) for i in one] == [("1", "0-2h"), ("3", "2-6h"), ("4", "6-24h"), ("21", "24h+"), ("8", "0-2h")], \
        "the rarest category of each bin (ties by name), then the speech question"
    assert [i["audio"] for i in one] == [False] * 4 + [True]
    two = egolife.select(items, table, lines, 1, per_age=2, speech=2, days=2)
    assert [i["id"] for i in two] == ["1", "18", "3", "6", "4", "20", "21", "5", "8", "9"] and all(i["pilot"] for i in two)
    assert egolife.select(items, table, lines, 1, per_age=2, speech=2, days=2) == two
    with pytest.raises(ValueError, match="2-6h age bin holds 2 questions .* fewer than 3"):
        egolife.select(items, table, lines, 1, per_age=3, speech=1, days=2)
    with pytest.raises(ValueError, match="speech pool holds 2 questions, fewer than 3"):
        egolife.select(items, table, lines, 1, per_age=1, speech=3, days=2)


def test_a_bin_is_taken_from_its_categories_in_turn_by_seed(table):
    queues = egolife.by_category([*synthetic("0-2h", "RelationMap", 5), *synthetic("0-2h", "EventRecall", 2),
                                  *synthetic("0-2h", "EntityLog", 3)])
    assert [q[0]["category"] for q in queues] == ["EventRecall", "EntityLog", "RelationMap"], "the rarest first"
    assert [i["category"] for i in egolife.rotate(queues, 7)] == ["EventRecall", "EntityLog", "RelationMap", "EventRecall", "EntityLog",
                                                                  "RelationMap", "EntityLog"], "in turn, a short queue skipped"
    pool = [i for b in egolife.AGE_NAMES for c, n in (("RelationMap", 12), ("EventRecall", 3), ("EntityLog", 8)) for i in synthetic(b, c, n)]
    chosen = {seed: egolife.select(pool, table, [], seed, per_age=6, speech=0, days=2) for seed in (1, 1, 2, 3)}
    assert chosen[1] == egolife.select(pool, table, [], 1, per_age=6, speech=0, days=2), "the same seed, the same questions"
    assert any([i["id"] for i in chosen[s]] != [i["id"] for i in chosen[1]] for s in (2, 3)), "another seed, others"
    for items in chosen.values():
        for name in egolife.AGE_NAMES:
            of = [i for i in items if i["age"] == name]
            assert [i["category"] for i in of] == ["EventRecall", "EntityLog", "RelationMap"] * 2 and len(of) == 6
            assert [i["pilot"] for i in of] == [True, True, False, False, False, False], "the first two of a bin are the pilot's"


def test_a_chosen_question_with_no_oracle_or_a_leaking_one_is_refused_by_its_id(source, table):
    days = DAYS
    without = {k: v for k, v in ORACLES.items() if k != "5"}
    with pytest.raises(ValueError, match="no oracle for question 5"):
        egolife.plan(source, 1, days, table, without, 2, 2)
    base = {"source": egolife.VIDEO, "question": SHADE}
    for change, message in (({"source": egolife.TEXT}, "source 'transcript'"), ({"question": "Is someone there"}, "not a question"),
                            ({"question": "Is someone working tonight?"}, "time word"),
                            ({"question": "Is someone cooking or cleaning?"}, "'and' or 'or'"),
                            ({"question": "Is Alice using a tool?"}, "names Alice"),
                            ({"question": "Is a tool in view?", "source": "screen"}, "source 'screen'")):
        with pytest.raises(ValueError, match=rf"oracle of question 4 .*{message}"):
            egolife.plan(source, 1, days, table, {**ORACLES, "4": {**base, **change}}, 2, 2)
    with pytest.raises(ValueError, match="oracle of question 9 .*source 'glasses'"):
        egolife.plan(source, 1, days, table, {**ORACLES, "9": base}, 2, 2)
    with pytest.raises(ValueError, match=r"oracle of question 9 holds an option's words \('Alice'\)"):
        egolife.plan(source, 1, days, table, {**ORACLES, "9": {"source": egolife.TEXT, "question": "Does the text mention Alice?"}}, 2, 2)
    made = egolife.plan(source, 1, days, table, ORACLES, 2, 2)
    assert made == egolife.plan(source, 1, days, table, ORACLES, 2, 2) and made.scale == "egolife"
    assert json.loads(json.dumps(made.questions)) == list(made.questions)


def test_the_plan_names_the_clips_neutrally_and_holds_the_transcript(source, table):
    made = egolife.plan(source, 1, DAYS, table, ORACLES, 2, 2)
    assert len(made.files) == 24 and made.files[0].path == "glasses/2026-10-29/11.09.42.mp4"
    assert made.files[0].name == "2026-10-29 11.09.42.mp4" and made.files[0].kind == "clip" and made.files[0].source == "glasses"
    assert made.files[0].t == ep(1, "11:09:42.08") and made.files[0].spec["src"] == table[0].rel
    assert not any("A1_JAKE" in f.path or "DAY" in f.path or "A1_JAKE" in f.name for f in made.files)
    assert sorted(made.logs) == ["2026-10-29", "2026-10-30"] and sum(len(v) for v in made.logs.values()) == 13
    q = next(q for q in made.questions if q["qid"] == "ego-001")
    assert (q["type"], q["template"], q["level"], q["source"], q["day"], q["absent"]) == ("video", "EntityLog", "0-2h", "glasses", 1, False)
    assert q["grade"] == {"kind": "choice", "truth": "B", "ignore": []} and q["options"] == OPTIONS
    assert q["truth"] == {"moments": [[ep(1, "11:10:45.00")] * 2], "others": []} and (q["oracle"], q["oracle_source"]) == (SHADE, "glasses")
    assert next(q for q in made.questions if q["qid"] == "ego-008")["source"] == "transcript"


# -- 4. the build -----------------------------------------------------------------------------------------------------------------

def test_a_run_is_built_with_neutral_frozen_data(built, source, table):
    names = view_files(built.data)
    assert len(names) == 26 and not any(re.search(r"A1_JAKE|DAY\d|egolife", n, re.I) for n in names)
    assert {n.split("/")[0] for n in names} == {"glasses", "transcript"}
    first = built.data / "glasses" / "2026-10-29" / "11.09.42.mp4"
    assert os.path.samefile(first, source / table[0].rel), "the data is the download's clips, linked"
    assert not any(os.access(built.data / n, os.W_OK) for n in names) and bench.changed(built.data, built.manifest) is None
    assert sorted(bench.read_json(built.manifest)) == names
    rows = [json.loads(line) for line in (built.data / "transcript" / "2026-10-29.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10 and all(set(r) == {"start", "end", "speaker", "en", "zh"} for r in rows)
    assert rows[0]["start"] == "2026-10-29T11:09:47.100" and rows[0]["speaker"] == "Jake" and rows[0]["zh"] == "好然后一个秒表"
    assert [r["end"] for r in rows] == sorted(r["end"] for r in rows)


def test_the_snapshot_holds_each_clip_and_each_line_at_its_time(built):
    plan = built.plan()
    with closing(sqlite3.connect(built.snapshot() / "index.sqlite")) as db:
        items = db.execute("SELECT name, t0, t1, state, kind FROM items WHERE source = 'glasses' ORDER BY t0").fetchall()
        records = [t for (t,) in db.execute("SELECT t FROM records WHERE source = 'transcript' ORDER BY t")]
    assert [(n, t0, t1) for n, t0, t1, *_ in items] == [(f.name, f.t, f.t + f.spec["length"]) for f in plan.files]
    assert {(state, kind) for *_, state, kind in items} == {("ok", "video")}
    assert records == sorted(datetime.fromisoformat(r["end"]).timestamp() for rows in plan.logs.values() for r in rows)
    assert not (built.snapshot() / "daemon.json").exists()
    assert os.path.exists(built.bin / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"))


def test_run_json_holds_what_a_run_needs(built):
    meta = bench.read_json(built.folder / "run.json")
    assert (meta["dataset"], meta["generator"], meta["seed"], meta["scale"]) == ("egolife", egolife.VERSION, 1, "egolife")
    assert meta["days"] == ["2026-10-29", "2026-10-30"] and meta["data"] == str(built.data) and meta["tag"] == built.tag
    assert meta["plan"] == bench.plan_hash(built.plan()) and meta["files"] == bench.files_hash(built.plan())
    assert meta["timeout_s"] == 900.0 and 0 < meta["decode_s"] < 1 and meta["probe"] == {"status": "none"} and meta["mms_commit"]
    assert (meta["per_age"], meta["speech"]) == (2, 2) and meta["source"] == str(built.source)
    assert meta["pools"] == {"video": 12, "speech": 2, "dropped": {
        "its query is after the last day": 1, "its time is a range or a list in one string": 1,
        "a target is outside the clips (a gap) or after the horizon": 1, "speech: no transcript line near a target": 1}}
    assert egolife.EgoRun.open(built.folder) == built and built.name.endswith("-egolife-s1")
    assert (built.folder / "clips.json").exists()


def test_a_run_is_refused_when_its_questions_changed(built, tmp_path, monkeypatch):
    changed = tmp_path / "oracles.json"
    changed.write_text(json.dumps({"version": egolife.ORACLE_VERSION, "items": {
        **ORACLES, "1": {"source": "glasses", "question": "Is a red frame in view?"}}}), encoding="utf-8")
    monkeypatch.setattr(egolife, "ORACLES", changed)
    with pytest.raises(ValueError, match="questions are not those its run.json recorded"):
        egolife.EgoRun.open(built.folder)
    monkeypatch.undo()
    monkeypatch.setattr(egolife, "VERSION", egolife.VERSION + 1)
    with pytest.raises(ValueError, match="made by egolife version 1"):
        egolife.EgoRun.open(built.folder)
    monkeypatch.undo()
    other = tmp_path / "haystack"
    other.mkdir()
    (other / "run.json").write_text(json.dumps({"generator": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="is a haystack run: use benchmarks/bench.py"):
        egolife.EgoRun.open(other)
    with pytest.raises(ValueError, match="is an egolife run: use benchmarks/egolife.py"):
        bench.Run.open(built.folder)


# -- 5. the mms horizon ------------------------------------------------------------------------------------------------------------

@pytest.fixture
def mms(built, tmp_path):
    """A daemon with the fake decider over a copy of the snapshot."""
    home = bench.copy_home(built.snapshot(), tmp_path / "home", FAKE_MODEL)
    daemon = Daemon(home).start()
    yield home, daemon.api
    daemon.stop()


def listed(api, types: list[str], **extra) -> list[dict]:
    return api.call("find", types=types, since=ep(1, "00:00:00.00"), until=ep(2, "23:59:00.00"), limit=200, scope="all", **extra)["results"]


def scan(api, **params) -> dict:
    return bench.ask(api, {"question": SHADE, "sources": ["glasses"], "since": ep(1, "11:00:00.00"), "until": ep(2, "23:00:00.00"),
                           "limit": 20, "candidates": 5, **params}, "id:bench-test")[0]


def test_set_horizon_hides_what_ended_after_it(built, mms):
    home, api = mms
    h = clip_ends(built)[1]  # the end of the second clip: the evidence clip starts there
    reference = scan(api, until=ep(1, "11:10:30.00"))  # the clips overlap by their last 0.06 s: a window of the next one is before H
    assert scan(api)["sightings"], "the red frame of the evidence clip is found while nothing is cut"
    evidence = next(r["id"] for r in listed(api, ["item"]) if r["name"] == "2026-10-29 11.10.30.mp4")
    assert len(listed(api, ["record"])) == 13
    assert egolife.set_horizon(home, h) == {"clips": 2, "clips_hidden": 22, "records": 2, "records_hidden": 11}
    # find shows no item and no record after H
    items, records = listed(api, ["item"]), listed(api, ["record"])
    assert len(items) == 2 and all(config.parse_when(r["t1"]) <= h + 0.001 for r in items)
    assert len(records) == 2 and all(config.parse_when(r["t0"]) <= h for r in records)
    # an ask over all time reads what an ask that ended at H read before the cut, and finds nothing of the evidence clip
    cut = scan(api)
    assert reference["sightings"] == cut["sightings"] == [] and cut["progress"]["windows"] > 0
    assert (cut["progress"]["windows"], cut["candidates"]) == (reference["progress"]["windows"], reference["candidates"])
    # a later item is not found by look, nor a later record by its source and time
    assert api.failure("look", item=evidence).code == "not_found"
    assert api.failure("look", source="transcript", t=ep(1, "11:12:36.00")).code == "not_found"
    shown = api.call("look", source="transcript", t=ep(1, "11:09:55.00"))
    assert "Okay, then we need a stopwatch" in shown["text"] and "screwdriver" not in shown["text"]


def test_moving_the_horizon_back_and_forward_restores_everything(built, mms):
    home, api = mms
    before = listed(api, ["record"])
    assert api.call("find", text="screwdriver", types=["record"])["results"]
    egolife.set_horizon(home, ep(1, "11:09:00.00"))
    assert listed(api, ["record"]) == [] and api.call("find", text="screwdriver", types=["record"])["results"] == []
    egolife.set_horizon(home, ep(1, "11:10:49.00"))
    assert len(listed(api, ["record"])) == 3, "the lines that ended by then: the screwdriver one ends at 11:10:47, the next at 11:10:49.5"
    assert api.call("find", text="screwdriver", types=["record"])["results"]
    egolife.set_horizon(home, ep(2, "23:59:00.00"))
    assert listed(api, ["record"]) == before, "the same rows, with the same ids"
    assert api.call("find", text="screwdriver", types=["record"])["results"], "and the search index finds a shown word again"
    with closing(sqlite3.connect(home / "index.sqlite")) as db:
        assert db.execute("SELECT COUNT(*) FROM bench_hidden_records").fetchone() == (0,)
        assert db.execute("SELECT COUNT(*) FROM items WHERE state = 'ok'").fetchone() == (24,)


# -- 6. the files arm's view and its deny list ------------------------------------------------------------------------------------

def lines_by(run: egolife.EgoRun, h: float) -> list[dict]:
    return [r for rows in run.plan().logs.values() for r in rows if datetime.fromisoformat(r["end"]).timestamp() <= h]


def test_the_files_view_holds_the_clips_and_lines_that_ended_by_h(built):
    ends, clips = clip_ends(built), built.clips()
    h = ends[4]
    view = egolife.files_view(built, 0, h)
    assert view == built.view_dir(0) and view.parent == built.temp
    expect = sorted(egolife.clip_path(built.days, c) for c, end in zip(clips, ends) if end <= h)
    got = [n for n in view_files(view) if n.startswith("glasses/")]
    assert got == expect and len(got) == 5
    assert all(os.path.samefile(view / n, built.data / n) for n in got), "the clips are the data's, linked"
    kept = (view / "transcript" / "2026-10-29.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in kept] == [r for r in lines_by(built, h) if r["end"].startswith("2026-10-29")]
    assert len(kept) == 5 and not (view / "transcript" / "2026-10-30.jsonl").exists(), "a later day has no file, not an empty one"
    assert not os.access(view / "transcript" / "2026-10-29.jsonl", os.W_OK)
    # a second, earlier H removes the later files; another slot's view is its own
    egolife.files_view(built, 1, ends[8])
    egolife.files_view(built, 0, ends[1])
    assert [n for n in view_files(view) if n.startswith("glasses/")] == expect[:2]
    assert len((view / "transcript" / "2026-10-29.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len([n for n in view_files(built.view_dir(1)) if n.startswith("glasses/")]) == 9
    # deleting the links left the data as it was: read-only, unchanged
    assert not any(os.access(built.data / n, os.W_OK) for n in view_files(built.data)) and bench.changed(built.data, built.manifest) is None
    egolife.release_views(built)
    assert not built.view_dir(0).exists() and not built.view_dir(1).exists()


def test_the_deny_list_keeps_an_agent_off_the_full_data_and_other_slots(built, tmp_path):
    deny = built.deny(built.slot(0))
    own, other = built.view_dir(0), built.view_dir(1)

    def reached(text: str, output: bool = False) -> bool:
        return codex_run.reached(text, deny, output) is not None

    clip = "glasses/2026-10-29/11.09.42.mp4"
    assert not reached(f"ffmpeg -i {own}/{clip} frame.png") and not reached(f"Input #0, from '{own}\\glasses\\x.mp4'", output=True)
    assert reached(f"ffmpeg -i {other}/{clip} frame.png") and reached(f"ffmpeg -i {built.data}/{clip} frame.png")
    assert reached(r"copy C:\bench-data\egolife\A1_JAKE\DAY2\DAY2_A1_JAKE_10050125.mp4 .")
    assert reached(f"cd ..\\{other.name}") and reached("dir C:\\mmsb-temp\\*-d1") and reached(f"dir {built.temp}\\mmsb-?-d2")
    assert reached(f"Input #0, from '{other}\\glasses\\x.mp4'", output=True), "ffmpeg echoes the path it reads"
    assert reached("Directory of C:\\x\\A1_JAKE\\DAY1", output=True) and reached(f"{other.name}/transcript/2026-10-29.jsonl", output=True)
    assert not reached(f"{other.name}\n{built.slot(1).name}", output=True), "a listing of the temp folder may name a view"
    assert reached(str(built.folder)) and reached(str(built.slot(1)))
    # the dataset's name is refused in a command and not in an output: a transcript line says "EGOlife" and an exec may print it
    assert reached("cd C:\\egolife-folder") and not reached("EGOlife都是过日子的人: Everyone here is living their lives.", output=True)
    assert deny == built.deny(built.slot(0)) and built.deny(built.slot(1)) != deny
    built.check_paths()
    with pytest.raises(ValueError, match="holds 'egolife'"):
        dataclasses.replace(built, temp=tmp_path / "egolife-temp").check_paths()


# -- 7. the decider tier ----------------------------------------------------------------------------------------------------------

def test_oracle_ranges_are_capped_at_h_and_merged():
    ranges = lambda until, *moments: egolife.oracle_ranges({"until": until, "truth": {"moments": [[m, m] for m in moments]}})  # noqa: E731
    assert ranges(1000.0, 900.0) == [(600.0, 1000.0)]
    assert ranges(1000.0, 500.0, 700.0) == [(200.0, 1000.0)], "overlapping ranges are one"
    assert ranges(3000.0, 400.0, 2000.0) == [(100.0, 700.0), (1700.0, 2300.0)], "far apart ranges are two"


@pytest.fixture(scope="module")
def decided(built):
    bench.run_decider(built, [FAKE_MODEL])
    return {q["qid"]: bench.read_json(built.folder / "results" / "decider" / "fake" / f"{q['qid']}.json") for q in built.plan().questions}


def test_the_decider_finds_the_evidence_with_the_fake(built, decided):
    plan = {q["qid"]: q for q in built.plan().questions}
    one = decided["ego-001"]
    assert set(one) == {"qid", "type", "template", "level", "model", "question", "source", "ranges", "found", "top5", "p", "moments",
                        "moments_found", "away", "windows", "seconds", "wall_s", "stopped", "warnings", "sightings", "candidates"}
    assert (one["found"], one["top5"], one["moments_found"], one["away"], one["moments"]) == (True, True, [0], 0, 1)
    assert one["p"] >= 0.9 and one["windows"] > 0 and one["model"] == "fake" and one["question"] == SHADE and one["source"] == "glasses"
    assert one["ranges"] == [list(r) for r in egolife.oracle_ranges(plan["ego-001"])] and one["ranges"][0][1] == plan["ego-001"]["until"]
    for qid in ("ego-003", "ego-004", "ego-005", "ego-020"):
        assert decided[qid]["found"] is True and decided[qid]["away"] == 0, qid
    # a time_list asks two ranges and finds the first of its two targets only; the second range is cut at H
    six = decided["ego-006"]
    assert (six["found"], six["moments_found"], len(six["ranges"])) == (True, [0], 2)
    assert six["ranges"][1][1] == plan["ego-006"]["until"] and six["ranges"][0][1] < six["ranges"][1][0]
    # nothing to find: the evidence clip has not ended by H, or is hours from the target
    for qid in ("ego-018", "ego-021"):
        assert (decided[qid]["found"], decided[qid]["moments_found"], decided[qid]["away"]) == (False, [], 0), qid
    assert all(r["stopped"] == "range_end" for r in decided.values()), "every range was scanned to its end"


def test_the_decider_reads_the_transcript_for_a_speech_question(built, decided):
    eight, nine = decided["ego-008"], decided["ego-009"]
    assert (eight["source"], eight["question"], eight["type"]) == ("transcript", WORD, "records")
    assert (eight["found"], eight["moments_found"]) == (True, [0]) and eight["p"] >= 0.9
    assert (nine["found"], nine["away"]) == (False, 0), "no line near its target says screwdriver"


def test_the_decider_tier_is_asked_once_and_reported(built, decided, monkeypatch):
    folder = built.folder / "results" / "decider" / "fake"
    assert sorted(p.stem for p in folder.glob("*.json")) == sorted(["warm", *decided]) and bench.read_json(folder / "warm.json")["model"] == "fake"
    monkeypatch.setattr(bench, "Daemon", lambda home: pytest.fail("a model with every record is asked nothing"))
    bench.run_decider(built, [FAKE_MODEL])
    tier = egolife.decider_figures(built)["fake"]
    assert (tier["all"]["asked"], tier["all"]["found"]) == (10, 7) and tier["all"]["top5"] >= 7 and tier["all"]["p50"] >= 7
    assert [(tier["types"][t]["asked"], tier["types"][t]["found"]) for t in ("video", "speech")] == [(8, 6), (2, 1)]
    assert tier["categories"]["EventRecall"]["asked"] == 3 and tier["first_s"] is not None
    text = "\n".join(egolife.decider_lines({"fake": tier}))
    assert "| fake | all | 10 |" in text and "| fake | speech | 2 |" in text and "no fault" in text


# -- 8. the agents, with a stub ---------------------------------------------------------------------------------------------------

VIDEO_QIDS = ("ego-001", "ego-003", "ego-004", "ego-005", "ego-006", "ego-018", "ego-020", "ego-021")
SPEECH_QIDS = ("ego-008", "ego-009")
RIGHT = {"blind": {"ego-001", "ego-003"}, "files": {"ego-001", "ego-003", "ego-004", "ego-005", "ego-008"},
         "mms": {*VIDEO_QIDS, *SPEECH_QIDS} - {"ego-021"}}  # who answers which question right


class Stub:
    """An agent that answers right for the questions in `RIGHT[arm]` (as `B` or `B) Alice`) and wrong for the rest (blind says an
    option's words for ego-018: no letter). It records what each exec was given; an mms exec asks over all time, and with `leak` a
    question's result also holds a recording past its horizon."""

    def __init__(self, *, pause: float = 0.0, leak: str | None = None):
        self.pause, self.leak = pause, leak
        self.seen: dict[tuple[str, str], dict] = {}
        self.log: list[tuple] = []
        self.lock = threading.Lock()

    def __call__(self, e: bench.Exec) -> dict:
        q, began = e.question, time.monotonic()
        seen = {"data": e.data, "home": e.home, "prompt": e.prompt, "work": e.work}
        tools = []
        if e.arm == "files":
            seen["files"] = view_files(e.data)
            seen["lines"] = [line for f in sorted((e.data / "transcript").glob("*.jsonl")) for line in f.read_text(encoding="utf-8").splitlines()]
        if e.arm == "mms":
            with closing(sqlite3.connect(e.home / "index.sqlite")) as db:
                seen["clips"] = db.execute("SELECT COUNT(*) FROM items WHERE source = 'glasses' AND state = 'ok'").fetchone()[0]
                seen["records"] = db.execute("SELECT COUNT(*) FROM records WHERE source = 'transcript'").fetchone()[0]
            far = ep(2, "23:00:00.00")
            until = q["until"] if q["qid"] == "ego-001" else far
            params = {"question": SHADE, "sources": ["glasses"], "since": q["since"], "until": until, "limit": 20, "candidates": 5}
            result, _ = bench.ask(Api(e.home), params, bench.exec_project(q))
            seen["ask"] = result
            if q["qid"] == self.leak:
                late = {"source": "glasses", "t0": config.iso(q["until"] + 60), "t1": config.iso(q["until"] + 61), "p": 0.9}
                result = {**result, "sightings": [*result["sightings"], late]}
            tools = [{"tool": "ask", "arguments": {**params, "until": config.iso(until)},
                      "result": {"content": [{"type": "text", "text": json.dumps({**result, "job": "j1"})}]}}]
        time.sleep(self.pause)
        truth = q["grade"]["truth"]
        if q["qid"] in RIGHT[e.arm]:
            answer = truth if q["qid"] < "ego-005" else f"{truth}) {q['options'][truth]}"
        else:
            answer = OPTIONS["A"] if (e.arm, q["qid"]) == ("blind", "ego-018") else next(x for x in "ABCD" if x != truth)
        with self.lock:
            self.seen[e.arm, q["qid"]] = seen
            self.log.append((e.arm, q["qid"], began, time.monotonic()))
        return {"answer": answer, "final": f"ANSWER: {answer}", "tools": tools, "seconds": 1.5, "problem": None, "turns": 2,
                "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1, "used_percent": 10.0,
                "model": "stub", "denials": 0}


@pytest.fixture(scope="module")
def ran(built):
    stub = Stub(pause=0.3)
    bench.run_agents(built, {"claude": stub}, [FAKE_MODEL], jobs=3)
    return stub


def result(run, arm: str, qid: str) -> dict:
    return bench.read_json(bench.result_path(run, "claude", arm, FAKE_MODEL if arm == "mms" else None, qid))


def test_each_arm_is_given_what_it_reads_cut_to_the_horizon(built, ran):
    plan = built.plan()
    ends = clip_ends(built)
    assert sorted(ran.seen) == sorted((arm, q["qid"]) for arm in bench.ARMS for q in plan.questions)
    for q in plan.questions:
        h = q["until"]
        clips, lines = sum(1 for end in ends if end <= h), lines_by(built, h)
        blind, files, mms = (ran.seen[arm, q["qid"]] for arm in ("blind", "files", "mms"))
        assert blind["data"] is None and blind["home"] is None and mms["data"] is None and files["home"] is None
        slot = next(k for k in range(4) if built.slot(k) == files["work"])
        assert files["data"] == built.view_dir(slot)
        assert len([n for n in files["files"] if n.startswith("glasses/")]) == clips
        assert [json.loads(line) for line in files["lines"]] == lines
        assert (mms["clips"], mms["records"]) == (clips, len(lines)), "the mms home was cut before the exec"
        assert mms["home"] == built.folder / "homes" / "claude-fake"
    assert not built.view_dir(0).exists(), "the slots' folders of recordings are removed at the end"
    # an mms exec that asks over all time reads no window past H: ego-018's evidence clip is past it
    assert ran.seen["mms", "ego-018"]["ask"]["sightings"] == [] and ran.seen["mms", "ego-018"]["ask"]["progress"]["windows"] == 2, \
        "a window for each of the two clips before H (a still clip is one run of frames)"
    assert ran.seen["mms", "ego-001"]["ask"]["sightings"], "and the same ask finds it for the question that has it by H"


def test_the_arms_differ_only_in_how_the_recordings_are_reached(built, ran):
    first = next(q for q in built.plan().questions if q["qid"] == "ego-001")
    blind, files, mms = (ran.seen[arm, "ego-001"]["prompt"] for arm in ("blind", "files", "mms"))
    head = "It is now Thursday 29 October 2026, 11:12 (day 1 of these recordings; day 1 was Thursday 29 October). I am Jake."
    tail = ("Question 1?\nA) Tasha\nB) Alice\nC) Shure\nD) Lucia\n\nIf what you can find does not settle it, give your best guess. "
            "End with one line ANSWER: <A, B, C or D>.")
    for text in (blind, files, mms):
        assert text.startswith(head) and text.endswith(tail), text
    assert "You cannot see my recordings." in blind and str(ran.seen["files", "ego-001"]["data"]) in files
    assert "glasses/<date>/" in files and "ffmpeg is on PATH (there is no ffprobe)" in files
    assert "mms has recorded these sources of mine from day 1 to now: glasses" in mms and "give since and until as dates and times" in mms
    assert not any(re.search(r"A1_JAKE|egolife|EgoLife|DAY\d", t) for t in (blind, files, mms))
    day2 = built.prompt(next(q for q in built.plan().questions if q["qid"] == "ego-005"), "blind", None)
    assert day2.startswith("It is now Friday 30 October 2026, 15:21 (day 2 of these recordings; day 1 was Thursday 29 October)")
    with pytest.raises(ValueError, match="arm is one of blind, files, mms"):
        built.prompt(first, "screen", None)


def test_results_are_graded_by_letter_and_audited(built, ran):
    for arm in bench.ARMS:
        for q in built.plan().questions:
            r = result(built, arm, q["qid"])
            assert r["passed"] is (q["qid"] in RIGHT[arm]) and r["problem"] is None and r["timeout_s"] == 900.0, (arm, q["qid"])
            assert r["arm"] == arm and r["level"] == q["level"] and r["template"] == q["template"] and r["absent"] is False
            assert ("after_horizon" in r) == (arm == "mms")
    mms = {q["qid"]: result(built, "mms", q["qid"]) for q in built.plan().questions}
    assert all((r["after_horizon"], r["asks"]) == (0, 1) for r in mms.values())
    assert {qid: r["asks_past_horizon"] for qid, r in mms.items()} == {qid: int(qid != "ego-001") for qid in mms}, "only ego-001 ended its ask at H"
    assert result(built, "blind", "ego-018")["abstained"] is False and result(built, "blind", "ego-018")["answer"] == "Tasha"


def test_mms_runs_one_at_a_time_and_the_others_in_parallel(ran):
    def overlap(arm: str) -> bool:
        spans = sorted((a, b) for x, _, a, b in ran.log if x == arm)
        return any(b > c for (_, b), (c, _) in zip(spans, spans[1:]))

    assert overlap("blind") and overlap("files") and not overlap("mms"), "jobs=3 for blind and files, one at a time for mms"


def test_a_second_command_asks_nothing_and_a_leak_ends_the_command(built, ran, tmp_path):
    again = Stub()
    bench.run_agents(built, {"claude": again}, [FAKE_MODEL], jobs=3)
    assert again.log == []

    def skip(where, names):
        return [n for n in names if n in ("results", "execs", "data", "models.json") or (os.path.basename(where) == "homes" and n != "snapshot")]

    shutil.copytree(built.folder, tmp_path / "run", ignore=skip)
    run = egolife.EgoRun.open(tmp_path / "run")
    with pytest.raises(egolife.HorizonLeak, match="ego-018: 2 recording times after the horizon"):
        bench.run_agents(run, {"claude": Stub(leak="ego-018")}, [FAKE_MODEL], jobs=1, only=["ego-018"])
    assert bench.result_path(run, "claude", "blind", None, "ego-018").exists() and bench.result_path(run, "claude", "files", None, "ego-018").exists()
    assert not bench.result_path(run, "claude", "mms", FAKE_MODEL, "ego-018").exists(), "the question stays to run again"
    assert not (run.snapshot() / "daemon.json").exists()


def test_the_subset_and_only_choose_questions():
    qs = [{"qid": f"ego-{n:03d}", "pilot": n < 3} for n in range(6)]
    plan, run = types.SimpleNamespace(questions=qs), egolife.EgoRun(Path("run"), 1, "egolife", DAYS, Path("data"), Path("temp"))
    chosen = lambda *args: [q["qid"] for q in run.questions(plan, *args)]  # noqa: E731
    assert chosen("all", []) == [q["qid"] for q in qs] and chosen("pilot", []) == ["ego-000", "ego-001", "ego-002"]
    assert chosen("pilot", [], ["ego-001"]) == ["ego-001"]
    with pytest.raises(ValueError, match="--only names questions that are not in the pilot questions: ego-004"):
        chosen("pilot", [], ["ego-004"])
    with pytest.raises(ValueError, match="a subset is one of all, pilot"):
        chosen("some", [])
    with pytest.raises(ValueError, match="jobs is 1 to 4"):
        bench.run_agents(run, {}, [], jobs=9)


# -- 9. the report ----------------------------------------------------------------------------------------------------------------

def test_the_report_counts_the_three_arms(built, ran, decided):
    figures = egolife.numbers(built)
    m = figures["agents"]["claude"]["fake"]
    assert (m["questions"], m["paired"]) == (10, 10) and m["video"]["n"] == 8 and m["speech"]["n"] == 2
    video = m["video"]
    assert [video["arms"][a]["correct"] for a in bench.ARMS] == [2, 4, 7]
    assert video["arms"]["blind"]["unparsed"] == 1 and video["arms"]["files"]["unparsed"] == 0
    assert [(video[k]["b"], video[k]["c"]) for k in ("mms_files", "files_blind", "mms_blind")] == [(3, 0), (2, 0), (5, 0)]
    assert video["mms_files"]["p"] == bench.mcnemar(3, 0) and video["mms_files"]["diff"] == pytest.approx(3 / 8) and video["mms_files"]["ci"] is None
    assert video["mms_files"]["base"]["correct"] == 4 and video["mms_files"]["other"]["correct"] == 7
    speech = m["speech"]
    assert [speech["arms"][a]["correct"] for a in bench.ARMS] == [0, 1, 2] and (speech["mms_files"]["b"], speech["mms_files"]["c"]) == (1, 0)
    assert [(a, g["n"]) for a, g in m["ages"].items()] == [(a, 2) for a in egolife.AGE_NAMES]
    assert m["categories"]["TaskMaster"] == {"n": 1, "blind": 0, "files": 0, "mms": 1}
    assert m["crossover"] is None and m["thin"] == {a: 2 for a in egolife.AGE_NAMES}
    assert (m["work"]["mms"]["ask_execs"], m["work"]["mms"]["n"]) == (8, 8) and m["cost"]["blind"]["correct"] == 2
    h = m["horizon"]
    assert (h["audited"], h["after_horizon"], h["asks"], h["asks_past_horizon"]) == (10, 0, 10, 9) and h["max_s"] > 0
    assert figures["run"]["video"] == [2, 2, 2, 2] and figures["run"]["speech"] == 2 and figures["totals"]["to_rerun"] == 0
    assert figures["decider"]["fake"]["all"]["found"] == 7 and json.loads(json.dumps(figures)) == figures


def test_the_report_text_has_its_tables(built, ran, decided):
    text = egolife.markdown(egolife.numbers(built))
    for line in ("# EgoLifeQA through the mms bench", "8 video questions (2 aged 0-2h, 2 aged 2-6h, 2 aged 6-24h, 2 aged 24h+;",
                 "No shortcut probe: a public dataset, so the blind arm is the floor",
                 "## claude with fake: 8 video questions of 10 (paired: done in all three arms)",
                 "| blind | 2/8 (25%, 7-59) |", "| files | 4/8 (50%,", "| mms | 7/8 (88%,",
                 "| mms vs files | 8 | +38 pts | 3 / 0 | 0.250 |",
                 "| files vs blind | 8 | +25 pts | 2 / 0 | 0.500 |",
                 "| mms vs blind | 8 | +62 pts | 5 / 0 | 0.062 |",
                 "A difference is the second arm less the first", "by age (mms vs files; blind is the floor)", "| 0-2h | 2 |",
                 "The youngest age at which mms leads files: none: at no level with 10 or more paired questions",
                 "too few paired questions at 0-2h (2), 2-6h (2), 6-24h (2), 24h+ (2)", "by category (right / n)",
                 "TaskMaster has few questions in each age", "the 2 speech questions, apart", "Answered from the transcripts alone",
                 "how each arm worked, and what a correct answer cost", "## The decider tier", "| fake | all | 10 |",
                 "Every arm saw the clips that ended by each question's horizon H (median 225 s before the question",
                 "Recording times after H in mms tool results: 0 in 10 audited execs; asks started past H (the agent's confusion about "
                 "mms's clock, harmless): 9/10.", "## Read with these",
                 "EgoLife has been public since 2025: the blind arm measures", "## Appendix: claude with fake, one row per question",
                 "| ego-001 | 0-2h | EntityLog | video | pass 2 s a0 | pass 2 s a0 | pass 2 s a1 |"):
        assert line in text, line
    assert text.count("| ego-0") == 10


def test_a_run_with_no_results_reports_none(built, tmp_path):
    def skip(where, names):
        return [n for n in names if n in ("results", "execs", "data", "models.json") or (os.path.basename(where) == "homes" and n != "snapshot")]

    shutil.copytree(built.folder, tmp_path / "run", ignore=skip)
    run = egolife.EgoRun.open(tmp_path / "run")
    figures = egolife.numbers(run)
    assert figures["agents"] == {} and figures["decider"] == {} and figures["totals"]["results"] == 0
    text = egolife.markdown(figures)
    assert "No agent results yet." in text and "No decider results yet." in text


# -- more: the audit, the estimate, the timeout, the oracles file, the command line -----------------------------------------------

def test_the_audit_counts_recordings_past_the_horizon_in_asks_finds_and_looks(built):
    h = built.plan().questions[0]["until"]
    iso = config.iso
    call = lambda tool, result, **args: {"tool": tool, "arguments": args, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}  # noqa: E731
    ask = call("ask", {"job": "j1", "sightings": [{"t0": iso(h - 5), "t1": iso(h)}], "candidates": [{"t": iso(h - 9)}]}, question="q",
               until=iso(h))
    assert egolife.audit([ask], h, None) == {"after_horizon": 0, "asks": 1, "asks_past_horizon": 0}, "a time at H is not past it"
    late = call("ask", {"job": "j2", "sightings": [{"t0": iso(h + 5), "t1": iso(h + 6)}], "candidates": [{"t": iso(h + 9)}]}, question="q")
    assert egolife.audit([late, call("ask", {"job": "j2"}, job="j2")], h, None) == {"after_horizon": 3, "asks": 1, "asks_past_horizon": 1}
    # an ask with `questions` counts; one mms refused (no job in its result) does not, whatever its until says
    several = call("ask", {"job": "j3"}, questions={"a": {}, "b": {}}, until="2026-10-30T21:30:00")
    refused = {"tool": "ask", "arguments": {"question": "q", "until": "yesterday morning"}, "result": "bad_request: until is a time"}
    assert egolife.audit([refused, several], h, None) == {"after_horizon": 0, "asks": 1, "asks_past_horizon": 1}
    # a find's rows, but not the labels the agent made itself (mms dates them by its clock, days later than the question)
    found = call("find", {"results": [{"type": "item", "t0": iso(h - 1), "t1": iso(h + 1)}, {"type": "record", "t0": iso(h + 30), "t1": None},
                                      {"type": "label", "t0": iso(h + 40), "t1": iso(h + 41)}]})
    assert egolife.audit([found], h, None)["after_horizon"] == 2
    with closing(sqlite3.connect(built.snapshot() / "index.sqlite")) as db:
        later = db.execute("SELECT id FROM items WHERE source = 'glasses' AND t1 > ? ORDER BY t1 LIMIT 1", (h,)).fetchone()[0]
        earlier = db.execute("SELECT id FROM items WHERE source = 'glasses' AND t1 <= ? ORDER BY t1 LIMIT 1", (h,)).fetchone()[0]
    assert egolife.audit([call("look", {"item": later, "kind": "image"})], h, built.snapshot())["after_horizon"] == 1
    assert egolife.audit([call("look", {"moments": [{"item": earlier}, {"error": {}}]})], h, built.snapshot())["after_horizon"] == 0
    # a look of several moments gives a JSON block for each, with a picture before it: all are audited, not the first alone
    blocks = {"content": [{"type": "image"}, {"type": "text", "text": json.dumps({"item": earlier})}, {"type": "image"},
                          {"type": "text", "text": json.dumps({"item": later})}]}
    assert egolife.audit([{"tool": "look", "arguments": {"moments": []}, "result": blocks}], h, built.snapshot())["after_horizon"] == 1
    assert egolife.audit([{"tool": "ask", "arguments": {"question": "q"}, "result": "not json"}], h, None) == \
        {"after_horizon": 0, "asks": 0, "asks_past_horizon": 0}


def test_the_audit_catches_a_sheet_cut_from_an_item_past_the_horizon_and_not_the_times_the_agent_asked_for(built):
    h = built.plan().questions[0]["until"]
    iso = config.iso
    call = lambda tool, result, **args: {"tool": tool, "arguments": args, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}  # noqa: E731
    with closing(sqlite3.connect(built.snapshot() / "index.sqlite")) as db:
        later = db.execute("SELECT id FROM items WHERE source = 'glasses' AND t1 > ? ORDER BY t1 LIMIT 1", (h,)).fetchone()[0]
        earlier = db.execute("SELECT id FROM items WHERE source = 'glasses' AND t1 <= ? ORDER BY t1 LIMIT 1", (h,)).fetchone()[0]
    tile = lambda n, item, **when: {"n": n, **when, "look": {"item": item, "at": 0}}  # noqa: E731
    sheet = lambda since, until, *tiles: {"kind": "sheet", "source": "glasses", "since": iso(since), "until": iso(until), "labels": [],
                                          "image": {}, "tiles": list(tiles)}  # noqa: E731
    leak = lambda result: egolife.audit([call("look", result, source="glasses")], h, built.snapshot())["after_horizon"]  # noqa: E731
    before = sheet(h - 20, h, tile(1, earlier, t=iso(h - 20)), tile(2, earlier, t0=iso(h - 10), t1=iso(h)))
    assert leak(before) == 0, "tiles cut from items that end by the horizon are no leak"
    asked = sheet(h - 20, h + 5000, tile(1, earlier, t=iso(h + 3)), tile(2, earlier, t0=iso(h + 4), t1=iso(h + 6)))
    assert leak(asked) == 0, "a range and tile times past it are what the agent asked for, as an ask's until"
    assert leak(sheet(h - 20, h, tile(1, later, t=iso(h - 20)))) == 1, "a tile cut from an item that ended past it"
    later_moments = {"moments": [before, {"error": {}}, sheet(h - 1, h + 2, tile(1, later, t=iso(h)))]}
    assert leak(later_moments) == 1, "a sheet among moments, and a failed moment"
    assert leak({"kind": "sheet", "tiles": [None, {"n": 1}]}) == 0, "a sheet's fields may be missing"


def test_the_audit_catches_a_time_given_as_t_in_a_sighting_or_a_candidate_and_in_a_check_in_that_leaves_out_the_rest(built):
    h = built.plan().questions[0]["until"]
    iso = config.iso
    call = lambda tool, result, **args: {"tool": tool, "arguments": args, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}  # noqa: E731
    before = {"source": "glasses", "t": iso(h - 5), "p": 0.9, "kind": "video", "look": {"item": 1, "at": 0}}
    spanning = {"source": "glasses", "t0": iso(h + 8), "t1": iso(h + 9), "p": 0.5, "kind": "audio", "look": {"item": 2, "at": 0}}
    # a sighting of one instant has only `t`, a candidate that spans has `t0` and `t1`: each is a time the agent saw
    late = call("ask", {"job": "j1", "sightings": [{**before, "t": iso(h + 7)}], "candidates": [spanning]}, question="q", until=iso(h))
    assert egolife.audit([late], h, None) == {"after_horizon": 3, "asks": 1, "asks_past_horizon": 0}
    assert egolife.audit([call("ask", {"job": "j1", "sightings": [before], "candidates": [{**before, "t": iso(h)}]}, question="q", until=iso(h))],
                         h, None)["after_horizon"] == 0, "a `t` at or before the horizon is no leak"
    # a later reply of the same job holds only what is new, and is audited as it is: its own sightings, none when it has none
    check_in = call("ask", {"job": "j1", "state": "running", "progress": {"windows": 3, "sightings": 1}, "sightings": [{**before, "t": iso(h + 11)}]}, job="j1")
    assert egolife.audit([check_in], h, None)["after_horizon"] == 1
    assert egolife.audit([call("ask", {"job": "j1", "state": "running", "progress": {"windows": 3}}, job="j1")], h, None)["after_horizon"] == 0


def test_the_estimate_is_the_decider_ask_at_the_cost_of_a_window_here(built, decided, mms):
    plan = built.plan()
    each = bench.window_s(FAKE_MODEL.about) - egolife.ask_script.DECODE_S + built.decode_s
    video = next(q for q in plan.questions if q["qid"] == "ego-001")
    windows = built.estimate(plan, FAKE_MODEL, video) / each
    _, api = mms
    read = sum(bench.ask(api, {"question": video["oracle"], "sources": [video["oracle_source"]], "since": a, "until": b,
                               "limit": bench.ORACLE_LIMIT, "candidates": bench.ORACLE_CANDIDATES, "max_s": bench.ORACLE_MAX_S,
                               "still_s": 0}, bench.ORACLE_PROJECT)[0]["progress"]["windows"] for a, b in egolife.oracle_ranges(video))
    assert read * 0.9 <= windows <= read * 1.1, "close to the windows an ask of a frame a second reads"
    assert 0 < decided["ego-001"]["windows"] <= windows, "and the ask with the gate reads no more: these still clips join their frames"
    speech = next(q for q in plan.questions if q["qid"] == "ego-008")
    assert built.estimate(plan, FAKE_MODEL, speech) == pytest.approx(bench.window_s(FAKE_MODEL.about, "records")), "a record window"
    assert built.needle(video) == "ego-001" != built.needle(speech)


def test_the_timeout_is_kept_in_the_run(built, tmp_path):
    def skip(where, names):
        return [n for n in names if n in ("results", "execs", "data", "homes", "models.json")]

    shutil.copytree(built.folder, tmp_path / "run", ignore=skip)
    run = egolife.set_timeout(egolife.EgoRun.open(tmp_path / "run"), 1200.0)
    assert run.timeout_s == 1200.0 and egolife.EgoRun.open(run.folder).timeout_s == 1200.0
    assert bench.read_json(run.folder / "run.json")["timeout_s"] == 1200.0 and built.timeout_s == 900.0


def test_the_oracles_file_follows_the_rules_a_machine_can_check():
    data = json.loads(Path(egolife.__file__).with_name("egolife_oracles.json").read_text(encoding="utf-8"))
    assert data["version"] == egolife.ORACLE_VERSION and len(data["items"]) == 50
    assert Counter(o["source"] for o in data["items"].values()) == {"glasses": 40, "transcript": 10}
    for id_, o in data["items"].items():
        assert set(o) == {"source", "question"} and o["question"].endswith("?") and o["question"].count("?") == 1, id_
        assert not egolife.ask_script.TIME_WORDS.search(o["question"]) and not egolife.ask_script.JOINED.search(o["question"]), id_
        assert not re.search(r"\b(now|just|last|first|usually|always|often|ago)\b", o["question"], re.I), f"{id_}: a time word"
        if o["source"] == "glasses":
            assert not any(re.search(rf"\b{p}\b", o["question"], re.I) for p in egolife.PEOPLE), f"{id_}: names a person"
            assert re.match(r"(Is|Are) ", o["question"]), id_
        else:
            assert o["question"].startswith("Does the text talk about "), id_


def test_the_command_line_needs_the_big_folders_named():
    parser = egolife.parser_of()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--source", "x"])
    args = parser.parse_args(["build", "--source", "x", "--out", "y", "--temp", "z"])
    assert (args.seed, args.out, args.temp) == (1, Path("y"), Path("z"))
    args = parser.parse_args(["agents", "run", "--subset", "pilot", "--timeout", "1200"])
    assert (args.models, args.agents, args.jobs, args.subset, args.timeout) == ("d1-3B", "claude,codex", bench.JOBS, "pilot", 1200.0)


def test_the_commands_say_what_they_will_run_and_report_writes_its_files(built, ran, decided, tmp_path, capsys):
    def skip(where, names):
        return [n for n in names if n in ("results", "execs", "data", "models.json", "report.md", "report.json")
                or (os.path.basename(where) == "homes" and n != "snapshot")]

    shutil.copytree(built.folder, tmp_path / "run", ignore=skip)
    fresh = egolife.EgoRun.open(tmp_path / "run")
    bench.show_agents(fresh, ["claude", "codex"], [FAKE_MODEL], [], 3, None, "pilot", ["ego-001"])
    out = capsys.readouterr().out
    assert "egolife, seed 1, 2 days, the pilot questions, only ego-001" in out and out.count("1 to run of 1") == 6
    assert re.search(r"claude\s+blind\s+1 to run of 1", out) and re.search(r"codex\s+mms fake\s+1 to run of 1", out)
    assert "6 execs to run" in out and "fake: this run's mms execs go one at a time" in out
    bench.show_agents(built, ["claude"], [FAKE_MODEL], [], 3)
    assert capsys.readouterr().out.count("0 to run of 10") == 3
    bench.show_decider(fresh, [FAKE_MODEL], [])
    assert re.search(r"10 oracle asks to run of 10, (about \d+ min|under a minute)", capsys.readouterr().out)
    assert egolife.main(["report", str(built.folder)]) == 0 and "# EgoLifeQA through the mms bench" in capsys.readouterr().out
    assert bench.read_json(built.folder / "report.json")["agents"]["claude"]["fake"]["paired"] == 10
    assert "## The decider tier" in (built.folder / "report.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        egolife.main(["agents", str(built.folder), "--agents", "gemini", "--models", "none"])
    assert egolife.main(["agents", str(tmp_path / "nowhere"), "--models", "none"]) == 1, "no run.json: an error line and exit 1"


# -- the review's cases -----------------------------------------------------------------------------------------------------------

def copy_run(built: egolife.EgoRun, tmp_path: Path, name: str = "copy") -> egolife.EgoRun:
    """A copy of the built run with no data, results, execs or homes but the snapshot (the data and the snapshot's media are shared
    with `built`: only read)."""
    def skip(where, names):
        return [n for n in names if n in ("results", "execs", "data", "models.json", "report.md", "report.json")
                or (os.path.basename(where) == "homes" and n != "snapshot")]

    shutil.copytree(built.folder, tmp_path / name, ignore=skip)
    return egolife.EgoRun.open(tmp_path / name)


def test_the_default_days_cross_no_change_of_the_clocks_and_build_refuses_days_that_do(source, tmp_path, monkeypatch):
    def offset(day: date) -> timedelta:  # the clocks go back during 1 November: the offset at its start is still summer's
        return timedelta(hours=-4) if day <= date(2026, 11, 1) else timedelta(hours=-5)

    days = lambda today: egolife.default_days(today, offset=offset)  # noqa: E731
    assert days(date(2026, 11, 20)) == (date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)), "the latest three before today"
    assert days(date(2026, 11, 5)) == (date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4))
    assert days(date(2026, 11, 3)) == (date(2026, 10, 29), date(2026, 10, 30), date(2026, 10, 31)), "the nights of 31 Oct to 2 Nov cross it"
    assert days(date(2026, 10, 20)) == (date(2026, 10, 17), date(2026, 10, 18), date(2026, 10, 19))
    assert egolife.default_days(date(2026, 11, 20)) and egolife.default_days(date(2026, 11, 20), n=1) == (date(2026, 11, 19),)
    for ok in ((date(2026, 10, 29), date(2026, 10, 30)), (date(2026, 11, 2), date(2026, 11, 3))):
        egolife.check_days(ok, offset)
    for crossing in ((date(2026, 10, 31), date(2026, 11, 1)), (date(2026, 11, 1),), (date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 1))):
        with pytest.raises(ValueError, match="cross a change of the local UTC offset"):
            egolife.check_days(crossing, offset)
    with pytest.raises(ValueError, match="consecutive, oldest first"):
        egolife.check_days((date(2026, 10, 30), date(2026, 10, 29)), offset)
    with pytest.raises(ValueError, match="consecutive, oldest first"):
        egolife.check_days((date(2026, 10, 29), date(2026, 10, 31)), offset)
    monkeypatch.setattr(egolife, "utc_offset", offset)
    with pytest.raises(ValueError, match="cross a change of the local UTC offset"):
        egolife.build(tmp_path / "out", source, 1, temp=tmp_path / "temp", days=(date(2026, 10, 31), date(2026, 11, 1)), per_age=2, speech=2)
    assert not (tmp_path / "out").exists(), "refused before anything was made"
    monkeypatch.undo()
    assert egolife.utc_offset(date(2026, 7, 1)) is not None and egolife.crosses_a_change((date(2026, 7, 1), date(2026, 7, 2))) is False


def test_a_transcript_line_that_a_files_exec_could_not_print_is_refused_at_build(built):
    made = built.plan()
    egolife.check_transcript(built, made)
    line = {"start": "2026-10-29T11:09:47.100", "end": "2026-10-29T11:09:50.766", "speaker": "Jake", "zh": "", "en": "x"}
    said = {"2026-10-29": ({**line, "zh": "EGOlife都是过日子的人", "en": "Everyone here is living their lives."},)}
    egolife.check_transcript(built, dataclasses.replace(made, logs=said))  # "egolife" is refused in a command only
    for word in ("bench-data", "A1_JAKE", ".claude", str(built.folder)):
        with pytest.raises(ValueError, match="2026-10-29T11:09:50.766 holds"):
            egolife.check_transcript(built, dataclasses.replace(made, logs={"2026-10-29": ({**line, "en": f"look in {word} now"},)}))


def test_an_oracle_may_not_borrow_a_word_of_an_option_that_its_question_lacks(source, table):
    item = {"id": "1", "audio": False, "question": "Who washed the dishes?", "options": {"A": "Wash dishes", "B": "Dry cups", "C": "Sweep", "D": "Me"}}
    ok = {"source": "glasses", "question": "Are dishes being washed?"}
    egolife.check_oracle(item, ok)  # `dishes` is in the question too
    with pytest.raises(ValueError, match="uses 'cups', a word of an option that the question does not use"):
        egolife.check_oracle(item, {**ok, "question": "Are cups in view?"})
    egolife.check_oracle(item, {**ok, "question": "Is someone drying a plate?"})  # words are compared whole, and `dry` is under 4 letters
    spoken_item = {**item, "audio": True, "options": {"A": "About two hours ago", "B": "About a day ago", "C": "x", "D": "y"}, "question": "When was AI mentioned?"}
    egolife.check_oracle(spoken_item, {"source": "transcript", "question": egolife.TEXT_ASK + "AI?"})  # the template's `about` is no borrowing
    with pytest.raises(ValueError, match="uses 'hours'"):
        egolife.check_oracle(spoken_item, {"source": "transcript", "question": egolife.TEXT_ASK + "hours?"})
    # the whole plan still passes with the rule on
    assert egolife.plan(source, 1, DAYS, table, ORACLES, 2, 2)


def line_at(run: egolife.EgoRun, after: bool) -> dict:
    """A transcript line of the run that ended after question ego-001's horizon, or by it."""
    h = next(q for q in run.plan().questions if q["qid"] == "ego-001")["until"]
    return next(r for rows in run.plan().logs.values() for r in rows
                if (datetime.fromisoformat(r["start"]).timestamp() > h) == after and (datetime.fromisoformat(r["end"]).timestamp() > h) == after)


def test_a_files_exec_whose_output_shows_a_time_or_a_clip_past_the_horizon_is_escaped(built, tmp_path):
    h = next(q for q in built.plan().questions if q["qid"] == "ego-001")["until"]
    late, early = line_at(built, True), line_at(built, False)
    seen = egolife.seen_past_horizon([json.dumps(late), json.dumps(early)], built, h)
    assert seen == sorted({late["start"], late["end"]}), "the times of the late line, none of the early one"
    texts = ["glasses/2026-10-29/12.30.30.mp4  2026-11-03T01:20:11.123  2026-10-29T99:99:99.123", "x glasses\\2026-10-29\\12.31.00.mp4 y",
             "glasses/2026-10-29/11.09.42.mp4 glasses/2026-10-29/11.11.30.mp4 glasses/2026-10-29/01.01.01.mp4"]
    assert egolife.seen_past_horizon(texts, built, h) == ["glasses/2026-10-29/12.30.30.mp4", "glasses/2026-10-29/12.31.00.mp4"], \
        "a clip that ended after H, either slash; not the clips in the view, an unknown name, the time now or a time that is none"
    # a path with its backslashes escaped, as a JSON or a Python repr prints it, is a path too
    escaped = "path: 'glasses\\\\2026-10-29\\\\12.31.00.mp4' and glasses\\\\\\\\2026-10-29\\\\\\\\12.30.30.mp4"
    assert egolife.seen_past_horizon([escaped], built, h) == ["glasses/2026-10-29/12.30.30.mp4", "glasses/2026-10-29/12.31.00.mp4"]
    # a time that is no transcript line's is the agent's own (a file's mtime, a computed time), after H or not
    own = f"mtime 2026-10-29T12:59:59.999 now {late['end'][:-3]}999 and {early['end']}"
    assert egolife.seen_past_horizon([own], built, h) == []
    # a clip name in a line that reports a missing file is a guess at the clip of "now": no escape, in any of the usual wordings
    guess = "glasses/2026-10-29/12.30.30.mp4"
    for said in (f"{guess}: No such file or directory", f"FileNotFoundError: [Errno 2] No such file or directory: '{guess}'",
                 f"Get-Content : Cannot find path 'C:\\x\\{guess}' because it does not exist.", f"bash: {guess}: not found",
                 f"Traceback ... FileNotFoundError: {guess}"):
        assert egolife.seen_past_horizon([f"{said}\nexit 1"], built, h) == [], said
        assert egolife.seen_past_horizon([said, guess], built, h) == [guess], "the same name on a line of its own is still read"
    codex = [{"type": "item.completed", "item": {"type": "command_execution", "command": "Get-Content x", "aggregated_output": json.dumps(late)}}]
    claude = [{"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": json.dumps(late)}]},
                                                      {"type": "tool_result", "tool_use_id": "u", "content": json.dumps(early)}]}},
              {"type": "assistant", "message": {"content": [{"type": "text", "text": json.dumps(late)}]}}]
    assert egolife.command_outputs(codex) == [json.dumps(late)] and egolife.command_outputs(claude) == [json.dumps(late), json.dumps(early)]
    q = next(q for q in built.plan().questions if q["qid"] == "ego-001")
    on = lambda arm, **out: built.extra(types.SimpleNamespace(arm=arm, question=q, agent="claude", home=None), {"problem": None, **out})  # noqa: E731
    for events in (codex, claude):
        got = on("files", events=events)
        assert got["seen_past_horizon"] == sorted({late["start"], late["end"]}) and got["passed"] is False
        assert got["problem"].startswith("escaped: its output showed recordings past the horizon: ") and got["timeout_s"] == 900.0
    assert on("files", events=[]) == {"timeout_s": 900.0, "seen_past_horizon": []}
    assert on("files") == {"timeout_s": 900.0, "seen_past_horizon": []}
    only_early = [{"type": "item.completed", "item": {"type": "command_execution", "aggregated_output": json.dumps(early)}}]
    assert on("files", events=only_early) == {"timeout_s": 900.0, "seen_past_horizon": []}
    assert "problem" not in on("files", events=codex, problem="timeout: no answer in 900 s"), "an exec that already failed keeps its problem"
    assert on("blind", events=codex) == {"timeout_s": 900.0}
    # through the loop: the result of the files arm says escaped
    run = copy_run(built, tmp_path)
    leaked = types.SimpleNamespace(calls=0)

    def agent(e: bench.Exec) -> dict:
        leaked.calls += 1
        return {"answer": "B", "final": "ANSWER: B", "tools": [], "events": codex, "seconds": 1.0, "problem": None, "turns": 1, "tokens": None,
                "cost_usd": None, "used_percent": 1.0, "model": "stub", "denials": 0}

    bench.run_agents(run, {"claude": agent}, [], jobs=1, only=["ego-001"])
    r = bench.read_json(bench.result_path(run, "claude", "files", None, "ego-001"))
    assert r["problem"].startswith("escaped: its output showed") and r["passed"] is False and r["seen_past_horizon"]
    assert bench.read_json(bench.result_path(run, "claude", "blind", None, "ego-001"))["problem"] is None


def test_the_report_reads_a_run_with_no_video_question_paired(built, tmp_path):
    run = copy_run(built, tmp_path)
    bench.run_agents(run, {"claude": Stub()}, [FAKE_MODEL], jobs=1, only=["ego-008"])  # a speech question alone
    figures = egolife.numbers(run)
    m = figures["agents"]["claude"]["fake"]
    assert m["video"] is None and m["speech"]["n"] == 1 and m["ages"] == {} and m["paired"] == 1
    text = egolife.markdown(figures)
    assert "## claude with fake: 0 video questions of 10" in text and "None paired yet." in text
    assert "the 1 speech questions, apart" in text and "| ego-008 | 0-2h | TaskMaster | speech |" in text


def test_a_dry_run_names_the_files_folder_and_makes_nothing(built, tmp_path, monkeypatch, capsys):
    run = copy_run(built, tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(bench, "shutil", types.SimpleNamespace(which=lambda name: None))
        patch.setattr(codex_run, "models_of", lambda command: {"models": [
            {"slug": codex_run.model(), "supported_reasoning_levels": [{"effort": codex_run.effort()}]}]})
        assert egolife.main(["agents", str(run.folder), "--agents", "claude,codex", "--models", "none", "--dry-run", "--timeout", "1200"]) == 0
    out = capsys.readouterr().out
    assert "nothing was run" in out and "You cannot see my recordings." in out and f"are in {run.view_dir(0)}:" in out
    assert not run.view_dir(0).exists() and not (run.folder / "results").exists() and not (run.folder / "homes" / "claude-fake").exists()
    for agent in ("claude", "codex"):
        for arm in ("blind", "files"):
            assert (run.folder / "dry-run" / agent / arm / "ego-001" / "prompt.txt").read_text(encoding="utf-8").startswith("It is now")
    assert bench.read_json(run.folder / "run.json")["timeout_s"] == 900.0, "a dry run keeps no timeout"


def test_a_view_that_cannot_be_removed_is_told_and_does_not_hide_the_error(built, monkeypatch, capsys):
    for k in (0, 1):
        built.view_dir(k).mkdir(parents=True, exist_ok=True)
    real = bench.remove_tree

    def busy(path):
        if path == built.view_dir(0):
            raise PermissionError("a child still holds a file")
        real(path)

    monkeypatch.setattr(bench, "remove_tree", busy)
    egolife.release_views(built)  # does not raise
    out = capsys.readouterr().out
    assert "could not remove" in out and "a child still holds a file" in out
    assert built.view_dir(0).exists() and not built.view_dir(1).exists(), "the others are still removed"
    monkeypatch.undo()
    egolife.release_views(built)
    assert not built.view_dir(0).exists()


def test_the_command_line_tells_a_failed_subprocess_as_an_error(tmp_path, monkeypatch, capsys):
    def build(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["icacls"], stderr="denied")

    monkeypatch.setattr(egolife, "build", build)
    assert egolife.main(["build", "--source", str(tmp_path), "--out", str(tmp_path / "o"), "--temp", str(tmp_path / "t")]) == 1
    assert "error: Command '['icacls']' returned non-zero exit status 1" in capsys.readouterr().err


def test_the_estimate_prices_a_blind_exec_at_its_own_figure(built, tmp_path, capsys):
    run = copy_run(built, tmp_path)
    bench.show_agents(run, ["claude", "codex"], [FAKE_MODEL], [], 3)
    hours = (2 * bench.BLIND_MIN / 3 + 2 * bench.EXEC_MIN / 3 + 2 * bench.EXEC_MIN / 1) * 10 / 60  # 10 questions in each of 6 arms
    assert f"60 execs to run, about {hours:.1f} h at {bench.EXEC_MIN:g} min an exec ({bench.BLIND_MIN:g} for a blind one) over 3 jobs" in \
        capsys.readouterr().out
