"""The benchmark's runner and report (TESTBED.md, Benchmark): a tiny week is built, copied into a home, asked with stub agents and
with the fake decider, and reported. No real agent or model runs. The run is built once for the module and worker, and shared by
the tests that read it; a test that must start clean works on a copy of its folder."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import threading
import time
import types
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import bench
import claude_run
import codex_run
import grade
import haystack
import pytest
from conftest import FAKE, Api, Daemon, model_folder

from mms import config, decider, decider_d1

DAYS = [date(2026, 10, 30), date(2026, 10, 31)]
FAKE_MODEL = bench.Model("fake", FAKE, decider_d1.ABOUT)  # the fake has d1-Omni-2B's facts: it is asked every question
# The tests land on any worker (xdist 3.8 does not honour xdist_group here), so each worker that needs the run builds its own.
pytestmark = pytest.mark.xdist_group("bench")


PROBED: list[dict] = []  # what the probe (a stand-in for `shortcuts.run`) was called with and found, by `built`


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A tiny week built once for the module; its shortcut probe is a stand-in that records its call (the real one has its own
    tests, and takes seconds)."""
    root = tmp_path_factory.mktemp("bench")

    def probe(plan, folder, out, *, ffmpeg=None, seed=None):
        PROBED.append({"plan": plan, "folder": Path(folder), "out": Path(out), "ffmpeg": ffmpeg, "seed": seed,
                       "data": any((Path(folder) / f.path).is_file() for f in plan.files),
                       "loaded": (Path(out) / "homes" / "snapshot" / "index.sqlite").exists()
                       and not (Path(out) / "homes" / "snapshot" / "daemon.json").exists(), "run_json": (Path(out) / "run.json").exists()})
        return []

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bench.shortcuts, "run", probe)
        return bench.build(root / "out", "tiny", 7, temp=root / "temp", days=DAYS)


def right(q: dict) -> str:
    """The answer that is right for a question (an absent one is `unknown`; a week question names its day)."""
    g = q["grade"]
    if g["kind"] == "absent":
        return "unknown"
    if g["kind"] in ("number", "count"):
        return g["truth"]
    answer = g["time"][:5] if g["kind"] == "time" else f"{g['truth']} at {g['time'][:5]}"
    return f"{answer} on {g['date']}" if g.get("date") else answer


def shape(plan) -> types.SimpleNamespace:
    """How many questions of each kind a plan has, so that a test does not hold the number of its week: `n` all of them, `texts`
    those of text and records (the controls), `media` the rest, `absent_media` and `needles` (media with something to find),
    `oracle` those with an oracle ask, `by_type` the count of each type and `absent_of` the absent ones of each."""
    qs = plan.questions
    media = [q for q in qs if q["type"] in bench.MEDIA]
    return types.SimpleNamespace(
        n=len(qs), texts=sum(1 for q in qs if q["type"] in bench.CONTROL_TYPES), media=len(media),
        absent_media=sum(1 for q in media if q["absent"]), needles=sum(1 for q in media if not q["absent"]),
        oracle=sum(1 for q in qs if q["oracle"] is not None), as_of=sum(1 for q in qs if q["source"] in bench.AS_OF),
        by_type={t: sum(1 for q in qs if q["type"] == t) for t in haystack.TYPES},
        absent_of={t: sum(1 for q in qs if q["type"] == t and q["absent"]) for t in haystack.TYPES})


def scan(api, **params) -> dict:
    job = api.call("jobs/start", script="ask", params=params, wait_s=50)
    while job["state"] in ("queued", "running"):
        job = api.call("jobs/get", job=job["job"], wait_s=50)
    assert job["state"] == "done", job
    return job["result"]


class Stub:
    """An agent that answers right for text and records in both arms and for every type with mms, `unknown` for pictures,
    video and sound without it. `first` changes the fields of its first result, `script[n]` those of its n-th (None: none), and
    `by[(arm, qid)]` those of that exec's; `pause` is how long each exec takes. `log` holds (arm, source, qid, start, end)."""

    def __init__(self, first: dict | None = None, *, script=(), by=None, pause: float = 0.0):
        self.execs: list[bench.Exec] = []
        self.marks: dict[tuple[str, str], str | None] = {}  # (arm, qid) -> the working folder's project file, when called
        self.first, self.script, self.by, self.pause = first or {}, list(script), by or {}, pause
        self.log: list[tuple] = []
        self.lock = threading.Lock()

    def __call__(self, e: bench.Exec) -> dict:
        q = e.question
        marker = e.work / ".mms-project"
        began = time.monotonic()
        with self.lock:
            self.execs.append(e)
            n = len(self.execs) - 1
            self.marks[(e.arm, q["qid"])] = marker.read_text(encoding="utf-8") if marker.exists() else None
        if e.arm == "mms" and q["template"] == "photo-sign":  # an ask with no caller: its labels are ambient
            scan(Api(e.home), question=q["fake"]["question"], sources=["phone"], since=q["since"], until=q["until"])
        time.sleep(self.pause)
        answer = "unknown" if e.arm == "files" and q["type"] in ("image", "video", "audio") else right(q)
        tools = ([{"tool": "ask", "arguments": {"question": q["question"]}}, {"tool": "find", "arguments": {"scope": "all"}}]
                 if e.arm == "mms" else [{"tool": "shell", "arguments": {"command": "ls"}}])
        out = {"answer": answer, "final": f"ANSWER: {answer}", "tools": tools, "seconds": 1.5, "problem": None, "turns": 2,
               "tokens": {"input": 1000, "output": 100, "cache_read": 0, "cache_write": 0}, "cost_usd": 0.1,
               "used_percent": 10.0, "model": "stub", "denials": 0}
        out.update(self.first if n == 0 else {})
        out.update((self.script[n] if n < len(self.script) else None) or {})
        out.update(self.by.get((e.arm, q["qid"]), {}))
        with self.lock:
            self.log.append((e.arm, q["source"], q["qid"], began, time.monotonic()))
        return out


@pytest.fixture(scope="module")
def ran(built):
    """The stub over the built run: the files arm, then the mms arm of the fake decider."""
    stub = Stub()
    bench.run_agents(built, {"claude": stub}, [FAKE_MODEL], jobs=2)
    return stub


def result(run, agent: str, folder: str, qid: str) -> dict:
    return json.loads((run.folder / "results" / agent / folder / f"{qid}.json").read_text(encoding="utf-8"))


def trimmed(built, tmp_path, homes: bool = False) -> bench.Run:
    """A copy of the built run with no results or execs (and with the snapshot home, when `homes`, but none of the homes copied
    from it): it shares the data, which is only read."""
    folder = tmp_path / "run"

    def skip(where, names):
        out = [n for n in names if n in ("results", "execs", "models.json") or n.startswith("report.")]
        if os.path.basename(where) == "homes":
            out += [n for n in names if n != "snapshot"]
        elif not homes:
            out += [n for n in names if n == "homes"]
        return out

    shutil.copytree(built.folder, folder, ignore=skip)
    return bench.Run.open(folder)


def reseeded(built, tmp_path, seed: int) -> bench.Run:
    """A trimmed copy that is another seed's run: its run.json holds that seed and the hash of that seed's plan."""
    run = trimmed(built, tmp_path)
    meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
    other = haystack.plan(seed, "tiny", run.days)
    meta.update(seed=seed, plan=bench.plan_hash(other), files=bench.files_hash(other))
    (run.folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return bench.Run.open(run.folder)


def test_a_run_is_built_copied_and_continued(built, ran, tmp_path):
    plan = built.plan()
    meta = json.loads((built.folder / "run.json").read_text(encoding="utf-8"))
    assert (meta["seed"], meta["scale"], meta["generator"]) == (7, "tiny", haystack.VERSION) and meta["mms_commit"]
    assert meta["plan"] == bench.plan_hash(plan) and len(meta["plan"]) == 64
    assert bench.Run.open(built.folder) == built and not (built.snapshot() / "daemon.json").exists()

    # the data is read-only and frozen: a change, an addition and a removal are each named
    files = [p for p in built.data.rglob("*") if p.is_file()]
    assert files and not any(os.access(p, os.W_OK) for p in files) and bench.changed(built.data, built.manifest) is None
    def frozen(name: str):
        folder = tmp_path / name
        (folder / "a").mkdir(parents=True)
        (folder / "a" / "one.txt").write_text("one")
        (folder / "two.txt").write_text("two")
        bench.freeze(folder, tmp_path / f"{name}.json")
        assert bench.changed(folder, tmp_path / f"{name}.json") is None and not os.access(folder / "two.txt", os.W_OK)
        return folder, tmp_path / f"{name}.json"

    folder, manifest = frozen("edited")
    (folder / "two.txt").chmod(stat.S_IWRITE)
    (folder / "two.txt").write_text("twenty")
    assert bench.changed(folder, manifest) == "changed: two.txt"
    folder, manifest = frozen("added")
    (folder / "a" / "three.txt").write_text("3")
    assert bench.changed(folder, manifest) == "added: a/three.txt"
    folder, manifest = frozen("removed")
    bench.remove_tree(folder / "a")
    assert bench.changed(folder, manifest) == "gone: a/one.txt"

    # removing a tree that holds a link to a read-only file (a home's media, a clip an agent linked) deletes the name and leaves
    # the file and its other links read-only; a read-only file with no other link is still removed
    folder, manifest = frozen("linked")
    link = tmp_path / "copy" / "one.txt"
    link.parent.mkdir()
    os.link(folder / "a" / "one.txt", link)
    bench.remove_tree(link.parent)
    assert not link.parent.exists() and not os.access(folder / "a" / "one.txt", os.W_OK) and bench.changed(folder, manifest) is None
    os.link(folder / "two.txt", tmp_path / "two-link.txt")
    bench.remove_tree(tmp_path / "two-link.txt")
    assert not (tmp_path / "two-link.txt").exists() and not os.access(folder / "two.txt", os.W_OK)
    bench.unlink(folder / "two.txt")
    assert not (folder / "two.txt").exists(), "unlink deletes a read-only file too"
    bench.remove_tree(folder / "a")
    assert not (folder / "a").exists()

    # a copy of the snapshot has the decider's model and every item and record the plan loaded; one with a daemon is refused
    home = bench.copy_home(built.snapshot(), tmp_path / "copy", FAKE_MODEL)
    assert config.load(home).decider_model == FAKE and bench.copy_home(built.snapshot(), home, FAKE_MODEL) == home
    daemon = Daemon(home).start()
    try:
        held = {s["name"]: s for s in daemon.api.call("status")["sources"]}
        for source in {f.source for f in plan.files}:
            assert held[source]["items"] == sum(1 for f in plan.files if f.source == source), source
        assert held[haystack.LOG_SOURCE]["records"] == sum(len(v) for v in plan.logs.values())
        assert daemon.api.call("status")["decider"]["model"] == FAKE
    finally:
        daemon.stop()
    busy = tmp_path / "busy"
    shutil.copytree(built.snapshot(), busy)
    (busy / "daemon.json").write_text("{}")
    with pytest.raises(RuntimeError, match="daemon"):
        bench.copy_home(busy, tmp_path / "never", FAKE_MODEL)
    assert not (tmp_path / "never").exists() and not (tmp_path / "never.part").exists()

    # both arms ran every question, the files arm first; the mms arm's slots held a project of their own, the files arm's none
    questions = [q["qid"] for q in plan.questions]
    n = len(questions)
    assert n >= 15 and len(ran.execs) == 2 * n
    arms = [e.arm for e in ran.execs]
    assert arms == ["files"] * n + ["mms"] * n, "files execs before mms execs"
    for e in ran.execs:
        assert e.work.parent == built.temp and e.work.name.startswith(f"mmsb-{built.key}-w") and e.deny == built.deny(e.work)
        assert e.data == (built.data if e.arm == "files" else None) and e.bin_dir == built.bin and e.seed == 7
        assert (e.model is None) == (e.arm == "files") and (e.home is None) == (e.arm == "files")
        assert ran.marks[(e.arm, e.question["qid"])] == (f"bench-{e.question['qid']}" if e.arm == "mms" else None)
        assert "mms has recorded" in e.prompt if e.arm == "mms" else str(built.data) in e.prompt
    for arm, folder in (("files", "files"), ("mms", "mms-fake")):
        got = {q: result(built, "claude", folder, q) for q in questions}
        assert sorted(p.stem for p in (built.folder / "results" / "claude" / folder).glob("*.json")
                      if p.stem != "ambient-labels") == sorted(questions)
        assert all(r["problem"] is None and r["agent"] == "claude" and r["arm"] == arm for r in got.values())
    sign = result(built, "claude", "mms-fake", "image-photo-sign-1")
    assert (sign["passed"], sign["abstained"], sign["model"], sign["seed"], sign["denials"]) == (True, False, "fake", 7, 0)
    assert (sign["calls"], sign["tools"], sign["scope_all"]) == (2, {"ask": 1, "find": 1}, 1)
    assert sign["exec"] == "execs/claude/mms-fake/image-photo-sign-1" and sign["agent_model"] == "stub" and sign["turns"] == 2
    blind = result(built, "claude", "files", "image-photo-sign-1")
    assert (blind["passed"], blind["abstained"], blind["model"], blind["scope_all"]) == (False, True, None, 0)

    # the label check is live: the stub's ask had no project, and the pinned ones have
    ambient = result(built, "claude", "mms-fake", "ambient-labels")
    assert ambient and all(r["by"] == "ask" and r["project"] is None for r in ambient)

    # a second command asks nothing; a deleted result is asked again alone
    again = Stub()
    bench.run_agents(built, {"claude": again}, [FAKE_MODEL], jobs=2)
    assert again.execs == []
    gone = built.folder / "results" / "claude" / "mms-fake" / "audio-mic-alarm-1.json"
    gone.unlink()
    once = Stub()
    bench.run_agents(built, {"claude": once}, [FAKE_MODEL], jobs=2)
    assert [(e.arm, e.question["qid"]) for e in once.execs] == [("mms", "audio-mic-alarm-1")] and gone.exists()


@pytest.mark.skipif(os.name != "nt", reason="a directory junction is a Windows link")
def test_remove_tree_removes_a_junction_and_never_what_it_points_to(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    work = tmp_path / "work"
    work.mkdir()
    (work / "mine.txt").write_text("mine")

    def junction(link: Path) -> Path:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
        assert link.is_junction() and not link.is_symlink()
        return link

    bench.remove_tree(junction(work / "link"))  # as `empty(work)` meets one an agent made: it is the child itself
    assert not (work / "link").exists() and (target / "keep.txt").read_text() == "keep" and (work / "mine.txt").exists()
    junction(work / "link")
    bench.remove_tree(work)  # and one met inside a tree
    assert not work.exists() and (target / "keep.txt").read_text() == "keep"
    work.mkdir()
    junction(work / "again")
    bench.empty(work)
    assert list(work.iterdir()) == [] and (target / "keep.txt").exists()


def test_the_decider_tier_scores_the_fake_questions(built, monkeypatch, capsys):
    plan = built.plan()
    # the long ranges (a screen, a porch clip or sound for a week) take the fake seconds each, and add nothing the shorter ones do not
    # prove: every kind of question stays (a part, a day and a `last` week one; a count; the absent ones). The screen's part question stays
    # whatever its windows (a frame an act: thousands), so the tier's screen scoring is covered.
    asked = [q for q in plan.questions if q["oracle"] is not None and (haystack.windows_in(plan, q["source"], q["since"], q["until"]) <= 60
                                                                        or (q["source"] == "screen" and q["level"] == "part" and not q["absent"]))]
    assert any(q["source"] == "screen" and haystack.windows_in(plan, "screen", q["since"], q["until"]) > 60 for q in asked)
    assert {"part", "day", "week"} <= {q["level"] for q in asked} and any(q.get("last") for q in asked) and any(q["absent"] for q in asked)
    assert any(q["fake"]["check"] == "count" for q in asked)
    bench.run_decider(built, [FAKE_MODEL], ask_with="fake", qids={q["qid"] for q in asked})
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("fake ")]
    assert len(lines) == len(asked) and all(" false=" in line and " windows in " in line for line in lines), lines
    folder = built.folder / "results" / "decider" / "fake"
    assert (folder / "warm.json").exists() and sorted(p.stem for p in folder.glob("*.json")) == sorted(["warm", *(q["qid"] for q in asked)])
    assert {q["template"] for q in plan.questions} - {q["template"] for q in plan.questions if q["oracle"] is not None} == {"config-asof"},         "every question but the config version has an oracle"
    for q in asked:
        r = json.loads((folder / f"{q['qid']}.json").read_text(encoding="utf-8"))
        assert (r["qid"], r["model"], r["type"], r["absent"]) == (q["qid"], "fake", q["type"], q["absent"])
        assert (r["level"], r["last"]) == (bench.level_of(q), bool(q.get("last")))
        assert r["limit"] == (bench.LAST_LIMIT if q.get("last") else 50 if q["fake"]["check"] == "count" else 20), q["qid"]
        if q["absent"]:
            assert (r["found"], r["top5"], r["false"], r["in_range"], r["p"], r["rank"]) == (None, None, 0, 0, None, None), q["qid"]
        else:
            assert r["found"] is True and r["top5"] is True and r["in_range"] >= 1 and r["rank"] >= 1, q["qid"]
            assert r["p"] is not None and r["moments_found"] and r["false"] == 0, q["qid"]
        assert r["windows"] > 0 and r["seconds"] >= 0 and r["wall_s"] >= 0 and r["sightings"] is not None
    tier = bench.numbers([built])["decider"]
    assert tier["missed"] == [] and tier["asks"] == len(asked) and tier["models"]["fake"]["first_s"] is not None
    types_ = tier["models"]["fake"]["types"].values()
    assert all(t["found"] == t["needles"] and t["absent_clean"] == t["absent"] for t in types_)
    text = bench.markdown(bench.numbers([built]))
    images = sum(1 for q in asked if q["type"] == "image" and not q["absent"])
    assert f"| fake | image | {images}/{images} (100%)" in text and "Every template was found" in text

    def no_daemon(*args, **kwargs):
        raise AssertionError("a model with every record is asked nothing, and its daemon is not started")

    monkeypatch.setattr(bench, "Daemon", no_daemon)
    bench.run_decider(built, [FAKE_MODEL], ask_with="fake", qids={q["qid"] for q in asked})


def test_the_report_counts_as_it_should(built, ran, tmp_path):
    plan = built.plan()
    k = shape(plan)
    assert all(grade.grade(q["grade"], right(q)) for q in plan.questions), "the stub's right answers pass the grader"
    assert not any(grade.grade(q["grade"], "unknown") for q in plan.questions if not q["absent"])
    figures = bench.numbers([built])
    model = figures["agents"]["claude"]["fake"]
    m = model["all"]
    assert (m["n"], model["questions"], model["left_out"], model["pending"]) == (k.n, k.n, 0, 0)
    right_without = k.texts + k.absent_media  # the text and records, and the absent media questions
    assert (m["files"]["correct"], m["mms"]["correct"]) == (right_without, k.n), "files: the controls, and `unknown` for what is absent"
    assert (m["b"], m["c"], m["p"]) == (k.needles, 0, bench.mcnemar(k.needles, 0)) and m["diff"] == pytest.approx(k.needles / k.n)
    assert m["ci"][0] > 0
    absent_all = sum(k.absent_of.values())
    for arm in ("files", "mms"):
        assert (m[arm]["absent"], m[arm]["absent_answered"], m[arm]["needles"]) == (absent_all, 0, k.n - absent_all)
    assert (m["files"]["abstained"], m["mms"]["abstained"]) == (k.needles, 0), "without mms the media needles got `unknown`"
    for type_ in ("image", "video", "audio"):
        assert model["types"][type_] == {"n": k.by_type[type_], "files": k.absent_of[type_], "mms": k.by_type[type_]}
    for type_ in ("text", "records"):
        assert model["types"][type_] == {"n": k.by_type[type_], "files": k.by_type[type_], "mms": k.by_type[type_]}
    assert (model["scope_all"], m["mms"]["median_calls"], m["files"]["median_calls"]) == (k.n, 2, 1)
    assert m["files"]["tokens"]["input"] == 1000 * k.n and m["files"]["median_s"] == 1.5
    assert m["mms"]["cost_usd"] == pytest.approx(0.1 * k.n) and model["kinds"] == {"files": {}, "mms": {}}

    # the media row: pictures, video and sound pooled, the part that depends on the model (the absent ones are right without)
    media = model["media"]
    assert (media["n"], media["files"]["correct"], media["mms"]["correct"], media["b"], media["c"]) ==         (k.media, k.absent_media, k.media, k.needles, 0)
    assert media["diff"] == pytest.approx(k.needles / k.media) and media["p"] == bench.mcnemar(k.needles, 0) < 0.05
    totals = figures["totals"]
    assert totals["results"] == 2 * k.n and totals["problems"] == {} and totals["to_rerun"] == 0
    assert totals["ambient_labels"] >= 1, "the stub's ask with no project"
    assert figures["agent_models"] == {"claude": {"stub": 2 * k.n}}
    text = bench.markdown(figures)
    pct = lambda right_, n: f"{right_}/{n} ({100 * right_ / n:.0f}%)"  # noqa: E731
    by_type = [f"{k.absent_of[t]}/{k.by_type[t]} -> {k.by_type[t]}/{k.by_type[t]}" for t in ("image", "video", "audio")]
    for line in ("## claude", f"| fake | all | {k.n}/{k.n} |", f"| fake | media | {k.media} |", pct(right_without, k.n), pct(k.n, k.n),
                 f"{100 * (k.n - right_without) / k.n:+.0f} pts", f"{100 * k.needles / k.media:+.0f} pts", f"{k.needles} / 0",
                 bench.chance(media["p"]), "## By data type", *by_type, "## The decider tier", "problems (files / mms)",
                 "would not be recorded at all", "mcnemar(6, 0) = 0.031", "mcnemar(5, 0) = 0.0625", "Runs pooled: 1; seeds 7",
                 f"Agents: claude stub ({2 * k.n} results).",
                 "Ask labels with no project", f"leak into another's scope): {totals['ambient_labels']}.",
                 "Problems by kind: none", "Multiple comparisons: this report has k = 1 agent-model rows",
                 "(2 p values). At p < 0.05 about 0.05 of k rows would fall below it by chance alone; 2 of the 2 p values here are."):
        assert line in text, line

    # runs are pooled by question: another seed adds new questions (here with no results yet)
    other = reseeded(built, tmp_path, 8)
    pooled = bench.numbers([built, other])
    assert [r["seed"] for r in pooled["runs"]] == [7, 8]
    assert (pooled["agents"]["claude"]["fake"]["all"]["n"], pooled["agents"]["claude"]["fake"]["pending"]) == (k.n, k.n)
    assert "Runs pooled: 2; seeds 7, 8" in bench.markdown(pooled)

    # the statistics
    assert bench.mcnemar(6, 0) == 0.03125 and bench.mcnemar(5, 0) == 0.0625 and bench.mcnemar(0, 0) == 1.0
    assert bench.mcnemar(3, 3) == 1.0 and bench.mcnemar(10, 2) == pytest.approx(0.0386, abs=1e-4)
    assert bench.mcnemar(2, 6) == bench.mcnemar(6, 2)


def test_the_report_counts_each_agents_model_and_effort():
    """Results of two settings pooled by mistake show in the report's Agents line."""
    medium, high = {"agent_model": "terra", "agent_effort": "medium"}, {"agent_model": "terra", "agent_effort": "high"}
    loaded = [(None, {"codex": {"files": {"a": medium, "b": high}, "mms-d1-3b": {"a": medium}}, "claude": {"files": {"a": {}}}})]
    assert bench.agent_models(loaded) == {"claude": {"unknown": 1}, "codex": {"terra at high effort": 1, "terra at medium effort": 2}}


def test_a_stopped_agent_starts_nothing_more(built, tmp_path, capsys):
    run = trimmed(built, tmp_path)  # no snapshot: an mms arm that started would fail, so none does
    total = len(run.plan().questions)
    results = run.folder / "results"

    limited = Stub({"problem": "limit: weekly", "answer": None})
    bench.run_agents(run, {"claude": limited}, [FAKE_MODEL], jobs=1)
    assert len(limited.execs) == 1 and len(list((results / "claude" / "files").glob("*.json"))) == 1
    assert "limit: weekly" in capsys.readouterr().out
    first = limited.execs[0].question["qid"]
    assert not bench.is_done(bench.load_result(results / "claude" / "files" / f"{first}.json"))

    # the next command asks the limited question again, and every other
    resumed = Stub()
    bench.run_agents(run, {"claude": resumed}, [], jobs=1)
    assert len(resumed.execs) == total and resumed.execs[0].question["qid"] == first

    # Codex stops at STOP_AT of the week's use, and when the use cannot be read
    codex = Stub({"used_percent": bench.STOP_AT + 1})
    bench.run_agents(run, {"codex": codex}, [], jobs=1)
    assert len(codex.execs) == 1 and f"{bench.STOP_AT + 1:g}" in capsys.readouterr().out
    assert bench.is_done(bench.load_result(results / "codex" / "files" / f"{codex.execs[0].question['qid']}.json"))
    blind = Stub({"used_percent": None})
    bench.run_agents(run, {"codex": blind}, [], jobs=1)
    assert len(blind.execs) == 1 and "cannot be read" in capsys.readouterr().out
    steady = Stub()
    bench.run_agents(run, {"codex": steady}, [], jobs=1)
    assert len(steady.execs) == total - 2, "under the limit and readable, the rest runs"

    # Claude's use of the week stops it at the same line, from the rate-limit event's number
    over = Stub({"used_percent": bench.STOP_AT})
    other = trimmed(built, tmp_path / "again")
    bench.run_agents(other, {"claude": over}, [], jobs=1)
    assert len(over.execs) == 1 and f"{bench.STOP_AT:g}%" in capsys.readouterr().out

    # a result that escaped is done, and is a failed answer; more than JOBS_MAX at once is refused
    escaped = Stub({"problem": "escaped: Read reached the answers", "answer": None})
    third = trimmed(built, tmp_path / "third")
    bench.run_agents(third, {"claude": escaped}, [], jobs=1)
    saved = bench.load_result(third.folder / "results" / "claude" / "files" / f"{escaped.execs[0].question['qid']}.json")
    assert saved["passed"] is False and bench.is_done(saved) and len(escaped.execs) == total
    with pytest.raises(ValueError, match="jobs"):
        bench.run_agents(third, {"claude": Stub()}, [], jobs=bench.JOBS_MAX + 1)


def private(built, tmp_path) -> bench.Run:
    """`trimmed`, with a data folder of its own (frozen again), for a test that changes or loses the data."""
    run = trimmed(built, tmp_path)
    data = tmp_path / "data"
    shutil.copytree(built.data, data)
    run = dataclasses.replace(run, data=data)
    bench.freeze(data, run.manifest)
    return run


@pytest.mark.skipif(os.name != "nt", reason="Windows permissions")
def test_every_data_file_is_left_with_the_permissions_its_folder_gives_so_codexs_sandbox_reads_it(tmp_path):
    # the pilot's screen, porch and mic files were moved out of a Python temporary folder, kept its owner-only permissions,
    # and Codex's sandbox could not open one: its files arm answered "unknown" to every video question
    import tempfile

    import haystack_media as hm
    # the data folder sits right in TEMP, as a run's does: pytest's tmp_path is owner-only itself, which hides the fault
    data, work = Path(tempfile.gettempdir()) / f"mmsb-acl-test-{os.getpid()}-{time.time_ns()}", Path(tempfile.mkdtemp(prefix="acl-"))
    try:
        (data / "screen").mkdir(parents=True)
        (data / "notes").mkdir()
        (work / "moved.mkv").write_bytes(b"x")
        shutil.move(str(work / "moved.mkv"), str(data / "screen" / "moved.mkv"))
        (work / "made.mkv").write_bytes(b"y")
        hm._finish(work / "made.mkv", data / "screen" / "made.mkv")
        (data / "notes" / "note.md").write_text("n", encoding="utf-8")
        assert bench.own_permissions(data) == ["screen/moved.mkv"], "a moved file keeps its folder's; a copied or new one does not"
        assert not (work / "made.mkv").exists() and (data / "screen" / "made.mkv").read_bytes() == b"y"
        bench.freeze(data, tmp_path / "manifest.json")
        assert bench.own_permissions(data) == [] and bench.changed(data, tmp_path / "manifest.json") is None
        assert not os.access(data / "screen" / "moved.mkv", os.W_OK), "still read-only"
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if data.exists():
            bench.remove_tree(data)


def test_a_changed_file_stops_the_files_arm_and_a_lost_one_is_written_again(built, tmp_path, capsys):
    run = private(built, tmp_path)
    victim = next(p for p in sorted(run.data.rglob("*")) if p.is_file())
    rel = victim.relative_to(run.data).as_posix()

    class Vandal(Stub):
        def __call__(self, e):
            victim.chmod(stat.S_IWRITE)
            victim.write_bytes(victim.read_bytes() + b"x")
            return super().__call__(e)

    vandal = Vandal()
    bench.run_agents(run, {"claude": vandal}, [], jobs=1)
    assert len(vandal.execs) == 1, "the arm stops when the data changed"
    saved = bench.load_result(run.folder / "results" / "claude" / "files" / f"{vandal.execs[0].question['qid']}.json")
    assert saved["problem"] == f"escaped: the data changed: changed: {rel}" and saved["passed"] is False
    with pytest.raises(RuntimeError, match="data changed before"):
        bench.run_agents(run, {"claude": Stub()}, [], jobs=1)

    # the temp folder lost the data (Windows cleans it): it is written again from the plan, and frozen
    bench.remove_tree(run.data)
    bench.remove_tree(run.bin)
    bench.ensure_data(run)
    assert "writing it again" in capsys.readouterr().out and bench.changed(run.data, run.manifest) is None
    assert sorted(bench.stats_of(run.data)) == sorted(bench.stats_of(built.data))
    assert (run.bin / "ffmpeg.exe").exists() == (os.name == "nt")
    again = Stub()
    bench.run_agents(run, {"claude": again}, [], jobs=1)
    assert len(again.execs) == len(run.plan().questions) - 1, "the escaped result is done: it counts as a failed answer"
    assert bench.changed(run.data, run.manifest) is None


def test_models_are_resolved_the_gpu_is_checked_and_the_prompt_differs_only_in_the_arm(tmp_path, built, monkeypatch):
    models = tmp_path / "models"
    model_folder(models, "d1-3B")
    model_folder(models, "h2o-lightning-4b")
    monkeypatch.setattr(bench.modelenv, "problem", lambda backend, folder, envs: None)
    monkeypatch.setattr(bench, "serving", lambda url: False)
    ready, skipped = bench.resolve_models(["d1-3b", "D3-EDGE", "h2o-lightning-4b", "nonsense", "d1-3B"], models)
    assert [m.name for m in ready] == ["d1-3B"] and ready[0].spec == (models / "d1-3B").resolve().as_posix()
    assert ready[0].url is None and ready[0].slug == "d1-3B" and ready[0].about is decider.known("d1-3B")[1]
    reasons = dict(line.split(": ", 1) for line in skipped)
    assert "mms model use d3-edge" in reasons["d3-edge"] and "not serving" in reasons["h2o-lightning-4b"]
    assert "not a model mms knows" in reasons["nonsense"] and len(skipped) == 3, "d1-3B named twice is one model"
    monkeypatch.setattr(bench, "serving", lambda url: True)
    served, _ = bench.resolve_models(["h2o-lightning-4b"], models)
    assert served[0].url == "http://127.0.0.1:8741" and not bench.needs_gpu(served[0])
    monkeypatch.setattr(bench.modelenv, "problem", lambda backend, folder, envs: "torch is missing")
    assert "env is not ready: torch is missing" in bench.resolve_models(["d1-3B"], models)[1][0]
    assert bench.needs_gpu(ready[0]) and not bench.needs_gpu(FAKE_MODEL)
    assert bench.slug("pplx-decider-flash-v1.2") == "pplx-decider-flash-v1-2"

    # nvidia-smi's compute apps: some are a busy GPU, none is free, and no nvidia-smi is an error
    def smi(output, code=0):
        return types.SimpleNamespace(run=lambda *a, **k: subprocess.CompletedProcess(a, code, output, "no device"))

    monkeypatch.setattr(bench, "shutil", types.SimpleNamespace(which=lambda name: "nvidia-smi"))
    monkeypatch.setattr(bench, "subprocess", smi("1234, python.exe\n"))
    assert bench.gpu_busy() == "1234, python.exe"
    with pytest.raises(RuntimeError, match="GPU is busy.*python.exe"):
        bench.check_gpu(ready[0])
    bench.check_gpu(FAKE_MODEL)
    monkeypatch.setattr(bench, "subprocess", smi("\n"))
    assert bench.gpu_busy() is None
    bench.check_gpu(ready[0])
    monkeypatch.setattr(bench, "subprocess", smi("", 9))
    with pytest.raises(RuntimeError, match="exited 9"):
        bench.gpu_busy()
    monkeypatch.setattr(bench, "shutil", types.SimpleNamespace(which=lambda name: None))
    with pytest.raises(RuntimeError, match="nvidia-smi is not on PATH"):
        bench.gpu_busy()

    # both arms carry the same facts: only how the sources are reached and how to answer differ
    q = built.plan().questions[6]
    now = datetime(2026, 11, 2, 14, 30).timestamp()
    mms, files = built.prompt(q, "mms", None, now), built.prompt(q, "files", built.data, now)
    week = "from Friday 30 October to Saturday 31 October"  # the week's own dates, not "the past 2 days": the day it runs moves
    assert mms.startswith(f"It is now Monday 02 November 2026, 14:30. mms has recorded these sources of mine {week}.")
    assert files.startswith(f"It is now Monday 02 November 2026, 14:30. The files of these sources, {week}, are in ")
    for text in (mms, files):
        assert q["question"] in text and "screen (my screen recordings), phone (photos and short videos from my phone)" in text
        assert "logs (the payments, web and auth services' logs)" in text and "ANSWER: unknown when" in text
    assert str(built.data) in files and "no ffprobe" in files and "ANSWER: unknown when mms does not hold it" in mms
    for arm in ("both", "blind"):  # the haystack has no blind arm
        with pytest.raises(ValueError, match="arm"):
            built.prompt(q, arm, None)

    # the files arm's deny list holds the run, the results folder, the repo and the owner's agent setup, and refuses a folder in one
    deny = built.deny()
    assert str(built.folder).lower().replace("\\", "/") in deny and ".claude" in deny and "/mms-bench/" in deny
    assert str(codex_run.ROOT).lower().replace("\\", "/") in deny
    for reached in ({"command": "ls ~/mms-bench"}, {"command": "ls ~/mms-bench/20261102-1200-standard-s1"}, "C:\\Users\\x\\mms-bench",
                    {"file_path": "/c/Users/x/mms-bench"}, {"command": "cd '/home/x/mms-bench' && ls"}):
        assert codex_run.reached(reached, deny), reached
    # `mms-bench` is a path part: this repo's worktree, `codex-mms-benchmarks`, is not it
    for harmless in ({"command": "ls /c/Users/x/.worktrees/codex-mms-benchmarks-6e8d00/tests"}, "C:\\mms-benchmarks\\data", "mms-bench"):
        assert codex_run.reached(harmless, deny) is None, harmless
    bench.check_paths(built)
    bench.check_paths(bench.Run(built.folder, 7, "tiny", built.days, tmp_path / "codex-mms-benchmarks-1" / "data", built.temp))
    inside = bench.Run(built.folder, 7, "tiny", built.days, tmp_path / ".codex" / "data", built.temp)
    with pytest.raises(ValueError, match=r"\.codex"):
        bench.check_paths(inside)


def test_the_command_line_dry_runs_runs_and_reports(built, tmp_path, monkeypatch, capsys):
    run = trimmed(built, tmp_path)
    first = run.plan().questions[0]["qid"]

    # a dry run writes the first exec's files per agent, prints both command lines, and runs nothing
    with monkeypatch.context() as patch:
        patch.setattr(bench, "shutil", types.SimpleNamespace(which=lambda name: None))
        patch.setattr(codex_run, "models_of", lambda command: {"models": [
            {"slug": codex_run.model(), "supported_reasoning_levels": [{"effort": codex_run.effort()}]}]})
        assert bench.main(["agents", str(run.folder), "--agents", "claude,codex", "--models", "none", "--dry-run"]) == 0
    out = capsys.readouterr().out
    for text in ("--restricted", "--strict-mcp-config", "sandbox_mode=workspace-write", "nothing was run", "ANTHROPIC_API_KEY",
                 f"question {first}", "environment:"):
        assert text in out, text
    for agent in ("claude", "codex"):
        assert (run.folder / "dry-run" / agent / "files" / first / "instructions.md").exists()
        prompt = (run.folder / "dry-run" / agent / "files" / first / "prompt.txt").read_text(encoding="utf-8")
        assert prompt.startswith("It is now")
    assert not (run.folder / "results").exists() and not (run.folder / "homes").exists()
    assert not (run.folder / "execs").exists(), "an exec folder may hold a run's record: a dry run writes beside it"

    # with a model, the mms arm is prepared too: its MCP config pins the project, and a model that is not ready is named
    with monkeypatch.context() as patch:
        patch.setattr(bench, "shutil", types.SimpleNamespace(which=lambda name: None))
        patch.setattr(bench, "resolve_models", lambda names, folder: ([FAKE_MODEL], ["d3-edge: not downloaded: mms model use d3-edge"]))
        assert bench.main(["agents", str(run.folder), "--agents", "claude", "--models", "fake,d3-edge", "--controls", "fake",
                           "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "skipped d3-edge: not downloaded" in out and "== claude mms-fake, question" in out and "--mcp-config" in out
        servers = json.loads((run.folder / "dry-run" / "claude" / "mms-fake" / first / "mcp.json").read_text(encoding="utf-8"))
        assert servers["mcpServers"]["mms"]["env"]["CLAUDE_PROJECT_DIR"] == str(run.slot(0))
        bench.show_decider(run, [FAKE_MODEL], [])
        assert f"{shape(run.plan()).oracle} oracle asks to run of {shape(run.plan()).oracle}" in capsys.readouterr().out

    # the same command with stub agents runs, after saying what it will run and an estimate; `--jobs` has a limit
    stub = Stub()
    monkeypatch.setattr(bench, "real_agents", lambda: {"claude": stub, "codex": stub})
    logins = []
    monkeypatch.setattr(bench.claude_run, "check_login", lambda: logins.append(1))
    assert bench.main(["agents", str(run.folder), "--agents", "claude", "--models", "none"]) == 0
    assert logins == [1], "Claude's login is checked before its first exec"
    out = capsys.readouterr().out
    n = shape(run.plan()).n
    assert "claude files" in out and f"{n} to run of {n}" in out
    assert f"{n} execs to run, about {n * bench.EXEC_MIN / 3 / 60:.1f} h at 2.5 min an exec over 3 jobs" in out
    assert len(stub.execs) == n
    with pytest.raises(SystemExit):
        bench.main(["agents", str(run.folder), "--jobs", str(bench.JOBS_MAX + 1)])
    with pytest.raises(SystemExit):
        bench.main(["agents", str(run.folder), "--agents", "gemini"])
    capsys.readouterr()

    # a report over a run with no results at all, and over the run that has some, is written in the first run's folder
    blank = trimmed(built, tmp_path / "blank")
    assert bench.main(["report", str(blank.folder)]) == 0
    out = capsys.readouterr().out
    assert "No agent results yet" in out and "No decider results yet" in out and "would not be recorded at all" in out
    assert "would not be recorded" in (blank.folder / "report.md").read_text(encoding="utf-8")
    assert json.loads((blank.folder / "report.json").read_text(encoding="utf-8"))["agents"] == {}
    assert bench.main(["report", str(run.folder), str(reseeded(built, tmp_path / "eight", 8).folder)]) == 0
    assert "Runs pooled: 2; seeds 7, 8" in (run.folder / "report.md").read_text(encoding="utf-8")
    assert bench.main(["report", str(run.folder), str(blank.folder)]) == 1, "the same seed and scale are the same questions"
    assert "same seed (7)" in capsys.readouterr().err
    assert bench.main(["report", str(tmp_path / "nowhere")]) == 1
    assert "error:" in capsys.readouterr().err


# -- problems, ordering, pooling and the record ------------------------------------------------------------------------

def test_a_timeout_a_cap_and_an_escape_are_wrong_answers_that_are_paired(built, tmp_path):
    run = trimmed(built, tmp_path, homes=True)
    kinds = {"image-photo-sign-1": ("timeout", "timeout: no answer in 900 s"), "video-screen-error-1": ("capped", "capped: max turns"),
             "audio-mic-doorbell-1": ("escaped", "escaped: Read reached the answers")}
    stub = Stub(by={("mms", qid): {"problem": problem, "answer": None} for qid, (_, problem) in kinds.items()})
    bench.run_agents(run, {"claude": stub}, [FAKE_MODEL], jobs=2)
    k = shape(run.plan())
    assert len(stub.execs) == 2 * k.n, "none of the three stops the agent"
    for qid, (kind, _) in kinds.items():
        saved = result(run, "claude", "mms-fake", qid)
        assert saved["problem"].startswith(kind) and saved["passed"] is False and bench.is_done(saved), qid

    figures = bench.numbers([run])
    model = figures["agents"]["claude"]["fake"]
    assert (model["all"]["n"], model["left_out"], model["pending"]) == (k.n, 0, 0), "the three are paired, not dropped"
    assert (model["all"]["files"]["correct"], model["all"]["mms"]["correct"]) == (k.texts + k.absent_media, k.n - 3)
    assert (model["all"]["b"], model["all"]["c"]) == (k.needles - 3, 0)
    assert (model["media"]["n"], model["media"]["mms"]["correct"], model["media"]["b"]) == (k.media, k.media - 3, k.needles - 3)
    assert model["kinds"] == {"files": {}, "mms": {"timeout": 1, "capped": 1, "escaped": 1}}
    assert figures["totals"]["problems"] == {"timeout": 1, "capped": 1, "escaped": 1} and figures["totals"]["to_rerun"] == 0
    text = bench.markdown(figures)
    assert "none / capped 1, escaped 1, timeout 1" in text and "Problems by kind: capped 1, escaped 1, timeout 1." in text
    again = Stub()
    bench.run_agents(run, {"claude": again}, [FAKE_MODEL], jobs=2)
    assert again.execs == [], "an exec that timed out is done: it is not asked again"


def test_a_session_that_was_not_the_one_asked_for_is_left_out_and_run_again(built, tmp_path, capsys):
    run = trimmed(built, tmp_path, homes=True)
    n = shape(run.plan()).n
    stub = Stub(script=[None] * n + [{"problem": "setup: the session has tools the arm does not", "answer": None}])
    bench.run_agents(run, {"claude": stub}, [FAKE_MODEL], jobs=1)
    assert [e.arm for e in stub.execs] == ["files"] * n + ["mms"], "the agent stops at the first setup problem"
    assert "stopping: claude: setup: the session has tools" in capsys.readouterr().out
    first = stub.execs[-1].question["qid"]
    saved = result(run, "claude", "mms-fake", first)
    assert saved["problem"].startswith("setup") and saved["passed"] is False and not bench.is_done(saved)

    figures = bench.numbers([run])
    model = figures["agents"]["claude"]["fake"]
    assert (model["all"]["n"], model["left_out"], model["pending"]) == (0, 1, n - 1), "left out of the pairs, not a wrong answer"
    assert model["kinds"]["mms"] == {"setup": 1} and model["all"]["diff"] is None and model["all"]["p"] is None
    assert f"| fake | all | 0/{n} | -" in bench.markdown(figures) and figures["totals"]["to_rerun"] == 1

    # the next command asks it again with the rest, and not the files arm
    resumed = Stub()
    bench.run_agents(run, {"claude": resumed}, [FAKE_MODEL], jobs=1)
    assert [e.arm for e in resumed.execs] == ["mms"] * n and resumed.execs[0].question["qid"] == first
    assert [e.question["until"] for e in resumed.execs] == sorted(e.question["until"] for e in resumed.execs), "in `until` order"
    assert bench.numbers([run])["agents"]["claude"]["fake"]["all"]["n"] == n


def test_a_lost_gpu_stops_the_mms_arm_and_its_exec_is_asked_again(built, tmp_path, capsys, monkeypatch):
    """After a driver crash mms's model cannot read, and an agent still answers from `find` and `look`: the exec that ends on a
    lost GPU is a setup problem, nothing more starts, and the next command asks it again."""
    run = trimmed(built, tmp_path, homes=True)
    n = shape(run.plan()).n
    bench.run_agents(run, {"claude": Stub()}, [], jobs=1)  # the files arm, before the GPU matters
    checks = iter([None, "nvidia-smi exited 6: GPU is lost.  Reboot the system to recover this GPU"])
    monkeypatch.setattr(bench, "needs_gpu", lambda model: True)
    monkeypatch.setattr(bench, "check_gpu", lambda model: None)
    monkeypatch.setattr(bench, "gpu_lost", lambda: next(checks))
    stub = Stub()
    bench.run_agents(run, {"claude": stub}, [FAKE_MODEL], jobs=1)
    assert len(stub.execs) == 2 and "stopping: claude: setup: the GPU was lost" in capsys.readouterr().out
    first, second = (e.question["qid"] for e in stub.execs)
    assert bench.is_done(result(run, "claude", "mms-fake", first))
    lost = result(run, "claude", "mms-fake", second)
    assert lost["passed"] is False and "GPU is lost" in lost["problem"] and not bench.is_done(lost)
    monkeypatch.setattr(bench, "gpu_lost", lambda: None)
    resumed = Stub()
    bench.run_agents(run, {"claude": resumed}, [FAKE_MODEL], jobs=1)
    assert len(resumed.execs) == n - 1 and resumed.execs[0].question["qid"] == second


@pytest.mark.parametrize("problem", ["setup: the session is not the one asked for", "auth: not logged in", "limit: weekly"])
def test_a_setup_auth_or_limit_problem_stops_the_agent_at_once(built, tmp_path, capsys, problem):
    run = trimmed(built, tmp_path)  # no snapshot: an mms arm that started would fail
    stub = Stub({"problem": problem, "answer": None})
    bench.run_agents(run, {"claude": stub}, [FAKE_MODEL], jobs=1)
    assert len(stub.execs) == 1 and f"stopping: claude: {problem}" in capsys.readouterr().out
    assert not bench.is_done(result(run, "claude", "files", stub.execs[0].question["qid"]))
    resumed = Stub()
    bench.run_agents(run, {"claude": resumed}, [], jobs=1)
    assert len(resumed.execs) == shape(run.plan()).n and resumed.execs[0].question["qid"] == stub.execs[0].question["qid"]


def test_three_errors_in_a_row_stop_the_agent_and_a_good_result_between_them_resets_the_count(built, tmp_path, capsys):
    run = trimmed(built, tmp_path)

    def error(n: int) -> dict:
        return {"problem": f"error: boom {n}", "answer": None}

    stub = Stub(script=[error(1), error(2), None, error(3), error(4), error(5)])
    bench.run_agents(run, {"claude": stub}, [], jobs=1)
    assert len(stub.execs) == 6, "two errors, a good result, then three errors in a row"
    assert "claude: 3 errors in a row, the last error: boom 5" in capsys.readouterr().out
    saved = [bench.load_result(run.folder / "results" / "claude" / "files" / f"{e.question['qid']}.json") for e in stub.execs]
    assert [bench.is_done(r) for r in saved] == [False, False, True, False, False, False]
    resumed = Stub()
    bench.run_agents(run, {"claude": resumed}, [], jobs=1)
    assert len(resumed.execs) == shape(run.plan()).n - 1, "the five errors are asked again, the good result is not"


def test_as_of_questions_run_one_at_a_time_in_time_order_and_the_rest_in_parallel(built, tmp_path):
    # the files arm's queue is in ascending `until` (the mms arm's, run one at a time, is checked where it is resumed above)
    first = trimmed(built, tmp_path / "ordered")
    one = Stub()
    bench.run_agents(first, {"claude": one}, [], jobs=1)
    untils = [e.question["until"] for e in one.execs]
    assert len(untils) == shape(first.plan()).n and untils == sorted(untils)

    # in the mms home the file-version sources never overlap, and run in ascending `until`; the others overlap
    second = trimmed(built, tmp_path / "parallel", homes=True)
    slow = Stub(pause=0.15)
    bench.run_agents(second, {"claude": slow}, [FAKE_MODEL], jobs=3)
    mms = sorted((x for x in slow.log if x[0] == "mms"), key=lambda x: x[3])
    as_of = [x for x in mms if x[1] in bench.AS_OF]
    assert {x[1] for x in as_of} == set(bench.AS_OF) and len(as_of) == shape(second.plan()).as_of
    assert all(a[4] <= b[3] for a, b in zip(as_of, as_of[1:])), "two file-version questions were running at once"
    until = {e.question["qid"]: e.question["until"] for e in slow.execs}
    assert [until[x[2]] for x in as_of] == sorted(until[x[2]] for x in as_of), "as-of questions in `until` order"
    others = [x for x in mms if x[1] not in bench.AS_OF]
    assert any(a[3] < b[4] and b[3] < a[4] for i, a in enumerate(others) for b in others[i + 1:]), "the others stay parallel"
    assert any(a[3] < b[4] and b[3] < a[4] for a in as_of for b in others), "and run beside a file-version question"


def test_a_run_whose_plan_changed_is_refused_and_a_seed_is_pooled_once(built, tmp_path):
    run = trimmed(built, tmp_path / "run")
    file = run.folder / "run.json"
    meta = json.loads(file.read_text(encoding="utf-8"))
    assert bench.Run.open(run.folder) == run and bench.numbers([run])["runs"][0]["seed"] == 7
    for change, why in (({"plan": "0" * 64}, "VERSION bump"), ({"plan": None}, "VERSION bump"),
                        ({"generator": haystack.VERSION + 1}, "generator")):
        file.write_text(json.dumps({**meta, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match=why):
            bench.Run.open(run.folder)
    file.write_text(json.dumps(meta), encoding="utf-8")
    assert bench.plan_hash(haystack.plan(7, "tiny", DAYS)) == meta["plan"] != bench.plan_hash(haystack.plan(8, "tiny", DAYS))

    # a second run of the same seed and scale is the same questions: it is refused, another seed is pooled
    twin = trimmed(built, tmp_path / "twin")
    with pytest.raises(ValueError, match=r"same seed \(7\) and scale \(tiny\)"):
        bench.numbers([built, twin])
    assert [r["seed"] for r in bench.numbers([built, reseeded(built, tmp_path / "eight", 8)])["runs"]] == [7, 8]


def test_a_result_holds_the_denials_and_abstains_by_the_questions_own_numbers(built, tmp_path):
    q = next(q for q in built.plan().questions if q["template"] == "absent-log")
    code = q["grade"]["ignore"][0]  # the reserved error code's digits: the question's own number

    def made(answer, **more):
        e = bench.Exec("claude", "mms", FAKE_MODEL, q, "", tmp_path / "execs" / "claude" / "mms-fake" / q["qid"], tmp_path / "w",
                       tmp_path / "home", built.data, built.bin, (), 7)
        return bench.record(e, {"answer": answer, "seconds": 2.0, "problem": None, "tools": [], **more})

    said = made(f"ERR-{code} was never logged", denials=3)
    assert (said["abstained"], said["passed"], said["denials"]) == (True, True, 3), "the code in the answer is the question's own"
    guess = made(f"ERR-{code}: 3 times", denials=0)
    assert (guess["abstained"], guess["passed"], guess["denials"]) == (False, False, 0)
    assert made("unknown")["denials"] is None, "Codex reports none"
    timed_out = made(None, problem="timeout: no answer in 900 s")
    assert (timed_out["passed"], timed_out["abstained"], timed_out["problem"][:7]) == (False, False, "timeout")


# -- hint levels, the oracle's rules, how each arm worked, cost, decider verdicts and the pilot ----------------------------
# These work on stub questions that carry the fields of the harder haystack (`level`, `last`, `truth.others`) and on stub results:
# a plan from today's haystack has none of them, and the report treats a missing `level` as `part`.

T0 = datetime(2026, 10, 30, 12, 0).timestamp()
DAY = 86400.0


def model_of(name: str) -> bench.Model:
    return bench.Model(name, f"/models/{name}", decider.known(name)[1])


def sq(qid: str, level: str = "part", *, last: bool = False, absent: bool = False, moments=None, others=(), oracle: str | None = "yes?",
       check: str = "overlap", oldest: bool = False, since: float = T0) -> dict:
    """A stub question with the fields the runner and the report read; `qid` is `type-template-n`."""
    type_, *rest = qid.split("-")
    template = "-".join(rest[:-1])
    moments = [] if absent else (moments or [[T0 + 3600, T0 + 3600]])
    return {"qid": qid, "type": type_, "template": template, "absent": absent, "source": "phone", "since": since, "until": T0 + 6 * 3600,
            "hint": "on Friday", "question": f"Question {qid}?", "level": level, "last": last,
            "truth": {"value": None, "moments": moments, "others": [list(o) for o in others]},
            "grade": {"kind": "absent" if absent else "time", "truth": None, "time": "13:00", "tol_min": 1, "wrong": [], "ignore": []},
            "oracle": oracle, "fake": {"question": "colour:green", "oldest_first": oldest, "limit": 20, "check": check}}


class Scripted:
    """An API that answers each ask job at once with `result`, and keeps the parameters it was given."""

    def __init__(self, result: dict):
        self.result, self.params = result, []

    def call(self, op: str, **body) -> dict:
        assert op == "jobs/start" and body["script"] == "ask"
        self.params.append(body["params"])
        return {"state": "done", "job": "j-1", "error": None, "result": self.result}


def asked(sightings=(), candidates=(), windows=40, seconds=2.5, stopped="range_end") -> dict:
    """What an ask job's result holds: sightings `(t0, t1, p)`, candidates `(t, p)`, and its progress. A sighting of one instant
    has `t`, as the API gives it."""
    return {"sightings": [{"t": config.iso(a), "p": p} if a == b else {"t0": config.iso(a), "t1": config.iso(b), "p": p}
                          for a, b, p in sightings],
            "candidates": [{"t": config.iso(t), "p": p} for t, p in candidates], "stopped": stopped, "warnings": [],
            "progress": {"windows": windows, "seconds": seconds}}


def test_an_oracle_ask_follows_the_level_rules_and_records_the_needles_p_and_moments():
    model = model_of("d3-edge")
    truth, other = T0 + 3600, T0 + 3600 - 2 * DAY

    # a `last` question asks for LAST_LIMIT sightings (5), newest first, with the 1,800 s any oracle ask is allowed; a sighting of
    # another needle of the template is no false sighting, and the needle's p is the best near the truth over sightings and candidates
    last = sq("image-photo-sign-3", "week", last=True, moments=[[truth, truth]], others=[[other, other]], oldest=True,
              since=T0 - 5 * DAY)
    api = Scripted(asked([(other, other, 0.91), (truth + 4, truth + 4, 0.83), (truth + 5000, truth + 5000, 0.9)],
                         [(truth + 3, 0.7), (truth + 500, 0.95), (T0 + 10, 0.99)], stopped="limit"))
    got = bench.oracle(api, last, model)
    assert bench.LAST_LIMIT == 5
    assert api.params[0] == {"question": "yes?", "sources": ["phone"], "since": T0 - 5 * DAY, "until": T0 + 6 * 3600,
                             "limit": 5,
                             "candidates": 5, "max_s": 1800}, "newest first: `oldest_first` is left out of a `last` ask"
    assert (got["level"], got["last"], got["limit"]) == ("week", True, 5)
    assert (got["found"], got["top5"], got["p"]) == (True, True, 0.83), "the best p within 10 s of the truth, not 0.95 or 0.99"
    assert (got["false"], got["in_range"], got["others"], got["others_found"]) == (1, 3, 1, 1)
    assert (got["moments"], got["moments_found"], got["stopped"], got["windows"], got["seconds"]) == (1, [0], "limit", 40, 2.5)
    assert got["rank"] == 2, "all three are in range; by end time truth+5000 (false) is first, truth+4 (the needle) second, `other` third"

    # the rank counts in-range sightings only: a false one newer than the needle makes it 2, one past `until` does not count
    late = T0 + 6 * 3600 + 100
    api = Scripted(asked([(late, late, 0.9), (truth + 2000, truth + 2000, 0.8), (truth + 2, truth + 2, 0.7)], stopped="limit"))
    got = bench.oracle(api, sq("image-photo-sign-3", "week", last=True, moments=[[truth, truth]]), model)
    assert (got["found"], got["rank"], got["in_range"], got["false"]) == (True, 2, 2, 1)
    # the newest sighting is the needle: 1; sightings given oldest first are ranked by their end time all the same
    got = bench.oracle(Scripted(asked([(truth - 900, truth - 900, 0.8), (truth + 1, truth + 1, 0.9)])), sq("image-photo-sign-3", "week", last=True), model)
    assert (got["found"], got["rank"], got["false"]) == (True, 1, 1)
    # no needle among the sightings (false ones, a sighting of an earlier needle): no rank, and the false ones are told
    api = Scripted(asked([(truth + 3000, truth + 3000, 0.9), (other, other, 0.8), (truth + 2000, truth + 2000, 0.7)], stopped="limit"))
    got = bench.oracle(api, sq("image-photo-sign-3", "week", last=True, moments=[[truth, truth]], others=[[other, other]], since=T0 - 5 * DAY), model)
    assert (got["found"], got["rank"], got["false"], got["in_range"], got["stopped"]) == (False, None, 2, 3, "limit")
    assert bench.outcome(got) == "stopped"

    # the others (part, day, a count): up to 20 sightings, a count up to 50, `oldest_first` kept; a needle only a candidate names
    part = sq("video-screen-error-1", oldest=True, moments=[[truth, truth]])
    api = Scripted(asked([], [(truth - 6, 0.68), (T0 + 99, 0.4)]))
    got = bench.oracle(api, part, model)
    assert api.params[0]["limit"] == 20 and api.params[0]["oldest_first"] is True and api.params[0]["max_s"] == 1800
    assert (got["found"], got["top5"], got["p"], got["moments_found"], got["false"]) == (False, True, 0.68, [], 0)
    count = sq("image-photo-count-1", "day", check="count", moments=[[truth, truth], [truth + 600, truth + 600], [truth + 900] * 2])
    api = Scripted(asked([(truth, truth, 0.9), (truth + 900, truth + 900, 0.9)]))
    got = bench.oracle(api, count, model)
    assert api.params[0]["limit"] == 50 and "oldest_first" not in api.params[0]
    assert (got["found"], got["moments"], got["moments_found"], got["p"]) == (True, 3, [0, 2], 0.9)

    # a needle the ask never names has no p; an absent question has none, and any sighting in range is false
    got = bench.oracle(Scripted(asked([], [(T0 + 99, 0.4)])), sq("video-porch-parcel-1"), model)
    assert (got["found"], got["top5"], got["p"], got["rank"]) == (False, False, None, None)
    got = bench.oracle(Scripted(asked([(T0 + 50, T0 + 60, 0.9)])), sq("video-absent-error-1", absent=True), model)
    assert (got["found"], got["top5"], got["p"], got["false"], got["moments_found"], got["rank"]) == (None, None, None, 1, [], None)
    # a question from before the levels (no level, no last, no others) is a `part` one
    old = sq("video-screen-error-1")
    for key in ("level", "last"):
        del old[key]
    del old["truth"]["others"]
    got = bench.oracle(Scripted(asked([(truth, truth, 0.9)])), old, model)
    assert (got["level"], got["last"], got["limit"], got["others"], got["found"]) == ("part", False, 20, 0, True)


def test_a_models_family_its_jobs_and_the_pilot_come_from_names_abouts_and_question_fields(built):
    names = ("d1-3B", "d1-2B", "d1-Omni-2B", "d3-edge", "pplx-decider-flash-v1.2", "h2o-lightning-4b", "fake")
    assert [bench.family(n) for n in names] == ["d1", "d1", "d1", "d3", "pplx", "h2o", "fake"]
    # a model above 0.15 s a window (mms's estimate from its About) runs one exec at a time; the files arm never does
    jobs = {n: bench.arm_jobs(model_of(n), 3) for n in ("d3-edge", "d1-3B", "h2o-lightning-4b", "d3-mini", "d3-flash", "pplx-decider-flash-v1.2")}
    assert jobs == {"d3-edge": 3, "d1-3B": 3, "h2o-lightning-4b": 3, "d3-mini": 1, "d3-flash": 1, "pplx-decider-flash-v1.2": 1}
    assert bench.arm_jobs(None, 3) == 3 and 0.15 < bench.window_s(decider.known("d3-mini")[1]) < 0.2

    # the text and records questions are asked of one model's mms arm (or of every model's, with no `controls`)
    questions = list(built.plan().questions)
    d13, omni, edge = model_of("d1-3B"), model_of("d1-Omni-2B"), model_of("d3-edge")
    assert bench.arm_questions(questions, d13, "d1-3B") == haystack.questions_for(questions, d13.about)
    assert bench.arm_questions(questions, edge) == haystack.questions_for(questions, edge.about)
    assert {q["type"] for q in bench.arm_questions(questions, edge, "d1-3B")} == {"image", "video"}
    assert {q["type"] for q in bench.arm_questions(questions, omni, "d1-3B")} == {"image", "video", "audio"}
    assert bench.asked_types(edge, "d1-3B") == ["image", "video"] and bench.asked_types(edge, "edge") == ["image", "video"]
    assert bench.asked_types(edge) == ["image", "records", "text", "video"] == bench.asked_types(edge, "d3-edge")

    # the pilot: the media questions of what some model reads (pictures and video with none), and 3 controls: a needle of text, one
    # of records and an absent one, each with an oracle and the least vague hint
    controls = bench.pick_controls(questions)
    assert [(q["type"], q["absent"]) for q in controls][:2] == [("text", False), ("records", False)]
    assert controls[2]["absent"] and controls[2]["type"] in bench.CONTROL_TYPES and all(q["oracle"] for q in controls)
    ids = {q["qid"] for q in controls}
    pilot = bench.pilot_questions(questions, [d13])
    assert [q["qid"] for q in pilot] == [q["qid"] for q in questions if q["type"] in ("image", "video") or q["qid"] in ids]
    assert "audio" in {q["type"] for q in bench.pilot_questions(questions, [d13, omni])}
    assert {q["type"] for q in bench.pilot_questions(questions, [])} == {"image", "video", "text", "records"}
    assert bench.selected(questions, "all", [d13]) == questions and bench.selected(questions, "pilot", [d13]) == pilot
    with pytest.raises(ValueError, match="subset"):
        bench.selected(questions, "half", [])
    two = [pilot[0]["qid"], pilot[-1]["qid"]]
    assert [q["qid"] for q in bench.selected(questions, "pilot", [d13], two)] == two
    with pytest.raises(ValueError, match="not in the pilot questions: audio-mic-alarm-1"):
        bench.selected(questions, "pilot", [d13], ["audio-mic-alarm-1"])
    # on stub questions, whatever the scale: a needle with an oracle and the least vague hint, the qid breaking ties
    stub = [sq("text-config-asof-1", "time", oracle=None), sq("text-note-code-1", "day"), sq("text-note-code-2", "part"),
            sq("records-log-count-1", "part"), sq("records-absent-log-1", "day", absent=True), sq("image-photo-sign-1")]
    assert [q["qid"] for q in bench.pick_controls(stub)] == ["text-note-code-2", "records-log-count-1", "records-absent-log-1"]


def test_agents_gives_each_model_its_questions_and_a_slow_model_one_exec_at_a_time(built, tmp_path, monkeypatch, capsys):
    run = trimmed(built, tmp_path, homes=True)
    calls = []
    monkeypatch.setattr(bench, "run_arm", lambda run_, name, agent, arm, model, questions, home, jobs, halt:
                        calls.append((name, model.name if model else None, jobs, [q["qid"] for q in questions])))
    monkeypatch.setattr(bench, "copy_home", lambda snapshot, target, model: target)
    monkeypatch.setattr(bench, "Daemon", lambda home: types.SimpleNamespace(start=lambda: types.SimpleNamespace(api=None, stop=lambda: None)))
    monkeypatch.setattr(bench, "warm", lambda api, t: 0.0)
    monkeypatch.setattr(bench, "ambient_labels", lambda run_, api: [])
    monkeypatch.setattr(bench, "check_gpu", lambda model: None)
    monkeypatch.setattr(bench, "kept_items", lambda home: set())
    monkeypatch.setattr(bench, "scrub_home", lambda home, kept, **kw: {})
    d13, slow = model_of("d1-3B"), model_of("d3-mini")
    questions = list(run.plan().questions)
    bench.run_agents(run, {"codex": object()}, [d13, slow], jobs=3, controls="d1-3B", subset="pilot")
    pilot = [q["qid"] for q in bench.pilot_questions(questions, [d13, slow])]
    media = [q for q in pilot if q.split("-")[0] in bench.MEDIA]
    assert calls == [("codex", None, 3, pilot), ("codex", "d1-3B", 3, pilot), ("codex", "d3-mini", 1, media)]
    assert len(pilot) < len(questions) and len(media) < len(pilot), "the controls are in the files arm and d1-3B's, not d3-mini's"
    # models.json holds the kinds of data each mms arm is asked about, and the report counts only those as asked
    noted = json.loads((run.folder / "models.json").read_text(encoding="utf-8"))
    assert noted["d3-mini"]["asked"] == ["image", "video"] and noted["d3-mini"]["readable"] == ["image", "records", "text", "video"]
    assert noted["d1-3B"]["asked"] == ["image", "records", "text", "video"]
    bench.note_model(run, d13)  # the decider tier notes a model too, and keeps what the agents wrote
    assert json.loads((run.folder / "models.json").read_text(encoding="utf-8"))["d1-3B"]["asked"] == noted["d1-3B"]["asked"]

    # no `controls`: every model's arm is asked everything, as the tests before this one did
    calls.clear()
    bench.run_agents(run, {"codex": object()}, [slow], jobs=2)
    assert [(c[1], c[2], len(c[3])) for c in calls] == [(None, 2, len(questions)), ("d3-mini", 1, len(bench.arm_questions(questions, slow)))]

    # the command line shows the same plan, and says why a model runs alone
    capsys.readouterr()
    bench.show_agents(run, ["codex"], [d13, slow], [], 3, "d1-3B", "pilot")
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    assert f"codex files {len(pilot)} to run of {len(pilot)}" in lines and f"codex mms d1-3B {len(pilot)} to run of {len(pilot)}" in lines
    assert f"codex mms d3-mini {len(media)} to run of {len(media)}" in lines
    assert any(line.startswith("d3-mini: 0.17 s a window, above 0.15 s: one exec at a time") for line in lines)
    assert not any(line.startswith("d1-3B:") for line in lines) and lines[-1].endswith("over 3 jobs; the GPU is used by each model's daemon")

    # `agents --subset pilot` runs the pilot's questions in the files arm; another subset is a usage error
    stub = Stub()
    monkeypatch.undo()
    monkeypatch.setattr(bench, "real_agents", lambda: {"claude": stub, "codex": stub})
    monkeypatch.setattr(bench.claude_run, "check_login", lambda: None)
    assert bench.main(["agents", str(run.folder), "--agents", "claude", "--models", "none", "--subset", "pilot"]) == 0
    assert sorted(e.question["qid"] for e in stub.execs) == sorted(q["qid"] for q in bench.pilot_questions(questions, []))
    with pytest.raises(SystemExit):
        bench.main(["agents", str(run.folder), "--subset", "half"])


def mms_result(payload) -> dict:
    """A tool result as Codex's tools.json holds an mms tool's: the JSON in one text block."""
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "structured_content": None}


def call(tool: str, arguments: dict, result=None) -> dict:
    return {"tool": tool, "arguments": arguments, "result": result, "error": None, "status": "completed"}


def job_result(job: str, state: str, windows: int, seconds: float, stopped: str | None = None) -> dict:
    """What an ask gives back, as the smoke's tools.json shows: the job, its state, and its progress and stop."""
    return {"job": job, "state": state, "error": None, "sightings": [], "candidates": [], "searched": {"scope": "default"},
            **({"stopped": stopped} if stopped else {}), "warnings": [],
            "progress": {"windows": windows, "of_about": 121, "seconds": seconds, "decider": "ready"}}


# an mms exec of Codex as the smoke recorded them (find, ask, look: the pictures of `look` left out as a hash)...
CODEX_MMS = [
    call("find", {"sources": ["screen"], "since": "2026-11-01T18:00:00", "limit": 20}, mms_result({"results": [], "next": None})),
    call("ask", {"question": "Is a red error screen visible?", "sources": ["screen"], "limit": 1}, mms_result(job_result("j-3c29", "running", 30, 2.0))),
    call("ask", {"job": "j-3c29"}, mms_result(job_result("j-3c29", "done", 121, 9.5, "max_s"))),
    call("ask", {"question": "Is a parcel visible?", "sources": ["porch"]}, mms_result(job_result("j-b7bb", "done", 4, 0.4, "limit"))),
    call("ask", {"question": "Is a sign visible?"}, mms_result(job_result("j-0001", "running", 10, 1.0))),
    call("ask", {"question": "?"}, "ask failed: bad_request: sources"),
    call("look", {"moments": [{"item": 36, "at": 80}]},
         {"content": [{"type": "image", "data": "<60296 characters, sha1 d0c4559cb9da>", "mimeType": "image/jpeg"},
                      {"type": "text", "text": "{\"kind\":\"image\",\"item\":36}"}], "structured_content": None}),
]
# ...Codex without mms (its `view_image` calls come from its rollout, with no result), and Claude's in both arms (its results are the
# blocks, or the text, of the tool)
CODEX_FILES = [call("shell", {"command": "ls"}, "a.mkv"), call("shell", {"command": "ffmpeg -i a.mkv -vf tile m.jpg"}, ""),
               call("view_image", {"path": "C:\\w\\m.jpg"}), call("shell", {"command": "ffmpeg ..."}, ""), call("view_image", {"path": "c.png"})]
CLAUDE_FILES = [call("Glob", {"pattern": "*"}, "a.mkv"), call("Bash", {"command": "ffmpeg -i a.mkv m.jpg"}, ""),
                call("Read", {"file_path": "C:\\w\\montage.jpg"}, [{"type": "text", "text": "[image]"}]),
                call("Read", {"file_path": "/data/notes/n.md"}, "text"), call("Read", {"file_path": "C:\\w\\FRAME.PNG"}),
                call("Grep", {"pattern": "ERR"}, "x"), call("Bash", {"command": "ffmpeg -y"}, "")]
CLAUDE_MMS = [call("ask", {"question": "Is a sign visible?"}, [{"type": "text", "text": json.dumps(job_result("j-c1", "done", 50, 3.0, "range_end"))}]),
             call("ask", {"question": "again"}, json.dumps(job_result("j-c2", "done", 7, 0.5, "cancelled"))),
             call("find", {"sources": ["phone"]}, [{"type": "text", "text": "{}"}]), call("look", {"moments": []}, "ok")]


def test_how_an_exec_worked_is_read_from_its_calls_whatever_the_agent(tmp_path):
    work = bench.work_of(CODEX_MMS)
    # a job counts once, by its last result (the second call collected the first's), and a job still running says so
    assert work == {"jobs": 3, "windows": 135, "seconds": 10.9, "views": 0, "shells": 0,
                    "stopped": {"max_s": 1, "limit": 1, "running": 1}}
    assert bench.work_of(CODEX_FILES) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 2, "shells": 3}
    assert bench.work_of(CLAUDE_FILES) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 2, "shells": 2}, \
        "a `Read` of a picture is a view whatever the case of its suffix; of a note it is not"
    assert bench.work_of(CLAUDE_MMS) == {"jobs": 2, "windows": 57, "seconds": 3.5, "views": 0, "shells": 0,
                                        "stopped": {"range_end": 1, "cancelled": 1}}
    assert bench.work_of([]) == {"jobs": 0, "windows": 0, "seconds": 0, "stopped": {}, "views": 0, "shells": 0}
    # a result that is not JSON, or not an object, or an ask that gave none, counts as no job
    for result in (None, "", "ask failed", "[1, 2]", [], [{"type": "image"}], {"content": None}, 7):
        assert bench.result_json(result) is None, result
        assert bench.work_of([call("ask", {"question": "?"}, result)])["jobs"] == 0
    assert bench.result_json({"content": [], "structured_content": {"job": "j-s"}}) == {"job": "j-s"}
    assert bench.result_json(["{\"a\": 1}"]) == {"a": 1}
    # a look of several moments has a JSON block for each (and a picture between): result_jsons gives all, result_json the first
    blocks = {"content": [{"type": "image"}, {"type": "text", "text": "{\"item\": 1}"}, {"type": "text", "text": "not json"},
                          {"type": "text", "text": "{\"item\": 2}"}, {"type": "text", "text": "[3]"}]}
    assert bench.result_jsons(blocks) == [{"item": 1}, {"item": 2}] and bench.result_json(blocks) == {"item": 1}
    assert bench.result_jsons(None) == [] and bench.result_jsons({"structured_content": {"job": "j"}, "content": blocks["content"]}) == [{"job": "j"}]

    # a tools.json of an exec is read for a result made before results held `work`; a missing or a bad one is None
    folder = tmp_path / "execs"
    folder.mkdir()
    (folder / "tools.json").write_text(json.dumps(CODEX_MMS), encoding="utf-8")
    assert bench.work_from_file(folder / "tools.json") == work
    assert bench.work_from_file(folder / "nothing.json") is None
    (folder / "bad.json").write_text("{", encoding="utf-8")
    (folder / "obj.json").write_text("{}", encoding="utf-8")
    assert bench.work_from_file(folder / "bad.json") is None and bench.work_from_file(folder / "obj.json") is None


def test_a_result_holds_its_level_and_how_it_worked(built, tmp_path):
    q = sq("video-screen-error-1", "day", last=True)
    e = bench.Exec("codex", "mms", FAKE_MODEL, q, "", tmp_path / "execs" / "codex" / "mms-fake" / q["qid"], tmp_path / "w", tmp_path / "home",
                   built.data, built.bin, (), 7)
    got = bench.record(e, {"answer": "13:00", "seconds": 4.0, "problem": None, "tools": CODEX_MMS})
    assert (got["level"], got["last"], got["passed"]) == ("day", True, True)
    assert got["tools"] == {"find": 1, "ask": 5, "look": 1} and got["work"]["windows"] == 135 and got["work"]["stopped"]["max_s"] == 1
    plain = sq("video-screen-error-1")
    del plain["level"], plain["last"]
    got = bench.record(dataclasses.replace(e, question=plain), {"answer": "13:00", "seconds": 4.0, "problem": None, "tools": []})
    assert (got["level"], got["last"], got["work"]["jobs"]) == ("part", False, 0)


def level_of_pairs(files_correct, mms_correct, n, files_s=100.0, mms_s=100.0, files_k=1000.0, mms_k=1000.0, b=None, c=None) -> dict:
    """A level as `by_level` gives it, from the counts: n pairs, each arm's correct answers, the seconds and the tokens an exec
    spends (so an arm's cost per correct answer is what its correct ones divide)."""
    def arm(correct, s, k):
        return {"correct": correct, "n": n, "accuracy": correct / n, "seconds": s * n if s is not None else 0, "median_s": s,
                "tokens_all": k * n if k is not None else None, "tokens_n": n if k is not None else 0, "tokens_mean": k}

    b = max(mms_correct - files_correct, 0) if b is None else b
    c = max(files_correct - mms_correct, 0) if c is None else c
    both_right = files_correct - c
    return {"n": n, "n_files": n, "diff": (b - c) / n, "ci": bench.interval(both_right, b, c, n - both_right - b - c), "b": b, "c": c,
            "files": arm(files_correct, files_s, files_k), "mms": arm(mms_correct, mms_s, mms_k)}


def test_the_crossover_is_the_first_level_with_enough_pairs_where_mms_leads_by_20_points_or_costs_half_without_being_worse():
    level = level_of_pairs
    # nothing: 10 more right out of 20 is no lead, a level of 10 pairs or more is needed, and an exact time is no level of the hint
    assert bench.crossover({"part": level(10, 11, 20), "day": level(10, 11, 20), "week": level(10, 11, 20)}) is None
    assert bench.crossover({"part": level(0, 5, 9)}) is None and bench.thin_levels({"part": level(0, 5, 9), "day": level(0, 5, 20)}) == {"part": 9}
    assert bench.crossover({"time": level(0, 20, 20, 10, 1)}) is None and bench.crossover({}) is None
    assert bench.crossover({"part": {**level(1, 1, 10), "n": 0}}) is None and bench.thin_levels({"part": {**level(1, 1, 10), "n": 0}}) == {}
    # a lead of 20 points is a lead (0.8 - 0.6 is 0.20000000000000007, and 0.6 - 0.4 is 0.1999...96)
    assert bench.crossover({"part": level(12, 16, 20)})["by"] == ["accuracy"]
    assert bench.crossover({"part": level(10, 11, 20), "day": level(8, 12, 20)})["level"] == "day"
    got = bench.crossover({"part": level(10, 14, 20), "day": level(5, 15, 20)})
    assert (got["level"], got["by"], got["lead"]) == ("part", ["accuracy"], pytest.approx(0.2))
    assert got["seconds"] == pytest.approx(10 / 14) == pytest.approx(got["tokens"]), "per correct answer: 14 right for what 10 cost"
    # the cost branch: half the seconds, or the tokens, per correct answer, with mms no less accurate
    assert bench.crossover({"part": level(10, 10, 20, 100, 90), "day": level(10, 10, 20, 100, 40)})["level"] == "day"
    got = bench.crossover({"week": level(10, 10, 20, 100, 100, 1000, 500)})
    assert (got["level"], got["by"], got["lead"], got["seconds"], got["tokens"]) == ("week", ["cost"], 0.0, 1.0, 0.5)
    both = bench.crossover({"day": level(8, 16, 20, 100, 40)})
    assert both["by"] == ["accuracy", "cost"] and both["seconds"] == pytest.approx(0.2)
    # per correct answer, not per exec: mms twice as fast with half the answers right costs the same per right answer
    assert bench.crossover({"day": level(16, 16, 20, 100, 50), "week": level(16, 8, 20, 100, 50, b=0, c=8)}) ["level"] == "day"
    assert bench.crossover({"week": level(16, 8, 20, 100, 50, b=0, c=8)}) is None
    # less accurate and cheaper is no crossover: a difference of -0.40 at a quarter of the cost, and -0.05 with an interval below it
    worse = level(18, 10, 20, 100, 25, 1000, 250, b=0, c=8)
    assert worse["diff"] == pytest.approx(-0.4) and bench.crossover({"day": worse}) is None
    assert bench.crossover({"day": level(10, 10, 20, 100, 25, b=1, c=2)}) is None, "-0.05, and too few discordant pairs for an interval"
    close = level(160, 158, 200, 100, 25, b=8, c=10)
    assert close["diff"] < 0 and close["ci"][0] > -bench.NOT_WORSE and bench.crossover({"day": close})["by"] == ["cost"]
    assert bench.crossover({"day": level(160, 140, 200, 100, 25, b=0, c=20)}) is None, "-10 points, interval below -0.10"
    # a cost per correct answer needs MIN_CORRECT right answers in each arm, and tokens that every exec gave
    assert bench.crossover({"day": level(4, 4, 10, 100, 10)}) is None
    nothing = level(10, 10, 20, None, None, None, None)
    assert bench.crossover({"part": nothing}) is None
    # the text says which branch fired, and why there is none
    cost_only = bench.crossover({"week": level(10, 10, 20, 100, 100, 1000, 500)})
    assert bench.crossover_text(cost_only) == "`week` by cost (accuracy +0 pts, 1.00x the seconds per correct answer, 0.50x the tokens per correct answer)"
    assert bench.crossover_text(bench.crossover({"day": level(8, 12, 20)})).startswith("`day` by accuracy (accuracy +20 pts")
    assert "too few paired questions at part (9)" in bench.crossover_text(None, {"part": 9})
    assert bench.crossover_text(None).startswith("none: at no level with 10 or more paired questions")


def test_the_difference_gets_newcombes_interval_only_from_ten_pairs_right_in_one_arm_only_and_mcnemar_speaks_for_the_rest():
    def rows(a, b, c, d):
        out = []
        for mms, files, count in ((True, True, a), (True, False, b), (False, True, c), (False, False, d)):
            def result(passed):
                return {"passed": passed, "absent": False, "answer": "x", "abstained": False, "seconds": 10.0, "calls": 1, "problem": None,
                        "tokens": None, "cost_usd": None, "agent": "codex", "work": None, "tools": {}}
            out += [({"type": "image", "level": "part"}, result(files), result(mms)) for _ in range(count)]
        return out

    # the case that a percentile bootstrap got wrong: 4 right with mms only, 0 the other way, 16 pairs: [.06, .50] with p = .125
    four = bench.paired(rows(6, 4, 0, 6))
    assert (four["b"], four["c"], four["p"], four["ci"], four["diff"]) == (4, 0, 0.125, None, 0.25)
    assert bench.newcombe(6, 4, 0, 6)[0] > 0, "the interval would exclude 0 where McNemar does not"
    assert bench.diff_cell(four) == "+25 pts (too few discordant pairs: b 4, c 0)"
    assert bench.lean(rows(6, 4, 0, 6))["ci"] is None and bench.paired(rows(6, 5, 4, 5))["ci"] is None, "9 discordant pairs"
    # from 10: the interval, in points, beside b, c and the p
    ten = bench.paired(rows(5, 10, 0, 5))
    assert ten["ci"] == pytest.approx([0.223, 0.675], abs=1e-3) and ten["p"] == pytest.approx(0.00195, abs=1e-4) and ten["diff"] == 0.5
    assert bench.diff_cell(ten) == "+50 pts [+22, +67]"
    even = bench.paired(rows(10, 5, 5, 10))
    assert even["ci"][0] < 0 < even["ci"][1] and even["p"] == 1.0 and bench.diff_cell(even).startswith("+0 pts [-")
    assert bench.paired([])["ci"] is None and bench.paired([])["diff"] is None and bench.diff_cell(bench.paired([])) == "-"
    # it agrees with McNemar where one speaks plainly: p of 0.2 or more has 0 inside, p of 0.01 or less has it outside
    for n in (12, 20, 30):
        for b in range(n + 1):
            for c in range(n + 1 - b):
                if b + c < bench.MIN_DISCORDANT:
                    assert bench.interval(n - b - c, b, c, 0) is None
                    continue
                for a in range(0, n - b - c + 1, max(1, (n - b - c) // 3)):
                    low, high = bench.newcombe(a, b, c, n - a - b - c)
                    p = bench.mcnemar(b, c)
                    assert not (p >= 0.2 and (low > 0 or high < 0)) and not (p <= 0.01 and low <= 0 <= high), (a, b, c, p, low, high)
    with pytest.raises(ValueError):
        bench.newcombe(0, 0, 0, 0)
    assert bench.wilson(0, 0) == (0.0, 1.0) and bench.wilson(5, 10)[0] < 0.5 < bench.wilson(5, 10)[1]


def test_a_pooled_difference_is_the_mean_of_the_paired_differences_whatever_the_files_arm_is_counted_once():
    def result(passed):
        return {"passed": passed, "absent": False, "answer": "x", "abstained": False, "seconds": 10.0, "calls": 1, "problem": None,
                "tokens": None, "cost_usd": None, "agent": "codex", "work": None}

    lost, won = result(False), result(True)  # one files result that was wrong, paired with three models; one that was right, with one
    q = {"type": "image", "level": "day"}
    pairs = [(q, lost, result(True)), (q, lost, result(True)), (q, lost, result(False)), (q, won, result(True))]
    got = bench.lean(pairs)
    assert (got["n"], got["n_files"]) == (4, 2) and (got["b"], got["c"]) == (2, 0)
    assert got["diff"] == pytest.approx(0.5), "the mean of (+1, +1, 0, 0), not 0.75 - 0.5 from a files side counted once"
    assert got["files"]["accuracy"] == pytest.approx(0.5) and got["mms"]["accuracy"] == pytest.approx(0.75)


def test_claudes_tokens_count_the_cache_it_read_and_wrote_as_codexs_input_already_does():
    claude = {"agent": "claude", "tokens": {"input": 100, "output": 50, "cache_read": 1000, "cache_write": 200}}
    codex = {"agent": "codex", "tokens": {"input": 1300, "output": 50, "cache_read": 1000, "cache_write": 0}}
    assert bench.token_total(claude) == 1350 == bench.token_total(codex) and bench.token_total({"agent": "claude", "tokens": None}) is None
    assert bench.token_total({"agent": "claude", "tokens": {"input": 5, "output": 1, "cache_read": None, "cache_write": None}}) == 6
    got = {**claude, "passed": True, "seconds": 2.0, "absent": False, "answer": "x", "abstained": False, "calls": 1, "problem": None,
           "cost_usd": None, "work": None}
    side = bench.side([got, {**got, "passed": False, "tokens": None}])
    assert (side["tokens_mean"], side["tokens_all"], side["tokens_n"], side["correct"], side["seconds"]) == (1350, 1350, 1, 1, 4.0)
    assert bench.per_correct([got])["tokens_k"] == pytest.approx(1.35) and bench.per_correct([got, {**got, "tokens": None}])["tokens_k"] is None
    assert bench.kilo(side) == "1" and bench.kilo(bench.side([])) == "-"


def rec(model: str, qid: str, found: bool, top5: bool, p: float | None, *, false: int = 0, level: str | None = "part",
        stopped: str = "range_end", absent: bool = False, candidates=()) -> dict:
    """A stub oracle record (the decider tier's), of the sort `oracle` writes; `level=None` leaves out `level` and `p`, as a record
    from before them does."""
    type_, *rest = qid.split("-")
    out = {"qid": qid, "type": type_, "template": "-".join(rest[:-1]), "absent": absent, "model": model, "question": "?", "source": "phone",
           "since": T0, "until": T0 + 6 * 3600, "found": found, "top5": top5, "in_range": 0, "false": false, "windows": 100,
           "seconds": 6.0, "wall_s": 6.0, "stopped": stopped, "warnings": [], "sightings": [], "candidates": list(candidates)}
    if level is not None:
        out.update(level=level, p=p)
    return out


def test_a_miss_is_the_haystacks_only_when_nobody_found_the_needle_and_two_families_scanned_it_without_it_in_their_top_5():
    def found(m):
        return rec(m, "video-screen-error-1", True, True, 0.97)

    def near(m):
        return rec(m, "video-screen-error-1", False, True, 0.68)

    def far(m):
        return rec(m, "video-screen-error-1", False, False, None)

    def stopped(m):  # a `last` ask that met its LAST_LIMIT sightings, false ones newer than the needle, before it reached it
        return rec(m, "video-screen-error-1", False, False, None, false=1, stopped="limit")

    def cut(m):
        return rec(m, "video-screen-error-1", False, False, None, stopped="max_s")

    # one model: the decider tier cannot tell a miss from a haystack fault; a near miss is below its model's hit
    assert bench.verdicts([far("d1-3B")]) == {"d1-3B": "one model ran: the agent arm's look, or a second family, tells a miss from a haystack fault"}
    assert bench.verdicts([near("d1-3B")]) == {"d1-3B": "below its model's hit (calibration)"}
    assert bench.verdicts([found("d1-3B")]) == {}
    # two models of one family are one family
    assert bench.verdicts([far("d1-3B"), far("d1-Omni-2B")]) == {n: bench.ONE_FAMILY for n in ("d1-3B", "d1-Omni-2B")}
    assert bench.ONE_FAMILY.startswith("one family ran:")
    # nobody found it, two families scanned it, neither had it in its top 5: the haystack is suspect
    assert bench.verdicts([far("d1-3B"), far("d3-edge")]) == {"d1-3B": "haystack suspect", "d3-edge": "haystack suspect"}
    # a model found it: every miss is that model's miss, of whichever family (a model of the same family included)
    assert bench.verdicts([found("d1-3B"), far("d3-edge")]) == {"d3-edge": "this model's miss"}
    assert bench.verdicts([found("d1-Omni-2B"), far("d1-3B"), far("d3-edge")]) == {"d1-3B": bench.MISS, "d3-edge": bench.MISS}
    assert bench.verdicts([far("d1-3B"), far("d1-Omni-2B"), found("d3-edge")]) == {"d1-3B": bench.MISS, "d1-Omni-2B": bench.MISS}
    assert bench.verdicts([found("d1-Omni-2B"), far("d1-3B")]) == {"d1-3B": bench.MISS}
    # another family had it in its top 5: this model's miss; a model in the top 5 is below its hit
    assert bench.verdicts([near("d1-3B"), far("d3-edge")]) == {"d1-3B": "below its model's hit (calibration)", "d3-edge": "this model's miss"}
    # a scan that ended before the needle is told as it is, and it does not make the haystack suspect or count as a family that ran
    assert bench.verdicts([stopped("d1-3B"), far("d3-edge")]) == {"d1-3B": bench.STOPPED, "d3-edge": bench.ONE_MODEL}
    assert bench.verdicts([cut("pplx-decider-flash-v1.2"), far("d3-edge"), far("d1-3B")]) == {
        "pplx-decider-flash-v1.2": bench.CUT, "d3-edge": bench.SUSPECT, "d1-3B": bench.SUSPECT}
    assert bench.verdicts([stopped("d1-3B"), found("d3-edge")]) == {"d1-3B": bench.STOPPED}
    assert bench.STOPPED == "stopped on a false sighting before the needle" and bench.CUT == "cut at max_s before the needle"
    # the outcomes: a needle with p of 0.5 or more is a near miss even outside the top 5; below 0.5 it is a miss
    assert bench.outcome(rec("d1-3B", "video-screen-error-1", False, False, 0.5)) == "near"
    assert bench.outcome(rec("d1-3B", "video-screen-error-1", False, False, 0.49)) == "missed"
    assert bench.outcome(rec("d1-3B", "video-screen-error-1", False, False, None)) == "missed"
    assert bench.outcome(found("d1-3B")) == "found" and bench.outcome(stopped("d1-3B")) == "stopped" and bench.outcome(cut("d1-3B")) == "cut"
    # the limit met with a sighting that was no false one, or the needle's own candidate in the top 5, are not a scan that ended early
    assert bench.outcome(rec("d1-3B", "video-screen-error-1", False, False, None, stopped="limit")) == "missed"
    assert bench.outcome(rec("d1-3B", "video-screen-error-1", False, True, 0.3, false=1, stopped="limit")) == "near"


NO_TOKENS = {"passed": True, "seconds": 1.0, "tokens": None, "work": None}


def wk(jobs: int = 0, windows: int = 0, seconds: float = 0.0, stopped: dict | None = None, views: int = 0, shells: int = 0) -> dict:
    return {"jobs": jobs, "windows": windows, "seconds": seconds, "stopped": stopped or {}, "views": views, "shells": shells}


# (passed, seconds, tool counts, work or None, (input, output) tokens, problem) of each exec, by qid
FILES_ARM = {
    "image-photo-sign-1": (True, 40, {"shell": 6, "view_image": 5}, wk(views=5, shells=6), (30000, 2000), None),
    "image-photo-sign-2": (False, 90, {"shell": 10, "view_image": 9}, wk(views=9, shells=10), (60000, 3000), None),
    "image-photo-sign-3": (False, 900, {"shell": 20, "view_image": 14}, wk(views=14, shells=20), (90000, 1000), "timeout: no answer in 900 s"),
    "video-screen-error-1": (True, 30, {"shell": 4, "view_image": 3}, wk(views=3, shells=4), (20000, 1000), None),
    "video-screen-error-2": (True, 80, {"shell": 9, "view_image": 8}, wk(views=8, shells=9), (50000, 2000), None),
    "video-porch-parcel-1": (False, 120, {"shell": 15, "view_image": 12}, wk(views=12, shells=15), (70000, 1500), None),
    "text-note-code-1": (True, 10, {"shell": 2}, wk(shells=2), (8000, 500), None)}
MMS_D13 = {
    "image-photo-sign-1": (True, 30, {"ask": 1, "find": 1, "look": 2}, wk(1, 50, 3.0, {"range_end": 1}), (20000, 1000), None),
    "image-photo-sign-2": (True, 20, {"ask": 2, "look": 1}, wk(1, 400, 25.0, {"max_s": 1}), (25000, 1500), None),
    "image-photo-sign-3": (True, 25, {"ask": 1, "look": 3}, wk(1, 90, 6.0, {"limit": 1}), (22000, 1000), None),
    "video-screen-error-1": (True, 35, {"find": 2, "look": 4}, wk(), (40000, 1200), None),
    "video-screen-error-2": (True, 50, {"ask": 1, "look": 2}, wk(1, 200, 12.0, {"range_end": 1}), (30000, 1000), None),
    "video-porch-parcel-1": (False, 60, {"ask": 1}, wk(1, 300, 20.0, {"max_s": 1}), (28000, 500), None),
    "text-note-code-1": (True, 8, {"find": 1}, wk(), (6000, 300), None)}
MMS_EDGE = {  # the last two are results from before `work` was kept: the first has no tools.json, the second one is on disk
    "image-photo-sign-1": (True, 38, {"ask": 1, "look": 1}, wk(1, 60, 5.0, {"range_end": 1}), (21000, 900), None),
    "image-photo-sign-2": (False, 30, {"ask": 1}, wk(1, 450, 80.0, {"max_s": 1}), (15000, 400), None),
    "video-screen-error-1": (True, 33, {"ask": 1, "look": 2}, None, (20000, 800), None),
    "video-screen-error-2": (True, 70, {"ask": 3, "look": 1}, None, (50000, 2000), None)}
EDGE_TOOLS = [call("ask", {"question": "Is a red error screen visible?"}, mms_result(job_result("j-a", "running", 100, 20.0))),
              call("ask", {"job": "j-a"}, mms_result(job_result("j-a", "done", 300, 60.0, "max_s"))),
              call("ask", {"question": "Again?"}, mms_result(job_result("j-b", "done", 80, 10.0, "limit"))),
              call("look", {"moments": [{"item": 1, "at": 3}]}, {"content": [{"type": "image", "data": "<5000 characters, sha1 abc>"}]})]
DECIDERS = {
    "d1-3B": [rec("d1-3B", "image-photo-sign-1", True, True, 0.96), rec("d1-3B", "image-photo-sign-3", False, False, None, false=1, level="week"),
              rec("d1-3B", "video-screen-error-1", True, True, 0.97), rec("d1-3B", "video-porch-parcel-1", False, False, None, level="week"),
              rec("d1-3B", "video-absent-error-1", None, None, None, absent=True),
              # a record from before the levels and the needle's p: the plan's level, and the p of the candidate at the truth
              rec("d1-3B", "image-photo-sign-2", False, True, None, level=None, candidates=[{"t": config.iso(T0 + 3602), "p": 0.68}])],
    "d3-edge": [rec("d3-edge", "image-photo-sign-1", True, True, 0.94), rec("d3-edge", "image-photo-sign-2", True, True, 0.90, level="day"),
                rec("d3-edge", "image-photo-sign-3", False, False, None, level="week"),
                rec("d3-edge", "video-screen-error-1", False, False, 0.3)]}


def stub_run(tmp_path, monkeypatch) -> bench.Run:
    """A run folder of hand-written results over stub questions that carry the new fields, with a plan that is theirs: two models
    asked the media questions of one agent, and d1-3B also its text control."""
    questions = [sq("image-photo-sign-1", "part"), sq("image-photo-sign-2", "day"),
                 sq("image-photo-sign-3", "week", last=True, others=[[T0 - 2 * DAY] * 2]), sq("video-screen-error-1", "part"),
                 sq("video-screen-error-2", "day"), sq("video-porch-parcel-1", "week", last=True), sq("text-note-code-1", "part")]
    by_qid = {q["qid"]: q for q in questions}
    days = (date(2026, 10, 29), date(2026, 10, 30))
    plan = haystack.Plan(1, "tiny", days, (), {}, tuple(questions))
    monkeypatch.setattr(bench.Run, "plan", lambda self: plan)
    run = bench.Run(tmp_path / "stub-run", 1, "tiny", days, tmp_path / "data", tmp_path)
    run.folder.mkdir()
    (run.folder / "run.json").write_text(json.dumps({"mms_commit": "abc123"}), encoding="utf-8")
    (run.folder / "models.json").write_text(json.dumps({
        "d1-3B": {"slug": "d1-3B", "readable": ["image", "records", "text", "video"], "asked": ["image", "records", "text", "video"]},
        "d3-edge": {"slug": "d3-edge", "readable": ["image", "records", "text", "video"], "asked": ["image", "video"]}}), encoding="utf-8")
    for folder, results in (("files", FILES_ARM), ("mms-d1-3B", MMS_D13), ("mms-d3-edge", MMS_EDGE)):
        for qid, (passed, seconds, tools, work, tokens, problem) in results.items():
            q = by_qid[qid]
            out = {"qid": qid, "type": q["type"], "template": q["template"], "level": q["level"], "last": q["last"], "absent": False,
                   "seed": 1, "agent": "codex", "arm": folder.split("-")[0], "model": None, "answer": "13:00" if passed else "14:00",
                   "passed": passed, "abstained": False, "problem": problem, "seconds": seconds, "turns": 1, "denials": None,
                   "calls": sum(tools.values()), "tools": tools, "scope_all": 0, "cost_usd": None, "used_percent": 10.0,
                   "tokens": {"input": tokens[0], "output": tokens[1], "cache_read": 0, "cache_write": 0},
                   "exec": f"execs/codex/{folder}/{qid}", "at": "2026-11-02T12:00:00", **({"work": work} if work is not None else {}),
                   **({"gpu_estimate_s": (1200.0 if qid == "video-porch-parcel-1" else 400.0) if q["level"] == "week" else 20.0,
                       "held_s": 4.0, "exclusive": q["level"] == "week", "beyond_timeout": qid == "video-porch-parcel-1"}
                      if folder == "mms-d1-3B" else {})}
            path = run.folder / "results" / "codex" / folder / f"{qid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out), encoding="utf-8")
    on_disk = run.folder / "execs" / "codex" / "mms-d3-edge" / "video-screen-error-2"
    on_disk.mkdir(parents=True)
    (on_disk / "tools.json").write_text(json.dumps(EDGE_TOOLS), encoding="utf-8")
    for model, records in DECIDERS.items():
        folder = run.folder / "results" / "decider" / model
        folder.mkdir(parents=True)
        (folder / "warm.json").write_text(json.dumps({"model": model, "seconds": 20.0, "at": "2026-11-02T12:00:00"}), encoding="utf-8")
        for r in records:
            (folder / f"{r['qid']}.json").write_text(json.dumps(r), encoding="utf-8")
    return run


def test_the_report_gives_the_levels_how_each_arm_worked_and_what_a_correct_answer_cost(tmp_path, monkeypatch):
    run = stub_run(tmp_path, monkeypatch)
    thin = bench.numbers([run])  # with its two pairs a level, nothing is told of a crossover
    assert thin["agents"]["codex"]["d1-3B"]["crossover"] is None and thin["agents"]["codex"]["d1-3B"]["thin"] == {"part": 2, "day": 2, "week": 2}
    assert "too few paired questions at part (2), day (2), week (2)" in bench.markdown(thin)
    monkeypatch.setattr(bench, "MIN_LEVEL_N", 2)  # the stub's levels have two pairs each
    figures = bench.numbers([run])
    d13, edge = figures["agents"]["codex"]["d1-3B"], figures["agents"]["codex"]["d3-edge"]
    assert (d13["questions"], d13["pending"], d13["all"]["n"], d13["media"]["n"]) == (7, 0, 7, 6)
    assert (edge["questions"], edge["pending"], edge["all"]["n"], edge["media"]["n"]) == (6, 2, 4, 4), "d3-edge is asked media alone"
    assert (d13["media"]["files"]["correct"], d13["media"]["mms"]["correct"], d13["media"]["b"], d13["media"]["c"]) == (3, 5, 2, 0)

    # accuracy, seconds, tokens and timeouts at each level of the hint
    assert list(d13["levels"]) == ["part", "day", "week"]
    part, day, week = d13["levels"]["part"], d13["levels"]["day"], d13["levels"]["week"]
    assert (part["n"], part["files"]["correct"], part["mms"]["correct"], part["files"]["median_s"], part["mms"]["median_s"]) == (2, 2, 2, 35.0, 32.5)
    assert (part["files"]["tokens_mean"], part["mms"]["tokens_mean"]) == (26500, 31100)
    assert (day["files"]["correct"], day["mms"]["correct"], day["diff"], day["files"]["median_s"], day["mms"]["median_s"]) == (1, 2, 0.5, 85.0, 35.0)
    assert (day["files"]["tokens_mean"], day["mms"]["tokens_mean"]) == (57500, 28750)
    assert (week["files"]["correct"], week["mms"]["correct"], week["files"]["timeouts"], week["mms"]["timeouts"]) == (0, 1, 1, 0)
    assert d13["crossover"] == {"level": "day", "by": ["accuracy"], "lead": 0.5, "seconds": None, "tokens": None}, \
        "two right answers in an arm make no cost per correct answer"
    assert (part["b"], part["c"], part["ci"], day["b"], day["c"], day["ci"]) == (0, 0, None, 1, 0, None)
    assert [lv["files"]["correct"] for lv in edge["levels"].values()] == [2, 1] and edge["crossover"] is None
    # pooled over the models: the files results count once (4 pairs at `day`, 2 files results), the mms results of both models
    pooled = figures["pooled"]["codex"]
    assert (pooled["levels"]["day"]["n"], pooled["levels"]["day"]["n_files"], pooled["levels"]["day"]["files"]["correct"],
            pooled["levels"]["day"]["mms"]["correct"], pooled["levels"]["day"]["diff"]) == (4, 2, 1, 3, 0.25)
    assert pooled["levels"]["day"]["mms"]["median_s"] == 40.0 and pooled["levels"]["week"]["n"] == 2
    assert pooled["crossover"]["level"] == "day" and pooled["crossover"]["lead"] == 0.25 and pooled["crossover"]["by"] == ["accuracy"]

    # how each arm worked, per exec on the paired media questions
    mms, files = d13["work"]["mms"], d13["work"]["files"]
    assert (mms["n"], mms["ask_execs"], mms["asks"], mms["finds"], mms["looks"], mms["known"]) == (6, 5, 1.0, 0.5, 2.0, 6)
    assert (mms["windows"], mms["gpu_s"], mms["max_s"], mms["views"]) == (pytest.approx(1040 / 6), 11.0, 2, 0.0)
    assert (files["n"], files["ask_execs"], files["views"], files["shells"]) == (6, 0, 8.5, pytest.approx(64 / 6))
    # a result with no `work` gets it from its exec's tools.json (two jobs, 380 windows, 70 GPU s), one with neither is unknown
    slim = edge["work"]["mms"]
    assert (slim["n"], slim["known"], slim["ask_execs"], slim["asks"], slim["looks"]) == (4, 3, 4, 1.5, 1.0)
    assert (slim["windows"], slim["gpu_s"], slim["max_s"]) == (pytest.approx(890 / 3), pytest.approx(155 / 3), 2)

    # seconds, tokens (k) and GPU seconds per correct answer: the GPU seconds only when every result says
    assert d13["cost"]["files"]["s"] == 420.0 and d13["cost"]["files"]["tokens_k"] == pytest.approx(330.5 / 3)
    assert (d13["cost"]["mms"]["s"], d13["cost"]["mms"]["tokens_k"], d13["cost"]["mms"]["gpu_s"]) == (44.0, pytest.approx(34.24), 13.2)
    assert (edge["cost"]["mms"]["correct"], edge["cost"]["mms"]["s"], edge["cost"]["mms"]["gpu_s"]) == (3, 57.0, None)
    assert bench.per_correct([])["s"] is None and bench.per_correct([{**NO_TOKENS, "passed": False}])["tokens_k"] is None

    # one row per question, in the plan's order: the files arm's cell and each model's (none where it was not asked)
    rows = figures["appendix"]["codex"]
    assert [r["qid"] for r in rows][:3] == ["image-photo-sign-1", "image-photo-sign-2", "image-photo-sign-3"] and len(rows) == 7
    assert rows[0] == {"seed": 1, "qid": "image-photo-sign-1", "level": "part", "type": "image",
                       "files": {"passed": True, "seconds": 40, "problem": None, "beyond": False, "views": 5, "asks": 0},
                       "mms": {"d1-3B": {"passed": True, "seconds": 30, "problem": None, "beyond": False, "views": 0, "asks": 1},
                              "d3-edge": {"passed": True, "seconds": 38, "problem": None, "beyond": False, "views": 0, "asks": 1}}}
    assert rows[2]["files"]["problem"] == "timeout" and list(rows[2]["mms"]) == ["d1-3B"] and list(rows[6]["mms"]) == ["d1-3B"]
    assert rows[5]["mms"]["d1-3B"]["passed"] is False and rows[5]["mms"]["d1-3B"]["beyond"] is True and mms["beyond"] == 1
    json.dumps(figures)

    # the markdown holds each table
    text = bench.markdown(figures)
    for line in (
            "## codex: by how vague the time hint is (the media questions, paired)",
            "| model | level | paired | without mms | with mms | difference (95% CI) | b / c | McNemar p | median s (files / mms) | "
            "tokens k per exec (files / mms) | timed out (files / mms) |",
            "| d1-3B | part | 2 | 2/2 (100%) | 2/2 (100%) | +0 pts (too few discordant pairs: b 0, c 0) | 0 / 0 | 1.000 | 35.0 / 32.5 | 26 / 31 | 0 / 0 |",
            "| d1-3B | day | 2 | 1/2 (50%) | 2/2 (100%) | +50 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 85.0 / 35.0 | 58 / 29 | 0 / 0 |",
            "| d1-3B | week | 2 | 0/2 (0%) | 1/2 (50%) | +50 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 510.0 / 42.5 | 81 / 26 | 1 / 0 |",
            "| all models | day | 4 | 1/2 (50%) | 3/4 (75%) | +25 pts (too few discordant pairs: b 1, c 0) | 1 / 0 | 1.000 | 85.0 / 40.0 | 58 / 31 | 0 / 0 |",
            "- d1-3B: `day` by accuracy (accuracy +50 pts)",
            "- d3-edge: none: at no level with 2 or more paired questions does mms lead by 20 points, or cost half as much per correct answer",
            "- all models: `day` by accuracy (accuracy +25 pts)",
            "## How each arm worked (the media questions, per exec)",
            "| agent | model | arm | execs | called ask | asks / finds / looks | ask windows | ask GPU s | asks stopped at max_s | "
            "GPU s estimated | held s (run alone) | estimated past the timeout | pictures viewed | shell commands |",
            "| codex | d1-3B | files | 6 | - | - | - | - | - | - | - | - | 8.5 | 10.7 |",
            "| codex | d1-3B | mms | 6 | 5/6 (83%) | 1.0 / 0.5 / 2.0 | 173.3 | 11.0 | 2 | 280.0 | 4.0 (2 alone) | 1 | - | - |",
            "| codex | d3-edge | mms | 4 | 4/4 (100%) | 1.5 / 0.0 / 1.0 | 296.7 (3/4) | 51.7 (3/4) | 2 | - | - | 0 | - | - |",
            "## Cost per correct answer (the media questions)",
            "| agent | model | arm | correct (k) | seconds | tokens k | GPU seconds |",
            "| codex | d1-3B | files | 3/6 | - | - | - |", "| codex | d1-3B | mms | 5/6 | 44.0 | 34.2 | 13.2 |",
            "| codex | d3-edge | mms | 3/4 | - | - | - |", "| codex | d3-edge | files | 3/4 | - | - | - |",
            "A cost per correct answer is told from 5 correct answers (k) in the arm; below that it is `-`.",
            "## Appendix: codex, one row per question",
            "| qid | level | type | files | mms d1-3B | mms d3-edge |",
            "| image-photo-sign-1 | part | image | pass 40 s v5 | pass 30 s a1 | pass 38 s a1 |",
            "| image-photo-sign-3 | week | image | FAIL timeout 900 s v14 | pass 25 s a1 | - |",
            "| text-note-code-1 | part | text | pass 10 s v0 | pass 8 s a0 | - |",
            "| video-porch-parcel-1 | week | video | FAIL 120 s v12 | FAIL 60 s a1 (est. past timeout) | - |",
            "- An mms exec whose ask may need more GPU seconds than 75% of the exec's time (675 s of 900:",
            "- The screen changes like a real one while the owner is active",
            "- `look` costs about 25 times more per frame than an ffmpeg montage"):
        assert line in text, line
    assert text.index("## Read with these") < text.index("## Appendix: codex")


def test_the_decider_tier_reports_each_templates_needle_p_and_verdict(tmp_path, monkeypatch):
    figures = bench.numbers([stub_run(tmp_path, monkeypatch)])
    tier = figures["decider"]
    d13, edge = tier["models"]["d1-3B"], tier["models"]["d3-edge"]
    # the record from before the levels and p got both from the plan: its level, and the p of the candidate at the truth
    assert d13["types"]["image"] == {"asked": 3, "needles": 3, "found": 1, "p50": 2, "top5": 2, "false": 1, "s_per_window": 0.06,
                                     "absent": 0, "absent_clean": 0}
    assert d13["types"]["video"]["absent"] == 1 and d13["types"]["video"]["absent_clean"] == 1 and d13["family"] == "d1"
    assert d13["media"] == {"asked": 6, "needles": 5, "found": 2, "p50": 3, "top5": 3, "false": 1, "s_per_window": 0.06}
    assert (edge["media"]["needles"], edge["media"]["found"], edge["media"]["p50"], edge["media"]["top5"]) == (4, 2, 2, 2)
    assert tier["missed"] == ["porch-parcel"] and tier["asks"] == 10
    sign = tier["templates"]["photo-sign"]
    assert sign["questions"] == 3
    assert sign["models"]["d1-3B"] == {"asked": 3, "found": 1, "near": 1, "stopped": 0, "cut": 0, "missed": 1, "p": pytest.approx(0.82)}
    assert sign["models"]["d3-edge"] == {"asked": 3, "found": 2, "near": 0, "stopped": 0, "cut": 0, "missed": 1, "p": pytest.approx(0.92)}
    assert sign["verdicts"] == {"below its model's hit (calibration)": {"models": ["d1-3B"], "levels": ["day"]},
                                "haystack suspect": {"models": ["d1-3B", "d3-edge"], "levels": ["week"]}}
    assert tier["templates"]["screen-error"]["verdicts"] == {"this model's miss": {"models": ["d3-edge"], "levels": ["part"]}}
    assert tier["templates"]["porch-parcel"]["verdicts"] == {bench.ONE_MODEL: {"models": ["d1-3B"], "levels": ["week"]}}

    text = bench.markdown(figures)
    for line in (
            "| model | type | recall (at its hit) | recall at p >= 0.5 | top 5 | absent with no false sighting | false sightings per ask |",
            "| d1-3B | image | 1/3 (33%) | 2/3 (67%) | 2/3 (67%) | - |",
            "Templates no model found: porch-parcel. The verdicts below say whether that is the haystack or the models.",
            "| template | needle questions | found by | nearly missed by (top 5 or p >= 0.5) | missed by | "
            "scan ended before the needle by (false sighting, max_s) | needle p (median) | verdict |",
            "| photo-sign | 3 | d1-3B 1/3, d3-edge 2/3 | d1-3B 1 | d1-3B 1, d3-edge 1 | - | d1-3B 0.82, d3-edge 0.92 | "
            "below its model's hit (calibration): d1-3B (day); haystack suspect: d1-3B, d3-edge (week) |",
            "| screen-error | 1 | d1-3B 1/1 | - | d3-edge 1 | - | d1-3B 0.97, d3-edge 0.30 | this model's miss: d3-edge (part) |",
            "| porch-parcel | 1 | - | - | d1-3B 1 | - | - | one model ran: the agent arm's look, or a second family, tells a miss from a "
            "haystack fault: d1-3B (week) |",
            "## Summary per model (the media questions)",
            "| model | family | recall (at its hit) | recall at p >= 0.5 | top 5 | false sightings per ask | median s per window | "
            "codex: mms accuracy | codex: s / k tokens / GPU s per correct |",
            "| d1-3B | d1 | 2/5 (40%) | 3/5 (60%) | 3/5 (60%) | 0.17 | 0.060 | 5/6 (83%) | 44.0 / 34.2 / 13.2 (k=5) |",
            "| d3-edge | d3 | 2/4 (50%) | 2/4 (50%) | 2/4 (50%) | 0.00 | 0.060 | 3/4 (75%) | - / - / - (k=3) |"):
        assert line in text, line


def test_the_decider_estimate_is_in_minutes_from_the_haystacks_window_counts_when_it_has_them(built, tmp_path, monkeypatch, capsys):
    run = trimmed(built, tmp_path)
    plan = run.plan()
    slow = model_of("d3-mini")
    monkeypatch.setattr(haystack, "windows_in", lambda plan_, source, since, until: 1000, raising=False)
    bench.show_decider(run, [FAKE_MODEL, slow], ["d3-edge: not downloaded: mms model use d3-edge"])
    lines = capsys.readouterr().out.splitlines()
    minutes = {}
    for model in (FAKE_MODEL, slow):
        mine = [q for q in haystack.questions_for(plan.questions, model.about) if q["oracle"] is not None]
        minutes[model.name] = sum(1000 * bench.window_s(model.about, bench.SOURCE_KINDS.get(q["source"], "video")) for q in mine) / 60
        line = next(line for line in lines if line.startswith(f"  {model.name}"))
        assert line.endswith(f"{len(mine)} oracle asks to run of {len(mine)}, about {minutes[model.name]:.0f} min"), line
    assert minutes["d3-mini"] > 2 * minutes["fake"] > 0, "the slow model takes longer on the same windows"
    total = f"about {sum(minutes.values()):.0f} min in all (each range's windows at the model's seconds a window from its About"
    assert any(total in line for line in lines) and any(line.startswith("  skipped d3-edge") for line in lines)

    # a haystack with no `windows_in`: today's words, and a range too short to count is under a minute
    monkeypatch.delattr(haystack, "windows_in")
    bench.show_decider(run, [FAKE_MODEL], [])
    out = capsys.readouterr().out
    asks = shape(plan).oracle
    assert f"{asks} oracle asks to run of {asks}" in out and ", about" not in out
    assert "each at most about a minute" in out
    assert bench.minutes_text(0.4) == "under a minute" and bench.minutes_text(42.4) == "about 42 min"
    assert bench.decider_minutes(plan, FAKE_MODEL, plan.questions) is None


# -- the shortcut probe in `build`, and in the report -----------------------------------------------------------------------

def stubbed_build(monkeypatch) -> None:
    """`build` with nothing written, loaded or started: its folders, its probe and its run.json are what is left to see."""
    monkeypatch.setattr(bench.haystack, "write", lambda plan, folder: folder.mkdir(parents=True))
    monkeypatch.setattr(bench.haystack, "load", lambda api, folder, plan, wait=120: None)
    monkeypatch.setattr(bench, "ensure_bin", lambda run: None)
    monkeypatch.setattr(bench, "Daemon", lambda home: types.SimpleNamespace(start=lambda: types.SimpleNamespace(api=None, stop=lambda: None)))


def test_the_probe_runs_over_the_written_and_loaded_week_with_the_files_arms_ffmpeg_and_is_recorded(built):
    (call,) = [c for c in PROBED if c["out"] == built.folder]
    assert bench.plan_hash(call["plan"]) == bench.plan_hash(built.plan()) and call["folder"] == built.data and call["seed"] == 7
    assert call["ffmpeg"] == bench.bin_exe(built) and call["ffmpeg"].is_file() and call["ffmpeg"].parent == built.bin
    assert call["data"] and call["loaded"] and not call["run_json"], "it runs once the data is written and loaded, before run.json"
    assert bench.probe_of(built.folder) == {"status": "passed", "failures": []}
    figures = bench.numbers([built])
    assert figures["runs"][0]["probe"] == {"status": "passed", "failures": []}
    assert "Shortcut probe" not in bench.markdown(figures), "a run that passed says nothing"


def test_a_probe_that_fails_or_is_skipped_is_recorded_and_a_build_that_fails_it_exits_1(tmp_path, monkeypatch, capsys):
    stubbed_build(monkeypatch)
    calls = []
    failures = ["video-screen-error-1 (part, screen): 3 states in scope, 250 needed", "audio-mic-alarm-2 (week, mic): 2 sounds in scope, 30 needed"]

    def probe(plan, folder, out, *, ffmpeg=None, seed=None):
        calls.append((plan.scale, seed, Path(out)))
        return list(failures) if seed == 2 else []

    monkeypatch.setattr(bench.shortcuts, "run", probe)
    # passed (seed 1), failed (seed 2: the run is kept, complete and flagged), skipped (seed 3: the probe is not called)
    for seed, want in ((1, {"status": "passed", "failures": []}), (2, {"status": "failed", "failures": failures}),
                       (3, {"status": "skipped", "failures": []})):
        run = bench.build(tmp_path / f"out{seed}", "tiny", seed, temp=tmp_path / "temp", days=DAYS, probe=seed != 3)
        assert bench.probe_of(run.folder) == want and bench.Run.open(run.folder) == run
    assert [(scale, seed) for scale, seed, _ in calls] == [("tiny", 1), ("tiny", 2)] and calls[0][2].parent == tmp_path / "out1"

    # the command line: a failed probe is printed failure by failure and exits 1 with the run kept; `--no-probe` skips it
    cli = tmp_path / "cli"
    base = ["build", "--scale", "tiny", "--out", str(cli), "--temp", str(tmp_path / "temp")]
    assert bench.main([*base, "--seed", "4"]) == 0
    out = capsys.readouterr()
    assert "skipped" not in out.out and calls[-1][1] == 4 and bench.probe_of(next(cli.glob("*-tiny-s4")))["status"] == "passed"
    monkeypatch.setattr(bench.shortcuts, "run", lambda plan, folder, out, *, ffmpeg=None, seed=None: list(failures))
    assert bench.main([*base, "--seed", "5"]) == 1
    out = capsys.readouterr()
    kept = next(cli.glob("*-tiny-s5"))
    assert str(kept) in out.out and (kept / "run.json").is_file() and bench.probe_of(kept)["status"] == "failed"
    assert all(f"probe: {failure}" in out.err for failure in failures) and "the shortcut probe failed (2)" in out.err
    n = len(calls)
    assert bench.main([*base, "--seed", "6", "--no-probe"]) == 0 and len(calls) == n
    assert "the shortcut probe was skipped (--no-probe)" in capsys.readouterr().out
    assert bench.probe_of(next(cli.glob("*-tiny-s6")))["status"] == "skipped"
    # the scales offered are the haystack's own, pilot among them
    assert "pilot" in haystack.SCALES and bench.parser_of().parse_args(["build", "--scale", "pilot"]).scale == "pilot"
    assert bench.parser_of().parse_args(["build"]).no_probe is False and bench.parser_of().parse_args(["build", "--no-probe"]).no_probe
    with pytest.raises(SystemExit):
        bench.main(["build", "--scale", "huge"])


def test_the_report_and_the_commands_say_when_a_runs_probe_failed_or_was_skipped_or_never_ran(built, tmp_path, capsys):
    outcomes = {"failed": {"status": "failed", "failures": ["a: x", "b: y"]}, "skipped": {"status": "skipped", "failures": []},
                "none": None, "passed": {"status": "passed", "failures": []}}
    notes = {}
    for label, outcome in outcomes.items():
        run = trimmed(built, tmp_path / label)
        meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
        meta.pop("probe", None)
        (run.folder / "run.json").write_text(json.dumps({**meta, **({"probe": outcome} if outcome else {})}), encoding="utf-8")
        lines = bench.markdown(bench.numbers([run])).splitlines()
        notes[label] = [line for line in lines if line.startswith("- Shortcut probe:")]
        if notes[label]:
            assert lines.index(notes[label][0]) < next(i for i, line in enumerate(lines) if line.startswith("- Runs pooled"))
        # `agents` and `decider` warn before they spend anything
        capsys.readouterr()
        bench.show_agents(run, ["codex"], [], [], 3)
        bench.show_decider(run, [], [])
        warned = [line for line in capsys.readouterr().out.splitlines() if line.startswith("warning: ")]
        assert len(warned) == (0 if label == "passed" else 2), label
        assert all(notes[label][0].endswith(w.removeprefix("warning: ") + ".") for w in warned)
    assert notes["passed"] == []
    assert "failed the shortcut probe (2 failures, in its shortcuts.md)" in notes["failed"][0]
    assert "was built with --no-probe" in notes["skipped"][0]
    assert "no shortcut probe record (it was built before the probe)" in notes["none"][0]
    # pooled runs: one line for each run that did not pass
    a, b = reseeded(built, tmp_path / "a", 8), reseeded(built, tmp_path / "b", 9)
    for run, outcome in ((a, outcomes["failed"]), (b, outcomes["skipped"])):
        meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
        (run.folder / "run.json").write_text(json.dumps({**meta, "probe": outcome}), encoding="utf-8")
    lines = bench.markdown(bench.numbers([built, a, b])).splitlines()
    assert [line.split(" ")[3] for line in lines if line.startswith("- Shortcut probe:")] == [a.name, b.name]


# -- isolation between execs, the GPU held by the long ones, the files arm's folders ---------------------------------------

class Probing:
    """An mms agent for the arm's scheduler and isolation: it looks, with scope all, at the labels in its home over the whole
    week, asks the fake decider its question under its own project (which labels each sighting), once marks the phone's items
    kept with `label`, looks again, and answers right. It keeps what it saw and the span of its run."""

    def __init__(self, run, pause: float = 0.0, keep_in: str | None = None):
        questions = run.plan().questions
        self.week = (min(q["since"] for q in questions), max(q["until"] for q in questions))
        self.pause, self.keep_in = pause, keep_in
        self.seen, self.after, self.spans, self.kept = {}, {}, {}, {}
        self.lock = threading.Lock()

    def labels(self, api) -> list[dict]:
        found = api.call("find", types=["label"], scope="all", since=self.week[0], until=self.week[1], limit=200)["results"]
        return [{"project": r.get("project"), "by": r.get("by"), "text": r["text"]} for r in found if r.get("by") in ("ask", "agent")]

    def __call__(self, e: bench.Exec) -> dict:
        q, api, project = e.question, Api(e.home), bench.exec_project(e.question)
        began = time.monotonic()
        before = self.labels(api)
        bench.ask(api, bench.oracle_params(q, q["fake"]["question"]), project)
        during = None
        if q["qid"] == self.keep_in:
            api.call("label", text="mine", since=q["since"], until=q["until"], source=q["source"], keep=True, caller={"project": project})
            during = bench.kept_items(e.home)
        time.sleep(self.pause)
        after = self.labels(api)
        with self.lock:
            self.seen[q["qid"]], self.after[q["qid"]], self.kept[q["qid"]] = before, after, during
            self.spans[q["qid"]] = (began, time.monotonic())
        return {"answer": right(q), "final": f"ANSWER: {right(q)}", "tools": [], "seconds": 1.0, "problem": None, "turns": 1,
                "tokens": {"input": 10, "output": 1, "cache_read": 0, "cache_write": 0}, "cost_usd": None, "used_percent": 1.0,
                "model": "stub", "denials": 0}


def labels_left(home: Path, pattern: str = "id:bench-%") -> tuple[int, int, int]:
    """How many labels, project rows and session rows of the bench's execs an mms home holds."""
    with closing(sqlite3.connect(home / "index.sqlite")) as db:
        return (db.execute("SELECT COUNT(*) FROM labels WHERE project LIKE ?", (pattern,)).fetchone()[0],
                db.execute("SELECT COUNT(*) FROM projects WHERE id LIKE ?", (pattern,)).fetchone()[0],
                db.execute("SELECT COUNT(*) FROM sessions WHERE project LIKE ?", (pattern,)).fetchone()[0])


def test_no_exec_sees_a_label_an_earlier_exec_left_and_a_long_exec_holds_the_gpu_alone(built, tmp_path, monkeypatch):
    run = trimmed(built, tmp_path, homes=True)
    plan = run.plan()
    used: set[str] = set()

    def pick(*templates: str) -> list[dict]:
        out = []
        for template in templates:
            q = next(q for q in plan.questions if q["template"] == template and q["qid"] not in used)
            used.add(q["qid"])
            out.append(q)
        return out

    first = pick("photo-sign", "screen-error")
    second = pick("note-code", "porch-parcel", "mic-doorbell", "photo-sign")
    third = pick("log-count", "absent-log", "log-last", "photo-count", "absent-note", "absent-sign", "absent-doorbell")
    assert all(bench.gpu_estimate(plan, FAKE_MODEL, q) <= bench.HEAVY_S / 4 for q in [*first, *second, *third]), "all cheap to ask (the screen's frames count: a part is 47 s): none holds the GPU alone"
    home = bench.copy_home(run.snapshot(), tmp_path / "home", FAKE_MODEL)
    daemon = Daemon(home).start()
    baseline = bench.kept_items(home)

    def arm(agent, questions, jobs):
        bench.run_arm(run, "claude", agent, "mms", FAKE_MODEL, questions, home, jobs, bench.Halt())

    try:
        # without the scrub, the later of two execs finds the earlier one's label (so the test below can fail)
        leaky = Probing(run)
        with monkeypatch.context() as patch:
            patch.setattr(bench, "scrub_home", lambda home_, kept, **kw: {})
            arm(leaky, first, 1)
        early, late = sorted(first, key=lambda q: (q["until"], q["qid"]))
        assert leaky.seen[early["qid"]] == [] and {r["project"] for r in leaky.seen[late["qid"]]} == {bench.exec_project(early)}
        swept = bench.scrub_home(home, baseline, prefix="id:bench-")
        assert swept["labels"] >= 2 and labels_left(home) == (0, 0, 0), "a sweep takes out what an interrupted run left"

        # with it, an exec starts from the snapshot's state: it sees no label of an exec that ended, and leaves none
        agent = Probing(run, keep_in=second[-1]["qid"])
        arm(agent, second, 1)
        for q in second:
            own = [r for r in agent.after[q["qid"]] if r["project"] == bench.exec_project(q)]
            rec = result(run, "claude", "mms-fake", q["qid"])
            assert agent.seen[q["qid"]] == [], q["qid"]
            assert own and {r["project"] for r in agent.after[q["qid"]]} == {bench.exec_project(q)}, "it sees its own label, and no other"
            assert rec["left"] == {"labels": len(own), "unkept": len(agent.kept[q["qid"]] - baseline) if q["qid"] == agent.keep_in else 0}
        assert agent.kept[agent.keep_in] > baseline and result(run, "claude", "mms-fake", agent.keep_in)["left"]["unkept"] > 0, "it kept items"
        assert bench.kept_items(home) == baseline and labels_left(home) == (0, 0, 0)
        assert next(r for r in agent.after[agent.keep_in] if r["by"] == "agent")["text"] == "mine"

        # the scheduler: two long execs (by an estimate of 400 s) run alone, two of one template never run together, and the
        # others stay parallel; each result holds its estimate, its wait and whether it ran alone
        long_ones = {"image-photo-count-1", "records-log-last-1"}
        monkeypatch.setattr(bench, "gpu_estimate", lambda plan_, model, q: 400.0 if q["qid"] in long_ones else 20.0)
        slow = Probing(run, pause=0.15)
        arm(slow, third, 3)
    finally:
        daemon.stop()
    spans = slow.spans
    assert long_ones <= set(spans), "the long ones are in this part"

    def together(a: str, b: str) -> bool:
        return spans[a][0] < spans[b][1] and spans[b][0] < spans[a][1]

    assert not any(together(h, x) for h in long_ones for x in spans if x != h), "an exec that holds the GPU runs alone"
    log_count, absent_log = (q["qid"] for q in third[:2])
    assert not together(log_count, absent_log), "a template's needle and the absent question in its words never run together"
    light = [q["qid"] for q in third if q["qid"] not in long_ones]
    assert any(together(a, b) for i, a in enumerate(light) for b in light[i + 1:]), "the others stay parallel"
    for q in third:
        rec = result(run, "claude", "mms-fake", q["qid"])
        assert (rec["gpu_estimate_s"], rec["exclusive"]) == ((400.0, True) if q["qid"] in long_ones else (20.0, False))
        assert rec["held_s"] >= 0 and rec["left"]["labels"] >= 0 and rec["problem"] is None and "queued_s" not in rec
        assert (rec["cancelled"], rec["unended"], rec["orphans"], rec["beyond_timeout"]) == ([], [], 0, False)
    assert labels_left(home) == (0, 0, 0) and result(run, "claude", "mms-fake", third[0]["qid"])["held_s"] < 60
    assert all(result(run, "claude", "mms-fake", h)["busy_at_start"] == 0 for h in long_ones), "a long exec starts with the daemon idle"
    assert result(run, "claude", "mms-fake", third[0]["qid"])["busy_at_start"] is None


def test_a_files_exec_may_open_only_its_own_slot_and_its_folders_have_no_name_that_tells_the_week(built, tmp_path):
    run = built
    # the folders in the temp folder are named by a tag, not by the run (whose name has the seed and the scale)
    assert re.fullmatch(r"[0-9a-f]{10}", run.tag) and run.key == run.tag
    for path in (run.data, run.bin, run.slot(0), run.slot(3)):
        assert path.parent == run.temp and path.name.startswith(f"mmsb-{run.tag}-")
        assert not any(word in path.name for word in (run.name, run.scale, f"s{run.seed}", "tiny", "-s7"))
    meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
    assert meta["tag"] == run.tag and Path(meta["data"]) == run.data and bench.Run.open(run.folder).tag == run.tag
    # a run from before the tags is named as it was
    old = trimmed(built, tmp_path)
    (old.folder / "run.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "tag"}), encoding="utf-8")
    old = bench.Run.open(old.folder)
    assert (old.tag, old.key, old.slot(0).name) == ("", old.name, f"mmsb-{old.name}-w0")

    # an exec's deny list: every slot but its own (for what it runs, not for what a listing prints), and the generator's words
    own, others = run.slot(1), [run.slot(k) for k in (0, 2, 3)]
    deny = run.deny(own)
    assert all(f"in:mmsb-{run.tag}-w{k}" in run.deny() for k in range(4)) and f"in:mmsb-{run.tag}-w1" not in deny, "no slot of its own: all"
    for k, other in zip((0, 2, 3), others):
        by_path = codex_run.reached({"command": f"Get-Content '{other}/montage.jpg'"}, deny)
        assert by_path and codex_run.reached({"command": str(other).replace(chr(92), "/") + "/m.png"}, deny)
        assert codex_run.reached(f"{other}/m.png", deny, output=True) is None, "a listing of the temp folder may name a slot"
        # by a relative path or a glob, which do not hold the absolute path
        for command in (f"Get-Content ..{chr(92)}mmsb-{run.tag}-w{k}{chr(92)}m.png", f"cd ../mmsb-{run.tag}-w{k}", f"ls $env:TEMP{chr(92)}mmsb-*-w{k}",
                        f"ls mmsb-{run.tag}-w*", f"ls mmsb-{run.tag}-w?", f"Get-ChildItem -Path . -Filter *-w{k}", f"cat ?-w{k}/m.png",
                        "Get-ChildItem $env:TEMP -Filter mmsb-*", "dir mmsb-[a-f]*"):
            assert codex_run.reached({"command": command}, deny), command
    assert codex_run.reached({"command": f"ffmpeg -i {run.data}/screen/a.mkv {own}/m.jpg"}, deny) is None
    for command in (f".{chr(92)}montage.jpg", "cd $PWD; ls", f"ffmpeg -i {run.data}{chr(92)}screen{chr(92)}a.mkv -vf fps=1 m.png",
                    f"Get-ChildItem {run.bin}", f"ls mmsb-{run.tag}-w1{chr(92)}", "Get-ChildItem *.jpg"):
        assert codex_run.reached({"command": command}, deny) is None, command
    # the generator, as code reaches it; prose that says haystack or truth, and what the data says, are no escape
    for command in ('python -c "import haystack"', "Get-Content benchmarks/haystack.py", "python -c 'from haystack import plan'",
                    "python -c \"print(haystack.plan(1, 'tiny', []))\"", "python -c \"print(q['truth'])\"",
                    'python -c "print(q[\\"truth\\"])"', "python -c \"print(q.get('truth'))\""):
        assert codex_run.reached({"command": command}, deny), command
    for command in ("echo needle in a haystack", "grep -r 'the truth' notes", "python -c \"def plan(x): return x\"", "echo truth"):
        assert codex_run.reached({"command": command}, deny) is None, command
    for printed in ("the truth is out there", "haystack.plan(a) in the notes", "import haystack", "q['truth']"):
        assert codex_run.reached(printed, deny, output=True) is None, "what the data says is not what the agent ran"
    bench.check_paths(built)
    # both agents refuse these, as a tool input and not as an output
    shell = {"type": "item.completed", "item": {"type": "command_execution", "command": f"cat {others[0]}/a.txt", "aggregated_output": ""}}
    assert codex_run.check_event(shell, "files", deny) == f"shell reached {codex_run.reached(shell['item']['command'], deny)}"
    import_it = {"type": "item.completed", "item": {"type": "command_execution", "command": "python -c 'import haystack'", "aggregated_output": ""}}
    assert codex_run.check_event(import_it, "files", deny) == "shell reached import haystack"
    listing = {"type": "item.completed", "item": {"type": "command_execution", "command": "dir ..", "aggregated_output": f"{others[0]}/x"}}
    assert codex_run.check_event(listing, "files", deny) is None
    use = {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "python benchmarks/haystack.py"}}]}}
    assert claude_run.check_event(use, "files", deny) == "Bash reached haystack.py"
    # Claude's free-text `description` is not what it ran: it may say haystack, and a path in it is not opened
    prose = {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t2", "name": "Bash", "input": {
        "command": "ffmpeg -i a.mkv m.jpg", "description": f"find the needle in a haystack of recordings, not {others[0]}/x"}}]}}
    assert claude_run.check_event(prose, "files", deny) is None
    read = {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t3", "name": "Read", "input": {
        "file_path": f"{others[0]}/x.png", "description": "a picture"}}]}}
    assert claude_run.check_event(read, "files", deny) == f"Read reached {codex_run.reached(f'{others[0]}/x.png', deny)}"

    # every exec of the files arm gets the deny list of its own slot, and its slot is empty once it ends
    class Messy(Stub):
        def __call__(self, e):
            (e.work / "montage.jpg").write_bytes(b"x")
            (e.work / "frames").mkdir()
            (e.work / "frames" / "f1.png").write_bytes(b"y")
            return super().__call__(e)

    def norm(path) -> str:
        return re.sub(r"[\\/]+", "/", str(path).lower())

    messy, files_run = Messy(), trimmed(built, tmp_path / "files")
    bench.run_agents(files_run, {"claude": messy}, [], jobs=2)
    assert messy.execs and all(e.deny == files_run.deny(e.work) for e in messy.execs)
    assert all("in:" + norm(e.work) not in e.deny and all("in:" + norm(files_run.slot(k)) in e.deny for k in range(4)
                                                           if files_run.slot(k) != e.work) for e in messy.execs)
    assert all(not any(files_run.slot(k).iterdir()) for k in range(2)), "what an exec wrote is gone when it ends"


def test_agents_refuses_a_run_whose_probe_failed_unless_anyway_and_only_warns_when_it_was_skipped(built, tmp_path, monkeypatch, capsys):
    run = trimmed(built, tmp_path)
    meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
    n = shape(run.plan()).n
    stub = Stub()
    monkeypatch.setattr(bench, "real_agents", lambda: {"claude": stub, "codex": stub})
    monkeypatch.setattr(bench.claude_run, "check_login", lambda: None)

    def probe_was(outcome: dict) -> list[str]:
        (run.folder / "run.json").write_text(json.dumps({**meta, "probe": outcome}), encoding="utf-8")
        return ["agents", str(run.folder), "--agents", "claude", "--models", "none"]

    failed = {"status": "failed", "failures": ["a: x", "b: y"]}
    assert bench.main(probe_was(failed)) == 1 and stub.execs == []
    err = capsys.readouterr().err
    assert "failed the shortcut probe (2 failures, in its shortcuts.md)" in err and "--anyway" in err
    assert bench.main([*probe_was(failed), "--anyway"]) == 0 and len(stub.execs) == n
    out = capsys.readouterr().out
    assert "warning: " in out and "failed the shortcut probe" in out, "run anyway, and said so"
    # a dry run spends nothing, so it is not refused
    with monkeypatch.context() as patch:
        patch.setattr(bench, "shutil", types.SimpleNamespace(which=lambda name: None))
        assert bench.main([*probe_was(failed), "--models", "none", "--dry-run"]) == 0
    capsys.readouterr()
    # a skipped probe runs, with the warning
    stub.execs.clear()
    for result_file in (run.folder / "results").rglob("*.json"):
        result_file.unlink()
    assert bench.main(probe_was({"status": "skipped", "failures": []})) == 0 and len(stub.execs) == n
    assert "built with --no-probe" in capsys.readouterr().out
    assert bench.parser_of().parse_args(["agents", "r"]).anyway is False and bench.parser_of().parse_args(["agents", "r", "--anyway"]).anyway


def test_a_check_in_that_leaves_out_what_did_not_change_still_counts_for_its_job():
    """A later reply holds `job`, `state` and `progress` (and `stopped` once set); the fields that are None or unchanged are gone."""
    first = job_result("j-a", "running", 10, 1.0)
    first.pop("error")
    check_in = {"job": "j-a", "state": "running", "progress": {"windows": 40, "sightings": 1, "of_about": 121, "seconds": 2.5, "decider": "ready"},
                "next": "Call ask"}
    last = {"job": "j-a", "state": "done", "stopped": "limit", "progress": {"windows": 61, "sightings": 1, "of_about": 121, "seconds": 3.5},
            "sightings": [{"source": "screen", "t": "2026-11-01T18:00:03.000-05:00", "p": 0.9, "kind": "video", "look": {"item": 1, "at": 3.0}}]}
    tools = [call("ask", {"question": "?"}, mms_result(first)), call("ask", {"job": "j-a"}, mms_result(check_in))]
    assert bench.open_jobs(tools) == ["j-a"]
    assert bench.work_of(tools) == {"jobs": 1, "windows": 40, "seconds": 2.5, "views": 0, "shells": 0, "stopped": {"running": 1}}
    tools.append(call("ask", {"job": "j-a"}, mms_result(last)))
    assert bench.open_jobs(tools) == []
    assert bench.work_of(tools) == {"jobs": 1, "windows": 61, "seconds": 3.5, "views": 0, "shells": 0, "stopped": {"limit": 1}}
    all_of_it = call("ask", {"job": "j-a", "all": True}, mms_result({**last, "searched": {}, "candidates": [], "warnings": []}))
    assert bench.work_of([*tools[:2], all_of_it])["windows"] == 61, "a reply for `all` is a reply like another"


def test_ask_jobs_an_exec_left_running_are_cancelled_when_it_ends():
    left_running = call("ask", {"question": "?"}, mms_result(job_result("j-a", "running", 10, 1.0)))
    collected = [call("ask", {"question": "?"}, mms_result(job_result("j-b", "running", 5, 0.5))),
                 call("ask", {"job": "j-b"}, mms_result(job_result("j-b", "done", 50, 3.0, "range_end")))]
    queued = call("ask", {"question": "?"}, mms_result(job_result("j-c", "queued", 0, 0.0)))
    cancelled = call("ask", {"job": "j-d"}, mms_result(job_result("j-d", "cancelled", 3, 0.2, "cancelled")))
    not_asks = [call("ask", {"question": "?"}, "ask failed: bad_request"), call("find", {"sources": ["phone"]}, mms_result({"job": "j-z", "state": "running"}))]
    tools = [left_running, *collected, queued, cancelled, *not_asks]
    assert bench.open_jobs(tools) == ["j-a", "j-c"] and bench.open_jobs([]) == []

    class Daemon_:
        def __init__(self):
            self.calls = []

        def call(self, op, **body):
            self.calls.append((op, body))
            if body["job"] == "j-c":
                raise RuntimeError("not_found: the job ended")
            return {}

    api = Daemon_()
    assert bench.cancel_open(api, tools) == ["j-a"] and api.calls == [("jobs/cancel", {"job": "j-a"}), ("jobs/cancel", {"job": "j-c"})]
    assert bench.cancel_open(Daemon_(), [collected[1]]) == []


# -- jobs left behind, hashed results, the files hash, the files arm's templates, asks past the timeout ----------------------

class JobApi:
    """The daemon's job API as a test scripts it: `jobs/cancel` returns at once with the job still running (a cancel that timed
    out), the job ends `ends_after` seconds after it (its late label lands then, noted in `events`), and `status` counts jobs
    from `counts` in turn, the last repeating. A job not in `known` is unknown to the daemon."""

    def __init__(self, events: list | None = None, ends_after: float = 0.4, counts=(0,), known=("j-late",)):
        self.events, self.ends_after, self.counts, self.known = events if events is not None else [], ends_after, list(counts), set(known)
        self.cancelled_at: dict[str, float] = {}
        self.calls: list[tuple] = []

    def call(self, op: str, **body) -> dict:
        self.calls.append((op, body))
        if op == "status":
            n = self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]
            return {"jobs": {"running": n, "queued": 0}}
        if body["job"] not in self.known:
            raise RuntimeError("not_found: no such job")
        if op == "jobs/cancel":
            self.cancelled_at.setdefault(body["job"], time.monotonic())
            return {"state": "running"}
        at = self.cancelled_at.get(body["job"])
        if at is None or time.monotonic() - at < self.ends_after:
            return {"state": "running"}
        if ("label lands", body["job"]) not in self.events:
            self.events.append(("label lands", body["job"]))
        return {"state": "cancelled"}


def test_an_exec_that_ended_waits_for_its_jobs_and_for_the_daemon_to_come_down_to_the_execs_still_running():
    open_ask = call("ask", {"question": "?"}, mms_result(job_result("j-late", "running", 10, 1.0)))
    # settle: a cancelled job that stops late is waited for, one that never stops is named, an unknown one is gone
    api = JobApi()
    api.call("jobs/cancel", job="j-late")
    began = time.monotonic()
    assert bench.settle(api, ["j-late", "j-gone"], 5) == [] and time.monotonic() - began >= 0.35
    assert api.events == [("label lands", "j-late")]
    stuck = JobApi(ends_after=99)
    stuck.call("jobs/cancel", job="j-late")
    assert bench.settle(stuck, ["j-late"], 0.3) == ["j-late"]
    # wait_idle: down to the allowed count, or how many are over it, or unreadable
    assert bench.wait_idle(JobApi(counts=[2, 0]), 0, 5) == 0 and bench.wait_idle(JobApi(counts=[1]), 1, 5) == 0
    assert bench.wait_idle(JobApi(counts=[3]), 1, 0.3) == 2 and bench.wait_idle(JobApi(counts=[4]), 0, 0) == 4

    class Unreadable:
        def __init__(self, answer):
            self.answer = answer

        def call(self, op, **body):
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    assert bench.wait_idle(Unreadable(RuntimeError("gone")), 0, 1) is None and bench.wait_idle(Unreadable({"jobs": {"running": None}}), 0, 1) is None
    # quiesce: cancel, wait for the job, then for the count
    events = []
    got = bench.quiesce(JobApi(events, counts=[1, 0]), [open_ask])
    assert got["cancelled"] == ["j-late"] and got["unended"] == [] and got["orphans"] == 0 and got["settle_s"] >= 0.35
    assert events == [("label lands", "j-late")], "it returned after the label had landed"
    # a job whose id was never logged (the exec timed out inside the ask that started it) is only in the count
    assert bench.quiesce(JobApi(counts=[1]), [], others=0, limit_s=0.3)["orphans"] == 1
    assert bench.quiesce(JobApi(counts=[1]), [], others=1, limit_s=0.3)["orphans"] == 0, "one for the exec still running"
    assert bench.quiesce(JobApi(counts=[3]), [], others=1, limit_s=0.3)["orphans"] == 2
    unended = bench.quiesce(JobApi(ends_after=99), [open_ask], limit_s=0.3)
    assert (unended["cancelled"], unended["unended"]) == (["j-late"], ["j-late"])


def test_the_next_exec_of_a_template_starts_after_the_late_label_of_the_last_one_has_landed_and_been_scrubbed(built, tmp_path, monkeypatch):
    run = trimmed(built, tmp_path)
    plan = run.plan()
    a, b = sorted((next(q for q in plan.questions if q["template"] == t) for t in ("log-count", "absent-log")),
                  key=lambda q: (q["until"], q["qid"]))
    other = next(q for q in plan.questions if q["template"] == "mic-doorbell")
    events, bases = [], []
    api = JobApi(events, counts=[0])
    monkeypatch.setattr(bench, "Api", lambda home: api)
    monkeypatch.setattr(bench, "kept_items", lambda home: bases.append(home) or set())
    monkeypatch.setattr(bench, "scrub_home", lambda home, kept, **kw: events.append(("scrub", tuple(kw.get("projects", ())))) or {"labels": 0, "unkept": 0})
    monkeypatch.setattr(bench, "gpu_estimate", lambda plan_, model, q: 10.0)

    def agent(e):
        qid = e.question["qid"]
        events.append(("start", qid))
        time.sleep(0.1)
        events.append(("end", qid))
        tools = [call("ask", {"question": "?"}, mms_result(job_result("j-late", "running", 10, 1.0)))] if qid == a["qid"] else []
        return {"answer": right(e.question), "final": "", "tools": tools, "seconds": 1.0, "problem": None, "turns": 1, "tokens": None,
                "cost_usd": None, "used_percent": 1.0, "model": "stub", "denials": 0}

    bench.run_arm(run, "claude", agent, "mms", FAKE_MODEL, [b, a, other], tmp_path / "home", 3, bench.Halt())
    at = {event: i for i, event in enumerate(events)}
    mine = ("scrub", (bench.exec_project(a),))
    assert at[("end", a["qid"])] < at[("label lands", "j-late")] < at[mine] < at[("start", b["qid"])], events
    assert at[("start", other["qid"])] < at[("label lands", "j-late")], "a question of another template is not held by it"
    assert [call_[0] for call_ in api.calls if call_[1].get("job") == "j-late"][:2] == ["jobs/cancel", "jobs/get"]
    # the baseline of the keeps is the snapshot's, not the live home, which an interrupted exec may have left kept
    assert bases and all(base == run.snapshot() for base in bases)
    rec_a, rec_b, rec_o = (result(run, "claude", "mms-fake", q["qid"]) for q in (a, b, other))
    assert (rec_a["cancelled"], rec_a["unended"], rec_a["orphans"]) == (["j-late"], [], 0) and rec_b["cancelled"] == []
    # the wait is how long the question was held back with a slot free
    assert rec_a["held_s"] == 0.0 and rec_b["held_s"] >= 0.3 and rec_o["held_s"] == 0.0
    assert (rec_a["beyond_timeout"], rec_a["busy_at_start"], rec_o["exclusive"]) == (False, None, False) and "queued_s" not in rec_a
    assert [c for c in api.calls if c[0] == "status"], "the daemon's jobs were counted"
    # a long exec, past the exec's timeout, starts when the daemon is idle: it waited for two jobs to end; and says how many were left
    for f in (run.folder / "results").rglob("*.json"):
        f.unlink()
    monkeypatch.setattr(bench, "gpu_estimate", lambda plan_, model, q: 1000.0)
    waited = JobApi(counts=[2, 0])
    monkeypatch.setattr(bench, "Api", lambda home: waited)
    bench.run_arm(run, "claude", agent, "mms", FAKE_MODEL, [other], tmp_path / "home", 1, bench.Halt())
    rec_o = result(run, "claude", "mms-fake", other["qid"])
    assert (rec_o["gpu_estimate_s"], rec_o["exclusive"], rec_o["beyond_timeout"], rec_o["busy_at_start"]) == (1000.0, True, True, 0)
    assert bench.worked([rec_a, rec_o])["beyond"] == 1 and bench.cell_text(bench.cell_of(rec_o), "asks") == "pass 1 s a0 (est. past timeout)"
    # past the run's timeout, not the module's: a run that allows 1,500 s (an ask may have 1,125) does not flag 1,000
    for f in (run.folder / "results").rglob("*.json"):
        f.unlink()
    monkeypatch.setattr(bench, "Api", lambda home: JobApi(counts=[0]))
    bench.run_arm(dataclasses.replace(run, timeout_s=1500.0), "claude", agent, "mms", FAKE_MODEL, [other], tmp_path / "home", 1, bench.Halt())
    assert result(run, "claude", "mms-fake", other["qid"])["beyond_timeout"] is False
    for f in (run.folder / "results").rglob("*.json"):
        f.unlink()
    busy = JobApi(counts=[3])
    monkeypatch.setattr(bench, "Api", lambda home: busy)
    monkeypatch.setattr(bench, "JOB_WAIT_S", 0.3)
    bench.run_arm(run, "claude", agent, "mms", FAKE_MODEL, [other], tmp_path / "home", 1, bench.Halt())
    assert result(run, "claude", "mms-fake", other["qid"])["busy_at_start"] == 3


def test_a_long_compact_ask_result_is_not_hashed_so_its_job_and_its_windows_are_still_read():
    said = "Isthereaparceltothefrontdoor?:yes(p0.97)," + "x" * 100
    text = json.dumps({"job": "j-big", "state": "running", "sightings": [{"said": said, "p": 0.97}] * 40,
                       "progress": {"windows": 40, "of_about": 100, "seconds": 2.5}}, separators=(",", ":"))
    assert len(text) > 3000 and " " not in text[:200], "a compact result, the first said with no space in it"
    result_ = {"content": [{"type": "text", "text": text}], "structured_content": None}
    assert codex_run.scrubbed(result_) == result_ and codex_run.scrubbed([{"type": "text", "text": text}]) == [{"type": "text", "text": text}]
    tools = [call("ask", {"question": "?"}, codex_run.scrubbed(result_))]
    assert bench.open_jobs(tools) == ["j-big"] and bench.work_of(tools)["windows"] == 40 and bench.work_of(tools)["seconds"] == 2.5
    # a picture's base64 and a data URL still are
    assert codex_run.scrubbed("A" * 5000).startswith("<5000 characters, sha1 ") and codex_run.scrubbed("data:image/png;base64," + "A" * 5000).startswith("<")
    assert codex_run.scrubbed(["QUJD" * 2000])[0].startswith("<8000 characters") and codex_run.scrubbed("word " * 2000) == "word " * 2000
    assert codex_run.scrubbed("[" + "1," * 3000 + "1]") == "[" + "1," * 3000 + "1]"


def test_a_changed_distractor_with_no_version_bump_is_caught_by_the_files_hash(built, tmp_path, monkeypatch):
    run = trimmed(built, tmp_path)
    meta = json.loads((run.folder / "run.json").read_text(encoding="utf-8"))
    real = haystack.plan
    assert meta["files"] == bench.files_hash(real(7, "tiny", DAYS)) and meta["files"] != bench.files_hash(real(8, "tiny", DAYS))

    def other_spec(seed, scale, days):
        plan = real(seed, scale, days)
        victim = plan.files[3]
        return dataclasses.replace(plan, files=tuple(dataclasses.replace(f, spec={**f.spec, "extra": 1}) if f is victim else f for f in plan.files))

    with monkeypatch.context() as patch:
        patch.setattr(haystack, "plan", other_spec)
        assert bench.plan_hash(haystack.plan(7, "tiny", DAYS)) == meta["plan"], "the questions are the same"
        with pytest.raises(ValueError, match="files are not those"):
            bench.Run.open(run.folder)
    assert bench.Run.open(run.folder) == run
    (run.folder / "run.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "files"}), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(haystack, "plan", other_spec)
        assert bench.Run.open(run.folder) == run, "a run from before the files hash has none to check"


def test_two_questions_about_one_needle_never_run_together_in_the_files_arm_either(built, tmp_path):
    run = trimmed(built, tmp_path)
    stub = Stub(pause=0.1)
    bench.run_agents(run, {"claude": stub}, [], jobs=3)
    spans = {x[2]: (x[3], x[4]) for x in stub.log if x[0] == "files"}
    by_base: dict[str, list[str]] = {}
    for q in run.plan().questions:
        by_base.setdefault(bench.base_template(q), []).append(q["qid"])

    def together(x: str, y: str) -> bool:
        return spans[x][0] < spans[y][1] and spans[y][0] < spans[x][1]

    pairs = [(x, y) for qids in by_base.values() for i, x in enumerate(qids) for y in qids[i + 1:]]
    assert pairs and not any(together(x, y) for x, y in pairs)
    qids = list(spans)
    assert any(together(x, y) for i, x in enumerate(qids) for y in qids[i + 1:]), "the others stay parallel"


def test_agents_names_the_questions_whose_ask_may_need_more_gpu_seconds_than_the_exec_has(built, tmp_path, monkeypatch, capsys):
    run = trimmed(built, tmp_path)
    q = next(q for q in run.plan().questions if q["template"] == "screen-error" and q["level"] == "week")
    monkeypatch.setattr(bench, "gpu_estimate", lambda plan, model, question: 988.0 if question["qid"] == q["qid"] else 5.0)
    bench.show_agents(run, ["codex"], [FAKE_MODEL], [], 3)
    out = capsys.readouterr().out
    assert "fake: an ask over the whole range of 1 questions may need more GPU seconds than the 675 s an exec leaves its ask" in out
    assert f"{q['qid']} (988 s)" in out and "its speed, not a miss" in out
    # an ask that fits the exec's 900 s but not with the agent's own turns (the screen's day asks of d1-3B at pilot scale: 880 s) is named; 670 s is not
    for seconds, named in ((700.0, True), (670.0, False)):
        monkeypatch.setattr(bench, "gpu_estimate", lambda plan, model, question, seconds=seconds: seconds if question["qid"] == q["qid"] else 5.0)
        bench.show_agents(run, ["codex"], [FAKE_MODEL], [], 3)
        assert (f"{q['qid']} ({seconds:.0f} s)" in capsys.readouterr().out) is named
    monkeypatch.setattr(bench, "gpu_estimate", lambda plan, model, question: 5.0)
    bench.show_agents(run, ["codex"], [FAKE_MODEL], [], 3)
    assert "a timeout there is its speed" not in capsys.readouterr().out


# -- the seam for another dataset: a blind arm, linked homes, the dataset's own parts ----------------------------------------

SEAM: list[tuple] = []  # what `Marked` and the stub agent were asked, in order, by the test that uses them


@dataclasses.dataclass(frozen=True)
class Marked(bench.Run):
    """A run as another dataset's would be: it says what each arm reads, is asked and adds, and what the exec is held by."""

    def view(self, q, arm, k, home):
        SEAM.append(("view", arm, q["qid"]))
        return None if arm == "blind" else self.temp / f"view-{arm}-{k}"

    def prompt(self, q, arm, data, now=None):
        return f"{arm}|{data}|{q['qid']}"

    def estimate(self, plan, model, q):
        return 77.0

    def needle(self, q):
        return q["qid"]

    def extra(self, e, out):
        SEAM.append(("extra", e.arm, e.question["qid"]))
        return {"audit": f"{e.arm}:{out['answer']}"}


def marked(run: bench.Run) -> Marked:
    return Marked(**{f.name: getattr(run, f.name) for f in dataclasses.fields(run)})


def stub_exec(seen: list, pause: float = 0.0):
    """An agent that keeps each exec it is given, notes it in SEAM and answers `x`."""
    def agent(e):
        SEAM.append(("agent", e.arm, e.question["qid"]))
        seen.append((e, time.monotonic()))
        time.sleep(pause)
        seen.append((e, time.monotonic()))
        return {"answer": "x", "final": "", "tools": [], "seconds": 1.0, "problem": None, "turns": 1, "tokens": None,
                "cost_usd": None, "used_percent": 1.0, "model": "stub", "denials": 0}

    return agent


def test_each_arm_is_given_what_the_runs_view_prompt_and_extra_make_and_blind_results_have_their_own_folder(built, tmp_path, monkeypatch):
    run = marked(trimmed(built, tmp_path))
    q = run.plan().questions[0]
    home = tmp_path / "home"
    SEAM.clear()
    # the mms arm needs a daemon only for its scrub, jobs and keeps, which are stood in for here
    monkeypatch.setattr(bench, "Api", lambda home_: JobApi(counts=[0]))
    monkeypatch.setattr(bench, "kept_items", lambda home_: set())
    monkeypatch.setattr(bench, "scrub_home", lambda home_, kept, **kw: SEAM.append(("scrub", tuple(kw.get("projects", ())))) or {})
    seen: list = []
    for arm, model in (("blind", None), ("files", None), ("mms", FAKE_MODEL)):
        SEAM.clear()
        bench.run_arm(run, "claude", stub_exec(seen), arm, model, [q], home if arm == "mms" else None, 1, bench.Halt())
        e = seen[-1][0]
        data = None if arm == "blind" else run.temp / f"view-{arm}-0"
        assert (e.arm, e.model, e.data, e.prompt) == (arm, model, data, f"{arm}|{data}|{q['qid']}")
        assert e.home == (home if arm == "mms" else None) and e.work == run.slot(0) and e.deny == run.deny(e.work)
        # the home is scrubbed, then the view is made, then the agent runs, then the audit is read
        steps = ["view", "agent", "extra"] if arm != "mms" else ["scrub", "view", "agent", "extra"]
        assert [s[0] for s in SEAM][:len(steps)] == steps
        folder = arm if arm != "mms" else "mms-fake"
        rec = result(run, "claude", folder, q["qid"])
        assert (rec["arm"], rec["audit"], rec["exec"]) == (arm, f"{arm}:x", f"execs/claude/{folder}/{q['qid']}")
        assert rec["model"] == (model.name if model else None) and ("gpu_estimate_s" in rec) == (arm == "mms")
        if arm == "mms":
            assert (rec["gpu_estimate_s"], rec["exclusive"]) == (77.0, False)
    assert sorted(p.name for p in (run.folder / "results" / "claude").iterdir()) == ["blind", "files", "mms-fake"]
    assert [bench.arm_dir(*pair) for pair in (("blind", None), ("files", None), ("mms", FAKE_MODEL))] == ["blind", "files", "mms-fake"]
    assert bench.result_path(run, "codex", "blind", None, "q-1") == run.folder / "results" / "codex" / "blind" / "q-1.json"
    for arm, model in (("blind", FAKE_MODEL), ("mms", None), ("both", None)):
        with pytest.raises(ValueError, match="arm is one of"):
            bench.arm_dir(arm, model)
    # a done blind result is not asked again
    again: list = []
    bench.run_arm(run, "claude", stub_exec(again), "blind", None, [q], None, 1, bench.Halt())
    assert again == []


def test_the_runs_needle_decides_which_questions_never_run_at_once(built, tmp_path):
    run = trimmed(built, tmp_path)
    by_base: dict[str, list[dict]] = {}
    for q in run.plan().questions:
        by_base.setdefault(bench.base_template(q), []).append(q)
    pair = next(qs[:2] for qs in by_base.values() if len(qs) >= 2)

    def overlap(of: bench.Run, arm: str) -> bool:
        seen: list = []
        bench.run_arm(of, "claude", stub_exec(seen, 0.2), arm, None, pair, None, 2, bench.Halt())
        (a0, a1), (b0, b1) = ((min(t for e, t in seen if e.question is q), max(t for e, t in seen if e.question is q)) for q in pair)
        return a0 < b1 and b0 < a1

    assert not overlap(run, "files"), "one needle: one at a time"
    shutil.rmtree(run.folder / "results")
    assert overlap(marked(run), "blind"), "Marked.needle is the qid: they run together, in the blind arm too"


def test_a_home_links_the_snapshots_media_and_copies_the_rest(built, tmp_path):
    snapshot = built.snapshot()
    home = bench.copy_home(snapshot, tmp_path / "linked", FAKE_MODEL)
    media = [p for folder in bench.MEDIA_FOLDERS for p in (snapshot / folder).rglob("*") if p.is_file()]
    assert media, "the snapshot holds the media it was loaded with"
    for path in media:
        twin = home / path.relative_to(snapshot)
        assert os.path.samefile(path, twin), path
        assert not os.access(path, os.W_OK), "the snapshot's media is read-only, and a link shares that"
        with pytest.raises(PermissionError):
            twin.open("ab")
    index = home / "index.sqlite"
    assert index.is_file() and not os.path.samefile(index, snapshot / "index.sqlite"), "the index is a copy"
    assert config.load(home).decider_model == FAKE
    others = [p for p in snapshot.rglob("*") if p.is_file() and p.relative_to(snapshot).parts[0] not in bench.MEDIA_FOLDERS
              and (home / p.relative_to(snapshot)).exists()]
    assert others and not any(os.path.samefile(p, home / p.relative_to(snapshot)) for p in others), "all else is copied"
    # the copy's index is its own: a write there leaves the snapshot's
    before = bench.kept_items(snapshot)
    with closing(sqlite3.connect(index)) as db:
        db.execute("UPDATE items SET keep = 1")
        db.commit()
    assert bench.kept_items(snapshot) == before != bench.kept_items(home)


def test_the_haystacks_open_refuses_a_run_of_another_dataset(built, tmp_path):
    run = trimmed(built, tmp_path)
    file = run.folder / "run.json"
    meta = json.loads(file.read_text(encoding="utf-8"))
    file.write_text(json.dumps({**meta, "dataset": "egolife"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"is an egolife run: use benchmarks/egolife\.py"):
        bench.Run.open(run.folder)
    file.write_text(json.dumps({**meta, "dataset": "haystack"}), encoding="utf-8")
    assert bench.Run.open(run.folder) == run
    assert "dataset" not in meta and bench.Run.open(built.folder) == built, "a run from before datasets is a haystack's"


def test_the_needles_p_takes_the_tolerance_it_is_given():
    truth = T0 + 3600
    moments = [[truth, truth]]
    sightings = [{"t0": config.iso(truth + 20), "t1": config.iso(truth + 22), "p": 0.8}]
    candidates = [{"t": config.iso(truth + 25), "p": 0.9}]
    assert haystack.TOL_S < 20
    assert bench.needle_p(moments, sightings, candidates) is None, "the haystack's tolerance"
    assert bench.needle_p(moments, sightings, candidates, tol=21) == 0.8
    assert bench.needle_p(moments, sightings, candidates, tol=30.0) == 0.9
    assert bench.needle_p(moments, [], [], tol=30.0) is None


def test_a_sighting_or_candidate_has_one_time_or_a_span_and_the_oracle_reads_both():
    at = T0 + 3600
    assert bench.span_of({"t": config.iso(at)}) == (pytest.approx(at, abs=0.001),) * 2
    assert bench.span_of({"t0": config.iso(at), "t1": config.iso(at + 5)}) == (pytest.approx(at, abs=0.001), pytest.approx(at + 5, abs=0.001))
    moments = [[at, at]]
    assert bench.needle_p(moments, [{"t": config.iso(at + 3), "p": 0.7}], []) == 0.7, "a sighting of one instant"
    assert bench.needle_p(moments, [{"t0": config.iso(at + 3), "t1": config.iso(at + 6), "p": 0.6}], []) == 0.6
    assert bench.needle_p(moments, [], [{"t0": config.iso(at - 4), "t1": config.iso(at - 1), "p": 0.5}]) == 0.5, "a candidate that spans"
    assert bench.needle_p(moments, [{"t": config.iso(at + 50), "p": 0.9}], [{"t": config.iso(at + 80), "p": 0.9}]) is None


def test_the_crossover_looks_at_the_levels_it_is_given_in_their_order():
    level = level_of_pairs
    ages = ("0-2h", "2-6h", "6-24h", "24h+")
    levels = {"24h+": level(2, 12, 20), "2-6h": level(10, 15, 20), "0-2h": level(10, 10, 20)}
    got = bench.crossover(levels, order=ages)
    assert (got["level"], got["by"], got["lead"]) == ("2-6h", ["accuracy"], pytest.approx(0.25))
    assert bench.crossover(levels) is None, "by default it reads part, day and week"
    thin = {"0-2h": level(0, 5, 9), "6-24h": level(0, 5, 20)}
    assert bench.thin_levels(thin, order=ages) == {"0-2h": 9} and bench.thin_levels(thin) == {}


def test_the_real_agents_pass_the_timeout_the_folder_and_the_notes_of_each_arm(tmp_path, monkeypatch):
    calls: dict[str, list] = {"claude": [], "codex": []}
    monkeypatch.setattr(claude_run, "ask_claude", lambda prompt_text, run_dir, work, **kw: calls["claude"].append(kw) or {})
    monkeypatch.setattr(codex_run, "ask_codex", lambda prompt_text, home, run_dir, **kw: calls["codex"].append((home, kw)) or {})
    q, notes, home, data = sq("image-photo-sign-1"), FAKE_MODEL.about.notes, tmp_path / "home", tmp_path / "data"
    arms = (("blind", None, None, None), ("files", None, None, data), ("mms", FAKE_MODEL, home, None))
    for agents, timeout in ((bench.real_agents(timeout=123.0), 123.0), (bench.real_agents(), bench.TIMEOUT_S)):
        for arm, model, home_, data_ in arms:
            e = bench.Exec("x", arm, model, q, "the prompt", tmp_path / "run", tmp_path / "w", home_, data_, tmp_path / "bin", ("d",), 1)
            agents["claude"](e)
            agents["codex"](e)
            kw = calls["claude"][-1]
            assert (kw["arm"], kw["home"], kw["data"], kw["timeout"], kw["deny"]) == (arm, home_, data_, timeout, ("d",))
            assert kw["notes"] == ((notes or None) if model else None)
            codex_home, kw = calls["codex"][-1]
            assert (kw["arm"], codex_home, kw["timeout"], kw["deny"], kw["work"]) == (arm, home_, timeout, ("d",), tmp_path / "w")
            assert kw["notes"] == (notes if model else ""), "only the mms arm reads a model's notes"
            assert "data" not in kw, "Codex is told the folder in its prompt, not its arguments"
