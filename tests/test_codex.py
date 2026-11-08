"""Codex recalls a synthetic day over `mms mcp` (TESTBED.md, Codex recall): 19 questions, one `codex exec` each, graded by
rule. It spends the Codex plan's quota, so it is opt-in and runs alone:

    uv run pytest -m "codex and not gpu" -n 0 -s     C1: 7 questions, `find` and `look` only (no decider), about 10 min
    uv run pytest -m "codex and gpu" -n 0 -s        C2: 12 questions with real d1, after the GPU tests pass, about 30 min

`MMS_CODEX_MODEL` (default gpt-6.1-terra), `MMS_CODEX_EFFORT` (high) and `MMS_CODEX_GUIDANCE=none` (mms's tool schemas only,
no instructions or skill) change the run. After each exec the week's plan use is read from that exec's own rollout file; at
60% or more, or when it cannot be read, the remaining questions are skipped. Codex is offered mms's `ask`, `find`, `look`
and `label` only. The prompts, every tool call and result, the answers and the grades go
to `<pytest basetemp>/codex-run/<stamp>/` (printed). One attempt per question: a sample, not a rate.

The graders are pure rules (`benchmarks/grade.py`: casefold, number words to digits, times to minutes), pinned by a fast test that
runs no Codex. A test of `-m gpu` alone never reaches Codex: these tests skip unless `codex` is in the `-m` expression."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import codex_run
import day
import pytest
from conftest import Daemon, write_config
from grade import code, near, norm, numbers, unknown

STOP_AT = 60.0  # percent of the plan's use at which the remaining questions are skipped
D1 = Path(os.environ["MMS_MODELS"], "d1-Omni-2B").as_posix() if os.environ.get("MMS_MODELS") else None
QUOTA: dict = {"used": None, "ran": False}  # the last weekly plan use read, in percent; whether an exec has run


# -- graders ----------------------------------------------------------------------------------------------------------

def invoice_total(a: str) -> bool:
    """342.50 as the day's invoice says it, or as a reader writes it ("342.5", "342,50 EUR"), or "unknown": an agent that
    cannot read a PDF may say so, and must not invent a total."""
    return unknown(a) or re.search(r"\b342[.,]50?\b", norm(a)) is not None


def rule_3(a: str) -> bool:
    counts = re.sub(r"(?:err[- ]?)?\b(?:503|429)\b", " ", norm(a))
    return code(a, "503") and not code(a, "429") and "5" in numbers(counts)


class Question(NamedTuple):
    n: int
    tier: str  # C1 (find and look, no decider) or C2 (real d1)
    text: str
    truth: str
    grade: Callable[[str], bool]


QUESTIONS = [
    Question(1, "C1", "What's the gate code from the email this morning?", "4821", lambda a: "4821" in numbers(a)),
    Question(2, "C1", "How many days do we keep logs for now?", "30 (45 was the morning's, 45 is wrong)",
             lambda a: "30" in numbers(a) and "45" not in numbers(a)),
    Question(3, "C1", "Which error showed up most in the server log, and how many times?", "ERR-503, 5", rule_3),
    Question(4, "C1", "When did the deploy finish?", "12:07", lambda a: near(a, "12:07", 1)),
    Question(5, "C1", "What did I note about the dentist?", "moved to Thursday 9:30",
             lambda a: "thursday" in norm(a) and near(a, "9:30", 0.5)),
    Question(6, "C1", "What's the Wi-Fi password?", "unknown", unknown),
    Question(7, "C2", "When did the doorbell ring last?", "16:40", lambda a: near(a, "16:40", 2)),
    Question(8, "C2", "How many times did the doorbell ring?", "3", lambda a: "3" in numbers(a)),
    Question(9, "C2", "What error code was on my screen in the morning?", "E-4417",
             lambda a: re.search(r"\be-?4417\b", norm(a)) is not None),
    Question(10, "C2", "When did the disk-full error first appear?", "10:00", lambda a: near(a, "10:00", 1)),
    Question(11, "C2", "What did the whiteboard say?", "SHIP 2.1 FRI",
             lambda a: all(w in norm(a) for w in ("ship", "2.1", "fri"))),
    Question(12, "C2", "Did a phone ring during the 2 pm call, and when?", "yes, 14:03",
             lambda a: re.search(r"\byes\b", norm(a)) is not None and near(a, "14:03", 1)),
    Question(13, "C2", "Was the garage light on or off when the garage camera last saw it?", "off",
             lambda a: re.search(r"\boff\b", norm(a)) is not None and re.search(r"\bon\b", norm(a)) is None),
    Question(14, "C2", "Which app was on my screen at 3 pm?", "VS Code", lambda a: "code" in norm(a)),
    Question(15, "C2", "What level did I park on?", "3", lambda a: "3" in numbers(a)),
    Question(16, "C2", "What did Sam's voice note say?", "bring the blue folder; unknown accepted",
             lambda a: unknown(a) or all(w in norm(a) for w in ("blue", "folder"))),
    Question(17, "C2", "When was the screen showing the error, and what did it say?", "10:00:20-10:00:50, E-4417 disk full",
             lambda a: near(a, "10:00:35", 1) and re.search(r"\be-?4417\b", norm(a)) is not None and "disk" in norm(a)),
    Question(18, "C1", "What's the total on the invoice PDF?", "342.50 (unknown accepted)", invoice_total),
    Question(19, "C2", "What's the total on the invoice PDF?", "342.50 (unknown accepted; d1's ask names the PDF as not read)",
             invoice_total),
]

# per question: answers that must pass, and answers that must not
ANSWERS = {
    1: (["4821", "The gate code is 4821.", "four eight two one"], ["1234", "4281", "unknown"]),
    2: (["30", "30 days", "thirty days"], ["45", "45 days", "30 or 45", "unknown"]),
    3: (["ERR-503, 5 times", "ERR-503 (five times)", "503, 5"], ["ERR-429, 2", "ERR-503, 2", "ERR-503", "unknown"]),
    4: (["12:07", "12:07:31", "12:08"], ["12:30", "12:05", "unknown"]),
    5: (["moved to Thursday 9:30", "Thursday at 9:30 am", "thursday 09:30"], ["Friday 9:30", "Thursday 10:30", "unknown"]),
    6: (["unknown", "Unknown."], ["hunter2", "the password is 12345678", "not known"]),
    7: (["16:40", "4:40 pm", "4:41 PM"], ["11:20", "13:05", "16:50", "unknown"]),
    8: (["3", "three times"], ["2", "4", "one", "unknown"]),
    9: (["E-4417", "e-4417 disk full"], ["E-4471", "E-4416", "unknown"]),
    10: (["10:00", "10:00:20", "10:01"], ["10:05", "09:00", "unknown"]),
    11: (["SHIP 2.1 FRI", "Ship 2.1 Friday"], ["SHIP 2.0 FRI", "SHIP 2.1", "unknown"]),
    12: (["yes, 14:03", "Yes at 2:03 pm", "yes 14:04"], ["no", "yes, 14:30", "yes", "14:03"]),
    13: (["off", "Off."], ["on", "it was on", "unknown"]),
    14: (["VS Code", "Visual Studio Code"], ["Chrome", "unknown"]),
    15: (["3", "level 3", "Level three"], ["2", "level 2", "unknown"]),
    16: (["bring the blue folder", "unknown"], ["bring the red folder", "call me back"]),
    17: (["10:00:20-10:00:50, E-4417 disk full", "10:00 E-4417: disk full"],
         ["E-4417 disk full", "10:00:20, disk full", "11:00, E-4417 disk full"]),
    18: (["342.50", "342.50 EUR", "342.5", "342,50 \u20ac", "Total 342.50 EUR", "unknown"],
         ["324.50", "34.25", "342", "1342.50", "3425", "the total is 34.5"]),
}
ANSWERS[19] = ANSWERS[18]  # the same question, put to Codex with real d1 behind `ask`


def test_the_graders_grade_as_the_table_says():
    assert [q.n for q in QUESTIONS] == list(range(1, 20)) and set(ANSWERS) == {q.n for q in QUESTIONS}
    for q in QUESTIONS:
        right, wrong = ANSWERS[q.n]
        assert all(q.grade(a) for a in right), f"question {q.n} must accept {[a for a in right if not q.grade(a)]}"
        assert not any(q.grade(a) for a in wrong), f"question {q.n} must refuse {[a for a in wrong if q.grade(a)]}"
    assert [q.tier for q in QUESTIONS].count("C1") == 7 and [q.tier for q in QUESTIONS].count("C2") == 12
    abstaining = {q.n for q in QUESTIONS if q.grade("unknown")}
    assert abstaining == {6, 16, 18, 19}, "only question 6 expects unknown; 16, 18 and 19 (the invoice PDF no tool reads) accept it"


def stop_reason() -> str | None:
    """Why the remaining questions are skipped (the week's use is high or unreadable), else None."""
    if QUOTA["ran"] and QUOTA["used"] is None:
        return "weekly use unreadable"
    if QUOTA["used"] is not None and QUOTA["used"] >= STOP_AT:
        return f"the plan is at {QUOTA['used']:g}% of its use; the run stops at {STOP_AT:g}%"
    return None


def test_the_harness_reads_answers_events_and_the_quota_and_refuses_what_it_must(tmp_path, monkeypatch):
    assert codex_run.answer_of("thinking\n**ANSWER: 4821**") == "4821"
    assert codex_run.answer_of("ANSWER: no\nmore\nANSWER: unknown") == "unknown" and codex_run.answer_of("4821") is None
    week, hours = {"used_percent": 14, "window_minutes": 10080, "resets_at": 1}, {"used_percent": 90, "window_minutes": 300}
    lines = [json.dumps({"payload": {"rate_limits": {"primary": {**week, "used_percent": 12.5}, "secondary": None}}}),
             "not json", json.dumps({"payload": {"info": 1}}), json.dumps({"payload": {"rate_limits": None}}),
             json.dumps({"payload": {"rate_limits": {"primary": week, "secondary": None}}})]
    assert codex_run.percent_in(lines) == 14.0 and codex_run.percent_in(["{}"]) is None
    # the week is read from either window, and never the 5-hour one
    assert codex_run.percent_in([json.dumps({"rate_limits": {"primary": hours, "secondary": week}})]) == 14.0
    for limits in ({"primary": hours, "secondary": None}, {"primary": None, "secondary": None}, {"primary": {"used_percent": 14}},
                   {"primary": {**week, "used_percent": None}}, {"primary": {**week, "used_percent": "14"}}):
        assert codex_run.percent_in([json.dumps({"rate_limits": limits})]) is None, limits
    monkeypatch.setitem(QUOTA, "ran", False)
    monkeypatch.setitem(QUOTA, "used", None)
    assert stop_reason() is None  # nothing has run yet
    monkeypatch.setitem(QUOTA, "ran", True)
    assert stop_reason() == "weekly use unreadable"
    monkeypatch.setitem(QUOTA, "used", 59.0)
    assert stop_reason() is None
    monkeypatch.setitem(QUOTA, "used", 60.0)
    assert "60%" in stop_reason()

    # the exec is offered mms's read tools only: a command line without --tools is refused
    run = tmp_path / "run"
    argv = codex_run.argv_of(["codex"], "m", "high", run, tmp_path / "home")
    assert "'--tools','ask,find,look,label'" in next(a for a in argv if a.startswith("mcp_servers.mms.args="))
    with pytest.raises(codex_run.CodexMissing, match="--tools"):
        codex_run.argv_of(["codex"], "m", "high", run, tmp_path / "home", tools="")

    call = {"type": "item.completed", "item": {"type": "mcp_tool_call", "server": "mms", "tool": "find"}}
    assert codex_run.check_event(call) is None and codex_run.check_event({"type": "turn.started"}) is None
    assert codex_run.check_event({"type": "item.started", "item": {"type": "command_execution"}})
    for plan in ("item.started", "item.updated", "item.completed"):  # Codex's own plan list stays inside the sandbox
        assert codex_run.check_event({"type": plan, "item": {"type": "todo_list", "items": []}}) is None
    assert codex_run.check_event({"type": "item.started", "item": {"type": "file_change"}})
    assert codex_run.check_event({"type": "item.started", "item": {"type": "mcp_tool_call", "server": "x", "tool": "t"}})
    assert codex_run.check_event({"type": "item.started", "item": {"type": "mcp_tool_call", "server": "codex",
                                                                    "tool": "list_mcp_resources"}}) is None

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    codex_run.check_personal()
    (tmp_path / "AGENTS.md").write_text("")
    codex_run.check_personal()  # an empty file adds no message
    (tmp_path / "AGENTS.md").write_text("always reply in French")
    with pytest.raises(codex_run.CodexMissing, match="personal Codex instructions"):
        codex_run.check_personal()


# -- the runs ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def run_root(request, tmp_path_factory):
    """Where this run's logs go; the guards that keep a `-m gpu` run from spending the plan."""
    if not re.search(r"(?<!not )\bcodex\b", request.config.getoption("markexpr") or ""):
        pytest.skip("asks Codex (the plan's quota): run with -m codex")
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("Codex runs one question at a time, in order: use -n 0")
    if not shutil.which("codex"):
        pytest.skip("the codex command is not on PATH")
    root = tmp_path_factory.getbasetemp() / "codex-run" / time.strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True)
    print(f"\ncodex run logs: {root}")
    return root


@pytest.fixture(scope="session")
def the_day(run_root):
    """The day of `tests/day.py` for yesterday, with Windows speech."""
    start = day.yesterday()
    try:
        events = day.build(run_root / "day", start, speech=True)
    except day.SpeechUnavailable as error:
        pytest.skip(str(error))
    return run_root / "day", events, start


def world(name: str, decider: str, run_root, the_day):
    folder, events, _ = the_day
    home = run_root / f"home-{name}"
    home.mkdir()
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\nremember_max_mb = 50\n" + decider)
    daemon = Daemon(home).start()
    try:
        day.load(daemon.api, folder, events)
    except BaseException:
        daemon.stop()
        raise
    return home, daemon


@pytest.fixture(scope="session")
def c1_home(run_root, the_day):
    """A daemon holding the day, with no decider: `ask` fails, so Codex has `find` and `look`."""
    home, daemon = world("c1", "", run_root, the_day)
    yield home
    daemon.stop()


@pytest.fixture(scope="session")
def c2_home(run_root, the_day):
    """A daemon holding the day, with d1."""
    if D1 is None:
        pytest.skip("C2 needs d1-Omni-2B: set MMS_MODELS to the folder that holds it")
    home, daemon = world("c2", f'[decider]\nmodel = "{D1}"\n', run_root, the_day)
    yield home
    daemon.stop()


def marks(q: Question) -> list:
    return [pytest.mark.codex, *([pytest.mark.gpu, pytest.mark.xdist_group("gpu")] if q.tier == "C2" else [])]


@pytest.mark.parametrize("question", [pytest.param(q, id=f"q{q.n:02d}-{q.tier}", marks=marks(q)) for q in QUESTIONS])
def test_codex_recalls_the_day(request, question, run_root, the_day):
    before = QUOTA["used"]
    if reason := stop_reason():
        pytest.skip(reason)
    home = request.getfixturevalue(f"{question.tier.lower()}_home")
    run_dir = run_root / f"q{question.n:02d}"
    result = codex_run.ask_codex(codex_run.prompt(question.text, the_day[2]), home, run_dir,
                                 guidance=os.environ.get("MMS_CODEX_GUIDANCE", "default"))
    used = result["used_percent"]
    QUOTA["used"], QUOTA["ran"] = used, True  # None stops the remaining questions: weekly use unreadable
    if result["problem"] and result["problem"].startswith("limit"):
        QUOTA["used"] = 100.0
    answer = result["answer"]
    passed = result["problem"] is None and answer is not None and question.grade(answer)
    (run_dir / "result.json").write_text(json.dumps({"question": question.text, "truth": question.truth, "answer": answer,
                                                     "passed": passed, "problem": result["problem"],
                                                     "seconds": result["seconds"], "used_percent": used}, indent=1))
    print(f"\nQ{question.n} {question.tier} {question.text}\n  answer: {answer!r}  (truth: {question.truth})  "
          f"{'PASS' if passed else 'FAIL'}  {result['seconds']:g} s  tools: "
          f"{[t['tool'] for t in result['tools']]}\n  plan use {before if before is not None else '?'} -> "
          f"{used if used is not None else '?'}%  logs: {run_dir}")
    assert result["problem"] is None, f"the exec did not finish cleanly: {result['problem']} (logs: {run_dir})"
    assert answer is not None, f"no ANSWER line in: {result['final'][-300:]!r} (logs: {run_dir})"
    assert passed, f"{answer!r} is not {question.truth!r} (logs: {run_dir})"
