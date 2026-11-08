"""The skill (DESIGN.md §15): valid for Claude Code and Codex, model-neutral, with one notes file for each family of models, and
every example call in it is a call `ask` accepts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import anyio
import pytest
from mcp_session import connected, text
from test_ask import api_under, facade

from mms import decider, install, jobs

FOLDER = Path(install.SKILL_SOURCE)
SKILL = FOLDER / "SKILL.md"
MODELS = FOLDER / "models"
TEXT = SKILL.read_text(encoding="utf-8")
NOTES = {file.name: file.read_text(encoding="utf-8") for file in sorted(MODELS.glob("*.md"))}


def examples_in(markdown: str) -> list[dict]:
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)```", markdown, re.DOTALL)]


EXAMPLES = examples_in(TEXT)
NOTE_EXAMPLES = {name: examples_in(body) for name, body in NOTES.items()}
EVERY_EXAMPLE = [("SKILL.md", example) for example in EXAMPLES] + [
    (name, example) for name, found in NOTE_EXAMPLES.items() for example in found]
TIME_WORDS = re.compile(r"\b(yesterday|today|tonight|this morning|last (week|night|month)|\d\s?(am|pm)|monday|tuesday|"
                        r"wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
TYPES = set(decider.TYPES)
EVERYWHERE = {"noul", "choice", "score"}  # the types every model answers
ABOUTS = decider.known_abouts()
MODEL_WORDS = re.compile(r"\b(d1|d3|h2o|pplx|omni|lightning|perplexity|liquid|vllm|qwen)\b", re.IGNORECASE)
NOT_IN_SKILL = ("0.8", "640", "1.6 million", "logit", "16 questions")
SKILL_LINES, NOTES_LINES = 400, 150


def head_of(markdown: str) -> dict:
    return dict(line.split(": ", 1) for line in markdown.split("---\n")[1].strip().splitlines())


def ids(pair) -> str:
    name, example = pair
    return f"{name}-{next(iter(example))}{len(str(example))}"


def test_the_frontmatter_names_the_skill_and_says_when_to_use_it():
    fields = head_of(TEXT)
    assert set(fields) == {"name", "description"}
    assert fields["name"] == SKILL.parent.name == "mms-questions"
    assert 40 < len(fields["description"]) <= 1024 and "Use it" in fields["description"]
    assert not MODEL_WORDS.search(fields["description"]), "the skill is for every model"


def test_the_skill_names_no_model_and_none_of_the_numbers_of_one():
    assert not MODEL_WORDS.search(TEXT), MODEL_WORDS.search(TEXT)
    for about in ABOUTS:
        assert about.name.lower() not in TEXT.lower(), about.name
    for banned in NOT_IN_SKILL:
        assert banned not in TEXT, banned


def test_the_skill_points_to_the_ask_description_and_to_the_models_notes():
    opening = TEXT.split("## Examples")[0]
    assert "`ask` tool description" in opening and "notes file" in opening and "models/" in opening
    assert "Read that file before your" in opening


def test_the_skill_is_under_400_lines_and_each_notes_file_under_150():
    assert TEXT.count("\n") < SKILL_LINES
    for name, body in NOTES.items():
        assert body.count("\n") < NOTES_LINES, name


def test_every_question_type_has_a_parser_in_ask(home):
    assert set(jobs.load_script(home, "ask").PARSERS) == set(decider.TYPES)


def test_the_examples_come_first():
    assert TEXT.index("## Examples") < min(TEXT.index(heading) for heading in
                                           ("## Which tool", "## Answer types", "## Cost", "## Jobs"))
    assert len(EXAMPLES) >= 5


def test_the_skills_examples_use_only_what_every_model_answers():
    for example in EXAMPLES:
        for named in (example.get("questions") or {"": {"type": "noul"}}).values():
            assert named["type"] in EVERYWHERE, example
            assert set(named) <= {"type", "instructions", "criteria", "hit"}, example
            if named["type"] == "noul":
                assert named.get("hit", True) is True, "a yes/no hit is left out or true, so the model's default applies"
        if "question" in example and "answers" not in example:
            assert example.get("hit", True) is True, example


@pytest.mark.parametrize("pair", EVERY_EXAMPLE, ids=ids)
def test_an_example_asks_nothing_the_design_forbids(pair):
    _, example = pair
    asked = [example["question"]] if "question" in example else [named["instructions"] for named in
                                                                  example["questions"].values()]
    for question in asked:
        assert not TIME_WORDS.search(question + str(example.get("context", ""))), "the range goes in since and until"
        assert " and " not in question.lower() and " or " not in question.lower(), "one thing per question"
    assert "since" in example, "every example gives its range"
    for named in (example.get("questions") or {}).values():
        assert named["type"] in TYPES
    if "answers" in example:
        hits = example["hit"] if isinstance(example["hit"], list) else [example["hit"]]
        assert len(example["answers"]) >= 2 and set(hits) <= set(example["answers"])
    assert len(example.get("context", "")) < 200


def test_every_example_is_a_call_the_ask_tool_takes(home, tmp_path, memory):
    """Through the MCP schema, which d1's fake decider answers: the shipped script takes every parameter as written."""
    async def main():
        async with connected(home, tmp_path) as client:
            for name, example in EVERY_EXAMPLE:
                result = await client.call_tool("ask", example)
                asked = text(result)
                assert not result.is_error and asked["state"] in ("done", "running"), (name, example, asked)
                assert "error" not in asked, (name, example, asked)
                assert all("unknown" not in warning for warning in asked.get("warnings", [])), (name, example, asked)

    anyio.run(main)


@pytest.mark.parametrize("about", ABOUTS, ids=lambda about: about.name)
def test_the_skills_examples_and_the_notes_of_a_model_run_against_it(home, tmp_path, served, about):
    """Each model of the backends takes every example of the skill and of its own notes: no refusal, no unknown parameter.
    The decider is a `python:` file with that model's About, answering as the fake decider does."""
    api = api_under(home, served, facade(tmp_path, about.name, known=True))
    mine = [("SKILL.md", example) for example in EXAMPLES] + [
        (about.notes, example) for example in NOTE_EXAMPLES.get(about.notes, [])]
    assert len(mine) > len(EXAMPLES), f"{about.name}'s notes {about.notes!r} have examples"
    for name, example in mine:
        job = api.call("jobs/start", script="ask", params=example, wait_s=50)
        while job["state"] in ("queued", "running"):
            job = api.call("jobs/get", job=job["job"], wait_s=50)
        assert job["state"] == "done" and job["error"] is None, (about.name, name, example, job["error"])
        assert all("unknown" not in warning for warning in job["result"].get("warnings", [])), (name, example)


def test_every_models_notes_exist_and_every_notes_file_is_some_models():
    assert ABOUTS, "no backend gave an About"
    named = {about.notes for about in ABOUTS}
    assert "" not in named, [about.name for about in ABOUTS if not about.notes]
    assert named <= set(NOTES), f"notes named by an About but not in models/: {sorted(named - set(NOTES))}"
    assert set(NOTES) <= named, f"notes in models/ that no About names: {sorted(set(NOTES) - named)}"


@pytest.mark.parametrize("about", ABOUTS, ids=lambda about: about.name)
def test_a_models_notes_say_what_its_about_says(about):
    body = NOTES[about.notes]
    low = body.lower()
    assert about.name.lower() in low, f"{about.notes} does not name {about.name}"
    for kind in about.types:
        assert re.search(rf"\b{kind}\b", body), f"{about.notes} does not mention the {kind} type of {about.name}"
    if "hit" not in about.measured:
        assert "assumed" in low and f"{about.hit:g}" in body, f"{about.notes}: the hit of {about.name} is assumed"
    if "audio" not in about.reads:
        assert "hears nothing" in low or "hear nothing" in low, f"{about.notes} must say {about.name} hears nothing"
    if about.questions_per_read == 1:
        assert re.search(r"one question a pass|one at a time", low), f"{about.notes}: a pass for each question"
    if about.questions_per_read > 1:
        assert f"{about.questions_per_read} questions" in body
    if set(about.question_keys) == {"criteria"}:
        assert "criteria" in body and "`multiple`" in body, f"{about.notes} must say the keys it refuses"
    if not set(decider.TYPES) - set(about.types):
        assert "all eight types" in low or "all 8 types" in low
