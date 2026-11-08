"""`ask` and d1's answer contract: one parser for each type, in d1's own shapes, a hit for a number,
unreadable answers as a counted warning, and `max_px`. The fixtures and helpers are test_ask's."""

from __future__ import annotations

import dataclasses
import json
import re

import pytest
from conftest import write_config
from test_ask import (SECONDS, T0, api, api_under, decider_log, facade, index, media, scan,  # noqa: F401  fixtures and helpers
                      store)

from mms import decider_d1, jobs, reader
from mms.store import Store

D1_ANSWERS = [  # (question, d1's answer, hit, whether it counts)
    ({"type": "noul"}, {"type": "noul", "noul": 0.93}, 0.8, True),
    ({"type": "noul"}, {"type": "noul", "noul": 0.2}, 0.8, False),
    ({"type": "choice", "criteria": {"a": None, "b": None}},
     {"type": "choice", "choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}, "a", True),
    ({"type": "choice", "criteria": {"a": None, "b": None}, "multiple": True},
     {"type": "choice", "choices": ["b"], "probabilities": {"a": 0.2, "b": 0.7}}, ["a", "b"], True),
    ({"type": "choice", "criteria": {"a": None, "b": None}, "multiple": True},
     {"type": "choice", "choices": [], "probabilities": {"a": 0.2, "b": 0.1}}, "a", False),
    ({"type": "score", "criteria": ["no", "yes"]},
     {"type": "score", "score": 1.0, "confidence": 0.9, "probabilities": {"0": 0.1, "1": 0.9}, "legend": {}}, 1, True),
    ({"type": "text"}, {"type": "text", "text": "v1.2", "span": [3, 7], "found": 0.9}, True, True),
    ({"type": "text"}, {"type": "text", "text": None, "span": None, "found": 0.7}, True, False),  # found, but no value
    ({"type": "date"}, {"type": "date", "date": "2026-10-31", "span": [0, 10], "found": 0.9, "text": "31 Oct"}, True, True),
    ({"type": "number"}, {"type": "number", "number": 71, "span": [4, 6], "found": 0.9, "text": "71", "unit": None}, True, True),
    ({"type": "number"}, {"type": "number", "number": None, "span": None, "found": 0.1, "text": None, "unit": None}, True, False),
    ({"type": "number", "range": [0, 10], "unit": "beeps"},  # a ranged number has an interval and no `found`
     {"type": "number", "number": 3, "interval": [2, 4], "unit": "beeps"}, 3, True),
    ({"type": "number", "range": [0, 10], "unit": "beeps"},
     {"type": "number", "number": 2, "interval": [1, 3], "unit": "beeps"}, 3, False),
    ({"type": "box"}, {"type": "box", "box": {"top": 0, "bottom": 0.5, "left": 0, "right": 0.5}, "found": 0.9,
                       "confidence": 0.8}, True, True),
    ({"type": "box"}, {"type": "box", "box": None, "found": 0.1, "confidence": 0.0}, True, False),
    ({"type": "boxes"}, {"type": "boxes", "boxes": [{"box": {"top": 0, "bottom": 0.5, "left": 0, "right": 0.5},
                                                      "found": 0.9, "confidence": 0.8}], "count": 1,
                         "at_least": False, "presence": [0.9]}, True, True),
    ({"type": "boxes"}, {"type": "boxes", "boxes": [], "count": 0, "at_least": False, "presence": [0.1]}, True, False),
]


@pytest.mark.parametrize("question,answer,hit,counts", D1_ANSWERS,
                         ids=[f"{q['type']}-{i}" for i, (q, *_) in enumerate(D1_ANSWERS)])
def test_each_type_of_d1_answer_is_read_by_its_own_parser(home, question, answer, hit, counts):
    ask = jobs.load_script(home, "ask")
    question = {"instructions": "x", **question}
    got, p, label, _ = ask.judge({"q": question}, {"q": ask._rule("q", question, hit, decider_d1.ABOUT)}, {"q": answer})
    assert got is counts and 0.0 <= p <= 1.0 and label.startswith("q: ")


EDGES = {"top": 0.1504, "bottom": 0.2396, "left": 0.0612, "right": 0.27}
WHERE = "top 0.15, left 0.06, bottom 0.24, right 0.27"
WHERE_FAKE = "top 0.00, left 0.00, bottom 0.50, right 0.50"  # the box the fake decider finds


def test_a_box_answer_says_its_edges_rounded_to_2_decimals_in_said(home):
    ask = jobs.load_script(home, "ask")
    question = {"type": "box", "instructions": "x"}

    def judged(answer) -> tuple:
        return ask.judge({"q": question}, {"q": ask._rule("q", question, True, decider_d1.ABOUT)}, {"q": answer})

    def said(answer) -> tuple:
        return judged(answer)[:3]

    assert said({"type": "box", "box": EDGES, "found": 0.9, "confidence": 0.8}) == (True, 0.9, f"q: at {WHERE} (p 0.90)")
    assert judged({"type": "box", "box": EDGES, "found": 0.9})[3] == f"at {WHERE}", "the result's said is the edges alone"
    assert said({"type": "box", "box": None, "found": 0.1, "confidence": 0.0}) == (False, 0.1, "q: none (p 0.10)")
    assert judged({"type": "box", "box": None, "found": 0.1})[3] == "none"
    unsure = said({"type": "box", "box": {**EDGES, "top": 0.1}, "found": 0.3, "confidence": 0.2})
    assert unsure == (False, 0.3, "q: at top 0.10, left 0.06, bottom 0.24, right 0.27 (p 0.30)"),         "a box below 0.5 is a miss, and its said does not claim it was found"
    assert said({"type": "box", "box": {"top": 0, "left": 0, "bottom": 1, "right": 1}, "found": 0.9})[2] ==         "q: at top 0.00, left 0.00, bottom 1.00, right 1.00 (p 0.90)", "0 and 1 are edges of the picture"
    for box, why in (({"top": 0.1, "left": 0.1, "bottom": 0.2}, "its box is"), ({**EDGES, "top": "high"}, "its top is 'high'"),
                     ({**EDGES, "right": None}, "its right is None"), ("left", "its box is 'left'"),
                     ({**EDGES, "right": 1.5}, "its right is 1.5, outside 0.0 to 1.0"),
                     ({**EDGES, "left": -0.1}, "its left is -0.1, outside 0.0 to 1.0"),
                     ({**EDGES, "top": 0.5}, "edge past the opposite one"), ({**EDGES, "left": 0.9}, "edge past the opposite one")):
        with pytest.raises(ask.Unreadable, match=why):
            said({"type": "box", "box": box, "found": 0.9})


def test_a_boxes_answer_says_its_count_then_the_edges_of_each_box_in_said(home):
    ask = jobs.load_script(home, "ask")
    question = {"type": "boxes", "instructions": "x"}

    def judged(answer) -> tuple:
        return ask.judge({"q": question}, {"q": ask._rule("q", question, True, decider_d1.ABOUT)}, {"q": answer})

    def said(answer) -> tuple:
        return judged(answer)[:3]

    first = {"box": EDGES, "found": 0.9, "confidence": 0.8}
    second = {"box": {"top": 0.5, "bottom": 0.9, "left": 0.5, "right": 1}, "found": 0.7, "confidence": 0.6}
    assert said({"type": "boxes", "boxes": [first, second], "count": 2}) == (
        True, 0.9, f"q: 2 found: {WHERE}; top 0.50, left 0.50, bottom 0.90, right 1.00 (p 0.90)")
    assert said({"type": "boxes", "boxes": [first], "count": 1, "at_least": False})[2] == f"q: 1 found: {WHERE} (p 0.90)"
    assert judged({"type": "boxes", "boxes": [first, second], "count": 2})[3] == (
        f"2 found: {WHERE}; top 0.50, left 0.50, bottom 0.90, right 1.00"), "the count and the edges, with no name or p"
    assert said({"type": "boxes", "boxes": [first], "count": 16, "at_least": True})[2] ==         f"q: at least 16 found: {WHERE} (p 0.90)", "a count that is a floor says so"
    assert said({"type": "boxes", "boxes": [], "count": 0}) == (False, 0.0, "q: 0 found (p 0.00)")
    for boxes, why in (([{"found": 0.9}], "its box is None"), ([{"box": {"top": 0.1}, "found": 0.9}], "its box is"),
                       ([{"box": {**EDGES, "bottom": 2}, "found": 0.9}], "its bottom is 2, outside 0.0 to 1.0"),
                       ([{"box": {**EDGES, "top": 0.5}, "found": 0.9}], "edge past the opposite one"),
                       (["a box"], "its boxes are"), ("none", "its boxes are")):
        with pytest.raises(ask.Unreadable, match=why):
            said({"type": "boxes", "boxes": boxes, "count": 1})
    with pytest.raises(ask.Unreadable, match="its at_least is 'yes'"):
        said({"type": "boxes", "boxes": [first], "count": 1, "at_least": "yes"})


def test_a_box_sighting_and_its_label_say_where_the_thing_is(home, api, store, media):  # noqa: F811
    index(store, "cam", "video", media["video"], T0, SECONDS)
    for kind, said in (("box", f"at {WHERE_FAKE}"), ("boxes", f"1 found: {WHERE_FAKE}")):
        (s,) = scan(api, questions={"mug": {"type": kind, "instructions": "Is it red?", "hit": True}})["sightings"]
        assert s["said"] == said, "the edges alone: the question's name and the p are not repeated"
        assert any(label.text.startswith(f"mug: {said} (p 0.95), ") for label in store.labels(by="ask")), store.labels(by="ask")


def test_a_text_or_date_with_no_value_says_no_value_in_the_label_and_in_said(home):
    ask = jobs.load_script(home, "ask")
    for kind in ("text", "date"):
        question = {"type": kind, "instructions": "x"}
        rules = {"q": ask._rule("q", question, True, decider_d1.ABOUT)}
        nothing = {"type": kind, kind: None, "span": None, "found": 0.7}
        assert ask.judge({"q": question}, rules, {"q": nothing}) == (False, 0.7, "q: no value (p 0.70)", "no value")
        there = {"type": kind, kind: "v1.2", "span": [0, 4], "found": 0.9}
        assert ask.judge({"q": question}, rules, {"q": there}) == (True, 0.9, "q: v1.2 (p 0.90)", "v1.2")


def test_a_yes_no_hit_of_true_counts_at_the_models_own_threshold(home):
    ask = jobs.load_script(home, "ask")
    question = {"type": "noul", "instructions": "x"}
    for own in (0.6, 0.8, 0.95):
        about = dataclasses.replace(decider_d1.ABOUT, hit=own)
        rules = ask.rules_of({"q": question}, {"q": True}, about)
        for p in (own - 0.05, own, own + 0.05):
            got, level, label, said = ask.judge({"q": question}, rules, {"q": {"type": "noul", "noul": p}})
            assert got is (p >= own) and level == p, (own, p)
            assert label == (f"q: yes (p {p:.2f})" if p >= own else f"q: below {own:g} (p {p:.2f})"), label
            assert said is None, "a yes/no says nothing beyond p"
    rules = ask.rules_of({"q": question}, {"q": 0.9}, dataclasses.replace(decider_d1.ABOUT, hit=0.6))
    assert ask.judge({"q": question}, rules, {"q": {"type": "noul", "noul": 0.7}})[0] is False, "a number is the asker's own"


def test_a_hit_written_as_text_is_read_as_its_number_and_other_text_is_refused(home):
    ask = jobs.load_script(home, "ask")
    noul = {"type": "noul", "instructions": "x"}
    rule = ask._rule("q", noul, "0.5", decider_d1.ABOUT)
    assert rule({"p": 0.6})[0] is True and rule({"p": 0.4})[0] is False, "\"0.5\" is 0.5, not d1's own 0.8"
    score = {"type": "score", "instructions": "x", "criteria": ["no", "some", "yes"]}
    rule = ask._rule("q", score, " 2 ", decider_d1.ABOUT)
    assert rule({"level": 2.0})[0] is True and rule({"level": 1.0})[0] is False, "\" 2 \" is level 2"
    number = {"type": "number", "instructions": "x", "range": [0, 10]}
    rule = ask._rule("q", number, "3", decider_d1.ABOUT)
    assert [rule({"value": v, "interval": [v - 1, v + 1], "unit": None})[0] for v in (3, 2)] == [True, False], "\"3\" is 3"
    for hit in ("high", "", "nan", "inf", "1.5", 10 ** 400, ["0.5"], None):
        with pytest.raises(ValueError, match=re.escape(f"hit is a number from 0 to 1 here, not {hit!r}")):
            ask._rule("q", noul, hit, decider_d1.ABOUT)
    with pytest.raises(ValueError, match="hit is a finite number here, not inf"):
        ask._rule("q", number, float("inf"), decider_d1.ABOUT)


def test_the_command_line_passes_hit_on_as_text_for_the_question_to_read(home):
    ask = jobs.load_script(home, "ask")
    assert ask.cli(["Is it red?", "--hit", "0.5"])["hit"] == "0.5"
    params = ask.cli(["How many?", "--answers", "1,2,3", "--hit", "2"])
    assert params["hit"] == "2", "an option that looks like a number stays the option"
    questions, hits = ask.build(params)
    rules = ask.rules_of(questions, hits, decider_d1.ABOUT)
    answer = {"type": "choice", "choice": "2", "confidence": 0.9, "probabilities": {"2": 0.9}}
    assert ask.judge(questions, rules, {"How many?": answer})[0] is True
    assert ask.cli(["Which?", "--answers", "a,b,c", "--hit", "a, b"])["hit"] == ["a", "b"]


def test_the_command_line_flag_best_first_is_the_param(home):
    ask = jobs.load_script(home, "ask")
    assert ask.cli(["Is it red?", "--best-first", "--limit", "3"]) == {"question": "Is it red?", "limit": 3, "best_first": True}
    assert "best_first" not in ask.cli(["Is it red?"]), "left out unless asked for"
    assert "best_first" in ask.PARAMS


def test_an_answer_that_is_not_in_d1_shape_is_unreadable_and_says_which(home):
    ask = jobs.load_script(home, "ask")
    question = {"type": "number", "instructions": "x", "range": [0, 10]}
    rules = {"q": ask._rule("q", question, 3, decider_d1.ABOUT)}
    for answer, why in (({"type": "number", "number": "three", "interval": [1, 5]}, "its number is 'three'"),
                        ({"type": "number", "number": 3, "interval": "wide"}, "interval"),
                        ({"type": "noul", "noul": 0.9}, "'noul' answer to a 'number' question"),
                        ("yes", "not an object")):
        with pytest.raises(ask.Unreadable, match=why):
            ask.judge({"q": question}, rules, {"q": answer})
    with pytest.raises(ask.Unreadable, match="no answer"):
        ask.judge({"q": question}, rules, {})
    plain = {"type": "number", "instructions": "x"}  # not ranged: d1 gives `found`
    with pytest.raises(ask.Unreadable, match="found"):
        ask.judge({"q": plain}, {"q": ask._rule("q", plain, True, decider_d1.ABOUT)}, {"q": {"type": "number", "number": 3}})


def test_a_number_hits_at_the_least_value_and_says_the_value_with_its_interval(home, api, store, media):  # noqa: F811
    index(store, "mic", "audio", media["audio"], T0, 19)
    beeps = {"type": "number", "instructions": "loud", "range": [0, 10], "unit": "beeps"}
    found = scan(api, questions={"beeps": {**beeps, "hit": 5}}, sources=["mic"])
    assert found["sightings"] and found["sightings"][0]["said"] == "10 (5 to 10) beeps"
    assert scan(api, questions={"beeps": {**beeps, "hit": 11}}, sources=["mic"])["sightings"] == [], "10 is below 11"
    store.add_records("thermo", [(T0, json.dumps({"note": "the kettle reads 71 degrees"}))])
    reading = {"type": "number", "instructions": "contains:kettle"}
    (sighting,) = scan(api, questions={"reading": {**reading, "hit": True}}, sources=["thermo"])["sightings"]
    assert sighting["said"] == "71"
    over = scan(api, questions={"reading": {**reading, "hit": 80}}, sources=["thermo"])
    assert over["sightings"] == [] and over["candidates"][0]["said"] == "71"


def test_a_hit_that_does_not_fit_the_type_fails_the_job_as_bad_request(home, api, store, media):  # noqa: F811
    for questions, text in (({"t": {"type": "text", "instructions": "x", "hit": 0.5}}, "hit is true"),
                            ({"n": {"type": "number", "instructions": "x", "range": [0, 9], "hit": True}}, "least value"),
                            ({"s": {"type": "noul", "instructions": "x", "hit": 2}}, "from 0 to 1"),
                            ({"s": {"type": "noul", "instructions": "x", "hit": "high"}}, "not 'high'")):
        job = api.call("jobs/start", script="ask", params={"questions": questions, "since": T0 - 10}, wait_s=40)
        assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and text in job["error"]["message"], job
    job = api.call("jobs/start", script="ask", params={"question": "Is it red?", "max_px": -1}, wait_s=40)
    assert job["state"] == "failed" and "max_px" in job["error"]["message"]


def test_a_questions_item_not_in_d1_format_fails_the_job_and_says_the_fix(home, api):
    trial = {"blue_bottle": {"question": "Is a blue bottle visible?", "hit": 0.65}}  # a benchmark trial's shape
    for questions, texts in (
            (trial, ("question 'blue_bottle' has no type", "a typed question", "leave out questions and give question, answers", "hit (all of them)")),
            ({"a": {"question": "Is a blue bottle visible?"}}, ("question 'a' has no type",)),  # before the hit is looked at
            ({"x": {"type": "foo", "instructions": "y", "hit": 0.5}}, ("question 'x': type is one of noul, choice", "not 'foo'")),
            ({"x": {"type": "noul", "hit": 0.5}}, ("question 'x' needs instructions",)),
            ({"x": {"type": "noul", "question": "Is it?", "hit": 0.5}}, ("needs instructions", "instructions, not question")),
            ({"x": {"type": "choice", "instructions": "Which?", "answers": ["a", "b"], "hit": ["a", "b"]}},
             ("question 'x': a choice has criteria", "answers goes with question")),
            ({"x": {"type": "choice", "instructions": "Which?", "criteria": {}, "hit": "a"}}, ("a choice has criteria",)),
            ({"x": {"type": "score", "instructions": "How red?", "hit": 1}}, ("question 'x': a score has criteria", "2 to 10")),
            ({"x": {"type": "score", "instructions": "How red?", "criteria": ["none"], "hit": 1}}, ("a score has criteria",)),
            ({"x": {"type": "noul", "instructions": " ", "hit": 0.5}}, ("question 'x' needs instructions",)),
            ({"x": {"type": "noul", "instructions": "y"}}, ("no question has a `hit`",))):
        job = api.call("jobs/start", script="ask", params={"questions": questions, "since": T0 - 10}, wait_s=40)
        assert job["state"] == "failed" and job["error"]["code"] == "bad_request", job
        assert all(text in job["error"]["message"] for text in texts), job["error"]["message"]


LAMP = {"type": "noul", "instructions": "Is the lamp lit?", "hit": True}


@pytest.mark.parametrize("criteria", [
    {"true": "lit"}, {"false": "dark"}, {"true": "lit", "false": "dark", "maybe": "dim"}, {"true": "lit", "dark": "off"},
    {"true": "lit", "false": None}, {"true": "lit", "false": 3}, {"true": "", "false": "dark"}, {"true": "lit", "false": " "},
    {}, ["lit", "dark"], "lit"])
def test_a_yes_no_with_less_or_more_than_both_sides_of_criteria_is_refused_before_a_scan_and_says_the_fix(home, api, store,
                                                                                                         criteria):
    store.add_version("notes", "n.md", T0, data=b"the kettle\n")
    params = {"questions": {"lit": {**LAMP, "criteria": criteria}}, "since": T0 - 10, "until": T0 + 10}
    job = api.call("jobs/start", script="ask", params=params, wait_s=40)
    assert job["state"] == "failed" and job["error"]["code"] == "bad_request", job
    assert ("question 'lit': a yes/no's criteria says what both answers mean: "
            '{"true": "...", "false": "..."}, or leave criteria out') in job["error"]["message"], job["error"]["message"]
    assert decider_log(home) == [], "no window was scanned"
    assert api.call("status")["decider"]["state"] == "unloaded", "and the model was never loaded"


def test_a_yes_no_with_both_sides_of_criteria_is_accepted_and_the_model_gets_them(home, api, store):
    store.add_version("notes", "n.md", T0, data=b"the kettle\n")
    both = {"true": "the kettle is mentioned", "false": "something else is"}
    result = scan(api, questions={"lit": {**LAMP, "instructions": "contains:kettle", "criteria": both}})
    assert [s["source"] for s in result["sightings"]] == ["notes"]
    assert {tuple(q["options"]) for call in decider_log(home) for q in call["questions"].values()} == {("true", "false")}


def test_a_top_level_hit_or_answers_without_question_is_an_error_not_ignored(home, api):
    one = {"type": "noul", "instructions": "Is it red?"}
    for params, text in (({"questions": {"a": one}, "hit": 0.9}, "hit goes inside each question of questions"),
                         ({"questions": {"a": {**one, "hit": 0.9}}, "answers": ["x", "y"]}, "answers goes with question"),
                         ({"hit": 0.9}, "answers and hit go with a question")):
        job = api.call("jobs/start", script="ask", params={**params, "since": T0 - 10}, wait_s=40)
        assert job["state"] == "failed" and job["error"]["code"] == "bad_request" and text in job["error"]["message"], job


def test_a_choice_hit_listing_options_counts_when_any_one_is_picked(home):
    ask = jobs.load_script(home, "ask")
    questions, hits = ask.build({"question": "Which is on the table?", "answers": ["mug", "cup"], "hit": ["mug", "cup"]})
    rules = ask.rules_of(questions, hits, decider_d1.ABOUT)
    for picked, counts in (("mug", True), ("cup", True), ("none", False)):
        answer = {"type": "choice", "choice": picked, "confidence": 0.9, "probabilities": {picked: 0.9}}
        got, _, label, said = ask.judge(questions, rules, {"Which is on the table?": answer})
        assert got is counts and f": {picked} " in label and said == picked, (picked, label, said)


def test_windows_whose_answers_cannot_be_read_are_a_counted_warning_not_a_silent_skip(home, tmp_path, served):
    broken = tmp_path / "broken_decider.py"
    broken.write_text("""
from mms.decider import About
ABOUT = About(name="broken")
def decide(state, questions, media):
    return {name: {"type": question["type"]} for name, question in questions.items()}  # no `noul`
""", encoding="utf-8")
    write_config(home, f'[memory]\nloop_s = 0.1\nfree_gb = 0\n[decider]\nmodel = "python:{broken.as_posix()}"\n')
    api = served().api
    store = Store(home)
    store.add_records("thermo", [(T0 + 10 * i, json.dumps({"n": i, "pad": "z" * 1100})) for i in range(6)])
    store.close()
    result = scan(api, question="Is it red?", sources=["thermo"])
    assert result["sightings"] == [] and result["progress"]["windows"] == 3
    (warning,) = [w for w in result["warnings"] if "skipped" in w]
    assert warning.startswith("3 windows were skipped") and "answer could not be read" in warning and "noul" in warning


def test_max_px_raises_the_pictures_d1_sees_up_to_its_cap_and_the_estimate_of_what_that_costs(home, api, store, media):  # noqa: F811
    index(store, "hd", "video", media["hd"], T0, 2)
    index(store, "cam", "video", media["video"], T0 + 100, SECONDS)
    cap = decider_d1.ABOUT.max_px
    at_cap = {tuple(reader.fit(1920, 1080, cap))}

    def sizes(**params) -> set:
        before = len(decider_log(home))
        scan(api, question="Is it red?", sources=["hd"], **params)
        return {tuple(size) for call in decider_log(home)[before:] for size in call["pictures"]}

    assert sizes() == {(640, 360)}, "the default"
    assert sizes(max_px=921_600) == {(1280, 720)}
    assert sizes(max_px=0) == at_cap != {(1920, 1080)}, "0 is what the model can see, not as stored"
    assert sizes(max_px=4 * cap) == at_cap, "a value above the cap is cut to it"

    def warned(**params) -> bool:  # 9 windows: about 0.6 s at the default size, about 4.5 s at d1's own cap
        return any("max_s" in w for w in scan(api, question="Is it red?", sources=["cam"], max_s=1, **params)["warnings"])

    assert not warned() and warned(max_px=0) and warned(max_px=4 * cap), "the estimate is at the cap too"


def test_the_lints_estimate_of_a_job_is_at_the_models_cap_whatever_max_px_asks(home, tmp_path, served, store, media):  # noqa: F811
    """9 windows; a picture costs 1 s at the model's cap and 100 s at the size 2,000,000 asks for, so a profile that is not
    cut to the cap on its way to the lint would estimate 900 s."""
    index(store, "cam", "video", media["video"], T0, SECONDS)
    steep = facade(tmp_path, "steep", max_px=500_000, picture_s=((230_400, 0.05), (500_000, 1.0), (2_000_000, 100.0)))
    api = api_under(home, served, steep)

    def estimate(**params) -> list[str]:
        return [w for w in scan(api, question="Is it red?", sources=["cam"], max_s=5, **params)["warnings"] if "max_s" in w]

    assert estimate() == [], "the default size costs 0.05 s a window"
    for asked in (0, 500_000, 2_000_000):
        (said,) = estimate(max_px=asked)
        assert said.startswith("about 9 windows, about 9 s against max_s of 5 s"), (asked, said)


def test_a_model_with_no_picture_cap_reads_max_px_0_as_stored(home, tmp_path, served, store, media):  # noqa: F811
    index(store, "hd", "video", media["hd"], T0, 2)
    api = api_under(home, served, facade(tmp_path, "uncapped", max_px=0))
    scan(api, question="Is it red?", max_px=0)
    assert {tuple(size) for call in decider_log(home) for size in call["pictures"]} == {(1920, 1080)}
