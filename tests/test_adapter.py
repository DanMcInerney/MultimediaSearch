"""`mms mcp` (DESIGN.md §12, TESTBED.md items 4, 12 and 14) as a process, driven over stdio by the MCP SDK's client as Claude
Code and Codex drive it, against a real daemon with the fake decider."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import anyio
import numpy as np
import pytest
from conftest import DECIDER, MODEL_FOLDERS, Api, approve_hint, mms, mms_json, model_folder, wait_for, write_config
from mcp import Client, types
from mcp_session import connected, drive, text
from PIL import Image

from mms import client as api_client
from mms import adapter, install
from mms import consent, decider
from mms import config as configuration
from mms.store import Store

ROOT = Path(__file__).resolve().parents[1]
DESIGN = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
HEAVY = {"torch", "av", "numpy", "PIL", "mms.store", "mms.reader", "mms.ops"}


@pytest.fixture
def work(tmp_path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def markdown_section(start: str, end: str) -> str:
    return DESIGN[DESIGN.index(start): DESIGN.index(end)]


async def ok(client, tool: str, /, **arguments) -> dict:
    """A tool call that must not be an error: its JSON."""
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, (tool, arguments, result.content)
    return text(result)


async def failed(client, tool: str, /, **arguments) -> dict:
    """A tool call that must be an error result: the error."""
    result = await client.call_tool(tool, arguments)
    assert result.is_error, (tool, arguments, result.content)
    return text(result)["error"]


def colour(data: bytes) -> tuple[float, float, float]:
    with Image.open(io.BytesIO(data)) as picture:
        r, g, b = np.asarray(picture.convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    return r, g, b


# the adapter itself

def test_it_starts_in_seconds_and_imports_nothing_heavy(home, work, memory, tmp_path, capsys):
    modules_file = tmp_path / "modules.json"
    timing = {}

    async def main():
        started = time.perf_counter()
        async with connected(home, work, modules=modules_file) as client:
            timing["initialize"] = time.perf_counter() - started
            await client.list_tools()
            await ok(client, "find", text="x")
            await ok(client, "sources")

    anyio.run(main)
    with capsys.disabled():
        print(f"\nadapter: spawn to initialize answered in {timing['initialize']:.2f} s")
    assert timing["initialize"] < 5.0
    loaded = set(json.loads(modules_file.read_text()))
    assert "mms.adapter" in loaded and not HEAVY & loaded, sorted(HEAVY & loaded)


def test_the_instructions_tools_and_schemas_are_the_designs(home, work, memory):
    quoted = [line[2:] if line.startswith("> ") else "" for line in
              markdown_section("### 12.4", "## 13.").splitlines() if line.startswith(">")]
    paragraphs = [" ".join(chunk.strip() for chunk in block.split("\n")) for block in "\n".join(quoted).split("\n\n")]
    rows = [line for line in markdown_section("### 12.3", "**Input schemas**").splitlines() if line.startswith("| `")]
    described = {re.match(r"\| `(\w+)`", row).group(1): row.split("|")[3].strip() for row in rows}

    async def steps(client):
        return client.instructions, (await client.list_tools()).tools

    instructions, tools = drive(steps, home, work)
    assert instructions == "\n\n".join(paragraphs)
    assert len(paragraphs[0]) <= 512 and len(instructions) < 4096
    assert [tool.name for tool in tools] == list(described)
    schemas = {}
    for tool in tools:
        fixed, _, model = tool.description.partition("\n\n")  # ask's ends with the paragraph of the model now
        assert fixed == described[tool.name].split("\n\n")[0] and tool.output_schema is None
        assert bool(model) == (tool.name == "ask"), tool.name
        schemas[tool.name] = tool.input_schema
        assert tool.input_schema["type"] == "object" and tool.input_schema["additionalProperties"] is False, tool.name
        assert "caller" not in tool.input_schema["properties"], tool.name

    ops = markdown_section("**Ops.**", "**`find`:**")
    for name in ("find", "look", "remember", "label"):
        row = next(line for line in ops.splitlines() if line.startswith(f"| `{name}` |"))
        assert set(schemas[name]["properties"]) == set(re.findall(r"`(\w+)`", row.split("|")[2])), name
    assert set(schemas["source_set"]["properties"]) == {
        "name", "kind", "device", "fps", "path", "glob", "command", "time_field", "max_gb", "when_full", "max_days",
        "remove"}
    assert schemas["source_set"]["required"] == ["name"] and schemas["remember"].get("required") is None
    assert schemas["sources"]["properties"] == {} and schemas["label"]["required"] == ["text", "since"]

    table = markdown_section("**Parameters:**", "**`hit`** per")
    named = {name for row in table.splitlines() if row.startswith("| `") for name in re.findall(r"`(\w+)`", row.split("|")[1])}
    ask = schemas["ask"]["properties"]
    assert set(ask) == named | {"job", "cancel", "all", "wait_s"}
    assert ask["answers"]["minItems"] == 2 and ask["context"]["maxLength"] == 4000
    assert (ask["limit"]["minimum"], ask["limit"]["maximum"], ask["limit"]["default"]) == (1, 50, 10)
    assert (ask["candidates"]["minimum"], ask["candidates"]["maximum"], ask["candidates"]["default"]) == (0, 20, 5)
    assert (ask["max_s"]["minimum"], ask["max_s"]["maximum"], ask["max_s"]["default"]) == (1, 3600, 600)
    assert (ask["every_s"]["default"], ask["merge_s"]["default"], ask["max_sighting_s"]["default"]) == (1, 10, 600)
    assert ask["scope"]["enum"] == ["default", "all"] and ask["wait_s"]["maximum"] == 50
    assert {"number", "string", "array"} == {option["type"] for option in ask["hit"]["anyOf"]}
    assert "the model's own if not given" in ask["hit"]["description"] and "0.8" not in ask["hit"]["description"]
    assert schemas["find"]["properties"]["limit"]["maximum"] == 200


def test_ask_takes_best_first_and_look_takes_a_region_of_four_fractions_also_in_each_moment(home, work, memory):
    schemas = drive(lambda client: client.list_tools(), home, work)
    tools = {tool.name: tool for tool in schemas.tools}
    assert tools["ask"].input_schema["properties"]["best_first"] == {
        "type": "boolean", "default": False, "description": "read the whole range and give the limit sightings with the "
        "highest p, best first, instead of the first limit found in scan order"}
    look = tools["look"].input_schema["properties"]
    moment = look["moments"]["items"]["properties"]
    for region in (look["region"], moment["region"]):
        assert region["type"] == "object" and region["additionalProperties"] is False
        assert set(region["properties"]) == set(region["required"]) == {"top", "left", "bottom", "right"}
        assert all(edge == {"type": "number", "minimum": 0, "maximum": 1} for edge in region["properties"].values())
        assert "part of the picture" in region["description"]
    assert "region" in tools["look"].description and "enlarged" in tools["look"].description


def test_tools_offers_only_the_named_tools_and_a_call_of_another_is_an_unknown_tool(home, work, memory):
    """`mms mcp --tools` (DESIGN.md §12.1): tools/list shows only the named tools, in the usual order, the instructions
    stay whole, and a call of a tool left out is the `not_found` an unknown tool gets, which the adapter never runs."""
    from mms import adapter

    (work / "secret.txt").write_text("not for the agent")  # `remember` would store it
    command = [sys.executable, "-m", "mms.cli", "mcp", "--tools", "look,find"]

    async def steps(client):
        called = await failed(client, "remember", path="secret.txt")
        return (client.instructions, [tool.name for tool in (await client.list_tools()).tools], called,
                await ok(client, "find", text="x"))

    instructions, names, called, found = drive(steps, home, work, command=command)
    assert names == ["find", "look"] and instructions == adapter.INSTRUCTIONS
    assert called["code"] == "not_found" and called["field"] == "tool" and "remember" in called["message"]
    assert found["results"] == []


def test_tools_names_the_valid_tools_when_one_is_unknown_or_empty(home):
    """`mms mcp --tools` is checked at the command line (exit 2, the valid tools named), not by a session that offers
    nothing."""
    for value in ("nope", "find,nope", "find,", ""):
        done = mms(home, "mcp", "--tools", value, check=False, timeout=30)
        assert done.returncode == 2 and "Traceback" not in done.stderr, (value, done.stderr)
        assert "ask, find, look, remember, label, sources, source_set" in done.stderr, (value, done.stderr)
        assert not done.stdout


def test_ask_cuts_its_wait_to_the_calls_budget(home, work, memory, monkeypatch):
    from mms import adapter, client

    monkeypatch.chdir(work)
    sent = []
    real = client.call

    def spy(home_, op, params=None, **kwargs):  # records what the adapter sends, and sends it
        sent.append((op, dict(params or {})))
        return real(home_, op, params, **kwargs)

    monkeypatch.setattr(client, "call", spy)
    session = adapter.Adapter(home)
    for wait in (50, 0, None):
        arguments = {"question": "Is it red?", "since": "1h"} | ({} if wait is None else {"wait_s": wait})
        content, is_error = session.call("ask", arguments, "test/1")
        assert json.loads(content[0].text)["state"] in ("running", "done") and len(content) == 1 and not is_error
    long, none, default = (params["wait_s"] for op, params in sent if op == "jobs/start")
    assert 40 < long <= 45, "the call's 50 s, less the margin and the time spent"
    assert none == 0 and 35 < default <= 40


# every tool, all the way down

def test_remember_find_ask_and_look_all_the_way_down(home, work, memory, red_clip):
    (work / ".mms-project").write_text("proj-demo\n")
    (work / "pics").mkdir()
    Image.new("RGB", (64, 36), (220, 20, 20)).save(work / "pics" / "red.png")

    async def steps(client):
        out = {}
        out["note"] = await ok(client, "remember", note="the kettle in the lab is on", t="90m", expires="30d")
        out["picture"] = await ok(client, "remember", path=os.path.join("pics", "red.png"))  # relative to the cwd
        out["recording"] = await ok(client, "remember", path=str(red_clip), source="clips", t="2h")
        out["kettle"] = await ok(client, "find", text="kettle")
        out["clips"] = await ok(client, "find", sources=["clips"])
        out["picture_found"] = await ok(client, "find", text="red.png", types=["item"])
        out["asked"] = await ok(client, "ask", question="Is it red?", since="3h", until="1h", sources=["clips"])
        look = await client.call_tool("look", out["asked"]["sightings"][0]["look"])
        out["look"] = look
        out["picture_look"] = await client.call_tool("look", out["picture_found"]["results"][0]["look"])
        start = configuration.parse_when(out["clips"]["results"][0]["t0"])
        out["label"] = await ok(client, "label", text="the red clip", since=start - 1, until=start + 3, source="clips")
        out["labels"] = await ok(client, "find", text="red clip", types=["label"])
        out["shown"] = await client.call_tool("look", {"source": "clips", "t": start + 1})
        out["sources"] = await ok(client, "sources")
        return out

    out = drive(steps, home, work)
    assert [row["type"] for row in out["note"]["stored"]] == ["record"] and out["note"]["expires"]
    assert out["picture"]["stored"][0]["name"] == "red.png" and out["picture"]["stored"][0]["type"] == "item"
    assert out["recording"]["stored"][0]["time_from"] == "given" and out["recording"]["stored"][0]["kind"] == "video"

    (kettle,) = out["kettle"]["results"]
    assert kettle["type"] == "record" and kettle["project"] == "work" and "kettle" in kettle["text"]
    assert out["kettle"]["searched"]["project"] == {"id": "id:proj-demo", "name": "work"}
    (clip,) = out["clips"]["results"]
    assert clip["kind"] == "video" and clip["look"] == {"item": clip["id"], "at": 0}
    assert out["picture_found"]["results"][0]["name"] == "red.png"

    asked = out["asked"]
    assert asked["state"] == "done" and "error" not in asked and "next" not in asked
    assert list(asked)[:2] == ["job", "state"] and asked["stopped"] == "range_end"
    (sighting,) = asked["sightings"]
    assert sighting["source"] == "clips" and sighting["look"]["item"] == clip["id"] and sighting["p"] == 0.95
    assert sighting["project"] == "work", "a job's labels carry the caller's project"

    for result, source in ((out["look"], "clips"), (out["picture_look"], "agent")):
        picture, words = result.content
        assert isinstance(picture, types.ImageContent) and picture.mime_type == "image/jpeg", "the picture comes first"
        data = base64.b64decode(picture.data)
        assert data[:2] == b"\xff\xd8", "a JPEG"
        r, g, b = colour(data)
        assert r > 150 and g < 60 and b < 60, "the frame the sighting is about, red"
        shown = json.loads(words.text)
        assert shown["kind"] == "image" and shown["source"] == source
        assert Path(shown["image"]["inline"]).read_bytes() == data, "the inline file is what the adapter sent"
        assert Path(shown["image"]["path"]).suffix == ".png" and Path(shown["image"]["path"]).is_file()
    labels = json.loads(out["shown"].content[1].text)["labels"]
    assert {(label["by"], label["text"]) for label in labels} >= {("agent", "the red clip")}, "ask's own label is there too"
    assert out["label"]["label"] and out["labels"]["results"][0]["by"] == "agent"
    assert out["labels"]["results"][0]["project"] == "work"
    sources = {one["name"]: one for one in out["sources"]["sources"]}
    assert {"agent", "clips"} <= set(sources) and sources["clips"]["items"] == 1
    assert out["sources"]["devices"]["microphones"] == [{"device": "Fake Mic"}] or os.name != "nt"
    for result in (out["shown"], out["look"]):  # compact JSON, whatever the block count
        raw = result.content[-1].text
        assert raw == json.dumps(json.loads(raw), separators=(",", ":"), ensure_ascii=False)


def test_look_with_moments_gives_each_its_picture_then_its_json_and_is_an_error_only_if_every_one_failed(home, work, memory):
    (work / "pics").mkdir()
    Image.new("RGB", (64, 36), (220, 20, 20)).save(work / "pics" / "red.png")
    Image.new("RGB", (64, 36), (20, 20, 220)).save(work / "pics" / "blue.png")

    async def steps(client):
        found = {}
        for name in ("red", "blue"):
            await ok(client, "remember", path=os.path.join("pics", f"{name}.png"))
            found[name] = (await ok(client, "find", text=f"{name}.png", types=["item"]))["results"][0]["look"]
        pictures = await client.call_tool("look", {"moments": [found["red"], found["blue"]]})
        mixed = await client.call_tool("look", {"moments": [{"item": 999}, found["blue"], {"source": "nowhere", "t": "now"}]})
        every = await client.call_tool("look", {"moments": [{"item": 999}, {"source": "nowhere", "t": "now"}]})
        single = await client.call_tool("look", found["red"])
        return found, pictures, mixed, every, single

    found, pictures, mixed, every, single = drive(steps, home, work)
    assert [type(block) for block in pictures.content] == [types.ImageContent, types.TextContent] * 2 and not pictures.is_error
    for picture, words, red in zip(pictures.content[::2], pictures.content[1::2], (True, False)):
        shown = json.loads(words.text)  # the picture is the moment's own frame, its text that moment's JSON
        data = base64.b64decode(picture.data)
        r, g, b = colour(data)
        assert shown["kind"] == "image" and Path(shown["image"]["inline"]).read_bytes() == data
        assert (r > 150 and b < 60) if red else (b > 150 and r < 60)
    assert [type(block) for block in single.content] == [types.ImageContent, types.TextContent]
    assert json.loads(single.content[1].text)["item"] == json.loads(pictures.content[1].text)["item"]
    assert [type(block) for block in mixed.content] == [types.TextContent, types.ImageContent, types.TextContent,
                                                        types.TextContent] and not mixed.is_error
    first, _, second, third = (json.loads(block.text) if isinstance(block, types.TextContent) else None for block in mixed.content)
    assert first["error"]["code"] == "not_found" and first["error"]["field"] == "item"
    assert second["kind"] == "image" and third["error"]["code"] == "not_found" and third["error"]["field"] == "source"
    assert every.is_error and [json.loads(block.text)["error"]["code"] for block in every.content] == ["not_found"] * 2


def test_look_over_a_range_gives_the_sheet_as_its_picture_then_its_json(home, work, memory):
    (work / "pics").mkdir()
    Image.new("RGB", (64, 36), (220, 20, 20)).save(work / "pics" / "red.png")
    Image.new("RGB", (64, 36), (20, 20, 220)).save(work / "pics" / "blue.png")
    asked = {"source": "agent", "since": "1h", "until": "now"}

    async def steps(client):
        await ok(client, "remember", path=os.path.join("pics", "red.png"), t="50m")
        await ok(client, "remember", path=os.path.join("pics", "blue.png"), t="20m")
        sheet = await client.call_tool("look", asked)
        several = await client.call_tool("look", {"moments": [asked, {"source": "nowhere", "since": "1h", "until": "now"}]})
        return sheet, several

    sheet, several = drive(steps, home, work)
    assert [type(block) for block in sheet.content] == [types.ImageContent, types.TextContent] and not sheet.is_error
    picture, words = sheet.content
    shown, data = json.loads(words.text), base64.b64decode(picture.data)
    assert shown["kind"] == "sheet" and [tile["n"] for tile in shown["tiles"]] == [1, 2] and data[:2] == b"\xff\xd8"
    assert Path(shown["image"]["inline"]).read_bytes() == data, "the inline file is what the adapter sent"
    with Image.open(io.BytesIO(data)) as inline:
        width, height = inline.size
        left, right = (np.asarray(inline.convert("RGB").crop(box), dtype=float).reshape(-1, 3).mean(axis=0)
                       for box in ((0, 0, width // 2, height), (width // 2, 0, width, height)))
    assert left[0] > 150 and left[2] < 60 and right[2] > 150 and right[0] < 60, "the red picture, then the blue one"
    assert [type(block) for block in several.content] == [types.ImageContent, types.TextContent, types.TextContent]
    assert not several.is_error and json.loads(several.content[1].text)["kind"] == "sheet"
    assert json.loads(several.content[2].text)["error"]["code"] == "not_found"


def test_a_moment_whose_picture_cannot_be_read_is_its_own_error_and_a_single_look_still_raises(home, work, memory, monkeypatch):
    from mms import adapter

    Image.new("RGB", (64, 36), (220, 20, 20)).save(work / "red.png")
    api_client.call(home, "remember", {"path": str(work / "red.png")})
    handle = api_client.call(home, "find", {"text": "red.png", "types": ["item"]})["results"][0]["look"]
    monkeypatch.chdir(work)
    real = api_client.call

    def lose_pictures(home_, op, params=None, **kwargs):  # the daemon's inline files are gone by the time they are read
        result = real(home_, op, params, **kwargs)
        for file in (home_ / "looks").glob("*.jpg"):
            file.unlink()
        return result

    monkeypatch.setattr(api_client, "call", lose_pictures)
    session = adapter.Adapter(home)
    content, is_error = session.call("look", {"moments": [handle, handle]}, "test/1")
    assert is_error and [block.type for block in content] == ["text", "text"]
    for block in content:
        error = json.loads(block.text)["error"]
        assert error["code"] == "internal" and "inline picture that cannot be read" in error["message"]
    with pytest.raises(adapter.ApiError) as raised:
        session.call("look", handle, "test/1")
    assert raised.value.code == "internal"


def test_a_record_found_by_its_handle_even_with_digits_below_the_millisecond(home, work, memory):
    stamp = time.time() - 600 + 0.0004567  # the handle's ISO time is rounded to the millisecond

    async def steps(client):
        await ok(client, "remember", note="the exact kettle", t=stamp)
        await ok(client, "remember", note="a later note", t=stamp + 300)
        row = (await ok(client, "find", text="exact"))["results"][0]
        return row, await ok(client, "look", **row["look"]), await ok(client, "look", source="agent", t=stamp)

    row, shown, direct = drive(steps, home, work)
    assert row["look"] == {"source": "agent", "t": row["t0"]} and re.search(r"\.\d{3}[+-]\d\d:\d\d$", row["t0"])
    for seen in (shown, direct):
        assert seen["kind"] == "text" and "the exact kettle" in seen["text"] and "a later note" not in seen["text"]


def test_a_long_ask_returns_a_job_before_the_budget_and_a_later_call_collects_the_rest(home, work, memory, red_clip):
    async def steps(client):
        await ok(client, "remember", path=str(red_clip), source="clips", t="2h")
        range_ = {"since": "3h", "until": "1h", "sources": ["clips"], "still_s": 0}  # a window a second: the scan's pace is timed
        started = time.monotonic()
        first = await ok(client, "ask", question="Is it red? slow:0.5", wait_s=0.5, **range_)
        spent = time.monotonic() - started
        later = await ok(client, "ask", job=first["job"], wait_s=30)
        slow = await ok(client, "ask", question="Is it red? slow:1.5", wait_s=0, **range_)
        cancelled = await ok(client, "ask", job=slow["job"], cancel=True)
        return first, spent, later, slow, cancelled

    first, spent, later, slow, cancelled = drive(steps, home, work)
    assert first["state"] == "running" and spent < 20
    assert first["job"] in first["next"] and "cancel" in first["next"] and first.get("progress", {}).get("windows", 0) < 6
    assert later["job"] == first["job"] and later["state"] == "done" and "next" not in later
    assert later["progress"]["windows"] >= 6 and len(later["sightings"]) == 1
    assert "searched" not in later, "the first reply gave it"
    assert slow["state"] == "running" and cancelled["state"] == "cancelled" and "next" not in cancelled


def test_a_check_in_gives_what_changed_and_all_gives_everything(home, work, memory, red_clip):
    async def steps(client):
        await ok(client, "remember", path=str(red_clip), source="clips", t="2h")
        range_ = {"since": "3h", "until": "1h", "sources": ["clips"], "still_s": 0}  # a window a second: the scan's pace is timed
        first = await ok(client, "ask", question="Is it red? slow:0.5", wait_s=0.5, **range_)
        later = await ok(client, "ask", job=first["job"], wait_s=30)
        again = await ok(client, "ask", job=first["job"], wait_s=0)
        everything = await ok(client, "ask", job=first["job"], all=True, wait_s=0)
        return first, later, again, everything

    first, later, again, everything = drive(steps, home, work)
    assert first["state"] == "running" and {"sightings", "candidates", "warnings", "searched", "progress"} <= set(first)
    assert "error" not in first and "stopped" not in first, "a field that is None is left out"
    assert later["state"] == "done" and not {"searched", "warnings"} & set(later), "what the first reply gave is not sent again"
    assert len(later["sightings"]) == 1 and later["stopped"] == "range_end"
    assert again["state"] == "done" and {"sightings", "candidates", "searched"} <= set(again), "a final reply forgets the job: all of it"
    assert everything == again


def test_errors_are_results_with_is_error_carrying_the_apis_code(home, work, served):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 100000000\n" + DECIDER)  # no disk is that big: the floor
    served()

    async def steps(client):
        out = {"unknown tool": await failed(client, "approve", source="mic")}
        out["field"] = await failed(client, "find", bogus=1)
        out["item"] = await failed(client, "look", item=999)
        out["floor"] = await failed(client, "remember", note="x")
        out["note"] = await failed(client, "remember", name="only a name")  # the schema asks for more: the daemon does
        out["kind"] = await failed(client, "source_set", name="only-a-name")
        out["cap"] = await failed(client, "source_set", name="no-cap", kind="folder", path=str(work))
        job = await client.call_tool("ask", {"question": "Is it red?", "limit": 99})  # the script's own check
        out["job"] = (job.is_error, text(job))
        started = await ok(client, "ask", question="Is it red?", since="1h")
        out["with job"] = await failed(client, "ask", job=started["job"], question="again")
        out["with since"] = await failed(client, "ask", job=started["job"], since="1h", cancel=True)
        out["cancel alone"] = await failed(client, "ask", cancel=True)
        out["all alone"] = await failed(client, "ask", all=True)
        out["nothing"] = await failed(client, "ask", since="1h")
        out["no job"] = await failed(client, "ask", job="j-00000000")
        return out

    out = drive(steps, home, work)
    for name, code, field in (("unknown tool", "not_found", "tool"), ("field", "bad_request", "bogus"),
                              ("item", "not_found", "item"), ("floor", "floor", None), ("note", "bad_request", "note"),
                              ("kind", "bad_request", "kind"), ("cap", "bad_request", "max_gb"),
                              ("with job", "bad_request", "question"), ("with since", "bad_request", "since"),
                              ("cancel alone", "bad_request", "cancel"), ("all alone", "bad_request", "all"),
                              ("nothing", "bad_request", "question"),
                              ("no job", "not_found", "job")):
        assert out[name]["code"] == code and out[name].get("field") == field, (name, out[name])
    is_error, failed_job = out["job"]
    assert is_error and failed_job["state"] == "failed" and failed_job["error"]["code"] == "bad_request"
    assert "limit" in failed_job["error"]["message"] and "next" not in failed_job
    assert not (home / "consent.json").exists()


def test_two_sessions_from_two_projects_see_each_others_notes_only_with_scope_all(home, tmp_path, memory):
    one, two, elsewhere = tmp_path / "one", tmp_path / "two", tmp_path / "elsewhere"
    for folder, name in ((one, "first"), (two, "second")):
        folder.mkdir()
        (folder / ".mms-project").write_text(name + "-project\n")
    elsewhere.mkdir()

    async def first(client):
        await ok(client, "remember", note="the first kettle")
        await ok(client, "remember", note="the first kettle again")
        return await ok(client, "find", text="kettle")

    async def second(client):  # its project is CLAUDE_PROJECT_DIR's, not the working directory's
        mine = await ok(client, "find", text="kettle")
        everyone = await ok(client, "find", text="kettle", scope="all")
        await ok(client, "remember", note="the second kettle")
        return mine, everyone, await ok(client, "find", text="kettle")

    seen_by_first = drive(first, home, one)
    mine, everyone, after = drive(second, home, elsewhere, env={"CLAUDE_PROJECT_DIR": str(two)})
    assert len(seen_by_first["results"]) == 2
    assert mine["results"] == [] and mine["searched"]["project"] == {"id": "id:second-project", "name": "two"}
    assert {json.loads(row["text"])["text"] for row in everyone["results"]} == {"the first kettle", "the first kettle again"}
    assert {row["project"] for row in everyone["results"]} == {"one"} and everyone["searched"]["scope"] == "all"
    assert [json.loads(row["text"])["text"] for row in after["results"]] == ["the second kettle"]

    store = Store(home)
    try:
        sessions = store.db().execute("SELECT id, project, client FROM sessions").fetchall()
        notes = store.db().execute("SELECT data, session, project FROM records WHERE source = 'agent' ORDER BY id").fetchall()
    finally:
        store.close()
    assert len(sessions) == 2 and {row["client"] for row in sessions} == {"testclient/9.9"}
    by_session = {row["id"]: row["project"] for row in sessions}
    assert set(by_session.values()) == {"id:first-project", "id:second-project"}
    assert notes[0]["session"] == notes[1]["session"] != notes[2]["session"], "one session ID for the process"
    assert [by_session[note["session"]] for note in notes] == ["id:first-project"] * 2 + ["id:second-project"]


def test_a_folder_outside_any_project_reads_ambient_rows_and_a_bad_project_fails_every_call_but_the_listing(
        home, work, tmp_path, memory):
    async def probe(client):
        return (await ok(client, "find"))["searched"]["project"]

    assert drive(probe, home, work) is None

    (work / ".mms-project").write_text("not valid!\n")

    async def steps(client):
        return (await client.list_tools()).tools, await failed(client, "find"), await failed(client, "sources")

    tools, found, sources = drive(steps, home, work)
    assert len(tools) == 7
    for error in (found, sources):
        assert error["code"] == "bad_project" and "mms-project" in error["message"] and error["hint"]


def test_a_session_whose_daemon_died_starts_another_and_asks_again_once(home, work, daemon, detached):
    from mms import client as api_client

    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n")
    first = daemon()

    def refused() -> bool:  # a call in the first moments after the kill can still be reset, which is not "no daemon"
        try:
            return api_client.connect(home) is None
        except api_client.DaemonFailed:
            return False

    async def steps(client):
        await ok(client, "remember", note="before the crash")
        first.kill()  # daemon.json is left behind, stale, as after a crash
        wait_for(refused, what="the dead daemon's port to refuse")
        return await ok(client, "find", text="crash"), await ok(client, "sources")

    found, sources = drive(steps, home, work)
    assert [json.loads(row["text"])["text"] for row in found["results"]] == ["before the crash"]
    assert sources["running"] and sources["pid"] != first.process.pid, "a new daemon, started by the adapter"


def test_with_no_daemon_the_first_call_starts_one_that_outlives_the_session(home, work, detached, capsys):
    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + DECIDER)
    assert not (home / "daemon.json").exists()
    timing = {}

    async def steps(client):
        started = time.perf_counter()
        remembered = await ok(client, "remember", note="the first call starts the daemon")
        timing["first call"] = time.perf_counter() - started
        return remembered, await ok(client, "find", text="daemon"), await ok(client, "sources")

    remembered, found, sources = drive(steps, home, work)
    with capsys.disabled():
        print(f"\nadapter: the first call, with no daemon running, took {timing['first call']:.2f} s")
    assert timing["first call"] < 50 and remembered["stored"] and len(found["results"]) == 1
    assert sources["running"] is True and sources["pid"] != os.getpid(), "a daemon of its own, started by the adapter"
    assert "mms daemon started" in (home / "daemon.log").read_text(encoding="utf-8"), "a detached daemon's log"
    # the session is over, and with it the job object the SDK's client put the adapter in: the daemon must outlive it
    assert api_client.connect(home).pid == sources["pid"], "the daemon the adapter started still answers"
    assert mms_json(home, "status")["running"] is True


def test_a_call_after_a_slow_restart_still_returns_a_job_id_within_its_budget(home, work, daemon, detached, red_clip, monkeypatch):
    """`ask` once sent the wait it had worked out before `ensure`, so a retry after a slow start waited longer than the
    call had left, the client gave up, and the agent got no job id while the job ran on."""
    from mms import adapter

    write_config(home, "[memory]\nloop_s = 0.1\nfree_gb = 0\n" + DECIDER)
    first = daemon()
    stored = api_client.call(home, "remember", {"path": str(red_clip), "source": "clips", "t": "2h"})["stored"]
    assert stored
    first.kill()
    wait_for(lambda: api_client.connect(home) is None, what="the dead daemon's port to refuse")

    monkeypatch.chdir(work)
    monkeypatch.setattr(adapter, "CALL_BUDGET_S", 18.0)  # the 50 s of the real thing, in scale
    real_ensure, sent = api_client.ensure, []

    def slow_ensure(home_, **kwargs):  # a daemon that is slow to start: 6 s before it is even asked for
        time.sleep(6)
        return real_ensure(home_, **kwargs)

    real_call = api_client.call
    monkeypatch.setattr(api_client, "ensure", slow_ensure)
    monkeypatch.setattr(api_client, "call", lambda home_, op, params=None, **kw: (sent.append((op, dict(params or {}), kw)),
                                                                                 real_call(home_, op, params, **kw))[1])
    session = adapter.Adapter(home)
    question = {"question": "Is it red? slow:4", "since": "3h", "until": "1h", "sources": ["clips"], "wait_s": 50, "still_s": 0}
    started = time.monotonic()
    content, is_error = session.call("ask", question, "test/1")
    output = json.loads(content[-1].text)
    took = time.monotonic() - started
    start = [(params, kw) for op, params, kw in sent if op == "jobs/start"]
    assert not is_error and output["state"] == "running" and output["job"].startswith("j-"), output
    assert took < 18, f"the call returned in {took:.1f} s"
    params, kw = start[-1]  # the retry
    assert params["wait_s"] <= 18 - 6 - adapter.MARGIN_S and kw["timeout_s"] <= 12, "worked out from the time left, in the retry"
    api_client.call(home, "jobs/cancel", {"job": output["job"]})


# what a check-in sends (DESIGN.md §12.1): the first reply is the whole result, a later one `job`, `state`, `progress` and each
# field whose value differs from the last sent, whole

def sighting(source: str = "cam", t0: str = "10:00:03", t1: str = "10:00:05", p: float = 0.9, **more) -> dict:
    span = {"t": t0} if t0 == t1 else {"t0": t0, "t1": t1}
    return {"source": source, **span, "p": p, "kind": "video", "look": {"item": 1, "at": 3.0}, **more}


def candidate(t: str, p: float) -> dict:
    return {"source": "cam", "t": t, "p": p, "kind": "video", "look": {"item": 1, "at": 0.0}}


def running(*, state: str = "running", error=None, windows: int = 0, **result) -> dict:
    """The daemon's `jobs/get` of an ask: `result` as the script reported it so far."""
    base = {"sightings": [], "candidates": [], "searched": {"scope": "default", "sources": ["cam"]}, "stopped": None,
            "warnings": [], "progress": {"windows": windows, "of_about": 9, "seconds": 0.0, "decider": "loaded"}}
    return {"job": "j-1", "script": "ask", "state": state, "started": "x", "ended": None, "error": error,
            "result": {**base, **result}}


def asks(session: adapter.Adapter, op: str, job: dict, everything: bool = False) -> dict:
    return session._reply(op, job, everything)


def lean() -> set:
    """A check-in with nothing to say: the job, its state (running) and progress, and what to do next."""
    return {"job", "state", "progress", "next"}


def test_the_first_reply_is_the_whole_result_without_the_fields_that_are_none(home):
    session = adapter.Adapter(home)
    out = asks(session, "jobs/start", running(sightings=[sighting()], candidates=[candidate("10:00:00", 0.4)],
                                             warnings=["no since: the last 24 h were searched"]))
    assert list(out) == ["job", "state", "sightings", "candidates", "searched", "warnings", "progress", "next"]
    assert "error" not in out and "stopped" not in out
    assert out["next"] == 'Call ask with job "j-1" to collect more, or with cancel: true to stop it.'
    empty = asks(adapter.Adapter(home), "jobs/start", running(state="queued"))
    assert empty["sightings"] == [] and empty["candidates"] == [] and empty["warnings"] == [], "as the script reported them"
    assert asks(adapter.Adapter(home), "jobs/start", {"job": "j-2", "state": "queued", "error": None, "result": {}}) == {
        "job": "j-2", "state": "queued", "next": 'Call ask with job "j-2" to collect more, or with cancel: true to stop it.'}


def test_a_check_in_with_nothing_new_carries_only_the_job_its_state_and_progress(home):
    session = adapter.Adapter(home)
    held = running(sightings=[sighting()], candidates=[candidate("10:00:00", 0.4)], warnings=["a warning"])
    asks(session, "jobs/start", held)
    again = running(sightings=[sighting()], candidates=[candidate("10:00:00", 0.4)], warnings=["a warning"], windows=4)
    out = asks(session, "jobs/get", again)
    assert out == {"job": "j-1", "state": "running", "progress": again["result"]["progress"],
                   "next": 'Call ask with job "j-1" to collect more, or with cancel: true to stop it.'}


def test_a_list_that_changed_comes_back_whole_a_sighting_that_grew_by_moving_its_start_included(home):
    """Newest first, a sighting grows by pulling its start earlier, and one may leave the list at its `limit`: no key names a
    sighting across replies, so each check-in that differs gives the list as it is, and the agent holds the last."""
    session = adapter.Adapter(home)
    older = sighting("mic", "09:00:00", "09:00:04")
    grown = [sighting("cam", "10:00:03", "10:00:05", 0.7)]
    asks(session, "jobs/start", running(sightings=[*grown, older]))
    held = [*grown, older]
    for start in range(55, 40, -1):  # the start moves earlier at each check-in, as windows join from the newest end
        grown = [sighting("cam", f"09:59:{start}", "10:00:05", 0.7 + (55 - start) / 100)]
        out = asks(session, "jobs/get", running(sightings=[*grown, older]))
        assert out["sightings"] == [*grown, older], "the whole list, not the one that moved"
        held = out["sightings"]
    assert held == [*grown, older]
    assert "sightings" not in asks(session, "jobs/get", running(sightings=[*grown, older])), "unchanged: left out"
    instant = sighting("cam", "10:05:00", "10:05:00")  # one instant has a `t`, and a new one in the list changes it
    assert asks(session, "jobs/get", running(sightings=[instant, *grown, older]))["sightings"] == [instant, *grown, older]
    gone = asks(session, "jobs/get", running(sightings=[instant, *grown]))  # `older` left the list: the agent is told
    assert gone["sightings"] == [instant, *grown]
    assert asks(session, "jobs/get", running(sightings=[]))["sightings"] == [], "an empty list is a change"


def test_candidates_are_sent_whole_when_the_list_differs_and_left_out_when_it_does_not(home):
    session = adapter.Adapter(home)
    best = [candidate("10:00:00", 0.6), candidate("10:01:00", 0.5), candidate("10:02:00", 0.4)]
    assert asks(session, "jobs/start", running(candidates=best))["candidates"] == best
    assert "candidates" not in asks(session, "jobs/get", running(candidates=list(best)))
    better = [candidate("10:03:00", 0.7), *best[:2]]  # one came in and one went out: a ranking, so all of it
    assert asks(session, "jobs/get", running(candidates=better))["candidates"] == better
    assert "candidates" not in asks(session, "jobs/get", running(candidates=better))
    assert asks(session, "jobs/get", running(candidates=[]))["candidates"] == [], "an empty list is a change too"
    assert "candidates" not in asks(session, "jobs/get", running(candidates=[]))


def test_warnings_come_back_whole_when_they_change_a_rewritten_line_included(home):
    session = adapter.Adapter(home)
    asks(session, "jobs/start", running(warnings=["one", "two"]))
    assert "warnings" not in asks(session, "jobs/get", running(warnings=["one", "two"]))
    assert asks(session, "jobs/get", running(warnings=["one", "two", "three"]))["warnings"] == ["one", "two", "three"]
    # a rolling line ("and N more items unreadable", the items not read) is rewritten in its place as more arrive
    rolled = ["one", "two", "and 7 more items unreadable"]
    assert asks(session, "jobs/get", running(warnings=["one", "two", "and 6 more items unreadable"]))["warnings"][-1].startswith("and 6")
    assert asks(session, "jobs/get", running(warnings=rolled))["warnings"] == rolled, "the whole list, with the line as it is now"
    assert "warnings" not in asks(session, "jobs/get", running(warnings=rolled))


def test_other_result_fields_are_sent_when_their_value_changed_and_searched_once(home):
    session = adapter.Adapter(home)
    queued = running(state="queued")
    del queued["result"]["searched"]  # the script has not reported it yet
    assert "searched" not in asks(session, "jobs/start", queued)
    got = asks(session, "jobs/get", running(estimate=3))
    assert got["searched"] == {"scope": "default", "sources": ["cam"]} and got["estimate"] == 3, "the first reply that has them"
    assert not {"searched", "estimate"} & set(asks(session, "jobs/get", running(estimate=3)))
    assert asks(session, "jobs/get", running(estimate=4))["estimate"] == 4


def test_stopped_and_error_are_sent_when_set_and_the_memory_goes_with_a_final_state(home):
    session = adapter.Adapter(home)
    asks(session, "jobs/start", running(sightings=[sighting()]))
    assert "j-1" in session._sent
    done = asks(session, "jobs/get", running(state="done", stopped="range_end", sightings=[sighting()]))
    assert done["state"] == "done" and done["stopped"] == "range_end" and "next" not in done and "sightings" not in done
    assert session._sent == {}, "a final state forgets the job"
    full = asks(session, "jobs/get", running(state="done", stopped="range_end", sightings=[sighting()]))
    assert full["sightings"] == [sighting()] and "searched" in full, "a job it has no memory of gets the whole result"
    for op, state, stopped, error in (("jobs/cancel", "cancelled", "cancelled", None),
                                      ("jobs/get", "failed", None, {"code": "bad_request", "message": "no"})):
        session = adapter.Adapter(home)
        asks(session, "jobs/start", running())
        out = asks(session, op, running(state=state, stopped=stopped, error=error))
        assert out == {"job": "j-1", "state": state, "progress": running()["result"]["progress"],
                       **({"stopped": stopped} if stopped else {}), **({"error": error} if error else {})}
        assert session._sent == {}
    assert asks(session, "jobs/start", running(state="done", stopped="range_end"))["state"] == "done"
    assert session._sent == {}, "a job that is done at its first reply is not remembered"


def test_all_gives_the_whole_current_result_and_a_job_the_session_has_not_seen_gets_it_too(home):
    session = adapter.Adapter(home)
    held = running(sightings=[sighting()], candidates=[candidate("10:00:00", 0.4)], warnings=["w"])
    asks(session, "jobs/start", held)
    assert set(asks(session, "jobs/get", held)) == lean()
    everything = asks(session, "jobs/get", held, everything=True)
    assert everything == asks(adapter.Adapter(home), "jobs/get", held), "all is what a job without memory gets"
    assert {"sightings", "candidates", "warnings", "searched"} <= set(everything)
    assert set(asks(session, "jobs/get", held)) == lean(), "and the check-ins go on from there"


def test_a_jobs_memory_is_its_own(home):
    session = adapter.Adapter(home)
    one, two = running(sightings=[sighting()]), running(sightings=[sighting("mic")])
    two["job"] = "j-2"
    asks(session, "jobs/start", one)
    asks(session, "jobs/start", two)
    assert set(asks(session, "jobs/get", one)) == lean()
    again = two | {"result": two["result"] | {"sightings": [sighting("mic"), sighting("cam")]}}
    assert asks(session, "jobs/get", again)["sightings"] == [sighting("mic"), sighting("cam")]
    assert set(asks(session, "jobs/get", one)) == lean(), "j-1 was not touched by j-2's"


def test_a_check_in_that_finishes_after_a_newer_one_gives_only_its_progress_and_records_nothing(home):
    session = adapter.Adapter(home)
    asks(session, "jobs/start", running(windows=0))
    newer = running(sightings=[sighting(), sighting("mic")], windows=8)
    assert asks(session, "jobs/get", newer)["sightings"] == [sighting(), sighting("mic")]
    older = running(sightings=[sighting()], candidates=[candidate("10:00:00", 0.4)], windows=3)
    late = asks(session, "jobs/get", older)
    assert late == {"job": "j-1", "state": "running", "progress": older["result"]["progress"], "next": late["next"]}
    assert set(asks(session, "jobs/get", newer)) == lean(), "the memory is the newer's: nothing to add, nothing taken back"
    equal = running(sightings=[sighting(), sighting("mic")], windows=8, warnings=["a new one"])
    assert asks(session, "jobs/get", equal)["warnings"] == ["a new one"], "as many windows is not older"


def test_a_running_snapshot_arriving_after_the_final_state_was_sent_is_not_remembered(home, monkeypatch):
    session = adapter.Adapter(home)
    asks(session, "jobs/start", running(sightings=[sighting()], windows=5))
    done = asks(session, "jobs/get", running(state="done", stopped="range_end", sightings=[sighting()], windows=9))
    assert done["state"] == "done" and session._sent == {}
    late = asks(session, "jobs/get", running(sightings=[sighting()], windows=7))
    assert set(late) == lean() and session._sent == {}, "no memory is made again, and the job is not shown as running afresh"
    again = asks(session, "jobs/get", running(state="done", stopped="range_end", sightings=[sighting()], windows=9))
    assert again["sightings"] == [sighting()], "asked for after it ended: the whole result"
    monkeypatch.setattr(adapter, "ENDED_KEPT", 3)
    for n in range(6):
        asks(session, "jobs/get", running(state="done", windows=9) | {"job": f"j-{n + 10}"})
    assert len(session._ended) == 3 and list(session._ended) == ["j-13", "j-14", "j-15"], "the ids kept are bounded"


def test_a_reply_is_recorded_when_it_is_given_back_not_when_the_daemon_answered(home, work, memory, red_clip, monkeypatch):
    monkeypatch.chdir(work)
    api_client.call(home, "remember", {"path": str(red_clip), "source": "clips", "t": "2h"})
    session = adapter.Adapter(home)
    ask = {"question": "Is it red? slow:0.3", "since": "3h", "until": "1h", "sources": ["clips"], "wait_s": 0.5, "still_s": 0}
    finish = session.start("ask", ask, "test/1")
    assert session._sent == {}, "the daemon's answer is in, and nothing says the agent has it"
    content, is_error = finish()
    first = json.loads(content[0].text)
    assert not is_error and first["state"] == "running" and set(session._sent) == {first["job"]}
    api_client.call(home, "jobs/cancel", {"job": first["job"]})


def test_a_check_in_the_client_cancelled_while_it_waited_records_nothing(home, work, memory, red_clip, monkeypatch):
    """The SDK abandons the thread of a cancelled call, which went on to the daemon's answer and, once, recorded the reply."""
    monkeypatch.chdir(work)
    given = []
    real = adapter.Adapter._reply

    def spy(self, op, job, everything):
        out = real(self, op, job, everything)
        given.append((op, out, dict(self._sent)))
        return out

    monkeypatch.setattr(adapter.Adapter, "_reply", spy)

    async def main():
        async with Client(adapter.build_server(home)) as client:
            await ok(client, "remember", path=str(red_clip), source="clips", t="2h")
            range_ = {"since": "3h", "until": "1h", "sources": ["clips"], "still_s": 0}  # a window a second: the scan's pace is timed
            first = await ok(client, "ask", question="Is it red? slow:0.4", wait_s=0, **range_)
            with anyio.move_on_after(1.0) as scope:  # the check-in waits for the job to end, and is cancelled
                await client.call_tool("ask", {"job": first["job"], "wait_s": 30})
            assert scope.cancelled_caught
            await anyio.sleep(6)  # the abandoned thread has the daemon's answer by now
            return first, await ok(client, "ask", job=first["job"], wait_s=30)

    first, later = anyio.run(main)
    assert [op for op, *_ in given] == ["jobs/start", "jobs/get"], "the cancelled check-in was never answered"
    assert later["state"] == "done" and len(later["sightings"]) == 1 and "stopped" in later, \
        "what the cancelled check-in would have carried is carried by this one"


# source_set

def test_source_set_reloads_a_folder_live_and_leaves_what_needs_the_owner_unapproved(home, work, memory):
    notes = work / "notes"
    notes.mkdir()
    (notes / "a.md").write_text("alpha")
    tick = [sys.executable, "-c", "import time; print('{\"hello\": \"tick\"}', flush=True); time.sleep(120)"]
    api = Api(home)

    async def until(check, what: str, timeout: float = 30):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if found := await check():
                return found
            await anyio.sleep(0.1)
        raise AssertionError(f"timed out waiting for {what}")

    async def steps(client):
        out = {"pid": (await ok(client, "sources"))["pid"]}
        out["folder"] = await ok(client, "source_set", name="notes", kind="folder", path="notes", glob="*.md", max_gb=1)

        async def ingested():
            rows = (await ok(client, "find", sources=["notes"]))["results"]
            return rows or None

        out["ingested"] = await until(ingested, "the folder source to ingest")
        out["path"] = configuration.load(home).sources["notes"].options["path"]
        out["unapproved"] = {}
        for name, table in (("watch", {"kind": "screen", "device": "0"}),
                            ("mic", {"kind": "microphone", "device": "Fake Mic"}),
                            ("cam", {"kind": "camera", "device": "Fake Cam"}),
                            ("tick", {"kind": "command", "command": tick})):
            out["unapproved"][name] = (await ok(client, "source_set", name=name, max_gb=1, **table))["source"]

        async def listed():  # the daemon reads the new tables at its next loop
            found = {one["name"]: one for one in (await ok(client, "sources"))["sources"]}
            return found if {"watch", "mic", "cam", "tick"} <= set(found) else None

        out["listed"] = await until(listed, "the daemon to read the new sources")
        out["no_records"] = (await ok(client, "find", sources=["tick"]))["results"]
        out["tools"] = [tool.name for tool in (await client.list_tools()).tools]
        out["approve_field"] = await failed(client, "source_set", name="mic", kind="microphone", max_gb=1, approved=True)
        approval = configuration.load(home).sources["tick"].what()
        consent.record(home, "tick", approval)  # what `mms approve` does once the owner has typed the name

        async def ticked():
            rows = (await ok(client, "find", sources=["tick"], text="hello"))["results"]
            return rows or None

        out["ticked"] = await until(ticked, "the approved command to run")
        out["after"] = {one["name"]: one["state"] for one in (await ok(client, "sources"))["sources"]}
        out["removed"] = await ok(client, "source_set", name="notes", remove=True)
        out["pid_after"] = (await ok(client, "sources"))["pid"]
        return out

    out = drive(steps, home, work)
    assert out["folder"]["source"]["state"] != "needs_approval" and "approve" not in out["folder"]["source"]
    assert os.path.normcase(out["path"]) == os.path.normcase(str(notes)), "a relative path is made absolute"
    assert out["ingested"][0]["name"] == "a.md" and out["pid_after"] == out["pid"], "no restart"
    for name, reply in out["unapproved"].items():
        assert reply["state"] in ("needs_approval", "pending_reload"), name  # until the daemon has read the new table
        assert reply["approve"] == approve_hint(name), name
        assert out["listed"][name]["state"] == "needs_approval" and out["listed"][name]["approve"] == approve_hint(name)
    assert out["no_records"] == [], "an unapproved command is not run"
    assert not any("approve" in name for name in out["tools"])
    assert out["approve_field"]["code"] == "bad_request" and out["approve_field"]["field"] == "approved"
    assert api.raw("approve", {"source": "tick"})[0] == 404
    assert consent.load(home)["tick"]["what"].startswith("command"), "the owner's, only"
    assert set(consent.load(home)) == {"tick"}
    assert out["after"]["tick"] != "needs_approval" and out["after"]["mic"] == "needs_approval"
    assert out["removed"]["source"] == {"name": "notes", "removed": True}
    assert "[sources.notes]" not in (home / "mms.toml").read_text(encoding="utf-8")


def test_nothing_reachable_through_mcp_writes_consent(home, work, memory):
    """Every tool, called with whatever it takes, leaves `consent.json` as it was (TESTBED.md item 4)."""
    async def steps(client):
        await ok(client, "source_set", name="mic", kind="microphone", device="Fake Mic", max_gb=1)
        for name, arguments in (("find", {"text": "x"}), ("look", {"source": "mic", "t": "now"}), ("remember", {"note": "n"}),
                                ("label", {"text": "l", "since": "1h"}), ("sources", {}),
                                ("ask", {"question": "Is it red?", "since": "1h"}),
                                ("source_set", {"name": "mic", "kind": "microphone", "device": "Fake Mic", "max_gb": 2}),
                                ("source_set", {"name": "mic", "remove": True})):
            await client.call_tool(name, arguments)

    drive(steps, home, work)
    assert not (home / "consent.json").exists()


# what the ask description tells the agent of the model (DESIGN.md §12.3)

def paragraph_for(home: Path, spec: str | None) -> str:
    """The model paragraph for a home whose `[decider] model` is `spec` (not set when None)."""
    model = "" if spec is None else spec.replace("\\", "/")  # a path in TOML has forward slashes
    write_config(home, "" if spec is None else f'[decider]\nmodel = "{model}"\n')
    return adapter.model_paragraph(home)


@pytest.mark.parametrize("name", [name for name in MODEL_FOLDERS if name != "unclaimed"])
def test_the_ask_description_gives_the_facts_of_each_model_a_folder_holds(home, tmp_path, name):
    about = decider.about_of(str(model_folder(tmp_path, name)))
    if about is None:
        pytest.skip(f"no backend claims a {name} folder yet")
    text = paragraph_for(home, str(tmp_path / name))
    assert text.startswith(f"The model now is {about.name}."), text
    assert f"Its default yes/no threshold (hit) is {about.hit:g}" in text
    assert ("(assumed, not yet measured here)" in text) == ("hit" not in about.measured), text
    listed = re.search(r"It answers (.*?)\.(?: |$)", text).group(1).split(" (")[0]
    assert re.split(r", | and ", listed) == list(about.types), text
    if "audio" in about.reads:
        assert "hears sound" in text and "hears nothing" not in text
        assert ("but not speech" in text) == (not about.speech)
    else:
        assert "It hears nothing: sound is not read." in text and "hears sound" not in text
    assert ("pictures" in text.split("It sees")[1].split(".")[0]) == ("image" in about.reads), text
    assert ("video frames" in text.split("It sees")[1].split(".")[0]) == ("video" in about.reads), text
    if about.questions_per_read == 1:
        assert "It reads one question a pass" in text
    elif about.questions_per_read:
        assert f"It reads {about.questions_per_read} questions in one pass." in text
    else:
        assert "any number of questions in one pass" in text
    if set(about.question_keys) != set(decider.QUESTION_KEYS):
        assert "A question carries only" in text and all(key in text for key in about.question_keys)
    else:
        assert "A question carries" not in text
    assert about.notes and f"Its notes: models/{about.notes} in the mms-questions skill." in text
    assert (install.SKILL_SOURCE / "models" / about.notes).is_file()
    assert len(text) < 800 and "\n" not in text


def test_the_ask_description_says_what_d1_omni_does_and_does_not_read(home, tmp_path):
    folder = model_folder(tmp_path, "d1-Omni-2B")
    text = paragraph_for(home, str(folder))
    assert text.startswith("The model now is d1-Omni-2B. It sees pictures and video frames and hears sound, but not speech. "
                           "It answers noul, choice, score, text, number, date, box and boxes (text, date and a number "
                           "with no range only from text and records). Its default yes/no threshold (hit) is 0.8")
    assert text.endswith("It reads 16 questions in one pass. Its notes: models/d1.md in the mms-questions skill."), text


def test_the_paragraph_is_made_from_the_about_alone_so_a_deaf_unmeasured_model_reads_so(home, tmp_path, monkeypatch):
    deaf = decider.About(name="deaf-one", reads={"video", "image"}, types=("noul", "choice"), extracts_from=(),
                         question_keys={"criteria"}, questions_per_read=1, hit=0.7, notes="d3.md")
    monkeypatch.setattr(decider, "about_of", lambda spec: deaf)
    text = paragraph_for(home, str(tmp_path))
    assert text == ("The model now is deaf-one. It sees pictures and video frames. It hears nothing: sound is not read. "
                    "It answers noul and choice. A question carries only criteria besides type, instructions and hit. "
                    "Its default yes/no threshold (hit) is 0.7 (assumed, not yet measured here). It reads one question a pass, "
                    "so each further question reads the window again. Its notes: models/d3.md in the mms-questions "
                    "skill.")
    measured = dataclasses.replace(deaf, measured={"hit"}, notes="", question_keys={"criteria", "range", "unit", "step",
                                                                                      "multiple"}, questions_per_read=0)
    monkeypatch.setattr(decider, "about_of", lambda spec: measured)
    text = paragraph_for(home, str(tmp_path))
    assert "assumed" not in text and "Its notes" not in text and "A question carries" not in text
    assert "It reads any number of questions in one pass." in text
    ears = dataclasses.replace(measured, reads={"audio"}, speech=True)
    monkeypatch.setattr(decider, "about_of", lambda spec: ears)
    assert paragraph_for(home, str(tmp_path)).startswith(
        "The model now is deaf-one. It sees nothing: pictures and video are not read. It hears sound and speech.")


def test_a_python_decider_and_no_model_have_their_own_lines(home, tmp_path):
    assert paragraph_for(home, "python:/somewhere/mine.py") == (
        "The model now is a python decider, mine.py; ask's warnings say what it does not read.")
    none = "The model now is none: no model is configured (or its folder cannot be read), so ask fails until the owner runs `mms model use NAME` (`mms model list` shows them)."
    assert paragraph_for(home, None) == none
    assert paragraph_for(home, str(tmp_path / "missing")) == none, "a folder that is not there"
    assert paragraph_for(home, str(model_folder(tmp_path, "unclaimed"))) == none, "a folder no backend reads"
    (home / "mms.toml").write_text("[decider\nmodel = ", encoding="utf-8")
    assert adapter.model_paragraph(home) == none, "a config that does not parse"
    long = "python:/x/" + "n" * 200 + ".py"
    assert len(paragraph_for(home, long)) < 160


def test_tools_list_carries_the_paragraph_within_a_thousand_characters_and_imports_nothing_heavy(
        home, work, memory, tmp_path):
    folder = model_folder(tmp_path, "d1-Omni-2B")
    write_config(home, f'[memory]\nloop_s = 0.1\nfree_gb = 0\n[decider]\nmodel = "{folder.as_posix()}"\n')
    modules_file = tmp_path / "modules.json"

    async def main():
        async with connected(home, work, modules=modules_file) as client:
            return {tool.name: tool.description for tool in (await client.list_tools()).tools}

    described = anyio.run(main)
    assert described["ask"].startswith(adapter.DESCRIPTIONS["ask"] + "\n\nThe model now is d1-Omni-2B. ")
    assert len(described["ask"]) <= 1024, len(described["ask"])
    assert all(described[name] == adapter.DESCRIPTIONS[name] for name in described if name != "ask")
    loaded = set(json.loads(modules_file.read_text()))
    assert "mms.adapter" in loaded and "mms.decider" in loaded and not HEAVY & loaded, sorted(HEAVY & loaded)


def test_every_paragraph_leaves_room_in_the_description_whatever_the_model(home, tmp_path):
    biggest = decider.About(name="n" * 40, reads=decider.MEDIA_KINDS, types=decider.TYPES, speech=False,
                            question_keys={"criteria"}, questions_per_read=1, hit=0.55, notes="x" * 20 + ".md")
    for about in [*decider.known_abouts(), biggest]:
        size = len(adapter.DESCRIPTIONS["ask"]) + 2 + len(adapter.about_paragraph(about))
        assert size <= 1024, (about.name, size)
