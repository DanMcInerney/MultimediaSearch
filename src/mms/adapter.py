"""`mms mcp` (DESIGN.md §12): one MCP session over stdio, translated to the daemon's API.

It keeps no memory of its own but what its replies to an `ask` job carried, so that a check-in sends only what changed.
A tool call is one API call: its arguments, plus `caller`, with relative paths made absolute. It imports the SDK,
`client.py`, `config.py`, `errors.py` and `project.py` only (and `decider.py`, for the model's facts below), so it
starts fast, and it answers `initialize` and `tools/list` before it touches git or the daemon. Only MCP messages go to
stdout. The one thing `tools/list` reads is the configured model's facts (`model_paragraph`), from the config file and
the model's folder, with no model loaded.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import anyio
from mcp import types
from mcp.server import Server

from mms import client, config, project
from mms.errors import ApiError

CALL_BUDGET_S = 50.0  # a tool call's whole time: Codex stops a call at 60 s
MARGIN_S = 5.0  # of it, kept for the daemon's own overhead beyond `ask`'s wait
WAIT_S = 40  # `ask`'s default wait, as the API's
ENDED = ("done", "failed", "cancelled")  # a job's final states, as jobs.py has them
ENDED_KEPT = 200  # the ended jobs whose ids are remembered, as many as the daemon keeps
Content = types.ImageContent | types.TextContent

INSTRUCTIONS = """\
MultimediaSearch is the owner's long memory of this machine: screen, microphone, cameras, folders, logs, programs and agent \
notes, kept as the raw recording. Use find for instant search by time, source and words. Use ask to have a local \
model look at or listen to the raw record in a time range: put the range in since and until, never in the \
question, and ask about one thing that could be seen or heard. Check each sighting with look before trusting it.

ask returns within about 45 s with what it has found so far and a job id: call ask with that job to collect more, or \
with cancel to stop it. A sighting gives a source, a time span, p and a look handle; candidates are the best windows \
that did not count, best first, to look at when the sightings do not hold what you need (look takes several at \
once in moments, and over a span, with source, since and until, shows up to 9 frames in one picture). Files the \
model does not read, such as PDFs, are named in ask's warnings, and look gives each as a file to open. Reads cover \
the ambient sources and this project unless scope is "all", and every result says what was searched. remember \
stores a note, file, folder or recording at its real time; label marks a time range. sources lists what is \
recorded and the capture devices; source_set adds or changes a source and needs a cap. Screen, microphone, camera \
and command sources wait until the owner runs `mms approve <name>` in a terminal: tell the owner, and never run it \
yourself. The mms-questions skill says how to write questions for ask."""

DESCRIPTIONS = {
    "ask": "Have a local model look at or listen to the raw record in a time range, and return sightings. "
           "Returns within about 45 s with what it found so far and a job id; call again with job to collect more "
           "(it returns what changed since the last call; all: true returns everything), or cancel.",
    "find": "Instant search of the memory by time, source and words: items, records and labels, newest first. "
            "When the result's next is set, call again with after set to it for the next page.",
    "look": "Show one moment, or up to 8 at once with moments: a picture inline (and its file path), a sound's file, "
            "text, or a copy of a file that is none of these (a PDF, say). Or a source over since and until: one picture "
            "of up to 9 numbered frames across the range. With region it shows only that part of the picture, enlarged. "
            "Check a sighting with it before trusting it.",
    "remember": "Store a note, file, folder or recording once, at its real time, with an optional expiry.",
    "label": "Mark a time range with text; with keep and a source, keep the media and files it covers from caps "
             "(records are not kept).",
    "sources": "List the sources, their state and size, the capture devices, the floor and the model.",
    "source_set": "Add, change or remove a continuous source: screen, microphone, camera, folder, log, command or "
                  "program. A cap is required; only the fields given change, and a cap or max_days can only stay or "
                  "rise (the owner lowers them in mms.toml). Screen, microphone, camera and command sources wait "
                  "for the owner's approval. The reply says so, and says while the daemon has not yet applied the "
                  "change.",
}


def _schema(properties: dict, required: tuple[str, ...] = ()) -> dict:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = list(required)
    return schema


_TIME = {"type": ["string", "number"], "description": "now, an age like 90s, 30m, 2h, 7d or 1w (that long ago), "
                                                      "ISO 8601 (no zone is local), or epoch seconds"}
_SOURCES = {"type": "array", "items": {"type": "string"}, "description": "narrow to these sources"}
_SCOPE = {"enum": ["default", "all"], "default": "default",
          "description": "default reads the ambient sources and this project; all reads every project"}
_SOURCE_NAME = {"type": "string", "pattern": config.NAME.pattern}

_ASK = _schema({
    "question": {"type": "string", "description": "one thing that could be seen or heard; yes/no unless answers"},
    "answers": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                "description": "the options of a choice question"},
    "hit": {"anyOf": [{"type": "number", "minimum": 0, "maximum": 1}, {"type": "string"},
                      {"type": "array", "items": {"type": "string"}}],
            "description": "what counts: a P(yes) threshold (the model's own if not given: this tool's description says "
                           "it), or the option or options that count (default: the first answer)"},
    "context": {"type": "string", "maxLength": 4000,
                "description": "state text given with every window: what the source shows, not what is hoped for"},
    "questions": {"type": "object", "additionalProperties": {"type": "object"},
                  "description": "instead of question, answers and hit: typed questions by name, as "
                                 "{\"mug\": {\"type\": \"noul\", \"instructions\": \"Is a mug in view?\", \"hit\": 0.8}}. "
                                 "A window counts only when every question with a hit hits; to find any of several "
                                 "things, leave out questions and give question, answers (the things) and hit (all of them)"},
    "since": _TIME, "until": _TIME, "sources": _SOURCES, "scope": _SCOPE,
    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10,
              "description": "sightings to find; 1 is last seen"},
    "candidates": {"type": "integer", "minimum": 0, "maximum": 20, "default": 5,
                   "description": "the best windows that did not count to give as well, best first, one per moment; "
                                  "0 for none"},
    "oldest_first": {"type": "boolean", "default": False},
    "best_first": {"type": "boolean", "default": False,
                   "description": "read the whole range and give the limit sightings with the highest p, best first, "
                                  "instead of the first limit found in scan order"},
    "every_s": {"type": "number", "minimum": 0, "default": 1,
                "description": "video: a frame per this many seconds of the clock; 0 for every frame"},
    "still_s": {"type": "number", "minimum": 0, "default": 30,
                "description": "video: a frame that looks the same as the one that began a window joins it, for at most "
                               "this many seconds; 0 reads every frame the step picks"},
    "max_px": {"type": "integer", "minimum": 0,
               "description": "the most pixels of a picture the model sees: 230400 (640x360) if not given, unless the "
                              "model sets its own; never above the model's own cap, which 0 gives. Raise it to read "
                              "small text, at several times the cost"},
    "merge_s": {"type": "number", "minimum": 0, "default": 10,
                "description": "hits of a source this close are one sighting"},
    "max_sighting_s": {"type": "number", "minimum": 0, "default": 600,
                       "description": "a sighting is cut here and marked capped; 0 for no cap"},
    "max_s": {"type": "number", "minimum": 1, "maximum": 3600, "default": 600,
              "description": "stop scanning after this many seconds, counted from the model's first answer (loading the "
                             "model is not counted)"},
    "job": {"type": "string", "description": "a job id from an earlier ask, to collect more; only cancel, all and "
                                             "wait_s may be given with it"},
    "cancel": {"type": "boolean", "description": "with job: stop the job"},
    "all": {"type": "boolean", "description": "with job: everything the job has found, not only what changed since the "
                                              "last call"},
    "wait_s": {"type": "number", "minimum": 0, "maximum": 50, "default": WAIT_S,
               "description": "wait up to this long for the job to end (the call returns within about 45 s)"},
})

_EDGES = ("top", "left", "bottom", "right")
_REGION = {**_schema({edge: {"type": "number", "minimum": 0, "maximum": 1} for edge in _EDGES}, _EDGES),
           "description": "only this part of the picture, enlarged: fractions from its top left, as ask gives a box's edges"}

_LOOK = {
    "item": {"type": "integer", "description": "an item id, with at"},
    "at": {"type": "number", "minimum": 0, "default": 0, "description": "seconds into the item"},
    "source": {"type": "string", "description": "with t, with run and tp, or with since and until"},
    "t": _TIME,
    "since": {**_TIME, "description": "with source and until: the start of the range a sheet of up to 9 frames covers"},
    "until": {**_TIME, "description": "with source and since: the end of the range, after since"},
    "run": {"type": "string", "description": "a program run, with source and tp"},
    "tp": {"type": "number", "description": "the program's own time in the run"},
    "offset": {"type": "integer", "minimum": 0, "description": "characters to skip when paging through text"},
    "region": _REGION,
}

SCHEMAS: dict[str, dict] = {
    "ask": _ASK,
    "find": _schema({
        "text": {"type": "string", "description": "words that must all match"},
        "since": _TIME, "until": _TIME, "sources": _SOURCES, "scope": _SCOPE,
        "types": {"type": "array", "items": {"enum": ["item", "record", "label"]},
                  "description": "some of item, record and label (default all three)"},
        "as_of": {**_TIME, "description": "instead of a range: the version of each file current at this time"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
        "after": {"type": "string", "description": "the next page: the next of an earlier result that was cut by limit"},
    }),
    "look": _schema({
        **_LOOK,
        "moments": {"type": "array", "minItems": 1, "maxItems": 8, "items": _schema(_LOOK),
                    "description": "instead of the fields above: up to 8 look handles (as sightings, candidates and "
                                   "find give them), shown in one call"},
    }),
    "remember": _schema({
        "note": {"type": "string", "maxLength": 100000, "description": "a note; give note or path"},
        "path": {"type": "string", "description": "a file, folder, picture or recording; relative to the working directory"},
        "t": {**_TIME, "description": "when it happened (default now, or a picture's, video's or sound's own time)"},
        "expires": {"type": ["string", "number"], "description": "a time ahead, or a span from now like 30d"},
        "name": {"type": "string", "description": "a note's name, or a file's (default its file name)"},
        "source": {**_SOURCE_NAME, "description": "a push source (default agent)"},
    }),
    "label": _schema({
        "text": {"type": "string"},
        "since": _TIME,
        "until": {**_TIME, "description": "the range's end (default since)"},
        "source": {"type": "string"},
        "keep": {"type": "boolean", "description": "keep what the label covers from caps; needs source"},
    }, ("text", "since")),
    "sources": _schema({}),
    "source_set": _schema({
        "name": {**_SOURCE_NAME, "description": "the source's name; with remove, the only other field"},
        "kind": {"enum": list(config.SET_KINDS)},
        "device": {"type": "string", "description": "screen, microphone, camera"},
        "fps": {"type": "number", "exclusiveMinimum": 0, "description": "screen, camera"},
        "path": {"type": "string", "description": "folder, log; relative to the working directory"},
        "glob": {"type": "string", "description": "folder, log"},
        "command": {"type": "array", "items": {"type": "string"}, "description": "command"},
        "time_field": {"type": "string", "description": "log, command: the JSON field holding a record's time"},
        "max_gb": {"type": "number", "exclusiveMinimum": 0, "description": "the cap; required unless remove"},
        "when_full": {"enum": list(config.WHEN_FULL), "default": "drop_oldest"},
        "max_days": {"type": "number", "exclusiveMinimum": 0},
        "remove": {"type": "boolean", "description": "delete the source's table; its history stays"},
    }, ("name",)),
}
TOOL_NAMES = tuple(SCHEMAS)
PATH_FIELDS = {"remember": "path", "source_set": "path"}  # arguments the daemon takes as absolute paths
DIRECT_OPS = {"find": "find", "look": "look", "remember": "remember", "label": "label", "source_set": "source_set"}


def _list(words: list[str]) -> str:
    return ", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else "".join(words)


def model_paragraph(home: Path) -> str:
    """The paragraph at the end of the `ask` description: the facts of the model `[decider] model` names, for an agent
    that has not asked yet (DESIGN.md §12.3). It reads the config file and the model's folder (`decider.about_of`), and
    loads and starts nothing."""
    from mms import decider

    try:
        spec = config.load(home).decider_model
        about = decider.about_of(spec)
    except (ValueError, OSError):  # a config that does not parse, or a folder that cannot be read
        spec, about = None, None
    if about is not None:
        return about_paragraph(about)
    if spec and spec.startswith("python:"):
        name = Path(spec[len("python:"):]).name
        return (f"The model now is a python decider, {name if len(name) <= 60 else name[:57] + '...'}; ask's "
                "warnings say what it does not read.")
    return ("The model now is none: no model is configured (or its folder cannot be read), so ask fails until the "
            "owner runs `mms model use NAME` (`mms model list` shows them).")


def about_paragraph(about) -> str:
    """What an agent needs of a model before its first `ask`, from its `decider.About`."""
    from mms.decider import QUESTION_KEYS

    sees = _list([word for kind, word in (("image", "pictures"), ("video", "video frames")) if kind in about.reads])
    hears = "sound" + (" and speech" if about.speech else ", but not speech")
    if sees and "audio" in about.reads:
        senses = f"It sees {sees} and hears {hears}."
    elif sees:
        senses = f"It sees {sees}. It hears nothing: sound is not read."
    else:
        senses = "It sees nothing: pictures and video are not read." + (" It hears " + hears + "."
                                                                        if "audio" in about.reads else "")
    text = [f"The model now is {about.name}.", senses]
    types = f"It answers {_list(list(about.types))}"
    if {"text", "date", "number"} & set(about.types):
        order = ("text", "records", "image", "video", "audio", "file")
        types += (" (text, date and a number with no range only from "
                  f"{_list([kind for kind in order if kind in about.extracts_from])})")
    text.append(types + ".")
    keys = [key for key in QUESTION_KEYS if key in about.question_keys]
    if len(keys) < len(QUESTION_KEYS):
        carried = f"only {_list(keys)} besides type, instructions and hit" if keys else "only type, instructions and hit"
        text.append(f"A question carries {carried}.")
    text.append(f"Its default yes/no threshold (hit) is {about.hit:g}"
                + ("" if "hit" in about.measured else " (assumed, not yet measured here)") + ".")
    passes = about.questions_per_read
    text.append("It reads any number of questions in one pass." if not passes else
                "It reads one question a pass, so each further question reads the window again." if passes == 1 else
                f"It reads {passes} questions in one pass.")
    if about.notes:
        text.append(f"Its notes: models/{about.notes} in the mms-questions skill.")
    return " ".join(text)


def describe(name: str, home: Path) -> str:
    """A tool's description: the fixed text, and for `ask` the model paragraph."""
    if name != "ask":
        return DESCRIPTIONS[name]
    return DESCRIPTIONS[name] + "\n\n" + model_paragraph(home)


def bad_request(message: str, field: str | None = None) -> ApiError:
    return ApiError("bad_request", message, field=field)


class Adapter:
    """The session: its ID, its project, and the one daemon every tool call goes to."""

    def __init__(self, home: Path | None = None, tools: tuple[str, ...] | None = None):
        self.home = Path(home) if home else config.default_home()
        self.offered = TOOL_NAMES if tools is None else tuple(name for name in TOOL_NAMES if name in tools)
        self.session = "s-" + secrets.token_hex(6)
        self.project: project.Project | None = None
        self._prepared = threading.Event()
        self._project_error: str | None = None
        self._start_error: ApiError | None = None
        self._started = False
        self._sent: dict[str, dict] = {}  # per ask job not yet ended, what the agent was sent (`_reply`)
        self._ended: dict[str, None] = {}  # the ids of jobs whose final state was sent, the newest ENDED_KEPT
        self._lock = threading.Lock()

    def prepare_in_background(self) -> None:
        """Find the project and make sure a daemon runs, in a thread; a tool call waits for it."""
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._prepare, name="mms-prepare", daemon=True).start()

    def _prepare(self) -> None:
        try:
            try:
                self.project = project.find(project.folder())
            except Exception as error:  # a bad .mms-project, or a git that fails: every call says so
                self._project_error = f"{type(error).__name__}: {error}"
                print(f"mms mcp: the project cannot be found: {self._project_error}", file=sys.stderr)
            try:
                client.ensure(self.home)
            except ApiError as error:
                self._start_error = error
                print(f"mms mcp: {error.code}: {error.message}", file=sys.stderr)
        finally:
            self._prepared.set()

    def call(self, name: str, arguments: dict[str, Any], client_name: str | None) -> tuple[list[Content], bool]:
        """Run tool `name`: (the content blocks, whether it is an error). Raises `ApiError`."""
        return self.start(name, arguments, client_name)()

    def start(self, name: str, arguments: dict[str, Any], client_name: str | None) -> Callable[[], tuple[list[Content], bool]]:
        """The part of a tool call that waits for the daemon, so it may run in a thread. It returns the function that gives
        the reply, which the caller runs once the reply is really going back, on one thread: for `ask` it also records what
        the agent is sent (`_reply`), so a call the client cancelled while it waited records nothing."""
        deadline = time.monotonic() + CALL_BUDGET_S
        if name not in self.offered:
            raise ApiError("not_found", f"there is no tool {name!r}", field="tool")
        op, params, wait = self._translate(name, dict(arguments))
        self.prepare_in_background()
        if not self._prepared.wait(max(0.0, deadline - MARGIN_S - time.monotonic())):
            raise client.DaemonUnavailable("the daemon is still starting")
        if self._project_error is not None:
            raise ApiError("bad_project", f"this session's project cannot be found: {self._project_error}",
                           hint="fix the .mms-project file or git, and restart the agent")
        with self._lock:
            error, self._start_error = self._start_error, None  # reported once; the next call tries again
        if error is not None:
            raise error
        caller = {"session": self.session, "project": self.project.id if self.project else None,
                  "project_name": self.project.name if self.project else None,
                  "project_root": self.project.root if self.project else None, "client": client_name}
        result = self._invoke(op, params, wait, caller, deadline)
        if name == "ask":
            everything = bool(arguments.get("all"))
            return lambda: ([_text_block(self._reply(op, result, everything))], result.get("state") == "failed")
        if name != "look":
            return lambda: ([_text_block(result)], False)
        looks = result["moments"] if "moments" in result else [result]  # each: its picture, if any, then its JSON
        blocks: list[Content] = []
        errors = 0
        for look in looks:
            try:
                picture = [_inline_image(look)] if look.get("kind") in ("image", "sheet") else []
            except ApiError as error:
                if "moments" not in result:
                    raise
                look, picture = {"error": error.body()}, []  # this moment's picture is lost; the others still show
            blocks += [*picture, _text_block(look)]
            errors += "error" in look
        return lambda: (blocks, errors == len(looks))

    def _translate(self, name: str, arguments: dict[str, Any]) -> tuple[str, dict, float | None]:
        """The API op, its parameters and (for a job call) the wait the caller asked for."""
        if name == "ask":
            return self._ask(arguments)
        if name == "sources":
            return "status", {"devices": True}, None
        field = PATH_FIELDS.get(name)
        if field and isinstance(arguments.get(field), str):
            arguments[field] = os.path.abspath(arguments[field])
        return DIRECT_OPS[name], arguments, None

    @staticmethod
    def _ask(arguments: dict[str, Any]) -> tuple[str, dict, float | None]:
        wait = arguments.pop("wait_s", None)
        wait = float(WAIT_S if wait is None else wait)
        job = arguments.pop("job", None)
        if job is not None:
            extra = sorted(set(arguments) - {"cancel", "all"})
            if extra:
                raise bad_request(f"with job, ask takes only cancel, all and wait_s, not {extra[0]}", extra[0])
            if arguments.get("cancel"):
                return "jobs/cancel", {"job": job}, None
            return "jobs/get", {"job": job}, wait
        if "cancel" in arguments:
            raise bad_request("cancel needs the job to cancel", "cancel")
        if "all" in arguments:
            raise bad_request("all needs the job to collect", "all")
        if "question" not in arguments and "questions" not in arguments:
            raise bad_request("ask needs question or questions (or job, to collect an earlier ask)", "question")
        return "jobs/start", {"script": "ask", "params": arguments}, wait

    def _reply(self, op: str, job: dict, everything: bool) -> dict:
        """The agent's view of an ask job (`_job_output`): the whole result for the call that started it, for `all` and for
        a job not remembered, else the fields that changed since the last reply. A job's memory goes with its final state.
        A reply is recorded when it is made, so this runs on one thread only, once the reply is really going back.
        Overlapping check-ins can finish out of order: a snapshot with fewer windows than the last sent, or of a job whose
        final state was sent, gets `job`, `state` and `progress` only, and records nothing."""
        job_id, final = job["job"], job["state"] in ENDED
        known = self._sent.get(job_id)
        windows = ((job.get("result") or {}).get("progress") or {}).get("windows") or 0
        if not final and (job_id in self._ended or (known is not None and windows < known["windows"])):
            return _job_output(job, _fields(job))
        sent = {} if op == "jobs/start" or everything or known is None else known["fields"]
        out = _job_output(job, sent)
        if final:
            self._sent.pop(job_id, None)
            self._ended[job_id] = None
            while len(self._ended) > ENDED_KEPT:
                del self._ended[next(iter(self._ended))]
        else:
            now = {key: value for key, value in out.items() if key not in ("job", "state", "progress", "next")}
            self._sent[job_id] = {"windows": windows, "fields": {**sent, **now}}
        return out

    def _invoke(self, op: str, params: dict, wait: float | None, caller: dict, deadline: float) -> dict:
        """One API call, and one more after `ensure` if no daemon was reached. The time left is worked out inside each
        attempt, so that a retry after a slow start does not ask the daemon to wait longer than the call has."""
        def attempt() -> dict:
            left = deadline - time.monotonic()
            body = dict(params) if wait is None else {**params, "wait_s": max(0.0, min(wait, left - MARGIN_S))}
            return client.call(self.home, op, body, caller=caller, timeout_s=max(1.0, left))

        try:
            return attempt()
        except client.DaemonUnavailable:  # the daemon stopped since the session began; the call never reached one
            client.ensure(self.home, wait_s=max(1.0, deadline - MARGIN_S - time.monotonic()))
            return attempt()


def _fields(job: dict) -> dict:
    """What a job reports beside its `job`, `state` and `progress`, without the None ones: its error, then its result."""
    return {key: value for key, value in {"error": job.get("error"), **(job.get("result") or {})}.items()
            if value is not None and key != "progress"}


def _job_output(job: dict, sent: dict) -> dict:
    """`{"job", "state", "progress", ...the fields that are not as `sent` has them, "next"?}` (§12.1). `sent` holds the
    fields the agent was sent before: a full reply passes none. A field goes whole or not at all, and a None one never:
    the sightings, the candidates, the warnings, `searched`, `stopped` and `error` each when their value differs."""
    out = {"job": job["job"], "state": job["state"]}
    for key, value in {"error": job.get("error"), **(job.get("result") or {})}.items():
        if value is not None and (key == "progress" or sent.get(key) != value):
            out[key] = value
    if job["state"] in ("queued", "running"):
        out["next"] = f'Call ask with job "{job["job"]}" to collect more, or with cancel: true to stop it.'
    return out


def _text_block(value: Any) -> types.TextContent:
    return types.TextContent(text=dumps(value))


def _inline_image(result: dict) -> types.ImageContent:
    """`look`'s inline JPEG (a file the daemon wrote, long side at most 1568 px) as an image block."""
    path = (result.get("image") or {}).get("inline")
    try:
        data = Path(path).read_bytes()
    except (OSError, TypeError) as error:
        raise ApiError("internal", f"look gave an inline picture that cannot be read: {path!r}: {error!r}") from error
    return types.ImageContent(data=base64.b64encode(data).decode("ascii"), mime_type="image/jpeg")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_server(home: Path | None = None, tools: tuple[str, ...] | None = None) -> Server:
    """The SDK server, not yet running. Over stdio, `main` runs it. It offers `tools` (default all of them)."""
    adapter = Adapter(home, tools)

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        result = types.ListToolsResult(tools=[types.Tool(name=name, description=describe(name, adapter.home),
                                                         input_schema=SCHEMAS[name]) for name in adapter.offered])
        adapter.prepare_in_background()  # after the answer is made, so initialize and tools/list never wait on it
        return result

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        info = getattr(getattr(ctx.session, "client_params", None), "client_info", None)
        client_name = f"{info.name}/{info.version}" if info else None
        try:
            finish = await anyio.to_thread.run_sync(
                adapter.start, params.name, params.arguments or {}, client_name, abandon_on_cancel=True)
            content, is_error = finish()  # on the event loop: a call cancelled while it waited never gets here
        except ApiError as error:
            return types.CallToolResult(content=[_text_block({"error": error.body()})], is_error=True)
        return types.CallToolResult(content=content, is_error=is_error)

    return Server("mms", instructions=INSTRUCTIONS, on_list_tools=list_tools, on_call_tool=call_tool)


async def _stdio(server: Server) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(home: Path | None = None, tools: tuple[str, ...] | None = None) -> int:
    """Serve one MCP session on stdin and stdout until the client closes them."""
    anyio.run(_stdio, build_server(home, tools))
    return 0


if __name__ == "__main__":
    sys.exit(main())
