"""The ops of DESIGN.md §11, from a JSON body to a JSON result: the daemon's API and the CLI call the same functions.

`call(env, op, body)` parses the body once into typed parameters (an unknown field, a wrong type or a value out of bounds
is `bad_request` naming the field), records the caller's project and session, and runs the op. Times in results are
ISO 8601 in the host's zone. The ops of jobs (`jobs/start`, `jobs/get`, `jobs/cancel`) are in jobs.py.
"""

from __future__ import annotations

import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from mms import config as configuration
from mms import consent, ingest
from mms.config import (NAME, PROJECT_ID, SET_KINDS, Config, ConfigError, iso, parse_ahead, parse_when,
                             reads_the_home, when)
from mms.errors import ApiError
from mms.store import (GB, SPANS, Scope, Store, cursor_key, decode_cursor, encode_cursor, item_dict,
                            item_matches, text_words)

SESSION = re.compile(r"^s-[0-9a-f]{12}$")
FIND_LIMIT = (1, 200)
EVERYTHING = Scope(all=True)  # `look` is given a handle, so it reads that item or record whatever its project
LOOK_NEAR_S = 30.0  # records this close to a time are what `look` shows there
NOTE_MAX = 100_000
TEXT_PEEK = 300  # characters of a record in a find result
LABELS_SHOWN = 10
LOOK_TILES = 9  # the times a sheet is made of, spread over its range, ends included


class Env:
    """What an op needs: the home, its store and the config in force. `live` is the running daemon (it knows the floor,
    the sources' states, the decider and the jobs) and None in the CLI. Setting `config` gives the store the floor and
    the caps its writes are checked against."""

    def __init__(self, home: Path, store: Store, config: Config, config_error: str | None = None, live=None):
        self.home, self.store, self.config_error, self.live = home, store, config_error, live
        self.config = config

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        self._config = config
        self.store.set_limits(config)


@dataclass(frozen=True)
class Caller:
    session: str | None = None
    project: str | None = None
    project_name: str | None = None
    project_root: str | None = None
    client: str | None = None

    def scope(self, name: str | None = "default") -> Scope:
        return Scope(self.project, name == "all")

    def asdict(self) -> dict:
        return {"session": self.session, "project": self.project, "project_name": self.project_name,
                "project_root": self.project_root, "client": self.client}


CALLER_FIELDS = ("session", "project", "project_name", "project_root", "client")


def parse_caller(raw) -> Caller:
    if raw is None:
        return Caller()
    if not isinstance(raw, dict):
        raise ApiError("bad_request", "caller is an object", field="caller")
    unknown = sorted(set(raw) - set(CALLER_FIELDS))
    if unknown:
        raise ApiError("bad_request", f"caller has session, project, project_name, project_root and client, not "
                                      f"{unknown}", field="caller")
    session, project = raw.get("session"), raw.get("project")
    if session is not None and not (isinstance(session, str) and SESSION.match(session)):
        raise ApiError("bad_request", "session is 's-' and 12 hex digits, or null", field="caller")
    if project is not None and not (isinstance(project, str) and PROJECT_ID.match(project)):
        raise ApiError("bad_request", "project is an ID such as git:0123456789abcdef, or null", field="caller")
    for key in ("project_name", "project_root", "client"):
        if raw.get(key) is not None and not isinstance(raw[key], str):
            raise ApiError("bad_request", f"{key} is text or null", field="caller")
    return Caller(**{key: raw.get(key) for key in CALLER_FIELDS})


# parameters

def _text(maximum: int | None = None, blank: bool = True):
    def check(value):
        if not isinstance(value, str) or (not blank and not value.strip()):
            raise ValueError("is text" + ("" if blank else " that is not blank"))
        if maximum is not None and len(value) > maximum:
            raise ValueError(f"is at most {maximum:,} characters")
        return value
    return check


def _when(value):
    return parse_when(value)


def _ahead(value):
    return parse_ahead(value)


def _name(value):
    if not isinstance(value, str) or not NAME.match(value):
        raise ValueError("is letters, digits, '_' or '-', up to 64 characters")
    return value


def _names(value):
    if not isinstance(value, list) or not all(isinstance(each, str) for each in value):
        raise ValueError("is a list of source names")
    return [_name(each) for each in value]


def _choice(*choices):
    def check(value):
        if value not in choices:
            raise ValueError("is one of " + ", ".join(map(str, choices)))
        return value
    return check


def _types(value):
    if not isinstance(value, list) or not value or not set(value) <= {"item", "record", "label"}:
        raise ValueError("is some of item, record and label")
    return list(dict.fromkeys(value))


def _int(low: int, high: int | None = None):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
            raise ValueError(f"is a whole number from {low}" + (f" to {high}" if high is not None else ""))
        return value
    return check


def _number(above: float | None = None, atleast: float | None = None):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("is a number")
        if above is not None and not value > above:
            raise ValueError(f"is a number above {above:g}")
        if atleast is not None and not value >= atleast:
            raise ValueError(f"is a number of at least {atleast:g}")
        return float(value)
    return check


def _region(value):
    """The part of a picture `look` shows, as a `reader.Region`: an object of its four edges, each a fraction from 0 to 1."""
    from mms.reader import Region
    shape = ("is an object with top, left, bottom and right, each a number from 0 to 1 (fractions of the picture from its "
             "top left), top above bottom and left before right")
    if not isinstance(value, dict) or set(value) != set(Region._fields):
        raise ValueError(shape)
    for edge, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 1:
            raise ValueError(f"{shape}; {edge} is {number!r}")
    region = Region(*(float(value[edge]) for edge in Region._fields))
    if not (region.top < region.bottom and region.left < region.right):
        raise ValueError(f"{shape}; top is {region.top:g} and bottom {region.bottom:g}, left {region.left:g} and right "
                         f"{region.right:g}")
    return region


def _flag(value):
    if not isinstance(value, bool):
        raise ValueError("is true or false")
    return value


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("is an absolute path")
    if not Path(value).expanduser().is_absolute():
        raise ValueError("is an absolute path")
    return value


def _args(value):
    if not isinstance(value, list) or not value or not all(isinstance(each, str) for each in value):
        raise ValueError("is a list of arguments")
    return value


def parse(op: str, body: dict, fields: dict, required=()) -> dict:
    """The body's parameters checked against `fields` (name to checker): `bad_request` names the field at fault."""
    for name in body:
        if name not in fields:
            raise ApiError("bad_request", f"{op} takes {', '.join(fields) or 'no parameters'}, not {name!r}", field=name)
    for name in required:
        if body.get(name) is None:
            raise ApiError("bad_request", f"{op} needs {name}", field=name)
    out = {}
    for name, value in body.items():
        if value is None:
            continue
        try:
            out[name] = fields[name](value)
        except ValueError as error:
            raise ApiError("bad_request", f"{name} {error}", field=name) from None
    return out


def _range(p: dict, until_default: float | None) -> tuple[float | None, float | None]:
    since, until = p.get("since"), p.get("until", until_default)
    if since is not None and until is not None and until < since:
        raise ApiError("bad_request", "until is before since", field="until")
    return since, until


def _after(t: float | None) -> float | None:
    """The bound a range's end becomes when the end is to be included."""
    return None if t is None else math.nextafter(t, math.inf)


# searched, and what a row looks like

def searched(env: Env, caller: Caller, since: float | None, until: float | None, sources, scope: str = "default") -> dict:
    """What a read covered (DESIGN.md §5.2)."""
    visible = caller.scope(scope)
    if sources:
        names = list(sources)
    else:
        names = sorted(set(env.store.source_names(visible)) | {
            name for name, source in env.config.sources.items()
            if visible.all or source.project is None or source.project == visible.project})
    project = None if caller.project is None else {"id": caller.project, "name": env.store.project_name(caller.project)}
    return {"now": iso(time.time()), "since": iso(since), "until": iso(until), "scope": scope or "default",
            "project": project, "sources": names}


def look_handle(row: dict) -> dict | None:
    """The `look` handle of a find result: an item and `at`, or a source and `t`."""
    if row["type"] == "item":
        return {"item": row["id"], "at": 0}
    if row["source"] is None:
        return None
    return {"source": row["source"], "t": iso(row["t0"])}


def result_row(env: Env, row: dict) -> dict:
    out = {"type": row["type"], "id": row["id"], "source": row["source"], "t0": iso(row["t0"]),
           "t1": iso(row["t1"]), "project": env.store.project_name(row["project"]), "look": look_handle(row)}
    if row["type"] == "item":
        out.update(kind=row["kind"], name=row["name"], media=row["meta"].get("media"), bytes=row["bytes"],
                   keep=bool(row["keep"]))
    elif row["type"] == "record":
        out["text"] = row["data"][:TEXT_PEEK]
        if env.store.has_clock(row["source"]):
            found = env.store.program_time(row["source"], row["t0"])
            if found:
                out.update(run=found[0], tp=found[1])
    else:
        out.update(text=row["text"], by=row["by"])
    return out


# find

def _cursor(value):
    if not isinstance(value, str):
        raise ValueError("is a cursor, the `next` of an earlier find")
    return decode_cursor(value)


def find(env: Env, caller: Caller, body: dict) -> dict:
    p = parse("find", body, {"text": _text(), "since": _when, "until": _when, "sources": _names,
                             "scope": _choice("default", "all"), "types": _types, "as_of": _when,
                             "limit": _int(*FIND_LIMIT), "after": _cursor})
    now = time.time()
    since, until = _range(p, now)
    limit = p.get("limit", 50)
    scope = caller.scope(p.get("scope"))
    types = p.get("types", ["item", "record", "label"])
    text = p.get("text")
    if text is not None and not text.strip():
        text = None  # blank is no text
    elif text is not None and not text_words(text):
        raise ApiError("bad_request", "text has no words to search for (letters or digits)", field="text")
    after = p.get("after")
    if "as_of" in p:
        rows = sorted((item_dict(item) for item in env.store.as_of(p["as_of"], p.get("sources"), scope)
                       if item_matches(item, text_words(text)) and "item" in types), key=cursor_key, reverse=True)
        found = [row for row in rows if after is None or cursor_key(row) < after][:limit + 1]
    else:
        found = env.store.find(since, _after(until), p.get("sources"), text, limit + 1, scope, types, after)
    more = len(found) > limit
    found = found[:limit]
    return {"results": [result_row(env, row) for row in found], "next": encode_cursor(found[-1]) if more else None,
            "searched": searched(env, caller, since, until, p.get("sources"), p.get("scope", "default"))}


# look

LOOK_FIELDS = {"item": _int(1), "at": _number(atleast=0), "source": _name, "t": _when, "run": _text(),
               "tp": _number(), "offset": _int(0), "since": _when, "until": _when, "region": _region}
LOOK_MOMENTS = 8
PICTURE_FORMS = ("video", "image")  # what `reader.form` gives for what `look` can show a part of


def look(env: Env, caller: Caller, body: dict) -> dict:
    if "moments" not in body:
        return look_moment(env, body)
    if len(body) > 1:
        raise ApiError("bad_request", "look takes moments alone, or the fields of one moment", field="moments")
    moments = body["moments"]
    if (not isinstance(moments, list) or not 1 <= len(moments) <= LOOK_MOMENTS
            or not all(isinstance(moment, dict) for moment in moments)):
        raise ApiError("bad_request", f"moments is a list of 1 to {LOOK_MOMENTS} looks, each an object", field="moments")
    out = []
    for moment in moments:
        try:
            out.append(look_moment(env, moment, len(moments)))
        except ApiError as error:  # this moment failed; the others still run
            out.append({"error": error.body()})
    return {"moments": out}


def _region_refused() -> ApiError:
    return ApiError("bad_request", "region is for pictures and video; this moment is not one", field="region")


def look_moment(env: Env, body: dict, of: int = 1) -> dict:
    """One moment, or a sheet of a range: the body of a `look` without `moments`. A text page is LOOK_PAGE divided by the
    `of` moments shown. A `region` shows that part of a picture or video frame, and is refused for anything else."""
    from mms import reader
    p = parse("look", body, LOOK_FIELDS)
    if "since" in p or "until" in p:
        return look_sheet(env, p)
    region = p.get("region")
    store, scope = env.store, EVERYTHING
    if "item" in p:
        for other in ("source", "t", "run", "tp"):
            if other in p:
                raise ApiError("bad_request", "look takes an item with at, or a source with t, or a source with run "
                                              "and tp", field=other)
        item = store.item(p["item"])
        if item is None or item.state != "ok":
            raise ApiError("not_found", f"no item {p['item']}", field="item")
        at = p.get("at", 0.0)
        t = item.t0 + at if item.kind in SPANS else item.t0
        extra: dict = {}
    else:
        if "source" not in p:
            raise ApiError("bad_request", "look needs an item, or a source with t or with run and tp", field="source")
        if "run" in p or "tp" in p:
            if "run" not in p or "tp" not in p:
                raise ApiError("bad_request", "a run is looked at with its tp", field="tp" if "run" in p else "run")
            t = store.wall_time(p["source"], p["run"], p["tp"])
            if t is None:
                raise ApiError("not_found", f"source {p['source']!r} has no run {p['run']!r}", field="run")
            extra = {"run": p["run"], "tp": p["tp"]}
        elif "t" in p:
            t, extra = p["t"], {}
        else:
            raise ApiError("bad_request", "look needs t, or run and tp, with a source", field="t")
        item = _moment(env, p["source"], t, scope)
        if item is not None and item.kind == "image":
            t = item.t0  # a picture is shown at its own time, not the time it was found from
        at = None if item is None else (t - item.t0 if item.kind in SPANS else 0.0)
    source = p.get("source") or item.source
    labels = _labels(store, source, t, t)
    if item is None:  # records within 30 s of t
        rows = store.records(t - LOOK_NEAR_S, t + LOOK_NEAR_S, [source], scope=scope)
        if not rows:
            raise ApiError("not_found", f"nothing of {source!r} at {iso(t)}", field="source", hint=_nearest(store, source, t))
        if region is not None:
            raise _region_refused()
        text = "\n".join(f"{iso(record.t)} {record.data}" for record in rows)
        shown = reader.page(text, p.get("offset", 0), reader.LOOK_PAGE // of)
        out = {"kind": "text", "source": source, "item": None, "t": iso(t), "at": None, "labels": labels, **shown}
    else:
        if region is not None and reader.form(item) not in PICTURE_FORMS:
            raise _region_refused()
        folder = env.home / "looks"
        folder.mkdir(exist_ok=True)
        try:
            made = reader.look(store, item, at, folder, p.get("offset", 0), reader.LOOK_PAGE // of, region)
        except reader.TooBig as error:
            raise ApiError("too_big", str(error), field="item") from None
        except ValueError as error:
            raise ApiError("bad_request", str(error), field="at") from None
        except OSError as error:
            raise ApiError("not_found", f"item {item.id}'s file is gone: {error}", field="item") from None
        out = {"kind": made.pop("kind"), "source": item.source, "item": item.id, "t": iso(t),
               "at": made.pop("at"), "labels": labels, **made}
    out.update(extra)
    if region is not None:
        out["region"] = region._asdict()
    if "run" not in out and env.store.has_clock(source) and item is None:
        found = store.program_time(source, t)
        if found:
            out.update(run=found[0], tp=found[1])
    return out


def look_sheet(env: Env, p: dict) -> dict:
    """A source over a range as one picture: `LOOK_TILES` times spread over it, each resolved as `look` resolves a `t`,
    those that read as a picture drawn by `reader.sheet`."""
    from mms import reader
    for other in ("item", "at", "t", "run", "tp", "offset"):
        if other in p:
            raise ApiError("bad_request", "look over a range takes source, since and until, and no other field", field=other)
    for needed in ("source", "since", "until"):
        if needed not in p:
            raise ApiError("bad_request", "look over a range needs source, since and until", field=needed)
    store, source, since, until = env.store, p["source"], p["since"], p["until"]
    if until <= since:
        raise ApiError("bad_request", "until is not after since", field="until")
    picks = []
    for k in range(LOOK_TILES):
        t = since + k * (until - since) / (LOOK_TILES - 1)
        item = _moment(env, source, t, EVERYTHING, since)
        if item is not None and reader.form(item) in PICTURE_FORMS:
            picks.append(reader.Pick(k, item.t0 if item.kind == "image" else t, item,
                                     t - item.t0 if item.kind in SPANS else 0.0))
    if not picks:
        raise ApiError("not_found", f"{source!r} has no video or pictures from {iso(since)} to {iso(until)}", field="source",
                       hint=_nearest(store, source, since))
    folder = env.home / "looks"
    folder.mkdir(exist_ok=True)
    try:
        made = reader.sheet(store, picks, folder, p.get("region"))
    except OSError as error:  # every picture's file is gone
        raise ApiError("not_found", f"a file of {source!r} is gone: {error}", field="source") from None
    except reader.UNREADABLE as error:  # no picture could be read
        raise ApiError("bad_request", str(error), field="source") from None
    out = {"kind": "sheet", "source": source, "since": iso(since), "until": iso(until),
           "labels": _labels(store, source, since, until), "image": made["image"],
           "tiles": [{"n": n, **when(tile["t0"], tile["t1"]), "look": {"item": tile["item"], "at": round(tile["at"], 3)}}
                     for n, tile in enumerate(made["tiles"], 1)]}
    if "region" in p:
        out["region"] = p["region"]._asdict()
    return out


def _labels(store: Store, source: str, t0: float, t1: float) -> list[dict]:
    """Up to LABELS_SHOWN of a source's labels over a time or a span."""
    return [{"t0": iso(label.t0), "t1": iso(label.t1), "by": label.by, "text": label.text}
            for label in store.labels(t0, _after(t1), [source], scope=EVERYTHING, limit=LABELS_SHOWN)]


def _moment(env: Env, source: str, t: float, scope: Scope, since: float | None = None):
    """What a source holds at `t`: the segment that covers it, else the newest of the versions current then and its
    latest picture at or before `t` (a picture covers only its own instant, so within a sheet's range, from `since`),
    else None (records)."""
    store = env.store
    covering = store.select(t, _after(t), [source], kinds=SPANS, newest_first=True, scope=scope)
    if covering:
        return covering[0]
    held = store.as_of(t, [source], scope)
    still = store.latest_still(source, t, since, scope)
    if still is not None:
        held.append(still)
    return max(held, key=lambda item: item.t0) if held else None


def _nearest(store: Store, source: str, t: float) -> str:
    """The `hint` of a look that found nothing: the source's nearest item or record."""
    item, record = store.nearest(source, t)
    parts = []
    if item is not None:
        parts.append(f"the nearest item is #{item[0]} at {iso(item[1])}")
    if record is not None:
        parts.append(f"the nearest record is at {iso(record)}")
    return "; ".join(parts) if parts else f"{source!r} holds nothing"  # look has no scope


# remember, label and keep

def remember(env: Env, caller: Caller, body: dict) -> dict:
    p = parse("remember", body, {"note": _text(NOTE_MAX), "path": _path, "t": _when, "expires": _ahead,
                                 "name": _text(blank=False), "source": _name})
    if ("note" in p) == ("path" in p):
        raise ApiError("bad_request", "remember takes a note or a path, one of them", field="note")
    source = p.get("source", "agent")
    ingest.check_push(env.config, source)
    now = time.time()
    if "expires" in p and p["expires"] <= now:
        raise ApiError("bad_request", "expires is a time ahead, or a span such as 30d", field="expires")
    who = {"project": caller.project, "session": caller.session, "expires": p.get("expires")}
    if "note" in p:
        out = ingest.push_note(env.store, source, p["note"], p.get("name"), p.get("t", now), **who)
    else:
        path = Path(p["path"]).expanduser()
        if reads_the_home(env.home, path):
            raise ApiError("bad_request", f"{path} is, holds or is inside the MultimediaSearch home, which is not remembered",
                           field="path")
        out = ingest.push_path(env.store, env.config, source, path, p.get("name"), p.get("t"), **who)
    for entry in out["stored"] + [one for one in out["unchanged"] if "t0" in one]:
        entry["t0"] = iso(entry["t0"])
        if entry.get("t1") is not None:
            entry["t1"] = iso(entry["t1"])
    return {**out, "source": source, "expires": iso(p.get("expires"))}


def _cap(env: Env, source: str | None) -> float | None:
    """The `max_gb` of a configured source: what the kept bytes of its items may not pass (DESIGN.md §5.1)."""
    configured = env.config.sources.get(source)
    return configured.keep.max_gb if configured else None


def label(env: Env, caller: Caller, body: dict) -> dict:
    p = parse("label", body, {"text": _text(blank=False), "since": _when, "until": _when, "source": _name,
                              "keep": _flag}, required=("text", "since"))
    since, until = p["since"], p.get("until", p["since"])
    if until < since:
        raise ApiError("bad_request", "until is before since", field="until")
    keeping = p.get("keep", False)
    if keeping and "source" not in p:
        raise ApiError("bad_request", "keep names the source whose items it keeps", field="source")
    made = env.store.label(since, until, p["text"], "agent", source=p.get("source"), keep=keeping,
                           project=caller.project, session=caller.session,
                           max_gb=_cap(env, p.get("source")) if keeping else None)
    return {"label": made, "kept": env.store.count_kept(p["source"], since, until) if keeping else 0}


def keep(env: Env, caller: Caller, body: dict) -> dict:
    p = parse("keep", body, {"source": _name, "since": _when, "until": _when, "clear": _flag},
              required=("source", "since"))
    since, until = p["since"], p.get("until", time.time())
    if until < since:
        raise ApiError("bad_request", "until is before since", field="until")
    clear = p.get("clear", False)
    items = env.store.set_keep(p["source"], since, until, not clear, max_gb=_cap(env, p["source"]))
    return {"items": items, "keep": not clear}


# status

def _entry(env: Env, source: configuration.Source | None, name: str, held: dict, states: dict, bytes_: int,
           state: str | None = None) -> dict:
    live = states.get(name, {})
    what = source.what() if source else None
    unapproved = what is not None and not consent.approved(env.home, name, what)
    # a running daemon says what it applied; with none (or before it has read a new table) the consent file does
    if state is not None:
        pass
    elif "state" in live:
        state = live["state"]
    elif unapproved:
        state = "needs_approval"
    else:
        state = "idle" if source is None else None
    out = {"name": name, "kind": source.kind if source else None, "state": state,
           "project": env.store.project_name(source.project) if source else None,
           "keep": {"max_gb": source.keep.max_gb if source else None,
                    "when_full": source.keep.when_full if source else None,
                    "max_days": source.keep.max_days if source else None},
           "bytes": bytes_, "items": held.get("items", 0), "records": held.get("records", 0),
           "newest": iso(held.get("newest"))}
    if live.get("error") and state != "needs_approval":
        out["error"] = live["error"]
    if unapproved and state in ("needs_approval", "pending_reload"):
        out["approve"] = f"mms approve {name}"
    return out


def status(env: Env, caller: Caller, body: dict) -> dict:
    p = parse("status", body, {"devices": _flag})
    config, store, live = env.config, env.store, env.live
    held = store.status()
    states = live.source_states() if live else {}
    entries = [_entry(env, config.sources.get(name), name, held.get(name, {}), states,
                      held.get(name, {}).get("item_bytes", 0) + held.get(name, {}).get("record_bytes", 0)
                      + held.get(name, {}).get("clock_bytes", 0))
               for name in sorted(config.sources.keys() | held.keys())]
    floor = live.floor() if live else _disk_floor(env)
    out = {"home": str(env.home), "running": live is not None, "pid": live.pid if live else None,
           "started": iso(live.started) if live else None,
           "config_error": env.config_error, "floor": floor,
           "decider": live.decider() if live else {"model": config.decider_model, "state": None, "error": None,
                                                    "waiting": None},
           "jobs": live.job_counts() if live else {"running": None, "queued": None}, "sources": entries}
    if p.get("devices"):
        from mms.capture import devices
        out["devices"] = devices(configuration.ffmpeg_exe(config))
    return out


def _disk_floor(env: Env) -> dict:
    free = shutil.disk_usage(env.home).free / GB
    return {"breached": free < env.config.free_gb, "free_gb": free, "floor_gb": env.config.free_gb}


# source_set

OPTION_FIELDS = {"device": ("screen", "microphone", "camera"), "fps": ("screen", "camera"),
                 "path": ("folder", "log"), "glob": ("folder", "log"), "command": ("command",),
                 "time_field": ("log", "command")}
OWNED = tuple(OPTION_FIELDS)  # the options source_set can name; a table's others (segment_s, rescan_s...) are the owner's
PENDING = "pending_reload"  # a source's `state` after `source_set`, until the daemon has read the change


def source_set(env: Env, caller: Caller, body: dict) -> dict:
    """Add a source, or change the fields given of one that exists: the rest of its table is kept as it is. A cap of an
    existing source can stay or rise here, never fall, and its project is the creator's for good. The reply says
    `pending_reload` until the running daemon has applied the change, and guesses no state."""
    p = parse("source_set", body, {"name": _name, "kind": _text(), "device": _text(blank=False), "fps": _number(above=0),
                                   "path": _path, "glob": _text(blank=False), "command": _args,
                                   "time_field": _text(blank=False), "max_gb": _number(above=0),
                                   "when_full": _choice(*configuration.WHEN_FULL), "max_days": _number(above=0),
                                   "remove": _flag}, required=("name",))
    name = p["name"]
    try:
        current = configuration.load(env.home)
    except ConfigError as error:
        raise ApiError("config_error", f"mms.toml does not parse: {error}", hint="ask the owner") from None
    existing = current.sources.get(name)
    if existing is not None and existing.kind not in SET_KINDS:
        raise ApiError("wrong_kind", f"source {name!r} is a {existing.kind} source, which the owner changes by hand",
                       field="name", hint="use another name")
    if p.get("remove"):
        extra = sorted(set(p) - {"name", "remove"})
        if extra:
            raise ApiError("bad_request", "remove takes only name", field=extra[0])
        if existing is None:
            raise ApiError("not_found", f"no source {name!r}", field="name")
        configuration.remove_source(env.home, name)
        return {"source": {"name": name, "removed": True},
                "pending_reload": env.live is not None and name in env.config.sources}
    kind = p.get("kind", existing.kind if existing else None)
    needed = ("kind", "max_gb") if existing is None else ()
    for field in needed:
        if field not in p:
            raise ApiError("bad_request", f"source_set needs {field} to add a source", field=field)
    if kind not in SET_KINDS:
        raise ApiError("bad_request", f"kind is one of {', '.join(SET_KINDS)}", field="kind")
    if existing is not None and existing.kind != kind:
        raise ApiError("wrong_kind", f"source {name!r} is a {existing.kind} source, not {kind}", field="name",
                       hint="use another name")
    for option, kinds in OPTION_FIELDS.items():
        if option in p and kind not in kinds:
            raise ApiError("bad_request", f"a {kind} source takes no {option}", field=option)
    if "path" in p and reads_the_home(env.home, p["path"]):
        raise ApiError("bad_request", f"{p['path']} is, holds or is inside the MultimediaSearch home, which is not scanned",
                       field="path")
    if existing is not None:
        for field, old in (("max_gb", existing.keep.max_gb), ("max_days", existing.keep.max_days)):
            if field in p and (old is None or p[field] < old):
                raise ApiError("would_delete", f"{field} of source {name!r} is {'unset' if old is None else f'{old:g}'}; "
                                               f"source_set can keep it or raise it, not lower it", field=field,
                               hint="the owner lowers it by editing mms.toml")
    held = env.store.source_bytes().get(name, 0)
    if "max_gb" in p and p["max_gb"] * GB < held:
        raise ApiError("would_delete", f"source {name!r} holds {held:,} bytes, more than max_gb {p['max_gb']:g} "
                                       f"would keep", field="max_gb", hint="the owner can edit mms.toml")
    if existing is None and "max_days" in p and env.store.holds_before(name, time.time() - p["max_days"] * 86400):
        raise ApiError("would_delete", f"source {name!r} holds history older than {p['max_days']:g} days",
                       field="max_days", hint="the owner can edit mms.toml")
    old = existing.keep if existing else configuration.Keep()
    keeping = {"max_gb": p.get("max_gb", old.max_gb), "when_full": p.get("when_full", old.when_full),
               "max_days": p.get("max_days", old.max_days)}
    table: dict = {"kind": kind, "keep": {key: value for key, value in keeping.items() if value is not None}}
    table.update(existing.options if existing else {})
    table.update({key: p[key] for key in OWNED if key in p})
    project = existing.project if existing else (caller.project if kind in ("folder", "log", "command") else None)
    if project is not None:
        table["project"] = project  # set when the source is made, and never moved by a later edit
    try:
        configuration.set_source(env.home, name, table)
    except ConfigError as error:  # the file went bad since it was read
        raise ApiError("config_error", f"mms.toml does not parse: {error}", hint="ask the owner") from None
    except ValueError as error:
        raise ApiError("bad_request", str(error)) from None
    source = configuration.load(env.home).sources[name]
    pending = env.live is not None and env.config.sources.get(name) != source
    states = env.live.source_states() if env.live else {}
    entry = _entry(env, source, name, env.store.status().get(name, {}), states, 0, PENDING if pending else None)
    return {"source": entry, "pending_reload": pending}


OPS = {"find": find, "look": look, "remember": remember, "label": label, "keep": keep, "status": status,
       "source_set": source_set}


def call(env: Env, op: str, body) -> dict:
    """Run an op: `body` is the JSON object of the call, the op's parameters and `caller`."""
    fn = OPS.get(op)
    if fn is None:
        raise ApiError("not_found", f"no op {op!r}", field="op")
    if not isinstance(body, dict):
        raise ApiError("bad_request", "the body is a JSON object")
    body = dict(body)
    caller = parse_caller(body.pop("caller", None))
    env.store.note_caller(caller.asdict())
    return fn(env, caller, body)
