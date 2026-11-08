"""The owner's `mms.toml` as a `Config` (DESIGN.md §4): every source's kind and options checked once, here, the
writer the daemon edits that file with, and the times the API and scripts take and give (`parse_when`, `iso`, `when`)."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import tomlkit

KINDS = ("screen", "microphone", "camera", "ffmpeg", "folder", "log", "command", "program", "push")
SET_KINDS = ("screen", "microphone", "camera", "folder", "log", "command", "program")  # what `source_set` may write
WHEN_FULL = ("thin", "drop_oldest")
PRESETS = ("screen", "camera", "audio")
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # a source's name is also its folder's
PROJECT_ID = re.compile(r"^(id|git|remote):[A-Za-z0-9_-]{4,64}$")
SECTIONS = ("memory", "decider", "sources")
FILE = "mms.toml"


class ConfigError(ValueError):
    """`mms.toml` does not parse, or holds something this version refuses."""


@dataclass(frozen=True)
class Keep:
    max_gb: float | None = None  # None: no cap
    when_full: str = "drop_oldest"
    max_days: float | None = None


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    keep: Keep
    options: dict  # every other key of the source's table, as written and checked
    project: str | None = None  # the project its rows belong to; None is ambient

    def get(self, key: str, default=None):
        return self.options.get(key, default)

    def device(self) -> str:
        """The device a screen, microphone or camera opens: as given, else the display (Windows output 0, else
        $DISPLAY)."""
        given = self.options.get("device")
        if given is not None or self.kind != "screen":
            return given
        return "0" if os.name == "nt" else os.environ.get("DISPLAY", ":0")

    def what(self) -> str | None:
        """What approval binds the source's name to (DESIGN.md §4.3): what it opens or runs; None when it needs none."""
        if self.kind in ("screen", "microphone", "camera"):
            return f"{self.kind}: {self.device()}"
        if self.kind == "ffmpeg":  # the argument list as JSON, so two lists never read alike
            return "ffmpeg: " + json.dumps(self.options["input"])
        if self.kind == "command":
            return "command: " + json.dumps(self.options["command"])
        return None


@dataclass(frozen=True)
class Config:
    home: Path
    free_gb: float = 20.0
    jobs: int = 4
    remember_max_mb: float = 2000.0
    ffmpeg: str | None = None
    decider_model: str | None = None
    decider_idle_s: float = 600.0
    decider_url: str | None = None  # where the owner serves the model, for a model that is served (loopback only)
    decider_envs: str | None = None  # where the model envs are; None: envs/ beside the model folder
    job_max_s: float = 3600.0
    loop_s: float = 0.25  # the daemon's loop
    janitor_s: float = 30.0  # between the janitor's passes
    sources: dict[str, Source] = field(default_factory=dict)


def reads_the_home(home: Path | str, path: Path | str) -> bool:
    """Whether reading `path` would read the MultimediaSearch home: it is the home, holds it (a folder source at `~`), or is
    inside it. Nothing the owner or an agent names may be scanned or remembered from there."""
    home, path = Path(home).expanduser().resolve(), Path(path).expanduser().resolve()
    return path == home or path in home.parents or home in path.parents


def default_home() -> Path:
    return Path(os.environ.get("MMS_HOME") or Path.home() / ".mms").expanduser()


# checking the file

def _number(label: str, value, *, above: float | None = None, atleast: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is a number, not {value!r}")
    if above is not None and not value > above:
        raise ValueError(f"{label} is a number above {above:g}, not {value!r}")
    if atleast is not None and not value >= atleast:
        raise ValueError(f"{label} is a number of at least {atleast:g}, not {value!r}")
    return float(value)


def _text(label: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is text, not {value!r}")
    return value


def _folder(label: str, value) -> str:
    """A path of the file that must be absolute (or start with `~`): a relative one would mean the cwd of whoever reads it,
    and the file's other paths are never resolved against the home either."""
    if not Path(_text(label, value)).expanduser().is_absolute():
        raise ValueError(f"{label} is an absolute path (or starts with ~), not {value!r}")
    return value


def _args(label: str, value) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(arg, str) for arg in value):
        raise ValueError(f"{label} is a list of arguments, such as [\"program\", \"arg\"]")
    return value


def _even_width(label: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2:
        raise ValueError(f"{label} is an even number of pixels, not {value!r}")
    return value


def _one_of(*choices):
    def check(label: str, value):
        if value not in choices:
            raise ValueError(f"{label} is one of {', '.join(choices)}, not {value!r}")
        return value
    return check


def _list_of_text(label: str, value) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(arg, str) for arg in value):
        raise ValueError(f"{label} is a list of ffmpeg arguments")
    return value


def _above_zero(label, value):
    return _number(label, value, above=0)


def _at_least_zero(label, value):
    return _number(label, value, atleast=0)


# the options each kind takes, and the ones it cannot do without
OPTIONS = {
    "screen": {"device": _text, "fps": _above_zero, "max_width": _even_width, "segment_s": _above_zero},
    "microphone": {"device": _text, "segment_s": _above_zero},
    "camera": {"device": _text, "fps": _above_zero, "segment_s": _above_zero},
    "ffmpeg": {"input": _args, "preset": _one_of(*PRESETS, "none"), "encode": _list_of_text,
               "media": _one_of("video", "audio"), "segment_s": _above_zero, "max_width": _even_width},
    "folder": {"path": _text, "glob": _text, "rescan_s": _above_zero, "copy_max_mb": _at_least_zero},
    "log": {"path": _text, "glob": _text, "rescan_s": _above_zero, "time_field": _text},
    "command": {"command": _args, "time_field": _text},
    "program": {"segment_s": _above_zero},
    "push": {},
}
REQUIRED = {"microphone": ("device",), "camera": ("device",), "ffmpeg": ("input",), "folder": ("path",),
            "log": ("path",), "command": ("command",)}


def _source(name: str, raw) -> Source:
    if not NAME.match(name):
        raise ValueError(f"source {name!r}: a name is letters, digits, '_' or '-', up to 64 characters")
    if not isinstance(raw, dict):
        raise ValueError(f"source {name!r}: is a table, [sources.{name}]")
    table = dict(raw)
    kind = table.pop("kind", None)
    if kind not in KINDS:
        raise ValueError(f"source {name!r}: kind is one of {', '.join(KINDS)}, not {kind!r}")
    keep_raw = table.pop("keep", {}) or {}
    if not isinstance(keep_raw, dict) or set(keep_raw) - {"max_gb", "when_full", "max_days"}:
        raise ValueError(f"source {name!r}: keep is {{ max_gb, when_full, max_days }}, not {keep_raw!r}")
    keep = Keep(max_gb=_keep_number(name, "max_gb", keep_raw.get("max_gb")),
                when_full=keep_raw.get("when_full", "drop_oldest"),
                max_days=_keep_number(name, "max_days", keep_raw.get("max_days")))
    if keep.when_full not in WHEN_FULL:
        raise ValueError(f"source {name!r}: when_full is one of {', '.join(WHEN_FULL)}, not {keep.when_full!r}")
    project = table.pop("project", None)
    if project is not None and not (isinstance(project, str) and PROJECT_ID.match(project)):
        raise ValueError(f"source {name!r}: project is an ID such as git:0123456789abcdef, not {project!r}")
    allowed = OPTIONS[kind]
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ValueError(f"source {name!r}: a {kind} source takes {', '.join(sorted(allowed)) or 'no options'}, "
                         f"not {unknown}")
    for key in REQUIRED.get(kind, ()):
        if key not in table:
            raise ValueError(f"source {name!r}: a {kind} source needs {key}")
    if kind == "ffmpeg" and table.get("preset", "none") == "none" and "encode" not in table:
        raise ValueError(f"source {name!r}: an ffmpeg source needs a preset ({', '.join(PRESETS)}) or an encode list")
    options = {key: allowed[key](f"source {name!r}: {key}", value) for key, value in table.items()}
    return Source(name, kind, keep, options, project)


def _keep_number(source: str, key: str, value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"source {source!r}: keep.{key} is a number of at least 0, not {value!r}")
    return float(value)


MEMORY = {"free_gb": ("free_gb", lambda v: _number("free_gb", v, atleast=0)),
          "jobs": ("jobs", lambda v: int(_number("jobs", v, atleast=1))),
          "remember_max_mb": ("remember_max_mb", lambda v: _number("remember_max_mb", v, above=0)),
          "ffmpeg": ("ffmpeg", lambda v: _text("ffmpeg", v)),
          "job_max_s": ("job_max_s", lambda v: _number("job_max_s", v, above=0)),
          "loop_s": ("loop_s", lambda v: _number("loop_s", v, above=0)),
          "janitor_s": ("janitor_s", lambda v: _number("janitor_s", v, above=0))}
LOOPBACK_HOSTS = {"127.0.0.1": "127.0.0.1", "localhost": "127.0.0.1", "::1": "::1"}  # host -> where mms connects


def loopback(url: str) -> tuple[str, int]:
    """(the host to connect to, the port) of a `[decider] url`: `http://` and one of 127.0.0.1, localhost or [::1], with
    a port, no credentials and no path, query or fragment. Anything else is a `ValueError` that says what is taken."""
    taken = ("a loopback address with a port, such as http://127.0.0.1:8741: http, host 127.0.0.1, localhost or [::1], "
             "no credentials, path, query or fragment")
    if not isinstance(url, str) or not url or "?" in url or "#" in url or url != url.strip():
        raise ValueError(f"[decider] url is {taken}, not {url!r}")
    try:
        parts = urlsplit(url)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise ValueError(f"[decider] url is {taken}, not {url!r}") from None
    if (parts.scheme != "http" or host not in LOOPBACK_HOSTS or port is None or not 0 < port < 65536
            or parts.username is not None or parts.password is not None or parts.path or "@" in parts.netloc):
        raise ValueError(f"[decider] url is {taken}, not {url!r}")
    return LOOPBACK_HOSTS[host], port


def _url(label: str, value) -> str:
    loopback(_text(label, value))
    return value


DECIDER = {"model": ("decider_model", lambda v: _text("[decider] model", v)),
           "idle_s": ("decider_idle_s", lambda v: _number("[decider] idle_s", v, atleast=0)),
           "url": ("decider_url", lambda v: _url("[decider] url", v)),
           "envs": ("decider_envs", lambda v: _folder("[decider] envs", v))}


def load_text(home: Path | str, text: str) -> Config:
    """The `Config` for the text of a `mms.toml`; `ConfigError` for anything that does not fit."""
    home = Path(home)
    try:
        raw = tomllib.loads(text)
        unknown = sorted(set(raw) - set(SECTIONS))
        if unknown:
            raise ValueError(f"the file has [memory], [decider] and [sources.<name>], not {unknown}")
        values: dict = {}
        for section, known in (("memory", MEMORY), ("decider", DECIDER)):
            table = raw.get(section, {})
            bad = sorted(set(table) - set(known))
            if bad:
                raise ValueError(f"[{section}] takes {', '.join(known)}, not {bad}")
            for key, value in table.items():
                attribute, check = known[key]
                values[attribute] = check(value)
        sources = {name: _source(name, table) for name, table in raw.get("sources", {}).items()}
    except (tomllib.TOMLDecodeError, ValueError) as error:
        raise ConfigError(str(error)) from None
    return Config(home=home, sources=sources, **values)


def load(home: Path | str | None = None) -> Config:
    home = Path(home).expanduser() if home else default_home()
    file = home / FILE
    return load_text(home, file.read_text(encoding="utf-8") if file.exists() else "")


def ffmpeg_exe(config: Config | None = None) -> str:
    """The configured ffmpeg, else ffmpeg on PATH, else the binary imageio-ffmpeg ships."""
    if config and config.ffmpeg:
        return config.ffmpeg
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# the writer (DESIGN.md §4.4)

_writing = threading.Lock()  # the daemon's own edits of the file take turns; the owner's are not locked out


def _edit(home: Path | str, change) -> object:
    """Apply `change(doc)` to the file with tomlkit, so comments and order stay. The old file must parse and the new
    text must load, or nothing is written: a `ConfigError` for the old file, a `ValueError` for the new."""
    home = Path(home)
    file = home / FILE
    with _writing:
        text = file.read_text(encoding="utf-8") if file.exists() else ""
        load_text(home, text)
        doc = tomlkit.parse(text)
        result = change(doc)
        new = tomlkit.dumps(doc)
        try:
            load_text(home, new)
        except ConfigError as error:
            raise ValueError(str(error)) from None
        if new != text:
            part = file.with_name(FILE + ".part")
            part.write_text(new, encoding="utf-8")
            os.replace(part, file)
        return result


def _table(table: dict):
    out = tomlkit.table()
    for key, value in table.items():
        if key == "keep":
            keep = tomlkit.inline_table()
            keep.update(value)
            out[key] = keep
        else:
            out[key] = value
    return out


def set_source(home: Path | str, name: str, table: dict) -> None:
    """Make `[sources.<name>]` the table given (kind, keep, project and the options), keeping the rest of the file."""
    def change(doc):
        sources = doc.get("sources")
        if sources is None:
            sources = tomlkit.table(is_super_table=True)
            doc["sources"] = sources
        existing = sources.get(name)
        if existing is None:
            sources[name] = _table(table)
            return
        for key in [key for key in existing if key not in table]:
            del existing[key]
        for key, value in _table(table).items():
            existing[key] = value
    _edit(home, change)


def remove_source(home: Path | str, name: str) -> bool:
    """Delete `[sources.<name>]`; False when there was none."""
    def change(doc):
        sources = doc.get("sources")
        if sources is None or name not in sources:
            return False
        del sources[name]
        return True
    return _edit(home, change)


def set_decider(home: Path | str, *, model: str, url: str | None) -> None:
    """Make `[decider] model` the folder given, and `url` the one given or, for None, none; the rest of the file stays."""
    def change(doc):
        table = doc.get("decider")
        if table is None:
            table = tomlkit.table()
            doc["decider"] = table
        table["model"] = model
        if url is not None:
            table["url"] = url
        elif "url" in table:
            del table["url"]
    _edit(home, change)


def ensure_program(home: Path | str, name: str, keep: dict) -> str:
    """A program stream's table, added with `keep` unless a table of that name exists. Returns the table's kind."""
    def change(doc):
        sources = doc.get("sources")
        if sources is not None and name in sources:
            return sources[name].get("kind")
        if sources is None:
            sources = tomlkit.table(is_super_table=True)
            doc["sources"] = sources
        sources[name] = _table({"kind": "program", "keep": keep})
        return "program"
    return _edit(home, change)


# times

AGO = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h|d|w)$")
UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_when(text: str | float | int | None, now: float | None = None) -> float | None:
    """UTC epoch seconds from "now", an age ("90s", "30m", "2h", "7d", "1w" ago), epoch seconds, or ISO 8601 (a time
    with no zone is local)."""
    if text is None:
        return None
    now = time.time() if now is None else now
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    if not isinstance(text, str):
        raise ValueError(f"a time is a string or a number, not {text!r}")
    text = text.strip()
    if text == "now":
        return now
    ago = AGO.match(text)
    if ago:
        return now - float(ago.group(1)) * UNIT_S[ago.group(2)]
    try:
        return float(text)
    except ValueError:
        pass
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"a time is 'now', an age such as 2h or 7d, epoch seconds, or ISO 8601, not {text!r}") from None
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.timestamp()


def parse_ahead(text: str | float | int, now: float | None = None) -> float:
    """A time ahead: a span from now ("30d", "2h") or any time `parse_when` takes."""
    now = time.time() if now is None else now
    if isinstance(text, str):
        span = AGO.match(text.strip())
        if span:
            return now + float(span.group(1)) * UNIT_S[span.group(2)]
    return parse_when(text, now)


def iso(t: float | None) -> str | None:
    """A time as ISO 8601 in the host's zone with its offset, to the millisecond: 2026-10-31T09:14:02.350-04:00."""
    if t is None:
        return None
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def when(t0: float, t1: float) -> dict:
    """A moment is `t`; a span is `t0` and `t1` (the same instant, to the millisecond `iso` writes, is a moment)."""
    start, end = iso(t0), iso(t1)
    return {"t": start} if start == end else {"t0": start, "t1": end}


def stamp(t: float) -> str:
    """A UTC time for a folder or file name: 20261030T210700Z."""
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
