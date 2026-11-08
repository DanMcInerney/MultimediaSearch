"""The `mms` command (DESIGN.md §10, §13): start, stop, status, find, look, remember, label, keep, approve, `ask` and
`run`, which hand the rest of the line to a script, `mcp`, `install` and `model`. The store's calls run in this process (`ops`);
jobs and `status` go to the daemon through client.py. It imports only what every command needs, so `mms mcp` starts
light: the adapter must not pull in the store or the reader."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mms import client, project
from mms import config as configuration
from mms.errors import ApiError

STOP_WAIT_S = 30.0
JOB_WAIT_S = 40.0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    home, rest = None, argv
    if argv[:1] == ["--home"] and len(argv) > 1:
        home, rest = argv[1], argv[2:]
    elif argv[:1] and argv[0].startswith("--home="):
        home, rest = argv[0].split("=", 1)[1], argv[1:]
    try:
        if rest[:1] in (["ask"], ["run"]):  # the script's own arguments pass through untouched
            return run_script(configuration.default_home() if home is None else Path(home).expanduser(), rest)
        if rest[:1] == ["install"]:  # install's own arguments, and its paths come from the environment
            from mms import install
            if home is not None:
                os.environ["MMS_HOME"] = str(Path(home).expanduser())
            return install.main(rest[1:])
        if rest[:1] == ["model"]:  # model's own arguments
            from mms import prepare
            return prepare.main(configuration.default_home() if home is None else Path(home).expanduser(), rest[1:])
        args = parser().parse_args(argv)
        args.home = Path(args.home).expanduser() if args.home else configuration.default_home()
        return args.command(args) or 0
    except ApiError as error:
        print(f"mms: {error.code}: {error.message}" + (f" ({error.hint})" if error.hint else ""), file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"mms: {error}", file=sys.stderr)
        return 2


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="mms", description="MultimediaSearch: a long memory for coding agents.")
    top.add_argument("--home", help="the memory's folder (default: MMS_HOME, else ~/.mms)")
    commands = top.add_subparsers(dest="name", required=True, metavar="command")

    def add(name: str, handler, help: str, json_flag: bool = False) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help, description=help)
        sub.set_defaults(command=handler)
        if json_flag:
            sub.add_argument("--json", action="store_true", help="print JSON")
        return sub

    sub = add("start", start, "run the daemon in this terminal until `mms stop` or Ctrl+C")
    sub.add_argument("--detach", action="store_true", help="start it in the background and return once it answers")
    sub.add_argument("--detached", action="store_true", help=argparse.SUPPRESS)
    add("stop", stop, "ask the running daemon to stop, and wait for it")
    sub = add("status", status, "the daemon, the free-space floor, the model and each source", True)
    sub.add_argument("--devices", action="store_true", help="also list the capture devices")

    sub = add("find", find, "items, records and labels in a time range, newest first", True)
    sub.add_argument("--since", help="a start: 2h, 7d, an ISO time or epoch seconds")
    sub.add_argument("--until", help="an end, in the same forms")
    sub.add_argument("--source", action="append", help="only this source; repeat for several")
    sub.add_argument("--text", help="only what contains these words")
    sub.add_argument("--as-of", help="the version of each file that was current at this time")
    sub.add_argument("--type", action="append", choices=["item", "record", "label"], help="only this type")
    sub.add_argument("--scope", choices=["default", "all"], default="default")
    sub.add_argument("--limit", type=int, default=50)
    sub.add_argument("--after", help="the next page: the cursor a find that was cut by --limit printed")

    sub = add("look", look, "show one moment: a frame or picture, a sound's file, text, or a copy of a file that is none of these; "
                            "or a range as a sheet of up to 9 frames", True)
    sub.add_argument("item", nargs="?", type=int, help="an item, with --at")
    sub.add_argument("--at", type=float, help="seconds into the item")
    sub.add_argument("--source", help="a source, with --t, with --run and --tp, or with --since and --until")
    sub.add_argument("--t", help="a time")
    sub.add_argument("--since", help="the start of a range, with --source and --until: a sheet of up to 9 numbered frames")
    sub.add_argument("--until", help="the end of the range")
    sub.add_argument("--run", help="a program's run")
    sub.add_argument("--tp", type=float, help="the program's own time in that run")
    sub.add_argument("--offset", type=int, help="characters into a text")

    sub = add("remember", remember, "store a note, file, folder or recording at its real time", True)
    sub.add_argument("text", nargs="?")
    sub.add_argument("--path", help="a file, folder or recording instead of a note")
    sub.add_argument("--t", help="when it happened (default now, or a picture's, video's or sound's own time)")
    sub.add_argument("--expires", help="a time ahead, or a span from now such as 30d")
    sub.add_argument("--source", default="agent")
    sub.add_argument("--name", help="a name for the note, or for the file")

    sub = add("label", label, "mark a time range with text", True)
    sub.add_argument("text")
    sub.add_argument("--since", required=True)
    sub.add_argument("--until")
    sub.add_argument("--source")
    sub.add_argument("--keep", action="store_true", help="keep what the range overlaps, of --source, from caps")

    sub = add("keep", keep, "keep a source's items in a time range from caps and thinning, or let them go", True)
    sub.add_argument("--source", required=True)
    sub.add_argument("--since", required=True)
    sub.add_argument("--until", help="default now")
    sub.add_argument("--clear", action="store_true", help="let them go: caps count and may delete them again")

    sub = add("approve", approve, "approve a source that opens a device or runs a program (a terminal, by the owner)")
    sub.add_argument("source")
    sub.add_argument("--revoke", action="store_true", help="remove the approval; needs no terminal")

    sub = add("mcp", mcp, "serve the memory to an agent over MCP (stdio)")
    sub.add_argument("--tools", type=tool_list, help="offer only these tools, comma-separated (default: all)")
    commands.add_parser("install", help="register the memory with Claude Code and Codex, download and configure its decision "
                        "model, and start at login")
    commands.add_parser("model", help="list the decision models, switch the one mms uses, downloading it when missing (`use`), "
                        "or build its env (`prepare`)")
    commands.add_parser("ask", help="ask the local model about the record: runs the ask script with the rest of the line")
    commands.add_parser("run", help="run a script, `mms run <script> ...`: the home's copy, else the shipped one")
    return top


def tool_list(text: str) -> tuple[str, ...]:
    """`--tools`' value as tool names; the adapter is imported only when the option is given."""
    from mms import adapter
    names = tuple(text.split(","))
    for name in names:
        if name not in adapter.TOOL_NAMES:
            raise argparse.ArgumentTypeError(f"{name!r} is not a tool; the tools are {', '.join(adapter.TOOL_NAMES)}")
    return names


def mcp(args) -> int:
    from mms import adapter
    return adapter.main(args.home, args.tools)


def caller() -> dict:
    """The CLI's caller (DESIGN.md §11): no session, and the project of the folder `mms` runs in."""
    found = project.find(Path.cwd())
    return {"session": None, "project": found.id if found else None, "project_name": found.name if found else None,
            "project_root": found.root if found else None, "client": "mms"}


def op(args, name: str, body: dict) -> dict:
    """An op, run by the daemon (started if none runs) for the CLI's caller: the daemon is the one writer of the store."""
    who = caller()
    client.ensure(args.home)
    return client.call(args.home, name, body, caller=who)


def given(**fields) -> dict:
    return {key: value for key, value in fields.items() if value is not None}


# the daemon

def start(args) -> int:
    from mms.daemon import Daemon, start_detached
    if args.detach:
        return start_detached(args.home)
    return Daemon(args.home, detached=args.detached).run()


def stop(args) -> int:
    from mms.daemon import STOP
    if not client.is_running(args.home):
        print("no daemon is running")
        return 0
    (args.home / STOP).write_text("stop", encoding="utf-8")
    end = time.monotonic() + STOP_WAIT_S
    while time.monotonic() < end:
        if not client.is_running(args.home):
            print("stopped")
            return 0
        time.sleep(0.1)
    print(f"mms: the daemon has not stopped after {STOP_WAIT_S:g} s", file=sys.stderr)
    return 1


def status(args) -> int:
    body, who = given(devices=args.devices or None), {"client": "mms"}  # status needs no project
    if client.connect(args.home):
        out = client.call(args.home, "status", body, caller=who)
    else:  # no daemon runs: the store and the config say what they can
        from mms import ops
        from mms.store import Store
        out = ops.call(ops.Env(args.home, Store(args.home), configuration.load(args.home)), "status",
                       {**body, "caller": who})
    if args.json:
        print(json.dumps(out))
        return 0
    floor = out["floor"]
    print(f"home    {out['home']}")
    print("daemon  " + (f"running (pid {out['pid']})" if out["running"] else "not running"))
    if out["config_error"]:
        print(f"config  ERROR, the last good one is in use: {out['config_error']}")
    free = f"{floor['free_gb']:.1f} GB free, " if floor.get("free_gb") is not None else ""
    print("floor   " + (f"BREACHED: {free}floor {floor['floor_gb']:g} GB, writing paused" if floor["breached"]
                        else f"ok: {free}floor {floor['floor_gb']:g} GB"))
    decider = out["decider"]
    print(f"decider {decider['model'] or 'no model configured'}" + (f" ({decider['state']})" if decider["state"] else ""))
    rows = [("source", "kind", "state", "items", "records", "size", "newest", "cap")]
    for one in out["sources"]:
        keep_ = one["keep"]
        cap = "" if keep_["max_gb"] is None else f"{keep_['max_gb']:g} GB {keep_['when_full']}"
        if keep_["max_days"] is not None:
            cap += f" {keep_['max_days']:g} d"
        rows.append((one["name"], one["kind"] or "?", one["state"] or "-", str(one["items"]), str(one["records"]),
                     size(one["bytes"]), one["newest"] or "-", cap.strip() or "-"))
        if one.get("approve"):
            rows.append(("", "", f"run: {one['approve']}", "", "", "", "", ""))
    if len(rows) > 1:
        table(rows)
    if "devices" in out:
        for kind in ("displays", "microphones", "cameras"):
            print(f"{kind}: " + (", ".join(d["device"] for d in out["devices"][kind]) or "none"))
        if out["devices"].get("error"):
            print(f"devices error: {out['devices']['error']}")
    return 0


# the memory

def find(args) -> int:
    out = op(args, "find", given(since=args.since, until=args.until, sources=args.source, text=args.text,
                                 as_of=args.as_of, types=args.type, scope=args.scope, limit=args.limit,
                                 after=args.after))
    if args.json:
        print(json.dumps(out))
        return 0
    for one in out["results"]:
        print(f"{one['t0']}  {one['type']:<6} #{one['id']:<6} {one['source'] or '-':<12} {summary(one)}")
    if out["next"]:
        print(f"more: repeat with --after {out['next']}")
    return 0


def summary(one: dict) -> str:
    if one["type"] == "item":
        what = one["name"] or ""
        return f"{one['kind']} {what} {size(one['bytes'])}".replace("  ", " ") + (" keep" if one["keep"] else "")
    if one["type"] == "record":
        return one["text"][:120]
    return f"{one['by']}: {one['text'][:120]}"


def look(args) -> int:
    out = op(args, "look", given(item=args.item, at=args.at, source=args.source, t=args.t, since=args.since,
                                 until=args.until, run=args.run, tp=args.tp, offset=args.offset))
    if args.json:
        print(json.dumps(out))
        return 0
    if out["kind"] in ("image", "sheet"):
        print(out["image"]["path"])
    elif out["kind"] == "audio":
        print(out["audio"]["path"])
    elif out["kind"] == "file":
        copy = out["file"]
        print(copy["path"])
        print(f"a copy of {copy['name'] or 'the file'}, {copy['media']}, {size(copy['bytes'])}", file=sys.stderr)
    else:
        print(out["text"])
        if out["more"]:
            print(f"more: repeat with --offset {out['next_offset']}", file=sys.stderr)
    return 0


def remember(args) -> int:
    if (args.text is None) == (args.path is None):
        raise ValueError("remember takes a note or --path, one of them")
    out = op(args, "remember", given(note=args.text, path=str(Path(args.path).resolve()) if args.path else None,
                                     t=args.t, expires=args.expires, source=args.source, name=args.name))
    if args.json:
        print(json.dumps(out))
        return 0
    for one in out["stored"]:
        print(f"{one['type']} {one['id']}")
    for one in out["unchanged"]:
        print(f"unchanged: {one['name']} is already stored (item {one['id']})")
    return 0


def label(args) -> int:
    out = op(args, "label", given(text=args.text, since=args.since, until=args.until, source=args.source,
                                  keep=True if args.keep else None))
    print(json.dumps(out) if args.json else f"label {out['label']}" + (f", kept {out['kept']} items" if args.keep else ""))
    return 0


def keep(args) -> int:
    out = op(args, "keep", given(source=args.source, since=args.since, until=args.until, clear=True if args.clear else None))
    print(json.dumps(out) if args.json else f"{'let go' if args.clear else 'kept'} {out['items']} items")
    return 0


def approve(args) -> int:
    from mms import consent
    source = configuration.load(args.home).sources.get(args.source)
    if source is None:
        raise ValueError(f"no source {args.source!r} in {args.home / configuration.FILE}")
    what = source.what()
    if what is None:
        raise ValueError(f"a {source.kind} source needs no approval")
    if args.revoke:
        print("approval removed" if consent.revoke(args.home, args.source) else f"{args.source} had no approval")
        return 0
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError("approve needs a terminal: the owner runs it, and types the source's name")
    print(f"{args.source} will open or run: {what}")
    if input(f"type {args.source} to approve it: ").strip() != args.source:
        raise ValueError("not approved: that was not the source's name")
    consent.record(args.home, args.source, what)
    print(f"approved {args.source}")
    return 0


# scripts

def run_script(home: Path, rest: list[str]) -> int:
    """`mms ask ...` and `mms run <script> ...`: the script's `cli(argv)` makes the params, the daemon runs the job
    (started if none runs), and the result is printed as JSON."""
    from mms import jobs
    if rest[0] == "ask":
        name, passed = "ask", rest[1:]
    elif len(rest) < 2:
        raise ValueError("run takes a script name: mms run <script> [arguments]")
    else:
        name, passed = rest[1].removesuffix(".py"), rest[2:]
    module = jobs.load_script(home, name)
    if not callable(getattr(module, "cli", None)):
        raise ValueError(f"script {name!r} has no cli(argv), so it cannot be run from the command line")
    try:
        params = module.cli(passed)
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    who = caller()
    client.ensure(home)
    job = client.call(home, "jobs/start", {"script": name, "params": params, "wait_s": JOB_WAIT_S}, caller=who,
                      timeout_s=JOB_WAIT_S + 20)
    while job["state"] in ("queued", "running"):
        job = client.call(home, "jobs/get", {"job": job["job"], "wait_s": JOB_WAIT_S}, caller=who,
                          timeout_s=JOB_WAIT_S + 20)
    if job["state"] == "failed":
        error = job["error"]
        print(f"mms: {error['code']}: {error['message']}", file=sys.stderr)
        return 2 if error["code"] == "bad_request" else 1
    print(json.dumps(job["result"], indent=2, ensure_ascii=False))
    note = getattr(module, "note", None)
    if callable(note):
        print(note(job["result"]), file=sys.stderr)
    return 0


# printing

def size(n: int | None) -> str:
    n = n or 0
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def table(rows: list[tuple]) -> None:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


if __name__ == "__main__":
    sys.exit(main())
