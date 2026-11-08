"""Codex asked a question over `mms mcp` (TESTBED.md, Codex recall): one `codex exec` for one prompt, locked down so that
the only thing it can reach is mms's tools.

`ask_codex` returns `{answer, final, tools, events, seconds, turns, tokens, cost_usd, used_percent, model, thread,
problem, argv, stderr}`: the `ANSWER:` line of the last message (None without one), what Codex called and got back, how
long it took, the turns and tokens it used, the plan usage that exec's own rollout file reports, and `problem` (None, or
`escaped`, `timeout`, `limit`, `auth` or `error` with the reason). It raises `CodexMissing` before any run when nothing
can run: no `codex` command, a model or effort it does not list, or personal instructions (`AGENTS.md`) in the Codex
home, whose existence and size are all that is looked at.

How the exec is made:
- `codex exec --json --skip-git-repo-check --ignore-user-config --ignore-rules --strict-config -m MODEL -C <empty folder>
  -- -`, the prompt on stdin (a `.cmd` shim cuts a command line at 8,000 characters and reads `[&|<>^%"\\r\\n]` as syntax,
  so an argument holding one is refused); every setting is a `-c`, nothing is written to `~/.codex/config.toml`;
- `--disable` for the shell, the browser, computer use, image generation, apps, plugins, multi-agent and the other
  features that reach files or the web; the instruction messages Codex adds are switched off;
- `model_catalog_json`: `codex debug models` with the model's entry patched to `tool_mode` `direct` (the gpt-6 models
  are code-mode only, which reaches MCP tools only through a script tool), no apply_patch, no extra tools;
- `model_instructions_file`: one line, then (guidance `default`) mms's server instructions, the skill and the notes of the
  bench's model (d1), which is what `mms install` gives Codex and what the skill tells it to read; skills are off in the
  run so an installed copy is never read. Guidance `none` is the line
  alone, so Codex has mms's tool schemas only (as the benchmark has);
- the one MCP server `mms`: this interpreter running `mms.cli --home HOME mcp --tools ask,find,look,label`, required,
  its calls pre-approved. `--tools` leaves out `remember` and `source_set`, which would copy any file the user can read
  into the run's home; a command line without it is refused;
- every event is checked as it arrives: an item other than a message, a thought, Codex's plan list or a call of `mms`
  stops the exec as `escaped`.
The files arm (`arm="files"`, the benchmark's baseline without mms) differs in four ways: no `mms` server; the shell and
`view_image` stay on, in a `workspace-write` sandbox with no network, whose cwd is the exec's working folder; the base
instructions are one line; and the only items allowed are messages, thoughts, the plan list and shell commands, none of
which may hold a `deny` string (a path or word that leads to the answers), in the command or in its output. Its shell
gets `shell_environment_policy` settings: only Codex's core variables, a PATH of `bin_dir` and the Windows folders (so no
`mms`, no `.venv`), and a MMS_HOME that does not exist. `cost_usd` is None: Codex reports none.
The blind arm (`arm="blind"`, what the model knows with no recordings) is the read-only settings with every feature disabled and
no server; its base instructions are one line, and any item but a message, a thought, an error or the plan list is `escaped`.
Problems, by what the benchmark does with them: `escaped`, `timeout` and `capped` (the context window ran out) count as
a wrong answer; `setup` (the exec was not the one asked for: a required MCP server or the configuration failed), `limit`
and `auth` stop the agent; `error` is run again.
`~/.codex/auth.json` is never read, copied or moved: Codex reads its own login.

    python benchmarks/codex_run.py --dry-run --home HOME --models models.json   # print the command, run nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "src" / "mms" / "skills" / "mms-questions" / "SKILL.md"
NOTES_DIR = SKILL.parent / "models"
NOTES = NOTES_DIR / "d1.md"  # the default notes: those of d1-Omni-2B, which the skill points to
SERVER = "mms"
BASE = "You answer the user's question from their memory, using the mms tools."
BASE_FILES = "You answer the user's question from their files, using your shell and view_image."
BASE_BLIND = "You answer the user's question from what you know; you have no tools."
ARMS = ("mms", "files", "blind")
NO_HOME = "no-mms-home"  # a folder in the working folder that is never made: the files arm shell MMS_HOME
TIMEOUT_S = 900.0
DISABLED = ("shell_tool", "unified_exec", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
            "computer_use", "image_generation", "apps", "plugins", "multi_agent", "multi_agent_v2", "view_image",
            "skill_search", "tool_suggest", "goals", "hooks", "workspace_dependencies", "remote_plugin", "sleep_tool")
CONFIG = ("sandbox_mode=read-only", "approval_policy=never", "web_search=disabled",
          "skills.include_instructions=false", "include_environment_context=false",
          "include_permissions_instructions=false", "include_collaboration_mode_instructions=false",
          "include_apps_instructions=false")
# The files arm: a `workspace-write` shell with no network, which is told what its shell is and where it may write
FILES_SWAP = {"sandbox_mode=read-only": "sandbox_mode=workspace-write",
              "include_environment_context=false": "include_environment_context=true",
              "include_permissions_instructions=false": "include_permissions_instructions=true"}
# On Windows the shell runs only in Codex's own sandbox: without `windows.sandbox` every command is "blocked by policy".
# "unelevated" needs no admin setup; it refuses writes to the home folder but lets the shell write TEMP, where the data
# is, so the data files are read-only and checked after each exec (bench.freeze, bench.changed).
CONFIG_FILES = (*(FILES_SWAP.get(c, c) for c in CONFIG), "sandbox_workspace_write.network_access=false",
                "windows.sandbox=unelevated", "shell_environment_policy.inherit=core",
                "shell_environment_policy.exclude=['PATH']")
DISABLED_FILES = tuple(f for f in DISABLED if f not in ("shell_tool", "unified_exec", "view_image"))
SERVER_CONFIG = ("required=true", "default_tools_approval_mode=approve", "startup_timeout_sec=60", "tool_timeout_sec=900")
CATALOG_PATCH = {"apply_patch_tool_type": None, "experimental_supported_tools": [], "multi_agent_version": None,
                 "tool_mode": "direct", "supports_search_tool": False}
TOOLS = "ask,find,look,label"  # the only mms tools Codex is offered: none that reads the disk
WEEK = 10080  # window_minutes of the weekly limit
ITEMS = ("agent_message", "reasoning", "error", "todo_list", "mcp_tool_call")  # what the events of an exec may hold
ITEMS_FILES = ("agent_message", "reasoning", "error", "todo_list", "command_execution")  # and of the files arm's
ITEMS_BLIND = ("agent_message", "reasoning", "error", "todo_list")  # and of the blind arm's: no tool at all
HELPERS = {"codex": ("list_mcp_resources", "list_mcp_resource_templates")}  # Codex's own listers: no resources exist
PERSONAL = ("AGENTS.md", "AGENTS.override.md")  # what Codex reads from its home into every exec's messages
UNSAFE = re.compile(r'[&|<>^%"\r\n]')  # what cmd.exe reads as syntax, in an argument given to a .cmd shim
CMD_LIMIT = 8000
SHELL_OUTPUT = 3000  # characters of a shell command's output kept in tools.json
LIMIT = re.compile(r"usage limit|usage_limit|rate limit|rate_limit|quota|usage not included|\b429\b", re.I)
AUTH = re.compile(r"not logged in|not signed in|codex login|log in again|unauthori[sz]ed|\b401\b|credentials|"
                  r"refresh token|sign in", re.I)
# Codex's own wordings, read from codex.exe's text: a required MCP server that did not start, a configuration it refused,
# and the context window running out ("Codex ran out of room in the model's context window...")
SETUP = re.compile(r"required MCP server|MCP client for .{0,80}(failed to start|timed out)|Error loading config", re.I)
CAPPED = re.compile(r"context window|context_length", re.I)
_MODELS: dict[tuple, dict] = {}  # `codex debug models`, read once a process


class CodexMissing(RuntimeError):
    """Nothing can run: no usable `codex`, an unlisted model or effort, or personal instructions in the Codex home."""


def model() -> str:
    return os.environ.get("MMS_CODEX_MODEL", "gpt-6.1-terra")


def effort() -> str:
    return os.environ.get("MMS_CODEX_EFFORT", "high")


def literal(text: str) -> str:
    """`text` as a TOML literal string: no escapes to get wrong in an argument."""
    if "'" in text or "\n" in text:
        raise CodexMissing(f"cannot pass {text!r} to codex: it holds a quote or a newline")
    return f"'{text}'"


def toml(value) -> str:
    """`value` as a TOML literal string with forward slashes."""
    return literal(str(value).replace("\\", "/"))


def system_path() -> list[str]:
    """The folders of the operating system itself: on Windows System32, the Windows folder, Wbem and Windows PowerShell."""
    if os.name != "nt":
        return ["/usr/bin", "/bin"]
    root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or "C:\\Windows"
    return [f"{root}\\System32", root, f"{root}\\System32\\Wbem", f"{root}\\System32\\WindowsPowerShell\\v1.0"]


def shell_path(bin_dir: Path | None, extra: tuple[str, ...] = ()) -> str:
    """The PATH of the files arm's shell: `bin_dir` (ffmpeg), `extra` and the system folders, and nothing of the owner's
    (so no `mms` and no `.venv`)."""
    return os.pathsep.join([*([str(bin_dir)] if bin_dir is not None else []), *extra, *system_path()])


def codex_command() -> list[str]:
    found = shutil.which("codex")
    if not found:
        raise CodexMissing("the codex command is not on PATH: install the Codex CLI and run `codex login`")
    return [found]


def codex_home() -> Path:
    return Path(os.environ["CODEX_HOME"]) if os.environ.get("CODEX_HOME") else Path.home() / ".codex"


def personal_files() -> list[Path]:
    """The non-empty personal instruction files of the Codex home; only their existence and size are looked at."""
    return [p for p in (codex_home() / name for name in PERSONAL) if p.is_file() and p.stat().st_size > 0]


def check_personal() -> None:
    if found := personal_files():
        raise CodexMissing(f"{found[0]} holds your personal Codex instructions, which Codex adds to every exec as a "
                           "message: move it aside for the run (and put it back after)")


def models_of(command: list[str]) -> dict:
    """`codex debug models`, parsed (once a process). Never run by a dry run, which is given a saved file."""
    key = tuple(command)
    if key not in _MODELS:
        done = subprocess.run([*command, "debug", "models"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, timeout=60)
        if done.returncode:
            raise CodexMissing(f"`codex debug models` exited {done.returncode}: {done.stderr.strip()[-300:]}")
        _MODELS[key] = json.loads(done.stdout)
    return _MODELS[key]


def catalog(models: dict, name: str, level: str) -> dict:
    """The model catalog with `name`'s entry patched for direct tool mode; CodexMissing for a model or effort it lacks."""
    entries = [dict(m) for m in models.get("models", [])]
    slugs = [m.get("slug") for m in entries]
    if name not in slugs:
        raise CodexMissing(f"codex does not list the model {name!r}; it lists {', '.join(map(str, slugs))}")
    for entry in entries:
        if entry["slug"] == name:
            listed = [x.get("effort") if isinstance(x, dict) else x for x in entry.get("supported_reasoning_levels") or ()]
            if listed and level not in listed:
                raise CodexMissing(f"codex lists {name!r} at the efforts {', '.join(map(str, listed))}, not {level!r}")
            entry.update(CATALOG_PATCH)
    return {"models": entries}


def notes_text(notes: str) -> str:
    """The text of a notes file of the skill: `About.notes`, a plain file name in its models folder."""
    if notes in (".", "..") or "/" in notes or "\\" in notes:
        raise ValueError(f"notes is a plain file name in the skill's models folder, not {notes!r}")
    return (NOTES_DIR / notes).read_text(encoding="utf-8")


def instructions(guidance: str, notes: str = NOTES.name, arm: str = "mms") -> str:
    """The base instructions file. The mms arm: `default` is one line, mms's server instructions, the skill and the notes
    file `notes` ("" for none); `none` the line alone. The files and blind arms are their one line, whatever the guidance."""
    if arm == "files":
        return BASE_FILES
    if arm == "blind":
        return BASE_BLIND
    if arm != "mms":
        raise ValueError(f"arm is one of {ARMS}, not {arm!r}")
    if guidance == "none":
        return BASE
    if guidance != "default":
        raise ValueError(f"guidance is 'default' or 'none', not {guidance!r}")
    from mms.adapter import INSTRUCTIONS
    parts = [BASE, INSTRUCTIONS, SKILL.read_text(encoding="utf-8"), *([notes_text(notes)] if notes else [])]
    return "\n\n".join(parts)


def prompt(question: str, day: float, now: float | None = None) -> str:
    """What Codex is asked: the time it is now, the day mms recorded, the question and how to answer."""
    now_ = datetime.fromtimestamp(now if now is not None else time.time())
    then = datetime.fromtimestamp(day)
    return (f"It is now {now_:%A %d %B %Y, %H:%M}. mms recorded my screen, phone, cameras, files and logs yesterday "
            f"({then:%A %d %B %Y}). {question} Answer from what mms holds. End with one line `ANSWER: <answer>`, or "
            f"`ANSWER: unknown` when mms does not hold it.")


def argv_of(command: list[str], name: str, level: str, run_dir: Path, home: Path | None, tools: str = TOOLS, *,
            arm: str = "mms", work: Path | None = None, bin_dir: Path | None = None) -> list[str]:
    """The command line of an exec (the prompt is on stdin). `work` is its working folder (default `run_dir/empty`). The
    mms arm is refused without `tools`: mms would offer `remember`. With `work` the server is told its project by
    `CLAUDE_PROJECT_DIR`, which the adapter reads first, so the project does not hang on the cwd Codex gives it. The
    files arm has no server and no `home`; its shell gets Codex's core variables, a PATH of `bin_dir` and the system
    folders only, and a MMS_HOME that does not exist (the codex process itself keeps its own environment). The blind
    arm has no server, no `home` and no tool: the read-only settings and every feature disabled."""
    if arm not in ARMS:
        raise ValueError(f"arm is one of {ARMS}, not {arm!r}")
    settings = [f"model_reasoning_effort={level}", f"model_catalog_json={toml(run_dir / 'catalog.json')}",
                f"model_instructions_file={toml(run_dir / 'instructions.md')}"]
    if arm == "files":
        settings = [*CONFIG_FILES, *settings, f"shell_environment_policy.set.PATH={literal(shell_path(bin_dir))}",
                    f"shell_environment_policy.set.MMS_HOME={toml((work or run_dir / 'empty') / NO_HOME)}"]
        disabled = DISABLED_FILES
    elif arm == "blind":
        settings = [*CONFIG, *settings]
        disabled = DISABLED
    else:
        if not tools:
            raise CodexMissing("mms mcp must be started with --tools: without it Codex is offered remember and source_set")
        if home is None:
            raise CodexMissing("the mms arm needs the home whose daemon Codex asks")
        server = [sys.executable, "-m", "mms.cli", "--home", str(home), "mcp", "--tools", tools]
        settings = [*CONFIG, *settings, f"mcp_servers.{SERVER}.command={toml(server[0])}",
                    f"mcp_servers.{SERVER}.args=[{','.join(toml(a) for a in server[1:])}]",
                    *(f"mcp_servers.{SERVER}.{c}" for c in SERVER_CONFIG),
                    *([f"mcp_servers.{SERVER}.env.CLAUDE_PROJECT_DIR={toml(work)}"] if work else [])]
        disabled = DISABLED
    args = [*command, "exec", "--json", "--skip-git-repo-check", "--ignore-user-config", "--ignore-rules",
            "--strict-config", "-m", name]
    for setting in settings:
        args += ["-c", setting]
    for feature in disabled:
        args += ["--disable", feature]
    return [*args, "-C", str(work or run_dir / "empty"), "--", "-"]


def check_argv(argv: list[str], command: list[str]) -> None:
    """A `.cmd` shim is run by cmd.exe: refuse an argument it would misread and a command line it would cut."""
    if os.name == "nt" and argv[0].lower().endswith((".cmd", ".bat")):
        for arg in argv[len(command):]:
            if UNSAFE.search(arg):
                raise CodexMissing(f"an argument holds a character that cmd.exe reads as syntax: {arg[:80]!r}")
        if (size := len(subprocess.list2cmdline(argv))) > CMD_LIMIT:
            raise CodexMissing(f"the command line is {size} characters, over the {CMD_LIMIT} a .cmd shim passes")


def prepare(run_dir: Path, name: str, level: str, guidance: str, home: Path | None, models: dict | None = None,
            command: list[str] | None = None, *, arm: str = "mms", work: Path | None = None,
            notes: str = NOTES.name, bin_dir: Path | None = None) -> list[str]:
    """Write `catalog.json`, `instructions.md` and the working folder (`work`, else the empty one) into `run_dir`, and
    return the command line. `models` is a saved `codex debug models` output (a dry run); else the installed Codex is
    asked."""
    command = command or codex_command()
    models = models if models is not None else models_of(command)
    run_dir.mkdir(parents=True, exist_ok=True)
    (work or run_dir / "empty").mkdir(parents=True, exist_ok=True)
    (run_dir / "catalog.json").write_text(json.dumps(catalog(models, name, level)), encoding="utf-8", newline="\n")
    (run_dir / "instructions.md").write_text(instructions(guidance, notes, arm), encoding="utf-8", newline="\n")
    argv = argv_of(command, name, level, run_dir, home, arm=arm, work=work, bin_dir=bin_dir)
    check_argv(argv, command)
    return argv


IN_ONLY = "in:"  # a `deny` entry that starts with this is checked in what the agent runs (a command, a path), not in what it reads


def reached(value, deny: tuple[str, ...], output: bool = False) -> str | None:
    """The first of `deny` that `value` (a command, a tool's input; with `output`, what a tool printed) holds, read as lower-case
    JSON with forward slashes (each run of slashes and backslashes is one `/`); else None. `deny` is lower case with forward
    slashes; an `IN_ONLY` entry is left out of an output (a listing may name a folder that a command may not open)."""
    text = re.sub(r"[\\/]+", "/", json.dumps(value, ensure_ascii=False).lower())
    for d in deny:
        if d.startswith(IN_ONLY):
            if output:
                continue
            d = d[len(IN_ONLY):]
        if d and d in text:
            return d
    return None


def check_event(event: dict, arm: str = "mms", deny: tuple[str, ...] = ()) -> str | None:
    """Why an event ends the exec, else None. The mms arm: an item that is not a message, a thought, a plan list or a call
    of mms. The files arm: one that is not a message, a thought, a plan list or a shell command, or a command or its output
    that holds a `deny` string. The blind arm: any tool item at all."""
    if event.get("type") not in ("item.started", "item.completed", "item.updated"):
        return None
    item = event.get("item") or {}
    what = item.get("type")
    if arm == "blind":
        return None if what in ITEMS_BLIND else f"codex used {what!r}, which the blind arm does not allow: it has no tools"
    if arm == "files":
        if what not in ITEMS_FILES:
            return f"codex used {what!r}, which the files arm does not allow: the restriction failed"
        if what == "command_execution" and (hit := reached(item.get("command"), deny)):
            return f"shell reached {hit}"
        if what == "command_execution" and (hit := reached(item.get("aggregated_output"), deny, output=True)):
            return f"shell output reached {hit}"
        return None
    if what not in ITEMS:
        return f"codex used {what!r}, which is not one of mms's tools: the restriction failed"
    if what != "mcp_tool_call" or item.get("tool") in HELPERS.get(item.get("server"), ()):
        return None
    if item.get("server") != SERVER:
        return f"codex called {item.get('tool')!r} of the server {item.get('server')!r}, not {SERVER!r}"
    if event["type"] == "item.completed" and "requires approval" in str((item.get("error") or {}).get("message", "")):
        return "a tool call was held for approval: the pre-approval did not apply"
    return None


BLOB = re.compile(r"[A-Za-z0-9+/=_-]{200}")  # the start of base64 (or base64url): 200 characters with no space or punctuation


def scrubbed(value):
    """`value` with long data (a `data:` URL, an image block's base64) as its hash, so a log holds no pictures. Text that starts
    like JSON is never hashed: an mms tool's result is a long compact JSON string, and `work_of` and `open_jobs` read it."""
    if isinstance(value, dict):
        return {k: scrubbed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrubbed(v) for v in value]
    if isinstance(value, str) and len(value) > 3000 and (value.startswith("data:") or BLOB.match(value)):
        return f"<{len(value)} characters, sha1 {hashlib.sha1(value.encode()).hexdigest()[:12]}>"
    return value


def kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()


def answer_of(text: str) -> str | None:
    """The text after the last `ANSWER:` line of a message."""
    found = re.findall(r"^\W*ANSWER:\s*(.+?)\s*$", text, re.I | re.M)
    return found[-1].strip(" *`") if found else None


def tools_of(events: list[dict]) -> list[dict]:
    """Each completed call of a tool: its name, arguments, result (without picture data) and error. A shell command is
    `shell`, its arguments the command and its result the start of its output."""
    out = []
    for e in events:
        item = e.get("item") or {}
        if e.get("type") != "item.completed":
            continue
        if item.get("type") == "mcp_tool_call":
            out.append({"tool": item.get("tool"), "arguments": item.get("arguments"),
                        "result": scrubbed(item.get("result")), "error": item.get("error"),
                        "status": item.get("status")})
        elif item.get("type") == "command_execution":
            code = item.get("exit_code")
            out.append({"tool": "shell", "arguments": {"command": item.get("command")},
                        "result": str(item.get("aggregated_output") or "")[:SHELL_OUTPUT],
                        "error": None if code in (0, None) else f"exit code {code}", "status": item.get("status")})
    return out


def usage_of(events: list[dict]) -> tuple[int, dict | None]:
    """How many turns completed, and the tokens of all of them (None when none reports usage). Codex's `input_tokens`
    includes the cached ones, which are also `cache_read`; it has no cache writes."""
    done = [e for e in events if e.get("type") == "turn.completed"]
    usages = [e["usage"] for e in done if isinstance(e.get("usage"), dict)]
    if not usages:
        return len(done), None

    def total(key: str) -> int:
        return sum(v for u in usages if isinstance(v := u.get(key), int))

    return len(done), {"input": total("input_tokens"), "output": total("output_tokens"),
                       "cache_read": total("cached_input_tokens"), "cache_write": 0}


def path_with(bin_dir: Path | None, env: dict | None = None) -> dict:
    """A copy of `env` (default: this process's environment) with `bin_dir` first on PATH."""
    out = dict(os.environ if env is None else env)
    if bin_dir is not None:
        out["PATH"] = os.pathsep.join([str(bin_dir), out.get("PATH", "")])
    return out


def weekly(limits: dict) -> float | None:
    """`used_percent` of the window of `limits` that is the week (`window_minutes` 10080, `primary` or `secondary`
    by plan); None when no window is the week or its percent is not a number."""
    for window in (limits.get("primary"), limits.get("secondary")):
        if isinstance(window, dict) and window.get("window_minutes") == WEEK:
            used = window.get("used_percent")
            return float(used) if isinstance(used, (int, float)) and not isinstance(used, bool) else None
    return None


def percent_in(lines) -> float | None:
    """The weekly use of the last `rate_limits` among the JSON lines of a rollout; None when it cannot be read."""
    last = None

    def walk(value):
        nonlocal last
        if isinstance(value, dict):
            if isinstance(value.get("rate_limits"), dict):
                last = weekly(value["rate_limits"])
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for line in lines:
        try:
            walk(json.loads(line))
        except ValueError:
            continue
    return last


def rollout_percent(thread: str, wait_s: float = 10.0) -> float | None:
    """The plan usage this exec's own rollout file reports: `~/.codex/sessions/**/rollout-*-<thread>.jsonl`, and no other
    session's file."""
    if not thread:
        return None
    end = time.monotonic() + wait_s
    while True:
        for file in (codex_home() / "sessions").glob(f"**/rollout-*-{thread}.jsonl"):
            with file.open(encoding="utf-8", errors="replace") as lines:
                if (found := percent_in(lines)) is not None:
                    return found
        if time.monotonic() > end:
            return None
        time.sleep(0.5)


def images_in(lines) -> list[dict]:
    """The `view_image` calls among the JSON lines of a rollout, as `tools_of` gives a call. `codex exec --json` has no
    item for them (seen 2026-11-02), so the rollout is the only place they are told."""
    out = []
    for line in lines:
        try:
            payload = json.loads(line).get("payload") or {}
        except (ValueError, AttributeError):
            continue
        if isinstance(payload, dict) and payload.get("type") == "function_call" and payload.get("name") == "view_image":
            try:
                arguments = json.loads(payload.get("arguments") or "{}")
            except ValueError:
                arguments = {"raw": payload.get("arguments")}
            out.append({"tool": "view_image", "arguments": arguments, "result": None, "error": None,
                        "status": "completed"})
    return out


def rollout_images(thread: str) -> list[dict]:
    """The `view_image` calls of this exec's own rollout file (read after `rollout_percent` waited for it)."""
    for file in (codex_home() / "sessions").glob(f"**/rollout-*-{thread}.jsonl") if thread else ():
        with file.open(encoding="utf-8", errors="replace") as lines:
            return images_in(lines)
    return []


def problem_of(code: int | None, events: list[dict], stderr: str) -> str | None:
    """`kind: reason` when the exec did not complete its turn, else None. The exit code alone is not enough. `setup`: a
    required MCP server or the configuration failed, so this was not the exec asked for. `capped`: the context window ran
    out. Then `limit` and `auth` by the last message (or stderr), else `error`."""
    types = {e.get("type") for e in events}
    if code == 0 and "turn.completed" in types and "turn.failed" not in types:
        return None
    messages = []
    for e in events:
        if e.get("type") in ("turn.failed", "error"):
            found = e["error"].get("message") if isinstance(e.get("error"), dict) else e.get("message")
            messages.append(str(found or ""))
    text = next((m for m in reversed(messages) if m), "") or stderr.strip()[-500:]
    text = text or f"codex exec exited {code} without finishing the turn"
    everything = "\n".join([*messages, stderr.strip()[-500:]])
    for kind, found in (("setup", SETUP), ("capped", CAPPED)):
        if found.search(everything):
            return f"{kind}: {next((m for m in [*messages, stderr] if found.search(m)), text)[:500]}"
    return f"{'limit' if LIMIT.search(text) else 'auth' if AUTH.search(text) else 'error'}: {text[:500]}"


def ask_codex(prompt_text: str, home: Path | None, run_dir: Path, name: str | None = None, level: str | None = None,
              guidance: str = "default", timeout: float = TIMEOUT_S, models: dict | None = None, *, arm: str = "mms",
              work: Path | None = None, notes: str = NOTES.name, deny: tuple[str, ...] = (),
              bin_dir: Path | None = None) -> dict:
    """Run one exec for `prompt_text` against the daemon of `home` (None in the files and blind arms), with its files and logs in
    `run_dir`. `work` is its working folder, `notes` the skill's notes file for the model behind mms, `deny` what the files
    arm's commands may not hold, and `bin_dir` a folder put first on its PATH."""
    check_personal()
    command = codex_command()
    name, level = name or model(), level or effort()
    argv = prepare(run_dir, name, level, guidance, home, models, command, arm=arm, work=work, notes=notes,
                   bin_dir=bin_dir)
    (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    events: list[dict] = []
    escaped: list[str] = []
    stderr_file = (run_dir / "stderr.txt").open("wb")
    began = time.monotonic()
    process = subprocess.Popen(argv, cwd=work or run_dir / "empty", env=path_with(bin_dir), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=stderr_file, start_new_session=os.name != "nt")

    def read() -> None:
        for raw in process.stdout:
            try:
                event = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if reason := check_event(event, arm, deny):
                escaped.append(reason)
                kill_tree(process)
                break

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        process.stdin.write(prompt_text.encode("utf-8"))
        process.stdin.close()
    except OSError:
        pass  # an exec that died before reading its prompt is told by its exit
    problem = None
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        process.wait()
        problem = f"timeout: no answer in {timeout:g} s"
    reader.join(30)
    stderr_file.close()
    seconds = round(time.monotonic() - began, 1)
    stderr = (run_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-2000:]
    if escaped:
        problem = f"escaped: {escaped[0]}"
    elif problem is None:
        problem = problem_of(process.returncode, events, stderr)
    thread = next((str(e.get("thread_id") or "") for e in events if e.get("type") == "thread.started"), "")
    messages = [str((e.get("item") or {}).get("text") or "") for e in events
                if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message"]
    final = messages[-1] if messages else ""
    turns, tokens = usage_of(events)
    used = rollout_percent(thread)
    tools = tools_of(events)
    if arm == "files":
        images = rollout_images(thread)
        tools += images
        hit = next((h for c in images if (h := reached(c["arguments"], deny))), None)
        if hit and not str(problem).startswith("escaped:"):
            problem = f"escaped: view_image reached {hit}"
    out = {"answer": answer_of(final), "final": final, "tools": tools, "events": [scrubbed(e) for e in events],
           "seconds": seconds, "turns": turns, "tokens": tokens, "cost_usd": None,
           "used_percent": used, "model": name, "effort": level, "thread": thread, "problem": problem, "argv": argv,
           "stderr": stderr}
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in out["events"]), encoding="utf-8")
    (run_dir / "tools.json").write_text(json.dumps(out["tools"], indent=1), encoding="utf-8")
    (run_dir / "final.txt").write_text(final, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask Codex one question over mms mcp, or with --dry-run print the command.")
    parser.add_argument("--home", type=Path, required=True, help="the mms home whose daemon Codex asks")
    parser.add_argument("--dry-run", action="store_true", help="write the catalog and instructions, print the command, run nothing")
    parser.add_argument("--models", type=Path, help="a saved `codex debug models` output (a dry run needs it)")
    parser.add_argument("--run-dir", type=Path, help="where the files go (default: a new temporary folder)")
    parser.add_argument("--model", default=model())
    parser.add_argument("--effort", default=effort())
    parser.add_argument("--guidance", default="default", choices=("default", "none"))
    parser.add_argument("--question", default="What's the gate code from the email this morning?")
    args = parser.parse_args(argv)
    run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix="codex-run-"))
    if args.dry_run:
        if args.models is None:
            parser.error("--dry-run needs --models, a saved `codex debug models` output: it runs no codex command")
        found = shutil.which("codex")
        command = [found] if found else ["codex"]
        cmd = prepare(run_dir, args.model, args.effort, args.guidance, args.home,
                      json.loads(args.models.read_text(encoding="utf-8")), command)
        first, last = cmd.index("-c"), cmd.index("-C")
        print(f"run dir: {run_dir} (nothing was run; the prompt goes on stdin)")
        print(" ".join(cmd[:first]))
        for setting in (cmd[i + 1] for i, a in enumerate(cmd) if a == "-c"):
            print("  -c", setting)
        print("  --disable", " ".join(cmd[i + 1] for i, a in enumerate(cmd) if a == "--disable"))
        print(" ".join(cmd[last:]))
        print("personal instructions in the Codex home:", [str(p) for p in personal_files()] or "none")
        print("unsafe characters in the arguments:", [a for a in cmd[1:] if UNSAFE.search(a)] or "none")
        print("prompt:", prompt(args.question, time.time() - 86400))
        return 0
    result = ask_codex(prompt(args.question, time.time() - 86400), args.home, run_dir, args.model, args.effort,
                       args.guidance)
    print(json.dumps({k: result[k] for k in ("answer", "final", "seconds", "used_percent", "problem")}, indent=1))
    print("logs:", run_dir)
    return 1 if result["problem"] else 0


if __name__ == "__main__":
    sys.exit(main())
