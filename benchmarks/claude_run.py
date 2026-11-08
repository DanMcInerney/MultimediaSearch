"""Claude Code asked a question (TESTBED.md, Benchmark): one `claude -p` for one prompt, locked down for one arm of the
benchmark, in the way `codex_run.py` locks down Codex. A dry run (`prepare`) runs nothing. It never reads, copies or moves a
file of `~/.claude`: Claude reads its own login.

`ask_claude` returns `{answer, final, tools, events, seconds, turns, tokens, cost_usd, used_percent, model, denials,
problem, argv, stderr}`: the `ANSWER:` line of the last message (None without one), what Claude called and got back (a
tool's bare name: `ask`, `find`, `look`, `label`, or `Read`, `Glob`, `Grep`, `Bash`), how long it took, its turns, tokens
and cost, the weekly plan use a `rate_limit_event` reports (None when none does), and `problem` (None, or a kind and the
reason). The kinds, by what the benchmark does with them: `escaped` (a tool outside the arm, or a `deny` string in a tool's
input or output), `timeout` and `capped` (`error_max_turns`, `error_max_budget_usd`) count as a wrong answer; `setup` (the
session is not the one asked for: a wrong init event, or an option `claude` refused), `limit` and `auth` stop the agent;
`error` is run again. It raises `ClaudeMissing` before any run when nothing can run.

The three arms:
- `blind`: `--tools ""`, no MCP server and no `--allowedTools`: what the model knows, with no recordings. A tool use is `escaped`.
- `mms`: the one MCP server `mms` (this interpreter running `mms.cli --home HOME mcp --tools ask,find,look,label`, its
  project pinned by `CLAUDE_PROJECT_DIR`) and no built-in tool. The server's instructions reach Claude through the server;
  the skill and the model's notes are appended to the system prompt, because skills are off in the run.
- `files`: Read, Glob, Grep and Bash that runs ffmpeg and the read-only commands only, over the working folder and the
  data folder, with no MCP server. No tool input or output may hold a `deny` string (a path or word that leads to the
  answers). Its PATH is `bin_dir`, the Windows folders and Git for Windows' (the Bash tool runs in it), so no `mms` and no
  `.venv`, and its MMS_HOME is a folder that does not exist.

How the exec is made:
- `claude -p --verbose --output-format stream-json --restricted --strict-mcp-config --permission-mode dontAsk ...`, the
  prompt on stdin. `--restricted` loads managed settings and `--settings` only (no user hooks, plugins or permission
  rules) and keeps the file tools in the working folder and `--add-dir`; `--strict-mcp-config` drops the owner's own `mms`
  registration, so only `--mcp-config` connects; `--tools` sets the built-in tools (MCP tools are not affected);
  `dontAsk` refuses whatever `--allowedTools` does not allow; `--disable-slash-commands` keeps installed skills out;
- the environment: the owner's API key out (`-p` would use it: the benchmark uses the plan), and `ENV_SET` in (no
  CLAUDE.md or auto-memory, no claude.ai connectors, no background tasks, no updates or telemetry);
- every event is checked as it arrives: an `init` whose tools, MCP servers or plugins are not the arm's, or that says the
  session is billed to an API key, or a missing `init`, stops the exec as `setup`; a tool the arm does not have, or
  (files arm) an input or an output that holds a `deny` string, as `escaped`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import codex_run

SKILL = codex_run.SKILL
ARMS = codex_run.ARMS
BASE_MMS = "You answer the user's question from their memory, using the mms tools."
BASE_BLIND = codex_run.BASE_BLIND
# The files arm's Bash runs ffmpeg alone (FILES_ALLOWED), and Claude is told so: in the pilot it was denied a `cd ... && find`,
# took Bash for gone and answered "unknown" without trying ffmpeg. Codex needs no such line: its shell is the sandbox's, whole.
BASE_FILES = ("You answer the user's question from their files, using your tools. Bash runs ffmpeg only: start each Bash "
              "command with ffmpeg (no cd, pipes, && or other programs), and write what it makes in your working folder. "
              "Find and read files with Glob, Grep and Read; Read shows you the pictures ffmpeg writes.")
TIMEOUT_S = 900.0
# `claude --help` (2.1.302) does not list --max-turns or --append-system-prompt-file, but both are taken (checked on
# 2026-11-02). If a later version refuses one, set PROMPT_FILE to False (the text then goes in --append-system-prompt,
# about 26k characters, under Windows' 32k command line) or MAX_TURNS to None (no cap).
PROMPT_FILE = True
MAX_TURNS: int | None = 60
BUDGET_USD = 5.0  # a list-price estimate per exec; the plan's own limit is separate
CMD_LIMIT = 32_000  # characters of a Windows command line (CreateProcess takes 32,767)
MMS_TOOLS = ("mcp__mms__ask", "mcp__mms__find", "mcp__mms__look", "mcp__mms__label")
FILES_TOOLS = ("Read", "Glob", "Grep", "Bash")
ARM_TOOLS = {"mms": MMS_TOOLS, "files": FILES_TOOLS, "blind": ()}  # the tools a session of each arm lists in its init event
FILES_ALLOWED = ("Read", "Glob", "Grep", "Bash(ffmpeg *)")
# Claude's built-in plugins load under --restricted too (2.1.302: these two, each adding a skill); off, the init lists none
BUILTIN_PLUGINS = ("cc-plugin-agents-md@builtin", "cc-plugin-plugin-authoring@builtin")
SETTINGS = {"disableAllHooks": True, "enabledPlugins": dict.fromkeys(BUILTIN_PLUGINS, False)}
ENV_SET = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",  # no user, project or auto-memory CLAUDE.md
           "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
           "ENABLE_CLAUDEAI_MCP_SERVERS": "false",  # no claude.ai connectors
           "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
           "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",  # a backgrounded call's result is lost in -p
           "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1"}
ENV_DROP = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")  # -p uses a key when one is set; the bench uses the plan
ENV_DROP_FILES = ("VIRTUAL_ENV",)  # and, from the files arm, these and every MMS_* (they point at the owner's mms)
MMS_PREFIX = "mcp__mms__"
TOKEN_KEYS = {"input": "input_tokens", "output": "output_tokens", "cache_read": "cache_read_input_tokens",
              "cache_write": "cache_creation_input_tokens"}  # the result's `usage`
LIMIT = re.compile(r"hit your .{0,20}limit|usage limit|rate limit|rate_limit|\b429\b|quota", re.I)
# also how Claude tells a login is needed: "Failed to authenticate: OAuth session expired and could not be refreshed"
AUTH = re.compile(codex_run.AUTH.pattern + r"|/login|invalid api key|failed to authenticate|oauth", re.I)
CAPPED = ("error_max_turns", "error_max_budget_usd")  # result subtypes: the turn or the dollar cap was reached
# what `claude` prints, from the text of claude.exe (commander's), when it refuses an option
SETUP = re.compile(r"error: (unknown option|unknown command|too many arguments|missing required argument|required option|"
                   r"option '[^']*' argument)", re.I)


class ClaudeMissing(RuntimeError):
    """Nothing can run: no usable `claude` command, or a command line Windows would cut."""


def model() -> str:
    return os.environ.get("MMS_CLAUDE_MODEL", "sonnet")


def effort() -> str:
    return os.environ.get("MMS_CLAUDE_EFFORT", "high")


def claude_command() -> list[str]:
    found = shutil.which("claude")
    if not found:
        raise ClaudeMissing("the claude command is not on PATH: install Claude Code and log in")
    if found.lower().endswith((".cmd", ".bat")):
        raise ClaudeMissing(f"{found} is a .cmd shim, which cmd.exe would misread the arguments of (an empty --tools, "
                            "parentheses): use the native claude.exe")
    return [found]


def check_login(command: list[str] | None = None) -> None:
    """Raise ClaudeMissing unless `claude auth status`, in an exec's environment (no API key), says the CLI is logged in.
    The CLI has a login of its own, which can be out of date while the desktop app's is fine."""
    done = subprocess.run([*(command or claude_command()), "auth", "status"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=60, env=env_of(None))
    try:
        status = json.loads(done.stdout)
    except ValueError:
        raise ClaudeMissing(f"`claude auth status` gave no status (exit {done.returncode}): "
                            f"{(done.stdout + done.stderr).strip()[-300:]}") from None
    if not isinstance(status, dict) or status.get("loggedIn") is not True:
        raise ClaudeMissing("the claude CLI is not logged in (`claude auth status`): run `claude auth login` in a "
                            "terminal, then the same command")


def instructions(arm: str, notes: str | None = None) -> str:
    """The text appended to Claude's system prompt. The mms arm: one line, the skill and the notes file `notes` (None for
    none). The files and blind arms: their one line."""
    if arm == "files":
        return BASE_FILES
    if arm == "blind":
        return BASE_BLIND
    if arm != "mms":
        raise ValueError(f"arm is one of {ARMS}, not {arm!r}")
    parts = [BASE_MMS, SKILL.read_text(encoding="utf-8"), *([codex_run.notes_text(notes)] if notes else [])]
    return "\n\n".join(parts)


def mcp_config(home: Path, work: Path) -> dict:
    """The `--mcp-config` of the mms arm: the one server, offering `ask, find, look, label`, in the project of `work`."""
    return {"mcpServers": {codex_run.SERVER: {
        "command": sys.executable,
        "args": ["-m", "mms.cli", "--home", str(home), "mcp", "--tools", codex_run.TOOLS],
        "env": {"CLAUDE_PROJECT_DIR": str(work)}}}}


def argv_of(command: list[str], arm: str, run_dir: Path, *, name: str, level: str, data: Path | None = None) -> list[str]:
    """The command line of an exec (the prompt is on stdin); `run_dir` holds the files `prepare` wrote."""
    if arm not in ARMS:
        raise ValueError(f"arm is one of {ARMS}, not {arm!r}")
    args = [*command, "-p", "--verbose", "--output-format", "stream-json", "--restricted", "--strict-mcp-config",
            "--permission-mode", "dontAsk", "--no-session-persistence", "--disable-slash-commands",
            "--model", name, "--effort", level, "--max-budget-usd", f"{BUDGET_USD:g}",
            "--settings", str(run_dir / "settings.json")]
    if MAX_TURNS:
        args += ["--max-turns", str(MAX_TURNS)]
    if PROMPT_FILE:
        args += ["--append-system-prompt-file", str(run_dir / "instructions.md")]
    else:
        args += ["--append-system-prompt", (run_dir / "instructions.md").read_text(encoding="utf-8")]
    if arm == "mms":
        return [*args, "--mcp-config", str(run_dir / "mcp.json"), "--tools", "", "--allowedTools", *MMS_TOOLS]
    if arm == "blind":
        return [*args, "--tools", ""]
    if data is None:
        raise ValueError("the files arm needs the data folder")
    return [*args, "--tools", ",".join(FILES_TOOLS), "--allowedTools", *FILES_ALLOWED, "--add-dir", str(data)]


def git_root(search: str | None = None) -> Path | None:
    """The root of Git for Windows, which Claude's Bash tool runs in. It is found from the folders of `search` (default:
    this process's PATH) that hold a `git.exe`, and failing that a `bash.exe`: going up from it to the first folder that
    holds `usr/bin/bash.exe`. (A `bash.exe` of System32 is WSL's: nothing above it matches.) None off Windows or when
    it is not found."""
    if os.name != "nt":
        return None
    folders = [Path(f) for f in (os.environ.get("PATH", "") if search is None else search).split(os.pathsep) if f]
    for exe in ("git.exe", "bash.exe"):
        for found in (folder / exe for folder in folders if (folder / exe).is_file()):
            for parent in found.parents:
                if (parent / "usr" / "bin" / "bash.exe").is_file():
                    return parent
    return None


def env_of(bin_dir: Path | None, arm: str = "mms", work: Path | None = None) -> dict:
    """This process's environment less the API key, plus `ENV_SET`, with `bin_dir` first on PATH. The files arm (with its
    working folder `work`) also gets a PATH of `bin_dir`, Git for Windows' folders and the Windows folders only, with
    Claude told where its bash is, a MMS_HOME that does not exist and none of the owner's other MMS_* or
    VIRTUAL_ENV: its Bash tool cannot reach the owner's `mms` or `.venv`."""
    env = {k: v for k, v in os.environ.items() if k not in ENV_DROP}
    if arm != "files":
        return codex_run.path_with(bin_dir, {**env, **ENV_SET})
    if work is None:
        raise ValueError("the files arm needs its working folder")
    env = {k: v for k, v in env.items() if k not in ENV_DROP_FILES and not k.startswith("MMS_")}
    root = git_root()
    if os.name == "nt" and root is None:
        raise ClaudeMissing("Git for Windows was not found from `git` or `bash` on PATH: Claude's Bash tool needs it")
    folders, bash = (), None
    if root:
        folders = tuple(str(root / sub) for sub in ("cmd", "bin", "usr/bin") if (root / sub).is_dir())
        bash = next((root / sub / "bash.exe" for sub in ("bin", "usr/bin") if (root / sub / "bash.exe").is_file()), None)
    env.update(ENV_SET)
    env["PATH"] = codex_run.shell_path(bin_dir, folders)
    env["MMS_HOME"] = str(work / codex_run.NO_HOME)
    if bash:
        env["CLAUDE_CODE_GIT_BASH_PATH"] = str(bash)
    return env


def prepare(run_dir: Path, arm: str, work: Path, *, home: Path | None = None, data: Path | None = None,
            notes: str | None = None, name: str | None = None, level: str | None = None,
            command: list[str] | None = None) -> list[str]:
    """Write `instructions.md`, `settings.json` and, for the mms arm, `mcp.json` into `run_dir`, make `work`, and return the
    command line. The mms arm needs `home`, the files arm `data`."""
    if arm == "mms" and home is None:
        raise ValueError("the mms arm needs the home whose daemon Claude asks")
    command = command or claude_command()
    run_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    (run_dir / "instructions.md").write_text(instructions(arm, notes), encoding="utf-8", newline="\n")
    (run_dir / "settings.json").write_text(json.dumps(SETTINGS), encoding="utf-8", newline="\n")
    if arm == "mms":
        (run_dir / "mcp.json").write_text(json.dumps(mcp_config(home, work), indent=1), encoding="utf-8", newline="\n")
    argv = argv_of(command, arm, run_dir, name=name or model(), level=level or effort(), data=data)
    if os.name == "nt" and (size := len(subprocess.list2cmdline(argv))) > CMD_LIMIT:
        raise ClaudeMissing(f"the command line is {size} characters, over the {CMD_LIMIT} Windows passes")
    return argv


# -- the stream -------------------------------------------------------------------------------------------------------

def blocks_of(event: dict) -> list[dict]:
    """The content blocks of an `assistant` or `user` event's message."""
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def is_init(event: dict) -> bool:
    return event.get("type") == "system" and event.get("subtype") == "init"


def check_init(event: dict, arm: str) -> str | None:
    """Why the session the init event describes is not the one asked for (its tools, MCP servers or plugins differ from
    the arm's, or it does not say, or it is billed to an API key), else None."""
    tools = event.get("tools")
    if not isinstance(tools, list):
        return "the init event lists no tools, so what the session can use is unknown"
    want, have = set(ARM_TOOLS[arm]), set(map(str, tools))
    if have != want:
        return f"the session's tools are not the {arm} arm's: extra {sorted(have - want)}, missing {sorted(want - have)}"
    servers = event.get("mcp_servers") or []
    if not isinstance(servers, list):
        return f"the init event's mcp_servers is not a list: {servers!r}"
    if arm == "mms":
        if len(servers) != 1 or not isinstance(servers[0], dict) or servers[0].get("name") != codex_run.SERVER:
            return f"the session's MCP servers are not just {codex_run.SERVER!r}: {servers!r}"
        if servers[0].get("status") != "connected":
            return f"the {codex_run.SERVER!r} server is {servers[0].get('status')!r}, not connected"
    elif servers:
        return f"the {arm} arm's session has MCP servers: {servers!r}"
    if event.get("plugins"):
        return f"the session has plugins: {event['plugins']!r}"
    if event.get("apiKeySource") not in (None, "none"):
        return f"the session is billed to an API key ({event['apiKeySource']}), not the plan's login"
    return None


def own_outputs(value, session: str | None):
    """`value` with the folder where Claude Code keeps this session's long tool outputs named `/agent-outputs` instead: a Bash
    output over Claude Code's limit is saved there and its result names the file (`<persisted-output>`), so the files arm may
    read its own output back, though the folder is under ~/.claude. Another session's folder, and a path that climbs out of
    this one (`..`), are left as they are."""
    if not session:
        return value
    folder = re.compile(r"[\\/]\.claude[\\/]+projects[\\/]+[^\\/\s\"']+[\\/]+" + re.escape(session)
                        + r"[\\/]+tool-results(?=[\\/\s\"']|$)(?![^\s\"']*\.\.)", re.IGNORECASE)

    def put(v):
        if isinstance(v, str):
            return folder.sub("/agent-outputs", v)
        if isinstance(v, list):
            return [put(x) for x in v]
        if isinstance(v, dict):
            return {k: put(x) for k, x in v.items()}
        return v

    return put(value)


def check_event(event: dict, arm: str, deny: tuple[str, ...] = (), session: str | None = None) -> str | None:
    """Why an assistant message ends the exec: a tool the arm does not have (the blind arm has none), or (files arm) an input
    holding a `deny` string, outside the `session`'s own outputs (`own_outputs`). Else None."""
    if event.get("type") != "assistant":
        return None
    allowed = ARM_TOOLS[arm]
    for block in blocks_of(event):
        if not str(block.get("type")).endswith("tool_use"):
            continue
        name = str(block.get("name"))
        if name not in allowed:
            return f"claude used {name!r}, which is not one of the {arm} arm's tools: the restriction failed"
        if arm == "files" and (hit := codex_run.reached(own_outputs(ran_of(block), session), deny)):
            return f"{name} reached {hit}"
    return None


def ran_of(block: dict):
    """What a tool use asks for, to be checked against the deny list: a shell command is its `command` alone (its `description`
    is the agent's prose about it, which may say anything), any other tool's input without its `description`."""
    value = block.get("input")
    if not isinstance(value, dict):
        return value
    if block.get("name") == "Bash" and "command" in value:
        return value["command"]
    return {k: v for k, v in value.items() if k != "description"}


def check_result(event: dict, arm: str, deny: tuple[str, ...], names: dict, session: str | None = None) -> str | None:
    """Why the results of tools in a `user` event end the exec: (files arm) output that holds a `deny` string, a picture's
    data and the `session`'s own outputs folder left out (`own_outputs`). `names` maps a tool use's id to its tool. Else None."""
    if arm != "files" or event.get("type") != "user":
        return None
    for block in blocks_of(event):
        if block.get("type") != "tool_result":
            continue
        content = block.get("content")
        if isinstance(content, list):
            content = [b for b in content if not (isinstance(b, dict) and b.get("type") == "image")]
        if hit := codex_run.reached(own_outputs(content, session), deny, output=True):
            return f"{names.get(block.get('tool_use_id'), 'a tool')} output reached {hit}"
    return None


def watcher(arm: str, deny: tuple[str, ...] = ()) -> Callable[[dict], str | None]:
    """A check for the events of one exec, in order: the init event, which must come before the first assistant message,
    then each assistant message and each tool's result. It returns the problem that must end the exec (`setup: ...` for a
    session that is not the one asked for, `escaped: ...` for what Claude did), else None."""
    inited = False
    session: str | None = None  # the init event's session_id: its own outputs folder may be read
    names: dict[str, str] = {}

    def watch(event: dict) -> str | None:
        nonlocal inited, session
        if is_init(event):
            inited = True
            session = str(event.get("session_id") or "") or None
            return f"setup: {why}" if (why := check_init(event, arm)) else None
        if event.get("type") == "assistant":
            if not inited:
                return "setup: an assistant message came before the init event, so what the session can use is unknown"
            names.update({str(b.get("id")): str(b.get("name")) for b in blocks_of(event) if b.get("type") == "tool_use"})
            why = check_event(event, arm, deny, session)
        else:
            why = check_result(event, arm, deny, names, session)
        return f"escaped: {why}" if why else None

    return watch


def tools_of(events: list[dict]) -> list[dict]:
    """Each call of a tool: its bare name, input, result (without picture data) and whether it failed. A call is once, by
    its id, however many events carry it."""
    results = {b.get("tool_use_id"): b for e in events if e.get("type") == "user" for b in blocks_of(e)
               if b.get("type") == "tool_result"}
    out, seen = [], set()
    for e in events:
        if e.get("type") != "assistant":
            continue
        for block in blocks_of(e):
            if block.get("type") != "tool_use" or block.get("id") in seen:
                continue
            seen.add(block.get("id"))
            got = results.get(block.get("id"), {})
            out.append({"tool": str(block.get("name")).removeprefix(MMS_PREFIX), "arguments": block.get("input"),
                        "result": codex_run.scrubbed(got.get("content")), "error": bool(got.get("is_error"))})
    return out


def used_percent_of(events: list[dict]) -> float | None:
    """The week's plan use in percent from the last `rate_limit_event` that gives it, else None: the seven-day window of its
    `unifiedWindows`, or its own `utilization` when `rateLimitType` is `seven_day`. Both are fractions of the window."""
    used = None
    for e in events:
        info = e.get("rate_limit_info") if e.get("type") == "rate_limit_event" else None
        if not isinstance(info, dict):
            continue
        windows = info.get("unifiedWindows")
        week = windows.get("seven_day") if isinstance(windows, dict) else None
        if isinstance(week, dict):
            fraction = week.get("utilization")
        elif info.get("rateLimitType") == "seven_day":
            fraction = info.get("utilization")
        else:
            continue
        if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
            used = float(fraction) * 100
    return used


def problem_of(result: dict | None, code: int | None, stderr: str) -> str | None:
    """`kind: reason` when the exec did not succeed, else None. `capped`: the turn or the dollar cap was reached. `setup`:
    `claude` exited without a result after refusing an option. A usage limit or a login problem is told from the result's
    text or stderr; no result at all is an error."""
    if result is not None and result.get("subtype") == "success" and not result.get("is_error"):
        return None
    if result is not None and result.get("subtype") in CAPPED:
        return f"capped: {result['subtype']}"
    if result is None and (refused := next((x for x in stderr.splitlines() if SETUP.search(x)), None)):
        return f"setup: {refused.strip()[:500]}"
    text = str((result or {}).get("result") or "").strip()
    tail = stderr.strip()[-500:]
    for kind, pattern in (("limit", LIMIT), ("auth", AUTH)):
        if pattern.search(text) or pattern.search(tail):
            return f"{kind}: {(text or tail)[:500]}"
    if result is None:
        return f"error: claude exited {code} without a result"
    subtype = result.get("subtype")
    why = ": ".join(part for part in (subtype if subtype != "success" else "", text[:500]) if part)
    return f"error: {why or 'the result is an error with no text'}"


def result_of(events: list[dict], code: int | None, stderr: str) -> dict:
    """What the stream says of the exec: `final` (the result's text, else the last assistant text), `problem`, `turns`,
    `tokens`, `cost_usd`, `model`, `denials` (permission denials) and `used_percent`."""
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    texts = [str(b.get("text") or "") for e in events if e.get("type") == "assistant" for b in blocks_of(e)
             if b.get("type") == "text"]
    final = str((result or {}).get("result") or "") or (texts[-1] if texts else "")
    usage = (result or {}).get("usage")
    tokens = {name: int(usage.get(key) or 0) for name, key in TOKEN_KEYS.items()} if isinstance(usage, dict) else None
    init = next((e for e in events if is_init(e)), None)
    first = next((e["message"].get("model") for e in events if e.get("type") == "assistant"
                  and isinstance(e.get("message"), dict) and e["message"].get("model")), None)
    cost = (result or {}).get("total_cost_usd")
    denials = (result or {}).get("permission_denials")
    return {"final": final, "problem": problem_of(result, code, stderr), "turns": (result or {}).get("num_turns"),
            "tokens": tokens, "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
            "model": (init or {}).get("model") or first, "denials": len(denials) if isinstance(denials, list) else 0,
            "used_percent": used_percent_of(events)}


def ask_claude(prompt_text: str, run_dir: Path, work: Path, *, arm: str, home: Path | None = None,
               data: Path | None = None, notes: str | None = None, name: str | None = None, level: str | None = None,
               timeout: float = TIMEOUT_S, deny: tuple[str, ...] = (), bin_dir: Path | None = None) -> dict:
    """Run one exec for `prompt_text` in the working folder `work`, with its files and logs in `run_dir`. The mms arm asks
    the daemon of `home` with the skill and `notes`; the files arm reads `data`. `bin_dir` goes first on PATH."""
    argv = prepare(run_dir, arm, work, home=home, data=data, notes=notes, name=name, level=level)
    (run_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    events: list[dict] = []
    stopped: list[str] = []
    watch = watcher(arm, deny)
    stderr_file = (run_dir / "stderr.txt").open("wb")
    began = time.monotonic()
    process = subprocess.Popen(argv, cwd=work, env=env_of(bin_dir, arm, work), stdin=subprocess.PIPE,
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
            if problem := watch(event):
                stopped.append(problem)
                codex_run.kill_tree(process)
                break

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        process.stdin.write(prompt_text.encode("utf-8"))
        process.stdin.close()
    except OSError:
        pass  # an exec that died before reading its prompt is told by its exit
    timed_out = None
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        codex_run.kill_tree(process)
        process.wait()
        timed_out = f"timeout: no answer in {timeout:g} s"
    reader.join(30)
    stderr_file.close()
    seconds = round(time.monotonic() - began, 1)
    stderr = (run_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-2000:]
    done = result_of(events, process.returncode, stderr)
    problem = stopped[0] if stopped else timed_out or done["problem"]
    out = {"answer": codex_run.answer_of(done["final"]), "final": done["final"], "tools": tools_of(events),
           "events": [codex_run.scrubbed(e) for e in events], "seconds": seconds, "turns": done["turns"],
           "tokens": done["tokens"], "cost_usd": done["cost_usd"], "used_percent": done["used_percent"],
           "model": done["model"], "denials": done["denials"], "problem": problem, "argv": argv, "stderr": stderr}
    if init := next((e for e in events if is_init(e)), None):
        (run_dir / "init.json").write_text(json.dumps(codex_run.scrubbed(init), indent=1), encoding="utf-8")
    (run_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in out["events"]), encoding="utf-8")
    (run_dir / "tools.json").write_text(json.dumps(out["tools"], indent=1), encoding="utf-8")
    (run_dir / "final.txt").write_text(done["final"], encoding="utf-8")
    return out
