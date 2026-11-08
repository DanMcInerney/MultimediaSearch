"""`mms install` (DESIGN.md §14): register the MCP adapter with Claude Code and Codex, copy the skill, start the daemon at
login, download and configure the decision model, and start the daemon now. It changes only what is MultimediaSearch's: an `mms`
server that is another program's, or a skill folder that is another skill's, is refused, not replaced. Every path comes
from the environment when it runs, so a test points them all into a temporary folder.
"""

from __future__ import annotations

import argparse
import contextlib
import filecmp
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tomlkit

from mms import client, fetch, prepare
from mms import config as configuration

SERVER = "mms"  # the name the adapter is registered under
SKILL = "mms-questions"
SKILL_SOURCE = Path(__file__).parent / "skills" / SKILL
STEPS = ("claude", "codex", "skills", "login", "model", "daemon")
DEFAULT_MODEL = "d1-3B"  # what a fresh install downloads and uses: a model must be set, and mms ships no weights
# Codex asks before every MCP tool call unless the server's tools are approved, and `codex exec` refuses them outright.
# mms's tools can neither approve a device, lower a cap nor delete history (DESIGN §4.3, §4.4), so they are approved.
APPROVAL, APPROVE = "default_tools_approval_mode", "approve"


class InstallError(Exception):
    """A step that cannot be done, with why."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mms install", description="Register MultimediaSearch with Claude Code and Codex, "
                                     "install its skill, start its daemon at login, and download and configure its "
                                     "decision model.")
    parser.add_argument("--only", nargs="+", action="extend", choices=STEPS, metavar="STEP",
                        help=f"only these steps: {', '.join(STEPS)} (default: all)")
    parser.add_argument("--remove", action="store_true", help="undo each step")
    parser.add_argument("--claude-exe", metavar="PATH", help="the claude to run (default: claude on PATH)")
    parser.add_argument("--model", metavar="NAME|FOLDER",
                        help=f"the decision model to download and use (default: {DEFAULT_MODEL}, when none is configured)")
    parser.add_argument("--models-dir", metavar="DIR", help="where model folders go (default: the folder of the "
                        "configured model, else <home>/models)")
    args = parser.parse_args(argv)
    if (args.model is not None or args.models_dir is not None) and (
            args.remove or (args.only is not None and "model" not in args.only)):
        parser.error("--model and --models-dir are for the model step: leave out --remove, and name model in --only")
    failed = False
    for step in (s for s in STEPS if args.only is None or s in args.only):
        try:
            status, detail = {"claude": lambda: claude(args.remove, args.claude_exe), "codex": lambda: codex(args.remove),
                              "skills": lambda: skills(args.remove), "login": lambda: login(args.remove),
                              "model": lambda: model(args.remove, args.model, args.models_dir),
                              "daemon": lambda: daemon(args.remove)}[step]()
        except Exception as error:  # reported with the step, and the other steps still run
            status, detail = "failed", str(error) if isinstance(error, InstallError) else f"{type(error).__name__}: {error}"
        failed |= status == "failed"
        print(f"{step}: {status}" + (f" ({detail})" if detail else ""))
    return 1 if failed else 0


# -- what is registered ----------------------------------------------------------------------------------------------

def mms_script() -> Path:
    """The `mms` script beside this Python (the project's venv), so `mms` need not be on PATH."""
    script = Path(sys.executable).parent / ("mms.exe" if os.name == "nt" else "mms")
    if not script.is_file():
        raise InstallError(f"there is no mms script beside {sys.executable}; install the project with `uv sync`")
    return script


def mms_home() -> str | None:
    """`MMS_HOME` as an absolute path when it is set: each registration passes it on."""
    value = os.environ.get("MMS_HOME")
    return os.path.abspath(os.path.expanduser(value)) if value else None


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


# -- steps: each returns (done | unchanged | skipped | failed, why) ----------------------------------------------------

def is_ours(command: str | None, args: list[str] | None) -> bool:
    """Whether a registered server is MultimediaSearch's: our `mms` script, run as `mms mcp`."""
    return (command is not None and os.path.normcase(os.path.abspath(command)) == os.path.normcase(str(mms_script()))
            and args == ["mcp"])


def claude(remove: bool, exe: str | None) -> tuple[str, str]:
    exe = exe or shutil.which("claude")
    if not exe:
        return "skipped", "claude not on PATH"
    shown = _run([exe, "mcp", "get", SERVER])
    was_there = shown.returncode == 0  # a non-zero exit means not registered; its wording is not read
    if was_there:
        fields = dict(re.findall(r"^\s*(Scope|Command|Args):[ \t]*(.*?)\s*$", shown.stdout, re.MULTILINE))
        if fields.get("Scope", "user").lower().startswith("user"):
            if not is_ours(fields.get("Command"), fields.get("Args", "").split()):
                raise InstallError(f"a user-scope MCP server named {SERVER} already exists and is not MultimediaSearch's (it runs "
                                   f"{fields.get('Command')} {fields.get('Args', '')}): remove it with `claude mcp remove "
                                   f"--scope user {SERVER}`, or rename it, and run this again")
            removed = _run([exe, "mcp", "remove", "--scope", "user", SERVER])
            if removed.returncode != 0:
                raise InstallError(f"claude mcp remove exited {removed.returncode}: {removed.stdout.strip()}")
        else:
            was_there = False  # another scope's entry is not the user scope's
    if remove:
        if was_there:
            return "done", "removed"
        return "unchanged", "was not registered" + (f" (claude mcp get: {shown.stdout.strip()})" if shown.returncode else "")
    command = [exe, "mcp", "add", "--scope", "user", SERVER]
    if home := mms_home():
        command += ["-e", f"MMS_HOME={home}"]
    added = _run([*command, "--", str(mms_script()), "mcp"])
    if added.returncode != 0:
        raise InstallError(f"claude mcp add exited {added.returncode}: {added.stdout.strip()}")
    return "done", "registered in user scope"


def codex(remove: bool) -> tuple[str, str]:
    file = codex_home() / "config.toml"
    before = file.read_bytes().decode("utf-8") if file.exists() else ""
    document = tomlkit.parse(before)
    servers = document.get("mcp_servers")
    existing = servers.get(SERVER) if servers is not None else None
    if remove:  # only what install sets goes: command, args, env.MMS_HOME and our approval mode
        if existing is None:
            return "unchanged", f"{file} has no [mcp_servers.{SERVER}]"
        for key in ("command", "args"):
            existing.pop(key, None)
        if existing.get(APPROVAL) == APPROVE:
            del existing[APPROVAL]
        if "env" in existing:
            existing["env"].pop("MMS_HOME", None)
            if not existing["env"]:
                del existing["env"]
        if not existing:
            del servers[SERVER]
        if not servers:
            del document["mcp_servers"]
        _write(file, _line_endings_of(before, document.as_string()))
        return "done", f"removed [mcp_servers.{SERVER}] from {file}"
    if servers is None:
        servers = tomlkit.table(is_super_table=True)
        document["mcp_servers"] = servers
    created = SERVER not in servers
    if created:
        servers[SERVER] = tomlkit.table()
    table = servers[SERVER]  # only command, args, env.MMS_HOME and the approval mode are ours
    table["command"] = str(mms_script())
    table["args"] = ["mcp"]
    if APPROVAL not in table:  # an approval mode the owner chose stays
        table[APPROVAL] = APPROVE
    home = mms_home()
    if home:
        if "env" not in table:
            table["env"] = tomlkit.inline_table()
        table["env"]["MMS_HOME"] = home
    elif "env" in table and "MMS_HOME" in table["env"]:
        del table["env"]["MMS_HOME"]
        if not table["env"]:
            del table["env"]
    if created:
        table.add(tomlkit.nl())  # a blank line before whatever follows
    after = _line_endings_of(before, document.as_string())
    if after == before:
        return "unchanged", f"{file}"
    _write(file, after)
    return "done", f"[mcp_servers.{SERVER}] in {file}"


def skills(remove: bool) -> tuple[str, str]:
    if not remove and not SKILL_SOURCE.is_dir():
        raise InstallError(f"the skill is missing from the package: {SKILL_SOURCE}")
    outcomes = []
    for target in (Path.home() / ".claude" / "skills" / SKILL, codex_home() / "skills" / SKILL):
        if target.exists() and not _is_our_skill(target):
            raise InstallError(f"{target} is not MultimediaSearch's skill (its SKILL.md is not named {SKILL}), so it is left alone")
        if remove:
            changed = target.exists()
            if changed:
                shutil.rmtree(target)
        elif target.is_dir() and _same_tree(SKILL_SOURCE, target):
            changed = False
        else:
            changed = True
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(SKILL_SOURCE, target, ignore=shutil.ignore_patterns("__pycache__"))
        outcomes.append((changed, target))
    status = "done" if any(changed for changed, _ in outcomes) else "unchanged"
    return status, ", ".join(str(target) for _, target in outcomes)


def model(remove: bool, name: str | None, models: str | None) -> tuple[str, str]:
    """Download and configure the decision model (`mms model use`): `name`, else d1-3B when none is configured. A model that
    is configured stays unless `name` is given. Removing keeps the weights and the setting. The work prints to stderr, so
    stdout stays one line per step."""
    home = configuration.default_home()
    configured = configuration.load(home).decider_model
    if remove:
        if configured:
            return "unchanged", (f"{configured} stays configured and its weights stay on disk: delete its folder by hand "
                                 "to free the space")
        return "unchanged", "no model is configured"
    if name is None and configured:
        if models is not None:
            raise InstallError("--models-dir says where a download goes, and nothing is downloaded: give --model NAME too")
        return "unchanged", f"the configured model stays ({configured}); mms model use NAME switches"
    try:
        with contextlib.redirect_stdout(sys.stderr):
            folder, about, changed = prepare.switch(home, name or DEFAULT_MODEL, models, None)
    except (prepare.PrepareError, fetch.FetchError, ValueError) as error:
        raise InstallError(str(error)) from None
    detail = f"{about.name} in {folder}"
    if note := prepare.hears_instead(about):
        detail += f"; it hears nothing: {note}"
    return ("done" if changed else "unchanged"), detail


def daemon(remove: bool) -> tuple[str, str]:
    """Start the daemon now, as the login entry will at the next login. Removing leaves it running: `mms stop` stops it."""
    home = configuration.default_home()
    if remove:
        return "unchanged", "a running daemon is left running; `mms stop` stops it"
    if client.connect(home):
        return "unchanged", f"already running for {home}"
    client.ensure(home)
    return "done", f"started for {home}"


def login(remove: bool) -> tuple[str, str]:
    windows = os.name == "nt"
    file = login_file(windows)
    if remove:
        if not file.exists():
            return "unchanged", f"{file} is not there"
        file.unlink()
        return "done", f"removed {file}"
    text = login_text(windows, mms_script(), mms_home())
    data = text.encode("oem" if windows else "utf-8")  # cmd reads its script in the console's code page
    if file.exists() and file.read_bytes() == data:
        return "unchanged", str(file)
    _write(file, data)
    return "done", str(file)


def login_file(windows: bool) -> Path:
    """The login entry: the Startup folder's `mms.cmd` (Task Scheduler's at-logon trigger needs an
    administrator, the Startup folder does not), or an XDG autostart entry."""
    if windows:
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise InstallError("APPDATA is not set, so the Startup folder cannot be found")
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "mms.cmd"
    config = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(config) / "autostart" / "mms.desktop"


def login_text(windows: bool, mms: Path, home: str | None) -> str:
    if windows:
        lines = ["@echo off"]
        if home:
            lines.append(f'set "MMS_HOME={home}"')
        lines.append(f'start "" /min "{mms}" start --detach')
        return "\r\n".join(lines) + "\r\n"
    command = f"{_quote(str(mms))} start --detach"
    if home:
        command = f"env {_quote('MMS_HOME=' + home)} {command}"
    return ("[Desktop Entry]\nType=Application\nName=MultimediaSearch\nComment=Starts the MultimediaSearch daemon\n"
            f"Exec={command}\nTerminal=false\nX-GNOME-Autostart-enabled=true\n")


# -- helpers --------------------------------------------------------------------------------------------------------

def _run(command: list[str]) -> subprocess.CompletedProcess:
    """A command's exit code and its output (stdout and stderr together)."""
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL, timeout=60)


def _write(file: Path, content: str | bytes) -> None:
    """Replace `file` whole (a `.part` beside it, then a rename), keeping its line endings as given."""
    file.parent.mkdir(parents=True, exist_ok=True)
    part = file.with_name(file.name + ".part")
    part.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    os.replace(part, file)


def _line_endings_of(original: str, text: str) -> str:
    """`text` with the line endings `original` has (tomlkit writes what it adds with LF)."""
    crlf, lf = "\r\n", "\n"
    return text.replace(crlf, lf).replace(lf, crlf) if crlf in original else text


def _is_our_skill(folder: Path) -> bool:
    """A real folder (not a link) whose SKILL.md names MultimediaSearch's skill; an empty folder is nobody's."""
    if folder.is_symlink() or not folder.is_dir():
        return False
    if not any(folder.iterdir()):
        return True
    try:
        text = (folder / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return False
    return re.search(rf"^name:\s*{re.escape(SKILL)}\s*$", text, re.MULTILINE) is not None


def _same_tree(left: Path, right: Path) -> bool:
    def files(root: Path) -> set[Path]:
        return {path.relative_to(root) for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}

    names = files(left)
    return names == files(right) and all(filecmp.cmp(left / name, right / name, shallow=False) for name in names)


def _quote(argument: str) -> str:
    """One argument of a desktop entry's Exec line."""
    if not re.search(r'[\s"\'\\<>~|&;$*?#()`]', argument):
        return argument
    return '"' + re.sub(r'(["`$\\])', r"\\\1", argument) + '"'


if __name__ == "__main__":
    sys.exit(main())
