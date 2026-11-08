"""`mms install` (DESIGN.md §14, TESTBED.md item 13) as a process, with every path in `tmp_path` and a fake `claude`.

Nothing here touches the real ~/.claude, ~/.codex, Startup folder or PATH: the environment the installer reads is
rebuilt in `tmp_path`, and the real places are checked unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path, PurePosixPath

import pytest
from fake_hub import FakeHub
from mcp_session import drive, text

from mms import config as configuration
from mms import decider_d1, install, modelenv

SOURCE = Path(install.SKILL_SOURCE)
WINDOWS = os.name == "nt"
MMS = Path(sys.executable).parent / ("mms.exe" if WINDOWS else "mms")
# the fake `claude`: records its arguments; a registered server is a file, which `get` shows as `claude mcp get` does
RECORDER = '''
import json, os, sys
from pathlib import Path
state = Path(os.environ["FAKE_CLAUDE_DIR"])
args = sys.argv[1:]
with open(state / "calls.jsonl", "a") as log:
    log.write(json.dumps(args) + "\\n")
registered = state / "registered"
mode = os.environ.get("FAKE_CLAUDE_MODE", "")
if args[:2] == ["mcp", "get"]:
    if not registered.exists():
        print('No MCP server named "mms". Run `claude mcp add` to add one.')
        sys.exit(1)
    command = json.loads(registered.read_text())
    command = command[command.index("--") + 1:]
    print("mms:\\n  Scope: User config (available in all your projects)\\n  Status: connected\\n  Type: stdio")
    print("  Command: " + command[0] + "\\n  Args: " + " ".join(command[1:]) + "\\n  Environment:")
elif args[:2] == ["mcp", "remove"]:
    if mode == "remove-fails":
        print("error: could not remove it", file=sys.stderr)
        sys.exit(2)
    registered.unlink()
elif args[:2] == ["mcp", "add"]:
    if mode == "add-fails":
        print("error: invalid configuration", file=sys.stderr)
        sys.exit(3)
    registered.write_text(json.dumps(args))
'''


class Setup:
    """The fake machine: home folders, the environment pointing at them, and the fake claude."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.user = tmp / "user"
        self.codex = tmp / "codex-home"
        self.appdata = tmp / "appdata"
        self.xdg = tmp / "xdg"
        self.fake_dir = tmp / "fake-claude-state"
        for folder in (self.user, self.fake_dir):
            folder.mkdir()
        recorder = tmp / "recorder.py"
        recorder.write_text(RECORDER)
        bin_dir = tmp / "fakebin"
        bin_dir.mkdir()
        if WINDOWS:
            self.claude = bin_dir / "claude.cmd"
            self.claude.write_text(f'@echo off\r\n"{sys.executable}" "{recorder}" %*\r\nexit /b %ERRORLEVEL%\r\n')
        else:
            self.claude = bin_dir / "claude"
            self.claude.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{recorder}" "$@"\n')
            self.claude.chmod(0o755)
        self.env = dict(os.environ, USERPROFILE=str(self.user), HOME=str(self.user), APPDATA=str(self.appdata),
                        CODEX_HOME=str(self.codex), XDG_CONFIG_HOME=str(self.xdg), FAKE_CLAUDE_DIR=str(self.fake_dir),
                        PYTHONUTF8="1", HF_HUB_OFFLINE="1")  # a test never downloads
        for name in ("MMS_HOME", "FAKE_CLAUDE_MODE"):
            self.env.pop(name, None)

    def run(self, *args: str, claude: bool = True, daemon: bool = False, **env: str) -> subprocess.CompletedProcess:
        """`mms install` as a process. It starts no daemon unless asked: the other steps are what these tests are about."""
        if "--only" not in args and not daemon:
            args = ("--only", "claude", "codex", "skills", "login", *args)
        command = [sys.executable, "-m", "mms.install", *args]
        if claude:
            command += ["--claude-exe", str(self.claude)]
        return subprocess.run(command, env={**self.env, **env}, capture_output=True, text=True, encoding="utf-8",
                              timeout=120)

    def calls(self) -> list[list[str]]:
        file = self.fake_dir / "calls.jsonl"
        return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []

    @property
    def login_file(self) -> Path:
        if WINDOWS:
            return self.appdata / "Microsoft/Windows/Start Menu/Programs/Startup/mms.cmd"
        return self.xdg / "autostart" / "mms.desktop"

    @property
    def skill_dirs(self) -> list[Path]:
        return [self.user / ".claude" / "skills" / "mms-questions", self.codex / "skills" / "mms-questions"]

    def files(self) -> dict[str, str]:
        """Every file under the fake machine, by relative path, with a hash (the fake claude's own records apart)."""
        skip = {self.fake_dir, self.tmp / "recorder.py", self.tmp / "fakebin"}
        out = {}
        for path in sorted(self.tmp.rglob("*")):
            if path.is_file() and not any(parent in skip for parent in (path, *path.parents)):
                out[path.relative_to(self.tmp).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out


@pytest.fixture
def machine(tmp_path) -> Setup:
    return Setup(tmp_path)


def real_places() -> dict[Path, int | None]:
    """The real files and folders an install must never touch, with their modification times (None: absent)."""
    real_home = Path(os.path.expanduser("~"))
    places = [real_home / ".claude" / "skills" / "mms-questions", real_home / ".codex" / "config.toml",
              real_home / ".codex" / "skills" / "mms-questions",
              real_home / ".config" / "autostart" / "mms.desktop", real_home / ".mms" / "mms.toml"]
    if os.environ.get("APPDATA"):
        places.append(Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup/mms.cmd")
    return {place: place.stat().st_mtime_ns if place.exists() else None for place in places}


def status_lines(done: subprocess.CompletedProcess) -> dict[str, str]:
    return {line.split(":")[0]: line.split(":", 1)[1].strip() for line in done.stdout.splitlines()}


def skill_files(folder: Path) -> dict[str, bytes]:
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def test_install_registers_copies_and_logs_in_and_writes_nothing_outside_tmp_path(machine, tmp_path):
    before = real_places()
    done = machine.run()
    assert done.returncode == 0, done.stdout + done.stderr
    assert {step: text.split(" ")[0] for step, text in status_lines(done).items()} == {
        "claude": "done", "codex": "done", "skills": "done", "login": "done"}

    # claude: remove, then add, with the mms script beside this python
    assert machine.calls() == [["mcp", "get", "mms"], ["mcp", "add", "--scope", "user", "mms", "--", str(MMS), "mcp"]]

    # codex: a table for the server
    config = tomllib.loads((machine.codex / "config.toml").read_text(encoding="utf-8"))
    assert config == {"mcp_servers": {"mms": {"command": str(MMS), "args": ["mcp"], "default_tools_approval_mode": "approve"}}}

    # the skill in both skills folders, as shipped
    expected = skill_files(SOURCE)
    assert "SKILL.md" in expected
    for folder in machine.skill_dirs:
        assert skill_files(folder) == expected

    # the login entry
    entry = machine.login_file.read_text(encoding="oem" if WINDOWS else "utf-8")
    assert (f'start "" /min "{MMS}" start --detach' in entry) if WINDOWS else (f"Exec={MMS} start --detach" in entry)

    # every file the install made is under tmp_path, and these are all of them
    names = set(machine.files())
    assert names == {"codex-home/config.toml",
                     *(f"{folder.relative_to(tmp_path).as_posix()}/{name}" for folder in machine.skill_dirs
                       for name in expected),
                     machine.login_file.relative_to(tmp_path).as_posix()}
    assert real_places() == before  # and the real ~/.claude, ~/.codex and Startup folder are as they were


def test_a_set_mms_home_is_passed_on_to_each_registration(machine, tmp_path):
    home = tmp_path / "my home"
    done = machine.run(MMS_HOME=str(home))
    assert done.returncode == 0, done.stdout + done.stderr
    assert machine.calls()[1] == ["mcp", "add", "--scope", "user", "mms", "-e", f"MMS_HOME={home}", "--", str(MMS),
                                  "mcp"]
    config = tomllib.loads((machine.codex / "config.toml").read_text(encoding="utf-8"))
    assert config["mcp_servers"]["mms"]["env"] == {"MMS_HOME": str(home)}
    entry = machine.login_file.read_text(encoding="oem" if WINDOWS else "utf-8")
    assert f"MMS_HOME={home}" in entry
    # unset again: the registration drops it
    machine.run()
    assert "env" not in tomllib.loads((machine.codex / "config.toml").read_text(encoding="utf-8"))["mcp_servers"]["mms"]


CODEX_BEFORE = """\
# my codex settings
model = "gpt-5"   # the default

[mcp_servers.other]
command = "other-server"   # keep this one
args = ["--flag"]

[profiles.fast]
model = "o4-mini"
"""


def test_the_codex_file_keeps_its_other_tables_and_comments_and_removal_restores_it(machine):
    (machine.codex).mkdir()
    file = machine.codex / "config.toml"
    file.write_bytes(CODEX_BEFORE.replace("\n", "\r\n").encode())  # its own line endings too
    assert machine.run("--only", "codex").returncode == 0
    text = file.read_bytes().decode()
    assert "[mcp_servers.mms]" in text and "# my codex settings" in text and "# keep this one" in text
    assert text.startswith(CODEX_BEFORE.split("[profiles.fast]")[0].replace("\n", "\r\n"))  # nothing before it moved
    assert text.endswith('[profiles.fast]\r\nmodel = "o4-mini"\r\n') and "\r\n\r\n[profiles.fast]" in text
    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["other"] == {"command": "other-server", "args": ["--flag"]}
    assert parsed["profiles"] == {"fast": {"model": "o4-mini"}} and parsed["mcp_servers"]["mms"]["args"] == ["mcp"]

    again = machine.run("--only", "codex")
    assert status_lines(again)["codex"].startswith("unchanged") and file.read_bytes().decode() == text

    removed = machine.run("--only", "codex", "--remove")
    assert status_lines(removed)["codex"].startswith("done")
    assert file.read_bytes().decode().rstrip() == CODEX_BEFORE.replace("\n", "\r\n").rstrip()
    gone = machine.run("--only", "codex", "--remove")
    assert status_lines(gone)["codex"].startswith("unchanged")


def test_a_codex_entry_with_the_owners_own_keys_keeps_them(machine):
    machine.codex.mkdir()
    (machine.codex / "config.toml").write_text('[mcp_servers.mms]\ncommand = "old"\nstartup_timeout_sec = 20\n')
    assert machine.run("--only", "codex").returncode == 0
    table = tomllib.loads((machine.codex / "config.toml").read_text())["mcp_servers"]["mms"]
    assert table == {"command": str(MMS), "args": ["mcp"], "startup_timeout_sec": 20, "default_tools_approval_mode": "approve"}


def test_an_approval_mode_the_owner_chose_stays_through_install_and_remove(machine):
    machine.codex.mkdir()
    (machine.codex / "config.toml").write_text('[mcp_servers.mms]\ndefault_tools_approval_mode = "prompt"\n')
    assert machine.run("--only", "codex").returncode == 0
    table = tomllib.loads((machine.codex / "config.toml").read_text())["mcp_servers"]["mms"]
    assert table["default_tools_approval_mode"] == "prompt" and table["args"] == ["mcp"]
    assert machine.run("--only", "codex", "--remove").returncode == 0
    assert tomllib.loads((machine.codex / "config.toml").read_text())["mcp_servers"] == {
        "mms": {"default_tools_approval_mode": "prompt"}}


def test_a_codex_file_that_does_not_parse_is_a_failure_and_is_left_alone(machine):
    machine.codex.mkdir()
    (machine.codex / "config.toml").write_text("this is [not toml")
    done = machine.run("--only", "codex")
    assert done.returncode == 1 and status_lines(done)["codex"].startswith("failed")
    assert (machine.codex / "config.toml").read_text() == "this is [not toml"


def test_claude_is_asked_before_it_is_changed_and_only_our_own_entry_is_replaced(machine):
    first = machine.run("--only", "claude")
    assert first.returncode == 0 and [call[1] for call in machine.calls()] == ["get", "add"]  # get said: not there
    second = machine.run("--only", "claude")  # now it is there, and ours: remove, then add
    assert second.returncode == 0 and [call[1] for call in machine.calls()] == ["get", "add", "get", "remove", "add"]

    stopped = machine.run("--only", "claude", FAKE_CLAUDE_MODE="remove-fails")
    assert stopped.returncode == 1 and "claude: failed" in stopped.stdout and "could not remove" in stopped.stdout
    assert [call[1] for call in machine.calls()][-1] == "remove"  # no add after a failed remove

    broken = machine.run("--only", "claude", FAKE_CLAUDE_MODE="add-fails")
    assert broken.returncode == 1 and "invalid configuration" in broken.stdout


def test_a_server_that_claude_mcp_get_does_not_show_is_not_registered(machine):
    broken = machine.run("--only", "claude", FAKE_CLAUDE_MODE="add-fails")  # a broken claude shows up as add's failure
    assert broken.returncode == 1 and "claude mcp add exited 3" in broken.stdout and "invalid configuration" in broken.stdout
    assert [call[1] for call in machine.calls()] == ["get", "add"]
    gone = machine.run("--only", "claude", "--remove")
    assert gone.returncode == 0 and status_lines(gone)["claude"] == (
        'unchanged (was not registered (claude mcp get: No MCP server named "mms". Run `claude mcp add` to add one.))')
    assert [call[1] for call in machine.calls()][-1] == "get", "nothing was removed"


def test_an_mms_server_that_is_not_ours_is_refused_and_left_alone(machine):
    (machine.fake_dir / "registered").write_text(json.dumps(
        ["mcp", "add", "--scope", "user", "mms", "--", "some-other-mms", "serve"]))
    done = machine.run("--only", "claude")
    assert done.returncode == 1 and "not MultimediaSearch's" in done.stdout and "some-other-mms" in done.stdout
    assert [call[1] for call in machine.calls()] == ["get"], "neither removed nor replaced"
    assert machine.run("--only", "claude", "--remove").returncode == 1
    (machine.fake_dir / "registered").write_text(json.dumps(  # our command with other arguments is not ours either
        ["mcp", "add", "--scope", "user", "mms", "--", str(MMS), "serve"]))
    assert machine.run("--only", "claude").returncode == 1


def test_removing_the_codex_entry_keeps_the_owners_other_keys(machine):
    machine.codex.mkdir()
    (machine.codex / "config.toml").write_text("[mcp_servers.mms]\nstartup_timeout_sec = 20\n")
    home = str(machine.tmp / "h")
    assert machine.run("--only", "codex", MMS_HOME=home).returncode == 0
    table = tomllib.loads((machine.codex / "config.toml").read_text())["mcp_servers"]["mms"]
    assert table == {"startup_timeout_sec": 20, "command": str(MMS), "args": ["mcp"], "env": {"MMS_HOME": home},
                     "default_tools_approval_mode": "approve"}
    assert machine.run("--only", "codex", "--remove").returncode == 0
    assert tomllib.loads((machine.codex / "config.toml").read_text())["mcp_servers"] == {"mms": {"startup_timeout_sec": 20}}


def test_a_skill_folder_that_is_not_ours_is_left_alone(machine):
    (machine.codex / "skills" / "mms-questions").mkdir(parents=True)
    (machine.codex / "skills" / "mms-questions" / "SKILL.md").write_text("---\nname: something-else\n---\nmine")
    done = machine.run("--only", "skills")
    assert done.returncode == 1 and "not MultimediaSearch's skill" in done.stdout
    assert (machine.codex / "skills" / "mms-questions" / "SKILL.md").read_text().endswith("mine")
    assert machine.run("--only", "skills", "--remove").returncode == 1


def test_install_starts_the_daemon_once_and_it_is_not_our_child(machine, home, detached):
    from mms import client

    done = machine.run("--only", "daemon", daemon=True, MMS_HOME=str(home))
    assert done.returncode == 0 and status_lines(done)["daemon"].startswith("done"), done.stdout + done.stderr
    first = client.connect(home)
    assert first is not None and first.pid != os.getpid()
    again = machine.run("--only", "daemon", daemon=True, MMS_HOME=str(home))
    assert status_lines(again)["daemon"].startswith("unchanged") and client.connect(home) == first
    removed = machine.run("--only", "daemon", "--remove", daemon=True, MMS_HOME=str(home))
    assert status_lines(removed)["daemon"].startswith("unchanged") and client.connect(home) == first, "left running"


def test_one_failed_step_does_not_stop_the_others_but_fails_the_run(machine):
    done = machine.run(FAKE_CLAUDE_MODE="add-fails")
    steps = status_lines(done)
    assert done.returncode == 1 and steps["claude"].startswith("failed")
    assert [steps[name].split(" ")[0] for name in ("codex", "skills", "login")] == ["done"] * 3


def test_without_claude_on_path_the_claude_step_is_skipped_with_why(machine, tmp_path):
    (tmp_path / "empty").mkdir()
    done = machine.run("--only", "claude", claude=False, PATH=str(tmp_path / "empty"))
    assert done.returncode == 0 and status_lines(done)["claude"] == "skipped (claude not on PATH)"
    assert machine.calls() == []


def test_running_it_twice_changes_nothing_more(machine):
    assert machine.run().returncode == 0
    after_first = machine.files()
    second = machine.run()
    assert second.returncode == 0
    assert machine.files() == after_first
    steps = status_lines(second)
    assert [steps[name].split(" ")[0] for name in ("codex", "skills", "login")] == ["unchanged"] * 3
    assert [call[1] for call in machine.calls()] == ["get", "add", "get", "remove", "add"]  # asked again, same effect


def test_an_older_copy_of_the_skill_is_replaced_whole(machine):
    for folder in machine.skill_dirs:
        (folder / "old").mkdir(parents=True)
        (folder / "old" / "stale.md").write_text("from an older version")
        (folder / "SKILL.md").write_text("---\nname: mms-questions\n---\nan older skill")
    done = machine.run("--only", "skills")
    assert done.returncode == 0 and status_lines(done)["skills"].startswith("done")
    for folder in machine.skill_dirs:
        assert skill_files(folder) == skill_files(SOURCE)


def test_remove_undoes_each_step(machine):
    assert machine.run().returncode == 0
    done = machine.run("--remove")
    assert done.returncode == 0, done.stdout + done.stderr
    assert [step.split(" ")[0] for step in status_lines(done).values()] == ["done"] * 4
    assert machine.calls()[-1] == ["mcp", "remove", "--scope", "user", "mms"]
    assert (machine.fake_dir / "registered").exists() is False
    assert not any(folder.exists() for folder in machine.skill_dirs) and not machine.login_file.exists()
    assert tomllib.loads((machine.codex / "config.toml").read_text()) == {}
    again = machine.run("--remove")
    assert [step.split(" ")[0] for step in status_lines(again).values()] == ["unchanged"] * 4


def test_only_picks_the_steps(machine):
    done = machine.run("--only", "skills", "login")
    assert done.returncode == 0 and set(status_lines(done)) == {"skills", "login"}
    assert machine.calls() == [] and not (machine.codex / "config.toml").exists()


@pytest.mark.skipif(not WINDOWS, reason="the Startup folder is Windows's")
def test_a_missing_appdata_fails_the_login_step_only(machine):
    env = dict(machine.env)
    env.pop("APPDATA")
    done = subprocess.run([sys.executable, "-m", "mms.install", "--only", "login", "skills"], env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    steps = status_lines(done)
    assert done.returncode == 1 and steps["login"].startswith("failed") and "APPDATA" in steps["login"]
    assert steps["skills"].startswith("done")


def test_the_linux_login_entry_quotes_what_needs_it():
    entry = install.login_text(False, PurePosixPath("/opt/my venv/bin/mms"), "/data/mms home")
    assert entry.splitlines()[0] == "[Desktop Entry]"
    assert 'Exec=env "MMS_HOME=/data/mms home" "/opt/my venv/bin/mms" start --detach' in entry
    plain = install.login_text(False, PurePosixPath("/venv/bin/mms"), None)
    assert "Exec=/venv/bin/mms start --detach\n" in plain
    assert install.login_file(False) == Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") \
        / "autostart" / "mms.desktop"


def test_the_registered_command_line_starts_the_adapter_and_it_answers(machine, home, memory):
    """`mms install` through the `mms` command (its `--home` is registered), then the command line it registered is the one a
    client starts."""
    done = subprocess.run([sys.executable, "-m", "mms.cli", "--home", str(home), "install", "--only", "claude", "codex",
                           "--claude-exe", str(machine.claude)], env=machine.env, capture_output=True, text=True,
                          encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    registered = json.loads((machine.fake_dir / "registered").read_text())
    assert registered[registered.index("-e") + 1] == f"MMS_HOME={home}", "--home is what is registered"
    command = registered[registered.index("--") + 1:]
    codex = tomllib.loads((machine.codex / "config.toml").read_text(encoding="utf-8"))["mcp_servers"]["mms"]
    assert command == [str(MMS), "mcp"] == [codex["command"], *codex["args"]] and codex["env"] == {"MMS_HOME": str(home)}

    async def steps(client):
        started = time.perf_counter()
        sources = text(await client.call_tool("sources", {}))
        return (await client.list_tools()).tools, sources, time.perf_counter() - started

    tools, sources, took = drive(steps, Path(codex["env"]["MMS_HOME"]), machine.tmp, command=command)
    assert len(tools) == 7 and sources["home"] == str(home) and sources["running"] is True and took < 20


# the model step: `install.main` in this process, the home in tmp_path, and the Hub faked (nothing is downloaded)

MMS_ENV = {name: pin or "1.0.0+cu128" for name, pin in map(modelenv.pinned, decider_d1.D1.requirements)}  # d1's pins, met


@pytest.fixture
def hub(monkeypatch) -> FakeHub:
    return FakeHub(monkeypatch)


@pytest.fixture
def modelling(home, hub, monkeypatch):
    """MMS_HOME in tmp_path, mms's own env meeting d1's pins, and the real places checked unchanged."""
    before = real_places()
    monkeypatch.setenv("MMS_HOME", str(home))
    monkeypatch.setattr(modelenv, "installed", lambda: dict(MMS_ENV))
    yield home
    assert real_places() == before


def install_model(capsys, *args: str) -> tuple[int, list[str], str]:
    """`mms install --only model ...`: its exit code, the lines of stdout and the text of stderr."""
    code = install.main(["--only", "model", *args])
    printed = capsys.readouterr()
    return code, printed.out.splitlines(), printed.err


def test_the_steps_end_with_the_model_and_then_the_daemon():
    assert install.STEPS == ("claude", "codex", "skills", "login", "model", "daemon")
    assert install.DEFAULT_MODEL == "d1-3B"


def test_a_fresh_install_downloads_d1_3b_and_says_it_hears_nothing(modelling, hub, capsys):
    folder = modelling / "models" / "d1-3B"
    code, out, err = install_model(capsys)
    assert code == 0
    assert out == [f"model: done (d1-3B in {folder}; it hears nothing: mms model use d1-Omni-2B gives one that hears)"]
    assert hub.calls == [("info", "LiquidAI/d1-3B"), ("download", "LiquidAI/d1-3B", folder, hub.commit)]
    assert "downloading LiquidAI/d1-3B at 0000000 (6.3 GB, licence lfm1.0) into" in err, "the work's output is on stderr"
    assert configuration.load(modelling).decider_model == folder.as_posix()


def test_a_model_that_is_configured_stays_and_nothing_is_asked_of_the_hub(modelling, hub, capsys):
    assert install_model(capsys)[0] == 0
    hub.calls.clear()
    before = (modelling / "mms.toml").read_bytes()
    code, out, err = install_model(capsys)
    configured = (modelling / "models" / "d1-3B").as_posix()
    assert code == 0 and err == ""
    assert out == [f"model: unchanged (the configured model stays ({configured}); mms model use NAME switches)"]
    assert hub.calls == [] and (modelling / "mms.toml").read_bytes() == before


def test_a_models_dir_with_a_model_configured_and_no_model_named_fails_the_step_rather_than_being_ignored(
        modelling, hub, tmp_path, capsys):
    assert install_model(capsys)[0] == 0
    hub.calls.clear()
    before = (modelling / "mms.toml").read_bytes()
    code, out, err = install_model(capsys, "--models-dir", str(tmp_path / "big"))
    assert code == 1 and out == ["model: failed (--models-dir says where a download goes, and nothing is downloaded: "
                                 "give --model NAME too)"]
    assert hub.calls == [] and (modelling / "mms.toml").read_bytes() == before and not (tmp_path / "big").exists()


def test_a_model_named_and_a_models_folder_given_are_downloaded_there_and_used(modelling, hub, tmp_path, capsys):
    assert install_model(capsys)[0] == 0  # d1-3B is configured
    hub.calls.clear()
    folder = tmp_path / "big" / "d1-Omni-2B"
    code, out, err = install_model(capsys, "--model", "d1-Omni-2B", "--models-dir", str(tmp_path / "big"))
    assert code == 0 and out == [f"model: done (d1-Omni-2B in {folder})"], "it hears: nothing to say"
    assert hub.calls == [("info", "LiquidAI/d1-Omni-2B"), ("download", "LiquidAI/d1-Omni-2B", folder, hub.commit)]
    assert configuration.load(modelling).decider_model == folder.as_posix()
    hub.calls.clear()
    again = install_model(capsys, "--model", "d1-Omni-2B", "--models-dir", str(tmp_path / "big"))
    assert again[:2] == (0, [f"model: unchanged (d1-Omni-2B in {folder})"]) and hub.calls == [], "it is there and set"


def test_a_model_that_cannot_be_used_fails_the_step_with_why_and_the_hub_is_not_asked(modelling, hub, monkeypatch, capsys):
    monkeypatch.setattr(modelenv, "installed", dict)  # mms's own env holds nothing
    code, out, err = install_model(capsys)
    assert code == 1 and len(out) == 1
    assert out[0].startswith("model: failed (d1-3B runs in mms's own env, which is not ready: ")
    assert "uv sync --group d1" in out[0]
    assert hub.calls == [] and not (modelling / "mms.toml").exists() and not (modelling / "models").exists()
    code, out, err = install_model(capsys, "--model", "nope")
    assert code == 1 and out[0].startswith("model: failed (there is no model folder 'nope' in ") and hub.calls == []


def test_removing_leaves_the_weights_and_the_setting_and_says_so(modelling, hub, capsys):
    code, out, err = install_model(capsys, "--remove")
    assert (code, out) == (0, ["model: unchanged (no model is configured)"])
    assert install_model(capsys)[0] == 0
    before = (modelling / "mms.toml").read_bytes()
    code, out, err = install_model(capsys, "--remove")
    configured = (modelling / "models" / "d1-3B").as_posix()
    assert code == 0 and out == [f"model: unchanged ({configured} stays configured and its weights stay on disk: delete its "
                                 "folder by hand to free the space)"]
    assert (modelling / "models" / "d1-3B" / "model.safetensors").is_file()
    assert (modelling / "mms.toml").read_bytes() == before


@pytest.mark.parametrize("args", [["--only", "claude", "--model", "d3-edge"], ["--only", "claude", "--models-dir", "x"],
                                  ["--only", "model", "--remove", "--model", "d3-edge"], ["--remove", "--models-dir", "x"]])
def test_model_options_are_refused_without_the_model_step_or_with_remove(args, capsys):
    with pytest.raises(SystemExit) as stop:
        install.main(args)
    assert stop.value.code == 2
    assert ("--model and --models-dir are for the model step: leave out --remove, and name model in --only"
            in capsys.readouterr().err)
