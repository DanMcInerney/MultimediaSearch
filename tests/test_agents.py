"""The benchmark's two agents, Claude Code and Codex, in both arms (TESTBED.md, Benchmark): the command lines, the checks of
what a session may use, and the reading of a stream. No real `claude` or `codex` runs: a command line is built and read, an
event is made up, and the few tests that start a process start a stand-in script that prints made-up events."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import claude_run
import codex_run
import pytest

NOTES = {name: (codex_run.NOTES_DIR / name).read_text(encoding="utf-8") for name in ("d1.md", "d1_3b.md", "d3.md")}
SKILL_TEXT = claude_run.SKILL.read_text(encoding="utf-8")
CONTEXT_FULL = ("Codex ran out of room in the model's context window. Start a new thread or clear earlier history before "
                "retrying.")  # as codex.exe words it
STAND_IN = """\
import json, os, sys, time
for line in open(sys.argv[1], encoding="utf-8"):
    sys.stdout.write(line)
    sys.stdout.flush()
json.dump({"env": dict(os.environ), "cwd": os.getcwd()}, open(sys.argv[3], "w"))
time.sleep(float(sys.argv[2]))
"""


def after(argv: list[str], flag: str) -> list[str]:
    """The values that follow `flag`, up to the next flag."""
    start = argv.index(flag) + 1
    return argv[start:next((i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv))]


def settings_of(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]


def disabled_of(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--disable"]


def lower(path: Path) -> str:
    return str(path).lower().replace("\\", "/")


# -- Claude: the command lines and the environment ----------------------------------------------------------------------

def test_claudes_command_lines_give_each_arm_its_tools_and_nothing_of_the_owners(tmp_path, monkeypatch):
    run, work, home, data = tmp_path / "run", tmp_path / "w0", tmp_path / "home", tmp_path / "data"
    argv = claude_run.prepare(run, "mms", work, home=home, notes="d1_3b.md", command=["claude"])
    assert argv[:3] == ["claude", "-p", "--verbose"] and after(argv, "--output-format") == ["stream-json"]
    for flag in ("--restricted", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in argv
    assert after(argv, "--permission-mode") == ["dontAsk"] and "--add-dir" not in argv
    assert after(argv, "--tools") == [""], "no built-in tool: only the MCP server's"
    assert after(argv, "--allowedTools") == ["mcp__mms__ask", "mcp__mms__find", "mcp__mms__look", "mcp__mms__label"]
    assert after(argv, "--mcp-config") == [str(run / "mcp.json")]
    assert after(argv, "--settings") == [str(run / "settings.json")]
    assert json.loads((run / "settings.json").read_text()) == {
        "disableAllHooks": True, "enabledPlugins": {"cc-plugin-agents-md@builtin": False,
                                                    "cc-plugin-plugin-authoring@builtin": False}}
    servers = json.loads((run / "mcp.json").read_text())["mcpServers"]
    assert list(servers) == ["mms"], "one server"
    assert servers["mms"] == {"command": sys.executable, "env": {"CLAUDE_PROJECT_DIR": str(work)},
                             "args": ["-m", "mms.cli", "--home", str(home), "mcp", "--tools", "ask,find,look,label"]}
    assert after(argv, "--append-system-prompt-file") == [str(run / "instructions.md")]
    assert (run / "instructions.md").read_text() == claude_run.instructions("mms", "d1_3b.md")
    assert work.is_dir()
    with pytest.raises(ValueError, match="home"):
        claude_run.prepare(tmp_path / "other", "mms", work, command=["claude"])

    argv = claude_run.prepare(tmp_path / "files-run", "files", work, data=data, command=["claude"])
    assert after(argv, "--tools") == ["Read,Glob,Grep,Bash"]
    assert after(argv, "--allowedTools") == ["Read", "Glob", "Grep", "Bash(ffmpeg *)"]
    assert after(argv, "--add-dir") == [str(data)]
    assert "--mcp-config" not in argv and "--strict-mcp-config" in argv, "no MCP server at all"
    assert not (tmp_path / "files-run" / "mcp.json").exists()
    assert (tmp_path / "files-run" / "instructions.md").read_text() == claude_run.BASE_FILES
    with pytest.raises(ValueError, match="data"):
        claude_run.argv_of(["claude"], "files", run, name="sonnet", level="high")

    # the two flags `claude --help` leaves out each have one switch
    assert after(argv, "--max-turns") == ["60"] and "--max-budget-usd" in argv
    monkeypatch.setattr(claude_run, "PROMPT_FILE", False)
    monkeypatch.setattr(claude_run, "MAX_TURNS", None)
    argv = claude_run.argv_of(["claude"], "files", tmp_path / "files-run", name="sonnet", level="high", data=data)
    assert after(argv, "--append-system-prompt") == [claude_run.BASE_FILES]
    assert "--append-system-prompt-file" not in argv and "--max-turns" not in argv
    # the text of any model's notes still fits the command line
    for notes in sorted(p.name for p in codex_run.NOTES_DIR.glob("*.md")):
        claude_run.prepare(tmp_path / f"in-the-line-{notes}", "mms", work, home=home, notes=notes, command=["claude"])


def test_claudes_environment_drops_the_api_key_and_the_instructions_hold_the_skill_and_the_notes(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-test")
    bin_dir = tmp_path / "bin"
    env = claude_run.env_of(bin_dir)
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
    assert all(env[k] == v for k, v in claude_run.ENV_SET.items())
    assert env["PATH"].startswith(str(bin_dir) + os.pathsep) and os.environ["ANTHROPIC_API_KEY"] == "sk-test"
    assert claude_run.env_of(None)["PATH"] == os.environ["PATH"]

    text = claude_run.instructions("mms", "d1_3b.md")
    assert SKILL_TEXT in text and NOTES["d1_3b.md"] in text and NOTES["d1.md"] not in text
    assert claude_run.instructions("mms", None) == f"{claude_run.BASE_MMS}\n\n{SKILL_TEXT}"
    assert claude_run.instructions("files", "d1_3b.md") == claude_run.BASE_FILES
    assert SKILL_TEXT not in claude_run.instructions("files") and NOTES["d1.md"] not in claude_run.instructions("files")
    with pytest.raises(ValueError, match="plain file name"):
        claude_run.instructions("mms", "../SKILL.md")


@pytest.fixture
def fake_git(tmp_path, monkeypatch):
    """A Git for Windows folder tree, with `claude_run.git_root` pointed at it: the real one is the owner's."""
    root = tmp_path / "Git"
    for sub in ("cmd", "bin", "usr/bin"):
        (root / sub).mkdir(parents=True)
    (root / "bin" / "bash.exe").write_text("")
    (root / "usr" / "bin" / "bash.exe").write_text("")
    monkeypatch.setattr(claude_run, "git_root", lambda: root)
    return root


def test_claudes_files_arm_shell_cannot_reach_the_owners_mms(tmp_path, monkeypatch, fake_git):
    """Under `uv run` PATH starts with the repo's venv folder (mms.exe), and MMS_HOME may be the owner's."""
    venv, bin_dir, work = tmp_path / ".venv" / "Scripts", tmp_path / "bin", tmp_path / "w0"
    venv.mkdir(parents=True)
    (venv / "mms.exe").write_text("")
    (venv / "mms.exe").chmod(0o755)
    monkeypatch.setenv("PATH", str(venv) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
    monkeypatch.setenv("MMS_HOME", str(tmp_path / "owners-home"))
    monkeypatch.setenv("MMS_MODELS", "C:/models")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert shutil.which("mms.exe", path=os.environ["PATH"]), "the owner's PATH does reach mms"

    env = claude_run.env_of(bin_dir, "files", work)
    folders = env["PATH"].split(os.pathsep)
    assert folders[0] == str(bin_dir)
    assert folders[1:4] == [str(fake_git / "cmd"), str(fake_git / "bin"), str(fake_git / "usr" / "bin")]
    assert folders[4:] == codex_run.system_path(), "then the system's folders and nothing else"
    assert not any(".venv" in f for f in folders) and shutil.which("mms.exe", path=env["PATH"]) is None
    assert env["MMS_HOME"] == str(work / "no-mms-home") and not Path(env["MMS_HOME"]).exists()
    assert not {"VIRTUAL_ENV", "MMS_MODELS", "ANTHROPIC_API_KEY"} & set(env)
    assert env["CLAUDE_CODE_GIT_BASH_PATH"] == str(fake_git / "bin" / "bash.exe")
    assert all(env[k] == v for k, v in claude_run.ENV_SET.items())

    env = claude_run.env_of(bin_dir)
    assert str(venv) in env["PATH"] and env["MMS_HOME"] == str(tmp_path / "owners-home"), "the mms arm is as it was"
    with pytest.raises(ValueError, match="working folder"):
        claude_run.env_of(bin_dir, "files")
    if os.name == "nt":
        monkeypatch.setattr(claude_run, "git_root", lambda: None)
        with pytest.raises(claude_run.ClaudeMissing, match="Git for Windows"):
            claude_run.env_of(bin_dir, "files", work)


@pytest.mark.skipif(os.name != "nt", reason="Git for Windows")
def test_git_for_windows_is_found_from_git_or_bash_and_not_from_wsls_bash(tmp_path):
    root, system = tmp_path / "Program Files" / "Git", tmp_path / "Windows" / "System32"
    for sub in ("cmd", "mingw64/bin", "usr/bin"):
        (root / sub).mkdir(parents=True)
    system.mkdir(parents=True)
    for exe in ("cmd/git.exe", "mingw64/bin/git.exe", "usr/bin/bash.exe"):
        (root / exe).write_text("")
    (system / "bash.exe").write_text("")
    assert claude_run.git_root(str(root / "cmd")) == root and claude_run.git_root(str(root / "mingw64" / "bin")) == root
    assert claude_run.git_root(str(system) + os.pathsep + str(root / "usr" / "bin")) == root, "from bash when no git"
    assert claude_run.git_root(str(system)) is None, "the bash of System32 is WSL's, not Git's"


# -- Claude: the stream ------------------------------------------------------------------------------------------------

def init(tools=claude_run.MMS_TOOLS, servers=({"name": "mms", "status": "connected"},), **extra) -> dict:
    return {"type": "system", "subtype": "init", "tools": list(tools), "mcp_servers": list(servers), "plugins": [],
            "model": "claude-sonnet-x", "apiKeySource": "none", **extra}


def use(name: str, arguments: dict, id: str = "t1") -> dict:
    return {"type": "assistant", "message": {"model": "claude-sonnet-x", "content": [
        {"type": "tool_use", "id": id, "name": name, "input": arguments}]}}


def says(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def finished(**fields) -> dict:
    return {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "result": "It was 47.\nANSWER: 47",
            "total_cost_usd": 0.0421, "permission_denials": [],
            "usage": {"input_tokens": 120, "output_tokens": 45, "cache_read_input_tokens": 3000,
                      "cache_creation_input_tokens": 800}, **fields}


def test_a_session_that_is_not_the_arms_is_a_setup_problem_and_a_tool_out_of_bounds_an_escape(tmp_path):
    files = {"tools": claude_run.FILES_TOOLS, "servers": ()}
    assert claude_run.check_init(init(), "mms") is None and claude_run.check_init(init(**files), "files") is None
    assert "Bash" in claude_run.check_init(init(tools=[*claude_run.MMS_TOOLS, "Bash"]), "mms")
    assert "mcp__mms__ask" in claude_run.check_init(init(tools=claude_run.MMS_TOOLS[1:]), "mms")
    extra = [{"name": "mms", "status": "connected"}, {"name": "claude.ai Gmail", "status": "connected"}]
    assert "Gmail" in claude_run.check_init(init(servers=extra), "mms")
    assert "not connected" in claude_run.check_init(init(servers=[{"name": "mms", "status": "failed"}]), "mms")
    assert claude_run.check_init(init(servers=[]), "mms"), "no server is not the mms arm"
    assert "MCP servers" in claude_run.check_init(init(tools=claude_run.FILES_TOOLS), "files"), "init() has the mms server"
    assert claude_run.check_init(init(tools=claude_run.MMS_TOOLS), "files"), "the mms arm's tools are not the files arm's"
    assert claude_run.check_init(init(plugins=[{"name": "codex", "path": "x"}]), "mms")
    assert "API key" in claude_run.check_init(init(apiKeySource="ANTHROPIC_API_KEY"), "mms")
    assert claude_run.check_init({"type": "system", "subtype": "init"}, "mms"), "an init that says nothing fails closed"

    deny = (lower(tmp_path / "answers"), ".claude")
    own = tmp_path / "data" / "notes" / "note-2026-10-28_09-12.md"
    assert claude_run.check_event(use("mcp__mms__find", {"scope": "all"}), "mms") is None
    assert "'Write'" in claude_run.check_event(use("Write", {"file_path": "x"}), "files", deny)
    assert "'Write'" in claude_run.check_event(use("Write", {"file_path": "x"}), "mms")
    assert claude_run.check_event(use("mcp__mms__remember", {"path": "x"}), "mms"), "remember is not offered"
    assert claude_run.check_event(use("Read", {"file_path": str(own)}), "mms"), "the mms arm has no Read"
    assert claude_run.check_event(use("Read", {"file_path": str(own)}), "files", deny) is None
    assert claude_run.check_event(use("Bash", {"command": f"ffmpeg -i {own} -vf fps=1 f%d.png"}), "files", deny) is None
    hit = claude_run.check_event(use("Read", {"file_path": str(tmp_path / "Answers" / "x.json")}), "files", deny)
    assert hit == f"Read reached {deny[0]}", "a path is read in lower case with forward slashes"
    assert claude_run.check_event(use("Grep", {"path": "C:\\Users\\user\\.Claude\\settings.json"}), "files", deny)
    assert claude_run.check_event(use("Glob", {"pattern": "**/*", "path": "~/.claude"}), "files", deny)
    assert claude_run.check_event(use("Read", {"file_path": ".claude"}), "files") is None, "no deny list, nothing denied"
    assert claude_run.check_event({"type": "assistant", "message": {"content": [
        {"type": "server_tool_use", "name": "web_search", "input": {}}]}}, "files", deny)
    assert claude_run.check_event(says("ANSWER: 47"), "files", deny) is None
    assert claude_run.check_event(init(), "mms") is None

    watch = claude_run.watcher("mms")
    missing = watch(use("mcp__mms__find", {}))
    assert missing.startswith("setup: ") and "init" in missing, "a message before the init event: not the session asked for"
    watch = claude_run.watcher("mms")
    wrong = watch(init(tools=[*claude_run.MMS_TOOLS, "Bash"]))
    assert wrong.startswith("setup: ") and "Bash" in wrong and watch(init()) is None
    assert watch(use("mcp__mms__ask", {})) is None and watch({"type": "user"}) is None
    assert watch(use("Bash", {})).startswith("escaped: claude used 'Bash'"), "what Claude does is its own"
    watch = claude_run.watcher("files", deny)
    assert watch(init(tools=claude_run.FILES_TOOLS, servers=())) is None
    assert watch(use("Read", {"file_path": ".claude/x"})) == "escaped: Read reached .claude"


def test_the_files_arm_may_read_back_its_own_saved_output_and_nothing_else_under_claude():
    """A Bash output over Claude Code's limit is saved under ~/.claude/projects/<folder>/<session>/tool-results, and its result
    names the file (2.1.302): the files arm may read that file back, not another session's or anything else under ~/.claude."""
    sid = "eb90273e-5f48-4dad-b3d2-2370a8a15ac1"
    saved = f"C:\\Users\\user\\.claude\\projects\\C--mms-bench-temp-mmsb-w2\\{sid}\\tool-results\\bbelb0265.txt"
    deny = (".claude",)
    big = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content":
           f"<persisted-output>\nOutput too large (2.4MB). Full output saved to: {saved}\n\nPreview (first 2KB):\nInput #0\n"}]}}

    def session(sid: str):
        watch = claude_run.watcher("files", deny)
        assert watch(init(tools=claude_run.FILES_TOOLS, servers=(), session_id=sid)) is None
        return watch

    watch = session(sid)
    assert watch(use("Bash", {"command": "ffmpeg -i a.ogg -af astats -f null -"})) is None and watch(big) is None
    assert watch(use("Read", {"file_path": saved, "offset": 100}, "t2")) is None
    assert watch(use("Read", {"file_path": saved.replace("\\", "/")}, "t3")) is None
    assert watch(use("Grep", {"pattern": "RMS", "path": saved.rsplit("\\", 1)[0]}, "t4")) is None, "the folder itself"
    for path in (saved.replace(sid, "0b7c-another-session"), saved.replace("bbelb0265.txt", "..\\..\\..\\..\\settings.json"),
                 "C:\\Users\\user\\.claude\\settings.json", saved.replace("tool-results", "subagents")):
        assert session(sid)(use("Read", {"file_path": path})) == "escaped: Read reached .claude", path
    assert session("0b7c-another-session")(big) == "escaped: a tool output reached .claude"
    assert claude_run.watcher("files", deny)(big) == "escaped: a tool output reached .claude", "no session known: none is own"


def test_a_made_up_run_gives_its_answer_tools_turns_tokens_and_cost():
    find = use("mcp__mms__find", {"since": "2026-10-28", "scope": "all"})
    got = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": False,
                                                    "content": [{"type": "text", "text": '{"items":[]}'}]}]}}
    events = [init(), find, got, says("It was 47.\nANSWER: 47"), finished()]
    done = claude_run.result_of(events, 0, "")
    assert done["problem"] is None and codex_run.answer_of(done["final"]) == "47"
    assert (done["turns"], done["cost_usd"], done["model"], done["denials"]) == (2, 0.0421, "claude-sonnet-x", 0)
    assert done["tokens"] == {"input": 120, "output": 45, "cache_read": 3000, "cache_write": 800}
    tools = claude_run.tools_of(events)
    assert [t["tool"] for t in tools] == ["find"] and tools[0]["arguments"]["scope"] == "all"
    assert tools[0]["result"] == [{"type": "text", "text": '{"items":[]}'}] and tools[0]["error"] is False
    twice = [init(), find, find, use("Read", {}, "t2"), finished()]
    assert [t["tool"] for t in claude_run.tools_of(twice)] == ["find", "Read"]

    assert claude_run.result_of([init(), says("thinking"), finished(result="")], 0, "")["final"] == "thinking"
    assert claude_run.result_of([init(), finished(permission_denials=[{"tool_name": "Bash"}] * 2)], 0, "")["denials"] == 2
    cut = claude_run.result_of([init(), says("partial"), {"type": "result", "subtype": "error_max_turns", "is_error": True,
                                                         "num_turns": 60}], 1, "")
    assert cut["problem"] == "capped: error_max_turns" and cut["final"] == "partial" and cut["tokens"] is None
    assert claude_run.result_of([init(), finished(subtype="error_max_budget_usd", is_error=True, result="")], 1,
                                "")["problem"] == "capped: error_max_budget_usd"
    assert claude_run.result_of([init(), finished(subtype="error_during_execution", is_error=True, result="")], 1,
                                "")["problem"] == "error: error_during_execution"
    assert claude_run.result_of([init(), finished(is_error=True, result="API Error: 500")], 1, "")["problem"] == \
        "error: API Error: 500"


def test_a_limit_a_login_and_a_missing_result_are_told_apart():
    def problem(result: dict | None, code: int = 1, stderr: str = "") -> str | None:
        return claude_run.result_of([init(), *([result] if result else [])], code, stderr)["problem"]

    assert problem(finished(is_error=True, result="You've hit your weekly limit \u00b7 resets Nov 5")).startswith("limit: ")
    assert problem(finished(is_error=True, result="Claude usage limit reached")).startswith("limit: ")
    assert problem(finished(is_error=True, result="API Error: 429 rate_limit_error")).startswith("limit: ")
    assert problem(finished(is_error=True, result="Invalid API key \u00b7 Please run /login")).startswith("auth: ")
    assert problem(finished(is_error=True, result="Not logged in")).startswith("auth: ")
    expired = "Failed to authenticate: OAuth session expired and could not be refreshed"  # seen in the 2026-11-02 smoke
    assert problem(finished(is_error=True, result=expired)) == f"auth: {expired}"
    assert problem(finished(is_error=True, result="API Error: 500"), stderr="rate limit exceeded").startswith("limit: ")
    assert problem(None, 3) == "error: claude exited 3 without a result"
    refused = "error: unknown option '--append-system-prompt-file'\n(Did you mean --append-system-prompt?)\n"
    assert problem(None, 1, refused) == "setup: error: unknown option '--append-system-prompt-file'"
    assert problem(None, 1, "error: option '--effort <level>' argument 'x' is invalid. Allowed choices are low.") \
        .startswith("setup: error: option '--effort <level>' argument 'x' is invalid")
    assert problem(None, 1, "error: required option '--model <m>' not specified").startswith("setup: ")
    assert problem(None, 1, "error: too many arguments. Expected 0 arguments but got 1.").startswith("setup: ")
    own = problem(finished(is_error=True, result="API Error: 500"), 1, refused)
    assert own == "error: API Error: 500", "a result is the session's own word"
    assert problem(finished(subtype="error_max_turns", is_error=True, result="")) == "capped: error_max_turns"
    spent = {"type": "result", "subtype": "error_max_budget_usd", "is_error": True}
    assert problem(spent) == "capped: error_max_budget_usd"
    assert problem(None, 1, "Claude AI usage limit reached").startswith("limit: ")
    assert problem(finished(result="the answer mentions a quota")) is None, "a success is never a problem"

    def week(**info) -> dict:
        return {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", **info}}

    assert claude_run.used_percent_of([init(), finished()]) is None
    assert claude_run.used_percent_of([week(rateLimitType="five_hour", utilization=0.9)]) is None
    assert claude_run.used_percent_of([week(rateLimitType="seven_day", utilization=0.25)]) == pytest.approx(25.0)
    windows = {"five_hour": {"utilization": 0.9, "resetsAt": 1}, "seven_day": {"utilization": 0.61, "resetsAt": 2}}
    both = week(rateLimitType="five_hour", utilization=0.9, unifiedWindows=windows)
    assert claude_run.used_percent_of([both]) == pytest.approx(61.0)
    later = week(rateLimitType="seven_day", utilization=0.3)
    assert claude_run.used_percent_of([week(unifiedWindows=windows), later]) == pytest.approx(30.0), "the last one counts"
    assert claude_run.used_percent_of([week(rateLimitType="seven_day", utilization="x")]) is None


def test_output_that_holds_a_deny_string_ends_the_files_arm_for_both_agents(tmp_path):
    deny = (lower(tmp_path / "mms-bench"), ".mms")
    leak = f"{tmp_path}/mms-bench/run1/results/claude/files/x.json:3: 47"

    def returned(content, id: str = "g1") -> dict:
        return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": id, "content": content}]}}

    watch = claude_run.watcher("files", deny)
    assert watch(init(tools=claude_run.FILES_TOOLS, servers=())) is None
    assert watch(use("Grep", {"pattern": "ERR-731", "path": str(tmp_path / "data")}, "g1")) is None
    assert watch(returned("data/logs/payments.jsonl:12: ERR-731")) is None
    assert watch(returned(leak)) == f"escaped: Grep output reached {deny[0]}"
    assert watch(returned([{"type": "text", "text": f"see {tmp_path / 'MMS-Bench' / 'run1'}"}])) == \
        f"escaped: Grep output reached {deny[0]}", "a path in a list of blocks, in any case"
    assert watch(returned("/home/user/.Mms/index.db", "unknown-id")) == "escaped: a tool output reached .mms"
    picture = {"type": "image", "source": {"type": "base64", "data": ".mms" * 20}}
    assert watch(returned([picture, {"type": "text", "text": "ok"}])) is None, "a picture's data is not text"
    watch = claude_run.watcher("mms", deny)
    assert watch(init()) is None and watch(use("mcp__mms__find", {}, "g1")) is None
    assert watch(returned(leak)) is None, "only the files arm has answers to keep from"
    assert claude_run.check_result(returned(leak), "files", (), {}) is None

    def output(text: str, kind: str = "item.completed") -> dict:
        return {"type": kind, "item": {"type": "command_execution", "command": "rg ERR-731 .", "aggregated_output": text,
                                        "exit_code": 0}}

    assert codex_run.check_event(output("data/logs/payments.jsonl:12: ERR-731"), "files", deny) is None
    assert codex_run.check_event(output(leak), "files", deny) == f"shell output reached {deny[0]}"
    assert codex_run.check_event(output(f"{tmp_path}\\MMS-BENCH\\run1\\x"), "files", deny) == \
        f"shell output reached {deny[0]}", "backslashes and case do not matter"
    home_file = output("C:/Users/user/.mms/index.db")
    assert codex_run.check_event(home_file, "files", deny) == "shell output reached .mms"
    assert codex_run.check_event(output(leak), "files") is None and codex_run.check_event(output(""), "files", deny) is None


# -- Claude: a stand-in process, for the reading of a stream, the escape, the timeout and the environment ----------

@pytest.fixture
def stand_in(tmp_path):
    """A script that prints the events of a file, writes its environment and cwd to a file, then sleeps, and the command
    that runs it in place of `claude` or `codex`: no real agent starts."""
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN, encoding="utf-8")
    probe = tmp_path / "probe.json"

    def command(events: list[dict], sleep: float = 0) -> list[str]:
        file = tmp_path / "events.jsonl"
        file.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        return [sys.executable, str(script), str(file), str(sleep), str(probe)]

    command.probe = lambda: json.loads(probe.read_text(encoding="utf-8"))
    return command


def test_claudes_stream_is_read_from_a_process_and_a_setup_problem_an_escape_or_a_timeout_ends_it(
        tmp_path, monkeypatch, stand_in, fake_git):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    run, work, home, data, bin_dir = (tmp_path / n for n in ("run", "w0", "home", "data", "bin"))
    find = use("mcp__mms__find", {"scope": "all"})
    got = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}}

    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([init(), find, got, says("ANSWER: 47"), finished()]))
    out = claude_run.ask_claude("What was the sign?", run, work, arm="mms", home=home, notes="d1_3b.md", bin_dir=bin_dir)
    assert out["problem"] is None and out["answer"] == "47" and out["turns"] == 2 and out["cost_usd"] == 0.0421
    assert [t["tool"] for t in out["tools"]] == ["find"] and out["tokens"]["cache_write"] == 800
    assert out["model"] == "claude-sonnet-x" and out["used_percent"] is None and out["argv"][0] == sys.executable
    assert (run / "prompt.txt").read_text() == "What was the sign?" and (run / "init.json").is_file()
    assert json.loads((run / "tools.json").read_text())[0]["tool"] == "find"
    assert (run / "final.txt").read_text().endswith("ANSWER: 47")
    assert len((run / "events.jsonl").read_text().splitlines()) == 5
    probe = stand_in.probe()
    assert Path(probe["cwd"]).samefile(work) and "ANTHROPIC_API_KEY" not in probe["env"]
    assert probe["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1" and probe["env"]["PATH"].startswith(str(bin_dir))

    deny = (lower(tmp_path / "answers"),)
    bad = use("Read", {"file_path": str(tmp_path / "answers" / "truth.json")})
    in_files = init(tools=claude_run.FILES_TOOLS, mcp_servers=[])
    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([in_files, bad, finished()], 60))
    out = claude_run.ask_claude("q", tmp_path / "run2", work, arm="files", data=data, deny=deny)
    assert out["problem"] == f"escaped: Read reached {deny[0]}" and out["seconds"] < 30 and out["answer"] is None

    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([in_files, says("ANSWER: 47"), finished()]))
    out = claude_run.ask_claude("q", tmp_path / "run2b", work, arm="files", data=data, deny=deny, bin_dir=bin_dir)
    probe = stand_in.probe()
    assert out["problem"] is None and probe["env"]["MMS_HOME"] == str(work / "no-mms-home")
    assert probe["env"]["PATH"].split(os.pathsep)[0] == str(bin_dir) and "ANTHROPIC_API_KEY" not in probe["env"]
    git_folders = tuple(str(fake_git / sub) for sub in ("cmd", "bin", "usr/bin"))
    assert probe["env"]["PATH"] == codex_run.shell_path(bin_dir, git_folders)

    grep = use("Grep", {"pattern": "ERR-731", "path": str(data)})
    leaked = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                                       "content": f"{tmp_path}/answers/truth.json:1: ERR-731"}]}}
    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([in_files, grep, leaked, finished()], 60))
    out = claude_run.ask_claude("q", tmp_path / "run2c", work, arm="files", data=data, deny=deny)
    assert out["problem"] == f"escaped: Grep output reached {deny[0]}" and out["seconds"] < 30 and out["answer"] is None

    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([init(tools=[*claude_run.MMS_TOOLS, "Bash"])], 60))
    out = claude_run.ask_claude("q", tmp_path / "run3", work, arm="mms", home=home)
    assert out["problem"].startswith("setup: ") and "Bash" in out["problem"] and out["seconds"] < 30

    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([init()], 60))
    out = claude_run.ask_claude("q", tmp_path / "run4", work, arm="mms", home=home, timeout=2)
    assert out["problem"] == "timeout: no answer in 2 s" and out["answer"] is None

    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([says("ANSWER: 47")]))
    out = claude_run.ask_claude("q", tmp_path / "run5", work, arm="mms", home=home)
    assert out["problem"].startswith("setup: ") and "init" in out["problem"], "no init event, no answer"


# -- Codex: the files arm -----------------------------------------------------------------------------------------------

def test_codexs_files_arm_has_its_shell_and_no_mms_and_the_mms_arm_is_pinned_to_its_project(tmp_path):
    run, work, home = tmp_path / "run", tmp_path / "w0", tmp_path / "home"
    argv = codex_run.argv_of(["codex"], "m", "high", run, None, arm="files", work=work)
    settings, disabled = settings_of(argv), disabled_of(argv)
    assert not any(s.startswith("mcp_servers.") for s in settings) and "--tools" not in argv
    assert not {"shell_tool", "unified_exec", "view_image"} & set(disabled)
    assert {"browser_use", "computer_use", "apps", "plugins", "multi_agent"} <= set(disabled), "the rest stay off"
    assert set(codex_run.DISABLED) - set(disabled) == {"shell_tool", "unified_exec", "view_image"}
    for setting in ("sandbox_mode=workspace-write", "include_environment_context=true",
                    "include_permissions_instructions=true", "sandbox_workspace_write.network_access=false",
                    "shell_environment_policy.inherit=core", "shell_environment_policy.exclude=['PATH']",
                    "windows.sandbox=unelevated", "approval_policy=never", "web_search=disabled"):
        assert setting in settings
    assert "shell_environment_policy.inherit=all" not in settings
    assert "sandbox_mode=read-only" not in settings and "include_environment_context=false" not in settings
    assert argv[argv.index("-C") + 1] == str(work) and argv[-2:] == ["--", "-"]

    # the shell sees ffmpeg's folder and the system's, and a MMS_HOME that does not exist: no mms, no .venv
    bin_dir = tmp_path / "bin"
    argv = codex_run.argv_of(["codex.CMD"], "m", "high", run, None, arm="files", work=work, bin_dir=bin_dir)
    codex_run.check_argv(argv, ["codex.CMD"])  # a .cmd shim reads none of it as syntax
    policy = dict(s.split("=", 1) for s in settings_of(argv) if s.startswith("shell_environment_policy.set."))
    assert list(policy) == ["shell_environment_policy.set.PATH", "shell_environment_policy.set.MMS_HOME"]
    shell_path = policy["shell_environment_policy.set.PATH"]
    assert shell_path.startswith("'") and shell_path.endswith("'")
    assert os.name != "nt" or ("\\" in shell_path and "/" not in shell_path), "a Windows PATH keeps its backslashes"
    assert shell_path.strip("'").split(os.pathsep) == [str(bin_dir), *codex_run.system_path()]
    assert not any(".venv" in folder for folder in shell_path.split(os.pathsep))
    assert policy["shell_environment_policy.set.MMS_HOME"] == f"'{str(work / 'no-mms-home').replace(chr(92), '/')}'"
    assert not (work / "no-mms-home").exists()
    argv = codex_run.argv_of(["codex"], "m", "high", run, None, arm="files")
    assert f"shell_environment_policy.set.MMS_HOME='{str(run / 'empty' / 'no-mms-home').replace(chr(92), '/')}'" in \
        settings_of(argv) and f"shell_environment_policy.set.PATH={codex_run.literal(codex_run.shell_path(None))}" in \
        settings_of(argv), "no bin_dir: the system's folders alone"

    argv = codex_run.argv_of(["codex"], "m", "high", run, home, work=work)
    assert not any(s.startswith("shell_environment_policy.") for s in settings_of(argv)), "the mms arm has no shell"
    pin = next(s for s in settings_of(argv) if s.startswith("mcp_servers.mms.env."))
    assert pin == f"mcp_servers.mms.env.CLAUDE_PROJECT_DIR='{str(work).replace(chr(92), '/')}'"
    assert "sandbox_mode=read-only" in settings_of(argv) and set(disabled_of(argv)) == set(codex_run.DISABLED)
    assert argv[argv.index("-C") + 1] == str(work)
    argv = codex_run.argv_of(["codex"], "m", "high", run, home)
    assert not any(".env." in s for s in settings_of(argv)) and argv[argv.index("-C") + 1] == str(run / "empty")
    with pytest.raises(codex_run.CodexMissing, match="home"):
        codex_run.argv_of(["codex"], "m", "high", run, None)
    with pytest.raises(ValueError, match="arm"):
        codex_run.argv_of(["codex"], "m", "high", run, home, arm="both")

    models = {"models": [{"slug": "m", "supported_reasoning_levels": [{"effort": "high"}]}]}
    argv = codex_run.prepare(run, "m", "high", "default", None, models, ["codex"], arm="files", work=work, notes="d3.md",
                             bin_dir=bin_dir)
    assert (run / "instructions.md").read_text() == codex_run.BASE_FILES and work.is_dir()
    assert argv[argv.index("-C") + 1] == str(work)
    assert json.loads((run / "catalog.json").read_text())["models"][0]["tool_mode"] == "direct"


def test_codexs_instructions_hold_the_models_notes_and_the_files_arms_line_alone():
    text = codex_run.instructions("default", "d3.md")
    assert NOTES["d3.md"] in text and NOTES["d1.md"] not in text and SKILL_TEXT in text
    assert NOTES["d1.md"] in codex_run.instructions("default"), "d1's notes stay the default"
    bare = codex_run.instructions("default", "")
    assert NOTES["d1.md"] not in bare and SKILL_TEXT in bare, "no notes: the skill alone"
    assert codex_run.instructions("none", "d3.md") == codex_run.BASE
    for guidance in ("default", "none"):
        assert codex_run.instructions(guidance, "d3.md", "files") == codex_run.BASE_FILES
    with pytest.raises(ValueError, match="plain file name"):
        codex_run.instructions("default", "..\\SKILL.md")


def test_codexs_problems_are_told_apart_by_what_the_benchmark_does_with_them():
    def turn_failed(message: str) -> dict:
        return {"type": "turn.failed", "error": {"message": message}}

    def error(message: str) -> dict:
        return {"type": "error", "message": message}

    def problem(events: list[dict], stderr: str = "", code: int = 1) -> str | None:
        return codex_run.problem_of(code, events, stderr)

    full = CONTEXT_FULL
    assert problem([turn_failed(full)]) == f"capped: {full}"
    assert problem([error(full), turn_failed("turn aborted")]).startswith("capped: Codex ran out of room")
    required = "required MCP servers failed to initialize: mms"
    assert problem([error(required), turn_failed(required)]) == f"setup: {required}"
    assert problem([error(required), turn_failed("turn aborted")]) == f"setup: {required}", "an earlier message counts"
    assert problem([turn_failed("required MCP server `mms` was not initialized")]).startswith("setup: ")
    assert problem([error("MCP client for `mms` failed to start: no such file")]).startswith("setup: ")
    assert problem([error("MCP client for `mms` timed out after 60 seconds.")]).startswith("setup: ")
    assert problem([], "Error loading configuration: unknown field `mcp_servers.mms.foo`").startswith("setup: Error loading")
    assert problem([turn_failed("You've hit your usage limit. Try again later.")]).startswith("limit: ")
    assert problem([turn_failed("401 Unauthorized: not logged in")]).startswith("auth: ")
    assert problem([turn_failed("stream disconnected before completion")]) == "error: stream disconnected before completion"
    assert problem([]) == "error: codex exec exited 1 without finishing the turn"
    assert problem([{"type": "turn.completed"}], code=0) is None
    assert problem([{"type": "turn.completed"}, turn_failed(full)], code=0).startswith("capped: ")


def test_codexs_files_arm_may_run_commands_and_nothing_else_and_none_may_reach_the_answers(tmp_path):
    deny = (lower(tmp_path / "answers"), ".codex")

    def item(kind: str, event: str = "item.started", **fields) -> dict:
        return {"type": event, "item": {"type": kind, **fields}}

    ffmpeg = item("command_execution", command=f'powershell -Command "ffmpeg -i {tmp_path}\\data\\phone\\a.jpg o.png"')
    assert codex_run.check_event(ffmpeg, "files", deny) is None
    for kind in ("agent_message", "reasoning", "error", "todo_list"):
        assert codex_run.check_event(item(kind), "files", deny) is None
    assert codex_run.check_event({"type": "turn.completed"}, "files", deny) is None
    hit = codex_run.check_event(item("command_execution", command=f"type {tmp_path}\\Answers\\truth.json"), "files", deny)
    assert hit == f"shell reached {deny[0]}"
    assert "shell reached .codex" in codex_run.check_event(
        item("command_execution", "item.completed", command="cat ~/.CODEX/auth.json"), "files", deny)
    assert codex_run.check_event(item("command_execution", command="cat ~/.codex/auth.json"), "files") is None
    for kind in ("mcp_tool_call", "file_change", "web_search", "collab_tool_call"):
        assert "files arm" in codex_run.check_event(item(kind, server="mms", tool="find"), "files", deny)

    # the mms arm is as it was: a command or a file change ends it, and so does a call of another server
    assert codex_run.check_event(ffmpeg)
    assert codex_run.check_event(item("file_change"), "mms", deny)
    assert codex_run.check_event(item("mcp_tool_call", server="mms", tool="find")) is None

    done = [{"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 9}},
            {"type": "turn.started"}, {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 4}}]
    assert codex_run.usage_of(done) == (2, {"input": 150, "output": 13, "cache_read": 60, "cache_write": 0})
    assert codex_run.usage_of([{"type": "turn.completed"}]) == (1, None) and codex_run.usage_of([]) == (0, None)
    shell = item("command_execution", "item.completed", command="ffmpeg -i a.mp4", aggregated_output="frame=1", exit_code=0,
                 status="completed")
    call = item("mcp_tool_call", "item.completed", server="mms", tool="ask", arguments={"scope": "all"}, status="completed")
    tools = codex_run.tools_of([item("command_execution"), shell, call])
    assert [t["tool"] for t in tools] == ["shell", "ask"] and tools[0]["arguments"] == {"command": "ffmpeg -i a.mp4"}
    assert tools[0]["result"] == "frame=1" and tools[0]["error"] is None and tools[1]["arguments"] == {"scope": "all"}
    failed = {**shell["item"], "exit_code": 1}
    assert codex_run.tools_of([{"type": "item.completed", "item": failed}])[0]["error"] == "exit code 1"

    env = codex_run.path_with(tmp_path / "bin")
    assert env["PATH"].startswith(str(tmp_path / "bin") + os.pathsep) and codex_run.path_with(None) == dict(os.environ)


def test_codexs_files_exec_is_read_from_a_process_and_a_command_at_the_answers_ends_it(tmp_path, monkeypatch, stand_in):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    work, bin_dir = tmp_path / "w0", tmp_path / "bin"
    models = {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]}
    deny = (lower(tmp_path / "answers"),)

    def command(name: str, kind: str = "command_execution", **fields) -> dict:
        return {"type": name, "item": {"type": kind, "command": "ffmpeg -i x.mp4 f.png", **fields}}

    said = {"type": "item.completed", "item": {"type": "agent_message", "text": "ANSWER: 47"}}
    usage = {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 9}}
    events = [{"type": "turn.started"}, command("item.started"), command("item.completed", exit_code=0), said, usage]
    monkeypatch.setattr(codex_run, "codex_command", lambda: stand_in(events))
    out = codex_run.ask_codex("What was the sign?", None, tmp_path / "run", "gpt-test", "high", models=models, arm="files",
                              work=work, deny=deny, bin_dir=bin_dir)
    assert out["problem"] is None and out["answer"] == "47" and out["model"] == "gpt-test" and out["cost_usd"] is None
    assert out["turns"] == 1 and out["tokens"] == {"input": 100, "output": 9, "cache_read": 60, "cache_write": 0}
    assert [t["tool"] for t in out["tools"]] == ["shell"] and out["used_percent"] is None
    assert Path(stand_in.probe()["cwd"]).samefile(work) and stand_in.probe()["env"]["PATH"].startswith(str(bin_dir))
    assert (tmp_path / "run" / "instructions.md").read_text() == codex_run.BASE_FILES

    bad = command("item.started", command=f"type {tmp_path}\\answers\\truth.json")
    monkeypatch.setattr(codex_run, "codex_command", lambda: stand_in([{"type": "turn.started"}, bad, said, usage], 60))
    out = codex_run.ask_codex("q", None, tmp_path / "run2", "gpt-test", "high", models=models, arm="files", work=work,
                              deny=deny)
    assert out["problem"] == f"escaped: shell reached {deny[0]}" and out["seconds"] < 30 and out["answer"] is None

    monkeypatch.setattr(codex_run, "codex_command",
                        lambda: stand_in([{"type": "turn.started"}, command("item.started", "mcp_tool_call")], 60))
    out = codex_run.ask_codex("q", None, tmp_path / "run3", "gpt-test", "high", models=models, arm="files", work=work)
    assert out["problem"].startswith("escaped: ") and "mcp_tool_call" in out["problem"]

    leak = command("item.completed", exit_code=0, aggregated_output=f"{tmp_path}/answers/truth.json:1: ERR-731")
    monkeypatch.setattr(codex_run, "codex_command", lambda: stand_in([{"type": "turn.started"}, leak, said, usage], 60))
    out = codex_run.ask_codex("q", None, tmp_path / "run4", "gpt-test", "high", models=models, arm="files", work=work,
                              deny=deny)
    assert out["problem"] == f"escaped: shell output reached {deny[0]}" and out["seconds"] < 30 and out["answer"] is None

    full = CONTEXT_FULL
    monkeypatch.setattr(codex_run, "codex_command",
                        lambda: stand_in([{"type": "turn.started"}, {"type": "turn.failed", "error": {"message": full}}]))
    out = codex_run.ask_codex("q", None, tmp_path / "run5", "gpt-test", "high", models=models, arm="files", work=work)
    assert out["problem"] == f"capped: {full}" and out["turns"] == 0 and out["answer"] is None

    required = "required MCP servers failed to initialize: mms"
    monkeypatch.setattr(codex_run, "codex_command", lambda: stand_in([{"type": "error", "message": required}]))
    out = codex_run.ask_codex("q", None, tmp_path / "run6", "gpt-test", "high", models=models, arm="files", work=work)
    assert out["problem"] == f"setup: {required}"


def test_codex_view_image_calls_are_read_from_the_execs_own_rollout(tmp_path, monkeypatch):
    """`codex exec --json` has no item for view_image, so the files arm counts them from the rollout file."""
    picture = r"C:\data\phone\IMG_1.jpg"
    lines = [json.dumps({"type": "session_meta", "payload": {"id": "t1"}}),
             json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                                                               "arguments": json.dumps({"cmd": "dir"})}}),
             json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "view_image",
                                                               "arguments": json.dumps({"path": picture})}}),
             "not json"]
    assert codex_run.images_in(lines) == [{"tool": "view_image", "arguments": {"path": picture}, "result": None,
                                           "error": None, "status": "completed"}]
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    day = tmp_path / "sessions" / "2026" / "11" / "02"
    day.mkdir(parents=True)
    (day / "rollout-2026-11-02T13-01-33-t1.jsonl").write_text("\n".join(lines), encoding="utf-8")
    assert [c["arguments"]["path"] for c in codex_run.rollout_images("t1")] == [picture]
    assert codex_run.rollout_images("t2") == [] and codex_run.rollout_images("") == []
    assert codex_run.reached({"path": picture}, ("c:/data",)) == "c:/data", "a picture's path is checked like a command"


def test_claude_login_is_checked_with_auth_status_before_a_run(tmp_path):
    def fake(stdout: str) -> list[str]:
        script = tmp_path / f"claude{len(list(tmp_path.iterdir()))}.py"
        script.write_text(f"import sys\nassert sys.argv[1:] == ['auth', 'status']\nprint({stdout!r})\n", encoding="utf-8")
        return [sys.executable, str(script)]

    claude_run.check_login(fake('{"loggedIn": true, "authMethod": "claude.ai"}'))
    with pytest.raises(claude_run.ClaudeMissing, match="claude auth login"):
        claude_run.check_login(fake('{"loggedIn": false, "authMethod": "none"}'))
    with pytest.raises(claude_run.ClaudeMissing, match="gave no status"):
        claude_run.check_login(fake("Usage: claude [options]"))


# -- the blind arm: no tools, no server, no recordings ------------------------------------------------------------------

BLIND = "You answer the user's question from what you know; you have no tools."


def test_the_blind_arm_has_one_line_of_instructions_and_no_tools_for_both_agents(tmp_path):
    assert codex_run.BASE_BLIND == claude_run.BASE_BLIND == BLIND and "blind" in codex_run.ARMS == claude_run.ARMS
    run, work, home = tmp_path / "run", tmp_path / "w0", tmp_path / "home"

    # Claude: `--tools ""`, the strict MCP flag with no config, and nothing allowed
    argv = claude_run.prepare(run, "blind", work, command=["claude"])
    assert after(argv, "--tools") == [""] and "--strict-mcp-config" in argv and "--restricted" in argv
    for flag in ("--mcp-config", "--allowedTools", "--add-dir"):
        assert flag not in argv, flag
    assert after(argv, "--permission-mode") == ["dontAsk"] and after(argv, "--settings") == [str(run / "settings.json")]
    assert (run / "instructions.md").read_text() == BLIND and not (run / "mcp.json").exists() and work.is_dir()
    assert claude_run.instructions("blind", "d1_3b.md") == BLIND, "no skill, no notes"
    given = claude_run.argv_of(["claude"], "blind", run, name="sonnet", level="high", data=tmp_path / "data")
    assert "--add-dir" not in given and after(given, "--tools") == [""], "no data folder either"
    assert claude_run.env_of(tmp_path / "bin", "blind")["PATH"].startswith(str(tmp_path / "bin"))

    # Codex: the mms arm's read-only settings with every feature off and no server
    argv = codex_run.argv_of(["codex"], "m", "high", run, None, arm="blind", work=work)
    settings, disabled = settings_of(argv), disabled_of(argv)
    assert not any(s.startswith(("mcp_servers.", "shell_environment_policy.")) for s in settings) and "--tools" not in argv
    assert set(disabled) == set(codex_run.DISABLED) and {"shell_tool", "unified_exec", "view_image"} <= set(disabled)
    for setting in codex_run.CONFIG:
        assert setting in settings, setting
    assert "sandbox_workspace_write.network_access=false" not in settings and "windows.sandbox=unelevated" not in settings
    assert argv[argv.index("-C") + 1] == str(work) and argv[-2:] == ["--", "-"]
    models = {"models": [{"slug": "m", "supported_reasoning_levels": [{"effort": "high"}]}]}
    codex_run.prepare(tmp_path / "codex-run", "m", "high", "default", None, models, ["codex"], arm="blind", work=work, notes="d3.md")
    assert (tmp_path / "codex-run" / "instructions.md").read_text() == BLIND
    for guidance in ("default", "none"):
        assert codex_run.instructions(guidance, "d3.md", "blind") == BLIND
    assert after(codex_run.argv_of(["codex"], "m", "high", run, home, arm="blind"), "-C")[0] == str(run / "empty")


def test_a_session_with_a_tool_or_a_server_is_not_the_blind_arms_and_any_tool_use_is_an_escape(tmp_path):
    assert claude_run.check_init(init(tools=(), servers=()), "blind") is None
    assert "Bash" in claude_run.check_init(init(tools=["Bash"], servers=()), "blind")
    assert "mcp__mms__ask" in claude_run.check_init(init(tools=claude_run.MMS_TOOLS, servers=()), "blind")
    assert "MCP servers" in claude_run.check_init(init(tools=(), servers=[{"name": "mms", "status": "connected"}]), "blind")
    assert claude_run.check_init(init(tools=(), servers=(), plugins=[{"name": "x"}]), "blind")

    for name in ("Read", "Bash", "Glob", "mcp__mms__find", "WebSearch"):
        hit = claude_run.check_event(use(name, {"command": "ffmpeg -i a.mp4", "file_path": "x"}), "blind")
        assert hit and repr(name) in hit and "blind arm" in hit, name
    assert claude_run.check_event({"type": "assistant", "message": {"content": [
        {"type": "server_tool_use", "name": "web_search", "input": {}}]}}, "blind")
    assert claude_run.check_event(says("ANSWER: B"), "blind") is None
    watch = claude_run.watcher("blind")
    assert watch(init(tools=(), servers=())) is None and watch(says("Let me think.")) is None
    assert watch(use("Read", {"file_path": "x"})).startswith("escaped: claude used 'Read'")

    # Codex: a message, a thought, an error and the plan list are the only items; a shell, a server call or a search is not
    def item(kind: str, event: str = "item.started", **fields) -> dict:
        return {"type": event, "item": {"type": kind, **fields}}

    for kind in ("agent_message", "reasoning", "error", "todo_list"):
        assert codex_run.check_event(item(kind), "blind") is None, kind
    assert codex_run.check_event({"type": "turn.completed"}, "blind") is None
    for kind in ("command_execution", "mcp_tool_call", "file_change", "web_search", "collab_tool_call"):
        for event in ("item.started", "item.completed"):
            hit = codex_run.check_event(item(kind, event, server="mms", tool="ask"), "blind")
            assert hit and repr(kind) in hit and "blind arm" in hit, (kind, event)


def test_a_blind_exec_is_read_from_a_process_and_a_tool_event_ends_it_as_escaped(tmp_path, monkeypatch, stand_in):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    work = tmp_path / "w0"
    blind_init = init(tools=(), servers=())
    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([blind_init, says("B."), finished(result="B.\nANSWER: B")]))
    out = claude_run.ask_claude("Which?", tmp_path / "claude-ok", work, arm="blind")
    assert out["problem"] is None and out["answer"] == "B" and out["tools"] == []
    assert (tmp_path / "claude-ok" / "prompt.txt").read_text() == "Which?" and "--tools" in out["argv"]
    assert "--mcp-config" not in out["argv"] and "--allowedTools" not in out["argv"]

    monkeypatch.setattr(claude_run, "claude_command",
                        lambda: stand_in([blind_init, use("Bash", {"command": "ffmpeg -i a.mp4"}), says("ANSWER: B")], 60))
    out = claude_run.ask_claude("Which?", tmp_path / "claude-tool", work, arm="blind")
    assert out["problem"].startswith("escaped: claude used 'Bash'") and out["seconds"] < 30 and out["answer"] is None

    monkeypatch.setattr(claude_run, "claude_command", lambda: stand_in([init(tools=["Bash"], servers=())], 60))
    out = claude_run.ask_claude("Which?", tmp_path / "claude-setup", work, arm="blind")
    assert out["problem"].startswith("setup: ") and "Bash" in out["problem"] and out["seconds"] < 30

    models = {"models": [{"slug": "gpt-test", "supported_reasoning_levels": [{"effort": "high"}]}]}
    said = {"type": "item.completed", "item": {"type": "agent_message", "text": "ANSWER: C"}}
    usage = {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 9}}
    monkeypatch.setattr(codex_run, "codex_command", lambda: stand_in([{"type": "turn.started"}, said, usage]))
    out = codex_run.ask_codex("Which?", None, tmp_path / "codex-ok", "gpt-test", "high", models=models, arm="blind", work=work)
    assert out["problem"] is None and out["answer"] == "C" and out["tools"] == []
    assert (tmp_path / "codex-ok" / "instructions.md").read_text() == BLIND

    shell = {"type": "item.started", "item": {"type": "command_execution", "command": "ffmpeg -i a.mp4 f.png"}}
    monkeypatch.setattr(codex_run, "codex_command", lambda: stand_in([{"type": "turn.started"}, shell, said, usage], 60))
    out = codex_run.ask_codex("Which?", None, tmp_path / "codex-tool", "gpt-test", "high", models=models, arm="blind", work=work)
    assert out["problem"].startswith("escaped: ") and "command_execution" in out["problem"] and "blind arm" in out["problem"]
    assert out["seconds"] < 30 and out["answer"] is None
