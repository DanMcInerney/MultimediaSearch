"""`mms mcp` as a process, driven over stdio by the MCP SDK's client as Claude Code and Codex drive it."""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters, types

# `mms mcp` as the registered command runs it, then the modules the process ended up with
WRAPPER = ("import json, sys; import mms.cli as c; c.main(['mcp']); "
           "open(sys.argv[1], 'w').write(json.dumps(sorted(sys.modules)))")


@asynccontextmanager
async def connected(home: Path, cwd: Path, *, env: dict | None = None, modules: Path | None = None,
                    info: tuple[str, str] = ("testclient", "9.9"), command: list[str] | None = None):
    """An MCP session with a fresh adapter process in `cwd`. `command` replaces the default `python -m mms.cli mcp`;
    `modules` is a file to write the adapter's imported modules to, once it has exited."""
    environment = dict(os.environ, MMS_HOME=str(home), PYTHONUTF8="1")
    environment.pop("CLAUDE_PROJECT_DIR", None)
    environment.update(env or {})
    if command is None:
        command = [sys.executable, *(["-c", WRAPPER, str(modules)] if modules else ["-m", "mms.cli", "mcp"])]
    params = StdioServerParameters(command=command[0], args=command[1:], env=environment, cwd=cwd)
    async with Client(params, client_info=types.Implementation(name=info[0], version=info[1])) as client:
        yield client


def drive(steps, home: Path, cwd: Path, **options):
    """Run `await steps(client)` against a fresh adapter process, and return what it returned."""
    async def main():
        async with connected(home, cwd, **options) as client:
            return await steps(client)
    return anyio.run(main)


def text(result: types.CallToolResult) -> dict:
    """The JSON of a tool result: its last block, after any picture."""
    assert isinstance(result.content[-1], types.TextContent)
    return json.loads(result.content[-1].text)
